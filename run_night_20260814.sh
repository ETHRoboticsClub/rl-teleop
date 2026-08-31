#!/usr/bin/env bash
# Two ACT policies from ONE night of recording — right arm, new ELE01 wrist camera.
#
#   A) PICK-AND-PLACE ("empty the box")  wide window, [t_close-4s, t_close+6s]
#      one training episode per CYCLE = reach → grasp → carry → place
#   B) GRASP (the usual)                 exporter defaults, [t_close-3s, t_close+2s]
#      one training episode per successful grasp. Measured 4.75 s mean after
#      clipping — which lands on the 4.58 s of the deployed 20260812 dataset, so
#      B reproduces the only window length proven to train a working policy here.
#
# Both are grasp-mode cuts at different widths, so the operator records ONE long
# take per box and gets a per-cycle corpus for both. --window-mode full is NOT
# used: it yields one demo per TAKE, which on the 2026-08-14 recordings meant 3
# demos from 66 grasps.
#
# Both are exported from THE SAME recordings. The exporter cuts grasp windows out
# of whole takes (that is how the deployed 20260812 lineage was made), so there is
# no reason to record twice. Record once, export twice, train twice.
#
# Usage:
#   ./run_night_20260814.sh                 # label, export, predecode, train both
#   ./run_night_20260814.sh --dry           # label + dry-run both exports, stop
#   DAY=20260814 ./run_night_20260814.sh    # a different recording day
#
# Run it AFTER you stop recording. It does not touch the arm or any camera.
set -uo pipefail
cd "$(dirname "$0")"

DAY="${DAY:-$(date +%Y%m%d)}"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

ARM=right
# camera_right (ELE01) -> observation.images.wrist, camera_top (D455) ->
# observation.images.top. Was wrist_right until the D455 came back on the ASMedia
# rebind at ~01:5x; the top view matters most for the PLACE half of policy A,
# which a wrist camera sees badly. Episodes recorded before the switch have no
# camera_top and the exporter rejects them by name — expected, it is 2 takes.
CAMS=wrist_right_top
# Steps are per-policy because A's windows are ~4x longer than B's, so at equal
# step counts A sees each demo ~4x less often. Measured on the 08-14 takes: A
# 441 s of window vs B 119 s. (The 20260812 grasp run converged at 135 epochs.)
PP_STEPS="${PP_STEPS:-200000}"
GR_STEPS="${GR_STEPS:-100000}"
# 20k, not the 10k the 20260812 run used: each checkpoint is 591 MB and two runs
# at 10k would write 11.8 GB into a filesystem that is already 97% full. 5 saves
# per run is still enough to pick a checkpoint by eval rather than by luck.
SAVE_FREQ="${SAVE_FREQ:-25000}"
BATCH="${BATCH:-8}"

REC="${REC:-recordings/$DAY}"   # overridable so --dry can rehearse on a copy
PP_REPO=ETHRC/yam_pickplace_right_$DAY      # A
GR_REPO=ETHRC/yam_grasp_right_$DAY          # B
HF=$HOME/.cache/huggingface/lerobot
PD=$HOME/.cache/lerobot-predecoded

PY=./.venv/bin/python3                      # NOT `uv run` — see rl-teleop/CLAUDE.md
mkdir -p outputs/train logs
LOG=logs/night_$DAY.log
say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "=== night run $DAY ==="

# ── 0. sanity ───────────────────────────────────────────────────────────────
EPS=( "$REC"/episode_* )
if [ ! -d "${EPS[0]:-}" ]; then
    say "FATAL: no episodes in $REC — nothing was recorded, or DAY is wrong"
    exit 1
fi
say "episodes recorded : ${#EPS[@]} in $REC"

FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk         : ${FREE_GB} GB"
if [ "$FREE_GB" -lt 12 ]; then
    say "FATAL: under 12 GB free. A full disk truncates mp4s and kills checkpoint"
    say "       writes SILENTLY (DATA-PIPELINE.md 2.6). Free space first, e.g.:"
    say "       rm -rf outputs/train/act_wrist_zone_v2 sweeps/outputs/<old runs>"
    exit 1
fi

# ── 1. label ────────────────────────────────────────────────────────────────
# record_kitting.sh runs the auto-labeller for the LEFT arm only (the detector and
# the grasp gate are calibrated in the left base frame), so right-arm episodes
# arrive with NO annotations_right.json and export B would reject every one of
# them. Label them here.
#
# --open-ref 1.0 --closed-ref 0.0 is NOT decoration. Without refs the labeller
# guesses polarity from ONE sample — the first of the episode — and if that
# sample lands on the closed side it inverts every success/slip/empty label in
# the take, with no flag (DATA-PIPELINE.md 2.3). Measured on this arm's mcaps the
# gripper channel already spans 0.001..0.999, so 1.0/0.0 are its true limits and
# the guess is removed rather than merely biased.
say "[1/5] labelling ${#EPS[@]} episodes (--arm $ARM, explicit gripper refs)..."
# A take recorded while a TRAINED-ON camera was unhealthy is not training data.
# Proven necessary 2026-08-14 03:10: the wrist camera was yanked mid-take and the
# session recorded for another 29 minutes — camera_top a clean 30.0 Hz, camera_right
# 5130 frames where ~55500 were due (9%), worst gap 7.55 s. Nothing else catches
# this: the frame COUNT is only checkable after the fact, the mp4 is valid, and the
# exporter resamples by nearest timestamp with only a 66.7 ms staleness cap on
# cameras — so a 7 s hole silently becomes 200 repeats of one stale frame.
#
# Only cameras in $CAMS matter. A degraded camera nobody trains on is noise.
say "[0/5] dropping takes degraded on a camera we train on..."
DEGRADED=()
for ep in "${EPS[@]}"; do
    hit=$($PY -c "
import json
m=json.load(open('$ep/session_meta.json'))
if not m.get('degraded'): print('')
else:
    bad=set(m.get('degraded_nodes') or [])
    want={'camera_right','camera_top'} if '$CAMS'=='wrist_right_top' else {'camera_right'}
    print(','.join(sorted(bad & want)))" 2>/dev/null)
    if [ -n "$hit" ]; then
        say "      $(basename "$ep")  DEGRADED on $hit — moving to recordings/.trash"
        mkdir -p recordings/.trash && mv "$ep" recordings/.trash/ && DEGRADED+=("$(basename "$ep")")
    fi
done
[ ${#DEGRADED[@]} -gt 0 ] && say "      ${#DEGRADED[@]} take(s) dropped as degraded"
# rebuild the list; .trash is never training data
EPS=( "$REC"/episode_* )
if [ ! -d "${EPS[0]:-}" ]; then say "FATAL: every take was degraded"; exit 1; fi

NO_LABEL=()
for ep in "${EPS[@]}"; do
    $PY -m robots_realtime.labeling.label_episode "$ep" --arm "$ARM" \
        --open-ref 1.0 --closed-ref 0.0 >>"$LOG" 2>&1
    # label_episode exits 0 when it SKIPS an incomplete episode, so its exit code
    # is not the test — the file it was supposed to write is.
    if [ ! -f "$ep/annotations_$ARM.json" ]; then
        say "      $(basename "$ep")  NO LABELS WRITTEN (see $LOG) — dropped from both datasets"
        NO_LABEL+=("$(basename "$ep")")
        continue
    fi
    n=$($PY -c "
import json
a=json.load(open('$ep/annotations_$ARM.json'))
g=a.get('grasp_attempts') or []
print(f\"{sum(1 for x in g if x.get('outcome')=='success')}/{len(g)}\")" 2>/dev/null || echo "?")
    say "      $(basename "$ep")  success/attempts: $n"
done
[ ${#NO_LABEL[@]} -gt 0 ] && say "      ${#NO_LABEL[@]}/${#EPS[@]} episodes produced no labels at all"

# ── 2. export ───────────────────────────────────────────────────────────────
# Zone gate: GRASP_WORKSPACE_X_MIN=0.25 was measured on the LEFT arm's mat. The
# right arm's grasps into the source bin measured x = 0.25..0.49 on 08-10/08-11,
# so the default gate happens to fit — but a couple sit right on the edge. The
# dry run prints every rejection with its reason; read it before believing a low
# episode count is a recording failure (DATA-PIPELINE.md 2.2).
#
# GRIPPER REFS. Both exports pass the jaws' physical limits (1.0/0.0, verified
# against this arm's mcaps: the channel spans 0.001..0.999). Without them each
# episode is scaled by its OWN percentiles, so with one cycle per take the same
# physical jaw opening lands on a different number in every episode — the exact
# case short takes create. Added as a flag today; the default is unchanged.
GREF=(--gripper-open-ref 1.0 --gripper-closed-ref 0.0)

# POLICY A IS CUT PER CYCLE, NOT PER TAKE. --window-mode full gives ONE demo per
# recorded take, so three 5-minute takes holding 66 grasps yielded three demos —
# and ACT would be learning "run 29 cycles back to back" instead of "run one".
#
# Widening the GRASP window instead gives one demo per cycle from the same takes.
# grasp_windows() clips adjacent windows so they never overlap, so asking for
# 4 s before / 6 s after the close auto-fits the actual cycle: measured on the
# 2026-08-14 takes, 60 windows, mean 7.36 s, median 7.56 s — each one a full
# reach → grasp → carry → place. 60 demos instead of 3, and 4.06 GB instead of 66.
#
# The honest cost: a window centred on the close starts mid-approach rather than
# from a rest pose, so the policy never sees "arm at rest" as an episode's first
# frame. ACT conditions on observations, not on episode boundaries, so this is a
# weaker demo boundary rather than a wrong one. A take whose cycle is much longer
# than the window (take1 at 36 s/cycle) contributes a partial cycle.
PP_PRE="${PP_PRE:-4}"    # seconds before the gripper closes  → the approach
PP_POST="${PP_POST:-6}"  # seconds after                      → lift, carry, place

export_both() {
    local extra="$1"
    say "[2/5] export A — pick-and-place (per cycle, -${PP_PRE}s/+${PP_POST}s around each close) $extra"
    $PY tools/export_lerobot.py --root "$REC" --repo-id "$PP_REPO" \
        --arms "$ARM" --cameras "$CAMS" --window-mode grasp \
        --pre-s "$PP_PRE" --post-s "$PP_POST" "${GREF[@]}" $extra 2>&1 | tee -a "$LOG"
    say "[2/5] export B — grasp windows (exporter defaults -3s/+2s) $extra"
    $PY tools/export_lerobot.py --root "$REC" --repo-id "$GR_REPO" \
        --arms "$ARM" --cameras "$CAMS" --window-mode grasp "${GREF[@]}" $extra 2>&1 | tee -a "$LOG"
}

if [ "$DRY" = 1 ]; then
    export_both --dry-run
    say "--dry: stopping here. Nothing was written."
    exit 0
fi
export_both ""

for r in "$PP_REPO" "$GR_REPO"; do
    if [ ! -f "$HF/$r/meta/info.json" ]; then
        say "FATAL: export produced no dataset at $HF/$r"
        exit 1
    fi
    say "      $r: $($PY -c "import json;d=json.load(open('$HF/$r/meta/info.json'));print(d['total_episodes'],'episodes,',d['total_frames'],'frames')")"
done

# ── 3. predecode ────────────────────────────────────────────────────────────
# torchcodec cannot load on this box, so this is what makes video reading WORK,
# not just what makes it fast.
say "[3/5] predecoding both datasets..."
for r in "$PP_REPO" "$GR_REPO"; do
    n=$(basename "$r")
    $PY tools/predecode_ffmpeg.py --dataset-root "$HF/$r" --output-root "$PD/$n" 2>&1 | tail -3 | tee -a "$LOG"
done

# ── 4. train, both at once ──────────────────────────────────────────────────
# One 5090, 32 GB. A single ACT run at batch 8 measured 0.032 s/step (100k steps
# ≈ 1 h) and is GPU-bound, so two side by side land at roughly 2 h each — well
# inside the night. The log must live OUTSIDE the output dir: LeRobot's
# cfg.validate() refuses to train into a directory that already exists.
# EACH RUN GETS ITS OWN TMUX SESSION. Previously these were `&` subshells of this
# script, so closing the terminal (or dropping SSH) SIGHUPed the script and killed
# both trainings hours in, with only a truncated log to show for it. In tmux they
# survive the terminal, this script, and each other — and `tmux attach -t <sess>`
# shows a live run rather than a tail of a file.
#
# The exit code is written to outputs/train/<run>.exit by the tmux command itself,
# which is what this script polls; if it is killed, the runs still finish and still
# leave their verdict on disk.
train_tmux() {
    local run="$1" repo="$2" seed="$3" steps="$4" sess="$5" aug="${6:-dark_noise}"
    local out="outputs/train/$run"
    rm -rf "$out"
    rm -f "outputs/train/$run.exit"
    tmux kill-session -t "$sess" 2>/dev/null
    tmux new-session -d -s "$sess" -c "$PWD" \
        "LEROBOT_PREDECODED_ROOT='$PD/$(basename "$repo")' ACT_AUG='$aug' $PY tools/train_act_dark_noise.py \
            --dataset.repo_id='$repo' \
            --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
            --steps=$steps --save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$seed \
            --output_dir='$out' > 'outputs/train/$run.log' 2>&1; \
         echo \$? > 'outputs/train/$run.exit'"
    sleep 2
    tmux has-session -t "$sess" 2>/dev/null \
        && say "      tmux session '$sess' up  (tmux attach -t $sess)" \
        || say "      WARNING: tmux session '$sess' did not start"
}

PP_RUN=act_pickplace_right_night_$DAY
GR_RUN=act_grasp_right_night_$DAY
say "[4/5] training both (batch $BATCH, save every $SAVE_FREQ)"
say "      A pick-and-place $PP_STEPS steps -> outputs/train/$PP_RUN.log"
say "      B grasp          $GR_STEPS steps -> outputs/train/$GR_RUN.log"
train_tmux "$PP_RUN" "$PP_REPO" 1000 "$PP_STEPS" train_pp dark_noise
sleep 60          # stagger: let A allocate and settle before B contends for VRAM

# The augmentation comparison runs on POLICY B, sequentially, in one tmux session.
#
# Sequential and not parallel because VRAM is the binding constraint, not time:
# ACT with TWO camera streams at batch 8 is untested on this box for footprint,
# and five concurrent runs on a 32 GB card is exactly how you get an OOM at 03:00
# with nothing to show. Two at a time (A + one variant) is the safe envelope.
#
# On B and not A because B is the cheap, proven half — 4.75 s windows, 100k steps,
# and a deployed lineage to compare against. Set AUG_LIST="" to skip the whole
# comparison, or AUG_LIST="none geometric" for a shorter night.
AUG_LIST="${AUG_LIST:-dark_noise none geometric occlusion}"
# Comparison runs want a VERDICT, not a checkpoint ladder: one mid save and the
# final is enough to rank recipes, and it keeps four runs inside ~4.7 GB.
VAR_SAVE_FREQ="${VAR_SAVE_FREQ:-50000}"

say "      augmentation comparison on policy B: $AUG_LIST"
QUEUE=""
for aug in $AUG_LIST; do
    # LEROBOT_PREDECODED_ROOT MUST BE INLINE, not exported before `tmux
    # new-session`. A tmux session inherits the tmux SERVER's environment, which
    # was captured whenever the server first started — not the environment of the
    # client creating the session. Exporting it here reached nothing, every run
    # fell back to torchcodec, and torchcodec cannot load on this box at all
    # (no libavutil): four runs died within seconds of launch, 2026-08-14 05:38.
    QUEUE="$QUEUE LEROBOT_PREDECODED_ROOT='$PD/$(basename "$GR_REPO")' ACT_AUG=$aug $PY tools/train_act_dark_noise.py \
        --dataset.repo_id='$GR_REPO' --policy.type=act --policy.device=cuda \
        --policy.push_to_hub=false --steps=$GR_STEPS --save_freq=$VAR_SAVE_FREQ \
        --batch_size=$BATCH --seed=2000 \
        --output_dir='outputs/train/${GR_RUN}_$aug' \
        > 'outputs/train/${GR_RUN}_$aug.log' 2>&1; \
        echo \$? > 'outputs/train/${GR_RUN}_$aug.exit';"
    rm -rf "outputs/train/${GR_RUN}_$aug" "outputs/train/${GR_RUN}_$aug.exit"
done
tmux kill-session -t train_gr 2>/dev/null
tmux new-session -d -s train_gr -c "$PWD" "$QUEUE echo done > outputs/train/${GR_RUN}_ALL.exit"
sleep 2
tmux has-session -t train_gr 2>/dev/null \
    && say "      tmux session 'train_gr' up, $(echo $AUG_LIST | wc -w) runs queued  (tmux attach -t train_gr)" \
    || say "      WARNING: tmux session 'train_gr' did not start"

# Poll for verdicts. This script may be killed at any point from here on without
# affecting the runs — they are in tmux and write their own .exit files.
say "      launched; waiting. Safe to Ctrl-C this script — tmux keeps training."
while [ ! -f "outputs/train/$PP_RUN.exit" ] || [ ! -f "outputs/train/${GR_RUN}_ALL.exit" ]; do
    sleep 60
done

# ── 5. verdict ──────────────────────────────────────────────────────────────
say "[5/5] done"
RUNS="$PP_RUN"
for aug in $AUG_LIST; do RUNS="$RUNS ${GR_RUN}_$aug"; done
for run in $RUNS; do
    rc=$(cat "outputs/train/$run.exit" 2>/dev/null || echo "?")
    last=$(grep -E "step:" "outputs/train/$run.log" 2>/dev/null | tail -1)
    say "      $run  exit=$rc"
    say "         ${last:-<no step lines — check the log>}"
    say "         checkpoints: $(ls outputs/train/$run/checkpoints 2>/dev/null | tr '\n' ' ')"
done

# Side-by-side final loss for the augmentation comparison. Loss across recipes is
# NOT a ranking — a harder augmentation legitimately trains to a higher loss and
# may still deploy better. This is here to spot a run that diverged or died, not
# to pick a winner. Pick the winner on the robot.
if [ -n "${AUG_LIST// /}" ]; then
    say ""
    say "      augmentation comparison (final train loss — NOT a deployment ranking):"
    for aug in $AUG_LIST; do
        l=$(grep -oE "loss:[0-9.]+" "outputs/train/${GR_RUN}_$aug.log" 2>/dev/null | tail -1)
        say "         $(printf '%-11s' "$aug") ${l:-<none>}"
    done
fi
say "=== night run $DAY finished ==="
