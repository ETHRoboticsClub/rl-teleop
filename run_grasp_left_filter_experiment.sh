#!/usr/bin/env bash
# The A/B/C data-filter experiment from loops/graspdata/REPORT.md (2026-09-04).
#
# A = the incumbent grasp_left_20260902_wristnative (all 126 windows, 150k steps)
#     -- ALREADY TRAINED, nothing to do.
# B = same recipe minus episode_165925 (101 windows). That episode is 20% of the
#     corpus and its leader-vs-follower EE error at the close is median 51 mm vs
#     13-24 mm for every other episode: it supervises "close before arriving".
# C = same recipe minus the 21 individually-flagged windows (105 windows) --
#     window-level curation instead of the episode-level cut.
#
# Everything except the keep-list is byte-identical to run_grasp_left_20260902.sh:
# same window rule, geometry (wrist_native), steps (150k), batch (8), seed (1000),
# aug (dark_noise). One variable per comparison, per data/research/act/06.
#
# EVAL RULE (before anyone celebrates a checkpoint): >=20 rollouts per variant on
# the held-out scenes (episodes 163624 + 164621), scored from the Step Monitor.
# Identical-config runs vary 19-49% success; 3-5 rollouts decide nothing.
#
# Usage:
#   ./run_grasp_left_filter_experiment.sh B        # one variant
#   ./run_grasp_left_filter_experiment.sh C
#   ./run_grasp_left_filter_experiment.sh chain    # B, wait for it, then C
#   ./run_grasp_left_filter_experiment.sh B --dry
set -uo pipefail
cd "$(dirname "$0")"

VARIANT="${1:?usage: $0 B|C|chain [--dry]}"
DRY=0; [ "${2:-}" = "--dry" ] && DRY=1

say(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ "$VARIANT" = chain ]; then
    "$0" B
    RUNB=grasp_left_20260902_B_no165925_wristnative
    say "chain: waiting for $RUNB to finish (poll outputs/train/$RUNB.exit)"
    while [ ! -f "outputs/train/$RUNB.exit" ]; do sleep 120; done
    say "chain: B exited with $(cat outputs/train/$RUNB.exit); waiting 90 s for GPU memory"
    sleep 90
    exec "$0" C
fi

case "$VARIANT" in
  B) RUN=grasp_left_20260902_B_no165925_wristnative
     REPO=ETHRC/yam_grasp_left_20260902_B_no165925
     KEEP=recordings/20260902/keep-train-B-no165925.json
     EXPECT_WINDOWS=101 ;;
  C) RUN=grasp_left_20260902_C_curated_wristnative
     REPO=ETHRC/yam_grasp_left_20260902_C_curated
     KEEP=recordings/20260902/keep-train-C-curated.json
     EXPECT_WINDOWS=105 ;;
  *) echo "unknown variant '$VARIANT'"; exit 2 ;;
esac

DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/$(basename "$REPO")
CACHE=${BASE}_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3

STEPS="${STEPS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-1000}"
AUG="${AUG:-dark_noise}"
WORKERS="${WORKERS:-6}"

OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
mkdir -p outputs/train

FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 40 ]; then
    say "FATAL: under 40 GB free. A full disk truncates checkpoint writes SILENTLY."
    exit 1
fi

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; exit 1; }

if [ "$DRY" = 1 ]; then
    echo "variant $VARIANT: keep=$KEEP expect=$EXPECT_WINDOWS repo=$REPO run=$RUN"
    echo "LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY"
    echo "tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --steps=$STEPS ..."
    exit 0
fi

if [ -f "$DS/meta/info.json" ]; then
    say "[1/4] dataset already exported at $DS -- skipping"
else
    say "[1/4] exporting $REPO ..."
    $PY tools/export_lerobot.py \
        --root recordings/20260902 \
        --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
        --chunk-frames 100 --close-idx-frac 1.00 \
        --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
        --keep "$KEEP" \
        --repo-id "$REPO" 2>&1 | tail -25
    [ -f "$DS/meta/info.json" ] || { say "FATAL: export produced no $DS/meta/info.json"; exit 1; }
fi

N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
say "      exported: $N_EP windows"
if [ "$N_EP" != "$EXPECT_WINDOWS" ]; then
    say "FATAL: expected $EXPECT_WINDOWS windows, got $N_EP. Keep-list and annotations"
    say "       disagree -- re-derive the keep-list rather than training on this."
    exit 1
fi

if [ -d "$BASE" ]; then
    say "[2/4] base cache exists at $BASE -- skipping"
else
    say "[2/4] predecoding to $BASE ..."
    $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -5
    [ -d "$BASE" ] || { say "FATAL: predecode produced nothing"; exit 1; }
fi

if [ -d "$CACHE" ]; then
    say "[3/4] baked cache exists at $CACHE -- skipping"
else
    say "[3/4] baking $GEOM geometry ..."
    $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -5
    [ -d "$CACHE" ] || { say "FATAL: resize produced nothing"; exit 1; }
fi

# LEROBOT_PREDECODED_ROOT MUST be inline on the command, not exported before
# `tmux new-session` (tmux server env trap -- cost a night on 2026-08-14).
CMD="LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='$OUT' > '$LOG' 2>&1; echo \$? > '$OUT.exit';"

say "[4/4] launching $RUN (geometry: $GEOM, $STEPS steps)"
rm -rf "$OUT" "$OUT.exit"
tmux kill-session -t "tr_$RUN" 2>/dev/null
tmux new-session -d -s "tr_$RUN" -c "$PWD" "$CMD"
sleep 25

if tmux has-session -t "tr_$RUN" 2>/dev/null; then
    say "running in tmux session tr_$RUN"
    say "  follow : tail -f $LOG"
else
    say "WARNING: tmux session gone after 25 s -- the run died at startup:"
    tail -25 "$LOG"
    exit 1
fi
tail -15 "$LOG"
