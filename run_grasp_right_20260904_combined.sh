#!/usr/bin/env bash
# run_grasp_right_20260904_combined.sh — right-arm grasp training on the COMBINED
# corpus: the 4 right episodes of 20260902 (the 104-window set behind
# ETHRC/yam_grasp_right_20260902) + everything recorded on 20260904/20260905.
#
# UNCONDITIONED by design (2026-09-04, Tommaso): no targetdot step, no fallback
# logic — the 3b conditioning stage of run_grasp_right_20260902.sh is deliberately
# absent. targetdot82 showed the dot is ignored without dot-dropout training.
#
# Corpus is UNCLEANED on purpose: no --keep list tonight; the operator reviews
# tomorrow and a curated re-export can follow (same pattern as left B/C variants).
#
# export_lerobot takes ONE --root, so multiple date dirs are combined through a
# staging root of per-episode symlinks (Path.rglob finds direct-child symlinked
# episode dirs on py3.11; it would NOT descend through a symlinked date dir).
# The exporter's --arms right filter skips the left episodes by itself — proven
# by the 20260902 export, whose root also held 11 left episodes.
set -uo pipefail
cd "$(dirname "$0")"

RUN=grasp_right_20260904_combined
REPO=ETHRC/yam_grasp_right_20260904_combined
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/yam_grasp_right_20260904_combined
CACHE=${BASE}_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3
DATES="${DATES:-20260902 20260904 20260905}"   # missing dirs are skipped
STAGE=$HOME/.cache/export-roots/right_20260904_combined
# Steps default to the epoch regime of the run that produced the working right
# checkpoint: grasp_right_20260902 ran 100k @ batch 8 over 13,362 frames = 59.9
# epochs (epch:59.87 in its log). Same regime, only the data changes.
EPOCHS="${EPOCHS:-60}"
SAVE_FREQ="${SAVE_FREQ:-25000}"; BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6
WINDOW_FLOOR="${WINDOW_FLOOR:-200}"
OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
STATUS=outputs/train/$RUN.status
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "=== RIGHT-ARM combined grasp training (UNCONDITIONED, uncleaned) ==="
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 30 ] || { say "FATAL: under 30 GB free (prune ~/.cache/lerobot-predecoded)"; echo "FAIL disk" > "$STATUS"; exit 1; }

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; echo "FAIL geometry" > "$STATUS"; exit 1; }

# ── staging root: symlink every episode of every listed date ────────────────
rm -rf "$STAGE"; mkdir -p "$STAGE"
n_link=0
for d in $DATES; do
  for ep in recordings/$d/episode_*; do
    [ -d "$ep" ] || continue
    ln -s "$(cd "$ep" && pwd)" "$STAGE/$(basename "$ep")"
    n_link=$((n_link+1))
  done
done
say "staged $n_link episodes from: $DATES"
[ "$n_link" -ge 5 ] || { say "FATAL: staging nearly empty — did tonight's recording land in recordings/<date>?"; echo "FAIL staging" > "$STATUS"; exit 1; }

# ── label any episode the live cockpit didn't (tunnel down = no scoring) ────
# Offline labeler is idempotent and never touches raw recordings. Same flags
# as the 20260902 corpus (run_night_20260814.sh lineage): open 1.0 closed 0.0.
for eplink in "$STAGE"/episode_*; do
  ep=$(readlink -f "$eplink")
  [ -f "$ep/yam_right.mcap" ] || continue           # left episodes: not ours
  [ -s "$ep/annotations_right.json" ] && continue
  say "  labeling $(basename "$ep") (no annotations_right.json)"
  $PY -m robots_realtime.labeling.label_episode "$ep" --arm right \
    --open-ref 1.0 --closed-ref 0.0 2>&1 | tail -2 | sed 's/^/    /'
done

# ── pre-export honesty report (frame yield is enforced NOWHERE else) ────────
say "[0/4] horizon report -> outputs/train/$RUN.horizon.txt (read it before trusting the run)"
$PY tools/act_horizon_report.py --root "$STAGE" --arm right --cameras wrist_right_top --pre-s 6.0 > "outputs/train/$RUN.horizon.txt" 2>&1 || say "  (horizon report failed — non-fatal, but look at it)"

if [ -f "$DS/meta/info.json" ]; then
  say "[1/4] dataset already at $DS -- skip"
else
  say "[1/4] exporting $REPO (right, grasp-pose, pre_s 6 / close-idx-frac 1.0) ..."
  $PY tools/export_lerobot.py --root "$STAGE" \
    --arms right --cameras wrist_right_top \
    --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
    --chunk-frames 100 --close-idx-frac 1.00 \
    --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
    --repo-id "$REPO" 2>&1 | tail -20
  [ -f "$DS/meta/info.json" ] || { say "FATAL: export produced nothing"; echo "FAIL export" > "$STATUS"; exit 1; }
fi
N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0)
say "      exported: $N_EP windows, $N_FR frames"
# On floor failure DELETE the export — otherwise a rerun after more recording
# would hit the "already at $DS -- skip" branch and train on the short dataset.
[ "$N_EP" -ge "$WINDOW_FLOOR" ] || { say "FATAL: only $N_EP windows (< $WINDOW_FLOOR floor — override with WINDOW_FLOOR=). Removing $DS so a rerun re-exports."; rm -rf "$DS"; echo "FAIL windows=$N_EP" > "$STATUS"; exit 1; }

STEPS="${STEPS:-$(( EPOCHS * N_FR / BATCH ))}"
say "      steps=$STEPS  (${EPOCHS} epochs over $N_FR frames @ batch $BATCH)"

if [ -d "$BASE" ]; then say "[2/4] base cache exists -- skip"; else
  say "[2/4] predecoding ..."; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -4
  [ -d "$BASE" ] || { say "FATAL: predecode nothing"; echo "FAIL predecode" > "$STATUS"; exit 1; }
fi
if [ -d "$CACHE" ]; then say "[3/4] baked cache exists -- skip"; else
  say "[3/4] baking $GEOM ..."; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -4
  [ -d "$CACHE" ] || { say "FATAL: resize nothing"; echo "FAIL resize" > "$STATUS"; exit 1; }
fi

say "[4/4] launching $RUN | UNCONDITIONED | cache=$CACHE | $STEPS steps"
echo "mode=UNCONDITIONED cache=$CACHE windows=$N_EP frames=$N_FR epochs=$EPOCHS steps=$STEPS dates='$DATES' started=$(date)" > "$STATUS"
rm -rf "$OUT" "$OUT.exit"
tmux kill-session -t "tr_$RUN" 2>/dev/null
CMD="LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.chunk_size=100 --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='$OUT' > '$LOG' 2>&1; echo \$? > '$OUT.exit'"
tmux new-session -d -s "tr_$RUN" "$CMD"
sleep 8
tmux has-session -t "tr_$RUN" 2>/dev/null && say "training launched in tmux tr_$RUN" || { say "FATAL: tmux session did not start"; echo "FAIL tmux" >> "$STATUS"; exit 1; }
say "=== done launching. tail -f $LOG ==="
