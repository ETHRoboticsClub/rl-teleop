#!/usr/bin/env bash
# Right-arm grasp ACT training, queued to run after the left training finishes.
# Built 2026-09-02 (operator away). Mirrors run_grasp_left_20260902.sh with right
# flags; adds a BEST-EFFORT dot-conditioning stage. If conditioning prep fails,
# trains the unconditioned baseline so the GPU night is never wasted -- and says so.
set -uo pipefail
cd "$(dirname "$0")"
RUN=grasp_right_20260902
REPO=ETHRC/yam_grasp_right_20260902
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/yam_grasp_right_20260902
CACHE=${BASE}_wristnative
DOTCACHE=${CACHE}_targetdot
GEOM=wrist_native
PY=./.venv/bin/python3
STEPS="${STEPS:-100000}"
SAVE_FREQ=10000; BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6
WINDOW_FLOOR=95
OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
STATUS=outputs/train/$RUN.status
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "=== RIGHT-ARM grasp training pipeline ==="
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 40 ] || { say "FATAL: under 40 GB free"; echo "FAIL disk" > "$STATUS"; exit 1; }

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; echo "FAIL geometry" > "$STATUS"; exit 1; }

# 1. EXPORT (auto-filtered, no keep-list -- operator away)
if [ -f "$DS/meta/info.json" ]; then
  say "[1/4] dataset already at $DS -- skip"
else
  say "[1/4] exporting $REPO (right, grasp-pose, pre_s 6 / close-idx-frac 1.0) ..."
  $PY tools/export_lerobot.py --root recordings/20260902 \
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
[ "$N_EP" -ge "$WINDOW_FLOOR" ] || { say "FATAL: only $N_EP windows (< $WINDOW_FLOOR floor)"; echo "FAIL windows=$N_EP" > "$STATUS"; exit 1; }

# 2-3. PREDECODE + BAKE GEOMETRY (baseline cache)
if [ -d "$BASE" ]; then say "[2/4] base cache exists -- skip"; else
  say "[2/4] predecoding ..."; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -4
  [ -d "$BASE" ] || { say "FATAL: predecode nothing"; echo "FAIL predecode" > "$STATUS"; exit 1; }
fi
if [ -d "$CACHE" ]; then say "[3/4] baked cache exists -- skip"; else
  say "[3/4] baking $GEOM ..."; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -4
  [ -d "$CACHE" ] || { say "FATAL: resize nothing"; echo "FAIL resize" > "$STATUS"; exit 1; }
fi

# 3b. CONDITIONING (best-effort): SAM2 label -> dot-dropout burn. On any failure, fall back to baseline.
TRAIN_CACHE="$CACHE"; MODE="UNCONDITIONED-baseline"
if [ -d "$DOTCACHE" ]; then
  say "[3b] dotted cache already exists -- using conditioned"; TRAIN_CACHE="$DOTCACHE"; MODE="CONDITIONED-targetdot"
else
  say "[3b] attempting dot-conditioning (SAM2 label + dropout burn) ..."
  if bash tools/targetdot/make_conditioned_cache.sh "$DS" "$CACHE" "$DOTCACHE" > outputs/train/$RUN.cond.log 2>&1; then
    if [ -d "$DOTCACHE" ]; then TRAIN_CACHE="$DOTCACHE"; MODE="CONDITIONED-targetdot"; say "[3b] conditioning OK -- training CONDITIONED"; fi
  else
    say "[3b] CONDITIONING FAILED (see $RUN.cond.log) -- falling back to UNCONDITIONED baseline"
  fi
fi

# 4. TRAIN
say "[4/4] launching $RUN | mode=$MODE | cache=$TRAIN_CACHE | $STEPS steps"
echo "mode=$MODE cache=$TRAIN_CACHE windows=$N_EP steps=$STEPS started=$(date)" > "$STATUS"
rm -rf "$OUT" "$OUT.exit"
tmux kill-session -t "tr_$RUN" 2>/dev/null
CMD="LEROBOT_PREDECODED_ROOT='$TRAIN_CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.chunk_size=100 --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='$OUT' > '$LOG' 2>&1; echo \$? > '$OUT.exit'"
tmux new-session -d -s "tr_$RUN" "$CMD"
sleep 8
tmux has-session -t "tr_$RUN" 2>/dev/null && say "training launched in tmux tr_$RUN ($MODE)" || { say "FATAL: tmux session did not start"; echo "FAIL tmux" >> "$STATUS"; exit 1; }
say "=== done launching. tail -f $LOG ==="
