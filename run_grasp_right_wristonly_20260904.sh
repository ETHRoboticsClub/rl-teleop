#!/usr/bin/env bash
# Right-arm WRIST-ONLY camera ablation, 2026-09-04.
#
# Question: does the top camera contribute anything to the right-arm grasp policy,
# or is it mostly noise (workshop background, people, the left arm, the kit box)?
# At the trained 480x270 the bin occupies ~17% of the top frame; AV-ALOHA-style
# within-trajectory ablation (data/research/act/03) says fewer views often wins.
#
# Design: byte-identical to run_grasp_right_20260902.sh EXCEPT --cameras
# wrist_right (drops observation.images.top). Same corpus, same auto-gates, same
# window flags, same steps/batch/seed/aug/geometry -- the camera set is the ONLY
# variable. The export must reproduce the baseline's exact 104 windows or the
# A/B is invalid; that is a hard gate below, not a floor.
#
# Model inputs after this change: observation.images.wrist (480x640x3, right
# wrist, wrist-native pass-through) + observation.state (7 = 6 joints + gripper).
# No conditioning stage -- this is an UNCONDITIONED ablation against the
# UNCONDITIONED grasp_right_20260902 baseline (ckpt 100000, exit 0).
#
# Deliberately NOT cleaned (graspdata audit's 10 flagged windows stay in): the
# baseline trained on them, so removing them here would confound the ablation.
# Cleaning + held-out carving belong to the next production retrain.
set -uo pipefail
cd "$(dirname "$0")"
RUN=grasp_right_20260902_wristonly
REPO=ETHRC/yam_grasp_right_20260902_wristonly
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/yam_grasp_right_20260902_wristonly
CACHE=${BASE}_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3
STEPS="${STEPS:-100000}"
SAVE_FREQ=10000; BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6
BASELINE_WINDOWS=104   # what run_grasp_right_20260902.sh exported; must match exactly
OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
STATUS=outputs/train/$RUN.status
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "=== RIGHT-ARM wrist-only ablation pipeline ==="
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 40 ] || { say "FATAL: under 40 GB free"; echo "FAIL disk" > "$STATUS"; exit 1; }

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; echo "FAIL geometry" > "$STATUS"; exit 1; }

# 1. EXPORT -- same corpus + flags as the baseline, camera set is the only change
if [ -f "$DS/meta/info.json" ]; then
  say "[1/4] dataset already at $DS -- skip"
else
  say "[1/4] exporting $REPO (right, WRIST ONLY, grasp-pose, pre_s 6 / close-idx-frac 1.0) ..."
  $PY tools/export_lerobot.py --root recordings/20260902 \
    --arms right --cameras wrist_right \
    --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
    --chunk-frames 100 --close-idx-frac 1.00 \
    --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
    --repo-id "$REPO" 2>&1 | tail -20
  [ -f "$DS/meta/info.json" ] || { say "FATAL: export produced nothing"; echo "FAIL export" > "$STATUS"; exit 1; }
fi
N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0)
say "      exported: $N_EP windows, $N_FR frames"
if [ "$N_EP" -ne "$BASELINE_WINDOWS" ]; then
  say "FATAL: $N_EP windows != baseline's $BASELINE_WINDOWS -- corpora differ, A/B invalid."
  say "       (Exporter or gates changed since 2026-09-02. Diff before training.)"
  echo "FAIL windows=$N_EP expected=$BASELINE_WINDOWS" > "$STATUS"; exit 1
fi

# 2-3. PREDECODE + BAKE GEOMETRY
if [ -d "$BASE" ]; then say "[2/4] base cache exists -- skip"; else
  say "[2/4] predecoding ..."; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -4
  [ -d "$BASE" ] || { say "FATAL: predecode nothing"; echo "FAIL predecode" > "$STATUS"; exit 1; }
fi
if [ -d "$CACHE" ]; then say "[3/4] baked cache exists -- skip"; else
  say "[3/4] baking $GEOM ..."; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -4
  [ -d "$CACHE" ] || { say "FATAL: resize nothing"; echo "FAIL resize" > "$STATUS"; exit 1; }
fi

# 4. TRAIN -- same regime as baseline: 100k steps @ batch 8 over the same frames
say "[4/4] launching $RUN | mode=WRISTONLY-ablation | cache=$CACHE | $STEPS steps"
echo "mode=WRISTONLY-ablation baseline=grasp_right_20260902 cache=$CACHE windows=$N_EP steps=$STEPS started=$(date)" > "$STATUS"
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
