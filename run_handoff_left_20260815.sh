#!/usr/bin/env bash
# Train the LEFT-arm handoff-grasp policy on the 2026-08-15 teleop takes, using
# the exact recipe that produced `matched_wristnative` (see
# run_matched_20260815.sh for why that recipe exists at all).
#
# DATA. ETHRC/yam_handoff_left_20260815 -- exported 2026-08-15 from
# recordings/20260815, the two takes recorded this morning:
#     episode_100759_9bcd2952   208 s   7 grasps  (1 bag, re-gripped)
#     episode_101622_18585b2d   376 s  29 grasps  (7 bags, 7 place cycles)
# --window-mode grasp cut those 36 successful grasps into 36 training windows
# (3 s before jaw-close -> 2 s after), 5,372 frames, 0 dropped. Both takes are
# `degraded: false` with all three cameras at full rate.
#
# GEOMETRY: wrist_native -- AND UNLIKE THE RIGHT ARM IT NEEDS NO RIG CHANGE.
# The right-arm lane had to have publish_resize deleted from camera_right before
# it could be deployed. The left arm is already there:
#     wrist  configs/yam/yam_left_handoff_teleop.yaml sets NO publish_resize on
#            camera_left, so the bus carries the native 640x480 = 480x640 (H,W).
#     top    cameras_right_top.yaml publishes camera_top at 270x480 and that is
#            what check_streams sees on the bus right now.
# That pair IS the `wrist_native` preset in tools/act_bus_geometry.py, so the
# checkpoint will declare exactly what the bus will hand it. The train/deploy
# scale error this whole line of work exists to kill is closed by construction
# here rather than by a follow-up config edit.
#
# CAVEAT FOR DEPLOY, NOT FOR TRAINING: the live `cams` session publishes
# camera_right + camera_top only. Nothing publishes camera_left/rgb yet, so the
# bimanual camera config still has to add that node before this checkpoint can
# run. Adding it must NOT add publish_resize, or the match above breaks.
#
# HELD IDENTICAL to matched_wristnative: dark_noise augmentation, batch 8,
# seed 1000, 6 workers, 100k steps, save every 50k, geometry baked into the
# cache by tools/predecode_resize.py.
#
# STEPS. 100k at batch 8 over 5,372 frames is 149 epochs. That is deliberately
# NOT the 44 epochs matched_wristnative ran: this is a small grasp-window set,
# and the 20260812 grasp run -- the only lineage proven to work on this arm --
# converged at 135 epochs on a comparable corpus. Same step count, same epoch
# regime as the thing that worked, because the dataset shrank.
#
# Usage:
#   ./run_handoff_left_20260815.sh          # launch
#   ./run_handoff_left_20260815.sh --dry    # print, launch nothing
set -uo pipefail
cd "$(dirname "$0")"

DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

RUN=handoff_left_wristnative
REPO=ETHRC/yam_handoff_left_20260815
CACHE=$HOME/.cache/lerobot-predecoded/yam_handoff_left_20260815_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3

STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-1000}"
AUG="${AUG:-dark_noise}"
WORKERS="${WORKERS:-6}"

mkdir -p outputs/train logs

# ── gates ───────────────────────────────────────────────────────────────────
if [ ! -d "$CACHE" ]; then
    echo "FATAL: no baked cache at $CACHE."
    echo "       Without one, training falls back to torchcodec, which cannot"
    echo "       load on this box (no libavutil). Build it with:"
    echo "         $PY tools/predecode_ffmpeg.py \\"
    echo "             --dataset-root \$HOME/.cache/huggingface/lerobot/$REPO \\"
    echo "             --output-root  \${CACHE%_wristnative}"
    echo "         $PY tools/predecode_resize.py --geometry $GEOM \\"
    echo "             --source \${CACHE%_wristnative} --dest $CACHE"
    exit 1
fi

FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
echo "free disk : ${FREE_GB} GB   (need ~1.2 GB for 2 checkpoints at 591 MB)"
if [ "$FREE_GB" -lt 4 ]; then
    echo "FATAL: under 4 GB free. A full disk truncates checkpoint writes"
    echo "       SILENTLY (DATA-PIPELINE.md 2.6). Free space first."
    exit 1
fi

echo "geometry selftest:"
$PY tools/act_bus_geometry.py || { echo "FATAL: geometry selftest failed"; exit 1; }

# LEROBOT_PREDECODED_ROOT MUST be inline on the command, not exported before
# `tmux new-session`: a tmux session inherits the tmux SERVER's environment,
# captured when the server first started, not the client's. Exporting it reaches
# nothing, the run silently falls back to torchcodec and dies within seconds.
# This cost a whole night on 2026-08-14.
CMD="LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='outputs/train/$RUN' > 'outputs/train/$RUN.log' 2>&1; \
echo \$? > 'outputs/train/$RUN.exit';"

echo
echo "── $RUN  (geometry: $GEOM) ──"
if [ "$DRY" = 1 ]; then
    echo "$CMD" | tr ';' '\n'
    exit 0
fi

rm -rf "outputs/train/$RUN" "outputs/train/$RUN.exit"
tmux kill-session -t "tr_$RUN" 2>/dev/null
tmux new-session -d -s "tr_$RUN" -c "$PWD" "$CMD"
sleep 2
tmux has-session -t "tr_$RUN" 2>/dev/null \
    && echo "  tmux tr_$RUN up -> outputs/train/$RUN.log" \
    || echo "  WARNING: tr_$RUN did NOT start"

cat <<EOF

watch:  tail -f outputs/train/$RUN.log
loss:   grep -h 'loss:' outputs/train/$RUN.log | tail
EOF
