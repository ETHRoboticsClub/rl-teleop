#!/usr/bin/env bash
# Train the LEFT-arm grasp policy on the 2026-09-02 teleop session.
#
# Chained so it survives the operator leaving: export -> predecode -> bake the
# deploy geometry into the cache -> train. Each stage gates the next.
#
# DATA. 11 episodes recorded 2026-09-02, 189 successful grasps. The operator
# reviewed all 189 windows in tools/review_grasps.py and kept 179
# (data_grasping_left.json). Three automatic gates then apply:
#     -16  episode_155339_ad02fb72   camera_left delivered 25% of its frames
#                                    (6.2 Hz, 143 gaps, worst 6.88 s). Twelve of
#                                    its windows would enter the set as 12-24
#                                    frame stubs -- shorter than one committed
#                                    action chunk. The review page CANNOT show
#                                    this: it renders 3 frames per window and
#                                    the starvation is in the ~120 between them.
#      -4  hold_s < 1.5 s
#     -19  close-index gate (see WINDOW below)
#   -> 140 windows, of which 14 are held out as two WHOLE episodes
#      (163624, 164621 -- both 100% frame yield) leaving 126 to train on.
#   Keep-lists: recordings/20260902/keep-train.json / keep-heldout.json
#
# WINDOW. --pre-s 6.0 --close-idx-frac 1.00, NOT the exporter's 3.0/0.8 defaults.
# close_idx is bounded by pre_s * fps, so at pre_s=3 the ceiling is frame 90 and
# 181 of 189 windows sit exactly on it -- "the working corpus sat at 0.90 of the
# chunk" is arithmetic (3.0*30/100), not evidence, and the gate cannot fire for
# the reason its docstring gives. Widen the search span and 24 of 189 windows
# turn out to have a real descent longer than one 100-frame chunk; at pre_s=3
# they were invisible, silently truncated to 90 and opening mid-descent.
# Full argument: ACT-DATA-READINESS-REPORT-20260902.md S3.
#
# GEOMETRY: wrist_native -- and it needs no rig change, same as the 20260815
# left lineage.
#     wrist  cameras_left_only.yaml sets NO publish_resize, so the bus carries
#            the native 640x480 = 480x640 (H,W). The mp4 is the same. Match.
#     top    the mp4 is 720x1280 but cameras_right_top.yaml publishes camera_top
#            at 270x480 (mode: pad). Training on the mp4 would declare a
#            checkpoint 2.67x off the geometry it is handed at inference -- the
#            error that cost ~57-65 mm of lateral tip on the right arm.
# tools/act_bus_geometry.py resizes through the SAME function camera_node
# publishes through (openpi_client.image_tools.resize_with_pad) and its selftest
# asserts the two agree byte for byte. 1280/720 == 480/270, so `pad` adds no
# letterboxing here -- it is a pure downscale, exactly as on the bus.
#
# STEPS. 150k at batch 8 over ~15.6k frames is ~77 epochs. The proven lineage
# (handoff_left_wristnative) ran 100k over 5,372 frames = 149 epochs. This
# corpus is ~3x bigger, so 150k is the CONSERVATIVE end in epoch terms, not the
# aggressive one. save_freq 10000 gives 15 checkpoints to pick from.
#
# Usage:
#   ./run_grasp_left_20260902.sh          # launch
#   ./run_grasp_left_20260902.sh --dry    # print the training command, run nothing
set -uo pipefail
cd "$(dirname "$0")"

DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

RUN=grasp_left_20260902_wristnative
REPO=ETHRC/yam_grasp_left_20260902
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/yam_grasp_left_20260902
CACHE=${BASE}_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3

STEPS="${STEPS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-1000}"
AUG="${AUG:-dark_noise}"
WORKERS="${WORKERS:-6}"
EXPECT_WINDOWS=126

# The log lives OUTSIDE $OUT on purpose: LeRobot's cfg.validate() refuses to
# train into a directory that already exists with resume=False, so creating it
# just to hold a log file kills the run before step 0. It did exactly that once.
OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
mkdir -p outputs/train

say(){ echo "[$(date +%H:%M:%S)] $*"; }

# ── gate: disk ───────────────────────────────────────────────────────────────
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 40 ]; then
    say "FATAL: under 40 GB free. A full disk truncates checkpoint writes SILENTLY."
    exit 1
fi

# ── gate: the geometry resize matches the bus byte for byte ──────────────────
say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; exit 1; }

if [ "$DRY" = 1 ]; then
    echo "LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY"
    echo "tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act"
    echo "--policy.device=cuda --policy.push_to_hub=false --steps=$STEPS"
    echo "--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS"
    echo "--output_dir='$OUT'"
    exit 0
fi

# ── 1. export ────────────────────────────────────────────────────────────────
if [ -f "$DS/meta/info.json" ]; then
    say "[1/4] dataset already exported at $DS -- skipping"
else
    say "[1/4] exporting $REPO ..."
    $PY tools/export_lerobot.py \
        --root recordings/20260902 \
        --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
        --chunk-frames 100 --close-idx-frac 1.00 \
        --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
        --keep recordings/20260902/keep-train.json \
        --repo-id "$REPO" 2>&1 | tail -25
    [ -f "$DS/meta/info.json" ] || { say "FATAL: export produced no $DS/meta/info.json"; exit 1; }
fi

N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])" 2>/dev/null || echo "?")
say "      exported: $N_EP windows, $N_FR frames"
if [ "$N_EP" != "$EXPECT_WINDOWS" ]; then
    say "FATAL: expected $EXPECT_WINDOWS windows, got $N_EP. The keep-list and the"
    say "       annotations disagree -- re-run the review rather than training on this."
    exit 1
fi

# ── 2. predecode ─────────────────────────────────────────────────────────────
# NOT an optimisation: torchcodec cannot load on this box (no libavutil), so
# this is what makes video reading work at all.
if [ -d "$BASE" ]; then
    say "[2/4] base cache exists at $BASE -- skipping"
else
    say "[2/4] predecoding to $BASE ..."
    $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -5
    [ -d "$BASE" ] || { say "FATAL: predecode produced nothing"; exit 1; }
fi

# ── 3. bake the deploy geometry into the cache ───────────────────────────────
if [ -d "$CACHE" ]; then
    say "[3/4] baked cache exists at $CACHE -- skipping"
else
    say "[3/4] baking $GEOM geometry ..."
    $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -5
    [ -d "$CACHE" ] || { say "FATAL: resize produced nothing"; exit 1; }
fi

# ── 4. train ─────────────────────────────────────────────────────────────────
# LEROBOT_PREDECODED_ROOT MUST be inline on the command, not exported before
# `tmux new-session`: a tmux session inherits the tmux SERVER's environment,
# captured when the server first started, not the client's. Exporting it reaches
# nothing, the run silently falls back to torchcodec and dies within seconds.
# This cost a whole night on 2026-08-14.
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
    say "  attach : tmux attach -t tr_$RUN"
else
    say "WARNING: tmux session gone after 25 s -- the run died at startup:"
    tail -25 "$LOG"
    exit 1
fi
tail -15 "$LOG"
