#!/usr/bin/env bash
# Retrain policy A with the image geometry MATCHED to what the bus publishes.
#
# WHY: every right-arm checkpoint so far declares the mp4 resolution (wrist
# 480x640, top 720x1280) and is deployed against bus frames at 240x320 / 270x480
# -- 18% of the vision tokens it trained on. Measured cost: ~57-65 mm of
# commanded tip x, z untouched, i.e. the lateral empty-grasp signature. Nothing
# raises, because ACT is resolution-agnostic at runtime. See
# tools/act_bus_geometry.py for the full mechanism.
#
# NO NEW DATA. Both lanes train on ETHRC/yam_pickplace_right_20260814 exactly as
# it sits on disk -- 82 episodes, 18,213 frames, the same corpus the deployed
# policy A used. act_bus_geometry resizes frames as they leave the decoder and
# redeclares the feature shapes, so this costs no disk and no re-export and
# reverts by unsetting one env var.
#
# TWO LANES, ONE VARIABLE. Both are dark_noise, both 200k steps, both seed 1000
# -- identical to the original policy A run. The ONLY thing that differs between
# them is the wrist resolution:
#
#   L1  bus           wrist 240x320  top 270x480   215 tokens
#       Deployable the moment it finishes: cameras_right_top.yaml already
#       publishes exactly this, act_runner already passes it straight through.
#
#   L2  wrist_native  wrist 480x640  top 270x480   435 tokens
#       The wrist at ACT/ALOHA canon (300 tokens, the paper's configuration).
#       Answers the one real risk in L1: whether 80 wrist tokens is enough to
#       find a packet. To DEPLOY this one you must first delete publish_resize
#       + publish_resize_mode from the camera_right node in
#       configs/yam/cameras_right_top.yaml, restart `cams` (NOT `arm`), and
#       confirm with tools/check_streams.py --watch 120 that the wrist still
#       holds 30 Hz at 27.6 MB/s through the ZMQ proxy.
#
# Augmentation is held constant on purpose. Four policy-B augmentation
# checkpoints are already on disk with no robot evaluation; adding a fifth
# untested variable here would make L1 vs L2 uninterpretable.
#
# Usage:
#   ./run_matched_20260815.sh          # launch both lanes
#   ./run_matched_20260815.sh --dry    # print what would run, launch nothing
set -uo pipefail
cd "$(dirname "$0")"

DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

# WHICH CORPUS. Overridable, and the choice is NOT cosmetic -- it must match the
# pose the deploy pipeline hands the policy at handover. Measured 2026-08-15,
# median tip at each export's first frame, against the x +0.472 y -0.011 z +0.130
# that goto_start_right.py actually produces:
#
#   yam_grasp_right_20260814     y +0.011   <- 2 cm out. THE ONE THAT FITS.
#   yam_grasp_right_20260812     y -0.125   <- 11 cm out
#   yam_pickplace_right_20260814 y +0.130   <- 14 cm out
#
# The pickplace window opens 4 s BEFORE the close, while the arm is still
# travelling back from the mat, so its demos start mid-transit. run_act_right.py
# is the ~3 s grasp window and parks the arm over the bin first. Train the
# pickplace corpus and hand it to this pipeline and the policy is out of
# distribution at tick 0 -- measured: descends to the right height, drifts 6 cm
# wide in y, never closes. 0/5 on hardware.
REPO="${REPO:-ETHRC/yam_grasp_right_20260814}"
PD_BASE="${PD_BASE:-$HOME/.cache/lerobot-predecoded/yam_grasp_right_20260814}"
RUN_PREFIX="${RUN_PREFIX:-graspmatched}"
PY=./.venv/bin/python3
# 100k, not the 200k policy A originally ran. Operator's call 2026-08-15 and the
# evidence backs it: 100k at batch 8 over 18,213 frames is 44 epochs, and the
# 20260812 grasp run -- the only lineage proven to work on this arm -- converged
# at 135 epochs on a quarter of the data. Policy A's own curve was already flat
# by then (0.045 at 100k, 0.031 at 200k, i.e. 100k more steps bought 0.014 of
# training loss, which ranks nothing on the robot).
#
# The first two runs were LAUNCHED at 200000 and stopped after the 100000
# checkpoint landed, so their train_config.json still says 200000. That is
# cosmetic -- but it also means `--resume=true` against checkpoints/100000 will
# happily continue to 200k if the 100k policy turns out to be under-trained.
STEPS="${STEPS:-100000}"
# 4 saves per lane. Each checkpoint is 591 MB, so two lanes at this frequency
# cost 4.7 GB -- the binding constraint on this box, not GPU time.
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-1000}"
AUG="${AUG:-dark_noise}"
# The geometry is BAKED INTO THE CACHE (tools/predecode_resize.py), so the
# dataloader only decodes small JPGs and never resizes. That is why 6 is enough.
# Measured the other two ways first, both worse, both on this box, 2 lanes:
#   resize-on-read, 4 workers    updt_s 0.036  data_s 0.093   0.129 s/step
#   resize-on-read, 10 workers   updt_s 0.101  data_s 0.039   0.140 s/step, GPU 24%
# Raising workers only moved the cost between the counters -- the box was
# CPU-bound on the resize itself. Baking it removed the work instead.
WORKERS="${WORKERS:-6}"

# lane name -> geometry (the cache suffix is the geometry with _ stripped)
declare -A LANES=( [${RUN_PREFIX}_bus]=bus [${RUN_PREFIX}_wristnative]=wrist_native )
declare -A CACHES=( [bus]="${PD_BASE}_bus" [wrist_native]="${PD_BASE}_wristnative" )

mkdir -p outputs/train logs

# ── gates ───────────────────────────────────────────────────────────────────
for geom in "${!CACHES[@]}"; do
    if [ ! -d "${CACHES[$geom]}" ]; then
        echo "FATAL: no baked cache for geometry '$geom' at ${CACHES[$geom]}."
        echo "       Without a predecoded cache training falls back to torchcodec,"
        echo "       which cannot load on this box (no libavutil). Build it with:"
        echo "         $PY tools/predecode_resize.py --geometry $geom \\"
        echo "             --source $PD_BASE --dest ${CACHES[$geom]}"
        exit 1
    fi
done

FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
echo "free disk : ${FREE_GB} GB   (need ~5 GB for 2 lanes x 4 checkpoints)"
if [ "$FREE_GB" -lt 12 ]; then
    echo "FATAL: under 12 GB free. A full disk truncates checkpoint writes"
    echo "       SILENTLY (DATA-PIPELINE.md 2.6). Free space first."
    exit 1
fi

# Prove the resize is the camera node's before spending 3 h of GPU on it.
echo "geometry selftest:"
$PY tools/act_bus_geometry.py || { echo "FATAL: geometry selftest failed"; exit 1; }

build() {   # build(run_name, geometry) -> shell string
    local run=$1 geom=$2
    # LEROBOT_PREDECODED_ROOT MUST be inline on the command, not exported before
    # `tmux new-session`: a tmux session inherits the tmux SERVER's environment,
    # captured whenever the server first started, not the client's. Exporting it
    # reaches nothing, every run silently falls back to torchcodec, and all of
    # them die within seconds. This cost a whole night on 2026-08-14.
    # ACT_GEOMETRY stays set even though the cache is already baked: the resize
    # is then a no-op, but the feature shapes are still redeclared, so pointing
    # this at a stale full-res cache by mistake trains correctly-but-slower
    # instead of silently wrong.
    echo "LEROBOT_PREDECODED_ROOT='${CACHES[$geom]}' ACT_AUG=$AUG ACT_GEOMETRY=$geom $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='outputs/train/$run' > 'outputs/train/$run.log' 2>&1; \
echo \$? > 'outputs/train/$run.exit';"
}

for run in "${!LANES[@]}"; do
    geom=${LANES[$run]}
    echo
    echo "── $run  (geometry: $geom) ──"
    if [ "$DRY" = 1 ]; then
        build "$run" "$geom" | tr ';' '\n'
        continue
    fi
    rm -rf "outputs/train/$run" "outputs/train/$run.exit"
    tmux kill-session -t "tr_$run" 2>/dev/null
    tmux new-session -d -s "tr_$run" -c "$PWD" "$(build "$run" "$geom")"
    sleep 2
    tmux has-session -t "tr_$run" 2>/dev/null \
        && echo "  tmux tr_$run up -> outputs/train/$run.log" \
        || echo "  WARNING: tr_$run did NOT start"
done

[ "$DRY" = 1 ] && exit 0

cat <<EOF

watch:    tail -f outputs/train/matched_bus.log
compare:  grep -h 'loss:' outputs/train/matched_*.log | tail
probe:    cd ../yam-pick-pipeline && YAM_ARM=right ./probe_policy_a.py \\
            ../rl-teleop/outputs/train/matched_bus/checkpoints/200000/pretrained_model --steps 300

Both lanes declare the geometry they trained at, so probe_policy_a.py and
act_runner see the real shape in config.json. matched_bus needs no rig change;
matched_wristnative needs publish_resize removed from camera_right first.
EOF
