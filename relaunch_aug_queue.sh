#!/usr/bin/env bash
# Relaunch policy B's augmentation comparison after the 2026-08-14 05:38 failure.
#
# WHAT FAILED: run_night_20260814.sh exported LEROBOT_PREDECODED_ROOT and then
# called `tmux new-session`. A tmux session inherits the tmux SERVER's environment
# (captured whenever the server first started), NOT the environment of the client
# creating it — so the variable reached nothing, every run fell back to torchcodec,
# and torchcodec cannot load on this box at all (no libavutil). All four runs died
# within seconds. Fixed in run_night_20260814.sh too; here the assignment is inline
# on the command, which is the only form that survives the tmux boundary.
#
# WHY TWO LANES AND NOT ONE QUEUE: measured on policy A, a two-camera ACT step
# costs 0.128 s (updt 0.104 + data 0.024), four times the 0.032 s of the
# single-camera lineage. Four sequential 100k runs would be ~14 h. Policy A also
# measured 11.07 GB of VRAM, so two concurrent runs fit a 32 GB card with room and
# three do not. Two lanes of two ≈ 7 h and stays inside the envelope.
set -uo pipefail
cd "$(dirname "$0")"

DAY=20260814
GR_REPO=ETHRC/yam_grasp_right_$DAY
GR_RUN=act_grasp_right_night_$DAY
PD=$HOME/.cache/lerobot-predecoded/yam_grasp_right_$DAY
PY=./.venv/bin/python3
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
BATCH="${BATCH:-8}"

if [ ! -d "$PD" ]; then
    echo "FATAL: no predecoded cache at $PD — training would fall back to"
    echo "       torchcodec, which cannot load here. Run predecode_ffmpeg.py first."
    exit 1
fi

# One lane = recipes run one after another in a single tmux session.
LANE_A="${LANE_A:-dark_noise geometric}"
LANE_B="${LANE_B:-none occlusion}"

build() {                      # build(recipes...) -> queued shell string
    local q=""
    for aug in $1; do
        rm -rf "outputs/train/${GR_RUN}_$aug" "outputs/train/${GR_RUN}_$aug.exit"
        q="$q LEROBOT_PREDECODED_ROOT='$PD' ACT_AUG=$aug $PY tools/train_act_dark_noise.py \
            --dataset.repo_id='$GR_REPO' --policy.type=act --policy.device=cuda \
            --policy.push_to_hub=false --steps=$STEPS --save_freq=$SAVE_FREQ \
            --batch_size=$BATCH --seed=2000 \
            --output_dir='outputs/train/${GR_RUN}_$aug' \
            > 'outputs/train/${GR_RUN}_$aug.log' 2>&1; \
            echo \$? > 'outputs/train/${GR_RUN}_$aug.exit';"
    done
    echo "$q"
}

rm -f "outputs/train/${GR_RUN}_ALL.exit"
for lane in a b; do tmux kill-session -t "train_gr_$lane" 2>/dev/null; done

tmux new-session -d -s train_gr_a -c "$PWD" "$(build "$LANE_A") echo done > outputs/train/${GR_RUN}_LANE_A.exit"
tmux new-session -d -s train_gr_b -c "$PWD" "$(build "$LANE_B") echo done > outputs/train/${GR_RUN}_LANE_B.exit"
sleep 3
for lane in a b; do
    tmux has-session -t "train_gr_$lane" 2>/dev/null \
        && echo "  train_gr_$lane up" || echo "  WARNING: train_gr_$lane did NOT start"
done
echo "  lane A: $LANE_A"
echo "  lane B: $LANE_B"

# Mark the whole comparison done once both lanes finish, so the still-polling
# run_night_20260814.sh can print its verdict.
tmux kill-session -t aug_watch 2>/dev/null
tmux new-session -d -s aug_watch -c "$PWD" \
    "while [ ! -f outputs/train/${GR_RUN}_LANE_A.exit ] || [ ! -f outputs/train/${GR_RUN}_LANE_B.exit ]; do sleep 60; done; echo done > outputs/train/${GR_RUN}_ALL.exit"
echo "  aug_watch up — writes ${GR_RUN}_ALL.exit when both lanes finish"
