#!/usr/bin/env bash
# ACT grasp policy, right arm — night of Wednesday 2026-08-12.
#
# Chained so it survives the operator leaving: waits for the export to finish,
# predecodes the mp4s to JPG (torchcodec cannot load on this box, so this is what
# makes video reading work at all, not just what makes it fast), then trains.
#
# Recipe is the one from sweeps/RESEARCH.md that produced the deployed lineage:
# dark_noise augmentation, batch 8, save every 10k. Steps raised to 100k.
set -uo pipefail
cd "$(dirname "$0")"

RUN=act_grasp_right_night_20260812
REPO=ETHRC/yam_grasp_right_20260812
DS=$HOME/.cache/huggingface/lerobot/$REPO
PD=$HOME/.cache/lerobot-predecoded/yam_grasp_right_20260812
OUT=outputs/train/$RUN
# The log must live OUTSIDE $OUT: LeRobot's cfg.validate() refuses to train into
# a directory that already exists (resume=False), so creating it just to hold a
# log file kills the run before step 0. It did exactly that once.
LOG=outputs/train/$RUN.log

mkdir -p outputs/train
rm -rf "$OUT"
echo "=== $RUN ===" | tee -a "$LOG"
date | tee -a "$LOG"

# ── 1. wait for the export ──────────────────────────────────────────────────
echo "[1/3] waiting for export to finish..." | tee -a "$LOG"
while pgrep -f "export_lerobot.py --root recordings/20260811" >/dev/null; do sleep 5; done
if [ ! -f "$DS/meta/info.json" ]; then
    echo "FATAL: export did not produce $DS/meta/info.json" | tee -a "$LOG"
    tail -20 /tmp/export_20260812.log | tee -a "$LOG"
    exit 1
fi
N_EP=$(python3 -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
echo "      export OK: $N_EP episodes in $DS" | tee -a "$LOG"

# ── 2. predecode ────────────────────────────────────────────────────────────
echo "[2/3] predecoding to $PD ..." | tee -a "$LOG"
./.venv/bin/python3 tools/predecode_ffmpeg.py \
    --dataset-root "$DS" --output-root "$PD" 2>&1 | tail -5 | tee -a "$LOG"

# ── 3. train ────────────────────────────────────────────────────────────────
echo "[3/3] training 100000 steps..." | tee -a "$LOG"
date | tee -a "$LOG"
LEROBOT_PREDECODED_ROOT=$PD ./.venv/bin/python3 tools/train_act_dark_noise.py \
    --dataset.repo_id=$REPO \
    --policy.type=act --policy.device=cuda \
    --policy.push_to_hub=false \
    --steps=100000 --save_freq=10000 --batch_size=8 --seed=1000 \
    --output_dir=$OUT 2>&1 | tee -a "$LOG"

echo "=== finished ===" | tee -a "$LOG"
date | tee -a "$LOG"
