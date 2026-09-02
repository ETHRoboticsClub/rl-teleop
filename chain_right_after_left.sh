#!/usr/bin/env bash
# Waits for the LEFT training (PID 2505198) to exit, then runs the right pipeline.
# Never signals the left process -- only polls its liveness. Safe to leave running.
set -uo pipefail
cd "$(dirname "$0")"
LOG=outputs/train/chain_right.log
mkdir -p outputs/train
exec >> "$LOG" 2>&1
echo "[$(date)] chain armed; waiting for left training PID 2505198 to finish"
while kill -0 2505198 2>/dev/null; do sleep 60; done
echo "[$(date)] left training PID 2505198 has exited -- waiting 90s for GPU memory to release"
sleep 90
echo "[$(date)] launching right pipeline"
bash run_grasp_right_20260902.sh
echo "[$(date)] right pipeline launcher returned (training now runs in tmux tr_grasp_right_20260902)"
