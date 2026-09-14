#!/usr/bin/env bash
# Waits for tonight's recording to finish, then launches the training runner.
# Independent of the agent session: survives it.
cd /home/tommaso/Desktop/kitting-v2/rl-teleop
ARM_T=$(date +%s); DEADLINE=$((ARM_T + 5400)); IDLE=900
LOG=outputs/train/isolate_pick_right_20260914.wait.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "armed; waiting for recorder idle + ${IDLE}s quiet after new files (deadline 90 min if nothing new)"
while true; do
  now=$(date +%s)
  rec=$(curl -s -m 3 http://127.0.0.1:8792/status | grep -o '"recording": *[a-z]*' | grep -o 'true\|false' || echo unknown)
  newest=$(find recordings/20260914 -type f -newermt "@$ARM_T" -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  last=$(find recordings/20260914 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  if [ "$rec" = "false" ] && [ -n "$newest" ] && [ $((now - last)) -ge $IDLE ]; then say "RECORDING FINISHED (last write $(date -d @$last +%H:%M))"; break; fi
  if [ "$rec" != "true" ] && [ -z "$newest" ] && [ $now -ge $DEADLINE ]; then say "NO NEW RECORDING in 90 min: training on the existing takes"; break; fi
  sleep 30
done
say "launching run_isolate_pick_right_20260914.sh"
./run_isolate_pick_right_20260914.sh >> outputs/train/isolate_pick_right_20260914.runner.log 2>&1
say "runner exited rc=$?"
