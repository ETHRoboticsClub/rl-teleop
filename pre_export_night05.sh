#!/usr/bin/env bash
# Pre-export remaining night05 datasets while the GPU trains (CPU-only lane).
# Atomic publish: build in staging, mv to the HF cache path only when complete —
# the training queue either finds a finished dataset or safely builds its own.
# (Recreated 06:07 after the 05:58 reboot wiped /tmp, where it first lived.)
set -uo pipefail
cd /home/tommaso/Desktop/kitting-v2/rl-teleop
PY=./.venv/bin/python3
ROOT=$HOME/.cache/export-roots/right_night_20260905
HF=$HOME/.cache/huggingface/lerobot/ETHRC
STG=$HOME/.cache/night05-staging
mkdir -p "$STG"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
build(){  # key cams chunk pre extra...
  local key=$1 cams=$2 chunk=$3 pre=$4; shift 4
  local final="$HF/yam_grasp_right_night05_$key"
  [ -f "$final/meta/info.json" ] && { say "$key: final exists — skip"; return; }
  rm -rf "$STG/$key"
  say "$key: exporting to staging ..."
  if $PY tools/export_lerobot.py --root "$ROOT" --arms right --cameras "$cams" \
      --window-mode grasp-pose --pre-s "$pre" --post-s 2.0 \
      --chunk-frames "$chunk" --close-idx-frac 1.00 \
      --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
      --repo-id "ETHRC/yam_grasp_right_night05_$key" --out "$STG/$key" "$@" \
      > "$STG/$key.log" 2>&1 && [ -f "$STG/$key/meta/info.json" ]; then
    if [ -f "$final/meta/info.json" ]; then say "$key: queue built it meanwhile — discarding staging"; rm -rf "$STG/$key"
    else mv "$STG/$key" "$final"; say "$key: PUBLISHED"; fi
  else say "$key: EXPORT FAILED (see $STG/$key.log) — leaving staging for autopsy"; fi
}
build plus1s_both wrist_right_top 130  8.0 --pregrasp-margin-s 1.5
build apex_both   wrist_right_top 200 10.0 --pregrasp-plane-m 0.15
build double_both wrist_right_top 200 10.0 --pregrasp-margin-s 3.0
build triple_both wrist_right_top 300 12.0 --pregrasp-margin-s 5.5
say "pre-export lane complete"
