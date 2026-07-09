#!/usr/bin/env bash
# One command to start a kitting recording session:
#   1. live label backend (cockpit /state + camera JPEG bridge)
#   2. teleop session (records yam_left.mcap etc., TUI in this terminal)
#   3. auto-opens viser + the cockpit in Firefox on the rig display
#
# Prereq (needs sudo, do once per boot): bring up the left follower CAN —
#   sudo ip link set can_follow_l down && \
#   sudo ip link set can_follow_l type can bitrate 1000000 && \
#   sudo ip link set can_follow_l up
#
# Usage:  ./record_kitting.sh   [config.yaml]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/yam/yam_left_kitting_teleop.yaml}"
COCKPIT_DIR="/home/tommaso/Desktop/sorting-red-box"
COCKPIT_PORT=8799
# Serve over http (NOT file://): the cockpit fetches /cam + /state from the cam
# server cross-origin; a file:// origin ("null") gets blocked, an http origin does not.
COCKPIT="http://localhost:${COCKPIT_PORT}/Buehler-Kitting-Cockpit.html"
VISER="http://localhost:8080"
export DISPLAY="${DISPLAY:-:2}"          # the rig display (where viser auto-opens)

# 0) serve the cockpit over http so its auto-connect to the cam server works.
if ! curl -s -o /dev/null --max-time 1 "http://localhost:${COCKPIT_PORT}/"; then
  ( cd "$COCKPIT_DIR" && python3 -m http.server "$COCKPIT_PORT" --bind 127.0.0.1 ) \
      > /tmp/cockpit_http.log 2>&1 &
  HTTP_PID=$!
  echo "cockpit http server → ${COCKPIT} (log: /tmp/cockpit_http.log)"
fi

# 1) live label backend — /state + /cam JPEG from the running session's camera bus.
#    default=camera_top/rgb so every cockpit panel shows something even before the
#    exact per-camera id mapping is set.
uv run python -m robots_realtime.labeling.live_server --live --arm left \
    --host 0.0.0.0 --port 8791 \
    --bus-cams "default=camera_top/rgb,egocentric=camera_top/rgb,scan=camera_scan/rgb,wristL=camera_left/rgb,wristR=camera_left/rgb" \
    > /tmp/live_label.log 2>&1 &
LIVE_PID=$!
echo "live label backend → http://localhost:8791 (log: /tmp/live_label.log)"

# 2) open viser + cockpit in Firefox once viser is up (background waiter)
(
  for _ in $(seq 1 60); do
    curl -s -o /dev/null --max-time 1 "$VISER" && break || sleep 0.5
  done
  firefox --new-window "$VISER" "$COCKPIT" >/dev/null 2>&1 &
) &

cleanup() { kill "$LIVE_PID" 2>/dev/null || true; kill "${HTTP_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# 3) teleop session (TUI here: [r] start · [1] save · [d] discard · [space] pause · [q] quit)
RS2_USE_RSUSB_BACKEND=true exec uv run rr-session "$CONFIG"
