#!/usr/bin/env bash
# Bring up the LEFT-ARM RECORDING pipeline, verified, in one command.
#
#   ./start_left_recording.sh
#
# Every check below exists because it silently failed on 2026-09-01 and cost
# real recording time. They are GATES, not advice: this script refuses to hand
# you a session that looks alive and records nothing.
#
#   1. control port      the cockpit hardcodes :8792 (cockpit-control.js
#                        DEFAULT_BASE). Overriding CONTROL_PORT makes the REC
#                        button post into a void -- it only prints "kein
#                        rr-session auf ..." in a status line. Five grasps were
#                        lost to this. We never override it and we verify it.
#   2. camera bridge     :8791 holds subscriptions that die with the session and
#                        never reconnect (/cam/top starts 503ing). So it is
#                        ALWAYS restarted after the session, never before.
#   3. gripper channel   a flat gripper channel cannot train a grasp policy --
#                        the exporter finds grasp windows BY the gripper closing.
#                        The left handle's gripper read 2290 with zero travel on
#                        2026-09-01 and 11 minutes of otherwise-perfect video was
#                        useless. This script makes you prove the gripper moves
#                        before it lets you record.
#   4. SSH forwarding    the operator's browser is NOT on this box. A local curl
#                        proving a port healthy proves nothing about what he can
#                        reach. We print the exact ports he must forward.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-configs/yam/yam_left_handoff_teleop_noscan.yaml}"
PY=./.venv/bin/python3
COCKPIT_PAGE="Buehler-Kitting-Recording.html"
SKIP_GRIPPER="${SKIP_GRIPPER:-0}"

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { red "✗ $*"; exit 1; }
port_up() { ss -tlnH 2>/dev/null | grep -q ":$1 "; }
pids_on() { ss -tlnpH 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | sort -u; }

echo "── left-arm recording bring-up ─────────────────────────────────────"

# ── 0. the leader handle's gripper MUST move ────────────────────────────────
# Done FIRST, with the serial port free, because it is the one failure that
# produces hours of perfect-looking but untrainable data.
if [ "$SKIP_GRIPPER" != "1" ]; then
  GPORT=$(grep -A8 'name: gello_left' "$CONFIG" | grep -oP 'port:\s*\K\S+' | head -1)
  GID=$(grep -A10 'name: gello_left' "$CONFIG" | grep -oP 'motor_ids:\s*\[\K[^\]]+' | head -1 | tr -d ' ' | awk -F, '{print $NF}')
  [ -n "$GPORT" ] && [ -n "$GID" ] || die "could not read gello_left port/motor_ids from $CONFIG"
  if port_up 8792; then
    ylw "  a session already holds the leader serial; skipping the gripper gate"
    ylw "  (stop it and re-run, or SKIP_GRIPPER=1 to bypass deliberately)"
  else
    echo "  gripper gate: squeeze the LEFT handle trigger fully closed and release."
    echo "               reading $GPORT motor $GID for 45 s ..."
    $PY - "$GPORT" "$GID" <<'PYEOF' || die "GRIPPER IS FLAT -- recording refused.
     The trigger does not move that servo. A grasp dataset with a constant
     gripper channel cannot train a grasp policy: the exporter locates grasp
     windows BY the gripper closing, so it would find none.
     Fix the handle, or swap in the handle whose gripper works and revert the
     leader swap (rl-teleop commit fee09a6). SKIP_GRIPPER=1 bypasses this gate
     if you genuinely mean to record without gripper data."
import sys, time
from dynamixel_sdk import PortHandler, PacketHandler
port, gid = sys.argv[1], int(sys.argv[2])
ph = PortHandler(port); pk = PacketHandler(2.0)
if not ph.openPort():
    print("   cannot open %s" % port); sys.exit(1)
ph.setBaudRate(1000000)
lo = hi = None; n = 0
t0 = time.time()
while time.time() - t0 < 45:
    pos, res, err = pk.read4ByteTxRx(ph, gid, 132)
    if res == 0 and err == 0:
        n += 1
        lo = pos if lo is None else min(lo, pos)
        hi = pos if hi is None else max(hi, pos)
        if hi - lo > 25:
            print("   gripper travels: %d..%d ticks (span %d)" % (lo, hi, hi - lo))
            ph.closePort(); sys.exit(0)
    time.sleep(0.03)
ph.closePort()
print("   no travel: %s reads, stuck at %s" % (n, lo))
sys.exit(1)
PYEOF
    grn "  ✓ gripper moves"
  fi
fi

# ── 1. bus ──────────────────────────────────────────────────────────────────
if port_up 5555 && port_up 5556; then
  grn "  ✓ bus already up (5555/5556)"
else
  echo "  starting bus ..."
  tmux kill-session -t bus 2>/dev/null
  tmux new-session -d -s bus -c "$PWD" './.venv/bin/rr-bus 2>&1 | tee /tmp/rr_bus.log'
  sleep 4
  port_up 5555 || die "bus did not come up -- see /tmp/rr_bus.log"
  grn "  ✓ bus up"
fi

# ── 2. the recording session ────────────────────────────────────────────────
# record_kitting.sh's own CONTROL_PORT default (8792) is what the cockpit talks
# to. We deliberately do NOT export CONTROL_PORT here.
echo "  starting recording session ($CONFIG) ..."
tmux kill-session -t rec 2>/dev/null; sleep 6
tmux new-session -d -s rec -c "$PWD" \
  "ATTACH_BUS=1 COCKPIT_FILE=$COCKPIT_PAGE ./record_kitting.sh $CONFIG 2>&1 | tee /tmp/record_left.log"

for i in $(seq 1 40); do port_up 8792 && break; sleep 2; done
port_up 8792 || die "control surface :8792 never came up -- see /tmp/record_left.log
     (if the log shows a camera preflight failure, a camera is missing or a
     device path drifted; check by FRAME CONTENT, not by remembered port)"
grn "  ✓ session up, control :8792 answering (the port the cockpit drives)"

# ── 3. camera bridge, ALWAYS restarted after the session ────────────────────
for p in $(pids_on 8791); do kill "$p" 2>/dev/null; done
sleep 2
setsid $PY -u -m robots_realtime.labeling.live_server --live --arm left \
  --host 0.0.0.0 --port 8791 --save-root recordings --episode-mode full \
  --control-url http://localhost:8792 \
  --bus-cams "default=camera_top/rgb,egocentric=camera_top/rgb,top=camera_top/rgb,wristL=camera_left/rgb" \
  > /tmp/live_label_left.log 2>&1 &
for i in $(seq 1 15); do port_up 8791 && break; sleep 1; done
port_up 8791 || die "camera bridge :8791 did not start -- see /tmp/live_label_left.log"

# ── 4. counter + cockpit ────────────────────────────────────────────────────
port_up 8806 || ( cd ../cockpit && setsid python3 recording_counter.py \
  --target-left 200 --target-right 300 </dev/null > /tmp/recording_counter.log 2>&1 & )
port_up 8799 || ( cd ../cockpit && setsid python3 -m http.server 8799 </dev/null > /tmp/cockpit_http.log 2>&1 & )
sleep 4

# ── 5. verify what the operator will actually see ───────────────────────────
echo "  verifying the panels ..."
fail=0
for id in top egocentric wristL; do
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:8791/cam/$id")
  [ "$code" = "200" ] && grn "    ✓ /cam/$id" || { red "    ✗ /cam/$id -> $code"; fail=1; }
done
timeout 25 $PY tools/check_streams.py --secs 5 2>&1 | grep -E "camera_(left|top)" | sed 's/^/    /'
[ "$fail" = "0" ] || die "a camera panel is dead -- do not record until it is fixed"

echo
grn "── ready ───────────────────────────────────────────────────────────"
cat <<EOF
  cockpit   http://localhost:8799/$COCKPIT_PAGE      (hard-reload: Ctrl+Shift+R)
  control   http://localhost:8792                    (the REC button drives this)
  counter   http://localhost:8806/counts

  IF YOU ARE ON SSH, forward these ports or the page will look dead even though
  every server here is healthy -- 8792, 8797, 8805 bind 127.0.0.1 only:

      8799  8791  8792  8806

  BEFORE RECORDING MORE THAN ONE TAKE:
    1. set TAKT to Auto      (each scored placement saves the take, one take = one grasp)
    2. press Aufnahme and watch the tile start counting seconds.
       If it stays "bereit", NOTHING IS BEING SAVED -- stop and fix it.

  Stop everything:  tmux kill-session -t rec
EOF
