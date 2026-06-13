#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="configs/yam/yam_bimanual_inference.yaml"
BITRATE="${CAN_BITRATE:-1000000}"
NO_TUI=0
RSUSB=1
CONFIG="$DEFAULT_CONFIG"
DURATION=""
EXTRA_ARGS=()

cd "$SCRIPT_DIR"

echo "Cloning any missing git submodules..."
# Only initialize submodules that aren't checked out yet (leading '-' in
# `git submodule status`). Mirrors teleop.sh: clones missing setups but
# does NOT force already-checked-out submodules back to the pinned commit.
git submodule status --recursive | awk '/^-/ {print $2}' | while read -r _sm_path; do
  echo "  initializing missing submodule: $_sm_path"
  git submodule update --init --recursive -- "$_sm_path"
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-tui)
      NO_TUI=1
      shift
      ;;
    --rsusb)
      RSUSB=1
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --config requires a path" >&2
        exit 1
      fi
      CONFIG="$2"
      shift 2
      ;;
    --duration)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --duration requires a value in seconds" >&2
        exit 1
      fi
      DURATION="$2"
      shift 2
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      CONFIG="$1"
      shift
      ;;
  esac
done

check_followers() {
  for iface in can_follow_l can_follow_r; do
    if ! ip link show dev "$iface" >/dev/null 2>&1; then
      echo "ERROR: $iface not found. Run ./custom/init-mapping.sh and replug CANables." >&2
      exit 1
    fi

    state="$(ip -brief link show dev "$iface" | awk '{print $2}')"
    if [[ "$state" != "UP" ]]; then
      echo "Bringing up $iface at ${BITRATE} bit/s..."
      sudo ip link set dev "$iface" down 2>/dev/null || true
      sudo ip link set dev "$iface" type can bitrate "$BITRATE"
      sudo ip link set dev "$iface" up
    fi
  done
}

check_followers

echo "CAN interfaces:"
ip -brief link show dev can_follow_l
ip -brief link show dev can_follow_r

CMD=(uv run rr-session "$CONFIG")
if [[ "$NO_TUI" -eq 1 ]]; then
  CMD+=(--no-tui)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$RSUSB" -eq 1 ]]; then
  export RS2_USE_RSUSB_BACKEND=true
fi

# Repo root on PYTHONPATH so `import adapters` resolves. The adapters/
# folder lives at the top level (outside the installed robots_realtime
# package) so user-edited model I/O stubs don't require a reinstall.
# Also add mimic-video/model so `cosmos_predict2` and `imaginaire` resolve
# at runtime — they are not installable Python packages.
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/mimic-video/model${PYTHONPATH:+:${PYTHONPATH}}"

# --duration wraps rr-session with `timeout`. SIGINT lets the session run
# its graceful shutdown (RobotNode ramps to shutdown_joint_pos); --kill-after
# is a hard backstop if any node ignores SIGINT.
if [[ -n "$DURATION" ]]; then
  exec timeout --preserve-status --signal=INT --kill-after=10 "$DURATION" "${CMD[@]}"
fi
exec "${CMD[@]}"
