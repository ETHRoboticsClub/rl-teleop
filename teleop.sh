#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/yam/yam_bimanual_gello_teleop.yaml}"
BITRATE="${CAN_BITRATE:-1000000}"

cd "$SCRIPT_DIR"

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

if [[ ! -e /dev/leader-left ]]; then
  echo "ERROR: /dev/leader-left not found. Run ./custom/init-mapping.sh and replug leaders." >&2
  exit 1
fi

if [[ ! -e /dev/leader-right ]]; then
  echo "ERROR: /dev/leader-right not found. Run ./custom/init-mapping.sh and replug leaders." >&2
  exit 1
fi

echo "CAN interfaces:"
ip -brief link show dev can_follow_l
ip -brief link show dev can_follow_r
echo "Leaders:"
ls -l /dev/leader-left /dev/leader-right

exec uv run rr-session "$CONFIG"
