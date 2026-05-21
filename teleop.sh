#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="configs/yam/yam_bimanual_dynamixel_gello_teleop.yaml"
BITRATE="${CAN_BITRATE:-1000000}"
LEADER_LEFT="${LEADER_LEFT:-/dev/leader-left}"
LEADER_RIGHT="${LEADER_RIGHT:-/dev/leader-right}"
DRY_RUN_LEADERS=0
NO_TUI=0
CONFIG="$DEFAULT_CONFIG"

cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-leaders)
      DRY_RUN_LEADERS=1
      shift
      ;;
    --no-tui)
      NO_TUI=1
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
    -* )
      echo "ERROR: unknown option: $1" >&2
      exit 1
      ;;
    *)
      CONFIG="$1"
      shift
      ;;
  esac
done

check_leaders() {
  if [[ ! -e "$LEADER_LEFT" ]]; then
    echo "ERROR: $LEADER_LEFT not found. Run ./custom/init-mapping.sh and replug leaders." >&2
    exit 1
  fi

  if [[ ! -e "$LEADER_RIGHT" ]]; then
    echo "ERROR: $LEADER_RIGHT not found. Run ./custom/init-mapping.sh and replug leaders." >&2
    exit 1
  fi
}

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

run_dry_leaders() {
  mkdir -p .omo/evidence
  LEADER_LEFT="$LEADER_LEFT" LEADER_RIGHT="$LEADER_RIGHT" CONFIG="$CONFIG" uv run python -c '
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

config_path = Path(os.environ["CONFIG"])
config = yaml.safe_load(config_path.read_text())
leader_overrides = {
    "gello_left": os.environ["LEADER_LEFT"],
    "gello_right": os.environ["LEADER_RIGHT"],
}
result = {}
agents = []
try:
    for node in config["nodes"]:
        node_name = node.get("name")
        if node.get("type") != "AgentNode" or node_name not in leader_overrides:
            continue
        module_name, class_name = node["agent_class"].split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        kwargs = dict(node.get("agent_kwargs") or {})
        kwargs["port"] = leader_overrides[node_name]
        agent = cls(**kwargs)
        agents.append(agent)

        samples = []
        for _ in range(5):
            action = agent.act({})
            arm_key = node.get("arm_key") or kwargs.get("robot_name")
            pos = action[arm_key]["pos"]
            samples.append(np.asarray(pos, dtype=np.float32))
            time.sleep(0.01)

        arr = np.stack(samples)
        diffs = np.diff(arr, axis=0)
        max_abs_delta = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        last = arr[-1]
        topic = node_name + "/joint_pos"
        result[topic] = {
            "joint_pos": last.tolist(),
            "length": int(last.shape[0]),
            "all_finite": bool(np.all(np.isfinite(arr))),
            "max_abs_delta_rad": max_abs_delta,
            "stale": False,
        }

    missing = {"gello_left/joint_pos", "gello_right/joint_pos"} - set(result)
    if missing:
        raise RuntimeError(f"missing dry-run topics: {sorted(missing)}")
    Path(".omo/evidence/task-7-dry-run-topics.json").write_text(json.dumps(result, indent=2) + "\n")
finally:
    for agent in agents:
        agent.close()
'
}

check_leaders

if [[ "$DRY_RUN_LEADERS" -eq 1 ]]; then
  run_dry_leaders
  echo "Dry-run leader topics written to .omo/evidence/task-7-dry-run-topics.json"
  exit 0
fi

check_followers

echo "CAN interfaces:"
ip -brief link show dev can_follow_l
ip -brief link show dev can_follow_r
echo "Leaders:"
ls -l "$LEADER_LEFT" "$LEADER_RIGHT"

if [[ "$NO_TUI" -eq 1 ]]; then
  exec uv run rr-session "$CONFIG" --no-tui
fi

exec uv run rr-session "$CONFIG"
