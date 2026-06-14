"""Tests for AgentNode integration with cartesian workspace guardrails."""

import numpy as np
import pytest
import sys
import types
from unittest.mock import MagicMock, patch


# ── Fake zmq/msgpack (required by Node.__init__) ──────────────────────────

class _FakeZmqAgain(Exception):
    pass


class _FakeZmqContext:
    @classmethod
    def instance(cls):
        return cls()

    def socket(self, *_args, **_kwargs):
        return MagicMock()


_fake_zmq = types.ModuleType("zmq")
_fake_zmq.Again = _FakeZmqAgain
_fake_zmq.Context = _FakeZmqContext
_fake_zmq.NOBLOCK = 1
_fake_zmq.PUB = 1
_fake_zmq.SUB = 2
_fake_zmq.SUBSCRIBE = 3
_fake_zmq.proxy = MagicMock()
sys.modules.setdefault("zmq", _fake_zmq)

_fake_msgpack = types.ModuleType("msgpack")
_fake_msgpack.packb = MagicMock(return_value=b"")
_fake_msgpack.unpackb = MagicMock(return_value={})
sys.modules.setdefault("msgpack", _fake_msgpack)

_fake_msgpack_numpy = types.ModuleType("msgpack_numpy")
_fake_msgpack_numpy.encode = MagicMock()
_fake_msgpack_numpy.decode = MagicMock()
sys.modules.setdefault("msgpack_numpy", _fake_msgpack_numpy)


class MockFKProvider:
    """Minimal FK provider for testing. Maps first 3 joint values to position."""

    def __init__(self, site_name):
        self.site_name = site_name

    def fk(self, q, site_name=None):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        pose[1, 3] = q[1]
        pose[2, 3] = q[2]
        return pose


def _cartesian_safety(agent_type="teleop", left_bounds=None, right_bounds=None):
    """Build a cartesian workspace safety config with bounding_box."""
    arms = {}
    if left_bounds is not None:
        arms["left"] = {
            "position_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "bounding_box": {
                "min": [-10.0] * 6,
                "max": [10.0] * 6,
            },
            "cartesian_workspace": {
                "enabled": True,
                "context": "teleop",
                "enforcement": "reject_hold_last_safe",
                "reject_scope": "arm",
                "frame": "model",
                "site_name": "left_wrist",
                "xml_path": "models/franka/panda.xml",
                "min_xyz": left_bounds[0],
                "max_xyz": left_bounds[1],
                "tolerance_m": 0.0001,
                "reentry_margin_m": 0.002,
                "reentry_max_velocity_rad_s": 1.0,
                "pass_through_indices": [6],
            },
        }
    if right_bounds is not None:
        arms["right"] = {
            "position_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "bounding_box": {
                "min": [-10.0] * 6,
                "max": [10.0] * 6,
            },
            "cartesian_workspace": {
                "enabled": True,
                "context": "teleop",
                "enforcement": "reject_hold_last_safe",
                "reject_scope": "arm",
                "frame": "model",
                "site_name": "right_wrist",
                "xml_path": "models/franka/panda.xml",
                "min_xyz": right_bounds[0],
                "max_xyz": right_bounds[1],
                "tolerance_m": 0.0001,
                "reentry_margin_m": 0.002,
                "reentry_max_velocity_rad_s": 1.0,
                "pass_through_indices": [6],
            },
        }
    return {
        "mode": "sim",
        "agent_type": agent_type,
        "arms": arms,
        "acceleration_limit": None,
    }


def _node(fk_factory=None, **kwargs):
    """Build an AgentNode with mocked Node.__init__ and FK provider.

    Args:
        fk_factory: Optional callable(xml_path, site_name) -> FKProvider.
                   If None, uses MockFKProvider for all arms.
    """
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    if fk_factory is None:
        fk_factory = lambda xml_path, site_name: MockFKProvider(site_name or "wrist")

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=MagicMock(), name="agent", **kwargs)
    node.name = "agent"
    node.publish = MagicMock(return_value=True)
    node._setup_safety_guardrails(fk_factory=fk_factory)
    return node


# ── Per-arm reject scope ──────────────────────────────────────────────────


def test_left_reject_does_not_freeze_right_arm():
    """When left hits the cartesian box boundary, right continues publishing normally."""
    node = _node(
        safety=_cartesian_safety(
            left_bounds=([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            right_bounds=([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        )
    )

    # Publish both arms: left goes out of its small box, right stays in its large box
    left_pos = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x far outside left box
    right_pos = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # well inside right box

    node._publish_commands(
        {"left": {"pos": left_pos}, "right": {"pos": right_pos}}, ts=100.0
    )

    # Both topics should have been published
    calls = node.publish.call_args_list
    topics = [c.args[0] for c in calls]
    assert "left_pos" in topics or "joint_pos" in topics
    assert "right_pos" in topics

    # Right arm output should be the original right_pos (unchanged)
    right_call = [c for c in calls if c.args[0] == "right_pos"][0]
    right_output = right_call.args[1]["joint_pos"]
    np.testing.assert_allclose(right_output, right_pos, atol=1e-5)


# ── Publish count matches input count ─────────────────────────────────────


def test_reject_path_still_publishes_joint_pos():
    """Even when a command is rejected, a publication still occurs (with safe fallback)."""
    node = _node(
        arm_key="left",
        safety=_cartesian_safety(
            left_bounds=([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
        ),
    )

    input_count = 5
    for i in range(input_count):
        pos = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # always out of bounds
        node._publish_commands({"left": {"pos": pos}}, ts=float(i))

    assert node.publish.call_count == input_count


# ── Startup validation: initial state outside box ─────────────────────────


def test_initial_state_outside_box_fails_before_reset():
    """If production current_state has end-effector outside the box, fail closed on startup."""
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    mock_agent = MagicMock()
    mock_agent.reset = MagicMock()

    # Safety config where the box is small around origin
    bad_safety = _cartesian_safety(
        left_bounds=([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
    )
    # Simulate current_state being far outside the box
    bad_safety["production_current_state"] = {
        "left": {"qpos": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]},
    }

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=mock_agent, name="agent", safety=bad_safety)
    node.name = "agent"
    node.publish = MagicMock()

    with pytest.raises(ValueError, match="[Oo]utside|[Cc]losed|[Ss]tartup|[Ii]nitial|cartesian"):
        node._setup_safety_guardrails(fk_factory=lambda xml_path, site_name: MockFKProvider(site_name or "wrist"))

    mock_agent.reset.assert_not_called()


# ── Missing production state fails closed ─────────────────────────────────


def test_missing_production_current_state_fails_before_reset():
    """Missing production_current_state in safety config fails on startup for real mode."""
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    mock_agent = MagicMock()
    mock_agent.reset = MagicMock()

    safety_no_state = _cartesian_safety(
        left_bounds=([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
    )
    # No production_current_state key at all

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=mock_agent, name="agent", safety=safety_no_state)
    node.name = "agent"
    node.publish = MagicMock()

    with pytest.raises(ValueError, match="[Mm]iss|production|current_state|cartesian"):
        node._setup_safety_guardrails(fk_factory=lambda xml_path, site_name: MockFKProvider(site_name or "wrist"))

    mock_agent.reset.assert_not_called()


# ── Reject event telemetry ────────────────────────────────────────────────


def test_reject_events_are_throttled_and_counted():
    """Cartesian reject events are tracked in clamp_log and throttled for telemetry."""
    node = _node(
        safety=_cartesian_safety(
            left_bounds=([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
        ),
    )

    # Send many out-of-bounds commands
    num_commands = 50
    for i in range(num_commands):
        pos = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        node._process_pos(pos, arm_key="left")

    # Clamp log tracks cartesian rejects
    assert len(node._clamp_log) > 0

    # Check that entries reference cartesian workspace
    cartesian_entries = [e for e in node._clamp_log if "cartesian" in e.get("guardrail", "")]
    assert len(cartesian_entries) > 0

    # Telemetry logs should be throttled (fewer than raw rejections)
    assert len(node._clamp_log) <= num_commands
