"""Safety configuration validation — Cartesian workspace only (no bounding_box)."""

import numpy as np
import pytest
import sys
import types
from unittest.mock import MagicMock, patch

from robots_realtime.runtime.safety.config import SafetyConfig, validate_safety_config


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


def _node(**kwargs):
    """Build an AgentNode with mocked Node.__init__."""
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=MagicMock(), name="agent", **kwargs)
    node.name = "agent"
    node.publish = MagicMock(return_value=True)
    node.setup()
    return node


def test_real_hardware_inference_requires_acceleration_limits():
    """Real hardware inference config requires acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "inference",
        "acceleration_limit": None,
    }
    with pytest.raises(ValueError, match="[Aa]cceleration"):
        validate_safety_config(cfg)


def test_real_hardware_teleop_does_not_require_acceleration_limits():
    """Teleop config should not require acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "teleop",
    }
    assert validate_safety_config(cfg) is True


def test_sim_config_accepts_explicit_numeric_acceleration():
    """Sim config with numeric acceleration_limit should pass validation."""
    cfg = {
        "mode": "sim",
        "agent_type": "inference",
        "acceleration_limit": 0.5,
    }
    assert validate_safety_config(cfg) is True


def test_invalid_nan_inf_acceleration_configs_fail():
    """NaN and inf acceleration_limit should fail validation."""
    nan_cfg = {"mode": "sim", "agent_type": "inference", "acceleration_limit": float('nan')}
    inf_cfg = {"mode": "sim", "agent_type": "inference", "acceleration_limit": float('inf')}
    neg_cfg = {"mode": "sim", "agent_type": "inference", "acceleration_limit": -1.0}

    for bad_cfg in [nan_cfg, inf_cfg, neg_cfg]:
        with pytest.raises(ValueError):
            validate_safety_config(bad_cfg)


def test_invalid_safety_config_fails_before_agent_reset():
    """Invalid safety config should raise before agent.reset() is called."""
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    bad_cfg = {
        "mode": "real",
        "agent_type": "inference",
        "arms": {},
        "acceleration_limit": None,
    }

    mock_agent = MagicMock()
    mock_agent.reset = MagicMock()

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=mock_agent, name="agent", safety=bad_cfg)
    node.name = "agent"
    node.publish = MagicMock()

    # _setup_safety_guardrails() runs before agent.reset() in setup()
    with pytest.raises(ValueError):
        node.setup()

    mock_agent.reset.assert_not_called()


def test_acceleration_clamp_event_is_logged_with_arm_and_guardrail_name():
    """Acceleration clamp events should be logged with arm key and guardrail name."""
    sfty = {
        "mode": "sim",
        "agent_type": "inference",
        "arms": {"left": {}},
        "acceleration_limit": 0.5,
    }
    node = _node(safety=sfty)

    # First call sets baseline, second call triggers acceleration clamp
    node._process_pos(np.zeros(6), arm_key="left")
    result = node._process_pos(np.array([10.0] * 6), arm_key="left")

    assert len(node._clamp_log) >= 1
    entry = node._clamp_log[0]
    assert entry["arm_key"] == "left"
    assert entry["guardrail"] == "accel"
    assert isinstance(entry["timestamp"], float)


def test_repeated_acceleration_clamps_warn_without_pausing():
    """Repeated acceleration clamps should log warnings but not pause/stop execution."""
    import logging

    sfty = {
        "mode": "sim",
        "agent_type": "inference",
        "arms": {"left": {}},
        "acceleration_limit": 0.1,
    }
    node = _node(safety=sfty)

    # Set baseline
    node._process_pos(np.zeros(6), arm_key="left")

    with patch.object(logging.getLogger("robots_realtime.runtime.agent_node"), "warning") as mock_warn:
        for _ in range(5):
            node._process_pos(np.array([99.0] * 6), arm_key="left")

        assert len(node._clamp_log) >= 5
        assert mock_warn.call_count >= 5


def test_nan_acceleration_limit_rejected():
    with pytest.raises(ValueError, match="[Nn]aN|[Aa]cceleration"):
        validate_safety_config(
            {"mode": "sim", "agent_type": "inference", "acceleration_limit": float("nan")}
        )


def test_negative_acceleration_limit_rejected():
    with pytest.raises(ValueError, match="[Pp]ositive|[Nn]egative|[Aa]cceleration"):
        validate_safety_config(
            {"mode": "sim", "agent_type": "inference", "acceleration_limit": -1.0}
        )


def test_infinite_acceleration_limit_rejected():
    with pytest.raises(ValueError, match="[Ii]nf|[Aa]cceleration"):
        validate_safety_config(
            {"mode": "sim", "agent_type": "inference", "acceleration_limit": float("inf")}
        )


# ── Migration RED tests: Cartesian-only safety (no bounding_box) ──────────


def test_cartesian_workspace_does_not_require_joint_bounding_box():
    """After migration, Cartesian workspace config should validate without bounding_box."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cw_cfg = {
        "agent_type": "teleop",
        "site_name": "left_tcp_site",
        "xml_path": "robots_realtime/sim/models/yam_bimanual_scene.xml",
        "frame": "model",
        "min_xyz": [-0.5, -0.5, -0.5],
        "max_xyz": [0.5, 0.5, 0.5],
        "enforcement": "reject_hold_last_safe",
    }
    assert validate_cartesian_workspace_config(cw_cfg) is True


def test_command_bounding_box_guardrail_removed():
    """After migration, CommandBoundingBoxGuardrail should not be importable from safety package."""
    try:
        from robots_realtime.runtime.safety.guardrails import CommandBoundingBoxGuardrail
        assert False, "CommandBoundingBoxGuardrail should be removed after migration"
    except (ImportError, AttributeError):
        pass
