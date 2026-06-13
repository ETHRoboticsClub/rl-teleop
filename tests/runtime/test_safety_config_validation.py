import logging
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

def test_real_hardware_requires_numeric_bbox_before_actuation():
    """Real hardware config with null bbox should fail validation."""
    cfg = {
        "mode": "real",
        "actuation": {"type": "hardware"},
        "bounding_box": None,  # Not filled in yet
    }
    with pytest.raises(ValueError, match="bounding_box"):
        validate_safety_config(cfg)

def test_real_hardware_inference_requires_acceleration_limits():
    """Real hardware inference config requires acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "inference",
        "actuation": {"type": "hardware"},
        "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        "acceleration_limit": None,  # Missing
    }
    with pytest.raises(ValueError, match="[Aa]cceleration"):
        validate_safety_config(cfg)

def test_real_hardware_teleop_does_not_require_acceleration_limits():
    """Teleop config should not require acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "teleop",
        "actuation": {"type": "hardware"},
        "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        # No acceleration_limit key at all — OK for teleop
    }
    # Should pass without error
    try:
        result = validate_safety_config(cfg)
        assert isinstance(result, dict) or result is True
    except NotImplementedError:
        raise NotImplementedError("SafetyConfig not implemented yet")

def test_sim_config_accepts_explicit_numeric_safety_limits():
    """Sim config with numeric values should pass validation."""
    cfg = {
        "mode": "sim",
        "agent_type": "inference",
        "bounding_box": {"min": [-np.pi, -np.pi, -np.pi], "max": [np.pi, np.pi, np.pi]},
        "acceleration_limit": 0.5,
    }
    try:
        result = validate_safety_config(cfg)
        assert result is True or isinstance(result, (dict, SafetyConfig))
    except NotImplementedError:
        raise NotImplementedError("SafetyConfig not implemented yet")

def test_invalid_nan_inf_and_min_greater_than_max_configs_fail():
    """NaN, inf, and min > max configs should fail validation."""
    nan_cfg = {"bounding_box": {"min": [float('nan')], "max": [1.0]}}
    inf_cfg = {"bounding_box": {"min": [0.0], "max": [float('inf')]}}
    inv_cfg = {"bounding_box": {"min": [2.0], "max": [1.0]}}
    
    for bad_cfg in [nan_cfg, inf_cfg, inv_cfg]:
        try:
            validate_safety_config(bad_cfg)
            assert False, f"Expected ValueError for {bad_cfg}"
        except ValueError:
            pass
        except NotImplementedError:
            raise NotImplementedError("SafetyConfig not implemented yet")

def test_invalid_safety_config_fails_before_agent_reset():
    """Invalid safety config should raise before agent.reset() is called."""
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node
    from unittest.mock import MagicMock, patch
    
    bad_cfg = {
        "mode": "real",
        "agent_type": "inference",
        "arms": {"left": {"bounding_box": None}},
        "acceleration_limit": None,
    }
    
    mock_agent = MagicMock()
    mock_agent.reset = MagicMock()
    
    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=mock_agent, name="agent", safety=bad_cfg)
    node.name = "agent"
    node.publish = MagicMock()
    
    # _setup_safety_guardrails() runs before agent.reset() in setup()
    with pytest.raises(ValueError, match="[Bb]ounding"):
        node.setup()
    
    # agent.reset() must NOT have been called
    mock_agent.reset.assert_not_called()

def test_clamp_event_is_logged_or_counted_with_arm_and_guardrail_name():
    sfty = {
        "mode": "real",
        "agent_type": "teleop",
        "arms": {"left": {"bounding_box": {"min": [-1.0] * 6, "max": [1.0] * 6}}},
        "acceleration_limit": None,
    }
    node = _node(safety=sfty)

    out_of_bounds = np.array([2.0] * 6)
    result = node._process_pos(out_of_bounds, arm_key="left")

    assert len(node._clamp_log) == 1
    entry = node._clamp_log[0]
    assert entry["arm_key"] == "left"
    assert entry["guardrail"] == "bbox"
    np.testing.assert_allclose(entry["original"], [2.0] * 6)
    np.testing.assert_allclose(entry["clamped"], [1.0] * 6)
    assert isinstance(entry["timestamp"], float)
    np.testing.assert_allclose(result, [1.0] * 6)

def test_repeated_clamps_warn_without_pausing_or_stopping():
    """Repeated clamps should log warnings but not pause/stop execution."""
    sfty = {
        "mode": "real",
        "agent_type": "teleop",
        "arms": {"left": {"bounding_box": {"min": [-0.5] * 6, "max": [0.5] * 6}}},
        "acceleration_limit": None,
    }
    node = _node(safety=sfty)

    with patch.object(logging.getLogger("robots_realtime.runtime.agent_node"), "warning") as mock_warn:
        for i in range(10):
            result = node._process_pos(np.array([99.0] * 6), arm_key="left")
            np.testing.assert_allclose(result, [0.5] * 6)

        assert len(node._clamp_log) == 10
        assert mock_warn.call_count == 10


def test_empty_bbox_min_fails_validation():
    with pytest.raises(ValueError, match="[Mm]in"):
        validate_safety_config({"bounding_box": {"max": [1.0, 1.0]}})


def test_empty_bbox_arrays_fail_validation():
    with pytest.raises(ValueError, match="[Ee]mpty|[Ll]ength"):
        validate_safety_config({"bounding_box": {"min": [], "max": []}})


def test_mismatched_bbox_lengths_fail_validation():
    with pytest.raises(ValueError, match="[Ll]ength|[Mm]ismatch"):
        validate_safety_config({"bounding_box": {"min": [0.0], "max": [1.0, 1.0]}})


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
