"""AgentNode safety integration: guardrails clamp commands on the publish path."""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.runtime.agent_node import AgentNode
from robots_realtime.runtime.safety.config import SafetyConfigError

pytestmark = [pytest.mark.safety]


def _node(safety, **kw):
    node = AgentNode(agent=object(), name="agent", safety=safety, **kw)
    node._build_guardrails()  # normally called in setup(); no bus needed
    return node


def _capture_publish(node):
    published = {}
    node.publish = lambda topic, data, ts=None, **k: published.__setitem__(topic, data)
    return published


BBOX = {"min": [-1] * 6, "max": [1] * 6}


def test_no_safety_is_passthrough():
    node = _node(None)
    out = node._finalize_pos(np.array([5.0] * 7), "default")
    np.testing.assert_array_equal(out, np.array([5.0] * 7, dtype=np.float32))


def test_bounding_box_clamps_published_command():
    node = _node({"command_source": "sim_teleop", "arms": {"default": {"bounding_box": BBOX}}})
    published = _capture_publish(node)
    node._publish_commands({"pos": np.array([9.0, -9.0, 0, 0, 0, 0, 0.5])}, ts=0.0)
    cmd = published["joint_pos"]["joint_pos"]
    np.testing.assert_allclose(cmd[:6], [1, -1, 0, 0, 0, 0])
    assert cmd[6] == pytest.approx(0.5)  # gripper passthrough


def test_teleop_speed_cap_limits_jump():
    node = _node(
        {
            "command_source": "teleop",
            "arms": {"default": {"bounding_box": {"min": [-10] * 6, "max": [10] * 6},
                                 "speed_limit": {"max_step_rad": 0.1}}},
        }
    )
    node._finalize_pos(np.zeros(7), "default")  # seed speed limiter at 0
    out = node._finalize_pos(np.array([5.0, 0, 0, 0, 0, 0, 0]), "default")
    assert out[0] == pytest.approx(0.1)  # WS4: teleop is now speed-capped


def test_multi_arm_independent_configs():
    node = _node(
        {
            "command_source": "sim_teleop",
            "arms": {
                "left": {"bounding_box": {"min": [-1] * 6, "max": [1] * 6}},
                "right": {"bounding_box": {"min": [-3] * 6, "max": [3] * 6}},
            },
        }
    )
    published = _capture_publish(node)
    node._publish_commands(
        {"left": {"pos": np.array([5.0] * 6)}, "right": {"pos": np.array([5.0] * 6)}},
        ts=0.0,
    )
    np.testing.assert_allclose(published["left_pos"]["joint_pos"], [1] * 6)
    np.testing.assert_allclose(published["right_pos"]["joint_pos"], [3] * 6)


def test_real_hardware_invalid_config_fails_closed_in_setup():
    # Missing bbox on real hardware must raise when guardrails are built (before any
    # command can be published).
    node = AgentNode(
        agent=object(),
        name="agent",
        safety={
            "command_source": "teleop",
            "is_real_hardware": True,
            "arms": {"left": {"speed_limit": {"max_step_rad": 0.05}}},
        },
    )
    with pytest.raises(SafetyConfigError):
        node._build_guardrails()


def test_nonfinite_command_holds_last_safe():
    # A NaN command must never pass through the chain (np.clip preserves NaN); once a safe
    # command exists, a later non-finite command is held to it.
    node = _node({"command_source": "sim_teleop", "arms": {"default": {"bounding_box": BBOX}}})
    safe = node._finalize_pos(np.array([0.5, 0, 0, 0, 0, 0, 0.5]), "default")
    held = node._finalize_pos(np.array([np.nan, 0, 0, 0, 0, 0, 0.5]), "default")
    assert np.all(np.isfinite(held))
    np.testing.assert_array_equal(held, safe)


def test_nonfinite_first_command_fails_closed():
    # A non-finite very first command has no safe pose to hold — refuse to publish.
    node = _node({"command_source": "sim_teleop", "arms": {"default": {"bounding_box": BBOX}}})
    with pytest.raises(RuntimeError, match="non-finite"):
        node._finalize_pos(np.array([np.inf, 0, 0, 0, 0, 0, 0.5]), "default")


def test_nonfinite_does_not_poison_speed_limiter():
    # Regression: a NaN command must not latch the speed limiter's reference to NaN and
    # disable the cap forever. After a held NaN, the next command is still clamped against
    # the last *safe* reference (0.0), not passed through.
    node = _node(
        {
            "command_source": "teleop",
            "arms": {"default": {"bounding_box": {"min": [-10] * 6, "max": [10] * 6},
                                 "speed_limit": {"max_step_rad": 0.1}}},
        }
    )
    node._finalize_pos(np.zeros(7), "default")                        # seed at 0
    node._finalize_pos(np.array([np.nan, 0, 0, 0, 0, 0, 0]), "default")  # held, must not latch
    out = node._finalize_pos(np.array([5.0, 0, 0, 0, 0, 0, 0]), "default")
    assert out[0] == pytest.approx(0.1)


def test_arm_key_path_uses_matching_config():
    node = _node(
        {"command_source": "sim_teleop", "arms": {"left": {"bounding_box": BBOX}}},
        arm_key="left",
    )
    published = _capture_publish(node)
    node._publish_commands({"left": {"pos": np.array([9.0] * 6)}}, ts=0.0)
    np.testing.assert_allclose(published["joint_pos"]["joint_pos"], [1] * 6)
