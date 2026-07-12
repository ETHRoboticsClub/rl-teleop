"""Unit tests for the pure safety guardrails + config (bounding box + speed limit)."""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.runtime.safety.config import (
    BoundingBoxConfig,
    CommandSource,
    SafetyConfigError,
    SpeedLimitConfig,
    build_safety_config,
)
from robots_realtime.runtime.safety.guardrails import (
    BoundingBoxGuardrail,
    SpeedLimitGuardrail,
)

pytestmark = [pytest.mark.safety]

ARM6 = tuple(range(6))


# ── Bounding box ────────────────────────────────────────────────────────────────


def _bbox(lo, hi, idx=ARM6):
    return BoundingBoxConfig(min=np.array(lo, float), max=np.array(hi, float), position_indices=idx)


def test_bbox_clamps_below_and_above():
    g = BoundingBoxGuardrail(_bbox([-1] * 6, [1] * 6))
    cmd = np.array([-2.0, 2.0, 0.0, 0.5, -0.5, 3.0, 0.7])  # index 6 = gripper
    out, ev = g.apply(cmd)
    np.testing.assert_allclose(out[:6], [-1, 1, 0, 0.5, -0.5, 1])
    assert out[6] == 0.7  # gripper untouched
    assert ev is not None and ev.guardrail == "bounding_box"


def test_bbox_preserves_in_bounds():
    g = BoundingBoxGuardrail(_bbox([-1] * 6, [1] * 6))
    cmd = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.9])
    out, ev = g.apply(cmd)
    np.testing.assert_array_equal(out, cmd)
    assert ev is None  # no event when nothing clamped


def test_bbox_does_not_mutate_input():
    g = BoundingBoxGuardrail(_bbox([-1] * 6, [1] * 6))
    cmd = np.array([5.0] * 7)
    original = cmd.copy()
    g.apply(cmd)
    np.testing.assert_array_equal(cmd, original)


# ── Speed limit ─────────────────────────────────────────────────────────────────


def _speed(step=0.1, idx=ARM6):
    return SpeedLimitConfig(max_step_rad=step, position_indices=idx)


def test_speed_first_command_seeds_no_clamp():
    g = SpeedLimitGuardrail(_speed(0.1))
    out, ev = g.apply(np.array([1.0, 2.0, 3.0, 0, 0, 0, 0.5]))
    assert ev is None  # first step establishes the reference
    np.testing.assert_array_equal(out[:6], [1, 2, 3, 0, 0, 0])


def test_speed_clamps_large_jump():
    g = SpeedLimitGuardrail(_speed(0.1))
    g.apply(np.zeros(7))  # seed at 0
    out, ev = g.apply(np.array([5.0, -5.0, 0.05, 0, 0, 0, 0.9]))
    # Arm joints move at most 0.1 per step.
    np.testing.assert_allclose(out[:6], [0.1, -0.1, 0.05, 0, 0, 0])
    assert out[6] == 0.9  # gripper passes through unclamped
    assert ev is not None


def test_speed_ramps_toward_target_over_steps():
    g = SpeedLimitGuardrail(_speed(0.1))
    g.apply(np.zeros(7))
    target = np.array([1.0, 0, 0, 0, 0, 0, 0])
    last = None
    for _ in range(20):
        last, _ = g.apply(target)
    # After enough steps it reaches the target (monotonic ramp, no overshoot).
    assert last[0] == pytest.approx(1.0, abs=1e-9)


def test_speed_seeds_from_current_state_when_given():
    g = SpeedLimitGuardrail(_speed(0.1))
    current = np.array([10.0, 0, 0, 0, 0, 0])
    out, ev = g.apply(np.array([10.5, 0, 0, 0, 0, 0, 0.0]), current=current)
    # Seeded from current=10.0, so a 0.5 jump is clamped to 0.1 -> 10.1.
    assert out[0] == pytest.approx(10.1)
    assert ev is not None


def test_speed_does_not_mutate_input():
    g = SpeedLimitGuardrail(_speed(0.1))
    g.apply(np.zeros(7))
    cmd = np.array([9.0] * 7)
    original = cmd.copy()
    g.apply(cmd)
    np.testing.assert_array_equal(cmd, original)


# ── Config parsing + fail-closed validation ─────────────────────────────────────


def test_build_none_when_absent_or_disabled():
    assert build_safety_config(None) is None
    assert build_safety_config({"enabled": False, "arms": {}}) is None


def test_build_sim_teleop_bbox_only():
    cfg = build_safety_config(
        {
            "command_source": "sim_teleop",
            "arms": {"default": {"bounding_box": {"min": [-1] * 6, "max": [1] * 6}}},
        }
    )
    assert cfg.command_source is CommandSource.SIM_TELEOP
    assert cfg.arm("default").bounding_box is not None
    assert cfg.arm("default").speed_limit is None


def test_speed_from_velocity_and_hz():
    cfg = build_safety_config(
        {
            "command_source": "teleop",
            "arms": {
                "default": {
                    "bounding_box": {"min": [-2] * 6, "max": [2] * 6},
                    "speed_limit": {"max_velocity_rad_s": 4.0, "control_hz": 200.0},
                }
            },
        }
    )
    # 4 rad/s at 200 Hz -> 0.02 rad/step.
    assert cfg.arm("default").speed_limit.max_step_rad == pytest.approx(0.02)


@pytest.mark.parametrize(
    "bbox",
    [
        {"min": [0, 0], "max": [1]},              # shape mismatch
        {"min": [1, 2], "max": [0, 3]},           # min > max
        {"min": [float("nan"), 0], "max": [1, 1]},  # non-finite
        {"max": [1, 1]},                          # missing min
    ],
)
def test_invalid_bbox_raises(bbox):
    with pytest.raises(SafetyConfigError):
        build_safety_config({"command_source": "teleop", "arms": {"a": {"bounding_box": bbox}}})


def test_real_hardware_requires_bbox_fail_closed():
    with pytest.raises(SafetyConfigError):
        build_safety_config(
            {
                "command_source": "teleop",
                "is_real_hardware": True,
                "arms": {"left": {"speed_limit": {"max_step_rad": 0.05}}},  # no bbox
            }
        )


def test_real_hardware_inference_requires_speed_limit():
    with pytest.raises(SafetyConfigError):
        build_safety_config(
            {
                "command_source": "inference",
                "is_real_hardware": True,
                "arms": {"left": {"bounding_box": {"min": [-1] * 6, "max": [1] * 6}}},  # no speed
            }
        )


def test_real_hardware_complete_config_ok():
    cfg = build_safety_config(
        {
            "command_source": "inference",
            "is_real_hardware": True,
            "arms": {
                "left": {
                    "bounding_box": {"min": [-1] * 6, "max": [1] * 6},
                    "speed_limit": {"max_step_rad": 0.05},
                }
            },
        }
    )
    assert cfg.is_real_hardware
    assert cfg.arm("left").bounding_box and cfg.arm("left").speed_limit
