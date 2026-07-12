"""Unit tests for the FK-based Cartesian workspace reject/hold guardrail (mock FK)."""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.runtime.safety.cartesian import (
    CartesianConfigError,
    CartesianWorkspaceConfig,
    CartesianWorkspaceRejectGuardrail,
)

pytestmark = [pytest.mark.safety]

# Mock FK: end-effector xyz == first 3 arm joints, so the Cartesian box maps directly to
# joint space and the behaviour is easy to reason about.
FK = lambda arm: np.asarray(arm[:3], dtype=float)


def _cfg(reentry=0.05):
    return CartesianWorkspaceConfig(
        min_xyz=np.array([-1.0, -1, -1]),
        max_xyz=np.array([1.0, 1, 1]),
        position_indices=tuple(range(6)),
        reentry_max_step_rad=reentry,
    )


def _g(reentry=0.05):
    g = CartesianWorkspaceRejectGuardrail(_cfg(reentry), FK, arm="left")
    g.reset(np.zeros(6))  # start safe at origin (inside box)
    return g


def test_in_box_command_is_accepted():
    g = _g()
    out, ev = g.apply(np.array([0.5, 0.2, -0.3, 0, 0, 0, 0.9]))
    np.testing.assert_allclose(out[:6], [0.5, 0.2, -0.3, 0, 0, 0])
    assert out[6] == 0.9
    assert ev is None


def test_out_of_box_holds_last_safe():
    g = _g()
    g.apply(np.array([0.5, 0, 0, 0, 0, 0, 0]))  # move to a safe pose
    out, ev = g.apply(np.array([5.0, 0, 0, 0, 0, 0, 0.3]))  # x=5 is outside the box
    np.testing.assert_allclose(out[:6], [0.5, 0, 0, 0, 0, 0])  # held at last safe
    assert out[6] == 0.3  # gripper still passes through while the arm holds
    assert ev is not None
    assert g.reject_count == 1


def test_reentry_is_rate_limited():
    g = _g(reentry=0.05)
    g.apply(np.array([9.0, 0, 0, 0, 0, 0, 0]))  # reject -> hold at origin
    out, ev = g.apply(np.array([0.5, 0, 0, 0, 0, 0, 0]))  # back in box, but far from hold
    # Re-entry step is limited to 0.05 rad, not the full 0.5.
    assert out[0] == pytest.approx(0.05)
    assert ev is not None


def test_reentry_converges_over_steps():
    g = _g(reentry=0.1)
    g.apply(np.array([9.0, 0, 0, 0, 0, 0, 0]))  # reject
    last = None
    for _ in range(50):
        last, _ = g.apply(np.array([0.5, 0, 0, 0, 0, 0, 0]))
    assert last[0] == pytest.approx(0.5, abs=1e-9)  # eventually reaches the leader


def test_reset_fails_closed_when_current_pose_outside_box():
    g = CartesianWorkspaceRejectGuardrail(_cfg(), FK, arm="left")
    with pytest.raises(CartesianConfigError):
        g.reset(np.array([5.0, 0, 0, 0, 0, 0]))  # x=5 outside the box


def test_does_not_mutate_input():
    g = _g()
    cmd = np.array([9.0] * 7)
    original = cmd.copy()
    g.apply(cmd)
    np.testing.assert_array_equal(cmd, original)


# ── Config validation ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "d",
    [
        {"min_xyz": [0, 0], "max_xyz": [1, 1, 1]},        # wrong shape
        {"min_xyz": [2, 0, 0], "max_xyz": [1, 1, 1]},     # min > max
        {"min_xyz": [float("nan"), 0, 0], "max_xyz": [1, 1, 1]},  # non-finite
        {"max_xyz": [1, 1, 1]},                            # missing min
    ],
)
def test_invalid_cartesian_config_raises(d):
    with pytest.raises(CartesianConfigError):
        CartesianWorkspaceConfig.from_dict(d, "left")
