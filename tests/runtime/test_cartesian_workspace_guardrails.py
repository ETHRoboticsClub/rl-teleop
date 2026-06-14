"""RED tests for Cartesian workspace guardrails — implementation does NOT exist yet."""

import numpy as np
import pytest


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


@pytest.fixture
def fk():
    return MockFKProvider("wrist")


@pytest.fixture
def min_xyz():
    return [0.0, 0.0, 0.0]


@pytest.fixture
def max_xyz():
    return [1.0, 1.0, 1.0]


@pytest.fixture
def guardrail(fk, min_xyz, max_xyz):
    from robots_realtime.runtime.safety.cartesian import CartesianWorkspaceRejectGuardrail

    return CartesianWorkspaceRejectGuardrail(
        fk_provider=fk,
        arm_key="test",
        site_name="wrist",
        min_xyz=min_xyz,
        max_xyz=max_xyz,
    )


# ── Core accept/reject semantics ──────────────────────────────────────────


def test_fk_accept_in_bounds(guardrail):
    """In-bounds candidate returns the candidate unchanged."""
    candidate = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    result = guardrail.apply(candidate)
    np.testing.assert_allclose(result.final_command, candidate)


def test_outside_candidate_publishes_last_safe_command(guardrail):
    """Out-of-bounds candidate falls back to last_safe arm joints."""
    # First, set last_safe
    last_safe = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    guardrail.mark_published_safe(last_safe)

    # Now try out-of-bounds candidate
    candidate = np.array([2.0, 0.5, 0.5, 0.0, 0.0, 0.0])  # x out of bounds
    result = guardrail.apply(candidate)
    expected = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(result.final_command, expected)


def test_gripper_passthrough_on_reject(guardrail):
    """Gripper (index 6+) passes through during reject; arm joints revert."""
    # Create guardrail with gripper pass-through
    from robots_realtime.runtime.safety.cartesian import CartesianWorkspaceRejectGuardrail

    fk = MockFKProvider("wrist")
    gr = CartesianWorkspaceRejectGuardrail(
        fk_provider=fk,
        arm_key="test",
        site_name="wrist",
        min_xyz=[0.0, 0.0, 0.0],
        max_xyz=[1.0, 1.0, 1.0],
        pass_through_indices=[6],
    )

    # Set last_safe
    last_safe = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1])
    gr.mark_published_safe(last_safe)

    # Out-of-bounds candidate with different gripper value
    candidate = np.array([2.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.9])
    result = gr.apply(candidate)

    # Arm joints (0-5) revert to last_safe
    np.testing.assert_allclose(result.final_command[:6], last_safe[:6])
    # Gripper (index 6) comes from candidate
    assert result.final_command[6] == pytest.approx(0.9)


# ── Re-entry rate limiting ────────────────────────────────────────────────


def test_reentry_rate_limited_command_is_fk_checked(guardrail):
    """When a rejected command is rate-limited on re-entry, the result is FK-checked."""
    # First, set last_safe inside the box
    last_safe = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    guardrail.mark_published_safe(last_safe)

    # Now try an out-of-bounds candidate — it should be rejected
    out_of_bounds = np.array([2.0, 0.5, 0.5, 0.0, 0.0, 0.0])
    result = guardrail.apply(out_of_bounds)

    # Result stays at last_safe (or within max_delta_per_step of it), never exceeds bounds
    final_xyz = result.final_xyz
    assert final_xyz[0] <= 1.0 + 1e-6  # x within box + epsilon


# ── Per-arm scoping ───────────────────────────────────────────────────────


def test_per_arm_scope_left_rejects_right_accepts():
    """Left rejects while right accepts independently via per-arm guardrails."""
    from robots_realtime.runtime.safety.cartesian import CartesianWorkspaceRejectGuardrail

    fk_left = MockFKProvider("left_wrist")
    fk_right = MockFKProvider("right_wrist")

    left_gr = CartesianWorkspaceRejectGuardrail(
        fk_provider=fk_left,
        arm_key="left",
        site_name="left_wrist",
        min_xyz=[-0.5, -0.5, -0.5],
        max_xyz=[0.5, 0.5, 0.5],
    )
    right_gr = CartesianWorkspaceRejectGuardrail(
        fk_provider=fk_right,
        arm_key="right",
        site_name="right_wrist",
        min_xyz=[-1.0, -1.0, -1.0],
        max_xyz=[1.0, 1.0, 1.0],
    )

    left_candidate = np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0])  # in left box
    right_candidate = np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0])  # in right box too

    left_safe = np.zeros(6)
    right_safe = np.zeros(6)

    # Initialize last_safe
    left_gr.mark_published_safe(left_safe)
    right_gr.mark_published_safe(right_safe)

    # Both in bounds — both pass
    left_out = left_gr.apply(left_candidate).final_command
    right_out = right_gr.apply(right_candidate).final_command
    np.testing.assert_allclose(left_out, left_candidate)
    np.testing.assert_allclose(right_out, right_candidate)

    # Now make left out of bounds but right still in bounds
    left_candidate_oo = np.array([2.0, 0.3, 0.3, 0.0, 0.0, 0.0])  # outside left box
    right_candidate_ok = np.array([0.8, 0.8, 0.8, 0.0, 0.0, 0.0])  # inside right box

    left_out = left_gr.apply(left_candidate_oo).final_command
    right_out = right_gr.apply(right_candidate_ok).final_command

    # Left falls back to last_safe
    np.testing.assert_allclose(left_out[:3], left_safe[:3])
    # Right passes through unchanged
    np.testing.assert_allclose(right_out, right_candidate_ok)


# ── Explicit state update ─────────────────────────────────────────────────


def test_mark_published_safe_updates_last_safe(guardrail):
    """Explicit method updates last_safe for future use."""
    new_safe = np.array([0.5, 0.5, 0.5, 0.1, 0.2, 0.3])
    guardrail.mark_published_safe(new_safe)

    # Now verify a subsequent check uses updated last_safe
    out_of_bounds = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
    result = guardrail.apply(out_of_bounds)
    # Should return new_safe's arm joints since candidate is out of bounds
    np.testing.assert_allclose(result.final_command[:6], new_safe[:6])
