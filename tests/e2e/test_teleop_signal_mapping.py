"""Suite B — teleop signal-correctness / characterization (golden) tests.

Freezes the current leader→follower joint mapping so a regression that flips an axis,
swaps a joint index, or rescales a joint is caught before it reaches a robot. The
mapping is a pure function of (raw reading, calibration), so these run with an injected
fake reader — no hardware.

Primary coverage is the Dynamixel GELLO mapping (richest transform + configurable
signs). GELLO/Feetech and YAM leaders need lerobot/i2rt and are guarded with importorskip
so they exercise in CI where those deps exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent import (
    DynamixelGelloLeaderAgent,
)

pytestmark = [pytest.mark.e2e]

TICKS_PER_REV = 4096
TICKS_TO_RAD = 2.0 * np.pi / TICKS_PER_REV
MID = 2048  # ticks → 0 rad (center of range, no wrap nearby)
N = 6


class _Reader:
    """Minimal injectable Dynamixel reader returning fixed ticks."""

    def __init__(self, ticks):
        self._ticks = np.asarray(ticks, dtype=np.float64)
        self._fail = False

    def set(self, ticks):
        self._ticks = np.asarray(ticks, dtype=np.float64)

    def get_positions(self):
        if self._fail:
            raise RuntimeError("simulated read failure")
        return self._ticks.copy()

    def seconds_since_last_read(self):
        return 0.0

    def close(self):
        pass


def _agent(reader, **kw):
    return DynamixelGelloLeaderAgent(reader=reader, **kw)


def _pos(agent):
    return np.asarray(agent.act({})[agent.robot_name]["pos"], dtype=np.float64)


def _baseline_ticks(gripper=MID):
    return [MID] * N + [gripper]


# ── Directionality / no inversion ──────────────────────────────────────────────


@pytest.mark.parametrize("joint", range(N))
def test_positive_tick_moves_output_positively_with_default_signs(joint):
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left")
    base = _pos(agent)

    bumped = _baseline_ticks()
    bumped[joint] += 100  # small, no wrap
    reader.set(bumped)
    after = _pos(agent)

    # Default joint_signs = +1 → increasing ticks increases the joint angle.
    assert after[joint] > base[joint], f"joint {joint} did not move in the + direction"


@pytest.mark.parametrize("joint", range(N))
def test_negative_sign_inverts_that_joint(joint):
    signs = [1] * N
    signs[joint] = -1
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left", joint_signs=signs)
    base = _pos(agent)

    bumped = _baseline_ticks()
    bumped[joint] += 100
    reader.set(bumped)
    after = _pos(agent)

    assert after[joint] < base[joint], f"sign=-1 did not invert joint {joint}"


# ── Proportionality / scale ────────────────────────────────────────────────────


def test_output_is_proportional_to_tick_delta():
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left")
    base = _pos(agent)

    delta_ticks = 200
    bumped = _baseline_ticks()
    for j in range(N):
        bumped[j] += delta_ticks
    reader.set(bumped)
    after = _pos(agent)

    expected = delta_ticks * TICKS_TO_RAD  # signs=1, scales=1
    np.testing.assert_allclose(after[:N] - base[:N], expected, atol=1e-6)


def test_joint_scale_multiplies_slope():
    scales = [2.0] * N
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left", joint_scales=scales)
    base = _pos(agent)

    bumped = _baseline_ticks()
    for j in range(N):
        bumped[j] += 50
    reader.set(bumped)
    after = _pos(agent)

    expected = 50 * 2.0 * TICKS_TO_RAD
    np.testing.assert_allclose(after[:N] - base[:N], expected, atol=1e-6)


# ── Single-axis isolation / no cross-talk ──────────────────────────────────────


@pytest.mark.parametrize("joint", range(N))
def test_single_axis_isolation(joint):
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left")
    base = _pos(agent)

    bumped = _baseline_ticks()
    bumped[joint] += 137
    reader.set(bumped)
    after = _pos(agent)

    for other in range(N):
        if other == joint:
            assert after[other] != base[other]
        else:
            # Untouched joints must be bit-identical — no cross-talk / index bleed.
            assert after[other] == base[other], f"joint {other} moved when only {joint} changed"


# ── Bimanual independence ──────────────────────────────────────────────────────


def test_bimanual_independence():
    left_reader = _Reader(_baseline_ticks())
    right_reader = _Reader(_baseline_ticks())
    left = _agent(left_reader, robot_name="left")
    right = _agent(right_reader, robot_name="right")

    right_base = _pos(right)
    # Move the left leader only.
    moved = _baseline_ticks()
    moved[3] += 300
    left_reader.set(moved)
    _pos(left)

    # Right follower command is unchanged — arms are independent.
    np.testing.assert_array_equal(_pos(right), right_base)


# ── Gripper mapping ─────────────────────────────────────────────────────────────


def test_gripper_open_closed_maps_to_unit_interval():
    open_ticks, closed_ticks = 2280, 1670
    reader = _Reader([MID] * N + [open_ticks])
    agent = _agent(
        reader,
        robot_name="left",
        include_gripper=True,
        gripper_open_ticks=open_ticks,
        gripper_closed_ticks=closed_ticks,
    )
    assert _pos(agent)[N] == pytest.approx(1.0)  # open → 1.0

    reader.set([MID] * N + [closed_ticks])
    assert _pos(agent)[N] == pytest.approx(0.0)  # closed → 0.0

    reader.set([MID] * N + [(open_ticks + closed_ticks) // 2])
    mid = _pos(agent)[N]
    assert 0.0 < mid < 1.0


def test_gripper_is_clipped_to_unit_interval():
    open_ticks, closed_ticks = 2280, 1670
    reader = _Reader([MID] * N + [open_ticks + 1000])  # past open
    agent = _agent(
        reader,
        robot_name="left",
        include_gripper=True,
        gripper_open_ticks=open_ticks,
        gripper_closed_ticks=closed_ticks,
    )
    assert _pos(agent)[N] == pytest.approx(1.0)

    reader.set([MID] * N + [closed_ticks - 1000])  # past closed
    assert _pos(agent)[N] == pytest.approx(0.0)


def test_gripper_range_mode_direction():
    reader = _Reader([MID] * N + [0])
    agent = _agent(
        reader,
        robot_name="left",
        include_gripper=True,
        gripper_range_ticks=1000,
    )
    # range mode: 1 - clip(|rad|/range). At 0 ticks → 1.0 (fully one side).
    assert _pos(agent)[N] == pytest.approx(1.0)


# ── Wrap boundary (discontinuity hazard) ───────────────────────────────────────


def test_output_always_within_pi():
    # Sweep the full tick range; every joint output must stay in [-pi, pi).
    for t in range(0, TICKS_PER_REV, 97):
        reader = _Reader([t] * N + [MID])
        agent = _agent(reader, robot_name="left")
        out = _pos(agent)[:N]
        # Output is float32, so allow a float32-sized epsilon around ±pi.
        assert np.all(out >= -np.pi - 1e-5)
        assert np.all(out < np.pi + 1e-5)


def test_center_tick_is_zero():
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left")
    np.testing.assert_allclose(_pos(agent)[:N], 0.0, atol=1e-9)


# ── Stale / non-finite fallback (currently SILENT — see fail-loud work) ─────────


def test_read_failure_reuses_last_command():
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left")
    good = _pos(agent)

    reader._fail = True
    # act() must not raise; it reuses the last good command. NOTE: this is currently
    # SILENT — a persistently-failing leader keeps the arm on a stale target. The
    # fail-loud / watchdog work is what surfaces this; this test pins today's behaviour.
    after = _pos(agent)
    np.testing.assert_array_equal(after, good)


# ── Safety: max_delta_rad per-step jump cap ────────────────────────────────────


def test_max_delta_rad_clamps_a_large_jump():
    """A huge single-step tick jump is rate-limited to max_delta_rad, not passed raw."""
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left", max_delta_rad=0.01)
    base = _pos(agent)  # seeds the reference at 0 rad

    jumped = _baseline_ticks()
    jumped[0] += 800  # ~1.2 rad jump, far beyond max_delta_rad=0.01
    reader.set(jumped)
    after = _pos(agent)

    # Joint 0 advances by at most one max_delta_rad step toward the target.
    assert abs(after[0] - base[0]) == pytest.approx(0.01, abs=1e-6)
    # Untouched joints do not move.
    np.testing.assert_allclose(after[1:N], base[1:N], atol=1e-6)


def test_max_delta_rad_ramps_toward_target_over_steps():
    """Repeated clamping walks the reference toward the leader (ramp, not a hard stop)."""
    reader = _Reader(_baseline_ticks())
    agent = _agent(reader, robot_name="left", max_delta_rad=0.05)
    _pos(agent)

    jumped = _baseline_ticks()
    jumped[0] += 800
    reader.set(jumped)
    outs = [_pos(agent)[0] for _ in range(4)]
    # Each step advances ~0.05 rad; monotonic ramp toward the (positive) target.
    assert outs == sorted(outs)
    assert outs[1] - outs[0] == pytest.approx(0.05, abs=1e-6)


def test_max_delta_rad_does_not_clamp_first_command():
    """The first read seeds the reference and is emitted un-clamped (matches the follower)."""
    ticks = _baseline_ticks()
    ticks[0] += 800  # first command is already far from the zero seed
    reader = _Reader(ticks)
    agent = _agent(reader, robot_name="left", max_delta_rad=0.01)
    first = _pos(agent)[0]
    assert abs(first) > 0.5  # not throttled down to 0.01 on the very first step


# ── CI-only: other leader mappings ─────────────────────────────────────────────


def test_gello_feetech_mapping_importable():
    pytest.importorskip("lerobot")
    from robots_realtime.agents.teleoperation import gello_leader_agent  # noqa: F401


def test_yam_leader_mapping_importable():
    pytest.importorskip("i2rt")
    from robots_realtime.agents.teleoperation import yam_leader_agent  # noqa: F401
