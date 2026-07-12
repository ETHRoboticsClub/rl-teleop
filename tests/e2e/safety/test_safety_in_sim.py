"""Suite C — safety guardrails in sim (workspace + speed) against real FK.

Proves the guardrails constrain the *realized* follower trajectory, not just the filter
math: commands are pushed through the guardrail chain and the resulting joint command is
run through real MuJoCo forward kinematics (kinematic sim realizes commands exactly, so
FK(output) is the follower's true end-effector). Asserts the end-effector never leaves
the Cartesian box and joint steps never exceed the speed cap.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robots_realtime.runtime.safety.cartesian import (
    CartesianWorkspaceConfig,
    CartesianWorkspaceRejectGuardrail,
)
from robots_realtime.runtime.safety.config import SpeedLimitConfig
from robots_realtime.runtime.safety.fk import make_mujoco_fk
from robots_realtime.runtime.safety.guardrails import SpeedLimitGuardrail

pytestmark = [pytest.mark.safety, pytest.mark.sim, pytest.mark.e2e]

SCENE = "robots_realtime/sim/models/yam_bimanual_scene.xml"
LEFT_JOINTS = [f"left_joint{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def fk():
    return make_mujoco_fk(SCENE, LEFT_JOINTS, "left_tcp_site")


def _tight_box(fk, home, half=0.05):
    c = fk(home)
    return c - half, c + half


def _find_exit_joint(fk, home, box_lo, box_hi):
    """Find a (joint, target_value) whose FK leaves the box, for a deterministic test."""
    for j in range(6):
        for val in (1.2, -1.2):
            q = np.array(home, dtype=float)
            q[j] = val
            xyz = fk(q)
            if np.any(xyz < box_lo) or np.any(xyz > box_hi):
                return j, val
    raise AssertionError("no joint move exits the box — widen the sweep")


# ── Workspace limits ────────────────────────────────────────────────────────────


def test_workspace_reject_keeps_tcp_in_box(fk):
    home = np.zeros(6)
    lo, hi = _tight_box(fk, home)
    j, target = _find_exit_joint(fk, home, lo, hi)

    cfg = CartesianWorkspaceConfig(
        min_xyz=lo, max_xyz=hi, position_indices=tuple(range(6)), tolerance_m=1e-4
    )
    guard = CartesianWorkspaceRejectGuardrail(cfg, fk, arm="left")
    guard.reset(home)

    # Ramp the leader command from home toward the out-of-box target.
    tol = cfg.tolerance_m + 1e-6
    ever_rejected = False
    for a in np.linspace(0, 1, 60):
        cmd = np.zeros(7)
        cmd[j] = a * target
        cmd[6] = 0.5  # gripper
        out, ev = guard.apply(cmd)
        realized = fk(out[:6])
        # The realized end-effector must never leave the box (this is the whole point).
        assert np.all(realized >= lo - tol) and np.all(realized <= hi + tol), (
            f"TCP left the box at a={a:.2f}: {realized}"
        )
        assert out[6] == 0.5  # gripper passes through even while the arm holds
        ever_rejected = ever_rejected or ev is not None
    assert ever_rejected, "sweep never triggered a reject — box/target not exercised"


def test_workspace_reentry_converges_without_leaving_box(fk):
    home = np.zeros(6)
    lo, hi = _tight_box(fk, home)
    j, target = _find_exit_joint(fk, home, lo, hi)
    cfg = CartesianWorkspaceConfig(lo, hi, tuple(range(6)), reentry_max_step_rad=0.02)
    guard = CartesianWorkspaceRejectGuardrail(cfg, fk, arm="left")
    guard.reset(home)

    # Push out (reject), then command a safe in-box pose and let it re-enter.
    out_cmd = np.zeros(7)
    out_cmd[j] = target
    guard.apply(out_cmd)  # reject -> hold

    safe = np.zeros(7)  # home is in the box
    tol = cfg.tolerance_m + 1e-6
    prev = None
    for _ in range(200):
        out, _ = guard.apply(safe)
        realized = fk(out[:6])
        assert np.all(realized >= lo - tol) and np.all(realized <= hi + tol)
        if prev is not None:
            assert abs(out[j] - prev) <= cfg.reentry_max_step_rad + 1e-9  # rate-limited
        prev = out[j]
    assert out[j] == pytest.approx(0.0, abs=1e-6)  # converged back home


# ── Speed limits ────────────────────────────────────────────────────────────────


def test_speed_cap_bounds_realized_joint_velocity(fk):
    cap = 0.05
    guard = SpeedLimitGuardrail(SpeedLimitConfig(max_step_rad=cap, position_indices=tuple(range(6))))
    home = np.zeros(6)
    guard.apply(np.concatenate([home, [0.0]]))  # seed at home

    target = np.zeros(7)
    target[0] = 1.5  # big jump
    prev = home.copy()
    for _ in range(200):
        out, _ = guard.apply(target)
        step = np.abs(out[:6] - prev)
        assert np.all(step <= cap + 1e-9), f"joint step {step.max():.4f} exceeded cap {cap}"
        prev = out[:6].copy()
    assert out[0] == pytest.approx(1.5, abs=1e-6)  # eventually reaches the target


def test_speed_cap_no_false_clamp_within_limit(fk):
    cap = 0.5
    guard = SpeedLimitGuardrail(SpeedLimitConfig(max_step_rad=cap, position_indices=tuple(range(6))))
    guard.apply(np.zeros(7))
    out, ev = guard.apply(np.array([0.1, -0.2, 0.05, 0, 0, 0, 0]))  # all within cap
    np.testing.assert_allclose(out[:6], [0.1, -0.2, 0.05, 0, 0, 0])
    assert ev is None


# ── Full AgentNode wiring against the sim FK ────────────────────────────────────


def test_agent_node_cartesian_and_speed_chain_holds_tcp_in_box(fk):
    from robots_realtime.runtime.agent_node import AgentNode

    home = np.zeros(6)
    lo, hi = _tight_box(fk, home)
    j, target = _find_exit_joint(fk, home, lo, hi)

    node = AgentNode(
        agent=object(),
        name="agent",
        safety={
            "command_source": "sim_teleop",
            "arms": {
                "default": {
                    "speed_limit": {"max_step_rad": 0.1},
                    "cartesian_workspace": {
                        "min_xyz": lo.tolist(),
                        "max_xyz": hi.tolist(),
                        "fk_xml_path": SCENE,
                        "fk_joint_names": LEFT_JOINTS,
                        "fk_site_name": "left_tcp_site",
                    },
                    "bounding_box": {"min": [-3.14] * 6, "max": [3.14] * 6},
                }
            },
        },
    )
    node._build_guardrails()  # builds speed + cartesian(FK) + bbox chain

    published = {}
    node.publish = lambda topic, data, ts=None, **k: published.__setitem__(topic, data)

    tol = 1e-4 + 1e-6
    for a in np.linspace(0, 1, 60):
        cmd = np.zeros(7)
        cmd[j] = a * target
        node._publish_commands({"pos": cmd}, ts=0.0)
        realized = fk(published["joint_pos"]["joint_pos"][:6])
        assert np.all(realized >= lo - tol) and np.all(realized <= hi + tol), (
            f"TCP left the box through the AgentNode chain at a={a:.2f}: {realized}"
        )
