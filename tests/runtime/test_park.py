"""There must always be a way to bring the arm home.

THE INCIDENT THESE TESTS ENCODE (2026-08-12). A policy agent node died at
startup. The right arm was then stuck, energised, in a reaching pose over the
source box, and every software route home was blocked at once:

  * `start_paused` gated the arm, so publishing on the command topic did nothing;
  * `Session.resume()` walks hosts in order and hung forever on the dead agent's
    control socket — it never reached the arm — while `/status` reported
    `paused: false`, so the session's own view of itself was wrong;
  * the agent that owned the command topic WAS the dead node, so nothing was
    left to command a pose with.

The only way out was to kill the session and have a person physically hold the
arm while it sagged. On an arm with no brakes, a bug must never be able to leave
the hardware there.

So: park() depends on nothing but the arm node itself, and a dead sibling node
can neither block it nor hide it.
"""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.runtime.environment.robot_node import RobotNode
from robots_realtime.runtime.session import NodeStatus, Session


class FakeRobot:
    """Records every commanded pose so a test can see where the arm went."""

    def __init__(self, start=(0.4, 2.0, 1.6, -0.7, 0.0, -0.5, 1.0)):
        self.q = np.asarray(start, dtype=np.float64)
        self.commands: list[np.ndarray] = []
        self.stopped = False

    def command_joint_pos(self, q) -> None:
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.commands.append(self.q.copy())

    def get_joint_pos(self):
        return self.q.copy()

    def get_observations(self):
        return {"joint_pos": self.q[:6].copy(), "gripper_pos": self.q[6:].copy()}

    def stop(self):
        self.stopped = True


HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _node(**kw) -> RobotNode:
    robot = FakeRobot()
    n = RobotNode(robot=robot, name="yam_right", cmd_topic=None,
                  startup_joint_pos=None, shutdown_joint_pos=HOME, **kw)
    return n


# ── the node itself ──────────────────────────────────────────────────────────


def test_park_ramps_the_arm_to_home() -> None:
    n = _node()
    n.park(duration_s=0.1)
    assert np.allclose(n._robot.q, HOME, atol=1e-6), f"ended at {n._robot.q}"
    assert len(n._robot.commands) > 1, "park must RAMP, not jump straight there"


def test_park_works_while_paused() -> None:
    """The gate must not be able to trap the arm.

    Being paused is the normal state when something has gone wrong — it is what
    start_paused does — so a park that waits politely for the gate is a park
    that is unavailable exactly when it is needed.
    """
    n = _node()
    n.pause()
    assert n.is_paused
    n.park(duration_s=0.1)
    assert np.allclose(n._robot.q, HOME, atol=1e-6)


def test_park_leaves_the_node_paused() -> None:
    """Parking takes control away and keeps it.

    Whatever was driving the arm when you decided to park it does not get the
    arm back without somebody saying so.
    """
    n = _node()
    n.park(duration_s=0.1)
    assert n.is_paused is True


def test_park_prefers_home_over_shutdown_pose() -> None:
    reach = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n = _node(home_joint_pos=reach)
    n.park(duration_s=0.1)
    assert np.allclose(n._robot.q, reach, atol=1e-6)


def test_park_refuses_loudly_with_nowhere_to_go() -> None:
    robot = FakeRobot()
    n = RobotNode(robot=robot, name="yam_right", cmd_topic=None,
                  startup_joint_pos=None, shutdown_joint_pos=None)
    with pytest.raises(RuntimeError, match="nothing to park to"):
        n.park(duration_s=0.1)


def test_park_forces_a_fresh_handoff_ramp_afterwards() -> None:
    """After a park, a resumed publisher must ramp from where the arm IS.

    Without clearing the command bookkeeping, the next command would be treated
    as a continuation of the stream that existed before the park and applied
    directly — snapping the arm from home back toward a target chosen for a
    completely different situation.
    """
    n = _node()
    n._last_msg_ts = 12345.0
    n._ramping = True
    n.park(duration_s=0.1)
    assert n._last_msg_ts == 0.0
    assert n._ramping is False


# ── the session ──────────────────────────────────────────────────────────────


class FakeHost:
    def __init__(self, name: str, alive: bool = True, raises: bool = False):
        self._name = name
        self._alive = alive
        self._raises = raises
        self.parked = False

    @property
    def node_name(self) -> str:
        return self._name

    @property
    def node_names(self) -> list[str]:
        return [self._name]

    def is_alive(self) -> bool:
        return self._alive

    def park(self, duration_s=None) -> str:
        if self._raises:
            raise RuntimeError("control socket timed out")
        self.parked = True
        return "parked at [0, 0, 0, 0, 0, 0, 0]"


def _session(tmp_path, hosts):
    s = Session([], save_root=tmp_path)
    s._hosts = list(hosts)                       # type: ignore[assignment]
    s._status = {h.node_name: NodeStatus(name=h.node_name) for h in hosts}
    return s


def test_a_dead_node_cannot_block_parking_the_arm(tmp_path) -> None:
    """THE REGRESSION. The dead node is FIRST, exactly as it was on the rig.

    Session.resume() walked the hosts in order and blocked forever on this one,
    so the arm behind it never got the message. park() must skip it.
    """
    dead = FakeHost("act_right", alive=False)
    arm = FakeHost("yam_right", alive=True)
    s = _session(tmp_path, [dead, arm])

    out = s.park()

    assert arm.parked is True, "the arm was not parked because a dead node came first"
    assert out["ok"] is True
    assert "skipped" in out["results"]["act_right"]


def test_a_failing_node_is_reported_not_swallowed(tmp_path) -> None:
    """'Probably parked' is not a state anyone can act on."""
    arm = FakeHost("yam_right", alive=True, raises=True)
    s = _session(tmp_path, [arm])

    out = s.park()

    assert out["ok"] is False
    assert out["failed"] == ["yam_right"]
    assert "FAILED" in out["results"]["yam_right"]


def test_parking_leaves_the_session_paused(tmp_path) -> None:
    """The session's own view must agree with what it just did to the hardware.

    The incident's second half was /status reporting `paused: false` while the
    arm node was still gated. Whatever else is true after a park, the session
    must not claim something is free to drive the arm.
    """
    arm = FakeHost("yam_right", alive=True)
    s = _session(tmp_path, [arm])
    s._is_paused = False

    s.park()

    assert s.is_paused is True
