"""Suite A — "teleop works" end-to-end loop in sim (no hardware).

Drives a real Session through the real ZMQ node graph: a scripted leader publishes a
target joint command, a follower applies it to a MuJoCo sim robot and republishes the
realized state. Proves the plumbing — spawn, publish/subscribe, command flow, liveness,
clean shutdown — works end to end and the follower actually tracks the leader.
"""

from __future__ import annotations

import os
import socket
import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("i2rt")

import i2rt
import numpy as np
import zmq

from robots_realtime.runtime.session import Session
from robots_realtime.runtime.transport.serialization import unpack
from tests.runtime._fake_nodes import ScriptedLeaderNode, SimFollowerNode

pytestmark = [pytest.mark.e2e, pytest.mark.sim]

YAM_XML = os.path.join(os.path.dirname(i2rt.__file__), "robot_models/arm/yam/yam.xml")
TARGET = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_leader_to_follower_command_flow(tmp_path):
    pub_port, sub_port = _free_port(), _free_port()
    session = Session(
        nodes=[
            ScriptedLeaderNode("leader", TARGET),
            SimFollowerNode("follower", "leader/joint_pos", YAM_XML),
        ],
        save_root=str(tmp_path),
    )
    # Propagate the ports to the nodes too (Session.__init__ only wires the bus).
    session.configure_bus_ports(pub_port=pub_port, sub_port=sub_port)

    # Independent bus subscriber to read what the follower actually publishes.
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{sub_port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"follower/joint_state")

    try:
        session.start()

        last_pos = None
        msg_count = 0
        tracked = False
        deadline = time.monotonic() + 8.0
        # Read for a sustained window so we can assert continuous publishing (not just a
        # single lucky frame), and confirm the follower converges to the commanded pose.
        while time.monotonic() < deadline and not (tracked and msg_count >= 5):
            if sock.poll(100):
                parts = sock.recv_multipart()
                if len(parts) >= 2:
                    env = unpack(parts[1])
                    data = env.get("data", {})
                    if "joint_pos" in data:
                        msg_count += 1
                        last_pos = np.asarray(data["joint_pos"])[:6]
                        tracked = tracked or np.allclose(last_pos, TARGET, atol=1e-3)

        # The follower published continuously and tracked the leader's commanded pose.
        assert last_pos is not None, "follower never published joint_state"
        assert msg_count >= 5, f"follower published only {msg_count} frames (not sustained)"
        assert tracked, f"follower did not track the target; last={last_pos}"

        statuses = {s.name: s for s in session.node_statuses()}
        assert statuses["leader"].alive and statuses["follower"].alive
        assert session.fatal_reason is None  # no critical failure
    finally:
        sock.close(linger=0)
        session.stop()
