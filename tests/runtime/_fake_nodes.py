"""Importable fake Node subclasses for fail-loud startup tests.

These live in their own module (not inside the test file) so they can be pickled and
re-imported by the ``multiprocessing`` spawn worker that ProcessHost uses.
"""

from __future__ import annotations

from robots_realtime.runtime.node import Node


class OkNode(Node):
    """A node that comes up fine and publishes a heartbeat."""

    poll_freq = 50.0

    def __init__(self, name: str, critical: bool = True) -> None:
        super().__init__(name=name, critical=critical)

    def setup(self) -> None:  # opens no hardware
        pass

    def step(self) -> None:
        self.publish("tick", {"n": 1})


class BusyNode(Node):
    """A node whose hardware is held by someone else — setup() fails loudly."""

    poll_freq = 50.0

    def __init__(self, name: str, critical: bool = True, device: str = "/dev/fake0") -> None:
        super().__init__(name=name, critical=critical)
        self._device = device

    def setup(self) -> None:
        from robots_realtime.runtime.preflight import (
            DeviceBusyError,
            DeviceReason,
            HolderInfo,
        )

        raise DeviceBusyError(
            self._device,
            DeviceReason.BUSY,
            [HolderInfo(pid=999, name="python", username="someone")],
        )

    def step(self) -> None:
        pass


class HangNode(Node):
    """A node whose setup() never returns — must be caught by the bring-up timeout."""

    poll_freq = 50.0

    def __init__(self, name: str, critical: bool = True) -> None:
        super().__init__(name=name, critical=critical)

    def setup(self) -> None:
        import time

        while True:
            time.sleep(0.1)

    def step(self) -> None:
        pass


class ScriptedLeaderNode(Node):
    """A stand-in leader: publishes a fixed target joint command at a fixed rate."""

    poll_freq = 100.0

    def __init__(self, name: str, target, critical: bool = True) -> None:
        super().__init__(name=name, critical=critical)
        self._target = list(target)

    def setup(self) -> None:
        pass

    def step(self) -> None:
        import numpy as np

        self.publish("joint_pos", {"joint_pos": np.asarray(self._target, dtype=np.float32)})


class SimFollowerNode(Node):
    """A follower: applies commands from ``cmd_topic`` to a MujocoSimRobot and
    republishes the realized joint state — the follower half of a teleop loop."""

    subscriber_driven = True
    poll_freq = 100.0
    published_topics = ["joint_state"]

    def __init__(self, name: str, cmd_topic: str, xml_path: str, critical: bool = True) -> None:
        self.subscribed_topics = [cmd_topic]
        super().__init__(name=name, critical=critical)
        self._cmd_topic = cmd_topic
        self._xml_path = xml_path
        self._robot = None

    def setup(self) -> None:
        from robots_realtime.robots.mujoco_sim_robot import MujocoSimRobot

        self._robot = MujocoSimRobot(xml_path=self._xml_path, render=False)

    def step(self) -> None:
        import numpy as np

        cmd = self.get_latest(self._cmd_topic)
        if cmd is not None:
            self._robot.command_joint_pos(np.asarray(cmd["joint_pos"], dtype=np.float64))
        self.publish("joint_state", self._robot.get_observations())
