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
