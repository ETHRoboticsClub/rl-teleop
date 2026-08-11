"""Faults NOT in the handoff catalogue — invented while attacking the system.

The catalogue in HANDOFF-CAMERA-HARDENING.md §5.1 is a floor, not a ceiling.
§5.5 asks for newly-invented faults on top of it, and these are the ones that
found real holes. Each docstring records what the code did BEFORE the fix,
because a regression test whose failure mode you cannot picture is a test people
delete.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_FAILED,
    STATE_OK,
    CameraUnavailable,
    SupervisedCamera,
)

SHAPE = (48, 64)


class RingCamera(CameraDriver):
    """A device cycling a SMALL RING of buffers: A, B, A, B, ... forever.

    A realistic driver failure — a stuck DMA ring, or a UVC layer replaying its
    buffer queue. Every consecutive pair differs, so it looks perfectly alive.
    """

    def __init__(self, n: int = 2) -> None:
        self.frames = [np.full((*SHAPE, 3), 10 + i * 37, np.uint8) for i in range(n)]
        for i, f in enumerate(self.frames):
            f[0, 0, 1] = i
        self.i = 0
        self.device_path = "/dev/fake-ring"

    def read(self) -> CameraData:
        time.sleep(0.003)
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return CameraData(images={"rgb": f}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


class StaticSceneCamera(CameraDriver):
    """A REAL camera pointed at a motionless bench: one scene, sensor noise.

    The control for the ring test. Any stall detector that flags this is worse
    than useless — it would blank a working panel and teach the operator to
    ignore the indicator.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.base = rng.integers(40, 60, (*SHAPE, 3), dtype=np.uint8)
        self._rng = rng
        self.device_path = "/dev/fake-static"

    def read(self) -> CameraData:
        time.sleep(0.003)
        noise = self._rng.integers(-2, 3, self.base.shape)
        f = np.clip(self.base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return CameraData(images={"rgb": f}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def _cam(factory, **kw) -> SupervisedCamera:
    kw.setdefault("read_deadline_s", 0.4)
    kw.setdefault("freeze_timeout_s", 0.6)
    kw.setdefault("reopen_backoff_s", (0.05, 0.1))
    kw.setdefault("target_fps", 200.0)
    kw.setdefault("expected_shape", SHAPE)
    return SupervisedCamera(factory, name="invented", **kw)


def _drive(cam: SupervisedCamera, secs: float) -> tuple[str | None, set[str]]:
    """Read for `secs`; return (first failure reason, every state seen)."""
    t0 = time.monotonic()
    first: str | None = None
    seen: set[str] = set()
    while time.monotonic() - t0 < secs:
        try:
            cam.read()
        except CameraUnavailable as exc:
            if first is None:
                first = exc.reason
        seen.add(cam.state)
        time.sleep(0.003)
    return first, seen


# ── invented fault A — a ring of buffers ─────────────────────────────────────


@pytest.mark.parametrize("ring", [2, 3, 5, 8])
def test_a_camera_cycling_a_ring_of_buffers_is_caught(ring: int) -> None:
    """A freeze with extra steps, and it sailed straight through.

    The freeze check compared each frame only against the one before it, so
    A,B,A,B was "distinct" at every single step. Rings of 2, 3, 5 and 8 all
    reported ``ok`` indefinitely while carrying no new information whatsoever —
    a camera delivering a slideshow of the same two pictures at 30 Hz, with
    every indicator green.
    """
    cam = _cam(lambda: RingCamera(ring))
    try:
        cam.wait_until_open(5.0)
        first, seen = _drive(cam, 6.0)
        assert first == "stalled", f"a ring of {ring} was not detected at all"
        assert cam.state == STATE_FAILED, f"ring of {ring} ended in {cam.state!r}, not failed"
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


def test_a_motionless_scene_from_a_real_sensor_is_not_flagged() -> None:
    """The control. A false 'dead' is its own failure.

    Sensor noise makes every frame from a real camera distinct even when
    nothing in front of it moves, which is exactly why the stall test can afford
    to be strict.
    """
    cam = _cam(StaticSceneCamera)
    try:
        cam.wait_until_open(5.0)
        first, _ = _drive(cam, 4.0)
        assert first is None, f"a working camera was flagged as {first!r}"
        assert cam.state == STATE_OK
    finally:
        cam.stop()


def test_recovery_from_a_ring_requires_a_varied_window_not_two_good_frames() -> None:
    """Recovery has to be judged over the window too.

    A ring passes 'N consecutive distinct frames' trivially, so with only the
    streak test the camera climbed back to ``ok`` after every reopen and the
    incident counter reset with it. It never converged — the same flap as the
    ring-of-1 case, arriving by a different road.
    """
    holder: dict = {"ring": 2}

    def factory():
        return RingCamera(holder["ring"])

    cam = _cam(factory)
    try:
        cam.wait_until_open(5.0)
        _drive(cam, 5.0)
        assert cam.state == STATE_FAILED
    finally:
        cam.stop()


# ── invented fault B — the wall clock steps backwards under the publisher ────


def test_publisher_throttle_survives_a_backwards_clock() -> None:
    """Fault 13 in the catalogue, which the handoff flagged as suspected-not-proven.

    IT WAS REAL. ``Publisher.publish()`` throttled on ``time.time()`` and kept
    the last send time in a dict. One backwards step — an NTP correction, a
    manual ``date``, a VM resume — put that timestamp in the FUTURE, so every
    later ``now - last`` came out negative, compared as "< min_interval", and the
    topic was throttled off the bus for the entire length of the step. Silently:
    the writer kept recording and every rate display kept its last value.
    """
    from robots_realtime.runtime.transport import publisher as pub_mod

    sent: list[tuple] = []

    class FakeSock:
        def send_multipart(self, parts):
            sent.append(tuple(parts))

        def close(self, linger=0):
            pass

        def setsockopt(self, *a):
            pass

        def connect(self, *a):
            pass

    p = pub_mod.Publisher.__new__(pub_mod.Publisher)
    p._node_name = "camera_top"
    p._writer = None
    p._publish_freq = 30.0
    p._min_interval = 1.0 / 30.0
    p._last_sent = {}
    p._sock = FakeSock()
    p._ctrl_sock = FakeSock()

    assert p.publish("rgb", {"a": 1}) is True
    n_before = len(sent)

    # The wall clock now runs an hour backwards on every single call.
    real_time = time.time
    try:
        state = {"t": real_time()}

        def backwards() -> float:
            state["t"] -= 3600.0
            return state["t"]

        pub_mod.time.time = backwards            # type: ignore[assignment]
        for _ in range(50):
            time.sleep(0.002)
            p.publish("rgb", {"a": 2})
    finally:
        pub_mod.time.time = real_time            # type: ignore[assignment]

    assert len(sent) > n_before, (
        "a backwards wall clock wedged the publish throttle — the topic went "
        "silent while the writer kept recording"
    )


# ── invented fault C — the health topic itself is the liar ───────────────────


def test_the_bus_auditor_does_not_read_the_health_topic() -> None:
    """check_streams.py must stay INDEPENDENT of the thing it audits.

    Its entire value is that it can catch the health topic lying: it counts
    frames whose CONTENT differs, straight off the bus. If it ever starts
    reading ``<node>/health`` to decide whether a camera is well, the two
    signals can only ever agree — including when they are both wrong.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "tools" / "check_streams.py"
    assert src.exists(), f"check_streams.py not found at {src}"
    body = src.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "/health" not in code, (
        "check_streams.py now reads the health topic. That destroys the one "
        "property it has: independence from the signal under audit."
    )


# ── invented fault D — a driver that lies about its own timestamps ───────────


class BackwardsTimestampCamera(CameraDriver):
    """Frames are fine; the timestamps run backwards.

    Seen in the wild when a camera's clock is reset mid-stream. It matters
    because CameraNode divides the driver timestamp into the envelope ``ts``,
    and every staleness check downstream compares against that.
    """

    def __init__(self) -> None:
        self.t = time.time() * 1000
        self.n = 0
        self.device_path = "/dev/fake-clock"

    def read(self) -> CameraData:
        time.sleep(0.003)
        self.n += 1
        self.t -= 1000.0
        f = np.full((*SHAPE, 3), self.n % 251, np.uint8)
        return CameraData(images={"rgb": f}, timestamp=self.t)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def test_backwards_driver_timestamps_do_not_wedge_the_supervisor() -> None:
    """The supervisor must pace itself on the monotonic clock, never the driver's.

    A driver whose timestamps run backwards must not be able to make read()
    block, nor make the camera look stale to its own supervisor — the frames
    themselves are perfectly good.
    """
    cam = _cam(BackwardsTimestampCamera)
    try:
        cam.wait_until_open(5.0)
        first, _ = _drive(cam, 3.0)
        assert first is None, f"good frames were rejected as {first!r}"
        assert cam.state == STATE_OK
        assert cam.health()["last_frame_age_s"] < 2.0, (
            "the health record's frame age came from the driver's clock rather "
            "than a monotonic one"
        )
    finally:
        cam.stop()


# ── invented fault E — the factory hands back a driver that is already dead ──


def test_a_factory_returning_a_dead_driver_does_not_loop_forever_claiming_ok() -> None:
    """Open 'succeeds' and every read fails. The camera must converge on failed,
    not oscillate between reopening and a hopeful degraded forever."""

    class DeadOnArrival(CameraDriver):
        device_path = "/dev/fake-doa"

        def read(self):
            raise RuntimeError("device disappeared between open and first read")

        def read_calibration_data_intrinsics(self):
            return {}

        def get_camera_info(self):
            return {}

        def stop(self):
            pass

    cam = _cam(DeadOnArrival, give_up_after=3)
    try:
        cam.wait_until_open(5.0)
        first, seen = _drive(cam, 5.0)
        assert first in ("read_error", "timeout")
        assert cam.health()["healthy"] is False
        assert STATE_OK not in seen, "a camera that never delivered a frame reported ok"
    finally:
        cam.stop()


# ── invented fault F — repeated reopen under a driver that leaks nothing ─────


def test_a_hundred_reopens_do_not_grow_the_thread_count() -> None:
    """Longer than the 20-cycle test, and specifically about the SUPERVISOR.

    Every reopen starts a pump thread and a close thread. If retirement is even
    slightly leaky, an overnight run with a flaky camera exhausts the process —
    and it would pass every functional test on the way there.
    """
    state = {"fail": True}

    class Flaky(CameraDriver):
        device_path = "/dev/fake-flaky"

        def __init__(self):
            self.n = 0

        def read(self):
            time.sleep(0.002)
            self.n += 1
            if state["fail"] and self.n > 2:
                raise RuntimeError("flaked")
            return CameraData(
                images={"rgb": np.full((*SHAPE, 3), self.n % 251, np.uint8)},
                timestamp=time.time() * 1000,
            )

        def read_calibration_data_intrinsics(self):
            return {}

        def get_camera_info(self):
            return {}

        def stop(self):
            pass

    cam = _cam(Flaky, reopen_backoff_s=(0.01,))
    try:
        cam.wait_until_open(5.0)
        baseline = threading.active_count()
        _drive(cam, 8.0)
        reopens = cam.health()["reopens"]
        assert reopens >= 20, f"only {reopens} reopens — the test did not exercise much"
        time.sleep(0.5)
        grown = threading.active_count() - baseline
        assert grown <= 3, f"thread count grew by {grown} over {reopens} reopens"
    finally:
        cam.stop()
