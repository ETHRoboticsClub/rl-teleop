"""Frame-drop watchdog (PLAN-FRAME-DROP-WATCHDOG.md, 2026-09-07).

Every camera counts its own gaps and publishes ONE tier — quiet / warn / loud —
so every dashboard shows the same verdict. These tests pin the three properties
that make that trustworthy:

  * counting is telemetry: a drop never changes `state`, never reopens;
  * both failure shapes are seen: bursts (gap events) AND a steady shortfall
    with no single long gap (the ledger);
  * incidents (freeze / stall / reopen / read failure) are loud outright.

Written against the 2026-09-07 right-wrist stall: 21.7 s of silence, no USB
disconnect, arm stationary, the supervisor reopened and recovered — the exact
event the operator had asked to be told about.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_OK,
    CameraUnavailable,
    SupervisedCamera,
)

SHAPE = (48, 64)


class ScriptedCamera(CameraDriver):
    """Distinct frames at a rate the test controls, with holes it can inject.

    ``hold`` blocks read() for that many seconds ONCE (a burst / stall);
    ``period_s`` is the steady rate (a shortfall when below target).
    """

    def __init__(self, period_s: float = 0.005) -> None:
        self.period_s = period_s
        self.n = 0
        self.hold = 0.0
        self.device_path = "/dev/fake-scripted"
        self._lock = threading.Lock()

    def read(self) -> CameraData:
        with self._lock:
            hold, self.hold = self.hold, 0.0
        if hold:
            time.sleep(hold)
        else:
            time.sleep(self.period_s)
        self.n += 1
        return CameraData(
            images={"rgb": np.full((*SHAPE, 3), self.n % 251, np.uint8)},
            timestamp=time.time() * 1000,
        )

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def _cam(factory, **kw) -> SupervisedCamera:
    kw.setdefault("read_deadline_s", 2.0)
    kw.setdefault("freeze_timeout_s", 5.0)
    kw.setdefault("reopen_backoff_s", (0.05, 0.1))
    kw.setdefault("target_fps", 100.0)
    kw.setdefault("expected_shape", SHAPE)
    kw.setdefault("slow_grace_s", 60.0)      # keep the existing slow-state out of the way
    kw.setdefault("alert_window_s", 10.0)
    return SupervisedCamera(factory, name="dropwatch", **kw)


def _drive(cam: SupervisedCamera, secs: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        try:
            cam.read()
        except CameraUnavailable:
            pass
        time.sleep(0.002)


def test_a_clean_camera_is_quiet_with_zero_counters() -> None:
    cam = _cam(ScriptedCamera)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 1.0)
        h = cam.health()
        assert h["alert"] == "quiet", h
        assert h["drops_total"] == 0 and h["frames_lost_total"] == 0
        assert h["worst_gap_ms"] < 1000
        assert h["incidents"] == 0
    finally:
        cam.stop()


def test_a_burst_is_a_drop_event_with_the_frames_it_swallowed() -> None:
    """One 300 ms hole at 100 Hz is one drop of ~29 frames — and NOT a state change."""
    drv = ScriptedCamera()
    cam = _cam(lambda: drv)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 0.5)
        assert cam.state == STATE_OK
        drv.hold = 0.3
        _drive(cam, 0.8)
        h = cam.health()
        assert h["drops_total"] == 1, h
        assert 20 <= h["frames_lost_total"] <= 40, h
        assert 250 <= h["worst_gap_ms"] <= 450, h
        # telemetry only: the hole did not touch the supervisor's verdict
        assert cam.state == STATE_OK
        assert h["reopens"] == 0
        assert h["incidents"] == 0
    finally:
        cam.stop()


def test_loss_percent_drives_the_tier_warn_then_loud() -> None:
    """Enough holes push loss over the warn line, then the loud line."""
    drv = ScriptedCamera(period_s=0.005)
    cam = _cam(lambda: drv, alert_window_s=10.0, alert_warn_pct=1.0, alert_loud_pct=5.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 1.5)               # ~150 frames at the 100 Hz pace
        drv.hold = 0.05                # ~4 lost of ~150 → 2-3 %, the warn band
        _drive(cam, 0.6)
        h = cam.health()
        assert h["alert"] == "warn", h
        assert "frames lost" in h["alert_reason"]
        for _ in range(4):             # pile on → loud
            drv.hold = 0.12
            _drive(cam, 0.3)
        h = cam.health()
        assert h["alert"] == "loud", h
        assert h["loss_pct"] >= 5.0, h
        assert cam.state == STATE_OK, "loss must never change the supervisor state"
    finally:
        cam.stop()


def test_a_steady_shortfall_with_no_long_gap_is_still_seen_by_the_ledger() -> None:
    """60 Hz steady where 100 was promised: no gap crosses 2.5 intervals, ledger says ~40 % lost."""
    drv = ScriptedCamera(period_s=1 / 60)
    cam = _cam(lambda: drv, target_fps=100.0, drop_gap_x=2.5, alert_window_s=30.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 6.0)               # ledger needs ≥5 s of span
        h = cam.health()
        assert h["drops_total"] == 0, "1.7-interval gaps are not drop events"
        assert h["loss_pct"] >= 25.0, h
        assert h["alert"] == "loud", h
    finally:
        cam.stop()


def test_a_webcam_at_its_native_29_5_hz_is_not_reported_as_loss() -> None:
    """29.5 Hz steady against a 30 fps target (both Innomakers, every session
    since 2026-08): 1.6 % short of the promise with zero gaps. That is the
    hardware's rate, not loss -- the ledger slack (2.5 % default) swallows it,
    so the wrist panels do not read "warn" on a healthy day (2026-09-11)."""
    drv = ScriptedCamera(period_s=1 / 29.5)
    cam = _cam(lambda: drv, target_fps=30.0, drop_gap_x=2.5, alert_window_s=30.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 6.0)
        h = cam.health()
        assert h["drops_total"] == 0, h
        assert h["loss_pct"] < 1.0, h
        assert h["alert"] == "quiet", h
    finally:
        cam.stop()


def test_the_ledger_slack_does_not_hide_a_real_steady_shortfall() -> None:
    """24 Hz where 30 was promised is 20 % short: slack takes 2.5 off, still loud."""
    drv = ScriptedCamera(period_s=1 / 24)
    cam = _cam(lambda: drv, target_fps=30.0, drop_gap_x=2.5, alert_window_s=30.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 6.0)
        h = cam.health()
        assert h["loss_pct"] >= 10.0, h
        assert h["alert"] == "loud", h
    finally:
        cam.stop()


def test_the_window_forgets_old_drops_and_the_alert_clears() -> None:
    drv = ScriptedCamera()
    cam = _cam(lambda: drv, alert_window_s=1.5, alert_warn_pct=1.0, alert_loud_pct=5.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 0.3)
        drv.hold = 0.2
        _drive(cam, 0.3)
        assert cam.health()["alert"] != "quiet"
        _drive(cam, 2.0)               # window rolls past the burst
        h = cam.health()
        assert h["alert"] == "quiet", h
        assert h["loss_pct"] < 1.0
        assert h["drops_total"] == 1, "totals are forever; only the window forgets"
    finally:
        cam.stop()


def test_a_stall_that_forces_a_reopen_is_loud_via_incidents() -> None:
    """The 2026-09-07 event: silence past read_deadline → reopen → recovery. Loud."""
    drv = ScriptedCamera()
    cam = _cam(lambda: drv, read_deadline_s=0.3, alert_window_s=10.0)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 0.4)
        drv.hold = 1.2                 # well past read_deadline_s
        _drive(cam, 2.5)
        h = cam.health()
        assert h["incidents"] >= 1, h
        assert h["alert"] == "loud", h
        assert "incident" in h["alert_reason"]
        # the reopen hands the pump a fresh driver, so the hole closes before the
        # full hold elapses; the incident, not the gap length, is what makes it loud
        assert h["worst_gap_ms"] >= 500, h
    finally:
        cam.stop()


@pytest.mark.parametrize("key", ["drops_total", "frames_lost_total", "worst_gap_ms",
                                 "loss_pct", "incidents", "freeze_incidents",
                                 "alert", "alert_reason", "alert_window_s"])
def test_the_health_record_carries_every_watchdog_field(key: str) -> None:
    cam = _cam(ScriptedCamera)
    try:
        assert cam.wait_until_open(5.0)
        assert key in cam.health()
    finally:
        cam.stop()
