from __future__ import annotations

import signal
from pathlib import Path

import pytest

from robots_realtime.sensors.cameras import realsense_camera
from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera


class FakeCameraInfo:
    serial_number = "serial_number"


class FakeStream:
    color = "color"
    depth = "depth"
    infrared = "infrared"


class FakeFormat:
    rgb8 = "rgb8"
    z16 = "z16"


class FakeProfile:
    def __init__(self, device: FakeDevice | None = None) -> None:
        self._device = device

    def get_device(self) -> FakeDevice | None:
        return self._device


class FakeConfig:
    def enable_device(self, _device_id: str) -> None:
        pass

    def enable_stream(self, *_args) -> None:
        pass


class FakeFrames:
    def __init__(self, state: dict[str, int | bool]) -> None:
        self._state = state

    def get_color_frame(self):
        return "color" if self._state.get("started") else None

    def get_infrared_frame(self):
        return "infrared" if self._state.get("started") else None


class FakePipeline:
    def __init__(self, state: dict[str, int | bool], device: FakeDevice | None = None) -> None:
        self._state = state
        self._device = device

    def start(self, _cfg: FakeConfig) -> FakeProfile:
        self._state["attempts"] = int(self._state["attempts"]) + 1
        if int(self._state["attempts"]) <= int(self._state["failures"]):
            raise RuntimeError("Device or resource busy")
        self._state["started"] = True
        return FakeProfile(self._device)

    def wait_for_frames(self, timeout_ms: int = 5000) -> FakeFrames:
        if not self._state.get("started"):
            raise RuntimeError("Frame didn't arrive within timeout")
        return FakeFrames(self._state)


class FakeDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.reset_count = 0

    def get_info(self, info: str) -> str:
        if info == FakeCameraInfo.serial_number:
            return self.serial
        return ""

    def hardware_reset(self) -> None:
        self.reset_count += 1


class FakeContext:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self._devices = devices

    def query_devices(self) -> list[FakeDevice]:
        return self._devices


class FakeRs:
    camera_info = FakeCameraInfo
    config = FakeConfig
    format = FakeFormat
    stream = FakeStream

    def __init__(self, devices: list[FakeDevice], state: dict[str, int | bool]) -> None:
        self._devices = devices
        self._state = state

    def context(self) -> FakeContext:
        return FakeContext(self._devices)

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self._state, self._devices[0] if self._devices else None)


def _camera_for_recovery(*, force: bool = False, serial: str = "123") -> RealSenseCamera:
    camera = object.__new__(RealSenseCamera)
    camera.device_id = serial
    camera.force_kill_stale_realsense_owners = force
    camera.stale_owner_grace_s = 0.0
    return camera


def _write_proc_owner(proc_root: Path, pid: int, device_path: Path, *, cmdline: str = "rr-session") -> None:
    proc_dir = proc_root / str(pid)
    fd_dir = proc_dir / "fd"
    fd_dir.mkdir(parents=True)
    (proc_dir / "cmdline").write_text(cmdline.replace(" ", "\0"))
    (proc_dir / "comm").write_text("python")
    fd_path = fd_dir / "7"
    fd_path.symlink_to(device_path)


def _write_usb_device(sys_root: Path, dev_root: Path, serial: str = "123") -> Path:
    usb_dir = sys_root / "1-1"
    usb_dir.mkdir(parents=True)
    (usb_dir / "serial").write_text(serial)
    (usb_dir / "busnum").write_text("1")
    (usb_dir / "devnum").write_text("2")
    device_path = dev_root / "bus" / "usb" / "001" / "002"
    device_path.parent.mkdir(parents=True)
    device_path.touch()
    return device_path


def test_stale_realsense_owner_takeover_is_graceful_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    sys_root = tmp_path / "sys" / "bus" / "usb" / "devices"
    dev_root = tmp_path / "dev"
    device_path = _write_usb_device(sys_root, dev_root)
    _write_proc_owner(proc_root, 42, device_path)

    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(realsense_camera, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(realsense_camera, "_SYS_USB_DEVICES", sys_root)
    monkeypatch.setattr(realsense_camera, "_DEV_ROOT", dev_root)
    monkeypatch.setattr(realsense_camera.os, "getpid", lambda: 999)
    monkeypatch.setattr(realsense_camera.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(realsense_camera.time, "sleep", lambda _seconds: None)

    assert _camera_for_recovery()._take_over_stale_realsense_owners()
    assert kills == [(42, signal.SIGTERM)]


def test_force_kill_flag_escalates_after_graceful_takeover_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    sys_root = tmp_path / "sys" / "bus" / "usb" / "devices"
    dev_root = tmp_path / "dev"
    device_path = _write_usb_device(sys_root, dev_root)
    _write_proc_owner(proc_root, 42, device_path)

    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(realsense_camera, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(realsense_camera, "_SYS_USB_DEVICES", sys_root)
    monkeypatch.setattr(realsense_camera, "_DEV_ROOT", dev_root)
    monkeypatch.setattr(realsense_camera.os, "getpid", lambda: 999)
    monkeypatch.setattr(realsense_camera.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(realsense_camera.time, "sleep", lambda _seconds: None)

    assert _camera_for_recovery(force=True)._take_over_stale_realsense_owners()
    assert kills == [(42, signal.SIGTERM), (42, signal.SIGKILL)]


def test_busy_open_resets_identified_device_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    device = FakeDevice("123")
    state: dict[str, int | bool] = {"attempts": 0, "failures": 2, "started": False}
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._width = 640
    camera._height = 480
    camera._depth_width = 640
    camera._depth_height = 480
    camera._pipeline = None
    camera._profile = None
    camera._align = None
    camera._depth_scale = 0.001
    camera._use_infrared = False
    camera.enable_depth = False
    camera.align_depth = False
    camera.fps = 30

    monkeypatch.setattr(camera, "_device_has_color_sensor", lambda: True)
    monkeypatch.setattr(camera, "_take_over_stale_realsense_owners", lambda: False)
    monkeypatch.setattr(realsense_camera.time, "sleep", lambda _seconds: None)

    camera._open_with_retries(max_retries=2)

    assert device.reset_count == 1
    assert state["started"] is True
    assert camera._pipeline is not None


def test_busy_open_skips_reset_when_auto_device_target_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [FakeDevice("123"), FakeDevice("456")]
    state: dict[str, int | bool] = {"attempts": 0, "failures": 99, "started": False}
    camera = _camera_for_recovery(serial=None)  # type: ignore[arg-type]
    camera.device_id = None
    camera._rs = FakeRs(devices, state)
    camera._width = 640
    camera._height = 480
    camera._depth_width = 640
    camera._depth_height = 480
    camera._pipeline = None
    camera._profile = None
    camera._align = None
    camera._depth_scale = 0.001
    camera._use_infrared = False
    camera.enable_depth = False
    camera.align_depth = False
    camera.fps = 30

    monkeypatch.setattr(camera, "_device_has_color_sensor", lambda: True)
    monkeypatch.setattr(camera, "_take_over_stale_realsense_owners", lambda: False)
    monkeypatch.setattr(realsense_camera.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="busy"):
        camera._open_with_retries(max_retries=1)

    assert [device.reset_count for device in devices] == [0, 0]
