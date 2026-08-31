"""``device_model`` fallback: survive a swapped RealSense unit without editing 20 configs.

The failure this guards against is quiet. A RealSense whose serial does not match
the pinned one fails at ``pipe.start()`` with "No device connected" — wording that
names neither the serial nor the camera and is indistinguishable from an unplugged
cable. The fallback must fix that case and ONLY that case: it must never resolve an
ambiguous match, and never let camera_scan open camera_top's device.
"""

from __future__ import annotations

import types

import pytest

from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera

D455_PINNED = "203522250539"   # librealsense serial for this rig's D455
D455_OTHER = "201523063286"    # a different D455 (also this unit's USB descriptor serial)
D435I_PINNED = "241222077246"


class _Info:
    serial_number = "serial"
    name = "name"


class _Dev:
    def __init__(self, serial: str, name: str) -> None:
        self._d = {"serial": serial, "name": name}

    def get_info(self, key: str) -> str:
        return self._d[key]


def _fake_rs(devices):
    ctx = types.SimpleNamespace(query_devices=lambda: devices)
    return types.SimpleNamespace(camera_info=_Info, context=lambda: ctx)


def _resolve(devices, pinned, model):
    cam = RealSenseCamera.__new__(RealSenseCamera)
    cam.device_id, cam.device_model, cam._rs = pinned, model, _fake_rs(devices)
    cam._resolve_device_identity()
    return cam.device_id


D455_A = _Dev(D455_OTHER, "Intel RealSense D455")
D455_B = _Dev(D455_PINNED, "Intel RealSense D455")
D435I = _Dev(D435I_PINNED, "Intel RealSense D435I")


def test_substitutes_when_pinned_serial_absent_and_model_unique():
    """The real rig case: the unit was swapped, exactly one D455 is present."""
    assert _resolve([D455_A, D435I], D455_PINNED, "D455") == D455_OTHER


def test_does_not_substitute_when_pinned_serial_present():
    """A present pin always wins, even alongside another camera of the same model."""
    assert _resolve([D455_A, D455_B], D455_PINNED, "D455") == D455_PINNED


def test_does_not_substitute_when_model_match_is_ambiguous():
    """Two candidate D455s is a question for a human, not a guess."""
    devices = [D455_A, _Dev("999999999999", "Intel RealSense D455")]
    assert _resolve(devices, D455_PINNED, "D455") == D455_PINNED


def test_scan_camera_never_steals_the_top_camera():
    """The D435i's config must not fall back onto a D455. This is the dangerous one:
    silently swapping top and scan would train a policy on the wrong viewpoint."""
    assert _resolve([D455_A], D435I_PINNED, "D435") == D435I_PINNED


def test_no_devices_keeps_the_pin_so_the_error_is_unchanged():
    assert _resolve([], D455_PINNED, "D455") == D455_PINNED


def test_absent_device_model_preserves_legacy_behaviour():
    assert _resolve([D455_A], D455_PINNED, None) == D455_PINNED


def test_enumeration_failure_is_not_fatal():
    """query_devices() can raise while a sick camera is on the bus; resolving is
    best-effort and must leave the pin alone rather than propagate."""
    cam = RealSenseCamera.__new__(RealSenseCamera)
    boom = types.SimpleNamespace(
        camera_info=_Info,
        context=lambda: types.SimpleNamespace(
            query_devices=lambda: (_ for _ in ()).throw(RuntimeError("usb busy"))
        ),
    )
    cam.device_id, cam.device_model, cam._rs = D455_PINNED, "D455", boom
    cam._resolve_device_identity()
    assert cam.device_id == D455_PINNED


def test_warns_loudly_on_substitution(caplog):
    """A silent substitution would be its own trap — the operator must see it."""
    with caplog.at_level("WARNING"):
        _resolve([D455_A, D435I], D455_PINNED, "D455")
    assert D455_OTHER in caplog.text
    assert "safety net" in caplog.text
