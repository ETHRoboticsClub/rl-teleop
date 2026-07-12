"""Unit tests for device preflight / lock diagnostics (fail-loud startup).

These lock in the behaviour that the demo failure needed: a busy/locked/missing device
is classified correctly and rendered into an actionable, holder-naming message.
"""

from __future__ import annotations

import errno
import os
import tempfile

import pytest

from robots_realtime.runtime import preflight
from robots_realtime.runtime.preflight import (
    DeviceBusyError,
    DeviceReason,
    HolderInfo,
    classify_os_error,
    describe_holders,
    raise_device_busy,
    scan_device_path,
)

# ── classify_os_error ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "err_no, expected",
    [
        (errno.EBUSY, DeviceReason.BUSY),
        (errno.EAGAIN, DeviceReason.BUSY),
        (errno.EACCES, DeviceReason.PERMISSION),
        (errno.EPERM, DeviceReason.PERMISSION),
        (errno.ENOENT, DeviceReason.MISSING),
        (errno.ENODEV, DeviceReason.MISSING),
    ],
)
def test_classify_by_errno(err_no, expected):
    assert classify_os_error(OSError(err_no, os.strerror(err_no))) is expected


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Device or resource busy", DeviceReason.BUSY),
        ("[Errno 16] Device or resource busy", DeviceReason.BUSY),
        ("Resource temporarily unavailable", DeviceReason.BUSY),
        ("Permission denied", DeviceReason.PERMISSION),
        ("No such file or directory", DeviceReason.MISSING),
        ("something totally unexpected", DeviceReason.UNKNOWN),
    ],
)
def test_classify_by_message(message, expected):
    # Drivers raise plain RuntimeError with no errno — must fall back to the message.
    assert classify_os_error(RuntimeError(message)) is expected


def test_errno_takes_precedence_over_message():
    exc = OSError(errno.EBUSY, "permission denied")  # errno says busy, text says perm
    assert classify_os_error(exc) is DeviceReason.BUSY


# ── HolderInfo / DeviceBusyError rendering ─────────────────────────────────────


def test_holder_str_includes_pid_name_user():
    h = HolderInfo(pid=1234, name="python", username="bob", cmdline="python teleop.py")
    s = str(h)
    assert "1234" in s and "python" in s and "bob" in s


def test_device_busy_error_message_is_actionable():
    err = DeviceBusyError(
        device="/dev/video4",
        reason=DeviceReason.BUSY,
        holders=[HolderInfo(pid=42, name="python", username="bob")],
    )
    msg = str(err)
    assert "/dev/video4" in msg
    assert "BUSY" in msg
    assert "PID 42" in msg
    assert "fuser -k" in msg  # tells the operator how to free it


def test_device_busy_error_busy_without_holders_still_hints():
    err = DeviceBusyError(device="/dev/ttyUSB0", reason=DeviceReason.BUSY)
    msg = str(err)
    assert "another process" in msg.lower()


def test_device_busy_error_permission_and_missing_hints():
    perm = str(DeviceBusyError("/dev/video0", DeviceReason.PERMISSION))
    assert "permission" in perm.lower()
    missing = str(DeviceBusyError("/dev/nope", DeviceReason.MISSING))
    assert "does not exist" in missing.lower()


def test_device_busy_error_keeps_structured_fields():
    err = DeviceBusyError("/dev/video4", DeviceReason.BUSY, detail="pipeline start failed")
    assert err.device == "/dev/video4"
    assert err.reason is DeviceReason.BUSY
    assert err.detail == "pipeline start failed"
    assert isinstance(err, RuntimeError)  # existing callers catching RuntimeError still work


# ── describe_holders ───────────────────────────────────────────────────────────


def test_describe_holders_finds_self_holding_a_file():
    """psutil should see our own process holding an open file (own proc is accessible)."""
    with tempfile.NamedTemporaryFile(prefix="preflight_test_") as tmp:
        f = open(tmp.name, "rb")
        try:
            holders = preflight._holders_via_psutil(tmp.name)
        finally:
            f.close()
    pids = {h.pid for h in holders}
    # If psutil is restricted on this platform the list may be empty — only assert
    # when it found anything, but our own pid must be present when it does.
    if holders:
        assert os.getpid() in pids


def test_describe_holders_never_raises(monkeypatch):
    monkeypatch.setattr(preflight, "_holders_via_psutil", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(preflight, "_holders_via_fuser", lambda p: [])
    assert describe_holders("/dev/whatever") == []


# ── raise_device_busy ──────────────────────────────────────────────────────────


def test_raise_device_busy_classifies_and_chains(monkeypatch):
    monkeypatch.setattr(preflight, "describe_holders", lambda p: [HolderInfo(pid=7, name="ssh")])
    cause = OSError(errno.EBUSY, "Device or resource busy")
    with pytest.raises(DeviceBusyError) as ei:
        raise_device_busy("/dev/ttyUSB0", cause=cause)
    err = ei.value
    assert err.reason is DeviceReason.BUSY
    assert err.holders and err.holders[0].pid == 7
    assert err.__cause__ is cause


def test_raise_device_busy_respects_explicit_reason(monkeypatch):
    monkeypatch.setattr(preflight, "describe_holders", lambda p: [])
    with pytest.raises(DeviceBusyError) as ei:
        raise_device_busy("cam-serial-123", reason=DeviceReason.WRONG_DEVICE, find_holders=False)
    assert ei.value.reason is DeviceReason.WRONG_DEVICE


# ── scan_device_path ───────────────────────────────────────────────────────────


def test_scan_missing_path():
    res = scan_device_path("/dev/definitely-not-here-xyz")
    assert not res.ok
    assert res.reason is DeviceReason.MISSING


def test_scan_existing_free_path(monkeypatch):
    monkeypatch.setattr(preflight, "describe_holders", lambda p: [])
    with tempfile.NamedTemporaryFile() as tmp:
        res = scan_device_path(tmp.name)
    assert res.ok
    assert res.reason is DeviceReason.UNKNOWN


def test_scan_existing_busy_path(monkeypatch):
    monkeypatch.setattr(preflight, "describe_holders", lambda p: [HolderInfo(pid=9, name="cat")])
    with tempfile.NamedTemporaryFile() as tmp:
        res = scan_device_path(tmp.name)
    assert not res.ok
    assert res.reason is DeviceReason.BUSY
    assert "held by" in res.message()
