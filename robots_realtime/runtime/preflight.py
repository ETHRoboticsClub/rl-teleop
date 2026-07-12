"""Device preflight and lock diagnostics for fail-loud hardware acquisition.

When a camera, arm, or serial port cannot be acquired at startup, teleop must fail
*loudly* with a diagnostic that names the device, *why* it failed (busy / permission /
missing / wrong device), and — when it is busy — *which process* is holding it. The old
behaviour was to swallow the error or retry silently, which is exactly how a live demo
died: another SSH session held a lock, a node crashed, and nothing surfaced it.

This module is dependency-light on purpose (only the stdlib plus an optional, lazily
imported ``psutil``) so it can be imported from driver code and the session startup path
without pulling in ``pyrealsense2`` / ``zmq`` / hardware SDKs.

Public surface:
  - ``DeviceReason``        — coarse classification of an acquisition failure.
  - ``classify_os_error``   — map a caught exception to a ``DeviceReason``.
  - ``HolderInfo`` / ``describe_holders`` — who holds a ``/dev`` path.
  - ``DeviceBusyError``     — rich exception carrying device + reason + holders.
  - ``raise_device_busy``   — classify, find holders, and raise in one call.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceReason(str, Enum):
    """Why a device could not be acquired.

    Inherits ``str`` so it formats and compares cleanly in log lines and tests.
    """

    BUSY = "busy"                 # held by another process (EBUSY / "resource busy")
    PERMISSION = "permission"     # EACCES / "permission denied"
    MISSING = "missing"           # ENOENT / device path does not exist
    WRONG_DEVICE = "wrong_device"  # opened, but not the expected serial / identity
    UNKNOWN = "unknown"           # could not be classified


# Substrings that different hardware SDKs use to signal "someone else has it".
_BUSY_MARKERS = (
    "busy",
    "resource temporarily unavailable",
    "device or resource busy",
    "already in use",
    "in use by another",
    "could not claim interface",
    "failed to set power state",  # realsense when another proc holds the device
)
_PERMISSION_MARKERS = (
    "permission denied",
    "access denied",
    "operation not permitted",
    "not permitted",
)
_MISSING_MARKERS = (
    "no such file",
    "no such device",
    "cannot find",
    "not found",
    "no device connected",
)


def classify_os_error(exc: BaseException) -> DeviceReason:
    """Best-effort classification of an acquisition failure.

    Looks at ``errno`` first (most reliable), then falls back to substring matching on
    the message, which is how camera/motor SDKs surface these conditions since they
    raise plain ``RuntimeError``/``Exception`` rather than ``OSError``.
    """
    err_no = getattr(exc, "errno", None)
    if err_no is not None:
        if err_no in (errno.EBUSY, errno.EAGAIN):
            return DeviceReason.BUSY
        if err_no in (errno.EACCES, errno.EPERM):
            return DeviceReason.PERMISSION
        if err_no in (errno.ENOENT, errno.ENODEV, errno.ENXIO):
            return DeviceReason.MISSING

    message = str(exc).lower()
    # Order matters: "permission" and "missing" are more specific than "busy", and a
    # message can contain multiple markers.
    if any(m in message for m in _PERMISSION_MARKERS):
        return DeviceReason.PERMISSION
    if any(m in message for m in _MISSING_MARKERS):
        return DeviceReason.MISSING
    if any(m in message for m in _BUSY_MARKERS):
        return DeviceReason.BUSY
    return DeviceReason.UNKNOWN


@dataclass(frozen=True)
class HolderInfo:
    """A process holding a device handle."""

    pid: int
    name: str = ""
    username: str = ""
    cmdline: str = ""

    def __str__(self) -> str:
        who = self.name or "?"
        parts = [f"PID {self.pid} ({who}"]
        if self.username:
            parts.append(f", user={self.username}")
        parts.append(")")
        base = "".join(parts)
        if self.cmdline and self.cmdline != self.name:
            base += f": {self.cmdline}"
        return base


def _resolve(dev_path: str) -> str:
    try:
        return str(Path(dev_path).resolve())
    except Exception:
        return dev_path


def _holders_via_psutil(dev_path: str) -> list[HolderInfo]:
    """Find holders by scanning open files across processes. Requires ``psutil``.

    Returns ``[]`` (never raises) if psutil is unavailable or every candidate process
    denies access — the caller degrades gracefully.
    """
    try:
        import psutil  # lazily imported; optional dependency
    except Exception:
        return []

    target = _resolve(dev_path)
    holders: list[HolderInfo] = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            for of in proc.open_files():
                if _resolve(of.path) == target:
                    info = proc.info
                    try:
                        cmdline = " ".join(proc.cmdline())[:200]
                    except Exception:
                        cmdline = ""
                    holders.append(
                        HolderInfo(
                            pid=int(info.get("pid", proc.pid)),
                            name=info.get("name") or "",
                            username=info.get("username") or "",
                            cmdline=cmdline,
                        )
                    )
                    break
        except Exception:
            # AccessDenied / NoSuchProcess / ZombieProcess — skip this process.
            continue
    return holders


def _holders_via_fuser(dev_path: str) -> list[HolderInfo]:
    """Fallback holder lookup via ``fuser`` (Linux). Returns ``[]`` on any failure."""
    fuser = shutil.which("fuser")
    if not fuser:
        return []
    try:
        proc = subprocess.run(
            [fuser, dev_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return []
    holders: list[HolderInfo] = []
    for token in proc.stdout.split():
        pid_str = "".join(ch for ch in token if ch.isdigit())
        if not pid_str:
            continue
        pid = int(pid_str)
        holders.append(HolderInfo(pid=pid, name=_proc_name(pid)))
    return holders


def _proc_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:
        return ""


def describe_holders(dev_path: str) -> list[HolderInfo]:
    """Return the processes currently holding ``dev_path`` (best effort, never raises).

    Tries ``psutil`` first, then ``fuser``. An empty list means "nobody found" *or*
    "could not determine" — callers must not treat empty as proof the device is free.
    """
    try:
        holders = _holders_via_psutil(dev_path)
        if holders:
            return holders
        return _holders_via_fuser(dev_path)
    except Exception as exc:  # diagnostics must never crash the caller
        logger.debug("describe_holders(%s) failed: %s", dev_path, exc)
        return []


class DeviceBusyError(RuntimeError):
    """A hardware device could not be acquired at startup.

    Carries structured context (device, reason, holders) and renders an actionable,
    operator-facing message. Raised in place of the bare ``RuntimeError`` that drivers
    used to throw, so the failure names the culprit instead of looking like a generic
    "device missing".
    """

    def __init__(
        self,
        device: str,
        reason: DeviceReason = DeviceReason.UNKNOWN,
        holders: list[HolderInfo] | None = None,
        detail: str = "",
    ) -> None:
        self.device = device
        self.reason = DeviceReason(reason)
        self.holders = holders or []
        self.detail = detail
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"Device {self.device!r} could not be acquired ({self.reason.value.upper()})"
        parts = [head]
        if self.detail:
            parts.append(f"— {self.detail}")
        if self.holders:
            who = "; ".join(str(h) for h in self.holders)
            parts.append(f"\n  Held by: {who}")
            # Prefer the resolved real path in the suggested fix so a symlink still works.
            parts.append(f"\n  Free it with:  fuser -k {_resolve(self.device)}")
        elif self.reason is DeviceReason.BUSY:
            parts.append(
                "\n  Another process holds it (holder PID could not be identified — "
                "check other SSH sessions / `fuser` / `lsof`)."
            )
        elif self.reason is DeviceReason.PERMISSION:
            parts.append("\n  Check device permissions / user groups (e.g. dialout, video).")
        elif self.reason is DeviceReason.MISSING:
            parts.append("\n  The device path does not exist — check it is plugged in / the symlink.")
        return " ".join(parts)


def raise_device_busy(
    device: str,
    cause: BaseException | None = None,
    reason: DeviceReason | None = None,
    detail: str = "",
    find_holders: bool = True,
) -> "DeviceBusyError":
    """Build and raise a :class:`DeviceBusyError` from a caught driver exception.

    Classifies ``cause`` (unless ``reason`` is given), looks up holders for a busy
    device, and raises — chaining the original exception. Returns for typing symmetry;
    it always raises.
    """
    resolved_reason = reason or (classify_os_error(cause) if cause is not None else DeviceReason.UNKNOWN)
    holders: list[HolderInfo] = []
    if find_holders and resolved_reason in (DeviceReason.BUSY, DeviceReason.PERMISSION):
        holders = describe_holders(device)
    if not detail and cause is not None:
        detail = str(cause)
    err = DeviceBusyError(device=device, reason=resolved_reason, holders=holders, detail=detail)
    raise err from cause


@dataclass
class PreflightResult:
    """Outcome of the cheap, non-destructive device scan run before nodes spawn."""

    device: str
    ok: bool
    reason: DeviceReason = DeviceReason.UNKNOWN
    holders: list[HolderInfo] = field(default_factory=list)
    detail: str = ""

    def message(self) -> str:
        if self.ok:
            return f"{self.device}: OK"
        line = f"{self.device}: {self.reason.value.upper()}"
        if self.holders:
            line += " — held by " + "; ".join(str(h) for h in self.holders)
        elif self.detail:
            line += f" — {self.detail}"
        return line


def scan_device_path(dev_path: str, require_free: bool = True) -> PreflightResult:
    """Non-destructively check whether ``dev_path`` exists and (optionally) is free.

    Does not open the device (that is the authoritative check done later by the driver);
    this exists so the operator sees *every* conflict at once before launch instead of
    one crash at a time.
    """
    resolved = _resolve(dev_path)
    if not os.path.exists(resolved):
        return PreflightResult(
            device=dev_path,
            ok=False,
            reason=DeviceReason.MISSING,
            detail="path does not exist",
        )
    holders = describe_holders(dev_path) if require_free else []
    if require_free and holders:
        return PreflightResult(
            device=dev_path,
            ok=False,
            reason=DeviceReason.BUSY,
            holders=holders,
        )
    return PreflightResult(device=dev_path, ok=True)
