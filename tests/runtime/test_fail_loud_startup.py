"""Fail-loud startup regression tests — the exact demo failure.

When another process holds a lock on a camera/arm, teleop must abort the whole session
with a diagnostic that names the device and holder — not report itself healthy over a
dead node. These tests drive a real Session (real ZMQ bus + subprocess nodes) with a
node whose setup() fails, and assert the loud-abort behaviour.
"""

from __future__ import annotations

import socket

import pytest

from robots_realtime.runtime.session import Session, SessionStartupError
from tests.runtime._fake_nodes import BusyNode, HangNode, OkNode


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_session(nodes, tmp_path) -> Session:
    return Session(
        nodes=nodes,
        save_root=str(tmp_path),
        pub_port=_free_port(),
        sub_port=_free_port(),
    )


def _all_dead(session: Session) -> bool:
    return all(h._proc is None or not h._proc.is_alive() for h in session._hosts)


def test_critical_busy_node_aborts_session_loudly(tmp_path):
    session = _make_session(
        [OkNode("cam_ok", critical=True), BusyNode("cam_busy", critical=True)],
        tmp_path,
    )
    try:
        with pytest.raises(SessionStartupError) as ei:
            session.start()
        msg = str(ei.value)
        # Names the failing node, the reason, and the holder.
        assert "cam_busy" in msg
        assert "BUSY" in msg
        assert "999" in msg  # holder PID surfaced
        # All hosts torn down — no orphan processes left running.
        assert _all_dead(session)
    finally:
        session.stop()


def test_optional_node_failure_degrades_not_aborts(tmp_path):
    session = _make_session(
        [OkNode("cam_ok", critical=True), BusyNode("cam_opt", critical=False)],
        tmp_path,
    )
    try:
        # Optional failure must NOT raise.
        session.start()
        statuses = {s.name: s for s in session.node_statuses()}
        assert statuses["cam_opt"].alive is False
        assert "BUSY" in statuses["cam_opt"].fatal_reason or "fake0" in statuses["cam_opt"].fatal_reason
        # The healthy node is unaffected.
        assert statuses["cam_ok"].alive is True
    finally:
        session.stop()


def test_all_critical_ok_starts_cleanly(tmp_path):
    session = _make_session(
        [OkNode("a", critical=True), OkNode("b", critical=True)],
        tmp_path,
    )
    try:
        session.start()  # must not raise
        assert not _all_dead(session)  # both alive
    finally:
        session.stop()


def test_mid_session_death_is_detected(tmp_path):
    import time

    session = _make_session(
        [OkNode("a", critical=True), OkNode("watchdog_target", critical=True)],
        tmp_path,
    )
    try:
        session.start()
        # Kill one node's subprocess out from under the running session.
        target = next(h for h in session._hosts if h.node_name == "watchdog_target")
        target._proc.terminate()
        # Monitor polls at ~4 Hz; give it up to ~2s to notice.
        deadline = time.monotonic() + 2.0
        statuses = {}
        while time.monotonic() < deadline:
            statuses = {s.name: s for s in session.node_statuses()}
            if not statuses["watchdog_target"].alive:
                break
            time.sleep(0.05)
        assert statuses["watchdog_target"].alive is False
        assert statuses["watchdog_target"].fatal_reason
        # Critical node death sets the session fatal reason + stop event.
        assert session.fatal_reason is not None
        assert "watchdog_target" in session.fatal_reason
    finally:
        session.stop()


@pytest.mark.slow
def test_hung_setup_aborts_via_timeout(tmp_path):
    session = _make_session([HangNode("cam_hang", critical=True)], tmp_path)
    try:
        with pytest.raises(SessionStartupError) as ei:
            # Short timeout so the test is fast; real default is 30s.
            for host in session._hosts:
                host.send_start = _shorten_timeout(host.send_start)
            session.start()
        assert "timed out" in str(ei.value)
        assert _all_dead(session)
    finally:
        session.stop()


def _shorten_timeout(bound_send_start):
    def wrapper(timeout=1.0):
        return bound_send_start(timeout=timeout)

    return wrapper
