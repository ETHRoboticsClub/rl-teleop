"""Keyboard dispatch for the recording TUI: arrow keys + quality-flag keys."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from robots_realtime.runtime.session import Session
from robots_realtime.runtime.tui import (
    _FLAG_KEYS,
    _advance,
    _default_instruction,
    _rerecord,
)


class _Stub:
    def __init__(self, recording: bool, mappings: dict):
        self.is_recording = recording
        self.instruction_mappings = mappings
        self.calls: list = []

    def start_episode(self):
        self.calls.append(("start",))

    def end_episode(self, save=True, instruction=None):
        self.calls.append(("end", save, instruction))

    def toggle_recording(self):
        self.calls.append(("toggle",))


def _wait(stub, n=1, timeout=2.0):
    """_advance/_rerecord dispatch on a daemon thread; wait for it to land."""
    end = time.time() + timeout
    while time.time() < end and len(stub.calls) < n:
        time.sleep(0.01)


def test_default_instruction_prefers_key_1():
    assert _default_instruction(_Stub(False, {"0": "play", "1": "kit", "2": "other"})) == "kit"
    assert _default_instruction(_Stub(False, {"0": "play", "3": "z"})) == "z"
    assert _default_instruction(_Stub(False, {})) is None


def test_right_arrow_idle_starts_episode():
    s = _Stub(recording=False, mappings={"1": "kit"})
    _advance(s, threading.Lock())
    _wait(s)
    assert s.calls == [("start",)]


def test_right_arrow_recording_saves_with_default_instruction():
    s = _Stub(recording=True, mappings={"1": "kit"})
    _advance(s, threading.Lock())
    _wait(s)
    assert s.calls == [("end", True, "kit")]


def test_left_arrow_discards_and_restarts():
    s = _Stub(recording=True, mappings={"1": "kit"})
    _rerecord(s, threading.Lock())
    _wait(s, n=2)
    assert s.calls == [("end", False, None), ("start",)]


def test_flag_keys_mapping():
    assert _FLAG_KEYS == {"g": "re_grasp", "x": "bad", "s": "slow"}


def test_flag_episode_writes_sidecar(tmp_path):
    obj = SimpleNamespace(_recording_lock=threading.Lock(), _episode_dir=tmp_path)
    assert Session.flag_episode(obj, "re_grasp") is True
    assert Session.flag_episode(obj, "slow") is True
    data = json.loads((tmp_path / "operator_flags.json").read_text())
    assert [f["tag"] for f in data["flags"]] == ["re_grasp", "slow"]


def test_flag_episode_noop_when_not_recording():
    obj = SimpleNamespace(_recording_lock=threading.Lock(), _episode_dir=None)
    assert Session.flag_episode(obj, "bad") is False
