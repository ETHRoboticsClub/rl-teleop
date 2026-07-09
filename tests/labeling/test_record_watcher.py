"""Tests for the auto-label record watcher: kit.json conversion + the filesystem
flow that writes kit.json on a new episode and labels it once session_meta appears."""
import json
import threading
import time

from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import episode_kit_json, record_watcher


def test_episode_kit_json_shape():
    packets = [
        {"bag_id": 1, "part": "UNN-10126-151", "name": "Flügelmutter", "comp": 5},
        {"part": "UNN-10015-007", "name": "", "comp": 3},          # no bag_id, empty name
    ]
    kit = episode_kit_json(packets)
    assert kit[0] == {"bag_id": 1, "part_no": "UNN-10126-151",
                      "name": "Flügelmutter", "compartment": 5}
    # bag_id defaults to position; empty name → None (label schema expects None, not "")
    assert kit[1] == {"bag_id": 2, "part_no": "UNN-10015-007",
                      "name": None, "compartment": 3}


def test_record_watcher_labels_on_end_not_start(tmp_path, monkeypatch):
    """The real recorder writes session_meta.json at START and re-writes it at END.
    The watcher must write kit.json at start but label only once meta is re-written —
    NOT while recording is still in progress."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-10126-151", "name": "x", "comp": 5}])

    labeled: list[str] = []
    monkeypatch.setattr("robots_realtime.labeling.live_server._run_label_episode",
                        lambda d, arm: labeled.append(d.name))

    stop = threading.Event()
    t = threading.Thread(target=record_watcher,
                         args=(str(tmp_path), labeler, "left", True, stop), daemon=True)
    t.start()

    # rr-session: create dir AND write session_meta.json at START (this is the real order)
    ep = tmp_path / "20260709" / "episode_120000_abcd"
    ep.mkdir(parents=True)
    (ep / "session_meta.json").write_text('{"phase":"start"}')
    time.sleep(1.6)
    kit = json.loads((ep / "kit.json").read_text())    # kit written at start
    assert kit[0]["part_no"] == "UNN-10126-151" and kit[0]["compartment"] == 5
    assert labeled == []                               # NOT labeled mid-recording (the bug)

    # rr-session end_episode: re-write session_meta.json (mtime advances)
    time.sleep(0.6)
    (ep / "session_meta.json").write_text('{"phase":"end","t_end":1.0}')
    time.sleep(1.6)
    assert labeled == ["episode_120000_abcd"]          # labeled exactly once, at end
    stop.set()


def test_record_watcher_idempotent_on_restart(tmp_path, monkeypatch):
    """A finished episode (already has annotations.json / kit.json) must not be
    re-labeled or have its kit clobbered when the watcher (re)starts."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "NEW-00000-000", "comp": 1}])
    ep = tmp_path / "20260709" / "episode_done"
    ep.mkdir(parents=True)
    (ep / "session_meta.json").write_text("{}")
    (ep / "annotations.json").write_text("{}")
    (ep / "kit.json").write_text('[{"bag_id":1,"part_no":"ORIG-11111-111"}]')

    labeled: list[str] = []
    monkeypatch.setattr("robots_realtime.labeling.live_server._run_label_episode",
                        lambda d, arm: labeled.append(d.name))
    stop = threading.Event()
    threading.Thread(target=record_watcher,
                     args=(str(tmp_path), labeler, "left", True, stop), daemon=True).start()
    time.sleep(1.6)
    assert labeled == []                                                  # not re-labeled
    assert "ORIG-11111-111" in (ep / "kit.json").read_text()             # kit not clobbered
    stop.set()
