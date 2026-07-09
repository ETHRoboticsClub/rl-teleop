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


def test_record_watcher_writes_kit_and_labels(tmp_path, monkeypatch):
    # a scanned kit in the labeler
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-10126-151", "name": "x", "comp": 5}])

    labeled: list[str] = []
    monkeypatch.setattr("robots_realtime.labeling.live_server._run_label_episode",
                        lambda d, arm: labeled.append(d.name))

    stop = threading.Event()
    t = threading.Thread(target=record_watcher,
                         args=(str(tmp_path), labeler, "left", True, stop), daemon=True)
    t.start()

    # rr-session creates an episode dir (recording starts)
    ep = tmp_path / "20260709" / "episode_120000_abcd"
    ep.mkdir(parents=True)
    time.sleep(1.4)
    kit = json.loads((ep / "kit.json").read_text())   # watcher wrote the scanned kit
    assert kit[0]["part_no"] == "UNN-10126-151" and kit[0]["compartment"] == 5
    assert labeled == []                               # not labeled until session_meta

    # rr-session finishes the episode (session_meta.json appears)
    (ep / "session_meta.json").write_text("{}")
    time.sleep(1.4)
    assert labeled == ["episode_120000_abcd"]          # auto-label fired exactly once
    stop.set()
