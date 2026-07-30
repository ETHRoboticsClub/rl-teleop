"""Tests for the auto-label record watcher: kit.json conversion + the filesystem
flow that writes kit.json on a new episode and labels it once session_meta appears."""
import json
import os
import threading
import time

from robots_realtime.labeling.detector import Detection
from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import (
    _state_with_detections, episode_kit_json, record_watcher,
)


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


def test_record_watcher_backfills_saved_but_unlabeled(tmp_path, monkeypatch):
    """REGRESSION: an episode saved while the watcher was DOWN (meta + a settled mcap,
    no annotations) must be labeled on the next watcher start. Previously it was stuck
    forever: meta0 initialized to the already-final mtime, so `m > meta0 + 0.5` could
    never fire. This is the silent-skip that lost the label for episode_181804."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-10126-151", "comp": 1}])
    ep = tmp_path / "20260709" / "episode_saved_offline"
    ep.mkdir(parents=True)
    (ep / "session_meta.json").write_text('{"phase":"end"}')
    mcap = ep / "yam_left.mcap"
    mcap.write_bytes(b"\x00" * 100)
    old = time.time() - 120                       # settled: finished writing 2 min ago
    os.utime(mcap, (old, old))

    labeled: list[str] = []
    monkeypatch.setattr("robots_realtime.labeling.live_server._run_label_episode",
                        lambda d, arm: labeled.append(d.name))
    stop = threading.Event()
    threading.Thread(target=record_watcher,
                     args=(str(tmp_path), labeler, "left", True, stop), daemon=True).start()
    time.sleep(1.6)
    assert labeled == ["episode_saved_offline"]   # backfilled on startup, not stuck
    stop.set()


def test_record_watcher_does_not_backfill_active_recording(tmp_path, monkeypatch):
    """The backfill staleness guard must NOT fire for an episode still recording: meta at
    start + a freshly-written (non-settled) mcap. Only a settled mcap means 'saved'."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-10126-151", "comp": 1}])
    ep = tmp_path / "20260709" / "episode_recording"
    ep.mkdir(parents=True)
    (ep / "session_meta.json").write_text('{"phase":"start"}')
    (ep / "yam_left.mcap").write_bytes(b"\x00" * 100)   # fresh mtime = now → not settled

    labeled: list[str] = []
    monkeypatch.setattr("robots_realtime.labeling.live_server._run_label_episode",
                        lambda d, arm: labeled.append(d.name))
    stop = threading.Event()
    threading.Thread(target=record_watcher,
                     args=(str(tmp_path), labeler, "left", True, stop), daemon=True).start()
    time.sleep(1.6)
    assert labeled == []                                # not backfilled mid-recording
    stop.set()


def test_state_no_wrong_box_for_duplicate_parts():
    """REGRESSION: when a kit has more same-part entries than live detections (dups
    picked/occluded/stale), the extra entries get NO box, not a borrowed lst[-1] one.
    A wrong box stacked on the wrong packet is the 'random bounding box' symptom."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-16022-009", "comp": 1},
                  {"bag_id": 2, "part": "UNN-16022-009", "comp": 1},
                  {"bag_id": 3, "part": "UNN-16022-009", "comp": 1}])

    class _OneDetection:
        def current(self):
            return [Detection([10, 10, 20, 20], "UNN-16022-009", 0.9)], [640, 480]

    st = _state_with_detections(labeler, _OneDetection())
    boxes = [p["bbox_px"] for p in st["packets"]]
    assert boxes[0] == [10, 10, 20, 20]               # first entry gets the real detection
    assert boxes[1] is None and boxes[2] is None      # no borrowed wrong box (was lst[-1])


def test_state_matches_suffix_not_position():
    """REGRESSION: two same-MIDDLE packets (10015-007 vs 10015-231) must each get the box
    of their OWN suffix. The old middle-only positional assignment swapped them by screen
    position → operator told the wrong compartment (a kitting error, not cosmetic)."""
    labeler = LiveLabeler()
    labeler.seed([{"bag_id": 1, "part": "UNN-10015-007", "comp": 1},
                  {"bag_id": 2, "part": "UNN-10015-231", "comp": 2}])

    class _TwoDetections:
        def current(self):
            # the -231 packet is detected ABOVE the -007 one, so a naive top-to-bottom
            # positional match would hand entry 0 (-007) the -231 box. Suffix match must not.
            return ([Detection([200, 100, 90, 60], "UNN-10015-231", 1.0),
                     Detection([800, 500, 90, 60], "UNN-10015-007", 0.99)], [1280, 720])

    st = _state_with_detections(labeler, _TwoDetections())
    by_part = {p["part"]: p["bbox_px"] for p in st["packets"]}
    assert by_part["UNN-10015-007"] == [800, 500, 90, 60]   # its own box, not the -231 one
    assert by_part["UNN-10015-231"] == [200, 100, 90, 60]
