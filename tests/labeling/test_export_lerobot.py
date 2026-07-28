"""Coverage for the mcap -> LeRobot grasp exporter.

Every filter and every branch here corresponds to a failure actually present in
the recorded corpus, not a hypothetical one. The counts in the comments are
measured over recordings/ as of 2026-07-27 (30 episodes, 28 with annotations).

The two failures worth the most: action/state swapped, and BGR/RGB swapped.
Neither raises. Both produce a dataset that trains to a plausible loss curve and
a policy that does not work, discovered an hour of GPU time later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from export_lerobot import (  # noqa: E402
    CAMERAS, MIN_WINDOW_FRAMES, N_DOF, build_features, grasp_windows,
    in_workspace, nearest_index, normalize_gripper, usable_grasps,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robots_realtime.labeling import constants as C  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────
def write_episode(tmp: Path, name: str, *, annotations=None, flags=None) -> Path:
    ep = tmp / name
    ep.mkdir(parents=True)
    if annotations is not None:
        (ep / "annotations.json").write_text(json.dumps(annotations))
    if flags is not None:
        (ep / "operator_flags.json").write_text(json.dumps(flags))
    return ep


def ann(outcome="success", attempts=(), **meta):
    return {"episode_meta": {"outcome": outcome, **meta},
            "segments": [], "grasp_attempts": list(attempts),
            "place_events": [], "tracking": [], "flags": []}


def grasp(t, outcome="success", x=0.43):
    """A grasp attempt. Default x sits mid-workspace (corpus mean is 0.432)."""
    return {"bag_id": 1, "attempt": 1, "arm": "left", "t": t, "outcome": outcome,
            "ee_pose": [x, -0.25, 0.12, 0.0, 0.0, 0.0, 1.0]}


# ── episode selection ───────────────────────────────────────────────────────
def test_accepts_a_clean_episode(tmp_path):
    ep = write_episode(tmp_path, "episode_a", annotations=ann(attempts=[grasp(10.0)]))
    good, why = usable_grasps(ep)
    assert why is None and len(good) == 1


def test_rejects_zero_grasp_episode_labelled_success(tmp_path):
    """The 17% case: `outcome` DEFAULTS to success and is not derived from
    whether anything was grasped. 3 of 18 'success' episodes hold no attempts.
    Training on these teaches the policy to stand still."""
    ep = write_episode(tmp_path, "episode_b", annotations=ann(outcome="success"))
    good, why = usable_grasps(ep)
    assert good == [] and "zero grasp_attempts" in why


def test_rejects_aborted(tmp_path):
    ep = write_episode(tmp_path, "episode_c",
                       annotations=ann(outcome="aborted", attempts=[grasp(1.0)]))
    good, why = usable_grasps(ep)
    assert good == [] and "aborted" in why


def test_rejects_missing_annotations(tmp_path):
    ep = write_episode(tmp_path, "episode_d")           # 2 of 30 episodes
    good, why = usable_grasps(ep)
    assert good == [] and "no annotations.json" in why


def test_rejects_operator_x_flag(tmp_path):
    """Operator pressed 'x' during the take. Their judgement outranks the
    labeller's — that is the whole point of the flag."""
    ep = write_episode(tmp_path, "episode_e",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": [{"tag": "x", "t": 5.2}]})
    good, why = usable_grasps(ep)
    assert good == [] and "'x'" in why


def test_other_flags_do_not_reject(tmp_path):
    """'g' (regrasp) and 's' (slow) are annotations, not rejections."""
    ep = write_episode(tmp_path, "episode_f",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": [{"tag": "g", "t": 5.2}, {"tag": "s", "t": 6.0}]})
    good, why = usable_grasps(ep)
    assert why is None and len(good) == 1


def test_keeps_only_successful_attempts_within_an_episode(tmp_path):
    ep = write_episode(tmp_path, "episode_g", annotations=ann(
        attempts=[grasp(1.0, "success"), grasp(5.0, "empty"), grasp(9.0, "success")]))
    good, why = usable_grasps(ep)
    assert why is None and [g["t"] for g in good] == [1.0, 9.0]


def test_rejects_when_no_attempt_succeeded(tmp_path):
    ep = write_episode(tmp_path, "episode_h",
                       annotations=ann(attempts=[grasp(1.0, "empty"), grasp(2.0, "empty")]))
    good, why = usable_grasps(ep)
    assert good == [] and "none with outcome=success" in why


def test_corrupt_annotations_rejected_not_raised(tmp_path):
    ep = tmp_path / "episode_i"
    ep.mkdir()
    (ep / "annotations.json").write_text("{not json")
    good, why = usable_grasps(ep)
    assert good == [] and why is not None


# ── windowing ───────────────────────────────────────────────────────────────
def test_window_is_pre_and_post_around_the_close_instant():
    """`segments` with phase='grasp' are INSTANTS (t_start == t_end) in the real
    data, so the trainable window has to be constructed around them."""
    (lo, hi), = grasp_windows([grasp(100.0)], t0=0.0, t1=200.0, pre_s=3.0, post_s=2.0)
    assert (lo, hi) == (97.0, 102.0)


def test_window_clipped_to_recorded_span():
    """A grasp 1s into the episode cannot have 3s of approach in front of it."""
    (lo, hi), = grasp_windows([grasp(1.0)], t0=0.0, t1=10.0, pre_s=3.0, post_s=2.0)
    assert lo == 0.0 and hi == 3.0


def test_adjacent_windows_never_overlap():
    """Overlap would put identical frames in two training episodes with
    different action labels — silent, and it inflates the dataset while teaching
    the policy contradictions."""
    ws = grasp_windows([grasp(10.0), grasp(12.0)], t0=0.0, t1=30.0,
                       pre_s=3.0, post_s=2.0)
    assert len(ws) == 2
    assert ws[0][1] <= ws[1][0], f"windows overlap: {ws}"


def test_many_grasps_yield_many_windows():
    """One recorded kitting run produces several training episodes — measured
    81 windows from 15 real episodes."""
    ts = [10.0 * i for i in range(1, 6)]
    assert len(grasp_windows([grasp(t) for t in ts], 0.0, 100.0, 3.0, 2.0)) == 5


def test_grasp_outside_recorded_span_is_dropped():
    assert grasp_windows([grasp(500.0)], t0=0.0, t1=100.0, pre_s=3.0, post_s=2.0) == []


# ── signals ─────────────────────────────────────────────────────────────────
def test_gripper_normalised_to_unit_range():
    """Raw limits are auto-calibrated on every boot (observed 5.2218/-0.0235),
    so raw values are not comparable across sessions."""
    out = normalize_gripper(np.array([-0.02, 2.6, 5.22], dtype=np.float32))
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


def test_gripper_normalisation_is_scale_invariant():
    """Two sessions with different calibration must produce the same normalised
    signal for the same physical motion. This is the whole reason to normalise."""
    a = normalize_gripper(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    b = normalize_gripper(np.array([-0.02, 2.6, 5.22], dtype=np.float32))
    assert np.allclose(a, b, atol=1e-6)


def test_degenerate_gripper_range_does_not_divide_by_zero():
    out = normalize_gripper(np.full(5, 3.0, dtype=np.float32))
    assert np.all(np.isfinite(out)) and np.all(out == 1.0)


def test_nearest_index_picks_the_closest_sample():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert list(nearest_index(t, np.array([-5.0, 0.4, 0.6, 2.9, 99.0]))) == [0, 0, 1, 3, 3]


def test_nearest_index_handles_rate_mismatch():
    """200Hz state resampled to a 30Hz grid must stay in order and in range."""
    fast = np.arange(0, 10, 1 / 200)
    grid = np.arange(0, 10, 1 / 30)
    idx = nearest_index(fast, grid)
    assert np.all(np.diff(idx) >= 0)
    assert np.max(np.abs(fast[idx] - grid)) < 1 / 200


# ── schema ──────────────────────────────────────────────────────────────────
def test_features_use_per_camera_resolution():
    """The rig records camera_top at 1280x720 and camera_left at 640x480.
    Assuming one shared resolution makes LeRobotDataset reject every wrist frame
    at add_frame() — caught only when it hit real data."""
    feats = build_features({"camera_top": (720, 1280), "camera_left": (480, 640)})
    assert feats["observation.images.top"]["shape"] == (720, 1280, 3)
    assert feats["observation.images.wrist"]["shape"] == (480, 640, 3)


def test_state_and_action_are_separate_features_of_the_right_width():
    feats = build_features({c: (64, 64) for c in CAMERAS})
    assert feats["observation.state"]["shape"] == (N_DOF,)
    assert feats["action"]["shape"] == (N_DOF,)
    assert feats["observation.state"]["names"][-1] == "gripper"


def test_scan_camera_is_excluded():
    """The scan cam watches the packet mat, not the workspace. It is ~49% of
    every episode's bytes and contributes nothing to a grasp policy."""
    assert "camera_scan" not in CAMERAS
    assert all("scan" not in k for k in build_features({c: (64, 64) for c in CAMERAS}))


def test_min_window_frames_guards_stub_episodes():
    assert MIN_WINDOW_FRAMES >= 10


# ── video ───────────────────────────────────────────────────────────────────
def _write_video(path: Path, colors_bgr, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not vw.isOpened():
        pytest.skip("no mp4 writer available")
    for bgr in colors_bgr:
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :] = bgr
        vw.write(frame)
    vw.release()


def test_camera_stream_returns_rgb_not_bgr(tmp_path):
    """cv2 decodes BGR; LeRobot stores RGB. Swapping them raises nothing, trains
    to a normal-looking loss, and produces a policy that fails on real images.
    Written as pure blue in BGR (255,0,0) — read back it must be RED-channel 0
    and BLUE-channel high."""
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(255, 0, 0)] * 5)            # BGR blue
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(5) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        img = s.frame_at(0)
    finally:
        s.close()
    assert img is not None
    r, g, b = (int(img[..., i].mean()) for i in range(3))
    assert b > 200 and r < 60, f"channels look BGR-ordered: r={r} g={g} b={b}"


def test_camera_stream_reads_forward_and_refuses_backward(tmp_path):
    """Sequential-only by design: cv2 seeking on h264 lands on the nearest
    keyframe and silently returns the wrong frame."""
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(4) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        assert s.frame_at(0) is not None
        assert s.frame_at(2) is not None            # forward is fine
        with pytest.raises(RuntimeError, match="backward"):
            s.frame_at(1)
    finally:
        s.close()


def test_camera_stream_past_end_returns_none(tmp_path):
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(0, 0, 0)] * 3)
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(3) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        assert s.frame_at(99) is None               # must not raise or hang
    finally:
        s.close()


# ── workspace gate ──────────────────────────────────────────────────────────
# Added 2026-07-28. The corpus holds 4 successful grasps at x 0.107..0.178 with
# z 0.162..0.226 against a corpus z mean of 0.120 -- near AND high, i.e. mid-air
# or mislabelled, not table grasps. Sorted x has a 176mm empty band between them
# and the other 77, so the gate is a fixed threshold rather than a moving
# statistical cut. See constants.GRASP_WORKSPACE_X_MIN.

def test_gate_keeps_a_grasp_on_the_mat():
    assert in_workspace(grasp(1.0, x=0.43)) is True


def test_gate_drops_a_grasp_off_the_mat():
    assert in_workspace(grasp(1.0, x=0.15)) is False


def test_gate_boundary_is_inclusive():
    assert in_workspace(grasp(1.0, x=C.GRASP_WORKSPACE_X_MIN)) is True
    assert in_workspace(grasp(1.0, x=C.GRASP_WORKSPACE_X_MIN - 1e-6)) is False


def test_gate_fails_open_on_a_missing_pose():
    """Unknown is not out-of-bounds. A grasp with no ee_pose is a labeller
    problem and must not be silently reclassified as a workspace result."""
    g = grasp(1.0)
    del g["ee_pose"]
    assert in_workspace(g) is True


def test_gate_is_grasp_level_not_episode_level(tmp_path):
    """THE regression this gate exists to avoid getting wrong. Both corpus
    episodes carrying out-of-workspace grasps also carry good ones. Rejecting
    whole episodes would drop 23 usable windows to remove 4 bad ones."""
    ep = write_episode(tmp_path, "episode_mixed", annotations=ann(attempts=[
        grasp(1.0, x=0.43),      # on the mat
        grasp(5.0, x=0.15),      # off the mat
        grasp(9.0, x=0.47),      # on the mat
    ]))
    good, why = usable_grasps(ep)
    assert why is None
    assert [g["t"] for g in good] == [1.0, 9.0]


def test_episode_with_only_off_mat_grasps_is_rejected(tmp_path):
    ep = write_episode(tmp_path, "episode_alloff", annotations=ann(
        attempts=[grasp(1.0, x=0.10), grasp(5.0, x=0.18)]))
    good, why = usable_grasps(ep)
    assert good == []
    assert why is not None and "outside the workspace" in why


def test_gate_can_be_disabled_for_review(tmp_path):
    """review_grasps.py passes workspace_gate=False so it can SHOW what was
    dropped. Nothing that writes training data may pass False."""
    ep = write_episode(tmp_path, "episode_review", annotations=ann(
        attempts=[grasp(1.0, x=0.43), grasp(5.0, x=0.15)]))
    good, why = usable_grasps(ep, workspace_gate=False)
    assert why is None and [g["t"] for g in good] == [1.0, 5.0]
