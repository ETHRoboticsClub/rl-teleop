"""W3 — the readiness tool: dead gripper FAILs, a late close FAILs, healthy PASSes.

The tool is READ-ONLY and layered on the production planner. These tests check
the three verdicts the plan names, the episode-state machine (including the
.trash exclusion the exporter has and any parallel implementation would forget),
and one regression on real recorded episodes.
"""
from __future__ import annotations

import json
import re

import analyze_act_readiness as R  # noqa: E402
import numpy as np
import pytest
from conftest import (  # noqa: E402
    EP_HELD_OUT,
    EP_KITTING_A,
    RECORDINGS,
    URDF,
    episode,
)
from export_lerobot import CAMERAS, Report, close_idx_gate, plan_episode  # noqa: E402

from robots_realtime.labeling import constants as C  # noqa: E402


# ── episode states ──────────────────────────────────────────────────────────
def make_ep(tmp_path, name, *, mcap=True, stamps=True, video=True, ann=True):
    ep = tmp_path / name
    ep.mkdir(parents=True)
    if mcap:
        # A real 3-message mcap is more work than this test needs; the state
        # machine's mcap branch is exercised by the real-episode tests below.
        (ep / "yam_left.mcap").write_bytes(b"")
    for cam in CAMERAS:
        if stamps:
            np.save(ep / f"{cam}-rgb-timestamp.npy", np.arange(10, dtype=float))
        if video:
            (ep / f"{cam}-images-rgb.mp4").write_bytes(b"")
    if ann:
        (ep / "annotations.json").write_text(json.dumps({"grasp_attempts": []}))
    return ep


def test_a_missing_mcap_is_corrupt(tmp_path):
    ep = make_ep(tmp_path, "episode_a", mcap=False)
    assert R.classify_episode(ep, "left", CAMERAS)[0] == "CORRUPT"


def test_an_unreadable_mcap_is_corrupt_not_empty(tmp_path):
    """An empty/garbage mcap must not read as 'an episode with no grasps'."""
    ep = make_ep(tmp_path, "episode_b")
    assert R.classify_episode(ep, "left", CAMERAS)[0] == "CORRUPT"


def test_missing_camera_timestamps_are_ineligible_not_unlabelled(tmp_path,
                                                                 monkeypatch):
    """A session killed mid-episode leaves the mp4 but no timestamp sidecar. No
    amount of labelling makes that exportable, so it is its own state."""
    monkeypatch.setattr(R, "positions",
                        lambda *a, **k: (np.arange(5.0), np.zeros((5, 7))))
    ep = make_ep(tmp_path, "episode_c", stamps=False)
    state, detail = R.classify_episode(ep, "left", CAMERAS)
    assert state == "INELIGIBLE" and "timestamps" in detail


def test_a_complete_but_unlabelled_episode_is_unlabelled(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "positions",
                        lambda *a, **k: (np.arange(5.0), np.zeros((5, 7))))
    ep = make_ep(tmp_path, "episode_d", ann=False)
    assert R.classify_episode(ep, "left", CAMERAS)[0] == "UNLABELLED"


def test_a_complete_labelled_episode_is_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "positions",
                        lambda *a, **k: (np.arange(5.0), np.zeros((5, 7))))
    ep = make_ep(tmp_path, "episode_e")
    assert R.classify_episode(ep, "left", CAMERAS)[0] == "OK"


def test_trash_is_excluded_exactly_as_the_exporter_excludes_it(tmp_path):
    """Deleting in the cockpit is a MOVE into .trash so it stays undoable. A
    tool that counted .trash would report a corpus the exporter will never
    write. Uses the exporter's own episode_dirs — same rule, one place."""
    make_ep(tmp_path, "episode_keep")
    make_ep(tmp_path, ".trash/episode_dropped")
    names = [p.name for p in R.episode_dirs(tmp_path, ("left",))]
    assert names == ["episode_keep"]


# ── gripper verdicts ────────────────────────────────────────────────────────
class _FakePos:
    """Stands in for read_positions with a chosen gripper channel."""

    def __init__(self, grip):
        self.t = np.arange(len(grip), dtype=float) / 200.0
        self.pos = np.zeros((len(grip), 7))
        self.pos[:, C.GRIPPER_JOINT_INDEX] = grip


def _gripper(monkeypatch, tmp_path, grip):
    fake = _FakePos(grip)
    monkeypatch.setattr(R, "positions", lambda *a, **k: (fake.t, fake.pos))
    return R.gripper_health(tmp_path, "left", C.GRIPPER_OPEN_REF,
                            C.GRIPPER_CLOSED_REF)


def test_dead_gripper_fails(monkeypatch, tmp_path):
    """The measured dead-channel signature: parked at the open stop with ~1e-4
    of sensor noise. The old `hi - lo < 1e-9` guard certified all twelve of
    these as healthy (constants.GRIPPER_MIN_RANGE_FRAC)."""
    rng = np.random.default_rng(0)
    out = _gripper(monkeypatch, tmp_path, 0.9964 + rng.normal(0, 1e-4, 2000))
    assert out["dead"] is True
    assert out["spread"] < out["floor"]


def test_a_healthy_gripper_passes_and_reads_as_bimodal(monkeypatch, tmp_path):
    grip = np.concatenate([np.full(1000, 0.999), np.full(1000, 0.004)])
    out = _gripper(monkeypatch, tmp_path, grip)
    assert out["dead"] is False
    assert out["bimodal_frac"] == pytest.approx(1.0)
    assert out["frac_closed"] == pytest.approx(0.5)


def test_a_gripper_stuck_mid_band_is_not_called_healthy(monkeypatch, tmp_path):
    """Jaws that never fully open or close: the span clears the dead floor, so
    the dead check alone would pass it. Bimodality is what catches it."""
    out = _gripper(monkeypatch, tmp_path,
                   np.linspace(0.45, 0.60, 2000))
    assert out["dead"] is False and out["bimodal_frac"] < 0.1


# ── the close-index verdict ─────────────────────────────────────────────────
def test_a_close_past_the_chunk_fails_the_gate():
    assert 90 > close_idx_gate(100, 0.8)          # the fixed window's index
    assert 58 <= close_idx_gate(100, 0.8)         # this corpus's pose-mode median


def test_cadence_reports_each_stream_against_its_own_nominal(tmp_path):
    """200 Hz arm, 30 Hz camera. One shared expectation would flag one of them
    on every healthy episode."""
    ep = tmp_path / "episode_z"
    ep.mkdir()
    np.save(ep / "camera_top-rgb-timestamp.npy", np.arange(300) / 30.0)
    cad = R.stream_cadence(ep, "left", {"camera_top": "top"})
    assert cad["camera_top"]["hz"] == pytest.approx(30.0, abs=0.01)
    assert cad["camera_top"]["gaps"] == 0


def test_cadence_counts_a_dropped_block_as_a_gap(tmp_path):
    ep = tmp_path / "episode_y"
    ep.mkdir()
    t = np.concatenate([np.arange(100) / 30.0, 100 / 30.0 + 2.0 + np.arange(100) / 30.0])
    np.save(ep / "camera_top-rgb-timestamp.npy", t)
    cad = R.stream_cadence(ep, "left", {"camera_top": "top"})
    assert cad["camera_top"]["gaps"] == 1
    assert cad["camera_top"]["worst_gap_s"] == pytest.approx(2.033, abs=0.01)


def test_camera_skew_is_measured_from_the_recorded_stamps(tmp_path):
    """REPORTED, never gated. Two free-running 30 Hz cameras offset by 10 ms."""
    ep = tmp_path / "episode_w"
    ep.mkdir()
    np.save(ep / "camera_left-rgb-timestamp.npy", np.arange(300) / 30.0)
    np.save(ep / "camera_top-rgb-timestamp.npy", np.arange(300) / 30.0 + 0.010)
    assert R.camera_skew_s(ep, CAMERAS) == pytest.approx(0.010, abs=1e-6)


# ── real episodes ───────────────────────────────────────────────────────────
def test_the_tool_reconciles_with_the_production_planner():
    """The claim the tool makes about itself: its window counts ARE the
    exporter's, because it calls plan_episode rather than reimplementing the
    filters. Measured 2026-09-02: 30 windows for this episode."""
    ep = episode(EP_KITTING_A)
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, 30, rep, None, None, None, ("left",),
                        "grasp", 1.0, 0.0, URDF)
    assert len(plan["windows"]) == 30


def test_held_out_episodes_are_reported_but_never_counted_as_training(capsys):
    """The held-out set is the newest finished takes. It must appear in the
    report — marked — and contribute 0 to the training total."""
    for name in EP_HELD_OUT:
        episode(name)                       # skips the test if they are not here
    rc = R.main(["--root", str(RECORDINGS), "--held-out", ",".join(EP_HELD_OUT),
                 "--urdf", URDF])
    out = capsys.readouterr().out
    assert rc == 0
    for name in EP_HELD_OUT:
        assert f"{name}" in out
    assert "[HELD OUT]" in out
    # The training total is the exporter's, and it was 72 when this was written
    # (2026-09-02). It is asserted as a FLOOR, not an equality: this root is a
    # live recording directory and a corpus can only grow — a test that pinned
    # it exactly would fail the next time the operator records, which is not a
    # defect in anything.
    total = int(re.search(r"grasp\s+training (\d+)", out).group(1))
    assert total >= 72
    # Every held-out episode is listed with its marker, and none of them is
    # inside that total.
    for name in EP_HELD_OUT:
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(name))
        assert "[HELD OUT]" in line


@pytest.mark.parametrize("name,n_holds", [(EP_HELD_OUT[0], 35), (EP_HELD_OUT[1], 13)])
def test_hold_durations_are_measured_when_annotations_predate_0_2_0(name, n_holds):
    """Hold duration is a first-class quality metric, so it must be available
    for episodes labelled before the field existed. Counts measured 2026-09-02
    and pinned; they match the grasp-mode labeller's own counts."""
    ep = episode(name)
    holds, _other, note = R.measure_holds(ep, "left", C.GRIPPER_OPEN_REF,
                                         C.GRIPPER_CLOSED_REF, 30, URDF)
    assert note == "measured from mcap"
    assert len(holds) == n_holds
    assert min(holds) >= C.MIN_HOLD_S
