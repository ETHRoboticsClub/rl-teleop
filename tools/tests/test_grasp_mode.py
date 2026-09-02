"""W1 — grasp-task labelling, and the pin that says kitting did not move.

The bug: ``MIN_TRANSPORT_M`` deletes a grasp that never travelled 10 cm. For a
kitting demo that is the right call (a close that re-opens where it closed is a
fumble at the pick, not a placement). For an approach+grasp+lift demo, which
never transports anything, it deletes the entire episode — measured on
episode_143533_ee94747f: four real grasps in, zero grasp_attempts and eight
flags out.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import EP_GRASP_ONLY, EP_HELD_OUT, URDF, episode  # noqa: E402

from robots_realtime.labeling import constants as C  # noqa: E402
from robots_realtime.labeling.fuse import GraspCandidate, build_annotations  # noqa: E402
from robots_realtime.labeling.label_episode import label_episode_dir  # noqa: E402
from robots_realtime.labeling.schema import Annotations, EpisodeMeta  # noqa: E402


def cand(t_close, t_open, *, dx=0.0, outcome="success", lifted=True, hold_norm=0.3):
    """A grasp that releases ``dx`` metres from where it closed."""
    return GraspCandidate(
        t_close=t_close, t_open=t_open, outcome=outcome, lifted=lifted,
        grasp_pose=[0.43, -0.25, 0.10, 1.0, 0.0, 0.0, 0.0],
        release_pose=[0.43 + dx, -0.25, 0.15, 1.0, 0.0, 0.0, 0.0],
        hold_norm=hold_norm)


# ── the deletion, and its fix ───────────────────────────────────────────────
def test_grasp_with_no_transport_is_deleted_by_the_kitting_task():
    """The live bug, pinned as behaviour so it cannot be 'fixed' by accident in
    the kitting path: a kitting demo really does require the transport."""
    ann = build_annotations("ep", "left", 0.0, 30.0,
                            [cand(10.0, 14.0, dx=0.01)],
                            min_transport_m=C.MIN_TRANSPORT_M,
                            task=C.TASK_KITTING)
    assert ann.grasp_attempts == []
    kinds = {f.kind for f in ann.flags}
    assert "no_transport" in kinds and "unplaced_grasp" in kinds


def test_grasp_with_no_transport_is_kept_by_the_grasp_task():
    ann = build_annotations("ep", "left", 0.0, 30.0,
                            [cand(10.0, 14.0, dx=0.01)],
                            min_transport_m=C.MIN_TRANSPORT_M,   # ignored in grasp mode
                            task=C.TASK_GRASP)
    assert len(ann.grasp_attempts) == 1
    g = ann.grasp_attempts[0]
    assert g.outcome == "success"
    assert g.hold_s == pytest.approx(4.0)
    assert g.lifted is True
    assert ann.place_events == []          # a grasp demo places nothing
    assert {f.kind for f in ann.flags} == set()


def test_a_close_that_never_lifted_is_kept_but_not_called_success():
    """Hold + lift is the whole outcome rule in grasp mode. A close with no lift
    is an adjustment; it is recorded (never silently dropped) with an outcome
    every existing consumer's `outcome == "success"` filter excludes."""
    ann = build_annotations("ep", "left", 0.0, 30.0,
                            [cand(10.0, 14.0, dx=0.01, lifted=False)],
                            task=C.TASK_GRASP)
    assert [g.outcome for g in ann.grasp_attempts] == ["no_lift"]
    assert [f.kind for f in ann.flags] == ["no_lift"]
    assert ann.episode_meta.outcome == "aborted"   # nothing was actually grasped


def test_a_grasp_still_shut_at_the_episode_end_holds_to_the_end():
    """t_open is None. The hold is at least (t_end - t_close), never zero."""
    ann = build_annotations("ep", "left", 0.0, 30.0, [cand(25.0, None)],
                            task=C.TASK_GRASP)
    g = ann.grasp_attempts[0]
    assert g.t_open is None and g.hold_s == pytest.approx(5.0)


def test_grasp_mode_keeps_every_attempt_in_time_order():
    ann = build_annotations("ep", "left", 0.0, 60.0,
                            [cand(20.0, 22.0), cand(10.0, 13.0),
                             cand(30.0, 31.0, outcome="empty", lifted=False)],
                            task=C.TASK_GRASP)
    assert [g.t for g in ann.grasp_attempts] == [10.0, 20.0, 30.0]
    assert [g.bag_id for g in ann.grasp_attempts] == [1, 2, 3]
    assert ann.grasp_attempts[-1].outcome == "empty"


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError):
        build_annotations("ep", "left", 0.0, 1.0, [], task="pickplace")


# ── the pin: kitting is untouched ───────────────────────────────────────────
def test_default_task_is_kitting_and_its_output_is_unchanged():
    """DEFAULT-BEHAVIOUR PIN. Same candidates, same kitting output — attempt by
    attempt, place by place, flag by flag — whether the task is defaulted or
    named. If grasp mode ever leaks into the default path this fails."""
    cands = [cand(10.0, 14.0, dx=0.30),                       # a real placement
             cand(20.0, 24.0, dx=0.01),                       # a fumble at the pick
             cand(30.0, 34.0, dx=0.25, outcome="slip", lifted=True)]
    default = build_annotations("ep", "left", 0.0, 60.0, cands,
                                min_transport_m=C.MIN_TRANSPORT_M)
    named = build_annotations("ep", "left", 0.0, 60.0, cands,
                              min_transport_m=C.MIN_TRANSPORT_M,
                              task=C.TASK_KITTING)
    assert default.to_dict() == named.to_dict()

    # And the kitting semantics themselves, spelled out: one placement, the
    # fumble flagged and unplaced, the slip pending.
    assert [(g.bag_id, g.attempt, g.outcome) for g in default.grasp_attempts] \
        == [(1, 1, "success")]
    assert len(default.place_events) == 1
    assert [f.kind for f in default.flags] == ["no_transport", "unplaced_grasp",
                                               "unplaced_grasp"]


def test_hold_metadata_is_additive_and_does_not_change_kitting_classification():
    """hold_s/t_open/lifted are recorded for the kitting task too. They are
    metadata: nothing above reads them, so no kitting verdict can move."""
    ann = build_annotations("ep", "left", 0.0, 60.0, [cand(10.0, 14.0, dx=0.30)],
                            min_transport_m=C.MIN_TRANSPORT_M)
    g = ann.grasp_attempts[0]
    assert (g.hold_s, g.t_open, g.lifted) == (pytest.approx(4.0), 14.0, True)
    assert g.outcome == "success" and len(ann.place_events) == 1


# ── atomic writes ───────────────────────────────────────────────────────────
def _ann() -> Annotations:
    return Annotations(episode_meta=EpisodeMeta(episode_id="ep", arm="left"))


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "annotations.json"
    _ann().save(p)
    assert json.loads(p.read_text())["episode_meta"]["episode_id"] == "ep"
    assert [f.name for f in tmp_path.iterdir()] == ["annotations.json"]


def test_a_failed_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """The reason for tmp+rename. write_text truncates first, so a crash — or a
    reader arriving mid-write — sees an empty annotations.json, which is
    indistinguishable from 'this episode has no grasps'."""
    p = tmp_path / "annotations.json"
    _ann().save(p)
    before = p.read_text()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        _ann().save(p)
    assert p.read_text() == before                      # untouched, not truncated
    assert list(tmp_path.iterdir()) == [p]              # temp file cleaned up


def test_save_replaces_rather_than_truncating(tmp_path):
    """A reader holding the old inode keeps reading a WHOLE file. That is the
    property os.replace buys and write_text does not."""
    p = tmp_path / "annotations.json"
    _ann().save(p)
    with open(p) as held:
        first = held.read()
        big = _ann()
        big.episode_meta.episode_id = "ep2"
        big.save(p)
        held.seek(0)
        assert held.read() == first                     # the open handle is stable
    assert json.loads(p.read_text())["episode_meta"]["episode_id"] == "ep2"


# ── one source of truth for the gripper refs ────────────────────────────────
def test_qa_label_uses_the_shared_gripper_refs():
    import qa_label
    assert (qa_label.OPEN_REF, qa_label.CLOSED_REF) == (C.GRIPPER_OPEN_REF,
                                                        C.GRIPPER_CLOSED_REF)


def test_live_server_ref_defaults_match_the_constants():
    """Pinned by TEXT, on purpose: importing live_server would pull in the whole
    live stack, and this file is read while a session may be running. The
    duplication is the thing being watched, so watching the source is honest."""
    src = (Path(__file__).resolve().parents[2]
           / "robots_realtime" / "labeling" / "live_server.py").read_text()
    assert f'ap.add_argument("--open-ref", type=float, default={C.GRIPPER_OPEN_REF})' in src
    assert f'ap.add_argument("--closed-ref", type=float, default={C.GRIPPER_CLOSED_REF})' in src


# ── real episodes ───────────────────────────────────────────────────────────
def test_the_four_deleted_grasps_come_back_in_grasp_mode():
    """REGRESSION ON RECORDED DATA. Counts measured 2026-09-02 on
    episode_143533_ee94747f and pinned here:

        kitting task : 0 grasp_attempts, 8 flags   (the bug)
        grasp task   : 4 grasp_attempts, all success, all lifted
        hold seconds : 5.275, 3.130, 3.500, 4.230

    write=False throughout: the live auto-labeller owns these files.
    """
    ep = episode(EP_GRASP_ONLY)
    kitting = label_episode_dir(ep, arm="left", urdf_path=URDF,
                                gripper_open_ref=C.GRIPPER_OPEN_REF,
                                gripper_closed_ref=C.GRIPPER_CLOSED_REF,
                                min_transport_m=C.MIN_TRANSPORT_M,
                                geometric_targets=True, write=False)
    assert len(kitting.grasp_attempts) == 0
    assert len(kitting.flags) == 8

    grasp = label_episode_dir(ep, arm="left", urdf_path=URDF,
                              gripper_open_ref=C.GRIPPER_OPEN_REF,
                              gripper_closed_ref=C.GRIPPER_CLOSED_REF,
                              task=C.TASK_GRASP, write=False)
    assert len(grasp.grasp_attempts) == 4
    assert all(g.outcome == "success" and g.lifted for g in grasp.grasp_attempts)
    holds = [round(g.hold_s, 3) for g in grasp.grasp_attempts]
    assert holds == [5.275, 3.130, 3.500, 4.230]


def test_the_on_disk_kitting_labels_are_reproduced_exactly():
    """The other half of the default pin, against the file the live labeller
    actually wrote: re-labelling in kitting mode reproduces it field for field
    (labeler_version excepted — 0.2.0 is the additive-fields bump)."""
    ep = episode(EP_GRASP_ONLY)
    on_disk = json.loads((ep / "annotations.json").read_text())
    fresh = label_episode_dir(ep, arm="left", urdf_path=URDF,
                              gripper_open_ref=C.GRIPPER_OPEN_REF,
                              gripper_closed_ref=C.GRIPPER_CLOSED_REF,
                              min_transport_m=C.MIN_TRANSPORT_M,
                              geometric_targets=True, write=False).to_dict()

    def strip(d):
        d = dict(d)
        d.pop("labeler_version", None)
        d["episode_meta"] = {k: v for k, v in d["episode_meta"].items()
                             if k != "labeler_version"}
        return d

    assert strip(fresh) == strip(on_disk)


@pytest.mark.parametrize("name,n_grasps", [(EP_HELD_OUT[0], 35), (EP_HELD_OUT[1], 13)])
def test_held_out_episodes_label_in_grasp_mode(name, n_grasps):
    """The HELD-OUT set: the newest finished takes of 2026-09-02. Grasp counts
    measured at implementation time and pinned. These episodes are the
    regression fixtures AND the test set — they must never be counted toward the
    training-window total (tools/analyze_act_readiness.py marks them)."""
    ep = episode(name)
    ann = label_episode_dir(ep, arm="left", urdf_path=URDF,
                            gripper_open_ref=C.GRIPPER_OPEN_REF,
                            gripper_closed_ref=C.GRIPPER_CLOSED_REF,
                            task=C.TASK_GRASP, write=False)
    ok = [g for g in ann.grasp_attempts if g.outcome == "success"]
    assert len(ok) == n_grasps
    assert all(g.hold_s is not None and g.hold_s >= C.MIN_HOLD_S for g in ok)
    assert np.median([g.hold_s for g in ok]) > 1.0
