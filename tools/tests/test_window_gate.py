"""W2 — pose-predicate window start and the close-index gate.

WHAT IS BEING PROTECTED. The fixed [t_close-3s, t_close+2s] window puts the jaw
close at frame 90 BY CONSTRUCTION, whatever the operator did — so a corpus
"measured" in that mode measures the exporter's constant, not the data. The pose
predicate makes the index a measurement, and the gate then refuses a window
whose close lands too deep into the policy's chunk to be reachable from one plan
(yam-pick-pipeline/check_handover_pose.py:17-31 — a 1 s window shift put 46% of
closes past frame 100 and the checkpoint scored 0/5 on hardware).
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import EP_GRASP_ONLY, EP_KITTING_A, EP_KITTING_B, URDF, episode  # noqa: E402
from export_lerobot import (  # noqa: E402
    CLOSE_IDX_CHUNK_FRAC,
    LEFT_CHUNK_FRAMES,
    PREGRASP_MARGIN_FRAMES,
    PREGRASP_PLANE_M,
    Report,
    close_frame_index,
    close_idx_gate,
    grasp_windows,
    grasp_windows_indexed,
    grasp_windows_pose_indexed,
    plan_episode,
    pregrasp_start,
)

FPS = 30
MARGIN_S = PREGRASP_MARGIN_FRAMES / FPS


def descent(t_close, *, hover_s, descent_s, z_hover=0.30, z_grasp=0.10, span=6.0):
    """A synthetic approach: hover at ``z_hover``, then descend over
    ``descent_s`` to ``z_grasp`` exactly at ``t_close``."""
    t = np.arange(t_close - span, t_close + 0.001, 1.0 / 200)
    z = np.full_like(t, z_hover)
    descending = t > t_close - descent_s
    z[descending] = np.interp(t[descending], [t_close - descent_s, t_close],
                              [z_hover, z_grasp])
    z[t < t_close - span + hover_s] = z_hover
    return t, z


def g(t, arm="left"):
    return {"bag_id": 1, "attempt": 1, "arm": arm, "t": t, "outcome": "success",
            "ee_pose": [0.43, -0.25, 0.10, 1.0, 0.0, 0.0, 0.0]}


# ── the predicate ───────────────────────────────────────────────────────────
def test_a_fast_approach_opens_the_window_at_the_descent():
    """1 s descent: the window opens 1 s + the margin before the close, not the
    fixed 3 s. That is the whole point — 90 frames becomes ~45."""
    t, z = descent(100.0, hover_s=2.0, descent_s=1.0)
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert mode == "pose"
    # crossing of z_grasp + 5cm on a 20cm/1s descent is 0.25 s before the close
    assert lo == pytest.approx(100.0 - 0.25 - MARGIN_S, abs=0.02)
    assert close_frame_index(100.0, lo, FPS) == 23


def test_a_slow_approach_is_clamped_to_the_historical_window():
    """Plunge, then creep: the arm drops through the plane at t-2.8 s and spends
    the rest of the approach inching down. The crossing minus the margin lands
    BEFORE the pre_s floor, and the floor wins — a pose start may only ever move
    the window later, never earlier than the historical one."""
    t = np.arange(94.0, 100.001, 1.0 / 200)
    z = np.where(t <= 97.2, 0.20, np.interp(t, [97.2, 100.0], [0.12, 0.10]))
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert mode == "clamped" and lo == 97.0
    assert close_frame_index(100.0, lo, FPS) == 90


def test_a_descent_that_began_before_the_search_span_falls_back_to_the_clamp():
    """Six seconds of continuous creep from 15.5 cm: nothing inside the last
    3 s was ever above the plane, so there is no crossing to find."""
    t = np.arange(94.0, 100.001, 1.0 / 200)
    z = np.interp(t, [94.0, 100.0], [0.155, 0.10])
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert (lo, mode) == (97.0, "clamp")


def test_a_moderate_descent_still_beats_the_fixed_window():
    """A 5 s descent from 30 cm: the plane crossing is 1.25 s before the close,
    so the window opens at 1.755 s and the close lands at frame 53 instead of
    90 — the fixed window's index is a constant, this one is a measurement."""
    t, z = descent(100.0, hover_s=0.5, descent_s=5.0)
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert mode == "pose"
    assert close_frame_index(100.0, lo, FPS) == 53


def test_no_crossing_falls_back_to_the_clamp():
    """A re-grasp from a low hover never rises above the plane. The predicate
    has nothing to measure and must produce the window this exporter has always
    produced, not a new one."""
    t = np.arange(97.0, 100.001, 1.0 / 200)
    z = np.full_like(t, 0.11)                     # never 5 cm above the grasp
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert (lo, mode) == (97.0, "clamp")


def test_the_last_crossing_wins_not_the_first():
    """Hover, dip, lift away, come back down. Only the final descent belongs to
    the grasp being windowed; taking the first crossing would open the window on
    an abandoned approach."""
    t = np.arange(94.0, 100.001, 1.0 / 200)
    z = np.full_like(t, 0.30)
    z[(t > 95.0) & (t < 95.5)] = 0.12             # the abandoned dip
    z[t > 99.5] = np.interp(t[t > 99.5], [99.5, 100.0], [0.30, 0.10])
    lo, mode = pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                              margin_s=MARGIN_S, floor_t=97.0)
    assert mode == "pose" and lo > 99.0


def test_the_plane_is_relative_to_each_grasps_own_height():
    """Two grasps 20 cm apart in z get the same window, because the plane is
    defined from the grasp's own z. A fixed table height would window one of
    them from the wrong instant."""
    outs = []
    for z_grasp in (0.10, 0.30):
        t, z = descent(100.0, hover_s=2.0, descent_s=1.0,
                       z_hover=z_grasp + 0.20, z_grasp=z_grasp)
        outs.append(pregrasp_start(100.0, t, z, plane_m=PREGRASP_PLANE_M,
                                   margin_s=MARGIN_S, floor_t=97.0))
    assert outs[0][1] == outs[1][1] == "pose"
    assert outs[0][0] == pytest.approx(outs[1][0], abs=1e-6)


# ── window assembly ─────────────────────────────────────────────────────────
def test_pose_windows_still_never_overlap():
    """The non-overlap rule is the exporter's, not the mode's: two windows
    sharing frames teach contradictory actions for one image."""
    grasps = [g(100.0), g(102.0)]
    lookup = lambda _g: descent(_g["t"], hover_s=2.0, descent_s=0.4)  # noqa: E731
    ws = grasp_windows_pose_indexed(grasps, 0.0, 200.0, 3.0, 2.0, lookup)
    assert len(ws) == 2
    (_, _lo0, hi0, _, _), (_, lo1, _hi1, _, _) = ws
    assert hi0 <= lo1 + 1e-9


def test_pose_mode_without_fk_reproduces_the_fixed_windows():
    """``ee_z`` returning None is the no-FK path. It must produce EXACTLY the
    windows grasp_windows_indexed produces — the fallback is the old behaviour,
    not a third definition."""
    grasps = [g(10.0), g(20.0), g(21.0)]
    fixed = grasp_windows_indexed(grasps, 0.0, 100.0, 3.0, 2.0)
    posed = grasp_windows_pose_indexed(grasps, 0.0, 100.0, 3.0, 2.0, lambda _g: None)
    assert [(i, lo, hi) for i, lo, hi, _, _ in posed] == fixed
    assert {m for *_, m in posed} == {"clamp"}


def test_the_gate_is_the_chunk_not_the_executed_horizon():
    """act_runner executes n_action_steps=16 and re-queries, so 16 is NOT the
    bound — the model only ever plans chunk_size ahead."""
    assert close_idx_gate(LEFT_CHUNK_FRAMES, CLOSE_IDX_CHUNK_FRAC) == 80
    assert close_idx_gate(100, 1.0) == 100
    assert close_idx_gate(60, 0.8) == 48


def test_close_frame_index_counts_from_the_window_start():
    assert close_frame_index(100.0, 97.0, 30) == 90
    assert close_frame_index(100.0, 99.0, 30) == 30


# ── default mode is untouched ───────────────────────────────────────────────
def test_default_grasp_windows_are_byte_identical():
    """DEFAULT-BEHAVIOUR PIN for the exporter: the fixed-window rule and its
    clipping are exactly what they were before grasp-pose existed."""
    grasps = [g(10.0), g(12.0), g(100.0)]
    assert grasp_windows(grasps, 0.0, 200.0, 3.0, 2.0) == [
        (7.0, 11.0), (11.0, 14.0), (97.0, 102.0)]


def test_plan_episode_rejects_an_unlabelled_episode_in_both_grasp_modes(tmp_path):
    ep = tmp_path / "episode_x"
    ep.mkdir()
    for mode in ("grasp", "grasp-pose"):
        rep = Report()
        assert plan_episode(ep, 3.0, 2.0, 30, rep, window_mode=mode) is None
        assert "no annotations.json" in rep.rejected[0].reason


# ── real episodes ───────────────────────────────────────────────────────────
def _plan(ep, mode, **kw):
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, FPS, rep, None, None, None, ("left",), mode,
                        1.0, 0.0, URDF, kw.get("chunk", LEFT_CHUNK_FRAMES),
                        kw.get("frac", CLOSE_IDX_CHUNK_FRAC))
    return plan, rep


@pytest.mark.parametrize("name,n_fixed,n_pose", [(EP_KITTING_A, 30, 27),
                                                 (EP_KITTING_B, 42, 33)])
def test_real_episode_window_counts(name, n_fixed, n_pose):
    """REGRESSION ON RECORDED DATA, measured 2026-09-02 and pinned. The fixed
    counts are what `export_lerobot.py --dry-run` writes today; the pose counts
    are what survives the 0.8-of-chunk gate."""
    ep = episode(name)
    fixed, _ = _plan(ep, "grasp")
    posed, rep = _plan(ep, "grasp-pose")
    assert len(fixed["windows"]) == n_fixed
    assert len(posed["windows"]) == n_pose
    assert len(rep.close_idx) == n_fixed        # every fixed window is a candidate
    assert all(idx <= 80 for _, idx, _, kept in rep.close_idx if kept)
    assert all(idx > 80 for _, idx, _, kept in rep.close_idx if not kept)


def test_the_pose_predicate_actually_moves_the_close_index():
    """The measurement that justifies W2: in the fixed window every close sits
    at frame 90 by construction; with the pose start the median drops to 58 over
    this corpus (measured 2026-09-02)."""
    _, rep = _plan(episode(EP_KITTING_B), "grasp-pose")
    idxs = np.array([i for _, i, _, _ in rep.close_idx])
    assert idxs.max() <= 90                      # the clamp is the ceiling
    assert np.median(idxs) < 75                  # measured: 58 over both episodes
    assert (np.array([m for _, _, m, _ in rep.close_idx]) == "pose").mean() > 0.5


def test_a_grasp_only_episode_still_needs_grasp_mode_labels():
    """W1 and W2 are independent: the pose predicate cannot rescue an episode the
    LABELLER deleted. episode_143533's annotations.json (kitting mode) has zero
    grasp_attempts, so both window modes reject it — the fix is to re-label."""
    ep = episode(EP_GRASP_ONLY)
    for mode in ("grasp", "grasp-pose"):
        plan, rep = _plan(ep, mode)
        assert plan is None
        assert "zero grasp_attempts" in rep.rejected[0].reason


def test_raising_the_gate_to_the_whole_chunk_keeps_every_window():
    """The 0.8 constant is a margin, not a measurement (PLAN-ACT-READINESS.md
    leaves it open). At frac=1.0 nothing in this corpus closes past the chunk —
    which is the property check_handover_pose.py actually tests."""
    ep = episode(EP_KITTING_A)
    posed, rep = _plan(ep, "grasp-pose", frac=1.0)
    assert len(posed["windows"]) == 30
    assert all(kept for *_, kept in rep.close_idx)
