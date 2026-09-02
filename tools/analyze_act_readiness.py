#!/usr/bin/env python3
"""Is the recorded corpus ready to train an ACT grasp policy? READ-ONLY.

    ./.venv/bin/python3 tools/analyze_act_readiness.py --root recordings/20260902

Opens no device, touches no bus, decodes no video, writes no file, and never
deletes an episode. It reads MCAP joint streams, camera timestamp sidecars and
annotations.json — nothing else.

WHAT IT REPORTS, AND WHY EACH ONE IS HERE
─────────────────────────────────────────
1. CLOSE FRAME INDEX vs the chunk. The one property that predicted the only
   checkpoint that has ever worked on this rig: a correctly-trained policy
   scored 0/5 because 46% of its training windows put the jaw close beyond
   frame 100, and one committed chunk cannot reach a close it was never shown
   inside a single plan (yam-pick-pipeline/check_handover_pose.py:17-31). The
   training loss cannot see this.
2. WINDOW-START FK SPREAD — REPORTED, NOT GATED. Free-play teleop starts wide
   by design; a 1-3 cm gate would fail every healthy episode. The number that
   matters is the spread of the windows that survive the W2 gate, so it is
   printed for exactly those.
3. GRIPPER span and bimodality, against the REAL dead-channel floor measured in
   constants.GRIPPER_MIN_RANGE_FRAC (a dead channel still shows ~1e-4 of noise,
   so an epsilon test certifies it as healthy).
4. PER-STREAM CADENCE. Each stream's OWN nominal rate and its gaps. Camera skew
   is REPORTED, never gated: the cameras free-run, so ±half a frame interval is
   what a healthy episode looks like and a 5 ms gate fails all of them.
5. HOLD DURATION. How long the jaws stayed shut per grasp. For a grasp-only
   policy this plus the FK lift IS the quality signal — transport distance
   cannot be, because a grasp demo transports nothing. Taken from annotations
   written by labeler 0.2.0+; for older files it is MEASURED from the mcap with
   the production segmenter and the row says so. It is never reported as zero
   because it is missing.
6. CORPUS TOTALS against the target, with the held-out set counted separately.

HOW IT STAYS HONEST. Every window count comes from the PRODUCTION planner,
`export_lerobot.plan_episode`, imported — not a reimplementation of its filters.
A number here that disagrees with `export_lerobot.py --dry-run` is a bug in one
of them, and that is the point: this tool must be checkable against the thing
that actually writes the dataset.

NO CACHE. Measured at ~5 min for the whole corpus, which is cheaper than the
first stale-cache debugging session would be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robots_realtime.labeling import constants as C  # noqa: E402
from robots_realtime.labeling.fk import ForwardKinematics  # noqa: E402
from robots_realtime.labeling.label_episode import annotations_path  # noqa: E402
from robots_realtime.labeling.mcap_io import read_positions  # noqa: E402
from robots_realtime.labeling.segmentation import (  # noqa: E402
    GripperRangeUnknown,
    normalize_width,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_lerobot import (  # noqa: E402
    ARM_SETS,
    CAMERA_SETS,
    CLOSE_IDX_CHUNK_FRAC,
    DEFAULT_FPS,
    DEFAULT_POST_S,
    DEFAULT_PRE_S,
    DEFAULT_URDF,
    LEFT_CHUNK_FRAMES,
    Report,
    close_idx_gate,
    episode_dirs,
    nearest_index,
    plan_episode,
)

# Corpus targets from PLAN-ACT-READINESS.md. Reported as progress, never as a
# gate: this tool does not decide when to stop recording, the operator does.
TARGET_TRAIN = (80, 120)
TARGET_HELD_OUT = (15, 20)

# An episode state, in the order the checks run. The order is the point: a
# missing mcap is CORRUPT whatever else is missing, and an episode killed before
# its camera sidecars were written is INELIGIBLE rather than "unlabelled",
# because no amount of labelling will make it exportable.
STATES = ("OK", "UNLABELLED", "INELIGIBLE", "CORRUPT")


# ── per-episode facts ───────────────────────────────────────────────────────
_POS: dict[str, tuple] = {}


def positions(path: Path, node: str):
    """read_positions, memoised FOR THIS PROCESS ONLY.

    Not the npz cache the plan rejected: nothing is written to disk, so there is
    no staleness trap — a re-run re-reads everything.
    """
    key = str(path)
    if key not in _POS:
        _POS[key] = read_positions(path, node)
    return _POS[key]


def measure_holds(ep: Path, arm: str, open_ref, closed_ref, fps: int,
                  urdf_path: str) -> tuple[list[float], list[float], str]:
    """Hold durations measured straight from the MCAP → (success, other, note).

    Used when annotations.json predates labeler 0.2.0 and therefore carries no
    ``hold_s``. It calls the PRODUCTION segmenter (``detect_grip_intervals``,
    the same one the labeller uses, with the same thresholds) — nothing about
    grasp detection is reimplemented here.

    The one shortcut: FK for the lift test is computed on the {fps} Hz grid and
    linearly interpolated back onto the 200 Hz joint timeline, instead of
    FK-ing all ~100k samples. The lift test asks whether the end-effector rose
    3 cm within 2 s, which no amount of 200 Hz detail changes, and the full-rate
    version costs ~20 s per episode for the same answer.
    """
    from robots_realtime.labeling.segmentation import detect_grip_intervals

    try:
        t, pos = positions(ep / f"yam_{arm}.mcap", f"yam_{arm}")
    except Exception as e:
        return [], [], f"unreadable: {str(e)[:60]}"
    if t.size < 2:
        return [], [], "no samples"

    fk = ForwardKinematics(urdf_path)
    n = max(2, int((float(t[-1]) - float(t[0])) * fps) + 1)
    grid = np.linspace(float(t[0]), float(t[-1]), n)
    idx = nearest_index(t, grid)        # the exporter's resampler, imported
    z_grid = fk.ee_positions(pos[idx, : C.N_ARM_JOINTS])[:, 2]
    ee_z = np.interp(t, grid, z_grid)

    try:
        ivs = detect_grip_intervals(t, pos[:, C.GRIPPER_JOINT_INDEX], ee_z=ee_z,
                                    open_ref=open_ref, closed_ref=closed_ref)
    except GripperRangeUnknown as e:
        return [], [], f"gripper unusable: {str(e)[:60]}"
    end = float(t[-1])
    ok, other = [], []
    for iv in ivs:
        d = float((iv.t_open if iv.t_open is not None else end) - iv.t_close)
        (ok if iv.outcome == "success" else other).append(d)
    return ok, other, "measured from mcap"


def classify_episode(ep: Path, arm: str, cameras: dict) -> tuple[str, str]:
    """→ (state, detail). Reads headers only; decodes nothing."""
    mcap = ep / f"yam_{arm}.mcap"
    if not mcap.exists():
        return "CORRUPT", f"no yam_{arm}.mcap"
    try:
        t, pos = positions(mcap, f"yam_{arm}")
    except Exception as e:                      # unreadable mcap = corrupt, not empty
        return "CORRUPT", f"mcap unreadable: {str(e)[:80]}"
    if t.size == 0 or pos.size == 0:
        return "CORRUPT", "mcap holds no joint samples"

    # A session killed mid-episode leaves the mp4 (ffmpeg flushes) but no
    # timestamp sidecar, and without timestamps nothing can be resampled onto
    # the export grid. That is not a labelling problem — it is unexportable.
    missing = [c for c in cameras
               if not (ep / f"{c}-rgb-timestamp.npy").exists()
               or not (ep / f"{c}-images-rgb.mp4").exists()]
    if missing:
        return "INELIGIBLE", f"no timestamps/video for {', '.join(sorted(missing))}"

    if not annotations_path(ep, arm).exists():
        return "UNLABELLED", f"no {annotations_path(ep, arm).name}"
    return "OK", ""


def stream_cadence(ep: Path, arm: str, cameras: dict) -> dict:
    """Per-stream nominal rate and gaps. Each stream against ITSELF, never a
    shared nominal: yam runs at ~200 Hz, gello at ~62.5 Hz and the cameras
    free-run near 30 Hz, so one global expectation would flag three healthy
    streams to catch a fourth."""
    out: dict[str, dict] = {}

    def add(name: str, t: np.ndarray) -> None:
        t = np.asarray(t, float)
        if t.size < 3:
            out[name] = {"n": int(t.size), "hz": None, "gaps": None}
            return
        dt = np.diff(t)
        med = float(np.median(dt))
        # A "gap" is a sample interval more than 3x the stream's own median —
        # a dropped block, not jitter.
        gaps = int((dt > 3 * med).sum()) if med > 0 else 0
        out[name] = {"n": int(t.size), "hz": (1.0 / med if med > 0 else None),
                     "gaps": gaps,
                     "worst_gap_s": float(dt.max()) if dt.size else 0.0,
                     "t0": float(t[0]), "t1": float(t[-1])}

    for node in (f"yam_{arm}", f"gello_{arm}"):
        f = ep / f"{node}.mcap"
        if f.exists():
            try:
                t, _ = positions(f, node)
                add(node, t)
            except Exception as e:
                out[node] = {"n": 0, "hz": None, "gaps": None, "error": str(e)[:60]}
    for cam in cameras:
        f = ep / f"{cam}-rgb-timestamp.npy"
        if f.exists():
            add(cam, np.load(f).astype(float).ravel())
    return out


def camera_skew_s(ep: Path, cameras: dict) -> float | None:
    """Median |Δt| between the two cameras' nearest frames. REPORTED ONLY.

    Free-running cameras are uniformly distributed within a frame interval of
    each other, so the expected value here is ~a quarter of a frame (~8 ms at
    30 Hz) on a perfectly healthy episode. It is a description of the rig, not
    a defect, which is why nothing gates on it.
    """
    stamps = []
    for cam in sorted(cameras):
        f = ep / f"{cam}-rgb-timestamp.npy"
        if f.exists():
            v = np.load(f).astype(float).ravel()
            if v.size > 2:
                stamps.append(np.sort(v))
    if len(stamps) < 2:
        return None
    ta, tb = stamps[0], stamps[1]        # the real recorded stamps, not a model of them
    idx = np.searchsorted(tb, ta).clip(1, tb.size - 1)
    d = np.minimum(np.abs(ta - tb[idx - 1]), np.abs(ta - tb[idx]))
    return float(np.median(d))


def gripper_health(ep: Path, arm: str, open_ref: float | None,
                   closed_ref: float | None) -> dict:
    """Span and bimodality of the gripper channel over the whole episode."""
    try:
        _, pos = positions(ep / f"yam_{arm}.mcap", f"yam_{arm}")
    except Exception as e:
        return {"error": str(e)[:80]}
    if pos.size == 0:
        return {"error": "no samples"}
    raw = pos[:, C.GRIPPER_JOINT_INDEX].astype(float)
    lo, hi = (float(v) for v in np.percentile(raw, [2, 98]))
    span = max(abs(hi), abs(lo), 1.0e-12)
    # The SAME relative floor the labeller uses, imported not re-derived.
    dead = (hi - lo) <= C.GRIPPER_MIN_RANGE_FRAC * span
    out = {"p2": lo, "p98": hi, "spread": hi - lo,
           "floor": C.GRIPPER_MIN_RANGE_FRAC * span, "dead": bool(dead)}
    try:
        n = normalize_width(raw, open_ref=open_ref, closed_ref=closed_ref)
        out["frac_closed"] = float((n < C.GRIPPER_CLOSE_ENTER).mean())
        out["frac_open"] = float((n > C.GRIPPER_CLOSE_EXIT).mean())
        # Bimodal = nearly all mass at the two ends. A channel that lives in the
        # middle is a jaw that never fully opened or closed.
        out["bimodal_frac"] = out["frac_closed"] + out["frac_open"]
        out["norm_min"] = float(n.min())
    except GripperRangeUnknown as e:
        out["unknown"] = str(e)[:100]
    return out


def hold_durations(ep: Path, arm: str) -> tuple[list[float], int, int]:
    """(hold seconds of successful grasps, n successful, n attempts missing hold).

    ``hold_s`` arrived with labeler 0.2.0. An older annotations.json has none,
    and that is reported as unknown — a missing hold is not a zero-length one.
    """
    f = annotations_path(ep, arm)
    if not f.exists():
        return [], 0, 0
    try:
        ann = json.loads(f.read_text())
    except Exception:
        return [], 0, 0
    holds, n_ok, n_missing = [], 0, 0
    for g in ann.get("grasp_attempts") or []:
        if g.get("outcome") != "success":
            continue
        n_ok += 1
        if g.get("hold_s") is None:
            n_missing += 1
        else:
            holds.append(float(g["hold_s"]))
    return holds, n_ok, n_missing


def window_start_poses(plan: dict, urdf_path: str) -> np.ndarray:
    """FK end-effector position at each window's first frame → (N, 3).

    The URDF path is passed in EXPLICITLY. ``fk.ForwardKinematics``'s default is
    "urdf/yam.urdf", relative to the current working directory, so a tool run
    from anywhere but the repo root would fail — or worse, load a different
    URDF and report a shifted spread as a data finding.
    """
    fk = ForwardKinematics(urdf_path)
    arm = plan["arms"][0]
    t_s, state, _, _ = plan["streams"][arm]
    out = []
    for lo, _hi in plan["windows"]:
        i = int(np.argmin(np.abs(t_s - lo)))
        out.append(fk.ee_pose(state[i, : C.N_ARM_JOINTS])[:3])
    return np.asarray(out, float).reshape(-1, 3)


# ── formatting ──────────────────────────────────────────────────────────────
def dist(values, fmt="{:.2f}") -> str:
    v = np.asarray(sorted(values), float)
    if v.size == 0:
        return "n=0"
    q = lambda p: fmt.format(float(np.percentile(v, p)))  # noqa: E731
    return (f"n={v.size}  min {fmt.format(float(v.min()))}  p10 {q(10)}  "
            f"median {q(50)}  p90 {q(90)}  max {fmt.format(float(v.max()))}")


def analyze(root: Path, arm: str, cameras: dict, args) -> int:
    arms = (arm,)
    eps = episode_dirs(root, arms)          # .trash excluded, exporter parity
    if not eps:
        print(f"no episodes under {root}", file=sys.stderr)
        return 2

    held_out = set(args.held_out)
    if args.held_out_newest:
        # Newest by directory name: episode_HHMMSS_hash sorts chronologically
        # inside a day directory, which is how the recorder names them.
        held_out |= {e.name for e in sorted(eps, key=lambda p: p.name)[-args.held_out_newest:]}

    states: dict[str, list[Path]] = {s: [] for s in STATES}
    per_ep: dict[str, dict] = {}

    print(f"root            : {root}")
    print(f"arm / cameras   : {arm} / {', '.join(sorted(cameras))}")
    print(f"urdf            : {args.urdf}")
    print(f"chunk           : {args.chunk_frames} frames @ {args.fps} Hz  "
          f"(gate <= {close_idx_gate(args.chunk_frames, args.close_idx_frac)} "
          f"= {args.close_idx_frac:g} x chunk)")
    if held_out:
        print(f"held out        : {', '.join(sorted(held_out))}  "
              "(analysed, NEVER counted toward the training total)")
    print()

    for ep in eps:
        state, detail = classify_episode(ep, arm, cameras)
        states[state].append(ep)
        rec: dict = {"state": state, "detail": detail,
                     "held_out": ep.name in held_out}
        per_ep[ep.name] = rec
        if state == "CORRUPT":
            continue

        rec["cadence"] = stream_cadence(ep, arm, cameras)
        rec["skew_s"] = camera_skew_s(ep, cameras)
        rec["gripper"] = gripper_health(ep, arm, args.gripper_open_ref,
                                        args.gripper_closed_ref)
        rec["holds"], rec["n_success"], rec["holds_missing"] = hold_durations(ep, arm)
        rec["holds_note"] = "from annotations"
        if not rec["holds"] and args.measure_holds:
            rec["holds"], rec["holds_other"], rec["holds_note"] = measure_holds(
                ep, arm, args.gripper_open_ref, args.gripper_closed_ref,
                args.fps, args.urdf)

        # The production planner, in both window definitions. Two Reports so a
        # rejection can be attributed to the mode that produced it.
        for mode in ("grasp", "grasp-pose"):
            r = Report()
            try:
                plan = plan_episode(
                    ep, args.pre_s, args.post_s, args.fps, r,
                    None, None, cameras, arms, mode,
                    args.gripper_open_ref, args.gripper_closed_ref,
                    args.urdf, args.chunk_frames, args.close_idx_frac)
            except Exception as e:            # a planner crash is a finding
                r.reject(ep.name, f"planner raised {type(e).__name__}: {str(e)[:80]}")
                plan = None
            rec[mode] = {"windows": len(plan["windows"]) if plan else 0,
                         "rejected": [x.reason for x in r.rejected],
                         "close_idx": list(r.close_idx)}
            if plan and mode == "grasp-pose":
                rec["start_xyz"] = window_start_poses(plan, args.urdf)

    # ── episode states ──────────────────────────────────────────────────────
    print("EPISODES")
    for ep in eps:
        r = per_ep[ep.name]
        tag = "  [HELD OUT]" if r["held_out"] else ""
        g = r.get("grasp", {}).get("windows", 0)
        gp = r.get("grasp-pose", {}).get("windows", 0)
        print(f"  {ep.name:<34} {r['state']:<11} windows grasp={g:<3} "
              f"grasp-pose={gp:<3}{tag}  {r['detail']}")
    print("  " + "  ".join(f"{s}={len(states[s])}" for s in STATES))

    # ── close index ─────────────────────────────────────────────────────────
    rows = [row for r in per_ep.values() if not r["held_out"]
            for row in r.get("grasp-pose", {}).get("close_idx", [])]
    ho_rows = [row for r in per_ep.values() if r["held_out"]
               for row in r.get("grasp-pose", {}).get("close_idx", [])]
    print("\nCLOSE FRAME INDEX (grasp-pose windows; the decisive check)")
    if rows:
        idxs = [i for _, i, _, _ in rows]
        modes: dict[str, int] = {}
        for _, _, m, _ in rows:
            modes[m] = modes.get(m, 0) + 1
        print(f"  training  {dist(idxs, '{:.0f}')}")
        print("  starts    " + "  ".join(f"{m}={n}" for m, n in sorted(modes.items()))
              + "   (clamp/clamped = no descent found, the old fixed window)")
        print(f"  passed    {sum(1 for r in rows if r[3])} / {len(rows)} "
              f"<= frame {close_idx_gate(args.chunk_frames, args.close_idx_frac)}")
    else:
        print("  no candidate windows")
    if ho_rows:
        print(f"  held out  {dist([i for _, i, _, _ in ho_rows], '{:.0f}')}")

    # ── window-start FK spread (REPORTED, not gated) ────────────────────────
    starts = [r["start_xyz"] for r in per_ep.values()
              if not r["held_out"] and r.get("start_xyz") is not None
              and len(r["start_xyz"])]
    print("\nWINDOW-START FK SPREAD (report only — free-play teleop is expected to be wide)")
    if starts:
        P = np.vstack(starts)
        c = P.mean(axis=0)
        d = np.linalg.norm(P - c, axis=1)
        print(f"  n={len(P)}  centroid [{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}] m")
        print(f"  per-axis range  x {P[:, 0].ptp():.3f}  y {P[:, 1].ptp():.3f}  "
              f"z {P[:, 2].ptp():.3f} m")
        print(f"  distance to centroid  {dist(d, '{:.3f}')} m")
    else:
        print("  no windows to measure")

    # ── gripper ─────────────────────────────────────────────────────────────
    print("\nGRIPPER (span vs the measured dead-channel floor, and bimodality)")
    for ep in eps:
        r = per_ep[ep.name]
        g = r.get("gripper")
        if not g:
            continue
        if "error" in g:
            print(f"  {ep.name:<34} unreadable: {g['error']}")
            continue
        verdict = ("DEAD" if g["dead"] else
                   "ok" if g.get("bimodal_frac", 0) >= 0.9 else "MID-BAND")
        print(f"  {ep.name:<34} spread {g['spread']:.4f} (floor {g['floor']:.4f})  "
              f"closed {g.get('frac_closed', float('nan')):.2f} / "
              f"open {g.get('frac_open', float('nan')):.2f}  {verdict}")

    # ── cadence ─────────────────────────────────────────────────────────────
    print("\nPER-STREAM CADENCE (each stream against its own nominal; skew reported, not gated)")
    for ep in eps:
        r = per_ep[ep.name]
        cad = r.get("cadence")
        if not cad:
            continue
        parts = []
        for name, v in cad.items():
            hz = f"{v['hz']:.1f}Hz" if v.get("hz") else "—"
            parts.append(f"{name.replace('camera_', 'cam_')} {hz}"
                         + (f" gaps={v['gaps']}" if v.get("gaps") else ""))
        skew = r.get("skew_s")
        print(f"  {ep.name:<34} " + "  ".join(parts)
              + (f"   skew {skew * 1000:.1f} ms" if skew is not None else ""))

    # ── hold duration ───────────────────────────────────────────────────────
    print("\nGRIPPER HOLD DURATION (seconds jaws shut per successful grasp)")
    all_holds, ho_holds, missing = [], [], 0
    for ep in eps:
        r = per_ep[ep.name]
        h = r.get("holds") or []
        missing += r.get("holds_missing", 0)
        if h:
            (ho_holds if r["held_out"] else all_holds).extend(h)
            print(f"  {ep.name:<34} {dist(h)}  [{r.get('holds_note', '')}]"
                  + ("  [HELD OUT]" if r["held_out"] else ""))
        elif r.get("n_success"):
            print(f"  {ep.name:<34} {r['n_success']} successful grasps, hold unknown "
                  f"({r.get('holds_note', 'no measurement')})")
    print(f"  CORPUS    {dist(all_holds)}")
    if ho_holds:
        print(f"  HELD OUT  {dist(ho_holds)}")
    if missing:
        print(f"  {missing} successful grasps in annotations carry no hold_s "
              "(written before labeler 0.2.0; the rows above marked 'measured from "
              "mcap' come from the production segmenter instead)")

    # ── corpus totals ───────────────────────────────────────────────────────
    print("\nCORPUS TOTALS")
    for mode in ("grasp", "grasp-pose"):
        tr = sum(r.get(mode, {}).get("windows", 0) for r in per_ep.values()
                 if not r["held_out"])
        ho = sum(r.get(mode, {}).get("windows", 0) for r in per_ep.values()
                 if r["held_out"])
        lo, hi = TARGET_TRAIN
        print(f"  {mode:<11} training {tr:<4} (target {lo}-{hi}, "
              f"{'MET' if tr >= lo else f'{lo - tr} short'})   held out {ho:<4} "
              f"(target {TARGET_HELD_OUT[0]}-{TARGET_HELD_OUT[1]})")
    reasons: dict[str, int] = {}
    for r in per_ep.values():
        for why in r.get("grasp", {}).get("rejected", []):
            key = why.split(" (")[0][:70]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("  rejections (grasp mode, the default export):")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}  {why}")
    print("\n  reconcile with:  tools/export_lerobot.py --root "
          f"{root} --dry-run   (grasp-mode training+held-out windows must match "
          "'grasp windows')")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="recordings")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="left")
    ap.add_argument("--cameras", choices=sorted(CAMERA_SETS), default="both")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--pre-s", type=float, default=DEFAULT_PRE_S)
    ap.add_argument("--post-s", type=float, default=DEFAULT_POST_S)
    ap.add_argument("--urdf", default=DEFAULT_URDF,
                    help="explicit URDF path; fk.py's default is cwd-relative")
    ap.add_argument("--chunk-frames", type=int, default=LEFT_CHUNK_FRAMES)
    ap.add_argument("--close-idx-frac", type=float, default=CLOSE_IDX_CHUNK_FRAC)
    ap.add_argument("--gripper-open-ref", type=float, default=C.GRIPPER_OPEN_REF)
    ap.add_argument("--gripper-closed-ref", type=float, default=C.GRIPPER_CLOSED_REF)
    ap.add_argument("--held-out", default="",
                    help="comma-separated episode directory names to hold out. They "
                         "are analysed and reported, and never counted toward the "
                         "training-window total.")
    ap.add_argument("--no-measure-holds", dest="measure_holds", action="store_false",
                    help="do not fall back to measuring hold durations from the mcap "
                         "when annotations.json predates labeler 0.2.0")
    ap.add_argument("--held-out-newest", type=int, default=0,
                    help="also hold out the N newest episodes under --root")
    a = ap.parse_args(argv)
    a.held_out = [x for x in (n.strip() for n in a.held_out.split(",")) if x]
    arms = ARM_SETS[a.arms]
    if len(arms) != 1:
        ap.error("--arms both is not supported: every per-episode number here is "
                 "per-arm, and averaging two arms would hide the dead one")
    return analyze(Path(a.root), arms[0], CAMERA_SETS[a.cameras], a)


if __name__ == "__main__":
    raise SystemExit(main())
