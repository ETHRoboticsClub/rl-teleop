#!/usr/bin/env python3
"""What chunk_size, n_action_steps and --close-idx-frac does THIS corpus ask for?

    ./.venv/bin/python3 tools/act_horizon_report.py --root recordings/20260902

READ-ONLY. Opens no device, touches no bus, decodes no video, writes no file
(unless you ask for --json). It calls the PRODUCTION planner
`export_lerobot.plan_episode` once per (episode, pre_s) and reads the windows it
returns — it does not reimplement a single filter. Numbers here that disagree
with `export_lerobot.py --dry-run` are a bug in one of the two.

WHY THIS EXISTS, AND WHAT analyze_act_readiness.py DOES NOT ANSWER
──────────────────────────────────────────────────────────────────
`analyze_act_readiness.py` answers "is the corpus healthy at the CURRENT
settings". Three decisions it deliberately does not make are still open, and
each of them is a number somebody has to type into a training config:

  chunk_size        how many future actions ACT predicts per query
  n_action_steps    how many of them the runtime commits before re-observing
  --close-idx-frac  which windows the grasp-pose exporter is allowed to keep

THE TRAP THIS TOOL EXISTS TO EXPOSE. `close_idx` is not a free measurement of
the data — it is bounded by the exporter's own `--pre-s`:

    close_idx <= pre_s * fps          (= 90 at the default pre_s=3.0, fps=30)

So on a corpus exported with the fixed window, EVERY unclipped window has
close_idx == 90 exactly, and "the working corpus sat at 0.90 of a 100-frame
chunk" is a restatement of pre_s=3.0 — not an empirical threshold. Any gate
expressed as a fraction of the chunk has to be read against that ceiling, which
is why this tool prints the ceiling next to every distribution and sweeps pre_s
instead of trusting one value of it.

WHAT IS ACTUALLY MEASURED, per window:

  close_idx        frame index of the jaw close inside the window
  start mode       pose  = the measured start of the final descent was used
                   clamp = no descent found in the search span -> pre_s fallback
                   clamped = a descent was found, but earlier than the fallback
  n_frames         int((hi - lo) * fps), the exporter's own window length
  post_idx         n_frames - close_idx, the lift supervision after the close

`descent_s` (close_idx / fps) at a LARGE pre_s is the honest, exporter-
independent duration of the final descent — the quantity a horizon should be
sized against. That is what --pre-s-sweep is for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_lerobot as EX  # noqa: E402
from export_lerobot import (  # noqa: E402
    CAMERA_SETS,
    DEFAULT_FPS,
    DEFAULT_POST_S,
    DEFAULT_PRE_S,
    DEFAULT_URDF,
    LEFT_CHUNK_FRAMES,
    MAX_CAM_STALENESS_S,
    MIN_WINDOW_FRAMES,
    Report,
    close_idx_gate,
    episode_dirs,
    load_keep_list,
    plan_episode,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robots_realtime.labeling import constants as C  # noqa: E402
from robots_realtime.labeling.fk import ForwardKinematics  # noqa: E402

# Big enough that the gate inside plan_episode never fires, so every CANDIDATE
# window is returned and the gating is done here, offline, at many thresholds.
UNGATED_FRAC = 1e6


def pct(values, q):
    return float(np.percentile(np.asarray(values, float), q)) if len(values) else float("nan")


def dist(values, fmt="{:.0f}"):
    v = sorted(values)
    if not v:
        return "n=0"
    f = fmt.format
    return (f"n={len(v)}  min {f(v[0])}  p10 {f(pct(v, 10))}  median {f(pct(v, 50))}  "
            f"p90 {f(pct(v, 90))}  max {f(v[-1])}")


def harvest(eps, arms, cameras, pre_s, post_s, fps, urdf, mode, tclose=None,
            fk=None):
    """One record per candidate window, straight out of the production planner."""
    rows, rejected = [], []
    for ep in eps:
        r = Report()
        try:
            plan = plan_episode(ep, pre_s, post_s, fps, r, None, None, cameras,
                                arms, mode, None, None, urdf,
                                LEFT_CHUNK_FRAMES, UNGATED_FRAC)
        except Exception as e:                      # a planner crash is a finding
            rejected.append((ep.name, f"planner raised {type(e).__name__}: {e}"))
            continue
        if plan is None:
            rejected += [(ep.name, x.reason) for x in r.rejected]
            continue
        meta = plan.get("window_meta")
        if not meta:
            # "grasp" mode carries no window_meta. Its close times come from the
            # pose-mode harvest of the SAME episode at the SAME pre_s: identical
            # grasp set, identical sort, identical non-overlap clipping, so the
            # k-th window is the k-th grasp in both. Mismatched lengths mean that
            # assumption broke, and the row says so instead of inventing a number.
            ts = (tclose or {}).get(ep.name) or []
            if len(ts) != len(plan["windows"]):
                ts = [None] * len(plan["windows"])
            meta = [{"t_close": t, "start": "fixed"} for t in ts]
        # Where the arm WAS at each window's first frame, and where the grasp
        # itself happened. The first is the pose a handoff would have to
        # reproduce; the second is where the packet was. Both in the base frame.
        t_s, state = plan["streams"][arms[0]][0], plan["streams"][arms[0]][1]
        grasp_xyz = {}
        if fk is not None:
            ann = json.loads((ep / f"annotations_{arms[0]}.json").read_text()) \
                if (ep / f"annotations_{arms[0]}.json").exists() \
                else json.loads((ep / "annotations.json").read_text())
            for g in ann.get("grasp_attempts", []):
                p = g.get("ee_pose") or []
                if len(p) >= 3 and g.get("t") is not None:
                    grasp_xyz[round(float(g["t"]), 3)] = [float(v) for v in p[:3]]

        for (lo, hi), m in zip(plan["windows"], meta):
            n = int((hi - lo) * fps)
            ci = m.get("close_idx")
            tc = m.get("t_close")
            if ci is None:
                ci = int(round((float(tc) - lo) * fps)) if tc is not None else None
            start_xyz = None
            if fk is not None:
                i = int(np.argmin(np.abs(t_s - lo)))
                start_xyz = [float(v) for v in
                             fk.ee_pose(state[i, : C.N_ARM_JOINTS])[:3]]
            rows.append({"episode": ep.name, "n_frames": n,
                         "close_idx": None if ci is None else int(ci),
                         "post_idx": None if ci is None else n - int(ci),
                         "start": m.get("start", "fixed"),
                         "t_close": tc, "lo": lo, "hi": hi,
                         "short": n < MIN_WINDOW_FRAMES,
                         "start_xyz": start_xyz,
                         "grasp_xyz": grasp_xyz.get(round(float(tc), 3))
                                      if tc is not None else None})
    return rows, rejected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="recordings/20260902")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--cameras", choices=sorted(CAMERA_SETS), default="both")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--pre-s", type=float, default=DEFAULT_PRE_S)
    ap.add_argument("--post-s", type=float, default=DEFAULT_POST_S)
    ap.add_argument("--pre-s-sweep", type=float, nargs="*",
                    default=[2.0, 3.0, 4.0, 6.0, 10.0],
                    help="pre_s values to re-plan at. A LARGE value stops the "
                         "clamp from binding, which is the only way to see the "
                         "true final-descent duration.")
    ap.add_argument("--chunks", type=int, nargs="*", default=[50, 60, 80, 100, 120],
                    help="candidate training chunk_size values for the yield table")
    ap.add_argument("--fracs", type=float, nargs="*",
                    default=[0.6, 0.7, 0.8, 0.9, 1.0],
                    help="candidate --close-idx-frac values for the yield table")
    ap.add_argument("--n-action-steps", type=int, default=16,
                    help="what the runtime actually commits per query "
                         "(yam-pick-pipeline/act_runner.py, left arm)")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--held-out-newest", type=int, default=0)
    ap.add_argument("--keep", default=None,
                    help="keep-list JSON from tools/review_grasps.py — restrict "
                         "every number to the windows the operator actually kept, "
                         "using the exporter's own filter")
    ap.add_argument("--json", default=None,
                    help="also dump the per-window records here (the ONLY write)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    arms = (a.arm,)
    cameras = CAMERA_SETS[a.cameras]
    eps = episode_dirs(root, arms)
    if not eps:
        print(f"no episodes under {root}", file=sys.stderr)
        return 2
    held_out = ({e.name for e in sorted(eps, key=lambda p: p.name)[-a.held_out_newest:]}
                if a.held_out_newest else set())
    if a.keep:
        # The exporter reads its keep-list from a module global. Setting the same
        # global is what makes this tool report the operator's ACTUAL selection
        # rather than a second, drifting interpretation of the same file.
        EX.KEEP = load_keep_list(a.keep)
        print(f"keep-list       : {sum(len(v) for v in EX.KEEP.values())} grasps "
              f"across {len(EX.KEEP)} episodes  ({a.keep})")
    fk = ForwardKinematics(a.urdf)

    print(f"root            : {root}")
    print(f"arm / cameras   : {a.arm} / {', '.join(sorted(cameras))}")
    print(f"fps             : {a.fps}   (1 action step = {1000 / a.fps:.1f} ms)")
    print(f"runtime commit  : n_action_steps={a.n_action_steps} "
          f"= {a.n_action_steps / a.fps:.2f} s of blind execution per query")
    if held_out:
        print(f"held out        : {', '.join(sorted(held_out))}")
    print()

    # ── 1. pose-predicate windows, swept over pre_s ─────────────────────────
    print("POSE-PREDICATE WINDOW (--window-mode grasp-pose), swept over --pre-s")
    print("  pre_s is a CEILING on close_idx, so a distribution that stops dead at")
    print("  the ceiling is measuring the exporter, not the descent.")
    print(f"  {'pre_s':>6} {'ceil':>5} {'n':>4} {'pose':>5} {'clamp':>6} {'clmpd':>6}  "
          f"{'close_idx (frames)':<52}  {'descent_s median':>16}")
    sweeps = {}
    for pre in a.pre_s_sweep:
        rows, _ = harvest(eps, arms, cameras, pre, a.post_s, a.fps, a.urdf,
                          "grasp-pose", fk=fk)
        tr = [r for r in rows if r["episode"] not in held_out]
        sweeps[pre] = rows
        ci = [r["close_idx"] for r in tr]
        modes = {m: sum(1 for r in tr if r["start"] == m)
                 for m in ("pose", "clamp", "clamped")}
        print(f"  {pre:>6.1f} {int(pre * a.fps):>5} {len(tr):>4} {modes['pose']:>5} "
              f"{modes['clamp']:>6} {modes['clamped']:>6}  {dist(ci):<52}  "
              f"{pct(ci, 50) / a.fps:>16.2f}")
    print()

    rows = sweeps.get(a.pre_s) or harvest(eps, arms, cameras, a.pre_s, a.post_s,
                                          a.fps, a.urdf, "grasp-pose", fk=fk)[0]
    train = [r for r in rows if r["episode"] not in held_out]

    # ── 2. the fixed window, for reference ──────────────────────────────────
    tclose: dict[str, list] = {}
    for r in rows:
        tclose.setdefault(r["episode"], []).append(r["t_close"])
    fixed, _ = harvest(eps, arms, cameras, a.pre_s, a.post_s, a.fps, a.urdf,
                       "grasp", tclose, fk=fk)
    train_fixed = [r for r in fixed
                   if r["episode"] not in held_out and r["close_idx"] is not None]
    ceil = int(a.pre_s * a.fps)
    print(f"FIXED WINDOW (--window-mode grasp, pre_s={a.pre_s:g} post_s={a.post_s:g}) "
          "— what the working right-arm corpus used")
    print(f"  ceiling on close_idx = pre_s x fps = {ceil}")
    print(f"  close_idx  {dist([r['close_idx'] for r in train_fixed])}")
    print(f"  n_frames   {dist([r['n_frames'] for r in train_fixed])}")
    print(f"  AT the ceiling: "
          f"{sum(1 for r in train_fixed if r['close_idx'] >= ceil)} / {len(train_fixed)}"
          f"  (these are the windows for which 'close_idx / chunk = 0.90' is just"
          f" pre_s x fps / chunk)")
    print()

    # ── 3. what the gate actually removes ───────────────────────────────────
    print(f"WHAT THE CLOSE-INDEX GATE REMOVES (at pre_s={a.pre_s:g}, "
          f"chunk={LEFT_CHUNK_FRAMES})")
    print(f"  {'frac':>5} {'gate':>5} {'kept':>6} {'dropped':>8}   dropped by start mode")
    for f in a.fracs:
        g = close_idx_gate(LEFT_CHUNK_FRAMES, f)
        drop = [r for r in train if r["close_idx"] > g]
        modes = {m: sum(1 for r in drop if r["start"] == m)
                 for m in ("pose", "clamp", "clamped")}
        print(f"  {f:>5.2f} {g:>5} {len(train) - len(drop):>6} {len(drop):>8}   "
              f"pose={modes['pose']} clamp={modes['clamp']} clamped={modes['clamped']}")
    print()
    print("  close_idx by start mode (a clamp/clamped window sits AT the ceiling by")
    print("  construction, so gating on close_idx is gating on 'the pose predicate")
    print("  failed', not on 'the close is late'):")
    for m in ("pose", "clamp", "clamped"):
        v = [r["close_idx"] for r in train if r["start"] == m]
        print(f"    {m:<8} {dist(v)}")
    print()

    # ── 4. yield table over (chunk_size, frac) ──────────────────────────────
    print("EXPORT YIELD  windows kept / " f"{len(train)} training candidates")
    print(f"  {'chunk':>6} " + " ".join(f"{f:>9.2f}" for f in a.fracs))
    for c in a.chunks:
        cells = []
        for f in a.fracs:
            g = close_idx_gate(c, f)
            k = sum(1 for r in train if r["close_idx"] <= g)
            cells.append(f"{k:>4} ({g:>2})")
        print(f"  {c:>6} " + " ".join(f"{x:>9}" for x in cells))
    print("  cell = kept windows (gate frame index)")
    print()

    # ── 4b. camera frame yield, per episode ─────────────────────────────────
    print("CAMERA FRAME YIELD  (how many of a window's frames actually get WRITTEN)")
    print("  The exporter drops any grid instant with no image inside")
    print(f"  MAX_CAM_STALENESS_S = {MAX_CAM_STALENESS_S * 1000:.0f} ms. A camera that")
    print("  dropped out mid-episode therefore does not fail the export — it")
    print("  silently shortens windows, and a window below one commit")
    print(f"  ({a.n_action_steps} frames) is not a demonstration of anything.")
    print(f"  session_meta 'degraded' is a WHOLE-EPISODE flag; this is the same")
    print("  question asked only of the frames that would be exported.")
    print(f"  {'episode':<34} {'win':>4} {'frames written':>18} {'stubs':>6} {'degraded':>9}")
    for ep in sorted({r["episode"] for r in rows}):
        stamps = {}
        try:
            for cam in cameras:
                stamps[cam] = np.load(root / ep / f"{cam}-rgb-timestamp.npy").astype(float)
        except Exception as e:
            print(f"  {ep:<34} timestamps unreadable: {type(e).__name__}")
            continue
        wrote = total = stubs = 0
        for r in (x for x in rows if x["episode"] == ep):
            grid = r["lo"] + np.arange(r["n_frames"]) / a.fps
            ok = np.ones(grid.size, bool)
            for t in stamps.values():
                i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
                ok &= np.minimum(np.abs(t[i] - grid),
                                 np.abs(t[i - 1] - grid)) <= MAX_CAM_STALENESS_S
            w = int(ok.sum())
            wrote += w
            total += grid.size
            stubs += w < a.n_action_steps
        meta = {}
        try:
            meta = json.loads((root / ep / "session_meta.json").read_text())
        except Exception:
            pass
        n = sum(1 for x in rows if x["episode"] == ep)
        flag = "  <-- REJECT" if stubs else ""
        print(f"  {ep:<34} {n:>4} {wrote:>8}/{total:<8} {100 * wrote / max(total, 1):>3.0f}%"
              f" {stubs:>6} {str(bool(meta.get('degraded'))):>9}{flag}")
    print()

    # ── 4c. the handoff / release pose ──────────────────────────────────────
    gated = [r for r in train
             if r["close_idx"] is not None
             and r["close_idx"] <= close_idx_gate(LEFT_CHUNK_FRAMES, a.fracs[-1])
             and r["start_xyz"] and r["grasp_xyz"]]
    if gated:
        S = np.array([r["start_xyz"] for r in gated], float)
        G = np.array([r["grasp_xyz"] for r in gated], float)
        D = S - G                                  # start relative to its own grasp
        print("HANDOFF / RELEASE POSE  — where the classical stack hands over to ACT")
        print(f"  over the {len(gated)} windows that pass the close-index gate\n")

        print("  A. window-start EE position, ABSOLUTE (base frame)")
        print(f"     centroid   [{S[:, 0].mean():+.3f} {S[:, 1].mean():+.3f} "
              f"{S[:, 2].mean():+.3f}] m")
        print(f"     per-axis range   x {S[:, 0].ptp():.3f}  y {S[:, 1].ptp():.3f}  "
              f"z {S[:, 2].ptp():.3f} m")
        e_fixed = np.linalg.norm(S - S.mean(0), axis=1)
        print(f"     error if ONE fixed pose is commanded: {dist(e_fixed, '{:.3f}')} m")
        print()

        print("  B. grasp EE position, ABSOLUTE — i.e. where the packets were")
        print(f"     per-axis range   x {G[:, 0].ptp():.3f}  y {G[:, 1].ptp():.3f}  "
              f"z {G[:, 2].ptp():.3f} m")
        print()

        print("  C. window start RELATIVE to its own grasp  (start - grasp)")
        for k, ax in enumerate("xyz"):
            print(f"     d{ax}  {dist(D[:, k], '{:+.3f}')} m")
        e_rel = np.linalg.norm(D - np.median(D, axis=0), axis=1)
        print(f"     |offset|  {dist(np.linalg.norm(D, axis=1), '{:.3f}')} m")
        print(f"     error if the MEDIAN offset is commanded off the detected packet:")
        print(f"       {dist(e_rel, '{:.3f}')} m")
        print()

        secs = np.array([r["close_idx"] for r in gated], float) / a.fps
        print("  D. seconds from that start to the jaw close")
        print(f"     {dist(secs, '{:.2f}')} s")
        print()
        print(f"  VERDICT  one fixed pose: p90 error {np.percentile(e_fixed, 90):.3f} m."
              f"   packet-relative: p90 error {np.percentile(e_rel, 90):.3f} m.")
        print(f"  median offset to command = [{np.median(D[:, 0]):+.3f} "
              f"{np.median(D[:, 1]):+.3f} {np.median(D[:, 2]):+.3f}] m off the packet.")
        print()

    # ── 5. horizons the data supports ───────────────────────────────────────
    n_frames = [r["n_frames"] for r in train]
    post = [r["post_idx"] for r in train]
    ci = [r["close_idx"] for r in train]
    print("HORIZONS THE DATA SUPPORTS")
    print(f"  window length      {dist(n_frames)} frames "
          f"= {pct(n_frames, 50) / a.fps:.2f} s median")
    print(f"  approach (to close) {dist(ci)} frames "
          f"= {pct(ci, 50) / a.fps:.2f} s median")
    print(f"  lift (after close) {dist(post)} frames "
          f"= {pct(post, 50) / a.fps:.2f} s median")
    short = sum(1 for r in train if r["short"])
    print(f"  windows shorter than MIN_WINDOW_FRAMES={MIN_WINDOW_FRAMES}: {short}")
    ns = a.n_action_steps
    print(f"  replans per window at n_action_steps={ns}: "
          f"{dist([n / ns for n in n_frames], '{:.1f}')}")
    print(f"  windows whose LIFT is shorter than one commit ({ns} frames): "
          f"{sum(1 for p in post if p < ns)} / {len(post)}")
    print(f"  windows whose APPROACH is shorter than one commit ({ns} frames): "
          f"{sum(1 for c in ci if c < ns)} / {len(ci)}")
    print()
    print("  chunk_size must cover the behaviour a single plan has to express;")
    print("  n_action_steps only bounds how long the policy runs blind. They are")
    print("  independent numbers and the runtime already overrides the second.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"root": str(root), "pre_s": a.pre_s, "post_s": a.post_s, "fps": a.fps,
             "held_out": sorted(held_out), "windows": rows,
             "sweep": {str(k): v for k, v in sweeps.items()}}, indent=1))
        print(f"\nper-window records -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
