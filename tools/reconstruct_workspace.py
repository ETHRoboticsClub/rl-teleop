#!/usr/bin/env python3
"""Reverse-engineer the cell geometry from what the operator actually did.

THE IDEA
========

Nothing in yam-pick-pipeline is calibrated for the right arm (BLOCKER 4), so
there is no measured model of where the source box and the mat are. But 8
recordings of a human doing the task contain that information implicitly:

  * every GRASP happened inside the source box, so the grasp cluster IS the box
    footprint, and the grasp heights are its floor;
  * every RELEASE happened over the mat, so the release cluster IS the drop zone;
  * and — the useful part — the operator never crashed. So every height at which
    the arm moved SIDEWAYS over the box footprint is a height that is
    demonstrably clear of the rim. The lowest such height is an empirical
    ceiling on how tall the box can be.

That last inference is what turns "roughly 15-20 cm" into a number derived from
the rig instead of from memory. It is a bound, not a measurement: the true rim
is at or below it. That direction is the safe one — planning above a bound that
is too high costs a little time; planning below a rim costs a box.

METHOD
======

Footprints are robust percentiles (p2/p98) of each cluster rather than min/max,
so one stray sample cannot inflate a box by 10 cm.

The crossing height uses only samples that are (a) horizontally inside the box
footprint, (b) moving laterally faster than ``--lateral-speed``, and (c) not part
of the vertical plunge in or out. Condition (b) is what separates "travelling
across the box" from "descending into it", where the arm moves almost purely in
z and low heights prove nothing about the rim.

    tools/reconstruct_workspace.py --root recordings/20260811 --plot ws.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from robots_realtime.labeling import constants as C          # noqa: E402
from robots_realtime.labeling.fk import ForwardKinematics    # noqa: E402
from robots_realtime.labeling.mcap_io import read_positions  # noqa: E402

CLOSED_BELOW, OPEN_ABOVE = 0.35, 0.65


def norm_grip(raw: np.ndarray) -> np.ndarray:
    lo, hi = float(raw.min()), float(raw.max())
    return np.full_like(raw, 1.0) if hi - lo < 1e-6 else (raw - lo) / (hi - lo)


def load(root: Path, fk: ForwardKinematics):
    """-> list of (xyz, grip, t) per episode, plus the grasp/release points."""
    eps, grasps, releases, skipped, carry_paths = [], [], [], [], []
    for ep in sorted(root.glob("episode_*")):
        f = ep / "yam_right.mcap"
        if not f.exists():
            continue
        try:
            t, q = read_positions(f, "yam_right")
        except Exception as exc:                                   # noqa: BLE001
            skipped.append((ep.name, f"{type(exc).__name__}"))
            continue
        if q.ndim != 2 or q.shape[1] <= C.GRIPPER_JOINT_INDEX:
            continue
        g = norm_grip(q[:, C.GRIPPER_JOINT_INDEX])
        xyz = fk.ee_positions(q)
        eps.append((xyz, g, t, ep.name))

        state = "open" if g[0] > OPEN_ABOVE else "closed"
        last_close = None
        for i in range(1, len(g)):
            if state == "open" and g[i] < CLOSED_BELOW:
                state, last_close = "closed", i
            elif state == "closed" and g[i] > OPEN_ABOVE:
                state = "open"
                if last_close is not None:
                    # Only real carries define the two zones. A re-grip that
                    # never went anywhere says nothing about where the mat is.
                    d = xyz[i, :2] - xyz[last_close, :2]
                    # DIRECTION MATTERS. The task is box -> mat, which is +y. Four
                    # of the 37 carries ran the other way (dy = -0.38, -0.30,
                    # -0.30, -0.28): the operator putting a bag BACK, or
                    # repositioning. Those are a different action, and including
                    # them put 4 grasps on the mat and 4 releases in the box —
                    # which smeared the two clusters into each other and made the
                    # box footprint overlap the drop zone almost completely. The
                    # reconstruction then "measured" a box -3 cm tall.
                    if np.linalg.norm(d) >= 0.15 and d[1] > 0:
                        grasps.append(xyz[last_close])
                        releases.append(xyz[i])
                        carry_paths.append(xyz[last_close:i + 1])
    return eps, np.array(grasps), np.array(releases), skipped, carry_paths


def footprint(pts: np.ndarray, lo=2, hi=98) -> dict:
    return {
        "x": [float(np.percentile(pts[:, 0], lo)), float(np.percentile(pts[:, 0], hi))],
        "y": [float(np.percentile(pts[:, 1], lo)), float(np.percentile(pts[:, 1], hi))],
        "z_median": float(np.median(pts[:, 2])),
        "z_p2": float(np.percentile(pts[:, 2], 2)),
        "n": int(len(pts)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "recordings/20260811")
    ap.add_argument("--lateral-speed", type=float, default=0.05,
                    help="m/s of horizontal motion that counts as 'travelling'")
    ap.add_argument("--pad", type=float, default=0.03,
                    help="metres to expand the box footprint by when testing "
                         "whether the arm is over it")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    fk = ForwardKinematics(urdf_path=REPO / "urdf/yam.urdf")
    eps, G, R, skipped, paths = load(a.root, fk)
    for name, why in skipped:
        print(f"  {name}: SKIPPED ({why})")
    if len(G) < 5:
        print("not enough carries to reconstruct anything", file=sys.stderr)
        return 1

    box, mat = footprint(G), footprint(R)
    bw = (box["x"][1] - box["x"][0]) * 100
    bd = (box["y"][1] - box["y"][0]) * 100
    mw = (mat["x"][1] - mat["x"][0]) * 100
    md = (mat["y"][1] - mat["y"][0]) * 100

    print(f"\n{'='*74}\nWORKSPACE RECONSTRUCTED FROM {len(G)} CARRIES "
          f"({len(eps)} episodes)\n{'='*74}")
    print("Arm base frame, wrist (link_6). +y is the direction of travel.\n")
    print(f"  SOURCE BOX  (the grasp cluster)")
    print(f"    x {box['x'][0]:+.3f} .. {box['x'][1]:+.3f}   y {box['y'][0]:+.3f} .. {box['y'][1]:+.3f}")
    print(f"    footprint  {bw:.0f} x {bd:.0f} cm      floor (grasp z) {box['z_median']:.3f} m")
    print(f"  DROP ZONE   (the release cluster)")
    print(f"    x {mat['x'][0]:+.3f} .. {mat['x'][1]:+.3f}   y {mat['y'][0]:+.3f} .. {mat['y'][1]:+.3f}")
    print(f"    footprint  {mw:.0f} x {md:.0f} cm      surface (release z) {mat['z_median']:.3f} m")
    sep = float(np.median(R[:, 1]) - np.median(G[:, 1]))
    print(f"\n  The two zones are separated by {sep*100:.0f} cm in y "
          f"(box on the -y side, drop zone on the +y side).")

    # ── the empirical rim, measured where it physically matters ──────────
    #
    # NOT "the lowest height anywhere over the box footprint": the arm descends
    # INTO the box to grasp, so low heights over that footprint are the inside of
    # the box and prove nothing about the rim. The rim is what the arm must clear
    # when it LEAVES the box toward the mat, so the honest measurement is the
    # height at the moment each carry crosses the box's +y edge.
    edge_y = box["y"][1]
    cross_z, cross_x = [], []
    for pth in paths:
        yy, zz = pth[:, 1], pth[:, 2]
        idx = np.where((yy[:-1] <= edge_y) & (yy[1:] > edge_y))[0]
        if not len(idx):
            continue
        i = int(idx[0])
        f = (edge_y - yy[i]) / max(yy[i + 1] - yy[i], 1e-9)
        cross_z.append(float(zz[i] + f * (zz[i + 1] - zz[i])))
        cross_x.append(float(pth[i, 0]))
    Z = np.array(cross_z)
    if len(Z) < 5:
        print("\n  too few edge crossings to bound the rim")
        return 1

    print(f"\n  CLEARING THE BOX EDGE (y = {edge_y:+.3f}) — {len(Z)} carries measured there")
    print(f"    lowest crossing  {Z.min():.3f} m")
    for p in (10, 25, 50, 90):
        print(f"    p{p:<3d} crossing    {np.percentile(Z, p):.3f} m")
    rim_bound = float(Z.min())
    box_h = rim_bound - box["z_median"]
    print(f"\n  The LOWEST any carry crossed that edge is {rim_bound:.3f} m, and none of")
    print(f"  them hit anything. Grasp floor is {box['z_median']:.3f} m, so relative to the")
    print(f"  bag the box wall is AT MOST {box_h*100:.0f} cm — an upper bound, not a")
    print(f"  measurement: the true rim is at or below it.")
    if 0.13 <= box_h <= 0.24:
        print(f"  That is consistent with your 15-20 cm estimate.")

    safe = float(max(np.percentile(Z, 25), rim_bound + 0.05))
    print(f"\n  SAFE TRAVEL HEIGHT: {safe:.3f} m")
    print(f"    = max(rim bound + 5 cm, the 25th percentile of what was actually flown)")
    print(f"    Median carry peaked at {float(np.median([p[:,2].max() for p in paths])):.3f} m, so this sits")
    print(f"    comfortably inside the demonstrated envelope.")

    print(f"\n  SAFE ZONE for the transport planner (arm base frame):")
    print(f"    while y <= {edge_y:+.3f} (over/near the box)  ->  keep z >= {safe:.3f}")
    print(f"    once y > {edge_y:+.3f}  ->  free to descend toward the drop surface {mat['z_median']:.3f}")
    print(f"    x over the box spans {box['x'][0]:.3f}..{box['x'][1]:.3f}; crossings happened at "
          f"x = {np.median(cross_x):.3f} median")
    print(f"\n  WRIST HEIGHTS. The bag hangs below the wrist, so subtract its length")
    print(f"  before trusting any of these as payload clearance.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        for xyz, g, t, _ in eps:
            ax[0].plot(xyz[:, 1], xyz[:, 0], color="0.85", lw=0.5)
        ax[0].scatter(G[:, 1], G[:, 0], s=26, c="tab:red", label=f"grasps ({len(G)})", zorder=3)
        ax[0].scatter(R[:, 1], R[:, 0], s=26, c="tab:blue", label=f"releases ({len(R)})", zorder=3)
        ax[0].add_patch(Rectangle((box["y"][0], box["x"][0]), box["y"][1]-box["y"][0],
                                  box["x"][1]-box["x"][0], fill=False, ec="tab:red", lw=2))
        ax[0].add_patch(Rectangle((mat["y"][0], mat["x"][0]), mat["y"][1]-mat["y"][0],
                                  mat["x"][1]-mat["x"][0], fill=False, ec="tab:blue", lw=2))
        ax[0].set_xlabel("y (m)"); ax[0].set_ylabel("x (m)")
        ax[0].set_title(f"top view — box {bw:.0f}x{bd:.0f} cm, drop {mw:.0f}x{md:.0f} cm")
        ax[0].legend(fontsize=8); ax[0].set_aspect("equal")

        for xyz, g, t, _ in eps:
            ax[1].plot(xyz[:, 1], xyz[:, 2], color="0.85", lw=0.5)
        ax[1].axhline(rim_bound, ls="--", color="k", label=f"rim upper bound {rim_bound:.3f}")
        ax[1].axhline(safe, ls=":", color="green", label=f"safe travel {safe:.3f}")
        ax[1].axvspan(box["y"][0], box["y"][1], color="tab:red", alpha=0.12, label="box footprint")
        ax[1].axvspan(mat["y"][0], mat["y"][1], color="tab:blue", alpha=0.12, label="drop zone")
        ax[1].set_xlabel("y (m)"); ax[1].set_ylabel("z (m)")
        ax[1].set_title("side view — everything the arm did"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"\n  wrote {a.plot}")

    if a.json:
        a.json.write_text(json.dumps({
            "frame": "right arm base_link, wrist link_6",
            "n_carries": len(G), "episodes": len(eps),
            "source_box": box, "drop_zone": mat,
            "box_footprint_cm": [bw, bd], "drop_footprint_cm": [mw, md],
            "rim_upper_bound_z": rim_bound,
            "edge_y": float(edge_y),
            "crossing_z_min": float(Z.min()),
            "box_height_upper_bound_m": box_h,
            "safe_travel_z": float(safe),
            "crossing_height_percentiles": {str(p): float(np.percentile(Z, p)) for p in (10, 25, 50, 90)},
        }, indent=2))
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
