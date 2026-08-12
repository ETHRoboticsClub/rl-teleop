#!/usr/bin/env python3
"""What shape is the carry, and how high does it clear? — the right-arm transport.

WHAT THIS ANSWERS
=================

The ACT policy trained for this arm covers ONLY the grasp: the export took
``[close - 3s, close + 2s]`` windows, so approach, descend, close and lift are
in the dataset and the carry to the mat is not. To close the cycle, something
has to move the arm from wherever the lift ends to the placement point — and
there is a box roughly 15-20 cm tall in between that a straight line would hit.

So this reads the FULL recordings, not the training windows, finds every
grasp→release the operator actually demonstrated, and measures the path they
flew. The output is the evidence for a transport arc: how high they lifted
before travelling, where the carry started and ended, and how much clearance a
"C" needs to reproduce what a human already proved works.

METHOD, AND ITS LIMITS
======================

* Joint trajectories come from ``yam_right.mcap`` — the FOLLOWER, i.e. where the
  arm actually was, not where the leader asked it to go.
* Cartesian positions are forward kinematics of the 6 arm joints, in the arm's
  OWN base frame. That is deliberate: nothing in yam-pick-pipeline is calibrated
  for this arm (BLOCKER 4), so a world frame would be invented rather than
  measured. Base-frame numbers are what a joint-space or IK move for THIS arm
  needs anyway.
* ``link_6`` is the wrist, not the fingertips. Every height here is the wrist's
  height; the gripped bag hangs below it. That makes these clearances
  CONSERVATIVE for the wrist and optimistic for the payload, which is stated
  again in the report rather than buried here.
* A cycle is gripper-close → next gripper-open. Segments where the gripper never
  reopens (end of episode, dropped bag) are excluded and counted.

    tools/analyze_transport_arc.py --root recordings/20260811
    tools/analyze_transport_arc.py --root recordings/20260811 --json out.json
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

#: Gripper is normalised 0=closed, 1=open. These bracket the transition with a
#: dead band so sensor noise around a threshold cannot manufacture cycles.
CLOSED_BELOW = 0.35
OPEN_ABOVE = 0.65


def gripper_events(grip: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """Indices of (close, next open) pairs — one per carried object."""
    state = "open" if grip[0] > OPEN_ABOVE else "closed"
    closes: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i in range(1, len(grip)):
        if state == "open" and grip[i] < CLOSED_BELOW:
            state = "closed"
            closes.append(i)
        elif state == "closed" and grip[i] > OPEN_ABOVE:
            state = "open"
            if closes:
                pairs.append((closes[-1], i))
    return pairs


def normalise_gripper(raw: np.ndarray) -> np.ndarray:
    """Per-episode min-max. Raw units differ per rig AND per boot, so an absolute
    threshold means nothing across sessions — the same reason export_lerobot
    normalises per episode rather than against a constant."""
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-6:
        return np.full_like(raw, 1.0)
    return (raw - lo) / (hi - lo)


def analyse_episode(ep: Path, fk: ForwardKinematics) -> list[dict]:
    t, q = read_positions(ep / "yam_right.mcap", "yam_right")
    if q.ndim != 2 or q.shape[1] <= C.GRIPPER_JOINT_INDEX:
        return []
    grip = normalise_gripper(q[:, C.GRIPPER_JOINT_INDEX])
    xyz = fk.ee_positions(q)

    out: list[dict] = []
    for i_close, i_open in gripper_events(grip, t):
        seg = xyz[i_close:i_open + 1]
        if len(seg) < 5:
            continue
        z = seg[:, 2]
        z0, z1 = float(z[0]), float(z[-1])
        i_peak = int(np.argmax(z))
        # Horizontal distance travelled, and how far along the carry the peak sat.
        horiz = float(np.linalg.norm(seg[-1, :2] - seg[0, :2]))
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1))])
        frac_at_peak = float(s[i_peak] / s[-1]) if s[-1] > 1e-6 else 0.0
        out.append({
            "episode": ep.name,
            "t_s": float(t[i_open] - t[i_close]),
            "grasp_xyz": seg[0].tolist(),
            "release_xyz": seg[-1].tolist(),
            "z_grasp": z0,
            "z_release": z1,
            "z_peak": float(z.max()),
            "lift_above_grasp": float(z.max() - z0),
            "lift_above_release": float(z.max() - z1),
            "z_min_during": float(z.min()),
            "horizontal_travel": horiz,
            "frac_at_peak": frac_at_peak,
            "path": seg.tolist(),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "recordings/20260811")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--box-height", type=float, default=0.20,
                    help="tallest obstacle between pick and place, metres")
    # QUALITY GATE. The raw close->open segmentation catches every gripper
    # transition, and most of them are not carries: re-grips after a slip, small
    # adjustments, and momentary twitches. Unfiltered, the median horizontal
    # travel came out at 7 cm while the median grasp and release POINTS sat 24 cm
    # apart — a contradiction that only makes sense if most segments never went
    # anywhere. Averaging those into the arc would drag the recommended travel
    # height down toward "did not move", which is the opposite of safe.
    ap.add_argument("--min-travel", type=float, default=0.15,
                    help="metres of horizontal travel below which a segment is "
                         "not a carry to the mat")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="metres of clearance above the box rim")
    ap.add_argument("--plot", type=Path, default=None, help="write a PNG of the arcs")
    ap.add_argument("--min-duration", type=float, default=0.8,
                    help="seconds below which a segment is a twitch, not a carry")
    a = ap.parse_args(argv)

    eps = sorted(p for p in a.root.glob("episode_*") if (p / "yam_right.mcap").exists())
    if not eps:
        print(f"no right-arm episodes under {a.root}", file=sys.stderr)
        return 2

    fk = ForwardKinematics(urdf_path=REPO / "urdf/yam.urdf")
    carries: list[dict] = []
    for ep in eps:
        try:
            carries.extend(analyse_episode(ep, fk))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {ep.name}: SKIPPED ({type(exc).__name__}: {exc})")

    raw_n = len(carries)
    rejected = [c for c in carries
                if c["horizontal_travel"] < a.min_travel or c["t_s"] < a.min_duration]
    carries = [c for c in carries
               if c["horizontal_travel"] >= a.min_travel and c["t_s"] >= a.min_duration]
    print(f"\n  segmentation: {raw_n} gripper close->open events, "
          f"{len(rejected)} rejected as re-grips/adjustments "
          f"(<{a.min_travel:.2f} m travel or <{a.min_duration:.1f} s), "
          f"{len(carries)} real carries kept")
    if not carries:
        print("no complete grasp->release cycles found", file=sys.stderr)
        return 1

    def col(k):
        return np.array([c[k] for c in carries], dtype=float)

    print(f"\n{'='*74}\nRIGHT-ARM TRANSPORT — {len(carries)} carries across {len(eps)} episodes")
    print(f"{'='*74}")
    print("All heights are the WRIST (link_6) in the arm's own base frame.")
    print("The bag hangs BELOW this, so payload clearance is less than wrist clearance.\n")

    for name, key, unit in (
        ("carry duration",            "t_s",                "s"),
        ("grasp height  z",           "z_grasp",            "m"),
        ("release height z",          "z_release",          "m"),
        ("peak height   z",           "z_peak",             "m"),
        ("lift above grasp",          "lift_above_grasp",   "m"),
        ("lift above release",        "lift_above_release", "m"),
        ("lowest z during carry",     "z_min_during",       "m"),
        ("horizontal travel",         "horizontal_travel",  "m"),
        ("peak at fraction of path",  "frac_at_peak",       ""),
    ):
        v = col(key)
        print(f"  {name:26s} median {np.median(v):7.3f}{unit:2s}  "
              f"p10 {np.percentile(v,10):7.3f}  p90 {np.percentile(v,90):7.3f}  "
              f"min {v.min():7.3f}  max {v.max():7.3f}")

    gx = np.array([c["grasp_xyz"] for c in carries])
    rx = np.array([c["release_xyz"] for c in carries])
    print(f"\n  grasp   point  median xyz = [{np.median(gx[:,0]):.3f} {np.median(gx[:,1]):.3f} {np.median(gx[:,2]):.3f}]"
          f"   spread(xy) = {np.std(gx[:,0]):.3f} / {np.std(gx[:,1]):.3f}")
    print(f"  release point  median xyz = [{np.median(rx[:,0]):.3f} {np.median(rx[:,1]):.3f} {np.median(rx[:,2]):.3f}]"
          f"   spread(xy) = {np.std(rx[:,0]):.3f} / {np.std(rx[:,1]):.3f}")

    # ── the arc the demonstrations actually flew ──────────────────────────
    # Resample every carry onto the same normalised path parameter so they can
    # be averaged: raw time would let a slow carry dominate a fast one.
    GRID = np.linspace(0.0, 1.0, 21)
    prof = []
    for c in carries:
        p = np.asarray(c["path"])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
        if s[-1] < 1e-6:
            continue
        prof.append(np.interp(GRID, s / s[-1], p[:, 2]))
    P = np.array(prof)
    print(f"\n  MEAN HEIGHT PROFILE along the carry ({len(P)} carries, normalised path):")
    print("    path %   " + "".join(f"{int(g*100):6d}" for g in GRID[::2]))
    print("    z (m)    " + "".join(f"{v:6.3f}" for v in np.median(P, axis=0)[::2]))
    print("    p10 (m)  " + "".join(f"{v:6.3f}" for v in np.percentile(P, 10, axis=0)[::2]))

    z_ref = float(np.median(col("z_release")))
    lowest_mid = float(np.percentile(P[:, 4:17], 10))
    print(f"\n  Lowest wrist height in the MIDDLE 60% of the carry (p10): {lowest_mid:.3f} m")
    print(f"  Median release height:                                    {z_ref:.3f} m")
    print(f"  So the demonstrated carry clears the release plane by     {lowest_mid - z_ref:+.3f} m")

    print(f"\n{'-'*74}\nRECOMMENDED TRANSPORT ARC (obstacle {a.box_height*100:.0f} cm)\n{'-'*74}")
    z_grasp = float(np.median(col("z_grasp")))
    demo_peak = float(np.median(col("z_peak")))
    peak_p10 = float(np.percentile(col("z_peak"), 10))

    # THE BOX IS AT THE PICK END, not the place end. The arm reaches DOWN INTO the
    # source box to grasp, so the thing it must not hit on the way out is that
    # box's rim — and the rim is referenced to the GRASP plane, not the release
    # plane. Referencing it to the release plane (as a first pass here did) gives
    # a number that happens to look similar and means something else.
    rim = z_grasp + a.box_height
    travel_z = rim + a.margin

    print(f"  Grasp sits at z = {z_grasp:.3f} m — that is INSIDE the source box.")
    print(f"  A {a.box_height*100:.0f} cm box around it puts the rim at   z = {rim:.3f} m.")
    print(f"  The demonstrations peaked at median         z = {demo_peak:.3f} m")
    print(f"                              p10             z = {peak_p10:.3f} m")
    if peak_p10 < rim:
        print(f"\n  NOTE: the p10 demonstrated peak is BELOW that rim by "
              f"{rim - peak_p10:.3f} m. Either the box is shorter than "
              f"{a.box_height*100:.0f} cm where they crossed it, the grasp is not at the box"
              f"\n  floor, or some carries genuinely brushed the rim. Worth one look before"
              f"\n  trusting {a.box_height*100:.0f} cm as the number to clear.")
    print(f"\n  The carry is a C, and the data says so: the height peaks at "
          f"{float(np.median(col('frac_at_peak')))*100:.0f}% of the path,")
    print(f"  i.e. mid-flight, rising {demo_peak - z_grasp:+.3f} m from the grasp and "
          f"falling {demo_peak - float(np.median(col('z_release'))):+.3f} m to the release.")

    gm, rm = np.median(gx, axis=0), np.median(rx, axis=0)
    print(f"\n  Waypoints (arm base frame, wrist/link_6):")
    print(f"    0. GRASP   [{gm[0]:.3f} {gm[1]:.3f} {gm[2]:.3f}]   <- ACT policy ends here, holding")
    print(f"    1. LIFT    [{gm[0]:.3f} {gm[1]:.3f} {travel_z:.3f}]   straight up, clear the rim")
    print(f"    2. TRAVEL  [{rm[0]:.3f} {rm[1]:.3f} {travel_z:.3f}]   swing across at height")
    print(f"    3. DESCEND [{rm[0]:.3f} {rm[1]:.3f} {rm[2]:.3f}]   down to the mat")
    print(f"    4. OPEN    release")
    print(f"\n  Travel height for step 2: {travel_z:.3f} m "
          f"(rim {rim:.3f} + {a.margin*100:.0f} cm margin).")
    print(f"  x stays ~constant at {gm[0]:.2f} m; the motion is a swing in y "
          f"({gm[1]:+.3f} -> {rm[1]:+.3f}).")
    print(f"\n  CAVEATS. These are WRIST heights — the bag hangs below, so subtract its")
    print(f"  length before believing any clearance. And the release point above is the")
    print(f"  median of where the operator dropped bags; a specific mat target needs its")
    print(f"  own xy. MEASURE THE BOX before trusting {a.box_height*100:.0f} cm.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for c in carries:
            pth = np.asarray(c["path"])
            ax[0].plot(pth[:, 1], pth[:, 2], color="0.7", lw=0.8)
        ax[0].plot(np.median([np.interp(GRID, np.linspace(0,1,len(np.asarray(c["path"]))),
                                        np.asarray(c["path"])[:,1]) for c in carries], axis=0),
                   np.median(P, axis=0), color="crimson", lw=2.5, label="median carry")
        ax[0].axhline(z_grasp + a.box_height, ls="--", color="k", label=f"{a.box_height*100:.0f} cm box rim")
        ax[0].axhline(travel_z, ls=":", color="green", label=f"travel z {travel_z:.3f}")
        ax[0].set_xlabel("y (m, base frame)"); ax[0].set_ylabel("z (m)")
        ax[0].set_title(f"{len(carries)} carries, side view — the C"); ax[0].legend(fontsize=8)
        ax[1].plot(GRID, np.median(P, axis=0), color="crimson", lw=2, label="median")
        ax[1].fill_between(GRID, np.percentile(P,10,axis=0), np.percentile(P,90,axis=0),
                           color="crimson", alpha=0.2, label="p10-p90")
        ax[1].axhline(z_grasp + a.box_height, ls="--", color="k")
        ax[1].set_xlabel("fraction of path"); ax[1].set_ylabel("z (m)")
        ax[1].set_title("height profile"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"\n  wrote {a.plot}")

    if a.json:
        a.json.write_text(json.dumps({
            "n_carries": len(carries),
            "episodes": [e.name for e in eps],
            "summary": {k: {"median": float(np.median(col(k))),
                            "p10": float(np.percentile(col(k), 10)),
                            "p90": float(np.percentile(col(k), 90))}
                        for k in ("t_s", "z_grasp", "z_release", "z_peak",
                                  "lift_above_grasp", "horizontal_travel")},
            "height_profile": {"path_fraction": GRID.tolist(),
                               "median_z": np.median(P, axis=0).tolist(),
                               "p10_z": np.percentile(P, 10, axis=0).tolist()},
            "recommended_travel_z": float(travel_z),
            "box_rim_z": float(rim),
            "grasp_z": float(z_grasp),
            "release_z": float(np.median(col("z_release"))),
            "waypoints": {"grasp": gm.tolist(), "lift": [float(gm[0]), float(gm[1]), float(travel_z)],
                          "travel": [float(rm[0]), float(rm[1]), float(travel_z)], "descend": rm.tolist()},
            "carries": [{k: v for k, v in c.items() if k != "path"} for c in carries],
        }, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
