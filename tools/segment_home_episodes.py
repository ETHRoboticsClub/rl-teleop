#!/usr/bin/env python3
"""Cut a home→home teleop take into training windows for export_lerobot --windows.

Protocol this serves (2026-09-14, right arm): the operator holds a raised home pose
above the source box, closes the jaws to push/isolate packets (NOT a grasp), grasps
exactly one, drops it in the next box and returns to home. One take holds many of
these. A grasp-window export would cut a window around every nudge, so the corpus
must be exported as whole home→home segments — this tool finds them.

    windows[episode] = [[t_abs_start, t_abs_end], ...]      → export_lerobot --windows
    holdout[episode] = [...]                                → whole segments, never trained

Home = the pose held in the first second of each take (median), checked against
the session median so a take that started somewhere else is flagged, not cut wrong.
A segment is the stretch between two dwells (>= --dwell-s within --thr rad of home),
padded by --pad-s on both sides so the policy sees the stop at home.

Per segment it reports gripper closes, releases split by base yaw (drop site vs
over the box), a category, and the CAMERA FRAME YIELD on the 30 fps grid — the
gate the export enforces nowhere (act-training skill §3). Segments under
--min-yield are excluded with the number printed.

Phase-tagged takes (2026-09-17): training_experiment/scripts/record_phase.sh stamps the session
instruction as "phase:<make_space|isolate|flip|pick>" (session_meta.json). Every home→home segment
of such a take IS that phase, so all categories are exported (yield gate and operator exclude
list still apply) and the segment carries "phase_tag" for phase_split.py.

Categories → exported? (clean_pick, multi_pick yes; the rest no — unless the take is phase-tagged):
  clean_pick       nudges then one grasp+drop, home
  multi_pick       two or more grasp+drops before home (kept whole)
  drop_then_inbox  grasp+drop, then more nudging before home
  no_drop          nudging only, nothing taken
  no_grasp         jaws never closed
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robots_realtime.labeling.mcap_io import read_positions  # noqa: E402

FPS = 30
STALE_S = 2.0 / FPS            # export_lerobot.MAX_CAM_STALENESS_S
EXPORT_CATS = ("clean_pick", "multi_pick")


def phase_tag(ep: Path) -> str | None:
    """'phase:<name>' stamped as the take's instruction by record_phase.sh -> '<name>', else None."""
    try:
        ins = str(json.loads((ep / "session_meta.json").read_text()).get("instruction") or "").strip()
    except Exception:
        return None
    return (ins[len("phase:"):].strip() or None) if ins.startswith("phase:") else None


def cycle_labels(ep: Path) -> list[dict]:
    """<episode>/cycle_labels.json written live by training_experiment/tools/phase_recorder.py:
    [{k, t_start, t_end, phase, bad}] -- the operator's per-cycle phase and bad marks."""
    try:
        return list(json.loads((ep / "cycle_labels.json").read_text()).get("cycles") or [])
    except Exception:
        return []


def cycle_for(cycles: list[dict], lo: float, hi: float) -> dict | None:
    """The live cycle overlapping [lo, hi] the most (>= half of the segment), else None."""
    best, ov = None, 0.0
    for c in cycles:
        o = min(hi, float(c["t_end"])) - max(lo, float(c["t_start"]))
        if o > ov:
            best, ov = c, o
    return best if best is not None and ov >= 0.5 * (hi - lo) else None


def dwells(t, d, thr, dwell_s):
    at = d < thr; iv = []; i = 0; n = len(t)
    while i < n:
        if at[i]:
            j = i
            while j < n and at[j]:
                j += 1
            if t[j - 1] - t[i] >= dwell_s:
                iv.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return iv


def frame_yield(stamps: np.ndarray, lo: float, hi: float) -> float:
    grid = lo + np.arange(int((hi - lo) * FPS)) / FPS
    if len(grid) == 0:
        return 0.0
    idx = np.clip(np.searchsorted(stamps, grid), 1, len(stamps) - 1)
    near = np.minimum(np.abs(stamps[idx] - grid), np.abs(stamps[idx - 1] - grid))
    return float((near <= STALE_S).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="recordings/<date>")
    ap.add_argument("--arm", default="right")
    ap.add_argument("--cameras", default="camera_top,camera_right")
    ap.add_argument("--dwell-s", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.20, help="rad, L2 over 6 arm joints")
    ap.add_argument("--pad-s", type=float, default=0.5)
    ap.add_argument("--drop-yaw", type=float, default=-0.2,
                    help="base yaw below this at a release = drop at the site")
    ap.add_argument("--min-yield", type=float, default=0.95)
    ap.add_argument("--min-s", type=float, default=4.0)
    ap.add_argument("--holdout-every", type=int, default=6,
                    help="every Nth exported segment goes to holdout (0 = none)")
    ap.add_argument("--home-drift-max", type=float, default=0.30)
    ap.add_argument("--out", default=None, help="default <root>/home_segments.json")
    ap.add_argument("--exclude", default=None,
                    help="JSON list of {episode, seg} the operator reviewed OUT "
                         "(default <root>/home_segments.exclude.json if it exists)")
    a = ap.parse_args()
    root = Path(a.root); cams = a.cameras.split(",")
    excl_path = Path(a.exclude) if a.exclude else root / "home_segments.exclude.json"
    excluded_ids = set()
    if excl_path.exists():
        for e in json.loads(excl_path.read_text()):
            excluded_ids.add((str(e["episode"]), int(e["seg"])))
        print(f"operator exclude list: {len(excluded_ids)} segments from {excl_path}")
    eps = sorted(p for p in root.glob("episode_*") if p.is_dir() and (p / f"yam_{a.arm}.mcap").exists())
    if not eps:
        sys.exit(f"no episode_* with yam_{a.arm}.mcap under {root}")

    loaded = []
    for ep in eps:
        t, p = read_positions(ep / f"yam_{a.arm}.mcap", f"yam_{a.arm}")
        t = np.asarray(t, float); p = np.asarray(p, float)
        if len(t) < FPS * 2:
            print(f"{ep.name}: only {len(t)} samples — skipped"); continue
        loaded.append((ep, t, p, np.median(p[t < t[0] + 1.0, :6], axis=0)))
    home_session = np.median(np.stack([h for *_, h in loaded]), axis=0)
    segs, windows, holdout, k_export = [], {}, {}, 0
    for ep, t, p, home_ep in loaded:
        drift = float(np.linalg.norm(home_ep - home_session))
        if drift > a.home_drift_max:
            print(f"{ep.name}: start pose {drift:.2f} rad from the session home — NOT cut (re-check the take)")
            continue
        tag = phase_tag(ep); live = cycle_labels(ep)
        q, g = p[:, :6], p[:, 6]
        d = np.linalg.norm(q - home_ep, axis=1)
        iv = dwells(t, d, a.thr, a.dwell_s)
        stamps = {c: np.load(ep / f"{c}-rgb-timestamp.npy").astype(float) for c in cams}
        closed = g < 0.3
        closes = np.where((~closed[:-1]) & closed[1:])[0]
        opens = np.where(closed[:-1] & (~closed[1:]))[0]
        for k in range(len(iv)):
            lo = iv[k][1]; hi = iv[k + 1][0] if k + 1 < len(iv) else None
            if hi is None or hi - lo < a.min_s:
                continue
            ev_o = [i for i in opens if lo <= t[i] <= hi]
            yaws = [float(q[i, 0]) for i in ev_o]
            drops = sum(y < a.drop_yaw for y in yaws)
            n_close = sum(1 for i in closes if lo <= t[i] <= hi)
            if n_close == 0: cat = "no_grasp"
            elif drops == 0: cat = "no_drop"
            elif drops == 1 and yaws[-1] < a.drop_yaw: cat = "clean_pick"
            elif drops >= 2: cat = "multi_pick"
            else: cat = "drop_then_inbox"
            # Pad into the surrounding dwells, but never past a dwell's midpoint:
            # the exporter's camera reader is forward-only, so two windows of one
            # episode must not overlap (cost a run on 2026-09-14 with 0.5 s pads
            # over 0.5 s dwells).
            pad_lo = min(a.pad_s, (iv[k][1] - iv[k][0]) / 2)
            pad_hi = min(a.pad_s, (iv[k + 1][1] - iv[k + 1][0]) / 2)
            wlo, whi = max(lo - pad_lo, float(t[0])), min(hi + pad_hi, float(t[-1]))
            yields = {c: frame_yield(stamps[c], wlo, whi) for c in cams}
            ok_yield = min(yields.values()) >= a.min_yield
            seg_no = len([s for s in segs if s["episode"] == ep.name]) + 1
            operator_out = (ep.name, seg_no) in excluded_ids
            seg_tag = tag; lc = cycle_for(live, wlo, whi) if live else None
            if lc is not None:                       # the operator's live marks win over the take's stamp
                seg_tag = lc.get("phase") or tag
                if lc.get("bad"):
                    operator_out = True
            exported = (cat in EXPORT_CATS or seg_tag is not None) and ok_yield and not operator_out
            split = None
            if exported:
                k_export += 1
                split = "holdout" if a.holdout_every and k_export % a.holdout_every == 0 else "train"
                (windows if split == "train" else holdout).setdefault(ep.name, []).append([wlo, whi])
            segs.append(dict(episode=ep.name, seg=seg_no,
                             t_abs=[wlo, whi], t_rel_s=[round(wlo - t[0], 1), round(whi - t[0], 1)],
                             dur_s=round(whi - wlo, 1), gripper_closes=n_close, site_drops=drops,
                             inbox_releases=len(yaws) - drops, category=cat, phase_tag=seg_tag,
                             frame_yield={c: round(v, 3) for c, v in yields.items()},
                             split=split or ("excluded:" + ("operator" if operator_out else "yield" if not ok_yield and (cat in EXPORT_CATS or seg_tag) else cat))))
        print(f"{ep.name}: {t[-1]-t[0]:.0f}s, home drift {drift:.2f} rad, {len(iv)} dwells, {'phase=' + tag + ', ' if tag else ''}{str(len(live)) + ' live cycles, ' if live else ''}"
              f"{sum(1 for s in segs if s['episode']==ep.name)} segments")

    cats = {}
    for s in segs:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    out = dict(
        made="segment_home_episodes.py", root=str(root), arm=a.arm,
        rule=dict(dwell_s=a.dwell_s, thr_rad=a.thr, pad_s=a.pad_s, drop_yaw=a.drop_yaw,
                  min_yield=a.min_yield, holdout_every=a.holdout_every),
        home_q_session=[round(float(x), 4) for x in home_session],
        summary=dict(segments=len(segs), categories=cats,
                     train=sum(len(v) for v in windows.values()),
                     holdout=sum(len(v) for v in holdout.values()),
                     excluded=sum(1 for s in segs if s["split"].startswith("excluded")),
                     packets_dropped_at_site=sum(s["site_drops"] for s in segs)),
        windows=windows, holdout=holdout, segments=segs)
    dst = Path(a.out) if a.out else root / "home_segments.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]))
    low = [s for s in segs if s["split"] == "excluded:yield"]
    for s in low:
        print(f"  LOW YIELD {s['episode']} seg {s['seg']}: {s['frame_yield']}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
