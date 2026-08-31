#!/usr/bin/env python3
"""Per-dataset-frame target points for ETHRC/yam_grasp_right_20260812.

Joins mapping.json (dataset frame -> camera_right frame index, from
map_dataset.py) with the SAM2 auto-label masks (study's work/ + this repo's
work/ for the events the study's long-hold filter missed) and writes
labels.json:

    [ { "episode_index": 0, "recording": ..., "t_close": ...,
        "points": [[x, y] | null, ...] },     # one per dataset frame_index
      ... ]

Label policy (documented for the PR):
  * pre-close frames: centroid of the tracked target mask at the NEAREST
    labeled camera frame within +-3 frames (the tracker ran at stride 2, so
    this bridges the stride but never invents a distant label);
  * frames at/after the seed frame (close-3, i.e. the close and the 2 s lift):
    the LAST tracked centroid, held constant — the packet is between the jaws
    and static relative to the wrist camera (the motion-still observation in
    the research doc); at inference the tracker equally keeps its last lock on
    the held packet;
  * frames with no tracked mask nearby (target not yet in the wrist camera's
    field of view during the early approach, or a lost track): NO DOT. The
    research doc measures ~67% approach coverage and attributes the remainder
    to the target being out of frame — drawing a dot there would fabricate a
    cue the deployed tracker cannot produce either.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_WORK = ("/home/tommaso/Desktop/kitting-v2-worktrees/yam-pick-pipeline/"
              "target-selector-offline/tools/target_selector_eval/work")
MY_WORK = os.path.join(HERE, "work")
RECORDINGS = "/home/tommaso/Desktop/kitting-v2/rl-teleop/recordings/20260811"

MATCH_TOL_S = 1.0     # keep grasp <-> auto-label event, close-time distance
NEAR_FRAMES = 3       # max |camera frame| distance to borrow a mask from


def load_events(recording: str, ts: np.ndarray) -> list[dict]:
    """Every auto-labeled event for one recording, with per-frame centroids."""
    events = []
    for src in (STUDY_WORK, MY_WORK):
        for p in sorted(glob.glob(os.path.join(src, recording, "event_*.npz"))):
            z = np.load(p)
            frames, masks = z["frames"], z["masks"]
            cents = {}
            for f, m in zip(frames, masks):
                ys, xs = np.nonzero(m)
                if len(xs):
                    cents[int(f)] = (float(xs.mean()), float(ys.mean()))
            events.append({
                "npz": p,
                "t_close": float(ts[int(z["close_frame"])]),
                "seed_frame": int(z["seed_frame"]),
                "cents": cents,
            })
    return events


def point_for(ev: dict, cam: int):
    """Target point for one camera frame index under the label policy above."""
    cents = ev["cents"]
    if not cents:
        return None
    if cam >= ev["seed_frame"]:
        last = max(f for f in cents)          # seed frame if tracked, else latest
        return cents[last]
    best = min(cents, key=lambda f: abs(f - cam))
    if abs(best - cam) <= NEAR_FRAMES:
        return cents[best]
    return None


def main() -> int:
    mapping = json.load(open(os.path.join(HERE, "mapping.json")))
    ts_cache, ev_cache = {}, {}
    out, dotted, total = [], 0, 0
    for ep in mapping:
        rec = ep["recording"]
        if rec not in ts_cache:
            ts_cache[rec] = np.load(os.path.join(
                RECORDINGS, rec, "camera_right-rgb-timestamp.npy")).astype(np.float64)
            ev_cache[rec] = load_events(rec, ts_cache[rec])
        events = ev_cache[rec]
        dts = [abs(e["t_close"] - ep["t_close"]) for e in events]
        i = int(np.argmin(dts))
        assert dts[i] <= MATCH_TOL_S, (
            f"episode {ep['episode_index']}: no auto-label event within "
            f"{MATCH_TOL_S}s of t_close={ep['t_close']:.2f} (best {dts[i]:.2f}s)")
        ev = events[i]
        pts = [point_for(ev, cam) for cam in ep["cam_idx"]]
        dotted += sum(p is not None for p in pts)
        total += len(pts)
        out.append({"episode_index": ep["episode_index"], "recording": rec,
                    "t_close": ep["t_close"], "event_npz": ev["npz"],
                    "points": [list(p) if p else None for p in pts]})
        n = sum(p is not None for p in pts)
        print(f"  ep {ep['episode_index']:>2} {rec} close={ep['t_close']:.2f}: "
              f"{n}/{len(pts)} frames dotted "
              f"(event {os.path.basename(ev['npz'])}, dt={dts[i]*1000:.0f}ms)")
    path = os.path.join(HERE, "labels.json")
    json.dump(out, open(path, "w"))
    print(f"\nlabel coverage: {dotted}/{total} frames "
          f"({100.0 * dotted / total:.1f}%) -> {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
