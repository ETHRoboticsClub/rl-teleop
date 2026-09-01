#!/usr/bin/env python3
"""Auto-label the dataset grasps the offline study's run did not cover.

The target-selector-offline study (tools/target_selector_eval/, its worktree)
already labeled every LONG-HOLD grasp event in the 20260811 + 20260814 right-arm
corpus — but its event detector (gripper-close hysteresis + >=50-frame hold)
misses some grasps the export marked successful: short holds and closes merged
by the hysteresis. This script labels exactly those, using the study's own
AutoLabeler (same SAM2 seed + backward-track pipeline), seeding at the export's
close time instead of the detector's.

Input: mapping-<dataset>.json from map_dataset.py — its per-episode t_close
values ARE the grasp list the dataset was cut around.

RUN WITH THE STUDY'S VENV (SAM2 lives only there):

    <target-selector-offline>/.venv-eval/bin/python \
        tools/targetdot/autolabel_missing.py --dataset 20260814

Output: work/<episode>/event_<close_frame>.npz next to this file, identical
format to the study's npz (frames, masks, seed_frame, close_frame, seed_points).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = ("/home/tommaso/Desktop/kitting-v2-worktrees/yam-pick-pipeline/"
         "target-selector-offline/tools/target_selector_eval")
sys.path.insert(0, STUDY)

from autolabel import AutoLabeler, label_event  # noqa: E402
from episode_io import GraspEvent, frame_at, load_episode  # noqa: E402

RECORDINGS = "/home/tommaso/Desktop/kitting-v2/rl-teleop/recordings"
DAY = {"20260812": "20260811", "20260814": "20260814"}  # dataset -> recording day
WORK = os.path.join(HERE, "work")
# A grasp whose t_close is within this of a study event's close time is
# COVERED by that event's masks (same approach, same packet).
MATCH_TOL_S = 1.0
APPROACH_S = 6.0
FPS = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DAY), default="20260812")
    a = ap.parse_args()
    mapping = json.load(open(os.path.join(HERE, f"mapping-{a.dataset}.json")))
    rec_root = os.path.join(RECORDINGS, DAY[a.dataset])

    al = None
    n_done = 0
    for ep_name in sorted({m["recording"] for m in mapping}):
        ep_path = os.path.join(rec_root, ep_name)
        ts = np.load(os.path.join(ep_path, "camera_right-rgb-timestamp.npy")
                     ).astype(np.float64)
        have = []
        for src in (os.path.join(STUDY, "work", ep_name),
                    os.path.join(WORK, ep_name)):
            for p in glob.glob(os.path.join(src, "event_*.npz")):
                have.append(float(ts[int(os.path.basename(p)[6:-4])]))
        for m in [m for m in mapping if m["recording"] == ep_name]:
            if have and min(abs(m["t_close"] - t) for t in have) <= MATCH_TOL_S:
                continue
            cf = frame_at(ts, int(m["t_close"] * 1e9))
            ev = GraspEvent(t_close_ns=int(m["t_close"] * 1e9),
                            t_open_ns=int(m["t_close"] * 1e9),
                            close_frame=cf,
                            approach_start_frame=max(0, cf - int(APPROACH_S * FPS)))
            if al is None:
                al = AutoLabeler()
            ep = load_episode(ep_path, side="right")
            t0 = time.time()
            path = label_event(al, ep, ev, os.path.join(WORK, ep_name))
            if path is None:
                print(f"[{ep_name}] ep {m['episode_index']} close@{cf}: SEED FAILED",
                      flush=True)
                continue
            have.append(float(ts[cf]))
            n_done += 1
            print(f"[{ep_name}] ep {m['episode_index']} close@{cf}: ok "
                  f"({time.time() - t0:.1f}s) -> {path}", flush=True)
    print(f"labeled {n_done} missing events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
