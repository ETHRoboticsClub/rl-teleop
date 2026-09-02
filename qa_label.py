#!/usr/bin/env python3
"""Label ONE episode with the correct physical gripper refs and emit qa.json.

Run in the project env (uv run). Fixes the live-labeler mislabel (it omits
gripper refs) and records recording-health so the review can tell a dead
gripper channel apart from a genuine operator abort.

  uv run python qa_label.py recordings/<date>/episode_xxx
  uv run python qa_label.py recordings/<date>/episode_xxx left --task grasp

--task grasp is for approach+grasp+lift demos (no transport, no placement).
In that mode the MIN_TRANSPORT_M gate is not applied, because on a grasp-only
demo it deletes every grasp rather than filtering any (measured 2026-09-02:
episode_143533_ee94747f, 4 real grasps → 0 grasp_attempts + 8 flags).
"""
import argparse
import json
import os

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.label_episode import label_episode_dir
from robots_realtime.labeling.mcap_io import read_positions

# The gripper's physical limits live in constants.py — ONE source of truth
# shared with live_server.py's --open-ref/--closed-ref defaults. They used to be
# hardcoded here, so a recalibration had to be found in two files.
OPEN_REF, CLOSED_REF = C.GRIPPER_OPEN_REF, C.GRIPPER_CLOSED_REF


def run(d, arm="left", task=C.TASK_KITTING, write=True):
    qa = {"episode": os.path.basename(d), "arm": arm, "task": task}
    mcap = os.path.join(d, f"yam_{arm}.mcap")
    try:
        t, pos = read_positions(mcap, f"yam_{arm}")
        g = pos[:, C.GRIPPER_JOINT_INDEX]
        norm = np.clip((g - CLOSED_REF) / (OPEN_REF - CLOSED_REF), 0.0, 1.0)
        qa["mcap_ok"] = True
        qa["n_samples"] = int(t.size)
        qa["gripper_min_norm"] = round(float(norm.min()), 4)
        # Did the gripper ever close enough to attempt a grasp? (< close-enter threshold)
        qa["gripper_actuated"] = bool(norm.min() < C.GRIPPER_CLOSE_ENTER)
    except Exception as e:
        qa["mcap_ok"] = False
        qa["error"] = str(e)[:200]

    if qa.get("mcap_ok"):
        try:
            ann = label_episode_dir(
                d, arm=arm,
                gripper_open_ref=OPEN_REF, gripper_closed_ref=CLOSED_REF,
                # The transport gate is a KITTING gate. In grasp mode it is not
                # passed at all rather than being passed and ignored downstream.
                min_transport_m=(C.MIN_TRANSPORT_M if task == C.TASK_KITTING else 0.0),
                geometric_targets=(task == C.TASK_KITTING),
                task=task, write=write)
            qa["outcome"] = ann.episode_meta.outcome
            qa["grasps"] = len(ann.grasp_attempts)
            qa["successes"] = sum(1 for a in ann.grasp_attempts if a.outcome == "success")
            qa["places"] = len(ann.place_events)
            holds = sorted(a.hold_s for a in ann.grasp_attempts
                           if a.outcome == "success" and a.hold_s is not None)
            if holds:
                qa["hold_s"] = {"n": len(holds),
                                "min": round(holds[0], 3),
                                "median": round(holds[len(holds) // 2], 3),
                                "max": round(holds[-1], 3)}
        except Exception as e:
            qa["label_error"] = str(e)[:200]

    if write:
        json.dump(qa, open(os.path.join(d, "qa.json"), "w"), indent=2)
    return qa


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episode_dir")
    ap.add_argument("arm", nargs="?", default="left")
    ap.add_argument("--task", choices=list(C.LABEL_TASKS), default=C.TASK_KITTING)
    ap.add_argument("--no-write", action="store_true",
                    help="report only — write neither annotations.json nor qa.json. "
                         "Use this while a recording session is live: the auto-labeller "
                         "in live_server writes the same files.")
    a = ap.parse_args(argv)
    q = run(a.episode_dir, a.arm, task=a.task, write=not a.no_write)
    health = ("CORRUPT" if not q.get("mcap_ok")
              else "DEAD-GRIPPER" if q.get("gripper_actuated") is False
              else "ok")
    print(f"{q['episode']}: health={health} task={q['task']} grasps={q.get('grasps')} "
          f"success={q.get('successes')} outcome={q.get('outcome')} hold={q.get('hold_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
