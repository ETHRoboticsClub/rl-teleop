#!/usr/bin/env python3
"""Export recorded teleop episodes to a LeRobot dataset for ACT training.

Scope: GRASPING ONLY. Each recorded episode contains a whole kitting run (reach,
grasp, carry, place, repeat). A grasp policy should not be trained on the carry
and place phases, so this cuts one short LeRobot episode per successful grasp.

    recorded episode (114s, 4 grasps)
    |---------------------------------------------------------------|
        [==]        [==]              [==]        [==]
         ^grasp      ^grasp            ^grasp      ^grasp
        4 LeRobot episodes, ~5s each

WHY A WINDOW AND NOT A SEGMENT: annotations.json does carry `segments` with
phase="grasp", but those are INSTANTS — t_start == t_end, the moment the gripper
closed. Only "transport" has real duration. So the trainable window has to be
constructed: PRE_S before the close (the descent/approach) through POST_S after
it (the lift). Defaults are deliberately conservative; tune with --pre-s/--post-s.

CLOCKS. Four streams, four rates, and the cameras do not even agree with each
other on frame count within one episode (measured: 3372 / 3426 / 3422 over the
same 114.3s). Nothing can be zipped by index. Everything is resampled onto a
uniform grid by TIMESTAMP:

    grid (30Hz) .....|.........|.........|.........|.....
    yam    200Hz ||||||||||||||||||||||||||||||||||||||||   nearest
    gello 62.5Hz  |   |   |   |   |   |   |   |   |   |     nearest
    cam_top  30Hz .   |    .   |    .   |    .   |    .     nearest, staleness-capped
    cam_left 31Hz .  |   .   |   .    |   .   |   .   .     nearest, staleness-capped

ACTION vs STATE. action = gello (leader, what the operator commanded),
observation.state = yam (follower, where the arm actually was). Swapping these
trains a policy to predict where the arm already is: the loss curve looks fine
and the policy is useless on hardware. Pinned by test.

GRIPPER. Raw motor position, and the limits are auto-calibrated on every boot
(observed 5.2218 / -0.0235 on one boot). Raw values are therefore NOT comparable
across sessions. Normalised per-episode to [0,1] (0=closed, 1=open) — the same
convention labeling/constants.py already uses, for the same reason. The scale is
taken over the WHOLE episode, not the window, so every window of one episode
shares one scale.

Usage:
    uv run python tools/export_lerobot.py --repo-id ETHRC/yam_grasp_v1
    uv run python tools/export_lerobot.py --root recordings/20260728 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.mcap_io import read_positions

# ── tunables ────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_PRE_S = 3.0      # descent/approach before the gripper closes
DEFAULT_POST_S = 2.0     # lift after it closes
# A grid point whose nearest camera frame is further away than this in time has
# no honest image. Two frames at 30Hz — beyond that we are inventing data.
MAX_CAM_STALENESS_S = 2.0 / DEFAULT_FPS
# A window needs at least this many frames to be worth training on.
MIN_WINDOW_FRAMES = 10

# recorded camera name -> LeRobot feature suffix. The scan camera is deliberately
# absent: it looks at the packet mat, not the workspace, contributes nothing to a
# grasp policy, and is ~49% of every episode's bytes.
CAMERAS = {"camera_top": "top", "camera_left": "wrist"}

JOINT_NAMES = [f"joint_{i + 1}" for i in range(C.N_ARM_JOINTS)] + ["gripper"]
N_DOF = C.N_ARM_JOINTS + 1

DEFAULT_TASK = "grasp the bag and lift it"


# ── episode selection ───────────────────────────────────────────────────────
@dataclass
class Rejection:
    episode: str
    reason: str


@dataclass
class Report:
    kept: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    windows: int = 0
    frames: int = 0

    def reject(self, ep: str, reason: str) -> None:
        self.rejected.append(Rejection(ep, reason))


def episode_dirs(root: Path) -> list[Path]:
    if (root / "yam_left.mcap").exists():
        return [root]
    # .trash holds episodes the operator threw away in the cockpit. Delete is a
    # move so it stays undoable — but it must never be training data.
    return sorted(p for p in root.rglob("episode_*")
                  if p.is_dir() and ".trash" not in p.parts)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def in_workspace(g: dict) -> bool:
    """Is this grasp inside the packet mat? See constants.GRASP_WORKSPACE_X_MIN.

    Grasp-level, NOT episode-level: both episodes holding out-of-workspace
    grasps also hold good ones, so rejecting the whole episode would throw away
    23 usable windows to remove 4 bad ones.

    Fails OPEN: a grasp with no ee_pose is unknown, not out-of-bounds, and this
    gate only removes grasps it can positively measure as outside. Silently
    dropping unlabelled data here would hide a labeller bug as a workspace
    result. (All 81 grasps in the corpus carry a pose; this is the guard, not
    the common path.)
    """
    pose = g.get("ee_pose")
    if not pose:
        return True
    return float(pose[0]) >= C.GRASP_WORKSPACE_X_MIN


def usable_grasps(ep: Path, workspace_gate: bool = True) -> tuple[list[dict], str | None]:
    """Successful grasp attempts in this episode, or (,reason-it-was-rejected).

    Every filter here corresponds to a real failure seen in the recorded corpus,
    not a hypothetical one.

    ``workspace_gate=False`` returns the out-of-workspace grasps too, so a review
    tool can show what was dropped and why. Nothing that writes training data
    should pass False.
    """
    ann = load_json(ep / "annotations.json")
    if ann is None:
        return [], "no annotations.json"           # 2 of 30 episodes

    meta = ann.get("episode_meta") or {}
    attempts = ann.get("grasp_attempts") or []

    # `outcome` DEFAULTS to "success" and is not derived from whether anything
    # was actually grasped: 3 of 18 "success" episodes hold zero grasp_attempts.
    # So the attempt list is the authority and outcome is only a veto.
    if meta.get("outcome") == "aborted":
        return [], "episode outcome=aborted"       # 10 of 28
    if not attempts:
        return [], "zero grasp_attempts (label says success, nothing was grasped)"

    # Operator flag 'x' = "that one was bad", pressed live during the take.
    flags = load_json(ep / "operator_flags.json") or {}
    tags = {f.get("tag") for f in flags.get("flags", [])}
    if "x" in tags:
        return [], "operator flagged 'x' (bad)"

    good = [a for a in attempts if a.get("outcome") == "success"]
    if not good:
        return [], f"{len(attempts)} grasp attempts, none with outcome=success"

    if workspace_gate:
        n_before = len(good)
        good = [a for a in good if in_workspace(a)]
        if not good:
            return [], f"all {n_before} grasps outside the workspace (x < {C.GRASP_WORKSPACE_X_MIN})"
    return good, None


def grasp_windows(grasps: list[dict], t0: float, t1: float,
                  pre_s: float, post_s: float) -> list[tuple[float, float]]:
    """[close - pre, close + post] per grasp, clipped so windows never overlap.

    Overlap matters: two grasps 2s apart with pre=3/post=2 would otherwise share
    frames, and the same frames appearing in two training episodes silently
    inflates the dataset while teaching contradictory actions for one image.
    """
    ts = sorted(float(g["t"]) for g in grasps if g.get("t") is not None)
    out = []
    for i, t in enumerate(ts):
        lo = max(t0, t - pre_s)
        hi = min(t1, t + post_s)
        if i > 0:                       # never reach back past the previous grasp
            lo = max(lo, (ts[i - 1] + t) / 2.0)
        if i + 1 < len(ts):             # never reach forward past the next one
            hi = min(hi, (t + ts[i + 1]) / 2.0)
        if hi > lo:
            out.append((lo, hi))
    return out


# ── signals ─────────────────────────────────────────────────────────────────
def normalize_gripper(col: np.ndarray) -> np.ndarray:
    """Raw gripper motor position -> [0,1], 0=closed 1=open, per episode.

    Same rationale as labeling/constants.py: the raw units differ per rig AND per
    boot (limits are auto-detected at startup), so an absolute value means
    nothing across sessions. A degenerate range (arm never moved the gripper)
    collapses to all-open rather than dividing by ~0.
    """
    lo, hi = float(np.min(col)), float(np.max(col))
    if hi - lo < 1e-6:
        return np.ones_like(col, dtype=np.float32)
    return ((col - lo) / (hi - lo)).astype(np.float32)


def nearest_index(sorted_t: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the nearest sample in sorted_t for each target time."""
    idx = np.searchsorted(sorted_t, targets)
    idx = np.clip(idx, 1, len(sorted_t) - 1)
    left, right = sorted_t[idx - 1], sorted_t[idx]
    return np.where(targets - left <= right - targets, idx - 1, idx)


class CameraStream:
    """One recorded mp4 + its per-frame timestamps, read strictly forward.

    Sequential decode, never seek. cv2 seeking on h264 lands on the nearest
    keyframe and silently returns the wrong frame; sequential reads are also an
    order of magnitude faster across a whole episode.
    """

    def __init__(self, mp4: Path, stamps: Path):
        import cv2
        self.t = np.load(stamps).astype(float)
        self.cap = cv2.VideoCapture(str(mp4))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {mp4}")
        self._pos = -1
        self._frame = None
        self._cv2 = cv2

    def frame_at(self, index: int):
        """RGB frame at the given index, decoding forward as needed."""
        if index < self._pos:
            raise RuntimeError("backward seek requested; windows must be sorted")
        while self._pos < index:
            ok, bgr = self.cap.read()
            if not ok:
                return None
            self._pos += 1
            self._frame = bgr
        if self._frame is None:
            return None
        # cv2 decodes BGR; LeRobot stores RGB. Getting this wrong costs a full
        # training run to notice, and only if someone looks at the images.
        return self._cv2.cvtColor(self._frame, self._cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


# ── export ──────────────────────────────────────────────────────────────────
def build_features(shapes: dict[str, tuple[int, int]]) -> dict:
    """shapes: recorded camera name -> (height, width).

    Per-camera, NOT one shared resolution: the rig records camera_top at
    1280x720 and camera_left at 640x480. Assuming they match makes
    LeRobotDataset reject every wrist frame at add_frame() time.
    """
    feats = {
        "observation.state": {"dtype": "float32", "shape": (N_DOF,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (N_DOF,), "names": JOINT_NAMES},
    }
    for cam, suffix in CAMERAS.items():
        h, w = shapes[cam]
        feats[f"observation.images.{suffix}"] = {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def probe_shapes(plan: dict) -> dict[str, tuple[int, int]] | None:
    """Decode one frame per camera to learn each one's real resolution."""
    shapes = {}
    for cam in CAMERAS:
        s = CameraStream(*plan["cams"][cam])
        try:
            img = s.frame_at(0)
            if img is None:
                return None
            shapes[cam] = img.shape[:2]
        finally:
            s.close()
    return shapes


def plan_episode(ep: Path, pre_s: float, post_s: float, fps: int, report: Report):
    """Everything needed to write this episode's windows, or None if unusable."""
    grasps, why = usable_grasps(ep)
    if why:
        report.reject(ep.name, why)
        return None

    try:
        t_yam, p_yam = read_positions(ep / "yam_left.mcap", "yam_left")
        t_gel, p_gel = read_positions(ep / "gello_left.mcap", "gello_left")
    except Exception as e:
        report.reject(ep.name, f"mcap read failed: {e}")
        return None
    if t_yam.size == 0 or t_gel.size == 0:
        report.reject(ep.name, "empty joint stream")
        return None

    cams = {}
    for cam in CAMERAS:
        mp4 = ep / f"{cam}-images-rgb.mp4"
        stamps = ep / f"{cam}-rgb-timestamp.npy"
        if not mp4.exists() or not stamps.exists():
            report.reject(ep.name, f"missing {cam}")
            return None
        cams[cam] = (mp4, stamps)

    # Gripper scale over the WHOLE episode so all its windows share one scale.
    state = p_yam.astype(np.float32).copy()
    action = p_gel.astype(np.float32).copy()
    state[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(state[:, C.GRIPPER_JOINT_INDEX])
    action[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(action[:, C.GRIPPER_JOINT_INDEX])

    t0 = max(float(t_yam[0]), float(t_gel[0]))
    t1 = min(float(t_yam[-1]), float(t_gel[-1]))
    windows = grasp_windows(grasps, t0, t1, pre_s, post_s)
    if not windows:
        report.reject(ep.name, "no grasp window inside the recorded span")
        return None

    ann = load_json(ep / "annotations.json") or {}
    task = (ann.get("episode_meta") or {}).get("instruction") or DEFAULT_TASK

    return {"ep": ep, "windows": windows, "task": task, "cams": cams,
            "t_yam": t_yam, "state": state, "t_gel": t_gel, "action": action}


def export(root: Path, repo_id: str, out: Path | None, fps: int,
           pre_s: float, post_s: float, dry_run: bool) -> Report:
    report = Report()
    eps = episode_dirs(root)
    if not eps:
        print(f"no episodes under {root}", file=sys.stderr)
        return report

    plans = []
    for ep in eps:
        plan = plan_episode(ep, pre_s, post_s, fps, report)
        if plan:
            plans.append(plan)
            report.kept.append(ep.name)
            report.windows += len(plan["windows"])

    if dry_run or not plans:
        return report

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shapes = probe_shapes(plans[0])
    if shapes is None:
        print("could not decode a probe frame", file=sys.stderr)
        return report

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps,
                               features=build_features(shapes),
                               root=str(out) if out else None,
                               robot_type="yam_left", use_videos=True)

    for plan in plans:
        # A camera re-plugged at a different resolution mid-corpus would other-
        # wise blow up add_frame() partway through a long export. Check once per
        # episode and drop that episode with a reason instead.
        got = probe_shapes(plan)
        if got != shapes:
            report.reject(plan["ep"].name,
                          f"camera resolution {got} != dataset schema {shapes}")
            continue

        streams = {c: CameraStream(*plan["cams"][c]) for c in CAMERAS}
        try:
            for (lo, hi) in plan["windows"]:
                n = int((hi - lo) * fps)
                if n < MIN_WINDOW_FRAMES:
                    report.reject(plan["ep"].name, f"window only {n} frames")
                    continue
                grid = lo + np.arange(n) / fps

                si = nearest_index(plan["t_yam"], grid)
                ai = nearest_index(plan["t_gel"], grid)
                ci = {c: nearest_index(streams[c].t, grid) for c in CAMERAS}
                stale = {c: np.abs(streams[c].t[ci[c]] - grid) > MAX_CAM_STALENESS_S
                         for c in CAMERAS}

                wrote = 0
                for k in range(n):
                    if any(stale[c][k] for c in CAMERAS):
                        continue                      # no honest image for this instant
                    frame = {
                        "observation.state": plan["state"][si[k]],
                        "action": plan["action"][ai[k]],
                        "task": plan["task"],
                    }
                    bad = False
                    for cam, suffix in CAMERAS.items():
                        img = streams[cam].frame_at(int(ci[cam][k]))
                        if img is None:
                            bad = True
                            break
                        frame[f"observation.images.{suffix}"] = img
                    if bad:
                        break
                    ds.add_frame(frame)
                    wrote += 1

                if wrote >= MIN_WINDOW_FRAMES:
                    ds.save_episode()
                    report.frames += wrote
                else:
                    # Drop the partial buffer rather than saving a stub episode.
                    ds.episode_buffer = None
                    report.reject(plan["ep"].name, f"only {wrote} usable frames")
        finally:
            for s in streams.values():
                s.close()

    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="recordings")
    ap.add_argument("--repo-id", default="ETHRC/yam_grasp_v1")
    ap.add_argument("--out", default=None, help="dataset root (default: HF cache)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--pre-s", type=float, default=DEFAULT_PRE_S)
    ap.add_argument("--post-s", type=float, default=DEFAULT_POST_S)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be exported, write nothing")
    a = ap.parse_args(argv)

    rep = export(Path(a.root), a.repo_id, Path(a.out) if a.out else None,
                 a.fps, a.pre_s, a.post_s, a.dry_run)

    print(f"\nepisodes kept   : {len(rep.kept)}")
    print(f"grasp windows   : {rep.windows}")
    if not a.dry_run:
        print(f"frames written  : {rep.frames}")
    print(f"episodes dropped: {len(rep.rejected)}")
    # Never silently drop data: every rejection is printed with its reason, so a
    # corpus that exports 3 of 100 episodes is obvious instead of looking fine.
    for r in rep.rejected:
        print(f"   {r.episode:<34} {r.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
