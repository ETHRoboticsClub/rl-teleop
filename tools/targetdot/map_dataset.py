#!/usr/bin/env python3
"""Reconstruct the recordings->LeRobot frame mapping for ETHRC/yam_grasp_right_20260812.

The dataset was exported (2026-08-12) with:

    tools/export_lerobot.py --root recordings/20260811 \
        --keep /tmp/keeplist/keep_20260812.json \
        --arms right --cameras wrist_right \
        --repo-id ETHRC/yam_grasp_right_20260812

(The keep-list was recovered from the session transcript that ran the export and
is checked in next to this file as keep_20260812.json — the /tmp original is gone.)

This script replays the exporter's PLANNING and WRITE loop (same code, imported
from tools/export_lerobot.py, no forked logic) without writing a dataset, and
emits mapping.json:

    [ { "episode_index": 0,                  # LeRobot episode
        "recording": "episode_233520_91895ddc",
        "t_close": 1786484127.4839892,       # keep-list grasp this window is cut around
        "lo": ..., "hi": ...,                # window bounds (epoch s)
        "cam_idx": [...],                    # per dataset frame_index: camera_right
                                             #   frame index in the recording mp4
        "cam_t":  [...] },                   # per dataset frame_index: camera timestamp
      ... ]

Then VERIFIES the reconstruction against the actual dataset:
  1. episode count and per-episode frame counts must equal meta/episodes exactly;
  2. a sample of frames is pixel-compared: recording mp4 frame at cam_idx vs the
     predecoded JPEG for (episode, frame_index). Must match better than either
     matches its temporal neighbours.

Frames the exporter SKIPPED (no camera frame within 2/30 s of the grid point)
are skipped here identically — that is why frame_index does not advance
uniformly with grid time and why the mapping has to be replayed, not computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # worktree root -> tools.*

import tools.export_lerobot as ex  # noqa: E402

RECORDINGS = Path("/home/tommaso/Desktop/kitting-v2/rl-teleop/recordings/20260811")
DATASET = Path.home() / ".cache/huggingface/lerobot/ETHRC/yam_grasp_right_20260812"
PREDECODED = Path.home() / ".cache/lerobot-predecoded/yam_grasp_right_20260812"
KEEP_JSON = HERE / "keep_20260812.json"
OUT = HERE / "mapping.json"

ARMS = ("right",)
CAMERAS = ex.CAMERA_SETS["wrist_right"]              # {"camera_right": "wrist"}
FPS = ex.DEFAULT_FPS
PRE_S, POST_S = ex.DEFAULT_PRE_S, ex.DEFAULT_POST_S


def build_mapping() -> list[dict]:
    ex.KEEP = ex.load_keep_list(KEEP_JSON)
    report = ex.Report()
    keep_entries = json.loads(KEEP_JSON.read_text())["keep"]

    episodes = []
    for ep in ex.episode_dirs(RECORDINGS, ARMS):
        plan = ex.plan_episode(ep, PRE_S, POST_S, FPS, report,
                               cameras=CAMERAS, arms=ARMS)
        if plan is None:
            continue
        cam_t = np.load(plan["cams"]["camera_right"][1]).astype(float)
        # keep-list t_close values for this recording, sorted: windows come out
        # of grasp_windows() in ascending time, one per kept grasp (unless the
        # window clips to nothing, which does not happen in this corpus).
        closes = sorted(k["t_close"] for k in keep_entries
                        if k["episode"] == ep.name)
        assert len(closes) == len(plan["windows"]), \
            f"{ep.name}: {len(closes)} keep grasps but {len(plan['windows'])} windows"

        for (lo, hi), t_close in zip(plan["windows"], closes):
            n = int((hi - lo) * FPS)
            if n < ex.MIN_WINDOW_FRAMES:
                continue                              # exporter: "window only N frames"
            grid = lo + np.arange(n) / FPS
            ci = ex.nearest_index(cam_t, grid)
            stale = np.abs(cam_t[ci] - grid) > ex.MAX_CAM_STALENESS_S
            kept = [int(ci[k]) for k in range(n) if not stale[k]]
            if len(kept) < ex.MIN_WINDOW_FRAMES:
                continue                              # exporter: "only N usable frames"
            episodes.append({
                "recording": ep.name,
                "t_close": t_close,
                "lo": lo, "hi": hi,
                "cam_idx": kept,
                "cam_t": [float(cam_t[i]) for i in kept],
            })
    for i, e in enumerate(episodes):
        e["episode_index"] = i
    for r in report.rejected:
        print(f"  planner rejected {r.episode}: {r.reason}")
    return episodes


def verify_counts(episodes: list[dict]) -> None:
    import pandas as pd
    meta = sorted((DATASET / "meta/episodes").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in meta]).sort_values("episode_index")
    assert len(df) == len(episodes), f"episode count {len(episodes)} != {len(df)}"
    for e, (_, row) in zip(episodes, df.iterrows()):
        assert len(e["cam_idx"]) == int(row["length"]), (
            f"episode {e['episode_index']} ({e['recording']} t_close={e['t_close']:.2f}): "
            f"reconstructed {len(e['cam_idx'])} frames, dataset has {int(row['length'])}")
        e["dataset_from_index"] = int(row["dataset_from_index"])
    print(f"  counts OK: {len(episodes)} episodes, "
          f"{sum(len(e['cam_idx']) for e in episodes)} frames match meta/episodes")


def verify_pixels(episodes: list[dict], n_checks: int = 12) -> None:
    """Compare recording mp4 frames against predecoded JPEGs at mapped indices."""
    import cv2
    rng = np.random.default_rng(0)
    picks = []
    for e in rng.choice(len(episodes), size=min(n_checks, len(episodes)),
                        replace=False):
        ep = episodes[int(e)]
        k = int(rng.integers(0, len(ep["cam_idx"])))
        picks.append((ep, k))
    picks.sort(key=lambda p: (p[0]["recording"], p[0]["cam_idx"][p[1]]))

    jpg_dir = PREDECODED / "videos/observation.images.wrist/chunk-000/file-000"
    worst = 0.0
    by_rec: dict[str, list] = {}
    for ep, k in picks:
        by_rec.setdefault(ep["recording"], []).append((ep, k))
    for rec, items in by_rec.items():
        mp4 = RECORDINGS / rec / "camera_right-images-rgb.mp4"
        items.sort(key=lambda p: p[0]["cam_idx"][p[1]])
        cap = cv2.VideoCapture(str(mp4))
        pos, frame = -1, None
        for ep, k in items:
            want = ep["cam_idx"][k]
            while pos < want + 1:                    # decode want-1, want, want+1
                ok, f = cap.read()
                if not ok:
                    break
                pos += 1
                if pos == want - 1:
                    prev = f.copy()
                if pos == want:
                    frame = f.copy()
                if pos == want + 1:
                    nxt = f.copy()
            g = int(ep["dataset_from_index"]) + k
            jpg = cv2.imread(str(jpg_dir / f"f{g:06d}.jpg"))
            d_here = float(np.mean(np.abs(jpg.astype(np.int16) - frame.astype(np.int16))))
            d_prev = float(np.mean(np.abs(jpg.astype(np.int16) - prev.astype(np.int16))))
            d_next = float(np.mean(np.abs(jpg.astype(np.int16) - nxt.astype(np.int16))))
            tag = "OK " if d_here <= min(d_prev, d_next) else "BAD"
            worst = max(worst, d_here - min(d_prev, d_next))
            print(f"  [{tag}] ep{ep['episode_index']:>3} frame {k:>3} -> {rec} "
                  f"cam {want:>5}: |diff|={d_here:5.2f} (prev {d_prev:5.2f}, next {d_next:5.2f})")
            assert tag == "OK ", "mapped frame matches a NEIGHBOUR better than itself"
        cap.release()
    print(f"  pixel check OK on {len(picks)} frames "
          f"(mapped frame always the best match)")


def main() -> int:
    episodes = build_mapping()
    print(f"reconstructed {len(episodes)} episodes / "
          f"{sum(len(e['cam_idx']) for e in episodes)} frames")
    verify_counts(episodes)
    verify_pixels(episodes)
    OUT.write_text(json.dumps(episodes))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
