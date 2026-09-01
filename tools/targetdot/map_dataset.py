#!/usr/bin/env python3
"""Reconstruct the recordings -> LeRobot frame mapping for a grasp-window export.

Presets (--dataset):

  20260812  ETHRC/yam_grasp_right_20260812 — exported 2026-08-12 with
            --arms right --cameras wrist_right --keep /tmp/keeplist/keep_20260812.json
            (keep-list recovered from the export session's transcript, checked in
            here as keep_20260812.json; the /tmp original is gone).
  20260814  ETHRC/yam_grasp_right_20260814 — exported by run_night_20260814.sh:
            --arms right --cameras wrist_right_top --window-mode grasp
            --gripper-open-ref 1.0 --gripper-closed-ref 0.0, no keep-list.
            Pixel check runs against the *_wristnative predecoded cache (wrist
            frames stored at native 640x480 there).

This replays the exporter's PLANNING and WRITE loop (same code, imported from
tools/export_lerobot.py, no forked logic) without writing a dataset, and emits
mapping-<name>.json:

    [ { "episode_index": 0, "recording": "episode_...",
        "t_close": ...,                      # the grasp this window is cut around
        "lo": ..., "hi": ...,                # window bounds (epoch s)
        "cam_idx": [...],                    # per dataset frame_index: wrist-cam
                                             #   frame index in the recording mp4
        "cam_t": [...],                      # per dataset frame_index: camera ts
        "dataset_from_index": ... },         # global frame offset (from meta)
      ... ]

Then VERIFIES the reconstruction against the actual dataset:
  1. episode count and per-episode frame counts must equal meta/episodes exactly;
  2. sampled frames are pixel-compared: recording mp4 frame at cam_idx vs the
     predecoded JPEG for (episode, frame_index). Must match better than either
     matches its temporal neighbours.

Frames the exporter SKIPPED (any camera >2/30 s stale at the grid point) are
skipped here identically — that is why frame_index does not advance uniformly
with grid time and why the mapping must be replayed, not computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # worktree root -> tools.*

import tools.export_lerobot as ex  # noqa: E402

RECORDINGS = Path("/home/tommaso/Desktop/kitting-v2/rl-teleop/recordings")
HF = Path.home() / ".cache/huggingface/lerobot/ETHRC"
PD = Path.home() / ".cache/lerobot-predecoded"

PRESETS = {
    "20260812": dict(
        recordings=RECORDINGS / "20260811",
        dataset=HF / "yam_grasp_right_20260812",
        cameras="wrist_right",
        keep=HERE / "keep_20260812.json",
        grip_refs=(None, None),
        # pixel check: the plain predecoded cache, wrist at native 640x480
        check_cache=PD / "yam_grasp_right_20260812",
    ),
    "20260814": dict(
        recordings=RECORDINGS / "20260814",
        dataset=HF / "yam_grasp_right_20260814",
        cameras="wrist_right_top",
        keep=None,
        grip_refs=(1.0, 0.0),
        # pixel check: the WRISTNATIVE cache (wrist stored at native 640x480,
        # which is also what the dot-burning step consumes)
        check_cache=PD / "yam_grasp_right_20260814_wristnative",
    ),
}

WRIST_KEY = "observation.images.wrist"


def build_mapping(cfg: dict) -> list[dict]:
    ex.KEEP = ex.load_keep_list(cfg["keep"]) if cfg["keep"] else None
    keep_entries = (json.loads(Path(cfg["keep"]).read_text())["keep"]
                    if cfg["keep"] else None)
    cameras = ex.CAMERA_SETS[cfg["cameras"]]
    open_ref, closed_ref = cfg["grip_refs"]
    report = ex.Report()

    episodes = []
    for ep in ex.episode_dirs(cfg["recordings"], ("right",)):
        plan = ex.plan_episode(ep, ex.DEFAULT_PRE_S, ex.DEFAULT_POST_S,
                               ex.DEFAULT_FPS, report, cameras=cameras,
                               arms=("right",), open_ref=open_ref,
                               closed_ref=closed_ref)
        if plan is None:
            continue
        cam_t = {c: np.load(plan["cams"][c][1]).astype(float) for c in cameras}
        wrist_t = cam_t["camera_right"]

        if keep_entries is not None:
            closes = sorted(k["t_close"] for k in keep_entries
                            if k["episode"] == ep.name)
        else:
            grasps, why = ex.usable_grasps(ep, arm="right")
            assert not why, (ep.name, why)
            closes = sorted(float(g["t"]) for g in grasps
                            if g.get("t") is not None)
        assert len(closes) == len(plan["windows"]), \
            f"{ep.name}: {len(closes)} grasps but {len(plan['windows'])} windows"

        for (lo, hi), t_close in zip(plan["windows"], closes):
            n = int((hi - lo) * ex.DEFAULT_FPS)
            if n < ex.MIN_WINDOW_FRAMES:
                continue                              # exporter: "window only N frames"
            grid = lo + np.arange(n) / ex.DEFAULT_FPS
            stale = np.zeros(n, dtype=bool)
            for c in cameras:                         # staleness across ALL cameras
                ci_c = ex.nearest_index(cam_t[c], grid)
                stale |= np.abs(cam_t[c][ci_c] - grid) > ex.MAX_CAM_STALENESS_S
            ci = ex.nearest_index(wrist_t, grid)
            kept = [int(ci[k]) for k in range(n) if not stale[k]]
            if len(kept) < ex.MIN_WINDOW_FRAMES:
                continue                              # exporter: "only N usable frames"
            episodes.append({
                "recording": ep.name,
                "t_close": t_close,
                "lo": lo, "hi": hi,
                "cam_idx": kept,
                "cam_t": [float(wrist_t[i]) for i in kept],
            })
    for i, e in enumerate(episodes):
        e["episode_index"] = i
    for r in report.rejected:
        print(f"  planner rejected {r.episode}: {r.reason}")
    return episodes


def verify_counts(cfg: dict, episodes: list[dict]) -> None:
    import pandas as pd
    meta = sorted((cfg["dataset"] / "meta/episodes").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in meta]).sort_values("episode_index")
    assert len(df) == len(episodes), f"episode count {len(episodes)} != {len(df)}"
    fcol = f"videos/{WRIST_KEY}/file_index"
    tcol = f"videos/{WRIST_KEY}/from_timestamp"
    for e, (_, row) in zip(episodes, df.iterrows()):
        assert len(e["cam_idx"]) == int(row["length"]), (
            f"episode {e['episode_index']} ({e['recording']} t_close={e['t_close']:.2f}): "
            f"reconstructed {len(e['cam_idx'])} frames, dataset has {int(row['length'])}")
        e["dataset_from_index"] = int(row["dataset_from_index"])
        e["wrist_file_index"] = int(row[fcol]) if fcol in row else 0
        # frame index within the wrist video FILE (files restart at t=0)
        e["wrist_file_frame0"] = int(round(float(row[tcol]) * ex.DEFAULT_FPS)) \
            if tcol in row else e["dataset_from_index"]
    print(f"  counts OK: {len(episodes)} episodes, "
          f"{sum(len(e['cam_idx']) for e in episodes)} frames match meta/episodes")


def jpg_path(cfg: dict, ep: dict, k: int) -> Path:
    return (cfg["check_cache"] / "videos" / WRIST_KEY /
            f"chunk-000/file-{ep['wrist_file_index']:03d}" /
            f"f{ep['wrist_file_frame0'] + k:06d}.jpg")


def verify_pixels(cfg: dict, episodes: list[dict], n_checks: int = 12) -> None:
    """Compare recording wrist frames against predecoded JPEGs at mapped indices."""
    import cv2
    rng = np.random.default_rng(0)
    picks = []
    for e in rng.choice(len(episodes), size=min(n_checks, len(episodes)),
                        replace=False):
        ep = episodes[int(e)]
        picks.append((ep, int(rng.integers(0, len(ep["cam_idx"])))))

    by_rec: dict[str, list] = {}
    for ep, k in picks:
        by_rec.setdefault(ep["recording"], []).append((ep, k))
    for rec, items in by_rec.items():
        mp4 = cfg["recordings"] / rec / "camera_right-images-rgb.mp4"
        items.sort(key=lambda p: p[0]["cam_idx"][p[1]])
        cap = cv2.VideoCapture(str(mp4))
        pos = -1
        prev = frame = nxt = None
        for ep, k in items:
            want = ep["cam_idx"][k]
            while pos < want + 1:
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
            jpg = cv2.imread(str(jpg_path(cfg, ep, k)))
            assert jpg is not None, jpg_path(cfg, ep, k)
            if jpg.shape != frame.shape:              # cache stores a resize
                jpg = cv2.resize(jpg, (frame.shape[1], frame.shape[0]))
            d_here = float(np.mean(np.abs(jpg.astype(np.int16) - frame.astype(np.int16))))
            d_prev = float(np.mean(np.abs(jpg.astype(np.int16) - prev.astype(np.int16))))
            d_next = float(np.mean(np.abs(jpg.astype(np.int16) - nxt.astype(np.int16))))
            tag = "OK " if d_here <= min(d_prev, d_next) else "BAD"
            print(f"  [{tag}] ep{ep['episode_index']:>3} frame {k:>3} -> {rec} "
                  f"cam {want:>5}: |diff|={d_here:5.2f} (prev {d_prev:5.2f}, next {d_next:5.2f})")
            assert tag == "OK ", "mapped frame matches a NEIGHBOUR better than itself"
        cap.release()
    print(f"  pixel check OK on {len(picks)} frames "
          f"(mapped frame always the best match)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="20260812")
    a = ap.parse_args()
    cfg = PRESETS[a.dataset]
    episodes = build_mapping(cfg)
    print(f"reconstructed {len(episodes)} episodes / "
          f"{sum(len(e['cam_idx']) for e in episodes)} frames")
    verify_counts(cfg, episodes)
    verify_pixels(cfg, episodes)
    out = HERE / f"mapping-{a.dataset}.json"
    out.write_text(json.dumps(episodes))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
