#!/usr/bin/env python3
"""Burn the target dot into a COPY of a predecoded frame cache.

Presets (--dataset):

  20260812  ~/.cache/lerobot-predecoded/yam_grasp_right_20260812
            -> ..._targetdot                                        (wrist only)
  20260814  ~/.cache/lerobot-predecoded/yam_grasp_right_20260814_wristnative
            -> ..._wristnative_targetdot          (wrist native 640x480 + top,
            top frames copied untouched — the dot is a wrist-image cue)

Every frame is first byte-copied; wrist frames with a label from
labels-<dataset>.json are then re-encoded with tools/target_dot.py's canonical
dot burned in at native 640x480, JPEG q=95 with 4:4:4 chroma (the source cache
was ffmpeg -q:v 2 yuvj444p; full chroma keeps the dot's edge crisp). Unlabeled
frames stay byte-identical to the source cache.

Refuses to run if the destination already exists — the training-run contract
is a fresh, never-mutated cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tools.target_dot import draw_target_dot  # noqa: E402

PD = Path.home() / ".cache/lerobot-predecoded"
PRESETS = {
    "20260812": dict(src=PD / "yam_grasp_right_20260812",
                     dst=PD / "yam_grasp_right_20260812_targetdot",
                     total=5910),
    "20260814": dict(src=PD / "yam_grasp_right_20260814_wristnative",
                     dst=PD / "yam_grasp_right_20260814_wristnative_targetdot",
                     total=11662),
}
WRIST_KEY = "observation.images.wrist"

ENC = [cv2.IMWRITE_JPEG_QUALITY, 95]
if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
    ENC += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="20260812")
    a = ap.parse_args()
    cfg = PRESETS[a.dataset]
    if cfg["dst"].exists():
        raise SystemExit(f"refusing: {cfg['dst']} already exists")
    mapping = json.load(open(HERE / f"mapping-{a.dataset}.json"))
    labels = json.load(open(HERE / f"labels-{a.dataset}.json"))
    meta = {m["episode_index"]: m for m in mapping}

    print(f"copying {cfg['src']} -> {cfg['dst']} ...")
    shutil.copytree(cfg["src"], cfg["dst"])

    n_dot = n_skip = 0
    for ep in labels:
        m = meta[ep["episode_index"]]
        folder = (cfg["dst"] / "videos" / WRIST_KEY /
                  f"chunk-000/file-{m.get('wrist_file_index', 0):03d}")
        base = m.get("wrist_file_frame0", m["dataset_from_index"])
        for k, pt in enumerate(ep["points"]):
            if pt is None:
                n_skip += 1
                continue
            p = folder / f"f{base + k:06d}.jpg"
            img = cv2.imread(str(p))            # BGR; dot color is RGB/BGR-symmetric
            assert img is not None, p
            draw_target_dot(img, pt[0], pt[1])
            ok = cv2.imwrite(str(p), img, ENC)
            assert ok, p
            n_dot += 1
    total = n_dot + n_skip
    print(f"burned {n_dot}/{total} frames ({n_skip} left undotted)")
    got = sum(1 for _ in (cfg["dst"] / "videos" / WRIST_KEY).rglob("f*.jpg"))
    assert got == total == cfg["total"], (got, total, cfg["total"])
    print(f"frame count intact: {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
