#!/usr/bin/env python3
"""Burn the target dot into a COPY of the predecoded frame cache.

Source:  ~/.cache/lerobot-predecoded/yam_grasp_right_20260812            (untouched)
Dest:    ~/.cache/lerobot-predecoded/yam_grasp_right_20260812_targetdot  (created here)

Every frame is first byte-copied; frames with a label from labels.json are then
re-encoded with tools/target_dot.py's canonical dot burned in at native
640x480, JPEG q=95 with 4:4:4 chroma (the source cache was ffmpeg -q:v 2
yuvj444p; full chroma keeps the dot's edge crisp). Unlabeled frames stay
byte-identical to the source cache.

Refuses to run if the destination already exists — the training-run contract
is a fresh, never-mutated cache.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tools.target_dot import draw_target_dot  # noqa: E402

SRC = Path.home() / ".cache/lerobot-predecoded/yam_grasp_right_20260812"
DST = Path.home() / ".cache/lerobot-predecoded/yam_grasp_right_20260812_targetdot"
REL = "videos/observation.images.wrist/chunk-000/file-000"

ENC = [cv2.IMWRITE_JPEG_QUALITY, 95]
if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
    ENC += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def main() -> int:
    if DST.exists():
        raise SystemExit(f"refusing: {DST} already exists")
    mapping = json.load(open(HERE / "mapping.json"))
    labels = json.load(open(HERE / "labels.json"))
    frm = {m["episode_index"]: m["dataset_from_index"] for m in mapping}

    print(f"copying {SRC} -> {DST} ...")
    shutil.copytree(SRC, DST)

    folder = DST / REL
    n_dot = n_skip = 0
    for ep in labels:
        base = frm[ep["episode_index"]]
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
    got = len(list(folder.glob("f*.jpg")))
    assert got == total == 5910, (got, total)
    print("frame count intact: 5910")
    return 0


if __name__ == "__main__":
    sys.exit(main())
