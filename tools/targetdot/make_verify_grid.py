#!/usr/bin/env python3
"""Visual check of the burned dots BEFORE training on them.

For a spread of dataset episodes, sample the dotted cache at five phases of the
grasp window (approach start / 25% / 50% / pre-close / lift) and tile them into
verify/grid_<dataset>_*.jpg (downscaled). The LIFT column doubles as ground
truth: the packet between the jaws there is by construction the packet that was
grasped, so the dot in the earlier columns must sit on that same packet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PD = Path.home() / ".cache/lerobot-predecoded"
PRESETS = {
    "20260812": dict(dst=PD / "yam_grasp_right_20260812_targetdot",
                     episodes=[0, 2, 6, 11, 15, 17, 21, 25, 30, 34, 38, 42]),
    "20260814": dict(dst=PD / "yam_grasp_right_20260814_wristnative_targetdot",
                     episodes=[0, 2, 8, 13, 21, 31, 37, 48, 54, 63, 71, 78]),
}
WRIST_KEY = "observation.images.wrist"
PHASES = [0.0, 0.25, 0.5, 0.62, 0.95]     # fraction of window; ~0.6 = pre-close, 0.95 = lift
CELL_W = 320                              # per-cell width in the grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="20260812")
    a = ap.parse_args()
    cfg = PRESETS[a.dataset]
    mapping = {m["episode_index"]: m
               for m in json.load(open(HERE / f"mapping-{a.dataset}.json"))}
    labels = {l["episode_index"]: l
              for l in json.load(open(HERE / f"labels-{a.dataset}.json"))}
    OUT = HERE / "verify"
    OUT.mkdir(exist_ok=True)

    eps = cfg["episodes"]
    for half, name in ((eps[:6], f"grid_{a.dataset}_a.jpg"),
                       (eps[6:], f"grid_{a.dataset}_b.jpg")):
        rows = []
        for e in half:
            m, lab = mapping[e], labels[e]
            folder = (cfg["dst"] / "videos" / WRIST_KEY /
                      f"chunk-000/file-{m.get('wrist_file_index', 0):03d}")
            base = m.get("wrist_file_frame0", m["dataset_from_index"])
            n = len(lab["points"])
            cells = []
            for ph in PHASES:
                k = min(n - 1, int(ph * n))
                img = cv2.imread(str(folder / f"f{base + k:06d}.jpg"))
                h, w = img.shape[:2]
                img = cv2.resize(img, (CELL_W, CELL_W * h // w))
                tag = f"ep{e} k={k}" + ("" if lab["points"][k] else " (no dot)")
                cv2.putText(img, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(img)
            rows.append(np.hstack(cells))
        grid = np.vstack(rows)
        cv2.imwrite(str(OUT / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"wrote {OUT / name}  {grid.shape[1]}x{grid.shape[0]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
