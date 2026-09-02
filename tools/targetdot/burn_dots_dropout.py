#!/usr/bin/env python3
"""Burn the target dot into a COPY of a predecoded cache, with dot-dropout.
Dot on every labeled wrist frame EXCEPT a random `--dropout-frac`, so the policy
learns 'no dot' is a valid state and the dot carries information (SutureBot: render
on every input; dropout makes presence informative). Generic (explicit src/dst/
labels/mapping), unlike the preset-locked burn_dots.py. Refuses if dst exists."""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tools.target_dot import draw_target_dot
WRIST_KEY = "observation.images.wrist"
ENC = [cv2.IMWRITE_JPEG_QUALITY, 95]
if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
    ENC += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--mapping", required=True)
    ap.add_argument("--dropout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists(): raise SystemExit(f"refusing: {dst} exists")
    rng = random.Random(a.seed)
    labels = json.load(open(a.labels)); mapping = json.load(open(a.mapping))
    meta = {m["episode_index"]: m for m in mapping}
    print(f"copytree {src} -> {dst}"); shutil.copytree(src, dst)
    n_dot = n_drop = n_none = bad = 0
    for ep in labels:
        m = meta[ep["episode_index"]]
        folder = dst / "videos" / WRIST_KEY / f"chunk-000/file-{m.get('wrist_file_index',0):03d}"
        base = m.get("wrist_file_frame0", m["dataset_from_index"])
        for k, pt in enumerate(ep["points"]):
            if pt is None: n_none += 1; continue
            if not (0 <= pt[0] < 640 and 0 <= pt[1] < 480): bad += 1; continue  # sanity: in-bounds
            if rng.random() < a.dropout_frac: n_drop += 1; continue            # dropout
            p = folder / f"f{base+k:06d}.jpg"; img = cv2.imread(str(p))
            if img is None: bad += 1; continue
            draw_target_dot(img, pt[0], pt[1]); cv2.imwrite(str(p), img, ENC); n_dot += 1
    print(f"dotted {n_dot}, dropped {n_drop}, no-target {n_none}, bad {bad}")
    if bad > n_dot * 0.05: raise SystemExit(f"too many bad labels ({bad}) -- refusing conditioned cache")
    if n_dot < 500: raise SystemExit(f"only {n_dot} dotted frames -- labels look wrong, refusing")
    return 0
if __name__ == "__main__": sys.exit(main())
