#!/usr/bin/env python3
"""Bake a geometry into a predecoded JPG cache, so training does not pay for it.

    python tools/predecode_resize.py --geometry bus \\
        --source ~/.cache/lerobot-predecoded/yam_pickplace_right_20260814 \\
        --dest   ~/.cache/lerobot-predecoded/yam_pickplace_right_20260814_bus

WHY. act_bus_geometry resizes frames as they leave the decoder, which is correct
and costs no disk -- but it costs CPU on every sample, forever. Measured on this
box (32 cores, 2 concurrent lanes, batch 8):

    workers=4    updt_s 0.036  data_s 0.093     -> 0.129 s/step
    workers=10   updt_s 0.101  data_s 0.039     -> 0.140 s/step, GPU util 24%

Raising workers just moved the cost between the two counters: the total is the
JPEG decode plus the PIL resize, and the box is CPU-bound on it either way. The
resize is the same for every epoch, so doing it 88 times per frame is waste.
Bake it once and the decode gets cheaper too -- a 240x320 JPG decodes far faster
than a 480x640 one.

COMPOSES WITH act_bus_geometry, DOES NOT REPLACE IT. Point training at a baked
cache AND keep ACT_GEOMETRY set: _resize_batch returns the tensor untouched when
it is already at the target, so the resize becomes a no-op while the feature
shapes are still redeclared. Keeping both means the geometry is asserted at
train time no matter which cache someone points at -- a stale full-res cache
gets silently resized rather than silently trained wrong.

A camera whose target equals its source resolution is SYMLINKED, not re-encoded:
no second JPEG generation, no disk, no time.

THE ONE LOSS, STATED SO NOBODY CHASES IT. Re-encoding adds a second JPEG pass on
top of the one predecode_ffmpeg.py already did. At q=95 on an image that has just
been downscaled 2-2.7x -- which removes the high-frequency content JPEG artifacts
live in -- this is far below the 2x scale error being fixed. It is not zero.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openpi_client.image_tools import resize_with_pad  # noqa: E402
from tools.act_bus_geometry import GEOMETRIES  # noqa: E402


def _one(args) -> int:
    src, dst, th, tw, quality = args
    img = np.asarray(Image.open(src).convert("RGB"))
    out = resize_with_pad(img, th, tw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst, "JPEG", quality=quality, subsampling=0)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--geometry", required=True, choices=sorted(GEOMETRIES))
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if --dest already exists")
    a = ap.parse_args(argv)

    plan = GEOMETRIES[a.geometry]
    src_videos = a.source / "videos"
    if not src_videos.is_dir():
        print(f"FATAL: {src_videos} is not a directory", file=sys.stderr)
        return 1
    if a.dest.exists():
        if not a.force:
            print(f"{a.dest} exists — pass --force to rebuild. Nothing done.")
            return 0
        shutil.rmtree(a.dest)

    jobs: list[tuple] = []
    for cam_dir in sorted(src_videos.iterdir()):
        if not cam_dir.is_dir():
            continue
        key = cam_dir.name
        if key not in plan:
            print(f"FATAL: camera {key!r} has no target in geometry "
                  f"{a.geometry!r} (covers {sorted(plan)})", file=sys.stderr)
            return 1
        th, tw = plan[key]
        probe = next(cam_dir.rglob("f*.jpg"), None)
        if probe is None:
            print(f"FATAL: no JPGs under {cam_dir}", file=sys.stderr)
            return 1
        h, w = np.asarray(Image.open(probe)).shape[:2]
        dst_dir = a.dest / "videos" / key
        if (h, w) == (th, tw):
            # Already right. Symlink the whole camera: no re-encode, no second
            # JPEG generation, no disk.
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            dst_dir.symlink_to(cam_dir.resolve(), target_is_directory=True)
            print(f"  {key:32} {h}x{w} already correct — symlinked")
            continue
        n = 0
        for jpg in cam_dir.rglob("f*.jpg"):
            jobs.append((jpg, a.dest / "videos" / key / jpg.relative_to(cam_dir),
                         th, tw, a.quality))
            n += 1
        print(f"  {key:32} {h}x{w} -> {th}x{tw}   {n} frames")

    if jobs:
        print(f"re-encoding {len(jobs)} frames on {a.workers} workers...", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = sum(ex.map(_one, jobs, chunksize=64))
        if done != len(jobs):
            print(f"FATAL: wrote {done} of {len(jobs)}", file=sys.stderr)
            return 1

    size = sum(f.stat().st_size for f in a.dest.rglob("f*.jpg") if not f.is_symlink())
    print(f"done: {a.dest}  ({size / 2**30:.2f} GiB of re-encoded frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
