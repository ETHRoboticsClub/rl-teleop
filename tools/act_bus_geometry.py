#!/usr/bin/env python3
"""Train ACT at the geometry the BUS publishes, not the geometry the mp4s hold.

    ACT_GEOMETRY=bus  python tools/train_act_dark_noise.py ...

WHY THIS EXISTS. Two pipelines set image geometry independently and nothing
joins them back up:

  training   export_lerobot.probe_shapes() decodes one frame from the FULL-RES
             mp4 on disk and writes that into meta/info.json. The checkpoint
             declares it: wrist 480x640, top 720x1280.
  deploy     publish_resize in configs/yam/cameras_right_top.yaml shrinks the
             frame BEFORE it hits the bus: wrist 240x320, top 270x480.
             act_runner.build_observation does /255 and HWC->CHW and nothing
             else; the checkpoint's 4-step preprocessor is rename -> batch ->
             device -> normalize. Nothing resizes, at either end.

So every right-arm checkpoint has been deployed at 1/2 (wrist) and 1/2.67 (top)
of its training image scale. It cannot raise: ACT's ResNet18 is fully
convolutional and its 2-D position embedding is generated from whatever feature
map arrives, so any resolution produces a valid action chunk. Measured cost on
hardware: ~57-65 mm of commanded tip x, z untouched -- a LATERAL targeting
error, which is exactly the empty-grasp signature.

WHAT THIS MODULE DOES. Resizes frames to the deploy geometry as they leave the
decoder, and rewrites the dataset's declared feature shapes to match, so the
checkpoint declares what it was actually trained on. The dataset on disk is
never touched -- this is a read path change, so it costs no disk and no
re-export, and reverts by unsetting one env var.

WHY NOT UPSCALE AT DEPLOY INSTEAD. Upstream, a resize inserted into the
inference path only (lerobot PR #4345 / issue #2980) took a working policy from
80-90% success to ~0%. F.interpolate upscaling is not the inverse of native
sensor detail: it produces a blurred image with the wrong texture statistics
where training saw real pixels. Fix the training side, keep the deploy path a
straight pass-through.

WHY THE SAME FUNCTION, NOT AN EQUIVALENT ONE. cv2.resize(INTER_LINEAR),
PIL.BILINEAR and torchvision.Resize(antialias=True) give measurably different
pixels on identical inputs. camera_node publishes through
openpi_client.image_tools.resize_with_pad (mode="pad"), so this imports that
exact function rather than reimplementing it. `--selftest` proves the two agree
byte-for-byte.

RESIDUAL, STATED SO NOBODY CHASES IT LATER. Training frames are JPEG/H.264
decoded and then downscaled; bus frames are raw sensor pixels and then
downscaled. That codec difference cannot be removed without recording the
downscaled stream, and it is orders of magnitude below the 2x scale error being
fixed here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# The function camera_node publishes through, imported rather than reimplemented.
from openpi_client.image_tools import resize_with_pad

# camera key -> (height, width) the policy should see. Keys are the LeRobot
# feature names, which are also the directory names under videos/, so a frame
# can be routed by its path with no guessing.
GEOMETRIES: dict[str, dict[str, tuple[int, int]]] = {
    # What the bus publishes TODAY. Deployable with zero changes to the rig:
    # cameras_right_top.yaml stays as it is, act_runner stays a pass-through.
    "bus": {
        "observation.images.wrist": (240, 320),
        "observation.images.top": (270, 480),
    },
    # The wrist at ACT/ALOHA canon (480x640 = 300 tokens, the paper's exact
    # configuration) and the top left where the bus already has it. Needs
    # `publish_resize` DELETED from the camera_right node before deploying --
    # see the note in DEPLOY_REQUIREMENTS below.
    "wrist_native": {
        "observation.images.wrist": (480, 640),
        "observation.images.top": (270, 480),
    },
}

DEPLOY_REQUIREMENTS = {
    "bus": "none -- cameras_right_top.yaml already publishes exactly this.",
    "wrist_native": (
        "DELETE publish_resize + publish_resize_mode from the camera_right node "
        "in configs/yam/cameras_right_top.yaml, restart the `cams` session (NOT "
        "`arm`), and verify with tools/check_streams.py --watch 120 that the "
        "wrist holds 30 Hz at 6.9 -> 27.6 MB/s through the ZMQ proxy."
    ),
}


def _resolve(name: str) -> dict[str, tuple[int, int]]:
    if name not in GEOMETRIES:
        raise SystemExit(
            f"ACT_GEOMETRY={name!r} is not a geometry; choose from "
            f"{sorted(GEOMETRIES)} (or leave it unset to train at the mp4 "
            f"resolution, which is what every checkpoint before 2026-08-15 did)")
    return GEOMETRIES[name]


def _camera_key_from_path(video_path) -> str | None:
    """videos/observation.images.wrist/chunk-000/file-000.mp4 -> the key.

    Routing by PATH and not by frame shape on purpose: two cameras that happen
    to share a resolution would silently share a target, and this rig has had
    exactly that (both Innomakers at 640x480, distinguishable only by port).
    """
    for part in Path(video_path).parts:
        if part.startswith("observation.images."):
            return part
    return None


def _resize_batch(frames: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """(T,C,H,W) float32 in [0,1] -> same, resized to `target`.

    Goes through uint8 because that is what the bus carries: the camera node
    resizes raw uint8 and publishes uint8, and act_runner divides by 255 only
    afterwards. Round-tripping here costs nothing -- x/255 is exact for uint8
    inputs, so (x/255*255).round() recovers the original byte -- and it keeps
    the training tensor on the same arithmetic path as the deployed one.
    """
    if tuple(frames.shape[-2:]) == target:
        return frames
    th, tw = target
    u8 = (frames * 255.0).round().clamp_(0, 255).to(torch.uint8)
    hwc = u8.permute(0, 2, 3, 1).contiguous().numpy()          # (T,H,W,C)
    out = resize_with_pad(hwc, th, tw)                          # PIL bilinear
    out = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2)
    return out.to(torch.float32).div_(255.0)


def install(name: str | None = None) -> dict[str, tuple[int, int]] | None:
    """Patch the decode path and the declared feature shapes. Idempotent-ish:
    call once, before the dataset is built and after predecoded_patch."""
    name = name if name is not None else os.environ.get("ACT_GEOMETRY", "")
    if not name or name == "none":
        return None
    plan = _resolve(name)

    from lerobot.datasets import lerobot_dataset as _ld
    from lerobot.datasets import video_utils as _vu
    from lerobot.scripts import lerobot_train as _lt

    # Wrap whatever is installed NOW, so this composes with predecoded_patch
    # instead of racing it. Import order is the contract: predecoded_patch
    # rebinds decode_video_frames on both modules, and this wraps that binding.
    inner = _vu.decode_video_frames
    if inner is not _ld.decode_video_frames:
        raise SystemExit(
            "video_utils.decode_video_frames and lerobot_dataset."
            "decode_video_frames are not the same object -- something patched "
            "one and not the other. Refusing to wrap a split decode path.")

    def _decode(video_path, timestamps, tolerance_s, backend=None):
        frames = inner(video_path, timestamps, tolerance_s, backend)
        key = _camera_key_from_path(video_path)
        if key is None:
            raise RuntimeError(
                f"cannot route {video_path} to a camera: no path component "
                f"starts with 'observation.images.'")
        if key not in plan:
            # Loud, not silent. A camera with no target would train at mp4
            # resolution while its neighbour trained at bus resolution -- the
            # exact class of bug this module exists to close.
            raise RuntimeError(
                f"camera {key!r} has no target in geometry {name!r} "
                f"(covers {sorted(plan)}). Add it to GEOMETRIES or train "
                f"without ACT_GEOMETRY.")
        return _resize_batch(frames, plan[key])

    _vu.decode_video_frames = _decode
    _ld.decode_video_frames = _decode

    # The frames now differ from what meta/info.json says. Fix the declaration
    # at the point the dataset is handed over, so dataset_to_policy_features()
    # -- and therefore the checkpoint's config.json, and therefore the shape
    # assert act_runner will do at load time -- describes the real tensor.
    _make_dataset = _lt.make_dataset

    def _make_dataset_patched(cfg, *a, **kw):
        ds = _make_dataset(cfg, *a, **kw)
        metas = [ds.meta] if hasattr(ds, "meta") else [d.meta for d in ds._datasets]
        for meta in metas:
            present = [k for k in meta.features if k.startswith("observation.images.")]
            missing = [k for k in present if k not in plan]
            if missing:
                raise SystemExit(
                    f"dataset has camera(s) {missing} with no target in geometry "
                    f"{name!r}. Every camera must be declared or none.")
            for key in present:
                th, tw = plan[key]
                old = tuple(meta.features[key]["shape"])
                meta.features[key]["shape"] = (th, tw, 3)
                if key in meta.info.get("features", {}):
                    meta.info["features"][key]["shape"] = (th, tw, 3)
                print(f"  {key:32} {old[0]}x{old[1]} -> {th}x{tw}", flush=True)
        return ds

    _lt.make_dataset = _make_dataset_patched

    print("=" * 66)
    print(f"geometry: {name} — frames resized on read, shapes redeclared")
    for key, (h, w) in sorted(plan.items()):
        print(f"  {key:32} -> {h}x{w}")
    print(f"  deploy requirement: {DEPLOY_REQUIREMENTS[name]}")
    print("=" * 66, flush=True)
    return plan


# ── selftest ────────────────────────────────────────────────────────────────
def _selftest() -> int:
    """Prove the resize is the camera node's, and that routing works, before
    committing hours of GPU to it."""
    ok = True

    # 1. byte-identical to what camera_node publishes
    from robots_realtime.runtime.environment.camera_node import _RESIZE_MODES
    rng = np.random.default_rng(0)
    for (h, w), (th, tw) in ((480, 640), (240, 320)), ((720, 1280), (270, 480)):
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        theirs = _RESIZE_MODES["pad"](img, th, tw)
        f = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        mine = (_resize_batch(f, (th, tw))[0].permute(1, 2, 0) * 255).round().numpy().astype(np.uint8)
        same = theirs.shape == mine.shape and np.array_equal(theirs, mine)
        print(f"  {h}x{w} -> {th}x{tw}  camera_node match: {same}")
        if not same:
            d = np.abs(theirs.astype(int) - mine.astype(int))
            print(f"    FAIL max|diff|={d.max()} shapes {theirs.shape} vs {mine.shape}")
            ok = False
        # letterbox must be a no-op at these exact-aspect targets
        if (h / w) != (th / tw):
            print(f"    FAIL {h}x{w}->{th}x{tw} changes aspect; pad bars would appear")
            ok = False

    # 2. routing
    for p, want in (
        ("videos/observation.images.wrist/chunk-000/file-000.mp4", "observation.images.wrist"),
        ("/x/y/videos/observation.images.top/chunk-000/file-003.mp4", "observation.images.top"),
        ("videos/chunk-000/file-000.mp4", None),
    ):
        got = _camera_key_from_path(p)
        print(f"  route {p[-46:]:46} -> {got}")
        if got != want:
            print(f"    FAIL expected {want}")
            ok = False

    # 3. a no-op target must return the identical tensor object
    f = torch.rand(1, 3, 480, 640)
    if _resize_batch(f, (480, 640)) is not f:
        print("    FAIL no-op resize copied the tensor")
        ok = False

    # 4. every geometry names a deploy requirement
    for g in GEOMETRIES:
        if g not in DEPLOY_REQUIREMENTS:
            print(f"    FAIL geometry {g!r} has no DEPLOY_REQUIREMENTS entry")
            ok = False

    print("  ALL OK" if ok else "  BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(_selftest())
