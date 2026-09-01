#!/usr/bin/env python3
"""THE canonical target-dot renderer for target-conditioned ACT.

One function, used by BOTH sides of the pipeline:

  * training: tools/targetdot/burn_dots.py burns the auto-labeled target point
    into every predecoded wrist frame before the target-dot training run;
  * inference: the deployment path (selector/operator click -> SAM2 track ->
    render) MUST call this same function on the wrist frame before the policy
    forward pass. The research notes (target-selector-offline worktree,
    research/target-conditioning-offline.md §3; prior doc §2 "rendering style at
    train/inference must match exactly") make matching styles a correctness
    requirement, not a nicety — do not reimplement or restyle the dot at
    deploy time; import this.

Style: a filled disk, radius 10 px on the native 640x480 wrist frame, pure
magenta (R=255, G=0, B=255), no outline, no alpha blending. Rationale:

  * a rendered POINT label is SutureBot's (arXiv 2510.20965) winning goal
    format for ACT — beat mask and distance-map conditioning;
  * magenta does not occur on the rig (mat is dark, bin is red, finger pads
    are orange, packets are white/blue/yellow) and, having R == B, it is
    INVARIANT under RGB<->BGR confusion — an entire channel-order bug class
    (cv2 vs torchvision) cannot silently move the cue;
  * radius 10 at 640x480 survives the dark_noise train-time augmentation
    (gaussian noise, blur sigma <= 2, contrast jitter) while staying well
    inside a packet's ~100 px footprint.
"""
from __future__ import annotations

import numpy as np

TARGET_DOT_RADIUS = 10
# (R, G, B) == (B, G, R): symmetric on purpose, see module docstring.
TARGET_DOT_COLOR = (255, 0, 255)


def draw_target_dot(img: np.ndarray, x: float, y: float,
                    radius: int = TARGET_DOT_RADIUS,
                    color: tuple[int, int, int] = TARGET_DOT_COLOR) -> np.ndarray:
    """Draw the target dot IN PLACE on an HxWx3 uint8 image; returns img.

    ``x``/``y`` are pixel coordinates in the image's own geometry (for this
    project: the native 640x480 wrist frame — burn the dot BEFORE any resize).
    A dot centered outside the frame is clipped like any other disk; callers
    pass the target point they have, unconditionally.
    """
    h, w = img.shape[:2]
    cx, cy = float(x), float(y)
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(w, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(h, int(np.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][disk] = np.asarray(color, dtype=img.dtype)
    return img
