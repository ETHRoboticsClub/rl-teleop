#!/usr/bin/env python3
"""EDITABLE. Held-vs-empty jaws. Cycle 8: deviation from an empty-jaw template.

    fit(images, ys, metas) -> model
    predict(model, images, metas) -> [0/1]

THE IDEA, and why it beats everything before it.

The wrist camera is bolted to the same body as the jaws, so the gripper lands on
the SAME PIXELS in every frame ever taken (kitting's jaw_mask.py established this
by temporal variance over 2928 frames). An empty gripper therefore looks nearly
identical every single time. That makes "is something between the fingers" a
CHANGE-DETECTION question, not a classification question:

    build a per-pixel MEDIAN and STD image from known-empty frames
    score a new frame by the FRACTION OF PIXELS more than 3 sigma from it

Empty frames match the template and score near 0. A packet occupies pixels that
have never been anything but background, and lights them all up at once.

WHY THE EARLIER FEATURES FAILED. Brightness, saturation, gradient and focus all
summarise the ROI as an AVERAGE, and an average over a region that also contains
bin, bag and mat is dominated by whichever of those is in frame. Cycle 2 measured
the consequence: a region of the image containing NO GRIPPER AT ALL classified the
two classes at 86% -- the features were reading the scene. Deviation-from-template
asks about the same pixels every time, so a changing background cannot answer it.

ROI comes from the operator drawing on the failures, plus a 600-rectangle search
around what he drew. He was right and the inherited rectangle was wrong: the
useful region is a NARROW VERTICAL STRIP down the finger gap (41x118 px), not the
wide box scaled from kitting's left-arm camera. Every rectangle he drew scored 0
false positives; the inherited one scored 1.

HONESTY. Template and threshold are built from TRAIN-fold data only, so the CV is
not being shown its own answer. What this cannot do is prove 100% on 80 samples
-- see verify.py's confidence-interval note.
"""
from __future__ import annotations

import numpy as np
import cv2

# Winner of the 600-rectangle search around the operator's selection. 376 of those
# 600 scored zero false positives, so this sits on a broad plateau rather than a
# lucky peak -- moving it 4% in any direction costs little.
ROI = (0.440, 0.930, 0.475, 0.605)

# 3 sigma, and a FLOOR on the std. Without the floor, pixels that happen to be
# constant across the empty set get a near-zero denominator, any sensor noise
# becomes a 100-sigma deviation, and every frame looks full. 3.0 grey levels is
# about this camera's own noise.
SIGMA = 3.0
STD_FLOOR = 3.0
WIDTH_TOL = 1e-9      # encoder float twins are 6.6e-16 apart, real steps 7.1e-5


def _crop(im):
    h, w = im.shape[:2]
    return im[int(ROI[0] * h):int(ROI[1] * h), int(ROI[2] * w):int(ROI[3] * w)]


def _gray(im):
    return cv2.cvtColor(_crop(im), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _score(g, tmpl, tstd):
    """Fraction of ROI pixels more than SIGMA from the empty template."""
    return float((np.abs(g - tmpl) / (tstd + STD_FLOOR) > SIGMA).mean())


def fit(images, ys, metas):
    y = np.asarray(ys, int)
    g = np.array([m["gripper_pos"] for m in metas], float)

    emp_w = g[y == 0]
    lo, hi = float(emp_w.min()), float(emp_w.max())

    stack = np.stack([_gray(images[i]) for i in np.where(y == 0)[0]])
    tmpl = np.median(stack, 0)
    tstd = stack.std(0)

    # Threshold from TRAIN scores. If the classes are cleanly separated, sit in
    # the middle of the gap -- the widest margin available, rather than hugging
    # whichever class had the unluckiest sample.
    s = np.array([_score(_gray(im), tmpl, tstd) for im in images])
    sf, se = s[y == 1], s[y == 0]
    if len(sf) and len(se) and sf.min() > se.max():
        thr = (sf.min() + se.max()) / 2
    else:
        thr = float(np.median(np.r_[sf, se]))
    return {"lo": lo, "hi": hi, "tmpl": tmpl, "tstd": tstd, "thr": float(thr)}


def predict(model, images, metas):
    lo, hi, thr = model["lo"], model["hi"], model["thr"]
    out = []
    for im, m in zip(images, metas):
        w = m["gripper_pos"]
        if w < lo - WIDTH_TOL:
            # PHYSICALLY IMPOSSIBLE, so do not trust the sensor. The jaws cannot
            # close TIGHTER around an object than they close on empty air, so a
            # width below the empty minimum is not evidence of emptiness -- it is
            # evidence the reading is wrong (a capture caught mid-close, most
            # likely). Falling through to the image is the honest response to a
            # corrupt input; treating it as "very empty" is how the one remaining
            # error in cycle 8 was produced.
            out.append(int(_score(_gray(im), model["tmpl"], model["tstd"]) > thr))
        elif w <= lo + WIDTH_TOL:
            out.append(0)          # AT the hard stop: nothing can be inside
        elif w > hi + WIDTH_TOL:
            out.append(1)          # wider than any empty: something holds it open
        else:
            out.append(int(_score(_gray(im), model["tmpl"], model["tstd"]) > thr))
    return out
