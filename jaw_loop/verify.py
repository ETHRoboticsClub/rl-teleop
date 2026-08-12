#!/usr/bin/env python3
"""FROZEN VERIFIER. The generator may not edit this file, labels.json or folds.json.

    ../.venv/bin/python3 jaw_loop/verify.py          # exit 0 = target met

Scores detector.py by 5-fold cross-validation on a split frozen at creation time.
Every fold is fitted on 4 folds and scored on the held-out one, so nothing is ever
scored on data it was fitted on.

WHY CV AND NOT ONE HOLDOUT. 50 samples. A single 80/20 holdout puts 10 images on
the scale and one flipped image moves the headline by 10 points, which is noise
wearing a decimal point. CV scores every image exactly once out-of-fold.

WHAT COUNTS AS FAILURE, and why they are not symmetric:
    false positive   predicted FULL, actually EMPTY. The arm believes it grasped
                     something it did not, carries air to the mat and 'places' it.
                     This is the bug that started all of this. Weighted hardest.
    false negative   predicted EMPTY, actually FULL. Costs a needless retry.
Target: ZERO false positives and >= 0.95 accuracy. 100% is the ask; 95% is the
floor that still ships.

THE HONEST CEILING, stated here so no cycle can quietly forget it: zero errors on
50 samples is a 95% confidence interval of roughly [0, 7%] on the true error rate.
This harness CANNOT establish 100%. It can establish "no worse than ~7% and no
false positives observed", which is a real result and not the same claim.

ANTI-GAMING. The generator sees this file's OUTPUT, never its internals at fit
time; it cannot drop samples (the label set is read here, not there), cannot pick
a split (folds.json is frozen), and cannot see the test fold's labels inside
fit(). If a detector needs a threshold, it must derive it from the TRAIN fold.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = json.load(open(HERE / "labels.json"))
FOLDS = np.array(json.load(open(HERE / "folds.json")))
Y = np.array([r["y"] for r in LABELS])

TARGET_ACC = 0.95
TARGET_FP = 0

sys.path.insert(0, str(HERE))
import detector  # noqa: E402  the ONE editable file


def load_images():
    from PIL import Image
    out = []
    for r in LABELS:
        out.append(np.asarray(Image.open(ROOT / "jaw_dataset" / r["file"]).convert("RGB")))
    return out


def main() -> int:
    imgs = load_images()
    meta = [{k: r[k] for k in ("gripper_pos", "joint_pos", "file", "id")} for r in LABELS]
    pred = np.full(len(Y), -1, int)
    t0 = time.time()

    for k in range(5):
        te = FOLDS == k
        tr = ~te
        model = detector.fit([imgs[i] for i in np.where(tr)[0]],
                             Y[tr].tolist(),
                             [meta[i] for i in np.where(tr)[0]])
        p = detector.predict(model,
                             [imgs[i] for i in np.where(te)[0]],
                             [meta[i] for i in np.where(te)[0]])
        p = np.asarray(p, int)
        if p.shape != (te.sum(),) or not set(np.unique(p)) <= {0, 1}:
            print(f"FAIL fold {k}: predict() must return {int(te.sum())} values in {{0,1}}")
            return 2
        pred[te] = p

    acc = float((pred == Y).mean())
    fp = int(((pred == 1) & (Y == 0)).sum())
    fn = int(((pred == 0) & (Y == 1)).sum())
    dt = time.time() - t0

    print(f"accuracy      {acc*100:.1f}%   ({int((pred==Y).sum())}/{len(Y)})")
    print(f"false POS     {fp}   (said FULL, was EMPTY  <- the dangerous one)")
    print(f"false NEG     {fn}   (said EMPTY, was FULL)")
    print(f"fit+predict   {dt:.1f}s total over 5 folds")
    if fp or fn:
        print("\nmisses:")
        for i in np.where(pred != Y)[0]:
            kind = "FP" if Y[i] == 0 else "FN"
            print(f"  {kind}  {LABELS[i]['file']}   gripper_pos {LABELS[i]['gripper_pos']:.6f}")

    ok = (fp <= TARGET_FP) and (acc >= TARGET_ACC)
    print(f"\n{'PASS' if ok else 'FAIL'}  (need FP<={TARGET_FP} and acc>={TARGET_ACC*100:.0f}%)")
    print(f"note: 0 errors on {len(Y)} samples => true error rate 95% CI ~[0, 7%]. "
          f"This cannot prove 100%.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
