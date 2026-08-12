#!/usr/bin/env python3
"""Re-freeze labels.json + folds.json from the live capture manifest.

    ../.venv/bin/python3 jaw_loop/refreeze.py          # show what would change
    ../.venv/bin/python3 jaw_loop/refreeze.py --write  # actually re-freeze

WHY THIS IS A SEPARATE, DELIBERATE STEP. verify.py reads the FROZEN labels and
the FROZEN folds, never the live manifest. That is the whole reason the score
means anything: if the split moved every time the detector was scored, "improved
by 3 points" could just as easily be "found a friendlier split". So new captures
do NOT silently enter the evaluation. Somebody has to say so.

Re-freezing invalidates comparison with earlier cycles, because the test set has
changed. experiments.md rows from before a re-freeze are still true, but they are
true about a different set -- the log records the sample count for that reason.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAN = ROOT / "jaw_dataset" / "manifest.jsonl"
WRITE = "--write" in sys.argv

rows = [json.loads(l) for l in open(MAN) if l.strip()]
rows = [r for r in rows if not r.get("deleted")]
rows.sort(key=lambda r: r["id"])

old = json.load(open(HERE / "labels.json")) if (HERE / "labels.json").exists() else []
print(f"frozen now : {len(old)} captures")
print(f"live now   : {len(rows)} captures  (+{len(rows)-len(old)})")
c = Counter((r["label"], r.get("band")) for r in rows)
print("\nlive breakdown (label, band):")
for k in sorted(c, key=str):
    print(f"  {k[0]:<6} {str(k[1]):<7} {c[k]}")
inband_full = c[("full", "in")]
print(f"\nFULL captures in the band: {inband_full}   (target ~20 -- these are the scarce ones)")

if not WRITE:
    print("\ndry run. add --write to re-freeze.")
    raise SystemExit(0)

labels = [{"id": r["id"], "file": r["file"], "y": 1 if r["label"] == "full" else 0,
           "gripper_pos": r["state"]["gripper_pos"], "joint_pos": r["state"]["joint_pos"],
           "band": r.get("band")} for r in rows]
json.dump(labels, open(HERE / "labels.json", "w"), indent=1)

# Stratify on (label, band) rather than label alone: the in-band FULLs are the
# scarce, load-bearing class, and a plain label-stratified split can put nearly
# all of them in one fold, which makes the CV number swing on where they landed.
y = np.array([r["y"] for r in labels])
strat = [f"{r['y']}_{r.get('band')}" for r in labels]
rng = np.random.default_rng(20260812)
folds = np.empty(len(y), int)
for s in sorted(set(strat)):
    idx = np.array([i for i, v in enumerate(strat) if v == s])
    rng.shuffle(idx)
    for k, i in enumerate(idx):
        folds[i] = k % 5
json.dump([int(f) for f in folds], open(HERE / "folds.json", "w"))
print(f"\nre-frozen: {len(labels)} captures, 5 folds stratified on (label, band)")
print("per-fold full-in-band:",
      [int(sum(1 for i in range(len(labels)) if folds[i] == k and labels[i]["y"] == 1
               and labels[i]["band"] == "in")) for k in range(5)])
