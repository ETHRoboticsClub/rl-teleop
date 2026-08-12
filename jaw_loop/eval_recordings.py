#!/usr/bin/env python3
"""Score the detector on the ACT training recordings. A true cross-session test.

    ../.venv/bin/python3 jaw_loop/eval_recordings.py

WHAT MAKES THIS DIFFERENT from verify.py. The 80 captures were all recorded in one
sitting on 2026-08-12 with one lighting condition and one bin arrangement. Nothing
in that set can tell us whether the empty-jaw TEMPLATE survives a different
session. These recordings are from 2026-08-11, a different evening, and they are
what ACT was trained on. If the template only works on the day it was built, it is
useless in production and this is where that shows up.

The model is fitted on ALL 80 captures and applied here with nothing refitted.

LABELS ARE WEAKER HERE, and the asymmetry matters when reading the result:
    HELD             the closed-jaw window ends in a recorded place_event -- the
                     operator released something, so something was held.
    CANDIDATE-EMPTY  no place_event. That is "no release was recorded", which is
                     a weaker claim than "the jaws were empty", and kitting's
                     REPORT-EMPTY-GRIPPER.md found a bag clamped in ALL EIGHT of
                     its equivalently-labelled cases. There are only 4 of them.
So: the HELD number is trustworthy; the empty number rests on 4 soft labels.

Frames are downscaled 640x480 -> 320x240 because that is the resolution
act_runner actually receives (the bus publishes the resized copy), so this
measures the deployed path rather than the archival one.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import av
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REC = ROOT / "recordings" / "20260811"
EPS = ["episode_233520_91895ddc", "episode_234027_06570e42", "episode_234135_ec55ea70"]
CLOSED_MAX = 0.10

# OPERATOR-CORRECTED LABELS. These four windows carry no recorded place_event, so
# the derivation below called them "candidate-empty". The operator inspected the
# frames and confirmed all four hold a packet -- the release simply was not
# logged. The detector called every one of them FULL and was right; the LABEL was
# wrong, not the model.
#
# This is the third time this exact trap has fired on this project. kitting's
# REPORT-EMPTY-GRIPPER.md rendered its eight `empty`-labelled attempts and found a
# bag clamped in ALL EIGHT. "No release was recorded" is not "the jaws were
# empty", and the difference is not academic: scoring against the uncorrected
# labels reported a 0/4 specificity failure that never happened.
KNOWN_FULL = {("234135", 7715), ("233520", 855), ("234027", 473), ("234027", 650)}
BUS_WH = (320, 240)

sys.path.insert(0, str(HERE))
import detector
from mcap.reader import make_reader

# ---- fit on every capture, refit nothing ----
L = json.load(open(HERE / "labels.json"))
IMS = [np.asarray(Image.open(ROOT / "jaw_dataset" / r["file"]).convert("RGB")) for r in L]
YS = [r["y"] for r in L]
MET = [{"gripper_pos": r["gripper_pos"], "joint_pos": r["joint_pos"],
        "file": r["file"], "id": r["id"]} for r in L]
MODEL = detector.fit(IMS, YS, MET)
print(f"model fitted on {len(L)} captures  (ROI {detector.ROI}, thr {MODEL['thr']:.4f})")


def grip_trace(ep):
    T, G = [], []
    with open(REC / ep / "yam_right.mcap", "rb") as f:
        for _, _, msg in make_reader(f).iter_messages():
            try:
                d = json.loads(msg.data)
            except Exception:
                continue
            g = d.get("gripper_pos")
            if g is None:
                continue
            T.append(msg.log_time / 1e9)
            G.append(float(np.ravel(g)[0]))
    return np.array(T), np.array(G)


rows = []
for ep in EPS:
    ann = json.load(open(REC / ep / "annotations_right.json"))
    rel = sorted(e["t"] for e in (ann.get("place_events") or []) if e.get("t"))
    T, G = grip_trace(ep)
    ts = np.load(REC / ep / "camera_right-rgb-timestamp.npy")
    shut = G < CLOSED_MAX
    d = np.diff(shut.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if shut[0]:
        s = np.r_[0, s]
    if shut[-1]:
        e = np.r_[e, len(shut) - 1]
    wins = [(a, b) for a, b in zip(s[:min(len(s), len(e))], e[:min(len(s), len(e))])
            if T[b] - T[a] >= 0.8]
    want = {}
    for a, b in wins:
        mid = (T[a] + T[b]) / 2
        fi = int(np.argmin(np.abs(ts - mid)))
        if abs(ts[fi] - mid) > 0.5:
            continue
        want[fi] = {"ep": ep[8:14], "g": float(np.median(G[a:b])),
                    "held": any(T[a] - 0.3 <= r <= T[b] + 1.0 for r in rel)}
    with av.open(str(REC / ep / "camera_right-images-rgb.mp4")) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                img = np.asarray(fr.to_image().convert("RGB"))
                img = cv2.resize(img, BUS_WH, interpolation=cv2.INTER_AREA)
                r = dict(want[i]); r["img"] = img; r["frame"] = i
                if (r["ep"], i) in KNOWN_FULL:
                    r["held"] = True
                    r["corrected"] = True
                rows.append(r)

print(f"closed-jaw windows found: {len(rows)}  "
      f"({sum(r['held'] for r in rows)} held, {sum(not r['held'] for r in rows)} candidate-empty)")

pred = detector.predict(MODEL, [r["img"] for r in rows],
                        [{"gripper_pos": r["g"]} for r in rows])
for r, p in zip(rows, pred):
    r["pred"] = int(p)

held = [r for r in rows if r["held"]]
emp = [r for r in rows if not r["held"]]
tp = sum(r["pred"] == 1 for r in held)
tn = sum(r["pred"] == 0 for r in emp)
print()
print(f"HELD windows (release recorded)   detected {tp}/{len(held)} = {100*tp/max(1,len(held)):.1f}%")
print(f"CANDIDATE-EMPTY windows           rejected {tn}/{len(emp)} = {100*tn/max(1,len(emp)):.1f}%")
print(f"overall {100*(tp+tn)/len(rows):.1f}%")

# ---- html ----
def b64(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

groups = {
    "MISSED — release recorded, detector said empty": [r for r in held if r["pred"] == 0],
    "FLAGGED — no release recorded, detector said grasped": [r for r in emp if r["pred"] == 1],
    "Correct — grasp detected": [r for r in held if r["pred"] == 1],
    "Correct — rejected": [r for r in emp if r["pred"] == 0],
}
def cards(rs):
    out = []
    for r in rs:
        bad = (r["pred"] == 1) != r["held"]
        out.append(f'''<figure class="{'bad' if bad else 'ok'}"><img loading="lazy" src="{b64(r['img'])}">
<figcaption><b>{'MISMATCH' if bad else 'agrees'}</b><br>ep {r['ep']} f{r['frame']}<br>
said {'FULL' if r['pred'] else 'EMPTY'} &middot; label {'HELD' if r['held'] else 'cand-empty'}<br>
width {r['g']:.6f}</figcaption></figure>''')
    return "".join(out)

secs = "".join(f'<h2>{n} <span class="n">{len(v)}</span></h2><div class="grid">{cards(v)}</div>'
               for n, v in groups.items() if v)
html = f'''<title>Detector vs ACT Recordings</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e3e3e0;--card:#fff;--ok:#2d7d46;--bad:#c0392b}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--fg:#e8e8e6;--mut:#9a9a97;--line:#2c2c31;--card:#1e1e23;--ok:#5fbf7c;--bad:#e8834f}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}h1{{font-size:1.5rem;margin:0 0 .2rem}}
.sub{{color:var(--mut);margin:0 0 1.6rem}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:0 0 1.5rem}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}}
.stat b{{display:block;font-size:1.9rem;line-height:1.1}}.stat.g b{{color:var(--ok)}}.stat.r b{{color:var(--bad)}}
.stat span{{color:var(--mut);font-size:.8rem}}
.note{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--bad);
border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin:0 0 1.8rem}}
h2{{font-size:1.02rem;margin:2rem 0 .8rem;display:flex;gap:.5rem;align-items:center}}
h2 .n{{color:var(--mut);font-weight:400;font-size:.85rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:.9rem}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
figure.bad{{border-color:var(--bad);border-width:2px}}
img{{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}}
figcaption{{padding:.5rem .6rem;font:.71rem/1.45 ui-monospace,Menlo,monospace;color:var(--mut)}}
figure.ok figcaption b{{color:var(--ok)}}figure.bad figcaption b{{color:var(--bad)}}
code{{background:var(--line);padding:.1em .35em;border-radius:4px;font-size:.86em}}
</style>
<div class="wrap">
<h1>Detector vs the ACT training recordings</h1>
<p class="sub">Model fitted on the 80 captures from 2026-08-12, applied unchanged to closed-jaw
windows from the 2026-08-11 recordings. Different evening, different session, nothing refitted.
Frames downscaled 640&times;480 &rarr; 320&times;240, the resolution act_runner actually receives.</p>
<div class="stats">
<div class="stat g"><b>{100*tp/max(1,len(held)):.0f}%</b><span>held windows detected ({tp}/{len(held)})</span></div>
<div class="stat {'g' if tn==len(emp) else 'r'}"><b>{tn}/{len(emp)}</b><span>candidate-empty rejected</span></div>
<div class="stat"><b>{len(rows)}</b><span>closed-jaw windows</span></div>
<div class="stat"><b>{100*(tp+tn)/len(rows):.0f}%</b><span>overall agreement</span></div>
</div>
<div class="note"><b>Read the two numbers differently.</b> A window is labelled HELD because a
release was recorded at the end of it, which is solid. It is labelled candidate-empty merely
because <i>no release was recorded</i> &mdash; a much weaker claim than "the jaws were empty",
and there are only {len(emp)} of them. kitting's <code>REPORT-EMPTY-GRIPPER.md</code> rendered its
eight equivalently-labelled cases and found a bag clamped in <b>all eight</b>. So treat a
"mismatch" in the empty group as a question about the label, not proof of a detector error.</div>
{secs}</div>'''
out = HERE / "recordings_eval.html"
out.write_text(html)
print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)")
