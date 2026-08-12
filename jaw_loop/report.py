#!/usr/bin/env python3
"""Per-class results + a visual page of every decision.

Uses the SAME frozen folds as verify.py, so every prediction here is out-of-fold:
each image is judged by a model that never saw it. Reporting in-fold numbers
would be the easiest lie available and they would all be ~100%.
"""
import base64, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import detector

L = json.load(open(HERE / "labels.json"))
F = np.array(json.load(open(HERE / "folds.json")))
Y = np.array([r["y"] for r in L])
IM = [np.asarray(Image.open(ROOT / "jaw_dataset" / r["file"]).convert("RGB")) for r in L]
META = [{k: r[k] for k in ("gripper_pos", "joint_pos", "file", "id")} for r in L]

pred = np.full(len(Y), -1, int)
for k in range(5):
    te, tr = F == k, F != k
    m = detector.fit([IM[i] for i in np.where(tr)[0]], Y[tr].tolist(),
                     [META[i] for i in np.where(tr)[0]])
    pred[te] = detector.predict(m, [IM[i] for i in np.where(te)[0]],
                                [META[i] for i in np.where(te)[0]])

full, empty = Y == 1, Y == 0
tp = int((pred[full] == 1).sum()); fn = int((pred[full] == 0).sum())
tn = int((pred[empty] == 0).sum()); fp = int((pred[empty] == 1).sum())

print("ON THE POSITIVES (you pressed D, a packet was held)")
print(f"  detected as grasped : {tp}/{full.sum()}  = {100*tp/full.sum():.1f}%")
print(f"  missed              : {fn}/{full.sum()}  = {100*fn/full.sum():.1f}%  (costs a retry)")
print()
print("ON THE NEGATIVES (you pressed E, jaws empty)")
print(f"  correctly rejected  : {tn}/{empty.sum()}  = {100*tn/empty.sum():.1f}%")
print(f"  FALSELY 'grasped'   : {fp}/{empty.sum()}  = {100*fp/empty.sum():.1f}%  <- the dangerous error")
print()
print(f"overall {100*(pred==Y).mean():.1f}%")

lo = float(np.array([r["gripper_pos"] for r in L])[Y == 0].min())
hi = float(np.array([r["gripper_pos"] for r in L])[Y == 0].max())


def b64(p):
    return "data:image/jpeg;base64," + base64.b64encode((ROOT / "jaw_dataset" / p).read_bytes()).decode()


def route(g):
    if g <= lo: return "width: at/below hard stop"
    if g > hi:  return "width: above empty band"
    return "IMAGE decided (in band)"


groups = {
    "MISSED — you held a packet, it said empty": [i for i in range(len(Y)) if Y[i] == 1 and pred[i] == 0],
    "FALSE ALARM — jaws empty, it said grasped": [i for i in range(len(Y)) if Y[i] == 0 and pred[i] == 1],
    "Correct — grasp detected": [i for i in range(len(Y)) if Y[i] == 1 and pred[i] == 1],
    "Correct — empty rejected": [i for i in range(len(Y)) if Y[i] == 0 and pred[i] == 0],
}

def cards(idx):
    out = []
    for i in idx:
        bad = pred[i] != Y[i]
        out.append(f'''<figure class="{'bad' if bad else 'ok'}">
<img loading="lazy" src="{b64(L[i]['file'])}">
<figcaption><b>{'WRONG' if bad else 'right'}</b> &middot; said {'FULL' if pred[i] else 'EMPTY'}<br>
width {L[i]['gripper_pos']:.6f}<br><span class="rt">{route(L[i]['gripper_pos'])}</span></figcaption></figure>''')
    return "".join(out)

secs = "".join(f'<h2>{name} <span class="n">{len(idx)}</span></h2><div class="grid">{cards(idx)}</div>'
                for name, idx in groups.items() if idx)

html = f'''<title>Jaw Detector Results</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e3e3e0;--card:#fff;--ok:#2d7d46;--bad:#c0392b}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--fg:#e8e8e6;--mut:#9a9a97;--line:#2c2c31;--card:#1e1e23;--ok:#5fbf7c;--bad:#e8834f}}}}
:root[data-theme=dark]{{--bg:#16161a;--fg:#e8e8e6;--mut:#9a9a97;--line:#2c2c31;--card:#1e1e23;--ok:#5fbf7c;--bad:#e8834f}}
*{{box-sizing:border-box}}
body{{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 .2rem;letter-spacing:-.02em}}
.sub{{color:var(--mut);margin:0 0 1.6rem}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:0 0 1.6rem}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}}
.stat b{{display:block;font-size:1.9rem;letter-spacing:-.02em;line-height:1.1}}
.stat.g b{{color:var(--ok)}} .stat.r b{{color:var(--bad)}}
.stat span{{color:var(--mut);font-size:.8rem}}
.why{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--bad);
border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin:0 0 1.6rem}}
.why h3{{margin:0 0 .5rem;font-size:1rem}}
.why p{{margin:.5rem 0}}
code{{background:var(--line);padding:.1em .35em;border-radius:4px;font-size:.86em}}
h2{{font-size:1.02rem;margin:2.2rem 0 .8rem;display:flex;align-items:center;gap:.5rem}}
h2 .n{{color:var(--mut);font-weight:400;font-size:.85rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.9rem}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
figure.bad{{border-color:var(--bad);border-width:2px}}
img{{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}}
figcaption{{padding:.5rem .6rem;font:.72rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut)}}
figure.ok figcaption b{{color:var(--ok)}} figure.bad figcaption b{{color:var(--bad)}}
.rt{{opacity:.75}}
</style>
<div class="wrap">
<h1>Jaw detector &mdash; every decision</h1>
<p class="sub">All 50 captures, judged out-of-fold: each image scored by a model that never saw it.
Red border = the detector was wrong.</p>

<div class="stats">
<div class="stat g"><b>{100*tp/full.sum():.0f}%</b><span>grasps detected ({tp}/{full.sum()})</span></div>
<div class="stat g"><b>{100*tn/empty.sum():.0f}%</b><span>empties rejected ({tn}/{empty.sum()})</span></div>
<div class="stat {'g' if fp==0 else 'r'}"><b>{fp}</b><span>false alarms (said grasped, was empty)</span></div>
<div class="stat r"><b>{fn}</b><span>missed grasps (costs a retry)</span></div>
</div>

<div class="why">
<h3>Why it is not 100%</h3>
<p><b>The jaws close the same amount on a thin packet as on nothing.</b> Gripper width splits
the set cleanly at both ends &mdash; at or below <code>{lo:.6f}</code> nothing can be inside, above
<code>{hi:.6f}</code> something must be. Between those two numbers the width tells us nothing, and
<b>31 of the 50 captures land there</b>.</p>
<p><b>Inside that band the images have a shortcut, not a signal.</b> A region of the frame
containing <i>no gripper at all</i> classifies your two classes at 86%. The strongest image
feature is saturation, and it is largely reading the red bin behind the jaws rather than what
is between them. So the image cannot be trusted to break the tie on its own.</p>
<p><b>And the band is starved of positives:</b> 26 empty against 5 full. Catching 2 more of
those 5 means loosening a boundary with 26 empties pressed against it &mdash; which is exactly
how false alarms get made. On this data, zero false alarms and 95% accuracy cannot both hold.</p>
<p>What fixes it: about 20 more <b>full</b> captures inside the band (thin packets that barely
move the jaws), and paired captures &mdash; D holding a packet, then E at the same pose without
moving &mdash; so the background is identical across the pair and cannot carry the label.</p>
</div>
{secs}
</div>'''

out = HERE / "results.html"
out.write_text(html)
print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
