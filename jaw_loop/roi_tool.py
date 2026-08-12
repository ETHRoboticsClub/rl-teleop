#!/usr/bin/env python3
"""Interactive ROI picker over the captures the detector gets WRONG.

    ../.venv/bin/python3 jaw_loop/roi_tool.py   &&   open :8804

The current ROI was inherited: kitting's jaw_mask.py measured the gripper by
temporal variance on 192x144 LEFT-arm frames, and it was scaled to our 240x320
right-wrist capture. Nobody checked it lands on the packet here. Since the camera
is bolted to the gripper, the right rectangle is a CONSTANT of this rig, so it is
worth getting right once by eye rather than inferring it from another camera.

Shows the misclassified captures first, because a rectangle that makes those
legible is the rectangle worth having. The drawn box persists as you page through
images -- a fixed ROI has to work on ALL of them, not on a favourite one.
"""
import base64
import json
import sys
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

wrong = [i for i in range(len(Y)) if pred[i] != Y[i]]
right_full = [i for i in range(len(Y)) if Y[i] == 1 and pred[i] == 1][:8]
right_empty = [i for i in range(len(Y)) if Y[i] == 0 and pred[i] == 0][:8]
order = wrong + right_full + right_empty

H, W = IM[0].shape[:2]
cur = detector.ROI  # (y0, y1, x0, x1) fractions


def b64(i):
    return "data:image/jpeg;base64," + base64.b64encode(
        (ROOT / "jaw_dataset" / L[i]["file"]).read_bytes()).decode()


items = [{"src": b64(i), "file": L[i]["file"],
          "truth": "FULL" if Y[i] == 1 else "EMPTY",
          "pred": "FULL" if pred[i] == 1 else "EMPTY",
          "ok": bool(pred[i] == Y[i]),
          "g": round(L[i]["gripper_pos"], 6)} for i in order]

html = f"""<title>ROI Picker</title>
<style>
:root{{--bg:#101014;--fg:#eceaea;--mut:#8e8e94;--line:#2a2a31;--card:#181820;--bad:#e0555a;--ok:#4fbf7c;--sel:#ffd970}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
.wrap{{max-width:1150px;margin:0 auto;padding:1.25rem 1rem 4rem}}
h1{{font-size:1.25rem;margin:0 0 .2rem}}
.sub{{color:var(--mut);font-size:.87rem;margin:0 0 1.2rem}}
.main{{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:1.2rem;align-items:start}}
@media(max-width:900px){{.main{{grid-template-columns:1fr}}}}
#stage{{position:relative;background:#000;border:1px solid var(--line);border-radius:10px;overflow:hidden;
user-select:none;cursor:crosshair}}
#stage img{{width:100%;display:block;-webkit-user-drag:none;user-drag:none;pointer-events:none}}
#old,#sel{{position:absolute;pointer-events:none}}
#old{{border:1px dashed #6b6b72}}
#sel{{border:2px solid var(--sel);background:rgba(255,217,112,.14)}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;margin:0 0 .8rem}}
.box h3{{margin:0 0 .5rem;font-size:.9rem;color:var(--mut);font-weight:600}}
pre{{margin:0;font:.82rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;
background:#0c0c10;border:1px solid var(--line);border-radius:7px;padding:.6rem .7rem;color:var(--sel)}}
button{{font:600 .85rem ui-sans-serif,system-ui,sans-serif;padding:.55rem .8rem;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;margin:.5rem .4rem 0 0}}
button:hover{{border-color:#4a4a55}}
.meta{{font:.75rem/1.5 ui-monospace,monospace;color:var(--mut)}}
.strip{{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:.5rem;margin-top:1.3rem}}
.strip figure{{margin:0;border:2px solid var(--line);border-radius:7px;overflow:hidden;cursor:pointer;position:relative}}
.strip figure.bad{{border-color:var(--bad)}}
.strip figure.on{{outline:2px solid var(--sel);outline-offset:1px}}
.strip img{{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}}
.strip figcaption{{font:.62rem/1.3 ui-monospace,monospace;padding:.2rem .3rem;color:var(--mut)}}
.hint{{color:var(--mut);font-size:.8rem;margin:.5rem 0 0}}
</style>
<div class="wrap">
<h1>Pick the region that decides the grasp</h1>
<p class="sub">Drag a rectangle on the image. The dashed box is what the detector uses today
(inherited from a different camera). Your box persists as you switch images &mdash; the camera is
bolted to the gripper, so ONE rectangle has to work for every frame. The failures are shown first.</p>

<div class="main">
  <div>
    <div id="stage"><img id="img" draggable="false"><div id="old"></div><div id="sel"></div></div>
    <p class="hint" id="cap"></p>
  </div>
  <div>
    <div class="box"><h3>Paste this into Claude Code</h3>
      <pre id="out">drag a box on the image</pre>
      <button onclick="copyOut()">copy this box</button>
      <button onclick="clearSel()">clear</button>
      <button onclick="exportAll()" style="border-color:#c8a02e;color:#ffd970">EXPORT ALL &rarr; paste to Claude</button>
      <p class="meta" id="prog"></p>
    </div>
    <div class="box"><h3>Current ROI (dashed)</h3>
      <pre>ROI = ({cur[0]}, {cur[1]}, {cur[2]}, {cur[3]})</pre>
      <p class="meta">y0, y1, x0, x1 as fractions &middot; frame {W}x{H}</p>
    </div>
    <div class="box"><h3>Keys</h3>
      <p class="meta">&larr; &rarr; page through images<br>
      red border = detector got it wrong</p>
    </div>
  </div>
</div>

<div class="strip" id="strip"></div>
</div>
<script>
const items = {json.dumps(items)};
const CUR = {json.dumps(list(cur))};
const stage=document.getElementById('stage'), img=document.getElementById('img'),
      oldb=document.getElementById('old'), sel=document.getElementById('sel'),
      out=document.getElementById('out'), cap=document.getElementById('cap');
let idx=0, box=null, drag=null;
// One box PER IMAGE, but a new image inherits the last one you drew. So doing
// nothing gives you one-rectangle-for-all; adjusting on an image overrides it
// just there. At the end every image carries a box and the winner is chosen by
// measurement, not by which one you happened to draw last.
const boxes={{}}; let touched=new Set();

function show(i){{
  idx=(i+items.length)%items.length; const it=items[idx];
  if(boxes[idx]) box=Object.assign({{}},boxes[idx]);
  else if(box) boxes[idx]=Object.assign({{}},box);   // inherit
  img.src=it.src;
  cap.textContent=`${{it.file}}  |  truth ${{it.truth}}  |  detector said ${{it.pred}}  |  width ${{it.g}}`;
  cap.style.color = it.ok ? '#8e8e94' : '#e0555a';
  document.querySelectorAll('.strip figure').forEach((f,k)=>f.classList.toggle('on',k===idx));
  draw(); prog();
}}
function draw(){{
  const w=stage.clientWidth, h=img.clientHeight||stage.clientHeight;
  oldb.style.cssText=`left:${{CUR[2]*w}}px;top:${{CUR[0]*h}}px;width:${{(CUR[3]-CUR[2])*w}}px;height:${{(CUR[1]-CUR[0])*h}}px;position:absolute`;
  oldb.className='';
  if(box){{
    sel.style.display='block';
    sel.style.cssText=`left:${{box.x0*w}}px;top:${{box.y0*h}}px;width:${{(box.x1-box.x0)*w}}px;height:${{(box.y1-box.y0)*h}}px;position:absolute;border:2px solid #ffd970;background:rgba(255,217,112,.14)`;
  }} else sel.style.display='none';
}}
function frac(e){{
  const r=stage.getBoundingClientRect();
  return {{x:Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),
           y:Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))}};
}}
stage.addEventListener('dragstart',e=>e.preventDefault());
stage.addEventListener('mousedown',e=>{{e.preventDefault(); drag=frac(e); box=null; draw();}});
addEventListener('mousemove',e=>{{
  if(!drag)return; const p=frac(e);
  box={{x0:Math.min(drag.x,p.x),x1:Math.max(drag.x,p.x),y0:Math.min(drag.y,p.y),y1:Math.max(drag.y,p.y)}};
  boxes[idx]=Object.assign({{}},box); touched.add(idx);
  draw(); emit();
}});
addEventListener('mouseup',()=>{{drag=null;}});
function emit(){{
  if(!box){{out.textContent='drag a box on the image';return;}}
  const f=v=>v.toFixed(3);
  out.textContent =
   `ROI = (${{f(box.y0)}}, ${{f(box.y1)}}, ${{f(box.x0)}}, ${{f(box.x1)}})\\n`+
   `# y0, y1, x0, x1 as fractions of a {W}x{H} frame\\n`+
   `# pixels: y ${{Math.round(box.y0*{H})}}..${{Math.round(box.y1*{H})}}  `+
   `x ${{Math.round(box.x0*{W})}}..${{Math.round(box.x1*{W})}}`;
}}
function copyOut(){{navigator.clipboard.writeText(out.textContent);}}
function exportAll(){{
  const rows=items.map((it,k)=>boxes[k]?{{
      file:it.file, truth:it.truth, pred:it.pred,
      y0:+boxes[k].y0.toFixed(4), y1:+boxes[k].y1.toFixed(4),
      x0:+boxes[k].x0.toFixed(4), x1:+boxes[k].x1.toFixed(4),
      adjusted:touched.has(k)}}:null).filter(Boolean);
  const txt='ROI_BOXES = '+JSON.stringify(rows,null,1);
  out.textContent=txt.slice(0,900)+(txt.length>900?' ... ('+rows.length+' boxes)':'');
  navigator.clipboard.writeText(txt);
  prog();
}}
function prog(){{
  document.getElementById('prog').textContent =
    Object.keys(boxes).length+' images have a box, '+touched.size+' adjusted by hand';
}}
function clearSel(){{box=null;draw();emit();}}
addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight'){{show(idx+1);}} else if(e.key==='ArrowLeft'){{show(idx-1);}}
}});
addEventListener('resize',draw);
img.addEventListener('load',draw);
document.getElementById('strip').innerHTML=items.map((it,k)=>
  `<figure class="${{it.ok?'':'bad'}}" onclick="show(${{k}})"><img loading=lazy src="${{it.src}}">
   <figcaption>${{it.truth}}&rarr;${{it.pred}}</figcaption></figure>`).join('');
show(0);
</script>"""

out = HERE / "roi_tool.html"
out.write_text(html)
print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
print(f"{len(wrong)} misclassified shown first, then correct examples for contrast")
print(f"current ROI (dashed): {cur}   frame {W}x{H}")
