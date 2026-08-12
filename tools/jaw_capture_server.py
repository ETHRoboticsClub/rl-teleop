#!/usr/bin/env python3
"""Label wrist frames as EMPTY jaws or FULL jaws, one keypress each.

    ./.venv/bin/python3 tools/jaw_capture_server.py            # :8802
    open http://localhost:8802/

    E  = jaws closed on NOTHING
    D  = jaws closed on a PACKET
    U  = undo the last capture

WHY THIS EXISTS. Deciding "did we actually grasp it" has exactly one honest
answer on this rig and it is not the one people reach for first:

    closure width   identical held vs empty -- 0.0033 vs 0.0032 median, measured
                    over 59 held and 4 candidate-empty windows in the 2026-08-11
                    right-arm recordings. The packets are thin enough that the
                    jaws shut fully around one.
    gripper effort  also identical: +0.310 vs +0.301. (kitting measured the same
                    thing on the left arm and recorded it as "effort is
                    INVERTED"; a hardware swap moved the empty baseline 0.085 ->
                    0.383, so it is not even stable.)
    the lift        works ONLY while the policy replans -- it is really "the
                    policy re-observed and moved on". At n_action_steps=100 the
                    chunk lifts unconditionally and the signal dies.

That leaves vision, and vision needs a negative class that does not exist.
kitting's REPORT-EMPTY-GRIPPER.md: "the corpus does not contain a single frame
of a closed, empty gripper being carried", and its witness therefore scores
1.0000 on empty jaws at home. This tool collects that missing class.

TWO WAYS THIS TOOL CAN SILENTLY RUIN THE DATASET, and what stops each:

  a dead camera   camera_right died twice on 2026-08-12 (usb 9-1.1, error -71).
                  A capture that saved the last frame anyway would write stale or
                  duplicate images into the class we have none of, and they would
                  look fine in the gallery. So a capture REFUSES if the newest
                  frame is older than MAX_STALE_S, and says which.

  the label       kitting rendered its eight `empty`-labelled attempts and found
                  a bag clamped in the jaws in ALL EIGHT. Operator labels are not
                  ground truth. So every capture keeps its raw frame and is
                  relabellable and deletable from the gallery afterwards -- the
                  verification pass is part of the tool, not a later script.

Joint state is stored with every frame even though nothing here uses it. It is
what makes the pose confound MEASURABLE later: on kitting's corpus a model on the
six joint angles alone scored AUC 0.9692-0.9841 against the vision witness's
0.9829, i.e. the headline number was reading arm pose, not jaws. Without these
six numbers on disk you cannot tell whether a future model has the same problem.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from robots_realtime.runtime.transport.subscriber import Subscriber  # noqa: E402
from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT  # noqa: E402

# 8802, NOT 8796. cockpit/INTEGRATION.md reserves 8765, 8787, 8791, 8796, 8797
# and 8799 for the live system -- 8796 is sort_server, the autonomous loop's own
# write path that Buehler-Kitting-Autonomous.html polls. Binding it here would
# make this tool and the sorter mutually exclusive, and the second one to start
# would die with EADDRINUSE mid-session. 8801 is the review cockpit.
PORT = int(os.environ.get("JAW_PORT", "8802"))
CAM_TOPIC = os.environ.get("JAW_CAM", "camera_right/rgb")
STATE_TOPIC = os.environ.get("JAW_STATE", "yam_right/joint_state")
ROOT = Path(os.environ.get("JAW_ROOT", HERE / "jaw_dataset"))
# 1.0 s, not the 0.25 s act_runner uses. That guard protects a 30 Hz control
# loop; this one only has to catch a camera that has actually stopped, and a
# human pressing a key does not need sub-second freshness. Too tight here would
# reject good captures during a momentary hiccup and train the operator to
# ignore the error.
MAX_STALE_S = 1.0
LABELS = {"empty": "E", "full": "D"}

# THE BAND. Measured over the first 50 captures: every EMPTY reading fell in
# [0.011959, 0.012030] and 23 of 24 FULL readings sat strictly above it. So the
# width alone decides the question OUTSIDE this band, and says nothing INSIDE it.
#
# That makes in-band FULL captures the only ones that teach anything new: there
# were 5 of them against 26 in-band empties, and that imbalance is the single
# reason the detector cannot reach 95% without inventing false alarms. The UI
# shows the live width and flags the band so the operator can aim for it instead
# of discovering afterwards that 45 of 50 captures were already-solved cases.
# FULL PRECISION, not the rounded values. gripper_pos comes off the encoder as
# 0.01203018223234... and a BAND_HI of "0.012030" is strictly BELOW that, so every
# top-of-band capture classified as "above the band -> already decided FULL". The
# backfill made it obvious: 0 in-band captures where 5 were known to exist. Round
# these and the UI confidently points the operator away from the only samples
# worth collecting.
BAND_LO = float(os.environ.get("JAW_BAND_LO", "0.01195899772209497"))
BAND_HI = float(os.environ.get("JAW_BAND_HI", "0.012030182232346202"))


# Above this the jaws are simply OPEN, and the question does not apply: there is
# no "held vs empty" to answer when nothing is clamped. Closed readings cluster at
# ~0.012, open sits at ~0.96, so anything past 0.05 is unambiguous. Without this
# an open gripper reads as "above the band" -> "already decided FULL", which is
# both wrong and the easiest way to fill the FULL class with pictures of nothing.
OPEN_ABOVE = float(os.environ.get("JAW_OPEN_ABOVE", "0.05"))


def band_of(g):
    if g is None:
        return "unknown"
    if g > OPEN_ABOVE:
        return "open"
    if g <= BAND_LO:
        return "below"      # at/under the hard stop -> already decided EMPTY
    if g > BAND_HI:
        return "above"      # jaws held open -> already decided FULL
    return "in"             # the only place the image is needed

ROOT.mkdir(parents=True, exist_ok=True)
for lb in LABELS:
    (ROOT / lb).mkdir(exist_ok=True)
MANIFEST = ROOT / "manifest.jsonl"


class Bus:
    """Newest wrist frame + joint state, with the age of each."""

    def __init__(self):
        self._cam = Subscriber([CAM_TOPIC], host="127.0.0.1", port=DEFAULT_SUB_PORT)
        self._st = Subscriber([STATE_TOPIC], host="127.0.0.1", port=DEFAULT_SUB_PORT)
        self._seen_ts = None
        self._seen_at = None
        self._lock = threading.Lock()

    def frame(self):
        """-> (rgb ndarray | None, age_s | None).

        Age is measured LOCALLY, from when a frame whose payload timestamp
        CHANGED last arrived -- not from the payload timestamp itself. The
        RealSense publishes a device clock and the USB cameras a host clock;
        subtracting one from the other is the bug act_runner.frame() documents,
        which reported an age of MINUS 173.9 ms and aborted four good cycles.
        """
        env = self._cam.get_latest(CAM_TOPIC)
        if env is None:
            return None, None
        ts = float(env.get("ts") or 0.0)
        now = time.monotonic()
        with self._lock:
            if self._seen_ts is None or ts != self._seen_ts:
                self._seen_ts, self._seen_at = ts, now
            age = now - self._seen_at
        img = ((env.get("data") or {}).get("images") or {}).get("rgb")
        return img, age

    def state(self):
        env = self._st.get_latest(STATE_TOPIC)
        if env is None:
            return None
        d = env.get("data") or {}
        try:
            q = np.asarray(d.get("joint_pos"), dtype=float).tolist()
            g = float(np.ravel(d.get("gripper_pos"))[0])
            ge = float(np.ravel(d.get("gripper_eff"))[0]) if d.get("gripper_eff") is not None else None
        except Exception:
            return None
        return {"joint_pos": q, "gripper_pos": g, "gripper_eff": ge}


BUS = Bus()


def encode_jpeg(rgb, quality=88):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def read_manifest():
    if not MANIFEST.exists():
        return []
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def rewrite_manifest(rows):
    MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in rows))


def counts(rows):
    c = {lb: 0 for lb in LABELS}
    c["full_in_band"] = 0        # the scarce class; the target is ~20
    for r in rows:
        if r.get("deleted"):
            continue
        c[r["label"]] = c.get(r["label"], 0) + 1
        if r["label"] == "full" and r.get("band") == "in":
            c["full_in_band"] += 1
    return c


def do_capture(label):
    """-> (ok, payload). Refuses on a stale or absent camera; see module docstring."""
    if label not in LABELS:
        return False, {"error": f"unknown label {label!r}"}
    img, age = BUS.frame()
    if img is None:
        return False, {"error": f"NO FRAME on {CAM_TOPIC} — is the camera session up?"}
    _st = BUS.state()
    if _st is not None and band_of(_st["gripper_pos"]) == "open":
        return False, {"error": f"JAWS ARE OPEN (width {_st['gripper_pos']:.3f}) — "
                                f"close them on the packet, or on nothing, then capture."}
    if age is not None and age > MAX_STALE_S:
        return False, {"error": f"camera STALE ({age:.1f}s old, max {MAX_STALE_S:.1f}s) — "
                                f"nothing captured. Check /cam or the USB link."}
    jpg = encode_jpeg(img)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S_%f")[:-3]
    name = f"{label}_{stamp}.jpg"
    (ROOT / label / name).write_bytes(jpg)
    row = {
        "id": stamp,
        "label": label,
        "file": f"{label}/{name}",
        "t": now.isoformat(),
        "frame_age_s": round(age, 3) if age is not None else None,
        "shape": list(np.asarray(img).shape),
        "state": BUS.state(),
    }
    _st = row["state"]
    row["band"] = band_of(_st["gripper_pos"] if _st else None)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(row) + "\n")
    rows = read_manifest()
    return True, {"row": row, "counts": counts(rows)}


PAGE = """<!doctype html><meta charset=utf-8><title>Jaw Capture</title>
<style>
:root{--bg:#101014;--fg:#eceaea;--mut:#8e8e94;--line:#2a2a31;--card:#181820;
--empty:#e8834f;--full:#4fbf7c;--bad:#e0555a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:1.25rem 1rem 4rem}
h1{font-size:1.2rem;margin:0 0 .15rem;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:.85rem;margin:0 0 1.1rem}
.top{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:1rem;align-items:start}
@media(max-width:800px){.top{grid-template-columns:1fr}}
.live{background:#000;border:1px solid var(--line);border-radius:10px;overflow:hidden;position:relative}
.live img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain}
#age{position:absolute;top:.5rem;right:.6rem;background:#000a;padding:.15rem .5rem;
border-radius:5px;font:600 .72rem ui-monospace,monospace}
.keys{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:0 0 .8rem}
button{font:600 .95rem ui-sans-serif,system-ui,sans-serif;padding:1rem .6rem;border-radius:9px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;text-align:center}
button:hover{border-color:#4a4a55}
button kbd{display:block;font-size:1.5rem;margin-bottom:.2rem}
#bE kbd{color:var(--empty)} #bD kbd{color:var(--full)}
.cnt{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:0 0 .8rem}
.c{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:.7rem .8rem}
.c b{display:block;font-size:1.8rem;line-height:1.1;letter-spacing:-.02em}
.c.e b{color:var(--empty)} .c.f b{color:var(--full)}
.c span{color:var(--mut);font-size:.75rem}
#band{font:600 .82rem ui-monospace,monospace;padding:.55rem .7rem;border-radius:8px;
border:1px solid var(--line);background:var(--card);margin:0 0 .6rem;text-align:center}
#band.in{border-color:#c8a02e;color:#ffd970;background:#2a2412}
#band.open{border-color:#e0555a;color:#ffb3b6;background:#2a1416}
#band.below{border-color:#3a3a44;color:var(--mut)}
#band.above{border-color:#2f6f47;color:#a7e3bf}
.c.t{margin:0 0 .8rem}
.c.t b{color:#ffd970}
.c.t small{font-size:.9rem;opacity:.6;font-weight:400}
#msg{min-height:2.6rem;font-size:.85rem;padding:.5rem .7rem;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--mut)}
#msg.bad{border-color:var(--bad);color:#ffb3b6}
#msg.ok{border-color:#2f6f47;color:#a7e3bf}
h2{font-size:.95rem;margin:1.8rem 0 .7rem;color:var(--mut);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.7rem}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:9px;overflow:hidden}
figure.empty{border-color:var(--empty)} figure.full{border-color:var(--full)}
figure img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
figcaption{padding:.4rem .5rem;font:.68rem/1.4 ui-monospace,monospace;color:var(--mut);
display:flex;gap:.3rem;align-items:center;justify-content:space-between}
figcaption button{padding:.15rem .4rem;font-size:.66rem;border-radius:5px}
</style>
<div class=wrap>
<h1>Jaw capture &mdash; empty vs full</h1>
<p class=sub>Press <b>E</b> when the jaws are closed on <b>nothing</b>, <b>D</b> when they are
closed on a <b>packet</b>. <b>U</b> undoes the last one. Every capture stores the wrist frame,
the six joint angles and the gripper state. Verify them in the gallery before training anything.</p>
<div class=top>
  <div class=live><img id=live src="/frame.jpg"><div id=age>&mdash;</div></div>
  <div>
    <div class=cnt>
      <div class="c e"><b id=nE>0</b><span>EMPTY (E)</span></div>
      <div class="c f"><b id=nD>0</b><span>FULL (D)</span></div>
    </div>
    <div id=band>&mdash;</div>
    <div class="c t"><b id=nB>0<small> / 20</small></b><span>FULL captures IN THE BAND &mdash; the ones that teach something</span></div>
    <div class=keys>
      <button id=bE onclick="cap('empty')"><kbd>E</kbd>empty jaws</button>
      <button id=bD onclick="cap('full')"><kbd>D</kbd>packet held</button>
    </div>
    <div id=msg>Ready.</div>
  </div>
</div>
<h2 id=gh>Captured</h2>
<div class=grid id=gal></div>
</div>
<script>
const $=s=>document.querySelector(s);
let busy=false;
function tick(){ $('#live').src='/frame.jpg?'+Date.now(); }
setInterval(tick,200);
async function age(){
  try{const r=await fetch('/health');const j=await r.json();
    $('#age').textContent = j.age===null?'no frame':(j.age.toFixed(2)+'s');
    $('#age').style.color = (j.age===null||j.age>1.0)?'#e0555a':'#a7e3bf';
    const b=$('#band'); const g=j.gripper_pos;
    if(g===null||g===undefined){ b.textContent='no gripper reading'; b.className=''; return; }
    const w=g.toFixed(6);
    if(j.band==='open')      b.textContent='width '+w+'  \u2022  JAWS OPEN \u2014 close them first, capture is blocked';
    else if(j.band==='in')   b.textContent='width '+w+'  \u2022  IN THE BAND \u2014 a FULL here is worth 10 elsewhere';
    else if(j.band==='below') b.textContent='width '+w+'  \u2022  at/below the hard stop \u2014 already decided EMPTY';
    else                      b.textContent='width '+w+'  \u2022  above the band \u2014 already decided FULL';
    b.className=j.band;
  }catch(e){}
}
setInterval(age,500);
function say(t,cls){const m=$('#msg');m.textContent=t;m.className=cls||'';}
async function cap(label){
  if(busy)return; busy=true;
  try{
    const r=await fetch('/capture?label='+label,{method:'POST'});
    const j=await r.json();
    if(!r.ok){ say(j.error||'capture failed','bad'); }
    else{ $('#nE').textContent=j.counts.empty; $('#nD').textContent=j.counts.full;
  if(j.counts.full_in_band!==undefined) $('#nB').innerHTML=j.counts.full_in_band+'<small> / 20</small>';
          say('saved '+j.row.file+'  (frame '+j.row.frame_age_s+'s old)','ok'); load(); }
  }catch(e){ say('capture failed: '+e,'bad'); }
  busy=false;
}
async function undo(){
  const r=await fetch('/undo',{method:'POST'}); const j=await r.json();
  if(!r.ok){say(j.error||'nothing to undo','bad');return;}
  $('#nE').textContent=j.counts.empty; $('#nD').textContent=j.counts.full;
  if(j.counts.full_in_band!==undefined) $('#nB').innerHTML=j.counts.full_in_band+'<small> / 20</small>';
  say('removed '+j.removed,'ok'); load();
}
async function relabel(id,to){ await fetch('/relabel?id='+id+'&label='+to,{method:'POST'}); load(); }
async function drop(id){ await fetch('/delete?id='+id,{method:'POST'}); load(); }
async function load(){
  const r=await fetch('/manifest'); const j=await r.json();
  $('#nE').textContent=j.counts.empty; $('#nD').textContent=j.counts.full;
  if(j.counts.full_in_band!==undefined) $('#nB').innerHTML=j.counts.full_in_band+'<small> / 20</small>';
  $('#gh').textContent='Captured — '+j.rows.length+' frames (newest first)';
  $('#gal').innerHTML=j.rows.slice().reverse().map(x=>`
   <figure class="${x.label}">
     <img loading=lazy src="/img/${x.file}">
     <figcaption><span>${x.label==='empty'?'EMPTY':'FULL'}</span>
       <span><button onclick="relabel('${x.id}','${x.label==='empty'?'full':'empty'}')">swap</button>
       <button onclick="drop('${x.id}')">del</button></span>
     </figcaption></figure>`).join('');
}
addEventListener('keydown',e=>{
  if(e.repeat||e.metaKey||e.ctrlKey||e.altKey)return;
  const k=e.key.toLowerCase();
  if(k==='e'){e.preventDefault();cap('empty');}
  else if(k==='d'){e.preventDefault();cap('full');}
  else if(k==='u'){e.preventDefault();undo();}
});
load(); age();
</script>"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/frame.jpg":
            img, age = BUS.frame()
            if img is None:
                return self._send(503, b"", "image/jpeg")
            return self._send(200, encode_jpeg(img, 80), "image/jpeg")
        if u.path == "/health":
            _, age = BUS.frame()
            st = BUS.state()
            g = st["gripper_pos"] if st else None
            return self._send(200, {"age": None if age is None else round(age, 3),
                                    "max_stale_s": MAX_STALE_S,
                                    "gripper_pos": g, "band": band_of(g),
                                    "lo": BAND_LO, "hi": BAND_HI})
        if u.path == "/manifest":
            rows = [r for r in read_manifest() if not r.get("deleted")]
            return self._send(200, {"rows": rows, "counts": counts(rows)})
        if u.path.startswith("/img/"):
            p = (ROOT / u.path[len("/img/"):]).resolve()
            if ROOT.resolve() not in p.parents or not p.exists():
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), "image/jpeg")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/capture":
            ok, payload = do_capture((q.get("label") or [""])[0])
            return self._send(200 if ok else 409, payload)
        rows = read_manifest()
        live = [r for r in rows if not r.get("deleted")]
        if u.path == "/undo":
            if not live:
                return self._send(409, {"error": "nothing to undo"})
            last = live[-1]
            for r in rows:
                if r["id"] == last["id"]:
                    r["deleted"] = True
            rewrite_manifest(rows)
            return self._send(200, {"removed": last["file"],
                                    "counts": counts(read_manifest())})
        if u.path in ("/relabel", "/delete"):
            rid = (q.get("id") or [""])[0]
            hit = False
            for r in rows:
                if r["id"] == rid:
                    hit = True
                    if u.path == "/delete":
                        r["deleted"] = True
                    else:
                        new = (q.get("label") or [""])[0]
                        if new not in LABELS:
                            return self._send(400, {"error": "bad label"})
                        src = ROOT / r["file"]
                        dst = ROOT / new / Path(r["file"]).name
                        # Relabelling MOVES the file so the directory layout never
                        # disagrees with the manifest. A trainer that globs the
                        # class directories and a trainer that reads the manifest
                        # must not be able to see different datasets.
                        if src.exists():
                            dst.parent.mkdir(exist_ok=True)
                            src.rename(dst)
                        r["label"] = new
                        r["file"] = f"{new}/{dst.name}"
            if not hit:
                return self._send(404, {"error": "no such id"})
            rewrite_manifest(rows)
            return self._send(200, {"counts": counts(read_manifest())})
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    time.sleep(1.0)  # let the subscribers connect before the first paint
    rows = read_manifest()
    print(f"jaw capture  ->  http://localhost:{PORT}/")
    print(f"  camera : {CAM_TOPIC}   state: {STATE_TOPIC}")
    print(f"  dataset: {ROOT}   (existing: {counts([r for r in rows if not r.get('deleted')])})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
