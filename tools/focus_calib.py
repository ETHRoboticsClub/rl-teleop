#!/usr/bin/env python3
"""Live camera calibration tool — tune the NEW wrist camera to match the OLD one.

Serves a page on :8804 that shows, once per second, the live camera_right frame
next to a reference frame from the ACT training set, with sharpness /
brightness / color metrics and green/red bands telling the operator whether the
current tuning (focus ring, exposure, color) is inside the training
distribution. Made for the 2026-08-13 Innomaker -> ELE01 wrist camera swap.

Targets are computed once at startup from the frozen jaw_ml train manifest —
the exact frames the deployed models were trained on. Frames come straight off
the ZMQ bus (camera_right/rgb, the same 240x320 copy the policy consumes), NOT
through the live_server bridge, so a flapping health state cannot blank the
tool while frames are in fact flowing.

Usage:
    ./.venv/bin/python3 tools/focus_calib.py            # http://<host>:8804
"""

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import zmq

from robots_realtime.runtime.transport.serialization import unpack
from robots_realtime.runtime.transport.subscriber import DEFAULT_SUB_PORT

PORT = 8805  # 8804 is an old static http.server someone left on this box
TOPIC = "camera_right/rgb"
# Reference = yesterday's jaw-capture frames (operator request 2026-08-13):
# the :8802 captures are the canonical "what the check view should look like",
# closer to the deployed crop than the broader ACT training frames.
JAW_DATASET = Path(__file__).resolve().parents[1] / "jaw_dataset"
MANIFEST = Path(__file__).resolve().parents[2] / "yam-pick-pipeline/jaw_ml/manifest_train.json"

state = {
    "frame": None,        # latest RGB frame from the bus
    "frame_ts": 0.0,
    "current": None,      # latest computed metrics
    "history": [],        # last 300 metric rows
    "target": None,
    "ref_jpeg": None,
}
lock = threading.Lock()


def metrics(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "sharpness_center": float(cv2.Laplacian(center, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "r": float(rgb[..., 0].mean()),
        "g": float(rgb[..., 1].mean()),
        "b": float(rgb[..., 2].mean()),
    }


def load_targets() -> None:
    # Prefer yesterday's :8802 jaw captures; fall back to the train manifest.
    paths = [str(p) for p in JAW_DATASET.glob("*/*.jpg")]
    if not paths:
        man = json.load(open(MANIFEST))
        items = man["items"] if isinstance(man, dict) and "items" in man else man
        paths = [e["path"] if isinstance(e, dict) else e for e in items]
    random.seed(0)
    rows, imgs = [], []
    for p in random.sample(paths, min(120, len(paths))):
        fp = p if str(p).startswith("/") else str(MANIFEST.parent / p)
        im = cv2.imread(fp)  # BGR
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rows.append(metrics(rgb))
        imgs.append((rows[-1]["sharpness"], im))
    keys = rows[0].keys()
    tgt = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        tgt[k] = {"med": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                  "p75": float(np.percentile(v, 75))}
    # reference image: the median-sharpness training frame
    imgs.sort(key=lambda t: abs(t[0] - tgt["sharpness"]["med"]))
    ok, buf = cv2.imencode(".jpg", imgs[0][1])
    with lock:
        state["target"] = tgt
        state["ref_jpeg"] = buf.tobytes() if ok else None
    print(f"targets from {len(rows)} training frames:",
          {k: round(v['med'], 1) for k, v in tgt.items()})


def bus_reader() -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{DEFAULT_SUB_PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, TOPIC)
    sock.RCVTIMEO = 2000
    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            continue
        try:
            env = unpack(parts[1])
            frame = env["data"]["images"]["rgb"]
        except Exception:
            continue
        with lock:
            state["frame"] = frame
            state["frame_ts"] = time.time()


def scorer() -> None:
    while True:
        time.sleep(1.0)
        with lock:
            frame = state["frame"]
            ts = state["frame_ts"]
        if frame is None or time.time() - ts > 3.0:
            with lock:
                state["current"] = None
            continue
        m = metrics(frame)
        m["ts"] = time.time()
        with lock:
            state["current"] = m
            state["history"].append(m)
            state["history"] = state["history"][-300:]


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Focus Calib</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 12px} .row{display:flex;gap:16px;flex-wrap:wrap}
.imgbox{text-align:center}.imgbox img{width:480px;image-rendering:pixelated;border:1px solid #333}
.imgbox .cap{color:#888;font-size:13px;margin-top:4px}
table{border-collapse:collapse;margin-top:16px;font-size:15px}
td,th{padding:6px 14px;border-bottom:1px solid #2a2a2a;text-align:right}
th{color:#888;font-weight:normal} td.name{text-align:left;color:#aaa}
.ok{color:#4caf50;font-weight:bold}.bad{color:#ff5252;font-weight:bold}
.big{font-size:22px} #spark{margin-top:10px;background:#181818;border:1px solid #333}
#status{color:#888;font-size:13px;margin-top:8px}
</style></head><body>
<h1>Wrist camera calibration &mdash; live vs. ACT training distribution (1&nbsp;Hz)</h1>
<div class="row">
 <div class="imgbox"><img id="live" src="/live.jpg"><div class="cap">LIVE &mdash; new camera (bus copy, what the policy sees)</div></div>
 <div class="imgbox"><img id="ref" src="/ref.jpg"><div class="cap">REFERENCE &mdash; median training frame (old camera)</div></div>
</div>
<table id="tbl"><tr><th></th><th>live</th><th>target (p25&ndash;p75)</th><th>verdict</th></tr></table>
<canvas id="spark" width="980" height="120"></canvas>
<div id="status"></div>
<script>
const ROWS=[["sharpness","Sharpness (Laplacian var)"],["sharpness_center","Sharpness center crop"],
["brightness","Brightness"],["contrast","Contrast"],["r","Red mean"],["g","Green mean"],["b","Blue mean"]];
async function tick(){
 try{
  const s=await (await fetch('/stats')).json();
  document.getElementById('live').src='/live.jpg?'+Date.now();
  const t=document.getElementById('tbl');
  while(t.rows.length>1)t.deleteRow(1);
  let st=document.getElementById('status');
  if(!s.current){st.textContent='NO FRAME from the bus (camera down or session stopped)';return;}
  st.textContent='last update '+new Date(s.current.ts*1000).toLocaleTimeString();
  for(const [k,label] of ROWS){
   const cur=s.current[k],tg=s.target[k];
   // sharpness: anything above p25 is fine (blur only hurts downward)
   const ok = (k.startsWith('sharpness')) ? cur>=tg.p25 : (cur>=tg.p25-0.15*(tg.p75-tg.p25+1) && cur<=tg.p75+0.15*(tg.p75-tg.p25+1));
   const r=t.insertRow();
   r.insertCell().outerHTML='<td class="name">'+label+'</td>';
   r.insertCell().outerHTML='<td class="big '+(ok?'ok':'bad')+'">'+cur.toFixed(1)+'</td>';
   r.insertCell().textContent=tg.p25.toFixed(0)+' \\u2013 '+tg.p75.toFixed(0)+'  (med '+tg.med.toFixed(0)+')';
   r.insertCell().outerHTML='<td class="'+(ok?'ok':'bad')+'">'+(ok?'\\u2713 in range':'\\u2717 off')+'</td>';
  }
  // sharpness sparkline
  const c=document.getElementById('spark'),x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const h=s.history.map(v=>v.sharpness); if(h.length>1){
   const max=Math.max(...h,s.target.sharpness.p75)*1.1;
   x.strokeStyle='#666';x.beginPath();
   const y25=c.height-(s.target.sharpness.p25/max)*c.height;
   x.moveTo(0,y25);x.lineTo(c.width,y25);x.stroke();
   x.fillStyle='#888';x.fillText('target p25',6,y25-4);
   x.strokeStyle='#4caf50';x.beginPath();
   h.forEach((v,i)=>{const px=i/(h.length-1)*c.width,py=c.height-(v/max)*c.height;i?x.lineTo(px,py):x.moveTo(px,py);});
   x.stroke();x.fillText('sharpness history (5 min)',6,12);
  }
 }catch(e){document.getElementById('status').textContent='fetch error: '+e;}
}
setInterval(tick,1000);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/live.jpg"):
            with lock:
                frame = state["frame"]
            if frame is None:
                self._send(503, "text/plain", b"no frame")
                return
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._send(200, "image/jpeg", buf.tobytes())
        elif self.path.startswith("/ref.jpg"):
            with lock:
                ref = state["ref_jpeg"]
            self._send(200 if ref else 503, "image/jpeg", ref or b"")
        elif self.path.startswith("/stats"):
            with lock:
                body = json.dumps({"current": state["current"], "target": state["target"],
                                   "history": state["history"][-300:]}).encode()
            self._send(200, "application/json", body)
        else:
            self._send(200, "text/html; charset=utf-8", PAGE.encode())


if __name__ == "__main__":
    load_targets()
    threading.Thread(target=bus_reader, daemon=True).start()
    threading.Thread(target=scorer, daemon=True).start()
    print(f"focus calib on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
