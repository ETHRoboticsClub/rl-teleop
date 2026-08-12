#!/usr/bin/env python3
"""Continuous camera_right forensic monitor -> /tmp/cam_monitor.jsonl

    tmux new-session -d -s cammon './.venv/bin/python3 -u tools/cam_monitor.py'

Logs every ~2 s: the node's own health (reopens, fps, failures) AND a tearing
signature computed from the actual pixels, because the health topic cannot see
tearing — a torn frame arrives on time, decodes, and counts as delivered.

TEARING SIGNATURE. A torn frame is a composite: some horizontal band comes from
the new exposure, the rest repeats the previous frame. So split each frame into
12 horizontal strips and diff each strip against the SAME strip of the previous
frame. A clean pair changes roughly uniformly (all strips move together, ratio
near 1); a torn pair splits — some strips near zero (repeated) while others jump
(new) — driving max/median ratio high. `tear_ratio` > ~8 with a nonzero median
is the smoking gun. The scene being static makes all strips ~0, which is why the
ratio uses the median as its floor and marks those samples `static`.

This is the loop's measurement instrument: the left camera was cut from the hub
at the same time this started, so reopens/hour and tear incidence before vs
after IS the verdict on the bandwidth hypothesis.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from robots_realtime.runtime.transport.subscriber import Subscriber  # noqa: E402
from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT  # noqa: E402

OUT = Path("/tmp/cam_monitor.jsonl")
STRIPS = 12

sub = Subscriber(["camera_right/rgb", "camera_right/health"],
                 host="127.0.0.1", port=DEFAULT_SUB_PORT)
time.sleep(2.0)

prev = None
prev_ts = None
n = 0
print(f"monitoring camera_right -> {OUT}")
while True:
    rec = {"t": time.time()}
    h = sub.get_latest("camera_right/health")
    if h:
        d = h.get("data") or {}
        rec.update({k: d.get(k) for k in
                    ("reopens", "open_failures", "frames", "fps",
                     "consecutive_failures", "state", "opened_at",
                     "last_frame_age_s")})
    e = sub.get_latest("camera_right/rgb")
    if e:
        ts = float(e.get("ts") or 0.0)
        img = ((e.get("data") or {}).get("images") or {}).get("rgb")
        if img is not None and ts != prev_ts:
            g = np.asarray(img).mean(axis=2).astype(np.float32)
            if prev is not None and prev.shape == g.shape:
                hgt = g.shape[0] // STRIPS
                diffs = [float(np.abs(g[i*hgt:(i+1)*hgt] - prev[i*hgt:(i+1)*hgt]).mean())
                         for i in range(STRIPS)]
                med = float(np.median(diffs))
                mx = float(np.max(diffs))
                rec["strip_med"] = round(med, 3)
                rec["strip_max"] = round(mx, 3)
                if med < 0.5:
                    rec["motion"] = "static"
                else:
                    rec["tear_ratio"] = round(mx / med, 2)
                    if mx / med > 8:
                        rec["TEAR"] = True
            prev, prev_ts = g, ts
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    n += 1
    if n % 150 == 0:
        print(f"  {n} samples, reopens={rec.get('reopens')}")
    time.sleep(2.0)
