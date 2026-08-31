#!/usr/bin/env python3
"""Vary the ELE01 wrist camera's exposure while the operator records captures.

Purpose (2026-08-13): the operator is hand-collecting jaw dataset episodes for
the new camera; cycling real sensor exposure during collection bakes genuine
lighting diversity into the dataset — stronger than synthetic augmentation,
and it directly covers the sun-through-the-window problem.

Every ~8 s: 30% chance -> auto exposure (aperture priority, what deployment
uses), otherwise manual exposure log-uniform in [16, 1300] (x100 us) with gain
jitter. Every change is appended to /tmp/exposure_wobble.jsonl with a
timestamp, so capture frames (manifest timestamps) can be joined to their
exposure setting afterwards.

Restores auto exposure + default gain on exit (Ctrl-C safe).
"""

import json
import random
import signal
import subprocess
import sys
import time

DEV = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:1:1.0-video-index0"
LOG = "/tmp/exposure_wobble.jsonl"
PERIOD_S = 8.0
EXP_LO, EXP_HI = 16, 1300      # x100us: 1.6ms .. 130ms
GAIN_DEFAULT = 17


def ctl(**kv):
    args = ",".join(f"{k}={v}" for k, v in kv.items())
    r = subprocess.run(["v4l2-ctl", "-d", DEV, f"--set-ctrl={args}"],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def log(row):
    row["ts"] = time.time()
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(time.strftime("%H:%M:%S"), row, flush=True)


def restore(*_):
    ok, err = ctl(auto_exposure=3, gain=GAIN_DEFAULT)
    log({"mode": "restored_auto", "ok": ok, "err": err})
    sys.exit(0)


signal.signal(signal.SIGINT, restore)
signal.signal(signal.SIGTERM, restore)

log({"mode": "start", "dev": DEV, "period_s": PERIOD_S})
while True:
    if random.random() < 0.30:
        ok, err = ctl(auto_exposure=3, gain=GAIN_DEFAULT)
        log({"mode": "auto", "ok": ok, "err": err})
    else:
        # log-uniform exposure; occasional gain kick for dim settings
        e = int(round(EXP_LO * (EXP_HI / EXP_LO) ** random.random()))
        g = random.choice([GAIN_DEFAULT] * 3 + [64, 128])
        ok1, e1 = ctl(auto_exposure=1)
        ok2, e2 = ctl(exposure_time_absolute=e, gain=g)
        log({"mode": "manual", "exposure": e, "gain": g,
             "ok": ok1 and ok2, "err": (e1 + " " + e2).strip()})
    time.sleep(PERIOD_S)
