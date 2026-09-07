#!/usr/bin/env python3
"""Watch one camera on the bus and log EVERY frame drop with the arm pose at that instant.

Written 2026-09-07 for the right wrist camera, whose cable/port history is bad, so
an operator can decide from evidence — not impressions — whether to keep working
with it or wait for the replacement cameras.

Bus-only: it subscribes to `<cam>/rgb`, `<cam>/health` and `<arm>/joint_state`
on the running session's publisher port. It NEVER opens the device (the session
owns it; a second opener corrupts the stream — bimanual-setup skill).

What counts as a drop, and why each one is here:

  DROP    the publisher's own frame timestamps jump by more than `--gap-x` nominal
          intervals (2.5 × 33 ms at 30 Hz). Judged on publisher timestamps, not
          arrival time, so a busy subscriber cannot fake a drop.
  FROZEN  N consecutive frames with byte-identical content. The stale-handle
          signature: the node keeps publishing at 30 Hz while the device is gone
          (CLAUDE.md "the recorder writes a perfect mp4 while publishing nothing").
  STALL   no frame at all for `--stall-s` seconds; a RESUME line closes it.
  HEALTH  the node's own health topic leaves `ok`, or its reopen / failure
          counters climb (a reopen means the device vanished and came back).

Every event carries: wall time, session-relative time, gap in ms, frames missed,
the arm's joint_pos (6) + gripper, |joint_vel| (was the arm moving?), and the age
of that joint sample. Events go to a JSONL file (everything) and, rate-limited,
to stdout (so a Monitor can wake an agent without flooding it). A last-good frame
is saved as JPEG next to the log for the first `--max-jpg` events.

Ctrl-C / SIGTERM prints the verdict: drops per minute, worst gap, frames lost,
and whether drops coincide with motion or with a joint region.

Usage (from rl-teleop, its venv):
  .venv/bin/python tools/watch_camera_drops.py --cam camera_right --arm yam_right
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import zmq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from robots_realtime.runtime.transport.serialization import unpack  # noqa: E402
from robots_realtime.runtime.transport.subscriber import DEFAULT_SUB_PORT  # noqa: E402


def fmt_pose(js: dict | None) -> str:
    if not js:
        return "pose=?"
    jp = ", ".join(f"{v:+.2f}" for v in js["joint_pos"])
    return f"j=[{jp}] grip={js['gripper']:.2f} |v|={js['speed']:.2f} age={js['age_s']:.2f}s"


class Watcher:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.nominal = 1.0 / a.fps
        self.gap_thresh = self.nominal * a.gap_x
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.out_dir = Path(a.out_dir) / f"{a.cam}_{stamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(self.out_dir / "events.jsonl", "a", buffering=1)
        self.t0 = time.time()
        self.frames = 0
        self.last_ts: float | None = None
        self.last_arrival: float | None = None
        self.last_digest: str | None = None
        self.same_run = 0
        self.frozen_open = False
        self.stalled = False
        self.last_frame: np.ndarray | None = None
        self.joint: dict | None = None
        self.health_prev: dict | None = None
        self.events: list[dict] = []
        self.jpgs = 0
        self.gaps_ms: list[float] = []
        self.speed_samples: list[float] = []
        self.last_stdout = 0.0
        self.pending: Counter = Counter()
        self.last_summary = self.t0
        self.stop = False

    # ── bus ───────────────────────────────────────────────────────────────────
    def run(self) -> int:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{self.a.sub_port}")
        for t in (f"{self.a.cam}/rgb", f"{self.a.cam}/health", f"{self.a.arm}/joint_state"):
            sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        print(f"watching {self.a.cam}/rgb + {self.a.arm}/joint_state on :{self.a.sub_port} "
              f"— log {self.out_dir}/events.jsonl  (gap>{self.gap_thresh*1000:.0f}ms=DROP, "
              f"{self.a.frozen_n}×identical=FROZEN, silence>{self.a.stall_s}s=STALL)", flush=True)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        try:
            while not self.stop:
                if sock.poll(100):
                    while True:
                        try:
                            parts = sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        if len(parts) >= 2:
                            self.on_msg(parts[0].decode(errors="replace"), parts[1])
                self.tick()
        except KeyboardInterrupt:
            pass
        finally:
            sock.close()
            self.report()
        return 0

    def on_msg(self, topic: str, raw: bytes) -> None:
        try:
            env = unpack(raw)
        except Exception:
            return
        data = env.get("data") or {}
        if topic == f"{self.a.arm}/joint_state":
            jp = np.asarray(data.get("joint_pos", []), dtype=float)
            jv = np.asarray(data.get("joint_vel", []), dtype=float)
            gp = np.asarray(data.get("gripper_pos", [np.nan]), dtype=float)
            self.joint = {"joint_pos": jp.round(3).tolist(), "gripper": float(gp[0]) if gp.size else float("nan"),
                          "speed": float(np.linalg.norm(jv)) if jv.size else float("nan"),
                          "ts": float(env.get("ts", time.time()))}
            self.speed_samples.append(self.joint["speed"])
        elif topic == f"{self.a.cam}/health":
            self.on_health(data)
        elif topic == f"{self.a.cam}/rgb":
            self.on_frame(env, data)

    def pose_now(self) -> dict | None:
        if not self.joint:
            return None
        j = dict(self.joint)
        j["age_s"] = round(time.time() - j.pop("ts"), 3)
        return j

    # ── detectors ─────────────────────────────────────────────────────────────
    def on_frame(self, env: dict, data: dict) -> None:
        now = time.time()
        ts = float(data.get("timestamp") or env.get("ts") or now)
        imgs = data.get("images") or {}
        frame = imgs.get("rgb") if isinstance(imgs, dict) else None
        if frame is None:
            frame = data.get("frame")
        self.frames += 1
        if self.stalled:
            self.stalled = False
            self.event("RESUME", now, gap_ms=(now - self.last_arrival) * 1000 if self.last_arrival else None)
        if self.last_ts is not None:
            gap = ts - self.last_ts
            if gap > self.gap_thresh:
                missed = max(0, int(round(gap / self.nominal)) - 1)
                self.gaps_ms.append(gap * 1000)
                self.event("DROP", now, gap_ms=gap * 1000, frames_missed=missed)
        if frame is not None:
            d = hashlib.md5(memoryview(np.ascontiguousarray(frame)).tobytes()).hexdigest()
            if d == self.last_digest:
                self.same_run += 1
                if self.same_run >= self.a.frozen_n and not self.frozen_open:
                    self.frozen_open = True
                    self.event("FROZEN", now, identical_frames=self.same_run + 1)
            else:
                if self.frozen_open:
                    self.frozen_open = False
                    self.event("UNFROZE", now, identical_frames=self.same_run + 1)
                self.same_run = 0
            self.last_digest = d
            self.last_frame = frame
        self.last_ts = ts
        self.last_arrival = now

    def on_health(self, h: dict) -> None:
        prev = self.health_prev
        self.health_prev = h
        if prev is None:
            return
        changes = {}
        if h.get("state") != prev.get("state"):
            changes["state"] = f"{prev.get('state')}→{h.get('state')} {h.get('reason', '')} {h.get('detail', '')}".strip()
        for k in ("reopens", "open_failures", "consecutive_failures"):
            if (h.get(k) or 0) > (prev.get(k) or 0):
                changes[k] = f"{prev.get(k)}→{h.get(k)}"
        if changes:
            self.event("HEALTH", time.time(), **changes)

    def tick(self) -> None:
        now = time.time()
        if self.last_arrival and not self.stalled and now - self.last_arrival > self.a.stall_s:
            self.stalled = True
            self.event("STALL", now, silent_s=round(now - self.last_arrival, 2))
        if now - self.last_summary >= self.a.summary_s:
            self.last_summary = now
            self.summary_line(now)
        self.flush_pending(now)

    # ── output ────────────────────────────────────────────────────────────────
    def event(self, kind: str, now: float, **extra) -> None:
        ev = {"kind": kind, "t_wall": time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}",
              "t_rel_s": round(now - self.t0, 3), "frame_n": self.frames, "pose": self.pose_now(), **extra}
        self.events.append(ev)
        self.jsonl.write(json.dumps(ev) + "\n")
        if kind in ("DROP", "FROZEN", "STALL", "HEALTH") and self.last_frame is not None and self.jpgs < self.a.max_jpg:
            try:
                import cv2
                cv2.imwrite(str(self.out_dir / f"{ev['t_rel_s']:08.2f}_{kind}.jpg"),
                            cv2.cvtColor(np.ascontiguousarray(self.last_frame), cv2.COLOR_RGB2BGR))
                self.jpgs += 1
            except Exception:
                pass
        self.pending[kind] += 1
        # stdout is rate-limited so a Monitor gets one line per burst, not one per frame
        if kind != "DROP" or now - self.last_stdout >= self.a.stdout_min_s:
            self.flush_pending(now, force=True)

    def flush_pending(self, now: float, force: bool = False) -> None:
        if not self.pending or (not force and now - self.last_stdout < self.a.stdout_min_s):
            return
        ev = self.events[-1]
        extra = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("kind", "t_wall", "t_rel_s", "frame_n", "pose"))
        burst = "" if sum(self.pending.values()) == 1 else f" (+{sum(self.pending.values()) - 1} more in burst: {dict(self.pending)})"
        print(f"{ev['t_wall']} {ev['kind']:6s} {extra}  {fmt_pose(ev['pose'])}{burst}", flush=True)
        self.pending.clear()
        self.last_stdout = now

    def summary_line(self, now: float) -> None:
        el = now - self.t0
        drops = [e for e in self.events if e["kind"] == "DROP"]
        lost = sum(e.get("frames_missed", 0) for e in drops)
        worst = max(self.gaps_ms) if self.gaps_ms else 0.0
        hz = self.frames / el if el else 0.0
        state = (self.health_prev or {}).get("state", "?")
        print(f"[{el/60:5.1f} min] {self.a.cam}: {self.frames} frames {hz:.1f} Hz | drops {len(drops)} "
              f"({len(drops)/max(el/60,1e-9):.1f}/min) lost≈{lost} worst {worst:.0f} ms | frozen "
              f"{sum(e['kind']=='FROZEN' for e in self.events)} stall {sum(e['kind']=='STALL' for e in self.events)} "
              f"health {sum(e['kind']=='HEALTH' for e in self.events)} node={state}", flush=True)

    def report(self) -> None:
        now = time.time()
        el = now - self.t0
        drops = [e for e in self.events if e["kind"] == "DROP"]
        lost = sum(e.get("frames_missed", 0) for e in drops)
        print("\n── verdict ──────────────────────────────────────────────", flush=True)
        self.summary_line(now)
        if drops:
            g = np.array(self.gaps_ms)
            print(f"  gap sizes ms: p50 {np.percentile(g,50):.0f}  p90 {np.percentile(g,90):.0f}  max {g.max():.0f}; "
                  f"frames lost ≈{lost} of {self.frames + lost} ({100*lost/max(self.frames+lost,1):.2f}%)")
            sp = [e["pose"]["speed"] for e in drops if e.get("pose") and np.isfinite(e["pose"]["speed"])]
            if sp and self.speed_samples:
                allv = np.array(self.speed_samples)
                moving_all = float((allv > self.a.moving_thresh).mean())
                moving_drop = float((np.array(sp) > self.a.moving_thresh).mean())
                print(f"  arm moving (|v|>{self.a.moving_thresh}) during {100*moving_drop:.0f}% of drops "
                      f"vs {100*moving_all:.0f}% of the session — "
                      + ("drops track MOTION (cable/connector flex)" if moving_drop > moving_all + 0.25
                         else "no motion correlation (bandwidth/port, not the cable flexing)"))
            jp = np.array([e["pose"]["joint_pos"] for e in drops if e.get("pose")])
            if len(jp) >= 3:
                print("  joint_pos at drops, per joint [min..max]: "
                      + " ".join(f"j{i+1}[{jp[:,i].min():+.2f}..{jp[:,i].max():+.2f}]" for i in range(jp.shape[1])))
        else:
            print("  no drops.")
        print(f"  events: {len(self.events)} → {self.out_dir}/events.jsonl  ({self.jpgs} jpg)", flush=True)
        verdict = {"cam": self.a.cam, "arm": self.a.arm, "elapsed_s": round(el, 1), "frames": self.frames,
                   "drops": len(drops), "drops_per_min": round(len(drops) / max(el / 60, 1e-9), 2),
                   "frames_lost": lost, "worst_gap_ms": round(max(self.gaps_ms), 1) if self.gaps_ms else 0,
                   "frozen": sum(e["kind"] == "FROZEN" for e in self.events),
                   "stalls": sum(e["kind"] == "STALL" for e in self.events),
                   "health": sum(e["kind"] == "HEALTH" for e in self.events)}
        (self.out_dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
        self.jsonl.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", default="camera_right")
    ap.add_argument("--arm", default="yam_right", help="joint_state publisher whose pose is logged")
    ap.add_argument("--fps", type=float, default=30.0, help="nominal camera rate")
    ap.add_argument("--gap-x", type=float, default=2.5, help="DROP when gap > this × nominal interval")
    ap.add_argument("--frozen-n", type=int, default=3, help="FROZEN after this many identical consecutive frames")
    ap.add_argument("--stall-s", type=float, default=1.0)
    ap.add_argument("--summary-s", type=float, default=60.0)
    ap.add_argument("--stdout-min-s", type=float, default=5.0, help="min seconds between DROP lines on stdout")
    ap.add_argument("--moving-thresh", type=float, default=0.15, help="|joint_vel| above which the arm counts as moving")
    ap.add_argument("--max-jpg", type=int, default=200)
    ap.add_argument("--sub-port", type=int, default=DEFAULT_SUB_PORT)
    ap.add_argument("--out-dir", default=str(REPO.parent / "data" / "camera-drops"))
    return Watcher(ap.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
