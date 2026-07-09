"""Live label backend for the kitting cockpit.

Serves the exact contract the cockpit polls:
    GET  /state            -> {seeded, packets:[{part,name,comp,bbox,status}], ti, ...}
    POST /seed             -> seed the kit, returns state
    GET  /cam/<id>         -> MJPEG proxied from --cam-base, else 404
    GET  /events           -> the live cockpit_events (debug / offline consistency)

Point the cockpit's "live URL" at http://localhost:<port>. As the operator
teleoperates, a feed pushes gripper samples into the LiveLabeler and the cockpit
shows each grasp/place labeled live (ti advances, packets flip to 'placed').

Feeds:
    replay_feed(...)   replay a recorded/synthetic episode (verifiable demo).
    bus_feed(...)      subscribe to the live rl-teleop joint stream (real wire-up).

Run a demo:
    uv run python -m robots_realtime.labeling.live_server --replay <episode_dir>
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.live import LiveLabeler

DEFAULT_KIT = [
    {"bag_id": 1, "part": "UNN-10126-151", "name": "Flügelmutter M8", "comp": 5, "bbox": [0.30, 0.55, 0.10, 0.10]},
    {"bag_id": 2, "part": "UNN-10015-007", "name": "Sechskantschraube", "comp": 3, "bbox": [0.55, 0.55, 0.10, 0.10]},
    {"bag_id": 3, "part": "DNN-15122-009", "name": "Sechskant", "comp": 1, "bbox": [0.20, 0.30, 0.10, 0.10]},
]


def _make_handler(labeler: LiveLabeler, cam_base: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.end_headers()

        def do_GET(self):
            if self.path == "/state":
                self._json(labeler.state())
            elif self.path == "/events":
                self._json(labeler.events)
            elif self.path.startswith("/cam/"):
                self._proxy_cam(self.path.split("/cam/", 1)[1])
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/seed":
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    kit = json.loads(raw)["packets"] if raw else DEFAULT_KIT
                except Exception:
                    kit = DEFAULT_KIT
                labeler.seed(kit)
                self._json(labeler.state())
            else:
                self.send_error(404)

        def _proxy_cam(self, cam_id: str):
            if not cam_base:
                self.send_error(404); return
            try:
                up = urllib.request.urlopen(f"{cam_base.rstrip('/')}/cam/{cam_id}", timeout=3)
            except Exception:
                self.send_error(502); return
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace"))
            self._cors(); self.end_headers()
            try:
                while True:
                    chunk = up.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except Exception:
                pass

    return Handler


class LiveLabelServer:
    def __init__(self, labeler: LiveLabeler, port: int = 8791, cam_base: str | None = None):
        self.labeler = labeler
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                          _make_handler(labeler, cam_base))
        self.port = port

    def serve_forever(self):
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def shutdown(self):
        self._httpd.shutdown()


def replay_feed(labeler: LiveLabeler, times, positions, realtime: bool = True,
                speed: float = 1.0) -> None:
    """Push a joint timeline into the labeler, optionally pacing in real time."""
    times = np.asarray(times, float)
    gripper = np.asarray(positions, float)[:, C.GRIPPER_JOINT_INDEX]
    t0 = times[0] if times.size else 0.0
    wall0 = time.monotonic()
    for i in range(times.size):
        if realtime:
            target = (times[i] - t0) / max(speed, 1e-6)
            dt = target - (time.monotonic() - wall0)
            if dt > 0:
                time.sleep(dt)
        labeler.push(float(times[i]), float(gripper[i]))


def _demo_episode():
    """Synthesize a 3-bag joint timeline (gripper only matters for live)."""
    dt = 0.02
    OPEN, HELD = 1.0, 0.35
    plan = []
    for _ in range(3):
        plan += [(1.0, OPEN), (0.5, HELD), (1.0, HELD), (0.6, OPEN)]
    times, pos, t = [], [], 0.0
    for dur, grip in plan:
        for _ in range(int(dur / dt)):
            pos.append([0, 0, 0, 0, 0, 0, grip]); times.append(t); t += dt
    return np.array(times), np.array(pos)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--replay", type=str, default=None,
                    help="episode dir to replay (yam_left.mcap); omit for a synthetic demo")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--cam-base", default=None, help="proxy /cam/<id> to this base URL")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--open-ref", type=float, default=1.0)
    ap.add_argument("--closed-ref", type=float, default=0.0)
    args = ap.parse_args(argv)

    labeler = LiveLabeler(open_ref=args.open_ref, closed_ref=args.closed_ref)
    labeler.seed(DEFAULT_KIT)
    server = LiveLabelServer(labeler, port=args.port, cam_base=args.cam_base)
    server.start_background()
    print(f"Live label backend on http://localhost:{args.port}  (point the cockpit here)")

    if args.replay:
        from robots_realtime.labeling.mcap_io import read_positions
        times, positions = read_positions(Path(args.replay) / f"yam_{args.arm}.mcap", f"yam_{args.arm}")
    else:
        times, positions = _demo_episode()

    print(f"Feeding {len(times)} samples (speed x{args.speed})... watch ti advance in the cockpit.")
    replay_feed(labeler, times, positions, realtime=True, speed=args.speed)
    print("Feed done. Final:", labeler.state()["done"], "/", labeler.state()["total"])
    # keep serving so the cockpit can still read the final state
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
