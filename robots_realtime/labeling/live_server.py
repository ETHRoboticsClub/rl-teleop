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


def encode_frame_jpeg(frame, quality: int = 80) -> bytes | None:
    """RGB (H,W,3) uint8 → JPEG bytes (BGR-corrected for the browser)."""
    import cv2
    import numpy as np
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class CameraBridge:
    """Serve the running session's camera topics as single JPEGs at /cam/<id>.

    During teleop the physical cameras are held by rr-session, so we can't open
    /dev/video directly — we subscribe to the ``<name>/rgb`` topics on the ZMQ bus
    ({"frame": (H,W,3) uint8}) and JPEG-encode the latest frame on request. The
    cockpit's <img src=".../cam/<id>?t=..."> cache-buster re-fetches it live.
    """

    def __init__(self, id_to_topic: dict[str, str], host: str = "127.0.0.1",
                 port: int | None = None):
        from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
        from robots_realtime.runtime.transport.subscriber import Subscriber
        self._id_to_topic = dict(id_to_topic)
        topics = sorted(set(id_to_topic.values()))
        self._sub = Subscriber(topics, host=host, port=port or DEFAULT_SUB_PORT)

    def jpeg(self, cam_id: str) -> bytes | None:
        # exact id → topic, else the 'default' topic so no panel is blank
        topic = self._id_to_topic.get(cam_id) or self._id_to_topic.get("default")
        if topic is None:
            return None
        env = self._sub.get_latest(topic)
        if not env:
            return None
        data = env.get("data") or {}
        # CameraNode publishes {"images": {"rgb": (H,W,3) uint8}, ...}; older
        # payloads used a bare {"frame": ...}.
        frame = data.get("frame")
        if frame is None:
            imgs = data.get("images")
            if isinstance(imgs, dict) and imgs:
                frame = imgs.get("rgb")
                if frame is None:
                    frame = next(iter(imgs.values()))
        return encode_frame_jpeg(frame) if frame is not None else None


def _make_handler(labeler: LiveLabeler, cam_base: str | None,
                  bridge: "CameraBridge | None" = None):
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
            # 1) live bus bridge (single JPEG from the session's camera topics)
            if bridge is not None:
                jpg = bridge.jpeg(cam_id)
                if jpg is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self._cors()
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
            # 2) optional MJPEG proxy fallback
            if not cam_base:
                self.send_error(503); return
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
    def __init__(self, labeler: LiveLabeler, port: int = 8791, cam_base: str | None = None,
                 bridge: "CameraBridge | None" = None, host: str = "127.0.0.1"):
        self.labeler = labeler
        self._httpd = ThreadingHTTPServer((host, port),
                                          _make_handler(labeler, cam_base, bridge))
        self.port = port

    def serve_forever(self):
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def shutdown(self):
        self._httpd.shutdown()


def _feed_from_source(labeler: LiveLabeler, next_sample, stop=None,
                      poll_s: float = 0.005) -> None:
    """Pump (ts, gripper_width) from ``next_sample()`` into the labeler until
    stop is set. ``next_sample`` returns (ts, width) or None when nothing new.
    Deduplicates on ts so the latest-per-topic subscriber isn't re-pushed."""
    last_ts = None
    while stop is None or not stop.is_set():
        s = next_sample()
        if s is not None:
            ts, width = s
            if ts != last_ts:
                last_ts = ts
                labeler.push(float(ts), float(width))
        time.sleep(poll_s)


def bus_feed(labeler: LiveLabeler, arm: str = "left", host: str = "127.0.0.1",
             port: int | None = None, stop=None) -> None:
    """Subscribe to the live rl-teleop joint stream and drive the labeler.

    The follower publishes ``yam_{arm}/joint_state`` = {"joint_pos": [...7...]}
    (gripper at index 6) on the ZMQ XPUB/XSUB broker. This taps it live so the
    cockpit shows each grasp/place labeled as it happens.
    """
    from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
    from robots_realtime.runtime.transport.subscriber import Subscriber

    topic = f"yam_{arm}/joint_state"
    sub = Subscriber([topic], host=host, port=port or DEFAULT_SUB_PORT)

    def next_sample():
        env = sub.get_latest(topic)
        if env is None:
            return None
        ts = env.get("ts")
        data = env.get("data") or {}
        # Follower publishes joint_pos (6 arm) + gripper_pos (1) as SEPARATE fields;
        # the gripper is NOT joint_pos[6].
        grip = data.get("gripper_pos")
        if grip is None or ts is None:
            return None
        width = float(grip[0]) if hasattr(grip, "__len__") else float(grip)
        return float(ts), width

    _feed_from_source(labeler, next_sample, stop=stop)


def write_cockpit_events(labeler: LiveLabeler, path: str | Path) -> None:
    """Persist the live place/grasp events as cockpit_events.jsonl so the
    offline labeler can fuse intent (part → compartment) later."""
    lines = [json.dumps(e) for e in labeler.events]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


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
    ap.add_argument("--live", action="store_true",
                    help="tap the running rl-teleop joint bus (real teleop)")
    ap.add_argument("--replay", type=str, default=None,
                    help="episode dir to replay (yam_left.mcap)")
    ap.add_argument("--record-events", type=str, default=None,
                    help="on exit, write cockpit_events.jsonl into this episode dir")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--cam-base", default=None, help="proxy /cam/<id> to this base URL")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 to reach it over Tailscale/LAN)")
    ap.add_argument("--bus-cams", default=None,
                    help="serve /cam/<id> from bus topics, e.g. "
                         "'top=camera_top/rgb,ego=camera_top/rgb,left=camera_left/rgb'")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--open-ref", type=float, default=1.0)
    ap.add_argument("--closed-ref", type=float, default=0.0)
    args = ap.parse_args(argv)

    labeler = LiveLabeler(open_ref=args.open_ref, closed_ref=args.closed_ref)
    labeler.seed(DEFAULT_KIT)

    bridge = None
    if args.bus_cams:
        id_to_topic = dict(p.split("=", 1) for p in args.bus_cams.split(",") if "=" in p)
        bridge = CameraBridge(id_to_topic, host="127.0.0.1")
        print(f"Camera bridge: {id_to_topic}")

    server = LiveLabelServer(labeler, port=args.port, cam_base=args.cam_base,
                             bridge=bridge, host=args.host)
    server.start_background()
    print(f"Live label backend on http://{args.host}:{args.port}  (point the cockpit here)")

    def _finish():
        if args.record_events:
            write_cockpit_events(labeler, Path(args.record_events) / "cockpit_events.jsonl")
            print(f"wrote {args.record_events}/cockpit_events.jsonl ({len(labeler.events)} events)")
        s = labeler.state()
        print(f"Final: {s['done']}/{s['total']} placed")

    try:
        if args.live:
            print("Tapping the live rl-teleop joint bus... teleoperate now.")
            bus_feed(labeler, arm=args.arm)          # blocks until Ctrl+C
        else:
            if args.replay:
                from robots_realtime.labeling.mcap_io import read_positions
                times, positions = read_positions(
                    Path(args.replay) / f"yam_{args.arm}.mcap", f"yam_{args.arm}")
            else:
                times, positions = _demo_episode()
            print(f"Feeding {len(times)} samples (speed x{args.speed})...")
            replay_feed(labeler, times, positions, realtime=True, speed=args.speed)
            _finish()
            while True:                              # keep serving final state
                time.sleep(1)
    except KeyboardInterrupt:
        _finish()
        server.shutdown()


if __name__ == "__main__":
    main()
