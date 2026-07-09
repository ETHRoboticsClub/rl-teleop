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
    # coerce to a clean uint8 C-contiguous buffer so cv2 never trips on an odd
    # stride/dtype (the frame content itself is trusted — see wrist MJPEG note).
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
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

    def frame(self, cam_id: str):
        """Latest RGB (H,W,3) uint8 frame for a cam id, or None."""
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
        return frame

    def jpeg(self, cam_id: str) -> bytes | None:
        frame = self.frame(cam_id)
        return encode_frame_jpeg(frame) if frame is not None else None


def _state_with_detections(labeler: LiveLabeler, detector) -> dict:
    """labeler.state() augmented with live scan detections: each kit packet gets a
    pixel bbox on the scan frame so the cockpit can draw the 'pick this next' box on
    the actual packet. When several packets share a part id (e.g. two UNN-16022-009),
    the physical detections are assigned to the kit entries POSITIONALLY so each entry
    boxes a distinct packet, not the same one twice. wh = scan size."""
    from collections import defaultdict
    from robots_realtime.labeling.detector import parse_part
    st = labeler.state()
    if detector is None:
        return st
    dets, wh = detector.current()
    st["wh"] = wh
    by_mid: dict[str, list] = defaultdict(list)     # 5-digit middle → detections (stable order)
    for d in dets:
        p = parse_part(d.part or "")
        if p:
            by_mid[p[1]].append(d)
    for lst in by_mid.values():
        lst.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    used: dict[str, int] = defaultdict(int)
    for pk in st["packets"]:
        p = parse_part(pk.get("part") or "")
        lst = by_mid.get(p[1], []) if p else []
        i = used[p[1]] if p else 0
        d = lst[i] if i < len(lst) else (lst[-1] if lst else None)
        if p and i < len(lst):
            used[p[1]] += 1
        pk["bbox_px"] = d.bbox if d else None
        pk["det_conf"] = d.conf if d else 0.0
    return st


def _make_handler(labeler: LiveLabeler, cam_base: str | None,
                  bridge: "CameraBridge | None" = None, detector=None):
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
                self._json(_state_with_detections(labeler, detector))
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
                kit = None
                if raw:
                    try:
                        kit = json.loads(raw)["packets"]
                    except Exception:
                        kit = None
                if kit is None and detector is not None:
                    # operator started recording → freeze the kit as the scan sees it NOW
                    from robots_realtime.labeling.detector import kit_from_detections
                    dets, _wh = detector.current()
                    kit = kit_from_detections(dets) or None
                labeler.seed(kit if kit is not None else DEFAULT_KIT)
                labeler.locked = True
                self._json(_state_with_detections(labeler, detector))
            elif self.path == "/advance":
                labeler.advance()
                self._json(labeler.state())
            else:
                self.send_error(404)

        def _proxy_cam(self, cam_id: str):
            # 1) live bus bridge. Firefox does NOT render multipart/x-mixed-replace
            #    inside an <img>, so serve a SINGLE JPEG per request — the cockpit
            #    re-polls with a ?t= cache-buster for live-ish video (works in all
            #    browsers). ?stream=1 opts into MJPEG for chrome/direct viewing.
            stream = self.path.split("?", 1)[1].startswith("stream=1") if "?" in self.path else False
            cam_id = cam_id.split("?", 1)[0]
            if bridge is not None:
                jpg = bridge.jpeg(cam_id)
                if jpg is not None and not stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self._cors()
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                if jpg is not None and stream:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self._cors(); self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            f = bridge.jpeg(cam_id)
                            if f is not None:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(f)).encode()
                                    + b"\r\n\r\n" + f + b"\r\n")
                            time.sleep(1 / 15.0)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return
            # 2) optional proxy fallback
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
                 bridge: "CameraBridge | None" = None, host: str = "127.0.0.1", detector=None):
        self.labeler = labeler
        self._httpd = ThreadingHTTPServer((host, port),
                                          _make_handler(labeler, cam_base, bridge, detector))
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


def episode_kit_json(packets: list[dict]) -> list[dict]:
    """Convert the live scanned kit → the kit.json shape label_episode reads
    ([{bag_id, part_no, name, compartment}, ...]) so each bag's part identity is
    saved with the recording and fused into annotations.json."""
    return [{"bag_id": p.get("bag_id", i + 1),
             "part_no": p.get("part"),
             "name": p.get("name") or None,
             "compartment": p.get("comp")}
            for i, p in enumerate(packets)]


def _run_label_episode(episode_dir: Path, arm: str) -> None:
    """Fire label_episode as an isolated subprocess → writes annotations.json.
    Isolated so a bad/incomplete episode (e.g. missing yam_<arm>.mcap) logs an
    error instead of taking down the server."""
    import subprocess
    import sys
    try:
        subprocess.Popen(
            [sys.executable, "-m", "robots_realtime.labeling.label_episode",
             str(episode_dir), "--arm", arm],
            cwd=str(Path.cwd()),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[auto-label] {episode_dir.name} → annotations.json")
    except Exception as e:
        print(f"[auto-label] failed to launch labeler for {episode_dir}: {e}")


def record_watcher(save_root: str, labeler: LiveLabeler, arm: str,
                   auto_label: bool, stop=None) -> None:
    """Watch save_root for episodes rr-session creates. On a NEW episode dir (record
    just started) write kit.json = the scanned kit. When session_meta.json appears
    (episode saved) run label_episode → annotations.json. Filesystem-coupled so the
    labeler process and the recorder process stay decoupled."""
    seen: dict[str, str] = {}
    root = Path(save_root)
    while stop is None or not stop.is_set():
        time.sleep(1.0)
        for d in sorted(root.glob("*/episode_*")) if root.exists() else []:
            if not d.is_dir():
                continue
            key, meta = str(d), d / "session_meta.json"
            if key not in seen:
                seen[key] = "recording"
                try:
                    (d / "kit.json").write_text(
                        json.dumps(episode_kit_json(labeler.packets), indent=2))
                    print(f"[auto-label] wrote {d.name}/kit.json ({len(labeler.packets)} bags)")
                except Exception as e:
                    print(f"[auto-label] kit.json write failed for {d}: {e}")
                # Box calibration is one-time and shared; if a canonical compartments.json
                # sits at the recordings root, copy it in so places get classified too.
                canon = root / "compartments.json"
                if canon.exists() and not (d / "compartments.json").exists():
                    try:
                        (d / "compartments.json").write_text(canon.read_text())
                    except Exception as e:
                        print(f"[auto-label] compartments copy failed for {d}: {e}")
            if seen[key] == "recording" and meta.exists():
                seen[key] = "done"
                if auto_label:
                    _run_label_episode(d, arm)


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
    ap.add_argument("--detect", action="store_true",
                    help="run the scan-cam packet detector (needs easyocr)")
    ap.add_argument("--detect-period", type=float, default=2.0,
                    help="seconds between scan-cam detection passes")
    ap.add_argument("--detect-gpu", action="store_true",
                    help="use GPU for OCR (leave off on the RTX 5090 — no torch kernels)")
    ap.add_argument("--save-root", default=None,
                    help="rr-session recordings root; watch it to save kit.json + auto-label")
    ap.add_argument("--auto-label", action="store_true",
                    help="run label_episode on each saved episode → annotations.json")
    args = ap.parse_args(argv)

    labeler = LiveLabeler(open_ref=args.open_ref, closed_ref=args.closed_ref)
    labeler.seed(DEFAULT_KIT)

    bridge = None
    if args.bus_cams:
        id_to_topic = dict(p.split("=", 1) for p in args.bus_cams.split(",") if "=" in p)
        bridge = CameraBridge(id_to_topic, host="127.0.0.1")
        print(f"Camera bridge: {id_to_topic}")

    # Packet detector: reads the scan cam, finds each kit packet's box + part id so
    # the cockpit can draw the "pick this next" box. Optional — needs easyocr; if it
    # is unavailable the cockpit falls back to kit-order highlighting.
    detector = None
    if bridge is not None and args.detect:
        try:
            from robots_realtime.labeling.detector import PacketDetector, kit_from_detections
            detector = PacketDetector(
                frame_source=lambda: bridge.frame("scan"),
                known_source=lambda: None,          # free-read EVERY packet on the table
                period_s=args.detect_period, gpu=args.detect_gpu).start()
            print(f"Packet detector on the scan cam (period {args.detect_period}s, gpu={args.detect_gpu})")

            # Auto-seed the kit from the scan: the pick-list = the packets actually on the
            # table (not a fixed list). Re-seed while idle so the box always reflects reality;
            # once the operator starts recording, POST /seed locks in the scanned kit.
            def _autoseed():
                while True:
                    time.sleep(args.detect_period)
                    dets, _wh = detector.current()
                    if dets and not labeler.locked:
                        kit = kit_from_detections(dets)
                        if kit and [p["part"] for p in kit] != [p.get("part") for p in labeler.packets]:
                            labeler.seed(kit)
            threading.Thread(target=_autoseed, daemon=True).start()
        except Exception as e:
            print(f"Packet detector disabled ({type(e).__name__}: {e})")

    # Watch recordings: save the scanned kit.json into each new episode and (optionally)
    # auto-run the offline labeler when the episode is saved, so every recording ends
    # up with a complete annotations.json (part id + compartment + grasp/place).
    if args.save_root:
        threading.Thread(target=record_watcher,
                         args=(args.save_root, labeler, args.arm, args.auto_label),
                         daemon=True).start()
        print(f"Watching {args.save_root} (kit.json + auto-label={args.auto_label})")

    server = LiveLabelServer(labeler, port=args.port, cam_base=args.cam_base,
                             bridge=bridge, host=args.host, detector=detector)
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
