"""Live packet detector for the kitting cockpit "pick this next" box.

The scan camera (D435i, top-down over the box) sees the loose Bühler packets. This
finds each packet's bounding box and reads its part id, so the cockpit can highlight
the exact packet the current kit step wants.

Pipeline (per frame, ~0.3-0.5 Hz — packets are static during a session):

    scan RGB frame
      │
      ▼  detect_blobs()  — bright packets on the dark mat (threshold + contours)
    [bbox_px, ...]
      │
      ▼  read_part_id()  — easyocr the crop, pull digit tokens, match the CLOSED
      │                    SKU set by the discriminative 5-digit middle field
    [{part, bbox, conf}, ...]

Barcode decoding was tried first and does not resolve at this camera range; OCR of
the large printed part number does. easyocr is optional — if it is missing or the GPU
kernel is unavailable, detections still carry bboxes with part=None and the cockpit
falls back to kit-order highlighting.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np

# A Bühler part id is PREFIX-MIDDLE-SUFFIX, e.g. UNN-10126-151. The MIDDLE 5 digits
# are the discriminative field OCR reads most reliably (conf ~0.9 on the real cam).
_PART_RE = re.compile(r"([A-Z]{2,4})[-\s]?(\d{5})[-\s]?(\d{3})")


def parse_part(part_id: str) -> tuple[str, str, str] | None:
    m = _PART_RE.search((part_id or "").upper().replace(" ", ""))
    return (m.group(1), m.group(2), m.group(3)) if m else None


@dataclass
class Detection:
    bbox: list[int]        # [x, y, w, h] in scan-frame pixels
    part: str | None       # matched kit part id, or None if unidentified
    conf: float            # OCR/match confidence 0..1


def detect_blobs(frame: np.ndarray, min_area: int = 1500,
                 thresh: int = 140) -> list[list[int]]:
    """Bright packets on the dark mat → list of [x,y,w,h] pixel bboxes."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 and frame.shape[2] == 3 else frame
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    H, W = gray.shape[:2]
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.6 * W * H:      # skip a near-full-frame blob (background)
            continue
        out.append([int(x), int(y), int(w), int(h)])
    return out


def match_sku(tokens: list[tuple[str, float]], known: list[str]) -> tuple[str | None, float]:
    """Match OCR tokens against the closed SKU set.

    Score each known part id: +2 if its 5-digit middle appears in the tokens,
    +1 for the 3-digit suffix, +0.5 for the prefix. The middle field is the
    reliable discriminator; suffix/prefix disambiguate duplicate middles.
    Returns (best_part_or_None, confidence).
    """
    digits5 = {t for (t, _) in tokens if re.fullmatch(r"\d{5}", t)}
    digits3 = {t for (t, _) in tokens if re.fullmatch(r"\d{3}", t)}
    prefixes = {t for (t, _) in tokens if re.fullmatch(r"[A-Z]{2,4}", t)}
    conf5 = {t: c for (t, c) in tokens if re.fullmatch(r"\d{5}", t)}
    best, best_score, best_conf = None, 0.0, 0.0
    for part in known:
        p = parse_part(part)
        if not p:
            continue
        pre, mid, suf = p
        score = 0.0
        if mid in digits5:
            score += 2.0
        if suf in digits3:
            score += 1.0
        if pre in prefixes:
            score += 0.5
        if score > best_score:
            best, best_score = part, score
            best_conf = conf5.get(mid, 0.5)
    # require at least the middle field to trust the match
    if best is None or best_score < 2.0:
        return None, 0.0
    return best, round(float(best_conf), 2)


class PacketDetector:
    """Detects packets on the scan camera and reads their part ids.

    Thread-safe: a background thread refreshes ``detections`` from the latest
    scan frame; the HTTP handler reads ``current()``. ``frame_source`` is a
    callable returning the latest scan RGB frame (H,W,3) uint8 or None.
    """

    def __init__(self, frame_source, known_source, period_s: float = 2.0,
                 gpu: bool = False):
        self._frame_source = frame_source          # () -> np.ndarray | None
        self._known_source = known_source          # () -> list[str] (kit part ids)
        self._period = period_s
        self._lock = threading.Lock()
        self._dets: list[Detection] = []
        self._wh: list[int] = [0, 0]
        self._reader = None
        self._gpu = gpu
        self._stop = threading.Event()

    def _ocr(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def detect_once(self, frame: np.ndarray, known: list[str]) -> list[Detection]:
        """OCR-region-driven: read the whole scan frame once, find the printed
        part-number tokens, and for each KNOWN kit SKU locate the packet by its
        high-confidence 5-digit middle field. The box is the OCR token region
        expanded to roughly cover the packet label.

        This is more robust than blob segmentation, which merges adjacent packets.
        Blob detection is kept (detect_blobs) as a fallback for the packet extent.
        """
        try:
            res = self._ocr().readtext(np.asarray(frame))
        except Exception:
            # OCR unavailable → fall back to unidentified blobs (cockpit uses order)
            return [Detection(b, None, 0.0) for b in detect_blobs(frame)]

        # tokens with pixel boxes + centers
        toks = []   # (text_upper, conf, [x,y,w,h], (cx,cy))
        for (box, txt, conf) in res:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            bb = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            cx, cy = bb[0] + bb[2] / 2, bb[1] + bb[3] / 2
            for t in re.split(r"[^A-Za-z0-9]+", txt.upper()):
                if t:
                    toks.append((t, float(conf), bb, (cx, cy)))

        mids = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"\d{5}", t)]
        sufs = [(t, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"\d{3}", t)]
        H, W = frame.shape[:2]
        dets: list[Detection] = []
        for part in known:
            p = parse_part(part)
            if not p:
                continue
            _pre, mid, suf = p
            # all label instances whose middle matches this SKU
            cands = [(c, bb, ctr) for (t, c, bb, ctr) in mids if t == mid]
            if not cands:
                continue
            # disambiguate duplicate middles (e.g. 10015-007 vs 10015-231) by whether
            # the SKU's own 3-digit suffix is printed nearby.
            def near_suffix(ctr):
                return min((abs(ctr[0] - sc[0]) + abs(ctr[1] - sc[1])
                            for (t, _bb, sc) in sufs if t == suf), default=1e9)
            best = min(cands, key=lambda cbc: (near_suffix(cbc[2]) > 120, -cbc[0]))
            conf, tokbb, _ctr = best
            # box on the packet: expand the label-number box (no blob snap — blobs merge)
            mx, my = tokbb[2], tokbb[3]
            box = [max(0, tokbb[0] - mx), max(0, tokbb[1] - my),
                   min(W, tokbb[2] + 2 * mx) if tokbb[0] - mx >= 0 else tokbb[2] + mx,
                   tokbb[3] + 3 * my]
            box[2] = min(box[2], W - box[0]); box[3] = min(box[3], H - box[1])
            dets.append(Detection([int(v) for v in box], part, round(conf, 2)))
        return dets

    def _loop(self):
        while not self._stop.is_set():
            frame = self._frame_source()
            known = self._known_source() or []
            if frame is not None:
                try:
                    dets = self.detect_once(np.asarray(frame), known)
                    with self._lock:
                        self._dets = dets
                        self._wh = [int(frame.shape[1]), int(frame.shape[0])]
                except Exception:
                    pass
            self._stop.wait(self._period)

    def start(self) -> "PacketDetector":
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def current(self) -> tuple[list[Detection], list[int]]:
        with self._lock:
            return list(self._dets), list(self._wh)

    def bbox_for(self, part: str) -> tuple[list[int] | None, float]:
        """Best (highest-conf) detected bbox whose part id matches ``part``."""
        with self._lock:
            cands = [d for d in self._dets if d.part == part]
        if not cands:
            return None, 0.0
        best = max(cands, key=lambda d: d.conf)
        return best.bbox, best.conf
