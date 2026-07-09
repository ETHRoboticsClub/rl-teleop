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


# Real Bühler part-id prefixes on the packets. OCR frequently mangles these (UNN → UMM /
# UMH / UMN / UNM; MDDY → MDO / MD / MDOT) so we snap each read to the nearest real prefix.
KNOWN_PREFIXES = ["UNN", "DNN", "MDDY"]


def _edit_dist(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def normalize_prefix(part: str) -> str:
    p = parse_part(part)
    if not p:
        return part
    pre, mid, suf = p
    best = min(KNOWN_PREFIXES, key=lambda k: _edit_dist(pre, k))
    return f"{best}-{mid}-{suf}"


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


def enhance_for_ocr(frame, gamma: float = 1.0):
    """Make the printed part numbers more readable for OCR: gray-world white balance
    (kills the D435i green cast), a gamma (software 'exposure' — <1 brighter, >1 darker),
    and CLAHE local contrast so the black-on-white text pops. Does NOT touch the camera
    or the recorded frames — this is applied only to the OCR input."""
    import cv2
    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6           # gray-world WB
    f = np.clip(f * (means.mean() / means), 0, 255)
    if gamma != 1.0:
        f = np.clip(((f / 255.0) ** gamma) * 255.0, 0, 255)
    out = f.astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


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


def _clean_dets(dets: list["Detection"], min_conf: float,
                merge_px: int = 75) -> list["Detection"]:
    """Turn raw OCR reads into one clean detection per physical packet:
    drop low-confidence noise (field labels, order numbers), keep the
    highest-confidence read per spatial cluster (dedupe prefix-misread twins),
    and snap the prefix to the nearest real Bühler prefix."""
    good = sorted((d for d in dets if d.conf >= min_conf), key=lambda d: -d.conf)
    kept: list[Detection] = []
    for d in good:
        cx, cy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
        if any(abs(cx - (k.bbox[0] + k.bbox[2] / 2)) < merge_px and
               abs(cy - (k.bbox[1] + k.bbox[3] / 2)) < merge_px for k in kept):
            continue                                   # same packet, lower-conf twin
        kept.append(Detection(d.bbox, normalize_prefix(d.part) if d.part else None, d.conf))
    return kept


class PacketDetector:
    """Detects packets on the scan camera and reads their part ids.

    Thread-safe: a background thread refreshes ``detections`` from the latest
    scan frame; the HTTP handler reads ``current()``. ``frame_source`` is a
    callable returning the latest scan RGB frame (H,W,3) uint8 or None.
    """

    def __init__(self, frame_source, known_source, period_s: float = 2.0,
                 gpu: bool = False, ttl_cycles: int = 4, min_conf: float = 0.5):
        self._frame_source = frame_source          # () -> np.ndarray | None
        self._known_source = known_source          # () -> list[str] | None
        self._period = period_s
        self._ttl_cycles = ttl_cycles              # hold a box this many missed cycles
        self.min_conf = min_conf                   # drop OCR reads below this confidence
        self._lock = threading.Lock()
        self._dets: list[Detection] = []
        self._wh: list[int] = [0, 0]
        self._reader = None
        self._gpu = gpu
        self._stop = threading.Event()
        self._expected = 0             # high-water mark of packets seen (recalibration target)
        self._recalibrate = True       # sweep exposure/gamma when we come up short

    def _ocr(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def detect_once(self, frame: np.ndarray, known=None,
                    gamma: float = 1.0) -> list[Detection]:
        """Read EVERY packet on the scan frame: enhance for OCR (white-balance +
        gamma + contrast), OCR, group each printed part number (prefix + 5-digit
        middle + 3-digit suffix) into one box, then dedupe/clean. ``gamma`` is the
        software exposure used by the recalibration sweep. ``known`` (optional) snaps
        a read to the closest catalog SKU."""
        try:
            res = self._ocr().readtext(enhance_for_ocr(frame, gamma))
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
        pres = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"[A-Z]{2,4}", t)]
        sufs = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"\d{3}", t)]
        H, W = frame.shape[:2]

        def nearest(cands, ctr, maxd):
            best, bd = None, maxd
            for c in cands:
                d = abs(c[3][0] - ctr[0]) + 2 * abs(c[3][1] - ctr[1])   # weight y (same line)
                if d < bd:
                    bd, best = d, c
            return best

        # Read EVERY packet: a Bühler part number is PREFIX MIDDLE(5) SUFFIX(3) printed on
        # one line. Anchor on each 5-digit middle that has a letter prefix right next to it
        # (this rejects order-number noise, which has no adjacent prefix).
        dets: list[Detection] = []
        for (mid, mconf, mbb, mctr) in mids:
            pre = nearest(pres, mctr, maxd=max(140, 3 * mbb[2]))
            if pre is None:
                continue                                   # no prefix → not a part number
            suf = nearest(sufs, mctr, maxd=max(140, 3 * mbb[2]))
            part = f"{pre[0]}-{mid}-{(suf[0] if suf else '000')}"
            mx, my = mbb[2], mbb[3]                         # expand the middle-number box onto the packet
            box = [max(0, mbb[0] - mx), max(0, mbb[1] - my), mbb[2] + 2 * mx, mbb[3] + 3 * my]
            box[2] = min(box[2], W - box[0]); box[3] = min(box[3], H - box[1])
            if known:                                       # optional: snap to catalog SKU
                snap, _ = match_sku([(pre[0], pre[1]), (mid, mconf)] +
                                    ([(suf[0], suf[1])] if suf else []), known)
                if snap:
                    part = snap
            dets.append(Detection([int(v) for v in box], part, round(mconf, 2)))
        return _clean_dets(dets, self.min_conf)

    def _loop(self):
        # Spatial tracker: one entry per PHYSICAL packet (keyed by box POSITION, not the
        # part text — OCR jitters the prefix/suffix across passes, so text keys would
        # accumulate duplicates of the same packet). Each position keeps its highest-conf
        # reading; a packet holds its box for _ttl_cycles of misses (arm occlusion) then
        # drops once it's been gone a while (picked / removed).
        held: list[list] = []          # [Detection, age]
        while not self._stop.is_set():
            frame = self._frame_source()
            known = self._known_source() or None
            if frame is not None:
                try:
                    frame = np.asarray(frame)
                    fresh = self.detect_once(frame, known)
                    # Auto-recalibrate: if we found fewer packets than we've seen before,
                    # a packet is likely washed out / too dark — sweep software exposure
                    # (gamma) and merge whatever extra reads that surfaces.
                    if self._recalibrate and len(fresh) < self._expected:
                        pool = list(fresh)
                        for g in (0.55, 1.7):
                            pool += self.detect_once(frame, known, gamma=g)
                        fresh = _clean_dets(pool, self.min_conf)
                    self._expected = max(self._expected, len(fresh))
                    used = [False] * len(fresh)

                    def _near(det):
                        cx, cy = det.bbox[0] + det.bbox[2] / 2, det.bbox[1] + det.bbox[3] / 2
                        for i, f in enumerate(fresh):
                            if used[i]:
                                continue
                            fx, fy = f.bbox[0] + f.bbox[2] / 2, f.bbox[1] + f.bbox[3] / 2
                            if abs(cx - fx) < 75 and abs(cy - fy) < 75:
                                return i
                        return -1

                    new_held: list[list] = []
                    for det, age in held:
                        i = _near(det)
                        if i >= 0:
                            used[i] = True
                            f = fresh[i]
                            new_held.append([f if f.conf >= det.conf else det, 0])
                        elif age + 1 < self._ttl_cycles:
                            new_held.append([det, age + 1])        # hold last-good
                    for i, f in enumerate(fresh):
                        if not used[i]:
                            new_held.append([f, 0])                # newly seen packet
                    held = new_held
                    with self._lock:
                        # final box-dedupe guard: the tracker can hold two co-located
                        # entries if a cycle briefly saw a packet twice; collapse them.
                        self._dets = _clean_dets([d for (d, _age) in held], self.min_conf)
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
        """Best detected bbox for ``part``, matched on the reliable 5-digit MIDDLE
        (OCR reads the middle far better than the suffix). Prefers an exact suffix
        match to disambiguate duplicate middles, else highest confidence."""
        p = parse_part(part)
        if not p:
            return None, 0.0
        _pre, mid, suf = p
        with self._lock:
            cands = [d for d in self._dets
                     if (parse_part(d.part) or (None, None, None))[1] == mid]
        if not cands:
            return None, 0.0
        exact = [d for d in cands if (parse_part(d.part) or (None, None, None))[2] == suf]
        best = max(exact or cands, key=lambda d: d.conf)
        return best.bbox, best.conf


def kit_from_detections(dets: list["Detection"], max_comp: int = 7) -> list[dict]:
    """Build the pick-list from what the scan cam actually sees: one entry per
    detected packet, ordered top-to-bottom then left-to-right, with a "grouped"
    part→compartment rule (identical parts share a compartment, assigned in
    first-seen order, capped at ``max_comp``)."""
    dets = _clean_dets(dets, 0.0)      # box-dedupe guard: never two entries per packet
    ordered = sorted(dets, key=lambda d: (d.bbox[1] // 120, d.bbox[0]))
    comp_of: dict[str, int] = {}
    kit = []
    for i, d in enumerate(ordered):
        if d.part and d.part not in comp_of:
            comp_of[d.part] = min(len(comp_of) + 1, max_comp)
        kit.append({"bag_id": i + 1, "part": d.part, "name": "",
                    "comp": comp_of.get(d.part), "bbox": [0.0, 0.0, 0.0, 0.0]})
    return kit
