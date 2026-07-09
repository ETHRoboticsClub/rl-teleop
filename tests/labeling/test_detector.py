"""Unit tests for the scan-cam packet detector's pure logic (blob detection + the
closed-SKU matching that turns OCR tokens into a part id)."""
import numpy as np

from robots_realtime.labeling.detector import (
    parse_part, match_sku, detect_blobs, PacketDetector, Detection,
)


def test_parse_part():
    assert parse_part("UNN-10126-151") == ("UNN", "10126", "151")
    assert parse_part("unn 10126 151") == ("UNN", "10126", "151")   # spaced/lowercase
    assert parse_part("MDDY-11065-001") == ("MDDY", "11065", "001")
    assert parse_part("garbage") is None


def test_match_sku_by_middle():
    known = ["UNN-10126-151", "UNN-10015-007", "DNN-15122-009"]
    # the 5-digit middle is the discriminator
    part, conf = match_sku([("UNN", 0.9), ("10126", 0.95), ("151", 0.5)], known)
    assert part == "UNN-10126-151" and conf == 0.95


def test_match_sku_needs_middle():
    known = ["UNN-10126-151"]
    # prefix alone (no middle) is not enough to trust the match
    assert match_sku([("UNN", 0.9)], known) == (None, 0.0)
    assert match_sku([("99999", 0.9)], known) == (None, 0.0)


def test_match_sku_disambiguates_duplicate_middle():
    # two SKUs share middle 10015; only the suffix separates them
    known = ["UNN-10015-007", "UNN-10015-231"]
    p007, _ = match_sku([("UNN", 0.9), ("10015", 0.95), ("007", 0.6)], known)
    p231, _ = match_sku([("UNN", 0.9), ("10015", 0.95), ("231", 0.6)], known)
    assert p007 == "UNN-10015-007"
    assert p231 == "UNN-10015-231"


def test_detect_blobs_finds_bright_rectangles():
    frame = np.zeros((480, 640, 3), np.uint8)          # dark mat
    frame[100:200, 120:320] = 230                      # one bright packet
    frame[300:380, 400:560] = 210                      # another
    boxes = detect_blobs(frame, min_area=1500)
    assert len(boxes) == 2
    # boxes are [x,y,w,h] and roughly cover the bright regions
    xs = sorted(b[0] for b in boxes)
    assert xs[0] < 200 and xs[1] > 350


def test_detect_blobs_ignores_full_frame():
    frame = np.full((480, 640, 3), 255, np.uint8)      # entirely bright
    assert detect_blobs(frame) == []                   # a near-full-frame blob is skipped


def test_bbox_for_prefers_highest_conf():
    det = PacketDetector(lambda: None, lambda: [])
    det._dets = [Detection([0, 0, 10, 10], "UNN-10015-007", 0.6),
                 Detection([5, 5, 20, 20], "UNN-10015-007", 0.9)]
    bbox, conf = det.bbox_for("UNN-10015-007")
    assert bbox == [5, 5, 20, 20] and conf == 0.9
    assert det.bbox_for("MISSING") == (None, 0.0)


def test_detect_once_falls_back_when_ocr_unavailable():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:200, 120:320] = 230
    det = PacketDetector(lambda: None, lambda: ["UNN-10126-151"])
    det._ocr = lambda: (_ for _ in ()).throw(RuntimeError("no easyocr"))  # force failure
    dets = det.detect_once(frame, ["UNN-10126-151"])
    # graceful: still returns unidentified blob boxes (part=None) for order fallback
    assert dets and all(d.part is None for d in dets)
