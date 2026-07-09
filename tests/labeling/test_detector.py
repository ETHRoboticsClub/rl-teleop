"""Unit tests for the scan-cam packet detector's pure logic (blob detection + the
closed-SKU matching that turns OCR tokens into a part id)."""
import numpy as np

from robots_realtime.labeling.detector import (
    parse_part, match_sku, detect_blobs, PacketDetector, Detection,
    kit_from_detections, normalize_prefix, _clean_dets,
)


def test_normalize_prefix():
    assert normalize_prefix("UMM-10126-151") == "UNN-10126-151"   # UMM → UNN
    assert normalize_prefix("MDO-11065-001") == "MDDY-11065-001"  # MDO → MDDY
    assert normalize_prefix("UNN-10015-007") == "UNN-10015-007"   # already good


def test_clean_dets_dedupes_thresholds_normalizes():
    dets = [Detection([480, 157, 100, 56], "UMM-10126-000", 0.76),   # real packet
            Detection([479, 157, 102, 56], "UMM-10426-000", 0.65),   # same packet twin
            Detection([520, 680, 78, 32], "VSDA-02290-000", 0.36),   # noise < threshold
            Detection([800, 590, 100, 64], "UMH-10015-007", 0.98)]   # another packet
    out = _clean_dets(dets, min_conf=0.5)
    assert sorted(d.part for d in out) == ["UNN-10015-007", "UNN-10126-000"]


class _FakeReader:
    """Stub easyocr: readtext returns fixed (box, text, conf) triples."""
    def __init__(self, results):
        self._results = results

    def readtext(self, _img):
        return self._results


def _word(txt, x, y, w=40, h=20, conf=0.9):
    return ([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], txt, conf)


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


def test_detect_once_reads_all_packets_generically():
    # two real packets + one order-number noise line (10 digits, no adjacent prefix)
    results = [
        _word("UNN", 100, 100), _word("10126", 145, 100), _word("151", 205, 100),
        _word("MDDY", 100, 300), _word("11065", 150, 300), _word("001", 210, 300),
        _word("3050765961", 100, 500),   # noise — not a 5-digit middle, ignored
    ]
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = _FakeReader(results)
    dets = det.detect_once(np.zeros((600, 640, 3), np.uint8), None)
    assert sorted(d.part for d in dets) == ["MDDY-11065-001", "UNN-10126-151"]


def test_detect_once_needs_prefix_near_middle():
    # a 5-digit number with NO letter prefix nearby is not a part number
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = _FakeReader([_word("48291", 100, 100), _word("77310", 400, 400)])
    assert det.detect_once(np.zeros((480, 640, 3), np.uint8), None) == []


def test_kit_from_detections_grouped_and_ordered():
    dets = [Detection([300, 50, 40, 20], "UNN-10015-007", 0.9),    # top row, right
            Detection([100, 50, 40, 20], "UNN-10126-151", 0.9),    # top row, left
            Detection([100, 300, 40, 20], "UNN-10015-007", 0.9)]   # lower row, dup part
    kit = kit_from_detections(dets)
    assert [k["part"] for k in kit] == ["UNN-10126-151", "UNN-10015-007", "UNN-10015-007"]
    assert kit[0]["comp"] == 1      # first distinct part
    assert kit[1]["comp"] == 2      # second distinct part
    assert kit[2]["comp"] == 2      # duplicate shares its part's compartment


def test_bbox_for_matches_by_middle_despite_suffix_flicker():
    det = PacketDetector(lambda: None, lambda: None)
    det._dets = [Detection([1, 2, 3, 4], "UNN-10015-000", 0.8)]   # suffix misread as 000
    bbox, _conf = det.bbox_for("UNN-10015-007")                   # kit wants -007
    assert bbox == [1, 2, 3, 4]                                   # matched on middle 10015


def test_detect_once_falls_back_when_ocr_unavailable():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:200, 120:320] = 230
    det = PacketDetector(lambda: None, lambda: ["UNN-10126-151"])
    det._ocr = lambda: (_ for _ in ()).throw(RuntimeError("no easyocr"))  # force failure
    dets = det.detect_once(frame, ["UNN-10126-151"])
    # graceful: still returns unidentified blob boxes (part=None) for order fallback
    assert dets and all(d.part is None for d in dets)
