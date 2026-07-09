"""Placement classification against a synthetic 2x2 compartment grid."""
from __future__ import annotations

from robots_realtime.labeling.placement import (
    Compartment,
    classify_release,
    compartment_at,
)

# 2x2 grid of 0.1x0.1 m compartments, small gaps between them.
GRID = [
    Compartment(1, 0.00, 0.10, 0.00, 0.10),
    Compartment(2, 0.12, 0.22, 0.00, 0.10),
    Compartment(3, 0.00, 0.10, 0.12, 0.22),
    Compartment(4, 0.12, 0.22, 0.12, 0.22),
]


def test_point_inside_compartment():
    assert compartment_at(0.05, 0.05, GRID, margin=0.0) == 1
    assert compartment_at(0.17, 0.17, GRID, margin=0.0) == 4


def test_point_outside_all():
    assert compartment_at(0.5, 0.5, GRID, margin=0.0) is None


def test_margin_expands_region():
    # 0.5cm outside compartment 1 (and nearer its center than comp 2's):
    # no margin → miss; 2cm margin → hit compartment 1
    assert compartment_at(0.105, 0.05, GRID, margin=0.0) is None
    assert compartment_at(0.105, 0.05, GRID, margin=0.02) == 1


def test_overlapping_margins_pick_nearest_center():
    # midpoint between 1 and 2 with big margin → nearest center wins
    assert compartment_at(0.109, 0.05, GRID, margin=0.05) == 1
    assert compartment_at(0.121, 0.05, GRID, margin=0.05) == 2


def test_classify_correct_placement():
    r = classify_release((0.05, 0.05), target_compartment=1, compartments=GRID)
    assert r.detected_compartment == 1
    assert r.in_target_region is True
    assert r.xy_offset_m < 0.02          # near center


def test_classify_wrong_compartment():
    r = classify_release((0.17, 0.17), target_compartment=1, compartments=GRID)
    assert r.detected_compartment == 4
    assert r.in_target_region is False
    # offset is distance to the TARGET (comp 1) center, so it's large
    assert r.xy_offset_m > 0.1


def test_classify_dropped_outside():
    r = classify_release((0.5, 0.5), target_compartment=2, compartments=GRID)
    assert r.detected_compartment is None
    assert r.in_target_region is False


def test_classify_no_target():
    r = classify_release((0.05, 0.05), target_compartment=None, compartments=GRID)
    assert r.detected_compartment == 1
    assert r.in_target_region is None
    assert r.xy_offset_m is None
