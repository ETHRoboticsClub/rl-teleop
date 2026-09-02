"""Shared paths and real-episode fixtures for the W1/W2/W3 tests.

The real-episode tests are REGRESSIONS ON RECORDED DATA, not smoke tests: the
numbers they assert were measured at implementation time (2026-09-02) and are
written down in the test that asserts them. They skip — never fail — when the
recordings are not on this machine, because a corpus that has moved is not a
code defect.

NOTHING HERE WRITES INTO AN EPISODE DIRECTORY. Every label run goes through
``write=False``: a recording session may be live, its auto-labeller writes the
same annotations.json, and a test is not allowed to race it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

URDF = str(REPO / "urdf" / "yam.urdf")
RECORDINGS = REPO / "recordings" / "20260902"

# The episodes these tests pin, and what each one is here to prove.
EP_GRASP_ONLY = "episode_143533_ee94747f"   # 4 grasps deleted by the transport gate
EP_KITTING_A = "episode_144835_983d9d07"    # 30 exported windows
EP_KITTING_B = "episode_145907_21b66012"    # 42 exported windows
# The newest finished takes: the HELD-OUT set. They must never be counted in a
# training total, and they are the regression fixtures for grasp-mode labelling.
EP_HELD_OUT = ("episode_150947_4d8802e5", "episode_152731_bb5a03ba")


def episode(name: str) -> Path:
    ep = RECORDINGS / name
    if not ep.is_dir() or not (ep / "yam_left.mcap").exists():
        pytest.skip(f"{ep} not on this machine")
    return ep
