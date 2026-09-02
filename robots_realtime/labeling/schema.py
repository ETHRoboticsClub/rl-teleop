"""Annotation schema for kitting episodes.

Two files per episode:
  annotations.json  — machine-generated, immutable, regenerable by label_episode.
  corrections.json  — human overrides (from the cockpit confirm sweep).

They are merged on READ (`load_annotations`): a correction replaces the matching
machine record by its key. Raw MCAP/MP4 recordings are never touched.

Everything is plain dataclasses with dict (de)serialization so the file is a
readable JSON any tool can consume, and `from_dict` tolerates unknown/missing
fields for forward compatibility (bump LABELER_VERSION on breaking changes).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bump the MINOR on additive fields, MAJOR when a re-label of old episodes is
# required. Stamped into every annotations.json so consumers can gate on it.
LABELER_VERSION = "0.2.0"   # 0.2.0: additive GraspAttempt hold fields, "no_lift"

# Phase segmentation vocabulary. The two "align" phases from the plan are
# disambiguated into align_grasp / align_place (eng-review finding).
PHASES = (
    "idle", "reach", "align_grasp", "grasp", "lift",
    "transport", "align_place", "place", "retract",
)
# Per-grasp-attempt outcomes. "empty" = closed on nothing.
#
# "no_lift" is emitted by the GRASP task only (constants.TASK_GRASP): the jaws
# closed on something wide enough not to read as empty, but the end-effector
# never rose MIN_LIFT_M within LIFT_WINDOW_S, so nothing was picked up. The
# kitting task expresses the same fact through the placement chain (no lift =>
# no terminal grasp => no place event), which a grasp-only demo does not have.
# It is deliberately NOT "success": every consumer filters on outcome ==
# "success", so a new non-success value is excluded by every existing filter
# without any of them being changed.
GRASP_OUTCOMES = ("success", "slip", "drop", "empty", "no_lift")
# Whole-episode outcomes.
EPISODE_OUTCOMES = ("success", "partial", "aborted", "unknown")

Pose = list  # [x, y, z, qw, qx, qy, qz] in the robot base frame


def _clean(d: dict) -> dict:
    """Drop None values so JSON stays compact."""
    return {k: v for k, v in d.items() if v is not None}


def _from_dict(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys (forward-compat)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class KitItem:
    """One planned bag in the kit (from OCR + the deterministic routing)."""
    bag_id: int
    part_no: str | None = None      # None = OCR could not read it → manual assign
    name: str | None = None
    compartment: int | None = None  # target 1..7; None if unrouted


@dataclass
class EpisodeMeta:
    episode_id: str
    arm: str = "left"
    box_id: str | None = None
    kitting_list: list[KitItem] = field(default_factory=list)
    outcome: str = "unknown"        # one of EPISODE_OUTCOMES
    t_start: float = 0.0
    t_end: float = 0.0
    # Offset added to cockpit-event timestamps to land them on the hardware
    # clock. 0.0 once the cockpit shares rl-teleop's clock (the chosen design).
    clock_offset_s: float = 0.0
    # Box pose in the robot base frame. None => assume the calibrated fixed pose.
    box_pose: list | None = None
    labeler_version: str = LABELER_VERSION

    def to_dict(self) -> dict:
        d = _clean(dataclasses.asdict(self))
        d["kitting_list"] = [_clean(dataclasses.asdict(k)) for k in self.kitting_list]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeMeta":
        data = dict(data)
        data["kitting_list"] = [_from_dict(KitItem, k) for k in data.get("kitting_list", [])]
        return _from_dict(cls, data)


@dataclass
class Segment:
    bag_id: int
    arm: str
    phase: str          # one of PHASES
    t_start: float
    t_end: float


@dataclass
class GraspAttempt:
    bag_id: int
    attempt: int
    arm: str
    t: float
    ee_pose: Pose | None = None
    close_width_norm: float | None = None   # normalized [0,1], 0=closed
    outcome: str = "success"                 # one of GRASP_OUTCOMES
    regrasp_of: int | None = None            # attempt # this re-grasps, if any
    # --- hold, added 0.2.0 -------------------------------------------------
    # HOW LONG THE JAWS STAYED SHUT is a first-class quality signal for a
    # grasp-only policy: together with the FK lift it is what separates a real
    # pick from a brush-and-release. Transport distance cannot play that role
    # here because a grasp demo never transports anything.
    # None on annotations written before 0.2.0 — consumers must treat missing
    # as unknown, never as zero.
    t_open: float | None = None              # release time; None = still shut at episode end
    hold_s: float | None = None              # t_open - t (episode end if never released)
    lifted: bool | None = None               # FK lift within LIFT_WINDOW_S; None = not evaluable


@dataclass
class PlaceEvent:
    bag_id: int
    t: float
    target_compartment: int | None = None
    detected_compartment: int | None = None  # compartment the release ACTUALLY landed in
    achieved_ee_pose: Pose | None = None      # (the demonstrated sorting label)
    in_target_region: bool | None = None     # None = no region / not evaluable
    xy_offset_m: float | None = None
    release_height_m: float | None = None


@dataclass
class TrackingError:
    """Commanded (gello leader / IK target) vs achieved (follower FK)."""
    phase: str
    arm: str
    rms_pos_err: float
    max_pos_err: float


@dataclass
class Flag:
    """A loud anomaly. The labeler NEVER silently drops data — it flags it so
    the human review sweep surfaces it. This is the anti-silent-mislabel guard."""
    kind: str        # e.g. "clock_out_of_window", "overlapping_grasp", "ocr_null"
    detail: str
    t: float | None = None
    bag_id: int | None = None


@dataclass
class Annotations:
    episode_meta: EpisodeMeta
    segments: list[Segment] = field(default_factory=list)
    grasp_attempts: list[GraspAttempt] = field(default_factory=list)
    place_events: list[PlaceEvent] = field(default_factory=list)
    tracking: list[TrackingError] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)

    # -- (de)serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "labeler_version": LABELER_VERSION,
            "episode_meta": self.episode_meta.to_dict(),
            "segments": [_clean(dataclasses.asdict(s)) for s in self.segments],
            "grasp_attempts": [_clean(dataclasses.asdict(g)) for g in self.grasp_attempts],
            "place_events": [_clean(dataclasses.asdict(p)) for p in self.place_events],
            "tracking": [_clean(dataclasses.asdict(t)) for t in self.tracking],
            "flags": [_clean(dataclasses.asdict(f)) for f in self.flags],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotations":
        return cls(
            episode_meta=EpisodeMeta.from_dict(data.get("episode_meta", {})),
            segments=[_from_dict(Segment, s) for s in data.get("segments", [])],
            grasp_attempts=[_from_dict(GraspAttempt, g) for g in data.get("grasp_attempts", [])],
            place_events=[_from_dict(PlaceEvent, p) for p in data.get("place_events", [])],
            tracking=[_from_dict(TrackingError, t) for t in data.get("tracking", [])],
            flags=[_from_dict(Flag, f) for f in data.get("flags", [])],
        )

    def save(self, path: str | Path) -> None:
        """Write annotations.json ATOMICALLY (temp file in the same dir + rename).

        The previous implementation was ``write_text``, which truncates the file
        first and then writes. A reader that opens it in that gap — the cockpit,
        review_corpus.py, the exporter, or a second labelling run — gets an empty
        or half-written file and reports the episode as having no grasps, which
        is indistinguishable from a genuinely empty label set. os.replace is
        atomic on POSIX within one filesystem, so a reader sees either the whole
        old file or the whole new one, never a torn one. The temp file is created
        in the SAME directory for exactly that reason (a cross-device rename is
        not atomic and would fall back to a copy).

        A crash mid-write now leaves the previous annotations.json intact plus a
        stray .annotations.json.* temp file, instead of a truncated one.
        """
        path = Path(path)
        payload = json.dumps(self.to_dict(), indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def merge_corrections(ann: Annotations, corrections: dict) -> Annotations:
    """Apply human overrides. Corrections win. Supported override blocks:

        {"episode_meta": {...partial...},
         "grasp_attempts": {"<bag_id>:<attempt>": {...partial...}},
         "place_events":   {"<bag_id>": {...partial...}},
         "kitting_list":   {"<bag_id>": {...partial...}}}

    A partial dict updates only the named fields of the matching machine record,
    so a human fixing a compartment doesn't wipe the machine-derived pose.
    """
    out = Annotations.from_dict(ann.to_dict())  # deep copy via round-trip

    for k, v in corrections.get("episode_meta", {}).items():
        if hasattr(out.episode_meta, k):
            setattr(out.episode_meta, k, v)

    kl_over = corrections.get("kitting_list", {})
    for item in out.episode_meta.kitting_list:
        ov = kl_over.get(str(item.bag_id))
        if ov:
            for k, v in ov.items():
                if hasattr(item, k):
                    setattr(item, k, v)

    ga_over = corrections.get("grasp_attempts", {})
    for g in out.grasp_attempts:
        ov = ga_over.get(f"{g.bag_id}:{g.attempt}")
        if ov:
            for k, v in ov.items():
                if hasattr(g, k):
                    setattr(g, k, v)

    pe_over = corrections.get("place_events", {})
    for p in out.place_events:
        ov = pe_over.get(str(p.bag_id))
        if ov:
            for k, v in ov.items():
                if hasattr(p, k):
                    setattr(p, k, v)

    return out


def load_annotations(episode_dir: str | Path) -> Annotations:
    """Load annotations.json and merge corrections.json if present."""
    episode_dir = Path(episode_dir)
    ann = Annotations.from_dict(json.loads((episode_dir / "annotations.json").read_text()))
    corr_path = episode_dir / "corrections.json"
    if corr_path.exists():
        ann = merge_corrections(ann, json.loads(corr_path.read_text()))
    return ann
