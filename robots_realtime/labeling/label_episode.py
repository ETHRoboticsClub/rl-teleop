"""Offline episode labeler — the orchestrator.

    label_from_arrays(...)  pure function: joint timeline → Annotations (testable).
    label_episode_dir(dir)  reads MCAP + optional cockpit_events/compartments/kit,
                            writes annotations.json.

Idempotent: same inputs → same annotations.json. Raw recordings are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.fk import ForwardKinematics, tracking_error
from robots_realtime.labeling.fuse import GraspCandidate, build_annotations
from robots_realtime.labeling.mcap_io import read_jsonl, read_positions
from robots_realtime.labeling.placement import Compartment, load_compartments
from robots_realtime.labeling.schema import (
    Annotations,
    Flag,
    KitItem,
    TrackingError,
)
from robots_realtime.labeling.segmentation import detect_grip_intervals


def _joints_at(times: np.ndarray, joints: np.ndarray, t: float) -> np.ndarray:
    """Arm-joint vector nearest to time t."""
    return joints[int(np.argmin(np.abs(times - t)))]


def label_from_arrays(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    fk: ForwardKinematics,
    arm: str = "left",
    episode_id: str = "episode",
    gripper_open_ref: float | None = None,
    gripper_closed_ref: float | None = None,
    cockpit_events: list[dict] | None = None,
    compartments: list[Compartment] | None = None,
    kitting_list: list[KitItem] | None = None,
    commanded_positions: tuple[np.ndarray, np.ndarray] | None = None,
    clock_offset_s: float = 0.0,
) -> Annotations:
    times = np.asarray(times, float)
    positions = np.asarray(positions, float)
    arm_joints = positions[:, : C.N_ARM_JOINTS]
    gripper = positions[:, C.GRIPPER_JOINT_INDEX]

    ee_pos = fk.ee_positions(arm_joints)
    ee_z = ee_pos[:, 2]

    intervals = detect_grip_intervals(
        times, gripper, ee_z=ee_z,
        open_ref=gripper_open_ref, closed_ref=gripper_closed_ref,
    )

    candidates: list[GraspCandidate] = []
    for iv in intervals:
        grasp_pose = fk.ee_pose(_joints_at(times, arm_joints, iv.t_close))
        release_pose = (fk.ee_pose(_joints_at(times, arm_joints, iv.t_open))
                        if iv.t_open is not None else None)
        candidates.append(GraspCandidate(
            iv.t_close, iv.t_open, iv.outcome, iv.lifted, grasp_pose, release_pose))

    t_start = float(times[0]) if times.size else 0.0
    t_end = float(times[-1]) if times.size else 0.0
    outcome = "success" if candidates else "aborted"

    ann = build_annotations(
        episode_id, arm, t_start, t_end, candidates,
        kitting_list=kitting_list, cockpit_events=cockpit_events,
        compartments=compartments, clock_offset_s=clock_offset_s, outcome=outcome,
    )

    # If the gripper range wasn't given and the signal never nears full close,
    # empty-vs-bag can't be told apart — flag it loudly (don't fail silently).
    if gripper_closed_ref is None:
        from robots_realtime.labeling.segmentation import normalize_width
        if normalize_width(gripper).min() > 0.15 and candidates:
            ann.flags.append(Flag(
                "gripper_range_unknown",
                "gripper never near full close and no known limits; empty-grasp "
                "detection may be unreliable — pass gripper limits",
            ))

    # Commanded-vs-achieved tracking error over each transport segment.
    if commanded_positions is not None:
        c_times, c_pos = commanded_positions
        c_times = np.asarray(c_times, float)
        c_arm = np.asarray(c_pos, float)[:, : C.N_ARM_JOINTS]
        for seg in [s for s in ann.segments if s.phase == "transport"]:
            m = (times >= seg.t_start) & (times <= seg.t_end)
            cm = (c_times >= seg.t_start) & (c_times <= seg.t_end)
            if m.sum() < 2 or cm.sum() < 2:
                continue
            achieved = ee_pos[m]
            commanded = fk.ee_positions(c_arm[cm])
            rms, mx = tracking_error(commanded, achieved)
            ann.tracking.append(TrackingError("transport", arm, rms, mx))

    return ann


def _load_kit(episode_dir: Path) -> list[KitItem] | None:
    """kit.json = [{bag_id, part_no, name, compartment}, ...] if the operator/cockpit saved one."""
    p = episode_dir / "kit.json"
    if not p.exists():
        return None
    return [KitItem(**{k: item.get(k) for k in ("bag_id", "part_no", "name", "compartment")})
            for item in json.loads(p.read_text())]


def label_episode_dir(episode_dir: str | Path, arm: str = "left",
                      urdf_path: str | Path = "urdf/yam.urdf",
                      gripper_open_ref: float | None = None,
                      gripper_closed_ref: float | None = None,
                      write: bool = True) -> Annotations:
    episode_dir = Path(episode_dir)
    fk = ForwardKinematics(urdf_path)

    times, positions = read_positions(episode_dir / f"yam_{arm}.mcap", f"yam_{arm}")
    if times.size == 0:
        raise RuntimeError(f"no joint data in {episode_dir}/yam_{arm}.mcap")

    commanded = None
    gello = episode_dir / f"gello_{arm}.mcap"
    if gello.exists():
        ct, cp = read_positions(gello, f"gello_{arm}")
        if ct.size:
            commanded = (ct, cp)

    comps = None
    comps_path = episode_dir / "compartments.json"
    if comps_path.exists():
        comps = load_compartments(comps_path)

    ann = label_from_arrays(
        times, positions, fk=fk, arm=arm, episode_id=episode_dir.name,
        gripper_open_ref=gripper_open_ref, gripper_closed_ref=gripper_closed_ref,
        cockpit_events=read_jsonl(episode_dir / "cockpit_events.jsonl") or None,
        compartments=comps, kitting_list=_load_kit(episode_dir),
        commanded_positions=commanded,
    )
    if write:
        ann.save(episode_dir / "annotations.json")
    return ann
