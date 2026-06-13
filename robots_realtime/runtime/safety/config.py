"""Safety configuration schema and validation for bounding boxes, acceleration limits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class SafetyConfig:
    """Runtime safety parameters clamped before actuation.

    Attributes:
        mode: "real" or "sim".
        agent_type: "inference" or "teleop".
        bounding_box: {"min": [...], "max": [...]} joint limits or None.
        acceleration_limit: Max joint acceleration (rad/s^2) or None.
    """

    mode: str = "sim"
    agent_type: str = "teleop"
    bounding_box: Optional[Dict[str, List[float]]] = None
    acceleration_limit: Optional[float] = None


def validate_safety_config(cfg: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
    """Validate a safety config dict; raise ValueError on invalid values.

    Rules:
      - Real hardware requires a numeric bounding_box (not None).
      - Real hardware + inference requires a numeric acceleration_limit.
      - Teleop does NOT require acceleration_limit.
      - Bounding box values must not contain NaN or inf.
      - Bounding box min must be <= max element-wise.

    Args:
        cfg: Config dict with keys like "mode", "agent_type", "bounding_box", "acceleration_limit".

    Returns:
        True (or the config dict) on success.

    Raises:
        ValueError: If any safety constraint is violated.
    """
    mode = cfg.get("mode", "sim")
    agent_type = cfg.get("agent_type", "teleop")
    bbox = cfg.get("bounding_box")
    accel = cfg.get("acceleration_limit")

    # ── Real hardware requires bounding box ────────────────────────────────
    if mode == "real" and bbox is None:
        raise ValueError(
            "Real hardware requires a numeric bounding_box; got None. "
            "Fill in bounding_box.min / bounding_box.max before running on real arms."
        )

    # ── Real hardware + inference requires acceleration limit ──────────────
    if mode == "real" and agent_type == "inference" and accel is None:
        raise ValueError(
            "Real hardware inference requires a numeric acceleration_limit. "
            "Set acceleration_limit to constrain joint acceleration."
        )

    # ── Validate acceleration_limit when provided ──────────────────────────
    if accel is not None:
        try:
            accel_f = float(accel)
        except (TypeError, ValueError):
            raise ValueError(
                f"acceleration_limit must be numeric, got {type(accel).__name__}"
            )
        if math.isnan(accel_f):
            raise ValueError("acceleration_limit is NaN")
        if math.isinf(accel_f):
            raise ValueError("acceleration_limit is inf")
        if accel_f <= 0:
            raise ValueError(f"acceleration_limit must be positive, got {accel_f}")

    # ── Validate bounding box contents ─────────────────────────────────────
    if bbox is not None:
        _validate_bbox(bbox)

    return True


def _validate_bbox(bbox: dict) -> None:
    """Check bounding box for missing keys, empty arrays, length mismatch, NaN, inf, and min > max."""
    if "min" not in bbox:
        raise ValueError("bounding_box missing 'min' key")
    if "max" not in bbox:
        raise ValueError("bounding_box missing 'max' key")
    mins = bbox["min"]
    maxs = bbox["max"]
    if len(mins) == 0:
        raise ValueError("bounding_box.min must not be empty")
    if len(mins) != len(maxs):
        raise ValueError(
            f"bounding_box.min length ({len(mins)}) != bounding_box.max length ({len(maxs)})"
        )
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        lo_f = float(lo)
        hi_f = float(hi)
        if math.isnan(lo_f):
            raise ValueError(f"bounding_box.min[{i}] is NaN")
        if math.isnan(hi_f):
            raise ValueError(f"bounding_box.max[{i}] is NaN")
        if math.isinf(lo_f):
            raise ValueError(f"bounding_box.min[{i}] is inf")
        if math.isinf(hi_f):
            raise ValueError(f"bounding_box.max[{i}] is inf")
        if lo_f > hi_f:
            raise ValueError(
                f"bounding_box.min[{i}] ({lo_f}) > bounding_box.max[{i}] ({hi_f})"
            )


def validate_cartesian_workspace_config(cfg: Dict[str, Any]) -> bool:
    """Validate a cartesian workspace config dict; raise ValueError on invalid values.

    Requires agent_type, site_name, xml_path, frame, min_xyz, max_xyz when present.
    Follows the same element-wise validation pattern as _validate_bbox.

    Args:
        cfg: Config dict with cartesian workspace keys.

    Returns:
        True on success.

    Raises:
        ValueError: If any constraint is violated.
    """
    # ── Required fields ────────────────────────────────────────────────────
    for key in ("agent_type", "site_name", "xml_path"):
        if key not in cfg:
            raise ValueError(f"cartesian workspace missing '{key}'")

    agent_type = cfg["agent_type"]
    site_name = cfg["site_name"]
    xml_path = cfg["xml_path"]

    # ── agent_type must be teleop or inference ─────────────────────────────
    if agent_type not in ("teleop", "inference"):
        raise ValueError(
            f"cartesian workspace agent_type must be 'teleop' or 'inference', got '{agent_type}'"
        )

    # ── site_name must be non-empty ────────────────────────────────────────
    if not isinstance(site_name, str) or not site_name.strip():
        raise ValueError("cartesian workspace site_name must be a non-empty string")

    # ── xml_path must be non-empty ─────────────────────────────────────────
    if not isinstance(xml_path, str) or not xml_path.strip():
        raise ValueError("cartesian workspace xml_path must be a non-empty string")

    # ── frame must be 'model' only ─────────────────────────────────────────
    if "frame" not in cfg:
        raise ValueError("cartesian workspace missing 'frame'")
    frame = cfg["frame"]
    if frame != "model":
        raise ValueError(
            f"cartesian workspace frame must be 'model', got '{frame}'"
        )

    # ── min_xyz / max_xyz presence ─────────────────────────────────────────
    if "min_xyz" not in cfg:
        raise ValueError("cartesian workspace missing 'min_xyz'")
    if "max_xyz" not in cfg:
        raise ValueError("cartesian workspace missing 'max_xyz'")

    mins = cfg["min_xyz"]
    maxs = cfg["max_xyz"]

    if len(mins) == 0:
        raise ValueError("cartesian workspace min_xyz must not be empty")
    if len(mins) != len(maxs):
        raise ValueError(
            f"cartesian workspace min_xyz length ({len(mins)}) != max_xyz length ({len(maxs)}), mismatch"
        )

    # ── Element-wise numeric validation (NaN, inf, min > max) ─────────────
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        lo_f = float(lo)
        hi_f = float(hi)
        if math.isnan(lo_f):
            raise ValueError(f"cartesian workspace min_xyz[{i}] is NaN")
        if math.isnan(hi_f):
            raise ValueError(f"cartesian workspace max_xyz[{i}] is NaN")
        if math.isinf(lo_f):
            raise ValueError(f"cartesian workspace min_xyz[{i}] is inf")
        if math.isinf(hi_f):
            raise ValueError(f"cartesian workspace max_xyz[{i}] is inf")
        if lo_f > hi_f:
            raise ValueError(
                f"cartesian workspace min_xyz[{i}] ({lo_f}) exceeds max_xyz[{i}] ({hi_f})"
            )

    # ── fk_ik_clamp not allowed for teleop ─────────────────────────────────
    enforcement = cfg.get("enforcement")
    if agent_type == "teleop" and enforcement == "fk_ik_clamp":
        raise ValueError(
            "fk_ik_clamp enforcement is not supported for teleop mode. "
            "Use 'reject_hold' or omit enforcement for teleop agents."
        )

    return True
