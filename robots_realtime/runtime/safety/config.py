"""Safety guardrail configuration + validation.

Parses per-arm command-space limits from a YAML ``safety:`` block and validates them.
Fail-closed policy: a real-hardware config must supply explicit finite numeric limits
for every guardrail it enables — null/placeholder/NaN values raise before actuation, so
a rig can never come up "guarded" while actually unbounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SafetyConfigError(ValueError):
    """Raised when a safety config is missing, malformed, or (real hardware) incomplete."""


class CommandSource(str, Enum):
    """Where the commands come from — decides which guardrails apply.

    - teleop / sim_teleop: bounding box + (optional) teleop speed cap + cartesian.
    - inference: bounding box + acceleration (speed) limit.
    """

    TELEOP = "teleop"
    SIM_TELEOP = "sim_teleop"
    INFERENCE = "inference"

    @property
    def is_teleop(self) -> bool:
        return self in (CommandSource.TELEOP, CommandSource.SIM_TELEOP)


def _as_float_array(values, field: str, arm: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SafetyConfigError(f"[{arm}] {field}: not a numeric array ({exc})") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise SafetyConfigError(f"[{arm}] {field}: must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise SafetyConfigError(f"[{arm}] {field}: contains non-finite (NaN/inf) values")
    return arr


@dataclass(frozen=True)
class BoundingBoxConfig:
    """Per-arm joint-position bounding box. Clamps ``position_indices`` to [min, max]."""

    min: np.ndarray
    max: np.ndarray
    position_indices: tuple[int, ...]

    @staticmethod
    def from_dict(d: dict, arm: str) -> "BoundingBoxConfig":
        if "min" not in d or "max" not in d:
            raise SafetyConfigError(f"[{arm}] bounding_box: requires 'min' and 'max'")
        lo = _as_float_array(d["min"], "bounding_box.min", arm)
        hi = _as_float_array(d["max"], "bounding_box.max", arm)
        if lo.shape != hi.shape:
            raise SafetyConfigError(
                f"[{arm}] bounding_box: min {lo.shape} and max {hi.shape} shapes differ"
            )
        if np.any(lo > hi):
            raise SafetyConfigError(f"[{arm}] bounding_box: some min > max")
        indices = d.get("position_indices", list(range(lo.size)))
        indices = tuple(int(i) for i in indices)
        if len(indices) != lo.size:
            raise SafetyConfigError(
                f"[{arm}] bounding_box: position_indices ({len(indices)}) "
                f"!= number of bounds ({lo.size})"
            )
        return BoundingBoxConfig(min=lo, max=hi, position_indices=indices)


@dataclass(frozen=True)
class SpeedLimitConfig:
    """Per-arm per-step joint-delta cap (used for inference accel and teleop speed cap).

    ``max_step_rad`` is the max absolute change per control step for each arm joint. It
    can be given directly, or derived from ``max_velocity_rad_s`` and ``control_hz``.
    """

    max_step_rad: float
    position_indices: tuple[int, ...]

    @staticmethod
    def from_dict(d: dict, arm: str, default_indices: tuple[int, ...]) -> "SpeedLimitConfig":
        max_step = d.get("max_step_rad")
        if max_step is None:
            vel = d.get("max_velocity_rad_s")
            hz = d.get("control_hz")
            if vel is None or hz is None:
                raise SafetyConfigError(
                    f"[{arm}] speed_limit: requires 'max_step_rad' or "
                    "('max_velocity_rad_s' and 'control_hz')"
                )
            if not (math.isfinite(vel) and math.isfinite(hz)) or vel <= 0 or hz <= 0:
                raise SafetyConfigError(
                    f"[{arm}] speed_limit: max_velocity_rad_s and control_hz must be positive finite"
                )
            max_step = float(vel) / float(hz)
        max_step = float(max_step)
        if not math.isfinite(max_step) or max_step <= 0:
            raise SafetyConfigError(f"[{arm}] speed_limit: max_step_rad must be positive finite")
        indices = d.get("position_indices")
        indices = tuple(int(i) for i in indices) if indices is not None else default_indices
        return SpeedLimitConfig(max_step_rad=max_step, position_indices=indices)


@dataclass(frozen=True)
class ArmSafety:
    """All guardrail configs for one arm.

    ``cartesian`` is the raw ``cartesian_workspace`` dict (parsed + FK-built by AgentNode,
    which owns the mujoco dependency) — kept unparsed here to avoid an import cycle.
    """

    bounding_box: BoundingBoxConfig | None = None
    speed_limit: SpeedLimitConfig | None = None
    cartesian: dict | None = None


@dataclass
class SafetyConfig:
    """Top-level safety config for an AgentNode."""

    command_source: CommandSource
    is_real_hardware: bool
    arms: dict[str, ArmSafety]

    def arm(self, arm_key: str) -> ArmSafety | None:
        return self.arms.get(arm_key)

    @property
    def any_enabled(self) -> bool:
        return any(a.bounding_box or a.speed_limit for a in self.arms.values())


def build_safety_config(params: dict | None) -> SafetyConfig | None:
    """Build a :class:`SafetyConfig` from a YAML ``safety:`` dict, or ``None`` if absent.

    Structure::

        safety:
          command_source: teleop | sim_teleop | inference
          is_real_hardware: false
          arms:
            left:
              bounding_box: { min: [...], max: [...], position_indices: [0..5] }
              speed_limit:  { max_step_rad: 0.08, position_indices: [0..5] }
            right: { ... }

    Fail-closed: for ``is_real_hardware: true`` every arm must supply the guardrails
    required by its command source with finite numeric values, or this raises.
    """
    if not params:
        return None
    if not params.get("enabled", True):
        return None

    try:
        source = CommandSource(params.get("command_source", "teleop"))
    except ValueError as exc:
        raise SafetyConfigError(f"unknown command_source: {params.get('command_source')!r}") from exc
    is_real = bool(params.get("is_real_hardware", False))

    arms_in = params.get("arms")
    if not isinstance(arms_in, dict) or not arms_in:
        raise SafetyConfigError("safety.arms must be a non-empty mapping of arm -> config")

    arms: dict[str, ArmSafety] = {}
    for arm_key, arm_cfg in arms_in.items():
        arm_cfg = arm_cfg or {}
        bbox = None
        if arm_cfg.get("bounding_box") is not None:
            bbox = BoundingBoxConfig.from_dict(arm_cfg["bounding_box"], arm_key)
        speed = None
        if arm_cfg.get("speed_limit") is not None:
            default_idx = bbox.position_indices if bbox else tuple(range(6))
            speed = SpeedLimitConfig.from_dict(arm_cfg["speed_limit"], arm_key, default_idx)
        cartesian = arm_cfg.get("cartesian_workspace")
        arms[str(arm_key)] = ArmSafety(
            bounding_box=bbox, speed_limit=speed, cartesian=cartesian
        )

    cfg = SafetyConfig(command_source=source, is_real_hardware=is_real, arms=arms)

    if is_real:
        _validate_real_hardware(cfg)
    return cfg


def _validate_real_hardware(cfg: SafetyConfig) -> None:
    """Fail closed: real hardware must have explicit limits for its command source."""
    for arm_key, arm in cfg.arms.items():
        if arm.bounding_box is None:
            raise SafetyConfigError(
                f"[{arm_key}] real-hardware safety requires an explicit bounding_box "
                "(fail-closed; fill in site-specific numeric limits before deploying)"
            )
        if cfg.command_source is CommandSource.INFERENCE and arm.speed_limit is None:
            raise SafetyConfigError(
                f"[{arm_key}] real-hardware inference requires an explicit speed_limit "
                "(acceleration/velocity bound)"
            )
