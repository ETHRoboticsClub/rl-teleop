"""FK-based Cartesian workspace reject/hold guardrail (teleop).

Bounds the follower's end-effector to a Cartesian box using forward kinematics only —
no IK — so it is cheap enough for the 200 Hz teleop loop. On a violation it *holds* the
last safe joint command (keeps publishing at cadence, never drops), and on re-entry it
rate-limits the joint move back toward the leader so the arm does not snap.

The FK is injected as a callable ``fk(arm_joints) -> xyz`` so this class is unit-testable
with a mock and works with any backend (mujoco / i2rt Kinematics) in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots_realtime.runtime.safety.guardrails import ClampEvent


class CartesianConfigError(ValueError):
    """Raised for a malformed cartesian workspace config."""


@dataclass(frozen=True)
class CartesianWorkspaceConfig:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    position_indices: tuple[int, ...]
    tolerance_m: float = 1e-4
    reentry_max_step_rad: float = 0.05

    @staticmethod
    def from_dict(d: dict, arm: str) -> "CartesianWorkspaceConfig":
        for key in ("min_xyz", "max_xyz"):
            if key not in d:
                raise CartesianConfigError(f"[{arm}] cartesian_workspace: requires '{key}'")
        lo = np.asarray(d["min_xyz"], dtype=np.float64)
        hi = np.asarray(d["max_xyz"], dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise CartesianConfigError(f"[{arm}] cartesian_workspace: min/max_xyz must be length-3")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise CartesianConfigError(f"[{arm}] cartesian_workspace: non-finite bounds")
        if np.any(lo > hi):
            raise CartesianConfigError(f"[{arm}] cartesian_workspace: min_xyz > max_xyz")
        indices = tuple(int(i) for i in d.get("position_indices", range(6)))
        tol = float(d.get("tolerance_m", 1e-4))
        reentry = float(d.get("reentry_max_step_rad", 0.05))
        if tol < 0 or reentry <= 0:
            raise CartesianConfigError(f"[{arm}] cartesian_workspace: bad tolerance/reentry")
        return CartesianWorkspaceConfig(lo, hi, indices, tol, reentry)


class CartesianWorkspaceRejectGuardrail:
    """Hold the last safe joint command whenever the FK end-effector leaves the box."""

    name = "cartesian_workspace"

    def __init__(self, config: CartesianWorkspaceConfig, fk, arm: str = "default") -> None:
        self._cfg = config
        self._fk = fk  # callable: arm_joints (ndarray) -> xyz (ndarray, shape (3,))
        self._arm = arm
        self._idx = np.asarray(config.position_indices, dtype=int)
        self._last_safe: np.ndarray | None = None  # last safe arm-joint vector
        self._holding = False
        self.reject_count = 0

    def _xyz(self, arm_joints: np.ndarray) -> np.ndarray:
        return np.asarray(self._fk(arm_joints), dtype=np.float64).reshape(3)

    def _in_box(self, xyz: np.ndarray) -> bool:
        tol = self._cfg.tolerance_m
        return bool(np.all(xyz >= self._cfg.min_xyz - tol) and np.all(xyz <= self._cfg.max_xyz + tol))

    def reset(self, current_joints: np.ndarray) -> None:
        """Seed the hold state from the current follower joints. Fails closed if the
        current pose is already outside the box (guarded teleop must not start unsafe)."""
        arm = np.asarray(current_joints, dtype=np.float64)
        valid = self._idx[self._idx < arm.size]
        arm_joints = arm[valid]
        if not self._in_box(self._xyz(arm_joints)):
            raise CartesianConfigError(
                f"[{self._arm}] cannot start guarded teleop: current pose is outside the "
                "Cartesian workspace box"
            )
        self._last_safe = arm_joints.copy()
        self._holding = False

    def apply(self, command: np.ndarray) -> tuple[np.ndarray, ClampEvent | None]:
        out = np.array(command, dtype=np.float64, copy=True)
        valid = self._idx[self._idx < out.size]
        if valid.size == 0:
            return out, None
        candidate = out[valid]

        if self._last_safe is None or self._last_safe.shape != candidate.shape:
            # No hold state yet — accept if in box and seed, else we cannot hold, so we
            # still publish the candidate (bounding box / motor limits remain downstream).
            if self._in_box(self._xyz(candidate)):
                self._last_safe = candidate.copy()
                self._holding = False
            return out, None

        if self._in_box(self._xyz(candidate)):
            if self._holding:
                # Re-entry: rate-limit the joint move back toward the leader, and only
                # accept the limited step if it is itself inside the box.
                step = np.clip(
                    candidate - self._last_safe,
                    -self._cfg.reentry_max_step_rad,
                    self._cfg.reentry_max_step_rad,
                )
                limited = self._last_safe + step
                if self._in_box(self._xyz(limited)):
                    out[valid] = limited
                    self._last_safe = limited.copy()
                    reached = np.allclose(limited, candidate)
                    self._holding = not reached
                    return out, ClampEvent(self._arm, self.name, float(np.abs(candidate - limited).max()))
                # Limited step still leaves the box — keep holding.
                out[valid] = self._last_safe
                self.reject_count += 1
                return out, ClampEvent(self._arm, self.name, float(np.abs(candidate - self._last_safe).max()))
            # Tracking normally.
            self._last_safe = candidate.copy()
            return out, None

        # Candidate outside the box — hold the last safe command.
        self._holding = True
        self.reject_count += 1
        out[valid] = self._last_safe
        return out, ClampEvent(self._arm, self.name, float(np.abs(candidate - self._last_safe).max()))
