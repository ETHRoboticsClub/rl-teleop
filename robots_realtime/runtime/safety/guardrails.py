"""Pure command-space guardrail filters (bounding box + speed limit).

Each guardrail takes a command vector and returns a (clamped command, event) pair. They
never mutate their input, never drop a command (clamp, not reject), and only touch the
configured arm-joint indices — gripper/other indices pass through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots_realtime.runtime.safety.config import BoundingBoxConfig, SpeedLimitConfig


@dataclass
class ClampEvent:
    """A structured record that a guardrail modified a command."""

    arm: str
    guardrail: str
    max_abs_change: float  # largest |clamped - original| across affected indices

    def __str__(self) -> str:
        return (
            f"{self.guardrail}[{self.arm}] clamped command "
            f"(max change {self.max_abs_change:.4f})"
        )


class BoundingBoxGuardrail:
    """Clamp configured joint indices into [min, max]; pass everything else through."""

    name = "bounding_box"

    def __init__(self, config: BoundingBoxConfig, arm: str = "default") -> None:
        self._cfg = config
        self._arm = arm
        self._idx = np.asarray(config.position_indices, dtype=int)
        self.clamp_count = 0

    def apply(self, command: np.ndarray) -> tuple[np.ndarray, ClampEvent | None]:
        out = np.array(command, dtype=np.float64, copy=True)
        valid = self._idx[self._idx < out.size]
        if valid.size == 0:
            return out, None
        # Align bounds to the indices that actually exist in this command.
        lo = self._cfg.min[: valid.size]
        hi = self._cfg.max[: valid.size]
        original = out[valid].copy()
        out[valid] = np.clip(original, lo, hi)
        change = np.abs(out[valid] - original)
        if np.any(change > 0):
            self.clamp_count += 1
            return out, ClampEvent(self._arm, self.name, float(change.max()))
        return out, None


class SpeedLimitGuardrail:
    """Clamp per-step joint deltas to ``max_step_rad`` from the last emitted command.

    Stateful: the reference is the command actually emitted last step (so repeated
    clamping ramps toward the target instead of snapping). Used both for inference
    acceleration limiting and the teleop speed cap.
    """

    name = "speed_limit"

    def __init__(self, config: SpeedLimitConfig, arm: str = "default") -> None:
        self._cfg = config
        self._arm = arm
        self._idx = np.asarray(config.position_indices, dtype=int)
        self._max_step = float(config.max_step_rad)
        self._last: np.ndarray | None = None
        self.clamp_count = 0

    def reset(self, current: np.ndarray | None = None) -> None:
        """Reset the reference to the current follower state (or clear it)."""
        if current is None:
            self._last = None
            return
        cur = np.asarray(current, dtype=np.float64)
        valid = self._idx[self._idx < cur.size]
        self._last = cur[valid].copy()

    def apply(
        self, command: np.ndarray, current: np.ndarray | None = None
    ) -> tuple[np.ndarray, ClampEvent | None]:
        out = np.array(command, dtype=np.float64, copy=True)
        valid = self._idx[self._idx < out.size]
        if valid.size == 0:
            return out, None

        if self._last is None:
            if current is not None:
                # Seed from the real follower state and clamp this command against it —
                # the first command must be rate-limited too, not snap from wherever the
                # arm actually is.
                cur = np.asarray(current, dtype=np.float64)
                self._last = cur[valid].copy() if cur.size > valid.max() else out[valid].copy()
            else:
                # No reference to clamp against — establish it from this command.
                self._last = out[valid].copy()
                return out, None

        ref = self._last
        if ref.shape != valid.shape:  # command width changed — reseed defensively
            self._last = out[valid].copy()
            return out, None

        desired = out[valid]
        delta = np.clip(desired - ref, -self._max_step, self._max_step)
        limited = ref + delta
        change = np.abs(desired - limited)
        out[valid] = limited
        self._last = limited.copy()
        if np.any(change > 1e-12):
            self.clamp_count += 1
            return out, ClampEvent(self._arm, self.name, float(change.max()))
        return out, None
