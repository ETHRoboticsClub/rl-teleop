"""FK-only Cartesian workspace reject guardrail — no IK in teleop loop.

Holds the last safe absolute joint command on out-of-bounds violation.
Re-entry is joint-rate-limited and FK-checked before accepting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np


class FKProvider(Protocol):
    """Forward kinematics provider interface.

    Implementations:
        - Production: i2rt.robots.kinematics.Kinematics
        - Tests: MockFKProvider
    """

    def fk(self, q: np.ndarray, site_name: str) -> np.ndarray:
        """Return 4x4 site pose in model/world frame."""
        ...


@dataclass
class GuardrailResult:
    """Result of applying the Cartesian workspace guardrail."""

    final_command: np.ndarray
    state: str  # "accepted" | "rejected" | "reentry"
    candidate_xyz: np.ndarray
    final_xyz: np.ndarray
    reason: str = ""
    event_payload: Dict[str, Any] = field(default_factory=dict)


class CartesianWorkspaceRejectGuardrail:
    """FK-only reject/hold-last-safe guardrail for Cartesian workspace.

    Constructor receives an FK provider object or factory, arm key, site name,
    XYZ bounds, tolerance, re-entry margin, re-entry max delta per cycle, and
    pass-through indices.

    Usage:
        gr = CartesianWorkspaceRejectGuardrail(
            fk_provider=fk,
            arm_key="left",
            site_name="wrist",
            min_xyz=[0.0, 0.0, 0.0],
            max_xyz=[1.0, 1.0, 1.0],
            tolerance_m=1e-4,
            reentry_margin_m=0.002,
            reentry_max_delta_per_cycle=0.1,
            pass_through_indices=[6],  # gripper
        )

        # Initialize last_safe from current follower state
        gr.mark_published_safe(current_q)

        # Apply to each incoming command
        result = gr.apply(candidate)
        # result.final_command is the command to publish
        # result.state is "accepted", "rejected", or "reentry"

        # After publishing (and after any other guardrails), update last_safe
        gr.mark_published_safe(final_published_command)
    """

    def __init__(
        self,
        fk_provider: FKProvider,
        arm_key: str,
        site_name: str,
        min_xyz: List[float],
        max_xyz: List[float],
        tolerance_m: float = 1e-4,
        reentry_margin_m: float = 0.002,
        reentry_max_delta_per_cycle: float = 0.1,
        pass_through_indices: Optional[List[int]] = None,
    ):
        self._fk = fk_provider
        self._arm_key = arm_key
        self._site_name = site_name

        self._min_xyz = np.array(min_xyz, dtype=np.float64)
        self._max_xyz = np.array(max_xyz, dtype=np.float64)
        self._tolerance = tolerance_m
        self._reentry_margin = reentry_margin_m
        self._reentry_max_delta = reentry_max_delta_per_cycle

        # Arm joint indices (everything NOT in pass_through_indices)
        if pass_through_indices is None:
            pass_through_indices = []
        self._pass_through = set(pass_through_indices)

        # State
        self._last_safe: Optional[np.ndarray] = None
        self._last_safe_xyz: Optional[np.ndarray] = None

        # Telemetry
        self._reject_count = 0
        self._reentry_count = 0
        self._last_log_time = 0.0
        self._log_hz = 1.0  # throttle to 1 Hz by default

    @property
    def arm_key(self) -> str:
        return self._arm_key

    @property
    def site_name(self) -> str:
        return self._site_name

    def _get_arm_indices(self, q_len: int) -> List[int]:
        """Get arm joint indices (all indices except pass-through)."""
        return [i for i in range(q_len) if i not in self._pass_through]

    def _fk_xyz(self, q: np.ndarray) -> np.ndarray:
        """Compute FK position (xyz) for joint configuration."""
        pose = self._fk.fk(q, self._site_name)
        return pose[:3, 3]

    def _is_in_bounds(self, xyz: np.ndarray) -> bool:
        """Check if xyz is within bounds (with tolerance)."""
        lo = self._min_xyz - self._tolerance
        hi = self._max_xyz + self._tolerance
        return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

    def _is_reentry(self, xyz: np.ndarray) -> bool:
        """Check if xyz is within re-entry margin of bounds."""
        lo = self._min_xyz - self._reentry_margin
        hi = self._max_xyz + self._reentry_margin
        return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

    def apply(
        self,
        candidate: np.ndarray,
        current_state: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> GuardrailResult:
        """Apply the guardrail to a candidate command.

        Args:
            candidate: Joint command array (arm joints + pass-through indices).
            current_state: Current follower state (optional, for re-entry tracking).
            now: Current timestamp for telemetry throttling.

        Returns:
            GuardrailResult with final_command to publish.
        """
        candidate = np.asarray(candidate, dtype=np.float64)

        # Initialize last_safe from first candidate if not set
        if self._last_safe is None:
            self._last_safe = candidate.copy()
            self._last_safe_xyz = self._fk_xyz(self._last_safe)

        # Compute candidate FK position
        candidate_xyz = self._fk_xyz(candidate)

        # Check bounds
        event: Dict[str, Any] = {}
        if self._is_in_bounds(candidate_xyz):
            # Accept candidate
            final = candidate.copy()
            final_xyz = candidate_xyz
            state = "accepted"
            reason = "in_bounds"
        else:
            # Reject — hold last_safe arm joints, pass through non-arm joints
            arm_idx = self._get_arm_indices(len(candidate))

            final = candidate.copy()
            final[arm_idx] = self._last_safe[arm_idx]

            # FK-check the held command
            final_xyz = self._fk_xyz(final)

            # Check if candidate is close enough for re-entry
            if self._is_reentry(candidate_xyz):
                # Rate-limit from last_safe toward candidate
                delta = candidate - self._last_safe
                arm_delta = delta[arm_idx]

                # Clamp per-joint delta to reentry_max_delta
                max_delta = self._reentry_max_delta
                sign = np.sign(arm_delta)
                clamped_delta = sign * np.minimum(np.abs(arm_delta), max_delta)

                # Apply rate-limited delta
                limited = self._last_safe.copy()
                limited[arm_idx] += clamped_delta

                # FK-check the rate-limited command
                limited_xyz = self._fk_xyz(limited)
                if self._is_in_bounds(limited_xyz):
                    final = limited
                    final_xyz = limited_xyz
                    state = "reentry"
                    reason = "reentry_rate_limited"
                    self._reentry_count += 1
                else:
                    # Rate-limited command still out of bounds — hold last_safe
                    state = "rejected"
                    reason = "out_of_bounds"
                    self._reject_count += 1
            else:
                # Far outside — hold last_safe
                state = "rejected"
                reason = "out_of_bounds"
                self._reject_count += 1

            # Build event payload
            now_ts = now if now is not None else time.time()
            if now_ts - self._last_log_time >= 1.0 / self._log_hz:
                event = {
                    "arm_key": self._arm_key,
                    "site_name": self._site_name,
                    "candidate_xyz": candidate_xyz.tolist(),
                    "final_xyz": final_xyz.tolist(),
                    "min_xyz": self._min_xyz.tolist(),
                    "max_xyz": self._max_xyz.tolist(),
                    "reason": reason,
                    "reject_count": self._reject_count,
                    "reentry_count": self._reentry_count,
                }
                self._last_log_time = now_ts

        return GuardrailResult(
            final_command=final.astype(np.float32),
            state=state,
            candidate_xyz=candidate_xyz,
            final_xyz=final_xyz,
            reason=reason,
            event_payload=event,
        )

    def mark_published_safe(self, final_command: np.ndarray) -> None:
        """Update last_safe after integration knows exact published command.

        This is the ONLY way to update last_safe. The apply() method does NOT
        update last_safe — it only reads it. This ensures last_safe equals the
        exact final command actually published after all guardrails and rate limits.
        """
        self._last_safe = np.asarray(final_command, dtype=np.float64).copy()
        self._last_safe_xyz = self._fk_xyz(self._last_safe)

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry counters."""
        return {
            "arm_key": self._arm_key,
            "site_name": self._site_name,
            "reject_count": self._reject_count,
            "reentry_count": self._reentry_count,
            "has_last_safe": self._last_safe is not None,
        }
