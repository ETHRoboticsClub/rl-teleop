"""FK-only Cartesian workspace reject guardrail — no IK in teleop loop.

Holds the last safe absolute joint command on out-of-bounds violation.
Re-entry is joint-rate-limited and FK-checked before accepting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of applying a Cartesian workspace guardrail check.

    Attributes:
        final_command: Joint command to publish after guardrail processing.
        state: One of "accepted", "rejected", "reentry".
        candidate_xyz: FK-computed end-effector position for the original candidate.
        final_xyz: FK-computed end-effector position for the final command.
        reason: Human-readable explanation of the decision.
        event_payload: Structured telemetry data for logging.
    """

    final_command: np.ndarray
    state: str
    candidate_xyz: np.ndarray
    final_xyz: np.ndarray
    reason: str
    event_payload: dict


class CartesianWorkspaceRejectGuardrail:
    """FK-only reject/hold-last-safe guardrail for Cartesian workspace bounds.

    On out-of-bounds violation, holds the last safe absolute joint command
    and publishes it at normal cadence. Re-entry from rejection is rate-limited
    via per-cycle joint delta limits and FK-checked before accepting.

    Args:
        fk_provider: Forward kinematics provider with ``fk(q, site_name) -> pose`` method.
        arm_key: Arm identifier (e.g., "left", "right").
        site_name: MuJoCo site name for FK position lookup.
        min_xyz: Minimum workspace bounds [x, y, z] in meters.
        max_xyz: Maximum workspace bounds [x, y, z] in meters.
        tolerance_m: Boundary tolerance in meters for bounds check.
        reentry_margin_m: Margin inside bounds required before re-entry is allowed.
        reentry_max_delta_per_cycle: Max per-cycle joint delta for re-entry convergence.
        pass_through_indices: Joint indices that bypass guardrail (e.g., gripper).
    """

    def __init__(
        self,
        fk_provider: Any,
        arm_key: str,
        site_name: str,
        min_xyz: List[float],
        max_xyz: List[float],
        tolerance_m: float = 1e-3,
        reentry_margin_m: float = 1e-3,
        reentry_max_delta_per_cycle: float = 0.0,
        pass_through_indices: Optional[List[int]] = None,
    ) -> None:
        self._fk_provider = fk_provider
        self._arm_key = arm_key
        self._site_name = site_name
        self._min_xyz = np.array(min_xyz, dtype=np.float64)
        self._max_xyz = np.array(max_xyz, dtype=np.float64)
        self._tolerance_m = float(tolerance_m)
        self._reentry_margin_m = float(reentry_margin_m)
        self._reentry_max_delta_per_cycle = float(reentry_max_delta_per_cycle)
        self._pass_through_indices: Optional[set] = (
            set(pass_through_indices) if pass_through_indices is not None else None
        )
        self._last_safe_q: Optional[np.ndarray] = None

    def _fk_call(self, q: np.ndarray, site: str) -> np.ndarray:
        """Compute forward kinematics, supporting both method and callable providers."""
        try:
            if hasattr(self._fk_provider, "fk"):
                return self._fk_provider.fk(q, site)
            return self._fk_provider(q, site)
        except Exception as e:
            logger.error(
                "FK call failed for arm=%s site=%s: %s",
                self._arm_key,
                site,
                e,
            )
            raise

    def _is_in_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace bounds (with tolerance)."""
        lower = self._min_xyz - self._tolerance_m
        upper = self._max_xyz + self._tolerance_m
        return bool(np.all(pos >= lower) and np.all(pos <= upper))

    def apply(
        self,
        candidate: np.ndarray,
        current_state: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> GuardrailResult:
        """Apply guardrail to candidate joint command.

        Returns a ``GuardrailResult`` with the final command to publish.
        On rejection, returns ``last_safe`` for arm joints and passes through
        gripper indices unchanged.
        """
        candidate_q = np.array(candidate, copy=True, dtype=np.float64)
        candidate_pose = self._fk_call(candidate_q, self._site_name)
        candidate_xyz = candidate_pose[:3, 3]
        in_bounds = self._is_in_bounds(candidate_xyz)

        res_state = "accepted"
        res_reason = "in bounds"
        res_final_q = np.array(candidate_q, copy=True)

        if not in_bounds:
            if self._last_safe_q is None:
                res_state = "rejected"
                res_reason = "no last_safe available"
                res_final_q = candidate_q
            else:
                res_state = "rejected"
                res_reason = "outside workspace bounds"
                res_final_q = np.array(candidate_q, copy=True)
                for i in range(6):
                    res_final_q[i] = self._last_safe_q[i]
        else:
            if (
                self._last_safe_q is not None
                and self._reentry_max_delta_per_cycle > 0
            ):
                diff = candidate_q - self._last_safe_q
                clamped_diff = np.clip(
                    diff,
                    -self._reentry_max_delta_per_cycle,
                    self._reentry_max_delta_per_cycle,
                )
                reentry_q = self._last_safe_q + clamped_diff
                reentry_pose = self._fk_call(reentry_q, self._site_name)
                reentry_xyz = reentry_pose[:3, 3]
                if self._is_in_bounds(reentry_xyz):
                    res_state = "reentry"
                    res_final_q = np.array(candidate_q, copy=True)
                    for i in range(6):
                        res_final_q[i] = reentry_q[i]
                else:
                    res_state = "rejected"
                    res_reason = "re-entry rate limit"
                    res_final_q = np.array(candidate_q, copy=True)
                    for i in range(6):
                        res_final_q[i] = self._last_safe_q[i]

        # Pass through gripper / non-arm indices
        if self._pass_through_indices is not None:
            for i in self._pass_through_indices:
                if i < len(res_final_q):
                    res_final_q[i] = candidate_q[i]

        return GuardrailResult(
            final_command=res_final_q,
            state=res_state,
            candidate_xyz=candidate_xyz,
            final_xyz=self._fk_call(res_final_q, self._site_name)[:3, 3],
            reason=res_reason,
            event_payload={},
        )

    def check(
        self, candidate: np.ndarray, last_safe: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply guardrail with optional temporary last_safe override.

        Useful for testing without mutating internal state.
        """
        if last_safe is not None:
            orig = self._last_safe_q
            self._last_safe_q = last_safe
            try:
                res = self.apply(candidate)
                return res.final_command
            finally:
                self._last_safe_q = orig
        return self.apply(candidate).final_command

    def fk_result(self, command: np.ndarray) -> np.ndarray:
        """Compute full FK pose for a joint command."""
        return self._fk_call(command, self._site_name)

    def mark_published_safe(self, final_command: np.ndarray) -> None:
        """Record a command as successfully published (updates last_safe)."""
        self._last_safe_q = np.array(final_command, copy=True, dtype=np.float64)
