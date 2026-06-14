import numpy as np
from typing import Callable, Dict, Any, List, Optional

class GuardrailResult:
    def __init__(self, final_command, state, candidate_xyz, final_xyz, reason, event_payload):
        self.final_command = final_command
        self.state = state
        self.candidate_xyz = candidate_xyz
        self.final_xyz = final_xyz
        self.reason = reason
        self.event_payload = event_payload

class CartesianWorkspaceRejectGuardrail:
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
        pass_through_indices: List[int] = None,
    ):
        self._fk_provider = fk_provider
        self._arm_key = arm_key
        self._site_name = site_name
        self._min_xyz = np.array(min_xyz)
        self._max_xyz = np.array(max_xyz)
        self._tolerance_m = tolerance_m
        self._reentry_margin_m = reentry_margin_m
        self._reentry_max_delta_per_cycle = float(reentry_max_delta_per_cycle)
        self._pass_through_indices = set(pass_through_indices) if pass_through_indices is not None else None
        self._last_safe_q: Optional[np.ndarray] = None

    def _fk_call(self, q: np.ndarray, site: str) -> np.ndarray:
        if hasattr(self._fk_provider, 'fk'):
            return self._fk_provider.fk(q, site)
        return self._fk_provider(q, site)

    def _is_in_bounds(self, pos: np.ndarray) -> bool:
        return np.all(pos >= (self._min_xyz - self._tolerance_m)) and \
               np.all(pos <= (self._max_xyz + self._tolerance_m))

    def apply(self, candidate: np.ndarray, current_state: Optional[np.ndarray] = None, now: Optional[float] = None) -> GuardrailResult:
        candidate_q = np.array(candidate, copy=True)
        candidate_pose = self._fk_call(candidate_q, self._site_name)
        candidate_xyz = candidate_pose[:3, 3]
        in_bounds = self._is_in_bounds(candidate_xyz)
        
        res_state = "accepted"
        res_reason = "in bounds"
        res_final_q = np.array(candidate_q, copy=True)

        if not in_bounds:
            if self._last_safe_q is None:
                res_state, res_reason, res_final_q = "rejected", "no last_safe available", candidate_q
            else:
                res_state, res_reason = "rejected", "outside workspace bounds"
                res_final_q = np.array(candidate_q, copy=True)
                for i in range(6):
                    res_final_q[i] = self._last_safe_q[i]
        else:
            if self._last_safe_q is not None and self._reentry_max_delta_per_cycle > 0:
                diff = candidate_q - self._last_safe_q
                clamped_diff = np.clip(diff, -self._reentry_max_delta_per_cycle, self._reentry_max_delta_per_cycle)
                reentry_q = self._last_safe_q + clamped_diff
                reentry_pose = self._fk_call(reentry_q, self._site_name)
                reentry_xyz = reentry_pose[:3, 3]
                if self._is_in_bounds(reentry_xyz):
                    res_state = "reentry"
                    res_final_q = np.array(candidate_q, copy=True)
                    for i in range(6):
                        res_final_q[i] = reentry_q[i]
                else:
                    res_state, res_reason = "rejected", "re-entry rate limit"
                    res_final_q = np.array(candidate_q, copy=True)
                    for i in range(6):
                        res_final_q[i] = self._last_safe_q[i]

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
            event_payload={}
        )

    def check(self, candidate: np.ndarray, last_safe: Optional[np.ndarray] = None) -> np.ndarray:
        if last_safe is not None:
            orig = self._last_safe_q
            self._last_safe_q = last_safe
            res = self.apply(candidate)
            self._last_safe_q = orig
            return res.final_command
        return self.apply(candidate).final_command

    def fk_result(self, command: np.ndarray) -> np.ndarray:
        return self._fk_call(command, self._site_name)

    def mark_published_safe(self, final_command: np.ndarray):
        self._last_safe_q = np.array(final_command, copy=True)
