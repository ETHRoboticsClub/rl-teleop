import numpy as np


class CommandBoundingBoxGuardrail:
    """Clamp command elements to [min_vals, max_vals] for joint dimensions; preserve gripper (index 6+)."""

    def __init__(self, min_vals: np.ndarray, max_vals: np.ndarray):
        self._min = np.asarray(min_vals)
        self._max = np.asarray(max_vals)

    def apply(self, cmd: np.ndarray) -> np.ndarray:
        result = cmd.copy()
        dim = len(self._min)
        result[:dim] = np.clip(result[:dim], self._min, self._max)
        return result


class InferenceAccelerationGuardrail:
    """Cap per-element delta between consecutive commands to ±max_delta_per_step."""

    def __init__(self, max_delta_per_step: float):
        self._max_delta = max_delta_per_step

    def apply(self, prev_cmd: np.ndarray, new_cmd: np.ndarray) -> np.ndarray:
        delta = new_cmd - prev_cmd
        clamped_delta = np.clip(delta, -self._max_delta, self._max_delta)
        return prev_cmd + clamped_delta
