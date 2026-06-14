"""Tests for InferenceAccelerationGuardrail — acceleration limiting only."""
import numpy as np
import pytest
from robots_realtime.runtime.safety.guardrails import InferenceAccelerationGuardrail


def test_inference_acceleration_clamps_large_delta():
    """Large acceleration delta should be clamped."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.05)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = accel.apply(prev_cmd, new_cmd)
    assert result[0] <= 0.05, "Delta of 0.5 should be clamped to max_delta_per_step"


def test_inference_acceleration_passes_small_delta():
    """Small acceleration delta should pass through unchanged."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.1)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = accel.apply(prev_cmd, new_cmd)
    np.testing.assert_allclose(result, new_cmd), "Small delta should pass through unchanged"


def test_acceleration_clamps_each_dim_independently():
    """Each dimension should be clamped independently based on its delta."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.1)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([0.5, 0.05, 0.0, 0.0, 0.0, 0.0])
    result = accel.apply(prev_cmd, new_cmd)
    assert result[0] == 0.1, "First dim delta of 0.5 should be clamped to 0.1"
    assert result[1] == 0.05, "Second dim delta of 0.05 should pass through"


def test_acceleration_handles_negative_delta():
    """Negative delta should also be clamped (in magnitude)."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.1)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = accel.apply(prev_cmd, new_cmd)
    assert result[0] >= -0.1, "Negative delta of -0.5 should be clamped to -0.1"


def test_acceleration_chained_steps_accumulate_correctly():
    """Multiple steps should accumulate within the per-step limit."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.1)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Step 1
    step1 = accel.apply(prev_cmd, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert step1[0] == 0.1
    # Step 2 (from step1 result)
    step2 = accel.apply(step1, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert step2[0] == 0.2
    # Step 3
    step3 = accel.apply(step2, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.isclose(step3[0], 0.3)
