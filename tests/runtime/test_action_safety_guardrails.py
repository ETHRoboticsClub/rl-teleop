import numpy as np
import pytest
from robots_realtime.runtime.safety.guardrails import CommandBoundingBoxGuardrail, InferenceAccelerationGuardrail

def test_command_bbox_clamps_below_min():
    """Command below min should be clamped to min values."""
    bbox = CommandBoundingBoxGuardrail(
        min_vals=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        max_vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    cmd = np.array([-0.5, -0.3, -0.1, 0.0, 0.5, 0.8])
    result = bbox.apply(cmd)
    assert np.allclose(result[0], 0.0), "First element below min should be clamped to min"
    assert np.allclose(result[1], 0.0), "Second element below min should be clamped to min"
    assert np.allclose(result[2], 0.0), "Third element below min should be clamped to min"

def test_command_bbox_clamps_above_max():
    """Command above max should be clamped to max values."""
    bbox = CommandBoundingBoxGuardrail(
        min_vals=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        max_vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    cmd = np.array([0.2, 0.5, 0.8, 1.1, 1.3, 1.5])
    result = bbox.apply(cmd)
    assert np.allclose(result[3], 1.0), "Fourth element above max should be clamped to max"
    assert np.allclose(result[4], 1.0), "Fifth element above max should be clamped to max"
    assert np.allclose(result[5], 1.0), "Sixth element above max should be clamped to max"

def test_command_bbox_preserves_in_bounds_values():
    """Command within bounds should pass through unchanged."""
    bbox = CommandBoundingBoxGuardrail(
        min_vals=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        max_vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    cmd = np.array([0.2, -0.5, 0.0, 0.7, -0.3, 0.9])
    result = bbox.apply(cmd)
    assert np.allclose(result, cmd), "In-bounds command should pass through unchanged"

def test_gripper_index_is_preserved_by_default():
    """Gripper index (6) should not be clamped by default."""
    bbox = CommandBoundingBoxGuardrail(
        min_vals=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        max_vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    # 7-dim command with gripper at index 6 set outside [0, 1]
    cmd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.0])
    result = bbox.apply(cmd)
    # First 6 dims should be clamped normally; gripper (index 6) should be untouched
    assert np.allclose(result[:6], np.clip(cmd[:6], 0.0, 1.0)), "Joint dims should be clamped"
    assert result[6] == 2.0, "Gripper index should not be clamped by default"

def test_inference_acceleration_clamps_large_delta():
    """Large acceleration delta should be clamped."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.05)
    prev_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = accel.apply(prev_cmd, new_cmd)
    assert result[0] <= 0.05, "Delta of 0.5 should be clamped to max_delta_per_step"

def test_acceleration_then_bbox_order_leaves_final_command_in_bounds():
    """Running acceleration then bbox should leave final command in bounds."""
    accel = InferenceAccelerationGuardrail(max_delta_per_step=0.1)
    bbox = CommandBoundingBoxGuardrail(
        min_vals=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        max_vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    prev_cmd = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_cmd = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    after_accel = accel.apply(prev_cmd, new_cmd)
    final = bbox.apply(after_accel)
    assert final[0] >= -1.0 and final[0] <= 1.0, "Final command must be within bbox"
    assert final[0] <= 0.9 + 0.1, "Final command must respect acceleration limit"
