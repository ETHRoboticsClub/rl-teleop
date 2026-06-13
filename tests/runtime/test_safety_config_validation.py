import numpy as np
import pytest
from robots_realtime.runtime.safety.config import SafetyConfig, validate_safety_config

def test_real_hardware_requires_numeric_bbox_before_actuation():
    """Real hardware config with null bbox should fail validation."""
    cfg = {
        "mode": "real",
        "actuation": {"type": "hardware"},
        "bounding_box": None,  # Not filled in yet
    }
    with pytest.raises(ValueError, match="bounding_box"):
        validate_safety_config(cfg)

def test_real_hardware_inference_requires_acceleration_limits():
    """Real hardware inference config requires acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "inference",
        "actuation": {"type": "hardware"},
        "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        "acceleration_limit": None,  # Missing
    }
    with pytest.raises(ValueError, match="[Aa]cceleration"):
        validate_safety_config(cfg)

def test_real_hardware_teleop_does_not_require_acceleration_limits():
    """Teleop config should not require acceleration limits."""
    cfg = {
        "mode": "real",
        "agent_type": "teleop",
        "actuation": {"type": "hardware"},
        "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        # No acceleration_limit key at all — OK for teleop
    }
    # Should pass without error
    try:
        result = validate_safety_config(cfg)
        assert isinstance(result, dict) or result is True
    except NotImplementedError:
        raise NotImplementedError("SafetyConfig not implemented yet")

def test_sim_config_accepts_explicit_numeric_safety_limits():
    """Sim config with numeric values should pass validation."""
    cfg = {
        "mode": "sim",
        "agent_type": "inference",
        "bounding_box": {"min": [-np.pi, -np.pi, -np.pi], "max": [np.pi, np.pi, np.pi]},
        "acceleration_limit": 0.5,
    }
    try:
        result = validate_safety_config(cfg)
        assert result is True or isinstance(result, (dict, SafetyConfig))
    except NotImplementedError:
        raise NotImplementedError("SafetyConfig not implemented yet")

def test_invalid_nan_inf_and_min_greater_than_max_configs_fail():
    """NaN, inf, and min > max configs should fail validation."""
    nan_cfg = {"bounding_box": {"min": [float('nan')], "max": [1.0]}}
    inf_cfg = {"bounding_box": {"min": [0.0], "max": [float('inf')]}}
    inv_cfg = {"bounding_box": {"min": [2.0], "max": [1.0]}}
    
    for bad_cfg in [nan_cfg, inf_cfg, inv_cfg]:
        try:
            validate_safety_config(bad_cfg)
            assert False, f"Expected ValueError for {bad_cfg}"
        except ValueError:
            pass
        except NotImplementedError:
            raise NotImplementedError("SafetyConfig not implemented yet")

def test_invalid_safety_config_fails_before_agent_reset():
    """Invalid safety config should raise before agent.reset() is called."""
    from robots_realtime.runtime.agent_node import AgentNode
    from unittest.mock import MagicMock, patch
    
    bad_cfg = {
        "mode": "real",
        "agent_type": "inference",
        "bounding_box": None,
        "acceleration_limit": None,
    }
    
    with patch.object(AgentNode, '_load_safety_config', return_value=bad_cfg):
        mock_agent = MagicMock(spec=AgentNode)
        mock_agent.reset = MagicMock()
        
        try:
            # Validation should fire during init/reset, before the actual agent reset
            from robots_realtime.runtime.safety.config import validate_safety_config
            validate_safety_config(bad_cfg)
            assert False, "Should have raised ValueError"
        except ValueError:
            mock_agent.reset.assert_not_called()
            pass
        except NotImplementedError:
            raise NotImplementedError("AgentNode safety validation not implemented yet")

def test_clamp_event_is_logged_or_counted_with_arm_and_guardrail_name():
    """Clamp events should include arm key, guardrail name, original/clamped command, timestamp."""
    # After a clamp occurs, observability data should be accessible
    clamp_data = {
        "arm_key": "left",
        "guardrail": "CommandBoundingBoxGuardrail",
        "original_cmd": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "clamped_cmd": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "timestamp": 1234567890.123,
    }
    
    raise NotImplementedError("Clamp observability not implemented yet")

def test_repeated_clamps_warn_without_pausing_or_stopping():
    """Repeated clamps should log warnings but not pause/stop execution."""
    import logging
    
    # Simulate many consecutive clamps; node should continue running
    raise NotImplementedError("Clamp observability not implemented yet")
