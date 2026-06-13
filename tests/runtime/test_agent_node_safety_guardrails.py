"""Verify AgentNode applies safety guardrails correctly per session mode."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

def test_teleop_applies_bbox_but_not_acceleration():
    """Teleop should only apply bbox, not acceleration limiting."""
    from robots_realtime.runtime.safety.guardrails import CommandBoundingBoxGuardrail, InferenceAccelerationGuardrail
    from robots_realtime.runtime.agent_node import AgentNode
    
    # Create mock agent node in teleop mode with a large command jump
    with patch.object(AgentNode, '__init__', return_value=None):
        node = AgentNode()
    
    raise NotImplementedError("AgentNode safety integration not implemented yet")

def test_inference_applies_bbox_and_acceleration():
    """Inference should apply both bbox and acceleration limiting."""
    from robots_realtime.runtime.safety.guardrails import CommandBoundingBoxGuardrail, InferenceAccelerationGuardrail
    from robots_realtime.runtime.agent_node import AgentNode
    
    with patch.object(AgentNode, '__init__', return_value=None):
        node = AgentNode()
    
    raise NotImplementedError("AgentNode safety integration not implemented yet")

def test_arm_key_path_applies_matching_arm_safety_config():
    """When arm_key is set, safety config for that arm should be applied."""
    from robots_realtime.runtime.safety.config import SafetyConfig
    from robots_realtime.runtime.agent_node import AgentNode
    
    # Bimanual setup: left and right arms each have their own bbox
    left_bbox = {
        "min_vals": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "max_vals": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    right_bbox = {
        "min_vals": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        "max_vals": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
    safety_config = SafetyConfig(arms={"left": left_bbox, "right": right_bbox})
    
    # When processing output for arm_key="right", right's bbox should apply
    cmd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    
    raise NotImplementedError("AgentNode safety integration not implemented yet")

def test_metadata_keys_record_chunk_images_are_not_guardrailed():
    """Metadata keys starting with _ should not be guardrailed."""
    from robots_realtime.runtime.agent_node import AgentNode
    
    # Output dict contains commands and metadata/image keys
    raw_output = {
        "pos": {"left": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                "right": np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])},
        "_chunk_images": {"cam_top": "base64image_data"},
        "_record_timestamp": 12345.678,
    }
    
    # Guardrails should only apply to 'pos', skip '_chunk_images' and '_record_timestamp'
    raise NotImplementedError("AgentNode safety integration not implemented yet")
