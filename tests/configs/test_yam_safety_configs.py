"""YAML config structure tests for safety settings."""
import pytest
import yaml


def test_yam_bimanual_inference_config_has_required_safety_structure():
    """Inference config should have safety structure (even with null placeholders)."""
    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # The YAML should have a safety section (or arm-level safety keys)
    # Even if values are null/placeholder, the structure must exist
    raise NotImplementedError("Safety config in YAML not added yet")


def test_yam_bimanual_inference_placeholders_fail_closed():
    """Inference config with null placeholders should fail validation until numeric values are filled."""
    import yaml
    from robots_realtime.runtime.safety.config import validate_safety_config

    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Null placeholders should cause validation to reject the config
    with pytest.raises(ValueError):
        validate_safety_config(cfg)

    raise NotImplementedError("Safety config validation not implemented yet")


def test_yam_sim_gello_teleop_config_has_bbox_but_no_acceleration():
    """Sim teleop config should have bbox but no acceleration limiter."""
    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Teleop configs have bounding_box but NOT acceleration_limit
    raise NotImplementedError("Safety config in YAML not added yet")


def test_yam_safety_config_documents_command_space_not_cartesian():
    """Safety config should document that values are command-space, not Cartesian."""
    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        raw = f.read()

    # Check for a comment or annotation clarifying coordinate space
    assert "command" in raw.lower() or "joint" in raw.lower(), \
        "Safety config should mention these are command/joint-space values, not Cartesian"

    raise NotImplementedError("Safety config documentation not added yet")
