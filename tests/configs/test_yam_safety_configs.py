"""YAML config structure tests for safety settings."""
import pytest
import yaml


def _get_policy_node(cfg):
    """Return the policy AgentNode from a bimanual inference config."""
    return next(
        (n for n in cfg["nodes"] if n.get("name") == "policy"), None
    )


def _get_agent_nodes(cfg):
    """Return all AgentNode entries from a config."""
    return [n for n in cfg["nodes"] if n.get("type") == "AgentNode"]


def test_yam_bimanual_inference_config_has_required_safety_structure():
    """Inference config should have safety structure with null placeholders.

    The policy node must contain a 'safety' key with per-arm bounding_box
    sections (left and right). Bounding box values may be null since real
    hardware validation is expected to reject them — but the structural keys
    must exist so they're discoverable before deployment.
    """
    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    policy = _get_policy_node(cfg)
    assert policy is not None, "Policy AgentNode not found"

    safety = policy.get("safety")
    assert safety is not None, "Safety section missing from policy node"

    # Check required top-level safety keys
    assert "mode" in safety, "safety.mode key missing"
    assert "agent_type" in safety, "safety.agent_type key missing"
    assert "arms" in safety, "safety.arms dict missing"

    # Per-arm structure
    arms = safety["arms"]
    assert "left" in arms, "safety.arms.left missing"
    assert "right" in arms, "safety.arms.right missing"

    for arm_name in ("left", "right"):
        arm = arms[arm_name]
        assert "bounding_box" in arm, f"safety.arms.{arm_name}.bounding_box missing"
        assert "position_indices" in arm, \
            f"safety.arms.{arm_name}.position_indices missing"
        assert "gripper_index" in arm, \
            f"safety.arms.{arm_name}.gripper_index missing"

        bbox = arm["bounding_box"]
        assert "min" in bbox, f"safety.arms.{arm_name}.bounding_box.min missing"
        assert "max" in bbox, f"safety.arms.{arm_name}.bounding_box.max missing"

    # acceleration_limit must exist (even if null placeholder)
    assert "acceleration_limit" in safety, \
        "safety.acceleration_limit key missing from inference node"


def test_yam_bimanual_inference_placeholders_fail_closed():
    """Inference config with null placeholders should fail validation until numeric values are filled.

    The safety validator (validate_safety_config) must raise ValueError when
    checking the inference YAML's safety section. This ensures deployment
    fails-closed until site-specific finite numeric values are filled in.
    """
    from robots_realtime.runtime.safety.config import validate_safety_config

    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    policy = _get_policy_node(cfg)
    safety = policy["safety"]

    # Build a flat SafetyConfig-style dict from the per-arm left config
    # (the validator expects mode, agent_type, bounding_box, acceleration_limit)
    left_arm = safety["arms"]["left"]
    safety_dict = {
        "mode": safety["mode"],
        "agent_type": safety["agent_type"],
        "bounding_box": left_arm["bounding_box"],
        "acceleration_limit": safety["acceleration_limit"],
    }

    # Null placeholders in real + inference must trigger ValueError
    with pytest.raises(ValueError):
        validate_safety_config(safety_dict)


def test_yam_sim_gello_teleop_config_has_bbox_but_no_acceleration():
    """Sim teleop config should have bbox but no acceleration limiter.

    Each gello leader AgentNode must define a bounding_box under its safety
    section. Teleop nodes do NOT use acceleration_limit — that constraint
    only applies to inference agents on real hardware.
    """
    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    agent_nodes = _get_agent_nodes(cfg)
    assert len(agent_nodes) >= 2, "Expected at least 2 gello agent nodes"

    for node in agent_nodes:
        safety = node.get("safety")
        assert safety is not None, \
            f"Node '{node['name']}' missing safety section"

        # Must have bounding box with numeric values
        arms = safety.get("arms", {})
        arm_key = node.get("arm_key", node["name"].split("_")[1])
        arm = arms.get(arm_key, list(arms.values())[0])

        bbox = arm.get("bounding_box")
        assert bbox is not None, \
            f"Node '{node['name']}' missing bounding_box"

        # Values must be lists of finite numbers (not None)
        mins = bbox.get("min", [])
        maxs = bbox.get("max", [])
        assert all(v is not None for v in mins), \
            f"Node '{node['name']}' has null values in bounding_box.min"
        assert all(v is not None for v in maxs), \
            f"Node '{node['name']}' has null values in bounding_box.max"

        # Teleop must NOT have acceleration_limit
        assert "acceleration_limit" not in safety, \
            f"Node '{node['name']}' must not include acceleration_limit (teleop)"


def test_yam_safety_config_documents_command_space_not_cartesian():
    """Safety config should document that values are command-space, not Cartesian.

    Both YAML files must contain a comment or annotation clarifying that
    bounding box values represent joint-space / command-space limits rather
    than end-effector Cartesian workspace limits. This prevents operators
    from misinterpreting the constraints.
    """
    # Check inference YAML
    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        raw = f.read()

    lower = raw.lower()
    assert "command" in lower or "joint" in lower, \
        "Inference safety config must mention command/joint-space, not Cartesian"

    # Also verify it mentions the distinction
    assert "cartesian" in lower, \
        "Inference safety config must explicitly reference Cartesian for contrast"

    # Check sim teleop YAML too
    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        sim_raw = f.read()

    sim_lower = sim_raw.lower()
    assert "command" in sim_lower or "joint" in sim_lower, \
        "Sim teleop safety config must mention command/joint-space, not Cartesian"
