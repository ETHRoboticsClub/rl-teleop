"""YAML config structure tests for safety settings — Cartesian workspace only."""
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

    The policy node must contain a 'safety' key with per-arm cartesian_workspace
    sections (left and right). Cartesian workspace values may be null since real
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
        assert "cartesian_workspace" in arm, \
            f"safety.arms.{arm_name}.cartesian_workspace missing"
        assert "position_indices" in arm, \
            f"safety.arms.{arm_name}.position_indices missing"
        assert "gripper_index" in arm, \
            f"safety.arms.{arm_name}.gripper_index missing"

        cw = arm["cartesian_workspace"]
        assert "enabled" in cw, f"safety.arms.{arm_name}.cartesian_workspace.enabled missing"
        assert "min_xyz" in cw, f"safety.arms.{arm_name}.cartesian_workspace.min_xyz missing"
        assert "max_xyz" in cw, f"safety.arms.{arm_name}.cartesian_workspace.max_xyz missing"

    # acceleration_limit must exist (even if null placeholder)
    assert "acceleration_limit" in safety, \
        "safety.acceleration_limit key missing from inference node"


def test_yam_bimanual_inference_placeholders_fail_closed():
    """Inference config with null placeholders should fail validation until numeric values are filled.

    The cartesian workspace validator must raise ValueError when
    checking the inference YAML's safety section. This ensures deployment
    fails-closed until site-specific finite numeric values are filled in.
    """
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    policy = _get_policy_node(cfg)
    safety = policy["safety"]

    # Build a cartesian workspace config dict from the per-arm left config
    left_arm = safety["arms"]["left"]
    cw = left_arm["cartesian_workspace"]
    cw_cfg = {
        "agent_type": safety["agent_type"],
        "site_name": cw.get("site_name"),
        "xml_path": cw.get("xml_path"),
        "frame": cw.get("frame"),
        "min_xyz": cw.get("min_xyz"),
        "max_xyz": cw.get("max_xyz"),
    }

    # Null placeholders in real mode must trigger ValueError
    with pytest.raises(ValueError):
        validate_cartesian_workspace_config(cw_cfg)


def test_yam_sim_gello_teleop_config_has_cartesian_workspace():
    """Sim teleop config should have cartesian workspace with numeric values.

    Each gello leader AgentNode must define a cartesian_workspace under its safety
    section. Values must be finite numbers (not None) for sim deployment.
    """
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    agent_nodes = _get_agent_nodes(cfg)
    assert len(agent_nodes) >= 2, "Expected at least 2 gello agent nodes"

    for node in agent_nodes:
        safety = node.get("safety")
        assert safety is not None, \
            f"Node '{node['name']}' missing safety section"

        # Must have cartesian workspace with numeric values
        arms = safety.get("arms", {})
        arm_key = node.get("arm_key", node["name"].split("_")[1])
        arm = arms.get(arm_key, list(arms.values())[0])

        cw = arm.get("cartesian_workspace")
        assert cw is not None, \
            f"Node '{node['name']}' missing cartesian_workspace"

        # Values must be lists of finite numbers (not None)
        mins = cw.get("min_xyz", [])
        maxs = cw.get("max_xyz", [])
        assert all(v is not None for v in mins), \
            f"Node '{node['name']}' has null values in cartesian_workspace.min_xyz"
        assert all(v is not None for v in maxs), \
            f"Node '{node['name']}' has null values in cartesian_workspace.max_xyz"

        # Validate the cartesian workspace config
        cw_cfg = {
            "agent_type": safety["agent_type"],
            "site_name": cw["site_name"],
            "xml_path": cw["xml_path"],
            "frame": cw["frame"],
            "min_xyz": cw["min_xyz"],
            "max_xyz": cw["max_xyz"],
        }
        if "enforcement" in cw:
            cw_cfg["enforcement"] = cw["enforcement"]

        assert validate_cartesian_workspace_config(cw_cfg) is True

        # Teleop must NOT have acceleration_limit
        assert "acceleration_limit" not in safety, \
            f"Node '{node['name']}' must not include acceleration_limit (teleop)"


def test_yam_safety_config_documents_cartesian_workspace():
    """Safety config should document that values are model-frame Cartesian meters.

    Both YAML files must contain a comment or annotation clarifying that
    cartesian workspace values represent model-frame Cartesian meters for the
    TCP site rather than joint-space limits. This prevents operators
    from misinterpreting the constraints.
    """
    # Check inference YAML
    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        raw = f.read()

    lower = raw.lower()
    assert "cartesian" in lower, \
        "Inference safety config must mention Cartesian workspace"
    assert "model" in lower or "frame" in lower, \
        "Inference safety config must document model frame"

    # Check sim teleop YAML too
    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        sim_raw = f.read()

    sim_lower = sim_raw.lower()
    assert "cartesian" in sim_lower, \
        "Sim teleop safety config must mention Cartesian workspace"
