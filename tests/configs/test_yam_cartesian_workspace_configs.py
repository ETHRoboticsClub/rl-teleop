"""YAML config structure tests for cartesian workspace settings."""
import pytest
import yaml


def _get_agent_nodes(cfg):
    """Return all AgentNode entries from a config."""
    return [n for n in cfg["nodes"] if n.get("type") == "AgentNode"]


def _get_policy_node(cfg):
    """Return the policy AgentNode from a bimanual inference config."""
    return next(
        (n for n in cfg["nodes"] if n.get("name") == "policy"), None
    )


def test_sim_teleop_cartesian_workspace_config_validates():
    """Sim teleop config should have cartesian_workspace sections that validate."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    agent_nodes = _get_agent_nodes(cfg)
    assert len(agent_nodes) >= 2, "Expected at least 2 gello agent nodes"

    for node in agent_nodes:
        safety = node.get("safety")
        assert safety is not None, f"Node '{node['name']}' missing safety section"

        arms = safety.get("arms", {})
        arm_key = node.get("arm_key", node["name"].split("_")[1])
        arm = arms.get(arm_key, list(arms.values())[0])

        cw = arm.get("cartesian_workspace")
        assert cw is not None, f"Node '{node['name']}' missing cartesian_workspace"

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


def test_real_cartesian_placeholders_fail_closed():
    """Real config placeholders should fail validation until numeric values are filled."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    with open("configs/yam/yam_bimanual_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    policy = _get_policy_node(cfg)
    assert policy is not None, "Policy AgentNode not found"

    safety = policy.get("safety")
    assert safety is not None, "Safety section missing from policy node"

    arms = safety.get("arms", {})
    for arm_name in ("left", "right"):
        arm = arms.get(arm_name)
        if arm is None:
            continue

        cw = arm.get("cartesian_workspace")
        if cw is None:
            continue

        # Build config dict - null values should fail validation
        cw_cfg = {
            "agent_type": safety["agent_type"],
            "site_name": cw.get("site_name"),
            "xml_path": cw.get("xml_path"),
            "frame": cw.get("frame"),
            "min_xyz": cw.get("min_xyz"),
            "max_xyz": cw.get("max_xyz"),
        }

        # Null placeholders must trigger ValueError
        with pytest.raises(ValueError):
            validate_cartesian_workspace_config(cw_cfg)


def test_cartesian_workspace_documents_model_frame():
    """Cartesian workspace config should document that values are model-frame Cartesian meters."""
    with open("configs/yam/yam_sim_gello_teleop.yaml", "r") as f:
        raw = f.read()

    lower = raw.lower()
    assert "cartesian" in lower, "Sim teleop must mention Cartesian workspace"
    assert "model" in lower or "frame" in lower, "Must document model frame"
