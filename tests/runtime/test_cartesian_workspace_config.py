"""RED tests for cartesian workspace config validation — implementation does NOT exist yet."""

import math
import pytest


def _cartesian_cfg(**overrides):
    """Build a valid cartesian workspace config base for teleop."""
    cfg = {
        "agent_type": "teleop",
        "site_name": "wrist",
        "xml_path": "models/franka/panda.xml",
        "frame": "model",
        "min_xyz": [-0.5, -0.5, -0.5],
        "max_xyz": [0.5, 0.5, 0.5],
    }
    cfg.update(overrides)
    return cfg


# ── Valid config ──────────────────────────────────────────────────────────


def test_valid_teleop_reject_hold_config_passes():
    """Valid teleop config should validate successfully."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg()
    assert validate_cartesian_workspace_config(cfg) is True


# ── fk_ik_clamp mode not allowed for teleop ───────────────────────────────


def test_teleop_rejects_fk_ik_clamp_enforcement():
    """fk_ik_clamp enforcement mode should raise ValueError for teleop agent_type."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg(enforcement="fk_ik_clamp")
    with pytest.raises(ValueError, match="[Cc]lamp|[Ff]k|[Ii]k|teleop"):
        validate_cartesian_workspace_config(cfg)


# ── Required field validations ────────────────────────────────────────────


def test_missing_site_name_raises():
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg()
    del cfg["site_name"]
    with pytest.raises(ValueError, match="[Ss]ite|[Mm]issing"):
        validate_cartesian_workspace_config(cfg)


def test_missing_xml_path_raises():
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg()
    del cfg["xml_path"]
    with pytest.raises(ValueError, match="[Xx]ml|[Mm]issing|[Pp]ath"):
        validate_cartesian_workspace_config(cfg)


# ── Frame validation ──────────────────────────────────────────────────────


def test_unsupported_frame_raises():
    """Only 'model' frame is supported; other values raise."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg(frame="world")
    with pytest.raises(ValueError, match="[Ff]rame|[Ss]upported|[Mm]odel"):
        validate_cartesian_workspace_config(cfg)


# ── NaN in bounds ─────────────────────────────────────────────────────────


def test_nan_xyz_raises():
    """NaN in min_xyz or max_xyz raises ValueError."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    nan_min = _cartesian_cfg(min_xyz=[float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="[Nn]aN|[Vv]alid"):
        validate_cartesian_workspace_config(nan_min)

    nan_max = _cartesian_cfg(max_xyz=[0.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="[Nn]aN|[Vv]alid"):
        validate_cartesian_workspace_config(nan_max)


# ── Inf in bounds ─────────────────────────────────────────────────────────


def test_inf_xyz_raises():
    """Inf in min_xyz or max_xyz raises ValueError."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    inf_min = _cartesian_cfg(min_xyz=[float("inf"), 0.0, 0.0])
    with pytest.raises(ValueError, match="[Ii]nf|[Vv]alid"):
        validate_cartesian_workspace_config(inf_min)

    neg_inf_max = _cartesian_cfg(max_xyz=[0.0, float("-inf"), 0.0])
    with pytest.raises(ValueError, match="[Ii]nf|[Vv]alid"):
        validate_cartesian_workspace_config(neg_inf_max)


# ── Min > Max ─────────────────────────────────────────────────────────────


def test_min_exceeds_max_raises():
    """If any min_xyz[i] > max_xyz[i], raises ValueError."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg(min_xyz=[1.0, 0.0, 0.0], max_xyz=[0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="[Mm]in.*[Mm]ax|[Ee]xceed|[Bb]ound"):
        validate_cartesian_workspace_config(cfg)


# ── Empty arrays ──────────────────────────────────────────────────────────


def test_empty_min_xyz_raises():
    """Empty min_xyz list raises ValueError."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg(min_xyz=[], max_xyz=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="[Ee]mpty|[Mm]in|[Ll]en"):
        validate_cartesian_workspace_config(cfg)


# ── Length mismatch ───────────────────────────────────────────────────────


def test_length_mismatch_raises():
    """min_xyz and max_xyz of different lengths raises ValueError."""
    from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config

    cfg = _cartesian_cfg(min_xyz=[0.0, 0.0], max_xyz=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="[Ll]en|[Mm]ismatch|[Dd]iffer"):
        validate_cartesian_workspace_config(cfg)
