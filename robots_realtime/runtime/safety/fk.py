"""Forward-kinematics provider for the Cartesian workspace guardrail.

Builds a lightweight ``fk(arm_joints) -> xyz`` callable from a MuJoCo model by writing
the arm's joint values into the named joints and reading a site position. Uses plain
MuJoCo (no IK / mink), so it is cheap enough for the teleop loop and has no extra deps
beyond mujoco itself.
"""

from __future__ import annotations

import numpy as np


def make_mujoco_fk(xml_path: str, joint_names: list[str], site_name: str):
    """Return ``fk(arm_joints) -> np.ndarray(3,)`` for ``site_name`` in ``xml_path``.

    ``joint_names`` are the arm's joints in command order; their MuJoCo qpos addresses
    are resolved once. The returned callable is stateful (reuses one MjData) and is not
    thread-safe — instantiate one per guardrail.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    qadr = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"FK model {xml_path!r} has no joint {jn!r}")
        qadr.append(int(model.jnt_qposadr[jid]))

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"FK model {xml_path!r} has no site {site_name!r}")

    qadr_arr = np.asarray(qadr, dtype=int)

    def fk(arm_joints: np.ndarray) -> np.ndarray:
        q = np.asarray(arm_joints, dtype=np.float64)
        data.qpos[qadr_arr] = q[: qadr_arr.size]
        mujoco.mj_kinematics(model, data)
        return np.array(data.site_xpos[sid], dtype=np.float64)

    return fk
