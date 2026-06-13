"""obs -> model input adapter.

Implement `transform(obs: dict) -> Any` to convert the AgentNode obs dict
(state_topics + image_topics flattened in) into whatever the loaded .pt
model expects as its first positional argument.

obs schema (from AgentNode at runtime):
    obs["timestamp"]:                float
    obs["left"]["joint_pos"]:        np.ndarray (7,)
    obs["left"]["gripper_pos"]:      np.ndarray (1,)
    obs["right"]["joint_pos"]:       np.ndarray (7,)
    obs["right"]["gripper_pos"]:     np.ndarray (1,)
    obs["top_camera"]["rgb"]:        np.ndarray (H, W, 3) uint8

(intentionally left empty — fill in once the .pt's expected input is known)
"""
