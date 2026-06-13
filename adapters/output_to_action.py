"""model output -> action adapter.

Implement `transform(model_output: Any) -> dict` to convert the raw model
output into the bimanual action dict that AgentNode fans out to
`policy/left_pos` and `policy/right_pos`:

    {
        "left":  {"pos": np.ndarray (7,) float32},
        "right": {"pos": np.ndarray (7,) float32},
    }

Each `pos` is [j1..j6, gripper] in radians (gripper in [0, 1]).

(intentionally left empty — fill in once the .pt's output shape is known)
"""
