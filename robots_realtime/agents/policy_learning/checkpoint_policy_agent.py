"""Generic checkpoint policy agent.

Pipeline:

    obs --[obs_adapter]--> model_input --[model]--> model_output --[action_adapter]--> action

The model is a `.pt` file loaded with ``torch.load`` (full-pickle, i.e.
``torch.save(model, ...)`` not just a state_dict). Adapters are resolved
from "module:callable" strings supplied via YAML, so the model-specific
preprocessing/postprocessing lives in the top-level ``adapters/`` folder
and the agent itself stays framework-agnostic. (``adapters/`` is not part
of the installed ``robots_realtime`` package; ``infer.sh`` puts the repo
root on ``PYTHONPATH`` so ``import adapters`` resolves.)
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec


def _resolve(path: str) -> Callable:
    """Resolve a "module.path:callable" string."""
    if ":" not in path:
        raise ValueError(
            f"Adapter spec '{path}' must be of the form 'module.path:callable'."
        )
    module_path, attr = path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"Adapter '{path}' references '{attr}', which does not exist in "
            f"'{module_path}'. Implement it in the adapter stub at "
            f"adapters/ (repo root)."
        ) from exc


class CheckpointPolicyAgent(PolicyAgent):
    def __init__(
        self,
        checkpoint_path: str,
        obs_adapter: str = "adapters.obs_to_input:transform",
        action_adapter: str = "adapters.output_to_action:transform",
        device: str = "cuda",
    ) -> None:
        import torch

        self._torch = torch
        self._device = device

        self._model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if hasattr(self._model, "eval"):
            self._model.eval()
        print(f"[CheckpointPolicyAgent] loaded checkpoint from {checkpoint_path}")

        self._obs_adapter = _resolve(obs_adapter)
        self._action_adapter = _resolve(action_adapter)

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = self._obs_adapter(obs)
        if model_input is None:
            return {}
        with self._torch.no_grad():
            model_output = self._model(model_input)
        return self._action_adapter(model_output)

    def action_spec(self) -> ActionSpec:
        return {
            "left":  {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    def close(self) -> None:
        pass
