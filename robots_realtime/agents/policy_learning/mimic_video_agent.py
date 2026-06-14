"""Mimic Video (Cosmos Predict 2 VAM) policy agent for the bimanual YAM.

Pipeline:

    N frames -> Video2WorldPipeline (denoise to tau steps)
              -> cross-attn embedding @ tau
              -> World2ActionPipeline (action DiT)
              -> action chunk (1, H, A)
              -> execute first `num_execute_actions` of H, refill on demand

Mirrors mimic-video/eval/libero/run.py::VAMInference, adapted for the YAM
bimanual obs/action schema.

The action decoder for YAM is not yet trained. `_convert_action` is a
clearly marked TODO that assumes the eventual head will emit
``(num_execute_actions, 14)`` absolute joint positions
(left7 + right7, gripper last in each arm, clipped to [0, 1]).
"""

from __future__ import annotations

import json
import pathlib
from collections import deque
from typing import Any, Deque, Dict

import numpy as np
from dm_env.specs import Array
from einops import rearrange

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec


_DTYPE_BY_NAME = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


class MimicVideoAgent(PolicyAgent):
    def __init__(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: str,
        prompt: str,
        img_horizon: int = 5,
        lowdim_horizon: int = 1,
        num_sampling_step: int = 35,
        stop_video_denoising_step: int = 20,
        num_execute_actions: int = 5,
        device: str = "cuda",
        dtype: str = "bfloat16",
        seed: int = 0,
        use_cuda_graphs: bool = True,
    ) -> None:
        import torch

        self._torch = torch
        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {list(_DTYPE_BY_NAME)}")
        self._dtype = getattr(torch, dtype)
        self._device = device
        self._prompt = prompt
        self._img_horizon = int(img_horizon)
        self._lowdim_horizon = int(lowdim_horizon)
        self._num_sampling_step = int(num_sampling_step)
        self._stop_after_step = int(stop_video_denoising_step)
        self._num_execute_actions = int(num_execute_actions)
        self._seed = int(seed)
        self._use_cuda_graphs = bool(use_cuda_graphs)

        if not (0 <= self._stop_after_step <= self._num_sampling_step):
            raise ValueError(
                f"stop_video_denoising_step={self._stop_after_step} must be in "
                f"[0, num_sampling_step={self._num_sampling_step}]"
            )

        self._model = self._load_pipeline(
            experiment_name=experiment_name,
            video_model_path=video_model_path,
            action_model_path=action_model_path,
            dataset_statistics_path=pathlib.Path(dataset_statistics_path),
        )

        # Lazy-pad history buffers — mirrors VAMInference.reset.
        self._image_history: Deque[np.ndarray] = deque(maxlen=(self._img_horizon - 1) * 4 + 1)
        self._lowdim_history: Deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self._action_buffer: np.ndarray | None = None
        self._action_buffer_idx = 0
        self._stub_warned = False

    def _load_pipeline(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: pathlib.Path,
    ):
        # Imports deferred: cosmos_predict2 / imaginaire are only on PYTHONPATH
        # at runtime (inference.sh extends PYTHONPATH with mimic-video/model).
        from cosmos_predict2.configs.config import make_config
        from cosmos_predict2.data.action.utils import extract_normalization_types
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
        from cosmos_predict2.pipelines.world2action import World2ActionPipeline
        from imaginaire.lazy_config import instantiate
        from imaginaire.utils.config_helper import override

        config = make_config()
        config = override(config, ["--", f"experiment={experiment_name}"])
        config.model.config.video_pipe_config.guardrail_config.enabled = False

        video_pipe = Video2WorldPipeline.from_config(
            config=config.model.config.video_pipe_config,
            dit_path=video_model_path,
            device=self._device,
            torch_dtype=self._dtype,
            load_ema_to_reg=False,
        )
        action_pipe = World2ActionPipeline.from_config(
            config.model.config.pipe_config,
            dit_path=action_model_path,
            device=self._device,
            dtype=self._dtype,
        )

        data_config = instantiate(config.data_config)
        with dataset_statistics_path.open("rb") as f:
            stats = json.load(f)
        action_pipe.normalizer.build_from_stats(
            stats,
            normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
            concat_groups=data_config.policy_io.concat_groups,
            device=self._device,
            dtype=self._dtype,
        )

        print(
            f"[MimicVideoAgent] loaded video backbone={video_model_path} "
            f"action decoder={action_model_path} experiment={experiment_name} "
            f"tau={self._stop_after_step}/{self._num_sampling_step}"
        )
        return Video2World2ActionPipeline(video_pipe, action_pipe).cuda()

    def reset(self) -> None:
        self._image_history.clear()
        self._lowdim_history.clear()
        self._action_buffer = None
        self._action_buffer_idx = 0

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rgb = obs.get("top_camera", {}).get("images", {}).get("rgb")
        if rgb is None:
            return {}
        try:
            state = self._state_from_obs(obs)
        except KeyError:
            return {}

        self._add_image(self._process_image(np.asarray(rgb)))
        self._add_lowdim(state)

        if self._action_buffer is None or self._action_buffer_idx >= self._num_execute_actions:
            self._query_policy()

        assert self._action_buffer is not None  # set by _query_policy above
        step = self._action_buffer[self._action_buffer_idx]
        self._action_buffer_idx += 1
        return self._convert_action(step)

    def _query_policy(self) -> None:
        torch = self._torch
        # Downsample 20fps → 5fps by taking every 4th frame; matches LIBERO eval.
        images = np.concatenate(list(self._image_history)[::4], axis=1)
        lowdims = np.stack(list(self._lowdim_history), axis=0)

        input_vid = torch.from_numpy(images[None]).to(device=self._device, dtype=self._dtype)
        state_tensor = torch.from_numpy(lowdims[None]).to(device=self._device, dtype=self._dtype)

        with torch.no_grad():
            pred = self._model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=self._prompt,
                num_sampling_step=self._num_sampling_step,
                stop_after_step=self._stop_after_step,
                seed=self._seed,
                use_cuda_graphs=self._use_cuda_graphs,
            )
        self._action_buffer = pred[0].float().cpu().numpy()
        self._action_buffer_idx = 0

    @staticmethod
    def _process_image(rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        tensor = rearrange(rgb, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image(self, image: np.ndarray) -> None:
        cap = self._image_history.maxlen or 0
        self._image_history.append(image)
        while len(self._image_history) < cap:
            self._image_history.append(image.copy())

    def _add_lowdim(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())

    @staticmethod
    def _state_from_obs(obs: Dict[str, Any]) -> np.ndarray:
        left = obs["left"]
        right = obs["right"]
        return np.concatenate(
            [
                np.asarray(left["joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(left["gripper_pos"], dtype=np.float32).reshape(-1),
                np.asarray(right["joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(right["gripper_pos"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

    def _convert_action(self, action: np.ndarray) -> Dict[str, Any]:
        # TODO(action-head): swap once the YAM bimanual action decoder is
        # trained. Until then we assume the trained head will emit a 14-dim
        # absolute joint position vector per step: [left7, right7], gripper
        # last in each arm clipped to [0, 1].
        if action.shape[-1] < 14:
            if not self._stub_warned:
                print(
                    f"[MimicVideoAgent] WARNING: action decoder emitted "
                    f"{action.shape[-1]}-dim output (<14). Padding with zeros — "
                    f"this is a placeholder until the YAM action head is trained."
                )
                self._stub_warned = True
            padded = np.zeros(14, dtype=np.float32)
            padded[: action.shape[-1]] = action
            action = padded
        else:
            action = action[:14].astype(np.float32, copy=False)

        left = action[:7].copy()
        right = action[7:14].copy()
        left[-1] = float(np.clip(left[-1], 0.0, 1.0))
        right[-1] = float(np.clip(right[-1], 0.0, 1.0))
        return {"left": {"pos": left}, "right": {"pos": right}}

    def action_spec(self) -> ActionSpec:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    def close(self) -> None:
        pass
