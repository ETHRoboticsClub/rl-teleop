"""Inference adapter for a video-action policy on the bimanual YAM.

Wired into ``CheckpointPolicyAgent`` (see
``robots_realtime/agents/policy_learning/checkpoint_policy_agent.py``)
via two module-level callables:

    obs --[obs_to_input]--> model_input --[model.pt]--> model_output --[output_to_action]--> action

The model:
  * input  : video clip ``(1, N_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)`` float32
             from the head (D405) RGB stream.
  * output : action chunk ``(1, T, 14)`` of absolute joint positions,
             7 dof per arm (last dof = gripper in [0, 1]).

State (frame ring buffer + chunk dequeue) lives at module scope because
the agent calls plain functions, not methods.
"""

from __future__ import annotations

import collections
from typing import Any, Deque, Dict, Optional

import numpy as np
import torch

from robots_realtime.agents.policy_learning.async_act_agent import _center_crop_and_resize

# --- Tunables -------------------------------------------------------------
N_FRAMES = 8
FRAME_SIZE = 224
DEVICE = "cuda"

# --- Module state ---------------------------------------------------------
_frame_buffer: Optional[Deque[np.ndarray]] = None
_chunk: Optional[np.ndarray] = None
_chunk_idx: int = 0


def reset() -> None:
    global _frame_buffer, _chunk, _chunk_idx
    _frame_buffer = None
    _chunk = None
    _chunk_idx = 0


def _preprocess_frame(rgb: np.ndarray) -> np.ndarray:
    cropped = _center_crop_and_resize(rgb, FRAME_SIZE, FRAME_SIZE)
    f = cropped.astype(np.float32) / 255.0
    return np.transpose(f, (2, 0, 1))


def obs_to_input(obs: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Build a ``(1, N_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)`` video clip.

    Returns None while the camera producer is still warming up.
    """
    global _frame_buffer

    cam = obs.get("top_camera")
    if cam is None:
        return None
    rgb = cam.get("images", {}).get("rgb")
    if rgb is None:
        return None

    frame = _preprocess_frame(np.asarray(rgb))

    if _frame_buffer is None:
        _frame_buffer = collections.deque(
            (frame.copy() for _ in range(N_FRAMES)), maxlen=N_FRAMES
        )
    else:
        _frame_buffer.append(frame)

    clip = np.stack(list(_frame_buffer), axis=0)
    return torch.from_numpy(clip).unsqueeze(0).to(DEVICE)


def _as_chunk_array(model_output: Any) -> np.ndarray:
    if isinstance(model_output, dict):
        model_output = model_output.get("action", model_output)
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if isinstance(model_output, torch.Tensor):
        arr = model_output.detach().cpu().numpy()
    else:
        arr = np.asarray(model_output)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 14:
        raise ValueError(
            f"Expected model output shape (T, 14) or (1, T, 14); got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def output_to_action(model_output: Any) -> Dict[str, Any]:
    """Dequeue one step (left7, right7) from the current action chunk.

    Refills the chunk when the previous one is exhausted; otherwise the
    freshly-computed chunk is dropped (typical ACT-style consumption).
    """
    global _chunk, _chunk_idx

    new_chunk = _as_chunk_array(model_output)
    if _chunk is None or _chunk_idx >= len(_chunk):
        _chunk = new_chunk
        _chunk_idx = 0

    step = _chunk[_chunk_idx]
    _chunk_idx += 1

    left = step[:7].copy()
    right = step[7:].copy()
    left[-1] = float(np.clip(left[-1], 0.0, 1.0))
    right[-1] = float(np.clip(right[-1], 0.0, 1.0))

    remaining = _chunk[_chunk_idx:]
    chunk_preview = None
    if remaining.shape[0] > 0:
        chunk_preview = {
            "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
            "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
        }

    action: Dict[str, Any] = {
        "left": {"pos": left.astype(np.float32)},
        "right": {"pos": right.astype(np.float32)},
    }
    if chunk_preview is not None:
        action["_chunk"] = chunk_preview
    return action
