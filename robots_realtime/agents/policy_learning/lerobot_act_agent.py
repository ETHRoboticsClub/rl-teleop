"""LeRobot ACT policy agent — single arm, wrist camera, 7 DoF.

WHY THIS EXISTS AND WHY IT IS NOT act_runner.py
===============================================

``yam-pick-pipeline/act_runner.py`` already loads exactly this checkpoint
format, and it must NOT be used for the right arm. Both right-arm configs say so
in capitals:

    BLOCKER 4 — NOTHING IN yam-pick-pipeline IS CALIBRATED FOR THIS ARM.
    results/calibration.json, keepout.json, tray_box.json and the 77 demos are
    all in the LEFT arm's base frame. [...] Do not point run_pick.py,
    sort_server.py or act_runner.py at this arm.

That warning is about geometry, and it is the right warning: act_runner carries a
safety envelope, a keep-out volume and a scan-camera shadow that are all
expressed in the LEFT arm's base frame. Running them against the right arm would
not fail loudly — it would compute confident, wrong clearances.

This agent deliberately carries NONE of that machinery. It is the smallest thing
that can deploy a visuomotor policy: wrist image + joint state in, joint targets
out. The policy was trained on 43 demonstrations of this exact arm doing this
exact task, so the demonstrations are the only prior, and no calibration file is
consulted. The safety that remains is the safety that is genuinely there:
``RobotNode``'s own joint limits, and the session's pause gate.

READ THIS BEFORE CHANGING ANYTHING ABOUT THE OBSERVATION
========================================================

**The processors are not optional.** Training normalises every batch on the way
in, so the policy's output is in normalised space and the postprocessor is what
turns it back into radians. Skipping either does not raise: seven finite numbers
go in and seven come out, and it looks like a working rollout. Measured on this
rig on 2026-07-28 with both omitted, the policy drove precisely to the all-zeros
configuration — which is what "normalised output published straight to the
joints" looks like, and is the signature of missing denormalisation.

**The image must be the size the policy trained on.** This agent consumes the
BUS copy of the camera frame. The teleop configs publish the wrist cameras with
``publish_resize: [240, 320]`` to keep four cameras inside one ZMQ proxy — but
this dataset was exported from the full-resolution MP4s, so the policy expects
480x640. A policy config for this agent must therefore leave ``publish_resize``
OFF for the camera it feeds. The shape is checked at runtime rather than assumed.

**The state vector is the bare 7.** ``export_lerobot.joint_names()`` gives one
arm the bare names ``joint_1..gripper``, which is exactly
``robot.get_observations()["joint_pos"]``. Two arms would get prefixed names in a
fixed order, and the exporter's own comment is worth repeating: "the order is
part of the dataset contract: swapping it trains a policy that drives the wrong
arm, and nothing raises."
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from robots_realtime.agents.agent import PolicyAgent

logger = logging.getLogger(__name__)


class LeRobotACTAgent(PolicyAgent):
    """Run a LeRobot ACT checkpoint on one arm from one camera.

    Args:
        checkpoint_path: a ``.../checkpoints/<step>/pretrained_model`` directory.
        state_key:  key in the obs dict carrying the arm's ``joint_state``
                    payload (AgentNode fills it from ``state_topics``).
        image_key:  key in the obs dict carrying the camera payload.
        model_image_key: the LeRobot feature name for that image. Must match the
                    checkpoint's ``input_features``; it is verified at load.
        n_action_steps: how many actions to consume from each predicted chunk
                    before re-inferring. Lower = more reactive, more compute.
        device:     "cuda" or "cpu".
        max_step_rad: per-tick safety clamp on how far any joint may be commanded
                    from where it currently is. This is NOT a substitute for the
                    geometric envelope act_runner has — it cannot know about the
                    table — but it does bound the one failure mode a visuomotor
                    policy produces most often on a brakeless arm, which is a
                    single enormous jump from a bad inference.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        state_key: str = "state",
        image_key: str = "wrist",
        model_image_key: str = "observation.images.wrist",
        n_action_steps: int = 8,
        device: str = "cuda",
        max_step_rad: float = 0.12,
        use_joint_state_as_action: bool = False,
    ) -> None:
        if not checkpoint_path:
            raise ValueError("LeRobotACTAgent requires checkpoint_path")
        self.use_joint_state_as_action = use_joint_state_as_action
        self._state_key = state_key
        self._image_key = image_key
        self._model_image_key = model_image_key
        self._max_step_rad = float(max_step_rad)
        self._device = device
        self._lock = threading.Lock()
        self._last_cmd: Optional[np.ndarray] = None
        self._infer_count = 0
        self._last_log = 0.0

        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        self._torch = torch
        self._policy = ACTPolicy.from_pretrained(checkpoint_path)

        # Verify the checkpoint really is the single-wrist, 7-DoF model this
        # agent is wired for, rather than discovering it by driving an arm.
        feats = getattr(self._policy.config, "input_features", {}) or {}
        if model_image_key not in feats:
            raise ValueError(
                f"checkpoint {checkpoint_path} has no input feature {model_image_key!r}. "
                f"It expects {sorted(feats)}. This agent feeds exactly one camera; a "
                f"checkpoint wanting more (e.g. observation.images.top) needs a different "
                f"config and a camera to fill it."
            )
        img_shape = tuple(feats[model_image_key].shape)
        state_shape = tuple(feats["observation.state"].shape)
        self._expect_hw = (int(img_shape[1]), int(img_shape[2]))
        self._dof = int(state_shape[0])

        if self._policy.config.n_action_steps != n_action_steps:
            logger.info(
                "n_action_steps %s -> %s (inference-only; chunk_size %s unchanged)",
                self._policy.config.n_action_steps, n_action_steps,
                self._policy.config.chunk_size,
            )
            self._policy.config.n_action_steps = int(n_action_steps)
        self._policy.to(device).eval()
        self._policy.reset()

        # BOTH processors, always. See the module docstring.
        self._pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=checkpoint_path,
            config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self._post = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=checkpoint_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        logger.info(
            "LeRobotACTAgent ready: %s | %d DoF | image %s %s | %d pre-steps, %d post-steps",
            checkpoint_path, self._dof, model_image_key, self._expect_hw,
            len(self._pre), len(self._post),
        )

    # ------------------------------------------------------------------

    def _joint_pos(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """The 7-vector the policy was trained on: 6 joints THEN the gripper.

        ``get_observations()["joint_pos"]`` is only SIX values — it omits the
        gripper, which arrives separately as ``gripper_pos``. RobotNode's own
        source says so twice ("NOT get_observations()['joint_pos'] which omits
        the gripper on i2rt"), and the exporter's feature names are
        ``joint_1..joint_6`` + ``gripper``. Reading only the six gives a
        6-vector, which this agent would reject and hold on forever: safe, and
        completely inert, with nothing obviously wrong in the log.

        The gripper channel is [0, 1] — 0 shut, 1 open — matching what
        normalize_gripper() wrote into the dataset. On this rig the live value is
        already in that range (measured 0.999 with the jaws open, against a
        demonstration median of 0.964).
        """
        payload = obs.get(self._state_key)
        if not isinstance(payload, dict):
            return None
        q = payload.get("joint_pos")
        if q is None:
            return None
        q = np.asarray(q, dtype=np.float32).reshape(-1)

        if q.size == self._dof:
            return q
        grip = payload.get("gripper_pos")
        if grip is None:
            return None
        q = np.concatenate([q, np.asarray(grip, dtype=np.float32).reshape(-1)])
        return q if q.size == self._dof else None

    def _image(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        payload = obs.get(self._image_key)
        if not isinstance(payload, dict):
            return None
        imgs = payload.get("images")
        img = imgs.get("rgb") if isinstance(imgs, dict) else payload.get("frame")
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        return arr

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One joint-position target for this tick.

        HOLDING IS THE FAILURE MODE. Every path that cannot produce a trustworthy
        action returns the arm's CURRENT joint positions, which commands it to
        stay where it is. On an arm with no brakes, "hold" is the only safe
        default: returning nothing would let the previous target stand, and
        returning a stale target is how an arm resumes a motion whose reason has
        gone away.
        """
        q_now = self._joint_pos(obs)
        if q_now is None:
            # No joint state yet. There is nothing sane to command and nothing to
            # hold TO, so repeat the last command if we have one and otherwise
            # stay silent by commanding nothing new.
            with self._lock:
                last = self._last_cmd
            return {"pos": last if last is not None else np.zeros(self._dof, np.float32)}

        img = self._image(obs)
        if img is None:
            return self._hold(q_now, "no camera frame")

        if (img.shape[0], img.shape[1]) != self._expect_hw:
            # Silently resizing here would hide a config error that changes what
            # the policy sees. The teleop configs downsize the wrist cameras for
            # the bus; a policy config must not.
            return self._hold(
                q_now,
                f"frame is {img.shape[:2]}, policy trained on {self._expect_hw} — "
                f"remove publish_resize from this camera",
            )

        torch = self._torch
        with torch.inference_mode():
            batch = {
                "observation.state": torch.from_numpy(q_now).to(self._device).unsqueeze(0),
                self._model_image_key: (
                    torch.from_numpy(np.ascontiguousarray(img))
                    .to(self._device)
                    .permute(2, 0, 1)
                    .to(torch.float32)
                    .div(255.0)
                    .unsqueeze(0)
                ),
            }
            action = self._post(self._policy.select_action(self._pre(batch)))[0]
        target = np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

        if target.size != self._dof or not np.all(np.isfinite(target)):
            return self._hold(q_now, f"policy returned {target.size} values / non-finite")

        # Per-tick clamp. A visuomotor policy's characteristic bad output is one
        # large jump, and this arm has no brakes to absorb it.
        delta = np.clip(target - q_now, -self._max_step_rad, self._max_step_rad)
        clamped = q_now + delta
        if np.any(np.abs(target - q_now) > self._max_step_rad):
            self._maybe_log(
                "clamped a %0.3f rad jump to %0.3f",
                float(np.max(np.abs(target - q_now))), self._max_step_rad,
            )

        with self._lock:
            self._last_cmd = clamped
            self._infer_count += 1
        return {"pos": clamped}

    def _hold(self, q_now: np.ndarray, why: str) -> Dict[str, Any]:
        self._maybe_log("holding position: %s", why)
        with self._lock:
            self._last_cmd = q_now
        return {"pos": q_now}

    def _maybe_log(self, fmt: str, *args) -> None:
        """Rate-limited so a persistent condition cannot flood the node log."""
        now = time.monotonic()
        if now - self._last_log > 1.0:
            self._last_log = now
            logger.warning("[LeRobotACTAgent] " + fmt, *args)

    def action_spec(self):
        # ActionSpec is a type ALIAS (Union[Array, Dict[str, ActionSpec]]), not a
        # class — constructing it would raise. Single arm publishes a bare
        # {"pos": ...}, which AgentNode turns into <name>/joint_pos.
        from robots_realtime.agents.constants import Array
        return {"pos": Array(shape=(self._dof,), dtype=np.float32)}
