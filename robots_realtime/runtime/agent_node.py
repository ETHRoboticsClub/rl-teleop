"""AgentNode — node wrapper for any Agent.

Bridges the Agent protocol (act / reset / close) onto the ZMQ bus.

Supports three loop modes:
    flat_out          — runs as fast as possible; use for hardware-paced agents
                        (e.g. GelloLeaderAgent paced by serial read)
    fixed_rate        — polls at poll_freq Hz (e.g. viser IK at 100 Hz)
    subscriber_driven — blocks on incoming observations; use for learned policies

Agent can be injected at construction (programmatic usage) or instantiated
from a dotted class path in YAML (via agent_class / agent_kwargs).

Action format (returned by agent.act()):
    {"pos": array}                         — single arm, published as joint_pos
    {"left": {"pos": array}, ...}          — multi arm; single non-_ key → joint_pos,
                                             multiple keys → {key}_pos each
    arm_key set                            — extract action[arm_key]["pos"] → joint_pos
    "_record" key                          — forwarded as record signal

Published topics:
    ``{name}/joint_pos``     — single-arm command
    ``{name}/{key}_pos``     — per-arm commands (multi-arm policies)
    ``{name}/record``        — forwarded record signal

Subscribed topics: state_topics.values() + image_topics.values()
"""

from __future__ import annotations

import importlib
import logging
import time

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole

logger = logging.getLogger(__name__)


class AgentNode(Node):
    """Wraps any Agent and bridges it onto the ZMQ bus.

    Args:
        agent:           Pre-built agent. If None, built from agent_class in setup().
        name:            Node name on the bus.
        agent_class:     Dotted import path, e.g.
                         "robots_realtime.agents.teleoperation.gello_leader_agent:GelloLeaderAgent".
                         Used when loading from YAML.
        agent_kwargs:    Keyword arguments forwarded to agent_class().
        loop_mode:       "flat_out" | "fixed_rate" | "subscriber_driven".
        poll_freq:       Hz for fixed_rate mode (or timeout in subscriber_driven).
        publish_freq:    Optional ZMQ send rate cap (Hz).
        state_topics:    {obs_key: bus_topic} — joint state inputs.
        image_topics:    {obs_key: bus_topic} — image inputs.
        arm_key:         If set, extract action[arm_key]["pos"] and publish as joint_pos.
                         Useful for agents that always return a multi-arm dict but are
                         deployed per-arm (e.g. GelloLeaderAgent).
        normalize_gripper: If True, map the last element of pos from raw degrees to [0,1].
        gripper_open_deg:  Raw degrees corresponding to gripper fully open (1.0).
        gripper_closed_deg: Raw degrees corresponding to gripper fully closed (0.0).
        writer:          Optional Writer injected at construction for recording.
    """

    role = NodeRole.CONTROLLER
    published_topics: list[str] = ["joint_pos", "record"]

    def __init__(
        self,
        agent=None,
        name: str = "agent",
        agent_class: str | None = None,
        agent_kwargs: dict | None = None,
        loop_mode: str = "subscriber_driven",
        poll_freq: float | None = None,
        publish_freq: float | None = None,
        state_topics: dict[str, str] | None = None,
        image_topics: dict[str, str] | None = None,
        arm_key: str | None = None,
        normalize_gripper: bool = False,
        gripper_open_deg: float = 85.0,
        gripper_closed_deg: float = 5.0,
        safety: dict | None = None,
        writer=None,
        **kwargs,
    ) -> None:
        self._state_topics = state_topics or {}
        self._image_topics = image_topics or {}
        self.subscribed_topics = (
            list(self._state_topics.values()) + list(self._image_topics.values())
        )

        if loop_mode == "subscriber_driven":
            self.subscriber_driven = True
            self.poll_freq = poll_freq
        elif loop_mode == "fixed_rate":
            self.subscriber_driven = False
            self.poll_freq = poll_freq
        elif loop_mode == "flat_out":
            self.subscriber_driven = False
            self.poll_freq = None
        else:
            raise ValueError(f"Unknown loop_mode: {loop_mode!r}")

        if publish_freq is not None:
            self.publish_freq = publish_freq

        super().__init__(name=name, writer=writer, **kwargs)

        self._agent = agent
        self._agent_class = agent_class
        self._agent_kwargs = agent_kwargs or {}
        self._arm_key = arm_key
        self._normalize_gripper = normalize_gripper
        self._gripper_open_deg = gripper_open_deg
        self._gripper_closed_deg = gripper_closed_deg
        # Command-space safety guardrails (bounding box + speed cap). Parsed/validated in
        # setup() so a bad real-hardware config fails closed *before* any command is sent.
        self._safety_params = safety
        self._guardrails: dict = {}
        self._clamp_counts: dict = {}
        self._last_clamp_log = 0.0

    # ------------------------------------------------------------------

    def setup(self) -> None:
        # Build + validate safety guardrails first so an invalid real-hardware config
        # fails closed before the agent resets or any command is published.
        self._build_guardrails()
        if self._agent is None:
            if self._agent_class is None:
                raise RuntimeError(
                    f"[{self.name}] AgentNode requires 'agent' or 'agent_class'"
                )
            self._agent = self._build_agent()
        if hasattr(self._agent, "reset"):
            self._agent.reset()

    def _build_guardrails(self) -> None:
        from robots_realtime.runtime.safety import (
            BoundingBoxGuardrail,
            SpeedLimitGuardrail,
            build_safety_config,
        )

        cfg = build_safety_config(self._safety_params)
        self._guardrails = {}
        if cfg is None:
            return
        for arm_key, arm in cfg.arms.items():
            chain = []
            # Order: speed cap first, Cartesian workspace reject next, bounding box last,
            # so the final published command is inside every limit.
            if arm.speed_limit is not None:
                chain.append(SpeedLimitGuardrail(arm.speed_limit, arm_key))
            if arm.cartesian is not None:
                chain.append(self._build_cartesian_guardrail(arm.cartesian, arm_key))
            if arm.bounding_box is not None:
                chain.append(BoundingBoxGuardrail(arm.bounding_box, arm_key))
            if chain:
                self._guardrails[arm_key] = chain

    def _build_cartesian_guardrail(self, cartesian: dict, arm_key: str):
        """Parse the cartesian_workspace dict and build its FK-backed guardrail.

        Parsing happens here (not in safety.config) so the mujoco FK dependency stays out
        of the config module and validation still runs in setup(), before actuation.
        """
        from robots_realtime.runtime.safety.cartesian import (
            CartesianConfigError,
            CartesianWorkspaceConfig,
            CartesianWorkspaceRejectGuardrail,
        )
        from robots_realtime.runtime.safety.fk import make_mujoco_fk

        ws = CartesianWorkspaceConfig.from_dict(cartesian, arm_key)
        for key in ("fk_xml_path", "fk_joint_names", "fk_site_name"):
            if key not in cartesian:
                raise CartesianConfigError(
                    f"[{arm_key}] cartesian_workspace: requires '{key}' to build FK"
                )
        fk = make_mujoco_fk(
            cartesian["fk_xml_path"],
            list(cartesian["fk_joint_names"]),
            cartesian["fk_site_name"],
        )
        return CartesianWorkspaceRejectGuardrail(ws, fk, arm_key)

    def _build_agent(self):
        ref = self._agent_class
        if ":" not in ref:
            raise ValueError(
                f"agent_class must be 'module.path:ClassName', got {ref!r}"
            )
        module_path, cls_name = ref.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)(**self._agent_kwargs)

    def step(self) -> None:
        obs: dict = {"timestamp": time.time(), "_paused": self._paused}
        for obs_key, topic in self._state_topics.items():
            data = self.get_latest(topic)
            if data is not None:
                obs[obs_key] = data
        for obs_key, topic in self._image_topics.items():
            data = self.get_latest(topic)
            if data is not None:
                obs[obs_key] = data

        action = self._agent.act(obs)
        ts = time.time()

        if "_record" in action:
            self.publish("record", {"record": bool(action["_record"])}, ts=ts)

        # Optional action-chunk snapshot for visualization consumers (e.g.
        # ViserMonitorNode rendering predicted end-effector positions). The
        # agent sets this under "_chunk" — keep it off the joint-command path.
        chunk = action.get("_chunk")
        if chunk is not None:
            self.publish("chunk", chunk, ts=ts)

        # Optional preprocessed-image snapshots — the exact frames fed to the
        # policy. Mirrored on image/{label} so downstream viewers (viser
        # monitor) can subscribe without reprocessing raw cameras themselves.
        # record=False: raw frames are already being written to MP4 by each
        # CameraNode's AsyncMp4Writer; recording them again here would route
        # 3× VGA-sized arrays through McapWriter's JSON fallback on the agent's
        # hot path (~45 ms/step → drops a 30 Hz loop to ~15 Hz during record).
        images = action.get("_images")
        if images:
            for label, img in images.items():
                self.publish(
                    f"image/{label}",
                    {"images": {"rgb": img}},
                    ts=ts,
                    record=False,
                )

        self._publish_commands(action, ts)

    def _publish_commands(self, action: dict, ts: float) -> None:
        if self._arm_key is not None:
            arm_action = action.get(self._arm_key)
            if arm_action is not None:
                pos = arm_action["pos"] if isinstance(arm_action, dict) else arm_action
                self.publish(
                    "joint_pos",
                    {"joint_pos": self._finalize_pos(pos, self._arm_key)},
                    ts=ts,
                )
        elif "pos" in action:
            self.publish(
                "joint_pos",
                {"joint_pos": self._finalize_pos(action["pos"], "default")},
                ts=ts,
            )
        else:
            arm_keys = [k for k in action if not k.startswith("_")]
            if len(arm_keys) == 1:
                arm_action = action[arm_keys[0]]
                if isinstance(arm_action, dict) and "pos" in arm_action:
                    self.publish(
                        "joint_pos",
                        {"joint_pos": self._finalize_pos(arm_action["pos"], arm_keys[0])},
                        ts=ts,
                    )
            else:
                for key in arm_keys:
                    arm_action = action[key]
                    if isinstance(arm_action, dict) and "pos" in arm_action:
                        self.publish(
                            f"{key}_pos",
                            {"joint_pos": self._finalize_pos(arm_action["pos"], key)},
                            ts=ts,
                        )

    def _finalize_pos(self, pos, arm_key: str) -> np.ndarray:
        """Normalize the gripper, then apply the safety guardrails for this arm."""
        return self._apply_safety(self._process_pos(pos), arm_key)

    def _process_pos(self, pos) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        if self._normalize_gripper and len(pos) > 6:
            span = self._gripper_open_deg - self._gripper_closed_deg
            pos = pos.copy()
            pos[-1] = float(np.clip((pos[-1] - self._gripper_closed_deg) / span, 0.0, 1.0))
        return pos

    def _apply_safety(self, pos: np.ndarray, arm_key: str) -> np.ndarray:
        """Run the per-arm guardrail chain (speed cap -> bounding box). Clamp, never drop."""
        chain = self._guardrails.get(arm_key) or self._guardrails.get("default")
        if not chain:
            return pos
        out = np.asarray(pos, dtype=np.float64)
        for guardrail in chain:
            out, event = guardrail.apply(out)
            if event is not None:
                self._record_clamp(event)
        return out.astype(np.float32)

    def _record_clamp(self, event) -> None:
        """Count clamp events; log at most ~1 Hz so telemetry can't stall the loop."""
        self._clamp_counts[event.guardrail] = self._clamp_counts.get(event.guardrail, 0) + 1
        now = time.monotonic()
        if now - self._last_clamp_log > 1.0:
            self._last_clamp_log = now
            logger.warning(
                "[%s] %s (total %d)", self.name, event, self._clamp_counts[event.guardrail]
            )

    def cleanup(self) -> None:
        if self._agent is not None and hasattr(self._agent, "close"):
            self._agent.close()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name": params["name"],
            "agent_class": params.get("agent_class"),
            "agent_kwargs": params.get("agent_kwargs") or {},
            "loop_mode": params.get("loop_mode", "subscriber_driven"),
            "poll_freq": params.get("poll_freq"),
            "publish_freq": params.get("publish_freq"),
            "state_topics": params.get("state_topics"),
            "image_topics": params.get("image_topics"),
            "arm_key": params.get("arm_key"),
            "normalize_gripper": params.get("normalize_gripper", False),
            "gripper_open_deg": params.get("gripper_open_deg", 85.0),
            "gripper_closed_deg": params.get("gripper_closed_deg", 5.0),
            "safety": params.get("safety"),
        }
