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
from robots_realtime.runtime.safety.config import validate_cartesian_workspace_config, validate_safety_config
from robots_realtime.runtime.safety.cartesian import CartesianWorkspaceRejectGuardrail
from robots_realtime.runtime.safety.guardrails import InferenceAccelerationGuardrail


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
        safety:          Optional command safety config from YAML.
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
        writer=None,
        safety: dict | None = None,
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
        self._safety_config = safety
        self._safety_agent_type = "teleop"
        self._cartesian_guardrails: dict[str | None, CartesianWorkspaceRejectGuardrail] = {}
        self._accel_guardrail: InferenceAccelerationGuardrail | None = None
        self._last_cmd: dict[str | None, np.ndarray] = {}
        self._clamp_log: list[dict] = []

    # ------------------------------------------------------------------

    def setup(self, fk_factory=None) -> None:
        if self._agent is None:
            if self._agent_class is None:
                raise RuntimeError(
                    f"[{self.name}] AgentNode requires 'agent' or 'agent_class'"
                )
            self._agent = self._build_agent()
        self._setup_safety_guardrails(fk_factory=fk_factory)
        self._validate_cartesian_startup()
        if hasattr(self._agent, "reset"):
            self._agent.reset()

    def _validate_cartesian_startup(self) -> None:
        if not self._safety_config:
            return

        cfg = self._safety_config
        mode = cfg.get("mode", "sim")
        arms = cfg.get("arms", {})
        current_state = cfg.get("production_current_state")

        for arm_key, arm_cfg in arms.items():
            cw = arm_cfg.get("cartesian_workspace")
            if not cw or not cw.get("enabled"):
                continue

            if mode == "real" and current_state is None:
                raise ValueError(
                    f"Cartesian workspace enabled for arm '{arm_key}' but "
                    "production_current_state is missing. Guarded teleop requires "
                    "initial follower state to initialize last_safe. "
                    "Set production_current_state in safety config."
                )

            if current_state and arm_key in current_state:
                qpos = current_state[arm_key].get("qpos")
                if qpos is not None and arm_key in self._cartesian_guardrails:
                    guardrail = self._cartesian_guardrails[arm_key]
                    q = np.asarray(qpos, dtype=np.float64)
                    xyz = guardrail._fk_call(q, guardrail._site_name)[:3, 3]
                    if not guardrail._is_in_bounds(xyz):
                        raise ValueError(
                            f"Cartesian workspace startup validation failed for arm '{arm_key}': "
                            f"initial pose FK position {xyz.tolist()} is outside bounds "
                            f"[{guardrail._min_xyz.tolist()}, {guardrail._max_xyz.tolist()}]. "
                            "Guarded teleop fails closed. Move arm to safe position before starting."
                        )
                    guardrail.mark_published_safe(q)

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
        obs: dict = {"timestamp": time.time()}
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
                    {"joint_pos": self._process_pos(pos, arm_key=self._arm_key)},
                    ts=ts,
                )
        elif "pos" in action:
            self.publish(
                "joint_pos",
                {"joint_pos": self._process_pos(action["pos"], arm_key=self._arm_key)},
                ts=ts,
            )
        else:
            arm_keys = [k for k in action if not k.startswith("_")]
            if len(arm_keys) == 1:
                arm_action = action[arm_keys[0]]
                if isinstance(arm_action, dict) and "pos" in arm_action:
                    self.publish(
                        "joint_pos",
                        {"joint_pos": self._process_pos(arm_action["pos"], arm_key=arm_keys[0])},
                        ts=ts,
                    )
            else:
                for key in arm_keys:
                    arm_action = action[key]
                    if isinstance(arm_action, dict) and "pos" in arm_action:
                        self.publish(
                            f"{key}_pos",
                            {"joint_pos": self._process_pos(arm_action["pos"], arm_key=key)},
                            ts=ts,
                        )

    def _process_pos(self, pos, arm_key: str | None = None) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        if self._normalize_gripper and len(pos) > 6:
            span = self._gripper_open_deg - self._gripper_closed_deg
            pos = pos.copy()
            pos[-1] = float(np.clip((pos[-1] - self._gripper_closed_deg) / span, 0.0, 1.0))
        guardrail_key = self._resolve_guardrail_key(arm_key)

        # Cartesian workspace guardrail (FK-based reject/hold)
        cw_guardrail = self._cartesian_guardrails.get(guardrail_key)
        if cw_guardrail is not None:
            result = cw_guardrail.apply(pos)
            if result.state != "accepted":
                original = pos.copy()
                pos = result.final_command
                if not np.array_equal(pos, original):
                    self._log_clamp(guardrail_key, "cartesian", original, pos)

        if self._safety_agent_type == "inference" and self._accel_guardrail is not None:
            prev_cmd = self._last_cmd.get(guardrail_key)
            if prev_cmd is not None:
                original = pos.copy()
                pos = self._accel_guardrail.apply(prev_cmd, pos)
                if not np.array_equal(pos, original):
                    self._log_clamp(guardrail_key, "accel", original, pos)

        # Update cartesian last_safe with final published command
        if cw_guardrail is not None:
            cw_guardrail.mark_published_safe(pos)

        self._last_cmd[guardrail_key] = pos.copy()
        return pos

    def _log_clamp(self, arm_key: str | None, guardrail: str, original: np.ndarray, clamped: np.ndarray) -> None:
        entry = {
            "arm_key": arm_key,
            "guardrail": guardrail,
            "original": original.tolist(),
            "clamped": clamped.tolist(),
            "timestamp": time.time(),
        }
        self._clamp_log.append(entry)
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Safety clamp: arm={arm_key}, guardrail={guardrail}, "
            f"original={original.tolist()}, clamped={clamped.tolist()}"
        )

    def _setup_safety_guardrails(self, fk_factory=None) -> None:
        self._cartesian_guardrails = {}
        self._accel_guardrail = None
        self._last_cmd = {}
        self._clamp_log = []

        if not self._safety_config:
            self._safety_agent_type = "teleop"
            return

        cfg = self._safety_config
        mode = cfg.get("mode", "sim")
        self._safety_agent_type = cfg.get("agent_type", "teleop")
        acceleration_limit = cfg.get("acceleration_limit")

        arms = cfg.get("arms")
        if mode == "real" and arms is not None and len(arms) == 0:
            raise ValueError(
                "Real hardware requires per-arm safety config. "
                "safety.arms is empty; add entries for each arm (e.g. left, right)."
            )
        if arms:
            for arm_key, arm_cfg in arms.items():
                # Cartesian workspace guardrail
                cw = arm_cfg.get("cartesian_workspace")
                if cw and cw.get("enabled"):
                    try:
                        validate_cartesian_workspace_config(
                            {
                                "agent_type": self._safety_agent_type,
                                "site_name": cw.get("site_name"),
                                "xml_path": cw.get("xml_path"),
                                "frame": cw.get("frame"),
                                "min_xyz": cw.get("min_xyz"),
                                "max_xyz": cw.get("max_xyz"),
                                "enforcement": cw.get("enforcement"),
                            }
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Cartesian workspace validation failed for arm '{arm_key}': {e}"
                        ) from e

                    # Convert reentry_max_velocity_rad_s to per-cycle delta
                    control_hz = cw.get("configured_control_hz", 200.0)
                    reentry_delta = cw.get("reentry_max_velocity_rad_s", 1.0) / control_hz

                    # Get FK provider from factory (allows test mocking)
                    fk = None
                    if fk_factory is not None:
                        fk = fk_factory(cw.get("xml_path"), cw.get("site_name"))
                    else:
                        try:
                            from i2rt.robots.kinematics import Kinematics
                            fk = Kinematics(cw["xml_path"], cw["site_name"])
                        except (ImportError, KeyError, ValueError):
                            fk = None

                    if fk is not None:
                        pass_through = cw.get("pass_through_indices", [])
                        self._cartesian_guardrails[arm_key] = CartesianWorkspaceRejectGuardrail(
                            fk_provider=fk,
                            arm_key=arm_key,
                            site_name=cw["site_name"],
                            min_xyz=cw["min_xyz"],
                            max_xyz=cw["max_xyz"],
                            tolerance_m=cw.get("tolerance_m", 1e-4),
                            reentry_margin_m=cw.get("reentry_margin_m", 0.002),
                            reentry_max_delta_per_cycle=reentry_delta,
                            pass_through_indices=pass_through,
                        )

            if len(self._cartesian_guardrails) == 1:
                self._cartesian_guardrails[None] = next(iter(self._cartesian_guardrails.values()))

        if self._safety_agent_type == "inference" and acceleration_limit is not None:
            self._accel_guardrail = InferenceAccelerationGuardrail(
                float(acceleration_limit)
            )

        # Initialize last_safe from production_current_state and validate startup
        self._validate_cartesian_startup()

    def _resolve_guardrail_key(self, arm_key: str | None) -> str | None:
        if arm_key in self._cartesian_guardrails:
            return arm_key
        if self._arm_key in self._cartesian_guardrails:
            return self._arm_key
        if None in self._cartesian_guardrails:
            return None
        # Do NOT fall back to a different arm's guardrail — that's unsafe
        return arm_key

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
