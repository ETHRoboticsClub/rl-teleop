"""Session — top-level orchestrator.

The Session's monitor thread subscribes to every topic on the bus and measures
live publish-Hz per node.  All recording is delegated to the nodes themselves
via start_recording(save_dir) / stop_recording().

  - Nodes own their writers — no MCAP/video writing in the monitor thread.
  - Session.start_episode() creates the episode directory and calls
    host.start_recording(save_dir) for all hosts.
  - Session.end_episode() calls host.stop_recording() for all hosts.
"""

from __future__ import annotations

import datetime
import json
import logging
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import zmq

from robots_realtime.runtime.node import Node, ProcessHost
from robots_realtime.runtime.transport.message_bus import (
    DEFAULT_PUB_PORT,
    DEFAULT_SUB_PORT,
    MessageBus,
)
from robots_realtime.runtime.transport.serialization import unpack


_HZ_WINDOW = 30

logger = logging.getLogger(__name__)


class SessionStartupError(RuntimeError):
    """Raised when one or more critical nodes fail to bring up their hardware.

    Carries the per-node failure details so the CLI can print exactly which device
    failed and why (busy / locked / missing) instead of leaving a green TUI over a
    dead node.
    """

    def __init__(self, failures: list[tuple[str, str]]) -> None:
        self.failures = failures  # list of (node_name, detail)
        lines = "\n".join(f"  - {name}: {detail}" for name, detail in failures)
        super().__init__(
            "Session aborted — critical node(s) failed to start:\n" + lines
        )


def _run_git(args: list[str], cwd: Path, timeout: float = 2.0) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_dirty(cwd: Path) -> bool | None:
    status = _run_git(["status", "--porcelain"], cwd)
    if status is None:
        return None
    return bool(status)


def _git_metadata() -> dict:
    repo_root_raw = _run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    if not repo_root_raw:
        return {"available": False}

    repo_root = Path(repo_root_raw)
    meta: dict = {
        "available": True,
        "root": str(repo_root),
        "commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "branch": _run_git(["branch", "--show-current"], repo_root),
        "dirty": _git_dirty(repo_root),
        "submodules": [],
    }

    submodule_lines = _run_git(["submodule", "status", "--recursive"], repo_root)
    if not submodule_lines:
        return meta

    for line in submodule_lines.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        state = line[0] if line else " "
        commit = parts[0].lstrip("+-U")
        path = parts[1]
        desc = parts[2].strip("()") if len(parts) >= 3 else ""
        sub_path = repo_root / path
        meta["submodules"].append(
            {
                "path": path,
                "commit": commit,
                "descriptor": desc,
                "state": state,
                "dirty": _git_dirty(sub_path) if sub_path.exists() else None,
            }
        )
    return meta


def _node_descriptor(node) -> dict:
    d: dict = {
        "name": node.name,
        "published_topics": list(getattr(node, "published_topics", [])),
        "subscribed_topics": list(getattr(node, "subscribed_topics", [])),
    }
    # Include sim node config so replay tools can auto-detect scene/task.
    sim_cfg: dict = {}
    if getattr(node, "_scene", None) is not None:
        sim_cfg["scene"] = node._scene
    if getattr(node, "_task", None) is not None:
        sim_cfg["task"] = node._task
    if sim_cfg:
        d["sim_config"] = sim_cfg
    return d


@dataclass
class NodeStatus:
    name: str
    alive: bool = True
    pub_hz: float = 0.0
    step_hz: float = 0.0
    # Set when the node died / failed to start; surfaced by the TUI and CLI so the
    # operator sees *why* instead of a silent 0 Hz.
    fatal_reason: str = ""
    _timestamps: dict[str, deque] = field(default_factory=dict, repr=False)

    @property
    def hz(self) -> float:
        """Backwards-compatible alias for pub_hz."""
        return self.pub_hz

    def record_message(self, topic_suffix: str) -> None:
        buf = self._timestamps.setdefault(topic_suffix, deque(maxlen=_HZ_WINDOW))
        buf.append(time.perf_counter())
        best = max(self._timestamps.values(), key=len)
        if len(best) >= 2:
            span = best[-1] - best[0]
            self.pub_hz = (len(best) - 1) / span if span > 0 else 0.0


class Session:
    """Orchestrates a graph of Nodes.

    Each node runs in its own subprocess via ProcessHost.  Recording is
    delegated to the nodes via start_recording() / stop_recording().

    Args:
        nodes:                List of Node instances to run.
        save_root:            Root directory for episode recordings.
        record_node_names:    Subset of node names to record; defaults to all.
        record_topic:         Full bus topic carrying the boolean record signal
                              (e.g. "gello_left/record").
        auto_record_duration: If set, automatically start recording on
                              session start and stop after this many seconds.
        episode_timeout:      If set, automatically stop recording and pause
                              after this many seconds from episode start.
        instruction:          Optional text instruction for the recorded episode.
        instruction_mappings: Mapping from number keys to episode instructions.
        pub_port:             MessageBus XSUB port.
        sub_port:             MessageBus XPUB port.
    """

    def __init__(
        self,
        nodes: list,
        save_root: str | Path = "recordings",
        record_node_names: list[str] | None = None,
        record_topic: str | None = None,
        auto_record_duration: float | None = None,
        start_paused: bool = False,
        record_on_unpause: bool = False,
        episode_timeout: float | None = None,
        instruction: str | None = None,
        instruction_mappings: dict[str, str] | None = None,
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self._pub_port = pub_port
        self._sub_port = sub_port
        self._save_root = Path(save_root)
        self._record_topic = record_topic
        self._auto_record_duration = auto_record_duration
        # start_paused: begin with RobotNode commands gated so the arms don't
        # start tracking the policy (or any cmd_topic producer) until the
        # operator explicitly hits space. Recommended for policy configs where
        # the arm could snap to an unexpected pose on startup.
        # record_on_unpause: when the operator unpauses, automatically start an
        # episode if one isn't already running. Useful for policy eval where
        # you want every rollout captured from the instant the policy takes over.
        self._start_paused = bool(start_paused)
        self._record_on_unpause = bool(record_on_unpause)
        self._episode_timeout = episode_timeout
        self._instruction = instruction or ""
        self._instruction_mappings = self._normalize_instruction_mappings(
            instruction_mappings
        )
        self._episode_timeout_timer: threading.Timer | None = None
        self._is_paused: bool = False
        self._session_start_time = time.time()

        self._hosts: list[ProcessHost] = []
        all_node_names: list[str] = []
        self._node_descriptors: list[dict] = []

        for item in nodes:
            if isinstance(item, ProcessHost):
                # Accept pre-built hosts (advanced usage)
                self._hosts.append(item)
                all_node_names.extend(item.node_names)
            elif isinstance(item, Node):
                self._hosts.append(ProcessHost(item))
                all_node_names.extend([item.name])
                self._node_descriptors.append(_node_descriptor(item))
            else:
                raise TypeError(f"Unexpected item in nodes list: {type(item)}")

        self._record_node_names: list[str] = record_node_names or all_node_names

        self._bus = MessageBus(pub_port=pub_port, sub_port=sub_port)
        # Propagate the bus ports to the nodes so they publish/subscribe on the same
        # ports as the bus. Without this a Session built with explicit ports would leave
        # its nodes on the default ports and no messages would flow.
        for host in self._hosts:
            node = getattr(host, "_node", None)
            if node is not None:
                node._pub_port = self._pub_port
                node._sub_port = self._sub_port

        self._status: dict[str, NodeStatus] = {
            name: NodeStatus(name=name) for name in all_node_names
        }

        self._stop_event = threading.Event()
        self._prev_record_signal = False
        self._monitor_thread: threading.Thread | None = None
        self._log_dir: Path | None = None

        # Recording state
        self._is_recording: bool = False
        self._episode_dir: Path | None = None
        self._episode_start_time: float | None = None
        self._recording_lock = threading.Lock()
        self._recording_error: str = ""
        # Set when a critical node dies mid-session; the CLI exits non-zero on it.
        self._fatal_reason: str | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def configure_bus_ports(
        self,
        pub_port: int | None = None,
        sub_port: int | None = None,
    ) -> None:
        """Set broker and node ZMQ ports before the session is started."""
        if self._hosts:
            for host in self._hosts:
                if getattr(host, "_proc", None) is not None:
                    raise RuntimeError("Cannot change MessageBus ports after nodes start")

        if pub_port is not None:
            self._pub_port = int(pub_port)
        if sub_port is not None:
            self._sub_port = int(sub_port)

        self._bus = MessageBus(pub_port=self._pub_port, sub_port=self._sub_port)
        for host in self._hosts:
            node = getattr(host, "_node", None)
            if node is not None:
                node._pub_port = self._pub_port
                node._sub_port = self._sub_port

    def start(self) -> None:
        import tempfile
        self._log_dir = Path(tempfile.mkdtemp(prefix="rr_logs_"))

        self._bus.start()
        time.sleep(0.1)

        for host in self._hosts:
            log_path = self._log_dir / f"{host.node_name}.log"
            host.start(log_path=log_path)

        # If configured to start paused, broadcast PAUSE to every subprocess
        # BEFORE calling send_start(). Each ProcessHost's control socket is
        # already bound once host.start() returns, so the PAUSE arrives before
        # the node's step loop begins — RobotNode.step()'s first tick will see
        # self._paused=True and skip command_joint_pos.
        if self._start_paused:
            self._is_paused = True
            self._broadcast_control("pause")

        # Authoritative bring-up: each host now reports whether its hardware actually
        # opened. A critical node's failure aborts the whole session loudly; an optional
        # node's failure is logged and surfaced (alive=False) but the session continues.
        critical_failures: list[tuple[str, str]] = []
        for host in self._hosts:
            result = host.send_start()
            if result.ok:
                continue
            if host.critical:
                logger.error(
                    "Critical node '%s' failed to start: %s", result.node, result.detail
                )
                critical_failures.append((result.node, result.detail))
            else:
                logger.error(
                    "Optional node '%s' failed to start (continuing without it): %s",
                    result.node,
                    result.detail,
                )
                if result.node in self._status:
                    self._status[result.node].alive = False
                    self._status[result.node].fatal_reason = result.detail

        if critical_failures:
            self._abort_startup(critical_failures)

        self._setup_signal_handlers()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SessionMonitor"
        )
        self._monitor_thread.start()

        if self._auto_record_duration is not None:
            t = threading.Thread(
                target=self._auto_record_timer,
                args=(self._auto_record_duration,),
                daemon=True,
            )
            t.start()

    def _abort_startup(self, failures: list[tuple[str, str]]) -> None:
        """Tear everything down and raise loudly after a critical bring-up failure."""
        self._stop_event.set()
        for host in self._hosts:
            try:
                host.stop()
            except Exception as exc:  # teardown is best-effort; the raise below is the point
                logger.warning("Error stopping host during startup abort: %s", exc)
        try:
            self._bus.stop()
        except Exception as exc:
            logger.warning("Error stopping bus during startup abort: %s", exc)
        raise SessionStartupError(failures)

    def stop(self) -> None:
        self._stop_event.set()
        if self._is_recording:
            self.end_episode(save=True)
        threads = [
            threading.Thread(target=host.stop, daemon=True)
            for host in self._hosts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8.0)
        self._bus.stop()

    def wait(self) -> None:
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── Recording controls ────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def fatal_reason(self) -> str | None:
        """Non-None when a critical node died mid-session; the CLI exits non-zero on it."""
        return self._fatal_reason

    @property
    def episode_start_time(self) -> float | None:
        return self._episode_start_time

    @property
    def save_root(self) -> Path:
        return self._save_root

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str | None) -> None:
        self._instruction = value or ""

    @property
    def instruction_mappings(self) -> dict[str, str]:
        return dict(self._instruction_mappings)

    @staticmethod
    def _normalize_instruction_mappings(
        instruction_mappings: dict[str, str] | None,
    ) -> dict[str, str]:
        if not instruction_mappings:
            return {}

        normalized: dict[str, str] = {}
        for raw_key, raw_instruction in instruction_mappings.items():
            key = str(raw_key)
            if len(key) != 1 or key not in "0123456789":
                raise ValueError(
                    f"Instruction mapping key must be a number key 0-9, got {raw_key!r}"
                )
            instruction = str(raw_instruction or "").strip()
            if not instruction:
                raise ValueError(f"Instruction for key {key!r} must not be empty")
            normalized[key] = instruction
        return normalized

    def start_episode(self) -> None:
        with self._recording_lock:
            if self._is_recording:
                return
            save_dir = self._make_episode_dir()
            self._episode_dir = Path(save_dir)
            self._episode_start_time = time.time()
            self._is_recording = True

        # Delegate recording to all hosts. A failure here is loud: if a node cannot
        # start recording (crashed, device gone), the operator must know the episode is
        # not fully captured rather than believing it is.
        failed = []
        for host in self._hosts:
            try:
                host.start_recording(save_dir)
            except Exception as exc:
                failed.append(host.node_name)
                logger.error(
                    "Node '%s' failed to start recording into %s: %s",
                    host.node_name, save_dir, exc,
                )
        self._recording_error = "; ".join(failed) if failed else ""

        if self._episode_timeout is not None:
            self._episode_timeout_timer = threading.Timer(
                self._episode_timeout, self._on_episode_timeout
            )
            self._episode_timeout_timer.daemon = True
            self._episode_timeout_timer.start()

    def _on_episode_timeout(self) -> None:
        self.end_episode(save=True)
        self.pause()

    def end_episode(
        self,
        save: bool = True,
        instruction: str | None = None,
    ) -> Path | None:
        if self._episode_timeout_timer is not None:
            self._episode_timeout_timer.cancel()
            self._episode_timeout_timer = None

        with self._recording_lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            episode_dir = self._episode_dir
            self._episode_dir = None
            self._episode_start_time = None

        # Delegate stop_recording to all hosts in parallel. Camera MP4 writers
        # may need to drain encoder queues on close; doing this serially makes
        # episode save latency add up across cameras.
        threads = [
            threading.Thread(target=self._safe_stop_recording, args=(host,), daemon=True)
            for host in self._hosts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=12.0)

        if not save and episode_dir is not None:
            import shutil
            shutil.rmtree(episode_dir, ignore_errors=True)
            return None

        if instruction is not None:
            self._instruction = instruction
            if episode_dir is not None:
                self._write_session_meta(episode_dir, instruction=instruction)

        return episode_dir

    @staticmethod
    def _safe_stop_recording(host) -> None:
        try:
            host.stop_recording()
        except Exception as exc:
            logger.error(
                "Node '%s' failed to stop recording (data may be incomplete): %s",
                getattr(host, "node_name", "?"), exc,
            )

    def toggle_recording(self) -> None:
        if self._is_recording:
            self.end_episode(save=True)
        else:
            self.start_episode()

    # ------------------------------------------------------------------
    # Pause / resume — gates RobotNode command output; other nodes keep
    # running (cameras, agents) so the TUI, viser, and inference stay live.
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return getattr(self, "_is_paused", False)

    def _broadcast_control(self, action: str) -> None:
        """Send a control call (pause/resume) to every host, logging any failure.

        Failures used to be silently swallowed — a host that ignored a pause left the
        arms live while the operator believed they were held. Now it is logged loudly.
        """
        for host in self._hosts:
            try:
                getattr(host, action)()
            except Exception as exc:
                logger.warning("Host '%s' %s() failed: %s", host.node_name, action, exc)

    def pause(self) -> None:
        if getattr(self, "_is_paused", False):
            return
        self._is_paused = True
        self._broadcast_control("pause")

    def resume(self) -> None:
        if not getattr(self, "_is_paused", False):
            return
        self._is_paused = False
        self._broadcast_control("resume")
        # Optional: prime recording when the operator unpauses. Lets a policy
        # eval config capture every rollout from the instant of handoff.
        if self._record_on_unpause and not self._is_recording:
            try:
                self.start_episode()
            except Exception as exc:
                logger.error("Failed to auto-start episode on unpause: %s", exc)

    def toggle_pause(self) -> None:
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    @property
    def log_dir(self) -> Path | None:
        return self._log_dir

    @property
    def web_endpoints(self) -> list[str]:
        urls = []
        for host in self._hosts:
            urls.extend(host._node.web_endpoints)
        return urls

    # ── Status ────────────────────────────────────────────────────────────────

    def node_statuses(self) -> list[NodeStatus]:
        return list(self._status.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_episode_dir(self) -> str:
        """Create the episode directory and write session_meta.json."""
        now = datetime.datetime.now()
        uid = uuid.uuid4().hex[:8]
        path = (
            self._save_root
            / now.strftime("%Y%m%d")
            / f"episode_{now.strftime('%H%M%S')}_{uid}"
        )
        path.mkdir(parents=True, exist_ok=True)
        self._write_session_meta(path, instruction=self._instruction)

        return str(path)

    def _write_session_meta(self, path: Path, instruction: str) -> None:
        meta_path = path / "session_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        if not meta:
            meta = {
                "session_start_time": self._session_start_time,
                "episode_start_time": time.time(),
                "episode_dir": str(path),
                "nodes": self._node_descriptors,
                "record_topic": self._record_topic,
                "save_root": str(self._save_root),
                "git": _git_metadata(),
            }
        meta["instruction"] = instruction
        meta["instruction_mappings"] = self._instruction_mappings
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        except Exception:
            pass

    def _monitor_loop(self) -> None:
        """Subscribe to all bus topics; measure Hz per node.

        Also watches record_topic for start/stop signals.
        """
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{self._sub_port}")
        sock.setsockopt(zmq.SUBSCRIBE, b"")

        node_names = set(self._status)
        hosts_by_name = {h.node_name: h for h in self._hosts}
        last_liveness_check = 0.0

        while not self._stop_event.is_set():
            # Poll process liveness (~4 Hz) so a node that dies mid-session (device
            # yanked, driver crash) is surfaced loudly instead of silently going 0 Hz.
            now_mono = time.monotonic()
            if now_mono - last_liveness_check > 0.25:
                last_liveness_check = now_mono
                self._check_node_liveness(hosts_by_name)

            while sock.poll(0):
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 2:
                    continue

                topic_b, payload_b = parts[0], parts[1]
                topic = topic_b.decode()
                parts = topic.split("/", 1)
                if not parts:
                    continue
                node_name = parts[0]
                topic_suffix = parts[1] if len(parts) > 1 else ""

                if node_name not in node_names:
                    continue

                # Internal step-rate report from the node process
                if topic_suffix == "_step_hz":
                    try:
                        envelope = unpack(payload_b)
                        self._status[node_name].step_hz = float(
                            envelope.get("data", {}).get("step_hz", 0.0)
                        )
                    except Exception:
                        pass
                    continue

                # Measure publish Hz
                self._status[node_name].record_message(topic_suffix)

                # Handle record signal from gello (or any configured topic)
                if self._record_topic and topic == self._record_topic:
                    try:
                        envelope = unpack(payload_b)
                        want = bool(envelope.get("data", {}).get("record", False))
                        if want and not self._prev_record_signal:
                            self.start_episode()
                        elif not want and self._prev_record_signal:
                            self.end_episode(save=True)
                        self._prev_record_signal = want
                    except Exception:
                        pass

            time.sleep(0.005)

        sock.close(linger=0)

    def _check_node_liveness(self, hosts_by_name: dict) -> None:
        """Flag nodes whose subprocess has died; abort the session if one was critical.

        Only detects live→dead transitions (each node is flagged once). A dead critical
        node sets ``_fatal_reason`` and the stop event so the operator's session tears
        down loudly with a non-zero exit instead of running on with a missing arm/camera.
        """
        for name, host in hosts_by_name.items():
            status = self._status.get(name)
            if status is None or not status.alive:
                continue  # already flagged, or unknown
            proc = getattr(host, "_proc", None)
            if proc is not None and not proc.is_alive():
                status.alive = False
                if not status.fatal_reason:
                    status.fatal_reason = f"node process exited (code {proc.exitcode})"
                logger.error(
                    "Node '%s' died mid-session: %s", name, status.fatal_reason
                )
                if getattr(host, "critical", True):
                    self._fatal_reason = (
                        f"critical node '{name}' died: {status.fatal_reason}"
                    )
                    self._stop_event.set()

    def _auto_record_timer(self, duration: float) -> None:
        # Brief warmup so ZMQ sockets connect and nodes start publishing
        time.sleep(0.3)
        self.start_episode()
        time.sleep(duration)
        self.end_episode(save=True)
        self._stop_event.set()

    def _setup_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGUSR1, lambda *_: self.toggle_recording())
            signal.signal(signal.SIGUSR2, lambda *_: self.end_episode(save=False))
        except (OSError, ValueError):
            pass
