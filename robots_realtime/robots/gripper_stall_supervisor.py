"""Gripper stall supervisor for the YAM parallel-jaw gripper.

Implements fixes 4 and 6 from ``research/gripper-mat-contact.md``:

* **Fix 4 — current/force-limited close.** The i2rt driver already implements
  the Dynamixel-style "current-based position mode" equivalent: a
  ``GripperForceLimiter`` capping gripper torque once the jaws clog
  (``limit_gripper_force`` in the robot YAML, in Newtons).
  :func:`set_gripper_force_limit` reconfigures that limiter at runtime so a
  close can run under a chosen force limit without reconstructing the robot.

* **Fix 6 — stall detection + bounded lift-and-regrasp.**
  :class:`GripperStallDetector` classifies a close as *stalled on the surface*
  when, for N consecutive samples, the encoder is static AND the position
  error is high AND the measured effort sits at the limit. Position error
  must be HIGH for a stall verdict: a thin packet held between nearly-closed
  jaws also freezes the encoder at limit current, but with a *small* position
  error — that is a successful grasp, not a stall.
  :class:`GripperStallSupervisor` wraps a close with that detector and a
  bounded recovery policy: open, lift 1-2 cm, small lateral shift, lower,
  retry — at most ``max_attempts`` closes, then report failure.

OPT-IN ONLY. Nothing in this repo constructs or activates the supervisor.
``GripperStallSupervisorConfig.enabled`` defaults to ``False`` and the
supervisor refuses to construct unless it is explicitly set ``True``.
Pipeline code (e.g. sort_server) must opt in via its own config and call
``supervised_close()`` at a moment when it owns the arm — never while a
policy is concurrently streaming joint commands.

The supervisor is deliberately hardware-agnostic: it talks to the gripper
through the tiny :class:`GripperIO` protocol and performs recovery arm motion
through the :class:`RecoveryMotion` protocol, both injectable — which is also
what makes it unit-testable with synthetic traces.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, runtime_checkable

import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only
    from i2rt.robots.motor_chain_robot import MotorChainRobot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GripperReading:
    """One raw gripper sample from the driver.

    ``pos`` is in the normalized command space used by
    ``MotorChainRobot`` (0 = fully closed, 1 = fully open); ``eff`` is the
    motor torque in Nm as reported by ``get_observations()['gripper_eff']``.
    """

    t: float
    pos: float
    vel: float
    eff: float


@dataclass(frozen=True)
class GripperSample:
    """A reading paired with the close target it was taken against."""

    t: float
    pos: float
    vel: float
    eff: float
    target: float


# ---------------------------------------------------------------------------
# Protocols (injection points)
# ---------------------------------------------------------------------------
@runtime_checkable
class GripperIO(Protocol):
    """Minimal gripper access the supervisor needs."""

    def read(self) -> GripperReading:  # pragma: no cover - protocol
        ...

    def command_gripper(self, pos: float) -> None:  # pragma: no cover - protocol
        ...


@runtime_checkable
class RecoveryMotion(Protocol):
    """Arm motion primitives for recovery, supplied by the pipeline.

    All deltas are meters in the world/base frame. Implementations own the
    actual cartesian motion (IK, speed limits, workspace clamps) — the
    supervisor only sequences them.
    """

    def lift(self, dz_m: float) -> None:  # pragma: no cover - protocol
        ...

    def shift(self, dy_m: float) -> None:  # pragma: no cover - protocol
        ...

    def lower(self, dz_m: float) -> None:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Fix 4 — runtime force/current limit on the close
# ---------------------------------------------------------------------------
def set_gripper_force_limit(robot: "MotorChainRobot", max_force_newtons: float) -> bool:
    """Set the gripper close force limit (Newtons) on a MotorChainRobot.

    Reconfigures the i2rt ``GripperForceLimiter`` the robot was constructed
    with (``limit_gripper_force`` in the robot YAML) so subsequent closes run
    under ``max_force_newtons``. This is the i2rt equivalent of Dynamixel
    current-based position mode: a blocked close settles at the force limit
    instead of wedging at full PD effort.

    Returns ``True`` on success, ``False`` if this robot's driver does not
    expose a usable force limiter (no gripper, or the gripper type has no
    limiter params). Never raises for a missing limiter so callers can treat
    it as a capability probe.
    """
    if max_force_newtons <= 0:
        raise ValueError(f"max_force_newtons must be > 0, got {max_force_newtons}")

    limiter = getattr(robot, "_gripper_force_limiter", None)
    if limiter is None or getattr(robot, "_gripper_index", None) is None:
        logger.warning("set_gripper_force_limit: robot has no gripper force limiter")
        return False

    # limiter.gripper_force_torque_map is partial(map_fn, <geometry kwargs>,
    # gripper_force=old) — functools.partial flattens the nested partials the
    # i2rt code builds, so .func is the raw map and .keywords carries the
    # gripper geometry (motor/gripper stroke etc.). Rebuild it with only
    # gripper_force replaced.
    old_map = limiter.gripper_force_torque_map
    base_map = getattr(old_map, "func", None)
    if base_map is None:
        logger.warning(
            "set_gripper_force_limit: gripper type exposes no force→torque map"
        )
        return False

    kwargs = dict(getattr(old_map, "keywords", None) or {})
    kwargs["gripper_force"] = float(max_force_newtons)
    limiter.max_force = float(max_force_newtons)
    limiter.gripper_force_torque_map = partial(
        base_map, *getattr(old_map, "args", ()), **kwargs
    )
    # MotorChainRobot.update() only consults the limiter when this is > 0.
    robot._limit_gripper_force = float(max_force_newtons)
    logger.info("gripper close force limit set to %.1f N", max_force_newtons)
    return True


# ---------------------------------------------------------------------------
# Fix 6a — stall detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StallDetectorConfig:
    """Thresholds for the stall verdict, in normalized gripper units / Nm.

    A sample counts toward a stall only when ALL three hold:
      * encoder static: |Δpos| between consecutive samples <= ``encoder_eps``
      * position error high: |target - pos| >= ``pos_error_threshold``
      * current at limit: |eff| >= ``effort_stall_threshold``

    ``pos_error_threshold`` must exceed the residual opening left by the
    thickest object you legitimately grasp, otherwise a successful grasp is
    misread as a stall. For thin flat packets the jaws close almost fully, so
    the default 0.15 (15 % of stroke) is safe.
    """

    encoder_eps: float = 0.004
    pos_error_threshold: float = 0.15
    effort_stall_threshold: float = 0.45  # Nm; linear_4310 clog threshold is 0.5
    consecutive_samples: int = 25  # ~0.5 s at 50 Hz


class GripperStallDetector:
    """Feed samples during a close; verdict latches once stalled.

    Pure logic, no I/O, no clock — drive it with real driver samples or
    synthetic traces alike.
    """

    def __init__(self, config: Optional[StallDetectorConfig] = None) -> None:
        self.config = config or StallDetectorConfig()
        self.reset()

    def reset(self) -> None:
        self._last: Optional[GripperSample] = None
        self._count = 0
        self._stalled = False

    @property
    def stalled(self) -> bool:
        return self._stalled

    @property
    def consecutive_stall_samples(self) -> int:
        return self._count

    def update(self, sample: GripperSample) -> bool:
        """Ingest one sample; returns True once the stall verdict latches."""
        if self._stalled:
            return True

        cfg = self.config
        encoder_static = (
            self._last is not None
            and abs(sample.pos - self._last.pos) <= cfg.encoder_eps
        )
        pos_error_high = abs(sample.target - sample.pos) >= cfg.pos_error_threshold
        current_at_limit = abs(sample.eff) >= cfg.effort_stall_threshold

        if encoder_static and pos_error_high and current_at_limit:
            self._count += 1
        else:
            self._count = 0

        self._last = sample
        if self._count >= cfg.consecutive_samples:
            self._stalled = True
        return self._stalled


# ---------------------------------------------------------------------------
# Fix 6b — supervisor with bounded recovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecoveryConfig:
    lift_height_m: float = 0.015  # 1-2 cm per the research note
    lateral_shift_m: float = 0.005  # "shift a few mm"
    max_attempts: int = 3  # total close attempts (so at most 2 recoveries)
    open_pos: float = 1.0  # normalized open command before lifting
    settle_time_s: float = 0.25


@dataclass(frozen=True)
class GripperStallSupervisorConfig:
    """Full supervisor config. ``enabled`` defaults to False — opt-in only."""

    enabled: bool = False
    close_force_newtons: Optional[float] = None  # if set, applied via fix 4
    close_target: float = 0.0  # normalized; 0 = fully closed
    close_reached_eps: float = 0.03  # |pos - target| below this = closed empty
    close_timeout_s: float = 4.0
    sample_rate_hz: float = 50.0
    detector: StallDetectorConfig = field(default_factory=StallDetectorConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "GripperStallSupervisorConfig":
        d = dict(d or {})
        detector = StallDetectorConfig(**(d.pop("detector", None) or {}))
        recovery = RecoveryConfig(**(d.pop("recovery", None) or {}))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown gripper supervisor config keys: {sorted(unknown)}")
        return cls(detector=detector, recovery=recovery, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "GripperStallSupervisorConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # allow either a flat file or a file with a single top-level section
        if "gripper_stall_supervisor" in raw:
            raw = raw["gripper_stall_supervisor"] or {}
        return cls.from_dict(raw)


class CloseVerdict(str, enum.Enum):
    CLOSED = "closed"  # reached target, or holding an object (held=True)
    STALLED = "stalled"  # friction stall on the surface
    TIMEOUT = "timeout"  # neither closed nor a recognizable stall in time


@dataclass
class CloseAttempt:
    index: int  # 1-based
    verdict: CloseVerdict
    held: bool  # True when closed on an object (encoder static at effort)
    duration_s: float
    final_pos: float
    final_eff: float
    n_samples: int


@dataclass
class SupervisedCloseResult:
    success: bool
    verdict: str  # "closed" | "stall_recovered" | "stall_unrecovered" | "timeout"
    held: bool
    attempts: List[CloseAttempt]
    recoveries: int
    force_limit_applied: Optional[float]

    @property
    def stalled_at_least_once(self) -> bool:
        return any(a.verdict is CloseVerdict.STALLED for a in self.attempts)


class GripperStallSupervisor:
    """Supervised gripper close with stall detection and bounded regrasp.

    Construction requires ``config.enabled`` to be True — a default config
    (or a YAML without ``enabled: true``) can never activate it by accident.

    Args:
        config: supervisor configuration (must have ``enabled=True``).
        io: gripper read/command access (see :class:`GripperIO`;
            :class:`MotorChainGripperIO` adapts an i2rt robot).
        motion: recovery motion primitives (see :class:`RecoveryMotion`).
        force_limit_setter: callable applying a force limit in Newtons;
            defaults to nothing. Pass e.g.
            ``lambda f: set_gripper_force_limit(robot, f)`` to enable fix 4.
        time_fn / sleep_fn: injectable clock, for tests.
    """

    def __init__(
        self,
        config: GripperStallSupervisorConfig,
        io: GripperIO,
        motion: RecoveryMotion,
        force_limit_setter: Optional[Callable[[float], bool]] = None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.enabled:
            raise ValueError(
                "GripperStallSupervisor is opt-in: refusing to construct with "
                "config.enabled=False. Set enabled: true in the pipeline's "
                "gripper_stall_supervisor config to activate it."
            )
        if config.recovery.max_attempts < 1:
            raise ValueError("recovery.max_attempts must be >= 1")
        self.config = config
        self.io = io
        self.motion = motion
        self._force_limit_setter = force_limit_setter
        self._time = time_fn
        self._sleep = sleep_fn
        self._detector = GripperStallDetector(config.detector)
        self._net_shift_m = 0.0

    # -- public ------------------------------------------------------------
    def supervised_close(self, target: Optional[float] = None) -> SupervisedCloseResult:
        """Close the gripper under supervision; recover from stalls, bounded.

        Returns a :class:`SupervisedCloseResult`; never raises for a failed
        grasp — pipeline code decides what a failed cycle means.
        """
        cfg = self.config
        close_target = cfg.close_target if target is None else float(target)

        force_applied: Optional[float] = None
        if cfg.close_force_newtons is not None and self._force_limit_setter is not None:
            if self._force_limit_setter(cfg.close_force_newtons):
                force_applied = cfg.close_force_newtons
            else:
                logger.warning(
                    "supervised_close: force limit %.1f N requested but the "
                    "driver did not accept it; closing without it",
                    cfg.close_force_newtons,
                )

        attempts: List[CloseAttempt] = []
        recoveries = 0
        self._net_shift_m = 0.0

        for attempt_idx in range(1, cfg.recovery.max_attempts + 1):
            attempt = self._close_once(attempt_idx, close_target)
            attempts.append(attempt)

            if attempt.verdict is CloseVerdict.CLOSED:
                verdict = "closed" if recoveries == 0 else "stall_recovered"
                return SupervisedCloseResult(
                    success=True,
                    verdict=verdict,
                    held=attempt.held,
                    attempts=attempts,
                    recoveries=recoveries,
                    force_limit_applied=force_applied,
                )

            if attempt.verdict is CloseVerdict.TIMEOUT:
                # Not a recognizable friction stall — do not blind-retry.
                logger.warning(
                    "supervised_close: attempt %d timed out without a stall "
                    "signature (pos=%.3f eff=%.3f); reporting failure",
                    attempt_idx,
                    attempt.final_pos,
                    attempt.final_eff,
                )
                return SupervisedCloseResult(
                    success=False,
                    verdict="timeout",
                    held=False,
                    attempts=attempts,
                    recoveries=recoveries,
                    force_limit_applied=force_applied,
                )

            # Stalled. Recover unless this was the last permitted attempt.
            logger.warning(
                "supervised_close: close attempt %d stalled at pos=%.3f "
                "(eff=%.3f)",
                attempt_idx,
                attempt.final_pos,
                attempt.final_eff,
            )
            if attempt_idx >= cfg.recovery.max_attempts:
                break
            self._recover(recovery_index=recoveries + 1)
            recoveries += 1

        return SupervisedCloseResult(
            success=False,
            verdict="stall_unrecovered",
            held=False,
            attempts=attempts,
            recoveries=recoveries,
            force_limit_applied=force_applied,
        )

    # -- internals ---------------------------------------------------------
    def _close_once(self, attempt_idx: int, target: float) -> CloseAttempt:
        cfg = self.config
        det = self._detector
        det.reset()
        period = 1.0 / cfg.sample_rate_hz

        self.io.command_gripper(target)
        t0 = self._time()
        deadline = t0 + cfg.close_timeout_s

        n_samples = 0
        held_count = 0
        last: Optional[GripperReading] = None

        while True:
            reading = self.io.read()
            n_samples += 1
            sample = GripperSample(
                t=reading.t,
                pos=reading.pos,
                vel=reading.vel,
                eff=reading.eff,
                target=target,
            )

            if det.update(sample):
                return self._attempt(attempt_idx, CloseVerdict.STALLED, False, t0, reading, n_samples)

            if abs(reading.pos - target) <= cfg.close_reached_eps:
                return self._attempt(attempt_idx, CloseVerdict.CLOSED, False, t0, reading, n_samples)

            # Closed on an object: encoder static at limit current but with a
            # position error too small to be a surface stall.
            encoder_static = (
                last is not None
                and abs(reading.pos - last.pos) <= cfg.detector.encoder_eps
            )
            if (
                encoder_static
                and abs(reading.eff) >= cfg.detector.effort_stall_threshold
                and abs(target - reading.pos) < cfg.detector.pos_error_threshold
            ):
                held_count += 1
                if held_count >= cfg.detector.consecutive_samples:
                    return self._attempt(attempt_idx, CloseVerdict.CLOSED, True, t0, reading, n_samples)
            else:
                held_count = 0

            last = reading
            if self._time() >= deadline:
                return self._attempt(attempt_idx, CloseVerdict.TIMEOUT, False, t0, reading, n_samples)
            self._sleep(period)

    def _attempt(
        self,
        idx: int,
        verdict: CloseVerdict,
        held: bool,
        t0: float,
        reading: GripperReading,
        n_samples: int,
    ) -> CloseAttempt:
        return CloseAttempt(
            index=idx,
            verdict=verdict,
            held=held,
            duration_s=self._time() - t0,
            final_pos=reading.pos,
            final_eff=reading.eff,
            n_samples=n_samples,
        )

    def _recover(self, recovery_index: int) -> None:
        """Open, lift, shift laterally, lower — ready for the next attempt.

        Lateral shifts alternate sides with growing magnitude relative to the
        original grasp pose: +s, -s, +2s, -2s, ... so retries probe fresh
        surface instead of re-snagging the same spot.
        """
        rec = self.config.recovery
        logger.info(
            "supervised_close: recovery %d — open, lift %.3f m, shift, retry",
            recovery_index,
            rec.lift_height_m,
        )
        self.io.command_gripper(rec.open_pos)
        self._sleep(rec.settle_time_s)

        self.motion.lift(rec.lift_height_m)

        magnitude = (recovery_index + 1) // 2
        sign = 1.0 if recovery_index % 2 == 1 else -1.0
        desired_net = sign * magnitude * rec.lateral_shift_m
        delta = desired_net - self._net_shift_m
        if abs(delta) > 0.0:
            self.motion.shift(delta)
            self._net_shift_m = desired_net

        self.motion.lower(rec.lift_height_m)
        self._sleep(rec.settle_time_s)


# ---------------------------------------------------------------------------
# Adapter for the real robot
# ---------------------------------------------------------------------------
class MotorChainGripperIO:
    """GripperIO adapter for an i2rt ``MotorChainRobot``/``SafeMotorChainRobot``.

    WARNING: ``command_gripper()`` re-commands the arm joints at their current
    position (the driver only accepts full joint vectors). The caller must own
    the arm — never use this while a policy or teleop leader is concurrently
    streaming joint commands to the same robot.
    """

    def __init__(self, robot: "MotorChainRobot", time_fn: Callable[[], float] = time.monotonic) -> None:
        if getattr(robot, "_gripper_index", None) is None:
            raise ValueError("robot has no gripper (gripper_index is None)")
        self.robot = robot
        self._time = time_fn

    def read(self) -> GripperReading:
        obs = self.robot.get_observations()
        return GripperReading(
            t=self._time(),
            pos=float(obs["gripper_pos"][0]),
            vel=float(obs["gripper_vel"][0]),
            eff=float(obs["gripper_eff"][0]),
        )

    def command_gripper(self, pos: float) -> None:
        joint_pos = self.robot.get_joint_pos().copy()
        joint_pos[self.robot._gripper_index] = float(pos)
        self.robot.command_joint_pos(joint_pos)
