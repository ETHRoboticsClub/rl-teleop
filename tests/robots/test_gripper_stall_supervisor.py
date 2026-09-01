"""Unit tests for the gripper stall supervisor (fixes 4 and 6,
research/gripper-mat-contact.md).

All tests run against a synthetic gripper with an injected clock — no
hardware, no CAN, no bus. The FakeGripper models the failure mode the
supervisor exists for: jaws that freeze against mat friction with the motor
effort pinned at the limit, versus jaws that freeze because they are holding
a packet (small position error — a GOOD grasp).
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType

from robots_realtime.robots.gripper_stall_supervisor import (
    CloseVerdict,
    GripperReading,
    GripperSample,
    GripperStallDetector,
    GripperStallSupervisor,
    GripperStallSupervisorConfig,
    RecoveryConfig,
    StallDetectorConfig,
    set_gripper_force_limit,
)

# ---------------------------------------------------------------------------
# Synthetic gripper
# ---------------------------------------------------------------------------


class FakeGripper:
    """Scripted gripper physics + recovery motion recorder.

    Normalized position space (0 closed, 1 open). Time only advances through
    ``sleep_fn`` — inject ``time_fn``/``sleep_fn`` into the supervisor.

    * ``stall_floor``: closing cannot go below this while the stall is active
      (mat friction); effort ramps to ``eff_limit`` while blocked.
    * ``clears_after_recoveries``: the stall floor disappears after this many
      recoveries (lift() calls), modeling a recovery that frees the snag.
    * ``object_floor``: an object between the jaws — also blocks closing at
      effort, but near-closed (a successful grasp, not a stall).
    * ``dead``: commands have no effect and no current flows (encoder static,
      effort ~0) — must yield TIMEOUT, not a stall verdict.
    """

    CLOSE_SPEED = 2.0  # normalized units / s
    EFF_RISE = 3.0  # Nm / s while blocked
    EFF_LIMIT = 0.7  # Nm at the force limit
    EFF_MOVING = 0.08  # Nm while freely moving

    def __init__(
        self,
        stall_floor=None,
        clears_after_recoveries=None,
        object_floor=None,
        dead=False,
    ):
        self.t = 0.0
        self.pos = 1.0
        self.target = 1.0
        self.eff = 0.0
        self.stall_floor = stall_floor
        self.clears_after_recoveries = clears_after_recoveries
        self.object_floor = object_floor
        self.dead = dead

        self.commands = []
        self.lifts = []
        self.shifts = []
        self.lowers = []
        self.recoveries = 0

    # -- clock -------------------------------------------------------------
    def time_fn(self):
        return self.t

    def sleep_fn(self, dt):
        # step physics finely so floors are hit exactly, not overshot
        steps = max(1, math.ceil(dt / 0.005))
        sub = dt / steps
        for _ in range(steps):
            self._step(sub)

    def _step(self, dt):
        self.t += dt
        if self.dead:
            self.eff = 0.0
            return

        floor = None
        candidates = [f for f in (self._active_stall_floor(), self.object_floor) if f is not None]
        if candidates:
            floor = max(candidates)

        delta = self.target - self.pos
        step = max(-self.CLOSE_SPEED * dt, min(self.CLOSE_SPEED * dt, delta))
        new_pos = self.pos + step
        if step < 0 and floor is not None:  # closing into a floor
            new_pos = max(new_pos, floor)

        blocked = abs(new_pos - self.pos) < 1e-12 and abs(self.target - new_pos) > 1e-9
        self.pos = new_pos
        if blocked:
            self.eff = min(self.EFF_LIMIT, self.eff + self.EFF_RISE * dt)
        elif abs(self.target - self.pos) > 1e-9:
            self.eff = self.EFF_MOVING
        else:
            self.eff = 0.0

    def _active_stall_floor(self):
        if self.stall_floor is None:
            return None
        if (
            self.clears_after_recoveries is not None
            and self.recoveries >= self.clears_after_recoveries
        ):
            return None
        return self.stall_floor

    # -- GripperIO ---------------------------------------------------------
    def read(self):
        return GripperReading(t=self.t, pos=self.pos, vel=0.0, eff=self.eff)

    def command_gripper(self, pos):
        self.commands.append(pos)
        self.target = float(pos)

    # -- RecoveryMotion ----------------------------------------------------
    def lift(self, dz_m):
        self.lifts.append(dz_m)
        self.recoveries += 1

    def shift(self, dy_m):
        self.shifts.append(dy_m)

    def lower(self, dz_m):
        self.lowers.append(dz_m)


def make_supervisor(fake, **cfg_overrides):
    cfg_overrides.setdefault("enabled", True)
    config = GripperStallSupervisorConfig(**cfg_overrides)
    return GripperStallSupervisor(
        config=config,
        io=fake,
        motion=fake,
        time_fn=fake.time_fn,
        sleep_fn=fake.sleep_fn,
    )


def trace(positions, effs, target=0.0, dt=0.02):
    """Build a list of GripperSamples from parallel pos/eff sequences."""
    return [
        GripperSample(t=i * dt, pos=p, vel=0.0, eff=e, target=target)
        for i, (p, e) in enumerate(zip(positions, effs, strict=True))
    ]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TestStallDetector:
    CFG = StallDetectorConfig(
        encoder_eps=0.004,
        pos_error_threshold=0.15,
        effort_stall_threshold=0.45,
        consecutive_samples=5,
    )

    def test_clean_close_never_stalls(self):
        det = GripperStallDetector(self.CFG)
        n = 50
        positions = [1.0 - i * (1.0 / n) for i in range(n + 1)]
        effs = [0.08] * (n + 1)
        for s in trace(positions, effs):
            assert not det.update(s)
        assert not det.stalled

    def test_friction_stall_latches_after_n_samples(self):
        det = GripperStallDetector(self.CFG)
        # closes 1.0 -> 0.6 then freezes at 0.6 with effort at the limit
        moving = trace([1.0, 0.9, 0.8, 0.7, 0.6], [0.08] * 5)
        frozen = trace([0.6] * 10, [0.7] * 10)
        for s in moving:
            assert not det.update(s)
        verdicts = [det.update(s) for s in frozen]
        # first 4 static samples accumulate, 5th latches
        assert verdicts[:4] == [False] * 4
        assert verdicts[4] is True
        assert det.stalled
        # verdict latches
        assert det.update(trace([0.5], [0.0])[0]) is True

    def test_movement_resets_counter(self):
        det = GripperStallDetector(self.CFG)
        frozen = trace([0.6] * 4, [0.7] * 4)
        for s in frozen:
            det.update(s)
        assert det.consecutive_stall_samples > 0
        # a real move (> encoder_eps) resets the streak
        det.update(GripperSample(t=1.0, pos=0.55, vel=0.0, eff=0.7, target=0.0))
        assert det.consecutive_stall_samples == 0
        assert not det.stalled

    def test_low_effort_freeze_is_not_a_stall(self):
        det = GripperStallDetector(self.CFG)
        for s in trace([0.6] * 20, [0.1] * 20):
            assert not det.update(s)

    def test_held_object_small_pos_error_is_not_a_stall(self):
        # jaws frozen at 0.05 from target 0.0 at limit current: a grasp
        det = GripperStallDetector(self.CFG)
        for s in trace([0.05] * 20, [0.7] * 20):
            assert not det.update(s)

    def test_reset_clears_state(self):
        det = GripperStallDetector(self.CFG)
        for s in trace([0.6] * 10, [0.7] * 10):
            det.update(s)
        assert det.stalled
        det.reset()
        assert not det.stalled
        assert det.consecutive_stall_samples == 0


# ---------------------------------------------------------------------------
# Fix 4 — force limit setter
# ---------------------------------------------------------------------------


class TestSetGripperForceLimit:
    def _real_limiter_robot(self, max_force=50.0):
        limiter = GripperForceLimiter(
            max_force=max_force,
            gripper_type=GripperType.LINEAR_4310,
            arm_type=ArmType.YAM,
            kp=20.0,
        )
        return SimpleNamespace(
            _gripper_force_limiter=limiter,
            _gripper_index=6,
            _limit_gripper_force=max_force,
        )

    def test_reconfigures_real_i2rt_limiter(self):
        robot = self._real_limiter_robot(max_force=50.0)
        assert set_gripper_force_limit(robot, 30.0) is True
        limiter = robot._gripper_force_limiter
        assert limiter.max_force == 30.0
        assert robot._limit_gripper_force == 30.0
        # linear_4310 map: torque = F * gripper_stroke / motor_stroke
        expected = 30.0 * 0.096 / 6.57
        got = limiter.gripper_force_torque_map(current_angle=1.0)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_no_limiter_returns_false(self):
        robot = SimpleNamespace(_gripper_force_limiter=None, _gripper_index=6)
        assert set_gripper_force_limit(robot, 30.0) is False

    def test_no_gripper_returns_false(self):
        robot = SimpleNamespace(
            _gripper_force_limiter=object(), _gripper_index=None
        )
        assert set_gripper_force_limit(robot, 30.0) is False

    def test_map_less_limiter_returns_false(self):
        limiter = SimpleNamespace(gripper_force_torque_map=SimpleNamespace(func=None))
        robot = SimpleNamespace(_gripper_force_limiter=limiter, _gripper_index=6)
        assert set_gripper_force_limit(robot, 30.0) is False

    def test_nonpositive_force_raises(self):
        robot = SimpleNamespace(_gripper_force_limiter=None, _gripper_index=6)
        with pytest.raises(ValueError):
            set_gripper_force_limit(robot, 0.0)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class TestSupervisor:
    def test_refuses_construction_unless_enabled(self):
        fake = FakeGripper()
        with pytest.raises(ValueError, match="opt-in"):
            GripperStallSupervisor(
                config=GripperStallSupervisorConfig(),  # enabled defaults False
                io=fake,
                motion=fake,
            )

    def test_clean_close_succeeds_without_recovery(self):
        fake = FakeGripper()
        sup = make_supervisor(fake)
        result = sup.supervised_close()
        assert result.success
        assert result.verdict == "closed"
        assert not result.held
        assert len(result.attempts) == 1
        assert result.attempts[0].verdict is CloseVerdict.CLOSED
        assert result.recoveries == 0
        assert fake.lifts == [] and fake.shifts == [] and fake.lowers == []

    def test_grasped_object_is_success_not_stall(self):
        # packet between the jaws: freeze at 0.05 with effort at the limit
        fake = FakeGripper(object_floor=0.05)
        sup = make_supervisor(fake)
        result = sup.supervised_close()
        assert result.success
        assert result.verdict == "closed"
        assert result.held
        assert result.recoveries == 0
        assert not result.stalled_at_least_once

    def test_friction_stall_recovers_on_second_attempt(self):
        fake = FakeGripper(stall_floor=0.6, clears_after_recoveries=1)
        sup = make_supervisor(fake)
        result = sup.supervised_close()
        assert result.success
        assert result.verdict == "stall_recovered"
        assert len(result.attempts) == 2
        assert result.attempts[0].verdict is CloseVerdict.STALLED
        assert result.attempts[0].final_pos == pytest.approx(0.6, abs=0.02)
        assert result.attempts[1].verdict is CloseVerdict.CLOSED
        assert result.recoveries == 1
        # recovery sequence: open command, lift, shift, lower
        rec = RecoveryConfig()
        assert fake.lifts == [rec.lift_height_m]
        assert fake.lowers == [rec.lift_height_m]
        assert fake.shifts == [rec.lateral_shift_m]
        # gripper was opened before lifting
        assert rec.open_pos in fake.commands

    def test_late_stall_detected_and_recovered(self):
        # jaws travel most of the stroke before snagging at 0.3
        fake = FakeGripper(stall_floor=0.3, clears_after_recoveries=1)
        sup = make_supervisor(fake)
        result = sup.supervised_close()
        assert result.success
        assert result.attempts[0].verdict is CloseVerdict.STALLED
        assert result.attempts[0].final_pos == pytest.approx(0.3, abs=0.02)
        assert result.recoveries == 1

    def test_persistent_stall_fails_after_max_attempts(self):
        fake = FakeGripper(stall_floor=0.6)  # never clears
        sup = make_supervisor(fake)
        result = sup.supervised_close()
        assert not result.success
        assert result.verdict == "stall_unrecovered"
        assert len(result.attempts) == 3  # max_attempts default
        assert all(a.verdict is CloseVerdict.STALLED for a in result.attempts)
        assert result.recoveries == 2  # attempts - 1
        assert len(fake.lifts) == 2 and len(fake.lowers) == 2
        # lateral shifts alternate sides: net offsets +s then -s => deltas +s, -2s
        s = RecoveryConfig().lateral_shift_m
        assert fake.shifts == pytest.approx([s, -2 * s])

    def test_dead_gripper_times_out_without_blind_retry(self):
        fake = FakeGripper(dead=True)
        sup = make_supervisor(fake, close_timeout_s=0.5)
        result = sup.supervised_close()
        assert not result.success
        assert result.verdict == "timeout"
        assert len(result.attempts) == 1  # no retries on a non-stall failure
        assert result.recoveries == 0
        assert fake.lifts == []

    def test_force_limit_applied_once_when_configured(self):
        fake = FakeGripper()
        calls = []

        def setter(newtons):
            calls.append(newtons)
            return True

        config = GripperStallSupervisorConfig(enabled=True, close_force_newtons=25.0)
        sup = GripperStallSupervisor(
            config=config,
            io=fake,
            motion=fake,
            force_limit_setter=setter,
            time_fn=fake.time_fn,
            sleep_fn=fake.sleep_fn,
        )
        result = sup.supervised_close()
        assert calls == [25.0]
        assert result.force_limit_applied == 25.0

    def test_force_limit_rejection_is_nonfatal(self):
        fake = FakeGripper()
        config = GripperStallSupervisorConfig(enabled=True, close_force_newtons=25.0)
        sup = GripperStallSupervisor(
            config=config,
            io=fake,
            motion=fake,
            force_limit_setter=lambda n: False,
            time_fn=fake.time_fn,
            sleep_fn=fake.sleep_fn,
        )
        result = sup.supervised_close()
        assert result.success
        assert result.force_limit_applied is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_are_disabled(self):
        assert GripperStallSupervisorConfig().enabled is False
        assert GripperStallSupervisorConfig.from_dict({}).enabled is False

    def test_from_dict_nested(self):
        cfg = GripperStallSupervisorConfig.from_dict(
            {
                "enabled": True,
                "close_force_newtons": 20.0,
                "detector": {"consecutive_samples": 10},
                "recovery": {"max_attempts": 2, "lift_height_m": 0.02},
            }
        )
        assert cfg.enabled is True
        assert cfg.close_force_newtons == 20.0
        assert cfg.detector.consecutive_samples == 10
        assert cfg.detector.pos_error_threshold == 0.15  # untouched default
        assert cfg.recovery.max_attempts == 2
        assert cfg.recovery.lift_height_m == 0.02

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown"):
            GripperStallSupervisorConfig.from_dict({"typo_key": 1})

    def test_example_yaml_loads_and_is_disabled(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            "yam",
            "gripper_stall_supervisor.yaml",
        )
        cfg = GripperStallSupervisorConfig.from_yaml(path)
        assert cfg.enabled is False  # the checked-in example must stay opt-out
        assert cfg.recovery.max_attempts == 3
