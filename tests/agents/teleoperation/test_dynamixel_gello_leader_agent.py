import logging
import time

import numpy as np
import pytest

from robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent import (
    DEFAULT_MOTOR_IDS,
    DynamixelGelloLeaderAgent,
)


class FakeDynamixelReader:
    """Fake reader for testing without hardware."""

    def __init__(self, positions_ticks=None, write_calls=None):
        if positions_ticks is None:
            positions_ticks = np.array([0] * 7)
        self._positions = np.asarray(positions_ticks)
        self._last_read_time = time.time()
        self.write_calls = [] if write_calls is None else write_calls
        self.fail_next = False
        self.close_calls = 0
        self.stale_seconds = None

    def get_positions(self) -> np.ndarray:
        self._last_read_time = time.time()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated read failure")
        return self._positions.copy()

    def seconds_since_last_read(self) -> float:
        if self.stale_seconds is not None:
            return float(self.stale_seconds)
        return time.time() - self._last_read_time

    def close(self) -> None:
        self.close_calls += 1

    def write(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))


def make_agent(reader, **kwargs):
    return DynamixelGelloLeaderAgent(reader=reader, **kwargs)


def action_pos(agent):
    return agent.act({})[agent.robot_name]["pos"]


def old_repo_map(raw_ticks, offsets=None, scales=None):
    raw_ticks = np.asarray(raw_ticks, dtype=np.float64)
    offsets = np.zeros(6, dtype=np.float64) if offsets is None else np.asarray(offsets, dtype=np.float64)
    scales = np.ones(6, dtype=np.float64) if scales is None else np.asarray(scales, dtype=np.float64)
    rad = ((raw_ticks[:6] + offsets) * scales) / 4096 * 2 * np.pi - np.pi
    return ((rad + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def test_action_shape_no_gripper():
    reader = FakeDynamixelReader(np.arange(7))
    agent = make_agent(reader, robot_name="left", include_gripper=False)

    action = agent.act({})
    pos = action["left"]["pos"]

    assert pos.shape == (6,)
    assert pos.dtype == np.float32
    assert reader.write_calls == []


def test_action_shape_with_gripper():
    reader = FakeDynamixelReader(np.arange(7))
    agent = make_agent(reader, robot_name="right", include_gripper=True)

    pos = agent.act({})["right"]["pos"]


    assert pos.shape == (7,)
    assert pos.dtype == np.float32
    assert reader.write_calls == []


def test_joint_sign_flip():
    ticks = np.array([4096, 2048, 1024, -1024, -2048, -4096, 0])
    reader = FakeDynamixelReader(ticks)
    agent = make_agent(reader, joint_signs=[-1, -1, -1, -1, -1, -1])

    expected = -old_repo_map(ticks)

    np.testing.assert_allclose(action_pos(agent), expected, rtol=1e-6)


def test_joint_offsets_rad():
    ticks = np.array([2048] * 7)
    offsets = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    reader = FakeDynamixelReader(ticks)
    agent = make_agent(reader, joint_offsets_rad=offsets)

    np.testing.assert_allclose(action_pos(agent), np.array(offsets, dtype=np.float32), rtol=1e-6)


def test_tick_to_radian_conversion():
    ticks = np.array([4096, 2048, 1024, 512, 256, 128, 0])
    reader = FakeDynamixelReader(ticks)
    agent = make_agent(reader)

    expected = old_repo_map(ticks)

    np.testing.assert_allclose(action_pos(agent), expected, rtol=1e-6)


def test_tick_offset_scale_and_wrap_to_pi():
    ticks = np.array([4096, 2048, 1024, 512, 256, 128, 0])
    offsets = [4, -1934, -502, -976, 2091, -32]
    scales = [1, 1, -1, 1, 1, -1]
    reader = FakeDynamixelReader(ticks)
    agent = make_agent(reader, joint_offsets_ticks=offsets, joint_scales=scales)

    expected = old_repo_map(ticks, offsets=offsets, scales=scales)

    np.testing.assert_allclose(action_pos(agent), expected, rtol=1e-6)
    assert np.all(action_pos(agent) >= -np.pi)
    assert np.all(action_pos(agent) <= np.pi)


def test_stale_read_warning(caplog):
    reader = FakeDynamixelReader(np.zeros(7))
    reader.stale_seconds = 2.5
    agent = make_agent(reader, stale_timeout_s=1.0)

    with caplog.at_level(logging.WARNING):
        agent.act({})

    assert "stale" in caplog.text.lower() or "no dynamixel reads" in caplog.text.lower()
    assert "2.50" in caplog.text or "2.5" in caplog.text


def test_no_write_on_init():
    reader = FakeDynamixelReader(np.zeros(7))

    make_agent(reader)

    assert reader.write_calls == []


def test_no_write_on_act():
    reader = FakeDynamixelReader(np.zeros(7))
    agent = make_agent(reader)

    agent.act({})

    assert reader.write_calls == []


def test_no_write_on_close():
    reader = FakeDynamixelReader(np.zeros(7))
    agent = make_agent(reader)


    agent.close()

    assert reader.close_calls == 1
    assert reader.write_calls == []


@pytest.mark.parametrize(("include_gripper", "shape"), [(False, (6,)), (True, (7,))])
def test_action_spec_shape(include_gripper, shape):
    reader = FakeDynamixelReader(np.zeros(7))
    agent = make_agent(reader, robot_name="left", include_gripper=include_gripper)

    spec = agent.action_spec()["left"]["pos"]

    assert spec.shape == shape
    assert spec.dtype == np.float32


def test_partial_read_failure():
    initial_ticks = np.array([4096, 2048, 1024, 512, 256, 128, 0])
    reader = FakeDynamixelReader(initial_ticks)
    agent = make_agent(reader)
    previous = action_pos(agent)

    reader._positions = np.array([999999] * 7)
    reader.fail_next = True
    pos = action_pos(agent)

    np.testing.assert_allclose(pos, previous)
    assert np.all(np.isfinite(pos))


def test_partial_read_failure_before_success_returns_finite_zero():
    reader = FakeDynamixelReader(np.array([999999] * 7))
    reader.fail_next = True
    agent = make_agent(reader)

    pos = action_pos(agent)

    np.testing.assert_allclose(pos, np.zeros(6, dtype=np.float32))
    assert np.all(np.isfinite(pos))


def test_gripper_open_closed_tick_mapping():
    ticks = np.zeros(7)
    ticks[6] = 2280
    reader = FakeDynamixelReader(ticks)
    agent = make_agent(
        reader,
        include_gripper=True,
        gripper_open_ticks=2280,
        gripper_closed_ticks=1670,
        gripper_scale=1.0,
    )

    open_pos = action_pos(agent)[-1]

    ticks[6] = 1670
    reader._positions = ticks
    closed_pos = action_pos(agent)[-1]

    ticks[6] = (2280 + 1670) / 2
    reader._positions = ticks
    half_open = action_pos(agent)[-1]

    assert open_pos == pytest.approx(1.0)
    assert closed_pos == pytest.approx(0.0)
    assert half_open == pytest.approx(0.5)
    assert reader.write_calls == []


def test_default_motor_ids_are_real_left_leader_ids():
    reader = FakeDynamixelReader(np.zeros(7))
    agent = make_agent(reader)

    assert DEFAULT_MOTOR_IDS == [1, 2, 3, 4, 5, 6, 7]
    assert agent.motor_ids == [1, 2, 3, 4, 5, 6, 7]
