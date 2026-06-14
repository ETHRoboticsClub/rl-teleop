"""Verify AgentNode applies safety guardrails correctly per session mode."""

import numpy as np
import sys
import types
from unittest.mock import MagicMock, patch


class _FakeZmqAgain(Exception):
    pass


class _FakeZmqContext:
    @classmethod
    def instance(cls):
        return cls()

    def socket(self, *_args, **_kwargs):
        return MagicMock()


_fake_zmq = types.ModuleType("zmq")
_fake_zmq.Again = _FakeZmqAgain
_fake_zmq.Context = _FakeZmqContext
_fake_zmq.NOBLOCK = 1
_fake_zmq.PUB = 1
_fake_zmq.SUB = 2
_fake_zmq.SUBSCRIBE = 3
_fake_zmq.proxy = MagicMock()
sys.modules.setdefault("zmq", _fake_zmq)

_fake_msgpack = types.ModuleType("msgpack")
_fake_msgpack.packb = MagicMock(return_value=b"")
_fake_msgpack.unpackb = MagicMock(return_value={})
sys.modules.setdefault("msgpack", _fake_msgpack)

_fake_msgpack_numpy = types.ModuleType("msgpack_numpy")
_fake_msgpack_numpy.encode = MagicMock()
_fake_msgpack_numpy.decode = MagicMock()
sys.modules.setdefault("msgpack_numpy", _fake_msgpack_numpy)


def _safety(agent_type="teleop", left_bbox=None, right_bbox=None, acceleration_limit=0.25):
    arms = {}
    if left_bbox is not None:
        arms["left"] = {"bounding_box": left_bbox}
    if right_bbox is not None:
        arms["right"] = {"bounding_box": right_bbox}
    return {
        "mode": "real",
        "agent_type": agent_type,
        "arms": arms,
        "acceleration_limit": acceleration_limit,
    }


def _node(**kwargs):
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=MagicMock(), name="agent", **kwargs)
    node.name = "agent"
    node.publish = MagicMock(return_value=True)
    node.setup()
    return node

def test_teleop_applies_bbox_but_not_acceleration():
    """Teleop should only apply bbox, not acceleration limiting."""
    node = _node(
        safety=_safety(
            agent_type="teleop",
            left_bbox={"min": [-10.0] * 6, "max": [10.0] * 6},
            acceleration_limit=0.25,
        )
    )

    first = node._process_pos(np.zeros(6), arm_key="left")
    second = node._process_pos(np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0]), arm_key="left")
    out_of_bounds = node._process_pos(np.array([99.0, -99.0, 0.0, 0.0, 0.0, 0.0]), arm_key="left")

    np.testing.assert_allclose(first, np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(second, np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out_of_bounds, np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

def test_inference_applies_bbox_and_acceleration():
    """Inference should apply both bbox and acceleration limiting."""
    node = _node(
        safety=_safety(
            agent_type="inference",
            left_bbox={"min": [-1.0] * 6, "max": [1.0] * 6},
            acceleration_limit=0.25,
        )
    )

    first = node._process_pos(np.zeros(6), arm_key="left")
    second = node._process_pos(np.array([9.0, -9.0, 0.1, 0.0, 0.0, 0.0]), arm_key="left")
    third = node._process_pos(np.array([9.0, -9.0, 5.0, 0.0, 0.0, 0.0]), arm_key="left")

    np.testing.assert_allclose(first, np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(second, np.array([0.25, -0.25, 0.1, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(third, np.array([0.5, -0.5, 0.35, 0.0, 0.0, 0.0], dtype=np.float32))

def test_arm_key_path_applies_matching_arm_safety_config():
    """When arm_key is set, safety config for that arm should be applied."""
    node = _node(
        arm_key="right",
        safety=_safety(
            agent_type="teleop",
            left_bbox={"min": [0.0] * 6, "max": [1.0] * 6},
            right_bbox={"min": [-0.5] * 6, "max": [0.5] * 6},
        ),
    )

    node._publish_commands({"right": {"pos": np.array([2.0] * 6)}}, ts=123.0)

    node.publish.assert_called_once()
    topic, payload = node.publish.call_args.args[:2]
    assert topic == "joint_pos"
    np.testing.assert_allclose(payload["joint_pos"], np.array([0.5] * 6, dtype=np.float32))

def test_metadata_keys_record_chunk_images_are_not_guardrailed():
    """Metadata keys starting with _ should not be guardrailed."""
    chunk = {"left": np.array([99.0, -99.0])}
    images = {"cam_top": np.array([[255]], dtype=np.uint8)}
    agent = MagicMock(
        act=MagicMock(
            return_value={
                "pos": np.array([2.0] * 6),
                "_record": True,
                "_chunk": chunk,
                "_images": images,
            }
        )
    )
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(
            agent=agent,
            name="agent",
            safety=_safety(
                agent_type="teleop",
                left_bbox={"min": [-1.0] * 6, "max": [1.0] * 6},
            ),
        )
    node.name = "agent"
    node.publish = MagicMock(return_value=True)
    node.setup()

    node.step()

    node.publish.assert_any_call("record", {"record": True}, ts=node.publish.call_args_list[0].kwargs["ts"])
    assert node.publish.call_args_list[1].args[:2] == ("chunk", chunk)
    assert node.publish.call_args_list[2].args[:2] == ("image/cam_top", {"images": {"rgb": images["cam_top"]}})
    np.testing.assert_allclose(
        node.publish.call_args_list[3].args[1]["joint_pos"],
        np.array([1.0] * 6, dtype=np.float32),
    )


# ── Migration RED tests: no bbox guardrails in AgentNode ──────────────────


def test_agent_node_has_no_bbox_guardrails():
    """After migration, AgentNode should not have _bbox_guardrails attribute."""
    # After migration, _bbox_guardrails should not exist on AgentNode
    from robots_realtime.runtime.agent_node import AgentNode
    from robots_realtime.runtime.node import Node

    with patch.object(Node, "__init__", return_value=None):
        node = AgentNode(agent=MagicMock(), name="agent")
    node.name = "agent"
    node.publish = MagicMock(return_value=True)
    node.setup()

    # _bbox_guardrails should not exist after migration
    assert not hasattr(node, "_bbox_guardrails"), (
        "AgentNode should not have _bbox_guardrails after migration to Cartesian-only safety"
    )
