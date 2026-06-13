# Lazy imports to avoid pulling zmq/pyrealsense2 at package-level import time.
# Safety config/guardrails are intentionally decoupled from Node dependencies.
def __getattr__(name):
    if name == "Node":
        from robots_realtime.runtime.node import Node as _Node
        return _Node
    if name == "ProcessHost":
        from robots_realtime.runtime.node import ProcessHost as _ProcessHost
        return _ProcessHost
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Node", "ProcessHost"]
