"""ZMQ subscriber — keeps only the latest message per topic.

Drains the socket in a background thread so callers never block on
socket I/O.  get_data() is always O(1).
"""

from __future__ import annotations

import threading

import zmq

from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
from robots_realtime.runtime.transport.serialization import unpack


class Subscriber:
    """Subscribes to one or more topic prefixes on the XPUB/XSUB broker.

    Maintains a latest-per-topic buffer updated by a background drain thread.
    Intermediate messages from fast producers are silently dropped — callers
    always get the freshest data.

    Args:
        topics: List of full topic strings to subscribe to,
                e.g. ``["gello_left/joint_pos", "camera_0/rgb"]``.
                An empty list subscribes to everything (use with care).
        host: Broker host.
        port: Broker XPUB port (subscribers connect here).
    """

    def __init__(
        self,
        topics: list[str],
        host: str = "127.0.0.1",
        port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self._latest: dict[str, dict] = {}
        self._lock = threading.Lock()
        # A SECOND LOCK, FOR THE SOCKET ITSELF. ZeroMQ sockets are not
        # thread-safe, and this class hands the same socket to two threads: the
        # background _drain_loop, and whichever thread calls drain()/drain_one().
        # Node._run_subscriber_driven() does exactly that on every tick, so any
        # subscriber-driven node had two threads inside recv_multipart() at once.
        #
        # The symptom is not a clean error. Interleaved recv_multipart() calls
        # tear message framing apart, so one caller receives a buffer containing
        # somebody else's bytes and msgpack raises
        #     ExtraData: unpack(b) received extra data
        # from deep inside deserialization, with nothing pointing at threading.
        # Observed 2026-08-12: it killed an ACT policy agent node ~15 s after
        # start, every time, once the subscribed topics included camera frames
        # (bigger messages widen the interleaving window).
        self._sock_lock = threading.Lock()
        self._stop = threading.Event()

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")

        if not topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for t in topics:
                self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())

        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="SubDrain"
        )
        self._thread.start()

    def _recv_pending(self) -> list[list[bytes]]:
        """Take every message currently queued, holding the socket lock.

        The lock is the point — see _sock_lock in __init__. Decoding happens
        OUTSIDE the lock so a large msgpack payload never blocks the other
        thread's receives.
        """
        out: list[list[bytes]] = []
        with self._sock_lock:
            try:
                while True:
                    out.append(self._sock.recv_multipart(zmq.NOBLOCK))
            except zmq.Again:
                pass
            except zmq.ZMQError:
                pass
        return out

    def _store(self, messages: list[list[bytes]]) -> int:
        stored = 0
        for parts in messages:
            if len(parts) < 2:
                continue
            try:
                envelope = unpack(parts[1])
            except Exception:
                # A single undecodable message must not kill the drain thread and
                # take the node with it.
                continue
            with self._lock:
                self._latest[parts[0].decode()] = envelope
            stored += 1
        return stored

    def _drain_loop(self) -> None:
        """Background thread: drain socket and keep latest per topic."""
        while not self._stop.is_set():
            messages = self._recv_pending()
            if not messages:
                self._stop.wait(0.002)
                continue
            self._store(messages)

    def drain_one(self, timeout_ms: int = 50) -> bool:
        """Block up to *timeout_ms* waiting for any new message."""
        import time
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest:
                    return True
            time.sleep(0.001)
        return False

    def get_latest(self, topic: str) -> dict | None:
        """Return the most recently received envelope for *topic*, or None."""
        with self._lock:
            return self._latest.get(topic)

    def get_data(self, topic: str) -> dict | None:
        """Convenience: return just the ``data`` field of the latest envelope."""
        with self._lock:
            env = self._latest.get(topic)
        return env["data"] if env is not None else None

    def get_timestamp(self, topic: str) -> float | None:
        """Return the hardware timestamp of the latest message on *topic*."""
        with self._lock:
            env = self._latest.get(topic)
        return env["ts"] if env is not None else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        with self._sock_lock:
            self._sock.close(linger=0)

    def drain(self) -> None:
        """Drain all pending messages from the socket (non-blocking).

        Safe to call from a different thread than the background drain loop —
        both go through _recv_pending(), which serialises socket access.
        """
        self._store(self._recv_pending())
