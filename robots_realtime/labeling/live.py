"""Live labeling — drive the cockpit's /state view as the operator works.

The cockpit polls  GET liveBase()/state -> {seeded, packets:[{part,name,comp,bbox}], ti}
and  POST /seed. So to make each step show up labeled live we run the SAME gripper
state machine online, and as grasps/releases are detected we advance ``ti`` and mark
packets done. Confirmation stays batched per-kit (design decision) — this is the live
*labeling* feed, not the confirm sweep.

Online normalization uses a running min/max (no episode percentiles available live),
oriented so the first sample is "open". The authoritative labels are still produced
offline by label_episode; this is the live preview + the cockpit_events.jsonl the
offline pass consumes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from robots_realtime.labeling import constants as C


@dataclass
class GripEvent:
    kind: str        # "close" | "release"
    t: float
    outcome: str | None = None   # for "release": success | slip | empty


class OnlineGripSegmenter:
    """Incremental hysteresis state machine over the gripper width.

    Feed samples with ``push(t, width_raw)``; it returns a GripEvent on a
    close→open ("release") or the committed open→close ("close"), else None.
    """

    def __init__(self, open_ref: float | None = None, closed_ref: float | None = None):
        self._open_ref = open_ref
        self._closed_ref = closed_ref
        self._lo = None
        self._hi = None
        self._oriented_flip = None
        self._closed = False
        self._t_close = 0.0
        self._hold_min = 1.0
        self._hold_sum = 0.0
        self._hold_n = 0
        self._committed = False   # emitted the "close" event for this interval

    def _norm(self, raw: float, first: bool) -> float:
        if self._open_ref is not None and self._closed_ref is not None:
            span = self._open_ref - self._closed_ref
            n = (raw - self._closed_ref) / span if abs(span) > 1e-9 else 0.0
        else:
            self._lo = raw if self._lo is None else min(self._lo, raw)
            self._hi = raw if self._hi is None else max(self._hi, raw)
            span = (self._hi - self._lo) if self._hi is not None else 0.0
            n = (raw - self._lo) / span if span > 1e-9 else 1.0
            if first:
                # first sample defines "open"; if it maps low, the signal is inverted
                self._oriented_flip = n < 0.5
            if self._oriented_flip:
                n = 1.0 - n
        return min(max(n, 0.0), 1.0)

    def push(self, t: float, width_raw: float) -> GripEvent | None:
        first = self._lo is None and self._open_ref is None
        w = self._norm(float(width_raw), first)
        ev: GripEvent | None = None
        if not self._closed:
            if w < C.GRIPPER_CLOSE_ENTER:
                self._closed = True
                self._t_close = t
                self._hold_min, self._hold_sum, self._hold_n = w, w, 1
                self._committed = False
        else:
            self._hold_min = min(self._hold_min, w)
            self._hold_sum += w
            self._hold_n += 1
            if not self._committed and t - self._t_close >= C.MIN_HOLD_S:
                self._committed = True
                ev = GripEvent("close", self._t_close)
            if w > C.GRIPPER_CLOSE_EXIT:
                if self._committed:
                    hold_mean = self._hold_sum / max(self._hold_n, 1)
                    if hold_mean < C.GRIPPER_EMPTY_CLOSE:
                        outcome = "empty"
                    elif self._hold_min < hold_mean - C.GRIPPER_SLIP_DROP:
                        outcome = "slip"
                    else:
                        outcome = "success"
                    ev = GripEvent("release", t, outcome)
                self._closed = False
        return ev


@dataclass
class LiveLabeler:
    """Holds the live kit state and turns gripper events into cockpit progress.

    Thread-safe: the feed thread calls push(); the HTTP server reads state().
    """
    open_ref: float | None = None
    closed_ref: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    packets: list[dict] = field(default_factory=list)
    ti: int = 0
    seeded: bool = False
    events: list[dict] = field(default_factory=list)   # live cockpit_events
    stage: str = "idle"

    def __post_init__(self):
        self._seg = OnlineGripSegmenter(self.open_ref, self.closed_ref)
        self._grasping = False

    def seed(self, packets: list[dict]) -> None:
        with self._lock:
            self.packets = [dict(p) for p in packets]
            for i, p in enumerate(self.packets):
                p.setdefault("status", "pending")
            self.ti = 0
            self.seeded = True
            self.stage = "reach"
            self.events = [{"t": time.monotonic(), "type": "seed",
                            "packets": len(self.packets)}]

    def advance(self) -> None:
        """Operator manually confirmed the current placement (cockpit 'Placed'
        button → POST /advance): mark the current packet placed and step ti."""
        with self._lock:
            if not self.seeded or self.ti >= len(self.packets):
                return
            p = self.packets[self.ti]
            p["status"] = "placed"
            self.events.append({"t": time.monotonic(), "type": "place_confirmed",
                                "bag_id": p.get("bag_id", self.ti + 1),
                                "comp": p.get("comp"), "outcome": "manual"})
            self.ti = min(self.ti + 1, len(self.packets))
            self.stage = "reach"

    def push(self, t: float, width_raw: float) -> None:
        with self._lock:
            if not self.seeded:
                return
            ev = self._seg.push(t, width_raw)
            if ev is None:
                return
            if ev.kind == "close":
                self._grasping = True
                self.stage = "grasp"
                self.events.append({"t": t, "type": "grasp", "bag_ti": self.ti})
            elif ev.kind == "release":
                self.stage = "place"
                if ev.outcome == "success" and self.ti < len(self.packets):
                    p = self.packets[self.ti]
                    p["status"] = "placed"
                    self.events.append({
                        "t": t, "type": "place_confirmed",
                        "bag_id": p.get("bag_id", self.ti + 1),
                        "comp": p.get("comp"), "outcome": "success",
                    })
                    self.ti = min(self.ti + 1, len(self.packets))
                    self.stage = "reach"
                else:  # slip/empty — stay on the same bag (regrasp coming)
                    self.events.append({"t": t, "type": "grasp_failed",
                                        "bag_ti": self.ti, "outcome": ev.outcome})
                self._grasping = False

    def state(self) -> dict:
        with self._lock:
            return {
                "seeded": self.seeded,
                "packets": [{"part": p.get("part"), "name": p.get("name"),
                             "comp": p.get("comp"), "bbox": p.get("bbox"),
                             "status": p.get("status", "pending")}
                            for p in self.packets],
                "ti": self.ti,
                "stage": self.stage,
                "done": sum(1 for p in self.packets if p.get("status") == "placed"),
                "total": len(self.packets),
            }
