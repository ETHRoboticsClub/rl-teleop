"""Gripper-width segmentation → grasp / release / slip events.

No gripper force sensor exists, so grasp and slip are read from the gripper
WIDTH (joint position, index 6) alone, plus end-effector height for lift
confirmation. The raw width unit and sign are unknown, so we:

  1. Normalize per-episode to [0, 1] via robust percentiles.
  2. Orient so 1 = open, 0 = fully closed, using the fact that the gripper
     starts open (first sample ≈ open).
  3. Run a hysteresis state machine (two thresholds) so jitter near the
     boundary can't emit a storm of open/close events.

A closed interval is classified:
  empty   — closed on nothing (hold width ≈ 0)
  slip    — held a bag, then width collapsed toward closed (bag fell)
  success — held to a normal release
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots_realtime.labeling import constants as C


@dataclass
class GripInterval:
    """One closed-gripper interval (a grasp attempt candidate)."""
    t_close: float
    t_open: float | None       # None if still closed at episode end
    hold_norm: float           # steady (median) normalized width while closed
    min_norm: float            # minimum width during hold (slip signal)
    outcome: str               # success | slip | empty
    lifted: bool | None        # True/False if ee_z given, else None


def normalize_width(width_raw, open_ref: float | None = None,
                    closed_ref: float | None = None) -> np.ndarray:
    """Map raw gripper width to [0, 1] with 1 = open, 0 = closed.

    Prefer the gripper's KNOWN physical limits (``open_ref``/``closed_ref`` from
    the robot config) so a bag-thickness hold normalizes to its true fraction.
    Without them, fall back to episode percentiles + auto-orientation — correct
    only when the episode actually spans the full open→closed range.
    """
    w = np.asarray(width_raw, dtype=float)
    if w.size == 0:
        return w
    if open_ref is not None and closed_ref is not None and abs(open_ref - closed_ref) > 1e-9:
        return np.clip((w - closed_ref) / (open_ref - closed_ref), 0.0, 1.0)
    lo, hi = np.percentile(w, [2, 98])
    if hi - lo < 1e-9:
        return np.zeros_like(w)          # gripper never moved
    norm = np.clip((w - lo) / (hi - lo), 0.0, 1.0)
    # Gripper starts open; if the first sample sits at the low end, the raw
    # signal is inverted (open = low raw) so flip it.
    if float(norm[0]) < 0.5:
        norm = 1.0 - norm
    return norm


def classify_hold(hold_vals: np.ndarray, t_close: float, t_arr: np.ndarray,
                  ee_z: np.ndarray | None) -> tuple[float, float, str, bool | None]:
    """Classify one closed-gripper hold → (hold_norm, min_norm, outcome, lifted).

    PUBLIC because the LIVE cockpit labeler (live.OnlineGripSegmenter) calls it on
    its accumulated hold buffer. Both labelers running the identical classifier is
    the only reliable way to keep them agreeing: when the live path had its own
    copy, it used the mean instead of the median and measured lift over the whole
    hold instead of LIFT_WINDOW_S, and a single 8-second hold then read
    lifted=True live / False offline. Do not fork this.
    """
    hold_norm = float(np.median(hold_vals))
    min_norm = float(np.min(hold_vals))
    slipped = min_norm < hold_norm - C.GRIPPER_SLIP_DROP

    # Did the end-effector lift during the hold? (bag picked up)
    lifted: bool | None = None
    if ee_z is not None and ee_z.size == t_arr.size:
        window = (t_arr >= t_close) & (t_arr <= t_close + C.LIFT_WINDOW_S)
        if window.any():
            z0 = float(np.interp(t_close, t_arr, ee_z))
            lifted = bool(float(np.max(ee_z[window])) - z0 >= C.MIN_LIFT_M)

    # Flat kitting bags let the gripper close almost fully even when holding one, so a
    # width-only test reads "empty" for real picks (no force sensor to tell them apart).
    # A LIFT is unambiguous evidence the grasp held an object → it is never empty.
    if lifted:
        outcome = "slip" if slipped else "success"
    elif hold_norm < C.GRIPPER_EMPTY_CLOSE:
        outcome = "empty"
    elif slipped:
        outcome = "slip"
    else:
        outcome = "success"
    return hold_norm, min_norm, outcome, lifted


def detect_grip_intervals(times, width_raw, ee_z=None,
                          open_ref: float | None = None,
                          closed_ref: float | None = None) -> list[GripInterval]:
    """Hysteresis state machine over the normalized gripper width."""
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return []
    w = normalize_width(width_raw, open_ref=open_ref, closed_ref=closed_ref)
    z = np.asarray(ee_z, dtype=float) if ee_z is not None else None

    intervals: list[GripInterval] = []
    closed = False
    t_close = 0.0
    hold: list[float] = []

    for i in range(t.size):
        if not closed:
            if w[i] < C.GRIPPER_CLOSE_ENTER:
                closed, t_close, hold = True, float(t[i]), [float(w[i])]
        else:
            hold.append(float(w[i]))
            if w[i] > C.GRIPPER_CLOSE_EXIT:
                hn, mn, oc, lifted = classify_hold(np.asarray(hold), t_close, t, z)
                intervals.append(GripInterval(t_close, float(t[i]), hn, mn, oc, lifted))
                closed = False
    if closed:
        hn, mn, oc, lifted = classify_hold(np.asarray(hold), t_close, t, z)
        intervals.append(GripInterval(t_close, None, hn, mn, oc, lifted))

    # Debounce: drop intervals shorter than MIN_HOLD_S (adjustment twitches).
    end = float(t[-1])
    return [iv for iv in intervals
            if (iv.t_open if iv.t_open is not None else end) - iv.t_close >= C.MIN_HOLD_S]


# ── Transport gate ────────────────────────────────────────────────────────────
# Shared by the OFFLINE labeler (fuse._transported) and the LIVE cockpit labeler
# (live.LiveLabeler). ONE implementation on purpose: these two segmenters have
# already drifted apart once (live had no lift check), and a threshold rule that
# exists in two places will drift again.
#
# Measured over 46 recorded grip intervals, the two populations are cleanly
# bimodal with an 8.7 cm empty gap, so the exact threshold is not sensitive
# (8/10/12/14 cm all classify identically):
#
#     re-grasp / fumble at the pick        real pick → place
#     ●●●●●●●●●●●●●●●●●● n=18              ●●●●●●●●●●●●●●●● n=28
#     0.1 cm ....... 5.8 cm       14.5 cm ....... 48.1 cm
#                       └──── 8.7 cm empty ────┘
#                                ▲
#                       C.MIN_TRANSPORT_M = 10 cm

def transport_distance_m(pose_a, pose_b) -> float | None:
    """Horizontal (XY) distance between two EE poses. None if either is missing."""
    if pose_a is None or pose_b is None:
        return None
    dx = float(pose_b[0]) - float(pose_a[0])
    dy = float(pose_b[1]) - float(pose_a[1])
    return (dx * dx + dy * dy) ** 0.5


def transport_ok(distance_m: float | None, min_transport_m: float) -> bool:
    """The threshold rule itself, expressed exactly once.

    FAILS OPEN (True) when the gate is disabled or the distance is unknown. A
    missing measurement must never silently freeze the kit pointer; callers
    surface that case as a visible warning instead.
    """
    if min_transport_m <= 0.0 or distance_m is None:
        return True
    return distance_m >= min_transport_m


def transported(pose_a, pose_b, min_transport_m: float) -> bool:
    """Did the EE travel far enough grasp→release to be a real placement?

    Pose-pair form, used by the OFFLINE labeler which has both poses in hand. The
    LIVE labeler measures the distance incrementally and calls transport_ok
    directly — both funnel through the same threshold rule.
    """
    return transport_ok(transport_distance_m(pose_a, pose_b), min_transport_m)
