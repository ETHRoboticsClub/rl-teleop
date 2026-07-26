"""Tuning constants for auto-labeling.

Gripper units differ across rigs and are not known a-priori (the recorded value
is a raw motor joint position, index 6 in the 7-DoF joint vector). So the
segmentation logic NEVER hardcodes an absolute width. It normalizes the gripper
signal per-episode to [0, 1] where 0 = fully closed, 1 = fully open, then applies
these fractional thresholds. This keeps the labeler unit-agnostic and robust to
open/closed calibration drift.
"""
from __future__ import annotations

# --- gripper segmentation (all fractions of the per-episode open→closed range) -
# Hysteresis (two thresholds) so sensor jitter near the boundary can't emit a
# storm of open/close events. Enter "closing" below CLOSE_ENTER, leave it above
# CLOSE_EXIT.
GRIPPER_CLOSE_ENTER = 0.45   # below this normalized width → gripper is closing
GRIPPER_CLOSE_EXIT = 0.60    # above this → gripper is opening/open
# A grasp must actually hold something: the closed width has to sit ABOVE this
# (i.e. not fully closed), otherwise the gripper closed on empty air.
GRIPPER_EMPTY_CLOSE = 0.08   # closed to < 8% of range = closed on nothing
# Slip/drop: while holding a bag, if the width collapses by more than this
# fraction below the grasp-hold width (toward fully closed), the bag is gone.
GRIPPER_SLIP_DROP = 0.15

# A grasp/release must persist at least this long to count (debounce brief
# adjustment twitches). Seconds.
MIN_HOLD_S = 0.20
# A grasp must be followed by a lift within this window to be a real grasp
# (a close with no lift is an adjustment, not a pick). Seconds.
LIFT_WINDOW_S = 2.0
# Minimum end-effector rise (metres, robot Z-up) to count as a "lift".
MIN_LIFT_M = 0.03
# Minimum horizontal (XY) distance the EE must travel between grasp and release
# for a grasp to count as a real pick-AND-place. A success grasp that closes and
# re-opens near the pick without transporting (a re-grip / fumble at the box) is
# NOT a placement — gating on this removes the "released-at-pick" false placements.
# 0.0 disables the gate (library default, back-compat); the real kitting pipeline
# passes ~0.10 (box1→box2 is always >=~0.15m apart).
MIN_TRANSPORT_M = 0.10

# --- placement ---------------------------------------------------------------
# A release counts as "in" a compartment if the end-effector's XY at release is
# within the compartment polygon expanded by this margin (metres). Small slop
# for the offset between the wrist link (link_6, what FK gives) and the actual
# release point.
IN_REGION_MARGIN_M = 0.02

# --- fusion / clock ----------------------------------------------------------
# A cockpit event whose timestamp is more than this far outside the episode's
# [start, end] window is rejected (clock desync guard). Seconds.
CLOCK_WINDOW_SLACK_S = 1.0

# --- FK ----------------------------------------------------------------------
# The arm has 6 revolute joints (joint1..joint6); the 7th recorded value is the
# gripper. FK is computed to this link. The real grasp point is offset from the
# wrist by the gripper length; callers may add a tool offset.
FK_EE_LINK = "link_6"
N_ARM_JOINTS = 6
GRIPPER_JOINT_INDEX = 6
