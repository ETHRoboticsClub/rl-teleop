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
#
# Raised 0.20 → 0.50 on 2026-07-27, measured against the whole recorded corpus
# (23 episodes, 192 grip intervals) with tools/diagnose_live_gate.py:
#
#     hold duration    n     p50      p75      p90
#     ─────────────────────────────────────────────
#     outcome=success  65    4.445s   6.999s   9.064s
#     outcome=empty   124    0.300s   0.427s   1.169s
#
# Real picks and twitches are separated by an order of magnitude, and the old
# 0.20 s let 124 sub-second open/close twitches into the event stream — they
# swamped the cockpit's event log and each one reset the grasp state machine
# mid-approach. Sweep of the cutoff:
#
#     0.3s → removes 62/124 twitches, loses 1/65 real grasps
#     0.5s → removes 98/124 twitches, loses 1/65 real grasps   ← chosen
#     0.6s → removes 101/124,         loses 3/65
#     0.8s → removes 106/124,         loses 7/65
#
# 0.5 s is the knee: 79 % of the noise for the same single lost grasp (that one
# is a 0.275 s pick, which is faster than a human teleop grasp can physically be
# and is itself probably a mislabel). Re-run the sweep before changing this.
MIN_HOLD_S = 0.50
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

# --- grasp workspace ---------------------------------------------------------
# Grasps whose end-effector x is below this are outside the packet mat and are
# not training data. Measured 2026-07-28 over the 81 successful grasps in the
# corpus: sorted x has a 176 mm empty band from 0.178 to 0.354, with 4 grasps
# below it. Those 4 also sit at z 0.162-0.226 against a corpus mean of 0.120 --
# near AND high, i.e. mid-air or mislabelled, not table grasps. They come from
# two different episodes (195935, 212500), so this is not one bad session.
#
# A fixed gate, not a 3-sigma rule: a statistical cut moves as the corpus grows,
# so the same recording would be in one week and out the next. Any threshold in
# [0.20, 0.30] drops exactly these 4; 0.25 sits mid-gap. Metres, robot base frame.
GRASP_WORKSPACE_X_MIN = 0.25
