#!/usr/bin/env python3
"""Is the gripper channel of a running teleop/recording session actually moving?

WHY THIS EXISTS
---------------
The 45 s squeeze gate at bring-up protected exactly one thing worth protecting:
TRAINING DATA.  A recorded corpus whose gripper channel never moves cannot
train a grasp policy, because the exporter locates grasp windows BY the
gripper closing — it would find none.  On 2026-09-01 eleven minutes of
otherwise-perfect video were useless for that reason.

But the gate proved it at the wrong moment and about the wrong signal: it read
the SERVO, before the session existed, and then trusted that forever.  This
watches the actual thing that gets recorded — the `<node>/joint_pos` messages
on the bus — while the operator teleoperates, and says so out loud if the
channel is flat.  That is strictly more of the truth: it also catches a
gripper that dies mid-session, a calibration that clamps the channel to a
constant, and a leader agent replaying its last reading
(dynamixel_gello_leader_agent.act() falls back to `self._last_pos`).

THE PREDICATE (why it does not cry wolf)
----------------------------------------
A flat gripper on a handle nobody is holding is NORMAL.  So a warning needs
two facts at once:

    the ARM joints have moved   (the operator is demonstrably driving)
  AND the GRIPPER channel has not moved at all, over the same window

Only then is the channel suspect.  Until the arm moves, this says nothing.

IT WARNS, IT NEVER BLOCKS.  Nothing it does can stop, kill or slow a session;
it holds no device and opens no serial port.  It is a SUB socket and a printf.

USAGE
    tools/gripper_watchdog.py --node gello_right          # until the channel moves
    tools/gripper_watchdog.py --node gello_left --secs 300
    tools/gripper_watchdog.py --node gello_right --once   # one verdict, then exit

EXIT
    0  the gripper channel was seen to move (or the window ended quietly with
       the arm never moving — nothing to judge)
    1  the arm moved and the gripper channel stayed flat  → the corpus being
       recorded right now has no grasp windows in it
    2  no messages on <node>/joint_pos at all — the leader node is not
       publishing (see runbook/leader-handle-identity-crossed.md: a dead
       handle still publishes replayed values, so silence means the NODE is
       down, not the handle)
"""
from __future__ import annotations

import argparse
import time

import zmq

from robots_realtime.runtime.transport.serialization import unpack
from robots_realtime.runtime.transport.subscriber import DEFAULT_SUB_PORT

RED = "\033[31m"
YLW = "\033[33m"
GRN = "\033[32m"
OFF = "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", required=True, help="gello node name, e.g. gello_right")
    ap.add_argument("--port", type=int, default=DEFAULT_SUB_PORT)
    ap.add_argument("--secs", type=float, default=600.0, help="give up watching after this")
    ap.add_argument("--window", type=float, default=20.0,
                    help="seconds of demonstrated arm motion before a flat gripper is a warning")
    ap.add_argument("--arm-move", type=float, default=0.15,
                    help="rad of arm travel that counts as 'the operator is driving'")
    ap.add_argument("--grip-move", type=float, default=0.05,
                    help="normalised gripper travel (0..1) that counts as 'the channel works'")
    ap.add_argument("--once", action="store_true", help="print one verdict and exit")
    ap.add_argument("--quiet-ok", action="store_true", help="say nothing when healthy")
    args = ap.parse_args()

    topic = f"{args.node}/joint_pos"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode())

    t0 = time.time()
    n = 0
    arm_lo: list[float] | None = None
    arm_hi: list[float] | None = None
    grip_lo = grip_hi = None
    warned = False
    driving_since: float | None = None

    try:
        while time.time() - t0 < args.secs:
            if not sock.poll(200):
                pass
            else:
                while True:
                    try:
                        parts = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(parts) < 2:
                        continue
                    try:
                        data = (unpack(parts[1]) or {}).get("data") or {}
                    except Exception:
                        continue
                    # joint_pos arrives as a numpy array, so `if not q` raises
                    # "truth value of an array is ambiguous" — convert first.
                    q = data.get("joint_pos")
                    if q is None:
                        continue
                    q = [float(x) for x in q]
                    if len(q) < 2:
                        continue
                    n += 1
                    arm, grip = q[:-1], q[-1]
                    if arm_lo is None:
                        arm_lo, arm_hi = list(arm), list(arm)
                        grip_lo = grip_hi = grip
                    else:
                        arm_lo = [min(a, b) for a, b in zip(arm_lo, arm)]
                        arm_hi = [max(a, b) for a, b in zip(arm_hi, arm)]
                        grip_lo, grip_hi = min(grip_lo, grip), max(grip_hi, grip)

            if arm_lo is None:
                continue
            arm_span = max(h - l for h, l in zip(arm_hi, arm_lo))
            grip_span = grip_hi - grip_lo

            if grip_span >= args.grip_move:
                if not args.quiet_ok:
                    print(f"{GRN}  ✓ {topic}: gripper channel moves "
                          f"(travel {grip_span:.2f} of 0..1 over {n} messages){OFF}", flush=True)
                return 0

            if arm_span >= args.arm_move:
                driving_since = driving_since or time.time()
                if time.time() - driving_since >= args.window and not warned:
                    print(f"\n{RED}  ✗ GRIPPER CHANNEL FLAT WHILE YOU ARE DRIVING THE ARM{OFF}",
                          flush=True)
                    print(f"{RED}    {topic}: arm has travelled {arm_span:.2f} rad over the last "
                          f"{args.window:.0f} s, gripper has travelled {grip_span:.3f} (of 0..1).{OFF}",
                          flush=True)
                    print(f"{YLW}    Anything you record from now has NO grasp windows — the "
                          f"exporter finds grasps BY the gripper closing.{OFF}", flush=True)
                    print(f"{YLW}    Triage, in this order (runbook/leader-handle-identity-crossed.md):{OFF}",
                          flush=True)
                    print(f"{YLW}      1. wrong handle — you are squeezing a trigger that is not on "
                          f"this port{OFF}", flush=True)
                    print(f"{YLW}      2. latched torque — stop the session, then "
                          f"tools/clear_leader_torque.py <port> <ids>{OFF}", flush=True)
                    print(f"{YLW}      3. mechanical — the trigger no longer turns its servo{OFF}",
                          flush=True)
                    warned = True
                    if args.once:
                        return 1
            # --once gives up only if the arm never started moving: once it has,
            # the verdict is still pending and cutting out here would report
            # "nothing to judge" on exactly the run that needed judging.
            if args.once and driving_since is None and time.time() - t0 > args.window:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)

    if n == 0:
        print(f"{RED}  ✗ nothing published on {topic} in {args.secs:.0f}s — the leader NODE is "
              f"not running. (A dead HANDLE still publishes replayed values, so silence is "
              f"the node, not the hardware.){OFF}", flush=True)
        return 2
    if warned:
        return 1
    if not args.quiet_ok:
        print(f"{YLW}  · {topic}: window ended with the arm barely moved — nothing to judge "
              f"({n} messages, gripper travel {(grip_hi - grip_lo):.3f}){OFF}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
