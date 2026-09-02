#!/usr/bin/env python3
"""Which PHYSICAL handle is on this serial port?  Wiggle it and watch.

WHAT THIS IS FOR
The GELLO leaders' udev names are CROSSED relative to where the handles sit on
the bench: on 2026-09-02 `/dev/leader-right` was proven to be wired to the
PHYSICALLY LEFT handle, and `/dev/leader-left` to the other one.  That confusion
cost two hours -- five bring-ups died on "GRIPPER IS FLAT" because the prompt
said "squeeze the LEFT handle" while the gate was reading the port belonging to
the other handle.  Neither handle nor gripper was broken.

There is no software signal that answers "which side is this?" -- the two FTDI
adapters have different serials but nothing ties a serial to a bench position,
and both handles report plausible joint values.  The only reliable oracle is to
make one MOVE and look at it.

THE TEST
Enables torque on motors 13 and 14 of the given port, oscillates them +/-60
ticks around wherever they currently are for 40 s, then disables torque and
restores the original goal position.  +/-60 ticks is ~5 deg: unmistakable to the
eye, far too small to hurt anyone or the handle, and easily overpowered by hand.

    WATCH WHICH HANDLE TWITCHES.  That handle is the one on --port.

USAGE
    ./.venv/bin/python3 tools/identify_leader_handle.py --port /dev/leader-right
    ./.venv/bin/python3 tools/identify_leader_handle.py --port /dev/leader-left \
        --ids 6,7 --amplitude 60 --secs 40

SAFETY
- This touches the LEADER handle only.  It opens a serial port and nothing else:
  no CAN channel, no follower, no camera.  The follower arm cannot move from
  here.
- It REFUSES to run while :8792 or :8794 is listening.  A live rr-session owns
  the leader serial, and two writers on one Dynamixel bus produce garbage; worse,
  a session that is mid-teleop would stream this wiggle straight to the follower.
  Stop the session first.
- Torque is disabled in a finally: block, so Ctrl-C leaves the handle limp and
  free, exactly as it was before.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

# Dynamixel X-series control table (same addresses the leader agent uses --
# robots_realtime/agents/teleoperation/dynamixel_gello_leader_agent.py).
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ON, TORQUE_OFF = 1, 0

BUSY_PORTS = (8792, 8794)


def listening_ports() -> set[int]:
    """TCP ports in LISTEN on this box, as ints. Empty set if ss is unavailable."""
    try:
        out = subprocess.run(
            ["ss", "-tlnH"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return set()
    ports = set()
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        _, _, port = local.rpartition(":")
        if port.isdigit():
            ports.add(int(port))
    return ports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="e.g. /dev/leader-right")
    ap.add_argument("--ids", default="13,14",
                    help="motor ids to wiggle (default 13,14 -- the wrist/gripper "
                         "pair of the left leader chain; use 6,7 for a 1-7 chain)")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--amplitude", type=int, default=60, help="ticks, peak (default 60 ~ 5 deg)")
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--hz", type=float, default=1.0, help="oscillation frequency")
    args = ap.parse_args()

    busy = sorted(set(BUSY_PORTS) & listening_ports())
    if busy:
        print(
            f"REFUSING: {', '.join(str(p) for p in busy)} is listening -- a live "
            f"rr-session owns a leader serial.\n"
            f"Two writers on one Dynamixel bus produce garbage, and a teleop session "
            f"would\nstream this wiggle to the follower arm. Stop the session, then "
            f"re-run.",
            file=sys.stderr,
        )
        return 2

    from dynamixel_sdk import PacketHandler, PortHandler  # imported late: needs the venv

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    ph = PortHandler(args.port)
    pk = PacketHandler(2.0)
    if not ph.openPort():
        print(f"cannot open {args.port}", file=sys.stderr)
        return 3
    ph.setBaudRate(args.baud)

    home: dict[int, int] = {}
    for mid in ids:
        pos, res, err = pk.read4ByteTxRx(ph, mid, ADDR_PRESENT_POSITION)
        if res != 0 or err != 0:
            print(f"motor {mid} does not answer on {args.port} (comm={res} err={err}) -- "
                  f"if NOTHING answers, that handle's power brick is off.", file=sys.stderr)
            continue
        home[mid] = pos

    if not home:
        ph.closePort()
        print(f"0/{len(ids)} servos answered on {args.port}: the handle is electrically "
              f"dead (unpowered brick or a cable out at motor 1). Nothing to wiggle.",
              file=sys.stderr)
        return 4

    print(f"wiggling {sorted(home)} on {args.port} for {args.secs:.0f} s, "
          f"+/-{args.amplitude} ticks around {home}")
    print(">>> WATCH WHICH PHYSICAL HANDLE TWITCHES. That handle is on this port. <<<")
    try:
        for mid in home:
            pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, TORQUE_ON)
        t0 = time.time()
        while time.time() - t0 < args.secs:
            phase = math.sin(2.0 * math.pi * args.hz * (time.time() - t0))
            for mid, base in home.items():
                pk.write4ByteTxRx(ph, mid, ADDR_GOAL_POSITION,
                                  int(base + args.amplitude * phase))
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\ninterrupted -- releasing torque")
    finally:
        # Restore first, THEN release: the handle ends where it started and limp.
        for mid, base in home.items():
            try:
                pk.write4ByteTxRx(ph, mid, ADDR_GOAL_POSITION, int(base))
            except Exception:
                pass
        time.sleep(0.3)
        for mid in home:
            try:
                pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except Exception:
                pass
        ph.closePort()
    print("done -- torque off, position restored.")
    print(f"If that was the handle you expected, {args.port} is correctly named for you.")
    print("If it was the OTHER one, the udev names are crossed (they were on 2026-09-02):")
    print("  /dev/leader-right drives the PHYSICALLY LEFT handle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
