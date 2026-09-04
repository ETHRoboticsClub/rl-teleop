#!/usr/bin/env python3
"""Clear latched Torque Enable on leader-handle servos.

2026-09-05: after a rec session was killed mid-flight (tmux kill-session, no
clean shutdown), the gripper servo (id 7 on /dev/leader-left) was left with
Torque Enable = 1. The motor then actively held ~1721 ticks, so the operator's
squeeze read as "GRIPPER IS FLAT" — thousands of clean reads, zero travel —
the same signature as the crossed-handle trap, but with the RIGHT handle.
A leader handle must be passive: torque belongs OFF on every servo.

Usage: clear_leader_torque.py <port> <comma-separated-ids>
Exits 0 whether or not anything was latched; nonzero only if the port is
unopenable. Safe pre-gate: reads/writes only Torque Enable (addr 64).
Runbook: runbook/gripper-gate-flat-torque-latched.md
"""
import sys

from dynamixel_sdk import PacketHandler, PortHandler

port, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x]
ph = PortHandler(port)
pk = PacketHandler(2.0)
if not (ph.openPort() and ph.setBaudRate(1000000)):
    sys.exit(f"cannot open {port}")
latched = []
for i in ids:
    v, r, _ = pk.read1ByteTxRx(ph, i, 64)
    if r == 0 and v:
        pk.write1ByteTxRx(ph, i, 64, 0)
        latched.append(i)
ph.closePort()
if latched:
    print(f"torque was LATCHED on ids {latched} — cleared (an unclean session "
          f"kill leaves this behind; the gripper gate would read flat)")
else:
    print("no latched torque on the leader chain")
