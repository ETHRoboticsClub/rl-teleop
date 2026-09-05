#!/usr/bin/env python3
"""Is every servo of a gello leader chain powered and answering? (~2 s, read-only)

The chain-ping stage -0.5 of start_left_recording.sh / start_right_recording.sh,
lifted into a tool so rig.sh can gate BOTH handles before a bimanual teleop
session (F3: gate logic is python-canonical, bash relays the exit code).

    tools/ping_leader_chain.py /dev/leader-right 8,9,10,11,12,13,14

Exit codes (the launcher turns each into its fix string):
    0  all ids answered
    2  NONE answered      -> handle electrically dead (5 V/12 V brick, first cable)
    3  SOME answered      -> chain incomplete; prints MISSING_IDS + the rail rule
    4  port unopenable    -> something else holds the serial, or the FTDI vanished

Rail rule: model 1020/1030 = XM430 on the 12 V rail; 1190/1200 = XL330 on the
5 V rail. A whole family missing = that brick is off. A contiguous missing TAIL
= the daisy-chain cable is out at the last id that answered.

Opens the serial device, so run it only when no session owns the port
(:8792/:8794 down). Runbook: runbook/leader-handle-identity-crossed.md
"""
import sys
import time

from dynamixel_sdk import PacketHandler, PortHandler


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 4
    port = sys.argv[1]
    ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
    ph = PortHandler(port)
    pk = PacketHandler(2.0)
    if not ph.openPort():
        print(f"   cannot open {port}")
        return 4
    ph.setBaudRate(1000000)

    alive, dead, models = [], [], {}
    # Every id gets at least one ping; the deadline only cuts RETRIES short, so
    # a slow bus can never make an untried id look missing.
    deadline = time.time() + 2.0
    for mid in ids:
        ok = False
        for attempt in range(3):
            if attempt and time.time() > deadline:
                break
            model, res, err = pk.ping(ph, mid)
            if res == 0 and err == 0:
                ok, models[mid] = True, model
                break
            time.sleep(0.01)
        (alive if ok else dead).append(mid)
    ph.closePort()

    print("   %d/%d servos answered on %s: %s" % (
        len(alive), len(ids), port,
        " ".join("id%d=model%s" % (m, models[m]) for m in alive) or "none"))
    if dead:
        print("   MISSING_IDS %s" % ",".join(str(d) for d in dead))
        xm = [m for m, v in models.items() if v in (1020, 1030)]
        xl = [m for m, v in models.items() if v in (1190, 1200)]
        print("   answering XM430 (12 V rail): %s" % (xm or "none"))
        print("   answering XL330 (5 V rail):  %s" % (xl or "none"))
    return 0 if not dead else (2 if not alive else 3)


if __name__ == "__main__":
    sys.exit(main())
