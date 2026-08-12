#!/usr/bin/env python3
"""Bring the arm home. The command you reach for when something has gone wrong.

    tools/park_arm.py                 # park every arm in the running session
    tools/park_arm.py --secs 8        # slower ramp
    tools/park_arm.py --port 8792     # a different control port
    tools/park_arm.py --status        # just ask where things stand

WHY THIS EXISTS
===============

On 2026-08-12 a policy agent node died at startup and the right arm could not be
brought home by any software route:

  * it was gated by `start_paused`, so publishing on the command topic did
    nothing;
  * `Session.resume()` walked the hosts in order and hung on the dead agent's
    control socket before it ever reached the arm, while `/status` reported
    `paused: false`;
  * the agent that owned the command topic was the thing that had died, so
    nothing was left to command a pose with.

The arm sat energised in a reaching pose over the source box and the only way
out was to kill the session and have a person hold it while it sagged. On an arm
with no brakes that is not an acceptable resting state for a bug to leave.

`POST /park` goes straight to each RobotNode's own control socket. It does not
use the bus, does not need an agent, takes the pause gate instead of waiting for
it, and skips nodes that are already dead instead of blocking on them.

WHAT IT DOES NOT DO
===================

It cannot help if there is no session running — once the motors are released
there is nothing listening and nothing holding the arm. If this says it cannot
reach the control port, the arm is unpowered, and the answer is hands, not
software.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _call(port: int, path: str, method: str = "POST", timeout: float = 60.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8792, help="rr-session control port")
    ap.add_argument("--secs", type=float, default=None, help="ramp duration")
    ap.add_argument("--status", action="store_true", help="only report status")
    a = ap.parse_args(argv)

    try:
        if a.status:
            print(json.dumps(_call(a.port, "/status", "GET", timeout=10), indent=2))
            return 0

        path = "/park" if a.secs is None else f"/park?secs={a.secs}"
        print(f"parking via http://127.0.0.1:{a.port}{path} ...")
        out = _call(a.port, path)
    except urllib.error.URLError as exc:
        print(f"\n  cannot reach the control server on port {a.port}: {exc}\n"
              f"  If no session is running, the motors are already released and\n"
              f"  there is nothing to command — the arm needs HANDS, not software.\n",
              file=sys.stderr)
        return 2
    except Exception as exc:                                       # noqa: BLE001
        print(f"  park request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for node, result in sorted((out.get("results") or {}).items()):
        print(f"  {node:16s} {result}")
    if out.get("ok"):
        print("\n  PARKED. The session is paused; nothing will drive the arm until you resume.")
        return 0
    print(f"\n  PARK INCOMPLETE — did not confirm: {', '.join(out.get('failed') or [])}\n"
          f"  Do NOT assume the arm is home. Look at it before releasing anything.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
