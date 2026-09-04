#!/usr/bin/env python3
"""Hardware-reset the D455 when it enumerates but delivers no frames.

2026-09-05: camera_top opened clean (preflight ok, uvc enumerated, speed 5000)
but the node sat at X-Cam-State: no_data and BOTH pyrealsense2 backends timed
out on wait_for_frames. dmesg showed NO xhci reset, so this was not the wedge
in runbook/camera-d455-xhci-reset.md — and the authorized 0/1 soft-replug did
NOT fix it. rs.device.hardware_reset() did, first try (139 frames / 5 s after).

Run with the session STOPPED — a node holding the device blocks the reset.
Runbook: runbook/camera-d455-xhci-reset.md
"""
import sys
import time

import pyrealsense2 as rs

serial = sys.argv[1] if len(sys.argv) > 1 else "203522250539"
devs = [d for d in rs.context().query_devices()
        if d.get_info(rs.camera_info.serial_number) == serial]
if not devs:
    sys.exit(f"no RealSense with serial {serial} enumerated — check cable/authorized")
devs[0].hardware_reset()
print("hardware_reset sent; waiting 12 s for re-enumeration ...")
time.sleep(12)
back = any(d.get_info(rs.camera_info.serial_number) == serial
           for d in rs.context().query_devices())
print("re-enumerated OK — restart the session now (stale handle rule)"
      if back else "DID NOT re-enumerate — physical replug is next")
sys.exit(0 if back else 1)
