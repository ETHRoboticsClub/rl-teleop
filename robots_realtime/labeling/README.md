# Kitting trajectory auto-labeling

Turns recorded kitting teleop episodes into structured, queryable labels, and
drives the operator cockpit's live view as each bag is grasped and placed.

Design + review history: `~/Desktop/sorting-red-box/kitting-labeling-plan.md`.

## What it produces

Per episode, an immutable sidecar next to the raw MCAP/MP4:

```
recordings/<episode>/
  yam_left.mcap  gello_left.mcap  camera_*.mp4  ...   (raw, never touched)
  cockpit_events.jsonl        (optional: live intent from the cockpit)
  compartments.json  kit.json  (optional: box calibration + OCR kit list)
  annotations.json            ← machine labels (regenerable)
  corrections.json            ← human overrides (merged on read, corrections win)
```

`annotations.json` carries, per bag: grasp attempts (6-DoF pose, outcome
success/slip/empty, regrasp chain), place events (target compartment, achieved
pose, in-region + xy offset), phase segments, commanded-vs-achieved tracking
error, and loud `flags` for anything ambiguous (never a silent drop).

## Key facts baked in
- **No gripper force sensor.** Grasp/slip come from gripper WIDTH (joint index 6)
  + end-effector lift, never "gripper effort".
- **Single-arm (left) kitting**; the schema is arm-agnostic.
- FK is to `link_6` on `urdf/yam.urdf` (a regression test guards joint-order drift).
- The box is **fixed** — compartments are calibrated once (`compartments.json`).

## Use it

Label one episode:
```bash
uv run python -m robots_realtime.labeling.label_episode recordings/<episode> \
    --arm left --open-ref <gripper_open_joint> --closed-ref <gripper_closed_joint>
```
(If you omit the gripper refs it normalizes per-episode and flags when it can't
tell an empty grasp from a bag.)

Build the process model over a corpus:
```bash
uv run python -m robots_realtime.labeling.aggregator recordings/
```

Live labeling in the cockpit (point the cockpit's live URL at this):
```bash
uv run python -m robots_realtime.labeling.live_server --port 8791 \
    --replay recordings/<episode>        # or omit --replay for a synthetic demo
    --cam-base http://localhost:8765      # optional: proxy camera MJPEG through
```
The cockpit polls `GET /state` and sees `ti` advance + packets flip to `placed`
as grasps/places are detected. Confirmation stays batched per kit (design choice).

## Prerequisites still to wire
1. Cockpit stamps its events on rl-teleop's hardware clock (shared source).
2. `compartments.json` = the 7 measured compartment rects in the robot base frame.
3. `bus_feed` for the live server to subscribe to the real rl-teleop joint stream
   (the replay/demo feed is wired and verified; the live-bus tap is the last hop).
