# PLAN — permanent leader-handle identity (no more crossed-label failures)

Date: 2026-09-02. Status: draft for /plan-eng-review.
Trigger: five bring-up failures today were the operator squeezing the wrong handle because
the udev names (`/dev/leader-left`, `/dev/leader-right`) are crossed relative to physical
bench sides. Operator asks: identify the handle by the controller connected to it, permanently.

## Identity facts (verified 2026-09-02)

- `/dev/leader-right` = FTDI serial **FTA2U44N** (ttyUSB1) = **physically LEFT** handle,
  servo chain ids **8–14** (XM430 8/9, XM430 10, XL330 11–14). Confirmed twice by servo-wiggle.
- `/dev/leader-left` = FTDI serial **FT94EW3S** (ttyUSB0) = physically RIGHT handle,
  servo chain ids **1–7** (per handoff doc; chain was unpowered today, unverified by ping).
- The FTDI serial identifies the **adapter cable**, not the handle: re-plugging a handle
  onto the other adapter silently breaks any serial-based mapping.
- The **servo chain IDs are baked into the handle's motors** — they are the handle's true,
  interrogable identity, immune to cable swaps.

## Design: identity by interrogation, three layers

1. **udev (adapter layer, exists):** keep serial-keyed symlinks, but add two NEW aliases that
   name the adapter honestly without a side claim: `/dev/gello-FTA2U44N`, `/dev/gello-FT94EW3S`
   (`SYMLINK+=`, additive rule file, old names kept for config compat). No config change today.
2. **software (handle layer, authoritative):** extend the bring-up chain ping (start_left_recording.sh
   stage -0.5): if 0/7 configured ids answer, ping the OTHER handle's id set (1–7); if those
   answer → die with "CABLES SWAPPED: the handle on $GPORT is the ids-1-7 handle; the config
   expects ids 8–14 — either replug or point the config at the other port", instead of the
   generic "electrically dead". Same logic added to `tools/identify_leader_handle.py --whoami`
   (prints which handle a port carries by pinging both id sets, no wiggle needed).
3. **physical (human layer):** operator tapes a label on each handle: "LEFT-ARM LEADER
   (ids 8–14)" / "RIGHT-ARM LEADER (ids 1–7)". Runbook entry updated with the id-set table.

## Deliverables

- udev rule file `/etc/udev/rules.d/98-gello-adapters.rules` (+ copy in
  `rl-teleop/tools/fault-injection/` beside the camera rule, with install note; needs one
  sudo install by operator — SUDO-QUEUE entry).
- `start_left_recording.sh`: swapped-cable branch in the chain ping (die message names both
  id sets and both fixes).
- `identify_leader_handle.py --whoami` (read-only ping of both id sets; wiggle stays as the
  visual fallback).
- Runbook `leader-handle-identity-crossed.md`: append the id-set identity table + the rule
  "identity = servo ids, never the port name, never the adapter serial".
- Tests: chain-ping branch unit-tested with stub dynamixel (right-ids-answer → swapped-cable
  message; pinned: default behavior unchanged when 7/7 answer).

## NOT in scope
- Renaming/removing the existing `leader-left/right` symlinks or editing teleop configs
  (every config references them; a rename is a coordinated change for a quiet day).
- Re-IDing servos to encode sides (flashing servo ids risks bricking a working handle).

## Failure modes
- Both id sets answer on one port (impossible with two handles, possible with a future
  combined chain) → report both, refuse to guess.
- Handle unpowered → both pings fail → existing "electrically dead" path (unchanged).
- udev rule missing after OS update → aliases vanish but nothing depends on them yet;
  software layer (2) is the authoritative gate either way.


## Review outcome (2026-09-02, eng review + outside voice)

The outside voice invalidated the original 3-layer design: the adapter-swap failure it
defended against has never occurred; the actual cause of the 2 h incident was prompt
wording + missing physical labels. Adopted its simpler alternative (auto-decided per
operator's standing instruction):

- FIXED (live bug): both prompt branches said "LEFT-side physical handle" — the
  leader-left branch now names ids 1-7 / physically RIGHT / UNVERIFIED.
- FIXED: die message no longer suggests repointing the config at the other port
  (joint offsets/scales are per-handle calibration; replug is the only safe fix).
- FIXED: comment-rot warning banner added to yam_right_teleop_only.yaml (its prose
  contradicted its declared motor_ids and fed today's confusion).
- OPERATOR ACTION: tape physical labels on both handles: "IDS 8-14 — LEFT-ARM LEADER"
  and "IDS 1-7 (unverified) — RIGHT-ARM LEADER".
- DROPPED: udev aliases (redundant — /dev/serial/by-id already provides serial-keyed
  stable paths), swapped-cable ping branch (defends an unobserved failure; the two
  id sets being disjoint is itself unverified until the dead handle is powered).
- DEFERRED: extracting the chain-ping heredoc to a testable tools/ping_leader_chain.py
  (prerequisite for any future layer-2 work); ping-gates for the other four leader
  entry points; config/comment consistency lint in preflight.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 4 issues folded; scope replaced by outside-voice alternative |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | codex unauthenticated; Claude subagent used |

- **CROSS-MODEL:** outside voice (Claude subagent) found 10 issues incl. a live prompt bug
  and a dangerous die-message suggestion; its simpler 30-minute alternative was adopted
  wholesale, shrinking the plan from ~a day to four small fixes (three landed immediately).
- **VERDICT:** ENG CLEARED — reduced scope fully implemented except the physical labels
  (operator) and the deferred items above.

**UNRESOLVED DECISIONS:**
- Whether to power the dead handle and verify its id set (blocks any future
  interrogation-based identity layer; harmless to leave unresolved today).
