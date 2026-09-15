#!/usr/bin/env python3
"""Persistent, machine-verifiable identity for the GELLO leader handles.

WHY THIS EXISTS
---------------
Every recording bring-up made the operator squeeze a leader trigger fully
closed and release, for up to 45 s, while the script watched the gripper
servo's Present Position (start_left_recording.sh:214-276,
start_right_recording.sh:215-275).  Ask what that squeeze actually PROVED and
it decomposes into four claims:

  1. the serial port opens and the gripper id answers
  2. Torque Enable is not latched on that servo
  3. the handle the operator is holding is the one wired to THIS port
  4. the trigger is mechanically coupled to that servo

(1) is already proven, in 2 s and with no operator, by the chain ping
(tools/ping_leader_chain.py, stage -0.5).  (2) is already fixed, with no
operator, by tools/clear_leader_torque.py.  So the squeeze was being paid
every single bring-up to re-prove two facts that were already proven, in
order to establish (3) and (4).

And (3)+(4) are STATIC.  Which physical handle hangs off which FTDI adapter,
and whether its trigger turns its servo, do not change between two bring-ups
five minutes apart.  They change when the HARDWARE changes: a replug, a
re-cable, a handle swap, a udev edit.  Re-proving a static fact on a timer is
what made the gate feel like a tax.

WHAT THIS REPLACES IT WITH
--------------------------
An attestation, written once by the operator, that binds the human-only fact
(which bench side this handle is on) to a fingerprint a machine can re-check
for free on every later bring-up:

    arm  ─ port ─ FTDI serial ─ physical side ─ motor ids ─ per-id models
                      │                                  │
                 stable across replugs         re-read by a 2 s ping
                 (udev rule keys on it)        (already run at stage -0.5)

plus the device node's birth time, which the kernel bumps on EVERY
re-enumeration.  That last field is what makes "squeeze once, ever" safe: any
replug of that adapter makes the device newer than the attestation and the
verifier demands a fresh squeeze.  Nothing else in this repo has to remember
to invalidate anything.

WHAT IT CANNOT PROVE, SAID OUT LOUD
-----------------------------------
If someone carries a whole handle assembly — FTDI adapter included — from the
left side of the bench to the right side, every field above still matches.
No software signal distinguishes bench positions; that is why
tools/identify_leader_handle.py (wiggle and LOOK) remains the only identity
oracle.  The mitigation is honest, not magic: re-attest after any physical
change, and the replug check above catches the far more common case where the
adapter was unplugged in the process.

FILE
----
configs/leader_identity.json (override with $LEADER_IDENTITY).  Absent by
design on a fresh checkout: every caller falls back to the old squeeze gate
when it is missing, so the default behaviour of this repo is unchanged until
an operator deliberately attests.

USAGE
-----
    # ONE TIME, per handle (the only squeeze you ever have to do):
    ./.venv/bin/python3 tools/leader_identity.py attest --arm right \
        --config configs/yam/yam_right_grasp_teleop_noscan.yaml

    # every bring-up, automatic, no operator, ~2 s:
    ./.venv/bin/python3 tools/leader_identity.py verify --arm right \
        --config configs/yam/yam_right_grasp_teleop_noscan.yaml

    ./.venv/bin/python3 tools/leader_identity.py show

EXIT CODES (verify)
    0   verified — the caller may skip the squeeze gate
    10  no attestation for this arm (file absent or arm missing) — callers
        MUST fall back to the old gate; this is not an error
    11  attestation present and CONTRADICTED — caller must refuse and say why
    12  attest could not proceed (port gone, a session owns the serial,
        or the chain is dead/incomplete) -- nothing was written

SAFETY
    verify is read-only apart from the chain ping, which opens the serial and
    is therefore skipped whenever :8792/:8794/:8797 is listening (a session
    owns the port) — the same guard stage -0.5 and identify_leader_handle.py
    use.  Nothing here touches CAN, a follower, or a camera.

Runbook: runbook/leader-handle-identity-crossed.md,
         runbook/gripper-gate-flat-torque-latched.md,
         runbook/teleop-leaders-crossed.md
Map:     docs/reference/RIG-ADDRESS-MAP.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "leader-identity/1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "configs" / "leader_identity.json"

# A live rr-session owns a leader serial; opening it a second time garbles the
# Dynamixel bus. Same ports identify_leader_handle.py refuses on, plus :8797.
BUSY_PORTS = (8792, 8794, 8797)

RC_OK = 0
RC_NO_ATTESTATION = 10
RC_CONTRADICTED = 11
RC_CANNOT_PROBE = 12

# Re-attestation nag. Not a failure: a handle nobody touched for months is not
# broken, but a year-old attestation deserves a glance.
STALE_DAYS = 120

# The device node's ctime is the enumeration time. Give it a second of slack so
# an attestation written in the same second as a udev event is not "before" it.
REPLUG_SLACK_S = 2.0


# ── pure helpers (unit-tested; no hardware, no filesystem) ───────────────────

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def record_for(doc: dict, arm: str) -> dict | None:
    """The attestation for one arm, or None."""
    for h in (doc or {}).get("handles", []):
        if h.get("arm") == arm:
            return h
    return None


def compare(rec: dict, obs: dict, cfg: dict | None = None, now: float | None = None) -> list[tuple[str, str]]:
    """Every way the world can contradict an attestation.

    Returns a list of (severity, message); severity is "FAIL" or "WARN".
    Pure: `obs` and `cfg` are plain dicts assembled by the probe functions, so
    every branch here is reachable from a unit test with no rig attached.
    """
    out: list[tuple[str, str]] = []
    now = time.time() if now is None else now

    # 1. the udev symlink still exists and still resolves
    if not obs.get("port_exists"):
        out.append(("FAIL", f"{rec['port']} does not exist — handle unplugged or the udev "
                            f"rule (99-yam-ft232h-leaders.rules) is gone"))
        return out  # nothing below can be evaluated

    # 2. the FTDI serial behind that symlink is the attested adapter.
    #    This is the load-bearing check: /dev/ttyUSBn RENUMBERS between boots
    #    (FTA2U44N was ttyUSB1 on 2026-09-02 and ttyUSB0 on 2026-09-15), the
    #    symlink NAME is known-crossed, but the serial is what the udev rule
    #    keys on and it follows the adapter.
    if rec.get("ftdi_serial") and obs.get("ftdi_serial") != rec["ftdi_serial"]:
        out.append(("FAIL", f"{rec['port']} now resolves to FTDI serial "
                            f"{obs.get('ftdi_serial') or '?'}, attested as {rec['ftdi_serial']} "
                            f"— the adapters were swapped or a udev rule changed"))

    # 3. has the device been re-enumerated since the attestation?
    born = obs.get("device_born_at")
    if born and rec.get("attested_at"):
        if parse_iso(born) > parse_iso(rec["attested_at"]) + REPLUG_SLACK_S:
            out.append(("FAIL", f"{rec['port']} was re-enumerated at {born}, AFTER the "
                                f"attestation at {rec['attested_at']} — something was "
                                f"replugged; the handle behind this port is no longer vouched for"))

    # 4. the servo fingerprint: id set AND per-id model numbers.
    #    Catches a re-flashed id, a lost servo, and swapping the adapters
    #    between the two chains (the id sets differ: 1-7 vs 8-14, and so do the
    #    model vectors — see the rail table in RIG-ADDRESS-MAP.md).
    chain = obs.get("chain")
    if chain is None:
        out.append(("WARN", "chain ping skipped (a session owns the serial) — servo "
                            "fingerprint not re-checked this bring-up"))
    else:
        want_ids = [int(i) for i in rec.get("motor_ids", [])]
        got_ids = sorted(int(k) for k in chain)
        if got_ids != sorted(want_ids):
            missing = sorted(set(want_ids) - set(got_ids))
            extra = sorted(set(got_ids) - set(want_ids))
            out.append(("FAIL", f"servo id set on {rec['port']} is {got_ids}, attested "
                                f"{sorted(want_ids)}"
                                + (f" — missing {missing}" if missing else "")
                                + (f" — unexpected {extra}" if extra else "")))
        want_models = {int(k): int(v) for k, v in (rec.get("models") or {}).items()}
        if want_models:
            bad = {i: (int(chain[str(i)]) if str(i) in chain else chain.get(i))
                   for i in want_models
                   if str(i) in chain and int(chain[str(i)]) != want_models[i]}
            if bad:
                out.append(("FAIL", f"servo MODELS changed on {rec['port']}: {bad} "
                                    f"(attested {want_models}) — a servo was replaced, or "
                                    f"this is a different chain"))

    # 5. the config that is about to be launched must point at this handle.
    #    The squeeze never checked this — it read the leader only, so a YAML
    #    with the arms crossed passed the gate every time
    #    (runbook/teleop-leaders-crossed.md).
    if cfg:
        if cfg.get("port") != rec["port"]:
            out.append(("FAIL", f"config node {cfg.get('node')} reads {cfg.get('port')}, but "
                                f"the {rec['arm']} arm is attested to {rec['port']} — the "
                                f"leader→arm mapping in the config is CROSSED"))
        if cfg.get("motor_ids") and [int(i) for i in cfg["motor_ids"]] != [int(i) for i in rec.get("motor_ids", [])]:
            out.append(("FAIL", f"config node {cfg.get('node')} uses motor_ids "
                                f"{cfg['motor_ids']}, attested {rec.get('motor_ids')}"))

    # 6. arm side vs bench side. On this rig the left arm is driven by the
    #    handle standing on the left. A record that says otherwise is either a
    #    deliberate transplant (attest with --allow-crossed) or a typo.
    side = rec.get("physical_side")
    if side and side != rec["arm"] and not rec.get("crossed_ok"):
        out.append(("FAIL", f"attestation says the {rec['arm']} arm is driven by the "
                            f"physically {side.upper()} handle. If that is deliberate, "
                            f"re-attest with --allow-crossed."))

    # 7. age — a nag, never a refusal.
    if rec.get("attested_at"):
        age_d = (now - parse_iso(rec["attested_at"])) / 86400.0
        if age_d > STALE_DAYS:
            out.append(("WARN", f"attestation is {age_d:.0f} days old (> {STALE_DAYS}); "
                                f"worth one confirming squeeze"))
    return out


def verdict(findings: list[tuple[str, str]]) -> int:
    return RC_CONTRADICTED if any(s == "FAIL" for s, _ in findings) else RC_OK


# ── probes (read-only except the ping, which opens the serial) ───────────────

def listening_ports() -> set[int]:
    try:
        out = subprocess.run(["ss", "-tlnH"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return set()
    ports = set()
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 4:
            _, _, p = f[3].rpartition(":")
            if p.isdigit():
                ports.add(int(p))
    return ports


def ftdi_serial_of(tty_path: str) -> str | None:
    """FTDI serial behind a resolved /dev/ttyUSBn, via sysfs then by-id."""
    name = os.path.basename(tty_path)
    p = Path("/sys/class/tty") / name / "device" / ".." / ".." / "serial"
    try:
        return p.read_text().strip()
    except Exception:
        pass
    byid = Path("/dev/serial/by-id")
    if byid.is_dir():
        for link in byid.iterdir():
            try:
                if os.path.realpath(link) == os.path.realpath(tty_path):
                    # usb-FTDI_USB__-__Serial_Converter_FT94EW3S-if00-port0
                    stem = link.name.rsplit("-if", 1)[0]
                    return stem.rsplit("_", 1)[-1]
            except Exception:
                continue
    return None


def ping_chain(port: str, ids: list[int], secs: float = 2.0) -> dict[str, int] | None:
    """{id: model} for every servo that answers. None if the port cannot open."""
    try:
        from dynamixel_sdk import PacketHandler, PortHandler  # late: needs the venv
    except Exception:
        return None
    ph = PortHandler(port)
    pk = PacketHandler(2.0)
    if not ph.openPort():
        return None
    ph.setBaudRate(1000000)
    models: dict[str, int] = {}
    deadline = time.time() + secs
    for mid in ids:
        for attempt in range(3):
            if attempt and time.time() > deadline:
                break
            model, res, err = pk.ping(ph, mid)
            if res == 0 and err == 0:
                models[str(mid)] = int(model)
                break
            time.sleep(0.01)
    ph.closePort()
    return models


def probe(port: str, ids: list[int], allow_open: bool = True) -> dict:
    obs: dict[str, Any] = {"port": port, "port_exists": os.path.exists(port)}
    if not obs["port_exists"]:
        return obs
    real = os.path.realpath(port)
    obs["tty"] = real
    obs["ftdi_serial"] = ftdi_serial_of(real)
    try:
        obs["device_born_at"] = iso(os.stat(real).st_ctime)
    except OSError:
        obs["device_born_at"] = None
    busy = sorted(set(BUSY_PORTS) & listening_ports())
    obs["busy_ports"] = busy
    obs["chain"] = None if (busy or not allow_open) else ping_chain(port, ids)
    return obs


def config_spec(cfg_path: str, node: str) -> dict:
    """{'node','port','motor_ids'} for one gello node of a session config."""
    import yaml
    doc = yaml.safe_load(Path(cfg_path).read_text())
    for n in doc.get("nodes", []):
        if n.get("name") == node:
            k = n.get("agent_kwargs") or {}
            return {"node": node, "port": k.get("port"), "motor_ids": k.get("motor_ids")}
    raise SystemExit(f"node {node!r} not found in {cfg_path}")


# ── document I/O ────────────────────────────────────────────────────────────

def doc_path(args) -> Path:
    return Path(getattr(args, "file", None) or os.environ.get("LEADER_IDENTITY") or DEFAULT_PATH)


def load_doc(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"{path} is not readable JSON: {exc}")


def save_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


# ── verbs ───────────────────────────────────────────────────────────────────

def cmd_verify(args) -> int:
    path = doc_path(args)
    doc = load_doc(path)
    rec = record_for(doc, args.arm)
    if rec is None:
        print(f"   no leader-identity attestation for the {args.arm} arm "
              f"({path}) — falling back to the squeeze gate")
        return RC_NO_ATTESTATION

    cfg = config_spec(args.config, args.node) if args.config else None
    obs = probe(rec["port"], [int(i) for i in rec.get("motor_ids", [])],
                allow_open=not args.no_open)
    findings = compare(rec, obs, cfg)
    rc = verdict(findings)

    if rc == RC_OK:
        chain = "fingerprint re-checked" if obs.get("chain") is not None else "fingerprint NOT re-checked"
        print(f"   ✓ {args.arm} leader identity verified: {rec['port']} "
              f"(FTDI {rec.get('ftdi_serial')}, ids {rec.get('motor_ids')}, physically "
              f"{str(rec.get('physical_side','?')).upper()} handle) — attested "
              f"{rec.get('attested_at')} by {rec.get('attested_by','?')}, {chain}")
        if rec.get("gripper_span_ticks"):
            print(f"     gripper proven to travel {rec['gripper_span_ticks']} ticks "
                  f"at attestation; the 45 s squeeze is not repeated")
    for sev, msg in findings:
        print(f"   {'✗' if sev == 'FAIL' else '⚠'} {msg}")
    if rc != RC_OK:
        print(f"   The attestation in {path} is CONTRADICTED by the rig. Do not trust it.")
        print("   Settle identity by eye, then re-attest:")
        print(f"     ./.venv/bin/python3 tools/identify_leader_handle.py --port {rec['port']}")
        print(f"     ./.venv/bin/python3 tools/leader_identity.py attest --arm {args.arm}"
              + (f" --config {args.config}" if args.config else ""))
    return rc


def cmd_attest(args) -> int:
    path = doc_path(args)
    doc = load_doc(path)
    doc.setdefault("schema", SCHEMA)
    doc.setdefault(
        "note",
        "Operator attestations binding each leader handle's BENCH SIDE (a fact only a "
        "human can establish) to a fingerprint a machine can re-check for free. Written "
        "by tools/leader_identity.py attest; read by the recording bring-ups to skip the "
        "45 s squeeze gate. Re-attest after ANY replug, re-cable or handle swap — the "
        "verifier also refuses on its own once the device node is newer than attested_at.",
    )

    if args.config:
        cfg = config_spec(args.config, args.node)
        port = args.port or cfg["port"]
        ids = [int(i) for i in (args.ids.split(",") if args.ids else cfg["motor_ids"])]
    else:
        if not (args.port and args.ids):
            raise SystemExit("attest needs --config, or both --port and --ids")
        port, ids = args.port, [int(i) for i in args.ids.split(",")]
    side = args.side or args.arm

    busy = sorted(set(BUSY_PORTS) & listening_ports())
    if busy:
        print(f"REFUSING: {busy} listening — a live session owns a leader serial. "
              f"Stop it and re-run.", file=sys.stderr)
        return RC_CANNOT_PROBE
    if not os.path.exists(port):
        print(f"REFUSING: {port} does not exist.", file=sys.stderr)
        return RC_CANNOT_PROBE

    # Torque latched by an unclean session kill reads exactly like a flat
    # gripper (runbook/gripper-gate-flat-torque-latched.md). Clear it first so
    # the attestation cannot bake in that trap.
    try:
        subprocess.run([sys.executable, str(REPO / "tools" / "clear_leader_torque.py"),
                        port, ",".join(str(i) for i in ids)], timeout=30)
    except Exception as exc:
        print(f"   (could not clear leader torque: {exc})")

    chain = ping_chain(port, ids)
    if not chain:
        print(f"REFUSING: 0/{len(ids)} servos answered on {port} — the handle is "
              f"electrically dead (power brick), not un-squeezed.", file=sys.stderr)
        return RC_CANNOT_PROBE
    if sorted(int(k) for k in chain) != sorted(ids):
        missing = sorted(set(ids) - {int(k) for k in chain})
        print(f"REFUSING: chain incomplete on {port}, missing ids {missing}. Fix the "
              f"chain before attesting (rail rule: model 1020/1030 = XM430 on 12 V, "
              f"1190/1200 = XL330 on 5 V).", file=sys.stderr)
        return RC_CANNOT_PROBE

    gid = ids[-1]
    print()
    print("  ── LEADER IDENTITY ATTESTATION — the one squeeze you have to do ──────")
    print(f"  Port {port} (FTDI {ftdi_serial_of(os.path.realpath(port))}), servo ids {ids}.")
    print(f"  You are recording that this port drives the {args.arm.upper()} arm and that")
    print(f"  its handle is the one standing on the PHYSICALLY {side.upper()} side of the")
    print("  bench. Do NOT go by the /dev name — the udev names are CROSSED.")
    print()
    print(f"  >>> Squeeze the trigger of the handle on the {side.upper()} side of the bench,")
    print(f"      fully closed, and release. Reading motor {gid} for {args.secs:.0f} s ...")
    span, lo, hi, n = _watch_travel(port, gid, args.secs, args.min_span)
    if span is None:
        print(f"\n  ✗ no travel: {n} reads, stuck at {lo}. NOTHING WAS WRITTEN.")
        print("    Thousands of clean reads with zero travel means the trigger you "
              "squeezed is\n    not on this port. Settle it by eye:")
        print(f"      ./.venv/bin/python3 tools/identify_leader_handle.py --port {port}")
        print(f"    and watch which handle twitches. That handle is the one on {port}.")
        return RC_CONTRADICTED
    print(f"  ✓ gripper travels {lo}..{hi} ticks (span {span})")

    real = os.path.realpath(port)
    rec = {
        "arm": args.arm,
        "node": args.node,
        "port": port,
        "ftdi_serial": ftdi_serial_of(real),
        "physical_side": side,
        "motor_ids": ids,
        "gripper_id": gid,
        "models": {str(k): int(v) for k, v in sorted(chain.items(), key=lambda kv: int(kv[0]))},
        "device_born_at": iso(os.stat(real).st_ctime),
        "gripper_span_ticks": span,
        "method": "squeeze",
        "attested_at": iso(time.time()),
        "attested_by": args.by or os.environ.get("USER", "operator"),
    }
    if args.allow_crossed:
        rec["crossed_ok"] = True
    handles = [h for h in doc.get("handles", []) if h.get("arm") != args.arm]
    handles.append(rec)
    doc["handles"] = sorted(handles, key=lambda h: h.get("arm", ""))
    save_doc(path, doc)
    print(f"  written to {path}")
    print(f"  From now on {args.arm}-arm bring-ups verify this in ~2 s and skip the squeeze.")
    print("  Re-run this after ANY replug of that adapter (the verifier will tell you).")
    return RC_OK


def _watch_travel(port: str, gid: int, secs: float, min_span: int):
    """The recording scripts' gate, verbatim in behaviour: span > min_span wins."""
    from dynamixel_sdk import PacketHandler, PortHandler
    ph = PortHandler(port)
    pk = PacketHandler(2.0)
    if not ph.openPort():
        return None, None, None, 0
    ph.setBaudRate(1000000)
    lo = hi = None
    n = 0
    t0 = time.time()
    try:
        while time.time() - t0 < secs:
            pos, res, err = pk.read4ByteTxRx(ph, gid, 132)
            if res == 0 and err == 0:
                n += 1
                lo = pos if lo is None else min(lo, pos)
                hi = pos if hi is None else max(hi, pos)
                if hi - lo > min_span:
                    return hi - lo, lo, hi, n
            time.sleep(0.03)
    finally:
        ph.closePort()
    return None, lo, hi, n


def cmd_show(args) -> int:
    path = doc_path(args)
    doc = load_doc(path)
    if not doc.get("handles"):
        print(f"no attestations in {path}")
        print("the recording bring-ups will keep asking for the 45 s squeeze until you run:")
        print("  tools/leader_identity.py attest --arm left  --config configs/yam/yam_left_handoff_teleop_noscan.yaml")
        print("  tools/leader_identity.py attest --arm right --config configs/yam/yam_right_grasp_teleop_noscan.yaml")
        return RC_NO_ATTESTATION
    print(f"{path}")
    for h in doc["handles"]:
        print(f"  {h['arm']:>5} arm ← {h['port']:<18} FTDI {h.get('ftdi_serial','?'):<10} "
              f"ids {h.get('motor_ids')} physically-{str(h.get('physical_side','?')).upper()}")
        print(f"        attested {h.get('attested_at')} by {h.get('attested_by','?')}, "
              f"gripper span {h.get('gripper_span_ticks')} ticks, "
              f"device born {h.get('device_born_at')}")
    return RC_OK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--file", help=f"identity file (default {DEFAULT_PATH}, $LEADER_IDENTITY)")
        p.add_argument("--arm", required=True, choices=("left", "right"))
        p.add_argument("--config", help="session config to cross-check the leader→arm mapping")
        p.add_argument("--node", help="gello node name in --config (default gello_<arm>)")

    v = sub.add_parser("verify", help="read-only check of the stored attestation")
    common(v)
    v.add_argument("--no-open", action="store_true",
                   help="skip the chain ping even when the port is free")

    a = sub.add_parser("attest", help="ONE-TIME: squeeze once, record the identity")
    common(a)
    a.add_argument("--port", help="override the port (default: from --config)")
    a.add_argument("--ids", help="override motor ids, comma separated")
    a.add_argument("--side", choices=("left", "right"),
                   help="bench side of the handle (default: same as --arm)")
    a.add_argument("--allow-crossed", action="store_true",
                   help="record a deliberate side↔arm transplant without failing verify")
    a.add_argument("--secs", type=float, default=45.0)
    a.add_argument("--min-span", type=int, default=25, help="ticks of travel that count")
    a.add_argument("--by", help="who attested (default $USER)")

    s = sub.add_parser("show", help="print the stored attestations")
    s.add_argument("--file")
    s.set_defaults(arm=None, config=None, node=None)

    args = ap.parse_args()
    if getattr(args, "arm", None) and not getattr(args, "node", None):
        args.node = f"gello_{args.arm}"
    return {"verify": cmd_verify, "attest": cmd_attest, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
