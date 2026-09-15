"""Unit tests for tools/leader_identity.py — the attestation comparator.

Every branch of `compare()` corresponds to a way the rig has actually lied
about which handle is on which port, so these are regressions on documented
incidents, not coverage padding:

  serial swap / id-set change  → runbook/leader-handle-identity-crossed.md
  crossed config mapping       → runbook/teleop-leaders-crossed.md
  replug since attestation     → the reason "squeeze once, ever" is safe at all

NOTHING HERE TOUCHES HARDWARE. `compare()` is pure: it takes the stored record,
a dict of observations, and (optionally) the config's leader spec. The probe
functions that build those dicts are the only code that opens a device, and
they are not exercised here.
"""
from __future__ import annotations

import time

import leader_identity as li
import pytest

NOW = li.parse_iso("2026-09-15T12:00:00Z")
ATTESTED = "2026-09-10T09:00:00Z"

GOOD_REC = {
    "arm": "right",
    "node": "gello_right",
    "port": "/dev/leader-left",          # crossed name: the RIGHT arm's handle
    "ftdi_serial": "FT94EW3S",
    "physical_side": "right",
    "motor_ids": [1, 2, 3, 4, 5, 6, 7],
    "gripper_id": 7,
    "models": {"1": 1020, "2": 1020, "3": 1020, "4": 1190, "5": 1190, "6": 1190, "7": 1190},
    "device_born_at": "2026-09-05T18:10:27Z",
    "gripper_span_ticks": 56,
    "attested_at": ATTESTED,
    "attested_by": "tommaso",
}

GOOD_OBS = {
    "port": "/dev/leader-left",
    "port_exists": True,
    "tty": "/dev/ttyUSB1",
    "ftdi_serial": "FT94EW3S",
    "device_born_at": "2026-09-05T18:10:27Z",
    "busy_ports": [],
    "chain": {"1": 1020, "2": 1020, "3": 1020, "4": 1190, "5": 1190, "6": 1190, "7": 1190},
}

GOOD_CFG = {"node": "gello_right", "port": "/dev/leader-left",
            "motor_ids": [1, 2, 3, 4, 5, 6, 7]}


def fails(findings):
    return [m for s, m in findings if s == "FAIL"]


def warns(findings):
    return [m for s, m in findings if s == "WARN"]


def test_happy_path_is_silent_and_verifies():
    f = li.compare(GOOD_REC, GOOD_OBS, GOOD_CFG, now=NOW)
    assert f == [], f
    assert li.verdict(f) == li.RC_OK


def test_missing_port_fails_and_stops_evaluating():
    obs = dict(GOOD_OBS, port_exists=False)
    f = li.compare(GOOD_REC, obs, GOOD_CFG, now=NOW)
    assert len(f) == 1 and "does not exist" in f[0][1]
    assert li.verdict(f) == li.RC_CONTRADICTED


def test_ftdi_serial_swap_fails():
    """The adapters were swapped: the port name is the same, the hardware is not.

    This is the check that survives /dev/ttyUSBn renumbering — FTA2U44N was
    ttyUSB1 on 2026-09-02 and ttyUSB0 on 2026-09-15, and neither is evidence
    of anything.
    """
    obs = dict(GOOD_OBS, ftdi_serial="FTA2U44N")
    f = li.compare(GOOD_REC, obs, GOOD_CFG, now=NOW)
    assert any("FTDI serial" in m for m in fails(f)), f


def test_replug_after_attestation_invalidates_it():
    """The kernel bumps the device node on every re-enumeration.

    This is what makes a one-time squeeze safe: nothing has to REMEMBER to
    invalidate the attestation after a replug — the device does it.
    """
    obs = dict(GOOD_OBS, device_born_at="2026-09-12T08:00:00Z")
    f = li.compare(GOOD_REC, obs, GOOD_CFG, now=NOW)
    assert any("re-enumerated" in m for m in fails(f)), f


def test_replug_before_attestation_is_fine():
    obs = dict(GOOD_OBS, device_born_at="2026-09-01T08:00:00Z")
    assert li.compare(GOOD_REC, obs, GOOD_CFG, now=NOW) == []


def test_missing_servo_fails_naming_the_id():
    chain = {k: v for k, v in GOOD_OBS["chain"].items() if k != "7"}
    f = li.compare(GOOD_REC, dict(GOOD_OBS, chain=chain), GOOD_CFG, now=NOW)
    assert any("missing [7]" in m for m in fails(f)), f


def test_wrong_chain_entirely_fails():
    """Swapping which adapter is plugged into which handle changes the id set."""
    chain = {str(i): 1190 for i in range(8, 15)}
    f = li.compare(GOOD_REC, dict(GOOD_OBS, chain=chain), GOOD_CFG, now=NOW)
    assert any("servo id set" in m for m in fails(f)), f


def test_servo_model_change_fails():
    chain = dict(GOOD_OBS["chain"], **{"7": 1020})
    f = li.compare(GOOD_REC, dict(GOOD_OBS, chain=chain), GOOD_CFG, now=NOW)
    assert any("MODELS changed" in m for m in fails(f)), f


def test_crossed_config_mapping_fails():
    """runbook/teleop-leaders-crossed.md: the squeeze gate NEVER caught this.

    It read the leader only, so a YAML that drove the right arm from the left
    handle passed every time. The stored identity is the first automated check
    that can see it.
    """
    cfg = dict(GOOD_CFG, port="/dev/leader-right")
    f = li.compare(GOOD_REC, GOOD_OBS, cfg, now=NOW)
    assert any("CROSSED" in m for m in fails(f)), f


def test_config_motor_ids_mismatch_fails():
    cfg = dict(GOOD_CFG, motor_ids=[8, 9, 10, 11, 12, 13, 14])
    f = li.compare(GOOD_REC, GOOD_OBS, cfg, now=NOW)
    assert any("motor_ids" in m for m in fails(f)), f


def test_side_arm_mismatch_fails_unless_declared():
    rec = dict(GOOD_REC, physical_side="left")
    f = li.compare(rec, GOOD_OBS, GOOD_CFG, now=NOW)
    assert any("physically LEFT handle" in m for m in fails(f)), f
    # a deliberate transplant (the 2026-09-02 leader swap was one) is declarable
    assert li.compare(dict(rec, crossed_ok=True), GOOD_OBS, GOOD_CFG, now=NOW) == []


def test_busy_port_downgrades_the_fingerprint_check_to_a_warning():
    """A live session owns the serial, so the ping is skipped — say so, do not fail."""
    f = li.compare(GOOD_REC, dict(GOOD_OBS, chain=None), GOOD_CFG, now=NOW)
    assert fails(f) == []
    assert any("not re-checked" in m for m in warns(f)), f
    assert li.verdict(f) == li.RC_OK


def test_old_attestation_nags_but_does_not_refuse():
    # The device must predate the attestation too, or this is a replug (which
    # is a FAIL) rather than mere age.
    rec = dict(GOOD_REC, attested_at="2025-01-01T00:00:00Z")
    obs = dict(GOOD_OBS, device_born_at="2024-12-01T00:00:00Z")
    f = li.compare(rec, obs, GOOD_CFG, now=NOW)
    assert fails(f) == []
    assert any("days old" in m for m in warns(f)), f


def test_no_config_means_no_mapping_check():
    """The recording scripts always pass --config; other callers may not."""
    assert li.compare(GOOD_REC, GOOD_OBS, None, now=NOW) == []


def test_record_for_picks_the_right_arm():
    doc = {"handles": [GOOD_REC, dict(GOOD_REC, arm="left", port="/dev/leader-right")]}
    assert li.record_for(doc, "right")["port"] == "/dev/leader-left"
    assert li.record_for(doc, "left")["port"] == "/dev/leader-right"
    assert li.record_for(doc, "middle") is None
    assert li.record_for({}, "right") is None


def test_iso_roundtrip():
    t = time.time()
    assert abs(li.parse_iso(li.iso(t)) - t) < 1.0


@pytest.mark.parametrize("findings,rc", [
    ([], li.RC_OK),
    ([("WARN", "x")], li.RC_OK),
    ([("WARN", "x"), ("FAIL", "y")], li.RC_CONTRADICTED),
])
def test_verdict(findings, rc):
    assert li.verdict(findings) == rc
