#!/usr/bin/env python3
"""Kitting data-collection review — corpus QA + pick/place distribution.

Standalone (stdlib only). Reads recordings/<date>/episode_*/annotations.json
(+ corrections.json overrides if present) and the shared compartments.json,
and emits:
  1. a corpus / label-quality audit (usable vs aborted vs mislabelled),
  2. pick-location statistics + per-bag consistency (the ACT "same op every
     time" signal),
  3. place accuracy vs the calibrated compartments,
  4. a clarity verdict,
  5. an HTML report with a top-down scatter (picks + places over compartments).

Usage:
  python3 review_corpus.py [recordings_dir] [--html out.html]
"""
from __future__ import annotations
import glob, json, math, os, sys, statistics as st

def load_ann(path):
    d = json.load(open(path))
    cpath = os.path.join(os.path.dirname(path), "corrections.json")
    if os.path.exists(cpath):
        try:
            corr = json.load(open(cpath))
            ga = {f"{g['bag_id']}:{g['attempt']}": g for g in d.get("grasp_attempts", [])}
            for k, ov in corr.get("grasp_attempts", {}).items():
                if k in ga: ga[k].update(ov)
            pe = {str(p['bag_id']): p for p in d.get("place_events", [])}
            for k, ov in corr.get("place_events", {}).items():
                if k in pe: pe[k].update(ov)
            for k, ov in corr.get("episode_meta", {}).items():
                d.setdefault("episode_meta", {})[k] = ov
        except Exception as e:
            print(f"  ! corrections merge failed for {path}: {e}", file=sys.stderr)
    return d

def fmt(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"

def hbar(frac, width=24):
    frac = max(0.0, min(1.0, frac))
    f = int(round(frac * width))
    return "█" * f + "·" * (width - f)

def main():
    args = [a for a in sys.argv[1:]]
    html_out = None
    if "--html" in args:
        i = args.index("--html"); html_out = args[i+1]; del args[i:i+2]
    root = args[0] if args else "recordings"

    comp_path = os.path.join(root, "compartments.json")
    comps = json.load(open(comp_path))["compartments"] if os.path.exists(comp_path) else []

    ann_files = sorted(glob.glob(os.path.join(root, "*", "episode_*", "annotations.json")))
    mcap_dirs = set(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "episode_*", "yam_left.mcap")))
    ann_dirs = set(os.path.dirname(p) for p in ann_files)

    episodes = []
    for f in ann_files:
        d = load_ann(f)
        ep = os.path.basename(os.path.dirname(f))
        m = d.get("episode_meta", {})
        g = d.get("grasp_attempts", [])
        p = d.get("place_events", [])
        qpath = os.path.join(os.path.dirname(f), "qa.json")
        qa = json.load(open(qpath)) if os.path.exists(qpath) else {}
        episodes.append(dict(ep=ep, dir=os.path.dirname(f), outcome=m.get("outcome"),
                             grasps=g, places=p, flags=d.get("flags", []),
                             segs=d.get("segments", []), kit=m.get("kitting_list", []), qa=qa))

    # ---- 1. corpus / label quality ----
    n = len(episodes)
    usable = [e for e in episodes if e["grasps"]]
    aborted = [e for e in episodes if e["outcome"] == "aborted"]
    mislabel = [e for e in episodes if e["outcome"] == "success" and not e["grasps"]]
    no_ann = sorted(mcap_dirs - ann_dirs)
    no_mcap = sorted(ann_dirs - mcap_dirs)

    # recording-health from qa.json sidecars (written by the watcher / qa_label).
    dead_gripper = [e for e in episodes if e["qa"].get("mcap_ok") and e["qa"].get("gripper_actuated") is False]
    corrupt = [e for e in episodes if e["qa"].get("mcap_ok") is False]
    true_abort = [e for e in aborted if e not in dead_gripper and e not in corrupt]

    all_grasps = [g for e in usable for g in e["grasps"]]
    all_places = [p for e in usable for p in e["places"]]
    by_outcome = {}
    for g in all_grasps:
        by_outcome[g.get("outcome", "?")] = by_outcome.get(g.get("outcome", "?"), 0) + 1
    regrasps = sum(1 for g in all_grasps if g.get("regrasp_of") is not None)

    L = []
    P = L.append
    P("=" * 72)
    P("KITTING DATA-COLLECTION REVIEW")
    P("=" * 72)
    P("")
    P("1. CORPUS & LABEL QUALITY")
    P("-" * 72)
    P(f"  labeled episodes ............ {n}")
    P(f"  usable (>=1 grasp) .......... {len(usable)}   ({len(usable)*100//max(n,1)}%)")
    P(f"  aborted (empty) ............. {len(aborted)}   (genuine operator abort: {len(true_abort)})")
    if dead_gripper or corrupt:
        P(f"  >>> DEAD GRIPPER (channel never closed, not a real abort) .. {len(dead_gripper)}  {[e['ep'] for e in dead_gripper]}")
        P(f"  >>> CORRUPT mcap (unreadable) ............................. {len(corrupt)}  {[e['ep'] for e in corrupt]}")
    P(f"  >>> MISLABEL: success w/ 0 grasp .. {len(mislabel)}  {[e['ep'] for e in mislabel]}")
    P(f"  >>> mcap dir w/ NO annotations ... {len(no_ann)}  {[os.path.basename(d) for d in no_ann]}")
    if no_mcap:
        P(f"  >>> annotations w/ NO yam mcap ... {len(no_mcap)}  {[os.path.basename(d) for d in no_mcap]}")
    P("")
    P(f"  grasp attempts (usable eps) . {len(all_grasps)}")
    for k in ("success", "slip", "drop", "empty"):
        c = by_outcome.get(k, 0)
        P(f"      {k:8} {c:4}  {hbar(c/max(len(all_grasps),1))}")
    P(f"  regrasp attempts ............ {regrasps}")
    P(f"  place events ................ {len(all_places)}")
    P("")
    P("  per-episode:")
    P(f"    {'episode':30}{'outcome':9}{'grasp':6}{'place':6}{'flags':6}")
    for e in episodes:
        tag = "  << mislabel" if (e["outcome"] == "success" and not e["grasps"]) else ""
        P(f"    {e['ep']:30}{str(e['outcome']):9}{len(e['grasps']):<6}{len(e['places']):<6}{len(e['flags']):<6}{tag}")

    # ---- 2. pick-location distribution ----
    P("")
    P("2. PICK-LOCATION DISTRIBUTION  (grasp ee_pose, robot base frame, metres)")
    P("-" * 72)
    succ = [g for g in all_grasps if g.get("ee_pose") and g.get("outcome") == "success"]
    xs = [g["ee_pose"][0] for g in succ]; ys = [g["ee_pose"][1] for g in succ]; zs = [g["ee_pose"][2] for g in succ]
    if len(succ) >= 2:
        def desc(v): return f"mean {fmt(st.mean(v))}  std {fmt(st.pstdev(v))}  min {fmt(min(v))}  max {fmt(max(v))}  range {fmt(max(v)-min(v))}"
        P(f"  successful grasps: {len(succ)}")
        P(f"    x (fwd)  {desc(xs)}")
        P(f"    y (lat)  {desc(ys)}")
        P(f"    z (up)   {desc(zs)}")
        spread = math.hypot(st.pstdev(xs), st.pstdev(ys))
        P(f"    planar spread (hypot of x,y std) = {fmt(spread)} m")
    else:
        P("  <2 successful grasps with pose — not enough to distribute yet.")

    # per-bag consistency across episodes (the "same operation every time" signal)
    P("")
    P("  per-bag pick consistency across episodes  (low std = repeatable = good for ACT):")
    bag_pts = {}
    for g in succ:
        bag_pts.setdefault(g["bag_id"], []).append(g["ee_pose"])
    P(f"    {'bag':5}{'n':4}{'x std':9}{'y std':9}{'z std':9}{'spread(m)':10}")
    for bag in sorted(bag_pts):
        pts = bag_pts[bag]
        if len(pts) >= 2:
            sx = st.pstdev([p[0] for p in pts]); sy = st.pstdev([p[1] for p in pts]); sz = st.pstdev([p[2] for p in pts])
            P(f"    {bag:<5}{len(pts):<4}{fmt(sx):9}{fmt(sy):9}{fmt(sz):9}{fmt(math.hypot(sx,sy)):10}")
        else:
            P(f"    {bag:<5}{len(pts):<4}{'—':9}{'—':9}{'—':9}{'(single sample)':10}")

    # ---- 3. place accuracy ----
    P("")
    P("3. PLACE ACCURACY  vs calibrated compartments")
    P("-" * 72)
    # A release whose XY sits on top of the same bag's grasp XY is a mis-detected
    # release (gripper hysteresis fired on a transient width blip over the pick,
    # not a real placement over box 2). Separate these from true placements.
    SUSPECT_M = 0.08
    ev, clean_ev, suspect = [], [], 0
    for e in usable:
        gxy = {g["bag_id"]: g["ee_pose"][:2] for g in e["grasps"] if g.get("ee_pose")}
        for p in e["places"]:
            if p.get("in_target_region") is None:
                continue
            ev.append(p)
            rp, gp = p.get("achieved_ee_pose"), gxy.get(p.get("bag_id"))
            if rp and gp and math.hypot(rp[0]-gp[0], rp[1]-gp[1]) < SUSPECT_M:
                suspect += 1
            else:
                clean_ev.append(p)
    if ev:
        def acc(evs, lbl):
            if not evs:
                P(f"  {lbl}: none"); return
            inreg = sum(1 for p in evs if p["in_target_region"])
            offs = [p["xy_offset_m"] for p in evs if p.get("xy_offset_m") is not None]
            P(f"  {lbl:28} in-region {inreg}/{len(evs)} ({inreg*100//len(evs)}%)  "
              f"offset mean {fmt(st.mean(offs)) if offs else 'n/a'} max {fmt(max(offs)) if offs else 'n/a'} m  {hbar(inreg/len(evs))}")
        P(f"  evaluable place events ...... {len(ev)}")
        acc(ev, "raw (all)")
        acc(clean_ev, "clean (real placements)")
        P(f"  >>> SUSPECT 'released-at-pick' (release within {int(SUSPECT_M*100)}cm of grasp) .. {suspect}/{len(ev)}")
        geo = any(f.get("kind") == "geometric_targets" for e in usable for f in e["flags"])
        if suspect:
            P("  note: those are a release mis-detect (fumble at the pick). Re-label with the")
            P("        transport gate (qa_label / label_episode --min-transport 0.10) to drop them.")
        elif geo:
            P("  note: targets GEOMETRY-ASSIGNED (--geometric-targets): each placement -> nearest")
            P("        distinct compartment, so this measures placement PRECISION (how centered),")
            P("        not SAP correctness. Residual offset = real placement scatter vs 7cm cells.")
            P("        (True SAP correctness needs per-pick OCR intent + scan<->base calibration.)")
        else:
            P("  note: 0 released-at-pick (transport gate active). Calibration is OK (good placements")
            P("        center ~5cm). Residual offset is bag->compartment ASSIGNMENT (pick order != kit")
            P("        order) — re-label with --geometric-targets, or wire cockpit OCR intent.")
    else:
        P("  no evaluable place events yet.")

    # ---- 4. clarity verdict ----
    P("")
    P("4. CLARITY VERDICT")
    P("-" * 72)
    issues = []
    if dead_gripper: issues.append(f"{len(dead_gripper)} episode(s) with a DEAD GRIPPER (follower gripper never closed) — check gripper teleop/CAN before recording more.")
    if corrupt: issues.append(f"{len(corrupt)} episode(s) with a CORRUPT mcap — unrecoverable, re-record.")
    if mislabel: issues.append(f"{len(mislabel)} 'success' episode(s) with 0 grasps (re-label offline with gripper --open-ref/--closed-ref).")
    if no_ann: issues.append(f"{len(no_ann)} recorded episode(s) never labelled.")
    if ev and sum(1 for p in ev if p['in_target_region']) == 0: issues.append("0% placements in target region — check compartment calibration.")
    if len(usable) < 20: issues.append(f"only {len(usable)} usable episodes — ACT typically wants 30-50+ clean demos.")
    if issues:
        P("  NOT CLEAR — blockers before this is ACT-ready:")
        for i, s in enumerate(issues, 1): P(f"    {i}. {s}")
    else:
        P("  CLEAR — corpus looks consistent and ready to scale.")
    P("")
    P("=" * 72)

    report = "\n".join(L)
    print(report)

    if html_out:
        write_html(html_out, comps, succ, all_grasps, all_places, report)
        print(f"\nHTML report -> {html_out}")

def write_html(path, comps, succ, all_grasps, all_places, report):
    # top-down: robot base frame, x forward (up on screen), y lateral (left on screen).
    pts = [(g["ee_pose"][0], g["ee_pose"][1], g.get("outcome")) for g in all_grasps if g.get("ee_pose")]
    plc = [(p["achieved_ee_pose"][0], p["achieved_ee_pose"][1], p.get("in_target_region"))
           for p in all_places if p.get("achieved_ee_pose")]
    allx = [x for x, y, *_ in pts+plc] + [c["x_max"] for c in comps] + [c["x_min"] for c in comps]
    ally = [y for x, y, *_ in pts+plc] + [c["y_max"] for c in comps] + [c["y_min"] for c in comps]
    if not allx: allx=[0,0.3]; ally=[-0.5,0]
    pad=0.03
    x0,x1=min(allx)-pad,max(allx)+pad; y0,y1=min(ally)-pad,max(ally)+pad
    W,H=760,520
    def sx(y): return (y1 - y)/(y1-y0)*W          # y lateral -> screen x (flip so +y is left)
    def sy(x): return (x1 - x)/(x1-x0)*H          # x forward -> screen y (up = forward)
    col={"success":"#3fb950","slip":"#d29922","drop":"#f85149","empty":"#8b949e",None:"#8b949e"}
    svg=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:#0b0f16;border:1px solid #242c37;border-radius:12px">']
    for c in comps:
        rx,ry=sx(c["y_max"]),sy(c["x_max"]); rw=sx(c["y_min"])-sx(c["y_max"]); rh=sy(c["x_min"])-sy(c["x_max"])
        svg.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" fill="none" stroke="#4da3ff" stroke-width="1.2"/>')
        svg.append(f'<text x="{rx+rw/2:.1f}" y="{ry+rh/2:.1f}" fill="#4da3ff" font-size="13" text-anchor="middle" font-family="monospace">{c["id"]}</text>')
    for x,y,inr in plc:  # place points = hollow squares
        c = "#3fb950" if inr else "#f85149"
        svg.append(f'<rect x="{sx(y)-4:.1f}" y="{sy(x)-4:.1f}" width="8" height="8" fill="none" stroke="{c}" stroke-width="1.5"/>')
    for x,y,o in pts:    # pick points = filled dots
        svg.append(f'<circle cx="{sx(y):.1f}" cy="{sy(x):.1f}" r="4.5" fill="{col.get(o,"#8b949e")}" opacity="0.85"/>')
    svg.append('</svg>')
    legend = ('<div style="font-family:monospace;font-size:12px;color:#9aa7b4;margin:10px 0">'
              '● pick: <span style="color:#3fb950">success</span> '
              '<span style="color:#d29922">slip</span> '
              '<span style="color:#f85149">drop</span> '
              '<span style="color:#8b949e">empty</span> &nbsp;|&nbsp; '
              '▢ place: <span style="color:#3fb950">in-region</span> '
              '<span style="color:#f85149">off</span> &nbsp;|&nbsp; '
              '<span style="color:#4da3ff">▭ compartments 1–7</span> &nbsp;·&nbsp; up = +x (forward), left = +y (lateral)</div>')
    html=(f'<!doctype html><meta charset="utf-8"><title>Kitting data review</title>'
          f'<body style="background:#0d1117;color:#cdd6e0;font-family:system-ui;max-width:900px;margin:24px auto;padding:0 16px">'
          f'<h1 style="color:#e6edf3">Kitting data-collection review</h1>'
          f'<h2 style="color:#e6edf3;font-size:16px">Top-down pick / place distribution</h2>'
          f'{"".join(svg)}{legend}'
          f'<h2 style="color:#e6edf3;font-size:16px">Full report</h2>'
          f'<pre style="background:#0b0f16;border:1px solid #242c37;border-radius:12px;padding:16px;overflow-x:auto;font-size:12.5px;line-height:1.5">{report}</pre>'
          f'</body>')
    open(path,"w").write(html)

if __name__ == "__main__":
    main()
