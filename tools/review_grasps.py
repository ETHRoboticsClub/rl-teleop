#!/usr/bin/env python3
"""Visual review of the grasp windows that will become ACT training data.

The point: before a 35k-step training run, look at every clip that goes in and
throw out the bad ones. This renders one card per grasp window showing the two
cameras the policy will actually see, at the three moments that matter, plus a
coverage map of where in the workspace the grasps happened.

WHY IT IMPORTS FROM export_lerobot RATHER THAN RE-DERIVING: a review tool that
computes its own episode selection and window boundaries is reviewing a dataset
that does not exist. `usable_grasps` and `grasp_windows` are imported, so what
you see here is what gets written, including the overlap clipping that makes
neighbouring windows shorter than pre_s + post_s.

THE THREE MOMENTS, and why these three:

    t_lo ............... t_close ............... t_hi
    window start        gripper closes         window end
    |                   |                      |
    the FIRST frame     the grasp itself       the lift
    the policy sees.    is the bag actually    did it come up,
    at deploy IK        under the fingers?     or did it slip?
    hands off here,
    so this frame must
    look reachable.

CAMERAS. Only camera_top and camera_left are shown, because those are the only
two the exporter writes (observation.images.top / .wrist). camera_scan looks at
the packet mat, is excluded from training, and showing it here would invite
judging the data on pixels the model never receives.

FRAME SEEKING. cv2's h264 seek lands on a keyframe, which can be most of a
second away -- at t_close that is the difference between "holding the bag" and
"about to touch it". So targets are visited in ascending order and decoded
FORWARD into position: seek to a little before, then read frame by frame to the
exact index. Accurate like a full decode, ~50x cheaper.

Usage:
    uv run python tools/review_grasps.py
    uv run python tools/review_grasps.py --pre-s 2.0 --post-s 1.0 --open
    uv run python tools/review_grasps.py --hover-x 0.43 --hover-y -0.25 --hover-z 0.17
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robots_realtime.labeling import constants as C
from tools.export_lerobot import (  # noqa: E402
    CAMERAS,
    episode_dirs,
    grasp_windows,
    load_json,
    usable_grasps,
)

THUMB_W = 300
JPEG_Q = 82

# Coverage grid. 6x6 over the mat region keeps cells ~3cm, which is the scale
# the packets were actually placed at; finer just makes every cell read n=1.
GRID_N = 6


# ── collect ─────────────────────────────────────────────────────────────────
def collect(root: Path, pre_s: float, post_s: float):
    """One record per grasp window, in exporter order. Also the rejected episodes."""
    records, rejects = [], []

    for ep in episode_dirs(root):
        # workspace_gate=False so the dropped grasps still appear, badged, with
        # the reason visible. The gate itself is applied below from the same
        # constant the exporter uses -- never re-derived here.
        grasps, why = usable_grasps(ep, workspace_gate=False)
        if why:
            rejects.append((ep.name, why))
            continue

        stamps = {}
        for cam in CAMERAS:
            f = ep / f"{cam}-rgb-timestamp.npy"
            if f.exists():
                stamps[cam] = np.load(f)
        if not stamps:
            rejects.append((ep.name, "no camera timestamps"))
            continue

        # Window clipping needs the recorded span. The camera streams bracket it
        # closely enough for review, and using them avoids opening both mcaps
        # (~2s per episode) just to read two numbers.
        t0 = max(float(s[0]) for s in stamps.values())
        t1 = min(float(s[-1]) for s in stamps.values())

        ordered = sorted((g for g in grasps if g.get("t") is not None),
                         key=lambda g: float(g["t"]))
        windows = grasp_windows(ordered, t0, t1, pre_s, post_s)
        if len(windows) != len(ordered):
            # grasp_windows drops any window that clips to nothing; without a
            # 1:1 match we cannot say which grasp a window belongs to.
            rejects.append((ep.name, f"{len(ordered)} grasps -> {len(windows)} windows"))
            continue

        ann = load_json(ep / "annotations.json") or {}
        kit = {k.get("bag_id"): k for k in ((ann.get("episode_meta") or {}).get("kitting_list") or [])}
        flags = load_json(ep / "operator_flags.json") or {}
        ep_tags = sorted({f.get("tag") for f in flags.get("flags", []) if f.get("tag")})

        for i, (g, (lo, hi)) in enumerate(zip(ordered, windows)):
            pose = g.get("ee_pose") or []
            x, y, z = (list(pose[:3]) + [None, None, None])[:3]
            bag = kit.get(g.get("bag_id")) or {}
            excluded = x is not None and x < C.GRASP_WORKSPACE_X_MIN
            records.append({
                "ep": ep.name,
                "date": ep.parent.name,
                "dir": str(ep),
                "idx": i,
                "uid": f"{ep.name}#{i}",
                "bag_id": g.get("bag_id"),
                "part_no": bag.get("part_no") or "",
                "part_name": bag.get("name") or "",
                "attempt": g.get("attempt"),
                "t_close": float(g["t"]),
                "t_lo": lo, "t_hi": hi,
                "dur": hi - lo,
                "x": x, "y": y, "z": z,
                "excluded": excluded,
                "ep_tags": ep_tags,
                "frames": {},
            })
    return records, rejects


# ── frames ──────────────────────────────────────────────────────────────────
def grab_frames(records, outdir: Path, quiet: bool = False):
    """Decode the 3 moments x 2 cameras for every record. One pass per video."""
    outdir.mkdir(parents=True, exist_ok=True)
    by_ep: dict[str, list] = {}
    for r in records:
        by_ep.setdefault(r["dir"], []).append(r)

    for n, (epdir, recs) in enumerate(sorted(by_ep.items()), 1):
        ep = Path(epdir)
        if not quiet:
            print(f"  [{n}/{len(by_ep)}] {ep.name}  ({len(recs)} grasps)", flush=True)

        for cam, suffix in CAMERAS.items():
            mp4 = ep / f"{cam}-images-rgb.mp4"
            npy = ep / f"{cam}-rgb-timestamp.npy"
            if not mp4.exists() or not npy.exists():
                continue
            ts = np.load(npy)

            # (frame index, record, moment) sorted ascending so one forward pass
            # can satisfy every target.
            targets = []
            for r in recs:
                for moment, t in (("lo", r["t_lo"]), ("close", r["t_close"]), ("hi", r["t_hi"])):
                    targets.append((int(np.argmin(np.abs(ts - t))), r, moment))
            targets.sort(key=lambda a: a[0])

            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                continue
            pos = -1
            try:
                for idx, r, moment in targets:
                    if pos < 0 or idx < pos or idx - pos > 90:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 45))
                        pos = max(0, idx - 45)
                    frame = None
                    while pos <= idx:
                        ok, f = cap.read()
                        if not ok:
                            break
                        frame = f
                        pos += 1
                    if frame is None:
                        continue
                    h, w = frame.shape[:2]
                    thumb = cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))),
                                       interpolation=cv2.INTER_AREA)
                    name = f"{r['ep']}_{r['idx']}_{suffix}_{moment}.jpg"
                    cv2.imwrite(str(outdir / name), thumb,
                                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
                    r["frames"][f"{suffix}_{moment}"] = name
            finally:
                cap.release()


# ── coverage ────────────────────────────────────────────────────────────────
def coverage(records, hover):
    """Bin kept grasps into a GRID_N x GRID_N map and locate the handoff pose.

    Bounds come from the KEPT grasps only. Including the excluded outliers would
    stretch the map 30cm to the west and squash everything real into two cells.
    """
    kept = [r for r in records if not r["excluded"] and r["x"] is not None]
    if not kept:
        return None
    xs = [r["x"] for r in kept]
    ys = [r["y"] for r in kept]
    pad = 1e-6
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad

    def cell(x, y):
        cx = int((x - x0) / (x1 - x0) * GRID_N)
        cy = int((y - y0) / (y1 - y0) * GRID_N)
        if not (0 <= cx < GRID_N and 0 <= cy < GRID_N):
            return None
        return cx, cy

    counts = [[0] * GRID_N for _ in range(GRID_N)]
    for r in kept:
        c = cell(r["x"], r["y"])
        if c:
            counts[c[0]][c[1]] += 1
            r["cell"] = f"{c[0]},{c[1]}"

    # The verdict reads the 3x3 NEIGHBOURHOOD, not the single cell. Cells are
    # ~30x49mm while the grasp scatter is ~80mm per axis and the calibration is
    # good to ~5mm -- so one cell is inside the noise, and a dense region with a
    # single empty cell in it would otherwise read as "no data here". Measured
    # case: the corpus mean (0.43, -0.25) lands in a cell of n=1 whose neighbours
    # hold 25 grasps. Reporting n=1 there would be true and useless.
    verdict, hcell, hood = "NO HANDOFF POSE GIVEN", None, None
    if hover is not None:
        c = cell(hover[0], hover[1])
        if c is None:
            verdict = "OUT OF DISTRIBUTION — outside the grasped region entirely"
        else:
            hcell = c
            hood = sum(counts[i][j]
                       for i in range(max(0, c[0] - 1), min(GRID_N, c[0] + 2))
                       for j in range(max(0, c[1] - 1), min(GRID_N, c[1] + 2)))
            n = counts[c[0]][c[1]]
            if hood == 0:
                verdict = "OUT OF DISTRIBUTION — no grasps within one cell"
            elif hood < 5:
                verdict = f"THIN — only {hood} grasps nearby (cell n={n})"
            else:
                verdict = f"IN DISTRIBUTION — {hood} grasps nearby (cell n={n})"

    return {"counts": counts, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "hcell": hcell, "hood": hood, "verdict": verdict,
            "hover": hover, "n": len(kept)}


# ── render ──────────────────────────────────────────────────────────────────
def render(records, rejects, cov, args, outdir: Path):
    kept = [r for r in records if not r["excluded"]]
    esc = html.escape

    def cov_html():
        if not cov:
            return "<p class='muted'>No kept grasps to map.</p>"
        c = cov
        vk = "good" if c["verdict"].startswith("IN DIST") else (
            "warn" if c["verdict"].startswith("THIN") else "bad")
        cells = []
        mx = max(max(row) for row in c["counts"]) or 1
        for cx in range(GRID_N - 1, -1, -1):          # high x at the top
            for cy in range(GRID_N):
                n = c["counts"][cx][cy]
                lvl = 0 if n == 0 else 1 + int(3.99 * n / mx)
                mark = " hov" if c["hcell"] == (cx, cy) else ""
                xa = c["x0"] + (c["x1"] - c["x0"]) * cx / GRID_N
                ya = c["y0"] + (c["y1"] - c["y0"]) * cy / GRID_N
                title = f"x {xa:+.3f} y {ya:+.3f} — {n} grasp{'' if n == 1 else 's'}"
                dis = "" if n else " disabled"
                cells.append(
                    f'<button class="cell l{lvl}{mark}" data-cell="{cx},{cy}"'
                    f' title="{title}"{dis}>{n or ""}</button>')
        vclass = vk
        hov = ""
        if c["hover"]:
            hov = (f'<div class="hovline">handoff pose '
                   f'<code>x {c["hover"][0]:+.3f}  y {c["hover"][1]:+.3f}  '
                   f'z {c["hover"][2]:+.3f}</code> '
                   f'<span class="verdict {vclass}">{esc(c["verdict"])}</span></div>')
        return (f'<div class="covwrap"><div class="axis-y">y (lateral) →</div>'
                f'<div class="covgrid">{"".join(cells)}</div>'
                f'<div class="axis-x">x (forward) ↑</div></div>{hov}'
                f'<p class="muted">Click a cell to filter. Bounds x '
                f'{c["x0"]:+.3f}..{c["x1"]:+.3f}, y {c["y0"]:+.3f}..{c["y1"]:+.3f} m, '
                f'{c["n"]} kept grasps.</p>')

    def card(r):
        thumbs = []
        for suffix, label in (("top", "top"), ("wrist", "wrist")):
            row = []
            for moment, mlabel in (("lo", "start"), ("close", "GRASP"), ("hi", "lift")):
                f = r["frames"].get(f"{suffix}_{moment}")
                cls = "shot key" if moment == "close" else "shot"
                if f:
                    row.append(f'<figure class="{cls}"><img loading="lazy" '
                               f'src="frames/{esc(f)}" alt="{suffix} {mlabel}">'
                               f'<figcaption>{mlabel}</figcaption></figure>')
                else:
                    row.append(f'<figure class="{cls} miss"><div class="ph">no frame</div>'
                               f'<figcaption>{mlabel}</figcaption></figure>')
            thumbs.append(f'<div class="camrow"><span class="camlabel">{label}</span>'
                          f'<div class="shots">{"".join(row)}</div></div>')

        badges = []
        if r["excluded"]:
            badges.append('<span class="badge bad">excluded · x &lt; 0.25</span>')
        for t in r["ep_tags"]:
            badges.append(f'<span class="badge warn">flag {esc(t)}</span>')
        if r["dur"] < args.pre_s + args.post_s - 0.05:
            badges.append(f'<span class="badge warn">clipped {r["dur"]:.2f}s</span>')

        checked = "" if r["excluded"] else " checked"
        pos = (f'{r["x"]:+.3f} {r["y"]:+.3f} {r["z"]:+.3f}'
               if r["x"] is not None else "no pose")
        part = ""
        if r["part_no"]:
            part = (f'<div class="part">{esc(r["part_no"])} '
                    f'{esc(r["part_name"])}</div>')
        excl_cls = " excl" if r["excluded"] else ""
        return (
            f'<article class="card{excl_cls}" '
            f'data-uid="{esc(r["uid"])}" data-ep="{esc(r["ep"])}" '
            f'data-cell="{esc(r.get("cell", ""))}" data-excl="{int(r["excluded"])}">'
            f'<header><label class="pick"><input type="checkbox"{checked}>'
            f'<span class="uid">{esc(r["ep"])}</span>'
            f'<span class="gi">#{r["idx"]}</span></label>'
            f'<div class="meta"><code>{pos}</code>'
            f'<span class="sep">·</span><span>{r["dur"]:.2f}s</span>'
            f'<span class="sep">·</span><span>bag {r["bag_id"]}</span>'
            f'{" ".join(badges)}</div>{part}'
            f'</header>{"".join(thumbs)}</article>')

    rej = "".join(f"<tr><td><code>{esc(e)}</code></td><td>{esc(w)}</td></tr>"
                  for e, w in rejects)

    payload = base64.b64encode(json.dumps(
        [{"uid": r["uid"], "ep": r["ep"], "dir": r["dir"], "idx": r["idx"],
          "t_close": r["t_close"], "t_lo": r["t_lo"], "t_hi": r["t_hi"],
          "excluded": r["excluded"]} for r in records]
    ).encode()).decode()

    return TEMPLATE.format(
        n_total=len(records), n_kept=len(kept),
        n_excl=len(records) - len(kept), n_eps=len({r["ep"] for r in records}),
        n_rej=len(rejects), pre=args.pre_s, post=args.post_s, grid=GRID_N,
        coverage=cov_html(), cards="".join(card(r) for r in records),
        rejects=rej or '<tr><td colspan="2" class="muted">none</td></tr>',
        payload=payload)


TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>Grasp review — {n_kept} training windows</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {{
  --bg:#0e1113; --panel:#161a1d; --line:#252b30; --line2:#1b2024;
  --ink:#e6e9ea; --dim:#8d979c; --faint:#5e686d;
  --ok:#5db87a; --warn:#d8a13e; --bad:#e0705c; --acc:#4aa8b8;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: light) {{
  :root {{ --bg:#f4f5f3; --panel:#fff; --line:#dcdfdc; --line2:#eceeec;
    --ink:#171a1b; --dim:#5c6568; --faint:#8c9497;
    --ok:#2f7d4a; --warn:#8a6410; --bad:#a8402c; --acc:#0f6d7d; }}
}}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; padding:22px }}
.wrap {{ max-width:1500px; margin:0 auto }}
h1 {{ font-size:19px; font-weight:650; letter-spacing:-.01em }}
h2 {{ font-size:11px; font-weight:600; letter-spacing:.13em; text-transform:uppercase;
     color:var(--dim); margin:26px 0 11px; padding-bottom:6px; border-bottom:1px solid var(--line) }}
code {{ font-family:var(--mono); font-size:12px }}
.muted {{ color:var(--faint); font-size:12px; margin-top:8px }}
.sep {{ color:var(--faint); margin:0 3px }}

.stats {{ display:flex; flex-wrap:wrap; gap:22px; margin:12px 0 4px; font-family:var(--mono); font-size:12px; color:var(--dim) }}
.stats b {{ color:var(--ink); font-weight:650; font-size:15px }}

.bar {{ position:sticky; top:0; z-index:20; background:var(--panel); border:1px solid var(--line);
  border-radius:9px; padding:11px 13px; margin:16px 0 6px; display:flex; flex-wrap:wrap; gap:9px; align-items:center }}
button {{ font:inherit; font-size:12.5px; color:var(--ink); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:6px 11px; cursor:pointer }}
button:hover:not(:disabled) {{ border-color:var(--acc); color:var(--acc) }}
button:disabled {{ opacity:.32; cursor:default }}
button.on {{ background:var(--acc); border-color:var(--acc); color:#04191d; font-weight:600 }}
.count {{ margin-left:auto; font-family:var(--mono); font-size:12px; color:var(--dim) }}
.count b {{ color:var(--acc); font-size:14px }}

.covwrap {{ display:inline-grid; grid-template-columns:auto auto; grid-template-rows:auto auto;
  gap:6px; align-items:center }}
.covgrid {{ display:grid; grid-template-columns:repeat({grid},34px); gap:3px }}
.axis-y {{ grid-column:2; font-size:10px; color:var(--faint); font-family:var(--mono) }}
.axis-x {{ grid-column:1; grid-row:2; writing-mode:vertical-rl; transform:rotate(180deg);
  font-size:10px; color:var(--faint); font-family:var(--mono) }}
.cell {{ height:34px; padding:0; border-radius:4px; font-family:var(--mono); font-size:11px }}
.cell.l0 {{ background:var(--line2); color:var(--faint); border-color:var(--line2) }}
.cell.l1 {{ background:color-mix(in srgb,var(--acc) 16%,transparent) }}
.cell.l2 {{ background:color-mix(in srgb,var(--acc) 34%,transparent) }}
.cell.l3 {{ background:color-mix(in srgb,var(--acc) 56%,transparent) }}
.cell.l4 {{ background:color-mix(in srgb,var(--acc) 78%,transparent); color:#04191d; font-weight:700 }}
.cell.hov {{ outline:2px solid var(--warn); outline-offset:1px }}
.hovline {{ margin-top:11px; font-size:12.5px }}
.verdict {{ font-family:var(--mono); font-size:11px; font-weight:700; letter-spacing:.06em;
  padding:3px 8px; border-radius:4px; margin-left:7px }}
.verdict.good {{ background:color-mix(in srgb,var(--ok) 20%,transparent); color:var(--ok) }}
.verdict.warn {{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn) }}
.verdict.bad {{ background:color-mix(in srgb,var(--bad) 20%,transparent); color:var(--bad) }}

.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(470px,1fr)); gap:11px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:11px }}
.card.excl {{ opacity:.5 }}
.card.sel {{ border-color:var(--acc) }}
.card.hide {{ display:none }}
.pick {{ display:flex; align-items:center; gap:7px; cursor:pointer }}
.pick input {{ width:15px; height:15px; accent-color:var(--acc); cursor:pointer }}
.uid {{ font-family:var(--mono); font-size:12px }}
.gi {{ font-family:var(--mono); font-size:11px; color:var(--faint) }}
.meta {{ display:flex; flex-wrap:wrap; align-items:center; gap:5px; margin-top:5px;
  font-size:11.5px; color:var(--dim) }}
.part {{ font-family:var(--mono); font-size:11px; color:var(--faint); margin-top:3px }}
.badge {{ font-family:var(--mono); font-size:10px; font-weight:600; padding:2px 6px; border-radius:3px }}
.badge.bad {{ background:color-mix(in srgb,var(--bad) 20%,transparent); color:var(--bad) }}
.badge.warn {{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn) }}

.camrow {{ display:flex; align-items:center; gap:8px; margin-top:9px }}
.camlabel {{ font-family:var(--mono); font-size:10px; color:var(--faint);
  writing-mode:vertical-rl; transform:rotate(180deg); letter-spacing:.09em; text-transform:uppercase }}
.shots {{ display:grid; grid-template-columns:repeat(3,1fr); gap:4px; flex:1 }}
.shot img {{ width:100%; display:block; border-radius:4px; background:var(--line2) }}
.shot figcaption {{ font-family:var(--mono); font-size:9px; color:var(--faint);
  text-align:center; margin-top:2px; letter-spacing:.05em }}
.shot.key img {{ outline:1.5px solid var(--acc); outline-offset:-1.5px }}
.shot.key figcaption {{ color:var(--acc); font-weight:700 }}
.ph {{ aspect-ratio:16/9; background:var(--line2); border-radius:4px; display:grid;
  place-items:center; font-size:10px; color:var(--faint) }}

table {{ border-collapse:collapse; font-size:12.5px; width:100% }}
td {{ padding:6px 12px 6px 0; border-bottom:1px solid var(--line2); color:var(--dim) }}
dialog {{ background:var(--panel); color:var(--ink); border:1px solid var(--line);
  border-radius:10px; padding:16px; max-width:min(760px,92vw) }}
dialog::backdrop {{ background:#000a; }}
textarea {{ width:100%; height:290px; background:var(--bg); color:var(--ink); font-family:var(--mono);
  font-size:11.5px; border:1px solid var(--line); border-radius:6px; padding:9px; resize:vertical }}
.dlgbar {{ display:flex; gap:8px; margin-top:11px; align-items:center }}
</style>
<div class="wrap">
<h1>Grasp review</h1>
<div class="stats">
  <span><b>{n_kept}</b> training windows</span>
  <span><b>{n_total}</b> grasps found</span>
  <span><b>{n_excl}</b> excluded</span>
  <span><b>{n_eps}</b> episodes</span>
  <span><b>{n_rej}</b> episodes rejected</span>
  <span>window &minus;{pre}s / +{post}s</span>
</div>

<h2>Where the grasps happened</h2>
{coverage}

<h2>Every window that goes into the model</h2>
<div class="bar">
  <button id="all">Select all</button>
  <button id="none">Select none</button>
  <button id="inv">Invert</button>
  <button id="fex" class="on">Hiding excluded</button>
  <button id="clr" disabled>Clear cell filter</button>
  <button id="copy">Copy selection</button>
  <span class="count"><b id="nsel">0</b> selected</span>
</div>
<div class="grid" id="grid">{cards}</div>

<h2>Episodes that never made it this far</h2>
<table><tbody>{rejects}</tbody></table>

<dialog id="dlg">
  <b>Selected grasp windows</b>
  <p class="muted">Episode list first, then the full keep-list as JSON.</p>
  <textarea id="out" readonly></textarea>
  <div class="dlgbar"><button id="cp">Copy to clipboard</button>
    <button id="close">Close</button><span class="muted" id="cpmsg"></span></div>
</dialog>
</div>
<script>
const DATA = JSON.parse(atob("{payload}"));
const byUid = Object.fromEntries(DATA.map(d => [d.uid, d]));
const cards = [...document.querySelectorAll('.card')];
let cellFilter = null, hideExcluded = true;

const boxOf = c => c.querySelector('input');
const sel = () => cards.filter(c => boxOf(c).checked);

function paint() {{
  cards.forEach(c => {{
    const okCell = !cellFilter || c.dataset.cell === cellFilter;
    const okExcl = !hideExcluded || c.dataset.excl === '0';
    c.classList.toggle('hide', !(okCell && okExcl));
    c.classList.toggle('sel', boxOf(c).checked);
  }});
  document.getElementById('nsel').textContent = sel().length;
}}

document.getElementById('grid').addEventListener('change', paint);

const visible = () => cards.filter(c => !c.classList.contains('hide'));
document.getElementById('all').onclick  = () => {{ visible().forEach(c => boxOf(c).checked = true);  paint(); }};
document.getElementById('none').onclick = () => {{ visible().forEach(c => boxOf(c).checked = false); paint(); }};
document.getElementById('inv').onclick  = () => {{ visible().forEach(c => boxOf(c).checked = !boxOf(c).checked); paint(); }};

const fex = document.getElementById('fex');
fex.onclick = () => {{
  hideExcluded = !hideExcluded;
  fex.classList.toggle('on', hideExcluded);
  fex.textContent = hideExcluded ? 'Hiding excluded' : 'Showing excluded';
  paint();
}};

const clr = document.getElementById('clr');
document.querySelectorAll('.cell[data-cell]:not([disabled])').forEach(b => {{
  b.onclick = () => {{
    cellFilter = (cellFilter === b.dataset.cell) ? null : b.dataset.cell;
    document.querySelectorAll('.cell').forEach(o => o.classList.toggle('on', o.dataset.cell === cellFilter));
    clr.disabled = !cellFilter;
    paint();
  }};
}});
clr.onclick = () => {{
  cellFilter = null;
  document.querySelectorAll('.cell').forEach(o => o.classList.remove('on'));
  clr.disabled = true; paint();
}};

const dlg = document.getElementById('dlg'), out = document.getElementById('out');
document.getElementById('copy').onclick = () => {{
  const picked = sel().map(c => byUid[c.dataset.uid]);
  const eps = [...new Set(picked.map(p => p.ep))].sort();
  out.value =
    `# ${{picked.length}} grasp windows across ${{eps.length}} episodes\\n` +
    eps.join('\\n') +
    `\\n\\n# keep-list\\n` +
    JSON.stringify({{keep: picked.map(p => ({{episode: p.ep, grasp: p.idx, t_close: p.t_close}}))}}, null, 2);
  dlg.showModal();
}};
document.getElementById('cp').onclick = async () => {{
  const msg = document.getElementById('cpmsg');
  try {{ await navigator.clipboard.writeText(out.value); msg.textContent = 'Copied.'; }}
  catch {{ out.select(); msg.textContent = 'Press Cmd/Ctrl+C.'; }}
  setTimeout(() => msg.textContent = '', 2500);
}};
document.getElementById('close').onclick = () => dlg.close();

paint();
</script>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="recordings")
    ap.add_argument("--out", default=".review-grasps")
    ap.add_argument("--pre-s", type=float, default=2.0)
    ap.add_argument("--post-s", type=float, default=1.0)
    ap.add_argument("--hover-x", type=float, default=None,
                    help="planned IK handoff pose, robot base frame (metres)")
    ap.add_argument("--hover-y", type=float, default=None)
    ap.add_argument("--hover-z", type=float, default=0.17)
    ap.add_argument("--no-frames", action="store_true",
                    help="skip video decode, reuse whatever is already in out/frames")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    root, outdir = Path(a.root), Path(a.out)
    records, rejects = collect(root, a.pre_s, a.post_s)
    if not records:
        print("no grasp windows found", file=sys.stderr)
        return 1

    kept = [r for r in records if not r["excluded"]]
    print(f"{len(records)} grasps, {len(kept)} kept, {len(records) - len(kept)} excluded "
          f"(x < {C.GRASP_WORKSPACE_X_MIN}), {len(rejects)} episodes rejected")

    frames_dir = outdir / "frames"
    if a.no_frames:
        for r in records:
            for suffix in CAMERAS.values():
                for moment in ("lo", "close", "hi"):
                    f = f"{r['ep']}_{r['idx']}_{suffix}_{moment}.jpg"
                    if (frames_dir / f).exists():
                        r["frames"][f"{suffix}_{moment}"] = f
    else:
        print(f"decoding {len(records) * 6} frames...")
        grab_frames(records, frames_dir)

    hover = None
    if a.hover_x is not None and a.hover_y is not None:
        hover = (a.hover_x, a.hover_y, a.hover_z)
    cov = coverage(records, hover)
    if cov:
        print(f"coverage: {cov['verdict']}")

    outdir.mkdir(parents=True, exist_ok=True)
    index = outdir / "index.html"
    index.write_text(render(records, rejects, cov, a, outdir))
    print(f"\n  {index.resolve()}")
    if a.open:
        import webbrowser
        webbrowser.open(index.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
