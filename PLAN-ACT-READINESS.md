# PLAN v2 — ACT data-readiness: pipeline fixes + slim evidence-based analysis

Date: 2026-09-02. Status: reviewed (eng review + cross-model outside voice), operator-approved.
Operator decisions: keep recording pure teleop today (post-processing fix is retroactive);
reshape approved WITH a mandatory manual review gate before any window enters the export.

## What the review established (evidence in repo)

- Start-pose-overlap testing (the old plan's B1/A6/A7 core) was ALREADY tried and retired:
  `yam-pick-pipeline/check_handover_pose.py:33-43` — it passed a known-bad checkpoint.
- The property that predicted the one working policy: **frame index of first jaw-close vs
  the executed chunk horizon** (`check_handover_pose.py:17-31`; a 1 s pre_s shift put 46%
  of closes beyond frame 100 → policy never closed on hardware).
- Left arm executes `n_action_steps=16` (`yam-pick-pipeline/act_runner.py:129-131`);
  chunk_size (training) ≠ n_action_steps (execution). Gates must use the left config.
- LIVE BUG: `MIN_TRANSPORT_M = 0.10` (labeling/constants.py, applied fuse.py:66-69,164-174)
  silently deletes grasp-only demos ("no_transport"/"unplaced_grasp") — episode_143533's
  4 real grasps vanished this way. For an approach+grasp+lift policy this filter is wrong.
- A2/A5-style gates measured the exporter's constants, not the data (window length 150 by
  construction). Camera skew is uniform ±16 ms by construction (free-running cameras);
  a 5 ms gate fails every healthy episode. Export-count truth lives in `plan_episode`
  (export_lerobot.py:301-350 gates), not the raw window function.
- Corpus stands at 72 verified-success windows (target 80–120 + 15–20 held out).

## Work items (in order)

### W1 — grasp-mode labelling (labeller fix, small + tested)
Add an explicit grasp-task mode to the labelling path (flag or min-transport=0 override)
so approach+grasp+lift demos are kept: no MIN_TRANSPORT deletion; outcome from hold +
FK lift only. Re-label today's episodes in that mode → recovers episode_143533's grasps.
Refs must be passed explicitly (qa_label.py hardcodes 1.0/0.0; live_server passes its
own — one source of truth). Never run concurrent with live_server writing the same dir;
annotation writes become atomic (tmp+rename, schema.py:172-173 today is write_text).
Tests: fixture with grasp-no-transport → kept in grasp mode, dropped in kitting mode;
atomic-write test.

### W2 — window definition + close-index gate (exporter change, tested)
Replace/augment the fixed [t_close−3s, +2s] window: pose-predicate start (last crossing
of a pre-grasp plane + 10–20 frame margin) with the −3 s clamp as fallback; and a HARD
export gate: first jaw-close frame index < executed horizon for the LEFT config (16-step
slices re-queried, so the binding constraint is close-index within chunk AND within the
demo's supervision-rich region — gate on close_idx ≤ 0.8 × chunk frames, report the
distribution). Tests: synthetic fast/slow approaches; regression: today's episodes.

### W3 — slim readiness tool (`tools/analyze_act_readiness.py`, read-only)
Layered ON `plan_episode`/dry-run (production filters, not reimplemented). Reports:
- close-frame-index distribution vs chunk (the decisive check, per W2 definition)
- window-start FK spread (report-only, no 1–3 cm gate — free-play teleop is expected
  to be wide; the number that matters is post-W2 windows)
- gripper span/bimodality (real dead-channel floor from constants.py), honest per-stream
  cadence (per-stream nominal, gaps; skew REPORTED not gated)
- corpus totals: windows by outcome/filter-reason, progress to 80–120 + held-out 15–20
- episode states: OK / UNLABELLED / INELIGIBLE (no timestamps: killed session) / CORRUPT;
  .trash excluded (exporter parity, export_lerobot.py:205-208); cwd-independent FK (pass
  urdf path explicitly, fk.py:73 default is cwd-relative). No npz cache (measured: full
  corpus ≈ 5 min; cache was unearned complexity + staleness trap). No video decode.
Tests: dead-gripper FAIL, close-past-chunk FAIL, healthy PASS, real-episode regression.

### W4 — manual review gate (operator-required, after recording ends)
`review_grasps.py` with pre/post MATCHED to the W2 window (imported, drift-proof) →
operator clicks keep/reject on every window → `export_lerobot --keep <list>`.
No window enters the training set without a human eye on its card. Heavy decode: runs
only after the session ends.

## NOT in scope
- IK-handoff recording (next session, own plan — operator deferred).
- p_tool revalidation; B-phase camera checks (post-recording, only if W3 raises questions).
- Any episode deletion; dashboards.

## What already exists (reused)
`plan_episode`/dry-run filters, `review_grasps.py` + keep-list, labelling stack
(hysteresis/hold/FK-lift — only the transport filter is touched), FK, `check_handover_pose.py`
(the close-index precedent), `review_corpus.py`.

## Failure modes
- W1 flag misused on kitting data → kitting export unaffected (mode is explicit, default
  unchanged); test pins default behavior.
- W2 pose-predicate misfires on an odd approach → falls back to −3 s clamp; gate then
  catches a late close rather than exporting it silently.
- Concurrent labelling vs live session → atomic writes + eligibility rule; documented.
- Tool import drift from exporter → single-source imports + regression test on real episode.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | FAILED (401 auth) | fell back to Claude subagent |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 6 issues (all folded, auto-decided per operator instruction) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **CROSS-MODEL:** outside voice (Claude subagent, fresh context, repo-verified) found 19
  issues the review missed, incl. 10 P1s: retired-check resurrection, wrong chunk number,
  false-alarming gates, a live labeller data-loss bug (MIN_TRANSPORT), unsafe --label
  concurrency, vacuous gates. Plan rewritten to v2 around its findings; the two operator
  decisions were put to the operator (recording flow: keep pure teleop; reshape: approved
  with manual-review gate). Tension resolution: outside voice adopted on all repo-verified
  points; its "IK-handoff now" strategic push was deferred by the operator to a next session.
- **VERDICT:** ENG CLEARED (v2) — ready to implement W1→W4.

**UNRESOLVED DECISIONS:**
- W2 gate constant (0.8 × chunk frames) is a starting point — revisit against the
  known-good corpus during implementation before pinning the number.
