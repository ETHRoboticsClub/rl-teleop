# ACT data-readiness report — left-arm grasp corpus, 2026-09-02

Read-only analysis of `recordings/20260902`. Nothing was exported, nothing was trained,
no device or bus was touched, and no episode was deleted. Writes were limited to the
labeller's own sidecars (`annotations.json`, `qa.json`) on finished episodes.

**Scope — the session is still recording.** Counts below are the **11 episodes finished
as of 17:12** (all labelled). They will grow; re-run the two tools when the session
closes. The conclusions are structural and did not move between the 9-episode and
11-episode passes — §3's central finding got stronger, not weaker.

*(Revision history: first written against the 9 episodes finished at 16:46. Every count
in §0–§4 has been re-derived on all 11. §6 step 2's episode gate was corrected — see the
`degraded`-flag note in §4.)*

Reproduce every number below with:

```bash
cd ~/Desktop/kitting-v2/rl-teleop
./.venv/bin/python3 tools/analyze_act_readiness.py --root recordings/20260902
./.venv/bin/python3 tools/act_horizon_report.py     --root recordings/20260902 --pre-s 6.0
./.venv/bin/python3 tools/act_horizon_report.py     --root recordings/20260902 --pre-s 6.0 \
    --keep recordings/20260902/keep-20260902.json          # §7, the operator's selection
./.venv/bin/python3 tools/export_lerobot.py --root recordings/20260902 --dry-run
```

`tools/act_horizon_report.py` is new and is the only thing added. It is read-only, it
calls `export_lerobot.plan_episode` (and its keep-list filter) rather than reimplementing
anything, and it exists to answer the numbers this report is about: `chunk_size`,
`n_action_steps`, `--close-idx-frac`, per-window camera frame yield, and the handoff
pose.

---

## 0. Headline

The corpus is **large enough and mechanically healthy**, the operator review is **done**,
and the export is **ready to run** at the settings in §8 — but not at the exporter's
current defaults, three of which are wrong for this data.

| | |
|---|---|
| Finished episodes labelled | 11 |
| Usable episodes | 9 — two were aborted with a dead gripper channel |
| Successful grasps | 189 |
| Kept by the operator review | 179 (`data_grasping_left.json`) |
| − camera-starved (`episode_155339`) | −16 |
| − short hold (`hold_s` < 1.5 s) | −4 |
| − close-index gate (descent > one chunk) | −19 |
| **Exportable training windows** | **140**, across 8 episodes |
| Corpus target from PLAN-ACT-READINESS | 80–120 training + 15–20 held out — **MET** |
| Handoff pose | packet-relative `[-46, +33, +103] mm`, p90 error 76 mm — §7 |
| Window length | ~2 s approach + ~2 s lift ≈ 4 s, `chunk_size = 100` |
| Verdict | **export now**; §9's four decisions are about the *tooling*, not this dataset |

Reconciliation, which is the check that matters: `analyze_act_readiness.py` and
`export_lerobot --dry-run` report the same 189 grasp-mode windows and the same 142 in
grasp-pose mode. The report and the thing that writes the dataset agree.

The single episode from `recordings/20260901` is the same task but is dead-gripper /
aborted, so `recordings/20260902` is the entire usable left-arm grasp corpus.

---

## 1. What was labelled

All nine finished episodes were re-labelled with `qa_label.py --task grasp`. Grasp mode
recovered `episode_143533`'s 4 grasps, which kitting mode had deleted via
`MIN_TRANSPORT_M` exactly as the plan predicted.

| episode | grasps | success | outcome | hold_s median | note |
|---|---|---|---|---|---|
| `episode_143533_ee94747f` | 4 | 4 | success | 3.86 | recovered by grasp mode |
| `episode_144346_839a5930` | 0 | 0 | aborted | — | **gripper channel dead** (208 s) |
| `episode_144715_4e3ba94f` | 0 | 0 | aborted | — | **gripper channel dead** (57 s) |
| `episode_144835_983d9d07` | 30 | 30 | success | 2.89 | |
| `episode_145907_21b66012` | 42 | 42 | success | 3.85 | |
| `episode_150947_4d8802e5` | 35 | 35 | success | 4.14 | |
| `episode_152731_bb5a03ba` | 15 | 13 | success | 4.98 | |
| `episode_155339_ad02fb72` | 17 | 17 | success | 5.18 | **camera starved — reject, see §4** |
| `episode_163624_4d26b4c8` | 12 | 12 | success | 5.18 | |
| `episode_164621_d1329cef` | 5 | 5 | success | 5.42 | `degraded` flag set, but 100% frame yield — **keep** |
| `episode_165925_7a4696f0` | 31 | 31 | success | 4.48 | |

The two aborted episodes have a gripper span of **exactly 0.0000** against a
dead-channel floor of 0.0499 — the channel was frozen for the whole take, not merely
unused. They bracket 14:43–14:48, and the memory note *"flat gripper = wrong handle
first"* (leader-handle identity, 2026-09-02) names this signature. Four and a half
minutes of teleop were recorded against a leader whose gripper was not driving the
follower. Both are correctly rejected by the exporter as `outcome=aborted`; no action
needed beyond recognising the signature faster next time.

---

## 2. (a) Seconds per action step, `n_action_steps`, `chunk_size`

**One action step = 1/30 s = 33.3 ms.** That is fixed by `DEFAULT_FPS = 30`, the grid
every stream is resampled onto.

From that: `n_action_steps = 16` commits **0.533 s** of blind execution per query, and
`chunk_size = 100` plans **3.33 s** ahead.

### Keep `n_action_steps = 16`

Measured over all 189 windows: the shortest **approach** (window start → jaw close) is
39 frames (1.30 s) and the shortest **lift** (close → window end) is 59 frames (1.97 s).
Both exceed 16 frames, in every single window. The runtime therefore never commits blind
through an entire phase of the task — the fastest demo still gets 2.4 re-observations
during its approach. Median window is 124 frames, i.e. ~7.8 replans per grasp.

Nothing in this data argues for changing 16. It is already an override applied at load
time in `act_runner.py:129-131`, and it does not need to match `chunk_size`.

### Keep `chunk_size = 100`

`chunk_size` has to cover the behaviour one plan must express: from the handoff pose to
the jaw close. Measured, with the exporter's clamp removed so the number is the real
descent and not an artifact (`--pre-s 6`):

```
approach duration   n=189   min 39   p10 48   median 64   p90 105   max 195 frames
                            1.30 s        2.13 s median        6.50 s max
```

At `chunk_size = 100` (3.33 s), 165 of 189 windows have their close inside one chunk
measured from their own start. Shrinking the chunk is expensive and buys nothing: at
`chunk_size = 60` only ~half the corpus survives the same criterion. 100 is also what
the only checkpoint that has ever worked on this rig was trained with, and what the
deployed left policy already carries, so keeping it means the retrain changes one
number (`--close-idx-frac`) rather than three.

### The 16-vs-100 distinction, stated once

The handoff is right that `chunk_size ≠ n_action_steps`, and there is a second half to
that which matters for how the close-index gate should be read:

`check_handover_pose.py` argues *"one committed chunk cannot reach a close it was never
shown"*. That argument is literally true **on the right arm**, where `act_runner.py:182`
sets `n_action_steps = 100 = chunk_size` — the runner commits the entire plan blind, so
a close at frame 120 is physically unreachable. **On the left arm it is not literally
true**: with `n_action_steps = 16` the policy re-observes every 0.53 s and can walk
forward indefinitely. The left-arm justification for bounding the close index is weaker
and different — it is about whether the policy is ever supervised to emit a *close* from
a handoff-like observation, not about running out of plan. This matters because the gate
threshold has been reasoned about as if the right-arm mechanism applied. It does not.

---

## 3. (c) The right `--close-idx-frac` — and why 0.8 is measuring the exporter

### The trap

`close_idx` is bounded by the exporter's own `--pre-s`:

```
close_idx ≤ pre_s × fps        = 90 at the default pre_s = 3.0
```

Measured on the fixed window (`--window-mode grasp`, the export shape the working
right-arm corpus used): **181 of 189 windows have `close_idx` exactly 90.**

So the belief that *"the corpus behind the only working policy sat at ~0.90"* is
arithmetic, not evidence: 0.90 = `pre_s × fps / chunk_size` = 3.0 × 30 / 100. Every
unclipped window in a fixed-window export lands there by construction. It contains no
information about where a threshold should sit, and the same is true of the three
right-arm corpora in `check_handover_pose.py`'s header — the two grasp corpora read
"median 90, p90 90" because they were exported with `pre_s = 3`, and the pickplace
corpus read "median 92, p90 120" because it was exported with `pre_s = 4`. The
distributions are reporting the export flag.

### What the 0.8 gate actually removes

At `pre_s = 3.0`, `chunk = 100`, gate = frame 80, on all 189 windows:

| dropped | count | what it is |
|---|---|---|
| `start=clamp` | 20 | the pose predicate found **no** descent inside its 3 s search span |
| `start=clamped` | 15 | it found one, but earlier than the 3 s floor |
| `start=pose` | 12 | genuine: a measured descent of 81–89 frames |
| **total** | **47** | 25% of the corpus |

`close_idx` by start mode makes it plain — a clamp window sits *at the ceiling by
construction*:

```
pose      n=153   min 39   p10 47   median 60   p90 77   max 89
clamp     n=20    min 90                                 max 90
clamped   n=16    min 79                                 max 90
```

So **three quarters of what the 0.8 gate removes, it removes because the pose predicate
hit its search-span limit** — not because the close was late. And at `pre_s = 3` no
window can possibly exceed the 100-frame chunk (the ceiling is 90), so the gate cannot
express the property its docstring describes.

### The failure the gate was built for is still in the data — hidden

Re-plan with the search span widened so the clamp never binds:

| `--pre-s` | ceiling | pose starts | windows with `close_idx > 100` |
|---|---|---|---|
| 3.0 | 90 | 153/189 | **0** — impossible, the ceiling is 90 |
| 4.0 | 120 | 178/189 | 24 |
| 6.0 | 180 | 188/189 | 24 |
| 10.0 | 300 | 189/189 | 24 |

**24 of 189 windows (13%) have a real final descent longer than one 100-frame chunk.**
At `pre_s = 3` they are invisible: each is silently truncated to exactly frame 90, and
its window opens mid-descent at an arbitrary height rather than at a pre-grasp pose. The
gate is currently blind to precisely the population it exists to catch, and it pays for
that blindness by deleting 19 healthy windows instead.

### Recommendation

```bash
./.venv/bin/python3 tools/export_lerobot.py --root recordings/20260902 \
    --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
    --chunk-frames 100 --close-idx-frac 1.00 \
    --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
    --keep <keep-list from review_grasps.py> --repo-id ETHRC/yam_grasp_left_v1
```

- **`--pre-s 6.0`** — `pre_s` is only a *search span* for the pose predicate. Widening it
  changes nothing for the 104 windows that already start at a measured descent; it
  converts the other 20 from arbitrary mid-descent starts into real ones. At 6.0 the
  predicate succeeds on 124/124.
- **`--close-idx-frac 1.00`** (gate = frame 100 = `chunk_size`) — now the gate means
  what it says: reject a window whose *measured* descent cannot fit in one chunk. It
  rejects the 24 genuinely-long descents and keeps 165.
- **Not 0.8.** At `pre_s = 6` the 0.8 gate would keep 142 and reject 47, of which 23 are
  ordinary 81–100-frame approaches with nothing wrong with them. There is no evidence
  in this corpus that a slower approach is a worse demo — `close_idx` and `hold_s`
  correlate at **+0.14**, i.e. not at all — so a threshold below the chunk is paying
  a quarter of the corpus for a property nobody has shown to matter.

| config | windows |
|---|---|
| `grasp` (fixed window) | 189 |
| `grasp-pose`, pre 3.0, frac 0.8 *(today's default)* | 142 |
| `grasp-pose`, pre 3.0, frac 0.9 | 189 |
| **`grasp-pose`, pre 6.0, frac 1.00 *(recommended)*** | **165** |

If you would rather not move `--pre-s`, the second-best option is `--pre-s 3.0
--close-idx-frac 0.90`: it keeps all 189 windows and leaves the gate armed as a
tripwire that fires the instant anyone raises `pre_s` — which is the exact mistake that
produced the pickplace corpus. It does **not** find the 24 hidden long descents.

---

## 4. (b) Is the data ACT-ready?

Ready on size, cadence, action/state wiring and gripper health. Three things are
missing or wrong.

### Healthy

- **Cadence.** `yam_left` 200.0 Hz, `gello_left` 62.5 Hz, `camera_top` 30.0 Hz,
  `camera_left` 31.2 Hz on all seven usable episodes, no gaps.
- **Camera skew** 8.3–8.5 ms, uniform. Free-running cameras; ~a quarter of a frame
  interval. Not a fault, do not chase it.
- **Gripper channel** spans 0.993 of its range on every usable episode against a
  0.0499 dead-channel floor, and is cleanly bimodal (closed ~0.25–0.39 / open
  ~0.59–0.78).
- **Hold durations** median 4.10 s, p10 2.64 s. This is a comfortably-held corpus.
- **Corpus size** meets the 80–120 + 15–20 target roughly 1.5× over in every candidate
  config.

### Missing 1 — one episode is camera-starved and nothing stops it

`episode_155339_ad02fb72`: `camera_left` delivered **1492 frames over 239.4 s — 6.2 Hz
effective**, with 143 gaps larger than two frames and a worst gap of **6.88 s**. About
5668 frames never arrived.

Nothing in the export path notices. `grep degraded tools/export_lerobot.py` returns
nothing, and the per-frame staleness drop (`MAX_CAM_STALENESS_S = 66.7 ms`) silently
shortens windows rather than failing them. Simulated over the windows that would be
written (`act_horizon_report.py`, CAMERA FRAME YIELD section):

| episode | windows | frames written | stubs < 16 frames | `degraded` |
|---|---|---|---|---|
| `episode_155339_ad02fb72` | 17 | 606 / 2468 = **25%** | **7** | true |
| `episode_164621_d1329cef` | 5 | 696 / 696 = 100% | 0 | true |
| all seven others | 167 | 100% | 0 | false |

`155339`'s per-window frame counts are `0, 8, 12, 12, 12, 12, 15, 16, 16, 16, 16, 20,
20, 24, 122, 128, 157`. Two fall below `MIN_WINDOW_FRAMES = 10` and are dropped; **the
rest — 12 to 24 frames, 0.4 to 0.8 seconds — are written into the dataset**, shorter
than the policy's own 16-step commit. Reject the episode outright. It is currently
excluded only because `--held-out-newest 3` happens to catch it, which is luck.

**The `degraded` flag alone is the wrong rule, and this is the correction to the first
draft of this report.** `episode_164621` carries the same flag but its dropouts (5 gaps,
worst 2.58 s) fall entirely *between* grasp windows: 100% frame yield, zero stubs. A
flag-based gate would throw away 5 good grasps. The right test is the one that asks the
question only of the frames that would actually be exported — which is why it is now a
section of `act_horizon_report.py` rather than a paragraph here. §6 step 2 uses it.

### Missing 2 — episode diversity, not window count

The 189 windows come from **seven** episodes, three of which (`145907`, `150947`,
`165925`) supply 57% of them. Window count is comfortably over target; scene diversity
does not follow from it. Everything was recorded in one session, one afternoon, one
lighting condition, one mat layout. That is a real generalisation risk and it is not
something the readiness report can measure. If more recording time is available, prefer
**new sessions over more grasps within a session** — the marginal value of grasp #190 in
this session is far below that of grasp #1 tomorrow morning.

### Missing 3 — the review gate has not been run

PLAN-ACT-READINESS W4 makes an operator pass mandatory before any window is exported.
It has not happened yet. §6 is the procedure. Note bug **B3** first — the review tool
currently shows you different windows from the ones that will be exported.

---

## 5. Bugs found — reported, not patched

Per the handoff, nothing here was changed. All four are in code that has already been
reviewed, so each needs a decision rather than a quiet fix.

**B1 — no camera-yield gate anywhere in the export path.** `session_meta.json` carries
`"degraded": true` and names the cameras; `export_lerobot.plan_episode` and
`analyze_act_readiness.classify_episode` both ignore it. A starved episode classifies as
`OK` and exports. The cadence row does surface the symptom (`cam_left 27.8Hz gaps=143`)
but nothing gates on it. *Suggested:* reject in `plan_episode` with a named reason,
overridable by an explicit flag — the same shape as every other rejection there. Gate on
**measured per-window frame yield**, not on the `degraded` flag: §4 shows the flag has a
false positive (`164621`) in this very session.

**B2 — `MIN_WINDOW_FRAMES = 10` is below `n_action_steps = 16`.** `export_lerobot.py:80`
admits training episodes shorter than a single committed action chunk. On healthy data
it never fires; on B1's starved data it is what lets twelve 12–24-frame stubs through.
The floor should be tied to the runtime commit, not to a round number.

**B3 — `review_grasps.py` reviews windows the exporter will not write.** It imports
`grasp_windows_indexed` — the **fixed** window — and defaults to `--pre-s 2.0
--post-s 1.0`, while the exporter's grasp-pose mode opens at the measured descent start
with `--pre-s 3.0 --post-s 2.0`. PLAN-ACT-READINESS W4 required the two to match
("imported, drift-proof"); they do not. The consequence is specific and lands on the
card the tool says matters most: its own docstring calls the first frame *"the FIRST
frame the policy sees; at deploy IK hands off here, so this frame must look
reachable"* — and that frame is judged at `t_close − 2.0 s` while the export opens it
anywhere from 1.3 s to 5.2 s before the close. The keep-list itself stays sound (it
matches on `t_close`, and a stale entry is a loud error), so this degrades the review's
value rather than corrupting the export.

**B4 — `--close-idx-frac 0.8` cannot express its stated property at the exporter's own
default `--pre-s 3.0`.** Full argument in §3. In short: the ceiling (90) is below the
chunk (100), so no window can fail for the documented reason; the gate instead removes
the slowest 23% of approaches, 19 of 28 of them merely because the pose predicate ran
out of search span. The `PLAN-ACT-READINESS.md` "UNRESOLVED DECISIONS" entry asked for
this constant to be revisited against data before being pinned. §3 is that revisit.

**Not a bug, but do it:** pass `--gripper-open-ref 1.0 --gripper-closed-ref 0.0` on the
export. Without them the dry-run reports *"per-episode percentile scale (no refs
given)"*, which is only valid while every episode spans the full range (all seven
currently do) and is not comparable across sessions. The labeller already uses those
absolute constants (`constants.GRIPPER_OPEN_REF/CLOSED_REF`); the export should share
them.

---

## 6. Operator data-validation procedure

189 cards for this session. Budget ~45 min. The decode is heavy; on this 32-core box it
ran `nice`d alongside a live recording with no measurable effect, but check
`/proc/loadavg` first and prefer an idle rig.

### Step 1 — label (2 min, safe while recording, on finished episodes only)

```bash
cd ~/Desktop/kitting-v2/rl-teleop
for d in recordings/20260902/episode_*/; do
  [ -f "$d/camera_left-rgb-timestamp.npy" ] || { echo "SKIP (mid-write): $d"; continue; }
  ./.venv/bin/python3 qa_label.py "$d" left --task grasp
done
```

The timestamp-sidecar test is the mid-write guard: a recording in progress has no
`*-timestamp.npy` and a 0-byte `gello_left.mcap`. Use `--no-write` if the auto-labeller
in `live_server` may be touching the same episode.

### Step 2 — automatic gates (nothing human yet)

```bash
./.venv/bin/python3 tools/analyze_act_readiness.py --root recordings/20260902 --held-out-newest 0
./.venv/bin/python3 tools/act_horizon_report.py     --root recordings/20260902 --pre-s 6.0
```

**Reject the whole episode** if any of these is true. None of them is a judgement call:

| # | test | where | why |
|---|---|---|---|
| E1 | any window with **< 16 written frames** ("stubs"), or episode frame yield < ~95% | horizon report, CAMERA FRAME YIELD | a starved camera silently shortens windows instead of failing; **not** enforced by the exporter (B1/B2) |
| E2 | gripper spread < 0.05 | readiness GRIPPER block | dead channel — wrong or dead leader handle, not a bad operator |
| E3 | `outcome = aborted`, `CORRUPT`, `UNLABELLED` | readiness EPISODES block | already enforced by the exporter |

Use E1, **not** `session_meta.degraded`. The flag is whole-episode and over-triggers:
`episode_164621` is flagged and is fine. E1 asks the same question of the frames that
would actually be written.

On this session E1 flags `episode_155339_ad02fb72` (25% yield, 7 stubs) and E2 flags
`144346` / `144715`. The other eight pass at 100% yield.

### Step 3 — automatic window gate

Take `--close-idx-frac 1.00` at `--pre-s 6.0` (§3). This is `export_lerobot`'s job, not
yours — it prints one loud line per rejected window with the measured frame index.

### Step 4 — the human pass

```bash
nice -n 15 ionice -c3 ./.venv/bin/python3 tools/review_grasps.py \
    --root recordings/20260902 --pre-s 3.0 --post-s 2.0 \
    --out .review-grasps-20260902 --repo-id ETHRC/yam_grasp_left_v1
```

Already generated for this session — 189 cards, 1134 decoded frames, 18 MB. Open it at
**http://localhost:8810/** (a static server is running out of
`.review-grasps-20260902/`; restart it with
`cd .review-grasps-20260902 && python3 -m http.server 8810`). Forward the port if you
are on SSH. The page's "keep-list" button emits the JSON that `--keep` consumes.

`--pre-s 3.0 --post-s 2.0` matches the exporter's clip lengths — the closest the tool
can currently get. Until B3 is fixed, treat the **first-frame** card as approximate: the
export may open that window somewhat earlier or later. The `t_close` and `t_hi` cards
are exact.

`episode_155339`'s 17 cards will look visibly stale (repeated / frozen wrist frames) —
that is E1 showing up by eye. Reject them; do not spend judgement on them.

Reject a window if **any** of:

| # | on which card | reject when |
|---|---|---|
| W1 | `t_close` | the jaws are not actually on a packet — closed on air, on the mat, or on the edge |
| W2 | `t_close` | **two or more packets** are between the fingers (see the multi-pick note in memory: these packets are too thin for jaw width to separate them) |
| W3 | `t_hi` | the packet is not clearly off the mat, or it visibly slips during the lift |
| W4 | `t_lo` | the wrist view does not contain the target packet — the policy would be asked to reach for something it cannot see |
| W5 | any | the top view is frozen, black, or identical to the neighbouring card's |

W5 is the manual backstop for a camera that went stale mid-episode without tripping E1.

### Step 5 — hold duration

`hold_s` is the quality signal for a grasp-only corpus, and it is **independent** of the
close index (corr +0.14) — passing Step 3 tells you nothing about it. From the readiness
report's HOLD DURATION block, over all 189 windows:

| `hold_s` | count | action |
|---|---|---|
| < 1.5 s | 5 | **reject** — shorter than the labeller's own confidence in a hold |
| 1.5 – 2.5 s | 10 | **confirm the lift by eye** on the `t_hi` card before keeping |
| > 2.5 s | 174 | keep if Step 4 passed |

Corpus reference: min 1.01, p10 2.64, median 4.10, p90 6.24, max 8.35 s.

### Step 6 — hold out whole episodes, then export

Hold out **episodes, not windows** — two windows from one episode share lighting, mat
layout and operator habit, so a window-level split leaks. Pick camera-healthy episodes
totalling 15–20 windows. `--held-out-newest 3` is a convenience, not a rule, and on this
session it picks the degraded `155339` and the mid-write `164621`; choose explicitly.

Then re-run the dry-run with the keep-list and confirm the window count matches what you
reviewed. A keep-list entry that matches no grasp is a hard error and means the
annotations were regenerated after the review — re-review, do not override.

---

## 7. The release / handoff pose — can it be clustered?

The operator's review is done: **179 of 189 windows kept**, in
`data_grasping_left.json`. Measured over the 156 of those that pass the close-index
gate, with FK on the follower at each window's first frame
(`act_horizon_report.py --keep ...`, HANDOFF / RELEASE POSE section):

### One fixed pose: no. Packet-relative: yes, but loosely.

| candidate handoff rule | p90 error vs the demos | within 50 mm |
|---|---|---|
| one fixed absolute pose (the corpus centroid) | **98 mm** | — |
| the median offset, applied to the detected packet | **76 mm** | 56% |

```
window start relative to its own grasp (start - grasp), n=156
  dx   median -0.046   p10 -0.092   p90 -0.013 m
  dy   median +0.033   p10 -0.017   p90 +0.081 m
  dz   median +0.103   p10 +0.070   p90 +0.136 m
```

**Command the handoff at `[-46, +33, +103] mm` relative to the detected packet** —
46 mm back in x, 33 mm across in +y, 103 mm above it. That is the modal starting
condition of the corpus, and it is the best-determined thing here.

### Why the offset is well-determined but the scatter is not fixable by re-recording

Two measurements say the *centre* is solid and the *spread* is operator noise:

- **The approach direction is unimodal.** 153 of 156 windows approach from the same
  quadrant (−x, +y); the xy-angle histogram is a single lobe around ~170°, with 3
  outliers. There is no second cluster to split off.
- **The variance is WITHIN episodes, not between them.** Every episode's own median
  offset agrees closely (dx −0.019…−0.059, dy +0.007…+0.058, dz +0.070…+0.131 m), while
  each episode's internal p90 spread is 47–96 mm. The rig setup was consistent; the
  operator's hand was not, grasp to grasp.

That second point is the actionable one: **you cannot tighten this by recording more
episodes the same way.** Only changing the protocol — starting each demo from a
commanded IK pose instead of free-play teleop — moves it. That is the deferred
"IK-handoff recording" item from PLAN-ACT-READINESS, and this is its quantitative case.

### Do not turn this into an export filter

Start-pose overlap has already been tried and retired on this rig:
`check_handover_pose.py:33-43` records that the handover pose was *equally atypical* for
all three right-arm corpora, **including the one that produced the only working policy**,
so the test passed a known-bad checkpoint. Use the offset above to decide **where to hand
off**. Do not use it to decide **which windows to keep** — that measurement has no
predictive record here, and gating on it would cost corpus size for nothing.

### How many seconds

From that start to the jaw close, over the same 156 windows:

```
  n=156   min 1.30   p10 1.58   median 2.00   p90 2.77   max 3.33 s
```

So: **~2 s of approach before the close, ~2 s of lift after it, ≈4 s per training
window** (median 124 frames at 30 Hz). `chunk_size = 100` (3.33 s) covers the approach
from the handoff for every window that survives the gate — by construction, since the
gate is what enforces it. Nothing here argues for a different chunk.

---

## 8. How to proceed — from review to a deployed checkpoint

Five steps. Steps 1–2 are yours, 3–5 are mechanical. Expect ~1 h of your attention and
~10 h of wall clock.

### Step 1 — review → keep-list  ✅ **done**

179 of 189 kept, in `data_grasping_left.json`. Two notes on it:

- **The file has a comment header and is not valid JSON**, so `--keep` rejects it
  (`load_keep_list` → `load_json` → `SystemExit`). The stripped copy is
  `recordings/20260902/keep-20260902.json`.
- **16 of `episode_155339`'s 17 windows were kept, and the review page could not have
  shown you why they are bad.** It renders three frames per window (start / close /
  lift); the starvation is in the ~120 frames *between* them. The eye gate structurally
  cannot see E1 — which is exactly why E1 is an automatic gate and must run anyway.

The funnel, with the automatic gates applied to your selection:

| | windows |
|---|---|
| reviewed and kept by you | 179 |
| − `episode_155339` (25% frame yield, E1) | −16 → 163 |
| − `hold_s` < 1.5 s | −4 → 159 |
| − close-index gate (descent > one chunk) | −19 |
| **exportable** | **140** |

`recordings/20260902/keep-20260902-final.json` is that 159-entry list; the exporter's
own dry-run confirms **140 windows across 8 episodes**. Comfortably above the 80–120
target.

### Step 2 — pick the held-out episodes  *(you, 2 min)*

Hold out **whole episodes**, never windows: two windows from one episode share lighting,
mat layout and operator habit, so a window-level split leaks and the validation number
lies. `episode_163624` (10 gated) + `episode_164621` (4 gated) ≈ 14 windows is a clean
pick — both 100% frame yield, both late in the session. Export them to a separate
repo-id, or just exclude them and keep them for the on-hardware test.

### Step 3 — export

```bash
cd ~/Desktop/kitting-v2/rl-teleop
./.venv/bin/python3 tools/export_lerobot.py \
    --root recordings/20260902 \
    --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
    --chunk-frames 100 --close-idx-frac 1.00 \
    --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
    --keep recordings/20260902/keep-20260902-final.json \
    --repo-id ETHRC/yam_grasp_left_20260902 --dry-run
```

Verified: **140 windows across 8 episodes**. A keep-list entry matching no grasp is a
hard error and means the annotations were regenerated after the review — re-review, do
not override. Then drop `--dry-run`.

### Step 4 — predecode, then train

`run_train_20260812.sh` is the recipe that produced the only deployed lineage on this
rig; copy it, don't reinvent it. Two things in it are load-bearing and easy to lose:

- **Predecode is not an optimisation.** `torchcodec` cannot load on this box, so
  `tools/predecode_ffmpeg.py` is what makes video reading work at all.
- **The log must live outside `--output_dir`.** LeRobot's `cfg.validate()` refuses to
  train into an existing directory with `resume=False`; creating it to hold a log file
  has killed a run at step 0 before.

```bash
REPO=ETHRC/yam_grasp_left_20260902
DS=$HOME/.cache/huggingface/lerobot/$REPO
PD=$HOME/.cache/lerobot-predecoded/yam_grasp_left_20260902
./.venv/bin/python3 tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$PD"

LEROBOT_PREDECODED_ROOT=$PD ./.venv/bin/python3 tools/train_act_dark_noise.py \
    --dataset.repo_id=$REPO --policy.type=act --policy.device=cuda \
    --policy.chunk_size=100 --policy.push_to_hub=false \
    --steps=100000 --save_freq=10000 --batch_size=8 --seed=1000 \
    --output_dir=outputs/train/act_grasp_left_20260902
```

**Do not pass `--policy.n_action_steps=16`.** `train_act_dark_noise.py`'s own header
(lines 19-23) already settles this: `n_action_steps` is read only by
`ACTPolicy.select_action` to size the action queue, while `chunk_size` is what shapes the
network. It is a deploy-time knob, `act_runner.py:1178-1181` overrides it to 16 at load
and prints that it did, and baking 16 into the checkpoint would just cost a run.
`chunk_size=100` is the number that matters and is the one §3's gate is expressed
against — pass it explicitly so the export config and the training config visibly agree.

Keep `batch_size=8` and `dark_noise` augmentation: that pairing is what
`sweeps/RESEARCH.md` identified, and changing two things at once between the last
working policy and this one makes a failure uninterpretable.

### Step 5 — gate the checkpoint BEFORE putting it on the arm

```bash
cd ~/Desktop/kitting-v2/yam-pick-pipeline
YAM_ARM=left ./check_handover_pose.py <checkpoint>
```

This reads the checkpoint and its training parquet, publishes nothing and touches no
motor. It re-measures the close-index distribution **on the exported corpus** — i.e. it
independently confirms what §3 predicted, after the export rather than before it. If the
two disagree, believe neither and find out why before the arm moves.

Then, and only then, run it through `/run-kitting` on the real rig with the arm-safety
rules in `rl-teleop/CLAUDE.md` in force. Score the held-out episodes' scenes, not the
training ones.

### What to record next

Not more grasps in this session (§4, Missing 2). The two highest-value additions:

1. **A second session on a different day / lighting.** Diversity is the corpus's weakest
   axis by a wide margin.
2. **IK-handoff recording** — demos that start at the pose the classical stack actually
   hands off from, rather than free-play teleop starts. That was deferred out of
   PLAN-ACT-READINESS as its own session, and it is the one change that would make the
   pose predicate, the close-index gate and the first-frame review card all measure the
   same thing.

---

## 9. Open questions for the operator

1. **B1/B2** — should a camera-starved episode be a hard rejection in `plan_episode`, or
   a warning with an override flag? It is the only one of the four bugs that can put
   garbage in a dataset without anyone noticing, and §6 step 2 is currently the only
   thing standing between `episode_155339` and the training set.
2. **B4** — adopt `--pre-s 6.0 --close-idx-frac 1.00` (165 windows, the 24 long
   descents rejected on measurement), or the conservative `--pre-s 3.0
   --close-idx-frac 0.90` (189 windows, gate armed as a tripwire only)?
3. **B3** — is it worth teaching `review_grasps.py` the grasp-pose window before this
   review, or do you accept an approximate first-frame card for this round?
4. Recording is continuing. Everything above is `recordings/20260902` as of 17:12 and
   should be re-run once the session closes; the two tools are idempotent and read-only.
