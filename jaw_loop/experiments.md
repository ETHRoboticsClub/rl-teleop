# Jaw detector — experiment log

Verifier: `verify.py` (FROZEN). 5-fold CV on a frozen split, 50 captures,
24 full / 26 empty. Target: **0 false positives and >= 95% accuracy.**

A false positive is "said FULL, was EMPTY" — the arm believing it grasped air.
That is the bug this whole line of work exists to kill, so it is weighted hardest.

**Ceiling, stated once:** 0 errors on 50 samples is a 95% CI of roughly [0, 7%]
on the true error rate. This harness cannot prove 100%. It can prove "no false
positives observed and no worse than ~7%".

| cycle | hypothesis | change | acc | FP | FN | verdict |
|---|---|---|---|---|---|---|
| 0 | baseline: the actuator, not the image | `gripper_pos` single threshold | 88.0% | — | — | reference bar. One "full" sample reads 0.00747, BELOW the empty hard stop, which is physically impossible for a held object — capture likely caught the jaws mid-close. |
| 1 | a held packet is close to the fixed-focus lens so it defocuses; high-frequency energy in the jaw ROI should drop | Laplacian variance + 7 other features, best single threshold, ROI y.45-1.0 x.10-.90 | 84.0% | 5 | 3 | FAIL. Direction was right (`lap_var` full 322 vs empty 579) but weak. Missed `full_170400_817`, the clearest held packet in the set. |
| 2 | which feature actually carries signal, and does it come from the jaws or the scene? | measurement only, no detector change | — | — | — | **Saturation is the strongest feature (AUC 0.886) and it is a SCENE ARTEFACT.** See below. |

## Cycle 2 — the finding that matters

Feature ranking over all 50 (AUC, and best single-threshold accuracy):

| feature | AUC | acc | full mean | empty mean |
|---|---|---|---|---|
| sat | 0.886 | 86.0% | 53.0 | 24.8 |
| grad_p90 | 0.183 | 82.0% | 101.6 | 122.2 |
| lap_var | 0.218 | 78.0% | 322.4 | 578.6 |
| bright_sd | 0.724 | 72.0% | 46.3 | 38.0 |

Then the ROI sweep, with a deliberate control — a region of the frame containing
**no gripper at all**:

| ROI | sat AUC | sat acc |
|---|---|---|
| bottomband y.75-1.0 (the jaws) | 0.897 | 88.0% |
| above y.35-.70 | 0.893 | 84.0% |
| **SCENE-ONLY y.00-.40 (no jaws)** | **0.873** | **86.0%** |
| tight jaw gap y.60-.90 x.35-.65 | 0.776 | 78.0% |

**The control scores as well as the jaws.** The classes are separable from the
background alone, so saturation is measuring what the camera was pointed at, not
what was between the fingers. Cut the scene away and accuracy falls to 78%.

This is the same class of defect kitting documented in `REPORT-EMPTY-GRIPPER.md`
§1, arrived at by a different route. There the shortcut was ARM POSE (joints-only
AUC 0.9841 vs the vision witness's 0.9829). Here arm pose is comparatively weak —
best single joint 74% — and the shortcut is the SCENE instead. Same lesson: a
headline number on this corpus does not demonstrate that anything read the jaws.

**Consequence for the target.** Chasing >=95% on this set as it stands optimises
for the shortcut. A detector could reach the number and still fail the first time
the bin is moved, refilled, or lit differently. The loop must not be run to
convergence on this data.

## Open

- 6 captures are physically inconsistent with their label on `gripper_pos`
  (5 "full" at exactly the empty hard stop 0.012030, 1 "full" below it at
  0.007474). Not dropped — the operator's labels are the ground truth by
  definition here, and silently discarding inconvenient samples is the classic
  way to manufacture a passing score. Listed for human adjudication in the
  capture gallery.

| 3 | the channels are unequally reliable, so branch instead of blending | cascade: width decides outside the empty band, image decides inside it | 90.0% | 1 | 4 | Better than either alone (image 88%, width 88%). The one FP sat exactly AT the hard stop. |
| 4 | jaws reaching the same closure as empty air have nothing in them, so the boundary is `<=` not `<` | one character | **92.0%** | **0** | 4 | **ZERO FALSE POSITIVES.** Target's safety half met. |

## Cycle 4 — where it stands, and why 95% is not reachable here

    width <= 0.011959  -> EMPTY   (at the hard stop; an object cannot close it further)
    width >  0.012030  -> FULL    (something is holding the jaws open)
    otherwise          -> the image decides

Remaining 4 errors are ALL false negatives, which cost a retry, not a phantom place:

    full_170102_006  gripper 0.007474  <- below the empty stop; physically impossible
                                          for a held object. Capture almost certainly
                                          caught the jaws mid-close.
    3x at exactly 0.012030             <- packets thin enough not to move the jaws

**The band is the whole remaining problem, and it is starved.** 31 of 50 samples
fall in it, but only 5 of them are FULL. Predicting "empty" for the entire band
already scores 83.9%; the best image feature reaches 93.5% in-band, and the
SCENE-ONLY control still reaches 90.3% in-band, so even that margin is partly the
background shortcut again.

Pushing in-band recall to catch 2 more fulls means loosening a decision boundary
that has 26 empties sitting against it. That is the exact move that produces false
positives. **On this corpus, 0 FP and >=95% are mutually exclusive.**

## What unblocks it — a specific data ask, not "more data"

Not 20 more of each. The set needs FULL captures INSIDE THE BAND: grasps of
packets thin enough that the jaws nearly close (0.011959-0.012030). There are 5
today. Twenty would make the band a real classification problem rather than a
5-positive curiosity.

Second, decorrelate the scene. Cycle 2's control showed a jaw-free region scores
86%. The cheap fix is paired capture: at ONE arm pose over ONE patch of bin,
press D holding a packet, release it, press E without moving. The background is
then identical across the pair and cannot carry the label.

---

# Re-freeze: 80 captures (was 50). Folds stratified on (label, band).

| cycle | hypothesis | change | acc | FP | FN | verdict |
|---|---|---|---|---|---|---|
| 5a | — | cycle-4 detector, new data | 82.5% | 5 | 9 | **REGRESSION.** Not the data: `g <= lo` compared float64 exactly, and each physical encoder reading appears as a PAIR of values 6.6e-16 apart. The larger twin of the hard-stop reading fell through to the image and became a false positive. |
| 5b | merge representation twins, keep the physics | `WIDTH_TOL = 1e-9` (clusters are 7.1e-5 apart, twins 6.6e-16) | 87.5% | 1 | 9 | Fixed the regression. |
| 6 | pick the in-band threshold for precision, not accuracy: lowest cut with no train FP | threshold from train negatives | 86.2% | 1 | 10 | REGRESSED, reverted. 10 in-band empties total; a threshold fitted on 8 does not hold on the other 2. |

## What the new data actually revealed

**The 92% / 0 FP on 50 captures was flattered by imbalance.** The band then held
26 empty against 5 full, so "predict empty in the band" scored 0 false positives
almost for free. The new captures balanced the band to **11 full / 10 empty**, and
that free lunch is gone: the same detector now shows 1 FP.

That is the new data doing its job. It did not make the detector worse, it made
the measurement honest.

Where the problem now lives, precisely:

    outside the band   59 of 80 captures, separated PERFECTLY by width alone
    inside the band    21 captures, image-only, honest CV 76.2% (baseline 47.6%)
                       -> catches 9/11 fulls but with 3 false alarms

So the image IS learning in the band (76% vs 48% baseline) and it is not learning
enough. Overall ceiling with in-band at 76% is about 94%; zero false positives is
not reliably reachable, because with 10 in-band empties any precision threshold is
fitted on 8 points.

## The technique this points at, which has never been tested

A thin packet between nearly-shut jaws looks like nothing in a single frame. That
is why the band is hard, and no amount of static-image feature work changes it.

**The micro-squeeze is the only proposed technique that targets the band directly,
and it cannot be evaluated from this dataset at all** -- it needs a live
interaction, not a captured frame:

    command a small extra close on already-shut jaws, then measure
      empty  -> already against the hard stop, width does not move
      held   -> the packet compresses, width moves and effort ramps

The band is exactly the set where static width is uninformative, and compliance is
a different measurement, not a better threshold on the same one. Static captures
can never answer it; a 2-minute live test can.

| 7 | the ROI was inherited from another camera; the operator draws it instead | operator-selected strip + 600-rectangle search | 93.8% | 0 | 5 | Big gain. EVERY rectangle he drew scored 0 FP; the inherited one scored 1. The useful region is a NARROW VERTICAL STRIP down the finger gap (41x118 px), not a wide box. 376/600 rectangles gave 0 FP -> broad plateau, not a lucky peak. |
| 8 | the jaws are rigid, so an EMPTY frame is nearly identical every time: score deviation from an empty template instead of averaging the ROI | per-pixel median+std template from TRAIN empties; feature = fraction of pixels >3 sigma | **98.8%** | **0** | 1 | The mechanism change that mattered. In-band separability went to 100% where every averaged feature had plateaued near 76%. |
| 9 | a width BELOW the empty hard stop is physically impossible, so the sensor is wrong and should not be trusted | impossible widths defer to the image | **100.0%** | **0** | **0** | PASS on the frozen split. See the robustness caveat below. |

## Why cycle 8 worked when six feature-engineering cycles did not

Every earlier feature -- brightness, saturation, gradient, focus, white fraction --
summarises the ROI as an AVERAGE. An average over a region that also contains bin,
bag and mat is dominated by whichever of those is in frame, which is exactly what
cycle 2's control caught: a region with NO GRIPPER IN IT scored 86%.

Deviation-from-template asks about THE SAME PIXELS every time. The camera is bolted
to the jaws, so an empty gripper is the same picture on every capture; a packet
occupies pixels that have never been anything but background. A changing background
cannot answer that question, so the shortcut is unavailable by construction rather
than by hope.

## ROBUSTNESS: the 100% does not survive re-splitting

    20 random 5-fold splits   mean 98.9%   min 97.5%   only 4/20 reach 100%
                              16/20 show at least one false positive
    81 ROI perturbations      45 still 100%, 27 produce a false positive

The frozen split says 100%. Twenty other splits say 98.9% on average, and false
positives reappear on most of them. After searching 600 rectangles against ONE
split, part of that 100% is selection, not detection.

**Report the honest number: ~98.9% mean, zero-FP not yet reliable.** The real gain
is 87.5% -> 98.9% and a mechanism that is principled rather than incidental. What
would make zero-FP robust is more in-band captures -- the band is still only 21 of
80 samples, so a single in-band empty landing in a test fold moves the FP count.
