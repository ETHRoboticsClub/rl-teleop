# Target-dot conditioned ACT — data pipeline + training run (2026-08-31)

Retrains the deployed right-arm grasp ACT with a **target point rendered into the
wrist image** (SutureBot's winning goal format; see
`target-selector-offline/research/target-conditioning-offline.md` in the
yam-pick-pipeline repo, §3 and §6.3). Same dataset, same recipe, only the pixels
change: every wrist frame that has an auto-label gets a magenta disk on the
packet that ends up grasped.

## The run

```
LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_right_20260812_targetdot \
  ./.venv/bin/python3 tools/train_act_dark_noise.py \
  --dataset.repo_id=ETHRC/yam_grasp_right_20260812 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --seed=1000 \
  --output_dir=outputs/train/act_grasp_right_targetdot_20260831
```

Identical to `run_train_20260812.sh` (the deployed lineage) except the
predecoded root: the parquet/state/action side of the dataset is the ORIGINAL
`ETHRC/yam_grasp_right_20260812` untouched; only the JPEG cache the
predecoded-patch feeds the policy is the dotted copy.

## Pipeline (in run order)

| step | file | output |
|---|---|---|
| 0 | `keep_20260812.json` | the operator keep-list the 2026-08-12 export was run with, recovered from the export session's transcript (the `/tmp` original is gone). 45 grasps; 2 windows were dropped at write time → 43 dataset episodes. |
| 1 | `map_dataset.py` | `mapping.json` — dataset (episode_index, frame_index) → camera_right frame index in recordings/20260811, by replaying the exporter's own planning + write loop (imports `tools/export_lerobot.py`, no forked logic). **Verified**: per-episode frame counts match `meta/episodes` exactly (43 ep / 5910 frames), and 12 sampled frames pixel-match their predecoded JPEG better than either temporal neighbour. |
| 2 | `autolabel_missing.py` | SAM2 auto-labels (run with the target-selector-offline study's `.venv-eval`) for the 9 keep-list grasps the study's long-hold event detector missed; same seed + backward-track pipeline, npz format identical. The other 36 events reuse the study's existing `work/` npz. |
| 3 | `build_labels.py` | `labels.json` — per dataset frame, the target point (tracked-mask centroid) in native 640x480 wrist pixels. Coverage 5600/5910 (94.8%). |
| 4 | `../target_dot.py` | THE canonical renderer (filled disk, r=10, magenta). Inference must import this same function — matching render style train/inference is a correctness requirement. |
| 5 | `burn_dots.py` | copies `~/.cache/lerobot-predecoded/yam_grasp_right_20260812` → `..._targetdot` and burns the dot into the 5600 labeled frames; the 310 unlabeled frames stay byte-identical. |
| 6 | `make_verify_grid.py` | `verify/grid_{a,b}.jpg` — 12 episodes x 5 phases, inspected before training. |

## Label policy (where the dot is, and when there is none)

- **approach (pre-close)**: centroid of the SAM2 backward-tracked mask at the
  nearest labeled camera frame within ±3 frames (tracker ran at stride 2).
- **close + lift**: last tracked centroid held constant — the packet is between
  the jaws and approximately static in the wrist frame; the deployed tracker
  keeps its last lock the same way.
- **no label** (310 frames, 5.2%): early-approach frames where the target is not
  yet in the wrist camera's field of view, or the track was lost. No dot is
  drawn — the research doc attributes the uncovered remainder to
  out-of-frame targets, and a fabricated dot there would be a cue the deployed
  tracker cannot produce either.

Dot style rationale (also in `target_dot.py`): point label beat mask and
distance-map in SutureBot's ablation; magenta occurs nowhere on the rig and is
RGB/BGR-symmetric, so a channel-order mixup cannot silently move the cue;
r=10 px at 640x480 survives the dark_noise augmentation.

## What is deliberately NOT in git

`mapping.json`, `labels.json`, `work/*.npz` (regenerable label dumps — steps
1–3), and the dotted predecoded cache. Regenerate with steps 1–5 above; steps
1, 3, 5, 6 run in this repo's venv, step 2 in the study worktree's `.venv-eval`.

## At inference

Selector/operator supplies the point → SAM2.1-small video tracker holds it
(~30 ms/frame) → `tools.target_dot.draw_target_dot(frame, x, y)` on the native
640x480 wrist frame before the policy forward pass. Undotted = "no target
chosen"; the 5.2% undotted training frames teach the policy that state too.
