# Target-dot conditioned ACT — data pipeline + training runs

Retrains the right-arm grasp ACT with a **target point rendered into the wrist
image** (SutureBot's winning goal format; see
`target-selector-offline/research/target-conditioning-offline.md` in the
yam-pick-pipeline repo, §3 and §6.3). Same datasets, same recipes as the
deployed lineage, only the pixels change: every wrist frame that has an
auto-label gets a magenta disk on the packet that ends up grasped.

## The runs

### Run 1 — 43 episodes (2026-08-31, stopped at 32k by the operator; 10k/20k/30k checkpoints kept)

```
LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_right_20260812_targetdot \
  ./.venv/bin/python3 tools/train_act_dark_noise.py \
  --dataset.repo_id=ETHRC/yam_grasp_right_20260812 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --seed=1000 \
  --output_dir=outputs/train/act_grasp_right_targetdot_20260831
```

Identical to `run_train_20260812.sh` (the deployed lineage) except the
predecoded root. Label coverage 5600/5910 frames (94.8%).

### Run 2 — 82 episodes / 11,662 frames (2026-08-31)

```
LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_right_20260814_wristnative_targetdot \
ACT_AUG=dark_noise ACT_GEOMETRY=wrist_native \
  ./.venv/bin/python3 tools/train_act_dark_noise.py \
  --dataset.repo_id=ETHRC/yam_grasp_right_20260814 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --seed=1000 \
  --output_dir=outputs/train/act_grasp_right_targetdot82_20260831
```

Identical to the `graspmatched_wristnative` lane (queue_priority_20260815.sh):
`ACT_GEOMETRY=wrist_native` trains the wrist at native 480x640 and the top at
the bus's 270x480, exactly like that checkpoint. Source cache is the
`*_wristnative` predecode (wrist stored native, which is the geometry the SAM2
labels are in). Label coverage 11,016/11,662 frames (94.5%). The `_bus` cache
variant is NOT supported: its frames are resize_with_pad-ed, so the dot would
need the same coordinate transform — follow-up work, noted in `build_labels.py`.

In both runs the parquet/state/action side of the dataset is the ORIGINAL
HF-cache dataset untouched; only the JPEG cache the predecoded-patch feeds the
policy is the dotted copy.

## Pipeline (in run order; `--dataset 20260812|20260814` on every script)

| step | file | output |
|---|---|---|
| 0 | `keep_20260812.json` | (20260812 only) the operator keep-list that export ran with, recovered from the export session's transcript (the `/tmp` original is gone). 45 grasps; 2 windows dropped at write time → 43 dataset episodes. The 20260814 export (`run_night_20260814.sh`: `--arms right --cameras wrist_right_top --gripper-open-ref 1.0 --gripper-closed-ref 0.0`) used no keep-list → 82 episodes. |
| 1 | `map_dataset.py` | `mapping-<ds>.json` — dataset (episode_index, frame_index) → wrist-camera frame index in the recordings, by replaying the exporter's own planning + write loop (imports `tools/export_lerobot.py`, no forked logic; staleness across ALL exported cameras). **Verified both datasets**: per-episode frame counts match `meta/episodes` exactly (43 ep / 5910 fr; 82 ep / 11,662 fr), and 12 sampled frames per dataset pixel-match their predecoded JPEG better than either temporal neighbour. |
| 2 | `autolabel_missing.py` | SAM2 auto-labels (run with the target-selector-offline study's `.venv-eval`) for the grasps the study's long-hold event detector missed — 9 for 20260812, 7 for 20260814; same seed + backward-track pipeline, npz format identical. The rest reuse the study's existing `work/` npz. |
| 3 | `build_labels.py` | `labels-<ds>.json` — per dataset frame, the target point (tracked-mask centroid) in native 640x480 wrist pixels. |
| 4 | `../target_dot.py` | THE canonical renderer (filled disk, r=10, magenta). Inference must import this same function — matching render style train/inference is a correctness requirement. |
| 5 | `burn_dots.py` | copies the source predecode cache → `*_targetdot` sibling and burns the dot into every labeled wrist frame; unlabeled wrist frames and all top frames stay byte-identical. |
| 6 | `make_verify_grid.py` | `verify/grid_*.jpg` — 12 episodes x 5 phases per dataset, inspected before training. |

## Label policy (where the dot is, and when there is none)

- **approach (pre-close)**: centroid of the SAM2 backward-tracked mask at the
  nearest labeled camera frame within ±3 frames (tracker ran at stride 2).
- **close + lift**: last tracked centroid held constant — the packet is between
  the jaws and approximately static in the wrist frame; the deployed tracker
  keeps its last lock the same way.
- **no label** (5.2% / 5.5% of frames): early-approach frames where the target
  is not yet in the wrist camera's field of view, or the track was lost. No dot
  is drawn — the research doc attributes the uncovered remainder to
  out-of-frame targets, and a fabricated dot there would be a cue the deployed
  tracker cannot produce either.

Dot style rationale (also in `target_dot.py`): point label beat mask and
distance-map in SutureBot's ablation; magenta occurs nowhere on the rig and is
RGB/BGR-symmetric, so a channel-order mixup cannot silently move the cue;
r=10 px at 640x480 survives the dark_noise augmentation.

## What is deliberately NOT in git

`mapping-*.json`, `labels-*.json`, `work/*.npz` (regenerable label dumps —
steps 1–3), and the dotted predecoded caches. Regenerate with steps 1–5; steps
1, 3, 5, 6 run in this repo's venv, step 2 in the study worktree's `.venv-eval`.

## At inference

Selector/operator supplies the point → SAM2.1-small video tracker holds it
(~30 ms/frame) → `tools.target_dot.draw_target_dot(frame, x, y)` on the native
640x480 wrist frame before the policy forward pass (before any resize, and for
a wrist_native checkpoint the wrist is not resized at all — but delete
`publish_resize` from the camera_right node per `act_bus_geometry.py`'s deploy
note). Undotted = "no target chosen"; the ~5% undotted training frames teach
the policy that state too.
