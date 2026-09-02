# Right-arm ACT training — queued 2026-09-02 (you were away)

## What will happen, automatically
A detached chain (`chain_right_after_left.sh`, session-leader, survives logout) is
polling the LEFT training's liveness. It **never signals the left process**. When the
left training exits, it waits 90 s for GPU memory, then runs `run_grasp_right_20260902.sh`:

1. geometry selftest (`act_bus_geometry.py` → ALL OK) — hard gate. PASSED at queue time.
2. export right dataset: `--arms right --cameras wrist_right_top`, grasp-pose window,
   `--pre-s 6.0 --post-s 2.0 --chunk-frames 100 --close-idx-frac 1.00`. Auto-filtered
   (no keep-list — you were away). Dry-run count: **104 windows** from 4 episodes / 133 grasps.
3. predecode + bake `wrist_native` geometry.
4. conditioning gate (see below).
5. train ACT: **100k steps**, `wrist_native`, chunk_size 100, batch 8, `dark_noise` aug,
   in tmux `tr_grasp_right_20260902`, log `outputs/train/grasp_right_20260902.log`.

## Conditioned vs baseline — READ THIS
You chose "condition properly." I built the dot-dropout machinery
(`tools/targetdot/burn_dots_dropout.py`, dropout frac 0.2, renders the SAM2-centroid dot
on every labeled wrist frame except a random fraction; sanity-gates out-of-bounds / too-few
labels). BUT the overnight run trains the **UNCONDITIONED baseline**, on purpose:

- The conditioning labeler runs SAM2 on the wrist frames to find the grasped packet.
  Its masks must be **eyeballed** — a wrong mask puts the dot on the wrong packet and
  trains the policy toward the wrong target, the exact opposite of the goal, and
  unverifiable while you're gone. Running it blind risks a poisoned 100k-step run.
- So `make_conditioned_cache.sh` **refuses** unless reviewed `labels-20260902.json` +
  `mapping-20260902.json` exist. They don't yet → the run falls back to baseline and
  says so in `outputs/train/grasp_right_20260902.status` (`mode=UNCONDITIONED-baseline`).

The baseline is a real, usable right-arm grasp checkpoint — not wasted GPU.

## To get the CONDITIONED run (≈30 min supervised, when you're back)
1. Build + REVIEW the SAM2 labels for this corpus (the fragile, must-be-watched step):
   run the targetdot label pipeline for 20260902 (needs a `20260902` preset added to
   `map_dataset.py` / `autolabel_missing.py` / `build_labels.py`; SAM2 in the study
   `.venv-eval`), eyeball the masks, write `tools/targetdot/labels-20260902.json` + `mapping-20260902.json`.
2. `rm -rf ~/.cache/lerobot-predecoded/yam_grasp_right_20260902_wristnative_targetdot`
   then re-run `run_grasp_right_20260902.sh` → it detects the labels, burns the dot-dropout
   cache, and trains CONDITIONED (`mode=CONDITIONED-targetdot`).
3. Gate it: `sanity_dot_follow.py` — require the grasp to move >50% of the dot's
   displacement (targetdot82 got ~2%). Only trust a checkpoint that passes.

## Data facts
- 4 right episodes, 133 grasps (37/28/34/34), all clean (none degraded).
- 104 windows pass the close-idx gate; 24 dropped because their descent is longer than
  one 100-frame chunk (right arm commits the whole chunk at deploy — a real constraint).
- Full gripper swing, median hold 3.5 s — healthy.

## Monitor
- `tail -f outputs/train/chain_right.log` — the chain (waiting → launched).
- `cat outputs/train/grasp_right_20260902.status` — mode + windows + start time.
- `tmux attach -t tr_grasp_right_20260902` — the training once it starts.
- Left training (untouched): tmux `tr_grasp_left_20260902_wristnative`, 150k steps.

## Untouched, as instructed
The left training, the recording session, and all bus/camera processes. Nothing was killed.
