#!/usr/bin/env bash
# Right-arm retrain on the OPERATOR-REVIEWED corpus — ready to run the moment the
# review's keep-list exists. Context: loops/graspdata/REPORT.md (2026-09-04 audit).
#
# The 20260902 right corpus was exported unattended, keep-list-less, with no held-out
# episodes. This launcher fixes both process gaps and changes NOTHING else vs the
# incumbent recipe (run_grasp_right_20260902.sh: 100k steps, batch 8, seed 1000,
# dark_noise, wrist_native, chunk 100, gripper refs 1.0/0.0):
#   - trains only windows the operator kept in the review page
#     (http://localhost:8811 -> keep-list button -> save as
#      recordings/20260902/keep-right-20260902.json)
#   - holds out one WHOLE episode for eval (default episode_183959_6c1f807f)
#
# KNOWN LIMIT (do not oversell this run): the dot-follow gate FAILED on both
# conditioned checkpoints (targetdot82: 2.4% gain; grasp_right_20260902_targetdot:
# 0.09% — tools/targetdot_runtime/results/dot_follow_metrics_*.json). Target
# selection is NOT fixable by retraining on this corpus; this run only buys the
# reviewed-data quality delta and an honest eval set. The targeting fix is a new
# assigned-target session and/or a supplied target at runtime.
#
# Usage:
#   ./run_grasp_right_reviewed.sh           # gate + export + predecode + train
#   ./run_grasp_right_reviewed.sh --dry     # print plan, run nothing
set -uo pipefail
cd "$(dirname "$0")"

DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

KEEP_ALL="${KEEP_ALL:-recordings/20260902/keep-right-20260902.json}"
HELDOUT_EP="${HELDOUT_EP:-episode_183959_6c1f807f}"
KEEP_TRAIN=recordings/20260902/keep-right-train.json
KEEP_HELD=recordings/20260902/keep-right-heldout.json

RUN=grasp_right_20260902_reviewed
REPO=ETHRC/yam_grasp_right_20260902_reviewed
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/$(basename "$REPO")
CACHE=${BASE}_wristnative
GEOM=wrist_native
PY=./.venv/bin/python3

STEPS="${STEPS:-100000}"
SAVE_FREQ=10000; BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6

OUT=outputs/train/$RUN
LOG=outputs/train/$RUN.log
mkdir -p outputs/train

say(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$KEEP_ALL" ] || { say "FATAL: no reviewed keep-list at $KEEP_ALL"; \
  say "review the cards at http://localhost:8811 first (W1-W5 checklist,"; \
  say "ACT-DATA-READINESS-REPORT-20260902.md §6), save its keep-list JSON there."; exit 1; }

# split: held-out episode goes to its own list, never into training
$PY - "$KEEP_ALL" "$KEEP_TRAIN" "$KEEP_HELD" "$HELDOUT_EP" <<'EOF'
import json, sys
src, ktrain, kheld, ep = sys.argv[1:5]
keep = json.load(open(src))["keep"]
tr = [k for k in keep if k["episode"] != ep]
hd = [k for k in keep if k["episode"] == ep]
json.dump({"keep": tr}, open(ktrain, "w"), indent=1)
json.dump({"keep": hd}, open(kheld, "w"), indent=1)
print(f"reviewed: {len(keep)}  train: {len(tr)}  heldout({ep}): {len(hd)}")
assert hd, f"held-out episode {ep} has no kept windows -- pick another"
EOF
[ $? -eq 0 ] || exit 1

FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 40 ] || { say "FATAL: under 40 GB free."; exit 1; }

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; exit 1; }

if [ "$DRY" = 1 ]; then
    $PY tools/export_lerobot.py --root recordings/20260902 \
        --arms right --cameras wrist_right_top \
        --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
        --chunk-frames 100 --close-idx-frac 1.00 \
        --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
        --keep "$KEEP_TRAIN" --repo-id "$REPO" --dry-run 2>&1 | tail -12
    exit 0
fi

if [ -f "$DS/meta/info.json" ]; then
    say "[1/4] dataset already exported -- skipping"
else
    say "[1/4] exporting $REPO ..."
    $PY tools/export_lerobot.py --root recordings/20260902 \
        --arms right --cameras wrist_right_top \
        --window-mode grasp-pose --pre-s 6.0 --post-s 2.0 \
        --chunk-frames 100 --close-idx-frac 1.00 \
        --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
        --keep "$KEEP_TRAIN" --repo-id "$REPO" 2>&1 | tail -25
    [ -f "$DS/meta/info.json" ] || { say "FATAL: export failed"; exit 1; }
fi
N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])")
say "      exported: $N_EP windows (reviewed train set)"

if [ -d "$BASE" ]; then say "[2/4] base cache exists -- skipping"; else
    say "[2/4] predecoding ..."
    $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -4
    [ -d "$BASE" ] || { say "FATAL: predecode failed"; exit 1; }
fi
if [ -d "$CACHE" ]; then say "[3/4] baked cache exists -- skipping"; else
    say "[3/4] baking $GEOM ..."
    $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -4
    [ -d "$CACHE" ] || { say "FATAL: resize failed"; exit 1; }
fi

# LEROBOT_PREDECODED_ROOT inline, not exported (tmux server-env trap, 2026-08-14).
CMD="LEROBOT_PREDECODED_ROOT='$CACHE' ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
tools/train_act_dark_noise.py --dataset.repo_id='$REPO' --policy.type=act \
--policy.device=cuda --policy.chunk_size=100 --policy.push_to_hub=false --steps=$STEPS \
--save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SEED --num_workers=$WORKERS \
--output_dir='$OUT' > '$LOG' 2>&1; echo \$? > '$OUT.exit';"

say "[4/4] launching $RUN ($STEPS steps)"
rm -rf "$OUT" "$OUT.exit"
tmux kill-session -t "tr_$RUN" 2>/dev/null
tmux new-session -d -s "tr_$RUN" -c "$PWD" "$CMD"
sleep 25
if tmux has-session -t "tr_$RUN" 2>/dev/null; then
    say "running in tmux tr_$RUN -- follow: tail -f $LOG"
else
    say "WARNING: run died at startup:"; tail -25 "$LOG"; exit 1
fi
