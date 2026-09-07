#!/usr/bin/env bash
# run_night_20260907.sh — three overnight retrains of the day's winner, launched
# autonomously 2026-09-07 while Tommaso slept ("based on our learning could we
# train 3 new models tonight").
#
# All three use the INCUMBENT's corpus and recipe (night05_base_both_c100: base
# window, wrist+top, chunk 100, dataset ETHRC/yam_grasp_right_night05_base_both,
# 260 windows / 31,525 frames). One factor changes per model, so each answers one
# question from data/research/eval-20260907-model-comparison.md §4:
#
#   night07_base_both_c100_200k       STEPS 200k (≈50 epochs)   T3: is the winner
#                                     winning because it is the most-trained?
#   night07_base_both_c100_seed2      seed 2000, 100k           variance floor: how
#                                     much of a 6/16 is luck? (identical-config runs
#                                     spread 19–49 % in the ACT research corpus)
#   night07_base_both_c100_geometric  ACT_AUG=geometric, 100k   the recipe written
#                                     for "the wrist camera was replaced/remounted"
#                                     (tools/act_aug.py) — the Innomaker swap is
#                                     exactly that
#
# Waits for the eval queue to go idle first (one GPU, and the night eval was
# running when this was written). Skips the export (datasets exist); rebuilds the
# predecode cache if it was cleaned (it was, 2026-09-07). Idempotent like the
# night05 script: a finished model (.exit == 0) is skipped on re-run.
#
# Run:  tmux new-session -d -s train 'cd ~/Desktop/kitting-v2/rl-teleop && ./run_night_20260907.sh 2>&1 | tee -a outputs/train/night07.log'
set -uo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python3
GEOM=wrist_native
BATCH=8; WORKERS=6
DSKEY=base_both
REPO=ETHRC/yam_grasp_right_night05_$DSKEY
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/night05_$DSKEY
CACHE=${BASE}_wristnative
CANDIDATES=../yam-pick-pipeline/tools/eval_candidates.json
MIN_FREE_GB=20
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }
free_gb(){ df --output=avail -BG . | tail -1 | tr -dc '0-9'; }

# name | steps | seed | aug
VARIANTS=(
  "night07_base_both_c100_200k|200000|1000|dark_noise"
  "night07_base_both_c100_seed2|100000|2000|dark_noise"
  "night07_base_both_c100_geometric|100000|1000|geometric"
)

say "=== NIGHT07: ${#VARIANTS[@]} retrains of the incumbent recipe ==="
[ -f "$DS/meta/info.json" ] || { say "FATAL: dataset $DS missing — export first (run_night_matrix_20260905.sh does it)"; exit 1; }
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; exit 1; }
$PY tools/act_aug.py >/dev/null 2>&1 || { say "FATAL: augmentation recipes selftest failed"; exit 1; }

# ── wait for the rig's eval queue to go idle (shared GPU) ────────────────────
eval_busy(){
  tmux has-session -t =eval 2>/dev/null && return 0
  local st; st=$(curl -s --max-time 3 http://127.0.0.1:8806/status 2>/dev/null) || return 1
  $PY - "$st" <<'EOF'
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    sys.exit(1)
busy = bool(d.get("running")) or (len(d.get("queue") or []) > 0 and not d.get("paused"))
sys.exit(0 if busy else 1)
EOF
}
waited=0
while eval_busy; do
  [ $((waited % 300)) -eq 0 ] && say "eval queue busy — waiting (${waited}s)"
  sleep 30; waited=$((waited+30))
  [ "$waited" -ge 7200 ] && { say "waited 2 h for the eval to finish — giving up, nothing trained"; exit 2; }
done
say "eval idle after ${waited}s — starting"

N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])")
N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])")
say "dataset $REPO: windows=$N_EP frames=$N_FR"

# ── predecode + bake once (identical to run_night_matrix_20260905.sh) ────────
if [ ! -d "$CACHE" ]; then
  say "[cache] predecode"; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" > outputs/train/night07.predecode.log 2>&1
  [ -d "$BASE" ] || { say "FAIL predecode (outputs/train/night07.predecode.log)"; exit 1; }
  say "[cache] bake $GEOM"; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" > outputs/train/night07.bake.log 2>&1
  [ -d "$CACHE" ] || { say "FAIL bake (outputs/train/night07.bake.log)"; exit 1; }
  HL_FAIL=0
  while IFS= read -r l; do
    t=$(readlink -f "$l") || { HL_FAIL=1; break; }
    rm "$l"
    if [ -d "$t" ]; then cp -al "$t" "$l" || { HL_FAIL=1; break; }
    else ln "$t" "$l" || { HL_FAIL=1; break; }
    fi
  done < <(find "$CACHE" -type l)
  if [ "$HL_FAIL" != "0" ] || [ -n "$(find "$CACHE" -xtype l 2>/dev/null | head -1)" ]; then
    say "FAIL hardlink conversion in $CACHE — keeping $BASE"; exit 1
  fi
  rm -rf "$BASE"
else say "[cache] baked cache exists — skip"; fi

# ── the queue ────────────────────────────────────────────────────────────────
for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r RUN STEPS SEED AUG <<< "$spec"
  OUT=outputs/train/$RUN; LOG=outputs/train/$RUN.log; STATUS=outputs/train/$RUN.status
  say "━━ $RUN  (steps $STEPS seed $SEED aug $AUG; epochs ≈ $(( STEPS * BATCH / N_FR )))"
  if [ -f "$OUT.exit" ] && [ "$(cat "$OUT.exit")" = "0" ]; then say "   already trained — skip"; continue; fi
  FREE=$(free_gb)
  [ "$FREE" -ge "$MIN_FREE_GB" ] || { say "   SKIP: only ${FREE} GB free"; echo "SKIP disk=${FREE}G" > "$STATUS"; continue; }
  echo "mode=UNCONDITIONED window=base cams=both train_chunk=100 export_chunk=100 windows=$N_EP frames=$N_FR steps=$STEPS seed=$SEED aug=$AUG recipe=night05_base_both_c100 dataset=$REPO started=$(date)" > "$STATUS"
  rm -rf "$OUT" "$OUT.exit"
  LEROBOT_PREDECODED_ROOT="$CACHE" ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
    tools/train_act_dark_noise.py --dataset.repo_id="$REPO" --policy.type=act \
    --policy.device=cuda --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.push_to_hub=false \
    --steps="$STEPS" --save_freq="$STEPS" --batch_size=$BATCH --seed=$SEED \
    --num_workers=$WORKERS --output_dir="$OUT" > "$LOG" 2>&1
  rc=$?; echo "$rc" > "$OUT.exit"
  if [ "$rc" = "0" ] && [ -f "$OUT/checkpoints/$STEPS/pretrained_model/config.json" ]; then
    say "   ✓ $RUN done"; echo "DONE $(date)" >> "$STATUS"
    # the eval dashboard lists it as pending until the checkpoint exists
    $PY - "$CANDIDATES" "$RUN" <<'EOF'
import json, sys
p, run = sys.argv[1], sys.argv[2]
d = json.load(open(p))
for m in d["models"]:
    if m["name"] == run: m.pop("pending", None)
json.dump(d, open(p, "w"), indent=2); open(p, "a").write("\n")
EOF
  else say "   ✗ $RUN FAILED rc=$rc (tail: $LOG)"; echo "FAIL train rc=$rc" >> "$STATUS"; fi
done

say "=== NIGHT07 COMPLETE ==="
for spec in "${VARIANTS[@]}"; do IFS='|' read -r RUN _ _ _ <<< "$spec"; s=outputs/train/$RUN.status; [ -f "$s" ] && echo "  $RUN: $(tail -1 "$s")" || echo "  $RUN: not started"; done
