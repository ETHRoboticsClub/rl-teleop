#!/usr/bin/env bash
# run_isolate_pick_right_20260914.sh — right-arm ACT on the ISOLATE-THEN-PICK task,
# trained on whole home→home episodes (not grasp windows).
#
# Task (Tommaso, 2026-09-14): start at a raised home above the red source box,
# close the jaws to push/isolate packets (NOT grasps), grasp exactly one, drop it
# in the next box, return home. Red box moved closer to the top camera tonight.
#
# Pipeline: tools/segment_home_episodes.py cuts every take at each return to home
# (frame-yield gate, category filter, 1-in-6 holdout) → export_lerobot
# --window-mode full --windows → predecode → bake wrist_native → train.
# A grasp-window export would cut a window around every isolation nudge, which
# is why the export is by time window.
#
# SIX runs in two waves of three concurrent trainings (Tommaso, 2026-09-14 23:40,
# "3 according to the paper and 3 according to what we found was best"):
#   wave 1 PAPER  (Zhao et al. 2023 as written; no augmentation, LeRobot ACT defaults
#                  = kl 10, lr 1e-5, ResNet18, 4 enc layers; ~50k grad steps ≈ 1.6× the
#                  paper's ~31k budget):
#     P1 c60        chunk 60 frames = the paper's 2 s horizon at this rig's 30 fps
#     P2 c100       the paper's literal chunk number (3.3 s here)
#     P3 c60_dec7   chunk 60 with the 7 decoder layers the paper states (LeRobot
#                   defaults to 1, reproducing the original code's effective depth)
#   wave 2 BEST   (this rig's proven recipe: chunk 100, dark_noise aug, batch 8):
#     B1 s1000_e30  30-epoch regime, seed 1000
#     B2 s2000_e30  same, seed 2000 (checkpoint variance here is 19–49 pts)
#     B3 s1000_e60  60-epoch regime, seed 1000 (the 20260902/20260904 lineage)
# n_action_steps is set equal to chunk_size only so ACTConfig validates (its default 100
# exceeds a 60-frame chunk); act_runner overrides it at deploy. Temporal ensembling is deploy-time.
set -uo pipefail
cd "$(dirname "$0")"

DATE="${DATE:-20260914}"
NAME=isolate_pick_right_$DATE
REPO=ETHRC/yam_$NAME
DS=$HOME/.cache/huggingface/lerobot/$REPO
BASE=$HOME/.cache/lerobot-predecoded/yam_$NAME
GEOM=wrist_native
CACHE=${BASE}_$GEOM
PY=./.venv/bin/python3
ROOT=recordings/$DATE
SEGS=$ROOT/home_segments.json
EPOCHS="${EPOCHS:-60}"
PAPER_STEPS="${PAPER_STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"; BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6
WINDOW_FLOOR="${WINDOW_FLOOR:-20}"
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }
STATUS=outputs/train/$NAME.status

say "=== ISOLATE-THEN-PICK right-arm training ($DATE) ==="
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
say "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 30 ] || { say "FATAL: under 30 GB free"; echo "FAIL disk" > "$STATUS"; exit 1; }

say "geometry selftest ($GEOM)"
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; echo "FAIL geometry" > "$STATUS"; exit 1; }
# The bus geometry this checkpoint will meet: wrist native 480x640, top 270x480 pad.
# Derived from configs/yam/cameras_right_top.yaml on 2026-09-14 (publish_resize
# commented out on camera_right). Re-derive if that file changed.

say "[0/5] segmenting $ROOT at every return to home"
$PY tools/segment_home_episodes.py --root "$ROOT" --out "$SEGS" > "outputs/train/$NAME.segments.txt" 2>&1 \
  || { say "FATAL: segmenter failed"; cat "outputs/train/$NAME.segments.txt"; echo "FAIL segment" > "$STATUS"; exit 1; }
tail -4 "outputs/train/$NAME.segments.txt"
N_TRAIN=$($PY -c "import json;print(json.load(open('$SEGS'))['summary']['train'])")
N_HOLD=$($PY -c "import json;print(json.load(open('$SEGS'))['summary']['holdout'])")
say "      train windows=$N_TRAIN holdout=$N_HOLD"
[ "$N_TRAIN" -ge "$WINDOW_FLOOR" ] || { say "FATAL: only $N_TRAIN train windows (< $WINDOW_FLOOR)"; echo "FAIL windows=$N_TRAIN" > "$STATUS"; exit 1; }

if [ -f "$DS/meta/info.json" ]; then
  say "[1/5] dataset already at $DS -- skip (rm -rf it to re-export after more recording)"
else
  say "[1/5] exporting $REPO (right, full windows from $SEGS)"
  $PY tools/export_lerobot.py --root "$ROOT" \
    --arms right --cameras wrist_right_top \
    --window-mode full --windows "$SEGS" \
    --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
    --repo-id "$REPO" > "outputs/train/$NAME.export.txt" 2>&1
  tail -12 "outputs/train/$NAME.export.txt"
  [ -f "$DS/meta/info.json" ] || { say "FATAL: export produced nothing"; echo "FAIL export" > "$STATUS"; exit 1; }
fi
N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0)
say "      exported: $N_EP episodes, $N_FR frames"
[ "$N_EP" -ge "$WINDOW_FLOOR" ] || { say "FATAL: only $N_EP episodes exported. Removing $DS."; rm -rf "$DS"; echo "FAIL exported=$N_EP" > "$STATUS"; exit 1; }
STEPS="${STEPS:-$(( EPOCHS * N_FR / BATCH ))}"
say "      steps=$STEPS  (${EPOCHS} epochs over $N_FR frames @ batch $BATCH)"

if [ -d "$BASE" ]; then say "[2/5] base cache exists -- skip"; else
  say "[2/5] predecoding ..."; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" 2>&1 | tail -3
  [ -d "$BASE" ] || { say "FATAL: predecode nothing"; echo "FAIL predecode" > "$STATUS"; exit 1; }
fi
if [ -d "$CACHE" ]; then say "[3/5] baked cache exists -- skip"; else
  say "[3/5] baking $GEOM ..."; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" 2>&1 | tail -3
  [ -d "$CACHE" ] || { say "FATAL: resize nothing"; echo "FAIL resize" > "$STATUS"; exit 1; }
fi

# [4/5] two waves of three concurrent runs. Each run = "name|chunk|aug|seed|steps|extra".
E30=$(( 30 * N_FR / BATCH )); E60=$(( 60 * N_FR / BATCH ))
WAVE1="P1_c60|60|none|1000|$PAPER_STEPS| P2_c100|100|none|1000|$PAPER_STEPS| P3_c60_dec7|60|none|1000|$PAPER_STEPS|--policy.n_decoder_layers=7"
WAVE2="B1_c100_s1000_e30|100|dark_noise|1000|$E30| B2_c100_s2000_e30|100|dark_noise|2000|$E30| B3_c100_s1000_e60|100|dark_noise|1000|$E60|"
launch(){  # one run, foreground; caller backgrounds it
  IFS='|' read -r TAG CH AUGR SD STP EXTRA <<< "$1"
  RUN=${NAME}_$TAG; OUT=outputs/train/$RUN; LOG=outputs/train/$RUN.log; ST=outputs/train/$RUN.status
  if [ -f "$OUT.exit" ] && [ "$(cat "$OUT.exit")" = "0" ]; then say "      $RUN already done -- skip"; return; fi
  say "[4/5] training $RUN | chunk=$CH aug=$AUGR seed=$SD steps=$STP $EXTRA"
  echo "task=isolate_pick mode=UNCONDITIONED windows=full chunk=$CH aug=$AUGR seed=$SD steps=$STP extra='$EXTRA' cache=$CACHE episodes=$N_EP frames=$N_FR dataset=$REPO holdout_segments=$N_HOLD started=$(date)" > "$ST"
  rm -rf "$OUT" "$OUT.exit"
  LEROBOT_PREDECODED_ROOT="$CACHE" ACT_AUG=$AUGR ACT_GEOMETRY=$GEOM $PY \
    tools/train_act_dark_noise.py --dataset.repo_id="$REPO" --policy.type=act \
    --policy.device=cuda --policy.chunk_size=$CH --policy.n_action_steps=$CH --policy.push_to_hub=false --steps=$STP \
    --save_freq=$SAVE_FREQ --batch_size=$BATCH --seed=$SD --num_workers=$WORKERS \
    $EXTRA --output_dir="$OUT" > "$LOG" 2>&1
  RC=$?; echo $RC > "$OUT.exit"
  if [ $RC -eq 0 ]; then echo "DONE $(date)" >> "$ST"; say "      $RUN done"; else echo "FAIL rc=$RC $(date)" >> "$ST"; say "      $RUN FAILED rc=$RC (see $LOG)"; fi
}
for W in 1 2; do
  eval "SPEC=\$WAVE$W"
  say "=== wave $W: $(echo $SPEC | tr ' ' '\n' | cut -d'|' -f1 | tr '\n' ' ')"
  PIDS=""
  for spec in $SPEC; do launch "$spec" & PIDS="$PIDS $!"; sleep 20; done
  wait $PIDS
  say "=== wave $W finished"
done
say "[5/5] all runs finished"
echo "ALL DONE $(date)" >> "$STATUS"
