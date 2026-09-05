#!/usr/bin/env bash
# run_night_matrix_20260905.sh — the night batch: window/camera/chunk variants of
# the combined right-arm grasp corpus, trained sequentially at 100k steps each.
#
# Corpus: 20260902 right episodes (104-window set) + everything in 20260904/05/06.
# All variants UNCONDITIONED, uncleaned (no --keep), --close-idx-frac 1.00,
# wrist_native geometry, dark_noise, batch 8, seed 1000. Three factors vary, one
# at a time:  WINDOW SHAPE  ×  CAMERA SET  ×  TRAINING CHUNK.
#
# NAMING (Tommaso 2026-09-05: "very clear names"):  night05_<window>_<cams>_c<chunk>
#   window: base    = current descent-start windows        (close median ~54)
#           plus1s  = +1.0 s lead  (--pregrasp-margin-s 1.5, export chunk 130)
#           apex    = start near top of approach (--pregrasp-plane-m 0.15, exp 200)
#           double  = +2.5 s lead  (--pregrasp-margin-s 3.0, export chunk 200)
#           triple  = +5.0 s lead  (--pregrasp-margin-s 5.5, export chunk 300)
#   cams:   both = wrist+top   wrist = wrist camera only
#   cN:     --policy.chunk_size at TRAINING (c50 twins reuse the parent dataset —
#           export windows are longer than the chunk, the sampler slices them)
#
# Idempotent: finished stages are skipped (dataset exists / cache exists /
# .exit==0), so a crash or trim can be resumed by re-running.
# Filter:  VARIANTS_ONLY="base_both_c100 base_wrist_c50" ./run_night_matrix_20260905.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python3
GEOM=wrist_native
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-100000}"     # final checkpoint only — disk is the constraint
BATCH=8; SEED=1000; AUG=dark_noise; WORKERS=6
DATES="${DATES:-20260902 20260904 20260905 20260906}"
STAGE=$HOME/.cache/export-roots/right_night_20260905
MIN_FREE_GB=20
mkdir -p outputs/train
say(){ echo "[$(date +%H:%M:%S)] $*"; }
free_gb(){ df --output=avail -BG . | tail -1 | tr -dc '0-9'; }

# window | cams | export_chunk | pre_s | extra export flags | train_chunk
# ORDER = factor-space coverage first (Tommaso 2026-09-05: "at least four
# DIFFERENT variations done by ~10:00"): the first four span baseline / wrist+c50
# / +1s window / apex window, so an early morning finds one of each family
# rather than four siblings. The rest fill in the pairs, cheap c50 twins early
# (they reuse the parent dataset), long-horizon exotics last.
VARIANTS=(
  "base|wrist_right_top|100|6.0||100"
  "base|wrist_right|100|6.0||50"
  "plus1s|wrist_right|130|8.0|--pregrasp-margin-s 1.5|130"
  "apex|wrist_right|200|10.0|--pregrasp-plane-m 0.15|200"
  "base|wrist_right|100|6.0||100"
  "base|wrist_right_top|100|6.0||50"
  "plus1s|wrist_right|130|8.0|--pregrasp-margin-s 1.5|50"
  "apex|wrist_right|200|10.0|--pregrasp-plane-m 0.15|50"
  "plus1s|wrist_right_top|130|8.0|--pregrasp-margin-s 1.5|130"
  "plus1s|wrist_right_top|130|8.0|--pregrasp-margin-s 1.5|50"
  "apex|wrist_right_top|200|10.0|--pregrasp-plane-m 0.15|200"
  "double|wrist_right_top|200|10.0|--pregrasp-margin-s 3.0|200"
  "triple|wrist_right_top|300|12.0|--pregrasp-margin-s 5.5|300"
)

say "=== NIGHT MATRIX: ${#VARIANTS[@]} variants, $STEPS steps each ==="
$PY tools/act_bus_geometry.py >/dev/null || { say "FATAL: geometry selftest failed"; exit 1; }

# ── stage once ──────────────────────────────────────────────────────────────
rm -rf "$STAGE"; mkdir -p "$STAGE"
n_link=0
for d in $DATES; do
  for ep in recordings/$d/episode_*; do
    [ -d "$ep" ] || continue
    ln -s "$(cd "$ep" && pwd)" "$STAGE/$(basename "$ep")"; n_link=$((n_link+1))
  done
done
say "staged $n_link episodes from: $DATES"
[ "$n_link" -ge 5 ] || { say "FATAL: staging nearly empty"; exit 1; }

# ── label once (cockpit-unscored episodes; idempotent) ──────────────────────
for eplink in "$STAGE"/episode_*; do
  ep=$(readlink -f "$eplink")
  [ -f "$ep/yam_right.mcap" ] || continue
  [ -f "$ep/camera_top-rgb-timestamp.npy" ] || { say "  $(basename "$ep") still open — skipped"; continue; }
  [ -s "$ep/annotations_right.json" ] && continue
  say "  labeling $(basename "$ep")"
  $PY -m robots_realtime.labeling.label_episode "$ep" --arm right \
    --open-ref 1.0 --closed-ref 0.0 2>&1 | tail -1 | sed 's/^/    /'
done

# ── honesty report once (frame yield is enforced nowhere else) ──────────────
$PY tools/act_horizon_report.py --root "$STAGE" --arm right --cameras wrist_right_top \
  --pre-s 6.0 > outputs/train/night05.horizon.txt 2>&1 || say "  (horizon report failed — non-fatal)"

# ── the queue ───────────────────────────────────────────────────────────────
for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r WIN CAMS XCHUNK PRE EXTRA TCHUNK <<< "$spec"
  case "$CAMS" in wrist_right_top) CS=both;; wrist_right) CS=wrist;; *) CS=$CAMS;; esac
  NAME=${WIN}_${CS}_c${TCHUNK}
  if [ -n "${VARIANTS_ONLY:-}" ]; then
    case " $VARIANTS_ONLY " in *" $NAME "*) ;; *) say "── $NAME: filtered out — skip"; continue;; esac
  fi
  RUN=night05_$NAME
  DSKEY=${WIN}_${CS}                       # dataset shared across train-chunk twins
  REPO=ETHRC/yam_grasp_right_night05_$DSKEY
  DS=$HOME/.cache/huggingface/lerobot/$REPO
  BASE=$HOME/.cache/lerobot-predecoded/night05_$DSKEY
  CACHE=${BASE}_wristnative
  OUT=outputs/train/$RUN
  LOG=outputs/train/$RUN.log
  STATUS=outputs/train/$RUN.status
  say "━━ $NAME  (dataset $DSKEY: export-chunk $XCHUNK pre_s $PRE $EXTRA · train chunk $TCHUNK)"

  if [ -f "$OUT.exit" ] && [ "$(cat "$OUT.exit")" = "0" ]; then say "   already trained — skip"; continue; fi
  FREE=$(free_gb)
  [ "$FREE" -ge "$MIN_FREE_GB" ] || { say "   SKIP: only ${FREE} GB free"; echo "SKIP disk=${FREE}G" > "$STATUS"; continue; }

  if [ -f "$DS/meta/info.json" ]; then say "   [1/4] dataset exists — skip"; else
    say "   [1/4] export"
    $PY tools/export_lerobot.py --root "$STAGE" \
      --arms right --cameras "$CAMS" \
      --window-mode grasp-pose --pre-s "$PRE" --post-s 2.0 \
      --chunk-frames "$XCHUNK" --close-idx-frac 1.00 \
      --gripper-open-ref 1.0 --gripper-closed-ref 0.0 \
      --repo-id "$REPO" $EXTRA > "$OUT.export.log" 2>&1
    [ -f "$DS/meta/info.json" ] || { say "   FAIL export (see $OUT.export.log)"; echo "FAIL export" > "$STATUS"; continue; }
  fi
  N_EP=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])")
  N_FR=$($PY -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])")
  say "   windows=$N_EP frames=$N_FR (epochs at ${STEPS}: $(( STEPS * BATCH / N_FR )))"
  [ "$N_EP" -ge 150 ] || { say "   FAIL: only $N_EP windows"; rm -rf "$DS"; echo "FAIL windows=$N_EP" > "$STATUS"; continue; }

  if [ ! -d "$CACHE" ]; then
    say "   [2/4] predecode"; $PY tools/predecode_ffmpeg.py --dataset-root "$DS" --output-root "$BASE" > "$OUT.predecode.log" 2>&1
    [ -d "$BASE" ] || { say "   FAIL predecode"; echo "FAIL predecode" > "$STATUS"; continue; }
    say "   [3/4] bake $GEOM"; $PY tools/predecode_resize.py --geometry "$GEOM" --source "$BASE" --dest "$CACHE" > "$OUT.bake.log" 2>&1
    [ -d "$CACHE" ] || { say "   FAIL bake"; echo "FAIL bake" > "$STATUS"; continue; }
    # predecode_resize SYMLINKS already-correct features into $CACHE — sometimes
    # whole DIRECTORIES (the wrist tree at wrist_native). Deleting $BASE naively
    # breaks them (cost 02:05), and `ln` cannot hardlink a directory (cost 02:20,
    # and the old exit-in-subshell didn't even stop the deletion). Replace every
    # symlink with a hardlink-copy of its target, verify, THEN drop the base.
    HL_FAIL=0
    while IFS= read -r l; do
      t=$(readlink -f "$l") || { HL_FAIL=1; break; }
      rm "$l"
      if [ -d "$t" ]; then cp -al "$t" "$l" || { HL_FAIL=1; break; }
      else ln "$t" "$l" || { HL_FAIL=1; break; }
      fi
    done < <(find "$CACHE" -type l)
    if [ "$HL_FAIL" != "0" ] || [ -n "$(find "$CACHE" -xtype l 2>/dev/null | head -1)" ]; then
      say "   FAIL hardlink conversion in $CACHE — keeping $BASE, not training on it"
      echo "FAIL hardlink" > "$STATUS"; continue
    fi
    rm -rf "$BASE"        # full-res predecode not needed once baked — disk
  else say "   [2-3/4] baked cache exists — skip"; fi

  say "   [4/4] train ($STEPS steps, chunk $TCHUNK)"
  echo "mode=UNCONDITIONED window=$WIN cams=$CS train_chunk=$TCHUNK export_chunk=$XCHUNK windows=$N_EP frames=$N_FR steps=$STEPS started=$(date)" > "$STATUS"
  rm -rf "$OUT" "$OUT.exit"
  LEROBOT_PREDECODED_ROOT="$CACHE" ACT_AUG=$AUG ACT_GEOMETRY=$GEOM $PY \
    tools/train_act_dark_noise.py --dataset.repo_id="$REPO" --policy.type=act \
    --policy.device=cuda --policy.chunk_size="$TCHUNK" --policy.n_action_steps="$TCHUNK" \
    --policy.push_to_hub=false \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --batch_size=$BATCH --seed=$SEED \
    --num_workers=$WORKERS --output_dir="$OUT" > "$LOG" 2>&1
  rc=$?; echo "$rc" > "$OUT.exit"
  if [ "$rc" = "0" ]; then say "   ✓ $NAME done"; echo "DONE $(date)" >> "$STATUS"
  else say "   ✗ $NAME train FAILED rc=$rc (tail: $LOG)"; echo "FAIL train rc=$rc" >> "$STATUS"; fi
done

say "=== MATRIX COMPLETE ==="
for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r WIN CAMS _ _ _ TCHUNK <<< "$spec"
  case "$CAMS" in wrist_right_top) CS=both;; *) CS=wrist;; esac
  s=outputs/train/night05_${WIN}_${CS}_c${TCHUNK}.status
  [ -f "$s" ] && echo "  ${WIN}_${CS}_c${TCHUNK}: $(tail -1 "$s")" || echo "  ${WIN}_${CS}_c${TCHUNK}: not started"
done
