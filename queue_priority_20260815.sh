#!/usr/bin/env bash
# Give handoff_left_wristnative the GPU to itself, and queue the two right-arm
# grasp lanes behind it. Operator's call 2026-08-15 12:15.
#
# THE PROBLEM. Three ACT runs landed on one 5090 within 4 minutes of each other
# (graspmatched_bus 10:45:29, graspmatched_wristnative 10:45:31,
# handoff_left_wristnative 10:48:47). Nothing is broken, but 3-way time-slicing
# put every lane at ~0.105 s/step where a solo lane runs at 0.060 -- so all
# three finish late and the one the operator actually needs finishes last.
#
# THE PLAN.
#   1. graspmatched_bus          -> stop once its 50k checkpoint is verified
#   2. graspmatched_wristnative  -> stop once its 50k checkpoint is verified
#   3. handoff_left_wristnative  -> runs alone to 100k  (PRIORITY 1)
#   4. graspmatched_bus          -> resume 50k -> 100k
#   5. graspmatched_wristnative  -> resume 50k -> 100k
#
# 4 and 5 run SEQUENTIALLY, not together: putting them back on the GPU at the
# same time recreates the contention this script exists to remove.
#
# NOTHING IS LOST. Both lanes stop on a complete 50k checkpoint and resume from
# it via --config_path + --resume=true, which restores optimizer state, RNG
# state and the step counter. The only cost is the steps run since 50k, which
# the resume redoes. Their train_config.json already says steps=100000, so a
# resumed lane runs to 100k and stops on its own.
#
# WHY THE CHECKPOINT IS VERIFIED BEFORE EVERY KILL. A SIGKILL landing in the
# middle of a checkpoint write leaves a truncated safetensors that loads without
# raising. The gate below requires all 11 files, a plausible size for both
# safetensors, a readable training_step.json, AND the same result twice 10 s
# apart, so a write in progress can never pass.
#
#   ./queue_priority_20260815.sh            # run it (use tmux, it is long-lived)
#   tail -f outputs/train/queue_priority.log
set -uo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python3
PD=$HOME/.cache/lerobot-predecoded
QLOG=outputs/train/queue_priority.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

# ── a checkpoint is only safe to kill on if it is COMPLETE ──────────────────
ckpt_complete(){   # <run> <step-dir>
    local d=outputs/train/$1/checkpoints/$2
    [ -d "$d" ] || return 1
    [ "$(find "$d" -type f 2>/dev/null | wc -l)" -eq 11 ] || return 1
    [ -s "$d/training_state/training_step.json" ] || return 1
    local m o
    m=$(stat -c%s "$d/pretrained_model/model.safetensors" 2>/dev/null || echo 0)
    o=$(stat -c%s "$d/training_state/optimizer_state.safetensors" 2>/dev/null || echo 0)
    [ "$m" -ge 200000000 ] && [ "$o" -ge 400000000 ]
}

stable_ckpt(){     # same answer twice, 10 s apart -> not mid-write
    ckpt_complete "$1" "$2" || return 1
    sleep 10
    ckpt_complete "$1" "$2"
}

stop_run(){
    local r=$1
    log "  stopping tr_$r"
    tmux kill-session -t "tr_$r" 2>/dev/null
    sleep 3
    pkill -f "train_act_dark_noise.py.*outputs/train/$r" 2>/dev/null
    sleep 3
    # the wrapper's `echo $? > .exit` may fire on the way down; that exit code
    # describes a kill, not a finish, so drop it or step 4 misreads it.
    rm -f "outputs/train/$r.exit"
    log "  tr_$r stopped, checkpoints on disk: $(ls outputs/train/$r/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' ')"
}

wait_ckpt_then_stop(){
    local r=$1
    if ! tmux has-session -t "tr_$r" 2>/dev/null; then
        log "$r: no session running, nothing to stop"
        return 0
    fi
    log "$r: waiting for a verified 050000 checkpoint"
    until stable_ckpt "$r" 050000; do
        if ! tmux has-session -t "tr_$r" 2>/dev/null; then
            log "$r: SESSION DIED before a complete checkpoint -- not queueing it"
            return 1
        fi
        sleep 30
    done
    log "$r: checkpoint 050000 verified complete"
    stop_run "$r"
}

wait_finish(){
    local r=$1
    log "$r: waiting for it to finish"
    while tmux has-session -t "tr_$r" 2>/dev/null; do sleep 60; done
    log "$r: finished (exit=$(cat "outputs/train/$r.exit" 2>/dev/null || echo '?'), last: $(grep -oE 'step:[0-9K]+ .*loss:[0-9.]+' "outputs/train/$r.log" 2>/dev/null | tail -1))"
}

resume_run(){      # <run> <cache> <geometry>
    local r=$1 cache=$2 geom=$3
    local free
    free=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
    log "$r: resuming to 100k (geometry $geom, ${free} GB free)"
    if [ "$free" -lt 3 ]; then
        log "$r: ABORT -- under 3 GB free, a checkpoint write would truncate silently"
        return 1
    fi
    if [ ! -e "outputs/train/$r/checkpoints/last/pretrained_model/train_config.json" ]; then
        log "$r: ABORT -- no checkpoints/last/pretrained_model/train_config.json to resume from"
        return 1
    fi
    # env inline on the tmux command, never exported: a tmux session inherits
    # the SERVER's environment, so an export reaches nothing and the run
    # silently falls back to torchcodec and dies. Cost a night on 2026-08-14.
    tmux new-session -d -s "tr_$r" -c "$PWD" \
"LEROBOT_PREDECODED_ROOT='$cache' ACT_AUG=dark_noise ACT_GEOMETRY=$geom $PY \
tools/train_act_dark_noise.py \
--config_path=outputs/train/$r/checkpoints/last/pretrained_model/train_config.json \
--resume=true >> 'outputs/train/$r.log' 2>&1; echo \$? > 'outputs/train/$r.exit';"
    sleep 8
    if tmux has-session -t "tr_$r" 2>/dev/null; then
        log "  tr_$r up"
    else
        log "  WARNING: tr_$r did NOT start -- see outputs/train/$r.log"
        return 1
    fi
}

# ── run ─────────────────────────────────────────────────────────────────────
log "=== priority queue start: handoff_left_wristnative gets the GPU alone ==="

wait_ckpt_then_stop graspmatched_bus
wait_ckpt_then_stop graspmatched_wristnative

log "PRIORITY: handoff_left_wristnative now has the GPU to itself"
wait_finish handoff_left_wristnative

log "=== priority run done, draining the queue sequentially ==="
resume_run graspmatched_bus "$PD/yam_grasp_right_20260814_bus" bus && wait_finish graspmatched_bus
resume_run graspmatched_wristnative "$PD/yam_grasp_right_20260814_wristnative" wrist_native && wait_finish graspmatched_wristnative

log "=== QUEUE COMPLETE ==="
