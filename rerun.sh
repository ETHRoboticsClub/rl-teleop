#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDINGS_DIR="$ROOT_DIR/recordings"

usage() {
    cat <<'EOF'
Usage:
  ./rerun.sh [episode-name|date/episode-name|episode-path] [--stride N] [--no-mcap]

Examples:
  ./rerun.sh
  ./rerun.sh episode_210619_bc3d7209
  ./rerun.sh 20260531/episode_210619_bc3d7209
  ./rerun.sh recordings/20260531/episode_210619_bc3d7209 --stride 2

Bash completion:
  source <(./rerun.sh --completion)
EOF
}

list_episodes() {
    [[ -d "$RECORDINGS_DIR" ]] || return 0
    find "$RECORDINGS_DIR" -mindepth 2 -maxdepth 2 -type d -name 'episode_*' -printf '%T@ %P\n' \
        | sort -nr \
        | awk '{print $2}'
}

list_completion_names() {
    local episode
    list_episodes | while IFS= read -r episode; do
        printf '%s\n' "$episode"
        printf '%s\n' "${episode##*/}"
    done | awk '!seen[$0]++'
}

latest_episode() {
    local latest
    latest="$(list_episodes | head -n 1)"
    [[ -n "$latest" ]] || return 1
    printf '%s\n' "$RECORDINGS_DIR/$latest"
}

resolve_episode() {
    local name="$1"
    local candidate match_count match

    if [[ -d "$name" ]]; then
        cd "$name"
        pwd
        return 0
    fi

    candidate="$ROOT_DIR/$name"
    if [[ -d "$candidate" ]]; then
        cd "$candidate"
        pwd
        return 0
    fi

    candidate="$RECORDINGS_DIR/$name"
    if [[ -d "$candidate" ]]; then
        cd "$candidate"
        pwd
        return 0
    fi

    match_count=0
    match=""
    while IFS= read -r candidate; do
        if [[ "${candidate##*/}" == "$name" ]]; then
            match="$candidate"
            match_count=$((match_count + 1))
        fi
    done < <(list_episodes)

    if [[ "$match_count" -eq 1 ]]; then
        printf '%s\n' "$RECORDINGS_DIR/$match"
        return 0
    fi

    if [[ "$match_count" -gt 1 ]]; then
        printf 'Ambiguous episode name: %s\n' "$name" >&2
        list_episodes | awk -v name="$name" -v root="$RECORDINGS_DIR" '{
            if ($0 ~ "/" name "$") {
                print "  " root "/" $0
            }
        }' >&2
        return 1
    fi

    printf 'Could not find episode: %s\n' "$name" >&2
    printf 'Try one of:\n' >&2
    list_completion_names | sed 's/^/  /' | head -n 20 >&2
    return 1
}

emit_completion() {
    cat <<'EOF'
_rl_teleop_rerun_complete() {
    local cur script
    cur="${COMP_WORDS[COMP_CWORD]}"
    script="${COMP_WORDS[0]}"

    case "$cur" in
        --*)
            COMPREPLY=($(compgen -W "--help --stride --no-mcap" -- "$cur"))
            return 0
            ;;
    esac

    if [[ -x "$script" ]]; then
        COMPREPLY=($(compgen -W "$("$script" --list-completions 2>/dev/null)" -- "$cur"))
    elif [[ -x ./rerun.sh ]]; then
        COMPREPLY=($(compgen -W "$(./rerun.sh --list-completions 2>/dev/null)" -- "$cur"))
    fi
}
complete -F _rl_teleop_rerun_complete rerun.sh ./rerun.sh
EOF
}

episode_arg=""
stride=1
log_mcap=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --completion)
            emit_completion
            exit 0
            ;;
        --list-completions)
            list_completion_names
            exit 0
            ;;
        --stride)
            [[ $# -ge 2 ]] || { printf '%s\n' '--stride requires a value' >&2; exit 1; }
            stride="$2"
            shift 2
            ;;
        --no-mcap)
            log_mcap=0
            shift
            ;;
        -*)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$episode_arg" ]]; then
                printf 'Only one episode argument is supported.\n\n' >&2
                usage >&2
                exit 1
            fi
            episode_arg="$1"
            shift
            ;;
    esac
done

if ! [[ "$stride" =~ ^[0-9]+$ ]] || [[ "$stride" -lt 1 ]]; then
    printf '%s\n' '--stride must be a positive integer' >&2
    exit 1
fi

if [[ -z "$episode_arg" ]]; then
    episode_dir="$(latest_episode)" || {
        printf 'No episodes found under %s\n' "$RECORDINGS_DIR" >&2
        exit 1
    }
else
    episode_dir="$(resolve_episode "$episode_arg")"
fi

printf 'Opening episode in Rerun: %s\n' "$episode_dir"

cd "$ROOT_DIR"
uv run python - "$episode_dir" "$stride" "$log_mcap" <<'PY'
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import rerun as rr
from mcap.reader import make_reader


def read_json_mcap(path: Path):
    rows = []
    with path.open("rb") as f:
        for _, channel, msg in make_reader(f).iter_messages():
            if channel.message_encoding != "json":
                continue
            rows.append((msg.log_time / 1e9, channel.topic, json.loads(msg.data)))
    return rows


def first_timestamp(episode: Path) -> float:
    starts: list[float] = []
    for ts_file in episode.glob("*-rgb-timestamp.npy"):
        arr = np.load(ts_file)
        if len(arr):
            starts.append(float(arr[0]))
    for mcap in episode.glob("*.mcap"):
        rows = read_json_mcap(mcap)
        if rows:
            starts.append(rows[0][0])
    return min(starts) if starts else 0.0


def log_cameras(episode: Path, t0: float, stride: int) -> None:
    for mp4 in sorted(episode.glob("*-images-rgb.mp4")):
        camera = mp4.name.removesuffix("-images-rgb.mp4")
        ts_file = episode / f"{camera}-rgb-timestamp.npy"
        timestamps = np.load(ts_file) if ts_file.exists() else None

        print(f"camera {camera}: {mp4.name}", flush=True)
        for i, frame in enumerate(iio.imiter(mp4, plugin="pyav")):
            if i % stride:
                continue
            ts = float(timestamps[i]) if timestamps is not None and i < len(timestamps) else float(i / 30.0 + t0)
            rr.set_time_seconds("episode_time", ts - t0)
            rr.log(f"cameras/{camera}", rr.Image(frame))


def log_mcaps(episode: Path, t0: float) -> None:
    for mcap in sorted(episode.glob("*.mcap")):
        print(f"mcap {mcap.name}", flush=True)
        stream = mcap.stem
        for ts, topic, data in read_json_mcap(mcap):
            rr.set_time_seconds("episode_time", ts - t0)
            topic_name = topic.strip("/").replace("/", "_")

            for key, value in data.items():
                if isinstance(value, list):
                    for i, x in enumerate(value):
                        if isinstance(x, (int, float)):
                            rr.log(f"signals/{stream}/{topic_name}/{key}_{i}", rr.Scalars(float(x)))
                elif isinstance(value, (int, float)):
                    rr.log(f"signals/{stream}/{topic_name}/{key}", rr.Scalars(float(value)))


def main() -> None:
    episode = Path(sys.argv[1]).resolve()
    stride = int(sys.argv[2])
    should_log_mcap = bool(int(sys.argv[3]))

    if not episode.is_dir():
        raise SystemExit(f"Not an episode directory: {episode}")

    rr.init(f"episode/{episode.name}", spawn=True)
    t0 = first_timestamp(episode)

    log_cameras(episode, t0, stride)
    if should_log_mcap:
        log_mcaps(episode, t0)

    print("Loaded into Rerun.", flush=True)
    time.sleep(2)


if __name__ == "__main__":
    main()
PY
