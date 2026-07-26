#!/usr/bin/env bash
# Live data-quality dashboard for kitting recording.
# On every saved episode it: (1) re-labels offline with the correct gripper refs
# (fixes the live-labeler mislabel), (2) writes a gripper-health qa.json, and
# (3) regenerates the review HTML, served over http so you watch it live.
#
# Usage:  ./review_watch.sh [recordings_dir] [http_port]
#   view on your laptop:  ssh -L 8792:localhost:8792 <rig> then open http://localhost:8792
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REC="${1:-recordings}"
PORT="${2:-8792}"
SERVE_DIR="$(mktemp -d)"
HTML="$SERVE_DIR/index.html"

echo "<!doctype html><meta http-equiv=refresh content=5><body style=background:#0d1117;color:#ccc;font-family:system-ui><p style=margin:40px>waiting for first episode…</p>" > "$HTML"
( cd "$SERVE_DIR" && python3 -m http.server "$PORT" --bind 0.0.0.0 ) >/tmp/review_http.log 2>&1 &
HTTP_PID=$!
cleanup(){ kill "$HTTP_PID" 2>/dev/null || true; rm -rf "$SERVE_DIR"; }
trap cleanup EXIT

echo "live review dashboard -> http://localhost:${PORT}  (auto-refreshes; regenerates on each saved episode)"
echo "watching: $REC   (Ctrl-C to stop)"

SIG_PREV=""
while true; do
  SIG="$(find "$REC" \( -name 'yam_left.mcap' -o -name 'annotations.json' -o -name 'qa.json' \) -printf '%p:%T@\n' 2>/dev/null | sort | md5sum)"
  if [ "$SIG" != "$SIG_PREV" ]; then
    # (1) re-label any episode whose qa.json is missing or older than its mcap.
    while IFS= read -r mcap; do
      [ -z "$mcap" ] && continue
      d="$(dirname "$mcap")"; qa="$d/qa.json"
      if [ ! -f "$qa" ] || [ "$mcap" -nt "$qa" ]; then
        echo "  [$(date +%H:%M:%S)] $(uv run python qa_label.py "$d" 2>/dev/null)"
      fi
    done < <(find "$REC" -name 'yam_left.mcap' | sort)
    # (2) regenerate the dashboard + a concise console summary.
    python3 review_corpus.py "$REC" --html "$HTML" >/tmp/review_last.txt 2>/dev/null
    grep -E "usable \(|DEAD GRIPPER|CORRUPT mcap|MISLABEL|in target region|NOT CLEAR|CLEAR —|^    [0-9]\." /tmp/review_last.txt | sed 's/^/    /'
    echo "  [$(date +%H:%M:%S)] dashboard refreshed"
    SIG_PREV="$SIG"
  fi
  sleep 3
done
