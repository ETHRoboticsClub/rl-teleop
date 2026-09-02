#!/usr/bin/env bash
# Build a dot-dropout conditioned cache from a baseline cache.
# Args: <DS_dir> <baseline_cache> <dot_cache_out>
# REQUIRES pre-built, human-reviewed labels-20260902.json + mapping-20260902.json
# in tools/targetdot/. Running SAM2 auto-labeling on a fresh corpus unsupervised
# can place dots on the WRONG packet (poisoning the policy toward a wrong target),
# so this refuses to fabricate labels: no reviewed labels -> exit 1 -> baseline.
set -uo pipefail
cd "$(dirname "$0")/../.."
DS="$1"; SRC="$2"; DST="$3"
LBL=tools/targetdot/labels-20260902.json
MAP=tools/targetdot/mapping-20260902.json
DROP="${DOT_DROPOUT_FRAC:-0.2}"
[ -f "$LBL" ] && [ -f "$MAP" ] || { echo "no reviewed labels ($LBL / $MAP) -- conditioning needs a supervised SAM2 label+review pass first"; exit 1; }
[ -d "$DST" ] && { echo "dot cache already exists: $DST"; exit 0; }
echo "burning dot-dropout cache (frac=$DROP): $SRC -> $DST"
./.venv/bin/python3 tools/targetdot/burn_dots_dropout.py --src "$SRC" --dst "$DST" --labels "$LBL" --mapping "$MAP" --dropout-frac "$DROP"
