#!/bin/bash
# Run one local practice trial end to end:
#   env (process driver) + this entry + scorer.
# usage: run-trial.sh <trial-dir> <out-dir> [port]
set -u
TRIAL=$1
OUT=$2
PORT=${3:-8080}
KIT=~/workspace/makefun/kits/free-model-aggregator-01-kit/free-model-aggregator-01-kit
ENTRY=~/workspace/makefun/free-model-aggregator-01
mkdir -p "$OUT/state" "$OUT/scores"
rm -rf "$OUT/state/export"

MAKEFUN_STATE_DIR="$OUT/state" MAKEFUN_ENV_DATA_DIR="$TRIAL/input/env" MAKEFUN_ENV_PORT="$PORT" \
  node "$KIT/program/env/server.mjs" > "$OUT/env.log" 2>&1 &
ENV_PID=$!
sleep 1

MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" MAKEFUN_PROMPT_FILE="$TRIAL/input/prompt.txt" PORT=$((PORT + 1)) \
  node "$ENTRY/main.ts" > "$OUT/server.log" 2>&1
SERVER_EXIT=$?
echo "server exit: $SERVER_EXIT"

kill -TERM $ENV_PID 2>/dev/null
wait $ENV_PID 2>/dev/null

if [ -f "$OUT/state/export/state.json" ]; then
  MAKEFUN_OUTPUT_DIR="$OUT/state/export" MAKEFUN_REFERENCE_DIR="$TRIAL/reference" MAKEFUN_SCORES_DIR="$OUT/scores" \
    node "$KIT/program/scorer/score.mjs" > "$OUT/score.log" 2>&1
  echo "scorer exit: $?"
  cat "$OUT/scores/score.json" 2>/dev/null || cat "$OUT/score.log"
else
  echo "NO state export; env log:"
  tail -20 "$OUT/env.log"
fi
