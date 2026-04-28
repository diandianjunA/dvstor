#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON_SCRIPT="$SCRIPT_DIR/start_memory_node.sh"

NUM_CLIENTS="${NUM_CLIENTS:-1}"
PORT="${PORT:-1237}"
MN_MEMORY="${MN_MEMORY:-50}"
STORAGE_ID="${STORAGE_ID:-3}"
ENABLE_STORAGE_OWNER="${ENABLE_STORAGE_OWNER:-1}"
INSERT_EXECUTION="${INSERT_EXECUTION:-storage_delta}"
STORAGE_PEER_HOST="${STORAGE_PEER_HOST:-127.0.0.1}"
STORAGE_PEERS="${STORAGE_PEERS:-$STORAGE_PEER_HOST:2234 $STORAGE_PEER_HOST:2235 $STORAGE_PEER_HOST:2236 $STORAGE_PEER_HOST:2237 $STORAGE_PEER_HOST:2238}"
DIM="${DIM:-1024}"
R="${R:-64}"
BEAM_WIDTH_CONSTRUCTION="${BEAM_WIDTH_CONSTRUCTION:-400}"
ALPHA="${ALPHA:-1.2}"
RABITQ_BITS="${RABITQ_BITS:-4}"
SEARCH_MODE="${SEARCH_MODE:-rabitq_gpu}"
INDEX_PREFIX="${INDEX_PREFIX:-/data/xjs/index/shine_gpu_index/1024dim50M}"
INDEX_FILE="${INDEX_FILE:-${INDEX_PREFIX}_node4_of5.dat}"

ARGS=(
  --num-clients "$NUM_CLIENTS"
  --port "$PORT"
  --mn-memory "$MN_MEMORY"
  --index-file "$INDEX_FILE"
)

if [[ "$ENABLE_STORAGE_OWNER" == "1" ]]; then
  # shellcheck disable=SC2206
  STORAGE_PEER_ARGS=( $STORAGE_PEERS )
  ARGS+=(
    --insert-execution "$INSERT_EXECUTION"
    --storage-id "$STORAGE_ID"
    --storage-peers "${STORAGE_PEER_ARGS[@]}"
    --dim "$DIM"
    --R "$R"
    --beam-width-construction "$BEAM_WIDTH_CONSTRUCTION"
    --alpha "$ALPHA"
    --rabitq-bits "$RABITQ_BITS"
    --search-mode "$SEARCH_MODE"
    --index-prefix "$INDEX_PREFIX"
  )
fi

exec "$COMMON_SCRIPT" "$@" "${ARGS[@]}"
