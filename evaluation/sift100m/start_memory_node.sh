#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <node-index 0..4> [start|stop|restart|status] [extra args...]" >&2
  exit 1
fi

NODE_INDEX="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

if ! [[ "$NODE_INDEX" =~ ^[0-9]+$ ]] || (( NODE_INDEX < 0 || NODE_INDEX >= MEMORY_NODES )); then
  echo "error: node-index must be in [0, $((MEMORY_NODES - 1))]" >&2
  exit 1
fi

COMMON_SCRIPT="$PROJECT_DIR/scripts/start_memory_node.sh"
NODE_NUMBER=$((NODE_INDEX + 1))
PORT="$((SERVER_PORT_BASE + NODE_INDEX))"
INDEX_FILE="${INDEX_PREFIX}_node${NODE_NUMBER}_of${MEMORY_NODES}.dat"
STORAGE_PEERS="$(join_storage_endpoints)"
COMMAND="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$INDEX_FILE" && "$COMMAND" != "stop" && "$COMMAND" != "status" ]]; then
  echo "warning: index shard not found yet: $INDEX_FILE" >&2
fi

# shellcheck disable=SC2206
storage_peer_args=( $STORAGE_PEERS )
args=(
  "$COMMAND"
  "$@"
  --num-clients "$NUM_CLIENTS"
  --port "$PORT"
  --mn-memory "$MN_MEMORY"
  --index-file "$INDEX_FILE"
  --insert-execution "$INSERT_EXECUTION"
  --storage-id "$NODE_INDEX"
  --storage-peers "${storage_peer_args[@]}"
  --dim "$DIM"
  --R "$R"
  --beam-width-construction "$BEAM_WIDTH"
  --alpha "$ALPHA"
  --rabitq-bits "$RABITQ_BITS"
  --search-mode "$SEARCH_MODE"
  --k "$K"
  --index-prefix "$INDEX_PREFIX"
  --storage-owner-batch-max "$STORAGE_OWNER_BATCH_MAX"
  --storage-owner-batch-wait-us "$STORAGE_OWNER_BATCH_WAIT_US"
  --storage-owner-cache-mb "$STORAGE_OWNER_CACHE_MB"
  --storage-owner-peer-rdma-tokens "$STORAGE_OWNER_PEER_RDMA_TOKENS"
  --storage-owner-rpc-depth "$STORAGE_OWNER_RPC_DEPTH"
  --storage-owner-rpc-timeout-ms "$STORAGE_OWNER_RPC_TIMEOUT_MS"
)

exec "$COMMON_SCRIPT" "${args[@]}"
