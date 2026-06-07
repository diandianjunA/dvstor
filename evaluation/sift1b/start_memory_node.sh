#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift1b_common.sh"

NODE_ID="${1:?usage: start_memory_node.sh <1-5> [profile]}"
PROFILE="${2:-$PROFILE}"
PROFILE_ENV="$SCRIPT_DIR/profiles/${PROFILE}.env"
if [[ ! -f "$PROFILE_ENV" ]]; then
  echo "unknown profile: $PROFILE" >&2
  exit 1
fi
source "$PROFILE_ENV"

if (( NODE_ID < 1 || NODE_ID > SHARDS )); then
  echo "node id must be in [1,$SHARDS]" >&2
  exit 1
fi

ensure_built dvstor_memory_node

PORT=$((BASE_PORT + NODE_ID - 1))
SHARD_FILE="$(shard_file "$NODE_ID")"
if [[ ! -f "$SHARD_FILE" ]]; then
  echo "missing shard file: $SHARD_FILE" >&2
  echo "build it first with: $SCRIPT_DIR/build_sift1b_index.sh" >&2
  exit 1
fi

PID_FILE="$PID_DIR/memory_node_${NODE_ID}.pid"
LOG_FILE="$LOG_DIR/memory_node_${NODE_ID}_${PROFILE}.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "memory node $NODE_ID already running: pid $(cat "$PID_FILE")" >&2
  exit 1
fi

read -r -a SERVER_ARGS <<< "$(server_endpoints)"
read -r -a RDMA_ARGS <<< "$(common_rdma_args)"
cmd=("$BUILD_DIR/dvstor_memory_node"
  --is-server
  --num-clients 1
  --servers "${SERVER_ARGS[@]}"
  --port "$PORT"
  "${RDMA_ARGS[@]}"
  --server-index-file "$SHARD_FILE"
  --index-prefix "$INDEX_PREFIX"
  --data-path "$(base_bin)"
  --load-index
  --dim "$DIM"
  --max-vectors "$MAX_VECTORS"
  --R "$R"
  --beam-width "$SEARCH_BEAM"
  --beam-width-construction "$BUILD_BEAM"
  --alpha "$ALPHA"
  --k "$K"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --mn-memory "$MN_MEMORY_GB"
  --insert-execution "$INSERT_EXECUTION"
  --storage-id "$((NODE_ID - 1))")

if [[ "$INSERT_EXECUTION" == "storage_owner" ]]; then
  cmd+=(--storage-peers "${SERVER_ARGS[@]}")
fi

printf '[memory-node-%s] command:' "$NODE_ID"; printf ' %q' "${cmd[@]}"; echo
nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "memory node $NODE_ID started: pid $(cat "$PID_FILE"), log $LOG_FILE"
