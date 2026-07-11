#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NODE_ID="${1:?usage: start_memory_node.sh <node-id> [profile]}"
PROFILE="${2:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

if (( NODE_ID < 1 || NODE_ID > SHARDS )); then
  echo "node id must be in [1,$SHARDS]" >&2
  exit 1
fi

ensure_built dvstor_memory_node

PORT=$((BASE_PORT + NODE_ID - 1))
validate_index_metadata storage "$NODE_ID"
SHARD_FILE="$(shard_file "$NODE_ID")"

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
  --threads "$SERVICE_THREADS"
  --dim "$DIM"
  --max-vectors "$MAX_VECTORS"
  --R "$R"
  --beam-width-construction "$BUILD_BEAM"
  --alpha "$ALPHA"
  --k "$K"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --mn-memory "$MN_MEMORY_GB"
  --storage-id "$((NODE_ID - 1))"
  --storage-peers "${SERVER_ARGS[@]}"
  --storage-owner-coroutines "${STORAGE_OWNER_COROUTINES:-4}"
  --storage-owner-batch-max "${STORAGE_OWNER_BATCH_MAX:-32}"
  --storage-owner-batch-wait-us "${STORAGE_OWNER_BATCH_WAIT_US:-100}"
  --storage-owner-peer-rdma-tokens "${STORAGE_OWNER_PEER_RDMA_TOKENS:-8}"
  --storage-owner-rpc-depth "${STORAGE_OWNER_RPC_DEPTH:-16}"
  --storage-owner-rpc-timeout-ms "${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
  --storage-owner-construction-beam-width "${STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH:-$BUILD_BEAM}"
  --storage-owner-search-snapshot-batch "${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-64}"
  --storage-owner-prune-max-candidates "${STORAGE_OWNER_PRUNE_MAX_CANDIDATES:-128}"
  --storage-owner-update-mode "${STORAGE_OWNER_UPDATE_MODE:-local_stitch}"
  --storage-owner-anchor-hints "${STORAGE_OWNER_ANCHOR_HINTS:-4}"
  --storage-owner-anchor-beam-width "${STORAGE_OWNER_ANCHOR_BEAM_WIDTH:-64}"
  --storage-owner-anchor-expand-cap "${STORAGE_OWNER_ANCHOR_EXPAND_CAP:-16}"
  --storage-owner-anchor-remote-rescue-cap "${STORAGE_OWNER_ANCHOR_REMOTE_RESCUE_CAP:-4}"
  --storage-owner-local-stitch-sync-fast-path "${STORAGE_OWNER_LOCAL_STITCH_SYNC_FAST_PATH:-true}"
  --storage-owner-maintenance-mode "${STORAGE_OWNER_MAINTENANCE_MODE:-finalize}"
  --storage-owner-maintenance-workers "${STORAGE_OWNER_MAINTENANCE_WORKERS:-8}"
  --storage-owner-reverse-mode "${STORAGE_OWNER_REVERSE_MODE:-async}"
  --storage-owner-reverse-queue-depth "${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
  --storage-owner-reverse-flush-us "${STORAGE_OWNER_REVERSE_FLUSH_US:-200}"
  --storage-owner-reverse-coalesce-max "${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}")

printf '[memory-node-%s] command:' "$NODE_ID"; printf ' %q' "${cmd[@]}"; echo
nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "memory node $NODE_ID started: pid $(cat "$PID_FILE"), log $LOG_FILE"
