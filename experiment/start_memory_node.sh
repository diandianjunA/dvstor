#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NODE_ID="${1:?usage: start_memory_node.sh <node-id> [profile]}"
PROFILE="${2:-${PROFILE:-00_baseline}}"
load_experiment_profile "$PROFILE"

if (( NODE_ID < 1 || NODE_ID > SHARDS )); then
  echo "node id must be in [1,$SHARDS]" >&2
  exit 1
fi

ensure_built dvstor_memory_node

PORT=$((BASE_PORT + NODE_ID - 1))
SHARD_FILE="$(shard_file "$NODE_ID")"
if [[ ! -f "$SHARD_FILE" ]]; then
  echo "missing shard file: $SHARD_FILE" >&2
  echo "build it first with: $EXPERIMENT_DIR/build_sift100m_index.sh" >&2
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
  cmd+=(--storage-owner-batch-max "${STORAGE_OWNER_BATCH_MAX:-32}")
  cmd+=(--storage-owner-batch-wait-us "${STORAGE_OWNER_BATCH_WAIT_US:-100}")
  cmd+=(--storage-owner-peer-rdma-tokens "${STORAGE_OWNER_PEER_RDMA_TOKENS:-8}")
  cmd+=(--storage-owner-rpc-depth "${STORAGE_OWNER_RPC_DEPTH:-16}")
  cmd+=(--storage-owner-rpc-timeout-ms "${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}")
  cmd+=(--storage-owner-construction-beam-width "${STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH:-$SEARCH_BEAM}")
  cmd+=(--storage-owner-search-snapshot-batch "${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-64}")
  cmd+=(--storage-owner-prune-max-candidates "${STORAGE_OWNER_PRUNE_MAX_CANDIDATES:-128}")
  update_mode="${STORAGE_OWNER_UPDATE_MODE:-exact}"
  if [[ "$update_mode" != "exact" ]]; then
    cmd+=(--storage-owner-update-mode "$update_mode")
  fi
  if [[ "$update_mode" == "anchored" ]]; then
    cmd+=(--storage-owner-anchor-hints "${STORAGE_OWNER_ANCHOR_HINTS:-4}")
    cmd+=(--storage-owner-anchor-beam-width "${STORAGE_OWNER_ANCHOR_BEAM_WIDTH:-64}")
    cmd+=(--storage-owner-anchor-expand-cap "${STORAGE_OWNER_ANCHOR_EXPAND_CAP:-16}")
    cmd+=(--storage-owner-anchor-remote-rescue-cap "${STORAGE_OWNER_ANCHOR_REMOTE_RESCUE_CAP:-4}")
    cmd+=(--storage-owner-anchor-audit-rate "${STORAGE_OWNER_ANCHOR_AUDIT_RATE:-256}")
    cmd+=(--storage-owner-anchor-min-overlap "${STORAGE_OWNER_ANCHOR_MIN_OVERLAP:-0.5}")
  fi
  cmd+=(--storage-owner-reverse-mode "${STORAGE_OWNER_REVERSE_MODE:-async}")
  cmd+=(--storage-owner-reverse-queue-depth "${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}")
  cmd+=(--storage-owner-reverse-flush-us "${STORAGE_OWNER_REVERSE_FLUSH_US:-200}")
  cmd+=(--storage-owner-reverse-coalesce-max "${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}")
fi

printf '[memory-node-%s] command:' "$NODE_ID"; printf ' %q' "${cmd[@]}"; echo
nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "memory node $NODE_ID started: pid $(cat "$PID_FILE"), log $LOG_FILE"
