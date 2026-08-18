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
validate_vector_id_namespace_size
resolve_mn_memory_gb
SHARD_FILE="$(shard_file "$NODE_ID")"

PID_FILE="$PID_DIR/memory_node_${NODE_ID}.pid"
LOG_FILE="$LOG_DIR/memory_node_${NODE_ID}_${PROFILE}.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "memory node $NODE_ID already running: pid $(cat "$PID_FILE")" >&2
  exit 1
fi

read -r -a SERVER_ARGS <<< "$(server_endpoints)"
read -r -a RDMA_ARGS <<< "$(common_rdma_args)"

# Several logical shards may intentionally share one storage host. Each
# process otherwise starts CoreAssignment at the same CPU and all shards pile
# onto an identical core set. Derive an internal local rank from equal endpoint
# hosts; a host that owns only one shard keeps the complete machine.
read -r -a CONFIGURED_HOSTS <<< "$HOSTS"
NODE_HOST="${CONFIGURED_HOSTS[NODE_ID - 1]}"
LOCAL_PROCESS_RANK=0
LOCAL_PROCESS_COUNT=0
for ((index = 0; index < SHARDS; ++index)); do
  if [[ "${CONFIGURED_HOSTS[index]}" == "$NODE_HOST" ]]; then
    if ((index < NODE_ID - 1)); then
      ((LOCAL_PROCESS_RANK += 1))
    fi
    ((LOCAL_PROCESS_COUNT += 1))
  fi
done
if ((LOCAL_PROCESS_COUNT == 0)); then
  echo "cannot derive the local storage process CPU partition" >&2
  exit 1
fi

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
  --vector-id-namespace-size "$VECTOR_ID_NAMESPACE_SIZE"
  --R "$R"
  --beam-width-construction "$BUILD_BEAM"
  --alpha "$ALPHA"
  --k "$K"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --mn-memory "$MN_MEMORY_GB"
  --storage-id "$((NODE_ID - 1))"
  --storage-peers "${SERVER_ARGS[@]}"
  --storage-owner-batch-max "${STORAGE_OWNER_BATCH_MAX:-32}"
  --storage-owner-batch-max-wait-us "${STORAGE_OWNER_BATCH_MAX_WAIT_US:-10000}"
  --storage-owner-stage2-batch-max-wait-us "${STORAGE_OWNER_STAGE2_BATCH_MAX_WAIT_US:-50}"
  --storage-owner-defer-stage1-prune "${STORAGE_OWNER_DEFER_STAGE1_PRUNE:-false}"
  --storage-owner-stage2-score-many "${STORAGE_OWNER_STAGE2_SCORE_MANY:-false}"
  --storage-owner-peer-qps-per-peer "${STORAGE_OWNER_PEER_QPS_PER_PEER:-8}"
  --storage-owner-peer-rdma-tokens "${STORAGE_OWNER_PEER_RDMA_TOKENS:-16}"
  --storage-owner-rpc-depth "${STORAGE_OWNER_RPC_DEPTH:-16}"
  --storage-owner-rpc-timeout-ms "${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
  --storage-owner-search-snapshot-batch "${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-256}"
  --storage-owner-maintenance-workers "${STORAGE_OWNER_MAINTENANCE_WORKERS:-8}"
  --storage-owner-reverse-queue-depth "${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
  --storage-owner-reverse-coalesce-max "${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}")

printf '[memory-node-%s] command:' "$NODE_ID"; printf ' %q' "${cmd[@]}"; echo
echo "[memory-node-$NODE_ID] local CPU partition rank=$LOCAL_PROCESS_RANK/$LOCAL_PROCESS_COUNT host=$NODE_HOST"
DVSTOR_LOCAL_PROCESS_RANK="$LOCAL_PROCESS_RANK" \
DVSTOR_LOCAL_PROCESS_COUNT="$LOCAL_PROCESS_COUNT" \
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "memory node $NODE_ID started: pid $(cat "$PID_FILE"), log $LOG_FILE"
