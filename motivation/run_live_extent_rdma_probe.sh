#!/usr/bin/env bash
set -euo pipefail

LIVE_EXTENT_MOTIVATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE_EXTENT_PROJECT_DIR="$(cd "$LIVE_EXTENT_MOTIVATION_DIR/.." && pwd)"
if [[ -n "${LIVE_EXTENT_CONFIG:-}" ]]; then
  if [[ ! -f "$LIVE_EXTENT_CONFIG" ]]; then
    echo "live-extent config does not exist: $LIVE_EXTENT_CONFIG" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$LIVE_EXTENT_CONFIG"
fi
source "$LIVE_EXTENT_PROJECT_DIR/experiment/common.sh"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
load_experiment_profile "$PROFILE"
resolve_mn_memory_gb

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  ensure_built dvstor_gpunetio_loopback_probe
fi

if [[ "${SKIP_STORAGE_PID_CHECK:-0}" != "1" ]]; then
  for ((node = 1; node <= SHARDS; ++node)); do
    pid_file="$PID_DIR/memory_node_${node}.pid"
    if [[ ! -f "$pid_file" ]] ||
       ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "memory node $node has no live local PID; for remotely managed " \
           "storage nodes set SKIP_STORAGE_PID_CHECK=1" >&2
      exit 1
    fi
  done
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${LIVE_EXTENT_RESULT_ROOT:-$LIVE_EXTENT_MOTIVATION_DIR/results/live_extent_rdma/$timestamp}"
mkdir -p "$RESULT_ROOT"
LOG_FILE="$RESULT_ROOT/probe.log"
CSV_FILE="$RESULT_ROOT/live_extent_rdma.csv"
HARDWARE_FILE="$RESULT_ROOT/hardware.txt"
COMMAND_FILE="$RESULT_ROOT/command.txt"

read -r -a SERVER_ARGS <<< "$(server_endpoints)"
read -r -a RDMA_ARGS <<< "$(common_rdma_args)"

cmd=("$BUILD_DIR/dvstor_gpunetio_loopback_probe"
  --servers "${SERVER_ARGS[@]}"
  --initiator
  --num-clients 1
  --port 2234
  "${RDMA_ARGS[@]}"
  --index-prefix "$INDEX_PREFIX"
  --threads "$SERVICE_THREADS"
  --dim "$DIM"
  --max-vectors "$MAX_VECTORS"
  --vector-id-namespace-size "$VECTOR_ID_NAMESPACE_SIZE"
  --k "$K"
  --R "$R"
  --beam-width-construction "$BUILD_BEAM"
  --alpha "$ALPHA"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --mn-memory "$MN_MEMORY_GB"
  --gpu-device "$GPU_DEVICE"
  --enable-updates false
  --gpu-query-slots "${GPU_QUERY_SLOTS:-256}"
  --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB:-40}"
  --gpu-memory-reserve-gb "${GPU_MEMORY_RESERVE_GB:-4}"
  --gpu-bootstrap-window-mb "${GPU_BOOTSTRAP_WINDOW_MB:-64}"
  --gpu-bootstrap-windows "${GPU_BOOTSTRAP_WINDOWS:-4}"
  --gpu-graph-prefetch-depth "${GPU_GRAPH_PREFETCH_DEPTH:-16}"
  --gpu-query-expansion-policy fixed
  --gpu-query-beam-merge-policy "${GPU_QUERY_BEAM_MERGE_POLICY:-stable-run}"
  --query-rdma-trace-mode off
  --gpu-traversal-beam-width "${GPU_TRAVERSAL_BEAM_WIDTH:-128}"
  --gpu-final-rerank-width "${GPU_FINAL_RERANK_WIDTH:-128}"
  --gpu-max-expansions "${GPU_MAX_EXPANSIONS:-384}"
  --gpu-rdma-qps "${GPU_RDMA_QPS:-32}"
  --gpu-direct-timeout-ms "${GPU_DIRECT_TIMEOUT_MS:-250}"
  --gpu-persistent-blocks-per-sm "${GPU_PERSISTENT_BLOCKS_PER_SM:-4}"
  --storage-id 0
  --storage-peers "${SERVER_ARGS[@]}"
  --storage-owner-batch-max "${STORAGE_OWNER_BATCH_MAX:-32}"
  --storage-owner-stage2-batch-max-wait-us \
    "${STORAGE_OWNER_STAGE2_BATCH_MAX_WAIT_US:-50}"
  --storage-owner-peer-qps-per-peer \
    "${STORAGE_OWNER_PEER_QPS_PER_PEER:-8}"
  --storage-owner-peer-rdma-tokens \
    "${STORAGE_OWNER_PEER_RDMA_TOKENS:-16}"
  --storage-owner-rpc-depth "${STORAGE_OWNER_RPC_DEPTH:-16}"
  --storage-owner-rpc-timeout-ms "${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
  --storage-owner-search-snapshot-batch \
    "${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-256}"
  --storage-owner-maintenance-workers \
    "${STORAGE_OWNER_MAINTENANCE_WORKERS:-8}"
  --storage-owner-maintenance-queue-depth \
    "${STORAGE_OWNER_MAINTENANCE_QUEUE_DEPTH:-65536}"
  --storage-owner-reverse-queue-depth \
    "${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
  --storage-owner-reverse-coalesce-max \
    "${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}")

{
  echo "date=$(date --iso-8601=seconds)"
  echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total \
    --format=csv,noheader || true
  ibv_devinfo -d "${IB_DEVICE:-mlx5_0}" | \
    grep -E 'hca_id|fw_ver|node_guid|transport|link_layer|active_speed|active_width' ||
    true
} > "$HARDWARE_FILE" 2>&1

{
  printf 'DVSTOR_GPUNETIO_PAYLOAD_SWEEP=1'
  printf ' DVSTOR_GPUNETIO_PAYLOAD_BYTES=%q' \
    "${LIVE_EXTENT_PAYLOAD_BYTES:-16,80,144,272,400,448,528,832}"
  printf ' DVSTOR_GPUNETIO_PAIRED_BODY_BYTES=%q' \
    "${LIVE_EXTENT_PAIRED_BODY_BYTES:-400,448}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST=%q' \
    "${LIVE_EXTENT_ACTIVE_QPS_LIST:-1,8,32,$((GPU_RDMA_QPS * SHARDS))}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS=%q' \
    "${LIVE_EXTENT_BATCH_READS:-16}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_WARMUP_ITERATIONS=%q' \
    "${LIVE_EXTENT_WARMUP_ITERATIONS:-32}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_ITERATIONS=%q' \
    "${LIVE_EXTENT_ITERATIONS:-512}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_REPEATS=%q' \
    "${LIVE_EXTENT_REPEATS:-3}"
  printf ' DVSTOR_GPUNETIO_PAYLOAD_REMOTE_SPAN_BYTES=%q' \
    "${LIVE_EXTENT_REMOTE_SPAN_BYTES:-67108864}"
  printf ' '
  printf '%q ' "${cmd[@]}"
  echo
} > "$COMMAND_FILE"

echo "This read-only probe consumes the storage nodes' single compute session."
echo "Restart the storage nodes before running another service or probe."

export DVSTOR_GPUNETIO_PAYLOAD_SWEEP=1
export DVSTOR_GPUNETIO_PAYLOAD_BYTES="${LIVE_EXTENT_PAYLOAD_BYTES:-16,80,144,272,400,448,528,832}"
export DVSTOR_GPUNETIO_PAIRED_BODY_BYTES="${LIVE_EXTENT_PAIRED_BODY_BYTES:-400,448}"
export DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST="${LIVE_EXTENT_ACTIVE_QPS_LIST:-1,8,32,$((GPU_RDMA_QPS * SHARDS))}"
export DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS="${LIVE_EXTENT_BATCH_READS:-16}"
export DVSTOR_GPUNETIO_PAYLOAD_WARMUP_ITERATIONS="${LIVE_EXTENT_WARMUP_ITERATIONS:-32}"
export DVSTOR_GPUNETIO_PAYLOAD_ITERATIONS="${LIVE_EXTENT_ITERATIONS:-512}"
export DVSTOR_GPUNETIO_PAYLOAD_REPEATS="${LIVE_EXTENT_REPEATS:-3}"
export DVSTOR_GPUNETIO_PAYLOAD_REMOTE_SPAN_BYTES="${LIVE_EXTENT_REMOTE_SPAN_BYTES:-67108864}"

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"

header="$(sed -n 's/^LIVE_EXTENT_RDMA_HEADER,//p' "$LOG_FILE" | head -n 1)"
if [[ -z "$header" ]]; then
  echo "probe produced no machine-readable header" >&2
  exit 1
fi
{
  echo "$header"
  sed -n 's/^LIVE_EXTENT_RDMA_CSV,//p' "$LOG_FILE"
} > "$CSV_FILE"

echo "live-extent RDMA probe complete"
echo "  CSV:      $CSV_FILE"
echo "  raw log:  $LOG_FILE"
echo "  hardware: $HARDWARE_FILE"
echo "  command:  $COMMAND_FILE"
