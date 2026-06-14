#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"

DATASET_DIR="${DATASET_DIR:-/data/xjs/datasets/sift1b}"
WORK_DIR="${WORK_DIR:-/data/xjs/index/dvstor_sift1b}"
CONVERTED_DIR="${CONVERTED_DIR:-$WORK_DIR/converted}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/index}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"

SHARDS="${SHARDS:-5}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-bfs}"
R="${R:-64}"
BUILD_BEAM="${BUILD_BEAM:-200}"
SEARCH_BEAM="${SEARCH_BEAM:-128}"
ALPHA="${ALPHA:-1.2}"
K="${K:-10}"
DIM="${DIM:-128}"
VECTOR_DATA_TYPE="${VECTOR_DATA_TYPE:-uint8}"
STORAGE_FORMAT="${STORAGE_FORMAT:-vamana_compact_v1}"
BUILD_THREADS="${BUILD_THREADS:-32}"
SERVICE_THREADS="${SERVICE_THREADS:-16}"
COROUTINES="${COROUTINES:-4}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
GPU_DEVICE="${GPU_DEVICE:-1}"
MAX_VECTORS="${MAX_VECTORS:-1000000000}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
GROUNDTRUTH_LABEL="${GROUNDTRUTH_LABEL:-1000M}"
GROUNDTRUTH_TOPK="${GROUNDTRUTH_TOPK:-10}"

estimate_node_bytes() {
  local component_size=4
  case "$VECTOR_DATA_TYPE" in
    uint8|int8) component_size=1 ;;
    float32|auto) component_size=4 ;;
  esac
  if [[ "$STORAGE_FORMAT" == "vamana_compact_v1" ]]; then
    local fixed_bytes=$((((16 + DIM * component_size + 15) / 16) * 16))
    local graph_bytes=$((((8 + R * 5 + 7) / 8) * 8))
    echo $((fixed_bytes + graph_bytes))
  else
    echo $((16 + DIM * component_size + R * 8))
  fi
}

estimate_mn_memory_gb() {
  local node_bytes vectors_per_shard bytes gib
  node_bytes="$(estimate_node_bytes)"
  vectors_per_shard=$(((MAX_VECTORS + SHARDS - 1) / SHARDS))
  bytes=$((vectors_per_shard * node_bytes))
  # 20% slack for partition imbalance, headers, allocator alignment, and online inserts, plus 4GB floor slack.
  gib=$(((bytes * 12 / 10 + 4 * 1024 * 1024 * 1024 + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024)))
  if (( gib < 8 )); then gib=8; fi
  echo "$gib"
}

if [[ -z "${CN_MEMORY_GB+x}" ]]; then
  CN_MEMORY_GB=16
  CN_MEMORY_GB_WAS_DEFAULT=1
else
  CN_MEMORY_GB_WAS_DEFAULT=0
fi
MN_MEMORY_GB="${MN_MEMORY_GB:-$(estimate_mn_memory_gb)}"

BASE_PORT="${BASE_PORT:-1234}"
HOSTS="${HOSTS:-127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1}"
IB_DEVICE="${IB_DEVICE:-}"
IB_PORT="${IB_PORT:-1}"
MAX_SEND_WRS="${MAX_SEND_WRS:-4096}"
MAX_RECEIVE_WRS="${MAX_RECEIVE_WRS:-4096}"
MAX_POLL_CQES="${MAX_POLL_CQES:-64}"

PROFILE="${PROFILE:-baseline}"
INDEX_PREFIX="${INDEX_PREFIX:-$INDEX_DIR/sift1b_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}}"

mkdir -p "$CONVERTED_DIR" "$INDEX_DIR" "$REPORT_DIR" "$LOG_DIR" "$PID_DIR"

base_suffix() {
  if [[ "$MAX_VECTORS" == "0" || "$MAX_VECTORS" == "1000000000" ]]; then
    echo ""
  else
    echo "_${MAX_VECTORS}"
  fi
}

query_suffix() {
  if [[ "$MAX_QUERIES" == "0" || "$MAX_QUERIES" == "10000" ]]; then
    echo ""
  else
    echo "_${MAX_QUERIES}"
  fi
}

base_bin() { echo "$CONVERTED_DIR/base$(base_suffix).u8bin"; }
query_bin() { echo "$CONVERTED_DIR/query$(query_suffix).u8bin"; }
groundtruth_bin() { echo "$CONVERTED_DIR/groundtruth_${GROUNDTRUTH_LABEL}.bin"; }

server_endpoints() {
  local idx=0
  local endpoints=()
  for host in $HOSTS; do
    if (( idx >= SHARDS )); then break; fi
    endpoints+=("${host}:$((BASE_PORT + idx))")
    idx=$((idx + 1))
  done
  if (( ${#endpoints[@]} != SHARDS )); then
    echo "HOSTS must contain $SHARDS entries; got ${#endpoints[@]}" >&2
    return 1
  fi
  printf '%s ' "${endpoints[@]}"
}

shard_file() {
  local node_id="$1"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.dat"
}

common_rdma_args() {
  local args=(--ib-port "$IB_PORT" --max-send-wrs "$MAX_SEND_WRS" --max-receive-wrs "$MAX_RECEIVE_WRS" --max-poll-cqes "$MAX_POLL_CQES")
  if [[ -n "$IB_DEVICE" ]]; then
    args+=(--ib-device "$IB_DEVICE")
  fi
  printf '%q ' "${args[@]}"
}

ensure_built() {
  cmake --build "$BUILD_DIR" -j --target "$@"
}


write_service_config() {
  local output="$1"
  local endpoints
  endpoints="$(server_endpoints)"
  {
    echo "servers = $endpoints"
    echo "initiator = true"
    echo "num-clients = 1"
    echo "port = 2234"
    echo "ib-port = $IB_PORT"
    if [[ -n "$IB_DEVICE" ]]; then echo "ib-device = $IB_DEVICE"; fi
    echo "max-send-wrs = $MAX_SEND_WRS"
    echo "max-receive-wrs = $MAX_RECEIVE_WRS"
    echo "max-poll-cqes = $MAX_POLL_CQES"
    echo "data-path = $(base_bin)"
    echo "index-prefix = $INDEX_PREFIX"
    echo "load-index = true"
    echo "no-recall = true"
    echo "vector-data-type = $VECTOR_DATA_TYPE"
    echo "dim = $DIM"
    echo "max-vectors = $MAX_VECTORS"
    echo "threads = $SERVICE_THREADS"
    echo "coroutines = $COROUTINES"
    echo "R = $R"
    echo "beam-width = $SEARCH_BEAM"
    echo "beam-width-construction = $BUILD_BEAM"
    echo "alpha = $ALPHA"
    echo "k = $K"
    echo "gpu-device = $GPU_DEVICE"
    echo "cn-memory = $CN_MEMORY_GB"
    echo "mn-memory = $MN_MEMORY_GB"
    echo "insert-workers = ${INSERT_WORKERS:-4}"
    echo "query-workers = ${QUERY_WORKERS:-12}"
    echo "insert-coroutines = ${INSERT_COROUTINES:-2}"
    echo "query-coroutines = ${QUERY_COROUTINES:-4}"
    echo "label = sift1b_${PROFILE_NAME:-$PROFILE}"
    if [[ "${GPUDIRECT_RDMA:-0}" == "1" ]]; then echo "gpudirect-rdma = true"; fi
    echo "insert-execution = ${INSERT_EXECUTION:-compute}"
    if [[ "${INSERT_EXECUTION:-compute}" == "storage_owner" ]]; then
      echo "storage-peers = $endpoints"
      echo "storage-owner-batch-max = ${STORAGE_OWNER_BATCH_MAX:-32}"
      echo "storage-owner-batch-wait-us = ${STORAGE_OWNER_BATCH_WAIT_US:-100}"
      echo "storage-owner-peer-rdma-tokens = ${STORAGE_OWNER_PEER_RDMA_TOKENS:-8}"
      echo "storage-owner-rpc-depth = ${STORAGE_OWNER_RPC_DEPTH:-16}"
      echo "storage-owner-rpc-timeout-ms = ${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
      echo "storage-owner-handoff-queue-depth = ${STORAGE_OWNER_HANDOFF_QUEUE_DEPTH:-0}"
      echo "storage-owner-construction-beam-width = ${STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH:-$BUILD_BEAM}"
      echo "storage-owner-search-snapshot-batch = ${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-64}"
      echo "storage-owner-prune-max-candidates = ${STORAGE_OWNER_PRUNE_MAX_CANDIDATES:-128}"
      echo "storage-owner-reverse-mode = ${STORAGE_OWNER_REVERSE_MODE:-async}"
      echo "storage-owner-reverse-queue-depth = ${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
      echo "storage-owner-reverse-flush-us = ${STORAGE_OWNER_REVERSE_FLUSH_US:-200}"
      echo "storage-owner-reverse-coalesce-max = ${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}"
      if [[ -n "${STORAGE_OWNER_SEARCH_MODE:-}" ]]; then
        echo "storage-owner-search-mode = ${STORAGE_OWNER_SEARCH_MODE}"
      fi
      if [[ -n "${STORAGE_OWNER_QDI_LOCAL_BEAM:-}" ]]; then
        echo "storage-owner-qdi-local-beam = ${STORAGE_OWNER_QDI_LOCAL_BEAM}"
      fi
      if [[ -n "${STORAGE_OWNER_QDI_RETURN_CANDIDATES:-}" ]]; then
        echo "storage-owner-qdi-return-candidates = ${STORAGE_OWNER_QDI_RETURN_CANDIDATES}"
      fi
      if [[ -n "${STORAGE_OWNER_QDI_EXACT_CANDIDATES:-}" ]]; then
        echo "storage-owner-qdi-exact-candidates = ${STORAGE_OWNER_QDI_EXACT_CANDIDATES}"
      fi
      if [[ -n "${STORAGE_OWNER_QDI_ENTRY_POINTS:-}" ]]; then
        echo "storage-owner-qdi-entry-points = ${STORAGE_OWNER_QDI_ENTRY_POINTS}"
      fi
      if [[ "${STORAGE_OWNER_TRANSITIVE_SEARCH:-0}" == "1" ]]; then
        echo "storage-owner-transitive-search = true"
      fi
    fi
  } > "$output"
}
