#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"

DATASET_DIR="${DATASET_DIR:-/data/xjs/datasets/sift1b}"
WORK_DIR="${WORK_DIR:-/data/xjs/index/dvstor_sift100m}"
CONVERTED_DIR="${CONVERTED_DIR:-$WORK_DIR/converted}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/index}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"

SHARDS="${SHARDS:-5}"
# PARTITION_STRATEGY="${PARTITION_STRATEGY:-bfs}"
# PARTITION_STRATEGY="${PARTITION_STRATEGY:-metis}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-balanced}"
R="${R:-96}"
BUILD_BEAM="${BUILD_BEAM:-128}"
SEARCH_BEAM="${SEARCH_BEAM:-100}"
ALPHA="${ALPHA:-1.2}"
K="${K:-10}"
DIM="${DIM:-128}"
VECTOR_DATA_TYPE="${VECTOR_DATA_TYPE:-uint8}"
STORAGE_FORMAT="${STORAGE_FORMAT:-vamana_compact_v1}"
BUILD_THREADS="${BUILD_THREADS:-112}"
SERVICE_THREADS="${SERVICE_THREADS:-16}"
COROUTINES="${COROUTINES:-4}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
GPU_DEVICE="${GPU_DEVICE:-1}"
MAX_VECTORS="${MAX_VECTORS:-100000000}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
GROUNDTRUTH_LABEL="${GROUNDTRUTH_LABEL:-100M}"
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
HOSTS="${HOSTS:-192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202}"
IB_DEVICE="${IB_DEVICE:-}"
IB_PORT="${IB_PORT:-1}"
MAX_SEND_WRS="${MAX_SEND_WRS:-4096}"
MAX_RECEIVE_WRS="${MAX_RECEIVE_WRS:-4096}"
MAX_POLL_CQES="${MAX_POLL_CQES:-64}"

PROFILE="${PROFILE:-baseline}"
INDEX_PREFIX="${INDEX_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}}"

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
insert_bin() { echo "${INSERT_FILE:-$CONVERTED_DIR/insert_test.u8bin}"; }

metadata_file() { echo "${INDEX_PREFIX}.meta.json"; }

validate_index_metadata() {
  local metadata
  metadata="$(metadata_file)"
  if [[ ! -f "$metadata" ]]; then
    echo "missing index metadata: $metadata" >&2
    return 1
  fi

  python3 - "$metadata" "$INDEX_PREFIX" "$R" "$BUILD_BEAM" "$DIM" "$MAX_VECTORS" "$SHARDS" "$VECTOR_DATA_TYPE" <<'PY_VALIDATE'
import json
import sys

path, expected_prefix, expected_r, expected_beam, expected_dim, expected_vectors, expected_shards, expected_dtype = sys.argv[1:]
with open(path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

expected = {
    'output_prefix': expected_prefix,
    'R': int(expected_r),
    'beam_width_construction': int(expected_beam),
    'dim': int(expected_dim),
    'num_vectors': int(expected_vectors),
    'num_memory_nodes': int(expected_shards),
    'vector_data_type': expected_dtype,
}
errors = []
for key, value in expected.items():
    if metadata.get(key) != value:
        errors.append(f"{key}: metadata={metadata.get(key)!r}, expected={value!r}")
if metadata.get('offline_builder_version', 0) < 2:
    errors.append('offline_builder_version is missing or older than 2; rebuild with the fixed per-node random graph initializer')
if metadata.get('random_graph_seed_scope') != 'per_node':
    errors.append('random_graph_seed_scope is not per_node; this index may contain duplicated initial neighbor lists')

if errors:
    print(f"incompatible or unsafe index metadata: {path}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)
PY_VALIDATE

  local node_id shard
  for ((node_id = 1; node_id <= SHARDS; ++node_id)); do
    shard="$(shard_file "$node_id")"
    if [[ ! -s "$shard" ]]; then
      echo "missing or empty shard file: $shard" >&2
      return 1
    fi
  done
}

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
  local insert_execution="${INSERT_EXECUTION:-compute}"
  local insert_workers query_workers insert_coroutines query_coroutines
  endpoints="$(server_endpoints)"

  if [[ "$insert_execution" == "storage_owner" ]]; then
    # Inserts execute on memory nodes; all compute-side workers remain available for queries.
    insert_workers=0
    query_workers="$SERVICE_THREADS"
    insert_coroutines=0
    query_coroutines="${QUERY_COROUTINES:-$COROUTINES}"
  else
    if [[ -z "${INSERT_WORKERS+x}" && -z "${QUERY_WORKERS+x}" ]]; then
      insert_workers=$((SERVICE_THREADS / 2))
      query_workers=$((SERVICE_THREADS - insert_workers))
    elif [[ -z "${INSERT_WORKERS+x}" ]]; then
      query_workers="$QUERY_WORKERS"
      insert_workers=$((SERVICE_THREADS - query_workers))
    elif [[ -z "${QUERY_WORKERS+x}" ]]; then
      insert_workers="$INSERT_WORKERS"
      query_workers=$((SERVICE_THREADS - insert_workers))
    else
      insert_workers="$INSERT_WORKERS"
      query_workers="$QUERY_WORKERS"
    fi
    insert_coroutines="${INSERT_COROUTINES:-$COROUTINES}"
    query_coroutines="${QUERY_COROUTINES:-$COROUTINES}"
  fi

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
    echo "insert-workers = $insert_workers"
    echo "query-workers = $query_workers"
    echo "insert-coroutines = $insert_coroutines"
    echo "query-coroutines = $query_coroutines"
    echo "label = sift100m_${PROFILE_NAME:-$PROFILE}"
    if [[ "${GPUDIRECT_RDMA:-0}" == "1" ]]; then echo "gpudirect-rdma = true"; fi
    if [[ -n "${ENABLE_BREAKDOWN:-}" ]]; then
      if [[ "${ENABLE_BREAKDOWN}" == "1" || "${ENABLE_BREAKDOWN}" == "true" ]]; then
        echo "enable-breakdown = true"
      else
        echo "enable-breakdown = false"
      fi
    fi
    if [[ "${OBSERVE_DEVICE_UTILIZATION:-0}" == "1" || "${OBSERVE_DEVICE_UTILIZATION:-0}" == "true" ]]; then
      echo "observe-device-utilization = true"
    fi
    if [[ -n "${EXPANSION_BATCH:-}" ]]; then echo "expansion-batch = ${EXPANSION_BATCH}"; fi
    if [[ "${CREDIT_AWARE_EXPANSION:-0}" == "1" || "${CREDIT_AWARE_EXPANSION:-0}" == "true" ]]; then
      echo "credit-aware-expansion = true"
      if [[ -n "${CREDIT_AWARE_MIN_K:-}" ]]; then echo "credit-aware-min-k = ${CREDIT_AWARE_MIN_K}"; fi
      if [[ -n "${CREDIT_AWARE_MAX_K:-}" ]]; then echo "credit-aware-max-k = ${CREDIT_AWARE_MAX_K}"; fi
      if [[ -n "${CREDIT_AWARE_TARGET_CANDIDATES:-}" ]]; then
        echo "credit-aware-target-candidates = ${CREDIT_AWARE_TARGET_CANDIDATES}"
      fi
      if [[ -n "${CREDIT_AWARE_MAX_LOOKAHEAD:-}" ]]; then
        echo "credit-aware-max-lookahead = ${CREDIT_AWARE_MAX_LOOKAHEAD}"
      fi
      if [[ "${CREDIT_AWARE_COST_GUARD:-0}" == "1" || "${CREDIT_AWARE_COST_GUARD:-0}" == "true" ]]; then
        echo "credit-aware-cost-guard = true"
        if [[ -n "${CREDIT_AWARE_COST_MAX_EXTRA_RATIO:-}" ]]; then
          echo "credit-aware-cost-max-extra-ratio = ${CREDIT_AWARE_COST_MAX_EXTRA_RATIO}"
        fi
        if [[ -n "${CREDIT_AWARE_COST_PROBE_ROUNDS:-}" ]]; then
          echo "credit-aware-cost-probe-rounds = ${CREDIT_AWARE_COST_PROBE_ROUNDS}"
        fi
      fi
    fi
    if [[ -n "${RDMA_QP_POOL_SIZE:-}" ]]; then echo "rdma-qp-pool-size = ${RDMA_QP_POOL_SIZE}"; fi
    if [[ -n "${RDMA_READ_BATCH_MODE:-}" ]]; then echo "rdma-read-batch-mode = ${RDMA_READ_BATCH_MODE}"; fi
    if [[ -n "${RDMA_READ_CHAIN_SIZE:-}" ]]; then echo "rdma-read-chain-size = ${RDMA_READ_CHAIN_SIZE}"; fi
    if [[ -n "${RDMA_READ_MAX_INFLIGHT_WRS:-}" ]]; then echo "rdma-read-max-inflight-wrs = ${RDMA_READ_MAX_INFLIGHT_WRS}"; fi
    if [[ -n "${QUERY_BATCH_SIZE:-}" ]]; then echo "query-batch-size = ${QUERY_BATCH_SIZE}"; fi
    if [[ "${USE_RABITQ:-0}" == "1" ]]; then echo "use-rabitq = true"; fi
    if [[ "${USE_RABITQ:-0}" == "1" ]]; then
      echo "rabitq-gate-width = ${RABITQ_GATE_WIDTH:-18}"
      echo "rabitq-gate-max-width = ${RABITQ_GATE_MAX_WIDTH:-36}"
      echo "rabitq-gate-margin = ${RABITQ_GATE_MARGIN:-0.08}"
      echo "rabitq-cache-max-ratio = ${RABITQ_CACHE_MAX_RATIO:-0.10}"
      echo "rabitq-dynamic-budget-mb = ${RABITQ_DYNAMIC_BUDGET_MB:-64}"
      echo "rabitq-coalesce-target = ${RABITQ_COALESCE_TARGET:-64}"
      echo "rabitq-coalesce-min = ${RABITQ_COALESCE_MIN:-32}"
      echo "rabitq-coalesce-wait-us = ${RABITQ_COALESCE_WAIT_US:-6}"
      echo "rabitq-warmup-exact-expansions = ${RABITQ_WARMUP_EXACT_EXPANSIONS:-6}"
      echo "rabitq-audit-period = ${RABITQ_AUDIT_PERIOD:-12}"
      if [[ "${RABITQ_STRICT_RECALL:-1}" == "1" ]]; then
        echo "rabitq-strict-recall = true"
      else
        echo "rabitq-strict-recall = false"
      fi
    fi
    echo "insert-execution = $insert_execution"
    if [[ "$insert_execution" == "storage_owner" ]]; then
      echo "storage-peers = $endpoints"
      echo "storage-owner-batch-max = ${STORAGE_OWNER_BATCH_MAX:-32}"
      echo "storage-owner-batch-wait-us = ${STORAGE_OWNER_BATCH_WAIT_US:-100}"
      echo "storage-owner-peer-rdma-tokens = ${STORAGE_OWNER_PEER_RDMA_TOKENS:-8}"
      echo "storage-owner-rpc-depth = ${STORAGE_OWNER_RPC_DEPTH:-16}"
      echo "storage-owner-rpc-timeout-ms = ${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
      echo "storage-owner-construction-beam-width = ${STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH:-$BUILD_BEAM}"
      echo "storage-owner-search-snapshot-batch = ${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-64}"
      echo "storage-owner-prune-max-candidates = ${STORAGE_OWNER_PRUNE_MAX_CANDIDATES:-128}"
      local update_mode="${STORAGE_OWNER_UPDATE_MODE:-exact}"
      if [[ "$update_mode" != "exact" ]]; then
        echo "storage-owner-update-mode = ${update_mode}"
      fi
      if [[ "$update_mode" == "local_stitch" ]]; then
        echo "storage-owner-anchor-hints = ${STORAGE_OWNER_ANCHOR_HINTS:-4}"
        echo "storage-owner-anchor-beam-width = ${STORAGE_OWNER_ANCHOR_BEAM_WIDTH:-64}"
        echo "storage-owner-anchor-expand-cap = ${STORAGE_OWNER_ANCHOR_EXPAND_CAP:-16}"
        echo "storage-owner-anchor-remote-rescue-cap = ${STORAGE_OWNER_ANCHOR_REMOTE_RESCUE_CAP:-4}"
      fi
      echo "storage-owner-maintenance-mode = ${STORAGE_OWNER_MAINTENANCE_MODE:-off}"
      echo "storage-owner-maintenance-workers = ${STORAGE_OWNER_MAINTENANCE_WORKERS:-0}"
      echo "storage-owner-reverse-mode = ${STORAGE_OWNER_REVERSE_MODE:-async}"
      echo "storage-owner-reverse-queue-depth = ${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
      echo "storage-owner-reverse-flush-us = ${STORAGE_OWNER_REVERSE_FLUSH_US:-200}"
      echo "storage-owner-reverse-coalesce-max = ${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}"
    fi
  } > "$output"
}
