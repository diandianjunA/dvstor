#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${BUILD_DIR+x}" ]]; then
  DVSTOR_BUILD_DIR_EXPLICIT=1
else
  DVSTOR_BUILD_DIR_EXPLICIT=0
fi
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"

DATASET_DIR="${DATASET_DIR:-/data/xjs/datasets/sift1b}"
WORK_DIR="${WORK_DIR:-/data/xjs/index/dvstor_sift100m}"
CONVERTED_DIR="${CONVERTED_DIR:-$WORK_DIR/converted}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/index}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"

SHARDS="${SHARDS:-5}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-metis}"
R="${R:-96}"
BUILD_BEAM="${BUILD_BEAM:-128}"
ALPHA="${ALPHA:-1.2}"
K="${K:-10}"
DIM="${DIM:-128}"
VECTOR_DATA_TYPE="${VECTOR_DATA_TYPE:-uint8}"
BUILD_THREADS="${BUILD_THREADS:-112}"
SERVICE_THREADS="${SERVICE_THREADS:-64}"
CLIENT_THREADS="${CLIENT_THREADS:-64}"
GPU_DEVICE="${GPU_DEVICE:-1}"
MAX_VECTORS="${MAX_VECTORS:-100000000}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
GROUNDTRUTH_LABEL="${GROUNDTRUTH_LABEL:-100M}"
GROUNDTRUTH_TOPK="${GROUNDTRUTH_TOPK:-10}"

BASE_PORT="${BASE_PORT:-1234}"
HOSTS="${HOSTS:-192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202}"
IB_DEVICE="${IB_DEVICE:-}"
IB_PORT="${IB_PORT:-1}"
MAX_SEND_WRS="${MAX_SEND_WRS:-4096}"
MAX_RECEIVE_WRS="${MAX_RECEIVE_WRS:-4096}"
MAX_POLL_CQES="${MAX_POLL_CQES:-64}"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
INDEX_PREFIX="${INDEX_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pq16}"

estimate_node_bytes() {
  local component_size=4
  case "$VECTOR_DATA_TYPE" in
    uint8|int8) component_size=1 ;;
    float32|auto) component_size=4 ;;
    *) echo "unsupported VECTOR_DATA_TYPE=$VECTOR_DATA_TYPE" >&2; return 1 ;;
  esac
  local fixed_bytes=$((((16 + DIM * component_size + 15) / 16) * 16))
  local graph_bytes=$((((8 + R * 5 + 7) / 8) * 8))
  echo $((fixed_bytes + graph_bytes + 16))
}

estimate_mn_memory_gb() {
  local node_bytes vectors_per_shard bytes gib
  node_bytes="$(estimate_node_bytes)"
  vectors_per_shard=$(((MAX_VECTORS + SHARDS - 1) / SHARDS))
  bytes=$((vectors_per_shard * node_bytes))
  gib=$(((bytes * 12 / 10 + 4 * 1024 * 1024 * 1024 + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024)))
  ((gib >= 8)) || gib=8
  echo "$gib"
}

MN_MEMORY_GB="${MN_MEMORY_GB:-$(estimate_mn_memory_gb)}"

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
model_file() { echo "${INDEX_PREFIX}.pq16"; }

shard_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.dat"
}

idmap_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.idmap"
}

navigation_code_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.pq16.codes"
}

validate_index_metadata() {
  local role="${1:-compute}"
  local node_id="${2:-0}"
  local metadata
  metadata="$(metadata_file)"
  if [[ ! -s "$metadata" ]]; then
    echo "missing index metadata: $metadata" >&2
    return 1
  fi

  python3 - "$metadata" "$INDEX_PREFIX" "$R" "$BUILD_BEAM" "$DIM" \
    "$MAX_VECTORS" "$SHARDS" "$VECTOR_DATA_TYPE" <<'PY_VALIDATE'
import json
import sys

path, prefix, degree, build_beam, dim, vectors, shards, dtype = sys.argv[1:]
with open(path, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)

expected = {
    'output_prefix': prefix,
    'schema_version': 14,
    'distance': 'l2',
    'node_layout': 'plain',
    'storage_format': 'vamana_compact_v1',
    'navigation_quantizer': 'opq_pq16',
    'navigation_format': 'opq_pq16_graph_v1',
    'navigation_execution': 'gpu_beam_v1',
    'R': int(degree),
    'beam_width_construction': int(build_beam),
    'dim': int(dim),
    'num_vectors': int(vectors),
    'num_memory_nodes': int(shards),
    'vector_data_type': dtype,
    'navigation_code_bytes': 16,
    'pq_subquantizers': 16,
    'pq_bits': 8,
}
errors = [
    f'{key}: metadata={metadata.get(key)!r}, expected={value!r}'
    for key, value in expected.items() if metadata.get(key) != value
]
if not metadata.get('navigation_model_checksum'):
    errors.append('navigation_model_checksum is missing')
for key in (
    'hot_graph_offsets',
    'hot_graph_entry_counts',
    'hot_graph_dynamic_base_offsets',
    'navigation_code_remote_offsets',
    'navigation_code_region_bytes',
):
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != int(shards):
        errors.append(f'{key} must contain one value per storage shard')
if metadata.get('anchor_format') != 'owner_anchor_v1':
    errors.append('owner_anchor_v1 sidecar is required for dynamic updates')
if errors:
    print(f'incompatible GPU index metadata: {path}', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    raise SystemExit(1)
PY_VALIDATE

  if [[ ! -s "${INDEX_PREFIX}.anchors" ]]; then
    echo "missing dynamic-update anchors: ${INDEX_PREFIX}.anchors" >&2
    return 1
  fi

  if [[ "$role" == "compute" ]]; then
    if [[ ! -s "$(model_file)" ]]; then
      echo "missing OPQ/PQ16 model: $(model_file)" >&2
      return 1
    fi
  elif [[ "$role" == "storage" ]]; then
    local first=1 last="$SHARDS"
    if ((node_id > 0)); then first="$node_id"; last="$node_id"; fi
    local current
    for ((current = first; current <= last; ++current)); do
      for artifact in "$(shard_file "$current")" "$(idmap_file "$current")" \
                      "$(navigation_code_file "$current")"; do
        if [[ ! -s "$artifact" ]]; then
          echo "missing storage artifact: $artifact" >&2
          return 1
        fi
      done
    done
  else
    echo "unknown index validation role: $role" >&2
    return 1
  fi
}

server_endpoints() {
  local index=0
  local endpoints=()
  for host in $HOSTS; do
    ((index >= SHARDS)) && break
    endpoints+=("${host}:$((BASE_PORT + index))")
    index=$((index + 1))
  done
  if ((${#endpoints[@]} != SHARDS)); then
    echo "HOSTS must contain $SHARDS entries; got ${#endpoints[@]}" >&2
    return 1
  fi
  printf '%s ' "${endpoints[@]}"
}

common_rdma_args() {
  local args=(--ib-port "$IB_PORT" --max-send-wrs "$MAX_SEND_WRS"
              --max-receive-wrs "$MAX_RECEIVE_WRS" --max-poll-cqes "$MAX_POLL_CQES")
  [[ -z "$IB_DEVICE" ]] || args+=(--ib-device "$IB_DEVICE")
  printf '%q ' "${args[@]}"
}

ensure_built() {
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "build directory is not configured: $BUILD_DIR" >&2
    return 1
  fi
  cmake --build "$BUILD_DIR" -j --target "$@"
}

use_storage_build() {
  if [[ "$DVSTOR_BUILD_DIR_EXPLICIT" == "0" ]]; then
    BUILD_DIR="$PROJECT_DIR/build-storage"
  fi
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] &&
     ! grep -q '^DVSTOR_STORAGE_NODE_ONLY:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"; then
    echo "storage command requires a storage-only build directory: $BUILD_DIR" >&2
    echo "configure a separate directory with -DDVSTOR_STORAGE_NODE_ONLY=ON" >&2
    return 1
  fi
}

write_service_config() {
  local output="${1:?output path is required}"
  local endpoints
  endpoints="$(server_endpoints)"
  validate_index_metadata compute

  {
    echo "servers = $endpoints"
    echo "initiator = true"
    echo "num-clients = 1"
    echo "port = 2234"
    echo "ib-port = $IB_PORT"
    [[ -z "$IB_DEVICE" ]] || echo "ib-device = $IB_DEVICE"
    echo "max-send-wrs = $MAX_SEND_WRS"
    echo "max-receive-wrs = $MAX_RECEIVE_WRS"
    echo "max-poll-cqes = $MAX_POLL_CQES"
    echo "index-prefix = $INDEX_PREFIX"
    echo "threads = $SERVICE_THREADS"
    echo "seed = ${SEED:-1234}"
    echo "vector-data-type = $VECTOR_DATA_TYPE"
    echo "dim = $DIM"
    echo "max-vectors = $MAX_VECTORS"
    echo "R = $R"
    echo "beam-width-construction = $BUILD_BEAM"
    echo "alpha = $ALPHA"
    echo "k = $K"
    echo "mn-memory = $MN_MEMORY_GB"
    echo "gpu-device = $GPU_DEVICE"
    echo "enable-breakdown = ${ENABLE_BREAKDOWN:-true}"
    echo "query-batch-min = ${GPU_QUERY_BATCH_MIN:-8}"
    echo "query-batch-target = ${GPU_QUERY_BATCH_TARGET:-64}"
    echo "query-batch-max = ${GPU_QUERY_BATCH_MAX:-256}"
    echo "query-batch-wait-us = ${GPU_QUERY_BATCH_WAIT_US:-10}"
    echo "gpu-memory-limit-gb = ${GPU_MEMORY_LIMIT_GB:-40}"
    echo "gpu-memory-reserve-gb = ${GPU_MEMORY_RESERVE_GB:-4}"
    echo "gpu-adjacency-cache-mb = ${GPU_ADJACENCY_CACHE_MB:-3072}"
    echo "gpu-adjacency-cache-ways = ${GPU_ADJACENCY_CACHE_WAYS:-4}"
    echo "gpu-exact-cache-mb = ${GPU_EXACT_CACHE_MB:-4096}"
    echo "gpu-exact-cache-ways = ${GPU_EXACT_CACHE_WAYS:-4}"
    echo "gpu-bootstrap-window-mb = ${GPU_BOOTSTRAP_WINDOW_MB:-64}"
    echo "gpu-bootstrap-windows = ${GPU_BOOTSTRAP_WINDOWS:-4}"
    echo "gpu-graph-prefetch-depth = ${GPU_GRAPH_PREFETCH_DEPTH:-8}"
    echo "gpu-graph-cache-ttl-us = ${GPU_GRAPH_CACHE_TTL_US:-0}"
    echo "gpu-traversal-beam-width = ${GPU_TRAVERSAL_BEAM_WIDTH:-128}"
    echo "gpu-final-rerank-width = ${GPU_FINAL_RERANK_WIDTH:-64}"
    echo "gpu-max-expansions = ${GPU_MAX_EXPANSIONS:-384}"
    echo "gpu-entry-seed-count = ${GPU_ENTRY_SEED_COUNT:-32}"
    echo "gpu-delta-anchor-probes = ${GPU_DELTA_ANCHOR_PROBES:-32}"
    echo "gpu-rdma-qps = ${GPU_RDMA_QPS:-8}"
    echo "gpu-persistent-blocks-per-sm = ${GPU_PERSISTENT_BLOCKS_PER_SM:-2}"
    echo "update-visibility-us = ${UPDATE_VISIBILITY_US:-10000}"
    echo "delta-max-ratio = ${DELTA_MAX_RATIO:-0.01}"
    echo "delta-budget-mb = ${DELTA_BUDGET_MB:-2048}"
    echo "merge-period-ms = ${MERGE_PERIOD_MS:-60000}"
    echo "storage-id = 0"
    echo "storage-peers = $endpoints"
    echo "storage-owner-coroutines = ${STORAGE_OWNER_COROUTINES:-4}"
    echo "storage-owner-batch-max = ${STORAGE_OWNER_BATCH_MAX:-32}"
    echo "storage-owner-batch-wait-us = ${STORAGE_OWNER_BATCH_WAIT_US:-100}"
    echo "storage-owner-peer-rdma-tokens = ${STORAGE_OWNER_PEER_RDMA_TOKENS:-8}"
    echo "storage-owner-rpc-depth = ${STORAGE_OWNER_RPC_DEPTH:-16}"
    echo "storage-owner-rpc-timeout-ms = ${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
    echo "storage-owner-construction-beam-width = ${STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH:-$BUILD_BEAM}"
    echo "storage-owner-search-snapshot-batch = ${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-64}"
    echo "storage-owner-prune-max-candidates = ${STORAGE_OWNER_PRUNE_MAX_CANDIDATES:-128}"
    echo "storage-owner-update-mode = ${STORAGE_OWNER_UPDATE_MODE:-local_stitch}"
    echo "storage-owner-anchor-hints = ${STORAGE_OWNER_ANCHOR_HINTS:-4}"
    echo "storage-owner-anchor-beam-width = ${STORAGE_OWNER_ANCHOR_BEAM_WIDTH:-64}"
    echo "storage-owner-anchor-expand-cap = ${STORAGE_OWNER_ANCHOR_EXPAND_CAP:-16}"
    echo "storage-owner-anchor-remote-rescue-cap = ${STORAGE_OWNER_ANCHOR_REMOTE_RESCUE_CAP:-4}"
    echo "storage-owner-local-stitch-sync-fast-path = ${STORAGE_OWNER_LOCAL_STITCH_SYNC_FAST_PATH:-true}"
    echo "storage-owner-maintenance-mode = ${STORAGE_OWNER_MAINTENANCE_MODE:-finalize}"
    echo "storage-owner-maintenance-workers = ${STORAGE_OWNER_MAINTENANCE_WORKERS:-8}"
    echo "storage-owner-reverse-mode = ${STORAGE_OWNER_REVERSE_MODE:-async}"
    echo "storage-owner-reverse-queue-depth = ${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
    echo "storage-owner-reverse-flush-us = ${STORAGE_OWNER_REVERSE_FLUSH_US:-200}"
    echo "storage-owner-reverse-coalesce-max = ${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}"
  } > "$output"
}
