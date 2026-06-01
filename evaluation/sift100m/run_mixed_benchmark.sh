#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

RUN_SCRIPT="$PROJECT_DIR/scripts/run_breakdown_test.sh"
GENERATED_CONFIG="${GENERATED_CONFIG:-$SCRIPT_DIR/generated_sift100m_storage_owner_gpucache.ini}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/reports/sift100m}"
LABEL="${LABEL:-sift100m_mixed_$(date +%Y%m%d_%H%M%S)}"
WORKLOAD="${WORKLOAD:-mixed}"
READ_RATIO="${READ_RATIO:-0.9}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
MEASURE_SECONDS="${MEASURE_SECONDS:-60}"

cat > "$GENERATED_CONFIG" <<EOF_CFG
servers = $(join_server_endpoints)
initiator = true
port = $SERVER_PORT_BASE
threads = $THREADS
coroutines = $COROUTINES
dim = $DIM
k = $K
R = $R
beam-width = 32
beam-width-construction = $BEAM_WIDTH
alpha = $ALPHA
rabitq-bits = $RABITQ_BITS
search-mode = $SEARCH_MODE
insert-execution = $INSERT_EXECUTION
storage-owner-cache-mb = $STORAGE_OWNER_CACHE_MB
storage-owner-peer-rdma-tokens = $STORAGE_OWNER_PEER_RDMA_TOKENS
storage-owner-rpc-depth = $STORAGE_OWNER_RPC_DEPTH
storage-owner-construction-beam-width = $STORAGE_OWNER_CONSTRUCTION_BEAM_WIDTH
storage-owner-search-snapshot-batch = $STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH
storage-owner-prune-max-candidates = $STORAGE_OWNER_PRUNE_MAX_CANDIDATES
storage-owner-reverse-mode = $STORAGE_OWNER_REVERSE_MODE
storage-owner-reverse-queue-depth = $STORAGE_OWNER_REVERSE_QUEUE_DEPTH
storage-owner-reverse-flush-us = $STORAGE_OWNER_REVERSE_FLUSH_US
storage-owner-reverse-coalesce-max = $STORAGE_OWNER_REVERSE_COALESCE_MAX
storage-id = 0
storage-peers = $(join_storage_endpoints)
load-index = true
index-prefix = $INDEX_PREFIX
gpu-device = $GPU_DEVICE
max-vectors = $MAX_VECTORS
cn-memory = $CN_MEMORY
mn-memory = $MN_MEMORY
query-workers = $QUERY_WORKERS
query-coroutines = $QUERY_COROUTINES
cache = $CACHE
cache-ratio = $CACHE_RATIO
disable-thread-pinning = $DISABLE_THREAD_PINNING
neighbor-cache-mb = $NEIGHBOR_CACHE_MB
neighbor-cache-invalidation-ms = $NEIGHBOR_CACHE_INVALIDATION_MS
neighbor-cache-invalidation-inserts = $NEIGHBOR_CACHE_INVALIDATION_INSERTS
gpudirect-rdma = $GPUDIRECT_RDMA
gpu-rabitq-cache-mb = $GPU_RABITQ_CACHE_MB
insert-start-id = $INSERT_START_ID
EOF_CFG

if [[ ! -f "$QUERY_FILE" ]]; then
  echo "error: query file not found: $QUERY_FILE" >&2
  exit 1
fi

args=(
  --service-config "$GENERATED_CONFIG"
  --workload "$WORKLOAD"
  --read-ratio "$READ_RATIO"
  --client-threads "$CLIENT_THREADS"
  --warmup-seconds "$WARMUP_SECONDS"
  --measure-seconds "$MEASURE_SECONDS"
  --query-file "$QUERY_FILE"
  --report-dir "$REPORT_DIR"
  --label "$LABEL"
  --insert-start-id "$INSERT_START_ID"
)

exec "$RUN_SCRIPT" "${args[@]}" "$@"
