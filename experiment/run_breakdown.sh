#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

WORKLOAD="${WORKLOAD:-mixed}"
BENCHMARK_CLIENT_THREADS="${BENCHMARK_CLIENT_THREADS:-128}"
READ_RATIO="${READ_RATIO:-0.5}"
MIXED_MODE="${MIXED_MODE:-fixed_threads}"
# MIXED_MODE="${MIXED_MODE:-probability}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_K="${RECALL_K:-$K}"
MIN_RECALL="${MIN_RECALL:--1}"
MIN_QUERY_QPS="${MIN_QUERY_QPS:-5000}"
MIN_INSERT_QPS="${MIN_INSERT_QPS:--1}"
MIN_STABILITY_RATIO="${MIN_STABILITY_RATIO:-0.90}"
MIN_WRITE_STABILITY_RATIO="${MIN_WRITE_STABILITY_RATIO:--1}"
TARGET_QUERY_QPS="${TARGET_QUERY_QPS:-0}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-0}"
QUERY_BASELINE_QPS="${QUERY_BASELINE_QPS:--1}"
QUERY_BASELINE_REPORT="${QUERY_BASELINE_REPORT:-}"
MIN_QUERY_BASELINE_RATIO="${MIN_QUERY_BASELINE_RATIO:--1}"
MAX_RECALL_DROP="${MAX_RECALL_DROP:--1}"
MAX_ZERO_COMPLETION_WINDOWS="${MAX_ZERO_COMPLETION_WINDOWS:--1}"
MAX_ZERO_QUERY_WINDOWS="${MAX_ZERO_QUERY_WINDOWS:--1}"
MAX_ZERO_WRITE_WINDOWS="${MAX_ZERO_WRITE_WINDOWS:--1}"
MAX_DRAIN_SECONDS="${MAX_DRAIN_SECONDS:--1}"
MIN_RATE_ATTAINMENT_RATIO="${MIN_RATE_ATTAINMENT_RATIO:--1}"
RECALL_MODE="${RECALL_MODE:-all}"
RECALL_BASE_ID_LIMIT="${RECALL_BASE_ID_LIMIT:-0}"
MAX_GPU_VISIBILITY_MS="${MAX_GPU_VISIBILITY_MS:--1}"
MAX_FINAL_MUTATION_CAPACITY_RESERVED="${MAX_FINAL_MUTATION_CAPACITY_RESERVED:--1}"
MAX_FINAL_DELTA_MUTABLE_ENTRIES="${MAX_FINAL_DELTA_MUTABLE_ENTRIES:--1}"
MAX_LATE_STORAGE_OWNER_RPCS="${MAX_LATE_STORAGE_OWNER_RPCS:--1}"
MAX_STAGE2_P99_MS="${MAX_STAGE2_P99_MS:--1}"
MAX_STAGE2_BACKLOG_SLOPE="${MAX_STAGE2_BACKLOG_SLOPE:--1}"
MAX_STAGE2_REMAINING="${MAX_STAGE2_REMAINING:--1}"
STAGE2_DRAIN_TIMEOUT_SECONDS="${STAGE2_DRAIN_TIMEOUT_SECONDS:-0}"
REQUIRE_COLD_BASELINE="${REQUIRE_COLD_BASELINE:-1}"

configure_insert_acceptance() {
  local client_threads="${1:?client thread count is required}"
  local min_insert_qps="${2:?minimum insert QPS is required}"
  WORKLOAD=insert
  BENCHMARK_CLIENT_THREADS="$client_threads"
  READ_RATIO=0
  MIXED_MODE=fixed_threads
  WARMUP_SECONDS=30
  MEASURE_SECONDS=120
  TARGET_QUERY_QPS=0
  TARGET_WRITE_QPS=0
  WRITE_INSERT_RATIO=1
  WRITE_UPSERT_RATIO=0
  WRITE_DELETE_RATIO=0
  RECALL_K=10
  MIN_RECALL=-1
  MIN_QUERY_QPS=-1
  MIN_INSERT_QPS="$min_insert_qps"
  MIN_STABILITY_RATIO=-1
  MIN_WRITE_STABILITY_RATIO=-1
  QUERY_BASELINE_QPS=-1
  QUERY_BASELINE_REPORT=""
  MIN_QUERY_BASELINE_RATIO=-1
  MAX_RECALL_DROP=-1
  MAX_ZERO_COMPLETION_WINDOWS=0
  MAX_ZERO_QUERY_WINDOWS=0
  MAX_ZERO_WRITE_WINDOWS=0
  MAX_DRAIN_SECONDS=-1
  MIN_RATE_ATTAINMENT_RATIO=-1
  RECALL_MODE=all
  RECALL_BASE_ID_LIMIT=0
  MAX_GPU_VISIBILITY_MS=10
  # ACK intentionally precedes GPU publication; this post-load gate waits for
  # the response executor so visibility_ns_max includes the final batch.
  MAX_FINAL_MUTATION_CAPACITY_RESERVED=0
  MAX_FINAL_DELTA_MUTABLE_ENTRIES=-1
  MAX_LATE_STORAGE_OWNER_RPCS=0
  MAX_STAGE2_P99_MS=-1
  MAX_STAGE2_BACKLOG_SLOPE=-1
  MAX_STAGE2_REMAINING=-1
  STAGE2_DRAIN_TIMEOUT_SECONDS=0
  REQUIRE_COLD_BASELINE=1
}

# UPDATE_ACCEPTANCE_PROFILE is deliberately separate from the experiment PROFILE:
# the former fixes workload/acceptance gates, while the latter selects service config.
UPDATE_ACCEPTANCE_PROFILE="${UPDATE_ACCEPTANCE_PROFILE:-}"
INSERT_ACCEPTANCE_SEGMENT_SIZE=1000000
case "$UPDATE_ACCEPTANCE_PROFILE" in
  "") ;;
  querybaseline)
    WORKLOAD=query
    BENCHMARK_CLIENT_THREADS=64
    READ_RATIO=1
    MIXED_MODE=fixed_threads
    WARMUP_SECONDS=30
    MEASURE_SECONDS=120
    RECALL_QUERIES=0
    MIN_RECALL=-1
    MIN_QUERY_QPS=5000
    MIN_INSERT_QPS=-1
    MIN_STABILITY_RATIO=0.90
    MIN_WRITE_STABILITY_RATIO=-1
    QUERY_BASELINE_QPS=-1
    QUERY_BASELINE_REPORT=""
    MIN_QUERY_BASELINE_RATIO=-1
    MAX_RECALL_DROP=-1
    MAX_ZERO_COMPLETION_WINDOWS=0
    MAX_ZERO_QUERY_WINDOWS=0
    MAX_ZERO_WRITE_WINDOWS=-1
    MAX_DRAIN_SECONDS=-1
    MIN_RATE_ATTAINMENT_RATIO=-1
    RECALL_MODE=all
    RECALL_BASE_ID_LIMIT=0
    MAX_GPU_VISIBILITY_MS=-1
    MAX_FINAL_MUTATION_CAPACITY_RESERVED=-1
    MAX_FINAL_DELTA_MUTABLE_ENTRIES=-1
    MAX_LATE_STORAGE_OWNER_RPCS=-1
    MAX_STAGE2_P99_MS=-1
    MAX_STAGE2_BACKLOG_SLOPE=-1
    MAX_STAGE2_REMAINING=-1
    STAGE2_DRAIN_TIMEOUT_SECONDS=0
    REQUIRE_COLD_BASELINE=1
    ;;
  insert24)
    # The checker uses >=, so the next representable decimal threshold encodes >760.
    configure_insert_acceptance 24 760.000001
    # Keep the two write-only runs disjoint even when they are executed against
    # the same deployed index. Acceptance profiles intentionally override any
    # ambient INSERT_START_ID.
    INSERT_START_ID="$MAX_VECTORS"
    ;;
  insert64)
    configure_insert_acceptance 64 1000
    INSERT_START_ID=$((MAX_VECTORS + INSERT_ACCEPTANCE_SEGMENT_SIZE))
    ;;
  mixed15m)
    WORKLOAD=mixed
    BENCHMARK_CLIENT_THREADS=64
    READ_RATIO=0.5
    MIXED_MODE=rate_limited
    WARMUP_SECONDS=60
    MEASURE_SECONDS=900
    TARGET_QUERY_QPS=5000
    TARGET_WRITE_QPS=1000
    WRITE_INSERT_RATIO=1
    WRITE_UPSERT_RATIO=0
    WRITE_DELETE_RATIO=0
    RECALL_QUERIES=1000
    RECALL_K=10
    MIN_RECALL=0.93
    MIN_QUERY_QPS=5000
    MIN_INSERT_QPS=1000
    MIN_STABILITY_RATIO=0.90
    MIN_WRITE_STABILITY_RATIO=0.90
    MIN_QUERY_BASELINE_RATIO=0.90
    MAX_RECALL_DROP=0.002
    MAX_ZERO_COMPLETION_WINDOWS=0
    MAX_ZERO_QUERY_WINDOWS=0
    MAX_ZERO_WRITE_WINDOWS=0
    MAX_DRAIN_SECONDS=5
    MIN_RATE_ATTAINMENT_RATIO=1
    RECALL_MODE=base_only
    RECALL_BASE_ID_LIMIT="$MAX_VECTORS"
    MAX_GPU_VISIBILITY_MS=10
    MAX_FINAL_MUTATION_CAPACITY_RESERVED=0
    MAX_FINAL_DELTA_MUTABLE_ENTRIES=0
    MAX_LATE_STORAGE_OWNER_RPCS=0
    MAX_STAGE2_P99_MS=5000
    MAX_STAGE2_BACKLOG_SLOPE=0
    MAX_STAGE2_REMAINING=0
    STAGE2_DRAIN_TIMEOUT_SECONDS=60
    REQUIRE_COLD_BASELINE=1
    ;;
  *)
    echo "unknown UPDATE_ACCEPTANCE_PROFILE=$UPDATE_ACCEPTANCE_PROFILE" >&2
    echo "expected one of: querybaseline, insert24, insert64, mixed15m" >&2
    exit 1
    ;;
esac

if [[ -n "$UPDATE_ACCEPTANCE_PROFILE" && "$SHARDS" != "5" ]]; then
  echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires the five-node deployment (SHARDS=5, got $SHARDS)" >&2
  exit 1
fi
if [[ -n "$UPDATE_ACCEPTANCE_PROFILE" ]]; then
  [[ "${STORAGE_OWNER_UPDATE_MODE:-local_stitch}" == "local_stitch" ]] || {
    echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires STORAGE_OWNER_UPDATE_MODE=local_stitch" >&2
    exit 1
  }
  [[ "${STORAGE_OWNER_MAINTENANCE_MODE:-finalize}" == "finalize" ]] || {
    echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires STORAGE_OWNER_MAINTENANCE_MODE=finalize" >&2
    exit 1
  }
  [[ "${STORAGE_OWNER_REVERSE_MODE:-async}" == "async" ]] || {
    echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires STORAGE_OWNER_REVERSE_MODE=async" >&2
    exit 1
  }
  [[ "${ENABLE_BREAKDOWN:-true}" == "true" ]] || {
    echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires ENABLE_BREAKDOWN=true" >&2
    exit 1
  }
fi

number_is_positive() {
  local value="$1"
  [[ "$value" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 > 0) }'
}

number_is_less_than_one() {
  local value="$1"
  [[ "$value" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 < 1) }'
}

needs_performance_query=0
needs_insert_data=0
needs_recall_data=1
case "$WORKLOAD" in
  query)
    needs_performance_query=1
    ;;
  insert)
    needs_insert_data=1
    ;;
  both)
    needs_performance_query=1
    needs_insert_data=1
    ;;
  mixed)
    if [[ "$MIXED_MODE" == "rate_limited" ]]; then
      if number_is_positive "$TARGET_QUERY_QPS"; then needs_performance_query=1; fi
      if number_is_positive "$TARGET_WRITE_QPS"; then needs_insert_data=1; fi
    else
      if number_is_positive "$READ_RATIO"; then needs_performance_query=1; fi
      if number_is_less_than_one "$READ_RATIO"; then needs_insert_data=1; fi
    fi
    ;;
  *)
    echo "unsupported WORKLOAD=$WORKLOAD; expected query, insert, both, or mixed" >&2
    exit 1
    ;;
esac

if [[ "$UPDATE_ACCEPTANCE_PROFILE" == "querybaseline" ||
      "$UPDATE_ACCEPTANCE_PROFILE" == "insert24" ||
      "$UPDATE_ACCEPTANCE_PROFILE" == "insert64" ]]; then
  needs_recall_data=0
fi

maintenance_logs=()
if [[ -n "${STORAGE_MAINTENANCE_LOGS:-}" ]]; then
  read -r -a maintenance_logs <<< "$STORAGE_MAINTENANCE_LOGS"
fi

if [[ "$UPDATE_ACCEPTANCE_PROFILE" == "mixed15m" ]]; then
  if [[ -z "$QUERY_BASELINE_REPORT" || ! -f "$QUERY_BASELINE_REPORT" ||
        ! -r "$QUERY_BASELINE_REPORT" ]]; then
    echo "mixed15m acceptance requires QUERY_BASELINE_REPORT=<readable querybaseline JSON report>" >&2
    echo "first run UPDATE_ACCEPTANCE_PROFILE=querybaseline and pass its json path" >&2
    exit 1
  fi
  if (( PERFORMANCE_QUERY_START < INSERT_VECTOR_END &&
        INSERT_VECTOR_START < PERFORMANCE_QUERY_END )); then
    echo "mixed15m acceptance requires disjoint declared query/insert source ranges" >&2
    echo "query=[$PERFORMANCE_QUERY_START,$PERFORMANCE_QUERY_END) insert=[$INSERT_VECTOR_START,$INSERT_VECTOR_END)" >&2
    exit 1
  fi
  if (( ${#maintenance_logs[@]} != 5 )); then
    echo "mixed15m acceptance requires STORAGE_MAINTENANCE_LOGS with five local log paths" >&2
    echo "got ${#maintenance_logs[@]} path(s)" >&2
    exit 1
  fi
  declare -A seen_maintenance_logs=()
  for maintenance_log in "${maintenance_logs[@]}"; do
    if [[ ! -f "$maintenance_log" || ! -r "$maintenance_log" ]]; then
      echo "mixed15m acceptance requires a readable storage maintenance log: $maintenance_log" >&2
      exit 1
    fi
    resolved_log="$(readlink -f -- "$maintenance_log")"
    if [[ -n "${seen_maintenance_logs[$resolved_log]:-}" ]]; then
      echo "mixed15m acceptance requires five distinct maintenance logs; duplicate: $maintenance_log" >&2
      exit 1
    fi
    seen_maintenance_logs[$resolved_log]=1
  done
fi

if [[ "$REQUIRE_COLD_BASELINE" == "1" ]]; then
  if (( ${GPU_ADJACENCY_CACHE_MB:-0} != 0 || ${GPU_EXACT_CACHE_MB:-0} != 0 )); then
    echo "cold baseline requires GPU_ADJACENCY_CACHE_MB=0 and GPU_EXACT_CACHE_MB=0" >&2
    exit 1
  fi
  [[ "${GPU_GRAPH_PREFETCH_DEPTH:-0}" == "32" ]] || {
    echo "official cold baseline requires GPU_GRAPH_PREFETCH_DEPTH=32" >&2
    exit 1
  }
  [[ "${GPU_PERSISTENT_BLOCKS_PER_SM:-0}" == "4" ]] || {
    echo "official cold baseline requires GPU_PERSISTENT_BLOCKS_PER_SM=4" >&2
    exit 1
  }
  [[ "${GPU_RDMA_QPS:-0}" == "32" ]] || {
    echo "official cold baseline requires GPU_RDMA_QPS=32" >&2
    exit 1
  }
fi

# Fixed acceptance runs must fail before building or materializing large input
# files when the compute deployment is missing any schema-15 routing sidecar.
# write_service_config() validates again immediately before launching the client.
if [[ -n "$UPDATE_ACCEPTANCE_PROFILE" ]]; then
  validate_index_metadata compute
fi

RECALL_QUERY_FILE=""
GROUNDTRUTH_PATH=""
PERFORMANCE_QUERY_PATH=""
INSERT_PATH=""
if (( needs_recall_data )); then
  RECALL_QUERY_FILE="$(query_bin)"
  GROUNDTRUTH_PATH="$(groundtruth_bin)"
fi
if (( needs_performance_query )); then PERFORMANCE_QUERY_PATH="$(performance_query_bin)"; fi
if (( needs_insert_data )); then INSERT_PATH="$(insert_bin)"; fi

ensure_built dvstor_breakdown_benchmark

# The current data preparer generates query and insert files together. If the one
# file needed by a pure workload already exists, skip that coupled step so an
# unused 5M query/insert file is neither required nor regenerated. Explicit
# PREPARE_BENCHMARK_DATA/PREPARE_INSERT still takes precedence.
prepare_benchmark_data="${PREPARE_BENCHMARK_DATA:-${PREPARE_INSERT:-1}}"
prepare_recall_query="${PREPARE_QUERY:-1}"
prepare_groundtruth="${PREPARE_GROUNDTRUTH:-1}"
if (( !needs_recall_data )); then
  prepare_recall_query=0
  prepare_groundtruth=0
fi
if [[ -z "${PREPARE_BENCHMARK_DATA+x}" && -z "${PREPARE_INSERT+x}" &&
      "${OVERWRITE_BENCHMARK_DATA:-0}" != "1" ]]; then
  if [[ "$UPDATE_ACCEPTANCE_PROFILE" == "insert24" ||
        "$UPDATE_ACCEPTANCE_PROFILE" == "insert64" ]]; then
    # Insert acceptance consumes only the held-out insert stream. Missing input
    # must fail explicitly rather than materializing an unrelated 5M query set.
    prepare_benchmark_data=0
  elif (( needs_performance_query && !needs_insert_data )) &&
     [[ -s "$PERFORMANCE_QUERY_PATH" ]]; then
    prepare_benchmark_data=0
  elif (( needs_insert_data && !needs_performance_query )) &&
       [[ -s "$INSERT_PATH" ]]; then
    prepare_benchmark_data=0
  fi
fi
PREPARE_BASE="${PREPARE_BASE:-0}" \
PREPARE_QUERY="$prepare_recall_query" \
PREPARE_GROUNDTRUTH="$prepare_groundtruth" \
PREPARE_BENCHMARK_DATA="$prepare_benchmark_data" \
  "$EXPERIMENT_DIR/prepare_sift100m_data.sh"

if (( needs_performance_query )); then
  if [[ ! -s "$PERFORMANCE_QUERY_PATH" ]]; then
    echo "missing performance query file: $PERFORMANCE_QUERY_PATH" >&2
    echo "set PERFORMANCE_QUERY_FILE to a large held-out .u8bin query set" >&2
    exit 1
  fi
  if (( needs_recall_data )) &&
     [[ "$(readlink -f "$RECALL_QUERY_FILE")" == "$(readlink -f "$PERFORMANCE_QUERY_PATH")" ]]; then
    echo "recall and performance query files must be different" >&2
    exit 1
  fi
fi
if (( needs_insert_data )) && [[ ! -s "$INSERT_PATH" ]]; then
  echo "missing insert file: $INSERT_PATH" >&2
  echo "set INSERT_FILE to a held-out .u8bin insert set" >&2
  exit 1
fi
if (( needs_performance_query && needs_insert_data )) &&
   [[ "$(readlink -f "$PERFORMANCE_QUERY_PATH")" == "$(readlink -f "$INSERT_PATH")" ]]; then
  echo "performance query and insert files must be different" >&2
  exit 1
fi

if [[ "$UPDATE_ACCEPTANCE_PROFILE" == "mixed15m" ||
      "$UPDATE_ACCEPTANCE_PROFILE" == "querybaseline" ]]; then
  read -r performance_query_rows performance_query_dim < <(
    od -An -tu4 -N8 -- "$PERFORMANCE_QUERY_PATH")
  performance_query_bytes="$(stat -c %s -- "$PERFORMANCE_QUERY_PATH")"
  expected_query_bytes=$((8 + 5000000 * DIM))
  if [[ "${performance_query_rows:-}" != "5000000" ||
        "${performance_query_dim:-}" != "$DIM" ||
        "$performance_query_bytes" != "$expected_query_bytes" ]]; then
    echo "$UPDATE_ACCEPTANCE_PROFILE acceptance requires one 5,000,000-row u8bin performance query file" >&2
    echo "file=$PERFORMANCE_QUERY_PATH rows=${performance_query_rows:-unknown} dim=${performance_query_dim:-unknown} bytes=$performance_query_bytes" >&2
    exit 1
  fi
fi

effective_insert_start_id=""
if (( needs_insert_data )); then
  default_insert_start_id=$((MAX_VECTORS + 1000000))
  if [[ "$WORKLOAD" == "mixed" ]]; then
    # Dataset row numbers and node IDs are distinct namespaces, but keeping the
    # generated IDs beyond the held-out query range makes logs/replay unambiguous.
    mixed_id_namespace_end="$MAX_VECTORS"
    if (( PERFORMANCE_QUERY_END > mixed_id_namespace_end )); then
      mixed_id_namespace_end="$PERFORMANCE_QUERY_END"
    fi
    default_insert_start_id=$((mixed_id_namespace_end + 1000000))
  fi
  effective_insert_start_id="${INSERT_START_ID:-$default_insert_start_id}"
  if [[ ! "$effective_insert_start_id" =~ ^[0-9]+$ ]] ||
     (( effective_insert_start_id > 4294967295 )); then
    echo "INSERT_START_ID must be a uint32 value: $effective_insert_start_id" >&2
    exit 1
  fi
  if (( effective_insert_start_id < MAX_VECTORS )); then
    echo "INSERT_START_ID must not overlap base IDs [0,$MAX_VECTORS): $effective_insert_start_id" >&2
    exit 1
  fi
  if [[ "$UPDATE_ACCEPTANCE_PROFILE" == "insert24" ||
        "$UPDATE_ACCEPTANCE_PROFILE" == "insert64" ]]; then
    acceptance_segment_end=$((effective_insert_start_id + INSERT_ACCEPTANCE_SEGMENT_SIZE))
    if (( acceptance_segment_end > 4294967296 )); then
      echo "$UPDATE_ACCEPTANCE_PROFILE ID segment exceeds uint32: [$effective_insert_start_id,$acceptance_segment_end)" >&2
      exit 1
    fi
    echo "[breakdown] $UPDATE_ACCEPTANCE_PROFILE ID segment=[$effective_insert_start_id,$acceptance_segment_end)"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPORT_DIR/$PROFILE"
mkdir -p "$OUT_DIR"
JSON_REPORT="$OUT_DIR/sift100m_${PROFILE}_${TS}.json"
TEXT_REPORT="$OUT_DIR/sift100m_${PROFILE}_${TS}.txt"
RUNTIME_CONFIG="$OUT_DIR/service_${PROFILE}_${TS}.ini"
write_service_config "$RUNTIME_CONFIG"

cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$RUNTIME_CONFIG"
  --workload "$WORKLOAD"
  --warmup-seconds "$WARMUP_SECONDS"
  --measure-seconds "$MEASURE_SECONDS"
  --client-threads "$BENCHMARK_CLIENT_THREADS"
  --read-ratio "$READ_RATIO"
  --mixed-mode "$MIXED_MODE"
  --target-query-qps "$TARGET_QUERY_QPS"
  --target-write-qps "$TARGET_WRITE_QPS"
  --write-insert-ratio "${WRITE_INSERT_RATIO:-1}"
  --write-upsert-ratio "${WRITE_UPSERT_RATIO:-0}"
  --write-delete-ratio "${WRITE_DELETE_RATIO:-0}"
  --min-query-qps "$MIN_QUERY_QPS"
  --min-insert-qps "$MIN_INSERT_QPS"
  --min-stability-ratio "$MIN_STABILITY_RATIO"
  --min-write-stability-ratio "$MIN_WRITE_STABILITY_RATIO"
  --query-baseline-qps "$QUERY_BASELINE_QPS"
  --min-query-baseline-ratio "$MIN_QUERY_BASELINE_RATIO"
  --max-recall-drop "$MAX_RECALL_DROP"
  --max-zero-completion-windows "$MAX_ZERO_COMPLETION_WINDOWS"
  --max-zero-query-windows "$MAX_ZERO_QUERY_WINDOWS"
  --max-zero-write-windows "$MAX_ZERO_WRITE_WINDOWS"
  --max-drain-seconds "$MAX_DRAIN_SECONDS"
  --min-rate-attainment-ratio "$MIN_RATE_ATTAINMENT_RATIO"
  --max-gpu-visibility-ms "$MAX_GPU_VISIBILITY_MS"
  --max-final-mutation-capacity-reserved "$MAX_FINAL_MUTATION_CAPACITY_RESERVED"
  --max-final-delta-mutable-entries "$MAX_FINAL_DELTA_MUTABLE_ENTRIES"
  --max-late-storage-owner-rpcs "$MAX_LATE_STORAGE_OWNER_RPCS"
  --max-stage2-p99-ms "$MAX_STAGE2_P99_MS"
  --max-stage2-backlog-slope "$MAX_STAGE2_BACKLOG_SLOPE"
  --max-stage2-remaining "$MAX_STAGE2_REMAINING"
  --stage2-drain-timeout-seconds "$STAGE2_DRAIN_TIMEOUT_SECONDS"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

if [[ -n "$QUERY_BASELINE_REPORT" ]]; then
  cmd+=(--query-baseline-report "$QUERY_BASELINE_REPORT")
fi

if (( needs_recall_data )); then
  cmd+=(--recall-query-file "$RECALL_QUERY_FILE"
        --groundtruth-file "$GROUNDTRUTH_PATH"
        --recall-queries "$RECALL_QUERIES"
        --recall-k "$RECALL_K"
        --recall-mode "$RECALL_MODE"
        --recall-base-id-limit "$RECALL_BASE_ID_LIMIT"
        --min-recall "$MIN_RECALL")
fi
if (( needs_performance_query )); then
  cmd+=(--performance-query-file "$PERFORMANCE_QUERY_PATH")
fi
if (( needs_insert_data )); then
  cmd+=(--insert-start-id "$effective_insert_start_id" --insert-file "$INSERT_PATH")
fi
for maintenance_log in "${maintenance_logs[@]}"; do
  cmd+=(--storage-maintenance-log "$maintenance_log")
done

if [[ -n "$UPDATE_ACCEPTANCE_PROFILE" ]]; then
  echo "[breakdown] update acceptance profile=$UPDATE_ACCEPTANCE_PROFILE"
fi

printf '[breakdown] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
