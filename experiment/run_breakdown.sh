#!/usr/bin/env bash
set -euo pipefail

# 普通用户只需修改 sift100m_common.sh 的 "Benchmark input files" 配置块，
# 或在命令行用同名环境变量覆盖。
show_help() {
  cat <<'EOF'
Usage: ./experiment/run_breakdown.sh [PROFILE]

常用负载：
  WORKLOAD=query|insert|both|mixed        默认 mixed

数据文件与声明范围（在 sift100m_common.sh 集中配置，也可用环境变量覆盖）：
  PERFORMANCE_QUERY_FILE                 默认 sift100m_to_105m_query.u8bin
  PERFORMANCE_QUERY_START/END            默认 [100000000,105000000)
  INSERT_FILE                            默认 sift103m_to_105m_insert.u8bin
  INSERT_VECTOR_START/END                默认 [103000000,105000000)

数据准备：
  PREPARE_BENCHMARK_DATA=0                默认；只读取预生成 u8bin，不需要 bigann_base.bvecs
  PREPARE_BENCHMARK_DATA=1                从 BENCHMARK_VECTOR_SOURCE 生成 benchmark 文件
  PREPARE_QUERY=1 PREPARE_GROUNDTRUTH=1   显式生成 recall 输入；默认也只读取

示例：
  WORKLOAD=insert BENCHMARK_CLIENT_THREADS=24 ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
  WORKLOAD=mixed BENCHMARK_CLIENT_THREADS=128 READ_RATIO=0.5 ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
  WORKLOAD=mixed MIXED_MODE=rate_limited TARGET_QUERY_QPS=5000 TARGET_WRITE_QPS=1000 \
    ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio

脚本只记录原始结果，不对 QPS、召回、稳定性或后台维护做通过/失败判断。
当前 profile 的各分片 memory-node 日志会被自动采集到 stage2 报告。
EOF
}

case "${1:-}" in
  -h|--help) show_help; exit 0 ;;
esac

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
TARGET_QUERY_QPS="${TARGET_QUERY_QPS:-0}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-0}"
RECALL_MODE="${RECALL_MODE:-all}"
RECALL_BASE_ID_LIMIT="${RECALL_BASE_ID_LIMIT:-0}"

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

# Pure-query clients do not initialize update ownership state. Any workload
# with writes additionally starts the update runtime.
if (( needs_insert_data )); then
  ENABLE_UPDATES=true
else
  ENABLE_UPDATES=false
fi

maintenance_logs=()
if [[ -n "${STORAGE_MAINTENANCE_LOGS:-}" ]]; then
  read -r -a maintenance_logs <<< "$STORAGE_MAINTENANCE_LOGS"
else
  # start_memory_node.sh uses this deterministic name for every shard. Feed
  # those logs to the benchmark automatically so the report contains the raw
  # stage2 observations from the measurement window. Missing/unreadable logs
  # remain visible as such in the report instead of silently disabling stage2
  # telemetry.
  for ((node_id = 1; node_id <= SHARDS; ++node_id)); do
    maintenance_logs+=(
      "$LOG_DIR/memory_node_${node_id}_${PROFILE}.log"
    )
  done
fi

# Validate the deployed index before preparing inputs or rebuilding the client.
# write_service_config() validates once more immediately before launch.
validate_index_metadata compute

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

# Benchmark inputs are runtime inputs, not compute-node build products. They are
# never generated unless the corresponding PREPARE_* switch is explicitly 1.
prepare_benchmark_data="${PREPARE_BENCHMARK_DATA:-0}"
prepare_recall_query="${PREPARE_QUERY:-0}"
prepare_groundtruth="${PREPARE_GROUNDTRUTH:-0}"
if (( !needs_recall_data )); then
  prepare_recall_query=0
  prepare_groundtruth=0
fi
if [[ "$prepare_recall_query" == "1" ||
      "$prepare_groundtruth" == "1" ||
      "$prepare_benchmark_data" == "1" ]]; then
  PREPARE_BASE=0 \
  PREPARE_QUERY="$prepare_recall_query" \
  PREPARE_GROUNDTRUTH="$prepare_groundtruth" \
  PREPARE_BENCHMARK_DATA="$prepare_benchmark_data" \
    "$EXPERIMENT_DIR/prepare_sift100m_data.sh"
fi

if (( needs_performance_query )); then
  echo "[breakdown] performance query: $PERFORMANCE_QUERY_PATH [$PERFORMANCE_QUERY_START,$PERFORMANCE_QUERY_END)"
fi
if (( needs_insert_data )); then
  echo "[breakdown] insert: $INSERT_PATH [$INSERT_VECTOR_START,$INSERT_VECTOR_END)"
fi

if (( needs_recall_data )); then
  [[ -s "$RECALL_QUERY_FILE" ]] || {
    echo "missing recall query file: $RECALL_QUERY_FILE" >&2
    echo "run PREPARE_QUERY=1 ./experiment/prepare_sift100m_data.sh once on a data-preparation node" >&2
    exit 1
  }
  [[ -s "$GROUNDTRUTH_PATH" ]] || {
    echo "missing groundtruth file: $GROUNDTRUTH_PATH" >&2
    echo "run PREPARE_GROUNDTRUTH=1 ./experiment/prepare_sift100m_data.sh once on a data-preparation node" >&2
    exit 1
  }
fi

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
fi

ensure_built dvstor_breakdown_benchmark

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
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

if (( needs_recall_data )); then
  cmd+=(--recall-query-file "$RECALL_QUERY_FILE"
        --groundtruth-file "$GROUNDTRUTH_PATH"
        --recall-queries "$RECALL_QUERIES"
        --recall-k "$RECALL_K"
        --recall-mode "$RECALL_MODE"
        --recall-base-id-limit "$RECALL_BASE_ID_LIMIT")
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

printf '[breakdown] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
