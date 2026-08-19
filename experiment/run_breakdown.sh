#!/usr/bin/env bash
set -euo pipefail

# 普通用户只需修改 sift100m_common.sh 的 "Benchmark input files" 配置块，
# 或在命令行用同名环境变量覆盖。
show_help() {
  cat <<'EOF'
Usage: ./experiment/run_breakdown.sh [PROFILE]

常用负载：
  WORKLOAD=query|insert|both|mixed        默认 mixed
  BENCHMARK_MODE=time|ops                 默认 time；ops 固定 warmup/measurement 工作量
  WARMUP_OPS/MEASURE_OPS                 ops 模式下的固定操作数（默认 100/1000）

并发负载：
  BENCHMARK_CLIENT_THREADS=auto           默认；由有界 GPU/RPC 容量推导
  BENCHMARK_CLIENT_THREAD_CAP=1024        auto 模式的安全上限；截断时会明确告警
  BENCHMARK_CLIENT_THREADS=N              显式固定并发，用于延迟/并发扫描
  MIXED_MODE=fixed_threads                默认；READ_RATIO 划分专用读/写 caller
  MIXED_MODE=probability                  READ_RATIO 控制闭环操作选择比例
  MIXED_MODE=rate_limited                 使用显式 query/write QPS 调度和达成率

数据文件与声明范围（在 sift100m_common.sh 集中配置，也可用环境变量覆盖）：
  PERFORMANCE_QUERY_FILE                 默认 sift100m_to_110m_query.u8bin
  PERFORMANCE_QUERY_START/END            默认 [100000000,110000000)
  INSERT_FILE                            默认 sift110m_to_120m_insert.u8bin
  INSERT_VECTOR_START/END                默认 [110000000,120000000)

数据准备：
  PREPARE_BENCHMARK_DATA=0                默认；只读取预生成 u8bin，不需要 bigann_base.bvecs
  PREPARE_BENCHMARK_DATA=1                从 BENCHMARK_VECTOR_SOURCE 生成 benchmark 文件
  PREPARE_QUERY=1 PREPARE_GROUNDTRUTH=1   显式生成 recall 输入；默认也只读取

示例：
  WORKLOAD=insert BENCHMARK_CLIENT_THREADS=24 ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
  WORKLOAD=mixed READ_RATIO=0.5 ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
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
validate_vector_id_namespace_size

WORKLOAD="${WORKLOAD:-mixed}"
BENCHMARK_CLIENT_THREADS_REQUEST="${BENCHMARK_CLIENT_THREADS:-auto}"
BENCHMARK_CLIENT_THREAD_CAP="${BENCHMARK_CLIENT_THREAD_CAP:-1024}"
READ_RATIO="${READ_RATIO:-0.5}"
MIXED_MODE="${MIXED_MODE:-fixed_threads}"
# MIXED_MODE="${MIXED_MODE:-probability}"
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
BENCHMARK_MODE="${BENCHMARK_MODE:-time}"
WARMUP_OPS="${WARMUP_OPS:-100}"
MEASURE_OPS="${MEASURE_OPS:-1000}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_K="${RECALL_K:-$K}"
TARGET_QUERY_QPS="${TARGET_QUERY_QPS:-0}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-0}"
RECALL_MODE="${RECALL_MODE:-all}"
RECALL_BASE_ID_LIMIT="${RECALL_BASE_ID_LIMIT:-0}"

if [[ "$BENCHMARK_MODE" != "time" && "$BENCHMARK_MODE" != "ops" ]]; then
  echo "BENCHMARK_MODE must be time or ops" >&2
  exit 1
fi
if [[ ! "$WARMUP_SECONDS" =~ ^(0|[1-9][0-9]*)$ ]] ||
   [[ ! "$MEASURE_SECONDS" =~ ^(0|[1-9][0-9]*)$ ]] ||
   [[ ! "$WARMUP_OPS" =~ ^(0|[1-9][0-9]*)$ ]] ||
   [[ ! "$MEASURE_OPS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "WARMUP_SECONDS and MEASURE_SECONDS must be non-negative integers" >&2
  exit 1
fi
if [[ "$BENCHMARK_MODE" == "time" ]] &&
   (( (WARMUP_SECONDS == 0) != (MEASURE_SECONDS == 0) )); then
  echo "WARMUP_SECONDS and MEASURE_SECONDS must either both be zero or both be positive" >&2
  exit 1
fi
if [[ "$BENCHMARK_MODE" == "ops" ]] &&
   (( WARMUP_OPS == 0 || MEASURE_OPS == 0 )); then
  echo "WARMUP_OPS and MEASURE_OPS must be positive in ops mode" >&2
  exit 1
fi

number_is_positive() {
  local value="$1"
  [[ "$value" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 > 0) }'
}

number_is_nonnegative() {
  local value="$1"
  [[ "$value" =~ ^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 >= 0) }'
}

number_is_at_most_one() {
  local value="$1"
  [[ "$value" =~ ^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 <= 1) }'
}

number_is_less_than_one() {
  local value="$1"
  [[ "$value" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    return 1
  awk -v value="$value" 'BEGIN { exit !(value + 0 < 1) }'
}

ceil_capacity_ratio() {
  local capacity="$1"
  local ratio="$2"
  awk -v capacity="$capacity" -v ratio="$ratio" '
    BEGIN {
      value = capacity / ratio
      rounded = int(value)
      if (rounded + 1e-12 < value) ++rounded
      print rounded
    }'
}

fixed_thread_split() {
  local threads="$1"
  local ratio="$2"
  awk -v threads="$threads" -v ratio="$ratio" '
    BEGIN {
      if (ratio <= 0) {
        reads = 0
      } else if (ratio >= 1) {
        reads = threads
      } else {
        reads = int(threads * ratio + 0.5)
        if (reads < 1) reads = 1
        if (reads > threads - 1) reads = threads - 1
      }
      print reads, threads - reads
    }'
}

if ! number_is_nonnegative "$READ_RATIO" ||
   ! number_is_at_most_one "$READ_RATIO"; then
  echo "READ_RATIO must be a number in [0,1]: $READ_RATIO" >&2
  exit 1
fi
if ! number_is_nonnegative "$TARGET_QUERY_QPS" ||
   ! number_is_nonnegative "$TARGET_WRITE_QPS"; then
  echo "TARGET_QUERY_QPS and TARGET_WRITE_QPS must be non-negative numbers" >&2
  exit 1
fi
case "$MIXED_MODE" in
  fixed_threads|probability|rate_limited) ;;
  *)
    echo "unsupported MIXED_MODE=$MIXED_MODE; expected fixed_threads, probability, or rate_limited" >&2
    exit 1
    ;;
esac

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

if [[ "$MIXED_MODE" == "rate_limited" ]]; then
  if [[ "$WORKLOAD" != "mixed" ]] ||
     { ! number_is_positive "$TARGET_QUERY_QPS" &&
       ! number_is_positive "$TARGET_WRITE_QPS"; }; then
    echo "MIXED_MODE=rate_limited requires WORKLOAD=mixed and at least one positive target rate" >&2
    exit 1
  fi
elif number_is_positive "$TARGET_QUERY_QPS" ||
     number_is_positive "$TARGET_WRITE_QPS"; then
  echo "target query/write rates require MIXED_MODE=rate_limited" >&2
  exit 1
fi

# The benchmark API is synchronous: one client thread contributes at most one
# in-flight operation. Derive the default closed-loop offered concurrency from
# the engine's explicit bounded capacities, not from a measured QPS or a
# dataset-specific constant. This does not enlarge any engine queue and does
# not discard work; excess callers wait behind the existing bounded admission
# paths.
if [[ ! "${GPU_QUERY_SLOTS:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_QUERY_SLOTS must be a positive integer: ${GPU_QUERY_SLOTS:-<unset>}" >&2
  exit 1
fi
if [[ ! "$SHARDS" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "${STORAGE_OWNER_RPC_DEPTH:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS and STORAGE_OWNER_RPC_DEPTH must be positive integers" >&2
  exit 1
fi
if [[ ! "$BENCHMARK_CLIENT_THREAD_CAP" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCHMARK_CLIENT_THREAD_CAP must be a positive integer: $BENCHMARK_CLIENT_THREAD_CAP" >&2
  exit 1
fi

query_concurrency_capacity="$GPU_QUERY_SLOTS"
write_concurrency_capacity=$((SHARDS * STORAGE_OWNER_RPC_DEPTH))
auto_client_threads=1
concurrency_derivation=""
case "$WORKLOAD" in
  query)
    auto_client_threads="$query_concurrency_capacity"
    concurrency_derivation="gpu_query_slots"
    ;;
  insert)
    auto_client_threads="$write_concurrency_capacity"
    concurrency_derivation="shards_x_storage_rpc_depth"
    ;;
  both)
    # The benchmark executes the insert and query phases sequentially.
    if (( query_concurrency_capacity > write_concurrency_capacity )); then
      auto_client_threads="$query_concurrency_capacity"
    else
      auto_client_threads="$write_concurrency_capacity"
    fi
    concurrency_derivation="max(gpu_query_slots,shards_x_storage_rpc_depth);phases_are_sequential"
    ;;
  mixed)
    if [[ "$MIXED_MODE" == "rate_limited" ]]; then
      auto_client_threads=0
      if number_is_positive "$TARGET_QUERY_QPS"; then
        auto_client_threads=$((auto_client_threads + query_concurrency_capacity))
      fi
      if number_is_positive "$TARGET_WRITE_QPS"; then
        auto_client_threads=$((auto_client_threads + write_concurrency_capacity))
      fi
      concurrency_derivation="sum(active_bounded_path_capacities);shared_rate_pacer"
    else
      query_required_threads=0
      write_required_threads=0
      if number_is_positive "$READ_RATIO"; then
        query_required_threads="$(ceil_capacity_ratio \
          "$query_concurrency_capacity" "$READ_RATIO")"
      fi
      if number_is_less_than_one "$READ_RATIO"; then
        write_ratio="$(awk -v ratio="$READ_RATIO" 'BEGIN { print 1 - ratio }')"
        write_required_threads="$(ceil_capacity_ratio \
          "$write_concurrency_capacity" "$write_ratio")"
      fi
      if (( query_required_threads > write_required_threads )); then
        auto_client_threads="$query_required_threads"
      else
        auto_client_threads="$write_required_threads"
      fi
      concurrency_derivation="max(ceil(gpu_query_slots/read_ratio),ceil(shards_x_storage_rpc_depth/write_ratio))"
    fi
    ;;
esac

client_threads_source="explicit"
auto_client_threads_capped=0
if [[ "$BENCHMARK_CLIENT_THREADS_REQUEST" == "auto" ]]; then
  client_threads_source="auto"
  BENCHMARK_CLIENT_THREADS="$auto_client_threads"
  if (( BENCHMARK_CLIENT_THREADS > BENCHMARK_CLIENT_THREAD_CAP )); then
    BENCHMARK_CLIENT_THREADS="$BENCHMARK_CLIENT_THREAD_CAP"
    auto_client_threads_capped=1
    echo "[breakdown][warning] auto concurrency requires $auto_client_threads threads but cap=$BENCHMARK_CLIENT_THREAD_CAP; bounded paths may be underfilled" >&2
  fi
elif [[ "$BENCHMARK_CLIENT_THREADS_REQUEST" =~ ^[1-9][0-9]*$ ]]; then
  BENCHMARK_CLIENT_THREADS="$BENCHMARK_CLIENT_THREADS_REQUEST"
else
  echo "BENCHMARK_CLIENT_THREADS must be 'auto' or a positive integer: $BENCHMARK_CLIENT_THREADS_REQUEST" >&2
  exit 1
fi

echo "[breakdown] concurrency clients=$BENCHMARK_CLIENT_THREADS source=$client_threads_source auto_required=$auto_client_threads auto_cap=$BENCHMARK_CLIENT_THREAD_CAP gpu_query_slots=$query_concurrency_capacity storage_rpc_inflight=$write_concurrency_capacity (${SHARDS}x${STORAGE_OWNER_RPC_DEPTH}) derivation=$concurrency_derivation"
projected_read_threads=-1
projected_write_threads=-1
if [[ "$WORKLOAD" == "mixed" && "$MIXED_MODE" == "fixed_threads" ]]; then
  read -r projected_read_threads projected_write_threads < <(
    fixed_thread_split "$BENCHMARK_CLIENT_THREADS" "$READ_RATIO")
  echo "[breakdown] fixed-thread offered split reads=$projected_read_threads writes=$projected_write_threads (READ_RATIO controls concurrent callers, not guaranteed completed-op ratio)"
fi

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
  read -r performance_query_rows performance_query_dim < <(
    od -An -N8 -t u4 "$PERFORMANCE_QUERY_PATH")
  if [[ ! "$performance_query_rows" =~ ^[1-9][0-9]*$ ]] ||
     [[ ! "$performance_query_dim" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid performance query header: $PERFORMANCE_QUERY_PATH" >&2
    exit 1
  fi
  if [[ "$BENCHMARK_MODE" == "time" ]] &&
     (( WARMUP_SECONDS + MEASURE_SECONDS > 0 )); then
    unique_query_budget_qps="$(awk \
      -v rows="$performance_query_rows" \
      -v seconds="$((WARMUP_SECONDS + MEASURE_SECONDS))" \
      'BEGIN { printf "%.1f", rows / seconds }')"
    echo "[breakdown] unique performance-query budget rows=$performance_query_rows total_timed_seconds=$((WARMUP_SECONDS + MEASURE_SECONDS)) max_average_query_qps_without_reuse=$unique_query_budget_qps"
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
  if (( effective_insert_start_id >= VECTOR_ID_NAMESPACE_SIZE )); then
    echo "INSERT_START_ID must be inside vector ID namespace [0,$VECTOR_ID_NAMESPACE_SIZE): $effective_insert_start_id" >&2
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
  --profile-name "$PROFILE"
  --system-variant-label "$SYSTEM_VARIANT_LABEL"
  --workload "$WORKLOAD"
  --warmup-ops "$WARMUP_OPS"
  --measure-ops "$MEASURE_OPS"
  --warmup-seconds "$([[ "$BENCHMARK_MODE" == "time" ]] && echo "$WARMUP_SECONDS" || echo 0)"
  --measure-seconds "$([[ "$BENCHMARK_MODE" == "time" ]] && echo "$MEASURE_SECONDS" || echo 0)"
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

# Keep the auto-load derivation beside the measured values. The benchmark
# already records the selected thread count and engine slots; this adds the
# driver-side source/cap decision without changing any measured field.
python3 - "$JSON_REPORT" \
  "$client_threads_source" "$BENCHMARK_CLIENT_THREADS" \
  "$auto_client_threads" "$BENCHMARK_CLIENT_THREAD_CAP" \
  "$auto_client_threads_capped" "$query_concurrency_capacity" \
  "$write_concurrency_capacity" "$SHARDS" "$STORAGE_OWNER_RPC_DEPTH" \
  "$concurrency_derivation" "$projected_read_threads" \
  "$projected_write_threads" <<'PY_CONCURRENCY_METADATA'
import json
import os
import sys

(
    report_path,
    source,
    selected,
    required,
    cap,
    capped,
    query_capacity,
    write_capacity,
    shards,
    rpc_depth,
    derivation,
    projected_reads,
    projected_writes,
) = sys.argv[1:]

with open(report_path, "r", encoding="utf-8") as stream:
    report = json.load(stream)

metadata = {
    "semantics": "closed_loop_synchronous_no_drop",
    "client_threads_source": source,
    "selected_client_threads": int(selected),
    "auto_required_threads": int(required),
    "auto_thread_cap": int(cap),
    "auto_cap_applied": bool(int(capped)),
    "gpu_query_slot_capacity": int(query_capacity),
    "storage_rpc_inflight_capacity": int(write_capacity),
    "storage_shards": int(shards),
    "storage_rpc_depth_per_shard": int(rpc_depth),
    "derivation": derivation,
}
if int(projected_reads) >= 0:
    metadata["fixed_thread_projected_read_threads"] = int(projected_reads)
    metadata["fixed_thread_projected_write_threads"] = int(projected_writes)

report.setdefault("meta", {})["benchmark_driver_concurrency"] = metadata
temporary_path = report_path + ".concurrency.tmp"
with open(temporary_path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
os.replace(temporary_path, report_path)
PY_CONCURRENCY_METADATA

echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
