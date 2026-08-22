#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
WARMUP_SECONDS="${WARMUP_SECONDS:-10}"
MEASURE_SECONDS="${MEASURE_SECONDS:-30}"
CLIENT_THREADS="${CLIENT_THREADS:-256}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program2_$(date +%Y%m%d_%H%M%S)}"
ACTION="${1:-all}"

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/manifest.tsv" ]]; then
  printf 'case\treport\n' > "$RUN_ROOT/manifest.tsv"
fi

record_case() {
  local case_name="$1"
  local report="$2"
  local temporary="$RUN_ROOT/manifest.tsv.tmp"
  awk -F '\t' -v name="$case_name" 'NR == 1 || $1 != name' \
    "$RUN_ROOT/manifest.tsv" > "$temporary"
  printf '%s\t%s\n' "$case_name" "$report" >> "$temporary"
  mv "$temporary" "$RUN_ROOT/manifest.tsv"
}

wait_storage() {
  local case_name="$1"
  echo
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  echo "  ./motivation/program2/start_storage_case.sh $case_name"
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

run_query_case() {
  local case_name="$1"
  local graph_policy="$2"
  local dynamic_extent="$3"
  local report_dir="$RUN_ROOT/$case_name"

  wait_storage "$case_name"
  mkdir -p "$report_dir"
  echo "[$(date --iso-8601=seconds)] query case=$case_name policy=$graph_policy"
  REPORT_DIR="$report_dir" \
  STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE=manual \
  GPU_QUERY_GRAPH_READ_POLICY="$graph_policy" \
  GPU_DYNAMIC_GRAPH_EXTENT="$dynamic_extent" \
  GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled \
  ENABLE_BREAKDOWN=false \
  WORKLOAD=query \
  BENCHMARK_MODE=time \
  WARMUP_SECONDS="$WARMUP_SECONDS" \
  MEASURE_SECONDS="$MEASURE_SECONDS" \
  BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
  RECALL_QUERIES="$RECALL_QUERIES" \
    "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | \
      tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'sift100m_*.json' -print -quit)"
  [[ -n "$report" ]] || {
    echo "missing JSON report for $case_name" >&2
    exit 1
  }
  record_case "$case_name" "$report"
}

run_probe() {
  local live_report live_ini report_dir
  live_report="$(awk -F '\t' '$1 == "live" {print $2}' "$RUN_ROOT/manifest.tsv")"
  [[ -n "$live_report" && -f "$live_report" ]] || {
    echo "probe requires the live query case in $RUN_ROOT" >&2
    exit 1
  }
  live_ini="$(find "$(dirname "$live_report")" -maxdepth 1 \
    -type f -name 'service_*.ini' -print -quit)"
  [[ -n "$live_ini" ]] || {
    echo "missing service INI beside $live_report" >&2
    exit 1
  }

  report_dir="$RUN_ROOT/probe"
  mkdir -p "$report_dir"
  local probe_log="$report_dir/probe.log"
  local expected_rows=$(( ${PROBE_REPEATS:-3} * 9 ))
  local observed_rows=0
  if [[ -f "$probe_log" ]]; then
    observed_rows="$(grep -c '^LIVE_EXTENT_RDMA_CSV,' "$probe_log" || true)"
  fi
  if (( observed_rows == expected_rows )); then
    echo "复用已完整采集的 RDMA probe：$probe_log ($observed_rows rows)"
    record_case probe "$probe_log"
    return
  fi

  wait_storage probe
  cmake --build "$PROJECT_DIR/build" -j --target dvstor_gpunetio_loopback_probe

  echo "[$(date --iso-8601=seconds)] RDMA protocol probe"
  DVSTOR_GPUNETIO_PAYLOAD_SWEEP=1 \
  DVSTOR_GPUNETIO_PAYLOAD_BYTES="400 832" \
  DVSTOR_GPUNETIO_PAIRED_BODY_BYTES="384" \
  DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST="1 32 160" \
  DVSTOR_GPUNETIO_PAYLOAD_REPEATS="${PROBE_REPEATS:-3}" \
  DVSTOR_GPUNETIO_PAYLOAD_WARMUP_ITERATIONS="${PROBE_WARMUP_ITERATIONS:-32}" \
  DVSTOR_GPUNETIO_PAYLOAD_ITERATIONS="${PROBE_ITERATIONS:-512}" \
  DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS="${PROBE_BATCH_READS:-16}" \
    "$PROJECT_DIR/build/dvstor_gpunetio_loopback_probe" \
      --service-config "$live_ini" 2>&1 | tee "$probe_log"
  record_case probe "$probe_log"
}

summarize() {
  python3 "$PROGRAM_DIR/summarize_program2.py" "$RUN_ROOT"
  python3 "$PROGRAM_DIR/plot_program2.py" "$RUN_ROOT"
  echo "结果目录：$RUN_ROOT"
}

case "$ACTION" in
  all)
    run_query_case fixed fixed false
    run_query_case header header-neighbor false
    run_query_case live live-extent true
    run_probe
    summarize
    ;;
  fixed)
    run_query_case fixed fixed false
    ;;
  header)
    run_query_case header header-neighbor false
    ;;
  live)
    run_query_case live live-extent true
    ;;
  probe)
    run_probe
    summarize
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 [all|fixed|header|live|probe|summarize]" >&2
    exit 2
    ;;
esac
