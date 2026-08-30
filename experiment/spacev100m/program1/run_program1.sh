#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/spacev100m"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
WARMUP_OPS="${WARMUP_OPS:-128}"
MEASURE_OPS="${MEASURE_OPS:-1000}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
QUALITY_STAGE2_DELAY_MS="${QUALITY_STAGE2_DELAY_MS:-15000}"
QUALITY_SETTLE_SECONDS="${QUALITY_SETTLE_SECONDS:-30}"
QUALITY_INSERT_COUNT="${QUALITY_INSERT_COUNT:-1000}"
QUALITY_RECALL_QUERIES="${QUALITY_RECALL_QUERIES:-1000}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program1_$(date +%Y%m%d_%H%M%S)}"
ACTION="${1:-all}"
MANIFEST="$RUN_ROOT/manifest.tsv"

mkdir -p "$RUN_ROOT"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'case\treport\n' > "$MANIFEST"
fi

record_case() {
  local case_name="$1" report="$2" temporary="$MANIFEST.tmp"
  awk -F '\t' -v name="$case_name" 'NR == 1 || $1 != name' \
    "$MANIFEST" > "$temporary"
  printf '%s\t%s\n' "$case_name" "$report" >> "$temporary"
  mv "$temporary" "$MANIFEST"
}

existing_report() {
  local case_name="$1"
  awk -F '\t' -v name="$case_name" '$1 == name {print $2; exit}' "$MANIFEST"
}

wait_storage() {
  local case_name="$1"
  echo
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  if [[ "$case_name" == quality ]]; then
    echo "  QUALITY_STAGE2_DELAY_MS=$QUALITY_STAGE2_DELAY_MS ./experiment/spacev100m/program1/start_storage_case.sh quality"
  else
    echo "  ./experiment/spacev100m/program1/start_storage_case.sh $case_name"
  fi
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

run_update_case() {
  local case_name="$1" mode="$2"
  local report_dir="$RUN_ROOT/$case_name"
  local existing
  existing="$(existing_report "$case_name")"
  if [[ -n "$existing" && -f "$existing" ]]; then
    echo "复用已完成 case=$case_name report=$existing"
    return
  fi

  wait_storage "$case_name"
  mkdir -p "$report_dir"
  echo "[$(date --iso-8601=seconds)] compute case=$case_name mode=$mode"
  REPORT_DIR="$report_dir" \
  STORAGE_OWNER_UPDATE_COMPLETION_MODE="$mode" \
  STORAGE_OWNER_STAGE2_INITIAL_DELAY_MS=0 \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
  GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled \
  ENABLE_BREAKDOWN=true \
  WORKLOAD=insert \
  BENCHMARK_MODE=ops \
  WARMUP_OPS="$WARMUP_OPS" \
  MEASURE_OPS="$MEASURE_OPS" \
  BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
  RECALL_QUERIES=1 \
    "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | \
      tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'spacev100m_*.json' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 {print $2}')"
  [[ -n "$report" ]] || { echo "missing report for $case_name" >&2; exit 1; }
  if [[ "$case_name" == baseline ]]; then
    python3 - "$report" <<'PY'
import json, sys
critical = json.load(open(sys.argv[1], encoding="utf-8")).get(
    "coupled_insert_critical_path", {})
if "rdma_wait_ns" not in critical:
    raise SystemExit("baseline report is missing rdma_wait_ns")
if critical["rdma_wait_ns"] > critical.get("total_ns", 0):
    raise SystemExit("baseline rdma_wait_ns exceeds total_ns")
PY
  fi
  record_case "$case_name" "$report"
}

run_quality() {
  local existing report_dir="$RUN_ROOT/quality"
  existing="$(existing_report quality)"
  if [[ -n "$existing" && -f "$existing" ]]; then
    echo "复用已完成 case=quality report=$existing"
    return
  fi

  wait_storage quality
  source "$EXPERIMENT_DIR/common.sh"
  load_experiment_profile "$PROFILE"
  export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
  export STORAGE_OWNER_STAGE2_INITIAL_DELAY_MS="$QUALITY_STAGE2_DELAY_MS"
  export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
  export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
  export ENABLE_BREAKDOWN=true

  ensure_built dvstor_sift101m_long_insert_recall
  local insert_path config report
  insert_path="$(insert_bin)"
  [[ -s "$insert_path" ]] || { echo "missing insert file: $insert_path" >&2; exit 1; }
  mkdir -p "$report_dir"
  config="$report_dir/service.ini"
  report="$report_dir/quality.json"
  write_service_config "$config"

  "$BUILD_DIR/dvstor_sift101m_long_insert_recall" \
    --service-config "$config" \
    --insert-file "$insert_path" \
    --insert-start-id "${QUALITY_INSERT_START_ID:-$((MAX_VECTORS + 1000000))}" \
    --insert-count "$QUALITY_INSERT_COUNT" \
    --insert-threads "${QUALITY_INSERT_THREADS:-16}" \
    --insert-batch-size "${QUALITY_INSERT_BATCH_SIZE:-16}" \
    --self-recall-queries "$QUALITY_RECALL_QUERIES" \
    --self-recall-k "${QUALITY_RECALL_K:-10}" \
    --settle-seconds "$QUALITY_SETTLE_SECONDS" \
    --report-json "$report" \
    --report-text "$report_dir/quality.txt" \
    2>&1 | tee "$report_dir/driver.log"
  record_case quality "$report"
}

summarize() {
  python3 "$PROGRAM_DIR/plot_program1.py" "$RUN_ROOT"
  echo "结果目录：$RUN_ROOT"
}

case "$ACTION" in
  all)
    run_update_case baseline coupled
    run_update_case solution decoupled
    run_quality
    summarize
    ;;
  baseline) run_update_case baseline coupled ;;
  solution) run_update_case solution decoupled ;;
  quality) run_quality ;;
  summarize) summarize ;;
  *) echo "usage: $0 [all|baseline|solution|quality|summarize]" >&2; exit 2 ;;
esac
