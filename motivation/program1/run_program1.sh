#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
WARMUP_OPS="${WARMUP_OPS:-128}"
MEASURE_OPS="${MEASURE_OPS:-1000}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program1_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"
printf 'case\treport\n' > "$RUN_ROOT/manifest.tsv"

wait_storage() {
  local case_name="$1"
  echo
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  echo "  ./motivation/program1/start_storage_case.sh $case_name"
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

run_case() {
  local case_name="$1"
  local mode="$2"
  local report_dir="$RUN_ROOT/$case_name"
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
    "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | tee "$report_dir/driver.log"
  report="$(find "$report_dir" -type f -name 'sift100m_*.json' -print -quit)"
  [[ -n "$report" ]] || { echo "missing report for $case_name" >&2; exit 1; }
  printf '%s\t%s\n' "$case_name" "$report" >> "$RUN_ROOT/manifest.tsv"
}

run_case baseline coupled
run_case solution decoupled

echo
echo "基线和方案一已完成。结果：$RUN_ROOT"
echo "接着运行精度实验："
echo "  RUN_ROOT=$RUN_ROOT ./motivation/program1/run_quality.sh"
