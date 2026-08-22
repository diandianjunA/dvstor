#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
REPEATS="${REPEATS:-3}"
WORKLOADS="${WORKLOADS:-query mixed}"
WARMUP_SECONDS="${WARMUP_SECONDS:-5}"
MEASURE_SECONDS="${MEASURE_SECONDS:-20}"
CLIENT_THREADS="${CLIENT_THREADS:-auto}"
CLIENT_THREAD_CAP="${CLIENT_THREAD_CAP:-512}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-500}"
WRITE_THREADS="${WRITE_THREADS:-16}"
MIN_WRITE_ATTAINMENT="${MIN_WRITE_ATTAINMENT:-0.95}"
GPU_COMMIT_WIDTH="${GPU_COMMIT_WIDTH:-16}"
VERIFY_CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-1}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program3_$(date +%Y%m%d_%H%M%S)}"
ACTION="${1:-all}"

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/manifest.tsv" ]]; then
  printf 'workload\tmode\trepeat\treport\n' > "$RUN_ROOT/manifest.tsv"
fi

record_case() {
  local workload="$1" mode="$2" repeat="$3" report="$4"
  local temporary="$RUN_ROOT/manifest.tsv.tmp"
  awk -F '\t' -v w="$workload" -v m="$mode" -v r="$repeat" \
    'NR == 1 || !($1 == w && $2 == m && $3 == r)' \
    "$RUN_ROOT/manifest.tsv" > "$temporary"
  printf '%s\t%s\t%s\t%s\n' "$workload" "$mode" "$repeat" "$report" \
    >> "$temporary"
  mv "$temporary" "$RUN_ROOT/manifest.tsv"
}

wait_storage() {
  local mode="$1" workload="$2" repeat="$3"
  echo
  echo "case: workload=$workload mode=$mode repeat=$repeat"
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  echo "  ./motivation/program3/start_storage_case.sh $mode"
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

validate_case() {
  local workload="$1" mode="$2" report="$3"
  python3 - "$workload" "$mode" "$report" "$MIN_WRITE_ATTAINMENT" <<'PY'
import json
import sys

workload, mode, path, min_write = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    report = json.load(stream)
meta = report.get("meta", {})
gpu = report.get("gpu_persistent", {})
throughput = report.get("throughput", {})
expected_early = mode == "early"
if meta.get("workload") != workload:
    raise SystemExit(f"{mode}: expected workload={workload}")
if meta.get("gpu_rdma_search_progression_mode") != "manual":
    raise SystemExit(f"{mode}: experiment did not use persistent manual mode")
if meta.get("gpu_query_beam_merge_policy") != "stable-run":
    raise SystemExit(f"{mode}: experiment did not use Stable-Run")
if int(meta.get("gpu_graph_issue_width", 0)) != int(meta.get("gpu_graph_commit_width", -1)):
    raise SystemExit(f"{mode}: issue width differs from commit width; speculative tail is not disabled")
if bool(meta.get("gpu_exact_frontier_early_issue", False)) != expected_early:
    raise SystemExit(f"{mode}: exact-frontier switch does not match case")
certificates = int(gpu.get("frontier_reusable_certificates", 0))
issued = int(gpu.get("frontier_reusable_issued_certificates", 0))
speculative = int(gpu.get("speculative_graph_reads", 0))
if expected_early and (certificates == 0 or issued == 0):
    raise SystemExit("early: no exact certificate/core wave was observed")
if not expected_early and (certificates != 0 or issued != 0):
    raise SystemExit("late: exact-frontier counters must remain zero")
if speculative != 0:
    raise SystemExit(f"{mode}: observed {speculative} speculative graph reads")
if workload == "mixed":
    attainment = float(throughput.get("write_rate_attainment_ratio", 0))
    if attainment < float(min_write):
        raise SystemExit(
            f"{mode}: write-rate attainment {attainment:.3f} is below {float(min_write):.3f}"
        )
print(f"[program3] valid workload={workload} mode={mode} certificates={certificates} issued={issued}")
PY
}

run_case() {
  local workload="$1" mode="$2" repeat="$3"
  local early=false
  [[ "$mode" == early ]] && early=true
  local report_dir="$RUN_ROOT/${workload}_${mode}_r${repeat}"
  local existing
  existing="$(awk -F '\t' -v w="$workload" -v m="$mode" -v r="$repeat" \
    '$1 == w && $2 == m && $3 == r { print $4; exit }' \
    "$RUN_ROOT/manifest.tsv")"
  if [[ -n "$existing" && -f "$existing" ]]; then
    echo "复用已完成 case: workload=$workload mode=$mode repeat=$repeat"
    return
  fi

  wait_storage "$mode" "$workload" "$repeat"
  mkdir -p "$report_dir"
  echo "[$(date --iso-8601=seconds)] workload=$workload mode=$mode repeat=$repeat"

  local -a workload_args
  if [[ "$workload" == mixed ]]; then
    workload_args=(
      WORKLOAD=mixed
      MIXED_MODE=write_rate_limited
      MIXED_WRITE_THREADS="$WRITE_THREADS"
      TARGET_QUERY_QPS=0
      TARGET_WRITE_QPS="$TARGET_WRITE_QPS"
      WRITE_INSERT_RATIO=1
      WRITE_UPSERT_RATIO=0
      WRITE_DELETE_RATIO=0
    )
  else
    workload_args=(WORKLOAD=query TARGET_QUERY_QPS=0 TARGET_WRITE_QPS=0)
  fi

  env \
    REPORT_DIR="$report_dir" \
    STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
    GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
    GPU_RDMA_SEARCH_PROGRESSION_MODE=manual \
    GPU_EXACT_FRONTIER_EARLY_ISSUE="$early" \
    GPU_GRAPH_COMMIT_WIDTH="$GPU_COMMIT_WIDTH" \
    GPU_GRAPH_ISSUE_WIDTH="$GPU_COMMIT_WIDTH" \
    GPU_QUERY_BEAM_MERGE_POLICY=stable-run \
    ENABLE_BREAKDOWN=false \
    BENCHMARK_MODE=time \
    WARMUP_SECONDS="$WARMUP_SECONDS" \
    MEASURE_SECONDS="$MEASURE_SECONDS" \
    BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
    BENCHMARK_CLIENT_THREAD_CAP="$CLIENT_THREAD_CAP" \
    RECALL_QUERIES="$RECALL_QUERIES" \
    "${workload_args[@]}" \
      "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | \
        tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'sift100m_*.json' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 { print $2 }')"
  [[ -n "$report" ]] || { echo "missing JSON report" >&2; exit 1; }
  validate_case "$workload" "$mode" "$report"
  record_case "$workload" "$mode" "$repeat" "$report"
}

run_workload() {
  local workload="$1"
  local repeat mode
  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    if (( repeat % 2 == 1 )); then
      for mode in late early; do run_case "$workload" "$mode" "$repeat"; done
    else
      for mode in early late; do run_case "$workload" "$mode" "$repeat"; done
    fi
  done
}

verify_exactness() {
  local verify_dir="$RUN_ROOT/identity"
  mkdir -p "$verify_dir"
  cmake --build "$PROJECT_DIR/build" -j --target \
    gpu_beam_merge_equivalence_test gpu_stable_frontier_preview_test
  {
    CUDA_VISIBLE_DEVICES="$VERIFY_CUDA_VISIBLE_DEVICES" \
      "$PROJECT_DIR/build/test/gpu_beam_merge_equivalence_test"
    CUDA_VISIBLE_DEVICES="$VERIFY_CUDA_VISIBLE_DEVICES" \
      "$PROJECT_DIR/build/test/gpu_stable_frontier_preview_test"
  } 2>&1 | tee "$verify_dir/exactness.log"
  if grep -q '^SKIP:' "$verify_dir/exactness.log"; then
    echo "exactness test was skipped; set CUDA_VISIBLE_DEVICES to an available GPU" >&2
    exit 1
  fi
}

summarize() {
  python3 "$PROGRAM_DIR/summarize_program3.py" "$RUN_ROOT"
  python3 "$PROGRAM_DIR/plot_program3.py" "$RUN_ROOT"
  echo "结果目录：$RUN_ROOT"
}

case "$ACTION" in
  all)
    verify_exactness
    for workload in $WORKLOADS; do run_workload "$workload"; done
    summarize
    ;;
  query|mixed) run_workload "$ACTION" ;;
  verify) verify_exactness ;;
  summarize) summarize ;;
  *)
    echo "usage: $0 [all|query|mixed|verify|summarize]" >&2
    exit 2
    ;;
esac
