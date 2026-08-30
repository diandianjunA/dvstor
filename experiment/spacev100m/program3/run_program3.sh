#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/spacev100m"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
MOTIVATION_WIDTHS="${MOTIVATION_WIDTHS:-1 4 8 16}"
PERF_WIDTH="${PERF_WIDTH:-16}"
PERF_WORKLOAD="${PERF_WORKLOAD:-query}"
WARMUP_SECONDS="${WARMUP_SECONDS:-5}"
MOTIVATION_SECONDS="${MOTIVATION_SECONDS:-10}"
PERFORMANCE_SECONDS="${PERFORMANCE_SECONDS:-20}"
CLIENT_THREADS="${CLIENT_THREADS:-auto}"
CLIENT_THREAD_CAP="${CLIENT_THREAD_CAP:-512}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-500}"
WRITE_THREADS="${WRITE_THREADS:-16}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program3_$(date +%Y%m%d_%H%M%S)}"
ACTION="${1:-all}"

mkdir -p "$RUN_ROOT"
MANIFEST="$RUN_ROOT/manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'kind\tlabel\twidth\treport\n' > "$MANIFEST"
fi

record_case() {
  local kind="$1" label="$2" width="$3" report="$4"
  local temporary="$MANIFEST.tmp"
  awk -F '\t' -v k="$kind" -v l="$label" -v w="$width" \
    'NR == 1 || !($1 == k && $2 == l && $3 == w)' "$MANIFEST" > "$temporary"
  printf '%s\t%s\t%s\t%s\n' "$kind" "$label" "$width" "$report" >> "$temporary"
  mv "$temporary" "$MANIFEST"
}

existing_report() {
  local kind="$1" label="$2" width="$3"
  awk -F '\t' -v k="$kind" -v l="$label" -v w="$width" \
    '$1 == k && $2 == l && $3 == w { print $4; exit }' "$MANIFEST"
}

wait_storage() {
  local mode="$1" width="$2" label="$3"
  echo
  echo "case: $label (commit/issue width=$width)"
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  echo "  GPU_COMMIT_WIDTH=$width ./experiment/spacev100m/program3/start_storage_case.sh $mode"
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

validate_report() {
  local kind="$1" mode="$2" width="$3" report="$4"
  python3 - "$kind" "$mode" "$width" "$report" <<'PY'
import json, sys
kind, mode, width, path = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))
m, g = d.get("meta", {}), d.get("gpu_persistent", {})
r = m.get("system_variant", {}).get("resolved_modes", {})
if r.get("gpu_rdma_search_progression_mode") != "manual":
    raise SystemExit("case did not use persistent manual progression")
if m.get("gpu_query_beam_merge_policy") != "stable-run":
    raise SystemExit("case did not use Stable-Run")
if int(m.get("gpu_graph_commit_width", 0)) != int(width) or int(m.get("gpu_graph_issue_width", 0)) != int(width):
    raise SystemExit("commit/issue width mismatch")
early = mode == "early"
if bool(m.get("gpu_exact_frontier_early_issue", False)) != early:
    raise SystemExit("exact-frontier switch mismatch")
cert = int(g.get("frontier_reusable_certificates", 0))
issued = int(g.get("frontier_reusable_issued_certificates", 0))
if early and (cert == 0 or issued == 0):
    raise SystemExit("early case produced no exact certificate/core issue")
if not early and (cert or issued):
    raise SystemExit("late case unexpectedly entered exact-frontier path")
if int(g.get("speculative_graph_reads", 0)) != 0:
    raise SystemExit("speculative graph reads must be zero")
if kind == "motivation" and m.get("workload") != "query":
    raise SystemExit("motivation sweep must use query-only workload")
print(f"[program3] valid kind={kind} mode={mode} width={width}")
PY
}

run_case() {
  local kind="$1" label="$2" mode="$3" width="$4" workload="$5" seconds="$6"
  local report_dir="$RUN_ROOT/$label"
  local existing
  existing="$(existing_report "$kind" "$label" "$width")"
  if [[ -n "$existing" && -f "$existing" ]]; then
    echo "复用已完成 case: $label"
    return
  fi

  local recovered=""
  if [[ -d "$report_dir" ]]; then
    recovered="$(find "$report_dir" -type f -name 'spacev100m_*.json' \
      -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 {print $2}')"
  fi
  if [[ -n "$recovered" ]]; then
    validate_report "$kind" "$mode" "$width" "$recovered"
    record_case "$kind" "$label" "$width" "$recovered"
    echo "恢复已有 case: $label"
    return
  fi

  wait_storage "$mode" "$width" "$label"
  mkdir -p "$report_dir"
  local early=false
  [[ "$mode" == early ]] && early=true
  local -a workload_args
  if [[ "$workload" == mixed ]]; then
    workload_args=(WORKLOAD=mixed MIXED_MODE=write_rate_limited
      MIXED_WRITE_THREADS="$WRITE_THREADS" TARGET_QUERY_QPS=0
      TARGET_WRITE_QPS="$TARGET_WRITE_QPS" WRITE_INSERT_RATIO=1
      WRITE_UPSERT_RATIO=0 WRITE_DELETE_RATIO=0)
  else
    workload_args=(WORKLOAD=query TARGET_QUERY_QPS=0 TARGET_WRITE_QPS=0)
  fi

  env REPORT_DIR="$report_dir" \
    STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
    GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
    GPU_RDMA_SEARCH_PROGRESSION_MODE=manual \
    GPU_EXACT_FRONTIER_EARLY_ISSUE="$early" \
    GPU_GRAPH_COMMIT_WIDTH="$width" GPU_GRAPH_ISSUE_WIDTH="$width" \
    GPU_QUERY_BEAM_MERGE_POLICY=stable-run ENABLE_BREAKDOWN=false \
    BENCHMARK_MODE=time WARMUP_SECONDS="$WARMUP_SECONDS" \
    MEASURE_SECONDS="$seconds" BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
    BENCHMARK_CLIENT_THREAD_CAP="$CLIENT_THREAD_CAP" \
    RECALL_QUERIES="$RECALL_QUERIES" "${workload_args[@]}" \
      "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'spacev100m_*.json' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 {print $2}')"
  [[ -n "$report" ]] || { echo "missing JSON report for $label" >&2; exit 1; }
  validate_report "$kind" "$mode" "$width" "$report"
  record_case "$kind" "$label" "$width" "$report"
}

run_motivation() {
  local width
  for width in $MOTIVATION_WIDTHS; do
    run_case motivation "motivation_c${width}" early "$width" query "$MOTIVATION_SECONDS"
  done
}

run_performance() {
  run_case performance "performance_late_c${PERF_WIDTH}" late "$PERF_WIDTH" "$PERF_WORKLOAD" "$PERFORMANCE_SECONDS"
  run_case performance "performance_early_c${PERF_WIDTH}" early "$PERF_WIDTH" "$PERF_WORKLOAD" "$PERFORMANCE_SECONDS"
}

summarize() {
  python3 "$PROGRAM_DIR/summarize_program3.py" "$RUN_ROOT"
  python3 "$PROGRAM_DIR/plot_program3.py" "$RUN_ROOT"
  echo "结果目录：$RUN_ROOT"
}

case "$ACTION" in
  all) run_motivation; run_performance; summarize ;;
  motivation) run_motivation ;;
  performance) run_performance ;;
  summarize) summarize ;;
  *) echo "usage: $0 [all|motivation|performance|summarize]" >&2; exit 2 ;;
esac
