#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/spacev100m"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
WARMUP_SECONDS="${WARMUP_SECONDS:-5}"
MEASURE_SECONDS="${MEASURE_SECONDS:-20}"
CLIENT_THREADS="${CLIENT_THREADS:-auto}"
CLIENT_THREAD_CAP="${CLIENT_THREAD_CAP:-512}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
TARGET_QUERY_QPS=0
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-500}"
WRITE_THREADS="${WRITE_THREADS:-16}"
MIN_DYNAMIC_EXPANDED="${MIN_DYNAMIC_EXPANDED:-100}"
MIN_DYNAMIC_SHARE="${MIN_DYNAMIC_SHARE:-0.001}"
MIN_WRITE_ATTAINMENT="${MIN_WRITE_ATTAINMENT:-0.95}"
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
  echo "  ./experiment/spacev100m/program2/start_storage_case.sh $case_name"
  read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
  [[ "$answer" != q && "$answer" != Q ]] || exit 130
}

validate_dynamic_case() {
  local case_name="$1"
  local report="$2"
  python3 - "$case_name" "$report" "$MIN_DYNAMIC_EXPANDED" \
    "$MIN_DYNAMIC_SHARE" "$MIN_WRITE_ATTAINMENT" <<'PY'
import json
import sys

case_name, report_path, min_dynamic, min_share, min_write = sys.argv[1:]
with open(report_path, encoding="utf-8") as stream:
    report = json.load(stream)
meta = report.get("meta", {})
throughput = report.get("throughput", {})
gpu = report.get("gpu_persistent", {})
if meta.get("workload") != "mixed":
    raise SystemExit(f"{case_name}: expected mixed workload")
if meta.get("mixed_dispatch_policy") != "write_rate_limited":
    raise SystemExit(f"{case_name}: expected write_rate_limited mixed workload")
dynamic = int(gpu.get("dynamic_expanded_parent_count", 0))
dynamic_share = float(gpu.get("dynamic_expanded_parent_ratio", 0))
write_attainment = float(throughput.get("write_rate_attainment_ratio", 0))
if dynamic < int(min_dynamic):
    raise SystemExit(
        f"{case_name}: only {dynamic} dynamic parents were expanded; "
        f"need >= {min_dynamic}. This is not a valid dynamic experiment."
    )
if dynamic_share < float(min_share):
    raise SystemExit(
        f"{case_name}: dynamic expansion share {100*dynamic_share:.4f}% is "
        f"below {100*float(min_share):.4f}%. Increase WARMUP_SECONDS or "
        "TARGET_WRITE_QPS; otherwise graph-read performance is still "
        "dominated by immutable nodes."
    )
if write_attainment < float(min_write):
    raise SystemExit(
        f"{case_name}: write-rate attainment {write_attainment:.3f} is below "
        f"{float(min_write):.3f}; lower TARGET_WRITE_QPS and rerun this case."
    )
print(
    f"[program2] valid dynamic case={case_name} "
    f"dynamic_expanded={dynamic} "
    f"dynamic_share={100*dynamic_share:.3f}% "
    f"write_attainment={write_attainment:.3f}"
)
PY
}

run_dynamic_case() {
  local case_name="$1"
  local graph_policy="$2"
  local dynamic_extent="$3"
  local report_dir="$RUN_ROOT/$case_name"

  wait_storage "$case_name"
  mkdir -p "$report_dir"
  echo "[$(date --iso-8601=seconds)] dynamic mixed case=$case_name policy=$graph_policy"
  REPORT_DIR="$report_dir" \
  STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE=manual \
  GPU_QUERY_GRAPH_READ_POLICY="$graph_policy" \
  GPU_DYNAMIC_GRAPH_EXTENT="$dynamic_extent" \
  GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled \
  ENABLE_BREAKDOWN=false \
  WORKLOAD=mixed \
  MIXED_MODE=write_rate_limited \
  MIXED_WRITE_THREADS="$WRITE_THREADS" \
  TARGET_QUERY_QPS="$TARGET_QUERY_QPS" \
  TARGET_WRITE_QPS="$TARGET_WRITE_QPS" \
  WRITE_INSERT_RATIO=1 \
  WRITE_UPSERT_RATIO=0 \
  WRITE_DELETE_RATIO=0 \
  BENCHMARK_MODE=time \
  WARMUP_SECONDS="$WARMUP_SECONDS" \
  MEASURE_SECONDS="$MEASURE_SECONDS" \
  BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
  BENCHMARK_CLIENT_THREAD_CAP="$CLIENT_THREAD_CAP" \
  RECALL_QUERIES="$RECALL_QUERIES" \
    "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE" 2>&1 | \
      tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'spacev100m_*.json' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 { print $2 }')"
  [[ -n "$report" ]] || {
    echo "missing JSON report for $case_name" >&2
    exit 1
  }
  validate_dynamic_case "$case_name" "$report"
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
  local build_jobs="${BUILD_JOBS:-16}"
  if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]] || ((build_jobs > 32)); then
    echo "BUILD_JOBS must be an integer in [1,32]: $build_jobs" >&2
    exit 1
  fi
  local probe_build_dir="${BUILD_DIR:-$PROJECT_DIR/build}"
  if [[ ! -f "$probe_build_dir/CMakeCache.txt" ]]; then
    echo "compute build directory is not configured: $probe_build_dir" >&2
    exit 1
  fi
  if grep -q '^DVSTOR_STORAGE_NODE_ONLY:BOOL=ON$' "$probe_build_dir/CMakeCache.txt"; then
    echo "RDMA probe requires a compute build; storage-only BUILD_DIR: $probe_build_dir" >&2
    exit 1
  fi
  cmake --build "$probe_build_dir" -j "$build_jobs" --target dvstor_gpunetio_loopback_probe

  echo "[$(date --iso-8601=seconds)] RDMA protocol probe"
  DVSTOR_GPUNETIO_PAYLOAD_SWEEP=1 \
  DVSTOR_GPUNETIO_PAYLOAD_BYTES="400 832" \
  DVSTOR_GPUNETIO_PAIRED_BODY_BYTES="384" \
  DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST="1 32 160" \
  DVSTOR_GPUNETIO_PAYLOAD_REPEATS="${PROBE_REPEATS:-3}" \
  DVSTOR_GPUNETIO_PAYLOAD_WARMUP_ITERATIONS="${PROBE_WARMUP_ITERATIONS:-32}" \
  DVSTOR_GPUNETIO_PAYLOAD_ITERATIONS="${PROBE_ITERATIONS:-512}" \
  DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS="${PROBE_BATCH_READS:-16}" \
    "$probe_build_dir/dvstor_gpunetio_loopback_probe" \
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
    # All three storage runs publish the same dynamic extent tag. Fixed and
    # Header→Neighbor ignore it; keeping publication identical avoids changing
    # update-path work between policies.
    run_dynamic_case fixed fixed true
    run_dynamic_case header header-neighbor true
    run_dynamic_case live live-extent true
    summarize
    ;;
  fixed)
    run_dynamic_case fixed fixed true
    ;;
  header)
    run_dynamic_case header header-neighbor true
    ;;
  live)
    run_dynamic_case live live-extent true
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
