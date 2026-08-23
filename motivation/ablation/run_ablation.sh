#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

# These are deliberately the publication settings used by
# experiment/run_breakdown.sh and by the reference full-system report.
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
CLIENT_THREADS="${CLIENT_THREADS:-auto}"
CLIENT_THREAD_CAP="${CLIENT_THREAD_CAP:-1024}"
READ_RATIO="${READ_RATIO:-0.5}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/ablation_$(date +%Y%m%d_%H%M%S)}"
REFERENCE_REPORT="${REFERENCE_REPORT:-$EXPERIMENT_DIR/reports/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260819_101650.json}"
ACTION="${1:-all}"
MANIFEST="$RUN_ROOT/manifest.tsv"

mkdir -p "$RUN_ROOT"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'case\tcode\tprofile\tupdate_mode\taccess_mode\tprogression_mode\treport\n' > "$MANIFEST"
fi

case_config() {
  local case_name="$1"
  case "$case_name" in
    baseline)
      CASE_CODE=000
      CASE_PROFILE=04_gpu_persistent_gpunetio_baseline
      CASE_UPDATE=coupled
      CASE_ACCESS=fixed
      CASE_PROGRESSION=coupled
      ;;
    program1)
      CASE_CODE=100
      CASE_PROFILE=04_gpu_persistent_gpunetio_baseline
      CASE_UPDATE=decoupled
      CASE_ACCESS=fixed
      CASE_PROGRESSION=coupled
      ;;
    program2)
      CASE_CODE=110
      CASE_PROFILE=04_gpu_persistent_gpunetio_baseline
      CASE_UPDATE=decoupled
      CASE_ACCESS=adaptive
      CASE_PROGRESSION=coupled
      ;;
    full)
      CASE_CODE=111
      CASE_PROFILE=04_gpu_persistent_gpunetio
      CASE_UPDATE=decoupled
      CASE_ACCESS=adaptive
      CASE_PROGRESSION=decoupled
      ;;
    *)
      echo "unknown ablation case: $case_name" >&2
      return 2
      ;;
  esac
}

record_case() {
  local case_name="$1"
  local report="$2"
  local temporary="$MANIFEST.tmp"
  awk -F '\t' -v name="$case_name" 'NR == 1 || $1 != name' \
    "$MANIFEST" > "$temporary"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_name" "$CASE_CODE" "$CASE_PROFILE" "$CASE_UPDATE" \
    "$CASE_ACCESS" "$CASE_PROGRESSION" "$report" >> "$temporary"
  mv "$temporary" "$MANIFEST"
}

wait_storage() {
  local case_name="$1"
  echo
  echo "请先在存储节点运行："
  echo "  cd $PROJECT_DIR"
  echo "  ./motivation/ablation/start_storage_case.sh $case_name"
  if [[ "${NO_PROMPT:-0}" != 1 ]]; then
    read -r -p "存储节点显示 ready 后按 Enter（输入 q 退出）: " answer
    [[ "$answer" != q && "$answer" != Q ]] || exit 130
  fi

  # The C++ startup handshake is authoritative. This shared-filesystem marker
  # catches the common mistake of starting the preceding case before the much
  # larger GPU/index bootstrap begins.
  local ready_file="$PROGRAM_DIR/.storage_ready.tsv"
  if [[ -f "$ready_file" ]]; then
    local ready_case ready_update ready_access ready_progression
    IFS=$'\t' read -r ready_case _ ready_update ready_access \
      ready_progression _ _ < "$ready_file"
    if [[ "$ready_case" != "$case_name" ||
          "$ready_update" != "$CASE_UPDATE" ||
          "$ready_access" != "$CASE_ACCESS" ||
          "$ready_progression" != "$CASE_PROGRESSION" ]]; then
      echo "storage ready marker does not match requested case=$case_name" >&2
      echo "marker: $(<"$ready_file")" >&2
      return 1
    fi
  else
    echo "[ablation] 未发现共享 ready 标记；将由计算/存储启动契约校验模式。"
  fi
}

validate_report() {
  local case_name="$1"
  local report="$2"
  python3 - "$case_name" "$CASE_CODE" "$CASE_UPDATE" "$CASE_ACCESS" \
    "$CASE_PROGRESSION" "$WARMUP_SECONDS" "$MEASURE_SECONDS" \
    "$READ_RATIO" "$report" <<'PY'
import json
import math
import sys

(case_name, code, update, access, progression, warmup, measure,
 read_ratio, report_path) = sys.argv[1:]
with open(report_path, encoding="utf-8") as stream:
    report = json.load(stream)

meta = report.get("meta", {})
system = meta.get("system_variant", {})
modes = system.get("resolved_modes", {})
expected = {
    "storage_owner_update_completion_mode": update,
    "gpu_dynamic_graph_access_mode": access,
    "gpu_rdma_search_progression_mode": progression,
}
errors = []
for key, wanted in expected.items():
    if modes.get(key) != wanted:
        errors.append(f"{key}={modes.get(key)!r}, expected {wanted!r}")
if meta.get("workload") != "mixed":
    errors.append(f"workload={meta.get('workload')!r}, expected 'mixed'")
if meta.get("mixed_dispatch_policy") != "fixed_threads":
    errors.append("mixed dispatch policy is not fixed_threads")
if int(meta.get("warmup_seconds", -1)) != int(warmup):
    errors.append("warmup duration differs from the requested setting")
if int(meta.get("measure_seconds", -1)) != int(measure):
    errors.append("measurement duration differs from the requested setting")
if not math.isclose(float(meta.get("read_ratio", -1)), float(read_ratio)):
    errors.append("read ratio differs from the requested setting")
if any(float(meta.get(key, -1)) != value for key, value in (
        ("write_insert_ratio", 1.0),
        ("write_upsert_ratio", 0.0),
        ("write_delete_ratio", 0.0))):
    errors.append("write mix is not append-only 1/0/0")
if code == "111" and system.get("profile_name") != "04_gpu_persistent_gpunetio":
    errors.append("111 is not using the formal full profile")
if errors:
    raise SystemExit(case_name + ": invalid ablation report:\n  - " +
                     "\n  - ".join(errors))

throughput = report.get("throughput", {})
print(
    f"[ablation] valid case={case_name} code={code} "
    f"query_qps={float(throughput.get('query_ops_per_sec', 0)):.1f} "
    f"write_qps={float(throughput.get('write_ops_per_sec', 0)):.1f} "
    f"total_qps={float(throughput.get('total_ops_per_sec', 0)):.1f}"
)
PY
}

run_case() {
  local case_name="$1"
  case_config "$case_name"
  wait_storage "$case_name"

  local report_dir="$RUN_ROOT/$case_name"
  mkdir -p "$report_dir"
  echo "[$(date --iso-8601=seconds)] case=$case_name code=$CASE_CODE profile=$CASE_PROFILE modes=$CASE_UPDATE/$CASE_ACCESS/$CASE_PROGRESSION"

  REPORT_DIR="$report_dir" \
  STORAGE_OWNER_UPDATE_COMPLETION_MODE="$CASE_UPDATE" \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE="$CASE_ACCESS" \
  GPU_RDMA_SEARCH_PROGRESSION_MODE="$CASE_PROGRESSION" \
  GPU_EXACT_FRONTIER_EARLY_ISSUE=false \
  ENABLE_BREAKDOWN=false \
  WORKLOAD=mixed \
  MIXED_MODE=fixed_threads \
  TARGET_QUERY_QPS=0 \
  TARGET_WRITE_QPS=0 \
  MIXED_WRITE_THREADS=0 \
  READ_RATIO="$READ_RATIO" \
  WRITE_INSERT_RATIO=1 \
  WRITE_UPSERT_RATIO=0 \
  WRITE_DELETE_RATIO=0 \
  BENCHMARK_MODE=time \
  WARMUP_SECONDS="$WARMUP_SECONDS" \
  MEASURE_SECONDS="$MEASURE_SECONDS" \
  BENCHMARK_CLIENT_THREADS="$CLIENT_THREADS" \
  BENCHMARK_CLIENT_THREAD_CAP="$CLIENT_THREAD_CAP" \
  RECALL_QUERIES="$RECALL_QUERIES" \
  RECALL_MODE=all \
  RECALL_BASE_ID_LIMIT=0 \
    "$EXPERIMENT_DIR/run_breakdown.sh" "$CASE_PROFILE" 2>&1 | \
      tee "$report_dir/driver.log"

  local report
  report="$(find "$report_dir" -type f -name 'sift100m_*.json' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 { print $2 }')"
  [[ -n "$report" ]] || {
    echo "missing JSON report for $case_name" >&2
    return 1
  }
  validate_report "$case_name" "$report"
  record_case "$case_name" "$report"
}

summarize() {
  local args=("$RUN_ROOT")
  [[ ! -f "$REFERENCE_REPORT" ]] || args+=("$REFERENCE_REPORT")
  python3 "$PROGRAM_DIR/summarize_ablation.py" "${args[@]}"
  python3 "$PROGRAM_DIR/plot_ablation.py" "$RUN_ROOT"
  echo "结果目录：$RUN_ROOT"
  echo "报告：$RUN_ROOT/消融实验分析报告.md"
}

case "$ACTION" in
  all)
    run_case baseline
    run_case program1
    run_case program2
    run_case full
    summarize
    ;;
  baseline|program1|program2|full)
    run_case "$ACTION"
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 [all|baseline|program1|program2|full|summarize]" >&2
    exit 2
    ;;
esac
