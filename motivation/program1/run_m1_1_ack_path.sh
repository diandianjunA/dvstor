#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
REPEATS="${REPEATS:-10}"
SCENARIOS="${SCENARIOS:-ack mixed}"
ACK_SECONDS="${ACK_SECONDS:-60}"
MIXED_SECONDS="${MIXED_SECONDS:-120}"
ACK_CLIENT_THREADS="${ACK_CLIENT_THREADS:-1}"
MIXED_CLIENT_THREADS="${MIXED_CLIENT_THREADS:-512}"
MIXED_READ_RATIO="${MIXED_READ_RATIO:-0.5}"
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
STORAGE_NODE_MODE="${STORAGE_NODE_MODE:-external}"
STORAGE_BEFORE_CASE_HOOK="${STORAGE_BEFORE_CASE_HOOK:-}"
ALLOW_SERVICE_RESTART="${ALLOW_SERVICE_RESTART:-0}"

if [[ "$SMOKE" == 1 ]]; then
  REPEATS=1
  ACK_SECONDS="${SMOKE_ACK_SECONDS:-3}"
  MIXED_SECONDS="${SMOKE_MIXED_SECONDS:-5}"
  WARMUP_SECONDS="${SMOKE_WARMUP_SECONDS:-1}"
  RECALL_QUERIES="${SMOKE_RECALL_QUERIES:-10}"
fi

for value_name in REPEATS ACK_SECONDS MIXED_SECONDS ACK_CLIENT_THREADS \
    MIXED_CLIENT_THREADS WARMUP_SECONDS RECALL_QUERIES; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || ((value == 0)); then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done

for scenario in $SCENARIOS; do
  case "$scenario" in ack|mixed) ;; *)
    echo "unsupported scenario: $scenario (expected ack and/or mixed)" >&2
    exit 2
  esac
done
case "$STORAGE_NODE_MODE" in
  external|local) ;;
  *)
    echo "STORAGE_NODE_MODE must be external or local: $STORAGE_NODE_MODE" >&2
    exit 2
    ;;
esac
if [[ -n "$STORAGE_BEFORE_CASE_HOOK" && ! -x "$STORAGE_BEFORE_CASE_HOOK" ]]; then
  echo "STORAGE_BEFORE_CASE_HOOK must be executable: $STORAGE_BEFORE_CASE_HOOK" >&2
  exit 2
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/m1_1_$RUN_STAMP}"

case_order() {
  local repeat="$1"
  if ((repeat % 2 == 1)); then
    echo 'coupled_one_stage two_stage'
  else
    echo 'two_stage coupled_one_stage'
  fi
}

mode_for_case() {
  case "$1" in
    coupled_one_stage) echo coupled ;;
    two_stage) echo decoupled ;;
    *) return 2 ;;
  esac
}

print_matrix() {
  local repeat scenario case_name order=0
  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    for scenario in $SCENARIOS; do
      order=0
      for case_name in $(case_order "$repeat"); do
        ((order += 1))
        printf 'repeat=%02d scenario=%s order=%d case=%s update_mode=%s\n' \
          "$repeat" "$scenario" "$order" "$case_name" \
          "$(mode_for_case "$case_name")"
      done
    done
  done
}

if [[ "$DRY_RUN" == 1 ]]; then
  echo "run_root=$RUN_ROOT profile=$PROFILE storage_node_mode=$STORAGE_NODE_MODE"
  print_matrix
  exit 0
fi

if [[ "$STORAGE_NODE_MODE" == local && "$ALLOW_SERVICE_RESTART" != 1 ]]; then
  echo "This experiment restarts all DVStor memory nodes." >&2
  echo "Re-run with STORAGE_NODE_MODE=local ALLOW_SERVICE_RESTART=1 after confirming the service is idle." >&2
  exit 2
fi
if [[ "$STORAGE_NODE_MODE" == external && -z "$STORAGE_BEFORE_CASE_HOOK" && ! -t 0 ]]; then
  echo "external storage mode needs an interactive terminal or STORAGE_BEFORE_CASE_HOOK" >&2
  exit 2
fi

if [[ -e "$RUN_ROOT" ]]; then
  echo "run root already exists; choose a new RUN_STAMP or RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
printf 'repeat\tscenario\torder\tcase\tupdate_mode\treport_dir\n' > "$RUN_ROOT/manifest.tsv"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "project_dir=$PROJECT_DIR"
  echo "profile=$PROFILE"
  echo "storage_node_mode=$STORAGE_NODE_MODE"
  echo "storage_before_case_hook=${STORAGE_BEFORE_CASE_HOOK:-manual}"
  echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git -C "$PROJECT_DIR" status --short 2>/dev/null || true
  echo "git_status_end"
  command -v nvidia-smi >/dev/null && nvidia-smi -L || true
  command -v ibv_devinfo >/dev/null && ibv_devinfo -l || true
} > "$RUN_ROOT/provenance.txt"

stop_nodes() {
  local pid_dir="$1"
  local pid_file pid
  local -a stopped_pids=()
  for pid_file in "$pid_dir"/memory_node_*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && stopped_pids+=("$pid")
  done
  PID_DIR="$pid_dir" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
  for pid in "${stopped_pids[@]}"; do
    for _ in {1..100}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "memory node pid $pid did not stop within 10 seconds" >&2
      return 1
    fi
  done
}

cleanup() {
  if [[ "$STORAGE_NODE_MODE" == local ]]; then
    stop_nodes "$RUN_ROOT/current_pids" || true
  fi
}
trap cleanup EXIT INT TERM

snapshot_nic_counters() {
  local output="$1" counter value
  printf 'counter\tvalue\n' > "$output"
  for counter in /sys/class/infiniband/*/ports/*/counters/* \
      /sys/class/infiniband/*/ports/*/hw_counters/*; do
    [[ -r "$counter" ]] || continue
    value="$(<"$counter")"
    printf '%s\t%s\n' "$counter" "$value" >> "$output"
  done
}

prepare_external_storage() {
  local repeat="$1" scenario="$2" order="$3" case_name="$4"
  local update_mode="$5" case_root="$6"
  local storage_log="$case_root/storage_before_case.log"

  if [[ -n "$STORAGE_BEFORE_CASE_HOOK" ]]; then
    STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode" \
    GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
    GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled \
    ENABLE_BREAKDOWN=true \
      "$STORAGE_BEFORE_CASE_HOOK" \
        "$repeat" "$scenario" "$order" "$case_name" "$case_root" \
        >"$storage_log" 2>&1
    return
  fi

  {
    echo
    echo "================================================================"
    echo "请在存储节点停止上一 case，并从同一静态索引重新启动全部 memory node："
    echo "  repeat=$repeat scenario=$scenario order=$order case=$case_name"
    echo "  STORAGE_OWNER_UPDATE_COMPLETION_MODE=$update_mode"
    echo "  GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive"
    echo "  GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled"
    echo "  ENABLE_BREAKDOWN=true"
    echo
    echo "存储节点配套命令："
    printf '  cd %q\n' "$PROJECT_DIR"
    printf '  PROFILE=%q ./motivation/program1/start_storage_case.sh %q\n' \
      "$PROFILE" "$case_name"
    echo "================================================================"
  } | tee "$storage_log"
  read -r -p "存储节点已全部就绪后按 Enter；输入 q 终止实验: " answer
  if [[ "$answer" == q || "$answer" == Q ]]; then
    echo "experiment stopped before case $case_name" >&2
    exit 130
  fi
}

run_case() {
  local repeat="$1" scenario="$2" order="$3" case_name="$4"
  local update_mode case_root report_root log_root pid_root measure_seconds
  update_mode="$(mode_for_case "$case_name")"
  case_root="$RUN_ROOT/repeat_$(printf '%02d' "$repeat")/$scenario/$case_name"
  report_root="$case_root/reports"
  log_root="$case_root/logs"
  pid_root="$RUN_ROOT/current_pids"
  mkdir -p "$report_root" "$log_root" "$pid_root"

  export REPORT_DIR="$report_root"
  export LOG_DIR="$log_root"
  export PID_DIR="$pid_root"
  export STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode"
  export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
  export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
  export ENABLE_BREAKDOWN=true

  if [[ "$STORAGE_NODE_MODE" == local ]]; then
    stop_nodes "$pid_root"
    # Also stop nodes started through the repository's default PID directory.
    stop_nodes "$EXPERIMENT_DIR/pids"
  else
    prepare_external_storage \
      "$repeat" "$scenario" "$order" "$case_name" "$update_mode" "$case_root"
  fi

  {
    echo "[$(date --iso-8601=seconds)] start repeat=$repeat scenario=$scenario order=$order case=$case_name"
    if [[ "$STORAGE_NODE_MODE" == local ]]; then
      "$EXPERIMENT_DIR/start_all_memory_nodes.sh" "$PROFILE"
    fi
    snapshot_nic_counters "$case_root/nic_before.tsv"

    if [[ "$scenario" == ack ]]; then
      measure_seconds="$ACK_SECONDS"
      WORKLOAD=insert \
      BENCHMARK_CLIENT_THREADS="$ACK_CLIENT_THREADS" \
      WARMUP_SECONDS="$WARMUP_SECONDS" \
      MEASURE_SECONDS="$measure_seconds" \
      RECALL_QUERIES="$RECALL_QUERIES" \
        "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE"
    else
      measure_seconds="$MIXED_SECONDS"
      WORKLOAD=mixed \
      MIXED_MODE=fixed_threads \
      READ_RATIO="$MIXED_READ_RATIO" \
      BENCHMARK_CLIENT_THREADS="$MIXED_CLIENT_THREADS" \
      WARMUP_SECONDS="$WARMUP_SECONDS" \
      MEASURE_SECONDS="$measure_seconds" \
      RECALL_QUERIES="$RECALL_QUERIES" \
        "$EXPERIMENT_DIR/run_breakdown.sh" "$PROFILE"
    fi
    snapshot_nic_counters "$case_root/nic_after.tsv"
    echo "[$(date --iso-8601=seconds)] finish"
  } 2>&1 | tee "$case_root/driver.log"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$scenario" "$order" "$case_name" "$update_mode" \
    "$report_root" >> "$RUN_ROOT/manifest.tsv"
  if [[ "$STORAGE_NODE_MODE" == local ]]; then
    stop_nodes "$pid_root"
  fi
}

repeat=0
for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
  for scenario in $SCENARIOS; do
    order=0
    for case_name in $(case_order "$repeat"); do
      ((order += 1))
      run_case "$repeat" "$scenario" "$order" "$case_name"
    done
  done
done

trap - EXIT INT TERM
cleanup
python3 "$PROGRAM_DIR/summarize_m1_1.py" "$RUN_ROOT"
echo "M1.1 complete: $RUN_ROOT"
