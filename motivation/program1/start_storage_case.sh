#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build-storage}"
STORAGE_PID_DIR="${STORAGE_PID_DIR:-$EXPERIMENT_DIR/pids}"
QUALITY_STAGE2_DELAY_MS="${QUALITY_STAGE2_DELAY_MS:-15000}"

case "${1:-}" in
  baseline|coupled_one_stage|coupled)
    case_name=baseline
    update_mode=coupled
    stage2_delay_ms=0
    ;;
  solution|two_stage|decoupled)
    case_name=solution
    update_mode=decoupled
    stage2_delay_ms=0
    ;;
  quality|stage1_only)
    case_name=quality
    update_mode=decoupled
    stage2_delay_ms="$QUALITY_STAGE2_DELAY_MS"
    ;;
  status)
    for file in "$STORAGE_PID_DIR"/memory_node_*.pid; do
      [[ -s "$file" ]] || continue
      pid="$(<"$file")"
      state=stopped
      kill -0 "$pid" 2>/dev/null && state=running
      printf '%s pid=%s %s\n' "$(basename "$file")" "$pid" "$state"
    done
    exit 0
    ;;
  stop)
    PID_DIR="$STORAGE_PID_DIR" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
    exit 0
    ;;
  *)
    echo "usage: $0 baseline|solution|quality|status|stop" >&2
    exit 2
    ;;
esac

if [[ ! -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]]; then
  echo "storage build not configured: $STORAGE_BUILD_DIR" >&2
  exit 1
fi

export BUILD_DIR="$STORAGE_BUILD_DIR"
export PID_DIR="$STORAGE_PID_DIR"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode"
export STORAGE_OWNER_STAGE2_INITIAL_DELAY_MS="$stage2_delay_ms"
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
export ENABLE_BREAKDOWN=true
export LOG_DIR="${STORAGE_LOG_ROOT:-$PROGRAM_DIR/storage_logs}/$(date +%Y%m%d_%H%M%S)_$case_name"

# Validate all five storage artifacts before stopping the previous case.
source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$PROFILE"
validate_vector_id_namespace_size
for ((node = 1; node <= SHARDS; ++node)); do
  validate_index_metadata storage "$node"
done

mkdir -p "$LOG_DIR" "$STORAGE_PID_DIR"
PID_DIR="$STORAGE_PID_DIR" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
cmake --build "$STORAGE_BUILD_DIR" -j --target dvstor_memory_node
"$EXPERIMENT_DIR/start_all_memory_nodes.sh" "$PROFILE"

sleep "${STORAGE_STARTUP_PROBE_SECONDS:-2}"
for ((node = 1; node <= SHARDS; ++node)); do
  pid_file="$STORAGE_PID_DIR/memory_node_${node}.pid"
  [[ -s "$pid_file" ]] || { echo "node $node has no PID file" >&2; exit 1; }
  kill -0 "$(<"$pid_file")" 2>/dev/null || {
    echo "node $node exited; inspect $LOG_DIR" >&2
    exit 1
  }
done

echo "storage case ready: $case_name mode=$update_mode stage2_delay_ms=$stage2_delay_ms"
echo "logs: $LOG_DIR"

