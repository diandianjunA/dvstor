#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build}"
STORAGE_PID_DIR="${STORAGE_PID_DIR:-$EXPERIMENT_DIR/pids}"

case "${1:-}" in
  baseline)
    case_name=baseline
    profile=04_gpu_persistent_gpunetio_baseline
    update_mode=coupled
    access_mode=fixed
    progression_mode=manual
    ;;
  program1)
    case_name=program1
    profile=04_gpu_persistent_gpunetio_baseline
    update_mode=decoupled
    access_mode=fixed
    progression_mode=manual
    ;;
  program3)
    case_name=program3
    profile=04_gpu_persistent_gpunetio_baseline
    update_mode=decoupled
    access_mode=fixed
    progression_mode=decoupled
    ;;
  full)
    case_name=full
    profile=04_gpu_persistent_gpunetio
    update_mode=decoupled
    access_mode=adaptive
    progression_mode=decoupled
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
    echo "usage: $0 baseline|program1|program3|full|status|stop" >&2
    exit 2
    ;;
esac

[[ -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]] || {
  echo "storage build not configured: $STORAGE_BUILD_DIR" >&2
  echo "set STORAGE_BUILD_DIR to the storage node build directory" >&2
  exit 1
}

export BUILD_DIR="$STORAGE_BUILD_DIR"
export PID_DIR="$STORAGE_PID_DIR"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode"
export GPU_DYNAMIC_GRAPH_ACCESS_MODE="$access_mode"
export GPU_RDMA_SEARCH_PROGRESSION_MODE="$progression_mode"
# Clear any manual motivation-test residue. In formal decoupled mode the
# complete progression bundle is selected by the top-level mode itself.
export GPU_EXACT_FRONTIER_EARLY_ISSUE=false
export GPU_GRAPH_COMMIT_WIDTH=16
export GPU_GRAPH_ISSUE_WIDTH=16
export GPU_QUERY_BEAM_MERGE_POLICY=stable-run
export ENABLE_BREAKDOWN=false
export LOG_DIR="${STORAGE_LOG_ROOT:-$PROGRAM_DIR/storage_logs}/$(date +%Y%m%d_%H%M%S)_$case_name"

source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$profile"
validate_vector_id_namespace_size
for ((node = 1; node <= SHARDS; ++node)); do
  validate_index_metadata storage "$node"
done

mkdir -p "$LOG_DIR" "$STORAGE_PID_DIR"
PID_DIR="$STORAGE_PID_DIR" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
cmake --build "$STORAGE_BUILD_DIR" -j --target dvstor_memory_node
"$EXPERIMENT_DIR/start_all_memory_nodes.sh" "$profile"

sleep "${STORAGE_STARTUP_PROBE_SECONDS:-2}"
for ((node = 1; node <= SHARDS; ++node)); do
  pid_file="$STORAGE_PID_DIR/memory_node_${node}.pid"
  [[ -s "$pid_file" ]] || {
    echo "node $node has no PID file" >&2
    exit 1
  }
  kill -0 "$(<"$pid_file")" 2>/dev/null || {
    echo "node $node exited; inspect $LOG_DIR" >&2
    exit 1
  }
done

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$case_name" "$profile" "$update_mode" "$access_mode" \
  "$progression_mode" "$LOG_DIR" "$(date --iso-8601=seconds)" \
  > "$PROGRAM_DIR/.storage_ready.tsv"

echo "storage ready: case=$case_name profile=$profile"
echo "modes: update=$update_mode access=$access_mode progression=$progression_mode"
echo "logs: $LOG_DIR"
