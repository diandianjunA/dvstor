#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build}"
STORAGE_PID_DIR="${STORAGE_PID_DIR:-$EXPERIMENT_DIR/pids}"

case "${1:-}" in
  fixed)
    case_name=fixed
    graph_policy=fixed
    dynamic_extent=false
    ;;
  live|probe)
    case_name="${1}"
    graph_policy=live-extent
    dynamic_extent=true
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
    echo "usage: $0 fixed|live|probe|status|stop" >&2
    exit 2
    ;;
esac

[[ -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]] || {
  echo "storage build not configured: $STORAGE_BUILD_DIR" >&2
  exit 1
}

export BUILD_DIR="$STORAGE_BUILD_DIR"
export PID_DIR="$STORAGE_PID_DIR"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=manual
export GPU_QUERY_GRAPH_READ_POLICY="$graph_policy"
export GPU_DYNAMIC_GRAPH_EXTENT="$dynamic_extent"
export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
export ENABLE_BREAKDOWN=false
export LOG_DIR="${STORAGE_LOG_ROOT:-$PROGRAM_DIR/storage_logs}/$(date +%Y%m%d_%H%M%S)_$case_name"

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

echo "storage case ready: $case_name graph_policy=$graph_policy dynamic_extent=$dynamic_extent"
echo "logs: $LOG_DIR"
