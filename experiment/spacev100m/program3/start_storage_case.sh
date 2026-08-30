#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/spacev100m"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build}"
STORAGE_PID_DIR="${STORAGE_PID_DIR:-$EXPERIMENT_DIR/pids}"
GPU_COMMIT_WIDTH="${GPU_COMMIT_WIDTH:-16}"

case "${1:-}" in
  late) early=false ;;
  early) early=true ;;
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
  *) echo "usage: $0 late|early|status|stop" >&2; exit 2 ;;
esac

[[ -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]] || {
  echo "storage build not configured: $STORAGE_BUILD_DIR" >&2
  exit 1
}

export BUILD_DIR="$STORAGE_BUILD_DIR"
export PID_DIR="$STORAGE_PID_DIR"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
export GPU_RDMA_SEARCH_PROGRESSION_MODE=manual
export GPU_EXACT_FRONTIER_EARLY_ISSUE="$early"
export GPU_GRAPH_COMMIT_WIDTH="$GPU_COMMIT_WIDTH"
export GPU_GRAPH_ISSUE_WIDTH="$GPU_COMMIT_WIDTH"
export GPU_QUERY_BEAM_MERGE_POLICY=stable-run
export ENABLE_BREAKDOWN=false
export LOG_DIR="${STORAGE_LOG_ROOT:-$PROGRAM_DIR/storage_logs}/$(date +%Y%m%d_%H%M%S)_${1}"

source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$PROFILE"
validate_vector_id_namespace_size
for ((node = 1; node <= SHARDS; ++node)); do
  validate_index_metadata storage "$node"
done

mkdir -p "$LOG_DIR" "$STORAGE_PID_DIR"
PID_DIR="$STORAGE_PID_DIR" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
cmake --build "$STORAGE_BUILD_DIR" -j "$BUILD_JOBS" --target dvstor_memory_node
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

echo "storage ready: mode=${1} exact_frontier_early_issue=$early commit/issue=$GPU_COMMIT_WIDTH"
echo "logs: $LOG_DIR"
