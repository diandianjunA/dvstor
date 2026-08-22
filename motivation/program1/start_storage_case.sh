#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build-storage}"
STORAGE_PID_DIR="${STORAGE_PID_DIR:-$EXPERIMENT_DIR/pids}"
STORAGE_LOG_ROOT="${STORAGE_LOG_ROOT:-$PROGRAM_DIR/storage_logs}"

usage() {
  cat <<'EOF'
Usage:
  ./motivation/program1/start_storage_case.sh coupled_one_stage
  ./motivation/program1/start_storage_case.sh two_stage
  ./motivation/program1/start_storage_case.sh status
  ./motivation/program1/start_storage_case.sh stop

The two case commands validate the storage build/index, stop the previous
DVStor memory nodes managed by STORAGE_PID_DIR, then start every shard with
the exact M1.1 feature-mode contract.
EOF
}

action="${1:-}"
if [[ -z "$action" || "$action" == -h || "$action" == --help ]]; then
  usage
  [[ -n "$action" ]] && exit 0 || exit 2
fi
if (( $# != 1 )); then
  usage >&2
  exit 2
fi

stop_nodes() {
  local pid_file pid
  local -a stopped_pids=()
  for pid_file in "$STORAGE_PID_DIR"/memory_node_*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && stopped_pids+=("$pid")
  done
  PID_DIR="$STORAGE_PID_DIR" "$EXPERIMENT_DIR/stop_memory_nodes.sh"
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

show_status() {
  local found=0 pid_file pid state
  if [[ -s "$STORAGE_PID_DIR/program1_current_case.env" ]]; then
    echo "current case:"
    sed 's/^/  /' "$STORAGE_PID_DIR/program1_current_case.env"
  fi
  for pid_file in "$STORAGE_PID_DIR"/memory_node_*.pid; do
    [[ -e "$pid_file" ]] || continue
    found=1
    pid="$(<"$pid_file")"
    state=stopped
    kill -0 "$pid" 2>/dev/null && state=running
    printf '%s pid=%s state=%s\n' "$(basename "$pid_file")" "$pid" "$state"
  done
  ((found != 0)) || echo "no managed memory-node PID files in $STORAGE_PID_DIR"
}

case "$action" in
  status)
    show_status
    exit 0
    ;;
  stop)
    stop_nodes
    rm -f "$STORAGE_PID_DIR/program1_current_case.env"
    echo "storage memory nodes stopped"
    exit 0
    ;;
  coupled_one_stage|coupled)
    case_name=coupled_one_stage
    update_mode=coupled
    expected_feature_modes=20
    ;;
  two_stage|decoupled)
    case_name=two_stage
    update_mode=decoupled
    expected_feature_modes=21
    ;;
  *)
    echo "unsupported storage case: $action" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]]; then
  echo "storage build directory is not configured: $STORAGE_BUILD_DIR" >&2
  exit 1
fi

export BUILD_DIR="$STORAGE_BUILD_DIR"
export PID_DIR="$STORAGE_PID_DIR"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode"
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
export ENABLE_BREAKDOWN=true

# Validate every shard before stopping a healthy previous case. This avoids a
# partial five-process deployment when a .dat/idmap/centroid/PQ artifact is
# missing or belongs to another build fingerprint.
source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$PROFILE"
validate_vector_id_namespace_size
for ((node = 1; node <= SHARDS; ++node)); do
  validate_index_metadata storage "$node"
done

run_stamp="$(date +%Y%m%d_%H%M%S)"
case_log_dir="$STORAGE_LOG_ROOT/${run_stamp}_${case_name}"
mkdir -p "$case_log_dir" "$STORAGE_PID_DIR"
export LOG_DIR="$case_log_dir"

stop_nodes
"$EXPERIMENT_DIR/start_all_memory_nodes.sh" "$PROFILE"

# start_all_memory_nodes.sh launches in the background. Catch immediate
# allocation/configuration failures while allowing the processes to continue
# loading their large shard files and accepting the compute-node connections.
sleep "${STORAGE_STARTUP_PROBE_SECONDS:-2}"
for ((node = 1; node <= SHARDS; ++node)); do
  pid_file="$STORAGE_PID_DIR/memory_node_${node}.pid"
  if [[ ! -s "$pid_file" ]]; then
    echo "memory node $node did not publish a PID file" >&2
    exit 1
  fi
  pid="$(<"$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "memory node $node exited during startup; inspect $case_log_dir" >&2
    exit 1
  fi
done

{
  echo "case=$case_name"
  echo "profile=$PROFILE"
  echo "storage_owner_update_completion_mode=$update_mode"
  echo "gpu_dynamic_graph_access_mode=adaptive"
  echo "gpu_rdma_search_progression_mode=decoupled"
  echo "feature_modes=$expected_feature_modes"
  echo "build_dir=$STORAGE_BUILD_DIR"
  echo "log_dir=$case_log_dir"
  echo "started_at=$(date --iso-8601=seconds)"
} > "$STORAGE_PID_DIR/program1_current_case.env"

echo "storage case started: $case_name"
echo "expected startup feature_modes=$expected_feature_modes"
echo "logs: $case_log_dir"
echo "The processes are loading/listening in the background; now start or confirm the compute case."

