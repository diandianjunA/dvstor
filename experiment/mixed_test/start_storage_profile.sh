#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET="${DATASET:-sift100m}"
PROFILE="${PROFILE:-}"
RUN_TOKEN="${RUN_TOKEN:-manual_$(date +%Y%m%d_%H%M%S)}"
STORAGE_BUILD_DIR="${STORAGE_BUILD_DIR:-$PROJECT_DIR/build}"

usage() {
  echo "usage: $0 --dataset NAME --profile NAME --run-token TOKEN" >&2
}
while (( $# > 0 )); do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --run-token) RUN_TOKEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "$DATASET" in sift100m|deep100m|spacev100m) ;; *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;; esac
case "$PROFILE" in
  baseline) PROFILE=04_gpu_persistent_gpunetio_baseline ;;
  full) PROFILE=04_gpu_persistent_gpunetio ;;
esac
EXPERIMENT_DIR="$PROJECT_DIR/experiment/$DATASET"
[[ -f "$EXPERIMENT_DIR/profiles/$PROFILE.env" ]] || {
  echo "unknown profile for $DATASET: $PROFILE" >&2; exit 2;
}
[[ -f "$STORAGE_BUILD_DIR/CMakeCache.txt" ]] || {
  echo "storage build is not configured: $STORAGE_BUILD_DIR" >&2
  echo "set STORAGE_BUILD_DIR to the storage-node build directory" >&2
  exit 1
}

# Prevent mode variables left by another experiment from silently changing the
# two publication profiles. Unknown/custom profiles retain their normal env
# override behavior.
case "$PROFILE" in
  04_gpu_persistent_gpunetio_baseline)
    export STORAGE_OWNER_UPDATE_COMPLETION_MODE=coupled
    export GPU_DYNAMIC_GRAPH_ACCESS_MODE=fixed
    export GPU_RDMA_SEARCH_PROGRESSION_MODE=coupled
    ;;
  04_gpu_persistent_gpunetio)
    export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
    export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
    export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
    ;;
esac

export BUILD_DIR="$STORAGE_BUILD_DIR"
export LOG_DIR="${STORAGE_LOG_DIR:-$SCRIPT_DIR/storage_logs/$(date +%Y%m%d_%H%M%S)_${DATASET}_${PROFILE}}"
source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$PROFILE"
validate_vector_id_namespace_size
for ((node = 1; node <= SHARDS; ++node)); do
  validate_index_metadata storage "$node"
done

mkdir -p "$LOG_DIR" "$PID_DIR"
"$EXPERIMENT_DIR/stop_memory_nodes.sh"
cmake --build "$STORAGE_BUILD_DIR" -j --target dvstor_memory_node
"$EXPERIMENT_DIR/start_all_memory_nodes.sh" "$PROFILE"

sleep "${STORAGE_STARTUP_PROBE_SECONDS:-2}"
for ((node = 1; node <= SHARDS; ++node)); do
  pid_file="$PID_DIR/memory_node_${node}.pid"
  [[ -s "$pid_file" ]] || { echo "node $node has no PID file" >&2; exit 1; }
  kill -0 "$(<"$pid_file")" 2>/dev/null || {
    echo "node $node exited; inspect $LOG_DIR" >&2; exit 1;
  }
done

printf '%s\t%s\t%s\t%s\t%s\n' "$RUN_TOKEN" "$DATASET" "$PROFILE" \
  "$LOG_DIR" "$(date --iso-8601=seconds)" > "$SCRIPT_DIR/.storage_ready.tsv"
echo "storage ready: dataset=$DATASET profile=$PROFILE token=$RUN_TOKEN"
echo "logs: $LOG_DIR"
