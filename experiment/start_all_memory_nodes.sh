#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"
for node in $(seq 1 "$SHARDS"); do
  "$EXPERIMENT_DIR/start_memory_node.sh" "$node" "$PROFILE"
done
