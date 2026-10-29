#!/usr/bin/env bash
# Generate one compute-node service config. Usage:
#   env overrides... gen_cn_config.sh <output.ini>
set -euo pipefail
cd "$HOME/experiment/dvstor"
source experiment/sift1b/common.sh
load_experiment_profile "${PROFILE:-04_gpu_persistent_gpunetio}"
export ENABLE_UPDATES="${ENABLE_UPDATES:-true}"
export NUM_CLIENTS=2
export MN_DYNAMIC_SLOTS_PER_SHARD=200000
write_service_config "$1"
echo "written: $1"
