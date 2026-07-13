#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

ROLE="${INDEX_ROLE:-compute}"
LOCAL_SHARD="${LOCAL_SHARD:-0}"
if [[ "$ROLE" != "compute" && "$ROLE" != "storage" ]]; then
  echo "INDEX_ROLE must be compute or storage" >&2
  exit 1
fi
if [[ ! "$LOCAL_SHARD" =~ ^[0-9]+$ ]] || ((LOCAL_SHARD > SHARDS)); then
  echo "LOCAL_SHARD must be in [0,$SHARDS]" >&2
  exit 1
fi
if [[ "$ROLE" == "storage" && "$LOCAL_SHARD" == "0" ]]; then
  echo "storage upgrade requires one-based LOCAL_SHARD" >&2
  exit 1
fi

ensure_built vamana_pq_indexer
cmd=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX"
  --upgrade-layout-only)
if ((LOCAL_SHARD > 0)); then cmd+=(--local-shard "$LOCAL_SHARD"); fi

printf '[schema15-upgrade] command:'; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
validate_index_metadata "$ROLE" "$LOCAL_SHARD"
echo "[schema15-upgrade] complete: role=$ROLE shard=$LOCAL_SHARD prefix=$INDEX_PREFIX"
