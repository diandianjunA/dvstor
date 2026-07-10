#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

CONVERSION_PROFILE="${1:-${CONVERSION_PROFILE:-04_gpu_persistent_gpunetio}}"
PROFILE_FILE="$SCRIPT_DIR/profiles/${CONVERSION_PROFILE}.env"
if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "unknown conversion profile: $CONVERSION_PROFILE" >&2
  exit 1
fi
source "$PROFILE_FILE"

missing=0
for ((node_id = 1; node_id <= SHARDS; ++node_id)); do
  shard="$(shard_file "$node_id")"
  if [[ ! -s "$shard" ]]; then
    echo "missing old shard: $shard" >&2
    missing=1
  fi
done
if (( missing != 0 )); then
  echo "The converter needs the original .dat shards because they contain the Vamana graph." >&2
  exit 1
fi

if [[ ! -x "$BUILD_DIR/vamana_gpu_sidecar_converter" ]]; then
  cmake -S "$PROJECT_DIR" -B "$BUILD_DIR"
fi
ensure_built vamana_gpu_sidecar_converter

mkdir -p "$(dirname "$INDEX_PREFIX")" "$LOG_DIR"
LOG_FILE="${CONVERSION_LOG:-$LOG_DIR/convert_gpu_sidecars_$(date +%Y%m%d_%H%M%S).log}"
cmd=("$BUILD_DIR/vamana_gpu_sidecar_converter"
  --index-prefix "$INDEX_PREFIX"
  --gpu-hot-degree "${GPU_HOT_DEGREE:-32}"
  --gpu-entry-points "${GPU_ENTRY_POINTS:-256}"
  --gpu-graph-page-bytes "${GPU_GRAPH_PAGE_BYTES:-4096}"
  --threads "${GPU_SIDECAR_THREADS:-$SHARDS}"
  --rabitq-source "${GPU_SIDECAR_RABITQ_SOURCE:-auto}")

if [[ "${GPU_SIDECAR_OVERWRITE:-0}" == "1" ||
      "${GPU_SIDECAR_OVERWRITE:-0}" == "true" ]]; then
  cmd+=(--overwrite)
fi

echo "[convert] old index prefix: $INDEX_PREFIX"
printf '[convert] command:'; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
validate_index_metadata
echo "[convert] log: $LOG_FILE"
