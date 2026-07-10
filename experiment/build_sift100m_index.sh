#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

BUILD_PROFILE="${1:-${BUILD_PROFILE:-}}"
if [[ -n "$BUILD_PROFILE" ]]; then
  PROFILE_FILE="$SCRIPT_DIR/profiles/${BUILD_PROFILE}.env"
  if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "unknown build profile: $BUILD_PROFILE" >&2
    exit 1
  fi
  source "$PROFILE_FILE"
fi

ensure_built vamana_offline_builder
"$SCRIPT_DIR/prepare_sift100m_data.sh"

mkdir -p "$(dirname "$INDEX_PREFIX")"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_${PARTITION_STRATEGY}_$(date +%Y%m%d_%H%M%S).log}"

cmd=("$BUILD_DIR/vamana_offline_builder"
  --data-path "$(base_bin)"
  --output-prefix "$INDEX_PREFIX"
  --memory-nodes "$SHARDS"
  --partition-strategy "$PARTITION_STRATEGY"
  --R "$R"
  --beam-width "$BUILD_BEAM"
  --alpha "$ALPHA"
  --threads "$BUILD_THREADS"
  --max-vectors "$MAX_VECTORS"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --storage-format "$STORAGE_FORMAT"
  --partition-max-degree "${PARTITION_MAX_DEGREE:-16}"
  --partition-imbalance "${PARTITION_IMBALANCE:-1.03}"
  --skip-sanity-check
  --use-rabitq
  --rabitq-cache-format "${RABITQ_CACHE_FORMAT:-budget}")

if [[ "${GPU_TIERED_INDEX:-0}" == "1" || "${GPU_TIERED_INDEX:-0}" == "true" ]]; then
  cmd+=(--gpu-tiered-index
        --gpu-entry-points "${GPU_ENTRY_POINTS:-256}")
fi

echo "[build] index prefix: $INDEX_PREFIX"
echo "[build] partition: $PARTITION_STRATEGY shards=$SHARDS R=$R build_beam=$BUILD_BEAM dtype=$VECTOR_DATA_TYPE"
echo "[build] distance execution: cpu-avx2"
printf '[build] command:'; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
validate_index_metadata
echo "[build] log: $LOG_FILE"
