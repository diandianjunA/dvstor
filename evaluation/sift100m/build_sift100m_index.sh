#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

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
  --partition-max-degree "${PARTITION_MAX_DEGREE:-48}"
  --partition-imbalance "${PARTITION_IMBALANCE:-1.03}"
  --skip-sanity-check
  --use-rabitq)

echo "[build] index prefix: $INDEX_PREFIX"
echo "[build] partition: $PARTITION_STRATEGY shards=$SHARDS R=$R build_beam=$BUILD_BEAM dtype=$VECTOR_DATA_TYPE"
echo "[build] distance execution: cpu-avx2"
printf '[build] command:'; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
validate_index_metadata
echo "[build] log: $LOG_FILE"
