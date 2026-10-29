#!/usr/bin/env bash
# GIST1M offline index build: graph -> OPQ/PQ -> Live-Extent sidecar.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

PARTITION_MAX_DEGREE="${PARTITION_MAX_DEGREE:-32}"
PQ_SUBQUANTIZERS="${PQ_SUBQUANTIZERS:-32}"
PQ_TRAIN_SAMPLES="${PQ_TRAIN_SAMPLES:-262144}"
PQ_THREADS="${PQ_THREADS:-64}"
BUILD_THREADS="${BUILD_THREADS:-64}"

BASE_DATA_PATH="$(base_bin)"
METADATA_FILE="${INDEX_PREFIX}.meta.json"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_gist1m_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$INDEX_DIR" "$LOG_DIR"

[[ -s "$BASE_DATA_PATH" ]] || { echo "missing base data: $BASE_DATA_PATH" >&2; exit 1; }

builder=("$BUILD_DIR/vamana_offline_builder"
  --data-path "$BASE_DATA_PATH" --output-prefix "$INDEX_PREFIX"
  --memory-nodes "$SHARDS" --partition-strategy "$PARTITION_STRATEGY"
  --R "$R" --beam-width "$BUILD_BEAM" --alpha "$ALPHA"
  --threads "$BUILD_THREADS" --max-vectors "$MAX_VECTORS"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --partition-max-degree "$PARTITION_MAX_DEGREE"
  --partition-imbalance "$PARTITION_IMBALANCE"
  --query-path "$(query_bin)" --groundtruth-path "$(groundtruth_bin)")
pq=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX" --subquantizers "$PQ_SUBQUANTIZERS"
  --train-samples "$PQ_TRAIN_SAMPLES" --threads "$PQ_THREADS"
  --seed "${SEED:-1234}" --overwrite)
[[ "${PQ_NO_OPQ:-0}" != "1" ]] || pq+=(--no-opq)
extent=("$BUILD_DIR/vamana_graph_extent_indexer"
  --index-prefix "$INDEX_PREFIX" --overwrite)

{
  echo "[build] profile=$PROFILE prefix=$INDEX_PREFIX shards=$SHARDS R=$R beam=$BUILD_BEAM pq=$PQ_SUBQUANTIZERS strategy=$PARTITION_STRATEGY"
  sha256sum "${builder[0]}" "${pq[0]}" "${extent[0]}"
  printf '[graph] command:'; printf ' %q' "${builder[@]}"; echo
  OMP_NUM_THREADS="$BUILD_THREADS" "${builder[@]}"
  printf '[pq] command:'; printf ' %q' "${pq[@]}"; echo
  OMP_NUM_THREADS="$PQ_THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "${pq[@]}"
  printf '[extent] command:'; printf ' %q' "${extent[@]}"; echo
  "${extent[@]}"
  echo "[build] complete: $INDEX_PREFIX"
} 2>&1 | tee "$LOG_FILE"
