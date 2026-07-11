#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"
use_storage_build

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

ensure_built vamana_offline_builder vamana_pq_indexer
"$EXPERIMENT_DIR/prepare_sift100m_data.sh"

mkdir -p "$(dirname "$INDEX_PREFIX")"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_${PARTITION_STRATEGY}_$(date +%Y%m%d_%H%M%S).log}"

builder=("$BUILD_DIR/vamana_offline_builder"
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
  --partition-max-degree "${PARTITION_MAX_DEGREE:-16}"
  --partition-imbalance "${PARTITION_IMBALANCE:-1.03}"
  --anchor-count-per-shard "${ANCHORS_PER_SHARD:-4096}"
  --skip-sanity-check)

pq=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX"
  --subquantizers "${PQ_SUBQUANTIZERS:-16}"
  --train-samples "${PQ_TRAIN_SAMPLES:-262144}"
  --opq-iterations "${PQ_OPQ_ITERATIONS:-20}"
  --pq-iterations "${PQ_ITERATIONS:-25}"
  --chunk-vectors "${PQ_ENCODE_CHUNK_VECTORS:-32768}"
  --entry-points "${GPU_ENTRY_POINTS:-256}"
  --threads "${PQ_THREADS:-32}"
  --seed "${SEED:-1234}")
if [[ -n "${PQ_REUSE_MODEL:-}" ]]; then pq+=(--reuse-model "$PQ_REUSE_MODEL"); fi
if [[ "${OVERWRITE_INDEX:-0}" == "1" ]]; then pq+=(--overwrite); fi

{
  echo "[build] schema-14 compact graph: $INDEX_PREFIX"
  printf '[build] command:'; printf ' %q' "${builder[@]}"; echo
  "${builder[@]}"
  printf '[pq] command:'; printf ' %q' "${pq[@]}"; echo
  OMP_NUM_THREADS="${PQ_THREADS:-32}" \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE \
  "${pq[@]}"
} 2>&1 | tee "$LOG_FILE"

validate_index_metadata storage
echo "[build] complete: $INDEX_PREFIX"
echo "[build] log: $LOG_FILE"
