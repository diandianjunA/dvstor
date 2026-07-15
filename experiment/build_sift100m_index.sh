#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

# Offline OPQ/PQ construction settings. They are deliberately kept here rather
# than in the online GPUNetIO profile: changing any of them requires rebuilding
# the schema-15 index and has no effect on an already-built index at runtime.
PQ_TRAIN_SAMPLES="${PQ_TRAIN_SAMPLES:-262144}"
PQ_OPQ_ITERATIONS="${PQ_OPQ_ITERATIONS:-20}"
PQ_ITERATIONS="${PQ_ITERATIONS:-25}"
PQ_ENCODE_CHUNK_VECTORS="${PQ_ENCODE_CHUNK_VECTORS:-32768}"
PQ_THREADS="${PQ_THREADS:-32}"
GPU_ENTRY_POINTS="${GPU_ENTRY_POINTS:-256}"

ensure_built vamana_offline_builder vamana_pq_indexer
PREPARE_BENCHMARK_DATA=0 "$EXPERIMENT_DIR/prepare_sift100m_data.sh"

mkdir -p "$(dirname "$INDEX_PREFIX")"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_${PARTITION_STRATEGY}_$(date +%Y%m%d_%H%M%S).log}"

artifacts=(
  "${INDEX_PREFIX}.meta.json"
  "${INDEX_PREFIX}.anchors"
  "${INDEX_PREFIX}.pq${PQ_SUBQUANTIZERS}"
)
for ((node = 1; node <= SHARDS; ++node)); do
  artifacts+=(
    "${INDEX_PREFIX}_node${node}_of${SHARDS}.dat"
    "${INDEX_PREFIX}_node${node}_of${SHARDS}.idmap"
    "${INDEX_PREFIX}_node${node}_of${SHARDS}.pq${PQ_SUBQUANTIZERS}.codes"
  )
done
existing=()
for artifact in "${artifacts[@]}"; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then existing+=("$artifact"); fi
done
if ((${#existing[@]} != 0)); then
  if [[ "${OVERWRITE_INDEX:-0}" != "1" ]]; then
    echo "index output already exists; choose a new PQ_INDEX_PREFIX or set OVERWRITE_INDEX=1:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    exit 1
  fi
  echo "[build] removing ${#existing[@]} old target artifacts before rebuild"
  rm -f -- "${existing[@]}"
fi

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
  --partition-max-degree "${PARTITION_MAX_DEGREE:-32}"
  --partition-imbalance "${PARTITION_IMBALANCE:-1.03}"
  --anchor-count-per-shard "${ANCHORS_PER_SHARD:-4096}"
  --skip-sanity-check)

pq=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX"
  --subquantizers "${PQ_SUBQUANTIZERS:-32}"
  --train-samples "$PQ_TRAIN_SAMPLES"
  --opq-iterations "$PQ_OPQ_ITERATIONS"
  --pq-iterations "$PQ_ITERATIONS"
  --chunk-vectors "$PQ_ENCODE_CHUNK_VECTORS"
  --entry-points "$GPU_ENTRY_POINTS"
  --threads "$PQ_THREADS"
  --seed "${SEED:-1234}")
if [[ -n "${PQ_REUSE_MODEL:-}" ]]; then pq+=(--reuse-model "$PQ_REUSE_MODEL"); fi
if [[ "${OVERWRITE_INDEX:-0}" == "1" ]]; then pq+=(--overwrite); fi

{
  echo "[build] schema-14 compact graph intermediate: $INDEX_PREFIX"
  printf '[build] command:'; printf ' %q' "${builder[@]}"; echo
  "${builder[@]}"
  printf '[pq] command:'; printf ' %q' "${pq[@]}"; echo
  OMP_NUM_THREADS="$PQ_THREADS" \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE \
  "${pq[@]}"
} 2>&1 | tee "$LOG_FILE"

validate_index_metadata storage
echo "[build] complete: schema-15 persistent OPQ/PQ index $INDEX_PREFIX"
echo "[build] log: $LOG_FILE"
