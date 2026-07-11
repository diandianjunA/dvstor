#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

SOURCE_PREFIX="${SOURCE_PREFIX:-${LEGACY_INDEX_PREFIX:?LEGACY_INDEX_PREFIX is required}}"
if [[ "$SOURCE_PREFIX" == "$INDEX_PREFIX" ]]; then
  echo "source and output index prefixes must differ" >&2
  exit 1
fi

ensure_built vamana_legacy_index_converter vamana_pq_indexer
mkdir -p "$(dirname "$INDEX_PREFIX")"

migration_cmd=("$BUILD_DIR/vamana_legacy_index_converter"
  --source-prefix "$SOURCE_PREFIX"
  --output-prefix "$INDEX_PREFIX"
  --io-threads "${MIGRATION_IO_THREADS:-$BUILD_THREADS}"
  --chunk-nodes "${MIGRATION_CHUNK_NODES:-65536}")
if [[ "${OVERWRITE_INDEX:-0}" == "1" ]]; then migration_cmd+=(--overwrite); fi

pq_threads="${PQ_THREADS:-32}"
pq_cmd=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX"
  --subquantizers "$PQ_SUBQUANTIZERS"
  --train-samples "${PQ_TRAIN_SAMPLES:-262144}"
  --opq-iterations "${PQ_OPQ_ITERATIONS:-20}"
  --pq-iterations "${PQ_ITERATIONS:-25}"
  --chunk-vectors "${PQ_ENCODE_CHUNK_VECTORS:-32768}"
  --entry-points "${GPU_ENTRY_POINTS:-256}"
  --threads "$pq_threads"
  --seed "${SEED:-1234}")
if [[ -n "${PQ_REUSE_MODEL:-}" ]]; then pq_cmd+=(--reuse-model "$PQ_REUSE_MODEL"); fi
if [[ "${OVERWRITE_INDEX:-0}" == "1" || "${OVERWRITE_PQ:-0}" == "1" ]]; then
  pq_cmd+=(--overwrite)
fi

metadata_path="${INDEX_PREFIX}.meta.json"
if [[ "${OVERWRITE_INDEX:-0}" != "1" && -f "$metadata_path" ]]; then
  if ! grep -Eq '"schema_version"[[:space:]]*:[[:space:]]*14' "$metadata_path"; then
    echo "existing output metadata is not a schema-14 migration checkpoint: $metadata_path" >&2
    exit 1
  fi
  echo "[migrate] schema-14 checkpoint exists; skipping completed migration"
else
  printf '[migrate] command:'; printf ' %q' "${migration_cmd[@]}"; echo
  "${migration_cmd[@]}"
fi

printf '[pq] command:'; printf ' %q' "${pq_cmd[@]}"; echo
OMP_NUM_THREADS="$pq_threads" \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OMP_DYNAMIC=FALSE \
"${pq_cmd[@]}"
validate_index_metadata storage
echo "[migrate] complete: $INDEX_PREFIX"
