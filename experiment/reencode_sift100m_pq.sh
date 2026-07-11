#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"
use_storage_build

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

SOURCE_PREFIX="${SOURCE_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pmd${PARTITION_MAX_DEGREE}_pq16}"
TARGET_PREFIX="$INDEX_PREFIX"
SOURCE_METADATA="${SOURCE_PREFIX}.meta.json"
TARGET_METADATA="${TARGET_PREFIX}.meta.json"

if [[ ! -s "$SOURCE_METADATA" ]]; then
  echo "missing schema-14 source metadata: $SOURCE_METADATA" >&2
  exit 1
fi
python3 - "$SOURCE_METADATA" <<'PY_VALIDATE_SOURCE'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)
if metadata.get('schema_version') != 14 or metadata.get('storage_format') != 'vamana_compact_v1':
    raise SystemExit('source index must already use schema-14 vamana_compact_v1')
PY_VALIDATE_SOURCE

ensure_built vamana_pq_indexer
mkdir -p "$(dirname "$TARGET_PREFIX")"

if [[ "$SOURCE_PREFIX" != "$TARGET_PREFIX" ]]; then
  link_artifact() {
    local source="$1" target="$2"
    if [[ -e "$target" || -L "$target" ]]; then
      if [[ "$(realpath "$target")" == "$(realpath "$source")" ]]; then
        return
      fi
      echo "target artifact already exists and is not the expected link: $target" >&2
      exit 1
    fi
    ln -s "$(realpath "$source")" "$target"
  }

  link_artifact "${SOURCE_PREFIX}.anchors" "${TARGET_PREFIX}.anchors"
  for ((node = 1; node <= SHARDS; ++node)); do
    link_artifact "${SOURCE_PREFIX}_node${node}_of${SHARDS}.dat" \
                  "${TARGET_PREFIX}_node${node}_of${SHARDS}.dat"
    link_artifact "${SOURCE_PREFIX}_node${node}_of${SHARDS}.idmap" \
                  "${TARGET_PREFIX}_node${node}_of${SHARDS}.idmap"
  done

  if [[ ! -e "$TARGET_METADATA" ]]; then
    python3 - "$SOURCE_METADATA" "$TARGET_METADATA" "$TARGET_PREFIX" <<'PY_COPY_METADATA'
import json
import sys

source, target, prefix = sys.argv[1:]
with open(source, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)
metadata['output_prefix'] = prefix
with open(target, 'w', encoding='utf-8') as stream:
    json.dump(metadata, stream, indent=2)
    stream.write('\n')
PY_COPY_METADATA
  fi
fi

pq_threads="${PQ_THREADS:-32}"
cmd=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$TARGET_PREFIX"
  --subquantizers "$PQ_SUBQUANTIZERS"
  --train-samples "${PQ_TRAIN_SAMPLES:-262144}"
  --opq-iterations "${PQ_OPQ_ITERATIONS:-20}"
  --pq-iterations "${PQ_ITERATIONS:-25}"
  --chunk-vectors "${PQ_ENCODE_CHUNK_VECTORS:-32768}"
  --entry-points "${GPU_ENTRY_POINTS:-256}"
  --threads "$pq_threads"
  --seed "${SEED:-1234}")
if [[ -n "${PQ_REUSE_MODEL:-}" ]]; then cmd+=(--reuse-model "$PQ_REUSE_MODEL"); fi
if [[ "${OVERWRITE_PQ:-0}" == "1" ]]; then cmd+=(--overwrite); fi

printf '[pq-reencode] command:'; printf ' %q' "${cmd[@]}"; echo
OMP_NUM_THREADS="$pq_threads" \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OMP_DYNAMIC=FALSE \
"${cmd[@]}"

validate_index_metadata storage
echo "[pq-reencode] complete: source=$SOURCE_PREFIX target=$TARGET_PREFIX PQ${PQ_SUBQUANTIZERS}"
