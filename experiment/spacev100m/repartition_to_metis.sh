#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/repartition_to_metis.env"
source "$SCRIPT_DIR/spacev100m_common.sh"

for flag in RUN_REPARTITIONER_TESTS GRAPH_ONLY VALIDATE_ONLY \
            DELETE_BALANCED_AFTER_SUCCESS; do
  if [[ "${!flag}" != 0 && "${!flag}" != 1 ]]; then
    echo "$flag must be 0 or 1: ${!flag}" >&2
    exit 2
  fi
done
for value_name in BUILD_JOBS REPARTITION_THREADS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || ((value > 32)); then
    echo "$value_name must be an integer in [1,32]: $value" >&2
    exit 2
  fi
done
if [[ ! "$PARTITION_MAX_DEGREE" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$PQ_CHUNK_VECTORS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARTITION_MAX_DEGREE and PQ_CHUNK_VECTORS must be positive integers" >&2
  exit 2
fi
if [[ "$GRAPH_ONLY" == 1 && "$DELETE_BALANCED_AFTER_SUCCESS" == 1 ]]; then
  echo "refusing to delete balanced source after a graph-only conversion" >&2
  exit 2
fi
if [[ "$VALIDATE_ONLY" == 1 && "$DELETE_BALANCED_AFTER_SUCCESS" == 1 ]]; then
  echo "refusing to delete balanced source in validate-only mode" >&2
  exit 2
fi

SOURCE_METADATA="${SOURCE_INDEX_PREFIX}.meta.json"
if [[ ! -s "$SOURCE_METADATA" ]]; then
  echo "missing balanced schema-16 source metadata: $SOURCE_METADATA" >&2
  exit 1
fi
mkdir -p "$(dirname "$METIS_INDEX_PREFIX")" "$SCRIPT_DIR/logs"

targets=(vamana_metis_repartitioner)
if [[ "$RUN_REPARTITIONER_TESTS" == 1 ]]; then
  targets+=(metis_repartitioner_test)
fi
ensure_built "${targets[@]}"
if [[ "$RUN_REPARTITIONER_TESTS" == 1 ]]; then
  "$BUILD_DIR/metis_repartitioner_test"
fi

command=("$BUILD_DIR/vamana_metis_repartitioner"
  --input-prefix "$SOURCE_INDEX_PREFIX"
  --output-prefix "$METIS_INDEX_PREFIX"
  --data-path "$REPARTITION_DATA_PATH"
  --partition-max-degree "$PARTITION_MAX_DEGREE"
  --partition-imbalance "$PARTITION_IMBALANCE"
  --threads "$REPARTITION_THREADS"
  --pq-chunk-vectors "$PQ_CHUNK_VECTORS")
[[ "$GRAPH_ONLY" == 0 ]] || command+=(--graph-only)
[[ "$VALIDATE_ONLY" == 0 ]] || command+=(--validate-only)

LOG_FILE="${REPARTITION_LOG:-$SCRIPT_DIR/logs/repartition_metis_$(date +%Y%m%d_%H%M%S).log}"
printf '[repartition] command:'
printf ' %q' "${command[@]}"
echo
echo "[repartition] log=$LOG_FILE"
OMP_NUM_THREADS="$REPARTITION_THREADS" \
OMP_THREAD_LIMIT=32 \
  "${command[@]}" 2>&1 | tee "$LOG_FILE"

delete_balanced_source() {
  local shards pq_subquantizers
  read -r shards pq_subquantizers < <(
    python3 - "$SOURCE_METADATA" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    metadata = json.load(stream)
if metadata.get("schema_version") != 16 or \
        metadata.get("partition_strategy") != "balanced":
    raise SystemExit("source metadata is no longer schema-16 balanced")
print(metadata["num_memory_nodes"], metadata["pq_subquantizers"])
PY
  )
  local artifacts=(
    "${SOURCE_INDEX_PREFIX}.meta.json"
    "${SOURCE_INDEX_PREFIX}.graph.meta.json"
    "${SOURCE_INDEX_PREFIX}.pq${pq_subquantizers}"
    "${SOURCE_INDEX_PREFIX}.gextent8"
    "${SOURCE_INDEX_PREFIX}.build.lock"
    "${SOURCE_INDEX_PREFIX}.graph-build.lock")
  local node
  for ((node = 1; node <= shards; ++node)); do
    artifacts+=(
      "${SOURCE_INDEX_PREFIX}_node${node}_of${shards}.dat"
      "${SOURCE_INDEX_PREFIX}_node${node}_of${shards}.idmap"
      "${SOURCE_INDEX_PREFIX}_node${node}_of${shards}.centroid"
      "${SOURCE_INDEX_PREFIX}_node${node}_of${shards}.pq${pq_subquantizers}.codes")
  done
  echo "[cleanup] conversion passed; deleting exact balanced-index artifacts"
  local artifact
  for artifact in "${artifacts[@]}"; do
    if [[ -e "$artifact" || -L "$artifact" ]]; then
      rm -f -- "$artifact"
      echo "[cleanup] removed $artifact"
    fi
  done
}

if [[ "$DELETE_BALANCED_AFTER_SUCCESS" == 1 ]]; then
  delete_balanced_source
else
  echo "[cleanup] balanced source retained; set DELETE_BALANCED_AFTER_SUCCESS=1 to remove it after a successful rerun"
fi

