#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"
PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

PQ_TRAIN_SAMPLES="${PQ_TRAIN_SAMPLES:-262144}"
PQ_OPQ_ITERATIONS="${PQ_OPQ_ITERATIONS:-20}"
PQ_ITERATIONS="${PQ_ITERATIONS:-25}"
PQ_ENCODE_CHUNK_VECTORS="${PQ_ENCODE_CHUNK_VECTORS:-32768}"
PQ_THREADS="${PQ_THREADS:-16}"
SEED="${SEED:-1234}"
RUN_INDEX_SELF_TESTS="${RUN_INDEX_SELF_TESTS:-1}"
REBUILD_GRAPH="${REBUILD_GRAPH:-0}"
REBUILD_PQ="${REBUILD_PQ:-0}"
REBUILD_EXTENT="${REBUILD_EXTENT:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
OVERWRITE_INDEX="${OVERWRITE_INDEX:-0}"

for flag in RUN_INDEX_SELF_TESTS REBUILD_GRAPH REBUILD_PQ REBUILD_EXTENT \
            VALIDATE_ONLY OVERWRITE_INDEX; do
  if [[ "${!flag}" != 0 && "${!flag}" != 1 ]]; then
    echo "$flag must be 0 or 1: ${!flag}" >&2
    exit 1
  fi
done
if [[ ! "$PQ_THREADS" =~ ^[1-9][0-9]*$ ]] ||
   ((PQ_THREADS > BUILD_THREAD_LIMIT)); then
  echo "PQ_THREADS must be an integer in [1,$BUILD_THREAD_LIMIT]: $PQ_THREADS" >&2
  exit 1
fi

BASE_DATA_PATH="$(base_bin)"
METADATA_FILE="${INDEX_PREFIX}.meta.json"
GRAPH_METADATA_FILE="${INDEX_PREFIX}.graph.meta.json"
LOCK_FILE="${INDEX_PREFIX}.build.lock"
PREFLIGHT="$SCRIPT_DIR/index_preflight.py"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_${PARTITION_STRATEGY}_$(date +%Y%m%d_%H%M%S).log}"

builder=("$BUILD_DIR/vamana_offline_builder"
  --data-path "$BASE_DATA_PATH" --output-prefix "$INDEX_PREFIX"
  --memory-nodes "$SHARDS" --partition-strategy "$PARTITION_STRATEGY"
  --R "$R" --beam-width "$BUILD_BEAM" --alpha "$ALPHA"
  --threads "$BUILD_THREADS" --max-vectors "$MAX_VECTORS"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --partition-max-degree "${PARTITION_MAX_DEGREE:-32}"
  --partition-imbalance "$PARTITION_IMBALANCE" --skip-sanity-check)
pq=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX" --subquantizers "$PQ_SUBQUANTIZERS"
  --train-samples "$PQ_TRAIN_SAMPLES" --opq-iterations "$PQ_OPQ_ITERATIONS"
  --pq-iterations "$PQ_ITERATIONS" --chunk-vectors "$PQ_ENCODE_CHUNK_VECTORS"
  --threads "$PQ_THREADS" --seed "$SEED" --overwrite)
[[ -z "${PQ_REUSE_MODEL:-}" ]] || pq+=(--reuse-model "$PQ_REUSE_MODEL")
extent=("$BUILD_DIR/vamana_graph_extent_indexer"
  --index-prefix "$INDEX_PREFIX" --overwrite)
preflight=(python3 "$PREFLIGHT" preflight
  --data "$BASE_DATA_PATH" --output-dir "$(dirname "$INDEX_PREFIX")"
  --max-vectors "$MAX_VECTORS" --dim "$DIM" --dtype "$VECTOR_DATA_TYPE"
  --degree "$R" --beam "$BUILD_BEAM" --shards "$SHARDS"
  --partition "$PARTITION_STRATEGY"
  --partition-max-degree "${PARTITION_MAX_DEGREE:-32}"
  --imbalance "$PARTITION_IMBALANCE" --alpha "$ALPHA"
  --pq-subquantizers "$PQ_SUBQUANTIZERS"
  --pq-train-samples "$PQ_TRAIN_SAMPLES"
  --pq-opq-iterations "$PQ_OPQ_ITERATIONS"
  --pq-iterations "$PQ_ITERATIONS"
  --pq-chunk-vectors "$PQ_ENCODE_CHUNK_VECTORS"
  --pq-seed "$SEED"
  --build-threads "$BUILD_THREADS" --pq-threads "$PQ_THREADS")
[[ -z "${PQ_REUSE_MODEL:-}" ]] ||
  preflight+=(--pq-reuse-model "$PQ_REUSE_MODEL")

all_outputs=("$METADATA_FILE" "$GRAPH_METADATA_FILE" "$(model_file)"
             "$(graph_extent_file)")
for ((node = 1; node <= SHARDS; ++node)); do
  all_outputs+=(
    "$(shard_file "$node")"
    "$(idmap_file "$node")"
    "$(centroid_file "$node")"
    "$(navigation_code_file "$node")")
done

validate_graph_stage() {
  local metadata="${1:?metadata path is required}"
  python3 "$PREFLIGHT" graph --metadata "$metadata" \
    --data "$BASE_DATA_PATH" --alpha "$ALPHA" \
    --imbalance "$PARTITION_IMBALANCE" \
    --prefix "$INDEX_PREFIX" --max-vectors "$MAX_VECTORS" --dim "$DIM" \
    --degree "$R" --beam "$BUILD_BEAM" --shards "$SHARDS" \
    --partition "$PARTITION_STRATEGY" \
    --partition-max-degree "${PARTITION_MAX_DEGREE:-32}"
}

metadata_schema() {
  python3 "$PREFLIGHT" schema --metadata "$1"
}

atomic_copy() {
  local source="${1:?source is required}"
  local destination="${2:?destination is required}"
  local temporary="${destination}.copy.$$"
  cp -- "$source" "$temporary"
  mv -f -- "$temporary" "$destination"
}

has_any_output() {
  local artifact
  for artifact in "${all_outputs[@]}"; do
    if [[ -e "$artifact" || -L "$artifact" ]]; then return 0; fi
  done
  return 1
}

validate_state_only() {
  local schema
  if [[ -s "$METADATA_FILE" ]]; then
    if ! schema="$(metadata_schema "$METADATA_FILE")"; then
      echo "[validate] corrupt metadata: $METADATA_FILE" >&2
      return 1
    fi
    case "$schema" in
      15)
        validate_graph_stage "$METADATA_FILE"
        echo "[validate] state=graph-complete next=pq"
        ;;
      16)
        if validate_index_metadata storage >/dev/null 2>&1 &&
           [[ -s "$(model_file)" ]]; then
          if validate_index_metadata compute >/dev/null 2>&1; then
            echo "[validate] state=complete"
          elif [[ ! -e "$(graph_extent_file)" &&
                  ! -L "$(graph_extent_file)" ]]; then
            echo "[validate] state=pq-complete next=extent"
          else
            echo "[validate] extent artifact exists but is invalid: $(graph_extent_file)" >&2
            validate_index_metadata compute || true
            return 1
          fi
        elif validate_graph_stage "$GRAPH_METADATA_FILE" >/dev/null 2>&1; then
          echo "[validate] schema-16 state is damaged; valid graph recovery is available for PQ retry" >&2
          return 1
        else
          echo "[validate] schema-16 state is incomplete and has no valid graph recovery commit" >&2
          validate_index_metadata storage || true
          validate_graph_stage "$GRAPH_METADATA_FILE" || true
          return 1
        fi
        ;;
      *)
        echo "[validate] unsupported metadata schema: $schema" >&2
        return 1
        ;;
    esac
  elif [[ -e "$METADATA_FILE" || -L "$METADATA_FILE" ]]; then
    echo "[validate] metadata exists but is empty: $METADATA_FILE" >&2
    return 1
  elif [[ -s "$GRAPH_METADATA_FILE" ]]; then
    validate_graph_stage "$GRAPH_METADATA_FILE"
    echo "[validate] state=graph-recoverable next=pq"
  elif has_any_output; then
    echo "[validate] partial artifacts exist without a valid commit" >&2
    return 1
  else
    echo "[validate] state=empty next=graph"
  fi
}

print_commands() {
  printf '[validate] builder preflight:'; printf ' %q' "${builder[@]}" --preflight-only; echo
  printf '[validate] graph command:'; printf ' %q' "${builder[@]}"; echo
  printf '[validate] pq command:'; printf ' %q' "${pq[@]}"; echo
  printf '[validate] extent command:'; printf ' %q' "${extent[@]}"; echo
}

"${preflight[@]}"
if [[ "$VALIDATE_ONLY" == 1 ]]; then
  for required_target in "${pq[0]}" "${extent[0]}"; do
    if [[ ! -x "$required_target" ]]; then
      echo "[validate] required executable is missing: $required_target" >&2
      exit 1
    fi
  done
  if [[ -x "${builder[0]}" ]]; then
    "${builder[@]}" --preflight-only
  else
    echo "[validate] builder does not exist yet: ${builder[0]}" >&2
    exit 1
  fi
  validate_state_only
  print_commands
  exit 0
fi

targets=(vamana_offline_builder vamana_pq_indexer vamana_graph_extent_indexer)
if [[ "$RUN_INDEX_SELF_TESTS" == 1 ]]; then
  targets+=(offline_local_id_set_test vamana_offline_graph_test)
fi
ensure_built "${targets[@]}"
"$SCRIPT_DIR/prepare_deep100m_data.sh"
"${builder[@]}" --preflight-only

mkdir -p "$(dirname "$INDEX_PREFIX")" "$LOG_DIR"
command -v flock >/dev/null || {
  echo "flock is required to serialize builds of one index prefix" >&2
  exit 1
}
exec {INDEX_LOCK_FD}>"$LOCK_FILE"
if ! flock -n "$INDEX_LOCK_FD"; then
  echo "another index build holds the prefix lock: $LOCK_FILE" >&2
  exit 1
fi

if [[ "$OVERWRITE_INDEX" == 1 && "$REBUILD_GRAPH" != 1 ]]; then
  echo "[safety] OVERWRITE_INDEX=1 no longer deletes a valid graph."
  echo "[safety] Use REBUILD_GRAPH=1 only for an intentional full rebuild."
fi

if [[ "$RUN_INDEX_SELF_TESTS" == 1 ]]; then
  "$BUILD_DIR/offline_local_id_set_test"
  "$BUILD_DIR/vamana_offline_graph_test"
fi

if [[ "$REBUILD_GRAPH" == 1 ]]; then
  "${preflight[@]}" --check-resources
  echo "[rebuild] explicitly removing only artifacts for prefix: $INDEX_PREFIX"
  for artifact in "${all_outputs[@]}"; do
    [[ ! -e "$artifact" && ! -L "$artifact" ]] || rm -f -- "$artifact"
  done
fi

stage=""
if [[ -s "$METADATA_FILE" ]]; then
  schema="$(metadata_schema "$METADATA_FILE" 2>/dev/null || echo invalid)"
  case "$schema" in
    15)
      validate_graph_stage "$METADATA_FILE"
      atomic_copy "$METADATA_FILE" "$GRAPH_METADATA_FILE"
      stage=pq
      ;;
    16)
      if [[ "$REBUILD_PQ" == 1 ]]; then
        validate_graph_stage "$GRAPH_METADATA_FILE" || {
          echo "REBUILD_PQ=1 requires valid graph recovery metadata." >&2
          exit 1
        }
        atomic_copy "$GRAPH_METADATA_FILE" "$METADATA_FILE"
        stage=pq
      elif validate_index_metadata storage >/dev/null 2>&1 &&
           [[ -s "$(model_file)" ]]; then
        if [[ "$REBUILD_EXTENT" != 1 ]] &&
           validate_index_metadata compute >/dev/null 2>&1; then
          stage=complete
        else
          stage=extent
        fi
      elif validate_graph_stage "$GRAPH_METADATA_FILE"; then
        echo "[resume] incomplete PQ stage; restoring schema-15 graph commit"
        atomic_copy "$GRAPH_METADATA_FILE" "$METADATA_FILE"
        stage=pq
      else
        echo "Incomplete schema-16 outputs and no valid graph recovery commit." >&2
        echo "Graph shards were preserved; use a new prefix or REBUILD_GRAPH=1." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported or corrupt metadata: $METADATA_FILE (schema=$schema)" >&2
      echo "Nothing was deleted; use a new prefix or REBUILD_GRAPH=1." >&2
      exit 1
      ;;
  esac
elif [[ -s "$GRAPH_METADATA_FILE" ]] &&
     validate_graph_stage "$GRAPH_METADATA_FILE"; then
  echo "[resume] restoring committed schema-15 graph metadata"
  atomic_copy "$GRAPH_METADATA_FILE" "$METADATA_FILE"
  stage=pq
elif has_any_output; then
  echo "Partial artifacts exist without a valid commit; nothing was deleted:" >&2
  for artifact in "${all_outputs[@]}"; do
    [[ ! -e "$artifact" && ! -L "$artifact" ]] || echo "  $artifact" >&2
  done
  echo "Use a new prefix or explicitly set REBUILD_GRAPH=1." >&2
  exit 1
else
  stage=graph
fi

if [[ "$stage" == graph ]]; then
  "${preflight[@]}" --check-resources
fi

{
  echo "[state] dataset=DEEP100M stage=$stage prefix=$INDEX_PREFIX"
  echo "[state] graph recovery metadata=$GRAPH_METADATA_FILE"
  sha256sum "${builder[0]}" "${pq[0]}" "${extent[0]}"

  if [[ "$stage" == graph ]]; then
    printf '[graph] command:'; printf ' %q' "${builder[@]}"; echo
    OMP_NUM_THREADS="$BUILD_THREADS" OMP_THREAD_LIMIT="$BUILD_THREAD_LIMIT" \
      "${builder[@]}"
    validate_graph_stage "$METADATA_FILE"
    atomic_copy "$METADATA_FILE" "$GRAPH_METADATA_FILE"
    echo "[graph] committed recovery metadata: $GRAPH_METADATA_FILE"
    stage=pq
  fi

  if [[ "$stage" == pq ]]; then
    validate_graph_stage "$METADATA_FILE"
    printf '[pq] command:'; printf ' %q' "${pq[@]}"; echo
    OMP_NUM_THREADS="$PQ_THREADS" OMP_THREAD_LIMIT="$BUILD_THREAD_LIMIT" \
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
      "${pq[@]}"
    validate_index_metadata storage
    [[ -s "$(model_file)" ]] || {
      echo "PQ model was not published: $(model_file)" >&2
      exit 1
    }
    stage=extent
  fi

  if [[ "$stage" == extent ]]; then
    printf '[extent] command:'; printf ' %q' "${extent[@]}"; echo
    "${extent[@]}"
  fi

  validate_index_metadata storage
  validate_index_metadata compute
  echo "[build] complete: $INDEX_PREFIX"
  echo "[build] resumable graph commit: $GRAPH_METADATA_FILE"
  echo "[build] log: $LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
