#!/usr/bin/env bash
set -euo pipefail

# Safe offline defaults are established before sourcing the historical SIFT
# common file. Explicit caller overrides are preserved after profile loading.
BUILD_THREAD_LIMIT=32
BUILD_JOBS="${BUILD_JOBS:-16}"
BUILD_THREADS="${BUILD_THREADS:-16}"
REQUESTED_PARTITION_STRATEGY="${PARTITION_STRATEGY:-balanced}"
REQUESTED_INDEX_PREFIX="${PQ_INDEX_PREFIX:-${INDEX_PREFIX:-}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

PARTITION_STRATEGY="$REQUESTED_PARTITION_STRATEGY"
PARTITION_MAX_DEGREE="${PARTITION_MAX_DEGREE:-32}"
PQ_SUBQUANTIZERS="${PQ_SUBQUANTIZERS:-32}"
if [[ -n "$REQUESTED_INDEX_PREFIX" ]]; then
  INDEX_PREFIX="$REQUESTED_INDEX_PREFIX"
else
  INDEX_PREFIX="$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pmd${PARTITION_MAX_DEGREE}_pq${PQ_SUBQUANTIZERS}_schema16"
fi

PQ_TRAIN_SAMPLES="${PQ_TRAIN_SAMPLES:-262144}"
PQ_OPQ_ITERATIONS="${PQ_OPQ_ITERATIONS:-20}"
PQ_ITERATIONS="${PQ_ITERATIONS:-25}"
PQ_ENCODE_CHUNK_VECTORS="${PQ_ENCODE_CHUNK_VECTORS:-32768}"
PQ_THREADS="${PQ_THREADS:-16}"
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
for name in BUILD_JOBS BUILD_THREADS PQ_THREADS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] ||
     ((value > BUILD_THREAD_LIMIT)); then
    echo "$name must be an integer in [1,$BUILD_THREAD_LIMIT]: $value" >&2
    exit 1
  fi
done
for name in PQ_TRAIN_SAMPLES PQ_OPQ_ITERATIONS PQ_ITERATIONS \
            PQ_ENCODE_CHUNK_VECTORS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer: $value" >&2
    exit 1
  fi
done
if [[ -n "${PQ_REUSE_MODEL:-}" && ! -s "$PQ_REUSE_MODEL" ]]; then
  echo "PQ_REUSE_MODEL is missing or empty: $PQ_REUSE_MODEL" >&2
  exit 1
fi

BASE_DATA_PATH="$(base_bin)"
METADATA_FILE="${INDEX_PREFIX}.meta.json"
GRAPH_METADATA_FILE="${INDEX_PREFIX}.graph.meta.json"
LOCK_FILE="${INDEX_PREFIX}.build.lock"
LOG_FILE="${BUILD_LOG:-$LOG_DIR/build_${PARTITION_STRATEGY}_$(date +%Y%m%d_%H%M%S).log}"

builder=("$BUILD_DIR/vamana_offline_builder"
  --data-path "$BASE_DATA_PATH" --output-prefix "$INDEX_PREFIX"
  --memory-nodes "$SHARDS" --partition-strategy "$PARTITION_STRATEGY"
  --R "$R" --beam-width "$BUILD_BEAM" --alpha "$ALPHA"
  --threads "$BUILD_THREADS" --max-vectors "$MAX_VECTORS"
  --vector-data-type "$VECTOR_DATA_TYPE"
  --partition-max-degree "$PARTITION_MAX_DEGREE"
  --partition-imbalance "$PARTITION_IMBALANCE" --skip-sanity-check)
pq=("$BUILD_DIR/vamana_pq_indexer"
  --index-prefix "$INDEX_PREFIX" --subquantizers "$PQ_SUBQUANTIZERS"
  --train-samples "$PQ_TRAIN_SAMPLES" --opq-iterations "$PQ_OPQ_ITERATIONS"
  --pq-iterations "$PQ_ITERATIONS" --chunk-vectors "$PQ_ENCODE_CHUNK_VECTORS"
  --threads "$PQ_THREADS" --seed "${SEED:-1234}" --overwrite)
[[ -z "${PQ_REUSE_MODEL:-}" ]] || pq+=(--reuse-model "$PQ_REUSE_MODEL")
extent=("$BUILD_DIR/vamana_graph_extent_indexer"
  --index-prefix "$INDEX_PREFIX" --overwrite)

# Embedded by design: SIFT remains deployable without sourcing another
# dataset's preflight implementation.
preflight_contract() {
  local check_resources="${1:-0}"
  python3 - "$BASE_DATA_PATH" "$(dirname "$INDEX_PREFIX")" \
    "$MAX_VECTORS" "$DIM" "$VECTOR_DATA_TYPE" "$R" "$BUILD_BEAM" \
    "$SHARDS" "$PARTITION_STRATEGY" "$PARTITION_MAX_DEGREE" \
    "$PARTITION_IMBALANCE" "$ALPHA" "$PQ_SUBQUANTIZERS" \
    "$PQ_TRAIN_SAMPLES" "$BUILD_THREADS" "$PQ_THREADS" \
    "$check_resources" <<'PY_PREFLIGHT'
import math
import os
from pathlib import Path
import struct
import sys

GIB = 1 << 30


def fail(message):
    raise ValueError(message)


def positive(name, raw):
    try:
        value = int(raw)
    except ValueError:
        fail(f'{name} must be an integer: {raw!r}')
    if value <= 0:
        fail(f'{name} must be > 0: {value}')
    return value


def finite_at_least(name, raw, minimum):
    try:
        value = float(raw)
    except ValueError:
        fail(f'{name} must be numeric: {raw!r}')
    if not math.isfinite(value) or value < minimum:
        fail(f'{name} must be finite and >= {minimum}: {raw!r}')
    return value


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def available_memory_bytes():
    try:
        fields = {}
        with open('/proc/meminfo', encoding='utf-8') as stream:
            for line in stream:
                key, raw = line.split(':', 1)
                fields[key] = int(raw.strip().split()[0]) * 1024
        return fields.get('MemAvailable', 0)
    except (OSError, ValueError):
        return 0


def cgroup_memory_available():
    candidates = (
        (Path('/sys/fs/cgroup/memory.max'),
         Path('/sys/fs/cgroup/memory.current')),
        (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
         Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')),
    )
    for limit_path, usage_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding='utf-8').strip()
            if raw_limit == 'max':
                return None
            limit = int(raw_limit)
            usage = int(usage_path.read_text(encoding='utf-8').strip())
            if limit >= 1 << 60:
                return None
            return max(0, limit - usage)
        except (OSError, ValueError):
            continue
    return None


def main():
    (raw_data, raw_output, raw_vectors, raw_dim, dtype, raw_degree,
     raw_beam, raw_shards, partition, raw_pmd, raw_imbalance, raw_alpha,
     raw_pq, raw_train, raw_build_threads, raw_pq_threads,
     raw_check_resources) = sys.argv[1:]
    data = Path(raw_data)
    output = Path(raw_output)
    max_vectors = positive('MAX_VECTORS', raw_vectors)
    dim = positive('DIM', raw_dim)
    degree = positive('R', raw_degree)
    beam = positive('BUILD_BEAM', raw_beam)
    shards = positive('SHARDS', raw_shards)
    pmd = positive('PARTITION_MAX_DEGREE', raw_pmd)
    pq = positive('PQ_SUBQUANTIZERS', raw_pq)
    train = positive('PQ_TRAIN_SAMPLES', raw_train)
    build_threads = positive('BUILD_THREADS', raw_build_threads)
    pq_threads = positive('PQ_THREADS', raw_pq_threads)
    imbalance = finite_at_least(
        'PARTITION_IMBALANCE', raw_imbalance, 1.0)
    alpha = finite_at_least('ALPHA', raw_alpha, 1.0)

    if dtype != 'uint8':
        fail(f'SIFT100M dtype must be uint8: {dtype!r}')
    if data.suffix != '.u8bin':
        fail(f'SIFT100M base must use .u8bin: {data}')
    if not data.is_file():
        fail(f'prepared base dataset is missing: {data}')
    with data.open('rb') as stream:
        header = stream.read(8)
    if len(header) != 8:
        fail(f'dataset header is truncated: {data}')
    vectors, file_dim = struct.unpack('<II', header)
    if vectors == 0 or file_dim == 0:
        fail(f'dataset header has zero vectors/dimension: '
             f'{vectors}x{file_dim}')
    if file_dim != dim:
        fail(f'dataset dim is {file_dim}, expected DIM={dim}')
    expected_bytes = 8 + vectors * file_dim
    actual_bytes = data.stat().st_size
    if actual_bytes != expected_bytes:
        fail(f'dataset size/header mismatch: bytes={actual_bytes}, '
             f'expected={expected_bytes}')
    if max_vectors > vectors:
        fail(f'MAX_VECTORS={max_vectors} exceeds dataset vectors={vectors}')
    if max_vectors > 0xFFFFFFFF:
        fail(f'vector count exceeds uint32 id capacity: {max_vectors}')
    if max_vectors <= degree:
        fail(f'dataset must contain at least R+1 vectors: '
             f'N={max_vectors}, R={degree}')
    if shards > 64:
        fail(f'SHARDS exceeds tagged-pointer capacity 64: {shards}')
    if degree > 255:
        fail(f'R exceeds one-byte graph-degree capacity 255: {degree}')
    if pmd > degree:
        fail(f'PARTITION_MAX_DEGREE must be <= R: {pmd} > {degree}')
    if partition not in {'balanced', 'bfs', 'metis'}:
        fail(f'unsupported PARTITION_STRATEGY: {partition}')
    if pq > 32:
        fail(f'PQ_SUBQUANTIZERS exceeds runtime maximum 32: {pq}')
    if dim % pq:
        fail(f'DIM={dim} is not divisible by PQ_SUBQUANTIZERS={pq}')
    if train < 256:
        fail('PQ_TRAIN_SAMPLES must be >= 256')
    if build_threads > 32 or pq_threads > 32:
        fail('offline build thread counts must not exceed 32')

    if raw_check_resources == '1':
        output.mkdir(parents=True, exist_ok=True)
        if not os.access(output, os.W_OK | os.X_OK):
            fail(f'index output directory is not writable: {output}')
        fixed_bytes = align_up(24 + align_up(dim, 8), 16)
        provisional_slots = min(15, max(2, (degree + 15) // 16))
        graph_bytes = align_up(
            16 + (degree + provisional_slots) * 8, 8)
        estimated_final = max_vectors * (
            fixed_bytes + graph_bytes + 24 + pq + 1)
        required_disk = math.ceil(estimated_final * 1.10)
        stat = os.statvfs(output)
        available_disk = stat.f_bavail * stat.f_frsize
        if available_disk < required_disk:
            fail('insufficient free disk for a fresh SIFT100M index: '
                 f'available={available_disk / GIB:.1f} GiB, '
                 f'required={required_disk / GIB:.1f} GiB')
        peak_memory = max(
            max_vectors * (dim * 4 + degree * 4 + 1),
            max_vectors * (fixed_bytes + graph_bytes + 288),
        )
        required_memory = math.ceil(peak_memory * 1.20)
        host_available = available_memory_bytes()
        if host_available and host_available < required_memory:
            fail('insufficient MemAvailable for build estimate: '
                 f'available={host_available / GIB:.1f} GiB, '
                 f'required={required_memory / GIB:.1f} GiB')
        cgroup_available = cgroup_memory_available()
        if (cgroup_available is not None and
                cgroup_available < required_memory):
            fail('cgroup memory headroom is too small: '
                 f'available={cgroup_available / GIB:.1f} GiB, '
                 f'required={required_memory / GIB:.1f} GiB')
        print('[preflight] resources: '
              f'estimated_final={estimated_final / GIB:.1f} GiB '
              f'disk_required_with_margin={required_disk / GIB:.1f} GiB '
              f'disk_available={available_disk / GIB:.1f} GiB '
              f'memory_required_with_margin={required_memory / GIB:.1f} GiB '
              f'memory_available={host_available / GIB:.1f} GiB')

    print(f'[preflight] SIFT100M: vectors={max_vectors} dim={dim} '
          f'dtype={dtype} R={degree} beam={beam} alpha={alpha} '
          f'partition={partition} pmd={pmd} imbalance={imbalance} '
          f'shards={shards} pq={pq}')


try:
    main()
except (OSError, ValueError, OverflowError) as error:
    print(f'SIFT100M index preflight failed: {error}', file=sys.stderr)
    raise SystemExit(1)
PY_PREFLIGHT
}

metadata_schema() {
  python3 - "${1:?metadata path is required}" <<'PY_SCHEMA'
import json
import sys
try:
    with open(sys.argv[1], encoding='utf-8') as stream:
        value = json.load(stream).get('schema_version')
    if not isinstance(value, int):
        raise ValueError(f'invalid schema_version: {value!r}')
    print(value)
except (OSError, ValueError, json.JSONDecodeError) as error:
    print(error, file=sys.stderr)
    raise SystemExit(1)
PY_SCHEMA
}

validate_graph_contract() {
  local metadata="${1:?metadata path is required}"
  local expected_schema="${2:?expected schema is required}"
  python3 - "$metadata" "$expected_schema" "$INDEX_PREFIX" \
    "$BASE_DATA_PATH" "$MAX_VECTORS" "$DIM" "$R" "$BUILD_BEAM" \
    "$SHARDS" "$PARTITION_STRATEGY" "$PARTITION_MAX_DEGREE" \
    "$PARTITION_IMBALANCE" "$ALPHA" <<'PY_GRAPH'
import json
import math
from pathlib import Path
import struct
import sys


def fail(message):
    raise ValueError(message)


def main():
    (raw_metadata, raw_schema, raw_prefix, raw_data, raw_vectors, raw_dim,
     raw_degree, raw_beam, raw_shards, partition, raw_pmd, raw_imbalance,
     raw_alpha) = sys.argv[1:]
    metadata_path = Path(raw_metadata)
    prefix = Path(raw_prefix)
    schema = int(raw_schema)
    vectors = int(raw_vectors)
    dim = int(raw_dim)
    degree = int(raw_degree)
    beam = int(raw_beam)
    shards = int(raw_shards)
    pmd = int(raw_pmd)
    imbalance = float(raw_imbalance)
    alpha = float(raw_alpha)
    try:
        with metadata_path.open(encoding='utf-8') as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        fail(f'cannot read metadata {metadata_path}: {error}')
    if not isinstance(metadata, dict):
        fail(f'metadata root is not an object: {metadata_path}')

    expected = {
        'schema_version': schema,
        'data_file': raw_data,
        'output_prefix': str(prefix),
        'distance': 'l2',
        'num_vectors': vectors,
        'dim': dim,
        'R': degree,
        'beam_width_construction': beam,
        'num_memory_nodes': shards,
        'node_layout': 'plain',
        'storage_format': 'vamana_tagged_v2',
        'remote_ptr_format': 'tagged_inc24_shard6_off34x16_v1',
        'vector_data_type': 'uint8',
        'partition_strategy': partition,
        'partition_max_degree': pmd,
        'idmap_format': 'owner_sharded_v2_bound',
        'centroid_state_format': 'physical_shard_centroid_v2_bound',
    }
    errors = [
        f'{key}: metadata={metadata.get(key)!r}, expected={wanted!r}'
        for key, wanted in expected.items()
        if metadata.get(key) != wanted
    ]
    for key, wanted in (('alpha', alpha),
                        ('partition_imbalance', imbalance)):
        actual = metadata.get(key)
        if (not isinstance(actual, (int, float)) or
                not math.isclose(float(actual), wanted,
                                 rel_tol=0.0, abs_tol=1e-12)):
            errors.append(
                f'{key}: metadata={actual!r}, expected={wanted!r}')

    fingerprints = metadata.get('shard_build_fingerprints')
    counts = metadata.get('hot_graph_entry_counts')
    if (not isinstance(fingerprints, list) or
            len(fingerprints) != shards or
            any(not isinstance(value, int) or value == 0
                for value in fingerprints)):
        errors.append('shard_build_fingerprints is invalid')
        fingerprints = [None] * shards
    if (not isinstance(counts, list) or len(counts) != shards or
            any(not isinstance(value, int) or value <= 0
                for value in counts) or sum(counts) != vectors):
        errors.append('hot_graph_entry_counts is invalid')

    for shard in range(1, shards + 1):
        dat = Path(f'{prefix}_node{shard}_of{shards}.dat')
        idmap = Path(f'{prefix}_node{shard}_of{shards}.idmap')
        centroid = Path(f'{prefix}_node{shard}_of{shards}.centroid')
        for sidecar in (idmap, centroid):
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                errors.append(f'missing graph-stage artifact: {sidecar}')
        try:
            actual_size = dat.stat().st_size
            with dat.open('rb') as stream:
                header = stream.read(16)
            if len(header) != 16:
                errors.append(f'shard header is truncated: {dat}')
                continue
            declared_size, fingerprint = struct.unpack('<QQ', header)
            if declared_size != actual_size:
                errors.append(
                    f'shard declared size mismatch: {dat}: '
                    f'declared={declared_size}, actual={actual_size}')
            expected_fingerprint = fingerprints[shard - 1]
            if (expected_fingerprint is not None and
                    fingerprint != expected_fingerprint):
                errors.append(f'shard fingerprint mismatch: {dat}')
        except OSError as error:
            errors.append(f'cannot inspect shard {dat}: {error}')

    if errors:
        print(f'invalid schema-{schema} graph contract: {metadata_path}',
              file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        raise SystemExit(1)
    print(f'[validate] schema-{schema} graph contract is complete: {prefix}')


try:
    main()
except (OSError, ValueError, TypeError, OverflowError) as error:
    print(f'SIFT100M graph validation failed: {error}', file=sys.stderr)
    raise SystemExit(1)
PY_GRAPH
}

validate_graph_stage() {
  validate_graph_contract "${1:?metadata path is required}" 15
}

validate_final_stage() {
  local role="${1:?validation role is required}"
  validate_graph_contract "$METADATA_FILE" 16 || return 1
  validate_index_metadata "$role"
}

ensure_offline_built() {
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "build directory is not configured: $BUILD_DIR" >&2
    return 1
  fi
  local targets=(vamana_offline_builder vamana_pq_indexer
                 vamana_graph_extent_indexer)
  if [[ "$RUN_INDEX_SELF_TESTS" == 1 ]]; then
    targets+=(offline_local_id_set_test vamana_offline_graph_test)
  fi
  echo "[build] offline-only targets dir=$BUILD_DIR jobs=$BUILD_JOBS"
  cmake --build "$BUILD_DIR" -j "$BUILD_JOBS" --target "${targets[@]}"
  local target
  for target in "${targets[@]}"; do
    if [[ ! -x "$BUILD_DIR/$target" ]]; then
      echo "required offline target was not produced: $BUILD_DIR/$target" >&2
      return 1
    fi
  done
}

require_existing_tools() {
  local tools=("${builder[0]}" "${pq[0]}" "${extent[0]}")
  if [[ "$RUN_INDEX_SELF_TESTS" == 1 ]]; then
    tools+=("$BUILD_DIR/offline_local_id_set_test"
            "$BUILD_DIR/vamana_offline_graph_test")
  fi
  local tool
  for tool in "${tools[@]}"; do
    if [[ ! -x "$tool" ]]; then
      echo "required offline executable is missing: $tool" >&2
      return 1
    fi
  done
}

atomic_copy() {
  local source="${1:?source is required}"
  local destination="${2:?destination is required}"
  local temporary="${destination}.copy.$$"
  cp -- "$source" "$temporary"
  mv -f -- "$temporary" "$destination"
}

all_outputs=("$METADATA_FILE" "$GRAPH_METADATA_FILE" "$(model_file)"
             "$(graph_extent_file)")
for ((node = 1; node <= SHARDS; ++node)); do
  all_outputs+=("$(shard_file "$node")" "$(idmap_file "$node")"
                "$(centroid_file "$node")"
                "$(navigation_code_file "$node")")
done

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
        if validate_final_stage storage >/dev/null 2>&1 &&
           [[ -s "$(model_file)" ]]; then
          if validate_final_stage compute >/dev/null 2>&1; then
            echo "[validate] state=complete"
          elif [[ ! -e "$(graph_extent_file)" &&
                  ! -L "$(graph_extent_file)" ]]; then
            echo "[validate] state=pq-complete next=extent"
          else
            echo "[validate] extent artifact exists but is invalid: $(graph_extent_file)" >&2
            validate_final_stage compute || true
            return 1
          fi
        elif validate_graph_stage "$GRAPH_METADATA_FILE" >/dev/null 2>&1; then
          echo "[validate] schema-16 state is damaged; valid graph recovery is available for PQ retry" >&2
          return 1
        else
          echo "[validate] schema-16 state is incomplete and has no valid graph recovery commit" >&2
          validate_final_stage storage || true
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
  printf '[validate] builder preflight:'
  printf ' %q' "${builder[@]}" --preflight-only
  echo
  printf '[validate] graph command:'
  printf ' %q' "${builder[@]}"
  echo
  printf '[validate] pq command:'
  printf ' %q' "${pq[@]}"
  echo
  printf '[validate] extent command:'
  printf ' %q' "${extent[@]}"
  echo
}

preflight_contract 0
if [[ "$VALIDATE_ONLY" == 1 ]]; then
  require_existing_tools
  "${builder[@]}" --preflight-only
  validate_state_only
  print_commands
  exit 0
fi

# This target list is intentionally CPU/offline-only. No compute-node or CUDA
# target is requested by this script.
ensure_offline_built
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
  preflight_contract 1
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
      elif validate_final_stage storage >/dev/null 2>&1 &&
           [[ -s "$(model_file)" ]]; then
        if [[ "$REBUILD_EXTENT" != 1 ]] &&
           validate_final_stage compute >/dev/null 2>&1; then
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
    [[ ! -e "$artifact" && ! -L "$artifact" ]] ||
      echo "  $artifact" >&2
  done
  echo "Use a new prefix or explicitly set REBUILD_GRAPH=1." >&2
  exit 1
else
  stage=graph
fi

if [[ "$stage" == graph ]]; then
  preflight_contract 1
fi

{
  echo "[state] dataset=SIFT100M stage=$stage prefix=$INDEX_PREFIX"
  echo "[state] graph recovery metadata=$GRAPH_METADATA_FILE"
  sha256sum "${builder[0]}" "${pq[0]}" "${extent[0]}"

  if [[ "$stage" == graph ]]; then
    printf '[graph] command:'
    printf ' %q' "${builder[@]}"
    echo
    OMP_NUM_THREADS="$BUILD_THREADS" OMP_THREAD_LIMIT="$BUILD_THREAD_LIMIT" \
      "${builder[@]}"
    validate_graph_stage "$METADATA_FILE"
    atomic_copy "$METADATA_FILE" "$GRAPH_METADATA_FILE"
    echo "[graph] committed recovery metadata: $GRAPH_METADATA_FILE"
    stage=pq
  fi

  if [[ "$stage" == pq ]]; then
    validate_graph_stage "$METADATA_FILE"
    printf '[pq] command:'
    printf ' %q' "${pq[@]}"
    echo
    OMP_NUM_THREADS="$PQ_THREADS" OMP_THREAD_LIMIT="$BUILD_THREAD_LIMIT" \
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
      "${pq[@]}"
    validate_final_stage storage
    [[ -s "$(model_file)" ]] || {
      echo "PQ model was not published: $(model_file)" >&2
      exit 1
    }
    stage=extent
  fi

  if [[ "$stage" == extent ]]; then
    printf '[extent] command:'
    printf ' %q' "${extent[@]}"
    echo
    "${extent[@]}"
  fi

  validate_final_stage storage
  validate_final_stage compute
  echo "[build] complete: $INDEX_PREFIX"
  echo "[build] resumable graph commit: $GRAPH_METADATA_FILE"
  echo "[build] log: $LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
