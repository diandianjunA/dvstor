#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${BUILD_DIR+x}" ]]; then
  DVSTOR_BUILD_DIR_EXPLICIT=1
else
  DVSTOR_BUILD_DIR_EXPLICIT=0
fi
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"

DATASET_DIR="${DATASET_DIR:-/data/xjs/datasets/sift1b}"
WORK_DIR="${WORK_DIR:-/data/xjs/index/dvstor_sift100m}"
CONVERTED_DIR="${CONVERTED_DIR:-$WORK_DIR/converted}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/index}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"

SHARDS="${SHARDS:-5}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-balanced}"
PARTITION_IMBALANCE="${PARTITION_IMBALANCE:-1.03}"
R="${R:-96}"
BUILD_BEAM="${BUILD_BEAM:-128}"
ALPHA="${ALPHA:-1.2}"
K="${K:-10}"
DIM="${DIM:-128}"
VECTOR_DATA_TYPE="${VECTOR_DATA_TYPE:-uint8}"
BUILD_THREAD_LIMIT=32
BUILD_JOBS="${BUILD_JOBS:-16}"
BUILD_THREADS="${BUILD_THREADS:-16}"

validate_build_thread_limits() {
  local name value
  for name in BUILD_JOBS BUILD_THREADS; do
    value="${!name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] ||
        ((value > BUILD_THREAD_LIMIT)); then
      echo "$name must be an integer in [1,$BUILD_THREAD_LIMIT]: $value" >&2
      return 1
    fi
  done
}
validate_build_thread_limits
SERVICE_THREADS="${SERVICE_THREADS:-64}"
GPU_DEVICE="${GPU_DEVICE:-1}"
PQ_SUBQUANTIZERS="${PQ_SUBQUANTIZERS:-32}"
MAX_VECTORS="${MAX_VECTORS:-100000000}"
# Logical IDs are sparse and authority state is allocated only for IDs that
# exist, so exposing the complete non-wrapping uint32 range does not reserve a
# dense 4B-entry table. UINT32_MAX itself remains excluded to prevent a
# benchmark's atomic ID generator from wrapping back to base ID 0.
VECTOR_ID_NAMESPACE_SIZE="${VECTOR_ID_NAMESPACE_SIZE:-4294967295}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
GROUNDTRUTH_LABEL="${GROUNDTRUTH_LABEL:-100M}"
GROUNDTRUTH_TOPK="${GROUNDTRUTH_TOPK:-10}"

# Benchmark input files. These are the only settings normally changed when
# moving the benchmark to another machine. Source row ranges describe how each
# pre-generated u8bin was extracted.
BENCHMARK_VECTOR_SOURCE="${BENCHMARK_VECTOR_SOURCE:-$DATASET_DIR/bigann_base.bvecs}"
PERFORMANCE_QUERY_FILE="${PERFORMANCE_QUERY_FILE:-$DATASET_DIR/sift100m_to_110m_query.u8bin}"
PERFORMANCE_QUERY_START="${PERFORMANCE_QUERY_START:-100000000}"
PERFORMANCE_QUERY_END="${PERFORMANCE_QUERY_END:-110000000}"
INSERT_FILE="${INSERT_FILE:-$DATASET_DIR/sift110m_to_120m_insert.u8bin}"
INSERT_VECTOR_START="${INSERT_VECTOR_START:-110000000}"
INSERT_VECTOR_END="${INSERT_VECTOR_END:-120000000}"

# Query and insert defaults are adjacent, non-overlapping held-out ranges.

BASE_PORT="${BASE_PORT:-1234}"
HOSTS="${HOSTS:-192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202}"
IB_DEVICE="${IB_DEVICE:-}"
IB_PORT="${IB_PORT:-1}"
MAX_SEND_WRS="${MAX_SEND_WRS:-4096}"
MAX_RECEIVE_WRS="${MAX_RECEIVE_WRS:-4096}"
MAX_POLL_CQES="${MAX_POLL_CQES:-64}"

PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
INDEX_PREFIX="${INDEX_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pq${PQ_SUBQUANTIZERS}}"

# MN_MEMORY_GB is the per-shard RDMA-registered storage region, not total
# process RSS. Resolve it only after the selected profile has established the
# final INDEX_PREFIX. An explicit environment/profile value always wins.
MN_DYNAMIC_HEADROOM_PERCENT="${MN_DYNAMIC_HEADROOM_PERCENT:-20}"
MN_DYNAMIC_SLOTS_PER_SHARD="${MN_DYNAMIC_SLOTS_PER_SHARD:-}"
MN_MEMORY_MIN_GB="${MN_MEMORY_MIN_GB:-8}"

validate_vector_id_namespace_size() {
  if [[ ! "$VECTOR_ID_NAMESPACE_SIZE" =~ ^[1-9][0-9]*$ ]] ||
      ((VECTOR_ID_NAMESPACE_SIZE < MAX_VECTORS ||
        VECTOR_ID_NAMESPACE_SIZE > 4294967295)); then
    echo "VECTOR_ID_NAMESPACE_SIZE must be an integer in [$MAX_VECTORS,4294967295]: $VECTOR_ID_NAMESPACE_SIZE" >&2
    return 1
  fi
}

estimate_mn_memory_gb() {
  local metadata="${INDEX_PREFIX}.meta.json"
  python3 - "$metadata" "$SHARDS" "$MAX_VECTORS" "$DIM" "$R" \
    "$VECTOR_DATA_TYPE" "$PQ_SUBQUANTIZERS" "$PARTITION_IMBALANCE" \
    "$MN_DYNAMIC_HEADROOM_PERCENT" "$MN_DYNAMIC_SLOTS_PER_SHARD" \
    "$MN_MEMORY_MIN_GB" <<'PY_MN_MEMORY'
import json
import os
import sys
from decimal import Decimal, InvalidOperation, ROUND_CEILING

GIB = 1 << 30
REMOTE_PTR_CAPACITY = 256 * GIB
CENTROID_HEADER_BYTES = 128
CENTROID_ENTRY_BYTES = 16
CENTROID_ENTRY_CAPACITY = 4
STORAGE_CONTROL_BYTES = 4096


def fail(message):
    raise ValueError(message)


def positive_int(name, raw, *, allow_zero=False):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        fail(f'{name} must be an integer: {raw!r}')
    if value < 0 or (value == 0 and not allow_zero):
        relation = 'non-negative' if allow_zero else 'positive'
        fail(f'{name} must be {relation}: {value}')
    return value


def decimal_value(name, raw, *, minimum):
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        fail(f'{name} must be numeric: {raw!r}')
    if not value.is_finite() or value < minimum:
        fail(f'{name} must be finite and >= {minimum}: {raw!r}')
    return value


def align_up(value, alignment):
    if value < 0 or alignment <= 0:
        fail('invalid alignment input')
    return ((value + alignment - 1) // alignment) * alignment


def centroid_publication_bytes(dim):
    centroid_end = CENTROID_HEADER_BYTES + dim * 4  # canonical FP32 route
    entries_offset = align_up(centroid_end, 8)
    return align_up(
        entries_offset + CENTROID_ENTRY_CAPACITY * CENTROID_ENTRY_BYTES, 64)


def metadata_layout(path, expected_shards):
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        fail(f'cannot read index metadata {path}: {error}')

    if metadata.get('schema_version') != 16 or \
            metadata.get('storage_format') != 'vamana_tagged_v2':
        fail(f'index metadata is not a schema-16 tagged index: {path}')
    shards = positive_int('metadata.num_memory_nodes',
                          metadata.get('num_memory_nodes'))
    if shards != expected_shards:
        fail(f'metadata shard count {shards} does not match SHARDS={expected_shards}')

    counts = metadata.get('hot_graph_entry_counts')
    bases = metadata.get('dynamic_node_base_offsets')
    if not isinstance(counts, list) or len(counts) != shards:
        fail('hot_graph_entry_counts must contain one value per shard')
    if not isinstance(bases, list) or len(bases) != shards:
        fail('dynamic_node_base_offsets must contain one value per shard')
    counts = [positive_int(f'hot_graph_entry_counts[{index}]', value)
              for index, value in enumerate(counts)]
    bases = [positive_int(f'dynamic_node_base_offsets[{index}]', value)
             for index, value in enumerate(bases)]
    num_vectors = positive_int('metadata.num_vectors', metadata.get('num_vectors'))
    if sum(counts) != num_vectors:
        fail('hot_graph_entry_counts do not sum to metadata.num_vectors')

    record_bytes = positive_int(
        'hot_graph_dynamic_record_bytes',
        metadata.get('hot_graph_dynamic_record_bytes'))
    allocation_size = positive_int(
        'allocation_size', metadata.get('allocation_size'))
    if allocation_size != record_bytes:
        fail('allocation_size does not match hot_graph_dynamic_record_bytes')
    dim = positive_int('metadata.dim', metadata.get('dim'))
    return counts, bases, record_bytes, dim, 'metadata'


def fallback_layout(shards, max_vectors, dim, degree, dtype, code_bytes,
                    partition_imbalance):
    if max_vectors == 0:
        fail('MAX_VECTORS=0 requires an existing schema-16 metadata file')
    component_sizes = {'uint8': 1, 'int8': 1, 'float32': 4, 'auto': 4}
    if dtype not in component_sizes:
        fail(f'unsupported VECTOR_DATA_TYPE={dtype!r}')
    if degree > 128:
        fail(f'R exceeds the GPU/runtime limit of 128: {degree}')
    if shards > 64:
        fail(f'SHARDS exceeds tagged RemotePtr capacity: {shards}')
    if code_bytes > dim:
        fail(f'PQ_SUBQUANTIZERS exceeds DIM: {code_bytes} > {dim}')

    vector_bytes = dim * component_sizes[dtype]
    fixed_bytes = align_up(24 + align_up(vector_bytes, 8), 16)
    provisional_slots = min(15, max(2, (degree + 15) // 16))
    graph_bytes = align_up(16 + (degree + provisional_slots) * 8, 8)
    dynamic_code_offset = fixed_bytes + graph_bytes
    # Four-byte incarnation/extent prefix plus the incarnation-bound snapshot
    # checksum used by one-READ dynamic PQ validation.
    record_bytes = align_up(dynamic_code_offset + 4 + code_bytes + 4, 16)

    projected = (Decimal(max_vectors) * partition_imbalance /
                 Decimal(shards)).to_integral_value(rounding=ROUND_CEILING)
    count = int(projected)
    fixed_end = 16 + count * fixed_bytes
    graph_header = align_up(fixed_end, 64)
    graph_offset = align_up(graph_header + 64, 64)
    dat_end = align_up(graph_offset + count * graph_bytes, 64)
    persistent_bytes = STORAGE_CONTROL_BYTES + count * code_bytes
    dynamic_base = dat_end + align_up(persistent_bytes, record_bytes)
    return ([count] * shards, [dynamic_base] * shards,
            record_bytes, dim, 'schema16-fallback')


def main():
    (metadata_path, raw_shards, raw_max_vectors, raw_dim, raw_degree,
     dtype, raw_code_bytes, raw_partition_imbalance,
     raw_headroom_percent, raw_absolute_slots, raw_min_gib) = sys.argv[1:]

    shards = positive_int('SHARDS', raw_shards)
    max_vectors = positive_int('MAX_VECTORS', raw_max_vectors, allow_zero=True)
    dim = positive_int('DIM', raw_dim)
    degree = positive_int('R', raw_degree)
    code_bytes = positive_int('PQ_SUBQUANTIZERS', raw_code_bytes)
    partition_imbalance = decimal_value(
        'PARTITION_IMBALANCE', raw_partition_imbalance, minimum=Decimal(1))
    headroom_percent = decimal_value(
        'MN_DYNAMIC_HEADROOM_PERCENT', raw_headroom_percent,
        minimum=Decimal(0))
    min_gib = positive_int('MN_MEMORY_MIN_GB', raw_min_gib)

    if os.path.exists(metadata_path):
        counts, bases, record_bytes, layout_dim, source = metadata_layout(
            metadata_path, shards)
    else:
        counts, bases, record_bytes, layout_dim, source = fallback_layout(
            shards, max_vectors, dim, degree, dtype, code_bytes,
            partition_imbalance)

    if raw_absolute_slots:
        absolute_slots = positive_int(
            'MN_DYNAMIC_SLOTS_PER_SHARD', raw_absolute_slots)
        headroom_slots = [absolute_slots] * shards
        policy = f'{absolute_slots} slots/shard'
    else:
        headroom_slots = [max(1, int(
            (Decimal(count) * headroom_percent / Decimal(100))
            .to_integral_value(rounding=ROUND_CEILING))) for count in counts]
        policy = f'{headroom_percent}% of base nodes'

    tail_bytes = centroid_publication_bytes(layout_dim)
    required_by_shard = [
        base + slots * record_bytes + tail_bytes
        for base, slots in zip(bases, headroom_slots)
    ]
    required_bytes = max(required_by_shard)
    estimated_gib = max(min_gib, (required_bytes + GIB - 1) // GIB)
    if estimated_gib * GIB > REMOTE_PTR_CAPACITY:
        fail(f'estimated storage region exceeds 256 GiB: {estimated_gib} GiB')

    limiting_shard = max(range(shards), key=required_by_shard.__getitem__)
    print(
        f'[mn-memory] required={estimated_gib} GiB source={source} '
        f'policy={policy} limiting_shard={limiting_shard + 1} '
        f'dynamic_slots={headroom_slots[limiting_shard]} '
        f'record_bytes={record_bytes}',
        file=sys.stderr)
    print(estimated_gib)


try:
    main()
except (ValueError, KeyError, TypeError) as error:
    print(f'mn-memory estimation failed: {error}', file=sys.stderr)
    raise SystemExit(1)
PY_MN_MEMORY
}

resolve_mn_memory_gb() {
  local required_mn_memory_gb
  required_mn_memory_gb="$(estimate_mn_memory_gb)" || return 1
  if [[ -z "${MN_MEMORY_GB:-}" ]]; then
    MN_MEMORY_GB="$required_mn_memory_gb"
  fi
  if [[ ! "$MN_MEMORY_GB" =~ ^[1-9][0-9]*$ ]] ||
      ((MN_MEMORY_GB > 256)); then
    echo "MN_MEMORY_GB must be an integer in [1,256]: $MN_MEMORY_GB" >&2
    return 1
  fi
  if ((MN_MEMORY_GB < required_mn_memory_gb)); then
    echo "MN_MEMORY_GB=$MN_MEMORY_GB is smaller than the index-required minimum $required_mn_memory_gb GiB" >&2
    return 1
  fi
}

mkdir -p "$CONVERTED_DIR" "$INDEX_DIR" "$REPORT_DIR" "$LOG_DIR" "$PID_DIR"

base_suffix() {
  if [[ "$MAX_VECTORS" == "0" || "$MAX_VECTORS" == "1000000000" ]]; then
    echo ""
  else
    echo "_${MAX_VECTORS}"
  fi
}

query_suffix() {
  if [[ "$MAX_QUERIES" == "0" || "$MAX_QUERIES" == "10000" ]]; then
    echo ""
  else
    echo "_${MAX_QUERIES}"
  fi
}

base_bin() { echo "$CONVERTED_DIR/base$(base_suffix).u8bin"; }
query_bin() { echo "$CONVERTED_DIR/query$(query_suffix).u8bin"; }
groundtruth_bin() { echo "$CONVERTED_DIR/groundtruth_${GROUNDTRUTH_LABEL}.bin"; }
insert_bin() { echo "$INSERT_FILE"; }
performance_query_bin() { echo "$PERFORMANCE_QUERY_FILE"; }
metadata_file() { echo "${INDEX_PREFIX}.meta.json"; }
model_file() { echo "${INDEX_PREFIX}.pq${PQ_SUBQUANTIZERS}"; }
graph_extent_file() { echo "${INDEX_PREFIX}.gextent8"; }

shard_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.dat"
}

idmap_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.idmap"
}

centroid_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.centroid"
}

navigation_code_file() {
  local node_id="${1:?node id is required}"
  echo "${INDEX_PREFIX}_node${node_id}_of${SHARDS}.pq${PQ_SUBQUANTIZERS}.codes"
}

validate_index_metadata() {
  local role="${1:-compute}"
  local node_id="${2:-0}"
  local metadata
  metadata="$(metadata_file)"
  if [[ ! -s "$metadata" ]]; then
    echo "missing index metadata: $metadata" >&2
    return 1
  fi

  python3 - "$metadata" "$INDEX_PREFIX" "$(base_bin)" "$R" "$BUILD_BEAM" "$DIM" \
    "$MAX_VECTORS" "$SHARDS" "$VECTOR_DATA_TYPE" "$PQ_SUBQUANTIZERS" \
    "$PARTITION_STRATEGY" "${PARTITION_MAX_DEGREE:-32}" "$ALPHA" \
    "$PARTITION_IMBALANCE" <<'PY_VALIDATE' || return 1
import json
import math
import sys

path, prefix, data_file, degree, build_beam, dim, vectors, shards, dtype, \
    subquantizers, partition_strategy, partition_max_degree, alpha, \
    partition_imbalance = sys.argv[1:]
with open(path, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)

expected = {
    'output_prefix': prefix,
    'data_file': data_file,
    'schema_version': 16,
    'distance': 'l2',
    'node_layout': 'plain',
    'storage_format': 'vamana_tagged_v2',
    'remote_ptr_format': 'tagged_inc24_shard6_off34x16_v1',
    'navigation_execution': 'gpu_beam_v1',
    'R': int(degree),
    'beam_width_construction': int(build_beam),
    'dim': int(dim),
    'num_vectors': int(vectors),
    'num_memory_nodes': int(shards),
    'vector_data_type': dtype,
    'navigation_code_bytes': int(subquantizers),
    'pq_subquantizers': int(subquantizers),
    'pq_bits': 8,
    'partition_strategy': partition_strategy,
    'partition_max_degree': int(partition_max_degree),
    'idmap_format': 'owner_sharded_v2_bound',
    'centroid_state_format': 'physical_shard_centroid_v2_bound',
    'hot_graph_pointer_bytes': 8,
}
errors = [
    f'{key}: metadata={metadata.get(key)!r}, expected={value!r}'
    for key, value in expected.items() if metadata.get(key) != value
]
for key, expected_value in (
        ('alpha', float(alpha)),
        ('partition_imbalance', float(partition_imbalance))):
    actual = metadata.get(key)
    if (not isinstance(actual, (int, float)) or isinstance(actual, bool) or
            not math.isfinite(actual) or
            not math.isclose(float(actual), expected_value,
                             rel_tol=1e-12, abs_tol=1e-12)):
        errors.append(
            f'{key}: metadata={actual!r}, expected={expected_value!r}')
if metadata.get('navigation_quantizer') != 'opq_pq':
    errors.append('navigation_quantizer must be opq_pq')
if metadata.get('navigation_format') != 'opq_pq_graph_v1':
    errors.append('navigation_format must be opq_pq_graph_v1')
if not metadata.get('navigation_model_checksum'):
    errors.append('navigation_model_checksum is missing')
if not metadata.get('index_build_fingerprint'):
    errors.append('index_build_fingerprint is missing')
shard_fingerprints = metadata.get('shard_build_fingerprints')
if (not isinstance(shard_fingerprints, list) or
        len(shard_fingerprints) != int(shards) or
        any(not isinstance(value, int) or value == 0
            for value in shard_fingerprints)):
    errors.append('shard_build_fingerprints must bind every storage shard')
if 'medoid' in metadata or 'navigation_entry_points' in metadata:
    errors.append('runtime metadata must not contain static query entry state')
for key in (
    'hot_graph_offsets',
    'hot_graph_entry_counts',
    'hot_graph_dynamic_base_offsets',
    'navigation_code_remote_offsets',
    'navigation_code_region_bytes',
    'storage_control_remote_offsets',
    'dynamic_node_base_offsets',
):
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != int(shards):
        errors.append(f'{key} must contain one value per storage shard')
dynamic_hot = metadata.get('hot_graph_dynamic_hot_offset', 0)
graph_entry = metadata.get('hot_graph_entry_size', 0)
dynamic_code = metadata.get('dynamic_navigation_code_offset', 0)
dynamic_record = metadata.get('hot_graph_dynamic_record_bytes', 0)
dynamic_validation = metadata.get('dynamic_navigation_code_validation_bytes', 0)
dynamic_checksum = metadata.get('dynamic_navigation_code_checksum_bytes', 4)
if dynamic_code != dynamic_hot + graph_entry:
    errors.append('dynamic PQ tag must immediately follow the compact graph record')
if dynamic_validation != 4 or dynamic_checksum != 4:
    errors.append('dynamic PQ snapshot validation must use 4B tag + 4B checksum')
if dynamic_record < (dynamic_code + dynamic_validation +
                     int(subquantizers) + dynamic_checksum):
    errors.append('persistent dynamic record is too small for validated PQ codes')
if errors:
    print(f'incompatible GPU index metadata: {path}', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    raise SystemExit(1)
PY_VALIDATE

  if [[ "$role" == "compute" ]]; then
    if [[ ! -s "$(model_file)" ]]; then
      echo "missing OPQ/PQ${PQ_SUBQUANTIZERS} model: $(model_file)" >&2
      return 1
    fi
    # The two formal profiles deliberately validate one deployable index
    # artifact set.  Fixed mode does not consume the extent classes while
    # serving queries, but it must not make an incomplete/mismatched index look
    # like a valid baseline that would need to be changed before running full.
    local require_graph_extent=false
    case "${GPU_DYNAMIC_GRAPH_ACCESS_MODE:-manual}" in
      fixed|adaptive) require_graph_extent=true ;;
      manual)
        [[ "${GPU_QUERY_GRAPH_READ_POLICY:-fixed}" == live-extent ]] &&
          require_graph_extent=true
        ;;
    esac
    if [[ "$require_graph_extent" == true &&
          ! -s "$(graph_extent_file)" ]]; then
      echo "missing Live-Extent sidecar: $(graph_extent_file)" >&2
      echo "generate it on a host that has every .dat shard, then copy it to the compute index prefix" >&2
      return 1
    fi
    if [[ "$require_graph_extent" == true ]]; then
      python3 - "$(graph_extent_file)" "$metadata" <<'PY_EXTENT_VALIDATE' || return 1
import json
import os
import struct
import sys

sidecar_path, metadata_path = sys.argv[1:]
with open(metadata_path, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)

HEADER_BYTES = 128
HEADER = struct.Struct('<8s10I10Q')
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def checksum64(data, state=FNV_OFFSET):
    for value in data:
        state ^= value
        state = (state * FNV_PRIME) & MASK64
    return state


errors = []
with open(sidecar_path, 'rb') as stream:
    raw_header = stream.read(HEADER_BYTES)
    if len(raw_header) != HEADER_BYTES:
        errors.append('header is truncated')
        values = None
    else:
        values = HEADER.unpack(raw_header)

    if values is not None:
        (magic, version, header_bytes, endian_marker, extent_quantum,
         class_bytes, pointer_bytes, entry_bytes, entry_capacity, shards,
         reserved0, nodes, payload_bytes, build_fingerprint,
         payload_checksum, header_checksum, *reserved) = values
        expected_entry_bytes = int(metadata.get('hot_graph_entry_size', 0))
        expected_capacity = (
            (expected_entry_bytes - 16) // 8
            if expected_entry_bytes >= 16 else 0)
        expected = {
            'magic': (magic, b'DVGEXT8\0'),
            'version': (version, 1),
            'header_bytes': (header_bytes, HEADER_BYTES),
            'endian_marker': (endian_marker, 0x01020304),
            'extent_quantum': (extent_quantum, 8),
            'class_bytes': (class_bytes, 1),
            'graph_pointer_bytes': (pointer_bytes, 8),
            'graph_entry_bytes': (entry_bytes, expected_entry_bytes),
            'graph_entry_capacity': (entry_capacity, expected_capacity),
            'num_shards': (shards, int(metadata.get('num_memory_nodes', 0))),
            'reserved0': (reserved0, 0),
            'num_nodes': (nodes, int(metadata.get('num_vectors', 0))),
            'payload_bytes': (payload_bytes, int(metadata.get('num_vectors', 0))),
            'build_fingerprint': (
                build_fingerprint,
                int(metadata.get('index_build_fingerprint', 0))),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                errors.append(f'{name}: sidecar={actual!r}, metadata={wanted!r}')
        if any(reserved):
            errors.append('reserved header fields are nonzero')

        checksum_header = bytearray(raw_header)
        checksum_header[80:88] = b'\0' * 8
        if checksum64(checksum_header) != header_checksum:
            errors.append('header checksum mismatch')

        payload_state = FNV_OFFSET
        remaining = payload_bytes
        maximum_class = (entry_capacity + 7) // 8
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                errors.append('payload is truncated')
                break
            if any(value > maximum_class for value in chunk):
                errors.append('payload contains an invalid extent class')
                remaining = 0
                break
            payload_state = checksum64(chunk, payload_state)
            remaining -= len(chunk)
        if remaining == 0 and payload_state != payload_checksum:
            errors.append('payload checksum mismatch')

        expected_file_bytes = HEADER_BYTES + payload_bytes
        if os.fstat(stream.fileno()).st_size != expected_file_bytes:
            errors.append(
                f'file size is {os.fstat(stream.fileno()).st_size}, '
                f'expected {expected_file_bytes}')

if errors:
    print(f'incompatible Live-Extent sidecar: {sidecar_path}', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    raise SystemExit(1)
PY_EXTENT_VALIDATE
    fi
  elif [[ "$role" == "storage" ]]; then
    local first=1 last="$SHARDS"
    if ((node_id > 0)); then first="$node_id"; last="$node_id"; fi
    local current
    for ((current = first; current <= last; ++current)); do
      for artifact in "$(shard_file "$current")" "$(idmap_file "$current")" \
                      "$(centroid_file "$current")" \
                      "$(navigation_code_file "$current")"; do
        if [[ ! -s "$artifact" ]]; then
          echo "missing storage artifact: $artifact" >&2
          return 1
        fi
      done
    done
  else
    echo "unknown index validation role: $role" >&2
    return 1
  fi
}

server_endpoints() {
  local index=0
  local endpoints=()
  for host in $HOSTS; do
    ((index >= SHARDS)) && break
    endpoints+=("${host}:$((BASE_PORT + index))")
    index=$((index + 1))
  done
  if ((${#endpoints[@]} != SHARDS)); then
    echo "HOSTS must contain $SHARDS entries; got ${#endpoints[@]}" >&2
    return 1
  fi
  printf '%s ' "${endpoints[@]}"
}

common_rdma_args() {
  local args=(--ib-port "$IB_PORT" --max-send-wrs "$MAX_SEND_WRS"
              --max-receive-wrs "$MAX_RECEIVE_WRS" --max-poll-cqes "$MAX_POLL_CQES")
  [[ -z "$IB_DEVICE" ]] || args+=(--ib-device "$IB_DEVICE")
  printf '%q ' "${args[@]}"
}

ensure_built() {
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "build directory is not configured: $BUILD_DIR" >&2
    return 1
  fi
  cmake --build "$BUILD_DIR" -j "$BUILD_JOBS" --target "$@"
}

write_service_config() {
  local output="${1:?output path is required}"
  local endpoints
  local enable_updates="${ENABLE_UPDATES:-true}"
  if [[ "$enable_updates" != "true" && "$enable_updates" != "false" ]]; then
    echo "ENABLE_UPDATES must be true or false: $enable_updates" >&2
    return 1
  fi
  endpoints="$(server_endpoints)"
  validate_index_metadata compute
  validate_vector_id_namespace_size
  resolve_mn_memory_gb

  {
    echo "servers = $endpoints"
    echo "initiator = true"
    echo "num-clients = 1"
    echo "port = 2234"
    echo "ib-port = $IB_PORT"
    [[ -z "$IB_DEVICE" ]] || echo "ib-device = $IB_DEVICE"
    echo "max-send-wrs = $MAX_SEND_WRS"
    echo "max-receive-wrs = $MAX_RECEIVE_WRS"
    echo "max-poll-cqes = $MAX_POLL_CQES"
    echo "index-prefix = $INDEX_PREFIX"
    echo "threads = $SERVICE_THREADS"
    echo "seed = ${SEED:-1234}"
    echo "vector-data-type = $VECTOR_DATA_TYPE"
    echo "dim = $DIM"
    echo "max-vectors = $MAX_VECTORS"
    echo "vector-id-namespace-size = $VECTOR_ID_NAMESPACE_SIZE"
    echo "R = $R"
    echo "beam-width-construction = $BUILD_BEAM"
    echo "alpha = $ALPHA"
    echo "k = $K"
    echo "mn-memory = $MN_MEMORY_GB"
    echo "gpu-device = $GPU_DEVICE"
    echo "enable-breakdown = ${ENABLE_BREAKDOWN:-true}"
    echo "enable-updates = $enable_updates"
    echo "storage-owner-update-completion-mode = $STORAGE_OWNER_UPDATE_COMPLETION_MODE"
    echo "gpu-dynamic-graph-access-mode = $GPU_DYNAMIC_GRAPH_ACCESS_MODE"
    echo "gpu-rdma-search-progression-mode = $GPU_RDMA_SEARCH_PROGRESSION_MODE"
    echo "gpu-exact-frontier-early-issue = ${GPU_EXACT_FRONTIER_EARLY_ISSUE:-false}"
    echo "gpu-query-slots = ${GPU_QUERY_SLOTS:-256}"
    echo "gpu-memory-limit-gb = ${GPU_MEMORY_LIMIT_GB:-40}"
    echo "gpu-memory-reserve-gb = ${GPU_MEMORY_RESERVE_GB:-4}"
    echo "gpu-bootstrap-window-mb = ${GPU_BOOTSTRAP_WINDOW_MB:-64}"
    echo "gpu-bootstrap-windows = ${GPU_BOOTSTRAP_WINDOWS:-4}"
    echo "gpu-graph-prefetch-depth = ${GPU_GRAPH_PREFETCH_DEPTH:-32}"
    echo "gpu-graph-commit-width = ${GPU_GRAPH_COMMIT_WIDTH:-0}"
    if [[ "$GPU_RDMA_SEARCH_PROGRESSION_MODE" == manual ]]; then
      echo "gpu-graph-issue-width = ${GPU_GRAPH_ISSUE_WIDTH:-0}"
      echo "gpu-query-beam-merge-policy = ${GPU_QUERY_BEAM_MERGE_POLICY:-legacy}"
    fi
    if [[ "$GPU_DYNAMIC_GRAPH_ACCESS_MODE" == manual ]]; then
      echo "gpu-query-graph-read-policy = ${GPU_QUERY_GRAPH_READ_POLICY:-fixed}"
      echo "gpu-dynamic-graph-extent = ${GPU_DYNAMIC_GRAPH_EXTENT:-true}"
    fi
    echo "query-rdma-trace-mode = ${QUERY_RDMA_TRACE_MODE:-off}"
    echo "query-rdma-trace-sample-rate = ${QUERY_RDMA_TRACE_SAMPLE_RATE:-1000}"
    [[ -z "${QUERY_RDMA_TRACE_OUTPUT:-}" ]] ||
      echo "query-rdma-trace-output = $QUERY_RDMA_TRACE_OUTPUT"
    echo "query-rdma-trace-events-per-query = ${QUERY_RDMA_TRACE_EVENTS_PER_QUERY:-1024}"
    echo "gpu-traversal-beam-width = ${GPU_TRAVERSAL_BEAM_WIDTH:-128}"
    echo "gpu-final-rerank-width = ${GPU_FINAL_RERANK_WIDTH:-128}"
    echo "gpu-max-expansions = ${GPU_MAX_EXPANSIONS:-384}"
    echo "gpu-rdma-qps = ${GPU_RDMA_QPS:-32}"
    echo "gpu-direct-timeout-ms = ${GPU_DIRECT_TIMEOUT_MS:-250}"
    echo "gpu-persistent-blocks-per-sm = ${GPU_PERSISTENT_BLOCKS_PER_SM:-4}"
    echo "storage-id = 0"
    echo "storage-peers = $endpoints"
    echo "storage-owner-batch-max = ${STORAGE_OWNER_BATCH_MAX:-32}"
    echo "storage-owner-batch-max-wait-us = ${STORAGE_OWNER_BATCH_MAX_WAIT_US:-10000}"
    echo "storage-owner-stage2-batch-max-wait-us = ${STORAGE_OWNER_STAGE2_BATCH_MAX_WAIT_US:-25000}"
    echo "storage-owner-stage2-initial-delay-ms = ${STORAGE_OWNER_STAGE2_INITIAL_DELAY_MS:-0}"
    echo "storage-owner-stage2-score-many = ${STORAGE_OWNER_STAGE2_SCORE_MANY:-true}"
    echo "storage-owner-stage2-home-rpc-combining = ${STORAGE_OWNER_STAGE2_HOME_RPC_COMBINING:-true}"
    echo "storage-owner-stage2-graph-issue-width = ${STORAGE_OWNER_STAGE2_GRAPH_ISSUE_WIDTH:-16}"
    echo "storage-owner-peer-qps-per-peer = ${STORAGE_OWNER_PEER_QPS_PER_PEER:-8}"
    echo "storage-owner-peer-rdma-tokens = ${STORAGE_OWNER_PEER_RDMA_TOKENS:-16}"
    echo "storage-owner-rpc-depth = ${STORAGE_OWNER_RPC_DEPTH:-16}"
    echo "storage-owner-rpc-timeout-ms = ${STORAGE_OWNER_RPC_TIMEOUT_MS:-30000}"
    echo "storage-owner-search-snapshot-batch = ${STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH:-256}"
    echo "storage-owner-maintenance-workers = ${STORAGE_OWNER_MAINTENANCE_WORKERS:-8}"
    echo "storage-owner-maintenance-queue-depth = ${STORAGE_OWNER_MAINTENANCE_QUEUE_DEPTH:-65536}"
    echo "storage-owner-reverse-queue-depth = ${STORAGE_OWNER_REVERSE_QUEUE_DEPTH:-65536}"
    echo "storage-owner-reverse-coalesce-max = ${STORAGE_OWNER_REVERSE_COALESCE_MAX:-256}"
  } > "$output"
}
