#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:?project directory is required}"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEST_DIR"' EXIT

fail() {
  echo "mn_memory_estimator_test: $*" >&2
  exit 1
}

assert_equal() {
  local expected="$1"
  local actual="$2"
  local context="$3"
  [[ "$actual" == "$expected" ]] ||
    fail "$context: expected=$expected actual=$actual"
}

PREFIX="$TEST_DIR/schema16"
python3 - "${PREFIX}.meta.json" <<'PY_FIXTURE'
import json
import sys

metadata = {
    'schema_version': 16,
    'storage_format': 'vamana_tagged_v2',
    'num_vectors': 100_000_000,
    'num_memory_nodes': 5,
    'dim': 128,
    'hot_graph_entry_counts': [
        19_964_936, 20_600_998, 19_416_534, 20_600_998, 19_416_534,
    ],
    'dynamic_node_base_offsets': [
        20_444_099_040, 21_095_426_384, 19_882_535_296,
        21_095_426_384, 19_882_535_296,
    ],
    'hot_graph_dynamic_record_bytes': 1040,
    'allocation_size': 1040,
}
with open(sys.argv[1], 'w', encoding='utf-8') as stream:
    json.dump(metadata, stream)
PY_FIXTURE

metadata_estimate="$({
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD
  WORK_DIR="$TEST_DIR/work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$PREFIX"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=96
  VECTOR_DATA_TYPE=uint8
  PQ_SUBQUANTIZERS=32
  PARTITION_IMBALANCE=1.03
  MN_DYNAMIC_HEADROOM_PERCENT=20
  MN_MEMORY_MIN_GB=8
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
} 2>/dev/null)"
assert_equal 24 "$metadata_estimate" "metadata-driven estimate"

profile_estimate="$({
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD INDEX_PREFIX
  WORK_DIR="$TEST_DIR/profile-work"
  INDEX_DIR="$WORK_DIR/index"
  PQ_INDEX_PREFIX="$PREFIX"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=96
  VECTOR_DATA_TYPE=uint8
  PQ_SUBQUANTIZERS=32
  MN_DYNAMIC_HEADROOM_PERCENT=20
  MN_MEMORY_MIN_GB=8
  source "$PROJECT_DIR/experiment/common.sh"
  [[ -z "${MN_MEMORY_GB:-}" ]] ||
    fail "MN_MEMORY_GB was resolved before the profile"
  load_experiment_profile 04_gpu_persistent_gpunetio
  resolve_mn_memory_gb
  printf '%s' "$MN_MEMORY_GB"
} 2>/dev/null)"
assert_equal 24 "$profile_estimate" "profile-delayed estimate"

explicit_estimate="$({
  MN_MEMORY_GB=37
  WORK_DIR="$TEST_DIR/explicit-work"
  INDEX_DIR="$WORK_DIR/index"
  PQ_INDEX_PREFIX="$PREFIX"
  source "$PROJECT_DIR/experiment/common.sh"
  load_experiment_profile 04_gpu_persistent_gpunetio
  resolve_mn_memory_gb
  printf '%s' "$MN_MEMORY_GB"
} 2>/dev/null)"
assert_equal 37 "$explicit_estimate" "explicit override"

fallback_estimate="$({
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD
  WORK_DIR="$TEST_DIR/fallback-work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$WORK_DIR/missing"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=96
  VECTOR_DATA_TYPE=uint8
  PQ_SUBQUANTIZERS=32
  PARTITION_IMBALANCE=1.03
  MN_DYNAMIC_HEADROOM_PERCENT=20
  MN_MEMORY_MIN_GB=8
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
} 2>/dev/null)"
assert_equal 24 "$fallback_estimate" "schema16 fallback estimate"

float_estimate="$({
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD
  WORK_DIR="$TEST_DIR/float-work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$WORK_DIR/missing"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=96
  VECTOR_DATA_TYPE=float32
  PQ_SUBQUANTIZERS=32
  PARTITION_IMBALANCE=1.03
  MN_DYNAMIC_HEADROOM_PERCENT=20
  MN_MEMORY_MIN_GB=1
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
} 2>/dev/null)"
assert_equal 33 "$float_estimate" "float32 fallback estimate"

r64_estimate="$({
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD
  WORK_DIR="$TEST_DIR/r64-work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$WORK_DIR/missing"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=64
  VECTOR_DATA_TYPE=uint8
  PQ_SUBQUANTIZERS=32
  PARTITION_IMBALANCE=1.03
  MN_DYNAMIC_HEADROOM_PERCENT=20
  MN_MEMORY_MIN_GB=1
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
} 2>/dev/null)"
assert_equal 18 "$r64_estimate" "R-sensitive fallback estimate"

absolute_slots_estimate="$({
  unset MN_MEMORY_GB
  MN_DYNAMIC_SLOTS_PER_SHARD=1000000
  WORK_DIR="$TEST_DIR/slots-work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$PREFIX"
  SHARDS=5
  MAX_VECTORS=100000000
  DIM=128
  R=96
  VECTOR_DATA_TYPE=uint8
  PQ_SUBQUANTIZERS=32
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
} 2>/dev/null)"
assert_equal 21 "$absolute_slots_estimate" "absolute dynamic-slot estimate"

BAD_PREFIX="$TEST_DIR/bad"
printf '{' > "${BAD_PREFIX}.meta.json"
if (
  unset MN_MEMORY_GB MN_DYNAMIC_SLOTS_PER_SHARD
  WORK_DIR="$TEST_DIR/bad-work"
  INDEX_DIR="$WORK_DIR/index"
  INDEX_PREFIX="$BAD_PREFIX"
  SHARDS=5
  source "$PROJECT_DIR/experiment/sift100m_common.sh"
  estimate_mn_memory_gb
) >/dev/null 2>&1; then
  fail "malformed metadata was silently accepted"
fi

echo "mn_memory_estimator_test passed"
