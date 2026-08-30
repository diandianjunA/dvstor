#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/sift100m"
PROFILE_DIR="$EXPERIMENT_DIR/profiles"
COMMON_PROFILE="$PROFILE_DIR/04_gpu_persistent_gpunetio_common.sh"
BASELINE_PROFILE=04_gpu_persistent_gpunetio_baseline
FULL_PROFILE=04_gpu_persistent_gpunetio
TEST_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEST_DIR"' EXIT

mapfile -t profile_files < <(
  find "$PROFILE_DIR" -maxdepth 1 -name '*.env' -printf '%f\n' | sort
)
expected_profiles=(
  04_gpu_persistent_gpunetio.env
  04_gpu_persistent_gpunetio_baseline.env
)
[[ "${profile_files[*]}" == "${expected_profiles[*]}" ]]

# A formal profile may select only its label and the three contribution-level
# modes.  Every lower-level parameter must come from the one shared file.
expected_profile_fields=(
  GPU_DYNAMIC_GRAPH_ACCESS_MODE
  GPU_RDMA_SEARCH_PROGRESSION_MODE
  STORAGE_OWNER_UPDATE_COMPLETION_MODE
  SYSTEM_VARIANT_LABEL
)
for profile in "$BASELINE_PROFILE" "$FULL_PROFILE"; do
  mapfile -t actual_profile_fields < <(
    sed -nE 's/^([A-Z][A-Z0-9_]*)=.*/\1/p' "$PROFILE_DIR/$profile.env" | sort
  )
  [[ "${actual_profile_fields[*]}" == "${expected_profile_fields[*]}" ]]
  grep -Fq 'source "$EXPERIMENT_DIR/profiles/04_gpu_persistent_gpunetio_common.sh"' \
    "$PROFILE_DIR/$profile.env"
done

mapfile -t common_fields < <(
  sed -nE 's/^([A-Z][A-Z0-9_]*)=.*/\1/p' "$COMMON_PROFILE"
)
[[ ${#common_fields[@]} -gt 0 ]]

run_clean_profile() {
  local profile="${1:?profile is required}"
  local action="${2:?action is required}"
  shift 2
  env -i \
    PATH="$PATH" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    INDEX_DIR=/tmp/dvstor-profile-contract \
    R=96 \
    BUILD_BEAM=128 \
    "$@" \
    bash --noprofile --norc -c '
      set -euo pipefail
      profile=$1
      action=$2
      shift 2
      source "$EXPERIMENT_DIR/profiles/$profile.env"
      case "$action" in
        identity)
          if [[ "$STORAGE_OWNER_UPDATE_COMPLETION_MODE" == coupled ]]; then
            update_api=append-only
          else
            update_api=insert-upsert-erase
          fi
          printf "%s|%s|%s|%s|%s\n" \
            "$SYSTEM_VARIANT_LABEL" \
            "$STORAGE_OWNER_UPDATE_COMPLETION_MODE" \
            "$GPU_DYNAMIC_GRAPH_ACCESS_MODE" \
            "$GPU_RDMA_SEARCH_PROGRESSION_MODE" \
            "$update_api"
          ;;
        common-snapshot)
          for field in "$@"; do
            printf "%s=%s\n" "$field" "${!field}"
          done
          ;;
        index-prefix)
          printf "%s\n" "$INDEX_PREFIX"
          ;;
        *)
          echo "unknown profile-contract action: $action" >&2
          exit 2
          ;;
      esac
    ' bash "$profile" "$action" "${common_fields[@]}"
}

baseline_identity="$(run_clean_profile "$BASELINE_PROFILE" identity)"
full_identity="$(run_clean_profile "$FULL_PROFILE" identity)"
[[ "$baseline_identity" == 'baseline|coupled|fixed|coupled|append-only' ]]
[[ "$full_identity" == \
   'full|decoupled|adaptive|decoupled|insert-upsert-erase' ]]

# Byte-for-byte compare a canonical name=value snapshot of every setting owned
# by the shared profile.  INDEX_PREFIX is also asserted explicitly because the
# headline comparison must use one physical index, not merely one schema.
baseline_snapshot="$(run_clean_profile "$BASELINE_PROFILE" common-snapshot)"
full_snapshot="$(run_clean_profile "$FULL_PROFILE" common-snapshot)"
[[ "$baseline_snapshot" == "$full_snapshot" ]]
[[ "$(run_clean_profile "$BASELINE_PROFILE" index-prefix)" == \
   "$(run_clean_profile "$FULL_PROFILE" index-prefix)" ]]

grep -Fq 'GPU_GRAPH_COMMIT_WIDTH=16' <<<"$baseline_snapshot"
grep -Fq 'GPU_GRAPH_ISSUE_WIDTH=32' <<<"$baseline_snapshot"
grep -Fq 'GPU_QUERY_BEAM_MERGE_POLICY=stable-run' <<<"$baseline_snapshot"
grep -Fq 'GPU_QUERY_GRAPH_READ_POLICY=live-extent' <<<"$baseline_snapshot"
grep -Fq 'GPU_DYNAMIC_GRAPH_EXTENT=true' <<<"$baseline_snapshot"
grep -Fq 'STORAGE_OWNER_STAGE2_SCORE_MANY=true' <<<"$baseline_snapshot"
grep -Fq 'STORAGE_OWNER_STAGE2_GRAPH_ISSUE_WIDTH=16' <<<"$baseline_snapshot"
grep -Fq 'STORAGE_OWNER_STAGE2_HOME_RPC_COMBINING=true' <<<"$baseline_snapshot"
grep -Fq 'ENABLE_BREAKDOWN=false' <<<"$baseline_snapshot"

# The full system is exactly the baseline with all three contribution switches
# enabled; no lower-level setting changes in this cumulative ablation.
baseline_all_on_identity="$(
  run_clean_profile "$BASELINE_PROFILE" identity \
    STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
    GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
    GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
)"
baseline_all_on_snapshot="$(
  run_clean_profile "$BASELINE_PROFILE" common-snapshot \
    STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled \
    GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive \
    GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
)"
[[ "$baseline_all_on_identity" == \
   'baseline|decoupled|adaptive|decoupled|insert-upsert-erase' ]]
[[ "${baseline_all_on_identity#*|}" == "${full_identity#*|}" ]]
[[ "$baseline_all_on_snapshot" == "$full_snapshot" ]]

# The coupled profile's append-only label is an execution boundary, not a
# documentation convention: its implementation may update the local centroid
# and remote graph records one-sided, but must not regain a target-side
# centroid or dynamic-reclaim RPC helper.
exact_execution="$PROJECT_DIR/src/memory_node/storage_owner_runtime/exact_execution.cc"
grep -Fq 'reconcile_reverse_ops_one_sided(' "$exact_execution"
grep -Fq 'apply_local_centroid_membership_ops(' "$exact_execution"
! grep -Fq 'apply_centroid_membership_fanout_and_wait(' "$exact_execution"
! grep -Fq 'control_dynamic_node_on_shard(' "$exact_execution"

# Invalid umbrella values must fail before either endpoint is started.
if run_clean_profile "$BASELINE_PROFILE" identity \
  STORAGE_OWNER_UPDATE_COMPLETION_MODE=invalid >/dev/null 2>&1; then
  exit 1
fi
if run_clean_profile "$BASELINE_PROFILE" identity \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE=invalid >/dev/null 2>&1; then
  exit 1
fi
if run_clean_profile "$BASELINE_PROFILE" identity \
  GPU_RDMA_SEARCH_PROGRESSION_MODE=invalid >/dev/null 2>&1; then
  exit 1
fi

# Both launch paths must forward the same three values: compute via INI and
# every memory node via CLI.
declare -A mode_variables=(
  [storage-owner-update-completion-mode]=STORAGE_OWNER_UPDATE_COMPLETION_MODE
  [gpu-dynamic-graph-access-mode]=GPU_DYNAMIC_GRAPH_ACCESS_MODE
  [gpu-rdma-search-progression-mode]=GPU_RDMA_SEARCH_PROGRESSION_MODE
)
for key in "${!mode_variables[@]}"; do
  mode_variable="${mode_variables[$key]}"
  grep -Fq "echo \"$key = \$$mode_variable\"" \
    "$EXPERIMENT_DIR/sift100m_common.sh"
  grep -Fq -- "--$key \"\$$mode_variable\"" \
    "$EXPERIMENT_DIR/start_memory_node.sh"
done

for runner in run_breakdown.sh; do
  grep -Fq -- '--profile-name "$PROFILE"' "$EXPERIMENT_DIR/$runner"
  grep -Fq -- '--system-variant-label "$SYSTEM_VARIANT_LABEL"' \
    "$EXPERIMENT_DIR/$runner"
done

# The server-side resolver sees the same common commit width as compute.  Full
# decoupled progression derives its issue width from this value; omitting it
# would combine the server default commit=32 with auto issue=32 and fail.
grep -Fq -- '--gpu-graph-commit-width "$GPU_GRAPH_COMMIT_WIDTH"' \
  "$EXPERIMENT_DIR/start_memory_node.sh"

generate_service_config() {
  local profile="${1:?profile is required}"
  local output="${2:?output is required}"
  shift 2
  (
    unset STORAGE_OWNER_UPDATE_COMPLETION_MODE
    unset GPU_DYNAMIC_GRAPH_ACCESS_MODE
    unset GPU_RDMA_SEARCH_PROGRESSION_MODE
    for assignment in "$@"; do
      export "$assignment"
    done
    WORK_DIR="$TEST_DIR/work-$profile"
    INDEX_DIR="$TEST_DIR/index"
    REPORT_DIR="$TEST_DIR/reports"
    LOG_DIR="$TEST_DIR/logs"
    PID_DIR="$TEST_DIR/pids"
    SHARDS=1
    HOSTS=127.0.0.1
    MN_MEMORY_GB=8
    source "$EXPERIMENT_DIR/common.sh"
    load_experiment_profile "$profile"
    # This test exercises config rendering, not index artifact validation.
    validate_index_metadata() { :; }
    resolve_mn_memory_gb() { :; }
    write_service_config "$output"
  )
}

baseline_ini="$TEST_DIR/baseline.ini"
full_ini="$TEST_DIR/full.ini"
manual_ini="$TEST_DIR/manual.ini"
generate_service_config "$BASELINE_PROFILE" "$baseline_ini"
generate_service_config "$FULL_PROFILE" "$full_ini"

for ini in "$baseline_ini" "$full_ini"; do
  grep -Fq 'storage-owner-update-completion-mode = ' "$ini"
  grep -Fq 'gpu-dynamic-graph-access-mode = ' "$ini"
  grep -Fq 'gpu-rdma-search-progression-mode = ' "$ini"
  ! grep -Fq 'gpu-query-graph-read-policy = ' "$ini"
  ! grep -Fq 'gpu-dynamic-graph-extent = ' "$ini"
  ! grep -Fq 'gpu-graph-issue-width = ' "$ini"
  ! grep -Fq 'gpu-query-beam-merge-policy = ' "$ini"
done

# Apart from the three umbrella lines, the executable compute manifests are
# byte-identical, including index-prefix and all common capacities.
sed -E '/^(storage-owner-update-completion-mode|gpu-dynamic-graph-access-mode|gpu-rdma-search-progression-mode) = /d' \
  "$baseline_ini" > "$TEST_DIR/baseline-common.ini"
sed -E '/^(storage-owner-update-completion-mode|gpu-dynamic-graph-access-mode|gpu-rdma-search-progression-mode) = /d' \
  "$full_ini" > "$TEST_DIR/full-common.ini"
cmp -s "$TEST_DIR/baseline-common.ini" "$TEST_DIR/full-common.ini"

# Fine-grained historical diagnostics remain possible only by explicitly
# selecting manual mode; only then are umbrella-owned children rendered.
generate_service_config "$BASELINE_PROFILE" "$manual_ini" \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE=manual \
  GPU_RDMA_SEARCH_PROGRESSION_MODE=manual \
  GPU_QUERY_GRAPH_READ_POLICY=fixed \
  GPU_DYNAMIC_GRAPH_EXTENT=false \
  GPU_GRAPH_ISSUE_WIDTH=16 \
  GPU_QUERY_BEAM_MERGE_POLICY=legacy
grep -Fq 'gpu-query-graph-read-policy = fixed' "$manual_ini"
grep -Fq 'gpu-dynamic-graph-extent = false' "$manual_ini"
grep -Fq 'gpu-graph-issue-width = 16' "$manual_ini"
grep -Fq 'gpu-query-beam-merge-policy = legacy' "$manual_ini"

# Baseline and full must accept exactly the same physical schema-16 artifact
# set.  In particular fixed mode may ignore extent classes at query time, but
# it must reject a missing, corrupt, or differently-built .gextent8 just as
# adaptive mode does.  Storage validation likewise requires the same bound PQ
# code sidecar for both profiles.
artifact_index_dir="$TEST_DIR/artifact-index"
artifact_converted_dir="$TEST_DIR/artifact-converted"
artifact_data_file="$artifact_converted_dir/base_1.u8bin"
artifact_partition_strategy=balanced
artifact_prefix="$artifact_index_dir/sift100m_R96_bw128_${artifact_partition_strategy}_pmd32_pq32_schema16"
python3 - "$artifact_prefix" "$artifact_data_file" \
  "$artifact_partition_strategy" <<'PY_ARTIFACT_FIXTURE'
import json
import os
import struct
import sys

prefix, data_file, partition_strategy = sys.argv[1:]
os.makedirs(os.path.dirname(prefix), exist_ok=True)
fingerprint = 0x123456789ABCDEF0
entry_bytes = 800
entry_capacity = (entry_bytes - 16) // 8
metadata = {
    'output_prefix': prefix,
    'data_file': data_file,
    'schema_version': 16,
    'distance': 'l2',
    'node_layout': 'plain',
    'storage_format': 'vamana_tagged_v2',
    'remote_ptr_format': 'tagged_inc24_shard6_off34x16_v1',
    'navigation_execution': 'gpu_beam_v1',
    'R': 96,
    'beam_width_construction': 128,
    'dim': 128,
    'num_vectors': 1,
    'num_memory_nodes': 1,
    'vector_data_type': 'uint8',
    'navigation_code_bytes': 32,
    'pq_subquantizers': 32,
    'pq_bits': 8,
    'partition_strategy': partition_strategy,
    'partition_imbalance': 1.03,
    'alpha': 1.2,
    'partition_max_degree': 32,
    'idmap_format': 'owner_sharded_v2_bound',
    'centroid_state_format': 'physical_shard_centroid_v2_bound',
    'hot_graph_pointer_bytes': 8,
    'navigation_quantizer': 'opq_pq',
    'navigation_format': 'opq_pq_graph_v1',
    'navigation_model_checksum': 0x31415926,
    'index_build_fingerprint': fingerprint,
    'shard_build_fingerprints': [0x27182818],
    'hot_graph_offsets': [4096],
    'hot_graph_entry_counts': [1],
    'hot_graph_dynamic_base_offsets': [8192],
    'navigation_code_remote_offsets': [12288],
    'navigation_code_region_bytes': [32],
    'storage_control_remote_offsets': [16384],
    'dynamic_node_base_offsets': [32768],
    'hot_graph_dynamic_hot_offset': 16,
    'hot_graph_entry_size': entry_bytes,
    'dynamic_navigation_code_offset': 16 + entry_bytes,
    'hot_graph_dynamic_record_bytes': 16 + entry_bytes + 4 + 32 + 4,
    'dynamic_navigation_code_validation_bytes': 4,
    'dynamic_navigation_code_checksum_bytes': 4,
}
with open(prefix + '.meta.json', 'w', encoding='utf-8') as stream:
    json.dump(metadata, stream)
for suffix in (
    '.pq32',
    '_node1_of1.dat',
    '_node1_of1.idmap',
    '_node1_of1.centroid',
    '_node1_of1.pq32.codes',
):
    with open(prefix + suffix, 'wb') as stream:
        stream.write(b'fixture')

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def checksum64(data):
    state = FNV_OFFSET
    for value in data:
        state ^= value
        state = (state * FNV_PRIME) & MASK64
    return state


payload = b'\0'
header_format = struct.Struct('<8s10I10Q')
header_values = (
    b'DVGEXT8\0', 1, 128, 0x01020304, 8, 1, 8,
    entry_bytes, entry_capacity, 1, 0,
    1, len(payload), fingerprint, checksum64(payload), 0,
    0, 0, 0, 0, 0,
)
header = bytearray(header_format.pack(*header_values))
header[80:88] = struct.pack('<Q', checksum64(header))
with open(prefix + '.gextent8', 'wb') as stream:
    stream.write(header)
    stream.write(payload)
PY_ARTIFACT_FIXTURE

validate_profile_artifacts() {
  local profile="${1:?profile is required}"
  local role="${2:?role is required}"
  env -i \
    PATH="$PATH" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    INDEX_DIR="$artifact_index_dir" \
    CONVERTED_DIR="$artifact_converted_dir" \
    R=96 \
    BUILD_BEAM=128 \
    DIM=128 \
    MAX_VECTORS=1 \
    SHARDS=1 \
    VECTOR_DATA_TYPE=uint8 \
    bash --noprofile --norc -c '
      set -euo pipefail
      source "$EXPERIMENT_DIR/common.sh"
      load_experiment_profile "$1"
      validate_index_metadata "$2" 1
    ' bash "$profile" "$role"
}

expect_both_profiles_reject_artifacts() {
  local role="${1:?role is required}"
  for profile in "$BASELINE_PROFILE" "$FULL_PROFILE"; do
    if validate_profile_artifacts "$profile" "$role" >/dev/null 2>&1; then
      echo "$profile unexpectedly accepted an incomplete/mismatched $role artifact set" >&2
      exit 1
    fi
  done
}

for profile in "$BASELINE_PROFILE" "$FULL_PROFILE"; do
  validate_profile_artifacts "$profile" compute
  validate_profile_artifacts "$profile" storage
done

mv "$artifact_prefix.gextent8" "$artifact_prefix.gextent8.held"
expect_both_profiles_reject_artifacts compute
mv "$artifact_prefix.gextent8.held" "$artifact_prefix.gextent8"

mv "$artifact_prefix.pq32" "$artifact_prefix.pq32.held"
expect_both_profiles_reject_artifacts compute
mv "$artifact_prefix.pq32.held" "$artifact_prefix.pq32"

mv "$artifact_prefix"_node1_of1.pq32.codes \
  "$artifact_prefix"_node1_of1.pq32.codes.held
expect_both_profiles_reject_artifacts storage
mv "$artifact_prefix"_node1_of1.pq32.codes.held \
  "$artifact_prefix"_node1_of1.pq32.codes

# A checksummed sidecar from another build is still the wrong artifact.
python3 - "$artifact_prefix.gextent8" <<'PY_REBIND_EXTENT'
import struct
import sys

path = sys.argv[1]
with open(path, 'rb') as stream:
    contents = bytearray(stream.read())
struct.pack_into('<Q', contents, 64, 0xBADF00D)
struct.pack_into('<Q', contents, 80, 0)
state = 1469598103934665603
for value in contents[:128]:
    state ^= value
    state = (state * 1099511628211) & ((1 << 64) - 1)
struct.pack_into('<Q', contents, 80, state)
with open(path, 'wb') as stream:
    stream.write(contents)
PY_REBIND_EXTENT
expect_both_profiles_reject_artifacts compute

# The build fingerprint is part of the common index contract, independent of
# which query access mode will consume the sidecar.
python3 - "$artifact_prefix.meta.json" <<'PY_ZERO_FINGERPRINT'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)
metadata['index_build_fingerprint'] = 0
with open(path, 'w', encoding='utf-8') as stream:
    json.dump(metadata, stream)
PY_ZERO_FINGERPRINT
expect_both_profiles_reject_artifacts compute
