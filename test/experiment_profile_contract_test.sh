#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"

prepare_profile_environment() {
  export EXPERIMENT_DIR INDEX_DIR=/tmp/dvstor-profile-contract
  export R=96 BUILD_BEAM=128
}

common_signature() {
  printf '%s\n' \
    "$PARTITION_STRATEGY|$PARTITION_MAX_DEGREE|$PQ_SUBQUANTIZERS|$INDEX_PREFIX" \
    "$SERVICE_THREADS|$GPU_QUERY_SLOTS|$GPU_MEMORY_LIMIT_GB|$GPU_MEMORY_RESERVE_GB" \
    "$GPU_BOOTSTRAP_WINDOW_MB|$GPU_BOOTSTRAP_WINDOWS|$GPU_GRAPH_PREFETCH_DEPTH" \
    "$GPU_GRAPH_COMMIT_WIDTH" \
    "$QUERY_RDMA_TRACE_MODE|$GPU_TRAVERSAL_BEAM_WIDTH|$GPU_FINAL_RERANK_WIDTH" \
    "$GPU_MAX_EXPANSIONS|$GPU_RDMA_QPS|$GPU_PERSISTENT_BLOCKS_PER_SM" \
    "$STORAGE_OWNER_BATCH_MAX|$STORAGE_OWNER_BATCH_MAX_WAIT_US" \
    "$STORAGE_OWNER_STAGE2_BATCH_MAX_WAIT_US|$STORAGE_OWNER_PEER_QPS_PER_PEER" \
    "$STORAGE_OWNER_PEER_RDMA_TOKENS|$STORAGE_OWNER_RPC_DEPTH" \
    "$STORAGE_OWNER_RPC_TIMEOUT_MS|$STORAGE_OWNER_SEARCH_SNAPSHOT_BATCH" \
    "$STORAGE_OWNER_MAINTENANCE_WORKERS|$STORAGE_OWNER_MAINTENANCE_QUEUE_DEPTH" \
    "$STORAGE_OWNER_REVERSE_QUEUE_DEPTH|$STORAGE_OWNER_REVERSE_COALESCE_MAX" \
    "$ENABLE_BREAKDOWN"
}

mapfile -t profile_files < <(
  find "$EXPERIMENT_DIR/profiles" -maxdepth 1 -name '*.env' -printf '%f\n' |
    sort
)
expected_profiles="04_gpu_persistent_gpunetio.env 04_gpu_persistent_gpunetio_baseline.env"
[[ "${profile_files[*]}" == "$expected_profiles" ]]

baseline_signature="$({
  prepare_profile_environment
  source "$EXPERIMENT_DIR/profiles/04_gpu_persistent_gpunetio_baseline.env"
  [[ "$GPU_GRAPH_COMMIT_WIDTH/$GPU_GRAPH_ISSUE_WIDTH" == "16/16" ]]
  [[ "$GPU_QUERY_BEAM_MERGE_POLICY" == legacy ]]
  [[ "$GPU_QUERY_GRAPH_READ_POLICY" == fixed ]]
  [[ "$GPU_DYNAMIC_GRAPH_EXTENT" == false ]]
  [[ "$STORAGE_OWNER_STAGE2_SCORE_MANY" == false ]]
  [[ "$STORAGE_OWNER_STAGE2_GRAPH_ISSUE_WIDTH" == 1 ]]
  [[ "$STORAGE_OWNER_STAGE2_HOME_RPC_COMBINING" == false ]]
  common_signature
})"

optimized_signature="$({
  prepare_profile_environment
  source "$EXPERIMENT_DIR/profiles/04_gpu_persistent_gpunetio.env"
  [[ "$GPU_GRAPH_COMMIT_WIDTH/$GPU_GRAPH_ISSUE_WIDTH" == "16/32" ]]
  [[ "$GPU_QUERY_BEAM_MERGE_POLICY" == stable-run ]]
  [[ "$GPU_QUERY_GRAPH_READ_POLICY" == live-extent ]]
  [[ "$GPU_DYNAMIC_GRAPH_EXTENT" == true ]]
  [[ "$STORAGE_OWNER_STAGE2_SCORE_MANY" == true ]]
  [[ "$STORAGE_OWNER_STAGE2_GRAPH_ISSUE_WIDTH" == 16 ]]
  [[ "$STORAGE_OWNER_STAGE2_HOME_RPC_COMBINING" == true ]]
  common_signature
})"

[[ "$baseline_signature" == "$optimized_signature" ]]

# A full-minus-one ablation uses the same profile and only overrides the
# selected runtime feature; no third profile is needed.
(
  prepare_profile_environment
  export STORAGE_OWNER_STAGE2_SCORE_MANY=false
  source "$EXPERIMENT_DIR/profiles/04_gpu_persistent_gpunetio.env"
  [[ "$STORAGE_OWNER_STAGE2_SCORE_MANY" == false ]]
  [[ "$GPU_QUERY_BEAM_MERGE_POLICY" == stable-run ]]
)
