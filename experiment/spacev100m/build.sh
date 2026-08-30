#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/spacev100m_common.sh"

case "${VALIDATE_ONLY:-0}" in
  0) ;;
  1)
    BUILD_DIR="$BUILD_DIR" \
      "$SCRIPT_DIR/build_spacev100m_index.sh" \
      "${PROFILE:-04_gpu_persistent_gpunetio}"
    exit 0
    ;;
  *)
    echo "VALIDATE_ONLY must be 0 or 1: ${VALIDATE_ONLY}" >&2
    exit 2
    ;;
esac

BUILD_ROLE="${BUILD_ROLE:-storage}"
case "$BUILD_ROLE" in
  auto|all|compute|storage|offline) ;;
  *) echo "BUILD_ROLE must be auto, all, compute, storage, or offline: $BUILD_ROLE" >&2; exit 2 ;;
esac

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  configure_storage_only=OFF
  [[ "$BUILD_ROLE" != storage && "$BUILD_ROLE" != offline ]] || configure_storage_only=ON
  cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDVSTOR_STORAGE_NODE_ONLY="$configure_storage_only"
fi

storage_only=OFF
if grep -q '^DVSTOR_STORAGE_NODE_ONLY:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"; then
  storage_only=ON
fi

resolved_role="$BUILD_ROLE"
if [[ "$resolved_role" == auto ]]; then
  if [[ "$storage_only" == ON ]]; then
    resolved_role=storage
  else
    resolved_role=all
  fi
fi
if [[ "$storage_only" == ON && ("$resolved_role" == compute || "$resolved_role" == all) ]]; then
  cat >&2 <<EOF_ERROR
BUILD_DIR is configured with DVSTOR_STORAGE_NODE_ONLY=ON and has no compute targets:
  $BUILD_DIR
Use BUILD_ROLE=storage for this directory, or configure a separate compute build:
  cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build-compute" \\
    -DCMAKE_BUILD_TYPE=Release -DDVSTOR_STORAGE_NODE_ONLY=OFF
  BUILD_DIR="$PROJECT_DIR/build-compute" BUILD_ROLE=compute \\
    "$SCRIPT_DIR/build.sh"
EOF_ERROR
  exit 1
fi

case "$resolved_role" in
  storage)
    targets=(dvstor_memory_node vamana_offline_builder vamana_pq_indexer
             vamana_graph_extent_indexer vamana_metis_repartitioner)
    default_build_index=1
    ;;
  offline)
    targets=(vamana_offline_builder vamana_pq_indexer
             vamana_graph_extent_indexer vamana_metis_repartitioner)
    default_build_index=1
    ;;
  compute)
    targets=(dvstor_compute_node dvstor_breakdown_benchmark
             dvstor_sift101m_long_insert_recall)
    default_build_index=0
    ;;
  all)
    targets=(dvstor_compute_node dvstor_memory_node dvstor_breakdown_benchmark
             dvstor_sift101m_long_insert_recall vamana_offline_builder
             vamana_pq_indexer vamana_graph_extent_indexer
             vamana_metis_repartitioner)
    default_build_index=1
    ;;
esac

echo "[build] role=$resolved_role storage_only=$storage_only dir=$BUILD_DIR jobs=$BUILD_JOBS"
cmake --build "$BUILD_DIR" -j "$BUILD_JOBS" --target "${targets[@]}"

"$SCRIPT_DIR/prepare_spacev100m_data.sh"
if [[ "${BUILD_INDEX:-$default_build_index}" == 1 ]]; then
  BUILD_DIR="$BUILD_DIR" \
    "$SCRIPT_DIR/build_spacev100m_index.sh" \
    "${PROFILE:-04_gpu_persistent_gpunetio}"
fi
