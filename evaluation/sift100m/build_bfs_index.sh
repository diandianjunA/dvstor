#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

BINARY="${BUILD_DIR:-$PROJECT_DIR/build}/vamana_offline_builder"
if [[ ! -x "$BINARY" ]]; then
  echo "error: missing builder: $BINARY" >&2
  echo "build it with: cmake --build ${BUILD_DIR:-$PROJECT_DIR/build} -j --target vamana_offline_builder" >&2
  exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "error: SIFT100M data file not found: $DATA_FILE" >&2
  exit 1
fi

mkdir -p "$INDEX_DIR"

echo "[SIFT100M BFS Build]"
echo "  data:              $DATA_FILE"
echo "  output prefix:     $INDEX_PREFIX"
echo "  memory nodes:      $MEMORY_NODES"
echo "  max vectors:       $MAX_VECTORS"
echo "  dim:               $DIM"
echo "  R:                 $R"
echo "  beam width:        $BEAM_WIDTH"
echo "  partition:         $PARTITION_STRATEGY"
echo "  reverse mode:      $OFFLINE_REVERSE_MODE"
echo "  skip sanity check: $SKIP_SANITY_CHECK"
echo "  node layout:       $NODE_LAYOUT"
echo "  rabitq bits:       $RABITQ_BITS"
echo "  threads:           $BUILD_THREADS"
echo "  gpu enabled:       $BUILD_USE_GPU"
echo "  gpu device:        $BUILD_GPU_DEVICE"
echo ""

args=(
  --data-path "$DATA_FILE"
  --output-prefix "$INDEX_PREFIX"
  --memory-nodes "$MEMORY_NODES"
  --max-vectors "$MAX_VECTORS"
  --R "$R"
  --beam-width "$BEAM_WIDTH"
  --alpha "$ALPHA"
  --rabitq-bits "$RABITQ_BITS"
  --node-layout "$NODE_LAYOUT"
  --partition-strategy "$PARTITION_STRATEGY"
  --partition-max-degree "$PARTITION_MAX_DEGREE"
  --partition-imbalance "$PARTITION_IMBALANCE"
  --offline-reverse-mode "$OFFLINE_REVERSE_MODE"
  --threads "$BUILD_THREADS"
)

if [[ "$BUILD_USE_GPU" == "true" || "$BUILD_USE_GPU" == "1" || "$BUILD_USE_GPU" == "yes" ]]; then
  args+=(--gpu-device "$BUILD_GPU_DEVICE")
else
  args+=(--no-gpu)
fi

if [[ "$SKIP_SANITY_CHECK" == "true" || "$SKIP_SANITY_CHECK" == "1" || "$SKIP_SANITY_CHECK" == "yes" ]]; then
  args+=(--skip-sanity-check)
fi

exec "$BINARY" "${args[@]}" "$@"
