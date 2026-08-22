#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_DIR/experiment"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio_baseline}"
RUN_ROOT="${RUN_ROOT:-$PROGRAM_DIR/results/program1_quality_$(date +%Y%m%d_%H%M%S)}"
QUALITY_STAGE2_DELAY_MS="${QUALITY_STAGE2_DELAY_MS:-15000}"
QUALITY_SETTLE_SECONDS="${QUALITY_SETTLE_SECONDS:-30}"
QUALITY_INSERT_COUNT="${QUALITY_INSERT_COUNT:-1000}"
QUALITY_RECALL_QUERIES="${QUALITY_RECALL_QUERIES:-1000}"

echo "请先在存储节点运行："
echo "  QUALITY_STAGE2_DELAY_MS=$QUALITY_STAGE2_DELAY_MS ./motivation/program1/start_storage_case.sh quality"
read -r -p "存储节点就绪后按 Enter（输入 q 退出）: " answer
[[ "$answer" != q && "$answer" != Q ]] || exit 130

source "$EXPERIMENT_DIR/common.sh"
load_experiment_profile "$PROFILE"
export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
export STORAGE_OWNER_STAGE2_INITIAL_DELAY_MS="$QUALITY_STAGE2_DELAY_MS"
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
export ENABLE_BREAKDOWN=true

ensure_built dvstor_sift101m_long_insert_recall
insert_path="$(insert_bin)"
[[ -s "$insert_path" ]] || { echo "missing insert file: $insert_path" >&2; exit 1; }
mkdir -p "$RUN_ROOT/quality"
config="$RUN_ROOT/quality/service.ini"
report="$RUN_ROOT/quality/quality.json"
write_service_config "$config"

"$BUILD_DIR/dvstor_sift101m_long_insert_recall" \
  --service-config "$config" \
  --insert-file "$insert_path" \
  --insert-start-id "${QUALITY_INSERT_START_ID:-$((MAX_VECTORS + 1000000))}" \
  --insert-count "$QUALITY_INSERT_COUNT" \
  --insert-threads "${QUALITY_INSERT_THREADS:-16}" \
  --insert-batch-size "${QUALITY_INSERT_BATCH_SIZE:-16}" \
  --self-recall-queries "$QUALITY_RECALL_QUERIES" \
  --self-recall-k "${QUALITY_RECALL_K:-10}" \
  --settle-seconds "$QUALITY_SETTLE_SECONDS" \
  --report-json "$report" \
  --report-text "$RUN_ROOT/quality/quality.txt" \
  2>&1 | tee "$RUN_ROOT/quality/driver.log"

if [[ -f "$RUN_ROOT/manifest.tsv" ]]; then
  printf 'quality\t%s\n' "$report" >> "$RUN_ROOT/manifest.tsv"
fi
echo "quality report: $report"
echo "汇总与画图：python3 $PROGRAM_DIR/summarize_program1.py $RUN_ROOT"
