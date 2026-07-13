#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

PREPARE_BASE="${PREPARE_BASE:-1}"
PREPARE_QUERY="${PREPARE_QUERY:-1}"
PREPARE_GROUNDTRUTH="${PREPARE_GROUNDTRUTH:-1}"
convert_args=()
if [[ "$PREPARE_BASE" != "1" ]]; then convert_args+=(--skip-base); fi
if [[ "$PREPARE_QUERY" != "1" ]]; then convert_args+=(--skip-query); fi
if [[ "$PREPARE_GROUNDTRUTH" != "1" ]]; then convert_args+=(--skip-groundtruth); fi

python3 "$SCRIPT_DIR/convert_sift100m.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$CONVERTED_DIR" \
  --groundtruth-label "$GROUNDTRUTH_LABEL" \
  --max-base "$MAX_VECTORS" \
  --max-query "$MAX_QUERIES" \
  --topk "$GROUNDTRUTH_TOPK" \
  --chunk-rows "${CONVERT_CHUNK_ROWS:-1000000}" \
  "${convert_args[@]}"

if [[ "$PREPARE_BASE" == "1" ]]; then echo "base:        $(base_bin)"; fi
if [[ "$PREPARE_QUERY" == "1" ]]; then echo "query:       $(query_bin)"; fi
if [[ "$PREPARE_GROUNDTRUTH" == "1" ]]; then echo "groundtruth: $(groundtruth_bin)"; fi

# Real, non-overlapping SIFT vectors for throughput queries and inserts.
# PREPARE_INSERT remains an alias for backward compatibility.
PREPARE_BENCHMARK_DATA="${PREPARE_BENCHMARK_DATA:-${PREPARE_INSERT:-1}}"
if [[ "$PREPARE_BENCHMARK_DATA" == "1" ]]; then
  benchmark_args=(
    --source "$BENCHMARK_VECTOR_SOURCE"
    --query-output "$(performance_query_bin)"
    --query-start "$PERFORMANCE_QUERY_START"
    --query-end "$PERFORMANCE_QUERY_END"
    --insert-output "$(insert_bin)"
    --insert-start "$INSERT_VECTOR_START"
    --insert-end "$INSERT_VECTOR_END"
    --chunk-rows "${BENCHMARK_CONVERT_CHUNK_ROWS:-1000000}")
  if [[ "${OVERWRITE_BENCHMARK_DATA:-0}" == "1" ]]; then
    benchmark_args+=(--overwrite)
  fi
  python3 "$SCRIPT_DIR/prepare_sift_benchmark_data.py" "${benchmark_args[@]}"
  echo "performance: $(performance_query_bin) [$PERFORMANCE_QUERY_START,$PERFORMANCE_QUERY_END)"
  echo "insert:      $(insert_bin) [$INSERT_VECTOR_START,$INSERT_VECTOR_END)"
fi
