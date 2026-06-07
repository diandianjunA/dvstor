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
