#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift1b_common.sh"

python3 "$SCRIPT_DIR/convert_sift1b.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$CONVERTED_DIR" \
  --groundtruth-label "$GROUNDTRUTH_LABEL" \
  --max-base "$MAX_VECTORS" \
  --max-query "$MAX_QUERIES" \
  --topk "$GROUNDTRUTH_TOPK" \
  --chunk-rows "${CONVERT_CHUNK_ROWS:-1000000}"

echo "base:        $(base_bin)"
echo "query:       $(query_bin)"
echo "groundtruth: $(groundtruth_bin)"
