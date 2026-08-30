#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/deep100m_common.sh"
python3 "$SCRIPT_DIR/prepare_deep100m_data.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$CONVERTED_DIR" \
  --recall-rows "${DEEP_RECALL_ROWS:-3334}" \
  --performance-rows "${DEEP_PERFORMANCE_ROWS:-3333}"
