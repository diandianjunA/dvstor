#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/spacev100m_common.sh"
python3 "$SCRIPT_DIR/prepare_spacev100m_data.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$CONVERTED_DIR" \
  --recall-rows "${SPACEV_RECALL_ROWS:-10000}" \
  --performance-rows "${SPACEV_PERFORMANCE_ROWS:-10000}"
