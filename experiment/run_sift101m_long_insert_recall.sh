#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-03_rabitq_expension_aware_two_stage_aldi_rdma}}"
load_experiment_profile "$PROFILE"

ensure_built dvstor_sift101m_long_insert_recall

# This is a graph-quality validation run, not a fine-grained performance breakdown.
# Keeping per-request samples disabled avoids storing 1M insert samples on the CN.
ENABLE_BREAKDOWN="${LONG_INSERT_ENABLE_BREAKDOWN:-0}"

SIFT_ROOT="${SIFT_ROOT:-$DATASET_DIR}"
INSERT_FILE_101M="${INSERT_FILE_101M:-$SIFT_ROOT/sift100m_insert_test.bvecs}"
QUERY_FILE_101M="${QUERY_FILE_101M:-$SIFT_ROOT/sift101m_query.fbin}"
GROUNDTRUTH_FILE_101M="${GROUNDTRUTH_FILE_101M:-$SIFT_ROOT/sift101m_groundtruth.bin}"
BASELINE_GROUNDTRUTH_FILE="${BASELINE_GROUNDTRUTH_FILE:-$SIFT_ROOT/gnd/idx_100M.ivecs}"

INSERT_START_ID="${INSERT_START_ID:-100000000}"
INSERT_COUNT="${INSERT_COUNT:-1000000}"
INSERT_ROW_OFFSET="${INSERT_ROW_OFFSET:-0}"
INSERT_THREADS="${INSERT_THREADS:-$CLIENT_THREADS}"
INSERT_BATCH_SIZE="${INSERT_BATCH_SIZE:-16}"
RECALL_QUERIES="${RECALL_QUERIES:-10000}"
RECALL_K="${RECALL_K:-$K}"
SETTLE_SECONDS="${SETTLE_SECONDS:-300}"
RESET_BREAKDOWN_EVERY="${RESET_BREAKDOWN_EVERY:-50000}"
MIN_POST_RECALL="${MIN_POST_RECALL:--1}"
MAX_RECALL_DROP="${MAX_RECALL_DROP:--1}"

for path in "$INSERT_FILE_101M" "$QUERY_FILE_101M" "$GROUNDTRUTH_FILE_101M"; do
  if [[ ! -s "$path" ]]; then
    echo "missing required input: $path" >&2
    exit 1
  fi
done
if [[ -n "$BASELINE_GROUNDTRUTH_FILE" && ! -s "$BASELINE_GROUNDTRUTH_FILE" ]]; then
  echo "missing baseline groundtruth: $BASELINE_GROUNDTRUTH_FILE" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPORT_DIR/$PROFILE"
mkdir -p "$OUT_DIR"
JSON_REPORT="$OUT_DIR/sift101m_long_insert_${PROFILE}_${TS}.json"
TEXT_REPORT="$OUT_DIR/sift101m_long_insert_${PROFILE}_${TS}.txt"
RUNTIME_CONFIG="$OUT_DIR/service_sift101m_long_insert_${PROFILE}_${TS}.ini"
write_service_config "$RUNTIME_CONFIG"

cmd=("$BUILD_DIR/dvstor_sift101m_long_insert_recall"
  --service-config "$RUNTIME_CONFIG"
  --insert-file "$INSERT_FILE_101M"
  --insert-start-id "$INSERT_START_ID"
  --insert-count "$INSERT_COUNT"
  --insert-row-offset "$INSERT_ROW_OFFSET"
  --insert-threads "$INSERT_THREADS"
  --insert-batch-size "$INSERT_BATCH_SIZE"
  --query-file "$QUERY_FILE_101M"
  --groundtruth-file "$GROUNDTRUTH_FILE_101M"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$RECALL_K"
  --settle-seconds "$SETTLE_SECONDS"
  --reset-breakdown-every "$RESET_BREAKDOWN_EVERY"
  --min-post-recall "$MIN_POST_RECALL"
  --max-recall-drop "$MAX_RECALL_DROP"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

if [[ -n "$BASELINE_GROUNDTRUTH_FILE" ]]; then
  cmd+=(--baseline-groundtruth-file "$BASELINE_GROUNDTRUTH_FILE")
fi

printf '[sift101m-long-insert] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
