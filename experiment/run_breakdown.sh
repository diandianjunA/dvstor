#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

ensure_built dvstor_breakdown_benchmark
PREPARE_BASE="${PREPARE_BASE:-0}" "$EXPERIMENT_DIR/prepare_sift100m_data.sh"

WORKLOAD="${WORKLOAD:-mixed}"
BENCHMARK_CLIENT_THREADS="${BENCHMARK_CLIENT_THREADS:-128}"
READ_RATIO="${READ_RATIO:-0.5}"
MIXED_MODE="${MIXED_MODE:-fixed_threads}"
# MIXED_MODE="${MIXED_MODE:-probability}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_K="${RECALL_K:-$K}"
MIN_RECALL="${MIN_RECALL:--1}"
RECALL_QUERY_FILE="$(query_bin)"
PERFORMANCE_QUERY_PATH="$(performance_query_bin)"
INSERT_PATH="$(insert_bin)"
if [[ ! -s "$PERFORMANCE_QUERY_PATH" ]]; then
  echo "missing performance query file: $PERFORMANCE_QUERY_PATH" >&2
  echo "set PERFORMANCE_QUERY_FILE to a large held-out .u8bin query set" >&2
  exit 1
fi
if [[ ! -s "$INSERT_PATH" ]]; then
  echo "missing insert file: $INSERT_PATH" >&2
  echo "set INSERT_FILE to a held-out .u8bin insert set" >&2
  exit 1
fi
if [[ "$(readlink -f "$RECALL_QUERY_FILE")" == "$(readlink -f "$PERFORMANCE_QUERY_PATH")" ]]; then
  echo "recall and performance query files must be different" >&2
  exit 1
fi
if [[ "$(readlink -f "$PERFORMANCE_QUERY_PATH")" == "$(readlink -f "$INSERT_PATH")" ]]; then
  echo "performance query and insert files must be different" >&2
  exit 1
fi
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPORT_DIR/$PROFILE"
mkdir -p "$OUT_DIR"
JSON_REPORT="$OUT_DIR/sift100m_${PROFILE}_${TS}.json"
TEXT_REPORT="$OUT_DIR/sift100m_${PROFILE}_${TS}.txt"
RUNTIME_CONFIG="$OUT_DIR/service_${PROFILE}_${TS}.ini"
write_service_config "$RUNTIME_CONFIG"

cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$RUNTIME_CONFIG"
  --workload "$WORKLOAD"
  --warmup-seconds "$WARMUP_SECONDS"
  --measure-seconds "$MEASURE_SECONDS"
  --client-threads "$BENCHMARK_CLIENT_THREADS"
  --read-ratio "$READ_RATIO"
  --mixed-mode "$MIXED_MODE"
  --write-insert-ratio "${WRITE_INSERT_RATIO:-0.5}"
  --write-upsert-ratio "${WRITE_UPSERT_RATIO:-0.4}"
  --write-delete-ratio "${WRITE_DELETE_RATIO:-0.1}"
  --recall-query-file "$RECALL_QUERY_FILE"
  --performance-query-file "$PERFORMANCE_QUERY_PATH"
  --groundtruth-file "$(groundtruth_bin)"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$RECALL_K"
  --min-recall "$MIN_RECALL"
  --insert-start-id "${INSERT_START_ID:-$((MAX_VECTORS + 1000000))}"
  --insert-file "$INSERT_PATH"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

printf '[breakdown] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
