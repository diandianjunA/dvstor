#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

ensure_built dvstor_breakdown_benchmark
PREPARE_BASE="${PREPARE_BASE:-0}" "$EXPERIMENT_DIR/prepare_sift100m_data.sh"

RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_CLIENT_THREADS="${RECALL_CLIENT_THREADS:-128}"
RECALL_K="${RECALL_K:-$K}"
MIN_RECALL="${MIN_RECALL:--1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPORT_DIR/recall_$PROFILE"
mkdir -p "$OUT_DIR"
JSON_REPORT="$OUT_DIR/sift100m_recall_${PROFILE}_${TS}.json"
TEXT_REPORT="$OUT_DIR/sift100m_recall_${PROFILE}_${TS}.txt"
RUNTIME_CONFIG="$OUT_DIR/service_${PROFILE}_${TS}.ini"
write_service_config "$RUNTIME_CONFIG"

cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$RUNTIME_CONFIG"
  --workload query
  --warmup-ops 0
  --measure-ops 1
  --client-threads "$RECALL_CLIENT_THREADS"
  --query-file "$(query_bin)"
  --groundtruth-file "$(groundtruth_bin)"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$RECALL_K"
  --min-recall "$MIN_RECALL"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

printf '[recall] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
