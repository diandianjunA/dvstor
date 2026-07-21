#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

if [[ "${PREPARE_RECALL_DATA:-0}" == "1" ]]; then
  PREPARE_BASE=0 PREPARE_QUERY=1 PREPARE_GROUNDTRUTH=1 \
  PREPARE_BENCHMARK_DATA=0 "$EXPERIMENT_DIR/prepare_sift100m_data.sh"
fi

RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_CLIENT_THREADS="${RECALL_CLIENT_THREADS:-128}"
RECALL_K="${RECALL_K:-$K}"
RECALL_QUERY_FILE="$(query_bin)"
GROUNDTRUTH_FILE="$(groundtruth_bin)"
# Recall is query-only and does not initialize the mutation executor.
ENABLE_UPDATES=false
validate_index_metadata compute
[[ -s "$RECALL_QUERY_FILE" ]] || {
  echo "missing recall query file: $RECALL_QUERY_FILE" >&2
  exit 1
}
[[ -s "$GROUNDTRUTH_FILE" ]] || {
  echo "missing groundtruth file: $GROUNDTRUTH_FILE" >&2
  exit 1
}

ensure_built dvstor_breakdown_benchmark

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
  --recall-only
  --warmup-ops 0
  --measure-ops 0
  --client-threads "$RECALL_CLIENT_THREADS"
  --recall-query-file "$RECALL_QUERY_FILE"
  --groundtruth-file "$GROUNDTRUTH_FILE"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$RECALL_K"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

printf '[recall] profile=%s command:' "$PROFILE"; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "json: $JSON_REPORT"
echo "text: $TEXT_REPORT"
