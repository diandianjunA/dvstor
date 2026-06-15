#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  echo "usage: $0 <experiment> <label> <config-name>" >&2
  exit 1
}

[[ $# -eq 3 ]] || usage
EXPERIMENT="$1"
LABEL="$2"
CONFIG_NAME="$3"
load_case_config "$CONFIG_NAME"

if [[ "$WORKLOAD" != "query" ]]; then
  echo "this motivation suite expects WORKLOAD=query" >&2
  exit 1
fi
if [[ "$USE_RABITQ" != "0" ]]; then
  echo "motivation cases must keep USE_RABITQ=0 to isolate the data path and scheduler" >&2
  exit 1
fi
if (( QUERY_WORKERS != 1 || INSERT_WORKERS != 0 )); then
  echo "motivation cases require QUERY_WORKERS=1 and INSERT_WORKERS=0" >&2
  exit 1
fi

ensure_built dvstor_breakdown_benchmark
PREPARE_BASE="${PREPARE_BASE:-0}" "$SIFT_DIR/prepare_sift100m_data.sh"
validate_index_metadata

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$MOTIVATION_REPORT_DIR/$EXPERIMENT/${LABEL}_${TS}"
mkdir -p "$CASE_DIR"
RUNTIME_CONFIG="$CASE_DIR/service.ini"
JSON_REPORT="$CASE_DIR/report.json"
TEXT_REPORT="$CASE_DIR/report.txt"
MANIFEST="$CASE_DIR/case.env"
GPU_DMON="$CASE_DIR/gpu_dmon.txt"
RESOURCE_USAGE="$CASE_DIR/resource_usage.txt"

PROFILE_NAME="motivation_${EXPERIMENT}_${LABEL}"
write_service_config "$RUNTIME_CONFIG"
write_case_manifest "$MANIFEST" "$EXPERIMENT" "$LABEL"

cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$RUNTIME_CONFIG"
  --workload query
  --warmup-seconds "$WARMUP_SECONDS"
  --measure-seconds "$MEASURE_SECONDS"
  --client-threads "$CLIENT_THREADS"
  --query-file "$(query_bin)"
  --groundtruth-file "$(groundtruth_bin)"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$K"
  --min-recall "$MIN_RECALL"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")

printf '[motivation] command:'; printf ' %q' "${cmd[@]}"; echo
start_device_sampling "$GPU_DMON"
trap stop_device_sampling EXIT
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v -o "$RESOURCE_USAGE" "${cmd[@]}"
else
  "${cmd[@]}"
fi
stop_device_sampling
trap - EXIT

python3 - "$JSON_REPORT" "$GPUDIRECT_RDMA" <<'PY_VERIFY_PATH'
import json
import sys

report_path, requested_gdr = sys.argv[1:]
with open(report_path, "r", encoding="utf-8") as source:
    report = json.load(source)
counters = report.get("query_breakdown", {}).get("counters", {})
direct = int(counters.get("query_rdma_to_staging_bytes", 0))
host = int(counters.get("query_host_staging_fallback_bytes", 0))
if requested_gdr == "1" and direct <= 0:
    raise SystemExit(
        "GPUDirect was requested but no candidate bytes reached GPU memory directly; "
        "check GPU/NIC topology, nvidia-peermem, and MR registration logs"
    )
if requested_gdr == "0" and host <= 0:
    raise SystemExit(
        "host staging was requested but no fallback bytes were recorded; "
        "the case does not match the intended baseline"
    )
print(f"[motivation] candidate path bytes: direct={direct} host={host}")
PY_VERIFY_PATH

echo "case: $CASE_DIR"
