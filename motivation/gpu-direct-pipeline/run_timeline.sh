#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"
load_case_config naive

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required for the optional CUDA/CPU timeline experiment" >&2
  exit 1
fi

ensure_built dvstor_breakdown_benchmark
PREPARE_BASE="${PREPARE_BASE:-0}" "$SIFT_DIR/prepare_sift100m_data.sh"
validate_index_metadata

GPUDIRECT_RDMA=0
COROUTINES=1
QUERY_COROUTINES=1
CLIENT_THREADS=1
EXPANSION_BATCH=1
QUERY_BATCH_SIZE=1
USE_RABITQ=0
WARMUP_SECONDS="${TIMELINE_WARMUP_SECONDS:-2}"
MEASURE_SECONDS="${TIMELINE_MEASURE_SECONDS:-5}"
PROFILE_NAME="motivation_timeline_naive"

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$MOTIVATION_REPORT_DIR/timeline/naive_${TS}"
mkdir -p "$CASE_DIR"
write_service_config "$CASE_DIR/service.ini"
write_case_manifest "$CASE_DIR/case.env" timeline naive

cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$CASE_DIR/service.ini"
  --workload query
  --warmup-seconds "$WARMUP_SECONDS"
  --measure-seconds "$MEASURE_SECONDS"
  --client-threads 1
  --query-file "$(query_bin)"
  --groundtruth-file "$(groundtruth_bin)"
  --recall-queries "${TIMELINE_RECALL_QUERIES:-10}"
  --recall-k "$K"
  --min-recall -1
  --report-json "$CASE_DIR/report.json"
  --report-text "$CASE_DIR/report.txt")

nsys profile --force-overwrite=true --trace=cuda,osrt,nvtx \
  --output "$CASE_DIR/timeline" "${cmd[@]}"
echo "timeline: $CASE_DIR/timeline.nsys-rep"
