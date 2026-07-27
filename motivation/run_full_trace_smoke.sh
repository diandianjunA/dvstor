#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
DEPTH="${DEPTH:-32}"
run_dir="$MOTIVATION_DIR/results/full_trace/depth_${DEPTH}"
mkdir -p "$run_dir"
trace_path="$run_dir/rdma_trace.jsonl"

(
  set -a
  source "$MOTIVATION_DIR/configs/common.env"
  source "$MOTIVATION_DIR/configs/prefetch_${DEPTH}.env"
  GPU_QUERY_EXPANSION_POLICY=fixed
  GPU_QUERY_BEAM_MERGE_POLICY=stable-run
  BENCHMARK_CLIENT_THREADS=1
  REPORT_DIR="$run_dir"
  QUERY_RDMA_TRACE_MODE=full
  QUERY_RDMA_TRACE_SAMPLE_RATE=1
  QUERY_RDMA_TRACE_OUTPUT="$trace_path"
  QUERY_RDMA_TRACE_EVENTS_PER_QUERY=8192
  WARMUP_SECONDS=0
  MEASURE_SECONDS=0
  RECALL_QUERIES="${FULL_TRACE_QUERIES:-100}"
  set +a
  "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
)
"$MOTIVATION_DIR/analyze_rdma_trace.py" \
  --include-round-details "$trace_path"
