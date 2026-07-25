#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
TRACE_CONCURRENCIES="${TRACE_CONCURRENCIES:-1 64}"
DEPTHS="${DEPTHS:-1 2 4 8 16 32}"
TRACE_MODE="${TRACE_MODE:-sampled}"
TRACE_SAMPLE_RATE="${TRACE_SAMPLE_RATE:-100}"

for depth in $DEPTHS; do
  for concurrency in $TRACE_CONCURRENCIES; do
    run_dir="$MOTIVATION_DIR/results/trace/depth_${depth}/concurrency_${concurrency}"
    mkdir -p "$run_dir"
    trace_path="$run_dir/rdma_trace.jsonl"
    (
      set -a
      source "$MOTIVATION_DIR/configs/common.env"
      source "$MOTIVATION_DIR/configs/prefetch_${depth}.env"
      BENCHMARK_CLIENT_THREADS="$concurrency"
      REPORT_DIR="$run_dir"
      QUERY_RDMA_TRACE_MODE="$TRACE_MODE"
      QUERY_RDMA_TRACE_SAMPLE_RATE="$TRACE_SAMPLE_RATE"
      QUERY_RDMA_TRACE_OUTPUT="$trace_path"
      # Detailed tracing is an analysis run, not a throughput result.
      WARMUP_SECONDS="${TRACE_WARMUP_SECONDS:-5}"
      MEASURE_SECONDS="${TRACE_MEASURE_SECONDS:-20}"
      RECALL_QUERIES="${TRACE_RECALL_QUERIES:-100}"
      set +a
      "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
    )
    "$MOTIVATION_DIR/analyze_rdma_trace.py" "$trace_path"
  done
done
