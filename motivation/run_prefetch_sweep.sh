#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
CONCURRENCIES="${CONCURRENCIES:-1 8 64 256}"
DEPTHS="${DEPTHS:-1 2 4 8 16 32}"

for depth in $DEPTHS; do
  for concurrency in $CONCURRENCIES; do
    run_dir="$MOTIVATION_DIR/results/sweep/depth_${depth}/concurrency_${concurrency}"
    mkdir -p "$run_dir"
    (
      set -a
      source "$MOTIVATION_DIR/configs/common.env"
      source "$MOTIVATION_DIR/configs/prefetch_${depth}.env"
      BENCHMARK_CLIENT_THREADS="$concurrency"
      REPORT_DIR="$run_dir"
      QUERY_RDMA_TRACE_MODE=off
      set +a
      "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
    )
  done
done
"$MOTIVATION_DIR/summarize_prefetch_sweep.py" \
  "$MOTIVATION_DIR/results/sweep"
