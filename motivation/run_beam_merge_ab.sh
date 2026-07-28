#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
CONCURRENCIES="${CONCURRENCIES:-1 8 64 256}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-16}"
RESULT_ROOT="${BEAM_MERGE_RESULT_ROOT:-$MOTIVATION_DIR/results/beam_merge}"

run_one() {
  local policy="$1"
  local concurrency="$2"
  local run_dir="$RESULT_ROOT/${policy}/concurrency_${concurrency}"
  mkdir -p "$run_dir"
  (
    set -a
    source "$MOTIVATION_DIR/configs/common.env"
    source "$MOTIVATION_DIR/configs/beam_merge_${policy//-/_}.env"
    GPU_GRAPH_PREFETCH_DEPTH="$PREFETCH_DEPTH"
    BENCHMARK_CLIENT_THREADS="$concurrency"
    REPORT_DIR="$run_dir"
    QUERY_RDMA_TRACE_MODE=off
    set +a
    "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
  )
}

for policy in legacy stable-run; do
  for concurrency in $CONCURRENCIES; do
    run_one "$policy" "$concurrency"
  done
done

echo "Beam-merge A/B reports: $RESULT_ROOT"
