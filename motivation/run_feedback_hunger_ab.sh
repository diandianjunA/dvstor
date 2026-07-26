#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
CONCURRENCIES="${CONCURRENCIES:-1 8 64 256}"
FIXED_DEPTHS="${FIXED_DEPTHS:-1 8 16 32}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"

run_one() {
  local policy="$1"
  local depth="$2"
  local concurrency="$3"
  local run_dir="$MOTIVATION_DIR/results/feedback_hunger/${policy}/concurrency_${concurrency}"
  mkdir -p "$run_dir"
  (
    set -a
    source "$MOTIVATION_DIR/configs/common.env"
    if [[ "$policy" == feedback-hunger || "$policy" == feedback-horizon-hunger ]]; then
      source "$MOTIVATION_DIR/configs/feedback_hunger.env"
    else
      source "$MOTIVATION_DIR/configs/prefetch_${depth}.env"
      GPU_QUERY_EXPANSION_POLICY=fixed
    fi
    BENCHMARK_CLIENT_THREADS="$concurrency"
    REPORT_DIR="$run_dir"
    QUERY_RDMA_TRACE_MODE=off
    set +a
    "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
  )
}

for depth in $FIXED_DEPTHS; do
  for concurrency in $CONCURRENCIES; do
    run_one "fixed_depth_${depth}" "$depth" "$concurrency"
  done
done

if [[ "$RUN_DYNAMIC" == 1 ]]; then
  for concurrency in $CONCURRENCIES; do
      run_one "feedback-horizon-hunger" 16 "$concurrency"
  done
fi

echo "Feedback-hunger A/B reports: $MOTIVATION_DIR/results/feedback_hunger"
