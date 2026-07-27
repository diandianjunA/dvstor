#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
RESULT_ROOT="${RESULT_ROOT:-$MOTIVATION_DIR/results/adjacency_certificate}"
CONCURRENCIES="${CONCURRENCIES:-1 64 256}"
REPETITIONS="${REPETITIONS:-1}"
WARMUP_SECONDS="${WARMUP_SECONDS:-3}"
MEASURE_SECONDS="${MEASURE_SECONDS:-10}"
BASELINE_JSON="${BASELINE_JSON:-$MOTIVATION_DIR/results/beam_merge_final/time/stable-run/concurrency_256/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260726_225928.json}"

sample_rate_for_concurrency() {
  case "$1" in
    1) printf '%s\n' "${TRACE_SAMPLE_RATE_C1:-50}" ;;
    64) printf '%s\n' "${TRACE_SAMPLE_RATE_C64:-1000}" ;;
    *) printf '%s\n' "${TRACE_SAMPLE_RATE_C256:-5000}" ;;
  esac
}

for concurrency in $CONCURRENCIES; do
  for repetition in $(seq 1 "$REPETITIONS"); do
    run_dir="$RESULT_ROOT/concurrency_${concurrency}/repeat_${repetition}"
    trace_path="$run_dir/adjacency_oracle.jsonl"
    mkdir -p "$run_dir"
    (
      set -a
      source "$MOTIVATION_DIR/configs/common.env"
      source "$MOTIVATION_DIR/configs/adjacency_certificate.env"
      BENCHMARK_CLIENT_THREADS="$concurrency"
      REPORT_DIR="$run_dir"
      QUERY_RDMA_TRACE_SAMPLE_RATE="$(
        sample_rate_for_concurrency "$concurrency")"
      QUERY_RDMA_TRACE_OUTPUT="$trace_path"
      WARMUP_SECONDS="$WARMUP_SECONDS"
      MEASURE_SECONDS="$MEASURE_SECONDS"
      set +a
      "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
    )
    "$MOTIVATION_DIR/analyze_adjacency_oracle.py" \
      "$trace_path" \
      --baseline-json "$BASELINE_JSON" \
      --output-json "$run_dir/adjacency_oracle_analysis.json" \
      --output-markdown "$run_dir/adjacency_oracle_analysis.md"
  done
done
