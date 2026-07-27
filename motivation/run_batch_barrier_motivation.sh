#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
RESULT_ROOT="${RESULT_ROOT:-$MOTIVATION_DIR/results/batch_barrier}"

# The default matrix contains a depth-1 negative control, depths around the
# measured optimum, low/high concurrency, and repetitions.  QUICK=1 is an
# end-to-end smoke, not a publishable run.
if [[ "${QUICK:-0}" == "1" ]]; then
  DEPTHS="${DEPTHS:-16}"
  CONCURRENCIES="${CONCURRENCIES:-1 256}"
  REPETITIONS="${REPETITIONS:-1}"
  PHASES="${PHASES:-performance trace}"
  PERF_WARMUP_SECONDS="${PERF_WARMUP_SECONDS:-1}"
  PERF_MEASURE_SECONDS="${PERF_MEASURE_SECONDS:-3}"
  TRACE_WARMUP_SECONDS="${TRACE_WARMUP_SECONDS:-1}"
  TRACE_MEASURE_SECONDS="${TRACE_MEASURE_SECONDS:-3}"
  RECALL_QUERIES="${RECALL_QUERIES:-32}"
else
  DEPTHS="${DEPTHS:-1 8 16 32}"
  CONCURRENCIES="${CONCURRENCIES:-1 8 64 256}"
  REPETITIONS="${REPETITIONS:-3}"
  PHASES="${PHASES:-performance trace}"
  PERF_WARMUP_SECONDS="${PERF_WARMUP_SECONDS:-10}"
  PERF_MEASURE_SECONDS="${PERF_MEASURE_SECONDS:-30}"
  TRACE_WARMUP_SECONDS="${TRACE_WARMUP_SECONDS:-5}"
  TRACE_MEASURE_SECONDS="${TRACE_MEASURE_SECONDS:-20}"
  RECALL_QUERIES="${RECALL_QUERIES:-1000}"
fi
TRACE_EVENTS_PER_QUERY="${TRACE_EVENTS_PER_QUERY:-4096}"

trace_sample_rate_for_concurrency() {
  local concurrency="$1"
  if [[ -n "${TRACE_SAMPLE_RATE:-}" ]]; then
    printf '%s\n' "$TRACE_SAMPLE_RATE"
  elif (( concurrency <= 1 )); then
    printf '5\n'
  elif (( concurrency <= 8 )); then
    printf '50\n'
  elif (( concurrency <= 64 )); then
    printf '500\n'
  else
    printf '1000\n'
  fi
}

run_benchmark() {
  local phase="$1"
  local depth="$2"
  local concurrency="$3"
  local repetition="$4"
  local run_dir="$RESULT_ROOT/$phase/depth_${depth}/concurrency_${concurrency}/repeat_${repetition}"
  local trace_path="$run_dir/rdma_trace.jsonl"
  mkdir -p "$run_dir"

  (
    set -a
    source "$MOTIVATION_DIR/configs/common.env"
    source "$MOTIVATION_DIR/configs/prefetch_${depth}.env"
    GPU_QUERY_EXPANSION_POLICY=fixed
    GPU_QUERY_BEAM_MERGE_POLICY=stable-run
    BENCHMARK_CLIENT_THREADS="$concurrency"
    REPORT_DIR="$run_dir"
    RECALL_QUERIES="$RECALL_QUERIES"
    if [[ "$phase" == "performance" ]]; then
      QUERY_RDMA_TRACE_MODE=off
      WARMUP_SECONDS="$PERF_WARMUP_SECONDS"
      MEASURE_SECONDS="$PERF_MEASURE_SECONDS"
    else
      QUERY_RDMA_TRACE_MODE=sampled
      QUERY_RDMA_TRACE_SAMPLE_RATE="$(
        trace_sample_rate_for_concurrency "$concurrency")"
      QUERY_RDMA_TRACE_EVENTS_PER_QUERY="$TRACE_EVENTS_PER_QUERY"
      QUERY_RDMA_TRACE_OUTPUT="$trace_path"
      WARMUP_SECONDS="$TRACE_WARMUP_SECONDS"
      MEASURE_SECONDS="$TRACE_MEASURE_SECONDS"
    fi
    set +a
    "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
  )

  if [[ "$phase" == "trace" ]]; then
    "$MOTIVATION_DIR/analyze_rdma_trace.py" "$trace_path"
  fi
}

for phase in $PHASES; do
  case "$phase" in
    performance|trace) ;;
    *)
      echo "unsupported PHASES entry: $phase" >&2
      exit 2
      ;;
  esac
  for depth in $DEPTHS; do
    if [[ ! -f "$MOTIVATION_DIR/configs/prefetch_${depth}.env" ]]; then
      echo "missing prefetch configuration for depth $depth" >&2
      exit 2
    fi
    for concurrency in $CONCURRENCIES; do
      for repetition in $(seq 1 "$REPETITIONS"); do
        echo "[batch-barrier] phase=$phase depth=$depth concurrency=$concurrency repetition=$repetition"
        run_benchmark "$phase" "$depth" "$concurrency" "$repetition"
      done
    done
  done
done

"$MOTIVATION_DIR/summarize_batch_barrier.py" "$RESULT_ROOT"
