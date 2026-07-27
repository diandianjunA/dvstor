#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
CONCURRENCIES="${CONCURRENCIES:-1 8 64 256}"
REPETITIONS="${REPETITIONS:-3}"
RESULT_ROOT="${LIVE_EXTENT_RESULT_ROOT:-$MOTIVATION_DIR/results/live_extent_ab}"
BEFORE_CASE_HOOK="${LIVE_EXTENT_BEFORE_CASE_HOOK:-}"

if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer: $REPETITIONS" >&2
  exit 1
fi
if [[ -n "$BEFORE_CASE_HOOK" && ! -x "$BEFORE_CASE_HOOK" ]]; then
  echo "LIVE_EXTENT_BEFORE_CASE_HOOK must name an executable: $BEFORE_CASE_HOOK" >&2
  exit 1
fi

# Resolve the exact profile prefix before consuming a storage-node session.
# The live case is intentionally fail-closed, so an A/B sweep must not run a
# valid fixed case first and only then discover that its paired sidecar is
# absent.
source "$PROJECT_DIR/experiment/common.sh"
load_experiment_profile "$PROFILE"
if [[ ! -s "${INDEX_PREFIX}.gextent8" ]]; then
  echo "missing Live-Extent sidecar: ${INDEX_PREFIX}.gextent8" >&2
  echo "generate it where all .dat shards are available before starting the A/B sweep" >&2
  exit 1
fi

run_one() {
  local policy="$1"
  local concurrency="$2"
  local repetition="$3"
  local config_file
  local run_dir

  case "$policy" in
    fixed)
      config_file="$MOTIVATION_DIR/configs/live_extent_fixed.env"
      ;;
    live-extent)
      config_file="$MOTIVATION_DIR/configs/live_extent_enabled.env"
      ;;
    *)
      echo "unsupported Live-Extent A/B policy: $policy" >&2
      exit 1
      ;;
  esac

  run_dir="$RESULT_ROOT/$policy/concurrency_${concurrency}/repeat_${repetition}"
  mkdir -p "$run_dir"
  if [[ -n "$BEFORE_CASE_HOOK" ]]; then
    "$BEFORE_CASE_HOOK" "$policy" "$concurrency" "$repetition"
  fi
  (
    set -a
    source "$MOTIVATION_DIR/configs/common.env"
    source "$config_file"
    BENCHMARK_CLIENT_THREADS="$concurrency"
    REPORT_DIR="$run_dir"
    set +a
    "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
  )
}

# Alternate policy order across repetitions so monotonic thermal/network drift
# is not consistently charged to the same policy.
for ((repetition = 1; repetition <= REPETITIONS; ++repetition)); do
  if (( repetition % 2 == 1 )); then
    policies=(fixed live-extent)
  else
    policies=(live-extent fixed)
  fi
  for concurrency in $CONCURRENCIES; do
    if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
      echo "CONCURRENCIES must contain positive integers: $concurrency" >&2
      exit 1
    fi
    for policy in "${policies[@]}"; do
      run_one "$policy" "$concurrency" "$repetition"
    done
  done
done

python3 "$MOTIVATION_DIR/summarize_live_extent_ab.py" "$RESULT_ROOT"
echo "Live-Extent A/B reports: $RESULT_ROOT"
