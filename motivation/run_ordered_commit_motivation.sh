#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$MOTIVATION_DIR/results/batch_barrier}"
TASK_GRANULARITY="${TASK_GRANULARITY:-tile}"
TASK_OVERHEADS_US="${TASK_OVERHEADS_US:-0,1,2,5,10}"

# Reuse the integrity-checked trace collection. QUICK=1 is useful only for
# validating the pipeline; formal evidence should retain the default repeated
# matrix and trace-off performance controls.
if [[ "${ANALYZE_ONLY:-0}" != "1" ]]; then
  "$MOTIVATION_DIR/run_batch_barrier_motivation.sh"
fi

mapfile -t traces < <(
  find "$RESULT_ROOT/trace" -type f -name 'rdma_trace.jsonl' | sort
)
if (( ${#traces[@]} == 0 )); then
  echo "no trace files found below $RESULT_ROOT/trace" >&2
  exit 2
fi

for trace in "${traces[@]}"; do
  echo "[ordered-commit-oracle] $trace"
  "$MOTIVATION_DIR/analyze_ordered_commit_oracle.py" \
    --task-granularity "$TASK_GRANULARITY" \
    --task-overheads-us "$TASK_OVERHEADS_US" \
    "$trace"
done

"$MOTIVATION_DIR/summarize_ordered_commit_motivation.py" "$RESULT_ROOT"
echo "Per-run oracle reports were written next to each rdma_trace.jsonl."
