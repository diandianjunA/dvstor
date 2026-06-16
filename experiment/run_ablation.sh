#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

profiles=("$@")
if (( ${#profiles[@]} == 0 )); then
  profiles=(
    00_baseline
    01_rabitq_gpu_pipeline
    02_rabitq_gpu_pipeline_aldi
    03_rabitq_gpu_pipeline_aldi_rdma
  )
fi

cleanup() {
  "$SCRIPT_DIR/stop_memory_nodes.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for profile in "${profiles[@]}"; do
  cleanup
  echo "=== profile: $profile ==="
  "$SCRIPT_DIR/start_all_memory_nodes.sh" "$profile"
  sleep "${STARTUP_SLEEP_SECONDS:-5}"
  "$SCRIPT_DIR/run_breakdown.sh" "$profile"
done

cleanup
python3 "$SCRIPT_DIR/summarize_reports.py" "${profiles[@]}"
