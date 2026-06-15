#!/usr/bin/env bash
set -euo pipefail

MOTIVATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/../.." && pwd)"
SIFT_DIR="$PROJECT_DIR/evaluation/sift100m"

# Set motivation-specific service defaults before loading the SIFT harness so
# its ${VAR:-default} assignments preserve these values.
source "$MOTIVATION_DIR/configs/defaults.env"

# Reuse data conversion, index naming, cluster endpoints, and config emission
# from the maintained SIFT100M harness.
source "$SIFT_DIR/sift100m_common.sh"

MOTIVATION_REPORT_DIR="${MOTIVATION_REPORT_DIR:-$MOTIVATION_DIR/reports}"
MOTIVATION_LOG_DIR="${MOTIVATION_LOG_DIR:-$MOTIVATION_DIR/logs}"
MOTIVATION_PID_DIR="${MOTIVATION_PID_DIR:-$MOTIVATION_DIR/pids}"
mkdir -p "$MOTIVATION_REPORT_DIR" "$MOTIVATION_LOG_DIR" "$MOTIVATION_PID_DIR"

load_case_config() {
  local config_name="$1"
  local config_path="$MOTIVATION_DIR/configs/${config_name}.env"
  if [[ ! -f "$config_path" ]]; then
    echo "unknown motivation config: $config_name" >&2
    return 1
  fi
  source "$config_path"
}

motivation_profile_for_memory_nodes() {
  # GPUDirect is a compute-node property. The baseline memory-node profile
  # loads the same exact index and is valid for every experiment in this suite.
  echo "baseline"
}

write_case_manifest() {
  local output="$1"
  local experiment="$2"
  local label="$3"
  cat > "$output" <<EOF
experiment=$experiment
label=$label
index_prefix=$INDEX_PREFIX
gpudirect_rdma=$GPUDIRECT_RDMA
service_threads=$SERVICE_THREADS
query_workers=$QUERY_WORKERS
coroutines=$COROUTINES
query_coroutines=$QUERY_COROUTINES
client_threads=$CLIENT_THREADS
expansion_batch=$EXPANSION_BATCH
query_batch_size=$QUERY_BATCH_SIZE
use_rabitq=$USE_RABITQ
warmup_seconds=$WARMUP_SECONDS
measure_seconds=$MEASURE_SECONDS
EOF
}

start_device_sampling() {
  local output="$1"
  DEVICE_SAMPLER_PID=""
  if [[ "$ENABLE_DEVICE_SAMPLING" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[motivation] nvidia-smi unavailable; device sampling skipped" >&2
    return
  fi
  nvidia-smi dmon -i "$GPU_DEVICE" -s pucvmet -d 1 > "$output" 2>&1 &
  DEVICE_SAMPLER_PID=$!
}

stop_device_sampling() {
  if [[ -n "${DEVICE_SAMPLER_PID:-}" ]] && kill -0 "$DEVICE_SAMPLER_PID" 2>/dev/null; then
    kill "$DEVICE_SAMPLER_PID" 2>/dev/null || true
    wait "$DEVICE_SAMPLER_PID" 2>/dev/null || true
  fi
  DEVICE_SAMPLER_PID=""
}
