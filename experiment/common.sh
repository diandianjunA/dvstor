#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$EXPERIMENT_DIR/.." && pwd)"

REPORT_DIR="${REPORT_DIR:-$EXPERIMENT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$EXPERIMENT_DIR/logs}"
PID_DIR="${PID_DIR:-$EXPERIMENT_DIR/pids}"

source "$EXPERIMENT_DIR/sift100m_common.sh"

load_experiment_profile() {
  local profile="${1:?profile name is required}"
  local profile_env="$EXPERIMENT_DIR/profiles/${profile}.env"
  if [[ ! -f "$profile_env" ]]; then
    if [[ "$profile" == "04_gpu_persistent_gpunetio_baseline" ]]; then
      echo "profile '$profile' was removed: it retained the proposed persistent/two-stage architecture" >&2
      echo "and was not a valid reference baseline. Use baseline/cpu-gpu-exact-safe@f304e99" >&2
      echo "with the exact+sync contract documented in experiment/README.md." >&2
      return 1
    fi
    echo "unknown experiment profile: $profile" >&2
    echo "available profiles:" >&2
    find "$EXPERIMENT_DIR/profiles" -maxdepth 1 -name '*.env' -printf '  %f\n' \
      | sed 's/\.env$//' >&2
    return 1
  fi
  PROFILE="$profile"
  source "$profile_env"
}
