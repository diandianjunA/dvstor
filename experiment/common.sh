#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$EXPERIMENT_DIR/.." && pwd)"
EVALUATION_DIR="$PROJECT_DIR/evaluation/sift100m"

REPORT_DIR="${REPORT_DIR:-$EXPERIMENT_DIR/reports}"
LOG_DIR="${LOG_DIR:-$EXPERIMENT_DIR/logs}"
PID_DIR="${PID_DIR:-$EXPERIMENT_DIR/pids}"

source "$EVALUATION_DIR/sift100m_common.sh"

load_experiment_profile() {
  local profile="${1:?profile name is required}"
  local profile_env="$EXPERIMENT_DIR/profiles/${profile}.env"
  if [[ ! -f "$profile_env" ]]; then
    echo "unknown experiment profile: $profile" >&2
    echo "available profiles:" >&2
    find "$EXPERIMENT_DIR/profiles" -maxdepth 1 -name '*.env' -printf '  %f\n' \
      | sed 's/\.env$//' >&2
    return 1
  fi
  PROFILE="$profile"
  source "$profile_env"
}

