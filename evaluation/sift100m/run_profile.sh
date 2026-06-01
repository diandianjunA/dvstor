#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat >&2 <<EOF
usage: $0 <profile> [run_mixed_benchmark args...]

profiles:
  baseline
  gpudirect_rdma
  gpudirect_slot_clock
  gpudirect_slot_clock_storage_owner
  gpudirect_gentile_storage_owner
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PROFILE="$1"
shift
PROFILE_FILE="$SCRIPT_DIR/profile_${PROFILE}.env"

if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "error: unknown profile: $PROFILE" >&2
  usage
  exit 1
fi

set -a
source "$PROFILE_FILE"
set +a

GENERATED_CONFIG="${GENERATED_CONFIG:-$SCRIPT_DIR/generated_sift100m_${PROFILE_NAME}.ini}"
LABEL="${LABEL:-sift100m_${PROFILE_NAME}_$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/reports/$PROFILE_NAME}"
export GENERATED_CONFIG LABEL REPORT_DIR

exec "$SCRIPT_DIR/run_mixed_benchmark.sh" "$@"

