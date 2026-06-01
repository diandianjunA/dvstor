#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat >&2 <<EOF
usage: $0 <profile> [start|stop|restart|status] [extra memory-node args...]

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
COMMAND="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

PROFILE_FILE="$SCRIPT_DIR/profile_${PROFILE}.env"
if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "error: unknown profile: $PROFILE" >&2
  usage
  exit 1
fi

set -a
source "$PROFILE_FILE"
set +a

exec "$SCRIPT_DIR/start_all_memory_nodes.sh" "$COMMAND" "$@"

