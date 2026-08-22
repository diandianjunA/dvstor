#!/usr/bin/env bash
set -euo pipefail

PROGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROGRAM_DIR/../.." && pwd)"

has_text() {
  local pattern="$1"
  shift
  rg -q -- "$pattern" "$@" 2>/dev/null
}

printf '%-38s %s\n' capability status
printf '%-38s %s\n' 'M1.1 coupled synchronous global' supported
printf '%-38s %s\n' 'M1.1 decoupled Stage1 ACK' supported

if has_text 'local-only' "$PROJECT_DIR/src/common/configuration.hh" \
    "$PROJECT_DIR/experiment/start_memory_node.sh"; then
  printf '%-38s %s\n' 'M1.2 explicit local-only mode' supported
else
  printf '%-38s %s\n' 'M1.2 explicit local-only mode' MISSING
fi

if has_text 'storage-owner-locality-repair-mode' \
    "$PROJECT_DIR/src/common/configuration.hh" \
    "$PROJECT_DIR/experiment/start_memory_node.sh"; then
  printf '%-38s %s\n' 'M1.3 relocation-only switch' supported
else
  printf '%-38s %s\n' 'M1.3 relocation-only switch' MISSING
fi

if has_text 'dynamic-reachability-report' "$PROJECT_DIR/src" "$PROJECT_DIR/tools"; then
  printf '%-38s %s\n' 'dynamic reachability scanner' supported
else
  printf '%-38s %s\n' 'dynamic reachability scanner' MISSING
fi

if has_text 'storage-owner-locality-snapshot' "$PROJECT_DIR/src" "$PROJECT_DIR/tools"; then
  printf '%-38s %s\n' 'full-graph locality snapshot' supported
else
  printf '%-38s %s\n' 'full-graph locality snapshot' MISSING
fi

