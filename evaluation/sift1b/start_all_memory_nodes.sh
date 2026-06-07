#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${1:-${PROFILE:-baseline}}"
for node in 1 2 3 4 5; do
  "$SCRIPT_DIR/start_memory_node.sh" "$node" "$PROFILE"
done
