#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/start_memory_node.sh" 3 "${1:-${PROFILE:-baseline}}"
