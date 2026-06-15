#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

profile="$(motivation_profile_for_memory_nodes)"
echo "[motivation] starting SIFT100M memory nodes with profile=$profile"
REPORT_DIR="$MOTIVATION_REPORT_DIR" \
LOG_DIR="$MOTIVATION_LOG_DIR" \
PID_DIR="$MOTIVATION_PID_DIR" \
  "$SIFT_DIR/start_all_memory_nodes.sh" "$profile"

