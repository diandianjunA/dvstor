#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

REPORT_DIR="$MOTIVATION_REPORT_DIR" \
LOG_DIR="$MOTIVATION_LOG_DIR" \
PID_DIR="$MOTIVATION_PID_DIR" \
  "$SIFT_DIR/stop_memory_nodes.sh"

