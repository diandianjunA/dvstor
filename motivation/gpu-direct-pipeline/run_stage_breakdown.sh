#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Characterize the unoptimized dependency chain with one outstanding query.
MOTIVATION_COROUTINES=1 MOTIVATION_QUERY_COROUTINES=1 \
MOTIVATION_CLIENT_THREADS=1 MOTIVATION_EXPANSION_BATCH=1 \
  "$SCRIPT_DIR/run_case.sh" stage-breakdown naive-single-query naive
