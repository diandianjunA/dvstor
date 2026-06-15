#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Controlled data-path experiment: all scheduling and graph-search parameters
# are identical; only the RDMA destination changes.
MOTIVATION_COROUTINES=1 MOTIVATION_QUERY_COROUTINES=1 \
MOTIVATION_CLIENT_THREADS=1 MOTIVATION_EXPANSION_BATCH=1 \
  "$SCRIPT_DIR/run_case.sh" data-path host-staging naive

MOTIVATION_COROUTINES=1 MOTIVATION_QUERY_COROUTINES=1 \
MOTIVATION_CLIENT_THREADS=1 MOTIVATION_EXPANSION_BATCH=1 \
  "$SCRIPT_DIR/run_case.sh" data-path gpudirect gpudirect
