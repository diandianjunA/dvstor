#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPANSION_VALUES="${EXPANSION_VALUES:-1 2 4 8}"

# Use one query coroutine to isolate intra-query batching. This experiment
# shows how many small graph reads/kernels the naive traversal creates and how
# much batching opportunity exists inside one query.
for expansion in $EXPANSION_VALUES; do
  MOTIVATION_COROUTINES=1 MOTIVATION_QUERY_COROUTINES=1 \
  MOTIVATION_CLIENT_THREADS=1 MOTIVATION_EXPANSION_BATCH="$expansion" \
    "$SCRIPT_DIR/run_case.sh" expansion-sweep "k${expansion}" naive
done
