#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 2 4 8 16}"

# Keep host staging and single-node expansion enabled. Increasing the number
# of resident query coroutines exposes cross-query overlap opportunity without
# changing the search algorithm or adding GPUDirect.
for concurrency in $CONCURRENCY_VALUES; do
  MOTIVATION_COROUTINES="$concurrency" \
  MOTIVATION_QUERY_COROUTINES="$concurrency" \
  MOTIVATION_CLIENT_THREADS="$concurrency" \
  MOTIVATION_EXPANSION_BATCH=1 \
    "$SCRIPT_DIR/run_case.sh" concurrency-sweep "c${concurrency}" naive
done
