#!/bin/bash
set -euo pipefail

COMMAND="${1:-start}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

case "$COMMAND" in
  start|stop|restart|status) ;;
  *)
    echo "usage: $0 [start|stop|restart|status] [extra args...]" >&2
    exit 1
    ;;
esac

if [[ "$COMMAND" == "start" || "$COMMAND" == "restart" ]]; then
  for ((i = 0; i < MEMORY_NODES; ++i)); do
    shard="${INDEX_PREFIX}_node$((i + 1))_of${MEMORY_NODES}.dat"
    if [[ ! -f "$shard" ]]; then
      echo "error: missing shard: $shard" >&2
      echo "build the index first: $SCRIPT_DIR/build_bfs_index.sh" >&2
      exit 1
    fi
  done
fi

for ((i = 0; i < MEMORY_NODES; ++i)); do
  echo "[SIFT100M] $COMMAND memory node $i"
  "$SCRIPT_DIR/start_memory_node.sh" "$i" "$COMMAND" "$@"
done
