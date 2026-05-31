#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"

DIM="${DIM:-32}"
BASE_COUNT="${BASE_COUNT:-128}"
QUERY_COUNT="${QUERY_COUNT:-32}"
TOPK="${TOPK:-10}"
R="${R:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-64}"
RABITQ_BITS="${RABITQ_BITS:-4}"
MIN_RECALL="${MIN_RECALL:-0.60}"
GPU_DEVICE="${GPU_DEVICE:-1}"
MN_MEMORY="${MN_MEMORY:-2}"
CN_MEMORY="${CN_MEMORY:-2}"
PORT="${PORT:-$((23000 + ($$ % 20000)))}"
SERVER="127.0.0.1:${PORT}"

WORK_DIR="${WORK_DIR:-/tmp/dvstor_offline_layout_smoke_${PORT}_$$}"
DATA_DIR="$WORK_DIR/data"
INDEX_PREFIX="$WORK_DIR/index/offline_layout"
CONFIG_FILE="$WORK_DIR/offline_layout_storage_owner.ini"
SHARD_FILE="${INDEX_PREFIX}_node1_of1.dat"
MEMORY_STARTED=0

cleanup() {
  if [[ "$MEMORY_STARTED" == "1" ]]; then
    "$PROJECT_DIR/scripts/start_memory_node.sh" stop -p "$PORT" >/dev/null 2>&1 || true
  fi
  if [[ "${KEEP_WORK_DIR:-0}" != "1" ]]; then
    rm -rf "$WORK_DIR"
  else
    echo "[offline-layout-smoke] kept work dir: $WORK_DIR"
  fi
}
trap cleanup EXIT

echo "[offline-layout-smoke] build targets"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD_DIR" -j --target vamana_offline_builder DvstorOfflineLayoutSmokeTest dvstor_memory_node

mkdir -p "$DATA_DIR" "$(dirname "$INDEX_PREFIX")"

echo "[offline-layout-smoke] generate synthetic data: dim=$DIM base=$BASE_COUNT queries=$QUERY_COUNT"
python3 - "$DATA_DIR" "$DIM" "$BASE_COUNT" "$QUERY_COUNT" "$TOPK" <<'PY'
import math
import pathlib
import struct
import sys

out = pathlib.Path(sys.argv[1])
dim = int(sys.argv[2])
base_count = int(sys.argv[3])
query_count = int(sys.argv[4])
topk = int(sys.argv[5])

def make_vec(i):
    values = [((i * 37 + d * 17) % 100) / 1000.0 for d in range(dim)]
    values[i % dim] += 5.0
    values[(i * 7 + 3) % dim] += 1.5
    values[(i * 11 + 5) % dim] += 0.03 * (i // max(1, dim))
    return values

base = [make_vec(i) for i in range(base_count)]
queries = base[:query_count]

def write_fbin(path, rows):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(rows), dim))
        for row in rows:
            f.write(struct.pack("<" + "f" * dim, *row))

write_fbin(out / "base.fbin", base)
write_fbin(out / "query.fbin", queries)

with open(out / "groundtruth.bin", "wb") as f:
    f.write(struct.pack("<II", query_count, topk))
    for q in queries:
        distances = []
        for idx, row in enumerate(base):
            dist = sum((q[d] - row[d]) ** 2 for d in range(dim))
            distances.append((dist, idx))
        distances.sort()
        f.write(struct.pack("<" + "I" * topk, *[idx for _, idx in distances[:topk]]))
PY

echo "[offline-layout-smoke] build offline index"
"$BUILD_DIR/vamana_offline_builder" \
  --data-path "$DATA_DIR/base.fbin" \
  --output-prefix "$INDEX_PREFIX" \
  --memory-nodes 1 \
  --threads 4 \
  --R "$R" \
  --beam-width "$BEAM_WIDTH" \
  --alpha 1.2 \
  --rabitq-bits "$RABITQ_BITS" \
  --node-layout rabitq_search_block \
  --max-vectors "$BASE_COUNT" \
  --no-gpu \
  --query-path "$DATA_DIR/query.fbin" \
  --groundtruth-path "$DATA_DIR/groundtruth.bin"

python3 - "$INDEX_PREFIX.meta.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    meta = json.load(f)
layout = meta.get("node_layout")
if layout != "rabitq_search_block":
    raise SystemExit(f"unexpected node_layout={layout!r}")
print(f"[offline-layout-smoke] metadata layout={layout} node_size={meta.get('node_size')}")
PY

cat > "$CONFIG_FILE" <<EOF
servers = $SERVER
initiator = true
port = $PORT
threads = 4
coroutines = 4
dim = $DIM
k = $TOPK
R = $R
beam-width = $BEAM_WIDTH
beam-width-construction = $BEAM_WIDTH
alpha = 1.2
rabitq-bits = $RABITQ_BITS
search-mode = rabitq_gpu
insert-execution = storage_owner
storage-owner-batch-max = 8
storage-owner-batch-wait-us = 100
storage-owner-cache-mb = 64
storage-owner-peer-rdma-tokens = 4
storage-owner-rpc-depth = 2
storage-owner-rpc-timeout-ms = 30000
storage-owner-construction-beam-width = $BEAM_WIDTH
storage-owner-search-snapshot-batch = 16
storage-owner-prune-max-candidates = 32
storage-owner-reverse-mode = async
storage-owner-reverse-queue-depth = 4096
storage-owner-reverse-flush-us = 100
storage-owner-reverse-coalesce-max = 64
storage-id = 0
storage-peers = $SERVER
load-index = true
index-prefix = $INDEX_PREFIX
gpu-device = $GPU_DEVICE
max-vectors = 512
cn-memory = $CN_MEMORY
mn-memory = $MN_MEMORY
query-workers = 4
query-coroutines = 2
cache = true
cache-ratio = 5
disable-thread-pinning = true
neighbor-cache-mb = 16
gpu-rabitq-cache-mb = 0
EOF

echo "[offline-layout-smoke] start memory node on $SERVER"
"$PROJECT_DIR/scripts/start_memory_node.sh" start \
  -p "$PORT" \
  -n 1 \
  --mn-memory "$MN_MEMORY" \
  --index-file "$SHARD_FILE" \
  --storage-id 0 \
  --insert-execution storage_owner \
  --storage-peers "$SERVER" \
  --dim "$DIM" \
  --R "$R" \
  --beam-width "$BEAM_WIDTH" \
  --beam-width-construction "$BEAM_WIDTH" \
  --k "$TOPK" \
  --rabitq-bits "$RABITQ_BITS" \
  --search-mode rabitq_gpu \
  --index-prefix "$INDEX_PREFIX" \
  --max-vectors 512 \
  --storage-owner-cache-mb 64 \
  --storage-owner-rpc-depth 2 \
  --storage-owner-search-snapshot-batch 16 \
  --storage-owner-prune-max-candidates 32 \
  --disable-thread-pinning
MEMORY_STARTED=1

echo "[offline-layout-smoke] run query/update/recall check"
"$BUILD_DIR/test/DvstorOfflineLayoutSmokeTest" \
  "$CONFIG_FILE" \
  "$DATA_DIR/base.fbin" \
  "$DATA_DIR/query.fbin" \
  "$DATA_DIR/groundtruth.bin" \
  "$MIN_RECALL"

echo "[offline-layout-smoke] passed"
