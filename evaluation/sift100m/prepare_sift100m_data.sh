#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/sift100m_common.sh"

PREPARE_BASE="${PREPARE_BASE:-1}"
PREPARE_QUERY="${PREPARE_QUERY:-1}"
PREPARE_GROUNDTRUTH="${PREPARE_GROUNDTRUTH:-1}"
convert_args=()
if [[ "$PREPARE_BASE" != "1" ]]; then convert_args+=(--skip-base); fi
if [[ "$PREPARE_QUERY" != "1" ]]; then convert_args+=(--skip-query); fi
if [[ "$PREPARE_GROUNDTRUTH" != "1" ]]; then convert_args+=(--skip-groundtruth); fi

python3 "$SCRIPT_DIR/convert_sift100m.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$CONVERTED_DIR" \
  --groundtruth-label "$GROUNDTRUTH_LABEL" \
  --max-base "$MAX_VECTORS" \
  --max-query "$MAX_QUERIES" \
  --topk "$GROUNDTRUTH_TOPK" \
  --chunk-rows "${CONVERT_CHUNK_ROWS:-1000000}" \
  "${convert_args[@]}"

if [[ "$PREPARE_BASE" == "1" ]]; then echo "base:        $(base_bin)"; fi
if [[ "$PREPARE_QUERY" == "1" ]]; then echo "query:       $(query_bin)"; fi
if [[ "$PREPARE_GROUNDTRUTH" == "1" ]]; then echo "groundtruth: $(groundtruth_bin)"; fi

# Insert test vectors (real SIFT data for benchmark, not synthetic)
INSERT_SRC="${INSERT_SRC:-/data/xjs/datasets/sift1b/sift100m_insert_test.bvecs}"
INSERT_BIN="$(insert_bin)"
PREPARE_INSERT="${PREPARE_INSERT:-1}"
if [[ "$PREPARE_INSERT" == "1" && ! -f "$INSERT_BIN" ]]; then
  if [[ -f "$INSERT_SRC" ]]; then
    echo "converting insert vectors: $INSERT_SRC → $INSERT_BIN"
    python3 -c "
import struct, sys
with open('$INSERT_SRC', 'rb') as fin, open('$INSERT_BIN', 'wb') as fout:
    vectors, dim = [], None
    while True:
        h = fin.read(4)
        if not h: break
        d = struct.unpack('<I', h)[0]
        if dim is None: dim = d
        vectors.append(fin.read(d))
    fout.write(struct.pack('<II', len(vectors), dim))
    for v in vectors: fout.write(v)
    print(f'OK: {len(vectors)} vectors, dim={dim}')
" && echo "insert:      $INSERT_BIN"
  else
    echo "WARNING: insert source not found at $INSERT_SRC — benchmark will use synthetic vectors"
  fi
fi
