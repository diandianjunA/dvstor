#!/bin/bash
# =============================================================================
# Recall Test: Build index and evaluate recall against ground truth
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/vamana_offline_builder"

# Defaults for 1024dim-1M dataset
DATA_PATH="${DATA_PATH:-/data/xjs/random_dataset/1024dim1M}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/data/xjs/index/shine_gpu_index/1024dim1M}"
QUERY_PATH="${QUERY_PATH:-${DATA_PATH}/queries/query-test.fbin}"
GT_PATH="${GT_PATH:-${DATA_PATH}/queries/groundtruth-test.bin}"
R="${R:-32}"
ALPHA="${ALPHA:-1.2}"
THREADS="${THREADS:-0}"
MAX_VECTORS="${MAX_VECTORS:-4294967295}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-path)         DATA_PATH="$2"; shift 2 ;;
        -o|--output-prefix)     OUTPUT_PREFIX="$2"; shift 2 ;;
        -q|--query-path)        QUERY_PATH="$2"; shift 2 ;;
        -g|--groundtruth-path)  GT_PATH="$2"; shift 2 ;;
        --R)                    R="$2"; shift 2 ;;
        --alpha)                ALPHA="$2"; shift 2 ;;
        -t|--threads)           THREADS="$2"; shift 2 ;;
        --max-vectors)          MAX_VECTORS="$2"; shift 2 ;;
        --no-gpu)               NO_GPU="--no-gpu"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -d, --data-path <path>       Base vectors directory (default: $DATA_PATH)"
            echo "  -o, --output-prefix <path>   Index output prefix (default: $OUTPUT_PREFIX)"
            echo "  -q, --query-path <path>      Query file (.fbin)"
            echo "  -g, --groundtruth-path <path> Ground truth file (.bin)"
            echo "      --R <n>                  Max out-degree (default: $R)"
            echo "      --alpha <f>              Alpha parameter (default: $ALPHA)"
            echo "  -t, --threads <n>            Thread count (default: auto)"
            echo "      --max-vectors <n>        Limit dataset size"
            echo "      --no-gpu                 Disable GPU"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found: $BINARY"
    echo "Build with: cd $PROJECT_DIR/build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "=== Recall Test Configuration ==="
echo "  data:        $DATA_PATH"
echo "  output:      $OUTPUT_PREFIX"
echo "  queries:     $QUERY_PATH"
echo "  groundtruth: $GT_PATH"
echo "  R=$R alpha=$ALPHA threads=$THREADS"
echo ""

exec "$BINARY" \
    --data-path "$DATA_PATH" \
    --output-prefix "$OUTPUT_PREFIX" \
    --query-path "$QUERY_PATH" \
    --groundtruth-path "$GT_PATH" \
    --R "$R" \
    --alpha "$ALPHA" \
    --threads "$THREADS" \
    --max-vectors "$MAX_VECTORS" \
    ${NO_GPU:-}
