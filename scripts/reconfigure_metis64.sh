#!/usr/bin/env bash
# Reconfigure the project with 64-bit METIS (IDXTYPEWIDTH=64).
# Required for large graphs where adjacency entry count exceeds 2^31-1.
#
# Usage: ./scripts/reconfigure_metis64.sh [build_dir]
#
# Prerequisites (one-time setup):
#   ./scripts/build_metis64.sh

set -euo pipefail

BUILD_DIR="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
METIS64_DIR="$PROJECT_DIR/third_party/metis64"
GKLIB_DIR="$HOME/local"

if [ ! -f "$METIS64_DIR/lib/libmetis.so" ]; then
    echo "ERROR: 64-bit METIS not found at $METIS64_DIR" >&2
    echo "Run: ./scripts/build_metis64.sh first." >&2
    exit 1
fi

mkdir -p "$PROJECT_DIR/$BUILD_DIR"
cd "$PROJECT_DIR/$BUILD_DIR"

cmake "$PROJECT_DIR" \
    -DMETIS_INCLUDE_DIR="$METIS64_DIR/include" \
    -DMETIS_LIBRARY="$METIS64_DIR/lib/libmetis.so" \
    -DGKLIB_LIBRARY="$GKLIB_DIR/lib/libGKlib.a" \
    -DCMAKE_BUILD_RPATH="$METIS64_DIR/lib;$GKLIB_DIR/lib" \
    -DCMAKE_INSTALL_RPATH="$METIS64_DIR/lib;$GKLIB_DIR/lib"

echo ""
echo "METIS 64-bit configured. Run 'make -j\$(nproc)' to build."
