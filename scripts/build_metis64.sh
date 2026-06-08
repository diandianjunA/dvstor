#!/usr/bin/env bash
# One-time setup: build METIS 5.2.1 with 64-bit idx_t and install locally.
# This is required for graphs where the adjacency entry count exceeds 2^31-1
# (~2.1 billion), such as SIFT100M with R=48 (~4.3B edges).
#
# After running this script once, use:
#   ./scripts/reconfigure_metis64.sh
# to configure the project to use the 64-bit METIS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
METIS64_DIR="$PROJECT_DIR/third_party/metis64"
GKLIB_DIR="$HOME/local"
BUILD_TMPDIR="/tmp/metis64_build_$$"

if [ -f "$METIS64_DIR/lib/libmetis.so" ] && [ -f "$GKLIB_DIR/lib/libGKlib.a" ]; then
    echo "64-bit METIS and GKlib already installed."
    echo "  METIS: $METIS64_DIR/lib/libmetis.so"
    echo "  GKlib: $GKLIB_DIR/lib/libGKlib.a"
    exit 0
fi

echo "=== Building GKlib ==="
if [ ! -f "$GKLIB_DIR/lib/libGKlib.a" ]; then
    rm -rf "$BUILD_TMPDIR/GKlib"
    git clone --depth 1 https://github.com/KarypisLab/GKlib.git "$BUILD_TMPDIR/GKlib" 2>/dev/null || {
        echo "WARNING: Could not clone GKlib from GitHub. Trying to use cached source..."
    }
    if [ -d "$BUILD_TMPDIR/GKlib" ]; then
        mkdir -p "$BUILD_TMPDIR/GKlib/build"
        cd "$BUILD_TMPDIR/GKlib/build"
        cmake .. -DCMAKE_INSTALL_PREFIX="$GKLIB_DIR" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        make -j$(nproc)
        make install
        echo "GKlib installed to $GKLIB_DIR"
    fi
else
    echo "GKlib already installed at $GKLIB_DIR"
fi

echo ""
echo "=== Building METIS (64-bit idx_t) ==="
if [ ! -f "$METIS64_DIR/lib/libmetis.so" ]; then
    rm -rf "$BUILD_TMPDIR/metis"
    git clone --depth 1 --branch v5.2.1 https://github.com/KarypisLab/METIS.git "$BUILD_TMPDIR/metis" 2>/dev/null || {
        echo "WARNING: Could not clone METIS from GitHub."
        exit 1
    }

    mkdir -p "$BUILD_TMPDIR/metis/build/xinclude"
    echo '#define IDXTYPEWIDTH 64' > "$BUILD_TMPDIR/metis/build/xinclude/metis.h"
    echo '#define REALTYPEWIDTH 32' >> "$BUILD_TMPDIR/metis/build/xinclude/metis.h"
    cat "$BUILD_TMPDIR/metis/include/metis.h" >> "$BUILD_TMPDIR/metis/build/xinclude/metis.h"
    cp "$BUILD_TMPDIR/metis/include/CMakeLists.txt" "$BUILD_TMPDIR/metis/build/xinclude/"

    cd "$BUILD_TMPDIR/metis/build"
    cmake "$BUILD_TMPDIR/metis" \
        -DCMAKE_VERBOSE_MAKEFILE=1 \
        -DGKLIB_PATH="$GKLIB_DIR" \
        -DCMAKE_INSTALL_PREFIX="$METIS64_DIR" \
        -DSHARED=1 \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -j$(nproc)
    make install
    echo "METIS 64-bit installed to $METIS64_DIR"
else
    echo "METIS 64-bit already installed at $METIS64_DIR"
fi

# Verify
echo ""
echo "=== Verification ==="
echo '#include <stdio.h>
#define IDXTYPEWIDTH 64
#include "metis.h"
int main() { printf("sizeof(idx_t)=%zu max=%ld\n", sizeof(idx_t), (long)IDX_MAX); return 0; }' | gcc -x c - -I"$METIS64_DIR/include" -L"$METIS64_DIR/lib" -L"$GKLIB_DIR/lib" -lmetis -lGKlib -lm -Wl,-rpath,"$METIS64_DIR/lib" -o /tmp/test_metis64_verify && /tmp/test_metis64_verify

echo ""
echo "Done. Now run: ./scripts/reconfigure_metis64.sh"
rm -rf "$BUILD_TMPDIR"
