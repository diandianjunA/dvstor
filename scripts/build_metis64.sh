#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${METIS64_SRC_DIR:-$PROJECT_DIR/thirdparty/src}"
INSTALL_DIR="${METIS64_INSTALL_DIR:-$PROJECT_DIR/thirdparty/metis64}"
METIS_REPO="${METIS_REPO:-https://github.com/KarypisLab/METIS.git}"
GKLIB_REPO="${GKLIB_REPO:-https://github.com/KarypisLab/GKlib.git}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$SRC_DIR" "$INSTALL_DIR"

if [[ ! -d "$SRC_DIR/METIS/.git" ]]; then
  git clone --recursive "$METIS_REPO" "$SRC_DIR/METIS"
else
  git -C "$SRC_DIR/METIS" fetch --tags --quiet || true
fi

if [[ ! -d "$SRC_DIR/METIS/GKlib" ]]; then
  if [[ ! -d "$SRC_DIR/GKlib/.git" ]]; then
    git clone "$GKLIB_REPO" "$SRC_DIR/GKlib"
  fi
  ln -sfn "$SRC_DIR/GKlib" "$SRC_DIR/METIS/GKlib"
fi

make -C "$SRC_DIR/METIS/GKlib" distclean >/dev/null 2>&1 || true
make -C "$SRC_DIR/METIS/GKlib" config prefix="$INSTALL_DIR" shared=1
make -C "$SRC_DIR/METIS/GKlib" -j"$JOBS"
make -C "$SRC_DIR/METIS/GKlib" install

make -C "$SRC_DIR/METIS" distclean >/dev/null 2>&1 || true
make -C "$SRC_DIR/METIS" config \
  prefix="$INSTALL_DIR" \
  gklib_path="$SRC_DIR/METIS/GKlib" \
  i64=1 \
  shared=1
make -C "$SRC_DIR/METIS" -j"$JOBS"
cmake --install "$SRC_DIR/METIS/build" --prefix "$INSTALL_DIR"

if ! grep -q '^#define IDXTYPEWIDTH 64' "$INSTALL_DIR/include/metis.h"; then
  echo "ERROR: $INSTALL_DIR/include/metis.h is not IDXTYPEWIDTH 64" >&2
  exit 1
fi

echo "METIS64 installed at: $INSTALL_DIR"
echo "Use CMake with: -DDVSTOR_METIS_ROOT=$INSTALL_DIR -DDVSTOR_METIS_PARTITION=ON"
