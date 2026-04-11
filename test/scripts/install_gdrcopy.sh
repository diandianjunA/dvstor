#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root."
  exit 1
fi

WORK_DIR="${1:-/tmp/gdrcopy-src}"
REPO_URL="${GDRCOPY_REPO_URL:-https://github.com/NVIDIA/gdrcopy.git}"
GDRDRV_VERSION="2.5.2"

patch_gdrdrv_makefile() {
  local makefile="/usr/src/gdrdrv-${GDRDRV_VERSION}/Makefile"
  if [[ ! -f "${makefile}" ]]; then
    return 0
  fi

  if grep -q "open-gpu-kernel-modules" "${makefile}"; then
    return 0
  fi

  if find /usr/src/open-gpu-kernel-modules-* -name "nv-p2p.c" -print -quit >/dev/null 2>&1; then
    echo "==> Patching gdrdrv Makefile for NVIDIA open kernel module layout"
    sed -i 's|find /usr/src/kernel-modules/nvidia-\* /usr/src/nvidia-\* -name "nv-p2p.c"|find /usr/src/kernel-modules/nvidia-* /usr/src/nvidia-* /usr/src/open-gpu-kernel-modules-* -name "nv-p2p.c"|' "${makefile}"
  fi
}

rebuild_gdrdrv_dkms() {
  echo "==> Rebuilding gdrdrv DKMS module"
  dkms remove -m gdrdrv -v "${GDRDRV_VERSION}" --all >/dev/null 2>&1 || true
  dkms add -m gdrdrv -v "${GDRDRV_VERSION}"
  dkms build -m gdrdrv -v "${GDRDRV_VERSION}"
  dkms install -m gdrdrv -v "${GDRDRV_VERSION}"
}

echo "==> Installing GDRCopy build prerequisites"
apt-get update
apt-get install -y git build-essential dkms linux-headers-"$(uname -r)"

echo "==> Preparing source tree in ${WORK_DIR}"
rm -rf "${WORK_DIR}"
git clone --depth=1 "${REPO_URL}" "${WORK_DIR}"

cd "${WORK_DIR}/packages"

echo "==> Building Debian packages"
CUDA=/usr/local/cuda ./build-deb-packages.sh

echo "==> Installing generated packages"
mapfile -t debs < <(find "${WORK_DIR}/packages" -maxdepth 1 -type f -name '*.deb' | sort)
if [[ ${#debs[@]} -eq 0 ]]; then
  echo "FAIL: no generated .deb packages found under ${WORK_DIR}/packages"
  exit 2
fi
dpkg -i "${debs[@]}" || true

patch_gdrdrv_makefile
rebuild_gdrdrv_dkms
dpkg --configure gdrdrv-dkms gdrcopy libgdrapi gdrcopy-tests

echo "==> Loading gdrdrv"
modprobe gdrdrv
lsmod | grep gdrdrv

echo "==> Running userspace sanity"
gdrcopy_sanity
