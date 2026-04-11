#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /etc/os-release

HOST_VERSION="${VERSION_ID:-unknown}"
DOCA_HOST_PKG="$(dpkg-query -W -f='${Version}\n' doca-host 2>/dev/null || true)"
FOCAL_DOCA_REPO_PRIMARY="https://linux.mellanox.com/public/repo/doca/3.2.1/ubuntu20.04/x86_64"
FOCAL_DOCA_REPO_FALLBACK="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu20.04/x86_64"
FOCAL_DOCA_LIST="/etc/apt/sources.list.d/doca-focal.list"
DISABLED_SOURCES=()

disable_problematic_sources() {
  local docker_sources
  mapfile -t docker_sources < <(grep -R -l "download.docker.com/linux/ubuntu" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null || true)
  for src in "${docker_sources[@]}"; do
    if [[ -f "${src}" ]]; then
      mv "${src}" "${src}.codex-disabled"
      DISABLED_SOURCES+=("${src}")
      echo "==> Temporarily disabled APT source: ${src}"
    fi
  done
}

restore_problematic_sources() {
  local src
  for src in "${DISABLED_SOURCES[@]}"; do
    if [[ -f "${src}.codex-disabled" ]]; then
      mv "${src}.codex-disabled" "${src}"
      echo "==> Restored APT source: ${src}"
    fi
  done
}

trap restore_problematic_sources EXIT

echo "==> Host OS: Ubuntu ${HOST_VERSION}"
if [[ -n "${DOCA_HOST_PKG}" ]]; then
  echo "==> Installed doca-host: ${DOCA_HOST_PKG}"
fi

if [[ "${HOST_VERSION}" == "20.04" ]]; then
  echo "==> Configuring Ubuntu 20.04-compatible DOCA repository"
  rm -f /etc/apt/sources.list.d/doca.list
  rm -f /etc/apt/sources.list.d/doca.list.save
  rm -f /etc/apt/sources.list.d/doca-host.list
  rm -f /etc/apt/sources.list.d/mlnx-doca.list
  rm -f /etc/apt/sources.list.d/doca-focal-2.5.5.list
  rm -f /etc/apt/sources.list.d/doca-focal-3.2.1.list
  if [[ -f /usr/share/keyrings/cuda-archive-keyring.gpg ]]; then
    echo "==> Using explicit Mellanox repo entry"
  fi
  if dpkg -s doca-host >/dev/null 2>&1; then
    echo "==> Removing mismatched doca-host local repo package"
    apt-get remove -y doca-host || true
  fi

  echo "==> Note: NVIDIA's support table for Ubuntu 20.04 is based on the 5.4 generic kernel."
  echo "==> Current kernel: $(uname -r)"
  disable_problematic_sources

  selected_repo=""
  selected_version=""
  selected_pkg_version=""
  for repo in "${FOCAL_DOCA_REPO_PRIMARY}" "${FOCAL_DOCA_REPO_FALLBACK}"; do
    cat > "${FOCAL_DOCA_LIST}" <<EOF
deb [trusted=yes] ${repo} ./
EOF
    echo "==> Trying DOCA repo: ${repo}"
    rm -f /var/lib/apt/lists/linux.mellanox.com_public_repo_doca_* || true
    update_ok=0
    for attempt in 1 2 3; do
      echo "APT update attempt ${attempt}/3"
      if apt-get update; then
        update_ok=1
        break
      fi
      sleep 5
      rm -f /var/lib/apt/lists/linux.mellanox.com_public_repo_doca_* || true
    done
    if [[ ${update_ok} -eq 1 ]]; then
      selected_repo="${repo}"
      if [[ "${repo}" == "${FOCAL_DOCA_REPO_PRIMARY}" ]]; then
        selected_version="3.2.1"
        selected_pkg_version="3.2.1025-1"
      else
        selected_version="3.1.0"
        selected_pkg_version="3.1.0105-1"
      fi
      break
    fi
  done

  if [[ -z "${selected_repo}" ]]; then
    echo "FAIL: Unable to refresh either DOCA 3.2.1 or DOCA 3.1.0 focal repositories."
    exit 5
  fi

  echo "==> Selected DOCA focal repo: ${selected_repo}"
  apt-get install -y \
    doca-sdk-gpunetio="${selected_pkg_version}" \
    libdoca-sdk-gpunetio-dev="${selected_pkg_version}" \
    doca-sdk-rdma="${selected_pkg_version}" \
    libdoca-sdk-rdma-dev="${selected_pkg_version}" \
    doca-sdk-verbs="${selected_pkg_version}" \
    libdoca-sdk-verbs-dev="${selected_pkg_version}" \
    nvidia-modprobe \
    doca-extra \
    dkms \
    build-essential \
    debhelper \
    devscripts \
    fakeroot \
    pkg-config

  echo "==> Creating NVIDIA device files if needed"
  nvidia-modprobe -u -c=0 || true

  echo "==> Loading peer memory module"
  modprobe nvidia_peermem

  if modinfo gdrdrv >/dev/null 2>&1; then
    echo "==> Loading existing gdrdrv module"
    modprobe gdrdrv
  else
    echo "==> gdrdrv module not found"
    echo "Build and install GDRCopy from the official source tree, then re-run this script."
    echo "Reference: https://github.com/NVIDIA/gdrcopy"
    exit 2
  fi

  echo "==> Re-running environment probe"
  "${ROOT_DIR}/test/scripts/check_gpu_rdma_env.sh"
  exit 0
fi

if [[ "${HOST_VERSION}" != "22.04" && "${HOST_VERSION}" != "24.04" ]]; then
  echo "FAIL: Unsupported Ubuntu release ${HOST_VERSION}."
  echo "Supported paths in this script are Ubuntu 20.04 with DOCA 2.5.5, or Ubuntu 22.04/24.04 with DOCA 3.x."
  exit 3
fi

if [[ -n "${DOCA_HOST_PKG}" && "${HOST_VERSION}" == "22.04" && "${DOCA_HOST_PKG}" == *ubuntu2404* ]]; then
  echo "FAIL: Installed doca-host package targets Ubuntu 24.04, but this machine is Ubuntu 22.04."
  echo "Install the Ubuntu 22.04 DOCA local repo package first, then re-run this script."
  exit 4
fi

if [[ -n "${DOCA_HOST_PKG}" && "${HOST_VERSION}" == "24.04" && "${DOCA_HOST_PKG}" != *ubuntu2404* ]]; then
  echo "FAIL: Installed doca-host package does not match Ubuntu 24.04."
  echo "Install the Ubuntu 24.04 DOCA local repo package first, then re-run this script."
  exit 4
fi

echo "==> Installing DOCA GPUNetIO/RDMA development packages"
apt-get update
apt-get install -y \
  doca-sdk-gpunetio \
  libdoca-sdk-gpunetio-dev \
  doca-sdk-rdma \
  libdoca-sdk-rdma-dev \
  doca-sdk-verbs \
  libdoca-sdk-verbs-dev \
  nvidia-modprobe \
  dkms \
  build-essential \
  debhelper \
  devscripts \
  fakeroot \
  pkg-config

echo "==> Creating NVIDIA device files if needed"
nvidia-modprobe -u -c=0 || true

echo "==> Loading peer memory module"
modprobe nvidia_peermem

if modinfo gdrdrv >/dev/null 2>&1; then
  echo "==> Loading existing gdrdrv module"
  modprobe gdrdrv
else
  echo "==> gdrdrv module not found"
  echo "Build and install GDRCopy from the official source tree, then re-run this script."
  echo "Reference: https://github.com/NVIDIA/gdrcopy"
  exit 2
fi

echo "==> Re-running environment probe"
"${ROOT_DIR}/test/scripts/check_gpu_rdma_env.sh"
