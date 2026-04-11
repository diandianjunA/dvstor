#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "This probe usually needs root on this host because GPU/RDMA device access is restricted."
  echo "Run: sudo $0"
  exit 1
fi

GPU_PCI="${GPU_PCI:-$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n1 || true)}"
IBDEV="${IBDEV:-$(ls /sys/class/infiniband | head -n1)}"
NIC_PCI="${NIC_PCI:-$(basename "$(readlink -f "/sys/class/infiniband/${IBDEV}/device")")}"

if [[ -z "${GPU_PCI}" || -z "${IBDEV}" || -z "${NIC_PCI}" ]]; then
  echo "FAIL: unable to auto-detect GPU/NIC PCI addresses"
  exit 1
fi

echo "Using GPU_PCI=${GPU_PCI} IBDEV=${IBDEV} NIC_PCI=${NIC_PCI}"

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build_probe" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT_DIR}/build_probe" -j --target GpuRdmaProbe

exec "${ROOT_DIR}/build_probe/test/GpuRdmaProbe" \
  --gpu-pci "${GPU_PCI}" \
  --nic-pci "${NIC_PCI}" \
  --ibdev "${IBDEV}"
