#!/usr/bin/env bash
set -euo pipefail

failures=0

check_cmd() {
  local name="$1"
  local cmd="$2"
  echo "==> ${name}"
  if bash -lc "${cmd}"; then
    echo "PASS: ${name}"
  else
    echo "FAIL: ${name}"
    failures=$((failures + 1))
  fi
  echo
}

check_path() {
  local name="$1"
  local path="$2"
  echo "==> ${name}"
  if [[ -e "${path}" ]]; then
    ls -l "${path}"
    echo "PASS: ${name}"
  else
    echo "FAIL: missing ${path}"
    failures=$((failures + 1))
  fi
  echo
}

check_cmd "CUDA compiler present" "nvcc --version"
check_cmd "NVIDIA driver responds" "nvidia-smi"
check_path "NVIDIA control device" "/dev/nvidiactl"
check_path "Infiniband verbs device" "/dev/infiniband/uverbs0"
check_cmd "NVIDIA kernel modules" "lsmod | rg 'nvidia|nvidia_uvm'"
check_cmd "RDMA kernel modules" "lsmod | rg 'mlx5_ib|ib_uverbs|rdma_cm'"
check_cmd "NVIDIA peer memory module loaded" "lsmod | rg 'nvidia_peermem'"
check_cmd "GDRCopy kernel module available" "modinfo gdrdrv >/dev/null"
check_cmd "IB port state" "ibstat"
check_cmd "Userspace verbs device discovery" "ibv_devices | tail -n +3 | rg -q '\\S'"
check_cmd "DOCA GPUNetIO packages" "dpkg -l | rg 'doca-sdk-gpunetio|libdoca-sdk-gpunetio-dev'"
check_cmd "DOCA RDMA packages" "dpkg -l | rg 'doca-sdk-rdma|libdoca-sdk-rdma-dev|doca-sdk-verbs|libdoca-sdk-verbs-dev'"
check_cmd "GDRCopy userspace sanity" "$(cd "$(dirname "$0")" && pwd)/check_gdrcopy_sanity.sh"

if [[ ${failures} -ne 0 ]]; then
  echo "Environment check failed with ${failures} issue(s)."
  exit 1
fi

echo "Environment check passed."
