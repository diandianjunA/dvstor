#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/check_gpu_rdma_env.sh"

echo "==> Looking for DOCA GPUNetIO libraries"
if [[ ! -f /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio.so ]]; then
  echo "FAIL: /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio.so is missing."
  echo "Install: apt install doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev"
  exit 1
fi

if [[ ! -f /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_rdma.so ]]; then
  echo "FAIL: /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_rdma.so is missing."
  exit 1
fi

echo "==> Looking for GPUNetIO RDMA sample binary"
SAMPLE_BIN="${1:-/opt/mellanox/doca/samples/doca_gpunetio/gpunetio_rdma_client_server_write/build/doca_gpunetio_rdma_client_server_write}"
if [[ ! -x "${SAMPLE_BIN}" ]]; then
  echo "FAIL: sample binary not found: ${SAMPLE_BIN}"
  echo "This host currently exposes only a stale build directory under /opt/mellanox/doca/samples."
  echo "After the SDK is fully installed, rebuild the official sample or point this script to your own GPUNetIO RDMA probe binary."
  exit 1
fi

echo "==> Running GPUNetIO RDMA probe"
echo "Binary: ${SAMPLE_BIN}"
echo "Note: GPUNetIO RDMA tests typically need two peers. Pass a rebuilt sample path as arg1 if needed."
exec "${SAMPLE_BIN}" "${@:2}"
