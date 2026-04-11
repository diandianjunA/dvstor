# DvstorIndex Smoke Test

This directory contains a small in-repository test for the in-process `DvstorIndex` wrapper.

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target DvstorIndexSmokeTest
```

Run on a single machine after a memory node is up:

```bash
./scripts/start_memory_node.sh -f --mn-memory 10
./build/test/DvstorIndexSmokeTest ./test/config/local_single_cn.ini
```

The test process itself acts as the compute node. It inserts a small synthetic dataset, checks
that self-query succeeds, stores the index, reloads it, and checks the query again.

It then runs a concurrent insertion stress test. You can override its default concurrency:

```bash
./build/test/DvstorIndexSmokeTest ./test/config/local_single_cn.ini /tmp/dvstor_test_idx 8 64
```

The last two arguments are:

- `concurrent_threads`
- `vectors_per_thread`

## GPUDirect RDMA and GPUNetIO Checks

Use the helper scripts below before moving traversal, insertion, or query orchestration fully onto the GPU:

```bash
./test/scripts/check_gpu_rdma_env.sh
./test/scripts/run_gpunetio_rdma_probe.sh
./test/scripts/run_gpu_rdma_probe.sh
```

`check_gpu_rdma_env.sh` validates the local prerequisites for GPUDirect RDMA:

- `nvidia-smi` and `/dev/nvidiactl`
- `/dev/infiniband/uverbs0`, `ibstat`, and `ibv_devices`
- DOCA GPUNetIO / RDMA packages
- `gdrcopy_sanity`

On recent CUDA/NVIDIA driver stacks, `gdrcopy_sanity` may report a single failure in `basic_with_tokens`. This repository treats that as a known compatibility exception because the token-based GPUDirect path is deprecated; the non-token path is the one that matters for current GPUDirect RDMA validation.

`run_gpunetio_rdma_probe.sh` is the next-stage entry point for GPU-initiated RDMA validation with NVIDIA GPUNetIO. It expects:

- `/opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio.so`
- `/opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_rdma.so`
- a built GPUNetIO RDMA sample or probe binary

`run_gpu_rdma_probe.sh` builds and runs the in-repo `GpuRdmaProbe` executable. It validates two concrete capabilities:

- GPU memory can be exported as DMABUF and registered to RNIC memory region state through `ibv_reg_dmabuf_mr`
- A local DOCA Verbs RC QP/CQ can be exported to GPUNetIO with `doca_gpu_verbs_export_qp`

If the machine is missing device nodes or DOCA libraries, fix the system environment first. On NVIDIA's DOCA packaging, the usual install line is:

```bash
sudo apt install doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev
```

For this repository, a fuller root-side setup helper is also provided:

```bash
sudo ./test/scripts/install_gpunetio_env.sh
```

That script first checks whether the host Ubuntu release matches the installed DOCA host repository, then installs the DOCA GPUNetIO / RDMA development packages, runs `nvidia-modprobe`, loads `nvidia_peermem`, and checks whether `gdrdrv` is present. If `gdrdrv` is still missing, build and install GDRCopy's kernel module from NVIDIA's official source:

- Ubuntu 20.04 prefers a focal-compatible DOCA 3.2.1 repository, and falls back to DOCA 3.1.0 if the 3.2.1 repo is temporarily inconsistent
- Ubuntu 22.04 and 24.04 use the newer DOCA 3.x package line

- GDRCopy: https://github.com/NVIDIA/gdrcopy
- DOCA GPUNetIO install reference: https://docs.nvidia.com/doca/archive/3-2-0/DOCA-GPUNetIO/index.html

For GDRCopy on this host, use:

```bash
sudo ./test/scripts/install_gdrcopy.sh
```

That helper clones NVIDIA's official GDRCopy repository, builds the Debian packages, installs them, loads `gdrdrv`, and runs `gdrcopy_sanity`.
