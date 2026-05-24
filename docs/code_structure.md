# Code Structure

This project keeps latency-sensitive Vamana, RDMA, and GPU paths close to their existing call sites. Structural changes should preserve data layout, avoid virtual dispatch on hot paths, and keep coroutine/RDMA helper functions inline when they are used inside tight loops.

## Runtime Modules

- `src/common/`: shared configuration, timing, statistics, and distance helpers.
- `src/gpu/`: CUDA buffer management, kernel launchers, and GPU RaBitQ cache code.
- `src/rdma/`: Vamana RDMA read, write, and atomic operation helpers.
- `src/service/`: compute-node orchestration, routing, breakdown accounting, and storage-owner client-side protocol helpers.
- `src/memory_node/`: memory-node support types that are not the top-level service loop.
- `src/vamana/`: core Vamana index structures and CPU/GPU search logic.

## Storage-Owner Split

Storage-owner logic has two sides:

- Compute-node client side:
  - `src/service/storage_owner_protocol.hh` defines wire-format request, response, and peer-RPC structures.
  - `src/service/storage_owner_client_helpers.hh` maps storage-owner timing counters into breakdown samples and owns small client-side utility helpers.
  - `src/service/compute_service.cc` owns request scheduling, RDMA send/receive completion, and result delivery.

- Memory-node owner side:
  - `src/memory_node/storage_owner_state.hh` owns storage-owner runtime state structures, local cache state, peer RPC buffers, and per-worker scratch state.
  - `src/memory_node.hh` owns the actual storage-owner execution, peer RDMA operations, reverse updates, and memory-node service loop.

## Refactoring Rules

- Prefer moving state-only structs and accounting helpers before splitting hot-path algorithms.
- Keep protocol structs in `src/service/storage_owner_protocol.hh`; do not duplicate wire layouts.
- Keep hot-path helpers inline or in headers unless profiling shows a negligible cost for out-of-line calls.
- Do not change RDMA work request ordering or buffer ownership while doing structural refactors.
- Add or update smoke tests for behavior changes; pure file splits should at least build `dvstor_memory_node` and `dvstor_breakdown_benchmark`.
