# GPU-Persistent Disaggregated Query Engine

## Execution Model

The `gpu_persistent` engine makes the GPU the query scheduler rather than a
distance-computation coprocessor. CPU client threads decode requests into pinned
staging slots, and an admission thread submits micro-batches to a mapped MPMC
ring. A single long-lived CUDA kernel consumes the ring and performs RaBitQ
navigation, remote fetch scheduling, exact reranking, and result publication.

The query data path is:

1. CPU admission into pinned query slots.
2. One batched H2D transfer and ring publication.
3. GPU-resident RaBitQ selection over shard-balanced entry points and hot-edge traversal.
4. Optional 4 KiB cold-adjacency reads through a GPU direct-mapped page cache.
5. Exact-vector RDMA reads into registered GPU memory.
6. GPU exact reranking and completion-ring publication.

`gpunetio` is the primary backend. It gives the persistent kernel multiple RC
QPs per memory node on one registered GPU memory region, allowing GPU-initiated
RDMA reads without a CPU CQ polling loop. `verbs_proxy` preserves the same
engine and GPU buffers while a CPU proxy posts batched RDMA reads. `local` is a
functional storage-file backend for development.

## Tiered Index

The offline builder emits two additional artifacts when
`--gpu-tiered-index` is enabled:

- `<prefix>.gpu.idx`: versioned node records, hot neighbors, full RaBitQ
  entries, shard regions, and the centroid.
- `<prefix>_nodeX_ofN.gpu.pages`: page-aligned full adjacency lists loaded into
  the memory node's registered region after the base shard.

The online engine keeps hot edges and RaBitQ entries resident. Cold pages are
read only for a bounded number of promising expansions and are shared across
queries by the GPU page cache.

## Dynamic Updates

Successful storage-owner mutations are acknowledged only after their vector or
tombstone is published to the GPU delta index. Queries capture a published
epoch at GPU admission and filter superseded base/delta generations against
that epoch. Repeated updates are compacted in the background. Existing base IDs
are merged into GPU node/RaBitQ records after active queries drain; dynamic IDs
remain as a compact live delta until a future full topology rebuild.

## SIFT100M Run

Build the required tiered artifacts and run the new profile:

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

Compare the generated report with a DVSTOR or OdinANN JSON report:

```bash
python3 experiment/compare_reports.py \
  --baseline /path/to/odinann.json \
  --candidate experiment/reports/04_gpu_persistent_gpunetio/report.json
```

Important report fields are under `gpu_persistent`: admission batch size,
submission/completion delay, RDMA bytes and operations, page-cache hit ratio,
delta visibility latency, and compaction progress.
