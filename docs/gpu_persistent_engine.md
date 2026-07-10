# GPU-Persistent Disaggregated Query Engine

## Converting an Existing Index

An existing schema-13 Vamana index can be reused without rebuilding either the
graph or its Metis placement. The converter reads the existing `.dat`, owner
`.idmap`, metadata, and full RaBitQ entries and emits only the GPU-tiered files:

```bash
./experiment/convert_sift100m_gpu_sidecars.sh 04_gpu_persistent_gpunetio
```

The output consists of `${INDEX_PREFIX}.gpu.idx` and one
`${INDEX_PREFIX}_nodeX_ofN.gpu.pages` file per storage shard. The old `.dat`
files remain unchanged. Conversion requires a quiescent, static schema-13 index
with dense IDs and `node_layout=rabitq`; persisted dynamic records must first be
compacted. Full `.rabitq12` sidecars are used when available, otherwise the
converter copies full RaBitQ entries from each fixed node. Set
`GPU_SIDECAR_OVERWRITE=1` to replace existing GPU sidecars.

## Execution Model

The `gpu_persistent` engine makes the GPU the query scheduler rather than a
distance-computation coprocessor. CPU client threads decode requests into pinned
staging slots, and an admission thread submits micro-batches to a mapped MPMC
ring. A single long-lived CUDA kernel consumes the ring and performs RaBitQ
navigation, remote fetch scheduling, exact reranking, and result publication.

The query data path is:

1. CPU admission into pinned query slots.
2. Batched admission with asynchronous H2D copies and ring publication.
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

The persistent grid launches multiple query blocks per SM (bounded by active
query slots) so blocks waiting for RDMA completions do not monopolize an SM.

## Dynamic Updates

Successful storage-owner mutations are acknowledged only after their vector or
tombstone is published to the GPU delta index. Queries capture a published
epoch at GPU admission and filter superseded base/delta generations against
that epoch. Repeated updates are compacted in the background. Existing base IDs
are merged into GPU node/RaBitQ records after active queries drain; dynamic IDs
remain as a compact live delta until a future full topology rebuild.

The delta signature filter is disabled by default because its Hamming-radius
cutoff is a lossy candidate filter. It may be enabled explicitly for throughput
experiments, but those runs must report recall separately.

## RaBitQ Correctness Scope

The GPU tiered format stores the complete one-bit code, centered-vector norm,
and per-vector correlation correction used by the asymmetric distance
estimator. Sidecar/tiered-layout tests cover padded and non-padded code widths.
Exact reranking then computes L2 on the selected remote vectors.

This is enough to represent and evaluate the estimator, but it does not by
itself guarantee end-to-end recall. The current transform is one fixed
random-sign Hadamard round, whereas the official fast RaBitQ implementation
uses a stronger multi-round randomized rotator. In addition, graph traversal,
cold-expansion limits, final gate width, and optional delta filtering can omit a
true neighbor before exact reranking. Measure the quantizer-only candidate
coverage independently with:

```bash
python3 experiment/audit_rabitq.py \
  --metadata /path/to/index.meta.json \
  --vectors /path/to/base.u8bin \
  --queries /path/to/query.u8bin
```

The reference profile exact-reranks all candidates in its 64-wide final beam.
Widths up to 128 are supported for recall/performance sweeps.

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

Before a distributed run, validate the local CUDA/DOCA/DMA-BUF registration
path for the selected GPU and RDMA device:

```bash
./build/dvstor_gpunetio_probe 1 mlx5_0
```

This probe verifies the local GPUNetIO capability and the exact GPU-memory
registration path used by the engine. A connected memory node is still required
to validate QP exchange and end-to-end RDMA reads.

Important report fields are under `gpu_persistent`: admission batch size,
submission/completion delay, RDMA bytes and operations, page-cache hit ratio,
delta visibility latency, and compaction progress.
