# GPU-Persistent Disaggregated Query Engine V4

## Design Boundary

V4 keeps the storage node's compact Vamana graph authoritative and removes the
compute-side graph replica. The only dense base array resident on the GPU is the
full RaBitQ entry (`code + norm + correction`, 24 bytes for SIFT). The compute
node does not keep a host copy of this array.

For SIFT1B the configured budget is:

- RaBitQ base codes: 24,000,000,000 bytes (22.35 GiB).
- Four-way adjacency cache: at most 7 GiB including tags and replacement state.
- Dynamic index: at most 2 GiB including vectors and sparse hash tables.
- Query workspaces, anchors, and metadata: charged by the startup budget model.
- Explicit engine allocations: at most 36 GiB; 4 GiB remains reserved, so the
  configured GPU limit is 40 GiB.

`gpu_memory_budget_test` checks both SIFT100M and a synthetic SIFT1B model
without preparing a one-billion-vector dataset.

## V4 Files

The distributed artifacts are:

- `<prefix>.meta.json`: storage layout and V4 metadata.
- `<prefix>.gpu.idx`: a small V4 manifest containing shard ranges, graph/code
  offsets, centroid, and entry-point ordinals.
- `<prefix>_nodeX_ofN.dat`: authoritative fixed nodes, exact vectors, and the
  compact 488-byte Vamana graph records for `R=96`.
- `<prefix>.anchors`: small routing anchors used for entry routing and dynamic
  candidate buckets.

`<prefix>_nodeX_ofN.gpu.codes` is optional. If present on a storage node, its
header, dimensions, offset, and checksum are validated before loading. If it is
absent, the storage node sequentially materializes the same contiguous RaBitQ
stream from the authoritative entries already present in its local `.dat`
shard. The stream occupies a reserved range immediately after the immutable
static image, and the dynamic allocator starts after that range.

The compute node needs only the manifest, metadata, and anchors on local or
shared storage. It does not need `.dat`, `.gpu.codes`, `.idmap`, `.rabitq12`, or
any graph snapshot locally in a distributed run. Storage nodes keep their own
`.dat`; storage-owner updates additionally use their local `.idmap` and anchor
files.

V3 `.gpu.idx` and `.gpu.pages` files are rejected with a migration error. Old
`.gpu.pages` files are never loaded. After validating V4, they can be removed
manually; the converter never deletes old files.

## Converting an Existing Index

The converter reuses the existing Vamana/Metis placement and never rebuilds the
graph. On a compute node where the `.dat` shards are remote, the script detects
that condition and writes only the small manifest:

```bash
GPU_SIDECAR_OVERWRITE=1 \
GPU_SIDECAR_RABITQ_SOURCE=nodes \
./experiment/convert_sift100m_gpu_sidecars.sh 04_gpu_persistent_gpunetio
```

In manifest-only mode no `.gpu.codes` file is created on the compute node. Each
storage node derives the advertised code-stream offset from the same persisted
`hot_graph_dynamic_base_offsets` metadata and materializes its local stream at
startup. When shards are local, `nodes` reads the authoritative full RaBitQ
entries from `.dat` and is the correctness-first default; `sidecar` and `auto`
remain optional conversion accelerators.

Conversion requires schema 13, `vamana_compact_v1`, `node_layout=rabitq`, and a
quiescent index with no persisted dynamic records. It performs sequential I/O
and is much cheaper than Vamana construction or Metis partitioning.

## Startup Protocol

1. Each storage node loads its local `.dat` shard into the registered RDMA
   region and validates the compact static layout.
2. It either validates and loads a local `.gpu.codes` payload or sequentially
   copies `code + norm + correction` from its authoritative fixed nodes into
   the manifest's reserved remote range.
3. The compute node allocates the final GPU layout once.
4. Two 64 MiB GPUDirect RDMA windows stream each shard directly into its final
   ordinal range. There is no full CPU staging array and no compute-side code
   snapshot file.
5. The persistent GPU kernel starts only after every streamed byte succeeds.

The storage stream consumes 24 bytes per SIFT vector across the storage nodes;
it consumes no compute-node disk space and is not mirrored in compute host
memory. The query backend can be `gpunetio`, `verbs_proxy`, or `local`. GPUNetIO owns
the steady-state query QPs; bootstrap uses the same remote registered regions
and writes directly to the final GPU allocation.

## Query Execution

Base nodes use a 30-bit ordinal handle. Dynamic nodes use a tagged delta handle.
Shard range metadata converts a base ordinal to its fixed-node and compact-graph
RDMA addresses without a dense node table.

For each admitted query, a persistent CUDA CTA:

1. Centers and applies the same deterministic signed Hadamard transform used to
   build RaBitQ entries, then creates byte lookup tables.
2. Seeds a bounded beam from shard-balanced anchor ordinals, with deterministic
   hash sampling only as a fallback.
3. Selects up to four frontier nodes and requests their complete compact graph
   records concurrently.
4. Reads records through a four-way, set-associative 512-byte GPU cache. Loading
   entries are deduplicated, readers pin a cache line against replacement, and
   checksum failures fail the query instead of silently dropping edges.
5. Decodes each five-byte remote pointer into a base ordinal or sparse dynamic
   handle and evaluates its full RaBitQ entry.
6. Probes the nearest dynamic anchor buckets; small deltas are scanned exactly.
7. RDMA-reads `id + generation + exact vector` for base finalists. Dynamic
   finalists use their resident exact vector. Exact L2 reranking publishes the
   external IDs to the completion ring.

Every expanded live node uses its complete adjacency list. V4 has no
`gpu_cold_expansions` cutoff and never substitutes a fixed resident hot-edge
prefix for the authoritative graph.

## Dynamic Updates

Storage-owner mutations are published with a record-level epoch. V4 stores
history in a bounded GPU delta and replaces the old `8 * N` override array with
two sparse open-addressing tables:

- base ordinal to first overriding epoch;
- dynamic remote pointer to delta slot.

Dynamic candidates are grouped by the nearest storage-shard anchor, avoiding an
`O(delta_count)` scan once the live delta is large. Publication orders vector,
RaBitQ entry, remote mapping, bucket link, override entry, and finally the
visible count. Superseded records remain visible to older admitted snapshots.

Mutation responses consume the storage invalidation payload and advance the
adjacency-cache generation. A short TTL bounds staleness from asynchronous
reverse-edge maintenance that completes after the foreground response.
Compaction drains active queries and rewrites only the live delta in place; it
does not pretend to merge data into the immutable base code stream. A future
storage compaction can drain queries and restream regenerated code ranges
without allocating a second 24 GiB array.

## RaBitQ Correctness Scope

The V4 entry format is sufficient to reproduce this implementation's asymmetric
RaBitQ estimator: it stores every sign bit, the centered-vector norm, and the
per-vector correlation correction as full-precision floats. Padded code widths
are normalized by the converter, the complete manifest and optional code
payloads are checksummed, and exact L2 uses the authoritative vector before an
ID is returned.

This guarantees format and estimator consistency, not perfect ANN recall.
Recall still depends on graph quality, beam width, entry points, anchor probes,
and the quality of the one-round deterministic signed Hadamard quantizer. The
current rotator is weaker than stronger multi-round randomized RaBitQ variants.
Use `GPU_SIDECAR_RABITQ_SOURCE=nodes` and measure candidate coverage plus final
recall before making a paper claim.

## Running SIFT100M

```bash
./experiment/convert_sift100m_gpu_sidecars.sh 04_gpu_persistent_gpunetio
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

Check local DOCA and GPU-memory registration first:

```bash
./build/dvstor_gpunetio_probe 1 mlx5_0
```

The probe validates local capability. A connected storage node is still needed
to validate QP exchange, bootstrap streaming, and end-to-end GPUNetIO reads.
