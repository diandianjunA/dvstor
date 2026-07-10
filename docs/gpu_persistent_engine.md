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
- `<prefix>_nodeX_ofN.dat`: authoritative fixed nodes, exact vectors, and the
  compact 488-byte Vamana graph records for `R=96`.
- `<prefix>.anchors`: small routing anchors used for entry routing and dynamic
  candidate buckets.

`<prefix>.gpu.idx` is an optional 2 KiB export/debug cache. The runtime never
requires or reads it: at startup it synthesizes the same control view in memory
from `.meta.json` and anchor pointers.

`<prefix>_nodeX_ofN.gpu.codes` is optional. If present on a storage node, its
header, dimensions, offset, and checksum are validated before loading. If it is
absent, the storage node sequentially materializes the same contiguous RaBitQ
stream from the authoritative entries already present in its local `.dat`
shard. The stream occupies a reserved range immediately after the immutable
static image, and the dynamic allocator starts after that range.

The compute node needs only metadata and anchors on local or shared storage. It
does not need `.gpu.idx`, `.dat`, `.gpu.codes`, `.idmap`, `.rabitq12`, or any
graph snapshot locally in a distributed run. Storage nodes keep their own
`.dat`; storage-owner updates additionally use their local `.idmap` and anchor
files.

V3 `.gpu.idx` and `.gpu.pages` files are rejected with a migration error. Old
`.gpu.pages` files are never loaded. After validating V4, they can be removed
manually; the converter never deletes old files.

## Converting an Existing Index

The converter reuses the existing Vamana/Metis placement and never rebuilds the
graph. The converter can still export a manifest for inspection, but it is not
part of the runtime dependency set:

```bash
GPU_SIDECAR_OVERWRITE=1 \
GPU_SIDECAR_RABITQ_SOURCE=nodes \
./experiment/convert_sift100m_gpu_sidecars.sh 04_gpu_persistent_gpunetio
```

In manifest-only mode no `.gpu.codes` file is created on the compute node. The
exported `.gpu.idx` may also be deleted without affecting query startup. Each
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
3. The compute node synthesizes ordinal ranges, remote offsets, centroid, and
   entry points from metadata/anchors, then allocates the final GPU layout once.
4. CPU-posted GPUDirect RDMA windows stream each shard directly into its final
   ordinal range. The CPU submits startup bulk transfers but never copies their
   payload; there is no host staging array or compute-side code snapshot file.
5. The persistent GPU kernel starts only after every streamed byte succeeds.

The storage stream consumes 24 bytes per SIFT vector across the storage nodes;
it consumes no compute-node disk space and is not mirrored in compute host
memory. The query backend can be `gpunetio`, `verbs_proxy`, or `local`. GPUNetIO
owns the steady-state query QPs. Bootstrap and automatic failover use one
equally sized pool of dedicated CPU-posted GPUDirect Verbs QPs rather than
sharing a control QP. A bounded GPU CQ poll disables the direct path after an
error or timeout, after which graph and exact-vector reads continue through the
fallback pool without copying payloads through host memory.
The benchmark JSON records the lifetime counter
`gpu_persistent.direct_path_failures`; a GPUNetIO performance result is valid
only when this value is zero.

Multi-gigabyte GPUNetIO allocations are registered through `nvidia-peermem`
with `ibv_reg_mr`, matching NVIDIA's Verbs GPUNetIO samples. The runtime falls
back to a DMA-BUF MR when peer-memory registration is unavailable. On the
current ConnectX-6/535 driver stack, a single DMA-BUF MR succeeds at 2 GiB but
fails at 4 GiB, while peer-memory registration has been validated at the full
36 GiB engine budget.

## Query Execution

Base nodes use a 30-bit ordinal handle. Dynamic nodes use a tagged delta handle.
Shard range metadata converts a base ordinal to its fixed-node and compact-graph
RDMA addresses without a dense node table.

Each persistent CUDA CTA owns at most one query descriptor. Thread 0 claims the
descriptor, publishes it through shared memory, and the entire CTA enters the
barrier-heavy search routine together. GPU-owned ring cursors reside in device
memory; only descriptors and publication sequences use mapped host memory.

For each admitted query, the CTA:

1. Centers and applies the same deterministic signed Hadamard transform used to
   build RaBitQ entries, then creates byte lookup tables.
2. RaBitQ-ranks shard-balanced entry ordinals, RDMA-fetches the selected exact
   vectors, and seeds the beam only with exact L2 distances.
3. Selects up to four frontier nodes and requests their complete compact graph
   records concurrently.
4. Reads records through a four-way, set-associative 512-byte GPU cache. Loading
   entries are deduplicated, readers pin a cache line against replacement, and
   checksum failures fail the query instead of silently dropping edges.
5. Decodes each five-byte remote pointer into a base ordinal or sparse dynamic
   handle and evaluates its full RaBitQ entry only to form a candidate gate.
6. During warmup and periodic audit rounds it exactifies the complete frontier;
   otherwise it exactifies the configured RaBitQ gate, including the uncertainty
   margin. Only exact L2 distances are inserted into and used to order the beam.
7. Probes the nearest dynamic anchor buckets and exactifies the selected resident
   delta vectors before beam insertion.
8. Carries the ID and exact distance obtained by the candidate fetch alongside
   each beam handle, so final sorting performs no duplicate RDMA read. Dynamic
   finalists use their resident ID and exact vector in the same beam format.

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

Mutation responses consume the storage invalidation payload and invalidate only
the matching cache sets after their active readers drain; foreground writes no
longer discard the entire multi-gigabyte adjacency cache. A short TTL bounds
staleness from asynchronous reverse-edge maintenance that completes after the
foreground response.
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

This guarantees format and estimator consistency, not perfect ANN recall. RaBitQ
is never treated as an exact metric and never directly orders the traversal
beam: it may only exclude candidates before authoritative exact-vector reads.
Warmup and audit rounds bound sustained navigation drift, but recall still
depends on graph quality, beam width, entry points, gate width, and anchor probes.
The current one-round deterministic rotator is weaker than stronger randomized
RaBitQ variants, so candidate coverage and final recall must both be reported.

## Running SIFT100M

```bash
./experiment/convert_sift100m_gpu_sidecars.sh 04_gpu_persistent_gpunetio
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

Restart every storage process after upgrading this engine. The GPUNetIO profile
reserves one configured QP pool for GPU-issued reads and a second equal pool for
bounded Verbs failover; an old storage process waits for a different QP count
and cannot complete the startup handshake.

Check local DOCA and GPU-memory registration first:

```bash
./build/dvstor_gpunetio_probe 1 mlx5_0

# Exercise the large peer-memory path used by SIFT100M.
./build/dvstor_gpunetio_probe 1 mlx5_0 12185894912 12185894912 peer
```

The probe validates local capability. A connected storage node is still needed
to validate QP exchange, bootstrap streaming, and end-to-end GPUNetIO reads.
