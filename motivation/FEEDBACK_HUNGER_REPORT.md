# Feedback-Horizon and Hardware-Hunger Expansion

## Status

The mechanism is implemented, but it is **not enabled by default**.  The
fixed-depth path remains unchanged and is the A/B baseline.  The dynamic path
is intentionally exposed as an experiment because the first end-to-end result
does not meet the performance target.

The implemented target is:

```text
B = min(remaining_budget,
        efficient_batch_cap,
        feedback_horizon + accepted_hardware_tiles * natural_tile)
```

`feedback_horizon` is produced inside the existing wide/compact merge.  The
hardware term is granted by the owner of the actual QP only during a
queue-empty, progress-balanced idle episode, and is consumed through a
per-QP, epoch-tagged WQE lease.  Ring-full and SQ-capacity defer revoke that
lease.  No speculative read, second beam, second sort, extra graph read, or
CPU control loop was added.

## Actual call path

`process_query()` still selects the authoritative Beam prefix, marks only the
selected parents expanded, calls `fetch_graph_records_batch()`, waits for the
existing per-shard descriptors, validates records, decodes neighbors, runs
visited/PQ, and invokes the existing wide or compact merge.  A dynamic
selection never exceeds
`persistent_score_chunk_capacity(graph_entry_capacity, traversal_capacity)`,
so it remains one decode/score/merge chunk.

During merge, the old-handle lookup now carries two bits:

```text
bit 0: restored expanded
bit 1: new candidate
```

The existing valid-prefix scan computes the earliest new output, old
unexpanded candidates before it, and the number of new candidates.  It then
writes only bit 0 back to the authoritative `beam_expanded` array.  There is
no independent post-merge Beam scan.

The owner-side lease is device memory, cache-line aligned, and indexed by the
same `(QP lane * remote_region_count + shard)` mapping used by
`direct_fetch_batch()`.  Query CTAs perform relaxed loads and nonblocking
claims; they never wait for a lease and stop at the first failed natural tile.
Claims are retained as round tokens until graph fetch reports which QP
descriptors reached publication.  Only unissued tokens are returned;
published tokens are consumed/revoked by the owner, avoiding a double-return
into an active SQ.
The lease is a performance permission only: Beam ordering, visited, expanded
flags, expansion budget, and termination remain authoritative.

## Resource and correctness checks

```text
cmake --build build -j8                                      PASS
ctest --test-dir build --output-on-failure -j8               PASS
gpu_feedback_horizon_test on GPU 1                          PASS
gpu_expansion_pressure_test on GPU 1                        PASS
gpu_compact_beam_merge_test on GPU 1                        PASS
```

The CUDA unit tests are skipped by CTest in a non-GPU environment, so the
three commands above are the explicit GPU runs.  The linked kernel resource
inspection reports:

```text
process_query: 254 registers/thread, LOCAL=0
persistent entry: 130 registers/thread, 116B static shared
entry spill stores/loads: 0/0
```

The start path resets pressure and per-QP lease state on every persistent
kernel launch, preventing stale active-query counts or offers after restart.
The query-side cap and all existing remote-record validation paths remain
unchanged.

## Reproduction

The complete matrix script is:

```bash
./motivation/run_feedback_hunger_ab.sh
```

It compares fixed depths `1 8 16 32` and the dynamic policy at client
concurrency `1 8 64 256`.  The dynamic configuration is:

```text
GPU_QUERY_EXPANSION_POLICY=feedback-horizon-hunger
GPU_GRAPH_PREFETCH_DEPTH=16   # ignored by the dynamic path
```

For a short A/B run after the five storage nodes are started:

```bash
WORKLOAD=query WARMUP_SECONDS=2 MEASURE_SECONDS=5 \
CLIENT_THREADS=256 RECALL_QUERIES=32 \
MIN_RECALL=-1 MIN_QUERY_QPS=-1 MIN_STABILITY_RATIO=-1 \
GPU_QUERY_EXPANSION_POLICY=fixed GPU_GRAPH_PREFETCH_DEPTH=16 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio

WORKLOAD=query WARMUP_SECONDS=2 MEASURE_SECONDS=5 \
CLIENT_THREADS=256 RECALL_QUERIES=32 \
MIN_RECALL=-1 MIN_QUERY_QPS=-1 MIN_STABILITY_RATIO=-1 \
GPU_QUERY_EXPANSION_POLICY=feedback-horizon-hunger \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

## Current measured result

These are fresh 5-second runs at concurrency 256 on the currently attached
SIFT100M/GPUNetIO setup:

| metric | fixed, depth 16 | feedback-horizon-hunger |
|---|---:|---:|
| QPS | 48,781 | 37,101 |
| mean latency | 5.241 ms | 6.889 ms |
| P99 latency | 6.445 ms | 9.754 ms |
| Recall@10 | 0.928125 | 0.928125 |
| average selected parents/round | 13.369 | 8.986 |
| graph rounds/query | 14.63 | 20.14 |
| graph reads/query | 195.53 | 181.01 |
| RDMA wait/query | 801.6 us | 1,052.4 us |
| Beam merge/query | 2,407.8 us | 3,442.4 us |
| extra parents | 0 | 23,431,747 |
| QP lease claims/rejects | 0/0 | 7,698,386 / 1,633,878 |

After the final control-only ledger change, a 1-second warmup/2-second
confirmation measured 36.8K QPS, 6.93 ms mean latency, and 9.82 ms P99.
Its 16-query Recall@10 sample was 0.9375 (too small to replace the 32-query
0.928125 comparison above); no query, QP, or CQ failures occurred.

The dynamic policy preserved Recall and reduced graph reads, but it selected
too few parents on average.  The resulting extra rounds and per-round lease /
selection work dominated the reduction in graph reads.  The marginal cost
ledger reported no failed probes in this run, which means that its current
integer allowance is permissive but does not compensate for QP lease
contention.

Therefore the current result is a negative performance result:
`feedback-horizon-hunger` is not yet better than fixed depth 16 and must not
become the default.  This is an implementation/design finding, not a reason
to alter Recall or expansion-budget accounting.  Work on this controller is
stopped: its smaller average batch increases search rounds and repeatedly pays
the dominant Beam-maintenance cost.  The follow-up experiment instead removes
that structural cost while retaining fixed authoritative expansion semantics;
see `motivation/STABLE_RUN_BEAM_REPORT.md`.

The final revision also removes per-claim/reject/return global counter atomics
from the query hot path.  Query-local completion counters still report claims
and rejects; device lease counters are reserved for owner-episode telemetry.
