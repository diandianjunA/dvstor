# Hardware-Native Stable-Run Beam Construction

## Status

This experiment replaces the full approximate-Beam rebuild with an exact
incremental construction path.  It does not change parent selection, expansion
budget, graph reads, visited semantics, PQ scoring, reranking, or result
publication.  The new path is opt-in:

```text
--gpu-query-beam-merge-policy=legacy|stable-run
```

The bare binary keeps `legacy` as its compatibility default, and the baseline
profile uses it. The default production experiment profile selects
`stable-run`.

## Observation and actual bottleneck

At fixed graph prefetch depth 16 and concurrency 256, the original query path
spent about 2.40 ms/query in Beam merge and about 0.79 ms/query waiting for
graph RDMA.  Beam construction was therefore the largest measured phase.

The old authoritative Beam is already in stable distance order, but the legacy
wide path inserts it into a new full radix sort together with every newly
scored candidate.  The compact path sorts two large intermediate runs and then
sorts their retained prefixes again.  Both paths repeatedly reconstruct order
which the previous round already established.

The performance problem is consequently not only "too many merge calls".  It
is also that each call performs work proportional to the full merge workspace
instead of only the unsorted delta.

## Exact stable-run algorithm

The implementation preserves the project's existing comparator:

```text
distance ascending
stable input order for equal distance
old Beam input before new candidates
```

It performs:

1. Keep the old authoritative Beam as an already sorted run.
2. Split only the new candidate array into hardware-native fixed-capacity
   runs.
3. Stable radix-sort each new run and retain at most Beam capacity `K` from
   each run.
4. Use block-parallel co-rank to stable-merge the old run and candidate runs.
5. Materialize only the exact global top `K`, restoring expanded/origin
   metadata in the same pass.

Dropping a run's item after its local top `K` is exact: that item already has
`K` items from the same run which are no worse and no later in stable order, so
it cannot enter the global top `K`.

For two sorted inputs `A` and `B`, where `A` wins equal-distance ties, the
co-rank partition for output diagonal `d` satisfies:

```text
i + j = d
A[i - 1] <= B[j]
B[j - 1] <  A[i]
```

Every CTA thread computes one or a small stride of output ranks.  This removes
the prototype's single-thread `O(K^2)` candidate-to-old lookup and serial
three-way materialization.

### Register-aware run sizing

The final compact path intentionally uses four 512-item stable sorts instead
of two 1024-item sorts:

```text
128-thread CTA: 128 threads * 4 items/thread = 512 items/run
256-thread CTA: 256 threads * 4 items/thread = 1024 items/run
```

These are derived from the actual CTA width and a low-register natural
items/thread tile; they are not dataset or latency knobs.  The 128-thread
path folds two candidate runs at a time while preserving a copy of the
original old-handle metadata in the existing `2*K` workspace.

The first two-run prototype was faster per merge but raised the linked
persistent kernel to 244 registers/thread, reducing occupancy from three to
two blocks/SM. Separating the radix helper and using 512-item compact runs
restored three blocks/SM. The current cleaned production kernel uses 142
registers/thread.

## Correctness

The stable-run path is exact with respect to the legacy approximate merge.
The GPU test matrix covers:

- 128- and 256-thread CTAs;
- Beam capacity 128 and 256;
- new candidates at the first, middle, and truncation positions;
- equal distances and candidate input order;
- `-0.0` / `+0.0` stable equivalence;
- invalid handles, `NaN`, infinities, and `FLT_MAX`;
- expanded old nodes and candidate handles equal to old handles;
- the 512/1024 run boundaries, including candidates at original indices
  1023 and 1024 entering the final Beam;
- full and non-full Beams.

Both policies are compared against the same CPU stable-merge reference for
handles, numeric distance values (with the existing signed-zero equivalence),
expanded flags, and valid count. The production visited path makes old/new
duplicate handles impossible, but the direct merge API retains the legacy
duplicate-old behavior.

Validation:

```text
cmake --build build -j8                                      PASS
ctest --test-dir build --output-on-failure -j8               PASS
gpu_beam_merge_equivalence_test on GPU 1                    PASS
gpu_compact_beam_merge_test on GPU 1                        PASS
gpu_stable_run_merge_microbench on GPU 1                    PASS
```

No dynamic allocation, second Beam, extra graph read, or CPU/GPU control
round-trip is introduced.

## Kernel resources

Final linked persistent kernel:

```text
registers/thread:                   142
static shared memory:               41,746 bytes
entry-kernel spill stores/loads:    0 / 0 bytes
selected CTA:                       128 threads
active blocks/SM:                   3
query CTAs:                         256
```

`-Xptxas=-v` also exposes call-local traffic which `cuobjdump`'s entry-level
`LOCAL=0` summary does not: each stable radix helper reports 60-byte spill
stores and loads, and the stable-run coordinator reports 156-byte spill stores
and loads.  This is a real residual cost, not hidden in the report.  The
occupancy-restored implementation still wins end to end, but eliminating that
call-local traffic is a valid follow-up optimization.

## GPU merge microbenchmark

Warmup 8 iterations followed by 64 measured iterations on GPU 1:

| CTA / Beam / candidates | legacy cycles | stable-run cycles | reduction |
|---|---:|---:|---:|
| 128 / 128 / 512 | 292,488 | 57,077 | 80.5% |
| 128 / 128 / 1536 | 316,999 | 97,461 | 69.3% |
| 256 / 128 / 512 | 157,697 | 43,193 | 72.6% |
| 256 / 128 / 1536 | 155,383 | 56,207 | 63.8% |

The compact path accepts a modestly higher per-merge cost than the earlier
two-1024-run prototype in exchange for restoring an additional resident query
CTA per SM.

## End-to-end result

SIFT100M, fixed prefetch depth 16, concurrency 256, identical build and
configuration except for Beam merge policy:

| metric | legacy | stable-run | change |
|---|---:|---:|---:|
| QPS | 48,846 | 58,681 | **+20.14%** |
| mean latency | 5.234 ms | 4.357 ms | **-16.75%** |
| P50 latency | 5.192 ms | 4.329 ms | -16.61% |
| P95 latency | 5.958 ms | 5.004 ms | -16.01% |
| P99 latency | 6.434 ms | 5.370 ms | **-16.54%** |
| P999 latency | 7.261 ms | 5.970 ms | -17.78% |
| Recall@10 | 0.935 | 0.935 | unchanged |
| graph reads/query | 195.54 | 195.19 | -0.18% |
| graph rounds/query | 14.626 | 14.614 | -0.08% |
| Beam merge/query | 2.403 ms | 1.089 ms | **-54.69%** |

The timed performance workload is single-pass and the faster policy consumes a
longer prefix of the query file, so the sub-percent graph-work differences are
query-mix differences rather than evidence of altered merge semantics.  The
GPU exact-reference tests establish per-merge equality; both recall passes use
the same fixed query set.  A per-query top-k/expansion-sequence hash is not yet
reported, so full-search bitwise equality is not claimed solely from these
aggregate reports.

At the higher achieved throughput, RDMA wait/query rises from about 0.79 ms to
1.03 ms.  This is a useful bottleneck shift: Beam work was removed, the system
issued more queries per second, and pressure moved toward the network while
overall GPU query time and wall latency still fell.

### Concurrency robustness

The same final binary was also compared at concurrency 1, 8, and 64.  These
rows use 2 s warmup plus 5 s measurement; all fixed Recall passes returned
0.935.

| concurrency | QPS change | mean latency change | P99 change | Beam merge change |
|---:|---:|---:|---:|---:|
| 1 | +45.07% | -31.06% | -32.69% | -56.41% |
| 8 | +40.91% | -29.04% | -28.96% | -55.10% |
| 64 | +35.11% | -25.98% | -26.86% | -55.58% |
| 256 | +20.14% | -16.75% | -16.54% | -54.69% |

The short timed concurrency-1 pair completed different query prefixes and its
aggregate graph reads/query rose by 1.073% (1.062% symmetric difference).  It
is retained under
`motivation/results/beam_merge_final/time_c1`, not silently discarded.  A
second count-mode pair ran the identical 100 warmup and 1,000 measured queries
on both policies.  It produced exactly:

```text
graph reads/query:  197.215 vs 197.215
graph rounds/query: 14.723 vs 14.723
Recall@10:           0.935 vs 0.935
```

The `time` result root (concurrency 8/64/256) and `fixed` result root
(concurrency 1) therefore both pass the analyzer's graph-read, round, and
Recall guards without relaxing their tolerances.

### Build provenance and measurement limits

All final rows were produced by one frozen benchmark/kernel build.  The
pre-run manifest records SHA-256 for the benchmark, GPU kernel library, kernel
sources, runtime configuration, and scripts:

```text
motivation/results/beam_merge_final/BUILD_PROVENANCE.txt
```

Every listed hash was checked after both the timed matrix and fixed-query run
and remained unchanged.  The final result directories are:

```text
motivation/results/beam_merge_final/time
motivation/results/beam_merge_final/time_c1
motivation/results/beam_merge_final/fixed
```

This is still a single policy-major A/B repetition, not a publication-grade
confidence interval.  The phase counters `prepare/sort/materialize` are
diagnostic approximations because their thread-0 timestamps do not put a
dedicated timing barrier around every CUB phase; total Beam merge and
end-to-end wall measurements are the authoritative performance values.

## Reproduction

```bash
# Timed throughput/latency matrix.
BEAM_MERGE_RESULT_ROOT="$PWD/motivation/results/beam_merge_final/time" \
CONCURRENCIES="8 64 256" WARMUP_SECONDS=2 MEASURE_SECONDS=5 \
RECALL_QUERIES=100 ./motivation/run_beam_merge_ab.sh

# Identical fixed query sequence for exact work-count validation.
BEAM_MERGE_RESULT_ROOT="$PWD/motivation/results/beam_merge_final/fixed" \
CONCURRENCIES=1 WARMUP_SECONDS=0 MEASURE_SECONDS=0 \
RECALL_QUERIES=100 ./motivation/run_beam_merge_ab.sh

./motivation/analyze_beam_merge_ab.py \
  motivation/results/beam_merge_final/time
./motivation/analyze_beam_merge_ab.py \
  motivation/results/beam_merge_final/fixed
```

The analyzer pairs legacy/stable reports, prints raw values and deltas, and
fails if Recall, graph reads/query, or graph rounds/query diverge beyond their
declared tolerances.  `BEAM_MERGE_RESULT_ROOT=<path>` can be used to isolate
longer repetitions or diagnostic runs without mixing them into the primary
result set.

## Research interpretation and scope

Stable-run is a substantial system optimization, but its individual
primitives (sorted-run reuse, stable radix delta sorting, local top-K
truncation, and co-rank merge) are established GPU techniques.  The stronger
research contribution is the observation that authoritative Beam
reconstruction, rather than RDMA, was the dominant barrier in the original
GPU-centric remote traversal.  Stable-run removes that measured cost without
changing the authoritative Beam, expansion order, graph reads, or Recall.

Before claiming hardware- or dataset-independent performance, the mechanism
still needs multiple AB/BA repetitions, other graph degrees and Beam widths,
at least two structurally different vector datasets, and a second GPU/NIC
configuration.  The implementation has no dataset-tuned threshold, but the
current measured claim is deliberately limited to the reported A800/SIFT100M
environment.
