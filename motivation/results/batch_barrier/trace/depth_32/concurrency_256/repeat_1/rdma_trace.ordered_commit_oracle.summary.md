# Execute-ready / commit-in-order motivation oracle

- Input: `/home/xjs/experiment/dvstor/motivation/results/batch_barrier/trace/depth_32/concurrency_256/repeat_1/rdma_trace.jsonl`
- Observable completion granularity: `shard_batch_owner_completion_boundary`
- Modeled task granularity: `tile`
- Verdict: **negative at observed granularity; parent/tile result remains unmeasured**

## Directly observed release dispersion

| Metric | Result |
|---|---:|
| Primary rounds | 2088 |
| Rounds with >=2 release boundaries | 36.54% |
| Strict spread P50/P90/P99 | 37.89 / 157.90 / 340.36 us |
| Natural tile ready >=10 us before tail | 18.20% |
| Completion-to-process handoff P50 | 4.10 us |

## Release-time oracle

The oracle moves only validation, decode, PQ scoring, and visited work. It leaves the authoritative Beam and all search decisions behind the epoch commit barrier.

| Queue/state overhead per task | Saved / GPU time P50 | Saved / RDMA wait P50 | Saved/query P50 |
|---:|---:|---:|---:|
| 0 us | 1.21% | 6.66% | 68.99 us |
| 1 us | 0.56% | 2.84% | 31.74 us |
| 2 us | 0.25% | 1.26% | 14.98 us |
| 5 us | 0.00% | 0.00% | 0.00 us |
| 10 us | 0.00% | 0.00% | 0.00 us |

## Preregistered screen

- [x] `integrity_clean`
- [x] `multi_release_coverage_ge_25pct`
- [x] `dispersion_ge_10us_p50_or_25us_p90`
- [ ] `zero_overhead_oracle_ge_8pct_gpu_time`

## Interpretation limit

Per-query validation+neighbor_decode+PQ+visited wall time is distributed linearly over graph parents. This is an oracle sensitivity model, not measured per-task service.

A shard-batch completion timestamp cannot establish parent-level dispersion inside that shard. A negative shard-granularity verdict therefore stops a shard-only reorder design, but does not silently stand in for a parent/tile-signaled experiment. Conversely, parent-weighted ready area is not wall-clock query time and is never reported as speedup.
