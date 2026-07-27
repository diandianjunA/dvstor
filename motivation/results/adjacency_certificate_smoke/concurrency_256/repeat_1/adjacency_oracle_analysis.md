# Query-selective adjacency transfer motivation

Verdict: **STOP_NO_LARGE_BENEFIT_EVIDENCE**

This is a fail-fast observation test. The perfect oracle includes every edge (including visited rejects); post-visited/final-Beam counters are diagnostic and are not credited as pre-transfer savings.

## Sample

- Queries with trace records: 152
- Score chunks: 2233
- Parents / edges: 29793 / 1395491
- Beam-not-full chunks: 152 (6.81%)
- Trace overflow: 0

## Edge usefulness funnel

| decoded | invalid | dynamic | visited survivors | finite scored | entered Beam |
|---:|---:|---:|---:|---:|---:|
| 1395491 | 0 | 0 | 1228537 (88.04%) | 1228537 (88.04%) | 112361 (8.05%) |

The current fixed 832-byte record also contains substantial unused degree capacity. Packing only the live edges would save **53.04%** of sampled graph payload before any query-dependent certificate. This is reported separately as a layout effect.

## Prefix + certified suffix (recommended one-sided layout)

| prefix | perfect live-edge skip | suffix live-edge skip | retain perfect | tail-free parents | WQE/parent | chunks needing stage 2 | certificate-only query gain upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7.71% | 0.00% | 0.00% | 0.09% | 1.999 | 100.00% | 0.00% |
| 16 | 10.59% | 0.00% | 0.00% | 2.39% | 1.976 | 99.78% | 0.00% |
| 32 | 8.71% | 0.00% | 0.00% | 28.41% | 1.716 | 96.51% | 0.00% |
| 48 | 4.50% | 0.00% | 0.00% | 60.43% | 1.396 | 82.94% | 0.00% |
| 64 | 1.72% | 0.00% | 0.00% | 80.49% | 1.195 | 62.92% | 0.00% |

## Arbitrary groups (diagnostic; requires remote runs)

| group | perfect skip | interval skip | retain perfect | interval certificate bytes saved | coalesced WQE/parent |
|---:|---:|---:|---:|---:|---:|
| 4 | 48.18% | 0.00% | 0.00% | 0.00% | 2.000 |
| 8 | 38.13% | 0.00% | 0.00% | 0.00% | 2.000 |
| 16 | 29.66% | 0.00% | 0.00% | 0.00% | 2.000 |
| 32 | 21.82% | 0.00% | 0.00% | 0.00% | 2.000 |

## Safety and gate

- Geometric lower-bound violations: 0
- Minimum measured safety margin: 0.0000000
- Best observed prefix: 16
- Failed gates: perfect_live_edge_skip_ge_70pct, suffix_retains_ge_80pct_perfect_skip, tail_free_parents_ge_92pct, certificate_only_query_gain_ge_20pct, 20pct_qps_wqe_rate_le_1p10x_observed

The model does not charge the second dependent RDMA RTT, the new version/checksum layout, or certificate evaluation. Therefore any reported query-time gain is an optimistic upper bound.
