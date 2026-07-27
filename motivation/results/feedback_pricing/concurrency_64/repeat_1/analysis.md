# Feedback-Priced Expansion Motivation

- sampled queries: 101
- sampled rounds: 1478
- top-16 Beam turnover mean/P50/P90: 0.742 / 1.000 / 1.000

## Candidate fate by origin and turnover

| origin | turnover | rank | eventually selected | samples |
|---|---:|---:|---:|---:|
| new | [.25,.50) | 0-15 | 100.000% | 1039 |
| new | [.25,.50) | 16-31 | 90.588% | 255 |
| new | [.50,.75) | 0-15 | 100.000% | 1475 |
| new | [.50,.75) | 16-31 | 95.806% | 453 |
| new | [.75,1] | 0-15 | 100.000% | 12379 |
| new | [.75,1] | 16-31 | 23.002% | 10186 |
| new | [0,.25) | 0-15 | 100.000% | 164 |
| new | [0,.25) | 16-31 | 90.323% | 62 |
| old | [.25,.50) | 0-15 | 100.000% | 1912 |
| old | [.25,.50) | 16-31 | 85.446% | 2027 |
| old | [.50,.75) | 0-15 | 100.000% | 1061 |
| old | [.50,.75) | 16-31 | 88.060% | 1742 |
| old | [.75,1] | 0-15 | 100.000% | 285 |
| old | [.75,1] | 16-31 | 85.329% | 1779 |
| old | [0,.25) | 0-15 | 100.000% | 999 |
| old | [0,.25) | 16-31 | 87.481% | 671 |

## Selected-parent immediate productivity

| prior origin | prior turnover | productive | samples |
|---|---:|---:|---:|
| new | [.25,.50) | 36.285% | 1039 |
| new | [.50,.75) | 49.424% | 1475 |
| new | [.75,1] | 87.665% | 12379 |
| new | [0,.25) | 23.780% | 164 |
| old | [.25,.50) | 28.295% | 1912 |
| old | [.50,.75) | 37.795% | 1061 |
| old | [.75,1] | 47.018% | 285 |
| old | [0,.25) | 20.821% | 999 |
| startup | startup | 100.000% | 404 |

## Productive-suffix oracle

- removable selected-parent suffix: 5.244%
- modeled round-work reduction after charging extra merge toll: -6.375%
> Optimistic upper bound: omitted parents are known not to change the immediate Beam; it does not replay changed visited or future traversal state, and it does not charge extra RDMA wait for smaller batches.

## Projection on the uninstrumented C16 baseline

- controlled GPU fraction: 85.528%
- projected end-to-end change: -5.452%
- projected QPS: 55646.8 (baseline 58680.9)

Gate: if this deliberately optimistic production projection is below 15%, adaptive batching does not have enough headroom to justify another controller.
