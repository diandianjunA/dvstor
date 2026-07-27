# Feedback-Priced Expansion Motivation

- sampled queries: 462
- sampled rounds: 6764
- top-16 Beam turnover mean/P50/P90: 0.735 / 1.000 / 1.000

## Candidate fate by origin and turnover

| origin | turnover | rank | eventually selected | samples |
|---|---:|---:|---:|---:|
| new | [.25,.50) | 0-15 | 100.000% | 4583 |
| new | [.25,.50) | 16-31 | 92.044% | 1194 |
| new | [.50,.75) | 0-15 | 100.000% | 7126 |
| new | [.50,.75) | 16-31 | 94.283% | 1994 |
| new | [.75,1] | 0-15 | 100.000% | 55913 |
| new | [.75,1] | 16-31 | 23.131% | 46669 |
| new | [0,.25) | 0-15 | 100.000% | 817 |
| new | [0,.25) | 16-31 | 86.826% | 334 |
| old | [.25,.50) | 0-15 | 100.000% | 8662 |
| old | [.25,.50) | 16-31 | 87.343% | 9157 |
| old | [.50,.75) | 0-15 | 100.000% | 5016 |
| old | [.50,.75) | 16-31 | 88.429% | 8210 |
| old | [.75,1] | 0-15 | 100.000% | 1283 |
| old | [.75,1] | 16-31 | 86.785% | 7900 |
| old | [0,.25) | 0-15 | 100.000% | 5245 |
| old | [0,.25) | 16-31 | 87.971% | 3691 |

## Selected-parent immediate productivity

| prior origin | prior turnover | productive | samples |
|---|---:|---:|---:|
| new | [.25,.50) | 36.548% | 4583 |
| new | [.50,.75) | 48.737% | 7126 |
| new | [.75,1] | 88.241% | 55913 |
| new | [0,.25) | 25.826% | 817 |
| old | [.25,.50) | 28.873% | 8662 |
| old | [.50,.75) | 37.640% | 5016 |
| old | [.75,1] | 49.961% | 1283 |
| old | [0,.25) | 18.627% | 5245 |
| startup | startup | 100.000% | 1848 |

## Productive-suffix oracle

- removable selected-parent suffix: 5.323%
- modeled round-work reduction after charging extra merge toll: -6.851%
> Optimistic upper bound: omitted parents are known not to change the immediate Beam; it does not replay changed visited or future traversal state, and it does not charge extra RDMA wait for smaller batches.

## Projection on the uninstrumented C16 baseline

- controlled GPU fraction: 85.528%
- projected end-to-end change: -5.859%
- projected QPS: 55432.9 (baseline 58680.9)

Gate: if this deliberately optimistic production projection is below 15%, adaptive batching does not have enough headroom to justify another controller.
