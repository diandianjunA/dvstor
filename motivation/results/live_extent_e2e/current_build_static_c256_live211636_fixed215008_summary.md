# Current-build static c256 Live-Extent A/B

This is a strict 120-second current-build pair. The two generated service
configurations differ only in `gpu-query-graph-read-policy`; the JSON control
fields also match. Both runs use 256 clients, fixed C16 expansion, stable-run
merge, Beam 128, 384 maximum expansions, rerank 128, and the same single-pass
query source.

## Result

| metric | fixed | live-extent | live vs fixed |
|---|---:|---:|---:|
| QPS | 57,912.54 | 63,065.85 | **+8.90%** |
| mean latency | 4,419.04 us | 4,057.86 us | **-8.17%** |
| P95 latency | 5,070.02 us | 4,669.35 us | **-7.90%** |
| P99 latency | 5,425.46 us | 4,997.48 us | **-7.89%** |
| P999 latency | 5,928.38 us | 5,481.37 us | **-7.54%** |
| GPU query | 4,346.08 us | 3,985.62 us | **-8.29%** |
| GPU graph stage | 1,862.70 us | 1,561.16 us | **-16.19%** |
| RDMA issue | 143.59 us | 158.22 us | +10.18% |
| RDMA wait | 1,029.97 us | 846.14 us | **-17.85%** |
| graph validation | 585.14 us | 395.99 us | **-32.32%** |
| graph bytes/query | 162,595.77 B | 81,878.40 B | **-49.64%** |
| total tracked RDMA bytes/query | 182,051.77 B | 101,334.40 B | **-44.34%** |

The QPS result is stable rather than a transient peak: fixed and live-extent
tail/head ratios are 0.99999 and 0.99934, respectively, with no zero-completion
windows.

## Search-work and correctness controls

| control | fixed | live-extent | difference |
|---|---:|---:|---:|
| Recall@10 before/after | 0.9401 / 0.9401 | 0.9401 / 0.9401 | 0 |
| logical graph reads/query | 195.4276 | 195.4297 | +0.0011% |
| selected parents/query | 195.4276 | 195.4297 | +0.0011% |
| graph rounds/query | 14.6182 | 14.6197 | +0.0100% |
| physical graph WQEs/query | 195.4276 | 195.4297 | +0.0011% |
| total tracked RDMA WQEs/query | 323.4276 | 323.4297 | +0.0006% |
| exact reads/query | 128 | 128 | 0 |
| fallback / underhint / promotion | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| direct-path / Stage2 failures | 0 / 0 | 0 / 0 | 0 |

This isolates the mechanism cleanly: Live-Extent does not reduce the number of
parents searched or requests issued. It transfers 418.97 rather than 832 bytes
per graph parent, which reduces graph-transfer and validation cost and yields
the measured throughput and latency improvement. The higher RDMA issue time
(+14.62 us/query) is paid back by 183.83 us/query less RDMA wait and 189.15
us/query less graph validation.

## Scope

This is one long paired run, not a multi-repeat confidence interval, and the
order was live-extent then fixed rather than randomized. Because the source is
single-pass and live-extent completes more work during the fixed duration, the
two runs consume different-length prefixes; work is therefore compared per
query. `rdma_read_bytes` and `rdma_read_ops` describe tracked GPU query reads,
not whole-system NIC wire traffic.
