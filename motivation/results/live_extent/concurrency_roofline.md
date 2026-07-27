# Live-extent concurrency byte roofline

Measured one-shot 8-edge extent payload ratio: **50.54%** of the fixed 832-byte graph record.

| concurrency | current QPS | current total GB/s | graph byte share | extent bytes/query | byte reduction | bandwidth-only QPS upper bound | upper-bound gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 326.36 | 0.060 | 89.36% | 101996.9 | 44.20% | 584.83 | 79.20% |
| 8 | 2522.32 | 0.454 | 89.20% | 100680.9 | 44.12% | 4513.80 | 78.95% |
| 64 | 18227.63 | 3.323 | 89.33% | 101763.1 | 44.18% | 32655.88 | 79.16% |
| 256 | 58680.86 | 10.671 | 89.30% | 101529.7 | 44.17% | 105104.99 | 79.11% |

The last two columns are a strict byte-proportional roofline: each run's observed RDMA byte rate is held fixed and every other cost is assigned zero marginal penalty. They are **not performance predictions**, do not establish NIC saturation, and must not be reported as expected QPS.
