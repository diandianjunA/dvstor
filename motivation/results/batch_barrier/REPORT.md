# Batch barrier motivation matrix

This report includes every discovered run. Trace QPS is never used as a performance result; performance rows were collected with tracing fully off.

## Mechanism trace

| depth | concurrency | reps | queries | multi-shard rounds | strict spread P50/P90 (us) | strict parent waste | ready tile +10us | upper bound / GPU time | screen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 1 | 50 | 0.0% | n/a/n/a | 0.0% | n/a | 0.0% | insufficient/negative |
| 8 | 256 | 1 | 212 | 30.5% | 11.26/34.20 | 5.0% | 26.1% | 1.0% | insufficient/negative |
| 16 | 1 | 1 | 136 | 38.6% | 2.05/5.12 | 0.8% | 0.4% | 0.4% | insufficient/negative |
| 16 | 256 | 1 | 234 | 34.7% | 15.36/69.63 | 7.3% | 33.8% | 2.1% | insufficient/negative |
| 32 | 256 | 1 | 186 | 36.9% | 37.38/155.03 | 9.1% | 49.4% | 3.6% | insufficient/negative |

The screen is preregistered and deliberately conservative: clean integrity, >=25% multi-shard primary attempts, >=5 us strict P50 spread, >=20% parent-weighted strict waste, a natural parent tile ready >=10 us early in >=50% of eligible attempts, and a per-query strict overlap upper bound >=10% of GPU residence. Failing the screen is retained as negative evidence.

## Trace-off performance controls

| depth | concurrency | repeat | QPS | mean/P99 latency (us) | Recall | GPU RDMA wait (us/query) | rounds/query |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 325.01 | 3075.76/3687.36 | 0.9281 | 490.11 | 14.66 |
| 16 | 256 | 1 | 58651.60 | 4355.48/5394.22 | 0.9281 | 1044.95 | 14.63 |

## Sampled-trace overhead sanity

| depth | concurrency | repeat | trace-off QPS | sampled QPS | delta |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 325.01 | 322.99 | -0.6% |
| 16 | 256 | 1 | 58651.60 | 58627.44 | -0.0% |

## Scope limit

A completion event is a query shard descriptor observed at its owner submission-group completion boundary. The experiment cannot see parent/WQE completion within a shard or within a shared final-CQE group. Therefore it can support only a shard-batch-granularity execute-ready/commit-in-order design. If most rounds contain one observable shard batch, the result is inconclusive at this interface rather than evidence that parent-level dispersion is absent.
