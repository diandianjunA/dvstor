# Ordered-commit motivation matrix

The zero-overhead column is an oracle upper bound. It already assumes that movable validation/decode/PQ/visited work can execute at each observable release without queueing, state-transfer, or scheduling cost. It is not a projected speedup.

| depth | concurrency | reps | queries | release coverage | strict spread P50/P90 | tile ready +10us | zero-overhead oracle / GPU P50 | 2us/tile oracle / GPU P50 | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 1 | 50 | 0.00% | n/a/n/a us | 0.00% | 0.00% | 0.00% | negative at shard granularity |
| 8 | 256 | 1 | 212 | 29.59% | 12.29/34.82 us | 7.96% | 0.71% | 0.36% | negative at shard granularity |
| 16 | 1 | 1 | 136 | 33.00% | 3.07/5.12 us | 0.15% | 0.41% | 0.00% | negative at shard granularity |
| 16 | 256 | 1 | 234 | 33.89% | 16.38/70.66 us | 11.74% | 1.08% | 0.27% | negative at shard granularity |
| 32 | 256 | 1 | 186 | 36.54% | 37.89/157.90 us | 18.20% | 1.21% | 0.25% | negative at shard granularity |

## Decision rule

Proceed only with clean integrity, >=25% rounds carrying multiple release boundaries, strict spread P50 >=10 us or P90 >=25 us, and a zero-overhead release-time oracle >=8% of query GPU residence at P50. A shard-level failure stops a shard-only design. It does not fabricate a conclusion about unobserved parent completion.
