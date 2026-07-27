# GPU graph-read batch barrier motivation

- Trace: `/home/xjs/experiment/dvstor/motivation/results/batch_barrier/trace/depth_16/concurrency_256/repeat_1/rdma_trace.jsonl`
- Integrity suitable for a headline result: **yes**
- Traced queries: 234
- Complete primary attempts: 3408
- Multi-shard primary attempts: 1184 (34.7%)

## Observable barrier evidence

| Metric | Result |
|---|---:|
| Completion spread P50 (eligible multi-shard) | 15.36 us |
| Completion spread P90 (eligible multi-shard) | 69.63 us |
| Strict post-issue wait spread P50 | 15.36 us |
| Strict post-issue wait spread P90 | 69.63 us |
| Parent-weighted strict barrier waste | 7.3% |
| Parents observable before tail completion | 15.8% |
| Parents ready >=10 us before tail | 10.2% |
| Query overlap upper bound / RDMA wait P50 | 9.5% |
| Query overlap upper bound / GPU time P50 | 2.1% |

The overlap values are upper bounds, not measured speedups. Parent-weighted waste has units parent·time. A single-shard attempt is unobservable at finer granularity and is excluded from spread percentiles rather than counted as zero.

## Integrity counters

```json
{
  "metadata_schema": 2,
  "observed_timestamp_quantum_ns": 1024,
  "route_attempt_present": true,
  "wait_phase_start_present": true,
  "query_records": 234,
  "shard_batch_events": 4769,
  "round_attempt_groups": 3408,
  "complete_round_attempt_groups": 3408,
  "incomplete_round_attempt_groups": 0,
  "invalid_timestamp_events": 0,
  "duplicate_target_shard_events": 0,
  "inconsistent_process_start_groups": 0,
  "inconsistent_wait_start_groups": 0,
  "missing_query_record_groups": 0,
  "trace_overflow_queries": 0,
  "failed_queries": 0,
  "query_event_count_mismatches": 0,
  "query_graph_round_count_mismatches": 0,
  "query_graph_batch_count_mismatches": 0,
  "query_graph_read_count_mismatches": 0,
  "queries_with_route_retry": 0
}
```

## Decision rule

Proceed to a shard-batch out-of-order execution prototype only if the trace is clean, multi-shard coverage is substantial, strict wait spread is material relative to query RDMA wait, and at least one natural parent tile is commonly ready before the tail. Otherwise this trace is negative or inconclusive evidence; it must not be presented as proof of a parent-level barrier.
