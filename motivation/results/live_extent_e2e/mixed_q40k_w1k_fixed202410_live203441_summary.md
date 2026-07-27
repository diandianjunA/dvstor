# Strict Live-Extent mixed-update paired summary

This is a strict causal pair. The only controlled algorithm variable is `gpu_query_graph_read_policy` (`fixed` vs `live-extent`).

## Contract audit

- Status: **PASS**; control-field mismatches: **0**.
- Workload: `mixed/rate_limited`, 40,000 query/s + 1,000 write/s, auto-derived 336 clients, 30 s warmup + 120 s measurement.
- Search: fixed C16, stable-run, beam 128, max-expansions 384, rerank 128.
- Both runs: warmup 30,000 writes; measurement 120,000 writes and 4,799,998 queries; the exact same maintenance target sequences reached their durable watermarks.
- Recall: 1,000 queries before and after; all result sets complete.
- Failures: zero direct-path failures, route timeouts, Stage2 failures, and late storage RPC completions.

Raw reports:

- fixed: `/home/xjs/experiment/dvstor/motivation/results/live_extent_e2e/fixed_mixed_q40k_w1k/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_202410.json`
- live-extent: `/home/xjs/experiment/dvstor/motivation/results/live_extent_e2e/live_mixed_q40k_w1k_repeat2/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_203441.json`

## Key paired metrics

| metric | fixed | live-extent | live-fixed | change |
|---|---:|---:|---:|---:|
| query_ops | 4799998.000000 | 4799998.000000 | 0.000000 | +0.0000% |
| write_ops | 120000.000000 | 120000.000000 | 0.000000 | +0.0000% |
| query_qps | 39999.983333 | 39999.983333 | 0.000000 | +0.0000% |
| query_latency_mean_us | 3883.946000 | 3913.719000 | 29.773000 | +0.7666% |
| query_latency_p50_us | 3865.374000 | 3899.791000 | 34.417000 | +0.8904% |
| query_latency_p95_us | 4456.273000 | 4566.219000 | 109.946000 | +2.4672% |
| query_latency_p99_us | 4786.773000 | 4904.242000 | 117.469000 | +2.4540% |
| query_latency_p999_us | 5333.304000 | 5432.118000 | 98.814000 | +1.8528% |
| write_latency_mean_us | 4014.623000 | 3746.998000 | -267.625000 | -6.6663% |
| write_latency_p99_us | 9573.486000 | 8893.929000 | -679.557000 | -7.0983% |
| write_latency_p999_us | 18064.581000 | 15269.114000 | -2795.467000 | -15.4749% |
| gpu_query_us | 3827.571099 | 3856.878808 | 29.307709 | +0.7657% |
| gpu_rdma_issue_us | 133.002705 | 167.669082 | 34.666377 | +26.0644% |
| gpu_rdma_wait_us | 727.298658 | 783.001413 | 55.702755 | +7.6589% |
| gpu_graph_validation_us | 549.161545 | 447.479551 | -101.681994 | -18.5159% |
| gpu_exact_us | 260.059926 | 246.779136 | -13.280790 | -5.1068% |
| gpu_other_us | 171.798851 | 226.255734 | 54.456883 | +31.6980% |
| logical_graph_reads_per_query | 191.497229 | 191.490027 | -0.007202 | -0.0038% |
| graph_bytes_per_query | 159325.694319 | 84506.772238 | -74818.922081 | -46.9597% |
| graph_bytes_per_logical_parent | 832.000001 | 441.311611 | -390.688389 | -46.9577% |
| physical_graph_wqes_per_query | 191.497229 | 196.129556 | 4.632327 | +2.4190% |
| physical_graph_wqes_per_logical_parent | 1.000000 | 1.024229 | 0.024229 | +2.4229% |
| total_rdma_bytes_per_query | 178782.635847 | 103963.715896 | -74818.919951 | -41.8491% |
| rdma_wqes_per_query | 319.523382 | 324.155768 | 4.632386 | +1.4498% |
| fallback_graph_reads_per_query | 0.000000 | 4.639529 | 4.639529 | n/a |
| graph_extent_fallback_ratio | 0.000000 | 0.024767 | 0.024767 | n/a |
| recall_before | 0.940100 | 0.940100 | 0.000000 | +0.0000% |
| recall_after | 0.939700 | 0.939500 | -0.000200 | -0.0213% |
| recall_change | -0.000400 | -0.000600 | -0.000200 | n/a |
| stage2_remaining | 7.000000 | 14.000000 | 7.000000 | +100.0000% |
| stage2_max_backlog | 10.000000 | 6.000000 | -4.000000 | -40.0000% |
| stage2_backlog_slope_per_sec | -0.016666 | 0.041666 | 0.058333 | n/a |
| stage2_p99_delay_upper_ms | 64.000000 | 32.000000 | -32.000000 | -50.0000% |
| stage2_completion_outstanding | 6.000000 | 17.000000 | 11.000000 | +183.3333% |
| stage2_max_completion_outstanding_per_shard | 4.000000 | 6.000000 | 2.000000 | +50.0000% |
| stage2_failures | 0.000000 | 0.000000 | 0.000000 | n/a |
| storage_late_rpc_completions | 0.000000 | 0.000000 | 0.000000 | n/a |

## Physical-I/O judgement

- Logical graph reads: 919,186,314 → 919,151,745 (effectively unchanged).
- Physical graph WQEs: 919,186,315 → 941,421,475; **+22,235,160**.
- Live-Extent fallback/retry reads: **22,269,730** (2.4767% of short reads).
- Extra physical graph WQEs minus fallbacks: -34,570. The WQE increase is therefore explained almost exactly by fallback/revalidation, not extra logical search expansion.
- Graph bytes/query: -46.96%; total RDMA bytes/query: -41.85%.

## Conclusions

- **bytes:** PASS: graph bytes/query fall 46.96% and total RDMA bytes/query fall 41.85%.
- **query_performance:** NO WIN: target QPS is fixed by the pacer; mean/P99/P999 query latency rise 0.77%/2.45%/1.85%, GPU query time rises 0.77%, and RDMA wait rises 7.66%.
- **write_and_stage2:** IMPROVED IN THIS PAIR: mean/P99/P999 write latency fall 6.67%/7.10%/15.47%; Stage2 max backlog falls 10 to 6 and its P99 delay histogram upper bound falls 64 ms to 32 ms. Remaining and completion-outstanding counters are higher, so the result is not uniformly better.
- **recall:** Initial Recall is identical at 0.9401. Post-workload Recall is 0.9397 fixed versus 0.9395 Live-Extent, an absolute -0.0002 difference; both runs completed the registered protocol.
- **failures:** PASS: no GPU direct-path failure, route timeout, Stage2 failure, or late storage RPC; all maintenance target watermarks became durable.
- **overall:** This controlled mixed-update pair validates byte elasticity but does not validate a query-performance gain. Update-induced fallback/revalidation converts the byte saving into extra WQEs and higher query RDMA wait.
