# Strict high-water Live-Extent mixed-update paired summary

This is a strict causal pair. The only controlled algorithm variable is `gpu_query_graph_read_policy` (`fixed` vs high-water `live-extent`).

## Contract audit

- Status: **PASS**; control-field mismatches: **0**.
- Workload: `mixed/rate_limited`, 40,000 query/s + 1,000 write/s, auto-derived 336 clients, 30 s warmup + 120 s measurement.
- Search: fixed C16, stable-run, beam 128, max-expansions 384, rerank 128.
- Both runs: warmup 30,000 writes; measurement 120,000 writes and 4,799,998 queries; identical maintenance target sequences reached their durable watermarks.
- Recall: 1,000 queries before and after; all result sets complete.
- Failures: zero direct-path failures, route timeouts, Stage2 failures, and late storage RPC completions.
- Schema compatibility: the legacy fixed report lacks underhint/promotion fields and is strictly interpreted as zero; fixed nonzero values are rejected by the analyzer.

Raw reports:

- fixed: `/home/xjs/experiment/dvstor/motivation/results/live_extent_e2e/fixed_mixed_q40k_w1k/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_202410.json`
- high-water live-extent: `/home/xjs/experiment/dvstor/motivation/results/live_extent_e2e/live_highwater_mixed_q40k_w1k/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_205653.json`

## Key paired metrics

| metric | fixed | high-water live | live-fixed | change |
|---|---:|---:|---:|---:|
| query_ops | 4799998.000000 | 4799998.000000 | 0.000000 | +0.0000% |
| write_ops | 120000.000000 | 120000.000000 | 0.000000 | +0.0000% |
| query_qps | 39999.983333 | 39999.983333 | 0.000000 | +0.0000% |
| query_latency_mean_us | 3883.946000 | 3750.411000 | -133.535000 | -3.4381% |
| query_latency_p50_us | 3865.374000 | 3738.325000 | -127.049000 | -3.2868% |
| query_latency_p95_us | 4456.273000 | 4377.488000 | -78.785000 | -1.7680% |
| query_latency_p99_us | 4786.773000 | 4713.836000 | -72.937000 | -1.5237% |
| query_latency_p999_us | 5333.304000 | 5172.237000 | -161.067000 | -3.0200% |
| write_latency_mean_us | 4014.623000 | 3688.451000 | -326.172000 | -8.1246% |
| write_latency_p95_us | 6732.720000 | 6102.983000 | -629.737000 | -9.3534% |
| write_latency_p99_us | 9573.486000 | 8686.565000 | -886.921000 | -9.2643% |
| write_latency_p999_us | 18064.581000 | 14394.831000 | -3669.750000 | -20.3146% |
| gpu_query_us | 3827.571099 | 3695.025335 | -132.545764 | -3.4629% |
| gpu_rdma_issue_us | 133.002705 | 147.337763 | 14.335058 | +10.7780% |
| gpu_rdma_wait_us | 727.298658 | 694.704737 | -32.593921 | -4.4815% |
| gpu_graph_validation_us | 549.161545 | 402.609572 | -146.551973 | -26.6865% |
| gpu_exact_us | 260.059926 | 245.969505 | -14.090421 | -5.4181% |
| gpu_other_us | 171.798851 | 223.232405 | 51.433554 | +29.9382% |
| logical_graph_reads_per_query | 191.497229 | 191.495712 | -0.001517 | -0.0008% |
| graph_bytes_per_query | 159325.694319 | 81555.222555 | -77770.471764 | -48.8123% |
| graph_bytes_per_logical_parent | 832.000001 | 425.885373 | -406.114628 | -48.8119% |
| physical_graph_wqes_per_query | 191.497229 | 191.631229 | 0.134000 | +0.0700% |
| physical_graph_wqes_per_logical_parent | 1.000000 | 1.000708 | 0.000708 | +0.0708% |
| total_rdma_bytes_per_query | 178782.635847 | 101012.163768 | -77770.472079 | -43.5000% |
| rdma_wqes_per_query | 319.523382 | 319.657374 | 0.133992 | +0.0419% |
| fallback_graph_reads_per_query | 0.000000 | 0.135517 | 0.135517 | n/a |
| graph_extent_fallback_ratio | 0.000000 | 0.000724 | 0.000724 | n/a |
| underhint_graph_reads_per_query | 0.000000 | 0.135517 | 0.135517 | n/a |
| extent_hint_promotions_per_query | 0.000000 | 0.135327 | 0.135327 | n/a |
| extent_underhint_ratio | 0.000000 | 0.000724 | 0.000724 | n/a |
| extent_hint_promotion_rate | 0.000000 | 0.998596 | 0.998596 | n/a |
| recall_before | 0.940100 | 0.940100 | 0.000000 | +0.0000% |
| recall_after | 0.939700 | 0.939500 | -0.000200 | -0.0213% |
| recall_change | -0.000400 | -0.000600 | -0.000200 | n/a |
| stage2_remaining | 7.000000 | 9.000000 | 2.000000 | +28.5714% |
| stage2_max_backlog | 10.000000 | 13.000000 | 3.000000 | +30.0000% |
| stage2_backlog_slope_per_sec | -0.016666 | 0.000001 | 0.016667 | n/a |
| stage2_p99_delay_upper_ms | 64.000000 | 32.000000 | -32.000000 | -50.0000% |
| stage2_completion_outstanding | 6.000000 | 8.000000 | 2.000000 | +33.3333% |
| stage2_max_completion_outstanding_per_shard | 4.000000 | 4.000000 | 0.000000 | +0.0000% |
| stage2_failures | 0.000000 | 0.000000 | 0.000000 | n/a |
| storage_late_rpc_completions | 0.000000 | 0.000000 | 0.000000 | n/a |

## Physical-I/O and high-water judgement

- Logical graph reads: 919,186,314 → 919,179,033 (effectively unchanged).
- Physical graph WQEs: 919,186,315 → 919,829,516; **+643,201** (+0.0700%).
- Underhints/fallbacks/promotions: **650,483 / 650,483 / 649,570**; promotion rate **99.8596%**.
- Only **913** observed underhints are not promoted; fallback is 0.07236% of short reads.
- Extra physical graph WQEs minus fallbacks: -7,282; the residual matches the tiny logical read-count difference rather than extra search expansion.
- Graph bytes/query: -48.81%; total RDMA bytes/query: -43.50%.

## Conclusions

- **bytes:** PASS: graph bytes/query fall 48.81% and total RDMA bytes/query fall 43.50%.
- **highwater:** PASS IN THIS PAIR: 650,483 underhints produce 649,570 promotions (99.86%); only 913 observed underhints are not promoted. Fallback is 0.07236% of short reads and physical graph-WQE amplification is 0.070%.
- **query_performance:** IMPROVED: target QPS is fixed by the pacer; mean/P99/P999 query latency fall 3.44%/1.52%/3.02%, GPU query time falls 3.46%, and RDMA wait falls 4.48%. RDMA issue and GPU other time rise 10.78% and 29.94%, respectively, but do not erase the net gain.
- **write_and_stage2:** IMPROVED LATENCY, MIXED QUEUE SIGNALS: mean/P99/P999 write latency fall 8.12%/9.26%/20.31%; the Stage2 P99 delay histogram upper bound falls 64 ms to 32 ms. Max backlog rises 10 to 13 and remaining rises 7 to 9, so Stage2 is not uniformly better.
- **recall:** Initial Recall is identical at 0.9401. Post-workload Recall is 0.9397 fixed versus 0.9395 high-water Live-Extent, an absolute -0.0002 difference; both runs complete the registered protocol.
- **failures:** PASS: no GPU direct-path failure, route timeout, Stage2 failure, or late storage RPC; identical maintenance target sequences become durable.
- **overall:** This single strict pair supports high-water extent learning: it keeps the byte reduction, nearly removes fallback WQE amplification, and improves both query and write latency. The rate-limited experiment cannot establish maximum-throughput gain, and repetitions are still required for confidence.
