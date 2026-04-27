# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 16
- **dim**: 1,024
- **effective_insert_start_id**: 50,010,000
- **insert_start_id**: 0
- **measure_mixed**
  - **completed_reads**: 26,742
  - **completed_writes**: 1,688
  - **issued_reads**: 26,742
  - **issued_writes**: 1,688
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 65,536
- **threads**: 16
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 12,983
  - **completed_writes**: 1,135
  - **issued_reads**: 12,983
  - **issued_writes**: 1,135
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1688
  latency_ms: mean=283.569 p50=297.55 p95=404.967 p99=480.027
  top_categories:
    gpu_ns: 396052 ms (82.7425%)
    rdma_ns: 41842.8 ms (8.74172%)
    transfer_ns: 33006 ms (6.89556%)
    cpu_ns: 7755.55 ms (1.62027%)
```

### query

```text
query breakdown
  count: 26742
  latency_ms: mean=17.8424 p50=16.1934 p95=29.0488 p99=37.0632
  top_categories:
    gpu_ns: 288814 ms (60.5424%)
    transfer_ns: 113813 ms (23.8579%)
    cpu_ns: 46956.1 ms (9.84313%)
    rdma_ns: 27461.2 ms (5.75653%)
```

## INSERT 分析

- 操作数：**1,688**
- 平均端到端延迟：**283.569 ms**
- P50 端到端延迟：**297.550 ms**
- P95 端到端延迟：**404.967 ms**
- P99 端到端延迟：**480.027 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 396052.249 ms | 82.74% |
| rdma_ns | 41842.813 ms | 8.74% |
| transfer_ns | 33006.035 ms | 6.90% |
| cpu_ns | 7755.551 ms | 1.62% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（82.74%）、`rdma_ns`（8.74%）、`transfer_ns`（6.90%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_filter_ns | 2650.254 ms | 34.17% |
| cpu_insert_stage_candidates_ns | 1956.576 ms | 25.23% |
| cpu_insert_runtime_overhead_ns | 1308.760 ms | 16.88% |
| cpu_insert_finalize_ns | 647.231 ms | 8.35% |
| cpu_insert_beam_update_ns | 530.311 ms | 6.84% |
| cpu_insert_select_ns | 349.984 ms | 4.51% |
| cpu_insert_overflow_prepare_ns | 138.245 ms | 1.78% |
| cpu_insert_pruned_neighbor_collect_ns | 53.625 ms | 0.69% |
| cpu_insert_candidate_collect_ns | 36.128 ms | 0.47% |
| cpu_insert_neighbor_collect_ns | 33.212 ms | 0.43% |
| cpu_insert_prune_prepare_ns | 27.055 ms | 0.35% |
| cpu_insert_preprune_sort_ns | 12.008 ms | 0.15% |
| cpu_insert_init_ns | 10.087 ms | 0.13% |
| cpu_cache_lookup_ns | 1.220 ms | 0.02% |
| cpu_insert_candidate_sort_ns | 0.595 ms | 0.01% |
| cpu_insert_neighbor_prepare_ns | 0.196 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.064 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_filter_ns`（34.17%）、`cpu_insert_stage_candidates_ns`（25.23%）、`cpu_insert_runtime_overhead_ns`（16.88%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 322713.594 ms | 81.48% |
| gpu_insert_distance_ns | 37439.697 ms | 9.45% |
| gpu_insert_prune_ns | 27326.463 ms | 6.90% |
| gpu_insert_overflow_distance_ns | 7934.128 ms | 2.00% |
| gpu_insert_quantize_ns | 638.367 ms | 0.16% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（81.48%）、`gpu_insert_distance_ns`（9.45%）、`gpu_insert_prune_ns`（6.90%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 14895.473 ms | 35.60% |
| rdma_neighbor_fetch_ns | 13581.942 ms | 32.46% |
| rdma_overflow_vec_fetch_ns | 4505.047 ms | 10.77% |
| rdma_neighbor_lock_ns | 2712.745 ms | 6.48% |
| rdma_pruned_neighbor_write_ns | 1557.265 ms | 3.72% |
| rdma_neighbor_list_read_ns | 1463.065 ms | 3.50% |
| rdma_neighbor_node_read_ns | 1462.923 ms | 3.50% |
| rdma_neighbor_unlock_ns | 1216.604 ms | 2.91% |
| rdma_candidate_fetch_ns | 285.322 ms | 0.68% |
| rdma_neighbor_list_write_ns | 82.728 ms | 0.20% |
| rdma_new_node_write_ns | 30.434 ms | 0.07% |
| rdma_alloc_ns | 26.080 ms | 0.06% |
| rdma_medoid_ptr_ns | 23.185 ms | 0.06% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（35.60%）、`rdma_neighbor_fetch_ns`（32.46%）、`rdma_overflow_vec_fetch_ns`（10.77%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 20636.406 ms | 62.52% |
| transfer_overflow_prune_d2h_ns | 4132.268 ms | 12.52% |
| transfer_overflow_dist_d2h_ns | 3003.723 ms | 9.10% |
| transfer_overflow_prune_inputs_h2d_ns | 2321.165 ms | 7.03% |
| transfer_overflow_query_h2d_ns | 1327.504 ms | 4.02% |
| transfer_overflow_candidate_h2d_ns | 1223.649 ms | 3.71% |
| transfer_quantize_d2h_ns | 219.612 ms | 0.67% |
| transfer_prune_d2h_ns | 83.941 ms | 0.25% |
| transfer_insert_query_h2d_ns | 30.702 ms | 0.09% |
| transfer_prune_h2d_ns | 27.065 ms | 0.08% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（62.52%）、`transfer_overflow_prune_d2h_ns`（12.52%）、`transfer_overflow_dist_d2h_ns`（9.10%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 478663.779 ms |
| mean_end_to_end_ns | 283.569 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 283.564 ms |
| p50_end_to_end_ns | 297.550 ms |
| p50_service_ns | 297.547 ms |
| p95_end_to_end_ns | 404.967 ms |
| p95_service_ns | 404.962 ms |
| p99_end_to_end_ns | 480.027 ms |
| p99_service_ns | 480.023 ms |
| queue_wait_ns | 7.131 ms |
| service_ns | 478656.648 ms |

### Counters

| 字段 | 值 |
|---|---|
| overflow_prune_global_load_bytes_upper_bound | 1,717,523,808,000 |
| rdma_read_bytes | 81,342,533,454 |
| vector_rdma_bytes | 80,794,704,896 |
| h2d_bytes | 27,273,096,176 |
| neighbor_rdma_bytes | 547,813,206 |
| overflow_prune_pair_checks_upper_bound | 209,352,000 |
| d2h_bytes | 103,487,904 |
| rdma_write_bytes | 64,800,046 |
| rdma_read_ops | 21,862,497 |
| vector_rdma_read_ops | 19,724,854 |
| overflow_prune_candidates | 6,542,250 |
| overflow_prune_kernel_threads | 6,542,250 |
| neighbor_rdma_read_ops | 2,135,724 |
| l2_kernels | 806,347 |
| lock_attempts | 441,666 |
| cas_failures | 347,372 |
| lock_retries | 347,372 |
| rdma_write_ops | 321,279 |
| prune_kernels | 102,552 |
| overflow_prune_kernel_blocks | 100,650 |
| overflow_prunes | 100,650 |
| vector_rdma_read_avg_bytes | 4,096.086333 |
| rdma_read_avg_bytes | 3,720.642407 |
| cache_hits | 1,919 |
| remote_allocations | 1,897 |
| neighbor_rdma_read_avg_bytes | 256.5 |
| rdma_write_avg_bytes | 201.693998 |
| overflow_prune_avg_candidates | 65 |
| overflow_prune_avg_kernel_threads | 65 |
| overflow_prune_max_candidates | 65 |
| overflow_prune_max_kernel_threads | 65 |
| overflow_prune_avg_kernel_blocks | 1 |
| cache_misses | 0 |
| exact_reranks | 0 |
| gpu_rabitq_cache_duplicate_fills | 0 |
| gpu_rabitq_cache_fallback_batches | 0 |
| gpu_rabitq_cache_fill_bytes | 0 |
| gpu_rabitq_cache_fills | 0 |
| gpu_rabitq_cache_hits | 0 |
| gpu_rabitq_cache_misses | 0 |
| neighbor_cache_hits | 0 |
| neighbor_cache_misses | 0 |
| query_host_staging_fallback_bytes | 0 |
| query_rdma_to_staging_bytes | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| rabitq_rdma_read_avg_bytes | 0 |
| rabitq_rdma_read_ops | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**26,742**
- 平均端到端延迟：**17.842 ms**
- P50 端到端延迟：**16.193 ms**
- P95 端到端延迟：**29.049 ms**
- P99 端到端延迟：**37.063 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 288813.828 ms | 60.54% |
| transfer_ns | 113812.896 ms | 23.86% |
| cpu_ns | 46956.075 ms | 9.84% |
| rdma_ns | 27461.171 ms | 5.76% |

- query 一级热点：占比最高的几项是 `gpu_ns`（60.54%）、`transfer_ns`（23.86%）、`cpu_ns`（9.84%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_filter_ns | 21555.766 ms | 45.91% |
| cpu_query_runtime_overhead_ns | 19400.969 ms | 41.32% |
| cpu_query_beam_update_ns | 2812.073 ms | 5.99% |
| cpu_cache_lookup_ns | 1279.700 ms | 2.73% |
| cpu_query_select_ns | 781.394 ms | 1.66% |
| cpu_query_finalize_ns | 505.380 ms | 1.08% |
| cpu_query_result_ids_ns | 498.864 ms | 1.06% |
| cpu_query_beam_sort_ns | 103.898 ms | 0.22% |
| cpu_query_rerank_collect_ns | 14.614 ms | 0.03% |
| cpu_query_rerank_update_ns | 3.416 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_filter_ns`（45.91%）、`cpu_query_runtime_overhead_ns`（41.32%）、`cpu_query_beam_update_ns`（5.99%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 285132.807 ms | 98.73% |
| gpu_query_prepare_ns | 2134.833 ms | 0.74% |
| gpu_query_rerank_ns | 1546.187 ms | 0.54% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.73%）、`gpu_query_prepare_ns`（0.74%）、`gpu_query_rerank_ns`（0.54%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 15853.537 ms | 57.73% |
| rdma_neighbor_fetch_ns | 10036.488 ms | 36.55% |
| rdma_rerank_fetch_ns | 1342.687 ms | 4.89% |
| rdma_medoid_ptr_ns | 228.459 ms | 0.83% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（57.73%）、`rdma_neighbor_fetch_ns`（36.55%）、`rdma_rerank_fetch_ns`（4.89%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 112535.017 ms | 98.88% |
| transfer_rerank_d2h_ns | 847.057 ms | 0.74% |
| transfer_query_h2d_ns | 430.822 ms | 0.38% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（98.88%）、`transfer_rerank_d2h_ns`（0.74%）、`transfer_query_h2d_ns`（0.38%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 477142.642 ms |
| mean_end_to_end_ns | 17.842 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 17.839 ms |
| p50_end_to_end_ns | 16.193 ms |
| p50_service_ns | 16.190 ms |
| p95_end_to_end_ns | 29.049 ms |
| p95_service_ns | 29.045 ms |
| p99_end_to_end_ns | 37.063 ms |
| p99_service_ns | 37.060 ms |
| queue_wait_ns | 98.672 ms |
| service_ns | 477043.970 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 22,349,672,510 |
| vector_rdma_bytes | 11,568,695,520 |
| query_rdma_to_staging_bytes | 11,479,547,904 |
| gpu_rabitq_cache_fill_bytes | 10,347,088,960 |
| rabitq_rdma_bytes | 10,344,029,800 |
| d2h_bytes | 573,094,936 |
| neighbor_rdma_bytes | 436,713,822 |
| gpu_rabitq_cache_hits | 120,591,058 |
| h2d_bytes | 119,562,240 |
| rdma_read_ops | 24,448,426 |
| gpu_rabitq_cache_fills | 19,898,248 |
| gpu_rabitq_cache_misses | 19,892,365 |
| rabitq_rdma_read_ops | 19,892,365 |
| visited_neighborlists | 3,999,720 |
| rabitq_kernels | 3,921,685 |
| neighbor_cache_hits | 3,148,102 |
| vector_rdma_read_ops | 2,824,302 |
| neighbor_rdma_read_ops | 1,702,588 |
| neighbor_cache_misses | 851,294 |
| cache_hits | 298,817 |
| exact_reranks | 29,194 |
| cache_misses | 22,164 |
| vector_rdma_read_avg_bytes | 4,096.125528 |
| rdma_read_avg_bytes | 914.155885 |
| rabitq_rdma_read_avg_bytes | 520 |
| neighbor_rdma_read_avg_bytes | 256.5 |
| cas_failures | 0 |
| gpu_rabitq_cache_duplicate_fills | 0 |
| gpu_rabitq_cache_fallback_batches | 0 |
| l2_kernels | 0 |
| lock_attempts | 0 |
| lock_retries | 0 |
| overflow_prune_avg_candidates | 0 |
| overflow_prune_avg_kernel_blocks | 0 |
| overflow_prune_avg_kernel_threads | 0 |
| overflow_prune_candidates | 0 |
| overflow_prune_global_load_bytes_upper_bound | 0 |
| overflow_prune_kernel_blocks | 0 |
| overflow_prune_kernel_threads | 0 |
| overflow_prune_max_candidates | 0 |
| overflow_prune_max_kernel_threads | 0 |
| overflow_prune_pair_checks_upper_bound | 0 |
| overflow_prunes | 0 |
| prune_kernels | 0 |
| query_host_staging_fallback_bytes | 0 |
| rdma_write_avg_bytes | 0 |
| rdma_write_bytes | 0 |
| rdma_write_ops | 0 |
| remote_allocations | 0 |
| visited_nodes | 0 |

## System Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 103,692,205,964 |
| h2d_bytes | 27,392,658,416 |
| d2h_bytes | 676,582,840 |
| rdma_write_bytes | 64,800,046 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 1.62% | 9.84% |
| gpu_ns | 82.74% | 60.54% |
| rdma_ns | 8.74% | 5.76% |
| transfer_ns | 6.90% | 23.86% |

- Insert 最大部分是 **gpu_ns**，占 **82.74%**。
- Query 最大部分是 **gpu_ns**，占 **60.54%**。
- Insert 更偏向 GPU 计算密集。
- Insert 的 RDMA 成本同样非常显著。
