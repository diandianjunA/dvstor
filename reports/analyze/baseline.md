# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 16
- **dim**: 1,024
- **effective_insert_start_id**: 50,010,000
- **insert_start_id**: 0
- **measure_mixed**
  - **completed_reads**: 4,397
  - **completed_writes**: 1,114
  - **issued_reads**: 4,397
  - **issued_writes**: 1,114
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
  - **completed_reads**: 1,741
  - **completed_writes**: 476
  - **issued_reads**: 1,741
  - **issued_writes**: 476
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1114
  latency_ms: mean=428.739 p50=405.696 p95=657.334 p99=804.361
  top_categories:
    gpu_ns: 196465 ms (41.135%)
    rdma_ns: 159675 ms (33.4321%)
    cpu_ns: 110240 ms (23.0816%)
    transfer_ns: 11230.1 ms (2.35131%)
```

### query

```text
query breakdown
  count: 4397
  latency_ms: mean=108.959 p50=101.456 p95=179.623 p99=222.822
  top_categories:
    rdma_ns: 247399 ms (51.6407%)
    cpu_ns: 179700 ms (37.5095%)
    gpu_ns: 38021.3 ms (7.93636%)
    transfer_ns: 13957.4 ms (2.9134%)
```

## INSERT 分析

- 操作数：**1,114**
- 平均端到端延迟：**428.739 ms**
- P50 端到端延迟：**405.696 ms**
- P95 端到端延迟：**657.334 ms**
- P99 端到端延迟：**804.361 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 196465.094 ms | 41.13% |
| rdma_ns | 159675.199 ms | 33.43% |
| cpu_ns | 110240.141 ms | 23.08% |
| transfer_ns | 11230.124 ms | 2.35% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（41.13%）、`rdma_ns`（33.43%）、`cpu_ns`（23.08%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 77515.756 ms | 70.32% |
| cpu_insert_runtime_overhead_ns | 25742.966 ms | 23.35% |
| cpu_insert_candidate_sort_ns | 4645.695 ms | 4.21% |
| cpu_insert_filter_ns | 1317.818 ms | 1.20% |
| cpu_insert_beam_update_ns | 336.826 ms | 0.31% |
| cpu_insert_finalize_ns | 257.827 ms | 0.23% |
| cpu_insert_select_ns | 247.379 ms | 0.22% |
| cpu_insert_overflow_prepare_ns | 80.775 ms | 0.07% |
| cpu_insert_pruned_neighbor_collect_ns | 24.808 ms | 0.02% |
| cpu_insert_neighbor_collect_ns | 19.371 ms | 0.02% |
| cpu_insert_prune_prepare_ns | 18.633 ms | 0.02% |
| cpu_insert_candidate_collect_ns | 16.954 ms | 0.02% |
| cpu_insert_preprune_sort_ns | 7.953 ms | 0.01% |
| cpu_insert_init_ns | 6.364 ms | 0.01% |
| cpu_cache_lookup_ns | 0.818 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.168 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.031 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（70.32%）、`cpu_insert_runtime_overhead_ns`（23.35%）、`cpu_insert_candidate_sort_ns`（4.21%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 159234.072 ms | 81.05% |
| gpu_insert_distance_ns | 17034.930 ms | 8.67% |
| gpu_insert_prune_ns | 16243.141 ms | 8.27% |
| gpu_insert_overflow_distance_ns | 3594.101 ms | 1.83% |
| gpu_insert_quantize_ns | 358.850 ms | 0.18% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（81.05%）、`gpu_insert_distance_ns`（8.67%）、`gpu_insert_prune_ns`（8.27%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 90773.239 ms | 56.85% |
| rdma_overflow_vec_fetch_ns | 39294.209 ms | 24.61% |
| rdma_neighbor_fetch_ns | 15192.098 ms | 9.51% |
| rdma_candidate_fetch_ns | 4879.663 ms | 3.06% |
| rdma_pruned_neighbor_write_ns | 2824.304 ms | 1.77% |
| rdma_neighbor_node_read_ns | 1907.377 ms | 1.19% |
| rdma_neighbor_lock_ns | 1868.942 ms | 1.17% |
| rdma_neighbor_list_read_ns | 1753.550 ms | 1.10% |
| rdma_neighbor_unlock_ns | 856.781 ms | 0.54% |
| rdma_neighbor_list_write_ns | 255.173 ms | 0.16% |
| rdma_new_node_write_ns | 32.833 ms | 0.02% |
| rdma_alloc_ns | 21.482 ms | 0.01% |
| rdma_medoid_ptr_ns | 15.549 ms | 0.01% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（56.85%）、`rdma_overflow_vec_fetch_ns`（24.61%）、`rdma_neighbor_fetch_ns`（9.51%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 5482.210 ms | 48.82% |
| transfer_candidate_h2d_ns | 2388.428 ms | 21.27% |
| transfer_overflow_prune_d2h_ns | 1047.861 ms | 9.33% |
| transfer_overflow_dist_d2h_ns | 800.328 ms | 7.13% |
| transfer_overflow_prune_inputs_h2d_ns | 604.535 ms | 5.38% |
| transfer_overflow_query_h2d_ns | 380.700 ms | 3.39% |
| transfer_overflow_candidate_h2d_ns | 346.902 ms | 3.09% |
| transfer_quantize_d2h_ns | 120.969 ms | 1.08% |
| transfer_prune_d2h_ns | 27.809 ms | 0.25% |
| transfer_prune_h2d_ns | 18.634 ms | 0.17% |
| transfer_insert_query_h2d_ns | 11.748 ms | 0.10% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（48.82%）、`transfer_candidate_h2d_ns`（21.27%）、`transfer_overflow_prune_d2h_ns`（9.33%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 477614.776 ms |
| mean_end_to_end_ns | 428.739 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 428.735 ms |
| p50_end_to_end_ns | 405.696 ms |
| p50_service_ns | 405.692 ms |
| p95_end_to_end_ns | 657.334 ms |
| p95_service_ns | 657.331 ms |
| p99_end_to_end_ns | 804.361 ms |
| p99_service_ns | 804.356 ms |
| queue_wait_ns | 4.217 ms |
| service_ns | 477610.559 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 52,144,558,191 |
| h2d_bytes | 52,034,990,728 |
| vector_rdma_bytes | 51,788,563,536 |
| neighbor_rdma_bytes | 355,984,551 |
| d2h_bytes | 64,426,064 |
| rdma_write_bytes | 38,787,586 |
| l2_kernels | 449,383 |
| lock_attempts | 332,899 |
| cas_failures | 276,724 |
| lock_retries | 276,724 |
| prune_kernels | 58,378 |
| overflow_prunes | 57,155 |
| cache_hits | 1,263 |
| remote_allocations | 1,226 |
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
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**4,397**
- 平均端到端延迟：**108.959 ms**
- P50 端到端延迟：**101.456 ms**
- P95 端到端延迟：**179.623 ms**
- P99 端到端延迟：**222.822 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 247398.551 ms | 51.64% |
| cpu_ns | 179699.533 ms | 37.51% |
| gpu_ns | 38021.272 ms | 7.94% |
| transfer_ns | 13957.412 ms | 2.91% |

- query 一级热点：占比最高的几项是 `rdma_ns`（51.64%）、`cpu_ns`（37.51%）、`gpu_ns`（7.94%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 168729.948 ms | 93.90% |
| cpu_query_rerank_prepare_ns | 4229.083 ms | 2.35% |
| cpu_query_runtime_overhead_ns | 2801.511 ms | 1.56% |
| cpu_query_filter_ns | 2223.525 ms | 1.24% |
| cpu_query_beam_update_ns | 466.886 ms | 0.26% |
| cpu_query_finalize_ns | 375.200 ms | 0.21% |
| cpu_query_result_ids_ns | 374.173 ms | 0.21% |
| cpu_cache_lookup_ns | 351.220 ms | 0.20% |
| cpu_query_select_ns | 126.902 ms | 0.07% |
| cpu_query_beam_sort_ns | 17.783 ms | 0.01% |
| cpu_query_rerank_collect_ns | 2.676 ms | 0.00% |
| cpu_query_rerank_update_ns | 0.626 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（93.90%）、`cpu_query_rerank_prepare_ns`（2.35%）、`cpu_query_runtime_overhead_ns`（1.56%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 37425.461 ms | 98.43% |
| gpu_query_rerank_ns | 323.218 ms | 0.85% |
| gpu_query_prepare_ns | 272.593 ms | 0.72% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.43%）、`gpu_query_rerank_ns`（0.85%）、`gpu_query_prepare_ns`（0.72%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 236094.908 ms | 95.43% |
| rdma_neighbor_fetch_ns | 6840.383 ms | 2.76% |
| rdma_rerank_fetch_ns | 4410.627 ms | 1.78% |
| rdma_medoid_ptr_ns | 52.633 ms | 0.02% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（95.43%）、`rdma_neighbor_fetch_ns`（2.76%）、`rdma_rerank_fetch_ns`（1.78%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 9278.464 ms | 66.48% |
| transfer_rabitq_h2d_ns | 4530.035 ms | 32.46% |
| transfer_rerank_d2h_ns | 71.946 ms | 0.52% |
| transfer_query_h2d_ns | 39.979 ms | 0.29% |
| transfer_rerank_h2d_ns | 36.988 ms | 0.27% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（66.48%）、`transfer_rabitq_h2d_ns`（32.46%）、`transfer_rerank_d2h_ns`（0.52%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 479091.737 ms |
| mean_end_to_end_ns | 108.959 ms |
| mean_queue_wait_ns | 0.003 ms |
| mean_service_ns | 108.955 ms |
| p50_end_to_end_ns | 101.456 ms |
| p50_service_ns | 101.452 ms |
| p95_end_to_end_ns | 179.623 ms |
| p95_service_ns | 179.619 ms |
| p99_end_to_end_ns | 222.822 ms |
| p99_service_ns | 222.819 ms |
| queue_wait_ns | 14.970 ms |
| service_ns | 479076.768 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 13,580,799,518 |
| h2d_bytes | 13,436,269,544 |
| query_host_staging_fallback_bytes | 13,417,403,368 |
| rabitq_rdma_bytes | 11,607,654,240 |
| vector_rdma_bytes | 1,846,185,056 |
| neighbor_rdma_bytes | 126,923,382 |
| d2h_bytes | 91,058,120 |
| visited_neighborlists | 620,311 |
| rabitq_kernels | 611,096 |
| neighbor_cache_hits | 372,902 |
| neighbor_cache_misses | 247,414 |
| cache_hits | 41,746 |
| cache_misses | 8,903 |
| exact_reranks | 4,603 |
| cas_failures | 0 |
| gpu_rabitq_cache_duplicate_fills | 0 |
| gpu_rabitq_cache_fallback_batches | 0 |
| gpu_rabitq_cache_fill_bytes | 0 |
| gpu_rabitq_cache_fills | 0 |
| gpu_rabitq_cache_hits | 0 |
| gpu_rabitq_cache_misses | 0 |
| l2_kernels | 0 |
| lock_attempts | 0 |
| lock_retries | 0 |
| overflow_prunes | 0 |
| prune_kernels | 0 |
| query_rdma_to_staging_bytes | 0 |
| rdma_write_bytes | 0 |
| remote_allocations | 0 |
| visited_nodes | 0 |

## System Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 65,725,357,709 |
| h2d_bytes | 65,471,260,272 |
| d2h_bytes | 155,484,184 |
| rdma_write_bytes | 38,787,586 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 23.08% | 37.51% |
| gpu_ns | 41.13% | 7.94% |
| rdma_ns | 33.43% | 51.64% |
| transfer_ns | 2.35% | 2.91% |

- Insert 最大部分是 **gpu_ns**，占 **41.13%**。
- Query 最大部分是 **rdma_ns**，占 **51.64%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
