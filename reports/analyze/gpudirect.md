# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 16
- **dim**: 1,024
- **effective_insert_start_id**: 50,010,000
- **insert_start_id**: 0
- **measure_mixed**
  - **completed_reads**: 24,635
  - **completed_writes**: 1,706
  - **issued_reads**: 24,635
  - **issued_writes**: 1,706
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
  - **completed_reads**: 12,002
  - **completed_writes**: 1,146
  - **issued_reads**: 12,002
  - **issued_writes**: 1,146
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1706
  latency_ms: mean=280.248 p50=289.14 p95=407.674 p99=457.045
  top_categories:
    gpu_ns: 378500 ms (79.1682%)
    rdma_ns: 62114.7 ms (12.9921%)
    transfer_ns: 30025.7 ms (6.28028%)
    cpu_ns: 7455.65 ms (1.55945%)
```

### query

```text
query breakdown
  count: 24635
  latency_ms: mean=19.3526 p50=17.7196 p95=31.5345 p99=39.6742
  top_categories:
    gpu_ns: 233293 ms (48.9435%)
    rdma_ns: 117401 ms (24.63%)
    transfer_ns: 96121.5 ms (20.1657%)
    cpu_ns: 29842.8 ms (6.26085%)
```

## INSERT 分析

- 操作数：**1,706**
- 平均端到端延迟：**280.248 ms**
- P50 端到端延迟：**289.140 ms**
- P95 端到端延迟：**407.674 ms**
- P99 端到端延迟：**457.045 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 378499.547 ms | 79.17% |
| rdma_ns | 62114.671 ms | 12.99% |
| transfer_ns | 30025.739 ms | 6.28% |
| cpu_ns | 7455.653 ms | 1.56% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（79.17%）、`rdma_ns`（12.99%）、`transfer_ns`（6.28%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_filter_ns | 2436.097 ms | 32.67% |
| cpu_insert_stage_candidates_ns | 1897.693 ms | 25.45% |
| cpu_insert_runtime_overhead_ns | 1418.832 ms | 19.03% |
| cpu_insert_beam_update_ns | 547.481 ms | 7.34% |
| cpu_insert_finalize_ns | 482.174 ms | 6.47% |
| cpu_insert_select_ns | 354.514 ms | 4.75% |
| cpu_insert_overflow_prepare_ns | 141.363 ms | 1.90% |
| cpu_insert_pruned_neighbor_collect_ns | 57.960 ms | 0.78% |
| cpu_insert_candidate_collect_ns | 36.773 ms | 0.49% |
| cpu_insert_neighbor_collect_ns | 33.589 ms | 0.45% |
| cpu_insert_prune_prepare_ns | 24.781 ms | 0.33% |
| cpu_insert_preprune_sort_ns | 12.173 ms | 0.16% |
| cpu_insert_init_ns | 10.059 ms | 0.13% |
| cpu_cache_lookup_ns | 1.254 ms | 0.02% |
| cpu_insert_candidate_sort_ns | 0.638 ms | 0.01% |
| cpu_insert_neighbor_prepare_ns | 0.189 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.083 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_filter_ns`（32.67%）、`cpu_insert_stage_candidates_ns`（25.45%）、`cpu_insert_runtime_overhead_ns`（19.03%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 302591.062 ms | 79.94% |
| gpu_insert_distance_ns | 39488.887 ms | 10.43% |
| gpu_insert_prune_ns | 27614.936 ms | 7.30% |
| gpu_insert_overflow_distance_ns | 8202.721 ms | 2.17% |
| gpu_insert_quantize_ns | 601.941 ms | 0.16% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（79.94%）、`gpu_insert_distance_ns`（10.43%）、`gpu_insert_prune_ns`（7.30%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_neighbor_fetch_ns | 21507.386 ms | 34.63% |
| rdma_vector_fetch_ns | 21336.068 ms | 34.35% |
| rdma_overflow_vec_fetch_ns | 6383.355 ms | 10.28% |
| rdma_neighbor_lock_ns | 3173.009 ms | 5.11% |
| rdma_pruned_neighbor_write_ns | 2710.726 ms | 4.36% |
| rdma_neighbor_list_read_ns | 2205.550 ms | 3.55% |
| rdma_neighbor_node_read_ns | 2185.452 ms | 3.52% |
| rdma_neighbor_unlock_ns | 1949.982 ms | 3.14% |
| rdma_candidate_fetch_ns | 407.599 ms | 0.66% |
| rdma_neighbor_list_write_ns | 129.330 ms | 0.21% |
| rdma_new_node_write_ns | 42.961 ms | 0.07% |
| rdma_medoid_ptr_ns | 42.871 ms | 0.07% |
| rdma_alloc_ns | 40.383 ms | 0.07% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_neighbor_fetch_ns`（34.63%）、`rdma_vector_fetch_ns`（34.35%）、`rdma_overflow_vec_fetch_ns`（10.28%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 18755.387 ms | 62.46% |
| transfer_overflow_prune_d2h_ns | 3732.894 ms | 12.43% |
| transfer_overflow_dist_d2h_ns | 2755.901 ms | 9.18% |
| transfer_overflow_prune_inputs_h2d_ns | 2114.112 ms | 7.04% |
| transfer_overflow_query_h2d_ns | 1225.053 ms | 4.08% |
| transfer_overflow_candidate_h2d_ns | 1103.916 ms | 3.68% |
| transfer_quantize_d2h_ns | 210.988 ms | 0.70% |
| transfer_prune_d2h_ns | 73.985 ms | 0.25% |
| transfer_insert_query_h2d_ns | 28.714 ms | 0.10% |
| transfer_prune_h2d_ns | 24.789 ms | 0.08% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（62.46%）、`transfer_overflow_prune_d2h_ns`（12.43%）、`transfer_overflow_dist_d2h_ns`（9.18%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 478102.880 ms |
| mean_end_to_end_ns | 280.248 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 280.244 ms |
| p50_end_to_end_ns | 289.140 ms |
| p50_service_ns | 289.137 ms |
| p95_end_to_end_ns | 407.674 ms |
| p95_service_ns | 407.670 ms |
| p99_end_to_end_ns | 457.045 ms |
| p99_service_ns | 457.042 ms |
| queue_wait_ns | 7.271 ms |
| service_ns | 478095.609 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 87,403,101,190 |
| vector_rdma_bytes | 86,814,473,408 |
| h2d_bytes | 29,506,691,960 |
| neighbor_rdma_bytes | 588,611,070 |
| d2h_bytes | 111,394,676 |
| rdma_write_bytes | 70,052,154 |
| l2_kernels | 866,932 |
| lock_attempts | 262,633 |
| cas_failures | 167,425 |
| lock_retries | 167,425 |
| prune_kernels | 110,937 |
| overflow_prunes | 108,887 |
| cache_hits | 2,090 |
| remote_allocations | 2,040 |
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

- 操作数：**24,635**
- 平均端到端延迟：**19.353 ms**
- P50 端到端延迟：**17.720 ms**
- P95 端到端延迟：**31.534 ms**
- P99 端到端延迟：**39.674 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 233292.975 ms | 48.94% |
| rdma_ns | 117400.716 ms | 24.63% |
| transfer_ns | 96121.458 ms | 20.17% |
| cpu_ns | 29842.833 ms | 6.26% |

- query 一级热点：占比最高的几项是 `gpu_ns`（48.94%）、`rdma_ns`（24.63%）、`transfer_ns`（20.17%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_filter_ns | 17065.567 ms | 57.18% |
| cpu_query_runtime_overhead_ns | 6697.985 ms | 22.44% |
| cpu_query_beam_update_ns | 2617.512 ms | 8.77% |
| cpu_cache_lookup_ns | 1118.415 ms | 3.75% |
| cpu_query_finalize_ns | 757.157 ms | 2.54% |
| cpu_query_result_ids_ns | 751.359 ms | 2.52% |
| cpu_query_select_ns | 720.143 ms | 2.41% |
| cpu_query_beam_sort_ns | 97.342 ms | 0.33% |
| cpu_query_rerank_collect_ns | 14.291 ms | 0.05% |
| cpu_query_rerank_update_ns | 3.062 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_filter_ns`（57.18%）、`cpu_query_runtime_overhead_ns`（22.44%）、`cpu_query_beam_update_ns`（8.77%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 230287.376 ms | 98.71% |
| gpu_query_prepare_ns | 1716.178 ms | 0.74% |
| gpu_query_rerank_ns | 1289.422 ms | 0.55% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.71%）、`gpu_query_prepare_ns`（0.74%）、`gpu_query_rerank_ns`（0.55%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 103745.000 ms | 88.37% |
| rdma_neighbor_fetch_ns | 11764.423 ms | 10.02% |
| rdma_rerank_fetch_ns | 1564.621 ms | 1.33% |
| rdma_medoid_ptr_ns | 326.673 ms | 0.28% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（88.37%）、`rdma_neighbor_fetch_ns`（10.02%）、`rdma_rerank_fetch_ns`（1.33%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 95049.401 ms | 98.88% |
| transfer_rerank_d2h_ns | 702.065 ms | 0.73% |
| transfer_query_h2d_ns | 369.992 ms | 0.38% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（98.88%）、`transfer_rerank_d2h_ns`（0.73%）、`transfer_query_h2d_ns`（0.38%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 476750.276 ms |
| mean_end_to_end_ns | 19.353 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 19.349 ms |
| p50_end_to_end_ns | 17.720 ms |
| p50_service_ns | 17.716 ms |
| p95_end_to_end_ns | 31.534 ms |
| p95_service_ns | 31.531 ms |
| p99_end_to_end_ns | 39.674 ms |
| p99_service_ns | 39.671 ms |
| queue_wait_ns | 92.293 ms |
| service_ns | 476657.983 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 84,085,158,008 |
| query_rdma_to_staging_bytes | 83,568,570,888 |
| rabitq_rdma_bytes | 72,272,318,560 |
| vector_rdma_bytes | 11,376,383,568 |
| d2h_bytes | 566,949,152 |
| neighbor_rdma_bytes | 436,226,472 |
| h2d_bytes | 117,551,104 |
| visited_neighborlists | 3,947,317 |
| rabitq_kernels | 3,870,361 |
| neighbor_cache_hits | 3,096,866 |
| neighbor_cache_misses | 850,344 |
| cache_hits | 292,894 |
| exact_reranks | 28,693 |
| cache_misses | 22,638 |
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
| query_host_staging_fallback_bytes | 0 |
| rdma_write_bytes | 0 |
| remote_allocations | 0 |
| visited_nodes | 0 |

## System Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 171,488,259,198 |
| h2d_bytes | 29,624,243,064 |
| d2h_bytes | 678,343,828 |
| rdma_write_bytes | 70,052,154 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 1.56% | 6.26% |
| gpu_ns | 79.17% | 48.94% |
| rdma_ns | 12.99% | 24.63% |
| transfer_ns | 6.28% | 20.17% |

- Insert 最大部分是 **gpu_ns**，占 **79.17%**。
- Query 最大部分是 **gpu_ns**，占 **48.94%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
