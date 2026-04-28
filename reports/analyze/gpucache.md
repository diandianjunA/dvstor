# Breakdown 分析报告

## 实验元信息

- **client\_threads**: 16
- **coroutines**: 16
- **dim**: 1,024
- **effective\_insert\_start\_id**: 50,010,000
- **insert\_start\_id**: 0
- **measure\_mixed**
  - **completed\_reads**: 26,742
  - **completed\_writes**: 1,688
  - **issued\_reads**: 26,742
  - **issued\_writes**: 1,688
- **measure\_ops**: 1,000
- **measure\_seconds**: 60
- **operation\_granularity**: single\_vector
- **read\_ratio**: 0.5
- **run\_mode**: time
- **search\_mode**: rabitq\_gpu
- **synthetic\_query\_vectors**: 65,536
- **threads**: 16
- **time\_completion\_policy**: drain
- **time\_issue\_policy**: bounded\_by\_observed\_call\_latency
- **warmup\_mixed**
  - **completed\_reads**: 12,983
  - **completed\_writes**: 1,135
  - **issued\_reads**: 12,983
  - **issued\_writes**: 1,135
- **warmup\_ops**: 100
- **warmup\_seconds**: 30
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

| 部分           | 时间            | 占比     |
| ------------ | ------------- | ------ |
| gpu\_ns      | 396052.249 ms | 82.74% |
| rdma\_ns     | 41842.813 ms  | 8.74%  |
| transfer\_ns | 33006.035 ms  | 6.90%  |
| cpu\_ns      | 7755.551 ms   | 1.62%  |

- insert 一级热点：占比最高的几项是 `gpu_ns`（82.74%）、`rdma_ns`（8.74%）、`transfer_ns`（6.90%）。

### Sub Breakdown 细分占比

#### cpu\_ns

| 部分                                         | 时间          | 占比     |
| ------------------------------------------ | ----------- | ------ |
| cpu\_insert\_filter\_ns                    | 2650.254 ms | 34.17% |
| cpu\_insert\_stage\_candidates\_ns         | 1956.576 ms | 25.23% |
| cpu\_insert\_runtime\_overhead\_ns         | 1308.760 ms | 16.88% |
| cpu\_insert\_finalize\_ns                  | 647.231 ms  | 8.35%  |
| cpu\_insert\_beam\_update\_ns              | 530.311 ms  | 6.84%  |
| cpu\_insert\_select\_ns                    | 349.984 ms  | 4.51%  |
| cpu\_insert\_overflow\_prepare\_ns         | 138.245 ms  | 1.78%  |
| cpu\_insert\_pruned\_neighbor\_collect\_ns | 53.625 ms   | 0.69%  |
| cpu\_insert\_candidate\_collect\_ns        | 36.128 ms   | 0.47%  |
| cpu\_insert\_neighbor\_collect\_ns         | 33.212 ms   | 0.43%  |
| cpu\_insert\_prune\_prepare\_ns            | 27.055 ms   | 0.35%  |
| cpu\_insert\_preprune\_sort\_ns            | 12.008 ms   | 0.15%  |
| cpu\_insert\_init\_ns                      | 10.087 ms   | 0.13%  |
| cpu\_cache\_lookup\_ns                     | 1.220 ms    | 0.02%  |
| cpu\_insert\_candidate\_sort\_ns           | 0.595 ms    | 0.01%  |
| cpu\_insert\_neighbor\_prepare\_ns         | 0.196 ms    | 0.00%  |
| cpu\_insert\_quantize\_prepare\_ns         | 0.064 ms    | 0.00%  |

- cpu\_ns 内部热点：占比最高的几项是 `cpu_insert_filter_ns`（34.17%）、`cpu_insert_stage_candidates_ns`（25.23%）、`cpu_insert_runtime_overhead_ns`（16.88%）。

#### gpu\_ns

| 部分                                  | 时间            | 占比     |
| ----------------------------------- | ------------- | ------ |
| gpu\_insert\_overflow\_prune\_ns    | 322713.594 ms | 81.48% |
| gpu\_insert\_distance\_ns           | 37439.697 ms  | 9.45%  |
| gpu\_insert\_prune\_ns              | 27326.463 ms  | 6.90%  |
| gpu\_insert\_overflow\_distance\_ns | 7934.128 ms   | 2.00%  |
| gpu\_insert\_quantize\_ns           | 638.367 ms    | 0.16%  |

- gpu\_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（81.48%）、`gpu_insert_distance_ns`（9.45%）、`gpu_insert_prune_ns`（6.90%）。

#### rdma\_ns

| 部分                                | 时间           | 占比     |
| --------------------------------- | ------------ | ------ |
| rdma\_vector\_fetch\_ns           | 14895.473 ms | 35.60% |
| rdma\_neighbor\_fetch\_ns         | 13581.942 ms | 32.46% |
| rdma\_overflow\_vec\_fetch\_ns    | 4505.047 ms  | 10.77% |
| rdma\_neighbor\_lock\_ns          | 2712.745 ms  | 6.48%  |
| rdma\_pruned\_neighbor\_write\_ns | 1557.265 ms  | 3.72%  |
| rdma\_neighbor\_list\_read\_ns    | 1463.065 ms  | 3.50%  |
| rdma\_neighbor\_node\_read\_ns    | 1462.923 ms  | 3.50%  |
| rdma\_neighbor\_unlock\_ns        | 1216.604 ms  | 2.91%  |
| rdma\_candidate\_fetch\_ns        | 285.322 ms   | 0.68%  |
| rdma\_neighbor\_list\_write\_ns   | 82.728 ms    | 0.20%  |
| rdma\_new\_node\_write\_ns        | 30.434 ms    | 0.07%  |
| rdma\_alloc\_ns                   | 26.080 ms    | 0.06%  |
| rdma\_medoid\_ptr\_ns             | 23.185 ms    | 0.06%  |

- rdma\_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（35.60%）、`rdma_neighbor_fetch_ns`（32.46%）、`rdma_overflow_vec_fetch_ns`（10.77%）。

#### transfer\_ns

| 部分                                         | 时间           | 占比     |
| ------------------------------------------ | ------------ | ------ |
| transfer\_distance\_d2h\_ns                | 20636.406 ms | 62.52% |
| transfer\_overflow\_prune\_d2h\_ns         | 4132.268 ms  | 12.52% |
| transfer\_overflow\_dist\_d2h\_ns          | 3003.723 ms  | 9.10%  |
| transfer\_overflow\_prune\_inputs\_h2d\_ns | 2321.165 ms  | 7.03%  |
| transfer\_overflow\_query\_h2d\_ns         | 1327.504 ms  | 4.02%  |
| transfer\_overflow\_candidate\_h2d\_ns     | 1223.649 ms  | 3.71%  |
| transfer\_quantize\_d2h\_ns                | 219.612 ms   | 0.67%  |
| transfer\_prune\_d2h\_ns                   | 83.941 ms    | 0.25%  |
| transfer\_insert\_query\_h2d\_ns           | 30.702 ms    | 0.09%  |
| transfer\_prune\_h2d\_ns                   | 27.065 ms    | 0.08%  |

- transfer\_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（62.52%）、`transfer_overflow_prune_d2h_ns`（12.52%）、`transfer_overflow_dist_d2h_ns`（9.10%）。

### Latency

| 延迟字段                   | 值             |
| ---------------------- | ------------- |
| end\_to\_end\_ns       | 478663.779 ms |
| mean\_end\_to\_end\_ns | 283.569 ms    |
| mean\_queue\_wait\_ns  | 0.004 ms      |
| mean\_service\_ns      | 283.564 ms    |
| p50\_end\_to\_end\_ns  | 297.550 ms    |
| p50\_service\_ns       | 297.547 ms    |
| p95\_end\_to\_end\_ns  | 404.967 ms    |
| p95\_service\_ns       | 404.962 ms    |
| p99\_end\_to\_end\_ns  | 480.027 ms    |
| p99\_service\_ns       | 480.023 ms    |
| queue\_wait\_ns        | 7.131 ms      |
| service\_ns            | 478656.648 ms |

### Counters

| 字段                                                 | 值                 |
| -------------------------------------------------- | ----------------- |
| overflow\_prune\_global\_load\_bytes\_upper\_bound | 1,717,523,808,000 |
| rdma\_read\_bytes                                  | 81,342,533,454    |
| vector\_rdma\_bytes                                | 80,794,704,896    |
| h2d\_bytes                                         | 27,273,096,176    |
| neighbor\_rdma\_bytes                              | 547,813,206       |
| overflow\_prune\_pair\_checks\_upper\_bound        | 209,352,000       |
| d2h\_bytes                                         | 103,487,904       |
| rdma\_write\_bytes                                 | 64,800,046        |
| rdma\_read\_ops                                    | 21,862,497        |
| vector\_rdma\_read\_ops                            | 19,724,854        |
| overflow\_prune\_candidates                        | 6,542,250         |
| overflow\_prune\_kernel\_threads                   | 6,542,250         |
| neighbor\_rdma\_read\_ops                          | 2,135,724         |
| l2\_kernels                                        | 806,347           |
| lock\_attempts                                     | 441,666           |
| cas\_failures                                      | 347,372           |
| lock\_retries                                      | 347,372           |
| rdma\_write\_ops                                   | 321,279           |
| prune\_kernels                                     | 102,552           |
| overflow\_prune\_kernel\_blocks                    | 100,650           |
| overflow\_prunes                                   | 100,650           |
| vector\_rdma\_read\_avg\_bytes                     | 4,096.086333      |
| rdma\_read\_avg\_bytes                             | 3,720.642407      |
| cache\_hits                                        | 1,919             |
| remote\_allocations                                | 1,897             |
| neighbor\_rdma\_read\_avg\_bytes                   | 256.5             |
| rdma\_write\_avg\_bytes                            | 201.693998        |
| overflow\_prune\_avg\_candidates                   | 65                |
| overflow\_prune\_avg\_kernel\_threads              | 65                |
| overflow\_prune\_max\_candidates                   | 65                |
| overflow\_prune\_max\_kernel\_threads              | 65                |
| overflow\_prune\_avg\_kernel\_blocks               | 1                 |
| cache\_misses                                      | 0                 |
| exact\_reranks                                     | 0                 |
| gpu\_rabitq\_cache\_duplicate\_fills               | 0                 |
| gpu\_rabitq\_cache\_fallback\_batches              | 0                 |
| gpu\_rabitq\_cache\_fill\_bytes                    | 0                 |
| gpu\_rabitq\_cache\_fills                          | 0                 |
| gpu\_rabitq\_cache\_hits                           | 0                 |
| gpu\_rabitq\_cache\_misses                         | 0                 |
| neighbor\_cache\_hits                              | 0                 |
| neighbor\_cache\_misses                            | 0                 |
| query\_host\_staging\_fallback\_bytes              | 0                 |
| query\_rdma\_to\_staging\_bytes                    | 0                 |
| rabitq\_kernels                                    | 0                 |
| rabitq\_rdma\_bytes                                | 0                 |
| rabitq\_rdma\_read\_avg\_bytes                     | 0                 |
| rabitq\_rdma\_read\_ops                            | 0                 |
| visited\_neighborlists                             | 0                 |
| visited\_nodes                                     | 0                 |

## QUERY 分析

- 操作数：**26,742**
- 平均端到端延迟：**17.842 ms**
- P50 端到端延迟：**16.193 ms**
- P95 端到端延迟：**29.049 ms**
- P99 端到端延迟：**37.063 ms**

### 一级 Breakdown 占比

| 部分           | 时间            | 占比     |
| ------------ | ------------- | ------ |
| gpu\_ns      | 288813.828 ms | 60.54% |
| transfer\_ns | 113812.896 ms | 23.86% |
| cpu\_ns      | 46956.075 ms  | 9.84%  |
| rdma\_ns     | 27461.171 ms  | 5.76%  |

- query 一级热点：占比最高的几项是 `gpu_ns`（60.54%）、`transfer_ns`（23.86%）、`cpu_ns`（9.84%）。

### Sub Breakdown 细分占比

#### cpu\_ns

| 部分                                | 时间           | 占比     |
| --------------------------------- | ------------ | ------ |
| cpu\_query\_filter\_ns            | 21555.766 ms | 45.91% |
| cpu\_query\_runtime\_overhead\_ns | 19400.969 ms | 41.32% |
| cpu\_query\_beam\_update\_ns      | 2812.073 ms  | 5.99%  |
| cpu\_cache\_lookup\_ns            | 1279.700 ms  | 2.73%  |
| cpu\_query\_select\_ns            | 781.394 ms   | 1.66%  |
| cpu\_query\_finalize\_ns          | 505.380 ms   | 1.08%  |
| cpu\_query\_result\_ids\_ns       | 498.864 ms   | 1.06%  |
| cpu\_query\_beam\_sort\_ns        | 103.898 ms   | 0.22%  |
| cpu\_query\_rerank\_collect\_ns   | 14.614 ms    | 0.03%  |
| cpu\_query\_rerank\_update\_ns    | 3.416 ms     | 0.01%  |

- cpu\_ns 内部热点：占比最高的几项是 `cpu_query_filter_ns`（45.91%）、`cpu_query_runtime_overhead_ns`（41.32%）、`cpu_query_beam_update_ns`（5.99%）。

#### gpu\_ns

| 部分                       | 时间            | 占比     |
| ------------------------ | ------------- | ------ |
| gpu\_query\_distance\_ns | 285132.807 ms | 98.73% |
| gpu\_query\_prepare\_ns  | 2134.833 ms   | 0.74%  |
| gpu\_query\_rerank\_ns   | 1546.187 ms   | 0.54%  |

- gpu\_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.73%）、`gpu_query_prepare_ns`（0.74%）、`gpu_query_rerank_ns`（0.54%）。

#### rdma\_ns

| 部分                        | 时间           | 占比     |
| ------------------------- | ------------ | ------ |
| rdma\_rabitq\_fetch\_ns   | 15853.537 ms | 57.73% |
| rdma\_neighbor\_fetch\_ns | 10036.488 ms | 36.55% |
| rdma\_rerank\_fetch\_ns   | 1342.687 ms  | 4.89%  |
| rdma\_medoid\_ptr\_ns     | 228.459 ms   | 0.83%  |

- rdma\_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（57.73%）、`rdma_neighbor_fetch_ns`（36.55%）、`rdma_rerank_fetch_ns`（4.89%）。

#### transfer\_ns

| 部分                          | 时间            | 占比     |
| --------------------------- | ------------- | ------ |
| transfer\_distance\_d2h\_ns | 112535.017 ms | 98.88% |
| transfer\_rerank\_d2h\_ns   | 847.057 ms    | 0.74%  |
| transfer\_query\_h2d\_ns    | 430.822 ms    | 0.38%  |

- transfer\_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（98.88%）、`transfer_rerank_d2h_ns`（0.74%）、`transfer_query_h2d_ns`（0.38%）。

### Latency

| 延迟字段                   | 值             |
| ---------------------- | ------------- |
| end\_to\_end\_ns       | 477142.642 ms |
| mean\_end\_to\_end\_ns | 17.842 ms     |
| mean\_queue\_wait\_ns  | 0.004 ms      |
| mean\_service\_ns      | 17.839 ms     |
| p50\_end\_to\_end\_ns  | 16.193 ms     |
| p50\_service\_ns       | 16.190 ms     |
| p95\_end\_to\_end\_ns  | 29.049 ms     |
| p95\_service\_ns       | 29.045 ms     |
| p99\_end\_to\_end\_ns  | 37.063 ms     |
| p99\_service\_ns       | 37.060 ms     |
| queue\_wait\_ns        | 98.672 ms     |
| service\_ns            | 477043.970 ms |

### Counters

| 字段                                                 | 值              |
| -------------------------------------------------- | -------------- |
| rdma\_read\_bytes                                  | 22,349,672,510 |
| vector\_rdma\_bytes                                | 11,568,695,520 |
| query\_rdma\_to\_staging\_bytes                    | 11,479,547,904 |
| gpu\_rabitq\_cache\_fill\_bytes                    | 10,347,088,960 |
| rabitq\_rdma\_bytes                                | 10,344,029,800 |
| d2h\_bytes                                         | 573,094,936    |
| neighbor\_rdma\_bytes                              | 436,713,822    |
| gpu\_rabitq\_cache\_hits                           | 120,591,058    |
| h2d\_bytes                                         | 119,562,240    |
| rdma\_read\_ops                                    | 24,448,426     |
| gpu\_rabitq\_cache\_fills                          | 19,898,248     |
| gpu\_rabitq\_cache\_misses                         | 19,892,365     |
| rabitq\_rdma\_read\_ops                            | 19,892,365     |
| visited\_neighborlists                             | 3,999,720      |
| rabitq\_kernels                                    | 3,921,685      |
| neighbor\_cache\_hits                              | 3,148,102      |
| vector\_rdma\_read\_ops                            | 2,824,302      |
| neighbor\_rdma\_read\_ops                          | 1,702,588      |
| neighbor\_cache\_misses                            | 851,294        |
| cache\_hits                                        | 298,817        |
| exact\_reranks                                     | 29,194         |
| cache\_misses                                      | 22,164         |
| vector\_rdma\_read\_avg\_bytes                     | 4,096.125528   |
| rdma\_read\_avg\_bytes                             | 914.155885     |
| rabitq\_rdma\_read\_avg\_bytes                     | 520            |
| neighbor\_rdma\_read\_avg\_bytes                   | 256.5          |
| cas\_failures                                      | 0              |
| gpu\_rabitq\_cache\_duplicate\_fills               | 0              |
| gpu\_rabitq\_cache\_fallback\_batches              | 0              |
| l2\_kernels                                        | 0              |
| lock\_attempts                                     | 0              |
| lock\_retries                                      | 0              |
| overflow\_prune\_avg\_candidates                   | 0              |
| overflow\_prune\_avg\_kernel\_blocks               | 0              |
| overflow\_prune\_avg\_kernel\_threads              | 0              |
| overflow\_prune\_candidates                        | 0              |
| overflow\_prune\_global\_load\_bytes\_upper\_bound | 0              |
| overflow\_prune\_kernel\_blocks                    | 0              |
| overflow\_prune\_kernel\_threads                   | 0              |
| overflow\_prune\_max\_candidates                   | 0              |
| overflow\_prune\_max\_kernel\_threads              | 0              |
| overflow\_prune\_pair\_checks\_upper\_bound        | 0              |
| overflow\_prunes                                   | 0              |
| prune\_kernels                                     | 0              |
| query\_host\_staging\_fallback\_bytes              | 0              |
| rdma\_write\_avg\_bytes                            | 0              |
| rdma\_write\_bytes                                 | 0              |
| rdma\_write\_ops                                   | 0              |
| remote\_allocations                                | 0              |
| visited\_nodes                                     | 0              |

## System Counters

| 字段                 | 值               |
| ------------------ | --------------- |
| rdma\_read\_bytes  | 103,692,205,964 |
| h2d\_bytes         | 27,392,658,416  |
| d2h\_bytes         | 676,582,840     |
| rdma\_write\_bytes | 64,800,046      |

## Insert / Query 对比

| 类别           | Insert 占比 | Query 占比 |
| ------------ | --------- | -------- |
| cpu\_ns      | 1.62%     | 9.84%    |
| gpu\_ns      | 82.74%    | 60.54%   |
| rdma\_ns     | 8.74%     | 5.76%    |
| transfer\_ns | 6.90%     | 23.86%   |

- Insert 最大部分是 **gpu\_ns**，占 **82.74%**。
- Query 最大部分是 **gpu\_ns**，占 **60.54%**。
- Insert 更偏向 GPU 计算密集。
- Insert 的 RDMA 成本同样非常显著。

