# 第 27 课：性能统计与 breakdown 报告

## 本课目标

本课学习项目的性能统计体系。学完后，你需要能够：

1. 区分 `ThreadStatistics` 和 per-request `breakdown::Sample`。
2. 理解 CPU、GPU、RDMA、transfer 统计的来源。
3. 读懂 benchmark JSON 中的 counters、latency、breakdown、utilization。
4. 判断统计代码可能引入的偏差。

代码入口：

- `src/common/statistics.hh`
- `src/service/breakdown/sample.hh`
- `src/service/breakdown/names.hh`
- `src/service/breakdown/aggregate.hh`
- `src/service/breakdown/json.hh`
- `src/service/breakdown/text.hh`
- `src/service/compute_service.cc`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`

## 1. 两套统计模型

项目中有两套互补统计：

1. `statistics::ThreadStatistics`
   - 存在于每个 `ComputeThread`。
   - 是线程级累计 counter。
   - 记录 bytes、ops、kernel count、visited nodes、queue wait 等。
   - 适合看总量。

2. `service::breakdown::Sample`
   - 每个 query/insert request 一个 sample。
   - 记录单请求时间窗口。
   - 开始时复制 thread counters。
   - 结束时再复制 thread counters。
   - 通过 diff 得到单请求 counter delta。
   - 适合做 latency breakdown 和 per-request 聚合。

这两套统计的关系：

- `ThreadStatistics` 是原始累计源。
- `Sample` 是 request 层的快照和细粒度时间片。
- `Aggregate` 把多个 sample 汇总。
- JSON/Text 把 aggregate 输出成报告。

## 2. ThreadStatistics 结构

`src/common/statistics.hh` 中 `ThreadStatistics` 包含大量字段，可分为几类：

1. 距离计算：
   - `distcomps`
   - `query_distcomps`
   - `build_distcomps`

2. RDMA bytes/ops：
   - `rdma_reads_in_bytes`
   - `rdma_writes_in_bytes`
   - `rdma_read_ops`
   - `rdma_write_ops`
   - query/build 分拆字段

3. neighbor/vector read：
   - `query_neighbor_rdma_reads_in_bytes`
   - `query_vector_rdma_reads_in_bytes`
   - `query_neighbor_rdma_read_ops`
   - `query_vector_rdma_read_ops`

4. vector batch planner：
   - `vector_rdma_batch_calls`
   - `vector_rdma_chunks`
   - `vector_rdma_active_nodes`
   - `vector_rdma_active_qps`
   - `vector_rdma_chain_wrs`
   - `vector_rdma_max_chain_wrs`
   - `vector_rdma_qp_high_water_wrs`
   - `vector_rdma_credit_waits`
   - `vector_rdma_credit_wait_ns`

5. GPU transfer/kernel：
   - `query_h2d_bytes`
   - `query_d2h_bytes`
   - `build_h2d_bytes`
   - `build_d2h_bytes`
   - `build_l2_kernels`
   - `build_prune_kernels`

6. RaBitQ：
   - `query_exact_reranks`
   - `query_rabitq_l0_candidates`
   - `query_rabitq_cache_misses`
   - `query_rabitq_l1_candidates`
   - `query_rabitq_l2_candidates`
   - `query_rabitq_forced_widen`
   - `query_rabitq_audit_expansions`

7. credit-aware expansion：
   - `query_credit_rounds`
   - `query_credit_expansions_issued`
   - `query_credit_precommit_expansions`
   - `query_credit_grow_events`
   - `query_credit_shrink_events`
   - `query_credit_credit_stalls`
   - `query_credit_cost_guard_events`

8. overflow prune：
   - `build_overflow_prunes`
   - `build_overflow_prune_candidates`
   - `build_overflow_prune_max_candidates`
   - kernel blocks/threads upper bound

9. processed 与 queue：
   - `processed_queries`
   - `processed_inserts`
   - `query_queue_wait_ns`
   - `insert_queue_wait_ns`

10. search graph：
   - `visited_nodes`
   - `visited_nodes_l0`
   - `visited_neighborlists`

这些字段不是全部都有细粒度时间，但它们是解释瓶颈的重要 counters。

## 3. Sample 的生命周期

`service::breakdown::Sample` 包含：

- operation：query 或 insert。
- `collect_fine_grained_breakdown`。
- `enqueued_at`
- `dequeued_at`
- `started_at`
- `finished_at`
- category/subcategory ns。
- start/end counters。
- queue/service/end-to-end ns。
- RDMA wait ns。
- GPU kernel ns。
- lock/cas counters。
- storage owner anchor counters。

创建位置：

- `ComputeService::search_local_result()`
- `ComputeService::search_local_raw_result()`
- insert/upsert/delete 相关路径。

开始位置：

- scheduler 从 queue 取 request 后调用 `mark_started(...)`。

结束位置：

- coroutine done 后调用 `mark_finished(...)`。

采样完成后：

- query sample 会 push 到 `completed_query_samples_`。
- insert sample 会 push 到 `completed_insert_samples_`。

`ComputeService::collect_breakdown_report()` 会遍历这两个 vector，汇总成 report。

## 4. category 与 subcategory

`src/service/breakdown/names.hh` 定义了四个大类：

- `cpu_ns`
- `gpu_ns`
- `rdma_ns`
- `transfer_ns`

subcategory 更细，例如 query：

- `cpu_query_select_ns`
- `cpu_query_filter_ns`
- `cpu_query_rabitq_gate_ns`
- `cpu_query_stage_candidates_ns`
- `cpu_query_beam_update_ns`
- `cpu_query_result_ids_ns`
- `gpu_query_distance_ns`
- `rdma_neighbor_fetch_ns`
- `rdma_vector_fetch_ns`
- `transfer_query_h2d_ns`
- `transfer_distance_d2h_ns`

insert：

- `cpu_insert_init_ns`
- `cpu_insert_select_ns`
- `cpu_insert_candidate_sort_ns`
- `cpu_insert_prune_prepare_ns`
- `gpu_insert_distance_ns`
- `gpu_insert_prune_ns`
- `rdma_alloc_ns`
- `rdma_new_node_write_ns`
- `rdma_neighbor_lock_ns`
- `rdma_neighbor_list_write_ns`
- `transfer_insert_query_h2d_ns`
- `transfer_prune_h2d_ns`

storage-owner：

- `cpu_storage_owner_queue_wait_ns`
- `cpu_storage_owner_search_ns`
- `cpu_storage_owner_prune_ns`
- `cpu_storage_owner_remote_reverse_ns`
- `rdma_storage_owner_send_ns`
- `rdma_storage_owner_search_neighbor_read_ns`

每个 subcategory 通过 `parent_category(...)` 映射到大类。

## 5. counter diff

`diff_thread_counters(end, start, operation)` 根据 operation 决定读取 query 字段还是 build/insert 字段。

对于 query：

- 使用 `end.query_rdma_reads_in_bytes - start.query_rdma_reads_in_bytes`。
- 使用 query neighbor/vector fields。
- 使用 query h2d/d2h。
- 使用 RaBitQ 和 credit-aware query counters。
- visited nodes 合并 `visited_nodes` 和 `visited_nodes_l0`。

对于 insert：

- 使用 build rdma read/write。
- 使用 build neighbor/vector fields。
- 使用 build h2d/d2h。
- 使用 l2/prune kernel。
- 使用 remote allocations。
- 使用 overflow prune counters。

这说明代码把 insert 统计称为 build 统计。阅读 report 时不要被字段名误导：`build_*` 在在线 insert 中也会增长。

## 6. add_subcategory

`Sample::add_subcategory(...)` 做两件事：

1. 增加该 subcategory 的 ns。
2. 增加 parent category 的 ns。

如果 operation 是 query 且 parent 是 RDMA，则额外增加：

```cpp
rdma_wait_ns += ns;
```

代码注释明确说明：

- RDMA wait 是软件等待 completion 的代理指标。
- 它不是物理 NIC line-rate utilization。

这个区别非常重要。看到 `rdma_completion_wait_ratio` 高，说明线程等待 RDMA completion 时间多，不等于网卡带宽打满。

## 7. GPU kernel time

`Sample::add_gpu_kernel_time(ns)`：

- 只对 query 增加 `gpu_kernel_ns`。
- 注释说明 GPU kernel time 是 CUDA event 包围距离 kernel 的执行时间。
- 不包括 CPU launch、stream queueing、D2H copy。

因此：

- `gpu_ns` category 可能包含 CPU 侧准备或等待范围。
- `gpu_kernel_busy_ns` 更接近 device kernel 执行时间。
- 二者不能混为一谈。

## 8. Aggregate 汇总

`add_sample(Aggregate&, Sample&)`：

1. 如果 sample 未 finished，跳过。
2. 增加 count。
3. 累计 queue/service/end-to-end。
4. 记录 latency 数组，用于 percentile。
5. 如果 sample 不收集 fine-grained，提前返回。
6. 累计 category/subcategory。
7. 计算 counter delta。
8. 累计 counters。
9. 维护 max 类字段。
10. 累计 lock/cas。

`Aggregate` 还提供：

- `measured_total_ns()`
- `cpu_other_ns()`

`cpu_other_ns()` 的设计是：

- 用 service total 减去 gpu/rdma/transfer 得到 CPU total。
- 再减去显式 CPU subcategory。
- 剩下的是 runtime overhead。

因此 JSON 中可能看到：

- `cpu_query_runtime_overhead_ns`
- `cpu_insert_runtime_overhead_ns`

它表示没有被细分埋点覆盖的 CPU 时间。

## 9. JSON 输出

`aggregate_to_json()` 输出：

1. operation/count。
2. latency：
   - total queue wait
   - total service
   - total end-to-end
   - mean queue wait
   - mean service
   - mean end-to-end
   - p50/p95/p99 end-to-end
   - p50/p95/p99 service

3. fine_grained_breakdown_observed。

4. utilization：
   - device utilization observed
   - gpu kernel busy ns/ratio
   - gpu kernel idle ratio
   - rdma completion wait ns/ratio
   - rdma payload bytes per service second

5. breakdown：
   - cpu_ns
   - gpu_ns
   - rdma_ns
   - transfer_ns

6. sub_breakdown：
   - 按大类嵌套每个 subcategory。

7. counters：
   - RDMA bytes/ops/avg bytes。
   - vector batch stats。
   - H2D/D2H bytes。
   - kernel counts。
   - RaBitQ stats。
   - credit-aware stats。
   - visited nodes。
   - overflow prune。
   - lock/cas。

这些字段构成后续优化的主要观测面。

## 10. 源码埋点位置

从 `rg` 搜索可看到：

- `src/vamana/vamana_search.ipp` 中有：
  - `cpu_query_select`
  - `cpu_query_filter`
  - `cpu_query_rabitq_gate`
  - `rdma_vector_fetch`
  - `gpu_query_distance`
  - `transfer_distance_d2h`
  - `cpu_query_beam_update`
  - `cpu_query_beam_sort`
  - `cpu_query_result_ids`

- `src/vamana/vamana_insert.ipp` 中有：
  - `rdma_medoid_ptr`
  - `rdma_alloc`
  - `rdma_new_node_write`
  - `rdma_medoid_update`
  - `cpu_insert_select`
  - `rdma_neighbor_fetch`
  - `rdma_vector_fetch`
  - `gpu_insert_distance`
  - `gpu_insert_prune`
  - `rdma_neighbor_lock`
  - `rdma_neighbor_list_read`
  - `rdma_neighbor_list_write`
  - overflow prune 相关 subcategory

- storage-owner runtime 中有：
  - queue wait
  - search
  - prune
  - write node
  - local/remote reverse
  - response send

学习时要把 report 中的字段反查到这些源码埋点，否则很容易只看 JSON 不知道瓶颈在哪里。

## 11. 统计偏差

当前统计体系可能有这些偏差：

1. fine-grained breakdown 自身有开销：
   - 每个埋点读取 clock。
   - sample 保存 arrays。
   - counters diff。

2. request sample 与 thread counter 的关系依赖 coroutine slot：
   - 同一 thread 上多个 coroutine 交错时，counter diff 可能包含其他 coroutine 的增量。
   - 如果 counters 是 thread 累计而非 coroutine 累计，这种偏差需要特别评估。

3. RDMA wait 不是硬件 utilization：
   - 只能解释软件等待 completion。

4. GPU kernel busy 不含 launch 和 copy：
   - 不能单独表示 GPU 总路径成本。

5. CPU other 是差值：
   - 可能包含未埋点 CPU、调度开销、统计误差。

6. p50/p95/p99 的 percentile 实现：
   - `idx = percentile * (n - 1)` 后取整。
   - 没有插值。

7. breakdown disabled 时：
   - 只有 count/latency 可能可用。
   - counters 和 sub_breakdown 不完整。

## 12. 优化前后必须观测的指标

查询优化：

- p50/p95/p99 end-to-end。
- p50/p95/p99 service。
- query queue wait。
- RDMA read bytes/ops。
- neighbor/vector rdma bytes。
- vector_rdma_batch_calls。
- vector_rdma_mean_active_nodes_per_batch。
- vector_rdma_mean_active_qps_per_batch。
- vector_rdma_credit_wait_ns。
- GPU kernel busy ratio。
- H2D/D2H bytes。
- visited nodes。
- exact reranks。
- recall@k。

插入优化：

- insert service latency。
- remote allocations。
- RDMA write/read ops。
- gpu insert distance/prune ns。
- lock attempts/retries/CAS failures。
- overflow prunes。
- neighbor lock/list read/list write ns。
- storage-owner queue wait。
- remote reverse update ns。

RaBitQ 优化：

- rabitq l0/l1/l2 candidates。
- cache misses。
- forced widen。
- audit expansions/candidates。
- exact vector reads。
- recall。

credit-aware 优化：

- credit rounds。
- issue k。
- precommit ratio。
- grow/shrink events。
- credit stalls。
- cost guard events。
- no progress rounds。

## 13. 设计异味

1. Thread counters 和 request sample 不是严格隔离：
   - 并发 coroutine 下可能混入其他请求 counter。

2. 统计字段命名混用 build/insert：
   - 对新读者不友好。

3. breakdown header-only 较多：
   - 编译耦合。

4. category/subcategory 枚举不断增长：
   - 容易变成全局杂物。

5. report JSON 结构庞大：
   - 缺少稳定 schema 文档和版本。

6. device utilization 语义需要注释才能理解：
   - 对自动分析工具不友好。

## 14. 可验证问题

1. breakdown disabled：
   - report 是否仍有 latency/count。

2. 多 coroutine 并发：
   - 单请求 counter delta 是否可能大于实际请求产生的操作。

3. query-only ops mode：
   - report count 是否等于 measure_ops。

4. mixed workload：
   - query_breakdown count 是否等于 completed_reads。
   - insert_breakdown count 是否等于 completed writes 中会产生 insert sample 的部分。

5. GPU kernel busy ratio：
   - 打开/关闭 `observe_device_utilization` 是否改变输出。

6. CPU other：
   - 是否随埋点增加而下降。

## 15. 学习任务

1. 画一张 breakdown subcategory 到源码埋点位置的索引表。
2. 选一个 query report JSON，手工解释每个 counters 字段代表的源码路径。
3. 设计一个 microbenchmark，验证 thread counter diff 是否受同线程其他 coroutine 影响。
4. 为 report JSON 设计一个 schema version 字段和兼容策略。
5. 制定一份优化前后必须比较的指标清单，并按 query/insert/mixed 分类。

