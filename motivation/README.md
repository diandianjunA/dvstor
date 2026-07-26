# GPU 图搜索批量同步扩展动机实验

## 1. 代码确认后的真实执行流程

本实验以当前代码为准，不假设每个父节点拥有独立 completion。

1. `process_query()` 的 traversal CTA 在 beam 中从低 rank 到高 rank 扫描未扩展项，最多选择
   `min(prefetch_depth, max_expansions - expansions)` 个父节点。父节点在发起读取前立即被置为
   `expanded=1`；实验代码没有改变此顺序或语义。
2. `fetch_graph_records_batch()` 为每个父节点准备一个 graph scratch slot、目标 shard、远端
   offset 和本地 IOVA。
3. 每次 snapshot attempt（最多 3 次）按 shard 聚合父节点。每个非空 shard 形成一个
   `DirectBatchDescriptor`，被投递到对应的 exclusive-QP owner queue。
4. owner warp 可合并多个已排队 descriptor，一次连续发布各 descriptor 的 READ WQE。一个
   shard batch 内只有最后一条 READ（需要 dump 时为 dump WQE）请求成功 CQE，其余 READ 只请求
   error CQE。因此现有数据路径只能观察 shard batch completion，不能观察单父节点或单 WQE
   completion。
5. 查询 CTA 先投递本轮所有非空 shard batch，再逐 shard 调用 `wait_direct_batch()`。所有
   shard status 都结束后经过 CTA barrier，才开始 graph record checksum/incarnation 验证。
   torn snapshot 会只重读失败记录，并再次形成 shard batch。
6. `fetch_graph_records_batch()` 完成全部验证后，traversal 才统一读取邻接表。结果按
   `persistent_score_chunk_capacity()` 分块：解码邻居、visited check/update、动态 PQ
   访问与 PQ scoring、merge 回 beam。该轮所有 chunk 完成后，才选择下一轮父节点。
7. traversal 完成后，对最终 beam 执行 exact record read/rerank。

所以当前关键同步点真实存在：每个 snapshot attempt 内是“全部 shard batch 完成 → 验证”，
每个搜索轮次内是“全部 graph fetch/验证完成 → 统一 decode/score/merge → 下一轮”。此外，
同一 shard 内多个父节点共享一个可观察 completion，不能据此推断父节点之间的完成离散。

相关实现：

- `src/gpu_search/persistent_kernel/query_traversal.cuh`
- `src/gpu_search/persistent_kernel/rdma_read.cuh`
- `src/gpu_search/persistent_kernel/runtime.cuh`

## 2. 新增埋点与准确口径

运行参数：

```text
--query-rdma-trace-mode=off|sampled|full
--query-rdma-trace-sample-rate=N
--query-rdma-trace-output=PATH
--query-rdma-trace-events-per-query=N
```

- `off`：不分配详细 trace arena，不执行 detailed timestamp，不拷贝或写文件。默认值。
- `sampled`：按 `request_id % sample_rate == 0` 采样。
- `full`：追踪全部查询，建议只用于单并发、小查询数。

详细事件使用每个 query slot 独占的预分配数组，没有热路径动态分配、文件 I/O 或跨查询全局
原子。查询结束后 completion thread 一次性拷贝该查询的事件到 CPU，输出 JSONL。容量不足时
保留前缀并设置 `overflow=1`；分析时不得静默丢弃 overflow 查询。

每个事件是 `(query, search_round, snapshot_attempt, target_shard)` 粒度，包含：

```text
request_id, search_round, snapshot_attempt, target_shard,
parent_count, bytes_per_parent,
issue_timestamp_ns, completion_timestamp_ns,
batch_process_start_timestamp_ns
```

`issue_timestamp_ns` 在查询 CTA 开始向 owner queue 投递该 shard descriptor 前取得，因此包含
可能的 owner-queue backpressure。`completion_timestamp_ns` 在 owner CTA CQ poll 返回后、
发布 status 前取得。它不是 host 提交时间，也不是 NIC 内部时间；它是当前软件接口可观察到的
最接近完成点。owner 可能一次处理多个 descriptor，较早 descriptor 的可观察完成时间会包含
该 owner submission group 的完成边界，这正是当前查询可见的同步语义。

详细时间戳使用 PTX `%globaltimer`，代码按纳秒记录，可跨 SM 比较。`clock64()` 只用于同一个
query CTA 内的 phase 相对时间，并使用启动时读取的 GPU clock kHz 转换。query CTA 在一次
`process_query()` 中不会迁移 CTA/SM；不对不同 CTA 的 `clock64()` 绝对值作比较。

始终汇总的 query phase 字段为：

```text
prepare
beam_selection
rdma_issue
rdma_wait
graph_validation
neighbor_decode
pq_score                  # 含动态 PQ access；dynamic_code_cycles 另列
visited
beam_merge
exact
other                     # gpu total 减去上述互斥大阶段，报告端可派生
```

原有 `graph_cycles/score_cycles/beam_cycles/exact_cycles` 保留兼容；新增字段初始化为 0。这里的
`rdma_wait` 是所有 shard descriptor 投递后的显式 wait wall time。部分请求在 issue 阶段已完成
时，其等待不会重复计入；这与关键路径分解一致。详细 trace 的 unused-ready 才衡量已完成数据
因 barrier 未被消费的 request-time。

## 3. 指标计算

`analyze_rdma_trace.py` 对每个 `(query, round, snapshot_attempt)` 计算 shard-batch completion
分布：

```text
min, median, p90, max
max - median
max - min
```

completion 以该 attempt 最早 issue 为共同零点。对于 shard batch `s`，可观察 barrier waste：

```text
waste_s = batch_process_start - completion_s
```

请求数加权口径把一个 shard batch 的 completion 赋给其中全部父节点（这是近似上限/下限都不
保证的 observable-granularity 估计，不冒充 per-parent measurement）：

```text
weighted_waste = sum_s(parent_count_s * waste_s)
B = sum_s(parent_count_s)
normalized_barrier_waste =
  weighted_waste / (B * (batch_process_start - min(issue_s)))
```

分析结果同时给出每轮平均/最大 unused-ready、每查询累计 request-ns、全局加权总量及相对 graph
read window 的比例。`unused_ready_ns_requests / query_gpu_ns` 也会输出，但其量纲是
“并行请求时间 / wall time”，不能解释为 wall-clock 百分比；是否能转化为可回收 query latency
仍需未来异步算法实验验证。另输出
`mean_parent_unused_over_query_gpu_time`，表示平均每个父节点的 unused-ready 占 query GPU
wall time 的比例。

## 4. prefetch depth sweep

代码约束为 `1 <= gpu_graph_prefetch_depth <= 32`，因此要求的 `1,2,4,8,16,32` 全部可用。
配置位于 `motivation/configs/`。

正式 sweep 默认测试四个闭环并发度 `1,8,64,256`，每个 depth 保持数据、beam、max
expansions、QP 数、warmup、measurement 和 recall 查询数相同。报告已有：

- query QPS、平均/P50/P95/P99/P999 latency、Recall@10；
- GPU query residence、prepare/各细分 phase；
- 每查询 graph parent reads、retry reads、总 RDMA ops/bytes；
- shard batch 数、graph round 数、exact/dynamic PQ reads。

详细 trace 补充每轮实际 parent count、shard batch count、completion spread 和 barrier waste。
`gpu_query_residence_ns` 不是硬件 GPU utilization；若需要真正 SM busy 百分比，应另做
Nsight/CUPTI profiler run，并将其与无 profiler 的性能结果分开，避免把 profiler 开销混入
主结论。

比较时必须保留全部结果，包括低 Recall、overflow、retry 和异常运行。吞吐主实验必须使用
trace off；sampled/full 结果只用于机制分析和估计埋点开销。

## 5. 运行命令

先按现有流程在各存储节点启动 memory node。然后在 compute 节点仓库根目录执行：

```bash
cmake --build build -j --target dvstor_breakdown_benchmark

# 主性能实验：6 depths × 4 concurrencies，trace 完全关闭
./motivation/run_prefetch_sweep.sh

# shard-batch 离散与 barrier waste：6 depths × (1,64) concurrency
./motivation/run_trace_sweep.sh

# 小规模全量 trace sanity check（默认 depth=32、100 queries）
./motivation/run_full_trace_smoke.sh
```

可缩小或覆盖扫描：

```bash
CONCURRENCIES="1 64" DEPTHS="1 8 32" ./motivation/run_prefetch_sweep.sh
TRACE_MODE=full TRACE_SAMPLE_RATE=1 TRACE_CONCURRENCIES=1 \
  DEPTHS=32 ./motivation/run_trace_sweep.sh
DEPTH=8 FULL_TRACE_QUERIES=50 ./motivation/run_full_trace_smoke.sh
```

输出位于 `motivation/results/`。每个 trace 旁会生成 `rdma_trace.summary.json`。正式判读时先检查
`trace_overflow_queries == 0`；否则增加 `QUERY_RDMA_TRACE_EVENTS_PER_QUERY` 后重跑详细实验，
不能选择性删除 overflow 查询。主 sweep 完成后还会生成
`motivation/results/sweep/prefetch_sweep.csv`；也可手工执行
`./motivation/summarize_prefetch_sweep.py` 重新汇总全部未过滤报告。

## 6. Feedback-Hunger 动态扩展

动态扩展实现、验证和当前负性能结论见
`motivation/FEEDBACK_HUNGER_REPORT.md`。完整 fixed/dynamic A/B：

```bash
./motivation/run_feedback_hunger_ab.sh
```
