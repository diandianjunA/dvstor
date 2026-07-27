# GPU 图搜索批量同步扩展动机实验

## Live-extent RDMA motivation

`LIVE_EXTENT_RDMA_MOTIVATION_REPORT.md` 检验一个存算分离特有的问题：存储记录为动态更新固定
预留的空槽，是否正在变成每次 GPU 查询都支付的网络流量。实验包含 live-degree byte oracle、
并发基线和真实 GPU-initiated one-sided RDMA payload sweep。

当前结论是：约 400–448 B 的**单次**图记录读取具有明确 transport 收益，值得继续做静态
端到端原型；每条记录先读 16 B header 再依赖式追读 body 在高并发下显著退化，不应继续。
端到端 A/B 完成后用 `python3 motivation/summarize_live_extent_ab.py` 严格成对汇总结果。
远端 storage nodes 启动后运行：

```bash
LIVE_EXTENT_CONFIG=motivation/configs/live_extent_rdma.env \
  ./motivation/run_live_extent_rdma_probe.sh
```

结果与完整限制见 `motivation/results/live_extent_rdma/` 和上述报告。

## Feedback-priced expansion motivation

`FEEDBACK_PRICING_MOTIVATION_REPORT.md` 检验 Beam old/new turnover 是否
包含足够的信息与性能空间，值得继续设计动态 batch controller。采样 trace 记录候选来源、
后续是否被选择以及每个父节点是否立刻贡献 Beam 入选子节点；离线 productive-suffix
oracle 给出一个刻意乐观的收益上限。

运行：

```bash
CONCURRENCIES="64 256" ./motivation/run_feedback_pricing_motivation.sh
```

## 1. 代码确认后的真实执行流程

本实验以当前代码为准，不假设每个父节点拥有独立 completion。

1. `process_query()` 的 traversal CTA 在 beam 中从低 rank 到高 rank 扫描未扩展项，最多选择
   `min(prefetch_depth, max_expansions - expansions)` 个父节点。父节点在发起读取前立即被置为
   `expanded=1`；实验代码没有改变此顺序或语义。
2. `fetch_graph_records_batch()` 为每个父节点准备一个 graph scratch slot、目标 shard、远端
   offset 和本地 IOVA。
3. 每次 snapshot attempt（最多 3 次）按 shard 聚合父节点。每个非空 shard 形成一个
   `DirectBatchDescriptor`，被投递到对应的 exclusive-QP owner queue。
4. owner warp 最多合并 8 个已排队 descriptor，一次连续发布其 READ WQE。整个 owner
   submission group 只有最后一条 READ（需要 dump 时为 dump WQE）请求成功 CQE；poll 到该
   CQE 后，owner 才依次发布组内所有 descriptor 的 completion/status。因此 trace 观察的是
   “shard descriptor 所属 owner submission group 的软件可见完成边界”，不能观察单父节点、
   单 WQE、单 descriptor 的物理完成时刻或 NIC 内部完成时刻。同一个 shard 内的多个父节点更
   完全不可区分。
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

每个事件是
`(query, route_attempt, search_round, snapshot_attempt, target_shard)` 粒度，包含：

```text
request_id, route_attempt, search_round, snapshot_attempt, target_shard,
parent_count, payload_bytes,
minimum_bytes_per_parent, maximum_bytes_per_parent,
issue_timestamp_ns, wait_phase_start_timestamp_ns,
completion_timestamp_ns,
batch_process_start_timestamp_ns
```

当前 RDMA trace schema 为 3。schema 1/2 的 graph shard event 使用统一的
`bytes_per_parent`；schema 3 允许同一 descriptor 内每条 graph READ 使用不同长度，
因此记录精确的 `payload_bytes` 以及最小/最大单父节点长度。分析脚本同时兼容两种
口径，不会用物理 record size 反推 live-extent 流量。

`issue_timestamp_ns` 在查询 CTA 开始向 owner queue 投递该 shard descriptor 前取得，因此包含
可能的 owner-queue backpressure。`wait_phase_start_timestamp_ns` 在本 attempt 的所有 shard
descriptor enqueue 完成后取得，是查询 CTA 最早能够消费已就绪 shard 的共同起点。
`completion_timestamp_ns` 在 owner CTA poll 到该 submission group 的最终 CQE 后、发布
descriptor status 前取得。它不是 host 提交时间，也不是 NIC 内部时间。`route_attempt` 防止
route 重新快照后的 round 编号从 0 开始而被错误合并。

详细时间戳使用 PTX `%globaltimer`，代码按纳秒记录，可跨 SM 比较。`clock64()` 只用于同一个
query CTA 内的 phase 相对时间，并使用启动时读取的 GPU clock kHz 转换。query CTA 在一次
`process_query()` 中不会迁移 CTA/SM；不对不同 CTA 的 `clock64()` 绝对值作比较。
本机 A800 实测 `%globaltimer` timestamp 的量化步长为 1024 ns，所以字段单位虽为 ns，低于
约 1 μs 的差异没有可解释精度。

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

`analyze_rdma_trace.py` 对每个
`(query, route_attempt, round, snapshot_attempt)` 计算可观察 completion 分布：

```text
min, median, p90, max
max - median
max - min
```

completion 以该 attempt 最早 issue 为共同零点。设 `I_s`、`W_0`、`C_s`、`C_max`、`P`
分别为 issue、统一 wait 起点、可观察 completion、最晚 completion 和 validation 开始。
分析器明确拆分：

```text
straggler_barrier_s = C_max - C_s
strict_wait_barrier_s = C_max - max(C_s, W_0)
post_completion_handoff = P - C_max
ready_until_process_s = P - C_s
```

其中 strict 口径不会把串行 issue/enqueue 期间已经发生的 completion 离散冒充为可执行机会。
请求数加权口径把一个 shard descriptor 的可观察 completion 赋给其中全部父节点：

```text
weighted_straggler_waste =
  sum_s(parent_count_s * (C_max - C_s))
B = sum_s(parent_count_s)
normalized_strict_wait_barrier_waste =
  sum_s(parent_count_s * (C_max - max(C_s, W_0)))
  / (B * (C_max - W_0))
```

所有 parent-weighted waste 的单位都是 `parent·ns`，只表示 ready work supply，不能冒充
query latency。每查询的 `sum(C_max-C_min)` 和
`sum(C_max-min(max(C_s,W_0)))` 只标记为顺序 attempt 的 observable/strict overlap-window
upper bound；它们不是已经测得的可回收延迟，更不是方案预期 speedup。

单 shard attempt 在现有接口下没有可比较的离散，分析器将其单独计数，并从 spread CDF 中排除，
而不是作为 0 混入。primary snapshot 与 retry 分层报告；overflow、非法 timestamp、不完整
group、缺 query record 和失败 query 全部进入 integrity 计数，不能静默过滤。

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

# shard-batch 离散与 barrier waste：6 depths × (1,8,64,256) concurrency
./motivation/run_trace_sweep.sh

# 小规模全量 trace sanity check（默认 depth=32、100 queries）
./motivation/run_full_trace_smoke.sh

# “正式工作乱序执行、搜索状态顺序提交”的批次屏障 motivation：
# trace-off 性能控制 + sampled mechanism trace，fixed + stable-run
./motivation/run_batch_barrier_motivation.sh
```

可缩小或覆盖扫描：

```bash
CONCURRENCIES="1 64" DEPTHS="1 8 32" ./motivation/run_prefetch_sweep.sh
TRACE_MODE=full TRACE_SAMPLE_RATE=1 TRACE_CONCURRENCIES=1 \
  DEPTHS=32 ./motivation/run_trace_sweep.sh
DEPTH=8 FULL_TRACE_QUERIES=50 ./motivation/run_full_trace_smoke.sh

# 只做端到端快速检查（不是正式数据）
QUICK=1 ./motivation/run_batch_barrier_motivation.sh

# 正式矩阵可显式覆盖；默认已经是 depth 1/8/16/32、
# concurrency 1/8/64/256、3 repetitions
DEPTHS="1 8 16 32" CONCURRENCIES="1 8 64 256" REPETITIONS=3 \
  ./motivation/run_batch_barrier_motivation.sh
```

输出位于 `motivation/results/`。每个 trace 旁会生成 `rdma_trace.summary.json` 和一份简短
Markdown；专用矩阵报告为 `motivation/results/batch_barrier/REPORT.md`。正式判读时先检查
所有 integrity 字段；overflow 时增加 `TRACE_EVENTS_PER_QUERY` 后完整重跑该格，不能选择性
删除 overflow 查询。主 sweep 完成后还会生成
`motivation/results/sweep/prefetch_sweep.csv`；也可手工执行
`./motivation/summarize_prefetch_sweep.py` 重新汇总全部未过滤报告。

专用报告预先注册了一个保守的 prototype screen，但它不是显著性检验，也不能替代未来原型的
wall-clock A/B。若多 shard coverage 很低、strict spread 很小，或提前 ready 的父节点不足一个
自然 CTA tile，实验应作为否定/不充分证据保留。特别是当前 SIFT100M 只有 5 个 shard；若大量
轮次只形成一个 shard descriptor，只能结论为“当前 completion 接口无法支持 shard-batch
乱序动机”，不能外推为 parent 级不存在离散。若要测 parent/WQE，必须改变 CQE 信号粒度并作为
独立 transport 实验，因为它会改变性能数据路径。

## 6. Feedback-Hunger 动态扩展

动态扩展实现、验证和当前负性能结论见
`motivation/FEEDBACK_HUNGER_REPORT.md`。完整 fixed/dynamic A/B：

```bash
./motivation/run_feedback_hunger_ab.sh
```

## 7. Beam merge policy A/B

设计、精确性证明、资源占用和最终性能结果见
`motivation/STABLE_RUN_BEAM_REPORT.md`。

固定扩展策略与同一 prefetch depth 下比较旧 merge 和 stable-run：

```bash
./motivation/run_beam_merge_ab.sh
```

可用 `PREFETCH_DEPTH` 和 `CONCURRENCIES` 缩小或覆盖矩阵：

```bash
PREFETCH_DEPTH=16 CONCURRENCIES="64 256" \
  ./motivation/run_beam_merge_ab.sh

./motivation/analyze_beam_merge_ab.py
```

## 8. Certified remote-adjacency transfer observation

只读 perfect-ADC 上界、可实现 PQ annulus/suffix 证书和 WQE 模型见
`motivation/ADJACENCY_CERTIFICATE_MOTIVATION.md`。复现命令：

```bash
./motivation/run_adjacency_certificate_motivation.sh
```

该 probe 复用 `query-rdma-trace-mode`，但使用独立的编译期 trace kernel；
`trace=off` 的生产 kernel 不包含 oracle 调用或其寄存器开销。采样运行的 QPS
不得作为性能数据。当前 SIFT100M/C16/c256 结果触发停止条件：可实现证书的
额外远端字节节省约为零。
