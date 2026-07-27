# “工作乱序执行、搜索状态顺序提交” Motivation Test

## 实验要证明什么

该方案需要同时满足两个事实：

1. 一轮正式父节点读取存在足够大的、软件可观察的 completion 离散；
2. 提前 READY 的记录携带足够多的 validation、decode、PQ 和 visited 工作，能够在最慢读取
   完成前真实执行，而不是只产生一块没有计算资源或没有足够工作可消费的生命周期面积。

只证明 `max(completion)-min(completion)>0` 不足以支撑完整实现。

## 三层证据

### 1. Trace-off 性能基线

使用 fixed expansion、stable-run merge、depth 1/8/16/32，
concurrency 1/8/64/256，至少 3 次重复。吞吐结果必须完全关闭详细 trace。

### 2. Completion 离散

当前 schema 2 能观察：

```text
query shard descriptor 所属 owner submission group 的最终 CQE 边界
```

它不能观察同一 shard 内的 parent/WQE completion。因此当前结果只能够支持或否定
shard-granularity 的乱序执行方案。

核心指标：

```text
strict spread = C_max - min_i(max(C_i, common_wait_start))
strict parent waste =
  sum_i parents_i * (C_max - max(C_i, common_wait_start))
```

single-shard round 不作为 0 混入 spread CDF。

### 3. Release-time 执行 oracle

`analyze_ordered_commit_oracle.py` 把每个 completion 当作 work release。它只允许以下工作越过
当前 I/O barrier：

```text
graph validation
neighbor decode
PQ scoring
visited check/update
```

Beam merge、下一轮父节点选择、expansion budget、termination 和 exact rerank 仍在
authoritative commit 之后。

当前 trace 没有逐 task service time，因此 oracle 将每个 query 的上述实测 wall time按
graph parent 数线性分摊。这是敏感性模型，不是性能预测。报告同时扣除每个自然 parent tile
0/1/2/5/10 μs 的 queue/state 管理成本。

预注册的继续门槛：

- trace integrity 全部通过；
- 至少 25% primary round 有两个以上 release boundary；
- strict spread P50 >= 10 μs，或 P90 >= 25 μs；
- **零调度开销**的 release-time oracle 在 query GPU residence 上 P50 >= 8%。

若最后一项失败，则即使 completion 离散真实，也不足以支持 shard-granularity 原型。

## 当前已有数据

SIFT100M、5 shards、depth 16、concurrency 256 的 schema 2 trace 显示：

- multi-shard primary rounds：34.7%；
- strict spread P50/P90：15.36/69.63 μs；
- parent-weighted strict barrier waste：7.3%；
- 简单的累计 spread 上界占 GPU query residence P50：2.1%。

release-time oracle 会进一步扣除“提前 READY 但其计算不足以覆盖完整 tail”的情况。当前
shard completion 数据因此预期只能给出很小的、零管理开销上界；最终数值以脚本输出为准。

这组结果证明批次屏障存在，但不能被选择性解释成巨大收益。若要继续，需要独立的
parent/tile-signaled transport probe。该 probe 必须默认关闭，并同时报告：

- 每 query 新增 CQE 数；
- CQ polling time；
- SQ/CQ occupancy 和 defer；
- trace-on 相对 trace-off 的 QPS/P99 扰动。

若 per-parent signaling 本身显著改变 transport，应把它仅作为诊断模式，不得当成正式方案的
数据路径。

## 运行

存储节点启动后，从项目根目录执行：

```bash
# 完整 trace-off + mechanism trace 矩阵，然后运行 oracle
./motivation/run_ordered_commit_motivation.sh

# 复用已有 trace，只重算 oracle
ANALYZE_ONLY=1 ./motivation/run_ordered_commit_motivation.sh

# 单个 trace
./motivation/analyze_ordered_commit_oracle.py \
  motivation/results/batch_barrier/trace/depth_16/concurrency_256/repeat_1/rdma_trace.jsonl

# 分析器单元测试
python3 -m unittest motivation/test_analyze_ordered_commit_oracle.py
```

每个 trace 旁生成：

```text
rdma_trace.ordered_commit_oracle.summary.json
rdma_trace.ordered_commit_oracle.summary.md
```
