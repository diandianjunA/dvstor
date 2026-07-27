# GPU 图搜索批次屏障 Motivation Test

## 结论先行

批次屏障和高并发下的 completion 离散都是真实的，但当前这组 SIFT100M、5-shard、
fixed + stable-run 实验**尚不足以支撑“正式工作乱序执行、搜索状态顺序提交”会带来巨大性能
收益**。

在最重要的 depth=16、concurrency=256 配置下：

- 34.7% 的 primary round 能观察到两个或更多 shard completion boundary；
- 这些 multi-shard round 的 strict completion spread P50/P90 为 15.36/69.63 μs；
- 但 65.3% 的 round 只有一个可观察 shard batch；
- parent-weighted strict barrier waste 为 7.3%；
- 每查询累计 strict overlap window 相对 RDMA wait 的 P50 上界为 9.5%；
- 相对整个 GPU query residence 的 P50 上界只有 2.1%，P90 上界为 8.1%。

因此，现有数据可以作为“批次越大，straggler 离散越明显”的机制证据，却不能诚实地宣称它是
当前查询吞吐的主要瓶颈。这里的数值还是理论机会**上界**；加入 ready queue、顺序 commit、
额外状态和调度以后，实际收益只会更低。

## 代码确认后的同步点

固定扩展的真实路径是：

1. query CTA 按 Beam 顺序选择父节点，并在发起 RDMA 前设置 `beam_expanded`；
2. `fetch_graph_records_batch()` 按目标 shard 形成 descriptor；
3. 所有非空 shard descriptor enqueue 完成；
4. query CTA 分工调用 `wait_direct_batch()`，随后执行 block barrier；
5. 最后一个 shard 完成后才统一 validation；
6. 函数返回后才执行 neighbor decode、visited、PQ score 和 Beam merge。

所以“已 READY 的 shard 不能被提前处理”的同步点真实存在。

但 completion 粒度有严格限制：owner warp 最多把 8 个 descriptor 合并为一次 submission，
整个 submission 只有最终 WQE/dump 请求成功 CQE。owner poll 到该 CQE 后才发布组内 descriptor
状态。因此本实验观察的是：

```text
shard descriptor 所属 owner submission group 的软件可见完成边界
```

而不是单 parent、单 WQE、单 descriptor 的物理完成，也不是 NIC 内部完成。同一 shard 内约
多个父节点的完成离散当前完全不可见。

## 为保证口径正确所做的修改

原有 trace 框架已存在，但分析器有两个阻断错误，并把两种时间混为 barrier waste。本次修正为：

- trace event 增加 `route_attempt`，避免 route retry 后从 round 0 重新编号而错误合并；
- 增加统一的 `wait_phase_start_timestamp_ns`，不把串行 issue/enqueue 时间计作乱序执行机会；
- owner submission group 内复用同一个 completion timestamp，避免 status 发布循环人为制造离散；
- 将原来的 `process_start - completion_i` 拆成：
  - straggler barrier：`C_max - C_i`；
  - strict wait barrier：`C_max - max(C_i, W_0)`；
  - completion 后 CTA handoff：`P - C_max`；
  - ready 到实际消费：`P - C_i`；
- primary snapshot 和 retry 分层；
- single-shard round 从 spread CDF 排除，而不是作为 0 混入；
- overflow、非法 timestamp、不完整 group、失败 query、事件数/round/read/batch 对账失败均显式
  报告；
- 增加 depth=1 负对照；6687 个 primary round 全部是单 shard，spread 正确报告为 N/A/0，
  没有伪造离散。

所有详细 timestamp 只在 sampled/full trace 下执行。OFF 模式仍不分配 trace arena。
schema 2 的 event 为 64 bytes；本次 256 query slots × 4096 events 约占 64 MiB。
timestamp 使用可跨 SM 比较的 PTX `%globaltimer`，名义单位为 ns；本机所有 timestamp 差值的
量化最大公约数为 1024 ns，因此实际时间分辨率约为 1.024 μs。query CTA 的 phase cycle 仍只用
同一 CTA 内的 `clock64()` 相对差并按 GPU clock kHz 换算。

## 指标定义

对 attempt \(a\) 中的 shard batch \(s\)：

- \(I_s\)：query CTA 开始 enqueue 的时间；
- \(W_0\)：所有 shard enqueue 完成后的统一 wait 起点；
- \(C_s\)：owner 发布的 completion boundary；
- \(C_{\max}\)：最后一个 completion boundary；
- \(P\)：所有 wait 和 CTA barrier 结束、开始 validation 的时间；
- \(n_s\)：该 shard descriptor 中的父节点数。

核心墙钟机会：

\[
D_a^{strict}
=
C_{\max}
-
\min_s \max(C_s,W_0)
\]

parent-weighted ready area：

\[
A_a^{strict}
=
\sum_s n_s
\left[
C_{\max}-\max(C_s,W_0)
\right]
\]

归一化 strict barrier waste：

\[
\frac{
\sum_a A_a^{strict}
}{
\sum_a B_a(C_{\max}-W_0)
}
\]

其中 parent-weighted area 的单位是 `parent·ns`，只衡量 READY work supply，绝不能称作查询
延迟。每查询 \(\sum_a D_a^{strict}\) 也只是不考虑计算资源冲突、调度开销和依赖的 overlap
window upper bound。

## 2026-07-27 快速实测

配置：

```text
dataset              SIFT100M / METIS / 5 shards
expansion policy     fixed
Beam merge policy    stable-run
concurrency          256（另有 depth16/concurrency1）
trace                sampled
GPU CTA              128 threads，natural parent tile = 4
```

| depth | concurrency | traced queries | multi-shard rounds | strict spread P50/P90 | strict parent waste | 一个 tile 提前 ≥10 μs 的 multi-shard round | query upper bound / GPU time P50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 50 | 0.0% | N/A | 0.0% | N/A | 0.0% |
| 8 | 256 | 212 | 30.5% | 11.26/34.20 μs | 5.0% | 26.1% | 1.0% |
| 16 | 1 | 136 | 38.6% | 2.05/5.12 μs | 0.8% | 0.4% | 0.4% |
| 16 | 256 | 234 | 34.7% | 15.36/69.63 μs | 7.3% | 33.8% | 2.1% |
| 32 | 256 | 186 | 36.9% | 37.38/155.03 μs | 9.1% | 49.4% | 3.6% |

所有实测 trace 的以下 integrity counter 均为 0：

```text
overflow
failed query
invalid timestamp
incomplete group
duplicate shard
query/event mismatch
graph round mismatch
graph batch mismatch
graph read/retry mismatch
```

depth16 的 sampled overhead sanity：

| concurrency | trace-off QPS | sampled QPS | 差值 |
|---:|---:|---:|---:|
| 1 | 325.01 | 322.99 | -0.6% |
| 256 | 58.65 K | 58.63 K | -0.04% |

这些只是单次短跑，用于验证机制和方向，不替代正式的 3-repetition 矩阵。

## 对创新方案的含义

1. **离散现象成立。** 高并发下 spread 明显高于单并发，且随 depth 8→16→32 单调上升。
   lone-straggler tail 也占主要部分，说明少量慢 completion 确实会拖住 multi-shard round。
2. **当前可见覆盖面不够。** depth16 下约三分之二 round 只有一个 shard descriptor；在这些
   round 中，现有 completion API 无法释放任何更细粒度的正式工作。
3. **吞吐收益上界偏小。** depth16 的中位 end-to-end headroom 仅 2.1%。即使把这部分全部
   消除，也难以形成希望的“大幅”系统提升。
4. **depth32 不是解决办法。** 它把 strict spread 和上界增至 37.38 μs、3.6%，但已有实验已
   证明 depth32 因 stale-beam over-expansion 降低整体 QPS。不能为了扩大屏障现象而改变算法
   工作量口径。
5. **最可能被漏掉的机会在 shard 内部。** 当前一个 shard descriptor 包含多个 parent read，
   但只由最终 CQE 一次性变为 READY。如果创新机制依赖 parent-level 乱序，必须先用独立诊断
   transport 模式证明 shard 内离散；现有结果不能替它作证。

现阶段建议：**不要直接实现完整 reorder buffer。** 下一阶段只值得做一个独立、默认关闭的
tile-signaled completion probe（例如每个自然 parent tile 一个 signaled boundary），同时测量
额外 CQE/CQ polling 对 transport 的扰动。如果该 probe 显示 shard 内存在足够大的 ready work，
且扣除 completion-granularity 成本后的保守收益仍显著，再进入“execute-ready、commit-in-order”
原型；否则应转向 owner queue/QP tail 和 Beam materialization 等更大的瓶颈。

## 运行方法

快速端到端验证：

```bash
QUICK=1 ./motivation/run_batch_barrier_motivation.sh
```

完整矩阵（默认值即如下配置）：

```bash
DEPTHS="1 8 16 32" \
CONCURRENCIES="1 8 64 256" \
REPETITIONS=3 \
./motivation/run_batch_barrier_motivation.sh
```

单独重算报告：

```bash
./motivation/summarize_batch_barrier.py \
  motivation/results/batch_barrier
```

主要产物：

- `motivation/results/batch_barrier/REPORT.md`
- `motivation/results/batch_barrier/trace_matrix.csv`
- 每个 run 的 `rdma_trace.jsonl`
- 每个 run 的 `rdma_trace.summary.json`
- 每个 run 的 `rdma_trace.summary.md`

若要在小规模 FULL trace 中内嵌全部派生 round row：

```bash
./motivation/analyze_rdma_trace.py \
  --include-round-details /path/to/rdma_trace.jsonl
```
