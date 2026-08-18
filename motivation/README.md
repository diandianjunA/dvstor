# GPU 查询优化实验

当前项目只保留两项已经通过端到端性能和正确性验证、且能够组合使用的查询优化：

1. **Stable-Run Beam Merge**：复用已有有序 Beam，只排序本轮候选 run，再执行稳定
   top-K merge。
2. **Live-Extent RDMA**：保留存储侧可更新的定长图记录，只通过 one-sided RDMA
   传输当前有效 extent；并发更新时由 GPU packed high-water 安全修复过期长度档。

正式运行入口只有一个 full profile：

```text
experiment/profiles/04_gpu_persistent_gpunetio.env
```

查询优化消融均从这个 profile 出发，只覆盖被测开关。比如 Beam merge A/B 只覆盖
`GPU_QUERY_BEAM_MERGE_POLICY`，Live-Extent A/B 只覆盖
`GPU_QUERY_GRAPH_READ_POLICY`/`GPU_DYNAMIC_GRAPH_EXTENT`；runner 固定其余 C16、
Beam、expansion budget、rerank、QP 和更新配置。这里的 feature-off 只用于单项因果
分析，不是论文的系统 baseline。系统 reference baseline 是
`baseline/cpu-gpu-exact-safe@f304e99` 的 CPU-driven GPU exact +
storage-owner exact/sync 版本，契约见
`experiment/README.md`。

## Stable-Run Beam Merge

设计、等价性、GPU 资源占用和正式结果见：

- `STABLE_RUN_BEAM_REPORT.md`
- `results/beam_merge_final/`

SIFT100M、fixed C16、concurrency 256 的正式对照中，Stable-Run 相对 legacy：

- QPS `+20.14%`
- 平均延迟 `-16.75%`
- P99 `-16.54%`
- Beam merge/query `-54.69%`
- Recall@10 不变

复现实验：

```bash
./motivation/run_beam_merge_ab.sh
./motivation/analyze_beam_merge_ab.py \
  motivation/results/beam_merge
```

## Live-Extent RDMA

动机、格式、安全 fallback、高水位更新语义和正式结果见：

- `LIVE_EXTENT_RDMA_MOTIVATION_REPORT.md`
- `../docs/live_extent_rdma.md`
- `results/live_extent_rdma/`
- `results/live_extent_e2e/`

当前构建的静态 C16/concurrency-256 对照中，Live-Extent 相对 fixed record：

- QPS `+8.90%`
- 平均延迟 `-8.17%`
- P99 `-7.89%`
- graph bytes/query `-49.64%`
- tracked RDMA bytes/query `-44.34%`
- Recall@10 不变

端到端 A/B：

```bash
./motivation/run_live_extent_ab.sh
python3 motivation/summarize_live_extent_ab.py \
  motivation/results/live_extent_ab
```

### 动态节点三组消融

动态更新场景使用独立的 fail-closed 入口，同时比较：

- `fixed`：所有图记录读取完整 832 B；
- `static-only`：静态节点使用 Live-Extent，动态节点仍读完整记录；
- `dynaextent`：静态节点使用 Live-Extent，动态节点使用 incarnation-tagged extent。

三组只改变 `GPU_QUERY_GRAPH_READ_POLICY` 与
`GPU_DYNAMIC_GRAPH_EXTENT`。runner 固定 mixed/rate-limited 40K query/s +
1K insert/s、336 clients、30 秒 warmup、120 秒 measure 和 1000 条 Recall，并按
3x3 轮转顺序消除固定的先后位置偏差。`REPETITIONS` 默认为 3，并且只接受 3 的
整数倍；分析器也拒绝缺号、缺边或不完整的 Latin-square cycle：

```bash
DYNAEXTENT_BEFORE_CASE_HOOK=/absolute/path/reset_snapshot.sh \
  ./motivation/run_dynaextent_mixed_ab.sh
```

reset hook 是强制项，必须非空、可执行且每个 case 成功返回；其 stdout/stderr 与退出码
保存到对应 case 的 `before_case_reset.log`。hook 接收
`policy concurrency repetition run_dir` 四个参数，并负责在返回前恢复同一个持久化快照。
此外它必须在 hook 输出中**恰好包含一行**：

```text
snapshot_id=<immutable-snapshot-id-or-content-digest>
```

这里的值必须是由 reset 系统给出的不可变版本 ID 或内容摘要，不能对所有状态硬编码同一个
常量。runner 会立即要求同一 repetition 的三种策略返回完全相同的 ID，并把 reset log 的
SHA-256、snapshot ID、策略、repeat 和 Latin position 写入对应 benchmark JSON 的
`dynaextent_reset` 证书。分析器重新计算 log SHA-256 后才接受该报告。缺 hook、hook
失败、缺/重复/非法 snapshot ID、同组 ID 不同、既有 case 目录或缺 sidecar 都会失败。

这条证书把报告绑定到了 trusted reset hook 的声明；分析器无法绕过 hook 独立读取所有
远端内存并证明 ID 确实对应其内容，因此 hook 本身仍是实验可信边界。

独立分析器会严格核对目录到报告的三策略映射、reset 日志、相同输入与 insert ID
范围、三边初始 Recall、warmup/measure 完成写入数，并输出 `static/fixed`、
`Dyna/static`、`Dyna/fixed`：

```bash
python3 motivation/summarize_dynaextent_mixed_ab.py \
  motivation/results/dynaextent_mixed_ab
```

这里机器检查的是同一 repetition 的 trusted-hook snapshot ID、绑定的 reset log、输入、
ID range、初始 Recall 和完成 update count。336 个并发 clients 之间的逐条提交顺序不保证
完全相同，也不声明存在 mutation-order hash；相同 ID/set/count 因而不能被写成“动态图
拓扑逐位相同”。六项 DynaExtent raw telemetry 及其派生量描述的是**物理 snapshot
attempt**；尤其 `dynamic_graph_short_reads + dynamic_graph_full_reads` 不能解释成动态图
logical read。摘要保留 raw totals 用于审计，但 headline 使用 per-query 动态计数、每次
物理读取的字节数和物理比例，避免把完成 query 数差异误写成机制差异。

40K query/s 是固定 offered rate，因此这里的 QPS 表示 target attainment，不是系统最大
capacity；该实验的主要性能量是延迟和 per-query RDMA 开销。最大吞吐结论需要另做饱和
target sweep。

只测 transport payload 的 GPUNetIO probe：

```bash
LIVE_EXTENT_CONFIG=motivation/configs/live_extent_rdma.env \
  ./motivation/run_live_extent_rdma_probe.sh
```

## C16 与性能诊断

`results/sweep/prefetch_sweep.csv` 保留固定 batch `1,2,4,8,16,32` 的完整汇总，
用于说明为什么 full profile 与对应 feature-off 消融都固定使用 C16。需要重新运行时：

```bash
./motivation/run_prefetch_sweep.sh
./motivation/summarize_prefetch_sweep.py \
  motivation/results/sweep
```

普通 shard-batch RDMA trace 是默认关闭的诊断能力，不是第三种查询算法。它不修改
Beam、visited、扩展顺序或 graph read 数量；`off` 模式不分配 detailed trace arena。

```bash
# 小规模 trace 完整性检查
./motivation/run_full_trace_smoke.sh

# 不同 depth/concurrency 的诊断矩阵
./motivation/run_trace_sweep.sh

# 分析单个 JSONL
./motivation/analyze_rdma_trace.py /path/to/rdma_trace.jsonl
```

当前 completion 粒度是 shard descriptor 所属 owner submission group 的软件可见完成
边界，不是单 parent、单 WQE 或 NIC 内部完成时间。分析结果只能按这一粒度解释。

## 结果管理

`motivation/results/` 只保存 Stable-Run、Live-Extent、C16 选择及通用 trace 所需的正式
证据。普通运行产生的日志和报告应写入 `experiment/logs/` 与
`experiment/reports/`，不作为查询优化实现的一部分。
