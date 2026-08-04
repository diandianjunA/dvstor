# GPU 查询优化实验

当前项目只保留两项已经通过端到端性能和正确性验证、且能够组合使用的查询优化：

1. **Stable-Run Beam Merge**：复用已有有序 Beam，只排序本轮候选 run，再执行稳定
   top-K merge。
2. **Live-Extent RDMA**：保留存储侧可更新的定长图记录，只通过 one-sided RDMA
   传输当前有效 extent；并发更新时由 GPU packed high-water 安全修复过期长度档。

正式运行入口只有两个自包含 profile：

```text
experiment/profiles/04_gpu_persistent_gpunetio_baseline.env
experiment/profiles/04_gpu_persistent_gpunetio.env
```

二者使用相同的 C16、Beam、expansion budget、rerank、QP 和更新配置。baseline 使用
legacy Beam merge 与 fixed-size graph read；默认 profile 启用 Stable-Run 和
Live-Extent。这样 A/B 不会通过改变搜索预算或 Recall 口径制造收益。

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

只测 transport payload 的 GPUNetIO probe：

```bash
LIVE_EXTENT_CONFIG=motivation/configs/live_extent_rdma.env \
  ./motivation/run_live_extent_rdma_probe.sh
```

## C16 与性能诊断

`results/sweep/prefetch_sweep.csv` 保留固定 batch `1,2,4,8,16,32` 的完整汇总，
用于说明为什么两个正式 profile 都固定使用 C16。需要重新运行时：

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
