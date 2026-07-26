# Feedback-Horizon and Hardware-Hunger Expansion

## 结论

方案已完整接入并保留 `fixed` A/B 路径，但当前硬件信号在 SIFT100M、并发 256
负载下没有产生有效的动态收缩：全局 credit 很快达到自动上限 4 tiles，之后没有
ring-full 或 SQ-defer 事件将其清零。因此实际 batch 几乎退化为 cap=16，而
Feedback Horizon 的 merge 元数据维护成为纯开销。

最终代码的 5 秒关键对照结果为：

| 指标 | fixed depth 16 | feedback-hunger | 动态相对变化 |
|---|---:|---:|---:|
| QPS | 49,111.14 | 48,489.23 | -1.27% |
| mean latency | 5.206 ms | 5.272 ms | +1.27% |
| P99 latency | 6.414 ms | 6.490 ms | +1.18% |
| P999 latency | 7.352 ms | 7.354 ms | +0.03% |
| Recall@10 | 0.928125 | 0.928125 | 一致 |
| graph reads/query | 195.548 | 195.522 | 基本一致 |
| graph bytes/query | 182,152 | 182,130 | 基本一致 |
| graph rounds/query | 14.626 | 14.626 | 基本一致 |
| selected parents/round | 13.370 | 13.368 | 基本一致 |
| RDMA wait/query | 804.37 us | 796.08 us | -1.03% |
| Beam merge/query | 2,378.75 us | 2,451.47 us | +3.06% |

这不是一个正向性能结果。当前离散 owner-idle/backpressure 控制规则把“批次间短暂
queue empty”当成了硬件饥饿；160 个 owner warp 中只需少数 owner 在 query scoring
期间观察到空队列，就会把全局 credit 加满。由于该负载没有真正 ring-full 或
SQ-capacity defer，credit 没有回收机会。最终没有减少搜索工作或 round，只增加约
72.7 us/query 的 merge 元数据成本。

## 实际调用链

`process_query()` 在 Beam 中按既有顺序选择未 expanded 父节点，选择时立即写
`beam_expanded=1`，随后调用 `fetch_graph_records_batch()`。后者准备 per-query
graph slots，按 shard 形成 `DirectBatchDescriptor`，经 `direct_fetch_batch()` 写入
bounded device ring。exclusive QP owner warp 可聚合多个 descriptor，准备 READ WQE，
一次 doorbell 提交，并由最终 signaled WQE 覆盖前序 reads。查询 CTA 在
`wait_direct_batch()` 后验证 record，再按
`persistent_score_chunk_capacity()` 执行 decode、visited、PQ score 和
`merge_approximate_into_beam()`。

原 fixed target 是：

```text
min(prefetch_depth, max_expansions - expansions)
```

若 selected count 超过 score chunk capacity，会形成多个独立 decode/score/merge
chunk。当前常用布局中 capacity 为 16。

wide merge 对 old Beam 与 new candidates 做一次 CUB radix sort，并通过原有
old-handle lookup 恢复 expanded。compact merge 先形成两个最多 K 项的 sorted run，
再排序 2K 项；`compact_scratch_expanded` 原来只保存布尔 expanded。

## 实现

新增配置：

```text
--gpu-query-expansion-policy=fixed|feedback-hunger
```

默认仍为 `fixed`。动态路径自动推导：

```text
tile = max(1, blockDim.x / 32)
efficient_batch_cap =
  min(kPersistentMaxPrefetch,
      persistent_score_chunk_capacity(graph_entry_capacity,
                                      traversal_capacity))
maximum_credit_tiles = ceil_div(efficient_batch_cap, tile)
```

每轮 target 为：

```text
min(remaining_budget,
    efficient_batch_cap,
    feedback_horizon + hunger_credit_tiles * tile)
```

选择仍严格遵循 Beam 顺序；没有 speculative read、额外 graph read、第二份 Beam、
第二次排序或 CPU/GPU 控制往返。

### Feedback Horizon

wide 和 compact 两条路径都在既有 old-handle lookup 中同时生成 origin bit 与
expanded bit。每线程只保存局部最小 new output position；每 warp 由 lane 0 对两个
shared scalar 做一次归约更新。

compact scratch 的 `u32` expanded 字段改为 bit flags：

```text
bit 0: expanded
bit 1: new
```

final pass 恢复完整 flags，写回 authoritative Beam 时只写 expanded bit。

`old_unexpanded_before_new` 和总未 expanded 数量直接融合进原有 thread-0
valid-count 扫描，不存在 merge 后第二次 Beam 遍历。初版 benchmark 曾暴露一次额外
遍历；该结果已废弃，修正后重新完成 CUDA 测试与性能实验。

### Hardware Hunger

新增 128-byte、cache-line aligned 的 device-only `ExpansionPressureState`。第一条
cache line 只含 packed active/credit/peak 控制值及 maximum；第二条 cache line 保存
诊断 counters。每个 query 仅在进入/退出时各执行一次 active CAS，每轮只做一次
relaxed device load。

owner warp 仅在：

```text
queue empty
active_queries > 0
no deferred descriptor
announced == completed
not announced in current idle episode
```

时发放一次 saturating credit。获得 batch 后才允许开始下一个 idle episode。
`direct_fetch_batch()` 只在首次 enqueue 失败且 ring sequence/position 确认真的 full
时清零 credit；普通 producer CAS race 不算 backpressure。owner 只有在真实
`total_wqes + needed + completion_wqes > sq_wqe_num` defer 时清零 credit。

## Telemetry

Completion/query telemetry 包含 policy、selected/horizon/credit sums、min/max batch、
min/max horizon 和 graph rounds。报告端派生 average selected batch、average horizon
和 average credit。

global pressure telemetry 包含 active peak、current/peak credit、maximum、grants、
idle episodes、congestion clears、ring backpressure 和 SQ defer。详细 query RDMA
trace 保持原机制，热路径不写文件。本实现没有增加可选的 round trace，因为 prompt
将其列为可选项，且 query/global 指标已经能解释本次控制行为。

本次动态并发 256 运行观察到：

```text
average feedback horizon = 2.105
average credit tiles     = 4.000
maximum credit tiles     = 4
hunger grants            = 4
idle owner episodes      = 3,900,572
ring backpressure        = 0
SQ defer                 = 0
```

这组数据直接证明 credit 长期饱和，而不是 Feedback Horizon 本身不稳定。

## 内存与编译资源

动态策略只增加一个 128-byte 全局 pressure state；没有 per-query global allocation，
没有新增 shared array。Horizon、query aggregate 和 merge reduction 使用固定 shared
scalar，使链接后 kernel static shared 增加 72 B。fixed 模式不分配 pressure state。

使用相同 Release/sm80 参数和 `-Xptxas=-v` 分别编译 Git HEAD baseline 与最终代码：

| 资源 | baseline | 最终代码 |
|---|---:|---:|
| persistent entry registers/thread | 132 | 132 |
| linked entry static shared | 41,150 B | 41,222 B |
| entry spill stores/loads | 0/0 | 0/0 |
| runtime selected CTA | 128 threads | 128 threads |
| hardware blocks/SM | 3 | 3 |

`process_query` 的 40 B spill 和 owner loop 的 228 B spill在 baseline 中已经存在，最终
数值相同。新增 merge helper 的 out-of-line callee 有 8 B spill；最终 persistent entry
增加 72 B static shared，但 registers、entry spill 和 blocks/SM 不变。

## 正确性验证

新增 `gpu_feedback_horizon_test` 覆盖：

- new candidate 插入首位、中部、末尾及多个 new；
- 无 new 进入、top-K 截断、invalid/non-finite；
- expanded old、Beam 未满/已满；
- 相同 distance 的稳定 tie 语义；
- 128-thread compact、256-thread wide、compact-final-256。

测试逐项比较最终 handle、distance、expanded、ID、earliest new、old-unexpanded
count、new count 和 horizon。

新增 `gpu_expansion_pressure_test` 覆盖：

- 无 active query 不发 credit；
- enter/exit 计数及 peak；
- idle grant saturation；
- query relaxed load 不修改 credit；
- ring/SQ clear counters；
- 连续 idle episode 去重和获得 batch 后重新 arm。

验证结果：

```text
cmake --build build -j8                         PASS
ctest --test-dir build --output-on-failure -j8 PASS (52/52; sandbox GPU tests skipped)
gpu_feedback_horizon_test on GPU 1             PASS
gpu_expansion_pressure_test on GPU 1           PASS
gpu_compact_beam_merge_test on GPU 1           PASS
4 persistent dispatch combinations on GPU 1   PASS
```

端到端 fixed-16 与 feedback-hunger 的 Recall@10 均为 0.928125，未出现 graph reread、
query failure、QP/CQ error 或非法访问。

## 实验复现

完整 A/B 配置位于：

- `motivation/configs/common.env`
- `motivation/configs/feedback_hunger.env`
- `motivation/configs/prefetch_{1,8,16,32}.env`
- `motivation/run_feedback_hunger_ab.sh`

存储节点启动后执行：

```bash
./motivation/run_feedback_hunger_ab.sh
```

默认运行 fixed depths `1,8,16,32` 和 feedback-hunger，在并发
`1,8,64,256` 下测试。可缩小矩阵：

```bash
CONCURRENCIES="64 256" FIXED_DEPTHS="8 16 32" \
  ./motivation/run_feedback_hunger_ab.sh
```

本报告关键原始结果：

- fixed-16:
  `experiment/reports/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260726_190031.json`
- feedback-hunger:
  `experiment/reports/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260726_190403.json`

## 判断

实现满足“只改变正式 selected parent 数量、不增加算法额外工作类型”的设计约束；
但按当前规定的离散控制规则，该方案在本负载上不值得默认启用，默认继续保持
`fixed`。失败原因不是 Horizon 选择造成额外 graph work，也不是 occupancy、spill、
RDMA 拥塞或 Recall，而是 owner queue 的瞬时空闲不能代表全局硬件饥饿，且 credit
只在严重 backpressure 时回收，导致长期饱和。

若继续研究，需要改变控制信号定义或增加 credit 衰减/消费语义；这将超出本任务明确
禁止经验阈值、AIMD/EWMA 和 query-side credit update 的约束，因此本实现没有私自加入。
