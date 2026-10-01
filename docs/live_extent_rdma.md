# Live-Extent RDMA

Live-Extent 保留图记录的固定物理大小和更新余量，只缩短查询时每条 RDMA READ 的 payload。
图 offset、搜索前沿、visited、PQ 和精排语义保持不变。长度档是可过期的性能提示，完整
header、checksum 和 incarnation 决定记录是否可用。

## 静态长度 sidecar

每个基础节点按物理 ordinal 保存一个 u8 class：

```text
class = ceil((stable_count + provisional_count) / 8)
bytes = min(graph_entry_bytes, 16 + class * 8 * sizeof(RemotePtr))
```

`<prefix>.gextent8` 包含 128 B header 和 `num_nodes` 个 class。header 绑定格式版本、
构建指纹、节点/分片数、图记录大小、容量和指针宽度；header 与 payload 分别校验 checksum。
记录按 metadata 分片顺序和 shard-local slot 顺序排列。100M 节点的 payload 为 100 MB，
不包含邻接表、handle 或向量。

在持有全部 `.dat` 分片的机器上生成：

```bash
cmake --build build-storage --target vamana_graph_extent_indexer -j16
./build-storage/vamana_graph_extent_indexer --index-prefix /path/to/index/prefix
```

工具全量校验 schema-16 图记录、active pointer 和零 suffix，写临时文件后原子 rename。
覆盖已有 sidecar 需显式 `--overwrite`。分布式部署时将 sidecar 放到计算节点的相同索引前缀。
数据集构建脚本已包含这一步。

## 读取策略

| `gpu-query-graph-read-policy` | 行为 |
|---|---|
| `fixed` | 一次读取完整图记录 |
| `header-neighbor` | 先读取 header，再按精确长度读取邻接正文 |
| `live-extent` | 根据 class 一次读取前缀，提示不足时完整读回退 |

贡献模式 `gpu-dynamic-graph-access-mode=fixed/adaptive` 会分别解析为完整读和启用动态
extent 的 Live-Extent；下层读取策略实验使用 `manual`。裸配置的读取策略默认为 `fixed`。
正式 baseline/full 为保持物理索引一致，都会校验 `.gextent8`；只有消费 Live-Extent 的
路径保留长度表。Persistent 为静态表分配 `align_up(N, 4)` B，另有每个 query slot 的
有界请求长度 scratch。

## 动态长度与槽复用

动态 PQ payload 使用以下布局：

```text
[ extent class:8 | incarnation:24 ] [ PQ code ] [ checksum:32 ]
```

checksum 绑定 incarnation 与 PQ code，不绑定可变的 extent 字节。动态邻接改写将 graph
和紧随其后的 4 B tag 一起发布，无需额外 WQE/CQE；新节点沿用完整记录、header-last
发布。R96 图记录为 832 B，邻接改写携带 tag 后为 836 B。SIFT 的 R96/PQ32 动态记录
stride 为 1040 B；其他维度、dtype 和 PQ 宽度由布局器计算，不能套用这一数值。

GPU 在原本需要的动态 PQ 读取中取得 class。PQ32 首次 miss 为 `4 + 32 + 4 = 40 B`，
一次 READ。class 与 PQ payload 共同存入按物理槽位映射的 arena；只有非 BUSY 且
incarnation 完全匹配的槽可提供 hint。冷节点、unknown class 或复用中的槽执行完整读。
PQ arena 命中在评分前后双采样 state，远端 miss 验证 incarnation 和 checksum；单次
RDMA READ 本身不是硬件原子快照，完整性仍有 32-bit checksum 的碰撞概率边界。

## 校验与自愈

每个请求的 scratch 仍按完整记录大小预留，短读不填充未传输 suffix。校验器扫描有效
前缀，并用 FNV prime 的模幂延续逻辑零 suffix，得到与完整记录相同的 checksum。
QP owner 按每条请求的长度构造 WQE，继续使用原有的聚合、doorbell 和 CQ ownership。

若 header 声明的邻接超出已传输前缀，或结构/checksum 校验失败，下一次 snapshot attempt
升级为完整读。仅明确的 underhint 且后续完整 snapshot 校验成功时才提升 class；torn
record/checksum 失败不训练长度表。重读后仍无效时沿用查询的有界失败策略，incarnation
变化则丢弃 stale 候选。

静态表通过 packed-u32 CAS 单调提升 high-water。动态图只修改同一 incarnation 的
arena class，旧查询不能更新复用后的新对象。checksum-valid 动态 snapshot 比缓存档位
低至少两档时降至 `observed + 1`，留一档滞回；UNKNOWN 可由完整 snapshot 首次精化，
不记为 fallback/promotion。

热节点无需每次边更新后重读 PQ tag：增长靠 short-read/full-fallback 升档，收缩靠完整
校验的 snapshot 降档。hint 与引擎同寿命，重置 benchmark 计数不会清空它，预热可以建立
测量期状态。稳定图版本修复后可避免重复 underhint；持续 churn 下，较旧的同-incarnation
snapshot 仍可能暂时降档，再由后续读取修复。该策略保证邻接不被截断，不保证 hint 与每次
图更新线性一致。

## 统计口径

| 字段 | 含义 |
|---|---|
| `graph_read_bytes` | 实际 admitted graph payload，包含重试与回退 |
| `graph_live_extent_reads` / `graph_full_record_reads` | 短读/完整读的物理 READ 数 |
| `graph_extent_fallback_reads` | extent 路径触发的完整读回退 |
| `graph_extent_underhint_reads` | 明确长度不足后成功 admitted 的首次完整回退 |
| `graph_extent_hint_promotions` | 成功提升的 class 数；竞争中别人已提升不重复计数 |
| `expanded_parent_count` / `expanded_neighbor_count_sum` | 权威展开节点数与真实邻居总数 |
| `expanded_degree_histogram` | 按 `ceil(degree/8)` 统计的查询加权展开分布 |
| `dynamic_graph_*` | 动态版本的 short/full/bytes/fallback/promotion/demotion 独立计数 |

`remote_pages` 仍是逻辑正式父节点读取数，不能用 `remote_pages * record_bytes` 推算
物理流量。动态 short/full 统计 snapshot attempts；一次 short→full 同时贡献一个 short、
一个 full 和一个 fallback。派生指标使用一致的分母：

```text
snapshot_attempts = short + full
nonfallback_full_attempts = max(full - fallback, 0)
short_physical_ratio = short / snapshot_attempts
fallback_ratio = fallback / short
```

`nonfallback_full_attempts` 也包含 checksum 重试，不能解释为纯 cold/unknown miss。
`dynamic_code_incarnation_rejects` 为保留的报告字段名，包含 incarnation 不匹配、arena
复用交叠和 trailer checksum 失败。RDMA trace 的 completion 时间对应 owner submission
组的 CQE，不能解释为逐 WQE 完成时间。

## 实验与验证

[Program 2](../experiment/README.md#program-2动态长度读取) 在同一动态 mixed workload
下比较 Fixed、Header→Neighbor 和 ClassExtent，并从实际权威展开的邻居数计算 Oracle
字节下界。每个 case 从同一静态索引重启，使用相同输入和更新压力。

应同时检查 Recall、逻辑展开数、物理 WQE、实际字节、fallback、动态访问占比、QPS 和尾延迟。
字节下降不等于同幅度吞吐提升，固定 offered rate 也不能证明最大容量提升。当前脚本不保证
不同 case 的逐操作 commit 顺序相同，不能将其当作逐查询 top-k identity 证明。

本地测试覆盖 sidecar 绑定/checksum、长度档、短读校验、槽复用和统计派生。GPU occupancy、
RDMA 混合长度读取、动态长期负载与召回率需要对应硬件验证；资源占用以本次构建和实际
测试为准。
