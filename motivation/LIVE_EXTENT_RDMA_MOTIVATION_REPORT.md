# Live-Extent RDMA motivation 实验报告

## 结论

本轮实验给出了三个层次清楚、但必须分别理解的结论：

1. **一次性 Live-Extent RDMA 在静态高并发查询上成立。**
2. **只依赖 build-time immutable extent class 的第一版在混合更新下失败。**
3. **由 GPU 在完整记录校验成功后单调提升 class 的 device high-water 修复，在一组严格受控
   的混合负载历史对照中消除了绝大多数重复 fallback，并恢复了净延迟收益。**

“先读 header、再依赖式追读 body”的两阶段方案仍应停止：它把几乎每次逻辑图读取变成两个
串行 READ，实测代价远大于少传字节的收益。

当前图记录为了支持原地更新，为每个节点固定保留 832 B；查询却在每次图读取时把这些尚未使用
的更新余量也搬过网络。SIFT100M 的两次独立查询 trace 中，被正式扩展的父节点平均只有
46.84–47.21 条 live edge。按 8 条边一档的一次性短读，只需 417.72–420.48 B/parent：

- 图记录 payload 减少 49.46%–49.79%；
- 按当前完整查询的 graph/exact read 构成投影，总 RDMA payload 减少
  44.17%–44.47%；
- graph READ 数、正式扩展数、decode/PQ/visited/Beam 工作量均不需要增加。

真实 GPU-initiated one-sided RDMA 扫参进一步证明，这些字节在当前硬件上不是“无害 padding”。
在 160 个 active QP 下，400 B 和 448 B 单次 READ 相对 832 B 分别获得
`1.669x` 和 `1.617x` 的 READ-WQE/s，batch P50 降至 `0.605x` 和 `0.616x`。
这两个 payload 位于实测平均 live extent 的两侧。

如果每次都先读 16 B header 再追读 body，160 QP 下逻辑 batch/s 反而只有相应单次短读的
`0.454x`/`0.497x`，P50 增加 129.024/109.568 us。因此，该设计的成败条件不是“能否少读
字节”，而是：

> **能否在发 WQE 前以很小的有界 metadata 得到安全的 extent hint，并把绝大多数图记录保持为
> 一个 READ。**

本报告把静态 c256 原型判定为 **performance GO**，把两阶段读取判定为 **STOP**。
正式的 30 s warmup + 120 s measure 反向顺序配对得到 QPS `+8.79%`、
mean `-8.08%`、P99 `-7.40%`、P999 `-7.25%`；graph payload 减少 `49.64%`，
而每查询正式父节点和 RDMA op 数不变。两个更早的短配对分别得到 `+8.22%` 和 `+9.08%`
QPS，主体收益已跨三组复现。跨并发、跨数据集和更新负载仍是独立 gate，不能从 c256
静态结果直接外推。

混合更新实验揭示了静态实验看不到的失败模式。immutable class 版本虽然仍减少 graph bytes
`46.96%`，但每查询发生 `4.6395` 次 fallback，物理 graph WQE 增加 `2.42%`，shard
batch 增加 `11.78%`，最终 mean/P99 查询延迟反而增加 `0.77%/2.45%`。原因不是搜索多扩展
了节点，而是插入引起的 base-record 增长使同一个过期 class 被反复命中；一次
underhint 会在后续每次访问时重复支付 `short READ + full READ`。

当前实现把设备侧 class table 改成四个 u8 class 共用一个对齐 u32 word。在短读 header
证明 extent 不足后，查询仍按原语义 full-read；只有该完整记录通过原有 full checksum
后，才以 packed `atomicCAS` 将对应 class 单调提升到已验证的 required extent。它不从
可能 torn/corrupt 的短 header 学习，不下调 class，也不修改磁盘上的 `.gextent8`。
同一 class transition 只会成功发布一次 promotion；在该 CAS 发布前，已经并发在途的查询
仍可能各自 fallback，但发布后的新查询不会继续因同一个旧 class 重复 fallback。

一组 40K query/s + 1K insert/s、30 s warmup + 120 s measurement 的受控历史对照得到：

- fallback 从 `22,269,730` 降至 `650,483`，减少 `97.08%`（`34.24x`）；
- high-water 中 `650,483` 次 underhint 对应 `649,570` 次成功 promotion，
  fallback 仅占 short read 的 `0.0724%`；
- 相对 fixed，graph bytes/query 减少 `48.81%`，全部已跟踪 query RDMA bytes/query
  减少 `43.50%`，物理 graph WQE 只增加 `0.070%`；
- 相对 fixed，mean/P99/P999 查询延迟减少 `3.44%/1.52%/3.02%`，GPU RDMA wait
  减少 `4.48%`；逻辑 graph reads/query 和 rounds/query 基本不变。

这组混合负载由 pacer 固定在 40K query/s，因此 **不能声称 QPS 或最大吞吐提高**；它只证明
在该负载点的延迟和容量余量改善。它还是一个跨 build、单次运行的历史对照，并且 30 s
warmup 会建立并保留 device high-water 状态，所以结论限于 warmed steady state。Recall
只在 base-only 1000-query 集合上测量，负载只有 insert，没有覆盖 upsert/delete，也没有
证明插入 ID 的检索质量或逐查询 top-k 完全一致。准确判断是：**high-water 修复通过了继续
复测和最大吞吐实验的 gate，但尚不能由这一组结果宣称混合更新下的最终性能结论。**

> 2026-07-27 实现状态：Live-Extent 已接入主查询路径，默认仍为 `fixed`。磁盘 sidecar
> 保持通用 u8/8-edge class，100M 节点约 100 MB；GPU 表仅为支持并发原子提升而按 u32
> 对齐打包，空间量级不变。实现继续沿用同 descriptor per-request length、逻辑零 suffix
> 的精确 FNV 延续、原 checksum 校验和有界 full-read fallback。静态与混合负载的正、负
> 原始结果均完整保留。

## 1. 直观故事

### 1.1 观察

当前系统把两种本应独立的大小绑定在一起：

```text
storage allocation size = 832 B
network transfer size   = 832 B
```

固定物理槽位对更新有价值：它允许邻接表增长、反向边暂存和原地发布。但查询只需要当前有效的
邻接前缀。于是系统出现一个与存算分离直接相关的矛盾：

> **存储侧为未来更新预留的空间，变成了计算侧每次查询都要支付的网络流量。**

### 1.2 方案方向

该方案称为：

**Elastic-at-Rest, Compact-in-Flight Graph Access**

中文可表述为：

**存储保留更新弹性、网络只传有效邻接的图访问**

它不压缩或复制整张图，不把图搬到 GPU，也不让存储 CPU 参与查询：

```text
存储节点：仍保留固定 832 B、可原地更新的物理记录
计算节点：根据预读可得的 extent hint，发起一次 one-sided 短 READ
GPU：只扫描有效前缀并以 FNV prime 幂延续逻辑零 suffix，沿用验证、decode 和搜索语义
```

所以该方向优化的是“更新友好的物理布局向查询网络流量泄漏”这一系统级问题，而不是跳过边、
预测候选或改变搜索算法。

## 2. 代码事实

当前 SIFT100M 配置为 `R=96`。代码为 provisional backlink 额外保留
`ceil(R/16)=6` 个槽位，因此总 capacity 为 102。记录布局是：

```text
16 B header + 102 × 8 B RemotePtr = 832 B
```

实际查询路径的事实如下：

1. `prepare_graph_record()` 只能从 handle 解出 shard 和远端 offset。
2. `fetch_graph_records_batch()` 对每个已选父节点传入统一的
   `params.graph_entry_bytes`，当前为 832 B。
3. `DirectBatchDescriptor` 只有一个 `bytes` 字段；同一 descriptor 内所有 READ 使用相同长度。
4. owner warp 仍采用 exclusive QP、批量 doorbell 和最后一个 signaled WQE 覆盖前置 READ。
5. record 的 authoritative `stable_count` 和 `provisional_count` 位于记录前两个字节，即当前只能
   在读取开始后看到。
6. checksum 覆盖完整记录；编码器先清零完整记录，再写 live prefix，unused suffix 保持为零。
7. `RemotePtr` 的 64 bit 已由 34 bit offset、6 bit shard 和 24 bit incarnation 用满，当前
   没有可直接塞入 extent class 的空闲 tag bit。

这意味着简单把 RDMA 长度改小并不成立。必须同时解决：

- READ issue 前的 extent hint；
- descriptor 内 per-request length；
- 短读后完整 snapshot/checksum 的等价验证；
- hint 落后于并发更新时的安全 fallback。

## 3. 实验 A：查询真正需要多少图记录字节

### 3.1 数据来源与口径

分析只使用已经正式选择并成功获取的父节点记录中的 live degree：

```text
stable_count + provisional_count
```

没有使用 visited 结果、后续 Beam 命中或 query-dependent edge oracle，所以不会把“后来没用到
的边”选择性删除。所有 live edge 都被保留。

两份 trace 均无 overflow：

| trace | queries | score chunks | parents | live edges | avg live edges/parent |
|---|---:|---:|---:|---:|---:|
| 主样本 | 462 | 6,764 | 90,493 | 4,271,919 | 47.207 |
| 独立复测 | 152 | 2,233 | 29,793 | 1,395,491 | 46.840 |

### 3.2 字节结果

`ideal live prefix` 是 16 B header 加精确 live handles；`8-edge extent` 将 edge count 向上
取整到 8 的倍数。

| trace | fixed | ideal live prefix | 8-edge extent | graph payload reduction |
|---|---:|---:|---:|---:|
| 主样本 | 832.00 B | 393.66 B | 420.48 B | 49.46% |
| 独立复测 | 832.00 B | 390.72 B | 417.72 B | 49.79% |

8-edge 分档相对不可能实现的 byte-exact layout 只多 6.81%–6.91%，说明 4-bit class
（0–13 档）已足够逼近 byte oracle，不需要复杂压缩格式。

当前 c256 基线每查询读取 195.19 个 graph record，graph payload 占完整查询 RDMA payload 的
89.30%。只替换 graph payload 后：

| 模型 | graph B/query | total RDMA B/query | total payload reduction |
|---|---:|---:|---:|
| 当前 fixed 832 B | 162,396.74 | 181,852.74 | 0% |
| 8-edge extent，主样本 | 82,073.67 | 101,529.67 | 44.17% |
| 8-edge extent，独立复测 | 81,534.54 | 100,990.54 | 44.47% |

这里是 byte projection，不是 QPS 预测。

### 3.3 为什么不能默认采用 header + tail

如果发 READ 前没有长度，先读取 8 条边的 prefix：

- 99.948% 的父节点仍需要 continuation；
- 即使 tail 合并成一个连续 READ，也需要 1.999 WQE/parent；
- 若按 8-edge extent 逐段追读，则需要 5.901–6.776 WQE/parent。

因此“先看 header 再读剩余部分”几乎把所有图读取变成依赖链。实验 B 直接测量了这项代价。

## 4. 实验 B：减少 payload 在真实 RDMA 路径上是否有价值

### 4.1 方法

新增的只读 transport probe 使用与生产路径相同的关键机制：

- A800 GPU warp 直接构造 one-sided RDMA READ WQE；
- 每个 warp 独占一个 QP；
- 每批 16 个 READ，与 `max_rd_atomic=16` 对齐；
- 一个最终成功 CQE 覆盖该批前置 READ；
- 1、8、32、160 active QP，其中 160 等于 5 个 shard × 32 QP；
- 每个 storage region 使用 64 MiB 确定性随机工作集；
- 每个 case 32 个 warmup batch、512 个 measured batch、3 次重复；
- 正向/反向 payload 顺序交替；
- payload 为 16、80、144、272、400、448、528、832 B；
- 额外测试 dependent `16+400` 与 `16+448`。

表中吞吐为应用请求字节，不是 NIC wire bytes。所有比较先按 repeat 配对，再报告配对比值的
中位数。

### 4.2 一次性短读结果

| active QPs | one-shot payload | READ-WQE/s vs 832 B | batch P50 vs 832 B | batch P99 vs 832 B |
|---:|---:|---:|---:|---:|
| 32 | 400 B | 2.007x | 0.459x | 0.435x |
| 32 | 448 B | 1.807x | 0.514x | 0.478x |
| 160 | 400 B | 1.669x | 0.605x | 0.613x |
| 160 | 448 B | 1.617x | 0.616x | 0.635x |

绝对中位数：

| active QPs | payload | READ WQE/s | requested-payload GB/s | batch P50 | batch P99 |
|---:|---:|---:|---:|---:|---:|
| 32 | 400 B | 27.978 M | 11.191 | 17.408 us | 20.480 us |
| 32 | 448 B | 25.224 M | 11.300 | 19.456 us | 22.528 us |
| 32 | 832 B | 13.959 M | 11.614 | 37.888 us | 47.104 us |
| 160 | 400 B | 22.986 M | 9.195 | 109.568 us | 144.384 us |
| 160 | 448 B | 22.577 M | 10.115 | 111.616 us | 147.456 us |
| 160 | 832 B | 13.915 M | 11.577 | 181.248 us | 230.400 us |

32 QP 时 400–832 B 的 requested-payload throughput 已聚集在
11.19–11.61 GB/s，而 READ rate 随 payload 缩短近似反比上升。这是直接的 bandwidth-limited
证据。160 QP 时短记录还会受到 WQE、packet 和调度开销限制，所以收益不是理想的 2x，但目标
范围仍提供 61.7%–66.9% 的 graph READ capacity 增量。

### 4.3 两阶段读取结果

dependent case 每个逻辑记录固定产生两个 READ WQE，并等待 header batch 完成后才 issue body。

| active QPs | dependent/reference | logical batch/s | P50 增量 | P99 倍率 |
|---:|---|---:|---:|---:|
| 32 | 16+400 / one-shot 400 | 0.628x | +11.264 us | 1.600x |
| 32 | 16+448 / one-shot 448 | 0.695x | +9.216 us | 1.455x |
| 160 | 16+400 / one-shot 400 | 0.454x | +129.024 us | 2.277x |
| 160 | 16+448 / one-shot 448 | 0.497x | +109.568 us | 2.278x |

两阶段 case 在每个 repeat 中都排在 one-shot sweep 之后，因此比较虽按 repeat 配对，但没有完全
做时间顺序 counterbalance。退化幅度远大于重复波动，已经足以否决“每条记录先读 header”的
默认设计。

## 5. 实验 C：当前查询是否已经处于传输压力区

固定 C=16、stable-run、Recall@10=0.935 的历史端到端结果：

| concurrency | QPS | graph reads/query | rounds/query | GPU query | RDMA wait | total app RDMA GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 326.36 | 196.30 | 14.66 | 2.995 ms | 0.495 ms | 0.060 |
| 8 | 2,522.32 | 193.17 | 14.48 | 3.100 ms | 0.506 ms | 0.454 |
| 64 | 18,227.63 | 195.74 | 14.63 | 3.447 ms | 0.622 ms | 3.323 |
| 256 | 58,680.86 | 195.19 | 14.61 | 4.279 ms | 1.035 ms | 10.671 |

从 c64 到 c256，搜索工作量几乎不变，但 RDMA wait/query 增加 0.413 ms，GPU query time
增加 0.832 ms。c256 当前产生约 11.454 M graph READ/s；transport probe 在同样 160 QP 下
对 uniform 832 B 测得 13.915 M READ/s。当前完整查询的 10.671 application GB/s 也达到
probe 的 uniform-832 requested-payload throughput 11.577 GB/s 的约 92.2%。

这不是 NIC 线速利用率证明，因为查询混有不同 request size，probe 也不含 Beam/PQ/owner
竞争；但它说明生产查询已经接近本机实测的 832 B 图读取 transport envelope，而不是停留在
“字节减少也不会影响性能”的低负载区。

把 c256 的 44.17% byte reduction 机械换算得到的 `1.791x` bandwidth-only roofline 只是严格
乐观上界，不是预期 speedup。真正端到端收益还会受 Beam、PQ、exact read、WQE 和 GPU
occupancy 限制。

## 6. 实验 D：端到端配对

### 6.1 受控条件与正确性

三组配对：

- repeat 1：fixed `20260727_141643`，optimized live `20260727_142804`
- repeat 2：fixed `20260727_143054`，optimized live `20260727_143911`
- long BA：optimized live `20260727_144537`，fixed `20260727_144952`

三组均为 concurrency 256、fixed expansion C16、stable-run Beam、Beam 128、
max-expansions 384、rerank 128 和 32 QP/shard。前两组为 2 s warmup + 5 s measure、
32-query Recall；long BA 为 30 s + 120 s、1000-query Recall，并采用 live 后 fixed 的
反向顺序抵消前两组 fixed 后 live 的顺序偏差。每组内部严格分析器检查的所有受控字段均一致，
唯一算法配置差异是 `gpu_query_graph_read_policy`。

long BA 的正确性与工作量：

| 指标 | fixed | live-extent | 变化 |
|---|---:|---:|---:|
| Recall@10（前/后） | 0.9401 / 0.9401 | 0.9401 / 0.9401 | 0 |
| logical parents/query | 195.4325 | 195.4258 | -0.0034% |
| rounds/query | 14.6183 | 14.6195 | +0.0082% |
| parent batch | 13.3690 | 13.3675 | -0.0116% |
| shard batches/query | 20.9577 | 20.9753 | +0.0836% |
| exact reads/query | 128 | 128 | 0 |
| retry / fallback / failure | 0 / 0 / 0 | 0 / 0 / 0 | 0 |

性能测试为 closed-loop time mode，因此快的一侧在相同时间内消费了更多但不重用的连续
query rows；表中的微小工作量差异来自查询样本区间，而不是算法新增或删除 graph read。
固定 Recall 集合前后完全一致；aggregate 报告尚未提供逐查询 top-k/expansion hash，不能把
aggregate Recall 相等夸大成逐位结果证明。

### 6.2 正式长配对结果

| 指标 | fixed | live-extent | 变化 |
|---|---:|---:|---:|
| QPS | 57,750.34 | 62,824.17 | **+8.79%** |
| mean latency | 4,431.50 us | 4,073.49 us | **-8.08%** |
| P50 | 4,403.27 us | 4,057.13 us | -7.86% |
| P95 | 5,094.62 us | 4,704.94 us | -7.65% |
| P99 | 5,471.97 us | 5,067.21 us | **-7.40%** |
| P999 | 6,020.23 us | 5,583.84 us | **-7.25%** |
| GPU query/query | 4,354.59 us | 3,995.81 us | -8.24% |
| GPU graph/query | 1,867.82 us | 1,567.23 us | -16.09% |
| GPU RDMA wait/query | 1,038.93 us | 854.03 us | -17.80% |
| graph validation/query | 582.90 us | 395.89 us | -32.08% |
| graph B/query | 162,599.82 | 81,883.66 | -49.64% |
| all RDMA B/query | 182,055.82 | 101,339.66 | -44.34% |

live 在正式 measurement 中完成 7,539,305 条查询；fixed 完成 6,930,411 条。live 的
1,473,374,972 次 graph physical read 全部为 short read；fixed 的 1,354,427,370 次全部为
full read。两边都没有 fallback、retry、snapshot reread、route timeout 或 direct-path
failure。按每查询归一化后 RDMA op 数均约 323.4，不以增加 WQE、descriptor 或 CQE 换取
字节收益。

阶段归因显示：

- RDMA wait 减少 184.89 us/query；
- prefix checksum/validation 减少 187.01 us/query；
- 相同 128 次 exact read 的精排阶段减少 80.87 us/query，说明 graph 流量释放的 NIC/QP
  容量也帮助了 rerank；
- per-request length issue 增加 15.97 us/query；
- 未细分的 live graph bookkeeping 增加约 55.35 us/query；
- score 和 Beam 分别增加 13.04 和 7.50 us/query。

所以不是全部字节收益都转化为查询收益；扣除 bookkeeping 与轻微计算干扰后，GPU query
净减少 358.78 us，与端到端 mean 减少 358.01 us 一致。

稳定性也没有显示热衰减：

| policy | head QPS | tail QPS | tail/head | minimum 5 s QPS |
|---|---:|---:|---:|---:|
| fixed | 57,598.0 | 57,795.3 | 1.0034 | 57,581.2 |
| live-extent | 62,924.2 | 62,792.5 | 0.9979 | 62,552.8 |

按相同 elapsed-time index 比较，24/24 个 5 秒窗口中 live 均高于 fixed，窗口优势范围为
`+8.14%` 到 `+9.66%`；这说明收益贯穿整个 120 秒 measurement，而不是启动瞬态。

### 6.3 三组复现与当前判断

| 指标 | smoke 1 | smoke 2 | long BA | 三组中位 |
|---|---:|---:|---:|---:|
| QPS | +8.22% | +9.08% | +8.79% | **+8.79%** |
| mean | -7.62% | -8.33% | -8.08% | **-8.08%** |
| P99 | -5.20% | -8.54% | -7.40% | **-7.40%** |
| P999 | +13.17% | -9.33% | -7.25% | -7.25% |
| RDMA wait | -16.38% | -18.64% | -17.80% | -17.80% |
| graph bytes | -49.25% | -49.24% | -49.64% | -49.25% |

第一次 5 秒 smoke 的 P999 退化没有在第二次 smoke 或 120 秒长测中复现。不能选择性删除
第一次结果，但更长的 7.54M-query measurement 支持它是偶发 tail episode，而不是该机制的
系统性尾延迟代价。

长配对同时达到预注册的 `QPS >= 8%`、`mean >= 8%` 和 `P99 不退化` performance-GO
条件。`RDMA wait >= 20%` 的 strong-GO 子条件没有达到，实际为 17.80%，因此不能把结果写成
网络等待已被完全消除。准确结论是：

> **在 SIFT100M 静态 c256 上，Live-Extent 用 95.4 MiB metadata 将 graph payload 减半，
> 不增加远程操作或改变正式搜索工作，并稳定换得约 8.8% 端到端吞吐提升。**

packed device high-water 合入后的 current build 又完成了一组 live→fixed、30 s warmup +
120 s measure 的严格配对：

| 指标 | fixed | live-extent | live 相对 fixed |
|---|---:|---:|---:|
| QPS | 57,912.54 | 63,065.85 | **+8.8984%** |
| mean | 4,419.04 us | 4,057.86 us | **-8.173%** |
| P50 | 4,392.74 us | 4,045.00 us | **-7.916%** |
| P95 | 5,070.02 us | 4,669.35 us | **-7.903%** |
| P99 | 5,425.46 us | 4,997.48 us | **-7.889%** |
| P999 | 5,928.38 us | 5,481.37 us | **-7.540%** |
| GPU graph | 1,862.70 us | 1,561.16 us | **-16.188%** |
| RDMA issue | 143.59 us | 158.22 us | +10.185% |
| RDMA wait | 1,029.97 us | 846.14 us | **-17.848%** |
| graph validation | 585.14 us | 395.99 us | **-32.325%** |
| graph B/query | 162,595.77 | 81,878.40 | **-49.643%** |
| tracked query RDMA B/query | 182,051.77 | 101,334.40 | **-44.338%** |

两侧 Recall@10 前后均为 `0.9401`，failure 均为 0。logical graph reads/query、
selected parents/query 和 physical graph WQE/query 仅相差 `+0.0011%`，total tracked
WQE/query 仅相差 `+0.0006%`，exact reads/query 均为 128；
fallback/underhint/promotion 均为 0。因此该 pair 隔离出的变化是传输和校验有效
prefix，而不是少搜索、少发请求或使用 fallback。live 的 24/24 个 5 秒窗口都高于
fixed，且 fixed/live tail-to-head ratio 分别为 `0.99999/0.99934`。

严格摘要与原始报告为：

- `motivation/results/live_extent_e2e/current_build_static_c256_live211636_fixed215008_summary.{json,md}`
- `motivation/results/live_extent_e2e/live_highwater_c256_static/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_211636.json`
- `motivation/results/live_extent_e2e/fixed_highwater_build_c256_static/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_215008.json`

这仍是一组、固定 live→fixed 顺序而非随机交错的 closed-loop pair。更快侧在固定
measurement 时间内消费了不同长度的 single-pass query 前缀，所以只按 query 比较
工作量。运行流程以及相同的 schema/control fields 支持两侧来自同一当前构建，但报告
没有记录 binary hash，因而不能由报告本身密码学证明同一 binary。aggregate Recall
相同也不等于逐查询 top-k identity。

### 6.4 c1/c8/c64 压力对照

c1 pair 使用 30 s warmup + 120 s measure，c8 和 c64 pair 使用 10 s + 30 s；三者
都使用 1000-query Recall。每个 pair 的严格受控字段仍只有 graph-read policy 不同；
Recall@10 均为 0.9401，正式搜索工作、RDMA op/query 和 exact read/query 一致，
fallback/retry/failure 均为零。

| 指标 | c1 | c8 | c64 | c256 long |
|---|---:|---:|---:|---:|
| fixed total app RDMA GB/s | 0.057 | 0.453 | 3.248 | 10.514 |
| QPS 变化 | +1.07% | +2.55% | +4.38% | +8.79% |
| mean 变化 | -1.06% | -2.49% | -4.19% | -8.08% |
| P99 变化 | -0.91% | -1.97% | -2.68% | -7.40% |
| GPU graph 变化 | -1.63% | -4.78% | -9.31% | -16.09% |
| RDMA wait 变化 | +13.08% | +8.10% | -2.04% | -17.80% |
| validation 变化 | -30.41% | -32.96% | -32.78% | -32.08% |
| exact rerank 变化 | -6.13% | -5.20% | -0.78% | -21.84% |
| graph bytes/query 变化 | -47.82% | -49.74% | -49.25% | -49.64% |

c1 的 P50/P95/P99/P999 分别变化 `-0.96%/-1.07%/-0.91%/-1.17%`。按相同
elapsed-time index，24 个 5 秒窗口中有 23 个 live 更快；唯一反向窗口只差
`-0.064%`，窗口优势范围为 `-0.064%` 到 `+2.25%`。fixed/live 的
tail/head 分别为 0.9939/0.9977。c1 采用 fixed-then-live，与 c8 的
live-then-fixed 顺序相反；两点都没有出现低压力净退化。

c8 的完整分位数也全部改善：P50/P95/P99/P999 分别变化
`-2.62%/-2.20%/-1.97%/-2.10%`。六个 5 秒窗口中 live 均快于 fixed，窗口 QPS
优势为 `+2.24%` 到 `+2.92%`。parents/query 只变化 `+0.0030%`，rounds/query
只变化 `-0.0182%`，这是 closed-loop 模式消费不同连续 query slice 产生的微差，而不是
减少搜索工作。

四个压力点的 prefix validation 都减少约 30%--33%，绝对节省分别为
143.95/160.95/175.06/187.01 us/query。这是只扫描记录有效 prefix、再以数学方式
续算逻辑零 suffix 带来的基础计算收益。网络侧则呈现完全不同的压力响应：

- c1 的 fixed payload 只有 0.057 GB/s，RDMA wait 增加 63.90 us/query；
- c8 的 fixed payload 只有 0.453 GB/s，RDMA wait 在这一个 pair 中反而增加
  42.31 us/query；短 414 B 左右不会消除 one-sided READ 的固定 RTT 和 owner 开销；
- c64 开始出现很小的等待收益，RDMA wait 只减少 12.71 us/query；
- c256 接近本机 transport envelope 后，RDMA wait 和 exact rerank 分别减少
  184.89 和 80.87 us/query。

低并发 exact 阶段的微小变化方向和幅度不稳定，不能归因于网络；只有 c256 重复出现的
约 80 us 降幅才支持“graph 流量释放共享 NIC/QP 容量”的解释。与此同时，live 的
issue 增量保持在约 12--16 us/query，未细分 graph bookkeeping 增量约
47--55 us/query。因此准确的机制结论是：

> **消除无效 suffix 校验提供跨压力水平的基础收益；消除无效 suffix 传输则在 NIC/QP
> 接近饱和时自然放大收益，使 QPS 从 c1 的 +1.07%、c8 的 +2.55%、c64 的 +4.38%
> 增长到 c256 的 +8.79%，无需按并发调参。**

不能把约 `48%--50%` 字节减少机械外推成同量级查询加速，也不能声称短读降低了 c1/c8
的 RDMA 延迟。c1/c8/c64 各只有一组 pair，低压力 wait 回升仍需同并发反向重复确认；
网络收益是否能在更新造成 class 变化时保留，则由下一节的 mixed-update 实验单独回答。

### 6.5 Mixed update：immutable class 的负结果

混合负载采用 `mixed/rate_limited`，目标为 40,000 query/s + 1,000 insert/s，
336 个自动推导的 client，30 s warmup + 120 s measurement。查询参数仍为 fixed C16、
stable-run、Beam 128、max-expansions 384、rerank 128。fixed 和 immutable-live 两侧都完成：

- warmup：1,199,998 query + 30,000 insert；
- measurement：4,799,998 query + 120,000 insert；
- 相同 maintenance target sequence 到达 durable watermark；
- direct-path、route、Stage2 和 late storage RPC failure 均为零。

受控字段审计无 mismatch。原始报告与严格摘要为：

- fixed：
  `motivation/results/live_extent_e2e/fixed_mixed_q40k_w1k/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_202410.json`
- immutable live：
  `motivation/results/live_extent_e2e/live_mixed_q40k_w1k_repeat2/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_203441.json`
- strict summary：
  `motivation/results/live_extent_e2e/mixed_q40k_w1k_fixed202410_live203441_summary.{json,md}`

结果不能用“字节减少”掩盖：

| 指标 | fixed | immutable live | 变化 |
|---|---:|---:|---:|
| target/achieved query rate | 40K / 39,999.98 | 40K / 39,999.98 | paced，相同 |
| mean query latency | 3,883.946 us | 3,913.719 us | **+0.77%** |
| P99 query latency | 4,786.773 us | 4,904.242 us | **+2.45%** |
| P999 query latency | 5,333.304 us | 5,432.118 us | +1.85% |
| GPU RDMA wait/query | 727.299 us | 783.001 us | **+7.66%** |
| graph bytes/query | 159,325.69 | 84,506.77 | -46.96% |
| total tracked query RDMA bytes/query | 178,782.64 | 103,963.72 | -41.85% |
| physical graph WQE/query | 191.497 | 196.130 | **+2.42%** |
| fallback/query | 0 | 4.6395 | +4.6395 |

immutable live 一共产生 `22,269,730` 次 fallback/retry，约为 short read 的 `2.4767%`；
physical graph WQE 增量几乎完全由这些 fallback 解释，而 logical graph reads/query
变化只有 `-0.0038%`。shard batch 总数从 `99,410,008` 增至 `111,119,604`
（`+11.78%`）。也就是说，查询并没有做更多算法扩展；过期 hint 把同一个逻辑读取拆成了
重复的物理读取和 owner/shard 调度工作。这个负结果判定 **immutable dynamic hint STOP**。

这也是 high-water 机制的直接动机，而不是事后加入的通用 cache：插入并非只产生一个新
record，它还会使已有 base record 的 stable/provisional 邻接增长。在 immutable class 下，
被更新到更高 extent 的热点节点每次被查询都会再次 underhint。正确的状态应是“已由完整
checksum 证明过的最大 extent”，而不是永远冻结在 build-time degree。

### 6.6 Device high-water：机制与受控历史对照

high-water 版本保留磁盘 u8 sidecar 和 one-sided 查询路径，只改变计算节点设备侧 hint：

1. 四个 u8 class 打包在一个对齐 u32 中，普通查询以一次 device load 提取自己的 byte。
2. 只有短读 header 自身有效且声明的 required bytes 超过 transfer bytes 时，才将该次
   fallback 记为 exact underhint。
3. 查询按原语义 full-read，不从短 header 更新 class。
4. 只有 authoritative full record 通过原完整 checksum 后，才计算 required class，并用
   packed `atomicCAS` 执行单调 `max(old, required)`；并发提升最终收敛到最大已验证 class，
   同一 word 中其他三个 byte 保持不变。
5. dynamic/unmapped handle 继续 full-read，unknown/full class 不被猜测替换；class 不随
   shrink 下调，因此最多多读少量字节，不会截断 decode。

设备表在 engine 生命周期内保留。benchmark 在 warmup/measurement 边界清零 telemetry，
但**不会清零 high-water class**；所以 30 s warmup 同时是在线学习/训练态，正式 measurement
衡量的是 warmed steady state，不是 cold-start。

高水位结果复用同一 fixed 历史基线，并执行相同 workload contract。原始结果与严格摘要为：

- high-water live：
  `motivation/results/live_extent_e2e/live_highwater_mixed_q40k_w1k/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_205653.json`
- strict summary：
  `motivation/results/live_extent_e2e/`
  `mixed_q40k_w1k_highwater_fixed202410_live205653_summary.{json,md}`

严格 analyzer 检查到受控字段 mismatch 为 0；旧 fixed schema 中不存在的
underhint/promotion 字段只允许按 0 解释，fixed 若出现非零值则拒绝比较。不过该 pair
来自不同 build、不是随机交错 A/B，而且 high-water 目前只有一次正式 mixed run，因此它是
**受控历史对照**，不能当作跨 build confound 已被重复实验完全排除。

| 指标 | fixed | high-water live | 变化 |
|---|---:|---:|---:|
| query ops / insert ops | 4,799,998 / 120,000 | 4,799,998 / 120,000 | 相同 |
| achieved query rate | 39,999.98/s | 39,999.98/s | paced，相同 |
| mean query latency | 3,883.946 us | 3,750.411 us | **-3.44%** |
| P99 query latency | 4,786.773 us | 4,713.836 us | **-1.52%** |
| P999 query latency | 5,333.304 us | 5,172.237 us | **-3.02%** |
| GPU query/query | 3,827.571 us | 3,695.025 us | -3.46% |
| GPU RDMA wait/query | 727.299 us | 694.705 us | -4.48% |
| graph validation/query | 549.162 us | 402.610 us | -26.69% |
| logical graph reads/query | 191.4972 | 191.4957 | -0.0008% |
| graph bytes/query | 159,325.69 | 81,555.22 | **-48.81%** |
| total tracked query RDMA bytes/query | 178,782.64 | 101,012.16 | **-43.50%** |
| physical graph WQE/query | 191.4972 | 191.6312 | +0.070% |
| fallback/underhint per query | 0 | 0.13552 | +0.13552 |

high-water run 中共有 `650,483` 次 underhint/fallback 和 `649,570` 次成功 class
promotion，promotion/underhint 为 `99.86%`；只有 913 次 underhint 没有对应成功提升，
fallback 仅占 short read 的 `0.07236%`。相对 immutable live：

- fallback 减少 `97.08%`，即 `34.24x`；
- physical graph WQE/query 从 `196.130` 降至 `191.631`；
- shard batch 从 `111,119,604` 降至 `100,058,139`（`-9.95%`），只比 fixed 高
  `0.65%`；
- mean/P99 查询延迟减少约 `4.17%/3.88%`；
- GPU RDMA wait 和 graph aggregate 分别减少约 `11.28%/10.00%`。

这构成了比单纯 fixed-vs-live 更强的机制归因：三个版本的 logical graph work 和查询配置
基本不变；immutable live 先证明重复 fallback 足以吞掉字节收益，high-water 再在保留短读
的同时把 fallback/WQE/shard-batch 放大几乎消除，并同步降低 issue、wait 和查询延迟。
它支持“已验证 high-water，而非 immutable build-time degree，才是动态 Live-Extent 的
正确 hint abstraction”。

但仍不得做以下宣称：

- 40K 是 pacer 目标，不是 capacity；`query_qps` 相同由实验设计决定，不能写成吞吐提升。
- write mean/P99/P999 在这一 pair 中分别改善 `8.12%/9.26%/20.31%`，Stage2 P99
  delay histogram 上界从 64 ms 降至 32 ms；但 remaining/max-backlog 等瞬时计数并非全部
  同向，单次历史对照不足以声称 update pipeline 普遍加速。
- fixed/live 的初始 base-only Recall 均为 `0.9401`，结束后为 `0.9397/0.9395`，绝对差
  `-0.0002`。这不是 inserted-ID quality、逐查询 top-k identity 或动态线性一致性的证明。
- 负载只包含 insert；upsert、delete、tombstone、relocation 和 cold-start promotion wave
  尚未形成同等强度的正式证据。

## 7. 可实施边界

### 7.1 当前 one-shot 数据路径

当前端到端实现的边界：

1. 离线为 base node 生成 u8 `ceil(live_edges/8)` extent-class sidecar；u8 能覆盖
   R128 加 provisional slots 等通用布局。
2. 100M 节点需要 100 MB metadata。它不是图副本，约为当前 83.2 GB fixed
   graph plane 的 0.12%。
3. 加载到 GPU 后按 u32 对齐打包，每节点仍只占一个 byte；query 根据静态 handle 的确定性
   slot ordinal 提取 class，在 issue WQE 前得到长度。
4. 给现有 request scratch 增加 per-request `u32 bytes` 数组，而不是按长度拆成更多 descriptor。
   32 requests/query、256 query slots 时只需约 32 KiB global scratch。
5. owner warp 对每个 WQE 读取自己的 length，继续保留原有 QP ownership、doorbell 和最终 CQE
   语义；steady state 的 graph READ/WQE 数不变。
6. 每个目标 scratch 仍为 832 B，但短读不写未传输 suffix；只扫描 header 声明的
   required prefix，并以模 `2^32` 的 FNV prime 幂延续逻辑零 suffix，得到与原完整
   record checksum 等价的结果。
7. 若 header count 超出 class，绝不截断 decode：丢弃短读并在下一 snapshot attempt
   使用完整记录。

该实现不修改 832 B 存储记录，不引入存储 CPU RPC，也不改变 Beam、visited、扩展顺序或
Recall。

### 7.2 动态更新使用已验证的单调 high-water

并发更新会使 build-time class 变旧。当前实现仍只把 class 当作 optimistic hint，而不是
authoritative length：

- 首次短读始终包含 authoritative count；
- 若 `stable_count + provisional_count` 超出已读 capacity，丢弃并 full-read；
- dynamic slot 或无法映射的 handle 默认 full-read；
- shrink 不要求同步，最多少节省字节。

与 immutable 版本不同，full-read 成功后会执行 device high-water promotion，但它受以下
安全顺序约束：

- zero suffix 是每个已提交版本的不变量；
- 短读补零后的 full-record checksum 与完整读取完全等价；
- short header 只能触发 full fallback，不能直接触发 promotion；
- promotion 必须位于 authoritative full checksum 成功之后；
- packed CAS 只允许单调增大目标 byte，不得覆盖相邻 class；
- torn publication、tombstone、generation/incarnation 和原有 retry 语义仍由完整
  validation 路径处理；
- telemetry reset 不清空 table，engine teardown 才结束该进程生命周期内的学习状态。

因此 stale hint 的最坏结果仍是多一次有界 full-read，而不是接受截断记录。成功 promotion
只改变下一次传输长度，不会让当前 short snapshot 被当成有效 graph record。当前正式
mixed run 证明这套顺序把 fallback ratio 降至 `0.0724%`，但跨更新类型和 cold-start
仍需继续验证。

### 7.3 当前不能采用的实现

- 每条记录 `16 B header -> wait -> body`：本实验已显示高并发显著退化。
- 存储 CPU 收到请求后打包回复：把 one-sided data plane 变成双边 RPC，会引入存储 CPU
  bottleneck，与当前架构目标冲突。
- 在 GPU 保存图结构或邻接 prefix cache：会破坏存算分离的容量目标。
- 仅修改 checksum 覆盖长度而不处理并发更新：可能接受截断或 torn record。
- 按 extent class 拆成许多 shard descriptor：可能把省下的字节换成更多 doorbell/CQE 和
  owner queue 竞争。当前实现继续使用 per-request length。

## 8. 已通过门槛与下一阶段标准

静态 c256 已达到 graph bytes `-49.64%`、total tracked RDMA bytes `-44.34%`、
QPS `+8.79%`、mean `-8.08%` 和 P99 不退化的 performance-GO。RDMA wait 为
`-17.80%`，没有达到预注册的 `-20%` strong-GO；而且 aggregate report 没有逐查询
top-k/expansion hash，所以只能说 Recall、正式工作量和错误语义一致，不能补写成逐位证明。

immutable mixed 的负结果也说明，原先的 `continuation < 10%` dynamic gate 定义错误：
`2.4767%` 看起来很小，但每查询约 191 次 graph read 使它放大成 `4.6395`
fallback/query，并造成 `+2.42%` graph WQE 和 `+11.78%` shard batch。动态 gate
必须同时看：

```text
fallback per query
physical graph WQE amplification
shard-batch amplification
bytes saved
query latency / capacity
```

high-water 在当前 warmed pair 中把这些量降至 `0.1355 fallback/query`、
`+0.070% graph WQE` 和约 `+0.65% shard batch vs fixed`，同时保留
`-48.81% graph bytes/query`，所以通过的是“继续复测”的机制 gate，不是最终 paper
performance gate。

下一阶段应预注册并完整保留以下结果：

1. **同 build 重复。** fixed/high-water 使用同一最终 binary，以 AB/BA 或随机交错顺序至少
   完成三组 30 s + 120 s 正式 pair，排除当前跨 build 历史对照的混杂。
2. **cold 与 warm 分离。** 明确在 measurement 前是否清空 device high-water；分别报告
   promotion wave、收敛时间、fallback/query 和稳态结果，不能只展示 30 s warmup 后的状态。
3. **capacity 而非 paced QPS。** 40K/1K 只比较 latency/headroom；最大查询吞吐或固定 tail-SLO
   下的承载能力必须使用不被 40K pacer 封顶的独立实验。
4. **正确性。** 在相同 query set 上加入逐查询 top-k ID/distance、expansion/read count
   对照；动态负载同时评估 inserted-ID quality，而不只使用 base-only Recall。
5. **更新覆盖。** insert 之外独立测试 upsert、delete/tombstone、record relocation 和地址
   复用，验证 full-checksum-before-CAS、incarnation 和 retry 语义。
6. **跨域。** 至少再覆盖一种 degree/数据分布、另一档并发和另一套 NIC/GPU/QP 配置；extent
   class 不需要人工调参，但收益仍可能随记录空洞率和网络压力变化。

若同 build 重复后 warmed high-water 仍满足：

- graph bytes/query 至少下降 45%，total tracked query RDMA bytes/query 至少下降 40%；
- physical graph WQE amplification 不超过 0.5%，没有新的 descriptor/CQE 放大；
- logical graph reads、正式搜索参数和正确性结果不变；
- mean/P99 不退化，并在 unpaced capacity 或固定 SLO 下得到可复现收益；

则可把 mixed-update high-water 判定为 performance GO。反之，如果 promotion 已收敛但
issue/bookkeeping 仍抵消网络收益，或 cold-start fallback wave 不可接受，应如实停止动态
默认启用，而保留已经成立的静态/immutable-base 使用范围。

## 9. 局限

1. live-degree trace 和现有端到端基线都来自 SIFT100M；尚未证明跨数据集分布稳定。
2. transport probe 是专用 GPU RDMA microbenchmark，不包含 Beam、decode、PQ、visited 和
   exact rerank。
3. `application_payload_GB/s` 是请求字节/墙钟，不是 NIC wire counter。
4. 64 MiB/shard 随机工作集小于完整索引；它验证的是 transport sensitivity，不是完整查询
   locality。
5. one-shot probe 使用 uniform payload；生产 variable-size mix 已测
   c1/c8/c64/c256，但每个低/中并发点只有一组 pair。
6. 静态正式长测包含旧版 live→fixed 和 current-build live→fixed 各一组；两个额外结果是
   5 秒 smoke，仍不等价于同一最终构建的三个随机/交错长重复。
7. immutable mixed 和 high-water mixed 各只有一次正式运行；high-water 复用更早 fixed
   report，属于跨 build 历史对照，而不是同一最终 binary 的 AB/BA 重复。
8. mixed query rate 被 pacer 固定为 40K/s，因此 latency 改善只表示该 offered load 下的
   headroom，不能转换成 QPS 或最大 capacity 提升。
9. high-water table 跨 30 s warmup 和 120 s measurement 保留；正式数据是 warmed steady
   state，不包含冷启动 promotion wave 的完整代价。
10. mixed correctness 使用 1000 条 base-only query。初始 Recall 相同、结束 Recall 只差
    0.0002，仍不能证明 inserted-ID quality、逐查询 top-k 一致或动态线性一致性。
11. 当前 mixed workload 只包含 insert；upsert、delete/tombstone、relocation 和地址复用尚未
    形成正式结果。
12. high-water 在 engine 生命周期内只增不减；若 provisional edges 后续收缩，它可能保留
    一个偏大的安全 class，正确性不受影响，但长期字节节省可能低于 build-time degree 所暗示
    的上界。
13. `rdma_read_bytes/ops` 是 GPU 查询路径已跟踪的 application request，不是 NIC wire
    counter，也不覆盖所有 Stage2/存储侧流量；不能把它称为全系统网络字节。
14. probe 检查 transport completion 和计数，但没有验证返回 payload 的图记录内容。
15. 每个 case 的 warmup 与 measured kernel 之间存在一次 host 同步和 telemetry 清零间隔；
    high-water class 本身不清零，所以结果不能当成完全连续的 cold-start trace。

因此当前最强的准确结论是：

> **SIFT100M 静态 c1/c8/c64/c256 均有正收益，current-build 静态 c256 获得 8.90% QPS
> 提升；在一组
> warmed、40K/1K、insert-only 的受控历史对照中，device high-water 把 immutable hint
> 造成的重复 fallback 减少 97.08%，并在相同 paced rate 下把 mean/P99 查询延迟降低
> 3.44%/1.52%。混合负载的同一最终构建重复、最大吞吐、冷启动、跨数据集和完整动态
> 正确性仍是开放 gate。**

## 10. 产物与复现

核心产物：

- `motivation/analyze_live_extent_motivation.py`
- `motivation/summarize_live_extent_motivation.py`
- `motivation/analyze_live_extent_rdma_probe.py`
- `motivation/run_live_extent_rdma_probe.sh`
- `motivation/summarize_live_extent_mixed_ab.py`
- `motivation/results/live_extent/c256/live_extent_analysis.{json,md}`
- `motivation/results/live_extent/c256_replication/live_extent_analysis.{json,md}`
- `motivation/results/live_extent/concurrency_roofline.{json,md}`
- `motivation/results/live_extent_rdma/20260727_125146/`
- `motivation/results/live_extent_e2e/live_c256_zero_elision_smoke/`
- `motivation/results/live_extent_e2e/fixed_c256_zero_elision_pair/`
- `motivation/results/live_extent_e2e/live_c256_zero_elision_repeat2/`
- `motivation/results/live_extent_e2e/live_c256_zero_elision_long_ba/`
- `motivation/results/live_extent_e2e/fixed_c256_zero_elision_long_ba/`
- `motivation/results/live_extent_e2e/live_c64_zero_elision/`
- `motivation/results/live_extent_e2e/fixed_c64_zero_elision/`
- `motivation/results/live_extent_e2e/live_c8_zero_elision/`
- `motivation/results/live_extent_e2e/fixed_c8_zero_elision/`
- `motivation/results/live_extent_e2e/live_c1_zero_elision/`
- `motivation/results/live_extent_e2e/fixed_c1_zero_elision/`
- `motivation/results/live_extent_e2e/live_highwater_c256_static/`
- `motivation/results/live_extent_e2e/fixed_highwater_build_c256_static/`
- `motivation/results/live_extent_e2e/current_build_static_c256_live211636_fixed215008_summary.{json,md}`
- `motivation/results/live_extent_e2e/fixed_mixed_q40k_w1k/`
- `motivation/results/live_extent_e2e/live_mixed_q40k_w1k_repeat2/`
- `motivation/results/live_extent_e2e/live_highwater_mixed_q40k_w1k/`
- `motivation/results/live_extent_e2e/mixed_q40k_w1k_fixed202410_live203441_summary.{json,md}`
- `motivation/results/live_extent_e2e/mixed_q40k_w1k_highwater_fixed202410_live205653_summary.{json,md}`

远端 storage nodes 启动后，在 compute node 执行：

```bash
LIVE_EXTENT_CONFIG=motivation/configs/live_extent_rdma.env \
  ./motivation/run_live_extent_rdma_probe.sh
```

该 probe 会消费 storage nodes 的单一 compute session；结束后需要重启 storage nodes，才能
运行下一个服务或 probe。

重新分析已有 CSV：

```bash
python3 motivation/analyze_live_extent_rdma_probe.py \
  motivation/results/live_extent_rdma/20260727_125146/live_extent_rdma.csv \
  --output-json \
    motivation/results/live_extent_rdma/20260727_125146/live_extent_rdma_summary.json \
  --output-markdown \
    motivation/results/live_extent_rdma/20260727_125146/live_extent_rdma_summary.md
```
