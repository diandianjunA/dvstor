# Live-Extent RDMA motivation 实验报告

## 结论

本轮实验给出了一个明确但有边界的结论：

> **值得继续实现“一次性 live-extent RDMA”端到端原型；不值得实现
> “先读 header、再依赖式追读 body”的两阶段方案。**

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

但是，这还不是端到端 QPS 提升结果。当前代码在发 READ 前不知道 live length；如果每次都先读
16 B header 再追读 body，160 QP 下逻辑 batch/s 反而只有相应单次短读的
`0.454x`/`0.497x`，P50 增加 129.024/109.568 us。因此，后续原型的成败条件不是“能否少读
字节”，而是：

> **能否在发 WQE 前以很小的有界 metadata 得到安全的 extent hint，并把绝大多数图记录保持为
> 一个 READ。**

本报告把当前阶段判定为 **conditional GO**：静态、一次性短读原型有充分 motivation；
两阶段读取为 **STOP**；动态更新下的 hint 失效与 checksum 语义必须单独验证。

> 2026-07-27 实现状态：conditional-GO 原型现已接入主查询路径。实现采用通用的
> u8/8-edge class（而非只适用于当前 R96 的 packed 4-bit）、同 descriptor
> per-request length、完整 scratch suffix 重构、原 checksum 校验和有界 full-read
> fallback。默认仍为 `fixed`；端到端 A/B 尚未完成，因此本报告中的性能结论仍只到
> motivation/transport gate，不把实现完成误写成 QPS 收益。

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

后续原型暂称：

**Elastic-at-Rest, Compact-in-Flight Graph Access**

中文可表述为：

**存储保留更新弹性、网络只传有效邻接的图访问**

它不压缩或复制整张图，不把图搬到 GPU，也不让存储 CPU 参与查询：

```text
存储节点：仍保留固定 832 B、可原地更新的物理记录
计算节点：根据预读可得的 extent hint，发起一次 one-sided 短 READ
GPU：在原 832 B scratch 中补零未读 suffix，沿用验证、decode 和搜索语义
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

## 6. 可实施边界

### 6.1 推荐的第一个端到端原型

第一个静态图、one-shot 短读原型的实现边界：

1. 离线为 base node 生成 u8 `ceil(live_edges/8)` extent-class sidecar；u8 能覆盖
   R128 加 provisional slots 等通用布局。
2. 100M 节点需要 100 MB metadata。它不是图副本，约为当前 83.2 GB fixed
   graph plane 的 0.12%。
3. query 根据静态 handle 的确定性 slot ordinal 读取 class，在 issue WQE 前得到长度。
4. 给现有 request scratch 增加 per-request `u32 bytes` 数组，而不是按长度拆成更多 descriptor。
   32 requests/query、256 query slots 时只需约 32 KiB global scratch。
5. owner warp 对每个 WQE 读取自己的 length，继续保留原有 QP ownership、doorbell 和最终 CQE
   语义；graph READ/WQE 数不变。
6. 每个目标 scratch 仍为 832 B。短读前把未读 suffix 清零，然后使用现有 record header 和
   checksum 做等价验证。
7. 若 header count 超出 class，绝不截断 decode：丢弃短读并在下一 snapshot attempt
   使用完整记录。

该原型不修改 832 B 存储记录，不引入存储 CPU RPC，也不改变 Beam、visited、扩展顺序或
Recall。

### 6.2 动态更新不能从静态结果直接外推

并发更新会使 build-time class 变旧。当前实现把 class 仅作为 optimistic hint：

- 首次短读始终包含 authoritative count；
- 若 `stable_count + provisional_count` 超出已读 capacity，丢弃并 full-read；
- dynamic slot 或无法映射的 handle 默认 full-read；
- shrink 不要求同步，最多少节省字节。

当前没有在线修改共享 class table；因此反复访问同一个已增长且超档的 static node
可能反复 fallback。该代价由 `graph_extent_fallback_reads` 单独报告，是否需要安全的
high-water cache 应由 mixed-update 实验决定，而不是默认加入。

但在启用前必须证明：

- zero suffix 是每个已提交版本的不变量；
- 短读补零后的 full-record checksum 与完整读取完全等价；
- torn publication、tombstone、generation/incarnation 变化不会被误接收；
- 16-bit checksum 的现有容错边界没有因重构而削弱；
- mixed update 下 fallback/continuation rate 足够低。

若这些条件不能被测试证明，就只能对 immutable base graph 使用短读，对动态/可变记录保留
832 B。不能把 stale hint 当成 authoritative length。

### 6.3 当前不能采用的实现

- 每条记录 `16 B header -> wait -> body`：本实验已显示高并发显著退化。
- 存储 CPU 收到请求后打包回复：把 one-sided data plane 变成双边 RPC，会引入存储 CPU
  bottleneck，与当前架构目标冲突。
- 在 GPU 保存图结构或邻接 prefix cache：会破坏存算分离的容量目标。
- 仅修改 checksum 覆盖长度而不处理并发更新：可能接受截断或 torn record。
- 按 extent class 拆成许多 shard descriptor：可能把省下的字节换成更多 doorbell/CQE 和
  owner queue 竞争。首个原型应使用 per-request length。

## 7. 下一阶段通过/停止标准

静态端到端 A/B 必须保持：

```text
same query set
same C=16 / Beam width / max expansions / rerank width
same selected parents and expansion sequence
same graph READ count
same top-k IDs/distances and Recall
same validation/retry semantics
```

建议预注册如下判断：

- correctness：top-k、扩展序列和 Recall 逐查询一致；
- bytes：graph payload/query 至少下降 45%，total RDMA payload/query 至少下降 40%；
- transport：不得增加 graph WQE、descriptor、CQE；
- performance GO：c256 QPS 至少提高 8%，mean latency 至少下降 8%，P99 不退化；
- strong GO：RDMA wait/query 下降至少 20%，且 QPS/P99 收益跨 3 次重复稳定；
- STOP：字节显著下降但 QPS < 3% 提升，或新 bookkeeping/zero-fill/validation 抵消大部分收益；
- dynamic gate：mixed update continuation < 10%，否则先解决 hint，而不是掩盖额外 WQE。

这些阈值用于防止在原型后选择性解释结果，不代表本 motivation 已经达到端到端性能门槛。

## 8. 局限

1. live-degree trace 和现有端到端基线都来自 SIFT100M；尚未证明跨数据集分布稳定。
2. transport probe 是专用 GPU RDMA microbenchmark，不包含 Beam、decode、PQ、visited 和
   exact rerank。
3. `application_payload_GB/s` 是请求字节/墙钟，不是 NIC wire counter。
4. 64 MiB/shard 随机工作集小于完整索引；它验证的是 transport sensitivity，不是完整查询
   locality。
5. one-shot probe 使用 uniform payload；生产路径已支持真实 variable-size mix，但仍需
   端到端 A/B 测量其 QPS/P99。
6. 当前没有可报告的 live-extent 端到端 QPS、P99 或 Recall 改善。
7. sidecar、fallback 和 CPU 侧 reconstruction/checksum 单元测试已完成；真实
   GPUNetIO mixed-length 内容等价性仍需在部署 sidecar 后由端到端查询验证。
8. probe 检查 transport completion 和计数，但没有验证返回 payload 的图记录内容。
9. 每个 case 的 warmup 与 measured kernel 之间存在一次 host 同步和计数清零间隔，结果不能
   当成完全连续的查询稳态 trace。

因此准确结论是“机会和 transport 因果均已得到支撑，值得做受控原型”，而不是“方案已经带来
某个百分比的查询加速”。

## 9. 产物与复现

核心产物：

- `motivation/analyze_live_extent_motivation.py`
- `motivation/summarize_live_extent_motivation.py`
- `motivation/analyze_live_extent_rdma_probe.py`
- `motivation/run_live_extent_rdma_probe.sh`
- `motivation/results/live_extent/c256/live_extent_analysis.{json,md}`
- `motivation/results/live_extent/c256_replication/live_extent_analysis.{json,md}`
- `motivation/results/live_extent/concurrency_roofline.{json,md}`
- `motivation/results/live_extent_rdma/20260727_125146/`

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
