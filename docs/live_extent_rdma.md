# Live-Extent RDMA

## 目标

Live-Extent RDMA 解耦远端图记录的两个尺度：

```text
存储分配：固定大小，保留原地更新余量
查询传输：一次 one-sided RDMA READ，只传当前邻接长度档覆盖的前缀
```

它不改变图记录的物理大小/offset、搜索父节点、Beam/visited、PQ、精排或存储 CPU
查询路径。
静态图稳态下，同一批次仍产生与 fixed 模式相同数量的 graph READ WQE、
descriptor、doorbell 和最终 CQE；唯一变化是每条 READ WQE 可以具有不同的
payload length。静态节点从离线 sidecar 初始化长度档；动态节点由存储写路径把长度档
搭载在原有 PQ incarnation tag 上，并由 GPU 按物理槽位和 incarnation 缓存。并发更新
第一次把记录撑出 GPU 已知档位时，允许一次安全的 short→full fallback；GPU 随后修复
该 incarnation 的档位，避免热门节点反复支付依赖式双读。

## 索引契约

离线工具为每个 immutable base-node physical ordinal 写一个 u8 class：

```text
class = ceil((stable_count + provisional_count) / 8)
bytes = min(graph_entry_bytes, 16 + class * 8 * sizeof(RemotePtr))
```

输出为 `<prefix>.gextent8`：

- 128 B header；
- `num_nodes` 个 u8 class，按 metadata shard 顺序和 shard-local slot 顺序排列；
- header 绑定 format version、build fingerprint、nodes、shards、record
  bytes/capacity 和 pointer width；
- header 与 payload 分别校验 checksum；
- 生成器全量校验每个 schema-16 graph record、active pointer 和零 suffix；
- 同目录临时文件完成后原子 rename，已有文件必须显式 `--overwrite`。

生成命令必须在持有 metadata 和全部 `.dat` shard 的机器执行：

```bash
cmake --build build --target vamana_graph_extent_indexer -j
./build/vamana_graph_extent_indexer \
  --index-prefix /path/to/index/prefix
```

分布式部署时，只需把生成的 `.gextent8` 复制到计算节点相同的 index prefix。
SIFT100M 的 payload 为 100,000,000 B；它不包含任何边、handle 或向量。

动态图不建立需要广播失效的全局 sidecar。现有 4 B dynamic PQ tag 重新编码为：

```text
[ extent class:8 | slot incarnation:24 ]
```

class 仍为 `ceil((stable + provisional) / 8)`。对已发布动态节点的邻接改写，
存储端把固定 832 B graph 与紧随其后的 4 B tag 合成一次
836 B 本地 copy 或 RDMA WRITE；因此写入 payload 只增加
`4/832=0.48%`，不增加 WQE、CQE 或网络往返。新节点仍沿用原有的
full-record/header-last 发布，tag 占用的是本来就存在的 4 B PQ 验证前缀，
并在 PQ 后保存 4 B incarnation-bound 完整性校验。当前 R96/PQ32 布局中该 trailer
占用原有 12 B 尾部对齐空洞，physical stride 仍为 1040 B；其他 PQ 宽度由布局器把
这 4 B 显式计入 record alignment。graph 更新既不改 PQ 也不改 trailer，因而仍不新增
WQE/CQE。tag 只是提示，graph header、incarnation 和
checksum 仍是正确性依据；即使读写并发导致 graph/tag 可见性不一致，也只会触发完整读
回退。

## GPU 数据路径

裸二进制在未加载实验 profile 时保留兼容默认 `fixed`：

```text
--gpu-query-graph-read-policy=fixed
```

查询算法的正式 baseline profile 为隔离 ASFE 变量而与 candidate 同样使用
live-extent；Live-Extent 自身的 fixed 对照由 `motivation/configs/live_extent_fixed.env`
显式设置。

默认生产 profile 启用：

```text
--gpu-query-graph-read-policy=live-extent
```

启动时 live-extent 模式严格加载并校验 sidecar，分配：

- `align_up(num_nodes, 4)` B 的 packed device class high-water table；
- `query_slots * kPersistentMaxPrefetch * sizeof(u32)` 的 request-length
  scratch。

fixed 模式不加载 sidecar，也不分配这两个数组。

每个 graph batch 的执行顺序为：

1. 解析 handle。static base node 通过确定性 physical ordinal 读取 class。动态图候选
   做 PQ 打分时读取 `4 B tag + PQ code + 4 B incarnation-bound checksum`；PQ32 的
   首次 miss 是 40 B、一个 READ WQE/一个 RTT。checksum 只绑定 low-24 incarnation 与
   PQ payload，不绑定会随图更新变化的 extent 高字节。GPU 把同一 tag 的 class 与 PQ
   payload 一起发布到一一映射的 physical-slot arena，因此不新增一次 metadata READ。
2. 动态 graph fetch 只有在 arena state 非 BUSY 且 incarnation 与 handle 完全相等时才
   使用 class。冷节点、unknown class、正在发布的槽位和已经复用的槽位都直接读取完整
   graph record。arena PQ 命中按 device-scope acquire/release 执行
   `state-before -> score payload -> fence -> state-after`；两次都必须是同 incarnation
   且非 BUSY。同 incarnation 的 extent CAS 不改 PQ，允许前后 class 不同；槽复用与评分
   重叠则丢弃该距离，并在同一批走 checksum-validated storage miss。
3. 保留原有固定大小 graph scratch，但短读不触碰未传输 suffix。校验器实际扫描
   header 声明的 required prefix，再用模 `2^32` 的 FNV prime 幂以 `O(log suffix)`
   延续逻辑零 suffix；class rounding 多取但未被 count 引用的 slots 也不必读取。
4. 按 shard 形成原有 descriptor，同时传入可选的 per-request length 数组。
5. QP owner 为每条 WQE 使用自己的长度；聚合、SQ credit、doorbell、最终 signaled
   WQE 和 CQ ownership 均保持不变。
6. completion 后先仅使用已到达 header 计算 exact required prefix。
7. 若 required prefix 超出本次 transfer，或逻辑补零后的完整结构/checksum 无效，
   下一 snapshot attempt 将该请求升级为 full read。
8. 只有“count 超出旧档位”触发的 full record 通过完整 checksum 后，CTA 才从该
   authoritative header 重新计算 class，并用 aligned-u32 packed CAS 单调提升对应
   byte。静态表保留单调 high-water；动态 arena 只对相同 incarnation 做 CAS，因而旧
   查询不能把旧 class 写到复用后的新对象上。
9. 动态图收缩时，checksum-valid snapshot 可以下调同 incarnation 的 class。只有实际
   class 至少低两档才下调，并保留一档 guard，吸收普通度数抖动；静态 sidecar 不下调。
10. full record 仍无效时沿用原有有界重读/fail-stop；dynamic slot incarnation 已变化
    时沿用原有 read-committed stale discard。checksum/torn-record fallback 不更新
    class。

边更新不会强制 GPU 重读已驻留的 PQ code/tag，否则每次更新都会变成额外
RDMA。冷节点或新 incarnation 在首次 PQ 读取时安装存储端发布的最新 class；
已驻留节点的增长由 short header 发现并在权威 full snapshot 后升档，收缩则由
checksum-valid snapshot 带滞回地降档。若同 incarnation arena 防御性地安装了
UNKNOWN，一次 checksum-valid full snapshot 会把它精化为实际 class；这是计划内 full
读的初次精化，不记为 short→full fallback、promotion 或 shrink demotion。因此
GPU hint 是按访问自愈，不需要失效广播。
新节点在候选打分时本来就需要读 `tag + PQ code + checksum`，它的第一次 graph fetch 因而已可
使用同批读回的 class，而不是固定传输整条大记录。

因此 extent class 只是性能 hint，不是正确性 authority。并发更新使 class 过期只会
让当前查询执行一次显式 full-read fallback 和后续 hint 修复，不会截断邻接表或绕过
checksum/incarnation/tombstone 语义。动态图 arena state 的低 24 bit 将 hint 生命周期
严格绑定到槽位 incarnation，复用无需广播清理。high-water 与 persistent engine 生命周期一致；
benchmark telemetry reset 不会清空它，因此 warmup 可以建立正式测量所需的稳态。
静态删除或收缩不会降低 class；动态节点则使用上述带滞回的安全下调。

动态降档验证了同一 incarnation 和完整 graph checksum，但没有与每一次 graph publication
建立额外的版本线性化点；持续增长与收缩交错时，较早的同-incarnation snapshot 可能暂时
把 class 降低，下一次 short read 会再次发现并修复。因此这里主张的是 incarnation-safe、
带滞回的最终自愈，而不是每次更新都线性一致的 hint。稳定图版本下，首个成功修复后才不会
为同一档位重复 fallback。

这里不把 PQ READ 描述成硬件原子快照。arena 路径由显式双采样排除复用交叠；远端
单 READ 由 32-bit incarnation-bound checksum 检测 mixed/torn body，保证边界与系统
现有 checksummed graph record 一致，但仍保留有限 checksum collision 的概率边界。

## Telemetry

query completion 和 benchmark JSON 新增：

```text
graph_read_bytes
graph_live_extent_reads
graph_full_record_reads
graph_extent_fallback_reads
graph_extent_underhint_reads
graph_extent_hint_promotions
expanded_parent_count
expanded_neighbor_count_sum
expanded_degree_histogram
dynamic_graph_short_reads
dynamic_graph_full_reads
dynamic_graph_read_bytes
dynamic_graph_fallback_reads
dynamic_graph_hint_promotions
dynamic_graph_hint_demotions
```

`expanded_degree_histogram` 是查询加权的权威 live degree 分布：只在父节点
完成关键读取、进入 authoritative expansion 时按 `ceil(degree/8)` 计数，配套的
parent count 和 neighbor sum 给出精确平均邻居数。它用于区分“整个索引的静态
degree 分布”和“查询真正访问的节点分布”，不会记录逐请求 trace。

其中 `graph_read_bytes` 是成功同步执行或成功进入 owner queue 的实际 graph payload
总和，包含 snapshot retry 和 fallback；不能用逻辑 `remote_pages * record_bytes`
替代。`short + full` 是物理 graph READ WQE 数，`remote_pages` 仍是逻辑正式父节点
读取数。`underhint` 只在首个 full fallback WQE 成功 admitted 时计数；
`promotions` 是 underhint 后成功的向上 class transition 数，同一节点可以随增长
多次提升，竞争中观察到别人已提升的 CTA 不重复计数。计划内 full read
对 UNKNOWN 的初次精化不包含在该计数中。

为保持报告 schema 兼容，`dynamic_code_incarnation_rejects` 沿用旧字段名，但它现在统计
所有被拒绝的动态 PQ snapshot：incarnation 不匹配、arena 读写复用交叠以及 trailer
checksum 失败；不能将该字段单独解释为 incarnation mismatch 数。

RDMA trace schema 3 的 shard-batch event 记录：

```text
parent_count
payload_bytes
minimum_bytes_per_parent
maximum_bytes_per_parent
```

completion 粒度仍是 owner submission group 的最终 CQE 边界，不伪造 per-WQE
completion timestamp。

动态计数单独报告，避免 100M 个 immutable base nodes 掩盖少量在线节点的行为。
`dynamic_graph_short_reads` 和 `dynamic_graph_full_reads` 统计物理 snapshot attempts；
checksum retry 会再次贡献一次 attempt，fallback 则同时贡献一个 short attempt、一个
full attempt 和一个 fallback。报告只派生具有同一物理语义的指标：

```text
dynamic_graph_snapshot_attempts = short + full
dynamic_graph_nonfallback_full_attempts = max(full - fallback, 0)
dynamic_graph_short_physical_ratio = short / snapshot_attempts
dynamic_graph_fallback_ratio = fallback / short
```

其中 `nonfallback_full_attempts` 只是排除已标记 fallback 的 full attempt，并不等价于
cold/unknown hint miss：checksum 导致的额外 full retry 也包含在内。这些物理计数不能
反推出动态图 logical reads；每次物理 snapshot attempt 的平均字节数同样以
`dynamic_graph_snapshot_attempts` 为分母。

## A/B

固定和 live-extent 配置只改变 graph read policy：

```bash
./motivation/run_live_extent_ab.sh
```

动态负载需要同一 binary 的三组消融：

```text
fixed:       GPU_QUERY_GRAPH_READ_POLICY=fixed
static-only: GPU_QUERY_GRAPH_READ_POLICY=live-extent GPU_DYNAMIC_GRAPH_EXTENT=false
DynaExtent:  GPU_QUERY_GRAPH_READ_POLICY=live-extent GPU_DYNAMIC_GRAPH_EXTENT=true
```

每个 case 必须从相同快照重启，使用相同输入、insert ID 区间和完成更新数；
只在前一个 case 的已修改图上继续运行会把动态图密度和 Stage2 进度混入 policy 差异。
当前并发 driver 不保证三组运行具有完全相同的 per-operation commit order，因而实验结论
不应声称 mutation-order hash 一致。

正式动态三组入口为：

```bash
DYNAEXTENT_BEFORE_CASE_HOOK=/absolute/path/reset_snapshot.sh \
  ./motivation/run_dynaextent_mixed_ab.sh
```

reset hook 必须为每个 case 恢复快照并恰好输出
`snapshot_id=<immutable-snapshot-id-or-content-digest>`。默认三个 repetitions 形成完整
3x3 Latin square；只允许以三个为单位增加完整 cycle。runner 要求同一 repetition 的
三种策略具有相同 snapshot ID，并把该 ID 与 reset log SHA-256、策略和 Latin position
写入 JSON 的 `dynaextent_reset`。分析器重新计算 SHA-256、核对三边 ID 和完整 Latin
cycle。该机制验证的是 trusted hook 的证书及其报告绑定，不声称分析器独立散列了所有
远端内存；hook 不能用与真实存储状态无关的常量 ID。

动态 raw counters 保留用于一致性审计，但正式 headline 使用 per-query short/full/
fallback/promotion/demotion/attempts、每 query 字节数和物理 attempt ratios。40K query/s
是固定 offered rate，QPS 仅表示 attainment；该实验用延迟和 per-query transport 评价
机制，最大 capacity 需要独立饱和 target sweep。

默认扫描 concurrency `1 8 64 256`，每点 3 次，并在相邻重复中交换策略顺序。
如每次 service run 前需要重新启动远端 storage session，可设置：

```bash
LIVE_EXTENT_BEFORE_CASE_HOOK=/path/to/restart-storage-hook \
  ./motivation/run_live_extent_ab.sh
```

报告必须同时比较 Recall、top-k、logical graph reads、physical graph WQE、
actual graph/RDMA bytes、fallback、QPS、mean/P95/P99 和 RDMA wait。字节减少但
QPS 不提升也应如实报告。

## c256 端到端结果

正式反向顺序 long pair 使用 30 s warmup、120 s measure 和 1000-query Recall。两边
Recall@10 均为 0.9401，logical graph reads/query、rounds/query、exact reads 和 RDMA
op/query 一致，无 fallback/retry/failure：

- graph bytes/query `-49.64%`，all RDMA bytes/query `-44.34%`；
- RDMA wait/query `-17.80%`，GPU query time `-8.24%`；
- QPS `+8.79%`；
- mean/P50/P95/P99/P999 分别
  `-8.08%/-7.86%/-7.65%/-7.40%/-7.25%`。

两个更早的 5 秒 smoke pair 分别得到 QPS `+8.22%` 和 `+9.08%`；三组的中位 QPS
变化为 `+8.79%`。第一次 smoke 的 P999 `+13.17%`，第二次和长测分别为
`-9.33%/-7.25%`，所有结果都保留。

结果位于：

```text
motivation/results/live_extent_e2e/live_c256_zero_elision_smoke/
motivation/results/live_extent_e2e/fixed_c256_zero_elision_pair/
motivation/results/live_extent_e2e/live_c256_zero_elision_repeat2/
motivation/results/live_extent_e2e/live_c256_zero_elision_long_ba/
motivation/results/live_extent_e2e/fixed_c256_zero_elision_long_ba/
```

该结果通过静态 SIFT100M c256 performance gate，但跨并发、跨数据集和更新负载仍须
独立验证，不能从高并发静态结果直接外推。

加入 packed high-water 后，当前构建又完成了一组 30 s warmup + 120 s measure 的
live→fixed 严格配对。两份 service 配置仅
`gpu-query-graph-read-policy` 不同，JSON control fields 一致：

| 指标（live 相对 fixed） | 变化 |
|---|---:|
| QPS | **+8.8984%** |
| mean / P50 | **-8.173% / -7.916%** |
| P95 / P99 / P999 | **-7.903% / -7.889% / -7.540%** |
| GPU graph / RDMA wait | **-16.188% / -17.848%** |
| graph validation | **-32.325%** |
| graph / tracked query RDMA bytes | **-49.643% / -44.338%** |
| RDMA issue | +10.185% |

logical graph reads/query、selected parents/query、physical graph WQE/query 的差异均只有
`+0.0011%`，total tracked WQE/query 差异为 `+0.0006%`，exact reads 均为 128；
fallback/underhint/promotion、direct-path 和 Stage2 failure 均为 0。两侧 Recall 前后
均为 `0.9401`。live 的 24/24 个 5 秒 QPS 窗口均高于 fixed，tail/head QPS 比为
`0.99934`，fixed 为 `0.99999`。这既确认 packed high-water 没有造成静态回退，也在
当前代码上复现了约 8.9% 的端到端收益。

严格摘要和原始报告位于：

```text
motivation/results/live_extent_e2e/current_build_static_c256_live211636_fixed215008_summary.{json,md}
motivation/results/live_extent_e2e/live_highwater_c256_static/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_211636.json
motivation/results/live_extent_e2e/fixed_highwater_build_c256_static/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260727_215008.json
```

边界是：这仍只是一组、顺序固定为 live→fixed 而非随机交错的 closed-loop pair；较快侧
在固定时间内消费了不同长度的 single-pass query 前缀，因此工作量只能按 query 归一化。
流程记录和一致 schema/control fields 支持“同构建”判定，但报告没有嵌入 binary hash，
不能把它表述为可由产物密码学证明的同一 binary。相同 aggregate Recall 也不是逐查询
top-k identity 证明。

## 1K insert/s mixed-update 结果

受控历史对照固定为 40K query/s + 1K insert/s、30 s warmup、120 s measure、
C16/stable-run、base-only Recall@10。两份配置除读取策略外一致，但 fixed 报告来自
加入 high-water telemetry 之前的构建，而且 warmup query 数相差 1；因此它不是论文
最终版所需的同一构建、逐查询严格 A/B。pacer 固定 offered query rate，因此这里比较
延迟而不是宣称 capacity QPS 提升。

最初只使用 immutable build-time class 时，虽然 graph/total query RDMA bytes 分别
下降 `46.96%/41.85%`，但更新使热门静态记录反复 short→full：

- fallback `4.6395/query`，占 short read `2.4767%`；
- physical graph WQE 增加 `2.42%`，shard batch 增加约 `11.78%`；
- query mean/P99 分别退化 `0.77%/2.45%`。

这组负结果没有被丢弃。它直接推动了 device high-water 修复。加入修复后的同合同
历史对照：

| 指标（high-water 相对 fixed） | 变化 |
|---|---:|
| graph bytes/query | **-48.81%** |
| tracked query RDMA bytes/query | **-43.50%** |
| physical RDMA WQE/query | +0.042% |
| GPU graph / RDMA wait | **-7.52% / -4.48%** |
| query mean / P99 / P999 | **-3.44% / -1.52% / -3.02%** |
| insert mean / P99 / P999 | **-8.12% / -9.26% / -20.31%** |

正式 measurement 仅有 `650,483` 次 underhint fallback（`0.1355/query`），其中
`649,570` 次成功推进 high-water。相对 immutable-class run 的 `22,269,730` 次
fallback，依赖式重读减少约 `97.1%`；`graph_read_retries == fallback`，没有额外
full→full snapshot retry。初始 Recall 均为 `0.9401`，更新后 fixed/high-water 为
`0.9397/0.9395`；direct、route、Stage2 和 late-RPC failure 均为零。

准确结论是：high-water 把 fallback 从“每次热点访问都付费”变为“每次实际 class
增长只发布一次新 high-water；仅已读取旧 class 的并发在途查询仍可能重复 fallback”。
measurement 前的 30 s warmup 已经训练同一张 device high-water 表，所以
这里证明的是稳态收益，不是冷启动收敛速度。当前结果只覆盖 insert-only mixed 和一个
跨构建历史对照；它不覆盖 upsert/delete、逐查询 top-k 一致性或无限期运行后的 class
膨胀。正式论文证据仍需在当前同一构建下交错重跑 fixed/live。

## c1/c8/c64 压力对照

c1 使用 30 s + 120 s，c8/c64 使用 10 s + 30 s；三个 pair 都保持
Recall=0.9401、正式搜索工作和远程操作数一致：

| 指标 | c1 | c8 | c64 | c256 long |
|---|---:|---:|---:|---:|
| fixed total app RDMA GB/s | 0.057 | 0.453 | 3.248 | 10.514 |
| graph bytes/query | -47.82% | -49.74% | -49.25% | -49.64% |
| validation | -30.41% | -32.96% | -32.78% | -32.08% |
| RDMA wait | +13.08% | +8.10% | -2.04% | -17.80% |
| QPS | +1.07% | +2.55% | +4.38% | +8.79% |
| mean | -1.06% | -2.49% | -4.19% | -8.08% |
| P99 | -0.91% | -1.97% | -2.68% | -7.40% |

c1 的 P50/P95/P99/P999 全部改善；24 个 5 秒窗口中 23 个 live 更快，唯一反向窗口
只差 0.064%。c8 的全部分位也改善，六个窗口全部快于 fixed。两点的 RDMA wait
反而上升：低压力下少传约一半 graph payload 不会消除 one-sided READ 的固定 RTT/owner
成本。四个压力点的 validation 都减少约 30%--33%，构成基础计算收益；只有到 c256
接近 transport envelope 后，payload reduction 才明显解除 NIC/QP 排队。

因此 QPS 收益随压力从 c1 的 `1.07%`、c8 的 `2.55%`、c64 的 `4.38%`
增长到 c256 的 `8.79%`。该趋势不需要按并发调参，但也不能把约 `48%--50%`
字节减少机械外推成同量级查询加速。

原始结果：

```text
motivation/results/live_extent_e2e/live_c64_zero_elision/
motivation/results/live_extent_e2e/fixed_c64_zero_elision/
motivation/results/live_extent_e2e/live_c8_zero_elision/
motivation/results/live_extent_e2e/fixed_c8_zero_elision/
motivation/results/live_extent_e2e/live_c1_zero_elision/
motivation/results/live_extent_e2e/fixed_c1_zero_elision/
```

## 已有验证与尚需硬件验证

CPU/C++ 测试覆盖 sidecar checksum/绑定/overwrite、global ordinal/class、
header-only 与 R128 class、短读补零后的 full checksum、under-sized stale class
检测和 benchmark telemetry 派生。

四个 persistent CUDA entry 模板已用 ptxas 构建，entry function 均为 0 spill；
部分未内联 helper 仍按 ptxas 报告使用原有 stack/spill。query-local force-full 在源码中只引入
两个 32-byte shared flag vector；编译器复用互斥阶段空间后，链接产物相对 fixed 的 static
shared 实增 24 B，未跨过 occupancy 阈值。
在 ROB 中没有任何 extent-underhint evidence 的常见轮次，单 warp ballot 会直接跳过
positional/certified/associative handle matching、flag remap 和第二个 CTA barrier；门控标量
与已结束的 tail-stale 阶段共用 4 B union，因此不再增加 shared memory。当前链接产物中
fixed/ASFE 的 static shared 分别为 48,968/48,992 B；四个生产 entry 的 ptxas 寄存器数为
128/138（fixed 128/256 线程）和 154/154（ASFE 128/256 线程），入口均为 0 spill。
复用安全的 PQ checksum/读后复验会改变部分非生产模板的寄存器数，因此资源数以每次
构建的 ptxas/occupancy test 为准，不再声称所有模板逐项不变。fixed policy 在运行时跳过这两个
vector 的初始化、lookup/remap 和内部 barrier。

真实 mixed-length GPUNetIO 静态查询已经完成两个 smoke pair、一个 120 秒正式 pair
以及 c1/c8/c64 压力 pair；insert-only mixed 更新完成了 immutable-class 负结果和
high-water 修复后的严格 pair。GPU 测试覆盖 packed CAS 的非回退、unknown class、
并发提升和末尾 partial word，compute-sanitizer 未报告错误。跨数据集、upsert/delete、
逐查询 top-k/expansion hash 和长期 high-water 膨胀仍需独立验证。
