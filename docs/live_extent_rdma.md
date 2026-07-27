# Live-Extent RDMA

## 目标

Live-Extent RDMA 解耦远端图记录的两个尺度：

```text
存储分配：固定大小，保留原地更新余量
查询传输：一次 one-sided RDMA READ，只传当前邻接长度档覆盖的前缀
```

它不改变图记录格式、搜索父节点、Beam/visited、PQ、精排或存储 CPU 查询路径。
静态图稳态下，同一批次仍产生与 fixed 模式相同数量的 graph READ WQE、
descriptor、doorbell 和最终 CQE；唯一变化是每条 READ WQE 可以具有不同的
payload length。并发更新第一次把某条静态记录撑出旧长度档时，允许一次安全的
short→full fallback；GPU 随后单调修复该档位，避免热门节点反复支付依赖式双读。

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

## GPU 数据路径

`fixed` 是默认策略：

```text
--gpu-query-graph-read-policy=fixed
```

启用：

```text
--gpu-query-graph-read-policy=live-extent
```

启动时 live-extent 模式严格加载并校验 sidecar，分配：

- `align_up(num_nodes, 4)` B 的 packed device class high-water table；
- `query_slots * kPersistentMaxPrefetch * sizeof(u32)` 的 request-length
  scratch。

fixed 模式不加载 sidecar，也不分配这两个数组。

每个 graph batch 的执行顺序为：

1. 解析 handle。static base node 通过确定性 physical ordinal 读取 class；
   dynamic node 使用完整记录长度。
2. 保留原有固定大小 graph scratch，但短读不触碰未传输 suffix。校验器实际扫描
   header 声明的 required prefix，再用模 `2^32` 的 FNV prime 幂以 `O(log suffix)`
   延续逻辑零 suffix；class rounding 多取但未被 count 引用的 slots 也不必读取。
3. 按 shard 形成原有 descriptor，同时传入可选的 per-request length 数组。
4. QP owner 为每条 WQE 使用自己的长度；聚合、SQ credit、doorbell、最终 signaled
   WQE 和 CQ ownership 均保持不变。
5. completion 后先仅使用已到达 header 计算 exact required prefix。
6. 若 required prefix 超出本次 transfer，或逻辑补零后的完整结构/checksum 无效，
   下一 snapshot attempt 将该请求升级为 full read。
7. 只有“count 超出旧档位”触发的 full record 通过完整 checksum 后，CTA 才从该
   authoritative header 重新计算 class，并用 aligned-u32 packed CAS 单调提升对应
   byte。一个 CAS 保留同 word 的另外三个 class；首个成功 CAS 后的新查询不再为同一
   档位 fallback，已经读取旧 class 的多个在途查询仍可能各自升级一次。
8. full record 仍无效时沿用原有有界重读/fail-stop；dynamic slot incarnation 已变化
   时沿用原有 read-committed stale discard。checksum/torn-record fallback 不会更新
   class，dynamic handle 也永远不会访问该表。

因此 extent class 只是性能 hint，不是正确性 authority。并发更新使 class 过期只会
产生一次显式 full-read fallback 和后续 high-water 修复，不会截断邻接表或绕过
checksum/incarnation/tombstone 语义。high-water 与 persistent engine 生命周期一致；
benchmark telemetry reset 不会清空它，因此 warmup 可以建立正式测量所需的稳态。
删除或收缩不会降低 class：这可能在长期运行后多读一个长度档，但不会产生 under-read。

## Telemetry

query completion 和 benchmark JSON 新增：

```text
graph_read_bytes
graph_live_extent_reads
graph_full_record_reads
graph_extent_fallback_reads
graph_extent_underhint_reads
graph_extent_hint_promotions
```

其中 `graph_read_bytes` 是成功同步执行或成功进入 owner queue 的实际 graph payload
总和，包含 snapshot retry 和 fallback；不能用逻辑 `remote_pages * record_bytes`
替代。`short + full` 是物理 graph READ WQE 数，`remote_pages` 仍是逻辑正式父节点
读取数。`underhint` 只在首个 full fallback WQE 成功 admitted 时计数；
`promotions` 是成功的 class transition 数，同一节点可以随增长多次提升，竞争中观察到
别人已提升的 CTA 不重复计数。

RDMA trace schema 3 的 shard-batch event 记录：

```text
parent_count
payload_bytes
minimum_bytes_per_parent
maximum_bytes_per_parent
```

completion 粒度仍是 owner submission group 的最终 CQE 边界，不伪造 per-WQE
completion timestamp。

## A/B

固定和 live-extent 配置只改变 graph read policy：

```bash
./motivation/run_live_extent_ab.sh
```

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

生产 CUDA kernel 已用 ptxas 构建：query entry kernel 158 registers/thread、
owner kernel 130 registers/thread，二者 entry kernel 均无 spill；实现没有新增
shared per-candidate array。

真实 mixed-length GPUNetIO 静态查询已经完成两个 smoke pair、一个 120 秒正式 pair
以及 c1/c8/c64 压力 pair；insert-only mixed 更新完成了 immutable-class 负结果和
high-water 修复后的严格 pair。GPU 测试覆盖 packed CAS 的非回退、unknown class、
并发提升和末尾 partial word，compute-sanitizer 未报告错误。跨数据集、upsert/delete、
逐查询 top-k/expansion hash 和长期 high-water 膨胀仍需独立验证。
