# SIFT100M Experiment

实验只维护 `main` 上两个正式系统 profile。它们使用同一份二进制、同一个
schema-16/tagged-v2 OPQ/PQ32 索引、相同的 GPU/RDMA/线程/容量参数和相同的更新卸载
基础路径。完整系统必须能够由 baseline 仅开启三个贡献级模式得到，不能通过更换索引
结构、重调容量或切换分支制造性能差异。

## 配置

两个正式 profile 为：

```text
profiles/04_gpu_persistent_gpunetio_baseline.env  # baseline：三个贡献级机制均关闭
profiles/04_gpu_persistent_gpunetio.env           # full：三个贡献级机制均开启
```

两者唯一允许不同的 profile 字段是 `SYSTEM_VARIANT_LABEL` 和下面三个顶层模式：

| 贡献级机制 | baseline | full | 机制边界 |
| --- | --- | --- | --- |
| 动态图更新完成语义解耦 | `STORAGE_OWNER_UPDATE_COMPLETION_MODE=coupled` | `decoupled` | append-only 单-owner 同步完成与两阶段 ACK/durable 解耦、迁移维护 |
| 动态图存储访问粒度调节 | `GPU_DYNAMIC_GRAPH_ACCESS_MODE=fixed` | `adaptive` | 固定记录读取与动态 extent/高水位访问调节 |
| GPU-RDMA 搜索推进解耦 | `GPU_RDMA_SEARCH_PROGRESSION_MODE=coupled` | `decoupled` | CPU-owned Beam/visited、CPU-posted RDMA 严格波次推进与 GPU-owned ahead-of-commit 流水推进 |

所有子级实现参数都定义在二者共同 source 的
`04_gpu_persistent_gpunetio_common.sh` 中。顶层 mode 由程序解析为完整机制 bundle；
诸如 issue width、Stage2 score-many、home RPC combining、Beam merge 或具体 graph-read
policy 不是另外的论文贡献开关，也不得在 baseline/full 主对照中单独重调。
正式 mode 写入 service INI 时不会再同时写入其拥有的 child 选项，避免把 common 中的
容量默认误判成显式半开配置。只有历史微实验显式选择 access/progression `manual` 时，
生成器才透传 graph-read/extent 或 issue-width/Beam-merge child knobs。

baseline 保留研究对象本身所必需的存算分离、GPU 查询资源、storage-owner 更新卸载、
相同的图/PQ 表示和相同的安全边界；它关闭的是上述三个创新机制，而不是换成另一套
CPU-only 系统或旧格式索引。

其中 update `coupled` 是严格的 append-only 完成语义，而不是把 decoupled 队列深度
调小：它只接受 fresh insert，单一 logical storage owner 同步完成 schema-16 全局 RDMA
搜索、剪枝、本地稳定写入、本地 centroid add 和跨分片 one-sided RDMA 反向边后才返回
ACK。该路径没有远端更新 CPU handler、Stage1/Stage2、accepted backlog 或 maintenance
migration，并保持 `maintenance_sequence=0`。upsert/delete 会在任何 authority lease
或物理副作用前明确报 unsupported；不会退回远端 centroid/reclaim helper。需要完整
insert/upsert/delete 语义时必须使用 `decoupled`，由它启用快速 Stage1 ACK、后台批量
Stage2、迁移维护和 durable watermark。
coupled 插入在普通反向边和本地 centroid publication 之后重新安装一条 mandatory
stable bridge；该最后一次图发布是本次操作的物理图线性化点，随后立即提交同代 authority。
这不持有跨提交的 parent lease：之后另一个已经线性化的 prune 合法地可以移除该边。

search `coupled` 选择 HostOrchestrated 后端：CPU 持有 Beam/visited，按 commit width
逐波次 post/poll one-sided RDMA，并用有限生命周期 CUDA kernel 批量完成 PQ 与精确
距离计算；该模式不启动 persistent query kernel，也不使用 GPUNetIO query transport。
`decoupled` 选择 Persistent 后端，由 GPU 持有查询状态并允许 ahead-of-commit issue、
乱序完成吸收与流水推进。两种执行 owner 读取完全相同的 schema-16 图记录、PQ code、
动态 header、路由和精确向量，因此切换的是搜索执行/推进机制，而不是索引结构或召回预算。

两端使用同一数据、R、构图参数、硬件、CPU 核、客户端并发和测量时长；查询参数应在
独立 validation queries 上调到相同 recall 后冻结。主表同时报告前台 completion QPS、
最终/durable QPS、backlog/drain 和 drain 后 recall。任何未在时限内达到 durable
watermark 的 baseline/full 运行都只能作为失败诊断，不能作为持续更新吞吐结果。

## 三项贡献级消融

正式主对照直接运行 baseline 与 full：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio_baseline
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio_baseline

./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

做逐项/累积消融时从 baseline 开始，只覆盖三个顶层 mode。例如 `100`、`110`、`111`
依次表示开启更新完成语义解耦、再开启访问粒度调节、最后开启搜索推进解耦；`111`
必须与 full profile 等价：

```bash
export STORAGE_OWNER_UPDATE_COMPLETION_MODE=decoupled
export GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive
export GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio_baseline
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio_baseline
unset STORAGE_OWNER_UPDATE_COMPLETION_MODE \
  GPU_DYNAMIC_GRAPH_ACCESS_MODE \
  GPU_RDMA_SEARCH_PROGRESSION_MODE
```

每个 case 都必须使用同一组 mode 整组重启计算端和所有存储端。三个 mode 支持显式环境
覆盖，所以在切换回正式 profile 前必须 `unset`；否则当前 shell 中遗留的导出值会覆盖
profile 默认值。主对照不得覆盖 common 中的低层参数。常用机器路径覆盖项：

```bash
export HOSTS="192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202"
export INDEX_DIR=/data/xjs/index/dvstor_sift100m/index
export GPU_DEVICE=1
export GPU_MEMORY_LIMIT_GB=40
export GPU_MEMORY_RESERVE_GB=4
```

本次清理同时把 peer RPC 协议升级到 v15，并删除了已经停用的 speculative
score-many traffic class。该协议没有独立的能力协商，因此计算节点和全部存储节点
必须使用同一版本二进制并一起重启，不能混跑 v14/v15。

`MN_MEMORY_GB` 表示每个存储分片的 RDMA 注册区容量，不是进程总 RSS。未显式
设置时，启动脚本会在 profile 确定最终 `INDEX_PREFIX` 后估算：已有 schema-16
metadata 时直接使用每个真实分片的 `dynamic_node_base_offsets`、节点数和动态记录
步长，并按最坏分片取整；索引尚不存在时才按当前 tagged-v2 fixed/graph/PQ 布局和
`PARTITION_IMBALANCE` 做保守估算。默认给每个分片预留相当于基图节点数 20% 的
动态槽，可按 workload 调整：

```bash
# 按比例预留峰值同时存活/迁移在途的动态记录
export MN_DYNAMIC_HEADROOM_PERCENT=20

# 或直接指定每个物理分片的绝对动态槽容量；设置后覆盖百分比策略
export MN_DYNAMIC_SLOTS_PER_SHARD=4000000

# 完全覆盖自动估算，仍要求整数 GiB 且不超过 tagged pointer 的 256 GiB 上限
export MN_MEMORY_GB=24
```

这里的槽数是峰值同时占用量，不是累计更新次数；已完成清理的删除/迁移源槽可以
复用。默认注册区下限由 `MN_MEMORY_MIN_GB` 控制，默认 8 GiB。

`MAX_VECTORS` 只表示不可变基图的节点数，并必须与索引 metadata 一致；动态插入
节点的合法 ID 上界由独立的 `VECTOR_ID_NAMESPACE_SIZE` 控制。默认使用
`4294967295` 作为排他的上界，允许所有不会让 uint32 ID 生成器回绕的 ID。该配置
不会分配一个同等大小的稠密表，authority/idmap 状态仍只为实际存在的动态节点
按需创建。计算节点与所有存储节点必须使用相同的值。

## 构建新索引

存储节点脚本默认使用独立的 `build-storage`，先按根目录 README 配置 CPU-only
构建，不能与计算节点的 `build` 共用。

该命令先构建 compact Vamana/Metis 分片，再训练 OPQ/PQ32、写码流并生成
Live-Extent 长度档：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

传入 `04_gpu_persistent_gpunetio_baseline` 会解析到完全相同的 `INDEX_PREFIX`；只需构建
一次，禁止为 baseline 生成不同的图或 PQ sidecar。

完整构建以 schema-15 tagged graph 为中间态，最终直接生成 schema-16
OPQ/PQ32 运行索引和 `<prefix>.gextent8`，无需随后重编码。已有完整
schema-16 分片也可在持有全部 `.dat` 的机器上单独生成：

```bash
./build/vamana_graph_extent_indexer \
  --index-prefix "$INDEX_PREFIX"
```

工具会流式校验所有 graph record，再用同目录临时文件原子发布 sidecar；已有输出
必须显式传入 `--overwrite`。
推荐使用 `PQ_INDEX_PREFIX=/new/prefix` 保留已有索引；只有明确要删除目标 prefix
下旧产物并原地重建时才设置 `OVERWRITE_INDEX=1`。

## 转换旧 compact-v1 索引

完整、未包含在线 mutation 的旧 schema-15 `vamana_compact_v1` 基图可以流式
转换，不需要重新运行 Vamana 或 METIS，也不需要重新训练 OPQ/PQ。转换保持每个
物理分片的 slot 顺序，因此保留原始向量字节、图拓扑、METIS placement 和跨分片边；
它会重写 fixed record、5-byte compact edge、所有 `RemotePtr`，并重新生成 bound
idmap v2 和物理分片 centroid v2。旧 PQ model 被复用，base PQ codes 从精确向量
重新编码。

先执行只读的全量校验：

```bash
./build/vamana_legacy_index_converter \
  --input-prefix "$OLD_PREFIX" \
  --output-prefix "$NEW_PREFIX" \
  --dry-run
```

校验通过后写到新的 prefix：

```bash
./build/vamana_legacy_index_converter \
  --input-prefix "$OLD_PREFIX" \
  --output-prefix "$NEW_PREFIX" \
  --chunk-vectors 65536 \
  --threads 32
```

转换器禁止原地执行或覆盖已有输出，并在所有分片、idmap 和 centroid 完成后最后
发布 metadata。输入至少需要 metadata、全部旧 `.dat` 和旧 `.pqM` model；旧
`.codes`、`.anchors`、原始 dataset 和旧 idmap 都不是恢复静态基图所必需的。
`--graph-only` 可停在新的 tagged schema-15 中间态。转换器拒绝 deleted、非零
generation 或含动态 slot 的运行时快照；这类在线状态没有足够的旧持久化语义可安全
映射到 incarnation/provisional/centroid 新契约，只能从一致的静态快照重新生成。

## 部署文件

计算节点（查询与更新）：

```text
<prefix>.meta.json
<prefix>.pq32
<prefix>.gextent8        # baseline/full 共用并在两者启动时校验
```

正式 baseline/full 都要求 `<prefix>.meta.json`、`<prefix>.pq32` 和
`<prefix>.gextent8`。baseline 的 fixed 路径运行时不消费 extent class，但启动时仍会
校验 sidecar 的完整性、布局和 build fingerprint，确保 full 不需要换索引或补文件。
只有显式的 manual+fixed 工程调试模式可以不部署 `.gextent8`。该文件只有
1 byte/base-node 的长度档，不包含邻接表。纯查询配置使用
`enable-updates = false`，不会启动
更新执行器。在线 mutation 会持续
更新 storage owner 的 centroid publication；每个计算节点从 storage 拉取同一版本化
快照，因此其他计算节点写入的新代表节点同样可见。

存储节点 X：

```text
<prefix>.meta.json
<prefix>_nodeX_ofN.dat
<prefix>_nodeX_ofN.idmap
<prefix>_nodeX_ofN.centroid
<prefix>_nodeX_ofN.pq32.codes
```

计算节点不需要 `.dat`、`.idmap`、`.pq32.codes` 或 `.gpu.idx`。Live-Extent
sidecar 应在持有全部 `.dat` 的构建/存储机器上生成，再复制到计算节点的同一 prefix。
METIS 只决定物理
placement；基础和动态 ID 的逻辑 authority 都由 `ID % N` 确定，其存储端 idmap
负责解析当前物理记录。因而增加计算节点不会复制一份 O(N) 的 ID 目录；每个存储
节点只加载自己的 `owner_sharded_v2_bound` idmap。该文件与整次构建和 owner
分片指纹强绑定，并校验完整长度、payload/header checksum、`ID % N`、tagged
`RemotePtr` 静态范围及重复 ID；旧 v1 会被直接拒绝。
加载后基础项只保留紧凑的 `ID -> RemotePtr`，完整的代际、提交回执和迁移状态仅为
实际参与 mutation 的 ID 分配。离线 writer 同样以每 owner 临时流单遍分桶，不在
内存中复制一份全量 idmap payload。
旧索引不能通过复制或改名 sidecar 升级：schema-16 运行格式、8-byte tagged
`RemotePtr`、构建/分片指纹、centroid sidecar v2 和 PQ code header 是同一次构建的
绑定契约。完整的 compact-v1 静态 `.dat` 可使用上面的转换器重写这些契约；缺失
`.dat` 或只有 anchor/idmap/PQ model 时信息不足，才必须重新构图。存储节点运行时
不需要 `.pq32` 模型。

## 启动

访问粒度 mode 的内部实现包含以下 graph-read policy：

```text
gpu-query-graph-read-policy = fixed        # fixed bundle：始终读取完整记录
gpu-query-graph-read-policy = live-extent  # adaptive bundle：一次 READ 读取有效前缀
```

`live-extent` 不改变图记录、Beam/visited、扩展顺序或存储 CPU 路径。静态稳态下，
每个逻辑 base-node graph fetch 仍只产生一个 READ WQE，只缩短 payload；dynamic
handle 始终读取完整记录且不访问长度档。磁盘 sidecar 是 1 byte/base-node，启动时
加载为 packed、aligned-u32 GPU high-water 表，占 `align_up(num_nodes, 4)` bytes；
SIFT100M 约 100 MB。

并发更新使档位落后时，短读只有在 header/count 可验证且
`required_bytes > transfer_bytes` 时才记为 underhint；其首个成功 admitted 的 full
fallback WQE 计入 `graph_extent_underhint_reads`。authoritative full record
通过完整 checksum 后，GPU 才重新计算档位并用 packed CAS 单调提升对应 byte，后续
查询直接使用新 high-water；已经读取旧 class 的并发在途 query 仍可能各自完成一次
fallback。checksum/torn-record 等其他 fallback 不学习档位，
删除或收缩也不会降低档位。fallback 继续沿用原有有界 snapshot retry 和错误语义，
因此更新期可能临时增加 WQE，但不会截断邻接表。相关统计包括
`graph_live_extent_reads`、`graph_full_record_reads`、
`graph_extent_fallback_reads`、`graph_extent_underhint_reads`、
`graph_extent_hint_promotions` 和实际 `graph_read_bytes`；promotions 统计成功的
class transition，同一节点随增长可多次提升。

shared common 固定使用 C16 authoritative expansion、`stable-run` Beam merge 和
`live-extent` 所需的容量上限，并关闭详细 RDMA trace。baseline 的顶层 fixed mode
会把自适应访问 bundle 作为整体关闭；这些 common 值不构成 baseline 中偷偷开启的
独立优化。profile 不写死 workload、更新开关或客户端并发，因此纯查询和混合负载可
使用同一套系统配置。

启动 baseline：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio_baseline
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio_baseline
```

启动 full：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

两个命令都在当前 `main` 源码树运行并解析同一个 `INDEX_PREFIX`。切换 profile 时
必须停止并整组重启所有节点；禁止混跑不同 mode 的计算端/存储端。

如果每个分片位于不同主机，可分别执行：

```bash
./experiment/start_memory_node.sh 1 04_gpu_persistent_gpunetio_baseline
# 或：./experiment/start_memory_node.sh 1 04_gpu_persistent_gpunetio
```

启动脚本会验证 schema、分片数、R、dtype、PQ checksum 和角色所需文件，
不兼容时在申请大块注册内存前退出。

运行格式固定为 schema 16，peer RPC 协议固定为版本 15。所有动态图指针携带
`{shard, offset, incarnation}`，记录头在读取和修改前都校验 incarnation；删除地址只在
对应维护序列 durable 后进入复用流程。查询采用 incarnation-tagged read-committed
语义，不要求动态加入的计算节点参与全局 ACK；incarnation 耗尽的槽位永久退休而不回绕。
动态目标分配使用无超时驱逐的 receipt：只有源记录进入终态且目标记录的精确身份已
确认后才结算，避免迟到重试把新对象误认为旧对象。混用旧二进制、旧分片、旧
centroid 或旧 PQ code 会在启动校验时失败。

Stage2 finalized 的等价边界是同一逻辑快照下延续 Stage1 的宽度 `L`
beam/visited/frontier，沿图中实际跨分片边完成 one-sided-RDMA 扩展后执行一次相同
RobustPrune，并等待本次 insert 所选邻居的反向边完成；它不会为每个分片重启独立
搜索，也不等价于离线 builder 的全候选构图。当前也没有
完整入边索引，所以 delete/upsert 不能同步清除所有历史未知入边；报告中的 durable
或 drained 仅表示已声明的 maintenance 任务完成，不应解释为全图整理已经完成。

控制页通过 descriptor 指向独立、可变长度的 centroid route publication；容量由
维度、标量类型和 live-entry 上限计算，不受 4 KiB 控制页固定槽位约束。旧存储
二进制没有该运行时扩展，因此新计算节点会在启动校验时拒绝混合部署；必须同步
升级全部二进制，并使用新 builder 重建或用上述工具完整转换索引。

maintenance observation 同时输出窗口可差分的 locality 计数器：Stage2 continuation
次数、远端 frontier/展开/评分记录数、迁移数，以及以 Stage1 home 和最终 home 计算的
跨分片边数。benchmark 只对测量窗口前后的单调累积值做差，报告
`home_match_rate`、`cross_edge_reduction_ratio` 和每次 continuation 的平均远端工作量；
这些指标包含窗口内全部请求，不使用请求抽样或数据集专用捷径。

动态节点的 PQ code 和图记录都以存储节点上的权威记录为准，GPU 查询通过
one-sided RDMA 按需读取，不维护需要广播、同步或回收的计算侧 dynamic-PQ 副本。
stage2 context 有界回收；存储节点记录同时保持逻辑 generation 与物理 incarnation
稳定，并在 durable watermark 后以 header-last 方式发布复用后的新 incarnation。

性能阶段结束时，driver 会一次性读取所有分片的
`next_maintenance_sequence - 1`，固定为全局 maintenance 前缀，再等待每个分片的
`durable_maintenance_sequence` 追平。这个边界同时覆盖跨分片 upsert 的旧 home
清理、新 home Stage2 以及迁移产生的远端退休任务；后续才提交的更大序号不会被
本轮 drain 无限等待。该控制读只发生在显式阶段边界，不进入逐更新热路径。

## 召回率与性能

测试负载参数不放在索引/系统 profile 中。`BENCHMARK_CLIENT_THREADS`、
`WORKLOAD`、`READ_RATIO`、`WARMUP_SECONDS`、`MEASURE_SECONDS` 和
`RECALL_QUERIES` 由运行脚本读取。`SERVICE_THREADS` 是计算服务 CPU 线程数，
不等于 benchmark 客户端并发数。

`BENCHMARK_CLIENT_THREADS` 默认为 `auto`。由于当前 load driver 每个线程同步
提交一个请求，一个线程最多贡献一个在途操作；`auto` 因此只使用系统的
显式有界容量推导闭环并发，不根据已测 QPS 调参：

- query 使用 `GPU_QUERY_SLOTS`；insert 使用
  `SHARDS * STORAGE_OWNER_RPC_DEPTH`；`both` 的两个阶段顺序执行，取两者最大值。
- `mixed/fixed_threads` 保证按 `READ_RATIO` 分配后，活跃路径至少有对应的
  容量数在途 caller。例如 256 个 GPU 查询槽、50/50 混合负载会推导为
  512 线程，即 256 读 + 256 写。`READ_RATIO` 在该模式表示 caller 比例，
  不保证不同延迟的读写最终完成量也恰好按此比例。
- `mixed/probability` 使每个闭环 caller 在上一操作完成后按 `READ_RATIO`
  选择下一操作，适合固定长时操作混合；它不保留专用读/写 caller。
- `mixed/rate_limited` 对激活的读、写路径容量求和；调度数、完成数和
  drain 均如实报告，不会把超载时未发出/未完成的计划请求算作完成来伪造达标。

`BENCHMARK_CLIENT_THREAD_CAP` 默认为 1024，只限制 `auto` 意外创建过多
OS 线程；如果截断了推导值，脚本会明确告警而不声称路径已饱和。
显式设置正整数 `BENCHMARK_CLIENT_THREADS` 可用于 1/2/4/... 延迟—吞吐扫描。
脚本会在终端输出完整推导，并在 JSON 的
`meta.benchmark_driver_concurrency` 中保留容量、来源、cap 和计算式。
`auto` 是一个与数据集无关的容量起点，不是“已饱和”的自动结论。论文实验应显式
公布对应 `auto/4`、`auto/2`、`auto`、`2*auto` 整数并发的完整延迟—吞吐曲线，并且只有在
零错误、召回不变且 durable backlog 有界时，平台点才能称为饱和吞吐。

`query.u8bin` 的 10K 标准查询仅供 recall 使用。性能阶段由
`PERFORMANCE_QUERY_FILE` 提供独立查询流，warmup 与 measure 共用一个单遍游标，
同一行不会再次执行；查询池耗尽时 benchmark 会失败而不是取模回绕。当前默认
性能查询池为 `[100M,110M)` 的 1000 万行，默认插入池为与之相邻且不重叠的
`[110M,120M)` 1000 万行。文件默认位于
`/data/xjs/datasets/sift1b`：

```text
sift100m_to_110m_query.u8bin
sift110m_to_120m_insert.u8bin
```

脚本会显示在 warmup + measure 全时段内不重用行所能支持的最大平均 query
QPS。例如 1000 万行与 30 + 120 秒只能为 query 部分提供平均
66,666.7 QPS；要完整测量 query-only 100K QPS，至少需要 1500 万不重复行，
或将两个时段总时长降到 100 秒以内。增加并发不会绕过这个数据诚信上限；
查询池耗尽仍会使 benchmark 失败。

`run_breakdown.sh` 默认只校验并读取预生成文件，不会在计算节点寻找
`bigann_base.bvecs`。只有显式设置 `PREPARE_BENCHMARK_DATA=1` 时才会调用数据准备。
可通过
`PERFORMANCE_QUERY_FILE`、`INSERT_FILE` 覆盖路径，或用以下变量
调整源区间：`PERFORMANCE_QUERY_START`、`PERFORMANCE_QUERY_END`、
`INSERT_VECTOR_START`、`INSERT_VECTOR_END`。常用选项可直接查看
`./experiment/run_breakdown.sh --help`。例如：

```bash
PERFORMANCE_QUERY_FILE=/data/xjs/datasets/sift/perf_queries_2m.u8bin \
INSERT_FILE=/data/xjs/datasets/sift/inserts_2m.u8bin \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

先做 query-only 召回验证：

```bash
RECALL_QUERIES=1000 \
./experiment/run_recall.sh 04_gpu_persistent_gpunetio
```

该脚本使用 `--recall-only`，不会执行 warmup/measure，也不会加载性能查询池。

再运行读写混合负载：

```bash
WORKLOAD=mixed READ_RATIO=0.5 \
WARMUP_SECONDS=30 MEASURE_SECONDS=120 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

这是饱和闭环吞吐测试，不是单客户端延迟测试。与单机 HNSW 比较时必须同时
固定数据集、召回率/搜索参数、更新语义和负载比例，并让两个系统都有
足够客户端达到各自饱和点。单纯 query-only HNSW 的 QPS 不能直接与包含权威
Stage2 drain 的混合读写 QPS 解释为同一指标。

如需比较不同运行，保持相同的索引、查询/插入文件和 GPU 参数即可。
报告只提供吞吐、延迟、召回、GPU 内存与 stage2 遥测；不包含自动验收结论。
JSON 的 `meta.system_variant`（其中 mode 位于 `resolved_modes`）和文本报告开头同时记录
profile label、程序解析后的三个 umbrella modes、`INDEX_PREFIX`、schema version 与
index build fingerprint；主对照应先核对这些字段，确保除三个 mode/label 外使用的是
同一个索引和配置契约。

短跑示例：

```bash
WORKLOAD=query RECALL_QUERIES=100 \
WARMUP_SECONDS=1 MEASURE_SECONDS=5 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

结果写入 `experiment/reports/04_gpu_persistent_gpunetio/`。报告保留下列原始指标，
由实验者结合目标负载自行分析：

- `gpu_persistent.direct_path_failures == 0`；
- 前后 recall 及其变化；
- 没有 unhealthy/fail-stop 日志；
- GPU 和 RDMA 指标显示多查询并发，而非单查询串行等待。

可与 OdinANN 或历史 JSON 比较：

```bash
python3 experiment/compare_reports.py \
  --baseline /path/to/odinann.json \
  --candidate experiment/reports/04_gpu_persistent_gpunetio/latest.json
```

比较工具只输出原始吞吐、延迟、加速比和 recall 差值，不给出自动通过/失败结论。

停止本机启动的存储进程：

```bash
./experiment/stop_memory_nodes.sh
```
