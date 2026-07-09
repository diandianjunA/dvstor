# DVSTOR 源码架构 30 课时学习大纲

生成日期：2026-07-08

## 编写原则

本大纲以代码为准，不以 README、实验说明或既有文档中的架构描述作为事实依据。课时主题来自以下代码入口的交叉阅读：

- 构建入口：`CMakeLists.txt`、`cmake/`
- 在线运行时：`src/main.cc`、`src/memory_node_main.cc`、`src/service/`、`src/memory_node/`
- Vamana 索引：`src/vamana/`
- RDMA 抽象：`src/rdma/`、`rdma-library/library/`
- GPU 执行：`src/gpu/`
- 路由与服务接口：`src/router/`、`src/http/`
- 离线构建与工具：`tools/vamana_offline/`、`tools/*.cc`
- 性能观测：`src/service/breakdown/`、`src/common/statistics.hh`、`tools/breakdown_benchmark/`

每课时默认按 60 到 90 分钟设计。课程正文暂不生成，本文件只给出学习路线、代码阅读入口、应掌握问题和阶段性产出。

## 总体学习目标

完成 30 课时后，应能够：

1. 从构建系统和入口函数恢复项目的真实模块边界。
2. 解释 compute node、memory node、RDMA、GPU、Vamana 图索引之间的运行关系。
3. 读懂查询、插入、离线构建、索引加载、storage-owner 更新、peer RDMA/RPC 的主路径。
4. 建立性能分析框架，能定位 CPU、GPU、RDMA、内存布局、调度策略的主要开销。
5. 识别设计不合理点，形成可验证的大规模重构或性能优化方案。

## 阶段划分

| 阶段 | 课时 | 主题 |
| --- | --- | --- |
| 阶段一 | 01-05 | 代码地图、构建目标、运行入口、基础类型、并发骨架 |
| 阶段二 | 06-11 | Vamana 图索引核心数据结构、搜索、插入、离线构建 |
| 阶段三 | 12-17 | RDMA 传输、远程内存、memory node、storage-owner、跨 shard 维护 |
| 阶段四 | 18-23 | GPU 执行、RaBitQ、查询流水线、多 compute node 路由 |
| 阶段五 | 24-27 | 索引格式、加载存储、服务接口、性能统计与实验工具 |
| 阶段六 | 28-30 | 测试缺口、性能建模、重构与优化路线设计 |

## 30 课时大纲

### 第 01 课：从构建系统恢复真实模块边界

代码入口：

- `CMakeLists.txt`
- `cmake/DvstorDependencies.cmake`
- `cmake/DvstorTargetHelpers.cmake`

学习重点：

- 区分 `dvstor_runtime`、`dvstor`、`dvstor_memory_node`、`dvstor_gpu_kernels`、离线工具和 benchmark 工具。
- 理解 `DVSTOR_STORAGE_NODE_ONLY`、`DVSTOR_BUILD_EXECUTABLES`、`DVSTOR_BUILD_TESTS`、`DVSTOR_METIS_PARTITION` 对源码编译边界的影响。
- 从 `DVSTOR_MEMORY_NODE_SOURCES`、`DVSTOR_OFFLINE_BUILDER_SOURCES`、`DVSTOR_RUNTIME_SOURCES` 推导真实依赖关系。

课时产出：

- 一张“target 到源码文件”的依赖表。
- 一张“哪些文件属于在线查询、在线插入、memory node、离线工具、GPU kernel”的归类表。

### 第 02 课：运行入口与进程角色

代码入口：

- `src/main.cc`
- `src/memory_node_main.cc`
- `src/common/configuration.hh`
- `rdma-library/library/configuration.hh`

学习重点：

- 理解 `config.is_server` 如何决定进程作为 compute node 或 memory node。
- 梳理 `ComputeService<L2Distance/IPDistance>` 与 `MemoryNode` 的生命周期入口。
- 识别 CLI 配置项如何影响搜索、插入、RDMA、GPU、storage-owner、RaBitQ、routing。

课时产出：

- 一张“命令行参数到子系统行为”的映射表。
- 一张 compute node 和 memory node 启动时序图草稿。

### 第 03 课：公共类型、ID、指针和向量 dtype

代码入口：

- `src/common/types.hh`
- `src/common/constants.hh`
- `src/common/vector_dtype.hh`
- `src/common/distance.hh`
- `src/remote_pointer.hh`

学习重点：

- 理解 `node_t`、`element_t`、`distance_t`、`byte_t` 等基础别名。
- 读懂 `RemotePtr` 的 memory node 编码和 byte offset 语义。
- 分析 float32、uint8、int8 查询和存储向量在距离计算中的解码路径。

课时产出：

- 一份“本项目核心类型词典”。
- 一张 `RemotePtr -> memory node -> RDMA offset -> VamanaNode` 的解释图。

### 第 04 课：线程、协程与工作队列骨架

代码入口：

- `src/coroutine.hh`
- `src/compute_thread.hh`
- `src/worker_pool.hh`
- `src/service/compute_service/lifecycle.ipp`

学习重点：

- 理解 `VamanaCoroutine`、`StorageOwnerInsertCoroutine` 的 promise 和调度模型。
- 分析 `ComputeThread` 如何用 `post_balances` 和 `gpu_post_balances` 同时等待 RDMA 与 GPU。
- 梳理 query worker、insert worker、coroutine 数量之间的关系。

课时产出：

- 一张“请求进入队列后如何被线程和协程执行”的状态流转图。
- 标注出容易产生饥饿、阻塞或过度轮询的位置。

### 第 05 课：全局资源上下文与内存分配

代码入口：

- `src/shared_context.hh`
- `src/buffer_allocator.hh`
- `rdma-library/library/hugepage.hh`
- `rdma-library/library/memory_region.hh`

学习重点：

- 理解 hugepage buffer、local memory region、remote token 的关系。
- 读懂 `SharedContext` 中 QP pool、completion slot、outstanding WR 统计的职责。
- 分析 `BufferAllocator` 在 RDMA staging、pointer slot、临时 buffer 中的作用。

课时产出：

- 一张 compute node 本地内存区域和 RDMA 注册区域布局图。
- 一个初步的内存生命周期风险清单。

### 第 06 课：Vamana 图索引算法在代码中的边界

代码入口：

- `src/vamana/vamana.hh`
- `src/vamana/vamana_helpers.ipp`
- `src/vamana/vamana_neighborlist.hh`
- `src/vamana/vamana_neighborlist.cc`

学习重点：

- 明确 `Vamana<Distance>` 模板持有的参数：`R`、`beam_width`、`beam_width_construction`、`alpha`、`k`、`dim`。
- 区分搜索路径、插入路径、辅助函数和 RDMA/GPU 调用边界。
- 理解 neighbor list 抽象与远端节点布局之间的关系。

课时产出：

- 一张 Vamana 类公开能力和内部配置的结构图。
- 一份“算法概念到源码函数”的索引表。

### 第 07 课：VamanaNode 存储布局与 hot graph

代码入口：

- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_node.cc`
- `src/vamana/storage_format.hh`
- `src/vamana/hot_graph.hh`

学习重点：

- 读懂 AoS、compact storage、hot graph plane 的节点布局。
- 理解 header bit、id、generation、edge_count、neighbors、vector、RaBitQ entry 的 offset 计算。
- 分析静态全局布局变量 `DIM`、`R`、`VECTOR_DTYPE`、`STORAGE_FORMAT` 带来的设计风险。

课时产出：

- 一张字节级节点布局图。
- 一份“布局相关隐式全局状态”的重构风险清单。

### 第 08 课：在线查询主路径之一：入口点、medoid 与 beam 初始化

代码入口：

- `src/service/compute_service/search.ipp`
- `src/vamana/vamana_search.ipp`
- `src/rdma/vamana_rdma_reads.hh`

学习重点：

- 从 `ComputeService::search` 跟踪到 query queue，再到 `Vamana::knn_raw`。
- 理解 entry point 来源：storage-owner anchor hints 或 medoid。
- 分析首次节点读取、距离计算、beam 初始化、visited set 的执行成本。

课时产出：

- 一张“search API 到 Vamana::knn_raw”的调用链。
- 标注查询路径中第一次 RDMA、第一次 CPU distance、第一次 GPU transfer 的位置。

### 第 09 课：在线查询主路径之二：批量 beam expansion

代码入口：

- `src/vamana/vamana_search.ipp`
- `src/rdma/vamana_rdma_reads.hh`
- `src/rdma/vector_batch_planner.hh`

学习重点：

- 理解 K-way batched beam expansion 的选择、预提交、邻居读取和候选过滤流程。
- 分析 `expansion_batch` 与 `credit_aware_expansion` 的动态控制逻辑。
- 理解 batch vector read 如何把候选按 memory node 和 QP 切分。

课时产出：

- 一张 beam expansion 单轮执行图。
- 一份 `expansion_batch`、QP credit、候选数、GPU batch size 之间的约束关系表。

### 第 10 课：在线查询主路径之三：GPU 距离计算与 rerank

代码入口：

- `src/vamana/vamana_search.ipp`
- `src/gpu/gpu_kernel_launcher.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/kernels/distance_kernels.cuh`

学习重点：

- 跟踪 query H2D、candidate vectors staging、kernel launch、distance D2H。
- 区分普通 host staging、GPUDirect candidate RDMA、indirect candidate pointer path。
- 分析 GPU kernel 粒度、同步点和 `GpuAwaitable` 对协程调度的影响。

课时产出：

- 一张“RDMA 读候选向量到 GPU distance 完成”的数据流图。
- 一个 GPU 路径潜在优化点清单。

### 第 11 课：在线插入与 RobustPrune

代码入口：

- `src/vamana/vamana_insert.ipp`
- `src/gpu/gpu_kernel_launcher.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/rdma/vamana_rdma_writes.hh`
- `src/rdma/vamana_rdma_atomics.hh`

学习重点：

- 理解 first insert 对 medoid 的 CAS 竞争处理。
- 跟踪插入时的 beam search、候选向量批量读取、GPU 距离计算。
- 理解 RobustPrune 候选收集、GPU prune、写新节点、反向边更新。

课时产出：

- 一张插入流程图。
- 一份“插入一致性和并发冲突点”检查表。

### 第 12 课：MemoryNode 基础架构与远程内存布局

代码入口：

- `src/memory_node/memory_node.hh`
- `src/memory_node/memory_node.cc`
- `src/memory_node/command_protocol.hh`
- `src/memory_node/storage_owner_state.hh`

学习重点：

- 理解 memory node 的职责：内存分配、索引 shard 存储、命令处理、peer 通信、storage-owner 插入。
- 分析固定 region 中 free pointer、entry pointer、节点区的组织方式。
- 读懂 LOAD、STORE、SHUTDOWN 命令协议。

课时产出：

- 一张 memory node 进程内部组件图。
- 一张 memory node RDMA region 逻辑布局图。

### 第 13 课：RDMA library 底层抽象

代码入口：

- `rdma-library/library/context.hh`
- `rdma-library/library/context.cc`
- `rdma-library/library/queue_pair.hh`
- `rdma-library/library/connection_manager.hh`
- `rdma-library/library/detached_qp.hh`

学习重点：

- 理解 protection domain、completion queue、queue pair、memory region 的封装。
- 区分 server/client connection manager 和 detached QP。
- 分析 QP 连接、post send、poll completion 的抽象边界。

课时产出：

- 一张 RDMA verbs 对象关系图。
- 一份底层 RDMA wrapper 的错误处理和资源释放风险清单。

### 第 14 课：Vamana RDMA read/write/atomic 封装

代码入口：

- `src/rdma/vamana_rdma_reads.hh`
- `src/rdma/vamana_rdma_writes.hh`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/rdma_send_chain.hh`

学习重点：

- 理解 medoid pointer、node、neighbor list、vector batch 的远程读取。
- 分析 chained READ WR、batch completion token、QP outstanding credit。
- 读懂 lock、CAS、header write、node write 的原子性边界。

课时产出：

- 一张 RDMA operation 到 `RemotePtr` offset 的映射表。
- 一份“哪些操作依赖远端内存一致性”的清单。

### 第 15 课：Storage-owner 插入协议

代码入口：

- `src/service/compute_service/storage_owner_insert.ipp`
- `src/service/storage_owner_protocol.hh`
- `src/service/storage_owner_client_helpers.hh`
- `src/memory_node/storage_owner_runtime.cc`
- `src/memory_node/storage_owner_index.cc`

学习重点：

- 理解 compute side 如何把 insert/upsert/erase 封装为 storage-owner RPC batch。
- 分析 owner storage 选择、batch wait、slot、request/response buffer 的实现。
- 跟踪 memory node 侧接收、解析、执行和返回结果。

课时产出：

- 一张 storage-owner mutation 的端到端时序图。
- 一份 batch size、timeout、RPC depth 对吞吐和尾延迟的影响假设。

### 第 16 课：Storage-owner 本地索引更新与 freshness

代码入口：

- `src/memory_node/storage_owner_index.cc`
- `src/memory_node/storage_owner_anchor.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/vamana/idmap.hh`

学习重点：

- 理解本地 owner 如何维护 id 到 `RemotePtr` 的映射、generation、deleted 标记。
- 分析 exact update 与 local-stitch update 的入口点和维护任务。
- 梳理 anchor hints、foreground search、background finalize 的职责差异。

课时产出：

- 一张 insert/upsert/delete 对 freshness table 和图边的影响图。
- 一份 storage-owner 路径与 compute-side 插入路径的差异表。

### 第 17 课：MemoryNode peer RDMA/RPC 与跨 shard 维护

代码入口：

- `src/memory_node/peer_rdma.cc`
- `src/memory_node/peer_rpc.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/memory_node/storage_owner_runtime.cc`

学习重点：

- 理解 peer 之间的 RDMA read/write/CAS 如何服务跨 shard 节点访问。
- 分析 reverse update、cleanup deleted、stitch search 的 peer RPC 消息路径。
- 识别 async/sync reverse mode 对一致性和延迟的影响。

课时产出：

- 一张跨 shard reverse update 时序图。
- 一份 peer RPC 队列、credit、timeout、重试/失败处理的风险清单。

### 第 18 课：GPU 资源管理与异步完成

代码入口：

- `src/gpu/gpu_buffer_manager.hh`
- `src/gpu/gpu_buffer_manager.cu`
- `src/gpu/gpu_awaitable.hh`
- `src/gpu/gpu_awaitable.cc`
- `src/gpu/compute_thread_gpu.cc`

学习重点：

- 理解每个 coroutine 的 stream、event、host pinned buffer、device buffer。
- 分析 candidate buffer 双缓冲和 GPUDirect RDMA 注册条件。
- 读懂 `poll_gpu_events` 如何解除 coroutine 的 GPU 等待。

课时产出：

- 一张 per-coroutine GPU buffer 布局图。
- 一份 GPU buffer size 与 `dim`、`R`、`beam_width`、`max_batch` 的关系表。

### 第 19 课：CUDA kernel 与距离计算实现

代码入口：

- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/kernels/distance_kernels.cuh`
- `src/common/distance.hh`

学习重点：

- 读懂 typed L2/IP distance 的 CPU/GPU 两套路径。
- 分析 single-query、multi-query、id-based、indirect pointer distance kernel 的差异。
- 识别 kernel launch overhead、memory coalescing、dtype decode 的优化空间。

课时产出：

- 一份 GPU kernel 分类表。
- 一份 kernel 级性能假设和待验证指标列表。

### 第 20 课：RaBitQ cache、sidecar 与 gate

代码入口：

- `src/vamana/rabitq_cache.hh`
- `src/vamana/vamana_node.hh`
- `tools/vamana_rabitq_sidecar_converter.cc`
- `tools/vamana_offline/anchor_builder.cc`

学习重点：

- 理解 RaBitQ entry、code bits、norm quantization、query LUT、lower bound 的实现。
- 分析 `select_gate_into` 如何选择需要 exact vector read 的候选。
- 跟踪 sidecar 和 dynamic slot 如何支持静态索引与在线变更。

课时产出：

- 一张 RaBitQ 估计距离到 exact rerank 的数据流图。
- 一份 recall 风险与 strict/audit 参数的关系表。

### 第 21 课：ComputeService 架构与生命周期

代码入口：

- `src/service/compute_service.hh`
- `src/service/compute_service/lifecycle.ipp`
- `src/service/compute_service/index_commands.ipp`
- `src/service/index_metadata.hh`
- `src/service/index_metadata.cc`

学习重点：

- 理解 `ComputeService` 持有的队列、worker、RDMA token、RPC state、index metadata。
- 梳理 start/stop/pause/resume worker 和 RPC 的职责。
- 分析 load/store index 命令如何协调 compute node 与 memory node。

课时产出：

- 一张 `ComputeService` 字段分组图。
- 一份 lifecycle 中异常退出、部分失败和资源清理风险清单。

### 第 22 课：查询服务路径、RPC routing 与结果汇总

代码入口：

- `src/service/compute_service/search.ipp`
- `src/service/compute_service/rpc_routing.ipp`
- `src/router/query_router.hh`
- `src/router/placement.hh`
- `src/router/kmeans.hh`
- `src/router/message_wrapper.hh`

学习重点：

- 区分 local search、initiator routing、proxy search、remote search response。
- 分析 `choose_destination`、routing centroid、inflight 计数与负载分配。
- 理解 multi-compute-node RPC message 的 header、payload 和 result 限制。

课时产出：

- 一张 query routing 决策树。
- 一份 routing 正确性和负载均衡的测试场景列表。

### 第 23 课：HTTP 服务层与外部请求模型

代码入口：

- `src/http/service_types.hh`
- `src/http/vamana_service_scheduler.hh`
- `src/service/compute_service.hh`

学习重点：

- 理解 QueryRequest、InsertRequest、QueryResult 等服务层类型。
- 分析 scheduler 如何把外部请求映射到 compute service 队列。
- 识别 API 层与核心索引层的耦合点。

课时产出：

- 一张 HTTP/service type 到内部队列对象的映射表。
- 一份未来重构 API 边界的候选方案。

### 第 24 课：离线 Vamana 构建器

代码入口：

- `tools/vamana_offline_builder.cc`
- `tools/vamana_offline/config.hh`
- `tools/vamana_offline/dataset_io.hh`
- `tools/vamana_offline/graph.hh`
- `tools/vamana_offline/graph.cc`

学习重点：

- 理解离线 Dataset、VamanaGraph、compute_medoid、beam_search、robust_prune。
- 对比离线构建与在线插入路径的算法和并发差异。
- 分析离线构建的锁粒度、线程并行和候选剪枝实现。

课时产出：

- 一张离线 build graph 流程图。
- 一份离线/在线 Vamana 实现差异表。

### 第 25 课：分区、shard 写出、metadata 与 index load

代码入口：

- `tools/vamana_offline/partitioning.hh`
- `tools/vamana_offline/partitioning.cc`
- `tools/vamana_offline/shard_writer.hh`
- `tools/vamana_offline/shard_writer.cc`
- `src/vamana/storage_layout_resolver.hh`
- `src/service/index_metadata.cc`

学习重点：

- 理解 balanced、BFS、METIS 分区策略的代码边界。
- 分析 shard writer 如何把图、向量、hot graph、RaBitQ、anchor 等数据写成 memory node 文件。
- 跟踪 online load 如何读取 metadata 并配置 VamanaNode 静态布局。

课时产出：

- 一张 offline shard 文件和 metadata 的字段关系图。
- 一份“metadata 和运行时静态布局不一致时会怎样”的风险分析。

### 第 26 课：实验工具与 benchmark 主路径

代码入口：

- `tools/dvstor_breakdown_benchmark.cc`
- `tools/breakdown_benchmark/args.hh`
- `tools/breakdown_benchmark/workload.hh`
- `tools/breakdown_benchmark/workload.cc`
- `tools/run_recall_test.sh`

学习重点：

- 理解 benchmark 如何构造 query、insert、mixed workload。
- 分析 warmup、measure、client threads、read ratio 对结果的影响。
- 判断工具是否足以支持后续性能优化验证。

课时产出：

- 一张 benchmark 参数到实际请求流的映射表。
- 一份现有 benchmark 覆盖不足的场景列表。

### 第 27 课：性能统计与 breakdown 报告

代码入口：

- `src/common/statistics.hh`
- `src/service/breakdown/sample.hh`
- `src/service/breakdown/aggregate.hh`
- `src/service/breakdown/json.hh`
- `src/service/breakdown/text.hh`
- `src/service/compute_service.cc`

学习重点：

- 理解 thread statistics 和 per-request breakdown sample 的粒度差异。
- 梳理 CPU、GPU、RDMA、transfer、queue wait、service time 的统计来源。
- 分析统计代码本身是否引入明显开销或偏差。

课时产出：

- 一张 breakdown subcategory 到源码埋点位置的索引。
- 一份优化前后必须观测的核心指标列表。

### 第 28 课：测试现状、可测性与安全重构护栏

代码入口：

- `CMakeLists.txt`
- `src/rdma/vector_batch_planner.hh`
- `tools/vamana_offline/recall_check.hh`
- `tools/vamana_offline/recall_check.cc`

学习重点：

- 基于代码确认当前仓库没有顶层 `test/` 目录时的测试缺口。
- 识别哪些逻辑是纯函数或近似纯函数，适合先补单元测试，例如 vector batch planner、partitioning、metadata parser。
- 设计 recall、latency、throughput、data consistency 的回归护栏。

课时产出：

- 一份测试金字塔规划。
- 一份“重构前必须补的最小测试集”。

### 第 29 课：性能优化候选点系统梳理

代码入口：

- `src/vamana/vamana_search.ipp`
- `src/rdma/vector_batch_planner.hh`
- `src/shared_context.hh`
- `src/gpu/gpu_buffer_manager.hh`
- `src/vamana/rabitq_cache.hh`
- `src/memory_node/storage_owner_runtime.cc`

学习重点：

- 从查询路径拆分 CPU selection/filter、RDMA neighbor/vector read、GPU distance、D2H、beam update。
- 从插入路径拆分 candidate search、RobustPrune、node write、reverse update、peer maintenance。
- 对每个优化候选点建立假设、指标、实验方法和失败判据。

课时产出：

- 一份性能优化 backlog。
- 每个 backlog 项包含：假设、涉及代码、预期收益、风险、验证指标。

### 第 30 课：大规模重构路线图设计

代码入口：

- `src/service/compute_service.hh`
- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`
- `src/memory_node/memory_node.hh`
- `src/common/configuration.hh`

学习重点：

- 识别超大类、模板/ipp 耦合、静态全局布局状态、配置膨胀、RDMA/GPU/算法交织等结构性问题。
- 设计渐进式重构路线：先加测试和观测，再抽边界，再替换实现。
- 明确哪些重构会改变性能、索引格式、网络协议或实验可比性。

课时产出：

- 一份三阶段重构路线图：
  - 阶段 A：可测性和边界固化。
  - 阶段 B：低风险模块拆分和配置收敛。
  - 阶段 C：高收益性能路径重写。
- 一份“不应立即重构”的高风险区域清单。

## 推荐阅读顺序

源码阅读不建议完全按目录顺序。推荐顺序如下：

1. `CMakeLists.txt`
2. `src/main.cc`、`src/memory_node_main.cc`
3. `src/common/configuration.hh`
4. `src/service/compute_service.hh`
5. `src/vamana/vamana.hh`
6. `src/vamana/vamana_node.hh`
7. `src/vamana/vamana_search.ipp`
8. `src/rdma/vamana_rdma_reads.hh`
9. `src/gpu/gpu_buffer_manager.hh`
10. `src/gpu/gpu_kernel_launcher.cu`
11. `src/vamana/vamana_insert.ipp`
12. `src/memory_node/memory_node.hh`
13. `src/service/compute_service/storage_owner_insert.ipp`
14. `src/memory_node/storage_owner_runtime.cc`
15. `src/memory_node/peer_rdma.cc`
16. `src/memory_node/peer_rpc.cc`
17. `tools/vamana_offline/graph.cc`
18. `tools/vamana_offline/shard_writer.cc`
19. `src/service/breakdown/`
20. `tools/breakdown_benchmark/`

## 后续生成课程正文时的约束

后续如果继续生成每课内容，建议遵守以下约束：

- 每课必须从代码证据出发，不能引用 README 或实验文档作为结论来源。
- 每课必须包含“调用链”“关键数据结构”“性能影响”“设计异味”“可验证问题”五类内容。
- 对所有涉及性能的判断，必须给出可观测指标和实验验证方式。
- 对所有涉及重构的建议，必须先说明行为不变量和回归测试。
- 不把论文术语或注释中的说法直接当成事实，必须回到实现验证。
