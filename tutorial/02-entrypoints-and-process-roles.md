# 第 02 课：运行入口与进程角色

## 本课目标

本课从进程入口出发，讲清楚项目运行时至少有两类角色：compute node 和 memory node。你需要掌握 `src/main.cc` 如何根据配置选择角色，`src/memory_node_main.cc` 如何启动纯 memory node，以及 CLI 配置如何一路进入服务、RDMA、GPU、storage-owner 和 routing。

## 代码证据

必须阅读：

- `src/main.cc`
- `src/memory_node_main.cc`
- `src/common/configuration.hh`
- `rdma-library/library/configuration.hh`
- `src/service/compute_service.hh`
- `src/memory_node/memory_node.hh`

## 两个入口文件

`src/main.cc` 是完整在线服务入口。它做三件事：

1. 构造 `configuration::IndexConfiguration config{argc, argv}`。
2. 如果 `config.is_server` 为 true，构造 `MemoryNode memory_node{config}`。
3. 否则按 `config.ip_distance` 构造 `ComputeService<IPDistance>` 或 `ComputeService<L2Distance>`，然后等待 SIGINT 或 SIGTERM。

简化后的角色选择是：

```text
main
  IndexConfiguration
  if config.is_server
    MemoryNode
  else
    ComputeService<Distance>
    wait_for_shutdown_signal
```

`src/memory_node_main.cc` 是 storage-node-only 可执行程序入口。它同样构造 `IndexConfiguration`，但只启动 `MemoryNode`。这保证在没有 CUDA 的机器上也能运行 memory node。

## 配置层级

`IndexConfiguration` 继承自 `rdma-library/library/configuration.hh` 中的 `Configuration`。你可以把配置分成两层：

底层 RDMA/集群配置：

- server/client 数量
- 端口和节点地址
- send/receive queue depth
- 是否是 server
- RDMA 设备相关配置

索引和服务配置：

- `dim`
- `R`
- `beam_width`
- `beam_width_construction`
- `alpha`
- `k`
- `num_threads`
- `num_coroutines`
- `vector_data_type`
- `gpu_device`
- `gpudirect_rdma`
- `expansion_batch`
- `credit_aware_expansion`
- `rdma_qp_pool_size`
- `rdma_read_batch_mode`
- `use_rabitq`
- `insert_execution`
- `storage_owner_*`
- `routing`

## 参数如何改变进程行为

重要配置项和影响如下：

| 配置项 | 影响 |
| --- | --- |
| `is_server` | 决定 `main.cc` 构造 `MemoryNode` 还是 `ComputeService` |
| `ip_distance` | 决定模板参数是 `IPDistance` 还是 `L2Distance` |
| `load_index` | compute node 和 memory node 启动时尝试读取 metadata 和 shard |
| `server_index_file` | memory node 启动时直接加载本地 shard 文件 |
| `use_rabitq` | compute side 启用 RaBitQ gate，并要求索引 metadata 匹配 |
| `gpudirect_rdma` | GPU buffer 是否尝试注册成 RDMA destination |
| `insert_execution` | 决定插入走 compute-side queue 还是 storage-owner RPC |
| `storage_owner_update_mode` | 决定 storage-owner 是否使用 local-stitch 和 anchor hints |
| `routing` | 决定是否启动 compute node 之间的 routing RPC |
| `rdma_read_batch_mode` | 决定向量批量 RDMA 读取是否使用 adaptive planner |

## compute node 启动时序

从 `ComputeService` 构造函数可以恢复大致时序：

```text
ComputeService(config)
  init_remote_tokens
  cm_.connect
  可选 pin main thread
  initiator 向 memory nodes 发送 configuration::Parameters
  receive_remote_access_tokens
  根据 metadata 配置 VamanaNode 静态布局
  gpu::gpu_init
  构造 vamana::Vamana<Distance>
  配置 expansion_batch、credit-aware、RaBitQ、query batch
  如果 use_rabitq 且 load_index，加载 rabitq::Cache
  如果 storage-owner local_stitch，加载 anchor index
  构造 WorkerPool
  为每个 ComputeThread 初始化 GPU buffer
  cm_.synchronize
  wait_for_load_or_store
  synchronize_clients_after_startup
  初始化 routing state
  start_workers
  start_rpc
  如果 storage-owner insert，start_storage_insert_runtime
  refresh_routing_state
```

这个时序说明一个重要事实：`VamanaNode` 的静态布局必须在大量运行时对象创建之前确定。metadata 校验失败会直接 `lib_assert` 或 `lib_failure`。

## memory node 启动时序

从 `MemoryNode::MemoryNode` 可以恢复大致时序：

```text
MemoryNode(config)
  context_ 和 connection manager
  cm_.connect_to_clients
  接收 initiator 发来的 configuration::Parameters
  读取 metadata，配置 VamanaNode 静态布局
  allocate_memory
  初始化 free pointer 为 16
  如果 server_index_file 非空，load_index_file
  如果需要 owner idmap，load_owner_idmap
  如果 local_stitch，加载 storage-owner anchor index
  注册 index region
  向所有 compute nodes 发送 MemoryRegionToken
  为每个 compute thread 连接 DetachedQP
  cm_.synchronize
  handle_command 处理启动命令
  如果 storage-owner insert
    setup_storage_peers
    setup_insert_runtime
    start_peer_reverse_update_runtime
    start_storage_owner_maintenance_runtime
    start_storage_owner_insert_workers
    service_storage_runtime
  否则循环 handle_command
```

memory node 构造函数非常长，本质上把启动、连接、metadata 校验、内存注册、命令处理和服务循环都放在一个构造函数中。这是后续重构时需要重点关注的结构性问题。

## configuration::Parameters 的作用

compute initiator 会向 memory node 发送：

```cpp
struct Parameters {
  u32 num_threads{};
  bool reserved{};
  bool routing{};
  u32 qp_pool_size{1};
};
```

memory node 接收后设置：

- `num_compute_threads_`
- `qp_pool_size_`

然后为 compute threads 建立足够数量的 QP。这个协议很小，但影响很大。它把 compute-side worker 数量传给 memory node，使 memory node 能按 compute thread 数量建立 RDMA 通道。

## shutdown 模型

`main.cc` 中 compute node 会阻塞等待 SIGINT 或 SIGTERM。收到信号后离开作用域，触发 `ComputeService` 析构：

```text
~ComputeService
  stop_storage_insert_runtime
  stop_rpc
  stop_workers
  shutdown_remote_if_requested
  destroy GPU buffers
  gpu_shutdown
```

如果构造 `ComputeService` 时 `shutdown_remote_on_stop=true`，并且当前是 initiator，且不是 storage-owner insert 模式，则会向 memory node 发送 `SHUTDOWN` 命令。

## 性能影响

启动阶段配置会影响后续性能上限：

- `num_threads` 决定 compute worker 数量。
- `num_coroutines` 决定每个 worker 可重叠的 RDMA/GPU coroutine 数。
- `rdma_qp_pool_size` 决定每个 memory node 对每个 `SharedContext` 的 QP 数。
- `expansion_batch` 放大每轮 beam expansion 的并行度，也放大 GPU candidate batch。
- `query_batch_size` 只有在非 RaBitQ、非 credit-aware 时才实际返回大于 1。
- `storage_owner_batch_max` 和 `storage_owner_rpc_depth` 直接决定 storage-owner insert 的微批和并发 RPC 上限。

如果后续做性能优化，首先要确认这些配置是否真的在生效。很多性能结论都可能只是参数导致。

## 设计异味

1. `ComputeService` 和 `MemoryNode` 的构造函数承担过多逻辑，不只是初始化。
2. 配置项集中在 `IndexConfiguration` 一个类中，算法、网络、GPU、实验、storage-owner 参数混在一起。
3. metadata 校验、运行时静态布局、GPU 初始化、worker 启动都在同一条构造路径里，难以单元测试。
4. `configuration::Parameters` 是轻量 wire protocol，但没有版本字段。
5. `shutdown_remote_if_requested` 对 storage-owner 模式有特殊分支，说明 shutdown 语义和插入执行模式耦合。

## 可验证问题

- `main.cc` 中什么时候会构造 `MemoryNode`？
- 为什么 `dvstor_memory_node` 不需要等待 SIGINT？
- `config.ip_distance` 如何影响模板实例化？
- memory node 如何知道 compute node 有多少线程？
- `load_index` 启动时 metadata 校验发生在 compute side 还是 memory side？

## 学习任务

1. 从 `src/main.cc` 开始，画出 compute node 启动调用链。
2. 从 `src/memory_node/memory_node.cc` 构造函数开始，画出 memory node 启动调用链。
3. 在 `src/common/configuration.hh` 中列出所有影响查询路径的参数。
4. 在 `src/common/configuration.hh` 中列出所有影响 storage-owner 插入路径的参数。
5. 思考：如果要把启动流程拆成可测试模块，应优先拆哪三个阶段？

