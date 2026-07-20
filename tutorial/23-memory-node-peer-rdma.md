# 第 23 课：存储节点主体与 peer RDMA

## 23.1 本课目标与涉及文件

第 22 课我们站在"读方"的视角讲解了计算节点用 GPUNetIO 直接读远端存储节点注册内存的工作流。本课把视角换到"被读方"，回答下面这一组问题：

- 一个 GPU 中心化的存储节点进程**到底加载什么、注册什么、向计算节点暴露什么**？
- `.dat` / `.idmap` / `.pq32.codes` 这三件磁盘产物在内存里是怎么拼起来的？PQ32 码流被安插到 metadata 指定的哪个偏移？
- 计算节点怎么拿到能 RDMA 读到的 token？为什么日志里会出现 "receive access tokens"？
- 所谓 "storage control block"（4 KiB 控制页）在缓冲区里具体长什么样？offset 1024 的 canonical route 快照怎么发布？reclaim ACK 槽是哪一段？
- 启动时做了哪些一致性校验，schema/分片数/R/dtype/PQ checksum 分别在哪儿检查？
- 多个存储节点之间为什么还要拉一条独立的 RDMA 网络（peer RDMA）？stage2 跨分片候选读取、反向边 ACK 在这一层是如何编排骨架的？QP/token/RDMA read 信用如何调度？
- 存储节点之间怎么握手？`storage_startup::Request` 那一个 4 字节魔数起到什么作用？

本课涉及的文件：

- `src/memory_node/memory_node.hh`（约 719 行）：`MemoryNode` 类的完整声明，包含生命周期、peer RDMA 传输、peer RPC、storage-owner 维护、storage-owner 插入运行时、storage-owner 索引操作六大段接口，以及全部成员变量。
- `src/memory_node/memory_node.cc`（约 755 行）：构造函数全流程（从加载 metadata、注册 MR、发 token，到 `service_storage_runtime` 主循环）、`load_index_file` 把 PQ 码流灌入指定偏移、`initialize_storage_owner_route_table`/`publish_storage_owner_route_table` 把自适应路由快照写到控制页 offset 1024，以及若干本地图访问辅助。
- `src/memory_node/peer_rdma.cc`（约 455 行）：存储节点之间的 peer RDMA 传输——`setup_storage_peers` 全连接 + token 交换、`remote_read_bytes`/`remote_write_bytes`/`remote_compare_and_swap` 三个同步原语、`post_peer_read_async` 协程异步读、信用（credit）控制、send CQ 轮询/完成回调、`try_lock_remote_header` 跨分片节点头加锁。
- `src/memory_node/startup_protocol.hh`：启动握手协议（`storage_startup::kMagic = 0x44565354`，即 ASCII "DVST"）。
- `src/memory_node/storage_owner_state.hh`：storage owner 运行时的核心值类型——`BeamEntry`、`NodeSnapshot`、`StorageOwnerCoroutineScratch`、`InsertRuntimeState`、`PeerRpcRuntimeState`、`PeerPendingSend`、`StorageOwnerThread`、`FreshnessEntry` 等。
- `src/memory_node/storage_owner_cpu_plan.hh`：CPU 侧执行计划 `StorageOwnerCpuPlan` 与 `derive_storage_owner_cpu_plan` 的分配算法。
- `src/memory_node/storage_reclaim.hh`：存储侧 reclaim 接口 `StorageReclaimQueue`，与第 16 课计算侧 ACK 直接对接。

与其他课的关系：第 5 课的 `Context`/`ServerConnectionManager`/`DetachedQP`/`MemoryRegion`/`MemoryRegionToken` 是本课的传输层底座；第 7 课 schema-15 索引格式（`CodeHeader`、`StorageControlBlock`、`StorageRoutePublication`）是本课加载与校验的对象；第 8 课的 `service::index_metadata::Metadata` 与 `service::storage_owner` 协议是本课入口契约；第 22 课 GPUNetIO 是"读方"，本课是"被读方"；第 24 课 peer RPC、第 25 课索引访问/图修改、第 26 课维护/wire protocol 都建立在 `MemoryNode` 这个外壳之上，本课给出这个外壳本身。第 16 课介绍了计算侧 reclaim ACK 的写法，本课给出存储侧消费 ACK 的 `StorageReclaimQueue` 与 `minimum_compute_reclaim_ack`。

## 23.2 MemoryNode 的整体结构

`MemoryNode` 是一个巨大的 PImpl 风格外壳。从 `memory_node.hh:52` 的类注释就能读出它的职责定位：

```cpp
/**
 * Owns one static vector/compact-graph shard, its PQ navigation-code stream,
 * and the mutable storage-owner region used by online updates. Compute nodes
 * access these regions directly; storage workers only execute update and
 * maintenance protocols.
 */
class MemoryNode {
```

这一段注释非常关键——它定义了"GPU 中心化存算分离"在存储侧的实现原则：

1. **每个 `MemoryNode` 持有一个静态分片**（`index_buffer_` 里的 `.dat` 数据）；
2. **同时持有一条 PQ 导航码流**（`gpu_navigation_code_bytes_` 指向的码流，被灌入控制页之后）；
3. **同时持有一片可变 storage-owner 区域**（`gpu_dynamic_node_base_` 之后的动态节点空间），供在线 upsert 使用；
4. **计算节点直接 RDMA 读这些区域**，存储侧 worker 不参与查询路径，只跑更新与维护协议。

也就是说，存储节点的查询路径是"零 CPU 参与"的——CPU 只负责装载、注册、维护。这一点决定了 `MemoryNode` 内部几乎所有的数据结构布局。

`MemoryNode` 没有公开接口（除了构造函数），全部 `private`。它的私有成员按职责可以粗分为六大段：

| 段落 | 关键成员 | 出现位置 |
|------|----------|----------|
| 生命周期与配置 | `context_`、`cm_`、`core_assignment_`、`num_clients_`、`storage_id_`、`num_storage_nodes_`、`mn_memory_bytes_` | `memory_node.hh:561-577` |
| 主缓冲区与 MR | `index_buffer_`、`index_region_` | `memory_node.hh:579-580` |
| peer RDMA 传输 | `peer_config_`、`peer_context_`、`peer_qps_`、`peer_remote_tokens_`、`peer_index_region_`、`peer_scratch_buffer_`、`peer_scratch_region_`、各 `peer_rdma_read_*_outstanding_`、`peer_qp_send_mutexes_` | `memory_node.hh:581-605` |
| peer RPC 运行时 | `peer_rpc_runtime_`、`peer_async_responses_`、`peer_request_deduplicator_`、`peer_rpc_responses_`、`peer_reverse_workers_`、`peer_stitch_search_workers_` 等 | `memory_node.hh:589-647` |
| storage-owner 维护 | `storage_owner_maintenance_workers_`、`storage_owner_repair_tasks_`、`storage_owner_reverse_outbox_`、`storage_owner_reclaim_queue_` 等 | `memory_node.hh:649-694` |
| storage-owner 插入运行时与索引状态 | `insert_runtime_`、`storage_insert_tasks_`、`storage_owner_threads_`、`storage_owner_route_table_`、`base_idmap_`、`dynamic_freshness_shards_` | `memory_node.hh:695-715` |

注意 `memory_node.hh:717-718` 还声明了两个 thread_local 指针：

```cpp
inline static thread_local StorageOwnerThread* current_storage_owner_thread_{nullptr};
inline static thread_local bool current_peer_rpc_progress_thread_{false};
```

这是因为 `MemoryNode` 内部跑着多套工作线程（插入 worker、维护 worker、peer 反向 worker、peer 搜索 worker、peer RPC progress 线程），它们都需要在不知道 `MemoryNode` 全貌的前提下"找到自己当前的执行上下文"。`current_storage_owner_thread_` 让任意一段代码拿到"当前线程的 scratch buffer / coroutine 状态"；`current_peer_rpc_progress_thread_` 用来识别"我现在的身份是 peer RPC progress 线程"，从而让 `wait_peer_sync_completion` 选择一条无锁的等待路径（见 23.4）。

类的私有嵌套类型集中在 `memory_node.hh:56-122`，它们刻画了 peer 反向更新与维护任务的全部数据载荷：

```cpp
struct PeerReverseUpdateTask {
  u32 source_shard{};
  service::storage_owner::PeerRpcHeader header{};
  vec<service::storage_owner::ReverseUpdateOp> ops;
  std::chrono::steady_clock::time_point received_at{};
};
```

`PeerReverseUpdateTask` 是从 peer 收到的"反向边更新请求"在存储侧排队时的载荷——`source_shard` 标识来源分片，`header` 是协议头（带 request_id、item_count 等），`ops` 是一批 `ReverseUpdateOp`，`received_at` 用于慢路径统计。这与第 24 课 peer RPC 的接收侧直接对应。

```cpp
struct StorageOwnerMaintenanceTask {
  StorageOwnerMaintenanceKind kind{StorageOwnerMaintenanceKind::stitch_insert};
  node_t id{};
  u32 generation{};
  u64 maintenance_sequence{};
  RemotePtr target;
  bool stitch_prepared{};
  vec<RemotePtr> stage1_candidates;
  vec<RemotePtr> stitch_base_neighbors;
  vec<RemotePtr> stitch_neighbors;
  bool cleanup_repair_only{};
  vec<RemotePtr> cleanup_neighbors;
  std::chrono::steady_clock::time_point queued_at{};
};
```

`StorageOwnerMaintenanceTask` 比较复杂，它是第 26 课维护协议的工作单元。这里只需要记住它分两类（`stitch_insert` / `cleanup_deleted_node`），携带了 stage1 已经收集好的候选邻居、当时被改写前的"基线邻居"（用于把 stage2 期间到达的反向边 rebase 进来而不是丢掉）、以及 cleanup 路径专用的 `cleanup_neighbors`。`maintenance_sequence` 与 reclaim 流程挂钩（见 23.6）。

类常量 `memory_node.hh:170-179` 也值得记一下：

```cpp
static constexpr u32 kPeerSyncWrOwner = std::numeric_limits<u32>::max();
static constexpr u32 kPeerAsyncWrOwner = std::numeric_limits<u32>::max() - 1;
static constexpr u32 kPeerSafeRdAtomic = 8;
static constexpr u32 kPeerRpcFlagNoResponse = 1u;
```

`kPeerSafeRdAtomic = 8` 是一条 QP 上同时挂起的 RDMA read 的安全上限——它直接决定 `peer_rdma_read_credit_limit_per_qp()`（见 23.4）。`kPeerSyncWrOwner` 与 `kPeerAsyncWrOwner` 是把 64 位 `wr_id` 拆成 `(owner, id)` 时的特殊 owner 值：同步路径用 `kPeerSyncWrOwner`、协程异步路径用 `kPeerAsyncWrOwner`，剩下的 owner 值就是 storage-owner 线程的 `thread_id`，这样可以复用同一个 send CQ 完成回调（见 23.4 的 `handle_peer_send_completion`）。

## 23.3 构造函数全流程：加载、注册、暴露

`memory_node.cc:12-219` 的构造函数是本课的核心。它是一条线性流水线，把一个"冷启动的存储进程"逐步变成"被计算节点 RDMA 读、被 peer 存储 RDMA 读、能处理 upsert"的运行态。我们按段拆开看。

### 23.3.1 初始化基本字段与连接计算节点

```cpp
MemoryNode::MemoryNode(Configuration& config)
    : context_(config), cm_(context_, config), num_clients_(config.num_clients),
      storage_id_(config.storage_id),
      num_storage_nodes_(config.storage_peers.empty() ? config.num_server_nodes()
                                                      : static_cast<u32>(config.storage_peers.size())),
      storage_owner_peer_rdma_tokens_(std::max<u32>(1, config.storage_owner_peer_rdma_tokens)),
      index_region_(context_),
      peer_rdma_read_outstanding_(num_storage_nodes_),
      mn_memory_bytes_(static_cast<u64>(config.mn_memory_gb) * 1073741824ul) {
  for (auto& credit : peer_rdma_read_outstanding_) {
    credit.store(0, std::memory_order_relaxed);
  }
  cm_.connect_to_clients();
```

- `context_(config)` 创建 RDMA `Context`（第 5 课），打开 IB device 与 PD。
- `cm_(context_, config)` 构造 `ServerConnectionManager`，构造函数引用 `memory_node.hh:562` 的字段声明。
- `num_storage_nodes_` 优先用 `config.storage_peers.size()`，否则回退到 `config.num_server_nodes()`——前者用于 storage-owner 模式下显式列出所有存储节点的 endpoint（peer RDMA 全连接要用，见 23.4）。
- `storage_owner_peer_rdma_tokens_` 至少为 1，是配置项 `storage_owner_peer_rdma_tokens` 决定的"全局 peer RDMA read 信用上限"，会被 `peer_rdma_read_credit_limit()` 钳制（见 23.4）。
- `mn_memory_bytes_ = config.mn_memory_gb * 1 GiB`。这就是后面 `index_buffer_` 的大页大小，也是 peer RDMA 校验"读不越界"的依据（`remote_offset + bytes <= mn_memory_bytes_`）。
- `peer_rdma_read_outstanding_` 是一个 `vec<std::atomic<u32>>`，每个 peer 一个原子计数器，记录"当前发给该 peer 的 in-flight RDMA read 数量"。构造函数把它们清零。
- `cm_.connect_to_clients()` 完成 ServerConnectionManager 的初始握手——这是第 5 课讲过的"启动时建立一条初始 QP（`initiator_qp`）+ 每个 client 一条 `client_qps`"的逻辑。

### 23.3.2 接收计算节点启动参数

```cpp
  // receive runtimes parameters from initiator
  configuration::Parameters p{};
  LocalMemoryRegion region{context_, &p, sizeof(configuration::Parameters)};

  cm_.initiator_qp->post_receive(region);
  context_.receive();

  num_compute_threads_ = p.num_threads;
  const u32 gpu_rdma_qps = p.gpu_rdma_qps;
  const filepath_t index_prefix = config.resolved_index_prefix();
  index_prefix_ = index_prefix;
  VectorDType startup_dtype = config.resolved_vector_dtype();
  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
```

注意，存储节点并不直接从命令行拿 `num_threads`、`gpu_rdma_qps`——这两个值是由"initiator"（启动协调者，通常就是 0 号计算节点）通过 `initiator_qp` 现场发过来的。`configuration::Parameters` 是一个 POJO，用一个临时 `LocalMemoryRegion` 包起来，`post_receive` + `context_.receive()` 是阻塞地把它收下来。这一步确立了"接下来要为每个计算节点建多少条 DetachedQP"。

紧接着读 `index_prefix` 与 `meta_file`——前者来自 `config.resolved_index_prefix()`，后者就是 `<prefix>.meta.json`。下面这一大段就是启动校验。

### 23.3.3 启动校验：schema / 分片数 / R / dtype / PQ checksum

`memory_node.cc:45-145` 这一段是存储节点启动时最严格的一致性检查，几乎每两行就是一个 `lib_assert`。下面把它分类列出：

**索引格式兼容性（schema-15 + compact OPQ/PQ）：**

```cpp
gpu_stream_layout_ = metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
  metadata.node_layout == "plain" &&
  metadata.storage_format == "vamana_compact_v1" &&
  compatible_quantizer && compatible_navigation;
lib_assert(gpu_stream_layout_,
           "storage node requires a schema-15 compact OPQ/PQ index");
```

只有 schema-15 + plain node layout + vamana_compact_v1 + opq_pq / opq_pq16 量化的索引才能被本存储节点加载。这是第 7 课 schema-15 索引格式与第 22 课 GPUNetIO 流式读取约定的硬约束。`gpu_stream_layout_` 这个 bool 决定后续能否把 PQ 码流拼到控制页之后。

**分片拓扑一致性：**

```cpp
lib_assert(metadata.num_memory_nodes == num_storage_nodes_, "index metadata storage-node count mismatch");
...
lib_assert(storage_id_ < num_storage_nodes_, "invalid GPU storage shard id");
lib_assert(metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
           "GPU storage metadata has invalid static shard counts");
lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_,
           "GPU storage metadata has invalid dynamic shard offsets");
lib_assert(metadata.storage_control_remote_offsets.size() == num_storage_nodes_ &&
             metadata.dynamic_node_base_offsets.size() == num_storage_nodes_,
           "GPU storage metadata has invalid control/dynamic-node offsets");
```

metadata 里记录的"分片数"必须等于本进程认为的 `num_storage_nodes_`；而且 metadata 里四个 per-shard 数组（静态节点数、动态基址、控制页远端偏移、动态节点基址）长度都得等于分片数。这些数组接下来会被本分片读取自己的对应元素：

```cpp
gpu_static_node_count_ = metadata.hot_graph_entry_counts[storage_id_];
gpu_static_dynamic_base_ = metadata.hot_graph_dynamic_base_offsets[storage_id_];
gpu_storage_control_offset_ = metadata.storage_control_remote_offsets[storage_id_];
gpu_dynamic_node_base_ = metadata.dynamic_node_base_offsets[storage_id_];
gpu_navigation_code_bytes_ = metadata.navigation_code_bytes;
```

这五个值定义了本存储节点在整个大缓冲区里的内存布局（见 23.5 的内存布局图）。

**维度与 PQ 模型校验：**

```cpp
lib_assert(gpu_navigation_code_bytes_ > 0 &&
             gpu_navigation_code_bytes_ <= gpu_search::format::kStorageRouteMaxCodeBytes,
           "navigation PQ width exceeds the fixed storage route publication");
```

`kStorageRouteMaxCodeBytes = 32`（`index_format.hh:37`），因为 `StorageRouteSlot::navigation_code` 是一个固定 32 字节的数组（`index_format.hh:100-105`）。如果单条 PQ 码大于 32 字节，路由发布页就装不下，所以这是硬约束。

```cpp
lib_assert(gpu_search::pq::read_model(
             index_path::navigation_model_file(index_prefix,
                                               metadata.pq_subquantizers),
             gpu_navigation_model_, &metadata_error),
           metadata_error);
lib_assert(gpu_navigation_model_.checksum() == metadata.navigation_model_checksum &&
             gpu_navigation_model_.code_bytes() == metadata.navigation_code_bytes &&
             gpu_navigation_model_.dim == metadata.dim,
           "storage-node PQ model does not match index metadata");
```

加载 PQ 模型（`.pq32.model`），并验证三件事：模型 checksum、码字节数、维度，都得跟 metadata 对上。这是 "PQ checksum" 校验的落点。`gpu_navigation_model_checksum_` 也会被用来跟接下来的 `.pq32.codes` 头部比对（见 23.3.5）。

**Vamana 静态布局一致性：**

```cpp
VamanaNode::disable_hot_graph();
VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
...
lib_assert(metadata.vector_component_size == VamanaNode::vector_component_size(),
           "index metadata vector component size mismatch on storage node");
lib_assert(metadata.vector_bytes == VamanaNode::vector_bytes(),
           "index metadata vector byte size mismatch on storage node");
lib_assert(metadata.node_size == VamanaNode::total_size(), "index metadata node size mismatch on storage node");
lib_assert(metadata.graph_hot_bytes == VamanaNode::graph_hot_bytes() &&
           metadata.vector_offset == VamanaNode::offset_vector(),
           "index metadata storage offsets mismatch on storage node");
```

`VamanaNode::disable_hot_graph()` 与 `init_static_storage(dim, R, dtype)` 配置 Vamana 节点的全局静态布局参数（第 6 课）。注意这里"先 disable_hot_graph 后 init_static_storage"——存储节点不需要 hot graph 视图（它是给计算节点用的），但它确实需要 `total_size()` / `vector_bytes()` / `graph_hot_bytes()` 这些常量来解释磁盘上的字节。随后逐字段比对 metadata 与 `VamanaNode` 计算出的偏移/大小，任何不一致都说明索引是用不同版本的 builder 跑出来的，必须拒绝。

**动态节点布局一致性：**

```cpp
lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_ &&
           metadata.dynamic_node_base_offsets.size() == num_storage_nodes_ &&
           metadata.hot_graph_dynamic_record_bytes >=
             metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
           metadata.hot_graph_dynamic_hot_offset >= VamanaNode::total_size() &&
           metadata.dynamic_navigation_code_offset >=
             metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
           metadata.hot_graph_dynamic_record_bytes >=
             metadata.dynamic_navigation_code_offset + metadata.navigation_code_bytes,
           "index dynamic hot graph metadata mismatch on storage node");
VamanaNode::configure_hot_graph(metadata.hot_graph_offsets,
                                metadata.hot_graph_entry_counts,
                                metadata.hot_graph_entry_size,
                                metadata.hot_graph_shard_bits,
                                metadata.dynamic_node_base_offsets,
                                metadata.hot_graph_dynamic_record_bytes,
                                metadata.hot_graph_dynamic_hot_offset,
                                metadata.dynamic_navigation_code_offset,
                                metadata.navigation_code_bytes);
```

这一段确认"动态节点记录里 hot graph 段与 PQ 码段都不溢出 record 边界，且 hot graph 段在向量之后、PQ 码在 hot graph 之后"。然后 `VamanaNode::configure_hot_graph(...)` 把这些偏移灌进 Vamana 的全局静态状态，让后续 `VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET` / `HOT_GRAPH_DYNAMIC_CODE_OFFSET` / `HOT_GRAPH_DYNAMIC_CODE_BYTES` 这些常量可用——它们在 23.3.5 的 `StorageControlBlock` 初始化里直接被引用。

校验完成后：

```cpp
owner_idmap_required_ = metadata.idmap_format == "owner_sharded_v1";
```

这决定后面要不要额外加载一份 owner-sharded idmap（`base_idmap_`），用于 storage-owner 模式下的 `prepare_mutation`/`observe_storage_owner_route` 校验"这个 id 在我的分片上吗"。

### 23.3.4 分配大页缓冲区

```cpp
  allocate_memory();
  // free-ptr is initialized to 16 (points to first free address in the buffer)
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;
```

`allocate_memory` 定义在 `memory_node.cc:279-290`：

```cpp
void MemoryNode::allocate_memory() {
  const auto t_allocate = timing_.create_enroll("allocate_index_buffer");
  std::cerr << "allocation size: " << mn_memory_bytes_ << std::endl;
  t_allocate->start();
  const size_t available_memory = index_buffer_.get_memory_size();
  lib_assert(mn_memory_bytes_ <= available_memory, "allocation failed");
  index_buffer_.allocate(mn_memory_bytes_);
  index_buffer_.touch_memory();
  t_allocate->stop();
}
```

`HugePage<byte_t> index_buffer_` 是 2 MiB 大页的封装（第 5 课）。`touch_memory()` 是逐页写一遍，强制内核把大页真分配出来，避免 RDMA MR 注册后再触发缺页。`get_memory_size()` 是系统允许的最大大页量，`mn_memory_bytes_` 必须不超过它。

注意构造函数体里那一行 `*reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;`——这是临时把"free pointer"（缓冲区头 8 字节，永远指向下一个可分配地址）初始化为 16，也就是 `kNodeBaseOffset`。这只是占位，`load_index_file` 在最后会用真实的 `gpu_dynamic_node_base_` 把它覆盖掉（见 23.3.5 末尾）。

### 23.3.5 `load_index_file`：把 .dat 与 PQ 码流拼到指定偏移

`memory_node.cc:307-430` 的 `load_index_file` 是本课第二个核心函数。它做四件事：

**1) 读 `.dat` 主索引文件：**

```cpp
std::pair<bool, str> MemoryNode::load_index_file(const str& path) {
  std::ifstream file{path, std::ios::binary};
  ...
  file.read(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), file_size);
  ...
```

把整个 `.dat` 一次性塞进 `index_buffer_` 起始处。这里没有偏移——磁盘上的 `.dat` 已经按"buffer 头 16 字节是 free pointer、16 字节之后是静态节点数组、之后是动态区"的格式布局（见第 7 课、第 29 课）。

**2) 校验磁盘上的 `.dat` 与 metadata 一致：**

```cpp
if (!gpu_stream_layout_ || gpu_static_node_count_ == 0 ||
    gpu_static_dynamic_base_ == 0) {
  return {false, "GPU storage metadata cannot materialize the PQ stream"};
}
const u64 persisted_free_pointer =
  *reinterpret_cast<const u64*>(index_buffer_.get_full_buffer());
if (persisted_free_pointer != gpu_static_dynamic_base_) {
  return {false, "GPU navigation requires a compacted static shard before startup"};
}
const u64 fixed_nodes_end = gpu_search::format::kNodeBaseOffset +
  gpu_static_node_count_ * VamanaNode::total_size();
if (fixed_nodes_end > gpu_static_dynamic_base_ ||
    gpu_static_dynamic_base_ > file_size) {
  return {false, "GPU storage shard is truncated or has inconsistent static metadata"};
}
```

关键校验是"磁盘上的 free pointer 必须等于 metadata 里的 `gpu_static_dynamic_base_`"——也就是说，磁盘上的 `.dat` 必须是经过"静态分片压缩"的产物，free pointer 已经被推进到动态区起点。如果不是（例如 builder 还没把静态区压缩完毕），存储节点就拒绝启动。

**3) 校验并加载 `.pq32.codes` 码流：**

```cpp
const u64 remote_offset = gpu_storage_control_offset_ +
  gpu_search::format::kStorageControlBytes;
const u64 payload_bytes = gpu_static_node_count_ * gpu_navigation_code_bytes_;
...
const filepath_t code_path = index_path::navigation_code_for_shard(
  path, gpu_navigation_code_bytes_);
gpu_search::format::CodeHeader header;
str error;
if (!gpu_search::format::read_code_header(code_path, header, &error) ||
    header.memory_node != storage_id_ ||
    header.node_size != VamanaNode::total_size() ||
    header.code_bytes != gpu_navigation_code_bytes_ ||
    header.model_checksum != gpu_navigation_model_checksum_ ||
    header.entry_count != gpu_static_node_count_ ||
    header.remote_offset != remote_offset ||
    header.payload_bytes != payload_bytes) {
  return {false, error.empty() ? "incompatible GPU PQ sidecar " + code_path.string()
                               : error};
}
```

码流的目标偏移 `remote_offset = gpu_storage_control_offset_ + kStorageControlBytes`——也就是说，码流紧跟在 4 KiB 控制页之后。`CodeHeader`（`index_format.hh:126-143`）里记录的 `remote_offset` / `payload_bytes` / `model_checksum` / `entry_count` 必须与运行时计算的完全一致，否则拒绝加载。这是防止"码流与 `.dat` 不是同一次构建产物"的硬约束。

接着分块（64 MiB 一块）把码流读到目标偏移，并一边读一边计算 checksum：

```cpp
std::ifstream codes{code_path, std::ios::binary};
codes.seekg(static_cast<std::streamoff>(sizeof(header)));
constexpr size_t chunk_bytes = 64ull << 20;
u64 checksum = gpu_search::format::checksum64_initial();
for (u64 offset = 0; offset < header.payload_bytes; offset += chunk_bytes) {
  const size_t bytes = static_cast<size_t>(
    std::min<u64>(chunk_bytes, header.payload_bytes - offset));
  byte_t* destination = index_buffer_.get_full_buffer() + header.remote_offset + offset;
  codes.read(reinterpret_cast<char*>(destination), static_cast<std::streamoff>(bytes));
  if (static_cast<size_t>(codes.gcount()) != bytes) {
    return {false, "short read from " + code_path.string()};
  }
  checksum = gpu_search::format::checksum64_update(checksum, destination, bytes);
}
if (checksum != header.payload_checksum) {
  return {false, "GPU PQ code sidecar payload checksum mismatch: " + code_path.string()};
}
```

64 MiB 分块是为了在巨大码流下也能稳态地推进。`checksum64_update` 是流式 64 位 checksum，最终与 `CodeHeader::payload_checksum` 比对——这就是日志里说的 "PQ checksum" 校验。

**4) 初始化 storage control block 与 route publication：**

```cpp
const u64 region_end = remote_offset + payload_bytes;
if (gpu_storage_control_offset_ !=
      gpu_search::format::align_up(gpu_static_dynamic_base_, 64) ||
    gpu_dynamic_node_base_ < region_end ||
    (gpu_dynamic_node_base_ - gpu_static_dynamic_base_) %
      VamanaNode::allocation_size() != 0 ||
    gpu_dynamic_node_base_ > index_buffer_.buffer_size) {
  return {false, "GPU storage control/dynamic-node layout is inconsistent"};
}
std::memset(index_buffer_.get_full_buffer() + gpu_storage_control_offset_, 0,
            gpu_search::format::kStorageControlBytes);
auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
  index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
*control = gpu_search::format::StorageControlBlock{
  .shard_id = storage_id_,
  .dynamic_record_bytes = static_cast<u32>(VamanaNode::allocation_size()),
  .dynamic_hot_offset = VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
  .dynamic_code_offset = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET,
  .code_bytes = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_BYTES,
  .compute_client_count = num_clients_,
  .dynamic_high_watermark = gpu_dynamic_node_base_,
};
```

控制页 4 KiB 被清零，然后写入 `StorageControlBlock`（`index_format.hh:80-98`）。注意 `StorageControlBlock` 自身只有 640 字节（`static_assert(sizeof(StorageControlBlock) == 640)`），后面到 1024 之间是预留空间；从 1024 开始放 `StorageRoutePublication`（448 字节），到 1472 结束；1472 到 4096 之间也是预留。`reclaim_ack_sequences` 是 `std::array<u64, 64>`，对应 `kMaxComputeClients = 64` 个计算节点的 reclaim ACK 序号槽（第 16 课计算侧每完成一轮 reclaim 就把自己的槽位写高，本存储侧用 `minimum_compute_reclaim_ack()` 读最小值决定哪些 retire 节点可以复用，见 23.6）。

接着初始化路由发布页：

```cpp
auto* route_publication = reinterpret_cast<
  gpu_search::format::StorageRoutePublication*>(
  index_buffer_.get_full_buffer() + gpu_storage_control_offset_ +
  gpu_search::format::kStorageRoutePublicationOffset);
*route_publication = gpu_search::format::StorageRoutePublication{
  .sequence_begin = 2,
  .shard_id = storage_id_,
  .code_bytes = gpu_navigation_code_bytes_,
  .sequence_end = 2,
};
route_publication->body_checksum =
  gpu_search::format::storage_route_body_checksum(*route_publication);
```

`sequence_begin = sequence_end = 2` 是一个"初始偶数版本"——`publish_storage_owner_route_table` 用"奇数 publish + 偶数 publish"的双段写序列（见 23.4 末尾），所以初始值必须是偶数。`body_checksum` 保证 body 在传输中不会被撕裂。

最后：

```cpp
if (num_clients_ == 0 || num_clients_ > gpu_search::format::kMaxComputeClients) {
  return {false, "compute client count exceeds the storage reclaim control capacity"};
}
*reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = gpu_dynamic_node_base_;
return {true, ""};
```

把缓冲区头的 free pointer 覆盖为 `gpu_dynamic_node_base_`——这是真实运行时的 free pointer 起点（动态节点从 `gpu_dynamic_node_base_` 开始往后分配）。`num_clients_` 不能超过 64，因为 `reclaim_ack_sequences` 只有 64 槽。

### 23.3.6 注册 MR、分发 access token、建 DetachedQP

回到构造函数 `memory_node.cc:163-194`：

```cpp
  print_status("register memory and distribute access token");
  index_region_.register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);
  MemoryRegionToken token = index_region_.createToken();

  // send access token to all compute nodes
  for (QP& qp : cm_.client_qps) {
    qp->post_send_inlined(std::addressof(token), sizeof(token), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
```

这一段就是日志里 "receive access tokens" 在存储侧的发送方。`index_region_.register_memory(..., true)` 注册整段 `index_buffer_` 为 MR，第三参数 `true` 表示开启 remote write（计算节点不仅要读，还要在 reclaim ACK 槽上写）。`createToken()` 产生一个 `MemoryRegionToken`（包含 `address` + `rkey` + 长度），然后通过每条 `client_qp` 用 `IBV_WR_SEND`（不是 RDMA write，是 send/recv）把这个 token 发给对应的计算节点。计算节点收到后就可以用这个 token 拼 `RemotePtr` 并发起 GPUNetIO RDMA read（第 22 课）。

```cpp
  // connect for each compute thread a new QP
  print_status("connect QPs of compute threads");
  vec<u_ptr<DetachedQP>> qps;
  // note: no need for QP sharing on the memory server side
  const u32 qps_per_node = gpu_rdma_qps;
  if (gpu_rdma_qps > 0) {
    print_status("reserving " + std::to_string(gpu_rdma_qps) +
                 " GPU/bootstrap QPs per compute node");
  }
  qps.reserve(num_clients_ * qps_per_node);

  for (QP& client_qp : cm_.client_qps) {
    for (u32 thread_id = 0; thread_id < qps_per_node; ++thread_id) {
      auto& qp = qps.emplace_back(std::make_unique<DetachedQP>(context_));
      qp->connect(context_, context_.get_lid(), client_qp);
    }
  }

  // notify compute nodes that we are ready
  cm_.synchronize();
```

`gpu_rdma_qps` 来自 initiator 推过来的 `p.gpu_rdma_qps`，每个计算节点要建多少条 DetachedQP（用于 GPUNetIO 流式 RDMA read，第 22 课）。注意"no need for QP sharing on the memory server side"——存储侧每条 QP 只服务一个计算节点的一个线程，不用像计算侧那样做 QP sharing，因为存储侧根本不主动发起 RDMA，它只是被读。`DetachedQP::connect` 是被动方一端的三次握手（第 5 课）。

`cm_.synchronize()` 是一个全局 barrier——所有存储节点都建完 DetachedQP 之后才放行。

### 23.3.7 启动 barrier、peer RDMA、各运行时

构造函数最后一段把所有后台运行时拉起来：

```cpp
  wait_for_start_signal();
  setup_storage_peers(config);
  setup_insert_runtime(config);
  storage_worker_config_ = std::make_unique<Configuration>(config);
  start_peer_reverse_update_runtime(config);
  start_storage_owner_maintenance_runtime(config);
  start_storage_owner_insert_workers(config);
  if (!config.disable_thread_pinning) {
    pin_main_thread(core_assignment_.get_available_core());
  }
  service_storage_runtime(config);

  storage_insert_shutdown_.store(true, std::memory_order_release);
  if (storage_insert_tasks_) storage_insert_tasks_->notify_all();
  for (auto& worker : storage_insert_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  stop_storage_owner_maintenance_runtime();
  stop_peer_reverse_update_runtime();

  print_status("memory node shutting down");
  std::cout << timing_ << std::endl;
}
```

执行顺序：

1. `wait_for_start_signal()` —— 等待计算节点发"开始"信号；
2. `setup_storage_peers(config)` —— 建立存储节点之间的 peer RDMA 全连接（23.4）；
3. `setup_insert_runtime(config)` —— 为 storage-owner 插入协议准备 buffer/MR/slot（第 25、26 课）；
4. `start_peer_reverse_update_runtime(config)` —— 启动 peer 反向边更新 RPC 工作线程（第 24 课）；
5. `start_storage_owner_maintenance_runtime(config)` —— 启动维护工作线程（第 26 课）；
6. `start_storage_owner_insert_workers(config)` —— 启动插入 worker（第 25 课）；
7. `pin_main_thread(...)` —— 把主线程钉到一个空闲核；
8. `service_storage_runtime(config)` —— 进入主循环，转发 insert 请求、回收 retire 节点。这一行会阻塞直到 shutdown。

退出时按相反顺序关停：先停插入 worker、再停维护、再停 peer 反向 RPC，最后打印 `timing_` 报告各阶段耗时。

`wait_for_start_signal` 的实现非常简单，`memory_node.cc:292-305`：

```cpp
void MemoryNode::wait_for_start_signal() {
  print_status("waiting for compute-node startup barrier");
  storage_startup::Request request{};
  LocalMemoryRegion region{context_, &request, sizeof(request)};
  cm_.initiator_qp->post_receive(region);
  context_.receive();
  const storage_startup::Response response{
    .ready = request.magic == storage_startup::kMagic,
  };
  cm_.initiator_qp->post_send_inlined(
    &response, sizeof(response), IBV_WR_SEND);
  context_.poll_send_cq_until_completion();
  lib_assert(response.ready, "invalid compute-node startup request");
}
```

`storage_startup::kMagic = 0x44565354`（ASCII "DVST"，`startup_protocol.hh:7`）。计算节点启动完毕后会发一个 `Request`，存储节点校验 magic 是否等于 `kMagic`，并回一个 `Response{ready}`。如果 magic 不对（比如把另一协议的包错投到 initiator_qp），存储节点会 `lib_assert` 失败退出。这是启动握手的全部——一个 4 字节魔数加一个 bool 回执。

## 23.4 peer RDMA：存储节点之间的传输骨架

`peer_rdma.cc` 是 `MemoryNode` 在存储节点之间建立的第二条 RDMA 网络。它和"计算节点 ↔ 存储节点"那条网络的本质区别是：

- **计算节点 ↔ 存储节点**：存储侧被动，计算侧主动发起 RDMA read。存储侧只需要注册 MR、发 token，不主动 post 任何 WR。
- **存储节点 ↔ 存储节点**：双向主动——任一存储节点都可能需要读另一存储节点的动态节点、邻居表，甚至 CAS 节点头来加锁。因此每个存储节点都既是 server 又是 client，需要全连接 + 双向 QP + 信用控制。

为什么需要这条网络？storage-owner 协议下，一个 upsert 的 stage2（最终剪枝）需要把候选集扩展到"跨分片"的邻居——也就是图里有反向边指向其他分片的节点。这时候必须去远端存储节点读它的邻居表、节点 snapshot。同时，插入完成后要把"我新加了节点 X，X 的反向边指向你那边的 Y"这件事通知 Y 所在的分片，让它把 Y 的邻居表加上 X——这就是 peer 反向更新 RPC（第 24 课），而 RPC 本身也跑在这条 peer RDMA 网络上。`peer_rdma.cc` 给出了这条网络的传输骨架；具体的 RPC 编排与语义留到第 24 课。

### 23.4.1 `setup_storage_peers`：全连接 + token 交换

`peer_rdma.cc:5-95`：

```cpp
void MemoryNode::setup_storage_peers(Configuration& config) {
  if (num_storage_nodes_ <= 1) {
    return;
  }
  lib_assert(config.storage_peers.size() == num_storage_nodes_,
             "storage_owner mode requires one storage peer endpoint per storage node");
  const auto self_endpoint = parse_endpoint(config.storage_peers[storage_id_], config.port);
```

单分片直接返回——没有 peer。多分片时要求 `config.storage_peers` 列出所有存储节点的 endpoint（与 `num_storage_nodes_` 等长）。`self_endpoint` 取自己那一项，注意它的端口可能被 `config.port` 覆盖（这是为了让一台机器上跑多个存储节点时可以共用 endpoint 字符串、靠 `--port` 区分）。

```cpp
  peer_config_ = std::make_unique<configuration::Configuration>(config);
  peer_config_->port = self_endpoint.port;
  peer_config_->is_server = true;
  peer_context_ = std::make_unique<Context>(*peer_config_);
  peer_context_->bind_to_port(self_endpoint.port);
```

peer RDMA 用一个**独立的 `Context`**（`peer_context_`），与计算节点那条网络的 `context_` 隔离。这是为了避免两条网络的 CQ 互相干扰，也方便独立 pin CPU、独立调send queue 深度。`is_server = true` 让 peer_context 把自己这一端当作 server（要 accept 其他 peer 的连接）。

```cpp
  peer_qps_per_peer_ = std::max<u32>(
    1, std::min<u32>(kMaxPeerQps, std::max<u32>(1, num_compute_threads_)));
  peer_qps_.resize(num_storage_nodes_);
  peer_qp_send_mutexes_.resize(num_storage_nodes_);
  peer_remote_tokens_.resize(num_storage_nodes_);
  peer_rdma_read_qp_outstanding_.clear();
  peer_rdma_read_qp_outstanding_.reserve(num_storage_nodes_);
  for (u32 i = 0; i < num_storage_nodes_; ++i) {
    auto& qp_credits = peer_rdma_read_qp_outstanding_.emplace_back(peer_qps_per_peer_);
    for (auto& credit : qp_credits) {
      credit.store(0, std::memory_order_relaxed);
    }
    if (i != storage_id_) {
      peer_qps_[i].resize(peer_qps_per_peer_);
      peer_qp_send_mutexes_[i].reserve(peer_qps_per_peer_);
      for (u32 qp_idx = 0; qp_idx < peer_qps_per_peer_; ++qp_idx) {
        peer_qp_send_mutexes_[i].push_back(std::make_unique<std::mutex>());
      }
      peer_remote_tokens_[i] = std::make_unique<MemoryRegionToken>();
    }
  }
```

每个 peer 建多少条 QP？`peer_qps_per_peer_ = min(kMaxPeerQps, max(1, num_compute_threads_))`——也就是跟计算节点的线程数对齐，但不超 `kMaxPeerQps`。多 QP 的目的是分摊 send queue 压力：`peer_rdma.cc:63-66` 的日志说明 QP0 留给 RPC，QP1..N 留给数据 RDMA read。

`peer_qp_send_mutexes_` 是一个 `vec<vec<unique_ptr<mutex>>>`——每条 QP 一把互斥锁，因为 ibverbs 的 `post_send` 不是线程安全的，多线程共享一条 QP 必须串行化。

`peer_remote_tokens_` 是 `vec<unique_ptr<MemoryRegionToken>>`，每个 peer 一个，用来存"远端存储节点 index_buffer 的 MR token"——本地发起 RDMA read 时要带上远端 rkey。

`peer_rdma_read_qp_outstanding_` 是一个 `vec<vec<atomic<u32>>>`，第一维是 peer，第二维是该 peer 的每条 QP——记录每条 QP 上当前 in-flight 的 RDMA read 数量，用于 per-QP 信用控制。

接着是"低 id 主动连、高 id 被动等"的全连接模式：

```cpp
  for (u32 peer_id = 0; peer_id < storage_id_; ++peer_id) {
    for (u32 qp_idx = 0; qp_idx < peer_qps_per_peer_; ++qp_idx) {
      const auto endpoint = parse_endpoint(config.storage_peers[peer_id], config.port);
      const u32 encoded_id = storage_id_ * peer_qps_per_peer_ + qp_idx;
      peer_qps_[peer_id][qp_idx] =
        peer_context_->connect_to_server(endpoint.address, endpoint.port, encoded_id);
    }
  }
  const u32 incoming_peer_count = num_storage_nodes_ - storage_id_ - 1;
  for (u32 i = 0; i < incoming_peer_count * peer_qps_per_peer_; ++i) {
    auto [qp, encoded_id] = peer_context_->wait_for_connection();
    const u32 peer_id = encoded_id / peer_qps_per_peer_;
    const u32 remote_qp_idx = encoded_id % peer_qps_per_peer_;
    lib_assert(peer_id < num_storage_nodes_, "invalid peer storage id");
    lib_assert(peer_id > storage_id_, "unexpected lower peer connection");
    lib_assert(remote_qp_idx < peer_qps_per_peer_, "invalid peer QP index");
    lib_assert(peer_qps_[peer_id][remote_qp_idx] == nullptr, "duplicate peer QP connection");
    peer_qps_[peer_id][remote_qp_idx] = std::move(qp);
  }
  peer_context_->close_server_socket();
```

这个设计很优雅：

- 对于 `peer_id < storage_id_`，本节点主动 `connect_to_server`，并且把 `(storage_id_, qp_idx)` 编码进 `encoded_id = storage_id_ * peer_qps_per_peer_ + qp_idx` 作为 connect 的"用户数据"传给对端——对端的 `wait_for_connection` 会拿到这个 `encoded_id`，从而知道"这条连接来自 peer `encoded_id / peer_qps_per_peer_`，对端的 QP idx 是 `encoded_id % peer_qps_per_peer_`"。
- 对于 `peer_id > storage_id_`，本节点被动 `wait_for_connection`，从对端发来的 `encoded_id` 解出 `peer_id` 与 `remote_qp_idx`，把这条 QP 存进 `peer_qps_[peer_id][remote_qp_idx]`。

注意 `peer_qps_[peer_id][remote_qp_idx]` 这里存的是"对端的 QP idx"——也就是说本地这一侧的 QP 顺序不重要，只按对端报上来的 idx 存。这避免了"双方对 QP 顺序的视图不一致"的协调问题。

全连接完成之后 `close_server_socket()` 关掉 listen socket，不再接受新连接。

**token 交换：**

```cpp
  peer_index_region_ = std::make_unique<MemoryRegion>(*peer_context_);
  peer_index_region_->register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);

  const MemoryRegionToken local_token = peer_index_region_->createToken();
  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    LocalMemoryRegion peer_token_region{*peer_context_, peer_remote_tokens_[peer_id].get(), sizeof(MemoryRegionToken)};
    peer_control_qp(peer_id)->post_receive(peer_token_region);
    peer_control_qp(peer_id)->post_send_inlined(&local_token, sizeof(local_token), IBV_WR_SEND);
    peer_context_->poll_send_cq_until_completion();
    peer_context_->receive();
  }
```

这一段把"自己的 index_buffer token"通过 QP0（`peer_control_qp`）发给每个 peer，同时接收 peer 的 token 存进 `peer_remote_tokens_[peer_id]`。注意它注册了一个**新的 MR**（`peer_index_region_`）——这是因为 peer 网络用的是 `peer_context_` 的 PD，与 `context_` 的 PD 不同，同一个内存区域必须在两个 PD 下分别注册才能被两条网络的 rkey 寻址。`peer_remote_tokens_[peer_id]` 拿到后，本地就能用 `peer_remote_tokens_[peer_id].get()` 作为 `post_send` 的 `remote` 参数发起 RDMA read/write/CAS 到对端的 `index_buffer_`。

**scratch buffer：**

```cpp
  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
  peer_scratch_buffer_.allocate(scratch_bytes);
  peer_scratch_buffer_.touch_memory();
  peer_scratch_region_ =
    std::make_unique<LocalMemoryRegion>(*peer_context_, peer_scratch_buffer_.get_full_buffer(), scratch_bytes);
  peer_send_wcs_.resize(std::max<i32>(1, peer_context_->get_config().max_send_queue_wr));
```

`peer_scratch_buffer_` 是本地侧的"RDMA 读落地缓冲区"——因为 RDMA read 要把数据读到一段本地注册的内存里。它至少 64 MiB，且至少能装下 4 个节点（`VamanaNode::total_size() * 4`），保证一次邻居表批量读有地方放。`peer_scratch_region_` 是它的 MR。注意后面 `post_peer_read_async` 路径用的是每线程的 `StorageOwnerThread::scratch_buffer` 而不是这个全局的，这个全局的只用于同步路径（`remote_read_bytes` 等）。

**启动 peer RPC 运行时 + 探活：**

```cpp
  setup_peer_rpc_runtime(config);

  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    u64 header_words[2]{};
    remote_read_bytes(peer_id, 0, header_words, sizeof(header_words), 0);
  }
}
```

`setup_peer_rpc_runtime` 准备 peer RPC 的 send/recv 槽（第 24 课）。最后这一段"每个 peer 读一次它 buffer 头 16 字节"是一个探活/预热——确保所有 peer QP 都跑通过一次 RDMA read，避免第一次真实读时撞上 lazy 初始化的延迟。

### 23.4.2 QP 选择与信用控制

QP 选择策略在 `peer_rdma.cc:97-118`：

```cpp
QP& MemoryNode::peer_control_qp(u32 shard_id) {
  lib_assert(shard_id < peer_qps_.size(), "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(!peer_qps_[shard_id].empty() && peer_qps_[shard_id][0] != nullptr,
             "peer control QP is not initialized for shard " + std::to_string(shard_id));
  return peer_qps_[shard_id][0];
}

u32 MemoryNode::peer_data_qp_index(u32 worker_id) const {
  lib_assert(peer_qps_per_peer_ > 0, "peer QP count is not initialized");
  if (peer_qps_per_peer_ == 1) {
    return 0;
  }
  return 1 + worker_id % (peer_qps_per_peer_ - 1);
}
```

QP0 永远是 control QP（RPC 用）；数据 QP 在 `1..peer_qps_per_peer_-1` 之间按 `worker_id` 取模分摊。当只有一条 QP 时，control 与 data 共用 QP0。

信用控制三个函数 `peer_rdma.cc:124-136`：

```cpp
u32 MemoryNode::peer_rdma_read_credit_limit_per_qp() const {
  return std::max<u32>(1, std::min<u32>(storage_owner_peer_rdma_tokens_, kPeerSafeRdAtomic));
}

u32 MemoryNode::peer_rdma_read_credit_limit() const {
  const u32 per_peer_safe = std::max<u32>(1, peer_qps_per_peer_) * kPeerSafeRdAtomic;
  return std::max<u32>(1, std::min<u32>(storage_owner_peer_rdma_tokens_, per_peer_safe));
}

u32 MemoryNode::peer_rdma_read_global_credit_limit() const {
  const u32 remote_peer_count = num_storage_nodes_ > 1 ? num_storage_nodes_ - 1 : 1;
  return std::max<u32>(1, peer_rdma_read_credit_limit() * remote_peer_count);
}
```

三层信用：

1. **per-QP 限制**：`min(storage_owner_peer_rdma_tokens_, 8)`——单条 QP 上最多 8 个 in-flight read（`kPeerSafeRdAtomic = 8`），这是 RDMA read 的硬件安全上限。
2. **per-peer 限制**：`min(storage_owner_peer_rdma_tokens_, peer_qps_per_peer_ * 8)`——同一个 peer 上所有 QP 加起来不超过这个值。
3. **全局限制**：`peer_rdma_read_credit_limit() * (num_storage_nodes_ - 1)`——所有 peer 加起来不超过这个值，防止一次 burst 把 `peer_scratch_buffer_` 撑爆。

`try_acquire_counter` 是 CAS 加 1 的通用原语：

```cpp
bool MemoryNode::try_acquire_counter(std::atomic<u32>& counter, u32 limit) {
  u32 current = counter.load(std::memory_order_acquire);
  while (current < limit) {
    if (counter.compare_exchange_weak(current,
                                      current + 1,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
```

两层信用获取 `peer_rdma.cc:151-170`：

```cpp
bool MemoryNode::try_acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  if (!try_acquire_counter(peer_rdma_read_outstanding_[shard_id], peer_rdma_read_credit_limit())) {
    return false;
  }
  if (try_acquire_counter(peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
                          peer_rdma_read_credit_limit_per_qp())) {
    return true;
  }
  peer_rdma_read_outstanding_[shard_id].fetch_sub(1, std::memory_order_acq_rel);
  return false;
}

void MemoryNode::acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  while (!try_acquire_peer_rdma_read_credit(shard_id, qp_idx)) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}
```

注意它的"先 per-peer 再 per-QP，失败要回滚 per-peer"——这是为了避免"per-peer 占住但 per-QP 失败"导致死锁。`acquire_peer_rdma_read_credit` 是阻塞版，忙等时调用 `poll_peer_send_cq()` 推进完成队列——这一点很重要：因为 credit 是在完成回调里归还的（见 23.4.3），不轮 CQ 就永远等不到 credit 释放。

### 23.4.3 wr_id 编码、完成回调、send CQ 轮询

`peer_rdma.cc:120-180`：

```cpp
u64 MemoryNode::peer_coroutine_wr_id(u32 thread_id, u32 coroutine_id) {
  return encode_64bit(thread_id, coroutine_id);
}

u64 MemoryNode::next_peer_sync_wr_id() {
  const u32 id = peer_sync_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
  return encode_64bit(kPeerSyncWrOwner, id);
}

u64 MemoryNode::next_peer_async_wr_id() {
  const u32 id = peer_async_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
  return encode_64bit(kPeerAsyncWrOwner, id);
}
```

64 位 `wr_id` 被拆成 `(owner, id)` 两段：

- `owner = kPeerSyncWrOwner`（`u32::max()`）：同步路径，完成回调把它插入 `peer_sync_completions_` 集合，`wait_peer_sync_completion` 轮询这个集合。
- `owner = kPeerAsyncWrOwner`（`u32::max() - 1`）：协程异步路径，完成回调查 `peer_pending_sends_[wr_id]` 拿到 `thread` 指针与 `coroutine_id`，把 `post_balances[coroutine_id]--`，让协程 forward 推进。
- `owner = thread_id`：协程异步路径的另一种编码方式（用 `peer_coroutine_wr_id(thread_id, coroutine_id)`），完成回调直接 `post_balances[coroutine_id]--`。
- `owner < storage_owner_threads_.size()`：storage-owner 线程发起的异步读，完成回调也走 `post_balances[id]--`。

`register_peer_pending_send_locked` 与 `handle_peer_send_completion` 是这套机制的枢纽，`peer_rdma.cc:182-237`：

```cpp
void MemoryNode::register_peer_pending_send_locked(u64 wr_id, PeerPendingSend pending) {
  std::lock_guard<std::mutex> lock(peer_completion_mutex_);
  peer_pending_sends_[wr_id] = pending;
}

void MemoryNode::handle_peer_send_completion(u64 wr_id) {
  PeerPendingSend pending;
  bool has_pending = false;
  {
    std::lock_guard<std::mutex> lock(peer_completion_mutex_);
    const auto pending_it = peer_pending_sends_.find(wr_id);
    if (pending_it != peer_pending_sends_.end()) {
      pending = pending_it->second;
      peer_pending_sends_.erase(pending_it);
      has_pending = true;
    }
  }
  if (has_pending) {
    if (pending.release_rpc_slot) {
      release_peer_rpc_send_slot(pending.target_shard, pending.rpc_slot_id);
      return;
    }
    if (pending.rdma_read_credit) {
      peer_rdma_read_outstanding_[pending.target_shard].fetch_sub(1, std::memory_order_acq_rel);
      if (pending.target_shard < peer_rdma_read_qp_outstanding_.size() &&
          pending.target_qp_idx < peer_rdma_read_qp_outstanding_[pending.target_shard].size()) {
        peer_rdma_read_qp_outstanding_[pending.target_shard][pending.target_qp_idx].fetch_sub(
          1, std::memory_order_acq_rel);
      }
    }
    if (pending.async) {
      lib_assert(pending.thread != nullptr, "async peer RDMA completion has no owner thread");
      lib_assert(pending.coroutine_id < pending.thread->post_balances.size(),
                 "async peer RDMA completion has invalid coroutine id");
      auto& balance = pending.thread->post_balances[pending.coroutine_id];
      --balance;
      peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
      return;
    }
  }

  const auto [owner, id] = decode_64bit(wr_id);
  if (owner == kPeerSyncWrOwner) {
    {
      std::lock_guard<std::mutex> lock(peer_completion_mutex_);
      peer_sync_completions_.insert(wr_id);
    }
    peer_completion_cv_.notify_all();
    return;
  }
  if (owner < storage_owner_threads_.size() && storage_owner_threads_[owner]) {
    auto& balance = storage_owner_threads_[owner]->post_balances[id];
    --balance;
    peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
  }
}
```

`PeerPendingSend`（`storage_owner_state.hh:138-148`）是一个"完成时要做的善后"结构：

```cpp
struct PeerPendingSend {
  u32 target_shard{};
  u32 target_qp_idx{};
  u32 thread_id{};
  u32 coroutine_id{};
  StorageOwnerThread* thread{};
  bool async{};
  bool rdma_read_credit{};
  bool release_rpc_slot{};
  u32 rpc_slot_id{};
};
```

完成回调优先级：

1. 如果 `release_rpc_slot`，归还 peer RPC 的 send 槽（第 24 课）。
2. 否则如果 `rdma_read_credit`，归还 per-peer + per-QP 信用。
3. 在 2 的基础上如果是异步读（`async`），还要 `post_balances[coroutine_id]--` 让协程 forward；同时 `peer_async_rdma_outstanding_--` 让全局异步限额有空间。
4. 如果 `peer_pending_sends_` 里没找到（说明是同步路径直接用 `next_peer_sync_wr_id()` 编码的 wr_id），按 `wr_id` 高 32 位 `owner` 分支：`kPeerSyncWrOwner` 进 `peer_sync_completions_` 集合并唤醒等待；`owner` 是 storage-owner 线程 id 时，`post_balances[id]--`。

`poll_peer_send_cq` 就是把 send CQ 上所有 WC 拉出来调一遍 `handle_peer_send_completion`：

```cpp
void MemoryNode::poll_peer_send_cq() {
  if (!peer_context_) {
    return;
  }
  std::lock_guard<std::mutex> lock(peer_send_cq_mutex_);
  Context::poll_send_cq(peer_send_wcs_.data(),
                        static_cast<i32>(peer_send_wcs_.size()),
                        peer_context_->get_send_cq(),
                        [&](u64 wr_id) { handle_peer_send_completion(wr_id); });
}
```

注意 `peer_send_cq_mutex_` 保护——多个线程都可能主动 poll CQ（比如协程在等 credit 时 poll、专门的 progress 线程也在 poll），必须串行化。

`wait_peer_sync_completion` 有两条路径，`peer_rdma.cc:260-278`：

```cpp
void MemoryNode::wait_peer_sync_completion(u64 wr_id) {
  if (peer_rpc_progress_running_.load(std::memory_order_acquire) &&
      !current_peer_rpc_progress_thread_) {
    std::unique_lock<std::mutex> lock(peer_completion_mutex_);
    peer_completion_cv_.wait(lock, [&]() {
      return peer_sync_completions_.contains(wr_id) ||
             !peer_rpc_progress_running_.load(std::memory_order_acquire);
    });
    const auto completion = peer_sync_completions_.find(wr_id);
    if (completion != peer_sync_completions_.end()) {
      peer_sync_completions_.erase(completion);
      return;
    }
  }
  while (!consume_peer_sync_completion(wr_id)) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}
```

如果当前有线程正在跑 `peer_rpc_progress_loop`（即 peer RPC progress 线程在持续 poll CQ），且当前线程不是那个 progress 线程自己，那就走条件变量等待——progress 线程会持续 poll CQ 并在完成时 `notify_all`。否则自己 busy-poll。这个优化避免了"每个调用方都自己 poll CQ"导致的争用。

### 23.4.4 三个同步原语：read / write / CAS

`peer_rdma.cc:321-444` 给出三个同步阻塞原语。它们的结构高度相似，以 `remote_read_bytes` 为例：

```cpp
void MemoryNode::remote_read_bytes(u32 shard_id, u64 remote_offset, void* dst, size_t bytes, size_t scratch_offset) {
  if (bytes == 0) return;
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
             "peer token is invalid for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + bytes <= mn_memory_bytes_,
             "peer RDMA read exceeds shard bounds: shard=" + std::to_string(shard_id) +
               " offset=" + std::to_string(remote_offset) +
               " bytes=" + std::to_string(bytes) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  lib_assert(scratch_offset + bytes <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
  byte_t* scratch = scratch_buffer.get_full_buffer() + scratch_offset;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, nullptr, false, true});
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
    qp->post_send(reinterpret_cast<u64>(scratch),
                  static_cast<u32>(bytes),
                  scratch_region.get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  peer_remote_tokens_[shard_id].get(),
                  remote_offset,
                  0,
                  wr_id);
  }
  wait_peer_sync_completion(wr_id);
  std::memcpy(dst, scratch, bytes);
}
```

要点：

1. **边界校验**：`remote_offset + bytes <= mn_memory_bytes_`，防止读越界（注意 `mn_memory_bytes_` 是每分片的 buffer 大小，因为所有分片的 `.dat` 都是同样大小）。
2. **QP 选择**：如果当前线程有 `StorageOwnerThread` 上下文，按 `owner_thread->id` 选 QP；否则用 worker 0。
3. **scratch buffer 选择**：优先用线程私有的（`owner_thread->scratch_buffer`），fallback 到全局 `peer_scratch_buffer_`。线程私有 scratch 避免了同步路径之间的争用。
4. **信用获取**：`acquire_peer_rdma_read_credit` 阻塞直到拿到 per-peer + per-QP 信用。
5. **wr_id 编码**：`next_peer_sync_wr_id()`，owner 是 `kPeerSyncWrOwner`。
6. **register pending**：注册 `PeerPendingSend{rdma_read_credit=true}`，这样完成回调会归还信用。
7. **post_send**：注意 `qp->post_send` 不是线程安全的，所以外面套了 `peer_qp_send_mutexes_[shard_id][qp_idx]`。
8. **wait + memcpy**：`wait_peer_sync_completion` 阻塞到完成，然后把数据从 scratch 拷到 `dst`。

`remote_write_bytes` 与 `remote_compare_and_swap` 结构相同，只是 `IBV_WR` 类型不同（WRITE / CAS），且 write 不需要 credit（write 不会因 read 限额被卡）。CAS 用 `qp->post_CAS`，返回值通过 scratch 拿回。

`try_lock_remote_header` 是 CAS 之上的语义封装，`peer_rdma.cc:446-455`：

```cpp
std::pair<bool, u64> MemoryNode::try_lock_remote_header(RemotePtr rptr) {
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &header, sizeof(header), 0);
  if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
    return {false, header};
  }
  const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
  const u64 original = remote_compare_and_swap(rptr.memory_node(), rptr.byte_offset(), header, desired, align_up(sizeof(header)));
  return {original == header, original};
}
```

这是跨分片节点加锁的 read-modify-CAS：先 read 节点头 8 字节，看锁位（`HEADER_NODE_LOCK`）；如果没锁，CAS 把锁位置 1，CAS 返回的 `original` 等于 read 时看到的值才算加锁成功。这是第 25 课图修改的"跨分片 stitch"和"反向边写入"用到的基础设施。

### 23.4.5 异步读：协程驱动

`post_peer_read_async` 是协程路径的入口，`peer_rdma.cc:280-319`：

```cpp
void MemoryNode::post_peer_read_async(StorageOwnerThread& thread,
                                      u32 shard_id,
                                      u64 remote_offset,
                                      byte_t* dst,
                                      size_t bytes,
                                      size_t local_offset) {
  if (bytes == 0) {
    return;
  }
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(thread.has_peer_scratch(), "storage-owner thread scratch is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + bytes <= mn_memory_bytes_, "peer RDMA read exceeds shard bounds");
  const u32 qp_idx = peer_data_qp_index(thread.id);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  while (peer_async_rdma_outstanding_.load(std::memory_order_acquire) >= peer_rdma_read_global_credit_limit()) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
  peer_async_rdma_outstanding_.fetch_add(1, std::memory_order_acq_rel);
  thread.track_post();
  const u64 wr_id = next_peer_async_wr_id();
  register_peer_pending_send_locked(
    wr_id,
    PeerPendingSend{shard_id, qp_idx, thread.id, thread.running_coroutine, &thread, true, true});
  std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
  qp->post_send(reinterpret_cast<u64>(dst),
                static_cast<u32>(bytes),
                thread.scratch_region->get_lkey(),
                IBV_WR_RDMA_READ,
                true,
                false,
                peer_remote_tokens_[shard_id].get(),
                remote_offset,
                local_offset,
                wr_id);
}
```

与同步版的关键差异：

1. `dst` 直接是协程提供的指针（通常是协程 scratch 里的某个偏移），不需要再 memcpy。
2. 多了一层全局异步限额 `peer_async_rdma_outstanding_ < peer_rdma_read_global_credit_limit()`，防止协程一次性 post 太多。
3. `thread.track_post()` 把 `post_balances[running_coroutine]++`——协程 forward 时会等到 `post_balances[coroutine_id] == 0` 才继续，这就是"等所有异步 RDMA 完成"的机制。
4. `PeerPendingSend{async=true, rdma_read_credit=true, thread=&thread, coroutine_id=thread.running_coroutine}`——完成回调会同时归还信用、减 `post_balances`、减全局异步计数。

协程路径在第 25 课插入流程里大量使用，这里只需理解"协程发 read → yield → progress 线程 poll CQ → 完成回调减 balance → 协程 forward 时检查 balance 为 0"。

## 23.5 关键数据结构与内存布局

把上面所有信息综合起来，存储节点启动后的内存布局与运行时拓扑如下。

### 23.5.1 部署拓扑图

```
┌──────────────────────────── 计算/发起端 ────────────────────────────┐
│  initiator (compute 0) ─── Parameters {num_threads, gpu_rdma_qps} ──┐│
│                                                                     ││
│  compute node i ── access token (rkey+addr) ──┐                     ││
│  compute node i ── N 条 DetachedQP (GPU RDMA) │                     ││
│  compute node i ── startup Request(magic) ────┘                     ││
└─────────────────────────────────────────────────────────────────────┘│
                                                                       │
       network 1: "compute↔memory" (context_, cm_)                     │
       - server: MemoryNode 被动 accept                                │
       - 计算 GPU 通过 GPUNetIO 直接 RDMA read index_buffer_            │
       - 计算 CPU 写 reclaim_ack_sequences[i]                          │
       │                                                                ▼
┌────────────────────────── 存储节点 storage_id_ ─────────────────────────┐
│                                                                          │
│  Context context_   ──┐   PD 1                                          │
│  ServerConnectionManager cm_                                              │
│    initiator_qp ──── Parameters / startup handshake                      │
│    client_qps[i] ─── access token SEND / synchronize() barrier            │
│    DetachedQPs ────  被 compute GPU RDMA read                             │
│                                                                          │
│  Context peer_context_ ──┐  PD 2 (与 context_ 隔离)                      │
│    peer_qps_[peer][0]    ├── control QP (RPC)                            │
│    peer_qps_[peer][1..N] └── data QP (RDMA read/write/CAS)               │
│    peer_index_region_  ──  index_buffer 在 PD2 下的 MR                    │
│    peer_scratch_region_──  peer_scratch_buffer 在 PD2 下的 MR             │
│                                                                          │
│  MemoryRegion index_region_  ── index_buffer 在 PD1 下的 MR              │
│    createToken() → 发给所有 compute node                                 │
│                                                                          │
│  HugePage<byte_t> index_buffer_  (mn_memory_bytes_, 2MiB 大页)           │
│    见 23.5.2 内存布局                                                     │
│                                                                          │
│  后台线程:                                                                │
│    storage_insert_workers_          (第 25 课)                            │
│    storage_owner_maintenance_workers_(第 26 课)                            │
│    peer_reverse_workers_            (第 24 课)                            │
│    peer_stitch_search_workers_      (第 24 课)                            │
│    peer_rpc_progress_thread_        (第 24 课)                            │
│    peer_reverse_response_thread_    (第 24 课)                            │
│    peer_reverse_outgoing_thread_    (第 24 课)                            │
│                                                                          │
│  主线程: service_storage_runtime → 主循环 (转发 insert, reclaim)          │
└──────────────────────────────────────────────────────────────────────────┘
       │
       │ network 2: "memory↔memory" (peer_context_, 全连接)
       │   - 每对 (i, j) 双向 QP，i<j 时 i 主动 connect、j 被动 wait
       │   - QP0 control: peer RPC send/recv (第 24 课)
       │   - QP1..N data: RDMA read 邻居表 / CAS 节点头加锁
       ▼
┌────────────────────────── 其他存储节点 ──────────────────────────────────┐
│   ... 同构 ...                                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 23.5.2 `index_buffer_` 内存布局

```
偏移                                          内容
─────────────────────────────────────────────────────────────────────────
0x0000                                        free pointer (u64)
                                              启动后 = gpu_dynamic_node_base_
0x0010 (kNodeBaseOffset)                      静态节点数组
  └─ 每节点 VamanaNode::total_size() 字节
     ┌─ header (u64)  : deleted / lock / generation 等
     ├─ id (node_t)
     ├─ generation (u32)
     ├─ vector       : vector_bytes() 字节
     └─ hot graph    : graph_hot_bytes() 字节 (compact pointer 邻居表)
  共 gpu_static_node_count_ 个

gpu_static_dynamic_base_                      静态区结束 / 动态区开始
  (与磁盘 .dat 的 persisted_free_pointer 一致)

gpu_storage_control_offset_                   4 KiB 控制页
= align_up(gpu_static_dynamic_base_, 64)
  ├─ [   0 .. 640 ) StorageControlBlock (640B)
  │     ├─ magic / version / header_bytes
  │     ├─ shard_id
  │     ├─ dynamic_record_bytes / dynamic_hot_offset / dynamic_code_offset / code_bytes
  │     ├─ compute_client_count = num_clients_
  │     ├─ next_maintenance_sequence / durable_maintenance_sequence
  │     ├─ dynamic_high_watermark = gpu_dynamic_node_base_
  │     ├─ reclaim_pending_nodes / reclaim_reused_nodes
  │     └─ reclaim_ack_sequences[64]  ← 计算侧写 ACK (第 16 课)
  │
  ├─ [ 640 ..1024 ) 预留
  │
  ├─ [1024 ..1472) StorageRoutePublication (448B)
  │     (kStorageRoutePublicationOffset = 1024)
  │     ├─ sequence_begin / sequence_end  ← 双段写序列号
  │     ├─ magic / version / header_bytes
  │     ├─ shard_id / slot_count=8 / code_bytes
  │     ├─ body_checksum
  │     ├─ slots[8]: StorageRouteSlot {remote_node, id, generation, navigation_code[32]}
  │     └─ sequence_end
  │
  └─ [1472 ..4096) 预留

gpu_storage_control_offset_ + 4096            PQ32 码流区
  (kStorageControlBytes = 4096)
  └─ gpu_static_node_count_ * gpu_navigation_code_bytes_ 字节
     每 static node 一条 PQ 码 (供 GPUNetIO 流式读)

gpu_dynamic_node_base_                        动态节点区起点
  └─ 在线 upsert 分配的节点
     每节点 VamanaNode::allocation_size() 字节
     (total_size + hot graph + PQ 码三段)
     └─ dynamic_high_watermark 推进
     └─ retire 后进 StorageReclaimQueue 等待复用

mn_memory_bytes_                              buffer 上限
```

几个关键约束（来自 `load_index_file` 的校验）：

- `gpu_storage_control_offset_ == align_up(gpu_static_dynamic_base_, 64)`：控制页必须紧跟静态区、64 字节对齐。
- `gpu_dynamic_node_base_ >= gpu_storage_control_offset_ + 4096 + payload_bytes`：动态区在 PQ 码流之后。
- `(gpu_dynamic_node_base_ - gpu_static_dynamic_base_) % VamanaNode::allocation_size() == 0`：动态区起点与静态区起点之间的距离必须是 allocation_size 的整数倍（保证动态节点的偏移可以统一寻址）。

### 23.5.3 `StorageControlBlock` 与 `StorageRoutePublication` 双段写

`publish_storage_owner_route_table`（`memory_node.cc:479-558`）是路由发布的核心。它的发布机制是经典的"双段写序列号"模式：

```cpp
auto* destination = reinterpret_cast<
  gpu_search::format::StorageRoutePublication*>(
  index_buffer_.get_full_buffer() + gpu_storage_control_offset_ +
  gpu_search::format::kStorageRoutePublicationOffset);
std::atomic_ref<u64> begin_sequence(destination->sequence_begin);
std::atomic_ref<u64> end_sequence(destination->sequence_end);
const u64 current = begin_sequence.load(std::memory_order_relaxed);
const u64 odd = (current & ~u64{1}) + 1;
const u64 even = odd + 1;
begin_sequence.store(odd, std::memory_order_release);
end_sequence.store(odd, std::memory_order_release);
std::memcpy(
  reinterpret_cast<byte_t*>(destination) +
    offsetof(gpu_search::format::StorageRoutePublication, magic),
  reinterpret_cast<const byte_t*>(&next) +
    offsetof(gpu_search::format::StorageRoutePublication, magic),
  offsetof(gpu_search::format::StorageRoutePublication, sequence_end) -
    offsetof(gpu_search::format::StorageRoutePublication, magic));
std::atomic_thread_fence(std::memory_order_release);
end_sequence.store(even, std::memory_order_release);
begin_sequence.store(even, std::memory_order_release);
```

流程：

1. 把 `sequence_begin` 与 `sequence_end` 同时写为同一个奇数 `odd`——这告诉读者"我正在改 body，别读"。
2. `memcpy` 覆盖 body（从 `magic` 到 `sequence_end` 之前的所有字段）。
3. memory fence 保证 body 可见先于"完成标记"。
4. 把 `sequence_end` 写为偶数 `even`，再把 `sequence_begin` 也写为 `even`——读者看到 `begin == end == 偶数` 才认为 body 是完整的。

读者（计算节点 GPUNetIO 一侧）的协议是：读 `sequence_begin` → 读 body → 再读 `sequence_begin`，如果两次 `sequence_begin` 都是同一个偶数且与 `sequence_end` 一致，body 可用；否则丢弃这次读，用上一次的快照。`body_checksum` 是额外一层校验。注释 `index_format.hh:107-111` 直接说明了这个设计。

`StorageControlBlock::reclaim_ack_sequences[64]` 是另一处被远端写入的地方——计算节点完成一轮 reclaim 后把自己的序号槽位写高，存储侧 `minimum_compute_reclaim_ack()` 取所有槽的最小值，与 `durable_maintenance_sequence` 一起决定哪些 retire 节点可以复用（见 23.6）。

## 23.6 `storage_reclaim.hh`：存储侧 reclaim 队列

`storage_reclaim.hh:13-52` 定义了 `StorageReclaimQueue`——一个非常紧凑的 retire/acquire 队列：

```cpp
class StorageReclaimQueue {
public:
  void retire(RemotePtr pointer, u64 maintenance_sequence) {
    if (pointer.is_null() || maintenance_sequence == 0) return;
    pending_[maintenance_sequence].push_back(pointer);
    ++size_;
  }

  std::optional<RemotePtr> acquire(u64 durable_sequence,
                                   u64 acknowledged_sequence) {
    const u64 safe_sequence = std::min(durable_sequence, acknowledged_sequence);
    while (!pending_.empty() && pending_.begin()->first <= safe_sequence) {
      auto nodes = std::move(pending_.begin()->second);
      pending_.erase(pending_.begin());
      ready_.insert(ready_.end(),
                    std::make_move_iterator(nodes.begin()),
                    std::make_move_iterator(nodes.end()));
    }
    if (ready_.empty()) return std::nullopt;
    const RemotePtr pointer = ready_.back();
    ready_.pop_back();
    --size_;
    ++reused_;
    return pointer;
  }
  ...
private:
  std::map<u64, std::vector<RemotePtr>> pending_;
  std::vector<RemotePtr> ready_;
  size_t size_{};
  u64 reused_{};
};
```

逻辑很清楚：

- `retire(pointer, seq)`：把一个被删/被替换的节点按 `maintenance_sequence` 入队。`pending_` 是 `map<seq, vec<RemotePtr>>`，按 seq 升序排列。
- `acquire(durable, acknowledged)`：`safe = min(durable, acknowledged)`。把所有 `seq <= safe` 的节点从 `pending_` 移到 `ready_`，然后从 `ready_` 末尾取一个返回。

这里的两个序号含义：

- `durable_sequence`：本存储节点"已持久化"的 maintenance 序号——也就是即便崩溃也已经恢复一致的序号（见第 26 课维护的水位线）。
- `acknowledged_sequence`：所有计算节点都已 ACK 的 reclaim 序号——也就是计算侧 `reclaim_ack_sequences[i]` 的最小值（`minimum_compute_reclaim_ack()`）。

为什么取 min？因为 retire 的节点只有在"维护已经持久化"且"所有计算节点都不再持有指向它的旧 RemotePtr"两个条件同时满足时才安全复用。任一条件未满足，复用都可能让某个计算节点读到被改写的内存，破坏 RCU 语义。这是第 16 课计算侧 reclaim ACK 的存储侧消费端，两者通过 `StorageControlBlock::reclaim_ack_sequences` 与 `StorageReclaimQueue` 闭合。

`MemoryNode` 持有的相关字段在 `memory_node.hh:692-694`：

```cpp
mutable std::mutex storage_owner_reclaim_mutex_;
memory_node_detail::StorageReclaimQueue storage_owner_reclaim_queue_;
std::atomic<u64> storage_owner_reclaim_candidates_{0};
```

`storage_owner_reclaim_mutex_` 保护 `storage_owner_reclaim_queue_`，因为 retire 与 acquire 可能从不同线程发起（维护 worker retire、插入 worker acquire）。`storage_owner_reclaim_candidates_` 是一个观测计数器。

## 23.7 `storage_owner_state.hh`：storage owner 的运行时值类型

`storage_owner_state.hh` 把 `MemoryNode` 内部用到的"值类型"集中定义在 `memory_node_detail` 命名空间里，避免 `MemoryNode` 类声明过长。这一节挑几个关键的讲。

### 23.7.1 `BeamEntry` 与 `NodeSnapshot`

```cpp
struct BeamEntry {
  RemotePtr rptr;
  distance_t distance{};
  bool expanded{false};
};

struct NodeSnapshot {
  RemotePtr rptr;
  u64 header{};
  node_t id{};
  u32 generation{};
  bool deleted{};
  vec<byte_t> vector_data;
};
```

`BeamEntry` 是 CPU beam search 的候选——`rptr` 指向节点、`distance` 是它到 query 的距离、`expanded` 标记是否已经展开过邻居。`NodeSnapshot` 是节点的完整快照，专门为"批量读邻居表/向量"设计——`vector_data` 是把向量字节拷出来，避免在排序/剪枝过程中反复 RDMA read 同一个节点。

### 23.7.2 `StorageOwnerCoroutineScratch`

```cpp
struct StorageOwnerCoroutineScratch {
  hashset_t<RemotePtr> visited;
  hashset_t<RemotePtr> empty_skip;
  vec<BeamEntry> beam;
  vec<RemotePtr> neighbors;
  vec<RemotePtr> unvisited;
  vec<RemotePtr> batch;
  vec<byte_t> neighbor_entry;
  vec<byte_t> neighbor_decoded;
  ...
};
```

这是协程的"工作内存"。注意它把所有 `vec` 都预声明在这里，并通过 `clear_search()` / `clear_prune()` / `clear_reverse_update()` 分阶段清空复用，而不是每次都 `new` 一遍——这是因为 storage-owner 插入流程在一轮里有 search、prune、reverse update 三个阶段，每阶段都需要大量临时容器，预分配 + clear 复用比反复分配快得多。这是第 25 课插入流程的核心数据结构。

### 23.7.3 `StorageOwnerThread`

```cpp
struct StorageOwnerThread {
  explicit StorageOwnerThread(u32 id, u32 num_coroutines, i32 max_send_queue_wr)
      : id(id),
        send_wcs(std::max<i32>(1, max_send_queue_wr)),
        post_balances(num_coroutines),
        coroutine_scratch_states(num_coroutines) {
    for (auto& balance : post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
  }

  void init_peer_scratch(Context& peer_context, size_t bytes, size_t per_coroutine_stride = 0) {
    scratch_stride = per_coroutine_stride == 0
                       ? align_to_cacheline(VamanaNode::total_size())
                       : align_to_cacheline(per_coroutine_stride);
    const size_t required_bytes = static_cast<size_t>(std::max<size_t>(1, post_balances.size())) * scratch_stride;
    scratch_buffer.allocate(std::max(bytes, required_bytes));
    scratch_buffer.touch_memory();
    scratch_region = std::make_unique<LocalMemoryRegion>(
      peer_context, scratch_buffer.get_full_buffer(), scratch_buffer.buffer_size);
  }
  ...
  u32 id{};
  vec<ibv_wc> send_wcs;
  vec<std::atomic<i32>> post_balances;
  vec<StorageOwnerCoroutineScratch> coroutine_scratch_states;
  StorageOwnerRequestScratch request_scratch;
  vec<u_ptr<StorageOwnerInsertCoroutine>> coroutines;
  HugePage<byte_t> scratch_buffer;
  std::unique_ptr<LocalMemoryRegion> scratch_region;
  u32 running_coroutine{};
  size_t scratch_stride{};
};
```

`StorageOwnerThread` 是 storage-owner 工作线程的执行上下文。它的设计有几个值得注意的点：

- **多协程共享一个线程**：`post_balances` 是 `vec<atomic<i32>>`，每个协程一个余额。协程 post 一个异步 RDMA read 时 `++balance`，完成回调 `--balance`；协程 forward 时检查 `balance == 0` 才继续。这是协作式多协程调度的经典"等所有 in-flight 完成"机制。
- **私有 scratch buffer**：`init_peer_scratch` 给每个线程分配一个独立的 `HugePage<byte_t>` + `LocalMemoryRegion`，用于 peer RDMA read 落地。每个协程在 scratch 里有一段独立的 stride（`scratch_stride`），互不干扰。
- **coroutine_scratch_states**：每个协程一份 `StorageOwnerCoroutineScratch`，在协程 yield/resume 之间保留工作内存。
- **send_wcs**：线程私有的 WC 数组，避免 poll CQ 时与其它线程争用。

`StorageOwnerThread` 是 23.4 提到的 `current_storage_owner_thread_` 指向的对象——任何代码都可以通过这个 thread_local 指针拿到"我现在的 scratch buffer、coroutine id、post balance"。

### 23.7.4 `PeerRpcRuntimeState`、`PeerPendingSend`、`FreshnessEntry`

```cpp
struct PeerRpcRuntimeState {
  HugePage<byte_t> buffer;
  std::unique_ptr<LocalMemoryRegion> region;
  size_t message_bytes{};
  size_t recv_region_bytes{};
  size_t sync_send_offset{};
  size_t async_send_offset{};
  u32 recv_slots_per_peer{1};
  u32 send_slots_per_peer{1};
};
```

`PeerRpcRuntimeState` 是 peer RPC 的 buffer/MR/slot 布局描述——所有 peer RPC 的 send/recv 槽都落在这一个大页里，按 offset 切分。具体的 send/recv 槽分配与协议在第 24 课讲。

`PeerPendingSend` 我们在 23.4.3 已经详细讲过。

`FreshnessEntry`（`storage_owner_state.hh:259-263`）：

```cpp
struct FreshnessEntry {
  RemotePtr current;
  u32 generation{};
  bool deleted{};
};
```

这是"某个 node id 当前最新版"的记录。`MemoryNode` 维护两套：

- `base_idmap_`（`memory_node.hh:713`）：从磁盘 idmap 加载的初始版本，是 owner-sharded 的"这个 id 在我分片上"的权威记录。
- `dynamic_freshness_shards_`（`memory_node.hh:714-715`）：256 个分片的 `DynamicFreshnessShard`，每个分片是 `{mutex, hashmap<node_t, FreshnessEntry>, hashset<node_t> mutations_inflight}`。在线 upsert 时把变更写到这里，让 `prepare_mutation`/`observe_storage_owner_route` 等快速判断"这个 id 现在是哪个指针、哪一代、是否被删"。256 个分片是为了减小锁争用。

## 23.8 `storage_owner_cpu_plan.hh`：CPU 侧执行计划

storage-owner 模式下一个存储节点要跑很多类线程，CPU 怎么分？`storage_owner_cpu_plan.hh:22-60` 给出了一份静态分配算法：

```cpp
inline StorageOwnerCpuPlan derive_storage_owner_cpu_plan(
    std::uint32_t available_cpus,
    std::uint32_t configured_threads,
    std::uint32_t rpc_parallelism,
    std::uint32_t configured_maintenance_workers,
    std::uint32_t remote_peer_count) {
  const std::uint32_t budget = std::max<std::uint32_t>(1, available_cpus);
  const std::uint32_t cpu_parallelism = std::max<std::uint32_t>(
    1, configured_threads / 2);
  const std::uint32_t fanout = std::max<std::uint32_t>(1, remote_peer_count + 1);

  StorageOwnerCpuPlan plan;
  plan.peer_progress_threads = remote_peer_count == 0 ? 0 : 3;
  plan.foreground_progress_threads = 1;
  plan.foreground_workers = std::min(
    std::max<std::uint32_t>(1, rpc_parallelism),
    std::max<std::uint32_t>(1,
      std::min(cpu_parallelism, budget / fanout)));
  plan.maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 10 : 1));
  plan.peer_reverse_workers = remote_peer_count == 0 ? 0 : std::min(
    std::uint32_t{8},
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 5 : 1));

  const std::uint64_t reserved =
    static_cast<std::uint64_t>(plan.peer_progress_threads) +
    plan.foreground_progress_threads +
    plan.foreground_workers + plan.maintenance_workers +
    plan.peer_reverse_workers;
  const std::uint32_t search_budget = reserved < budget
    ? static_cast<std::uint32_t>(budget - reserved) : 1;
  plan.peer_search_workers = remote_peer_count == 0 ? 0 :
    std::min(cpu_parallelism, std::max<std::uint32_t>(1, search_budget));
  return plan;
}
```

输出结构 `StorageOwnerCpuPlan`（`storage_owner_cpu_plan.hh:8-15`）：

```cpp
struct StorageOwnerCpuPlan {
  std::uint32_t foreground_workers{};
  std::uint32_t maintenance_workers{};
  std::uint32_t peer_search_workers{};
  std::uint32_t peer_reverse_workers{};
  std::uint32_t peer_progress_threads{};
  std::uint32_t foreground_progress_threads{};
};
```

分配原则：

1. **`peer_progress_threads = 3`**（仅当有远端 peer）：peer RPC 的 progress 线程数固定 3。这些线程专职 poll send/recv CQ，不能太少否则 CQ 跟不上。
2. **`foreground_progress_threads = 1`**：`service_storage_runtime` 主循环独占一个 CPU。
3. **`foreground_workers = min(rpc_parallelism, min(cpu_parallelism, budget / fanout))`**：插入 worker 数受配置 `rpc_parallelism` 上限、`configured_threads / 2`、以及"budget / fanout"（每个 fanout 路径至少分一个核）三者钳制。
4. **`maintenance_workers`**：维护 worker，最多 `budget / 10`（且不少于 1），上限是配置值。
5. **`peer_reverse_workers`**：peer 反向边更新 worker，最多 `budget / 5`，硬上限 8。
6. **`peer_search_workers`**：剩余预算全给 peer 搜索 worker——因为每次 finalize insert 都要在每个远端分片跑一次完整搜索，是最重的活。

注释 `storage_owner_cpu_plan.hh:18-21` 解释了为什么 search 拿剩余预算：每个 finalized insert 都要在每个远端分片跑一次完整 search，所以 search 是最重的负载，应该拿剩余预算；foreground 和 maintenance 的配置值只是上限。

## 23.9 `startup_protocol.hh`：启动握手协议

`startup_protocol.hh` 全文只有 17 行：

```cpp
namespace storage_startup {

inline constexpr u32 kMagic = 0x44565354;  // DVST

struct Request {
  u32 magic{kMagic};
};

struct Response {
  bool ready{};
};

}  // namespace storage_startup
```

`kMagic = 0x44565354` 是 ASCII "DVST"。`Request` 只有一个字段——4 字节魔数。`Response` 只有一个 bool——`ready`。

这个协议用在 23.3.7 提到的 `wait_for_start_signal`：

- 计算节点启动完毕后，通过 `initiator_qp` 发一个 `Request{magic = 0x44565354}`。
- 存储节点收到后，校验 `request.magic == kMagic`，回 `Response{ready = true}`。
- 存储节点在收到这个信号之前不会继续启动后续运行时（peer RDMA、插入 worker 等）。

这个 barrier 的意义是确保所有计算节点都已经注册好 MR、建好 QP、准备好接收 token 之后，存储节点才开始做"会触发计算节点动作"的事（比如发 access token、建 DetachedQP）。`cm_.synchronize()` 是更早的 barrier（建 DetachedQP 之前），`wait_for_start_signal` 是更晚的 barrier（建完 DetachedQP 之后、启动运行时之前）。

注意 magic 选 "DVST"（项目名 dvstor 的前 4 字符）是一个有意的、易识别的值——任何把别的协议数据误投到 initiator_qp 的 bug 都会被这个 magic 校验拦下，而不是导致存储节点莫名其妙地启动一半。

## 23.10 与其他模块的关系

本课是存储节点子系统的"外壳"课，把所有跨分片、跨进程的传输与启动逻辑收拢在 `MemoryNode` 这一个类里。后续四课都是在这个外壳内部展开：

- **第 24 课 peer RPC**：基于本课 23.4 的 peer RDMA 骨架（QP0 control + scratch buffer + 完成回调），定义 `service::storage_owner::PeerRpcHeader`、`ReverseUpdateOp` 等协议结构，实现 `send_peer_op_batch_async` / `wait_for_peer_reverse_update_response` 等 RPC 编排。`MemoryNode` 类里 `memory_node.hh:222-337` 那一大段 `peer_rpc_*` 接口就是第 24 课的内容。
- **第 25 课 索引访问/图修改**：基于本课的 `index_buffer_` 内存布局与 `local_node_ptr` / `load_local_node_header_acquire` 等本地图访问辅助，实现 `write_new_node` / `write_neighbor_list` / `apply_local_reverse_update` / `beam_search_candidates_async` 等。`StorageOwnerCoroutineScratch`（23.7.2）是这一课的工作内存。
- **第 26 课 维护/wire protocol**：基于本课的 `StorageOwnerMaintenanceTask`（23.2）与 `storage_owner_maintenance_*` 字段，实现 stitch_insert / cleanup_deleted 两类维护工作流；同时定义 `service_storage_runtime` 主循环的 wire protocol（怎么从计算节点收 insert 请求、怎么回响应）。
- **第 16 课 存储回收 RCU（计算侧）**：与 23.5.3 / 23.6 严格对应。计算侧写 `reclaim_ack_sequences[i]`，本存储侧用 `minimum_compute_reclaim_ack()` 读最小值；计算侧维护"上一个稳定快照"，本存储侧用 `StorageReclaimQueue` 维护"哪些 retire 节点可以复用"。两者通过 `StorageControlBlock::reclaim_ack_sequences[64]` 与 `durable_maintenance_sequence` 闭合。
- **第 22 课 GPUNetIO**：本课是"被读方"。计算节点 GPU 通过 GPUNetIO 直接 RDMA read `index_buffer_` 的静态节点区、PQ 码流区、控制页（含路由发布页）。本课 23.3.6 的 `index_region_.register_memory(..., true)` 与 token 分发是 GPUNetIO 能工作的前提。
- **第 7 课 schema-15 索引格式**：本课 23.3.3 的全部启动校验都是在检查 schema-15 的 metadata 字段。`StorageControlBlock` / `StorageRoutePublication` / `CodeHeader` 都来自第 7 课。
- **第 8 课 元数据/owner map/存储协议**：`service::index_metadata::Metadata` 的加载与 `service::storage_owner` 协议常量是本课的入口契约。
- **第 5 课 RDMA 传输库（上）**：本课大量使用 `Context`、`ServerConnectionManager`、`DetachedQP`、`MemoryRegion`、`MemoryRegionToken`、`LocalMemoryRegion`、`HugePage` 等基础组件。

## 23.11 小结

本课讲解了 dvstor 存储节点的主体类 `MemoryNode` 与 peer RDMA 传输骨架。关键点：

1. **`MemoryNode` 是一个 PImpl 风格外壳**，把存储节点进程的全部状态收拢在一个类里：主缓冲区、两条 RDMA 网络（compute↔memory 与 memory↔memory）、六类后台线程、storage-owner 路由表与 freshness 表。

2. **启动是一条严格校验的线性流水线**：连接计算节点 → 收 Parameters → 加载 metadata 并逐字段校验（schema-15、分片数、R、dtype、PQ checksum、动态布局）→ 分配大页 → 加载 `.dat` 与 `.pq32.codes` 到指定偏移 → 初始化 4 KiB 控制页（`StorageControlBlock` + offset 1024 的 `StorageRoutePublication`）→ 注册 MR → 分发 access token → 建 DetachedQP → 启动 barrier → 启动各运行时 → 进入主循环。

3. **`load_index_file` 把三件磁盘产物拼到 `index_buffer_` 的指定偏移**：`.dat` 从 0 开始、PQ 码流紧跟控制页之后、动态区在 PQ 码流之后。控制页 offset 1024 的 `StorageRoutePublication` 用"奇偶双段写序列号 + body_checksum"实现无锁的无撕裂发布。

4. **peer RDMA 是存储节点之间的第二条网络**，用独立 `Context`（独立 PD/CQ）。全连接采用"低 id 主动连、高 id 被动等"模式，把 `(storage_id, qp_idx)` 编码进 connect 的 user data 实现自动对齐。QP0 留给 RPC，QP1..N 给数据 RDMA。

5. **三层信用控制**：per-QP（≤8）、per-peer（≤peer_qps_per_peer_ × 8）、全局（≤per-peer × (num_storage_nodes_ - 1)）。完成回调在 `handle_peer_send_completion` 里归还信用、减 `post_balances`、唤醒同步等待。

6. **三个同步原语** `remote_read_bytes` / `remote_write_bytes` / `remote_compare_and_swap` 用同样的"信用获取 + wr_id 编码 + post_send + wait_completion"模式。`try_lock_remote_header` 在 CAS 之上实现跨分片节点头加锁，是 stitch 与反向边写入的基础。

7. **`StorageReclaimQueue` 与第 16 课计算侧 reclaim ACK 闭合**：retire 节点按 maintenance_sequence 入队，acquire 时取 `min(durable_sequence, acknowledged_sequence)` 作为安全序号，序号 ≤ 安全序号的节点才能复用。

8. **`StorageOwnerThread` 是多协程共享的执行上下文**：每协程一个 `post_balance`、一份 `StorageOwnerCoroutineScratch`、一段独立 scratch stride。`current_storage_owner_thread_` thread_local 指针让任意代码找到当前上下文。

9. **`derive_storage_owner_cpu_plan` 把 CPU 分成 6 类**：progress（peer 3 个 + foreground 1 个）、foreground workers、maintenance workers、peer_reverse workers、peer_search workers。search 拿剩余预算因为每个 finalize insert 都要在每个远端分片跑一次完整 search。

10. **`storage_startup::kMagic = "DVST"`** 是启动握手的全部——4 字节魔数 + bool 回执，确保所有计算节点准备好之后存储节点才启动后续运行时。

至此我们完成了存储节点"外壳与传输"层的讲解。下一课（第 24 课）进入 `peer_rpc/` 目录，讲解在 23.4 的骨架之上构建的 peer RPC 协议——反向边更新请求/响应、stitch search 请求/响应、cleanup deleted 请求的编排与去重。
