# 第 21 课：ComputeService 架构与生命周期

## 本课目标

学完本课后，你需要能从代码层面回答三个问题：

1. `ComputeService` 为什么是 compute node 的总装配点。
2. 构造、启动、暂停、恢复、停止分别会触碰哪些资源。
3. 后续做性能优化或大规模重构时，哪些状态不能随意移动。

本课只基于这些代码：

- `src/service/compute_service.hh`
- `src/service/compute_service/lifecycle.ipp`
- `src/service/compute_service/index_commands.ipp`
- `src/service/index_metadata.hh`
- `src/service/index_metadata.cc`

## 1. ComputeService 的定位

`ComputeService<Distance>` 是 compute node 的主对象。`src/main.cc` 选择 `L2Distance` 或 `IPDistance` 后，会实例化它；之后查询、插入、load/store、routing、breakdown 都经过它暴露的 public API。

从 `compute_service.hh` 可以把字段分成九类：

1. 配置和连接：
   - `Configuration config_`
   - `Context context_`
   - `ClientConnectionManager cm_`
   - `num_servers_`
   - `shutdown_remote_on_stop_`

2. 远端内存访问：
   - `MemoryRegionTokens remote_access_tokens_`
   - `init_remote_tokens()`
   - `receive_remote_access_tokens()`

3. worker 与 coroutine 调度：
   - `std::unique_ptr<WorkerPool> worker_pool_`
   - `ServiceProfile service_profile_`
   - `InsertQueue insert_queue_`
   - `QueryQueue query_queue_`
   - `vec<std::thread> workers_`
   - `workers_paused_`
   - `workers_idle_count_`
   - `shutdown_`
   - `stopped_`

4. Vamana 算法实例：
   - `std::unique_ptr<vamana::Vamana<Distance>> vamana_`
   - `std::unique_ptr<vamana::rabitq::Cache> rabitq_cache_`
   - `std::unique_ptr<vamana::anchor::Index> anchor_index_`

5. RPC routing：
   - `rpc_buffer_`
   - `rpc_region_`
   - `rpc_freelist_`
   - `outbound_rpc_queue_`
   - `pending_queries_`
   - `pending_registration_acks_`
   - `routing_centroids_`
   - `routing_inflight_`
   - `registered_remote_clients_`

6. storage-owner 写入路径：
   - `storage_insert_completion_thread_`
   - `storage_insert_shutdown_`
   - `storage_insert_owners_`
   - `StorageOwnerSenderState`
   - `StorageOwnerRpcSlot`
   - `StorageOwnerResponseSlot`

7. compute-side id map：
   - `compute_side_idmap_`
   - `initialize_compute_side_idmap()`
   - `lookup_compute_side_id()`
   - `publish_compute_side_id()`
   - `mark_remote_deleted()`

8. index command 与元数据：
   - `mn_command_mutex_`
   - `load_index()`
   - `store_index()`
   - `validate_index_metadata()`
   - `send_index_command()`

9. breakdown：
   - `breakdown_mutex_`
   - `breakdown_enabled_`
   - `completed_query_samples_`
   - `completed_insert_samples_`

这说明 `ComputeService` 不是单纯的“服务入口”。它同时拥有网络连接、RDMA token、GPU 初始化、算法对象、线程池、队列、RPC buffer、metadata 校验和性能采样。后续如果要重构，第一步不是拆文件，而是先识别这些状态的生命周期边界。

## 2. 构造函数调用链

`src/service/compute_service/lifecycle.ipp` 中构造函数的大致顺序如下：

1. 保存配置并创建 RDMA context、connection manager。
2. `init_remote_tokens()` 为每个 memory node 创建 `MemoryRegionToken` 容器。
3. `cm_.connect()` 建立 compute node 与 memory node、其他 compute node 的连接。
4. 如果不是 `disable_thread_pinning`，主线程绑定到 core。
5. initiator 向 memory node 发送 `configuration::Parameters`。
6. `receive_remote_access_tokens()` 从 memory node 接收远端 memory region token。
7. 初始化 `VamanaNode` 静态布局为默认状态：
   - `disable_rabitq()`
   - `disable_hot_graph()`
   - `set_storage_format(aos_v1)`
8. 如果配置了 `load_index` 且存在 metadata：
   - 读取 `.meta.json`
   - 校验 vector dtype
   - 解析 storage format
   - `VamanaNode::init_static_storage(...)`
   - 如 metadata 是 RaBitQ 布局，则启用 RaBitQ 并设置 centroid
   - 如 compact layout，则配置 hot graph offset、entry count、dynamic base 等
9. `gpu::gpu_init(...)` 初始化 GPU。
10. `resolve_service_profile()` 决定 insert/query worker 数和 coroutine 数。
11. 构造 `vamana::Vamana<Distance>`。
12. 将搜索和插入相关配置写入 Vamana：
    - expansion batch
    - credit-aware expansion
    - breakdown device utilization
    - query batch size
    - RaBitQ gate/runtime
13. 如果启用 RaBitQ 且加载 index：
    - 创建 `vamana::rabitq::Cache`
    - 加载 sidecar
    - 校验 cache ratio
    - `vamana_->set_rabitq_cache(...)`
14. 如果 storage-owner insert 使用 `local_stitch`：
    - 加载 anchor sidecar
    - 失败时直接拒绝启动
15. 创建 `WorkerPool`，分配 compute threads、shared contexts、QP pool、buffer allocator。
16. 为每个 compute thread 初始化 GPU buffers。
17. `cm_.synchronize()` 做连接后的同步。
18. 如有必要加载 compute-side idmap。
19. `wait_for_load_or_store()` 处理启动时 load/store 命令。
20. `synchronize_clients_after_startup()` 同步 compute clients。
21. 初始化 routing centroid/inflight 数组。
22. `start_workers()`
23. `start_rpc()`
24. 如果启用 storage-owner insert，则 `start_storage_insert_runtime()`
25. `refresh_routing_state(true)`

这条链路有一个重要结论：`ComputeService` 构造函数已经完成了大多数资源的创建和启动。它不是轻量对象，不能随意在测试中频繁构造，也不能把它当作普通的无状态 facade。

## 3. VamanaNode 静态全局状态的风险

构造函数和 `validate_index_metadata()` 都会调用 `VamanaNode` 的静态方法：

- `set_storage_format`
- `init_static_storage`
- `enable_rabitq`
- `disable_rabitq`
- `set_rabitq_centroid`
- `configure_hot_graph`
- `disable_hot_graph`

这意味着当前系统将 index layout 作为进程级静态状态，而不是作为 `Vamana` 或 `ComputeService` 的实例状态。它带来几个后果：

1. 同一进程内同时服务两个不同 layout 的 index 很困难。
2. load index 时必须暂停 worker 和 RPC，避免运行中的 coroutine 使用旧 layout。
3. 单元测试如果在同一进程中按不同参数初始化 `VamanaNode`，可能互相污染。
4. metadata 校验不是局部逻辑，它会改变全局 layout。

后续重构如果想把 layout 状态实例化，必须同步修改：

- `VamanaNode`
- RDMA read/write wrapper
- offline shard writer
- memory node load
- compute node metadata validation
- hot graph encode/decode
- RaBitQ cache sidecar

这类重构属于高风险阶段，不能作为第一步。

## 4. worker 生命周期

`start_workers()` 负责启动 service worker。它从 `service_profile_` 读取：

- `insert_workers`
- `query_workers`
- `insert_coroutines`
- `query_coroutines`

如果启用 storage-owner insert：

- `resolve_service_profile()` 会把 `insert_workers` 设为 0。
- 所有 service worker 都变成 query worker。
- insert/upsert/delete 由 storage-owner RPC 路径处理。

如果没有启用 storage-owner insert：

- 默认将线程拆成 insert/query 两部分。
- 如果用户配置了 `insert_workers` 或 `query_workers`，则按配置推导另一部分。
- 要求 insert 和 query worker 都至少有一个。

`start_workers()` 中每个线程都会再次调用 `gpu::gpu_init(config_.gpu_device)`。这说明 GPU context 不只是主线程初始化，worker 线程也需要绑定到 GPU 设备。

insert worker 调用：

```cpp
service::vamana_service_schedule_inserts(...)
```

query worker 调用：

```cpp
service::vamana_service_schedule_queries(...)
```

这两个 scheduler 会：

- 轮询请求队列。
- 维护每个 coroutine slot 的活跃请求。
- 调用 `thread->poll_cq()`。
- 调用 `thread->poll_gpu_events()`。
- 在 coroutine ready 时 resume。
- 在 coroutine done 时 set promise。

因此，worker 线程实际承担三类职责：

1. service queue 调度。
2. RDMA completion progress。
3. GPU event progress。

这也是为什么单纯增加线程数不一定提升性能：每个线程既是执行者，也是 progress engine。

## 5. pause/resume 的语义

`pause_workers()` 逻辑是：

1. 设置 `workers_paused_ = true`。
2. 循环等待 `workers_idle_count_ >= config_.num_threads`。
3. 等待过程中只 `yield`。

scheduler 中只有在 `all_idle` 时才会响应 pause：

```cpp
if (paused.load(...)) {
  idle_count.fetch_add(...);
  while (paused.load(...)) {
    std::this_thread::yield();
  }
  idle_count.fetch_sub(...);
}
```

所以 pause 不是抢占式暂停。它要求 worker 当前没有活跃 coroutine，或者活跃 coroutine 最终能走到完成状态。如果某个 coroutine 卡在 RDMA completion 或 GPU event 上，`pause_workers()` 会一直等。

`pause_rpc()` 类似：

- 设置 `rpc_paused_ = true`。
- 等待 `rpc_idle_`。
- RPC loop 在看到 pause 后设置 idle 并 yield。

load/store 会先暂停 worker 和 RPC：

- `load_index()`：`pause_workers()` -> `pause_rpc()` -> memory node LOAD -> validate metadata -> resume
- `store_index()`：`pause_workers()` -> `pause_rpc()` -> memory node STORE -> resume

这说明 load/store 被设计成全局静止点。优化这个路径时必须关注：

- 队列中已入队请求是否会长时间等待。
- routing pending query 是否会被暂停影响。
- storage-owner insert runtime 是否也需要 pause。
- metadata 更新是否和正在运行的 query/insert 隔离。

## 6. stop/destructor 的资源释放顺序

析构函数顺序是：

1. `stop_storage_insert_runtime()`
2. `stop_rpc()`
3. `stop_workers()`
4. `shutdown_remote_if_requested()`
5. 销毁每个 compute thread 的 GPU buffers
6. `gpu::gpu_shutdown()`

这个顺序说明：

- storage-owner insert runtime 在 worker 前停止。
- RPC 在 worker 前停止。
- worker 通过 `shutdown_` 退出。
- GPU buffer 在 worker join 后销毁。

如果改动这里的顺序，需要考虑：

1. worker 是否仍会访问 `gpu_buffers`。
2. RPC loop 是否仍会使用 `rpc_buffer_`。
3. storage-owner sender 是否仍在等待 response slot。
4. `shutdown_remote_if_requested()` 是否会依赖 still-alive server QPs。

`shutdown_remote_if_requested()` 只在以下条件满足时发送 remote shutdown：

- `shutdown_remote_on_stop_` 为 true。
- 当前 compute node 是 initiator。
- 没有使用 storage-owner insert。

这避免 storage-owner 模式下由 compute destructor 随意关闭 memory node。

## 7. load/store index 的协议

`send_index_command()` 在 `index_commands.ipp` 中实现。流程是：

1. 用 `mn_command_mutex_` 序列化 command。
2. 遍历所有 memory server QP。
3. 对每个 server 构造 shard path：
   - `index_path::shard_file(path, i + 1, num_memory_servers)`
4. 发送 `mn_command::Request`。
5. 如果 path 非空，再发送 path 字符串。
6. 再遍历所有 server 接收 `mn_command::Response`。
7. 如果 response 有 message，再额外 receive message payload。

它的正确性依赖 memory node 的命令处理路径按完全相同顺序 receive 请求、path、发送 response。

性能上，load/store 不是高频路径，但它暴露了设计上的耦合：

- compute node 知道 shard 文件命名规则。
- memory node command protocol 使用 SEND/RECV 串行完成。
- metadata validation 在 compute node 本地完成，但实际数据由 memory node 加载。
- 部分 memory node 成功、部分失败时，compute 只返回 false，没有自动 rollback。

## 8. validate_index_metadata 的检查点

`validate_index_metadata()` 首先判断 metadata 文件是否存在：

- 如果不存在：回退到 runtime config，初始化默认 AoS 布局，并按配置启用 RaBitQ。
- 如果存在：严格读取并校验。

关键校验包括：

- `vector_data_type` 与 runtime 配置是否匹配。
- `storage_format` 是否可解析。
- `schema_version` 是否等于 13。
- `dim` 是否等于 `config_.dim`。
- `R` 是否等于 `config_.R`。
- `vector_component_size` 是否等于 `VamanaNode::vector_component_size()`。
- `vector_bytes` 是否等于 `VamanaNode::vector_bytes()`。
- `node_size` 是否等于 `VamanaNode::total_size()`。
- `graph_hot_bytes`、`vector_offset`、`neighbors_offset`、`rabitq_offset` 是否匹配。
- `num_memory_nodes` 是否等于当前连接的 memory node 数。
- compact layout 下 hot graph offset/count/dynamic base 是否完整。
- `beam_width_construction` 如果存在，必须匹配当前配置。

这个函数是 index format 与 runtime 之间的主要防线。后续改 metadata 字段、layout 或 builder 输出时，必须同步更新它。

## 9. compute-side idmap

当不使用 storage-owner insert 时，`initialize_compute_side_idmap()` 会加载 owner idmap sidecar：

- 要求 `metadata.idmap_format == "owner_sharded_v1"`。
- 对每个 owner 读取 `index_path::owner_idmap_file(...)`。
- 校验 magic、version、owner shard、shard count。
- 将 `id -> {RemotePtr, deleted, owner}` 写入 `compute_side_idmap_`。

这张表用于：

- upsert 时找到旧节点。
- delete 时标记旧节点 deleted。
- 根据 id 找 owner。

如果 idmap 缺失，代码会清空 `compute_side_idmap_` 并返回 false。对于基础索引里的 id，这意味着 upsert/delete 能力会下降。课程后面做一致性测试时，需要专门覆盖 idmap 缺失、损坏、schema 不匹配的情况。

## 10. 性能影响

`ComputeService` 生命周期对性能的影响主要有：

1. 构造成本高：
   - RDMA 连接。
   - token 接收。
   - metadata load。
   - GPU init。
   - worker pool 初始化。
   - GPU buffer 分配。
   - RaBitQ cache load。
   - anchor sidecar load。

2. pause 是 busy-yield：
   - `pause_workers()` 和 `pause_rpc()` 都用 `yield` 等待。
   - 如果用于频繁 load/store，会浪费 CPU。

3. worker 同时负责 progress：
   - RDMA CQ 和 GPU event 的 poll 在 scheduler 内部。
   - 如果请求队列为空，线程仍可能频繁 yield。
   - 如果 coroutine 数配置过大，单线程扫描 slot 的开销会上升。

4. routing centroid 计算会读 medoid：
   - `compute_local_routing_centroid()` 使用 compute thread 0，读 medoid probe。
   - load 后 refresh routing 会依赖此路径。

5. RaBitQ cache 加载可能占启动时间：
   - `rabitq_cache_->load(...)` 会读取每个 shard sidecar。
   - 还会校验 cache 与 raw vector bytes 的 ratio。

## 11. 设计异味

从代码可以直接看到以下结构性问题：

1. `ComputeService` 过大：
   - 生命周期、routing、index command、idmap、storage-owner、breakdown 都在同一个模板类中。

2. 静态 layout 状态：
   - `VamanaNode` 静态字段让多 index、多配置测试和热切换变复杂。

3. 构造函数承担太多工作：
   - 难以局部测试。
   - 失败时资源清理路径复杂。
   - 很难模拟部分初始化状态。

4. pause/stop 缺少超时：
   - 如果 worker 不进入 idle，load/store 会无限等待。

5. metadata validation 有副作用：
   - 它既验证 metadata，也更新 `VamanaNode` 全局状态和 `config_.vector_data_type`。

6. service profile 与 storage-owner 模式耦合：
   - 是否启用 storage-owner insert 会改变 worker 类型分配。

## 12. 可验证问题

阅读源码后，你应该设计这些验证：

1. metadata 不存在时：
   - 是否回退到 runtime config。
   - 是否按 `config_.use_rabitq` 启用 RaBitQ。

2. metadata 存在但 `dim` 不匹配：
   - `load_index()` 是否返回 false。
   - worker/RPC 是否恢复。

3. compact metadata 缺少 hot graph arrays：
   - `validate_index_metadata()` 是否拒绝。

4. `pause_workers()` 在有长时间 RDMA 请求时：
   - 是否可能长时间等待。
   - CPU 使用率是否上升。

5. storage-owner insert 模式：
   - `resolve_service_profile()` 是否确实没有 insert worker。
   - insert 请求是否走 storage-owner runtime。

6. routing enabled 且非 initiator：
   - `refresh_routing_state(true)` 是否等待 ack。

## 13. 学习任务

1. 画一张 `ComputeService` 字段分组图，把本课第 1 节的九类字段放进去。
2. 手写构造函数时序图，标出哪些步骤会修改 `VamanaNode` 静态状态。
3. 从 `validate_index_metadata()` 提取一张 metadata 字段与 runtime 校验关系表。
4. 思考一个重构方案：如何把 metadata validation 拆成“纯校验”和“应用 layout”两步。
5. 思考一个优化方案：如何让 pause 从 busy-yield 变成 condition variable 或 event 驱动，但不改变现有行为。

