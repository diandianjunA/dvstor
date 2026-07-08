# 第 12 课：MemoryNode 基础架构与远程内存布局

## 本课目标

本课讲 memory node 的真实职责。它不是简单内存服务器，而是集成了 RDMA region、索引 shard、命令协议、storage-owner 插入、peer RDMA/RPC 和维护线程的复杂服务。

## 代码证据

必须阅读：

- `src/memory_node/memory_node.hh`
- `src/memory_node/memory_node.cc`
- `src/memory_node/command_protocol.hh`
- `src/memory_node/storage_owner_state.hh`
- `src/vamana/storage_layout_resolver.hh`

## MemoryNode 构造函数主线

`MemoryNode::MemoryNode(Configuration& config)` 做了大量工作：

```text
初始化 RDMA context 和 connection manager
连接 compute clients
接收 configuration::Parameters
读取 metadata 并配置 VamanaNode 静态布局
allocate_memory
初始化 free pointer
可选加载 server_index_file
可选加载 owner idmap
可选加载 anchor index
注册 index memory region
发送 MemoryRegionToken 给 compute nodes
为 compute thread 连接 DetachedQP
同步
处理启动 command
根据 storage-owner 模式进入不同服务循环
```

这说明 `MemoryNode` 既是启动器，又是服务对象，又是协议处理器。

## index region 布局

memory node 的核心内存是 `index_buffer_`：

```text
offset 0: free pointer, u64
offset 8: global medoid pointer, u64
offset 16..: node records
```

构造函数中初始化：

```cpp
*reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;
```

离线 shard writer 也按同样约定写：

- shard 文件 offset 0 写 shard size。
- shard 0 offset 8 写 medoid pointer。
- nodes 从 offset 16 开始。

注意：运行时 free pointer 在加载 shard 后应该来自 shard 文件 offset 0 的内容。

## memory node 的 RDMA token

memory node 注册整块 index buffer：

```text
index_region_.register_memory(index_buffer, size, remote_access=true)
token = index_region_.createToken()
```

然后向所有 compute node 发送 `MemoryRegionToken`。compute side 后续 RDMA READ/WRITE/CAS/FAA 都使用这个 token 的 address 和 rkey。

## compute QP 连接

memory node 在接收 `configuration::Parameters` 后知道：

- `num_compute_threads_`
- `qp_pool_size_`

于是为每个 client、每个 thread/shared-context、每个 QP pool lane 建立 `DetachedQP`。

```text
qps_per_node = min(num_compute_threads_, MAX_QPS) * qp_pool_size_
for each client_qp:
  for thread_id in qps_per_node:
    DetachedQP connect
```

这配合 compute side `SharedContext` 中的 `qps[memory_node][pool_index]`。

## 命令协议

`src/memory_node/command_protocol.hh` 定义：

```cpp
enum Command : u32 { NOOP = 0, LOAD = 1, STORE = 2, SHUTDOWN = 3 };
```

请求和响应：

- `Request { command, path_length }`
- `Response { success, message_length }`

compute side `send_index_command` 发送 LOAD/STORE/SHUTDOWN。memory node `handle_command` 执行本地 `load_index_file` 或 `store_index_file`。

## storage-owner 相关状态

`storage_owner_state.hh` 定义了 memory node 内部状态结构：

- `BeamEntry`
- `NodeSnapshot`
- `InsertRuntimeState`
- `PeerRpcRuntimeState`
- `StorageOwnerInsertTask`
- `StorageOwnerThread`
- `StorageOwnerInsertJob`
- `FreshnessEntry`

特别是 `FreshnessEntry`：

```cpp
struct FreshnessEntry {
  RemotePtr current;
  u32 generation;
  bool deleted;
};
```

它是 storage-owner idmap 的运行时状态，用于 upsert/delete 判断和 generation 更新。

## load metadata

memory node 启动时如果发现 `<index_prefix>.meta.json`：

1. 校验 `dim`、`R`、`num_memory_nodes`。
2. 校验 vector dtype。
3. 设置 `VamanaNode::STORAGE_FORMAT`。
4. `VamanaNode::init_static_storage`。
5. 如果 metadata node layout 是 RaBitQ，启用 RaBitQ 并设置 centroid。
6. 校验 node size、vector offset、neighbor offset、hot graph metadata。
7. 记录是否需要 owner idmap。

compute node 和 memory node 都会做类似校验，这保证双方对远端内存布局的理解一致。

## storage-owner 模式和普通模式

启动 command 之后：

如果 `use_storage_owner_insert_`：

```text
setup_storage_peers
setup_insert_runtime
start_peer_reverse_update_runtime
start_storage_owner_maintenance_runtime
start_storage_owner_insert_workers
service_storage_runtime
```

否则：

```text
while running:
  running = handle_command()
```

这表示非 storage-owner 模式下 memory node 主要被 compute-side one-sided RDMA 操作访问；storage-owner 模式下 memory node 还会处理 SEND/RECV RPC 和 peer RPC。

## 性能影响

- index region 是否使用 hugepage 会影响 TLB 和内存访问。
- metadata 校验决定 hot graph 是否启用，进而影响 neighbor RDMA bytes。
- QP 数量由 compute thread 和 qp pool size 决定，影响 vector RDMA 并行。
- storage-owner 模式下 memory node CPU 成为插入路径瓶颈之一。
- peer RDMA/RPC 会额外占用 memory node RDMA resources。

## 设计异味

1. `MemoryNode` 构造函数过长，包含启动流程和服务循环。
2. 内存布局 offset 0/8/16 是硬编码协议，缺少统一 header struct。
3. metadata 校验逻辑在 compute node 和 memory node 重复。
4. storage-owner 模式和普通 command 模式在构造函数中分支，职责不清。
5. memory node 既处理 one-sided RDMA 暴露，又处理 SEND/RECV RPC 和 peer RPC，模块边界复杂。

## 可验证问题

- free pointer 和 medoid pointer 分别在哪个 offset？
- memory node 什么时候发送 MemoryRegionToken？
- `server_index_file` 和 `load_index` 有什么区别？
- non storage-owner 模式下 memory node 是否启动 insert worker？
- owner idmap 什么时候必须加载？

## 学习任务

1. 画出 memory node index region 的字节布局。
2. 跟踪 `MemoryNode` 启动时 metadata 校验的每一步。
3. 画出 compute node 和 memory node QP 连接数量计算。
4. 找到 `handle_command`，解释 LOAD/STORE/SHUTDOWN 的处理。
5. 思考：如果要重构 `MemoryNode`，构造函数应拆成哪些阶段？

