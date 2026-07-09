# 第 17 课：MemoryNode peer RDMA/RPC 与跨 shard 维护

## 本课目标

本课讲 storage-owner 模式下 memory node 之间如何通信。重点是 peer RDMA 用于远程读写/CAS，peer RPC 用于 reverse update、cleanup deleted、stitch search。这是理解多 shard 插入一致性和性能的关键。

## 代码证据

必须阅读：

- `src/memory_node/peer_rdma.cc`
- `src/memory_node/peer_rpc.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/memory_node/storage_owner_runtime.cc`
- `src/service/storage_owner_protocol.hh`

## peer RDMA 建连

`setup_storage_peers` 只在 storage-owner 模式且 storage node 数大于 1 时执行。

主线：

1. 要求 `config.storage_peers.size() == num_storage_nodes_`。
2. 每个 memory node 根据 `storage_id_` 决定主动连低编号 peer 或等待高编号 peer。
3. `peer_qps_per_peer_ = min(MAX_QPS, num_compute_threads_)`。
4. 每个 peer 建多个 QP。
5. 注册本地 `index_buffer_` 为 peer remote accessible MR。
6. 和 peer 交换 `MemoryRegionToken`。
7. 分配 peer scratch buffer。
8. 初始化 peer RPC runtime。

约定：

- QP0 是 control/RPC lane。
- 如果 QP 数大于 1，数据 RDMA 使用 QP1..。

## peer RDMA credit

peer read credit 有三层：

- per peer total: `peer_rdma_read_credit_limit`
- per peer per QP: `peer_rdma_read_credit_limit_per_qp`
- global async: `peer_rdma_read_global_credit_limit`

`try_acquire_peer_rdma_read_credit` 同时增加 peer total 和 per-QP counter。completion 时在 `handle_peer_send_completion` 中释放。

这避免 storage-owner peer RDMA read 打爆 peer QP 的 RDMA read atomic 能力。

## sync 和 async peer RDMA

同步 read/write/CAS：

- `remote_read_bytes`
- `remote_write_bytes`
- `remote_compare_and_swap`

它们会 post WR，然后等待 sync completion。

异步 read：

- `post_peer_read_async`

它绑定 `StorageOwnerThread` 和 coroutine id：

```text
thread.track_post()
register PeerPendingSend
post RDMA READ
completion -> thread.post_balances[coroutine_id]--
```

这让 storage-owner coroutine 能 overlap peer RDMA。

## peer RPC runtime

`setup_peer_rpc_runtime`：

- 计算最大 reverse update request size。
- 计算 stitch search request/response size。
- 统一取最大 message bytes。
- 分配 receive region、sync send region、async send region。
- 为每个 peer 和 receive slot post receive。

RPC message 使用 `PeerRpcHeader`：

- magic
- type
- source_shard
- item_count
- request_id
- status
- reserved

RPC 类型：

- `reverse_update_request`
- `reverse_update_response`
- `cleanup_deleted_request`
- `stitch_search_request`
- `stitch_search_response`

## reverse update

当某个 shard 插入新节点后，需要更新其他 shard 上节点的反向边，就会产生 `ReverseUpdateOp`：

```cpp
target_raw
candidate_raw
```

接收方：

```text
handle_peer_reverse_update_request
  创建 PeerReverseUpdateTask
  apply_peer_reverse_update_task
    grouped[target].push_back(candidate)
    apply_local_reverse_update(target, candidates)
  如果需要 response，发送 reverse_update_response
```

可配置：

- `storage_owner_reverse_mode`: async 或 sync。
- `storage_owner_reverse_queue_depth`
- `storage_owner_reverse_flush_us`
- `storage_owner_reverse_coalesce_max`

## cleanup deleted

cleanup deleted request 用于让其他 shard 清理指向 deleted node 的边。它同样通过 peer RPC 传递 `ReverseUpdateOp` 格式，但语义不同。

读这条路径时，要区分：

- reverse update 是添加或修正反向边。
- cleanup deleted 是移除或修正无效边。

## stitch search

local-stitch 可能需要远端 shard 帮忙搜索候选。peer RPC 提供：

- `stitch_search_request`
- `stitch_search_response`

这条路径会把目标节点 snapshot 或搜索请求发给目标 shard，让目标 shard 在本地执行部分候选搜索，再返回候选 `RemotePtr`。

## peer progress threads

`start_peer_reverse_update_runtime` 启动：

- `peer_rpc_progress_thread_`
- `peer_reverse_response_thread_`
- `peer_reverse_outgoing_thread_`
- reverse update workers
- stitch search workers

这些线程和 storage-owner insert workers 并行运行，共享 peer context、queues 和 scratch。

## 性能影响

- QP0 作为 control lane，可以避免大数据 RDMA 挤压 RPC。
- peer RDMA read credit 限制保护 QP，但 credit wait 会增加 insert latency。
- async reverse update 提高 foreground 吞吐，但延迟一致性更复杂。
- sync reverse update 更强，但会阻塞 foreground。
- coalescing 可以减少 RPC 数，但增加 flush wait。
- stitch search remote expansions 会放大跨 shard RDMA/RPC。

## 设计异味

1. peer RDMA 和 peer RPC 都实现在 `MemoryNode` 方法中，类职责过大。
2. control/data QP 约定靠索引 0 和注释，缺少类型封装。
3. peer RPC wire format 手写，版本和兼容性不足。
4. async reverse update 的最终一致性依赖后台队列，不易测试。
5. peer credit、async outstanding、worker scratch 多套状态交织，调试难度高。

## 可验证问题

- peer 之间 QP0 用来做什么？
- `post_peer_read_async` completion 后减少哪个计数？
- reverse update request 的 payload 是什么？
- async 和 sync reverse mode 差异在哪里？
- stitch search 为什么需要 peer RPC？

## 学习任务

1. 画出两个 memory node 之间 peer QP 建连图。
2. 跟踪一次 remote node snapshot 读取的 peer RDMA 路径。
3. 画出 reverse update request 和 response 的时序。
4. 搜索所有 `PeerRpcType` 分支，列出每个 handler。
5. 思考：如果要独立测试 peer RPC，应该先抽出哪些纯协议编码函数？

