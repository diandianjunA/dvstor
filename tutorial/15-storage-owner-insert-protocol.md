# 第 15 课：Storage-owner 插入协议

## 本课目标

本课讲 `insert_execution=storage_owner` 相关的 compute side 到 memory node 的 RPC 协议。你需要理解 batch、slot、request/response buffer、SEND/RECV、mutation kind、anchor hints、timeout 和 completion loop。

## 代码证据

必须阅读：

- `src/service/compute_service/storage_owner_insert.ipp`
- `src/service/storage_owner_protocol.hh`
- `src/service/storage_owner_client_helpers.hh`
- `src/memory_node/storage_owner_runtime.cc`
- `src/memory_node/storage_owner_state.hh`

## 为什么有 storage-owner 插入

普通 compute-side 插入通过 one-sided RDMA 直接修改远端图，需要 compute node 负责搜索、剪枝、写节点、反向边更新。storage-owner 模式把插入执行移动到 owner memory node：

```text
compute node
  只负责路由、编码 request、发 SEND、收 response

memory node owner
  解码 request
  本地执行插入/upsert/erase
  必要时 peer RDMA/RPC
  返回结果
```

这样可以减少 compute-side 直接修改远端图的复杂度，也让 storage node 掌握 freshness/idmap。

## 协议结构

`storage_owner_protocol.hh` 定义：

- `kInsertMagic`
- `kMutationMagic`
- `kPeerRpcMagic`

mutation kind：

- `insert`
- `upsert`
- `erase`

mutation status：

- `ok`
- `not_found`
- `already_exists`
- `already_deleted`
- `failed`

request header：

- `InsertBatchRequestHeader`
- `MutationBatchRequestHeader`

response header：

- `InsertBatchResponseHeader`

response payload 包含：

- per-item status
- per-item mutation result
- breakdown counters
- invalidated neighbor raws

## compute side runtime 初始化

`start_storage_insert_runtime`：

1. owner count = memory node 数。
2. rpc depth = `storage_owner_rpc_depth`。
3. 根据 batch max、dim、anchor hint count 计算 request/response bytes。
4. 为每个 owner 创建 `StorageOwnerSenderState`。
5. 每个 owner 有多个 `StorageOwnerRpcSlot`。
6. 每个 owner 有多个 response receive slot。
7. 预先 post response receives。
8. 启动 completion loop 线程。
9. 每个 owner 启动一个 sender 线程。

`StorageOwnerRpcSlot` 保存：

- request buffer/region
- response buffer/region
- tasks
- samples
- send/response done flags
- batch id 到 slot 映射
- timing fields

## insert API 路由

storage-owner insert 中：

```text
for each InsertItem:
  创建 StorageInsertTask
  route_storage_owner_update(item)
  task.anchor_hints = route.hints
  owner_storage = route.owner
  push 到 storage_insert_owners_[owner].queue
  notify owner sender
```

如果没有 anchor index，owner 通常按 id hash 或 override 决定。local-stitch 模式会使用 anchor index 产生 hints。

## sender 线程

每个 owner 一个 sender 线程：

```text
run_storage_insert_sender(owner)
  等待 queue 非空且 free_slots 非空
  可等待 storage_owner_batch_wait_us 聚合更多任务
  取一个 free slot
  从 queue 取最多 storage_owner_batch_max 个 task
  post_storage_owner_batch
```

这形成 per-owner 微批。

## request 编码

`post_storage_owner_batch`：

1. 判断是否 mutation request。
2. 计算 request size 和 response size。
3. 填 header：
   - magic
   - dim
   - owner_storage
   - source_client
   - item_count
   - vector_dtype
   - vector_bytes
   - anchor_hint_count
   - batch_id
4. 写 ids。
5. 写 mutation kinds。
6. 用 `encode_float_vector_to_storage` 写 vectors。
7. 写 anchor hints raw address。
8. 记录 slot 状态。
9. `qp.post_send_with_id(... IBV_WR_SEND ...)`。

注意：这是 SEND，不是 RDMA WRITE。memory node 必须提前 post receive。

## completion loop

compute side `run_storage_insert_completion_loop` 同时 poll：

- send CQ
- receive CQ

send completion：

```text
handle_storage_owner_send_completion(owner, slot)
  slot.send_done = true
  maybe_release_storage_owner_slot_locked
```

response receive：

```text
handle_storage_owner_response(owner, response_slot)
  根据 response batch_id 找原 request slot
  拷贝 response 到 slot.response_buffer
  slot.response_done = true
  maybe_release_storage_owner_slot_locked
  重新 post receive
```

slot 只有 send_done 和 response_done 都 true 才释放。

## memory node 接收与处理

memory node `service_storage_runtime` 会接收 insert request，形成 `StorageOwnerInsertTask`。worker 中：

```text
storage_owner_insert_worker_loop
  从 storage_insert_tasks_ 取一批 task
  process_storage_owner_insert_tasks
```

`process_storage_owner_insert_tasks`：

1. 解码 request header。
2. 收集 batch ids、kinds、vectors、anchor hints。
3. decode storage vector 到 float。
4. 调 `execute_storage_owner_batch_items_async` 或同步版本。
5. 组装 response。
6. SEND response 给 source client。

## response 处理和本地状态更新

compute side 收到 response 后，`maybe_release_storage_owner_slot_locked` 会：

- 读取 statuses。
- 读取 mutation results。
- 对每个 task 设置 promise。
- 合并 breakdown。
- 释放 slot 到 free_slots。
- `storage_insert_inflight_--`。

上层 `ComputeService::insert` 等待 futures，统计成功数量和超时。

## 性能影响

关键参数：

- `storage_owner_batch_max`
- `storage_owner_batch_wait_us`
- `storage_owner_rpc_depth`
- `storage_owner_rpc_timeout_ms`
- `storage_owner_anchor_hints`

可观测指标：

- sender queue wait
- batch wait
- request prepare
- RDMA SEND
- response wait unaccounted
- storage-owner queue wait
- storage-owner search/prune/write/reverse

权衡：

- batch 大提升吞吐，但增加排队等待。
- rpc depth 大提高并发，但占用 CQ 和 receive slots。
- timeout 太短会误判慢请求，太长会掩盖死锁。

## 设计异味

1. compute side storage-owner 逻辑在 `ComputeService` 内部，类膨胀明显。
2. request/response payload 通过手写 offset 函数解析，缺少版本协商。
3. SEND/RECV slot 管理复杂，容易出现 unmatched response。
4. 每个 owner 一个 sender 线程，owner 多时线程数增加。
5. batch wait 是固定时间策略，不能根据负载动态调整。

## 可验证问题

- storage-owner request 是 RDMA WRITE 还是 SEND？
- `batch_id` 的作用是什么？
- response 如何找到原 slot？
- insert 和 upsert/erase 的 request header 有什么区别？
- `storage_owner_batch_wait_us` 增大会怎样影响延迟和吞吐？

## 学习任务

1. 画出 storage-owner insert 的 compute side 时序图。
2. 手算一个 batch 中 request buffer 的布局。
3. 跟踪 response 从 memory node SEND 到 future.set_value 的路径。
4. 找出所有 unmatched response 的日志路径。
5. 思考：如果要支持协议版本升级，应在 header 中增加哪些字段？

