# 第 24 课 · 存储侧 peer RPC

> 范围：`src/memory_node/peer_rpc/` 目录下的五个文件——`runtime.cc`、`request_handlers.cc`、`workers.cc`、`client_requests.cc`、`async_response.hh`，以及配套的 `detail.hh`。它们一起构成了存储节点之间"控制面"的 RPC 层：在 schema-15 存算分离的图上，反向边（reverse update）、删除清理（cleanup deleted）和跨分片候选搜索（stitch search）必须由归属分片（owner shard）发往候选所在分片（peer shard）去落地，peer RPC 就是这条跨分片链路的全部实现。

---

## 24.1 本课目标与涉及文件

本课要回答四个问题：

1. **生命周期**：一个存储节点在启动时如何把 RDMA buffer 切成接收/同步发送/异步发送三段，如何为每个 peer 配置接收 slot 与发送 slot，如何拉起 worker 线程池与 CQ 轮询线程（`runtime.cc`）。
2. **入站处理**：当一个 peer RPC 报文从 RDMA recv CQ 落到 buffer 里，CQ 进度线程如何识别它、去重、分派到反向更新队列或 stitch 搜索队列，worker 线程如何批处理与合批（`request_handlers.cc` + `workers.cc`）。
3. **出站构造**：当本分片需要给某个 peer 发反向边/清理/stitch 请求时，调用方有哪几种 API（同步直发、异步 fan-out、异步 outbox、异步 stage2 post），它们如何申请发送 slot、构造 `PeerRpcHeader`、注册 future、等待或轮询响应（`client_requests.cc`）。
4. **异步响应登记表**：`PeerAsyncResponseRegistry` 与 `PeerRequestDeduplicator` 这两个定长开地址哈希表如何在没有堆分配的前提下完成"请求—响应"匹配、超时重发、接收 slot 占用转移，以及接收端对重复请求 ID 的去重与回放（`async_response.hh`）。

涉及文件一览（均为绝对路径）：

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/runtime.cc` | 402 | peer RPC 运行时的初始化、生命周期、buffer 布局、send slot 调度、同步发送原语 |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/request_handlers.cc` | 369 | 入站 RPC 的派发与各 handler：反向更新、cleanup、stitch 搜索，及响应入队 |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/workers.cc` | 388 | CQ 进度循环、reverse worker、stitch worker、响应发送线程、出站 outbox 线程 |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/client_requests.cc` | 571 | 客户端侧请求构造：同步直发、fan-out、async post、outbox 入队、响应消费 |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/async_response.hh` | 570 | `PeerAsyncResponseRegistry`、`PeerRequestDeduplicator` 两个核心数据结构 |
| `/home/xjs/experiment/dvstor/src/memory_node/peer_rpc/detail.hh` | 9 | 三个 `.cc` 共享的 include 入口（只引入 `memory_node.hh` + 原子工具） |

辅助但需要交叉理解的文件：

- `/home/xjs/experiment/dvstor/src/service/storage_owner_protocol.hh`：wire 层的 `PeerRpcHeader`、`PeerRpcType`、`ReverseUpdateOp`、`StitchSearchItem`、`StitchSearchCandidate`，以及所有 `*_bytes()` / 偏移访问器（第 8 课、第 26 课）。
- `/home/xjs/experiment/dvstor/src/memory_node/storage_owner_state.hh`：`PeerRpcRuntimeState`、`PeerPendingSend`、`PeerRpcMessage` 三个 POD 结构。
- `/home/xjs/experiment/dvstor/src/memory_node/peer_rdma.cc`：`poll_peer_send_cq`、`next_peer_sync_wr_id`、`next_peer_async_wr_id`、`register_peer_pending_send_locked`、`wait_peer_sync_completion`、`handle_peer_send_completion`——peer RPC 的发送完成依赖这一层（第 23 课）。
- `/home/xjs/experiment/dvstor/rdma-library/library/utils.hh:70`：`encode_64bit/decode_64bit`——把 `(peer_id, slot_id)` 压进一个 64 位 WR ID。
- `/home/xjs/experiment/dvstor/src/memory_node/storage_owner_maintenance/worker.cc`：stage2 维护协程调用 `post_stitch_search_request_async` / `post_peer_op_batch_async` / `try_consume_peer_rpc_response` 的实际现场（第 25 课）。
- `/home/xjs/experiment/dvstor/src/memory_node/storage_owner_runtime/wire_protocol.cc:465` 与 `batch_execution.cc:132`：在线插入/更新的 stage2 同步路径调用 `send_reverse_update_batch`（第 26 课）。

---

## 24.2 wire 层契约回顾

在进入 `peer_rpc/` 之前，先锁定 wire 格式（`src/service/storage_owner_protocol.hh`）：

```cpp
// storage_owner_protocol.hh:12-13
constexpr u32 kPeerRpcMagic = 0x53505250;  // "SPRP"
constexpr u32 kPeerRpcVersion = 3;
```

`PeerRpcHeader`（第 145–154 行）是所有 peer RPC 报文的固定头：

```cpp
struct PeerRpcHeader {
  u32 magic{kPeerRpcMagic};
  u32 version{kPeerRpcVersion};
  u32 type{};
  u32 source_shard{};
  u32 item_count{};
  u64 request_id{};
  u32 status{static_cast<u32>(InsertStatus::failed)};
  u32 reserved{};
};
```

`type` 取自 `enum class PeerRpcType`（第 35–42 行），共六种：`reverse_update_request/response`、`cleanup_deleted_request/response`、`stitch_search_request/response`。`status` 用 `InsertStatus` 的 `ok/failed/overloaded` 三态。`reserved` 在 stitch 请求/响应里复用为 `candidate_capacity`，在反向更新请求里复用为 flag 位——`memory_node.hh:173` 定义 `kPeerRpcFlagNoResponse = 1u`，置位表示"不要回 ACK"。

报文体紧跟在头后，三种 payload 的字节计算函数（第 298–333 行）：

- `reverse_update_request_bytes(item_count) = sizeof(PeerRpcHeader) + item_count * sizeof(ReverseUpdateOp)`，`ReverseUpdateOp` 是 16 字节 `{target_raw, candidate_raw}` 两个 `u64`。
- `stitch_search_request_bytes(item_count) = 头 + item_count*sizeof(StitchSearchItem)` 对齐到 8 字节，再加 `item_count * vector_bytes`。
- `stitch_search_response_bytes(item_count, candidate_capacity)` 含每项一个 `u32` count、`candidate_capacity` 个 `StitchSearchCandidate`，以及 `candidate_capacity * vector_bytes` 的向量数据。

这些 `*_bytes()` 函数在 `runtime.cc` 初始化时被用来确定单个 RPC 消息的最大长度，所有 peer 共用一个定长 `message_bytes`。

---

## 24.3 `runtime.cc`：运行时初始化与生命周期

### 24.3.1 `setup_peer_rpc_runtime`：buffer 布局与 slot 分配

`runtime.cc:4-100` 的 `setup_peer_rpc_runtime` 在存储节点启动期被调用一次，它的职责是把 peer RPC 所需的全部 RDMA 资源一次性配齐。函数开头是空操作守卫：

```cpp
// runtime.cc:4-7
void MemoryNode::setup_peer_rpc_runtime(const Configuration& config) {
  if (!peer_context_ || num_storage_nodes_ <= 1) {
    return;
  }
```

`peer_context_` 为空说明本节点没有 peer RDMA 上下文（第 23 课），`num_storage_nodes_ <= 1` 说明集群里只有一个存储节点，根本没有 peer，两种情况都直接返回。

接下来按"最大的可能报文"决定 `message_bytes`：

```cpp
// runtime.cc:9-27
const u64 max_reverse_update_ops =
  static_cast<u64>(config.R) * static_cast<u64>(config.storage_owner_batch_max);
lib_assert(max_reverse_update_ops <= std::numeric_limits<u32>::max(), ...);
const size_t reverse_update_bytes =
  service::storage_owner::reverse_update_request_bytes(static_cast<u32>(max_reverse_update_ops));
const size_t stitch_request_bytes =
  service::storage_owner::stitch_search_request_bytes(config.storage_owner_batch_max);
const size_t stitch_response_bytes =
  service::storage_owner::stitch_search_response_bytes(
    config.storage_owner_batch_max,
    config.resolved_storage_owner_construction_width());
peer_rpc_runtime_.message_bytes = align_up(
  std::max({reverse_update_bytes,
            service::storage_owner::reverse_update_response_bytes(),
            stitch_request_bytes,
            stitch_response_bytes}));
```

关键点是反向更新请求的 op 数上限是 `R * storage_owner_batch_max`——一批 `storage_owner_batch_max` 个节点，每个节点有最多 `R` 条反向边，所以一次反向更新 RPC 最多携带 `R * batch_max` 个 `ReverseUpdateOp`。`stitch` 的请求/响应都按 `batch_max` 项算，响应还要乘 `candidate_capacity`（即 `resolved_storage_owner_construction_width()`，见 `configuration.hh:105`）。四者取最大再 `align_up`，就是每个 peer RPC 消息的统一缓冲大小。`lib_assert` 保证它不超过 verbs SGE 的 32 位长度上限。

接下来算接收/发送 slot 数量：

```cpp
// runtime.cc:28-35
const u32 remote_peer_count = num_storage_nodes_ - 1;
const u32 max_recv_wr = static_cast<u32>(std::max<i32>(1, config.max_recv_queue_wr));
const u32 max_slots_per_peer = std::max<u32>(1, max_recv_wr / remote_peer_count);
const u32 desired_slots_per_peer = std::max<u32>(16, config.storage_owner_rpc_depth * 4);
peer_rpc_runtime_.recv_slots_per_peer = std::min(desired_slots_per_peer, max_slots_per_peer);
peer_rpc_runtime_.send_slots_per_peer = std::min(
  std::max<u32>(1, config.storage_owner_rpc_depth),
  peer_rpc_runtime_.recv_slots_per_peer);
```

逻辑很清晰：

- 每个 peer 的接收 WR 预算是"总 recv WR / peer 数"，这是硬件队列长度的硬约束。
- 期望值是 `max(16, rpc_depth * 4)`，即至少 16 个、否则 4 倍 `rpc_depth`（`storage_owner_rpc_depth` 默认 8，见 `configuration.hh:72`）。
- 实际取两者较小值。
- 发送 slot 数量取 `max(1, rpc_depth)` 与接收 slot 的较小值——发送侧不需要像接收侧那么深，因为发送是主动的、可以反压。

随后算 buffer 三段布局：

```cpp
// runtime.cc:36-43
peer_rpc_runtime_.recv_region_bytes =
  peer_rpc_runtime_.message_bytes * num_storage_nodes_ * peer_rpc_runtime_.recv_slots_per_peer;
peer_rpc_runtime_.sync_send_offset = peer_rpc_runtime_.recv_region_bytes;
peer_rpc_runtime_.async_send_offset =
  peer_rpc_runtime_.sync_send_offset + peer_rpc_runtime_.message_bytes * num_storage_nodes_;
const size_t async_send_bytes = peer_rpc_runtime_.message_bytes * num_storage_nodes_ *
                                peer_rpc_runtime_.send_slots_per_peer;
peer_rpc_runtime_.buffer.allocate(peer_rpc_runtime_.async_send_offset + async_send_bytes);
peer_rpc_runtime_.buffer.touch_memory();
peer_rpc_runtime_.region = std::make_unique<LocalMemoryRegion>(
  *peer_context_, peer_rpc_runtime_.buffer.get_full_buffer(), peer_rpc_runtime_.buffer.buffer_size);
```

整块大缓冲被切成三段（按偏移区分用途，物理上是同一块 RDMA MR）：

1. **接收区** `[0, sync_send_offset)`：`num_storage_nodes_ * recv_slots_per_peer` 个 `message_bytes` 槽位。注意是按 `num_storage_nodes_`（包含自己）而不是 `remote_peer_count` 索引，这样 `peer_id` 可以直接做下标，省一次减法。
2. **同步发送区** `[sync_send_offset, async_send_offset)`：每个 peer 一个 `message_bytes` 槽位，专用于阻塞式发送（响应回送、同步直发请求）。
3. **异步发送区** `[async_send_offset, end)`：每个 peer `send_slots_per_peer` 个槽位，用于多路并发异步发送。

`buffer.allocate` 是 `HugePage<byte_t>`（见 `storage_owner_state.hh:126` 的 `PeerRpcRuntimeState`），`touch_memory` 触发页表预填，避免首次发送时的缺页抖动。最后把整块注册成单个 `LocalMemoryRegion`，所有收发都复用它。

下一步初始化发送 slot 的 free list（按 send class 分桶）：

```cpp
// runtime.cc:47-66
std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
peer_rpc_free_send_slots_.clear();
peer_rpc_free_send_slots_.resize(num_storage_nodes_);
peer_rpc_sync_send_mutexes_.clear();
peer_rpc_sync_send_mutexes_.resize(num_storage_nodes_);
for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
  if (peer_id == storage_id_) {
    continue;
  }
  peer_rpc_sync_send_mutexes_[peer_id] = std::make_unique<std::mutex>();
  for (u32 slot_id = 0;
       slot_id < peer_rpc_runtime_.send_slots_per_peer;
       ++slot_id) {
    const size_t send_class = static_cast<size_t>(
      peer_rpc_send_slot_class(slot_id));
    peer_rpc_free_send_slots_[peer_id][send_class].push_back(slot_id);
  }
}
```

`peer_rpc_free_send_slots_` 在 `memory_node.hh:608` 声明为 `vec<std::array<std::deque<u32>, 3>>`——每个 peer 三条 lane，对应 `PeerRpcSendClass` 的 `stitch_search / graph_update / control` 三态。`peer_rpc_send_slot_class`（`runtime.cc:263-269`）按 slot 编号的奇偶性把 slot 静态划分到 stitch 或 graph_update lane：

```cpp
MemoryNode::PeerRpcSendClass MemoryNode::peer_rpc_send_slot_class(
    u32 slot_id) const {
  const u32 slot_count = peer_rpc_runtime_.send_slots_per_peer;
  if (slot_count <= 1) return PeerRpcSendClass::control;
  return slot_id % 2 == 0 ? PeerRpcSendClass::stitch_search
                          : PeerRpcSendClass::graph_update;
}
```

这种"按奇偶切 lane"的设计保证 stitch 搜索（吞吐敏感）与反向更新（latency 敏感）不会在同一个发送 slot 上互相阻塞——即使总发送 slot 很少，两类请求也各有独立 lane。

随后创建异步响应登记表与请求去重表：

```cpp
// runtime.cc:67-80
const size_t registry_peer_count = std::max<u32>(1, num_storage_nodes_ - 1);
const size_t response_capacity = std::max<size_t>(
  1024,
  static_cast<size_t>(config.storage_owner_rpc_depth) *
    std::max<u32>(1, config.storage_owner_maintenance_workers) *
    registry_peer_count * 4);
peer_async_responses_ =
  std::make_unique<PeerAsyncResponseRegistry>(response_capacity);
const size_t dedup_capacity = std::max<size_t>(
  1024,
  static_cast<size_t>(config.storage_owner_reverse_queue_depth) *
    registry_peer_count * 2);
peer_request_deduplicator_ =
  std::make_unique<PeerRequestDeduplicator>(dedup_capacity);
```

容量推导：响应表容量按"`rpc_depth * maintenance_workers * peer_count * 4`"取，去重表按"`reverse_queue_depth * peer_count * 2`"取，下限都是 1024。这两个数会传给 `normalize_capacity`（`async_response.hh:340-346`）向上取整为 2 的幂，用作开地址哈希的 `mask_`。

`runtime.cc:81-89` 打印三条状态行。最后是接收 WR 的批量预投递：

```cpp
// runtime.cc:90-99
for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
  if (peer_id == storage_id_) continue;
  for (u32 slot_id = 0; slot_id < peer_rpc_runtime_.recv_slots_per_peer; ++slot_id) {
    peer_control_qp(peer_id)->post_receive(
      *peer_rpc_runtime_.region,
      static_cast<u32>(peer_rpc_runtime_.message_bytes),
      encode_64bit(peer_id, slot_id),
      peer_rpc_receive_offset(peer_id, slot_id));
  }
}
```

每个 (peer, slot) 对都 `post_receive` 一次，WR ID 用 `encode_64bit(peer_id, slot_id)`（`utils.hh:70`）把两个 32 位数压进 64 位，CQ 出队时 `decode_64bit` 解回来。`peer_rpc_receive_offset` 的实现：

```cpp
// runtime.cc:246-250
size_t MemoryNode::peer_rpc_receive_offset(u32 peer_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(peer_id) * peer_rpc_runtime_.recv_slots_per_peer + slot_id;
  return slot_index * peer_rpc_runtime_.message_bytes;
}
```

即"行主序"：peer 维度在外，slot 维度在内。这一布局在 `workers.cc` 的 CQ 处理循环里直接复用。

### 24.3.2 `start_peer_reverse_update_runtime`：拉起线程池

`runtime.cc:102-196` 的 `start_peer_reverse_update_runtime` 在 `setup_peer_rpc_runtime` 之后被调用，负责把运行期的状态结构清零并启动所有 worker 线程。开头同样有空操作守卫（102–105 行），随后清空响应 map（107–112 行）。

接下来把所有 shutdown/done 标志清零，把队列容量与计数器清零：

```cpp
// runtime.cc:114-134
peer_reverse_shutdown_.store(false, std::memory_order_release);
peer_reverse_workers_done_.store(false, std::memory_order_release);
peer_reverse_response_done_.store(false, std::memory_order_release);
peer_reverse_task_queue_limit_ =
  std::max<size_t>(1024, static_cast<size_t>(config.storage_owner_reverse_queue_depth));
peer_stitch_search_task_queue_limit_ = peer_reverse_task_queue_limit_;
peer_reverse_outgoing_queue_limit_ = peer_reverse_task_queue_limit_;
peer_reverse_responses_ =
  std::make_unique<bounded::Queue<PeerReverseUpdateResponse>>(
    peer_reverse_task_queue_limit_);
// ... 11 个 atomic<u64> 计数器全部 .store(0) ...
```

`peer_reverse_responses_` 是个有界阻塞队列（`common/bounded_queue.hh`），承载 worker 计算完毕后待发送的响应。三个队列容量都取 `max(1024, storage_owner_reverse_queue_depth)`。

然后是 CPU 规划与 worker 数推导：

```cpp
// runtime.cc:136-164
const u32 rpc_parallelism = std::max<u32>(
  1, static_cast<u32>(num_clients_) *
     std::max<u32>(1, config.storage_owner_rpc_depth));
const auto cpu_plan = memory_node_detail::derive_storage_owner_cpu_plan(
  core_assignment_.available_core_count(), num_compute_threads_,
  rpc_parallelism, config.storage_owner_maintenance_workers,
  num_storage_nodes_ > 0 ? num_storage_nodes_ - 1 : 0);
const u32 reverse_worker_count = cpu_plan.peer_reverse_workers;
const u32 stitch_worker_count = cpu_plan.peer_search_workers;
```

`derive_storage_owner_cpu_plan`（`storage_owner_cpu_plan.hh`）把可用核分配给 stage1/stage2/maintenance/peer-reverse/peer-search 等角色，`rpc_parallelism` 是 `num_clients_ * rpc_depth`，反映"同一时刻最多有多少条在飞 RPC"。两个 worker 数从 plan 里取出来。

随后为每个 reverse worker 分配协程 scratch（`StorageOwnerThread` + `init_peer_scratch`），但 stitch worker 不分配 scratch（它走同步 `send_peer_rpc_message`，不需要协程）：

```cpp
// runtime.cc:145-152
const size_t coroutine_scratch_stride =
  align_up(std::max<size_t>(VamanaNode::total_size(),
                            std::max(neighbor_stride,
                                     snapshot_stride *
                                       std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
const size_t scratch_bytes = coroutine_scratch_stride;
peer_reverse_worker_states_.reserve(reverse_worker_count);
for (u32 i = 0; i < reverse_worker_count; ++i) {
  auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
  worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
  peer_reverse_worker_states_.push_back(std::move(worker));
}
```

注意 `StorageOwnerThread` 的构造参数 `1` 是协程数——reverse worker 各自只跑 1 个协程（peer RPC 的处理是顺序的，不需要协程并发）。stitch worker 直接构造不带 scratch（159–164 行）。

最后启动五类线程并绑核（166–187 行）：

- `peer_rpc_progress_thread_`：CQ 进度主循环（`workers.cc:3`）。
- `peer_reverse_response_thread_`：响应发送线程（`workers.cc:307`）。
- `peer_reverse_outgoing_thread_`：出站 outbox 发送线程（`workers.cc:321`）。
- `reverse_worker_count` 个 `peer_reverse_workers_`：反向更新 worker（`workers.cc:203`）。
- `stitch_worker_count` 个 `peer_stitch_search_workers_`：stitch 搜索 worker（`workers.cc:261`）。

`pin_thread` 把每个线程钉到 `core_assignment_.get_available_core()` 返回的核上，关闭 `disable_thread_pinning` 时跳过。

### 24.3.3 `stop_peer_reverse_update_runtime`：有序关闭

`runtime.cc:198-244` 是对称的关闭流程。先把 shutdown 标志置位，notify 所有 cv，让阻塞在 `wait` 上的线程醒来：

```cpp
// runtime.cc:198-206
peer_reverse_shutdown_.store(true, std::memory_order_release);
peer_reverse_tasks_cv_.notify_all();
peer_stitch_search_tasks_cv_.notify_all();
if (peer_reverse_responses_) peer_reverse_responses_->notify_all();
peer_reverse_outgoing_cv_.notify_all();
peer_rpc_responses_cv_.notify_all();
peer_completion_cv_.notify_all();
```

然后按依赖顺序 join：先 outbox（207–209），再 reverse worker（210–214），再 stitch worker（215–219）。worker 全部退出后置 `peer_reverse_workers_done_`，让响应队列的 `pop_wait`（`bounded::Queue` 支持关闭谓词）能返回 false，从而响应线程退出（220–224）。最后 join CQ 进度线程（225–227）。

收尾工作（228–243）：`peer_async_responses_->drain_completed()` 把所有还挂在登记表里的 complete 状态响应的接收描述符取出来，逐个 `repost_peer_rpc_receive`——否则这些接收 slot 会泄露，下次启动就没有可用的 recv WR。然后清空 worker 容器、`reset` 响应队列、清空 `peer_rpc_*_` 三个 map。

### 24.3.4 send slot 调度与同步发送原语

`try_acquire_peer_rpc_send_slot`（`runtime.cc:271-295`）是发送侧的"准入控制"：

```cpp
bool MemoryNode::try_acquire_peer_rpc_send_slot(
    u32 peer_id,
    PeerRpcSendClass send_class,
    u32& slot_id) {
  lib_assert(peer_id < peer_rpc_free_send_slots_.size() && peer_id != storage_id_, ...);
  std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
  auto& lanes = peer_rpc_free_send_slots_[peer_id];
  auto try_lane = [&](PeerRpcSendClass lane) {
    auto& free_slots = lanes[static_cast<size_t>(lane)];
    if (free_slots.empty()) return false;
    slot_id = free_slots.front();
    free_slots.pop_front();
    return true;
  };

  if (peer_rpc_runtime_.send_slots_per_peer == 1) {
    return try_lane(PeerRpcSendClass::control);
  }
  if (send_class == PeerRpcSendClass::control) {
    return try_lane(PeerRpcSendClass::stitch_search) ||
           try_lane(PeerRpcSendClass::control);
  }
  return try_lane(send_class);
}
```

三种取法：

- 只有 1 个 slot 时，所有发送都走 `control` lane（实际就是唯一那个 slot）。
- `control` 类（运行时控制报文）可以挤占 `stitch_search` lane 或自己的 lane。
- 否则按请求自身的 class 取对应 lane，互不挤占——这就是 stage2 stitch 搜索与反向更新不会互相阻塞的保证。

`release_peer_rpc_send_slot`（297–307）是对称的归还。

`post_peer_rpc_send_slot`（320–347）是异步发送的核心：拿到 slot 后把 `wr_id = next_peer_async_wr_id()` 与一个 `PeerPendingSend{release_rpc_slot=true, rpc_slot_id=slot_id}` 关联登记，然后 `post_send_with_id`。发送完成时 `handle_peer_send_completion`（`peer_rdma.cc:187`）看到 `pending.release_rpc_slot` 为 true，就调用 `release_peer_rpc_send_slot` 自动归还（见 `peer_rdma.cc:200-203`）。这套机制让异步发送的 slot 回收完全自动化。

`send_peer_rpc_message`（349–382）是同步发送原语，给响应回送和直发请求用：

```cpp
void MemoryNode::send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes) {
  lib_assert(peer_context_ != nullptr, "peer context not initialized");
  lib_assert(peer_id < peer_rpc_sync_send_mutexes_.size() &&
               peer_rpc_sync_send_mutexes_[peer_id] != nullptr,
             "peer RPC sync send buffer is not initialized");
  lib_assert(bytes <= peer_rpc_runtime_.message_bytes, "peer rpc message too large");
  lib_assert(!current_peer_rpc_progress_thread_,
             "peer CQ progress thread must not execute a blocking response send");
  std::lock_guard<std::mutex> sync_lock(*peer_rpc_sync_send_mutexes_[peer_id]);
  const size_t offset = peer_rpc_sync_send_offset(peer_id);
  std::memcpy(peer_rpc_runtime_.buffer.get_full_buffer() + offset, payload, bytes);
  const u64 wr_id = next_peer_sync_wr_id();
  register_peer_pending_send_locked(wr_id, PeerPendingSend{.target_shard = peer_id, .target_qp_idx = 0});
  {
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[peer_id][0]);
    peer_control_qp(peer_id)->post_send_with_id(..., offset);
  }
  wait_peer_sync_completion(wr_id);
}
```

几个要点：

- **每个 peer 一把 `peer_rpc_sync_send_mutexes_` 互斥锁**：同步发送区每 peer 只有一个槽位，多个线程同时回 ACK 必须串行化。锁是 per-peer 的，不同 peer 之间不互斥。
- **`current_peer_rpc_progress_thread_`** 守卫（356–357）：CQ 进度线程绝不能调用同步发送，因为 `wait_peer_sync_completion` 需要进度线程来推进 CQ，自己等自己会死锁。
- **`wait_peer_sync_completion(wr_id)`**（`peer_rdma.cc:260-278`）：如果进度线程在跑，就在 `peer_completion_cv_` 上等 `peer_sync_completions_` 出现该 `wr_id`；否则自己 `poll_peer_send_cq` + `yield` 轮询。后者用于进度线程还没启动或已停止的窗口期。

### 24.3.5 响应构造

`make_peer_reverse_update_response`（`runtime.cc:384-402`）是纯函数，根据请求头构造响应头：magic/version 还原，type 由请求类型映射到对应响应类型（cleanup 请求映射到 cleanup 响应，否则映射到 reverse_update 响应），`source_shard` 填自己，`item_count`/`request_id` 回填，`status` 取 `ok`/`failed`。这个函数被 `request_handlers.cc` 和 `workers.cc` 复用。

---

## 24.4 `request_handlers.cc`：入站 RPC 的派发与处理

### 24.4.1 `handle_peer_rpc_request`：总入口

`request_handlers.cc:249-305` 是入站 RPC 的总派发。它先做最小合法性校验：

```cpp
// request_handlers.cc:249-258
bool MemoryNode::handle_peer_rpc_request(const PeerRpcMessage& message, const Configuration& config) {
  if (message.payload.size() < sizeof(service::storage_owner::PeerRpcHeader)) {
    return false;
  }
  const auto* header =
    reinterpret_cast<const service::storage_owner::PeerRpcHeader*>(message.payload.data());
  if (header->magic != service::storage_owner::kPeerRpcMagic ||
      header->version != service::storage_owner::kPeerRpcVersion) {
    return false;
  }
```

magic/version 不符就拒绝（不回响应——这通常是版本不匹配或脏数据）。然后按 `header->type` 分派：

- `reverse_update_request`（260–267）：检查 `payload.size() >= reverse_update_request_bytes(item_count)`，取出 `ops` 指针，调 `handle_peer_reverse_update_request`。
- `cleanup_deleted_request`（268–275）：同上，调 `handle_peer_cleanup_deleted_request`。
- `stitch_search_request`（276–302）：除了字节校验，还要校验 `item_count` 在 `(0, storage_owner_batch_max]` 范围内、`reserved == resolved_storage_owner_construction_width()`（即 `candidate_capacity` 必须和本节点配置一致）。任一不符就发 `send_peer_stitch_search_failed_response` 并返回 false；否则构造 `PeerStitchSearchTask` 入队 `enqueue_peer_stitch_search_task`，入队失败也发 failed 响应。

注意 stitch 请求是**入队异步处理**，而 reverse/cleanup 是**直接同步处理**——这一差异源于两者的耗时模型：stitch 搜索要做实际图遍历（数百微秒到毫秒级），必须交给专门 worker；reverse/cleanup 只是修改邻接表（微秒级），直接在调用线程做即可。但这条规则只对 `handle_peer_rpc_request` 这条同步路径成立；真正的 CQ 进度线程走的是 `workers.cc` 里的去重+入队路径（见 24.5.1）。

### 24.4.2 `handle_peer_reverse_update_request` 与 cleanup

`request_handlers.cc:87-105`：

```cpp
bool MemoryNode::handle_peer_reverse_update_request(u32 source_shard,
                                        const service::storage_owner::PeerRpcHeader& header,
                                        const service::storage_owner::ReverseUpdateOp* ops,
                                        const Configuration& config) {
  PeerReverseUpdateTask task;
  task.source_shard = source_shard;
  task.header = header;
  task.received_at = std::chrono::steady_clock::now();
  task.ops.assign(ops, ops + header.item_count);
  const bool success = apply_peer_reverse_update_task(task, config);
  if ((header.reserved & kPeerRpcFlagNoResponse) == 0) {
    PeerReverseUpdateResponse response;
    response.destination_shard = source_shard;
    response.header = make_peer_reverse_update_response(header, success);
    response.queued_at = std::chrono::steady_clock::now();
    send_peer_reverse_update_response(response);
  }
  return success;
}
```

把 `ops` 拷到 `task.ops`（POD 拷贝），调 `apply_peer_reverse_update_task` 真正落地，然后除非 `kPeerRpcFlagNoResponse` 置位，否则回送 ACK。`handle_peer_cleanup_deleted_request`（107–130）结构完全一致，只是 `apply_peer_reverse_update_task` 内部会根据 `request_type` 走不同的 apply 路径。

### 24.4.3 `apply_peer_reverse_update_tasks`：批量落地

`request_handlers.cc:9-63` 是真正修改本地图的入口：

```cpp
// request_handlers.cc:9-41
bool MemoryNode::apply_peer_reverse_update_tasks(const vec<PeerReverseUpdateTask>& tasks, const Configuration& config) {
  if (tasks.empty()) return true;
  const auto apply_started = std::chrono::steady_clock::now();
  const auto request_type = static_cast<service::storage_owner::PeerRpcType>(
    tasks.front().header.type);
  lib_assert(request_type == service::storage_owner::PeerRpcType::reverse_update_request ||
               request_type == service::storage_owner::PeerRpcType::cleanup_deleted_request, ...);
  dense_hashmap_t<u64, vec<RemotePtr>> grouped;
  size_t item_count = 0;
  for (const PeerReverseUpdateTask& task : tasks) {
    lib_assert(task.header.type == tasks.front().header.type, ...);
    item_count += task.ops.size();
  }
  grouped.reserve(item_count);
  for (const PeerReverseUpdateTask& task : tasks) {
    for (const auto& op : task.ops) {
      const RemotePtr target{op.target_raw};
      const RemotePtr candidate{op.candidate_raw};
      // Peer payloads are a trust boundary ...
      if (!valid_local_storage_node_pointer(target)) {
        return false;
      }
      grouped[target.raw_address].push_back(candidate);
    }
  }
```

关键步骤：

1. **同批合并**：多个 task 的 ops 按 `target.raw_address` 聚合到 `grouped`——同一目标节点的多个候选边会被一次性写入，减少节点锁次数。这是 worker 主循环里 `coalesce` 的延续。
2. **信任边界**：`valid_local_storage_node_pointer(target)` 校验 target 指针确实落在本地分片、对齐、在范围内、且当前已分配。注释明确指出 peer payload 是信任边界，恶意/错误/版本不匹配的 target 指针必须在此处拦截，否则会成为本地 OOB。校验失败直接 `return false`（整批失败，不部分应用）。

然后按请求类型选择 apply 路径：

```cpp
// request_handlers.cc:43-46
const bool success =
  request_type == service::storage_owner::PeerRpcType::cleanup_deleted_request
    ? remove_local_neighbors_batched(grouped, config)
    : apply_local_reverse_updates_batched(grouped, config);
```

`apply_local_reverse_updates_batched` 和 `remove_local_neighbors_batched` 属于 `storage_owner_index/`（第 25 课）。两者都是按 target 节点加锁、批量改邻接表。

结尾有一段慢日志（47–61）：如果 `apply_ns > 1s`，输出前 16 次的 task_count/item_count/grouped_targets/elapsed_ms。`static std::atomic<u32> slow_apply_logs` 限制日志量避免刷屏。

### 24.4.4 `send_peer_reverse_update_response` 与响应队列

`request_handlers.cc:65-85` 的 `send_peer_reverse_update_response` 直接调 `send_peer_rpc_message`（同步发送），然后慢日志（>1s 输出前 16 次，含 `queued_ms` 即从 `queued_at` 到真正发送的等待时间）。

`enqueue_peer_reverse_update_response`（341–349）构造一个 `PeerReverseUpdateResponse` 并 `try_enqueue_peer_reverse_update_response`。后者（351–369）尝试 `peer_reverse_responses_->try_push`，失败则丢弃并打日志：

```cpp
// request_handlers.cc:357-367
// A successful reverse operation remains in the bounded receiver dedup
// cache. Dropping only this ACK is safe: the source retries the identical
// request ID and receives a replay without applying the graph operation
// twice. Failed operations are retryable by definition as well.
```

这段注释是理解整个 peer RPC 重试语义的关键：**ACK 丢失是安全的**，因为接收端的 `PeerRequestDeduplicator` 会缓存成功响应，发送端用相同 `request_id` 重试时接收端会直接回放缓存的响应，不会重复应用图操作。失败响应不缓存（见 24.6.2），重试会真的重新执行——但失败操作本身就是幂等的（重试要么还是失败，要么成功）。

### 24.4.5 `handle_peer_stitch_search_request`：候选搜索

`request_handlers.cc:132-221` 是 stitch 搜索的核心，它在接收端为每个请求项（target + 向量）执行一次本地 Vamana 搜索，返回 `candidate_capacity` 个候选。

开头校验 `candidate_capacity`（即 `header.reserved`）必须等于 `resolved_storage_owner_construction_width()`，否则发 failed 响应。然后分配响应缓冲并填头：

```cpp
// request_handlers.cc:143-155
const size_t response_bytes = service::storage_owner::stitch_search_response_bytes(
  header.item_count, candidate_capacity);
vec<byte_t> response(response_bytes, 0);
auto* response_header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(response.data());
response_header->magic = ...; response_header->version = ...;
response_header->type = static_cast<u32>(PeerRpcType::stitch_search_response);
response_header->source_shard = storage_id_;
response_header->item_count = header.item_count;
response_header->request_id = header.request_id;
response_header->status = static_cast<u32>(InsertStatus::ok);
response_header->reserved = candidate_capacity;
```

如果 `storage_owner_route_table_` 为空（本节点还没建好路由表），直接把 status 改 `failed` 并发回。

接下来逐项搜索：

```cpp
// request_handlers.cc:172-217
thread_local vec<element_t> query;
query.resize(VamanaNode::DIM);
bool success = true;
for (u32 i = 0; i < header.item_count; ++i) {
  const byte_t* raw_vector = vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
  decode_storage_vector_to_float(raw_vector, VamanaNode::vector_dtype(), VamanaNode::DIM, query.data());
  const auto components = span<const element_t>{query.data(), query.size()};
  vec<RemotePtr> entries = storage_owner_route_entries(components);
  if (entries.empty()) {
    response_header->status = static_cast<u32>(InsertStatus::failed);
    success = false;
    continue;
  }
  vec<RemotePtr> candidates =
    partition_local_search_candidates(components, entries, config, nullptr, raw_vector);
  u32 written = 0;
  hashset_t<RemotePtr> seen;
  for (const RemotePtr& candidate : candidates) {
    if (written >= candidate_capacity) break;
    if (candidate.is_null() || !local_shard(candidate.memory_node()) ||
        candidate.raw_address == items[i].target_raw ||
        !seen.insert(candidate).second) {
      continue;
    }
    const byte_t* candidate_vector = local_live_vector(candidate);
    if (candidate_vector == nullptr) continue;
    const byte_t* candidate_node = local_node_ptr(candidate);
    const size_t slot = static_cast<size_t>(i) * candidate_capacity + written;
    candidate_slots[slot].raw = candidate.raw_address;
    candidate_slots[slot].generation = *reinterpret_cast<const u32*>(
      candidate_node + VamanaNode::offset_generation());
    std::memcpy(candidate_vectors + slot * VamanaNode::vector_bytes(),
                candidate_vector, VamanaNode::vector_bytes());
    ++written;
  }
  counts[i] = written;
}
send_peer_rpc_message(source_shard, response.data(), response.size());
```

流程：

1. `decode_storage_vector_to_float` 把磁盘编码的向量解码成 float 查询向量。
2. `storage_owner_route_entries` 从路由表（`adaptive_route_table.hh`，第 8 课）取出本地入口点。
3. `partition_local_search_candidates` 跑一次 beam search（stage1 候选搜索的本地版本），返回一组候选 `RemotePtr`。
4. 对每个候选做去重（`seen` hashset）、跳过 target 自己、跳过非本地、跳过已删除（`local_live_vector == nullptr`）。
5. 写入 `candidate_slots`（含 `raw` 和 `generation`，generation 用于 RCU 一致性，第 16 课）和 `candidate_vectors`（向量数据，供请求方做最终 prune）。
6. `counts[i]` 记录实际写入数（可能少于 `candidate_capacity`）。

`send_peer_stitch_search_failed_response`（223–247）是失败响应构造，注释明确："Never size a local response from an untrusted or differently configured peer. All shards in a deployment still have to agree on L."——`candidate_capacity` 永远取本地配置，不用请求方的 `reserved`，防止恶意/错误请求触发大缓冲分配。

### 24.4.6 入队原语

`enqueue_peer_reverse_update_task`（307–324）与 `enqueue_peer_stitch_search_task`（326–339）是带背压的入队：

```cpp
// request_handlers.cc:307-324
bool MemoryNode::enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task) {
  const u64 item_count = task.ops.size();
  size_t queue_size = 0;
  std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
  if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
      peer_reverse_tasks_.size() >= peer_reverse_task_queue_limit_) {
    return false;
  }
  peer_reverse_tasks_.push_back(std::move(task));
  queue_size = peer_reverse_tasks_.size();
  lock.unlock();
  peer_reverse_update_enqueued_.fetch_add(1, std::memory_order_relaxed);
  peer_reverse_update_items_enqueued_.fetch_add(item_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(peer_reverse_update_max_queue_, static_cast<u64>(queue_size));
  peer_reverse_tasks_cv_.notify_one();
  return true;
}
```

队列满时返回 false，让调用方决定如何反压（CQ 进度线程会发失败响应、abandon 去重表项）。计数器更新放在锁外（relaxed 序），`update_max_relaxed` 跟踪历史峰值，供运维观测。`notify_one` 唤醒一个等待的 worker。

---

## 24.5 `workers.cc`：四个线程主循环

### 24.5.1 `peer_rpc_progress_loop`：CQ 进度主循环

`workers.cc:3-201` 是 peer RPC 的心脏——单线程轮询 peer context 的 recv CQ，把每个收到的报文按类型路由。开头设 `current_peer_rpc_progress_thread_ = true`，这正是 `send_peer_rpc_message` 里那个死锁守卫的目标。

主循环结构：

```cpp
// workers.cc:9-39
vec<ibv_wc> recv_wcs(std::max<i32>(1, peer_context_->get_config().max_recv_queue_wr));
for (;;) {
  poll_peer_send_cq();
  const i32 num_received =
    peer_context_->poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
  if (num_received <= 0) {
    bool responses_empty = false;
    bool outgoing_empty = false;
    bool sends_empty = false;
    if (peer_reverse_response_done_.load(std::memory_order_acquire)) {
      responses_empty = peer_reverse_responses_ == nullptr || peer_reverse_responses_->empty();
      { std::lock_guard<std::mutex> lock(peer_reverse_outgoing_mutex_); outgoing_empty = peer_reverse_outgoing_.empty(); }
      { std::lock_guard<std::mutex> lock(peer_completion_mutex_); sends_empty = peer_pending_sends_.empty(); }
    }
    if (peer_reverse_response_done_.load(std::memory_order_acquire) &&
        responses_empty && outgoing_empty && sends_empty) {
      peer_rpc_progress_running_.store(false, std::memory_order_release);
      peer_completion_cv_.notify_all();
      current_peer_rpc_progress_thread_ = false;
      return;
    }
    std::this_thread::yield();
    continue;
  }
```

每一轮先 `poll_peer_send_cq()`——把发送完成事件处理掉（释放发送 slot、归还 RDMA read credit、唤醒等待同步完成的线程）。然后轮询 recv CQ。如果没有报文，检查是否所有 worker 都已退出（`peer_reverse_response_done_` 置位）、响应队列为空、outbox 队列为空、没有挂起的发送——全部满足时进度线程退出。退出前 `peer_rpc_progress_running_.store(false)` 并 `notify` `peer_completion_cv_`，让可能阻塞在 `wait_peer_sync_completion` 上的线程解锁。

接下来遍历每个收到的 WC：

```cpp
// workers.cc:41-61
for (i32 i = 0; i < num_received; ++i) {
  bool hold_receive_slot = false;
  const auto [peer_id, slot_id] = decode_64bit(recv_wcs[i].wr_id);
  if (peer_id >= num_storage_nodes_ || slot_id >= peer_rpc_runtime_.recv_slots_per_peer) continue;
  const size_t offset = peer_rpc_receive_offset(peer_id, slot_id);
  const byte_t* payload = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
  const size_t bytes = recv_wcs[i].byte_len;
  if (bytes < sizeof(service::storage_owner::PeerRpcHeader)) {
    repost_peer_rpc_receive(peer_id, slot_id); continue;
  }
  const auto* header = reinterpret_cast<const PeerRpcHeader*>(payload);
  if (header->magic != kPeerRpcMagic || header->version != kPeerRpcVersion ||
      header->source_shard != peer_id) {
    repost_peer_rpc_receive(peer_id, slot_id); continue;
  }
```

`hold_receive_slot` 标志是关键：默认处理完会 `repost_peer_rpc_receive`（把 slot 重新投递回 recv WR，准备接下一帧）；但如果响应走异步登记表（`try_deliver` 成功），slot 必须保留——payload 还在那里，等 stage2 协程来 `try_consume_peer_rpc_response` 拷走。注意校验 `header->source_shard != peer_id`：WR ID 里的 peer_id 是"哪个 QP 收到的"，header 里的 source_shard 是"谁发的"，两者必须一致，防止 spoofing。

接下来按类型分派。先看反向更新请求的处理：

```cpp
// workers.cc:76-104
if (header->type == static_cast<u32>(PeerRpcType::reverse_update_request)) {
  const size_t expected_bytes = reverse_update_request_bytes(header->item_count);
  if (bytes >= expected_bytes) {
    const auto decision = peer_request_deduplicator_->begin(peer_id, *header, true);
    if (decision.action == PeerRequestAction::execute) {
      const auto* ops = reverse_update_ops(payload);
      PeerReverseUpdateTask task;
      task.source_shard = peer_id; task.header = *header;
      task.received_at = std::chrono::steady_clock::now();
      task.ops.assign(ops, ops + header->item_count);
      if (!enqueue_peer_reverse_update_task(std::move(task))) {
        peer_request_deduplicator_->abandon(peer_id, *header);
        if ((header->reserved & kPeerRpcFlagNoResponse) == 0) {
          enqueue_peer_reverse_update_response(peer_id, *header, false);
        }
      }
    } else if (decision.action == PeerRequestAction::replay &&
               (header->reserved & kPeerRpcFlagNoResponse) == 0) {
      PeerReverseUpdateResponse response;
      response.destination_shard = peer_id;
      response.header = decision.response;
      response.queued_at = std::chrono::steady_clock::now();
      (void)try_enqueue_peer_reverse_update_response(std::move(response));
    }
  }
}
```

这是去重表的核心使用场景：

- `peer_request_deduplicator_->begin(...)` 返回 `PeerRequestDecision`，action 有四种：`execute`（首次或可重做，真的入队执行）、`duplicate_inflight`（已在执行中，忽略）、`replay`（已完成且响应可回放，直接回缓存的响应）、`conflict`/`full`（异常）。
- `execute` 分支：构造 task 入队。入队失败（队列满）时 `abandon` 去重表项（让下次重试能再次 execute），并回失败响应。
- `replay` 分支：直接把缓存的响应头入队回送，**不重新执行图操作**——这就是 ACK 丢失安全性的实现。
- `duplicate_inflight` 和 `conflict`/`full` 都不回响应——发送方会因 `rpc_timeout_ms` 重试。

cleanup 请求（105–133）的处理结构完全相同，只是 `response_replayable` 参数也是 true。stitch 请求（134–151）的 `response_replayable` 是 false——stitch 响应太大不缓存，重试会重新搜索，但去重表仍会用 `duplicate_inflight` 状态合并并发重复请求（同一 `request_id` 的并发请求只执行一次）。

响应类型（152–196）走另一条路径：

```cpp
// workers.cc:152-171
} else if (header->type == static_cast<u32>(PeerRpcType::reverse_update_response) ||
           header->type == static_cast<u32>(PeerRpcType::cleanup_deleted_response)) {
  const bool valid_response = bytes >= reverse_update_response_bytes();
  if (valid_response && peer_async_responses_ != nullptr &&
      peer_async_responses_->try_deliver(peer_id, slot_id, bytes, *header)) {
    hold_receive_slot = true;
    storage_owner_maintenance_cv_.notify_all();
  } else {
    bool accepted = false;
    {
      std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
      if (peer_rpc_pending_responses_.contains(header->request_id)) {
        peer_rpc_responses_[header->request_id] = *header;
        accepted = true;
      }
    }
    if (accepted) peer_rpc_responses_cv_.notify_all();
  }
}
```

两条路径：

1. **异步登记表路径**：`try_deliver` 成功意味着这个响应对应一个 stage2 注册的 `request_id`，slot 被登记表"接管"（`hold_receive_slot = true`，不 repost），并唤醒 stage2 协程（`storage_owner_maintenance_cv_`）来消费。
2. **同步等待路径**：如果 `try_deliver` 失败（响应没在异步登记表里，或者元数据不匹配），回退到同步 `peer_rpc_pending_responses_` map——只有当 `request_id` 在这个 set 里时才接受，并 `notify` `peer_rpc_responses_cv_` 唤醒 `wait_for_peer_reverse_update_response`。

这种"异步优先，同步兜底"的设计让同一套 CQ 处理能同时服务 stage2 协程（异步）和 stage2 同步直发（`send_peer_op_batch_direct` + `wait_for_peer_reverse_update_response`）。

stitch 响应（172–196）的差别在于多保存了 payload（`peer_rpc_response_payloads_`），因为 stitch 响应体很大，不能只存 header。

每个 WC 处理完，如果 `!hold_receive_slot` 就 `repost_peer_rpc_receive`（198 行）。

### 24.5.2 `peer_reverse_update_worker_loop`：反向更新 worker

`workers.cc:203-259` 是 reverse worker 主循环。它从 `peer_reverse_tasks_` 队列取 task，做合批后调用 `apply_peer_reverse_update_tasks`：

```cpp
// workers.cc:203-236
void MemoryNode::peer_reverse_update_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = peer_reverse_worker_states_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    vec<PeerReverseUpdateTask> tasks;
    tasks.reserve(8);
    {
      std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
      peer_reverse_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) || !peer_reverse_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) && peer_reverse_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      tasks.push_back(std::move(peer_reverse_tasks_.front()));
      peer_reverse_tasks_.pop_front();
      const u32 request_type = tasks.back().header.type;
      size_t coalesced_ops = tasks.back().ops.size();
      while (!peer_reverse_tasks_.empty() &&
             coalesced_ops < config.storage_owner_reverse_coalesce_max) {
        if (peer_reverse_tasks_.front().header.type != request_type) break;
        const size_t next_ops = peer_reverse_tasks_.front().ops.size();
        if (!tasks.empty() && coalesced_ops + next_ops > config.storage_owner_reverse_coalesce_max) break;
        tasks.push_back(std::move(peer_reverse_tasks_.front()));
        peer_reverse_tasks_.pop_front();
        coalesced_ops += next_ops;
      }
    }
    peer_reverse_tasks_cv_.notify_one();
```

合批逻辑：

1. 先取队首一个 task。
2. 继续从队列里取**同 request_type** 的 task，累加 ops 数。
3. 累加到 `storage_owner_reverse_coalesce_max`（默认 256）就停。
4. 类型不同的 task 不合批（reverse 和 cleanup 不能混）。
5. 出锁后 `notify_one` 让其他 worker 继续取剩余 task。

合批后调 `apply_peer_reverse_update_tasks`（242 行），然后更新计数器（243–248）：

```cpp
peer_reverse_update_processed_.fetch_add(tasks.size(), std::memory_order_relaxed);
peer_reverse_update_items_processed_.fetch_add(processed_items, std::memory_order_relaxed);
if (!success) peer_reverse_update_failed_.fetch_add(1, std::memory_order_relaxed);
```

最后为每个 task 调 `peer_request_deduplicator_->complete`（缓存成功响应供后续 replay）和 `enqueue_peer_reverse_update_response`（回 ACK）：

```cpp
// workers.cc:249-257
for (const PeerReverseUpdateTask& task : tasks) {
  const auto response_header = make_peer_reverse_update_response(task.header, success);
  peer_request_deduplicator_->complete(task.source_shard, task.header, response_header);
  if ((task.header.reserved & kPeerRpcFlagNoResponse) == 0) {
    enqueue_peer_reverse_update_response(task.source_shard, task.header, success);
  }
}
```

注意 `success` 是整批的成败——所有 task 共享一个 success，因为它们改的是同一批节点锁，要么都成功要么都失败。

### 24.5.3 `peer_stitch_search_worker_loop`：stitch worker

`workers.cc:261-305` 与 reverse worker 结构类似，但有几个关键差异：

```cpp
// workers.cc:278-291
peer_stitch_search_active_workers_.fetch_add(1, std::memory_order_acq_rel);
atomic_utils::CounterDecrementGuard active_slot(peer_stitch_search_active_workers_);

PeerStitchSearchTask task;
{
  std::lock_guard<std::mutex> lock(peer_stitch_search_tasks_mutex_);
  if (peer_stitch_search_tasks_.empty()) continue;
  task = std::move(peer_stitch_search_tasks_.front());
  peer_stitch_search_tasks_.pop_front();
}
```

`peer_stitch_search_active_workers_` 跟踪"正在处理"的 worker 数，供 shutdown 时判断是否还有在飞的 stitch 任务。`CounterDecrementGuard` 是 RAII 守卫，函数返回时自动减 1。

注意 stitch worker 不做合批——每个 task 独立处理，因为 stitch 搜索的耗时与 batch 大小近线性，合批没有收益反而增加尾延迟。

```cpp
// workers.cc:293-304
const bool success = handle_peer_stitch_search_request(
  task.source_shard, task.header, task.payload.data(), config);
// Stitch responses are not cached because their payload is large. The
// request is read-only, so a same-ID retry after this response may safely
// recompute it; duplicates that arrived while it ran were coalesced.
peer_request_deduplicator_->abandon(task.source_shard, task.header);
peer_stitch_search_processed_.fetch_add(1, std::memory_order_relaxed);
if (success) peer_stitch_search_items_.fetch_add(task.header.item_count, std::memory_order_relaxed);
```

`handle_peer_stitch_search_request` 内部已经 `send_peer_rpc_message` 回送响应，所以这里只需 `abandon` 去重表项（让相同 ID 的重试能重新 execute）。注释解释了为什么可以 abandon：stitch 是只读操作，重试重新计算是安全的；并发重复请求在执行期间已经被去重表合并（`duplicate_inflight`）。

### 24.5.4 `peer_reverse_response_loop`：响应发送线程

`workers.cc:307-319` 极简：

```cpp
void MemoryNode::peer_reverse_response_loop() {
  for (;;) {
    PeerReverseUpdateResponse response;
    lib_assert(peer_reverse_responses_ != nullptr, ...);
    if (!peer_reverse_responses_->pop_wait(response, peer_reverse_workers_done_)) {
      peer_reverse_response_done_.store(true, std::memory_order_release);
      return;
    }
    send_peer_reverse_update_response(response);
  }
}
```

专线程串行发送 ACK。`pop_wait` 阻塞在有界队列上，队列关闭（`peer_reverse_workers_done_` 为 true 且队列空）时返回 false，置 `peer_reverse_response_done_` 让进度线程退出。这个串行化是必要的：`send_peer_rpc_message` 是同步阻塞的，如果 worker 自己发响应会阻塞 worker 主循环；用一个专门线程把"计算"和"发送"解耦，worker 可以立即回去取下一个 task。

### 24.5.5 `peer_reverse_outgoing_loop`：出站 outbox 线程

`workers.cc:321-388` 处理本节点主动发起的反向更新（async mode）：

```cpp
// workers.cc:321-363
void MemoryNode::peer_reverse_outgoing_loop() {
  const Configuration& config = *storage_worker_config_;
  const u64 wire_max_u64 = std::max<u64>(1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  const u32 wire_max = static_cast<u32>(std::min<u64>(wire_max_u64, std::numeric_limits<u32>::max()));
  const u32 coalesce_max = std::min(wire_max, std::max<u32>(1, config.storage_owner_reverse_coalesce_max));
  for (;;) {
    PeerReverseOutgoingTask task;
    {
      std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_);
      peer_reverse_outgoing_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) || !peer_reverse_outgoing_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) && peer_reverse_outgoing_.empty()) return;
      task = std::move(peer_reverse_outgoing_.front());
      peer_reverse_outgoing_.pop_front();
      size_t coalesced_ops = task.ops.size();
      size_t scanned = 0;
      constexpr size_t kOutboxCoalesceScanLimit = 64;
      for (auto it = peer_reverse_outgoing_.begin();
           it != peer_reverse_outgoing_.end() && coalesced_ops < coalesce_max &&
           scanned < kOutboxCoalesceScanLimit;) {
        ++scanned;
        if (it->target_shard != task.target_shard || it->rpc_type != task.rpc_type) { ++it; continue; }
        const size_t next_ops = it->ops.size();
        if (coalesced_ops + next_ops > coalesce_max) break;
        task.ops.insert(task.ops.end(), it->ops.begin(), it->ops.end());
        coalesced_ops += next_ops;
        it = peer_reverse_outgoing_.erase(it);
      }
    }
    peer_reverse_outgoing_cv_.notify_one();
    const auto send_started = std::chrono::steady_clock::now();
    const bool success = send_peer_op_batch_direct(task.target_shard, task.ops, task.rpc_type, false, config);
    ...
  }
}
```

与 reverse worker 的合批类似，但这里合批的是**出站任务**（同 target_shard + 同 rpc_type），上限 `min(wire_max, coalesce_max)`。`kOutboxCoalesceScanLimit = 64` 限制扫描队列的深度，避免队列很长时合批扫描本身成为瓶颈。合批后调 `send_peer_op_batch_direct(..., wait_for_response=false, ...)`——outbox 模式不等待 ACK（依赖去重表的重试保证可靠性）。

慢日志（369–386）输出 `queued_ms`（从入队到发出的等待）和 `elapsed_ms`（发送耗时），>1s 或失败时各打前 16 次。

---

## 24.6 `async_response.hh`：两个核心数据结构

### 24.6.1 `PeerAsyncResponseRegistry`：请求-响应关联表

`async_response.hh:58-362` 是一个定容开地址哈希表，把"逻辑请求 ID"映射到"响应接收描述符"。它的核心约束写在头注释里（53–57）：

> Payloads remain in their registered RDMA receive slots until the stage2 executor consumes them, so CQ progress performs neither allocation nor payload copy. All methods are short critical sections; callers perform copies and reposts after the registry lock has been released.

即：响应 payload 不拷贝，直接留在 RDMA recv buffer 里，登记表只存描述符 `(peer_id, receive_slot, bytes, header)`；stage2 协程消费时再拷贝，然后 `repost_peer_rpc_receive`。

容量归一化（340–346）：

```cpp
static size_t normalize_capacity(size_t requested) {
  requested = std::max<size_t>(2, requested);
  if (requested > (size_t{1} << 62)) throw std::invalid_argument(...);
  return std::bit_ceil(requested);
}
```

`std::bit_ceil` 向上取整为 2 的幂，`mask_ = capacity_ - 1` 用于快速取模。`hash_request_id`（348–355）是 SplitMix64 风格的整数哈希，分布均匀。

每个 slot 有五态（228–234）：

```cpp
enum class State : std::uint8_t {
  empty,       // 从未使用
  pending,     // 已注册请求，等响应
  complete,    // 收到响应，等消费
  retryable,   // 响应被拒（payload 校验失败），可重发
  retired,     // 已消费或已取消，可复用
};
```

`find_locked`（301–322）是开放寻址线性探测，同时返回 `found`（已存在项）和 `insertion`（可插入位置，优先用 retired 槽，否则用第一个 empty）：

```cpp
[[nodiscard]] Lookup find_locked(u64 request_id) const {
  Lookup result;
  size_t first_retired = npos;
  size_t index = hash_request_id(request_id) & mask_;
  for (size_t probe = 0; probe < capacity_; ++probe) {
    const Slot& slot = slots_[index];
    if (slot.state == State::empty) {
      result.insertion = first_retired == npos ? index : first_retired;
      return result;
    }
    if (slot.request_id == request_id) { result.found = index; return result; }
    if (slot.state == State::retired && first_retired == npos) first_retired = index;
    index = (index + 1) & mask_;
  }
  result.insertion = first_retired;
  return result;
}
```

`register_send_attempt`（90–98）和 `register_request`（75–83）的差别在于是否允许"复活" retired 槽——前者用于 stage2 协程的真实发送尝试，可以复活；后者用于首次注册，不能复活。两者都调 `register_request_locked`（253–299）：

```cpp
PeerResponseRegistration register_request_locked(u64 request_id, ..., bool revive_retired) {
  if (request_id == 0) return PeerResponseRegistration::conflict;
  const Lookup lookup = find_locked(request_id);
  if (lookup.found != npos) {
    Slot& slot = slots_[lookup.found];
    if (!metadata_matches(slot, ...)) return PeerResponseRegistration::conflict;
    if (slot.state == State::pending) return PeerResponseRegistration::retry;
    if (slot.state == State::complete) return PeerResponseRegistration::already_complete;
    if (slot.state == State::retryable) {
      slot.response = {};
      slot.state = State::pending;
      return PeerResponseRegistration::retry;
    }
    if (!revive_retired) return PeerResponseRegistration::retired;
    slot.response = {};
    slot.state = State::pending;
    ++size_;
    return PeerResponseRegistration::retry;
  }
  if (size_ == capacity_ || lookup.insertion == npos) return PeerResponseRegistration::full;
  Slot& slot = slots_[lookup.insertion];
  slot.request_id = request_id; slot.expected_shard = expected_shard;
  slot.expected_type = expected_type; slot.expected_item_count = expected_item_count;
  slot.response = {}; slot.state = State::pending;
  ++size_;
  return PeerResponseRegistration::registered;
}
```

返回值六种：`registered`（新注册）、`retry`（已有同 ID 同元数据项，重发即可）、`already_complete`（响应已到，无需重发）、`retired`（已消费且不允许复活）、`conflict`（元数据不匹配，可能是 ID 重用冲突）、`full`（表满）。

`try_deliver`（103–129）在 CQ 进度线程里被调用：

```cpp
bool try_deliver(u32 peer_id, u32 receive_slot, size_t bytes,
                 const service::storage_owner::PeerRpcHeader& header) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Lookup lookup = find_locked(header.request_id);
  if (lookup.found == npos) return false;
  Slot& slot = slots_[lookup.found];
  if (slot.state != State::pending ||
      !metadata_matches(slot, peer_id, static_cast<PeerRpcType>(header.type), header.item_count) ||
      header.source_shard != peer_id) {
    return false;
  }
  slot.response = PeerResponseDescriptor{
    .peer_id = peer_id, .receive_slot = receive_slot, .bytes = bytes, .header = header};
  slot.state = State::complete;
  return true;
}
```

只有 `pending` 状态 + 元数据完全匹配（shard/type/item_count）+ `source_shard == peer_id` 才接受。返回 true 时 slot 进入 `complete`，CQ 进度线程会 `hold_receive_slot = true` 不 repost。返回 false 时 CQ 进度线程会尝试同步路径或 repost。

`try_take`（131–166）是 stage2 协程的消费侧：

```cpp
TryPeerResponse try_take(u64 request_id, u32 expected_shard, PeerRpcType expected_type,
                         u32 expected_item_count, PeerResponseDescriptor& response) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Lookup lookup = find_locked(request_id);
  if (lookup.found == npos) return TryPeerResponse::stale;
  Slot& slot = slots_[lookup.found];
  if (!metadata_matches(slot, expected_shard, expected_type, expected_item_count) ||
      slot.state == State::retired) return TryPeerResponse::stale;
  if (slot.state == State::pending || slot.state == State::retryable) return TryPeerResponse::pending;
  if (slot.state != State::complete) return TryPeerResponse::stale;
  response = slot.response;
  const bool success = response.header.status == static_cast<u32>(InsertStatus::ok);
  if (success) retire_locked(slot);
  else { slot.response = {}; slot.state = State::retryable; }
  return success ? TryPeerResponse::success : TryPeerResponse::failure;
}
```

关键设计：

- **成功响应消费后 `retire`**——slot 进入 retired，可被复用；接收 slot 由调用方 `repost`。
- **失败响应消费后进入 `retryable`**——保留 request_id 注册，允许 stage2 用相同 ID 重发。注释（147–150）解释了为什么 `retryable` 在 `try_take` 里报 `pending` 而不是 `stale`：如果报 stale，stage2 会 cancel 项再重新 register，但 cancel 一个 retryable 项会把它 retire（见 `cancel` 170–185），导致后续 register 失败，造成永久中毒。
- **`pending` 和 `retryable` 都报 pending**——stage2 协程会继续轮询。

`mark_retryable`（190–211）允许把 `retired` 状态的项复活为 `retryable`，用于 stage2 拿到响应描述符但 payload 校验失败时（如 candidate count 越界），让相同 ID 的重发能继续匹配。

`cancel`（170–185）取消请求，如果是 `complete` 状态还要返回接收描述符让调用方 repost slot。`drain_completed`（213–225）用于 shutdown，把所有 complete 项的描述符取出来供 `stop_peer_reverse_update_runtime` repost。

### 24.6.2 `PeerRequestDeduplicator`：接收端去重表

`async_response.hh:368-568` 是接收端的对偶——以 `(source_shard, request_id)` 为键，缓存已完成请求的响应，使重试不重复执行。注释（364–367）：

> Bounded receiver-side de-duplication for retries that reuse a request ID. Reverse/cleanup completions replay their cached fixed-size response. Stitch search completions are read-only and may be recomputed after completion; concurrent duplicates are still coalesced while the first search runs.

状态四态（467–472）：`empty / inflight / complete / retired`。`Slot` 比 response registry 多了 `last_used`（用于 LRU 驱逐）和完整的请求元数据（`type/item_count/reserved`）。

`begin`（375–428）是核心：

```cpp
PeerRequestDecision begin(u32 source_shard, const PeerRpcHeader& request, bool response_replayable) {
  std::lock_guard<std::mutex> lock(mutex_);
  Lookup lookup = find_locked(source_shard, request.request_id);
  if (lookup.found != npos) {
    Slot& slot = slots_[lookup.found];
    if (!metadata_matches(slot, source_shard, request)) return {.action = PeerRequestAction::conflict};
    if (slot.state == State::inflight) return {.action = PeerRequestAction::duplicate_inflight};
    if (slot.state == State::complete && response_replayable) {
      return {.action = PeerRequestAction::replay, .response = slot.response};
    }
    if (slot.state == State::complete) {
      slot.state = State::inflight; slot.last_used = ++clock_;
      return {.action = PeerRequestAction::execute};
    }
    if (slot.state == State::retired) {
      slot.response = {}; slot.last_used = ++clock_; slot.state = State::inflight; ++size_;
      return {.action = PeerRequestAction::execute};
    }
  }
  if (lookup.insertion == npos || size_ == capacity_) {
    evict_oldest_complete_locked();
    lookup = find_locked(source_shard, request.request_id);
  }
  if (lookup.insertion == npos || size_ == capacity_) return {.action = PeerRequestAction::full};
  // ... 新建 inflight 项 ...
}
```

关键分支：

- **`inflight`**：同一请求正在执行，返回 `duplicate_inflight`，调用方忽略（不发响应）。
- **`complete` + `response_replayable`**：返回 `replay` + 缓存响应，调用方直接回放。reverse/cleanup 走这条。
- **`complete` + 不可回放**（stitch）：把状态改回 `inflight` 重新执行。这是 stitch 重试的处理——不回放旧响应（可能已经过期），重新计算。
- **`retired`**：复活为 `inflight` 重新执行。
- 表满时 `evict_oldest_complete_locked`（527–536）驱逐最久未用的 complete 项，再试一次。

`complete`（430–452）在 worker 执行完毕后调用：

```cpp
void complete(u32 source_shard, const PeerRpcHeader& request, const PeerRpcHeader& response) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Lookup lookup = find_locked(source_shard, request.request_id);
  if (lookup.found == npos) return;
  Slot& slot = slots_[lookup.found];
  if (slot.state != State::inflight || !metadata_matches(slot, source_shard, request)) return;
  // A failed/overloaded operation is retryable with the same ID. Successful
  // operations remain cached so a lost ACK never causes a second apply.
  if (response.status != static_cast<u32>(InsertStatus::ok)) {
    retire_locked(slot);
    return;
  }
  slot.response = response;
  slot.last_used = ++clock_;
  slot.state = State::complete;
}
```

关键策略（注释 443–444）：**成功响应缓存，失败响应不缓存**。成功的图操作必须缓存，否则 ACK 丢失后重试会重复应用（虽然 `apply_local_reverse_updates_batched` 可能幂等，但每次都要重新加锁改图，浪费且可能违反一致性）。失败响应不缓存——失败本身就是可重试的，下次重试重新执行即可。

`abandon`（454–464）在入队失败或 stitch 完成时调用，把 `inflight` 项 retire，让重试能重新 execute。

`evict_oldest_complete_locked`（527–536）是简单的线性扫描找 `last_used` 最小的 complete 项驱逐——容量通常很大（1024+），驱逐是罕见事件，O(n) 扫描可接受。

---

## 24.7 `client_requests.cc`：客户端侧请求构造

`client_requests.cc` 是从本节点主动向 peer 发请求的全部入口。按调用模式分四类。

### 24.7.1 同步轮询路径：`pump_peer_rpcs` 与 `handle_peer_rpc_requests`

`client_requests.cc:3-10` 是 `handle_peer_rpc_requests`——简单遍历 `requests` 调 `handle_peer_rpc_request`。

`pump_peer_rpcs_locked`（12–72）是同步轮询 recv CQ 的版本，供未启动进度线程时（或需要同步处理时）使用。结构与 `peer_rpc_progress_loop` 类似但更简单：不走去重表，直接把请求压入 `requests` vector，响应直接写入 `peer_rpc_responses_`/`peer_rpc_response_payloads_`。

`pump_peer_rpcs`（74–85）是包装：`wait_for_event` 决定用 `lock` 还是 `try_lock`，调 `pump_peer_rpcs_locked` 后再 `handle_peer_rpc_requests`。这条路径主要用于初始化阶段或维护任务的同步等待。

### 24.7.2 同步等待响应：`wait_for_peer_reverse_update_response`

`client_requests.cc:87-132` 是同步直发请求的响应等待：

```cpp
bool MemoryNode::wait_for_peer_reverse_update_response(u64 request_id, u32 target_shard,
                                           u32 item_count, PeerRpcType response_type,
                                           const Configuration& config) {
  const auto wait_started = std::chrono::steady_clock::now();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  std::unique_lock<std::mutex> lock(peer_rpc_mutex_);
  for (;;) {
    const auto it = peer_rpc_responses_.find(request_id);
    if (it != peer_rpc_responses_.end()) {
      const auto& header = it->second;
      const bool success = header.magic == kPeerRpcMagic && header.version == kPeerRpcVersion &&
                           header.type == static_cast<u32>(response_type) &&
                           header.source_shard == target_shard &&
                           header.item_count == item_count &&
                           header.status == static_cast<u32>(InsertStatus::ok);
      peer_rpc_responses_.erase(it);
      peer_rpc_pending_responses_.erase(request_id);
      lock.unlock();
      log_slow_peer_reverse_update_response(wait_started, request_id, target_shard, item_count, success);
      return success;
    }
    if (peer_rpc_responses_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      peer_rpc_pending_responses_.erase(request_id);
      peer_rpc_responses_.erase(request_id);
      peer_rpc_response_payloads_.erase(request_id);
      // ... 慢日志 ...
      return false;
    }
  }
}
```

要点：

- **deadline = now + `storage_owner_rpc_timeout_ms`**（默认 30s，`configuration.hh:73`）。这是 `rpc_timeout_ms` 在 peer RPC 里的实际作用点。
- **响应校验**：除了在 `peer_rpc_responses_` 里找到 `request_id`，还要验证 magic/version/type/source_shard/item_count/status 全部匹配。任一不符视为失败。
- **超时清理**：timeout 时把 `request_id` 从 `peer_rpc_pending_responses_` 和三个 map 里全删，防止后续乱序到达的响应污染状态。
- **慢日志**：`log_slow_peer_reverse_update_response`（547–571）在等待 >1s 时打前 16 次。

注意 `peer_rpc_pending_responses_` 是 `unordered_set<u64>`（`memory_node.hh:592`），只存 `request_id`——它是"我在等谁"的集合。CQ 进度线程收到响应时只在 `request_id` 在这个集合里时才接受，避免被无关响应污染。

### 24.7.3 同步直发：`send_peer_op_batch_direct` 与 fan-out

`client_requests.cc:367-436` 是同步直发批量操作：

```cpp
bool MemoryNode::send_peer_op_batch_direct(u32 target_shard,
                                      const vec<ReverseUpdateOp>& ops,
                                      PeerRpcType rpc_type, bool wait_for_response,
                                      const Configuration& config) {
  if (ops.empty()) return true;
  const u64 max_items_u64 = std::max<u64>(1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(max_items_u64 <= std::numeric_limits<u32>::max(), ...);
  const u32 max_items = static_cast<u32>(max_items_u64);
  for (size_t begin = 0; begin < ops.size(); begin += max_items) {
    const u32 item_count = static_cast<u32>(std::min<size_t>(ops.size() - begin, max_items));
    const size_t bytes = reverse_update_request_bytes(item_count);
    lib_assert(bytes <= peer_rpc_runtime_.message_bytes, ...);
    vec<byte_t> message(bytes);
    auto* header = reinterpret_cast<PeerRpcHeader*>(message.data());
    header->magic = kPeerRpcMagic; header->version = kPeerRpcVersion;
    header->type = static_cast<u32>(rpc_type);
    header->source_shard = storage_id_;
    header->item_count = item_count;
    header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (!wait_for_response) {
      header->reserved |= kPeerRpcFlagNoResponse;
    } else {
      lib_assert(rpc_type == PeerRpcType::reverse_update_request ||
                   rpc_type == PeerRpcType::cleanup_deleted_request, ...);
      std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
      peer_rpc_pending_responses_.insert(header->request_id);
    }
    auto* payload_ops = reverse_update_ops(message.data());
    std::memcpy(payload_ops, ops.data() + begin, item_count * sizeof(ReverseUpdateOp));
    const auto send_started = std::chrono::steady_clock::now();
    send_peer_rpc_message(target_shard, message.data(), bytes);
    // ... 慢日志 ...
    if (wait_for_response) {
      const auto response_type = rpc_type == PeerRpcType::cleanup_deleted_request
        ? PeerRpcType::cleanup_deleted_response : PeerRpcType::reverse_update_response;
      if (!wait_for_peer_reverse_update_response(header->request_id, target_shard, item_count, response_type, config)) {
        return false;
      }
    }
  }
  return true;
}
```

流程：

1. 按 `max_items = R * batch_max` 分片（wire 上限）。
2. 每片构造一个 `message`，填头 + 拷贝 ops。
3. `wait_for_response` 决定是否置 `kPeerRpcFlagNoResponse`。要等响应时先在 `peer_rpc_pending_responses_` 注册 `request_id`（让 CQ 进度线程知道要接这个响应）。
4. `send_peer_rpc_message` 同步发送（阻塞到发送完成）。
5. 要等响应时调 `wait_for_peer_reverse_update_response`。

`send_reverse_update_batch_direct`（438–447）是 `rpc_type=reverse_update_request` 的便捷封装。

`send_reverse_update_batch`（449–456）根据 `storage_owner_reverse_mode`（"async"/"sync"）选择 outbox 入队或同步直发：

```cpp
bool MemoryNode::send_reverse_update_batch(u32 target_shard, const vec<ReverseUpdateOp>& ops, const Configuration& config) {
  if (config.storage_owner_reverse_mode == "async") {
    return enqueue_reverse_update_batch(target_shard, ops, config);
  }
  return send_reverse_update_batch_direct(target_shard, ops, true, config);
}
```

这是第 26 课 `wire_protocol.cc:465` 和 `batch_execution.cc:132` 调用的入口——在线插入路径的反向边派发。

`send_peer_op_fanout_and_wait`（468–535）是多目标 fan-out：先对所有 target_shard 各自发送（注册 pending），再统一等待所有响应。`send_reverse_update_fanout_and_wait` 和 `send_cleanup_deleted_fanout_and_wait` 是它的两个特化。这是"先全部发出再统一等"的批量化模式，比"发一个等一个"省 RTT。

### 24.7.4 outbox 异步模式：`enqueue_reverse_update_batch`

`client_requests.cc:330-365` 把出站任务入队，由 `peer_reverse_outgoing_loop` 线程消费：

```cpp
bool MemoryNode::enqueue_reverse_update_batch(u32 target_shard, const vec<ReverseUpdateOp>& ops, const Configuration& config) {
  if (ops.empty()) return true;
  const u64 max_items_u64 = std::max<u64>(1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(max_items_u64 <= std::numeric_limits<u32>::max(), ...);
  const size_t max_items = static_cast<size_t>(max_items_u64);
  for (size_t begin = 0; begin < ops.size(); begin += max_items) {
    const size_t count = std::min(max_items, ops.size() - begin);
    std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_);
    peer_reverse_outgoing_cv_.wait(lock, [&]() {
      return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
             peer_reverse_outgoing_.size() < peer_reverse_outgoing_queue_limit_;
    });
    if (peer_reverse_shutdown_.load(std::memory_order_acquire)) return false;
    PeerReverseOutgoingTask task;
    task.target_shard = target_shard;
    task.ops.assign(ops.begin() + static_cast<std::ptrdiff_t>(begin),
                    ops.begin() + static_cast<std::ptrdiff_t>(begin + count));
    task.queued_at = std::chrono::steady_clock::now();
    peer_reverse_outgoing_.push_back(std::move(task));
    lock.unlock();
    peer_reverse_outgoing_cv_.notify_one();
  }
  return true;
}
```

要点：

- 按 `max_items` 分片入队。
- **背压**：队列满时 `wait` 在 cv 上，直到有空间或 shutdown。这与 `enqueue_peer_reverse_update_task` 的"满了返回 false"不同——outbox 模式承诺入队成功（调用方不需要重试），代价是调用方可能阻塞。
- 注释（352–354）强调："Allocate/copy only after a bounded queue slot is owned"——先拿到队列槽位再拷贝 ops，避免拷贝完发现队列满又丢弃的浪费。

### 24.7.5 异步 post：`post_stitch_search_request_async` 与 `post_peer_op_batch_async`

这两个是 stage2 协程使用的异步入口（第 25 课 `worker.cc` 调用）。

`post_stitch_search_request_async`（142–209）：

```cpp
bool MemoryNode::post_stitch_search_request_async(u32 target_shard, const vec<NodeSnapshot>& targets,
    u64 request_id, u32& item_count, const Configuration& config) {
  item_count = 0;
  if (targets.empty() || target_shard == storage_id_) return true;
  if (peer_async_responses_ == nullptr || target_shard >= num_storage_nodes_ ||
      request_id == 0 || targets.size() > config.storage_owner_batch_max) return false;
  item_count = static_cast<u32>(targets.size());
  const size_t bytes = stitch_search_request_bytes(item_count);
  if (bytes > peer_rpc_runtime_.message_bytes) return false;
  for (const NodeSnapshot& target : targets) {
    if (target.vector_data.size() < VamanaNode::vector_bytes()) return false;
  }
  const auto registration = peer_async_responses_->register_send_attempt(
    request_id, target_shard, PeerRpcType::stitch_search_response, item_count);
  if (registration == PeerResponseRegistration::already_complete) return true;
  if (registration != PeerResponseRegistration::registered &&
      registration != PeerResponseRegistration::retry) return false;
  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(target_shard, PeerRpcSendClass::stitch_search, slot_id)) return false;
  // ... 构造 message ...
  post_peer_rpc_send_slot(target_shard, slot_id, bytes);
  return true;
}
```

与同步直发的关键差别：

1. **`request_id` 由调用方传入**（来自 `allocate_peer_request_id`），不是函数内部分配——stage2 协程需要用同一个 ID 注册超时跟踪器、重发、cancel。
2. **`register_send_attempt` 先于 `try_acquire_peer_rpc_send_slot`**：先在响应登记表占位，再申请发送 slot。如果 `already_complete`（响应已到）直接返回 true 不发送；如果 `registered`/`retry` 才真发。
3. **发送 slot 拿不到时返回 false**——调用方会 `mark_retry` 让 stage2 调度器稍后重试相同 ID。
4. **`post_peer_rpc_send_slot` 异步发送**——不等完成，发送 slot 由 `handle_peer_send_completion` 自动释放。

`post_peer_op_batch_async`（211–274）结构类似，但用 `PeerRpcSendClass::graph_update` lane，且响应类型根据请求类型推导（cleanup 请求 → cleanup 响应，否则 → reverse_update 响应）。

### 24.7.6 异步响应消费：`try_consume_peer_rpc_response`

`client_requests.cc:276-310` 是 stage2 协程消费响应的入口：

```cpp
MemoryNode::TryPeerResponse MemoryNode::try_consume_peer_rpc_response(
    u64 request_id, u32 expected_shard, PeerRpcType expected_type, u32 expected_item_count,
    PeerRpcHeader& header, vec<byte_t>& payload) {
  if (peer_async_responses_ == nullptr) return TryPeerResponse::stale;
  memory_node_detail::PeerResponseDescriptor response;
  const TryPeerResponse result = peer_async_responses_->try_take(
    request_id, expected_shard, expected_type, expected_item_count, response);
  if (result == TryPeerResponse::pending || result == TryPeerResponse::stale) return result;
  header = response.header;
  const bool valid_descriptor = response.peer_id < num_storage_nodes_ &&
    response.receive_slot < peer_rpc_runtime_.recv_slots_per_peer &&
    response.bytes >= sizeof(PeerRpcHeader) && response.bytes <= peer_rpc_runtime_.message_bytes;
  if (valid_descriptor) {
    const size_t offset = peer_rpc_receive_offset(response.peer_id, response.receive_slot);
    const byte_t* source = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
    payload.assign(source, source + response.bytes);
  } else {
    payload.clear();
    (void)peer_async_responses_->mark_retryable(request_id, expected_shard, expected_type, expected_item_count);
  }
  repost_peer_rpc_receive(response.peer_id, response.receive_slot);
  return valid_descriptor ? result : TryPeerResponse::failure;
}
```

要点：

- 调 `try_take` 从登记表取出响应描述符。
- `pending`/`stale` 直接返回，不 repost（pending 时 slot 还被登记表持有；stale 时本来就没持有）。
- 拿到描述符后**拷贝 payload**——这一刻 payload 还在 RDMA recv buffer 里，必须拷出来才能 `repost_peer_rpc_receive` 释放 slot 给下一帧用。
- 描述符本身不合理（peer_id/slot/bytes 越界）时不拷贝，`mark_retryable` 让相同 ID 重发，返回 `failure`。
- 无论成功失败，只要拿了描述符就要 `repost_peer_rpc_receive`——slot 的所有权从登记表转回 RDMA recv WR。

`rearm_peer_rpc_response`（312–320）调 `mark_retryable`，用于 stage2 拿到响应但 payload 解析失败（如 candidate count 不符）时让相同 ID 的重发能继续匹配。`cancel_peer_rpc_response`（322–328）调 `cancel`，如果项是 complete 状态还要 repost 接收 slot。

### 24.7.7 `allocate_peer_request_id`

`client_requests.cc:134-140` 极简：

```cpp
u64 MemoryNode::allocate_peer_request_id() {
  for (;;) {
    const u64 request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (request_id != 0) return request_id;
  }
}
```

`next_peer_request_id_` 初值 1（`memory_node.hh:612`），原子递增。跳过 0——0 在登记表里是非法值（`register_request_locked` 第 259 行 `if (request_id == 0) return conflict`）。`for(;;)` 保证即使 fetch_add 回绕到 0 也能继续取下一个非零值。

---

## 24.8 关键数据结构与流程图

### 24.8.1 buffer 布局

```
peer_rpc_runtime_.buffer (单块 HugePage, 单个 LocalMemoryRegion)
┌────────────────────────────────────────────────────────────────────────────┐
│ 接收区  [0, sync_send_offset)                                              │
│   peer 0 slot 0 | peer 0 slot 1 | ... | peer 0 slot (recv-1)              │
│   peer 1 slot 0 | ...                                                       │
│   ...                                                                       │
│   peer (N-1) slot (recv-1)                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ 同步发送区  [sync_send_offset, async_send_offset)                          │
│   peer 0 (1 slot) | peer 1 (1 slot) | ... | peer (N-1) (1 slot)            │
├────────────────────────────────────────────────────────────────────────────┤
│ 异步发送区  [async_send_offset, end)                                       │
│   peer 0 slot 0 | peer 0 slot 1 | ... | peer 0 slot (send-1)              │
│   peer 1 slot 0 | ...                                                       │
│   ...                                                                       │
│   peer (N-1) slot (send-1)                                                  │
└────────────────────────────────────────────────────────────────────────────┘
每个槽位 = message_bytes (align_up(max(所有报文类型最大长度)))
```

### 24.8.2 线程模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    peer_rpc_progress_thread                      │
│  (单线程, current_peer_rpc_progress_thread_ = true)              │
│                                                                  │
│  loop:                                                           │
│    poll_peer_send_cq()  ── 处理发送完成, 释放 slot                │
│    poll recv CQ                                                  │
│    ├─ reverse_update_request  → dedup.begin → enqueue task       │
│    │                             或 dedup.replay → enqueue ACK   │
│    ├─ cleanup_deleted_request → (同上)                           │
│    ├─ stitch_search_request   → dedup.begin → enqueue task       │
│    ├─ *_response              → async_registry.try_deliver       │
│    │                             (hold slot) 或 sync map         │
│    └─ repost_receive_slot (除非 hold)                            │
└─────────────────────────────────────────────────────────────────┘
        │ enqueue                    │ try_deliver (hold slot)
        ▼                            ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│ peer_reverse_workers │   │ stage2 协程 (maintenance/worker.cc)    │
│  (N 个, 合批 coalesce)│   │  try_consume_peer_rpc_response        │
│  apply_local_reverse │   │   → 拷贝 payload + repost slot        │
│  → dedup.complete    │   │  rearm / cancel                       │
│  → enqueue ACK       │   └──────────────────────────────────────┘
└──────────┬───────────┘
           │ try_push
           ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│ peer_reverse_responses   │───▶│ peer_reverse_response_thread     │
│ (bounded::Queue)         │    │  send_peer_rpc_message (同步)    │
└──────────────────────────┘    └─────────────────────────────────┘

┌──────────────────────────┐    ┌─────────────────────────────────┐
│ peer_reverse_outgoing    │───▶│ peer_reverse_outgoing_thread     │
│ (deque, per-peer 合批)    │    │  send_peer_op_batch_direct       │
│  (async mode 入口)        │    │   (wait_for_response=false)      │
└──────────────────────────┘    └─────────────────────────────────┘

┌──────────────────────┐
│ peer_stitch_workers  │  handle_peer_stitch_search_request
│  (M 个, 不合批)       │  → send_peer_rpc_message (同步, 在 worker 内)
│  → dedup.abandon     │
└──────────────────────┘
```

### 24.8.3 跨分片 mutation 时序图

```
计算节点                 owner 分片 (storage_id_ = S0)              peer 分片 (storage_id_ = S1)
   │                            │                                          │
   │ mutation RPC (stage1)      │                                          │
   ├───────────────────────────▶│                                          │
   │                            │ stage1 完成, 产生反向边                   │
   │                            │ remote_ops_by_peer[S1] = [...]           │
   │                            │                                          │
   │                            │ ─── stage2 同步路径 (wire_protocol) ───  │
   │                            │ send_reverse_update_batch(S1, ops)       │
   │                            │   mode="sync": send_peer_op_batch_direct │
   │                            │   ├─ register request_id in pending set  │
   │                            │   ├─ send_peer_rpc_message (阻塞)        │
   │                            │   ├─────────────────────────────────────▶│ recv CQ
   │                            │   │                                       │ progress_loop:
   │                            │   │                                       │ dedup.begin → execute
   │                            │   │                                       │ enqueue reverse task
   │                            │   │                                       │
   │                            │   │                                       │ reverse_worker:
   │                            │   │                                       │ apply_local_reverse_updates_batched
   │                            │   │                                       │ dedup.complete (缓存 ACK)
   │                            │   │                                       │ enqueue ACK → response_thread
   │                            │   │                                       │
   │                            │   │                                       │ response_thread:
   │                            │   │                                       │ send_peer_rpc_message (ACK)
   │                            │   │◀─────────────────────────────────────│
   │                            │   │                                       │
   │                            │   │ progress_loop:                        │
   │                            │   │   try_deliver 失败 (同步 pending set) │
   │                            │   │   → peer_rpc_responses_[id] = header  │
   │                            │   │   notify peer_rpc_responses_cv_       │
   │                            │   │                                       │
   │                            │   │ wait_for_peer_reverse_update_response │
   │                            │   │   (从 cv 醒来, 校验, 返回 success)     │
   │                            │◀──┘                                       │
   │                            │                                          │
   │                            │ ─── stage2 异步路径 (maintenance worker) │
   │                            │ post_peer_op_batch_async(S1, ops, id)    │
   │                            │   ├─ register_send_attempt (registry)    │
   │                            │   ├─ try_acquire send slot (graph_update)│
   │                            │   ├─ post_peer_rpc_send_slot (异步)      │
   │                            │   ├─────────────────────────────────────▶│ recv CQ
   │                            │   │                                       │ (同上处理, 回 ACK)
   │                            │   │◀─────────────────────────────────────│
   │                            │   │                                       │
   │                            │   │ progress_loop:                        │
   │                            │   │   try_deliver 成功 (hold slot)        │
   │                            │   │   notify maintenance_cv_              │
   │                            │   │                                       │
   │                            │ stage2 协程轮询:                          │
   │                            │   try_consume_peer_rpc_response(id)      │
   │                            │   ├─ try_take → success                  │
   │                            │   ├─ 拷贝 payload                        │
   │                            │   └─ repost_peer_rpc_receive (释放 slot) │
   │                            │                                          │
   │ mutation response          │                                          │
   │◀───────────────────────────│                                          │
```

### 24.8.4 ACK 丢失与重试时序

```
owner 分片                                    peer 分片
   │                                             │
   │ reverse_update_request (request_id=42)      │
   ├────────────────────────────────────────────▶│ dedup.begin → execute (inflight)
   │                                             │ reverse_worker: apply, complete (缓存 ACK)
   │                                             │ enqueue ACK
   │                                             │
   │                          ACK 丢失 ✗         │ response_thread: send ACK
   │◀ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
   │                                             │
   │  rpc_timeout_ms 到期, 重试相同 id=42         │
   ├────────────────────────────────────────────▶│ dedup.begin → replay (返回缓存 ACK)
   │                                             │   不重新执行图操作!
   │                                             │ enqueue 缓存的 ACK
   │◀────────────────────────────────────────────│
   │ 成功                                        │
```

---

## 24.9 与其他模块的关系

- **第 3 课（并发原语与协程）**：`bounded::Queue`、`StorageOwnerThread` 的协程框架、cv/mutex 使用模式都源自第 3 课。peer RPC 的 reverse worker 各跑 1 个协程，但实际未在 peer RPC 代码里 `co_await`——协程基础设施是为 stage2 维护协程（第 25 课）准备的，reverse worker 只复用 `StorageOwnerThread` 的 scratch 分配。
- **第 8 课（storage_owner_protocol wire 层）**：`PeerRpcHeader`、`PeerRpcType`、`ReverseUpdateOp`、`StitchSearchItem/Candidate`、所有 `*_bytes()` 函数都在 `service/storage_owner_protocol.hh`。peer RPC 是 wire 层的"内部协议"使用者，与计算节点↔存储节点的 `InsertBatchRequestHeader`/`MutationBatchRequestHeader` 是平级的另一套 wire 格式。
- **第 16 课（存储回收 RCU）**：stitch 响应里的 `generation` 字段、`local_live_vector`/`local_node_ptr` 的有效性检查都依赖 RCU 回收机制。peer RPC 在 `apply_peer_reverse_update_tasks` 里通过 `valid_local_storage_node_pointer` 拦截过期指针。
- **第 23 课（存储节点主体/peer RDMA）**：`peer_context_`、`peer_control_qp`、`poll_peer_send_cq`、`next_peer_sync_wr_id`、`wait_peer_sync_completion`、`handle_peer_send_completion` 都在 `peer_rdma.cc`。peer RPC 完全建立在第 23 课的 QP 与 CQ 基础设施之上，自身只管 buffer 布局与协议解析。
- **第 25 课（索引访问/图修改）**：`apply_local_reverse_updates_batched`、`remove_local_neighbors_batched`、`partition_local_search_candidates`、`storage_owner_route_entries`、`local_live_vector`、`local_node_ptr` 都在 `storage_owner_index/`。peer RPC 是这些图修改函数的"远程触发器"。
- **第 25 课（stage2 维护协程）**：`storage_owner_maintenance/worker.cc` 是 `post_stitch_search_request_async`、`post_peer_op_batch_async`、`try_consume_peer_rpc_response`、`rearm_peer_rpc_response`、`cancel_peer_rpc_response` 的主要调用方。stage2 协程通过 `Stage2RequestTracker`（独立的超时跟踪器）与 `PeerAsyncResponseRegistry` 协同——前者管"何时重发"，后者管"响应匹配"。
- **第 26 课（wire protocol）**：`storage_owner_runtime/wire_protocol.cc:465` 和 `batch_execution.cc:132` 在在线插入路径里调用 `send_reverse_update_batch`。第 26 课的 `storage_owner_runtime` 是"计算节点→owner 分片"的 wire 协议，peer RPC 则是"owner 分片→peer 分片"的内部协议，两者接力完成一次跨分片 mutation。

分层关系总结：

```
计算节点 mutation
    │
    │ (第 26 课 wire protocol: MutationBatchRequestHeader)
    ▼
owner 分片 storage_owner_runtime/wire_protocol.cc
    │
    │ stage1: 本地 prune + 写节点
    │ stage2: 反向边派发
    │
    ├── sync mode: send_reverse_update_batch → send_peer_op_batch_direct
    │                                       → wait_for_peer_reverse_update_response
    │   (本课 client_requests.cc)
    │
    └── async mode: enqueue_reverse_update_batch → peer_reverse_outgoing_loop
                                                 → send_peer_op_batch_direct(no_wait)
        (本课 workers.cc)                        (本课 client_requests.cc)

    维护任务 (第 25 课 stage2):
        post_stitch_search_request_async  ─┐
        post_peer_op_batch_async          ─┤  (本课 client_requests.cc)
        try_consume_peer_rpc_response     ─┤
        rearm/cancel_peer_rpc_response    ─┘
                    │
                    │ (本课 wire: PeerRpcHeader + payload)
                    ▼
        peer 分片 peer_rpc_progress_loop (workers.cc)
                    │
                    │ dedup.begin → enqueue task
                    ▼
        reverse_worker / stitch_worker (workers.cc)
                    │
                    │ apply_local_reverse_updates_batched (第 25 课)
                    │ 或 partition_local_search_candidates (第 25 课)
                    ▼
        response_thread → send ACK
                    │
                    │ (本课 wire: PeerRpcHeader response)
                    ▼
        owner 分片 progress_loop → try_deliver / sync map
                    │
                    ▼
        stage2 协程消费 / wait_for_peer_reverse_update_response 返回
```

---

## 24.10 小结

peer RPC 是 dvstor 存算分离架构里"跨分片图一致性"的全部实现。它的设计有三个值得记住的特点：

1. **单 buffer 三段布局 + 编码 WR ID**：一块 HugePage MR 切成接收/同步发送/异步发送三段，`(peer_id, slot_id)` 压进 64 位 WR ID，CQ 出队时 `decode_64bit` 直接定位 buffer 偏移。这让 peer RPC 在不依赖额外内存注册的前提下支持多 peer 并发收发。

2. **两条可靠性路径**：
   - 同步路径（`send_peer_op_batch_direct` + `wait_for_peer_reverse_update_response`）用于在线插入的 stage2，靠 `peer_rpc_pending_responses_` set + `peer_rpc_responses_cv_` + `rpc_timeout_ms` 实现请求-响应匹配与超时。
   - 异步路径（`post_*_async` + `PeerAsyncResponseRegistry`）用于 stage2 维护协程，靠定容开地址哈希表 + slot 占用转移实现零拷贝等待，超时与重发由 stage2 调度器（第 25 课 `Stage2RequestTracker`）管理。
   两条路径共享同一套 CQ 进度线程，靠"异步优先、同步兜底"的分派逻辑统一处理响应。

3. **接收端去重表保证 ACK 丢失安全**：`PeerRequestDeduplicator` 缓存成功响应，发送端用相同 `request_id` 重试时接收端直接回放缓存响应，不重复执行图操作。失败响应不缓存，重试重新执行（失败操作幂等）。stitch 响应因 payload 太大不缓存，但并发重复请求在执行期间被 `duplicate_inflight` 合并，执行完毕后 `abandon` 让重试重新计算（只读安全）。

这三点合起来，让 peer RPC 在 RDMA 不可靠丢包（虽然 verbs SEND 本身可靠，但 ACK 发送可能因 buffer 压力被丢）和队列背压下都能保持图的一致性——这是存算分离向量索引能在线更新的基础。
