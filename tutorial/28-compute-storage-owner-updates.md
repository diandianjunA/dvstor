# 第 28 课 计算侧 storage owner 更新

> 课号 28 / 30 · 课题：计算侧 storage owner 更新
> 涉及代码：`src/service/compute_service/storage_owner/{lifecycle.cc, sender.cc, completion.cc, public_mutations.cc, response_validation.hh}`，辅以 `src/service/storage_owner_protocol.hh`、`src/service/storage_owner_client_helpers.hh`、`src/gpu_search/persistent_engine.cc`、`src/gpu_search/persistent_engine/delta_publication.cc`、`src/service/compute_service/index_commands.cc`、`src/service/compute_service.hh`。

## 28.1 本课目标与涉及文件

dvstor 是一个 GPU 中心化的存算分离向量检索系统。客户端发起的 `insert` / `upsert` / `erase` 既不能直接落到 GPU 上，也不能直接写到存储节点，必须由**计算节点**作为协调者，完成两件事：

1. **stage1**：通过 RDMA `SEND` 把一批 mutation 发给对应 `storage owner`，让 owner 在 owner-memory（owner 节点的本地内存）里完成实际的图修改、反向索引更新、PQ slot 分配等动作，并返回 stage1 ACK（包含每个 item 的状态、新/旧 `RemotePtr`、generation、维护序列号、被反向失效的图节点列表，以及细粒度 breakdown 计数）。
2. **GPU 可见性发布**：在 stage1 ACK 返回后，计算节点把 ACK 中描述的"已 commit"mutation 翻译成 `gpu_search::DeltaMutation`，按可见性窗口合并、按 generation 去重，再交给 `PersistentSearchEngine::publish_mutations` 上传到 GPU delta 表，最终通过 `DeltaCoordinator` 的 epoch 发布让查询路径"看得见"这些更新。

本课围绕"计算侧如何把客户端 mutation 一路推进到 GPU 可见"这一条端到端链路展开，把计算侧 storage owner 子系统拆成五个文件逐一精读：

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `lifecycle.cc` | ~189 | 启动/停止 storage owner 客户端运行时；为每个 owner 预分配请求/响应缓冲、MR、RPC slot、task pool；拉起 progress / completion 两条线程 |
| `sender.cc` | ~231 | progress 线程主循环：轮询 send/recv CQ、按 owner 轮转 drain 提交队列、组 batch、`post_send`；以及 slot 回收 |
| `completion.cc` | ~438 | completion 线程主循环：响应匹配、响应校验、commit 每个 item、调 `publish_compute_side_id`、构造 `DeltaMutation`、发布到 GPU、回收容量 |
| `public_mutations.cc` | ~155 | 对外 `insert/upsert/erase` 入口；路由到 owner、容量预约、task 入队、同步等待完成 |
| `response_validation.hh` | ~38 | 响应头部字段校验（magic/owner/item_count/batch_id/字节数） |

辅助文件：

- `storage_owner_protocol.hh`：stage1 请求/响应 wire 格式（与第 8 课一脉相承，与第 26 课 wire protocol 对应）。
- `storage_owner_client_helpers.hh`：WR ID 编解码、breakdown 拆分助手。
- `persistent_engine.cc` 的 `publish_mutations / reserve_mutation_capacity / release_mutation_capacity / mark_committed_mutation_gap`：GPU 侧发布/容量控制。
- `delta_publication.cc`：GPU 侧真正上传 delta 记录、提交 CUDA 命令的实现。
- `index_commands.cc`：`publish_compute_side_id` / `known_storage_owner_for_id` / `claim_storage_owner_for_mutation`，是计算侧的"路由表 + generation 表"。
- `compute_service.hh` 内的 `StorageInsertTask / StorageOwnerRpcSlot / StorageOwnerSenderState / StorageOwnerReadySlot / StorageOwnerReleasedSlot` 结构体定义。

读完后你应当能回答：

- 一个客户端 `insert(vec)` 调用，从同步 API 进入到 GPU 可见，中间经过哪些队列、线程、RDMA 操作？
- 计算侧如何决定一个 ID 应当发给哪个 owner？如何防止"多个计算节点对同一未观察 ID 选出不同 owner"？
- stage1 ACK 的 batch_id 匹配怎么做？为什么需要单独的 `validate_storage_owner_response`？
- "已 commit 但发布到 GPU delta 失败"会发生什么？什么是 `mark_committed_mutation_gap`？
- RPC depth / batch max / task capacity 三层容量是如何联动限制 inflight 的？

## 28.2 关键数据结构（先看 `compute_service.hh`）

在精读实现前，先把 `compute_service.hh:79-142` 中定义的几个结构体摆出来，它们贯穿全部五个文件。

### `StorageInsertTask`（`compute_service.hh:79-85`）

```cpp
struct StorageInsertTask {
  InsertItem item;
  service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
  u32 completion_id{std::numeric_limits<u32>::max()};
  std::chrono::steady_clock::time_point enqueued_at{};
  std::chrono::steady_clock::time_point sender_dequeued_at{};
};
```

一个 task = 一条 mutation（一个 ID + 一条向量 + 一个 kind）。`completion_id` 是 `bounded::CompletionPool` 的同步凭证，`public_mutations.cc` 调用方就靠它阻塞等待。`enqueued_at` / `sender_dequeued_at` 用于 breakdown，把"入队→出队"和"出队→post_send"两段时间分别记账。

### `StorageOwnerRpcSlot`（`compute_service.hh:87-114`）

```cpp
struct StorageOwnerRpcSlot {
  u32 owner_storage{};
  u32 slot_id{};
  bool in_use{false};
  bool send_done{false};
  bool response_done{false};
  bool results_completed{false};
  bool completion_claimed{false};
  bool response_valid{false};
  u32 response_slot_id{std::numeric_limits<u32>::max()};
  u32 gpu_reserved_items{};
  u32 item_count{};
  u64 batch_id{};
  // ... breakdown 时间戳 ...
  vec<byte_t> request_buffer;
  std::unique_ptr<LocalMemoryRegion> request_region;
  vec<u32> tasks;
  vec<gpu_search::DeltaMutation> publication_mutations;
  vec<u64> publication_invalidated_graph_nodes;
  u32 publication_mutation_count{};
  u32 publication_reserved_items{};
};
```

这是**计算侧对一条 stage1 RPC 在途时的全部状态**。注意几组状态机标志位：

- `in_use`：slot 已被分配给某次 batch（`post_storage_owner_batch` 置 true，`reclaim_storage_owner_slots` 置 false）。
- `send_done`：发送 WR 的 CQE 已收到。
- `response_done`：对应 `batch_id` 的响应已匹配上。
- `completion_claimed`：`queue_storage_owner_completion` 已把它推进 `storage_ready_slots_`，防止重复入队。
- `results_completed`：completion 线程的 `commit_storage_owner_slot` 已结束，slot 可被回收。
- `response_valid`：响应头部字段校验通过。

只有当 `in_use && send_done && response_done && !results_completed && !completion_claimed` 同时成立时，才会被推进 ready 队列（见 `completion.cc:94-106`）。

`request_buffer` 是 wire 协议报文，`request_region` 是它对应的 RDMA MR；`tasks` 是这一批 task_id 列表；`publication_mutations` 是**预分配**的 `DeltaMutation` 数组，stage1 ACK 解析后按 `publication_mutation_count` 填充，再交给 GPU 发布；`publication_invalidated_graph_nodes` 是 owner 返回的"反向失效"图节点列表，发布时一并传给 GPU。

### `StorageOwnerSenderState`（`compute_service.hh:123-131`）

```cpp
struct StorageOwnerSenderState {
  u32 task_capacity{};
  std::unique_ptr<bounded::Queue<u32>> queue;          // 提交队列（task_id）
  std::unique_ptr<bounded::Queue<u32>> free_tasks;      // 空闲 task_id 池
  std::unique_ptr<StorageInsertTask[]> tasks;           // task 容量数组
  vec<StorageOwnerRpcSlot> slots;                       // RPC slot 数组（rpc_depth 个）
  vec<StorageOwnerResponseSlot> response_slots;         // 响应接收 slot
  vec<u32> free_slots;                                  // 空闲 RPC slot 栈
};
```

每个 owner 一份。`queue` 是 `public_mutations.cc` push、`sender.cc` drain 的通道；`free_tasks` / `tasks` 是 task pool；`slots` 是 `rpc_depth` 个发送 slot（每个 slot 持有完整请求缓冲），`response_slots` 是 `rpc_depth` 个接收缓冲。

### `StorageOwnerReadySlot` / `StorageOwnerReleasedSlot`（`compute_service.hh:133-142`）

```cpp
struct StorageOwnerReadySlot { u32 owner_storage; u32 slot_id; };
struct StorageOwnerReleasedSlot { u32 owner_storage; u32 slot_id; u32 response_slot_id; };
```

两条跨线程 handoff 队列的载荷。ready 队列把"send+recv 都完成"的 slot 交给 completion 线程；released 队列把"commit 也做完"的 slot 交回 progress 线程回收。注意 `ReleasedSlot` 多带一个 `response_slot_id`——因为接收缓冲要在 progress 线程里重新 `post_receive`，而不是在 completion 线程里。

### `DeltaMutation`（`gpu_search/delta_index.hh:20-33`）

```cpp
struct DeltaMutation {
  node_t id{};
  service::storage_owner::MutationKind kind{...};
  u32 generation{};
  u64 epoch{};
  u64 remote_node{};
  u64 old_remote_node{};
  u64 anchor_hint{};
  u64 maintenance_sequence{};
  u32 owner_storage{};
  bool durable{};
  std::vector<byte_t> vector;
  std::chrono::steady_clock::time_point enqueued_at{};
};
```

这是 stage1 ACK → GPU 发布之间的统一中间表示。`generation` 来自 owner 响应（见 `MutationResult`）；`remote_node` / `old_remote_node` 来自 `result.new_rptr_raw` / `result.old_rptr_raw`；`enqueued_at` 用 stage1 响应到达时间填充，GPU 发布后会用它计算"可见性延迟"。

## 28.3 `lifecycle.cc`：storage owner 客户端生命周期

`lifecycle.cc` 提供三个函数：`start_storage_insert_runtime` / `stop_storage_insert_runtime` / `release_storage_insert_runtime`，构成"启动 → 停止 → 释放"三段式生命周期。

### 28.3.1 `start_storage_insert_runtime`（`lifecycle.cc:5-123`）

函数一开始是幂等保护：

```cpp
if (!storage_insert_owners_.empty()) return;
```

随后从配置推导四个核心参数：

```cpp
const u32 owner_count = std::max<u32>(1, num_servers_);
const u32 rpc_depth = std::max<u32>(1, config_.storage_owner_rpc_depth);
const size_t request_bytes = std::max(
  service::storage_owner::insert_batch_request_bytes(config_.storage_owner_batch_max),
  service::storage_owner::mutation_batch_request_bytes(config_.storage_owner_batch_max));
const size_t response_bytes =
  service::storage_owner::insert_batch_response_bytes(config_.storage_owner_batch_max);
```

注意 `request_bytes` 取 `insert_batch_request_bytes` 和 `mutation_batch_request_bytes` 的较大值。这两者的区别在 `storage_owner_protocol.hh:180-191`：

```cpp
inline size_t insert_batch_request_bytes(u32 item_count) {
  return sizeof(InsertBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}
inline size_t mutation_batch_request_bytes(u32 item_count) {
  return sizeof(MutationBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(u32) +       // kinds[]
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}
```

mutation 版本多一段 `u32 kinds[]`（每条 mutation 的 `MutationKind`），所以总是不小于 insert 版本。计算侧实际发送时按这一批 task 是否全为 `insert` 来选 magic 和布局（见 28.4.3），但缓冲要按最坏情况预留——这就是 `std::max` 的作用。

接下来是一组 `lib_assert`，确保这些预分配不会撑爆 RDMA verbs 的 SGE 字段（u32）和 send/recv CQ 容量：

```cpp
lib_assert(request_bytes <= std::numeric_limits<u32>::max() &&
             response_bytes <= std::numeric_limits<u32>::max(),
           "storage_owner RPC message is too large for verbs SGEs; "
           "reduce batch size or vector dimension");
const size_t max_inflight = static_cast<size_t>(owner_count) * rpc_depth;
lib_assert(max_inflight <= static_cast<size_t>(config_.max_send_queue_wr),
           "storage_owner RPC depth exceeds compute send CQ capacity");
lib_assert(max_inflight <= static_cast<size_t>(config_.max_recv_queue_wr),
           "storage_owner RPC depth exceeds compute receive CQ capacity");
```

`max_inflight = owner_count * rpc_depth`：每个 owner 同时最多有 `rpc_depth` 条在途 RPC，全部 owner 加起来就是 CQ 上可能同时存在的 WR 数上限。这是"协议并发度"层面的硬上界。

随后是 task 容量推导：

```cpp
const u64 requested_task_capacity = std::max<u64>(
  256,
  static_cast<u64>(rpc_depth) * config_.storage_owner_batch_max * 8);
```

注释明确说"从协议并发度推导，不暴露为 benchmark 旋钮"。`rpc_depth * batch_max * 8` 中，`rpc_depth * batch_max` 是同时在途的 item 数上限，再乘 8 是给"closed-loop writer 突发"留的余量。`task_capacity` 是**每个 owner** 的 task pool 容量；全部 owner 加起来 `total_task_capacity_u64 = owner_count * task_capacity`，作为 `CompletionPool` 和 breakdown 样本数组的容量。

接着初始化跨线程队列：

```cpp
storage_ready_slots_ =
  std::make_unique<bounded::Queue<StorageOwnerReadySlot>>(max_inflight);
storage_released_slots_ =
  std::make_unique<bounded::Queue<StorageOwnerReleasedSlot>>(max_inflight);
storage_completion_pool_ = std::make_unique<bounded::CompletionPool>(
  static_cast<u32>(total_task_capacity_u64));
storage_completion_samples_ =
  std::make_unique<service::breakdown::Sample[]>(total_task_capacity_u64);
```

`storage_ready_slots_` 和 `storage_released_slots_` 的容量都是 `max_inflight`——因为每个在途 RPC slot 至多产生一条 ready 和一条 released，不会超过在途数。`CompletionPool` 和 sample 数组按"全部 task 容量"分配，因为每个 task 都需要一个 completion 凭证和一个 breakdown sample。

下面进入 per-owner 状态构造（`lifecycle.cc:54-97`）：

```cpp
for (u32 owner = 0; owner < owner_count; ++owner) {
  auto state = std::make_unique<StorageOwnerSenderState>();
  state->task_capacity = task_capacity;
  state->queue = std::make_unique<bounded::Queue<u32>>(task_capacity);
  state->free_tasks = std::make_unique<bounded::Queue<u32>>(task_capacity);
  state->tasks = std::make_unique<StorageInsertTask[]>(task_capacity);
  for (u32 task_id = 0; task_id < task_capacity; ++task_id) {
    auto& task = state->tasks[task_id];
    task.item.values.reserve(config_.dim);
    lib_assert(state->free_tasks->try_push(task_id),
               "failed to initialize storage-owner task pool");
  }
```

每个 task 的 `values` 预先 `reserve(dim)`，避免后续 `assign` 时再分配。所有 task_id 一开始进 `free_tasks`。

然后是 RPC slot 构造（`lifecycle.cc:68-85`）：

```cpp
state->slots.resize(rpc_depth);
state->free_slots.reserve(rpc_depth);
for (u32 slot_id = 0; slot_id < rpc_depth; ++slot_id) {
  auto& slot = state->slots[slot_id];
  slot.owner_storage = owner;
  slot.slot_id = slot_id;
  slot.request_buffer.assign(request_bytes, 0);
  slot.request_region = std::make_unique<LocalMemoryRegion>(
    context_, slot.request_buffer.data(), slot.request_buffer.size());
  slot.tasks.reserve(config_.storage_owner_batch_max);
  slot.publication_mutations.resize(config_.storage_owner_batch_max);
  for (auto& mutation : slot.publication_mutations) {
    mutation.vector.resize(VamanaNode::vector_bytes());
  }
  slot.publication_invalidated_graph_nodes.reserve(
    static_cast<size_t>(config_.storage_owner_batch_max) * config_.R);
  state->free_slots.push_back(slot_id);
}
```

每个 slot 拥有：自己的请求缓冲 + MR、预分配到 `batch_max` 的 task_id 列表、预分配到 `batch_max` 的 `DeltaMutation` 数组（且每个 `DeltaMutation::vector` 都按 `VamanaNode::vector_bytes()` 预分配），以及预分配到 `batch_max * R` 的失效图节点列表。**所有这些预分配都是为了在 hot path 上零分配**——stage1 ACK 到达后，completion 线程只做 `memcpy` 进预分配缓冲，不触发任何 `new`。

注意 `publication_invalidated_graph_nodes` 的容量是 `batch_max * R`：因为每个 item 至多在反向索引更新时让 `R` 个图节点失效（`R` 是 Vamana 的度数上限），这与 `response_invalidation_capacity(item_count) = item_count * VamanaNode::R`（`storage_owner_protocol.hh:294-296`）对齐。

响应 slot 构造类似（`lifecycle.cc:86-95`），每个 slot 一份 `response_bytes` 缓冲 + MR。

启动时还要预先 post 所有响应接收（`lifecycle.cc:99-104`）：

```cpp
for (u32 owner = 0; owner < owner_count; ++owner) {
  for (u32 response_slot_id = 0;
       response_slot_id < rpc_depth; ++response_slot_id) {
    post_storage_owner_response_receive(owner, response_slot_id);
  }
}
```

这一步是 RDMA `RECV` 的常规操作：必须先 post receive，对端 `SEND` 来了才有 buffer 接。每个 owner 的 `rpc_depth` 个 response slot 全部 post 上，对应 `max_inflight` 个在途 RPC。

最后拉起两条线程（`lifecycle.cc:106-118`）：

```cpp
storage_insert_completion_thread_ =
  std::thread([this]() { run_storage_insert_completion_loop(); });
storage_insert_progress_thread_ =
  std::thread([this]() { run_storage_insert_progress_loop(); });
if (!config_.disable_thread_pinning) {
  const u32 progress_core = core_assignment_.get_available_core();
  const u32 completion_core = core_assignment_.get_available_core();
  pin_thread(storage_insert_progress_thread_, progress_core);
  pin_thread(storage_insert_completion_thread_, completion_core);
  ...
}
```

注意先起 completion 线程，再起 progress 线程——避免 progress 已经发出 RPC 而 completion 还没就绪。线程绑核由 `core_assignment_` 分配，progress 跑在独立核（CQ 轮询+post_send），completion 跑在另一个核（响应解析+GPU 发布）。

最后的 `print_status` 把架构总结成一句话：

```
storage-owner acknowledgement=owner-memory stage1 publication;
GPU visibility=ordered asynchronous publication;
submission=bounded owner rings; progress=single work-conserving executor
```

四个分号分别点出：stage1 ACK 仅意味着 owner-memory 已落定；GPU 可见性是另一条有序异步发布流；提交走 bounded owner ring（每个 owner 一个有界队列）；progress 是单线程 work-conserving 执行器。

### 28.3.2 `stop_storage_insert_runtime`（`lifecycle.cc:125-163`）

停止流程先置 shutdown 标志，并唤醒所有可能阻塞的队列：

```cpp
storage_insert_shutdown_.store(true, std::memory_order_release);
for (auto& state : storage_insert_owners_) {
  if (state && state->queue) state->queue->notify_all();
  if (state && state->free_tasks) state->free_tasks->notify_all();
}
if (storage_ready_slots_) storage_ready_slots_->notify_all();
if (storage_released_slots_) storage_released_slots_->notify_all();
```

`notify_all` 是 `bounded::Queue` 的接口，唤醒所有在 `pop_wait` / `push_wait` 上阻塞的线程，让它们重新检查 shutdown 标志。

然后 join progress 线程：

```cpp
if (storage_insert_progress_thread_.joinable()) {
  storage_insert_progress_thread_.join();
}
if (storage_ready_slots_) storage_ready_slots_->notify_all();
if (storage_insert_completion_thread_.joinable()) {
  storage_insert_completion_thread_.join();
}
```

progress 线程 join 后再 notify 一次 `storage_ready_slots_`——因为 completion 线程可能在 `pop_wait` 上等 ready slot，而 progress 已经不再生产 ready slot 了。

join 完成后，所有在途和未发送的 task 都要 fail 掉（`lifecycle.cc:142-160`）：

```cpp
for (u32 owner = 0; owner < storage_insert_owners_.size(); ++owner) {
  auto& state = *storage_insert_owners_[owner];
  u32 task_id = 0;
  while (state.queue && state.queue->try_pop(task_id)) {
    vec<u32> failed{task_id};
    fail_storage_owner_tasks(owner, failed);
  }
  for (auto& slot : state.slots) {
    if (slot.in_use && !slot.results_completed) {
      fail_storage_owner_tasks(owner, slot.tasks);
    }
    slot.in_use = false;
    // ... 清空所有状态标志 ...
  }
}
storage_insert_inflight_.store(0, std::memory_order_release);
storage_insert_progress_done_.store(true, std::memory_order_release);
```

`fail_storage_owner_tasks`（见 28.5.6）会把 task 标记为失败、释放 GPU 容量预约、回收 task_id。这一步保证 shutdown 后所有阻塞在 `CompletionPool::wait` 上的调用方都能被唤醒并看到失败结果。

### 28.3.3 `release_storage_insert_runtime`（`lifecycle.cc:165-189`）

`stop_*` 之后还要 `release_*`，差异在于：`stop_*` 只把运行时"停下来"，结构体还在（用于后续 status 查询）；`release_*` 才真正释放 MR、清空 vector、reset unique_ptr。注意每个 slot 的 `request_region.reset()` 必须在 `tasks.clear()` 之前，因为 MR 持有 `request_buffer.data()` 的注册，必须先撤销注册再释放底层内存。最后 `storage_insert_owners_.clear()` 把所有 per-owner state 释放掉。

## 28.4 `sender.cc`：mutation batch 发送与 progress 线程

`sender.cc` 是 storage owner 客户端的"发送引擎"，单线程 work-conserving，负责四件事：轮询 CQ、drain 提交队列组 batch、post_send、回收已完成的 slot。

### 28.4.1 `run_storage_insert_progress_loop`（`sender.cc:5-54`）

主循环：

```cpp
vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));
vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
u32 first_owner = 0;
auto previous_poll = std::chrono::steady_clock::now();

for (;;) {
  bool progressed = false;
  reclaim_storage_owner_slots();

  const auto poll_started = std::chrono::steady_clock::now();
  storage_insert_current_cq_gap_ns_ =
    duration_ns(previous_poll, poll_started);
  previous_poll = poll_started;
```

`first_owner` 是轮转起点，每次循环 +1，实现 owner 间的公平调度。`storage_insert_current_cq_gap_ns_` 记录"上一次 poll 到本次 poll 的间隔"——这是 progress 线程的"轮询节奏"指标，会附在 slot 的 breakdown 上（`cpu_storage_owner_cq_progress_gap` 子项）。

接着轮询 send CQ：

```cpp
const i32 send_count = Context::poll_send_cq(
  send_wcs.data(), static_cast<i32>(send_wcs.size()),
  context_.get_send_cq(), [&](u64 wr_id) {
    const auto [owner_storage, slot_id] = decode_64bit(wr_id);
    handle_storage_owner_send_completion(owner_storage, slot_id);
  });
progressed = send_count > 0;
```

WR ID 是 64 位编码的 `(owner, slot)` 对（见 `storage_owner_client_helpers.hh:35-37` 的 `storage_owner_wr_id`），`decode_64bit` 解回来。`handle_storage_owner_send_completion`（见 28.5.1）只做"置 `send_done=true`、推进 ready 队列"两件事。

然后轮询 recv CQ：

```cpp
const i32 recv_count = context_.poll_recv_cq(
  recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
progressed = progressed || recv_count > 0;
for (i32 i = 0; i < recv_count; ++i) {
  const auto [owner_storage, slot_id] = decode_64bit(recv_wcs[i].wr_id);
  handle_storage_owner_response(
    owner_storage, slot_id, recv_wcs[i].byte_len);
}
```

注意 recv 这边是 `recv_wcs[i].wr_id`——响应 buffer 的 WR ID 是 `(owner, response_slot_id)`，但 `handle_storage_owner_response` 拿到 `response_slot_id` 后会在所有 in-use slot 中按 `batch_id` 匹配真正的发送 slot（见 28.5.2）。这种解耦是因为 send 和 recv 走不同 WR，无法用同一个 WR ID 串起来。

接着 drain 提交队列：

```cpp
progressed = drain_storage_owner_submissions(first_owner) || progressed;
reclaim_storage_owner_slots();
```

`drain_storage_owner_submissions` 是真正"组 batch + post_send"的地方（见 28.4.2）。`reclaim_storage_owner_slots` 在循环开头和结尾各调一次——开头回收上一轮 completion 线程释放的 slot，结尾回收本轮 CQ 完成可能触发的释放（虽然 completion 线程是异步的，但提前回收能减少下一轮的空闲 slot 短缺）。

退出条件：

```cpp
bool submissions_empty = true;
for (const auto& state : storage_insert_owners_) {
  submissions_empty = submissions_empty && state->queue->empty();
}
if (storage_insert_shutdown_.load(std::memory_order_acquire) &&
    submissions_empty &&
    storage_insert_inflight_.load(std::memory_order_acquire) == 0) {
  storage_insert_progress_done_.store(true, std::memory_order_release);
  storage_insert_progress_done_.notify_all();
  storage_ready_slots_->notify_all();
  return;
}
if (!progressed) std::this_thread::yield();
```

退出三个条件：shutdown 标志置位、所有 owner 的提交队列空、inflight 归零。inflight 计数在 `post_storage_owner_batch` +1、`reclaim_storage_owner_slots` -1，所以归零意味着所有在途 RPC 都已回收。无进展时 `yield` 让出 CPU，避免空转。

### 28.4.2 `drain_storage_owner_submissions`（`sender.cc:56-87`）

```cpp
const u32 owner_count = static_cast<u32>(storage_insert_owners_.size());
if (owner_count == 0) return false;
bool progressed = false;
for (u32 offset = 0; offset < owner_count; ++offset) {
  const u32 owner = (first_owner + offset) % owner_count;
  auto& state = *storage_insert_owners_[owner];
  while (!state.free_slots.empty()) {
    u32 first_task = 0;
    if (!state.queue->try_pop(first_task)) break;

    const u32 slot_id = state.free_slots.back();
    state.free_slots.pop_back();
    auto& slot = state.slots[slot_id];
    slot.tasks.clear();
    slot.tasks.push_back(first_task);
    u32 task_id = 0;
    while (slot.tasks.size() < config_.storage_owner_batch_max &&
           state.queue->try_pop(task_id)) {
      slot.tasks.push_back(task_id);
    }
    const auto dequeued_at = std::chrono::steady_clock::now();
    for (const u32 id : slot.tasks) {
      state.tasks[id].sender_dequeued_at = dequeued_at;
    }
    post_storage_owner_batch(owner, slot_id);
    progressed = true;
  }
}
first_owner = (first_owner + 1) % owner_count;
return progressed;
```

关键点：

1. **owner 公平轮转**：从 `first_owner` 开始 round-robin，每个 owner 内部 `while (!free_slots.empty() && !queue.empty())` 尽量发。`first_owner` 每轮循环 +1，保证所有 owner 都有机会被先服务。
2. **批大小自适应**：从队列里 pop 第一个 task 后，继续 `try_pop` 直到达到 `storage_owner_batch_max` 或队列空。这意味着低负载时一个 batch 可能只有 1 个 task（低延迟），高负载时自动凑满（高吞吐）。
3. **`sender_dequeued_at` 统一标记**：一批 task 共享同一个 dequeue 时间戳，避免为每个 task 单独取时钟。
4. **`free_slots` 是栈**：`back()` + `pop_back()`，LIFO。这不影响正确性，但有助于缓存局部性（最近用过的 slot 缓冲可能还在 cache）。

注意 `drain` 只在 `free_slots` 非空且 `queue` 非空时才发，所以**永远不会出现"task 已 dequeue 但找不到 slot"的情况**——这是后续容量推导能成立的基础。

### 28.4.3 `post_storage_owner_batch`（`sender.cc:129-231`）

这是把一批 task 真正编码成 wire 报文并 `post_send` 的函数，hot path。

先取出 slot、确认非空：

```cpp
auto& state = *storage_insert_owners_[owner_storage];
auto& slot = state.slots[slot_id];
if (slot.tasks.empty()) return;
```

然后确定 batch 元数据：

```cpp
const u32 item_count = static_cast<u32>(slot.tasks.size());
const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
bool collect_breakdown = false;
bool mutation_request = false;
for (const u32 task_id : slot.tasks) {
  const auto& task = state.tasks[task_id];
  const auto& sample = storage_completion_samples_[task.completion_id];
  collect_breakdown = collect_breakdown || sample.collects_breakdown();
  mutation_request = mutation_request ||
    task.kind != service::storage_owner::MutationKind::insert;
}
```

`batch_id` 来自全局原子计数器 `next_request_id_`，是响应匹配的 key。`mutation_request` 是一个 OR：**只要这批里有一个不是 insert，整批就走 mutation 协议**（带 `kinds[]` 数组）。这是兼容性设计：mutation 协议是 insert 协议的超集，混发时统一用 mutation magic。

`collect_breakdown` 同样是 OR：只要有一个 task 开了 breakdown，整批都收集——breakdown 是 per-item 的，但响应里的 `InsertBreakdownCounters` 是整批共享的（见 `storage_owner_protocol.hh:268-276`），要么一起收要么都不收。

接下来按是否 mutation 选报文布局：

```cpp
const auto prepare_start = collect_breakdown
  ? std::chrono::steady_clock::now()
  : std::chrono::steady_clock::time_point{};
const size_t request_size = mutation_request
  ? service::storage_owner::mutation_batch_request_bytes(item_count)
  : service::storage_owner::insert_batch_request_bytes(item_count);
const size_t response_size =
  service::storage_owner::insert_batch_response_bytes(item_count);
```

注意 `response_size` 总是用 `insert_batch_response_bytes`——响应格式不分 insert/mutation，都是同一个 `InsertBatchResponseHeader` 加 statuses、results、breakdown、invalidation。

随后填写请求头部：

```cpp
auto* request = reinterpret_cast<
  service::storage_owner::InsertBatchRequestHeader*>(
    slot.request_buffer.data());
request->magic = mutation_request
  ? service::storage_owner::kMutationMagic
  : service::storage_owner::kInsertMagic;
request->dim = config_.dim;
request->owner_storage = owner_storage;
request->source_client = cm_.client_id;
request->item_count = item_count;
request->vector_dtype = static_cast<u32>(VamanaNode::vector_dtype());
request->vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
request->anchor_hint_count = 0;
request->batch_id = batch_id;
```

注意这里有个 wire 协议的细节：`InsertBatchRequestHeader` 和 `MutationBatchRequestHeader` 是两个独立结构体（`storage_owner_protocol.hh:44-66`），但二者 `static_assert(sizeof(...) == 40)` 且字段偏移完全一致，包括 `anchor_hint_count` 在 offset 28、`batch_id` 在 offset 32。所以用 `InsertBatchRequestHeader*` 强转 mutation 请求缓冲也是安全的——magic 字段会区分两者，存储侧按 magic 解析（见第 23 课）。

`anchor_hint_count = 0` 是 schema-15 兼容性要求：这个字段在 wire 上保留，但本实现只接受 0（见 `storage_owner_protocol.hh:68-75` 的注释）。

然后填 ids、kinds、vectors：

```cpp
node_t* ids = mutation_request
  ? service::storage_owner::mutation_request_ids(slot.request_buffer.data())
  : service::storage_owner::request_ids(slot.request_buffer.data());
byte_t* vectors = mutation_request
  ? service::storage_owner::mutation_request_vectors(slot.request_buffer.data(), item_count)
  : service::storage_owner::request_vectors(slot.request_buffer.data(), item_count);
u32* kinds = mutation_request
  ? service::storage_owner::mutation_request_kinds(slot.request_buffer.data())
  : nullptr;
for (u32 i = 0; i < item_count; ++i) {
  const auto& task = state.tasks[slot.tasks[i]];
  ids[i] = task.item.id;
  if (kinds != nullptr) kinds[i] = static_cast<u32>(task.kind);
  byte_t* vector_output =
    vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
  if (task.kind == service::storage_owner::MutationKind::erase) {
    std::memset(vector_output, 0, VamanaNode::vector_bytes());
  } else {
    encode_float_vector_to_storage(
      task.item.values.data(), config_.dim,
      VamanaNode::vector_dtype(), vector_output);
  }
}
```

这里有几个要点：

- **layout helper**：`request_ids` / `request_vectors` / `mutation_request_ids` / `mutation_request_vectors` / `mutation_request_kinds` 都在 `storage_owner_protocol.hh:202-242`，是按 header 偏移计算的指针强转，零开销。
- **erase 把向量清零**：erase 不需要向量内容，但 wire 格式仍然占 `vector_bytes()` 空间，所以填零。这避免了"未初始化内存走 RDMA"的潜在信息泄露。
- **`encode_float_vector_to_storage`**：把 `element_t`（host float）编码成存储/GPU 期望的 dtype（可能是 fp16/fp32/bf16，由 `VamanaNode::vector_dtype()` 决定）。这与第 6 课 Vamana 节点格式、第 9 课 PQ 模型对齐。

填完报文，更新 slot 状态：

```cpp
slot.in_use = true;
slot.send_done = false;
slot.response_done = false;
slot.results_completed = false;
slot.completion_claimed = false;
slot.response_valid = false;
slot.response_slot_id = std::numeric_limits<u32>::max();
slot.gpu_reserved_items = item_count;
slot.publication_mutation_count = 0;
slot.publication_reserved_items = 0;
slot.publication_invalidated_graph_nodes.clear();
slot.item_count = item_count;
slot.batch_id = batch_id;
slot.request_prepare_ns = collect_breakdown
  ? duration_ns(prepare_start, std::chrono::steady_clock::now()) : 0;
slot.request_size = request_size;
slot.response_size = response_size;
slot.cq_progress_gap_ns = 0;
slot.send_posted_at = std::chrono::steady_clock::now();
storage_insert_inflight_.fetch_add(1, std::memory_order_acq_rel);
```

`gpu_reserved_items = item_count`：本批所有 item 都已在 `public_mutations.cc` 里预约了 GPU mutation 容量（每个 item 1 个），这里记录"需要释放的预约数"。如果后续某些 item 失败（owner 返回非 0 status），completion 线程会释放"未 commit"那部分的预约（见 28.5.4）。

最后 post_send：

```cpp
cm_.server_qps[owner_storage]->post_send_with_id(
  *slot.request_region,
  static_cast<u32>(request_size),
  IBV_WR_SEND,
  storage_owner_wr_id(owner_storage, slot_id),
  true,
  nullptr,
  0,
  0);
```

`IBV_WR_SEND` 是 RDMA verbs 的 send 操作（对端需要 post receive）。`storage_owner_wr_id(owner, slot_id)` 编码进 WR ID，CQE 回来时 `decode_64bit` 解出。`true` 是 signaling flag——每个 send 都产生 CQE（不做 unsignaled 优化，因为 batch 频率不高，简化实现）。

### 28.4.4 `reclaim_storage_owner_slots`（`sender.cc:89-127`）

```cpp
StorageOwnerReleasedSlot released;
while (storage_released_slots_->try_pop(released)) {
  lib_assert(released.owner_storage < storage_insert_owners_.size(), ...);
  auto& state = *storage_insert_owners_[released.owner_storage];
  lib_assert(released.slot_id < state.slots.size() &&
               released.response_slot_id < state.response_slots.size(), ...);
  auto& slot = state.slots[released.slot_id];
  lib_assert(slot.in_use && slot.results_completed, ...);
  slot.in_use = false;
  slot.send_done = false;
  slot.response_done = false;
  slot.results_completed = false;
  slot.completion_claimed = false;
  slot.response_valid = false;
  slot.response_slot_id = std::numeric_limits<u32>::max();
  slot.gpu_reserved_items = 0;
  slot.publication_mutation_count = 0;
  slot.publication_reserved_items = 0;
  slot.publication_invalidated_graph_nodes.clear();
  slot.item_count = 0;
  slot.batch_id = 0;
  // ... 清空所有时间戳和 size ...
  slot.tasks.clear();
  state.free_slots.push_back(released.slot_id);
  post_storage_owner_response_receive(
    released.owner_storage, released.response_slot_id);
  storage_insert_inflight_.fetch_sub(1, std::memory_order_acq_rel);
}
```

这个函数把 completion 线程通过 `storage_released_slots_` 交回的 slot 重置成"空闲"状态：清空所有标志位、清空 task 列表、把 slot_id 推回 `free_slots` 栈，并把对应的 response_slot 重新 post receive（这一步必须在 progress 线程做，因为 `cm_.server_qps` 的访问没有锁，只能 progress 线程碰）。最后 inflight -1。

注意 `lib_assert(slot.in_use && slot.results_completed, ...)`：这是不变量检查——只有"已 commit 的在途 slot"才能被释放。如果 violation，说明 ready/released 队列协议被破坏，立即 abort。

## 28.5 `completion.cc`：stage1 ACK 完成处理与 GPU 发布

`completion.cc` 是计算侧最复杂的文件，包含 9 个函数。我们按数据流顺序讲解：CQ 完成 → 响应匹配 → 校验 → ready 入队 → commit → GPU 发布 → 释放。

### 28.5.1 `handle_storage_owner_send_completion`（`completion.cc:6-16`）

```cpp
void ComputeService::handle_storage_owner_send_completion(
    u32 owner_storage, u32 slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (slot_id >= state.slots.size()) return;
  auto& slot = state.slots[slot_id];
  if (!slot.in_use) return;
  slot.send_done = true;
  slot.send_completed_at = std::chrono::steady_clock::now();
  queue_storage_owner_completion(slot);
}
```

send CQE 到达只做两件事：置 `send_done=true`、记录时间戳、尝试推进 ready 队列。前两个 `if` 是防御性检查（CQE 可能在 shutdown 后才到达，此时 owner/slot 可能已失效）。

### 28.5.2 `handle_storage_owner_response`（`completion.cc:18-80`）

这是响应到达的处理函数，最关键的是"按 batch_id 匹配发送 slot"。

```cpp
auto& response_slot = state.response_slots[response_slot_id];
const auto* response = reinterpret_cast<const
  service::storage_owner::InsertBatchResponseHeader*>(
    response_slot.buffer.data());
StorageOwnerRpcSlot* matched = nullptr;
if (received_bytes >=
    sizeof(service::storage_owner::InsertBatchResponseHeader)) {
  for (auto& slot : state.slots) {
    if (slot.in_use && !slot.response_done &&
        slot.batch_id == response->batch_id) {
      matched = &slot;
      break;
    }
  }
}
```

注意：recv WR ID 里的 `slot_id` 是 response_slot_id，**不是发送 slot_id**。响应报文头部带 `batch_id`，发送时记录在每个 slot 里，所以匹配靠"扫描所有 in_use 且未 response_done 的 slot，比对 batch_id"。

为什么要这样？因为 RDMA SEND/RECV 是两个独立的 WR 队列，send 的 WR ID 和 recv 的 WR ID 无法直接关联。计算侧的选择是：recv buffer 用 `(owner, response_slot_id)` 编码，响应报文自带 `batch_id`，靠应用层匹配。

为什么不直接用 response_slot_id 配对？因为 response_slot 是"任意 recv buffer"，对端 SEND 时不知道哪个 response_slot 会接到——verbs 层只是把 send 投递到对端的 recv 队列，由对端 post_receive 的顺序决定。所以 batch_id 是唯一的关联手段。

匹配失败的处理：

```cpp
if (matched == nullptr) {
  static std::atomic<u32> unknown_response_logs{0};
  const u32 log_index = unknown_response_logs.fetch_add(
    1, std::memory_order_relaxed);
  if (log_index < 16) {
    std::cerr << "[storage-owner] unmatched insert response"
              << " owner=" << owner_storage
              << " response_slot=" << response_slot_id
              << " magic=0x" << std::hex << response->magic << std::dec
              << " response_owner=" << response->owner_storage
              << " batch_id=" << response->batch_id
              << " item_count=" << response->item_count
              << " received_bytes=" << received_bytes << std::endl;
  }
  post_storage_owner_response_receive(owner_storage, response_slot_id);
  return;
}
```

匹配不上可能是：迟到的响应、重复响应、损坏报文。前 16 次打 stderr，之后静默（避免日志洪水），无论如何都把 response_slot 重新 post receive，让该 buffer 能继续接下一个响应。

匹配成功后调用 `validate_storage_owner_response` 做头部校验：

```cpp
const auto* request = reinterpret_cast<const
  service::storage_owner::InsertBatchRequestHeader*>(
    matched->request_buffer.data());
const auto validation = validate_storage_owner_response(
  *response,
  received_bytes,
  response_slot.buffer.size(),
  request->magic,
  owner_storage,
  matched->item_count,
  matched->batch_id,
  matched->response_size);
lib_assert(validation != StorageOwnerResponseValidation::unmatched,
           "batch-id matched response was classified as unmatched");
matched->response_done = true;
matched->response_valid =
  validation == StorageOwnerResponseValidation::matched_valid;
matched->response_slot_id = response_slot_id;
matched->response_completed_at = std::chrono::steady_clock::now();
matched->cq_progress_gap_ns = storage_insert_current_cq_gap_ns_;
queue_storage_owner_completion(*matched);
```

校验返回三态：`unmatched`（batch_id 不一致，不应发生，因为前面已经匹配过）、`matched_invalid`（字段不一致）、`matched_valid`。`response_valid` 只在 `matched_valid` 时为 true，后续 `commit_storage_owner_slot` 会用它决定是否信任响应体。

注意 `cq_progress_gap_ns` 在这里赋值——这是 progress 线程最近一次的"两次 poll 间隔"，用来衡量"响应到达但 progress 还没轮到处理"的延迟。它会被 per-item 平摊到 breakdown 子项 `cpu_storage_owner_cq_progress_gap`。

最后一句注释：

```cpp
// The receive is reposted only after the response executor has finished
// parsing this buffer. This removes the large CQ-thread memcpy.
```

这是关键设计：响应 buffer **不在这里重新 post receive**，而是等到 completion 线程 commit 完成后，通过 `release_storage_owner_slot` → `storage_released_slots_` → `reclaim_storage_owner_slots` 在 progress 线程里 post。这样 completion 线程可以慢慢解析 buffer，不用担心 buffer 被下一个响应覆盖；progress 线程的 CQ 回调也不做重活。

### 28.5.3 `post_storage_owner_response_receive`（`completion.cc:82-92`）

```cpp
void ComputeService::post_storage_owner_response_receive(
    u32 owner_storage, u32 response_slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (response_slot_id >= state.response_slots.size()) return;
  auto& response_slot = state.response_slots[response_slot_id];
  cm_.server_qps[owner_storage]->post_receive(
    *response_slot.region,
    static_cast<u32>(response_slot.buffer.size()),
    storage_owner_wr_id(owner_storage, response_slot_id));
}
```

纯粹的 RDMA RECV post：把整个 response buffer（按 `response_bytes` 上限分配）注册的 MR 投递到对端的 recv 队列，WR ID 编码 `(owner, response_slot_id)`。注意投递的是 buffer 满容量，而不是"预期响应大小"——因为 verbs RECV 不知道对端会发多少字节，只要不超过 buffer 就行。

### 28.5.4 `queue_storage_owner_completion`（`completion.cc:94-106`）

```cpp
bool ComputeService::queue_storage_owner_completion(
    StorageOwnerRpcSlot& slot) {
  if (!slot.in_use || !slot.send_done || !slot.response_done ||
      slot.results_completed || slot.completion_claimed) {
    return false;
  }
  slot.completion_claimed = true;
  const bool queued = storage_ready_slots_->try_push(
    StorageOwnerReadySlot{slot.owner_storage, slot.slot_id});
  lib_assert(queued,
             "storage-owner ready queue exhausted despite RPC-slot bound");
  return true;
}
```

这是 send_done 和 response_done 两条事件汇合点。两条事件都来自 progress 线程（send CQE 和 recv CQE），但可能乱序到达——send_done 可能先于 response_done（罕见，因为 send 完成通常早于响应），也可能后于（如果响应到达时 send CQE 还没轮询到）。无论谁先到，都通过这个函数尝试推进。

`completion_claimed` 防止重复入队：第一条到达的事件置 `send_done` 或 `response_done`，第二条到达时检查"两个都 done 且未 claimed"，置 claimed 并入队。

`lib_assert(queued, "ready queue exhausted despite RPC-slot bound")`：ready 队列容量是 `max_inflight = owner_count * rpc_depth`，正好等于所有 slot 数；每个 slot 至多产生一条 ready（因为 `completion_claimed` 阻止重复），所以队列不会满。这是一个不变量证明。

### 28.5.5 `run_storage_insert_completion_loop`（`completion.cc:108-133`）

completion 线程主循环：

```cpp
for (;;) {
  StorageOwnerReadySlot ready;
  if (!storage_ready_slots_->pop_wait(
        ready, storage_insert_progress_done_)) {
    return;
  }

  auto& state = *storage_insert_owners_[ready.owner_storage];
  auto& slot = state.slots[ready.slot_id];
  commit_storage_owner_slot(ready.owner_storage, ready.slot_id);

  if (slot.publication_mutation_count == 0) {
    if (persistent_search_ != nullptr &&
        slot.publication_reserved_items != 0) {
      persistent_search_->release_mutation_capacity(
        slot.publication_reserved_items);
    }
  } else {
    publish_storage_owner_mutations(slot);
  }
  // Publication consumes slot-owned spans synchronously. Reuse is safe only
  // after the GPU command and coordinator handoff have both returned.
  release_storage_owner_slot(ready.owner_storage, ready.slot_id);
}
```

`pop_wait` 阻塞等 ready slot，第二个参数 `storage_insert_progress_done_` 是 shutdown 信号——progress 线程退出后会置位并 notify，`pop_wait` 收到信号后返回 false 让 completion 线程也退出。

每个 ready slot 走三步：

1. `commit_storage_owner_slot`：解析响应、更新 compute-side idmap、构造 `DeltaMutation` 列表、回收未用的 GPU 容量预约、完成 task。这一步会把 `slot.publication_mutation_count` 和 `slot.publication_reserved_items` 填好。
2. **GPU 发布分支**：如果有 mutation（`publication_mutation_count > 0`），调 `publish_storage_owner_mutations` 发布到 GPU delta；否则（整批都失败，或者 `persistent_search_ == nullptr`）直接释放预约容量。
3. `release_storage_owner_slot`：把 slot 推进 `storage_released_slots_`，让 progress 线程回收。

注释"Publication consumes slot-owned spans synchronously"强调：`publish_mutations` 是同步调用，返回后 GPU 命令已提交、coordinator epoch 已发布，slot 内的 `publication_mutations` 和 `publication_invalidated_graph_nodes` 缓冲可以安全复用。这就是为什么 `lifecycle.cc` 里能预分配这些缓冲——发布是同步的，不存在"发布后还在用缓冲"的悬挂引用。

### 28.5.6 `commit_storage_owner_slot`（`completion.cc:135-352`）

这是计算侧最长的函数，分阶段讲解。

#### 阶段 A：不变量检查 + 响应头校验（`completion.cc:135-171`）

```cpp
auto& state = *storage_insert_owners_[owner_storage];
lib_assert(slot_id < state.slots.size(), ...);
auto& slot = state.slots[slot_id];
lib_assert(slot.in_use && slot.send_done && slot.response_done &&
             slot.completion_claimed && !slot.results_completed &&
             slot.response_slot_id < state.response_slots.size(),
           "storage-owner completion claimed a slot in an invalid state");
slot.results_completed = true;
const auto response_executor_started = std::chrono::steady_clock::now();

const byte_t* response_buffer =
  state.response_slots[slot.response_slot_id].buffer.data();
const auto* response = reinterpret_cast<const
  service::storage_owner::InsertBatchResponseHeader*>(response_buffer);
const auto* request = reinterpret_cast<const
  service::storage_owner::InsertBatchRequestHeader*>(
    slot.request_buffer.data());
bool response_ok = slot.response_valid &&
  (response->magic == service::storage_owner::kInsertMagic ||
   response->magic == service::storage_owner::kMutationMagic) &&
  response->magic == request->magic &&
  response->owner_storage == slot.owner_storage &&
  response->batch_id == slot.batch_id &&
  response->item_count == slot.item_count;
```

`response_ok` 是"响应整体可信"的总开关，综合了：

- `slot.response_valid`：`validate_storage_owner_response` 给出的字段校验结果（含字节数匹配）。
- magic 一致：响应 magic 与请求 magic 一致（避免 insert 请求被 mutation 响应回答）。
- owner/batch_id/item_count 三元组匹配。

注意 `validate_storage_owner_response` 已经检查过这些字段，这里再检查一次是为了**双重防御**——任何一环出错都让 `response_ok = false`，然后整批都按失败处理。

接下来提取 invalidation_count：

```cpp
u32 invalidation_count = 0;
if (response_ok) {
  invalidation_count =
    *service::storage_owner::response_invalidation_count(
      response_buffer, slot.item_count);
  response_ok = invalidation_count <=
    service::storage_owner::response_invalidation_capacity(
      slot.item_count);
  if (!response_ok) invalidation_count = 0;
}
```

`response_invalidation_capacity(item_count) = item_count * R`（`storage_owner_protocol.hh:294-296`）。owner 返回的 `invalidation_count` 不能超过这个上限，否则视为响应损坏。

#### 阶段 B：取各段指针 + breakdown 决策（`completion.cc:173-195`）

```cpp
const u32* statuses =
  service::storage_owner::response_statuses(response_buffer);
const auto* results =
  service::storage_owner::response_mutation_results(
    response_buffer, slot.item_count);
const bool mutation_request =
  request->magic == service::storage_owner::kMutationMagic;
const byte_t* request_vectors = mutation_request
  ? service::storage_owner::mutation_request_vectors(
      slot.request_buffer.data(), slot.item_count)
  : service::storage_owner::request_vectors(
      slot.request_buffer.data(), slot.item_count);

bool collect_breakdown = false;
for (const u32 task_id : slot.tasks) {
  collect_breakdown = collect_breakdown ||
    storage_completion_samples_[state.tasks[task_id].completion_id]
      .collects_breakdown();
}
const auto* breakdown = collect_breakdown && response_ok
  ? service::storage_owner::response_breakdown(
      response_buffer, slot.item_count)
  : nullptr;
```

`statuses` 是每个 item 的状态码（`MutationStatus` 或 `InsertStatus`，0 表示成功）。`results` 是每个 item 的 `MutationResult`（含 new_rptr / old_rptr / generation / maintenance_sequence）。`request_vectors` 是回指请求缓冲里的向量——因为 `DeltaMutation` 发布到 GPU 时需要向量，而向量已经在请求缓冲里编好码了，直接复用。

`breakdown` 指针仅在 `collect_breakdown && response_ok` 时非空——失败响应的 breakdown 字段不可信。

#### 阶段 C：失败响应日志（`completion.cc:196-212`）

```cpp
if (!response_ok) {
  static std::atomic<u32> bad_response_logs{0};
  const u32 log_index = bad_response_logs.fetch_add(
    1, std::memory_order_relaxed);
  if (log_index < 16) {
    std::cerr << "[storage-owner] invalid insert response"
              << " owner=" << slot.owner_storage
              << " slot=" << slot.slot_id
              << " magic=0x" << std::hex << response->magic << std::dec
              << " response_owner=" << response->owner_storage
              << " expected_owner=" << slot.owner_storage
              << " batch_id=" << response->batch_id
              << " expected_batch_id=" << slot.batch_id
              << " item_count=" << response->item_count
              << " expected_item_count=" << slot.item_count << std::endl;
  }
}
```

同样的"前 16 次"日志限流模式，全文件出现 5 处（unknown_response、bad_response、failed_status、gpu_delta_failure、late_rpc_completion）。这是 dvstor 一致的诊断策略：异常情况打少量日志供调试，避免日志洪水拖垮系统。

#### 阶段 D：breakdown 时间拆分（`completion.cc:214-223`）

```cpp
const u64 memory_breakdown_ns = breakdown == nullptr
  ? 0 : breakdown->total();
const u64 send_ns = collect_breakdown
  ? duration_ns_clamped(slot.send_posted_at, slot.send_completed_at) : 0;
const u64 response_wait_ns = collect_breakdown
  ? duration_ns_clamped(
      slot.send_completed_at, slot.response_completed_at) : 0;
const u64 response_wait_unaccounted_ns =
  collect_breakdown && response_wait_ns > memory_breakdown_ns
    ? response_wait_ns - memory_breakdown_ns : 0;
```

这一段把"send 完成到响应到达"这段等待时间拆成两部分：owner 内部各种动作（`breakdown->total()`）已经记账的部分，和"未解释"的部分（`response_wait_unaccounted_ns`）。后者会被记到 breakdown 子项 `cpu_storage_owner_response_wait_unaccounted`，帮助定位"owner 声称没花时间但实际 RPC 慢"的情况（ verbs 层延迟、网络延迟等）。

#### 阶段 E：失效图节点收集（`completion.cc:225-238`）

```cpp
slot.publication_mutation_count = 0;
slot.publication_reserved_items = 0;
slot.publication_invalidated_graph_nodes.clear();
if (persistent_search_ != nullptr && response_ok) {
  const u64* invalidated_raws =
    service::storage_owner::response_invalidated_raws(
      response_buffer, slot.item_count);
  for (u32 index = 0; index < invalidation_count; ++index) {
    if (invalidated_raws[index] != 0) {
      slot.publication_invalidated_graph_nodes.push_back(
        invalidated_raws[index]);
    }
  }
}
```

owner 在反向索引更新时，可能让某些图节点的反向指针失效（例如 old_rptr 被替换、被删除）。这些"失效图节点 raw"会传给 GPU，让 GPU 端的 graph cache 失效对应条目（见 `delta_publication.cc:261` 的 `graph_cache_keys` 和 `refresh_anchor_graph_records`）。

注意 `invalidated_raws[index] != 0` 的过滤——0 是哨兵值，表示空条目，跳过。

#### 阶段 F：per-item commit 循环（`completion.cc:240-325`）

这是函数的核心：

```cpp
u32 committed_items = 0;
for (u32 i = 0; i < slot.item_count; ++i) {
  const u32 task_id = slot.tasks[i];
  auto& task = state.tasks[task_id];
  auto& sample = storage_completion_samples_[task.completion_id];
  const bool committed = response_ok && statuses[i] == 0;
  if (response_ok && !committed) {
    // ... 失败日志 ...
  }
```

每个 item 的 `committed` 取决于两个条件：响应整体 OK，且该 item 的 status 为 0。任一不满足都视为该 item 失败。

breakdown 记账：

```cpp
if (sample.collects_breakdown()) {
  add_storage_owner_sender_breakdown(
    &sample,
    duration_ns_clamped(task.enqueued_at, task.sender_dequeued_at),
    slot.request_prepare_ns,
    send_ns,
    response_wait_unaccounted_ns,
    slot.item_count);
  sample.add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_dequeue_to_post,
    duration_ns_clamped(task.sender_dequeued_at, slot.send_posted_at));
  sample.add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_cq_progress_gap,
    per_item_ns(slot.cq_progress_gap_ns, slot.item_count));
  sample.add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_response_executor_queue,
    duration_ns_clamped(
      slot.response_completed_at, response_executor_started));
  if (breakdown != nullptr) {
    add_storage_owner_breakdown(&sample, *breakdown, slot.item_count);
  }
}
```

`add_storage_owner_sender_breakdown`（`storage_owner_client_helpers.hh:101-119`）把"发送侧"的 4 段延迟记到 sample：sender_queue_wait、request_prepare、rdma_send、response_wait_unaccounted。`add_storage_owner_breakdown`（同文件 39-99）把 owner 返回的 breakdown 计数（medoid/search/prune/write_node/reverse/...）记到 sample。两者都用 `per_item_ns(total, item_count)` 平摊到每个 item。

注意 `response_executor_queue` 子项：响应到达 progress 线程后，要等 completion 线程从 ready 队列 pop 出来才开始处理，这段时间就是"响应已到但未被处理"的调度延迟。

如果 committed，进入 compute-side idmap 更新和 DeltaMutation 构造：

```cpp
if (committed) {
  const auto& result = results[i];
  const bool newest_generation = publish_compute_side_id(
    task.item.id,
    RemotePtr{task.kind == service::storage_owner::MutationKind::erase
                ? result.old_rptr_raw : result.new_rptr_raw},
    task.kind == service::storage_owner::MutationKind::erase,
    slot.owner_storage,
    result.generation);
  // Responses from different owner RPC slots may complete out of order.
  // Only the newest generation may enter the ordered GPU publication
  // stream; publishing an older completion after a newer one could revive
  // a tombstoned/upserted route representative.
  if (persistent_search_ != nullptr && newest_generation) {
    lib_assert(slot.publication_mutation_count <
                 slot.publication_mutations.size(),
               "storage-owner publication slot overflow");
    gpu_search::DeltaMutation& mutation =
      slot.publication_mutations[slot.publication_mutation_count];
    mutation.id = task.item.id;
    mutation.kind = task.kind;
    mutation.generation = result.generation;
    mutation.remote_node = result.new_rptr_raw;
    mutation.old_remote_node = result.old_rptr_raw;
    mutation.anchor_hint = 0;
    mutation.maintenance_sequence = result.maintenance_sequence;
    mutation.owner_storage = owner_storage;
    mutation.durable = false;
    mutation.epoch = 0;
    mutation.enqueued_at = slot.response_completed_at;
    if (mutation.kind != service::storage_owner::MutationKind::erase) {
      const byte_t* vector = request_vectors +
        static_cast<size_t>(i) * VamanaNode::vector_bytes();
      lib_assert(mutation.vector.size() == VamanaNode::vector_bytes(),
                 "storage-owner publication vector was not preallocated");
      std::memcpy(
        mutation.vector.data(), vector, VamanaNode::vector_bytes());
    }
    ++slot.publication_mutation_count;
    ++committed_items;
  }
}
```

这里有几个关键设计：

**1. `publish_compute_side_id` 做 generation 检查**（`index_commands.cc:14-29`）：

```cpp
bool ComputeService::publish_compute_side_id(node_t id,
                                             RemotePtr ptr,
                                             bool deleted,
                                             u32 owner_storage,
                                             u32 generation) {
  auto& shard = compute_side_idmap_[static_cast<size_t>(id) % kComputeSideIdShardCount];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto existing = shard.entries.find(id);
  if (existing != shard.entries.end() &&
      existing->second.generation >= generation) {
    return false;
  }
  shard.entries[id] = ComputeSideIdEntry{
    ptr, deleted, owner_storage, generation};
  return true;
}
```

compute-side idmap 是一个 sharded map（按 ID 哈希分片，每片独立 mutex），记录每个 ID 当前的"最新 generation"对应的 `RemotePtr`、`deleted` 标志、`owner_storage`。如果已有 generation >= 本次 generation，返回 false——**这意味着同 ID 的旧响应被丢弃，不进入 GPU 发布流**。这是"乱序完成"的关键防护：两个不同 batch 的同 ID mutation 可能 out-of-order 完成，只有最新的能进 GPU。

**2. erase 用 `old_rptr_raw`，insert/upsert 用 `new_rptr_raw`**：erase 后 ID 不再有有效 rptr，所以用 old（用于 GPU 端识别"被删的旧位置"）；insert/upsert 则用新位置。

**3. `mutation.remote_node = result.new_rptr_raw`**：注意 erase 时 `new_rptr_raw` 可能为 0，但 `remote_node` 仍然填它——GPU 端 `delta_publication.cc:124-132` 会按 `mutation.kind == erase` 分支处理，`record_remote = mutation.remote_node != 0 ? mutation.remote_node : mutation.old_remote_node`，所以 erase 时实际用 `old_remote_node`。

**4. `mutation.durable = false`、`mutation.epoch = 0`**：durable 由后续维护流程提升（见第 15 课）；epoch 由 `publish_mutations` 内部 `delta_.reserve_epoch()` 分配（见 28.6.1）。

**5. `enqueued_at = slot.response_completed_at`**：用响应到达时间作为"入队时间"，GPU 发布后会用它计算"stage1 ACK 到 GPU 可见"的延迟（SLO 关键指标）。

**6. 向量复用请求缓冲**：`mutation.vector` 是预分配的 `vector_bytes()` 缓冲（见 `lifecycle.cc:79-81`），这里 `memcpy` 从请求缓冲把对应 item 的向量拷过来。erase 不拷（mutation.vector 留作未定义，GPU 端会忽略）。

`lib_assert(mutation.vector.size() == VamanaNode::vector_bytes(), ...)` 是对预分配不变量的检查。

#### 阶段 G：完成 task 和 breakdown 收尾（`completion.cc:327-352`）

```cpp
const auto response_processed_at = std::chrono::steady_clock::now();
const u64 response_process_ns = duration_ns(
  response_executor_started, response_processed_at);
for (u32 i = 0; i < slot.item_count; ++i) {
  const u32 task_id = slot.tasks[i];
  auto& task = state.tasks[task_id];
  auto& sample = storage_completion_samples_[task.completion_id];
  if (sample.collects_breakdown()) {
    sample.add_subcategory(
      service::breakdown::Subcategory::cpu_storage_owner_response_process,
      per_item_ns(response_process_ns, slot.item_count));
  }
  sample.mark_finished(response_processed_at);
  complete_storage_owner_task(
    owner_storage, task_id, response_ok && statuses[i] == 0);
}

lib_assert(committed_items <= slot.gpu_reserved_items,
           "committed storage mutations exceeded reserved GPU capacity");
slot.publication_reserved_items = committed_items;
const u32 release_reserved_items =
  slot.gpu_reserved_items - committed_items;
if (persistent_search_ != nullptr && release_reserved_items != 0) {
  persistent_search_->release_mutation_capacity(release_reserved_items);
}
```

`response_process_ns` 是 commit 整批的耗时，per-item 平摊到 breakdown `cpu_storage_owner_response_process`。然后 `sample.mark_finished` 标记 sample 完成，`complete_storage_owner_task`（见 28.5.7）回收 task 并 complete CompletionPool——这一步会唤醒阻塞在 `wait` 上的调用方。

最后是 GPU 容量预约的"找零"：`gpu_reserved_items` 是发送时预约的整批 item 数（全部 item 都预约了 1 个），`committed_items` 是实际 commit 的数量，差额 `release_reserved_items` 是"未 commit 的预约"——这部分立即释放回 GPU 容量池。`publication_reserved_items = committed_items` 记录"待发布完后再释放的预约数"，这部分在 `publish_storage_owner_mutations` 返回后释放。

`lib_assert(committed_items <= slot.gpu_reserved_items, ...)` 是不变量：commit 数不可能超过预约数，因为每个 item 至多预约 1 个。

### 28.5.7 `release_storage_owner_slot` / `complete_storage_owner_task` / `fail_storage_owner_tasks`

`release_storage_owner_slot`（`completion.cc:354-368`）：

```cpp
void ComputeService::release_storage_owner_slot(
    u32 owner_storage, u32 slot_id) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(slot_id < state.slots.size(), ...);
  auto& slot = state.slots[slot_id];
  lib_assert(slot.in_use && slot.results_completed &&
               slot.response_slot_id < state.response_slots.size(),
             "storage-owner released a slot before completion");
  const bool queued = storage_released_slots_->try_push(
    StorageOwnerReleasedSlot{
      owner_storage, slot_id, slot.response_slot_id});
  lib_assert(queued,
             "storage-owner release queue exhausted despite RPC-slot bound");
}
```

把 slot 推进 `storage_released_slots_`，等 progress 线程的 `reclaim_storage_owner_slots` 回收。注意载荷带 `response_slot_id`——因为 response buffer 要在 progress 线程重新 post receive。

`complete_storage_owner_task`（`completion.cc:408-422`）：

```cpp
void ComputeService::complete_storage_owner_task(
    u32 owner_storage, u32 task_id, bool success) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(task_id < state.task_capacity, ...);
  auto& task = state.tasks[task_id];
  const u32 completion_id = task.completion_id;
  task.item.values.clear();
  task.enqueued_at = {};
  task.sender_dequeued_at = {};
  task.completion_id = std::numeric_limits<u32>::max();
  const bool freed = state.free_tasks->try_push(task_id);
  lib_assert(freed, "storage-owner task pool overflow");
  storage_completion_pool_->complete(completion_id, success);
}
```

清空 task 内容、推回 `free_tasks` 池、`complete` 唤醒调用方。`storage_completion_pool_->complete(completion_id, success)` 是 bounded::CompletionPool 的接口，让阻塞在 `wait(completion_id)` 的调用方返回 `success` 或 `failure`。

`fail_storage_owner_tasks`（`completion.cc:424-438`）：

```cpp
void ComputeService::fail_storage_owner_tasks(
    u32 owner_storage, vec<u32>& tasks) {
  if (tasks.empty()) return;
  if (persistent_search_ != nullptr) {
    persistent_search_->release_mutation_capacity(tasks.size());
  }
  const auto finished_at = std::chrono::steady_clock::now();
  for (const u32 task_id : tasks) {
    auto& task = storage_insert_owners_[owner_storage]->tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    if (!sample.finished_flag) sample.mark_finished(finished_at);
    complete_storage_owner_task(owner_storage, task_id, false);
  }
  tasks.clear();
}
```

shutdown 时调用。先释放这些 task 的 GPU 容量预约（`tasks.size()` 个），再逐个 mark_finished + complete_storage_owner_task(false)，让调用方收到失败结果。注意 `if (!sample.finished_flag)` 的检查——某些 task 可能已经被 mark_finished（例如 stop_storage_insert_runtime 里同时处理 queue 和 slot 中的 task 时可能重复），避免覆盖已完成 sample 的时间戳。

### 28.5.8 `publish_storage_owner_mutations`（`completion.cc:370-406`）

```cpp
void ComputeService::publish_storage_owner_mutations(
    StorageOwnerRpcSlot& slot) {
  const std::span<gpu_search::DeltaMutation> mutations{
    slot.publication_mutations.data(), slot.publication_mutation_count};
  if (persistent_search_ == nullptr || mutations.empty()) {
    if (persistent_search_ != nullptr &&
        slot.publication_reserved_items != 0) {
      persistent_search_->release_mutation_capacity(
        slot.publication_reserved_items);
    }
    return;
  }

  auto& invalidated = slot.publication_invalidated_graph_nodes;
  std::sort(invalidated.begin(), invalidated.end());
  invalidated.erase(
    std::unique(invalidated.begin(), invalidated.end()), invalidated.end());
  try {
    if (!persistent_search_->publish_mutations(
          mutations, invalidated)) {
      persistent_search_->mark_committed_mutation_gap(
        "persistent GPU mutation publication returned false");
    }
  } catch (const std::exception& error) {
    persistent_search_->mark_committed_mutation_gap(error.what());
    // ... 失败日志 ...
  }
  persistent_search_->release_mutation_capacity(
    slot.publication_reserved_items);
}
```

三步：

1. **invalidated 去重**：owner 可能返回重复的失效图节点 raw（多个 item 失效同一节点），sort + unique 去重，避免 GPU 端重复处理。
2. **调 `publish_mutations`**：把 `DeltaMutation` span 和失效节点 span 交给 `PersistentSearchEngine`。
3. **释放容量**：无论发布成功失败，都释放 `publication_reserved_items` 个预约容量（这些 item 已经从"预约"变成"已发布"或"已放弃"，预约额度必须归还）。

**关键：失败处理是 `mark_committed_mutation_gap`**。这个函数（`persistent_engine.cc:251-255`）调用 `impl_->mark_unhealthy("storage committed a mutation that is not GPU-visible: " + reason)`，把引擎标记为不健康。原因是：stage1 ACK 已经告诉客户端"commit 成功"，但 GPU 没看到——这是 dvstor 的"一致性裂缝"，不能继续正常服务。系统会进入不健康状态，触发维护流程或人工介入。

这与第 15 课 delta_publication 的"stage1 ACK 在前、GPU 可见在后"模型一致：stage1 承诺 owner-memory 已落定，GPU 可见性是异步的，但**不允许"已 commit 但永远 GPU 不可见"**——一旦发生，引擎标记不健康。

## 28.6 `public_mutations.cc`：对外 mutation 入口

`public_mutations.cc` 是客户端调用的同步入口，包含 `insert / upsert / erase` 三个公开方法和共享的 `submit_storage_owner_mutations` 实现。

### 28.6.1 `insert / upsert / erase`（`public_mutations.cc:131-155`）

```cpp
size_t ComputeService::insert(const vec<InsertItem>& batch) {
  const size_t inserted = submit_storage_owner_mutations(
    batch, service::storage_owner::MutationKind::insert);
  vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
  return inserted;
}

size_t ComputeService::upsert(const vec<InsertItem>& batch) {
  const size_t updated = submit_storage_owner_mutations(
    batch, service::storage_owner::MutationKind::upsert);
  vectors_inserted_.fetch_add(updated, std::memory_order_relaxed);
  return updated;
}

size_t ComputeService::erase(const vec<node_t>& ids) {
  vec<InsertItem> items;
  items.reserve(ids.size());
  for (const node_t id : ids) {
    InsertItem item;
    item.id = id;
    items.push_back(std::move(item));
  }
  return submit_storage_owner_mutations(
    items, service::storage_owner::MutationKind::erase);
}
```

三个方法都是 `submit_storage_owner_mutations` 的薄包装：insert/upsert 直接转发（kind 不同），erase 先把 `vec<node_t>` 包装成 `vec<InsertItem>`（不带 values）。`vectors_inserted_` 是个统计计数器，insert/upsert 累加，erase 不减（语义上是"曾经插入过的向量数"）。

### 28.6.2 `submit_storage_owner_mutations`（`public_mutations.cc:5-129`）

这是计算侧 mutation 的核心同步逻辑。

#### 前置检查（`public_mutations.cc:5-21`）

```cpp
if (!config_.enable_updates) {
  throw std::runtime_error(
    "compute updates are disabled by enable-updates=false");
}
if (storage_insert_owners_.empty()) {
  throw std::runtime_error("storage_owner mutation runtime is not initialized");
}
if (kind != service::storage_owner::MutationKind::erase) {
  for (const auto& item : items) {
    if (item.values.size() != config_.dim) {
      throw std::invalid_argument("mutation dimension mismatch");
    }
  }
}
```

三道闸：配置开关、运行时已启动、维度匹配（erase 不检查，因为不需要向量）。

#### thread_local pending 缓冲（`public_mutations.cc:23-30`）

```cpp
thread_local vec<u32> pending;
pending.clear();
pending.reserve(std::min<size_t>(
  items.size(), storage_completion_pool_->capacity()));
size_t committed = 0;
```

`thread_local` 是关键——调用方线程在 `submit_storage_owner_mutations` 内部循环提交 + 等待，`pending` 缓冲记录"已提交但未等待"的 completion_id。`reserve` 取 `min(items.size, pool capacity)`，避免无意义的大分配。

注释解释了为什么需要这个缓冲：单条提交时不想每次分配；同时支持任意大小的公开 batch（caller 可能一次传 1000 条）。

#### `consume_one` lambda（`public_mutations.cc:32-62`）

```cpp
const auto consume_one = [&]() {
  lib_assert(!pending.empty(), "missing storage-owner completion");
  const u32 completion_id = pending.back();
  const auto result = storage_completion_pool_->wait(completion_id);
  auto& sample = storage_completion_samples_[completion_id];
  if (sample.collects_breakdown()) {
    sample.add_subcategory(
      service::breakdown::Subcategory::cpu_storage_owner_caller_wake,
      duration_ns_clamped(
        sample.finished_at, std::chrono::steady_clock::now()));
  }
  sample.mark_finished(std::chrono::steady_clock::now());
  if (sample.finished_flag) {
    std::lock_guard<std::mutex> lock(breakdown_mutex_);
    service::breakdown::add_sample(
      completed_breakdown_report_.insert, sample);
  }
  if (sample.end_to_end_ns >
      static_cast<u64>(config_.storage_owner_rpc_timeout_ms) * 1'000'000ull) {
    // ... 超时日志 ...
  }
  committed += result == bounded::CompletionPool::Result::success ? 1 : 0;
  storage_completion_pool_->release_consumer(completion_id);
  pending.pop_back();
};
```

`consume_one` 是"等一个 completion_id 完成"的操作：

1. `wait(completion_id)` 阻塞直到 completion 线程 `complete` 这个 id（见 28.5.7）。
2. 记录 `cpu_storage_owner_caller_wake` 子项：从 sample.finished_at（completion 线程 mark_finished 的时间）到现在（调用方被唤醒并执行到这里）的间隔——这是 CompletionPool 唤醒调度的延迟。
3. `sample.mark_finished(now)` 再次更新 finished 时间为"调用方真正看到结果"的时间。这一步让 sample.end_to_end_ns 包含唤醒延迟。
4. 如果 sample 完整（finished_flag 为 true），把它加到 `completed_breakdown_report_.insert`——breakdown 报告的 insert 子项。
5. 如果 end_to_end 超过 RPC timeout，记日志（不抛异常，因为这是软超时）。
6. 累加 committed（仅 success）。
7. `release_consumer` 释放 CompletionPool 的 consumer 槽位（让该 completion_id 可以被复用）。

#### 主循环（`public_mutations.cc:64-127`）

```cpp
for (const auto& item : items) {
  const auto operation_started = std::chrono::steady_clock::now();
  u32 owner_storage = 0;
  const std::optional<u32> known_owner =
    known_storage_owner_for_id(item.id);
  if (known_owner.has_value()) {
    owner_storage = *known_owner;
  } else {
    const u32 proposed_owner = num_servers_ == 0
      ? 0
      : static_cast<u32>(item.id % num_servers_);
    owner_storage = claim_storage_owner_for_mutation(
      item.id, proposed_owner);
  }
  lib_assert(owner_storage < storage_insert_owners_.size(),
             "storage-owner route selected an invalid owner");
```

**路由决策**：

- 如果 compute-side idmap 或 base_owner_map 已知该 ID 的 owner（`known_storage_owner_for_id`），直接用。
- 否则按 `id % num_servers_` 提议一个 owner，调 `claim_storage_owner_for_mutation` 抢占。

`claim_storage_owner_for_mutation`（`index_commands.cc:54-75`）：

```cpp
u32 ComputeService::claim_storage_owner_for_mutation(
    node_t id, u32 proposed_owner) {
  auto& shard =
    compute_side_idmap_[static_cast<size_t>(id) % kComputeSideIdShardCount];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto existing = shard.entries.find(id);
  if (existing != shard.entries.end()) {
    return existing->second.owner_storage;
  }
  if (const auto base_owner = base_owner_map_.owner_for(id)) {
    return *base_owner;
  }
  shard.entries.emplace(
    id, ComputeSideIdEntry{RemotePtr{}, true, proposed_owner, 0});
  return proposed_owner;
}
```

这是"计算侧首次见到该 ID 时的 owner 选择"：

1. 加锁后再查一次——避免两个线程同时进 `known_storage_owner_for_id` 都返回 nullopt，然后各自 claim 不同 owner。
2. 如果 base_owner_map 有记录（来自索引元数据，见第 8 课），用 base owner——这是"静态 owner"，对所有计算节点一致。
3. 否则用 `proposed_owner = id % num_servers_`——**这是确定性的**：所有计算节点对同一未观察 ID 都会提议同一 owner。注释明确说："Every compute node proposes the same owner for an unseen ID."

这段解决了"多计算节点协同下的 owner 一致性"问题：即使没有 base owner map，靠 `id % num_servers_` 的确定性也能保证所有计算节点对同一 ID 选出同一 owner，避免 split ownership。

`generation = 0` 表示"本地路由 claim，不是已发布的 mutation"——第一个 stage1 ACK 返回时 generation 从 1 开始，会覆盖这条 claim（见 `publish_compute_side_id` 的 `existing->second.generation >= generation` 检查：generation 0 < 1，会被覆盖）。

#### completion_id 获取（`public_mutations.cc:84-91`）

```cpp
u32 completion_id = 0;
while (!storage_completion_pool_->try_acquire(completion_id)) {
  if (pending.empty()) {
    completion_id = storage_completion_pool_->acquire();
    break;
  }
  consume_one();
}
```

这是**反压流控**的核心：

- `try_acquire` 非阻塞地拿一个 completion_id。
- 如果拿不到（CompletionPool 容量满），且本地 `pending` 也空（没有未等待的 completion），才阻塞 `acquire`。
- 否则 `consume_one()` 等一个已提交的 completion 完成，释放出容量。

这个循环保证：**调用方的 inflight 数被 CompletionPool 容量限制**。CompletionPool 容量是 `total_task_capacity = owner_count * task_capacity`，而 task_capacity 又由 `rpc_depth * batch_max * 8` 推导。所以反压链是：

```
caller inflight ≤ CompletionPool capacity = owner_count * task_capacity
              = owner_count * max(256, rpc_depth * batch_max * 8)
sender inflight ≤ owner_count * rpc_depth  (slot bound)
```

caller 的 inflight 上限远大于 sender 的 inflight 上限（8 倍余量），所以 caller 可以"超前提交"，让 sender 始终有活干，但又不至于把内存撑爆。

#### sample 初始化 + 路由 breakdown（`public_mutations.cc:92-101`）

```cpp
auto& sample = storage_completion_samples_[completion_id];
sample = service::breakdown::Sample(
  service::breakdown::Operation::insert,
  breakdown_enabled_.load(std::memory_order_acquire));
sample.enqueued_at = operation_started;
sample.mark_started(operation_started, operation_started);
const auto route_finished = std::chrono::steady_clock::now();
sample.add_subcategory(
  service::breakdown::Subcategory::cpu_storage_owner_route,
  duration_ns(operation_started, route_finished));
```

`sample.mark_started(start, start)` 两个参数都是 `operation_started`——start_time 是操作开始，end_time 也是 operation_started（因为 sample 还没结束）。`mark_started` 内部会设置 `started_at` 和初始化 end-to-end 计时。`cpu_storage_owner_route` 子项记录路由耗时（`known_storage_owner_for_id` + `claim_storage_owner_for_mutation`）。

#### GPU 容量预约（`public_mutations.cc:103-107`）

```cpp
if (persistent_search_ != nullptr) {
  persistent_search_->reserve_mutation_capacity(1);
}
```

**关键：在 task 进入发送队列之前就预约 GPU 容量**。注释说："Backpressure happens here, never after owner memory has been mutated."——反压必须发生在 owner memory 被修改之前，否则就会出现"owner 已 commit 但 GPU 没容量发布"的困境。

`reserve_mutation_capacity(1)`（`persistent_engine.cc:181-234`）会阻塞等待，直到 GPU delta 表和 resident PQ 表都有 90%/95% 水位以下的余量。这是 dvstor 的"硬反压点"：当 GPU 更新积压到接近容量上限时，新的 mutation 在这里被挡住，让 owner 也跟着降速（owner 的 stage1 ACK 必须等计算侧发出 RPC 才能产生）。

#### task 入队（`public_mutations.cc:109-124`）

```cpp
auto& state = *storage_insert_owners_[owner_storage];
u32 task_id = 0;
state.free_tasks->pop_wait(task_id);
auto& task = state.tasks[task_id];
task.item.id = item.id;
if (kind == service::storage_owner::MutationKind::erase) {
  task.item.values.clear();
} else {
  task.item.values.assign(item.values.begin(), item.values.end());
}
task.kind = kind;
task.completion_id = completion_id;
task.enqueued_at = std::chrono::steady_clock::now();
task.sender_dequeued_at = {};
state.queue->push_wait(task_id);
pending.push_back(completion_id);
```

`free_tasks->pop_wait` 阻塞等一个空闲 task_id——这是另一道反压：如果 task pool 满了（所有 task 都在 queue 里或 in-flight），调用方阻塞在这里。

填充 task：erase 清空 values（节省内存），其他 kind 拷贝向量。`enqueued_at` 是入队时间，sender drain 时会读它算 `sender_queue_wait`。`push_wait` 把 task_id 推进 owner 的提交队列，`pending.push_back` 记录 completion_id 等待后续 `consume_one`。

#### 收尾（`public_mutations.cc:127-128`）

```cpp
while (!pending.empty()) consume_one();
return committed;
```

所有 item 提交完后，把 `pending` 里的 completion 全部 wait 完。返回 `committed`（成功的 item 数）。注意如果 caller 传了 1000 条，前面 999 条可能在循环里就被 `consume_one` 等完了（因为 CompletionPool 容量限制），最后只剩少量 pending。

## 28.7 `response_validation.hh`：响应校验

`response_validation.hh` 是个 inline 头文件，只定义一个枚举和一个校验函数。

```cpp
enum class StorageOwnerResponseValidation {
  unmatched,
  matched_invalid,
  matched_valid,
};

inline StorageOwnerResponseValidation validate_storage_owner_response(
    const service::storage_owner::InsertBatchResponseHeader& response,
    size_t received_bytes,
    size_t response_buffer_bytes,
    u32 expected_magic,
    u32 expected_owner,
    u32 expected_item_count,
    u64 expected_batch_id,
    size_t expected_response_bytes) {
  if (response.batch_id != expected_batch_id) {
    return StorageOwnerResponseValidation::unmatched;
  }
  const bool valid =
    response.magic == expected_magic &&
    response.owner_storage == expected_owner &&
    response.item_count == expected_item_count &&
    expected_response_bytes <= response_buffer_bytes &&
    received_bytes == expected_response_bytes;
  return valid
    ? StorageOwnerResponseValidation::matched_valid
    : StorageOwnerResponseValidation::matched_invalid;
}
```

三态设计：

- `unmatched`：`batch_id` 不一致。这种情况在 `handle_storage_owner_response` 里理论上不会发生（因为已经按 batch_id 匹配过），但作为防御性检查保留——如果发生说明 ready 队列协议被破坏。
- `matched_invalid`：batch_id 匹配，但其他字段不一致。这表示响应损坏或协议不匹配，整批按失败处理。
- `matched_valid`：全部字段一致，响应可信。

校验项：

1. `magic`：响应 magic 与请求 magic 一致。
2. `owner_storage`：响应来自预期的 owner。
3. `item_count`：响应包含的 item 数与请求一致。
4. `expected_response_bytes <= response_buffer_bytes`：预期响应大小不超过缓冲容量（不会越界读）。
5. `received_bytes == expected_response_bytes`：实际收到的字节数与预期完全一致（既不多也不少）。

注意第 5 项是**严格相等**——verbs RECV 不会返回部分数据，但可能因为对端发错大小而 mismatch。严格检查能及时发现协议 bug。

## 28.8 端到端时序图

把上面所有片段串起来，一个客户端 mutation 的完整生命周期：

```
                调用方线程                progress 线程              completion 线程            owner (storage 节点)
                   |                          |                           |                          |
insert(items)      |                          |                           |                          |
  known_storage_owner_for_id / claim          |                           |                          |
  reserve_mutation_capacity(1)  ──────────────┼─────── (阻塞等待 GPU 容量)                          |
  free_tasks.pop_wait(task_id)                |                           |                          |
  queue.push_wait(task_id)  ───────────────►  |                           |                          |
  pending.push_back(completion_id)            |                           |                          |
                   |                          |                           |                          |
                   |     (循环 poll CQ)       |                           |                          |
                   |     drain_storage_owner_submissions                  |                          |
                   |     drain: pop task_id, 组 batch (≤ batch_max)       |                          |
                   |     post_storage_owner_batch                         |                          |
                   |       encode ids/kinds/vectors → request_buffer      |                          |
                   |       post_send ──────────────────────────────────────────────────────────────►  |
                   |                          |                           |                          |
                   |                          |                           |        stage1 处理        |
                   |                          |                           |        (图修改/反向/PQ)   |
                   |                          |                           |                          |
                   |                          |     ◄──────────────────────────────────────────────  post_send (response)
                   |     poll_recv_cq         |                           |                          |
                   |     handle_storage_owner_response                    |                          |
                   |       按 batch_id 匹配发送 slot                       |                          |
                   |       validate_storage_owner_response                |                          |
                   |       queue_storage_owner_completion (send_done && response_done)
                   |       storage_ready_slots_.try_push ──────────────►  |                          |
                   |                          |                           |                          |
                   |                          |     pop_wait(ready)       |                          |
                   |                          |     commit_storage_owner_slot                        |
                   |                          |       |                   |                          |
                   |                          |       | response_ok 校验 |                          |
                   |                          |       | per-item:        |                          |
                   |                          |       |   publish_compute_side_id (generation 检查)  |
                   |                          |       |   填充 DeltaMutation (预分配缓冲)            |
                   |                          |       | complete_storage_owner_task (唤醒 caller)    |
                   |                          |       |                   |                          |
                   |                          |     publish_storage_owner_mutations                  |
                   |                          |       sort+unique invalidated                        |
                   |                          |       persistent_search_->publish_mutations ─► GPU   |
                   |                          |         reserve_epoch                                |
                   |                          |         upload_mutations (CUDA 命令)                 |
                   |                          |         delta_.publish_metadata (epoch 发布)        |
                   |                          |       release_mutation_capacity                      |
                   |                          |     release_storage_owner_slot                       |
                   |                          |       storage_released_slots_.try_push ─►           |
                   |                          |                           |                          |
                   |     reclaim_storage_owner_slots                      |                          |
                   |       重置 slot 状态      |                           |                          |
                   |       free_slots.push_back(slot_id)                  |                          |
                   |       post_storage_owner_response_receive (重新 post recv buffer)              |
                   |       inflight -1        |                           |                          |
  consume_one:     |                          |                           |                          |
    completion_pool.wait(completion_id) ◄─────┼───────────────────────────  complete(success)       |
    sample.mark_finished + add_sample         |                           |                          |
    release_consumer                          |                           |                          |
  return committed                            |                           |                          |
```

关键阶段总结：

| 阶段 | 触发者 | 动作 | 反压点 |
| --- | --- | --- | --- |
| 路由 + 容量预约 | 调用方 | 选 owner、reserve GPU 容量 | `reserve_mutation_capacity` 阻塞 |
| task 入队 | 调用方 | `free_tasks.pop_wait` + `queue.push_wait` | task pool 满 / 提交队列满 |
| batch 组装 + post_send | progress | drain queue、组 batch、post_send | free_slots 空（rpc_depth 用完） |
| stage1 处理 | owner | 图修改、反向、PQ | owner 端（见第 23 课） |
| 响应匹配 + 校验 | progress | poll recv CQ、batch_id 匹配、validate | 无（无阻塞） |
| ready 入队 | progress | `queue_storage_owner_completion` | ready 队列满（不可能，见 28.5.4） |
| commit + GPU 发布 | completion | `commit_storage_owner_slot` + `publish_mutations` | `publish_mutations` 同步阻塞 |
| slot 回收 | progress | `reclaim_storage_owner_slots` | released 队列满（不可能） |
| 调用方唤醒 | 调用方 | `completion_pool.wait` 返回 | 无（被 completion 线程唤醒） |

## 28.9 与其他模块的关系

- **第 8 课（schema-15 索引格式 / 存储协议）**：`storage_owner_protocol.hh` 的 wire 格式（Insert/MutationBatchRequestHeader、InsertBatchResponseHeader、MutationResult、InsertBreakdownCounters）是第 8 课协议定义的具体落地。`anchor_hint_count = 0` 是 schema-15 兼容性约束。
- **第 15 课（增量发布）**：`publish_storage_owner_mutations` → `PersistentSearchEngine::publish_mutations` 是第 15 课 delta_publication 的入口。`reserve_epoch`、`upload_mutations`、`publish_metadata` 三步构成 GPU 可见性发布。`mark_committed_mutation_gap` 是第 15 课"一致性裂缝"检测的触发点。
- **第 16 课（存储回收 RCU）**：`reserve_mutation_capacity` / `release_mutation_capacity` 与第 16 课的"retired slot 回收"联动——`reclaim_retired_delta_slots_locked`（`delta_publication.cc:301-357`）在 reserve 时被调用，回收已被 RCU 退休的 delta slot。`query_ticket_barrier_passed` 是 RCU grace period 的等价物。
- **第 23 课（存储节点主体 / peer RDMA）**：本课的 `post_send` 投递到 owner 的 recv 队列，由第 23 课的存储节点主体接收并处理。响应方向反过来：owner post_send，计算侧 post_receive。`kInsertMagic` / `kMutationMagic` 在两端共享。
- **第 26 课（维护 / wire protocol）**：`PeerRpcHeader` / `kPeerRpcMagic`（`storage_owner_protocol.hh:12-13, 145-154`）是 owner 之间反向更新、stitch search 的 wire 协议，与第 26 课对应。本课的 stage1 响应里的 `invalidated_raws` 就来自 owner 间的反向更新。
- **第 27 课（ComputeService 主体）**：本课所有函数都是 `ComputeService` 的方法，`compute_service.hh` 的成员（`storage_insert_owners_`、`persistent_search_`、`cm_`、`context_`、`base_owner_map_`、`compute_side_idmap_` 等）由第 27 课构造和初始化。`start_storage_insert_runtime` / `stop_storage_insert_runtime` / `release_storage_insert_runtime` 由 ComputeService 的 `start` / `stop` 流程调用。
- **第 10 课（delta / 动态路由 / 预算）**：`DeltaMutation` 的 `owner_storage`、`remote_node`、`old_remote_node` 是第 10 课动态路由的关键字段。`publish_compute_side_id` 维护的 compute-side idmap 是第 10 课"计算侧路由表"的实现。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`PersistentSearchEngine` 的 PImpl 模式（`impl_` unique_ptr）由第 11 课定义。本课调用的 `publish_mutations` / `reserve_mutation_capacity` / `release_mutation_capacity` / `mark_committed_mutation_gap` 都是 PImpl 转发到 `Impl`。
- **第 30 课（breakdown benchmark）**：本课大量的 `sample.add_subcategory`、`add_storage_owner_sender_breakdown`、`add_storage_owner_breakdown` 调用是第 30 课 breakdown 报告的数据来源。`storage_completion_samples_` 数组按 `total_task_capacity` 分配，与第 30 课的采样容量对齐。

## 28.10 小结

本课讲解了计算侧 storage owner 更新子系统的五个文件：

1. **`lifecycle.cc`**：per-owner 预分配请求/响应缓冲、MR、RPC slot、task pool；启动 progress + completion 两条线程。所有 hot path 缓冲都在启动时预分配，零运行时分配。
2. **`sender.cc`**：progress 线程单线程 work-conserving，轮询 send/recv CQ、owner round-robin drain 提交队列组 batch、post_send。批大小自适应（1 到 batch_max），按 batch_id 关联 send/recv。
3. **`completion.cc`**：completion 线程在 stage1 ACK 到达后做四件事——响应校验、compute-side idmap generation 检查（防止乱序完成复活旧路由）、构造预分配的 `DeltaMutation` 列表、同步发布到 GPU。失败路径 `mark_committed_mutation_gap` 标记引擎不健康。
4. **`public_mutations.cc`**：同步入口 `insert/upsert/erase`，通过 `submit_storage_owner_mutations` 实现"路由 + GPU 容量预约 + task 入队 + 同步等待"。两层反压：`reserve_mutation_capacity`（owner memory 之前）+ CompletionPool 容量（caller inflight 限制）。`claim_storage_owner_for_mutation` 靠 `id % num_servers_` 的确定性保证多计算节点 owner 一致性。
5. **`response_validation.hh`**：三态响应校验（unmatched / matched_invalid / matched_valid），严格字节数检查。

核心设计原则：

- **stage1 ACK 与 GPU 可见性解耦**：stage1 只承诺 owner-memory 已落定，GPU 可见性是另一条有序异步发布流。但"已 commit 必须 GPU 可见"是一致性硬约束，违反时引擎标记不健康。
- **预分配 + 同步发布**：所有 hot path 缓冲在 lifecycle 预分配；GPU 发布是同步调用，返回后缓冲可立即复用。这避免了"发布后缓冲悬挂引用"的复杂性。
- **generation 防乱序**：compute-side idmap 按 generation 拒绝旧响应，GPU 发布流只接受 newest generation。这关闭了"旧响应在新响应之后完成导致复活旧路由"的窗口。
- **三层容量联动**：`rpc_depth`（per-owner 在途 RPC 上限）× `batch_max`（per-batch item 上限）× 8 = `task_capacity`（per-owner task pool 上限）；`owner_count * task_capacity` = CompletionPool 容量（caller inflight 上限）。反压逐级传递，最终源头在 `reserve_mutation_capacity`。
- **确定性 owner 路由**：`id % num_servers_` + base_owner_map 保证所有计算节点对同一 ID 选出同一 owner，无需协调。

下一课（第 29 课）将讲解离线构建与迁移流程，与本课的"在线 mutation"互补——离线构建产生初始索引，在线 mutation 在初始索引上叠加 delta。
