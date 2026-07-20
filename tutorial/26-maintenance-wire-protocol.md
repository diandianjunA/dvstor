# 第 26 课 · 存储侧维护与 runtime wire protocol

> 配套代码：`src/memory_node/storage_owner_maintenance/`（stage2 维护子系统）与 `src/memory_node/storage_owner_runtime/`（foreground runtime / wire 协议解码）。两套子系统共同构成存储节点对"mutation 批"的完整处理链路：runtime 接收并执行 stage1，maintenance 在 stage1 ACK 之后异步执行 stage2 finalize，把分散的临时反向边收敛成权威邻接表，并推进 durable sequence，触发计算侧回收。

## 本课目标与涉及文件

读完本课你应当能回答：

1. 计算/前台路径发出一个 mutation 批之后，存储节点如何从 RDMA recv CQ 一路走到 stage1 完成 + stage2 入队？
2. stage2 维护 worker 的"local_ready → remote_search_pending → prune_ready → reverse_pending → finalized"状态机是如何驱动的？谁负责并发外部分片候选、谁负责合并 beam、谁负责 RobustPrune、谁负责权威反向边 ACK？
3. durable watermark（`durable_maintenance_sequence`）是如何被推进的？为什么必须保留至少一个 maintenance worker 防止 watermark 饿死？
4. mutation 批的 wire 编码（`InsertBatchRequestHeader` / `MutationBatchRequestHeader` / `PeerRpcHeader` / `ReverseUpdateOp`）与第 8 课 `storage_owner_protocol` 是如何对接的？

涉及文件（按出场顺序）：

- `src/memory_node/storage_owner_runtime/detail.hh`（runtime 子系统的模式判定 helper）
- `src/memory_node/storage_owner_runtime/lifecycle.cc`（runtime 生命周期：recv CQ 循环、insert worker 启停、slot 偏移）
- `src/memory_node/storage_owner_runtime/workers.cc`（前台 insert worker 主循环 + 同步执行路径 `execute_storage_owner_batch_items`）
- `src/memory_node/storage_owner_runtime/batch_execution.cc`（异步执行路径 `execute_storage_owner_batch_items_async` + 协程调度）
- `src/memory_node/storage_owner_runtime/wire_protocol.cc`（前台 RPC 主循环 `service_storage_runtime` 与同步 mutation 处理 `handle_storage_insert_request`）
- `src/memory_node/storage_owner_maintenance/detail.hh`（维护子系统公共常量/小工具）
- `src/memory_node/storage_owner_maintenance/admission_policy.hh`（stage2 准入策略）
- `src/memory_node/storage_owner_maintenance/cleanup_policy.hh`（cleanup 邻居选择与 rebase 合并）
- `src/memory_node/storage_owner_maintenance/stage2_tracker.hh`（stage2 状态机 + 请求跟踪器，499 行）
- `src/memory_node/storage_owner_maintenance/reverse_outbox.hh`（反向边发件箱：coalesce/async/sync，555 行）
- `src/memory_node/storage_owner_maintenance/graph_tasks.cc`（图任务原语：node lock / task current / preserved neighbor / 批量删边）
- `src/memory_node/storage_owner_maintenance/queue.cc`（有界维护队列、watermark 发布、调度入队、背压，296 行）
- `src/memory_node/storage_owner_maintenance/runtime.cc`（维护 runtime 启停 + 观测日志，487 行）
- `src/memory_node/storage_owner_maintenance/worker.cc`（stage2 worker 主循环，1217 行）

---

## 一、子系统定位与两阶段总览

dvstor 的"storage owner"是 schema-15 索引（见第 7 课）的权威持有者，每个存储分片负责本分片节点的物理分配、邻接表写入、反向边维护与删除回收。计算侧通过 mutation 批（insert/upsert/erase 混合）发起变更（见第 15 课增量发布与第 28 课计算侧 sender），mutation 的执行被刻意拆成两个阶段：

- **stage1（前台 / foreground）**：执行 prepare_mutation、allocate_node、`write_new_node`、`publish_mutation`、`robust_prune_cpu` 选出"临时"出边集合、把临时的本地反向边立即应用、把跨分片临时反向边通过 peer RPC fire-and-forget。stage1 必须在 mutation 响应回程之前完成，因为它要给计算侧返回 `MutationResult`（含 `new_rptr_raw`、`old_rptr_raw`、`generation`、`maintenance_sequence`）。
- **stage2（后台 / maintenance / finalize）**：对 stage1 留下的 intent 做最终收敛——向外部分片发起 `stitch_search_request` 收集候选、把所有候选（本地 + 远端）做一次 RobustPrune、把权威反向边以 `reverse_update_request` 形式向每个相关分片发送并等待 ACK，ACK 齐了之后推进 `durable_maintenance_sequence`。计算侧看到 durable 推进就可以回收旧的物理节点（见第 16 课存储回收 RCU）。

之所以要拆两段，是因为 stage1 必须快（响应延迟敏感），而真正的图收敛需要跨分片 RPC、需要等待 ACK、需要 rebase——这些都不能阻塞前台响应。`storage_owner_update_mode == "local_stitch"` 时，stage1 只做本地收敛，stage2 承担全部跨分片工作；否则 stage1 已经做完跨分片临时反向边，stage2 只做权威收敛。无论哪种模式，stage2 都是 durable watermark 推进的唯一来源。

`storage_owner_maintenance_mode` 配置取值 `"finalize"` 且 `storage_owner_maintenance_workers > 0` 时 stage2 才启用（见 `runtime.cc:6-9`），否则 stage1 直接 commit、维护序立即 complete，watermark 在 stage1 路径里推进。

---

## 二、runtime 子系统：mutation 批的入口

### 2.1 `detail.hh`：模式判定

```cpp
// src/memory_node/storage_owner_runtime/detail.hh
inline bool storage_owner_local_stitch_mode(const configuration::IndexConfiguration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}
inline bool storage_owner_batch_is_local_stage1(const configuration::IndexConfiguration& config) {
  return storage_owner_local_stitch_mode(config);
}
```

`local_stitch` 是 stage1 的两种执行模式的开关。开（`"local_stitch"`）：stage1 只做本分片 medoid 路由 + 本地 beam search + 本地 prune + 本地反向边，跨分片候选全部留给 stage2。关：stage1 走完整的 `beam_search_candidates`，并直接把跨分片反向边用 `send_reverse_update_batch` 发出去（见 `wire_protocol.cc:420-435`）。该 helper 同时被 `workers.cc`、`batch_execution.cc`、`lifecycle.cc` 使用。

### 2.2 `lifecycle.cc`：runtime 生命周期与 RDMA slot 布局

`setup_insert_runtime`（`lifecycle.cc:6-38`）负责一次性分配 RPC 缓冲区与 MR：

```cpp
const size_t insert_request_bytes = align_up(std::max(
    service::storage_owner::insert_batch_request_bytes(config.storage_owner_batch_max),
    service::storage_owner::mutation_batch_request_bytes(config.storage_owner_batch_max)));
insert_runtime_.request_bytes = insert_request_bytes;
insert_runtime_.request_slot_count = std::max<u32>(1, config.storage_owner_rpc_depth);
```

注意 `insert_request_bytes` 取 `insert_batch_request_bytes` 与 `mutation_batch_request_bytes` 的最大值再 `align_up`——这是因为同一套 slot 既要能接"纯 insert"（`kInsertMagic`）又要能接"mutation 批"（`kMutationMagic`，见 `storage_owner_protocol.hh:10-11`）。mutation 批比纯 insert 多一段 `u32 kinds[item_count]`（见 `storage_owner_protocol.hh:186-191`），所以 slot 容量按 mutation 取。

slot 总数 = `num_clients_ * request_slot_count`，且 `lib_assert(slot_count <= config.max_recv_queue_wr)`——这是协议层与 verbs 接收 CQ 容量的硬约束：每个 (client, slot) 恰好一条 recv WR，预注册、循环 repost，永不阻塞 ingress。缓冲区布局是 `[request_slots... | response_slots...]`，`response_offset = request_bytes * slot_count`（`lifecycle.cc:26`）。

`start_storage_owner_insert_workers`（`lifecycle.cc:40-117`）启动前台 worker 池。关键点：

1. CPU 计划由 `derive_storage_owner_cpu_plan` 统一分配（见第 23 课存储节点主体），把 foreground / maintenance / peer_search / peer_reverse / progress 各类线程按可用核数与 rpc 并行度切成不重叠的集合。
2. 每个 `StorageOwnerThread` 在非 local_stitch 模式下会 `init_peer_scratch`，预分配协程 scratch（`snapshot_stride * snapshot_batch + max(total_size, neighbor_stride)`），保证协程恢复时不分配堆内存。
3. worker 数 = `cpu_plan.foreground_workers`（work-conserving：可能与配置值不同，受核数约束）。
4. `storage_owner_async_candidates_` 是 `[worker_count][coroutine_count]` 的二维候选缓冲，给协程切换用。

`storage_owner_insert_worker_loop`（`lifecycle.cc:119-134`）是 worker 主循环：

```cpp
for (;;) {
  StorageOwnerInsertTask task;
  if (!storage_insert_tasks_->pop_wait(task, storage_insert_shutdown_)) {
    current_storage_owner_thread_ = nullptr;
    return;
  }
  mark_storage_owner_foreground_activity();
  storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
  process_storage_owner_insert_task(task);
  storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
  mark_storage_owner_foreground_activity();
}
```

`storage_insert_tasks_` 是有界 `bounded::Queue<StorageOwnerInsertTask>`，容量 = slot 总数。`storage_owner_insert_active_workers_` 是 stage2 准入判断的"前台忙"信号之一（见 §3.4）。`mark_storage_owner_foreground_activity` 把 `storage_owner_foreground_last_active_ns_` 推到当前，给 stage2 背压用。

### 2.3 `wire_protocol.cc`：RDMA CQ 主循环与 mutation 解码

`service_storage_runtime`（`wire_protocol.cc:5-104`）是存储节点对计算侧的 RPC 入口主循环。它在启动时为每个 (client, slot) `post_receive` 一次，然后进入 `for(;;)`：

1. `context_.poll_send_cq` 取回已完成的 send WR（响应已发出），并对该 (client, slot) 重新 `post_receive`——这是"一收一发"的 repost 模型。
2. `context_.poll_recv_cq` 取回新到达的请求。对每条 recv WC：
   - 取 `payload`（即 `insert_runtime_.buffer` 中对应 slot 偏移），读 `InsertBatchRequestHeader`（注意 mutation 与 insert 共用同一种 header 布局，仅 `magic` 不同）。
   - 校验 `magic`、`dim`、`owner_storage == storage_id_`、`item_count ∈ (0, storage_owner_batch_max]`、`vector_dtype`、`vector_bytes`、`anchor_hint_count == 0`、`bytes >= expected_bytes`。任何不符则落到 `handle_storage_insert_request` 同步路径返回错误。
   - 全部满足则构造 `StorageOwnerInsertTask` 并 `storage_insert_tasks_->try_push`。注意 `lib_assert(...try_push...)`：因为队列容量 = slot 总数，而每个被占用的 slot 至多一条 task，所以 try_push 不可能失败——这是协议层给出的容量不变式。

`handle_storage_insert_request`（`wire_protocol.cc:110-208`）是同步 fallback 路径（当无法异步派发时直接在 poll 线程内执行）。它与 `process_storage_owner_insert_task` 几乎对称，都走 `execute_storage_owner_batch_items`（同步版），区别只是同步 fallback 不进入 worker 池。

`execute_storage_owner_batch_items`（`wire_protocol.cc:210-471`）是同步 mutation 批执行的核心。**这是 stage1 的真正实现**。流程：

1. **预占维护序**（`wire_protocol.cc:240-253`）：

   ```cpp
   vec<u32> reserved_maintenance_work_items(item_count);
   for (size_t idx = 0; idx < item_count; ++idx) {
     const auto kind = kinds == nullptr ? MutationKind::insert : kinds[idx];
     reserved_maintenance_work_items[idx] =
         storage_owner_maintenance_work_items(kind, config);
   }
   const u64 first_reserved_maintenance_sequence =
       begin_storage_owner_maintenance_batch(reserved_maintenance_work_items);
   ```

   `storage_owner_maintenance_work_items`（`queue.cc:147-164`）按 mutation kind 决定 stage2 工作量：insert/erase = 1，upsert = 2（既要 stitch 又要 cleanup）。`begin_storage_owner_maintenance_batch`（`queue.cc:82-93`）调用 `SlidingCompletionRing::reserve_batch` 一次性分配整批 sequence，保证 batch 内 sequence 连续、且整个 batch 能落入 admission window（`storage_owner_maintenance_admission_limit_`）。

   `queue.cc:152-155` 的注释值得专门读：

   > Even when stage2 is disabled, keep one completion unit until the maintenance intent has been published. Returning zero lets reserve_batch finalize and recycle the modulo slot before schedule_storage_owner_maintenance writes the intent, so concurrent foreground workers can race while writing the same non-atomic intent fields.

   也就是说即便 stage2 关闭，也必须返回 1（而不是 0）作为 work item——`reserve_batch` 一旦返回 0 就会立即 finalize 并回收 modulo 槽位，而此时 `schedule_storage_owner_maintenance` 还没写入 intent 字段，并发的前台 worker 可能踩到同一槽位。这是 stage1/stage2 协调的细节关键。

2. **逐项执行 mutation**（`wire_protocol.cc:284-436`）：对每条 item：
   - `prepare_mutation` 查 dynamic freshness shard 与 base idmap，返回旧 `FreshnessEntry` 与新一代 `generation`。
   - erase：`mark_node_deleted`、`publish_mutation(deleted=true)`、`invalidate_storage_owner_route`、`schedule_storage_owner_maintenance` 入 cleanup 队列。
   - insert/upsert：根据 `local_stitch` 与 medoid 是否存在选择候选来源；`robust_prune_cpu` 选出 stage1 临时邻居；`allocate_local_node` + `write_new_node`；`publish_mutation`；`observe_storage_owner_route`；`schedule_storage_owner_maintenance` 入 stitch 队列（local_stitch 模式下把 `candidates` 与 `selected_neighbors` 一并交给 stage2 做 rebase 基线）。
   - stage2 关闭时，在 stage1 内直接把反向边应用：本地走 `local_updates` + `apply_partition_local_reverse_update`，远端走 `remote_updates` + `send_reverse_update_batch`。

3. **批量反向边**（`wire_protocol.cc:438-469`）：本地反向边批量 apply，远端反向边按目标分片批量发送。这是 stage1 fire-and-forget 的临时反向边。

`process_storage_owner_insert_task`（`batch_execution.cc:5-141`）是 worker 路径的入口，与 `handle_storage_insert_request` 对称：解码 wire → 选择同步/异步执行 → 构造响应 → `post_storage_owner_response`。响应布局是 `InsertBatchResponseHeader + statuses[item_count] + MutationResult[item_count] + InsertBreakdownCounters + invalidation_count + invalidated_raws[item_count*R]`（见 `storage_owner_protocol.hh:193-200`）。`invalidated_raws` 是计算侧路由失效通知——告诉计算侧"这些 raw 节点指针的图结构变了，缓存的路由要失效"。

### 2.4 `batch_execution.cc`：异步执行与协程调度

`execute_storage_owner_batch_items_async`（`batch_execution.cc:5-146`）是 local_stitch 模式下的 stage1 执行路径（`workers.cc:60-82` 的 `local_stage1` 判定选择 sync 还是 async）。与同步版相比，差异在：

1. 同样 `begin_storage_owner_maintenance_batch` 预占序（`batch_execution.cc:19-27`）。
2. 把每条 item 包成 `StorageOwnerInsertJob`，把 `reserved_maintenance_sequence + idx` 作为该 item 的序（`batch_execution.cc:36-37`）。
3. 用 `coroutine_count = thread.post_balances.size()` 个协程并发跑 job（`batch_execution.cc:48-87`）：每个协程是 `execute_storage_owner_insert_job_async` 返回的 `StorageOwnerInsertCoroutine`。主循环每轮 `poll_peer_send_cq`，然后对每个协程：done 且有剩余 job → 销毁重建；ready → resume；否则跳过。全部 done 时退出。
4. 协程内部会做 peer RDMA 读（取候选节点向量），`thread.is_ready(coroutine_id)` 判断对应 post balance 是否完成。
5. 协程全部结束后，再统一处理本地反向边（`apply_partition_local_reverse_update`）与远端反向边（`send_reverse_update_batch`）。

协程原语见第 3 课。这里只需要注意：异步路径下 stage1 不直接发跨分片临时反向边——因为 local_stitch 模式下，跨分片候选留给 stage2 的 `stitch_search_request` 完成。`batch_execution.cc:89-92` 销毁所有协程句柄，避免悬挂。

`post_storage_owner_response`（`batch_execution.cc:143-178`）通过 `cm_.client_qps[client_id]->post_send_with_id` 把响应发回计算侧，wr_id 编码 `(client_id, slot_id)` 以便 send CQ 回调时 repost。

---

## 三、maintenance 子系统：stage2 finalize

### 3.1 `detail.hh`：常量与小工具

```cpp
// src/memory_node/storage_owner_maintenance/detail.hh
inline constexpr u64 kMaintenanceObservationPeriodNs = 5ull * 1000ull * 1000ull * 1000ull;  // 5s
inline constexpr u64 kStitchCompactionMaxDelayNs = 10ull * 1000ull * 1000ull;  // 10ms
inline constexpr size_t kForegroundQueueYieldMultiplier = 2;
inline constexpr std::array<u64, 18> kFinalizeLatencyBucketUpperNs{ ... };
```

`kMaintenanceObservationPeriodNs` 是观测日志的最小间隔（5 秒）。`kStitchCompactionMaxDelayNs` 是 stitch 压实的最大允许延迟（10ms），超过意味着 stage2 跟不上。`kFinalizeLatencyBucketUpperNs` 是 finalize 延迟的 18 个桶上界（1ms → 30s → max），用于 p99 直方图。

`queue_near_limit`（`detail.hh:73-77`）取 3/4 阈值；`counter_above_fraction(value, limit, num, den)` 判断 `value >= max(1, limit*num/den)`。这两个是 stage2 背压探测的统一阈值工具。

### 3.2 `admission_policy.hh`：stage2 准入

```cpp
// src/memory_node/storage_owner_maintenance/admission_policy.hh
template <class ForegroundPressureProbe>
Stage2AdmissionDecision decide_stage2_admission(
    bool local_contexts_full, bool shutting_down,
    ForegroundPressureProbe&& foreground_pressure) {
  if (local_contexts_full || shutting_down) {
    return Stage2AdmissionDecision::unavailable;
  }
  return std::forward<ForegroundPressureProbe>(foreground_pressure)()
           ? Stage2AdmissionDecision::foreground_pressure
           : Stage2AdmissionDecision::admit;
}
```

三态决策：`unavailable`（contexts 满或关停中）→ 不准入；`foreground_pressure`（前台忙）→ 让出；`admit` → 准入。注意注释强调："production probe 可能 poll 共享 peer CQ，所以 shutdown/full 路径必须无副作用"——即 probe 是惰性求值的，前两个分支不会调用它。

### 3.3 `cleanup_policy.hh`：cleanup 邻居选择与 rebase

```cpp
// src/memory_node/storage_owner_maintenance/cleanup_policy.hh
inline vec<RemotePtr> select_cleanup_neighbors(
    bool repair_only,
    span<const RemotePtr> preserved_neighbors,
    span<const RemotePtr> supplemental_neighbors) {
  vec<RemotePtr> selected;
  selected.reserve((repair_only ? 0 : preserved_neighbors.size()) +
                   supplemental_neighbors.size());
  const auto append_unique = [&](span<const RemotePtr> neighbors) { ... };
  if (!repair_only) {
    append_unique(preserved_neighbors);
  }
  append_unique(supplemental_neighbors);
  return selected;
}
```

cleanup 的两类邻居来源：

- **preserved**：从被删除节点的 hot graph entry（见第 6 课 Vamana 图格式）里读出的"保留邻接表"——`read_preserved_neighbor_list`（`graph_tasks.cc:43-77`）通过 `edge_count + checksum16` 校验后解码。
- **supplemental**：stitch 失败时由 `handoff_stitch_cleanup` 传入的"这次 stitch 已经装上去的反向边目标"。

`repair_only == true` 时只取 supplemental（即只回滚这次 stitch 装的反向边），不取 preserved。注释（`cleanup_policy.hh:10-14`）解释为什么必须分开：stale stitch 的修复只拥有"它自己装的那些反向边"的撤销权；让 stitch 变 stale 的那次 erase/upsert 有自己的 ordinary cleanup intent 负责 preserved 邻接。两套合一可能达到 2R 操作，超出 schema-15 每条 item 的 R 上限，所以必须拆开。

`merge_stage2_rebase_candidates`（`cleanup_policy.hh:44-66`）是 stage2 finalize 时 commit 前的 rebase：把 stage2 全局 prune 的结果（`globally_pruned`）与"stage1 之后才出现的新邻居"（`observed_neighbors - stage1_neighbors`）合并。注释（`cleanup_policy.hh:42-44`）解释：stage1 的临时邻接在 stage2 远程搜索期间可能已经被另一个并发 stage2 装了反向边进来，这些"新邻居"是已完成的工作，不能丢。

### 3.4 `queue.cc`：有界队列、watermark、调度

#### 入队与背压

`enqueue_storage_owner_maintenance`（`queue.cc:5-35`）：

```cpp
std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
storage_owner_maintenance_cv_.wait(lock, [&]() {
  return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
         storage_owner_stitch_tasks_.size() + storage_owner_cleanup_tasks_.size() <
             config.storage_owner_maintenance_queue_depth;
});
```

这是有界背压：当 stitch + cleanup 队列总和达到 `storage_owner_maintenance_queue_depth` 时，入队者（即 stage1 的 `schedule_storage_owner_maintenance`）会被挂起在条件变量上，直到 stage2 worker 消费掉一些。这一步是 stage1 反压 stage2 的关键——队列不是无界的，stage2 慢了 stage1 会停下来等。

注意分两个 deque：`storage_owner_stitch_tasks_` 与 `storage_owner_cleanup_tasks_`。`enqueue_insert_stitch` 与 `enqueue_deleted_node_cleanup`（`queue.cc:37-75`）分别按 kind 入队，并各自维护独立的 `finalize_enqueued` / `cleanup_enqueued` 计数。`task.queued_at = std::chrono::steady_clock::now()` 记录入队时刻，给 finalize 延迟统计用。

#### watermark 发布

`publish_storage_owner_maintenance_watermarks`（`queue.cc:95-118`）把 completion ring 的 `next_sequence()` 与 `finalized()` 写入 `StorageControlBlock`：

```cpp
auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
std::atomic_ref<u64> next(control->next_maintenance_sequence);
std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
// CAS 推进 next 与 durable（单调，只升不降）
```

`StorageControlBlock` 是索引缓冲区开头的控制块（见第 7 课 schema-15 与第 16 课回收），计算侧通过 RDMA read 或本地映射观察 `durable_maintenance_sequence` 来判断哪些旧节点可以回收。`next` 是"已分配的下一个维护序"，`durable` 是"已 finalize 的最高序"。两个 CAS 循环保证单调推进——`observed_next < desired_next` 时才 CAS，避免回退。

`complete_storage_owner_maintenance_sequence`（`queue.cc:124-133`）在 stage2 finalize 完一条 task 或 stage1 取消一条预留时调用，`completion_ring_->complete(seq, work_items)` + `publish_watermarks` + `notify_all`。`notify_all` 是为了唤醒：①等待背压的入队者；②等待 durable 推进的回收路径；③等待 shutdown drain 的 stop 函数。

#### cleanup 就绪判定

`storage_owner_cleanup_ready`（`queue.cc:135-145`）：

```cpp
bool MemoryNode::storage_owner_cleanup_ready(u64 sequence) const {
  if (sequence <= 1) return true;
  ...
  const u64 durable = std::atomic_ref<u64>(durable_storage).load(std::memory_order_acquire);
  return durable >= sequence - 1;
}
```

cleanup task 必须等"前一序" finalize 之后才能执行。原因（见 `worker.cc:895-908` 的注释）：一次 upsert 既有 stitch（新节点入图）又有 cleanup（旧节点回收），cleanup 必须等 stitch 完成确保新节点的反向边已经装好，否则先 cleanup 旧节点会破坏正在进行的 stitch。`sequence <= 1` 直接放行——首序无前驱。

#### schedule_storage_owner_maintenance：stage1 → stage2 的桥

`schedule_storage_owner_maintenance`（`queue.cc:166-215`）是 stage1 收尾的核心：

```cpp
auto& intent = storage_owner_maintenance_intents_[
    static_cast<size_t>((reserved_sequence - 1) %
                        storage_owner_maintenance_intent_capacity_)];
intent.id = id;
intent.generation = generation;
intent.kind = kind;
intent.new_ptr = new_ptr;
intent.old_ptr = old_ptr;
intent.published_at = std::chrono::steady_clock::now();
intent.sequence.store(reserved_sequence, std::memory_order_release);
```

intent 数组是按 `(sequence-1) % capacity` 索引的环形缓冲，capacity = `completion_capacity`（见 §3.6 runtime 启动），保证每个活跃 sequence 有独立 intent 槽。`sequence.store(..., release)` 是发布屏障——stage2 worker 看到 intent 字段全部写完才能消费。

随后按需 enqueue stitch / cleanup，并对"预占但未实际使用"的 work items 立即 complete：

```cpp
if (reserved_work_items > actual_work_items) {
  complete_storage_owner_maintenance_sequence(
      reserved_sequence, reserved_work_items - actual_work_items);
}
```

例如 erase 预占了 1 个 work item（`storage_owner_maintenance_work_items` 返回 1），但 `old_ptr.is_null()` 时 `needs_cleanup = false`，`actual_work_items = 0`，这里就把那 1 个预留立即 complete 掉，否则 watermark 永远推不动。

#### 前台忙探测

`storage_owner_maintenance_foreground_busy`（`queue.cc:221-276`）是 stage2 准入的背压探针，分四档：

1. `storage_owner_insert_active_workers_ != 0` → 前台有活跃 worker。
2. `storage_insert_tasks_->approximate_size() >= max(4, threads * 2)` → 前台队列接近满（`kForegroundQueueYieldMultiplier = 2`）。
3. `peer_reverse_tasks_` 或 `peer_reverse_outgoing_` 队列 near limit（3/4）→ peer 反向边路径忙。
4. `peer_context_` 存在时，poll send CQ 并检查 RDMA 读信用：全局 `peer_async_rdma_outstanding_` 与每 peer `peer_rdma_read_outstanding_` 都按 `pressure_num/pressure_den`（前台活跃时 1/2，否则 3/4）判定。

只要任一档命中就返回 true，stage2 暂停准入。这保证 stage2 的跨分片 RPC 不会与前台 peer RDMA 抢信用。

#### 全局 context 准入

`try_acquire_storage_owner_maintenance_slot`（`queue.cc:278-296`）是全局 stage2 context 数量的上限：

```cpp
const u64 configured_contexts =
    static_cast<u64>(std::max<size_t>(1, storage_owner_maintenance_worker_states_.size())) *
    std::max<u32>(1, config.storage_owner_rpc_depth);
const u32 max_contexts = static_cast<u32>(std::min<u64>(configured_contexts, ...));
```

注释（`queue.cc:279-281`）强调：这个计数器是"全局活 context 数"，不是"物理 worker 数"。每个 worker 拥有 `rpc_depth` 个 context 池，全局上限就是 `workers * rpc_depth`。`storage_owner_maintenance_active_workers_` 用 CAS 自增到该上限，超了就返回 false（不准入）。这个上限与 peer RPC 信用是独立的——peer send 信用是 try-only 的，由 `peer_async_rdma_outstanding_` 单独约束。

### 3.5 `stage2_tracker.hh`：状态机与请求跟踪

#### Stage2Phase 状态枚举

```cpp
// src/memory_node/storage_owner_maintenance/stage2_tracker.hh
enum class Stage2Phase : std::uint8_t {
  local_ready,
  remote_search_pending,
  prune_ready,
  reverse_pending,
  finalized,
};
```

这是 stage2 context 的五态机。一个 context 一次承载一个 batch（最多 `storage_owner_batch_max` 个 task），按顺序流转：

```
local_ready → remote_search_pending → prune_ready → reverse_pending → finalized
```

特殊路径：cleanup task 没有 remote_search 阶段，`begin_remote_search(handle, 0)` 直接跳到 `prune_ready`（`stage2_tracker.hh:125-127`、`worker.cc:546-551`）。

#### Stage2ContextHandle：slot + generation

```cpp
struct Stage2ContextHandle {
  std::uint32_t slot{};
  std::uint32_t generation{};
  bool operator==(const Stage2ContextHandle&) const = default;
};
```

`(slot, generation)` 双字段。slot 是物理槽位下标，generation 是该槽位的"代际"——每次 `try_acquire` 都 `++slot.generation`（`stage2_tracker.hh:80`），且避开 0（`if (slot.generation == 0) ++slot.generation`）。这样 handle 是 ABA 安全的：一个 worker 拿着旧 handle 访问已释放并重新分配的 slot，`resolve` 会因 generation 不匹配返回 nullptr（`stage2_tracker.hh:248-253`）。

#### Stage2StateTracker：状态机存储

```cpp
class Stage2StateTracker {
 public:
  std::optional<Stage2ContextHandle> try_acquire();
  bool release(Stage2ContextHandle handle);
  std::optional<Stage2ContextSnapshot> snapshot(Stage2ContextHandle handle) const;
  Stage2EventResult begin_remote_search(handle, expected_peer_mask);
  Stage2EventResult record_remote_search_response(handle, peer_index);
  Stage2EventResult begin_reverse(handle, expected_peer_mask);
  Stage2EventResult record_reverse_ack(handle, peer_index);
  Stage2EventResult finalize(handle);
  bool awaits(handle, kind, peer_index) const;
  ...
};
```

固定容量、预分配、无内部锁——所有调用者（worker）必须串行化访问自己的 slot。注释（`stage2_tracker.hh:54-56`）明确："owner must serialize calls; the tracker deliberately performs no allocation after construction and no internal locking"。

状态机的关键迁移函数：

- `begin_remote_search`（`stage2_tracker.hh:112-129`）：从 `local_ready` 进入。`expected_peer_mask == 0`（无远程分片或 cleanup）直接跳 `prune_ready`，否则进 `remote_search_pending` 并记录 `expected_search_mask`。
- `record_remote_search_response`（`stage2_tracker.hh:131-157`）：逐位置位 `completed_search_mask`。全部置齐（`completed == expected`）则迁移到 `prune_ready`，返回 `phase_advanced`；否则返回 `accepted`。重复响应返回 `duplicate`。
- `begin_reverse`（`stage2_tracker.hh:159-176`）：从 `prune_ready` 进入。`expected_peer_mask == 0`（无远程反向边）直接返回 `ready_to_finalize`，但**不迁移 phase**——调用者需要继续调用 `finalize`。否则进 `reverse_pending`。
- `record_reverse_ack`（`stage2_tracker.hh:178-202`）：逐位置位 `completed_reverse_mask`。全部置齐返回 `ready_to_finalize`，否则 `accepted`。
- `finalize`（`stage2_tracker.hh:204-217`）：从 `reverse_pending` 进 `finalized`。强制检查 `completed_reverse_mask == expected_reverse_mask`，不等返回 `incomplete`。
- `release`（`stage2_tracker.hh:88-99`）：必须在 `finalized` 状态才能 release，归还 slot。

`awaits`（`stage2_tracker.hh:219-234`）是请求注册前置检查：判断该 (handle, kind, peer) 是否确实在等待响应——phase 必须匹配、expected 位必须为 1、completed 位必须为 0。

#### Stage2RequestTracker：请求注册与重试

```cpp
class Stage2RequestTracker {
 public:
  Stage2RequestRegisterResult try_register(request_id, context, kind, peer_index,
                                            sent_at, deadline, const Stage2StateTracker& states);
  std::optional<Stage2RequestMetadata> find(request_id) const;
  bool retry_due(request_id, now) const;
  std::optional<Stage2RequestMetadata> mark_retry(request_id, sent_at, deadline);
  Stage2EventResult record_response(request_id, Stage2StateTracker& states);
  bool erase(request_id);
  ...
};
```

固定容量开地址哈希表（`buckets_` 容量 = `2 * capacity`，向上取 2 的幂，见 `hash_capacity` `stage2_tracker.hh:431-448`），混合 `mix`（`stage2_tracker.hh:450-457`，Murmur3 finalizer）打散 64-bit request_id。tombstone 复用（`bucket_for_insert` 优先用 first_tombstone，`stage2_tracker.hh:475-491`）。

`try_register`（`stage2_tracker.hh:310-341`）的顺序很重要：

1. `states.is_current(context)` — context 没被释放。
2. `states.awaits(context, kind, peer_index)` — context 确实在等这个响应。
3. `find_bucket(request_id).has_value()` — 防重复注册。
4. `free_records_.empty()` — 容量检查。

`record_response`（`stage2_tracker.hh:371-397`）的关键设计：

```cpp
// Generation validation precedes duplicate detection so a late response
// cannot be mistaken for a response to a context that reused the slot.
if (!states.is_current(metadata.context)) {
  return Stage2EventResult::stale_context;
}
if (metadata.response_seen) return Stage2EventResult::duplicate;
```

generation 校验先于重复检测——迟到的响应不会被误判成"对复用 slot 的新 context 的响应"。`response_seen` 标志位防重复处理。

`mark_retry`（`stage2_tracker.hh:357-369`）：如果已经 `response_seen` 就不再 retry（返回 nullopt），否则 `++attempt_count`、更新 `last_send_time` 与 `deadline`。`retry_due`（`stage2_tracker.hh:350-355`）= `!response_seen && now >= deadline`。

### 3.6 `runtime.cc`：维护 runtime 启停

#### 启动

`start_storage_owner_maintenance_runtime`（`runtime.cc:11-167`）：

1. **控制块校验**（`runtime.cc:12-22`）：读 `StorageControlBlock`，校验 magic/version。关键断言 `initial_next == initial_durable + 1`——重启时不能有未完成的内存维护。如果上次 crash 留下了 `next > durable+1`，必须先离线修复，不能直接重启。

2. **completion ring 容量**（`runtime.cc:23-30`）：
   ```cpp
   const size_t completion_capacity = std::max<size_t>(
       std::max<size_t>(1, config.storage_owner_batch_max),
       config.storage_owner_maintenance_queue_depth / 2);
   ```
   取 `batch_max` 与 `queue_depth/2` 的最大值。注释（`runtime.cc:22-24`）：upsert 一序预占 2 个 work item，所以 capacity 至少要能容纳 queue_depth 的一半，保证每条入队 task 都能 publish 全部 intent。

3. **admission limit**（`runtime.cc:31-39`）：
   ```cpp
   const u64 requested_admission_limit = storage_owner_maintenance_enabled(config)
       ? std::max<u64>(std::max<u32>(1, config.storage_owner_batch_max),
                       static_cast<u64>(std::max<u32>(1, ...workers)) *
                       std::max<u32>(1, config.storage_owner_rpc_depth) * 4)
       : completion_capacity;
   storage_owner_maintenance_admission_limit_ =
       static_cast<size_t>(std::min<u64>(completion_capacity, requested_admission_limit));
   ```
   admission window 是 `min(completion_capacity, max(batch_max, workers*rpc_depth*4))`。stage2 关闭时取 `completion_capacity`——不限制，因为 stage1 自己 complete。

4. **intent 数组**（`runtime.cc:40-42`）：`storage_owner_maintenance_intents_` 是 `StorageOwnerMaintenanceIntent[]`，capacity 同 completion ring。

5. **stage2 启用时的资源**（`runtime.cc:44-147`）：
   - 重置所有计数器。
   - 计算 worker_count = `cpu_plan.maintenance_workers`（CPU 计划）。
   - 创建 `Stage2ReverseOutbox`：容量 = `worker_count * rpc_depth * (num_storage_nodes - 1)`，wire_max_ops = `R * batch_max`（一条聚合消息的最大 op 数）。注释（`runtime.cc:96-105`）强调这是"shared per-peer work-conserving aggregation"——所有 worker 共享一个 outbox。
   - 每 worker 一个 `bounded::Queue<Stage2ReverseCompletion>`，容量 = `rpc_depth * (num_storage_nodes - 1)`。
   - `storage_owner_repair_tasks_`：stale stitch 修复队列，容量 = `max(completion_capacity, 2 * workers * rpc_depth * batch_max)`——保证修复队列不会满。
   - 创建 worker 线程，按 `disable_thread_pinning` 决定是否绑核。

#### 停止

`stop_storage_owner_maintenance_runtime`（`runtime.cc:169-229`）：

1. **drain**（`runtime.cc:176-196`）：前台 insert worker 已 join（注释 `runtime.cc:170-171`），所以不会再有新 intent。等待 stitch/cleanup/repair 队列全部空、`active_workers == 0`。超时（`max(5s, min(60s, rpc_timeout*3))`）后不再等——注释（`runtime.cc:172-175`）解释：schema-15 没有 cross-shard shutdown barrier，某个分片可能已经下线，无限重试同 ID 会死锁进程关闭。
2. `storage_owner_maintenance_shutdown_.store(true)` + `notify_all`：唤醒所有 worker。
3. join 所有 worker。
4. 清空队列、释放 outbox、释放 repair queue。
5. `log_storage_owner_maintenance_observation(..., final=true)` 输出总结。

worker 主循环看到 shutdown 标志后（`worker.cc:1167-1189`）会：①`erase_queued_worker` 删除自己在 outbox 里排队的 dispatch；②`discard_owned_aggregate` 逐个丢弃自己拥有的聚合（并 `cancel_peer_rpc_response`）；③对每个还活着的 context，`cancel_peer_rpc_response` 它的 search request，并 `active_workers_.fetch_sub(1)`；④返回。

#### 观测日志

`log_storage_owner_maintenance_observation`（`runtime.cc:231-468`）输出 70+ 字段的总结，包括：

- 计数：enqueued / stitch_enqueued / cleanup_enqueued / stitch_tasks_done / stitched_live / cleanup_processed / stale / failed / rpc_timeouts。
- 延迟：avg_stitch_delay_ms / p99_stitch_delay_upper_ms / max_stitch_delay_ms / stitch_delay_histogram（18 桶直方图）。
- 批量：stitch_batches / avg_stitch_batch_size / reverse_aggregate_batches / avg_reverse_aggregate_logicals / avg_reverse_aggregate_ops。
- 背压：max_backlog / admission_window / completion_outstanding / pressure_yields。
- peer：peer_stitch_* / peer_reverse_* / external_search_requests / external_candidates。
- 回收（见第 16 课）：reclaim_ack / reclaim_pending / reclaim_reused / dynamic_high_watermark。
- 剩余：stitch_remaining / cleanup_remaining / active_contexts / repair_remaining / remaining（取计数器剩余与队列剩余的最大值，注释 `runtime.cc:338-344` 解释：队列基数在 task 进入 context 后就归零，但 RPC 可能还在等，所以取最大值避免"虚假清空"）。

`maybe_log_storage_owner_maintenance_observation`（`runtime.cc:470-487`）由 worker 主循环周期性调用，CAS 抢 `last_observation_ns` 保证 5 秒内只有一个 worker 输出日志。

### 3.7 `reverse_outbox.hh`：反向边发件箱

`Stage2ReverseOutbox` 是 stage2 反向边的"逻辑发件箱 + 物理聚合器"。注释（`reverse_outbox.hh:72-77`）总结设计：

> Fixed-capacity MPMC logical outbox plus fixed-capacity aggregate storage. All vectors reserve their maximum in the constructor and never grow beyond it. Queueing is work-conserving: form_aggregate consumes the currently available prefix for one peer and RPC type up to the unchanged wire bound; it never waits for a batching timer.

核心思想：

- **逻辑层**：每个 stage2 context 想给某 peer 发 N 条 op，就在 outbox 里入队一条 `Stage2ReverseDispatch`（含 `logical_request_id`、`context`、`worker_id`、`peer_index`、`request_type`、`item_count`、`ops` 指针）。逻辑层是 MPMC 的——任意 worker 都能入队。
- **聚合层**：任意 worker 调用 `form_aggregate(peer_index, ...)` 把该 peer 队列头部连续若干 dispatch 合并成一条 `Stage2ReverseAggregate`（一条 schema-15 wire 消息），ops 被拷贝进固定缓冲。聚合层是 work-conserving 的——有多少合多少，不等 timer。
- **租约（lease）**：`claim_ready_to_post` / `claim_awaiting_response` 用 `leased` 标志 + `owner_worker_id` 实现"一个聚合同时只有一个 worker 在操作"。

#### try_enqueue（`reverse_outbox.hh:112-148`）

```cpp
std::lock_guard<std::mutex> lock(mutex_);
for (const Entry& entry : entries_) {
  if (!entry.in_use || entry.dispatch.logical_request_id != dispatch.logical_request_id) continue;
  return entry.dispatch.same_request(dispatch)
           ? Stage2ReverseEnqueueResult::duplicate
           : Stage2ReverseEnqueueResult::conflict;
}
if (free_entries_.empty()) return Stage2ReverseEnqueueResult::full;
```

入队时全表扫一遍防同 ID 冲突：同 ID 同请求 = `duplicate`（幂等返回），同 ID 不同请求 = `conflict`（逻辑错误）。随后把 dispatch 追加到该 peer 的 FIFO 链表尾部（`PeerQueue` 维护 head/tail，`Entry` 维护 previous/next）。

#### form_aggregate（`reverse_outbox.hh:150-214`）

```cpp
while (peer.head != npos) {
  const std::uint32_t entry_index = peer.head;
  Entry& entry = entries_[entry_index];
  const Stage2ReverseDispatch& dispatch = entry.dispatch;
  if (dispatch.ready_at_ns > now_ns ||
      dispatch.request_type != aggregate.snapshot.request_type ||
      aggregate.ops.size() + dispatch.item_count > wire_max_ops_) {
    break;
  }
  unlink_queued(entry_index);
  ...
  aggregate.ops.insert(aggregate.ops.end(), dispatch.ops, dispatch.ops + dispatch.item_count);
  ++aggregate.snapshot.logical_count;
}
```

聚合条件：①`ready_at_ns <= now`（已就绪，可能因 backoff 延迟）；②`request_type` 相同（`reverse_update_request` 与 `cleanup_deleted_request` 不能混）；③合并后不超过 `wire_max_ops_`（= `R * batch_max`）。满足这三个条件就从 peer FIFO 头部摘链，把 ops 拷贝进聚合缓冲。摘链是为了"先到先聚合"——避免后入队的饿死。

`logical_count == 0` 时释放聚合返回 nullopt——空聚合不发。

#### claim_ready_to_post / claim_awaiting_response

两个租约接口：

- `claim_ready_to_post(worker_id, now, cursor)`（`reverse_outbox.hh:224-240`）：扫描 `aggregates_`，找一个 `ready_to_post && !leased && ready_at_ns <= now` 的聚合，置 `leased=true`、`owner_worker_id=worker_id`，返回 snapshot。`cursor` 是扫描游标，避免每次从头扫。
- `claim_awaiting_response(worker_id, cursor)`（`reverse_outbox.hh:242-256`）：找一个 `awaiting_response && !leased && owner_worker_id == worker_id` 的聚合。

#### finish_post / release_poll / finish_success

状态机：

- `finish_post(..., sent, ready_or_deadline_ns)`（`reverse_outbox.hh:274-292`）：`ready_to_post → awaiting_response`（sent 成功，记 deadline）或留在 `ready_to_post`（sent 失败，记新 ready_at_ns 等重试）。
- `release_poll(..., retry, ready_at_ns)`（`reverse_outbox.hh:294-310`）：`awaiting_response → ready_to_post`（retry=true，重置 ready_at_ns）或留在 `awaiting_response`（retry=false，继续等）。
- `finish_success(...)`（`reverse_outbox.hh:338-353`）：释放所有成员 entry + 释放聚合槽。

#### copy_completions（`reverse_outbox.hh:312-336`）

ACK 到达后，把聚合的所有 logical 成员展开成 `Stage2ReverseCompletion` 列表（每个成员一条 `{logical_request_id, context, worker_id, peer_index}`）。调用者（worker）把这些 completion 推到对应 worker 的 `storage_owner_reverse_completions_[worker_id]` 队列，让目标 worker 在自己的循环里 `record_response`。

注释（`worker.cc:385-390`）解释为什么必须先 `finish_success` 再 fan-out completion：

> The copied completions are value snapshots. Release every logical outbox entry before making an ACK visible to a destination worker: that worker may consume its final ACK, reuse the context slot, and enqueue replacement work immediately. Keeping the old entries until after fan-out would create a transient false-full at exact capacity.

即：completion 是值拷贝，目标 worker 处理完 ACK 可能立即复用 slot 入新 dispatch，如果此时旧 entry 还没释放，outbox 会在满容量时假性 full。

#### erase_queued_worker / discard_owned_aggregate

shutdown 路径用：前者删除某 worker 在 FIFO 里的所有 dispatch；后者逐个丢弃某 worker 拥有的聚合并返回 wire_request_id 以便 cancel。两者都持同一 `mutex_`，保证与 `form_aggregate` 的 ops 拷贝不竞争（注释 `reverse_outbox.hh:356-358`）。

### 3.8 `graph_tasks.cc`：图任务原语

#### try_lock_node / lock_node / unlock_node

```cpp
// graph_tasks.cc:5-24
bool MemoryNode::try_lock_node(RemotePtr rptr) {
  ...
  auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
  std::atomic_ref<u64> ref(*header_ptr);
  for (u32 attempt = 0; attempt < 8; ++attempt) {
    u64 header = ref.load(std::memory_order_acquire);
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) return false;
    const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
    if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
```

节点锁是节点 header 的一个 bit（`HEADER_NODE_LOCK`），用 CAS 自旋 8 次获取。这是 stage2 commit 反向边时串行化访问节点邻接表的细粒度锁——不会阻塞整个分片，只阻塞同一节点的并发修改。

#### storage_owner_task_current

```cpp
// graph_tasks.cc:26-41
bool MemoryNode::storage_owner_task_current(node_t id, u32 generation, RemotePtr target) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto dynamic = shard.entries.find(id);
  if (dynamic != shard.entries.end()) {
    return !dynamic->second.deleted &&
           dynamic->second.generation == generation &&
           dynamic->second.current == target;
  }
  const auto& immutable_base = base_idmap_;
  const auto base = immutable_base.find(id);
  return base != immutable_base.end() && !base->second.deleted &&
         base->second.generation == generation && base->second.current == target;
}
```

判断 stage2 task 是否仍然"当前"：①id 在 dynamic shard 里且未删除且 generation 与 target 都匹配；②否则在 base idmap 里查同样条件。这是 stage2 检测 stale 的核心——如果 stage1 之后该 id 又被 upsert/erase 了，stage2 看到的 task 就 stale 了，必须走 repair 路径（`complete_stale_stitch`）。

#### read_preserved_neighbor_list

`graph_tasks.cc:43-77`：从被删除节点的 hot graph entry 读出"保留邻接表"。校验 `edge_count <= R` 与 `checksum16`，按 `decode_remote_ptr` 解码每个邻居。这是 cleanup task 取"旧节点原本指向谁"的依据——删节点时要把这些反向边也撤掉。

#### remove_local_neighbor / remove_local_neighbors_batched

`graph_tasks.cc:79-141`：本地节点的反向边删除。`remove_local_neighbors_batched` 按 `target_raw → deleted_ptrs[]` 分组，对每个 target 加锁、读邻接、`std::remove_if` 删除所有命中 deleted_ptrs 的邻居、写回。批量版本是 stage2 cleanup 在本地分片上做反向边撤销的入口（见 `worker.cc:936`）。

### 3.9 `worker.cc`：stage2 worker 主循环（1217 行）

`storage_owner_maintenance_worker_loop`（`worker.cc:11-1217`）是 stage2 的核心。整个函数是一个超大的 lambda 集合——所有 helper 都用 `[&, this]` 捕获，共享 `Stage2Context` 池、`Stage2StateTracker states`、`Stage2RequestTracker requests`、`Stage2ReverseOutbox* storage_owner_reverse_outbox_`。

#### 资源预分配（`worker.cc:43-91`）

每个 worker 拥有：

- `contexts[context_capacity]`：每个 context 预分配 `tasks/targets/candidate_storage/candidate_counts/remote_ops_by_peer/search_request_ids/...`。`candidate_storage` 容量 = `batch_max * num_storage_nodes * construction_width`，每个 `NodeSnapshot` 的 `vector_data` 预分配 `VamanaNode::vector_bytes()`。
- `Stage2StateTracker states(context_capacity, num_storage_nodes_)`：状态机。
- `Stage2RequestTracker requests(context_capacity * (num_storage_nodes_ - 1))`：请求跟踪器，容量按"每 context 最多对每个远端 peer 一条请求"算。
- `reverse_wire_ops` / `reverse_completion_scratch` / `reverse_response_payload`：scratch 缓冲。

`reset_context`（`worker.cc:93-111`）清空所有 context 字段——每次准入新 batch 时调用。

#### 记录 finalize 延迟

`record_finalized_live`（`worker.cc:113-127`）：

```cpp
const u64 latency_ns = ...;
storage_owner_maintenance_finalize_latency_ns_.fetch_add(latency_ns, ...);
storage_owner_maintenance_finalize_latency_buckets_[finalize_latency_bucket(latency_ns)].fetch_add(1, ...);
atomic_utils::update_max_relaxed(storage_owner_maintenance_finalize_max_latency_ns_, latency_ns);
storage_owner_maintenance_finalized_live_.fetch_add(1, ...);
```

延迟从 `task.queued_at`（stage1 入队时刻）算到 stage2 finalize 时刻，分桶累加，给观测日志的 p99 与直方图用。`finalized_live` 是"成功 finalize 且未 stale"的计数，是 stitch_completion_ratio 的分子。

#### commit_rebased_stitch_neighbors（`worker.cc:129-179`）

stage2 finalize 的最后一步——把 prune 结果写回目标节点：

```cpp
lock_node(task.target);
const bool target_deleted = (load_local_node_header_acquire(task.target) & HEADER_DELETED) != 0;
const bool current = !target_deleted && storage_owner_task_current(task.id, task.generation, task.target);
if (!current) { unlock_node(task.target); return false; }
// rebase：合并 stage2 prune 结果与 stage1 之后出现的新邻居
const vec<RemotePtr> observed_neighbors = read_neighbor_list(task.target);
vec<RemotePtr> rebased_candidates = merge_stage2_rebase_candidates(...);
vec<NodeSnapshot> rebased_snapshots = read_node_snapshots_batched(rebased_candidates, config);
hashset_t<RemotePtr> skip; skip.insert(task.target);
task.stitch_neighbors = robust_prune_snapshots_cpu(
    index_buffer_.get_full_buffer() + target_vector_address.offset,
    VamanaNode::vector_dtype(), rebased_snapshots, skip, config, config.R);
write_neighbor_list(task.target, task.stitch_neighbors);
unlock_node(task.target);
return true;
```

关键点：

1. **持锁重检**：lock 之后再次检查 deleted 与 current。因为 stage2 远程搜索期间，另一个 mutation 可能已经把这个节点删了或换了 generation。stale 不写（注释 `worker.cc:142-149`）。
2. **rebase**：`merge_stage2_rebase_candidates`（见 §3.3）把 stage2 全局 prune 结果与"stage1 snapshot 之后才出现的新邻居"合并。这一步保证并发 stage2 装的反向边不丢。
3. **二次 prune**：合并后可能超 R，所以在持锁状态下再做一次 `robust_prune_snapshots_cpu`，把结果压到 R 以内。
4. **写回**：`write_neighbor_list` 把最终邻接表写回目标节点。

注释（`worker.cc:142-149`）专门解释为什么不重写 stale/tombstoned 节点："A prepared stitch may already have installed reverse edges, and the union of those targets with the adjacency preserved at deletion can exceed R."——stale 节点交给 cleanup repair 处理，而不是在这里强行重写。

#### handoff_stitch_cleanup（`worker.cc:181-216`）

stale stitch 的修复交接：

```cpp
StorageOwnerMaintenanceTask cleanup;
cleanup.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
cleanup.maintenance_sequence = task.maintenance_sequence;
cleanup.target = task.target;
cleanup.cleanup_repair_only = true;
cleanup.cleanup_neighbors = std::move(cleanup_neighbors);
cleanup.queued_at = std::chrono::steady_clock::now();
lib_assert(storage_owner_repair_tasks_->try_push(std::move(cleanup)), ...);
```

把 stale stitch 已经装上去的反向边目标列表交给 repair 队列，由后续 cleanup task 用 `repair_only=true` 模式撤销。这保证 stale stitch 装的反向边不会泄漏。注释（`worker.cc:1082-1085`）强调 repair 优先于新 stitch 入场："Repair continuations own an already-admitted maintenance sequence and therefore take priority over new stitch work. This removes the stale stitch's attempted backlinks before advancing the watermark and proves the dedicated queue cannot grow across successive admission waves."

#### reverse outbox 驱动

三个 lambda 串成 reverse outbox 的处理流水线：

- `poll_owned_reverse_aggregates`（`worker.cc:358-446`）：claim 自己 own 的 `awaiting_response` 聚合，调 `try_consume_peer_rpc_response` 取响应。success → `copy_completions` + `finish_success` + 把 completion 推到对应 worker 队列；failure/stale → `release_poll(retry=true)` 重新排队，stale 还要 `cancel_peer_rpc_response` + `rearm_peer_rpc_response`（tombstone 复用场景的最佳努力重建，注释 `worker.cc:414-418`）；timeout → 计数 + `release_poll`。
- `form_reverse_aggregates`（`worker.cc:448-471`）：对每个 peer 调 `can_form_aggregate` + `form_aggregate`，把 FIFO 头部 dispatch 合并成 wire 聚合。统计 `reverse_aggregate_batches / logical_requests / ops`。
- `post_owned_reverse_aggregates`（`worker.cc:473-509`）：claim `ready_to_post` 聚合，`copy_ops` 拷贝 ops，`post_peer_op_batch_async` 发送，`finish_post` 记结果。

`drive_reverse_outbox`（`worker.cc:511-516`）= poll || form || post，每轮主循环调用两次（开头一次、`try_admit_context` 之后一次，注释 `worker.cc:1191-1194`）。

#### drain_reverse_completions（`worker.cc:518-543`）

从自己的 `storage_owner_reverse_completions_[worker_id]` 队列 pop completion，校验 `metadata.context == completion.context` 等一致性，调 `requests.record_response` 推进 stage2 状态机（`record_reverse_ack`），最后 `requests.erase`。

#### prepare_local（`worker.cc:545-622`）

stage2 入场第一步。对每个 task：

1. cleanup kind 直接 `begin_remote_search(handle, 0)` 跳 `prune_ready`，return。
2. 非 local_shard 的 target：complete 该 task 序，跳过（stage2 只处理本分片 target）。
3. stale（`!storage_owner_task_current`）→ `complete_stale_stitch`（交接 repair）。
4. 读 target snapshot，若 deleted → `complete_stale_stitch`。
5. 否则把 target snapshot 入 `context.targets`，读 stage1 candidates 的本地 snapshot 拷进 `candidate_storage`。

最后计算 `expected_mask`（除本分片外所有 peer），`begin_remote_search(handle, expected_mask)`，`register_search_requests`。

`register_search_requests`（`worker.cc:305-327`）对每个 expected peer：分配 request_id、`requests.try_register`、`post_search`（实际 `post_stitch_search_request_async`）。注意两段循环：先全部注册再全部 post——"Attempt every shard before returning to the rest of the pipeline"，避免单 shard 阻塞影响其他 shard。

#### poll_search_responses / parse_search_response / retry_search_if_due

`poll_search_responses`（`worker.cc:743-794`）对每个 pending shard 调 `try_consume_peer_rpc_response`：

- success + parse ok → `record_response`（推进 mask，可能 phase_advanced）+ `erase`。
- failure/stale/success-but-parse-fail → `mark_retry(now, now+backoff)`，stale 还要 cancel+rearm。
- 否则 → `retry_search_if_due` 检查 deadline 是否到，到则重发。

`parse_search_response`（`worker.cc:624-714`）做严格校验：

- `header.reserved == construction_width`、`header.item_count == tasks.size()`、`payload.size() >= expected_bytes`。
- 每个 candidate：`pointer.memory_node() < num_storage_nodes`、`== header.source_shard`、`ptr_in_bounds`、`hot_graph_entry_available`。dynamic offset 的 candidate 还要求 `generation != 0`。
- 校验通过后把 candidate 拷进 `candidate_storage`，统计 `external_candidates`。

任何校验失败返回 false，外层当作 parse 失败重试。

#### prepare_stitch_reverse（`worker.cc:796-870`）

prune_ready 阶段对 stitch kind 的处理：

1. 对每个 task：再次检查 deleted/current（stale → `complete_stale_stitch`）。
2. 若 `task.stitch_prepared`（已经 prune 过，retry 场景）直接用旧结果；否则 `robust_prune_snapshots_cpu(target_vector, candidate_storage[item], skip={target}, config, config.R)`。
3. 标 `stitch_prepared=true`，存 `stitch_neighbors`。
4. 把每个 neighbor 按 memory_node 分流：本地 → `local_updates`；远端 → `remote_ops_by_peer[shard]`。
5. `apply_local_reverse_updates_batched(local_updates)` 装本地反向边。
6. 计算 `expected_mask`（有远端 op 的 shard 集合），`begin_reverse(handle, expected_mask)`，`register_reverse_requests`。

`begin_reverse` 返回 `ready_to_finalize`（无远端）或 `phase_advanced`（有远端）——前者直接 finalize，后者等 ACK。

#### prepare_cleanup_reverse（`worker.cc:872-956`）

cleanup kind 的 prune_ready 处理：

1. `task.target.is_null()` → complete 序，跳过。
2. 读 target snapshot。
3. 非 deleted 且非 `repair_only` → `complete_stale_cleanup`（节点没被删，cleanup 多余）。
4. `!repair_only` 时读 `preserved_neighbor_list`（删节点保留的旧邻接）。
5. `select_cleanup_neighbors(repair_only, preserved, supplemental)` 选出要撤销反向边的目标集合（见 §3.3）。
6. 断言 `old_neighbors.size() <= R`（schema-15 每 item 上限）。
7. 分流本地/远端，`remove_local_neighbors_batched` 撤本地反向边。
8. `begin_reverse(handle, expected_mask)` + `register_reverse_requests`。

注释（`worker.cc:895-904`）再次强调 repair vs ordinary cleanup 的分离：repair 只撤销 supplemental（stale stitch 装的反向边），ordinary 只撤销 preserved（被删节点原本的邻接），二者不能混。

#### finalize_context（`worker.cc:958-1005`）

```cpp
const Stage2EventResult transition = states.finalize(context.handle);
lib_assert(transition == Stage2EventResult::phase_advanced, "stage2 finalized before all reverse ACKs");
```

强制要求 `completed_reverse_mask == expected_reverse_mask` 才能 finalize。finalize 后：

- stitch kind：对每个 task 调 `commit_rebased_stitch_neighbors`。`!current` → `handoff_stitch_cleanup`（stale 修复）；否则 `record_finalized_live` + `complete_storage_owner_maintenance_sequence`。
- cleanup kind：对每个 task，非 `repair_only` 时 `retire_local_dynamic_node(task.target, task.maintenance_sequence)`（把物理节点归还动态分配池，见第 16 课回收），`complete_storage_owner_maintenance_sequence`。

最后 `reset_context` + `states.release(handle)` + `active_workers_.fetch_sub(1)` + `notify_all`。

#### drive_context（`worker.cc:1007-1048`）

状态机驱动器：

```cpp
switch (snapshot->phase) {
  case Stage2Phase::local_ready:
    (void)prepare_local(context); progressed = true; continue;
  case Stage2Phase::remote_search_pending:
    progressed = poll_search_responses(context) || progressed;
    if (still pending) return progressed;
    continue;
  case Stage2Phase::prune_ready:
    prepared = (kind == stitch_insert) ? prepare_stitch_reverse : prepare_cleanup_reverse;
    if (!prepared) return progressed;
    progressed = true; continue;
  case Stage2Phase::reverse_pending:
    if (still pending && completed != expected) return progressed;
    finalize_context(context); return true;
  case Stage2Phase::finalized:
    lib_failure("stage2 context remained active after finalization");
}
```

每个 phase 自己推进自己，直到阻塞（pending）或完成（finalized）。`remote_search_pending` 与 `reverse_pending` 在未完成时 return，让主循环去驱动其他 context 或 outbox。

#### try_admit_context（`worker.cc:1050-1164`）

准入新 batch 的入口：

1. `decide_stage2_admission`：contexts full/shutdown → null；foreground pressure → null 并计数。
2. `try_acquire_storage_owner_maintenance_slot`：全局 context 上限。失败 → null 并计数。
3. **repair 优先**：先从 `storage_owner_repair_tasks_` try_pop。`storage_owner_cleanup_ready(repair.maintenance_sequence)` 检查前一序是否已 durable，未就绪则 push 回去 break；就绪则装入 context，继续 pop 直到 batch 满或遇到未就绪。
4. **普通 task**：取 `storage_owner_maintenance_mutex_` 锁，`find_if` 找第一个 `cleanup_ready` 的 cleanup task。比较 stitch front 与 ready cleanup 的 `queued_at`，选早的（FIFO 跨 kind）。stitch 从 front pop 到 batch 满；cleanup 用 `erase` 从 deque 中删除（保持其他未就绪 cleanup 顺序）。

注释（`worker.cc:1082-1085`、`worker.cc:1113-1130`）强调两个不变式：①repair 优先保证 stale stitch 的反向边在 watermark 推进前被撤销；②stitch vs cleanup 按 queued_at 跨 kind FIFO，避免任一 kind 饿死。

#### 主循环（`worker.cc:1166-1216`）

```cpp
for (;;) {
  if (shutdown) { ... 清理 ...; return; }
  bool progressed = drive_reverse_outbox();
  progressed = drain_reverse_completions() || progressed;
  for (Stage2Context& context : contexts) {
    if (context.active) progressed = drive_context(context) || progressed;
  }
  while (Stage2Context* context = try_admit_context()) {
    progressed = true;
    (void)drive_context(*context);
  }
  progressed = drive_reverse_outbox() || progressed;
  progressed = drain_reverse_completions() || progressed;
  maybe_log_storage_owner_maintenance_observation();
  if (!progressed) {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait_for(lock, std::chrono::milliseconds(1));
  }
}
```

每轮：①驱动 outbox（poll+form+post）；②drain completion；③驱动所有 active context；④尝试准入新 context 直到准入失败；⑤再驱动一次 outbox + completion（注释 `worker.cc:1191-1194`：prune 可能产生新 dispatch，第二遍 catch 住，不等 timer）；⑥观测日志；⑦无进展则 `wait_for(1ms)`——1ms 超时保证即使错过 notify 也能及时醒。

---

## 四、关键数据结构与流程图

### 4.1 数据结构关系

```
StorageControlBlock (索引缓冲区头部)
  ├─ next_maintenance_sequence      (atomic u64, reserve_batch 推进)
  ├─ durable_maintenance_sequence   (atomic u64, finalize 推进, 计算侧观察)
  ├─ reclaim_pending_nodes          (见第 16 课)
  ├─ reclaim_reused_nodes
  └─ dynamic_high_watermark

SlidingCompletionRing (storage_owner_maintenance_completion_ring_)
  ├─ reserve_batch(work_items[], admission_limit) → first_sequence
  ├─ complete(sequence, work_items)
  ├─ next_sequence() / finalized()
  └─ outstanding()

StorageOwnerMaintenanceIntent[capacity]  (环形, 按 (seq-1)%capacity 索引)
  ├─ id / generation / kind / new_ptr / old_ptr
  ├─ published_at
  └─ sequence (atomic, release 写)

storage_owner_stitch_tasks_ / storage_owner_cleanup_tasks_  (deque, 有界)
storage_owner_repair_tasks_  (bounded::Queue, stale stitch 修复)

Stage2StateTracker (每 worker 一个, capacity = rpc_depth)
  └─ Slot[capacity] { generation, in_use, Stage2ContextSnapshot }

Stage2RequestTracker (每 worker 一个, capacity = rpc_depth * (peers-1))
  └─ 开地址哈希: request_id → Stage2RequestMetadata

Stage2ReverseOutbox (全局共享)
  ├─ entries_[logical_capacity]  (逻辑 dispatch, 按 peer FIFO)
  ├─ aggregates_[aggregate_capacity]  (物理 wire 聚合, 含 ops 拷贝)
  └─ peers_[peer_count]  (每 peer FIFO head/tail)

storage_owner_reverse_completions_[worker_count]  (每 worker 一个 bounded::Queue)
```

### 4.2 完整时序图

```
计算侧                存储节点 (storage_owner_runtime)           stage2 worker
  │                         │                                       │
  │ mutation batch (RDMA)   │                                       │
  │ ───────────────────────>│                                       │
  │                         │ recv CQ → try_push(insert_task)       │
  │                         │ worker: process_storage_owner_insert_task
  │                         │   begin_storage_owner_maintenance_batch
  │                         │     reserve_batch → first_seq           │
  │                         │   for each item:                        │
  │                         │     prepare_mutation / allocate_node    │
  │                         │     write_new_node / publish_mutation   │
  │                         │     robust_prune_cpu (stage1 临时邻居)  │
  │                         │     schedule_storage_owner_maintenance: │
  │                         │       write intent[(seq-1)%cap]         │
  │                         │       enqueue_insert_stitch ────────────>│ stitch_tasks_.push
  │                         │       (或 enqueue_deleted_node_cleanup)─>│ cleanup_tasks_.push
  │                         │     (local_stitch off: 本地+远端临时反向边)
  │                         │   build response (含 maintenance_sequence)
  │ <───────────────────────│ post_storage_owner_response             │
  │                         │                                       │
  │                         │                                       │ try_admit_context
  │                         │                                       │   decide_stage2_admission
  │                         │                                       │   try_acquire_slot
  │                         │                                       │   pop stitch/cleanup/repair → context
  │                         │                                       │ drive_context:
  │                         │                                       │   local_ready:
  │                         │                                       │     prepare_local (读 target snapshot
  │                         │                                       │       + 本地 stage1 candidates)
  │                         │                                       │     begin_remote_search(expected_mask)
  │                         │                                       │     register_search_requests
  │                         │                                       │   remote_search_pending:
  │                         │                                       │     post stitch_search_request ───> peer
  │                         │                                       │     poll_search_responses <────────── peer
  │                         │                                       │       parse (校验 source_shard / bounds)
  │                         │                                       │     record_remote_search_response
  │                         │                                       │   prune_ready:
  │                         │                                       │     prepare_stitch_reverse:
  │                         │                                       │       robust_prune_snapshots_cpu (合并候选)
  │                         │                                       │       apply_local_reverse_updates_batched
  │                         │                                       │     begin_reverse(expected_mask)
  │                         │                                       │     register_reverse_requests
  │                         │                                       │       enqueue_reverse_dispatch ──> outbox
  │                         │                                       │   (outbox: form_aggregate → claim_ready_to_post
  │                         │                                       │     → copy_ops → post_peer_op_batch_async ──> peer)
  │                         │                                       │   reverse_pending:
  │                         │                                       │     poll_owned_reverse_aggregates <────── peer ACK
  │                         │                                       │       finish_success → copy_completions
  │                         │                                       │       → push completion to owner worker
  │                         │                                       │     drain_reverse_completions:
  │                         │                                       │       record_reverse_ack (置位 mask)
  │                         │                                       │   (all ACKs received)
  │                         │                                       │   finalize_context:
  │                         │                                       │     commit_rebased_stitch_neighbors:
  │                         │                                       │       lock_node; rebase + 二次 prune; write_neighbor_list
  │                         │                                       │     complete_storage_owner_maintenance_sequence
  │                         │                                       │       completion_ring.complete(seq, work)
  │                         │                                       │       publish_storage_owner_maintenance_watermarks:
  │                         │                                       │         CAS durable_maintenance_sequence 上升
  │                         │                                       │       cv.notify_all
  │                         │                                       │
  │ RDMA read durable_seq   │                                       │
  │ <───────────────────────│ (计算侧观察 durable 推进)              │
  │                         │                                       │
  │ 回收旧节点 (RCU)        │                                       │
  │ (见第 16 课)            │                                       │
```

### 4.3 stage2 状态机

```
                ┌───────────────────────────────────────────────────────┐
                │                                                       │
                ▼                                                       │
    ┌──────────────────┐                                                │
    │   local_ready    │  prepare_local:                                │
    │                  │    - cleanup: begin_remote_search(mask=0) ─────────────┐
    │                  │    - stitch:  读 target+本地 candidates,                │ │
    │                  │               begin_remote_search(expected_mask)        │ │
    └────────┬─────────┘                                                 │ │
             │ expected_mask == 0 (cleanup 或无远端)                     │ │
             │                                                           │ │
             │ expected_mask != 0                                        │ │
             ▼                                                           │ │
    ┌──────────────────┐                                                │ │
    │ remote_search_   │  register_search_requests → post_search         │ │
    │   pending        │  poll_search_responses:                         │ │
    │                  │    parse + record_remote_search_response        │ │
    │                  │    (逐位置位 completed_search_mask)             │ │
    └────────┬─────────┘                                                │ │
             │ completed_search_mask == expected_search_mask             │ │
             ▼                                                           │ │
    ┌──────────────────┐  ←──────────────────────────────────────────────┘ │
    │   prune_ready    │  prepare_stitch_reverse / prepare_cleanup_reverse: │
    │                  │    robust_prune_snapshots_cpu (stitch)              │
    │                  │    select_cleanup_neighbors (cleanup)               │
    │                  │    apply_local_reverse / remove_local_neighbors     │
    │                  │    begin_reverse(expected_reverse_mask)             │
    └────────┬─────────┘                                                  │
             │ expected_reverse_mask == 0 → ready_to_finalize              │
             │ expected_reverse_mask != 0 → phase_advanced                 │
             ▼                                                            │
    ┌──────────────────┐                                                  │
    │ reverse_pending  │  register_reverse_requests → outbox.try_enqueue   │
    │                  │  drive_reverse_outbox:                            │
    │                  │    form_aggregate → claim_ready_to_post →         │
    │                  │    copy_ops → post_peer_op_batch_async            │
    │                  │  poll_owned_reverse_aggregates:                   │
    │                  │    try_consume_peer_rpc_response                  │
    │                  │    finish_success → copy_completions              │
    │                  │  drain_reverse_completions:                       │
    │                  │    record_reverse_ack (置位 completed_reverse_mask)│
    └────────┬─────────┘                                                  │
             │ completed_reverse_mask == expected_reverse_mask             │
             ▼                                                            │
    ┌──────────────────┐                                                  │
    │   finalized      │  finalize_context:                               │
    │                  │    commit_rebased_stitch_neighbors (stitch)      │
    │                  │    retire_local_dynamic_node (cleanup)           │
    │                  │    complete_storage_owner_maintenance_sequence   │
    │                  │    publish_watermarks (CAS durable ↑)            │
    │                  │    states.release(handle)                        │
    └──────────────────┘                                                  │
                                                                          │
  任意阶段检测到 stale (storage_owner_task_current == false):              │
    complete_stale_stitch → handoff_stitch_cleanup → repair_tasks_ ────────┘
```

---

## 五、与其他模块的关系

### 5.1 与第 3 课（并发原语与协程）

- `bounded::Queue` / `bounded::SlidingCompletionRing` 是有界并发原语，stage1/stage2 之间通过它们传递 task 与序号。
- `execute_storage_owner_insert_job_async` 返回的 `StorageOwnerInsertCoroutine` 是 C++20 协程，`thread.is_ready(coroutine_id)` 基于 peer RDMA post balance 判断协程是否可恢复。协程切换不分配堆内存，scratch 全部预分配。
- `std::atomic_ref<u64>` 用于在 `StorageControlBlock` 的非原子字段上做原子访问（`queue.cc:100-117`）。

### 5.2 与第 8 课（schema-15 / storage_owner_protocol）

- `InsertBatchRequestHeader` / `MutationBatchRequestHeader` 共用 40 字节布局，仅 magic 不同（`storage_owner_protocol.hh:44-75`）。`anchor_hint_count` 字段保留但必须为 0（schema-15 兼容）。
- `PeerRpcHeader`（`storage_owner_protocol.hh:145-154`）是 stage2 peer RPC（stitch_search / reverse_update / cleanup_deleted）的统一头。`PeerRpcType`（`storage_owner_protocol.hh:35-42`）区分 6 种消息。
- `ReverseUpdateOp{target_raw, candidate_raw}`（`storage_owner_protocol.hh:156-159`）是反向边 wire 操作。`wire_max_ops_ = R * batch_max` 是一条聚合消息的 op 上限，`select_cleanup_neighbors` 与 `prepare_stitch_reverse` 都断言不超过 R 每 item——这是 schema-15 的硬约束。
- `stitch_search_response_bytes / counts / candidates / candidate_vectors`（`storage_owner_protocol.hh:328+`）是 stitch 搜索响应的 wire 解码 helper，被 `parse_search_response` 调用。
- response 布局 `InsertBatchResponseHeader + statuses + MutationResult + InsertBreakdownCounters + invalidation_count + invalidated_raws`，`invalidated_raws` 是计算侧路由失效通知。

### 5.3 与第 15 课（增量发布）和第 28 课（计算侧 sender）

- 计算侧 sender 把 mutation 批通过 RDMA send 发到存储节点的 recv CQ（`service_storage_runtime`）。`MutationResult.maintenance_sequence` 回程给计算侧，计算侧用它跟踪"这个 mutation 何时 durable"。
- `invalidated_raws` 触发计算侧路由失效（见第 15 课），保证后续查询不会用到旧图结构。
- stage2 finalize 推进 `durable_maintenance_sequence` 后，计算侧观察到的 durable watermark 上升，可以释放对旧 `RemotePtr` 的引用。

### 5.4 与第 16 课（存储回收 RCU）

- `durable_maintenance_sequence` 是 RCU 回收的 grace period 信号。计算侧 `minimum_compute_reclaim_ack()` 跟踪所有计算节点已观察到的 durable 序号（见 `runtime.cc:353` 的 `reclaim_ack` 统计）。
- cleanup task 的 `retire_local_dynamic_node(task.target, task.maintenance_sequence)`（`worker.cc:990`）把物理节点归还动态分配池，但实际复用要等 durable 推进到该序之后——这是 RCU 的延迟回收语义。
- `StorageControlBlock.reclaim_pending_nodes / reclaim_reused_nodes / dynamic_high_watermark` 是回收统计，在观测日志里输出（`runtime.cc:347-352`）。

### 5.5 与第 23/24/25 课（storage owner 主体 / peer RPC / 索引访问）

- 第 23 课讲述存储节点主体（`MemoryNode`）的初始化与 peer 上下文；stage2 worker 的 `peer_context_`、`peer_async_rdma_outstanding_`、`peer_rdma_read_outstanding_` 都来自该层。
- 第 24 课讲述 peer RPC 的发送/接收路径（`post_stitch_search_request_async`、`post_peer_op_batch_async`、`try_consume_peer_rpc_response`、`cancel_peer_rpc_response`、`rearm_peer_rpc_response`）。stage2 worker 通过这些原语与外部分片通信。
- 第 25 课讲述索引访问与图修改原语（`lock_node`、`read_neighbor_list`、`write_neighbor_list`、`read_node_snapshot`、`allocate_local_node`、`mark_node_deleted`、`robust_prune_snapshots_cpu`、`apply_partition_local_reverse_update`、`send_reverse_update_batch` 等）。stage2 的 `commit_rebased_stitch_neighbors`、`prepare_stitch_reverse`、`prepare_cleanup_reverse` 全部建立在这些原语之上。

### 5.6 stage1/stage2 协调要点

- **stage1 ACK 后入维护队列**：`schedule_storage_owner_maintenance` 写 intent + enqueue stitch/cleanup，这一步发生在 stage1 返回响应之前。响应里的 `maintenance_sequence` 告诉计算侧"这个 mutation 的 stage2 序号"。
- **stage2 finalize**：worker 从队列取 task、走状态机、finalize 后 `complete_storage_owner_maintenance_sequence`。
- **durable sequence 推进**：`publish_storage_owner_maintenance_watermarks` 把 completion ring 的 `finalized()` CAS 进 `StorageControlBlock.durable_maintenance_sequence`。计算侧观察 durable 上升 → 回收旧节点。
- **至少保留一个 maintenance worker**：`storage_owner_maintenance_enabled` 要求 `workers > 0`（`runtime.cc:6-9`）。否则 stage1 自己 complete，但 durable watermark 永远不会因 stage2 而推进——其实 stage2 关闭时 stage1 直接 commit 并 complete，watermark 在 stage1 路径里推进，所以不会有饿死问题。但是 stage2 启用时，必须有 worker 才能消费队列、推进 durable——若 worker 数为 0 而 mode=finalize，durable 会永远卡在初始值，cleanup task 永远 `cleanup_ready=false`，整个维护队列死锁。所以 `workers > 0` 是 stage2 启用的硬条件。
- **背压**：stage2 慢 → 队列满 → stage1 入队阻塞在 `cv_.wait` → 计算侧 mutation 响应延迟上升 → 自然反压。stage2 的 `foreground_busy` 探测反过来让 stage2 在前台忙时主动让出，避免抢 RDMA 信用。

---

## 六、小结

本课拆解了存储节点对 mutation 批的"两阶段处理"：

1. **runtime 子系统（stage1）**：`service_storage_runtime` 接 RDMA recv CQ，`storage_owner_insert_worker_loop` 消费 `storage_insert_tasks_`，`execute_storage_owner_batch_items[_async]` 执行 mutation：预占维护序、prepare/allocate/write/publish、stage1 临时 prune、`schedule_storage_owner_maintenance` 写 intent 并入维护队列。响应回程携带 `maintenance_sequence` 与 `invalidated_raws`。
2. **maintenance 子系统（stage2 finalize）**：`storage_owner_maintenance_worker_loop` 维护五态状态机 `local_ready → remote_search_pending → prune_ready → reverse_pending → finalized`。通过 `Stage2ReverseOutbox` 把跨分片反向边 coalesce 成 wire 聚合消息，发出去等 ACK；ACK 齐了 `finalize_context` 做 rebase + 二次 prune + 写回邻接表，`complete_storage_owner_maintenance_sequence` 推进 `durable_maintenance_sequence`。stale task 走 `handoff_stitch_cleanup` → repair 队列，保证装上去的反向边不泄漏。
3. **协调**：stage1 入队有界（`queue_depth`），背压反传到计算侧；stage2 准入有界（`admission_limit` + `foreground_busy` + 全局 context 上限），不抢前台 RDMA 信用。durable watermark 是 RCU 回收的 grace period 信号，必须至少一个 maintenance worker 才能推进。
4. **wire 协议**：mutation 批复用 `InsertBatchRequestHeader` 布局，magic 区分 insert/mutation；peer RPC 用 `PeerRpcHeader` 统一头；`ReverseUpdateOp` 是反向边 wire 操作；`stitch_search_response` 是候选回传格式。所有这些与第 8 课 `storage_owner_protocol` 严格对齐，schema-15 的容量约束（R/batch_max）在多处 `lib_assert` 里强制。

至此，存储侧的"前台 mutation → 后台 finalize → durable 推进 → 计算侧回收"完整闭环已经清晰。下一课（第 27 课）将转向计算服务主体，看计算侧如何发起 mutation、如何观察 durable 推进、如何调度查询。
