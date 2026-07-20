# 第 14 课：查询执行、路由与完成处理

## 本课目标

dvstor 的 GPU 中心化检索系统把"接收 query → 在 GPU 上跑持久化 kernel → 把结果交回 CPU"这条路径切成三段，由三个独立线程驱动：

1. **admission 线程**（CPU）：从空闲 slot 池领 slot，把 query 拷进预分配的 GPU 输入区，立即把 `QueryDescriptor` 推到 host→device ring，不凑批。
2. **maintenance 线程的 storage-canonical 路由刷新分支**（CPU）：定期从每个存储 shard 的 4 KiB control page offset 1024 拉取 `StorageRoutePublication`，用 seqlock + checksum 校验，按 epoch 更新 GPU 上的动态路由槽，并发布 route-only delta command。同时 `routing.cc` 负责路由失效后的 anchor graph 刷新。
3. **completion 线程**（CPU）：从 device→host completion ring 读 `CompletionDescriptor`，组装最终 ID 距离列表，归还 slot，累加各阶段 cycle/remote_pages 等遥测，status 非零时调用 `mark_unhealthy` 进入 fail-stop。

本课逐行讲解这三个文件，并画出"admission → device ring → kernel → completion ring → 归还 slot"的端到端时序图。

### 涉及文件

- `src/gpu_search/persistent_engine/query_execution.cc` — `search()` 入口与 `admission_loop()`。
- `src/gpu_search/persistent_engine/routing.cc` — delta mutation 解码、anchor 最近邻、graph cache key 计算、`refresh_anchor_graph_records()`。
- `src/gpu_search/persistent_engine/completion.cc` — `report_direct_path_failure()` 与 `completion_loop()`。
- 配套读取（本课交叉引用，不展开）：`src/gpu_search/persistent_engine/impl.hh`（`Impl` 结构与所有成员）、`src/gpu_search/persistent_engine/health.cc`（`reject_submission`/`mark_unhealthy`/`reject_all_pending`）、`src/gpu_search/persistent_engine/lifecycle.cc`（线程生命周期）、`src/gpu_search/persistent_engine/storage_reclaim.cc`（路由快照拉取，见下文）、`src/gpu_search/mapped_ring.hh`（`MappedRing::try_push/try_pop`）、`src/gpu_search/types.hh`（`QueryDescriptor`/`CompletionDescriptor`）、`src/gpu_search/index_format.hh`（`StorageRoutePublication`/`kStorageRoutePublicationOffset`）。

> **关于"routing.cc"命名的说明**：任务原文提到的"storage-canonical 路由快照拉取（从 4 KiB control page offset 1024）、checksum/seqlock 校验、按 epoch 更新 GPU 动态路由槽、route-only command 发布"在源码中实际位于 `storage_reclaim.cc` 的 `read_storage_route_publications()` 与 `synchronize_storage_routes()`（这两个函数由 maintenance 线程调用）。本课按照"基于真实代码"的原则：`routing.cc` 部分严格讲解 `routing.cc` 文件实际内容（mutation 解码 + anchor 路由 + anchor graph 刷新），路由快照拉取部分则贴 `storage_reclaim.cc` 的真实代码并明确标注文件路径，避免臆造。

---

## 一、`query_execution.cc`：CPU admission 线程

这个文件只有两个函数：对外入口 `search()` 与后台 `admission_loop()`。两者通过 `admission_queue` 这个 `std::deque<PendingSubmission>` 解耦——`search()` 是调用方线程（计算服务的 RPC worker），`admission_loop()` 是单独的 admission 线程。

### 1.1 数据结构回顾（来自 `impl.hh`）

在进入代码前，先把 `Impl` 里和本课相关的成员列出来（`src/gpu_search/persistent_engine/impl.hh:65-414`）：

```cpp
struct PendingQuery {                    // impl.hh:66
  u32 slot{};
  std::chrono::steady_clock::time_point submitted_at{};
  std::promise<service::QueryResult> promise;
};

struct PendingSubmission {               // impl.hh:72
  QueryDescriptor descriptor{};
  std::chrono::steady_clock::time_point enqueued_at{};
};
```

- `PendingQuery` 是"已经发给 admission 但还没收到 completion"的 per-request 状态，含一个 `std::promise`，`search()` 返回的 `future` 就由它支撑。
- `PendingSubmission` 是"已进入 admission_queue 但还没 push 到 device ring"的条目，记录 `enqueued_at` 用于统计 admission 等待时延。

线程协调用的成员（`impl.hh:387-412`）：

```cpp
std::atomic<bool> accepting{true};        // 是否还接受新 query（停机时翻 false）
std::atomic<bool> healthy{true};          // fail-stop 总开关
std::atomic<bool> shutdown{false};        // admission/completion 线程停机标志
std::atomic<u64>  active_gpu_queries{0};  // 已 push 到 ring 但未 completion 的数量
std::atomic<u64>  next_query_ticket{1};   // 全局单调递增的 query ticket（RCU barrier 用）
std::atomic<u64>  next_request_id{1};     // request_id 分配器
std::atomic<u64>  pending_count{0};       // pending_queries 表大小（停机 drain 用）
std::mutex admission_mutex;
std::condition_variable admission_cv;
std::deque<PendingSubmission> admission_queue;
std::string health_error;
std::mutex slot_mutex;
std::condition_variable slot_cv;
std::vector<u32> free_slots;              // 空闲 slot 池（栈式使用）
std::unique_ptr<std::atomic<u64>[]> active_query_tickets;   // per-slot ticket
std::unique_ptr<std::atomic<u64>[]> active_query_snapshots; // per-slot snapshot_epoch+1
std::mutex query_snapshot_mutex;
std::mutex pending_mutex;
std::unordered_map<u64, std::shared_ptr<PendingQuery>> pending_queries;
MappedRing<QueryDescriptor> submissions;        // host→device ring
MappedRing<CompletionDescriptor> completions;   // device→host ring
```

`submissions` 与 `completions` 是 `MappedRing`（`src/gpu_search/mapped_ring.hh`）——一种用 `cudaHostAllocMapped` 分配的环形队列，CPU 端写 host 指针、GPU 端通过 device 指针读，绕过显式 copy。`try_push`/`try_pop` 基于 sequence 数组实现无锁 MPSC/MSPC 语义（`mapped_ring.hh:74-96`）。这两个 ring 的具体实现见第 17 课。

### 1.2 `search()`：对外入口

`src/gpu_search/persistent_engine/query_execution.cc:7-78`：

```cpp
service::QueryResult PersistentSearchEngine::Impl::search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (!accepting.load(std::memory_order_acquire)) {
    throw std::runtime_error("persistent GPU search engine is stopping");
  }
  if (!healthy.load(std::memory_order_acquire)) {
    throw std::runtime_error(unhealthy_message());
  }
  if (query_data == nullptr || static_cast<u32>(query_dtype) > 2 ||
      k == 0 || k > result_capacity) {
    throw std::invalid_argument("invalid persistent GPU query");
  }
```

第 8-10 行检查 `accepting`：析构函数（`lifecycle.cc:328`）会把它翻成 false 并通知所有 CV，此时新 query 直接抛"stopping"。第 11-13 行检查 `healthy`，fail-stop 后调用 `unhealthy_message()`（`health.cc:7`，锁 `admission_mutex` 读 `health_error`）抛带原因的异常。第 14-17 行做参数校验：`query_dtype` 用 `static_cast<u32>` 比较，合法值是 0/1/2（fp32/fp16/bf16，见第 9 课）；`k` 不能超过 `result_capacity`（GPU 结果缓冲区每个 slot 的容量）。

接下来是 slot 分配：

```cpp
  u32 slot = 0;
  {
    std::unique_lock<std::mutex> lock(slot_mutex);
    slot_cv.wait(lock, [&] {
      return !free_slots.empty() || !accepting.load() || !healthy.load();
    });
    if (!accepting.load()) throw std::runtime_error("persistent GPU search engine stopped");
    if (!healthy.load()) {
      lock.unlock();
      throw std::runtime_error(unhealthy_message());
    }
    slot = free_slots.back();
    free_slots.pop_back();
  }
```

`free_slots` 是一个 vector 当栈用（`back()`/`push_back()`）。`slot_cv.wait` 的谓词同时关心三个条件：有空闲 slot、`accepting` 翻 false、`healthy` 翻 false。被唤醒后再次检查后两个，避免在停机/不健康时还领 slot。`lock.unlock()` 在抛异常前显式释放，避免 `~unique_lock` 在 unwind 时二次 unlock 的潜在警告（其实 unique_lock 析构会处理，这里写法偏保守）。

slot 拿到后，把 query 拷到预分配的 host 输入区：

```cpp
  const size_t query_bytes = vector_dtype_bytes(query_dtype, config.dim);
  byte_t* query_slot = query_input_host + static_cast<size_t>(slot) * query_input_stride;
  std::memcpy(query_slot, query_data, query_bytes);
```

`query_input_host` 是 `cudaHostAllocMapped` 的 pinned host 缓冲，`query_input_stride` 是按最大 dtype（fp32）对齐后的每 slot 字节数，GPU 端通过 `d_query_input`（device 指针）直接读。这里 **CPU 只写 host 端，GPU 直接读 device 端**，没有显式 `cudaMemcpy`——这是 mapped memory 的核心好处。注意 stride 而非 `query_bytes`：同一个 slot 复用时 dtype 可能变，必须按最大 stride 预留空间。

接着登记 pending，准备 promise/future：

```cpp
  const u64 request_id = next_request_id.fetch_add(1, std::memory_order_relaxed);
  const auto submitted_at = std::chrono::steady_clock::now();
  auto pending = std::make_shared<PendingQuery>();
  pending->slot = slot;
  pending->submitted_at = submitted_at;
  auto future = pending->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    pending_queries.emplace(request_id, pending);
    pending_count.fetch_add(1, std::memory_order_relaxed);
  }
```

`request_id` 用 relaxed 自增即可——它只是 `pending_queries` 的 key，不需要和其他原子量建立顺序。`submitted_at` 现在取，后续 completion 用它算 end-to-end 时延。`pending` 用 `shared_ptr` 是因为 `reject_submission`（health.cc）和 `completion_loop` 都可能从 `pending_queries` 摘出它，`shared_ptr` 让 promise 的归属权在两条路径间安全转移。`pending_count` 是停机 drain 的计数器（`lifecycle.cc:340`）。

构造 `QueryDescriptor`：

```cpp
  QueryDescriptor descriptor{
    .request_id = request_id,
    .snapshot_epoch = engine.delta_.published_epoch(),
    .query_device_address = reinterpret_cast<u64>(
      d_query_input + static_cast<size_t>(slot) * query_input_stride),
    .result_device_address = reinterpret_cast<u64>(
      d_result_ids + static_cast<size_t>(slot) * result_capacity),
    .query_slot = slot,
    .result_capacity = result_capacity,
    .dim = static_cast<u16>(config.dim),
    .k = static_cast<u16>(k),
    .query_dtype = static_cast<u8>(query_dtype),
  };
```

`QueryDescriptor` 的字段见 `src/gpu_search/types.hh:15-27`。几个关键点：

- `snapshot_epoch = engine.delta_.published_epoch()`：先填一个"当前已发布"的 epoch 作为占位。admission 线程稍后会在 push 前用 **最新的** `published_epoch()` 覆盖它（`query_execution.cc:111`），保证 query 看到的是 push 那一刻的快照。`engine.delta_` 是 delta 发布器（见第 10 课、第 15 课）。
- `query_device_address` / `result_device_address` 是 GPU 端指针，kernel 直接用这些地址读 query、写结果。
- `query_slot` 重复传一遍是因为 kernel 完成后回写 `CompletionDescriptor.query_slot` 时要用它定位 slot（见 1.3）。

接着把 descriptor 入 admission_queue 并通知 admission 线程：

```cpp
  bool rejected = false;
  std::string rejection_message;
  {
    std::lock_guard<std::mutex> lock(admission_mutex);
    if (!healthy.load(std::memory_order_relaxed)) {
      rejected = true;
      rejection_message = health_error;
    } else {
      admission_queue.push_back({.descriptor = descriptor, .enqueued_at = submitted_at});
    }
  }
  if (rejected) {
    reject_submission({.descriptor = descriptor, .enqueued_at = submitted_at},
                      rejection_message);
  } else {
    admission_cv.notify_one();
  }
  engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
  return future.get();
```

这里有一个 **TOCTOU**：`search()` 开头检查 `healthy` 用 acquire，但入队时再检查一次 relaxed——因为 `mark_unhealthy`（`health.cc:41`）会在持 `admission_mutex` 时翻 `healthy` 并 swap 走整个 `admission_queue`。如果 `search()` 在 `mark_unhealthy` 之后才拿到 `admission_mutex`，这里就能观察到 `healthy==false`，于是走 `reject_submission` 把已经登记的 pending 摘掉、set_exception、归还 slot。如果不做这次复检，descriptor 会进 queue 但 admission 线程已经退出，永远没人处理。

`reject_submission`（`health.cc:12-39`）做的事情和 `completion_loop` 的归还路径几乎对称：从 `pending_queries` 摘 pending → 清 `active_query_tickets`/`active_query_snapshots` → `promise.set_exception` → 归还 slot → `pending_count--` → 通知 `slot_cv` 和 `maintenance_cv`。

最后 `future.get()` 阻塞调用方线程，直到 completion 线程（或 reject 路径）set_value/set_exception。这就是 dvstor 的同步查询语义：RPC worker 线程在这里挂住，等 GPU 跑完。

### 1.3 `admission_loop()`：把 descriptor 推到 device ring

`src/gpu_search/persistent_engine/query_execution.cc:80-158`。这是 admission 线程的主体，构造于 `construction.cc:1016`：

```cpp
admission_thread = std::thread([this] { admission_loop(); });
```

主体：

```cpp
void PersistentSearchEngine::Impl::admission_loop() {
  std::vector<PendingSubmission> batch;
  batch.reserve(config.gpu_query_slots);
  size_t submitted_count = 0;
  try {
    bind_cuda_device("cudaSetDevice(GPU navigation admission)");
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      submitted_count = 0;
      {
        std::unique_lock<std::mutex> lock(admission_mutex);
        admission_cv.wait(lock, [&] {
          return !admission_queue.empty() ||
                 !healthy.load() || shutdown.load();
        });
        if (!healthy.load(std::memory_order_acquire) || shutdown.load()) return;
        if (admission_queue.empty()) continue;
        const size_t count = std::min<size_t>(
            admission_queue.size(), config.gpu_query_slots);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(admission_queue.front());
          admission_queue.pop_front();
        }
        active_gpu_queries.fetch_add(count, std::memory_order_release);
      }
```

`bind_cuda_device`（`health.cc:106`）确保线程绑到 `config.gpu_device`——admission 线程要操作 mapped memory 和 device ring，必须先 set device。主循环等 `admission_cv`，谓词和 `search()` 的入队路径对称：queue 非空、不健康、停机三者任一满足就唤醒。

被唤醒后先复检 `healthy`/`shutdown`，然后 **一次性把 queue 里最多 `gpu_query_slots` 个条目摘到本地 `batch`**。这一步设计上有两个取舍：

- **摘到本地再处理**：避免在 push device ring（可能 yield 让出）时还持着 `admission_mutex`，阻塞 `search()` 入队。
- **上限 `gpu_query_slots`**：device ring 容量有限，且 `active_gpu_queries` 用来给停机 drain 判断"还有多少在 GPU 上飞"，不能一次摘太多。

`active_gpu_queries.fetch_add(count)` 在释放锁前做，保证 `mark_unhealthy` 看到 `active_gpu_queries > 0` 时能正确等待 completion 归还（实际 `mark_unhealthy` 不等，但停机析构会等 `pending_count==0`）。

接下来是关键段：**snapshot_epoch 绑定 + ticket 分配**：

```cpp
      if (batch.empty()) continue;
      const auto admitted_at = std::chrono::steady_clock::now();
      u64 wait_ns = 0;
      {
        std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
        for (PendingSubmission& submission : batch) {
          submission.descriptor.snapshot_epoch = engine.delta_.published_epoch();
          const u64 query_ticket =
              next_query_ticket.fetch_add(1, std::memory_order_acq_rel);
          const u32 slot = submission.descriptor.query_slot;
          active_query_snapshots[slot].store(
              submission.descriptor.snapshot_epoch + 1,
              std::memory_order_release);
          active_query_tickets[slot].store(query_ticket, std::memory_order_release);
        }
      }
```

`query_snapshot_mutex` 是 delta 引擎和 completion 共享的快照锁（见第 10 课、第 16 课）。在锁内做三件事：

1. **刷新 `snapshot_epoch`**：`search()` 里填的是入队时刻的 `published_epoch()`，这里覆盖成 push 前一刻的最新值。这样 query 看到的 delta 快照尽可能新。`+1` 存到 `active_query_snapshots`（0 保留为"slot 空闲"哨兵，所以 epoch 要 +1）。
2. **分配 `query_ticket`**：单调递增的全局票号。delta 回收（第 16 课 RCU）用 ticket barrier 判断"所有读过旧快照的 query 都已完成"——`active_query_tickets[slot]` 让 reclaim 线程能查每个在飞 query 的 ticket。
3. **acq_rel 语义**：`next_query_ticket` 用 acq_rel 是因为 enqueue_storage_reclaim_barriers（`storage_reclaim.cc:311`）会用 `next_query_ticket.load()-1` 当 barrier，必须看到此前所有 push 的副作用。

注意 `admitted_at` 在锁外取，`wait_ns` 累加的是每个 submission 的 `admitted_at - enqueued_at`——即"在 admission_queue 里排队等了多久"。这是 admission 路径的关键时延指标。

接下来 push 到 device ring：

```cpp
      for (PendingSubmission& submission : batch) {
        while (!submissions.try_push(submission.descriptor)) {
          if (shutdown.load(std::memory_order_acquire) ||
              !healthy.load(std::memory_order_acquire)) {
            throw std::runtime_error("GPU submission ring stopped making progress");
          }
          std::this_thread::yield();
        }
        ++submitted_count;
        wait_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            admitted_at - submission.enqueued_at).count());
      }
```

**这里体现了"立即发布 ring descriptor（不凑批）"**：摘到 `batch` 是为了减少锁持有时间，但每个 descriptor 是逐个 `try_push` 到 ring 的，**没有刻意等凑满 N 个再批量发布**。如果 ring 满（GPU 消费慢），`try_push` 返回 false，admission 线程 yield 让出，直到 ring 有空位。每次 yield 前检查停机/健康，避免在 ring 永远不满的情况下死循环。

`try_push`（`mapped_ring.hh:74-84`）是 sequence-based 无锁 push：读 `enqueue` 位置 → 检查 `sequences[slot]==position`（空）→ 写 entry → `sequences[slot]=position+1`（发布）→ `enqueue++`。GPU 端的 dispatcher kernel 通过 device 端的 `dequeue_position`（`device_owned_position_`，`mapped_ring.hh:51`）消费。这条 ring 是 host→device 方向，所以 dequeue 端在 device（见第 17 课 device ring）。

异常处理：

```cpp
      engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.batch_queries.fetch_add(batch.size(), std::memory_order_relaxed);
      engine.telemetry_.submission_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    }
  } catch (const std::exception& error) {
    for (size_t index = submitted_count; index < batch.size(); ++index) {
      reject_submission(batch[index], error.what());
    }
    const size_t rejected_count = batch.size() - submitted_count;
    if (rejected_count != 0) {
      active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
      maintenance_cv.notify_all();
    }
    mark_unhealthy(std::string{"GPU admission failed: "} + error.what());
  } catch (...) {
    for (size_t index = submitted_count; index < batch.size(); ++index) {
      reject_submission(batch[index], "unknown GPU admission failure");
    }
    const size_t rejected_count = batch.size() - submitted_count;
    if (rejected_count != 0) {
      active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
      maintenance_cv.notify_all();
    }
    mark_unhealthy("unknown GPU admission failure");
  }
}
```

遥测三连：`batches`（admission 循环次数）、`batch_queries`（本批 query 数）、`submission_wait_ns`（排队等待纳秒）。注意 `batches` 计的是"admission 被唤醒并摘到非空 batch 的次数"，不是 query 数。

异常路径很讲究：`submitted_count` 记录本批已成功 push 的数量。catch 时只 reject `batch.size() - submitted_count` 个（即还没 push 的），因为已 push 的会走正常 completion 路径（kernel 会回 CompletionDescriptor，哪怕 status 非零）。`active_gpu_queries` 只减掉未 push 的部分——已 push 的要等 completion 来减。最后 `mark_unhealthy` 进入 fail-stop：翻 `healthy=false`、swap 走整个 `admission_queue` 并 reject、通知所有 CV（`health.cc:41-57`）。`catch(...)` 兜底未知异常，行为对称。

> **fail-stop 语义**：admission 任何一步失败（ring 长期不满、CUDA 错误、抛异常）都直接 `mark_unhealthy`，整个引擎停止接受新 query，已 in-flight 的 query 通过 reject 或 completion 的 status 非零路径拿到异常。这是 dvstor 一致性策略的核心——不尝试局部恢复，而是停机让人工介入。与第 11 课 lifecycle 的停机流程衔接。

### 1.4 admission 与第 11 课 lifecycle、第 17 课 device ring 的关系

- **lifecycle**（第 11 课）：`admission_thread` 在 `construction.cc:1016` 创建，在 `~Impl`（`lifecycle.cc:322-357`）中通过 `shutdown.store(true)` + `admission_cv.notify_all()` 唤醒后 join。停机顺序很重要：先停 maintenance，再停 admission（`lifecycle.cc:333-336`），drain `pending_count`（`lifecycle.cc:340`），最后停 completion（`lifecycle.cc:357`）。这保证 admission 不再 push 后，completion 还能处理完在飞的。
- **device ring**（第 17 课）：`submissions` 是 host→device 方向的 `MappedRing<QueryDescriptor>`，`completions` 是 device→host 方向的 `MappedRing<CompletionDescriptor>`。admission 写 submissions 的 host 端，GPU dispatcher kernel 读 device 端；GPU 写 completions 的 device 端（通过 `DeviceRingView`），completion 线程读 host 端。两条 ring 用 `cudaHostAllocMapped` 实现 zero-copy 跨界通信。

---

## 二、`routing.cc`：delta 解码、anchor 路由与 anchor graph 刷新

这个文件包含四个函数，都是 **admission/completion 之外的辅助路由逻辑**，被 delta 发布（第 15 课）和 maintenance 调用：

1. `decode_mutation_payload` — 把存储侧发来的 delta mutation 向量解码成 fp32。
2. `nearest_anchor` — 给定一个向量，找它所在 shard 的最近 anchor（路由种子）。
3. `graph_cache_key` / `graph_cache_keys` — 把存储侧的 raw node 指针换算成 graph cache 的 key。
4. `refresh_anchor_graph_records` — 当 anchor graph 失效时，通过 RDMA 重读对应 slot 的图记录并校验。

### 2.1 `decode_mutation_payload()`

`src/gpu_search/persistent_engine/routing.cc:7-20`：

```cpp
void PersistentSearchEngine::Impl::decode_mutation_payload(const DeltaMutation& mutation,
                             std::vector<f32>& decoded) const {
  std::fill(decoded.begin(), decoded.end(), 0.0f);
  if (mutation.kind == service::storage_owner::MutationKind::erase) return;
  if (mutation.vector.size() == static_cast<size_t>(config.dim) * sizeof(f32)) {
    std::memcpy(decoded.data(), mutation.vector.data(), mutation.vector.size());
  } else if (mutation.vector.size() ==
             vector_dtype_bytes(config.resolved_vector_dtype(), config.dim)) {
    decode_storage_vector_to_float(mutation.vector.data(), config.resolved_vector_dtype(),
                                   config.dim, decoded.data());
  } else {
    throw std::invalid_argument("GPU delta mutation vector has an invalid size");
  }
}
```

`DeltaMutation` 是存储侧推来的增量（见第 10 课、第 15 课）。`decoded` 是调用方预分配好的 `dim` 长 fp32 缓冲。逻辑：

- 先清零，`erase` 类型直接返回（删除不需要向量，用零向量占位）。
- 第一种尺寸匹配：存储侧直接给了 fp32 原始向量，memcpy。
- 第二种尺寸匹配：存储侧给的是压缩 dtype（fp16/bf16，由 `config.resolved_vector_dtype()` 决定），调 `decode_storage_vector_to_float` 解码。
- 都不匹配：抛异常——这是协议级错误，说明存储和计算侧的 dim/dtype 配置不一致。

`decode_storage_vector_to_float` 是第 9 课的 GPU 类型工具。这个函数被 delta 发布路径调用（第 15 课），把存储侧的向量转成 fp32 后才能和 anchor 比较、写进 GPU delta buffer。

### 2.2 `nearest_anchor()`

`src/gpu_search/persistent_engine/routing.cc:22-45`：

```cpp
u32 PersistentSearchEngine::Impl::nearest_anchor(const std::vector<f32>& vector, u64 remote_node) const {
  if (anchor_table.count() == 0) return 0;
  const u32 shard = static_cast<u32>(remote_node >> 48);
  if (shard >= index.shards.size()) return 0;
  const u32 begin = anchor_table.shard_offsets[shard];
  const u32 end = anchor_table.shard_offsets[shard + 1];
  if (begin == end) return 0;
  u32 best = begin;
  f32 best_distance = std::numeric_limits<f32>::max();
  for (u32 anchor = begin; anchor < end; ++anchor) {
    f32 distance = 0.0f;
    const f32* candidate = anchor_table.vectors.data() +
      static_cast<size_t>(anchor) * config.dim;
    for (u32 dimension = 0; dimension < config.dim; ++dimension) {
      const f32 difference = vector[dimension] - candidate[dimension];
      distance += difference * difference;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best = anchor;
    }
  }
  return best;
}
```

`anchor_table`（`impl.hh:51-61`、`impl.hh:166`）是 CPU 侧的 anchor 索引（第 6 课 Vamana anchor）。`remote_node >> 48` 取高 16 位作为 shard id（这是 dvstor 的 `RemotePtr` 编码，见第 6 课）。每个 shard 的 anchor 在 `shard_offsets[shard..shard+1]` 区间内。

函数做暴力最近邻（L2 平方距离，不开方）：遍历该 shard 所有 anchor，找距离最小的返回 anchor 全局序号。这是 O(anchor_count_per_shard × dim) 的 CPU 计算，只在 delta 发布时为每个 mutation 调一次（不是每 query 调），所以暴力扫可接受。返回的 anchor 序号用于：mutation 写进 GPU delta bucket 时定位 bucket head（第 10 课 delta overlay）。

注意 `best` 初始化为 `begin` 而非 0——如果所有距离都相等（理论上不会但要防御），返回该 shard 的第一个 anchor 而非全局 0（可能是别的 shard 的 anchor）。`begin==end` 时返回 0 是"无 anchor"哨兵。

### 2.3 `graph_cache_key()` 与 `graph_cache_keys()`

`src/gpu_search/persistent_engine/routing.cc:47-91`。这两个函数把存储侧报来的 raw node 指针换算成 graph cache 用的 64-bit key。

```cpp
u64 PersistentSearchEngine::Impl::graph_cache_key(u64 raw) const {
  const u32 shard = static_cast<u32>(raw >> 48);
  const u64 node_offset = (raw << 16) >> 16;
  if (raw == 0 || shard >= index.shards.size()) {
    throw std::runtime_error("storage returned an invalid GPU graph-cache invalidation");
  }
  const format::ShardRegion& region = index.shards[shard];
  u64 graph_offset = 0;
  if (node_offset >= region.node_base_offset && region.node_stride != 0) {
    const u64 relative = node_offset - region.node_base_offset;
    if (relative % region.node_stride == 0 &&
        relative / region.node_stride < region.node_count) {
      graph_offset = region.graph_base_offset +
        (relative / region.node_stride) * index.layout.graph_entry_bytes;
        }
  }
  if (graph_offset == 0) {
    if (node_offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0 ||
        (node_offset - region.dynamic_base_offset) % region.dynamic_record_bytes != 0) {
      throw std::runtime_error("storage returned a misaligned GPU graph-cache invalidation");
    }
    graph_offset = node_offset + region.dynamic_hot_offset;
  }
  return (static_cast<u64>(shard) << 48) | graph_offset;
}
```

raw 指针编码：高 16 位 shard，低 48 位字节偏移。函数要做的是把这个偏移从"node record 偏移"或"dynamic record 偏移"换算成"graph entry 偏移"——因为 graph cache 是按 graph entry 索引的，而存储侧失效通知用的是 node/dynamic 偏移。

两条路径：

1. **base node 路径**：`node_offset` 落在 `[node_base_offset, node_base_offset + node_count*node_stride)` 且对齐到 `node_stride`，说明是 base 节点。`graph_offset = graph_base_offset + ordinal * graph_entry_bytes`。
2. **dynamic node 路径**：如果上面没命中（`graph_offset==0`），检查是否在 dynamic 区间且对齐 `dynamic_record_bytes`。dynamic 节点的 graph entry 在 record 内部的 `dynamic_hot_offset` 处，所以 `graph_offset = node_offset + dynamic_hot_offset`。

两条都不命中就抛异常——存储侧报了一个无法解释的偏移。最后把 shard 编码回高 16 位，形成 cache key。这个 key 用来索引 GPU graph cache（第 19 课 RDMA cache）的失效。

`graph_cache_keys()` 是批量版本（`routing.cc:73-91`）：

```cpp
std::vector<u64> PersistentSearchEngine::Impl::graph_cache_keys(std::span<const u64> raw_nodes) const {
  std::vector<u64> keys;
  keys.reserve(raw_nodes.size());
  for (u64 raw : raw_nodes) {
    const u64 key = graph_cache_key(raw);
    if (graph_cache_sets == 0 &&
        !std::binary_search(anchor_graph_keys_host.begin(),
                            anchor_graph_keys_host.end(), key)) {
      continue;
    }
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  if (keys.size() > graph_invalidation_capacity) {
    throw std::runtime_error("GPU navigation graph invalidation batch exceeds capacity");
  }
  return keys;
}
```

两个过滤条件：

- `graph_cache_sets == 0`：graph cache 未启用时，只保留那些属于 anchor graph 的 key（`anchor_graph_keys_host` 是预排序的 anchor graph key 列表，用 `binary_search` 查）。非 anchor 的失效通知在无 cache 模式下没意义（kernel 会现场 RDMA 读）。
- 启用 graph cache 时不过滤——所有失效都要处理。

之后排序+去重（同一 batch 可能报多次同一节点），超 `graph_invalidation_capacity` 抛异常（GPU 端失效缓冲区有上限）。返回的 keys 用于 delta 发布时随 `DeltaPublishDescriptor.invalidation_count` 一起推给 kernel（第 15 课）。

### 2.4 `refresh_anchor_graph_records()`

`src/gpu_search/persistent_engine/routing.cc:93-186`。这是本文件最长的函数，处理 anchor graph 记录的 RDMA 刷新。anchor graph 是 GPU 上的静态路由图（第 6 课 anchor index 的图部分），当存储侧重写某个 anchor 对应的图记录后，需要重读到 GPU。

```cpp
void PersistentSearchEngine::Impl::refresh_anchor_graph_records(std::span<const u64> invalidation_keys) {
  if (invalidation_keys.empty() || anchor_graph_keys_host.empty()) return;
  std::vector<u32> route_slots;
  route_slots.reserve(invalidation_keys.size());
  for (u64 key : invalidation_keys) {
    const auto iterator = std::lower_bound(
      anchor_graph_keys_host.begin(), anchor_graph_keys_host.end(), key);
    if (iterator != anchor_graph_keys_host.end() && *iterator == key) {
      route_slots.push_back(static_cast<u32>(
        iterator - anchor_graph_keys_host.begin()));
    }
  }
  if (route_slots.empty()) return;
```

入参 `invalidation_keys` 是失效的 graph key。第一步把 key 换算成 anchor graph 在 GPU 上的 slot 序号——`anchor_graph_keys_host` 是排序好的 key 数组，slot 就是数组下标。不在数组里的 key 直接忽略（不是 anchor graph 节点）。

接下来要 **等 GPU 上的读者退出**：

```cpp
  const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
    config.storage_owner_rpc_timeout_ms, 1000, 5000));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    check_cuda(cudaMemcpyAsync(
                 anchor_graph_readers_host, d_anchor_graph_readers,
                 anchor_graph_keys_host.size() * sizeof(u32),
                 cudaMemcpyDeviceToHost, route_refresh_stream),
               "cudaMemcpyAsync(anchor route readers)");
    check_cuda(cudaStreamSynchronize(route_refresh_stream),
               "cudaStreamSynchronize(anchor route readers)");
    bool busy = false;
    for (u32 slot : route_slots) {
      if (anchor_graph_readers_host[slot] != 0) {
        busy = true;
        break;
      }
    }
    if (!busy) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        "anchor route graph refresh timed out waiting for active readers");
    }
    std::this_thread::yield();
  }
```

`d_anchor_graph_readers` 是 GPU 上的 per-slot 读者计数（kernel 进入临界区时 `atomicAdd`，退出时 `atomicSub`）。CPU 用 `route_refresh_stream`（专用 CUDA stream，`impl.hh:380`，在 `construction.cc:836` 创建）异步拷回来，同步后检查目标 slot 的计数是否全 0。这是 RCU 风格的等待——不阻塞读者，而是等读者自然退出。超时 1-5 秒（按配置钳位），超时抛异常触发 fail-stop。

读者退出后，构造 RDMA 读请求重读图记录：

```cpp
  std::vector<NavigationRead> requests;
  std::vector<i32> statuses(route_slots.size(), -EIO);
  requests.reserve(route_slots.size());
  for (u32 slot : route_slots) {
    const u64 key = anchor_graph_keys_host[slot];
    requests.push_back(NavigationRead{
      .remote_offset = (key << 16) >> 16,
      .destination_address = reinterpret_cast<u64>(
        d_anchor_graph_records +
        static_cast<size_t>(slot) * index.layout.graph_entry_bytes),
      .memory_node = static_cast<u16>(key >> 48),
    });
  }
  control_bootstrapper->read(requests, statuses);
```

`NavigationRead`（`src/gpu_search/navigation_bootstrapper.hh:15`）是 RDMA 读请求：`remote_offset` 取 key 低 48 位（图 entry 在存储侧的字节偏移），`destination_address` 是 GPU 上 anchor graph 记录区的对应 slot，`memory_node` 取 key 高 16 位（shard）。`control_bootstrapper` 是 RDMA 传输层（第 4-5 课）。这是把存储侧最新的图记录直接 RDMA DMA 到 GPU 显存。

读完后 **逐条校验**：

```cpp
  for (size_t request = 0; request < statuses.size(); ++request) {
    if (statuses[request] <= 0) {
      throw std::runtime_error(
        "anchor route graph refresh RDMA read failed: slot=" +
        std::to_string(route_slots[request]) + " status=" +
        std::to_string(statuses[request]));
    }
    check_cuda(cudaMemcpyAsync(
                 anchor_graph_validation_host,
                 d_anchor_graph_records +
                   static_cast<size_t>(route_slots[request]) *
                     index.layout.graph_entry_bytes,
                 index.layout.graph_entry_bytes, cudaMemcpyDeviceToHost,
                 route_refresh_stream),
               "cudaMemcpyAsync(anchor route validation)");
    check_cuda(cudaStreamSynchronize(route_refresh_stream),
               "cudaStreamSynchronize(anchor route validation)");
    const u16 expected = vamana::hot_graph::load_u16_le(
      anchor_graph_validation_host + 2);
    const u16 actual = vamana::hot_graph::checksum16(
      anchor_graph_validation_host, index.layout.graph_entry_bytes);
    if (anchor_graph_validation_host[0] > index.layout.graph_degree ||
        expected != actual) {
      throw std::runtime_error(
        "anchor route graph refresh produced an invalid record at slot " +
        std::to_string(route_slots[request]));
    }
  }
```

每个 slot 单独处理：先看 RDMA status（`<=0` 即失败，`navigation_bootstrapper` 用正值表示成功字节数），再把刚写的图 entry 拷回 host 校验。校验两项：

- `anchor_graph_validation_host[0]`（图 entry 的第一个字节，是实际度数）不能超过 `graph_degree`。
- `checksum16`：entry 偏移 2 处存的是预期的 checksum（小端 u16），用 `vamana::hot_graph::checksum16` 重算整个 entry 比较。这是 vamana hot graph 格式的自校验（第 6 课）。

任一不符抛异常。注意这里 **每个 slot 都做一次 device→host copy + sync**——慢但正确，因为校验必须看实际写到 GPU 的字节，而 RDMA 写完成后 GPU L1/L2 可能还有缓存。逐 slot 同步保证校验看到的是 RDMA 真正落盘的字节。

最后发布"ready"状态：

```cpp
  check_cuda(cudaMemcpyAsync(
               d_anchor_graph_states, anchor_graph_ready_states_host.data(),
               anchor_graph_ready_states_host.size() * sizeof(u32),
               cudaMemcpyHostToDevice, route_refresh_stream),
             "cudaMemcpyAsync(anchor route ready states)");
  check_cuda(cudaStreamSynchronize(route_refresh_stream),
             "cudaStreamSynchronize(anchor route ready states)");
  engine.telemetry_.graph_route_refreshes.fetch_add(
    route_slots.size(), std::memory_order_relaxed);
}
```

`d_anchor_graph_states` 是 GPU 上的 per-slot 状态机：0=未就绪，非 0=就绪。`anchor_graph_ready_states_host` 预填了就绪值，整体 H2D 拷过去后，kernel 才会使用这些刷新过的图记录。`graph_route_refreshes` 遥测累加刷新的 slot 数。这条路径和第 19 课 RDMA cache 的失效/刷新机制互补——anchor graph 是"静态常驻"的图记录，不走通用 graph cache，而是有自己的专属刷新流。

---

## 三、路由快照拉取（`storage_reclaim.cc`）

任务原文说的"从 4 KiB control page offset 1024、checksum/seqlock 校验、按 epoch 更新 GPU 动态路由槽、route-only command 发布"实际由 `storage_reclaim.cc` 的两个函数实现，由 maintenance 线程调用。这里按真实代码讲解。

### 3.1 control page 布局

`src/gpu_search/index_format.hh:25-36`：

```cpp
inline constexpr u32 kStorageControlBytes = 4096;             // control page 总大小
inline constexpr u64 kStorageControlMagic = 0x314c525443565344ULL;  // "DSVCTRL1"
inline constexpr u32 kStorageControlVersion = 2;
// ...
inline constexpr u32 kStorageRoutePublicationOffset = 1024;   // 路由发布区在 page 内偏移
inline constexpr u64 kStorageRoutePublicationMagic = 0x3154554f52565344ULL;  // "DSVROUT1"
inline constexpr u32 kStorageRoutePublicationVersion = 1;
inline constexpr u32 kStorageRouteSlots = 8;
```

每个存储 shard 在自己的 control page（4 KiB）里维护：开头是 `StorageControlBlock`（640 字节，`index_format.hh:80-98`，含 magic/version/shard_id/维护序列号/reclaim ACK 数组等），偏移 1024 处是 `StorageRoutePublication`（448 字节，`index_format.hh:112-124`）。`static_assert`（`index_format.hh:150-152`）保证 1024 ≥ sizeof(StorageControlBlock) 且 1024+448 ≤ 4096，不挤占其他区。

`StorageRoutePublication` 结构（`index_format.hh:112-124`）：

```cpp
struct alignas(64) StorageRoutePublication {
  u64 sequence_begin{};
  u64 magic{kStorageRoutePublicationMagic};
  u32 version{kStorageRoutePublicationVersion};
  u32 header_bytes{sizeof(StorageRoutePublication)};
  u32 shard_id{};
  u32 slot_count{kStorageRouteSlots};
  u32 code_bytes{};
  u32 reserved{};
  u64 body_checksum{};
  std::array<StorageRouteSlot, kStorageRouteSlots> slots{};
  u64 sequence_end{};
};
```

注释（`index_format.hh:107-111`）说明这是 **seqlock + checksum** 双重保护：`sequence_begin`/`sequence_end` 是设备端 seqlock（写者更新时先 `sequence_begin++` 变奇数，写完 `sequence_end++` 变偶数），`body_checksum` 让撕裂的 body 可检测。计算读者额外在 body RDMA 前后各读一次 `sequence_begin`，排除"旧 body 配新 sequence"的相干读。

### 3.2 `read_storage_route_publications()`

`src/gpu_search/persistent_engine/storage_reclaim.cc:74-194`。这是 maintenance 线程拉路由快照的核心。

```cpp
std::vector<format::StorageRoutePublication>
PersistentSearchEngine::Impl::read_storage_route_publications() {
  if (control_bootstrapper == nullptr || index.shards.empty()) return {};
  std::vector<NavigationRead> requests(index.shards.size());
  std::vector<i32> before_statuses(index.shards.size(), -EIO);
  std::vector<i32> body_statuses(index.shards.size(), -EIO);
  std::vector<i32> after_statuses(index.shards.size(), -EIO);
  std::vector<format::StorageRoutePublication> publications(index.shards.size());
  std::vector<u64> sequences_before(index.shards.size());
  std::vector<u64> sequences_after(index.shards.size());
  std::string last_error;
  bool last_failure_was_transient = false;
  bool saw_nontransient_failure = false;
```

每个 shard 要做三次 RDMA：before（读 `sequence_begin`）、body（读整个 publication）、after（再读 `sequence_begin`）。三次的状态分别用三个 statuses 数组追踪。`sequences_before/after` 存前后两次读到的 sequence。

重试循环最多 2 次：

```cpp
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    last_failure_was_transient = false;
    for (size_t shard = 0; shard < index.shards.size(); ++shard) {
      requests[shard] = NavigationRead{
        .remote_offset = index.shards[shard].control_remote_offset +
          format::kStorageRoutePublicationOffset +
          offsetof(format::StorageRoutePublication, sequence_begin),
        .destination_address = reinterpret_cast<u64>(
          d_storage_route_sequence_before + shard),
        .bytes = sizeof(u64),
        .memory_node = static_cast<u16>(shard),
      };
      before_statuses[shard] = -EIO;
    }
    control_bootstrapper->read(requests, before_statuses);
```

`control_remote_offset` 是该 shard control page 在远存的起始偏移，加上 `kStorageRoutePublicationOffset`（1024）再加上 `sequence_begin` 的成员偏移，得到 `sequence_begin` 字段的绝对远端地址。读 8 字节到 `d_storage_route_sequence_before[shard]`。这是 **before** 读。

body 读：

```cpp
    for (size_t shard = 0; shard < index.shards.size(); ++shard) {
      requests[shard] = NavigationRead{
        .remote_offset = index.shards[shard].control_remote_offset +
          format::kStorageRoutePublicationOffset,
        .destination_address = reinterpret_cast<u64>(
          d_storage_route_snapshots + shard),
        .bytes = sizeof(format::StorageRoutePublication),
        .memory_node = static_cast<u16>(shard),
      };
      body_statuses[shard] = -EIO;
    }
    control_bootstrapper->read(requests, body_statuses);
```

读整个 448 字节 publication 到 `d_storage_route_snapshots[shard]`。注意这里直接 DMA 到 GPU 显存（`d_storage_route_snapshots` 是 device 指针，`impl.hh:284`），不经过 host。

after 读（同 before，目标 `d_storage_route_sequence_after`，`storage_reclaim.cc:115-127`）。

三次 RDMA 完成后拷回 host 校验：

```cpp
    check_cuda(cudaMemcpy(publications.data(), d_storage_route_snapshots, ...));
    check_cuda(cudaMemcpy(sequences_before.data(), d_storage_route_sequence_before, ...));
    check_cuda(cudaMemcpy(sequences_after.data(), d_storage_route_sequence_after, ...));
    bool valid = true;
    for (size_t shard = 0; shard < publications.size(); ++shard) {
      if (before_statuses[shard] <= 0 || body_statuses[shard] <= 0 ||
          after_statuses[shard] <= 0) {
        last_error = "RDMA read failed for shard " + std::to_string(shard);
        saw_nontransient_failure = true;
        valid = false;
        break;
      }
      if (sequences_before[shard] != sequences_after[shard] ||
          sequences_before[shard] != publications[shard].sequence_begin) {
        last_error = "shard " + std::to_string(shard) +
          ": storage route changed across RDMA snapshot";
        last_failure_was_transient = true;
        valid = false;
        break;
      }
```

校验三连：

1. **RDMA status 全正**：任一非正即 RDMA 失败，标记 `saw_nontransient_failure`（非瞬态——网络/权限问题不会自愈）。
2. **seqlock 一致**：`before == after == publication.sequence_begin`。不等说明 body 读期间存储侧正在写（sequence_begin 变奇再变偶），body 可能撕裂。这是瞬态——下一拍重试即可。
3. **publication 自校验**（下一步 `validate_storage_route_publication`）：检查 magic/version/header_bytes/shard_id/slot_count/body_checksum。其中 "overlaps publication" 和 "checksum mismatch" 被判为瞬态（`storage_reclaim.cc:164-166`），其他（magic 错、version 错）是非瞬态。

还有一项 `code_bytes` 匹配检查（`storage_reclaim.cc:172-178`）：路由 PQ code 宽度必须和计算侧索引一致，否则非瞬态失败。

```cpp
  if (last_failure_was_transient && !saw_nontransient_failure) {
    // Route metadata is advisory. A torn low-frequency control-page read must
    // never fail queries or the mutation engine; retain the previous GPU
    // snapshot and retry on the next maintenance tick.
    engine.telemetry_.dynamic_route_snapshot_skips.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  throw std::runtime_error(
    "storage route snapshot unavailable after retry: " + last_error + ...);
}
```

重试两次都失败后的处理体现 **"路由是建议性元数据"** 的设计哲学：

- 瞬态失败（seqlock 撕裂、checksum 不匹配）：不抛异常，返回空 vector，`dynamic_route_snapshot_skips++`，下一拍 maintenance 再试。GPU 上保留旧快照继续服务 query。
- 非瞬态失败（RDMA 错、magic/version 错、code_bytes 不匹配）：抛异常，被 `maintenance_loop` catch 后 `mark_unhealthy`（`storage_reclaim.cc:580-583`）。因为这表明部署不一致，继续跑会出错。

### 3.3 `synchronize_storage_routes()`

`src/gpu_search/persistent_engine/storage_reclaim.cc:196-265`。把读到的 publication 落到 GPU 动态路由槽。

```cpp
bool PersistentSearchEngine::Impl::synchronize_storage_routes() {
  const std::vector<format::StorageRoutePublication> publications =
    read_storage_route_publications();
  if (publications.empty()) return false;
  if (dynamic_route_snapshot.size() != dynamic_route_capacity) {
    throw std::logic_error("canonical storage route snapshot capacity mismatch");
  }
  for (u32 shard = 0; shard < publications.size(); ++shard) {
    for (u32 local_slot = 0; local_slot < format::kStorageRouteSlots;
         ++local_slot) {
      const auto& source = publications[shard].slots[local_slot];
      const u32 slot = shard * format::kStorageRouteSlots + local_slot;
      dynamic_route_snapshot[slot] =
        vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot{
          .shard = shard,
          .slot = local_slot,
          .initialized = source.remote_node != 0 || source.generation != 0,
          .live = source.remote_node != 0,
          .id = source.id,
          .generation = source.generation,
          .entry = RemotePtr{source.remote_node},
        };
    }
  }
```

`publications.empty()` 即"瞬态跳过"——直接返回 false，maintenance 不做后续 reclaim barrier。否则把每个 shard 的 8 个 slot 展平成 `dynamic_route_snapshot`（全局 `shard*8+local`）。`RouteSlotSnapshot` 是 `vamana::routing::AdaptiveRouteTable` 的快照结构（第 6 课 adaptive_route_table）。`live = remote_node != 0` 表示该路由有效（非零远指针），`initialized` 还包容 generation 非零的"已分配但暂无远端"状态。

统计 live 槽位 + diff：

```cpp
  const u64 live_routes = static_cast<u64>(std::count_if(
    dynamic_route_snapshot.begin(), dynamic_route_snapshot.end(),
    [](const auto& slot) { return slot.live; }));
  engine.telemetry_.dynamic_route_live_slots.store(live, std::memory_order_relaxed);

  // prepare() compares only canonical slot contents.  Epoch 1 is a harmless
  // placeholder; reserve the real ordered query epoch only when something
  // actually changed.
  dynamic_route_diff->prepare(
    dynamic_route_snapshot, 1, dynamic_route_update_scratch);
  if (dynamic_route_update_scratch.empty()) return true;
```

`DynamicRouteOverlayDiff`（`src/gpu_search/dynamic_route_overlay.hh`，第 10 课）做增量 diff：把新快照和上次 commit 的快照比，把变化项写进 `dynamic_route_update_scratch`。没有变化就返回 true（说明本轮 maintenance 成功，可以推进 reclaim barrier）但不发布任何 command。注释说明 `prepare` 只比内容不比 epoch，所以传 1 占位；真实 epoch 等确定有变化才 `reserve_epoch()`。

有变化时构造 route-only delta command：

```cpp
  const u64 epoch = engine.delta_.reserve_epoch();
  for (size_t update_index = 0;
       update_index < dynamic_route_update_scratch.size(); ++update_index) {
    DynamicRouteUpdate& update = dynamic_route_update_scratch[update_index];
    update.epoch = epoch;
    std::memcpy(
      dynamic_route_code_updates_host + update_index * code_bytes,
      publications[update.shard]
        .slots[update.slot % format::kStorageRouteSlots]
        .navigation_code.data(),
      code_bytes);
  }
  std::memcpy(dynamic_route_updates_host,
              dynamic_route_update_scratch.data(),
              dynamic_route_update_scratch.size() * sizeof(DynamicRouteUpdate));
  submit_delta_publication(DeltaPublishDescriptor{
    .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
    .final_count = static_cast<u32>(delta_records_host.size()),
    .dynamic_route_count = static_cast<u32>(dynamic_route_update_scratch.size()),
  });
  dynamic_route_diff->commit(dynamic_route_update_scratch);
```

关键步骤：

1. **`reserve_epoch()`**：在 delta 发布器（第 10 课、第 15 课）预订一个新的有序 query epoch。这个 epoch 会被写进每个 `DynamicRouteUpdate.epoch`，kernel 端只对 `snapshot_epoch >= update.epoch` 的 query 应用新路由——这就是第 10 课 dynamic_route_overlay 的可见性闸门。
2. **PQ code 拷贝**：每个路由槽带一份 `navigation_code`（路由专用 PQ 码，`kStorageRouteMaxCodeBytes=32`），逐 slot memcpy 到 `dynamic_route_code_updates_host`。
3. **`submit_delta_publication`**：发一个 `DeltaPublishDescriptor`，其中 `dynamic_route_count` 非零、其他 count 字段为零——这就是 **route-only command**。`final_count` 传当前 delta 表大小（保持 kernel 的 final 计数一致）。发布细节见第 15 课。
4. **`dynamic_route_diff->commit`**：把本轮 scratch 作为新基线，下次 diff 基于此。

最后发布 barrier：

```cpp
  engine.telemetry_.dynamic_route_publications.fetch_add(1, std::memory_order_relaxed);
  engine.telemetry_.dynamic_route_slot_updates.fetch_add(
    dynamic_route_update_scratch.size(), std::memory_order_relaxed);
  // Queries acquire this epoch only after the control CTA has made both the
  // PQ bytes and route seqlocks visible.
  engine.delta_.publish_barrier(epoch);
  return true;
}
```

`publish_barrier(epoch)`（第 15 课）等 control CTA 把 PQ code 和路由 seqlock 都写完，然后才让 `published_epoch()` 推进到 `epoch`。此后新 admission 的 query 才会拿到这个 epoch 作为 `snapshot_epoch`，从而看到新路由。注释点明了"双可见性"——PQ 字节和 seqlock 都要就绪，缺一不可。

### 3.4 与第 10 课、第 6 课的关系

- **第 10 课 dynamic_route_overlay**：`DynamicRouteOverlayDiff`、`reserve_epoch`/`publish_barrier`/`published_epoch` 都是该模块的 API。本课只展示调用点，diff 算法和 epoch 发布器的内部实现在第 10 课。
- **第 6 课 adaptive_route_table**：`RouteSlotSnapshot` 是 `vamana::routing::AdaptiveRouteTable` 的快照类型，anchor graph（`anchor_graph_keys_host`/`d_anchor_graph_records`）来自第 6 课的 anchor index。`routing.cc` 的 `nearest_anchor`/`refresh_anchor_graph_records` 都在操作第 6 课定义的 anchor 表。
- **route refresh 遥测**：`graph_route_refreshes`（`routing.cc:184`）、`dynamic_route_snapshot_skips`（`storage_reclaim.cc:186`）、`dynamic_route_publications`/`dynamic_route_slot_updates`/`dynamic_route_live_slots`（`storage_reclaim.cc:223/257/259`）是路由子系统的核心遥测，见第 9 课遥测、第 30 课 breakdown benchmark。

---

## 四、`completion.cc`：CPU completion 线程

这个文件含两个函数：`report_direct_path_failure()`（GPUNetIO 直读失败上报）和 `completion_loop()`（主循环）。

### 4.1 `report_direct_path_failure()`

`src/gpu_search/persistent_engine/completion.cc:9-37`：

```cpp
void PersistentSearchEngine::Impl::report_direct_path_failure() {
  if (direct_disabled_host == nullptr || direct_disabled_device == nullptr ||
      direct_error_host == nullptr || direct_error_device == nullptr) return;
  check_cuda(cudaMemcpyAsync(direct_disabled_host, direct_disabled_device,
                             sizeof(u32), cudaMemcpyDeviceToHost, rdma_stream),
             "cudaMemcpyAsync(GPUNetIO failure flag)");
  check_cuda(cudaMemcpyAsync(direct_error_host, direct_error_device,
                             sizeof(i32), cudaMemcpyDeviceToHost, rdma_stream),
             "cudaMemcpyAsync(GPUNetIO failure status)");
  check_cuda(cudaStreamSynchronize(rdma_stream),
             "cudaStreamSynchronize(GPUNetIO failure status)");
  if (*direct_disabled_host == 0) return;
  bool expected = false;
  if (!direct_failure_logged.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel)) return;
  const i32 direct_error = *direct_error_host;
  const bool graph_snapshot_error = direct_error == -EBADMSG;
  std::cerr << "[gpu-search] "
            << (graph_snapshot_error
                  ? "graph snapshot validation failed after bounded rereads"
                  : "GPUNetIO direct read failed")
            << " with status=" << direct_error
            << "; strict query mode rejects the query\n";
  engine.telemetry_.direct_path_failures.fetch_add(1, std::memory_order_relaxed);
  mark_unhealthy(std::string(graph_snapshot_error
                 ? "graph snapshot validation failed with status "
                 : "GPUNetIO direct read failed with status ") +
                 std::to_string(direct_error));
}
```

这个函数在 completion 发现 `status != 0` 时调用。`direct_disabled_device`/`direct_error_device` 是 GPU 上的两个标量（kernel 检测到 GPUNetIO 直读失败时置位，第 22 课）。CPU 用 `rdma_stream` 把它们拷回 host 同步读。

如果 `direct_disabled_host == 0`（kernel 没置位），说明 status 非零不是 GPUNetIO 直读失败，直接返回让 completion 走通用错误路径。否则用 `direct_failure_logged` 这个 atomic + CAS 保证只记一次日志（避免每个失败 query 都刷屏）。`-EBADMSG` 特殊处理为"图快照校验失败"（bounded rereads 后仍不一致），其他错误码算"GPUNetIO 直读失败"。最后 `mark_unhealthy`——GPUNetIO 直读失败是 fail-stop 触发条件，因为继续跑会污染 query 结果。GPUNetIO 传输细节见第 22 课。

### 4.2 `completion_loop()`：主循环

`src/gpu_search/persistent_engine/completion.cc:39-182`。

```cpp
void PersistentSearchEngine::Impl::completion_loop() {
  while (!shutdown.load(std::memory_order_acquire) ||
         pending_count.load(std::memory_order_acquire) != 0) {
    CompletionDescriptor completion;
    if (!completions.try_pop(completion)) {
      std::this_thread::yield();
      continue;
    }
    if (completion.status != 0) report_direct_path_failure();
```

循环条件很关键：`shutdown || pending_count != 0` 时继续。即 **停机后还要把在飞的 query 处理完**（或等到析构的 drain 超时）。`try_pop`（`mapped_ring.hh:86-96`）从 device→host ring 非阻塞取一个 `CompletionDescriptor`。取不到就 yield，不阻塞——completion 线程是纯轮询。

`try_pop` 失败时 yield 而非 sleep，是因为 GPU kernel 可能随时写完成，sleep 会引入微秒级尾延迟。这种 busy-poll 是低延迟检索引擎的常见取舍。

拿到 completion 后，`status != 0` 先调 `report_direct_path_failure`（4.1）。注意这里 **不 return**——即使 GPUNetIO 失败，还要继续走下面的流程把 pending 摘掉、归还 slot、set_exception，否则 slot 永不归还，资源泄漏。

摘 pending：

```cpp
    std::shared_ptr<PendingQuery> pending;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      const auto it = pending_queries.find(completion.request_id);
      if (it != pending_queries.end()) {
        pending = std::move(it->second);
        pending_queries.erase(it);
      }
    }
    if (!pending) {
      if (completion.query_slot < query_slots) {
        active_query_tickets[completion.query_slot].store(0, std::memory_order_release);
        active_query_snapshots[completion.query_slot].store(0, std::memory_order_release);
      }
      active_gpu_queries.fetch_sub(1, std::memory_order_release);
      maintenance_cv.notify_all();
      continue;
    }
```

用 `completion.request_id` 在 `pending_queries` 表里查。如果找不到（`!pending`），说明这个 query 已经被 `reject_submission` 或 `reject_all_pending` 摘走了（停机路径）。此时仍要清 slot 的 ticket/snapshot、减 `active_gpu_queries`、通知 maintenance——否则 reclaim barrier 永远等不到这个 slot 空闲。注意 `completion.query_slot < query_slots` 的防御：恶意/损坏的 completion 可能带越界 slot。

找到了就进入正常完成路径。先算时延：

```cpp
    const auto completed_at = std::chrono::steady_clock::now();
    const u64 gpu_ns = completion.gpu_cycles * 1000000ULL / gpu_clock_khz;
    const auto phase_ns = [&](u64 cycles) {
      return cycles * 1000000ULL / gpu_clock_khz;
    };
    const u64 end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        completed_at - pending->submitted_at).count());
```

`gpu_clock_khz`（`impl.hh:229`）是启动时量得的 GPU 时钟频率（kHz）。`completion.gpu_cycles` 是 kernel 用 GPU 时钟计数器量得的总 cycle 数，乘 1e6 除以 kHz 得 ns（`cycle / (kHz * 1000) * 1e9 = cycle * 1e6 / kHz`）。`phase_ns` lambda 对各阶段 cycle 做同样换算。`end_to_end_ns` 是 CPU steady_clock 量的"search() 入队到 completion"总时延，含 CPU 排队 + GPU 计算 + ring 往返。

慢查询日志（采样）：

```cpp
    if (end_to_end_ns >= 10000000ULL &&
        slow_query_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
      std::cerr << "[gpu-search] slow query e2e_us=" << end_to_end_ns / 1000
                << " gpu_us=" << gpu_ns / 1000
                << " prepare_us=" << completion.prepare_cycles * 1000ULL / gpu_clock_khz
                << " graph_us=" << completion.graph_cycles * 1000ULL / gpu_clock_khz
                << " score_us=" << completion.score_cycles * 1000ULL / gpu_clock_khz
                << " beam_us=" << completion.beam_cycles * 1000ULL / gpu_clock_khz
                << " exact_us=" << completion.exact_cycles * 1000ULL / gpu_clock_khz
                << " delta_scan_us=" << completion.delta_scan_cycles * 1000ULL / gpu_clock_khz
                << " delta_scan_records=" << completion.delta_scan_records
                << " delta_scan_scored=" << completion.delta_scan_scored
                << " delta_scan_truncated_buckets=" << completion.delta_scan_truncated_buckets
                << " graph_reads=" << completion.remote_pages
                << " graph_rereads=" << completion.graph_read_retries
                << " graph_batches=" << completion.remote_batches
                << " graph_rounds=" << completion.graph_rounds
                << " graph_hits=" << completion.cache_hits
                << " route_hits=" << completion.route_hits
                << " exact_reads=" << completion.exact_vectors
                << " exact_hits=" << completion.exact_cache_hits << '\n';
    }
```

阈值 10ms（`10000000ULL` ns），且用 `slow_query_logs` atomic 限流到前 16 条——避免启动期或抖动期刷屏。每条慢查询日志含完整的阶段分解（prepare/graph/score/beam/exact/delta_scan 各自 us）和 RDMA 统计（reads/rereads/batches/rounds/hits）。这些字段对应 `CompletionDescriptor`（`types.hh:29-53`）。阶段语义见第 18-20 课（评分、cache、遍历主循环）。

结果组装：

```cpp
    try {
      if (completion.status != 0) {
        const std::string message = "persistent GPU query failed with status " +
          std::to_string(completion.status);
        mark_unhealthy(message);
        throw std::runtime_error(message);
      }
      const size_t offset = static_cast<size_t>(pending->slot) * result_capacity;
      service::QueryResult result;
      result.reserve(completion.result_count);
      for (u32 index = 0; index < completion.result_count; ++index) {
        result.push_back({result_ids_host[offset + index],
                          result_distances_host[offset + index]});
      }
      pending->promise.set_value(std::move(result));
    } catch (...) {
      pending->promise.set_exception(std::current_exception());
    }
```

- **status != 0**：`mark_unhealthy` + 抛异常 → catch 里 `set_exception`。这是 fail-stop 的另一触发点——kernel 内部检测到无法恢复的错误（除 GPUNetIO 直读失败外，比如 delta 校验失败、图损坏等）会回非零 status。
- **status == 0**：从 `result_ids_host`/`result_distances_host`（pinned host 缓冲，GPU 端 `d_result_ids`/`d_result_distances` 直接写）按 slot 偏移拷出 `result_count` 条 (id, distance) 对。`service::QueryResult` 是 `vector<pair<node_t, float>>`（见第 27 课计算服务主体）。
- try/catch 兜 promise 异常：任何异常都通过 `set_exception` 传给 `search()` 的 `future.get()`，避免 promise 析构时未设值导致 `broken_promise`。

归还 slot 与遥测：

```cpp
    {
      active_query_tickets[pending->slot].store(0, std::memory_order_release);
      active_query_snapshots[pending->slot].store(0, std::memory_order_release);
      std::lock_guard<std::mutex> lock(slot_mutex);
      free_slots.push_back(pending->slot);
    }
    slot_cv.notify_one();
    pending_count.fetch_sub(1, std::memory_order_release);
    active_gpu_queries.fetch_sub(1, std::memory_order_release);
    maintenance_cv.notify_all();
```

四步归还：

1. 清 slot 的 ticket/snapshot（0 = 空闲），让 reclaim barrier 能通过。
2. 持 `slot_mutex` 把 slot push 回 `free_slots`。
3. `slot_cv.notify_one()` 唤醒一个等 slot 的 `search()`。
4. `pending_count--`（停机 drain 计数）、`active_gpu_queries--`、`maintenance_cv.notify_all()`（唤醒可能等 barrier 的 maintenance 线程）。

注意 `active_query_tickets/snapshots` 的清零在 `slot_mutex` 外做——它们是 atomic，不需要锁保护，但 slot 的归还必须在锁内和 `free_slots` 一起做，避免 `search()` 拿到 slot 后看到旧 ticket。这里有个微妙的顺序：先清 ticket/snapshot，再 push slot，保证 `search()` 拿到 slot 时 ticket 已经是 0。

最后是庞大的遥测累加块（`completion.cc:128-181`）：

```cpp
    engine.telemetry_.queries_completed.fetch_add(1, std::memory_order_relaxed);
    engine.telemetry_.gpu_active_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
    engine.telemetry_.gpu_prepare_ns.fetch_add(phase_ns(completion.prepare_cycles), ...);
    engine.telemetry_.gpu_graph_ns.fetch_add(phase_ns(completion.graph_cycles), ...);
    engine.telemetry_.gpu_score_ns.fetch_add(phase_ns(completion.score_cycles), ...);
    engine.telemetry_.gpu_beam_ns.fetch_add(phase_ns(completion.beam_cycles), ...);
    engine.telemetry_.gpu_exact_ns.fetch_add(phase_ns(completion.exact_cycles), ...);
    engine.telemetry_.gpu_delta_scan_ns.fetch_add(phase_ns(completion.delta_scan_cycles), ...);
    engine.telemetry_.completion_wait_ns.fetch_add(end_to_end_ns, ...);
    if (completion.snapshot_epoch != 0) {
      engine.telemetry_.delta_queries.fetch_add(1, ...);
    }
    engine.telemetry_.delta_scan_records.fetch_add(completion.delta_scan_records, ...);
    engine.telemetry_.delta_scan_scored.fetch_add(completion.delta_scan_scored, ...);
    engine.telemetry_.delta_scan_truncated_buckets.fetch_add(completion.delta_scan_truncated_buckets, ...);
    const u64 physical_graph_reads =
      static_cast<u64>(completion.remote_pages) + completion.graph_read_retries;
    engine.telemetry_.rdma_read_ops.fetch_add(
      static_cast<u64>(completion.exact_vectors) + physical_graph_reads, ...);
    engine.telemetry_.rdma_read_bytes.fetch_add(
      static_cast<u64>(completion.exact_vectors) * node_record_bytes +
      physical_graph_reads * index.layout.graph_entry_bytes, ...);
    if (physical_graph_reads > completion.remote_batches) {
      engine.telemetry_.rdma_merged_requests.fetch_add(
        physical_graph_reads - completion.remote_batches, ...);
    }
    engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors, ...);
    engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages, ...);
    engine.telemetry_.graph_read_retries.fetch_add(completion.graph_read_retries, ...);
    engine.telemetry_.graph_dependency_rounds.fetch_add(completion.graph_rounds, ...);
    engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits, ...);
    engine.telemetry_.graph_route_hits.fetch_add(completion.route_hits, ...);
    engine.telemetry_.exact_vector_cache_hits.fetch_add(completion.exact_cache_hits, ...);
  }
}
```

几个值得点的细节：

- **`snapshot_epoch != 0` → `delta_queries++`**：epoch 0 是"无 delta"哨兵（`search()` 里 `published_epoch()` 在没有任何 delta 发布时返回 0），非零说明这个 query 看到了 delta 快照，计入 delta query 数。
- **`physical_graph_reads = remote_pages + graph_read_retries`**：物理 RDMA 读次数 = 成功页数 + 重试次数。重试是 cache 一致性协议的一部分（第 19 课 RDMA cache，读到撕裂数据会重读）。
- **`rdma_merged_requests`**：`physical_graph_reads - remote_batches`。一次 batch 可能含多个页，所以物理读数 > batch 数时，差额就是合并掉的请求数——衡量 batch 效率。
- **`rdma_read_bytes`**：exact vector 按 `node_record_bytes` 算，graph 按 `graph_entry_bytes` 算，分开统计因为两者记录大小不同。

这些遥测字段对应 `TelemetrySnapshot`（`types.hh:140+`，见第 9 课），是第 30 课 breakdown benchmark 的数据源。

### 4.3 与第 9 课 CompletionDescriptor、第 11 课 health 的关系

- **第 9 课 CompletionDescriptor**：`completion.cc` 消费的每个字段都在 `types.hh:29-53` 定义，`static_assert(sizeof(CompletionDescriptor)==128)` 保证和 kernel 端布局一致。第 9 课讲字段语义和 GPU 端如何填写，本课讲 CPU 端如何消费。
- **第 11 课 health**：`completion.cc` 的 `mark_unhealthy` 调用（`completion.cc:33`、`104`）触发 `health.cc:41` 的 fail-stop 流程——翻 `healthy=false`、swap 走 `admission_queue` 并 reject、通知所有 CV。completion 线程自己不退出（`shutdown || pending_count!=0` 的循环条件让它继续处理在飞 query），但新 query 进不来。停机顺序见 1.4。
- **fail-stop 的三个触发点**：(1) admission 失败（`query_execution.cc:146/156`）、(2) GPUNetIO 直读失败（`completion.cc:33`）、(3) kernel 返回非零 status（`completion.cc:104`）。三者殊途同归到 `mark_unhealthy`，体现"不局部恢复，停机人工介入"的一致性策略。

---

## 五、端到端时序图

下面画出一条 query 从 `search()` 入口到 `future.get()` 返回的完整时序，标注 CPU/GPU 分界和各阶段计时点。

```
RPC worker 线程            admission 线程            GPU (persistent kernel)        completion 线程
=================          ================          ========================        ================

search()
  ├─ check accepting/healthy
  ├─ slot_cv.wait ──┐
  │                 │ (slot 池空时阻塞)
  ├─ get slot ◀─────┘
  ├─ memcpy query → query_input_host[slot]            ┌── (mapped memory, GPU 可直接读)
  ├─ pending_queries.emplace(request_id)              │
  ├─ admission_queue.push_back(descriptor)            │
  ├─ admission_cv.notify_one() ──┐                    │
  └─ future.get() (阻塞)          │                    │
                                  ▼                    │
                          admission_loop()              │
                          ├─ batch.pop_back(count)     │
                          ├─ snapshot_lock:            │
                          │   snapshot_epoch =         │
                          │     published_epoch()  ◀───┘ (第 10/15 课 delta 发布器)
                          │   active_query_snapshots[slot] = epoch+1
                          │   active_query_tickets[slot] = ticket
                          ├─ submissions.try_push(descriptor) ──┐
                          │   (host→device ring, 无锁)          │
                          ├─ telemetry.batches++                │
                          │                                     ▼
                          │                          dispatcher CTA
                          │                          ├─ try_pop(submissions)
                          │                          ├─ 分配给 query CTA
                          │                          │   ├─ prepare_cycles (PQ LUT)
                          │                          │   ├─ graph_cycles  (图遍历+RDMA)
                          │                          │   │   ├─ remote_pages × RDMA read (第 22 课)
                          │                          │   │   ├─ cache_hits (第 19 课)
                          │                          │   │   └─ route_hits (第 6/10 课)
                          │                          │   ├─ score_cycles (第 18 课)
                          │                          │   ├─ beam_cycles
                          │                          │   ├─ exact_cycles (精排+RDMA)
                          │                          │   └─ delta_scan_cycles (第 20 课)
                          │                          ├─ 写 d_result_ids[slot]
                          │                          ├─ 写 CompletionDescriptor
                          │                          └─ completions.try_push ◀──── (device→host ring) ──┐
                          │                                                                     │
                          │                                                                     ▼
                          │                                                          completion_loop()
                          │                                                          ├─ try_pop(completions)
                          │                                                          ├─ status!=0? report_direct_path_failure()
                          │                                                          ├─ pending_queries.find(request_id)
                          │                                                          ├─ 计算 gpu_ns / phase_ns / end_to_end_ns
                          │                                                          ├─ (慢查询日志)
                          │                                                          ├─ 组装 QueryResult (从 result_ids_host)
                          │                                                          ├─ pending->promise.set_value(result)
                          │                                                          ├─ clear active_query_tickets[slot]
                          │                                                          ├─ free_slots.push_back(slot)
                          │                                                          ├─ slot_cv.notify_one()
                          │                                                          ├─ pending_count--
                          │                                                          ├─ active_gpu_queries--
                          │                                                          ├─ maintenance_cv.notify_all()
                          │                                                          └─ telemetry 累加 (各阶段 ns/reads/hits)
                          │                                                                     │
  ◀───────────────────────────────────────────────────────────────────────────────────────────────── future.get() 返回 result
```

### 计时点标注

| 计时点 | 量取位置 | 含义 |
|--------|----------|------|
| `submitted_at` | `query_execution.cc:36`（search 入队前） | query 进入系统的时间基准 |
| `enqueued_at` | `query_execution.cc:67`（= submitted_at） | 入 admission_queue 时刻 |
| `admitted_at` | `query_execution.cc:106`（admission 唤醒后） | admission 开始处理本批时刻 |
| `submission_wait_ns` | `query_execution.cc:130-131`（`admitted_at - enqueued_at`） | 在 admission_queue 排队等待 |
| `gpu_cycles` | kernel 端 GPU clock counter | GPU 总计算 cycle |
| `prepare/graph/score/beam/exact/delta_scan_cycles` | kernel 端各阶段 cycle counter | GPU 各阶段耗时（第 18-20 课） |
| `completed_at` | `completion.cc:68`（completion 摘到 pending 后） | completion 处理时刻 |
| `end_to_end_ns` | `completion.cc:73-75`（`completed_at - submitted_at`） | 端到端时延（含 CPU 排队 + GPU + ring） |
| `gpu_ns` | `completion.cc:69`（`gpu_cycles * 1e6 / gpu_clock_khz`） | GPU 实际计算时间 |

### CPU/GPU 分界

- **CPU 侧**：`search()` → admission_queue → admission_loop → `submissions.try_push`（host 端写）。这一段是纯 CPU，含 mutex/CV 协调、memcpy、无锁 ring push。
- **跨界（zero-copy）**：`submissions` 是 `cudaHostAllocMapped`，CPU 写 host 指针 = GPU 读 device 指针，无显式 copy。`completions` 反向同理。
- **GPU 侧**：dispatcher CTA → query CTA（prepare/graph/score/beam/exact/delta_scan）→ 写 `d_result_ids`/`CompletionDescriptor` → `completions.try_push`（device 端写）。这一段是 persistent kernel 内部，见第 17-21 课。
- **CPU 侧**：`completion_loop` → `try_pop`（host 端读）→ 组装 result → set_value → 归还 slot → 遥测。

整条路径只有两次真正的数据拷贝：(1) `search()` 把 query memcpy 到 `query_input_host`（用户缓冲 → pinned mapped）、(2) `completion_loop` 把结果从 `result_ids_host` 拷进 `QueryResult` vector（pinned → 用户缓冲）。中间所有 CPU↔GPU 通信都走 mapped memory zero-copy。

---

## 六、与其他模块的关系

- **第 9 课（GPU 类型/遥测/PQ 模型）**：`CompletionDescriptor`/`QueryDescriptor` 字段语义、`TelemetrySnapshot` 累加项、`gpu_clock_khz` 量取。本课消费这些定义。
- **第 10 课（delta/动态路由/预算）**：`engine.delta_.published_epoch()`/`reserve_epoch()`/`publish_barrier()`、`DynamicRouteOverlayDiff`、`active_query_snapshots`/`active_query_tickets` 的 RCU 语义。本课展示调用点。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`Impl` 结构、`accepting`/`healthy`/`shutdown` 三态机、`~Impl` 的停机顺序（maintenance → admission → drain → completion）、`mark_unhealthy` 的 fail-stop 语义。本课是这三态机的具体使用者。
- **第 15 课（增量发布）**：`submit_delta_publication`、`DeltaPublishDescriptor.dynamic_route_count`（route-only command）、`publish_barrier` 的可见性闸门。本课 `synchronize_storage_routes` 是其调用方。
- **第 16 课（存储回收 RCU）**：`active_query_tickets` 作为 ticket barrier、`enqueue_storage_reclaim_barriers`/`publish_ready_storage_reclaim_acks`（`storage_reclaim.cc:309/328`）依赖 completion 清 ticket。本课 completion 的 `active_query_tickets[slot].store(0)` 是 barrier 推进的必要条件。
- **第 17 课（kernel 启动器/上下文/device ring）**：`MappedRing` 的 host/device 双视图、`DeviceRingView`、persistent kernel 启动/就绪握手。本课 `submissions`/`completions` 两条 ring 是其消费方。
- **第 18 课（候选评分）**：`CompletionDescriptor.score_cycles`/`exact_cycles`。本课只统计，评分逻辑在第 18 课。
- **第 19 课（RDMA cache）**：`completion.cache_hits`/`graph_read_retries`/`remote_pages`/`remote_batches`。本课统计物理读与合并，cache 一致性协议在第 19 课。
- **第 20 课（查询遍历主循环）**：`prepare/graph/beam/delta_scan` 各阶段在 kernel 内的具体行为。本课从 CPU 侧看这些 cycle 的来源。
- **第 22 课（GPUNetIO 传输/probe）**：`direct_disabled_device`/`direct_error_device`、`report_direct_path_failure` 的 GPUNetIO 失败语义。
- **第 30 课（breakdown benchmark）**：本课累加的所有 `telemetry_` 字段是 benchmark 报告的数据源。

---

## 七、小结

本课讲解了 dvstor GPU 检索引擎的 CPU 侧三段流水线：

1. **`query_execution.cc`** — `search()` 在 RPC worker 线程领 slot、拷 query、入 `admission_queue`、阻塞 `future.get()`；`admission_loop()` 在专用线程摘 batch、在 `query_snapshot_mutex` 内绑定 `snapshot_epoch` + 分配 ticket、逐个 `try_push` 到 host→device ring（不凑批，立即发布），失败走 `reject_submission` + `mark_unhealthy`。关键设计：mapped memory zero-copy、snapshot_epoch 在 push 前刷新、ticket 用于 RCU barrier。

2. **`routing.cc`** — delta mutation 解码（`decode_mutation_payload`）、anchor 最近邻（`nearest_anchor`，为 delta bucket 定位）、graph cache key 换算（`graph_cache_key(s)`，把存储侧 raw node 偏移换算成 cache key）、anchor graph RDMA 刷新（`refresh_anchor_graph_records`，RCU 等读者退出 + RDMA 重读 + checksum16 校验 + 发布 ready 状态）。配合 `storage_reclaim.cc` 的 `read_storage_route_publications`/`synchronize_storage_routes`（从 4 KiB control page offset 1024 三次 RDMA + seqlock + checksum 校验拉路由快照，diff 后发 route-only delta command，`publish_barrier` 保证 PQ code 与 seqlock 双可见性）完成动态路由刷新。

3. **`completion.cc`** — `completion_loop()` 轮询 device→host completion ring，摘 `CompletionDescriptor`、（失败时）`report_direct_path_failure`、摘 pending、算各阶段时延、（慢查询）采样日志、组装 `QueryResult`、`set_value`/`set_exception`、清 ticket/snapshot、归还 slot、累加全部遥测。`status != 0` 是 fail-stop 的 kernel 侧触发点。

三者通过 `MappedRing`（zero-copy host↔device）、`pending_queries`（request_id → promise）、`active_query_tickets/snapshots`（per-slot RCU 状态）、`free_slots`（slot 池）四个数据结构耦合，由 `accepting`/`healthy`/`shutdown` 三态机统一管控生命周期。整条路径只有两次真拷贝（query 入、result 出），中间全程 zero-copy，是低延迟 GPU 检索的关键。fail-stop 在 admission 失败、GPUNetIO 失败、kernel status 非零三个触发点统一到 `mark_unhealthy`，体现"不局部恢复"的一致性策略。
