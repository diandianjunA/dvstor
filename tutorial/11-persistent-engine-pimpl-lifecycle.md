# 第 11 课：持久化引擎 PImpl 与生命周期

> 本课是 Part III「GPU 搜索引擎」的入口课。我们将拆开 `PersistentSearchEngine` 这扇对外只露一个 `search()`/`publish_mutations()` 的薄壳，看清楚 PImpl 把"一片 GPU 卡上几十个 device 指针、4 条 CUDA stream、3 条 host 线程、1 个常驻 grid"全部藏到 `Impl` 里的真实做法；再逐行解读引擎从"构造 → 启动 → 稳态 → shutdown"的状态机，以及 `health.cc` 的 fail-stop 模型。后续第 12–16 课都建立在本课给出的资源归属与线程结构之上。

## 11.1 本课目标与涉及文件

本课回答四个问题：

1. **PImpl 边界在哪？** 哪些状态由 `PersistentSearchEngine` 自己持有，哪些被下沉进 `Impl`？
2. **引擎怎么被"装配"出来？** `Impl::Impl` 的 1000 多行构造函数按什么顺序把 GPU 卡变成一个能吃查询的常驻引擎？
3. **线程与资源谁负责停？** 三条 host 线程、四条 CUDA stream、上百个 device buffer、几十个 pinned host buffer 的回收顺序是怎样的？
4. **失败如何短路？** GPUNetIO 读错、kernel 起不来、delta 发布失败之后，引擎怎么让"已经在 admission 队列里"和"还没到"的查询都立刻失败？

涉及文件（全部在 `src/gpu_search/persistent_engine/` 下，根路径 `/home/xjs/experiment/dvstor`）：

| 文件 | 行数 | 角色 |
|---|---|---|
| `persistent_engine.hh` | 59 | 公开 PImpl 头：`PersistentSearchEngine` 类与 `MutationCapacityError` |
| `persistent_engine.cc` | 277 | 公开方法的转发实现；mutation 发布协议在 `publish_mutations` |
| `impl.hh` | 416 | `PersistentSearchEngine::Impl` 完整成员声明 |
| `lifecycle.cc` | 468 | 启动后期的 bootstrap 流、kernel 起/停、`~Impl()` 析构序列 |
| `construction.cc` | 1021 | `Impl::Impl` 主构造函数（真正的装配入口） |
| `health.cc` | 114 | fail-stop：`mark_unhealthy`/`reject_submission`/`reject_all_pending` |
| `cuda_helpers.hh` | 77 | `check_cuda`/`device_allocate`/`mapped_host_allocate` 等基础设施 |
| `query_execution.cc` | 160 | `search()` 与 `admission_loop()` —— 稳态主回路入口 |
| `completion.cc` | 184 | `completion_loop()` 与 `report_direct_path_failure()` |

> 注意：任务大纲里把"启动序列"挂在 `lifecycle.cc` 名下，但实际启动序列最关键的 `Impl::Impl` 函数体落在 `construction.cc`，而 `lifecycle.cc` 承载的是"启动后期的 RDMA bootstrap 流、kernel 起/停、析构"。本课按真实代码归属讲解：§11.5 讲 `construction.cc` 的 `Impl::Impl`，§11.6 讲 `lifecycle.cc` 的 `start_persistent_kernel`/`stop_persistent_kernel`/`~Impl`，§11.4 讲 `lifecycle.cc` 头部的两个 bootstrap 函数。

## 11.2 公开门面：`persistent_engine.hh`

整个 GPU 引擎对外的头文件只有 59 行，先看全貌（`src/gpu_search/persistent_engine.hh:18-59`）：

```cpp
namespace gpu_search {

class MutationCapacityError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class PersistentSearchEngine {
public:
  PersistentSearchEngine(configuration::IndexConfiguration& config,
                         Context& channel_context,
                         ClientConnectionManager& connection_manager,
                         const MemoryRegionTokens& remote_regions);
  ~PersistentSearchEngine();

  PersistentSearchEngine(const PersistentSearchEngine&) = delete;
  PersistentSearchEngine& operator=(const PersistentSearchEngine&) = delete;

  service::QueryResult search(VectorDType query_dtype, const byte_t* query_data, u32 k);
  service::QueryResult search(std::span<const element_t> query, u32 k);

  bool publish_mutations(
    std::span<DeltaMutation> mutations,
    std::span<const u64> invalidated_graph_nodes = {});
  bool try_reserve_mutation_capacity(size_t mutation_count);
  void reserve_mutation_capacity(size_t mutation_count);
  void release_mutation_capacity(size_t mutation_count);
  void mark_committed_mutation_gap(const std::string& reason);
  DeltaCoordinator& delta() { return delta_; }
  const DeltaCoordinator& delta() const { return delta_; }
  TelemetrySnapshot telemetry() const { return telemetry_.snapshot(); }
  void reset_telemetry();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DeltaCoordinator delta_;
  Telemetry telemetry_;
  std::mutex mutation_publish_mutex_;
};

}  // namespace gpu_search
```

这是教科书式的 PImpl：`Impl` 只前置声明（`struct Impl;`），定义在 `impl.hh` 里。外头编译单元只要包含 `persistent_engine.hh`，就不需要拉 CUDA 头、不需要看到那 100+ 个 `d_*` device 指针，因而修改 `Impl` 内部布局不会触发上层重编。需要重点理解的是 `PersistentSearchEngine` 自己保留的四个成员（`persistent_engine.hh:53-56`）——它们没有下沉到 `Impl`，是有意为之：

### 11.2.1 `DeltaCoordinator delta_` 与 `Telemetry telemetry_`：跨线程共享所有权

`DeltaCoordinator`（见第 10 课）维护已发布 epoch、`node_t → generation` 的版本表、retired 队列。`Telemetry`（见第 9 课）是一堆 `std::atomic` 计数器。这两块状态的生命周期横跨"宿主类 + Impl"两侧：

- 宿主类直接持有，使 `delta()`/`telemetry()` 这两个内联 getter 不必经 `Impl` 间接访问；
- `Impl` 通过 `engine.delta_` 与 `engine.telemetry_`（见 `impl.hh:162` 的 `PersistentSearchEngine& engine;`）反向引用，使得 admission/completion/maintenance 三条线程在记录遥测、推进 epoch 时无需跨 PImpl 边界加锁。

这样做的代价是 `DeltaCoordinator` 与 `Telemetry` 的析构顺序必须在 `Impl` 之前。`persistent_engine.cc:30-32` 的析构函数显式 `impl_.reset();`，正是为了让 `Impl` 先于 `delta_`/`telemetry_` 析构——三条线程先 join、device buffer 先释放，再让 coordinator/telemetry 走默认析构。

### 11.2.2 `std::mutex mutation_publish_mutex_`：mutation 发布与 route 维护的串行化点

这个互斥锁的存在把"mutation 发布"与"route-only 维护"放在同一临界区外。`persistent_engine.cc:44-47` 注释写得很清楚：

```cpp
  std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
  if (mutations.empty()) {
    throw std::invalid_argument("GPU mutation publication requires a non-empty epoch batch");
  }
  // Epoch reservation and publication share this mutex with route-only
  // maintenance commands.  Therefore a later route barrier can never overtake
  // an earlier mutation whose GPU records have not been committed yet.
```

这把锁解决了一个具体的时序问题：维护线程（见第 16 课 storage_reclaim）会执行 route-only 命令，它的"路由 barrier"必须不能跨过一个还没把 GPU records 提交完的 mutation。`mutation_publish_mutex_` 落在宿主类而不是 `Impl` 里，是因为维护线程和发布 RPC 入口都要抢它，把它放在 PImpl 外面更直观。

### 11.2.3 `MutationCapacityError`：可恢复的容量拒绝

`persistent_engine.hh:20-23` 单独定义一个异常类继承 `std::runtime_error`，目的是让上游 RPC 能区分"容量不够"（可重试）和"引擎真坏了"（不可恢复）。`persistent_engine.cc:98-104` 显示这种区分是在 `publish_mutations` 的 catch 链里完成的：

```cpp
  try {
    graph_cache_invalidations =
      impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
  } catch (const MutationCapacityError&) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    throw;
  } catch (const std::exception& error) {
    impl_->mark_unhealthy(std::string{"GPU mutation publication failed: "} + error.what());
    throw;
  }
```

`MutationCapacityError` 被原样 rethrow（仅记一个拒绝计数），其它 `std::exception` 一律 `mark_unhealthy`——把引擎切到 fail-stop。fail-stop 模型见 §11.7。

### 11.2.4 公开 API 的语义分组

把头文件里 9 个公开成员函数按语义分组：

| 组 | 成员 | 说明 |
|---|---|---|
| 查询 | `search()` × 2 重载 | 同步阻塞返回 `QueryResult`；内部走 `Impl::search` → admission → kernel → completion（见第 14/20 课） |
| 发布 | `publish_mutations()` | 把一批 `DeltaMutation` 写到 GPU delta tier 并 publish epoch；见 §11.3 |
| 容量预约 | `try_reserve_mutation_capacity`/`reserve_mutation_capacity`/`release_mutation_capacity` | 上游 RPC 在收到 mutation batch 之前先预约 delta 容量；§11.3.2 |
| 故障通知 | `mark_committed_mutation_gap` | 存储侧报告"已 commit 但 GPU 不可见"的 mutation，立刻 fail-stop |
| 访问器 | `delta()` × 2 / `telemetry()` / `reset_telemetry()` | 给 RPC 层、benchmark 工具读状态用 |

`search()` 的两个重载里，`std::span<const element_t>` 版本只是把 `float32` 路径归一到字节版（`persistent_engine.cc:39-42`）：

```cpp
service::QueryResult PersistentSearchEngine::search(std::span<const element_t> query, u32 k) {
  return search(VectorDType::float32,
                reinterpret_cast<const byte_t*>(query.data()), k);
}
```

真正的查询入口在 `Impl::search`（`query_execution.cc:7`，见第 14 课）。`PersistentSearchEngine::search` 的字节版只是 `return impl_->search(...)`（`persistent_engine.cc:34-37`）——纯转发。

## 11.3 公开方法的转发实现：`persistent_engine.cc`

### 11.3.1 构造/析构转发

`persistent_engine.cc:18-32`：

```cpp
PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions)
    : delta_() {
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU navigation engine)");
  impl_ = std::make_unique<Impl>(*this, config, channel_context,
                                 connection_manager, remote_regions);
}

PersistentSearchEngine::~PersistentSearchEngine() {
  impl_.reset();
}
```

注意构造函数初始化列表只写了 `delta_()`（`DeltaCoordinator` 默认构造，`published_epoch_` 从 0 起）。`telemetry_` 与 `mutation_publish_mutex_` 都走默认构造。这里有一个**关键细节**：在 `make_unique<Impl>` 之前先 `cudaSetDevice(config.gpu_device)`。这是 PImpl 装配的入口定位——`Impl::Impl` 内部所有 `cudaMalloc`/`cudaHostAlloc` 都依赖当前线程的 device binding，而 `Impl::Impl` 自己在 `construction.cc:85` 又调了一次 `bind_cuda_device("cudaSetDevice(GPU navigation construction)")`（§11.5 会讲）。宿主构造里这次 `cudaSetDevice` 是为了在 `make_unique` 抛异常时也能给出一个清晰的 CUDA 错误，而不是把 device 不匹配的错误埋进 `Impl::Impl` 的某次 `cudaMalloc`。

析构只有一行 `impl_.reset();`：`unique_ptr` 在析构点触发 `Impl::~Impl()`（`lifecycle.cc:322`，§11.6.3 详解）。`delta_`/`telemetry_`/`mutation_publish_mutex_` 的析构在 `impl_.reset()` 返回后按声明逆序发生——这正是我们想要的：Impl 先死、coordinator 后死。

### 11.3.2 mutation 容量预约三件套

`try_reserve_mutation_capacity`/`reserve_mutation_capacity`/`release_mutation_capacity` 是 `publish_mutations` 的"预约层"，解决一个并发问题：上游 RPC 拿到一个 mutation batch 后要先解码、校验、对 generation 排序，这些步骤可能耗时；如果直接调 `publish_mutations` 才发现 delta 槽位不够，整批都得回滚。所以 RPC 层先预约容量，再做解码，最后 `publish_mutations` 真正占用 delta 槽位，`release` 释放预约。

#### `try_reserve_mutation_capacity`：非阻塞尝试

`persistent_engine.cc:150-179`：

```cpp
bool PersistentSearchEngine::try_reserve_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return true;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  impl_->reclaim_retired_delta_slots_locked();
  const size_t active_slots = impl_->active_delta_slots_locked();
  const size_t hard_watermark = static_cast<size_t>(impl_->delta_capacity) * 9 / 10;
  const size_t active_resident_pq = impl_->active_resident_pq_slots_locked();
  const size_t resident_pq_hard_watermark =
    std::max<size_t>(1, static_cast<size_t>(impl_->resident_pq_capacity) * 95 / 100);
  if (mutation_count > hard_watermark ||
      active_slots > hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        hard_watermark - mutation_count - active_slots ||
      mutation_count > resident_pq_hard_watermark ||
      active_resident_pq > resident_pq_hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        resident_pq_hard_watermark - mutation_count - active_resident_pq) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  impl_->reserved_mutation_capacity += mutation_count;
  // ... 更新 telemetry 的 mutation_capacity_reserved / _max ...
  return true;
}
```

关键点：

- 加 `impl_->delta_mutex`，与 `publish_mutations`、维护线程的 retire/reclaim 共用一把锁。
- `reclaim_retired_delta_slots_locked()` 先回收已经退役但还没释放物理槽位的 delta（退役 barrier 由 `query_ticket_barrier_passed` 决定，见第 16 课）。
- 两条硬水位线：delta 槽位用 90%、resident PQ 槽位用 95%。每个水位检查 6 个条件，覆盖"本次请求本身超水位"、"当前活跃超水位减去本次请求"、"已有预约 + 本次请求 > 水位减去活跃"。
- 拒绝时只记 `mutation_capacity_rejections` 计数，不抛异常；接受时把 `reserved_mutation_capacity` 加上去并更新两个 telemetry gauge（当前值 + 历史最大值，后者用 CAS 循环）。

#### `reserve_mutation_capacity`：阻塞版本

`persistent_engine.cc:181-234` 是 `try_` 版本的阻塞兄弟：同样的水位检查，但失败时不返回 false，而是在 `delta_capacity_cv` 上 `wait_for(1ms)` 后重试。这里有两个遥测增量：

```cpp
telemetry_.mutation_capacity_wait_events.fetch_add(1, std::memory_order_relaxed);
// ...
telemetry_.mutation_capacity_wait_ns.fetch_add(...);
```

注释（`persistent_engine.cc:229-231`）解释了为什么 1ms 是必要的：发布线程在释放预约后会 `delta_capacity_cv.notify_all()`，但维护线程也会回收退役槽位，它不知道有谁在等，所以等待方必须主动重试。

```cpp
// Publication releases reservations and notifies directly. The bounded
// wait also rechecks capacity reclaimed by the independent maintenance
// thread, which does not need to know about submitters.
impl_->delta_capacity_cv.wait_for(lock, std::chrono::milliseconds(1));
```

#### `release_mutation_capacity`：释放与记账下溢保护

`persistent_engine.cc:236-249`：

```cpp
void PersistentSearchEngine::release_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  if (mutation_count > impl_->reserved_mutation_capacity) {
    impl_->mark_unhealthy("GPU mutation capacity reservation accounting underflow");
    impl_->reserved_mutation_capacity = 0;
  } else {
    impl_->reserved_mutation_capacity -= mutation_count;
  }
  telemetry_.mutation_capacity_reserved.store(
    static_cast<u64>(impl_->reserved_mutation_capacity),
    std::memory_order_release);
  impl_->delta_capacity_cv.notify_all();
}
```

`release > reserved` 被当作严重 bug：直接 `mark_unhealthy`，把 `reserved_mutation_capacity` 清零（不让一个坏状态继续累积），但仍然 `notify_all` 唤醒可能在等的 `reserve_mutation_capacity`。`store` 用 `memory_order_release` 是因为 `delta_capacity_cv.notify_all()` 配对的 `wait` 会 `acquire`——锁本身已经提供 happens-before，这里的 release 语义只是额外保证 telemetry 读到的值不比锁内旧。

### 11.3.3 `publish_mutations`：发布协议本体

`persistent_engine.cc:44-148` 是本课最长的方法。它做的事可以拆成 6 步：

**第 1 步：取锁、非空校验、reserve epoch**（`persistent_engine.cc:47-54`）

```cpp
std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
if (mutations.empty()) {
  throw std::invalid_argument("GPU mutation publication requires a non-empty epoch batch");
}
const u64 epoch = delta_.reserve_epoch();
```

`reserve_epoch()` 在 `DeltaCoordinator` 内部原子地分配一个新 epoch 号（第 10 课）。这个 epoch 在第 6 步 `publish_metadata` 之前不会对外可见，所以中间任何步骤失败都可以丢弃。

**第 2 步：批内去重 + generation 单调化**（`persistent_engine.cc:55-82`）

这是批内对同一 `node_t` 的多次 mutation 做合并：

```cpp
size_t accepted_count = 0;
for (size_t index = 0; index < mutations.size(); ++index) {
  DeltaMutation& candidate = mutations[index];
  const auto current = delta_.version(candidate.id);
  u32 accepted_generation = current ? current->generation : 0;
  for (size_t accepted = 0; accepted < accepted_count; ++accepted) {
    if (mutations[accepted].id == candidate.id) {
      accepted_generation = std::max(
        accepted_generation, mutations[accepted].generation);
    }
  }
  if (candidate.generation == 0) {
    candidate.generation = accepted_generation + 1;
  } else if (candidate.generation <= accepted_generation) {
    continue;
  }
  if (accepted_count != index) {
    std::swap(mutations[accepted_count], mutations[index]);
  }
  ++accepted_count;
}
mutations = mutations.first(accepted_count);
if (mutations.empty()) {
  return true;
}
```

逻辑：对每个 candidate，先取 `delta_.version(id)` 得到 coordinator 当前已知的 generation（没有就是 0），再扫描已 accept 列表里同 id 的最大 generation，取两者较大值 `accepted_generation`。

- 如果 candidate 的 `generation == 0`（调用方未指定），赋 `accepted_generation + 1`；
- 如果 candidate 的 `generation <= accepted_generation`（过时），跳过；
- 否则接受。

接受时用 `std::swap` 而不是 `remove_if`+move，注释（`persistent_engine.cc:55-57`）解释："every preallocated vector buffer remains owned by one RPC-slot element"——RPC slot 预分配了 vector buffer，move-assignment 会让原 slot 失去 buffer 所有权，swap 则保留所有权交换。

**第 3 步：测量 publication queue 时延**（`persistent_engine.cc:83-93`）

```cpp
const auto publication_started = std::chrono::steady_clock::now();
u64 publication_queue_ns = 0;
for (const DeltaMutation& mutation : mutations) {
  if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
  publication_queue_ns += static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      publication_started - mutation.enqueued_at).count());
}
telemetry_.publication_queue_ns_total.fetch_add(publication_queue_ns,
                                                std::memory_order_relaxed);
```

每个 mutation 的 `enqueued_at` 是它在 RPC 入口被排进队列的时刻；到 `publish_mutations` 开始执行之间的等待被累加进 `publication_queue_ns_total`。`enqueued_at` 为默认值（time_point{}）表示调用方没填，跳过。

**第 4 步：上传到 GPU delta tier**（`persistent_engine.cc:94-104`）

```cpp
size_t graph_cache_invalidations = 0;
try {
  graph_cache_invalidations =
    impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
} catch (const MutationCapacityError&) {
  telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
  throw;
} catch (const std::exception& error) {
  impl_->mark_unhealthy(std::string{"GPU mutation publication failed: "} + error.what());
  throw;
}
```

`upload_mutations` 见第 15 课（delta_publication.cc）。它做三件事：分配 delta slot、把 vector/PQ code 写到 device、向 kernel 发 `DeltaPublishDescriptor`。`MutationCapacityError` 与其它异常的区分在 §11.2.3 已讲。

**第 5 步：测量 GPU 上传到完成的可见时延**（`persistent_engine.cc:105-117`）

```cpp
const auto gpu_upload_completed_at = std::chrono::steady_clock::now();
u64 visibility_ns_total = 0;
u64 visibility_ns_max = 0;
u64 visibility_sample_count = 0;
for (const DeltaMutation& mutation : mutations) {
  if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
  const u64 visibility_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      gpu_upload_completed_at - mutation.enqueued_at).count());
  visibility_ns_total += visibility_ns;
  visibility_ns_max = std::max(visibility_ns_max, visibility_ns);
  ++visibility_sample_count;
}
```

注意 `visibility_ns` 是从 RPC 入口到 GPU 上传完成的总时间，不是只到 `upload_mutations` 返回——后面还要加 coordinator publish 时间。

**第 6 步：coordinator publish + 更新 telemetry**（`persistent_engine.cc:118-147`）

```cpp
try {
  if (!delta_.publish_metadata(mutations, epoch)) {
    impl_->mark_unhealthy("GPU mutation publication lost its coordinator epoch");
    return false;
  }
} catch (const std::exception& error) {
  impl_->mark_unhealthy(std::string{"GPU epoch publication failed: "} + error.what());
  throw;
}
// Queries cannot select this epoch until the coordinator publish above.
// Include that final handoff in the stage1-response-to-visible SLO.
const u64 coordinator_publish_ns = static_cast<u64>(
  std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - gpu_upload_completed_at).count());
visibility_ns_total += coordinator_publish_ns * visibility_sample_count;
if (visibility_sample_count != 0) {
  visibility_ns_max += coordinator_publish_ns;
}
telemetry_.mutations_published.fetch_add(mutation_count, std::memory_order_relaxed);
telemetry_.delta_publications.fetch_add(1, std::memory_order_relaxed);
telemetry_.graph_cache_invalidations.fetch_add(
  graph_cache_invalidations, std::memory_order_relaxed);
telemetry_.visibility_ns_total.fetch_add(visibility_ns_total,
                                         std::memory_order_relaxed);
telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
u64 current_max = telemetry_.visibility_ns_max.load(std::memory_order_relaxed);
while (current_max < visibility_ns_max &&
       !telemetry_.visibility_ns_max.compare_exchange_weak(
         current_max, visibility_ns_max, std::memory_order_relaxed)) {}
return true;
```

`publish_metadata` 失败（返回 false）的语义是"coordinator 的 epoch 被别人抢走了"——本批次的 epoch 已经无效。这种情况是 fatal，直接 `mark_unhealthy`。

注释 `// Queries cannot select this epoch until the coordinator publish above.` 解释了为什么 `coordinator_publish_ns` 要加到 visibility 时延里：在 `publish_metadata` 之前，新 epoch 不会出现在 `published_epoch()` 里，admission 线程（`query_execution.cc:111`）拿到的 snapshot_epoch 不会是这次发布的 epoch。所以"mutation 真正对查询可见"的时点就是 `publish_metadata` 返回的那一刻。

最后一行 CAS 循环更新 `visibility_ns_max`（历史最大可见时延），是 telemetry 里少数几个需要"取最大值"语义的计数器。

### 11.3.4 `mark_committed_mutation_gap`：存储侧 → 引擎的 fail-stop 通道

`persistent_engine.cc:251-255`：

```cpp
void PersistentSearchEngine::mark_committed_mutation_gap(const std::string& reason) {
  impl_->mark_unhealthy(
    "storage committed a mutation that is not GPU-visible: " + reason);
}
```

这是存储侧（见第 28 课"计算侧 storage owner 更新"）报告"我已经 commit 了一个 mutation，但发现 GPU delta tier 里没有对应记录"时的通道。直接 `mark_unhealthy`——这种状态不可恢复，因为存储侧已经向客户端返回成功，但 GPU 查询永远看不到这条 mutation。fail-stop 是唯一安全选择。

### 11.3.5 `reset_telemetry`：带锁快照同步

`persistent_engine.cc:257-275` 把 telemetry 全部清零后，重新从 `Impl` 抓取 6 个 gauge 的当前值（`delta_physical_entries`/`delta_mutable_entries`/`delta_durable_entries`/`resident_pq_capacity`/`resident_pq_entries`/`resident_pq_peak_entries`/`mutation_capacity_reserved`）。这些值只有拿到 `delta_mutex` 才能安全读，所以函数末尾再加 `impl_->delta_mutex`。注意 `delta_live_entries` 在加锁前就 store 了——它来自 `delta_.delta_size()`，由 `DeltaCoordinator` 内部锁保护，不依赖 `Impl::delta_mutex`。

## 11.4 `lifecycle.cc` 头部：bootstrap 流与 delta 重置

`lifecycle.cc` 的前 217 行是两个 RDMA bootstrap 函数和一个 delta 设备状态重置函数。它们都由 `Impl::Impl` 在 `construction.cc` 里调用，但逻辑独立，先讲。

### 11.4.1 `stream_codes_to_gpu`：PQ code 的流式 RDMA 上传 + 抽样审计

`lifecycle.cc:7-110`。函数签名：

```cpp
void PersistentSearchEngine::Impl::stream_codes_to_gpu(NavigationBootstrapper& source) {
  const u64 window_bytes = static_cast<u64>(config.gpu_bootstrap_window_mb) << 20;
  std::vector<NavigationRead> requests;
  std::vector<i32> statuses;
  requests.reserve(config.gpu_bootstrap_windows);
  u64 streamed = 0;
  for (const format::ShardRegion& shard : index.shards) {
    for (u64 offset = 0; offset < shard.code_bytes;) {
      requests.clear();
      for (u32 window = 0; window < config.gpu_bootstrap_windows &&
           offset < shard.code_bytes; ++window) {
        const u32 bytes = static_cast<u32>(std::min<u64>(
          window_bytes, shard.code_bytes - offset));
        requests.push_back(NavigationRead{
          .remote_offset = shard.code_remote_offset + offset,
          .destination_address = reinterpret_cast<u64>(d_pq_codes +
            shard.ordinal_base * code_bytes + offset),
          .bytes = bytes,
          .memory_node = static_cast<u16>(shard.memory_node),
        });
        offset += bytes;
      }
      statuses.assign(requests.size(), -EIO);
      source.read(requests, statuses);
      for (size_t request_index = 0; request_index < statuses.size(); ++request_index) {
        if (statuses[request_index] <= 0) {
          // ... 抛 runtime_error，附 status/shard/offset/bytes/destination ...
        }
      }
      for (const NavigationRead& request : requests) streamed += request.bytes;
    }
  }
  const u64 expected = index.layout.num_nodes * code_bytes;
  if (streamed != expected) throw std::runtime_error("GPU PQ code bootstrap size mismatch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GPU PQ bootstrap)");
```

要点：

- **窗口化 RDMA**：`gpu_bootstrap_window_mb` 控制单次 `NavigationRead` 字节数；`gpu_bootstrap_windows` 控制每批并发请求数。两者一起决定 RDMA 风暴的深度。
- **destination 直接写到 final 位置**：`d_pq_codes + shard.ordinal_base * code_bytes + offset`——PQ code 不经过临时 buffer，直接 RDMA 写到 `d_remote_buffer`（GPUNetIO 区域，见 §11.5.4）。
- **status ≤ 0 视为失败**：`NavigationBootstrapper::read` 用 `i32` 状态码返回，0 是成功，负数是 `-errno`。
- **size 校验**：`streamed != expected` 直接抛错。`expected = num_nodes * code_bytes` 是全索引的 PQ code 总字节。
- **device 同步**：RDMA 写完成不代表 GPU 可见，必须 `cudaDeviceSynchronize`。

接下来是抽样审计（`lifecycle.cc:50-109`），从每个 shard 取头、中、尾三个 ordinal，单独 RDMA 读到 `d_exact_records`（exact 区，用作"权威源"），再 `cudaMemcpy` 回 host，与 `d_pq_codes` 对应位置的内容比较：

```cpp
struct AuditSample {
  u32 shard{};
  u64 slot{};
  u64 ordinal{};
};
std::vector<AuditSample> samples;
samples.reserve(index.shards.size() * 3);
for (const format::ShardRegion& shard : index.shards) {
  const std::array<u64, 3> shard_slots{0, shard.node_count / 2, shard.node_count - 1};
  for (size_t sample_index = 0; sample_index < shard_slots.size(); ++sample_index) {
    if (sample_index != 0 && shard_slots[sample_index] == shard_slots[sample_index - 1]) {
      continue;
    }
    // ...
  }
}
std::vector<byte_t> authoritative(code_bytes);
std::vector<byte_t> resident(code_bytes);
for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
  // ... 单条 RDMA 读到 d_exact_records ...
  check_cuda(cudaMemcpy(authoritative.data(), d_exact_records, authoritative.size(),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(GPU PQ audit source)");
  check_cuda(cudaMemcpy(
    resident.data(),
    d_pq_codes + sample.ordinal * code_bytes,
    resident.size(), cudaMemcpyDeviceToHost),
    "cudaMemcpy(GPU PQ audit resident)");
  if (!std::equal(resident.begin(), resident.end(), authoritative.begin())) {
    throw std::runtime_error(
      "GPU PQ ordinal mapping mismatch: shard=" + ...);
  }
}
```

这个审计是冷启动正确性的兜底：如果 `shard.ordinal_base` 算错、或 `code_remote_offset` 偏移错位，抽样会立刻发现。`shard_slots` 用 `if (sample_index != 0 && shard_slots[sample_index] == shard_slots[sample_index - 1]) continue;` 跳过重复（当 `node_count` 很小时头/中/尾可能重合）。

### 11.4.2 `stream_anchor_graph_to_gpu`：静态 fallback 路由图

`lifecycle.cc:112-174`。Anchor 图是"当动态路由没命中时的静态 fallback"，见第 6/19 课。函数先检查 `anchor_graph_keys_host.empty()`，空则打日志返回（`lifecycle.cc:113-116`）：

```cpp
if (anchor_graph_keys_host.empty()) {
  std::cerr << "[gpu-search] static fallback route graph disabled\n";
  return;
}
```

否则按 4096 一批从存储侧读图边记录：

```cpp
constexpr size_t kBootstrapBatch = 4096;
std::vector<NavigationRead> requests;
std::vector<i32> statuses;
requests.reserve(kBootstrapBatch);
for (size_t begin = 0; begin < anchor_graph_keys_host.size();
     begin += kBootstrapBatch) {
  const size_t end = std::min(begin + kBootstrapBatch,
                              anchor_graph_keys_host.size());
  requests.clear();
  for (size_t slot = begin; slot < end; ++slot) {
    const u64 key = anchor_graph_keys_host[slot];
    const u32 shard = static_cast<u32>(key >> 48);
    if (shard >= index.shards.size()) {
      throw std::runtime_error("anchor route graph key has an invalid shard");
    }
    requests.push_back(NavigationRead{
      .remote_offset = (key << 16) >> 16,
      .destination_address = reinterpret_cast<u64>(
        d_anchor_graph_records +
        slot * index.layout.graph_entry_bytes),
      .bytes = index.layout.graph_entry_bytes,
      .memory_node = static_cast<u16>(shard),
    });
  }
  statuses.assign(requests.size(), -EIO);
  source.read(requests, statuses);
  // ... status 检查 ...
}
```

`anchor_graph_keys_host[slot]` 是 64 位 key：高 16 位是 shard 编号，低 48 位是 shard 内偏移。`(key << 16) >> 16` 把高 16 位清零得到偏移。

之后是 checksum 审计（`lifecycle.cc:153-169`）：

```cpp
const size_t audit_count = std::min<size_t>(15, anchor_graph_keys_host.size());
std::vector<byte_t> record(index.layout.graph_entry_bytes);
for (size_t audit = 0; audit < audit_count; ++audit) {
  const size_t slot = audit_count == 1 ? 0 :
    audit * (anchor_graph_keys_host.size() - 1) / (audit_count - 1);
  check_cuda(cudaMemcpy(
               record.data(),
               d_anchor_graph_records + slot * index.layout.graph_entry_bytes,
               record.size(), cudaMemcpyDeviceToHost),
             "cudaMemcpy(anchor route graph audit)");
  const u16 expected = vamana::hot_graph::load_u16_le(record.data() + 2);
  const u16 actual = vamana::hot_graph::checksum16(record.data(), record.size());
  if (record[0] > index.layout.graph_degree || expected != actual) {
    throw std::runtime_error(
      "anchor route graph audit failed at slot " + std::to_string(slot));
  }
}
```

每条图边记录的第一个字节是 degree（必须 ≤ `graph_degree`），第 2-3 字节是 `checksum16`（Vamana 图格式，见第 6/7 课）。审计最多 15 条，均匀分布在 keys 上（`slot = audit * (size - 1) / (count - 1)`）。

### 11.4.3 `clear_delta_device_state`：delta tier 的"全归零"

`lifecycle.cc:176-217`。构造时和某些重置场景要把 delta tier 的 device 状态清空。它依次 `cudaMemsetAsync` 11 个 delta 相关的 device buffer：

| Buffer | 填充值 | 含义 |
|---|---|---|
| `d_delta_records` | `0` | 所有 delta record 清零 |
| `d_delta_next` / `d_delta_prev` | `0xff` | 链表 next/prev 指针 = `UINT32_MAX`（空指针哨兵） |
| `d_delta_remote_positions` | `0xff` | remote 位置 = 空 |
| `d_base_override_keys` | `0xff` | override hash 表 key = 空 |
| `d_base_override_epochs` | `0` | override epoch = 0 |
| `d_permanent_override_bits` | `0` | 永久 override 位图 = 全 0 |
| `d_delta_remote_keys` | `0` | remote key 表 = 0 |
| `d_delta_remote_slots` | `0xff` | remote slot 表 = 空 |
| `d_delta_count` | `0` | delta 计数器 = 0 |
| `d_delta_bucket_heads`（可选） | `0xff` | anchor bucket 链头 = 空 |

注意 `0xff` 填 `u32` 等于 `UINT32_MAX`，这是 kernel 侧约定的"空槽"标记。最后 `cudaStreamSynchronize(stream)` 确保所有 memset 完成才返回——`stream` 参数允许调用方传入 `delta_stream` 与其它操作并发。

## 11.5 装配入口：`Impl::Impl`（`construction.cc:73-1019`）

这是本课最长的一节。`Impl::Impl` 体长近 950 行，按顺序做 9 件事。逐段讲。

### 11.5.1 第 1 步：成员初始化列表 + 启动校验（`construction.cc:73-97`）

```cpp
PersistentSearchEngine::Impl::Impl(PersistentSearchEngine& owner,
     configuration::IndexConfiguration& config_in,
     Context& channel_context,
     ClientConnectionManager& connection_manager,
     const MemoryRegionTokens& remote_regions)
    : engine(owner), config(config_in),
      submissions(config.gpu_query_slots * 2,
                  MappedRing<QueryDescriptor>::Direction::host_to_device),
      completions(config.gpu_query_slots * 2,
                  MappedRing<CompletionDescriptor>::Direction::device_to_host),
      delta_submissions(8, MappedRing<DeltaPublishDescriptor>::Direction::host_to_device),
      delta_completions(8, MappedRing<DeltaPublishCompletion>::Direction::device_to_host) {
  bind_cuda_device("cudaSetDevice(GPU navigation construction)");
  compute_client_id = connection_manager.client_id;
  compute_client_count = connection_manager.num_total_clients;
  if (compute_client_count == 0 ||
      compute_client_count > format::kMaxComputeClients ||
      compute_client_id >= compute_client_count) {
    throw std::runtime_error("compute client identity exceeds storage reclaim capacity");
  }
  if (config.gpu_traversal_beam_width > kPersistentMaxBeam ||
      config.gpu_final_rerank_width > kPersistentMaxExact ||
      config.R > kPersistentMaxGraphDegree) {
    throw std::invalid_argument("GPU navigation beam/exact/degree limit exceeded");
  }
```

成员初始化列表里建了 4 个 `MappedRing`（见第 17 课）：

- `submissions` / `completions`：查询描述符的双向 ring，容量 `gpu_query_slots * 2`（host→device / device→host）；
- `delta_submissions` / `delta_completions`：delta 发布命令的 ring，容量 8——delta 命令频率远低于查询，不需要大 ring。

启动校验有三类：

1. **计算客户端身份**：`compute_client_count` 不能超过 `kMaxComputeClients`（存储侧 reclaim ack 表大小限制），`compute_client_id` 必须在范围内。这是 storage reclaim 协议的硬约束（见第 16 课）。
2. **kernel 容量上限**：`beam_width`/`rerank_width`/`R` 必须不超过 `kPersistentMaxBeam=128`/`kPersistentMaxExact=256`/`kPersistentMaxGraphDegree=128`（`persistent_kernel.hh:13-18`）。这些是 kernel 编译期常量，超了 kernel 会越界。
3. **配置与索引 layout 一致性**：下一节讲。

### 11.5.2 第 2 步：索引元数据合成与一致性校验（`construction.cc:99-137`）

```cpp
std::string load_error;
bool used_anchor_entry_points = false;
if (!format::synthesize_distributed_view(
      config.resolved_index_prefix(), index,
      format::SynthesisOptions{
        .entry_points = 0,
        .seed = static_cast<u64>(static_cast<u32>(config.seed)),
      },
      &used_anchor_entry_points, &load_error)) {
  throw std::runtime_error(load_error);
}
std::cerr << "[gpu-search] synthesized navigation manifest in memory from metadata"
          << (used_anchor_entry_points ? " and anchors\n" : "\n");
if (!pq::read_model(index_path::navigation_model_file(
      config.resolved_index_prefix(), index.layout.pq_subquantizers),
      pq_model, &load_error)) {
  throw std::runtime_error(load_error);
}
if (index.layout.dim != config.dim || index.layout.graph_degree != config.R ||
    index.layout.num_shards != remote_regions.size() ||
    index.layout.num_shards > kPersistentMaxShards ||
    index.layout.pq_subquantizers != pq_model.subquantizers ||
    index.layout.pq_subquantizers > kPersistentMaxSubquantizers ||
    index.layout.pq_bits != pq_model.bits_per_code ||
    index.layout.code_bytes != pq_model.code_bytes() ||
    index.layout.model_checksum != pq_model.checksum() ||
    index.layout.graph_entry_bytes != VamanaNode::hot_graph_entry_size() ||
    index.layout.graph_shard_bits != VamanaNode::HOT_GRAPH_SHARD_BITS ||
    index.layout.vector_dtype != static_cast<u32>(config.resolved_vector_dtype()) ||
    index.entry_points.size() > kPersistentMaxEntryPoints) {
  throw std::runtime_error("GPU navigation manifest does not match runtime metadata");
}
```

`synthesize_distributed_view` 把分散在多个存储节点上的元数据合成一个内存里的 `format::View`（见第 7/8 课）。PQ 模型从 `navigation_model_file` 读出（见第 9 课）。

13 项一致性校验覆盖维度、度数、shard 数、PQ 参数、checksum、图条目大小、shard bits、dtype。任何一项不匹配都直接抛错——这种"启动即校验"是为了避免 kernel 跑起来才越界崩溃。与第 7 课的 schema-15 校验衔接。

接着是 merge 容量校验（`construction.cc:131-137`）：

```cpp
const u64 max_merge_candidates =
  static_cast<u64>(config.gpu_traversal_beam_width) +
  static_cast<u64>(std::min(config.gpu_graph_prefetch_depth,
                            kPersistentScoreChunk)) * config.R;
if (max_merge_candidates > kPersistentMaxMergeCandidates) {
  throw std::invalid_argument("GPU navigation prefetch/degree exceeds parallel top-k capacity");
}
```

`max_merge_candidates` 是一次 top-k merge 候选数的上界：beam 宽度 + 预取深度 × 度数。超过 `kPersistentMaxMergeCandidates=2048`（`persistent_kernel.hh:20`）会让 kernel 的 candidate 数组越界。

### 11.5.3 第 3 步：anchor 表 + dynamic route overlay 初始化（`construction.cc:139-168`）

```cpp
anchor_table = load_anchor_table(config.resolved_index_prefix(), config.dim,
                                 index.layout.num_shards, index);
dynamic_route_capacity = static_cast<u32>(index.shards.size()) *
  kDynamicRouteSlotsPerShard;
dynamic_route_diff =
  std::make_unique<DynamicRouteOverlayDiff>(
    static_cast<u32>(index.shards.size()));
if (dynamic_route_diff->capacity() != dynamic_route_capacity) {
  throw std::logic_error("GPU dynamic route capacity mismatch");
}
dynamic_route_snapshot.resize(dynamic_route_capacity);
dynamic_route_update_scratch.reserve(dynamic_route_capacity);
for (u32 anchor = 0; anchor < anchor_table.raw_pointers.size(); ++anchor) {
  anchor_buckets_by_raw.emplace(anchor_table.raw_pointers[anchor], anchor);
  anchor_graph_keys_host.push_back(
    graph_cache_key(anchor_table.raw_pointers[anchor]));
}
std::sort(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end());
anchor_graph_keys_host.erase(
  std::unique(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end()),
  anchor_graph_keys_host.end());
if (anchor_graph_keys_host.size() > std::numeric_limits<u32>::max()) {
  throw std::runtime_error("GPU anchor route table exceeds uint32 capacity");
}
entry_handles = index.entry_points;
```

`load_anchor_table`（`construction.cc:16-70`）从 anchor sidecar 文件读 anchor 向量、handle、raw pointer，按 shard 组织。`DynamicRouteOverlayDiff` 是动态路由 overlay 的 diff 容器（见第 10 课）。`anchor_graph_keys_host` 排序去重后存的是 anchor 对应的图边 key（高 16 位 shard + 低 48 位偏移），后续 `stream_anchor_graph_to_gpu` 按这个顺序读图边记录。

### 11.5.4 第 4 步：内存预算 + region 布局（`construction.cc:184-374`）

这是构造函数最长的一段。先算 `engine_budget`（配置上限）和 `physically_available`（GPU 实际空闲减 reserve）取较小值，然后调 `memory_budget::estimate` 得到 `budget` 结构（codes/delta/cache/exact 各分多少）：

```cpp
const u64 engine_budget = static_cast<u64>(
  config.gpu_memory_limit_gb - config.gpu_memory_reserve_gb) << 30;
size_t free_gpu_bytes = 0;
size_t total_gpu_bytes = 0;
check_cuda(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes), "cudaMemGetInfo(GPU navigation budget)");
const u64 runtime_reserve = static_cast<u64>(config.gpu_memory_reserve_gb) << 30;
const u64 physically_available = free_gpu_bytes > runtime_reserve
  ? static_cast<u64>(free_gpu_bytes) - runtime_reserve : 0;
const u64 usable_budget = std::min(engine_budget, physically_available);
const auto budget = memory_budget::estimate(memory_budget::Request{ ... });
if (!budget.fits) {
  throw std::runtime_error(
    "GPU navigation allocations exceed the configured memory budget; codes=" +
    std::to_string(budget.code_bytes) + " fixed=" +
    std::to_string(budget.fixed_bytes));
}
```

`budget.fits` 为 false 时直接抛错——不会带着不够的内存往下走。然后从 `budget` 提取 `delta_capacity`/`delta_table_capacity`/`graph_cache_sets`/`graph_cache_slots`/`exact_cache_*` 等容量参数填到 `Impl` 成员里。

接下来是一段几十行的"额外 scratch 字节估算"（`construction.cc:240-288`）：`dynamic_code_scratch_bytes`、`dynamic_request_scratch_bytes`、`navigation_candidate_bytes`、`query_dispatch_bytes`、`direct_queue_bytes`、`graph_scratch_bytes`、`cache_admission_bytes`、`route_graph_bytes`。每个都按 `query_slots * 常量 * sizeof(...)` 算。这些 scratch 字节加起来不能超过 `usable_budget - budget.explicit_bytes`，否则抛错。

`resident_pq_capacity` 单独算（`construction.cc:289-305`）：先算可用字节，再算请求字节（`gpu_resident_pq_budget_mb`），取较小值，再 `choose_resident_pq_capacity` 算实际容量。`resident_pq_capacity < delta_capacity` 抛错——resident PQ tier 必须能容纳全部 delta tier 的 PQ code，否则 delta 升级为 resident 时放不下。

然后是 region 布局（`construction.cc:347-374`）：所有 region 都在一个 `d_remote_buffer` 大区里按 offset 切分：

```cpp
anchor_graph_region_offset = static_cast<size_t>(
  align_up(code_region_bytes, 512));
dynamic_code_region_offset = static_cast<size_t>(align_up(
  anchor_graph_region_offset + route_graph_record_bytes, 256));
exact_region_offset = static_cast<size_t>(align_up(
  dynamic_code_region_offset + dynamic_code_scratch_bytes, 256));
graph_scratch_offset = static_cast<size_t>(align_up(
  exact_region_offset + exact_bytes, 512));
exact_cache_offset = static_cast<size_t>(align_up(
  graph_scratch_offset + graph_scratch_bytes, 256));
graph_cache_offset = static_cast<size_t>(
  align_up(exact_cache_offset + exact_cache_bytes, 512));
control_region_offset = static_cast<size_t>(
  align_up(graph_cache_offset + graph_cache_bytes, 256));
// ...
const size_t remote_buffer_bytes = control_region_offset + control_region_bytes;
```

对齐值（256/512）由 GPUNetIO 的 RDMA 对齐要求决定。`align_up` 在 `cuda_helpers.hh:26-28`：

```cpp
inline u64 align_up(u64 value, u64 alignment) {
  return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}
```

### 11.5.5 第 5 步：GPUNetIO transport 建链（`construction.cc:375-413`）

```cpp
#ifdef DVSTOR_HAVE_GPUNETIO
  direct_transport = std::make_unique<gpu::GpuNetioPersistentTransport>(
    config, remote_buffer_bytes, channel_context, connection_manager, remote_regions);
  direct_view = direct_transport->view();
  if (direct_view.data == nullptr || direct_view.data_bytes < remote_buffer_bytes) {
    throw std::runtime_error("GPUNetIO returned an undersized GPU data region");
  }
  d_remote_buffer = direct_view.data;
  owns_remote_buffer = false;
#else
  throw std::runtime_error("GPU query engine requires DOCA GPUNetIO support");
#endif
d_pq_codes = d_remote_buffer;
d_anchor_graph_records = d_remote_buffer + anchor_graph_region_offset;
d_dynamic_code_records = d_remote_buffer + dynamic_code_region_offset;
d_exact_records = d_remote_buffer + exact_region_offset;
d_graph_scratch = d_remote_buffer + graph_scratch_offset;
d_exact_cache = d_remote_buffer + exact_cache_offset;
d_graph_cache = d_remote_buffer + graph_cache_offset;
d_control_snapshots = reinterpret_cast<format::StorageControlBlock*>(
  d_remote_buffer + control_region_offset);
// ... route_snapshot / route_sequence_before / route_after ...
```

这是与第 22 课（GPUNetIO）的衔接点。`GpuNetioPersistentTransport` 建 QP、probe、暴露一个 GPU 直接可见的 `data` 大区。`owns_remote_buffer = false` 标记这块内存归 transport 拥有，析构时不要 `device_free`——`~Impl` 里 `if (owns_remote_buffer) device_free(d_remote_buffer);`（`lifecycle.cc:444`）。

`#else throw` 说明：本引擎**强依赖** GPUNetIO，没有它无法运行。

接着建 `NavigationBootstrapper`（`construction.cc:404-408`）——这是 CPU posted RDMA 的 bootstrap 通道，区别于 query 期 GPU 发起的 GPUNetIO：

```cpp
control_bootstrapper = std::make_unique<NavigationBootstrapper>(
  config, channel_context, connection_manager, remote_regions,
  d_remote_buffer, remote_buffer_bytes);
std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
             "queries=strict GPU-initiated GPUNetIO\n";
initialize_storage_reclaim_ack();
(void)read_storage_route_publications();
stream_codes_to_gpu(*control_bootstrapper);
stream_anchor_graph_to_gpu(*control_bootstrapper);
```

注意三件事：

1. `initialize_storage_reclaim_ack()` 先初始化 reclaim ack 表（第 16 课）；
2. `read_storage_route_publications()` 强制读一次路由发布——失败也忽略（`(void)`），因为并发的存储发布可能给空结果，维护线程会重试；
3. `stream_codes_to_gpu` / `stream_anchor_graph_to_gpu` 是 §11.4 讲的两个 bootstrap 函数。

### 11.5.6 第 6 步：device buffer 批量分配（`construction.cc:417-798`）

这一大段是几百行 `device_allocate` + `cudaMemcpy` + `cudaMemset`，按 region 顺序分配并填充：

- **shard / OPQ / PQ centroids / entry points**（417-434）：`cudaMemcpy` host→device；
- **anchor graph metadata**（435-468）：keys/states/readers + pinned host snapshot；
- **anchor vectors + handles + PQ codes**（469-503）：anchor PQ code 用 `launch_gather_anchor_codes` 从 `d_pq_codes` 收集；
- **delta bucket heads**（498-502）：`0xff` 填充；
- **query input / transformed / LUT / candidates / visited**（505-523）；
- **dynamic code request scratch**（524-531）；
- **query dispatch ring**（533-553）：enqueue/dequeue/sequences/entries；
- **GPUNetIO direct batch queue**（555-611）：`direct_batch_queue_count = qps_per_node * remote_region_count`，每个 queue 容量 64（`kDirectBatchQueueCapacity`）；
- **graph cache + admission**（613-675）：keys/generations/timestamps/states/readers/victims + admission keys/victims；
- **exact cache + admission**（677-707）；
- **result ids/distances**（709-723）：pinned host mapped，host 与 device 共享指针；
- **delta records/vectors/pq_codes/links/override/resident_pq**（725-783）；
- **dynamic route slots/codes**（785-797）；
- `clear_delta_device_state()`（798）：§11.4.3 的全归零。

这段代码是典型的"装备清单式"代码：每个 `device_allocate` 都带一个描述性 tag（如 `"cudaMalloc(GPU navigation shards)"`），失败时 `cuda_helpers.hh:45-49` 的 `device_allocate` 实现会把 tag + CUDA 错误 + free/total memory 一起拼进异常消息：

```cpp
template <class T>
void device_allocate(T*& pointer, size_t count, const char* operation) {
  if (count == 0) {
    pointer = nullptr;
    return;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(operation) + ": allocation size overflow");
  }
  const size_t bytes = count * sizeof(T);
  const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer), bytes);
  if (status != cudaSuccess) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status) +
      " requested=" + std::to_string(bytes) +
      " free=" + std::to_string(free_bytes) +
      " total=" + std::to_string(total_bytes));
  }
}
```

注意 `count == 0` 时 `pointer = nullptr`——这让 `~Impl` 的 `device_free(pointer)` 能安全跳过（`device_free` 检查 `nullptr`，§11.6.4）。

`mapped_host_allocate`（`cuda_helpers.hh:59-75`）是 pinned + mapped 的双指针分配：host 端拿 `host_pointer`、device 端拿 `device_pointer`，两者指向同一块内存。这是 query input、staging buffer、kernel ready 标志等"host/device 共享"内存的标准做法。

### 11.5.7 第 7 步：stop flag + stream 创建 + kernel 参数组装（`construction.cc:800-862`）

```cpp
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                         cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation stop staging)");
*stop_host = 0;
device_allocate(stop_device, 1, "cudaMalloc(GPU navigation stop)");
check_cuda(cudaMemset(stop_device, 0, sizeof(u32)),
           "cudaMemset(GPU navigation stop)");
// ... direct_disabled_host/device, direct_error_host/device ...
// ... query/dispatcher/control kernel_ready host/device ...
check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking),
           "cudaStreamCreate(GPU navigation kernel)");
check_cuda(cudaStreamCreateWithFlags(&delta_stream, cudaStreamNonBlocking),
           "cudaStreamCreate(GPU navigation delta)");
check_cuda(cudaStreamCreateWithFlags(&rdma_stream, cudaStreamNonBlocking),
           "cudaStreamCreate(GPU navigation RDMA owners)");
check_cuda(cudaStreamCreateWithFlags(&route_refresh_stream,
                                     cudaStreamNonBlocking),
           "cudaStreamCreate(GPU anchor route refresh)");
```

四条 stream 都是 `cudaStreamNonBlocking`——不与默认 stream 同步，避免 NULL stream 的隐式同步拖累。四条 stream 的分工：

| Stream | 用途 |
|---|---|
| `kernel_stream` | 持久化查询 kernel（owner/query/dispatcher/control CTAs） |
| `delta_stream` | delta publication kernel（`submit_delta_publication`） |
| `rdma_stream` | GPUNetIO owner kernel + stop flag 信号 |
| `route_refresh_stream` | anchor 路由图刷新（`refresh_anchor_graph_records`） |

接着算 kernel 块数（`construction.cc:839-862`）：

```cpp
cudaDeviceProp properties{};
check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
           "cudaGetDeviceProperties(GPU navigation)");
gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
constexpr u32 warp_width = 32;
const u32 owner_warps_per_block = kPersistentQueryThreads / warp_width;
owner_kernel_blocks =
  (direct_batch_queue_count + owner_warps_per_block - 1) /
  owner_warps_per_block;
const u32 resident_blocks = static_cast<u32>(
  std::max(1, properties.multiProcessorCount));
constexpr u32 control_blocks = 2;
if (owner_kernel_blocks + control_blocks >= resident_blocks) {
  throw std::runtime_error(
    "GPU has too few SMs to keep GPUNetIO owners and control resident");
}
const u64 requested_blocks = static_cast<u64>(
  std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
const u64 useful_blocks = std::max<u64>(1, config.num_threads);
const u64 resident_query_blocks =
  resident_blocks - owner_kernel_blocks - control_blocks;
kernel_blocks = static_cast<u32>(std::min({
  static_cast<u64>(query_slots), requested_blocks, useful_blocks,
  resident_query_blocks}));
```

`kPersistentQueryThreads = 256`（每 CTA 256 线程，即 8 warp）。`owner_kernel_blocks` 按"每个 block 容纳 8 个 owner warp"算需要多少 block 给所有 GPUNetIO owner warp。`control_blocks = 2`（dispatcher + control）。三者相加必须 < SM 数，否则抛错——必须留出至少 1 个 block 给 query kernel。

`kernel_blocks` 是 query kernel 的 block 数，取 `query_slots`、`requested_blocks`（SM 数 × 每 SM 块数）、`useful_blocks`（`num_threads` 配置）、`resident_query_blocks`（剩余可驻留块数）的最小值。

### 11.5.8 第 8 步：`PersistentKernelParams` 装配（`construction.cc:864-1014`）

这一段是给 kernel 传参的超大结构体（见 `persistent_kernel.hh:81` 起的 `PersistentKernelParams` 定义）。150 个字段，每个都指向 §11.5.6 分配的 device buffer 或 §11.5.4 算出的 offset。这段代码没有逻辑，只是"把成员绑到 params 字段"。重点看几个特殊字段：

```cpp
.direct_timeout_ns = 20000000ULL,
.graph_cache_ttl_ns = static_cast<u64>(
  config.gpu_graph_cache_ttl_us == 0
    ? config.update_visibility_us
    : std::min(config.gpu_graph_cache_ttl_us,
               config.update_visibility_us)) * 1000,
```

`direct_timeout_ns = 20ms` 是 GPUNetIO 单次 RDMA 读的超时。`graph_cache_ttl_ns` 取 `gpu_graph_cache_ttl_us` 与 `update_visibility_us` 的较小值——图缓存的 TTL 不能超过 mutation 可见时延，否则会读到过期图边。

### 11.5.9 第 9 步：启动 kernel + 三条 host 线程（`construction.cc:1015-1018`）

```cpp
start_persistent_kernel();
admission_thread = std::thread([this] { admission_loop(); });
completion_thread = std::thread([this] { completion_loop(); });
maintenance_thread = std::thread([this] { maintenance_loop(); });
}
```

`start_persistent_kernel` 在 `lifecycle.cc:219`，下一节讲。三条线程：

- `admission_thread` → `admission_loop()`（`query_execution.cc:80`）：从 admission_queue 取查询，push 到 submissions ring；
- `completion_thread` → `completion_loop()`（`completion.cc:39`）：从 completions ring pop 结果，set promise；
- `maintenance_thread` → `maintenance_loop()`（storage_reclaim.cc，第 16 课）：retire delta、回收 slot、推进 reclaim ack。

到这里 `Impl::Impl` 返回，`PersistentSearchEngine` 构造完成，可以接受查询。

## 11.6 `lifecycle.cc` 后半：kernel 起/停与析构

### 11.6.1 `start_persistent_kernel`：kernel 启动 + 就绪 barrier（`lifecycle.cc:219-302`）

```cpp
void PersistentSearchEngine::Impl::start_persistent_kernel() {
  bind_cuda_device("cudaSetDevice(GPU navigation kernel start)");
  *stop_host = 0;
  *direct_disabled_host = 0;
  *direct_error_host = 0;
  check_cuda(cudaMemset(stop_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation start flag)");
  check_cuda(cudaMemset(direct_disabled_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation direct failure flag)");
  check_cuda(cudaMemset(direct_error_device, 0, sizeof(i32)),
             "cudaMemset(GPU navigation direct error)");
  (void)cudaGetLastError();
  std::fill_n(direct_owner_phases_host, direct_batch_queue_count, 0u);
  *query_kernel_ready_host = 0;
  *dispatcher_kernel_ready_host = 0;
  *control_kernel_ready_host = 0;
  std::atomic_thread_fence(std::memory_order_release);
```

启动前把所有 ready 标志、stop 标志、direct 错误标志全部清零。`std::atomic_thread_fence(std::memory_order_release)` 保证前面的写入在 kernel 启动前对 device 可见——mapped memory 的可见性靠这个 fence。

接着组装 `launch_params`（从 `kernel_params` 复制，补三个运行时字段：`direct_owner_block_count`/`query_block_count`/三个 ready_count 指针），调 `launch_persistent_search`：

```cpp
PersistentKernelParams launch_params = kernel_params;
launch_params.direct_owner_block_count = owner_kernel_blocks;
launch_params.query_block_count = kernel_blocks;
launch_params.query_kernel_ready_count = d_query_kernel_ready;
launch_params.dispatcher_kernel_ready_count = d_dispatcher_kernel_ready;
launch_params.control_kernel_ready_count = d_control_kernel_ready;
const u32 total_blocks = owner_kernel_blocks + kernel_blocks + 2;
launch_persistent_search(kernel_stream, launch_params, total_blocks,
                         kPersistentQueryThreads);
check_cuda(cudaGetLastError(), "launch_persistent_search(unified navigation)");
```

`total_blocks = owner + query + 2`（2 = dispatcher + control）。`launch_persistent_search` 的实现在第 17/21 课。`cudaGetLastError` 检查 launch 是否成功（launch 是异步的，但 launch 本身的参数错误会立即返回）。

接下来是就绪 barrier（`lifecycle.cc:247-294`）：

```cpp
const auto ready_deadline = std::chrono::steady_clock::now() +
  std::chrono::seconds(3);
u32 ready_owners = 0;
for (;;) {
  ready_owners = 0;
  for (u32 qp = 0; qp < direct_batch_queue_count; ++qp) {
    ready_owners +=
      *reinterpret_cast<volatile u32*>(direct_owner_phases_host + qp) == 1
        ? 1u : 0u;
  }
  const u32 ready_queries =
    *reinterpret_cast<volatile u32*>(query_kernel_ready_host);
  const u32 ready_dispatchers =
    *reinterpret_cast<volatile u32*>(dispatcher_kernel_ready_host);
  const u32 ready_controls =
    *reinterpret_cast<volatile u32*>(control_kernel_ready_host);
  if (ready_owners == direct_batch_queue_count &&
      ready_queries == kernel_blocks && ready_dispatchers == 1 &&
      ready_controls == 1) {
    break;
  }
  if (std::chrono::steady_clock::now() >= ready_deadline) {
    // ... 拼 owners/queries/dispatcher/control 的就绪数 + 第一个未就绪 owner phase ...
    *stop_host = 1;
    (void)cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                          cudaMemcpyHostToDevice, rdma_stream);
    (void)cudaStreamSynchronize(rdma_stream);
    (void)cudaStreamSynchronize(kernel_stream);
    throw std::runtime_error(
      "unified GPU grid did not become fully resident: owners=" + ...);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
kernel_running = true;
```

就绪条件四项全满足：所有 owner phase == 1、所有 query block ready、dispatcher ready、control ready。3 秒超时——超时后设置 `stop_host = 1` 并把 stop 信号同步到 device，等 kernel 退出后抛错。这种"超时即 fail-stop"是冷启动的硬约束：kernel 起不来就不能接受查询。

`volatile u32*` 读 mapped memory——host 端不能缓存，必须每次重新读。`sleep_for(1ms)` 是轮询间隔。

成功后 `kernel_running = true`，打日志：

```cpp
std::cerr << "[gpu-search] unified persistent CTAs=" << owner_kernel_blocks
          << "-owner+" << kernel_blocks
          << "-query+1-dispatch+1-control"
          << " QP-owner-warps=" << direct_batch_queue_count
          << " threads/CTA=" << kPersistentQueryThreads
          << " query_slots=" << query_slots << '\n';
```

### 11.6.2 `stop_persistent_kernel`：优雅停 kernel（`lifecycle.cc:304-320`）

```cpp
void PersistentSearchEngine::Impl::stop_persistent_kernel() {
  if (!kernel_running) return;
  bind_cuda_device("cudaSetDevice(GPU navigation kernel stop)");
  *stop_host = 1;
  check_cuda(cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                             cudaMemcpyHostToDevice, rdma_stream),
             "cudaMemcpyAsync(GPU navigation stop)");
  check_cuda(cudaStreamSynchronize(rdma_stream),
             "cudaStreamSynchronize(GPU navigation stop signal)");
  const cudaError_t query_status = cudaStreamSynchronize(kernel_stream);
  const cudaError_t control_status = cudaStreamSynchronize(delta_stream);
  const cudaError_t rdma_status = cudaStreamSynchronize(rdma_stream);
  kernel_running = false;
  check_cuda(query_status, "cudaStreamSynchronize(GPU navigation stop)");
  check_cuda(control_status, "cudaStreamSynchronize(GPU delta control stop)");
  check_cuda(rdma_status, "cudaStreamSynchronize(GPU RDMA owner stop)");
}
```

停 kernel 的协议：把 `stop_host` 设为 1，异步拷到 `stop_device`，等 `rdma_stream` 同步完（保证 stop 信号到达 device），然后同步 kernel/delta/rdma 三条 stream。注意这里**先同步、后检查错误**——三条 stream 的 status 先存下来，`kernel_running = false` 先置位（保证析构不会重复 stop），再依次 check。如果某条 stream 同步返回错误，会在这里抛异常——但此时 `kernel_running` 已经是 false，析构不会再走 stop 逻辑。

### 11.6.3 `~Impl`：析构序列（`lifecycle.cc:322-466`）

这是本课最关键的"资源回收"代码。先看整体顺序：

```cpp
PersistentSearchEngine::Impl::~Impl() {
  const cudaError_t device_status = cudaSetDevice(static_cast<int>(config.gpu_device));
  if (device_status != cudaSuccess) {
    std::cerr << "[gpu-search] failed to bind CUDA device during teardown: "
              << cudaGetErrorString(device_status) << '\n';
  }
  accepting.store(false, std::memory_order_release);
  maintenance_shutdown.store(true, std::memory_order_release);
  maintenance_cv.notify_all();
  admission_cv.notify_all();
  slot_cv.notify_all();
  if (maintenance_thread.joinable()) maintenance_thread.join();
  shutdown.store(true, std::memory_order_release);
  admission_cv.notify_all();
  if (admission_thread.joinable()) admission_thread.join();
  reject_queued_submissions("persistent GPU query engine is stopping");
  const auto drain_deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  while (pending_count.load(std::memory_order_acquire) != 0 &&
         std::chrono::steady_clock::now() < drain_deadline) {
    std::this_thread::yield();
  }
  if (kernel_running) {
    // ... stop kernel（同 §11.6.2 的内联版本） ...
  }
  reject_all_pending("persistent GPU query engine stopped before completion");
  if (completion_thread.joinable()) completion_thread.join();
  // ... cudaStreamDestroy × 4 ...
  // ... cudaFreeHost × N（pinned host buffer） ...
  // ... device_free × N（device buffer） ...
  // ... control_bootstrapper.reset() ...
  // ... if (owns_remote_buffer) device_free(d_remote_buffer) ...
  // ... direct_transport.reset() (GPUNetIO) ...
}
```

逐阶段讲：

**阶段 A：绑定 device + 设标志 + join maintenance**（322-333）

`cudaSetDevice` 失败不抛异常只打日志——析构不能抛。设 `accepting = false`（拒绝新查询）、`maintenance_shutdown = true`，唤醒三条 cv。先 join maintenance——它负责 retire/reclaim，必须先停，否则它可能在后面 stop kernel 时还在写 delta 设备状态。

**阶段 B：设 shutdown + join admission + reject queued**（334-337）

`shutdown = true` 让 admission_loop 退出。`admission_cv.notify_all()` 唤醒可能在等的 admission。join 完后 `reject_queued_submissions` 把 admission_queue 里还没 push 到 ring 的查询全部 reject（§11.7.3）。

**阶段 C：drain pending + stop kernel**（338-355）

```cpp
const auto drain_deadline = std::chrono::steady_clock::now() +
  std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
while (pending_count.load(std::memory_order_acquire) != 0 &&
       std::chrono::steady_clock::now() < drain_deadline) {
  std::this_thread::yield();
}
if (kernel_running) {
  *stop_host = 1;
  if (rdma_stream != nullptr) {
    (void)cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                          cudaMemcpyHostToDevice, rdma_stream);
    (void)cudaStreamSynchronize(rdma_stream);
  }
  if (kernel_stream != nullptr) cudaStreamSynchronize(kernel_stream);
  if (delta_stream != nullptr) cudaStreamSynchronize(delta_stream);
  if (rdma_stream != nullptr) cudaStreamSynchronize(rdma_stream);
  kernel_running = false;
}
```

drain 阶段等 `pending_count == 0`（所有在飞查询都完成），但最多等 `storage_owner_rpc_timeout_ms`。超时后强制 stop kernel——此时可能还有未完成的查询，下一阶段 reject。

注意 stop kernel 这段是 `~Impl` 内联的，与 `stop_persistent_kernel` 几乎一样但不调用它——因为 `stop_persistent_kernel` 会 `check_cuda` 抛异常，析构不能抛。这里全部 `(void)` 忽略错误。

**阶段 D：reject_all_pending + join completion**（356-357）

```cpp
reject_all_pending("persistent GPU query engine stopped before completion");
if (completion_thread.joinable()) completion_thread.join();
```

`reject_all_pending`（§11.7.4）把所有还在 `pending_queries` 里的查询 set exception。completion_thread 在 `pending_count == 0` + `shutdown == true` 后才会退出（`completion.cc:40-41` 的循环条件）。

**阶段 E：销毁 stream + 释放 pinned host + 释放 device**（358-466）

按顺序：

1. `cudaStreamDestroy` × 4（route_refresh / rdma / delta / kernel）；
2. `cudaFreeHost` × N（pinned host buffer：direct_disabled/direct_error/direct_owner_phases/control/dispatcher/query kernel_ready/stop/result_ids/result_distances/delta_staging_*/delta_override/delta_durable/resident_pq_erase/dynamic_route_*/delta_supersede/graph_invalidation/anchor_graph_validation/anchor_graph_readers/query_input）；
3. `device_free` × N（device buffer，按分配逆序释放）；
4. `control_bootstrapper.reset()`（先于 remote_buffer 释放，因为 bootstrapper 持有 remote_buffer 指针）；
5. `if (owns_remote_buffer) device_free(d_remote_buffer)`（GPUNetIO 时 owns=false，跳过）；
6. `direct_transport.reset()`（GPUNetIO 释放 QP + remote_buffer）。

`device_free`（`cuda_helpers.hh:53-57`）：

```cpp
template <class T>
void device_free(T*& pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
}
```

置 nullptr 是防御性的——析构里重复 free 也不会出问题。

## 11.7 fail-stop 模型：`health.cc`

`health.cc` 只有 114 行，定义了引擎如何"从健康变不健康"以及"不健康时如何处理在飞查询"。

### 11.7.1 `unhealthy_message`：读取错误消息

`health.cc:7-10`：

```cpp
std::string PersistentSearchEngine::Impl::unhealthy_message() {
  std::lock_guard<std::mutex> lock(admission_mutex);
  return health_error.empty() ? "persistent GPU query engine is unhealthy" : health_error;
}
```

`health_error` 是 `std::string`，访问它必须持 `admission_mutex`——与 `mark_unhealthy` 写入时同一把锁。`search()` 入口（`query_execution.cc:11-13`）在 `healthy` 为 false 时调这个函数取错误消息。

### 11.7.2 `reject_submission`：拒绝单个在飞查询

`health.cc:12-39`：

```cpp
void PersistentSearchEngine::Impl::reject_submission(const PendingSubmission& submission,
                       const std::string& message) {
  std::shared_ptr<PendingQuery> pending;
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    const auto iterator = pending_queries.find(submission.descriptor.request_id);
    if (iterator != pending_queries.end()) {
      pending = std::move(iterator->second);
      pending_queries.erase(iterator);
    }
  }
  if (!pending) return;
  if (active_query_tickets != nullptr) {
    active_query_tickets[pending->slot].store(0, std::memory_order_release);
  }
  if (active_query_snapshots != nullptr) {
    active_query_snapshots[pending->slot].store(0, std::memory_order_release);
  }
  pending->promise.set_exception(
    std::make_exception_ptr(std::runtime_error(message)));
  {
    std::lock_guard<std::mutex> lock(slot_mutex);
    free_slots.push_back(pending->slot);
  }
  slot_cv.notify_one();
  pending_count.fetch_sub(1, std::memory_order_release);
  maintenance_cv.notify_all();
}
```

步骤：

1. 持 `pending_mutex` 从 `pending_queries` 取出 `PendingQuery`（找不到说明已经被 completion 处理过，直接返回）；
2. 把 `active_query_tickets[slot]` 和 `active_query_snapshots[slot]` 清零——这是 retire barrier 检查"该 slot 是否还有查询在读"的标志（见第 16 课 storage_reclaim）；
3. `promise.set_exception` 让 `search()` 的 `future.get()` 抛异常；
4. 持 `slot_mutex` 把 slot 放回 `free_slots`，`slot_cv.notify_one()` 唤醒可能在等 slot 的 `search()`；
5. `pending_count.fetch_sub(1)` + `maintenance_cv.notify_all()`——maintenance 线程在等 `pending_count == 0` 时可以推进 retire barrier。

### 11.7.3 `mark_unhealthy` 与 `reject_queued_submissions`：核心 fail-stop

`health.cc:41-57`：

```cpp
void PersistentSearchEngine::Impl::mark_unhealthy(const std::string& message) {
  std::deque<PendingSubmission> rejected;
  {
    std::lock_guard<std::mutex> lock(admission_mutex);
    if (!healthy.load(std::memory_order_relaxed)) return;
    health_error = message;
    healthy.store(false, std::memory_order_release);
    rejected.swap(admission_queue);
  }
  admission_cv.notify_all();
  slot_cv.notify_all();
  for (const PendingSubmission& submission : rejected) {
    reject_submission(submission, message);
  }
  std::cerr << "[gpu-search] query engine entered fail-stop mode: "
            << message << '\n';
}
```

这是 fail-stop 的核心：

1. 持 `admission_mutex`，检查 `healthy`——已经 unhealthy 就直接返回（幂等）；
2. 写 `health_error`、`healthy = false`、`swap` 取出整个 `admission_queue`；
3. 释放锁后 `notify_all` 两条 cv（admission/slot）——admission 线程会因 `healthy == false` 退出，`search()` 在 `slot_cv.wait` 里也会因 `!healthy` 醒来；
4. 对每个被 reject 的 submission 调 `reject_submission`（§11.7.2）；
5. 打 fail-stop 日志。

关键点：**`admission_queue` 在锁内 swap 出来，reject 在锁外做**。这避免了 reject 时持 `pending_mutex`/`slot_mutex` 又持 `admission_mutex` 的锁顺序问题。

`reject_queued_submissions`（`health.cc:59-68`）是 `mark_unhealthy` 的"轻量版"——只 reject 队列里所有 submission 但不设 `healthy = false`，用于 shutdown 时：

```cpp
void PersistentSearchEngine::Impl::reject_queued_submissions(const std::string& message) {
  std::deque<PendingSubmission> rejected;
  {
    std::lock_guard<std::mutex> lock(admission_mutex);
    rejected.swap(admission_queue);
  }
  for (const PendingSubmission& submission : rejected) {
    reject_submission(submission, message);
  }
}
```

`~Impl` 在 join 完 admission 线程后调它（`lifecycle.cc:337`），清理 admission 线程来不及处理的队列。

### 11.7.4 `reject_all_pending`：shutdown 终极清理

`health.cc:70-104`：

```cpp
void PersistentSearchEngine::Impl::reject_all_pending(const std::string& message) {
  std::vector<std::shared_ptr<PendingQuery>> rejected;
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    rejected.reserve(pending_queries.size());
    for (auto& [request_id, pending] : pending_queries) {
      (void)request_id;
      rejected.push_back(std::move(pending));
    }
    pending_queries.clear();
  }
  if (rejected.empty()) return;
  {
    std::lock_guard<std::mutex> lock(slot_mutex);
    for (const auto& pending : rejected) {
      if (active_query_tickets != nullptr) {
        active_query_tickets[pending->slot].store(0, std::memory_order_release);
      }
      if (active_query_snapshots != nullptr) {
        active_query_snapshots[pending->slot].store(0, std::memory_order_release);
      }
      free_slots.push_back(pending->slot);
    }
  }
  for (const auto& pending : rejected) {
    try {
      pending->promise.set_exception(
        std::make_exception_ptr(std::runtime_error(message)));
    } catch (const std::future_error&) {
    }
  }
  pending_count.fetch_sub(rejected.size(), std::memory_order_release);
  slot_cv.notify_all();
  maintenance_cv.notify_all();
}
```

与 `reject_submission` 的区别：

- 不按 request_id 找，而是把 `pending_queries` 全部 move 出来；
- 一次性持 `slot_mutex` 批量放回 free_slots；
- `promise.set_exception` 用 try/catch 包住——`std::future_error` 可能在 promise 已经被 set（比如 completion 线程抢先 set 了结果）时抛出，这种情况下忽略。

`~Impl` 在 stop kernel 后调它（`lifecycle.cc:356`），处理 drain 超时还没完成的查询。

### 11.7.5 `bind_cuda_device`：线程入口的 device 绑定

`health.cc:106-112`：

```cpp
void PersistentSearchEngine::Impl::bind_cuda_device(const char* operation) const {
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU navigation)");
  if (current_device != static_cast<int>(config.gpu_device)) {
    check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)), operation);
  }
  return;
}
```

每条线程入口（`admission_loop`、`maintenance_loop`、构造函数各阶段）都会调它。因为线程可能在其它设备上被调度（多 GPU 机器），必须确保 CUDA 调用落在 `config.gpu_device` 上。`cudaGetDevice` 先读当前，只在不一样时 `cudaSetDevice`——避免无谓的系统调用。

## 11.8 `cuda_helpers.hh`：基础设施

`cuda_helpers.hh` 只 77 行，定义了本课反复用到的几个工具。除 §11.5.6 已讲的 `device_allocate`/`device_free`/`mapped_host_allocate` 和 §11.5.4 的 `align_up`，还有：

### 11.8.1 `check_cuda`：错误抛出

`cuda_helpers.hh:19-24`：

```cpp
inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}
```

把 `cudaError_t` 翻译成 `std::runtime_error`，错误消息 = operation + ": " + CUDA 错误字符串。本课所有 CUDA 调用都套了它。

### 11.8.2 几个常量

`cuda_helpers.hh:14-17`：

```cpp
inline constexpr u32 kDirectBatchQueueCapacity = 64;
inline constexpr u32 kCacheAdmissionWays = 4;
inline constexpr u32 kMaxCacheAdmissionSets = 1u << 18;
inline constexpr u32 kResidentRouteReady = 2;
```

- `kDirectBatchQueueCapacity = 64`：每个 GPUNetIO owner queue 的容量（§11.5.6）；
- `kCacheAdmissionWays = 4`：cache admission 表的 way 数；
- `kMaxCacheAdmissionSets = 262144`：cache admission 表的最大 set 数（防止 admission 表占太多内存）；
- `kResidentRouteReady = 2`：resident route 的 ready 状态值（§11.5.6 的 `anchor_graph_ready_states_host` 初始化用）。

### 11.8.3 `check_doca` 在哪？

任务大纲提到 `check_doca`，但 `cuda_helpers.hh` 里没有——本课文件范围内只有 `check_cuda`。`check_doca` 应该在第 22 课 GPUNetIO transport 的代码里（`gpu/gpunetio_transport.hh`），本课不讲。

## 11.9 引擎状态机与线程/资源归属图

把前几节的信息综合成两张图。

### 11.9.1 状态机

```
                  ┌─────────────────────┐
                  │   Constructing      │
                  │ (Impl::Impl 体里)   │
                  └──────────┬──────────┘
                             │ 9 步装配完成
                             │ (start_persistent_kernel 成功 + 3 线程已起)
                             ▼
                  ┌─────────────────────┐
                  │   Accepting         │  ←── healthy=true, accepting=true
                  │ (稳态：查询/delta)   │      kernel_running=true
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       mark_unhealthy    ~PersistentSearchEngine   kernel 启动超时
       (GPUNetIO/kernel/   (析构触发)               (start_persistent_kernel)
        delta/校验失败)
              │              │              │
              ▼              ▼              ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │  Unhealthy  │  │  Stopping   │  │ Construct   │
       │ (fail-stop) │  │ (drain+stop)│  │  Failed     │
       │ 查询立即抛  │  │             │  │ (异常抛出)  │
       └──────┬──────┘  └──────┬──────┘  └─────────────┘
              │                │
              │                │ kernel stop + reject_all_pending
              │                │ + join completion + 释放资源
              │                ▼
              │           ┌─────────────┐
              │           │  Destroyed  │
              │           │ (Impl 已析构)│
              │           └─────────────┘
              │
              │ ~PersistentSearchEngine 后同样走 Stopping → Destroyed
              └──────────────► (等析构)
```

注意 `Unhealthy` 不是终态——它仍可被析构（`~PersistentSearchEngine` → `~Impl` 走 Stopping 流程）。fail-stop 只影响查询路径，不影响 shutdown 路径。

### 11.9.2 线程/资源归属图

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PersistentSearchEngine (宿主)                       │
│                                                                     │
│  DeltaCoordinator delta_     ◄──┐                                   │
│  Telemetry telemetry_           │ 反向引用 engine.delta_/telemetry_  │
│  mutex mutation_publish_mutex_  │                                   │
│  unique_ptr<Impl> impl_ ────────┼──► ┐                              │
└─────────────────────────────────┼────┼──────────────────────────────┘
                                  │    │
                                  │    │
                                  ▼    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Impl (PImpl)                                  │
│                                                                     │
│  ┌─── Host 线程 ──────────────────────────────────────────────┐    │
│  │  admission_thread   → admission_loop (query_execution.cc)  │    │
│  │  completion_thread  → completion_loop (completion.cc)      │    │
│  │  maintenance_thread → maintenance_loop (storage_reclaim.cc)│    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── CUDA Stream ────────────────────────────────────────────┐    │
│  │  kernel_stream       → persistent search kernel             │    │
│  │  delta_stream        → delta publication kernel             │    │
│  │  rdma_stream         → GPUNetIO owner kernel + stop signal  │    │
│  │  route_refresh_stream→ anchor route refresh                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Device 内存（d_remote_buffer 大区，GPUNetIO 拥有）──────┐    │
│  │  d_pq_codes | d_anchor_graph_records | d_dynamic_code_*    │    │
│  │  d_exact_records | d_graph_scratch | d_exact_cache          │    │
│  │  d_graph_cache | d_control_snapshots | d_storage_route_*    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Device 内存（独立 cudaMalloc，~Impl 释放）──────────────┐    │
│  │  d_shards/d_opq_matrix/d_pq_centroids/d_entry_points       │    │
│  │  d_anchor_vectors/handles/pq_codes                          │    │
│  │  d_queries/d_transformed_queries/d_query_luts               │    │
│  │  d_navigation_candidate_*/d_visited                         │    │
│  │  d_query_dispatch_*/d_direct_batch_*                        │    │
│  │  d_graph_cache_*/d_graph_admission_*                        │    │
│  │  d_exact_cache_*/d_exact_admission_*                        │    │
│  │  d_delta_records/vectors/pq_codes/next/prev/...             │    │
│  │  d_resident_pq_*/d_dynamic_route_*                          │    │
│  │  stop_device/direct_disabled_device/direct_error_device     │    │
│  │  d_query/dispatcher/control_kernel_ready                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Pinned/Mapped Host（cudaHostAlloc，~Impl 释放）─────────┐    │
│  │  query_input_host ←→ d_query_input (mapped)                 │    │
│  │  result_ids_host ←→ d_result_ids (mapped)                   │    │
│  │  result_distances_host ←→ d_result_distances (mapped)       │    │
│  │  stop_host/direct_disabled_host/direct_error_host           │    │
│  │  query/dispatcher/control_kernel_ready_host (mapped)        │    │
│  │  direct_owner_phases_host (mapped)                          │    │
│  │  delta_staging_slots/records/vectors_host (mapped)          │    │
│  │  delta_supersede/override/durable_updates_host (mapped)     │    │
│  │  resident_pq_erase_updates_host (mapped)                    │    │
│  │  dynamic_route_updates/code_updates_host (mapped)           │    │
│  │  graph_invalidation_keys_host (mapped)                      │    │
│  │  anchor_graph_readers_host/anchor_graph_validation_host     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Ring（MappedRing，host/device 共享）────────────────────┐    │
│  │  submissions (host→device, 容量 query_slots*2)              │    │
│  │  completions (device→host, 容量 query_slots*2)              │    │
│  │  delta_submissions (host→device, 容量 8)                    │    │
│  │  delta_completions (device→host, 容量 8)                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── 外部组件 ───────────────────────────────────────────────┐    │
│  │  NavigationBootstrapper control_bootstrapper (CPU posted)   │    │
│  │  GpuNetioPersistentTransport direct_transport (GPU direct)  │    │
│  │  DynamicRouteOverlayDiff dynamic_route_diff                 │    │
│  │  AnchorTable anchor_table                                   │    │
│  │  format::View index  (合成视图)                              │    │
│  │  pq::Model pq_model                                          │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.9.3 析构顺序（与图对应）

`~Impl` 的释放顺序严格对应"先停线程 → 再停 kernel → 再释放 device → 最后释放外部组件"：

1. **停线程**：maintenance → admission → (drain) → completion（§11.6.3 阶段 A-D）；
2. **停 kernel**：stop flag → 同步三条 stream（阶段 C 内联）；
3. **销毁 stream**：route_refresh → rdma → delta → kernel（阶段 E）；
4. **释放 pinned host**：所有 `cudaFreeHost`（阶段 E）；
5. **释放 device**：所有 `device_free`，按分配逆序（阶段 E）；
6. **释放外部组件**：`control_bootstrapper.reset()` → `if (owns_remote_buffer) device_free(d_remote_buffer)` → `direct_transport.reset()`（阶段 E 末尾）。

`control_bootstrapper` 必须先于 `d_remote_buffer` 释放，因为它持有 `d_remote_buffer` 指针；`direct_transport` 最后释放，因为它拥有 `d_remote_buffer`（GPUNetIO 模式下 `owns_remote_buffer = false`）。

## 11.10 与其他模块的关系

本课代码与其它课的衔接点：

- **第 2 课（公共类型与配置）**：`configuration::IndexConfiguration` 是 `PersistentSearchEngine` 构造参数；`MemoryRegionTokens`/`Context`/`ClientConnectionManager` 来自第 3-5 课的并发/RDMA 原语。
- **第 7 课（schema-15 索引格式）**：`format::synthesize_distributed_view`/`format::StorageControlBlock`/`format::StorageRoutePublication` 是启动校验的对象；`construction.cc:117-130` 的 13 项一致性校验直接依赖 schema-15 的字段。
- **第 8 课（元数据/owner map）**：`index.shards`/`index.entry_points`/`index.layout` 在 §11.5.2 合成。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`Telemetry`/`TelemetrySnapshot`/`pq::Model`；`engine.telemetry_.*` 的所有计数器在 `publish_mutations`/`reset_telemetry` 里更新。
- **第 10 课（delta/动态路由/预算）**：`DeltaCoordinator`/`DeltaMutation`/`DynamicRouteOverlayDiff`/`memory_budget`；`publish_mutations` 的 epoch 协议、`try_reserve_mutation_capacity` 的水位检查都依赖第 10 课的实现。
- **第 12-13 课（construction 上/下）**：本课只讲 `Impl::Impl` 的"装配"部分（construction.cc 的 9 步），第 12 课会讲 construction 的更细节（anchor 表加载、QP 建立、probe）；第 13 课讲离线构建如何产生这些文件。
- **第 14 课（查询执行/路由/完成）**：`search()` → `Impl::search` → admission → kernel → completion 的完整链路；本课只给入口（`query_execution.cc:7`）。
- **第 15 课（增量发布）**：`upload_mutations`/`submit_delta_publication` 在 `delta_publication.cc`；本课只讲 `publish_mutations` 的上层协议。
- **第 16 课（存储回收 RCU）**：`maintenance_loop`/`retire_durable_delta`/`reclaim_retired_delta_slots_locked`/`pending_storage_reclaim_acks` 在 `storage_reclaim.cc`；本课只讲 maintenance 线程的启动/停止。
- **第 17 课（kernel 启动器/上下文/device ring）**：`launch_persistent_search`/`PersistentKernelParams`/`MappedRing`/`DeviceRingView`；本课只讲启动调用点。
- **第 19 课（RDMA cache）**：`d_graph_cache_*`/`d_exact_cache_*` 是 graph cache 与 exact cache 的 device 状态。
- **第 20 课（查询遍历主循环）**：kernel 内的遍历逻辑；本课只讲 kernel 怎么起。
- **第 21 课（kernel 运行时/角色调度）**：owner/query/dispatcher/control 四种 CTA 的协作；本课只讲它们的 block 数计算与 ready barrier。
- **第 22 课（GPUNetIO 传输/probe）**：`GpuNetioPersistentTransport`/`direct_view`/`check_doca`；本课只讲 transport 在 `Impl::Impl` 的建链点。
- **第 28 课（计算侧 storage owner 更新）**：`mark_committed_mutation_gap` 是存储侧 → 计算侧的 fail-stop 通道。

## 11.11 小结

本课拆开了 `PersistentSearchEngine` 的 PImpl 外壳，讲清了四件事：

1. **PImpl 边界**：`DeltaCoordinator`/`Telemetry`/`mutation_publish_mutex_` 留在宿主类（跨线程共享、不需经 Impl 间接访问），其余全部下沉到 `Impl`（device 指针、stream、线程、ring）。
2. **装配序列**：`Impl::Impl` 的 9 步——校验 → 合成 view → anchor 表 → 内存预算 → region 布局 → GPUNetIO 建链 → device buffer 分配 → stream/kernel 参数 → kernel 启动 + 三线程。每一步都有"启动即校验"，失败直接抛异常。
3. **shutdown 顺序**：`~Impl` 的 6 阶段——绑 device → 停 maintenance → 停 admission → drain pending → stop kernel → reject_all_pending → 释放 stream/host/device/外部组件。析构全程不抛异常，CUDA 错误用 `(void)` 忽略。
4. **fail-stop 模型**：`mark_unhealthy` 原子地把 `healthy` 翻 false 并 reject 整个 admission_queue；`reject_submission`/`reject_all_pending` 处理在飞查询；`MutationCapacityError` 是唯一可恢复的异常（仅拒绝计数，不 fail-stop）。

后续第 12 课会深入 `construction.cc` 的 anchor 表加载与 QP 建立；第 14 课讲 `search()` 的完整链路；第 16 课讲 maintenance 线程的 RCU 回收——它们都建立在本课给出的资源归属与生命周期框架之上。
