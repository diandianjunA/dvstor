# 第 16 课 存储回收 RCU

> 本课讲解 dvstor 计算侧如何把"存储节点已经 durable 的旧版本向量"安全地从 GPU 上退休，并通过一个远端 ACK 槽把"安全 sequence"回写给存储节点，使后者可以复用动态物理地址。整套机制是一段典型的读拷贝更新（RCU）：存储侧先把数据写到新地址，计算侧在确认没有任何在飞查询还会读到旧地址之后，再放行旧地址的回收。

## 本课目标与涉及文件

读完本课你应当能回答：

1. 计算侧怎么知道存储侧已经把某条 mutation 真正 durable 了？`durable_maintenance_sequence` 怎么变成 `safe_durable_sequences`？
2. 退休一条 delta 记录和它对应的常驻 PQ 槽为什么不能立刻回收？`retired_delta_batches` / `retired_resident_pq_batches` 上的 `query_ticket_barrier` 在等什么？
3. `reclaim_ack_sequences[compute_client_id]` 这个远端 8 字节槽是怎么被写入的？为什么它是"每计算节点独立"的？
4. 前台 `reserve_mutation_capacity` / `release_mutation_capacity` 怎么防止后台 maintenance worker 被"饿死"？
5. 任何一步失败时引擎怎么 `mark_unhealthy`，避免把脏指针暴露给查询？

涉及文件：

- `src/gpu_search/persistent_engine/storage_reclaim.cc`（约 587 行，本课主线）
- `src/memory_node/storage_reclaim.hh`（存储侧回收队列，仅头文件，53 行）
- `src/gpu_search/persistent_engine/impl.hh`（`Impl` 中所有 reclaim 相关成员声明）
- `src/gpu_search/delta_index.hh` / `src/gpu_search/delta_index.cc`（`DeltaCoordinator::retire_durable`，回收的源头）
- `src/gpu_search/persistent_engine/delta_publication.cc`（`query_ticket_barrier_passed` / `durable_snapshot_safe` / `reclaim_retired_delta_slots_locked` 三个 RCU 原语）
- `src/gpu_search/persistent_engine.cc`（`reserve_mutation_capacity` / `release_mutation_capacity` 的容量预算逻辑）
- `src/gpu_search/index_format.hh`（`StorageControlBlock` schema-15 control block version 2，`reclaim_ack_sequences` 数组）
- `src/gpu_search/persistent_kernel.hh`（`DeviceDeltaRecord`、`kDeltaDurable`/`kDeltaDeleted`/`kBaseOverrideEmpty` 等常量）
- `src/gpu_search/types.hh`（`storage_reclaim_ack_writes`/`storage_reclaim_ack_sequence`/`resident_pq_*`/`delta_reclaim_batches`/`mutation_capacity_*` 等遥测字段，第 243–272 行的 atomic 版本）

## 逐文件逐函数讲解

### 1. `src/memory_node/storage_reclaim.hh` —— 存储侧的回收队列

虽然本课主线在计算侧，但要理解"ACK 写回去之后存储侧会做什么"，必须先看存储侧的队列。这是一个非常小的头文件，但它定义了整条 RCU 链条最后一步的语义。

```cpp
// src/memory_node/storage_reclaim.hh:13
namespace memory_node_detail {

class StorageReclaimQueue {
public:
  void retire(RemotePtr pointer, u64 maintenance_sequence) {
    if (pointer.is_null() || maintenance_sequence == 0) return;
    pending_[maintenance_sequence].push_back(pointer);
    ++size_;
  }
```

存储侧在把一个动态节点写到新地址（发布 `maintenance_sequence = N`）之后，把"旧地址"通过 `retire(pointer, N)` 入队。注意它用 `std::map<u64, std::vector<RemotePtr>>`（第 46 行 `pending_`）按 sequence 分桶——这正是为了配合计算侧 ACK 的"按 sequence 推进"语义。

```cpp
// src/memory_node/storage_reclaim.hh:23
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
```

`acquire` 是存储侧在分配新动态地址时调用的：`durable_sequence` 是存储自己认为已经 durable 的 watermark，`acknowledged_sequence` 是从某个计算节点的 `reclaim_ack_sequences[client_id]` 槽读回来的 ACK 值。**两者取 min**——这条规则是本课的核心约束的镜像：

> 存储节点只有同时满足"我自己已经 durable"且"所有计算节点都已 ACK 看到这个 sequence"两个条件，才能复用一个旧物理地址。

`std::min` 在这里体现的是"木桶效应"：哪怕一个计算节点 ACK 落后，存储也不能复用它发出的某个旧地址，因为这个计算节点上可能还有在飞查询在用旧地址做 RDMA 读。`reused_` 计数器对应计算侧的 `resident_pq_reclaimed`，是端到端回收速率的对账指标。

`ready_` 是一个扁平 `std::vector<RemotePtr>`，所有已经"安全"的指针在这里排队等待被新写入复用。`pending_` → `ready_` 的迁移是一次性批量搬移，保证 `ready_` 里的指针都已经度过了 RCU 宽限期。

### 2. `src/gpu_search/index_format.hh` —— schema-15 control block version 2

RCU 协议落地在存储 4 KiB control 页里。先看 layout：

```cpp
// src/gpu_search/index_format.hh:25
inline constexpr u32 kStorageControlBytes = 4096;
inline constexpr u64 kStorageControlMagic = 0x314c525443565344ULL;  // "DSVCTRL1"
inline constexpr u32 kStorageControlVersion = 2;
inline constexpr u32 kMaxComputeClients = 64;
```

`kStorageControlVersion = 2` 就是题目里强调的 "schema-15 control block version 2"——schema-15 是索引整体的元数据 schema（见第 7 课），而 storage control page 自身有独立的 `kStorageControlVersion`。version 1 没有 `reclaim_ack_sequences` 数组；version 2 加入了一个最多 64 个计算客户端的 ACK 槽数组，这是支持"多计算节点 + 动态路由"的前提（见第 10、15 课）。

```cpp
// src/gpu_search/index_format.hh:80
struct alignas(64) StorageControlBlock {
  u64 magic{kStorageControlMagic};
  u32 version{kStorageControlVersion};
  u32 header_bytes{sizeof(StorageControlBlock)};
  u32 shard_id{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
  u32 code_bytes{};
  u32 compute_client_count{};
  u32 reserved0{};
  u64 next_maintenance_sequence{1};
  u64 durable_maintenance_sequence{};
  u64 dynamic_high_watermark{};
  u64 reclaim_pending_nodes{};
  u64 reclaim_reused_nodes{};
  u64 reserved1{};
  std::array<u64, kMaxComputeClients> reclaim_ack_sequences{};
};
// index_format.hh:146
static_assert(sizeof(StorageControlBlock) == 640);
static_assert(sizeof(StorageControlBlock) <= kStorageControlBytes);
```

关键字段：

- `next_maintenance_sequence`：存储侧每次发布新动态节点时分配的序号，单调递增。
- `durable_maintenance_sequence`：存储侧已 fsync/durable 到的最大 sequence。计算侧读这个字段决定能开始退休哪些 mutation。
- `reclaim_ack_sequences[client_id]`：**每个计算客户端独占一个 8 字节槽**。计算侧把"我已经安全度过了 RCU 宽限期的 sequence"写回这里。`kMaxComputeClients = 64` 决定了最多 64 个计算节点能并发接入一个 shard。
- `reclaim_pending_nodes` / `reclaim_reused_nodes`：存储侧对账计数，分别对应还在等 ACK 的指针数和已经复用的指针数。

`alignas(64)` + `static_assert(sizeof == 640)` 保证字段布局稳定，跨节点 RDMA 读不会因为 padding 漂移。control page 剩余的尾部空间 (`kStorageRoutePublicationOffset = 1024`，见第 15 课) 用来放 `StorageRoutePublication`——本课的 `synchronize_storage_routes` 会先处理它，再发 ACK。

### 3. `storage_reclaim.cc` 顶部 —— control block 校验与读取

```cpp
// src/gpu_search/persistent_engine/storage_reclaim.cc:7
void PersistentSearchEngine::Impl::validate_storage_control(
    const format::StorageControlBlock& control, size_t shard) const {
  if (control.magic != format::kStorageControlMagic ||
      control.version != format::kStorageControlVersion ||
      control.header_bytes != sizeof(format::StorageControlBlock) ||
      control.shard_id != shard ||
      control.compute_client_count != compute_client_count ||
      control.dynamic_record_bytes != index.shards[shard].dynamic_record_bytes ||
      control.dynamic_hot_offset != index.shards[shard].dynamic_hot_offset ||
      control.dynamic_code_offset != index.shards[shard].dynamic_code_offset ||
      control.code_bytes != index.layout.code_bytes) {
    std::ostringstream message;
    message << "storage maintenance control mismatch for shard " << shard
            << ": expected{...} actual{...}. Rebuild and restart every storage "
               "node from the current dev branch before starting the compute node.";
    throw std::runtime_error(message.str());
  }
}
```

这个函数在每次读 control block 后调用，校验 8 个不变量。注意第 13 行 `control.compute_client_count != compute_client_count`：存储侧写 control page 时必须知道总共有几个计算客户端，否则 `reclaim_ack_sequences` 数组的语义会错位。任何一项不匹配都直接抛异常——本课里所有失败路径都会被 `maintenance_loop` 的 `catch` 抓住并 `mark_unhealthy`，后面会看到。错误信息明确要求"rebuild and restart every storage node"，因为 schema-15 control page 是跨版本不兼容的。

```cpp
// src/gpu_search/persistent_engine/storage_reclaim.cc:45
std::vector<format::StorageControlBlock>
PersistentSearchEngine::Impl::read_storage_controls() {
  if (control_bootstrapper == nullptr || index.shards.empty()) return {};
  std::vector<NavigationRead> requests(index.shards.size());
  std::vector<i32> statuses(index.shards.size(), -EIO);
  for (size_t shard = 0; shard < index.shards.size(); ++shard) {
    requests[shard] = NavigationRead{
      .remote_offset = index.shards[shard].control_remote_offset,
      .destination_address = reinterpret_cast<u64>(d_control_snapshots + shard),
      .bytes = sizeof(format::StorageControlBlock),
      .memory_node = static_cast<u16>(shard),
    };
  }
  control_bootstrapper->read(requests, statuses);
  std::vector<format::StorageControlBlock> controls(index.shards.size());
  check_cuda(cudaMemcpy(controls.data(), d_control_snapshots,
                        controls.size() * sizeof(format::StorageControlBlock),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(storage maintenance controls)");
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    if (statuses[shard] <= 0) {
      throw std::runtime_error(
          "storage maintenance control read failed for shard " +
          std::to_string(shard));
    }
    validate_storage_control(controls[shard], shard);
  }
  return controls;
}
```

一次 fan-out 读所有 shard 的 control block。`d_control_snapshots` 是 GPU 显存里的一组 `StorageControlBlock`（每个 shard 一份），`NavigationRead` 把远端 4 KiB control 页的前 640 字节直接 RDMA 拉到 GPU 显存，再 `cudaMemcpy` 回 host 做校验。`control_bootstrapper` 是第 17 课讲的 kernel 启动器上下文里那个独立的 bootstrap QP——它和查询 kernel 用的 data QP 隔离，保证 maintenance 读不会被查询流量阻塞。任何一个 shard 读失败 (`statuses[shard] <= 0`) 都抛异常，从而触发 `maintenance_loop` 的 fail-stop。

### 4. `read_storage_route_publications` —— 三段式 seqlock 读

这个函数虽然主要服务于第 15 课的动态路由，但它是 RCU ACK 路径的前置步骤，且其"防撕裂读"思路和 ACK 协议高度相关，必须讲。

```cpp
// src/gpu_search/persistent_engine/storage_reclaim.cc:74
std::vector<format::StorageRoutePublication>
PersistentSearchEngine::Impl::read_storage_route_publications() {
  if (control_bootstrapper == nullptr || index.shards.empty()) return {};
  std::vector<NavigationRead> requests(index.shards.size());
  std::vector<i32> before_statuses(index.shards.size(), -EIO);
  std::vector<i32> body_statuses(index.shards.size(), -EIO);
  std::vector<i32> after_statuses(index.shards.size(), -EIO);
  ...
  for (u32 attempt = 0; attempt < 2; ++attempt) {
```

每个 shard 做三次 RDMA 读：先读 `sequence_begin`（在 `StorageRoutePublication` 偏移 0，第 113 行），再读整个 448 字节 body，最后再读一次 `sequence_begin`（实际上读的是同一个字段，但通过第二次读来检测 body 读取期间存储是否在原地原子换页）。

```cpp
// storage_reclaim.cc:144
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

三次读必须一致：`before == after == body.sequence_begin`。否则说明存储在 body 读期间正好在切换 publication——这是"RCU publish"的典型撕裂，对路由这种 advisory 元数据来说直接重试一次即可（外层 `for (u32 attempt = 0; attempt < 2; ++attempt)`）。

```cpp
// storage_reclaim.cc:182
  if (last_failure_was_transient && !saw_nontransient_failure) {
    // Route metadata is advisory. A torn low-frequency control-page read must
    // never fail queries or the mutation engine; retain the previous GPU
    // snapshot and retry on the next maintenance tick.
    engine.telemetry_.dynamic_route_snapshot_skips.fetch_add(
      1, std::memory_order_relaxed);
    return {};
  }
  throw std::runtime_error(
    "storage route snapshot unavailable after retry: " + last_error + ...);
```

两次重试都撕裂则记 `dynamic_route_snapshot_skips` 并返回空 `vector`——`synchronize_storage_routes` 见到空就 `return false`，**ACK 路径因此不会被推进**（见第 5 节 `maintenance_loop` 中的 `if (synchronize_storage_routes()) { enqueue_storage_reclaim_barriers(); ... }`）。这个设计至关重要：路由快照是 ACK 推进的"闸门"，一次撕裂读绝不能让存储过早复用某个还可能被旧路由引用的物理地址。

### 5. `synchronize_storage_routes` —— 路由先于 ACK

```cpp
// storage_reclaim.cc:196
bool PersistentSearchEngine::Impl::synchronize_storage_routes() {
  const std::vector<format::StorageRoutePublication> publications =
    read_storage_route_publications();
  if (publications.empty()) return false;
  ...
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

把 8 × shard 个路由槽展平到一个一维 snapshot vector。`live = source.remote_node != 0`——0 是 tombstone 哨兵，表示这个槽已经被存储侧收回。`dynamic_route_diff->prepare(...)` 做 diff，如果有变化，分配一个新 epoch 并通过 `submit_delta_publication` 把新路由 + 新 navigation code 推到 GPU（见第 15 课）。

```cpp
// storage_reclaim.cc:261
  // Queries acquire this epoch only after the control CTA has made both the
  // PQ bytes and route seqlocks visible.
  engine.delta_.publish_barrier(epoch);
  return true;
}
```

`publish_barrier` 把 `published_epoch_` CAS 到新 epoch（见 `delta_index.cc:80`）。返回 `true` 表示"路由侧确实有新发布"。`maintenance_loop` 见到 `true` 才会调 `enqueue_storage_reclaim_barriers`，否则跳过本轮 ACK——这是"先发布新路由 tombstone/替换，再发 ACK 让旧地址可回收"的严格顺序保证。

### 6. `write_storage_reclaim_acks` —— 写远端 ACK 槽

```cpp
// storage_reclaim.cc:267
void PersistentSearchEngine::Impl::write_storage_reclaim_acks(
    std::span<const u64> sequences) {
  if (sequences.size() != index.shards.size()) {
    throw std::invalid_argument("storage reclaim ACK cardinality mismatch");
  }
  std::vector<NavigationWrite> requests(index.shards.size());
  std::vector<i32> statuses(index.shards.size(), -EIO);
  for (size_t shard = 0; shard < index.shards.size(); ++shard) {
    u64* device_ack =
      &d_control_snapshots[shard].reclaim_ack_sequences[compute_client_id];
    check_cuda(cudaMemcpy(device_ack, &sequences[shard], sizeof(u64),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(storage reclaim ACK)");
    requests[shard] = NavigationWrite{
      .remote_offset = index.shards[shard].control_remote_offset +
        offsetof(format::StorageControlBlock, reclaim_ack_sequences) +
        static_cast<u64>(compute_client_id) * sizeof(u64),
      .source_address = reinterpret_cast<u64>(device_ack),
      .bytes = sizeof(u64),
      .memory_node = static_cast<u16>(shard),
    };
  }
  control_bootstrapper->write(requests, statuses);
  for (size_t shard = 0; shard < statuses.size(); ++shard) {
    if (statuses[shard] <= 0) {
      throw std::runtime_error(
          "storage reclaim ACK write failed for shard " +
          std::to_string(shard));
    }
  }
}
```

逐行拆：

- 第 274–275 行：每个 shard 在 GPU 显存的 `d_control_snapshots[shard]` 里有一个完整的 `StorageControlBlock` 副本。ACK 只动其中 `reclaim_ack_sequences[compute_client_id]` 那 8 字节。`compute_client_id` 在引擎初始化时确定（见第 11 课生命周期），是这个计算节点在集群里的唯一身份。
- 第 276–278 行：先把 host 上的 `sequences[shard]` `cudaMemcpy` 到那 8 字节。
- 第 279–286 行：构造 `NavigationWrite`，远端偏移 = shard 的 control page 基址 + `reclaim_ack_sequences` 字段在结构体里的偏移 + `compute_client_id * 8`。注意这是**带偏移的 RDMA 写**，只覆盖远端 4 KiB control page 里的 8 字节，不动其他字段。
- 第 288 行：`control_bootstrapper->write(...)` fan-out 写所有 shard。
- 第 289–295 行：任何一个 shard 写失败都抛异常 → `maintenance_loop` 捕获 → `mark_unhealthy`。

**为什么每计算节点独立 ACK？** 因为每个计算节点的在飞查询进度独立。A 节点可能已经 ACK 到 sequence 100，B 节点还在 80。存储侧 `StorageReclaimQueue::acquire(durable_seq, ack_seq_for_client_B)` 会被 B 卡住——这是正确的，B 上可能还有查询在读 sequence 80 之前的旧地址。多计算节点的 RCU 宽限期是"取最慢者"的，本课第 8 节的状态机图会展示这一点。

### 7. `initialize_storage_reclaim_ack` —— 启动时复位 ACK

```cpp
// storage_reclaim.cc:298
void PersistentSearchEngine::Impl::initialize_storage_reclaim_ack() {
  (void)read_storage_controls();
  pending_storage_reclaim_acks.resize(index.shards.size());
  enqueued_reclaim_ack_sequences.assign(index.shards.size(), 0);
  published_reclaim_ack_sequences.assign(index.shards.size(), 0);
  const std::vector<u64> reset_sequences(index.shards.size(), 0);
  write_storage_reclaim_acks(reset_sequences);
  std::cerr << "[gpu-search] storage reclaim RCU client=" << compute_client_id
            << '/' << compute_client_count << " ACK reset complete\n";
}
```

启动时：

1. 读一次 control block 验证 magic/version（`(void)` 丢弃返回值，纯粹做校验）。
2. 三个 per-shard vector 清零：`pending_storage_reclaim_acks`（待发布的 ACK 队列）、`enqueued_reclaim_ack_sequences`（已入队但还没 publish 的最大 sequence）、`published_reclaim_ack_sequences`（已经成功写到远端的 sequence）。
3. 主动写一全 0 的 ACK 数组到每个 shard——这告诉存储"这个计算节点重新启动了，之前任何 ACK 都作废"。存储侧据此把所有该 client 的 pending 指针重新视为"未 ACK"。

这一步是计算节点 fail-restart 的关键：如果计算节点崩溃重启后不主动复位 ACK，存储侧会一直以为旧 ACK 仍有效，从而可能在还有重启中查询读旧地址时就复用掉它。详见第 23 课存储侧 lifecycle。

### 8. `enqueue_storage_reclaim_barriers` —— 把 durable sequence 关联到 query ticket barrier

```cpp
// storage_reclaim.cc:309
void PersistentSearchEngine::Impl::enqueue_storage_reclaim_barriers() {
  std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
  const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
  for (size_t shard = 0; shard < safe_durable_sequences.size(); ++shard) {
    const u64 sequence = safe_durable_sequences[shard];
    if (sequence <= enqueued_reclaim_ack_sequences[shard]) continue;
    auto& queue = pending_storage_reclaim_acks[shard];
    if (!queue.empty() && queue.back().query_ticket_barrier == barrier) {
      queue.back().maintenance_sequence = sequence;
    } else {
      queue.push_back(PendingStorageReclaimAck{
        .maintenance_sequence = sequence,
        .query_ticket_barrier = barrier,
      });
    }
    enqueued_reclaim_ack_sequences[shard] = sequence;
  }
}
```

这是 RCU 宽限期的"注册"步骤。逐行：

- 第 310 行：锁 `query_snapshot_mutex`——这把锁同时保护 `next_query_ticket` 和 `active_query_tickets`/`active_query_snapshots` 数组（见 `delta_publication.cc:282` 的 `query_ticket_barrier_passed`）。
- 第 311 行：`barrier = next_query_ticket.load() - 1`。`next_query_ticket` 是下一个要分配的 ticket，所以 `barrier` 是"当前已经发出的最大 ticket"。任何 ticket ≤ barrier 的查询都是"在我注册 barrier 之前就已经在飞"的查询——它们可能读到旧版本。`acquire` 序保证读到的 ticket 值与查询线程把它写入 `active_query_tickets[slot]` 的顺序一致。
- 第 313–314 行：跳过"没有进展"的 shard——`safe_durable_sequences[shard]` 没变化就不重复入队。
- 第 316–323 行：合并优化。如果队尾已经是同一个 barrier（即本轮 maintenance tick 没有新查询提交），就直接把队尾的 `maintenance_sequence` 抬到新值。否则 push 一个新条目。
- 第 324 行：`enqueued_reclaim_ack_sequences[shard]` 记录"已入队的最大 sequence"，避免下一轮重复入队同一个 sequence。

`PendingStorageReclaimAck` 的两个字段（见 `impl.hh:87`）就是 RCU 宽限期的全部状态：

```cpp
// src/gpu_search/persistent_engine/impl.hh:87
struct PendingStorageReclaimAck {
  u64 maintenance_sequence{};
  u64 query_ticket_barrier{};
};
```

含义："等到所有 `ticket ≤ query_ticket_barrier` 的查询都退出后，就可以把 `maintenance_sequence` 作为 ACK 写回远端。"

### 9. `publish_ready_storage_reclaim_acks` —— 宽限期到了就发 ACK

```cpp
// storage_reclaim.cc:328
void PersistentSearchEngine::Impl::publish_ready_storage_reclaim_acks() {
  if (!healthy.load(std::memory_order_acquire)) return;
  if (!retired_resident_pq_batches.empty()) return;
  std::vector<u64> targets = published_reclaim_ack_sequences;
  bool advanced = false;
  for (size_t shard = 0; shard < pending_storage_reclaim_acks.size(); ++shard) {
    auto& queue = pending_storage_reclaim_acks[shard];
    while (!queue.empty() &&
           query_ticket_barrier_passed(queue.front().query_ticket_barrier)) {
      targets[shard] = queue.front().maintenance_sequence;
      queue.pop_front();
      advanced = true;
    }
  }
  if (!advanced) return;
  write_storage_reclaim_acks(targets);
  published_reclaim_ack_sequences = std::move(targets);
  engine.telemetry_.storage_reclaim_ack_writes.fetch_add(
    1, std::memory_order_relaxed);
  engine.telemetry_.storage_reclaim_ack_sequence.store(
    *std::min_element(published_reclaim_ack_sequences.begin(),
                      published_reclaim_ack_sequences.end()),
    std::memory_order_relaxed);
}
```

这是 RCU 宽限期的"检查并推进"步骤：

- 第 329 行：`healthy` 是 fail-stop 的总闸（见第 9 节）。一旦 `mark_unhealthy` 置位，所有 ACK 推进立刻停止——这是为了避免在引擎已经处于不可恢复状态时还往远端写"安全"信号。
- 第 330 行：`retired_resident_pq_batches` 非空时直接返回。这条限制很微妙：常驻 PQ 的物理退休（resident_pq_erase）和远端 ACK 必须按"先本地退休、后远端 ACK"的顺序推进。如果本地还有 PQ 退休批次没消费完，说明上一轮的 retire 工作还没做完，此时发新 ACK 可能让存储侧过早复用一个本地 GPU 上还在被擦除的常驻 PQ 槽对应的远端地址。这是一道额外保护。
- 第 333–340 行：遍历每个 shard 的 pending 队列，只要队首的 `query_ticket_barrier` 已经 passed（即没有任何在飞查询的 ticket ≤ barrier），就把它的 `maintenance_sequence` 提升为新的 ACK 目标。`while` 循环是因为多个连续 barrier 可能同时满足。
- 第 342–343 行：只有真的推进了才写远端——避免无意义的 RDMA 写。
- 第 344 行：`published_reclaim_ack_sequences` 用新值覆盖。
- 第 345–350 行：遥测。`storage_reclaim_ack_writes` 记写次数（一次写覆盖所有 shard），`storage_reclaim_ack_sequence` 取所有 shard 的**最小值**——因为存储侧的 `acquire` 也是取 min，最小值反映的是"集群角度的 RCU 进度"。

`query_ticket_barrier_passed` 的实现在 `delta_publication.cc:282`：

```cpp
// src/gpu_search/persistent_engine/delta_publication.cc:282
bool PersistentSearchEngine::Impl::query_ticket_barrier_passed(u64 barrier) const {
  for (u32 slot = 0; slot < query_slots; ++slot) {
    const u64 ticket = active_query_tickets[slot].load(std::memory_order_acquire);
    if (ticket != 0 && ticket <= barrier) return false;
  }
  return true;
}
```

这是 RCU 宽限期的核心：扫描所有查询槽位（`query_slots` 个），只要有一个槽位上挂着一个 ticket ≤ barrier 的活跃查询，就返回 false。`active_query_tickets` 是一个 `std::atomic<u64>[]`（`impl.hh:403`），查询线程进入时把自己的 ticket 写入分配的 slot，退出时清零。`acquire` 序配对查询线程的 `release` 序，保证"看到 ticket=0 即该查询已经不再读任何旧版本"。

注意 `barrier = next_query_ticket - 1` 的设计：后续新提交的查询 ticket 都 > barrier，它们读到的会是新版本（因为新版本已经通过 `publish_barrier` 可见），所以不需要等它们。这正是 RCU "grace period" 的精髓。

### 10. `retire_durable_delta` —— durable watermark 推进

```cpp
// storage_reclaim.cc:353
std::vector<DeltaMutation> PersistentSearchEngine::Impl::retire_durable_delta() {
  if (control_bootstrapper == nullptr || index.shards.empty()) return {};
  const std::vector<format::StorageControlBlock> controls =
    read_storage_controls();
  if (durable_sequence_history.size() != index.shards.size()) {
    durable_sequence_history.resize(index.shards.size());
    observed_durable_sequences.assign(index.shards.size(), 0);
    safe_durable_sequences.assign(index.shards.size(), 0);
  }
  const auto now = std::chrono::steady_clock::now();
  const auto visibility_grace =
    std::chrono::microseconds(config.update_visibility_us);
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    const auto& control = controls[shard];
    if (control.durable_maintenance_sequence > observed_durable_sequences[shard]) {
      observed_durable_sequences[shard] = control.durable_maintenance_sequence;
      durable_sequence_history[shard].emplace_back(
        control.durable_maintenance_sequence, now);
    }
    auto& history = durable_sequence_history[shard];
    while (!history.empty() && now - history.front().second >= visibility_grace) {
      safe_durable_sequences[shard] = history.front().first;
      history.pop_front();
    }
  }
  return engine.delta_.retire_durable(
    safe_durable_sequences, delta_command_capacity);
}
```

这是把"存储侧 durable watermark"翻译成"计算侧安全 watermark"的关键函数。它实现了一个**带可见性宽限期的 watermark 推进**：

1. 第 357 行：读所有 shard 的 control block（含 `durable_maintenance_sequence`）。
2. 第 367–371 行：如果某个 shard 的 durable sequence 涨了，把 `(sequence, now)` 入 `durable_sequence_history[shard]` 这个 deque。
3. 第 373–376 行：从队首开始 pop 那些"已经等待了 `visibility_grace` 时间"的历史条目，每 pop 一个就把 `safe_durable_sequences[shard]` 抬到它的 sequence。

为什么要 `visibility_grace`？因为查询线程从"读到新 control block"到"实际开始用新版本"之间有一段窗口——查询线程可能在 `read_storage_controls` 之后被调度走，几微秒后才真正发起 RDMA 读。如果计算侧一看到 `durable_maintenance_sequence = N` 就立刻认为 N 之前的所有地址都可回收，那些还在用 N 之前快照的查询就会被存储侧的地址复用打穿。`config.update_visibility_us` 就是这段"等查询线程把新 watermark 真正用上"的宽限期，通常和查询的 visibility window 对齐（见第 10 课 delta_index 的可见性窗口）。

注意 `durable_sequence_history` 是 `std::deque<std::pair<u64, time_point>>` 的 per-shard vector（`impl.hh:357`），允许存储侧一次性跳多个 sequence（比如 maintenance worker 批量 durable 了一批 mutation），计算侧会逐个按时间窗口释放。

第 378–379 行：把 `safe_durable_sequences`（per-shard vector）传给 `DeltaCoordinator::retire_durable`。这个函数在 `delta_index.cc:106` 实现：

```cpp
// src/gpu_search/delta_index.cc:106
std::vector<DeltaMutation> DeltaCoordinator::retire_durable(
    std::span<const u64> durable_sequences, size_t max_items) {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  std::vector<DeltaMutation> retired;
  if (max_items == 0 || durable_sequences.empty() ||
      durable_candidates_.empty()) return retired;
  retired.reserve(std::min(max_items, delta_.size()));
  const size_t owner_count = std::min(
    durable_sequences.size(), durable_candidates_.size());
  const size_t first_owner = durable_owner_cursor_ % owner_count;
  for (size_t offset = 0;
       offset < owner_count && retired.size() < max_items; ++offset) {
    const size_t owner = (first_owner + offset) % owner_count;
    DurableQueue& candidates = durable_candidates_[owner];
    const u64 durable_sequence = durable_sequences[owner];
    while (!candidates.empty() && retired.size() < max_items &&
           candidates.top().maintenance_sequence <= durable_sequence) {
      const DurableCandidate candidate = candidates.top();
      candidates.pop();
      const auto mutation_iterator = delta_.find(candidate.id);
      if (mutation_iterator == delta_.end()) continue;
      DeltaMutation& mutation = mutation_iterator->second;
      if (mutation.durable || mutation.owner_storage != owner ||
          mutation.maintenance_sequence != candidate.maintenance_sequence ||
          mutation.epoch != candidate.epoch ||
          mutation.generation != candidate.generation) {
        continue;
      }
      mutation.durable = true;
      const auto version = versions_.find(mutation.id);
      if (version != versions_.end() &&
          version->second.epoch <= mutation.epoch) {
        version->second.in_delta = false;
      }
      retired.push_back(std::move(mutation));
      delta_.erase(mutation_iterator);
    }
  }
  durable_owner_cursor_ = (first_owner + 1) % owner_count;
  return retired;
}
```

逻辑：

- `durable_candidates_` 是 per-owner-storage 的优先队列（`delta_index.hh:90`），按 `maintenance_sequence` 升序。每次 `publish_impl` 时（`delta_index.cc:64`），如果 mutation 带 `maintenance_sequence != 0`，就 push 进对应 owner 的队列。
- `retire_durable` 用 `durable_owner_cursor_` 轮询所有 owner，避免一个 owner 独占 retire 预算。每轮从 `first_owner` 开始，绕一圈，每个 owner 弹出所有 `maintenance_sequence <= durable_sequence` 的 candidate。
- 第 128–133 行的 5 项校验：candidate 可能已经过期（mutation 被后续 update 覆盖、generation 升级等），任何不一致都跳过。
- 第 134 行 `mutation.durable = true` + 第 135–139 行 `version->second.in_delta = false`：把这条 mutation 从"在 delta 里"标记为"已经 durable"。
- 第 140–141 行：move 出来返回给 caller，同时从 `delta_` erase。

返回的 `retired` vector 会进入 `maintenance_loop` 的 `pending_durable_retirements` multimap，再经 `mark_durable_delta_records_locked` 退休到 GPU。

### 11. `mark_durable_delta_records_locked` —— 退休 GPU L0 记录与常驻 PQ 槽

这是本课最长、最关键的函数。它把 durable 的 mutation 翻译成 GPU 端的"打标记 + 延迟回收"操作。

```cpp
// storage_reclaim.cc:382
void PersistentSearchEngine::Impl::mark_durable_delta_records_locked(
    std::span<const DurableRetirement> retired) {
  std::vector<DeltaDurableUpdate> updates;
  std::vector<u32> retiring_slots;
  std::vector<ResidentPqEraseUpdate> retiring_resident_pq;
  std::unordered_set<u64> retained_resident_pq;
  retained_resident_pq.reserve(retired.size());
  for (const DurableRetirement& mutation : retired) {
    if (mutation.kind != service::storage_owner::MutationKind::erase &&
        mutation.remote_node != 0) {
      retained_resident_pq.insert(mutation.remote_node);
    }
```

第一个 pass：收集"被保留的 remote_node"。一条 durable mutation 如果不是 erase 且指向一个新 remote_node，那么这个 remote_node 对应的常驻 PQ 槽**不能**被回收——因为新版本的向量还活在那个地址上。`retained_resident_pq` 会在后面用来过滤。

```cpp
// storage_reclaim.cc:394
    if (mutation.old_remote_node != 0 &&
        mutation.old_remote_node != mutation.remote_node) {
      const auto resident = resident_pq_slots_by_remote.find(
        mutation.old_remote_node);
      if (resident != resident_pq_slots_by_remote.end()) {
        retiring_resident_pq.push_back(ResidentPqEraseUpdate{
          .remote_node = mutation.old_remote_node,
          .slot = resident->second,
        });
      }
    }
```

第二条 pass：如果 mutation 替换了 remote_node（update 场景，`old_remote_node != 0` 且不等于新 `remote_node`），那么 `old_remote_node` 对应的常驻 PQ 槽要被退休。`resident_pq_slots_by_remote` 是 `unordered_map<u64, u32>`（`impl.hh:353`），记录每个 remote_node 在常驻 PQ 表里的 slot。

```cpp
// storage_reclaim.cc:405
    std::vector<u32> retained_superseded;
    const auto superseded = superseded_delta_slots.find(mutation.id);
    if (superseded != superseded_delta_slots.end()) {
      retained_superseded.reserve(superseded->second.size());
      for (u32 slot : superseded->second) {
        if (slot < delta_records_host.size() &&
            delta_records_host[slot].epoch <= mutation.epoch) {
          retiring_slots.push_back(slot);
        } else {
          retained_superseded.push_back(slot);
        }
      }
      if (retained_superseded.empty()) {
        superseded_delta_slots.erase(superseded);
      } else {
        superseded->second = std::move(retained_superseded);
      }
    }
```

第三条 pass：处理"被这条 mutation 覆盖的旧 delta slot"。`superseded_delta_slots` 是 `unordered_map<node_t, std::vector<u32>>`（`impl.hh:348`），记录每个 node id 上被新版本覆盖但还没回收的 delta slot。只有 `epoch <= mutation.epoch` 的旧 slot 才能被退休——更新版本的 slot 保留。

```cpp
// storage_reclaim.cc:423
    const auto latest = latest_delta_slot.find(mutation.id);
    if (latest != latest_delta_slot.end() &&
        latest->second < delta_records_host.size() &&
        delta_records_host[latest_delta_slot[mutation.id]].epoch <= mutation.epoch) {
      DeviceDeltaRecord& record = delta_records_host[latest->second];
      if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
          record.superseded_epoch == 0) {
        if (mutable_delta_entries == 0) {
          throw std::runtime_error("GPU mutable delta accounting underflow");
        }
        --mutable_delta_entries;
      }
      retiring_slots.push_back(latest->second);
      latest_delta_slot.erase(latest);
    }
  }
```

第四条 pass：处理"当前最新的 delta slot"。`latest_delta_slot` 是 `unordered_map<node_t, u32>`（`impl.hh:352`），每个 node id 当前在 GPU 上的活跃 delta slot。如果它的 epoch ≤ mutation.epoch，说明这条 durable mutation 已经覆盖了它，可以退休：

- 第 428–433 行：如果这条 record 既没删除也没 durable、且没被 superseded（`superseded_epoch == 0`），那它是一条"mutable"delta 记录，退休时要减少 `mutable_delta_entries` 计数。`mutable_delta_entries == 0` 时抛异常——这是 fail-stop 的会计保护。
- 第 435 行：push 进 retiring_slots。
- 第 436 行：从 `latest_delta_slot` erase。

```cpp
// storage_reclaim.cc:439
  std::sort(retiring_slots.begin(), retiring_slots.end());
  retiring_slots.erase(
    std::unique(retiring_slots.begin(), retiring_slots.end()),
    retiring_slots.end());
```

去重：同一条 slot 可能从 `superseded_delta_slots` 和 `latest_delta_slot` 两条路径都被加入，sort + unique 去重。

```cpp
// storage_reclaim.cc:443
  for (u32 slot : retiring_slots) {
    const u64 remote_node = delta_records_host[slot].remote_node;
    if (remote_node == 0 || retained_resident_pq.contains(remote_node)) continue;
    const auto resident = resident_pq_slots_by_remote.find(remote_node);
    if (resident != resident_pq_slots_by_remote.end()) {
      retiring_resident_pq.push_back(ResidentPqEraseUpdate{
        .remote_node = remote_node,
        .slot = resident->second,
      });
    }
  }
```

第五条 pass：扫描所有 retiring slot，如果某个 slot 的 remote_node 在常驻 PQ 表里且**没有**被 retained_resident_pq 保留（即这条 durable mutation 是 erase 或者 remote_node 变了），那么对应的常驻 PQ 槽也要退休。这是"delta 记录退休 → 物理常驻 PQ 退休"的级联。

```cpp
// storage_reclaim.cc:454
  std::sort(retiring_resident_pq.begin(), retiring_resident_pq.end(),
            [](const ResidentPqEraseUpdate& lhs,
               const ResidentPqEraseUpdate& rhs) {
              if (lhs.remote_node != rhs.remote_node) {
                return lhs.remote_node < rhs.remote_node;
              }
              return lhs.slot < rhs.slot;
            });
  retiring_resident_pq.erase(
    std::unique(retiring_resident_pq.begin(), retiring_resident_pq.end(),
                [](const ResidentPqEraseUpdate& lhs,
                   const ResidentPqEraseUpdate& rhs) {
                  return lhs.remote_node == rhs.remote_node &&
                    lhs.slot == rhs.slot;
                }),
    retiring_resident_pq.end());
```

对 `retiring_resident_pq` 按 `(remote_node, slot)` 排序去重——同一个 remote_node 可能从 update 路径和 delta-slot 路径都被加入。

```cpp
// storage_reclaim.cc:470
  updates.reserve(retiring_slots.size());
  for (u32 slot : retiring_slots) {
    DeviceDeltaRecord& record = delta_records_host[slot];
    updates.push_back(DeltaDurableUpdate{
      .slot = slot,
      .epoch = record.epoch,
    });
  }
  for (size_t begin = 0; begin < updates.size(); begin += delta_command_capacity) {
    const size_t count = std::min<size_t>(
      delta_command_capacity, updates.size() - begin);
    std::memcpy(delta_durable_updates_host, updates.data() + begin,
                count * sizeof(DeltaDurableUpdate));
    const u32 live_count = static_cast<u32>(delta_records_host.size());
    submit_delta_publication(DeltaPublishDescriptor{
      .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
      .final_count = live_count,
      .durable_count = static_cast<u32>(count),
    });
  }
```

构造 `DeltaDurableUpdate` 批量（每条带 `slot` 和 `epoch`），分批发给 GPU。每批最多 `delta_command_capacity` 条——这是 kernel 一次能处理的 delta 命令上限（见第 17 课 kernel 启动器）。`DeltaPublishDescriptor.durable_count` 告诉 kernel 这次要处理多少条 durable 标记。kernel 侧会把对应 slot 的 `DeviceDeltaRecord.flags` 置上 `kDeltaDurable`（见 `persistent_kernel.hh:28`）。

注意第 483 行 `live_count = delta_records_host.size()`——这个数字包含已经退休但还没物理回收的 slot，所以叫 `final_count` 而不是 `active_count`。kernel 用它来知道 delta 表的物理大小。

```cpp
// storage_reclaim.cc:490
  if (!retiring_slots.empty() || !retiring_resident_pq.empty()) {
    for (u32 slot : retiring_slots) {
      DeviceDeltaRecord& record = delta_records_host[slot];
      record.flags |= kDeltaDurable;
      if (record.superseded_epoch == 0) record.superseded_epoch = record.epoch;
      if (record.base_ordinal != kBaseOverrideEmpty) {
        const auto override = base_override_epochs.find(record.base_ordinal);
        if (override != base_override_epochs.end() &&
            override->second <= record.epoch) {
          base_override_epochs.erase(override);
        }
      }
    }
```

本地 host 端把对应 slot 的 `flags` 置上 `kDeltaDurable`，并把 `superseded_epoch` 填上（如果还没填过）。`base_override_epochs` 是 `unordered_map<u32, u64>`（`impl.hh:356`），记录每个 base override ordinal 的 epoch；如果 override 的 epoch ≤ record.epoch，说明这个 override 已经被 durable 覆盖，可以 erase。

```cpp
// storage_reclaim.cc:503
    std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
    const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
    retired_delta_batches.push_back(RetiredDeltaBatch{
      .query_ticket_barrier = barrier,
      .slots = std::move(retiring_slots),
    });
    if (!retiring_resident_pq.empty()) {
      retired_resident_pq_batches.push_back(RetiredResidentPqBatch{
        .query_ticket_barrier = barrier,
        .entries = std::move(retiring_resident_pq),
      });
    }
    reclaim_retired_delta_slots_locked();
  }
```

注册 RCU 宽限期！这里和 `enqueue_storage_reclaim_barriers` 是同一思路：锁 `query_snapshot_mutex`，抓当前 `next_query_ticket - 1` 作为 barrier，把退休的 slot/entries 挂到 `retired_delta_batches` / `retired_resident_pq_batches` 队列（`impl.hh:349`、`impl.hh:350`）。然后立即调 `reclaim_retired_delta_slots_locked()` 尝试回收——如果宽限期已过（没有在飞查询），就直接回收。

```cpp
// storage_reclaim.cc:517
  engine.telemetry_.delta_mutable_entries.store(
    mutable_delta_entries, std::memory_order_relaxed);
  engine.telemetry_.delta_durable_entries.store(
    durable_delta_entries, std::memory_order_relaxed);
  engine.telemetry_.delta_entries_retired.fetch_add(
    updates.size(), std::memory_order_relaxed);
}
```

更新遥测：`delta_mutable_entries`（还在变的 delta 数）、`delta_durable_entries`（已经 durable 但还没物理回收的 delta 数）、`delta_entries_retired`（累计退休的 delta 条数，单调增）。

### 12. `reclaim_retired_delta_slots_locked` —— 真正的物理回收

这个函数在 `delta_publication.cc:301`，但它和本课强相关，必须讲。

```cpp
// src/gpu_search/persistent_engine/delta_publication.cc:301
void PersistentSearchEngine::Impl::reclaim_retired_delta_slots_locked() {
  u64 reclaimed = 0;
  while (!retired_delta_batches.empty() &&
         query_ticket_barrier_passed(
           retired_delta_batches.front().query_ticket_barrier)) {
    RetiredDeltaBatch batch = std::move(retired_delta_batches.front());
    retired_delta_batches.pop_front();
    reclaimed += batch.slots.size();
    free_delta_slots.insert(free_delta_slots.end(),
                            batch.slots.begin(), batch.slots.end());
  }
```

第一个 while：弹出所有已经度过宽限期的 delta 批次，把 slot 还回 `free_delta_slots`——这些 slot 现在可以被新 mutation 复用了。这是 GPU L0 delta 表的物理回收。

```cpp
// delta_publication.cc:312
  u64 resident_pq_reclaimed = 0;
  while (!retired_resident_pq_batches.empty() &&
         query_ticket_barrier_passed(
           retired_resident_pq_batches.front().query_ticket_barrier)) {
    RetiredResidentPqBatch batch =
      std::move(retired_resident_pq_batches.front());
    retired_resident_pq_batches.pop_front();
    for (size_t begin = 0; begin < batch.entries.size();
         begin += delta_command_capacity) {
      const size_t count = std::min<size_t>(
        delta_command_capacity, batch.entries.size() - begin);
      std::memcpy(resident_pq_erase_updates_host,
                  batch.entries.data() + begin,
                  count * sizeof(ResidentPqEraseUpdate));
      submit_delta_publication(DeltaPublishDescriptor{
        .command_id = next_delta_command_id.fetch_add(
          1, std::memory_order_relaxed),
        .final_count = static_cast<u32>(delta_records_host.size()),
        .resident_pq_erase_count = static_cast<u32>(count),
      });
      for (size_t index = 0; index < count; ++index) {
        const ResidentPqEraseUpdate& update = batch.entries[begin + index];
        const auto resident = resident_pq_slots_by_remote.find(
          update.remote_node);
        if (resident == resident_pq_slots_by_remote.end() ||
            resident->second != update.slot) {
          continue;
        }
        resident_pq_slots_by_remote.erase(resident);
        free_resident_pq_slots.push_back(update.slot);
        ++resident_pq_reclaimed;
      }
    }
  }
```

第二个 while：常驻 PQ 槽的回收。每批最多 `delta_command_capacity` 条 `ResidentPqEraseUpdate`，发 `DeltaPublishDescriptor.resident_pq_erase_count` 给 kernel——kernel 会把 GPU 上 `d_resident_pq_codes`/`d_resident_pq_keys`/`d_resident_pq_slots`/`d_resident_pq_positions` 里对应 slot 清掉。同时 host 端从 `resident_pq_slots_by_remote` erase 并把 slot 还回 `free_resident_pq_slots`。

注意第 336–339 行的乐观校验：如果在等待宽限期的过程中，这个 slot 已经被新 mutation 复用（`resident->second != update.slot`），就跳过——这是正常的，因为 `free_resident_pq_slots` 已经在上一轮回收时被新 mutation 拿走了。

```cpp
// delta_publication.cc:346
  if (reclaimed != 0) {
    engine.telemetry_.delta_reclaim_batches.fetch_add(1, std::memory_order_relaxed);
  }
  if (resident_pq_reclaimed != 0) {
    engine.telemetry_.resident_pq_reclaimed.fetch_add(
      resident_pq_reclaimed, std::memory_order_relaxed);
  }
  engine.telemetry_.delta_physical_entries.store(
    active_delta_slots_locked(), std::memory_order_relaxed);
  engine.telemetry_.resident_pq_entries.store(
    active_resident_pq_slots_locked(), std::memory_order_relaxed);
}
```

遥测：`delta_reclaim_batches`（物理回收批次数）、`resident_pq_reclaimed`（累计回收的常驻 PQ 槽数）、`delta_physical_entries`/`resident_pq_entries`（当前活跃数）。`resident_pq_capacity`/`resident_pq_peak_entries` 在别处更新（见 `persistent_engine.cc` 的 `allocate_resident_pq_slot_locked`）。

### 13. `maintenance_loop` —— 把所有步骤串起来

```cpp
// storage_reclaim.cc:525
void PersistentSearchEngine::Impl::maintenance_loop() {
  bind_cuda_device("cudaSetDevice(GPU navigation maintenance)");
  const auto period = std::chrono::milliseconds(std::max<u32>(
    1, std::min<u32>(config.gpu_delta_maintenance_period_ms,
                     std::max<u32>(1, config.update_visibility_us / 1000))));
  while (!maintenance_shutdown.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(maintenance_mutex);
      maintenance_cv.wait_for(lock, period, [&] { return maintenance_shutdown.load(); });
    }
    if (maintenance_shutdown.load()) break;
    std::vector<DeltaMutation> retired;
    try {
      std::lock_guard<std::mutex> publish_lock(engine.mutation_publish_mutex_);
      retired = retire_durable_delta();
```

maintenance 线程的 tick 周期是 `gpu_delta_maintenance_period_ms`，但被 `update_visibility_us / 1000` 限制——后者是可见性窗口，前者不能比它大太多，否则 RCU 宽限期推进太慢。`std::max<u32>(1, ...)` 保证至少 1ms。

每个 tick：

1. 锁 `mutation_publish_mutex_`——这把锁串行化 maintenance 和前台 mutation publish（见第 14 课查询执行/路由/完成）。
2. `retire_durable_delta()` 读 control block，推进 `safe_durable_sequences`，调 `DeltaCoordinator::retire_durable` 拿到 durable 的 mutation list。

```cpp
// storage_reclaim.cc:540
      for (const DeltaMutation& mutation : retired) {
        pending_durable_retirements.emplace(
          mutation.epoch,
          DurableRetirement{
            .id = mutation.id,
            .kind = mutation.kind,
            .epoch = mutation.epoch,
            .remote_node = mutation.remote_node,
            .old_remote_node = mutation.old_remote_node,
          });
      }
```

把 retired mutation 转成 `DurableRetirement` 放进 `pending_durable_retirements`（`impl.hh:351`，`std::multimap<u64, DurableRetirement>`，按 epoch 排序）。multimap 而非 vector 是因为要按 epoch 顺序消费——后面 `durable_snapshot_safe` 检查。

```cpp
// storage_reclaim.cc:551
      std::vector<DurableRetirement> snapshot_safe;
      snapshot_safe.reserve(std::min<size_t>(
        delta_command_capacity, pending_durable_retirements.size()));
      {
        std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
        while (!pending_durable_retirements.empty() &&
               snapshot_safe.size() < delta_command_capacity) {
          auto oldest = pending_durable_retirements.begin();
          if (!durable_snapshot_safe(oldest->first)) break;
          snapshot_safe.push_back(std::move(oldest->second));
          pending_durable_retirements.erase(oldest);
        }
      }
```

从 multimap 队首开始消费，每条检查 `durable_snapshot_safe(epoch)`——这是 RCU 的另一道保护。`durable_snapshot_safe` 在 `delta_publication.cc:290`：

```cpp
// src/gpu_search/persistent_engine/delta_publication.cc:290
bool PersistentSearchEngine::Impl::durable_snapshot_safe(u64 durable_epoch) const {
  for (u32 slot = 0; slot < query_slots; ++slot) {
    const u64 encoded_snapshot =
      active_query_snapshots[slot].load(std::memory_order_acquire);
    if (encoded_snapshot != 0 && encoded_snapshot - 1 < durable_epoch) {
      return false;
    }
  }
  return true;
}
```

`active_query_snapshots[slot]` 存的是查询线程进入时观察到的 `published_epoch_ + 1`（编码方式：0 表示空槽，非 0 值减 1 才是真实 epoch）。如果某查询的 snapshot epoch < `durable_epoch`，说明它进入时还没看到这条 durable mutation 对应的新版本——它可能还在用旧版本，所以不能 retire。这是"可见性窗口"的硬件级实现：查询线程用 `release` 序写 `active_query_snapshots`，maintenance 线程用 `acquire` 序读，构成 happens-before。

```cpp
// storage_reclaim.cc:564
      {
        std::lock_guard<std::mutex> delta_lock(delta_mutex);
        if (!snapshot_safe.empty()) {
          mark_durable_delta_records_locked(snapshot_safe);
        }
        reclaim_retired_delta_slots_locked();
      }
```

锁 `delta_mutex`，调 `mark_durable_delta_records_locked`（第 11 节讲过），然后立刻调 `reclaim_retired_delta_slots_locked` 尝试物理回收上一轮（甚至本轮）注册的退休批次。`delta_mutex` 是所有 delta 表操作的总锁，和前台 `upload_mutations` 互斥。

```cpp
// storage_reclaim.cc:571
      // A reclaim ACK may allow storage to reuse a dynamic address. Publish
      // the canonical route tombstone/replacement first, then capture a query
      // ticket barrier that covers every query which could have read the old
      // route. A torn route snapshot therefore advances neither the barrier
      // nor the remote ACK.
      if (synchronize_storage_routes()) {
        enqueue_storage_reclaim_barriers();
        publish_ready_storage_reclaim_acks();
      }
```

最后一步——但**只在路由快照成功推进时**才做。注释解释了为什么顺序是"路由 → barrier → ACK"：ACK 一写回去，存储就可能复用某个动态地址；在 ACK 之前必须先让所有查询看到新路由（tombstone 或替换），否则一个查询可能拿着旧路由去读已经被复用的地址。`synchronize_storage_routes()` 返回 false 时（路由快照撕裂或没变化），整个 ACK 推进跳过——下一 tick 再试。

```cpp
// storage_reclaim.cc:580
    } catch (const std::exception& error) {
      mark_unhealthy(std::string{"storage maintenance watermark failed: "} + error.what());
      break;
    }
  }
}
```

任何异常都 `mark_unhealthy` 并退出循环——这是 fail-stop。maintenance 线程一旦退出，`safe_durable_sequences` 不再推进，`storage_reclaim_ack_sequence` 停在最后值，存储侧永远不会复用任何该计算节点之后才能回收的地址。系统进入"读安全但写不进"的降级状态，等待运维介入。

### 14. `mutation_capacity` —— 前台写入 vs maintenance worker 的反饥饿

虽然这部分的代码在 `persistent_engine.cc` 而非 `storage_reclaim.cc`，但它是 RCU 链条的"反压"机制，必须讲。

```cpp
// src/gpu_search/persistent_engine.cc:150
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
  const u64 reserved = static_cast<u64>(impl_->reserved_mutation_capacity);
  telemetry_.mutation_capacity_reserved.store(reserved, std::memory_order_relaxed);
  u64 current_max = telemetry_.mutation_capacity_reserved_max.load(
    std::memory_order_relaxed);
  while (current_max < reserved &&
         !telemetry_.mutation_capacity_reserved_max.compare_exchange_weak(
           current_max, reserved, std::memory_order_relaxed)) {}
  return true;
}
```

`try_reserve_mutation_capacity` 是非阻塞版本。它检查两个表的水位：

- delta 表：`delta_capacity * 90%`（hard watermark）
- 常驻 PQ 表：`resident_pq_capacity * 95%`（hard watermark）

`reserved_mutation_capacity` 是"已预订但还没真正写入"的额度——前台 RPC 在拿到 mutation 之前先预订，写完再 release。这个额度确保即使 100 个并发 RPC 同时到达，也不会超出 hard watermark。

公式 `active_slots > hard_watermark - mutation_count` 等价于 `active_slots + mutation_count > hard_watermark`，但用减法形式避免无符号下溢。`reserved_mutation_capacity > hard_watermark - mutation_count - active_slots` 等价于 `active_slots + mutation_count + reserved_mutation_capacity > hard_watermark`——同时考虑已预订额度。

拒绝时 `mutation_capacity_rejections++`。`reserve_mutation_capacity`（阻塞版本，第 181 行）则用 `delta_capacity_cv.wait_for(lock, 1ms)` 等 maintenance 线程回收出空间。注释（第 229 行）解释了为什么用 1ms 超时而不是无限等：

```cpp
// src/gpu_search/persistent_engine.cc:229
    // Publication releases reservations and notifies directly. The bounded
    // wait also rechecks capacity reclaimed by the independent maintenance
    // thread, which does not need to know about submitters.
```

maintenance 线程不知道有谁在等，但它回收后会调 `reclaim_retired_delta_slots_locked`，间接让 `active_slots` 下降。1ms 超时让前台线程定期重检，避免错过 maintenance 的回收。

`release_mutation_capacity`（第 236 行）在 mutation 写完后调用：

```cpp
// src/gpu_search/persistent_engine.cc:236
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

会计下溢（release 比 reserve 多）直接 `mark_unhealthy`——这是 bug 的信号，不能继续。`notify_all` 唤醒所有等容量的前台线程。`std::memory_order_release` 配对 `Telemetry::snapshot` 的 `acquire`（`types.cc:10`）——保证快照读到的 reservation 值之后的代码能看到所有 publisher 完成的遥测更新。

### 15. fail-stop：`mark_unhealthy`

`mark_unhealthy` 在 `impl.hh:111` 声明，实现在 `persistent_engine.cc`。它做三件事：

1. `healthy.store(false, acquire)`——让 `publish_ready_storage_reclaim_acks` 第 329 行的检查立即生效，停止 ACK 推进。
2. 写入 `health_error` 字符串。
3. 唤醒所有等条件变量的线程，让它们看到 unhealthy 后退出。

本课中触发 `mark_unhealthy` 的路径：

- `storage_reclaim.cc:581`：`maintenance_loop` 捕获任何异常（control block 读失败、ACK 写失败、retire 会计下溢等）。
- `persistent_engine.cc:240`：`release_mutation_capacity` 会计下溢。

注意 `mark_durable_delta_records_locked` 第 431 行的 `throw std::runtime_error("GPU mutable delta accounting underflow")` 会被 `maintenance_loop` 的 catch 捕获，转成 `mark_unhealthy`。所有会计问题都 fail-stop，绝不带病运行——因为 RCU 协议一旦会计错乱，可能让存储过早复用地址，造成查询读到脏数据。

## 关键数据结构 / 流程图

### 数据结构总览

| 名字 | 类型 | 位置 | 作用 |
|---|---|---|---|
| `StorageControlBlock` | struct (640B, alignas 64) | `index_format.hh:80` | schema-15 control page v2，含 `durable_maintenance_sequence` 和 `reclaim_ack_sequences[64]` |
| `StorageReclaimQueue` | class | `storage_reclaim.hh:15` | 存储侧回收队列，`pending_` map + `ready_` vector |
| `PendingStorageReclaimAck` | struct | `impl.hh:87` | 计算侧待发布的 ACK 条目，`(maintenance_sequence, query_ticket_barrier)` |
| `RetiredDeltaBatch` / `RetiredResidentPqBatch` | struct | `impl.hh:77` / `impl.hh:82` | 已退休但等宽限期的 GPU L0/PQ 批次 |
| `DurableRetirement` | struct | `impl.hh:92` | 从 `DeltaCoordinator` retire 出来的 durable mutation |
| `DeltaDurableUpdate` / `ResidentPqEraseUpdate` | struct | `persistent_kernel.hh` | 发给 kernel 的退休命令 |
| `DeviceDeltaRecord` | struct | `persistent_kernel.hh:58` | GPU 上每条 delta 记录，`flags` 含 `kDeltaDurable` |
| `durable_sequence_history` | `vector<deque<pair<u64,time_point>>>` | `impl.hh:357` | per-shard 的 watermark 历史，驱动可见性宽限期 |
| `pending_durable_retirements` | `multimap<u64, DurableRetirement>` | `impl.hh:351` | 按 epoch 排序的待退休队列 |
| `pending_storage_reclaim_acks` | `vector<deque<PendingStorageReclaimAck>>` | `impl.hh:361` | per-shard 的待发布 ACK 队列 |

### RCU 状态机 / 时序图

下面是完整的状态机，从存储侧 durable 到存储侧物理回收：

```
[存储侧]                                   [计算侧 maintenance 线程]                [计算侧查询线程]                  [存储侧 acquire]
                                                                                  
  发布 dynamic addr v2 ──┐                                                                                                  
  (maintenance_seq = N)  │                                                                                                  
                         v                                                                                                  
  durable_maintenance_seq                                                                                                   
      = N 写入 control ─────┐                                                                                              
                             │                                                                                              
                             v                                                                                              
                       read_storage_controls()                                                                              
                       (storage_reclaim.cc:45)                                                                              
                             │                                                                                              
                             v                                                                                              
                       durable_sequence_history                                                                              
                       .emplace_back(N, now)                                                                                
                             │                                                                                              
                             │  等 visibility_grace                                                                         
                             v                                                                                              
                       safe_durable_sequences[shard] = N                                                                    
                       (storage_reclaim.cc:374)                                                                              
                             │                                                                                              
                             v                                                                                              
                       DeltaCoordinator::retire_durable()                                                                   
                       (delta_index.cc:106)                                                                                 
                             │                                                                                              
                             v                                                                                              
                       pending_durable_retirements                                                                           
                       (multimap by epoch)                                                                                   
                             │                                                                                              
                             │  durable_snapshot_safe(epoch)? ──── no ──→ 等下一 tick                                       
                             v                                                                                              
                       mark_durable_delta_records_locked()                                                                  
                       (storage_reclaim.cc:382)                                                                             
                             │                                                                                              
                             ├─→ submit DeltaDurableUpdate 给 kernel                                                    
                             ├─→ host 端 flags |= kDeltaDurable                                                           
                             └─→ retired_delta_batches.push_back(                                                           
                                     {barrier=next_ticket-1, slots})                                                        
                                  retired_resident_pq_batches.push_back(...)                                               
                             │                                                                                              
                             v                                                                                              
                       reclaim_retired_delta_slots_locked()                                                                 
                       (delta_publication.cc:301)                                                                           
                             │                                                                                              
                             │  query_ticket_barrier_passed(barrier)? ── no ──→ 等下一 tick                                
                             │           ▲                                                                                  
                             │           │ 查询线程退出时 active_query_tickets[slot]=0 (release)                          
                             │           │                                                                                  
                             │           └── 查询线程进入时 active_query_tickets[slot]=ticket (release)                   
                             v                                                                                              
                       free_delta_slots / free_resident_pq_slots                                                           
                       物理回收完成                                                                                          
                             │                                                                                              
                             v                                                                                              
                       synchronize_storage_routes()                                                                         
                       (storage_reclaim.cc:196)                                                                             
                             │                                                                                              
                             │  路由有变化? ── no ──→ 跳过 ACK 推进                                                         
                             v                                                                                              
                       enqueue_storage_reclaim_barriers()                                                                   
                       (storage_reclaim.cc:309)                                                                             
                             │                                                                                              
                             │  barrier = next_query_ticket - 1                                                           
                             │  pending_storage_reclaim_acks[shard].push_back(                                            
                             │      {maintenance_sequence=N, barrier})                                                     
                             v                                                                                              
                       publish_ready_storage_reclaim_acks()                                                                 
                       (storage_reclaim.cc:328)                                                                             
                             │                                                                                              
                             │  query_ticket_barrier_passed(barrier)? ── no ──→ 等下一 tick                                
                             v                                                                                              
                       write_storage_reclaim_acks(targets)                                                                 
                       (storage_reclaim.cc:267)                                                                             
                             │                                                                                              
                             │  RDMA 写 8B 到 control page                                                                
                             │  reclaim_ack_sequences[client_id] = N                                                      
                             v                                                                                              
  StorageReclaimQueue::acquire(durable_seq, ack_seq_for_client)                                                             
  (storage_reclaim.hh:23)                                                                                                   
                             │                                                                                              
                             │  safe_sequence = min(durable, ack)                                                          
                             │  pending_ 中 seq <= safe_sequence 的指针 → ready_                                          
                             v                                                                                              
  ready_.pop_back() → 复用为新的 dynamic addr v3                                                                             
```

关键不变量：

1. **顺序不变量**：`durable_seq` → `safe_durable_seq` → `retire` → `mark_durable` → `reclaim` → `route_sync` → `barrier_enqueue` → `ack_publish` → 存储复用。每一步都依赖前一步的可见性。
2. **双宽限期不变量**：retire 时一个 `durable_snapshot_safe` 检查 + 一个 `query_ticket_barrier`；ACK 时又一个 `query_ticket_barrier`。两次宽限期保证"没有任何在飞查询读到旧版本"。
3. **路由先于 ACK 不变量**：`if (synchronize_storage_routes())` 是 ACK 的闸门——路由快照撕裂时绝不推进 ACK。
4. **多计算节点木桶不变量**：存储侧 `acquire` 取 `min(durable, ack)`，每计算节点独立 ACK 槽，最慢者决定全局回收速率。

## 与其他模块的关系

- **第 7 课（schema-15 索引格式）**：`StorageControlBlock` 是 schema-15 control page 的运行时结构，version 2 加入了 `reclaim_ack_sequences` 数组。`kStorageControlVersion = 2` 与 `kMetadataSchemaVersion = 15` 是两个独立版本号——前者管 control page layout，后者管整个索引元数据。
- **第 8 课（元数据/owner map/存储协议）**：reclaim ACK 协议是第 8 课存储协议的"回收半边"。`StorageControlBlock` 的 8 字节 ACK 槽 + `StorageReclaimQueue` 的 min(durable, ack) 规则共同定义了跨节点回收契约。
- **第 10 课（delta/动态路由/预算）**：本课的 `safe_durable_sequences` 推进直接消费 `DeltaCoordinator::retire_durable`；`durable_snapshot_safe` 用的 `active_query_snapshots` 就是第 10 课可见性窗口的镜像；`update_visibility_us` 同时驱动可见性窗口和 `durable_sequence_history` 的宽限期。`synchronize_storage_routes` 把第 10 课的动态路由 overlay 推到 GPU，是 ACK 的前置闸门。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`maintenance_thread` 在 `Impl` 析构时通过 `maintenance_shutdown` + `maintenance_cv` 优雅停止；`initialize_storage_reclaim_ack` 在引擎启动时复位 ACK，是 fail-restart 的关键。
- **第 14 课（查询执行/路由/完成）**：`active_query_tickets` / `active_query_snapshots` 由查询执行路径维护（查询进入时写 ticket + snapshot epoch，退出时清零）。`mutation_publish_mutex_` 串行化 maintenance 和前台 publish。
- **第 15 课（增量发布）**：`synchronize_storage_routes` 是第 15 课路由发布的消费者侧；`submit_delta_publication` 的 `durable_count` / `resident_pq_erase_count` / `dynamic_route_count` 三个字段分别驱动本课的三个退休路径。
- **第 17 课（kernel 启动器/上下文/device ring）**：`control_bootstrapper` 是 maintenance 专用的 bootstrap QP，与查询 kernel 用的 data QP 隔离。`delta_command_capacity` 决定每批 retire 命令的上限。
- **第 23 课（存储节点主体/peer RDMA）**：`StorageReclaimQueue` 是存储节点维护线程的回收队列；存储侧在分配新 dynamic addr 时 `acquire`，在 durable 后 `retire` 旧指针。
- **第 26 课（维护/wire protocol）**：存储侧的 maintenance worker 推进 `durable_maintenance_sequence`，触发本课的整个 RCU 链条；wire protocol 上的 control page 读写是本课 RDMA 操作的远端目标。

## 小结

本课讲解的是 dvstor 计算侧的 RCU 回收机制。核心是三段宽限期：

1. **可见性宽限期**（`durable_sequence_history` + `visibility_grace`）：等查询线程真正用上新 watermark。
2. **快照宽限期**（`durable_snapshot_safe` + `active_query_snapshots`）：等所有 snapshot epoch < durable_epoch 的查询退出。
3. **ticket 宽限期**（`query_ticket_barrier_passed` + `active_query_tickets`）：等所有 ticket ≤ barrier 的查询退出，用于 GPU L0/PQ 物理回收和远端 ACK 推进。

三段宽限期叠加，保证"存储侧 durable → 计算侧退休 → 远端 ACK → 存储复用"整条链路上，没有任何查询会读到被复用的旧地址。`StorageControlBlock.reclaim_ack_sequences[client_id]` 是计算侧向存储侧报告"我已经安全"的唯一通道，每计算节点独立，存储侧取 min。schema-15 control block version 2 引入这个 64 槽数组，是多计算节点动态路由的物理基础。

`mutation_capacity` 的 reserve/release 是反饥饿机制——前台 RPC 预订额度，maintenance 线程通过 `reclaim_retired_delta_slots_locked` 回收空间，两者通过 `delta_capacity_cv` 协调。会计下溢直接 `mark_unhealthy`，体现"宁可停服也不读脏数据"的设计哲学。

下一课（第 17 课）将进入 kernel 启动器和 device ring 的世界，本课频繁出现的 `submit_delta_publication`、`delta_command_capacity`、`control_bootstrapper` 都会在那里得到完整的设备侧解释。
