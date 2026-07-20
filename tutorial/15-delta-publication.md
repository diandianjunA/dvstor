# 第 15 课：增量发布（Delta Publication）

## 本课目标

dvstor 把"对索引的一次插入/删除"视作一次协议事件：从存储节点 ACK（stage1 完成）到 GPU 上查询真正看见这条变更，中间要走一段由计算节点驱动、CPU 与常驻 control CTA 协作的"发布流水线"。这段流水线既不能阻塞 GPU 的查询 kernel，也不能让一个尚未完全编码好的 delta 行被正在进行的查询读到。本课要解决的就是这个可见性协议。

本课以 `src/gpu_search/persistent_engine/delta_publication.cc` 为主线，完整讲解：

1. `publish_mutations` 的 6 步协议（epoch 预留 → 批内去重 → queue 时延统计 → GPU 上传 → 可见时延统计 → coordinator publish + telemetry）。
2. `upload_records_locked` 如何在 pinned staging 里把原始 dtype 向量摆好，而**不启动 side kernel、不同步 H2D**，把真正的 OPQ/PQ 编码完全交给常驻 control CTA。
3. `submit_delta_publication` 如何通过 `MappedRing` 把 `DeltaPublishDescriptor` 投递给 GPU 并等回 `DeltaPublishCompletion`，从而做到 0 显式 stream 同步。
4. control CTA（`runtime.cuh`）如何批量完成 OPQ/PQ 编码、更新 delta hash/bucket/override，并以 `__threadfence_system()` + `atomicExch(delta_count)` 的方式原子发布 delta count 与 snapshot epoch。
5. `DeltaPublishDescriptor` 各字段的语义、`kDeltaCommandReset` / `kDeltaCommandPromoteOverrides` 两个专用 flag、route-only command 如何先写 PQ code、再用 device-scope seqlock 发布 `{epoch,pointer,id,generation}`，最后推进 `published_epoch_`。
6. 可见性窗口 `update_visibility_us` 与 `publication_queue_ns_total` / `publication_prepare_ns_total` / `publication_command_ns_total` 三段遥测的来源。

涉及文件（均按行号引用）：

- 主文件：`src/gpu_search/persistent_engine/delta_publication.cc`
- 入口与 6 步协议：`src/gpu_search/persistent_engine.cc:44-148`
- 类型定义：`src/gpu_search/types.hh`、`src/gpu_search/persistent_kernel.hh`
- 协调器：`src/gpu_search/delta_index.hh` / `.cc`
- 私有头：`src/gpu_search/persistent_engine/impl.hh`
- 路由辅助：`src/gpu_search/persistent_engine/routing.cc`
- control CTA 实现：`src/gpu_search/persistent_kernel/runtime.cuh`
- 查询可见性判定：`src/gpu_search/persistent_kernel/candidate_scoring.cuh`
- 查询快照获取：`src/gpu_search/persistent_engine/query_execution.cc:48`
- route-only 发布：`src/gpu_search/persistent_engine/storage_reclaim.cc:229-264`
- stage1 ACK 触发：`src/service/compute_service/storage_owner/completion.cc:108-406`

---

## 一、协议总览：stage1 ACK → 查询可见

先把整条链路画出来，后面逐段拆。存储节点返回的 `InsertBatchResponseHeader` 被 compute_service 的 CQ 线程解析后，会把"已提交的 mutation"塞回 RPC slot 的 `publication_mutations` 缓冲，再由 completion loop 同步地喂给持久化引擎：

```
存储节点                    compute_service CQ 线程          completion loop                PersistentSearchEngine           GPU control CTA
  |  InsertBatchResponse ----> handle_storage_owner_response ----> run_storage_insert_         |                                |
  |                              (匹配 batch_id,                  completion_loop()             |                                |
  |                               queue_storage_owner_completion)  |                           |                                |
  |                                                                v                           |                                |
  |                                                  commit_storage_owner_slot()                |                                |
  |                                                  - 解析 statuses/results                     |                                |
  |                                                  - publish_compute_side_id() (见第28课)       |                                |
  |                                                  - 组装 slot.publication_mutations[]         |                                |
  |                                                                |                           |                                |
  |                                                                v                           |                                |
  |                                                  publish_storage_owner_mutations(slot)       |                                |
  |                                                                |                           |                                |
  |                                                                v                           |                                |
  |                                          persistent_search_->publish_mutations(mutations,   |                                |
  |                                              invalidated_graph_nodes)                         |                                |
  |                                                                                            v                                |
  |                                                  [1] delta_.reserve_epoch()       (CPU)                                       |
  |                                                  [2] 批内 generation 去重                                                  |
  |                                                  [3] publication_queue_ns 统计                                          |
  |                                                  [4] impl_->upload_mutations()                                          |
  |                                                       └─ upload_records_locked()                                       |
  |                                                            ├─ 分配 free_delta_slots                                       |
  |                                                            ├─ supersede 旧记录                                             |
  |                                                            ├─ 分配 resident_pq_slot                                        |
  |                                                            ├─ 组装 DeviceDeltaRecord                                       |
  |                                                            ├─ memcpy 原始 dtype 向量 → pinned staging                      |
  |                                                            └─ submit_delta_publication() ─────────►  device_ring_try_pop    |
  |                                                                                            │                                |
  |                                                                                            │  OPQ 变换 + PQ 编码             |
  |                                                                                            │  delta hash/bucket/override    |
  |                                                                                            │  resident_pq 表插入            |
  |                                                                                            │  dynamic route seqlock         |
  |                                                                                            │  atomicExch(delta_count)       |
  |                                                                                            │  __threadfence_system()        |
  |                                                                                            │  device_ring_push(completion)  |
  |                                                  [5] 可见时延 visibility_ns 统计 ◄──────────────┘                                |
  |                                                  [6] delta_.publish_metadata(epoch) ──► published_epoch_ CAS                |
  |                                                       telemetry.delta_publications++                                         |
  |                                                                                            |                                |
  |                                                                                            v                                |
  |                                                                                  查询: descriptor.snapshot_epoch =       |
  |                                                                                        delta_.published_epoch()         |
  |                                                                                  ─────────────────────────────────►    delta_visible()
```

关键点：**snapshot epoch 不是在 GPU 上推进的，而是 CPU 端 `DeltaCoordinator::publish_metadata` 通过 CAS 推进的**。查询在 admission 时（`query_execution.cc:48`）读取 `published_epoch()` 并写入 `QueryDescriptor.snapshot_epoch`，之后 kernel 内部用 `delta_visible(record, snapshot_epoch)` 判断某条 delta 是否对这次查询可见。control CTA 发布 `delta_count` 只是让"新槽位"在物理上可被读到；要真正进入查询的可见集合，还必须等 `published_epoch_` 被 CAS 到 ≥ `record.epoch`。这两步合起来就是 dvstor 的"双阶段可见"。

---

## 二、入口：`PersistentSearchEngine::publish_mutations`（6 步协议）

文件：`src/gpu_search/persistent_engine.cc:44-148`。这是被 compute_service 直接调用的公开入口。

### 2.1 协议锁与 epoch 预留（步骤 1）

```cpp
bool PersistentSearchEngine::publish_mutations(
    std::span<DeltaMutation> mutations,
    std::span<const u64> invalidated_graph_nodes) {
  std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
  if (mutations.empty()) {
    throw std::invalid_argument("GPU mutation publication requires a non-empty epoch batch");
  }
  // Epoch reservation and publication share this mutex with route-only
  // maintenance commands.  Therefore a later route barrier can never overtake
  // an earlier mutation whose GPU records have not been committed yet.
  const u64 epoch = delta_.reserve_epoch();
```

`mutation_publish_mutex_` 是整个发布协议的序列化锁。它不仅保护 mutation 发布，还和 `storage_reclaim.cc` 里的 route-only 发布（`synchronize_storage_routes` → `publish_barrier`）共享。注释明确指出：这把锁保证"晚到的 route barrier 永远不会赶在一批尚未 GPU 提交的 mutation 前面"。这是一个非常关键的顺序保证——如果没有它，可能出现"route 指针已经指向新 epoch，但 delta 行还没编码完"的窗口。

`delta_.reserve_epoch()` 见 `delta_index.cc:6-8`：

```cpp
u64 DeltaCoordinator::reserve_epoch() {
  return next_epoch_.fetch_add(1, std::memory_order_relaxed);
}
```

`next_epoch_` 从 1 开始（`delta_index.hh:92`）。注意这里用 `relaxed`：序保证由 `mutation_publish_mutex_` 提供，原子只是为了多读者（查询读 `published_epoch_`）安全。

### 2.2 批内 generation 去重（步骤 2）

`publish_mutations` 的输入是"一个 RPC slot 内积累的 mutation 数组"。由于不同客户端可能对同一 id 发起多次 mutation，且响应可能乱序到达，必须在进入 GPU 之前剔除"已被同批更新 generation 覆盖"的旧条目：

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

逻辑：

1. `delta_.version(candidate.id)` 取协调器里已记录的最新 generation（`delta_index.hh:56`，`delta_index.cc:100-104`）。
2. 再扫一遍本批已接受的 mutation，若同 id 已被接受，就把 `accepted_generation` 抬到本批最大。
3. 若 `candidate.generation == 0`，说明存储节点没赋值，自动 +1。
4. 若 `candidate.generation <= accepted_generation`，说明这条是同批或历史更老的版本，**直接 `continue` 丢弃**。
5. 否则用 `std::swap` 把它紧凑地换到 `accepted_count` 位置——注释（`persistent_engine.cc:55-57`）强调：用 swap 而不是 `remove_if` + move-assignment，是为了让每个预分配的 vector buffer 仍然绑在 RPC slot 的某个元素上，避免重新分配。

注意：`publish_compute_side_id`（在 `commit_storage_owner_slot` 里调用，`completion.cc:284-290`）已经做过一次 generation 判定，只把 `newest_generation` 的结果放进 `publication_mutations`。这里的二次去重是为同 id 在**同一批**里出现多次做最后兜底。

### 2.3 queue 时延统计（步骤 3）

```cpp
  const size_t mutation_count = mutations.size();
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

`mutation.enqueued_at` 是在 `commit_storage_owner_slot` 里设为 `slot.response_completed_at`（`completion.cc:311`）。这一段统计的是"stage1 响应完成 → 本批 mutation 真正开始发布"的等待，对应 telemetry 里的 `publication_queue_ns_total`。这是 SLO 的一部分：它度量的是 completion loop 是否被前面的批次阻塞。

### 2.4 GPU 上传（步骤 4）

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
  const auto gpu_upload_completed_at = std::chrono::steady_clock::now();
```

`upload_mutations` 是真正的 CPU 准备 + GPU 编码触发，详见第三节。这里把异常分成两类：`MutationCapacityError`（容量不足，可重试，见第 10 课）只记 `mutation_capacity_rejections`；其他异常会把引擎 `mark_unhealthy`。这是 dvstor 的"故障即停"策略——一旦发布协议损坏，宁可让所有后续查询失败也不要让脏数据进入 GPU。

`gpu_upload_completed_at` 是后面可见性时延的截止点。

### 2.5 可见时延统计（步骤 5）

```cpp
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

这一段度量"从 stage1 ACK 到 GPU 编码完成"的总时延——也就是 `upload_mutations` 返回为止。但注意：**此时查询还看不到这批 mutation**，因为 `published_epoch_` 还没推进。注释（`persistent_engine.cc:127-128`）说得很清楚：

```cpp
  // Queries cannot select this epoch until the coordinator publish above.
  // Include that final handoff in the stage1-response-to-visible SLO.
```

所以下一步 coordinator publish 的耗时也会被加进 `visibility_ns_*`：

### 2.6 coordinator publish + telemetry（步骤 6）

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

`publish_metadata` 见 `delta_index.cc:14-17`，它调用 `publish_impl(..., retain_vectors=false)`——也就是说协调器里**只保留元数据**，不拷贝 vector buffer。注释（`delta_index.hh:46-49`）解释：同步 GPU 上传路径已经用完了 vector buffer，再保留一份会阻止 RPC slot 复用。这是把 CPU 内存开销和 RPC slot 生命周期解耦的关键设计。

`publish_impl` 内部（`delta_index.cc:19-78`）做三件事：

1. 对每条 mutation 做一次 generation 检查（与 `publish_mutations` 里的检查呼应），把 `versions_[id]` 更新为新的 `{generation, epoch, deleted, in_delta=true}`。
2. 若 `retain_vectors=false`，只拷贝元数据字段（id/kind/generation/epoch/remote_node/old_remote_node/anchor_hint/maintenance_sequence/owner_storage/durable/enqueued_at），不拷 `vector`。
3. 若 `maintenance_sequence != 0`，把 `{maintenance_sequence, epoch, id, generation}` 压入 `durable_candidates_[owner_storage]`（优先队列），留给后续 `retire_durable`（见第 16 课）使用。

最后 `publish_barrier(epoch)`（`delta_index.cc:80-89`）：

```cpp
void DeltaCoordinator::publish_barrier(u64 epoch) {
  if (epoch == 0) {
    throw std::invalid_argument("delta publication barrier requires a non-zero epoch");
  }
  u64 current = published_epoch_.load(std::memory_order_relaxed);
  while (current < epoch &&
         !published_epoch_.compare_exchange_weak(current, epoch,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed)) {}
}
```

这是**本课的核心可见性点**：CAS 用 `release` 顺序，确保前面 `upload_mutations` 对 `delta_count` 的发布（GPU 端 `__threadfence_system`）、对 `delta_records` 的写入、对 `resident_pq_codes` 的写入，对后续读 `published_epoch_` 的查询线程（`acquire`，见 `delta_index.cc:91-93`）全部可见。换句话说，这一行 CAS 是"GPU 物理可见"和"查询逻辑可见"之间的发布线。

`coordinator_publish_ns` 被乘以 `visibility_sample_count` 加到 total，并加到 max——把协调器发布的开销摊到每条 mutation 上。最后用 CAS 循环更新 `visibility_ns_max`（无锁 max）。

至此 6 步协议走完。下面拆开步骤 4（`upload_mutations` → `upload_records_locked`）和 control CTA 的工作。

---

## 三、`upload_mutations` 与 `upload_records_locked`：CPU 侧准备

文件：`src/gpu_search/persistent_engine/delta_publication.cc:258-357`。

### 3.1 容量检查与 retired 槽回收

```cpp
size_t PersistentSearchEngine::Impl::upload_mutations(std::span<DeltaMutation> mutations, u64 epoch,
                        std::span<const u64> invalidated_graph_nodes) {
  if (mutations.empty()) return 0;
  const std::vector<u64> invalidation_keys = graph_cache_keys(invalidated_graph_nodes);
  std::lock_guard<std::mutex> lock(delta_mutex);
  reclaim_retired_delta_slots_locked();
  const size_t active_slots = active_delta_slots_locked();
  const size_t hard_watermark = static_cast<size_t>(delta_capacity) * 9 / 10;
  if (active_slots + mutations.size() > hard_watermark) {
    throw MutationCapacityError(
      "bounded GPU update tier reached its hard watermark; "
      "storage maintenance has not retired updates quickly enough");
  }
  for (DeltaMutation& mutation : mutations) {
    mutation.epoch = epoch;
  }
  upload_records_locked(mutations, invalidation_keys);
  return invalidation_keys.size();
}
```

注意三件事：

1. `graph_cache_keys`（`routing.cc:73-91`）在持 `delta_mutex` **之外**调用，因为它可能抛异常（容量超限）。它把存储节点返回的 raw node pointer 转成 graph-cache key，去重并截断到 `graph_invalidation_capacity`。
2. `reclaim_retired_delta_slots_locked()`（见第 16 课）先把已经被 query ticket barrier 释放的 retired slot 回收到 `free_delta_slots`。这是 RCU 回收的入口——只有当所有早于某 ticket 的查询都结束，对应的 delta slot 才能重用。
3. 90% 硬水位：`hard_watermark = delta_capacity * 9 / 10`。超过就抛 `MutationCapacityError`，让上层（`try_reserve_mutation_capacity`，见第 10 课）能拒掉新 RPC 而不是在这里崩。

最后把 `mutation.epoch = epoch` 统一打标——这步在 `delta_.publish_metadata` 之前，所以 `upload_records_locked` 写入 GPU 的 `record.epoch` 和协调器里 `versions_[id].epoch` 是同一个值。

### 3.2 `upload_records_locked`：主循环

`delta_publication.cc:70-256`。这是 CPU 侧最重的一段。我们按段拆。

#### 3.2.1 准备与容量再检查

```cpp
void PersistentSearchEngine::Impl::upload_records_locked(std::span<DeltaMutation> mutations,
                           std::span<const u64> invalidation_keys) {
  const auto prepare_started = std::chrono::steady_clock::now();
  bind_cuda_device("cudaSetDevice(GPU navigation delta publication)");
  (void)cudaGetLastError();
  const size_t available_slots = free_delta_slots.size() +
    (delta_capacity - delta_records_host.size());
  if (mutations.size() > available_slots) {
    throw std::runtime_error("GPU navigation delta live set exceeds its configured capacity");
  }
  const size_t vector_bytes = VamanaNode::vector_bytes();
  std::vector<DeviceDeltaRecord> records;
  std::vector<u32> destination_slots;
  std::vector<byte_t> vectors(static_cast<size_t>(mutations.size()) * vector_bytes);
  records.reserve(mutations.size());
  destination_slots.reserve(mutations.size());
  std::unordered_map<u32, size_t> staged_record_indices;
  std::vector<DeltaSupersedeUpdate> superseded_updates;
  std::vector<DeltaOverrideUpdate> override_updates;
  std::vector<f32> decoded(config.dim);
```

`bind_cuda_device` 确保 CUDA 上下文绑定到配置的 device；`cudaGetLastError()` 清掉残留错误（因为可能是从其他线程切过来）。`available_slots` 是"可重用 retired slot + 完全新 slot"的总和，这是第二次容量检查（第一次在 `upload_mutations` 用 `hard_watermark`，更保守）。

`vectors` 是一个紧凑的连续 buffer，按 `vector_bytes = VamanaNode::vector_bytes()` 对齐——这就是要 memcpy 到 pinned staging 的源数据。注意它**按存储 dtype 紧凑排布**，而不是按 f32 排布，编码时 OPQ 矩阵才在 GPU 上做 dtype → f32 → OPQ 变换（见 3.2.5 和第四节）。

`staged_record_indices` 把"本批已 staged 的 slot → 在 `records` 里的下标"映射起来，用于后面同 id 多次 mutation 时的"批内 supersede"——见 3.2.3。

#### 3.2.2 槽位分配

```cpp
  for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
    DeltaMutation& mutation = mutations[mutation_index];
    bool decoded_ready = false;
    u32 slot = UINT32_MAX;
    if (!free_delta_slots.empty()) {
      slot = free_delta_slots.back();
      free_delta_slots.pop_back();
    } else {
      slot = static_cast<u32>(delta_records_host.size());
      delta_records_host.emplace_back();
    }
```

优先复用 `free_delta_slots`（已回收的 retired slot），否则在高水位以内追加新 slot。`delta_records_host` 是 CPU 侧的"全量 delta 记录镜像"，它的大小单调增长直到 `delta_capacity`，所以一旦 `free_delta_slots` 非空就优先用它——这是 dvstor 不做紧凑整理、只做 free-list 回收的策略。

#### 3.2.3 旧记录 supersede

```cpp
    const auto previous = latest_delta_slot.find(mutation.id);
    if (previous != latest_delta_slot.end()) {
      DeviceDeltaRecord& previous_record = delta_records_host[previous->second];
      if (previous_record.superseded_epoch == 0 &&
          (previous_record.flags & kDeltaDeleted) == 0) {
        if ((previous_record.flags & kDeltaDurable) != 0) {
          --durable_delta_entries;
        } else {
          --mutable_delta_entries;
        }
      }
      previous_record.superseded_epoch = mutation.epoch;
      superseded_delta_slots[mutation.id].push_back(previous->second);
      const auto staged = staged_record_indices.find(previous->second);
      if (staged != staged_record_indices.end()) {
        records[staged->second].superseded_epoch = mutation.epoch;
      } else {
        superseded_updates.push_back(DeltaSupersedeUpdate{
          .slot = previous->second,
          .epoch = mutation.epoch,
        });
      }
    }
```

`latest_delta_slot[id]` 记录这个 id 当前最新版本所在的 slot。如果存在，就要把它 supersede：

1. 调整 `durable_delta_entries` / `mutable_delta_entries` 计数（用于 telemetry 和容量管理）。
2. 在 CPU 镜像里把 `superseded_epoch` 置为 `mutation.epoch`。
3. 把它塞进 `superseded_delta_slots[id]`，留待后续 `mark_durable_delta_records_locked`（第 16 课）回收。
4. **关键优化**：如果这个旧 slot 已经在本批 staged（`staged_record_indices` 命中），就直接改 `records[...]` 里的 `superseded_epoch`；否则才生成一个 `DeltaSupersedeUpdate` 发给 GPU。这就是"批内去重"在 GPU 侧的对应：同 id 在一批里出现两次，旧的那条根本不会进 GPU，supersede 也在 CPU 镜像里就完成了。

`DeltaSupersedeUpdate` 结构（`types.hh:57-61`）只有 `slot`/`reserved`/`epoch` 三个字段——非常紧凑，因为它要在 control CTA 里逐条处理。

#### 3.2.4 owner 校验与 anchor bucket 分配

```cpp
    const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
    const u64 record_remote = mutation.remote_node != 0
      ? mutation.remote_node : mutation.old_remote_node;
    const u32 route_shard = static_cast<u32>(record_remote >> 48);
    if (record_remote == 0 || route_shard >= index.shards.size() ||
        route_shard != mutation.owner_storage) {
      throw std::runtime_error(
        "storage returned an invalid physical owner for GPU dynamic routing");
    }
    // Reuse the graph-address validator so an acknowledged but misaligned
    // dynamic pointer can never enter either the delta map or route overlay.
    (void)graph_cache_key(record_remote);
    u32 bucket = 0;
    if (!deleted) {
      const auto hinted = anchor_buckets_by_raw.find(mutation.anchor_hint);
      if (hinted == anchor_buckets_by_raw.end()) {
        if (!decoded_ready) {
          decode_mutation_payload(mutation, decoded);
          decoded_ready = true;
        }
        bucket = nearest_anchor(decoded, record_remote);
      } else {
        bucket = hinted->second;
      }
    }
```

这里做三件大事：

1. **物理 owner 校验**：`record_remote >> 48` 取高位 shard，必须等于 `mutation.owner_storage`。否则说明存储节点返回了一个不属于它自己的指针——这会破坏 GPU 动态路由表（每个 shard 的 route slot 只能指向本 shard 的节点），直接抛异常。
2. **graph 地址校验**：`graph_cache_key(record_remote)`（`routing.cc:47-71`）会验证 raw pointer 落在某个 shard 的 node 或 dynamic 区间内、且 stride 对齐。注释强调"复用 graph-cache 校验器"，让"被 ACK 但对齐错误的动态指针"永远进不了 delta map 或 route overlay。这是一道防御性校验，把存储节点的 bug 挡在 GPU 之外。
3. **anchor bucket 分配**：delta 在 GPU 上按 anchor bucket 链表组织（`delta_bucket_heads[anchor_bucket]` 是链头），查询时只扫与查询最近的几个 bucket（见第 20 课）。`mutation.anchor_hint` 是存储节点回填的提示（非零时直接用 `anchor_buckets_by_raw` 查表）；否则 CPU 解码向量并调用 `nearest_anchor`（`routing.cc:22-45`）在 shard 内做暴力最近邻找 anchor。注意 `decode_mutation_payload`（`routing.cc:7-20`）支持 f32 直传和存储 dtype 解码两种情况，所以 CPU 侧解码是惰性的（`decoded_ready` 标志）。

#### 3.2.5 base override 与 DeviceDeltaRecord 组装

```cpp
    u32 base_ordinal = kBaseOverrideEmpty;
    if (format::remote_to_ordinal(
          index, RemotePtr{mutation.old_remote_node}, base_ordinal)) {
      const auto [it, inserted] =
        base_override_epochs.emplace(base_ordinal, mutation.epoch);
      if (inserted) {
        override_updates.push_back(DeltaOverrideUpdate{
          .ordinal = base_ordinal,
          .epoch = mutation.epoch,
        });
      } else if (mutation.epoch < it->second) {
        it->second = mutation.epoch;
        override_updates.push_back(DeltaOverrideUpdate{
          .ordinal = base_ordinal,
          .epoch = mutation.epoch,
        });
      }
    } else {
      base_ordinal = kBaseOverrideEmpty;
    }
```

这段处理"base override"——即"某个 base 索引里的节点已经被本次 mutation 替换/删除，查询时应当跳过它"。`format::remote_to_ordinal`（见第 7 课）尝试把 `old_remote_node` 反解成 base 索引 ordinal。成功则：

- 在 CPU 镜像 `base_override_epochs` 里记录"这个 ordinal 从 epoch 起被覆盖"。
- 生成一个 `DeltaOverrideUpdate`（`types.hh:63-67`）发往 GPU。注意只有"新插入"或"epoch 更小"时才生成 update——避免重复刷写。

`base_override_epochs` 是一个 hash map，但它只在 CPU 用于维护；GPU 上是 `base_override_keys` / `base_override_epochs` 两个并行数组组成的开放寻址 hash（见第四节 runtime.cuh）。

```cpp
    DeviceDeltaRecord record{
      .id = mutation.id,
      .generation = std::max<u32>(1, mutation.generation),
      .flags = (deleted ? kDeltaDeleted : 0u) |
        (mutation.durable ? kDeltaDurable : 0u),
      .base_ordinal = base_ordinal,
      .epoch = mutation.epoch,
      .remote_node = record_remote,
      .anchor_bucket = bucket,
      .resident_pq_slot = deleted
        ? UINT32_MAX : allocate_resident_pq_slot_locked(record_remote),
    };
    delta_records_host[slot] = record;
    records.push_back(record);
    destination_slots.push_back(slot);
    staged_record_indices.emplace(slot, records.size() - 1);
    latest_delta_slot[mutation.id] = slot;
    if (!deleted) {
      if (mutation.durable) {
        ++durable_delta_entries;
      } else {
        ++mutable_delta_entries;
      }
    }
```

组装 `DeviceDeltaRecord`（`persistent_kernel.hh:58-68`）：

- `generation` 用 `max(1, mutation.generation)` 保证非零（generation 0 在 `dynamic_route_atomic_*` 里是"未初始化"哨兵）。
- `flags` 编码 deleted/durable 两位（`kDeltaDeleted = 1u`、`kDeltaDurable = 1u << 1`，`persistent_kernel.hh:27-28`）。
- `base_ordinal`：成功反解则填，否则 `kBaseOverrideEmpty = UINT32_MAX`。
- `resident_pq_slot`：deleted 留 `UINT32_MAX`，否则调 `allocate_resident_pq_slot_locked` 分配。

`allocate_resident_pq_slot_locked`（`delta_publication.cc:39-68`）值得单独看：

```cpp
u32 PersistentSearchEngine::Impl::allocate_resident_pq_slot_locked(u64 remote_node) {
  if (remote_node == 0) {
    throw std::runtime_error(
      "cannot allocate resident GPU PQ for a null remote node");
  }
  if (resident_pq_slots_by_remote.contains(remote_node)) {
    throw std::runtime_error(
      "storage reused a dynamic remote node before its resident GPU PQ was reclaimed");
  }
  u32 slot = UINT32_MAX;
  if (!free_resident_pq_slots.empty()) {
    slot = free_resident_pq_slots.back();
    free_resident_pq_slots.pop_back();
  } else if (resident_pq_high_watermark < resident_pq_capacity) {
    slot = resident_pq_high_watermark++;
  } else {
    throw MutationCapacityError(
      "resident GPU PQ tier is full; increase --gpu-resident-pq-budget-mb "
      "or consolidate dynamic vectors into a new base generation");
  }
  resident_pq_slots_by_remote.emplace(remote_node, slot);
  const u64 live = active_resident_pq_slots_locked();
  engine.telemetry_.resident_pq_entries.store(live, std::memory_order_relaxed);
  u64 peak = engine.telemetry_.resident_pq_peak_entries.load(
    std::memory_order_relaxed);
  while (peak < live &&
         !engine.telemetry_.resident_pq_peak_entries.compare_exchange_weak(
           peak, live, std::memory_order_relaxed)) {}
  return slot;
}
```

resident PQ 是 GPU 上一块**常驻的 PQ code 缓存**，专门给动态路由用——每个 live 的 dynamic route pointer 在这里有一份 PQ code，查询时不用走 RDMA 读 PQ。分配策略和 delta slot 一样：free-list 优先，否则高水位增长。注意第二个 if：`resident_pq_slots_by_remote.contains(remote_node)` 时**直接抛异常**——因为同一个 remote_node 不应该有两个 resident slot，这表明存储节点复用了 dynamic pointer 而 GPU 端还没回收（RCU 没追上）。这是 dvstor 一致性的硬保证。

#### 3.2.6 向量写入：memcpy 到紧凑 buffer，不启动 H2D

```cpp
    byte_t* stored_vector = vectors.data() + mutation_index * vector_bytes;
    if (deleted) {
      std::memset(stored_vector, 0, vector_bytes);
    } else if (mutation.vector.size() == vector_bytes) {
      std::memcpy(stored_vector, mutation.vector.data(), vector_bytes);
    } else {
      if (!decoded_ready) {
        decode_mutation_payload(mutation, decoded);
        decoded_ready = true;
      }
      encode_float_vector_to_storage(decoded.data(), config.dim,
                                     config.resolved_vector_dtype(), stored_vector);
    }
  }
```

**这是本课的一个关键设计点**。三种情况：

1. `deleted`：直接 memset 0。GPU 侧 OPQ/PQ 编码时检测 `kDeltaDeleted` flag 会跳过——见第四节 runtime.cuh:311 `if ((record.flags & kDeltaDeleted) == 0)`。
2. `mutation.vector.size() == vector_bytes`：vector 已经是存储 dtype 的紧凑表示，直接 memcpy。
3. 否则：vector 是 f32（从 RPC 请求来），需要 CPU 侧 `encode_float_vector_to_storage` 编码到存储 dtype。

注意第 3 种情况用的是 `decode_mutation_payload` + `encode_float_vector_to_storage`——即先把 f32 解码（其实只是 memcpy），再编码到目标 dtype。这看起来多余，其实是为了处理 dtype 不一致的情况（比如 mutation.vector 是 f32 但 config 要求 u8）。

**整个 `upload_records_locked` 没有调用任何 `cudaMemcpyAsync` 或 `cudaMemcpy` H2D**。所有数据先在 CPU 端的 `records` / `destination_slots` / `vectors` / `superseded_updates` / `override_updates` 里准备好，然后一次性 memcpy 到 pinned staging：

#### 3.2.7 批量拷贝到 pinned staging

```cpp
  if (records.size() > delta_command_capacity ||
      superseded_updates.size() > delta_command_capacity ||
      override_updates.size() > delta_command_capacity ||
      invalidation_keys.size() > graph_invalidation_capacity) {
    throw std::runtime_error("GPU navigation delta control batch exceeds capacity");
  }

  std::memcpy(delta_staging_records_host, records.data(),
              records.size() * sizeof(DeviceDeltaRecord));
  std::memcpy(delta_staging_slots_host, destination_slots.data(),
              destination_slots.size() * sizeof(u32));
  std::memcpy(delta_staging_vectors_host, vectors.data(), vectors.size());
  if (!superseded_updates.empty()) {
    std::memcpy(delta_supersede_updates_host, superseded_updates.data(),
                superseded_updates.size() * sizeof(DeltaSupersedeUpdate));
  }
  if (!override_updates.empty()) {
    std::memcpy(delta_override_updates_host, override_updates.data(),
                override_updates.size() * sizeof(DeltaOverrideUpdate));
  }
  if (!invalidation_keys.empty()) {
    std::memcpy(graph_invalidation_keys_host, invalidation_keys.data(),
                invalidation_keys.size() * sizeof(u64));
  }
```

`delta_staging_*_host` 是 **mapped pinned memory**——`cudaHostAlloc` 分配、`cudaHostGetDevicePointer` 拿到 device 端指针（`d_delta_staging_*`）。CPU 写入后，GPU 通过统一虚拟地址空间直接可见，**不需要任何 H2D 拷贝或 stream 同步**。这是 dvstor 的核心零拷贝设计：control CTA 用 `params.delta_staging_*`（device 指针）直接读 CPU 写好的数据。

容量校验：四个数组都不能超过 `delta_command_capacity`，graph invalidation 不能超过 `graph_invalidation_capacity`。这些容量在 `construction.cc` 里根据 `--gpu-delta-command-capacity` 等配置分配（见第 13 课）。

#### 3.2.8 提交与 telemetry

```cpp
  const u32 count = static_cast<u32>(delta_records_host.size());
  const auto command_started = std::chrono::steady_clock::now();
  engine.telemetry_.publication_prepare_ns_total.fetch_add(
    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      command_started - prepare_started).count()), std::memory_order_relaxed);
  submit_delta_publication(DeltaPublishDescriptor{
    .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
    .record_count = static_cast<u32>(records.size()),
    .final_count = count,
    .invalidation_count = static_cast<u32>(invalidation_keys.size()),
    .superseded_count = static_cast<u32>(superseded_updates.size()),
    .override_count = static_cast<u32>(override_updates.size()),
  });
  refresh_anchor_graph_records(invalidation_keys);
  engine.telemetry_.publication_command_ns_total.fetch_add(
    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - command_started).count()),
    std::memory_order_relaxed);
  engine.telemetry_.delta_physical_entries.store(
    count - free_delta_slots.size(), std::memory_order_relaxed);
  engine.telemetry_.delta_mutable_entries.store(
    mutable_delta_entries, std::memory_order_relaxed);
  engine.telemetry_.delta_durable_entries.store(
    durable_delta_entries, std::memory_order_relaxed);
}
```

- `prepare_started` 到 `command_started` 这段计入 `publication_prepare_ns_total`——即 CPU 侧准备（解码、查找 anchor、组装 record）的耗时。
- `submit_delta_publication` 阻塞等 GPU 完成（见第五节），所以 `command_started` 到返回这段计入 `publication_command_ns_total`——即"control CTA 处理本批 + completion 回读"的耗时。
- `refresh_anchor_graph_records`（`routing.cc:93-186`）处理 graph invalidation：如果 invalidating 的 key 对应 anchor route slot，会 RDMA 重读 anchor graph record。这步在 `submit_delta_publication` 之后做——因为 control CTA 在处理本批时已经把 `anchor_graph_states[slot]` 从 `kGraphCacheReady` CAS 到 `kGraphCacheStale`（见第四节 runtime.cuh:379），所以这里重读不会和查询冲突。
- 最后更新三个容量 telemetry。

`DeltaPublishDescriptor` 字段填写：

| 字段 | 值 | 含义 |
|------|----|----|
| `command_id` | `next_delta_command_id++` | 全局单调递增，用于 completion 匹配与 route seqlock 的 staleness 检查 |
| `first_slot` | 0（默认） | 未使用，保留给未来"按 slot 范围发布"的优化 |
| `record_count` | `records.size()` | 本批新 staged 的记录数（不含 supersede） |
| `final_count` | `delta_records_host.size()` | 发布后 GPU `delta_count` 应当被设成的值（高水位） |
| `invalidation_count` | `invalidation_keys.size()` | graph cache 失效 key 数 |
| `superseded_count` | `superseded_updates.size()` | 需要在 GPU 上打 superseded_epoch 的旧 slot 数 |
| `override_count` | `override_updates.size()` | 需要在 GPU 上插入 base override 的 ordinal 数 |
| `durable_count` | 0（默认） | 本批 durable 提升数（仅 `mark_durable_delta_records_locked` 路径填） |
| `resident_pq_erase_count` | 0（默认） | 本批 resident PQ 擦除数（仅 reclaim 路径填） |
| `dynamic_route_count` | 0（默认） | 本批 dynamic route 更新数（仅 `synchronize_storage_routes` 路径填） |
| `flags` | 0（默认） | `kDeltaCommandReset` / `kDeltaCommandPromoteOverrides` 位 |

`final_count` 用 `delta_records_host.size()` 而不是 `records.size()` 是有意的——`delta_count` 在 GPU 上是一个"高水位"：它告诉查询"0 到 delta_count-1 的 slot 都是合法的 delta 记录"。重用 retired slot 时下标不变，所以 `delta_count` 单调增长直到 `delta_capacity`，回收的 slot 只是放进 free-list，不会让 `delta_count` 缩小。这点在 `query_traversal.cuh:129` 附近有注释强调。

---

## 四、control CTA：批量编码与原子发布

文件：`src/gpu_search/persistent_kernel/runtime.cuh:83-676`。这是常驻 control CTA 的主循环（`enable_delta` 角色见第 21 课）。我们只看 delta 发布分支。

### 4.1 取 descriptor 与参数校验

```cpp
    if (threadIdx.x == 0) {
      have_delta_submission = enable_delta &&
        params.delta_submissions.entries != nullptr &&
        device_ring_try_pop(params.delta_submissions, delta_descriptor) ? 1u : 0u;
    }
    __syncthreads();
    if (have_delta_submission != 0) {
      if (threadIdx.x == 0) {
        delta_status = 0;
        const bool reset = (delta_descriptor.flags & kDeltaCommandReset) != 0;
        const bool promote =
          (delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0;
        constexpr u32 known_flags =
          kDeltaCommandReset | kDeltaCommandPromoteOverrides;
        if ((delta_descriptor.flags & ~known_flags) != 0 ||
            (reset && promote) || ...
```

`device_ring_try_pop` 从 `MappedRing<DeltaPublishDescriptor>` 取一条命令。`MappedRing` 是映射到 GPU 的固定容量环形队列（见第 17 课），CPU `try_push`、GPU `try_pop`。

校验逻辑非常细致（`runtime.cuh:97-167`）：

- `flags & ~known_flags` 必须为 0——未知 flag 直接 `delta_status = -EINVAL`。
- `reset` 和 `promote` 互斥。
- `reset` 模式下：`first_slot` 必须为 0，`record_count` ≤ `delta_capacity`，其他 count 字段都必须为 0，且所有相关 device 指针非空。
- 非 reset 模式下：`final_count`、`record_count` ≤ `delta_capacity`；若 `record_count != 0`，staging 指针、`delta_remote_*` 表、PQ 编码所需的所有指针（`delta_pq_codes`、`delta_encode_scratch`、`pq_centroids`、`resident_pq_*`）都必须非空且容量非零。
- `durable_count != 0` 时 `delta_durable_updates` 必须非空；`resident_pq_erase_count != 0` 时所有 `resident_pq_erase_*` 指针非空；`dynamic_route_count != 0` 时所有 `dynamic_route_*` 指针非空。
- `promote && override_count != 0` 时 `permanent_override_bits` 必须非空；非 promote 时 `base_override_*` 必须非空。

这些校验全部由 thread 0 完成，结果存进 `__shared__ i32 delta_status`，`__syncthreads` 后全 block 可见。这是 GPU 端的"防御性编程"——任何一个不合法字段都会让本批失败、completion 回传 -EINVAL，CPU 端 `submit_delta_publication` 抛异常、引擎 `mark_unhealthy`。

### 4.2 `kDeltaCommandReset` 分支：清空 delta 表

`runtime.cuh:171-229`。这是引擎生命周期切换（重建索引、reset）时用的。逐段看：

```cpp
      if ((delta_descriptor.flags & kDeltaCommandReset) != 0) {
        if (delta_status == 0) {
          for (u32 index = threadIdx.x; index < delta_descriptor.record_count;
               index += blockDim.x) {
            const DeviceDeltaRecord record = params.delta_records[index];
            const u32 remote_position = params.delta_remote_positions[index];
            if (record.remote_node != 0 &&
                remote_position < params.delta_remote_capacity &&
                load_cg(params.delta_remote_slots + remote_position) == index) {
              atomicCAS(reinterpret_cast<unsigned long long*>(
                          params.delta_remote_keys + remote_position),
                        record.remote_node, kDeltaRemoteTombstone);
              atomicExch(params.delta_remote_slots + remote_position, UINT32_MAX);
            }
            if (record.base_ordinal < params.num_nodes) {
              const u32 mask = params.base_override_capacity - 1;
              u32 position = hash32(record.base_ordinal) & mask;
              for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
                const u32 key = load_cg(params.base_override_keys + position);
                if (key == record.base_ordinal) {
                  if (atomicCAS(params.base_override_keys + position,
                                record.base_ordinal,
                                kBaseOverrideTombstone) == record.base_ordinal) {
                    params.base_override_epochs[position] = 0;
                  }
                  break;
                }
                if (key == kBaseOverrideEmpty) break;
                position = (position + 1) & mask;
              }
            }
            params.delta_records[index] = {};
            params.delta_records[index].base_ordinal = kBaseOverrideEmpty;
            params.delta_next[index] = UINT32_MAX;
            params.delta_prev[index] = UINT32_MAX;
            params.delta_remote_positions[index] = UINT32_MAX;
          }
          for (u32 index = threadIdx.x; index < params.anchor_count;
               index += blockDim.x) {
            params.delta_bucket_heads[index] = UINT32_MAX;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0 && delta_status == 0) {
          __threadfence();
          atomicExch(params.delta_count, 0u);
          __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          device_ring_push(params.delta_completions, DeltaPublishCompletion{
            .command_id = delta_descriptor.command_id,
            .status = delta_status,
            .final_count = 0,
          });
        }
        __syncthreads();
        continue;
      }
```

对每条记录（stride = `blockDim.x`）：

1. 把 `delta_remote_keys[position]` 从 `record.remote_node` CAS 成 `kDeltaRemoteTombstone`（`UINT64_MAX`），`delta_remote_slots[position]` 设为 `UINT32_MAX`——这从 remote→slot 反查表里抹掉这条记录。
2. 若 `base_ordinal` 有效，在 `base_override_keys` 开放寻址 hash 里找到它，CAS 成 `kBaseOverrideTombstone`（`UINT32_MAX - 1`），epoch 清零。
3. 把 `delta_records[index]` 清零（保留 `base_ordinal = kBaseOverrideEmpty`），链表指针 `delta_next` / `delta_prev` 设为 `UINT32_MAX`。

之后 thread 0 做发布：

```cpp
        if (threadIdx.x == 0 && delta_status == 0) {
          __threadfence();
          atomicExch(params.delta_count, 0u);
          __threadfence_system();
        }
```

`__threadfence`（block 内 + device 内可见）保证前面所有清零对全 device 可见；`atomicExch(delta_count, 0)` 是发布；`__threadfence_system` 保证对 CPU（mapped memory、completion 轮询）也可见。这个"threadfence → atomicExch → threadfence_system"的序列就是 dvstor 在 GPU 上做发布的**标准模式**——和后面正常发布的 `atomicExch(delta_count, final_count)` 完全一致。

最后 push completion，`continue` 回到主循环。**注意 Reset 路径不推进 `published_epoch_`**——epoch 推进是 CPU 侧 `DeltaCoordinator` 的事，GPU 只负责把 `delta_count` 归零。

### 4.3 正常发布分支：校验 slot 范围与 dynamic route

```cpp
      if (delta_status == 0) {
        for (u32 index = threadIdx.x;
             index < delta_descriptor.record_count;
             index += blockDim.x) {
          const u32 slot = params.delta_staging_slots[index];
          if (slot >= delta_descriptor.final_count || slot >= params.delta_capacity) {
            atomicExch(&delta_status, -EINVAL);
          }
        }
        __syncthreads();
      }
```

每个 staged slot 必须落在 `[0, final_count)` 且 `< delta_capacity`。这是为了防止 CPU 端的 free-list bug 把一个越界 slot 写进 staging。

```cpp
      if (delta_status == 0) {
        for (u32 index = threadIdx.x;
             index < delta_descriptor.dynamic_route_count;
             index += blockDim.x) {
          const DynamicRouteUpdate update =
            params.dynamic_route_updates[index];
          const bool live = (update.flags & kDynamicRouteLive) != 0;
          bool duplicate_slot = false;
          for (u32 prior = 0; prior < index; ++prior) {
            duplicate_slot = duplicate_slot ||
              params.dynamic_route_updates[prior].slot == update.slot;
          }
          if (update.slot >= params.dynamic_route_capacity ||
              update.shard >= params.num_shards ||
              update.epoch == 0 ||
              update.slot / kDynamicRouteSlotsPerShard != update.shard ||
              (update.flags & ~kDynamicRouteLive) != 0 ||
              (live &&
               (update.remote_node == 0 ||
                static_cast<u32>(update.remote_node >> 48) != update.shard)) ||
              (!live && update.remote_node != 0) || duplicate_slot) {
            atomicExch(&delta_status, -EINVAL);
            continue;
          }
          const DeviceDynamicRouteSlot& current =
            params.dynamic_route_slots[update.slot];
          const u64 current_command =
            dynamic_route_atomic_load(current.command_id);
          const u32 current_id = dynamic_route_atomic_load(current.id);
          const u32 current_generation =
            dynamic_route_atomic_load(current.generation);
          if (current_command >= delta_descriptor.command_id ||
              (current_id == update.id &&
               current_generation > update.generation)) {
            atomicExch(&delta_status, -ESTALE);
          }
        }
        __syncthreads();
      }
```

dynamic route 校验（仅 route-only 路径会触发，`dynamic_route_count > 0`）：

- slot/shard 一致性：`slot / kDynamicRouteSlotsPerShard == shard`（每个 shard 有 8 个 route slot，`types.hh:85`）。
- live route 的 `remote_node` 高位 shard 必须匹配；非 live route 的 `remote_node` 必须为 0。
- 批内 slot 不能重复（O(n²) 检查，但 `dynamic_route_capacity` 很小，`kPersistentMaxShards * kDynamicRouteSlotsPerShard ≤ 256`）。
- **staleness 检查**：如果当前 slot 的 `command_id >= 本批 command_id`，说明已经有更新的命令覆盖了它（命令乱序到达），或者同 id 但 generation 更高——直接 `delta_status = -ESTALE`，本批失败。

这是 dvstor 对"命令乱序"的防御：虽然 `mutation_publish_mutex_` 保证了 CPU 端的提交顺序，但 mapped ring 可能让 GPU 端以不同顺序看到命令（理论上不会，但防御性校验仍然必要）。

### 4.4 拷贝 staging → device 与 OPQ 变换

```cpp
      if (delta_status == 0) {
        for (u32 index = threadIdx.x;
             index < delta_descriptor.record_count;
             index += blockDim.x) {
          const u32 slot = params.delta_staging_slots[index];
          params.delta_records[slot] = params.delta_staging_records[index];
          params.delta_next[slot] = UINT32_MAX;
          params.delta_prev[slot] = UINT32_MAX;
        }
        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.vector_bytes;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.vector_bytes);
          const u32 byte = static_cast<u32>(index % params.vector_bytes);
          const u32 slot = params.delta_staging_slots[record_index];
          params.delta_vectors[static_cast<u64>(slot) * params.vector_bytes + byte] =
            params.delta_staging_vectors[index];
        }
        __syncthreads();
```

第一步：把 `delta_staging_records`（CPU 写好的 `DeviceDeltaRecord`）拷到 `delta_records[slot]`，同时初始化链表指针。注意这里每个 thread 处理一条记录（stride `blockDim.x`）。

第二步：把 `delta_staging_vectors`（CPU 写好的紧凑 dtype 向量）按 byte 拷到 `delta_vectors[slot * vector_bytes ...]`。这里用 byte 粒度并行——`record_count * vector_bytes` 个 byte，每个 thread 拷一个 byte。这种"细粒度并行拷贝"是 GPU 上的常见模式，比 thread 拷整条向量负载更均衡。

```cpp
        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.dim;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.dim);
          const u32 row = static_cast<u32>(index % params.dim);
          const u32 slot = params.delta_staging_slots[record_index];
          const DeviceDeltaRecord record = params.delta_records[slot];
          f32 transformed = 0.0f;
          if ((record.flags & kDeltaDeleted) == 0) {
            const u8* vector = params.delta_vectors +
              static_cast<size_t>(slot) * params.vector_bytes;
            if (params.opq_matrix == nullptr) {
              transformed = storage_component(params, vector, row);
            } else {
              const f32* matrix_row = params.opq_matrix +
                static_cast<size_t>(row) * params.dim;
              for (u32 column = 0; column < params.dim; ++column) {
                transformed += matrix_row[column] *
                  storage_component(params, vector, column);
              }
            }
          }
          params.delta_encode_scratch[index] = transformed;
        }
        __syncthreads();
```

**OPQ 变换**：对每条记录的每个维度（`record_count * dim` 个元素，每 thread 一个），把存储 dtype 解码成 f32（`storage_component`），再乘 OPQ 矩阵。`opq_matrix == nullptr` 时跳过矩阵乘法（没用 OPQ 的索引）。结果写进 `delta_encode_scratch[record_index * dim + row]`——一个临时的 f32 buffer，按 record 紧凑排布（而不是按 slot）。

注意 `kDeltaDeleted` 的记录直接写 0——后面 PQ 编码会跳过。

### 4.5 PQ 编码与 resident PQ 写入

```cpp
        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.pq_code_bytes;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.pq_code_bytes);
          const u32 subquantizer = static_cast<u32>(index % params.pq_code_bytes);
          const u32 slot = params.delta_staging_slots[record_index];
          u8 best_code = 0;
          if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
            const f32* transformed = params.delta_encode_scratch +
              static_cast<size_t>(record_index) * params.dim +
              static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
            const f32* centroids = params.pq_centroids +
              static_cast<size_t>(subquantizer) * 256 * params.pq_subvector_dim;
            f32 best_distance = FLT_MAX;
            for (u32 centroid = 0; centroid < 256; ++centroid) {
              f32 distance = 0.0f;
              for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
                const f32 difference = transformed[dimension] -
                  centroids[static_cast<size_t>(centroid) * params.pq_subvector_dim + dimension];
                distance += difference * difference;
              }
              if (distance < best_distance) {
                best_distance = distance;
                best_code = static_cast<u8>(centroid);
              }
            }
          }
          params.delta_pq_codes[
            static_cast<size_t>(slot) * params.pq_code_bytes + subquantizer] = best_code;
          const u32 resident_slot = params.delta_records[slot].resident_pq_slot;
          if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
            if (resident_slot >= params.resident_pq_capacity) {
              atomicExch(&delta_status, -ENOSPC);
            } else {
              params.resident_pq_codes[
                static_cast<size_t>(resident_slot) * params.pq_code_bytes +
                subquantizer] = best_code;
            }
          }
        }
        __threadfence();
        __syncthreads();
```

**PQ 编码**：对每条记录的每个子量化器（`record_count * pq_code_bytes` 个 byte，每 thread 一个），在 256 个 centroid 里暴力找最近邻。这是标准的 PQ 编码流程（见第 9 课 PQ 模型）。结果写两处：

1. `delta_pq_codes[slot * pq_code_bytes + subquantizer]`——delta 行的 PQ code，查询时用 `approximate_handle` 读它（`candidate_scoring.cuh:443-462`）。
2. `resident_pq_codes[resident_slot * pq_code_bytes + subquantizer]`——resident PQ 缓存，dynamic route 查询时用。

如果 `resident_slot >= resident_pq_capacity`，说明 CPU 分配的 slot 越界（不应该发生，因为 `allocate_resident_pq_slot_locked` 已经检查过），直接 `delta_status = -ENOSPC`。

最后 `__threadfence`——保证 PQ code 写入对后续 `delta_count` 发布可见。这是"先写数据，再发发布标记"的标准模式。

### 4.6 graph invalidation 与 supersede

```cpp
        for (u32 index = threadIdx.x;
             index < delta_descriptor.invalidation_count;
             index += blockDim.x) {
          const u64 key = params.graph_invalidation_keys[index];
          const u32 route_slot = anchor_graph_slot(params, key);
          if (route_slot != UINT32_MAX &&
              params.anchor_graph_states != nullptr) {
            atomicCAS(params.anchor_graph_states + route_slot,
                      kGraphCacheReady, kGraphCacheStale);
          }
          if (params.graph_cache_sets == 0 ||
              params.graph_cache_states == nullptr ||
              params.graph_cache_keys == nullptr) {
            continue;
          }
          const u32 set = hash64(key) % params.graph_cache_sets;
          for (u32 way = 0; way < params.graph_cache_ways; ++way) {
            const u32 slot = set * params.graph_cache_ways + way;
            for (;;) {
              const u32 state = *reinterpret_cast<volatile u32*>(
                params.graph_cache_states + slot);
              if (load_cg(params.graph_cache_keys + slot) != key ||
                  state == kGraphCacheEmpty || state == kGraphCacheStale ||
                  state == kGraphCacheFillInvalidated) {
                break;
              }
              if (state == kGraphCacheReady) {
                if (atomicCAS(params.graph_cache_states + slot, kGraphCacheReady,
                              kGraphCacheStale) == kGraphCacheReady) {
                  break;
                }
                continue;
              }
              if (state == kGraphCacheFilling) {
                if (atomicCAS(params.graph_cache_states + slot, kGraphCacheFilling,
                              kGraphCacheFillInvalidated) == kGraphCacheFilling) {
                  break;
                }
                continue;
              }
              break;
            }
          }
        }
```

graph cache invalidation：对每个失效 key，先在 anchor route 表里找对应 slot，CAS `kGraphCacheReady → kGraphCacheStale`；再在 graph cache 的组相联 hash 里找所有匹配 key 的 way，根据当前 state 分别处理：

- `kGraphCacheReady`：CAS 到 `kGraphCacheStale`。
- `kGraphCacheFilling`：CAS 到 `kGraphCacheFillInvalidated`——这告诉正在 fill 的线程"你读到的是旧数据，别 commit"。
- 其他状态（Empty/Stale/FillInvalidated）：不动。

这是 dvstor graph cache 的 RCU 失效协议（见第 19 课）。

```cpp
        if (threadIdx.x == 0) {
          for (u32 index = 0; index < delta_descriptor.superseded_count; ++index) {
            const DeltaSupersedeUpdate update = params.delta_supersede_updates[index];
            if (update.slot >= delta_descriptor.final_count) {
              delta_status = -EINVAL;
              continue;
            }
            DeviceDeltaRecord& record = params.delta_records[update.slot];
            record.superseded_epoch = update.epoch;
            unlink_mutable_delta(params, update.slot);
          }
        }
```

supersede 处理由 thread 0 串行做（数量很少）：在 GPU 上把 `record.superseded_epoch` 设为 `update.epoch`，并 `unlink_mutable_delta` 把它从 anchor bucket 链表里摘掉。摘链后查询扫 bucket 时就不会再遇到这条记录（即使 `delta_visible` 还没判定它不可见，链表结构上就跳过了）。

### 4.7 base override：`kDeltaCommandPromoteOverrides` 语义

```cpp
        if ((delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0) {
          for (u32 index = threadIdx.x;
               index < delta_descriptor.override_count;
               index += blockDim.x) {
            const u32 ordinal = params.delta_override_updates[index].ordinal;
            if (ordinal >= params.num_nodes) {
              atomicExch(&delta_status, -EINVAL);
              continue;
            }
            atomicOr(params.permanent_override_bits + ordinal / 32,
                     1u << (ordinal % 32));
          }
        } else if (threadIdx.x == 0) {
          const u32 mask = params.base_override_capacity - 1;
          for (u32 index = 0; index < delta_descriptor.override_count; ++index) {
            const DeltaOverrideUpdate update = params.delta_override_updates[index];
            u32 position = hash32(update.ordinal) & mask;
            u32 first_tombstone = UINT32_MAX;
            bool inserted = false;
            for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
              const u32 key = params.base_override_keys[position];
              if (key == update.ordinal) {
                params.base_override_epochs[position] = min(
                  params.base_override_epochs[position], update.epoch);
                inserted = true;
                break;
              }
              if (key == kBaseOverrideTombstone && first_tombstone == UINT32_MAX) {
                first_tombstone = position;
              }
              if (key == kBaseOverrideEmpty) {
                const u32 destination = first_tombstone == UINT32_MAX
                  ? position : first_tombstone;
                params.base_override_epochs[destination] = update.epoch;
                __threadfence();
                params.base_override_keys[destination] = update.ordinal;
                inserted = true;
                break;
              }
              position = (position + 1) & mask;
            }
            if (!inserted && first_tombstone != UINT32_MAX) {
              params.base_override_epochs[first_tombstone] = update.epoch;
              __threadfence();
              params.base_override_keys[first_tombstone] = update.ordinal;
              inserted = true;
            }
            if (!inserted) delta_status = -ENOSPC;
          }
        }
```

**`kDeltaCommandPromoteOverrides` 的语义**：当引擎决定把一批 delta 永久化（durable promotion，见第 16 课 `mark_durable_delta_records_locked`），它会把对应的 base override 从"临时 hash 表"提升到"永久 bitmap"——`permanent_override_bits` 是一个 bit 数组，每个 bit 对应一个 base ordinal，置位表示"这个 base 节点永久跳过"。这是为了在 retire durable delta 后清理 base override 表——retire 后 base 节点确实不存在了，bitmap 让查询永久跳过它，hash 表里的临时 override 可以被 Reset 清掉。

非 promote 路径是标准的开放寻址 hash 插入：linear probing，tombstone 复用，`__threadfence` 保证 epoch 先于 key 可见（这样查询看到 key 时 epoch 一定已写好）。注意 `min(existing_epoch, update.epoch)`——如果同一 ordinal 已经有更新 epoch 的 override，保留更老的（因为更老的 epoch 对更多查询可见）。

`base_overridden` 查询判定（`candidate_scoring.cuh:393-414`）：

```cpp
__device__ bool base_overridden(const PersistentKernelParams& params,
                                u32 ordinal, u64 snapshot_epoch) {
  if (params.permanent_override_bits != nullptr &&
      ordinal < params.num_nodes) {
    const u32 word = load_cg(params.permanent_override_bits + ordinal / 32);
    if ((word & (1u << (ordinal % 32))) != 0) return true;
  }
  if (params.base_override_capacity == 0) return false;
  const u32 mask = params.base_override_capacity - 1;
  u32 position = hash32(ordinal) & mask;
  for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
    const u32 key = load_cg(params.base_override_keys + position);
    if (key == ordinal) {
      const u64 epoch = load_cg(params.base_override_epochs + position);
      return epoch != 0 && epoch <= snapshot_epoch;
    }
    if (key == kBaseOverrideEmpty) return false;
    position = (position + 1) & mask;
  }
  return false;
}
```

先查 permanent bitmap（永久跳过，不看 epoch），再查临时 hash 表（epoch ≤ snapshot 才跳过）。这两层结构让"已 retire 的 durable delta"查询零开销，"还在 delta 里的临时 override"按 epoch 过滤。

### 4.8 durable / resident_pq_erase 处理

```cpp
        if (threadIdx.x == 0) {
          for (u32 index = 0; index < delta_descriptor.durable_count; ++index) {
            const DeltaDurableUpdate update = params.delta_durable_updates[index];
            if (update.slot >= delta_descriptor.final_count) {
              delta_status = -EINVAL;
              continue;
            }
            DeviceDeltaRecord& record = params.delta_records[update.slot];
            if (record.epoch == update.epoch) {
              if (record.superseded_epoch == 0) {
                record.superseded_epoch = update.epoch;
              }
              unlink_mutable_delta(params, update.slot);
              const u32 remote_position = params.delta_remote_positions[update.slot];
              if (record.remote_node != 0 &&
                  remote_position < params.delta_remote_capacity &&
                  load_cg(params.delta_remote_slots + remote_position) == update.slot) {
                atomicCAS(reinterpret_cast<unsigned long long*>(
                            params.delta_remote_keys + remote_position),
                          record.remote_node, kDeltaRemoteTombstone);
                atomicExch(params.delta_remote_slots + remote_position, UINT32_MAX);
              }
              params.delta_remote_positions[update.slot] = UINT32_MAX;
              if (record.base_ordinal < params.num_nodes) {
                atomicOr(params.permanent_override_bits + record.base_ordinal / 32,
                         1u << (record.base_ordinal % 32));
                // ... 清掉 base_override hash 表里的临时条目 ...
              }
            }
          }
          for (u32 index = 0;
               index < delta_descriptor.resident_pq_erase_count; ++index) {
            erase_resident_pq(params, params.resident_pq_erase_updates[index]);
          }
        }
```

`DeltaDurableUpdate` 处理（仅 `mark_durable_delta_records_locked` 路径触发）：对每个 durable slot，如果 epoch 匹配，就：

1. 设 `superseded_epoch = epoch`（如果还没设）。
2. `unlink_mutable_delta` 从 bucket 链表摘掉。
3. 从 `delta_remote_*` 反查表抹掉。
4. 把 base ordinal 写进 permanent bitmap，并清掉临时 hash 条目。

这就是 durable retire 的 GPU 侧动作——之后这条 delta 仍然在 `delta_records` 里（直到 Reset），但所有查询都会跳过它（`superseded_epoch != 0`，`delta_visible` 返回 false）。

`resident_pq_erase` 处理：从 resident PQ 表里抹掉指定 remote_node 的条目，free 出 slot。这用于 dynamic route 被替换时回收旧 route 的 resident PQ。

### 4.9 resident PQ 插入与 delta bucket 链表

```cpp
      if (delta_status == 0) {
        if (threadIdx.x == 0) {
          const u32 mask = params.delta_remote_capacity - 1;
          for (u32 index = 0; index < delta_descriptor.record_count; ++index) {
            const u32 slot = params.delta_staging_slots[index];
            const DeviceDeltaRecord record = params.delta_records[slot];
            if ((record.flags & kDeltaDeleted) == 0 &&
                !insert_resident_pq(
                  params, record.remote_node, record.resident_pq_slot)) {
              delta_status = -ENOSPC;
              break;
            }
            params.delta_remote_positions[slot] = UINT32_MAX;
            if (record.remote_node != 0 && params.delta_remote_capacity != 0) {
              // ... 开放寻址插入 delta_remote_keys/delta_remote_slots ...
            }
            if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
                record.superseded_epoch == 0 &&
                params.delta_bucket_heads != nullptr) {
              const u32 old_head = params.delta_bucket_heads[record.anchor_bucket];
              params.delta_prev[slot] = UINT32_MAX;
              params.delta_next[slot] = old_head;
              if (old_head < params.delta_capacity) {
                params.delta_prev[old_head] = slot;
              }
              params.delta_bucket_heads[record.anchor_bucket] = slot;
            }
          }
        }
      }
      __syncthreads();
```

thread 0 串行处理（数量少）：

1. `insert_resident_pq`：把 `{remote_node → resident_pq_slot}` 插入 GPU 上的 resident PQ hash 表。失败（表满）则 `delta_status = -ENOSPC`。
2. `delta_remote_keys/delta_remote_slots` 插入：开放寻址，把 `{remote_node → slot}` 写进反查表。查询时用 `delta_slot_from_raw` 从 remote_node 反查 slot（见第 18 课）。
3. **delta bucket 链表插入**：头插法，把 slot 接到 `delta_bucket_heads[anchor_bucket]` 链头。只有 `!deleted && !durable && superseded_epoch == 0` 的记录才进链表——durable 和 superseded 的记录通过其他路径（`delta_slot_from_raw` 反查）访问，不进 bucket 扫描。

### 4.10 dynamic route seqlock 发布

```cpp
      if (delta_status == 0) {
        // Canonical storage-route codes become visible before a route slot can
        // point at them. Mark every changing slot odd before touching either
        // its code or metadata; query scoring rechecks the same sequence after
        // consuming both.
        for (u32 index = threadIdx.x;
             index < delta_descriptor.dynamic_route_count;
             index += blockDim.x) {
          const DynamicRouteUpdate update =
            params.dynamic_route_updates[index];
          DeviceDynamicRouteSlot& destination =
            params.dynamic_route_slots[update.slot];
          cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
            destination.sequence);
          sequence.fetch_add(1, cuda::memory_order_acq_rel);
        }
        __syncthreads();
        for (u64 byte = threadIdx.x;
             byte < static_cast<u64>(delta_descriptor.dynamic_route_count) *
                      params.pq_code_bytes;
             byte += blockDim.x) {
          const u32 update_index = static_cast<u32>(
            byte / params.pq_code_bytes);
          const u32 code_byte = static_cast<u32>(
            byte % params.pq_code_bytes);
          const DynamicRouteUpdate update =
            params.dynamic_route_updates[update_index];
          if ((update.flags & kDynamicRouteLive) != 0) {
            params.dynamic_route_pq_codes[
              static_cast<size_t>(update.slot) * params.pq_code_bytes +
              code_byte] = params.dynamic_route_code_updates[byte];
          }
        }
        __threadfence();
        __syncthreads();
        for (u32 index = threadIdx.x;
             index < delta_descriptor.dynamic_route_count;
             index += blockDim.x) {
          const DynamicRouteUpdate update =
            params.dynamic_route_updates[index];
          DeviceDynamicRouteSlot& destination =
            params.dynamic_route_slots[update.slot];
          cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
            destination.sequence);
          dynamic_route_atomic_store(
            destination.command_id, delta_descriptor.command_id);
          dynamic_route_atomic_store(destination.epoch, update.epoch);
          dynamic_route_atomic_store(
            destination.remote_node, update.remote_node);
          dynamic_route_atomic_store(destination.id, update.id);
          dynamic_route_atomic_store(
            destination.generation, update.generation);
          dynamic_route_atomic_store(destination.shard, update.shard);
          dynamic_route_atomic_store(destination.flags, update.flags);
          __threadfence();
          sequence.fetch_add(1, cuda::memory_order_release);
        }
      }
```

**这是 route-only command 的核心：device-scope seqlock**。注释（`types.hh:99-103`）已经说清楚了：

> sequence is a device-scope seqlock. The control CTA is the only writer: odd means an update is in progress, even means the remaining fields form a stable snapshot. Query CTAs never wait for a writer; they skip an unstable dynamic seed and continue with the static route.

流程：

1. **第一步：所有 slot 的 sequence +1**（变成奇数，标记"正在更新"）。`acq_rel` 保证这一步对所有 query CTA 可见。
2. **第二步：写 PQ code**。把 `dynamic_route_code_updates`（CPU 预编码好的 PQ code）拷到 `dynamic_route_pq_codes[slot]`。只有 live route 才写（非 live route 的 PQ code 不被查询用）。
3. `__threadfence`：保证 PQ code 写入对后续 metadata 写入之前可见。
4. **第三步：写 metadata**（command_id/epoch/remote_node/id/generation/shard/flags），再 `sequence +1`（变成偶数，标记"稳定快照"）。`release` 保证 metadata 写入对偶数 sequence 的读者可见。

查询侧的 seqlock 读（`candidate_scoring.cuh` 里 `dynamic_route_atomic_*` 的 load）：

- 读 sequence。奇数 → 跳过（用 static route），偶数 → 读 metadata，再读 sequence，如果相同则用，不同则跳过。

这种"写两次 sequence + 读两次 sequence"的经典 seqlock 让查询**无锁、无等待**地读到一致的 dynamic route 快照。注释里强调"Query CTAs never wait for a writer"——查询永远不会被发布阻塞，最坏情况是跳过这次更新用 static route。

### 4.11 delta_count 发布与 completion

```cpp
      __syncthreads();
      if (threadIdx.x == 0 && delta_status == 0) {
        __threadfence();
        atomicExch(params.delta_count, delta_descriptor.final_count);
        __threadfence_system();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        device_ring_push(params.delta_completions, DeltaPublishCompletion{
          .command_id = delta_descriptor.command_id,
          .status = delta_status,
          .final_count = delta_status == 0 ? delta_descriptor.final_count : 0u,
        });
      }
      __syncthreads();
      if (threadIdx.x == 0) idle_cycles = 256u;
      __syncthreads();
      continue;
```

**最终发布**：`__threadfence`（device 可见）→ `atomicExch(delta_count, final_count)` → `__threadfence_system`（CPU 可见）。这就是 4.2 节提到的"threadfence → atomicExch → threadfence_system"标准模式。

`delta_count` 是查询扫 delta 表的上界——`query_traversal.cuh:114` `delta_count_snapshot = min(load_cg(params.delta_count), params.delta_capacity)`。注释（`query_traversal.cuh:129`）说 `delta_count` 是"reused-slot high watermark"，回收 slot 不会让它缩小。

注意：**`delta_count` 发布不等于查询可见**。查询可见还需要 `published_epoch_ >= record.epoch`。`delta_count` 只是"物理上 slot 0..N-1 都有合法数据"，`published_epoch_` 是"逻辑上这些数据对当前查询可见"。

最后 push completion。`final_count` 字段在失败时回 0——CPU 端 `submit_delta_publication` 会检测到 `final_count != descriptor.final_count` 并抛异常。

---

## 五、`submit_delta_publication`：CPU↔GPU 命令/完成通道

文件：`src/gpu_search/persistent_engine/delta_publication.cc:7-33`。

```cpp
void PersistentSearchEngine::Impl::submit_delta_publication(const DeltaPublishDescriptor& descriptor) {
  const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
    config.storage_owner_rpc_timeout_ms, 1000, 5000));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!delta_submissions.try_push(descriptor)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("persistent GPU delta command queue is not making progress");
    }
    std::this_thread::yield();
  }

  DeltaPublishCompletion completion{};
  while (!delta_completions.try_pop(completion)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("persistent GPU delta publication timed out");
    }
    std::this_thread::yield();
  }
  if (completion.command_id != descriptor.command_id || completion.status != 0 ||
      completion.final_count != descriptor.final_count) {
    throw std::runtime_error(
      "persistent GPU delta publication failed: command=" +
      std::to_string(completion.command_id) + " status=" +
      std::to_string(completion.status) + " count=" +
      std::to_string(completion.final_count));
  }
}
```

**这是 CPU 与 control CTA 之间的同步点**。流程：

1. `delta_submissions.try_push(descriptor)`：把 descriptor 推进 `MappedRing<DeltaPublishDescriptor>`。如果 ring 满（GPU 还没消费上一批），`yield` 重试直到 deadline。
2. `delta_completions.try_pop(completion)`：等 GPU push 回的 completion。同样 `yield` 重试。
3. 校验：`command_id` 匹配（防止乱序）、`status == 0`、`final_count` 匹配。任一不符抛异常。

`MappedRing` 是 mapped pinned memory 上的 SPSC/MPSC 环形队列（见第 17 课）。CPU 写 `delta_submissions`、GPU 读；GPU 写 `delta_completions`、CPU 读。两端都用原子操作 + 内存屏障保证无锁正确性。

`storage_owner_rpc_timeout_ms` 被 clamp 到 [1000, 5000] ms——这是 stage1 RPC 超时复用为 delta 发布超时。超时意味着 control CTA 卡死（GPU hang 或死锁），引擎会因抛异常而 `mark_unhealthy`。

**关键设计**：`submit_delta_publication` 是**同步阻塞**的。`publish_mutations` 持有 `mutation_publish_mutex_`，所以这里阻塞意味着 completion loop 也会阻塞——这就是为什么 `publication_command_ns_total` 重要：它度量的是 control CTA 处理一批的端到端时延，直接决定 mutation 发布的吞吐。dvstor 选择同步阻塞而不是异步回调，是为了简化协议顺序——`publish_metadata` 必须在 GPU 编码完成后才推进 `published_epoch_`，异步会让这个顺序难以保证。

---

## 六、route-only command：canonical storage route 发布

文件：`src/gpu_search/persistent_engine/storage_reclaim.cc:229-264`（`synchronize_storage_routes` 的一部分）。

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
              dynamic_route_update_scratch.size() *
                sizeof(DynamicRouteUpdate));
  submit_delta_publication(DeltaPublishDescriptor{
    .command_id = next_delta_command_id.fetch_add(
      1, std::memory_order_relaxed),
    .final_count = static_cast<u32>(delta_records_host.size()),
    .dynamic_route_count = static_cast<u32>(
      dynamic_route_update_scratch.size()),
  });
  dynamic_route_diff->commit(dynamic_route_update_scratch);
  engine.telemetry_.dynamic_route_publications.fetch_add(
    1, std::memory_order_relaxed);
  engine.telemetry_.dynamic_route_slot_updates.fetch_add(
    dynamic_route_update_scratch.size(), std::memory_order_relaxed);
  // Queries acquire this epoch only after the control CTA has made both the
  // PQ bytes and route seqlocks visible.
  engine.delta_.publish_barrier(epoch);
  return true;
```

route-only command 的特点：

1. **`final_count = delta_records_host.size()`**：不改变 delta_count，只是把当前高水位再发一次。这是为了让 control CTA 进入正常发布分支（非 Reset），但实际 `record_count = 0`，所以不会动 `delta_records`。
2. **`dynamic_route_count > 0`**：触发 4.3/4.4/4.10 节的 dynamic route 处理。
3. **先 `submit_delta_publication`（等 GPU 完成 seqlock 发布），再 `publish_barrier(epoch)`**。注释明确："Queries acquire this epoch only after the control CTA has made both the PQ bytes and route seqlocks visible."——这与 mutation 发布的顺序一致：GPU 物理可见 → CPU 逻辑可见。

route-only command 不创建 delta 行（`record_count = 0`），但它推进 `published_epoch_`。这是 `DeltaCoordinator::publish_barrier` 存在的原因——`delta_index.hh:50-52` 注释："Route-only GPU publications still need to become visible at one ordered query snapshot, but they must not create a synthetic mutation/delta row."

存储节点定期广播 canonical route publication（每个 shard 8 个 slot 的 live representative），compute 节点用 `dynamic_route_diff->prepare` 算出 diff，再通过 route-only command 安装到 GPU。这让所有 compute 节点的 dynamic route overlay 收敛到存储节点的 canonical 视图（`types.hh:81-85` 注释）。

---

## 七、可见性窗口与遥测

### 7.1 `update_visibility_us`

`src/common/configuration.hh:63`：`u32 update_visibility_us{10'000};`——默认 10ms。这是一个**软 SLO**：从 stage1 ACK 到查询可见的目标时延。它被用在两处：

1. `storage_reclaim.cc:364` `retire_durable_delta`：durable sequence 历史的 grace period。一个 durable sequence 必须被观察到至少 `update_visibility_us` 才被认为 safe——确保所有 in-flight 查询都过了旧 epoch。
2. `storage_reclaim.cc:529` `maintenance_loop`：maintenance period 上限为 `update_visibility_us / 1000` ms——保证 retire 足够频繁。

### 7.2 三段发布遥测

| telemetry 字段 | 测量区间 | 来源 |
|---|---|---|
| `publication_queue_ns_total` | `mutation.enqueued_at` → `publication_started` | `persistent_engine.cc:85-93` |
| `publication_prepare_ns_total` | `prepare_started` → `command_started` | `delta_publication.cc:234-236` |
| `publication_command_ns_total` | `command_started` → `submit_delta_publication` 返回 | `delta_publication.cc:246-249` |
| `visibility_ns_total` / `visibility_ns_max` | `mutation.enqueued_at` → `coordinator publish` 完成 | `persistent_engine.cc:106-146` |

`visibility_ns_*` 是端到端 SLO 指标，包含前三段加 coordinator publish。breakdown benchmark（第 30 课）会把这些导出成 per-stage 时延。

### 7.3 delta 容量遥测

- `delta_physical_entries`：`delta_records_host.size() - free_delta_slots.size()`（物理占用 slot 数）。
- `delta_mutable_entries` / `delta_durable_entries`：分别计数 mutable/durable delta。
- `delta_live_entries`：`delta_.delta_size()`（协调器里的 live mutation 数，含未 retire 的 durable candidate）。
- `resident_pq_entries` / `resident_pq_peak_entries` / `resident_pq_reclaimed`：resident PQ 表的占用、峰值、累计回收。
- `dynamic_route_publications` / `dynamic_route_slot_updates` / `dynamic_route_live_slots` / `dynamic_route_snapshot_skips`：route overlay 的发布与查询跳过统计。

---

## 八、关键数据结构与流程图

### 8.1 数据结构总览

```
CPU 侧 (PersistentSearchEngine::Impl)
├── delta_records_host: vector<DeviceDeltaRecord>       # 全量镜像，单调增长
├── free_delta_slots: vector<u32>                       # 可重用 slot
├── latest_delta_slot: unordered_map<node_t, u32>       # id → 最新 slot
├── superseded_delta_slots: unordered_map<node_t, vector<u32>>  # id → 旧 slot 列表
├── resident_pq_slots_by_remote: unordered_map<u64, u32>        # remote_node → resident slot
├── free_resident_pq_slots: vector<u32>
├── base_override_epochs: unordered_map<u32, u64>       # ordinal → epoch (CPU 镜像)
├── delta_submissions: MappedRing<DeltaPublishDescriptor>       # CPU→GPU 命令
├── delta_completions: MappedRing<DeltaPublishCompletion>       # GPU→CPU 完成
└── delta_staging_*_host: mapped pinned (DeviceDeltaRecord/slots/vectors/supersede/override/...)

GPU 侧 (PersistentKernelParams)
├── delta_records: DeviceDeltaRecord[delta_capacity]            # 主表
├── delta_vectors: u8[delta_capacity * vector_bytes]            # 原始 dtype 向量
├── delta_pq_codes: u8[delta_capacity * pq_code_bytes]          # PQ 编码
├── delta_encode_scratch: f32[record_count * dim]               # OPQ 变换临时
├── delta_next/delta_prev: u32[delta_capacity]                  # bucket 链表
├── delta_bucket_heads: u32[anchor_count]                       # 链头
├── delta_count: u32 (atomic)                                   # 高水位发布标记
├── delta_remote_keys/slots/positions: 反查表 (remote_node ↔ slot)
├── base_override_keys/epochs: 开放寻址 hash (临时 override)
├── permanent_override_bits: u32 bitmap (永久 override)
├── resident_pq_codes/keys/slots/positions: resident PQ 表
├── dynamic_route_slots: DeviceDynamicRouteSlot[dynamic_route_capacity]  # seqlock
├── dynamic_route_pq_codes: u8[dynamic_route_capacity * pq_code_bytes]
└── delta_staging_*: mapped pinned (CPU 写, GPU 读)

协调器 (DeltaCoordinator)
├── delta_: unordered_map<node_t, DeltaMutation>        # 元数据 (无 vector)
├── versions_: unordered_map<node_t, VersionEntry>      # generation/epoch/deleted/in_delta
├── durable_candidates_: vector<priority_queue>         # per-owner retire 队列
├── next_epoch_: atomic<u64>                            # epoch 分配器
└── published_epoch_: atomic<u64>                       # 查询可见性发布线
```

### 8.2 可见性协议状态机

```
                  CPU publish_mutations                     GPU control CTA                  Query CTA
                  ────────────────────                     ────────────────                  ─────────
mutation 到达 ──► reserve_epoch() ──┐
                                    │
                                    ├── upload_records_locked()
                                    │   ├─ 分配 slot
                                    │   ├─ supersede 旧记录
                                    │   ├─ 组装 DeviceDeltaRecord
                                    │   ├─ memcpy → pinned staging
                                    │   └─ submit_delta_publication() ──► device_ring_try_pop
                                    │                                       │
                                    │                                       ├─ 校验 descriptor
                                    │                                       ├─ 拷贝 staging → device
                                    │                                       ├─ OPQ 变换 (delta_encode_scratch)
                                    │                                       ├─ PQ 编码 (delta_pq_codes + resident_pq_codes)
                                    │                                       ├─ graph invalidation
                                    │                                       ├─ supersede / override / durable / resident_pq_erase
                                    │                                       ├─ resident PQ 插入
                                    │                                       ├─ delta bucket 链表头插
                                    │                                       ├─ [route-only] dynamic route seqlock 发布
                                    │                                       ├─ __threadfence
                                    │                                       ├─ atomicExch(delta_count, final_count)
                                    │                                       ├─ __threadfence_system
                                    │                                       └─ device_ring_push(completion) ──► try_pop
                                    │                                                                       │
                                    ◄──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├── visibility_ns 统计
                                    ├── publish_metadata(epoch) ──► versions_ 更新
                                    │                              └─ publish_barrier(epoch)
                                    │                                  └─ CAS published_epoch_ (release)
                                    │                                          │
                                    │                                          │ acquire
                                    │                                          ▼
                                    │                                  query admission: descriptor.snapshot_epoch = published_epoch_
                                    │                                          │
                                    │                                          ▼
                                    │                                  kernel: delta_visible(record, snapshot_epoch)
                                    │                                          │
                                    │                                          ▼ record.epoch ≤ snapshot_epoch && (superseded==0 || >snapshot) && !deleted && !durable
                                    │                                          可见 ✓
```

### 8.3 stage1 ACK → staging → control CTA → epoch 发布 → 查询可见 时序图

```
 storage_owner          compute_service                PersistentSearchEngine            GPU control CTA           Query CTA
      │  InsertBatchResponse │                              │                              │                          │
      │ ───────────────────► │                              │                              │                          │
      │                      │ handle_storage_owner_response│                              │                          │
      │                      │ commit_storage_owner_slot()  │                              │                          │
      │                      │  - 组装 publication_mutations│                              │                          │
      │                      │ publish_storage_owner_mutations                              │                          │
      │                      │ ──────────────────────────► │                              │                          │
      │                      │                              │ publish_mutations():         │                          │
      │                      │                              │  [1] reserve_epoch()         │                          │
      │                      │                              │  [2] 批内去重                 │                          │
      │                      │                              │  [3] publication_queue_ns   │                          │
      │                      │                              │  [4] upload_records_locked()│                          │
      │                      │                              │      ├─ memcpy → staging ──► │ (mapped pinned)         │
      │                      │                              │      └─ submit_delta_publication                         │
      │                      │                              │           try_push(descriptor) ─► device_ring_try_pop   │
      │                      │                              │              ▲                    │                     │
      │                      │                              │              │ 阻塞              │ 校验 + 拷贝          │
      │                      │                              │              │                    │ OPQ + PQ 编码        │
      │                      │                              │              │                    │ supersede/override   │
      │                      │                              │              │                    │ bucket 链表          │
      │                      │                              │              │                    │ __threadfence        │
      │                      │                              │              │                    │ atomicExch(delta_count)
      │                      │                              │              │                    │ __threadfence_system │
      │                      │                              │              │                    │ device_ring_push(completion)
      │                      │                              │              │  ◄─────────────────┘                      │
      │                      │                              │  [5] visibility_ns 统计                              │
      │                      │                              │  [6] publish_metadata(epoch)                         │
      │                      │                              │      └─ CAS published_epoch_ (release) ─────────────┐
      │                      │                              │           │                                          │ acquire
      │                      │                              │           │ telemetry                               │
      │                      │                              │ ◄─────────┘                                          │
      │                      │ ◄──────────────────────────                                                                │
      │                      │                                                                                            │
      │                      │                          下一查询 search():                                                │
      │                      │                          snapshot_epoch = published_epoch_ ──────────────────────────────►│
      │                      │                                                                                            │
      │                      │                                                                                            │ delta_visible()
      │                      │                                                                                            │ 可见 ✓
```

---

## 九、与其他模块的关系

- **第 9 课（GPU 类型/遥测/PQ 模型）**：`DeviceDeltaRecord`、`DeltaPublishDescriptor`、`DeltaPublishCompletion`、`Telemetry` 全部定义在 `types.hh`；OPQ/PQ 编码用的 `pq_centroids`、`opq_matrix`、`pq_subvector_dim` 来自 PQ model。本课的编码流程（4.4/4.5）是 PQ 模型的 GPU 端实例。
- **第 10 课（delta/动态路由/预算）**：`DeltaCoordinator` 的 `reserve_epoch`/`publish_metadata`/`publish_barrier` 是本课的协调器侧；`MutationCapacityError`、`try_reserve_mutation_capacity`、`release_mutation_capacity` 是本课的容量预算机制；`DynamicRouteOverlayDiff` 是 route-only command 的 diff 计算。本课是第 10 课协议的 GPU 执行侧。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`PersistentSearchEngine::Impl` 的 `delta_mutex`、`mutation_publish_mutex_`、`delta_stream`、所有 `d_delta_*` 指针的生命周期；`clear_delta_device_state` 用 `kDeltaCommandReset`。
- **第 16 课（存储回收 RCU）**：`reclaim_retired_delta_slots_locked`、`retired_delta_batches`、`retired_resident_pq_batches`、`query_ticket_barrier_passed`、`mark_durable_delta_records_locked`、`kDeltaCommandPromoteOverrides` 的 durable promote 路径。本课的"发布"和第 16 课的"回收"是 delta 生命周期的两端。
- **第 17 课（kernel 启动器/上下文/device ring）**：`MappedRing`、`device_ring_try_pop`/`device_ring_push`、`DeviceRingView`、persistent kernel 启动。
- **第 18 课（候选评分）**：`delta_visible`、`delta_code_visible`、`base_overridden`、`approximate_handle`、`delta_slot_from_raw`——这些是本课发布结果在查询侧的消费者。
- **第 19 课（RDMA cache）**：graph cache invalidation（4.6 节）是 RDMA cache 失效协议的触发方；`refresh_anchor_graph_records` 是失效后的 RDMA 重读。
- **第 20 课（查询遍历主循环）**：`delta_count_snapshot`、bucket 链表扫描、`delta_visible` 过滤——本课发布的 `delta_count` 和 bucket 链表在这里被消费。
- **第 21 课（kernel 运行时/角色调度）**：`enable_delta` 角色、control CTA 的 `idle_cycles` 退避、`kernel_ready_count` 同步——本课的 control CTA 是第 21 课角色调度的一个角色。
- **第 28 课（计算侧 storage owner 更新）**：`commit_storage_owner_slot`、`publish_storage_owner_mutations`、`publish_compute_side_id`——本课的 stage1 ACK 触发链。

---

## 十、小结

本课讲解了 dvstor 的"增量发布协议"，核心是**双阶段可见**：

1. **物理可见**（GPU 侧）：control CTA 在 `__threadfence` 后 `atomicExch(delta_count, final_count)`，让新 slot 进入查询的扫描范围。route-only command 用 device-scope seqlock 发布 `{epoch, remote_node, id, generation}`，让查询无锁读到一致的 dynamic route 快照。
2. **逻辑可见**（CPU 侧）：`DeltaCoordinator::publish_barrier` 用 `release` CAS 推进 `published_epoch_`，查询在 admission 时 `acquire` 读它作为 `snapshot_epoch`，kernel 里用 `delta_visible(record, snapshot_epoch)` 判定每条 delta 是否对本次查询可见。

这两步合起来保证："一条 mutation 从 stage1 ACK 到被查询看见，中间没有任何一个查询会看到半编码状态"。协议的关键设计点：

- **零拷贝 staging**：CPU 写 mapped pinned、GPU 直接读，不启动 side kernel、不同步 H2D。
- **常驻 control CTA 批量编码**：OPQ/PQ 编码、hash 表更新、链表插入都在 control CTA 里批量做，避免为每批 mutation 启动新 kernel。
- **seqlock 无锁发布**：dynamic route 用 device-scope seqlock，查询永不等待发布，最坏跳过用 static route。
- **同步阻塞 `submit_delta_publication`**：简化协议顺序，`publish_metadata` 一定在 GPU 编码完成后才推进 `published_epoch_`。
- **`mutation_publish_mutex_` 全局序列化**：保证 mutation 发布和 route-only barrier 的顺序，杜绝 epoch 与 GPU 状态错位。
- **双阶段 override**：临时 hash 表（按 epoch 过滤）+ 永久 bitmap（durable retire 后零开销）。

下一课（第 16 课）将讲解这套 delta 的回收侧——`reclaim_retired_delta_slots_locked`、`retire_durable_delta`、`mark_durable_delta_records_locked` 如何在 RCU 与 durable promotion 下安全地回收 slot。
