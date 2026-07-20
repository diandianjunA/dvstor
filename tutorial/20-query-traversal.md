# 第 20 课 查询遍历主循环

## 本课目标与涉及文件

持久化 GPU kernel 把"一条 top-k 向量查询"的全部工作放进一个 CTA(block) 中执行。这个 block 既是 CPU 提交的消费者，也是结果回写的生产者，中间要完成：解码查询向量 → OPQ 旋转 → PQ 查找表构建 → 入口（静态 anchor + 动态 route）打分初始化 beam → 反复"选未展开候选 → 并行发远端图读 → 解码 compact graph 邻居 → 用常驻 PQ 评分 → 去重/合并/裁剪 beam" → 收敛后精排（远端读 fixed record，精确 L2）→ delta overlay 补救 → top-k 写出。整个过程与 RDMA cache、dynamic route seqlock、base override、snapshot epoch 紧密耦合，且每个阶段都用 `clock64()` 打点写回 `CompletionDescriptor`。

本课逐行讲解这套主循环，所涉及文件：

- `src/gpu_search/persistent_kernel/query_traversal.cuh`（893 行，本课主菜）
- `src/gpu_search/persistent_kernel/candidate_scoring.cuh`（beam/visited/排序/delta 可见性等通用工具，第 18 课已铺垫）
- `src/gpu_search/persistent_kernel/rdma_cache.cuh`（图 cache、精排 cache、批量远端读，第 19 课已铺垫）
- `src/gpu_search/persistent_kernel/context.cuh`（CUB block radix sort 类型别名）
- `src/gpu_search/persistent_kernel.hh`（`PersistentKernelParams` / `QueryDescriptor` / `CompletionDescriptor` 以及各 `kPersistent*` 容量常量）
- `src/gpu_search/types.hh`（`CompletionDescriptor` 字段定义）
- `src/gpu_search/delta_scan_budget.hh` / `initial_seed_budget.hh` / `dynamic_route_consistency.hh`（与 delta 扫描 / 入口预算 / seqlock 窗口相关的算法常量）
- `src/vamana/vamana_node.hh` 与 `src/vamana/hot_graph.hh`（compact graph 5 字节 RemotePtr 编解码、checksum16、布局）
- `src/remote_pointer.hh`（`[16b memory_node | 48b byte_offset]` 的 RemotePtr 表示）

与其它课的关系：评分函数族与 beam/visited 工具见**第 18 课**；RDMA cache + GPUNetIO 直接读见**第 19 课**；kernel 启动器、CTA 角色调度（query/owner/dispatcher/control）见**第 17 课**与**第 21 课**；完成回写与慢查询日志见**第 14 课**；delta 发布与 base override 写入见**第 15 课**；RCU 回收见**第 16 课**。

---

## 总览：单查询 block 的执行阶段

`process_query` 是一条查询的全部状态机。把它压扁成一张表（阶段→device 函数→关键 cycle 计时点）：

| 阶段 | 主要 device 代码 | cycle 写回字段 |
|------|------------------|----------------|
| 0. 入口校验 | `process_query` 开头 | `completion.gpu_cycles`（整条总时长，错误时立即写） |
| 1. prepare：query→float + OPQ 旋转 + LUT 构建 | 第 281–333 行 | `completion.prepare_cycles`（在主循环开始前由 thread0 写入） |
| 2. 入口打分：anchor / entry points + dynamic route → 初始化 beam | 第 335–572 行 | 计入 prepare 周期内 |
| 3. 主循环：选未展开 → `fetch_graph_records_batch` → 解码邻居 → `approximate_handles_batch` → `merge_approximate_into_beam` | 第 608–788 行 | `graph_phase_cycles` / `score_phase_cycles` / `beam_phase_cycles` 分别累加 |
| 4. 精排：`exactify_into_beam` | 第 790–823 行 | `exact_phase_cycles` |
| 5. delta overlay：`add_delta_candidates` | 第 825–842 行 | `completion.delta_scan_cycles`（外加 records/scored/truncated 计数） |
| 6. 终排序 + top-k 写出 + 完成 push | 第 843–891 行 | 全部字段写入 `CompletionDescriptor` |

接下来逐文件、逐函数讲解。

---

## 文件头与命名空间

```cpp
// query_traversal.cuh:1-10
#pragma once

#include "gpu_search/delta_scan_budget.hh"
#include "gpu_search/dynamic_route_consistency.hh"
#include "gpu_search/initial_seed_budget.hh"
#include "gpu_search/persistent_kernel/rdma_cache.cuh"

#include <cuda/atomic>

namespace gpu_search::persistent_kernel_detail {
```

这个头文件只被持久化 kernel 的运行时实现包含（见第 17、21 课）。它把所有 host 端不可见的 device-only 帮助函数集中在 `persistent_kernel_detail` 命名空间内。`rdma_cache.cuh` 又会传递性引入 `candidate_scoring.cuh`、`context.cuh` 与 `persistent_kernel.hh`，所以本文件可以直接用 `PersistentKernelParams`、`QueryDescriptor`、`CompletionDescriptor`、`fetch_graph_records_batch`、`approximate_handles_batch`、`exactify_into_beam`、`merge_approximate_into_beam`、`release_graph_record`、`graph_record_pointer` 等符号，而无需重复 include。

---

## decode_compact_raw：5 字节 RemotePtr → 16/48 位 raw 地址

```cpp
// query_traversal.cuh:12-21
__device__ u64 decode_compact_raw(const u8* source, u32 shard_bits) {
  u64 packed = 0;
  for (u32 byte = 0; byte < 5; ++byte) packed |= static_cast<u64>(source[byte]) << (8 * byte);
  if (packed == ((1ull << 40) - 1ull) || shard_bits >= 16) return 0;
  const u32 offset_bits = 40 - shard_bits;
  const u64 offset_mask = (1ull << offset_bits) - 1ull;
  const u32 shard = static_cast<u32>(packed >> offset_bits);
  const u64 offset = (packed & offset_mask) * 8;
  return (static_cast<u64>(shard) << 48) | offset;
}
```

这是与 `src/vamana/hot_graph.hh` 的 `encode_remote_ptr`/`decode_remote_ptr` 严格对应的 device 内联版本，但**返回值不是 `RemotePtr` 结构体，而是已经放回 `[16b memory_node | 48b byte_offset]` 标准布局的 `u64 raw`**（见 `src/remote_pointer.hh:9`）。这样后续可直接喂给 `handle_from_raw`、`resolve_handle`、`delta_slot_from_raw` 等函数。

逐行解读：

1. **小端 5 字节解包**：`kCompactPointerBytes = 5`（`hot_graph.hh:15`），用循环把 5 个字节拼成 40 位的 `packed`，低字节在前。
2. **null/shard_bits 守卫**：`(1ull<<40)-1ull` 是 `kNullCompactPointer`（`hot_graph.hh:17`），表示"空邻居"，这里直接返回 0（即 `RemotePtr{}` 的 `raw_address == 0`）。`shard_bits >= 16` 是编码端的硬上限（`encode_remote_ptr` 在该条件下会写入全 0xff 并返回 false），device 端同样视作 null。
3. **位拆分**：高 `shard_bits` 位是 `shard`（存储节点 ordinal），低 `40 - shard_bits` 位是 **8 字节为单位的偏移单位数**——注意 `offset = (packed & offset_mask) * 8` 把"8 字节槽号"还原成字节偏移。这是 compact 表示法节省空间的关键：所有 graph record 都 8 字节对齐（`encode_remote_ptr` 里的 `ptr.byte_offset() % 8 != 0` 校验），所以 48 位 byte offset 可以无损压成 40 位 unit offset。
4. **重组标准 raw**：`(shard << 48) | offset` 就是 `RemotePtr::store_address` 的二进制布局。

注意 `shard_bits` 来自 `params.graph_shard_bits`（`persistent_kernel.hh:102`），它由 host 端根据 `shard_count` 计算（`hot_graph.hh:48-56` 的 `shard_bits_for`），最多 15，确保 `offset_bits >= 25`，即单 shard 至少能表达 32 MB × 8 = 256 MB 的图区域，足以覆盖典型 graph plane。

---

## DynamicRouteSnapshot 与 seqlock 读

```cpp
// query_traversal.cuh:23-43
struct DynamicRouteSnapshot {
  u64 epoch{};
  u64 remote_node{};
  u32 id{};
  u32 generation{};
  u32 shard{};
  u32 flags{};
};

template <typename T>
__device__ T dynamic_route_atomic_load(const T& value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(
    const_cast<T&>(value));
  return reference.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ void dynamic_route_atomic_store(T& destination, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(destination);
  reference.store(value, cuda::memory_order_relaxed);
}
```

`DynamicRouteSnapshot` 是 `DeviceDynamicRouteSlot`（`types.hh:103-112`）去掉 `sequence` 和 `command_id` 的"数据视图"。`DeviceDynamicRouteSlot.sequence` 是 device-scope seqlock 的版本号（`types.hh:99-102` 注释说明：奇数表示 writer 正在更新，偶数表示数据稳定）。

`dynamic_route_atomic_load` / `_store` 是对 `cuda::atomic_ref` 的薄封装，用 `memory_order_relaxed`——因为 seqlock 的 acquire 语义已经在 `sequence` 字段的两次 `memory_order_acquire` 加载中保证（见下文 `score_dynamic_route_slot`）。这两个模板只是为了在 const 视图上做原子读（`const_cast` 是必要的，因为 `atomic_ref` 要求非 const 引用）。

### score_dynamic_route_slot：seqlock 双重校验 + 可见性 + PQ 评分

```cpp
// query_traversal.cuh:45-97
__device__ bool score_dynamic_route_slot(
    const PersistentKernelParams& params, u32 slot_index,
    u64 snapshot_epoch, const f32* query_lut,
    DynamicRouteSnapshot& result, f32& distance) {
  if (params.dynamic_route_slots == nullptr ||
      params.dynamic_route_pq_codes == nullptr ||
      params.pq_code_bytes == 0 ||
      slot_index >= params.dynamic_route_capacity) {
    return false;
  }
  const DeviceDynamicRouteSlot& source =
    params.dynamic_route_slots[slot_index];
  cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
    const_cast<u64&>(source.sequence));
  // A writer window is very short.  Two attempts recover from the common
  // boundary race without ever making a query wait on mutation publication.
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    const u64 before = sequence.load(cuda::memory_order_acquire);
    if ((before & 1u) != 0) continue;
    DynamicRouteSnapshot candidate{
      .epoch = dynamic_route_atomic_load(source.epoch),
      .remote_node = dynamic_route_atomic_load(source.remote_node),
      .id = dynamic_route_atomic_load(source.id),
      .generation = dynamic_route_atomic_load(source.generation),
      .shard = dynamic_route_atomic_load(source.shard),
      .flags = dynamic_route_atomic_load(source.flags),
    };
    const u64 after = sequence.load(cuda::memory_order_acquire);
    if (!dynamic_route_window_stable(before, after)) continue;
    const bool live = (candidate.flags & kDynamicRouteLive) != 0;
    if (!live || (candidate.flags & ~kDynamicRouteLive) != 0 ||
        candidate.epoch == 0 || candidate.epoch > snapshot_epoch ||
        candidate.remote_node == 0 ||
        candidate.shard >= params.num_shards ||
        slot_index / kDynamicRouteSlotsPerShard != candidate.shard ||
        static_cast<u32>(candidate.remote_node >> 48) != candidate.shard) {
      return false;
    }
    const f32 candidate_distance = approximate_entry(
      params, query_lut,
      params.dynamic_route_pq_codes +
        static_cast<size_t>(slot_index) * params.pq_code_bytes);
    // PQ bytes are part of the same slot transaction. A writer marks the
    // sequence odd before changing either code or metadata; revalidate only
    // after scoring so an old pointer can never be paired with a new code.
    const u64 scored_after = sequence.load(cuda::memory_order_acquire);
    if (!dynamic_route_window_stable(before, scored_after)) continue;
    result = candidate;
    distance = candidate_distance;
    return true;
  }
  return false;
}
```

这是 dynamic route overlay 的核心读路径。逐段剖析：

1. **守卫**：`dynamic_route_slots`、`dynamic_route_pq_codes` 都得非空，`pq_code_bytes != 0`，`slot_index < dynamic_route_capacity`。任何一个不满足说明 host 没启用 dynamic route，直接 false。
2. **seqlock 入口**：拿到 `source.sequence` 的 `atomic_ref`。注释明确："writer window 非常短，两次尝试即可从最常见的边界竞争恢复，且永远不让查询等待 mutation 发布"。这与 `types.hh:99-102` 的设计一致——query CTA 永远不 spin 等 writer。
3. **attempt 循环**：
   - `before = sequence.load(acquire)`：第一次 acquire 读取版本号。
   - `(before & 1u) != 0` 说明 writer 正在更新，`continue` 进下一次 attempt（最多两次）。
   - 用 `dynamic_route_atomic_load` 逐字段读 `epoch`、`remote_node`、`id`、`generation`、`shard`、`flags`（都是 relaxed，因为 acquire 已经在 sequence 上做过了）。
   - `after = sequence.load(acquire)`：第二次 acquire。`dynamic_route_window_stable(before, after)`（`dynamic_route_consistency.hh:17-20`）要求 `before` 是偶数且 `before == after`，即整个读窗口内 writer 没动过。如果不稳定，`continue`。
4. **可见性与一致性校验**（一大块条件）：
   - `live = (flags & kDynamicRouteLive) != 0`：必须被标记为活跃（`types.hh:86` 的 `kDynamicRouteLive = 1u`）。
   - `(flags & ~kDynamicRouteLive) != 0`：除 live 位外其它位非零——这是预留扩展位的保守检查，任何未知 flag 都拒绝。
   - `epoch == 0`：未初始化。
   - `epoch > snapshot_epoch`：发布时间晚于本查询的快照——**这正是 snapshot_epoch 绑定可见性的关键检查**（见第 15 课 delta 发布）。
   - `remote_node == 0`：null RemotePtr。
   - `shard >= num_shards`：非法 shard。
   - `slot_index / kDynamicRouteSlotsPerShard != candidate.shard`：slot 必须落在它声称的 shard 分区内（`kDynamicRouteSlotsPerShard = 8`，`types.hh:85`；slot 分区由 host 安装时保证，这里是防御性检查）。
   - `static_cast<u32>(remote_node >> 48) != candidate.shard`：RemotePtr 的高 16 位（memory_node）必须与 `shard` 字段一致——防止 writer 在更新 `remote_node` 和 `shard` 时出现跨字段错位。
5. **PQ 评分**：`approximate_entry`（`candidate_scoring.cuh:431-441`）用 `query_lut` 与 `dynamic_route_pq_codes + slot_index * pq_code_bytes` 做 PQ ADC 估计。注意 PQ code 与 metadata 是**同一个事务**的一部分——writer 在改任何一边之前都会把 sequence 置奇。
6. **scored 后再校验**：`scored_after = sequence.load(acquire)`。注释解释：PQ 字节也是 slot 事务的一部分，writer 在改 code 或 metadata 前都会置奇；只在评分后再校验一次，确保"旧的 pointer 永远不会配上新的 code"。如果 unstable，`continue` 进下一次 attempt。
7. **返回**：成功则写 `result` 与 `distance`，返回 true。两次 attempt 都不稳定则返回 false，调用方就跳过这个 slot。

这个函数把 seqlock 的"读窗口"扩展到了"metadata + PQ code"两段，确保不会读到 torn state。它是 dynamic route overlay 在遍历中参与的入口之一（另一个是 `add_delta_candidates` 里的 delta scan）。

---

## add_delta_candidates：delta overlay 注入 beam

```cpp
// query_traversal.cuh:99-254
__device__ void add_delta_candidates(const PersistentKernelParams& params,
                                     const QueryDescriptor& descriptor,
                                     const f32* query, const f32* query_lut,
                                     u32* beam_handles,
                                     u32* beam_ids, f32* beam_distances,
                                     u8* beam_expanded, u32& beam_count,
                                     u32 beam_capacity,
                                     const u32* selected_anchors,
                                     u32 selected_anchor_count,
                                     u32* scan_slots,
                                     u32& scanned_records,
                                     u32& scored_records,
                                     u32& truncated_buckets) {
  __shared__ u32 delta_count_snapshot;
  if (threadIdx.x == 0) {
    delta_count_snapshot = min(load_cg(params.delta_count), params.delta_capacity);
    scanned_records = 0;
    scored_records = 0;
    truncated_buckets = 0;
  }
  __syncthreads();
  const u32 count = delta_count_snapshot;
  if (count == 0) return;
```

入口：thread0 用 `load_cg`（`candidate_scoring.cuh:80-90`，`ld.global.cg` 全局可见但不在 L2 缓存的 load）读 `delta_count`——这是 delta 表的高水位线（reused-slot），clip 到 `delta_capacity`。三个统计字段清零。

```cpp
  __shared__ u32 candidate_handles[256];
  __shared__ u32 candidate_slots[256];
  __shared__ f32 candidate_distances[256];
  __shared__ u32 selected_bucket_nonempty;
  u32 local_slot = UINT32_MAX;
  f32 local_approximation = FLT_MAX;
```

`candidate_*` 是每线程一个候选（最多 256 线程，与 `kPersistentQueryThreads = 256` 对齐）。每个线程会扫描自己分到的 delta slot，保留 PQ 距离最小的那个作为本线程的候选。

### 快速跳过：anchor 已选但分桶为空

```cpp
  // query_traversal.cuh:129-150
  if (params.anchor_count != 0 && selected_anchor_count != 0) {
    if (threadIdx.x == 0) selected_bucket_nonempty = 0;
    __syncthreads();
    for (u32 probe = threadIdx.x; probe < selected_anchor_count;
         probe += blockDim.x) {
      const u32 selected_anchor = selected_anchors[probe];
      if (selected_anchor != UINT32_MAX) {
        const u32 head = load_cg(
          params.delta_bucket_heads + selected_anchor);
        if (head != UINT32_MAX && head < count) {
          atomicExch(&selected_bucket_nonempty, 1u);
        }
      }
    }
    __syncthreads();
    if (selected_bucket_nonempty == 0) return;
  }
```

注释解释：`delta_count` 是 reused-slot 的高水位线，即使所有 mutable record 都被 unlink 了它也可能保持非零。在 anchor-backed 的常规配置下，如果本查询选中的 anchor 桶都没有链表头（`delta_bucket_heads[anchor] == UINT32_MAX` 或越界），就直接 return，**不触碰 fixed scan scratch**。这避免了无意义的全表扫描。注意"发布如果与本查询竞争，它带的是更新的 epoch，对本快照不可见"——所以这种跳过是安全的。

### 扫描槽位分配

```cpp
  // query_traversal.cuh:152-157
  static_assert(kDeltaScanRecordBudget <= kPersistentMaxMergeCandidates);
  for (u32 index = threadIdx.x; index < kDeltaScanRecordBudget;
       index += blockDim.x) {
    scan_slots[index] = UINT32_MAX;
  }
  __syncthreads();
```

`kDeltaScanRecordBudget = 2048`（`delta_scan_budget.hh:12`），与 `kPersistentMaxMergeCandidates` 对齐（复用 `navigation_handles` 这块 scratch，见下文 `process_query` 调用处）。先把所有槽位置 `UINT32_MAX`。

#### 无 anchor 分桶：扫最近高水位窗口

```cpp
  // query_traversal.cuh:159-172
  if (params.anchor_count == 0 || selected_anchor_count == 0) {
    const u32 scan_count = min(count, kDeltaScanRecordBudget);
    // Without anchor buckets, prefer the append-most-recent high-watermark
    // window. Slot reuse can make this approximate, but the work remains
    // bounded and the graph/dynamic route remain the authoritative paths.
    const u32 scan_begin = count - scan_count;
    for (u32 index = threadIdx.x; index < scan_count;
         index += blockDim.x) {
      scan_slots[index] = scan_begin + index;
    }
    if (threadIdx.x == 0) {
      scanned_records = scan_count;
      truncated_buckets = count > scan_count ? 1 : 0;
    }
  }
```

无 anchor 时退化为"扫最近 2048 个 slot 的 append 窗口"。注释承认 slot reuse 会让这个窗口是近似的，但工作量有上界，且 graph + dynamic route 仍是权威路径。

#### 有 anchor 分桶：沿链表收集每个 anchor 的前缀

```cpp
  // query_traversal.cuh:173-206
  } else {
    // Bucket insertion is at the head, so this covers the newest fixed-budget
    // prefix of every selected anchor. One thread follows each singly-linked
    // list; unlike the old partitioned loop, links are never redundantly
    // traversed by every worker assigned to the same anchor.
    u32 local_discovered = 0;
    u32 local_truncated = 0;
    for (u32 probe = threadIdx.x; probe < selected_anchor_count;
         probe += blockDim.x) {
      const DeltaScanSegment segment = delta_scan_segment(
        probe, selected_anchor_count, kDeltaScanRecordBudget);
      const u32 selected_anchor = selected_anchors[probe];
      u32 slot = selected_anchor == UINT32_MAX
        ? UINT32_MAX : load_cg(params.delta_bucket_heads + selected_anchor);
      u32 discovered = 0;
      while (slot != UINT32_MAX && slot < count && discovered < segment.count) {
        scan_slots[segment.offset + discovered] = slot;
        slot = load_cg(params.delta_next + slot);
        ++discovered;
      }
      local_discovered += discovered;
      if (discovered == segment.count && slot != UINT32_MAX && slot < count) {
        ++local_truncated;
      }
    }
    if (local_discovered != 0) {
      atomicAdd(&scanned_records, local_discovered);
    }
    // This is structural prefix truncation, not a claim that every record
    // beyond the prefix would be visible to this query snapshot.
    if (local_truncated != 0) {
      atomicAdd(&truncated_buckets, local_truncated);
    }
  }
  __syncthreads();
```

`delta_scan_segment(probe, selected_anchor_count, 2048)`（`delta_scan_budget.hh:22-33`）把 2048 的预算**按 selected_anchor_count 均分**：每个 anchor 分到 `base = 2048 / segment_count` 个槽，前 `remainder = 2048 % segment_count` 个 anchor 多分 1 个。返回 `{offset, count}`。

每个线程负责一个 `probe`（即一个 selected anchor），从 `delta_bucket_heads[anchor]` 开始沿 `delta_next` 单链表走，最多走 `segment.count` 步，把 slot 号写进 `scan_slots[segment.offset + discovered]`。如果走满 budget 还有后续节点，记一次 `local_truncated`。

注释强调两点：
- "桶插入在头部，所以这覆盖了每个选中 anchor 最新的 fixed-budget 前缀"——delta 链表是 LIFO，最新插入在最前。
- "这是结构性前缀截断，不代表截断后的记录对本快照都可见"——可见性在评分阶段用 `delta_visible` 再判一次。

### 评分与去重

```cpp
  // query_traversal.cuh:209-236
  u32 local_scored = 0;
  for (u32 index = threadIdx.x; index < kDeltaScanRecordBudget;
       index += blockDim.x) {
    const u32 slot = scan_slots[index];
    if (slot == UINT32_MAX || slot >= count) continue;
    const DeviceDeltaRecord& record = params.delta_records[slot];
    if (!delta_visible(record, descriptor.snapshot_epoch)) continue;
    ++local_scored;
    const f32 approximation = approximate_entry(
      params, query_lut,
      params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
    if (approximation < local_approximation) {
      local_approximation = approximation;
      local_slot = slot;
    }
  }
  if (local_scored != 0) {
    atomicAdd(&scored_records, local_scored);
  }
  candidate_slots[threadIdx.x] = local_slot;
  candidate_handles[threadIdx.x] = local_slot == UINT32_MAX
    ? UINT32_MAX
    : handle_from_raw(params, params.delta_records[local_slot].remote_node);
  candidate_distances[threadIdx.x] = local_slot == UINT32_MAX
    ? FLT_MAX
    : exact_storage_distance(params, query,
        params.delta_vectors + static_cast<size_t>(local_slot) * params.vector_bytes);
  __syncthreads();
```

每个线程跨步扫描 `scan_slots`（stride = blockDim.x）。对每个非空 slot：
- 读 `DeviceDeltaRecord`（`persistent_kernel.hh:58-68`）。
- `delta_visible(record, snapshot_epoch)`（`candidate_scoring.cuh:416-421`）：要求 `record.epoch <= snapshot_epoch` && (`superseded_epoch == 0` || `> snapshot_epoch`) && `flags` 没有 `kDeltaDeleted` 也没有 `kDeltaDurable`——**这就是 snapshot_epoch 绑定可见性的另一处检查，以及 tombstone（kDeltaDeleted）过滤**。注意这里用 `delta_visible` 而不是 `delta_code_visible`：因为这里要把候选加入精排 beam（要拿真实向量），所以 durable 也要排除（durable 表示已落盘到 base，由 base override 接管）。
- 不可见则跳过；可见则 `approximate_entry` 用 `delta_pq_codes` 评分，更新本线程最小。
- 累计 `local_scored` 后 atomicAdd 到全局 `scored_records`。

线程结束扫描后，把"本线程最佳 slot"写进 `candidate_slots[threadIdx.x]`，并：
- `candidate_handles` 用 `handle_from_raw`（`candidate_scoring.cuh:343-361`）把 delta 的 `remote_node` 转 handle。注意 delta 的 `remote_node` 是 dynamic 节点（落在 `dynamic_base_offset`），所以 `handle_from_raw` 会返回 `kDeltaHandleBit | ...` 形式的 dynamic handle。
- `candidate_distances` 用 `exact_storage_distance`（`rdma_cache.cuh:14-23`）对 `delta_vectors`（delta 的原向量缓冲）做精确 L2——**这是 delta overlay 的关键优势：它的原向量就在本地 GPU 显存，不需要远端读**。

### 单线程合并进 beam

```cpp
  // query_traversal.cuh:237-253
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < min(blockDim.x, 256u); ++index) {
      const u32 handle = candidate_handles[index];
      if (handle == UINT32_MAX) continue;
      bool duplicate = false;
      for (u32 beam = 0; beam < beam_count; ++beam) {
        if (beam_handles[beam] == handle) duplicate = true;
      }
      if (!duplicate) {
        const u32 slot = candidate_slots[index];
        beam_insert(beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                    beam_capacity, handle, params.delta_records[slot].id,
                    candidate_distances[index]);
      }
    }
  }
  __syncthreads();
}
```

thread0 串行遍历每个线程的候选，对每个 handle 在 beam 里做线性去重（`beam_count <= beam_capacity = 128`，所以 O(256×128) 可接受），未重复则 `beam_insert`（`candidate_scoring.cuh:479-499`）。`beam_insert` 在 beam 未满时 append，已满时替换最差——这里传的 `beam_capacity` 是 `params.final_rerank_width`（见调用处第 832 行），即精排宽度。

注意 delta 候选的 `id` 直接取 `params.delta_records[slot].id`，距离是精确 L2 而非 PQ 近似——这意味着 delta 候选**跳过了 PQ 评分阶段，直接以精确距离进入精排 beam**。这与 delta overlay 的定位一致：它是图尚未追上时的短期可见性补救，所以给它最短路径。

---

## process_query：主循环

### 入口校验与错误早退

```cpp
// query_traversal.cuh:256-279
__device__ void process_query(const PersistentKernelParams& params,
                              const QueryDescriptor& descriptor) {
  const u32 query_slot = descriptor.query_slot;
  __shared__ u64 query_started_cycles;
  if (threadIdx.x == 0) query_started_cycles = clock64();
  __syncthreads();
  CompletionDescriptor completion{
    .request_id = descriptor.request_id,
    .snapshot_epoch = descriptor.snapshot_epoch,
    .query_slot = query_slot,
  };
  if (query_slot >= params.query_slots || descriptor.dim != params.dim ||
      descriptor.query_dtype > 2 || params.decoded_queries == nullptr ||
      params.navigation_candidate_handles == nullptr ||
      params.navigation_candidate_distances == nullptr ||
      descriptor.k == 0 || descriptor.k > descriptor.result_capacity) {
    if (threadIdx.x == 0) {
      completion.status = -EINVAL;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
```

- thread0 在 `query_started_cycles` 记录起始 cycle，这是整条查询的"挂钟"。
- 默认构造 `CompletionDescriptor`，预填 `request_id` / `snapshot_epoch` / `query_slot`。
- 一长串校验：`query_slot` 越界、`dim` 不匹配、`query_dtype > 2`（`candidate_scoring.cuh:129-141` 的 `query_component` 支持 0=f32、1=u8、2=i8）、关键缓冲 null、`k==0` 或 `k > result_capacity`。任一失败：thread0 写 `-EINVAL`、记 `gpu_cycles`、`device_ring_push` 推完成事件，然后 block 同步后 return。

这个"早退"路径是第 14 课完成回写的最小形态——即使查询非法，也要把完成事件推回 host，否则 `completion_loop` 会一直等。

### 计时变量与 prepare 阶段开始

```cpp
// query_traversal.cuh:281-296
  const u8* query_input = reinterpret_cast<const u8*>(descriptor.query_device_address);
  __shared__ u64 prepare_started_cycles;
  __shared__ u64 graph_phase_cycles;
  __shared__ u64 score_phase_cycles;
  __shared__ u64 beam_phase_cycles;
  __shared__ u64 exact_phase_cycles;
  __shared__ u64 delta_scan_started_cycles;
  __shared__ u64 phase_started_cycles;
  if (threadIdx.x == 0) {
    prepare_started_cycles = clock64();
    graph_phase_cycles = 0;
    score_phase_cycles = 0;
    beam_phase_cycles = 0;
    exact_phase_cycles = 0;
  }
  __syncthreads();
```

`query_input` 是 host 注册的设备指针（`query_device_address`），指向原始字节（dtype 由 `query_dtype` 决定）。六个 cycle 计数器：`prepare_started_cycles` 标记 prepare 起点；`graph/score/beam/exact_phase_cycles` 是累加器（主循环每轮都加）；`phase_started_cycles` 是"当前轮当前阶段"的临时起点，会被反复覆盖；`delta_scan_started_cycles` 在最后 delta scan 阶段用。

thread0 把四个累加器清零（delta_scan 不需要累加，单次）。

### 阶段 1a：query → float 解码

```cpp
// query_traversal.cuh:297-301
  f32* query = params.decoded_queries + static_cast<size_t>(query_slot) * params.dim;
  for (u32 dimension = threadIdx.x; dimension < params.dim; dimension += blockDim.x) {
    query[dimension] = query_component(query_input, descriptor.query_dtype, dimension);
  }
  __syncthreads();
```

`decoded_queries` 是 per-slot 的 f32 缓冲（`query_slot * dim`）。每个线程跨步把 `query_input` 的某个维度按 `query_dtype` 解码成 f32（`candidate_scoring.cuh:129-141`）。这一步把异构 dtype（f32/u8/i8）统一成 f32，后续 OPQ/LUT/精确距离都基于这个 f32 buffer。

### 阶段 1b：OPQ 旋转

```cpp
// query_traversal.cuh:302-316
  f32* transformed = params.transformed_queries +
    static_cast<size_t>(query_slot) * params.dim;
  for (u32 row = threadIdx.x; row < params.dim; row += blockDim.x) {
    if (params.opq_matrix == nullptr) {
      transformed[row] = query[row];
      continue;
    }
    f32 value = 0.0f;
    const f32* matrix_row = params.opq_matrix + static_cast<size_t>(row) * params.dim;
    for (u32 column = 0; column < params.dim; ++column) {
      value += matrix_row[column] * query[column];
    }
    transformed[row] = value;
  }
  __syncthreads();
```

OPQ（Optimized Product Quantization）旋转：`transformed = opq_matrix · query`。`opq_matrix` 是 `dim × dim` 行主序矩阵，每行一个输出维度。如果 `opq_matrix == nullptr`（未启用 OPQ），直接拷贝。这个旋转把查询变换到 PQ 训练时的旋转空间，与存储侧的 PQ code 对齐。

注意这里每个线程算一行（内层 `column` 循环串行），是典型的"每线程一行"的 GEMV。`dim` 通常 96~512，256 线程足够覆盖。

### 阶段 1c：PQ 查找表（LUT）构建

```cpp
// query_traversal.cuh:317-333
  f32* query_lut = params.query_luts +
    static_cast<size_t>(query_slot) * params.pq_subquantizers * 256;
  const u32 table_entries = params.pq_subquantizers * 256;
  for (u32 index = threadIdx.x; index < table_entries; index += blockDim.x) {
    const u32 subquantizer = index / 256;
    const f32* query_subvector = transformed +
      static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
    const f32* centroid_subvector = params.pq_centroids +
      static_cast<size_t>(index) * params.pq_subvector_dim;
    f32 distance = 0.0f;
    for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
      const f32 difference = query_subvector[dimension] - centroid_subvector[dimension];
      distance += difference * difference;
    }
    query_lut[index] = distance;
  }
  __syncthreads();
```

PQ ADC（Asymmetric Distance Computation）的 LUT：对每个 subquantizer 的 256 个码字，预先算出"查询子向量与该码字的 L2 距离"。布局：
- `query_lut` 大小 = `pq_subquantizers × 256`。
- `index = subquantizer * 256 + code`，所以 `subquantizer = index / 256`。
- `query_subvector = transformed + subquantizer * pq_subvector_dim`。
- `centroid_subvector = pq_centroids + index * pq_subvector_dim`（`pq_centroids` 是 `[pq_subquantizers × 256, pq_subvector_dim]` 行主序）。
- 算 L2 距离平方，存进 `query_lut[index]`。

之后 `approximate_entry`（`candidate_scoring.cuh:431-441`）只需对每个 subquantizer 查一次 `query_lut[subquantizer*256 + code[subquantizer]]` 并累加——这就是 PQ ADC 的 O(M) 评分，比精确 L2 快得多。LUT 构建是一次性 O(M×256×subvector_dim) 的工作，分摊到每个候选评分上极便宜。

### 阶段 2：beam/visited/merge scratch 初始化

```cpp
// query_traversal.cuh:335-364
  __shared__ u32 shared_beam_handles[kPersistentMaxBeam];
  __shared__ u32 shared_beam_ids[kPersistentMaxBeam];
  __shared__ f32 shared_beam_distances[kPersistentMaxBeam];
  __shared__ u8 shared_beam_expanded[kPersistentMaxBeam];
  __shared__ CandidateWorkspace candidate_workspace;
  u32* merge_handles = candidate_workspace.arrays.handles;
  u32* merge_ids = candidate_workspace.arrays.ids;
  f32* merge_distances = candidate_workspace.arrays.distances;
  u8* merge_expanded = candidate_workspace.arrays.expanded;
  u32* navigation_handles = params.navigation_candidate_handles +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  f32* navigation_distances = params.navigation_candidate_distances +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  u32* beam_handles = shared_beam_handles;
  u32* beam_ids = shared_beam_ids;
  f32* beam_distances = shared_beam_distances;
  u8* beam_expanded = shared_beam_expanded;
  const u32 traversal_capacity = min(kPersistentMaxBeam, params.traversal_beam_width);
  u32* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < traversal_capacity; index += blockDim.x) {
    beam_handles[index] = UINT32_MAX;
    beam_ids[index] = UINT32_MAX;
    beam_distances[index] = FLT_MAX;
    beam_expanded[index] = 0;
  }
  for (u32 index = threadIdx.x; index < params.visited_capacity; index += blockDim.x) {
    visited[index] = UINT32_MAX;
  }
  __syncthreads();
```

关键数据结构（全部 device shared memory 或 global scratch）：

- **beam**（`shared_beam_*`，128 槽）：traversal beam。`kPersistentMaxBeam = 128`（`persistent_kernel.hh:13`）。`traversal_capacity = min(128, params.traversal_beam_width)`，实际生效宽度由 host 配置。每槽 4 字段：`handle`（RemotePtr-derived 32 位句柄，static 是 ordinal、dynamic 是 `kDeltaHandleBit | ...`）、`id`（向量逻辑 id）、`distance`（PQ 近似或精确距离）、`expanded`（是否已展开过图邻居）。
- **CandidateWorkspace**（`candidate_scoring.cuh:20-23`）：含 `arrays`（`kPersistentMaxExact * 2 = 512` 槽的 handles/ids/distances/expanded）+ sort 用的 CUB radix sort temp storage。这是 merge scratch，beam 合并时把"现有 beam + 新候选"放进 `merge_*`，排序后取 top。
- **navigation_handles / navigation_distances**：global scratch，`kPersistentMaxMergeCandidates = 2048` 槽（`persistent_kernel.hh:20`）。这里复用三处：(1) 阶段 2 存入口候选；(2) 主循环里存"flatten 后的邻居 handle"；(3) `add_delta_candidates` 里当 `scan_slots` 用（见第 832-834 行调用）。复用是安全的，因为这三处时序不重叠。
- **visited**（`visited_hash`，global）：开放寻址哈希表，`visited_capacity` 槽（host 配置，2 的幂）。`UINT32_MAX` 表示空。去重靠 `insert_visited`（`candidate_scoring.cuh:209-219`），下文详解。

初始化把 beam 前 `traversal_capacity` 槽置空，visited 全清空。

### 阶段 2a：anchor 入口打分

```cpp
// query_traversal.cuh:366-458
  __shared__ u32 beam_count;
  __shared__ u32 rerank_handles[kPersistentMaxExact];
  __shared__ u32 rerank_ids[kPersistentMaxExact];
  __shared__ f32 rerank_distances[kPersistentMaxExact];
  __shared__ u32 rerank_count;
  __shared__ u32 total_exact_reads;
  __shared__ u32 total_exact_cache_hits;
  __shared__ u32 seed_count;
  __shared__ u32 dynamic_seed_count;
  __shared__ u32 selected_anchor_count;
  __shared__ u32 anchor_best_indices[256];
  if (params.anchor_count != 0 && params.anchor_vectors != nullptr &&
      params.anchor_handles != nullptr && params.anchor_pq_codes != nullptr) {
    constexpr u32 local_anchor_candidates = 2;
    const u32 candidates_per_thread =
      blockDim.x == kApproximateSortThreadsCompact ? 2u : 1u;
    u32 local_anchors[local_anchor_candidates];
    u32 local_handles[local_anchor_candidates];
    f32 local_distances[local_anchor_candidates];
    for (u32 index = 0; index < local_anchor_candidates; ++index) {
      local_anchors[index] = UINT32_MAX;
      local_handles[index] = UINT32_MAX;
      local_distances[index] = FLT_MAX;
    }
    for (u32 anchor = threadIdx.x; anchor < params.anchor_count; anchor += blockDim.x) {
      const u32 handle = params.anchor_handles[anchor];
      const f32 distance = approximate_entry(
        params, query_lut,
        params.anchor_pq_codes + static_cast<size_t>(anchor) * params.pq_code_bytes);
      u32 worst = 0;
      for (u32 index = 1; index < candidates_per_thread; ++index) {
        if (candidate_less(local_handles[worst], local_distances[worst],
                           local_handles[index], local_distances[index])) {
          worst = index;
        }
      }
      if (candidate_less(handle, distance,
                         local_handles[worst], local_distances[worst])) {
        local_anchors[worst] = anchor;
        local_handles[worst] = handle;
        local_distances[worst] = distance;
      }
    }
    for (u32 index = 0; index < candidates_per_thread; ++index) {
      const u32 output = threadIdx.x * candidates_per_thread + index;
      merge_handles[output] = local_handles[index];
      merge_ids[output] = local_anchors[index];
      merge_distances[output] = local_distances[index];
      merge_expanded[output] = 0;
    }
    __syncthreads();
```

anchor 是 host 预选的"入口代表点"（见第 6 课 anchor/idmap）。如果有 anchor：
- 每个线程跨步扫所有 anchor，用 `approximate_entry` 对 `anchor_pq_codes` 评分。
- 每线程保留 `candidates_per_thread` 个最佳（compact block=128 线程时 2 个，wide block=256 线程时 1 个——这是为了控制 merge scratch 总量在 `blockDim.x * candidates_per_thread` 内）。
- `candidate_less`（`candidate_scoring.cuh:501-505`）是距离优先、handle 次之的比较谓词，保证全序。
- 每线程的最佳写进 `merge_handles/output`（`output = threadIdx.x * candidates_per_thread + index`），共 `blockDim.x * candidates_per_thread` 个候选。

```cpp
    // query_traversal.cuh:417-431
    const u32 approximate_anchor_candidates =
      blockDim.x * candidates_per_thread;
    sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                    approximate_anchor_candidates);
    if (threadIdx.x == 0) {
      u32 valid = 0;
      while (valid < approximate_anchor_candidates &&
             merge_ids[valid] != UINT32_MAX &&
             isfinite(merge_distances[valid]) &&
             merge_distances[valid] != FLT_MAX) {
        ++valid;
      }
      selected_anchor_count = min(valid, 256u);
    }
    __syncthreads();
```

`sort_candidates`（`candidate_scoring.cuh:513-553`）是 bitonic 排序，把 `merge_*` 按距离升序。thread0 数有效候选数，clip 到 256（`anchor_best_indices[256]` 的上限）。

```cpp
    // query_traversal.cuh:432-450
    for (u32 index = threadIdx.x; index < selected_anchor_count;
         index += blockDim.x) {
      merge_distances[index] = exact_anchor_distance(params, query, merge_ids[index]);
      merge_expanded[index] = 0;
    }
    __syncthreads();
    sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                    selected_anchor_count);
    if (threadIdx.x == 0) {
      selected_anchor_count = min(
        selected_anchor_count,
        max(min(params.entry_seed_count, traversal_capacity),
            min(params.delta_anchor_probes, kPersistentMaxAnchorProbes)));
      seed_count = min(selected_anchor_count,
                       min(params.entry_seed_count, traversal_capacity));
      for (u32 index = 0; index < selected_anchor_count; ++index) {
        anchor_best_indices[index] = merge_ids[index];
      }
    }
    __syncthreads();
```

关键一步：**对 top anchor 用 `exact_anchor_distance`（`rdma_cache.cuh:25-35`）做精确 L2 重排**。`anchor_vectors` 是 SoA 布局（`params.anchor_vectors[dimension * anchor_count + anchor]`），在本地显存，所以精确距离很便宜。重排后再 sort。

`selected_anchor_count` 被压到 `max(min(entry_seed_count, traversal_capacity), min(delta_anchor_probes, kPersistentMaxAnchorProbes))`——这个 max 同时服务于"入口 seed 数"和"delta anchor probe 数"两个用途（delta scan 时会复用 `anchor_best_indices`）。`seed_count` 进一步 clip 到 `min(entry_seed_count, traversal_capacity)`，即真正进入 beam 的 anchor 数。

`anchor_best_indices[]` 记录选中的 anchor 索引（不是 handle），供 delta scan 用。

```cpp
    // query_traversal.cuh:452-458
    for (u32 seed = threadIdx.x; seed < seed_count; seed += blockDim.x) {
      const u32 handle = params.anchor_handles[anchor_best_indices[seed]];
      merge_handles[seed] = handle;
      merge_distances[seed] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      merge_expanded[seed] = 0;
    }
  }
```

最后，对选中的 seed anchor，用 `approximate_handle`（`candidate_scoring.cuh:443-477`）重新评分——这里 `approximate_handle` 会根据 handle 是 static 还是 dynamic 选择不同的 PQ code 源（base `pq_codes` / delta `delta_pq_codes` / resident `resident_pq_codes`），并做可见性判断。**注意 seed 进 beam 用的是 PQ 近似，不是精确距离**——因为 anchor 还要走图遍历展开，精确重排留到最后。

### 阶段 2b：无 anchor 时的静态入口

```cpp
  // query_traversal.cuh:459-475
  } else {
    if (threadIdx.x == 0) {
      seed_count = min(
        min(params.entry_point_count, params.entry_seed_count),
        traversal_capacity);
      selected_anchor_count = 0;
    }
    for (u32 index = threadIdx.x; index < seed_count; index += blockDim.x) {
      const u32 handle = params.entry_points[index];
      merge_handles[index] = handle;
      merge_distances[index] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      merge_expanded[index] = 0;
    }
  }
  __syncthreads();
  const u32 static_seed_count = seed_count;
  if (threadIdx.x == 0) dynamic_seed_count = 0;
  __syncthreads();
```

无 anchor 时退化为 `params.entry_points`（预配置的静态入口点，`persistent_kernel.hh:92`）。`seed_count = min(entry_point_count, entry_seed_count, traversal_capacity)`。同样用 `approximate_handle` 评分。`selected_anchor_count = 0`，所以后续 `add_delta_candidates` 会走"无 anchor"路径扫最近窗口。

`static_seed_count` 锁定静态 seed 数，`dynamic_seed_count` 清零准备累加。

### 阶段 2c：dynamic route 入口打分

```cpp
  // query_traversal.cuh:478-500
  for (u32 slot = threadIdx.x; slot < params.dynamic_route_capacity;
       slot += blockDim.x) {
    DynamicRouteSnapshot dynamic_route;
    f32 distance = FLT_MAX;
    if (!score_dynamic_route_slot(
          params, slot, descriptor.snapshot_epoch, query_lut,
          dynamic_route, distance)) {
      continue;
    }
    const u32 handle = handle_from_raw(params, dynamic_route.remote_node);
    if (handle == UINT32_MAX) continue;
    if (!isfinite(distance) || distance == FLT_MAX) {
      continue;
    }
    const u32 rank = atomicAdd(&dynamic_seed_count, 1u);
    const u32 destination = static_seed_count + rank;
    if (destination >= kPersistentMaxExact * 2) continue;
    merge_handles[destination] = handle;
    merge_ids[destination] = dynamic_route.id;
    merge_distances[destination] = distance;
    merge_expanded[destination] = 0;
  }
  __syncthreads();
```

跨步扫所有 dynamic route slot，每个 slot 用前面讲过的 `score_dynamic_route_slot` 做 seqlock 读 + 可见性 + PQ 评分。成功的转 handle、检查有限性，然后 `atomicAdd` 抢一个 rank，写进 `merge_*[static_seed_count + rank]`。`kPersistentMaxExact * 2 = 512` 是 merge scratch 的容量上限，超过的丢弃。

```cpp
  // query_traversal.cuh:501-513
  if (threadIdx.x == 0) {
    // atomicAdd counts every valid canonical slot, including ranks that did
    // not fit in the fixed merge scratch.  Only the contiguous prefix below
    // was materialized and may participate in the combined route ranking.
    dynamic_seed_count = min(
      dynamic_seed_count,
      static_cast<u32>(kPersistentMaxExact * 2) - static_seed_count);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    seed_count = static_seed_count + dynamic_seed_count;
  }
  __syncthreads();
```

注释解释：`atomicAdd` 统计了所有有效 slot，但只有 contiguous 前缀真的写进了 merge scratch。所以 thread0 把 `dynamic_seed_count` clip 到"剩余容量"（`kPersistentMaxExact*2 - static_seed_count`），然后 `seed_count = static_seed_count + dynamic_seed_count`。

### 阶段 2d：static + dynamic 合并去重，初始化 beam

```cpp
  // query_traversal.cuh:515-563
  // Adaptive routes are replacements inside the configured entry-seed
  // budget, not an extra tier which can silently enlarge the query beam.  Rank
  // the static fallback and canonical dynamic entries in the same PQ distance
  // space, then keep the best unique handles.  This retains the immutable
  // fallback whenever it is more useful while allowing a closer dynamic entry
  // to displace it.  In particular, the usual 32-static + 40-dynamic setup
  // still starts traversal with at most 32 entries rather than 72.
  sort_candidates(merge_handles, nullptr, merge_distances, merge_expanded,
                  seed_count);
  if (threadIdx.x == 0) {
    const u32 initial_seed_capacity = initial_seed_budget(
      params.entry_seed_count, traversal_capacity);
    u32 unique_count = 0;
    for (u32 input = 0;
         input < seed_count && unique_count < initial_seed_capacity; ++input) {
      const u32 handle = merge_handles[input];
      if (handle == UINT32_MAX || !isfinite(merge_distances[input]) ||
          merge_distances[input] == FLT_MAX) {
        continue;
      }
      bool duplicate = false;
      for (u32 prior = 0; prior < unique_count; ++prior) {
        if (merge_handles[prior] == handle) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) continue;
      if (unique_count != input) {
        merge_handles[unique_count] = handle;
        merge_distances[unique_count] = merge_distances[input];
        merge_expanded[unique_count] = 0;
      }
      ++unique_count;
    }
    seed_count = unique_count;
    beam_count = unique_count;
    rerank_count = 0;
    total_exact_reads = 0;
    total_exact_cache_hits = 0;
    for (u32 index = 0; index < beam_count; ++index) {
      beam_handles[index] = merge_handles[index];
      beam_ids[index] = UINT32_MAX;
      beam_distances[index] = merge_distances[index];
      beam_expanded[index] = 0;
      insert_visited(visited, params.visited_capacity, beam_handles[index]);
    }
  }
  __syncthreads();
```

这段注释非常关键：**"adaptive route 是 entry-seed 预算内的替换，不是额外层级"**。典型配置 32 static + 40 dynamic，但 beam 初始仍最多 32 个——dynamic 只在它比某个 static 更好时替换。

`initial_seed_budget(entry_seed_count, traversal_capacity)`（`initial_seed_budget.hh:10-14`）就是 `min(configured, traversal_capacity)`。thread0 在排序后的 merge 里线性扫描，跳过 invalid/duplicate，把前 `initial_seed_capacity` 个唯一 handle 压到前面。然后把它们写进 beam，并 `insert_visited` 标记已访问（防止后续图展开时重复访问）。

`insert_visited`（`candidate_scoring.cuh:209-219`）是开放寻址哈希表插入：`hash32(handle) & mask` 起始，线性探测，`atomicCAS` 抢空槽或发现已存在。返回 true 表示首次插入（未访问过），false 表示已存在。

### 阶段 2e：beam 为空的早退

```cpp
  // query_traversal.cuh:564-572
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
```

如果连一个 seed 都没有（anchor/entry/dynamic route 全空），返回 `-EIO`。这是"无可导航入口"的硬错误。

### 阶段 3 前置：主循环 scratch 初始化

```cpp
  // query_traversal.cuh:574-607
  __shared__ u32 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 cache_hits_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 route_hits_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_cache_slots[kPersistentMaxPrefetch];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_remote_batches;
  __shared__ u32 total_graph_read_retries;
  __shared__ u32 total_graph_rounds;
  __shared__ u32 total_cache_hits;
  __shared__ u32 total_route_hits;
  __shared__ u32 graph_failed;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_remote_batches = 0;
    total_graph_read_retries = 0;
    total_graph_rounds = 0;
    total_cache_hits = 0;
    total_route_hits = 0;
    graph_failed = 0;
  }
  __syncthreads();

  __shared__ u32 expansions;
  if (threadIdx.x == 0) expansions = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    completion.prepare_cycles = clock64() - prepare_started_cycles;
  }
  __syncthreads();
```

主循环 scratch：
- `selected_handles`：本轮选中的未展开候选 handle（最多 `kPersistentMaxPrefetch = 32`）。
- `neighbor_counts` / `neighbor_offsets` / `flattened_neighbors`：每个 selected 的邻居数 + prefix sum + 总数。
- `remote_reads_by_lane` / `cache_hits_by_lane` / `route_hits_by_lane`：per-selected 的统计（区分远端读、cache 命中、route 命中）。
- `graph_cache_slots`：每个 selected 在 graph cache / scratch / route 中的槽位号（含 `kGraphScratchBit` / `kGraphRouteBit` 标志位）。
- 全局累加器：`total_remote_reads`、`total_remote_batches`、`total_graph_read_retries`、`total_graph_rounds`、`total_cache_hits`、`total_route_hits`、`graph_failed`。
- `expansions`：已展开节点总数，与 `params.max_expansions` 比较。

最后 thread0 写 `completion.prepare_cycles`——**这是 prepare 阶段的 cycle 计时点**。注意它发生在主循环开始之前，所以 prepare 包含了 query 解码、OPQ、LUT、入口打分、beam 初始化的全部时间。

### 阶段 3：主循环（graph 展开 + 评分 + beam 合并）

```cpp
  // query_traversal.cuh:608-622
  while (expansions < params.max_expansions) {
    if (threadIdx.x == 0) phase_started_cycles = clock64();
    __syncthreads();
    if (threadIdx.x == 0) {
      selected_count = 0;
      graph_failed = 0;
      const u32 target = min(params.prefetch_depth, params.max_expansions - expansions);
      for (u32 index = 0; index < beam_count && selected_count < target; ++index) {
        if (beam_expanded[index] != 0) continue;
        beam_expanded[index] = 1;
        selected_handles[selected_count++] = beam_handles[index];
      }
    }
    __syncthreads();
    if (selected_count == 0) break;
    if (threadIdx.x == 0) ++total_graph_rounds;
    __syncthreads();
```

循环条件：`expansions < max_expansions`。每轮：
1. thread0 记 `phase_started_cycles`。
2. thread0 串行扫 beam，选最多 `target = min(prefetch_depth, max_expansions - expansions)` 个**未展开**候选进 `selected_handles`。`prefetch_depth` 控制"每轮并行发多少个图读"（默认 ≤ `kPersistentMaxPrefetch = 32`）。
3. `selected_count == 0` 说明 beam 里所有候选都已展开——遍历收敛，break。
4. `total_graph_rounds++`。

#### 阶段 3a：批量发远端图读

```cpp
    // query_traversal.cuh:625-644
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    if (!fetch_graph_records_batch(
          params, descriptor, selected_handles, selected_count,
          graph_cache_slots, remote_reads_by_lane, cache_hits_by_lane,
          route_hits_by_lane,
          &total_remote_batches, &total_graph_read_retries)) {
      if (threadIdx.x == 0) graph_failed = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      graph_phase_cycles += clock64() - phase_started_cycles;
      for (u32 selected = 0; selected < selected_count; ++selected) {
        total_remote_reads += remote_reads_by_lane[selected];
        total_cache_hits += cache_hits_by_lane[selected];
        total_route_hits += route_hits_by_lane[selected];
      }
    }
    __syncthreads();
```

`fetch_graph_records_batch`（`rdma_cache.cuh:842-1036`，第 19 课详解）是批量图读的核心：
- 每个 selected handle 经过 `prepare_graph_record`：先查 anchor graph route（`anchor_graph_slot` 二分 + `kGraphCacheReady` 状态），命中则 `acquired_slot = kGraphRouteBit | route_slot`；再查 graph cache（多 way 组相联，带 generation + TTL 校验），命中则 `acquired_slot = slot`；再尝试 admit + 占一个 Filling 槽发远端读；最后退路是 `graph_scratch`（per-query-slot 的 scratch buffer，`kGraphScratchBit | request_index`）。
- 对每个 shard 调 `direct_fetch_batch`（GPUNetIO verbs，见第 22 课）批量发读请求。
- 读回后用 `valid_graph_record`（`rdma_cache.cuh:669-674`）校验 checksum16——**这是 compact graph 记录的乐观快照验证**，因为 graph entry 可能被 stage2/reverse-edge worker 原地更新，RDMA 完成可能读到新旧混合的 torn 状态。最多重试 `kGraphSnapshotAttempts = 3` 次。

返回 false 则 `graph_failed = 1`。thread0 累加 `graph_phase_cycles`（**graph 阶段计时点**），并汇总 per-lane 统计到 total。

#### 阶段 3b：图读失败的处理

```cpp
    // query_traversal.cuh:645-672
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        const u32 slot = graph_cache_slots[selected];
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
        if (lane_in_warp == 0) graph_cache_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_rounds = total_graph_rounds;
        completion.cache_hits = total_cache_hits;
        completion.route_hits = total_route_hits;
        completion.exact_vectors = total_exact_reads;
        completion.exact_cache_hits = total_exact_cache_hits;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
```

图读失败时，每个 warp 协作释放已获取的 graph record（`release_graph_record`，`candidate_scoring.cuh:109-121`：route bit 走 `anchor_graph_readers`，普通走 `graph_cache_readers`）。注意 `kGraphScratchBit` 的槽不需要 release（scratch 是 per-query 私有，无引用计数）。thread0 写 `-EIO` 完成事件，带上所有累计统计，return。**这是 strict 模式的硬失败**——见第 14 课 `report_direct_path_failure`，`-EIO` 触发 direct path failure 上报。

#### 阶段 3c：解码 compact graph 邻居（分 chunk）

```cpp
    // query_traversal.cuh:674-756
    for (u32 chunk_begin = 0; chunk_begin < selected_count;
         chunk_begin += kPersistentScoreChunk) {
      const u32 chunk_count = min(kPersistentScoreChunk,
                                  selected_count - chunk_begin);
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_cache_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        if (lane_in_warp == 0) {
          neighbor_counts[local] = record != nullptr && (record[1] & 1u) == 0
            ? min(static_cast<u32>(record[0]), params.graph_degree) : 0;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_offsets[0] = 0;
        for (u32 local = 0; local < chunk_count; ++local) {
          neighbor_offsets[local + 1] =
            neighbor_offsets[local] + neighbor_counts[local];
        }
        flattened_neighbors = neighbor_offsets[chunk_count];
        phase_started_cycles = clock64();
      }
      __syncthreads();
```

`kPersistentScoreChunk = 16`，每轮处理 16 个 selected（控制 shared memory 用量）。对每个 selected：
- `graph_record_pointer`（`rdma_cache.cuh:825-840`）根据 `acquired_slot` 的标志位返回指针：route bit → `anchor_graph_records`，scratch bit → `graph_scratch`，普通 → `graph_cache`。
- lane0 读 `record[0]`（edge_count，1 字节）和 `record[1]`（flags，含 `HOT_GRAPH_DELETED = 1`）。如果未删除，`neighbor_counts[local] = min(record[0], graph_degree)`。

thread0 做 prefix sum 得 `neighbor_offsets` 和总数 `flattened_neighbors`。同时记 `phase_started_cycles = clock64()`——**这标志着 graph 阶段结束、score 阶段开始**。

##### 解码 5 字节 RemotePtr 邻居

```cpp
      // query_traversal.cuh:700-722
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_cache_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        __syncwarp();
        const u32 count = neighbor_counts[local];
        for (u32 neighbor = lane_in_warp; neighbor < count; neighbor += warp_width) {
          const u64 raw = decode_compact_raw(record + 8 + neighbor * 5,
                                             params.graph_shard_bits);
          navigation_handles[neighbor_offsets[local] + neighbor] =
            handle_from_raw(params, raw);
        }
        __syncwarp();
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
        if (lane_in_warp == 0) graph_cache_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
```

每个 warp 处理一个 selected 的邻居。**逐字节解读 compact graph 记录布局**（与 `vamana_node.hh:198-219` 的 `encode_hot_graph_entry` 对应）：

- `record[0]`：edge_count（1 字节，≤ R）。
- `record[1]`：flags（1 字节，bit0 = `HOT_GRAPH_DELETED`）。
- `record[2..3]`：checksum16（小端 u16，`hot_graph.hh:119-123` 的 `load_u16_le`）。
- `record[4..7]`：generation（小端 u32，`store_u32_le(out + 4, generation)`）。
- `record[8 + i*5 .. 8 + i*5 + 4]`：第 i 个邻居的 5 字节 compact RemotePtr（`hot_graph.hh:94-96` 的 `neighbor_offset(i) = 8 + i*5`）。

所以 `decode_compact_raw(record + 8 + neighbor * 5, graph_shard_bits)` 解码出第 `neighbor` 个邻居的 raw RemotePtr，再 `handle_from_raw` 转成 32 位 handle。所有邻居 handle 写进 `navigation_handles[neighbor_offsets[local] + neighbor]`。

注意 `record[0] <= params.graph_degree` 的约束——`graph_degree` 是 host 配置的最大度数（`persistent_kernel.hh:101`），与 `VamanaNode::R` 一致。`record + 8 + neighbor * 5` 的最大偏移是 `8 + (graph_degree-1)*5`，对典型 R=64 是 323 字节，远小于 `kPersistentGraphCacheLineBytes = 512`（`persistent_kernel.hh:24`）——所以一条 cache line 装得下整条 compact graph 记录，RDMA 单次读 512B 即可。

解码完后 lane0 release graph record（`__threadfence` 保证读可见后再 release），并清空 `graph_cache_slots[selected]`。`__syncwarp` 保证 warp 内对 `record` 的读在 release 之前完成。

##### 去重 + 批量 PQ 评分

```cpp
      // query_traversal.cuh:723-744
      const u32 candidate_count = flattened_neighbors;
      for (u32 flat = threadIdx.x; flat < candidate_count; flat += blockDim.x) {
        const u32 handle = navigation_handles[flat];
        if (handle == UINT32_MAX ||
            !insert_visited(visited, params.visited_capacity, handle)) {
          navigation_handles[flat] = UINT32_MAX;
        }
      }
      __syncthreads();
      if (!approximate_handles_batch(params, descriptor, query_lut,
                                     navigation_handles,
                                     candidate_count,
                                     navigation_distances)) {
        if (threadIdx.x == 0) graph_failed = 1;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        score_phase_cycles += clock64() - phase_started_cycles;
        phase_started_cycles = clock64();
      }
      __syncthreads();
      if (graph_failed != 0) break;
```

去重：跨步遍历所有 flatten 后的邻居 handle，`insert_visited` 返回 false（已访问）的置 `UINT32_MAX`。这一步把"已展开过的候选"和"本批重复的候选"都过滤掉。

`approximate_handles_batch`（`rdma_cache.cuh:328-436`，第 19 课详解）批量评分：
- static handle（`kDeltaHandleBit == 0`）：直接 `approximate_handle`（base `pq_codes` / delta `delta_pq_codes` / resident `resident_pq_codes`）。
- dynamic handle：先查 `delta_slot_from_raw`，命中且 `delta_code_visible` 则用 `delta_pq_codes`；否则查 `resident_pq_slot_from_raw`，命中则用 `resident_pq_codes`；否则发远端读 `shards[shard].dynamic_code_offset` 处的 PQ code（`direct_fetch_batch` 批量）。
- 不可见（已删除或被 supersede）的返回 `FLT_MAX`。

返回 false 则 `graph_failed = 1`。thread0 累加 `score_phase_cycles`（**score 阶段计时点**），并重置 `phase_started_cycles` 标记 beam 阶段开始。`graph_failed` 则 break 出 chunk 循环。

##### 合并进 beam

```cpp
      // query_traversal.cuh:745-756
      merge_approximate_into_beam(
        navigation_handles, navigation_distances,
        candidate_count, beam_handles, beam_ids, beam_distances,
        beam_expanded, beam_count, traversal_capacity,
        merge_handles, merge_ids, merge_distances, merge_expanded,
        rerank_handles, rerank_ids, rerank_distances,
        candidate_workspace);
      if (threadIdx.x == 0) {
        beam_phase_cycles += clock64() - phase_started_cycles;
      }
      __syncthreads();
    }
```

`merge_approximate_into_beam`（`candidate_scoring.cuh:705-740`）把"现有 beam + 新评分的邻居"合并排序，取 top `traversal_capacity`。内部根据 `blockDim.x` 选择 wide（256 线程，单次 CUB radix sort）或 compact（128 线程，两 pass + final sort）路径——见第 18 课。`beam_count` 更新为有效候选数。

thread0 累加 `beam_phase_cycles`（**beam 阶段计时点**）。

#### 阶段 3d：chunk 循环后的失败处理

```cpp
    // query_traversal.cuh:757-788
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        const u32 slot = graph_cache_slots[selected];
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_rounds = total_graph_rounds;
        completion.cache_hits = total_cache_hits;
        completion.route_hits = total_route_hits;
        completion.exact_vectors = total_exact_reads;
        completion.exact_cache_hits = total_exact_cache_hits;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {
      expansions += selected_count;
    }
    __syncthreads();
  }
```

如果 chunk 循环里 `approximate_handles_batch` 失败导致 `graph_failed`，同样释放剩余 graph record、写 `-EIO` 完成、return。否则 `expansions += selected_count`，进下一轮主循环。

注意 `expansions` 加的是 `selected_count`（本轮展开的节点数），不是邻居数。所以 `max_expansions` 是"最多展开多少个节点"。

### 阶段 4：精确重排（exactify）

```cpp
  // query_traversal.cuh:790-823
  for (u32 index = threadIdx.x; index < beam_count; index += blockDim.x) {
    merge_handles[index] = beam_handles[index];
    merge_distances[index] = beam_distances[index];
    merge_expanded[index] = 0;
  }
  __syncthreads();
  sort_candidates(merge_handles, nullptr, merge_distances, merge_expanded, beam_count);
  if (threadIdx.x == 0) {
    rerank_count = 0;
    for (u32 index = 0;
         index < beam_count && rerank_count < params.final_rerank_width;
         ++index) {
      const u32 handle = merge_handles[index];
      if ((handle & kDeltaHandleBit) == 0 &&
          base_overridden(params, handle, descriptor.snapshot_epoch)) {
        continue;
      }
      rerank_handles[rerank_count] = handle;
      rerank_ids[rerank_count] = UINT32_MAX;
      rerank_distances[rerank_count] = merge_distances[index];
      ++rerank_count;
    }
    phase_started_cycles = clock64();
  }
  __syncthreads();
  exactify_into_beam(params, descriptor, query, rerank_handles, rerank_ids, rerank_distances,
                     rerank_count, beam_handles, beam_ids, beam_distances, beam_expanded,
                     beam_count, &total_exact_reads, &total_exact_cache_hits,
                     params.final_rerank_width, true, merge_handles, merge_ids,
                     merge_distances, merge_expanded);
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();
```

精确重排的预处理：
1. 把 beam 拷进 merge，sort by PQ distance。
2. thread0 取前 `final_rerank_width` 个候选进 `rerank_handles`，但**跳过被 base override 的 static handle**：`base_overridden`（`candidate_scoring.cuh:392-414`）检查两处——permanent override bitmap（`permanent_override_bits`，全量 bitmap）和 base override hash table（`base_override_keys/epochs`，per-ordinal epoch）。`base_overridden` 返回 true 表示该 ordinal 在 `snapshot_epoch` 之前被 override（即被 delta 取代或删除），跳过。**这就是 override epoch 屏蔽**。注意 dynamic handle（`kDeltaHandleBit`）不查 base override——它本身就是 override。
3. `rerank_ids` 置 `UINT32_MAX`（待 exactify 填充），`rerank_distances` 暂存 PQ 距离（会被精确距离覆盖）。

`exactify_into_beam`（`rdma_cache.cuh:438-656`，第 19 课详解）：
- 对每个 rerank candidate：
  - dynamic handle 且 delta 可见：用 `delta_vectors`（本地）做精确 L2，取 `delta_records[slot].id`。
  - 否则查 exact cache（`exact_cache`，多 way 组相联），命中且 `exact_record_visible`（header 无 lock/deleted）则用 cache 里的 id + 精确 L2。
  - 未命中且 admit 通过：占一个 cache 槽发远端读 `node_meta_offset` 处的 fixed record（含 header + id + generation + vector）。
  - 远端读回后 `exact_record_visible` 校验，取 id + 精确 L2，更新 cache 状态为 ready。
- 把"现有 beam（reset_beam=true 所以清空）+ 精确评分候选"合并 sort，取 top `final_rerank_width` 写回 beam。

`reset_beam = true` 意味着精排后 beam 完全替换为精确距离排序的结果——PQ 近似距离被丢弃。

thread0 累加 `exact_phase_cycles`（**exact 阶段计时点**）。

### 阶段 5：delta overlay 补救

```cpp
  // query_traversal.cuh:825-842
  __shared__ u32 delta_scan_records;
  __shared__ u32 delta_scan_scored;
  __shared__ u32 delta_scan_truncated_buckets;
  if (threadIdx.x == 0) delta_scan_started_cycles = clock64();
  __syncthreads();
  add_delta_candidates(params, descriptor, query, query_lut,
                       beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                       params.final_rerank_width,
                       anchor_best_indices, selected_anchor_count,
                       navigation_handles, delta_scan_records,
                       delta_scan_scored, delta_scan_truncated_buckets);
  if (threadIdx.x == 0) {
    completion.delta_scan_cycles = clock64() - delta_scan_started_cycles;
    completion.delta_scan_records = delta_scan_records;
    completion.delta_scan_scored = delta_scan_scored;
    completion.delta_scan_truncated_buckets = delta_scan_truncated_buckets;
  }
  __syncthreads();
```

精排后调用 `add_delta_candidates`（前面已详解）。这里 `beam_capacity = params.final_rerank_width`，所以 delta 候选会被 `beam_insert` 进精排 beam（已按精确距离排序）。注意 delta 候选用的是精确 L2 距离（`exact_storage_distance` on `delta_vectors`），与 beam 里其它候选的精确距离可比。

`anchor_best_indices` 和 `selected_anchor_count` 复用阶段 2a 的结果——delta scan 沿这些 anchor 的链表走。`navigation_handles` 当 `scan_slots` 用（2048 槽）。

thread0 写 `completion.delta_scan_cycles`（**delta_scan 阶段计时点**）外加三个计数。

### 阶段 5b：beam 为空再判

```cpp
  // query_traversal.cuh:843-859
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_rounds = total_graph_rounds;
      completion.cache_hits = total_cache_hits;
      completion.route_hits = total_route_hits;
      completion.exact_vectors = total_exact_reads;
      completion.exact_cache_hits = total_exact_cache_hits;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
```

如果精排 + delta 后 beam 仍空（所有候选都被 override / 不可见 / 读失败），返回 `-EIO`。

### 阶段 6：终排序 + top-k 写出 + 完成 push

```cpp
  // query_traversal.cuh:860-891
  sort_candidates(beam_handles, beam_ids, beam_distances, beam_expanded,
                  beam_count);
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < beam_count && beam_ids[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid])) ++valid;
    const u32 result_count = min(static_cast<u32>(descriptor.k), valid);
    u32* output_ids = reinterpret_cast<u32*>(descriptor.result_device_address);
    f32* output_distances = params.result_distances +
      static_cast<size_t>(query_slot) * descriptor.result_capacity;
    for (u32 index = 0; index < result_count; ++index) {
      output_ids[index] = beam_ids[index];
      output_distances[index] = beam_distances[index];
    }
    completion.result_count = result_count;
    completion.status = 0;
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.graph_cycles = graph_phase_cycles;
    completion.score_cycles = score_phase_cycles;
    completion.beam_cycles = beam_phase_cycles;
    completion.exact_cycles = exact_phase_cycles;
    completion.remote_pages = total_remote_reads;
    completion.remote_batches = total_remote_batches;
    completion.graph_read_retries = total_graph_read_retries;
    completion.graph_rounds = total_graph_rounds;
    completion.cache_hits = total_cache_hits;
    completion.route_hits = total_route_hits;
    completion.exact_vectors = total_exact_reads;
    completion.exact_cache_hits = total_exact_cache_hits;
    device_ring_push(params.completions, completion);
  }
}
```

最终 beam（含精排 + delta）按精确距离 sort。thread0：
1. 数有效候选（`beam_ids != UINT32_MAX` 且 `isfinite(distances)`）。
2. `result_count = min(k, valid)`。
3. 把 top-k 的 id 写进 `descriptor.result_device_address`（host 提供的输出缓冲，只写 id），距离写进 `params.result_distances[query_slot * result_capacity + index]`（per-slot 距离缓冲）。
4. 填满 `CompletionDescriptor` 全部字段：`result_count`、`status=0`、`gpu_cycles`（总时长）、五个阶段 cycle（`graph/score/beam/exact/delta_scan`）、所有统计计数。
5. `device_ring_push` 推完成事件。

`device_ring_push`（`device_ring.cuh:78`）是 SPSC ring 的阻塞推入，host 端 `completion_loop`（第 14 课）会 pop 并回写 PendingQuery。

---

## 关键数据结构与流程图

### beam / visited / merge scratch 数据结构

```
┌─ shared memory (per-block) ──────────────────────────────────────┐
│                                                                  │
│  shared_beam_handles[128]   ┐                                    │
│  shared_beam_ids[128]       ├─ traversal beam (PQ 近似 → 精确)   │
│  shared_beam_distances[128] │   traversal_capacity =             │
│  shared_beam_expanded[128]  ┘   min(128, traversal_beam_width)    │
│                                                                  │
│  rerank_handles[256]        ┐                                    │
│  rerank_ids[256]            ├─ 精排候选 (final_rerank_width)     │
│  rerank_distances[256]      ┘                                    │
│                                                                  │
│  CandidateWorkspace                                              │
│   ├─ arrays.handles[512]     ┐                                   │
│   ├─ arrays.ids[512]         ├─ merge scratch (sort 输入)        │
│   ├─ arrays.distances[512]   │                                   │
│   └─ arrays.expanded[512]    ┘                                   │
│   └─ sort (CUB radix temp)                                      │
│                                                                  │
│  selected_handles[32]       ┐                                    │
│  neighbor_counts[32]        ├─ 主循环每轮 scratch                │
│  neighbor_offsets[33]       │                                    │
│  graph_cache_slots[32]      │                                    │
│  remote_reads_by_lane[32]   │                                    │
│  cache_hits_by_lane[32]     │                                    │
│  route_hits_by_lane[32]     ┘                                    │
│                                                                  │
│  anchor_best_indices[256]   ── 选中 anchor 索引 (delta scan 复用)│
│                                                                  │
│  cycle 计时器 × 7                                                │
└──────────────────────────────────────────────────────────────────┘

┌─ global scratch (per-query-slot) ────────────────────────────────┐
│  navigation_candidate_handles[2048]  ┐ 邻居 flatten + delta scan  │
│  navigation_candidate_distances[2048]┘ scan_slots (时分复用)       │
│  visited_hash[visited_capacity]      ── 开放寻址去重              │
│  decoded_queries / transformed_queries / query_luts ── per-slot   │
│  dynamic_code_request_{shards,offsets,local_iovas}[2048] ─ RDMA  │
│  exact_records[exact_width × node_record_bytes] ── 精排远端读落点│
│  result_distances[result_capacity] ── top-k 距离输出              │
└──────────────────────────────────────────────────────────────────┘
```

### 完整状态机

```
                  ┌──────────────────────────┐
                  │  process_query 入口       │
                  │  query_started_cycles     │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  入口校验                  │
                  │  EINVAL → push completion │
                  │             return        │
                  └────────────┬─────────────┘
                               │ OK
                  ┌────────────▼─────────────┐
                  │  prepare_started_cycles   │
   prepare        │  query → float (dtype)    │
   阶段           │  OPQ 旋转                  │
                  │  PQ LUT 构建               │
                  ├───────────────────────────┤
                  │  anchor 打分(可选)         │
                  │   ├─ PQ 近似 sort         │
                  │   ├─ exact_anchor 重排    │
                  │   └─ approximate_handle   │
                  │  或 entry_points 打分     │
                  │  dynamic route 打分       │
                  │   └─ seqlock 双校验       │
                  │  static+dynamic 合并去重  │
                  │  → beam 初始化            │
                  │  beam_count==0 → EIO return│
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  completion.prepare_cycles│
                  │  = clock - prepare_started│
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
       ┌────────→│  while expansions < max   │
       │         │   phase_started = clock    │
       │         └────────────┬─────────────┘
       │                      │
       │         ┌────────────▼─────────────┐
       │         │  选未展开候选 (≤prefetch) │
       │         │  selected_count==0 break  │
       │         └────────────┬─────────────┘
       │                      │
       │         ┌────────────▼─────────────┐
       │ graph   │  fetch_graph_records_batch│
       │ 阶段    │  (route/cache/scratch/    │
       │         │   RDMA + checksum 校验)   │
       │         │  graph_failed → release + │
       │         │   EIO push + return       │
       │         └────────────┬─────────────┘
       │                      │
       │         ┌────────────▼─────────────┐
       │         │  graph_phase_cycles +=    │
       │         │   clock - phase_started   │
       │         └────────────┬─────────────┘
       │                      │
       │         ┌────────────▼─────────────┐
       │         │  for chunk in selected   │
       │         │   (kPersistentScoreChunk) │
       │         ├───────────────────────────┤
       │ score   │  decode compact graph     │
       │ +beam   │   record[0]=count         │
       │ 阶段    │   record[1]=flags         │
       │         │   record[2..3]=checksum   │
       │         │   record[4..7]=generation │
       │         │   record[8+i*5]=neighbor  │
       │         │    RemotePtr (compact)    │
       │         │  insert_visited 去重      │
       │         │  approximate_handles_batch│
       │         │   (base/delta/resident/   │
       │         │    RDMA code)             │
       │         │  merge_approximate_into_  │
       │         │   beam (CUB radix sort)   │
       │         └────────────┬─────────────┘
       │                      │
       │         ┌────────────▼─────────────┐
       │         │  score_phase_cycles +=    │
       │         │  beam_phase_cycles +=     │
       │         └────────────┬─────────────┘
       │                      │
       └──────────────────────┘ (expansions += selected_count)

                  ┌────────────▼─────────────┐
                  │  精排预处理                │
   exact         │  beam → merge sort         │
   阶段          │  skip base_overridden      │
                  │  → rerank_handles[]       │
                  │  exactify_into_beam       │
                  │   (delta local / cache /  │
                  │    RDMA fixed record +    │
                  │    exact L2)              │
                  │  reset_beam=true          │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  exact_phase_cycles +=    │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
   delta_scan    │  add_delta_candidates      │
   阶段          │   (anchor 链表 / 高水位)   │
                  │   delta_visible 过滤      │
                  │   exact_storage_distance  │
                  │   beam_insert             │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  delta_scan_cycles +=     │
                  │  delta_scan_records/scored│
                  │  /truncated_buckets       │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  beam_count==0 → EIO      │
                  │  sort beam (精确距离)     │
                  │  top-k 写 result_device   │
                  │  填满 CompletionDescriptor│
                  │  device_ring_push         │
                  └──────────────────────────┘
```

### 每阶段 cycle 计时图

```
query_started_cycles ────────────────────────────────────────────────► clock64
│                                                                     │
│ ├─ prepare (query→float, OPQ, LUT, anchor/route 打分, beam init)   │
│ │   └── completion.prepare_cycles = t1 - t0                         │
│ │                                                                     │
│ ├─ while loop ──────────────────────────────────────────┐            │
│ │   │ ├─ graph_phase (fetch_graph_records_batch)        │            │
│ │   │ ├─ score_phase (decode + approximate_handles_batch)│           │
│ │   │ └─ beam_phase (merge_approximate_into_beam)       │            │
│ │   ↳ (每轮累加，共 total_graph_rounds 轮)                │            │
│ │   └────────────────────────────────────────────────────┘            │
│ │                                                                     │
│ ├─ exact_phase (exactify_into_beam)                                 │
│ │   └── completion.exact_cycles                                      │
│ │                                                                     │
│ ├─ delta_scan_phase (add_delta_candidates)                          │
│ │   └── completion.delta_scan_cycles                                 │
│ │                                                                     │
│ └─ 终排序 + top-k 写出 + push                                        │
│                                                                     │
└── completion.gpu_cycles = t_final - query_started_cycles            │
   completion.graph_cycles = Σ graph_phase_cycles                     │
   completion.score_cycles = Σ score_phase_cycles                     │
   completion.beam_cycles  = Σ beam_phase_cycles                      │
```

这些 cycle 计数在 host 端 `completion.cc:69-72` 用 `cycles * 1000000ULL / gpu_clock_khz` 换算成 ns，喂给慢查询日志和 telemetry（见第 14 课、第 9 课）。

---

## delta overlay 在遍历中的参与

delta overlay 在 `process_query` 中有**三处**参与：

1. **入口打分阶段**（`approximate_handle`，第 455、470 行）：当 handle 是 dynamic（`kDeltaHandleBit` set），`approximate_handle`（`candidate_scoring.cuh:443-477`）会先查 `delta_slot_from_raw`——如果 delta 表里有该 remote_node 且 `delta_code_visible`，用 `delta_pq_codes` 评分；如果 delta 不可见（已删除或被 supersede），返回 `FLT_MAX` 屏蔽；否则查 resident PQ 或发远端读。**这是 delta overlay 在图遍历中的核心参与点**：图邻居指向的节点可能已经被 delta 取代，PQ 评分必须用 delta 的最新 PQ code。

2. **精排阶段**（`exactify_into_beam`，第 815 行）：dynamic handle 且 delta 可见时，用 `delta_vectors`（本地显存）做精确 L2，跳过远端读（`rdma_cache.cuh:474-484`）。delta 不可见时退回 exact cache / RDMA。**delta overlay 让最新插入的节点无需远端读即可精确评分**。

3. **delta scan 补救**（`add_delta_candidates`，第 830 行）：沿 anchor 链表或高水位窗口扫 delta 表，把"图还没追上的最新 delta"直接以精确距离注入精排 beam。`delta_visible` 过滤掉不可见（deleted/durable/superseded/未来 epoch）的记录。`delta_records[slot].id` 提供逻辑 id，`delta_vectors` 提供原向量。

### override epoch 屏蔽与 tombstone 过滤

- **override epoch 屏蔽**：`base_overridden(params, handle, snapshot_epoch)`（`candidate_scoring.cuh:392-414`）检查两处——permanent override bitmap（全量 ordinal bitmap，bit set 表示该 ordinal 永久被 delta 取代）+ base override hash table（per-ordinal epoch，`epoch <= snapshot_epoch` 表示在快照前已 override）。精排预处理（第 803-806 行）跳过被 override 的 static handle，避免读到旧版本向量。
- **tombstone 过滤**：`delta_visible`（`candidate_scoring.cuh:416-421`）排除 `kDeltaDeleted` flag；`delta_code_visible`（423-429）只排除 `kDeltaDeleted`（不排除 durable，因为 durable 表示已落盘 base，PQ code 仍可用于近似评分）。`exact_record_visible`（`rdma_cache.cuh:294-297`）检查 fixed record header 的 `kNodeDeletedMask`（`candidate_scoring.cuh:58`）和 `kNodeLockMask`——deleted 的 fixed record 在精排时被跳过。

### snapshot_epoch 绑定可见性

`descriptor.snapshot_epoch` 是查询提交时 host 绑定的快照 epoch（见第 14 课查询提交）。它在遍历中三处生效：
- `score_dynamic_route_slot`：`candidate.epoch > snapshot_epoch` 的 dynamic route 被拒（第 76 行）。
- `approximate_handle` / `approximate_handles_batch` / `exactify_into_beam`：`delta_visible` / `delta_code_visible` 要求 `record.epoch <= snapshot_epoch` 且 `superseded_epoch > snapshot_epoch`（或 0）。
- `base_overridden`：`base_override_epochs[position] <= snapshot_epoch` 才算 override 生效。

这保证了整条查询看到的是一个一致的快照视图——晚于 `snapshot_epoch` 的发布对本查询不可见，早于的 supersede 对本查询生效。

---

## 与其它模块的关系

- **第 18 课（候选评分）**：`approximate_entry` / `approximate_handle` / `exact_storage_distance` / `beam_insert` / `sort_candidates` / `merge_approximate_into_beam` / `insert_visited` / `handle_from_raw` / `resolve_handle` / `base_overridden` / `delta_visible` / `delta_code_visible` 全部来自 `candidate_scoring.cuh`。本课是这些工具的"主消费者"。
- **第 19 课（RDMA cache）**：`fetch_graph_records_batch` / `prepare_graph_record` / `graph_record_pointer` / `release_graph_record` / `valid_graph_record` / `approximate_handles_batch` / `exactify_into_beam` / `direct_fetch_batch` / `wait_direct_batch` 来自 `rdma_cache.cuh`。graph cache（多 way 组相联 + generation + TTL + admission）、exact cache、anchor graph route 都在那里详解。
- **第 17 课（kernel 启动器/上下文）**：`PersistentKernelParams` 的所有字段由 host 端 `PersistentSearchEngine::Impl` 装配，`launch_persistent_search` 启动 query block。`query_slot` 的分配、`visited_hash` 的 per-slot 划分都在那里。
- **第 21 课（kernel 运行时/角色调度）**：query block 与 owner block（`launch_direct_read_owners`）、dispatcher、control CTA 的协作。`direct_batch_queues` / `direct_batch_statuses` / `direct_owner_phases` 是 owner block 的接口，本课里 `direct_fetch_batch` 会通过这些队列委托 owner block 发 GPUNetIO 读（见 `rdma_cache.cuh:137-195`）。
- **第 14 课（查询执行/路由/完成）**：`device_ring_push(params.completions, completion)` 推完成事件，host `completion_loop`（`completion.cc:39`）pop 后换算 cycle→ns，更新 telemetry，唤醒 PendingQuery。`-EIO` / `-EBADMSG` 触发 `report_direct_path_failure`。
- **第 15 课（增量发布）**：delta 表、base override、resident PQ、dynamic route slots 由 control CTA 通过 `DeltaPublishDescriptor` 发布。本课读这些结构时用 seqlock / atomic / `load_cg` 保证不阻塞 writer。
- **第 16 课（存储回收 RCU）**：`release_graph_record` 递减 `graph_cache_readers` / `anchor_graph_readers`，让 control CTA 可以安全回收被淘汰的 cache 槽。`exact_cache_readers` 同理。
- **第 22 课（GPUNetIO 传输/probe）**：`direct_fetch_batch` 的 verbs 路径（`doca_gpu_dev_verbs_*`）、`poll_direct_cq`、`lock_direct_qp` 在 `candidate_scoring.cuh:143-206` 与 `rdma_cache.cuh:37-262`，本课只调 `direct_fetch_batch` / `wait_direct_batch`。

---

## 小结

`query_traversal.cuh` 的 `process_query` 是 dvstor 单查询的完整状态机，它把"解码 → 入口 → 图遍历 → 精排 → delta 补救 → top-k"压进一个 CTA，靠 shared memory beam + global visited hash + CUB radix sort 实现高效近邻搜索，靠 seqlock + atomic + `load_cg` 与并发的 delta 发布/RCU 回收协调，靠 `clock64()` 七个计时点把每阶段开销精确反馈给 host telemetry。关键设计：

1. **入口自适应**：static anchor/entry + dynamic route 在同一 PQ 距离空间排序去重，dynamic 是替换不是叠加，保证 beam 初始宽度不膨胀。
2. **图遍历批量 pipelining**：每轮选 `prefetch_depth` 个未展开候选，批量发 RDMA 图读（route/cache/scratch 三级降级），解码 5 字节 compact RemotePtr 邻居，批量 PQ 评分（base/delta/resident/RDMA 四级降级），CUB radix sort 合并进 beam。
3. **compact graph 记录 512B cache line 对齐**：header(8) + neighbor(R×5) ≤ 328B，单次 RDMA 读一条 cache line 即可拿到全部邻居。
4. **乐观快照校验**：graph record 的 checksum16 是乐观验证，最多重试 3 次应对 stage2 原地更新导致的 torn read。
5. **精排 + delta 双路**：精排走 exact cache / RDMA fixed record，delta scan 走本地 `delta_vectors`，两者都产出精确 L2 距离，在最终 beam 里公平竞争。
6. **snapshot_epoch 一致性**：dynamic route epoch、delta epoch/superseded_epoch、base override epoch 都绑定 `descriptor.snapshot_epoch`，保证整条查询看到一致快照。
7. **七点 cycle 计时**：prepare / graph / score / beam / exact / delta_scan / total，写回 `CompletionDescriptor`，供 host 慢查询分析与 telemetry 累加。

下一课（第 21 课）将讲 query/owner/dispatcher/control 四种 CTA 角色的调度与协作，把本课的 `process_query` 放进整个持久化 kernel 的运行时上下文。
