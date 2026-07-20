# 第 25 课 存储侧索引访问与图修改

## 本课目标与涉及文件

前几课我们看完了 GPU 持久化 kernel 的查询遍历（第 20 课）与计算侧的 storage owner 远端缓存（第 19 课、第 28 课），但一直把“图是怎么写出来的”当作黑盒。本课终于打开这个黑盒：**当一条 insert/upsert/delete 到达存储节点（storage owner）后，存储节点是如何分配物理空间、如何在分片内做构造搜索、如何剪枝、如何把反向边写回邻居、又是如何推进 finalized watermark 的。**

这是 dvstor“GPU 中心化存算分离”的另一条腿：查询路径走 GPU（第 17–22 课），而**写入路径走 CPU 协程 + 跨分片 RDMA**，原因是写入要做 RobustPrune 剪枝和反向边 ACK，逻辑复杂、批量小、对延迟不敏感，CPU 上更顺手。GPU 只读 hot-graph 邻居表（schema-15 紧凑图），所以存储侧必须把图维护成“GPU 可见、checksum 自校验、并发安全”的形式。

涉及文件（全部在 `src/memory_node/storage_owner_index/` 下）：

| 文件 | 行数 | 作用 |
|------|------|------|
| `allocation.cc` | 248 | 为新 insert/upsert 分配动态节点空间、idmap 加载、freshness 发布、medoid 读写 |
| `graph_access.cc` | 504 | 紧凑图记录的读写原语：节点快照、邻居列表、节点锁、global medoid |
| `candidate_search.cc` | 400 | owner 分片内 beam search（同步 + async 协程两版）、partition-local search |
| `graph_mutation.cc` | 567 | insert/upsert/delete 主流程、RobustPrune CPU、反向边本地应用 |
| `reverse_batch.cc` | 279 | 反向边批量应用（stage2 维护期使用），coalesce + snapshot cache |
| `robust_prune_policy.hh` | 52 | alpha RobustPrune 通用选择策略（模板） |
| `two_stage_insert_oracle.hh` | 241 | 两阶段插入算法 oracle（direct vs two-stage 等价性参考） |
| `partition_local_search.hh` | 171 | 分片内 beam search 原语（算法层，无 I/O） |
| `reverse_batch_policy.hh` | 37 | 反向边候选筛选策略（去重 + 在锁内做 liveness） |
| `detail.hh` | 159 | 共用别名、awaitable 定义、snapshot 解析 |

阅读本课前建议先看：第 6 课（Vamana 图格式与 anchor/idmap）、第 7 课（schema-15 索引格式，特别是 `VamanaNode` 与 `hot_graph_entry`）、第 8 课（存储协议与 owner map）、第 23 课（storage_owner_state 与 `StorageOwnerThread`/`StorageOwnerCoroutineScratch`）、第 24 课（peer RPC 与 `send_reverse_update_batch`）。本课内容会直接被第 26 课（maintenance stage2 worker）调用。

---

## 逐文件逐函数讲解

### 1. `detail.hh`：共用别名与 awaitable

这个头文件被本目录所有 `.cc` 包含，是“粘合层”。它先给一堆长类型名起短别名（`detail.hh:15-26`）：

```cpp
using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerCoroutineScratch = memory_node_detail::StorageOwnerCoroutineScratch;
using StorageOwnerPruneCandidateInfo = memory_node_detail::StorageOwnerPruneCandidateInfo;
using StorageOwnerScoredSnapshot = memory_node_detail::StorageOwnerScoredSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
```

这些 `memory_node_detail::*` 都定义在 `src/memory_node/storage_owner_state.hh`（第 23 课），`NodeSnapshot` 就是 `{rptr, header, id, generation, deleted, vector_data}`，是存储节点之间传递节点状态的统一载荷。

接着三个内联小工具（`detail.hh:28-59`）：

- `snapshot_buffer_bytes()` = `VamanaNode::size_until_vector_end()`，即一个节点从 header 到 vector 末尾的字节数，是 peer-read 的最小单元。
- `aligned_snapshot_bytes()` 把它对齐到 cache line，方便在协程 scratch 里按 stride 切片，避免相邻快照的 false sharing。
- `storage_owner_construction_width(config)` 取 `config.resolved_storage_owner_construction_width()`，即 beam 宽度 L。
- `storage_owner_snapshot_batch_size(config, thread)` 综合配置的 `storage_owner_search_snapshot_batch` 和 thread 的 scratch 容量，取较小值。**这是把“算子批大小”与“协程 scratch 物理容量”绑定的关键点**——批大小不能超过 scratch 能放下的快照数，否则会越界。
- `local_stitch_enabled(config)`：`config.storage_owner_update_mode == "local_stitch"` 时为 true，表示走“本地缝合”模式而非跨分片两阶段。

`parse_remote_snapshot`（`detail.hh:61-73`）把 peer-read 回来的裸字节解析成 `NodeSnapshot`：header（u64）、id（u32）、generation（u32）、deleted 标志（从 header 位提取）、vector_data（按 `VamanaNode::vector_bytes()` 拷贝）。注意它**只解析到 vector 末尾，不解析 hot-graph 邻居表**——邻居表有独立的 checksum 和解码路径（见 `graph_access.cc` 的 `read_neighbor_list`）。

最后四个 awaitable 类型（`detail.hh:77-159`）是手写的 `co_await` 适配器，它们都不挂起调度器（`await_suspend` 为空），而是把“是否就绪”交给调用方在协程循环里 poll。这种设计在第 23 课的协程调度器里有详细说明。关键点：

- `GlobalMedoidReadAwaitable`：本地分片直接读 `index_buffer_ + 8`；远端分片靠 `buffer` 里 peer-read 回来的 8 字节。
- `NodeSnapshotReadAwaitable` / `NodeSnapshotsReadAwaitable`：ready=true 时已经同步算完；ready=false 时 buffer 里有 peer-read 的原始字节，`await_resume` 调 `parse_remote_snapshot`。
- `NeighborListReadAwaitable`：ready=false 时在 `await_resume` 里 decode hot-graph entry，**如果 decode 失败会 fallback 到同步 `read_neighbor_list`**——这是对“peer-read 期间邻居表正在被并发写”的容忍。

---

### 2. `allocation.cc`：物理空间分配与 freshness

#### `allocate_local_node()`（`allocation.cc:5-30`）

这是存储节点给新 insert/upsert 节点分配物理地址的入口。核心逻辑：

```cpp
size_t node_size = VamanaNode::allocation_size();
while (node_size % 8 != 0) {
  node_size += 4;
}
```

先把节点大小对齐到 8 字节。`VamanaNode::allocation_size()` 是 header + id + generation + vector + PQ code + hot_graph_entry 的总和（见第 7 课），已经对齐到 16，这里再保 8 字节对齐是为了让 `std::atomic_ref<u64>` 的 fetch_add 不会跨缓存行。

然后是一段**非常重要**的注释（`allocation.cc:11-15`）：

> Schema-15 reverse-update operations carry physical pointers but no target generation. Reusing a tombstoned address while an old cross-shard request can still retry would let that stale operation mutate an unrelated node. Keep vector/PQ/node storage generation-stable in this protocol version; only bounded stage2/GPU-delta metadata is reclaimed.

这段话解释了为什么本课的存储侧**不复用 tombstone 地址**：schema-15 的反向边 RPC 只携带物理 `RemotePtr`，不带目标 generation（见第 24 课 peer RPC）。如果一个节点被删了、地址被回收复用给新节点，那么一个还在路上的旧反向边 RPC 就会写到新节点上，造成数据错乱。所以这一版协议里，**vector/PQ/节点存储是 generation-stable 的，永不复用**；只有 stage2 metadata 和 GPU delta 这种有界的、不会被跨分片 RPC 引用的元数据才回收。

分配本身是一个原子 bump allocator（`allocation.cc:17-20`）：

```cpp
auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
std::atomic_ref<u64> alloc_ref(*free_ptr);
const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
```

`index_buffer_` 的第 0 个 u64 是分配指针（free pointer），每次 `fetch_add(node_size)` 推进。这是无锁的，多协程并发安全。`lib_assert` 是 dvstor 的断言宏，OOM 直接 abort（生产环境宁可崩也不要静默写坏）。

接着更新 `dynamic_high_watermark`（`allocation.cc:21-28`）：

```cpp
auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
  index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
std::atomic_ref<u64> high_watermark(control->dynamic_high_watermark);
u64 observed = high_watermark.load(std::memory_order_relaxed);
while (observed < offset + node_size &&
       !high_watermark.compare_exchange_weak(
         observed, offset + node_size,
         std::memory_order_release, std::memory_order_relaxed)) {}
```

`StorageControlBlock` 是 schema-15 在每个存储分片开头的控制块（第 7 课、第 9 课），`dynamic_high_watermark` 是 GPU 查询遍历“动态节点区”时能读到的最高偏移。**这个 watermark 必须单调递增**，所以用 CAS 循环只在 `offset + node_size > observed` 时推进。GPU 读邻居时如果看到 `rptr.byte_offset() > high_watermark`，就认为该指针无效（见 `valid_local_storage_node_pointer`）。最后返回 `RemotePtr{storage_id_, offset}`。

#### `retire_local_dynamic_node()`（`allocation.cc:32-47`）

```cpp
if (pointer.is_null() || pointer.memory_node() != storage_id_ ||
    pointer.byte_offset() < gpu_dynamic_node_base_ || maintenance_sequence == 0) {
  return;
}
```

这个函数本意是回收动态节点空间，但当前实现**什么都没回退**——它只是把 `storage_owner_reclaim_candidates_` 和 `control->reclaim_pending_nodes` 清零（`allocation.cc:42-46`）。这跟上面的注释一致：**这一版协议不真正回收节点物理空间**，只是把“待回收计数”清零，避免 RCU（第 16 课）的回收队列无限增长。`maintenance_sequence` 参数被 `(void)` 掉了，是为未来版本预留的。

注意对齐断言 `allocation.cc:39-40`：要回收的指针必须落在 `gpu_dynamic_node_base_` 之上且按 `node_size` 对齐——这是防止误传一个静态区指针进来。

#### `minimum_compute_reclaim_ack()`（`allocation.cc:49-62`）

```cpp
const u32 client_count = control->compute_client_count;
if (client_count == 0 || client_count > gpu_search::format::kMaxComputeClients) {
  return 0;
}
u64 minimum = std::numeric_limits<u64>::max();
for (u32 client = 0; client < client_count; ++client) {
  std::atomic_ref<u64> ack(control->reclaim_ack_sequences[client]);
  minimum = std::min(minimum, ack.load(std::memory_order_acquire));
}
return minimum;
```

这是 RCU（第 16 课）在存储侧的入口：存储节点要回收某段空间前，必须等所有 compute client 都 ACK 了“我已经离开这段临界区”。这里取所有 client 的 `reclaim_ack_sequences` 的**最小值**——任何一个 client 没跟上，min 就推进不了。返回 0 表示没有 compute client（纯存储模式或刚启动），可以激进回收。

#### `load_owner_idmap()`（`allocation.cc:64-112`）

加载本 owner 分片的 idmap sidecar。先清空 `base_idmap_` 和所有 `dynamic_freshness_shards_`（`allocation.cc:65-69`）。`base_idmap_` 是**不可变基础层**——离线构建时写好的 id→RemotePtr 映射；`dynamic_freshness_shards_` 是**动态层**——在线 insert/upsert/delete 的增量，分多个 shard 降低锁竞争（`kDynamicFreshnessShardCount`）。

`index_path::owner_idmap_file(index_prefix, storage_id_ + 1, num_storage_nodes_)` 构造路径，注意是 `storage_id_ + 1`（1-based）。打开后读 `vamana::idmap::Header`（第 6 课），校验 magic/version/owner_shard/shard_count。`allocation.cc:88` 计算动态 headroom：

```cpp
const u64 dynamic_headroom = std::max<u64>(1024ull * 1024ull, header.entry_count / 20);
```

即“至少 100 万条动态容量，否则按基础条目数的 5% 预留”。这个 headroom 平摊到 `kDynamicFreshnessShardCount` 个 shard 上做 `reserve`，避免动态插入时反复 rehash。

`allocation.cc:97-105` 逐条读 `vamana::idmap::Entry`，构造 `FreshnessEntry{RemotePtr, generation, deleted}` 插入 `base_idmap_`。注意 `deleted` 是从 `flags & vamana::idmap::kDeleted` 提取的——离线 idmap 里也可能有 tombstone。

#### `publish_mutation()` / `prepare_mutation()`（`allocation.cc:114-162`）

这两个是 freshness map 的核心写接口。

`publish_mutation(id, ptr, generation, deleted)`（`allocation.cc:114-119`）：

```cpp
DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
std::lock_guard<std::mutex> lock(shard.mutex);
shard.entries[id] = FreshnessEntry{ptr, generation, deleted};
shard.mutations_inflight.erase(id);
```

按 `id` 选 shard（hash 分片），加锁写入 `entries[id]`，并从 `mutations_inflight` 中移除——表示这条 mutation 已经“落盘”到 freshness map，可以被 compute 侧 owner 更新（第 28 课）读到。

`prepare_mutation(id, kind, old_entry, new_generation)`（`allocation.cc:121-162`）是 mutation 的“预检 + 占位”：

```cpp
if (shard.mutations_inflight.contains(id)) {
  return service::storage_owner::MutationStatus::failed;
}
```

**同一 id 同时只允许一个 mutation 在飞**，防止两个并发 insert 互相覆盖。这是基于 `mutations_inflight` set 做的乐观锁。

`allocation.cc:131-141` 查找当前 entry：先查 dynamic shard，没有再查 immutable base。`exists` = dynamic 或 base 任一命中；`current` 取命中的那个，都没命中就构造空 `FreshnessEntry{}`。`live = exists && !current.deleted`。

`allocation.cc:143-152` 把旧 entry 和新 generation 写回给调用方：

```cpp
if (old_entry != nullptr) {
  *old_entry = current;
  if (old_entry->deleted) {
    old_entry->current = RemotePtr{};
  }
}
const u32 previous_generation = exists ? current.generation : 0;
if (new_generation != nullptr) {
  *new_generation = previous_generation + 1;
}
```

**注意这个细节**：如果旧 entry 是 deleted 状态，`old_entry->current` 被清成 null RemotePtr。这是因为 deleted entry 的 rptr 已经是 tombstone，调用方不应该再去访问它的物理空间——upsert 时 `old_entry.current.is_null()` 表示“无需 mark_node_deleted 旧节点”。

`new_generation = previous_generation + 1`：generation 单调递增，新插入的节点用这个 generation 标记，compute 侧 owner 更新时通过 generation 判断“这条更新我是否已经见过”（第 28 课）。

`allocation.cc:153-159` 做语义校验：insert 且已 live → `already_exists`；erase 但不存在 → `not_found`；erase 但已 deleted → `already_deleted`。最后 `shard.mutations_inflight.insert(id)` 占位，返回 `ok`。**调用方必须在后续要么 `publish_mutation` 要么显式 erase 掉 inflight 占位**，否则该 id 会被永久卡住——`graph_mutation.cc` 的失败路径都显式调用了 `complete_storage_owner_maintenance_sequence`，间接清理。

#### `mark_node_deleted()`（`allocation.cc:164-204`）

把一个节点标记为 deleted。这分两步：**header 位 + hot_graph_entry 位**。为什么要两处？因为 GPU 查询遍历只看 hot_graph_entry（紧凑图），而 CPU 维护代码看 header（完整记录）。两处都要置 tombstone，否则 GPU 可能还会把一个被删节点当成 live 邻居展开。

`allocation.cc:166-182` 处理 header：

```cpp
const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
const bool local = local_shard(rptr.memory_node());
if (local) {
  auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
  std::atomic_ref<u64> ref(*header_ptr);
  ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED), std::memory_order_acq_rel);
  lock_node(rptr);
  locked = true;
} else {
  lock_node(rptr);
  locked = true;
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
  header |= static_cast<u64>(VamanaNode::HEADER_DELETED);
  remote_write_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
}
```

注意本地和远端的锁顺序不同：**本地先置位再上锁**（因为 `fetch_or` 本身是原子的，置位不需要锁保护，但接下来要改 hot_graph_entry 需要锁）；**远端先上锁再 read-modify-write**（因为远端 fetch_or 没有 RDMA 原语支持，只能 read+write，这两步必须锁保护防止丢更新）。这是分布式系统里“本地有原子原语、远端只有 RPC”的典型不对称处理。

`HEADER_DELETED` 的位值在 `src/vamana/vamana_node.hh:21` 是 `0b1000000000000000000000000`（bit 25），刻意选高位避开 lock 位（bit 0-1）和 medoid 位（bit 16）。

`allocation.cc:183-199` 处理 hot_graph_entry：

```cpp
const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
if (local_shard(rptr.memory_node())) {
  byte_t* entry = index_buffer_.get_full_buffer() + hot_offset;
  entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
  vamana::hot_graph::store_u32_le(entry + 4, generation);
  const u16 checksum =
    vamana::hot_graph::checksum16(entry, VamanaNode::hot_graph_entry_size());
  vamana::hot_graph::store_u16_le(entry + 2, checksum);
}
```

`entry[1]` 是标志字节（`HOT_GRAPH_DELETED = 1<<0`），`entry[4..8]` 是 generation（little-endian u32），`entry[2..4]` 是 checksum16。**每次改 hot_graph_entry 都要重算 checksum**——GPU 和 CPU 读邻居表时都先验 checksum，失败就重试或视为 tombstone。这就是 schema-15 紧凑图的“自校验”机制（第 7 课）。远端分支用 read-modify-write 整个 entry，逻辑相同。

#### `read_global_medoid()` / `async_read_global_medoid()` / `write_global_medoid()` / `try_set_global_medoid()`（`allocation.cc:206-248`）

这四个函数管理 **global medoid**——整个索引（所有分片共享）的入口点。它存在 `index_buffer_` 的 offset 8（前 8 字节是 free pointer，见 `allocate_local_node`），且**只在 storage_id_==0 的主节点上是权威值**，其他分片要 read/write 都得跨分片 RDMA 访问节点 0。

`allocation.cc:206-214`：本地（storage_id_==0）直接读 `*(u64*)(buffer+8)`；远端 `remote_read_bytes(0, 8, &raw, sizeof(raw), 0)`。

`allocation.cc:216-224` 的 async 版本构造 `GlobalMedoidReadAwaitable`，本地 ready=true 立即返回；远端 post 一个 async peer-read，awaitable 携带 buffer 指针，resume 时解析。

`write_global_medoid`（`allocation.cc:226-232`）逻辑对称。

`try_set_global_medoid`（`allocation.cc:234-248`）是 CAS：

```cpp
if (storage_id_ == 0) {
  auto* slot = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8);
  std::atomic_ref<u64> ref(*slot);
  u64 current = expected.raw_address;
  const bool ok =
    ref.compare_exchange_strong(current, desired.raw_address,
      std::memory_order_acq_rel, std::memory_order_acquire);
  observed = RemotePtr{current};
  return ok;
}
const u64 original = remote_compare_and_swap(0, 8, expected.raw_address, desired.raw_address, 0);
observed = RemotePtr{original};
return original == expected.raw_address;
```

本地用 `std::atomic_ref` 的 CAS，远端用 `remote_compare_and_swap`（第 4-5 课 RDMA 库提供的 RDMA CAS）。`observed` 传出当前值，调用方据此判断“我是否抢到了 medoid 位”。这个 CAS 在 `execute_storage_owner_insert_job_async` 里用来“第一个插入的节点成为 medoid”。

---

### 3. `graph_access.cc`：紧凑图记录的存储侧访问

#### `read_node_snapshot()`（`graph_access.cc:5-50`）

读一个节点的完整快照（header + id + generation + vector，不含邻居表）。这是构造搜索和 RobustPrune 的核心读原语。

`graph_access.cc:9-13` 是边界断言：

```cpp
const auto vector_addr = vamana::StorageLayoutResolver::vector(rptr);
lib_assert(vector_addr.offset + vector_addr.size <= mn_memory_bytes_,
           "node snapshot read exceeds shard bounds: ...");
```

`StorageLayoutResolver` 把 `RemotePtr` 解析成各种子区域的 `(offset, size)`（第 6 课），这里用 `vector()` 得到 vector 区域，断言不越界。

本地分支（`graph_access.cc:19-29`）：

```cpp
const byte_t* base = index_buffer_.get_full_buffer();
const byte_t* ptr = base + rptr.byte_offset();
snapshot.header = load_local_node_header_acquire(rptr);
snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
snapshot.generation = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation());
snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
std::memcpy(snapshot.vector_data.data(), base + vector_addr.offset, VamanaNode::vector_bytes());
```

`load_local_node_header_acquire` 是 acquire 序的原子读 header（保证看到 deleted 位之前所有写操作都可见）。注意 `offset_id() = HEADER_SIZE`、`offset_generation() = HEADER_SIZE + ID_SIZE`，跟 `vamana_node.hh:64-65` 对应。

远端分支（`graph_access.cc:31-49`）更有意思：先选 scratch buffer（优先用当前 `StorageOwnerThread` 的 scratch，没有就退回 `peer_scratch_buffer_`），然后**两次 peer-read**：

```cpp
byte_t prefix[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
remote_read_bytes(rptr.memory_node(), vector_addr.offset, read_buffer, read_size, 0);
```

第一次读 header+id+generation（24 字节，栈上 prefix），第二次读 vector（进 scratch buffer）。为什么不一次读？因为紧凑图布局里 hot_graph_entry 紧跟 vector 后面，而 vector 长度可能远大于 24 字节，**一次读会把 hot_graph_entry 也读进来浪费带宽**——而 hot_graph_entry 有自己的解码路径（`read_neighbor_list`），不在这里读。这是 schema-15 紧凑布局带来的带宽优化。

#### `valid_local_storage_node_pointer()`（`graph_access.cc:52-73`）

校验一个本地 `RemotePtr` 是否“看起来合法”：

```cpp
if (rptr.is_null() || !local_shard(rptr.memory_node()) ||
    !VamanaNode::hot_graph_entry_available(rptr)) {
  return false;
}
```

`hot_graph_entry_available(rptr)` 检查 rptr 的对齐和区域是否允许有 hot_graph_entry（动态区和静态区都行，但某些特殊地址没有）。然后检查 header 不越界。关键在 `graph_access.cc:62-72`：

```cpp
if (rptr.byte_offset() < gpu_dynamic_node_base_) {
  return true;
}
const auto* control = ...;
const u64 high_watermark = std::atomic_ref<const u64>(
  control->dynamic_high_watermark).load(std::memory_order_acquire);
const u64 node_bytes = VamanaNode::allocation_size();
return rptr.byte_offset() <= high_watermark &&
       node_bytes <= high_watermark - rptr.byte_offset();
```

**静态区**（offset < `gpu_dynamic_node_base_`）的指针总是合法（离线构建时分配，永不回收）。**动态区**的指针必须在 `[base, high_watermark]` 区间内且能放下一个完整节点。`high_watermark` 就是 `allocate_local_node` 推进的那个值——这里 acquire 读它，保证看到 watermark 之前的所有节点初始化写操作。

#### `storage_owner_node_live()`（`graph_access.cc:75-100`）

判断一个节点当前是否 live（未 deleted）。这是反向边应用、prune 候选筛选的关键判据。

```cpp
if (rptr.is_null() || rptr.memory_node() >= num_storage_nodes_) return false;
if (!VamanaNode::hot_graph_entry_available(rptr)) return false;
if (local_shard(rptr.memory_node()) && !valid_local_storage_node_pointer(rptr)) return false;
```

先做物理合法性检查。然后读 header，看 `HEADER_DELETED` 位：

```cpp
u64 header = 0;
if (local_shard(rptr.memory_node())) {
  header = load_local_node_header_acquire(rptr);
} else {
  remote_read_bytes(rptr.memory_node(), header_address.offset, &header, sizeof(header), 0);
}
return (header & VamanaNode::HEADER_DELETED) == 0;
```

本地原子读，远端 peer-read 8 字节。**注意这是 point-in-time 检查**——返回 true 之后节点可能立刻被删，所以所有调用方在写邻居表前都要在节点锁内重新校验（见 `reverse_batch_policy.hh` 的注释）。

#### `read_neighbor_list()`（`graph_access.cc:102-149`）

读一个节点的邻居列表（紧凑图 hot_graph_entry 解码）。这是查询构造搜索里被调用最频繁的函数。

`graph_access.cc:102-113` 选 buffer：本地用栈上 `local_entry`，远端用 scratch。然后是**带重试的解码循环**（`graph_access.cc:116-137`）：

```cpp
constexpr u32 kMaxReadAttempts = 3;
for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
  if (local_shard(rptr.memory_node())) {
    std::memcpy(read_buffer, index_buffer_.get_full_buffer() + VamanaNode::hot_graph_entry_offset(rptr),
                VamanaNode::hot_graph_entry_size());
  } else {
    remote_read_bytes(rptr.memory_node(), VamanaNode::hot_graph_entry_offset(rptr),
                      read_buffer, VamanaNode::hot_graph_entry_size(), 0);
  }
  decoded_ok = VamanaNode::decode_hot_graph_entry(read_buffer, decoded.data());
  if (decoded_ok) break;
  std::this_thread::yield();
}
if (!decoded_ok) {
  return {};
}
```

为什么要重试？因为**hot_graph_entry 的写不是原子的**——`write_hot_graph_entry` 是 read-modify-write 整个 entry，期间 checksum 会暂时不一致。并发读时如果读到“写了一半”的 entry，checksum 校验失败，`decode_hot_graph_entry` 返回 false。重试 3 次通常能撞上一次完整的写。`yield()` 给写者让 CPU。如果 3 次都失败就返回空 vector——调用方会把这个节点当叶子处理（不展开）。**这是性能与正确性的权衡**：3 次失败极罕见，强行重试只会拖慢整条搜索路径。

`graph_access.cc:138-148` 解析 decoded buffer：

```cpp
const u8 edge_count = *reinterpret_cast<const u8*>(parse_buffer + VamanaNode::neighbor_count_offset_in_read());
const auto* slots = reinterpret_cast<const RemotePtr*>(parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
vec<RemotePtr> neighbors;
neighbors.reserve(edge_count);
for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
  if (!slots[i].is_null()) {
    neighbors.push_back(slots[i]);
  }
}
```

`neighbor_read_size() = 8 + R * sizeof(RemotePtr)`（第 7 课），前 8 字节是元数据（edge_count 等），后 `R * 8` 字节是 `RemotePtr` 数组。**双重上界**：`i < edge_count`（实际数量）和 `i < R`（物理容量），防止 entry 损坏时读到垃圾 edge_count 越界。跳过 null slot——删除邻居时通常写 null 而不是 compact。

#### `read_local_neighbor_list()`（`graph_access.cc:151-191`）

跟 `read_neighbor_list` 的本地分支逻辑相同，但**复用调用方提供的 buffer**（`entry`、`decoded`、`neighbors`），避免每次分配。这是 `partition_local_search_candidates` 在搜索循环里调用的版本——beam search 每轮都要读邻居，反复分配 vector 会成为热点。注意 `graph_access.cc:155-160` 的断言强制本地：

```cpp
lib_assert(local_shard(rptr.memory_node()),
           "local neighbor lookup received a remote pointer");
```

#### `async_read_node_snapshot()` / `async_read_node_snapshots()`（`graph_access.cc:193-251`）

协程版的快照读。单点版（`graph_access.cc:193-205`）：本地直接同步读，远端 post async peer-read 到 `thread.coroutine_scratch()`，返回 awaitable。

批量版 `async_read_node_snapshots`（`graph_access.cc:207-251`）更复杂：

```cpp
const size_t snapshot_size = snapshot_buffer_bytes();
const size_t snapshot_stride = aligned_snapshot_bytes();
const u32 max_batch = storage_owner_snapshot_batch_size(config, &thread);
lib_assert(rptrs.size() <= max_batch, "storage-owner snapshot batch exceeds configured limit");
```

batch 大小受 `storage_owner_snapshot_batch_size` 限制（见 detail.hh）。然后遍历 rptrs：本地的同步读直接进 `awaitable.snapshots`；远端的算出 scratch offset，post async peer-read，记录到 `awaitable.pending`：

```cpp
const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
lib_assert(scratch_offset + snapshot_size <= thread.scratch_stride, ...);
byte_t* buffer = thread.coroutine_scratch(scratch_offset);
post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                     VamanaNode::size_until_vector_end());
awaitable.pending.push_back(NodeSnapshotsReadAwaitable::PendingRead{rptr, buffer});
```

`scratch_stride` 是每个快照在 scratch 里的步长（cacheline 对齐），`remote_slot` 是远端快照的序号（本地快照不占 scratch 槽）。断言保证不越界。`awaitable.ready = false` 让协程调度器知道需要 poll CQ。

`read_node_snapshots_batched`（`graph_access.cc:253-328`）是同步包装：在每个 batch 内 post 所有 peer-read，然后 poll CQ 直到全部完成，再 `parse_remote_snapshot` 收集结果。它被 `robust_prune_cpu`、`apply_local_reverse_update` 等非协程路径调用。

#### `write_hot_graph_entry()`（`graph_access.cc:348-416`）

写一个节点的邻居列表（紧凑图）。这是图修改的“最后一公里”——所有 insert/reverse-update 最终都调到这里。

`graph_access.cc:358-367` 先读旧 entry（本地 memcpy 或远端 peer-read），用于保留 generation 和 deleted 标志：

```cpp
vec<byte_t> previous(entry_size, 0);
if (local_shard(rptr.memory_node())) {
  std::memcpy(previous.data(), index_buffer_.get_full_buffer() + hot_offset, entry_size);
} else {
  remote_read_bytes(rptr.memory_node(), hot_offset, previous.data(), previous.size(), 0);
}
```

`graph_access.cc:369-393` 解码旧 entry 的 generation 和 deleted：

```cpp
const bool previous_valid =
  vamana::hot_graph::load_u16_le(previous.data() + 2) ==
    vamana::hot_graph::checksum16(previous.data(), previous.size());
u32 generation = previous_valid ? vamana::hot_graph::load_u32_le(previous.data() + 4) : 0;
bool deleted = previous_valid && (previous[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
if (!previous_valid) {
  // fallback: 从 header 和 compact meta 重新读
  ...
}
```

如果旧 entry checksum 不对（并发写、初始化未完成），就从 header 和 compact metadata 重新读 generation 和 deleted。这是“自愈”逻辑——一个损坏的 entry 不会让 write 失败，而是从权威源头重建。`graph_access.cc:394-395` 允许调用方 override generation 和 deleted（用于 `write_new_node` 强制设 deleted=false）。

`graph_access.cc:397-410` 编码新 entry：

```cpp
vec<byte_t> entry(entry_size, 0);
const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
VamanaNode::encode_hot_graph_entry(entry.data(), edge_count, neighbors.data(), edge_count,
                                   VamanaNode::HOT_GRAPH_SHARD_BITS, generation, false);
if (deleted) {
  entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
  const u16 checksum = vamana::hot_graph::checksum16(entry.data(), entry.size());
  vamana::hot_graph::store_u16_le(entry.data() + 2, checksum);
}
```

`encode_hot_graph_entry` 内部已经算过一次 checksum，但如果再置 deleted 位就要重算（因为标志位变了）。**deleted 节点保留邻居列表**——注释 `graph_access.cc:404-406` 解释：删除后 GPU 看 tombstone 忽略 payload，但维护代码可能还要用邻居做 cleanup，所以 adjacency 要保留。

`graph_access.cc:411-415` 写回：本地 memcpy，远端 peer-write。

#### `write_neighbor_list()`（`graph_access.cc:418-428`）

`write_neighbor_list` 是 `write_hot_graph_entry` 的薄封装，只多了一个 neighbor_addr 越界断言。所有“写邻居表”的路径都走这里。

#### `write_dynamic_navigation_code()`（`graph_access.cc:430-443`）

写 PQ 导航码（GPU 查询时用来做粗排，第 9 课）。注意断言 `graph_access.cc:432-434`：**PQ 码只能写给本地动态节点**。`gpu_navigation_model_` 是 PQ 模型，`encode` 把 float 向量编码成 `code_bytes()` 长的字节串，写到 `VamanaNode::dynamic_navigation_code_offset(rptr)`。`thread_local vec<f32> transformed` 是 encode 内部的 scratch，避免反复分配。

#### `write_new_node()`（`graph_access.cc:445-459`）

写一个全新节点，组合了上面几个原语：

```cpp
byte_t* ptr = local_node_ptr(rptr);
std::memset(ptr, 0, VamanaNode::allocation_size());
*reinterpret_cast<u64*>(ptr) = 0;
*reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
*reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                               ptr + VamanaNode::offset_vector());
write_dynamic_navigation_code(rptr, components);
write_hot_graph_entry(rptr, neighbors, generation, false);
```

清零整个节点区域，写 header=0（无 lock、无 deleted、无 medoid）、id、generation，编码 float vector，写 PQ 码，最后写邻居表（generation override = 新 generation，deleted = false）。**只支持本地节点**（`local_node_ptr` 不接受远端）——新节点总是分配在本 owner 分片，跨分片写通过反向边 RPC（第 24 课）。

#### `lock_node()` / `unlock_node()`（`graph_access.cc:461-504`）

节点级自旋锁，保护邻居表的 read-modify-write。锁位是 `HEADER_NODE_LOCK`（bit 0-1，`vamana_node.hh:19`）。

本地分支 `lock_node`（`graph_access.cc:462-477`）：

```cpp
auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + ...header.offset);
std::atomic_ref<u64> ref(*header_ptr);
for (;;) {
  u64 header = ref.load(std::memory_order_acquire);
  if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
    std::this_thread::yield();
    continue;
  }
  const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
  if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
}
```

经典的 test-and-test-and-set 自旋锁：先 acquire 读，看到锁空闲再 CAS 抢。CAS 失败说明有人抢了，重试。`yield()` 避免空转浪费 CPU。

远端分支 `graph_access.cc:479-487` 调 `try_lock_remote_header`，返回 `(success, header)`。看到锁被占就 yield。

`unlock_node` 本地 `fetch_and(~LOCK)` 清锁位（`graph_access.cc:490-497`）。远端分支 `graph_access.cc:499-503` 有意思：

```cpp
const byte_t unlock = 0;
remote_write_bytes(rptr.memory_node(),
                   vamana::StorageLayoutResolver::header(rptr).offset +
                     VamanaNode::HEADER_UNTIL_LOCK,
                   &unlock, 1, 0);
```

`HEADER_UNTIL_LOCK = 0`——远端解锁是写 header 的第 0 字节为 0。这暗示远端锁实现是“header 第 0 字节非零即锁”，比本地用整个 u64 的 CAS 弱。这是 RDMA 层的限制：单字节 write 比 8 字节 CAS 便宜得多。这种不对称是分布式锁的常见妥协。

---

### 4. `candidate_search.cc`：owner 分片内构造搜索

#### `beam_search_candidates()`（`candidate_search.cc:6-141`）

这是同步版的 owner 分片内 beam search，用于 stage1 构造搜索。算法是标准 Vamana search（第 6 课）：从 medoid 出发，维护一个 beam，每轮选最近未展开节点，展开其邻居，更新 beam，直到无未展开节点。

`candidate_search.cc:10-26` 初始化：

```cpp
hashset_t<RemotePtr> visited;
vec<BeamEntry> beam;
...
NodeSnapshot medoid_snapshot;
read_node_snapshot(medoid, medoid_snapshot);
const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
beam.push_back({medoid, medoid_dist, false});
visited.insert(medoid);
```

`BeamEntry = {rptr, distance, expanded}`（`storage_owner_state.hh:20-24`）。读 medoid 快照，算距离，入 beam。

主循环 `candidate_search.cc:37-111`：

```cpp
for (;;) {
  i32 best_idx = -1;
  distance_t best_dist = std::numeric_limits<distance_t>::max();
  for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
    if (!beam[i].expanded && beam[i].distance < best_dist) {
      best_dist = beam[i].distance;
      best_idx = i;
    }
  }
  if (best_idx < 0) break;
  beam[best_idx].expanded = true;
  const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
  ...
}
```

`best_idx < 0` 即“无未展开节点”，是 Vamana search 的收敛条件。**注意没有独立的深度/扩展数上限**——beam 宽度 L 隐式限制了扩展数（beam 满了之后只能替换更远的，不会无限扩展）。

`candidate_search.cc:63-77` 收集未访问邻居：

```cpp
vec<RemotePtr> unvisited_neighbors;
for (const RemotePtr& neighbor : neighbors) {
  if (neighbor.is_null() || visited.contains(neighbor)) continue;
  visited.insert(neighbor);
  unvisited_neighbors.push_back(neighbor);
}
```

**先去重再批量读快照**——避免对同一节点反复 peer-read。

`candidate_search.cc:83-110` 分批读快照、算距离、更新 beam：

```cpp
const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
const u32 construction_width = storage_owner_construction_width(config);
for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
  ...
  vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
  for (const NodeSnapshot& snapshot : snapshots) {
    if (snapshot.deleted) continue;
    const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
    insert_into_beam(beam, snapshot.rptr, dist, construction_width);
  }
}
```

跳过 deleted 节点（不进 beam），`insert_into_beam` 把新候选插入 beam 并裁剪到 `construction_width`。`breakdown` 是性能计数器（第 30 课 breakdown benchmark）。

`candidate_search.cc:128-140` 排序输出候选列表（按距离升序）。

#### `beam_search_candidates_async()`（`candidate_search.cc:143-288`）

协程版，逻辑相同但用 `co_await` 调 async 读原语。关键差异：

`candidate_search.cc:148-159` 从协程 scratch 取所有 buffer，避免每次分配：

```cpp
StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
scratch.clear_search();
hashset_t<RemotePtr>& visited = scratch.visited;
vec<BeamEntry>& beam = scratch.beam;
vec<RemotePtr>& unvisited_neighbors = scratch.unvisited;
vec<RemotePtr>& batch = scratch.batch;
```

`candidate_search.cc:162` 用 `co_await async_read_node_snapshot(medoid, thread)` 读 medoid——本地立即返回，远端挂起让调度器 poll CQ。

主循环里 `co_await async_read_neighbor_list` 和 `co_await async_read_node_snapshots` 同理。`candidate_search.cc:275-287` 把结果写到 `storage_owner_async_candidates_[thread.id][thread.running_coroutine]`——这是预先分配的二维 slot，协程 id + 协程内 slot 唯一定位一个候选列表。调用方（`execute_storage_owner_insert_job_async`）通过这个 slot 拿结果。

#### `partition_local_search_candidates()`（`candidate_search.cc:290-400`）

这是**分片内局部搜索**版本，跟 `beam_search_candidates` 的区别是：它只在**本 owner 分片**内搜索，不跨分片读邻居。用于 stage2 维护期的局部搜索（第 26 课）。

`candidate_search.cc:296-312` 优先用协程 scratch，否则栈上临时 vector。`candidate_search.cc:314-318` 决定是否走“精确整型 L2”路径：

```cpp
const bool exact_integral_query = integral_raw_query != nullptr &&
  integral_byte_l2_sum_exact_in_float(config.dim) &&
  (dtype == VectorDType::uint8 || dtype == VectorDType::int8);
```

注释 `candidate_search.cc:330-335` 解释：peer stage2 携带精确 uint8/int8 字节，DIM=258 时整型平方和能在 float 里精确表示，**同 dtype AVX2 路径保持候选排序且避免反复 byte-to-float 转换**。这是性能优化。

`candidate_search.cc:319-343` 的 `score` lambda：

```cpp
auto score = [&](RemotePtr candidate) -> std::optional<distance_t> {
  const byte_t* vector = local_live_vector(candidate);
  if (vector == nullptr) return std::nullopt;
  const distance_t distance = exact_integral_query
    ? typed_l2_distance(integral_raw_query, dtype, vector, dtype, config.dim)
    : distance_to_stored_vector(query, vector, config);
  return distance;
};
```

`local_live_vector(candidate)` 返回本地 live 节点的 vector 指针，null 表示节点不 live（deleted 或无效）。返回 `nullopt` 让搜索跳过该候选。

`candidate_search.cc:344-367` 的 `expand` lambda：

```cpp
auto expand = [&](RemotePtr candidate, auto&& visit) {
  bool decoded = read_local_neighbor_list(candidate, neighbors, neighbor_entry, neighbor_decoded);
  if (!decoded) {
    lock_node(candidate);
    decoded = read_local_neighbor_list(candidate, neighbors, neighbor_entry, neighbor_decoded);
    unlock_node(candidate);
    lib_assert(decoded, "partition-local construction search could not decode a locked adjacency snapshot");
  }
  for (const RemotePtr neighbor : neighbors) {
    visit(neighbor);
  }
};
```

**fallback 加锁**：如果乐观读（无锁）3 次都解码失败，就上锁再读一次。注释 `candidate_search.cc:349-353` 解释：并发 adjacency 发布可能让所有乐观 checksum 都失败，上锁保证读到一致快照，避免把热点节点误当叶子。

`candidate_search.cc:371-375` 用 `thread_local` 的 `PartitionLocalSearchBeam` 复用——这个 wrapper 永不挂起，所以一个 OS thread 一个状态不会并发。

`candidate_search.cc:377-386` 最后再 filter 一次，确保所有候选都还 live：

```cpp
filter_final_partition_local_beam(final_beam, [&](RemotePtr candidate) {
  return storage_owner_node_live(candidate);
});
```

**因为搜索期间可能有节点被删**，filter 保证返回的候选都是 point-in-time live。但调用方在写邻居表前还要再校验一次（reverse_batch_policy.hh 的注释强调）。

---

### 5. `robust_prune_policy.hh`：alpha RobustPrune 通用策略

这个头文件只有一个模板函数 `select_alpha_robust_pruned_sorted`（`robust_prune_policy.hh:15-50`），是 Vamana alpha-RobustPrune 的算法核心，被 4 处复用：

- `graph_mutation.cc` 的 `robust_prune_cpu`（新节点剪枝）
- `graph_mutation.cc` 的 `robust_prune_snapshots_cpu`（反向边溢出剪枝）
- `reverse_batch.cc` 的批量剪枝
- `two_stage_insert_oracle.hh` 的 oracle 剪枝

算法（`robust_prune_policy.hh:31-49`）：

```cpp
for (size_t candidate_index = 0;
     candidate_index < sorted_candidates.size() &&
     selected.size() < result_limit;
     ++candidate_index) {
  const Candidate& candidate = sorted_candidates[candidate_index];
  bool pruned = false;
  for (const size_t selected_index : selected_indices) {
    if (alpha * pair_distance(candidate, sorted_candidates[selected_index]) <=
        source_distance_of(candidate)) {
      pruned = true;
      break;
    }
  }
  if (!pruned) {
    selected.push_back(pointer_of(candidate));
    selected_indices.push_back(candidate_index);
  }
}
```

**前置条件**：`sorted_candidates` 必须按到 source 的距离升序排好。算法逐个考虑候选（最近的先选），对每个候选检查：是否存在已选中的候选 `retained`，使得 `alpha * dist(candidate, retained) <= dist(candidate, source)`。如果是，说明 `retained` 已经“覆盖”了 `candidate` 的方向（在 alpha 放大下 `retained` 比 `candidate` 更接近 source 的某个邻居方向），剪掉 `candidate`。

`alpha > 1` 时剪枝更激进（保留更少、更多样的邻居），`alpha = 1` 时退化为严格 greedy。`result_limit` 是 `config.R`（最大度）。模板参数 `PointerOf`/`SourceDistanceOf`/`PairDistance` 是三个回调，让算法独立于候选的具体存储形式（`StorageOwnerPruneCandidateInfo`、`StorageOwnerScoredSnapshot`、`PartitionLocalSearchEntry` 等都能用）。

**注意 `<=` 而非 `<`**：等号情况也剪掉，这是 Vamana 论文的标准定义，保证幂等性（同一候选集多次剪枝结果相同）。

---

### 6. `reverse_batch_policy.hh`：反向边候选筛选

`select_fresh_reverse_candidates_locked`（`reverse_batch_policy.hh:17-35`）是反向边应用的“候选净化”函数，**调用方必须持有 target 节点锁**（注释 `reverse_batch_policy.hh:11-15`）：

```cpp
template <class IsLive>
void select_fresh_reverse_candidates_locked(
    const vec<RemotePtr>& current_neighbors,
    const vec<RemotePtr>& candidates,
    IsLive&& is_live,
    vec<RemotePtr>& selected) {
  selected.clear();
  selected.reserve(candidates.size());
  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() ||
        std::find(current_neighbors.begin(), current_neighbors.end(), candidate) != current_neighbors.end() ||
        std::find(selected.begin(), selected.end(), candidate) != selected.end() ||
        !is_live(candidate)) {
      continue;
    }
    selected.push_back(candidate);
  }
}
```

筛选规则：
1. 跳过 null
2. 跳过已经在 current_neighbors 里的（避免重复边）
3. 跳过已经选中的（去重）
4. 跳过不 live 的（`is_live` 回调，通常 = `storage_owner_node_live`）

注释特别强调 **liveness 在锁内检查**的原因：删节点的 cleanup 如果在锁前完成，这里看到 dead；如果在锁后开始，cleanup 必须等锁，会在写完反向边后把新写的 backlink 清掉。这就是“锁边界 = liveness 边界”的并发安全保证。

---

### 7. `partition_local_search.hh`：分片内 beam search 原语

这个头文件是**算法层**，完全不涉及 I/O，可单元测试。

`PartitionLocalSearchEntry`（`partition_local_search.hh:18-22`）：`{rptr, distance, expanded}`，跟 `BeamEntry` 同构但独立定义，避免算法层依赖存储层。

`PartitionLocalSearchBeam` 类（`partition_local_search.hh:24-100`）维护一个**始终有序、宽度不超过 L** 的 beam：

- `try_visit(pointer)`（`partition_local_search.hh:47-52`）：只接受本分片的非 null 指针，去重。**访问与评分分离**——rejected/deleted 候选也标记 visited，避免重复 score。
- `add_visited(pointer, distance)`（`partition_local_search.hh:56-64`）：`lower_bound` 找插入位置，保持有序，超宽时 `resize(beam_width_)` 截断。
- `take_closest_unexpanded()`（`partition_local_search.hh:68-77`）：返回最近的未展开节点并标记 expanded。`nullopt` 是收敛条件。

`partition_local_construction_search_into`（`partition_local_search.hh:108-138`）是完整搜索循环：

```cpp
search.reset(partition_id, beam_width);
auto consider = [&](RemotePtr pointer) {
  if (!search.try_visit(pointer)) return;
  const std::optional<distance_t> distance = std::invoke(score, pointer);
  if (distance.has_value()) {
    search.add_visited(pointer, *distance);
  }
};
for (const RemotePtr entry : entry_points) {
  consider(entry);
}
while (const std::optional<RemotePtr> current = search.take_closest_unexpanded()) {
  std::invoke(expand, *current, consider);
}
return search.mutable_final_beam();
```

`score`/`expand` 是回调，让算法层不依赖存储层。`filter_final_partition_local_beam`（`partition_local_search.hh:159-169`）是 point-in-time liveness 过滤，注释 `partition_local_search.hh:155-158` 强调“调用方在 mutation boundary 还要再校验”。

---

### 8. `two_stage_insert_oracle.hh`：两阶段插入决策参考

这个头文件是**算法 oracle**——它不是运行时代码，而是用来验证“direct 插入”和“two-stage 插入”在静态图快照下结果等价的参考实现。注释 `two_stage_insert_oracle.hh:14-29` 说得很清楚：

> Algorithm-only reference for the partitioned insertion semantics. It is deliberately independent of RPCs, queues, retries, and graph mutation... Direct-vs-staged equality assumes both observe the same logical graph snapshot. The production runtime protects the inserted target's generation and revalidates final neighbors, but it deliberately does not freeze every shard for the duration of stage2. Under concurrent graph changes the claim is therefore quiescent/reference equivalence plus eventual cleanup, not a byte-for-byte linearizable construction history.

**关键结论**：生产环境**不冻结所有分片**做 stage2，所以两阶段插入在并发下不是 byte-for-byte 线性化的，而是“静态等价 + 最终清理”。

`PartitionedInsertStage1`（`two_stage_insert_oracle.hh:36-45`）保存 stage1 的 owner beam 和 temporary neighbors。注释 `two_stage_insert_oracle.hh:188-190` 强调：

> Prune a copy: the complete owner beam is the stage boundary and must not be reduced to the temporary outgoing edge set before stage 2.

**stage1 的临时出边是 reduced 的（剪枝后），但 owner beam 必须保留完整的 width-L**——因为 stage2 要拿完整 owner beam 跟其他分片的 beam 合并再剪枝，临时出边只是为了“立即可见性”，不参与等价性证明。

三个模板函数：
- `partitioned_direct_insert_reference`（`two_stage_insert_oracle.hh:119-153`）：所有分片都搜一遍，merge 所有 beam，RobustPrune 一次。这是 baseline。
- `partitioned_two_stage_insert_begin`（`two_stage_insert_oracle.hh:155-192`）：只搜 owner 分片，存 beam + 临时出边。
- `partitioned_two_stage_insert_finalize`（`two_stage_insert_oracle.hh:194-239`）：复用 stage1 的 owner beam（不重搜），搜其他分片，merge，RobustPrune。

`partitioned_insert_candidate_capacity`（`two_stage_insert_oracle.hh:47-64`）= `partition_count * beam_width`，是 merged beam 的最大容量。`append_partitioned_insert_beam`（`two_stage_insert_oracle.hh:75-86`）有边界检查，超过 capacity 抛 `length_error`。

**生产代码不直接调这些函数**，但 `graph_mutation.cc` 的 `execute_storage_owner_insert_job_async` 是它的运行时对应物：stage1 在 owner 分片 beam_search（`graph_mutation.cc:373-385`），stage2 的跨分片候选合并由 maintenance worker（第 26 课）异步完成。理解这个 oracle 就理解了 dvstor 两阶段插入的设计意图。

---

### 9. `graph_mutation.cc`：图修改主流程

#### `robust_prune_cpu()`（`graph_mutation.cc:7-105`）

新节点出边的 RobustPrune。输入：source 向量（新插入节点的向量）、候选列表（beam search 结果）、skip set（要排除的指针，通常是新节点自己）。

`graph_mutation.cc:14-38` 取 scratch buffer，算 `result_limit`（默认 `config.R`，可 override 但不超过 R）。

`graph_mutation.cc:40-45` 预过滤：跳过 null 和 skip 集合。

`graph_mutation.cc:47-69` 分批读候选快照、算距离，构造 `StorageOwnerPruneCandidateInfo{rptr, dist, vector_data}`：

```cpp
for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
  ...
  vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
  for (NodeSnapshot& snapshot : snapshots) {
    if (snapshot.deleted) continue;
    const distance_t dist = distance_between_vectors(source, source_dtype,
                                                     snapshot.vector_data.data(),
                                                     VamanaNode::vector_dtype(), config);
    infos.push_back({snapshot.rptr, dist, std::move(snapshot.vector_data)});
  }
}
```

**保留 vector_data 在 info 里**——RobustPrune 的 pair_distance 需要候选之间的距离，必须缓存向量避免反复 peer-read。

`graph_mutation.cc:71-78` 按 dist 升序排序。`graph_mutation.cc:80-102` 调 `select_alpha_robust_pruned_sorted`，pair_distance lambda 用缓存的 vector_data 算候选间距离。返回选中的 `RemotePtr` 列表。

#### `robust_prune_snapshots_cpu()`（`graph_mutation.cc:107-183`）

跟 `robust_prune_cpu` 类似，但**输入是已经读好的 `NodeSnapshot` 列表**（调用方已经批量读过），不再做 peer-read。用于反向边溢出剪枝——`apply_local_reverse_update` 已经把当前邻居 + 新候选都快照过了，直接复用。

`graph_mutation.cc:142-155` 用 `seen` set 去重（同一 rptr 多次出现只保留第一次），跳过 deleted 和 vector 不足的 snapshot。

#### `apply_partition_local_reverse_update()`（`graph_mutation.cc:185-304`）

**分片内反向边应用**——被 `batch_execution.cc` 在 stage1 完成后调用，处理本 owner 分片内的反向边。注释 `graph_mutation.cc:193-194` 强制 target 本地，`graph_mutation.cc:205-206` 强制候选也本地。

`graph_mutation.cc:199-214` 去重候选。`graph_mutation.cc:216-221` 上锁后检查 target 是否 deleted——deleted 节点不接反向边。

`graph_mutation.cc:223-243` 读当前邻居，分类：

```cpp
vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
vec<RemotePtr> preserved_external;
vec<RemotePtr> local_candidates;
bool changed = false;
for (const RemotePtr& neighbor : current_neighbors) {
  if (neighbor.is_null()) { changed = true; continue; }
  if (!storage_owner_node_live(neighbor)) { changed = true; continue; }
  if (local_shard(neighbor.memory_node())) {
    local_candidates.push_back(neighbor);
  } else {
    preserved_external.push_back(neighbor);
  }
}
```

**保留 external（跨分片）邻居**，只对 local 邻居 + 新候选做剪枝。这是分片-local 优化的关键：跨分片邻居的修改要走 RPC，本函数只管本地。`changed` 标记是否有 null/dead 邻居被清理。

`graph_mutation.cc:245-254` 加入新候选（liveness 校验 + 去重）。

`graph_mutation.cc:260-280` 算 local 容量并剪枝：

```cpp
const u32 local_capacity = preserved_external.size() >= config.R
                             ? 0
                             : config.R - static_cast<u32>(preserved_external.size());
vec<RemotePtr> selected_local;
if (local_candidates.size() <= local_capacity) {
  selected_local = std::move(local_candidates);
} else if (local_capacity > 0) {
  ...
  selected_local = robust_prune_cpu(target_vector, VamanaNode::vector_dtype(),
                                    local_candidates, skip, config, nullptr, local_capacity);
}
```

**local_capacity = R - preserved_external.size()**——给 external 邻居预留槽位。如果 local 候选数超过容量，调 `robust_prune_cpu` 用 `local_capacity` 作为 override limit 剪枝。

`graph_mutation.cc:282-298` 合并 external + selected_local，校验不超过 R，写回。`changed_neighbors` 判断是否真的变了（避免无谓写）。

#### `execute_storage_owner_insert_job_async()`（`graph_mutation.cc:306-421`）

**这是本课的核心**——insert/upsert/delete 的主流程协程。逐段讲解。

`graph_mutation.cc:312-318` 准备 mutation：

```cpp
const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                               VamanaNode::DIM};
FreshnessEntry old_entry{};
u32 generation = 0;
const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
job.old_ptr = old_entry.current;
job.generation = generation;
```

`prepare_mutation` 做 inflight 占位 + 旧 entry 查询 + 新 generation 分配。`job.old_ptr` 给调用方用于 invalidate 旧节点（compute 侧 owner 更新）。

`graph_mutation.cc:319-326` 失败路径：

```cpp
const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
if (status != service::storage_owner::MutationStatus::ok) {
  complete_storage_owner_maintenance_sequence(job.maintenance_sequence, job.reserved_maintenance_work);
  job.status = status;
  job.ok = false;
  co_return;
}
```

失败时调 `complete_storage_owner_maintenance_sequence` 释放预留的维护 slot（第 26 课），避免维护队列卡住。

`graph_mutation.cc:327-341` **delete 分支**：

```cpp
if (job.kind == service::storage_owner::MutationKind::erase) {
  job.ok = mark_node_deleted(old_entry.current, generation);
  job.status = job.ok ? ...ok : ...failed;
  if (job.ok) {
    publish_mutation(job.id, old_entry.current, generation, true);
    job.maintenance_sequence = schedule_storage_owner_maintenance(
        job.id, generation, job.kind, RemotePtr{}, old_entry.current,
        job.maintenance_sequence, job.reserved_maintenance_work, config);
  } else {
    complete_storage_owner_maintenance_sequence(...);
  }
  co_return;
}
```

delete 不分配新节点，直接 `mark_node_deleted` 置 tombstone，`publish_mutation(deleted=true)` 让 freshness map 反映删除，`schedule_storage_owner_maintenance` 调度 stage2 cleanup（第 26 课 maintenance worker 会清理反向边）。

`graph_mutation.cc:342-370` **medoid 空特判**：

```cpp
lib_assert(!local_stitch_enabled(config),
           "local stage1 must run on its dedicated CPU executor");
RemotePtr medoid_ptr{};
const vec<RemotePtr>* candidates = nullptr;
medoid_ptr = co_await async_read_global_medoid(thread);
if (medoid_ptr.is_null()) {
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  write_new_node(new_ptr, job.id, components, {}, generation);
  RemotePtr observed;
  if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
    job.ok = true;
    ...
    publish_mutation(job.id, new_ptr, generation, false);
    job.maintenance_sequence = schedule_storage_owner_maintenance(...);
    co_return;
  }
  medoid_ptr = observed;
}
```

**第一个插入的节点成为 medoid**：如果 global medoid 是 null，分配新节点、写空邻居表、CAS 把自己设为 medoid（expected=null）。CAS 成功或 observed 仍 null 都算成功。CAS 失败（有人抢先）就用 observed 的 medoid 继续搜索。这是无锁的 medoid 初始化。

`graph_mutation.cc:372-385` **stage1 beam search**：

```cpp
auto t_search = std::chrono::steady_clock::now();
auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
co_await std::suspend_always{};
while (!search.handle.done()) {
  if (thread.is_ready(thread.running_coroutine)) {
    search.handle.resume();
  } else {
    co_await std::suspend_always{};
  }
}
search.handle.destroy();
breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
candidates = &storage_owner_async_candidates_[thread.id][thread.running_coroutine];
```

**嵌套协程**：`execute_storage_owner_insert_job_async` 自己是协程，它内部又启动 `beam_search_candidates_async` 协程并手动驱动。`co_await std::suspend_always{}` 让出当前协程给调度器 poll CQ，`thread.is_ready` 检查 peer-read 是否完成，完成就 `resume` 内层协程。`search.handle.destroy()` 手动销毁内层协程帧（结果已经拷到 `storage_owner_async_candidates_`）。

`graph_mutation.cc:387-393` **RobustPrune 选最终邻居**：

```cpp
StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
scratch.empty_skip.clear();
vec<RemotePtr> selected_neighbors = robust_prune_cpu(
    reinterpret_cast<const byte_t*>(components.data()),
    VectorDType::float32, *candidates, scratch.empty_skip, config, &breakdown);
```

source = 新节点向量（float32），candidates = beam search 结果，skip = 空。返回的就是新节点的最终出边。

`graph_mutation.cc:394-405` **分配新节点 + 写入 + 发布**：

```cpp
const RemotePtr new_ptr = allocate_local_node();
job.new_ptr = new_ptr;
write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
  mark_node_deleted(old_entry.current, old_entry.generation);
}
publish_mutation(job.id, new_ptr, generation, false);
job.maintenance_sequence = schedule_storage_owner_maintenance(
    job.id, generation, job.kind, new_ptr, old_entry.current,
    job.maintenance_sequence, job.reserved_maintenance_work, config);
```

**upsert 特判**：如果是 upsert 且旧节点 live（`!old_entry.deleted`），mark 旧节点 deleted。注意 `old_entry.deleted` 为 true 时 `old_entry.current` 已被 `prepare_mutation` 清成 null（见 allocation.cc:145-147），这里不会误删。

`schedule_storage_owner_maintenance` 调度 stage2 maintenance——异步处理反向边、跨分片候选合并等（第 26 课）。

`graph_mutation.cc:407-418` **maintenance 禁用时立即处理反向边**：

```cpp
if (!maintenance_enabled) {
  for (const RemotePtr& neighbor_ptr : selected_neighbors) {
    if (local_shard(neighbor_ptr.memory_node())) {
      local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
      job.invalidated_neighbors.push_back(neighbor_ptr.raw_address);
    } else {
      remote_updates[neighbor_ptr.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
      job.invalidated_neighbors.push_back(neighbor_ptr.raw_address);
    }
  }
}
```

**两条路径**：本地邻居进 `local_updates`（后续 `apply_partition_local_reverse_update` 处理），跨分片邻居进 `remote_updates`（后续 `send_reverse_update_batch` RPC，第 24 课）。`job.invalidated_neighbors` 记录所有受影响的邻居，给 compute 侧 invalidate 缓存用（第 28 课）。

如果 maintenance 启用，反向边由 stage2 maintenance worker 异步处理（第 26 课），这里不立即做。

#### `apply_local_reverse_update()`（`graph_mutation.cc:423-567`）

**单目标反向边应用**，比 `apply_partition_local_reverse_update` 更完整（支持跨分片候选、有冲突重试）。被 `reverse_batch.cc` 的冲突 fallback 路径调用。

`graph_mutation.cc:432-448` 读 target 向量、去重候选。

`graph_mutation.cc:450-453` 的 `target_deleted` lambda 是锁内重检 deleted 的工具。

`graph_mutation.cc:462-487` **快速路径**：

```cpp
for (;;) {
  lock_node(target_ptr);
  if (target_deleted()) { unlock_node(target_ptr); return true; }
  current_neighbors = read_neighbor_list(target_ptr);
  select_fresh_reverse_candidates_locked(current_neighbors, unique_candidates,
    [this](const RemotePtr& candidate) { return storage_owner_node_live(candidate); },
    fresh_candidates);
  if (fresh_candidates.empty()) { unlock_node(target_ptr); return true; }
  if (current_neighbors.size() + fresh_candidates.size() <= config.R) {
    current_neighbors.insert(current_neighbors.end(), fresh_candidates.begin(), fresh_candidates.end());
    write_neighbor_list(target_ptr, current_neighbors);
    unlock_node(target_ptr);
    return true;
  }
  unlock_node(target_ptr);
  ...
}
```

**快速路径**：如果当前邻居 + 新候选不超过 R，直接 append 写回，无需剪枝。这是常见情况——大多数反向边不会让邻居溢出。

`graph_mutation.cc:490-511` **溢出路径**（注释 `graph_mutation.cc:490-494` 解释）：snapshot 当前+fresh 集合，在锁外做 alpha RobustPrune（`robust_prune_snapshots_cpu`）。注释强调：

> The final locked compare/revalidation below makes this optimistic calculation safe under concurrent reverse updates and deletes.

`graph_mutation.cc:513-545` **锁内 revalidation + 写回**：

```cpp
lock_node(target_ptr);
if (target_deleted()) { unlock_node(target_ptr); return true; }
const vec<RemotePtr> observed_neighbors = read_neighbor_list(target_ptr);
const bool unchanged = observed_neighbors.size() == current_neighbors.size() &&
                       std::equal(observed_neighbors.begin(), observed_neighbors.end(),
                                  current_neighbors.begin());
if (!unchanged) { ++conflicts; unlock_node(target_ptr); continue; }
select_fresh_reverse_candidates_locked(observed_neighbors, fresh_candidates,
  [this](const RemotePtr& candidate) { return storage_owner_node_live(candidate); },
  revalidated_candidates);
const bool candidates_unchanged = revalidated_candidates.size() == fresh_candidates.size() &&
                                   std::equal(revalidated_candidates.begin(),
                                              revalidated_candidates.end(),
                                              fresh_candidates.begin());
if (!candidates_unchanged) { ++conflicts; unlock_node(target_ptr); continue; }
write_neighbor_list(target_ptr, selected_neighbors);
unlock_node(target_ptr);
```

**乐观锁协议**：
1. 锁前快照 current_neighbors 和 fresh_candidates
2. 锁外做 RobustPrune（耗时，不持锁）
3. 重新加锁，检查 current_neighbors 是否变了（被人插队）
4. 检查 fresh_candidates 是否还 live 且去重后一致
5. 任一不一致就 `++conflicts` 重试（回到 for 循环开头）

`graph_mutation.cc:548-564` 慢日志：如果整个 update 超过 1 秒，记录前 16 次的 target/conflicts/elapsed，帮助诊断热点 target。

**conflicts 是本函数的关键指标**——高 conflicts 说明某 target 被并发反向边频繁争抢，可能需要调度优化。

---

### 10. `reverse_batch.cc`：反向边批量应用

`apply_local_reverse_updates_batched()`（`reverse_batch.cc:32-279`）是 stage2 maintenance worker（第 26 课）批量处理反向边的入口。它比 `apply_local_reverse_update` 更高效：**一次性 snapshot 所有需要的向量，批量 RobustPrune，减少 peer-read 次数**。

`reverse_batch.cc:43-52` **候选 liveness cache**：

```cpp
dense_hashmap_t<u64, bool> candidate_liveness;
const auto candidate_live = [&](const RemotePtr& candidate) {
  const auto found = candidate_liveness.find(candidate.raw_address);
  if (found != candidate_liveness.end()) return found->second;
  const bool live = storage_owner_node_live(candidate);
  candidate_liveness.emplace(candidate.raw_address, live);
  return live;
};
```

注释 `reverse_batch.cc:39-42`：stage2 batch 常把同一个新候选带给多个 target，这个 cache 只做 early rejection（已知 dead 就跳过），**positive 结果仍要在最终锁内 revalidate**。

`reverse_batch.cc:54-60` **target 排序**：

```cpp
vec<u64> target_raws;
for (const auto& [target_raw, candidates] : updates) {
  target_raws.push_back(target_raw);
}
std::sort(target_raws.begin(), target_raws.end());
```

按 raw_address 排序处理 target——**降低跨 target 的 peer-read 局部性差异**（相邻 raw 往往在同一 cache line / 同一 RDMA segment）。

`reverse_batch.cc:67-132` **第一阶段：锁内取快照**。对每个 target：
- 去重候选 + liveness 过滤
- 上锁，检查 deleted
- 读 current_neighbors，select_fresh_reverse_candidates_locked
- 快速路径（不溢出）：直接 append 写回，unlock，continue
- 溢出：unlock，把 target + current + fresh 记入 `pending`，收集所有需要 snapshot 的 rptr

`reverse_batch.cc:134-154` **批量 snapshot**：

```cpp
std::sort(snapshots_needed.begin(), snapshots_needed.end(), ...);
snapshots_needed.erase(std::unique(snapshots_needed.begin(), snapshots_needed.end()), ...);
vec<NodeSnapshot> snapshots = read_node_snapshots_batched(snapshots_needed, config);
dense_hashmap_t<u64, size_t> snapshot_index;
for (size_t index = 0; index < snapshots.size(); ++index) {
  snapshot_index[snapshots[scope].rptr.raw_address] = index;
}
```

注释 `reverse_batch.cc:145-148`：

> Snapshot all vectors before reacquiring any target lock. The final locked pass only needs its narrow liveness-boundary header checks; bulk remote vector reads and alpha pruning stay outside the critical section.

**核心优化**：所有 peer-read 和 RobustPrune 都在锁外做，锁内只做 header 检查 + 写回。这极大缩短了锁持有时长。

`reverse_batch.cc:156-225` 的 `robust_prune_cached` lambda：用 cached snapshots 算距离，调 `select_alpha_robust_pruned_sorted`。`reverse_batch.cc:230-235` 对每个 pending target 跑一遍 prune，得到 `selected_neighbors`。

`reverse_batch.cc:237-272` **第二阶段：锁内 revalidation + 写回**：

```cpp
for (PendingReverseUpdate& update : pending) {
  lock_node(update.target);
  if (target_deleted) { unlock_node(update.target); continue; }
  const vec<RemotePtr> observed_neighbors = read_neighbor_list(update.target);
  if (!same_neighbors(observed_neighbors, update.current_neighbors)) {
    unlock_node(update.target);
    conflicted[update.target.raw_address] = std::move(update.candidates);
    continue;
  }
  vec<RemotePtr> fresh_candidates;
  select_fresh_reverse_candidates_locked(observed_neighbors, update.candidates,
    [this](const RemotePtr& candidate) { return storage_owner_node_live(candidate); },
    fresh_candidates);
  if (fresh_candidates.empty()) { unlock_node(update.target); continue; }
  if (!same_neighbors(fresh_candidates, update.candidates)) {
    robust_prune_cached(update.target, observed_neighbors, fresh_candidates, update.selected_neighbors);
  }
  write_neighbor_list(update.target, update.selected_neighbors);
  unlock_node(update.target);
}
```

跟 `apply_local_reverse_update` 的乐观锁协议相同：检查 current_neighbors 是否变了，变了就丢进 `conflicted` map；检查 fresh_candidates 是否一致，不一致就重算 prune；一致就直接写 cached 的 selected_neighbors。

`reverse_batch.cc:274-278` **冲突 fallback**：

```cpp
bool success = true;
for (const auto& [target_raw, candidates] : conflicted) {
  success &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
}
return success;
```

冲突的 target 用单 target 版本重试——它的内部循环会处理冲突。**两阶段（批量 + 单 target fallback）是性能与正确性的折中**：批量快但冲突时要重算，单 target 慢但冲突时自适应重试。

---

## 关键数据结构与流程图

### 数据结构关系

```
┌─────────────────────────────────────────────────────────────────┐
│ MemoryNode (storage owner 分片实例)                              │
│                                                                 │
│  index_buffer_  ──┬──> [0..8] free pointer (bump alloc)         │
│                   ├──> [8..16] global medoid (storage_id==0)    │
│                   ├──> static node region                       │
│                   │     └── VamanaNode records (idmap base)     │
│                   ├──> gpu_dynamic_node_base_                   │
│                   │     └── dynamic node region                 │
│                   │           └── bump-allocated by             │
│                   │               allocate_local_node           │
│                   └── StorageControlBlock                       │
│                         ├── dynamic_high_watermark              │
│                         ├── reclaim_pending_nodes               │
│                         └── reclaim_ack_sequences[clients]      │
│                                                                 │
│  base_idmap_        : id -> FreshnessEntry (immutable, offline) │
│  dynamic_freshness_shards_[kDynamicFreshnessShardCount]:        │
│      each shard:                                                │
│        mutex                                                    │
│        entries: id -> FreshnessEntry                            │
│        mutations_inflight: set<id>                              │
│                                                                 │
│  storage_owner_async_candidates_[thread.id][coroutine_id]:      │
│      vec<RemotePtr>  (beam search 输出 slot)                    │
│                                                                 │
│  StorageOwnerThread.coroutines[i]                               │
│    └── StorageOwnerCoroutineScratch                             │
│          ├── visited, beam, unvisited, batch  (search)          │
│          ├── prune_infos, scored_snapshots, selected (prune)    │
│          └── reverse_*  (reverse update)                        │
└─────────────────────────────────────────────────────────────────┘
```

### insert 主流程状态机

```
                    ┌──────────────────────┐
                    │ prepare_mutation     │
                    │ (inflight 占位 +     │
                    │  generation 分配)    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │ status == ok?               │
                ▼                             ▼
        ┌───────────────┐            ┌────────────────┐
        │ ok            │            │ failed/exists  │
        │               │            │ complete_seq   │
        │               │            │ co_return      │
        └──────┬────────┘            └────────────────┘
               │
       ┌───────┴───────┐
       │ kind == erase?│
       ▼               ▼
┌────────────┐   ┌────────────────────────────────────────┐
│ mark_node_ │   │ async_read_global_medoid               │
│ deleted    │   │                                        │
│ publish    │   │  ┌── null? ──────────────────────┐     │
│ schedule_  │   │  │                               │     │
│ maintenance│   │  ▼                               │     │
│ co_return  │   │ allocate_local_node              │     │
└────────────┘   │ write_new_node(neighbors={})     │     │
                 │ try_set_global_medoid            │     │
                 │  ┌─ success ──┐  ┌─ fail ──┐     │     │
                 │  │ publish    │  │ use     │     │     │
                 │  │ schedule   │  │ observed│     │     │
                 │  │ co_return  │  │ medoid  │     │     │
                 │  └────────────┘  └────┬────┘     │     │
                 └───────────────────────┼──────────┘     │
                                         │                 │
                                         ▼                 │
                ┌────────────────────────────────────────┐  │
                │ STAGE 1: beam_search_candidates_async  │  │
                │ (owner 分片内 width-L 构造搜索)         │  │
                │                                        │  │
                │  medoid snapshot -> beam               │  │
                │  loop:                                 │  │
                │    best unexpanded                     │  │
                │    read_neighbor_list                  │  │
                │    batch read_node_snapshots           │  │
                │    insert_into_beam (width L)          │  │
                │  until no unexpanded                   │  │
                │  output -> storage_owner_async_        │  │
                │             candidates_[thread][coro]  │  │
                └───────────────────┬────────────────────┘  │
                                    │                       │
                                    ▼                       │
                ┌────────────────────────────────────────┐  │
                │ RobustPrune (alpha)                    │  │
                │ source = new vector                    │  │
                │ candidates = beam                      │  │
                │ -> selected_neighbors (<= R)           │  │
                └───────────────────┬────────────────────┘  │
                                    │                       │
                                    ▼                       │
                ┌────────────────────────────────────────┐  │
                │ allocate_local_node (new RemotePtr)    │  │
                │ write_new_node(id, vector, neighbors,  │  │
                │                generation)             │  │
                │ upsert? mark old deleted               │  │
                │ publish_mutation(live, new generation) │  │
                └───────────────────┬────────────────────┘  │
                                    │                       │
                       ┌────────────┴──────────────┐        │
                       │ maintenance_enabled?      │        │
                       ▼                           ▼        │
              ┌─────────────────┐        ┌────────────────┐ │
              │ schedule_       │        │ immediate:     │ │
              │ maintenance     │        │ local_updates  │ │
              │ (stage2 async,  │        │   (local       │ │
              │  第26课)        │        │    neighbors)  │ │
              │                 │        │ remote_updates │ │
              │ stage2 worker:  │        │   (cross-shard │ │
              │  - cross-shard  │        │    via RPC     │ │
              │    candidate    │        │    第24课)     │ │
              │    merge        │        └────────────────┘ │
              │  - reverse_batch│                           │
              │    apply        │                           │
              │  - finalize     │                           │
              └─────────────────┘                           │
```

### 反向边应用流程（`apply_local_reverse_updates_batched`）

```
updates: {target_raw -> [candidates]}

Phase 1: 锁内取快照 (per target)
  for target in sorted(updates):
    lock(target)
    if deleted: unlock, skip
    current = read_neighbor_list(target)
    fresh = select_fresh(current, candidates, is_live)
    if fresh.empty: unlock, skip
    if |current| + |fresh| <= R:        # 快速路径
      write_neighbor_list(target, current + fresh)
      unlock, continue
    unlock(target)
    pending.push({target, current, fresh})
    snapshots_needed += current + fresh

Phase 2: 批量 snapshot (锁外)
  dedup(snapshots_needed)
  snapshots = read_node_snapshots_batched(snapshots_needed)
  build snapshot_index: raw -> idx

Phase 3: 批量 prune (锁外)
  for update in pending:
    robust_prune_cached(target, current, fresh)
      -> update.selected_neighbors

Phase 4: 锁内 revalidation + 写回 (per target)
  for update in pending:
    lock(target)
    if deleted: unlock, skip
    observed = read_neighbor_list(target)
    if observed != current:                  # 冲突
      unlock, conflicted[target] = candidates, continue
    revalidated = select_fresh(observed, candidates, is_live)
    if revalidated.empty: unlock, skip
    if revalidated != fresh:                 # 候选变了
      robust_prune_cached(target, observed, revalidated)
    write_neighbor_list(target, selected)
    unlock(target)

Phase 5: 冲突 fallback
  for target, candidates in conflicted:
    apply_local_reverse_update(target, candidates)   # 单 target 重试
```

### stage1 / stage2 与 finalized watermark 的时间线

```
t0  prepare_mutation (inflight 占位)
t1  stage1: beam_search + RobustPrune + write_new_node
t2  publish_mutation (freshness map 可见, generation=g)
t3  schedule_storage_owner_maintenance (入队 stage2)
    ---- insert 协程返回, job.ok=true ----
t4  batch_execution 收集 local_updates / remote_updates
t5  apply_partition_local_reverse_update (本地反向边立即应用)
t6  send_reverse_update_batch (跨分片 RPC, 第24课)
    ---- 调用方拿到 MutationResult ----
    ---- compute 侧 owner 更新 (第28课) 看到 generation=g ----

stage2 (异步, maintenance worker, 第26课):
t7  跨分片候选合并 (其他分片 beam_search)
t8  reverse_batch apply (反向边批量应用)
t9  finalized watermark 推进 (RCU 回收旧节点元数据)
t10 complete_storage_owner_maintenance_sequence
```

**关键点**：insert 在 t3 就对调用方返回 ok，但 stage2 反向边要等 t7-t10 才完成。期间查询能看到新节点（freshness map t2 已发布），但新节点的反向边（邻居指向新节点）可能还没建好——这就是 `two_stage_insert_oracle.hh` 注释说的“quiescent equivalence + eventual cleanup”：静态快照下两阶段等价，并发下靠最终清理收敛。

---

## 与其他模块的关系

- **与第 6 课（Vamana 图格式）**：本课所有 `VamanaNode::xxx` 调用（`offset_id`、`offset_generation`、`hot_graph_entry_offset`、`encode_hot_graph_entry`、`decode_hot_graph_entry`、`HEADER_DELETED`、`HEADER_NODE_LOCK`、`HOT_GRAPH_DELETED`、`R`、`DIM`、`vector_bytes`、`allocation_size`）都来自 `src/vamana/vamana_node.hh`。`StorageLayoutResolver`（`src/vamana/storage_layout_resolver.hh`）把 `RemotePtr` 解析成 header/vector/neighbor_read/neighbor_slots 等子区域。`vamana::hot_graph::checksum16`/`store_u16_le`/`load_u16_le` 是紧凑图 entry 的 checksum 工具。

- **与第 7 课（schema-15 索引格式）**：`StorageControlBlock`（`src/gpu_search/index_format.hh`）的 `dynamic_high_watermark`、`reclaim_pending_nodes`、`reclaim_ack_sequences`、`compute_client_count` 字段在 `allocation.cc` 直接原子读写。`gpu_dynamic_node_base_` 和 `gpu_storage_control_offset_` 是 schema-15 在 `MemoryNode::init` 时算出的分区偏移（`src/memory_node/memory_node.cc:72-73`）。

- **与第 8 课（存储协议）**：`service::storage_owner::MutationKind`（insert/upsert/erase）、`MutationStatus`（ok/failed/already_exists/not_found/already_deleted）、`ReverseUpdateOp`、`MutationResult` 都来自 `src/service/storage_owner_protocol.hh`。`prepare_mutation` 的状态机就是协议规范的实现。

- **与第 16 课（存储回收 RCU）**：`minimum_compute_reclaim_ack` 是 RCU 回收的入口；`retire_local_dynamic_node` 当前不真正回收（注释 `allocation.cc:11-15` 解释原因），只清零计数。compute 侧的 `reclaim_ack_sequences` 推进见第 28 课。

- **与第 23 课（storage_owner_state）**：`StorageOwnerThread`、`StorageOwnerCoroutineScratch`、`BeamEntry`、`NodeSnapshot`、`StorageOwnerPruneCandidateInfo`、`StorageOwnerScoredSnapshot` 都定义在 `src/memory_node/storage_owner_state.hh`。`StorageOwnerCoroutineScratch` 的 `clear_search`/`clear_prune`/`clear_reverse_update` 方法让协程复用 buffer。`storage_owner_async_candidates_` 的二维 slot（`thread.id × coroutine_id`）在 `memory_node.hh:700` 声明。

- **与第 24 课（peer RPC）**：`remote_read_bytes`/`remote_write_bytes`/`remote_compare_and_swap` 是 peer RPC 的低层接口（第 4-5 课 RDMA 库）。`post_peer_read_async` + `poll_peer_send_cq` 是 async 版本。`send_reverse_update_batch` 把 `remote_updates` 发给其他分片。`try_lock_remote_header` 是远端节点锁。`reverse_batch.cc` 的批量 snapshot 和 `graph_mutation.cc` 的嵌套协程都依赖这些 RPC 原语。

- **与第 26 课（maintenance stage2）**：`schedule_storage_owner_maintenance` 把 stage2 任务入队（`src/memory_node/storage_owner_maintenance/queue.cc:166`）。stage2 worker（`src/memory_node/storage_owner_maintenance/worker.cc`）在 t7-t10 执行：跨分片候选合并、`apply_local_reverse_updates_batched`（`worker.cc:850`）、`send_reverse_update_batch`、`complete_storage_owner_maintenance_sequence`。`robust_prune_snapshots_cpu` 被 stage2 worker 直接调用（`worker.cc:820` 附近）。

- **与第 28 课（compute 侧 storage owner 更新）**：`publish_mutation` 写入 `dynamic_freshness_shards_` 后，compute 侧通过 owner map RPC（第 24 课）拉取增量，用 generation 判断是否见过。`job.invalidated_neighbors` 让 compute 侧 invalidate 缓存的邻居表。

- **与第 30 课（breakdown benchmark）**：`InsertBreakdownCounters` 的 `storage_owner_medoid_ns`/`storage_owner_search_*_ns`/`storage_owner_prune_*_ns`/`storage_owner_write_node_ns`/`storage_owner_local_reverse_ns`/`storage_owner_remote_reverse_ns` 字段在 `candidate_search.cc`、`graph_mutation.cc`、`batch_execution.cc` 各处用 `elapsed_ns_since` 累积，是写入路径性能剖析的核心。

---

## 小结

本课覆盖了存储侧索引访问与图修改的完整链条：

1. **空间分配**（`allocation.cc`）：bump allocator + `dynamic_high_watermark`，schema-15 物理地址 generation-stable 不复用，避免跨分片 RPC 错乱。
2. **紧凑图访问**（`graph_access.cc`）：节点快照读（header+vector，不含邻居）、邻居表读（带 checksum 重试）、节点写（read-modify-write + checksum 重算）、节点锁（本地原子 CAS / 远端单字节 write）。所有读写都区分本地/远端两条路径。
3. **构造搜索**（`candidate_search.cc`）：sync 版 `beam_search_candidates` + async 协程版 `beam_search_candidates_async` + partition-local 版。beam 宽度 L，无独立深度上限，靠 L 隐式收敛。
4. **图修改主流程**（`graph_mutation.cc`）：insert/upsert/delete 协程，medoid 空特判、stage1 beam search、RobustPrune、write_new_node、publish_mutation、schedule maintenance。反向边应用有 partition-local（快速）、单 target（带冲突重试）两版。
5. **批量反向边**（`reverse_batch.cc`）：两阶段（锁内取快照 → 锁外批量 prune → 锁内 revalidation 写回 → 冲突 fallback），把 peer-read 和 prune 移出锁外，缩短临界区。
6. **算法层**（`robust_prune_policy.hh`、`partition_local_search.hh`、`two_stage_insert_oracle.hh`、`reverse_batch_policy.hh`）：与 I/O 解耦的纯算法模板，`select_alpha_robust_pruned_sorted` 被 4 处复用，`two_stage_insert_oracle` 是 direct vs two-stage 等价性的参考实现。

核心设计思想：**写入路径走 CPU 协程 + 跨分片 RDMA，查询路径走 GPU**。两阶段插入（stage1 owner 分片本地 + stage2 跨分片异步）让 insert 快速返回，反向边和跨分片候选合并由 maintenance worker 异步收敛。所有图修改都通过“锁内 liveness 校验 + checksum 自校验 + generation 单调递增”保证并发安全，schema-15 物理地址不复用避免跨分片 RPC 的 generation 携带开销。这是 dvstor 在“GPU 中心化存算分离”架构下对写入路径的核心权衡。
