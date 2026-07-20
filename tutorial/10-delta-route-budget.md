# 第 10 课：Delta 索引、动态路由 overlay 与预算

> 本课是 Part III "GPU 搜索引擎" 的第二课。第 9 课把 GPU 类型、遥测、PQ 模型铺开
> 之后，本课进入"计算侧如何在静态 base 图之上叠加一层可变层"——这是 dvstor 实现
> 存算分离下"低延迟可见性"的核心机制。本课围绕四个相互独立但语义耦合的小模块展开：
>
> - `DeltaCoordinator`：CPU 侧的 delta 索引协调器，记录 mutation 的代际与发布次序
> - `DynamicRouteOverlayDiff`：把存储侧的 canonical 路由快照换算成 GPU 槽位更新
> - `memory_budget`：启动前对 GPU 显存做显式核算，决定 delta/resident PQ/cache 的容量
> - `NavigationBootstrapper`：用 RDMA 把 base PQ code 与 anchor 兜底层拉到 GPU
>
> 它们合在一起回答一个问题：**在 immutable base 索引之外，GPU 上要驻留哪些可变结
> 构，它们各占多少显存，谁在什么 epoch 下让它们对查询可见。**

---

## 本课目标与涉及文件

学完本课你应该能够：

1. 说清 `DeltaCoordinator` 内部 `delta_` / `versions_` / `durable_candidates_` 三张
   表的职责，以及 `reserve_epoch → publish_metadata → publish_barrier → retire_durable`
   这条生命周期对应的发布与回收语义。
2. 描述 `DynamicRouteOverlayDiff` 如何把 `AdaptiveRouteTable::RouteSlotSnapshot`
   折叠为 `DynamicRouteUpdate` 列表，并理解 seqlock 视窗约束
   `dynamic_route_window_stable`。
3. 复述 `memory_budget::estimate` 的输入/输出字段，并能把它逐字段映射到
   `[gpu-search] navigation budget ...` 日志上。
4. 解释 `NavigationBootstrapper` 为什么是一份"只做 RDMA read/write 的瘦适配器"，
   以及它和 `synthesized navigation manifest in memory` 日志的关系。
5. 画出 GPU 内存上"静态 anchors + 动态 8 槽路由 + delta overlay + resident PQ"
   的布局图与可见性时序图。

涉及文件（均为绝对路径，行号以 HEAD 为准）：

| 文件 | 行数 | 作用 |
|---|---|---|
| `src/gpu_search/delta_index.hh` | 1–97 | `DeltaMutation` / `VersionEntry` / `DeltaCoordinator` 声明 |
| `src/gpu_search/delta_index.cc` | 1–149 | 上述结构的实现 |
| `src/gpu_search/dynamic_route_overlay.hh` | 1–42 | `DynamicRouteOverlayDiff` 声明 |
| `src/gpu_search/dynamic_route_overlay.cc` | 1–99 | diff 计算 + compute-side mirror 提交 |
| `src/gpu_search/dynamic_route_consistency.hh` | 1–25 | seqlock 视窗谓词 |
| `src/gpu_search/delta_scan_budget.hh` | 1–36 | delta 扫描预算常量与分段 |
| `src/gpu_search/initial_seed_budget.hh` | 1–17 | 初始 seed 预算的 min 裁剪 |
| `src/gpu_search/memory_budget.hh` | 1–212 | 显式 GPU 显存核算 |
| `src/gpu_search/navigation_bootstrapper.hh` | 1–54 | `NavigationBootstrapper` 接口 |
| `src/gpu_search/navigation_bootstrapper.cc` | 1–268 | RDMA read/write 实现 |

辅助引用文件：

- `src/gpu_search/types.hh:75–115`（`DynamicRouteUpdate` / `DeviceDynamicRouteSlot`）
- `src/gpu_search/persistent_kernel.hh:20–70`（`kDeltaHandleMask` / `DeviceShardRegion` / `DeviceDeltaRecord`）
- `src/vamana/adaptive_route_table.hh:40–113`（`RouteSlotSnapshot` / `kSlotsPerShard`）
- `src/gpu_search/persistent_engine/construction.cc:80–360`（预算核算主流程）

---

## 1. `DeltaCoordinator`：CPU 侧 delta 索引协调器

### 1.1 数据模型：`DeltaMutation` 与 `VersionEntry`

打开 `src/gpu_search/delta_index.hh:20`：

```cpp
struct DeltaMutation {
  node_t id{};
  service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
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

这是一条 mutation 在计算侧的完整描述。逐字段解释：

- `id`：向量在全局 `node_t` 空间里的 ID（见第 2 课）。
- `kind`：`MutationKind::insert` / `erase`，来自存储 owner 协议（见第 8 课、
  第 24 课）。注意第 34 行 `deleted = mutation.kind == ...::erase`——`erase` 在
  delta 层就是"软删除"，对应 `DeviceDeltaRecord` 里的 `kDeltaDeleted` 标志位（见
  `persistent_kernel.hh:27`）。
- `generation`：同一 `id` 的代际号。`publish_impl` 第 29–33 行会做单调性检查：
  如果调用方没填（`== 0`），就 `current_generation + 1`；如果填了但
  `<= current_generation`，**直接 `continue` 跳过**——这是"延迟到达的旧代"被静默
  丢弃的关键。
- `epoch`：发布次序。`reserve_epoch()` 给出，`publish_barrier()` 把它写到
  `published_epoch_`。查询路径会用 `published_epoch()` 当作"可见快照边界"（见
  `src/gpu_search/persistent_engine/query_execution.cc:48`）。
- `remote_node` / `old_remote_node`：把向量编码成 `RemotePtr` 的 raw 值。本课
  `dynamic_route_overlay.cc:50` 就用 `source.entry.memory_node()` 校验 shard。
- `anchor_hint`：插入时建议挂在哪个 anchor 桶下；GPU 侧的 `DeviceDeltaRecord` 也
  存了 `anchor_bucket`（`persistent_kernel.hh:67`），见第 19 课 RDMA cache。
- `maintenance_sequence`：存储侧维护流水号，决定何时可以"晋升为 durable"。第
  64–74 行用它建堆。
- `owner_storage`：这条 mutation 由哪个 storage owner 持久化。`durable_candidates_`
  按 owner 分桶（第 65–66 行），回收也按 owner 轮询（第 113–118 行）。
- `durable`：本条 mutation 是否已被存储侧确认落盘。`retire_durable` 在移出
  delta 前会把它置 `true`（第 134 行）。
- `vector`：原始浮点向量。**只在同步 GPU 上传路径上需要**；走
  `publish_metadata` 时它会被丢弃（见第 1.3 节）。
- `enqueued_at`：入队时间戳，供遥测/调试。

`VersionEntry`（第 35 行）则是 ID→代际视图的极简摘要：

```cpp
struct VersionEntry {
  u32 generation{};
  u64 epoch{};
  bool deleted{};
  bool in_delta{};
};
```

`in_delta` 表示"该 ID 是否还驻留在 delta 表里"。一旦 mutation 被
`retire_durable` 拿走，`versions_[id].in_delta` 被置为 `false`（第 138 行），但
`versions_[id]` 本身保留——这样后续同样的 ID 再来一次插入，仍能从
`current_generation + 1` 继续。

### 1.2 私有成员：三张表 + 两个原子计数器

`delta_index.hh:87` 列出私有成员：

```cpp
mutable std::shared_mutex state_mutex_;
std::unordered_map<node_t, DeltaMutation> delta_;
std::unordered_map<node_t, VersionEntry> versions_;
std::vector<DurableQueue> durable_candidates_;
size_t durable_owner_cursor_{};
std::atomic<u64> next_epoch_{1};
std::atomic<u64> published_epoch_{0};
```

三张表：

1. `delta_`：**当前还活着的 mutation**。key 是 `node_t`，value 是完整 `DeltaMutation`
   （可能含 raw vector，也可能只含 metadata）。
2. `versions_`：**所有出现过的 ID 的代际摘要**，即便 mutation 已经被回收也保留。
   这是"代际单调性"判定的依据。
3. `durable_candidates_`：按 `owner_storage` 分桶的最小堆，等待"被存储侧确认
   durable"的候选。每个桶是一个 `std::priority_queue<..., DurableCandidateGreater>`，
   `DurableCandidateGreater`（第 73 行）让 `maintenance_sequence` 小的优先出堆——
   即"先入队的先回收"。

两个原子计数器：

- `next_epoch_`：从 1 开始单调递增，`reserve_epoch()` 用 `fetch_add` 取号（第 7 行）。
  注意起始值是 1，而 0 是"非法 epoch"——`publish_impl` 第 22 行
  `epoch == 0` 直接返回 false，`publish_barrier` 第 81–83 行对 `epoch == 0` 抛异常。
  这是为了让 GPU 侧能用 `0` 表示"未初始化"。
- `published_epoch_`：当前已经发布完成的最大 epoch。查询路径
  `published_epoch()` 用 `acquire` 读（第 92 行），与 `publish_barrier` 的
  `release` 配对，构成跨线程的 happens-before。

`state_mutex_` 是 `shared_mutex`：`delta_size()` / `version()` 用 `shared_lock`
（第 96、101 行）做并发读，`publish_impl` / `retire_durable` 用 `unique_lock`
（第 23、108 行）做独占写。注意 `published_epoch_` 本身是 atomic，**不在锁内**
更新——查询路径不需要拿锁就能读到一致快照边界。

### 1.3 `publish` / `publish_metadata` / `publish_impl`

`delta_index.cc:10` 起三个 publish 入口：

```cpp
bool DeltaCoordinator::publish(std::vector<DeltaMutation> mutations, u64 epoch) {
  return publish_impl(std::span<DeltaMutation>{mutations}, epoch, true);
}

bool DeltaCoordinator::publish_metadata(
    std::span<DeltaMutation> mutations, u64 epoch) {
  return publish_impl(mutations, epoch, false);
}
```

两者区别在 `retain_vectors` 参数：

- `publish`（`retain_vectors=true`）：把完整 `DeltaMutation`（含 `vector`）搬进
  `delta_`，调用方放弃所有权。
- `publish_metadata`（`retain_vectors=false`）：只存 metadata 字段，**不保留 raw
  vector**。

文件头注释（`delta_index.hh:45–49`）解释了为什么需要这两种入口：

> The synchronous GPU publication path owns reusable per-RPC mutation buffers.
> Persist only coordinator metadata so those vector buffers remain with their
> RPC slot and can be reused without another allocation.

具体说：同步 GPU 上传路径已经在调 `publish_metadata` 之前把 `vector` 直接灌进
GPU delta 区（通过 `DeltaPublishDescriptor`，见 `persistent_kernel.hh:120` 与第
15 课），所以 coordinator 这边没必要再留一份拷贝。RPC slot 的 buffer 因此能被下
一个 RPC 复用，避免每条 mutation 触发一次堆分配。这是 dvstor 在热路径上反复出
现的"零额外拷贝"模式。

`publish_impl`（第 19 行）核心逻辑分四步：

```cpp
if (mutations.empty() || epoch == 0) return false;
std::unique_lock<std::shared_mutex> lock(state_mutex_);
for (DeltaMutation& mutation : mutations) {
  mutation.epoch = epoch;
  const auto current = versions_.find(mutation.id);
  const u32 current_generation = current == versions_.end()
                                   ? 0 : current->second.generation;
  if (mutation.generation == 0) {
    mutation.generation = current_generation + 1;
  } else if (mutation.generation <= current_generation) {
    continue;
  }
  ...
}
publish_barrier(epoch);
return true;
```

第一步：**epoch 写入 + 代际单调性检查**。代际号要么由调用方填（信任来源），要么
coordinator 自增。如果调用方填了一个比当前小的代际，直接 `continue` 跳过——这条
mutation 是"过期的重复发布"，丢弃即可。注意这里 `continue` 之后 `publish_barrier`
依然会被调用，意味着这次 publish 仍然推进 `published_epoch_`，只是 delta 表内容
不变。这是"幂等重放"的语义：重放方可以安全地把整批 mutations 重发，coordinator
保证最终状态一致。

第二步：**写 versions_ 表**（第 34–40 行）：

```cpp
const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
versions_[mutation.id] = VersionEntry{
  .generation = mutation.generation,
  .epoch = epoch,
  .deleted = deleted,
  .in_delta = true,
};
```

`in_delta = true` 表示这条记录现在在 delta 表里活跃。

第三步：**写 delta_ 表**，分 `retain_vectors` 两种情况（第 42–62 行）。注释
（第 46–48 行）再次强调：metadata-only 路径不能把 raw vector 留到 stage2 回收，
否则会卡住 RPC slot buffer 复用。

第四步：**入 durable_candidates_ 堆**（第 64–74 行）：

```cpp
if (stored.maintenance_sequence != 0) {
  if (durable_candidates_.size() <= stored.owner_storage) {
    durable_candidates_.resize(static_cast<size_t>(stored.owner_storage) + 1);
  }
  durable_candidates_[stored.owner_storage].push(DurableCandidate{...});
}
```

只有带 `maintenance_sequence` 的 mutation 才入堆（`== 0` 表示这条 mutation 还没
进入存储侧维护流水，没有"等待 durable"的资格）。`durable_candidates_` 按 owner
分桶，桶按需懒扩容——这避免了在 coordinator 构造时就要知道 owner 总数。

### 1.4 `publish_barrier`：发布可见性的关键

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

这是个经典的"单调推进 CAS"：只要 `epoch` 比当前值大就尝试 CAS，CAS 失败说明有
别的线程已经推到更大的值，重读后继续比。两个关键点：

1. **CAS 成功用 `release`**：与 `published_epoch()` 的 `acquire` 配对（第 92 行），
   保证查询线程读到新 `published_epoch_` 时，前面 `publish_impl` 在锁内对
   `delta_` / `versions_` 的写入也对它可见。注意锁内的写虽然由 `state_mutex_`
  保护，但读侧（`published_epoch()`）并不拿锁——它只读原子变量，可见性完全靠
   release/acquire。
2. **`current < epoch` 才推**：重放旧 epoch 不会回退。这就是为什么
   `publish_metadata` 在丢过期 mutation 后仍要调 `publish_barrier`——重放方可以
   把整批重发，coordinator 保证 `published_epoch_` 单调。

文件头注释（`delta_index.hh:51–53`）解释了"route-only"路径为什么也走
`publish_barrier`：

> Route-only GPU publications still need to become visible at one ordered query
> snapshot, but they must not create a synthetic mutation/delta row.

即：动态路由 overlay 的更新（见第 2 节）不产生 delta 行，但它的可见性也要由
`published_epoch_` 序列化，否则查询可能看到"新路由 + 旧 delta"的不一致组合。
`src/gpu_search/persistent_engine/storage_reclaim.cc:233,263` 是这条路径的调用点：
先 `reserve_epoch`，做完路由更新后 `publish_barrier`。

### 1.5 `retire_durable`：stage2 回收

`delta_index.cc:106` 的 `retire_durable` 是 delta 索引"自我清理"的入口。它的目标：
把存储侧已经 durable 的 mutation 从 delta 表里移走，返回给上层去做 stage2
晋升（即真正改写 base 图）。

签名：

```cpp
std::vector<DeltaMutation> retire_durable(
  std::span<const u64> durable_sequences,
  size_t max_items = std::numeric_limits<size_t>::max());
```

`durable_sequences` 是"每个 storage owner 当前已 durable 到哪个
`maintenance_sequence`"的数组，下标对应 `owner_storage`。`max_items` 是这次回收
的上限，避免一次回收把所有 owner 的堆都掏空。

实现要点（第 108–145 行）：

1. **owner 轮询**：`durable_owner_cursor_` 记录上次回收停在哪个 owner，本次从
   `(cursor + 1) % owner_count` 开始（第 115 行）。这避免"某个 owner 的堆总是被
   掏空、其他 owner 的堆总是被忽略"的饥饿。
2. **堆顶判定**：`candidates.top().maintenance_sequence <= durable_sequence` 才能
   出堆（第 122 行）。`<=` 而非 `<` 是因为 `maintenance_sequence` 是"包含
   性"的——存储侧说"durable 到 N"意味着 N 及之前的都已落盘。
3. **二次校验**（第 128–133 行）：出堆后再到 `delta_` 表里确认这条 mutation
   没有被同 ID 的新代覆盖。如果 `mutation.owner_storage != owner` 或者
   `generation / epoch / maintenance_sequence` 任意一个对不上，跳过——这是防止
   "出堆候选项已过期"的安全网。
4. **更新 versions_**（第 135–139 行）：把 `versions_[id].in_delta` 置 false。
   注意条件 `version->second.epoch <= mutation.epoch`——如果同 ID 已经有更新
   代（新代 `epoch` 更大），旧代的回收不应该把 `in_delta` 置 false（新代仍然
   在 delta 里）。
5. **从 delta_ 移除并返回**（第 140–141 行）：`std::move` 出来后 `erase` 掉表项。

返回的 `retired` 会进入 stage2 晋升流程（见第 15 课"增量发布"和第 16 课"存储
回收 RCU"）。`persistent_engine/impl.hh:157` 与 `storage_reclaim.cc:539` 是上层
调用点。

### 1.6 可见性 overlay 语义

把第 1.1–1.5 节合起来，`DeltaCoordinator` 提供的是一种 **overlay 可见性** 语义：

- base 索引（图 + PQ codes）是 **不可变** 的，由存储侧发布（见第 7、15 课）。
- delta 索引是 base 之上的 **薄层覆盖**：插入的新 ID 在 delta 里出现，删除的 ID
  在 delta 里被标 `deleted`。
- 查询时，GPU kernel 会先查 delta 命中（见第 19、20 课），miss 才回退到 base。
- "什么时候让一批 delta 对查询可见"由 `published_epoch_` 决定，查询快照边界取
  `published_epoch()`（见 `query_execution.cc:48,111`）。
- 一旦 mutation durable 且被 stage2 晋升进 base，就从 delta 移除
  （`retire_durable`）。因此 delta 大小不会随维护积累而无限增长——它只反映"已
  发布但还没落盘"的部分。

这条语义与第 15 课"增量发布"（GPU 侧 staging → promote overrides）和第 16 课
"存储回收 RCU"（base 图替换时保护读者）紧密耦合，详见对应课程。

---

## 2. `DynamicRouteOverlayDiff`：8 槽动态路由 overlay

### 2.1 设计动机：storage-canonical 路由

`src/gpu_search/types.hh:81` 的注释直击设计核心：

```cpp
// The dynamic query-route overlay is deliberately tiny and fixed-capacity.
// Static anchors remain the bootstrap/fallback. Storage owners publish the
// canonical live representatives, so every compute node installs identical
// slot identities even when mutations originate from different clients.
inline constexpr u32 kDynamicRouteSlotsPerShard = 8;
inline constexpr u32 kDynamicRouteLive = 1u;
```

关键点：

1. **固定小容量**：每 shard 8 个槽，整个 overlay 是
   `shard_count * 8` 条 `DeviceDynamicRouteSlot`（48 字节，见
   `types.hh:103–115`）。这是 GPU 上的常驻结构，必须小到可以塞进 SM 的 L1/L2
   热路径。
2. **静态 anchor 是兜底**：如果动态槽不可用（写入中、未初始化、被 invalidate），
   查询退回静态 anchor 表。这就是 `construction.cc:164` 日志里的
   `"query routing=storage-canonical adaptive routes +static recall fallback"`。
3. **storage-canonical**：路由槽的身份由存储 owner 统一发布，所有 compute 节点
   安装完全相同的槽位。这一点至关重要：如果每个 compute 节点各自维护路由，跨
   节点的查询会看到不同的入口选择，导致结果不稳定。`DynamicRouteOverlayDiff`
   的 `commit` 里对 `shard / remote_node` 的一致性检查（见 2.3）就是在保证这一
   点。

### 2.2 `DynamicRouteOverlayDiff` 类

`src/gpu_search/dynamic_route_overlay.hh:14`：

```cpp
// Converts the fixed storage-canonical route snapshot into the minimal set of
// GPU slot updates. prepare() is side-effect free; commit() advances the
// compute-side mirror only after the control CTA has acknowledged the command.
class DynamicRouteOverlayDiff {
public:
  explicit DynamicRouteOverlayDiff(u32 shard_count);

  u32 capacity() const { return static_cast<u32>(slots_.size()); }

  void prepare(
    span<const vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot> snapshot,
    u64 epoch,
    std::vector<DynamicRouteUpdate>& updates) const;

  void commit(std::span<const DynamicRouteUpdate> updates);

private:
  struct Slot {
    u64 epoch{};
    u64 remote_node{};
    u32 shard{};
    u32 id{};
    u32 generation{};
    u32 flags{};
  };

  u32 shard_count_{};
  std::vector<Slot> slots_;
};
```

这是 **计算侧的 mirror**，记录"上次提交到 GPU 的槽位状态"。注释强调
`prepare()` 无副作用——它只算 diff，不修改 mirror；`commit()` 才推进 mirror。
这种两段式设计让上层可以：

1. `prepare()` 算出需要更新的槽位列表；
2. 把列表通过 `DeltaPublishDescriptor` 推到 GPU 的 control CTA（见
   `persistent_kernel.hh:161` 的 `dynamic_route_updates` 字段）；
3. 等 control CTA ack（通过 `DeltaPublishCompletion`）；
4. `commit()` 推进本地 mirror。

如果在步骤 2–3 之间查询路径读到的是旧 mirror，没有问题——GPU 上的
`DeviceDynamicRouteSlot` 仍然是旧值，seqlock 保证查询看到的是一致的旧快照（见第
2.4 节）。

`slots_` 是按 `shard * kDynamicRouteSlotsPerShard + local` 一维展开的，构造函数
（`dynamic_route_overlay.cc:10`）顺便给每个 slot 预填 `shard` 字段：

```cpp
DynamicRouteOverlayDiff::DynamicRouteOverlayDiff(u32 shard_count)
    : shard_count_(shard_count),
      slots_(static_cast<size_t>(shard_count) *
             kDynamicRouteSlotsPerShard) {
  if (shard_count == 0) {
    throw std::invalid_argument(
      "dynamic route overlay requires at least one shard");
  }
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (u32 local = 0; local < kDynamicRouteSlotsPerShard; ++local) {
      slots_[static_cast<size_t>(shard) * kDynamicRouteSlotsPerShard + local]
        .shard = shard;
    }
  }
}
```

文件顶部还有一条 `static_assert`（`dynamic_route_overlay.cc:7`）：

```cpp
static_assert(kDynamicRouteSlotsPerShard ==
              vamana::routing::AdaptiveRouteTable::kSlotsPerShard);
```

强制 GPU 侧的槽位数和存储侧 `AdaptiveRouteTable` 的槽位数一致（都是 8，见
`adaptive_route_table.hh:19`）。这是"storage-canonical"约束的编译期保证。

### 2.3 `prepare`：从 snapshot 算 diff

`dynamic_route_overlay.cc:26`：

```cpp
void DynamicRouteOverlayDiff::prepare(
    span<const vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot> snapshot,
    u64 epoch,
    std::vector<DynamicRouteUpdate>& updates) const {
  if (epoch == 0 || snapshot.size() != slots_.size()) {
    throw std::invalid_argument("invalid adaptive route snapshot for GPU overlay");
  }

  updates.clear();
  if (updates.capacity() < slots_.size()) {
    throw std::invalid_argument(
      "GPU dynamic route update buffer was not preallocated");
  }
  for (const auto& source : snapshot) {
    if (source.shard >= shard_count_ ||
        source.slot >= kDynamicRouteSlotsPerShard) {
      throw std::invalid_argument("adaptive route snapshot contains an invalid slot");
    }
    const u32 slot = source.shard * kDynamicRouteSlotsPerShard + source.slot;
    const Slot& current = slots_[slot];
    const u32 flags = source.live ? kDynamicRouteLive : 0u;
    const u64 remote_node = source.live ? source.entry.raw_address : 0u;
    if (source.live &&
        (remote_node == 0 ||
         source.entry.memory_node() != source.shard)) {
      throw std::invalid_argument("adaptive route snapshot contains an invalid live entry");
    }
    if (current.shard == source.shard && current.id == source.id &&
        current.generation == source.generation &&
        current.remote_node == remote_node && current.flags == flags) {
      continue;
    }
    updates.push_back(DynamicRouteUpdate{
      .epoch = epoch,
      .remote_node = remote_node,
      .slot = slot,
      .shard = source.shard,
      .id = source.id,
      .generation = source.generation,
      .flags = flags,
    });
  }
}
```

逐段解释：

1. **入参校验**（第 30–38 行）：`epoch != 0`、snapshot 大小必须等于 `slots_.size()`
   （即 `shard_count * 8`）、`updates` 容量必须预分配到 `slots_.size()`。最后一
   条是为了避免 `push_back` 触发 reallocation——这条路径在 maintenance tick 上
   跑，分配抖动会影响延迟尾部队列。
2. **逐槽映射**（第 39–44 行）：`RouteSlotSnapshot` 用 `(shard, slot)` 二维定位，
   `DynamicRouteUpdate` 用 `slot` 一维定位，这里做转换。`RouteSlotSnapshot` 的
   字段（`adaptive_route_table.hh:54`）包含 `shard / slot / initialized / live /
   id / generation / entry`，但 GPU 端只关心 `id / generation / remote_node /
   live`——centers 和 representative 向量留在存储侧，GPU 只拿"指针 + 元数据"。
3. **live 校验**（第 46–52 行）：`live` 槽必须有非零 `remote_node`，且
   `entry.memory_node() == shard`。后者是"storage-canonical"的核心约束：8 槽里
   第 i 个槽必须指向 shard i 的内存节点。如果违反，说明存储侧
   `AdaptiveRouteTable::observe` 或 `snapshot_route_slots` 有 bug，立即抛异常
   而非把错误数据送上 GPU。
4. **diff 跳过**（第 53–57 行）：如果 mirror 里这个槽的 `shard/id/generation/
   remote_node/flags` 全部相同，就 `continue` 不产生 update。这一步把
   "snapshot 没变化的槽"过滤掉，显著减少 GPU 上需要 atomic 写的槽数。
5. **构造 update**（第 58–66 行）：填充 `DynamicRouteUpdate`（结构定义见
   `types.hh:88`）。注意 `slot` 字段是一维下标，`shard` 是冗余字段（可由
   `slot / 8` 算出），但保留在结构里让 GPU 端的 control CTA 不必再算一次。

### 2.4 `commit`：推进 mirror

`dynamic_route_overlay.cc:70`：

```cpp
void DynamicRouteOverlayDiff::commit(
    std::span<const DynamicRouteUpdate> updates) {
  for (const DynamicRouteUpdate& update : updates) {
    if (update.slot >= slots_.size() || update.shard >= shard_count_ ||
        update.slot / kDynamicRouteSlotsPerShard != update.shard ||
        update.epoch == 0 || (update.flags & ~kDynamicRouteLive) != 0 ||
        ((update.flags & kDynamicRouteLive) != 0 &&
         (update.remote_node == 0 ||
          static_cast<u32>(update.remote_node >> 48) != update.shard))) {
      throw std::invalid_argument("invalid committed GPU dynamic route update");
    }
    const Slot& current = slots_[update.slot];
    if ((current.epoch != 0 && update.epoch <= current.epoch) ||
        (current.id == update.id &&
         current.generation > update.generation)) {
      throw std::invalid_argument("stale committed GPU dynamic route update");
    }
    slots_[update.slot] = Slot{
      .epoch = update.epoch,
      .remote_node = update.remote_node,
      .shard = update.shard,
      .id = update.id,
      .generation = update.generation,
      .flags = update.flags,
    };
  }
}
```

`commit` 的职责是"上层确认 GPU 已经 ack 后，推进本地 mirror"。两个关键校验：

1. **结构一致性**（第 73–80 行）：除了重复 prepare 的校验，还检查
   `update.slot / 8 == update.shard`——一维 slot 与二维 shard 必须自洽。另
   外 `static_cast<u32>(update.remote_node >> 48) != update.shard` 是
   `RemotePtr` 编码的 shard 字段（高 16 位）必须等于 `update.shard`，这是
   storage-canonical 的二次防御。
2. **时序单调**（第 81–86 行）：`update.epoch` 必须 `> current.epoch`（除非
   current 还未初始化，即 `epoch == 0`）；同 ID 的 `generation` 必须严格递增。
   违反则抛"stale"异常——这通常是上层调度出错的信号，宁可早爆也不静默吞掉。

通过校验后直接覆盖 `slots_[update.slot]`。注意这里没有原子操作——`commit` 由
维护线程独占调用，查询路径读的是 GPU 上的 `DeviceDynamicRouteSlot`
（`persistent_kernel.hh:163`），不读 mirror。

### 2.5 seqlock 视窗：`dynamic_route_consistency.hh`

`src/gpu_search/dynamic_route_consistency.hh:17` 是一个看起来很小但语义关键的
谓词：

```cpp
// A route read is usable only when the writer sequence stayed at the same
// even value across metadata validation *and* PQ scoring.  Keeping this tiny
// predicate host-testable makes it difficult to accidentally narrow the
// protected window back to metadata alone.
DVSTOR_ROUTE_HD inline constexpr bool dynamic_route_window_stable(
    u64 before, u64 after) {
  return (before & 1u) == 0 && before == after;
}
```

对应的 `DeviceDynamicRouteSlot`（`types.hh:103`）：

```cpp
// sequence is a device-scope seqlock.  The control CTA is the only writer:
// odd means an update is in progress, even means the remaining fields form a
// stable snapshot.  Query CTAs never wait for a writer; they skip an unstable
// dynamic seed and continue with the static route.
struct DeviceDynamicRouteSlot {
  u64 sequence{};
  u64 command_id{};
  u64 epoch{};
  u64 remote_node{};
  u32 id{};
  u32 generation{};
  u32 shard{};
  u32 flags{};
};
```

这是经典 seqlock 模式：

1. control CTA 写入前 `sequence++`（变奇数）；
2. 写 `command_id / epoch / remote_node / id / generation / shard / flags`；
3. 再 `sequence++`（变偶数）。

查询 CTA 读：

1. 读 `before = sequence`；
2. 读其它字段，做 metadata 校验（`shard` 合法、`generation` 在范围内等）；
3. 做 PQ 评分（用 `remote_node` 指向的 PQ code 算距离）；
4. 读 `after = sequence`；
5. 调 `dynamic_route_window_stable(before, after)`：仅当
   `before` 是偶数 **且** `before == after` 时，这次读才算数。

注释强调"across metadata validation *and* PQ scoring"——视窗必须覆盖整个评分
过程，而不是只覆盖 metadata 读。如果只覆盖 metadata，那么读 metadata 后、PQ 评分
前 control CTA 可能把 `remote_node` 改了，查询会用旧 metadata 配新指针算分，
得到错误距离。这个谓词留在 host 头文件且 `__host__ __device__`，就是为了让
host 侧的单测也能验证它，避免有人手滑把视窗缩回 metadata 阶段。

查询遇到 unstable 视窗怎么办？`types.hh:99–102` 注释说："Query CTAs never wait
for a writer; they skip an unstable dynamic seed and continue with the static
route." 即放弃这一条动态 seed，退回静态 anchor——这是"动态 overlay 是优化而非
正确性依赖"的体现。

### 2.6 与第 8/24/28 课的关系

`AdaptiveRouteTable` 的 `observe / invalidate / snapshot_route_slots` 在存储侧
运行（见第 8 课"元数据/owner map/存储协议"），决定 8 槽里到底放哪些 ID。compute
侧只通过 `RouteSlotSnapshot` 拿到一份只读快照。第 24 课"peer RPC"会讲 storage
owner 如何把这份快照推给 compute；第 28 课"计算侧 storage owner 更新"会讲
compute 收到推送后如何调 `prepare / commit` 把它落到 GPU。

---

## 3. 两个小工具：`delta_scan_budget.hh` 与 `initial_seed_budget.hh`

### 3.1 `delta_scan_budget.hh`

`src/gpu_search/delta_scan_budget.hh:12`：

```cpp
// Query-time delta injection is a short-lived visibility aid while stage2 is
// making a mutation reachable through the authoritative graph.  Its work must
// not grow with the maintenance backlog.  This is an algorithm constant (and
// deliberately not another deployment knob): the normal graph search and
// stage2 construction widths remain unchanged.
inline constexpr std::uint32_t kDeltaScanRecordBudget = 2048;
```

注释把设计意图说得非常清楚：**delta 扫描不是另一条搜索宽度，而是一个算法常
量**。查询时如果允许扫描整个 delta 表，那么 delta 越大（maintenance 积压越多）
查询越慢——这违背了"delta 只是 stage2 晋升前的临时可见性兜底"的语义。所以固定
2048 条上限，与配置无关。

`DeltaScanSegment`（第 14 行）和 `delta_scan_segment`（第 22 行）是分片工具：

```cpp
struct DeltaScanSegment {
  std::uint32_t offset{};
  std::uint32_t count{};
};

constexpr DeltaScanSegment delta_scan_segment(
    std::uint32_t index,
    std::uint32_t segment_count,
    std::uint32_t budget = kDeltaScanRecordBudget) {
  if (segment_count == 0 || index >= segment_count) return {};
  const std::uint32_t base = budget / segment_count;
  const std::uint32_t remainder = budget % segment_count;
  return DeltaScanSegment{
    .offset = index * base + (index < remainder ? index : remainder),
    .count = base + (index < remainder ? 1u : 0u),
  };
}
```

它把 `budget`（默认 2048）均分给 `segment_count` 个段，第 `i` 段拿到
`base + (i < remainder ? 1 : 0)` 条记录，offset 累加。例如 `segment_count=3,
budget=2048`：base=682, remainder=2，三段分别是 683/683/682 条。这是个常见的"
余数前缀分配"模式，保证总和不超 budget 且各段差不超过 1。

`__host__ __device__` 标记让它能在 GPU kernel 里直接 `constexpr` 求
值——查询 kernel 启动时各 CTA 用自己的 `index` 算出负责的 delta 区间。具体调用
见第 20 课"查询遍历主循环"。

### 3.2 `initial_seed_budget.hh`

`src/gpu_search/initial_seed_budget.hh:10`：

```cpp
constexpr std::uint32_t initial_seed_budget(
    std::uint32_t configured,
    std::uint32_t traversal_capacity) {
  return configured < traversal_capacity ? configured : traversal_capacity;
}
```

这是 `std::min` 的封装，但它单独成文件是为了语义：**初始 seed 数量受两个约束**
——配置项 `gpu_entry_seed_count`（用户想用多少个入口点）和 `traversal_capacity`
（beam 容量上限，由 `gpu_traversal_beam_width * graph_degree * 8` 等决定）。取
两者较小值，避免 seed 数量超过 beam 能容纳的容量。

`construction.cc:167` 日志里的 `seeds=...` 就走这个裁剪。这看似简单，但它把"用
户配置"与"算法容量"两个不同来源的约束在同一处统一，避免 kernel 启动时再做
隐式截断。

---

## 4. `memory_budget`：启动前显式核算 GPU 显存

### 4.1 为什么需要显式核算

dvstor 的 GPU 内存同时驻留多个独立结构：base PQ codes、resident PQ、delta overlay、
graph cache、exact cache、route graph、各种 scratch。如果"按需分配"，启动后才
在某次维护 push 时 OOM，会让系统进入无法恢复的状态。`memory_budget.hh` 的设计
是：**在启动 construction 之前，一次性把所有显式结构的容量算清楚，超预算直接
启动失败**。

`memory_budget.hh:12` 定义输入：

```cpp
struct Request {
  u64 nodes{};
  u64 max_delta_vectors{};
  u64 usable_bytes{};
  u64 requested_cache_bytes{};
  u64 requested_exact_cache_bytes{};
  u64 delta_budget_bytes{};
  u32 dim{};
  u32 pq_subquantizers{};
  u32 code_bytes{};
  u32 vector_bytes{};
  u32 query_slots{};
  u32 beam_width{};
  u32 graph_degree{};
  u32 exact_width{};
  u32 exact_record_bytes{};
  u32 anchor_count{};
  u32 shard_count{};
  u32 entry_point_count{};
  u32 cache_ways{4};
  u32 exact_cache_ways{4};
};
```

对应 `construction.cc:193` 的 `memory_budget::Request{...}` 字段填充。注意
`usable_bytes` 来自 `construction.cc:192`：

```cpp
const u64 engine_budget = static_cast<u64>(
  config.gpu_memory_limit_gb - config.gpu_memory_reserve_gb) << 30;
...
const u64 physically_available = free_gpu_bytes > runtime_reserve
  ? static_cast<u64>(free_gpu_bytes) - runtime_reserve : 0;
const u64 usable_budget = std::min(engine_budget, physically_available);
```

即 `usable_bytes = min(配置上限, 物理可用)`，其中配置上限是
`gpu_memory_limit_gb - gpu_memory_reserve_gb`（转字节），物理可用是
`cudaMemGetInfo` 报告的 free 减去运行时 reserve。两者取 min 保证既不超配置也不
超物理。

### 4.2 工具函数：`next_power_of_two` / `delta_footprint` / `choose_delta_capacity`

`memory_budget.hh:60`：

```cpp
inline u32 next_power_of_two(u64 value) {
  if (value >= (1u << 31)) return 1u << 31;
  return std::max<u32>(2, std::bit_ceil(static_cast<u32>(value)));
}
```

返回不小于 `value` 的最小 2 的幂，下限 2，上限 `1u<<31`（GPU 侧用 u32 索引，
2^31 是安全上限）。`std::bit_ceil` 是 C++20 起的标准库函数。

`delta_footprint`（第 65 行）算给定 `capacity` 的 delta 表占多少字节：

```cpp
inline u64 delta_footprint(u32 capacity, u32 vector_bytes, u32 code_bytes) {
  if (capacity == 0) return 0;
  const u32 table_capacity = next_power_of_two(static_cast<u64>(capacity) * 2);
  return static_cast<u64>(capacity) *
      (sizeof(DeviceDeltaRecord) + vector_bytes +
       code_bytes + 3 * sizeof(u32)) +
    static_cast<u64>(table_capacity) *
      (sizeof(u32) + sizeof(u64) + sizeof(u64) + sizeof(u32));
}
```

delta 区由两部分组成：

1. **记录数组**（`capacity` 条）：每条包含 `DeviceDeltaRecord`（24 字节，见
   `persistent_kernel.hh:58`）+ 原始向量（`vector_bytes`）+ PQ code
   （`code_bytes`）+ 3 个 u32（链表 `next/prev` + remote_position，对应
   `persistent_kernel.hh:137–139` 的 `delta_next/delta_prev/delta_remote_positions`）。
2. **hash table**（`table_capacity` 桶）：每桶含 `u32` head + `u64` key +
   `u64` epoch + `u32` slot，对应 `persistent_kernel.hh:148–150` 的
   `delta_remote_keys/delta_remote_slots` + `base_override_keys/epochs`。注意
   `table_capacity = next_power_of_two(capacity * 2)`——load factor 上限 50%，
   这是开放地址 hash table 的常见选择。

`choose_delta_capacity`（第 75 行）用二分搜索找最大 `capacity` 使
`delta_footprint(capacity, ...) <= budget`：

```cpp
inline u32 choose_delta_capacity(u64 budget, u64 max_vectors,
                                 u32 vector_bytes, u32 code_bytes) {
  u32 low = 1;
  u32 high = static_cast<u32>(std::min<u64>(
    std::min<u64>(max_vectors, kDeltaHandleMask),
    budget / std::max<u64>(1, vector_bytes)));
  if (high == 0) return 0;
  while (low < high) {
    const u32 middle = low + (high - low + 1) / 2;
    if (delta_footprint(middle, vector_bytes, code_bytes) <= budget) low = middle;
    else high = middle - 1;
  }
  return delta_footprint(low, vector_bytes, code_bytes) <= budget ? low : 0;
}
```

- `high` 上限是 `min(max_vectors, kDeltaHandleMask, budget / vector_bytes)`：
  `kDeltaHandleMask = 0x7fffffffu`（`persistent_kernel.hh:26`），因为 GPU 侧用
  u31 作为 delta handle 索引（最高位 `kDeltaHandleBit` 是标志位，区分 delta vs
  base）。
- `low` 起始为 1，二分向上找。注意 `middle = low + (high - low + 1) / 2` 是
  "上取整"二分，避免 `low = middle` 死循环。
- 找到 `low` 后再 sanity check 一次，不满足则返回 0（调用方会因此启动失败）。

`resident_pq_footprint` / `choose_resident_pq_capacity`（第 90、97 行）是
resident PQ 层的对应版本，结构类似但更简单（无 vector，无链表），只算 PQ code
+ hash table。`resident_pq` 是"热向量的 PQ code 常驻 GPU"层，避免每次查询都
走 RDMA 拉 PQ code（见第 19 课）。

### 4.3 `estimate`：主核算函数

`memory_budget.hh:112` 是核心函数。逐段讲：

**入参校验**（第 114–124 行）：

```cpp
if (request.nodes == 0 || request.nodes >= (1ull << 30) || request.dim == 0 ||
    request.pq_subquantizers == 0 || request.code_bytes == 0 ||
    request.vector_bytes == 0 ||
    request.code_bytes != request.pq_subquantizers ||
    request.dim % request.pq_subquantizers != 0 ||
    request.query_slots == 0 || request.beam_width == 0 ||
    request.graph_degree == 0 || request.exact_width == 0 ||
    request.exact_record_bytes == 0 || request.cache_ways == 0 ||
    request.exact_cache_ways == 0) {
  return result;
}
```

任意一项非法就返回空 `Result`（`fits` 默认 false）。注意
`code_bytes != pq_subquantizers` 和 `dim % pq_subquantizers != 0`——这两条把 PQ
编码约束（每子量化器 1 字节、dim 必须被子量化器数整除）做成了预算核算的硬前
提。

**delta 容量选择**（第 125–130 行）：

```cpp
result.delta_capacity = choose_delta_capacity(
  request.delta_budget_bytes, request.max_delta_vectors,
  request.vector_bytes, request.code_bytes);
if (result.delta_capacity == 0) return result;
result.delta_table_capacity = next_power_of_two(
  static_cast<u64>(result.delta_capacity) * 2);
```

delta 预算由配置 `delta_budget_mb` 决定（`construction.cc:199`）。如果预算装不
下任何一条 delta 记录，直接返回失败。

**visited 容量**（第 131–132 行）：

```cpp
result.visited_capacity = next_power_of_two(
  std::max<u32>(256, request.beam_width * request.graph_degree * 8));
```

visited bitmap 容量按 `beam * degree * 8` 算，下限 256，向上取 2 的幂。这是
查询遍历时"已访问节点"集合的容量（见第 18、20 课）。

**fixed bytes**（第 133–157 行）：把"必须有的"结构字节加起来：

```cpp
result.code_bytes = request.nodes * request.code_bytes;          // base PQ codes
result.delta_bytes = delta_footprint(...);                       // delta 区
result.delta_code_bytes = static_cast<u64>(result.delta_capacity) * request.code_bytes;
result.query_workspace_bytes = ...;                              // query 工作区
result.exact_bytes = ...;                                        // exact rerank 区
result.metadata_bytes = ...;                                     // shard/anchor/entry 元数据
result.permanent_override_bytes = ((request.nodes + 31) / 32) * sizeof(u32);
result.fixed_bytes = result.code_bytes + result.delta_bytes +
  result.query_workspace_bytes + result.exact_bytes + result.metadata_bytes +
  result.permanent_override_bytes;
if (result.fixed_bytes >= request.usable_bytes) return result;
```

逐项映射到 `construction.cc:326` 日志：

| `Result` 字段 | 日志 key | 含义 |
|---|---|---|
| `code_bytes` | `codes=` | base PQ codes 总字节 |
| `delta_bytes` | `delta=` | delta 区总字节（含 hash table） |
| `delta_capacity` | `delta_capacity=` | delta 表能装多少条 |
| `delta_code_bytes` | `delta_codes=` | delta 里 PQ code 部分字节 |
| `permanent_override_bytes` | `permanent_overrides=` | 永久 override bitmap |

`permanent_override_bytes` 用 bitmap 标记"base 节点已被 delta 覆盖，不要再从
base 读"，每 32 节点 1 个 u32（`persistent_kernel.hh:146`）。

`metadata_bytes`（第 144 行）是各种小结构的总和：

```cpp
result.metadata_bytes = static_cast<u64>(request.shard_count) *
    sizeof(DeviceShardRegion) +                                  // shard 元数据
  (static_cast<u64>(request.dim) * request.dim +
   static_cast<u64>(request.dim) * 256) * sizeof(f32) +          // opq matrix + pq centroids
  static_cast<u64>(request.entry_point_count) * sizeof(u32) +    // entry points
  static_cast<u64>(request.anchor_count) * request.dim * sizeof(f32) +  // anchor 向量
  static_cast<u64>(request.anchor_count) * sizeof(u32) +         // anchor handles
  static_cast<u64>(request.anchor_count) * request.code_bytes +  // anchor PQ codes
  (64ull << 20);                                                 // 64 MiB 杂项预留
```

最后一项 `64ull << 20`（64 MiB）是个固定的"杂项预留"，覆盖各种小结构（如
`stop / kernel_ready_count` 等控制标志），避免逐项核算的复杂度。

**cache 分配**（第 160–188 行）：剩下的空间分给 graph cache 和 exact cache。

```cpp
const u64 bytes_per_set = static_cast<u64>(request.cache_ways) *
    (kPersistentGraphCacheLineBytes + 3 * sizeof(u64) + 2 * sizeof(u32)) +
  sizeof(u32);
```

每个 cache set 含 `cache_ways` 路，每路 = cache line（512 字节，见
`persistent_kernel.hh:24`）+ 3 个 u64（key/generation/timestamp）+ 2 个 u32
（state/readers）+ 1 个 u32（set 头）。这是组相联 cache 的标准布局，详见第 19
课 RDMA cache。

cache 必须能容纳至少 `query_slots` 个 set（每个查询至少能缓存一个 set 的邻居），
否则返回失败（第 183 行）。`cache_budget` 取 `min(requested, available)`，保证
既满足用户配置上限，也不超可用空间。

**explicit bytes 与 fits 判定**（第 205–208 行）：

```cpp
result.explicit_bytes = result.fixed_bytes + result.cache_total_bytes +
  result.exact_cache_total_bytes;
result.fits = result.explicit_bytes <= request.usable_bytes;
return result;
```

`explicit_bytes` 是"显式核算的总和"，`fits` 是最终判定。注意 `explicit_bytes`
**不包括** `additional_scratch_bytes`——后者在 `construction.cc:280` 单独核算，
包括 dynamic_code_scratch / query_dispatch / direct_queue / graph_scratch /
cache_admission / route_graph 等"运行时 scratch"。`construction.cc:285` 检查
`additional_scratch_bytes > usable_budget - budget.explicit_bytes` 时同样抛异
常。最后 `construction.cc:306` 把所有显式 + scratch + resident_pq 加起来存到
`explicit_gpu_bytes`，对应遥测字段 `gpu_memory_explicit_bytes`（见第 9 课）。

### 4.4 resident PQ 容量校验

`construction.cc:289–301`：

```cpp
const u64 available_resident_pq_bytes =
  usable_budget - budget.explicit_bytes - additional_scratch_bytes;
const u64 requested_resident_pq_bytes =
  static_cast<u64>(config.gpu_resident_pq_budget_mb) << 20;
const u64 resident_pq_budget_bytes = std::min(
  requested_resident_pq_bytes, available_resident_pq_bytes);
resident_pq_capacity = memory_budget::choose_resident_pq_capacity(
  resident_pq_budget_bytes, kDeltaHandleMask, code_bytes);
if (resident_pq_capacity < delta_capacity) {
  throw std::runtime_error(
    "GPU resident dynamic-PQ budget is too small for the bounded update tier; "
    "increase --gpu-resident-pq-budget-mb or reduce --delta-budget-mb");
}
```

这里有个关键约束：**resident PQ 容量必须 ≥ delta 容量**。原因：每条 delta 记录
的 PQ code 都要能进 resident PQ 层（`DeviceDeltaRecord::resident_pq_slot`，
`persistent_kernel.hh:67`），否则查询命中 delta 时还得走 RDMA 拉 PQ code，违背
resident PQ 的设计目的。这条约束直接对应日志里的 `resident_pq_capacity=...`。

### 4.5 完整日志映射

`construction.cc:326–345` 的日志：

```cpp
std::cerr << "[gpu-search] navigation budget codes=" << budget.code_bytes
          << " delta=" << budget.delta_bytes
          << " delta_capacity=" << budget.delta_capacity
          << " delta_codes=" << budget.delta_code_bytes
          << " resident_pq=" << resident_pq_bytes
          << " resident_pq_capacity=" << resident_pq_capacity
          << " permanent_overrides=" << budget.permanent_override_bytes
          << " adjacency_total=" << budget.cache_total_bytes
          << " exact_cache_total=" << budget.exact_cache_total_bytes
          << " dynamic_code_scratch=" << dynamic_code_scratch_bytes
          << " dynamic_request_scratch=" << dynamic_request_scratch_bytes
          << " navigation_candidates=" << navigation_candidate_bytes
          << " direct_queue_scratch=" << direct_queue_bytes
          << " graph_scratch=" << graph_scratch_bytes
          << " cache_admission=" << cache_admission_bytes
          << " anchor_route=" << anchor_route_bytes
          << " dynamic_route=" << dynamic_route_bytes
          << " dynamic_route_codes=" << dynamic_route_code_bytes
          << " explicit=" << explicit_gpu_bytes
          << " limit=" << engine_budget << " bytes\n";
```

逐项映射：

| 日志 key | 来源 | 含义 |
|---|---|---|
| `codes` | `budget.code_bytes` | base PQ codes（`nodes * code_bytes`） |
| `delta` | `budget.delta_bytes` | delta 区总字节 |
| `delta_capacity` | `budget.delta_capacity` | delta 表能装多少条 |
| `delta_codes` | `budget.delta_code_bytes` | delta PQ code 部分 |
| `resident_pq` | `resident_pq_bytes` | resident PQ 区总字节 |
| `resident_pq_capacity` | `resident_pq_capacity` | resident PQ 能装多少条 |
| `permanent_overrides` | `budget.permanent_override_bytes` | 永久 override bitmap |
| `adjacency_total` | `budget.cache_total_bytes` | graph cache 总字节 |
| `exact_cache_total` | `budget.exact_cache_total_bytes` | exact rerank cache 总字节 |
| `dynamic_code_scratch` | `dynamic_code_scratch_bytes` | 动态 PQ code 评分 scratch |
| `dynamic_request_scratch` | `dynamic_request_scratch_bytes` | 动态 RDMA 请求 scratch |
| `navigation_candidates` | `navigation_candidate_bytes` | 候选合并缓冲 |
| `direct_queue_scratch` | `direct_queue_bytes` | direct RDMA 队列 scratch |
| `graph_scratch` | `graph_scratch_bytes` | 图遍历 scratch |
| `cache_admission` | `cache_admission_bytes` | cache admission filter |
| `anchor_route` | `anchor_route_bytes` | 静态 anchor 路由图 |
| `dynamic_route` | `dynamic_route_bytes` | 8 槽动态路由结构 |
| `dynamic_route_codes` | `dynamic_route_code_bytes` | 8 槽 PQ code |
| `explicit` | `explicit_gpu_bytes` | 显式总和 |
| `limit` | `engine_budget` | 配置上限（`limit - reserve`）|

启动时这行日志是排查"为什么启动失败"或"为什么查询慢"的第一手资料——所有显存项
一目了然。

---

## 5. `NavigationBootstrapper`：把 base PQ 与 anchor 拉到 GPU

### 5.1 类接口

`src/gpu_search/navigation_bootstrapper.hh:15` 定义两个 POD：

```cpp
struct NavigationRead {
  u64 remote_offset{};
  u64 destination_address{};
  u32 bytes{};
  u16 memory_node{};
};

struct NavigationWrite {
  u64 remote_offset{};
  u64 source_address{};
  u32 bytes{};
  u16 memory_node{};
};
```

每个 read/write 请求 = `(remote_offset, local_gpu_address, bytes, memory_node)`
四元组。`destination_address` / `source_address` 是 GPU 上的虚拟地址
（`void*` 转 `u64`），`remote_offset` 是远端 MR 内偏移，`memory_node` 是哪个
存储节点。

`NavigationBootstrapper` 类（第 29 行）：

```cpp
class NavigationBootstrapper {
public:
  NavigationBootstrapper(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions,
    void* gpu_destination_base,
    size_t gpu_destination_bytes);
  ~NavigationBootstrapper();

  NavigationBootstrapper(const NavigationBootstrapper&) = delete;
  NavigationBootstrapper& operator=(const NavigationBootstrapper) = delete;

  void read(std::span<const NavigationRead> requests,
            std::span<i32> statuses);
  void write(std::span<const NavigationWrite> requests,
             std::span<i32> statuses);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

经典的 PImpl 模式（见第 11 课持久化引擎 PImpl）。构造函数接受：

- `config`：索引配置（含 `gpu_rdma_qps` 等）。
- `channel_context` / `connection_manager`：RDMA 连接上下文（见第 4、5 课
  RDMA 传输库）。
- `remote_regions`：各存储节点的远端 MR token 列表。
- `gpu_destination_base` / `gpu_destination_bytes`：GPU 上目标缓冲区基址与大小。
  **所有 read/write 的 local 地址必须落在这个区间内**——`Impl::read` 第 84–89
  行会校验。

`read` / `write` 都是批量接口：一次提交多个请求，`statuses` 返回每个请求的结
果（1 成功，负数 errno）。

### 5.2 `Impl` 构造：每节点一组 DetachedQP

`navigation_bootstrapper.cc:38` 起 `Impl` 构造：

```cpp
struct NavigationBootstrapper::Impl {
  explicit Impl(const BootstrapContext& context)
      : config_(context.config), channel_context_(context.channel_context),
        connection_manager_(context.connection_manager), remote_regions_(context.remote_regions),
        data_context_(config_), gpu_region_(data_context_, context.gpu_destination_base,
                                           context.gpu_destination_bytes),
        gpu_base_(reinterpret_cast<u64>(context.gpu_destination_base)),
        gpu_bytes_(context.gpu_destination_bytes), next_qp_(remote_regions_.size(), 0) {
    const u32 qps_per_node = std::max<u32>(1, config_.gpu_rdma_qps);
    qps_.resize(remote_regions_.size());
    for (u32 node = 0; node < qps_.size(); ++node) {
      qps_[node].reserve(qps_per_node);
      for (u32 index = 0; index < qps_per_node; ++index) {
        auto qp = std::make_unique<DetachedQP>(data_context_);
        qp->connect(channel_context_, data_context_.get_lid(),
                    connection_manager_.server_qps[node]);
        qps_[node].push_back(std::move(qp));
      }
    }
    ...
  }
```

要点：

1. **`data_context_` 是独立的 RDMA Context**：不复用 `channel_context_`，因为
   bootstrap 走的是 GPUDirect RDMA 路径，需要绑到 GPU device 上。
2. **`gpu_region_` 是 GPU 缓冲区的 LocalMemoryRegion**：注册到 `data_context_`，
   提供 `lkey` 给 RDMA WR 用（第 102 行 `gpu_region_.get_lkey()`）。
3. **每节点 `gpu_rdma_qps` 个 QP**：`qps_` 是二维数组
   `[node][qp_index]`。多 QP 让多个 RDMA WR 并行（一个 QP 同一时刻只能有一个
   outstanding WR 在某些配置下），降低尾延迟。
4. **`next_qp_` 是 round-robin 游标**：每个节点一个 u32，下次选哪个 QP 用
   `next_qp_[node]++ % qps_per_node`（第 92 行）。

接着第 57–65 行探测 GPUDirect RDMA flush 能力：

```cpp
check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
           "cudaSetDevice(PQ bootstrap init)");
int flush_options = 0;
check_cuda(cudaDeviceGetAttribute(
             &flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions,
             static_cast<int>(config_.gpu_device)),
           "cudaDeviceGetAttribute(PQ bootstrap GPUDirect flush)");
flush_supported_ =
  (flush_options & cudaFlushGPUDirectRDMAWritesOptionHost) != 0;
```

`cudaFlushGPUDirectRDMAWrites` 的作用是：RDMA write 完成后，GPU 侧的缓存可能还
没刷到可见域，需要显式 flush 才能让 kernel 读到最新数据。read 路径在所有 WR
完成后调一次 flush（第 143–148 行）。write 路径不 flush——因为 write 是从 GPU
写到远端存储，不涉及 GPU 读自己的数据。

### 5.3 `read`：批量 RDMA read 到 GPU

`navigation_bootstrapper.cc:68`：

```cpp
void read(std::span<const NavigationRead> requests,
          std::span<i32> statuses) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  if (requests.size() != statuses.size()) {
    throw std::invalid_argument("PQ bootstrap status cardinality mismatch");
  }
  if (failed_) throw std::runtime_error("PQ bootstrap RDMA backend is unavailable");
  struct QpBatch {
    DetachedQP* qp{};
    std::vector<size_t> request_indices;
  };
  std::unordered_map<DetachedQP*, size_t> batch_by_qp;
  std::vector<QpBatch> batches;
  for (size_t i = 0; i < requests.size(); ++i) {
    const NavigationRead& request = requests[i];
    statuses[i] = -EINVAL;
    const u64 destination_offset = request.destination_address >= gpu_base_
      ? request.destination_address - gpu_base_ : gpu_bytes_;
    if (request.memory_node >= qps_.size() || request.bytes == 0 ||
        request.destination_address < gpu_base_ ||
        destination_offset > gpu_bytes_ || request.bytes > gpu_bytes_ - destination_offset) {
      continue;
    }
    auto& node_qps = qps_[request.memory_node];
    DetachedQP* qp = node_qps[next_qp_[request.memory_node]++ % node_qps.size()].get();
    auto [it, inserted] = batch_by_qp.emplace(qp, batches.size());
    if (inserted) batches.push_back(QpBatch{.qp = qp, .request_indices = {}});
    batches[it->second].request_indices.push_back(i);
  }
```

第一步是**按 QP 分批**。每个请求先做边界校验（`memory_node` 合法、`bytes > 0`、
local 地址落在 GPU 缓冲区内、不越界），然后用 round-robin 选一个 QP，按 QP 聚
合到 `batches`。边界校验失败的请求 `statuses[i]` 留 `-EINVAL`，不进入批次。

第二步**逐批 post_send**（第 98–106 行）：

```cpp
for (QpBatch& batch : batches) {
  for (size_t request_index : batch.request_indices) {
    const NavigationRead& request = requests[request_index];
    batch.qp->qp->post_send(
      request.destination_address, request.bytes, gpu_region_.get_lkey(),
      IBV_WR_RDMA_READ, true, false, remote_regions_[request.memory_node].get(),
      request.remote_offset, 0, request_index + 1);
  }
}
```

`post_send` 的参数（来自第 4 课 RDMA 传输库）：local 地址、字节数、lkey、WR 类型
（`IBV_WR_RDMA_READ`）、`solicited`（true）、`signaled`（false，最后一条由
batch 末尾的 poll 决定）、远端 MR、远端偏移、`wr_id = request_index + 1`。
`+1` 是为了让 `wr_id == 0` 保留给"无效"语义——poll 时 `completions[i].wr_id - 1`
还原回 request_index（第 130 行）。

第三步**逐批 poll CQ**（第 107–136 行）：

```cpp
std::vector<ibv_wc> completions(64);
for (QpBatch& batch : batches) {
  size_t remaining = batch.request_indices.size();
  const u32 timeout_ms = std::min<u32>(config_.storage_owner_rpc_timeout_ms, 1000);
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(timeout_ms);
  while (remaining > 0) {
    const i32 count = batch.qp->poll_send_cq(
      completions.data(), static_cast<i32>(std::min<size_t>(completions.size(), remaining)));
    if (count == 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        failed_ = true;
        throw std::runtime_error("PQ bootstrap RDMA read timed out");
      }
      std::this_thread::yield();
      continue;
    }
    if (count < 0) {
      failed_ = true;
      throw std::runtime_error("PQ bootstrap CQ polling failed");
    }
    remaining -= static_cast<size_t>(count);
    for (i32 i = 0; i < count; ++i) {
      const size_t request_index = static_cast<size_t>(completions[i].wr_id - 1);
      if (request_index < statuses.size()) {
        statuses[request_index] = completions[i].status == IBV_WC_SUCCESS ? 1 : -EIO;
      }
    }
  }
}
```

每批一个 deadline（取 `storage_owner_rpc_timeout_ms` 与 1000ms 的较小值），超时
就标记 `failed_ = true` 并抛异常——一旦失败，后续所有 read/write 都会立即抛
"backend unavailable"（第 74 行）。这是 fail-fast 设计：bootstrap 失败说明
RDMA 链路有问题，继续重试没意义。

`completions[i].status == IBV_WC_SUCCESS ? 1 : -EIO`——成功标 1，失败标
`-EIO`。`statuses` 数组最终所有项要么是 1（成功），要么是负数（EINVAL/EIO）。

最后第 137–148 行做 flush：

```cpp
thread_local int selected_gpu = -1;
if (selected_gpu != static_cast<int>(config_.gpu_device)) {
  check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
             "cudaSetDevice(PQ bootstrap fetch)");
  selected_gpu = static_cast<int>(config_.gpu_device);
}
if (flush_supported_) {
  check_cuda(cudaDeviceFlushGPUDirectRDMAWrites(
               cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
               cudaFlushGPUDirectRDMAWritesToOwner),
             "cudaDeviceFlushGPUDirectRDMAWrites(PQ bootstrap fetch)");
}
```

`thread_local selected_gpu` 缓存当前 thread 选中的 GPU，避免每次 read 都调
`cudaSetDevice`（虽然有副作用开销很小，但 hot path 上能省则省）。flush 把
GPUDirect RDMA 写入的数据刷到 GPU 可见域，确保后续 kernel 能读到。

### 5.4 `write`：从 GPU push 到远端

`navigation_bootstrapper.cc:151` 的 `write` 与 `read` 结构几乎相同，区别只在
WR 类型（`IBV_WR_RDMA_WRITE` vs `IBV_WR_RDMA_READ`）和 local 地址语义
（`source_address` vs `destination_address`）。write 路径不做 flush——因为
数据是 GPU 写出去，不需要 GPU 再读回来。

write 路径主要用于把 anchor 状态（`anchor_graph_states` /
`anchor_graph_readers`，见 `persistent_kernel.hh:172`）写回存储侧，或者在
maintenance 时把 GPU 上的 staging 数据同步回去。这部分详见第 15 课。

### 5.5 "synthesized navigation manifest in memory" 日志

`construction.cc:110`：

```cpp
std::cerr << "[gpu-search] synthesized navigation manifest in memory from metadata"
          << (used_anchor_entry_points ? " and anchors\n" : "\n");
```

这条日志对应的是 `construction.cc:101–109` 的 `format::synthesize_distributed_view`
调用——它把存储侧的 metadata（shard 布局、anchor 表、entry points）合并成一
份"内存中的 navigation manifest"，**不直接读 base 索引数据**。后续
`NavigationBootstrapper` 才用这份 manifest 知道"从哪个 memory_node 的哪个
offset 读多少字节"——即 manifest 描述"读什么"，bootstrapper 执行"怎么读"。

具体说，这份 manifest 包含：

1. **shard 布局**（`DeviceShardRegion`，`persistent_kernel.hh:36`）：每个 shard
   的 ordinal_base / node_count / node_stride / graph_base_offset /
   code_remote_offset 等——告诉 bootstrapper 从远端哪个偏移读 PQ code。
2. **anchor 表**（`anchor_table`，`construction.cc:139`）：anchor 的 raw pointer
   到 anchor bucket 的映射，以及 anchor 的 PQ code 与向量。
3. **entry points**（`entry_handles`，`construction.cc:163`）：查询的起点 ID 列表。
4. **PQ model**（`pq_model`，`construction.cc:112`）：PQ centroids + opq matrix，
   用于 GPU 上做 PQ 评分。

"synthesized" 强调这份 manifest 是**在内存里拼出来的**，不是从某个文件加载——
它来自存储侧推送的 metadata（见第 8、28 课）。这与第 11 课"持久化引擎 PImpl"
的生命周期紧密耦合：engine 构造时调 `synthesize_distributed_view` 得到 manifest，
然后 `NavigationBootstrapper` 用 manifest 把 base PQ code 与 anchor 兜底层拉
到 GPU，最后启动 persistent kernel（见第 17 课）。

---

## 6. 关键数据结构与流程图

### 6.1 GPU 内存布局图

```
┌─────────────────────────────────────────────────────────────────────┐
│ GPU 显存（按 construction.cc:347–359 偏移布局）                       │
├─────────────────────────────────────────────────────────────────────┤
│ codes (base PQ)         │ budget.code_bytes = nodes * code_bytes     │ ← NavigationBootstrapper.read 拉入
├─────────────────────────┼───────────────────────────────────────────┤
│ anchor_graph_records    │ anchor_graph_keys_host.size() * graph_entry_bytes
│  + anchor_graph_metadata│  + ... * (sizeof(u64) + 2*sizeof(u32))     │ ← 静态 anchor 路由图
├─────────────────────────┼───────────────────────────────────────────┤
│ dynamic_route_pq_codes  │ dynamic_route_capacity * code_bytes        │ ← 8 槽动态路由 PQ
├─────────────────────────┼───────────────────────────────────────────┤
│ dynamic_route_slots     │ dynamic_route_capacity * 48 (DeviceDynamicRouteSlot)
│                         │  = shard_count * 8 * 48                    │ ← seqlock 保护
├─────────────────────────┼───────────────────────────────────────────┤
│ exact rerank 区         │ budget.exact_bytes                         │
├─────────────────────────┼───────────────────────────────────────────┤
│ graph_scratch           │ graph_scratch_bytes                       │
├─────────────────────────┼───────────────────────────────────────────┤
│ exact_cache             │ exact_cache_bytes                         │
├─────────────────────────┼───────────────────────────────────────────┤
│ graph_cache (adjacency) │ graph_cache_bytes                         │
├─────────────────────────┼───────────────────────────────────────────┤
│ delta_records           │ delta_capacity * sizeof(DeviceDeltaRecord)
│ delta_vectors           │  + delta_capacity * vector_bytes          │ ← DeltaCoordinator 管理
│ delta_pq_codes          │  + delta_capacity * code_bytes            │
│ delta hash table        │  + delta_table_capacity * (u32+u64+u64+u32)
├─────────────────────────┼───────────────────────────────────────────┤
│ resident_pq_codes       │ resident_pq_capacity * code_bytes          │ ← resident PQ 兜底层
│ resident_pq hash table  │  + resident_pq_table_capacity * (u64+u32)  │   容量 ≥ delta_capacity
├─────────────────────────┼───────────────────────────────────────────┤
│ permanent_override_bits │ (nodes + 31) / 32 * sizeof(u32)            │ ← 永久 override bitmap
├─────────────────────────┼───────────────────────────────────────────┤
│ various scratch         │ dynamic_code / query_dispatch / direct_queue / ...
└─────────────────────────────────────────────────────────────────────┘
```

图中每个区域都对应 `construction.cc:326` 日志的一个 key。`delta_*` 与
`resident_pq_*` 是本课 `DeltaCoordinator` 与 `memory_budget` 协同决定大小的部分；
`dynamic_route_*` 是第 2 节 `DynamicRouteOverlayDiff` 维护的部分；
`codes` 与 `anchor_graph_*` 由 `NavigationBootstrapper` 拉入。

### 6.2 可见性时序图

```
时间 →

存储侧:   mutation 提交       storage owner 推送 RouteSlotSnapshot
              │                          │
              ▼                          ▼
计算侧:   reserve_epoch()            prepare(snapshot) → updates
              │                          │
              │ (同步 GPU 上传 delta)      │ (推 DeltaPublishDescriptor 给 control CTA)
              ▼                          ▼
              publish_metadata(mutations, epoch)
              │
              │  (delta_ / versions_ 更新, 持锁)
              ▼
              publish_barrier(epoch)
              │
              │  published_epoch_ CAS release
              ▼
              ────────── happens-before ──────────
              │
              ▼
查询路径:   snapshot_epoch = published_epoch()  (acquire)
              │
              │  GPU kernel 用此 snapshot_epoch 决定 delta 可见性
              │  + 读 DeviceDynamicRouteSlot（seqlock 视窗）
              ▼
            返回结果

后台:       storage owner 确认 durable
              │
              ▼
            retire_durable(durable_sequences)
              │
              │  delta_.erase, versions_[id].in_delta = false
              ▼
            返回 retired → stage2 晋升（见第 15 课）
```

关键约束：

- `publish_barrier` 的 `release` 与查询路径 `published_epoch()` 的 `acquire`
  构成 happens-before，保证查询读到的 delta 表内容至少新到 `snapshot_epoch`。
- 动态路由 overlay 的 `prepare` 不依赖 `publish_barrier`，但 `commit` 必须在
  `publish_barrier` 之后（见 `storage_reclaim.cc:233,263`）——即路由更新与 delta
  更新共享同一个 epoch 序列。
- `retire_durable` 在 stage2 晋升后调用，此时查询已经能看到 base 图里的新内
  容，delta 里的旧副本可以安全移除。

### 6.3 动态路由 overlay 时序

```
control CTA 写 DeviceDynamicRouteSlot:
  sequence++                  (奇数)
  command_id = ...
  epoch = ...
  remote_node = ...
  id = ...
  generation = ...
  shard = ...
  flags = ...
  sequence++                  (偶数)

query CTA 读:
  before = sequence
  读 command_id/epoch/remote_node/id/generation/shard/flags
  做 metadata 校验
  用 remote_node 做 PQ 评分
  after = sequence
  if dynamic_route_window_stable(before, after):
      使用这条动态 seed
  else:
      放弃，退回静态 anchor
```

`dynamic_route_window_stable` 强制视窗覆盖 metadata + PQ 评分全过程，保证不会
出现"metadata 是旧的、remote_node 是新的"错配。

---

## 7. 与其他模块的关系

- **第 2 课（公共类型与配置）**：`node_t`、`u32`/`u64` 等基础类型，以及
  `IndexConfiguration` 的 `gpu_memory_limit_gb / gpu_memory_reserve_gb /
  delta_budget_mb / gpu_resident_pq_budget_mb / gpu_rdma_qps` 等配置项。
- **第 4–5 课（RDMA 传输库）**：`NavigationBootstrapper` 用的
  `DetachedQP`、`Context`、`LocalMemoryRegion`、`MemoryRegionTokens`、
  `ClientConnectionManager` 都来自 RDMA 传输库。
- **第 7 课（schema-15 索引格式）**：`format::synthesize_distributed_view`
  解析存储侧 metadata 得到 shard 布局。
- **第 8 课（元数据/owner map/存储协议）**：`service::storage_owner::MutationKind`
  定义 `insert` / `erase`；`AdaptiveRouteTable` 维护 storage-canonical 路由。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`DynamicRouteUpdate` /
  `DeviceDynamicRouteSlot` / `DeviceDeltaRecord` / `DeviceShardRegion` 等
  GPU 结构；遥测字段 `gpu_memory_explicit_bytes` 等。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`PersistentSearchEngine` 持有
  `DeltaCoordinator delta_` 与 `NavigationBootstrapper`，本课是其"可变层"的
  实现细节。
- **第 14 课（查询执行/路由/完成）**：查询路径读 `delta_.published_epoch()`
  作快照边界，读 `DeviceDynamicRouteSlot` 取动态路由入口。
- **第 15 课（增量发布）**：`publish_metadata` 之后 GPU 侧 staging → promote
  overrides，把 delta 内容固化进 base override 层。
- **第 16 课（存储回收 RCU）**：`retire_durable` 返回的 retired mutations
  触发 base 图替换，RCU 保护替换期间的读者。
- **第 17 课（kernel 启动器/上下文/device ring）**：
  `DeltaPublishDescriptor` 通过 device ring 推给 control CTA。
- **第 19 课（RDMA cache）**：resident PQ 与 delta 的 PQ code 命中逻辑。
- **第 20 课（查询遍历主循环）**：`delta_scan_segment` 在 kernel 内分段扫
  delta；`dynamic_route_window_stable` 在 kernel 内判视窗。
- **第 22 课（GPUNetIO 传输/probe）**：`NavigationBootstrapper` 是
  GPUDirect RDMA 路径，与 GPUNetIO 传输互补（一个走 Verbs，一个走 DOCA）。
- **第 24 课（peer RPC）**：storage owner 通过 peer RPC 推送
  `RouteSlotSnapshot` 给 compute。
- **第 28 课（计算侧 storage owner 更新）**：compute 收到推送后调
  `DynamicRouteOverlayDiff::prepare / commit`。

---

## 8. 小结

本课讲了 dvstor 计算侧"在不可变 base 之上叠加可变层"的四个支柱：

1. **`DeltaCoordinator`** 用 `delta_ / versions_ / durable_candidates_` 三张
   表，配合 `reserve_epoch / publish_metadata / publish_barrier / retire_durable`
   四个接口，实现"代际单调 + epoch 序列化可见性 + durable 回收"的 overlay 语义。
   `publish_metadata` 走"零额外拷贝"路径，让 RPC slot buffer 可复用；
   `publish_barrier` 用 release/acquire 配对保证查询读到的快照与 delta 表内容
   一致。

2. **`DynamicRouteOverlayDiff`** 把存储侧 8 槽 snapshot 折叠成最小更新集，
   `prepare` 无副作用算 diff，`commit` 在 control CTA ack 后推进 mirror。
   GPU 侧的 `DeviceDynamicRouteSlot` 用 seqlock 让查询无等待地读，遇到不稳定
   视窗就退回静态 anchor。`dynamic_route_window_stable` 强制视窗覆盖 metadata
   + PQ 评分全过程，防止错配。

3. **`memory_budget`** 在启动前一次性核算所有显式结构（base PQ、delta、
   resident PQ、cache、route graph、scratch），超预算直接启动失败。
   `choose_delta_capacity` / `choose_resident_pq_capacity` 用二分搜索找最大
   容量；`estimate` 输出的每个字段都映射到 `[gpu-search] navigation budget`
   日志的一个 key。resident PQ 容量必须 ≥ delta 容量，是"delta 命中即 GPU 命中"
   的硬约束。

4. **`NavigationBootstrapper`** 是一份瘦 RDMA 适配器，用 DetachedQP 把
   base PQ code 与 anchor 兜底层从存储节点 GPUDirect RDMA 拉到 GPU。它执行
   `synthesize_distributed_view` 拼出的"navigation manifest"——manifest 描述
   "读什么"，bootstrapper 执行"怎么读"。

这四者合在一起，构成了 dvstor 查询路径的"可见性基础设施"：base 索引不可变，
delta overlay 提供低延迟可见性，动态路由 overlay 提供自适应入口，resident PQ
避免 RDMA 拉 code，memory budget 保证启动即确定显存边界。下一课（第 11 课）将
讲把这些组件装配起来的 `PersistentSearchEngine` 生命周期。
