# 第 12 课：引擎构造与资源分配（上）

## 本课目标

`PersistentSearchEngine::Impl` 的构造函数（`construction.cc:73`）是整个 GPU 中心化向量检索系统的"出生时刻"：它把第 2 课讲的 `IndexConfiguration`、第 7 课讲的 schema-15 索引格式、第 8 课讲的存储协议元数据、第 9 课讲的 PQ 模型、第 10 课讲的 `memory_budget` 预算核算、第 11 课讲的 PImpl 生命周期，全部"装配"成一块可以直接被持久化 CUDA Kernel（见第 17 课、第 21 课）和 GPUNetIO 传输层（见第 22 课）使用的 GPU 显存布局。

本课（第 12 课，上篇）覆盖 `construction.cc` 的前半部分，讲解到查询 scratch 分配完成为止——即"内存分配 + 引导"阶段结束、开始创建 GPUNetIO QP / owner 运行时之前的自然分界。具体包括：

1. 构造函数主体流程概览，与第 11 课生命周期的衔接。
2. `cudaSetDevice` / 设备绑定与构造期前置校验。
3. navigation manifest 在内存中的合成与一致性校验。
4. 显式预算核算（调用第 10 课 `memory_budget`），各数组 `cudaMalloc` 的顺序与尺寸来源：PQ base codes、resident PQ、delta（hash/bucket/override/remote slot）、mutable L0、scratch（query/OPQ out/LUT/beam/visited/route/dynamic route/direct queue/graph scratch）、结果区。
5. PQ code 批量 RDMA 引导（bootstrap window/windows，从远端 `.pq32.codes` 区间拉到最终 GPU 数组，payload 不经主机内存），首/中/尾抽样比对。
6. 静态 anchors PQ code 聚集（`launch_gather_anchor_codes`，连续兜底层），对应日志 `static_fallback_entries=...`。
7. navigation manifest 在 GPU 上的指针/偏移装配（远程缓冲区的分区偏移）。
8. 失败路径（任一分配/校验失败抛异常、已分配资源如何回滚）。
9. 本课结束时的 GPU 内存布局图。

第 13 课（下篇）覆盖后半：QP 装配、direct batch queue、kernel launch、control/delta CTA 资源、`PersistentKernelParams` 巨型结构体的填充、`start_persistent_kernel` 与三个工作线程的启动。

### 涉及文件

| 文件 | 角色 |
|------|------|
| `src/gpu_search/persistent_engine/construction.cc` | 本课主角，构造函数主体（1021 行） |
| `src/gpu_search/persistent_engine/impl.hh` | `Impl` 结构体字段定义（所有 `d_*` 指针） |
| `src/gpu_search/persistent_engine/cuda_helpers.hh` | `check_cuda` / `align_up` / `device_allocate` / `mapped_host_allocate` |
| `src/gpu_search/memory_budget.hh` | 预算核算 `estimate` / `choose_delta_capacity` / `choose_resident_pq_capacity` |
| `src/gpu_search/navigation_bootstrapper.cc` | CPU-posted GPUDirect RDMA 引导器（PQ code 拉取） |
| `src/gpu_search/persistent_engine/lifecycle.cc` | `stream_codes_to_gpu` / `stream_anchor_graph_to_gpu` 实现 |
| `src/gpu_search/index_format.hh` | `NavigationLayout` / `ShardRegion` / `StorageControlBlock` 字段 |
| `src/gpu_search/persistent_kernel.hh` | `kPersistentMax*` 常量、`gather_anchor_codes_kernel` 签名 |
| `src/gpu_search/persistent_kernel/runtime.cuh` | `gather_anchor_codes_kernel` 设备端实现 |

---

## 构造函数主体流程概览

构造函数签名（`construction.cc:73`）：

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
      delta_completions(8, MappedRing<CompletionPublishCompletion>::Direction::device_to_host) {
```

这里发生了几件事：

- `engine` 是对公开类 `PersistentSearchEngine` 的引用——后续所有 `engine.telemetry_.*` 的写入都走这条引用（见第 9 课遥测）。
- `config` 是外部 `IndexConfiguration` 的**引用**而非拷贝。这意味着构造期之后，引擎内部对 `config` 字段的读取仍然是外部同一份对象——调用方必须保证 `IndexConfiguration` 的生命周期不短于引擎。
- 成员初始化列表里初始化了 4 个 `MappedRing`：`submissions`/`completions` 是查询通路（容量 `gpu_query_slots * 2`，见第 3 课协程原语与第 11 课生命周期）；`delta_submissions`/`delta_completions` 是 delta 发布通路，容量固定为 8（delta 是低频控制通路，不需要查询那么深）。

`MappedRing` 在初始化列表里构造，说明它们不是延迟构造的——这与第 11 课 PImpl"构造即就绪"的设计一致。

构造函数主体从 `construction.cc:85` 的 `bind_cuda_device` 开始，到 `construction.cc:1018` 启动三个工作线程结束，全长约 930 行。按职责可以划分为以下阶段：

| 行号区间 | 阶段 | 本课/第 13 课 |
|----------|------|---------------|
| 85–97 | 设备绑定 + 客户端身份校验 + 容量上限校验 | 本课 |
| 99–137 | navigation manifest 合成 + 一致性校验 | 本课 |
| 139–181 | anchor table 装载 + 动态路由 overlay + free slot 池初始化 | 本课 |
| 183–345 | 预算核算（`memory_budget::estimate`）+ scratch 字节逐项累加 + 日志 | 本课 |
| 347–413 | 远程缓冲区布局（偏移对齐）+ GPUNetIO 传输层创建 + 控制块引导 + reclaim ACK 初始化 + 路由 publication 读取 | 本课 |
| 414–415 | PQ code 批量 RDMA 引导 + anchor graph 引导 | 本课 |
| 417–503 | shards/OPQ/PQ centroids/entry points H2D + anchor vectors/handles/PQ codes 聚集 + delta bucket heads | 本课 |
| 505–523 | 查询 scratch（queries/transformed/LUT/visited/candidates） | 本课（结尾） |
| 524–553 | dynamic code request scratch + query dispatch ring | **第 13 课起点**（QP/runtime 前的过渡） |
| 555–611 | direct batch queue / GPUNetIO owner runtime | 第 13 课 |
| 613–707 | graph cache / exact cache / admission filter | 第 13 课 |
| 709–723 | 结果区（host mapped） | 第 13 课 |
| 725–798 | delta records / resident PQ / dynamic route | 第 13 课 |
| 800–838 | stop flag / direct error flag / 4 条 CUDA stream | 第 13 课 |
| 839–862 | SM 数探测 + CTA 切分 | 第 13 课 |
| 864–1014 | `PersistentKernelParams` 巨型结构体填充 | 第 13 课 |
| 1015–1018 | `start_persistent_kernel` + 三个工作线程 | 第 13 课 |

本课讲到 **505–523 行的查询 scratch 分配**为止，第 13 课从 **524 行的 dynamic code request scratch** 起接。

---

## 1. 设备绑定与前置校验（`construction.cc:85–97`）

### 1.1 `bind_cuda_device`

```cpp
bind_cuda_device("cudaSetDevice(GPU navigation construction)");
```

`bind_cuda_device` 定义在 `health.cc:106`：

```cpp
void PersistentSearchEngine::Impl::bind_cuda_device(const char* operation) const {
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU navigation)");
  if (current_device != static_cast<int>(config.gpu_device)) {
    check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)), operation);
  }
}
```

它是一个**幂等的当前设备绑定**：先用 `cudaGetDevice` 查当前设备，只有在不一致时才 `cudaSetDevice`。这样每次进入引擎代码路径都不会无谓地切换上下文，避免在多引擎/多 GPU 场景下抖动。注意构造函数在这里只调用一次，但第 11 课生命周期里 `start_persistent_kernel`、`stop_persistent_kernel`、`clear_delta_device_state` 都会再次调用——因为构造结束后引擎代码可能在任意线程上跑，设备上下文不能假定保留。

> **关于 `cudaFree(0)` 预热**：题目要求中提到 `cudaFree(0)` 预热，但实际 `construction.cc` 中并未显式调用 `cudaFree(0)`。真正起"建立主上下文 + 设备绑定"作用的就是这里的 `cudaSetDevice`（经由 `bind_cuda_device`）以及后续 `cudaMemGetInfo`、`cudaMalloc`。`NavigationBootstrapper::Impl` 构造时（`navigation_bootstrapper.cc:57`）也会 `cudaSetDevice`。可以说"`cudaFree(0)` 预热"的等价效果由这些调用共同达成。请不要臆造不存在的代码。

### 1.2 客户端身份校验

```cpp
compute_client_id = connection_manager.client_id;
compute_client_count = connection_manager.num_total_clients;
if (compute_client_count == 0 ||
    compute_client_count > format::kMaxComputeClients ||
    compute_client_id >= compute_client_count) {
  throw std::runtime_error("compute client identity exceeds storage reclaim capacity");
}
```

dvstor 是存算分离架构（见第 1 课），多个计算客户端共享同一组存储节点。每个计算客户端在存储侧的 `StorageControlBlock::reclaim_ack_sequences[client_id]`（见第 8 课、第 16 课 RCU 回收）中占一个槽位，因此客户端 ID 必须 `< kMaxComputeClients`（`index_format.hh:28`，值为 64）。这是构造期最早的硬失败之一——身份不合法直接抛异常，不分配任何 GPU 资源。

### 1.3 容量上限校验

```cpp
if (config.gpu_traversal_beam_width > kPersistentMaxBeam ||
    config.gpu_final_rerank_width > kPersistentMaxExact ||
    config.R > kPersistentMaxGraphDegree) {
  throw std::invalid_argument("GPU navigation beam/exact/degree limit exceeded");
}
```

`kPersistentMaxBeam=128`、`kPersistentMaxExact=256`、`kPersistentMaxGraphDegree=128`（`persistent_kernel.hh:13–17`）。这些是 kernel 内部的定长数组上限（见第 18 课候选评分、第 20 课查询遍历主循环），用户配置一旦超出直接拒绝，避免 kernel 内越界。

---

## 2. navigation manifest 合成与一致性校验（`construction.cc:99–137`）

### 2.1 合成 distributed view

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
```

`format::synthesize_distributed_view` 是第 7 课 schema-15 索引格式的核心装配函数：它读取磁盘上的元数据头（`NavigationLayout`）、各 shard 的 `ShardRegion`、可选的 anchor 入口点，组装出 `format::View index`（`impl.hh:164` 的成员 `index`）。`SynthesisOptions::entry_points = 0` 表示不在此处合成随机入口点——入口点稍后由 `index.entry_points` 提供（`construction.cc:163` 的 `entry_handles = index.entry_points`）。`used_anchor_entry_points` 标志位决定日志措辞，反映是否用了 anchor sidecar 提供的入口点。

### 2.2 PQ 模型装载

```cpp
if (!pq::read_model(index_path::navigation_model_file(
      config.resolved_index_prefix(), index.layout.pq_subquantizers),
      pq_model, &load_error)) {
  throw std::runtime_error(load_error);
}
```

`pq::Model`（`impl.hh:165`）包含 OPQ 旋转矩阵 `rotation`、PQ 码本 `centroids`、子量化器数 `subquantizers`、每码位数 `bits_per_code`、`code_bytes()`。文件路径由 `index_path::navigation_model_file` 构造（见第 2 课配置）。`pq_subquantizers` 被用作模型文件名的区分维度，因为同一索引可能有多套 PQ 模型（见第 9 课 PQ 模型）。

### 2.3 一致性校验

```cpp
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

这是一段防御式编程：磁盘元数据（`index.layout`）、PQ 模型文件（`pq_model`）、运行时配置（`config`）、图节点格式（`VamanaNode`，见第 6 课 Vamana 图格式）必须两两一致。任何一项不匹配都说明索引构建期与运行期的版本/参数漂移，直接拒绝启动。关键校验包括：

- `num_shards == remote_regions.size()`：磁盘上的 shard 数必须等于实际连上的存储节点数（`remote_regions` 来自构造参数，见第 23 课存储节点主体）。
- `code_bytes == pq_model.code_bytes()`：PQ 码宽必须与模型匹配，否则 kernel 评分会错位。
- `model_checksum == pq_model.checksum()`：模型二进制级一致性校验。
- `graph_entry_bytes == VamanaNode::hot_graph_entry_size()`：图节点热区条目大小必须与 `VamanaNode` 编译期布局一致。
- `graph_shard_bits == VamanaNode::HOT_GRAPH_SHARD_BITS`：图内分片位数（见第 6 课）必须一致。

### 2.4 merge 候选容量校验

```cpp
const u64 max_merge_candidates =
  static_cast<u64>(config.gpu_traversal_beam_width) +
  static_cast<u64>(std::min(config.gpu_graph_prefetch_depth,
                            kPersistentScoreChunk)) * config.R;
if (max_merge_candidates > kPersistentMaxMergeCandidates) {
  throw std::invalid_argument("GPU navigation prefetch/degree exceeds parallel top-k capacity");
}
```

`max_merge_candidates` 是单次遍历扩展时可能并发的候选总数：当前 beam 宽度 + 预取深度 × 图度数。`kPersistentScoreChunk=16`（`persistent_kernel.hh:19`）是 kernel 单次评分块的容量上限；预取深度不会超过它。`kPersistentMaxMergeCandidates=2048` 是 kernel 内并行 top-k 缓冲区的定长上限（见第 18 课、第 20 课）。超出则拒绝。

---

## 3. anchor table 装载 + 动态路由 overlay + free slot 池（`construction.cc:139–181`）

### 3.1 `load_anchor_table`

```cpp
anchor_table = load_anchor_table(config.resolved_index_prefix(), config.dim,
                                 index.layout.num_shards, index);
```

`load_anchor_table` 是文件作用域内的辅助函数（`construction.cc:16–70`），读取磁盘上的 anchor sidecar（见第 6 课 anchor/idmap）。关键点：

```cpp
const filepath_t path = index_path::anchor_file(prefix);
std::ifstream input(path, std::ios::binary);
if (!input.good()) {
  std::cerr << "[gpu-search] warning: no anchor sidecar; large deltas use a full scan\n";
  return result;
}
```

如果 sidecar 不存在，只是警告返回空表——后续 delta 路由会退化到全量扫描，不是硬失败。

```cpp
vamana::anchor::Header header;
input.read(reinterpret_cast<char*>(&header), sizeof(header));
if (!input.good() || header.magic != vamana::anchor::kMagic ||
    header.version != vamana::anchor::kVersion || header.dim != expected_dim ||
    header.shard_count != expected_shards || header.total_anchors > (1u << 24)) {
  throw std::runtime_error("invalid anchor sidecar for GPU delta buckets: " + path.string());
}
```

sidecar 头部校验：magic、version、dim、shard count，以及 `total_anchors <= 16M`（防磁盘损坏导致的离谱值）。每个 anchor 条目里存的是 `RemotePtr`（远端节点指针），通过 `format::remote_to_ordinal(index_view, RemotePtr{entry.rptr_raw}, handle)` 反查到本地 ordinal——如果反查失败说明 anchor 指向了一个非静态 GPU 入口（动态节点），不允许：

```cpp
u32 handle = UINT32_MAX;
if (!format::remote_to_ordinal(index_view, RemotePtr{entry.rptr_raw}, handle)) {
  throw std::runtime_error("anchor sidecar contains a non-static GPU entry point");
}
```

`AnchorTable` 字段（`impl.hh:51`）：

```cpp
struct AnchorTable {
  u32 dim{};
  std::vector<f32> vectors;          // anchor 原始向量，按 [anchor][dim] 排布
  std::vector<u32> handles;          // 每个 anchor 对应的全局 ordinal
  std::vector<u64> raw_pointers;     // 每个 anchor 的远端 RemotePtr（用于图缓存键）
  std::vector<u32> shard_offsets;    // 每个 shard 在 vectors 里的起止偏移
  ...
};
```

`count()` 是 `vectors.size() / dim`——这正是日志里 `static_fallback_entries=...` 的来源（`construction.cc:166` 的 `anchor_table.count()`）。

### 3.2 动态路由 overlay

```cpp
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
```

`kDynamicRouteSlotsPerShard = 8`（`types.hh:85`），与 schema-15 的 `kStorageRouteSlots` 用 `static_assert` 对齐（`construction.cc:6`）。每个 shard 有 8 个"自适应路由槽位"，存储侧会通过 `StorageRoutePublication`（见第 8 课、第 10 课动态路由）发布这些槽位的内容。`DynamicRouteOverlayDiff` 是第 10 课讲的差分叠加层，用来追踪"当前快照 vs 上次同步到 GPU"的差集。

### 3.3 anchor → 图缓存键的映射

```cpp
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
```

每个 anchor 的 `raw_pointer` 转换成一个 64 位 `graph_cache_key`（`routing.cc:47`）：高 16 位是 shard 编号，低 48 位是该 anchor 对应的图条目在 shard 内的字节偏移。这些键排序去重后作为"静态兜底路由表"的键集合——后续 `stream_anchor_graph_to_gpu` 会用这些键把 anchor 的图邻接条目（`graph_entry_bytes` 每条）从远端一次性拉到 GPU。`anchor_buckets_by_raw` 是反向映射，供 delta 路由时按 raw pointer 查 anchor 编号。

### 3.4 入口点、free slot、查询票据

```cpp
entry_handles = index.entry_points;
std::cerr << "[gpu-search] query routing=storage-canonical adaptive routes"
          << "+static recall fallback"
          << " static_fallback_entries=" << anchor_table.count()
          << " adaptive_slots_per_shard=" << kDynamicRouteSlotsPerShard
          << " seeds=" << config.gpu_entry_seed_count << '\n';
query_slots = config.gpu_query_slots;
query_dispatch_capacity = memory_budget::next_power_of_two(query_slots * 2);
result_capacity = std::max<u32>(config.k, config.gpu_final_rerank_width);
exact_width = kPersistentMaxExact;
code_bytes = index.layout.code_bytes;
free_slots.resize(query_slots);
for (u32 slot = 0; slot < query_slots; ++slot) free_slots[slot] = slot;
active_query_tickets = std::make_unique<std::atomic<u64>[]>(query_slots);
active_query_snapshots = std::make_unique<std::atomic<u64>[]>(query_slots);
for (u32 slot = 0; slot < query_slots; ++slot) {
  active_query_tickets[slot].store(0, std::memory_order_relaxed);
  active_query_snapshots[slot].store(0, std::memory_order_relaxed);
}
```

这一段设置查询运行时的核心容量参数：

- `entry_handles`：图遍历的起始 ordinal 列表。
- `query_slots`：并发查询槽位数（来自配置 `gpu_query_slots`）。
- `query_dispatch_capacity`：查询派发环容量，必须是 2 的幂（`MappedRing` 要求 mask = capacity - 1），所以用 `next_power_of_two(query_slots * 2)`。
- `result_capacity`：每个查询的结果区大小，取 `k` 与 `gpu_final_rerank_width` 的较大值。
- `exact_width = kPersistentMaxExact = 256`：精确重排宽度固定为上限（kernel 内定长）。
- `code_bytes`：PQ 码宽，后续所有 `* code_bytes` 的尺寸都来自它。
- `free_slots`：初始时所有槽位空闲，按 0..N-1 入栈。
- `active_query_tickets` / `active_query_snapshots`：每槽位一对原子计数器，用于 RCU 式的查询快照同步（见第 16 课存储回收 RCU）。每次查询提交时 `active_query_tickets[slot]++`，完成时 `active_query_snapshots[slot]++`；维护线程通过比较两者判断该槽位是否已穿越回收屏障。

---

## 4. 显式预算核算（`construction.cc:183–345`）

这是本课最厚重的一段。它把"配置里写的 GB 数"翻译成"每个数组的精确字节数"，并在每一步都做显式核算。

### 4.1 节点记录尺寸与可用预算

```cpp
node_record_bytes = static_cast<u32>(VamanaNode::size_until_vector_end());
const u64 engine_budget = static_cast<u64>(
  config.gpu_memory_limit_gb - config.gpu_memory_reserve_gb) << 30;
size_t free_gpu_bytes = 0;
size_t total_gpu_bytes = 0;
check_cuda(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes), "cudaMemGetInfo(GPU navigation budget)");
const u64 runtime_reserve = static_cast<u64>(config.gpu_memory_reserve_gb) << 30;
const u64 physically_available = free_gpu_bytes > runtime_reserve
  ? static_cast<u64>(free_gpu_bytes) - runtime_reserve : 0;
const u64 usable_budget = std::min(engine_budget, physically_available);
```

三个层次：

- `engine_budget`：用户声明的引擎预算 = `gpu_memory_limit_gb - gpu_memory_reserve_gb`（GB → 字节）。
- `physically_available`：当前 GPU 实际空闲字节减去 `runtime_reserve`（保留给 CUDA runtime/驱动开销）。
- `usable_budget`：两者取小，即"既不超过用户声明，也不超过物理可用"。

`cudaMemGetInfo` 在这里既是预算探测，也顺带触发了主上下文的惰性初始化——这是题目中"`cudaFree(0)` 预热"的等价行为。

### 4.2 `memory_budget::estimate`

```cpp
const auto budget = memory_budget::estimate(memory_budget::Request{
  .nodes = index.layout.num_nodes,
  .max_delta_vectors = config.max_vectors,
  .usable_bytes = usable_budget,
  .requested_cache_bytes = static_cast<u64>(config.gpu_adjacency_cache_mb) << 20,
  .requested_exact_cache_bytes = static_cast<u64>(config.gpu_exact_cache_mb) << 20,
  .delta_budget_bytes = static_cast<u64>(config.delta_budget_mb) << 20,
  .dim = config.dim,
  .pq_subquantizers = pq_model.subquantizers,
  .code_bytes = code_bytes,
  .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
  .query_slots = query_slots,
  .beam_width = config.gpu_traversal_beam_width,
  .graph_degree = config.R,
  .exact_width = exact_width,
  .exact_record_bytes = node_record_bytes,
  .anchor_count = anchor_table.count(),
  .shard_count = static_cast<u32>(index.shards.size()),
  .entry_point_count = static_cast<u32>(entry_handles.size()),
  .cache_ways = config.gpu_adjacency_cache_ways,
  .exact_cache_ways = config.gpu_exact_cache_ways,
});
```

这是第 10 课 `memory_budget::estimate` 的实际调用。`Request` 结构（`memory_budget.hh:12`）传入所有决定显存占用的参数；`Result` 结构（`memory_budget.hh:35`）返回每个子区域的字节数与容量。`estimate` 内部的关键计算（见 `memory_budget.hh:112`）：

- `code_bytes = nodes * code_bytes`（PQ base codes 总量）。
- `delta_capacity = choose_delta_capacity(...)`：在 `delta_budget_bytes` 内二分搜索最大 delta 容量（`memory_budget.hh:75`），每个 delta 槽位的 footprint 是 `DeviceDeltaRecord + vector_bytes + code_bytes + 3*sizeof(u32)`（链表 next/prev/remote_position）+ hash 表 `(u32+u64+u64+u32)`。
- `delta_table_capacity = next_power_of_two(delta_capacity * 2)`：delta hash 表按 2 倍 load factor 取整为 2 的幂。
- `visited_capacity = next_power_of_two(max(256, beam * degree * 8))`：每查询的访问位图容量。
- `exact_bytes = query_slots * exact_width * exact_record_bytes`：精确重排暂存区。
- `metadata_bytes`：shard 描述表 + OPQ 矩阵 + PQ 码本 + entry points + anchor 向量/handles/codes + 64 MiB 余量。
- `permanent_override_bytes = ((nodes + 31) / 32) * sizeof(u32)`：永久覆盖位图。
- `fixed_bytes = code + delta + query_workspace + exact + metadata + permanent_override`。
- 图缓存与精确缓存的组相联分配：`cache_sets`、`cache_slots`、`cache_payload_bytes` 等。

### 4.3 预算核算第一道硬失败

```cpp
if (!budget.fits) {
  throw std::runtime_error(
    "GPU navigation allocations exceed the configured memory budget; codes=" +
    std::to_string(budget.code_bytes) + " fixed=" +
    std::to_string(budget.fixed_bytes));
}
```

`budget.fits`（`memory_budget.hh:207`）只有 `explicit_bytes <= usable_bytes` 时为 `true`。`explicit_bytes = fixed_bytes + cache_total_bytes + exact_cache_total_bytes`——即"基础数据 + 两级缓存"。如果不 fits，构造直接失败，抛出带详细字节数的异常。

### 4.4 把 budget 写回 Impl 字段

```cpp
delta_capacity = budget.delta_capacity;
delta_table_capacity = budget.delta_table_capacity;
permanent_override_words = static_cast<u32>((index.layout.num_nodes + 31) / 32);
visited_capacity = budget.visited_capacity;
graph_cache_sets = budget.cache_sets;
graph_cache_slots = budget.cache_slots;
graph_cache_bytes = static_cast<size_t>(budget.cache_payload_bytes);
exact_cache_sets = budget.exact_cache_sets;
exact_cache_slots = budget.exact_cache_slots;
exact_cache_stride = budget.exact_cache_stride;
exact_cache_bytes = static_cast<size_t>(budget.exact_cache_payload_bytes);
graph_admission_sets = std::min(graph_cache_sets, kMaxCacheAdmissionSets);
exact_admission_sets = std::min(exact_cache_sets, kMaxCacheAdmissionSets);
```

- `permanent_override_words` 在 budget 外重新计算一次（与 `memory_budget.hh:154` 同公式），用于 kernel 参数。
- `graph_admission_sets` / `exact_admission_sets` 截断到 `kMaxCacheAdmissionSets = 1 << 18`（`cuda_helpers.hh:16`），因为 admission filter 是组相联的旁路结构，组数过多会爆 16 位索引。

### 4.5 图失效容量

```cpp
const u64 invalidation_capacity = static_cast<u64>(
  std::max(config.storage_owner_batch_max, config.gpu_query_slots)) * config.R;
if (invalidation_capacity > std::numeric_limits<u32>::max()) {
  throw std::runtime_error("GPU navigation graph invalidation capacity exceeds uint32");
}
graph_invalidation_capacity = static_cast<u32>(std::max<u64>(1, invalidation_capacity));
```

delta 发布时需要把被覆盖的图节点标记失效，单批最大失效数 = `max(storage_owner_batch_max, gpu_query_slots) * R`。这是 host staging buffer 的容量上限。

### 4.6 scratch 字节逐项累加

接下来是一长串 `u64` 字节计算，每一项都对应一块 GPU scratch：

```cpp
const u64 dynamic_code_scratch_bytes =
  static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates * code_bytes;
const u64 dynamic_request_scratch_bytes =
  static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
  (sizeof(u32) + 2 * sizeof(u64));
const u64 navigation_candidate_bytes =
  static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
  (sizeof(u32) + sizeof(f32));
```

- `dynamic_code_scratch`：每查询槽位预留给"动态 PQ 码请求结果"的缓冲区，最多 `kPersistentMaxMergeCandidates=2048` 个候选 × `code_bytes` 字节。当 delta 路由需要动态拉取某个非静态节点的 PQ 码时，结果落在这里。
- `dynamic_request_scratch`：每个候选的请求元数据（shard `u32` + offset `u64` + local IOVA `u64`）。
- `navigation_candidate_bytes`：遍历候选的 (handle `u32` + distance `f32`)。

```cpp
const u64 estimated_direct_queue_count =
  static_cast<u64>(config.gpu_rdma_qps) * index.shards.size();
const u64 query_dispatch_bytes = 2 * sizeof(u64) +
  static_cast<u64>(query_dispatch_capacity) *
    (sizeof(u64) + sizeof(QueryDescriptor));
const u64 direct_queue_bytes = estimated_direct_queue_count *
  (2 * sizeof(u64) + sizeof(DeviceRingView<DirectBatchDescriptor>) +
   static_cast<u64>(kDirectBatchQueueCapacity) *
     (sizeof(u64) + sizeof(DirectBatchDescriptor))) +
  static_cast<u64>(query_slots) * index.shards.size() * sizeof(i32);
```

- `estimated_direct_queue_count`：GPUNetIO owner 队列数 = QP 数 × shard 数。每个 (QP, shard) 对应一个 owner 队列（见第 17 课、第 22 课）。
- `query_dispatch_bytes`：查询派发环 = enqueue/dequeue 两个 `u64` 头 + `query_dispatch_capacity` 个 (sequence `u64` + entry `QueryDescriptor`)。
- `direct_queue_bytes`：GPUNetIO owner 队列总字节，每队列 = 2 个 `u64` 头 + 1 个 `DeviceRingView` + `kDirectBatchQueueCapacity=64` 个 (sequence + entry)，外加每 (query_slot, shard) 一个 `i32` 完成状态。

```cpp
const u64 graph_scratch_bytes = static_cast<u64>(query_slots) *
  kPersistentMaxPrefetch * kPersistentGraphCacheLineBytes;
const u64 cache_admission_bytes =
  static_cast<u64>(graph_admission_sets) *
    (kCacheAdmissionWays * sizeof(u64) + sizeof(u32)) +
  static_cast<u64>(exact_admission_sets) *
    (kCacheAdmissionWays * sizeof(u32) + sizeof(u32));
```

- `graph_scratch_bytes`：图邻接预取暂存，每查询槽位 `kPersistentMaxPrefetch=32` 条 × `kPersistentGraphCacheLineBytes=512` 字节。
- `cache_admission_bytes`：两级缓存的 admission filter（4 路组相联），每个 set = 4 个键 + 1 个 victim 位。

### 4.7 路由图字节

```cpp
const u64 route_graph_record_bytes =
  static_cast<u64>(anchor_graph_keys_host.size()) *
  index.layout.graph_entry_bytes;
const u64 route_graph_metadata_bytes =
  static_cast<u64>(anchor_graph_keys_host.size()) *
  (sizeof(u64) + 2 * sizeof(u32));
const u64 dynamic_route_bytes =
  static_cast<u64>(dynamic_route_capacity) *
  sizeof(DeviceDynamicRouteSlot);
const u64 dynamic_route_code_bytes =
  static_cast<u64>(dynamic_route_capacity) * index.layout.code_bytes;
const u64 anchor_route_bytes =
  route_graph_record_bytes + route_graph_metadata_bytes;
route_graph_bytes = anchor_route_bytes + dynamic_route_bytes +
  dynamic_route_code_bytes;
```

"路由图"是静态 anchor 路由 + 动态 adaptive 路由的总和：

- `route_graph_record_bytes`：所有 anchor 的图邻接条目本身（每条 `graph_entry_bytes`）。
- `route_graph_metadata_bytes`：每条 anchor 路由的元数据（key `u64` + state `u32` + reader `u32`）。
- `dynamic_route_bytes`：动态路由槽位（每槽 `DeviceDynamicRouteSlot`）。
- `dynamic_route_code_bytes`：动态路由对应的 PQ 码（每槽 `code_bytes`）。

### 4.8 预算核算第二道硬失败

```cpp
const u64 additional_scratch_bytes =
  dynamic_code_scratch_bytes + dynamic_request_scratch_bytes +
  navigation_candidate_bytes + query_dispatch_bytes + direct_queue_bytes +
  graph_scratch_bytes +
  cache_admission_bytes + route_graph_bytes;
if (additional_scratch_bytes > usable_budget - budget.explicit_bytes) {
  throw std::runtime_error(
    "GPU navigation dynamic-code scratch exceeds the configured memory budget");
}
```

`budget.explicit_bytes` 是第 10 课 `estimate` 已经核算过的"基础数据 + 两级缓存"；这里把所有 scratch 累加为 `additional_scratch_bytes`，检查 `explicit + additional <= usable_budget`。这是第二道预算硬失败。

### 4.9 resident PQ 容量选择

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
resident_pq_table_capacity = memory_budget::next_power_of_two(
  static_cast<u64>(resident_pq_capacity) * 2);
resident_pq_bytes = memory_budget::resident_pq_footprint(
  resident_pq_capacity, code_bytes);
```

resident PQ 是"常驻 GPU 的动态 PQ 码缓存"，专门存最近写入的 delta 向量的 PQ 码（避免每次评分都走 RDMA 拉取）。容量选择用 `choose_resident_pq_capacity`（`memory_budget.hh:97`）在预算内二分搜索最大容量，上限是 `kDeltaHandleMask = 0x7fffffffu`（`persistent_kernel.hh`，delta 句柄掩码）。

**第三道硬失败**：`resident_pq_capacity < delta_capacity`。这意味着 resident PQ 必须至少能容纳所有 delta 槽位的码——否则某些 delta 的 PQ 码无家可归。错误信息直接提示用户调 `--gpu-resident-pq-budget-mb` 或降 `--delta-budget-mb`。

`resident_pq_table_capacity` 是 resident PQ 的 hash 表容量，2 倍 load factor 取 2 的幂。`resident_pq_bytes` 是最终占用（`memory_budget.hh:90`）。

### 4.10 显式总字节与遥测上报

```cpp
explicit_gpu_bytes = budget.explicit_bytes + additional_scratch_bytes +
  resident_pq_bytes;
engine.telemetry_.gpu_memory_explicit_bytes.store(
  explicit_gpu_bytes, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_base_pq_bytes.store(
  budget.code_bytes, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_resident_pq_bytes.store(
  resident_pq_bytes, std::memory_order_relaxed);
engine.telemetry_.resident_pq_capacity.store(
  resident_pq_capacity, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_route_graph_bytes.store(
  route_graph_bytes, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_delta_reserved_bytes.store(
  budget.delta_bytes, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_graph_cache_bytes.store(
  budget.cache_total_bytes, std::memory_order_relaxed);
engine.telemetry_.gpu_memory_exact_cache_bytes.store(
  budget.exact_cache_total_bytes, std::memory_order_relaxed);
```

`explicit_gpu_bytes` 是"显式分配"的总字节——即 `explicit (固定+缓存) + scratch + resident_pq`。这个值会写入 `engine.telemetry_`（第 9 课遥测），供运维监控与 breakdown benchmark（第 30 课）使用。注意这里用 `std::memory_order_relaxed`——遥测不参与同步，只求最终一致。

### 4.11 预算日志

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

这是构造期最长的一行日志，把所有预算项一次性打出来。运维侧可以根据这行日志判断预算是否合理、哪个子区域吃掉了大部分显存。

---

## 5. 远程缓冲区布局与 GPUNetIO 传输层创建（`construction.cc:347–413`）

### 5.1 区域偏移对齐

```cpp
const size_t code_region_bytes = static_cast<size_t>(base_code_region_bytes);
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
```

`align_up`（`cuda_helpers.hh:26`）是手写的向上取整对齐函数。所有区域按 256 或 512 字节对齐——512 字节对齐是为了图缓存行（`kPersistentGraphCacheLineBytes = 512`）和 RDMA 对齐要求。区域顺序：

1. **PQ base codes**（`code_region_bytes`，从 offset 0 开始）
2. **anchor graph records**（`anchor_graph_region_offset`，512 对齐）
3. **dynamic code scratch**（`dynamic_code_region_offset`，256 对齐）
4. **exact records**（`exact_region_offset`，256 对齐）
5. **graph scratch**（`graph_scratch_offset`，512 对齐）
6. **exact cache**（`exact_cache_offset`，256 对齐）
7. **graph cache**（`graph_cache_offset`，512 对齐）
8. **control region**（`control_region_offset`，256 对齐）

### 5.2 控制区子布局

```cpp
const size_t control_snapshot_bytes =
  index.shards.size() * sizeof(format::StorageControlBlock);
const size_t route_snapshot_offset = static_cast<size_t>(align_up(
  control_snapshot_bytes, alignof(format::StorageRoutePublication)));
const size_t route_snapshot_bytes =
  index.shards.size() * sizeof(format::StorageRoutePublication);
const size_t route_sequence_before_offset = static_cast<size_t>(align_up(
  route_snapshot_offset + route_snapshot_bytes, alignof(u64)));
const size_t route_sequence_after_offset = route_sequence_before_offset +
  index.shards.size() * sizeof(u64);
const size_t control_region_bytes = route_sequence_after_offset +
  index.shards.size() * sizeof(u64);
const size_t remote_buffer_bytes = control_region_offset + control_region_bytes;
```

控制区内含三块：

- `StorageControlBlock` 快照数组（每 shard 一份，640 字节，见 `index_format.hh:80`）。
- `StorageRoutePublication` 快照数组（每 shard 一份，448 字节，见第 8 课、第 10 课动态路由）。
- 两组 `u64` 序列号数组（before/after），用于 bracket 读检测撕裂（见 `index_format.hh:112` 注释）。

`remote_buffer_bytes` 是整个远程缓冲区的总字节——它会传给 GPUNetIO 传输层作为"GPU 直连 RDMA 的本地目标区"。

### 5.3 GPUNetIO 传输层创建

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
```

`GpuNetioPersistentTransport` 是第 22 课的主角——它创建一块**直接注册为 RDMA 本地内存的 GPU 缓冲区**（`direct_view.data`），并建立到各存储节点的 QP。注意 `owns_remote_buffer = false`：这块缓冲区由 transport 拥有，析构时由 transport 释放，不由 `~Impl` 直接 `device_free`（见 `lifecycle.cc:444` 的 `if (owns_remote_buffer) device_free(d_remote_buffer)`）。如果没编译 GPUNetIO 支持则直接抛异常——引擎无法工作。

### 5.4 指针派生

```cpp
d_pq_codes = d_remote_buffer;
d_anchor_graph_records = d_remote_buffer + anchor_graph_region_offset;
d_dynamic_code_records = d_remote_buffer + dynamic_code_region_offset;
d_exact_records = d_remote_buffer + exact_region_offset;
d_graph_scratch = d_remote_buffer + graph_scratch_offset;
d_exact_cache = d_remote_buffer + exact_cache_offset;
d_graph_cache = d_remote_buffer + graph_cache_offset;
d_control_snapshots = reinterpret_cast<format::StorageControlBlock*>(
  d_remote_buffer + control_region_offset);
d_storage_route_snapshots = reinterpret_cast<
  format::StorageRoutePublication*>(
    d_remote_buffer + control_region_offset + route_snapshot_offset);
d_storage_route_sequence_before = reinterpret_cast<u64*>(
  d_remote_buffer + control_region_offset + route_sequence_before_offset);
d_storage_route_sequence_after = reinterpret_cast<u64*>(
  d_remote_buffer + control_region_offset + route_sequence_after_offset);
```

所有"大块"GPU 指针都是 `d_remote_buffer` 的偏移派生——**不单独 `cudaMalloc`**。这样设计的关键好处：整个远程缓冲区是**单块连续的 RDMA 注册内存**，远端存储节点可以把数据直接 RDMA 写到任意子区域（PQ codes、control snapshots、route publications 等），不需要 pinned memory 拼接。这是 GPUDirect RDMA 的核心约束（见第 4 课 RDMA 传输库、第 22 课 GPUNetIO 传输）。

### 5.5 控制块引导器与初始化

```cpp
control_bootstrapper = std::make_unique<NavigationBootstrapper>(
  config, channel_context, connection_manager, remote_regions,
  d_remote_buffer, remote_buffer_bytes);
std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
             "queries=strict GPU-initiated GPUNetIO\n";
initialize_storage_reclaim_ack();
// Fail before accepting queries when storage nodes do not expose the
// canonical fixed-route extension. A concurrent publication may produce a
// transient empty result and will simply be retried by maintenance.
(void)read_storage_route_publications();
```

- `NavigationBootstrapper`（`navigation_bootstrapper.hh`）是"CPU posted GPUDirect RDMA"引导器——用普通的 verbs QP（`DetachedQP`）把数据从远端拉到 `d_remote_buffer`。它跟查询期的 GPUNetIO（GPU-initiated）是两套通路：引导期 CPU 主动 posted，查询期 GPU kernel 主动发起。这行日志明确区分了两者。
- `initialize_storage_reclaim_ack()`（`storage_reclaim.cc:298`）：读取每个 shard 的 `StorageControlBlock`，初始化 reclaim ACK 序列数组，并把所有 ACK 序列重置为 0（`storage_reclaim.cc:304` 的 `write_storage_reclaim_acks(reset_sequences)`）。这是第 16 课 RCU 回收的起点。
- `read_storage_route_publications()`（`storage_reclaim.cc:75`）：读取每个 shard 的 `StorageRoutePublication`（自适应路由发布）。`(void)` 强制丢弃返回值——这里只是触发"首次读取 + 校验"。注释说明：并发发布可能产生瞬时空结果，由维护循环重试，不在此处失败。

---

## 6. PQ code 批量 RDMA 引导（`construction.cc:414` → `lifecycle.cc:7`）

```cpp
stream_codes_to_gpu(*control_bootstrapper);
stream_anchor_graph_to_gpu(*control_bootstrapper);
```

这两行是本课的高潮：把所有 PQ base codes 和 anchor graph records 从远端存储节点拉到 GPU。实现在 `lifecycle.cc` 而非 `construction.cc`，因为它们也属于"生命周期"（重建引擎时也会调用）。

### 6.1 `stream_codes_to_gpu` 主体

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
          ...throw std::runtime_error("RDMA PQ code bootstrap failed: ...");
        }
      }
      for (const NavigationRead& request : requests) streamed += request.bytes;
    }
  }
```

关键设计：

- **window/windows 双参数**：`gpu_bootstrap_window_mb` 是单个 RDMA read 请求的最大字节数（窗口大小），`gpu_bootstrap_windows` 是单批最多并发多少个窗口。两者组合控制单次 `source.read` 的并发度与单请求大小。
- **payload 不经主机内存**：`destination_address` 直接是 `d_pq_codes + ordinal_base * code_bytes + offset`——即 GPU 显存地址。`NavigationBootstrapper::Impl::read`（`navigation_bootstrapper.cc:68`）把这个地址作为 `post_send` 的 local address，lkey 用 `gpu_region_.get_lkey()`（注册过的 GPU MR）。这就是 GPUDirect RDMA：数据从远端 NIC 直接 DMA 到 GPU 显存，不经过主机内存。
- **目标偏移计算**：`shard.ordinal_base * code_bytes + offset`。`ordinal_base` 是该 shard 在全局 ordinal 空间的起始编号（见第 7 课 schema-15），PQ codes 数组按 ordinal 连续排布，所以 shard 的 code 区间映射到 `d_pq_codes[ordinal_base * code_bytes .. (ordinal_base + node_count) * code_bytes]`。
- **失败处理**：任一窗口的 status `<= 0` 都抛异常，附带 shard/offset/bytes/destination 调试信息。

### 6.2 总量校验与同步

```cpp
const u64 expected = index.layout.num_nodes * code_bytes;
if (streamed != expected) throw std::runtime_error("GPU PQ code bootstrap size mismatch");
check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GPU PQ bootstrap)");
```

拉取的总字节数必须严格等于 `num_nodes * code_bytes`——任何不一致都是 bug。`cudaDeviceSynchronize` 等待所有 RDMA 写入在 GPU 侧可见（GPUDirect RDMA 写入需要显式 flush/sync 才能被后续 kernel 读到）。

### 6.3 首/中/尾抽样比对

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
    const u64 slot = shard_slots[sample_index];
    samples.push_back(AuditSample{
      .shard = shard.memory_node,
      .slot = slot,
      .ordinal = shard.ordinal_base + slot,
    });
  }
}
std::vector<byte_t> authoritative(code_bytes);
std::vector<byte_t> resident(code_bytes);
for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
  const AuditSample& sample = samples[sample_index];
  const format::ShardRegion& shard = index.shards[sample.shard];
  requests.assign(1, NavigationRead{
    .remote_offset = shard.code_remote_offset + sample.slot * code_bytes,
    .destination_address = reinterpret_cast<u64>(d_exact_records),
    .bytes = code_bytes,
    .memory_node = static_cast<u16>(sample.shard),
  });
  statuses.assign(1, -EIO);
  source.read(requests, statuses);
  if (statuses.front() <= 0) {
    throw std::runtime_error("GPU PQ ordinal audit RDMA read failed: ...");
  }
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
      "GPU PQ ordinal mapping mismatch: shard=... slot=... ordinal=...");
  }
}
std::cerr << "[gpu-search] streamed " << streamed
          << " PQ bytes directly into final GPU storage; ordinal audit passed for "
          << samples.size() << " entries\n";
```

这是非常严密的校验：

- 每个 shard 抽 3 个 slot：首（0）、中（`node_count/2`）、尾（`node_count-1`）。若 shard 只有 1 个节点则只抽 1 个。
- 对每个抽样：单独再发一个 RDMA read，把"权威源"（`shard.code_remote_offset + slot * code_bytes`，按 shard-local slot 编号）拉到 `d_exact_records`（临时暂存区）。
- 然后 `cudaMemcpy` 把 `d_exact_records`（权威源）和 `d_pq_codes + ordinal * code_bytes`（resident，按全局 ordinal 编号）都拷回 host，逐字节比较。
- 任何不匹配都说明"shard-local slot → 全局 ordinal"的映射错位（`ordinal_base` 算错或 code_bytes 不一致），直接抛异常。

这段校验是 dvstor 对"分布式索引装配正确性"的硬保证——一旦通过，后续 kernel 就可以放心地按全局 ordinal 直接索引 `d_pq_codes`。

### 6.4 `stream_anchor_graph_to_gpu`

```cpp
void PersistentSearchEngine::Impl::stream_anchor_graph_to_gpu(NavigationBootstrapper& source) {
  if (anchor_graph_keys_host.empty()) {
    std::cerr << "[gpu-search] static fallback route graph disabled\n";
    return;
  }
  constexpr size_t kBootstrapBatch = 4096;
  ...
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
    ...
  }
```

anchor graph 引导与 PQ code 引导同构，但有几个不同点：

- **批量固定 4096**：anchor 数量通常远少于节点数，固定批大小即可。
- **key 解码**：`key >> 48` 是 shard 编号，`(key << 16) >> 16` 是低 48 位的图条目字节偏移（即 `graph_base_offset + slot * graph_entry_bytes`）。这与 `graph_cache_key`（`routing.cc:47`）的编码方式对称。
- **目标排布**：anchor graph records 按 `anchor_graph_keys_host` 的顺序连续排布在 `d_anchor_graph_records`，每条 `graph_entry_bytes`。后续 kernel 通过二分查找 `anchor_graph_keys` 定位某个 anchor 的图条目。

### 6.5 anchor graph 审计

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

anchor graph 的审计不是逐字节比对（因为没法用第二次 RDMA read 作权威源——图条目是按 key 偏移拉的，权威源就是同一偏移），而是校验**图条目自身的校验和**：`checksum16`（见第 6 课 Vamana 图格式）。同时检查 `record[0]`（度数）不超过 `graph_degree`。这是"自校验"——如果 RDMA 拉错了偏移，校验和几乎必然不匹配。

---

## 7. 静态元数据 H2D 与 anchor PQ code 聚集（`construction.cc:417–503`）

### 7.1 shards / OPQ / PQ centroids / entry points

```cpp
device_allocate(d_shards, index.shards.size(), "cudaMalloc(GPU navigation shards)");
device_allocate(d_opq_matrix, pq_model.rotation.size(), "cudaMalloc(OPQ matrix)");
device_allocate(d_pq_centroids, pq_model.centroids.size(), "cudaMalloc(PQ centroids)");
device_allocate(d_entry_points, entry_handles.size(), "cudaMalloc(GPU navigation entries)");
check_cuda(cudaMemcpy(d_shards, index.shards.data(),
                      index.shards.size() * sizeof(format::ShardRegion),
                      cudaMemcpyHostToDevice), "cudaMemcpy(GPU navigation shards)");
if (!pq_model.rotation.empty()) {
  check_cuda(cudaMemcpy(d_opq_matrix, pq_model.rotation.data(),
                        pq_model.rotation.size() * sizeof(f32),
                        cudaMemcpyHostToDevice), "cudaMemcpy(OPQ matrix)");
}
check_cuda(cudaMemcpy(d_pq_centroids, pq_model.centroids.data(),
                      pq_model.centroids.size() * sizeof(f32),
                      cudaMemcpyHostToDevice), "cudaMemcpy(PQ centroids)");
check_cuda(cudaMemcpy(d_entry_points, entry_handles.data(),
                      entry_handles.size() * sizeof(u32), cudaMemcpyHostToDevice),
           "cudaMemcpy(GPU navigation entries)");
```

`device_allocate`（`cuda_helpers.hh:31`）是对 `cudaMalloc` 的封装，count=0 时直接置 nullptr（不分配），溢出时抛 `overflow_error`，失败时打印 free/total 字节辅助排查。这四块是"静态元数据"：

- `d_shards`：`ShardRegion` 数组（88 字节/个，见 `index_format.hh:145` 的 `static_assert`），kernel 用它把全局 ordinal 翻译到 (shard, slot) 与远端偏移。
- `d_opq_matrix`：OPQ 旋转矩阵（`dim * dim` 个 f32）。查询时先把查询向量乘以旋转矩阵，再做 PQ 量化。
- `d_pq_centroids`：PQ 码本（`subquantizers * 256 * subvector_dim` 个 f32），用于生成查询 LUT。
- `d_entry_points`：入口点 ordinal 数组。

OPQ 矩阵的 `cudaMemcpy` 用 `if (!pq_model.rotation.empty())` 保护——允许退化到无旋转的纯 PQ。

### 7.2 anchor graph keys/states/readers

```cpp
const u32 anchor_graph_count =
  static_cast<u32>(anchor_graph_keys_host.size());
device_allocate(d_anchor_graph_keys, anchor_graph_count,
                "cudaMalloc(GPU anchor route keys)");
device_allocate(d_anchor_graph_states, anchor_graph_count,
                "cudaMalloc(GPU anchor route states)");
device_allocate(d_anchor_graph_readers, anchor_graph_count,
                "cudaMalloc(GPU anchor route readers)");
anchor_graph_ready_states_host.assign(anchor_graph_count,
                                      kResidentRouteReady);
if (anchor_graph_count != 0) {
  check_cuda(cudaMemcpy(d_anchor_graph_keys, anchor_graph_keys_host.data(),
                        anchor_graph_keys_host.size() * sizeof(u64),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(GPU anchor route keys)");
  check_cuda(cudaMemcpy(d_anchor_graph_states,
                        anchor_graph_ready_states_host.data(),
                        anchor_graph_ready_states_host.size() * sizeof(u32),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(GPU anchor route states)");
  check_cuda(cudaMemset(d_anchor_graph_readers, 0,
                        anchor_graph_keys_host.size() * sizeof(u32)),
             "cudaMemset(GPU anchor route readers)");
  check_cuda(cudaHostAlloc(
                 reinterpret_cast<void**>(&anchor_graph_readers_host),
                 anchor_graph_keys_host.size() * sizeof(u32),
                 cudaHostAllocPortable),
             "cudaHostAlloc(GPU anchor route reader snapshot)");
  check_cuda(cudaHostAlloc(
                 reinterpret_cast<void**>(&anchor_graph_validation_host),
                 index.layout.graph_entry_bytes,
                 cudaHostAllocPortable),
             "cudaHostAlloc(GPU anchor route validation record)");
}
```

anchor 路由表在 GPU 上有三个并行数组：

- `d_anchor_graph_keys`：排序去重后的 64 位键，用于二分查找。
- `d_anchor_graph_states`：每条路由的状态字。初始全部置为 `kResidentRouteReady = 2`（`cuda_helpers.hh:17`），表示"图条目已就绪可用"。维护线程刷新某条路由时会把状态置为非 ready，刷新完再置回。
- `d_anchor_graph_readers`：每条路由的读者计数，用于 RCU 式的刷新同步（kernel 读时 inc，维护线程等所有读者退出再覆写）。

host 侧还分配了 `anchor_graph_readers_host`（pinned，用于 CPU 轮询读者快照）和 `anchor_graph_validation_host`（pinned，单条图记录大小，用于刷新后校验）。

### 7.3 anchor 向量与 handles

```cpp
if (!anchor_table.vectors.empty()) {
  std::vector<f32> transposed_anchors(anchor_table.vectors.size());
  for (u32 anchor = 0; anchor < anchor_table.count(); ++anchor) {
    for (u32 dimension = 0; dimension < anchor_table.dim; ++dimension) {
      transposed_anchors[
        static_cast<size_t>(dimension) * anchor_table.count() + anchor] =
          anchor_table.vectors[
            static_cast<size_t>(anchor) * anchor_table.dim + dimension];
    }
  }
  device_allocate(d_anchor_vectors, anchor_table.vectors.size(),
                  "cudaMalloc(GPU navigation anchors)");
  check_cuda(cudaMemcpy(d_anchor_vectors, transposed_anchors.data(),
                        transposed_anchors.size() * sizeof(f32), cudaMemcpyHostToDevice),
             "cudaMemcpy(GPU navigation anchors)");
  device_allocate(d_anchor_handles, anchor_table.handles.size(),
                  "cudaMalloc(GPU navigation anchor handles)");
  check_cuda(cudaMemcpy(d_anchor_handles, anchor_table.handles.data(),
                        anchor_table.handles.size() * sizeof(u32), cudaMemcpyHostToDevice),
             "cudaMemcpy(GPU navigation anchor handles)");
```

anchor 向量需要**转置**存储：磁盘上是行主序 `[anchor][dim]`，GPU 上转成列主序 `[dim][anchor]`。这是因为 kernel 评分时按维度并发访问所有 anchor（计算查询向量到每个 anchor 的距离），列主序让同一维度的所有 anchor 连续——coalesced memory access。`anchor_table.dim` 等于 `config.dim`。

`d_anchor_handles` 是每个 anchor 对应的全局 ordinal，用于从 `d_pq_codes` 取该 anchor 的 PQ 码（见下一步）。

### 7.4 anchor PQ code 聚集（`launch_gather_anchor_codes`）

```cpp
device_allocate(d_anchor_pq_codes,
                static_cast<size_t>(anchor_table.count()) * code_bytes,
                "cudaMalloc(GPU navigation anchor PQ codes)");
launch_gather_anchor_codes(nullptr, d_pq_codes, d_anchor_handles,
                           d_anchor_pq_codes, anchor_table.count(), code_bytes,
                           static_cast<u32>(index.layout.num_nodes));
check_cuda(cudaGetLastError(), "launch_gather_anchor_codes");
check_cuda(cudaStreamSynchronize(nullptr),
           "cudaStreamSynchronize(GPU navigation anchor PQ codes)");
```

这里用 `launch_gather_anchor_codes`（`persistent_kernel.hh:235`）启动一个 device kernel，把每个 anchor 的 PQ 码从 `d_pq_codes`（按全局 ordinal 排布的完整 PQ base codes 数组）"聚集"到 `d_anchor_pq_codes`（按 anchor 顺序连续排布）。kernel 实现（`runtime.cuh:1031`）：

```cpp
__global__ void gather_anchor_codes_kernel(const u8* base_codes,
                                           const u32* anchor_handles,
                                           u8* anchor_codes,
                                           u32 anchor_count,
                                           u32 code_bytes,
                                           u32 node_count) {
  const u64 byte = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
  const u64 total = static_cast<u64>(anchor_count) * code_bytes;
  if (byte >= total) return;
  const u32 anchor = static_cast<u32>(byte / code_bytes);
  const u32 code_byte = static_cast<u32>(byte % code_bytes);
  const u32 handle = anchor_handles[anchor];
  anchor_codes[byte] = handle < node_count
    ? base_codes[static_cast<u64>(handle) * code_bytes + code_byte]
    : 0;
}
```

每个线程负责一个字节：算出它属于哪个 anchor、哪个字节偏移，从 `base_codes[handle * code_bytes + code_byte]` 取值写入 `anchor_codes[byte]`。`handle < node_count` 的兜底——如果 anchor 的 handle 越界（不应该发生，但防御式）则填 0。

**为什么需要这一步？** 因为 kernel 评分时，anchor 路由需要快速访问每个 anchor 的 PQ 码（用于把动态路由请求的 PQ 码与 anchor 码比对，决定路由方向）。如果每次都从 `d_pq_codes` 按 ordinal 随机访问，访存模式不连续；聚集到 `d_anchor_pq_codes` 后，anchor 码连续排布，且与 `d_anchor_vectors`/`d_anchor_handles` 同序，便于 kernel 协同访问。这就是"连续兜底层"的含义——把稀疏的 anchor PQ 码聚集成连续数组。

注意 `launch_gather_anchor_codes(nullptr, ...)` 第一个参数是 stream，传 `nullptr` 表示默认流；后续 `cudaStreamSynchronize(nullptr)` 等待完成。这是构造期少见的同步点——必须等 anchor 码聚集完才能继续。

### 7.5 delta bucket heads

```cpp
device_allocate(d_delta_bucket_heads, anchor_table.count(),
                "cudaMalloc(GPU navigation delta buckets)");
check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                      static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
           "cudaMemset(GPU navigation delta buckets)");
}
```

delta bucket 是 anchor 数量的 hash 桶链表头：每个 anchor 一个桶，桶头初始为 `0xffffffff`（空链表尾标记）。delta 发布时，新 delta 按其最近 anchor 入桶，形成链表（`d_delta_next`/`d_delta_prev`）。这是第 10 课动态路由/预算的核心数据结构——anchor 提供"粗路由"，delta 在 anchor 桶内提供"细路由"。注意这段在 `if (!anchor_table.vectors.empty())` 块内——没有 anchor 时 `d_delta_bucket_heads` 保持 nullptr，delta 路由退化到全量扫描。

---

## 8. 查询 scratch 分配（`construction.cc:505–523`）

这是本课的收尾——查询通路的 scratch 缓冲区。

```cpp
query_input_stride = static_cast<size_t>(config.dim) * sizeof(f32);
device_allocate(d_queries, static_cast<size_t>(query_slots) * config.dim,
                "cudaMalloc(GPU decoded queries)");
mapped_host_allocate(query_input_host, d_query_input,
                     static_cast<size_t>(query_slots) * query_input_stride,
                     "cudaHostAlloc(GPU navigation query input)");
device_allocate(d_transformed_queries, static_cast<size_t>(query_slots) * config.dim,
                "cudaMalloc(GPU transformed queries)");
device_allocate(d_query_luts,
                static_cast<size_t>(query_slots) * pq_model.subquantizers * 256,
                "cudaMalloc(GPU PQ query LUTs)");
device_allocate(d_navigation_candidate_handles,
                static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates,
                "cudaMalloc(GPU navigation candidate handles)");
device_allocate(d_navigation_candidate_distances,
                static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates,
                "cudaMalloc(GPU navigation candidate distances)");
device_allocate(d_visited, static_cast<size_t>(query_slots) * visited_capacity,
                "cudaMalloc(GPU navigation visited)");
```

逐项：

- `query_input_stride = dim * sizeof(f32)`：每个查询输入向量的字节步长。
- `d_queries`：解码后的查询向量（`query_slots * dim` 个 f32）。如果输入是 fp16/bf16，先解码到 fp32 落这里。
- `query_input_host` / `d_query_input`：**pinned + mapped** 的查询输入缓冲区。`mapped_host_allocate`（`cuda_helpers.hh:60`）用 `cudaHostAllocMapped | cudaHostAllocPortable` 分配 pinned host 内存，再 `cudaHostGetDevicePointer` 取设备指针。这样 CPU 写入 host 端、GPU 读 device 端，零拷贝。host 端指针给 admission 线程写查询输入，device 端指针给 kernel 读。
- `d_transformed_queries`：OPQ 旋转后的查询向量（`query_slots * dim` 个 f32）。kernel 评分前先把 `d_queries` 乘以 `d_opq_matrix` 落这里。
- `d_query_luts`：PQ 查找表（`query_slots * subquantizers * 256` 个 f32）。每个查询生成一张 LUT：对每个子量化器、每个码字（256），预计算该码字对该查询的距离贡献。评分时只需查表累加。
- `d_navigation_candidate_handles` / `d_navigation_candidate_distances`：遍历候选的 (handle, distance) 数组，每查询 `kPersistentMaxMergeCandidates=2048` 个。
- `d_visited`：访问位图，每查询 `visited_capacity` 个 u32（位图字），容量来自 `memory_budget::estimate` 的 `next_power_of_two(max(256, beam * degree * 8))`。

---

## 关键数据结构与 GPU 内存布局图

### 本课结束时的 GPU 内存布局

本课结束时（查询 scratch 分配完，dynamic code request scratch 即将开始），GPU 上有两类内存：

**A. 单块连续的远程缓冲区 `d_remote_buffer`**（GPUNetIO 注册，RDMA 可直达）：

```
偏移 0                                              code_region_bytes
├─────────────────────────────────────────────────┤
│ PQ base codes (d_pq_codes)                       │  ← num_nodes * code_bytes
├─────────────────────────────────────────────────┤  512 对齐
│ anchor graph records (d_anchor_graph_records)    │  ← anchor_count * graph_entry_bytes
├─────────────────────────────────────────────────┤  256 对齐
│ dynamic code scratch (d_dynamic_code_records)    │  ← query_slots * MaxMergeCandidates * code_bytes
├─────────────────────────────────────────────────┤  256 对齐
│ exact records (d_exact_records)                  │  ← query_slots * exact_width * node_record_bytes
├─────────────────────────────────────────────────┤  512 对齐
│ graph scratch (d_graph_scratch)                  │  ← query_slots * MaxPrefetch * 512
├─────────────────────────────────────────────────┤  256 对齐
│ exact cache (d_exact_cache)                      │  ← exact_cache_payload_bytes
├─────────────────────────────────────────────────┤  512 对齐
│ graph cache (d_graph_cache)                      │  ← graph_cache_payload_bytes
├─────────────────────────────────────────────────┤  256 对齐
│ control region:                                  │
│   ├─ StorageControlBlock[shards]                 │  ← shards * 640
│   ├─ StorageRoutePublication[shards]             │  ← shards * 448 (alignof 对齐)
│   ├─ route_sequence_before[shards] (u64)         │  ← shards * 8
│   └─ route_sequence_after[shards]  (u64)         │  ← shards * 8
└─────────────────────────────────────────────────┘
```

**B. 独立 `cudaMalloc` 的设备数组**（按分配顺序）：

| 指针 | 尺寸 | 来源 |
|------|------|------|
| `d_shards` | `shards.size() * sizeof(ShardRegion)` | `index_format.hh:145` (88 字节/个) |
| `d_opq_matrix` | `pq_model.rotation.size() * sizeof(f32)` | `dim * dim` |
| `d_pq_centroids` | `pq_model.centroids.size() * sizeof(f32)` | `subquantizers * 256 * subvector_dim` |
| `d_entry_points` | `entry_handles.size() * sizeof(u32)` | `index.entry_points` |
| `d_anchor_graph_keys` | `anchor_graph_count * sizeof(u64)` | 排序去重后的 anchor keys |
| `d_anchor_graph_states` | `anchor_graph_count * sizeof(u32)` | 初始 `kResidentRouteReady=2` |
| `d_anchor_graph_readers` | `anchor_graph_count * sizeof(u32)` | 初始 0 |
| `d_anchor_vectors` | `anchor_count * dim * sizeof(f32)` | 转置后的列主序 |
| `d_anchor_handles` | `anchor_count * sizeof(u32)` | anchor → ordinal |
| `d_anchor_pq_codes` | `anchor_count * code_bytes` | `gather_anchor_codes_kernel` 输出 |
| `d_delta_bucket_heads` | `anchor_count * sizeof(u32)` | 初始 `0xff..ff` |
| `d_queries` | `query_slots * dim * sizeof(f32)` | 解码后查询 |
| `d_transformed_queries` | `query_slots * dim * sizeof(f32)` | OPQ 旋转后 |
| `d_query_luts` | `query_slots * subquantizers * 256 * sizeof(f32)` | PQ 查找表 |
| `d_navigation_candidate_handles` | `query_slots * MaxMergeCandidates * sizeof(u32)` | 遍历候选 |
| `d_navigation_candidate_distances` | `query_slots * MaxMergeCandidates * sizeof(f32)` | 遍历候选距离 |
| `d_visited` | `query_slots * visited_capacity * sizeof(u32)` | 访问位图 |

**C. Pinned + mapped host 内存**（CPU 写 / GPU 零拷贝读）：

| host 指针 | device 指针 | 尺寸 |
|-----------|-------------|------|
| `query_input_host` | `d_query_input` | `query_slots * dim * sizeof(f32)` |
| `anchor_graph_readers_host` | — | `anchor_graph_count * sizeof(u32)` (pinned only) |
| `anchor_graph_validation_host` | — | `graph_entry_bytes` (pinned only) |

### 关键数据结构速查

**`AnchorTable`**（`impl.hh:51`）：anchor 侧表的内存表示，含原始向量（行主序）、handles、raw_pointers、shard_offsets。构造期装载，运行期只读。

**`memory_budget::Request` / `Result`**（`memory_budget.hh:12` / `35`）：预算核算的输入输出。`Request` 是配置 + 索引元数据的投影；`Result` 是每个子区域的字节数与容量。

**`NavigationRead`**（`navigation_bootstrapper.hh:15`）：单次 RDMA read 请求描述符，含远端偏移、本地目标地址、字节数、目标 memory_node。

**`ShardRegion`**（`index_format.hh:62`）：单 shard 的布局描述，含 `ordinal_base`（全局 ordinal 起点）、`code_remote_offset`（PQ 码在远端的字节偏移）、`code_bytes`（该 shard 的 PQ 码总字节）、`memory_node`（存储节点编号）。88 字节，`static_assert` 保证布局稳定。

### 流程图：构造期资源装配

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. bind_cuda_device + 身份/容量校验                              │
│    └─ 失败 → 抛异常，无资源泄漏                                  │
├─────────────────────────────────────────────────────────────────┤
│ 2. synthesize_distributed_view + read PQ model + 一致性校验     │
│    └─ 失败 → 抛异常，无 GPU 资源                                 │
├─────────────────────────────────────────────────────────────────┤
│ 3. load_anchor_table + 动态路由 overlay + free slot 池          │
├─────────────────────────────────────────────────────────────────┤
│ 4. memory_budget::estimate                                      │
│    ├─ !fits → 抛异常                                            │
│    ├─ additional_scratch > 余量 → 抛异常                        │
│    └─ resident_pq_capacity < delta_capacity → 抛异常            │
├─────────────────────────────────────────────────────────────────┤
│ 5. 远程缓冲区布局对齐 + GPUNetIO transport 创建                 │
│    └─ transport 创建失败 / 缓冲区过小 → 抛异常                  │
├─────────────────────────────────────────────────────────────────┤
│ 6. NavigationBootstrapper + reclaim ACK init + route publication│
├─────────────────────────────────────────────────────────────────┤
│ 7. stream_codes_to_gpu (PQ base codes RDMA 引导 + 抽样审计)    │
│    └─ 任一窗口失败 / 总量不符 / 审计不匹配 → 抛异常             │
├─────────────────────────────────────────────────────────────────┤
│ 8. stream_anchor_graph_to_gpu (anchor 图条目引导 + 校验和审计)  │
├─────────────────────────────────────────────────────────────────┤
│ 9. 静态元数据 H2D (shards/OPQ/centroids/entries)                │
├─────────────────────────────────────────────────────────────────┤
│ 10. anchor 路由表 (keys/states/readers) + anchor 向量转置       │
├─────────────────────────────────────────────────────────────────┤
│ 11. launch_gather_anchor_codes (PQ 码聚集) + delta bucket heads │
├─────────────────────────────────────────────────────────────────┤
│ 12. 查询 scratch (queries/input/transformed/LUT/candidates/visited) │
└─────────────────────────────────────────────────────────────────┘
        ↓ 第 13 课续
        dynamic code request scratch + query dispatch ring
        direct batch queue / GPUNetIO owner runtime
        graph cache / exact cache / admission filter
        结果区 / delta records / resident PQ / dynamic route
        stop flag / 4 条 CUDA stream / SM 探测 / CTA 切分
        PersistentKernelParams 填充 + start_persistent_kernel + 3 线程
```

---

## 失败路径与回滚

构造期任一步骤失败都抛 `std::runtime_error` / `std::invalid_argument` / `std::logic_error`。由于 `Impl` 是 PImpl（见第 11 课），构造函数抛异常时：

1. **成员初始化列表构造的对象**（`MappedRing` 4 个、`engine` 引用、`config` 引用）会被正常析构——它们是 `Impl` 的成员，异常时自动逆序析构。
2. **`unique_ptr` 成员**（`dynamic_route_diff`、`control_bootstrapper`、`direct_transport`）会被自动释放。
3. **裸 `d_*` 指针**（`cudaMalloc` 分配的）——如果异常发生在它们分配之后、`~Impl` 之前，**不会自动释放**。这是 PImpl 构造期的一个已知风险点。

但实际上，构造函数内的失败点分布使得"已分配但未释放"的风险很低：

- 步骤 1–4（校验与预算）失败时，**还没有任何 `cudaMalloc`**（只有 `MappedRing` 等成员）。
- 步骤 5（GPUNetIO transport 创建）失败时，`d_remote_buffer` 由 transport 拥有，transport 的 `unique_ptr` 会释放。
- 步骤 7–8（PQ/anchor 引导）失败时，已经分配的 `d_remote_buffer` 由 `direct_transport` 拥有（`owns_remote_buffer = false`），`direct_transport` 释放。
- 步骤 9 之后（静态元数据 H2D）失败时，`d_shards`/`d_opq_matrix` 等已分配——**这些会泄漏**。但这一段几乎只可能因 `cudaMalloc` 失败而失败，而 `device_allocate` 失败时该指针保持 nullptr，已分配的前序指针仍会泄漏。

> **实践提示**：构造期 `cudaMalloc` 失败的概率极低（预算已核算过），且 dvstor 是长生命周期服务进程，构造失败通常意味着环境问题，进程会退出由 systemd/k8s 重启——少量泄漏可接受。`~Impl`（`lifecycle.cc:322`）针对**正常析构**做了完整的 `device_free` 清理（约 50 个 `device_free` 调用），覆盖所有 `d_*` 指针。

---

## 与其他模块的关系

- **第 1 课（项目总览）**：本课是"GPU 中心化"理念的落地——所有索引数据、PQ 码、图条目最终都落在 GPU 显存，CPU 只做控制面。
- **第 2 课（公共类型与配置）**：`IndexConfiguration` 的所有 `gpu_*` 字段在这里被消费。
- **第 3 课（并发原语与协程）**：`MappedRing` 是构造函数初始化列表里最早就绪的成员。
- **第 6 课（Vamana 图格式）**：`VamanaNode::hot_graph_entry_size()`、`HOT_GRAPH_SHARD_BITS`、`size_until_vector_end()` 在一致性校验和 `node_record_bytes` 计算中被用。
- **第 7 课（schema-15 索引格式）**：`format::synthesize_distributed_view` 是本课的入口；`ShardRegion` 字段（`ordinal_base`/`code_remote_offset`/`code_bytes`）决定 PQ 码引导的目标偏移。
- **第 8 课（元数据/owner map/存储协议）**：`StorageControlBlock`、`StorageRoutePublication`、`kStorageRouteSlots` 在控制区布局与路由 publication 读取中被用。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`pq::Model`、`engine.telemetry_.*` 在预算上报中被用。
- **第 10 课（delta/动态路由/预算）**：`memory_budget::estimate` 是本课预算核算的核心；`DynamicRouteOverlayDiff`、`kDynamicRouteSlotsPerShard` 在动态路由 overlay 中被用。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`Impl` 构造函数即本课；`stream_codes_to_gpu`/`stream_anchor_graph_to_gpu` 实现在 `lifecycle.cc`。
- **第 13 课（construction 下）**：本课讲到查询 scratch 结束；第 13 课从 dynamic code request scratch 起，覆盖 QP/runtime、缓存、delta/resident PQ/dynamic route、stream 创建、CTA 切分、`PersistentKernelParams` 填充、kernel launch 与工作线程。
- **第 17 课（kernel 启动器/上下文/device ring）**：`d_direct_batch_queues`、`DeviceRingView` 在第 13 课装配，本课仅提到 `estimated_direct_queue_count` 的预算。
- **第 18 课（候选评分）**：`d_query_luts`、`d_navigation_candidate_*` 是评分的 scratch。
- **第 19 课（RDMA cache）**：`d_graph_cache_*`、`d_exact_cache_*` 在第 13 课装配。
- **第 20 课（查询遍历主循环）**：`d_visited`、`d_transformed_queries` 是遍历的 scratch。
- **第 22 课（GPUNetIO 传输/probe）**：`GpuNetioPersistentTransport`、`direct_view` 在本课创建；查询期的 GPU-initiated GPUNetIO 在第 22 课讲。
- **第 16 课（存储回收 RCU）**：`initialize_storage_reclaim_ack`、`active_query_tickets`/`active_query_snapshots` 是 RCU 的基础设施。

---

## 小结

本课（第 12 课，上篇）覆盖了 `construction.cc` 的前半部分——从构造函数入口（`construction.cc:73`）到查询 scratch 分配（`construction.cc:523`），共约 450 行核心逻辑。要点回顾：

1. **设备绑定与前置校验**：`bind_cuda_device` 幂等绑定，客户端身份与容量上限校验在前，确保不合法配置不分配任何 GPU 资源。
2. **manifest 合成与一致性校验**：`synthesize_distributed_view` + `pq::read_model` + 14 项字段校验，保证磁盘元数据、PQ 模型、运行时配置、图节点格式两两一致。
3. **anchor table 装载**：从 sidecar 读取 anchor 向量/handles/raw_pointers，转换成 `graph_cache_key` 集合作为静态兜底路由表。
4. **显式预算核算**：三道硬失败（`!budget.fits`、scratch 超余量、resident_pq < delta_capacity），每一项 scratch 字节都有明确公式与来源，全部写入遥测。
5. **远程缓冲区布局**：单块连续 RDMA 注册内存，按 256/512 对齐划分为 8 个子区域，所有"大块"指针是偏移派生——GPUDirect RDMA 的核心约束。
6. **PQ code 批量 RDMA 引导**：window/windows 双参数控制并发，payload 直接落 GPU 显存不经主机，首/中/尾抽样逐字节比对保证 ordinal 映射正确。
7. **anchor graph 引导**：按 key 编码（shard | offset）从远端拉图条目，校验和自校验。
8. **anchor PQ code 聚集**：`gather_anchor_codes_kernel` 把稀疏 anchor 码聚集成连续数组，作为 delta 路由的连续兜底层——这正是日志 `static_fallback_entries=...` 的含义。
9. **查询 scratch**：解码查询、OPQ 旋转查询、PQ LUT、候选、访问位图，每查询槽位一份。

### 与第 13 课的分界

本课讲到 **`construction.cc:523`**（`d_visited` 分配完成）为止。第 13 课从 **`construction.cc:524`** 的 `dynamic_request_elements` 起接：

- 524–553：dynamic code request scratch + query dispatch ring（查询派发环装配）
- 555–611：direct batch queue / GPUNetIO owner runtime（每 (QP, shard) 一个 owner 队列）
- 613–707：graph cache / exact cache / admission filter（两级缓存的 keys/states/readers/victims/admission）
- 709–723：结果区（host mapped 的 result_ids/result_distances）
- 725–798：delta records / resident PQ / dynamic route slots
- 800–838：stop flag / direct error flag / 4 条 CUDA stream
- 839–862：SM 数探测 + CTA 切分（owner kernel blocks / query kernel blocks）
- 864–1014：`PersistentKernelParams` 巨型结构体填充（约 150 个字段）
- 1015–1018：`start_persistent_kernel` + 三个工作线程（admission/completion/maintenance）

续见第 13 课。
