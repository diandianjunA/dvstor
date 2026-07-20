# 第 17 课：Kernel 启动器、上下文与 device ring

## 本课目标与涉及文件

第 16 课我们解决了"图节点和 PQ 码在 GPU 上的物理布局、缓存替换"等静态问题。本课把视角抬到**整个持久化 CUDA kernel 的入口与跨设备数据通路**：CPU 怎么把查询请求送进 GPU、GPU kernel 启动后多个 CTA 怎么分工消费请求、device ring 在 GPU 上的无锁实现，以及 CPU↔GPU 的 mapped memory ring 如何与 device ring 协作构成完整的请求/完成通路。

读完本课应当能回答：

- `PersistentKernelParams` 这个上百字段的"大杂烩"结构体怎么分组、谁负责填、kernel 怎么用？
- 6 个 `launch_*` 函数各自的 grid/block 是怎么算出来的？为什么有的按 warp、有的按字节？
- `DeviceRingView` 的 `try_pop`/`try_push` 为什么用 `.acquire.sys`/`.release.sys` 这种 system-scope 内存序？为什么 GPU 端必须 `__threadfence_system`？
- `MappedRing` 怎么用 `cudaHostAllocMapped` 实现"一段内存、两个地址"，并让 CPU 和 GPU 各自只拥有一个 position？
- 5 字节紧凑 RemotePtr 在 GPU 端怎么解码回 8 字节的 `[node|offset]`？
- "CPU mapped ring → device ring → kernel block 消费 → completion ring → CPU" 这条环形数据流的具体拓扑是什么？

涉及文件：

- `src/gpu_search/persistent_kernel.hh` — `PersistentKernelParams` 结构体、`launch_*` 声明。
- `src/gpu_search/persistent_kernel.cu` — 6 个 launch 函数的 grid/block 计算。
- `src/gpu_search/persistent_kernel/context.cuh` — kernel 内部头文件聚合（cub 排序别名）。
- `src/gpu_search/persistent_kernel/runtime.cuh` — kernel 主体 `persistent_search_kernel`、`direct_read_owner_loop`、几个 probe kernel 的实现（launch 函数指向的真实代码）。
- `src/gpu_search/persistent_kernel/query_traversal.cuh` — `decode_compact_raw`（5 字节 RemotePtr 解码）。
- `src/gpu_search/device_ring.cuh` — `DeviceRingView`、`device_ring_try_pop/try_push/push`、`device_ring_load_acquire/store_release`。
- `src/gpu_search/mapped_ring.hh` — `MappedRing` host↔device mapped SPSC ring。
- `src/gpu_search/types.hh` — `QueryDescriptor`/`CompletionDescriptor`/`DeltaPublishDescriptor`/`DeltaPublishCompletion`。
- `src/remote_pointer.hh` — C++ 侧 `RemotePtr` 编码（与 GPU 侧解码镜像对应，见第 3 课）。

---

## 逐文件逐函数讲解

### 1. `persistent_kernel.hh`：参数结构与启动接口

整个文件分三段：编译期常量、设备侧辅助结构体、`PersistentKernelParams` 巨型参数块、6 个 `launch_*` 声明。

#### 1.1 编译期容量常量（`persistent_kernel.hh:13-34`）

```cpp
inline constexpr u32 kPersistentMaxBeam = 128;
inline constexpr u32 kPersistentMaxExact = 256;
inline constexpr u32 kPersistentMaxSubquantizers = 32;
inline constexpr u32 kPersistentMaxEntryPoints = 512;
inline constexpr u32 kPersistentMaxGraphDegree = 128;
inline constexpr u32 kPersistentMaxPrefetch = 32;
inline constexpr u32 kPersistentScoreChunk = 16;
inline constexpr u32 kPersistentMaxMergeCandidates = 2048;
inline constexpr u32 kPersistentMaxShards = 16;
inline constexpr u32 kPersistentMaxAnchorProbes = 64;
inline constexpr u32 kPersistentQueryThreads = 256;
inline constexpr u32 kPersistentGraphCacheLineBytes = 512;
```

这些是 kernel 内部分配 `__shared__` 数组、cub 排序模板实例化时的固定上限。`kPersistentQueryThreads = 256` 是 query CTA 的固定线程数（`launch_persistent_search` 的 `threads` 参数最终就喂这个值，见 `lifecycle.cc:244`）。`kPersistentMaxMergeCandidates = 2048` 与第 18 课的候选评分/归并排序直接相关。`kPersistentMaxShards * kDynamicRouteSlotsPerShard <= kPersistentMaxExact * 2` 的 `static_assert`（`persistent_kernel.hh:33-34`）保证动态路由槽数（每个 shard 8 个）不会超过 exact 阶段能容纳的候选上限。

接着是 delta 相关的位标志与哨兵值（`persistent_kernel.hh:25-32`）：

```cpp
inline constexpr u32 kDeltaHandleBit = 0x80000000u;
inline constexpr u32 kDeltaHandleMask = 0x7fffffffu;
inline constexpr u32 kDeltaDeleted = 1u;
inline constexpr u32 kDeltaDurable = 1u << 1;
inline constexpr u32 kBaseOverrideEmpty = UINT32_MAX;
inline constexpr u32 kBaseOverrideTombstone = UINT32_MAX - 1;
inline constexpr u64 kDeltaRemoteEmpty = 0;
inline constexpr u64 kDeltaRemoteTombstone = UINT64_MAX;
```

`kBaseOverrideEmpty`/`kBaseOverrideTombstone` 是 base override 哈希表的空槽/墓碑标记，`kDeltaRemoteEmpty`/`kDeltaRemoteTombstone` 是 remote→slot 哈希表的对应标记。这些值在 `runtime.cuh` 的 delta 发布路径里被反复 `atomicCAS`，是第 15 课增量发布的设备侧落点。

#### 1.2 设备侧辅助结构体（`persistent_kernel.hh:36-79`）

```cpp
struct DeviceShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 control_remote_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
};
```

每个 shard 的远程布局描述。`ordinal_base/node_count` 把全局 ordinal 切成 shard 内 ordinal；`node_base_offset/node_stride` 是存储节点上向量记录的物理偏移与步长；`graph_base_offset` 是该 shard 的 Vamana 图在远端的基地址；`control_remote_offset`/`code_remote_offset` 分别是控制段（degree/anchor 等）和 PQ 码段的远端偏移；`memory_node` 是该 shard 所在的远端节点编号（与 `RemotePtr` 高 16 位一致）。`dynamic_*` 字段服务于第 10 课的动态路由记录布局。

```cpp
struct DirectRemoteRegion {
  u64 address{};
  u32 rkey{};
  u32 reserved{};
};
```

GPUNetIO 直接读取的远端内存区域：`address` 是远端基地址，`rkey` 是 RDMA rkey（见第 22 课）。

```cpp
struct DeviceDeltaRecord {
  u32 id{};
  u32 generation{};
  u32 flags{};
  u32 base_ordinal{kBaseOverrideEmpty};
  u64 epoch{};
  u64 superseded_epoch{};
  u64 remote_node{};
  u32 anchor_bucket{};
  u32 resident_pq_slot{UINT32_MAX};
};
```

设备侧的 delta 记录。`remote_node` 就是 `RemotePtr::raw_address`（`[node|offset]` 编码），`resident_pq_slot` 指向 resident PQ 表里的位置（见第 15 课）。

```cpp
struct DirectBatchDescriptor {
  const u32* request_shards{};
  const u64* remote_offsets{};
  const u64* local_iova_offsets{};
  i32* completion_status{};
  u32 request_count{};
  u32 memory_node{};
  u32 bytes{};
  u32 reserved{};
};
```

GPUNetIO 批量读请求描述符。`request_shards/remote_offsets/local_iova_offsets` 是三个并行数组，第 `i` 项表示"从 `request_shards[i]` 这个 memory node 的 `remote_offsets[i]` 处读 `bytes` 字节到本地 IOVA `local_iova_offsets[i]`"。`completion_status` 是每批一个的状态字，kernel 完成后 `atomicExch` 写入（见 `runtime.cuh:703-708`）。

#### 1.3 `PersistentKernelParams`（`persistent_kernel.hh:81-223`）

这是整个持久化 kernel 的"配置单"，按功能可以分成 9 组。下面逐组讲解。

**组 1：5 个 ring view（`persistent_kernel.hh:82-86`）**

```cpp
DeviceRingView<QueryDescriptor> submissions;
DeviceRingView<QueryDescriptor> device_submissions;
DeviceRingView<CompletionDescriptor> completions;
DeviceRingView<DeltaPublishDescriptor> delta_submissions;
DeviceRingView<DeltaPublishCompletion> delta_completions;
```

- `submissions`：CPU 提交查询的入口 ring（host_to_device 方向）。
- `device_submissions`：dispatcher CTA 把 `submissions` 转发到的"设备内部"ring，供 query CTA 消费。这是两段式提交：CPU → dispatcher → query，目的是让 CPU 不直接竞争 query CTA 的取队列。
- `completions`：query CTA 写完成描述符，CPU 取（device_to_host 方向）。
- `delta_submissions`/`delta_completions`：delta 发布通路，结构与查询通路对称。

这 5 个 view 都来自 `MappedRing::device_view()`，但**只有 CPU 拥有的那一端的 position 是 mapped memory，另一端是 device 独占的 `cudaMalloc`**（见 `mapped_ring.hh:46-54`）。这点是后面理解 ring 数据流的关键。

**组 2：shard 与索引数据指针（`persistent_kernel.hh:87-106`）**

```cpp
const DeviceShardRegion* shards{};
u32 num_shards{};
const u8* pq_codes{};
const f32* opq_matrix{};
const f32* pq_centroids{};
const u32* entry_points{};
u32 entry_point_count{};
u32 num_nodes{};
u32 medoid_ordinal{};
u32 dim{};
u32 pq_subquantizers{};
u32 pq_subvector_dim{};
u32 pq_code_bytes{};
u32 graph_entry_bytes{};
u32 graph_degree{};
u32 graph_shard_bits{};
u32 node_meta_offset{};
u32 node_record_bytes{};
u32 vector_bytes{};
u32 vector_dtype{};
```

`shards` 指向 `num_shards` 个 `DeviceShardRegion`，是远端访问的元数据。`pq_codes/opq_matrix/pq_centroids/entry_points` 是设备本地的常驻索引数据（PQ 码表、OPQ 旋转矩阵、PQ 码本、入口点表）。`graph_shard_bits` 是 5 字节紧凑 RemotePtr 解码时用来切分 shard 编号与偏移的位数（见下文 `decode_compact_raw`）。`vector_dtype` 标识向量是 u8/i8/f16/f32（见第 2 课 `common/vector_dtype.hh`）。

**组 3：查询预算与缓存容量（`persistent_kernel.hh:107-116`）**

```cpp
u32 traversal_beam_width{};
u32 final_rerank_width{};
u32 entry_seed_count{};
u32 exact_width{};
u32 max_expansions{};
u32 prefetch_depth{};
u32 visited_capacity{};
u32 query_slots{};
u32 direct_region_count{};
u32 direct_qps_per_node{};
u32 direct_local_mkey{};
u64 direct_local_iova_base{};
u64 direct_timeout_ns{};
```

`traversal_beam_width/final_rerank_width/exact_width` 是图遍历和精确重排的预算（见第 18、20 课）。`query_slots` 是引擎同时可挂起的查询数（决定 `__shared__` scratch 的尺寸）。`direct_*` 字段是 GPUNetIO 直接读的配置：`direct_region_count` 个远端节点、每个节点 `direct_qps_per_node` 个 QP、本地 mkey 与 IOVA 基址、超时阈值。

**组 4：GPUNetIO 直接读资源（`persistent_kernel.hh:120-129`）**

```cpp
const DirectRemoteRegion* direct_regions{};
void* const* direct_qps{};
i32* direct_qp_locks{};
const DeviceRingView<DirectBatchDescriptor>* direct_batch_queues{};
i32* direct_batch_statuses{};
u32 direct_batch_queue_count{};
u32* direct_owner_phases{};
u8* direct_dump{};
u32* direct_disabled{};
i32* direct_error{};
```

`direct_regions` 是 `direct_region_count` 个 `DirectRemoteRegion`；`direct_qps` 是 `direct_batch_queue_count` 个 DOCA GPUNetIO QP 指针（每个 owner warp 一个）；`direct_batch_queues` 是 `direct_batch_queue_count` 个 `DeviceRingView<DirectBatchDescriptor>`，即每个 owner warp 一个独立的 batch 输入 ring。`direct_owner_phases` 是 CPU 用来观察 owner warp 生命周期阶段的状态字（1=就绪、2=收到 batch、3=已 submit、4=已 poll、5=出错、6=成功完成，见 `runtime.cuh:771-925`）。`direct_disabled`/`direct_error` 是故障熔断开关。

**组 5：delta 数据结构与哈希表（`persistent_kernel.hh:130-156`）**

```cpp
DeviceDeltaRecord* delta_records{};
u8* delta_vectors{};
u8* delta_pq_codes{};
const u32* delta_staging_slots{};
const DeviceDeltaRecord* delta_staging_records{};
const u8* delta_staging_vectors{};
f32* delta_encode_scratch{};
u32* delta_next{};
u32* delta_prev{};
u32* delta_remote_positions{};
u32* delta_bucket_heads{};
u32* delta_count{};
u32 delta_capacity{};
u32* base_override_keys{};
u64* base_override_epochs{};
u32 base_override_capacity{};
u32* permanent_override_bits{};
u32 permanent_override_words{};
u64* delta_remote_keys{};
u32* delta_remote_slots{};
u32 delta_remote_capacity{};
u8* resident_pq_codes{};
u64* resident_pq_keys{};
u32* resident_pq_slots{};
u32* resident_pq_positions{};
u32 resident_pq_capacity{};
u32 resident_pq_table_capacity{};
```

这一大段是 delta 发布在设备侧的全部状态：

- `delta_records/delta_vectors/delta_pq_codes`：delta 主表（记录、原始向量、PQ 码）。
- `delta_staging_*`：CPU 通过 `cudaMemcpy` 拷上来的暂存区，control CTA 校验后搬到主表。
- `delta_next/delta_prev/delta_bucket_heads`：按 anchor bucket 的链表，用于查询时扫描同 bucket 的 delta。
- `delta_remote_keys/delta_remote_slots/delta_remote_positions`：`remote_node → slot` 的开放寻址哈希表（双向指针）。
- `base_override_keys/base_override_epochs`：`ordinal → epoch` 的 override 表（"这个 ordinal 在某 epoch 后被 delta 覆盖"）。
- `permanent_override_bits`：durable delta 永久覆盖位图。
- `resident_pq_*`：resident PQ 表（把 hot delta 的 PQ 码常驻设备内存，避免每次查询都 RDMA 读）。
- `delta_count`：当前 delta 表的可见条数（control CTA 发布完后 `atomicExch` 写入，查询 CTA `load_cg` 读）。

这些字段的运行时行为在第 15 课已经讲过，本课关注的是它们怎么被 `PersistentKernelParams` 打包传给 kernel。

**组 6：delta 更新数组与动态路由（`persistent_kernel.hh:157-165`）**

```cpp
const DeltaSupersedeUpdate* delta_supersede_updates{};
const DeltaOverrideUpdate* delta_override_updates{};
const DeltaDurableUpdate* delta_durable_updates{};
const ResidentPqEraseUpdate* resident_pq_erase_updates{};
const DynamicRouteUpdate* dynamic_route_updates{};
const u8* dynamic_route_code_updates{};
DeviceDynamicRouteSlot* dynamic_route_slots{};
u8* dynamic_route_pq_codes{};
u32 dynamic_route_capacity{};
const u64* graph_invalidation_keys{};
```

每个 delta 命令可能携带多类更新数组（supersede/override/durable/erase/route），`runtime.cuh` 在 `delta_descriptor` 校验阶段会按各类 count 逐项检查（见 `runtime.cuh:95-165`）。`dynamic_route_slots` 是第 10 课动态路由表在设备侧的落点，`dynamic_route_pq_codes` 是各路由槽的 PQ 码（查询时用 `score_dynamic_route_slot` 评分，见 `query_traversal.cuh:45-97`）。`graph_invalidation_keys` 是本次发布要失效的图缓存键。

**组 7：anchor 相关（`persistent_kernel.hh:167-176`）**

```cpp
const f32* anchor_vectors{};
const u32* anchor_handles{};
const u8* anchor_pq_codes{};
const u64* anchor_graph_keys{};
const u8* anchor_graph_records{};
u32* anchor_graph_states{};
u32* anchor_graph_readers{};
u32 anchor_graph_count{};
u32 anchor_count{};
u32 delta_anchor_probes{};
```

anchor 表与 anchor 图缓存。`anchor_graph_keys/states/readers` 是 anchor→graph 的 RCU 风格缓存表（见第 16 课）。`delta_anchor_probes` 是查询时每个 anchor bucket 要探针的 delta 数量上限。

**组 8：kernel 协调信号（`persistent_kernel.hh:177-183`）**

```cpp
u32* stop{};
u32* kernel_ready_count{};
u32 direct_owner_block_count{};
u32 query_block_count{};
u32* query_kernel_ready_count{};
u32* dispatcher_kernel_ready_count{};
u32* control_kernel_ready_count{};
```

`stop` 是 CPU 设置的停止标志（mapped memory，kernel 每轮循环读）。`kernel_ready_count` 是"非统一调度"模式下的就绪计数；统一调度模式下拆成三个：`query_kernel_ready_count`/`dispatcher_kernel_ready_count`/`control_kernel_ready_count`，分别对应 query CTA、dispatcher CTA、control CTA。`direct_owner_block_count`/`query_block_count` 是统一调度时各类 CTA 的数量，kernel 用 `blockIdx.x` 与这两个值比较来决定自己的角色（见 `runtime.cuh:11-26`）。

**组 9：图缓存、exact 缓存、scratch、结果（`persistent_kernel.hh:184-222`）**

```cpp
u8* graph_cache{};
u8* graph_scratch{};
u64* graph_cache_keys{};
u64* graph_cache_generations{};
u64* graph_cache_timestamps{};
u32* graph_cache_states{};
u32* graph_cache_readers{};
u32* graph_cache_victims{};
u64* graph_admission_keys{};
u32* graph_admission_victims{};
u32 graph_admission_sets{};
const u64* graph_cache_generation{};
u32 graph_cache_sets{};
u32 graph_cache_ways{};
u64 graph_cache_ttl_ns{};
f32* decoded_queries{};
f32* transformed_queries{};
f32* query_luts{};
u32* navigation_candidate_handles{};
f32* navigation_candidate_distances{};
u32* visited_hash{};
u8* exact_records{};
u8* dynamic_code_records{};
u32* dynamic_code_request_shards{};
u64* dynamic_code_request_offsets{};
u64* dynamic_code_request_local_iovas{};
u8* exact_cache{};
u32 exact_cache_stride{};
u32 exact_cache_sets{};
u32 exact_cache_ways{};
u32* exact_cache_keys{};
u32* exact_cache_states{};
u32* exact_cache_readers{};
u32* exact_cache_victims{};
u32* exact_admission_keys{};
u32* exact_admission_victims{};
u32 exact_admission_sets{};
u32* result_ids{};
f32* result_distances{};
```

`graph_cache*`/`graph_admission*` 是第 16 课的图缓存与准入表。`decoded_queries/transformed_queries/query_luts` 是每个 query slot 的解码/OPQ 变换/LUT scratch。`navigation_candidate_handles/distances` 是导航阶段候选。`visited_hash` 是访问位图哈希。`exact_*` 是 exact 阶段的记录缓存与 admission 表。`dynamic_code_request_*` 是动态路由码的 GPUNetIO 读请求 scratch。`result_ids/result_distances` 是最终结果输出。

**小结**：`PersistentKernelParams` 是一个按值传递（`PersistentKernelParams params`，不是引用）给 kernel 的结构体。`runtime.cuh:11` 的 `persistent_search_kernel(PersistentKernelParams params)` 接收时整个结构体被拷贝到每个 block 的寄存器/本地内存。结构体本身只有指针和整数，拷贝开销可忽略，但让 kernel 内部访问字段时不需要再去 global memory 读配置——这是性能上的关键设计。

#### 1.4 `launch_*` 函数声明（`persistent_kernel.hh:225-248`）

```cpp
void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads);
void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads);
void launch_gpunetio_owner_read_probe(
  cudaStream_t stream, const PersistentKernelParams& params,
  u32* request_shards, u64* remote_offsets, u64* local_iova_offsets,
  u8* destinations, u32 destination_stride, i32* statuses,
  u32* completed, u32* phases, u32 queue_count);
void launch_gather_anchor_codes(cudaStream_t stream, const u8* base_codes,
                                const u32* anchor_handles, u8* anchor_codes,
                                u32 anchor_count, u32 code_bytes,
                                u32 node_count);
void launch_gpunetio_locked_read_probe(cudaStream_t stream,
                                       const PersistentKernelParams& params,
                                       u8* destinations, u32 destination_stride,
                                       i32* statuses, u32* completed,
                                       u32 blocks, u32 iterations);
void launch_gpunetio_batched_read_probe(cudaStream_t stream,
                                        const PersistentKernelParams& params,
                                        u8* destinations, u32 destination_stride,
                                        i32* statuses, u32* completed,
                                        u32 blocks, u32 batch_size);
```

注意 `launch_persistent_search` 接收的是 `const PersistentKernelParams&`，但 kernel 启动时是按值传给 device（`persistent_kernel.cu:10` 的 `<<<blocks, threads, 0, stream>>>(params)`）。CUDA runtime 会把 host 端的结构体按值拷贝到 kernel 参数区。这是允许结构体里全是设备指针的前提：host 端填充的是设备地址，按值拷贝到 kernel 后 device 直接用。

---

### 2. `persistent_kernel.cu`：6 个 launch 函数的 grid/block 计算

这个文件极短，只是把 launch 接口翻译成 `<<<grid, block, 0, stream>>>` 调用。但每个 grid/block 的算法都值得讲。

#### 2.1 `launch_persistent_search`（`persistent_kernel.cu:8-11`）

```cpp
void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}
```

grid/block 完全由调用方决定。看 `lifecycle.cc:236-244`：

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
```

`total_blocks = owner_kernel_blocks + kernel_blocks + 2`：前 `owner_kernel_blocks` 个 block 跑 direct owner 循环（GPUNetIO RDMA owner），中间 `kernel_blocks` 个 block 跑 query，倒数第 2 个 block 跑 dispatcher，最后 1 个 block 跑 control（delta 发布）。`threads = kPersistentQueryThreads = 256`。这是"一次 launch、多角色"的统一调度模式，kernel 内部用 `blockIdx.x` 与 `direct_owner_block_count`/`query_block_count` 比较来分流（见 `runtime.cuh:11-26`）。

#### 2.2 `launch_direct_read_owners`（`persistent_kernel.cu:13-19`）

```cpp
void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads) {
  const u32 warps_per_block = max(1u, threads / 32);
  const u32 blocks = (queue_count + warps_per_block - 1) / warps_per_block;
  direct_read_owner_kernel<<<blocks, threads, 0, stream>>>(params, queue_count);
}
```

这是"独立调度 owner"模式（非统一调度）。每个 owner warp 处理一个 batch queue，所以 grid 数量按 warp 计算：`warps_per_block = threads/32`，`blocks = ceil(queue_count / warps_per_block)`。例如 `threads=256` 时 `warps_per_block=8`，`queue_count=16` 则 `blocks=2`，共 16 个 warp，每个 warp 一个 queue。`direct_read_owner_kernel` 内部 `warp = owner_block * warps_per_block + warp_in_block` 决定自己负责的 queue（见 `runtime.cuh:717-722`）。

#### 2.3 `launch_gpunetio_owner_read_probe`（`persistent_kernel.cu:21-31`）

```cpp
void launch_gpunetio_owner_read_probe(
    cudaStream_t stream, const PersistentKernelParams& params,
    u32* request_shards, u64* remote_offsets, u64* local_iova_offsets,
    u8* destinations, u32 destination_stride, i32* statuses,
    u32* completed, u32* phases, u32 queue_count) {
  constexpr u32 threads = 128;
  const u32 blocks = (queue_count + threads - 1) / threads;
  gpunetio_owner_read_probe_kernel<<<blocks, threads, 0, stream>>>(
    params, request_shards, remote_offsets, local_iova_offsets,
    destinations, destination_stride, statuses, completed, phases, queue_count);
}
```

固定 `threads=128`，`blocks = ceil(queue_count/128)`。每个线程负责一个 QP 的探测（`qp_index = blockIdx.x * blockDim.x + threadIdx.x`，见 `runtime.cuh:1003`）。这是 GPUNetIO 健康探测，每个 QP 读一个 u64，统计成功数到 `completed`，状态写到 `statuses`，阶段写到 `phases` 供 CPU 观察。

#### 2.4 `launch_gather_anchor_codes`（`persistent_kernel.cu:33-43`）

```cpp
void launch_gather_anchor_codes(cudaStream_t stream, const u8* base_codes,
                                const u32* anchor_handles, u8* anchor_codes,
                                u32 anchor_count, u32 code_bytes,
                                u32 node_count) {
  const u64 bytes = static_cast<u64>(anchor_count) * code_bytes;
  if (bytes == 0) return;
  constexpr u32 threads = 256;
  const u32 blocks = static_cast<u32>((bytes + threads - 1) / threads);
  gather_anchor_codes_kernel<<<blocks, threads, 0, stream>>>(
    base_codes, anchor_handles, anchor_codes, anchor_count, code_bytes, node_count);
}
```

按**字节总数**算 grid：`bytes = anchor_count * code_bytes`，`blocks = ceil(bytes/256)`。每个线程搬一个字节。`bytes == 0` 时直接 return，避免空 kernel 启动。这是构造期辅助 kernel，把 anchor 对应的 base PQ 码从大表里 gather 出来存成紧凑数组（见第 13 课 construction.cc:492）。

#### 2.5 `launch_gpunetio_locked_read_probe`（`persistent_kernel.cu:45-52`）

```cpp
void launch_gpunetio_locked_read_probe(cudaStream_t stream,
                                       const PersistentKernelParams& params,
                                       u8* destinations, u32 destination_stride,
                                       i32* statuses, u32* completed,
                                       u32 blocks, u32 iterations) {
  gpunetio_locked_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, iterations);
}
```

固定 `threads=128`（即 4 个 warp），`blocks` 与 `iterations` 由调用方决定。kernel 内部 `worker = threadIdx.x / 32`，每 warp 一个 stream，`stream = blockIdx.x * worker_count + worker`（见 `runtime.cuh:951-966`）。每个 stream 跑 `iterations` 次 `direct_fetch`，成功一次就 `atomicAdd(completed, 1)`。这是"锁定 QP、连续探测"的压测 kernel。

#### 2.6 `launch_gpunetio_batched_read_probe`（`persistent_kernel.cu:54-61`）

```cpp
void launch_gpunetio_batched_read_probe(cudaStream_t stream,
                                        const PersistentKernelParams& params,
                                        u8* destinations, u32 destination_stride,
                                        i32* statuses, u32* completed,
                                        u32 blocks, u32 batch_size) {
  gpunetio_batched_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, batch_size);
}
```

固定 `threads=128`，`blocks` 与 `batch_size` 由调用方决定。每个 block 处理一个 memory node，用 `__shared__` 数组准备 `batch_size` 个请求，由 thread 0 调用 `direct_fetch_batch` 提交，再补一次 `direct_fetch` 探测完成（见 `runtime.cuh:968-996`）。成功后 `atomicAdd(completed, batch_size+1)`。这是"批量提交 + 单次确认"的压测 kernel。

**grid/block 设计哲学**：query/owner/delta 这种长寿命 kernel 用"统一调度一次 launch"；probe/gather 这种短寿命 kernel 用"按工作量算 grid"。前者避免多次 launch 开销，后者让 grid 自动适配数据量。

---

### 3. `context.cuh`：kernel 内部共享上下文

这个文件其实是 kernel 内部头文件的聚合点，不是单独的"上下文类"。

```cpp
#pragma once

#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <cerrno>
#include <cmath>
#include <cstdint>

#ifdef DVSTOR_HAVE_GPUNETIO
#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#endif

namespace gpu_search::persistent_kernel_detail {
inline constexpr u32 kApproximateSortThreadsWide = 256;
inline constexpr u32 kApproximateSortItemsWide =
  kPersistentMaxMergeCandidates / kApproximateSortThreadsWide;
inline constexpr u32 kApproximateSortThreadsCompact = 128;
inline constexpr u32 kApproximateSortItemsCompactPass = 8;
inline constexpr u32 kApproximateSortItemsCompactFinal = 2;

using ApproximateBlockSortWide = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWide, u64>;
using ApproximateBlockSortCompactPass = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactPass, u64>;
using ApproximateBlockSortCompactFinal = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactFinal, u64>;
}
```

它做了三件事：

1. **引入 GPUNetIO 设备头**：`doca_gpunetio_dev_verbs_onesided.cuh`，并补定义 `IBV_WC_DRIVER1/2/3` 三个 driver 自定义完成码（135/136/137，见第 22 课）。`#ifndef IBV_WC_DRIVER1` 是因为某些版本的 GPUNetIO 头没定义这些宏。

2. **定义 cub 排序别名**：三个 `cub::BlockRadixSort<f32, threads, items, u64>` 实例。`Wide` 版本 256 线程、每线程 8 项（2048/256=8），用于候选归并排序；`CompactPass`/`CompactFinal` 版本 128 线程、每线程 8 项或 2 项，用于分阶段归并。`(f32, u64)` 是 key-value 排序：key 是距离，value 是 handle。

3. **命名空间 `persistent_kernel_detail`**：所有 kernel 内部辅助函数都放在这里，避免污染 `gpu_search` 命名空间。

**真正的"共享上下文"在 `runtime.cuh:38-50`**：

```cpp
__shared__ QueryDescriptor descriptor;
__shared__ QueryDescriptor dispatch_descriptor;
__shared__ DeltaPublishDescriptor delta_descriptor;
__shared__ u32 have_submission;
__shared__ u32 dispatch_pending;
__shared__ u32 have_delta_submission;
__shared__ u32 stop_requested;
__shared__ u32 idle_cycles;
__shared__ i32 delta_status;
```

这是每个 block 的共享内存协调区。`descriptor`/`dispatch_descriptor`/`delta_descriptor` 是当前 block 正在处理的描述符副本（thread 0 从 ring pop 出来后写到 `__shared__`，全 block 共享）。`have_submission`/`dispatch_pending`/`have_delta_submission` 是"本轮是否拿到任务"的标志。`stop_requested` 是停止信号的 block 内广播。`idle_cycles` 是自适应退避的当前睡眠时长（`runtime.cuh:49` 初始化为 `256 + (blockIdx.x*131 & 1023)`，每个 block 不同以避免同步唤醒）。`delta_status` 是 delta 发布的状态码（0=成功、-EINVAL=参数错、-ENOSPC=表满、-ESTALE=过期命令）。

`context.cuh` 本身不定义这些 `__shared__` 变量，它只提供 cub 排序类型和 GPUNetIO 头。`__shared__` 协调区在 `runtime.cuh` 的 kernel 体内直接声明——这是 CUDA 的惯用法：共享上下文就是 kernel 函数里的 `__shared__` 局部变量。

---

### 4. `device_ring.cuh`：device 端 SPSC/MPSC ring

这是 GPU 端无锁 ring 的核心。文件极小但每个细节都重要。

#### 4.1 `DeviceRingView`（`device_ring.cuh:7-15`）

```cpp
template <class T>
struct DeviceRingView {
  unsigned long long* enqueue_position{};
  unsigned long long* dequeue_position{};
  unsigned long long* sequences{};
  T* entries{};
  unsigned int capacity{};
  unsigned int mask{};
};
```

这是一个**非拥有视图**：5 个指针 + 2 个整数。`enqueue_position`/`dequeue_position` 是 64 位位置计数器（不回绕，单调递增），`sequences` 是每个 slot 的序列号数组（长度 = `capacity`），`entries` 是数据数组。`capacity` 是 2 的幂，`mask = capacity - 1`，slot 索引 = `position & mask`。

这是 Dmitry Vyukov 的 bounded MPMC 队列的 GPU 变体：每个 slot 的 sequence 号初始为 slot 索引（`MappedRing` 构造时 `sequences_host_[index] = index`，见 `mapped_ring.hh:35`）。生产者写到 slot `i` 后把 sequence 设为 `i+1`；消费者读到 slot `i` 后把 sequence 设为 `i+capacity`。这样 sequence 号在"可写"和"可读"之间交替，无需独立锁。

#### 4.2 `device_ring_relax`（`device_ring.cuh:19-27`）

```cpp
__device__ __forceinline__ void device_ring_relax(unsigned int cycles = 64) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __nanosleep(cycles);
#else
  for (volatile unsigned int i = 0; i < cycles; ++i) {
    asm volatile("");
  }
#endif
}
```

GPU 端没有 `std::this_thread::yield`，退避只能"睡"。Volta（`__CUDA_ARCH__ >= 700`）以上有 `__nanosleep` 指令（底层是 `nanosleep.u32`），可以精确睡几十到几千纳秒。老架构用 `volatile` 循环 + `asm volatile("")` 防止优化。默认 64 周期，`runtime.cuh:49` 的自适应退避会从 256 一直翻倍到 16384。

#### 4.3 `device_ring_load_acquire`/`store_release`（`device_ring.cuh:29-45`）

```cpp
__device__ __forceinline__ unsigned long long device_ring_load_acquire(
    const unsigned long long* address) {
  unsigned long long value = 0;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void device_ring_store_release(
    unsigned long long* address, unsigned long long value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}
```

这是 GPU 端 ring 的灵魂。两点关键：

**1. `.acquire`/`.release` 内存序**：CUDA 的 `volatile` 与 `__ldcg` 不保证 release/acquire 语义。这里用 PTX 内联汇编直接发射 `ld.acquire.sys`/`st.release.sys` 指令。`acquire` 保证之后的读不会重排到这条 load 之前；`release` 保证之前的写不会重排到这条 store 之后。这是 Vyukov 队列正确性的基础：消费者读到 `sequence == position+1` 后才读 `entries[slot]`，必须保证读 entries 在读 sequence 之后；生产者写完 `entries[slot]` 才写 `sequence = position+1`，必须保证写 entries 在写 sequence 之前。

**2. `.sys` scope（system）**：PTX load/store 的 scope 有 `.cta`/`.gpu`/`.sys` 三档。`.sys` 保证跨 CPU+GPU 的可见性。为什么用 `.sys` 而不是 `.gpu`？因为 `MappedRing` 把 position 和 sequences 放在 mapped memory 上，CPU 端会直接读写同一个地址（见 `mapped_ring.hh:74-96` 的 `try_push`/`try_pop`）。如果 GPU 用 `.gpu` scope 写 sequence，CPU 可能看不到——必须用 `.sys` 才能保证 CPU 端 `std::atomic_ref<u64>::load(acquire)` 能观察到 GPU 的写。

这就是题目里"为何 GPU 端用 `__threadfence_system`"的答案：device ring 的 sequence 号是 CPU↔GPU 共享变量，任何 release 写都要让 CPU 看见，所以内存序必须 system scope。`runtime.cuh` 里每次发布 delta 后的 `__threadfence_system()`（如 `runtime.cuh:217`、`662`）同理——把 delta 表的修改刷到对 CPU 可见。

#### 4.4 `device_ring_try_pop`（`device_ring.cuh:47-61`）

```cpp
template <class T>
__device__ __forceinline__ bool device_ring_try_pop(DeviceRingView<T> ring, T& value) {
  const unsigned long long position = __ldcg(ring.dequeue_position);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  const unsigned long long sequence = device_ring_load_acquire(ring.sequences + slot);
  bool claimed = false;
  if (sequence == position + 1ULL) {
    claimed = atomicCAS(ring.dequeue_position, position, position + 1ULL) == position;
    if (claimed) {
      value = ring.entries[slot];
      device_ring_store_release(ring.sequences + slot, position + ring.capacity);
    }
  }
  return claimed;
}
```

逐行：

- `__ldcg(ring.dequeue_position)`：用 `__ldcg`（cache-global，不缓存到 L1）读 dequeue 位置。ring 的位置计数器是高频共享变量，缓存会导致看到过期值。
- `slot = position & mask`：定位 slot。
- `device_ring_load_acquire(ring.sequences + slot)`：acquire 读 sequence 号。
- `if (sequence == position + 1ULL)`：检查 slot 是否"可读"。Vyukov 约定下，可读时 sequence == position+1。
- `atomicCAS(dequeue_position, position, position+1)`：CAS 抢占这个 position。多消费者时只有一个能成功，失败的返回 false（注意这里没重试，是 try_pop 语义）。
- `value = entries[slot]`：抢占成功后读数据。
- `device_ring_store_release(sequences + slot, position + capacity)`：release 写 sequence 号为 `position + capacity`，标记此 slot 现在可写（下一个生产者会看到 `sequence == position + capacity`，等价于"slot 已回到可写状态"）。

注意 `entries[slot]` 的读在 acquire 之后、release 之前，符合 acquire-release 配对：生产者的 release 写 sequence 之前的 `entries[slot]` 写对消费者的 acquire 读 sequence 之后的 `entries[slot]` 读可见。

#### 4.5 `device_ring_try_push`（`device_ring.cuh:63-75`）

```cpp
template <class T>
__device__ __forceinline__ bool device_ring_try_push(DeviceRingView<T> ring,
                                                     const T& value) {
  const unsigned long long position = atomicAdd(ring.enqueue_position, 0ULL);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  if (device_ring_load_acquire(ring.sequences + slot) != position) return false;
  if (atomicCAS(ring.enqueue_position, position, position + 1ULL) != position) {
    return false;
  }
  ring.entries[slot] = value;
  device_ring_store_release(ring.sequences + slot, position + 1ULL);
  return true;
}
```

- `atomicAdd(enqueue_position, 0ULL)`：原子读 enqueue 位置（`atomicAdd(_,0)` 等价于原子 load，但比 `__ldcg` 多了原子语义，多生产者时能保证读到一致的值）。
- `slot = position & mask`。
- `device_ring_load_acquire(sequences + slot) != position`：检查 slot 是否"可写"。可写时 sequence == position。
- `atomicCAS(enqueue_position, position, position+1)`：抢占 position。失败则返回 false。
- `entries[slot] = value`：写数据。
- `device_ring_store_release(sequences + slot, position + 1ULL)`：release 写 sequence 为 `position + 1`，标记 slot 可读。

`try_push` 与 `try_pop` 对称：可写条件 `sequence == position` vs 可读条件 `sequence == position + 1`；写完后 sequence 设为 `position + 1`（可读）vs `position + capacity`（可写）。

#### 4.6 `device_ring_push`（`device_ring.cuh:77-86`）

```cpp
template <class T>
__device__ __forceinline__ void device_ring_push(DeviceRingView<T> ring, const T& value) {
  const unsigned long long position = atomicAdd(ring.enqueue_position, 1ULL);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  while (device_ring_load_acquire(ring.sequences + slot) != position) {
    device_ring_relax();
  }
  ring.entries[slot] = value;
  device_ring_store_release(ring.sequences + slot, position + 1ULL);
}
```

阻塞版 push：`atomicAdd(_, 1ULL)` 直接抢占 position（多生产者时天然互斥），然后 while 循环等 slot 变可写。这是 `runtime.cuh` 里 `device_ring_push(delta_completions, ...)`（如 `runtime.cuh:221`、`666`）用的——delta 完成事件必须入队，不能丢，所以用阻塞版。query 提交用 `try_push`（`runtime.cuh:68`）是因为 dispatcher 可以重试。

**MPMC vs SPSC**：device ring 的实现是 MPMC（多生产者多消费者）安全的，但实际使用场景多数是 SPSC：query 通路是 dispatcher 单生产者 → query CTA 单消费者（每个 block 一个）；delta 通路是 control CTA 单生产者 → CPU 单消费者。completion 通路是 query CTA 多生产者 → CPU 单消费者。MPMC 实现保证在任何配置下都正确。

---

### 5. `mapped_ring.hh`：CPU↔GPU mapped memory ring

这是 device ring 的 host 端搭档。它用 `cudaHostAllocMapped` 分配一段 CPU 和 GPU 都能访问的内存，构造一个 `DeviceRingView` 给 kernel 用。

#### 5.1 `Direction` 枚举与构造（`mapped_ring.hh:20-67`）

```cpp
enum class Direction {
  host_to_device,
  device_to_host,
};

MappedRing(u32 requested_capacity, Direction direction)
    : capacity_(normalize_capacity(requested_capacity)) {
  try {
    allocate_mapped(&enqueue_host_, 1, "cudaHostAlloc(ring enqueue)");
    allocate_mapped(&dequeue_host_, 1, "cudaHostAlloc(ring dequeue)");
    allocate_mapped(&sequences_host_, capacity_, "cudaHostAlloc(ring sequences)");
    allocate_mapped(&entries_host_, capacity_, "cudaHostAlloc(ring entries)");

    *enqueue_host_ = 0;
    *dequeue_host_ = 0;
    for (u32 index = 0; index < capacity_; ++index) sequences_host_[index] = index;
    ...
  }
}
```

四个 mapped 数组：enqueue/dequeue position 各 1 个 u64，sequences 是 `capacity` 个 u64，entries 是 `capacity` 个 T。初始 `sequences[i] = i`，符合 Vyukov 约定（slot 0 可写时 sequence==0、可读时 sequence==1...）。

`normalize_capacity`（`mapped_ring.hh:114-117`）把容量 round up 到 2 的幂（`std::bit_ceil`），上限 `1u<<31`，下限 2。这是为了 `mask = capacity - 1` 能用位与代替取模。

#### 5.2 双指针与"谁拥有 position"（`mapped_ring.hh:37-54`）

```cpp
u64* enqueue_device = device_pointer(enqueue_host_, ...);
u64* dequeue_device = device_pointer(dequeue_host_, ...);
u64* sequences_device = device_pointer(sequences_host_, ...);
T* entries_device = device_pointer(entries_host_, ...);

check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_owned_position_), sizeof(u64)), ...);
check_cuda(cudaMemset(device_owned_position_, 0, sizeof(u64)), ...);
if (direction == Direction::host_to_device) {
  dequeue_device = device_owned_position_;
} else {
  enqueue_device = device_owned_position_;
}
```

这是 `MappedRing` 最巧妙的设计。`cudaHostAllocMapped` 给每个 host 指针配一个 device 指针，两者指向同一物理内存。但 position 计数器不能两端都 mapped——如果 CPU 和 GPU 各自原子修改"同一个" mapped position，硬件缓存一致性虽然保证最终可见，但 CAS 会因为缓存行乒乓而极慢。

解决方案：**只有 ring 的"生产者端 position"在 mapped memory 上，消费者端 position 用 `cudaMalloc` 单独分配在 device 上**。

- `host_to_device`（CPU 生产、GPU 消费）：enqueue（生产者位置）用 mapped（CPU 写、GPU 读），dequeue（消费者位置）用 device-only（GPU 独占写）。CPU `try_push` 写 enqueue；GPU `try_pop` 写 dequeue。
- `device_to_host`（GPU 生产、CPU 消费）：enqueue 用 device-only（GPU 独占写），dequeue 用 mapped（GPU 写、CPU 读）。GPU `try_push` 写 enqueue；CPU `try_pop` 写 dequeue。

这样生产者独占自己的 position（无竞争 CAS），消费者也独占自己的 position。`sequences` 和 `entries` 仍是 mapped（生产者写、消费者读），但用 `.acquire.sys`/`.release.sys` 保证跨端可见。

`device_view_` 字段（`mapped_ring.hh:55-62`）把上述指针打包成 `DeviceRingView`，kernel 用这个 view 操作 ring。

#### 5.3 host 端 `try_push`/`try_pop`（`mapped_ring.hh:74-96`）

```cpp
bool try_push(const T& value) {
  std::atomic_ref<u64> enqueue(*enqueue_host_);
  const u64 position = enqueue.load(std::memory_order_relaxed);
  const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
  std::atomic_ref<u64> sequence(sequences_host_[slot]);
  if (sequence.load(std::memory_order_acquire) != position) return false;
  entries_host_[slot] = value;
  sequence.store(position + 1, std::memory_order_release);
  enqueue.store(position + 1, std::memory_order_release);
  return true;
}

bool try_pop(T& value) {
  std::atomic_ref<u64> dequeue(*dequeue_host_);
  const u64 position = dequeue.load(std::memory_order_relaxed);
  const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
  std::atomic_ref<u64> sequence(sequences_host_[slot]);
  if (sequence.load(std::memory_order_acquire) != position + 1) return false;
  value = entries_host_[slot];
  sequence.store(position + capacity_, std::memory_order_release);
  dequeue.store(position + 1, std::memory_order_release);
  return true;
}
```

这是 device ring `try_push`/`try_pop` 的 C++ 镜像，逻辑完全对称。区别：

- 用 `std::atomic_ref<u64>` 而不是 PTX 内联汇编（C++ 端有标准原子）。
- `memory_order_acquire`/`release` 而不是 `.acquire.sys`/`.release.sys`——因为 host 端 `std::atomic` 默认是 system scope（x86 上 acquire/release 就是普通的 `mov` + 隐式 fence）。
- host 端 `try_push` 不需要 `atomicCAS(enqueue_position)`——因为 `host_to_device` 方向下 CPU 是唯一生产者，`enqueue` 不需要原子修改，普通 `store` 即可。`try_pop` 同理（`device_to_host` 方向下 CPU 是唯一消费者）。

这就解释了为什么 device ring 需要 `atomicCAS` 而 host ring 不需要：device ring 是 MPMC 通用实现，host ring 是 SPSC 专用。

#### 5.4 `device_view()` 与 device ring 的协作

```cpp
DeviceRingView<T> device_view() const { return device_view_; }
```

`MappedRing` 把构造好的 view 暴露给 kernel：

```cpp
// construction.cc:865
.submissions = submissions.device_view(),
.completions = completions.device_view(),
.delta_submissions = delta_submissions.device_view(),
.delta_completions = delta_completions.device_view(),
```

`PersistentKernelParams` 里的 5 个 ring view 就是这么来的。kernel 用 `device_ring_try_pop(params.submissions, descriptor)` 等 device ring 函数操作它们。

**`device_submissions` 的特殊性**：它不是 `MappedRing`，而是设备内的 `DeviceRingView`（position 和 sequences 都在 device 上，CPU 不参与）。这是 dispatcher CTA → query CTA 的纯设备内 ring，不需要 mapped memory。它由 `construction.cc` 用 `cudaMalloc` 单独分配，再填到 `PersistentKernelParams::device_submissions`。

---

### 6. RemotePtr 5 字节紧凑编码的 GPU 侧解码

第 3 课讲了 C++ 侧的 `RemotePtr`：

```cpp
// src/remote_pointer.hh:7-22
struct RemotePtr {
  u64 raw_address{};  // [ memory node (16b) | byte offset (48b) ]
  u32 memory_node() const { return raw_address >> 48; }
  u64 byte_offset() const { return (raw_address << 16) >> 16; }
  void store_address(u32 memory_node, u64 byte_offset) {
    raw_address = (static_cast<u64>(memory_node) << 48) | byte_offset;
  }
};
```

`raw_address` 是 `[memory_node(16b) | byte_offset(48b)]`。这是 8 字节的全量编码，用于设备内存中的 `DeviceDeltaRecord::remote_node`、`DeviceDynamicRouteSlot::remote_node` 等字段。

但**图边**（Vamana 邻接表里的邻居指针）不能存 8 字节——一个节点 128 个邻居、每个 8 字节就是 1KB，远端 RDMA 读太浪费。所以图格式用了 5 字节紧凑编码。GPU 侧解码在 `query_traversal.cuh:12-21`：

```cpp
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

逐行：

- 5 字节小端读入 `packed`（40 bit）。
- `packed == (1<<40)-1`（全 1）是"空邻居"哨兵，返回 0。
- `shard_bits >= 16` 防御性检查（shard_bits 不能超过 16，因为 memory_node 只有 16 位）。
- `offset_bits = 40 - shard_bits`：低 `offset_bits` 位是字节偏移（除以 8 后存储，所以乘 8 还原），高 `shard_bits` 位是 shard 编号。
- `shard = packed >> offset_bits`：取高 `shard_bits` 位。
- `offset = (packed & offset_mask) * 8`：取低 `offset_bits` 位并乘 8 还原字节偏移。
- `return (shard << 48) | offset`：组装成 `RemotePtr::raw_address` 格式。

例如 `shard_bits=4` 时，40 位 = 4 位 shard + 36 位偏移（实际偏移 39 位 = 512GB），shard 编号 0~15，对应 `memory_node` 0~15。这与 `persistent_kernel.hh:102` 的 `graph_shard_bits` 字段对应，由索引格式决定（见第 7 课 schema-15）。

调用点在 `query_traversal.cuh:709`：

```cpp
const u64 raw = decode_compact_raw(record + 8 + neighbor * 5, params.graph_shard_bits);
```

图记录格式：每个 record 第 0~7 字节是 degree/anchor 等控制字段，从第 8 字节开始是邻居数组，每个邻居 5 字节。GPU 遍历图时逐个解码邻居指针，再用 `remote_node` 去 delta remote 表查是否有覆盖。

**C++ 侧镜像**：第 3 课 `remote_pointer.hh` 的 `RemotePtr(u32 memory_node, u64 byte_offset)` 是 8 字节编码的构造。5 字节紧凑编码的 C++ 侧在索引格式代码里（见第 7 课 schema-15 索引格式），与这里的 `decode_compact_raw` 互为编解码对。

---

### 7. `runtime.cuh`：kernel 主体与角色调度

虽然本课重点是 launcher/ring/context，但 launch 函数指向的 kernel 实现就在 `runtime.cuh`，必须讲清楚角色调度才能理解"device ring → kernel block 消费"这条数据流。

#### 7.1 `persistent_search_kernel` 的角色分流（`runtime.cuh:11-37`）

```cpp
__global__ void persistent_search_kernel(PersistentKernelParams params) {
  const bool unified_dispatch = params.direct_owner_block_count != 0;
  if (unified_dispatch && blockIdx.x < params.direct_owner_block_count) {
    direct_read_owner_loop(params, params.direct_batch_queue_count, blockIdx.x);
    return;
  }

  bool enable_queries = true;
  bool enable_dispatcher = false;
  bool enable_delta = true;
  if (unified_dispatch) {
    const u32 role_block = blockIdx.x - params.direct_owner_block_count;
    enable_queries = role_block < params.query_block_count;
    enable_dispatcher = role_block == params.query_block_count;
    enable_delta = role_block == params.query_block_count + 1;
    if (!enable_queries && !enable_dispatcher && !enable_delta) return;
    if (threadIdx.x == 0) {
      u32* ready_count = enable_queries ? params.query_kernel_ready_count
        : enable_dispatcher ? params.dispatcher_kernel_ready_count
                           : params.control_kernel_ready_count;
      if (ready_count != nullptr) atomicAdd(ready_count, 1u);
      __threadfence_system();
    }
  } else if (threadIdx.x == 0 && params.kernel_ready_count != nullptr) {
    atomicAdd(params.kernel_ready_count, 1u);
    __threadfence_system();
  }
  ...
}
```

统一调度模式下，block 按 `blockIdx.x` 分流：

- `[0, direct_owner_block_count)`：直接进 `direct_read_owner_loop`，跑 GPUNetIO owner 循环，return 后不再走后面的 query/dispatcher/control 逻辑。
- `[direct_owner_block_count, direct_owner_block_count + query_block_count)`：`enable_queries = true`，跑查询主循环。
- `direct_owner_block_count + query_block_count`：`enable_dispatcher = true`，跑 dispatcher 循环（CPU submissions → device_submissions）。
- `direct_owner_block_count + query_block_count + 1`：`enable_delta = true`，跑 control 循环（delta 发布）。

每个角色 block 的 thread 0 把自己的 ready 计数 +1 并 `__threadfence_system()`，让 CPU 能观察到。CPU 在 `lifecycle.cc:247-267` 轮询这些计数，等所有角色都 ready 后才认为 kernel 启动成功。`__threadfence_system()` 是必须的——`atomicAdd` 只在 device scope 可见，CPU 看不到；必须 fence 到 system scope 才能让 mapped memory 上的 ready 计数对 CPU 可见。这与 device ring 用 `.sys` scope 同理。

非统一调度模式（`direct_owner_block_count == 0`）下，所有 block 都是 query block，用 `kernel_ready_count` 单一计数。

#### 7.2 dispatcher 循环：CPU ring → device ring（`runtime.cuh:58-82`）

```cpp
if (enable_dispatcher) {
  if (threadIdx.x == 0) {
    bool progressed = false;
    if (dispatch_pending == 0 && params.submissions.entries != nullptr &&
        device_ring_try_pop(params.submissions, dispatch_descriptor)) {
      dispatch_pending = 1;
      progressed = true;
    }
    if (dispatch_pending != 0 &&
        params.device_submissions.entries != nullptr &&
        device_ring_try_push(params.device_submissions,
                             dispatch_descriptor)) {
      dispatch_pending = 0;
      progressed = true;
    }
    if (progressed) {
      idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
    } else {
      device_ring_relax(idle_cycles);
      idle_cycles = min(idle_cycles * 2, 16384u);
    }
  }
  __syncthreads();
  continue;
}
```

dispatcher 的唯一职责：从 `submissions`（CPU 写的 mapped ring）pop，push 到 `device_submissions`（设备内 ring）。`dispatch_pending` 表示"手里有一个还没塞进 device ring 的描述符"——如果上一轮 pop 出来了但 push 失败，这一轮先尝试 push。`progressed` 控制退避：有进展就重置 `idle_cycles`，没进展就翻倍退避（最大 16384 周期）。

为什么需要 dispatcher？因为 query CTA 有 `kernel_blocks` 个（可能几十个），如果都直接 `try_pop(params.submissions)`，会有几十个 block 在同一个 mapped ring 上 CAS 竞争 `dequeue_position`。dispatcher 把竞争收敛到 1 个 block，再把描述符分发到 device ring（device ring 的 `enqueue_position` 只有 dispatcher 一个生产者，无竞争）。

#### 7.3 control 循环：delta 发布（`runtime.cuh:83-676`）

`enable_delta` 时，block 先尝试 `device_ring_try_pop(params.delta_submissions, delta_descriptor)`。pop 成功后进入漫长的校验+应用流程：

1. **参数校验**（`runtime.cuh:90-168`）：检查 flags 互斥、count 不超容量、各指针非空、override 槽位合法等。任何一项失败设 `delta_status = -EINVAL`。
2. **reset 命令分支**（`runtime.cuh:171-229`）：如果是 reset 命令，清空 delta 表、remote 表、override 表，push 一个 final_count=0 的完成事件。
3. **staging 搬运与 OPQ/PQ 编码**（`runtime.cuh:231-368`）：把 `delta_staging_*` 搬到主表，对每个 delta 向量做 OPQ 变换和 PQ 编码，结果写到 `delta_pq_codes` 和 `resident_pq_codes`。
4. **图缓存失效**（`runtime.cuh:372-415`）：遍历 `graph_invalidation_keys`，把图缓存里对应 slot 的状态从 `kGraphCacheReady` 改成 `kGraphCacheStale`。
5. **supersede/override/durable 处理**（`runtime.cuh:417-529`）：更新 `superseded_epoch`、`base_override_epochs`、`permanent_override_bits` 等。
6. **remote 表与 bucket 链表插入**（`runtime.cuh:533-598`）：把 delta 记录按 `remote_node` 插入 `delta_remote_keys/slots` 哈希表，按 `anchor_bucket` 插入 `delta_bucket_heads` 链表。
7. **动态路由发布**（`runtime.cuh:600-657`）：用 seqlock 风格更新 `dynamic_route_slots`——先 `sequence.fetch_add(1)` 变奇数（写中），写 code 和 metadata，再 `sequence.fetch_add(1)` 变偶数（写完）。
8. **可见性发布**（`runtime.cuh:659-674`）：`__threadfence()` + `atomicExch(delta_count, final_count)` + `__threadfence_system()`，让查询 CTA 看到 final_count 时 delta 表已完全就绪。最后 `device_ring_push(delta_completions, ...)`。

这里的 `__threadfence_system()` 是关键：`delta_count` 在 mapped memory 上，CPU 会轮询它来判断 delta 是否发布完成（见第 15 课）。`__threadfence()` 保证 device 内的写（delta 表内容）在 `delta_count` 写之前对其他 GPU 线程可见；`__threadfence_system()` 保证对 CPU 可见。这与 device ring 的 `.sys` scope 同一逻辑。

#### 7.4 query 循环：消费 device_submissions（`runtime.cuh:678-699`）

```cpp
if (threadIdx.x == 0) {
  const DeviceRingView<QueryDescriptor> query_queue =
    params.device_submissions.entries != nullptr
      ? params.device_submissions : params.submissions;
  have_submission = enable_queries && query_queue.entries != nullptr &&
    device_ring_try_pop(query_queue, descriptor) ? 1u : 0u;
}
__syncthreads();
if (have_submission == 0) {
  if (threadIdx.x == 0) {
    device_ring_relax(idle_cycles);
    idle_cycles = min(idle_cycles * 2, 16384u);
  }
  __syncthreads();
  continue;
}
if (threadIdx.x == 0) {
  idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
}
__syncthreads();
process_query(params, descriptor);
__syncthreads();
```

query CTA 优先从 `device_submissions`（dispatcher 转发）pop，退化到 `submissions`（直接 CPU 提交，非统一调度模式）。pop 成功后调 `process_query`（第 20 课详讲）。每轮 `__syncthreads()` 保证全 block 看到一致的 `have_submission` 与 `descriptor`。

#### 7.5 `direct_read_owner_loop`：GPUNetIO owner（`runtime.cuh:710-938`）

这是统一调度下 owner block 跑的循环。每个 owner warp 一个 batch queue：

```cpp
const u32 warp = owner_block * warps_per_block + warp_in_block;
...
const u32 memory_node = warp % params.direct_region_count;
auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[warp]);
...
const DeviceRingView<DirectBatchDescriptor> queue = params.direct_batch_queues[warp];
```

`warp % direct_region_count` 决定这个 warp 服务哪个远端节点。每个 warp 独占一个 QP 和一个 batch queue。

主循环（`runtime.cuh:778-932`）：

1. 检查 `stop` 标志（`__shfl_sync` 广播给全 warp）。
2. lane 0 从 batch queue `try_pop` 最多 8 个 `DirectBatchDescriptor`，统计每个 batch 需要的 WQE 数，超过 QP 容量则 defer 到下一轮。
3. 全 warp 协作准备 WQE：每个 lane 处理一个 request，用 `__ballot_sync`/`__popc` 算自己在 batch 内的 rank，调用 `doca_gpu_dev_verbs_wqe_prepare_read` 准备 RDMA read WQE。
4. lane 0 `doca_gpu_dev_verbs_submit` 提交所有 WQE，然后 `poll_direct_cq` 等完成。
5. 每个 batch 完成后 `complete_direct_batch`：`__threadfence_system()` + `atomicExch(completion_status, status)`，让 CPU 看到完成。

`complete_direct_batch` 的 `__threadfence_system()`（`runtime.cuh:706`）又是同一逻辑：`completion_status` 在 mapped memory 上，CPU 在等。这里的 fence 保证 batch 的数据写（RDMA read 的目标 buffer）在 status 写之前对 CPU 可见。

---

## 关键数据结构与流程图

### 数据流：CPU mapped ring → device ring → kernel block → completion ring → CPU

```
                      CPU 侧                                   GPU 侧
+-------------------------------------------------+    +------------------------------------------+
|                                                 |    |                                          |
|  PersistentSearchEngine::Impl                  |    |  persistent_search_kernel (统一调度)     |
|  ├─ submissions: MappedRing<QueryDescriptor>   |    |                                          |
|  │  (host_to_device)                            |    |  block[0..owner-1]: direct_read_owner_   |
|  │  enqueue_host_ ──mapped──► [enqueue_device] |    |     loop(warp → direct_batch_queues[w]) │
|  │  dequeue ──────device_only──► [dequeue_dev] |    |                                          |
|  │                                              |    |  block[owner]: dispatcher               |
|  ├─ device_submissions: DeviceRingView         |    |     try_pop(submissions)  ──┐            |
|  │  (device-only, cudaMalloc)                   |    │     try_push(device_submissions)         |
|  │                                              |    |                              │            |
|  ├─ completions: MappedRing<CompletionDescriptor>|   |  block[owner+1..owner+query]: query      |
|  │  (device_to_host)                            |    │     try_pop(device_submissions) ◄─┘      |
|  │  enqueue ──────device_only──► [enqueue_dev] |    │     process_query(descriptor)            |
|  │  dequeue_host_ ──mapped──► [dequeue_device] |    │       │                                  |
|  │                                              |    │       ▼                                  |
|  ├─ delta_submissions: MappedRing<...>          |    │     try_push(completions) ──mapped──►    |
|  ├─ delta_completions: MappedRing<...>          |    │                                          |
|  │                                              |    |  block[owner+query+1]: control (delta)   |
|  └─ stop/ready_count: mapped u32                |    │     try_pop(delta_submissions)           |
|                                                 |    │     apply delta → push(delta_completions)│
|  try_push(submissions, QueryDescriptor)         |    |                                          |
|  try_pop(completions, CompletionDescriptor)     |    |  *每轮读 stop (mapped u32)               |
|                                                 |    |                                          |
+-------------------------------------------------+    +------------------------------------------+
        ▲                                                                  │
        │                                                                  │
        └──────────── completion ring (mapped, device_to_host) ◄───────────┘

图例：
  ──mapped──►    cudaHostAllocMapped, CPU 和 GPU 同一物理内存
  ──device_only──►  cudaMalloc, GPU 独占
  try_push/try_pop  使用 .acquire.sys/.release.sys 或 std::atomic acquire/release
```

**关键路径**：

1. **提交**：CPU `submissions.try_push(QueryDescriptor)` → mapped enqueue_position 与 sequences（release 写）→ GPU dispatcher `device_ring_try_pop(submissions)`（acquire 读 sequence）。
2. **分发**：dispatcher `device_ring_try_push(device_submissions, descriptor)` → device-only enqueue_position → query CTA `device_ring_try_pop(device_submissions)`。
3. **执行**：query CTA `process_query(descriptor)` → 写 `result_ids/result_distances`。
4. **完成**：query CTA `device_ring_push(completions, CompletionDescriptor)` → mapped sequences（release 写）→ CPU `completions.try_pop()`（acquire 读）。
5. **delta 通路对称**：CPU `delta_submissions.try_push` → control CTA `device_ring_try_pop` → 应用 → `device_ring_push(delta_completions)` → CPU `delta_completions.try_pop`。

### 内存序与 fence 一览

| 操作 | 位置 | 内存序/scope | 原因 |
|---|---|---|---|
| `device_ring_load_acquire(sequences)` | GPU | `ld.acquire.sys` | sequence 是 CPU↔GPU 共享，需 system scope |
| `device_ring_store_release(sequences)` | GPU | `st.release.sys` | 同上 |
| `__ldcg(dequeue_position)` | GPU | cache-global | 避免缓存过期 position |
| `atomicCAS(enqueue_position)` | GPU | device scope atomic | position 是 device-only 或 mapped，device scope 足够 |
| `std::atomic_ref::load(acquire)` | CPU | system scope | x86 acquire 天然 system |
| `atomicAdd(ready_count, 1)` + `__threadfence_system()` | GPU | system fence | ready_count 是 mapped u32，CPU 轮询 |
| `atomicExch(delta_count, final_count)` + `__threadfence_system()` | GPU | system fence | delta_count 是 mapped u32，CPU 轮询 |
| `complete_direct_batch`: `__threadfence_system()` + `atomicExch(status)` | GPU | system fence | status 是 mapped i32，CPU 轮询 |

---

## 与其他模块的关系

- **与第 3 课（并发原语与协程）**：`RemotePtr` 的 C++ 侧 8 字节编码（`src/remote_pointer.hh`）与本课的 `decode_compact_raw` 5 字节解码互为镜像。第 3 课讲的 `std::atomic` acquire/release 是本课 device ring `.acquire.sys`/`.release.sys` 的 host 端对应。
- **与第 7 课（schema-15 索引格式）**：`graph_shard_bits` 字段决定 5 字节紧凑 RemotePtr 的 shard/offset 切分，由索引格式在加载时填入 `PersistentKernelParams`。
- **与第 13 课（construction 下）**：`MappedRing` 的构造（`construction.cc:80-84`）与 `launch_gather_anchor_codes` 的调用（`construction.cc:492`）都在引擎构造期完成。`kernel_params` 的字段填充（`construction.cc:865` 起）把 5 个 ring view 装进 `PersistentKernelParams`。
- **与第 14 课（查询执行/路由/完成）**：CPU 侧的 admission（提交查询到 `submissions`）与 completion（从 `completions` 取结果）直接调用 `MappedRing::try_push`/`try_pop`。本课讲的是这些调用的 device 端落点。
- **与第 15 课（增量发布）**：delta 发布的设备侧实现在 `runtime.cuh:83-676`，本课讲的是它的入口（`delta_submissions` ring）和出口（`delta_completions` ring）。
- **与第 16 课（存储回收 RCU）**：图缓存的 RCU 读者写者协调（`graph_cache_states/readers`）在 `runtime.cuh:372-415` 的失效路径里调用。
- **与第 18 课（候选评分）**：`context.cuh` 定义的 `ApproximateBlockSortWide` 等 cub 排序别名在第 18 课的评分归并里使用。
- **与第 20 课（查询遍历主循环）**：`process_query` 是 query CTA 的核心，本课只讲它被调用的位置，第 20 课详讲内部。
- **与第 21 课（kernel 运行时/角色调度）**：本课的统一调度角色分流（`runtime.cuh:11-37`）是第 21 课的入口，第 21 课会讲更细的角色协同与退避策略。
- **与第 22 课（GPUNetIO 传输/probe）**：`direct_read_owner_loop` 与 3 个 probe kernel 是第 22 课的主角，本课只讲它们的 launch 接口。

---

## 小结

本课把持久化 CUDA kernel 的"入口与通路"讲清楚了：

1. **`PersistentKernelParams`** 是一个上百字段的按值传递参数块，按 ring view / shard 元数据 / 查询预算 / GPUNetIO 资源 / delta 表 / 动态路由 / anchor / 协调信号 / 缓存与 scratch 九组组织。host 端在引擎构造期填充，kernel 启动时按值拷贝到 device。

2. **6 个 `launch_*` 函数**的 grid/block 计算反映了两类设计：长寿命 kernel（`launch_persistent_search`）用统一调度一次 launch 跑多角色，grid 由 `owner_kernel_blocks + kernel_blocks + 2` 决定；短寿命 kernel（probe/gather）按工作量（队列数、字节数）算 grid。

3. **`context.cuh`** 是 kernel 内部头文件聚合点，提供 cub 排序别名和 GPUNetIO 头；真正的 block 共享上下文是 `runtime.cuh` kernel 体内的 `__shared__` 变量（`descriptor`/`dispatch_descriptor`/`delta_descriptor`/`have_submission`/`idle_cycles` 等）。

4. **`device_ring.cuh`** 是 Vyukov MPMC 队列的 GPU 变体，用 `ld.acquire.sys`/`st.release.sys` PTX 内联汇编保证 CPU↔GPU 的 acquire-release 配对。`try_pop`/`try_push` 是非阻塞版（CAS 失败返回 false），`push` 是阻塞版（while + relax）。GPU 端必须用 `.sys` scope 是因为 sequence 号在 mapped memory 上，CPU 端 `std::atomic` 会直接读。

5. **`mapped_ring.hh`** 用 `cudaHostAllocMapped` 分配 CPU↔GPU 共享内存，并把"消费者端 position"单独 `cudaMalloc` 在 device 上，避免 position CAS 的缓存乒乓。`Direction` 枚举决定哪个 position 是 device-only。host 端 `try_push`/`try_pop` 是 device ring 的 C++ 镜像，但 SPSC 所以不需要 CAS。

6. **5 字节紧凑 RemotePtr** 在 `decode_compact_raw` 里解码：5 字节小端 → 40 位 packed，高 `shard_bits` 位是 shard，低 `40-shard_bits` 位是字节偏移（乘 8 还原），组装成 `[shard<<48 | offset]` 的 8 字节 `raw_address` 格式。这是图边在 GPU 端的解码，与第 3 课 C++ 侧 8 字节 `RemotePtr` 互为镜像。

7. **环形数据流**：CPU `submissions.try_push` → mapped ring → dispatcher `device_ring_try_pop` + `try_push` 到 device ring → query CTA `try_pop` + `process_query` → `device_ring_push(completions)` → mapped ring → CPU `completions.try_pop`。delta 通路对称。每个跨端写都配 `__threadfence_system()` 或 `.sys` scope store-release，保证 CPU 可见性。

下一课（第 18 课）进入 `process_query` 内部，讲候选评分与归并排序——本课的 `context.cuh` 定义的 cub 排序别名在那里登场。
