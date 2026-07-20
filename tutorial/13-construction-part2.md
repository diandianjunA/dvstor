# 第 13 课 · 引擎构造与资源分配（下）

> 承接第 12 课。第 12 课已经走完 `PersistentSearchEngine::Impl::Impl` 构造函数前半段（`src/gpu_search/persistent_engine/construction.cc:1`–`416`）——主要覆盖：构造函数初始化列表里的 `MappedRing`（提交/完成/delta 提交/delta 完成）、计算客户端身份校验、`format::synthesize_distributed_view` 合成 manifest、PQ 模型加载、anchor 表加载、动态路由 overlay 容量、`memory_budget::estimate` 内存预算、各 region 偏移计算、`GpuNetioPersistentTransport` 首次出现（拿到 `d_remote_buffer`）、以及 `stream_codes_to_gpu` / `stream_anchor_graph_to_gpu` 的 PQ 码与 anchor 图 RDMA 引导。
>
> 本课从第 12 课留下的"device 端已被 `d_remote_buffer` 单一巨型缓冲覆盖、PQ 码与 anchor 图记录已就位"这一状态继续，覆盖 `construction.cc:417`–`1021`，把余下的 **shard/anchor 元数据 H2D 拷贝**、**device ring 初始化**、**delta/resident-PQ/dynamic-route 设备表清零**、**stop/ready 标志与 CUDA stream 创建**、**block 角色计数（owner/query/dispatcher/control-delta）**、**`PersistentKernelParams` 逐字段装配**、**`start_persistent_kernel` 的统一 grid launch 与就绪 barrier**，以及**最后三条后台线程的启动**讲透。

---

## 本课目标与涉及文件

本课结束时，你应当能够回答以下问题：

1. `Impl::Impl` 在第 12 课结尾已经把 PQ 码、anchor 图记录流式拉到 GPU 上，为什么后面还要做一次 `cudaMemcpy(d_shards, ...)` / `d_anchor_graph_keys` / `d_anchor_vectors` 这一批"小" H2D？这些常驻元数据的角色是什么？
2. `MappedRing`（host↔device 双向 ring，第 3 课）与 `DeviceRingView`（device 端纯 GPU ring，第 17 课）在本构造函数里分别承担什么角色？为什么"提交 ring"用 `MappedRing`，而 query dispatch ring、direct batch queue 用裸 `DeviceRingView`？
3. `direct_batch_queue_count = qps_per_node * remote_region_count` 这一等式为什么是 GPUNetIO QP 装配正确性的硬约束？
4. 持久化 kernel 的 grid 是怎么划分的？为什么 `total_blocks = owner_kernel_blocks + kernel_blocks + 2`？`+2` 里的两个 block 分别是什么角色？
5. `start_persistent_kernel` 在 launch 之后为什么要等 3 秒的 ready barrier？它检查的 4 类 ready 计数（`direct_owner_phases_host`、`query_kernel_ready_host`、`dispatcher_kernel_ready_host`、`control_kernel_ready_host`）分别对应 kernel 内部的哪段代码？
6. 为什么构造函数最后才启动 `admission_thread` / `completion_thread` / `maintenance_thread`？在这三条线程启动之前，参数侧已经做好了哪些就绪保证？

涉及文件（全部绝对路径）：

- 主文件：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/construction.cc`（本课覆盖 `417`–`1021` 行；前半 `1`–`416` 行在第 12 课讲过）
- PImpl 与所有字段声明：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/impl.hh`
- 构造函数调用的 launch 与生命周期：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/lifecycle.cc`
- `PersistentKernelParams` 结构体定义与 launch 入口：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel.hh`、`/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel.cu`
- 持久化 kernel 主体（角色块划分、ready barrier、delta 控制块）：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/runtime.cuh`
- device ring 原语：`/home/xjs/experiment/dvstor/src/gpu_search/device_ring.cuh`
- mapped ring（host↔device）：`/home/xjs/experiment/dvstor/src/gpu_search/mapped_ring.hh`
- CUDA 小工具（`device_allocate` / `mapped_host_allocate` / `align_up` / `kDirectBatchQueueCapacity` 等）：`/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/cuda_helpers.hh`
- GPUNetIO 传输层视图（`GpuNetioPersistentView`）：`/home/xjs/experiment/dvstor/src/gpu/gpunetio_transport.hh`

---

## 逐函数讲解

### 1. shard/entry/anchor 元数据的 H2D 拷贝（`construction.cc:417`–`503`）

第 12 课结尾用 `stream_codes_to_gpu` 和 `stream_anchor_graph_to_gpu` 把**大块**的 PQ 码与 anchor 图记录通过 RDMA 直接拉进 `d_remote_buffer`。接下来这一段做的是"小而碎"的元数据 H2D，把 host 侧已经准备好的 `index.shards` / `pq_model` / `entry_handles` / `anchor_table` / `anchor_graph_keys_host` 推到 device 上，供 kernel 常驻索引使用。

```cpp
// construction.cc:417-434
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

要点：

- `d_shards` 是 `DeviceShardRegion*`（见 `persistent_kernel.hh:36`–`50`），与 `format::ShardRegion` 二进制兼容——`construction.cc:14` 那条 `static_assert(sizeof(DeviceShardRegion) == sizeof(format::ShardRegion))` 正是为此处的 `cudaMemcpy` 直接搬运保驾护航。每个 shard 携带 `ordinal_base`、`node_count`、`node_base_offset`、`code_remote_offset`、`memory_node` 等字段，是 kernel 把 ordinal 翻译成"哪块远端内存、偏移多少"的关键。schema-15 索引格式与 `ShardRegion` 字段含义见第 7 课、第 8 课。
- `d_opq_matrix` 仅当 PQ 模型带 OPQ 旋转（`pq_model.rotation` 非空）时才拷贝；第 9 课讲过 PQ 模型结构。kernel 在对 delta 向量重新编码时会根据 `params.opq_matrix == nullptr` 选择是否做矩阵乘法（`runtime.cuh:314`–`323`）。
- `d_pq_centroids` 是 `subquantizers * 256 * subvector_dim` 个 `f32`，是 delta 重编码时最近邻码本搜索的依据（`runtime.cuh:340`–`352`）。
- `d_entry_points` 是 `u32*`，每个元素是一个 ordinal，是搜索起点。第 12 课讲过 `entry_handles = index.entry_points` 的来源。

接下来是 anchor 路由图（与第 12 课的 anchor 表不同：anchor 表是"原始向量+handle"，anchor 路由图是"按 graph cache key 去重后的 graph 记录索引"）：

```cpp
// construction.cc:435-468
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

这段做四件事：

1. `d_anchor_graph_keys` / `d_anchor_graph_states` / `d_anchor_graph_readers` 三组等长数组，长度都是 `anchor_graph_count`。`keys` 是路由表的键（由 `graph_cache_key(raw)` 算出，第 12 课讲过去重排序），`states` 是 cache 行状态机（`kResidentRouteReady == 2` 表示常驻可用，见 `cuda_helpers.hh:17`），`readers` 是读者计数（用于 RCU 风格的回收，见第 16 课、第 19 课）。
2. 把 `states` 初始化为 `kResidentRouteReady`——因为这些 anchor 图记录已经在第 12 课的 `stream_anchor_graph_to_gpu` 里通过 RDMA 拉到 `d_anchor_graph_records` 了，状态直接置为"常驻就绪"。
3. `anchor_graph_readers_host` 是 `cudaHostAllocPortable` 的 pinned host 内存，用于 CPU 侧（maintenance 线程）读取 device 写入的 reader 计数快照——`refresh_anchor_graph_records` 会用它判断何时安全覆盖一条 anchor 图记录（RCU 回收，第 16 课）。
4. `anchor_graph_validation_host` 是单条 graph 记录的暂存区，maintenance 在覆盖前用它做 checksum 校验。

接着是 anchor 向量与 anchor PQ 码：

```cpp
// construction.cc:469-503
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
  device_allocate(d_anchor_pq_codes,
                  static_cast<size_t>(anchor_table.count()) * code_bytes,
                  "cudaMalloc(GPU navigation anchor PQ codes)");
  launch_gather_anchor_codes(nullptr, d_pq_codes, d_anchor_handles,
                             d_anchor_pq_codes, anchor_table.count(), code_bytes,
                             static_cast<u32>(index.layout.num_nodes));
  check_cuda(cudaGetLastError(), "launch_gather_anchor_codes");
  check_cuda(cudaStreamSynchronize(nullptr),
             "cudaStreamSynchronize(GPU navigation anchor PQ codes)");
  device_allocate(d_delta_bucket_heads, anchor_table.count(),
                  "cudaMalloc(GPU navigation delta buckets)");
  check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                        static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
             "cudaMemset(GPU navigation delta buckets)");
}
```

要点：

- **转置**：`anchor_table.vectors` 是行主序（anchor × dim），但 kernel 需要按维度做矩阵乘法时是列主序更高效，所以这里把它转置成 `dim × anchor_count` 的布局写入 `d_anchor_vectors`。`nearest_anchor` 的语义见第 10 课。
- **anchor PQ 码不是从 host 拷贝，而是用 `launch_gather_anchor_codes` 在 GPU 上现采**（`persistent_kernel.cu:33`–`43`）：以 `d_pq_codes` 为源、`d_anchor_handles` 为索引，gather 出每个 anchor 对应 base 节点的 PQ 码到 `d_anchor_pq_codes`。这样 anchor 与 base 的 PQ 码永远一致，不会因为 host 侧编码路径不同而漂移。同步用 `cudaStreamSynchronize(nullptr)`，即默认流。
- **`d_delta_bucket_heads`** 是 delta 链表的桶头数组，长度等于 anchor 数；初始化为 `0xff`（即 `UINT32_MAX`，表示空链）。第 10 课讲过 delta 按 anchor bucket 分桶组织，第 15 课讲发布流程。

### 2. 查询侧 device 缓冲（`construction.cc:505`–`531`）

```cpp
// construction.cc:505-531
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
                "cudaMalloc(GPU navigation visited"));
const size_t dynamic_request_elements =
  static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates;
device_allocate(d_dynamic_code_request_shards, dynamic_request_elements,
                "cudaMalloc(dynamic PQ request shards)");
device_allocate(d_dynamic_code_request_offsets, dynamic_request_elements,
                "cudaMalloc(dynamic PQ request offsets)");
device_allocate(d_dynamic_code_request_local_iovas, dynamic_request_elements,
                "cudaMalloc(dynamic PQ request local IOVAs)");
```

每个 query slot 对应一组工作区：

- `d_query_input`（**mapped pinned host**，CPU 写入 → GPU 直接读）：admission 线程把用户查询向量写到这里，再 push 到 `submissions` ring；`d_queries` 是 kernel 解码后的 float 缓冲。`mapped_host_allocate`（`cuda_helpers.hh:59`–`75`）用 `cudaHostAllocMapped | cudaHostAllocPortable` 分配并 `cudaHostGetDevicePointer` 拿到 device 句柄。
- `d_transformed_queries`：OPQ 旋转后的查询向量。
- `d_query_luts`：PQ ADC 查找表，每个 slot `subquantizers * 256` 个 `f32`（每个子量化器 256 个码字距离）。
- `d_navigation_candidate_handles` / `d_navigation_candidate_distances`：每次 beam 扩展产生的候选节点合并缓冲，长度 `kPersistentMaxMergeCandidates == 2048`（`persistent_kernel.hh:20`）。第 12 课讲过 `max_merge_candidates` 容量校验。
- `d_visited`：每个 slot 一个 `visited_capacity` 大小的 hash 表，用于去重。第 18 课讲评分时还会提到。
- `d_dynamic_code_request_{shards,offsets,local_iovas}`：当查询需要动态拉取远端 PQ 码（resident-PQ miss）时，构造 GPUNetIO 读请求的 scratch。`local_iovas` 是 GPU 本地 IOVA，配合 `direct_local_mkey` 做 RDMA 写入目标地址。这三个 scratch 与第 19 课（RDMA cache）、第 22 课（GPUNetIO）紧密相关。

### 3. query dispatch ring（device 内部二级 ring，`construction.cc:533`–`553`）

```cpp
// construction.cc:533-554
device_allocate(d_query_dispatch_enqueue, 1,
                "cudaMalloc(GPU query dispatch enqueue)");
device_allocate(d_query_dispatch_dequeue, 1,
                "cudaMalloc(GPU query dispatch dequeue)");
device_allocate(d_query_dispatch_sequences, query_dispatch_capacity,
                "cudaMalloc(GPU query dispatch sequences)");
device_allocate(d_query_dispatch_entries, query_dispatch_capacity,
                "cudaMalloc(GPU query dispatch entries)");
check_cuda(cudaMemset(d_query_dispatch_enqueue, 0, sizeof(u64)),
           "cudaMemset(GPU query dispatch enqueue)");
check_cuda(cudaMemset(d_query_dispatch_dequeue, 0, sizeof(u64)),
           "cudaMemset(GPU query dispatch deenqueue)");
std::vector<u64> query_dispatch_sequences(query_dispatch_capacity);
for (u32 slot = 0; slot < query_dispatch_capacity; ++slot) {
  query_dispatch_sequences[slot] = slot;
}
check_cuda(cudaMemcpy(d_query_dispatch_sequences,
                      query_dispatch_sequences.data(),
                      query_dispatch_sequences.size() * sizeof(u64),
                      cudaMemcpyHostToDevice),
           "cudaMemcpy(GPU query dispatch sequences)");
```

这是**第一个纯 device-side 的 `DeviceRingView`**，容量 `query_dispatch_capacity = next_power_of_two(query_slots * 2)`（第 12 课已算）。它和 `MappedRing` 的根本区别在于：`MappedRing` 的 enqueue 或 dequeue 一端在 host pinned 内存里，CPU 可以直接 push/pop；而 `d_query_dispatch_*` 全在 device 内存里，只有 kernel 能访问。

它存在的目的是**解耦 admission 与 query CTA**：admission 线程把 `QueryDescriptor` push 进 `submissions`（MappedRing，host→device），dispatcher CTA 从 `submissions` pop 出来、再 push 进 `device_submissions`（这个 device ring），query CTA 从 `device_submissions` pop。这样 query CTA 不必去争抢 host pinned 内存（那会触发 PCIe 读），所有热路径都在 GPU 显存里。

`sequences` 初始化为 `[0, 1, 2, ..., capacity-1]`，这是 Dmitry Vyukov bounded MPMC ring 的标准初始化（每个槽位的"期望序列号"等于初始位置），第 3 课讲 `MappedRing` 时也见到过同样的 `sequences_host_[index] = index`。`DeviceRingView` 的 push/pop 实现见 `device_ring.cuh:47`–`86`，用 `ld.acquire.sys` / `st.release.sys` 保证跨 host/device 的顺序（见第 17 课）。

注意第 543 行那条 `"cudaMemset(GPU query dispatch deenqueue)"` 是个无害的拼写错误（写成 `deenqueue`），但只是日志字符串，不影响行为。

### 4. GPUNetIO direct batch queue 装配（`construction.cc:555`–`611`）

这是本课的重点之一：把"每条 QP 一个 device ring"的 owner 队列装配出来。

```cpp
// construction.cc:555-559
direct_batch_queue_count = direct_view.qps_per_node * direct_view.remote_region_count;
if (direct_batch_queue_count == 0 ||
    direct_batch_queue_count != estimated_direct_queue_count) {
  throw std::runtime_error("GPUNetIO QP count does not match the GPU owner queues");
}
```

`direct_view` 是 `GpuNetioPersistentTransport::view()` 返回的视图（`gpunetio_transport.hh:16`–`27`），其 `qps_per_node` / `remote_region_count` 字段描述了传输层实际建好的 QP 拓扑：**每个存储节点建 `qps_per_node` 条 QP**，总共 `qps_per_node * remote_region_count` 条。`estimated_direct_queue_count` 是第 12 课里算预算时用的估算值 `config.gpu_rdma_qps * index.shards.size()`（`construction.cc:248`–`249`）。这里强校验二者相等——若不等就抛异常，因为后面 kernel 里的 owner warp 会按 `warp = owner_block * warps_per_block + warp_in_block` 直接索引 `direct_batch_queues[warp]` 与 `direct_qps[warp]`，下标越界会直接炸。

```cpp
// construction.cc:560-577
const size_t direct_queue_slots =
  static_cast<size_t>(direct_batch_queue_count) * kDirectBatchQueueCapacity;
device_allocate(d_direct_batch_enqueue, direct_batch_queue_count,
                "cudaMalloc(GPUNetIO owner enqueue positions)");
device_allocate(d_direct_batch_dequeue, direct_batch_queue_count,
                "cudaMalloc(GPUNetIO owner dequeue positions)");
device_allocate(d_direct_batch_sequences, direct_queue_slots,
                "cudaMalloc(GPUNetIO owner queue sequences)");
device_allocate(d_direct_batch_entries, direct_queue_slots,
                "cudaMalloc(GPUNetIO owner queue entries)");
device_allocate(d_direct_batch_queues, direct_batch_queue_count,
                "cudaMalloc(GPUNetIO owner queue views)");
device_allocate(d_direct_batch_statuses,
                static_cast<size_t>(query_slots) * index.shards.size(),
                "cudaMalloc(GPUNetIO owner completion statuses"));
mapped_host_allocate(direct_owner_phases_host, d_direct_owner_phases,
                     direct_batch_queue_count,
                     "cudaHostAlloc(GPUNetIO owner runtime phases)");
```

每个 owner queue 是一个 `DeviceRingView<DirectBatchDescriptor>`，容量 `kDirectBatchQueueCapacity == 64`（`cuda_helpers.hh:14`）。所有队列共享一段连续的 `sequences` / `entries` 大数组（`direct_queue_slots = queue_count * 64`），每个队列在 `queue_base = queue * 64` 处取自己的 64 槽切片。`d_direct_batch_queues` 是 `DeviceRingView` 数组，kernel 通过它索引到每条队列。

`d_direct_batch_statuses` 是**每个 (query_slot, shard) 二元组一个 i32**——每个 query 在每个 shard 上的 GPUNetIO 读完成状态。kernel 发起 RDMA 读后会把状态写到这里，查询 CTA 轮询它判断完成。

`direct_owner_phases_host` / `d_direct_owner_phases` 是 **mapped pinned** 的调试/就绪通道：每个 owner warp 一个 `u32` phase，kernel 在生命周期不同阶段写不同值（`1` = 就绪、`2` = 首次取到 batch、`3` = 提交 WQE、`4` = 等 CQ、`5` = 出错、`6` = 成功，见 `runtime.cuh:771`–`925`）。CPU 侧的 `start_persistent_kernel` 与 `report_direct_path_failure` 通过 host 指针读取这些 phase 值判断 owner warp 是否健康。

接下来把所有 enqueue/dequeue 清零、sequences 初始化为 `[0..63]` 重复 `queue_count` 次：

```cpp
// construction.cc:578-611
check_cuda(cudaMemset(d_direct_batch_enqueue, 0,
                      static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
           "cudaMemset(GPUNetIO owner enqueue positions)");
check_cuda(cudaMemset(d_direct_batch_dequeue, 0,
                      static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
           "cudaMemset(GPUNetIO owner dequeue positions)");
std::vector<u64> direct_sequences(direct_queue_slots);
std::vector<DeviceRingView<DirectBatchDescriptor>> direct_queues(
  direct_batch_queue_count);
for (u32 queue = 0; queue < direct_batch_queue_count; ++queue) {
  const size_t queue_base = static_cast<size_t>(queue) * kDirectBatchQueueCapacity;
  for (u32 slot = 0; slot < kDirectBatchQueueCapacity; ++slot) {
    direct_sequences[queue_base + slot] = slot;
  }
  direct_queues[queue] = {
    .enqueue_position = reinterpret_cast<unsigned long long*>(
      d_direct_batch_enqueue + queue),
    .dequeue_position = reinterpret_cast<unsigned long long*>(
      d_direct_batch_dequeue + queue),
    .sequences = reinterpret_cast<unsigned long long*>(
      d_direct_batch_sequences + queue_base),
    .entries = d_direct_batch_entries + queue_base,
    .capacity = kDirectBatchQueueCapacity,
    .mask = kDirectBatchQueueCapacity - 1,
  };
}
check_cuda(cudaMemcpy(d_direct_batch_sequences, direct_sequences.data(),
                      direct_sequences.size() * sizeof(u64), cudaMemcpyHostToDevice),
           "cudaMemcpy(GPUNetIO owner queue sequences)");
check_cuda(cudaMemcpy(d_direct_batch_queues, direct_queues.data(),
                      direct_queues.size() *
                        sizeof(DeviceRingView<DirectBatchDescriptor>),
                      cudaMemcpyHostToDevice),
           "cudaMemcpy(GPUNetIO owner queue views)");
```

注意 `mask = capacity - 1`，因为 `kDirectBatchQueueCapacity == 64` 是 2 的幂；这是 bounded MPMC ring 的硬要求。`direct_queues` 这个 `DeviceRingView` 数组整体 H2D 一次拷贝，kernel 之后通过 `params.direct_batch_queues[warp]` 直接拿到一个 view 结构体（内含 4 个指针 + capacity + mask），所有后续 push/pop 都在 device 内存里完成。QP 数组本身（`direct_view.qp_array`）由 `GpuNetioPersistentTransport` 在第 22 课讲的那一侧创建并 export 给 GPU。

### 5. graph cache / exact cache 的 admission 与状态机（`construction.cc:613`–`707`）

这一段把图缓存（邻接表缓存）与精确重排缓存的 device 数组全部建出来并清零。第 12 课已经算好了 `graph_cache_sets` / `graph_cache_slots` / `graph_cache_bytes` 与 `exact_cache_*` 的容量，本段是把它们物化。

```cpp
// construction.cc:613-675
device_allocate(d_graph_cache_keys, graph_cache_slots, "cudaMalloc(navigation cache keys)");
device_allocate(d_graph_cache_generations, graph_cache_slots,
                "cudaMalloc(navigation cache generations)");
device_allocate(d_graph_cache_timestamps, graph_cache_slots,
                "cudaMalloc(navigation cache timestamps)");
device_allocate(d_graph_cache_states, graph_cache_slots, "cudaMalloc(navigation cache states)");
device_allocate(d_graph_cache_readers, graph_cache_slots, "cudaMalloc(navigation cache readers)");
device_allocate(d_graph_cache_victims, graph_cache_sets, "cudaMalloc(navigation cache victims)");
device_allocate(d_graph_admission_keys,
                static_cast<size_t>(graph_admission_sets) * kCacheAdmissionWays,
                "cudaMalloc(navigation admission keys)");
device_allocate(d_graph_admission_victims, graph_admission_sets,
                "cudaMalloc(navigation admission victims)");
device_allocate(d_graph_cache_generation, 1, "cudaMalloc(navigation cache generation)");
delta_command_capacity = std::max({1u, config.storage_owner_batch_max,
                                   config.gpu_query_slots});
mapped_host_allocate(graph_invalidation_keys_host, d_graph_invalidation_keys,
                     graph_invalidation_capacity,
                     "cudaHostAlloc(navigation graph invalidation staging)");
mapped_host_allocate(delta_supersede_updates_host, d_delta_supersede_updates,
                     delta_command_capacity,
                     "cudaHostAlloc(navigation delta supersede staging)");
mapped_host_allocate(delta_override_updates_host, d_delta_override_updates,
                     delta_command_capacity,
                     "cudaHostAlloc(navigation delta override staging)");
mapped_host_allocate(delta_durable_updates_host, d_delta_durable_updates,
                     delta_command_capacity,
                     "cudaHostAlloc(navigation delta durable staging)");
mapped_host_allocate(resident_pq_erase_updates_host,
                     d_resident_pq_erase_updates,
                     delta_command_capacity,
                     "cudaHostAlloc(resident dynamic PQ erase staging)");
mapped_host_allocate(dynamic_route_updates_host,
                     d_dynamic_route_updates,
                     dynamic_route_capacity,
                     "cudaHostAlloc(dynamic query route staging)");
mapped_host_allocate(dynamic_route_code_updates_host,
                     d_dynamic_route_code_updates,
                     static_cast<size_t>(dynamic_route_capacity) *
                       index.layout.code_bytes,
                     "cudaHostAlloc(dynamic query route code staging)");
if (graph_cache_slots != 0) {
  check_cuda(cudaMemset(d_graph_cache_states, 0,
                        static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
             "cudaMemset(navigation cache states)");
  check_cuda(cudaMemset(d_graph_cache_readers, 0,
                        static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
             "cudaMemset(navigation cache readers)");
  check_cuda(cudaMemset(d_graph_cache_victims, 0,
                        static_cast<size_t>(graph_cache_sets) * sizeof(u32)),
             "cudaMemset(navigation cache victims)");
  check_cuda(cudaMemset(d_graph_admission_keys, 0xff,
                        static_cast<size_t>(graph_admission_sets) *
                          kCacheAdmissionWays * sizeof(u64)),
             "cudaMemset(navigation admission keys)");
  check_cuda(cudaMemset(d_graph_admission_victims, 0,
                        static_cast<size_t>(graph_admission_sets) * sizeof(u32)),
             "cudaMemset(navigation admission victims)");
}
const u64 initial_cache_generation = 1;
check_cuda(cudaMemcpy(d_graph_cache_generation, &initial_cache_generation,
                      sizeof(initial_cache_generation), cudaMemcpyHostToDevice),
           "cudaMemcpy(navigation cache generation)");
```

要点：

- graph cache 是一个**组相联**的 cache：`graph_cache_sets` 个 set、每个 set `config.gpu_adjacency_cache_ways` 路，总槽位 `graph_cache_slots = sets * ways`。每个槽位有 5 个并行的数组：`keys`（缓存行键，64 位）、`generations`（失效代次）、`timestamps`（LRU 时间戳）、`states`（状态机：empty/filling/ready/stale/fill-invalidated）、`readers`（活跃读者计数，RCU 回收用）。这套状态机与 anchor 图路由的状态机一致，第 19 课（RDMA cache）会详细讲。
- `d_graph_admission_keys` 是 admission filter（布隆过滤器替代品）：`graph_admission_sets * kCacheAdmissionWays` 个 64 位键，初始化为 `0xff...ff`（表示空）。它的作用是在请求 GPUNetIO 拉图前先做一次廉价检查，避免重复请求已经在缓存里的节点。`kCacheAdmissionWays == 4`（`cuda_helpers.hh:15`）。
- `d_graph_cache_generation` 是一个单 `u64`，初始为 `1`——任何全局失效会递增它，cache 命中时若发现 generation 不匹配则视为 miss。第 16 课（RCU 回收）会用到。
- 接下来一串 `mapped_host_allocate` 是**delta 命令的 staging 缓冲**：`delta_command_capacity = max(1, storage_owner_batch_max, gpu_query_slots)`。这些是 CPU 写入 → GPU 读取的 mapped pinned 内存，用于 `upload_mutations`（第 15 课）把 delta 发布命令的载荷（`DeltaSupersedeUpdate` / `DeltaOverrideUpdate` / `DeltaDurableUpdate` / `ResidentPqEraseUpdate` / `DynamicRouteUpdate` + 对应 PQ 码）传给 control-delta CTA。`graph_invalidation_keys_host` 也是 mapped，用于发布 delta 时告诉 kernel 哪些图缓存行要失效。

`exact_cache` 部分对称（`construction.cc:677`–`707`），只是缓存内容是"精确向量"而非"邻接表"，admission key 是 `u32` 而非 `u64`（因为 exact 缓存的键是 ordinal），其余结构与 graph cache 完全一致，不再赘述。

### 6. 结果缓冲（mapped，`construction.cc:709`–`723`）

```cpp
// construction.cc:709-723
const size_t result_elements = static_cast<size_t>(query_slots) * result_capacity;
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_ids_host),
                         result_elements * sizeof(u32),
                         cudaHostAllocMapped | cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation result ids)");
check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_ids),
                                    result_ids_host, 0),
           "cudaHostGetDevicePointer(GPU navigation result ids)");
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_distances_host),
                         result_elements * sizeof(f32),
                         cudaHostAllocMapped | cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation result distances)");
check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_distances),
                                    result_distances_host, 0),
           "cudaHostGetDevicePointer(GPU navigation result distances)");
```

结果数组用 `cudaHostAllocMapped | cudaHostAllocPortable` 而不是 `cudaMalloc`——这样 kernel 写完 `d_result_ids`，CPU 侧的 completion 线程不用做 `cudaMemcpy` 就能直接通过 `result_ids_host` 读到（PCIe 一致性）。`result_capacity = max(config.k, config.gpu_final_rerank_width)`（第 12 课算过）。每个 query slot 一份独立的结果缓冲，避免 query 之间互相覆盖。

### 7. delta / resident-PQ / dynamic-route 设备表（`construction.cc:725`–`798`）

这一段把第 12 课算好容量的所有 delta 相关设备表全部 `cudaMalloc` 出来并初始化。

```cpp
// construction.cc:725-748
device_allocate(d_delta_records, delta_capacity, "cudaMalloc(navigation delta records)");
device_allocate(d_delta_vectors,
                static_cast<size_t>(delta_capacity) * VamanaNode::vector_bytes(),
                "cudaMalloc(navigation delta vectors)");
if (budget.delta_code_bytes !=
    static_cast<u64>(delta_capacity) * this->code_bytes) {
  throw std::logic_error("GPU delta-code budget does not match the PQ code width");
}
device_allocate(d_delta_pq_codes,
                static_cast<size_t>(budget.delta_code_bytes),
                "cudaMalloc(PQ delta codes)");
mapped_host_allocate(delta_staging_slots_host, d_delta_staging_slots,
                     delta_command_capacity,
                     "cudaHostAlloc(navigation delta slot staging)");
mapped_host_allocate(delta_staging_records_host, d_delta_staging_records,
                     delta_command_capacity,
                     "cudaHostAlloc(navigation delta record staging)");
mapped_host_allocate(delta_staging_vectors_host, d_delta_staging_vectors,
                     static_cast<size_t>(delta_command_capacity) *
                       VamanaNode::vector_bytes(),
                     "cudaHostAlloc(navigation delta vector staging)");
device_allocate(d_delta_encode_scratch,
                static_cast<size_t>(delta_command_capacity) * config.dim,
                "cudaMalloc(navigation delta encode scratch)");
device_allocate(d_delta_next, delta_capacity, "cudaMalloc(navigation delta links)");
device_allocate(d_delta_prev, delta_capacity,
                "cudaMalloc(navigation delta reverse links)");
device_allocate(d_delta_remote_positions, delta_capacity,
                "cudaMalloc(navigation delta remote positions)");
```

要点：

- `d_delta_records` / `d_delta_vectors` / `d_delta_pq_codes` 是 delta 三件套（记录、原始向量、PQ 码），每个 `delta_capacity` 槽。`DeviceDeltaRecord` 结构见 `persistent_kernel.hh:58`–`68`，包含 `id`、`generation`、`flags`、`base_ordinal`、`epoch`、`superseded_epoch`、`remote_node`、`anchor_bucket`、`resident_pq_slot`。
- 那条 `delta_code_bytes != delta_capacity * code_bytes` 的 `logic_error` 是个不变量检查：`memory_budget` 算出来的 `delta_code_bytes` 必须正好等于 `delta_capacity * code_bytes`，否则 PQ 码布局就不对。
- `delta_staging_*` 是 mapped pinned 的 staging（CPU 写 → GPU 读），control-delta CTA 从这里读取一批 delta 命令的载荷。`delta_encode_scratch` 是 device 内存，用于 kernel 对 delta 向量做 OPQ 变换与 PQ 编码的 scratch（见 `runtime.cuh:303`–`368`）。
- `d_delta_next` / `d_delta_prev` 是 delta 链表的前后向指针，按 anchor bucket 组织成双向链表。`d_delta_remote_positions` 记录每个 delta slot 在 `d_delta_remote_keys/slots` 哈希表中的位置，用于 O(1) 删除。

```cpp
// construction.cc:754-783
device_allocate(d_base_override_keys, delta_table_capacity,
                "cudaMalloc(navigation override keys)");
device_allocate(d_base_override_epochs, delta_table_capacity,
                "cudaMalloc(navigation override epochs)");
device_allocate(d_permanent_override_bits, permanent_override_words,
                "cudaMalloc(navigation permanent override bits)");
device_allocate(d_delta_remote_keys, delta_table_capacity,
                "cudaMalloc(navigation delta remote keys)");
device_allocate(d_delta_remote_slots, delta_table_capacity,
                "cudaMalloc(navigation delta remote slots"));
device_allocate(d_resident_pq_codes,
                static_cast<size_t>(resident_pq_capacity) * code_bytes,
                "cudaMalloc(resident dynamic PQ codes)");
device_allocate(d_resident_pq_keys, resident_pq_table_capacity,
                "cudaMalloc(resident dynamic PQ keys)");
device_allocate(d_resident_pq_slots, resident_pq_table_capacity,
                "cudaMalloc(resident dynamic PQ slots)");
device_allocate(d_resident_pq_positions, resident_pq_capacity,
                "cudaMalloc(resident dynamic PQ positions)");
check_cuda(cudaMemset(d_resident_pq_keys, 0,
                      static_cast<size_t>(resident_pq_table_capacity) *
                        sizeof(u64)),
           "cudaMemset(resident dynamic PQ keys)");
check_cuda(cudaMemset(d_resident_pq_slots, 0xff,
                      static_cast<size_t>(resident_pq_table_capacity) *
                        sizeof(u32)),
           "cudaMemset(resident dynamic PQ slots)");
check_cuda(cudaMemset(d_resident_pq_positions, 0xff,
                      static_cast<size_t>(resident_pq_capacity) * sizeof(u32)),
           "cudaMemset(resident dynamic PQ positions)");
```

这里有两张哈希表（开放寻址，线性探测）：

1. **base override 表**（`d_base_override_keys` / `d_base_override_epochs`）：键是 `base_ordinal`（u32），值是 `epoch`（u64）。当 delta 覆盖了某个 base 节点，就在这张表里登记"该 base 节点的有效 epoch"。查询时若 base 节点的 epoch 小于表中的 epoch，则视为已被 supersede。`d_permanent_override_bits` 是一个 bitmap，标记哪些 base ordinal 已经被"永久"覆盖（即对应 delta 已经 durable，不会再回滚）。第 10 课、第 15 课详细讲。
2. **delta remote 表**（`d_delta_remote_keys` / `d_delta_remote_slots`）：键是 `remote_node`（u64，高 16 位是 shard），值是 `delta_slot`（u32）。这是"远端节点 ID → delta slot"的反向索引，查询时根据远端节点快速定位到 delta。`kDeltaRemoteEmpty == 0`、`kDeltaRemoteTombstone == UINT64_MAX`（`persistent_kernel.hh:31`–`32`）。

resident-PQ 是"常驻 GPU 的动态 PQ 码"缓存，第 10 课讲过：当 delta 节点还没被 promote 到 base 时，它的 PQ 码存在这里，查询时直接 GPU 取，不用走 GPUNetIO。`d_resident_pq_keys` / `d_resident_pq_slots` 是"远端节点 → resident slot"的哈希表，`d_resident_pq_positions` 是反向索引（resident slot → 在 keys 表中的位置）。`resident_pq_table_capacity = next_power_of_two(resident_pq_capacity * 2)`（第 12 课算过），load factor 0.5。

```cpp
// construction.cc:784-797
device_allocate(d_delta_count, 1, "cudaMalloc(navigation delta count)");
device_allocate(d_dynamic_route_slots, dynamic_route_capacity,
                "cudaMalloc(dynamic query route slots)");
device_allocate(d_dynamic_route_pq_codes,
                static_cast<size_t>(dynamic_route_capacity) * code_bytes,
                "cudaMalloc(dynamic query route PQ codes"));
check_cuda(cudaMemset(d_dynamic_route_slots, 0,
                      static_cast<size_t>(dynamic_route_capacity) *
                        sizeof(DeviceDynamicRouteSlot)),
           "cudaMemset(dynamic query route slots)");
check_cuda(cudaMemset(d_dynamic_route_pq_codes, 0,
                      static_cast<size_t>(dynamic_route_capacity) *
                        code_bytes),
           "cudaMemset(dynamic query route PQ codes)");
clear_delta_device_state();
```

- `d_delta_count` 是单 `u32`，记录当前活跃 delta 数量。control-delta CTA 在每次发布完成后用 `atomicExch` 写入 `final_count`（`runtime.cuh:661`）。
- `d_dynamic_route_slots` 是动态路由表，每个 shard `kDynamicRouteSlotsPerShard == 8` 个 slot（`types.hh:85`），总容量 `dynamic_route_capacity = num_shards * 8`。每个 `DeviceDynamicRouteSlot` 包含 `command_id`、`epoch`、`remote_node`、`id`、`generation`、`shard`、`flags`、`sequence`，是 storage-canonical 自适应路由的核心数据结构（第 10 课讲过设计，第 15 课讲发布流程，`runtime.cuh:600`–`656` 讲发布时的双 sequence 写法）。
- `clear_delta_device_state()`（`lifecycle.cc:176`–`217`）把所有 delta 相关的 device 数组清零或填 `0xff`（空标记），保证 kernel 启动时看到的是干净的初始状态。这一步在 `construction.cc` 末尾和每次 delta reset 命令时都会调用，是构造期与运行期共用的复位函数。

### 8. stop / ready 标志与 CUDA stream（`construction.cc:800`–`838`）

```cpp
// construction.cc:800-822
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                         cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation stop staging)");
*stop_host = 0;
device_allocate(stop_device, 1, "cudaMalloc(GPU navigation stop)");
check_cuda(cudaMemset(stop_device, 0, sizeof(u32)),
           "cudaMemset(GPU navigation stop)");
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_disabled_host), sizeof(u32),
                         cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation direct failure staging)");
*direct_disabled_host = 0;
device_allocate(direct_disabled_device, 1,
                "cudaMalloc(GPU navigation direct failure flag)");
check_cuda(cudaMemset(direct_disabled_device, 0, sizeof(u32)),
           "cudaMemset(GPU navigation direct failure flag)");
check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_error_host), sizeof(i32),
                         cudaHostAllocPortable),
           "cudaHostAlloc(GPU navigation direct error staging)");
*direct_error_host = 0;
device_allocate(direct_error_device, 1,
                "cudaMalloc(GPU navigation direct error)");
check_cuda(cudaMemset(direct_error_device, 0, sizeof(i32)),
           "cudaMemset(GPU navigation direct error)");
```

三个标志位：

- `stop_host` / `stop_device`：kernel 主循环每个迭代开头都会读 `*params.stop`（`runtime.cuh:53`–`57`），非零则 `return`。CPU 侧通过 `cudaMemcpyAsync(stop_device, stop_host, ...)` 让 kernel 退出（`lifecycle.cc:307`–`311`）。`cudaHostAllocPortable`（非 mapped）意味着 CPU 写完要用 cudaMemcpy 才能让 device 看到——这是刻意的，避免 kernel 误读部分写入。
- `direct_disabled_host` / `direct_disabled_device`：GPUNetIO 直读路径被禁用的标志。owner warp 在 CQ 出错时 `atomicExch(params.direct_disabled, 1u)`（`runtime.cuh:928`），之后所有 direct batch 立即返回 `-EHOSTDOWN`。CPU 侧的 completion 线程会读 `direct_disabled_host` 判断是否需要回退到 bootstrap RDMA 路径。
- `direct_error_host` / `direct_error_device`：第一条 GPUNetIO 错误码。`atomicCAS(params.direct_error, 0, status)`（`runtime.cuh:927`）保留首错。

```cpp
// construction.cc:823-829
mapped_host_allocate(query_kernel_ready_host, d_query_kernel_ready, 1,
                     "cudaHostAlloc(GPU query kernel readiness)");
mapped_host_allocate(dispatcher_kernel_ready_host,
                     d_dispatcher_kernel_ready, 1,
                     "cudaHostAlloc(GPU dispatcher kernel readiness)");
mapped_host_allocate(control_kernel_ready_host, d_control_kernel_ready, 1,
                     "cudaHostAlloc(GPU control kernel readiness)");
```

三个 ready 计数器，每个一个 `u32`。**这三个用 `mapped_host_allocate`（mapped pinned）而非 portable**——因为 kernel 启动时 CPU 要忙等这三个值（见 `start_persistent_kernel`），mapped 让 CPU 直接通过 host 指针读到 device 的 `atomicAdd` 写入，无需 cudaMemcpy。`__threadfence_system()`（`runtime.cuh:32`）保证 kernel 端的写入对 host 可见。

```cpp
// construction.cc:830-838
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

四条 `cudaStreamNonBlocking` 流：

- `kernel_stream`：持久化搜索 kernel 在此流上 launch（`lifecycle.cc:243`）。
- `delta_stream`：delta 编码 kernel（`upload_mutations` 用，第 15 课）。
- `rdma_stream`：CPU posted RDMA（bootstrap、stop 信号、direct_disabled/direct_error 回读，见 `completion.cc:13`–`18`）。
- `route_refresh_stream`：anchor 路由图刷新（`refresh_anchor_graph_records`，第 16 课）。

`NonBlocking` 让这些流不与默认流（NULL stream）同步，避免构造期临时 kernel（如 `launch_gather_anchor_codes`）意外阻塞持久化 kernel。

### 9. block 角色计数与 SM 容量校验（`construction.cc:839`–`862`）

这是本课的核心之一：决定持久化 kernel 的 grid 形状。

```cpp
// construction.cc:839-842
cudaDeviceProp properties{};
check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
           "cudaGetDeviceProperties(GPU navigation)");
gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
```

先取设备属性。`gpu_clock_khz` 用于把配置里的超时换算成时钟周期（第 19 课、第 22 课）。

```cpp
// construction.cc:843-854
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
```

- `kPersistentQueryThreads == 256`（`persistent_kernel.hh:23`），所以 `owner_warps_per_block = 256 / 32 = 8` 个 warp 一个 block。`runtime.cuh:715` 里的 `max_warps_per_block == 8` 正好对应——这是 direct owner kernel 的硬上限。
- `owner_kernel_blocks = ceil(direct_batch_queue_count / 8)`：每 block 8 个 warp，每个 warp 服务一条 QP。例如 `direct_batch_queue_count == 32`（4 个 shard × 8 QPs/shard）则 `owner_kernel_blocks == 4`。
- `resident_blocks = multiProcessorCount`（SM 数）。`control_blocks == 2` 是 dispatcher 与 control-delta 各占一个 block（见下文 `runtime.cuh:24`–`25`）。
- 校验：`owner_kernel_blocks + 2 < SM 数`，否则 query block 一个都分不到，直接抛错。这保证持久化 grid 至少有 1 个 query block。

```cpp
// construction.cc:855-862
const u64 requested_blocks = static_cast<u64>(
  std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
const u64 useful_blocks = std::max<u64>(1, config.num_threads);
const u64 resident_query_blocks =
  resident_blocks - owner_kernel_blocks - control_blocks;
kernel_blocks = static_cast<u32>(std::min({
  static_cast<u64>(query_slots), requested_blocks, useful_blocks,
  resident_query_blocks}));
```

`kernel_blocks` 是 query CTA 数，取四个上限的最小：

1. `query_slots`：每个 slot 至少一个 query CTA 服务，多了浪费。
2. `requested_blocks = SM 数 * gpu_persistent_blocks_per_sm`：用户配置的"每 SM 几个持久化 block"。
3. `useful_blocks = max(1, config.num_threads)`：用户线程数（CPU 侧并发上限），再多 query CTA 也没用，CPU 喂不满。
4. `resident_query_blocks = SM 数 - owner_kernel_blocks - 2`：SM 减去 owner 与 control 占用的，剩下的才能给 query。

这个公式保证 grid 不会 oversubscribe SM——所有 block 都能同时常驻，不会因抢占导致 owner warp 被踢出（那会让 GPUNetIO CQ 永远没人 poll，最终超时）。

### 10. `PersistentKernelParams` 装配（`construction.cc:864`–`1014`）

这是本课最长的一段，但内容直接：把前面所有 `d_*` 指针、容量、配置塞进一个 `PersistentKernelParams` 结构体（定义在 `persistent_kernel.hh:81`–`223`）。这个结构体会被**按值**传给 kernel（`persistent_kernel.cu:10` 的 `<<<blocks, threads, 0, stream>>>(params)`），所以它必须能在 constant/参数内存里装下——结构体大约 600 字节，刚好在 CUDA 单 kernel 参数上限（通常 4KB）内。

逐块讲：

#### 10.1 ring 视图（`construction.cc:865`–`879`）

```cpp
// construction.cc:865-879
kernel_params = PersistentKernelParams{
  .submissions = submissions.device_view(),
  .device_submissions = {
    .enqueue_position = reinterpret_cast<unsigned long long*>(
      d_query_dispatch_enqueue),
    .dequeue_position = reinterpret_cast<unsigned long long*>(
      d_query_dispatch_dequeue),
    .sequences = reinterpret_cast<unsigned long long*>(
      d_query_dispatch_sequences),
    .entries = d_query_dispatch_entries,
    .capacity = query_dispatch_capacity,
    .mask = query_dispatch_capacity - 1,
  },
  .completions = completions.device_view(),
  .delta_submissions = delta_submissions.device_view(),
  .delta_completions = delta_completions.device_view(),
```

- `submissions`：`MappedRing<QueryDescriptor>::device_view()`（`mapped_ring.hh:98`），host→device 方向，admission 线程 push、dispatcher CTA pop。
- `device_submissions`：上面 5 节建的纯 device ring，dispatcher CTA push、query CTA pop。
- `completions`：`MappedRing<CompletionDescriptor>::device_view()`，device→host 方向，query CTA push、completion 线程 pop。
- `delta_submissions` / `delta_completions`：delta 命令的双向 ring，容量都是 8（`construction.cc:83`–`84`）。CPU 侧 `submit_delta_publication` push 命令，control-delta CTA pop 处理后 push 完成回执。

#### 10.2 索引元数据（`construction.cc:880`–`899`）

```cpp
// construction.cc:880-899
  .shards = d_shards,
  .num_shards = static_cast<u32>(index.shards.size()),
  .pq_codes = d_pq_codes,
  .opq_matrix = d_opq_matrix,
  .pq_centroids = d_pq_centroids,
  .entry_points = d_entry_points,
  .entry_point_count = static_cast<u32>(entry_handles.size()),
  .num_nodes = static_cast<u32>(index.layout.num_nodes),
  .medoid_ordinal = index.layout.medoid_ordinal,
  .dim = config.dim,
  .pq_subquantizers = pq_model.subquantizers,
  .pq_subvector_dim = pq_model.subvector_dim(),
  .pq_code_bytes = pq_model.code_bytes(),
  .graph_entry_bytes = index.layout.graph_entry_bytes,
  .graph_degree = index.layout.graph_degree,
  .graph_shard_bits = index.layout.graph_shard_bits,
  .node_meta_offset = 0,
  .node_record_bytes = node_record_bytes,
  .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
  .vector_dtype = static_cast<u32>(config.resolved_vector_dtype()),
```

这里全是 12 课准备好的常量与 device 指针。`node_meta_offset = 0` 是 Vamana 节点记录内 meta 段的偏移（schema-15 固定为 0，第 7 课）。`graph_shard_bits` 来自 `VamanaNode::HOT_GRAPH_SHARD_BITS`（第 12 课校验过）。`vector_bytes` 是单个 delta 向量的字节数（与 dtype 无关，由 `VamanaNode::vector_bytes()` 决定）。

#### 10.3 查询/搜索参数（`construction.cc:900`–`907`）

```cpp
// construction.cc:900-907
  .traversal_beam_width = config.gpu_traversal_beam_width,
  .final_rerank_width = config.gpu_final_rerank_width,
  .entry_seed_count = config.gpu_entry_seed_count,
  .exact_width = exact_width,
  .max_expansions = config.gpu_max_expansions,
  .prefetch_depth = config.gpu_graph_prefetch_depth,
  .visited_capacity = visited_capacity,
  .query_slots = query_slots,
```

搜索行为的可调参数。`exact_width = kPersistentMaxExact == 256`（第 12 课），`visited_capacity` 由 memory_budget 算出。`max_expansions` 是单次查询最多扩展多少节点，防止病态查询拖死 GPU。第 18 课（候选评分）、第 20 课（查询遍历主循环）会详细用这些。

#### 10.4 GPUNetIO 直读通道（`construction.cc:908`–`922`）

```cpp
// construction.cc:908-922
  .direct_region_count = direct_view.remote_region_count,
  .direct_qps_per_node = direct_view.qps_per_node,
  .direct_local_mkey = direct_view.local_mkey,
  .direct_local_iova_base = direct_view.local_iova_base,
  .direct_timeout_ns = 20000000ULL,
  .direct_regions = reinterpret_cast<const DirectRemoteRegion*>(direct_view.remote_regions),
  .direct_qps = direct_view.qp_array,
  .direct_qp_locks = direct_view.qp_locks,
  .direct_batch_queues = d_direct_batch_queues,
  .direct_batch_statuses = d_direct_batch_statuses,
  .direct_batch_queue_count = direct_batch_queue_count,
  .direct_owner_phases = d_direct_owner_phases,
  .direct_dump = direct_view.dump,
  .direct_disabled = direct_disabled_device,
  .direct_error = direct_error_device,
```

这一块是 GPUNetIO 直读路径的全部 device 句柄。`direct_local_mkey` 与 `direct_local_iova_base` 是 GPUDirect RDMA 的本地内存键与基址（DOCA gpu epoll mode 需要的 IOVA 计算，第 22 课详讲）。`direct_timeout_ns = 20ms` 是单次 CQ poll 的超时，owner warp 在 `poll_direct_cq` 里用它判断是否放弃（`runtime.cuh:917`）。`direct_dump` 是 DOCA GPUNetIO 的 dump WQE 目标缓冲，用于需要 dump 模式时校验 WQE 内容。`direct_regions` 是每个远端内存节点的 `{address, rkey}` 数组，`DirectRemoteRegion` 与 `format::ShardRegion` 的远端部分二进制兼容。

`direct_qp_locks` 字段当前在 unified dispatch 模式下不用（因为每个 warp 独占一条 QP，无需加锁），但在 probe kernel（`launch_gpunetio_locked_read_probe`）里会用到——见第 22 课。

#### 10.5 delta / resident-PQ / override 表（`construction.cc:923`–`958`）

```cpp
// construction.cc:923-958
  .delta_records = d_delta_records,
  .delta_vectors = d_delta_vectors,
  .delta_pq_codes = d_delta_pq_codes,
  .delta_staging_slots = d_delta_staging_slots,
  .delta_staging_records = d_delta_staging_records,
  .delta_staging_vectors = d_delta_staging_vectors,
  .delta_encode_scratch = d_delta_encode_scratch,
  .delta_next = d_delta_next,
  .delta_prev = d_delta_prev,
  .delta_remote_positions = d_delta_remote_positions,
  .delta_bucket_heads = d_delta_bucket_heads,
  .delta_count = d_delta_count,
  .delta_capacity = delta_capacity,
  .base_override_keys = d_base_override_keys,
  .base_override_epochs = d_base_override_epochs,
  .base_override_capacity = delta_table_capacity,
  .permanent_override_bits = d_permanent_override_bits,
  .permanent_override_words = permanent_override_words,
  .delta_remote_keys = d_delta_remote_keys,
  .delta_remote_slots = d_delta_remote_slots,
  .delta_remote_capacity = delta_table_capacity,
  .resident_pq_codes = d_resident_pq_codes,
  .resident_pq_keys = d_resident_pq_keys,
  .resident_pq_slots = d_resident_pq_slots,
  .resident_pq_positions = d_resident_pq_positions,
  .resident_pq_capacity = resident_pq_capacity,
  .resident_pq_table_capacity = resident_pq_table_capacity,
  .delta_supersede_updates = d_delta_supersede_updates,
  .delta_override_updates = d_delta_override_updates,
  .delta_durable_updates = d_delta_durable_updates,
  .resident_pq_erase_updates = d_resident_pq_erase_updates,
  .dynamic_route_updates = d_dynamic_route_updates,
  .dynamic_route_code_updates = d_dynamic_route_code_updates,
  .dynamic_route_slots = d_dynamic_route_slots,
  .dynamic_route_pq_codes = d_dynamic_route_pq_codes,
  .dynamic_route_capacity = dynamic_route_capacity,
```

全是从前面 6/7 节建好的 device 指针与容量。注意 `delta_staging_*` 这几个是 mapped pinned 的 device 端地址（`mapped_host_allocate` 写入 `d_*`），kernel 通过它们读 CPU 端 `upload_mutations` 写入的 delta 命令载荷。`base_override_capacity` 与 `delta_remote_capacity` 都是 `delta_table_capacity`（同一张表的两个用途，第 12 课算容量时已说明）。第 15 课（增量发布）会详细讲 control-delta CTA 如何消费这些字段。

#### 10.6 anchor / 路由（`construction.cc:959`–`969`）

```cpp
// construction.cc:959-969
  .graph_invalidation_keys = d_graph_invalidation_keys,
  .anchor_vectors = d_anchor_vectors,
  .anchor_handles = d_anchor_handles,
  .anchor_pq_codes = d_anchor_pq_codes,
  .anchor_graph_keys = d_anchor_graph_keys,
  .anchor_graph_records = d_anchor_graph_records,
  .anchor_graph_states = d_anchor_graph_states,
  .anchor_graph_readers = d_anchor_graph_readers,
  .anchor_graph_count = anchor_graph_count,
  .anchor_count = anchor_table.count(),
  .delta_anchor_probes = config.gpu_delta_anchor_probes,
```

`anchor_vectors` 是转置后的 anchor 向量矩阵（第 1 节讲过转置）。`anchor_graph_records` 是第 12 课 `stream_anchor_graph_to_gpu` 拉进来的邻接表记录，按 `anchor_graph_keys` 顺序排列。`delta_anchor_probes` 是 delta 查找时最多 probe 多少个 anchor bucket（限制最坏情况复杂度，第 10 课）。

#### 10.7 graph cache / exact cache（`construction.cc:970`–`1011`）

```cpp
// construction.cc:970-1011
  .stop = stop_device,
  .graph_cache = d_graph_cache,
  .graph_scratch = d_graph_scratch,
  .graph_cache_keys = d_graph_cache_keys,
  .graph_cache_generations = d_graph_cache_generations,
  .graph_cache_timestamps = d_graph_cache_timestamps,
  .graph_cache_states = d_graph_cache_states,
  .graph_cache_readers = d_graph_cache_readers,
  .graph_cache_victims = d_graph_cache_victims,
  .graph_admission_keys = d_graph_admission_keys,
  .graph_admission_victims = d_graph_admission_victims,
  .graph_admission_sets = graph_admission_sets,
  .graph_cache_generation = d_graph_cache_generation,
  .graph_cache_sets = graph_cache_sets,
  .graph_cache_ways = config.gpu_adjacency_cache_ways,
  .graph_cache_ttl_ns = static_cast<u64>(
    config.gpu_graph_cache_ttl_us == 0
      ? config.update_visibility_us
      : std::min(config.gpu_graph_cache_ttl_us,
                 config.update_visibility_us)) * 1000,
```

注意 `graph_cache_ttl_ns`：如果用户没配 `gpu_graph_cache_ttl_us`（为 0），则用 `update_visibility_us` 作为 TTL；否则取二者较小值。这保证缓存不会比"delta 可见性延迟"活得更久，避免缓存里看到陈旧的邻接表。第 19 课（RDMA cache）会详细讲 cache 状态机与 TTL。

`.stop = stop_device` 是唯一在 launch 前已经设好、运行期会被 CPU 改写的字段（通过 `cudaMemcpyAsync`）。

exact cache 部分（`construction.cc:1001`–`1011`）与 graph cache 对称，字段含义同 5 节。

```cpp
// construction.cc:1012-1014
  .result_ids = d_result_ids,
  .result_distances = d_result_distances,
};
```

最后是结果缓冲的 device 句柄（mapped pinned 的 device 端）。

**注意**：装配时**没有**填 `direct_owner_block_count`、`query_block_count`、`query_kernel_ready_count`、`dispatcher_kernel_ready_count`、`control_kernel_ready_count` 这五个字段——它们在 `start_persistent_kernel` 里临时补上（见下节）。这是因为这些字段与具体某次 launch 的 grid 形状绑定，而 `kernel_params` 是 `Impl` 成员、可能在重启 kernel 时复用，所以每次 launch 前现填。

### 11. `start_persistent_kernel`：unified grid launch 与 ready barrier（`lifecycle.cc:219`–`302`）

虽然这个函数在 `lifecycle.cc` 里，但它是 `Impl::Impl` 构造函数末尾 `construction.cc:1015` 调用的，是构造期的最后一步实质工作。

```cpp
// lifecycle.cc:219-235
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

启动前清零所有运行期标志：stop、direct_disabled、direct_error、owner phases、三个 ready 计数。`(void)cudaGetLastError()` 清掉前面残留的异步错误状态（构造期可能有过非致命的异步错误）。`std::atomic_thread_fence` 保证清零在 launch 之前对 device 可见。

```cpp
// lifecycle.cc:236-245
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

**unified dispatch 模式**：把 owner / query / dispatcher / control-delta 全部塞进一个 `persistent_search_kernel` 的 grid。`total_blocks = owner_kernel_blocks + kernel_blocks + 2`——那个 `+2` 就是 dispatcher 与 control-delta 各一个 block。kernel 内部用 `blockIdx.x` 划分角色（见下文 runtime.cuh 段）。

为什么用统一 grid 而不是分多次 launch？因为持久化 kernel 一旦启动就不退出（直到 stop），分多次 launch 会让 owner warp 与 query CTA 在不同 kernel 里、无法共享 `__shared__` 内存与同步原语；更重要的是 GPUNetIO 的 QP 是"独占资源"，owner warp 必须与 query CTA 同生命周期，否则 QP 会在 owner warp 被换出时超时。统一 grid 保证所有角色 block 同生共死。

`kPersistentQueryThreads == 256` 是所有 block 的统一线程数——query CTA 用满 256 线程做评分，owner block 用 8 个 warp（256/32）服务 8 条 QP，dispatcher 与 control-delta 实际只用少量线程但保持 block 形状一致以简化 launch。

#### 11.1 ready barrier

```cpp
// lifecycle.cc:247-294
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
      u32 first_owner_phase = 0;
      for (u32 qp = 0; qp < direct_batch_queue_count; ++qp) {
        const u32 phase =
          *reinterpret_cast<volatile u32*>(direct_owner_phases_host + qp);
        if (phase != 1) {
          first_owner_phase = phase;
          break;
        }
      }
      *stop_host = 1;
      (void)cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                            cudaMemcpyHostToDevice, rdma_stream);
      (void)cudaStreamSynchronize(rdma_stream);
      (void)cudaStreamSynchronize(kernel_stream);
      throw std::runtime_error(
        "unified GPU grid did not become fully resident: owners=" +
        std::to_string(ready_owners) + "/" +
        std::to_string(direct_batch_queue_count) +
        " queries=" + std::to_string(ready_queries) + "/" +
        std::to_string(kernel_blocks) +
        " dispatcher=" + std::to_string(ready_dispatchers) + "/1" +
        " control=" + std::to_string(ready_controls) + "/1" +
        " first_owner_phase=" + std::to_string(first_owner_phase));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  kernel_running = true;
```

这是构造期最关键的"启动就绪 barrier"。3 秒超时内，必须达成四个条件：

1. **每个 owner warp 的 `direct_owner_phases_host[qp] == 1`**：`runtime.cuh:771`–`774` 里 owner warp 在完成参数校验、拿到 QP 句柄后写 `params.direct_owner_phases[warp] = 1` 并 `__threadfence_system()`。`ready_owners == direct_batch_queue_count` 表示所有 QP 都有 warp 在 poll。
2. **`query_kernel_ready_host == kernel_blocks`**：每个 query CTA 的 thread 0 在 `runtime.cuh:28`–`33` 里 `atomicAdd(params.query_kernel_ready_count, 1u)` 后 `__threadfence_system()`。`ready_queries == kernel_blocks` 表示所有 query CTA 都进入了主循环。
3. **`dispatcher_kernel_ready_host == 1`**：dispatcher CTA 同样 `atomicAdd`。只期待 1 个。
4. **`control_kernel_ready_host == 1`**：control-delta CTA 同样。

为什么需要这个 barrier？因为持久化 kernel 是"启动后永不退出"的，如果某个 block 因 SM 调度问题没起来（例如 grid 太大），构造函数返回后 admission 线程就会往 ring 里 push 查询，但没有 CTA 在 pop，查询会永远卡住。barrier 保证构造函数返回时所有角色 block 都已经进入主循环。

**失败路径**（`lifecycle.cc:268`–`292`）：超时后：

1. 记录第一个未就绪 owner warp 的 phase 值（用于错误信息诊断——phase `0x100 | invalid` 表示参数校验失败，`0x200` 表示 QP 句柄为空，见 `runtime.cuh:735`–`759`）。
2. `*stop_host = 1` 并 `cudaMemcpyAsync` 到 `stop_device`，让 kernel 退出。
3. 同步 `rdma_stream`（确保 stop 信号到达）与 `kernel_stream`（确保 kernel 真的退出了）。
4. 抛 `runtime_error`，构造函数失败，`Impl::~Impl` 走回滚路径（见第 11 课、本课下节）。

成功则 `kernel_running = true`，打印一行启动日志：

```cpp
// lifecycle.cc:296-301
  std::cerr << "[gpu-search] unified persistent CTAs=" << owner_kernel_blocks
            << "-owner+" << kernel_blocks
            << "-query+1-dispatch+1-control"
            << " QP-owner-warps=" << direct_batch_queue_count
            << " threads/CTA=" << kPersistentQueryThreads
            << " query_slots=" << query_slots << '\n';
```

### 12. kernel 内部的角色块划分（`runtime.cuh:11`–`51`）

这段虽然在 kernel 文件里，但它是理解 `start_persistent_kernel` 的 grid 划分所必需的，本课一并讲。

```cpp
// runtime.cuh:11-37
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
```

`blockIdx.x` 划分：

- `[0, direct_owner_block_count)`：owner block，进入 `direct_read_owner_loop`，每个 block 8 个 warp、每 warp 服务一条 QP，**return 后退出 kernel**（不进入主循环）。
- `[direct_owner_block_count, direct_owner_block_count + query_block_count)`：query CTA，`enable_queries = true`。
- `blockIdx.x == direct_owner_block_count + query_block_count`：dispatcher CTA，`enable_dispatcher = true`。
- `blockIdx.x == direct_owner_block_count + query_block_count + 1`：control-delta CTA，`enable_delta = true`。
- 其他：return（不应发生，因为 `total_blocks` 正好是这四段之和）。

每个非 owner block 的 thread 0 在进入主循环前 `atomicAdd` 对应的 ready 计数器，然后 `__threadfence_system()`——这一行就是 `start_persistent_kernel` ready barrier 在等的东西。`__threadfence_system` 保证 `atomicAdd` 的写入对 host 可见（跨 host/device 的内存序，CUDA 的 system scope）。

#### 12.1 dispatcher CTA 的角色

```cpp
// runtime.cuh:58-82
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

dispatcher 是 admission 与 query CTA 之间的中继：从 `submissions`（host→device MappedRing）pop，push 到 `device_submissions`（纯 device ring）。它存在的意义是把"host PCIe 写入"与"query CTA 热路径"解耦——query CTA 只需要从 device ring pop，不用碰 host pinned 内存。`dispatch_pending` 是个单 slot 缓冲，避免 `submissions` pop 成功但 `device_submissions` push 失败时丢消息。idle 时指数退避（`256 → 16384` 周期）。

#### 12.2 control-delta CTA 的角色

control-delta CTA（`enable_delta == true`）在主循环里先尝试 `device_ring_try_pop(params.delta_submissions, delta_descriptor)`，pop 到 delta 命令后做大量参数校验（`runtime.cuh:90`–`167`），然后执行 reset 或正常发布流程（`runtime.cuh:171`–`675`）：拷贝 staging 到正式表、OPQ 变换、PQ 重编码、resident-PQ 插入、delta remote 表插入、bucket 链表插入、动态路由 sequence 双写、`delta_count` 发布、完成回执 push。这一整套是第 15 课的主题，本课只讲构造期给它准备好了什么参数。

注意 `enable_delta` 在 unified dispatch 模式下**只有 `role_block == query_block_count + 1` 这一个 block 为 true**——也就是说 control-delta 是单 CTA 串行处理 delta 命令。这是刻意的：delta 发布涉及大量全局表的写操作，多 CTA 并发会需要复杂的锁，而 delta 命令频率远低于 query，单 CTA 足够。

#### 12.3 query CTA 的角色

query CTA（`enable_queries == true`）在主循环里从 `device_submissions`（或回退到 `submissions`）pop 查询，调用 `process_query`（第 20 课）。本课不展开 `process_query`，只强调构造期为它准备好的工作区：`d_queries` / `d_transformed_queries` / `d_query_luts` / `d_navigation_candidate_*` / `d_visited` / `d_result_*`，每个 query slot 一份。

### 13. 后台线程启动（`construction.cc:1016`–`1018`）

```cpp
// construction.cc:1015-1018
  start_persistent_kernel();
  admission_thread = std::thread([this] { admission_loop(); });
  completion_thread = std::thread([this] { completion_loop(); });
  maintenance_thread = std::thread([this] { maintenance_loop(); });
}
```

构造函数最后启动三条后台线程：

- `admission_thread` → `admission_loop`：从用户 API 接查询、分配 slot、写入 `d_query_input`、push 到 `submissions` ring。第 14 课详讲。
- `completion_thread` → `completion_loop`：从 `completions` ring pop 完成回执、读 `result_ids_host` / `result_distances_host`、resolve promise 给用户。同时监控 `direct_disabled_host` / `direct_error_host`，在直读路径失败时上报。第 14 课详讲。
- `maintenance_thread` → `maintenance_loop`：周期性做存储路由刷新、delta durable 推进、resident-PQ 回收、anchor 图刷新、storage reclaim ack 发布。第 15 课、第 16 课详讲。

**为什么必须先 `start_persistent_kernel` 再启动线程？** 因为 admission 线程会立刻往 `submissions` ring push 查询——如果 kernel 还没起来，dispatcher CTA 就没人 pop，ring 会很快填满（容量 `query_slots * 2`），admission 会阻塞。ready barrier 保证 kernel 已经在主循环里 pop ring，admission 才能放心 push。

**为什么 completion 线程也排在 kernel 后面？** 因为 completion 线程会读 `direct_disabled_host` 判断直读路径状态——如果 kernel 没起来，`direct_disabled_host` 是 0 但没有 owner warp 在 poll CQ，completion 会误以为直读路径正常。kernel 起来后若 owner warp 起不来，会写 `direct_disabled = 1`，completion 才能正确回退。

**maintenance 线程排在最后**：它需要 `kernel_running == true` 才能做 storage reclaim ack 的发布（否则 ack 不会被 kernel 消费）、anchor 图刷新（需要 kernel 不在用 anchor_graph_records）。

### 14. 失败路径与回滚

构造函数里任何一步抛异常，C++ 异常机制会展开 `Impl::Impl`，已经分配的资源通过 `Impl::~Impl` 回滚。`~Impl` 在 `lifecycle.cc:322`–`466`：

- 先 `accepting.store(false)`、唤醒所有 cv、join `maintenance_thread`（若已启动）。
- `shutdown.store(true)`、join `admission_thread`（若已启动）。
- drain `pending_count`（最多等 `storage_owner_rpc_timeout_ms`）。
- 若 `kernel_running`：`*stop_host = 1` + cudaMemcpyAsync + 同步三条流，让 kernel 退出。
- `reject_all_pending`、`completion_thread.join()`。
- 销毁 4 条 stream。
- 一长串 `cudaFreeHost` / `device_free` 把所有 mapped pinned 与 device 内存释放，顺序与构造期分配相反。
- `control_bootstrapper.reset()`、`direct_transport.reset()`、若 `owns_remote_buffer` 则 `device_free(d_remote_buffer)`。

注意：在 GPUNetIO 模式下 `owns_remote_buffer == false`（`construction.cc:383`），`d_remote_buffer` 的所有权在 `direct_transport` 里，由 `direct_transport.reset()` 释放——避免 double free。

构造期失败最常见的几种：

- `cudaMalloc` 失败：`device_allocate` 抛 `runtime_error`，错误信息含 requested/free/total 字节数（`cuda_helpers.hh:42`–`49`）。
- `GpuNetioPersistentTransport` 构造失败：抛在 `construction.cc:376`–`381`，可能因为 DOCA 初始化失败、QP 建立失败、远端 MR 注册失败等——第 22 课详讲。
- `direct_batch_queue_count != estimated_direct_queue_count`：`construction.cc:557`，说明传输层实际建好的 QP 数与预算不一致，通常是配置 `gpu_rdma_qps` 与传输层协商结果不匹配。
- `owner_kernel_blocks + 2 >= multiProcessorCount`：`construction.cc:851`，SM 太少。
- ready barrier 超时：`lifecycle.cc:283`，最常见的原因是 grid 太大塞不进 SM、或 GPUNetIO QP 句柄为空（`runtime.cuh:754` 返回 phase `0x200`）。

---

## 关键数据结构与流程图

### 图 1：block 角色划分（unified dispatch 模式）

```
persistent_search_kernel<<<total_blocks, 256>>>
total_blocks = owner_kernel_blocks + kernel_blocks + 2

blockIdx.x 区间                                         角色
[0, owner_kernel_blocks)                                owner block
                                                        每 block 8 warp, 每 warp 服务 1 条 QP
                                                        直接进入 direct_read_owner_loop, 不回主循环
                                                        ready 信号: direct_owner_phases[warp] = 1

[owner_kernel_blocks,
 owner_kernel_blocks + kernel_blocks)                   query CTA (kernel_blocks 个)
                                                        从 device_submissions pop 查询
                                                        调用 process_query
                                                        ready 信号: atomicAdd(query_kernel_ready_count, 1)

owner_kernel_blocks + kernel_blocks                     dispatcher CTA (1 个)
                                                        submissions → device_submissions 中继
                                                        ready 信号: atomicAdd(dispatcher_kernel_ready_count, 1)

owner_kernel_blocks + kernel_blocks + 1                 control-delta CTA (1 个)
                                                        从 delta_submissions pop 命令
                                                        执行 reset / 发布 / promote
                                                        ready 信号: atomicAdd(control_kernel_ready_count, 1)
```

### 图 2：kernel launch 配置与 ready barrier 时序

```
CPU (Impl::Impl)                         GPU (persistent_search_kernel)
─────────────────                        ─────────────────────────────────
start_persistent_kernel()
  clear stop/disabled/error/ready flags
  fill direct_owner_phases_host = 0
  fence release
  launch_params = kernel_params
  launch_params.direct_owner_block_count = owner_kernel_blocks
  launch_params.query_block_count        = kernel_blocks
  launch_params.*_ready_count            = d_*_ready
  total_blocks = owner + query + 2
  launch_persistent_search(kernel_stream,
                           launch_params,
                           total_blocks, 256)  ──►  [所有 block 调度到 SM]
                                                          │
  ready_deadline = now + 3s                            每个 block:
  loop:                                                  ├─ owner block: 校验 → direct_owner_phases[warp]=1
    ready_owners = count(phases == 1)                   ├─ query CTA:   atomicAdd(query_kernel_ready_count,1)
    ready_queries = *query_kernel_ready_host            ├─ dispatcher:  atomicAdd(dispatcher_kernel_ready_count,1)
    ready_dispatchers = *dispatcher_kernel_ready_host   ├─ control:     atomicAdd(control_kernel_ready_count,1)
    ready_controls = *control_kernel_ready_host         └─ __threadfence_system()  ← 让 host 看到写入
    if all match: break
    if now > deadline:
      *stop_host = 1
      cudaMemcpyAsync(stop_device, ...)
      sync rdma_stream, kernel_stream
      throw "did not become fully resident"
    sleep 1ms
  kernel_running = true
  spawn admission/completion/maintenance threads
                                                       [所有 block 进入主循环, 开始 pop ring]
```

### 图 3：device ring 拓扑（构造期建立的 5 类 ring）

```
                host pinned (mapped)              device memory
                ─────────────────────             ──────────────

admission thread ──push──► submissions (MappedRing, h2d)
                                  │ device_view()
                                  ▼
                          [dispatcher CTA pop]
                                  │
                                  ▼ push
                          device_submissions (DeviceRingView, dev-only) ◄──┐
                                  │                                         │
                                  ▼ pop                                     │
                          [query CTA] ────push──► completions (MappedRing, d2h) ──► completion thread
                                  │
                                  ▼ push (GPUNetIO 请求)
                          direct_batch_queues[qp] (DeviceRingView, dev-only)
                                  │
                                  ▼ pop
                          [owner warp] ──DOCA GPUNetIO RDMA read──► remote storage

submit_delta_publication ──push──► delta_submissions (MappedRing, h2d)
                                  │
                                  ▼ pop
                          [control-delta CTA] ──push──► delta_completions (MappedRing, d2h) ──► delta waiter
```

五类 ring：

1. `submissions` / `completions`：MappedRing，admission↔dispatcher、query↔completion。容量 `query_slots * 2`。
2. `device_submissions`：纯 device ring，dispatcher↔query。容量 `next_power_of_two(query_slots * 2)`。
3. `direct_batch_queues`：纯 device ring 数组，每条 QP 一个，query↔owner。容量 64。
4. `delta_submissions` / `delta_completions`：MappedRing，CPU↔control-delta。容量 8。

为什么容量差异这么大？submissions/completions 与 device_submissions 容量大，因为查询吞吐高、需要 burst 容量；direct_batch_queue 容量小（64），因为每条 QP 串行处理、深度大于 64 会让 owner warp 来不及 poll CQ；delta ring 容量极小（8），因为 delta 命令低频、串行处理，深度大于 8 反而会让 control-delta CTA 处理时 admission 阻塞。

---

## 与其他模块的关系

- **第 3 课（并发原语与协程）**：`MappedRing` 是本课五类 ring 中三类的实现基础（submissions/completions/delta_*）。第 3 课讲了 Vyukov bounded MPMC ring 的序列号初始化与 host/device 共享内存模型；本课的 `DeviceRingView` 是同一套算法的 device-only 版本，`sequences` 同样初始化为 `[0..capacity-1]`，但 `enqueue_position` / `dequeue_position` 一端用 `cudaMalloc` 的 device 内存（`MappedRing` 在 `mapped_ring.hh:46`–`54` 根据 direction 决定哪端用 device_owned）。
- **第 10 课（delta/动态路由/预算）**：本课装配的 `d_delta_records` / `d_resident_pq_*` / `d_dynamic_route_slots` 等设备表，正是第 10 课设计的 delta 三层结构（mutable delta / resident-PQ / dynamic-route）的 device 物化。`delta_capacity`、`resident_pq_capacity`、`dynamic_route_capacity` 的容量推导在第 10 课与第 12 课讲过。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`Impl::~Impl` 的回滚顺序、`kernel_running` 标志、`shutdown` / `maintenance_shutdown` 原子的语义在第 11 课讲过；本课的失败路径与之一致。
- **第 12 课（construction 上）**：本课直接承接，第 12 课的 `memory_budget::estimate`、`d_remote_buffer` 布局、`stream_codes_to_gpu` / `stream_anchor_graph_to_gpu` 是本课所有 `d_*` 指针能正确指向的前提。分界在 `construction.cc:416`：416 行之前是"预算与远端数据搬运"，417 行起是"device 元数据 H2D 与 ring/params 装配"。
- **第 14 课（查询执行/路由/完成）**：`admission_loop` 与 `completion_loop` 是本课最后启动的两条线程的入口；它们如何使用 `submissions` / `completions` ring、如何处理 `direct_disabled` 回退，在第 14 课详讲。
- **第 15 课（增量发布）**：`submit_delta_publication` / `upload_mutations` / `maintenance_loop` 的 delta durable 推进，消费本课装配的 `delta_staging_*`、`delta_supersede_updates` 等 mapped 缓冲与 `d_delta_*` 设备表；control-delta CTA 的发布主循环在 `runtime.cuh:171`–`675`。
- **第 16 课（存储回收 RCU）**：`d_graph_cache_readers` / `d_anchor_graph_readers` 这两个读者计数数组、`refresh_anchor_graph_records` 用 `anchor_graph_readers_host` 判断回收时机，都在第 16 课讲。
- **第 17 课（kernel 启动器/上下文/device ring）**：`DeviceRingView` 的 `ld.acquire.sys` / `st.release.sys` 指令、`device_ring_try_pop` / `device_ring_push` 的实现、`__threadfence_system` 的语义，第 17 课会从 kernel 视角深入讲。
- **第 18 课（候选评分）**：`d_navigation_candidate_handles` / `d_navigation_candidate_distances` / `d_query_luts` 的使用方在评分 kernel 里。
- **第 19 课（RDMA cache）**：`d_graph_cache_*` / `d_exact_cache_*` 的状态机（empty/filling/ready/stale/fill-invalidated）与 admission filter 的运行期行为在第 19 课讲。
- **第 20 课（查询遍历主循环）**：`process_query` 的完整流程，本课只提了它被 query CTA 调用。
- **第 21 课（kernel 运行时/角色调度）**：`persistent_search_kernel` 主循环的退避策略、`enable_dispatcher` / `enable_delta` 的协调、stop 信号的处理，第 21 课会更系统地讲。
- **第 22 课（GPUNetIO 传输/probe）**：`GpuNetioPersistentTransport` 如何建 QP、export `qp_array` / `local_mkey` / `local_iova_base`、`direct_read_owner_loop` 里 WQE 准备与 CQ poll 的细节，全在第 22 课。本课只讲"装配点"：`construction.cc:376`–`381` 构造 transport、`376` 拿 `direct_view`、`913`–`914` 把 `qp_array` / `remote_regions` 塞进 `PersistentKernelParams`。

---

## 小结

本课（第 13 课）承接第 12 课，覆盖 `construction.cc:417`–`1021` 共约 600 行构造函数尾部，主要内容是：

1. **shard / anchor 元数据 H2D**（417–503）：把 `index.shards`、PQ 模型、entry points、anchor 表、anchor 路由图键/状态/读者计数从 host 推到 device；anchor PQ 码用 `launch_gather_anchor_codes` 在 GPU 上现采而非 host 拷贝；anchor 向量做转置以匹配 kernel 的矩阵乘法布局。
2. **查询工作区与 query dispatch ring**（505–553）：每个 query slot 一组解码/变换/LUT/候选/visited/dynamic-request scratch；建立**第一个纯 device ring** `device_submissions`，把 admission 与 query CTA 解耦。
3. **GPUNetIO direct batch queue 装配**（555–611）：`direct_batch_queue_count = qps_per_node * remote_region_count`，强校验与预算一致；每条 QP 一个 `DeviceRingView<DirectBatchDescriptor>`，容量 64；mapped pinned 的 `direct_owner_phases_host` 作为 CPU↔kernel 调试/就绪通道。
4. **graph cache / exact cache**（613–707）：组相联 cache 的 keys/generations/timestamps/states/readers/victims 六个数组 + admission filter；mapped pinned 的 delta 命令 staging 缓冲（supersede/override/durable/resident-pq-erase/dynamic-route 五类）。
5. **结果缓冲 + delta/resident-PQ/dynamic-route 设备表**（709–798）：mapped pinned 结果数组；delta 三件套（records/vectors/pq-codes）+ 链表 + 两张哈希表（base override / delta remote）+ resident-PQ 缓存 + 动态路由表；`clear_delta_device_state` 复位。
6. **stop/ready 标志 + 4 条 CUDA stream**（800–838）：stop/disabled/error 三个 portable pinned 标志 + 三个 mapped pinned ready 计数器；NonBlocking 流避免与默认流同步。
7. **block 角色计数与 SM 容量校验**（839–862）：`owner_kernel_blocks = ceil(qp_count / 8)`，`kernel_blocks = min(query_slots, SM*blocks_per_sm, num_threads, SM - owner - 2)`，保证 grid 不过载。
8. **`PersistentKernelParams` 装配**（864–1014）：约 150 个字段逐个填入，分 7 大组（ring/索引元数据/搜索参数/GPUNetIO/delta-resident-override/anchor-路由/cache-结果）；`direct_owner_block_count` 等 5 个字段留给 launch 时补。
9. **`start_persistent_kernel`**（lifecycle.cc:219–302）：unified grid launch（`total_blocks = owner + query + 2`），3 秒 ready barrier 等待 4 类就绪信号；失败时 stop + 同步 + 抛异常。
10. **后台线程启动**（1015–1018）：kernel 就绪后启动 admission / completion / maintenance 三条线程，构造函数返回，引擎进入运行态。

本课的核心张力在于"**静态装配**"与"**动态就绪**"的衔接：构造函数前半（第 12 课）算好预算、拉好远端数据，本课把预算物化成几百个 device 指针与容量、塞进一个 `PersistentKernelParams`，然后 launch 一个永不退出的 kernel，用 ready barrier 确认它真的起来了，最后才敢启动后台线程往 ring 里喂数据。任何一步失败都要能干净回滚——这是 PImpl 析构链（第 11 课）的责任。

下一课（第 14 课）将讲 `admission_loop` 与 `completion_loop` 如何使用本课建立的 ring 与工作区，以及直读路径失败时的回退策略。
