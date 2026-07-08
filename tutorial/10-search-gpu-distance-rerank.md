# 第 10 课：在线查询主路径之三：GPU 距离计算与 rerank

## 本课目标

本课聚焦查询中从候选向量 RDMA 到 GPU distance，再到 D2H 和 beam update 的路径。你需要掌握普通 host staging、GPUDirect candidate RDMA、indirect pointer path、RaBitQ exact rerank 在代码中的差异。

## 代码证据

必须阅读：

- `src/vamana/vamana_search.ipp`
- `src/gpu/gpu_buffer_manager.hh`
- `src/gpu/gpu_buffer_manager.cu`
- `src/gpu/gpu_awaitable.hh`
- `src/gpu/compute_thread_gpu.cc`
- `src/gpu/gpu_kernel_launcher.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/kernels/distance_kernels.cuh`

## 查询 GPU 路径总览

非 RaBitQ 单查询路径：

```text
query H2D
neighbor RDMA -> all_unvisited
batch_read_vectors(all_unvisited)
  direct to GPU 或 host buffers
如果 host buffers:
  memcpy host buffers -> pinned h_candidate_vecs
  H2D h_candidate_vecs -> d_candidate_vecs
GPU typed L2 distance
co_await GpuAwaitable
D2H distances
beam update
```

RaBitQ 路径：

```text
CPU rotated query + LUT
neighbor RDMA -> all_unvisited
RaBitQ estimate/gate
exact_ptrs = gate selected
如果第一次 exact:
  query H2D
batch_read_vectors(exact_ptrs)
GPU typed L2 exact rerank
D2H distances
beam update only selected candidates
```

## GpuBufferManager 资源

每个 `ComputeThread` 有一个 `GpuBufferManager`。每个 coroutine 有一套 `CoroutineGpuState`：

host pinned:

- `h_query`
- `h_candidate_vecs`
- `h_candidate_dists`
- `h_candidate_order`
- `h_distances`
- `h_candidate_ptrs`
- `h_pruned_indices`
- `h_pruned_count`

device:

- `d_query`
- `d_candidate_vecs`
- `d_candidate_vecs_alt`
- `d_candidate_dists`
- `d_candidate_order`
- `d_distances`
- `d_candidate_ptrs`
- `d_pruned_indices`
- `d_pruned_count`

stream/event:

- `stream`
- `event`
- optional `kernel_start_event`

GPUDirect:

- `d_candidate_vecs_mr`
- `d_candidate_vecs_alt_mr`
- `d_candidate_vecs_lkey`
- `d_candidate_vecs_alt_lkey`

## 双缓冲 candidate vectors

查询路径会调用：

```cpp
gs.flip_query_candidate_buffer();
uint8_t* staging = gs.current_query_candidate_vecs();
```

这在 `d_candidate_vecs` 和 `d_candidate_vecs_alt` 之间切换。目的：

- 避免下一轮 RDMA 写入覆盖当前 GPU 还可能读取的 candidate buffer。
- 支持 precommit neighbor read 和 GPU/D2H 之间更多重叠。

注意：双缓冲只针对 candidate vector device buffer，不是完整 pipeline 的所有 buffer 都双缓冲。

## host staging 路径

如果不能 direct to GPU：

```text
batch_read_vectors -> host_buffers
for each candidate:
  memcpy host_buffer -> gs.h_candidate_vecs
  free host_buffer
cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs, bytes, H2D)
```

统计：

- `query_host_staging_fallback_bytes`
- `transfer_candidate_h2d`
- `cpu_query_stage_candidates`

这个路径有两次拷贝：

1. RDMA 写入 compute hugepage host buffer。
2. CPU memcpy 到 pinned host buffer，再 H2D。

## GPUDirect candidate RDMA 路径

如果：

```cpp
gpu.gpudirect_candidate_ready() && gs.d_candidate_vecs_rdma_registered
```

则 `batch_read_vectors` 可以把 RDMA destination 设置为 GPU device buffer 和对应 lkey。

统计：

- `query_rdma_to_staging_bytes`

这个路径避免 host staging 和 H2D candidate copy，但要求：

- RDMA device 支持注册 CUDA device memory。
- `ibv_reg_mr(pd, d_candidate_vecs, candidate_bytes, IBV_ACCESS_LOCAL_WRITE)` 成功。

代码中如果任一 coroutine candidate buffer 注册失败，会清理所有注册并关闭 GPUDirect candidate ready。

## indirect candidate pointer path

查询中有：

```cpp
const bool use_indirect_candidate_path =
  use_gpudirect_candidate_rdma && thread->reserved_query_state[1] != nullptr;
```

如果启用，会把每个 candidate 的 device pointer 写到 `h_candidate_ptrs`，再 H2D 到 `d_candidate_ptrs`，调用：

```cpp
launch_batch_typed_query_l2_distances_indirect
```

这个路径适合候选不是连续布局时使用。当前代码中普通 GPUDirect read 会把候选按 batch 连续写入 staging，因此 indirect path 不是默认必要路径。

## kernel launcher

主要查询 launcher：

- `launch_batch_typed_query_l2_distances`
- `launch_batch_typed_multi_query_l2_distances`
- `launch_batch_typed_query_l2_distances_indirect`

它们根据 query dtype 和 candidate dtype 选择模板 kernel：

- float/float
- float/uint8
- float/int8
- uint8/uint8
- int8/int8
- uint8/int8 等

对 integral dtype，launcher 会判断 dim 下 `int32_t` accumulator 是否安全，否则使用 `int64_t` accumulator。

## kernel 粒度

`gpu_kernel_launcher.cu` 中：

```cpp
TILE_SIZE = 4
BLOCK_SIZE = 512
total_threads = n_candidates * TILE_SIZE
num_blocks = ceil(total_threads / BLOCK_SIZE)
```

也就是说每个 candidate distance 用一个 tile 的 4 个线程协作完成。kernel 内用 cooperative groups reduce。

这对短向量和小 batch 的 kernel launch overhead 很敏感。候选太少时，GPU 可能利用率很低。

## GpuAwaitable

launcher 在 kernel 后记录 event：

```cpp
cudaEventRecord(event, stream)
```

然后查询代码：

```cpp
co_await gpu::GpuAwaitable{thread.get()}
```

`GpuAwaitable` 设置 `gpu_post_balances[coro_id]++`。scheduler 轮询：

```cpp
cudaEventQuery(gpu_buffers.event(coro_id))
if success:
  gpu_post_balances[coro_id] = 0
```

这让同一 OS thread 可以在等待 GPU 时推进其他 coroutine。

## D2H 和 beam update

GPU 完成后：

```text
cudaMemcpyAsync(h_distances, d_distances, n_batch * sizeof(float), D2H)
cudaStreamSynchronize(stream)
for each distance:
  insert_into_beam(beam, ptr, distance, beam_width)
```

注意这里有 `cudaStreamSynchronize`，D2H 阶段是同步等待。代码会在 D2H 前 precommit 下一轮 neighbor reads，以重叠部分 RDMA latency。

## final result id read

beam 搜索结束后：

```text
sort beam
for beam entries until k:
  read_vamana_id(rptr)
  if header not deleted:
    results.push_back({id, distance})
```

`read_vamana_id` 是远程小读，读取 header 和 id。如果节点 deleted，返回 `max node_t` 并跳过。

这一步保证 upsert/delete 后不返回已删除节点，但也给查询尾部增加最多 `k` 次小 RDMA。

## 性能影响

主要可观测指标：

- `query_h2d_bytes`
- `query_d2h_bytes`
- `query_vector_rdma_reads_in_bytes`
- `query_rdma_to_staging_bytes`
- `query_host_staging_fallback_bytes`
- `query_distcomps`
- `query_exact_reranks`
- `gpu_query_distance_ns`
- `transfer_distance_d2h_ns`
- `vector_rdma_credit_wait_ns`

优化方向通常围绕：

- 增大 batch 提升 GPU 利用率。
- 减少 exact candidate 数。
- 启用 GPUDirect 减少 H2D。
- 减少 D2H 同步次数。
- 减少 final id 小读。

## 设计异味

1. GPU buffer sizing 在 `ComputeService` 构造函数中按多个参数推导，逻辑较隐蔽。
2. query path 中 CPU、RDMA、GPU、统计、RaBitQ gate 交织在一个函数里。
3. D2H 后使用 `cudaStreamSynchronize`，没有完全事件化。
4. final id read 是串行 `co_await`，对大 k 可能影响尾延迟。
5. kernel launcher 只暴露 C 风格函数，缺少对 dtype pair 的可测试策略层。

## 可验证问题

- GPUDirect candidate RDMA 成功后是否还需要 candidate H2D？
- `GpuAwaitable` 是否自己检查 CUDA event？
- 为什么 candidate buffer 有两个 device buffer？
- integral dtype 距离为什么要判断 accumulator 是否安全？
- 查询最终结果为什么还要读 id？

## 学习任务

1. 跟踪非 GPUDirect 查询路径中每一次内存拷贝。
2. 跟踪 GPUDirect 查询路径中 local lkey 的来源。
3. 在 `gpu_kernel_launcher.cu` 中列出所有 dtype pair 分支。
4. 用 breakdown 指标设计一个实验：验证 GPUDirect 是否真的减少 H2D bytes。
5. 思考：如果要把 D2H 和 beam update 也流水线化，需要增加哪些 buffer 和状态？

