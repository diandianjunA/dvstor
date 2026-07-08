# 第 18 课：GPU 资源管理与异步完成

## 本课目标

本课从资源管理角度讲 GPU：每个 compute thread 和 coroutine 有哪些 CUDA 资源，如何 sizing，GPUDirect candidate buffer 如何注册，GPU event 如何和协程调度结合。

## 代码证据

必须阅读：

- `src/gpu/gpu_buffer_manager.hh`
- `src/gpu/gpu_buffer_manager.cu`
- `src/gpu/gpu_awaitable.hh`
- `src/gpu/gpu_awaitable.cc`
- `src/gpu/compute_thread_gpu.cc`
- `src/service/compute_service/lifecycle.ipp`

## 初始化位置

compute service 构造函数中：

1. `gpu::gpu_init(config_.gpu_device)`。
2. 构造 `WorkerPool` 和 `ComputeThread`。
3. 计算 `max_batch`。
4. 对每个 compute thread 调：

```cpp
thread->gpu_buffers.init(
  num_coroutines,
  dim,
  max_batch,
  R,
  query_buffer_bytes,
  candidate_buffer_bytes,
  protection_domain,
  gpudirect_rdma,
  enable_kernel_timing)
```

worker thread 内部也会调用 `gpu::gpu_init` 设置当前 CUDA device。

## max_batch 计算

`ComputeService` 中：

```text
query_frontier_batch = max(R, beam_width) * expansion_batch * query_batch_factor
construction_batch = beam_width_construction
overflow_prune_batch = R + 1
max_batch = max(query_frontier_batch, construction_batch, overflow_prune_batch)
```

这说明 GPU candidate buffer 必须同时覆盖：

- 查询候选。
- 插入 construction candidate。
- overflow prune candidate。

如果参数设置过大，GPU memory 也会线性增加。

## 每 coroutine 资源

`CoroutineGpuState` 每个 coroutine 一份：

stream/event：

- `stream`
- `event`
- optional `kernel_start_event`

host pinned：

- query
- candidate vectors
- candidate dists
- candidate order
- distances
- candidate pointers
- pruned indices
- pruned count

device：

- query
- candidate vectors A/B
- candidate dists
- candidate order
- distances
- candidate pointers
- pruned indices
- pruned count

这意味着 GPU buffer memory 随：

```text
num_threads * num_coroutines * max_batch
```

增长。

## query buffer bytes

`query_buffer_bytes` 取：

```text
max(dim * sizeof(float) * query_batch_factor,
    rabitq_code_bits * sizeof(float))
```

原因：

- 普通查询需要存 query batch。
- RaBitQ 需要存 rotated query，长度为 next power of two。

## candidate buffer bytes

`candidate_buffer_bytes` 取：

```text
max(VamanaNode::vector_bytes(), VamanaNode::rabitq_entry_size())
```

原因：

- exact distance 需要完整 vector。
- RaBitQ 或其他路径可能读取 entry。

## GPUDirect RDMA 注册

如果 `enable_gpudirect_rdma && rdma_pd != nullptr`：

对每个 coroutine：

```text
ibv_reg_mr(pd, d_candidate_vecs, candidate_bytes, IBV_ACCESS_LOCAL_WRITE)
ibv_reg_mr(pd, d_candidate_vecs_alt, candidate_bytes, IBV_ACCESS_LOCAL_WRITE)
```

全部成功时：

- `gpudirect_candidate_ready_ = true`
- `gpudirect_rdma_enabled_ = true`

任意失败则清理所有 MR，退回 host staging。

## GPU awaitable 完成模型

kernel launcher 负责记录 event：

```text
cudaEventRecord(event, stream)
```

`GpuAwaitable::await_suspend`：

```text
thread->track_gpu_post()
```

`ComputeThread::poll_gpu_events`：

```text
for each coro:
  if gpu_post_balances[coro] > 0:
    status = cudaEventQuery(event)
    if success:
      gpu_post_balances[coro] = 0
```

调度器只有在 RDMA 和 GPU balance 都为 0 时 resume coroutine。

## kernel timing

如果 `enable_breakdown && observe_device_utilization`：

- event 创建时保留 timing。
- 额外创建 `kernel_start_event`。
- `begin_query_gpu_kernel_timing` 在 kernel 前 record start。
- `finish_query_gpu_kernel_timing` 用 `cudaEventElapsedTime` 记录 kernel 时间。

注意这个 kernel time 排除了 CPU launch、排队和 D2H。

## destroy

`GpuBufferManager::destroy` 会：

- `cudaFree` device buffers。
- `ibv_dereg_mr` GPUDirect MRs。
- `cudaFreeHost` pinned buffers。
- `cudaEventDestroy`。
- `cudaStreamDestroy`。

析构函数如果仍 initialized，也会调用 `destroy`。

## 性能影响

- 每 coroutine 一 stream 支持并发，但过多 stream 也会增加调度开销。
- pinned host buffer 加快 H2D/D2H，但占用 pinned memory。
- GPUDirect 成功时减少 candidate H2D，失败时自动 fallback。
- 双 candidate device buffer 支持更好的查询流水线。
- kernel timing event 如果启用可能带来少量开销。

## 设计异味

1. GPU buffer sizing 逻辑在 `ComputeService` 中，不在 GPU manager 内部。
2. 所有 coroutine 预分配最大 buffer，内存占用可能很大。
3. GPUDirect 注册要么全部成功要么全部关闭，没有 per-coroutine 降级策略。
4. `GpuAwaitable` 不校验 event 是否真的记录，依赖调用方约定。
5. `poll_gpu_events` 对所有 coroutine 轮询，coroutine 多时开销增加。

## 可验证问题

- `max_batch` 为什么要考虑 `R + 1`？
- GPUDirect 注册失败后系统是否还能运行？
- `gpu_post_balances` 何时加一、何时清零？
- kernel timing 测到的是端到端 GPU 时间吗？
- 每个 compute thread 是否共享同一个 CUDA stream？

## 学习任务

1. 根据你的配置估算 GPU buffer 总内存。
2. 找出所有使用 `d_candidate_vecs_alt` 的位置。
3. 设计一个指标检查 GPUDirect 是否启用成功。
4. 跟踪 `GpuBufferManager::destroy` 的资源释放顺序。
5. 思考：如果要按需分配 GPU buffers，哪些路径需要改变？

