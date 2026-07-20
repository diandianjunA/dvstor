# 第 19 课：RDMA cache 与请求合并

## 本课目标与涉及文件

在第 17 课里我们看到了 `direct_batch_queues` 这一组 per-QP 的 `DeviceRing`，以及 `PersistentKernelParams` 里那一长串以 `direct_*`、`graph_cache_*`、`exact_cache_*` 打头的字段——它们就是 GPU 端发起 RDMA read、缓存图页/精确记录、合并请求的全部“接线”。第 18 课我们讲了候选评分，落到了“某些 PQ code 必须从远端拉”这一步。本课要回答的核心问题是：

> 在一个 GPU 持久化 kernel 里，几百个并发查询 CTA 如何把“我要读远端这一页图/这一条向量”这个意图，安全、低延迟、且高扇入地变成一次 GPUNetIO RDMA read，再把结果回填给正确的查询？

`rdma_cache.cuh` 就是这个问题的答案。它不是一个传统意义上的“缓存数据结构”，而是一整套：**请求合并（merge）→ per-QP dispatch → CQ 轮询 → 结果落点（scratch / adjacency cache / exact cache）→ 失败重试/失败即停（fail-stop）**的 GPU 端协作流程。本课涉及的文件：

- `src/gpu_search/persistent_kernel/rdma_cache.cuh`（约 1038 行，本课主体）
- `src/gpu_search/persistent_kernel/runtime.cuh`（`direct_read_owner_loop`、`direct_read_owner_kernel`，owner warp 侧的消费者）
- `src/gpu/gpunetio_transport.cc`（CPU 端装配 send_cq/recv_cq/qp_wq/qp_dbr umem、跑 startup probe，对应第 22 课）
- `src/gpu_search/persistent_kernel.hh`（`DirectRemoteRegion`、`DirectBatchDescriptor`、`PersistentKernelParams` 的 `direct_*` / `graph_cache_*` / `exact_cache_*` 字段）
- `src/gpu_search/persistent_kernel/candidate_scoring.cuh`（`kGraphCache*` 状态机常量、`poll_direct_cq`、`lock_direct_qp`、`anchor_graph_slot`、`release_graph_record`）
- `src/gpu_search/types.hh`（`CompletionDescriptor` 的 `remote_batches` / `graph_read_retries` / `exact_cache_hits` 等遥测；聚合统计里的 `rdma_merged_requests` / `graph_page_cache_hits` / `graph_dependency_rounds`）
- `src/common/configuration.hh`（`gpu_graph_prefetch_depth=32`、`gpu_adjacency_cache_ways=4` 默认值）
- `src/gpu_search/persistent_kernel/query_traversal.cuh`（调用点：`fetch_graph_records_batch`、`approximate_handles_batch`、`exactify_into_beam`）

读完本课你应该能画出：查询 CTA → per-shard owner warp → GPUNetIO WQE → CQ → scratch/cache 回填 的完整状态机，并理解“为什么默认 `gpu_adjacency_cache_ways=4` 但 cache 容量却可能是 0”、以及 `graph_read_retries` 和 fail-stop 的边界。

---

## 逐文件逐函数讲解

### 一、距离计算的基础工具（rdma_cache.cuh:1–35）

文件开头是两个不直接涉及 RDMA、但被 cache 路径反复使用的小工具：

```cpp
__device__ __forceinline__ f32 storage_component(
    const PersistentKernelParams& params, const u8* vector, u32 dimension) {
  if (params.vector_dtype == 0) return reinterpret_cast<const f32*>(vector)[dimension];
  if (params.vector_dtype == 1) return static_cast<f32>(vector[dimension]);
  return static_cast<f32>(reinterpret_cast<const int8_t*>(vector)[dimension]);
}
```

`vector_dtype` 取 0/1/2 分别对应 f32 / u8 / int8 三种存储格式（与第 9 课讲的 PQ 模型一致）。`exact_storage_distance` 与 `exact_anchor_distance` 是两个朴素的 L2 距离实现——前者在远端 record 拉回后由 `exactify_into_beam` 调用，后者在 anchor 距离比较时使用。注意 anchor 向量布局是 `(dim, anchor_count)` 的 SoFA（column-major）：

```cpp
const f32 component = params.anchor_vectors[
  static_cast<size_t>(dimension) * params.anchor_count + anchor];
```

这是因为同一查询要顺序扫所有 anchor，column-major 让连续 thread 读连续地址，匹配 GPU 合并访问。这两段不是本课重点，但会出现在 cache 命中后的“就地算距离”分支里。

### 二、direct_fetch：单条 RDMA read 的原语（rdma_cache.cuh:37–113）

这是整条链路最底层、最贴近硬件的函数。它把“对某个 `memory_node` 读 `bytes` 字节、落到 `destination`”这一件事直接映射成一条 GPUNetIO RDMA READ WQE。它**不**走 owner warp/合并路径，而是当前线程亲自锁 QP、亲自 doorbell、亲自 poll CQ。它主要被 startup probe 与 `gpunetio_locked_read_probe_kernel` 用，运行时查询路径基本走下面的 `direct_fetch_batch`。但它的代码揭示了所有关键约定，所以逐段讲：

```cpp
__device__ i32 direct_fetch(const PersistentKernelParams& params,
                            u32 memory_node, u64 remote_offset,
                            u8* destination, u32 bytes, u32 lane) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qp_locks == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return -EHOSTDOWN;
```

第一段是 6 项前置检查。`direct_disabled` 是一个全局的“fail-stop 闸门”——任何一个 QP 上发生不可恢复错误，就把这个 u32 置 1（见后面的 `atomicExch(params.direct_disabled, 1u)`），此后所有路径立刻短路返回 `-EHOSTDOWN`。`volatile` + `reinterpret_cast` 是为了跨 SM/CTA 可见（GPU 上 `atomicExch` 自带 fence，但这里读侧用 volatile 也保险）。

```cpp
  const u32 qp_index = (lane % params.direct_qps_per_node) *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
```

QP 的索引布局是 `[lane][memory_node]`——每个 lane（即 `direct_qps_per_node` 中的一个“流”）对每个远端节点都有独立 QP。第 22 课 transport 装配时就是按 `for lane: for server:` 双层循环创建的（`gpunetio_transport.cc:245–411`），并填到 `d_qp_array` 里。`lane` 参数是调用方传的“我属于哪条流”，使得多查询间能在不同 QP 上并发，避免一个 QP 成为瓶颈。

```cpp
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  doca_gpu_dev_verbs_addr remote{.addr = region.address + remote_offset, .key = region.rkey};
  doca_gpu_dev_verbs_addr local{
    .addr = reinterpret_cast<u64>(destination) - params.direct_local_iova_base,
    .key = params.direct_local_mkey,
  };
```

`DirectRemoteRegion` 在 `persistent_kernel.hh:52` 定义，只有 `address`/`rkey` 两个字段——对应远端 MR 的基址和远程 key。`remote.addr = region.address + remote_offset` 说明 `remote_offset` 是相对该 region 基址的字节偏移，调用方已经知道目标页/目标记录在自己 shard 的远端 MR 里的位置（见 `prepare_graph_record` 里 `request_offset = graph_offset`、`approximate_handles_batch` 里 `request_offsets[index] = node_offset + params.shards[shard].dynamic_code_offset`）。

`local.addr` 的写法值得注意：`reinterpret_cast<u64>(destination) - params.direct_local_iova_base`。`destination` 是 GPU 显存指针（虚拟地址），减去 `direct_local_iova_base` 得到的是“在已注册 MR 里的偏移”——这正是 GPUNetIO WQE 期望的 IOVA。`direct_local_iova_base` 在 `gpunetio_transport.cc:432/441` 设定：如果 `ibv_reg_mr` peer_memory 路径成功则为 0（此时 IOVA == daddr），否则走 dmabuf 路径、`local_iova_base = registered_base`（IOVA 是相对基址的偏移）。这两条路径对 kernel 是透明的——kernel 只管做减法。

```cpp
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    if (status != -ECANCELED) atomicExch(params.direct_disabled, 1u);
    return status;
  }
```

`lock_direct_qp`（`candidate_scoring.cuh:181`）是一个 spin-CAS 锁。GPUNetIO 的 QP WQ 不是多 producer 安全的——两个 warp 同时往同一 SQ 塞 WQE 会撕裂 ticket 计算。所以每个 QP 配一个 `i32` lock（`d_qp_locks`，`gpunetio_transport.cc:501` 分配），任何想直接门铃的路径都得先拿锁。`-ECANCELED` 表示是 stop/disabled 主动取消，不算错误；其它非零 status 都把 `direct_disabled` 置 1，进入 fail-stop。

```cpp
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, -E2BIG);
    atomicExch(params.direct_disabled, 1u);
    unlock_direct_qp(params.direct_qp_locks + qp_index);
    return -E2BIG;
  }
```

单条 WQE 有最大传输长度（由 ConnectX 硬件限制，通常 2GB 但实际配置更小）。超大的请求直接 fail-stop——因为继续往下走会生成一条硬件无法执行的 WQE，CQ 会报错，但错误恢复更麻烦，不如在源头拒绝。

```cpp
  const doca_gpu_dev_verbs_ticket_t read_ticket = qp->sq_wqe_pi;
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const doca_gpu_dev_verbs_ticket_t completion_ticket =
    doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &completion_queue->cqe_ci);
```

`sq_wqe_pi` 是 SQ 的 producer index（下一个要写 WQE 的位置），`cqe_ci` 是 CQ 的 consumer index。`ticket` 的语义：我要 poll 的是“在我提交这一条 WQE 之前 CQ 已经消费到的位置之后的那一个 CQE”。EXCLUSIVE 模式表示这个 QP 不会被 CPU 共享，GPU 可以放心地用 relaxed load 读取元数据。

```cpp
  auto* read_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, read_ticket);
  const bool need_dump = qp->need_dump;
  doca_gpu_dev_verbs_wqe_prepare_read(
    qp, read_wqe, read_ticket,
    need_dump ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE
              : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    remote.addr, remote.key, local.addr, local.key, bytes);
```

`wqe_prepare_read` 把 RDMA READ 的所有参数（远端 addr/rkey、本地 IOVA/mkey、字节数、CQ 事件策略）填进 WQE。`need_dump` 是一个调试开关——为真时，每次 RDMA read 后追加一条 “dump” WQE（用 `MLX5_WQE_CTRL_CQ_ERROR_UPDATE` 强制生成一个 CQE，便于确认硬件确实把前面的 READ 跑完了）。正常生产路径 `need_dump=false`，只用 `CQ_UPDATE`——只有这一条 READ 完成时才产 CQE。

```cpp
  doca_gpu_dev_verbs_ticket_t final_ticket = read_ticket;
  if (need_dump) {
    final_ticket = read_ticket + 1;
    auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_ticket);
    doca_gpu_dev_verbs_wqe_prepare_dump(
      qp, dump_wqe, final_ticket, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
      dump.addr, dump.key, 1);
  }
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      qp, final_ticket + 1);
```

`submit` 才是真正“敲门铃”——它把 `qp->sq_wqe_pi` 推进到 `final_ticket + 1`，并通过 UAR 写 doorbell 通知 NIC 有新 WQE。**注意 prepare 只是写 WQE 内存，submit 才让硬件看见。** 这正是为什么后面 owner warp 路径能把多条 WQE 一次性 submit——它批量 prepare，最后一次 submit。

```cpp
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
  unlock_direct_qp(params.direct_qp_locks + qp_index);
  return status;
```

`poll_direct_cq`（`candidate_scoring.cuh:144`）是个自旋循环：它读 `completion->op_own` 的 owner bit，对比 `ticket` 推算出的期望 owner，判断 CQE 是否就绪；就绪后检查 opcode 是不是 `MLX5_CQE_REQ_ERR`，并用 `atomic_max` 推进 `cqe_ci`。它带 `direct_timeout_ns` 超时与 stop/disabled 主动取消。**任何非零返回都直接 fail-stop**——单条直发路径不容忍硬件错误，因为继续重试同一个 QP 很可能再次失败，让全局 disabled 反而能让上层一致地切到 fallback。

`#else` 分支（行 104–112）说明：没有 GPUNetIO 编译选项时这个函数返回 `-ENOTSUP`，让上层一致走 fallback（CPU 代理 RPC）。这就是为什么整个 `rdma_cache.cuh` 大量 `#ifdef DVSTOR_HAVE_GPUNETIO`——它是一个可选加速路径。

### 三、direct_fetch_batch：批量合并的入口（rdma_cache.cuh:115–262）

这是查询路径真正调用的函数。它的契约是：**给定一组 `(request_shards[], remote_offsets[], local_iova_offsets[])` 共 `request_count` 条请求，把其中 `request_shards[i] == memory_node` 的那些条目合并成一次 GPUNetIO 提交。**

```cpp
__device__ i32 direct_fetch_batch(const PersistentKernelParams& params,
                                  u32 memory_node, const u32* request_shards,
                                  const u64* remote_offsets, u32 request_count,
                                  u8* destination_base, u32 destination_stride,
                                  u32 bytes, u32 lane,
                                  const u64* local_iova_offsets = nullptr,
                                  i32* owner_completion = nullptr,
                                  bool defer_owner_wait = false,
                                  u32* owner_progress = nullptr) {
```

参数说明：
- `memory_node`：本次调用只处理目标节点是它的那些请求（合并的“分桶”维度）。
- `request_shards[]`：每条请求的目标节点。`UINT32_MAX` 表示“这条已经命中 cache 不用读”。
- `remote_offsets[]`：每条请求在远端 MR 里的偏移。
- `local_iova_offsets[]`：每条请求的本地落点 IOVA 偏移。**这是合并的关键**——不同条目可以落到不同本地地址（scratch 不同槽、cache 不同 slot），一次 RDMA 批量读就把多查询的请求并行下发。
- `bytes`：所有条目读相同的字节数（图页固定 512B、PQ code 固定 `pq_code_bytes`、精确记录固定 `node_record_bytes`）。
- `owner_completion`：如果非空，走 owner warp 异步路径。
- `defer_owner_wait`：true 表示“我提交完就返回 `-EINPROGRESS`，结果由我自己稍后 `wait_direct_batch` 取”。

```cpp
  u32 matching = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] == memory_node) ++matching;
  }
  if (matching == 0) return 0;
```

先数本次要发几条 WQE。0 条就直接返回 0（成功，但无事可做）。

```cpp
  if (params.direct_batch_queues != nullptr && owner_completion != nullptr) {
    if (qp_index >= params.direct_batch_queue_count) return -EINVAL;
    const u64 started = global_time_ns();
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 2;
      __threadfence_system();
    }
    atomicExch(owner_completion, -EINPROGRESS);
    __threadfence();
```

如果 `direct_batch_queues` 存在且调用方提供了 `owner_completion`，就**不亲自发 RDMA**，而是把请求塞进 per-QP ring，由专门的 owner warp 异步消费。`owner_progress` 是给 startup probe 用的调试字段（phase=2 表示“已入队”）。`atomicExch(owner_completion, -EINPROGRESS)` 在 fence 前后写入，让 owner warp 完成后能通过同一个原子变量通知本 CTA。

```cpp
    const DirectBatchDescriptor descriptor{
      .request_shards = request_shards,
      .remote_offsets = remote_offsets,
      .local_iova_offsets = local_iova_offsets,
      .completion_status = owner_completion,
      .request_count = request_count,
      .memory_node = memory_node,
      .bytes = bytes,
    };
    while (!device_ring_try_push(params.direct_batch_queues[qp_index], descriptor)) {
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(owner_completion, -ECANCELED);
        return -ECANCELED;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        atomicExch(owner_completion, -EHOSTDOWN);
        return -EHOSTDOWN;
      }
      if (global_time_ns() - started >= params.direct_timeout_ns) {
        atomicExch(owner_completion, -ETIMEDOUT);
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -ETIMEDOUT);
        }
        atomicExch(params.direct_disabled, 1u);
        return -ETIMEDOUT;
      }
      device_ring_relax(128);
    }
```

`DirectBatchDescriptor`（`persistent_kernel.hh:70`）就是把入参打包——注意它存的是指针，调用方必须保证这些数组在 owner warp 消费前不被释放/覆盖（实际上它们是 `params.dynamic_code_request_*` 或 query slot 本地的 shared 数组，生命周期覆盖整个 fetch 阶段）。`device_ring_try_push` 失败时自旋等待，但带 stop/disabled/timeout 三种逃逸条件。**入队超时也算 fail-stop**——说明 owner warp 严重积压，继续等下去会让整个查询 CTA 卡死。

```cpp
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 3;
      __threadfence_system();
    }
    if (defer_owner_wait) return -EINPROGRESS;
    for (;;) {
      const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
      if (status != -EINPROGRESS) return status;
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) return -ECANCELED;
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        return -EHOSTDOWN;
      }
      if (global_time_ns() - started >= params.direct_timeout_ns) {
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -ETIMEDOUT);
        }
        atomicExch(params.direct_disabled, 1u);
        return -ETIMEDOUT;
      }
      device_ring_relax(128);
    }
  }
```

phase=3 表示“已入队，开始等结果”。`defer_owner_wait=true` 时立刻返回 `-EINPROGRESS`，让调用方继续干别的事（比如给别的 shard 入队），稍后统一 `wait_direct_batch`；`defer_owner_wait=false` 时就地自旋等 owner 写入最终 status。

**接下来的 else 分支（行 196–246）是“没有 owner warp 时的直接批量路径”**——逻辑和 `direct_fetch` 几乎一致，但循环 prepare 多条 WQE：

```cpp
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      matching + (qp->need_dump ? 1u : 0u) > qp->sq_wqe_num) return -E2BIG;
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) return status;
  ...
  const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
  u32 posted = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    const doca_gpu_dev_verbs_ticket_t ticket = first_wqe + posted;
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
    const bool last_read = posted + 1 == matching;
    const auto flags = !qp->need_dump && last_read
      ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
      : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    const u64 local_iova = local_iova_offsets != nullptr
      ? local_iova_offsets[index]
      : reinterpret_cast<u64>(destination_base) +
          static_cast<u64>(index) * destination_stride - params.direct_local_iova_base;
    doca_gpu_dev_verbs_wqe_prepare_read(
      qp, wqe, ticket, flags, region.address + remote_offsets[index], region.rkey,
      local_iova, params.direct_local_mkey, bytes);
    ++posted;
  }
```

关键优化：**只有最后一条 READ 用 `CQ_UPDATE`，前面的都用 `CQ_ERROR_UPDATE`**（不在正常完成时产 CQE，只在出错时产 error CQE）。这样 N 条合并请求只 poll 1 个 CQE，把 CQ 压力降下来。这是 `rdma_merged_requests` 遥测对应的真实合并行为——多个查询/多个候选对同一远端节点的读被合并成一次 submit、一次 CQE。

```cpp
  doca_gpu_dev_verbs_ticket_t final_wqe = first_wqe + posted - 1;
  if (qp->need_dump) { ... }
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      qp, final_wqe + 1);
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
```

一次 submit 推进整批 WQE。`poll_direct_cq` 一次就拿到整批的完成状态。如果批里任何一条出错，CQE 会是 error opcode，`poll_direct_cq` 返回 `-EIO`，然后整批 fail-stop。

### 四、wait_direct_batch：异步路径的取结果（rdma_cache.cuh:264–292）

```cpp
__device__ i32 wait_direct_batch(const PersistentKernelParams& params,
                                 i32* owner_completion) {
  if (owner_completion == nullptr) return -EINVAL;
  const u64 started = global_time_ns();
  for (;;) {
    const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
    if (status != -EINPROGRESS) return status;
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      return -EHOSTDOWN;
    }
    if (global_time_ns() - started >= params.direct_timeout_ns) {
      if (params.direct_error != nullptr) {
        atomicCAS(params.direct_error, 0, -ETIMEDOUT);
      }
      atomicExch(params.direct_disabled, 1u);
      return -ETIMEDOUT;
    }
    device_ring_relax(128);
  }
}
```

简单自旋等 `owner_completion` 变成最终 status。和 `direct_fetch_batch` 的内联等待逻辑一样，单独提出来是因为查询路径用 `defer_owner_wait=true` 入队多个 shard 后，需要统一在外面等所有 shard 完成。`device_ring_relax(128)` 是一个 `__nanosleep` 类的指令（`device_ring.cuh:19`），让 spin 不烧满 SM 占用调度槽。

### 五、admission cache：四路组相联的“该不该缓存”过滤器（rdma_cache.cuh:294–326）

在讲 graph cache 主流程前，先看一个看起来不起眼但很重要的辅助结构。`exact_record_visible` 用 record 头部的 `kNodeLockMask | kNodeDeletedMask` 判断记录是否可见（被锁/被删的记录要跳过）。然后是两个 admission 函数：

```cpp
__device__ bool admit_graph_cache(const PersistentKernelParams& params, u64 key) {
  if (params.graph_admission_sets == 0 || params.graph_admission_keys == nullptr ||
      params.graph_admission_victims == nullptr) return true;
  const u32 set = hash64(key) % params.graph_admission_sets;
  const u32 base = set * kCacheAdmissionWays;
  for (u32 way = 0; way < kCacheAdmissionWays; ++way) {
    if (load_cg(params.graph_admission_keys + base + way) == key) return true;
  }
  const u32 way = atomicAdd(params.graph_admission_victims + set, 1u) %
    kCacheAdmissionWays;
  atomicExch(reinterpret_cast<unsigned long long*>(
               params.graph_admission_keys + base + way), key);
  return false;
}
```

`admit_graph_cache` 回答的问题是：“这个 key 第一次见到吗？”——只在“第一次见到”时返回 false（让调用方把记录装进真正的 cache），后续命中返回 true（调用方就知道这页已经在缓存里或正在被缓存，自己直接走 scratch 落点）。

它的实现是一个四路组相联的小指纹表（`kCacheAdmissionWays = 4`，`cuda_helpers.hh:15`），每个 set 4 个 slot。查重用 `load_cg`（`ld.global.cg`，cache-global 一致性——能看见其他 SM 的写入但可能不持久）；victim 用 `atomicAdd` 轮转。`admit_exact_cache` 结构完全一样，只是 key 是 u32（handle），hash 用 `hash32`。

**为什么需要 admission cache？** 因为 GPU 的 graph cache 容量有限（`graph_cache_sets * graph_cache_ways` 个 512B 槽），如果允许所有并发查询 CTA 都往里塞，会产生大量无谓的 filling→eviction 抖动。admission filter 让“第一次见”的 key 才有资格去抢 filling 槽，其他查询要么命中 ready 槽、要么等 filling 完成、要么走 scratch。这是多查询并发下的反抖动设计。

注意 `graph_admission_sets` 默认可以是 0——`construction.cc:232` 写了 `graph_admission_sets = std::min(graph_cache_sets, kMaxCacheAdmissionSets)`，所以**当 `graph_cache_sets=0`（即 graph cache 容量为 0）时，admission 也禁用，`admit_graph_cache` 永远返回 true**——意思是“没有 cache 时所有读都走 scratch”。这就是本课目标里“可选 adjacency cache（默认容量 0）”的真实含义。

### 六、approximate_handles_batch：PQ code 批量拉取与评分（rdma_cache.cuh:328–436）

这是查询遍历主循环里被调用的第一个批量函数：给定一组 candidate handle，把它们对应的 PQ code 拉回来（如果不在 delta/resident/dynamic route 里），算近似距离。它展示了“落点选择”的完整决策树。

```cpp
__device__ bool approximate_handles_batch(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          const f32* query_lut,
                                          u32* handles,
                                          u32 count,
                                          f32* distances) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
```

每个 query slot 在 `dynamic_code_request_*` 里有自己专属的 `kPersistentMaxMergeCandidates=2048` 条请求槽位（`construction.cc:241-246` 分配）。这是 per-query scratch 落点的“请求元数据”部分。

```cpp
  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 handle = handles[index];
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iova_offsets[index] = 0;
    distances[index] = FLT_MAX;
    if (handle == UINT32_MAX) continue;
    if ((handle & kDeltaHandleBit) == 0) {
      distances[index] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      continue;
    }
```

第一阶段：每个 thread 处理一个 handle。先默认 `request_shards = UINT32_MAX`（意为“不读远端”）。如果 handle 不是 delta handle（`kDeltaHandleBit = 0x80000000`），就直接用本地 `pq_codes` 算近似距离（`approximate_handle`，见第 18 课）；delta handle 才需要进一步判断。

```cpp
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    const u32 delta_slot = delta_slot_from_raw(params, raw);
    const u32 delta_count = min(load_cg(params.delta_count), params.delta_capacity);
    if (delta_slot < delta_count &&
        params.delta_records[delta_slot].remote_node == raw) {
      const DeviceDeltaRecord& record = params.delta_records[delta_slot];
      if (delta_code_visible(record, descriptor.snapshot_epoch)) {
        distances[index] = approximate_entry(
          params, query_lut,
          params.delta_pq_codes +
            static_cast<size_t>(delta_slot) * params.pq_code_bytes);
        continue;
      }
      const u64 superseded = load_cg(&record.superseded_epoch);
      if (record.epoch <= descriptor.snapshot_epoch &&
          ((record.flags & kDeltaDeleted) != 0 ||
           (superseded != 0 && superseded <= descriptor.snapshot_epoch))) {
        continue;
      }
    }
```

delta 路径：如果该 raw node 在 delta 表里且可见，直接用 `delta_pq_codes`；如果被删除/被取代（在快照 epoch 之前），跳过（distance 留 FLT_MAX）。这是第 10 课“delta/动态路由”在查询侧的体现。

```cpp
    const u32 resident_slot = resident_pq_slot_from_raw(params, raw);
    if (resident_slot != UINT32_MAX && params.resident_pq_codes != nullptr) {
      distances[index] = approximate_entry(
        params, query_lut,
        params.resident_pq_codes +
          static_cast<size_t>(resident_slot) * params.pq_code_bytes);
      continue;
    }
```

resident PQ 路径：某些热点节点的 PQ code 常驻 GPU 显存（`resident_pq_codes`），命中就用它。

```cpp
    if (params.dynamic_code_records == nullptr || shard >= params.num_shards) continue;
    const u64 node_offset = (raw << 16) >> 16;
    request_shards[index] = shard;
    request_offsets[index] = node_offset + params.shards[shard].dynamic_code_offset;
    u8* destination = params.dynamic_code_records +
      (request_base + index) * params.pq_code_bytes;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
  }
```

**真正的远端读路径**：落点是 `dynamic_code_records` 里 `(query_slot, index)` 对应的 per-query 槽。`request_local_iova_offsets[index]` 存的是这个槽的 IOVA 偏移。`node_offset` 是 raw node 的低 48 位（高 16 位是 shard），加上 shard 的 `dynamic_code_offset` 得到该 PQ code 在远端 MR 里的偏移。

```cpp
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
      params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, count,
        params.dynamic_code_records + request_base * params.pq_code_bytes,
        params.pq_code_bytes, params.pq_code_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iova_offsets, owner_completion, true);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion = params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();
```

这是合并的核心循环：对每个 shard 调用一次 `direct_fetch_batch`，让它把 `request_shards[i] == shard` 的所有请求合并成一次提交。`direct_batch_statuses` 是 per-query per-shard 的状态字（`descriptor.query_slot * num_shards + shard`），让 owner warp 完成后能写回正确位置。lane 选 `(query_slot + shard) % direct_qps_per_node`——这样不同 query 对同一 shard 会分散到不同 QP，避免单 QP 拥塞，同时同一 query 的不同 shard 也分散。

第二轮循环把所有 `-EINPROGRESS` 的 shard 收尾——这就是 `defer_owner_wait=true` 的意义：先全部入队、再统一等结果，让 owner warp 有机会并行处理多个 shard 的请求。

```cpp
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard == UINT32_MAX) continue;
    if (shard_status[shard] != 0) {
      atomicExch(&failed, 1u);
      continue;
    }
    const u8* code = params.dynamic_code_records +
      (request_base + index) * params.pq_code_bytes;
    distances[index] = approximate_entry(params, query_lut, code);
  }
  __syncthreads();
  return failed == 0;
}
```

结果回填：远端拉回的 PQ code 现在在 `dynamic_code_records[request_base + index]`，调用 `approximate_entry`（查 LUT 算近似距离）填回 `distances[index]`。任何 shard 失败都让整个 batch 失败（`failed=1`），返回 false 让上层把查询标为失败。

### 七、exactify_into_beam：精确记录的缓存与回填（rdma_cache.cuh:438–656）

这是最复杂的函数——它做精确重排（rerank）：把候选 handle 对应的完整向量记录拉回来、算精确 L2 距离、合并进 beam。和 `approximate_handles_batch` 的关键区别是它**会使用 exact_cache**——一个真正的、有 eviction 的、reader pin 的缓存。

```cpp
__shared__ u32 request_cache_slots[kPersistentMaxExact];
__shared__ u32 request_delta_slots[kPersistentMaxExact];
__shared__ u8 request_cache_owned[kPersistentMaxExact];
__shared__ i32 shard_status[kPersistentMaxShards];
```

per-CTA shared 数组：`request_cache_slots[index]` 记录每个候选对应的 cache slot（`UINT32_MAX` 表示不进 cache，直接落 scratch）；`request_cache_owned[index]=1` 表示这个 slot 是本 CTA 抢到的 filling slot，成功后要推进到 ready，失败要释放。

```cpp
for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
  ...
  request_offsets[index] = ((raw << 16) >> 16) + params.node_meta_offset;
  u32 cache_slot = UINT32_MAX;
  if (!dynamic && params.exact_cache_sets != 0 && params.exact_cache_ways != 0) {
    const u32 set = hash32(handle) % params.exact_cache_sets;
    bool cache_hit = false;
    for (u32 way = 0; way < params.exact_cache_ways; ++way) {
      const u32 slot = set * params.exact_cache_ways + way;
      const u32 state = *reinterpret_cast<volatile u32*>(
        params.exact_cache_states + slot);
      if (state == 2 && load_cg(params.exact_cache_keys + slot) == handle) {
        atomicAdd(params.exact_cache_readers + slot, 1u);
        __threadfence();
        if (*reinterpret_cast<volatile u32*>(params.exact_cache_states + slot) == 2 &&
            load_cg(params.exact_cache_keys + slot) == handle) {
          const u8* record = params.exact_cache +
            static_cast<size_t>(slot) * params.exact_cache_stride;
          if (exact_record_visible(record)) {
            candidate_ids[index] =
              *reinterpret_cast<const u32*>(record + kNodeIdOffset);
            candidate_distances[index] = exact_storage_distance(
              params, query, record + kNodeVectorOffset);
            atomicAdd(exact_cache_hits, 1u);
            cache_hit = true;
          }
        }
        atomicSub(params.exact_cache_readers + slot, 1u);
        if (cache_hit) break;
      }
    }
    if (cache_hit) continue;
```

这一段是 reader pin 模式：先检查 slot 状态是不是 `2`（ready），如果是就 `atomicAdd readers +1`，fence 后再 double-check 状态和 key 没变（防止在 fence 之间被 evictor 改写），命中就读记录算距离，记一次 `exact_cache_hits` 遥测，最后 `atomicSub readers -1` 放行。**double-check 是必须的**：reader pin 之间，evictor 可能正在把 slot 从 ready 改成 filling——pin 后再确认一次才能保证读到的一致数据。如果可见性检查失败（record 被锁/被删），同样算未命中。

```cpp
    if (cache_slot == UINT32_MAX && admit_exact_cache(params, handle)) {
      const u32 start_way = atomicAdd(params.exact_cache_victims + set, 1u) %
        params.exact_cache_ways;
      for (u32 attempt = 0; attempt < params.exact_cache_ways; ++attempt) {
        const u32 slot = set * params.exact_cache_ways +
          (start_way + attempt) % params.exact_cache_ways;
        const u32 state = *reinterpret_cast<volatile u32*>(
          params.exact_cache_states + slot);
        if (state == 1 ||
            atomicCAS(params.exact_cache_states + slot, state, 1u) != state) {
          continue;
        }
        u32 wait = 0;
        while (*reinterpret_cast<volatile u32*>(params.exact_cache_readers + slot) != 0 &&
               *reinterpret_cast<const volatile u32*>(params.stop) == 0 &&
               wait++ < kCacheWaitRounds) {
          device_ring_relax(128);
        }
        if (*reinterpret_cast<const volatile u32*>(params.exact_cache_readers + slot) != 0) {
          atomicCAS(params.exact_cache_states + slot, 1u, state);
          continue;
        }
        params.exact_cache_keys[slot] = handle;
        __threadfence();
        cache_slot = slot;
        request_cache_owned[index] = 1;
        break;
      }
    }
  }
```

未命中且 admission 通过时，尝试抢一个 filling 槽：从轮转 victim 开始扫 4 路，找到第一个能 CAS 到 state=1（filling）的 slot。抢到后等当前所有 reader 退出（`readers == 0`），写 key，fence，标记 `request_cache_owned[index]=1`。**等 reader 退出有 `kCacheWaitRounds=64` 上限**——超时就把 slot 还原回原 state，放弃这次缓存机会（落点会退到 per-query scratch）。这是为了避免 evictor 被慢 reader 卡死。

```cpp
  request_cache_slots[index] = cache_slot;
  request_shards[index] = shard;
  const u8* destination = cache_slot != UINT32_MAX
    ? params.exact_cache + static_cast<size_t>(cache_slot) * params.exact_cache_stride
    : params.exact_records +
        (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
          params.node_record_bytes;
  request_local_iova_offsets[index] =
    reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
}
```

落点选择：抢到 cache slot 就落 cache，否则落 per-query scratch（`exact_records[query_slot * exact_width + index]`）。这就是“图读 miss 直接进 scratch”的精确记录版本——同一个机制。

接下来（行 555–578）和 `approximate_handles_batch` 一样的合并/等待循环。**注意一个重要差异**：合并 fetch 时的 `destination_base` 是 `exact_records` 起始、`destination_stride = node_record_bytes`——但 `local_iova_offsets` 已经在前面 per-thread 算好了（cache slot 或 scratch），所以 `direct_fetch_batch` 内部会优先用 `local_iova_offsets[index]` 而不是 `destination_base + index*stride`（见 `direct_fetch_batch` 行 218–221 的三元运算符）。这让同一批 fetch 里混合 cache 落点和 scratch 落点成为可能。

```cpp
for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
  const u32 delta_slot = request_delta_slots[index];
  if (delta_slot != UINT32_MAX) { ... continue; }
  const u32 shard = request_shards[index];
  const u32 cache_slot = request_cache_slots[index];
  const bool cache_owned = request_cache_owned[index] != 0;
  if (shard != UINT32_MAX && shard_status[shard] != 0) {
    if (cache_owned) {
      __threadfence();
      atomicExch(params.exact_cache_states + cache_slot, 0u);
    }
    continue;
  }
  if (shard == UINT32_MAX && cache_slot == UINT32_MAX) continue;
  const u8* record = cache_slot != UINT32_MAX
    ? params.exact_cache + static_cast<size_t>(cache_slot) * params.exact_cache_stride
    : params.exact_records + ...;
  if (exact_record_visible(record)) {
    candidate_ids[index] = *reinterpret_cast<const u32*>(record + kNodeIdOffset);
    candidate_distances[index] = exact_storage_distance(
      params, query, record + kNodeVectorOffset);
  }
  if (shard != UINT32_MAX) atomicAdd(exact_reads, 1u);
  if (cache_owned) {
    __threadfence();
    atomicExch(params.exact_cache_states + cache_slot, 2u);
  }
}
```

结果回填阶段三种情况：
1. **远端读失败**（`shard_status != 0`）：如果抢了 cache slot，把 state 还原为 0（empty，让别的查询可以重新抢），跳过本候选。
2. **delta 路径**：直接用 `delta_vectors` 算距离。
3. **远端读成功**：record 现在在 cache slot 或 scratch，读出 id 和向量算距离。如果抢了 cache slot，fence 后把 state 推进到 `2`（ready）——**这一刻 cache 槽对外可见，其他 query 即可命中**。`exact_reads` 遥测 +1（仅对真正远端读的条目）。

函数剩余部分（行 622–655）是把 candidate 合并进 beam：把现有 beam + 新 candidate 拼成 merge 数组，`sort_candidates` 排序，截取 top-`beam_capacity` 写回 beam。这是 `exactify_into_beam` 的“into beam”部分，和 cache 无关，不展开。

### 八、graph_checksum / valid_graph_record：图页校验（rdma_cache.cuh:658–674）

```cpp
__device__ u16 graph_checksum(const u8* data, u32 bytes) {
  u32 hash = 2166136261u;
  for (u32 index = 0; index < bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}

__device__ bool valid_graph_record(const PersistentKernelParams& params, const u8* record) {
  const u16 stored = static_cast<u16>(record[2]) |
    static_cast<u16>(static_cast<u16>(record[3]) << 8);
  return record[0] <= params.graph_degree &&
    stored == graph_checksum(record, params.graph_entry_bytes);
}
```

FNV-1a 变体，跳过 byte 2/3（因为那两个字节本身存的就是 checksum）。`valid_graph_record` 还检查 `record[0] <= graph_degree`（度数不能越界）。**这个校验是图页 RDMA 读的关键**：存储侧 stage2/reverse-edge worker 会就地修改 compact 图条目（见第 15 课），所以一次 RDMA read 可能读到一个“半新半旧”的撕裂条目——checksum 不匹配就说明撕裂了，需要重读。代码里 `rdma_cache.cuh:921-926` 有专门的注释解释这一点。

### 九、prepare_graph_record：图页落点的三级决策（rdma_cache.cuh:676–823）

这是图页（navigation graph adjacency）的“落点选择”函数，比 exact 版本多了一级“anchor route cache”。它的决策树：**anchor_graph 路由命中 → graph_cache 命中 → graph_cache 抢 filling → graph_scratch**。

```cpp
__device__ bool prepare_graph_record(const PersistentKernelParams& params,
                                     u32 handle,
                                     u32 query_slot,
                                     u32 request_index,
                                     u32& acquired_slot,
                                     u32& request_shard,
                                     u64& request_offset,
                                     u64& request_local_iova,
                                     bool& cache_hit,
                                     bool& route_hit) {
  acquired_slot = UINT32_MAX;
  ...
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return false;
  const u64 graph_key = (static_cast<u64>(shard) << 48) | graph_offset;
```

`graph_key` 是 `(shard, graph_offset)` 的 64 位打包，作为 cache 的 key。`resolve_handle` 把 handle 拆成 raw node id、shard、graph_offset（图页在远端的偏移）。

```cpp
  const u32 route_slot = anchor_graph_slot(params, graph_key);
  if (route_slot != UINT32_MAX && params.anchor_graph_states != nullptr &&
      params.anchor_graph_readers != nullptr &&
      load_cg(params.anchor_graph_states + route_slot) == kGraphCacheReady) {
    atomicAdd(params.anchor_graph_readers + route_slot, 1u);
    __threadfence();
    if (load_cg(params.anchor_graph_states + route_slot) == kGraphCacheReady &&
        load_cg(params.anchor_graph_keys + route_slot) == graph_key) {
      acquired_slot = kGraphRouteBit | route_slot;
      route_hit = true;
      return true;
    }
    atomicSub(params.anchor_graph_readers + route_slot, 1u);
  }
```

**第一级：anchor_graph 路由缓存**。`anchor_graph_slot`（`candidate_scoring.cuh:92`）是二分查找——`anchor_graph_keys` 是预先排好序的“热点图页 key”数组（anchor 节点邻接表）。命中后用 reader pin 模式（和 exact_cache 一样的 double-check），返回 `kGraphRouteBit | route_slot` 作为 acquired_slot 标记。`route_hit=true` 让调用方记遥测（`route_hits`）。

anchor_graph 是一个**只读、固定容量**的路由层——它的内容由 CPU 在 `refresh_anchor_graph_records` 时一次性灌满，不像 graph_cache 那样有 eviction。所以命中它最便宜：根本不需要走 RDMA。

```cpp
  const u64 generation = load_cg(params.graph_cache_generation);
  const u32 set = params.graph_cache_sets == 0
    ? 0 : hash64(graph_key) % params.graph_cache_sets;
  const u32 way_count = params.graph_cache_ways;

  bool contended = false;
  for (u32 lookup_round = 0;
       lookup_round < 2 && params.graph_cache_sets != 0 && way_count != 0;
       ++lookup_round) {
    bool retry_lookup = false;
    for (u32 way = 0; way < way_count; ++way) {
      const u32 slot = set * way_count + way;
      const u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == kGraphCacheReady &&
          load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        atomicAdd(params.graph_cache_readers + slot, 1u);
        __threadfence();
        const u64 timestamp = load_cg(params.graph_cache_timestamps + slot);
        const u64 now = global_time_ns();
        if (*reinterpret_cast<volatile u32*>(params.graph_cache_states + slot) ==
              kGraphCacheReady &&
            load_cg(params.graph_cache_keys + slot) == graph_key &&
            load_cg(params.graph_cache_generations + slot) == generation &&
            (params.graph_cache_ttl_ns == 0 || now - timestamp <= params.graph_cache_ttl_ns)) {
          acquired_slot = slot;
          cache_hit = true;
          return true;
        }
        atomicSub(params.graph_cache_readers + slot, 1u);
      }
```

**第二级：graph_cache 命中**。和 exact_cache 的 reader pin 一模一样，但多了两个检查：
- `generations[slot] == generation`：generation 是全局失效版本号，delta 发布批量失效时会 bump（`runtime.cuh:374-415` 把对应 slot 改成 stale）。如果 generation 变了，缓存的内容可能已过时，不能命中。
- `ttl_ns`：可选的 TTL，`now - timestamp <= ttl_ns`。`ttl_ns=0` 表示永不超时（默认）。

```cpp
      if ((state == kGraphCacheFilling || state == kGraphCacheFillInvalidated) &&
          load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        u32 wait = 0;
        for (; wait < kCacheWaitRounds; ++wait) {
          const u32 current = *reinterpret_cast<volatile u32*>(
            params.graph_cache_states + slot);
          if ((current != kGraphCacheFilling &&
               current != kGraphCacheFillInvalidated) ||
              *reinterpret_cast<volatile u32*>(params.stop) != 0) break;
          device_ring_relax(128);
        }
        const u32 current = *reinterpret_cast<volatile u32*>(
          params.graph_cache_states + slot);
        retry_lookup = current != kGraphCacheFilling &&
          current != kGraphCacheFillInvalidated;
        contended = !retry_lookup;
        break;
      }
    }
    if (retry_lookup) continue;
    break;
  }
```

如果 slot 正在被别人 filling（state=1 或 4），等 `kCacheWaitRounds=64` 轮。等到了（state 变成 ready/stale/empty）就 `retry_lookup=true` 重查；超时了就 `contended=true` 放弃抢 filling（避免多个查询同时挤一个 slot）。`kGraphCacheFillInvalidated=4` 是一个特殊中间态：filling 过程中被失效（delta 发布），等 filling 完成后会被推到 stale 而不是 ready。

```cpp
  if (!contended && params.graph_cache_sets != 0 && way_count != 0 &&
      admit_graph_cache(params, graph_key)) {
    const u32 start_way = atomicAdd(params.graph_cache_victims + set, 1u) % way_count;
    for (u32 attempt = 0; attempt < way_count; ++attempt) {
      const u32 slot = set * way_count + (start_way + attempt) % way_count;
      u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == kGraphCacheFilling || state == kGraphCacheFillInvalidated ||
          atomicCAS(params.graph_cache_states + slot, state,
                    kGraphCacheFilling) != state) continue;
      u32 wait = 0;
      while (*reinterpret_cast<volatile u32*>(params.graph_cache_readers + slot) != 0 &&
             *reinterpret_cast<const volatile u32*>(params.stop) == 0 &&
             wait++ < kCacheWaitRounds) {
        device_ring_relax(128);
      }
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
        return false;
      }
      if (*reinterpret_cast<volatile u32*>(params.graph_cache_readers + slot) != 0) {
        const u32 current = *reinterpret_cast<volatile u32*>(
          params.graph_cache_states + slot);
        if (current == kGraphCacheFilling) {
          atomicCAS(params.graph_cache_states + slot, kGraphCacheFilling, state);
        } else if (current == kGraphCacheFillInvalidated) {
          atomicCAS(params.graph_cache_states + slot,
                    kGraphCacheFillInvalidated, kGraphCacheStale);
        }
        continue;
      }
      params.graph_cache_keys[slot] = graph_key;
      params.graph_cache_generations[slot] = generation;
      __threadfence();
      u8* destination = params.graph_cache +
        static_cast<size_t>(slot) * kPersistentGraphCacheLineBytes;
      acquired_slot = slot;
      request_shard = shard;
      request_offset = graph_offset;
      request_local_iova = reinterpret_cast<u64>(destination) -
        params.direct_local_iova_base;
      return true;
    }
  }
```

**第三级：抢 filling 槽**。和 exact_cache 一样的 CAS→wait readers→写 key 流程，但多了 generation 写入和 fill_invalidated 处理。抢到后落点是 `graph_cache[slot]`，准备发起 RDMA read。

```cpp
  if (*reinterpret_cast<volatile u32*>(params.stop) != 0 ||
      params.graph_scratch == nullptr || request_index >= kPersistentMaxPrefetch) {
    return false;
  }
  u8* destination = params.graph_scratch +
    (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
      kPersistentGraphCacheLineBytes;
  acquired_slot = kGraphScratchBit | request_index;
  request_shard = shard;
  request_offset = graph_offset;
  request_local_iova = reinterpret_cast<u64>(destination) -
    params.direct_local_iova_base;
  return true;
}
```

**第四级（兜底）：graph_scratch**。per-query 的 `kPersistentMaxPrefetch=32` 个 512B 槽（`construction.cc:259` 分配）。`acquired_slot = kGraphScratchBit | request_index`——最高位标记“这是 scratch”。注意 `request_index >= kPersistentMaxPrefetch` 时直接返回 false（失败），这是每查询 outstanding 的硬上限——每轮 prefetch 最多 32 个图页（`gpu_graph_prefetch_depth=32`，见 `configuration.hh:50`），超出就拒绝。

### 十、graph_record_pointer：从 acquired_slot 还原指针（rdma_cache.cuh:825–840）

```cpp
__device__ u8* graph_record_pointer(const PersistentKernelParams& params,
                                    u32 query_slot, u32 acquired_slot) {
  if ((acquired_slot & kGraphRouteBit) != 0) {
    const u32 route_slot = acquired_slot & kGraphSlotMask;
    return const_cast<u8*>(params.anchor_graph_records) +
      static_cast<size_t>(route_slot) * params.graph_entry_bytes;
  }
  if ((acquired_slot & kGraphScratchBit) != 0) {
    const u32 request_index = acquired_slot & kGraphSlotMask;
    return params.graph_scratch +
      (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
        kPersistentGraphCacheLineBytes;
  }
  return params.graph_cache +
    static_cast<size_t>(acquired_slot) * kPersistentGraphCacheLineBytes;
}
```

`acquired_slot` 是一个“带 tag 的句柄”：高 2 位是 tag（route / scratch），低 30 位是真正的 slot 索引。这个函数根据 tag 还原出 record 的 GPU 指针。三个分支对应三种落点。`kGraphSlotMask = ~(kGraphScratchBit | kGraphRouteBit) = 0x3fffffff`，能编码 10 亿个 slot——绰绰有余。

### 十一、fetch_graph_records_batch：图页批量拉取主流程（rdma_cache.cuh:842–1036）

这是被 `query_traversal.cuh:628` 调用的入口。它把一组 handle 转成 `(shard, offset, local_iova)` 三元组，按 shard 合并发起 RDMA，校验 checksum，重试失败条目。

```cpp
__device__ bool fetch_graph_records_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const u32* handles,
    u32 count,
    u32* acquired_slots,
    u32* remote_reads,
    u32* cache_hits,
    u32* route_hits,
    u32* remote_batches,
    u32* graph_read_retries) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  __shared__ u32 retry_pending;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iovas =
    params.dynamic_code_request_local_iovas + request_base;

  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    acquired_slots[index] = UINT32_MAX;
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iovas[index] = 0;
    remote_reads[index] = 0;
    cache_hits[index] = 0;
    route_hits[index] = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();
```

初始化阶段。`request_shards/offsets/local_iovas` 复用 `dynamic_code_request_*` 这同一块 per-query scratch——因为图页阶段和 PQ code 阶段不会同时进行，可以共享。遥测数组（`remote_reads/cache_hits/route_hits`）是 per-index 的，最后会被 `query_traversal` 累加到 per-query 总数。

```cpp
  constexpr u32 warp_width = 32;
  const u32 warp = threadIdx.x / warp_width;
  const u32 lane_in_warp = threadIdx.x % warp_width;
  const u32 warp_count = max(1u, blockDim.x / warp_width);
  if (lane_in_warp == 0) {
    for (u32 index = warp; index < count; index += warp_count) {
      bool cache_hit = false;
      bool route_hit = false;
      if (!prepare_graph_record(params, handles[index], descriptor.query_slot,
                                index, acquired_slots[index],
                                request_shards[index], request_offsets[index],
                                request_local_iovas[index], cache_hit,
                                route_hit)) {
        atomicExch(&failed, 1u);
      } else if (route_hit) {
        route_hits[index] = 1;
      } else if (cache_hit) {
        cache_hits[index] = 1;
      } else {
        remote_reads[index] = 1;
      }
    }
  }
  __syncthreads();
```

**只有每个 warp 的 lane 0 调用 `prepare_graph_record`**——因为该函数内部有 shared-memory lock、CAS、spin-wait，多 lane 并发反而会增加 contention。warp 间通过 `index += warp_count` 分摊工作。结果分四类：失败 / route 命中 / cache 命中 / 远端读。

```cpp
  if (failed != 0) {
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const u32 slot = acquired_slots[index];
      if (slot == UINT32_MAX) continue;
      if ((slot & kGraphRouteBit) != 0) {
        release_graph_record(params, slot);
      } else if ((slot & kGraphScratchBit) != 0) {
        acquired_slots[index] = UINT32_MAX;
      } else if (cache_hits[index] != 0) {
        release_graph_record(params, slot);
      } else {
        atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
      }
      acquired_slots[index] = UINT32_MAX;
    }
    __syncthreads();
    return false;
  }
```

如果有任何 handle 在 prepare 阶段失败（比如 stop 被设置、scratch 满），回滚所有已 acquired 的 slot——route/cache 命中的要 `release_graph_record`（`atomicSub readers`），filling 中的要还原成 empty。这是 fail-fast 一致性：要么全部准备好，要么全部回滚。

```cpp
  constexpr u32 kGraphSnapshotAttempts = 3;
  for (u32 attempt = 0; attempt < kGraphSnapshotAttempts; ++attempt) {
    if (threadIdx.x == 0) retry_pending = 0;
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      shard_status[shard] = 0;
    }
    __syncthreads();

    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      u32 matching = 0;
      for (u32 index = 0; index < count; ++index) {
        matching += request_shards[index] == shard ? 1u : 0;
      }
      if (matching != 0) {
        atomicAdd(remote_batches, 1u);
        if (attempt != 0 && graph_read_retries != nullptr) {
          atomicAdd(graph_read_retries, matching);
        }
      }
      i32* owner_completion = ...;
      shard_status[shard] = direct_fetch_batch(
          params, shard, request_shards, request_offsets, count,
          params.graph_cache, kPersistentGraphCacheLineBytes,
          params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, owner_completion, true);
    }
```

**最多 3 次 snapshot 重试**。每次重试都重新发起 direct_fetch_batch——但只对还没成功的 shard（`request_shards[index]` 没被改成 UINT32_MAX 的）会真正发 WQE，已成功的 shard `matching=0` 会被 `direct_fetch_batch` 第一段短路。

`remote_batches` 遥测：每个 shard 的每次 attempt 都 +1，这就是“批次计数”——衡量合并后实际下了几批 RDMA。`graph_read_retries` 仅在重试时累加 `matching`，衡量重试涉及的条目数。

```cpp
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != -EINPROGRESS) continue;
      i32* owner_completion = ...;
      shard_status[shard] = wait_direct_batch(params, owner_completion);
    }
    __syncthreads();

    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const u32 shard = request_shards[index];
      if (shard == UINT32_MAX) continue;
      const u32 slot = acquired_slots[index];
      const bool scratch = (slot & kGraphScratchBit) != 0;
      u8* record = graph_record_pointer(params, descriptor.query_slot, slot);
      const i32 status = shard_status[shard];
      const bool valid = status == 0 && valid_graph_record(params, record);
      if (valid) {
        if (!scratch) {
          params.graph_cache_timestamps[slot] = global_time_ns();
          params.graph_cache_readers[slot] = 1;
          __threadfence();
          const u32 state = atomicCAS(params.graph_cache_states + slot,
                                      kGraphCacheFilling, kGraphCacheReady);
          if (state == kGraphCacheFillInvalidated) {
            atomicCAS(params.graph_cache_states + slot,
                      kGraphCacheFillInvalidated, kGraphCacheStale);
          } else if (state != kGraphCacheFilling) {
            atomicExch(&failed, 1u);
          }
        }
        request_shards[index] = UINT32_MAX;
        continue;
      }
```

校验阶段。`valid = (status == 0) && valid_graph_record(...)`——RDMA 成功但 checksum 不匹配也算 invalid（撕裂读）。valid 的情况：
- 如果是 graph_cache 槽（非 scratch），把 timestamp 设为当前时间（用于 TTL），`readers=1`（自己 pin 住，下面 traversal 用完会 `release_graph_record`），fence 后 CAS `filling → ready`。
- 如果 CAS 返回 `kGraphCacheFillInvalidated`，说明 filling 期间被失效了，推到 `stale`（让后续查询重新拉）。
- 如果 CAS 返回的不是 filling 也不是 fill_invalidated（说明被别人改了，异常），fail。
- `request_shards[index] = UINT32_MAX` 把这条移出后续重试批次，但 `acquired_slots[index]` 保留——traversal 还要用。

```cpp
      if (status == 0 && attempt + 1 < kGraphSnapshotAttempts) {
        atomicAdd(&retry_pending, 1u);
        continue;
      }

      __threadfence();
      if (!scratch) {
        atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
      }
      acquired_slots[index] = UINT32_MAX;
      request_shards[index] = UINT32_MAX;
      atomicExch(&failed, 1u);
      if (status == 0) {
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -EBADMSG);
        }
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncthreads();
    if (retry_pending == 0) break;
    if (threadIdx.x == 0) device_ring_relax(128);
    __syncthreads();
  }
```

invalid 处理：
- RDMA 成功但 checksum 失败，且还有重试次数：`retry_pending++`，下一轮重发。
- RDMA 失败，或重试用完：把 slot 还原 empty，fail。如果是 checksum 失败（status==0 但 valid==false）用尽重试，记 `-EBADMSG` 并 fail-stop——因为持续撕裂说明存储侧有严重问题。

`retry_pending == 0` 跳出循环。`device_ring_relax(128)` 在重试间给存储侧一点时间让 in-flight 发布落地。

```cpp
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    if (route_hits[index] == 0) continue;
    const u32 slot = acquired_slots[index];
    const u8* record = graph_record_pointer(
      params, descriptor.query_slot, slot);
    if (!valid_graph_record(params, record)) {
      atomicExch(&failed, 1u);
      if (params.direct_error != nullptr) {
        atomicCAS(params.direct_error, 0, -EBADMSG);
      }
      atomicExch(params.direct_disabled, 1u);
    }
  }
  __syncthreads();
  return failed == 0;
}
```

最后还要校验 route 命中的条目——anchor_graph 是 CPU 灌的只读缓存，理论上不会撕裂，但万一存储侧发布了坏数据并被 CPU 抓取灌进来，这里能 catch 住。同样 fail-stop。

### 十二、direct_read_owner_loop：owner warp 的消费循环（runtime.cuh:710–938）

现在我们终于能看到“被合并的请求是谁在消费”了。`direct_read_owner_kernel`（`runtime.cuh:940`）是一个独立的 `__global__` kernel，由 `lifecycle.cc:237` 通过 `direct_owner_block_count` 调度——它在 `persistent_search_kernel`（`runtime.cuh:11`）里被优先分配 blockIdx：`blockIdx.x < direct_owner_block_count` 的 block 都跑 owner loop，剩下的才跑查询/dispatcher/delta。

```cpp
__device__ void direct_read_owner_loop(PersistentKernelParams params,
                                       u32 queue_count,
                                       u32 owner_block) {
  constexpr u32 warp_width = 32;
  constexpr u32 max_warps_per_block = 8;
  constexpr u32 max_submit_batches = 8;
  const u32 lane = threadIdx.x % warp_width;
  const u32 warps_per_block = blockDim.x / warp_width;
  const u32 warp_in_block = threadIdx.x / warp_width;
  const u32 warp = owner_block * warps_per_block + warp_in_block;
  if (warps_per_block == 0 || warps_per_block > max_warps_per_block ||
      warp >= queue_count) return;
```

每个 owner block 最多 8 个 warp（256 thread / 32）。每个 warp 绑定到一个 QP（`warp = owner_block * warps_per_block + warp_in_block`，对应 `direct_batch_queues[warp]` 和 `direct_qps[warp]`）。`queue_count = direct_batch_queue_count = qps_per_node * remote_region_count`（`construction.cc:555`），即“lane × memory_node”的全组合。**每个 QP 都有一个专属 owner warp 在 spin**——这是为什么 owner path 不会成为瓶颈：QP 间完全独立。

```cpp
  u32 invalid = 0;
  invalid |= params.direct_batch_queues == nullptr ? 1u : 0u;
  invalid |= params.direct_qps == nullptr ? 2u : 0u;
  ...
  if (invalid != 0) {
    if (lane == 0 && params.direct_owner_phases != nullptr) {
      params.direct_owner_phases[warp] = 0x100u | invalid;
      __threadfence_system();
    }
    return;
  }
```

启动自检：把任何一项缺失记到位掩码写到 `direct_owner_phases[warp]`，让 CPU 侧能诊断（`lifecycle.cc:231` 等 owner 启动时会读这个数组）。

```cpp
  __shared__ DirectBatchDescriptor shared_batches
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_matching_counts
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_wqe_offsets
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_batch_counts[max_warps_per_block];
  __shared__ u32 shared_total_wqes[max_warps_per_block];

  const u32 memory_node = warp % params.direct_region_count;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[warp]);
  ...
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const bool need_dump = qp->need_dump;
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  const DeviceRingView<DirectBatchDescriptor> queue =
    params.direct_batch_queues[warp];
```

per-warp 状态：QP、CQ、远端 region、绑定到的 ring。`memory_node = warp % direct_region_count`——和 QP 索引布局一致。

```cpp
  const u32 initial_idle_cycles = 256u + ((warp * 97u) & 2047u);
  u32 idle_cycles = initial_idle_cycles;
  for (;;) {
    const u32 stop_requested = lane == 0
      ? *reinterpret_cast<const volatile u32*>(params.stop)
      : 0u;
    if (__shfl_sync(0xffffffffu, stop_requested, 0) != 0) break;

    if (lane == 0) {
      u32 batch_count = 0;
      u32 total_wqes = 0;
      while (batch_count < max_submit_batches) {
        DirectBatchDescriptor descriptor{};
        u32 matching = 0;
        if (have_deferred) {
          descriptor = deferred;
          matching = deferred_matching;
          have_deferred = false;
        } else if (!device_ring_try_pop(queue, descriptor)) {
          break;
        } else {
          ...
          if (descriptor.memory_node == memory_node && ...) {
            for (u32 index = 0; index < descriptor.request_count; ++index) {
              matching += descriptor.request_shards[index] == memory_node ? 1u : 0;
            }
          }
        }
        ...
        const u32 needed = matching + (need_dump ? 1u : 0u);
        if (needed > qp->sq_wqe_num) {
          complete_direct_batch(descriptor, -E2BIG);
          continue;
        }
        if (batch_count != 0 && total_wqes + needed > qp->sq_wqe_num) {
          deferred = descriptor;
          deferred_matching = matching;
          have_deferred = true;
          break;
        }
        shared_batches[warp_in_block][batch_count] = descriptor;
        shared_matching_counts[warp_in_block][batch_count] = matching;
        shared_wqe_offsets[warp_in_block][batch_count] = total_wqes;
        ++batch_count;
        total_wqes += needed;
      }
      shared_batch_counts[warp_in_block] = batch_count;
      shared_total_wqes[warp_in_block] = total_wqes;
    }
    __syncwarp();
```

**这是“合并”最核心的逻辑**：lane 0 在一个循环里从 ring 尽量多 pop batch（最多 8 个），统计每个 batch 需要的 WQE 数，累计 `total_wqes`。如果加上下一个 batch 会超过 `sq_wqe_num`（SQ 容量），就把该 batch `deferred` 留到下一轮，跳出循环。这样一轮最多 submit `sq_wqe_num` 条 WQE，把多个查询 CTA 提交的多个 batch **二次合并**成一次大 submit——这就是 `rdma_merged_requests` 遥测度量的“合并”。

注意 `complete_direct_batch` 在错误时直接写 `completion_status` 让调用方 CTA 收到失败——但 owner 不会因此停（除非是 disabled 引起的）。

```cpp
    const u32 batch_count = shared_batch_counts[warp_in_block];
    if (batch_count == 0) {
      if (lane == 0) device_ring_relax(idle_cycles);
      __syncwarp();
      idle_cycles = min(idle_cycles * 2, 16384u);
      continue;
    }
    idle_cycles = initial_idle_cycles;
```

空转退避：`idle_cycles` 从 `256 + warp*97 % 2048` 起步（不同 warp 错峰），每次空翻倍到 16384 上限。有活干就重置回初始值。

```cpp
    const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
    const doca_gpu_dev_verbs_ticket_t first_completion =
      doca_gpu_dev_verbs_load_relaxed<...>(&completion_queue->cqe_ci);
    for (u32 batch = 0; batch < batch_count; ++batch) {
      const DirectBatchDescriptor descriptor =
        shared_batches[warp_in_block][batch];
      const u32 matching = shared_matching_counts[warp_in_block][batch];
      const u32 batch_offset = shared_wqe_offsets[warp_in_block][batch];
      u32 matched_before = 0;
      for (u32 base = 0; base < descriptor.request_count; base += warp_width) {
        const u32 index = base + lane;
        const bool matching_request = index < descriptor.request_count &&
          descriptor.request_shards[index] == memory_node;
        const u32 matching_mask = __ballot_sync(0xffffffffu, matching_request);
        if (matching_request) {
          const u32 lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
          const u32 rank = __popc(matching_mask & lower_lanes);
          const u32 matched = matched_before + rank;
          const doca_gpu_dev_verbs_ticket_t ticket =
            first_wqe + batch_offset + matched;
          auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
          const bool last_read = matched + 1 == matching;
          const auto flags = !need_dump && last_read
            ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
          doca_gpu_dev_verbs_wqe_prepare_read(
            qp, wqe, ticket, flags,
            region.address + descriptor.remote_offsets[index], region.rkey,
            descriptor.local_iova_offsets[index], params.direct_local_mkey,
            descriptor.bytes);
        }
        matched_before += __popc(matching_mask);
      }
      ...
    }
```

**WQE prepare 全 warp 并行**：每个 lane 处理 request 数组的一个 index，用 `__ballot_sync` + `__popc` 算出“自己是这一批里第几个 matching 的”，对应到 `first_wqe + batch_offset + matched` 这个 WQE 位置。这是非常精巧的 warp-level 并行——32 个 lane 同时填 32 个 WQE，比 lane 0 串行快 32 倍。`matched_before` 跨 `base` 循环累加，保证 ticket 连续。

```cpp
    __syncwarp();
    if (lane == 0) {
      ...
      doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
          qp, first_wqe + shared_total_wqes[warp_in_block]);
      ...

      i32 status = 0;
      for (u32 batch = 0; batch < batch_count; ++batch) {
        if (status == 0) {
          status = poll_direct_cq(completion_queue, first_completion + batch,
                                  params.direct_timeout_ns, params.stop,
                                  params.direct_disabled);
        }
        complete_direct_batch(shared_batches[warp_in_block][batch], status);
      }
      ...
      if (status != 0 && status != -ECANCELED) {
        if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncwarp();
  }
}
```

lane 0 单线程 submit + 串行 poll CQ。**关键设计：每个 batch 只 poll 一个 CQE**（`first_completion + batch`，因为每个 batch 内只有最后一条 READ 用 `CQ_UPDATE`）。如果某个 batch 失败，后续 batch 直接 `complete_direct_batch(..., status)` 用同一个 status 完成它们——因为 QP 失败后通常需要 reset，后续 batch 也不会成功。非 ECANCELED 的失败都 fail-stop。

### 十三、direct_read_owner_kernel / probe kernels（runtime.cuh:940–1029）

`direct_read_owner_kernel` 就是 `__global__` 包装的 `direct_read_owner_loop`，给 `lifecycle.cc` 调用。

后面三个 probe kernel 是诊断工具：
- `gpunetio_locked_read_probe_kernel`（行 945）：每个 warp 串行调 `direct_fetch`，测 locked 直发路径。
- `gpunetio_batched_read_probe_kernel`（行 968）：测 `direct_fetch_batch` 同步路径（`defer_owner_wait=false`）。
- `gpunetio_owner_read_probe_kernel`（行 998）：测 owner warp 异步路径，把 `phases` 写到 `owner_progress` 让 CPU 观测。

`gpunetio_transport.cc:514` 启动时跑的就是这些 probe——确认每条 QP 都能正常 RDMA read 才让查询 kernel 启动。Probe 失败会抛异常带详细的 debug 字段（cqe/ticket/opown 等）。

---

## 关键数据结构与流程图

### 图缓存状态机

```
                 admit_graph_cache 通过
                         │
                         │  CAS(state, 任意, Filling)
                         ▼
  ┌─────────┐  CAS    ┌─────────┐  RDMA ok   ┌──────────┐
  │  Empty  │ ──────► │ Filling │ ─────────► │   Ready  │ ◄──┐
  │  (0)    │         │  (1)    │            │   (2)    │    │
  └─────────┘         └─────────┘            └──────────┘    │
       ▲                  │ CAS                  │           │
       │                  │ FillInvalidated      │ delta     │
       │                  │ (4)                  │ invalid.  │
       │                  ▼                      ▼           │
       │             ┌─────────┐  RDMA ok   ┌─────────┐      │
       └─────────◄── │ FillInv │ ─────────► │ Stale   │ ─────┘
         CAS empty   │  (4)    │            │  (3)    │ 重读
                     └─────────┘            └─────────┘
```

- Empty: 可被新 filling 抢占
- Filling: 某查询正在 RDMA 拉取
- Ready: 可读，reader pin 模式
- Stale: delta 发布过期，可被重读刷新
- FillInvalidated: filling 过程中被 delta 失效，完成后落 Stale 而非 Ready

### acquired_slot 的 tag 编码

```
  bit 31    bit 30      bit 29..0
 ┌────────┬──────────┬─────────────────┐
 │scratch │  route   │   slot index    │
 │  bit   │   bit    │                 │
 └────────┴──────────┴─────────────────┘
   1         0          scratch 索引   → graph_scratch
   0         1          route slot     → anchor_graph_records
   0         0          cache slot     → graph_cache
```

### 查询 CTA 提交读请求 → owner warp 合并 → CQ 完成 → 回填 状态机

```
 查询 CTA (per shard)                  owner warp (per QP)                 NIC/CQ
 ─────────────────────                  ──────────────────                  ──────

 1. prepare_graph_record
    选落点: route / cache / scratch
    抢 filling (CAS state=1)
        │
        ▼
 2. direct_fetch_batch(defer=true)
    构造 DirectBatchDescriptor
    device_ring_try_push ──────────►  3. lane 0 循环 pop batch
                                         累计 WQE，超 sq_wqe_num 则 defer
                                         全 warp 并行 prepare WQE
                                         lane 0 submit ──────────────►  4. NIC 执行 READ
                                                                                                     │
                                         5. lane 0 poll_direct_cq ◄──────────────────────────────────  CQ 产 CQE
                                            (每 batch 1 个 CQE)
                                         6. complete_direct_batch
                                            atomicExch(completion, status) ◄──── 7. status (0/err)
    │                                                                                               │
    ▼
 8. wait_direct_batch
    spin on owner_completion
        │
        ▼
 9. valid_graph_record (checksum)
    ├─ valid: cache slot → CAS filling→ready, scratch → 直接用
    └─ invalid:
       ├─ attempt<3: retry_pending++, 下一轮重发
       └─ attempt=3: fail-stop (-EBADMSG → direct_disabled=1)
```

### 多查询并发图

```
 Q0 CTA ─┐
         ├──► direct_batch_queues[Q0.lane × nodes + shard0] ──┐
 Q1 CTA ─┤                                                    │
         ├──► direct_batch_queues[Q1.lane × nodes + shard0] ──┤  owner warp[q0]
 Q2 CTA ─┘                                                    ├── (合并 pop 多 batch)
         ├──► direct_batch_queues[Q2.lane × nodes + shard0] ──┘  (一次 submit ≤ sq_wqe_num 条 WQE)
                                                              │
                                                              ▼
                                                            QP[lane × nodes + shard0]
                                                              │
                                                              ▼
                                                          NIC → 远端 shard0
```

**每个 (lane, memory_node) 组合一个 ring + 一个 owner warp + 一个 QP**。查询 CTA 按 `(query_slot + shard) % direct_qps_per_node` 选 lane，使得不同 query 对同一 shard 散列到不同 ring，进一步减少单 QP 竞争。

---

## 与其他模块的关系

- **第 17 课（kernel 启动器/上下文/device ring）**：`direct_batch_queues` 就是 device ring，`PersistentKernelParams` 的装配在第 17 课讲过。本课所有 `device_ring_try_push/try_pop` 都建立在第 17 课的 `DeviceRingView` 上。
- **第 18 课（候选评分）**：`approximate_handle` / `approximate_entry` / `resolve_handle` / `delta_slot_from_raw` 等都在 `candidate_scoring.cuh`，本课的 `approximate_handles_batch` 是它们的批量包装。
- **第 20 课（查询遍历主循环）**：`process_query`（`query_traversal.cuh`）在主循环里调用 `fetch_graph_records_batch`（prefetch 阶段）→ `approximate_handles_batch`（PQ 评分阶段）→ `exactify_into_beam`（rerank 阶段）。本课是这三个调用的实现。
- **第 21 课（kernel 运行时/角色调度）**：`persistent_search_kernel` 的“owner block 优先分配 + 查询/dispatcher/delta block 跟进”模式在 `runtime.cuh:11–37`，第 21 课会详细讲角色调度。
- **第 22 课（GPUNetIO 传输/probe）**：`gpunetio_transport.cc` 装配 send_cq/recv_cq/qp_wq/qp_dbr umem 的细节、QP 状态机（INIT/RTR/RTS）、startup probe 的 debug 字段含义都在第 22 课讲。本课只在 `direct_fetch` / `direct_fetch_batch` 里使用这些装配好的 QP。
- **第 15 课（增量发布）**：`runtime.cuh:374–415` 的 graph_cache 失效逻辑（`Ready → Stale`、`Filling → FillInvalidated`）由 delta 发布触发。本课的 `prepare_graph_record` 通过 `generation` 检查感知失效。
- **第 10 课（delta/动态路由/预算）**：`approximate_handles_batch` 里的 delta/resident/dynamic route 决策树是第 10 课在查询侧的兑现。

---

## 小结

`rdma_cache.cuh` 把“GPU 持久化 kernel 里发起 RDMA read”这件事拆成了三层：

1. **落点决策层**（`prepare_graph_record` / `exactify_into_beam` 的 cache 分支 / `approximate_handles_batch` 的请求构造）：决定每条读请求的数据应该落到哪里——优先 anchor_graph 路由（只读、最快）→ graph_cache 命中（reader pin）→ graph_cache filling（CAS 抢占 + 等 reader 退出）→ graph_scratch（per-query 兜底）。每查询 scratch 槽位数 (`kPersistentMaxPrefetch=32`) 是 outstanding 上限，`gpu_graph_prefetch_depth=32` 控制每轮 prefetch 量。

2. **合并调度层**（`direct_fetch_batch` + `direct_read_owner_loop`）：查询 CTA 不亲自敲 doorbell，而是把 `DirectBatchDescriptor` 入 per-QP ring；owner warp 在 ring 上多批 pop、累计 WQE 数到 `sq_wqe_num` 上限，全 warp 并行 prepare、一次 submit。每个 batch 内只有最后一条 READ 产 CQE，把 CQ 压力降到 `批次数` 而非 `WQE 数`。`rdma_merged_requests` 遥测度量这一层的合并比。

3. **完成与校验层**（`poll_direct_cq` + `valid_graph_record` + `complete_direct_batch`）：owner warp poll CQ，把 status 通过 `atomicExch(completion_status, ...)` 通知查询 CTA。查询 CTA 用 `wait_direct_batch` 收尾，对结果做 checksum 校验——失败重试最多 3 次（`graph_read_retries` 遥测），重试用尽或硬件错误就 `atomicExch(direct_disabled, 1u)` 进入 fail-stop。

整个体系的精髓在于：**把 RDMA 的“发起—完成”解耦成 producer（查询 CTA）和 consumer（owner warp）**，用 per-QP ring 做背压安全的扇入，用 per-batch CQE 做批量完成，用 reader-pin + CAS 状态机做并发安全的 cache 准入/淘汰，用 generation + checksum 做 delta 一致性与撕裂读防护。当你看到 `graph_page_cache_hits` 上升、`rdma_merged_requests` 上升、`graph_read_retries` 接近 0 时，就说明这套机制运转良好；任何一个 fail-stop 触发都会让 `direct_disabled=1`，所有查询立刻返回 `-EHOSTDOWN`，由上层（见第 27 课计算服务主体）切到 CPU 代理 RPC fallback。
