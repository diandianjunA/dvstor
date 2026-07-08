# 第 05 课：全局资源上下文与内存分配

## 本课目标

本课讲清 compute node 本地内存、RDMA memory region、QP pool、completion slot、buffer allocator 的关系。理解这层后，你才能判断一次 RDMA READ 的本地地址、lkey、远端地址、rkey、completion 是如何组织起来的。

## 代码证据

必须阅读：

- `src/shared_context.hh`
- `src/buffer_allocator.hh`
- `rdma-library/library/hugepage.hh`
- `rdma-library/library/memory_region.hh`
- `src/worker_pool.hh`
- `src/rdma/vamana_rdma_reads.hh`

## WorkerPool 如何分配资源

`WorkerPool` 构造时创建一个全局 `BufferAllocator`：

```cpp
BufferAllocator buffer_allocator_(num_compute_threads, buffer_size_bytes)
```

然后 `allocate_worker_threads` 做两类对象：

1. 创建 `SharedContext<ComputeThread>`，数量最多 `MAX_QPS`。
2. 创建 `ComputeThread`，每个 thread 注册到一个 shared context。

映射关系：

```text
ComputeThread id
  -> shared_contexts_[id % MAX_QPS]
```

这意味着多个 compute thread 可能共享一个 `SharedContext`，从而共享其中的 QP pool 和 local memory region。

## BufferAllocator

`BufferAllocator` 在 hugepage 上分配一整块本地 buffer：

```text
HugePage<byte_t> local_buffer_
byte_t* buffer_ptr_
atomic bump_pointer_
freelists_by_size_
```

分配逻辑：

- `allocate_buffer(size)`：按 cacheline 对齐，先查对应 size freelist，没有则 bump allocate。
- `allocate_vamana_node(thread_id)`：分配 `VamanaNode::total_size()`。
- `allocate_pointer()`：分配 8 字节 pointer slot。
- `free_buffer(ptr, size)`：按对齐 size 放回 freelist，8 字节及以下不回收。

重要事实：这是 compute node 上的 RDMA registered staging buffer。很多 RDMA READ 都读到这里，然后再构造 `VamanaNode` 或 `VamanaNeighborlist` view。

## BufferAllocator 的生命周期

典型例子：

```text
read_vamana_node
  allocate_buffer(read_size)
  RDMA READ 到 local buffer
  await_resume 返回 shared_ptr<VamanaNode>
  VamanaNode 析构
  owner_->buffer_allocator.free_buffer(buffer_slice_, buffer_size_)
```

neighbor list 类似：

```text
read_vamana_neighbors
  allocate_buffer(read_size)
  RDMA READ
  await_resume 返回 shared_ptr<VamanaNeighborlist>
  VamanaNeighborlist 析构
  free_buffer
```

这说明 buffer 的释放依赖 RAII wrapper 析构。如果手动提前释放或 wrapper 生命周期错误，就会出现 use-after-free。

## SharedContext 的职责

`SharedContext<T>` 封装 compute node 侧对所有 memory node 的 RDMA 通道：

```text
Context context
qps[memory_node][pool_index]
qp_runtime[memory_node][pool_index]
registered_threads
memory_region
remote_mrts
batch_options
completion_slots
qp_tie_breakers
```

构造时：

1. 为每个 memory node 创建 `qp_pool_size` 个 `DetachedQP`。
2. 连接到 memory node 提供的 server QP。
3. 创建本地 `LocalMemoryRegion`，注册 `BufferAllocator` 的 hugepage buffer。
4. 初始化 batch completion slots。

## QP runtime state

每个 QP 有：

- `send_wr_capacity`
- `outstanding_wrs`
- `outstanding_chunks`
- `high_water_wrs`

核心方法：

- `qp_credit_limit(node, qp)`
- `try_reserve_qp_wrs(node, qp, wr_count)`
- `try_reserve_bulk_qp_wrs(node, preferred_qp, wr_count, selected_qp)`

这套逻辑服务于批量 RDMA READ。它避免向同一个 QP 过量 post WR，留下 reserve 空间，降低 ENOMEM/EAGAIN/EBUSY 风险。

## completion slot

批量 RDMA READ chain 通常只在最后一个 WR 上设置 signaled。为了让 completion 能归还多个 outstanding WR credit，`SharedContext` 创建了 `CompletionSlot`：

```cpp
struct CompletionSlot {
  atomic<bool> in_use;
  u32 thread_index;
  u32 coroutine_id;
  u32 memory_node;
  u32 qp_index;
  u32 wr_count;
};
```

`try_create_batch_completion` 返回一个带高位标记的 `wr_id`：

```text
kBatchCompletionFlag | (slot_index + 1)
```

completion 到达后 `complete_send`：

- 从 slot 找到 memory node、QP、wr_count、thread、coroutine。
- `outstanding_wrs -= wr_count`
- `outstanding_chunks -= 1`
- 对应 coroutine 的 `post_balances -= 1`
- slot 释放。

## 普通 completion 和 batch completion 的区别

普通 RDMA：

```text
wr_id = encode_64bit(ctx_tid, coroutine_id)
completion -> post_balances[coroutine_id]--
```

批量 vector RDMA：

```text
wr_id = kBatchCompletionFlag | slot
completion -> 释放 QP credit + post_balances[coroutine_id]--
```

区别在于批量 completion 同时承担 QP outstanding accounting。

## 本地和远端地址

一次 RDMA READ 需要：

- local addr
- local lkey
- remote addr
- remote rkey

local addr 常见来源：

- `BufferAllocator` 分配的 host buffer。
- GPU device buffer，前提是 GPUDirect RDMA 注册成功。

local lkey 常见来源：

- `thread->ctx->get_lkey()`：compute local hugepage memory region。
- `gs.d_candidate_vecs_lkey`：GPU candidate buffer MR 的 lkey。

remote addr/rkey 来源：

- `thread->ctx->get_remote_mrt(memory_node)`。
- `remote_mrt->address + resolved_offset`。

## 内存布局图

```text
Compute node process
  BufferAllocator hugepage
    pointer slots
    RDMA node read buffers
    neighbor read buffers
    vector staging buffers
  LocalMemoryRegion
    registers whole hugepage
  SharedContext
    qps[MN][QP]
    completion slots
    remote memory tokens

Memory node process
  index_buffer_ hugepage
    offset 0: free pointer
    offset 8: medoid pointer
    offset 16..: nodes
  MemoryRegionToken
    sent to compute node
```

## 性能影响

1. 全局 hugepage buffer 降低 TLB 压力，但 bump allocator 不真正归还整体空间。
2. freelist 按 size 复用 buffer，可减少热路径 malloc/free。
3. `freelist_mutex_` 是全局 mutex，在高并发不同线程释放 buffer 时可能成为争用点。
4. batch completion 减少 CQE 数量，但需要 completion slot 池。slot 不足会产生 `vector_rdma_completion_token_waits`。
5. QP credit reserve 避免过载，但 credit wait 会增加查询延迟。

## 设计异味

1. `BufferAllocator` 构造参数 `num_threads` 当前未使用。
2. `allocated_buffers_` 记录但没有明显释放单个 buffer 的策略，更多是 bump + freelist。
3. `free_buffer` 不回收 8 字节 pointer，这需要明确说明，否则容易误判内存泄漏。
4. `SharedContext` 同时管理 QP pool、completion slots、memory region、thread registry，职责较重。
5. batch completion 使用 wr_id 高位编码，协议隐式，适合封装成更强类型。

## 可验证问题

- `BufferAllocator` 分配的内存什么时候注册为 RDMA MR？
- 为什么批量 RDMA chain 只需要一个 signaled completion？
- completion slot 不足时会发生什么？
- GPUDirect RDMA 成功时 local lkey 来自哪里？
- `qp_credit_limit` 为什么要保留 reserve？

## 学习任务

1. 跟踪 `batch_read_vectors` 中 local addr 和 remote addr 的构造。
2. 画出 `CompletionSlot` 从创建、使用到释放的生命周期。
3. 搜索 `vector_rdma_completion_token_waits`，看它如何进入统计报告。
4. 思考：如果 freelist mutex 成为瓶颈，可以如何改成 per-thread freelist？
5. 思考：如果支持多个索引，`BufferAllocator` 和 `SharedContext` 是否需要拆分？

