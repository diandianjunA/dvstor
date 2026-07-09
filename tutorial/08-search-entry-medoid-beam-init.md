# 第 08 课：在线查询主路径之一：入口点、medoid 与 beam 初始化

## 本课目标

本课从 `ComputeService::search` 开始跟踪到 `Vamana::knn_raw`，讲清一次查询如何进入队列、如何选择 entry point、如何读取 medoid、如何初始化 beam 和 visited。理解这个开头，后续批量 expansion 和 GPU 路径才能连起来。

## 代码证据

必须阅读：

- `src/service/compute_service/search.ipp`
- `src/http/vamana_service_scheduler.hh`
- `src/vamana/vamana_search.ipp`
- `src/rdma/vamana_rdma_reads.hh`
- `src/vamana/anchor_index.hh`

## ComputeService 搜索入口

公开方法：

- `search(const vec<element_t>& query, u32 k)`
- `search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k)`

如果没有 routing，`search` 直接走：

```text
search
  -> search_local
  -> search_local_result
```

`search_local_result` 创建 `QueryRequest`：

```text
QueryRequest
  components 或 raw_components
  entry_points
  query_dtype
  k
  enqueued_at
  breakdown_sample
  promise<QueryResult>
```

然后：

```text
query_queue_.enqueue(request)
future.get()
```

所以 `ComputeService::search` 对调用者是同步 API，对内部 worker 是异步队列。

## service scheduler 接管请求

query worker 在 `vamana_service_schedule_queries` 中：

1. 从 `query_queue_` 取 `QueryRequest*`。
2. 设置当前 coroutine 的 breakdown sample。
3. 如果 request 有 `raw_components`，调用 `vamana_idx.knn_raw`。
4. 否则拷贝 float components 到 staging slot，调用 `vamana_idx.knn`。
5. coroutine 完成后从 `thread->query_results[q_id]` 取结果，设置 promise。

这里的 `q_id` 不是用户业务 query id，而是 scheduler 内部 slot id。

## entry points 的来源

`ComputeService::storage_owner_query_entry_points` 在 local-stitch 模式下提供 anchor entry points：

```text
if storage_owner_update_mode != "local_stitch"
  return empty
if anchor_index_ == nullptr or empty
  return empty
nearest_shards(query, 2)
for each shard:
  nearest_anchors(query, shard, storage_owner_anchor_hints)
deduplicate RemotePtr
```

如果 entry points 为空，`Vamana::knn_raw` 会回退到 medoid。

这说明查询入口点有两个来源：

- local-stitch anchor hints。
- 全局 medoid pointer。

## knn_raw 初始化

`Vamana::knn_raw` 开始时：

```text
++thread->stats.processed
++thread->stats.processed_queries
coro_state = thread->current_vamana_coroutine()
beam = coro_state.beam
visited = coro_state.visited_nodes
gpu = thread->gpu_buffers
gs = gpu.state(coro_id)
```

如果不使用 RaBitQ，会立即把 query 拷贝到 pinned host buffer，然后 H2D：

```text
memcpy(gs.h_query, query_data)
cudaMemcpyAsync(gs.d_query, gs.h_query, query_bytes, H2D)
track_query_h2d
```

如果使用 RaBitQ，则这里不会立刻上传 exact query，而是在 RaBitQ gate 需要 exact rerank 时再上传。

## add_start_point

`knn_raw` 内部定义 `add_start_point`：

```text
if ptr null or visited contains ptr:
  return
read_node(ptr)
distance_to_stored_vector(query, node->vector_data())
beam.push_back({ptr, dist, false})
visited.insert(ptr)
```

注意这里距离是在 CPU 上算的，不是 GPU。原因是 entry point 数量很少，读一个节点后直接算初始距离更简单。

## medoid 读取

如果没有 entry points 或 entry points 全部无效：

```text
RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread)
add_start_point(medoid_ptr)
```

`read_medoid_ptr` 本质是从 memory node 0 的 offset 8 读取 8 字节：

```text
QP to memory node 0
RDMA READ remote offset 8
local pointer slot
await_resume -> RemotePtr{*pointer_slot}
```

这依赖 shard 文件和 memory node 初始化约定：

- offset 0: free pointer
- offset 8: medoid pointer
- offset 16 起：节点

## 初始 beam 排序

添加 entry point 或 medoid 后：

```cpp
std::sort(beam.begin(), beam.end(), distance ascending)
```

之后 `select_best` 会查找最小距离且未 expanded 的 beam entry。

## 查询开始阶段调用链

```text
用户调用 ComputeService::search
  search_local
    search_local_result
      QueryRequest
      query_queue_.enqueue

query worker
  vamana_service_schedule_queries
    queue.try_dequeue
    vamana_idx.knn / knn_raw

Vamana::knn_raw
  准备 query buffer
  beam.clear
  visited.clear
  add entry points
  如果 beam empty:
    read_medoid_ptr
    add_start_point(medoid)
  sort beam
  cold start read first neighbor
```

## 第一次 RDMA、CPU、GPU

第一次 RDMA 可能是：

- 如果有 entry point：`read_vamana_node(entry)`。
- 如果无 entry point：先 `read_medoid_ptr`，再 `read_vamana_node(medoid)`。

第一次 CPU distance：

- `distance_to_stored_vector` 对 entry point 或 medoid 计算距离。

第一次 GPU transfer：

- 非 RaBitQ：query H2D 在 `knn_raw` 开始处。
- RaBitQ：rotated query 在 CPU 计算，exact query H2D 延迟到 exact candidate rerank。

第一次 neighbor RDMA：

- cold start 中对当前 best beam entry 调 `read_vamana_neighbors`。

## 性能影响

- medoid pointer 是所有无 anchor 查询的固定远端小读，可能成为 latency 下限的一部分。
- entry point 节点读取是完整 `size_until_vector_end`，比只读 id 或 neighbor list 大。
- 初始距离在 CPU 上算，少量 entry point 下合理，但如果 anchor hints 很多，CPU 初始距离可能可见。
- visited set 从一开始就去重 entry points，避免重复扩展。
- query H2D 提前做可以和后续 neighbor RDMA 有部分重叠，但当前代码在进入 expansion 前完成调用，不等待 stream。

## 设计异味

1. `add_start_point` 是嵌套 coroutine，内部手动 resume 子协程，阅读和调试复杂。
2. medoid 固定存放在 memory node 0 offset 8，是硬编码协议。
3. 初始 entry point distance 走 CPU，后续 candidate distance 走 GPU，路径不完全统一。
4. `direct_node_reads_` 后续结果 id 读取为 true，结果阶段还会产生小 RDMA 读。
5. local-stitch query entry points 由 storage-owner 配置控制，查询优化和插入更新策略耦合。

## 可验证问题

- 没有 anchor entry points 时，查询从哪个远端地址读 medoid？
- `QueryRequest` 的生命周期由谁释放？
- `knn` 和 `knn_raw` 的差别是什么？
- RaBitQ 模式下 query 是否一开始就上传到 GPU？
- entry points 是如何去重的？

## 学习任务

1. 手动画出从 `ComputeService::search` 到 `Vamana::knn_raw` 的调用链。
2. 在 `vamana_search.ipp` 中标出初始化阶段所有 `co_await`。
3. 在 `rdma/vamana_rdma_reads.hh` 中找到 `read_medoid_ptr`，解释它为什么用 pointer slot。
4. 构造一个查询路径表：有 anchor、无 anchor、RaBitQ、非 RaBitQ 四种组合的初始化差异。
5. 思考：如果要缓存 medoid pointer，正确性和更新时机会遇到什么问题？

