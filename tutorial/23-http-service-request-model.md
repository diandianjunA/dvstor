# 第 23 课：HTTP 服务层与外部请求模型

## 本课目标

本课学习“服务层类型”和“scheduler 如何把外部请求映射到 Vamana coroutine”。注意：代码目录叫 `src/http/`，但从当前源码看，这一层主要定义 service request types 和 scheduler，并没有在这些文件中实现完整 HTTP server。学习时不要因为目录名假设存在 REST/HTTP 网络服务。

代码入口：

- `src/http/service_types.hh`
- `src/http/vamana_service_scheduler.hh`
- `src/service/compute_service.hh`
- `src/service/compute_service/search.ipp`
- `src/service/compute_service/storage_owner_insert.ipp`

## 1. service_types 是内部请求模型

`src/http/service_types.hh` 定义了三个核心类型：

```cpp
struct QueryResultItem {
  node_t id;
  distance_t distance;
};

using QueryResult = vec<QueryResultItem>;
```

查询结果保留了 id 和 distance。但 `ComputeService::search()` public API 最后只返回 `vec<node_t>`，会丢弃 distance。distance 主要用于内部 breakdown、local result 或未来扩展。

插入请求：

```cpp
struct InsertRequest {
  node_t id;
  vec<element_t> components;
  MutationKind kind;
  RemotePtr old_ptr;
  RemotePtr new_ptr;
  std::promise<bool> result;
  time_point enqueued_at;
  shared_ptr<breakdown::Sample> breakdown_sample;
};
```

查询请求：

```cpp
struct QueryRequest {
  vec<element_t> components;
  vec<byte_t> raw_components;
  vec<RemotePtr> entry_points;
  VectorDType query_dtype;
  u32 k;
  std::promise<QueryResult> result;
  time_point enqueued_at;
  shared_ptr<breakdown::Sample> breakdown_sample;
};
```

队列类型：

```cpp
using InsertQueue = concurrent_queue<InsertRequest*>;
using QueryQueue = concurrent_queue<QueryRequest*>;
```

这些类型是外部 API 和核心 Vamana coroutine 之间的桥：

- API 调用方负责构造 request。
- scheduler 从 queue 取 request。
- Vamana coroutine 执行。
- scheduler 通过 promise 返回结果。

## 2. QueryRequest 的两种 query 表示

`QueryRequest` 同时支持：

- `components`：float vector。
- `raw_components`：按 `query_dtype` 存储的原始 byte vector。

Scheduler 中的逻辑是：

```cpp
if (!req->raw_components.empty()) {
  vamana_idx.knn_raw(...);
} else {
  copy components to staging;
  vamana_idx.knn(...);
}
```

这说明 raw query 是优先路径。如果 `raw_components` 非空，`components` 不参与实际计算。

这对优化有意义：

1. 对 float32 query，直接用 `components` 可以避免 raw decode。
2. 对 float16/int8 query，raw path 可以保留输入 dtype，减少 host 端转换。
3. 但 routing enabled 时，`search_raw()` 会先 decode 成 float 并调用 `search()`，从而失去 raw path。

## 3. entry_points 的作用

`QueryRequest` 中的 `entry_points` 是 optional 起点集合。它主要由 `ComputeService::storage_owner_query_entry_points()` 填充。

在 local-stitch 模式下：

1. `anchor_index_->nearest_shards(query, 2)` 找最近 shard。
2. 每个 shard 取 `storage_owner_anchor_hints` 个 anchor。
3. 去重后作为 entry points。

Scheduler 会把 `entry_points` 指针传给：

- `vamana_idx.knn(...)`
- `vamana_idx.knn_raw(...)`

如果为空，则传 `nullptr`，Vamana search 会按默认 medoid 路径启动。

这说明 service layer 不理解 Vamana search 细节，但它能通过 `entry_points` 改变搜索起点。

## 4. InsertRequest 的 kind、old_ptr、new_ptr

`InsertRequest` 支持 mutation kind：

- insert
- upsert
- erase

字段含义：

- `old_ptr`：用于 upsert/delete 的旧节点指针。
- `new_ptr`：insert/upsert 后新节点位置。
- `kind`：区分插入、更新、删除。

在普通 compute-side insert 模式下，scheduler 调用 `vamana_idx.insert(...)`，并把 `&req->new_ptr` 传进去。

在 storage-owner insert 模式下，`ComputeService` 不使用 insert worker，而是将外部 insert/upsert/delete 封装为 `StorageInsertTask`，走 owner RPC。此时 `InsertRequest` scheduler 不是主路径。

## 5. scheduler 的整体结构

`src/http/vamana_service_scheduler.hh` 定义两个模板函数：

- `vamana_service_schedule_inserts`
- `vamana_service_schedule_queries`

它们都是长期运行的 worker loop。

共同特点：

1. `thread->reset()`。
2. 设置 service role：
   - insert worker：`ServiceWorkerRole::insert`
   - query worker：`ServiceWorkerRole::query`
3. 创建 staging database。
4. 为每个 coroutine slot 创建 dummy coroutine。
5. 进入无限循环。
6. 每轮遍历所有 coroutine slot。
7. 每个 slot 都 poll RDMA CQ 和 GPU events。
8. 如果 coroutine done：
   - 完成当前 request。
   - 尝试从 queue 取新 request。
   - destroy 旧 coroutine handle。
   - reset coroutine state。
   - 创建新的 Vamana coroutine。
9. 如果 coroutine 未 done 且 `thread->is_ready(cid)`：
   - resume。
10. 如果所有 slot idle：
    - 检查 shutdown。
    - 检查 paused。
    - 否则 yield。

这套 scheduler 是核心服务执行模型。它不是线程池任务提交模型，而是“每个 worker 固定扫描自己的 coroutine slots”。

## 6. insert scheduler 细节

insert scheduler 初始化：

```cpp
io::Database<element_t> staging;
staging.allocate(dim, num_coroutines);
vec<InsertRequest*> active_requests(num_coroutines, nullptr);
```

取到 request 后：

1. 将 `req->components` 复制到 staging 的 coroutine slot。
2. `staging.set_id(cid, req->id)`。
3. `active_requests[cid] = req`。
4. 如果有 breakdown sample：
   - 记录 queue wait。
   - `mark_started(...)`。
5. destroy dummy/old coroutine。
6. reset coroutine state。
7. 调用：

```cpp
vamana_idx.insert(req->id, slot_components, thread, &req->new_ptr)
```

coroutine 完成时：

1. 如果有 sample，`mark_finished(...)`。
2. `thread->set_active_sample(cid, nullptr)`。
3. `active_requests[cid]->result.set_value(true)`。
4. 清空 active request。

当前 scheduler 对 insert coroutine 的异常没有显式捕获。Vamana insert 内部如果抛异常或断言失败，通常会终止流程，而不是给 promise 返回 false。

## 7. query scheduler 细节

query scheduler 也创建 staging：

```cpp
io::Database<element_t> staging;
staging.allocate(dim, num_coroutines);
```

还创建：

```cpp
vec<node_t> slot_ids(num_coroutines);
slot_ids[i] = i;
```

这里 `slot_ids[cid]` 作为 query id 传给 Vamana。Vamana search 结束后结果写到：

```cpp
thread->query_results[q_id]
```

coroutine 完成时：

1. 从 `thread->query_results` 查 `slot_ids[cid]`。
2. 找到则 move 到 promise。
3. 找不到则返回空 `QueryResult`。
4. 清理 active request。

取到新 request 后：

- 如果 raw query 非空：
  - `vamana_idx.knn_raw(slot_ids[cid], req->raw_components.data(), req->query_dtype, thread, entry_points)`
- 否则：
  - copy float components 到 staging。
  - `vamana_idx.knn(slot_ids[cid], slot_components, thread, entry_points)`

这说明 query result 的关联不是通过 request pointer，而是通过 worker thread 内部的 `query_results` map 和 slot id。重构时必须保持这个映射不乱。

## 8. breakdown 在 service layer 的接入点

service layer 记录两个时刻：

- enqueue time：由 `ComputeService::search_local_result()` 或 insert API 创建 request 时填入。
- dequeue/start time：scheduler 从 queue 取到 request 时记录。

insert/query scheduler 都会：

```cpp
queue_wait = now - req->enqueued_at;
thread->stats.xxx_queue_wait_ns += queue_wait;
req->breakdown_sample->mark_started(now, now, thread->stats);
```

coroutine done 时：

```cpp
req->breakdown_sample->mark_finished(now, thread->stats);
```

因此 breakdown 的 service time 覆盖的是：

- coroutine 执行。
- RDMA/GPU 等待。
- scheduler 轮询 resume。

但不覆盖：

- public API 调用方构造 request 前的时间。
- RPC routing 中等待 outbound/receive 的全部细节。
- future.get() 前后用户态业务逻辑。

## 9. pause/shutdown 与 scheduler 的关系

Scheduler 只在 `all_idle` 时处理 pause：

- 如果某个 coroutine 未完成，`all_idle = false`。
- 如果 queue 中有新请求，`all_idle = false`。
- 如果 coroutine ready 并 resume，`all_idle = false`。

当所有 slots 都 idle：

1. 如果 `shutdown` 为 true，break。
2. 如果 `paused` 为 true：
   - `idle_count++`
   - while paused yield
   - `idle_count--`
3. 否则 yield。

所以 scheduler 的 pause 语义是“等所有当前活跃请求完成后暂停接新请求”。它不是立即停止。

优化 load/store 或在线切 index 时，必须考虑这个语义：

- pause 前已入队但尚未取出的请求可能继续等待。
- pause 前已取出的请求会跑完。
- 如果 Vamana coroutine 卡住，pause 也会卡住。

## 10. API 层与核心索引层的耦合点

虽然 `service_types.hh` 看起来很薄，但它已经和核心实现耦合：

1. `QueryRequest` 包含 `RemotePtr entry_points`：
   - API 层知道 Vamana 的远端节点起点。

2. `InsertRequest` 包含 `RemotePtr old_ptr/new_ptr`：
   - API 层知道节点指针和 mutation 布局。

3. `VectorDType` 下沉到 QueryRequest：
   - API 层知道 storage vector dtype。

4. `breakdown::Sample` 下沉到 request：
   - API 层知道性能采样模型。

5. `MutationKind` 来自 storage-owner protocol：
   - API 层与 storage-owner update 模式耦合。

这些耦合不是一定错误，但重构时应该明确边界。更清晰的分层可能是：

- public API request：只包含用户输入。
- execution request：包含 RemotePtr、breakdown、layout、dtype、entry points。
- protocol request：用于 RPC/storage-owner 的编码。

## 11. 性能影响

service scheduler 的性能特征：

1. 每个 worker 扫描所有 coroutine slots：
   - `num_coroutines` 越大，空转扫描成本越高。
   - 但 coroutine 数太小，RDMA/GPU overlap 不够。

2. 每轮 slot 都 poll：
   - `thread->poll_cq()`
   - `thread->poll_gpu_events()`
   - 如果 poll 操作成本不低，扫描开销会明显。

3. queue 使用 concurrent_queue：
   - 多 producer、多 consumer 场景下有同步成本。
   - insert/query 分队列，避免读写请求互相抢同一 queue。

4. query staging copy：
   - float path 会把 query copy 到 staging。
   - raw path 避免 float staging copy，但 Vamana 内部可能需要 decode 或 H2D。

5. future/promise 每请求分配：
   - public API 同步等待 future。
   - 高频小请求下，promise/future 开销可能可见。

6. request 使用 `new` 和 `delete`：
   - `ComputeService::search_local_result()` 为每个 query new 一个 `QueryRequest`。
   - insert API 同理需要查看实现路径。
   - 可考虑 request pool，但要先测。

## 12. 设计异味

1. `src/http` 命名不准确：
   - 当前文件更像 service runtime，不是 HTTP adapter。

2. request 同时包含 API 字段和执行字段：
   - 例如 `entry_points`、`RemotePtr`、`breakdown_sample`。

3. scheduler 是 header-only 模板：
   - 编译成本高。
   - 单测隔离困难。

4. coroutine lifecycle 手工管理：
   - 显式 `handle.destroy()`。
   - dummy coroutine 初始化。
   - reset 状态函数需要知道 coroutine 内部字段。

5. request completion 假设成功：
   - insert done 后直接 `set_value(true)`。
   - 异常/失败路径不是显式 API。

6. query result 通过 thread map 和 slot id 传递：
   - 隐式约定较强。

## 13. 可验证问题

1. query raw path：
   - `raw_components` 非空时，是否不会读取 `components`。

2. query float path：
   - components 是否正确 copy 到 staging slot。

3. query result map：
   - Vamana search 是否总是写入 `thread->query_results[slot_id]`。
   - 找不到 result 时返回空是否会掩盖错误。

4. insert completion：
   - Vamana insert 内部失败时 promise 会发生什么。

5. pause 行为：
   - 活跃请求完成前 pause 是否阻塞。
   - pause 后是否不再取新请求。

6. breakdown queue wait：
   - `query_queue_wait_ns` 是否只统计 queue 等待，不包含 service time。

7. coroutine 数量：
   - 提高 `num_coroutines` 后 latency/throughput 是否单调变化。

## 14. 重构候选方案

低风险重构：

1. 将 `service_types.hh` 从 `src/http` 移到 `src/service`，或新增 wrapper 保持兼容。
2. 把 scheduler 中的 request completion 逻辑提取为小函数。
3. 把 query raw/float dispatch 提取为 `start_query_coroutine(...)`。
4. 为 request 创建 RAII wrapper，避免手动 delete。
5. 为 `thread->query_results` result missing 增加统计或错误日志。

中风险重构：

1. 引入 request pool，降低分配开销。
2. 将 `breakdown_sample` 从 request 中拆成 execution context。
3. 将 `entry_points` 的生成从 `ComputeService` 下沉到 query planner。

高风险重构：

1. 替换 scheduler 的扫描模型。
2. 把 RDMA/GPU progress 从 worker 中拆到独立 progress thread。
3. 改 public API 为异步 API。

## 15. 学习任务

1. 画一张 `QueryRequest` 从 public API 到 Vamana coroutine 的生命周期图。
2. 画一张 insert scheduler 的 slot 状态机：idle、active、ready、done、paused。
3. 找出 Vamana search 将结果写入 `thread->query_results` 的位置，解释为什么 slot id 能关联结果。
4. 设计一个 benchmark：固定 worker 数，调整 coroutine 数，观察 queue wait、service time、RDMA bytes、GPU kernel busy ratio。
5. 设计一个重构小步：把 scheduler 中 raw/float query dispatch 提取为函数，并列出需要保持的行为不变量。

