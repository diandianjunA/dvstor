# 第 04 课：线程、协程与工作队列骨架

## 本课目标

本课解释在线服务如何把请求变成协程，如何在同一个 compute thread 内同时等待 RDMA 和 GPU，为什么项目没有用传统 blocking future 去执行每个查询。理解这一层后，后续读 `Vamana::knn_raw` 和 `Vamana::insert` 的 `co_await` 才不会迷失。

## 代码证据

必须阅读：

- `src/coroutine.hh`
- `src/compute_thread.hh`
- `src/worker_pool.hh`
- `src/http/vamana_service_scheduler.hh`
- `src/service/compute_service/lifecycle.ipp`
- `src/vamana/scheduler.hh`

## 三类协程

`src/coroutine.hh` 定义三类 coroutine wrapper：

| 类型 | 用途 | initial suspend | final suspend |
| --- | --- | --- | --- |
| `MinorCoroutine` | 子协程，如读节点辅助逻辑 | `suspend_never` | `suspend_always` |
| `VamanaCoroutine` | 查询和普通在线插入主协程 | `suspend_always` | `suspend_always` |
| `StorageOwnerInsertCoroutine` | memory node storage-owner 异步插入 | `suspend_always` | `suspend_always` |

`VamanaCoroutine` 内部保存查询/插入热状态：

- `beam`
- `visited_nodes`
- `scratch_unvisited`
- `reserved_ptrs_a`
- `reserved_ptrs_b`
- `indirect_candidate_ptrs`
- `scratch_distances`
- `scratch_entry_ptrs`
- `scratch_indices_a`
- `scratch_indices_b`
- `scratch_flags`

这说明 coroutine frame 不只是控制流，也是查询热路径的临时内存容器。

## ComputeThread 的职责

`ComputeThread` 继承自 `Thread`，它不是简单线程句柄，而是每个 worker 的执行上下文。关键字段：

- `node_id`
- `send_wcs`
- `buffer_allocator`
- `ctx`: `SharedContext<ComputeThread>*`
- `ctx_tid`
- `query_results`
- `vamana_coroutines`
- `post_balances`
- `gpu_post_balances`
- `gpu_buffers`
- `reserved_query_state`
- `stats`
- `active_samples_`
- `service_role_`

最关键的判断是：

```cpp
bool is_ready(u32 coroutine_id) const {
  return post_balances[coroutine_id] == 0 && gpu_post_balances[coroutine_id] == 0;
}
```

这表示协程能否继续运行取决于 RDMA outstanding 和 GPU outstanding 都为 0。

## RDMA 等待模型

当 Vamana 代码发起 RDMA 操作时，会调用：

```cpp
thread->track_post();
```

这会增加当前 coroutine 的 `post_balances[running_coroutine_]`。

RDMA completion 到达时：

- 普通 WR completion 由 `SharedContext::complete_send` 解码 `wr_id`。
- 对应 coroutine 的 `post_balances` 减 1。

调度器每轮调用：

```cpp
thread->poll_cq();
```

所以 RDMA 等待不是阻塞当前 OS thread，而是让当前 coroutine 暂停，线程继续轮询其他 coroutine。

## GPU 等待模型

GPU kernel 发出后，代码调用：

```cpp
co_await gpu::GpuAwaitable{thread.get()};
```

`GpuAwaitable::await_suspend` 只做一件事：

```cpp
thread->track_gpu_post();
```

真正完成判断在：

```cpp
ComputeThread::poll_gpu_events()
  cudaEventQuery(gpu_buffers.event(coro_id))
  如果成功，gpu_post_balances[coro_id] = 0
```

因此 GPU 也是非阻塞式等待。协程暂停，线程继续 poll RDMA/GPU 并尝试推进其他 coroutine。

## 服务模式 scheduler

`src/http/vamana_service_scheduler.hh` 有两个服务模式循环：

- `vamana_service_schedule_inserts`
- `vamana_service_schedule_queries`

它们的共同模式是：

```text
初始化 staging database
初始化 num_coroutines 个 dummy coroutine
循环：
  对每个 coroutine id：
    poll_cq
    poll_gpu_events
    如果 coroutine done：
      完成旧 request promise
      从 queue try_dequeue 新 request
      构造新的 vamana_idx.insert 或 vamana_idx.knn coroutine
    否则如果 thread->is_ready(cid)：
      resume coroutine
    否则继续轮询
  如果全部 idle：
    如果 shutdown，退出
    如果 paused，进入 pause idle 协议
    yield
```

这就是 compute service 处理在线请求的核心循环。

## 离线 scheduler

`src/vamana/scheduler.hh` 是另一套批量调度器，服务离线式插入/查询数据库对象。它使用 `io::Database` 和 `query_router`，不直接处理 `InsertRequest` 或 `QueryRequest`。

这说明项目中至少有两套 scheduler：

- service scheduler：在线服务队列。
- database scheduler：批量数据处理和旧模式查询。

## 请求进入队列后的流程

普通查询：

```text
ComputeService::search_local_result
  new QueryRequest
  query_queue_.enqueue(request)
  future.get()

query worker
  vamana_service_schedule_queries
  queue.try_dequeue(req)
  set active sample
  构造 vamana_idx.knn 或 knn_raw coroutine
  循环 poll RDMA/GPU
  coroutine done 后从 thread->query_results 取结果
  req->result.set_value
```

普通插入：

```text
ComputeService::insert
  new InsertRequest
  insert_queue_.enqueue(request)
  future.get()

insert worker
  vamana_service_schedule_inserts
  queue.try_dequeue(req)
  拷贝 components 到 staging slot
  构造 vamana_idx.insert coroutine
  coroutine done 后 req->result.set_value(true)
```

## pause/resume 机制

`ComputeService::pause_workers` 设置 `workers_paused_ = true`，然后等待：

```cpp
workers_idle_count_ < config_.num_threads
```

service scheduler 只有在 `all_idle` 时才递增 idle count 并停在 pause loop。这个设计意味着 pause 不是抢占式的，正在执行的 coroutine 会继续跑到当前 worker 变 idle。

这对 `load_index` 和 `store_index` 很重要：它们会 pause workers，避免索引加载/存储期间仍有查询或插入修改远端数据。

## 性能影响

关键性能点：

- coroutine 数量决定单线程内可重叠 RDMA/GPU 的并发度。
- 每轮 scheduler 都对每个 coroutine 调 `poll_cq` 和 `poll_gpu_events`，轮询成本随 coroutine 数增长。
- `std::this_thread::yield()` 出现在 idle、credit wait、completion token wait 等路径，过多 yield 可能影响低延迟。
- `VamanaCoroutine` 使用 `std::unordered_set` 做 visited，会在热路径频繁查找。
- query/insert worker 拆分影响资源隔离。`resolve_service_profile` 决定有多少线程处理查询和插入。

## 设计异味

1. scheduler 是手写轮询循环，公平性和抢占性都靠代码约定。
2. `post_balances` 和 `gpu_post_balances` 使用普通计数，缺少更强类型的 pending token。
3. pause 只能等待 worker idle，不能立即阻止长查询继续推进。
4. `QueryRequest` 和 `InsertRequest` 用裸指针进入队列，生命周期靠调用者和 scheduler 约定。
5. coroutine state 和算法 scratch 混在一个结构中，复用方便但不利于分层测试。

## 可验证问题

- `co_await rdma read` 后，哪个计数让 coroutine 暂停？
- GPU kernel 完成如何通知 scheduler？
- query worker 怎么把结果交还给 `ComputeService::search_local_result`？
- pause workers 是否能中断正在跑的查询？
- 为什么 `VamanaCoroutine` 里保存 `visited_nodes`？

## 学习任务

1. 跟踪一次 `QueryRequest` 从创建到 `promise.set_value` 的完整路径。
2. 跟踪一次 GPU kernel launch 后 `gpu_post_balances` 从加一到清零的路径。
3. 画出一个 compute thread 内 4 个 coroutine 的轮询状态图。
4. 找出所有调用 `track_post` 和 `track_gpu_post` 的地方。
5. 思考：如果要降低轮询开销，可以替换成什么事件模型？需要改哪些边界？

