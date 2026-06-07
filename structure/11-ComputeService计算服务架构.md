# 第11课：ComputeService计算服务架构

## 学习目标
- 掌握`ComputeService`的整体架构与生命周期
- 理解WorkerPool、SharedContext、线程池管理
- 理解服务启动/停止/暂停/恢复的生命周期

## 内容大纲

### 1. ComputeService职责划分
```cpp
template <class Distance>
class ComputeService {
    // 对外API
    search(query, k) → vec<node_t>    // 搜索接口
    insert(batch) → size_t             // 插入接口
    load_index(path) → bool            // 加载索引
    store_index(path) → bool           // 存储索引
    status() → Status                  // 获取状态

    // 内部组件
    WorkerPool        // 计算线程池
    Vamana<Distance>  // Vamana图索引
    QueryRouter       // 多CN查询路由
    // RPC系统
    // 存储Owner插入系统
};
```

### 2. 初始化流程 (`lifecycle.ipp`)
```
构造:
1. 解析配置 (IndexConfiguration)
2. 初始化RDMA Context + ClientConnectionManager
3. 接收远程Memory Region Token (remote_access_tokens_)
4. 初始化Vamana (R, beam_width, beam_width_construction, alpha, k, dim)
5. 创建WorkerPool
6. 如果load_index: 发送load命令到所有Memory Node
7. 如果routing: 启动RPC子系统
8. 如果storage_owner: 启动Storage Owner插入运行时
9. 启动workers
10. 如果shutdown_remote_on_stop: 注册退出钩子
```

### 3. WorkerPool (`src/worker_pool.hh`)
```cpp
class WorkerPool {
    // 创建线程和共享上下文
    allocate_worker_threads(context, cm, remote_mrts, num_coroutines);

    // 共享上下文 (每组MAX_QPS个线程共享Context+QP)
    vec<u_ptr<SharedCtx>> shared_contexts_;

    // 线程级同步 (Latch栅栏)
    Latch start_latch_;  // 所有线程同时开始
    Latch end_latch_;    // 所有线程完成后同步

    // 调度入口
    process_vamana_inserts(vamana_idx, next_insert_idx, database, num_coroutines, thread_id);
    process_vamana_queries(vamana_idx, next_query_idx, queries, query_router, num_coroutines, thread_id);
};
```

### 4. SharedContext (`src/shared_context.hh`)
```cpp
template <typename T>
class SharedContext {
    Context context;                   // 独立的RDMA Context
    vec<u_ptr<DetachedQP>> qps;       // 每个Memory Node一个DetachedQP
    u_ptr<LocalMemoryRegion> memory_region;  // 注册的本地大页Buffer
    vec<T*> registered_threads;       // 共享此Context的线程列表

    void register_thread(T* thread) {
        registered_threads.push_back(thread);
        thread->ctx = this;           // 反向指针
        thread->ctx_tid = registered_threads.size() - 1;  // 分配ID
    }
};
```

### 5. 线程模型
```
ComputeService 的线程组成:
┌─────────────────────────────────────────────────┐
│  ComputeThread[0]  ──  SharedContext[0]         │
│  ComputeThread[1]  ──  SharedContext[0]         │
│  ComputeThread[2]  ──  SharedContext[1]         │
│  ComputeThread[3]  ──  SharedContext[1]         │
│  ...                                            │
│  ComputeThread[N-1] ── SharedContext[N%MAX_QPS] │
├─────────────────────────────────────────────────┤
│  rpc_thread_              (路由/多CN通信)       │
│  storage_insert_completion_thread_              │
│  storage_insert_sender threads (每storage一个)  │
└─────────────────────────────────────────────────┘
```
MAX_QPS=4: 最多4个共享Context，限制每节点QP总数

### 6. Worker角色分配
```cpp
ServiceProfile resolve_service_profile() {
    if (insert_workers > 0 && query_workers > 0) {
        // 显式分配: insert_workers + query_workers = num_threads
        return {insert_workers, query_workers, insert_coroutines, query_coroutines};
    }
    // 默认: 75% query, 25% insert
    u32 query_workers = max(1u, num_threads * 3 / 4);
    u32 insert_workers = num_threads - query_workers;
    return {insert_workers, query_workers, ...};
}
```
每个worker被标记为`ServiceWorkerRole::insert`或`ServiceWorkerRole::query`

## 课后任务
1. 跟踪`ComputeService`从构造到析构的完整生命周期
2. 分析：为什么MAX_QPS设为4？更大值会有什么影响？
3. 实验：修改75/25的worker分配比例，观察对混合负载的影响

## 参考文件
- `src/service/compute_service.hh`
- `src/service/compute_service.cc`
- `src/worker_pool.hh`
- `src/shared_context.hh`
