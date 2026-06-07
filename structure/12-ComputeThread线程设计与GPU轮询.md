# 第12课：ComputeThread线程设计与GPU轮询

## 学习目标
- 掌握`ComputeThread`的完整成员变量与方法
- 理解RDMA CQ轮询与GPU事件轮询的双路径完成检测
- 理解`poll_gpu_events()`的实现

## 内容大纲

### 1. ComputeThread完整视图 (`src/compute_thread.hh`)
```cpp
class ComputeThread : public Thread {
    // === 身份标识 ===
    const u32 node_id;           // 所属Compute Node ID
    u32 ctx_tid;                // 在SharedContext中的线程索引

    // === RDMA相关 ===
    vec<ibv_wc> send_wcs;       // 发送完成队列条目缓冲
    vec<std::atomic<i32>> post_balances;       // 每协程RDMA未完成计数
    vec<std::atomic<i32>> gpu_post_balances;   // 每协程GPU未完成计数
    vec<u64*> pointer_slots_;                 // 每协程的指针槽（RDMA CAS/FAA结果）

    // === 协程管理 ===
    vec<u_ptr<VamanaCoroutine>> vamana_coroutines;
    u32 running_coroutine_;     // 当前活动协程ID

    // === GPU资源 ===
    gpu::GpuBufferManager gpu_buffers;  // 每线程的GPU资源管理器
    void* reserved_query_state[2];      // 预分配查询状态

    // === 内存管理 ===
    BufferAllocator& buffer_allocator;

    // === 统计与结果 ===
    statistics::ThreadStatistics stats;
    hashmap_t<node_t, service::QueryResult> query_results;

    // === 服务角色 ===
    ServiceWorkerRole service_role_;  // none/insert/query
};
```

### 2. WR ID编码
```cpp
u64 create_wr_id() const {
    return encode_64bit(ctx_tid, running_coroutine_);
}
// [32位 ctx_tid | 32位 coroutine_id]
```
使CQ完成回调能将事件路由到正确的线程和协程

### 3. RDMA完成轮询
```cpp
void poll_cq() {
    Context::poll_send_cq(send_wcs.data(), max_send_queue_wr_, ctx->get_cq(),
        [&](u64 wr_id) {
            auto [ctx_offset, coroutine_id] = decode_64bit(wr_id);
            --ctx->registered_threads[ctx_offset]->post_balances[coroutine_id];
        });
}
```
每次轮询处理最多`max_send_queue_wr_`个完成事件

### 4. GPU完成轮询 (`compute_thread_gpu.cc`)
```cpp
void ComputeThread::poll_gpu_events() {
    for (u32 coro_id = 0; coro_id < vamana_coroutines.size(); ++coro_id) {
        if (gpu_post_balances[coro_id] == 0) continue;
        auto& gs = gpu_buffers.state(coro_id);
        cudaError_t err = cudaEventQuery(gs.event);
        if (err == cudaSuccess) {
            --gpu_post_balances[coro_id];  // GPU工作完成
        } else if (err != cudaErrorNotReady) {
            // 真正的错误
        }
    }
}
```
使用`cudaEventQuery`非阻塞检查GPU kernel完成状态

### 5. 就绪检查
```cpp
bool is_ready(u32 coroutine_id) const {
    return post_balances[coroutine_id] == 0      // 所有RDMA完成
        && gpu_post_balances[coroutine_id] == 0;  // 所有GPU完成
}
```
调度器每次循环对所有协程调用此检查

### 6. 跟踪操作
```cpp
void track_post()     { ++post_balances[running_coroutine_]; }
void track_gpu_post() { ++gpu_post_balances[running_coroutine_]; }
```
- `track_post()`: 在每次`qp->post_send()`前调用
- `track_gpu_post()`: 在每次GPU kernel launch + event record后调用

### 7. 随机Memory Node选择
```cpp
u32 get_random_memory_node() { return dist_(generator_); }
// uniform_int_distribution over [0, num_memory_nodes)
```
用于节点分配时的负载均衡

## 课后任务
1. 画一张完整的状态图：协程从挂起到恢复的完整过程
2. 分析：如果`poll_gpu_events()`改为阻塞式`cudaEventSynchronize`会怎样？
3. 写一段伪代码展示协程的生命周期

## 参考文件
- `src/compute_thread.hh`
- `src/gpu/compute_thread_gpu.cc`
- `src/gpu/gpu_buffer_manager.hh`
