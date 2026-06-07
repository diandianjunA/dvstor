# 第14课：存储Owner插入协议详解

## 学习目标
- 理解storage_owner插入的完整协议流程
- 掌握Compute Node与Memory Node之间的RPC交互
- 理解批量、超时、重试等机制

## 内容大纲

### 1. 协议架构

```
Compute Node                          Memory Node (Storage Owner)
┌──────────────┐                     ┌──────────────────────┐
│ InsertQueue  │                     │ InsertRuntime        │
│     ↓        │  RDMA SEND (req)    │     ↓                │
│ SenderThread ├────────────────────→│ InsertWorker[0..N]   │
│  per storage │                     │     ↓                │
│     ↑        │  RDMA SEND (resp)   │  execute_insert_job  │
│ ComplThread  │←────────────────────┤     ↓                │
│     ↓        │                     │  Peer Reverse RPC    │
│  resolve     │                     │  (跨Storage Node)    │
│  promises   │                     └──────────────────────┘
└──────────────┘
```

### 2. Compute Node端协议 (`storage_owner_insert.ipp`)

#### 插入入队
```cpp
// insert() → 创建StorageInsertTask → 放入对应storage的队列
struct StorageInsertTask {
    InsertItem item;             // id + vector
    shared_ptr<Sample> sample;   // breakdown tracking
    promise<bool> result;        // 异步结果
    time_point enqueued_at;
    time_point sender_dequeued_at;
};
```

#### 发送端线程 (`run_storage_insert_sender`)
```cpp
void run_storage_insert_sender(u32 owner_storage) {
    // 1. 等待任务到达或超时（batch_wait_us）
    // 2. 收集最多batch_max个任务
    // 3. 序列化为RDMA SEND请求
    // 4. 投递发送
    // 5. 记录batch→slot映射
}
```

#### 完成端线程 (`run_storage_insert_completion_loop`)
```cpp
// 1. 轮询CQ获取响应
// 2. 根据batch_id找到对应的slot和tasks
// 3. 解析响应 → 设置promise结果
// 4. 回收slot
```

### 3. Memory Node端协议 (`memory_node.cc`)

#### 请求处理
```cpp
size_t handle_storage_insert_request(u32 client_id, const byte_t* payload, size_t bytes) {
    // 1. 反序列化: ids[], vectors[]（可能量化）
    // 2. 入队到storage_insert_tasks_
    // 3. 通知worker线程
}
```

#### Worker循环 (`storage_owner_insert_worker_loop`)
```cpp
void storage_owner_insert_worker_loop(u32 worker_id) {
    // 1. 从队列取任务
    // 2. 执行execute_storage_owner_batch_items
    //    a. Beam Search（本地内存访问）
    //    b. CPU RobustPrune
    //    c. 分配+写入新节点
    //    d. 收集local_updates和remote_updates
    // 3. 本地更新: apply_local_reverse_update
    // 4. 远程更新: enqueue_reverse_update_batch (→ Peer RPC)
    // 5. 构造响应并RDMA SEND回CN
}
```

### 4. RPC深度控制
```cpp
struct StorageOwnerRpcSlot {
    u32 owner_storage, slot_id;
    bool in_use, send_done, response_done, results_completed;
    vec<byte_t> request_buffer, response_buffer;
    unique_ptr<LocalMemoryRegion> request_region, response_region;
    vec<unique_ptr<StorageInsertTask>> tasks;
};

// 空闲slot管理: free_slots队列
// 发送前: 从freelist取slot → 标记in_use
// 完成后: 清理 → 归还freelist
```
深度限制：最多`storage_owner_rpc_depth`个并发批次

### 5. 批量与超时
```
批量化参数:
- storage_owner_batch_max (默认16):  每批最多插入数
- storage_owner_batch_wait_us (默认250μs): 最大等待时间
- storage_owner_rpc_timeout_ms (默认30s): 单个RPC超时

策略: 先到先收集，达到batch_max或超时即发送
```

### 6. 错误处理
- 超时: `storage_insert_timeout_logs_` 计数器
- 连接断开: slot清理 + promise.set_exception
- 批量失败: `fail_storage_owner_tasks()`

## 课后任务
1. 画出一次storage_owner插入的完整时序图
2. 分析：batch_wait_us=250μs对延迟和吞吐量的影响
3. 思考：如果Memory Node崩溃，Compute Node如何恢复？

## 参考文件
- `src/service/compute_service/storage_owner_insert.ipp`
- `src/service/storage_owner_protocol.hh`
- `src/service/storage_owner_client_helpers.hh`
- `src/memory_node/storage_owner_runtime.cc`
