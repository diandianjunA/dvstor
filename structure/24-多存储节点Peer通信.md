# 第24课：多存储节点Peer通信

## 学习目标
- 理解跨存储节点的RDMA和RPC通信架构
- 掌握Peer RDMA信用系统
- 理解反向更新RPC的完整流程

## 内容大纲

### 1. Peer通信架构
```
Memory Node A                    Memory Node B
┌──────────────────┐            ┌──────────────────┐
│ Index Buffer     │            │ Index Buffer     │
│ PeerScratchBuffer│            │ PeerScratchBuffer│
│                  │            │                  │
│ peer_qps_[B]  ───┼─RDMA READ─→│ index_region_   │
│ peer_qps_[B]  ───┼─RDMA WRITE→│ index_region_   │
│ peer_qps_[B]  ───┼─RDMA CAS──→│ index_region_   │
│                  │            │                  │
│ peer_rpc (SEND)──┼───RPC─────→│ peer_rpc handler│
│                  │←──RPC──────│                  │
└──────────────────┘            └──────────────────┘
```

### 2. Peer QP管理
```cpp
void setup_storage_peers(Configuration& config) {
    // 1. 解析所有peer endpoint
    // 2. 为每个远端创建Context + QP
    // 3. 交换Memory Region Token
    // 4. 分配peer_scratch_buffer_ (RDMA暂存)
    // 5. 交换index_region_的rkey
}

QP& peer_control_qp(u32 shard_id);      // 控制/小数据
QP& peer_data_qp(u32 shard_id, qp_idx); // 数据/大传输
```

### 3. Peer RDMA信用系统
```cpp
// 全局信用
std::atomic<u32> peer_async_rdma_outstanding_;

// 每QP信用
vec<vec<std::atomic<u32>>> peer_rdma_read_qp_outstanding_;

bool try_acquire_peer_rdma_read_credit(shard_id, qp_idx) {
    // 检查全局和局部信用
    return try_acquire_counter(peer_async_rdma_outstanding_, global_limit) &&
           try_acquire_counter(peer_rdma_qp_outstanding[shard][qp], per_qp_limit);
}
```
目的：防止过度发送RDMA请求导致远端QP溢出

### 4. Peer同步/异步操作
```cpp
// 同步: 发送后阻塞等待CQ
u64 next_peer_sync_wr_id();
void wait_peer_sync_completion(wr_id);
// 用于锁定/解锁等关键路径操作

// 异步: 发送后继续，稍后轮询
u64 next_peer_async_wr_id();
void poll_peer_send_cq();
// 用于向量读取等可并发的操作
```

### 5. 反向更新RPC协议
```
请求消息:
  PeerRpcHeader: [request_id(8B)|item_count(4B)|flags(4B)]
  ReverseUpdateOp[]: [{target_ptr(8B)|candidate_count(4B)|pad(4B)|candidate_ptrs[]}]

响应消息:
  PeerRpcHeader: [request_id(8B)|item_count(4B)|success(1B)|pad(3B)]
```

#### 发送端（发起反向更新的存储节点）
```cpp
bool enqueue_reverse_update_batch(target_shard, ops, config) {
    // 1. 按target_shard分组
    // 2. 按coalesce_max合并
    // 3. 入队到peer_reverse_outgoing_
}

void peer_reverse_outgoing_loop() {
    // 1. 从队列取任务
    // 2. 序列化为RPC消息
    // 3. RDMA SEND到目标
    // 4. 如果sync模式: 等待响应
}
```

#### 接收端（处理反向更新请求）
```cpp
void peer_reverse_update_worker_loop(u32 worker_id) {
    // 1. 从peer_reverse_tasks_取任务
    // 2. 对每个ReverseUpdateOp:
    //    a. peer_rdma_read 读取target节点
    //    b. CPU/GPU RobustPrune
    //    c. peer_rdma_write 写入更新后的邻居
    // 3. 构造响应 → 入队peer_reverse_responses_
}
```

### 6. 队列管理
```
peer_reverse_tasks_:       接收到的反向更新请求队列
peer_reverse_responses_:   待发送的响应队列
peer_reverse_outgoing_:    待发送的请求队列

每个队列有:
- mutex + condition_variable (生产者-消费者)
- 深度限制 (storage_owner_reverse_queue_depth)
```

## 课后任务
1. 画一张Peer RPC的完整时序图（含RDMA和RPC两种路径）
2. 分析：如果反向更新队列满了会发生什么？
3. 模拟一个需要跨两个存储节点的插入场景

## 参考文件
- `src/memory_node/memory_node.hh`（Peer相关成员）
- `src/memory_node/peer_rdma.cc`
- `src/memory_node/peer_rpc.cc`
