# 第13课：MemoryNode存储节点架构

## 学习目标
- 掌握`MemoryNode`的整体架构与索引存储设计
- 理解存储节点端的插入执行（storage_owner模式）
- 理解Peer-to-Peer RDMA与反向更新RPC

## 内容大纲

### 1. MemoryNode核心职责
```cpp
class MemoryNode {
    // === 索引存储 ===
    HugePage<byte_t> index_buffer_;    // 大页索引缓冲区
    MemoryRegion index_region_;        // RDMA注册的索引区域

    // === 节点级操作（通过RDMA被Compute Node访问） ===
    allocate_local_node()     // FAA → 返回RemotePtr
    read_global_medoid()      // RDMA读medoid_ptr
    write_global_medoid()     // CAS写medoid_ptr
    read_node_snapshot()      // 读节点快照
    lock_node/unlock_node()   // CAS锁/Write解锁

    // === 存储端插入执行（storage_owner模式） ===
    beam_search_candidates()  // 本地Beam Search找候选
    robust_prune_cpu()        // CPU端RobustPrune
    execute_storage_owner_insert_job_async()  // 异步插入Job

    // === Peer-to-Peer通信 ===
    peer_rdma_read/write/CAS  // 跨存储节点RDMA
    peer_reverse_update RPC   // 反向边更新RPC
};
```

### 2. 内存布局
```
Memory Node Index Buffer:
Offset 0:   free_ptr (8B)      ← FAA分配节点用
Offset 8:   medoid_ptr (8B)    ← CAS交换Medoid用
Offset 16+: VamanaNode[0], VamanaNode[1], ...
```

### 3. 存储端插入模式 (storage_owner)
与Compute Node端插入的关键区别：
- **Beam Search在存储节点本地执行**：直接访问本地内存，无需RDMA
- **RobustPrune用CPU执行**：`robust_prune_cpu()`
- **反向边更新通过Peer RPC**：需要跨存储节点通信

### 4. Peer-to-Peer RDMA架构
```cpp
// 每个存储节点建立到其他存储节点的连接
void setup_storage_peers(Configuration& config) {
    peer_config_ = make peer config;
    peer_context_ = new Context;
    // 为每个远端存储节点创建 QP
    peer_qps_[shard_id] = connect to peer;
    // 交换Memory Region Token
    // 分配Peer Scratch Buffer用于RDMA暂存
}

// 跨节点读
void remote_read_bytes(shard_id, remote_offset, dst, bytes, scratch_offset) {
    // 使用peer_scratch_buffer_暂存
    // RDMA READ from remote shard's index_region
}

// 跨节点写
void remote_write_bytes(shard_id, remote_offset, src, bytes, scratch_offset) {
    // 先写到本地scratch → RDMA WRITE到远程
}
```

### 5. 反向更新RPC协议
```
当存储节点A上的节点p引用了存储节点B上的节点q时:
Storage Node A → Storage Node B:
    PeerRpcHeader{request_id, item_count, flags}
    ReverseUpdateOp[]{target_ptr, candidate_ptrs[]}
Storage Node B → Storage Node A:
    PeerRpcHeader{request_id, item_count, success}
```
支持三种完成模式：
- **async**: 发送后不等待响应（后台队列处理）
- **sync**: 等待远端确认

### 6. 并发流程控制
```cpp
// Peer RDMA读取信用系统
u32 peer_rdma_read_credit_limit_per_qp();
bool try_acquire_peer_rdma_read_credit(shard_id, qp_idx);
void acquire_peer_rdma_read_credit(shard_id, qp_idx);  // 阻塞等待
```
防止过多in-flight RDMA读取耗尽Remote QP资源

## 课后任务
1. 对比compute端插入和storage_owner端插入的延迟差异
2. 画图展示一次storage_owner插入中的跨节点通信
3. 分析：peer_rdma_read_credit_limit对吞吐量的影响

## 参考文件
- `src/memory_node/memory_node.hh`
- `src/memory_node/memory_node.cc`
- `src/memory_node/storage_owner_runtime.cc`
- `src/memory_node/storage_owner_index.cc`
- `src/memory_node/storage_owner_state.hh`
