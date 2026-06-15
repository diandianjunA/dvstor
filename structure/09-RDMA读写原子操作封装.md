# 第9课：RDMA读写原子操作封装

## 学习目标
- 理解`src/rdma/`中Vamana专用RDMA操作的设计模式
- 掌握awaitable模式的协程集成
- 理解GPUDirect RDMA与Host Staging的两种数据路径

## 内容大纲

### 1. RDMA封装三层结构
```
rdma::vamana::read_vamana_node()      ← 高层语义操作
  └── qp->post_send(...RDMA_READ...)  ← QP底层接口
        └── ibv_post_send()           ← libibverbs系统调用
```

### 2. RDMA Read操作 (`vamana_rdma_reads.hh`)

| 函数 | RDMA次数 | 说明 |
|------|----------|------|
| `read_medoid_ptr` | 1次读8B | 读Memory Node 0偏移8处的Medoid指针 |
| `read_vamana_node` | 1次读 | 读header+meta+vector |
| `read_vamana_node_full` | 1次读 | 读完整节点（含邻居） |
| `read_vamana_id` | 1次读4B | 仅读节点ID |
| `read_vamana_neighbors` | 2次读 | 分别读edge_count和neighbor_slots |
| `batch_read_vectors` | N个READ WR | 批量读N个节点的向量；每条signaled WR链只产生一个完成事件 |
| `read_vamana_nodes` | N次读 | 批量读N个完整节点 |

### 3. 统计追踪模式
```cpp
inline void track_total_rdma_read(thread, bytes, ops=1) {
    thread->stats.rdma_reads_in_bytes += bytes;
    thread->stats.rdma_read_ops += ops;
    // 同时按worker角色（insert/query）分类统计
}
```
所有RDMA操作都经过三级统计：总数 + 按角色 + 按数据类型（邻居/向量）

### 4. Awaitable模式详解
```cpp
struct awaitable {
    byte_t* node_ptr;
    size_t read_size;
    RemotePtr rptr;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }  // 总是挂起
    static void await_suspend(std::coroutine_handle<>) {}  // 空操作
    s_ptr<VamanaNode> await_resume() {  // 恢复时构造VamanaNode
        return std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get());
    }
};
```
- `await_ready() = false`: 总是挂起等待RDMA完成
- `await_suspend()`: 不做任何事（CQ polling由调度器统一处理）
- `await_resume()`: 将RDMA已完成的缓冲区包装为VamanaNode

### 5. RDMA Write操作 (`vamana_rdma_writes.hh`)

| 函数 | 说明 |
|------|------|
| `write_vamana_node` | 写入完整新节点（header+meta+vector+neighbors） |
| `write_vamana_neighbors` | 写入邻居列表（edge_count+neighbor_slots，两次RDMA） |
| `write_medoid_ptr` | 写入Medoid指针（8B inlined write） |
| `write_vamana_header` | 写入节点Header（inlined write） |
| `unlock_vamana_node` | 写入0到锁位（1B inlined write） |

### 6. RDMA Atomic操作 (`vamana_rdma_atomics.hh`)

| 函数 | RDMA操作 | 说明 |
|------|----------|------|
| `try_lock_vamana_node` | CAS | 尝试加锁，返回(success, original_value) |
| `spinlock_vamana_node` | CAS循环 | 自旋直到成功加锁 |
| `allocate_vamana_node` | FAA | 原子递增free_ptr分配节点空间 |
| `swap_medoid_ptr` | CAS | 原子交换Medoid指针 |

### 7. GPUDirect RDMA路径
```cpp
// 直接路径: RDMA→GPU d_candidate_vecs
batch_read_vectors(..., gs.d_candidate_vecs, gs.d_candidate_vecs_lkey);
// → QP使用GPU内存的lkey，RDMA网卡直接DMA到GPU内存

// 间接指针路径: RDMA→分散的GPU staging buffers
batch_read_vectors(..., &destinations);
// → 每个向量可能写入不同的GPU地址
```

注册流程在`gpu_buffer_manager.cu`中：`ibv_reg_mr(pd, d_candidate_vecs, bytes, IBV_ACCESS_LOCAL_WRITE)`

### 8. 自适应向量READ批处理

向量地址通常不连续，因此不能把N个向量直接变成一个RDMA READ。当前实现仍为每个向量构造一个READ WR，但会：

- 按memory node分组，并将bulk读取分散到QP1..N；QP0保留给medoid、neighbor和原子操作。
- 根据QP outstanding WR选择负载最低的QP，而不是每批都从QP0开始。
- 默认按`2 * max_qp_init_rd_atom`限制每条WR链长度，链尾才设置`IBV_SEND_SIGNALED`。
- 每条链调用一次`track_post()`，CQE到达后同时释放协程余额和QP send credit。
- 对SQ临时满产生的partial post从`bad_wr`继续提交，不静默丢失链尾完成事件。

相关配置为`rdma-read-batch-mode`、`rdma-qp-pool-size`、`rdma-read-chain-size`和
`rdma-read-max-inflight-wrs`。`adaptive`为默认模式，`legacy`配合自动QP配置时恢复单QP旧行为。

## 课后任务
1. 统计：一次搜索调用中每种RDMA操作的次数
2. 对比`write_vamana_neighbors`的两次RDMA写与单次RDMA写的优劣
3. 画出GPUDirect RDMA的数据流图

## 参考文件
- `src/rdma/vamana_rdma_reads.hh`
- `src/rdma/vamana_rdma_writes.hh`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/vamana_rdma_operations.hh`
