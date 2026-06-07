# 第8课：RDMA库底层实现

## 学习目标
- 掌握`rdma-library`的完整架构
- 理解QP管理、内存注册、连接建立
- 熟悉RDMA Read/Write/Atomic三种操作的使用场景

## 内容大纲

### 1. RDMA库架构 (`rdma-library/library/`)
```
context.hh/.cc         — RDMA设备上下文（ibv_context, PD, CQ管理）
configuration.hh/.cc   — 网络配置（IP、端口、设备选择）
connection_manager.hh/.cc — 连接建立（Server/Client模式）
queue_pair.hh/.cc      — QP操作封装（post_send, post_CAS, post_FAA）
memory_region.hh/.cc   — 内存注册（Local/Remote MR）
hugepage.hh            — 大页内存分配
thread.hh              — 线程抽象
latch.hh               — 线程同步屏障
detached_qp.hh         — 分离QP（共享CQ的多连接场景）
batched_read.hh        — 批量RDMA读
types.hh               — 基本类型定义（u32, u64, vec, str等）
```

### 2. Context: RDMA设备管理
```cpp
class Context {
    ibv_device* device_;         // RDMA设备
    ibv_context* context_;       // 设备上下文
    ibv_pd* protection_domain_;  // 保护域(PD)
    ibv_cq* send_cq_;            // 发送完成队列
    ibv_cq* receive_cq_;         // 接收完成队列
    ibv_srq* shared_receive_cq_; // 共享接收队列(可选)
    ibv_port_attr port_attributes_; // 端口属性(LID等)
};
```

### 3. ConnectionManager: 连接建立协议
**Server模式** (Memory Node使用):
```cpp
ServerConnectionManager(context, config) {
    // 1. 绑定TCP端口
    // 2. 等待所有客户端连接 (accept + QP交换)
    // 3. 同步: 广播确认消息
}
```

**Client模式** (Compute Node使用):
```cpp
ClientConnectionManager(context, config) {
    // 1. 连接所有服务器
    // 2. 客户端间互联
    // 3. 分发客户端ID
    // 4. 同步确认
}
```

### 4. QueuePair: RDMA操作封装
```cpp
class QueuePair {
    // 核心操作
    post_send(local_addr, size, lkey, opcode, signaled, inlined, remote_mrt, remote_offset, imm_data, wr_id);
    post_send_inlined(data, size, opcode, signaled, remote_mrt, remote_offset, wr_id);
    post_CAS(local_addr, lkey, remote_mrt, remote_offset, compare, swap, signaled, wr_id);
    post_FAA(local_addr, lkey, remote_mrt, remote_offset, add, signaled, wr_id);
    post_receive(mr, size, wr_id, offset);
};
```

### 5. 三种RDMA操作的使用场景

| 操作 | 用途 | 示例 |
|------|------|------|
| RDMA READ | 单边读取远程数据 | 读取VamanaNode、邻居列表、向量 |
| RDMA WRITE | 单边写入远程数据 | 写入新节点、更新邻居、解锁 |
| RDMA CAS | 原子比较并交换 | 锁定节点、设置Medoid |
| RDMA FAA | 原子加并返回旧值 | 分配节点（移动free_ptr） |

### 6. Memory Region管理
- **LocalMemoryRegion**: 注册本地内存供RDMA访问
- **MemoryRegionToken**: 远程MR的rkey+地址信息，用于RDMA操作
- **HugePage**: 大页分配（2MB/1GB），减少TLB miss

### 7. 完成队列轮询
```cpp
// 发送完成轮询（带回调）
Context::poll_send_cq(wcs, max_cqes, cq, [&](u64 wr_id) {
    auto [ctx_offset, coroutine_id] = decode_64bit(wr_id);
    --ctx->registered_threads[ctx_offset]->post_balances[coroutine_id];
});
```
WR ID承载了线程和协程的身份信息，使完成事件能路由到正确的协程。

## 课后任务
1. 跟踪`ServerConnectionManager::synchronize()`和`ClientConnectionManager::synchronize()`的完整执行路径
2. 画出一次RDMA READ从post_send到CQ polling的时序图
3. 实验：如果MAX_QPS从4增加到8，需要修改哪些代码？

## 参考文件
- `rdma-library/library/context.hh`、`context.cc`
- `rdma-library/library/connection_manager.hh`、`connection_manager.cc`
- `rdma-library/library/queue_pair.hh`、`queue_pair.cc`
- `rdma-library/library/memory_region.hh`、`memory_region.cc`
