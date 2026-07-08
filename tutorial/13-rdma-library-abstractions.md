# 第 13 课：RDMA library 底层抽象

## 本课目标

本课讲 `rdma-library/library` 里的底层 RDMA 封装。它不理解 Vamana，只负责 verbs 对象、TCP 交换 QP 信息、QP 状态迁移、memory region 注册、SEND/RECV、RDMA READ/WRITE、CAS、FAA 和 CQ polling。

## 代码证据

必须阅读：

- `rdma-library/library/context.hh`
- `rdma-library/library/context.cc`
- `rdma-library/library/queue_pair.hh`
- `rdma-library/library/queue_pair.cc`
- `rdma-library/library/memory_region.hh`
- `rdma-library/library/memory_region.cc`
- `rdma-library/library/connection_manager.hh`
- `rdma-library/library/connection_manager.cc`

## Context

`Context` 封装：

- `ibv_context`
- `ibv_pd`
- send CQ
- receive CQ
- optional SRQ
- port attributes
- device attributes
- TCP server socket for QP info exchange

构造时：

```text
ibv_get_device_list
选择 ib_device 或 device_idx
ibv_open_device
ibv_query_device
ibv_alloc_pd
ibv_query_port
ibv_create_cq(send)
ibv_create_cq(receive)
optional ibv_create_srq
```

析构时按相反顺序销毁 SRQ、CQ、PD、device，并关闭 server socket。

## QP 连接过程

`Context::wait_for_connection`：

1. 创建本地 `QueuePair`。
2. TCP accept。
3. 接收远端 `QPInfo`。
4. 发送本地 `QPInfo`。
5. `transition_to_rtr`。
6. `transition_to_rts`。

`Context::connect_to_server`：

1. 创建本地 `QueuePair`。
2. TCP connect。
3. 发送本地 `QPInfo`。
4. 接收远端 `QPInfo`。
5. `transition_to_rtr`。
6. `transition_to_rts`。

`QPInfo` 包含：

- `lid`
- `qp_number`
- `node_id`

## QueuePair 状态迁移

`QueuePair` 构造后立即：

```text
RESET -> INIT
```

`transition_to_init` 设置：

- `qp_state = IBV_QPS_INIT`
- `pkey_index`
- `port_num`
- access flags:
  - remote write
  - remote read
  - local write
  - remote atomic

`transition_to_rtr` 设置：

- MTU
- remote QP number
- remote LID
- `max_dest_rd_atomic`
- RNR timer

`transition_to_rts` 设置：

- timeout
- retry count
- RNR retry
- `max_rd_atomic`

这说明项目使用 reliable connected QP。

## MemoryRegion 和 token

`MemoryRegion` 包装 `ibv_reg_mr`。`MemoryRegionToken` 是远端访问需要的三元组：

```cpp
struct MemoryRegionToken {
  u64 address;
  u32 lkey;
  u32 rkey;
};
```

对远端来说最重要的是：

- `address`
- `rkey`

本地 post send 时最重要的是：

- local address
- local lkey

compute node 需要 memory node 发来的 token 才能做 one-sided RDMA。

## QueuePair post 接口

封装了：

- `post_receive`
- `post_send_inlined`
- `post_send`
- `post_send_with_id`
- `post_CAS`
- `post_FAA`

`post_send` 根据 opcode 决定是否需要 remote token：

- `IBV_WR_SEND`: 不需要 remote token。
- `IBV_WR_RDMA_READ`: 需要 remote token。
- `IBV_WR_RDMA_WRITE`: 需要 remote token。

inline send 要求 size 不超过 `INLINE_SIZE = 256`。

## CAS 和 FAA

`post_CAS`：

- opcode: `IBV_WR_ATOMIC_CMP_AND_SWP`
- remote offset 必须 8B 对齐。
- local SGE 长度 8。
- completion 后 local buffer 中是 original value。

`post_FAA`：

- opcode: `IBV_WR_ATOMIC_FETCH_AND_ADD`
- 用于 free pointer 分配节点。

在 Vamana 中：

- CAS 用于 node lock、medoid swap。
- FAA 用于远程 bump allocation。

## CQ polling

`Context::poll_recv_cq`：

- 调用 `ibv_poll_cq`。
- 检查 `wc.status == IBV_WC_SUCCESS`。
- 如果提供 `ReceiveInfo`，根据 wr_id 恢复 MemoryRegion pointer 和 byte_len。

`Context::poll_send_cq`：

- 调用 `ibv_poll_cq`。
- 检查状态。
- 对每个 completion 调用 `id_handler(wr_id)`。

上层用 `wr_id` 编码 coroutine id、completion slot 或 peer request id。

## ConnectionManager

两类 connection manager：

- `ServerConnectionManager`
- `ClientConnectionManager`

memory node 使用 server connection manager 连接 compute clients。

compute node 使用 client connection manager：

- initiator 负责更多客户端间连接和 ID 分发。
- `server_qps` 指向 memory nodes。
- initiator 有 `client_qps` 指向其他 compute clients。
- non-initiator 有 `initiator_qp`。

这为 routing RPC 和 memory node RDMA 都提供连接基础。

## 性能影响

- CQ size 来自 `max_send_queue_wr` 和 `max_recv_queue_wr`。
- QP `sq_sig_all = 0`，只有显式 signaled WR 产生 CQE。批量 RDMA 依赖这个降低 CQE 开销。
- `max_qp_read_atomic` 和 `max_qp_dest_read_atomic` 被限制到最多 16，用于 RDMA read credit。
- inline SEND 减少小消息本地 MR 依赖，但受 `INLINE_SIZE` 限制。
- TCP 只用于建连，不在数据路径。

## 设计异味

1. verbs 错误处理大量用 `lib_assert`，生产环境恢复能力有限。
2. `Context` 既管理 RDMA device，又管理 TCP server socket。
3. QP 参数固定，如 MTU、timeout、retry count，没有配置化。
4. `QueuePair::post_send` 对 `bad_work_request` 没有像 batch chain 那样做 retry。
5. `MemoryRegionToken` 暴露 lkey，但远端访问主要只需要 address/rkey，语义略混杂。

## 可验证问题

- QP 是哪种类型？
- 为什么 CAS remote offset 必须 8B 对齐？
- 哪些 WR 会产生 completion？
- TCP 在 RDMA 数据路径中是否参与？
- `MemoryRegionToken` 在 compute node 侧如何使用？

## 学习任务

1. 画出 Context、PD、CQ、QP、MR 的关系图。
2. 跟踪一次 `connect_to_server` 的 QP 信息交换。
3. 找出所有调用 `post_CAS` 和 `post_FAA` 的上层代码。
4. 比较 `post_send_inlined` 和 `post_send_with_id` 的使用场景。
5. 思考：如果要支持 RoCE 或不同 MTU，当前哪些参数需要配置化？

