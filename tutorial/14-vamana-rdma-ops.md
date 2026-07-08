# 第 14 课：Vamana RDMA read/write/atomic 封装

## 本课目标

本课讲 `src/rdma/vamana_rdma_*` 如何把底层 RDMA verbs 封装成 Vamana 语义操作。学完后，你应该能解释 medoid 读取、节点读取、邻居读取、向量批量读取、新节点写入、邻居写回、lock、allocate、CAS medoid 的具体远端地址和完成方式。

## 代码证据

必须阅读：

- `src/rdma/vamana_rdma_reads.hh`
- `src/rdma/vamana_rdma_writes.hh`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/rdma_send_chain.hh`
- `src/vamana/storage_layout_resolver.hh`
- `src/shared_context.hh`

## RDMA read 分类

主要 read 操作：

| 函数 | 读取内容 | 远端 offset |
| --- | --- | --- |
| `read_medoid_ptr` | medoid RemotePtr | memory node 0 offset 8 |
| `read_vamana_node` | header 到 vector end | `rptr.byte_offset()` |
| `read_vamana_node_prefix` | node prefix | `rptr.byte_offset()` |
| `read_vamana_node_full` | full node | `rptr.byte_offset()` |
| `read_vamana_id` | header + id | `StorageLayoutResolver::header(rptr)` |
| `read_vamana_neighbors` | neighbor read buffer 或 hot graph entry | `StorageLayoutResolver::neighbor_read(rptr)` |
| `batch_read_vectors` | vector 或指定 offset bytes | `StorageLayoutResolver::vector(rptr)` |

## awaitable 模式

典型 read：

```text
allocate local buffer
track stats
thread->track_post()
qp.post_send(... IBV_WR_RDMA_READ ..., wr_id=thread->create_wr_id())
return awaitable
```

awaitable：

- `await_ready` 返回 false。
- `await_suspend` 不做事。
- `await_resume` 把 local buffer 包成 `VamanaNode`、`VamanaNeighborlist` 或结果对象。

真正等待依赖 scheduler 轮询 CQ，减少 `post_balances`。

## neighbor read 特殊性

`read_vamana_neighbors` 使用 `StorageLayoutResolver::neighbor_read`：

- AoS：直接读取 id、edge_count、neighbors。
- compact：读取 hot graph entry。

如果 compact：

1. RDMA READ hot graph entry。
2. `await_resume` 调 `decode_hot_graph_entry`。
3. 解码到传统 neighbor read buffer。
4. 返回 `VamanaNeighborlist`。

如果 checksum 失败，会重试最多 3 次。

## vector batch read

`batch_read_vectors` 是最复杂的 read 封装。它负责：

1. 确定每个 request 的 memory node。
2. 获取每个 node 的 QP 数。
3. 获取 outstanding WR snapshot。
4. 调 `plan_vector_read_batch` 生成 chunks。
5. 为每个 request 确定 local addr/lkey 和 remote addr/rkey。
6. 为每个 chunk 构造 WR chain。
7. reserve QP WR credit。
8. 申请 batch completion token。
9. 只在 chain 最后一个 WR 上 signaled。
10. `post_send_chain_with_retry`。

它既支持：

- host buffer destination
- GPU device buffer destination
- 自定义 destination array
- 自定义 node offset 和 read size

因此 RaBitQ、vector read、其他 offset read 都复用这套函数。

## RDMA write 分类

主要 write 操作：

| 函数 | 写入内容 |
| --- | --- |
| `write_vamana_node` | 完整新节点，另可写 hot graph entry |
| `write_vamana_neighbors` | 邻居列表，AoS 写 edge_count 和 slots，compact 写 hot graph |
| `write_medoid_ptr` | medoid pointer |
| `write_vamana_header` | header |
| `unlock_vamana_node` | header lock byte 清 0 |

写操作也会：

- 构造本地 buffer。
- `thread->track_post()`。
- RDMA WRITE。
- await 完成后释放 buffer 或返回 node view。

## atomic 操作

`try_lock_vamana_node`：

```text
compare = expected_header without NODE_LOCK
swap = compare | NODE_LOCK
RDMA CAS header offset
await_resume:
  original = pointer_slot
  success = original == compare
```

`spinlock_vamana_node`：

- 循环 `try_lock_vamana_node`。
- 失败时更新 lock retries 和 cas failures。
- 每 100000 次失败输出诊断。

`allocate_vamana_node`：

- 随机 memory node。
- 对 offset 0 free pointer FAA 加 node allocation size。
- 返回 old value 作为新节点 offset。

`swap_medoid_ptr`：

- 对 memory node 0 offset 8 做 CAS。

## StorageLayoutResolver 的价值

不要直接在业务代码里手算 offset。resolver 负责：

- header
- id
- generation
- edge_count
- vector
- rabitq
- neighbor_read
- neighbor_slots

对于 compact storage，neighbor 相关 offset 会转到 hot graph entry。

## 统计埋点

read/write wrappers 更新：

- `rdma_reads_in_bytes`
- `rdma_writes_in_bytes`
- `rdma_read_ops`
- `rdma_write_ops`
- query/build 分组
- neighbor/vector 分组
- batch calls/chunks/active nodes/active qps/credit waits/post retries

这些统计是第 27 和第 29 课优化分析的基础。

## 性能影响

- 小 RDMA READ，如 medoid、id、neighbor，受 latency 主导。
- vector batch READ 受带宽、QP 并行、chain size、credit 主导。
- CAS spinlock 在竞争下可能放大 tail latency。
- FAA 分配简单快速，但导致随机 memory node 分配，不考虑局部性。
- compact hot graph 减少 neighbor read bytes，但多一次 decode 和 checksum。

## 设计异味

1. RDMA wrapper 是 header-only，函数很长，编译和阅读成本高。
2. `batch_read_vectors` 过于通用，职责横跨 planning、posting、统计、buffer ownership。
3. `allocate_vamana_node` 随机选择 memory node，和后续 locality-aware placement 思路冲突。
4. atomic lock 没有 backoff，只是 yield 和诊断。
5. `read_vamana_id` 为结果阶段逐个小读，可能适合批量化。

## 可验证问题

- medoid pointer 读哪个 memory node 和 offset？
- AoS 和 compact 的 neighbor read offset 有何不同？
- batch vector read 为什么只 track 一个 coroutine post？
- CAS 返回的 original value 放在哪里？
- `write_vamana_neighbors` 在 compact 模式写什么？

## 学习任务

1. 做一张 RDMA operation 到 offset/size 的表。
2. 跟踪 `allocate_vamana_node` 从 FAA 到 `RemotePtr` 的路径。
3. 跟踪 `spinlock_vamana_node` 的失败统计如何进入 breakdown。
4. 将 `read_vamana_id` 改成理论上的 batch read 方案，写出需要的接口。
5. 思考：RDMA wrapper 如何拆成 address resolver、buffer manager、poster、statistics 四层？

