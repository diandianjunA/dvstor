# 第 09 课：在线查询主路径之二：批量 beam expansion

## 本课目标

本课讲查询最核心的循环：如何选择多个未展开 beam 节点，如何批量读取 neighbor list，如何过滤 unvisited candidate，如何用 RDMA vector batch planner 把候选向量读取拆到不同 memory node 和 QP。学完本课，你应该能够解释 `expansion_batch`、`credit_aware_expansion`、QP credit 和 candidate batch size 之间的关系。

## 代码证据

必须阅读：

- `src/vamana/vamana_search.ipp`
- `src/rdma/vamana_rdma_reads.hh`
- `src/rdma/vector_batch_planner.hh`
- `src/shared_context.hh`
- `src/common/statistics.hh`

## expansion 循环概览

`knn_raw` 初始化 beam 后进入循环。每一轮大致分成：

```text
Phase 1:
  consume pending neighbor reads
  decode neighbor list
  filter visited
  collect all_unvisited

Phase 2:
  如果使用 RaBitQ:
    estimate/gate
    exact vector RDMA for selected candidates
  否则:
    vector RDMA for all_unvisited

Phase 3:
  GPU distance
  可能提前 issue 下一轮 neighbor reads
  D2H distances
  beam update
  credit controller record round
```

第一轮 cold start 只发起一个 neighbor read：

```text
select_best
mark expanded
read_vamana_neighbors
pending_K = 1
```

之后每轮可以根据 `expansion_batch_` 和 credit controller 发起多个 neighbor read。

## select_best

`select_best` 是线性扫描：

```text
best = -1
best_d = max
for i in beam:
  if !expanded && distance < best_d:
    best = i
return best
```

这个实现简单，但 beam 大或 expansion 轮数多时 CPU 成本可见。它和 `insert_into_beam` 一起构成查询 CPU 热路径。

## issue_plain_neighbor_reads

`issue_plain_neighbor_reads(desired, start_slot, precommit)` 做：

1. 从 `start_slot` 开始填 `pf_neighbors`。
2. 每次调用 `select_best`。
3. 将 beam entry 标记 expanded。
4. 发起 `rdma::vamana::read_vamana_neighbors`。
5. 更新 credit-aware 统计。

这个函数只是发起 neighbor RDMA，不等待完成。等待发生在下一轮 consume 阶段。

## consume neighbor reads

循环中对 `pending_K` 个 neighbor awaitable：

```text
thread->poll_cq()
如果当前 coroutine post_balances == 0:
  pf_neighbors[k].mark_ready()
nlist = co_await pf_neighbors[k]
for neighbor in nlist->view():
  if not null and not visited:
    visited.insert
    all_unvisited.push_back
```

注意：这里用当前 coroutine 的 `post_balances` 判断全部 RDMA 是否完成，而不是逐个 awaitable 自带 completion 状态。多个 neighbor read 都绑定同一个 coroutine balance。

## batch vector read

没有 RaBitQ 时，对本轮所有候选：

```text
rdma::vamana::batch_read_vectors(all_unvisited, thread, optional gpu_buffer, optional lkey)
```

如果 GPUDirect candidate buffer 可用，local destination 是 GPU device buffer；否则读到 host buffer，再 memcpy 到 pinned host staging，再 H2D 到 GPU。

## VectorReadPlanner

`src/rdma/vector_batch_planner.hh` 是纯 planning helper。输入：

- `request_nodes`: 每个候选属于哪个 memory node。
- `qp_counts`: 每个 memory node 有多少 QP。
- `outstanding_wrs`: 当前各 QP outstanding WR。
- `tie_breakers`: 打散同负载选择。
- `max_chain_wrs`: 每条 chain 最多 WR 数。
- `adaptive`: 是否 adaptive。

输出：

- `chunks`
- `request_order`
- `active_nodes`
- `active_qps`
- `max_chain_wrs`

每个 `VectorReadChunkPlan`：

```cpp
struct VectorReadChunkPlan {
  u32 memory_node;
  u32 qp_index;
  u32 request_offset;
  u32 request_count;
};
```

## 非 adaptive 模式

如果 `adaptive=false`：

- 每个 node 的请求按 `i % qp_counts[node]` 分散到 QP。
- 每个非空 QP 生成一个 chunk。
- 不考虑当前 outstanding WR。

优点是简单可预测。缺点是在热点 memory node 和已有 outstanding 负载不均时，可能继续打到繁忙 QP。

## adaptive 模式

如果 `adaptive=true`：

- QP0 在有 bulk lane 时保留为低延迟控制 lane。
- bulk QP 从 1 开始。
- 根据 `outstanding_wrs` 初始化 projected load。
- 按 `chain_limit` 把该 node 的请求切块。
- 每块选择 projected load 最小的 QP，tie 用 tie breaker 打散。

这就是 adaptive multi-QP RDMA scheduling 的核心。

## chained READ WR

`batch_read_vectors` 对每个 chunk 构造一条 WR chain：

```text
wr[0] -> wr[1] -> ... -> wr[n-1]
```

只有最后一个 WR：

- 设置 `wr_id = completion_id`
- 设置 `IBV_SEND_SIGNALED`

然后调用 `post_send_chain_with_retry`。如果 `ibv_post_send` 返回 ENOMEM/EAGAIN/EBUSY，会从 `bad` WR 继续重试，并在重试中 poll CQ。

## QP credit 和 completion token

发 chain 之前：

```text
try_reserve_bulk_qp_wrs(node, preferred_qp, wr_count, selected_qp)
```

如果失败：

- `vector_rdma_credit_waits++`
- 累计 `vector_rdma_credit_wait_ns`
- poll CQ + yield

然后申请 batch completion：

```text
try_create_batch_completion(thread_index, coroutine_id, node, qp, wr_count)
```

如果 completion slot 不足：

- `vector_rdma_completion_token_waits++`
- poll CQ + yield

## credit-aware expansion

`CreditExpansionController` 根据每轮表现动态调节 issue width。

关键输入：

- issued expansions
- frontier candidates
- exact candidates
- 是否 credit stall
- 是否 progressed
- graph degree
- target candidates
- cost guard

可能行为：

- underfilled 且 progressed，增加 `issue_k`。
- credit stall、overfilled、cost too high、连续无进展，减少 `issue_k`。
- lookahead 也会随 stall/no-progress 收缩或扩张。

这表示 `expansion_batch` 是上限，而 credit-aware 会在运行时选择实际 issue 宽度。

## 约束关系

| 参数或状态 | 影响 |
| --- | --- |
| `R` | 每个 expanded node 最多产生 R 个候选 |
| `expansion_batch` | 每轮最多展开多少 beam node |
| `beam_width` | beam 最大保存候选数 |
| `gpu.max_batch()` | 本轮 exact candidates 上限 |
| `rdma_read_chain_size` | 每个 chunk 最多 WR 数 |
| `rdma_read_max_inflight_wrs` | 每个 QP outstanding 上限 |
| QP pool size | 同一 memory node 可并行的 bulk lane 数 |
| partition strategy | 候选分布到多少 memory node |
| RaBitQ gate | exact vector read candidate 数 |

## 性能影响

- `all_unvisited` 越大，RDMA vector bytes 和 GPU distance 成本越高。
- `expansion_batch` 增大能减少迭代次数和 kernel launch 次数，但可能过度扩展无用候选。
- adaptive QP planner 能提升热点 shard 的 RDMA 并行性。
- chain WR 减少 CQE，但过长可能占用 QP credit，增加 tail latency。
- precommit neighbor read 能和 D2H 重叠，但可能发起后来证明没必要的 neighbor read。

## 设计异味

1. neighbor read completion readiness 依赖整个 coroutine 的 `post_balances`，粒度较粗。
2. `CreditExpansionController` 是 `knn_raw` 内部局部 struct，难以单元测试。
3. `batch_read_vectors` 同时做 planning、buffer 准备、WR 构造、credit、post、统计，职责很重。
4. adaptive planner 是纯函数，但目前没有顶层 `test/` 目录覆盖。
5. `select_best` 每次线性扫描，随着 beam 宽度增大可能成为 CPU 开销。

## 可验证问题

- `expansion_batch` 是否一定等于每轮展开数？
- adaptive 模式下为什么 QP0 可能被保留？
- 一条 RDMA chain 有多少 CQE？
- completion slot 不足时会记录哪个统计？
- RaBitQ gate 如何改变 exact vector RDMA 数量？

## 学习任务

1. 用一个例子手算：10 个候选分布在两个 memory node，QP 数分别为 3 和 2，adaptive planner 会生成哪些 chunk。
2. 在 `batch_read_vectors` 中标出 local addr、remote addr、lkey、rkey 的来源。
3. 搜索 `query_credit_*` 统计字段，理解每个字段的含义。
4. 对比 `adaptive=true/false` 的 chunk 分配差异。
5. 思考：如果 `select_best` 改为 heap，需要如何处理 beam 中 distance 更新和 expanded 标记？

