# 第 16 课：Storage-owner 本地索引更新与 freshness

## 本课目标

本课讲 memory node 在 storage-owner 模式下如何维护逻辑 ID、当前 RemotePtr、generation、deleted 状态，以及 exact update 和 local-stitch update 如何影响插入/upsert/delete。

## 代码证据

必须阅读：

- `src/memory_node/storage_owner_index.cc`
- `src/memory_node/storage_owner_anchor.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/memory_node/storage_owner_runtime.cc`
- `src/vamana/idmap.hh`
- `src/vamana/anchor_index.hh`

## owner idmap

离线 `shard_writer` 会为每个 owner 写 idmap sidecar：

```text
owner = dataset.id(i) % num_memory_nodes
Entry:
  id
  rptr_raw
  generation = 0
  flags = 0
```

memory node 启动时如果 metadata 要求 `owner_sharded_v1`，调用：

```text
load_owner_idmap(index_prefix)
```

加载到：

```cpp
hashmap_t<node_t, FreshnessEntry> idmap_;
```

`FreshnessEntry`：

- current `RemotePtr`
- generation
- deleted

## mutation 准备

`prepare_mutation` 在 `idmap_mutex_` 下执行：

1. 如果 id 已在 `mutations_inflight_`，返回 failed。
2. 查 idmap。
3. 判断 exists/live。
4. 计算 new generation = previous + 1。
5. insert 如果 live，返回 already_exists。
6. erase 如果不存在，返回 not_found。
7. erase 如果已删除，返回 already_deleted。
8. 将 id 放入 `mutations_inflight_`。
9. 返回 ok。

这避免同一个 logical id 的并发 mutation 交错。

## publish mutation

执行成功后：

```text
publish_mutation(id, ptr, generation, deleted)
  idmap_[id] = FreshnessEntry{ptr, generation, deleted}
  mutations_inflight_.erase(id)
```

如果执行失败，必须确保 inflight 被清理，否则后续同 ID mutation 会被永久拒绝。这是检查 storage-owner 代码时的重要一致性点。

## mark_node_deleted

删除或 upsert 旧节点时调用：

```text
mark_node_deleted(rptr, generation)
```

逻辑：

- 如果 local shard，直接 atomic fetch_or `HEADER_DELETED`。
- 如果 remote shard，先 lock node，再 remote read header、设置 deleted、remote write。
- compact storage 下还会更新 hot graph entry 的 deleted flag 和 generation/checksum。
- 最后 unlock。

这说明删除不是物理回收，而是 logical tombstone。

## exact update 与 local-stitch

`storage_owner_update_mode` 有两个重要模式：

- `exact`
- `local_stitch`

exact 模式倾向于完整搜索/剪枝，成本高但图质量更可控。

local-stitch 模式使用 anchor hints：

```text
anchor_hints -> anchor_search_candidates
  先读取 hints
  本地/远程有限扩展
  生成候选
```

相关参数：

- `storage_owner_anchor_hints`
- `storage_owner_anchor_beam_width`
- `storage_owner_anchor_expand_cap`
- `storage_owner_anchor_remote_rescue_cap`
- `storage_owner_search_snapshot_batch`

## anchor_search_candidates

`storage_owner_anchor.cc` 中：

1. 对 anchor hints 去重和过滤。
2. 批量读取 node snapshots。
3. 对非 deleted snapshot 计算距离。
4. 插入 beam。
5. 最多扩展 `storage_owner_anchor_expand_cap` 次。
6. remote expansion 受 `storage_owner_anchor_remote_rescue_cap` 限制。
7. 返回按距离排序的 candidates。

这是一种局部修补式候选生成，不等价于完整 Vamana search。

## NodeSnapshot

storage-owner 路径使用 `NodeSnapshot`：

- rptr
- header
- id
- generation
- edge_count
- deleted
- vector_data

snapshot 可以本地读，也可以通过 peer RDMA 读远端 shard。它让 storage-owner 不必构造 compute-side `VamanaNode`。

## background maintenance

`storage_owner_maintenance.cc` 提供后台维护任务，例如：

- stitch insert finalize
- cleanup deleted node

维护任务通过队列和 worker 执行，目的是把 foreground local-stitch 的近似更新补成更准确或更干净的状态。

读这部分时要关注：

- 任务何时入队。
- generation 如何避免旧任务覆盖新 mutation。
- cleanup 是否跨 shard 发送 peer RPC。

## storage-owner 与 compute-side 插入差异

| 维度 | compute-side insert | storage-owner insert |
| --- | --- | --- |
| 执行位置 | compute node | owner memory node |
| 请求传输 | one-sided RDMA | SEND/RECV RPC |
| idmap 权威 | compute side sidecar/dynamic | owner memory node idmap |
| 删除 | compute side mark_remote_deleted | owner mark_node_deleted |
| 反向边 | compute node 逐邻居更新 | owner 本地/peer reverse update |
| anchor | 查询 entry hints 可用 | local-stitch mutation hints |
| GPU | 强依赖 compute GPU | storage-owner 主体 CPU/RDMA |

## 性能影响

- idmap mutex 是同 ID mutation 串行点。
- local-stitch 减少 foreground search 成本，但可能需要后台维护。
- snapshot batch size 影响 peer RDMA 并发和 scratch buffer 使用。
- generation 能防止旧结果污染新状态，但也增加 hot graph 更新成本。
- deleted tombstone 不回收空间，长期 upsert/delete 会增加无效节点。

## 设计异味

1. freshness/idmap、图更新、peer RPC、maintenance 逻辑都在 `MemoryNode` 内。
2. tombstone 删除缺少空间回收策略。
3. local-stitch 和 exact 路径共享部分状态，但语义差异较大。
4. generation 更新分布在 fixed node、hot graph、idmap，容易不一致。
5. background maintenance 的正确性需要更多测试护栏。

## 可验证问题

- 同一 id 并发 upsert 会被哪里挡住？
- delete 是否释放远端内存？
- hot graph entry 如何表示 deleted？
- local-stitch 的 anchor hints 从哪里来？
- generation 在 upsert/delete 中起什么作用？

## 学习任务

1. 画出 insert/upsert/erase 对 idmap 的状态机。
2. 跟踪 `prepare_mutation` 到 `publish_mutation` 的所有失败路径。
3. 画出 `anchor_search_candidates` 的 beam 扩展流程。
4. 找出所有写 `HEADER_DELETED` 的位置。
5. 思考：如果要支持空间回收，需要解决哪些 RemotePtr 和 neighbor list 问题？

