# 第 11 课：在线插入与 RobustPrune

## 本课目标

本课讲普通 compute-side 在线插入路径，也就是 `insert_execution` 不走 storage-owner 时的 `Vamana::insert`。你需要理解 first insert、medoid CAS、插入时 beam search、GPU RobustPrune、新节点写入、反向边更新和 overflow prune。

## 代码证据

必须阅读：

- `src/service/compute_service/storage_owner_insert.ipp`
- `src/http/vamana_service_scheduler.hh`
- `src/vamana/vamana_insert.ipp`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/vamana_rdma_writes.hh`
- `src/rdma/vamana_rdma_reads.hh`
- `src/gpu/gpu_kernel_launcher.cu`

## 普通插入入口

当 `config_.use_storage_owner_insert()` 为 false 时：

```text
ComputeService::insert
  为每个 InsertItem 创建 InsertRequest
  insert_queue_.enqueue(request)
  等待 future
```

insert worker 在 `vamana_service_schedule_inserts` 中：

```text
queue.try_dequeue(req)
拷贝 req->components 到 staging slot
coroutine.handle = vamana_idx.insert(req->id, slot_components, thread, &req->new_ptr)
coroutine done 后 req->result.set_value(true)
```

插入成功后 compute side 会：

- `publish_compute_side_id(id, new_ptr, false, new_ptr.memory_node())`
- 如果有 RaBitQ cache，更新 dynamic cache。

## first insert

`Vamana::insert` 开始先读 medoid：

```text
medoid_ptr = read_medoid_ptr()
```

如果 medoid 为空：

1. `allocate_vamana_node` 远程 FAA 分配节点。
2. `write_vamana_node` 写一个无邻居节点。
3. `swap_medoid_ptr(RemotePtr{}, new_ptr)` 尝试 CAS 设置 medoid。
4. 如果 CAS 成功，写 header 标记 `HEADER_IS_MEDOID`。
5. 如果 CAS 失败，说明其他线程先完成 first insert，使用 observed medoid 继续普通插入。

这个流程用 CAS 解决空索引并发 first insert 竞争。

## 插入候选搜索

普通插入在 medoid 非空时：

```text
read medoid node
CPU distance(query, medoid)
beam = {medoid}
visited = {medoid}
query H2D
while true:
  select best unexpanded
  read neighbor list
  filter unvisited
  batch_read_vectors(unvisited)
  candidate vectors to GPU
  launch_batch_typed_l2_distances
  D2H distances
  insert_into_beam(..., beam_width_construction_)
```

注意插入搜索使用 `beam_width_construction_`，不是查询用的 `beam_width_`。

插入路径不使用 RaBitQ gate，候选距离是 exact L2 GPU 距离。

## RobustPrune 选择新节点邻居

候选搜索结束后：

```text
sort beam by distance
candidate_rptrs = beam rptrs
batch_read_vectors(candidate_rptrs)
stage candidate vectors and candidate distances
launch_robust_prune_typed(... alpha_, R_)
D2H pruned indices and count
selected_neighbors = candidate_rptrs[pruned_indices]
```

`alpha_` 和 `R_` 在这里发挥作用：

- `R_` 控制新节点最多多少邻居。
- `alpha_` 控制 RobustPrune 的多样性剪枝。

## 新节点写入

新节点写入分两步：

```text
new_ptr = allocate_vamana_node(thread)
new_node = write_vamana_node(new_ptr, id, components, selected_neighbors, pruned_count)
```

`allocate_vamana_node`：

- 随机选择 memory node。
- 对该 memory node offset 0 的 free pointer 执行 FAA。
- 返回 `RemotePtr{memory_node, old_free_ptr}`。

`write_vamana_node`：

- 在 compute local buffer 组装节点。
- 写 header、id、edge_count 或 generation、vector、neighbors、可选 RaBitQ entry。
- 如果 hot graph 可用，额外写 hot graph entry。
- RDMA WRITE 到 `new_ptr.byte_offset()`。

## 反向边更新

对每个 `selected_neighbor`：

1. 读 neighbor node prefix。
2. CAS spinlock neighbor header。
3. 读 neighbor 的 neighbor list。
4. 如果 neighbor degree < R，append 新节点并写回。
5. 否则做 overflow prune。
6. unlock neighbor。

这一步让图尽量保持双向可达性，但它也是普通在线插入最重的部分。

## overflow prune

当某个 neighbor 已满时，需要从：

```text
neighbor 当前邻居 + new_ptr
```

中重新选最多 R 个。

流程：

1. 批量读取原邻居向量。
2. 把 neighbor 自己的 vector 上传为 query。
3. 把所有候选向量上传到 GPU，最后一个是新插入向量。
4. GPU 计算 neighbor 到每个候选的距离。
5. CPU sort 候选距离，上传 distances 和 order。
6. `launch_robust_prune_typed`。
7. D2H pruned indices。
8. `write_vamana_neighbors` 写回 neighbor list。

这条路径的统计很多：

- `build_overflow_prunes`
- `build_overflow_prune_candidates`
- `build_overflow_prune_pair_checks_upper_bound`
- `build_overflow_prune_global_load_bytes_upper_bound`
- `build_overflow_prune_kernel_threads`

## 插入一致性边界

普通插入的一致性依赖：

- free pointer FAA 保证新节点分配不重叠。
- medoid pointer CAS 处理空索引竞争。
- neighbor header CAS lock 保护 neighbor list 更新。
- unlock 通过写 header lock byte 清除。

但它不是事务：

- 新节点写入成功后，部分反向边更新可能已经发生，后续失败没有完整回滚。
- upsert/delete 通过 deleted bit 和 compute-side idmap 处理，而不是物理删除节点。

## 性能影响

主要开销：

- 插入搜索会产生大量 neighbor RDMA 和 vector RDMA。
- RobustPrune 需要读取所有候选向量并跑 GPU kernel。
- 反向边更新对每个 selected neighbor 串行进行。
- neighbor full 时 overflow prune 成本很高。
- CAS spinlock 在高并发插入同一区域时可能产生重试。

可观测指标：

- `build_distcomps`
- `build_vector_rdma_reads_in_bytes`
- `build_l2_kernels`
- `build_prune_kernels`
- `remote_allocations`
- `lock_attempts`
- `lock_retries`
- `cas_failures`
- `build_overflow_prunes`

## 设计异味

1. 普通插入把搜索、剪枝、写节点、反向边维护放在一个长函数中。
2. 反向边更新逐个 neighbor 串行执行，无法充分利用 batch。
3. overflow prune 混合 CPU sort、GPU distance、GPU prune，数据移动复杂。
4. insert path 强绑定 GPU，即使 memory node 有 CPU 也不能独立完成普通插入。
5. 插入失败没有显式事务状态或补偿机制。

## 可验证问题

- first insert 如何避免两个线程同时设置 medoid？
- 插入搜索为什么使用 `beam_width_construction_`？
- 新节点分配为什么用 FAA？
- neighbor list 更新为什么要 CAS lock？
- overflow prune 什么时候触发？

## 学习任务

1. 画出 `Vamana::insert` 的五个阶段：medoid、search、prune、write、reverse update。
2. 在 `vamana_insert.ipp` 标出每个 GPU kernel launch。
3. 跟踪一次 neighbor full 的 overflow prune 数据流。
4. 统计插入路径中所有 RDMA READ、WRITE、CAS、FAA。
5. 思考：如果要优化插入尾延迟，应该先减少 selected_neighbors 数、批量反向边，还是改 overflow prune？

