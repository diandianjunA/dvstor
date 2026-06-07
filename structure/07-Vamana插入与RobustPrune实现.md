# 第7课：Vamana插入与RobustPrune实现

## 学习目标
- 逐行理解`vamana_insert.ipp`的完整插入流程
- 掌握GPU RobustPrune和溢出剪枝（Overflow Prune）
- 理解反向边更新和并发锁机制

## 内容大纲

### 1. 插入流程概览

#### Phase 0: 首次插入（空索引处理）
```cpp
if (medoid_ptr.is_null()) {
    // 1. RDMA FAA分配节点
    // 2. RDMA写入空邻居的新节点
    // 3. RDMA CAS设置Medoid指针（可能race失败）
}
```

#### Phase 1: Beam Search找候选邻居
与搜索流程相同，但使用`beam_width_construction_`（通常更大，如200 vs 128）：
- 读取Medoid入口
- Beam Search遍历图
- 收集beam中的所有候选（不仅是top-k）

#### Phase 2: GPU RobustPrune选择邻居
```cpp
// 1. 按距离排序所有候选
std::sort(beam.begin(), beam.end(), ...);
// 2. 收集候选RemotePtr
vec<RemotePtr> candidate_rptrs;
// 3. 批量读取候选向量 (RDMA)
// 4. 上传候选向量和距离到GPU
// 5. 启动RobustPrune kernel
gpu::launch_robust_prune_typed(stream, event,
    d_candidate_vecs, candidate_dtype, d_candidate_dists,
    d_candidate_order, n_candidates, dim, alpha, R,
    d_pruned_indices, d_pruned_count);
co_await gpu::GpuAwaitable{thread.get()};
// 6. 下载剪枝结果 (D2H)
// 7. 将剪枝索引映射回RemotePtr
```

#### Phase 3: 分配并写入新节点
```cpp
RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);
s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
    new_ptr, id, components, selected_neighbors, pruned_count, false, false, thread);
```

#### Phase 5: 反向边更新（双向连接）
对每个选中的邻居节点：
1. **RDMA读取**邻居节点
2. **CAS自旋锁**锁定邻居
3. **RDMA读取**邻居的邻居列表
4. 如果邻居未满（`edge_count < R`）：直接追加new_ptr
5. 如果邻居已满：执行**Overflow Prune**
6. **RDMA写入**更新后的邻居列表
7. **RDMA写入**解锁邻居

### 2. Overflow Prune详解
当邻居节点的出度已达上限时触发：
```cpp
// 1. 收集: [已有邻居 + 新节点]
vec<RemotePtr> all_candidate_ptrs = [existing_neighbors..., new_ptr];
// 2. 批量读取已有邻居的向量 (RDMA)
// 3. 将邻居自身向量解码并上传到GPU作为查询
// 4. 编码新节点的存储向量
// 5. GPU计算所有候选的距离
// 6. CPU排序（按距离）
// 7. GPU RobustPrune选择R个
// 8. 写入剪枝结果
```
统计指标：`build_overflow_prunes`、`build_overflow_prune_candidates`

### 3. 细粒度并发控制
```
节点锁 (bit 0):  CAS compare(不含锁) → swap(含锁)
Medoid锁 (bit 8): 用于第一次插入时的race保护

锁操作:
- try_lock: RDMA CAS, 失败则自旋重试
- spinlock: 循环CAS直到成功
- unlock: RDMA WRITE清零锁位
```
每个锁操作都是独立的RDMA atomic操作

### 4. 插入时的性能细分
| 阶段 | 主要开销 |
|------|---------|
| Phase 0 | FAA + Write + CAS (3次RDMA) |
| Phase 1 | 多次RDMA读 + GPU距离计算 |
| Phase 2 | 批量RDMA读 + GPU RobustPrune kernel |
| Phase 3 | FAA + Write (2次RDMA) |
| Phase 5 | 对每个邻居: Read+Lock+Read+Write+Unlock (5+次RDMA) |

## 课后任务
1. 模拟一个插入操作，列出所有RDMA操作的顺序
2. 分析：为什么Overflow Prune比正常Prune开销大很多？
3. 思考：如何优化反向边更新阶段（Phase 5）的延迟？

## 参考文件
- `src/vamana/vamana_insert.ipp`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/vamana_rdma_writes.hh`
