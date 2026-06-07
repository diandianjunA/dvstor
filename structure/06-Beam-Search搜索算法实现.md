# 第6课：Beam Search搜索算法实现

## 学习目标
- 逐行理解`vamana_search.ipp`中Beam Search的完整实现
- 掌握RDMA批量读取+GPU距离计算的交互流程
- 理解Direct to GPU路径与Host Staging回退路径

## 内容大纲

### 1. 搜索入口
```cpp
VamanaCoroutine knn(node_t q_id, const span<element_t> components, thread) {
    // 1. 读取Medoid入口指针 (RDMA read, 8B)
    // 2. 读取Medoid节点 (RDMA read, header+vector)
    // 3. 上传查询向量到GPU (H2D transfer)
    // 4. 初始化Beam: beam = [(medoid, dist(q, medoid), false)]
    // 5. 进入Beam Search主循环
}
```

### 2. Beam Search主循环分解

#### Step 1: 选择最佳未扩展节点 (CPU)
```cpp
i32 best_idx = -1;
for (i32 i = 0; i < beam.size(); ++i) {
    if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
    }
}
if (best_idx < 0) break;  // 所有节点已扩展
```

#### Step 2: 读取邻居列表 (RDMA)
```cpp
auto nlist = co_await rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, thread);
// 两次RDMA读: edge_count(1B) + neighbor_slots(R*8B)
```

#### Step 3: 过滤重复访问 (CPU)
```cpp
for (const RemotePtr& n_ptr : nlist->view()) {
    if (!visited.contains(n_ptr)) {
        visited.insert(n_ptr);
        unvisited.push_back(n_ptr);
    }
}
```

#### Step 4: 批量读取向量 (RDMA) — 两条路径

**路径A: GPUDirect RDMA（若可用）**
```cpp
// 直接用RDMA写到GPU内存
auto vec_read = co_await rdma::vamana::batch_read_vectors(
    unvisited, thread, gs.d_candidate_vecs, gs.d_candidate_vecs_lkey);
// 优点: 零拷贝，减少PCIe流量
```

**路径B: Host Staging（回退方案）**
```cpp
auto vec_read = co_await rdma::vamana::batch_read_vectors(unvisited, thread);
// 读取到host pinned memory → cudaMemcpyAsync到GPU
// 优点: 兼容所有硬件
```

**路径C: 间接指针路径（GPUDirect + 非连续向量）**
```cpp
// 每个向量写入独立的GPU位置，记录指针表
// GPU kernel通过指针表间接访问向量
```

#### Step 5: GPU距离计算
```cpp
gpu::launch_batch_typed_query_l2_distances(stream, event,
    d_query, query_dtype, d_candidate_vecs, candidate_dtype, d_distances, n_batch, dim);
co_await gpu::GpuAwaitable{thread.get()};  // 挂起等待GPU完成
```

#### Step 6: 距离回传 (D2H) + Beam更新
```cpp
cudaMemcpyAsync(h_distances, d_distances, n_batch*sizeof(float), D2H, stream);
cudaStreamSynchronize(stream);  // 确保距离数据就绪
for (u32 i = 0; i < n_batch; ++i) {
    insert_into_beam(beam, unvisited[i], h_distances[i], beam_width_);
}
```

### 3. 搜索结果收集
```cpp
std::sort(beam.begin(), beam.end(), ...);  // 按距离排序
for (u32 i = 0; i < k; ++i) {
    node_t id = co_await rdma::vamana::read_vamana_id(beam[i].rptr, thread);
    results.push_back({id, beam[i].distance});
}
```

### 4. 性能分解跟踪
每个步骤都有nanosecond级别的计时：
- `rdma_medoid_ptr`: 读取Medoid指针的RDMA延迟
- `cpu_node_read`: CPU处理节点数据的开销
- `rdma_neighbor_fetch`: 读取邻居列表的RDMA延迟
- `rdma_vector_fetch`: 批量读取向量的RDMA延迟
- `gpu_query_distance`: GPU距离计算延迟
- `cpu_query_beam_update`: Beam插入排序开销

## 课后任务
1. 画一张时序图，展示一次搜索中的RDMA和GPU操作交替
2. 分析：n_batch从1到beam_width变化时，各步骤的延迟占比如何变化
3. 解释间接指针路径存在的场景和原因

## 参考文件
- `src/vamana/vamana_search.ipp`
- `src/rdma/vamana_rdma_reads.hh`
- `src/gpu/gpu_kernel_launcher.hh`
