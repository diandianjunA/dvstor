# 第19课：离线Vamana图构建器

## 学习目标
- 理解`vamana_offline_builder`的构建流程
- 掌握图分区策略（Balanced/BFS/METIS）
- 理解离线构建与在线加载的集成

## 内容大纲

### 1. 离线构建流程
```
main() in vamana_offline_builder.cc:
1. 解析配置 (VamanaBuildConfig)
2. 读取数据集 (Dataset)
3. GPU初始化（可选）
4. 构建Vamana图 (build_vamana_graph)
5. 可选: 与外部groundtruth比对recall
6. 写入Shard文件 (write_vamana_shards)
```

### 2. 图构建核心 (`graph.cc`)
```cpp
void build_vamana_graph(VamanaGraph& graph, const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         BuilderGpuContext* gpu_contexts, size_t num_gpu_contexts) {
    // 1. 初始化: 第一个点作为medoid
    // 2. 随机打乱其余点
    // 3. 依次插入每个点:
    //    a. Beam Search找到候选
    //    b. 如果有GPU: GPU计算距离, GPU RobustPrune
    //    c. 否则: CPU计算距离, CPU RobustPrune
    //    d. 更新图 (正向边 + 反向边)
    // 4. 多线程并行 (OpenMP或自定义线程池)
}
```

### 3. GPU上下文 (`BuilderGpuContext`)
```cpp
struct BuilderGpuContext {
    cudaStream_t stream;
    cudaEvent_t event;
    float* d_query;
    float* d_candidates;    // 可以复用d_base_vectors
    float* d_distances;
    uint32_t* d_candidate_ids;
    uint32_t* d_pruned_indices;
    uint32_t* d_pruned_count;
    // 如果全部向量在GPU: d_base_vectors (共享)
};
```
与在线GpuBufferManager的区别：
- 不需要双缓冲（无并发RDMA写入）
- 可能有d_base_vectors（全量数据集在GPU）

### 4. 图分区策略 (`partitioning.cc`)
```cpp
enum PartitionStrategy { balanced, bfs, metis };

// Balanced: 按节点ID均匀分配
partition_balanced(graph, num_shards, ...)
    → shard[i] = nodes[i*N/k : (i+1)*N/k]

// BFS: 从Medoid出发BFS遍历，连续BFS区间分配到一个shard
partition_bfs(graph, num_shards, ...)
    → 图相邻的节点更可能在同一个shard

// METIS: 使用METIS图分区库
partition_metis(graph, num_shards, max_degree, imbalance, ...)
    → 最小化跨shard的边（精确图分区）
```

### 5. Shard文件格式 (`shard_writer.cc`)
```
输出文件:
  dvstor_index_node{1..N}_ofN.dat  — 每个shard的二进制数据
  dvstor_index.meta.json            — 元数据
  dvstor_index.rotation.bin        — RaBitQ旋转矩阵（可选）

Shard .dat文件格式:
  [free_ptr(8B)] [medoid_ptr(8B)] [VamanaNode...]
  (与Memory Node内存布局完全一致)
```

### 6. 在线加载
```
Memory Node启动时:
  ./dvstor_memory_node --server-index-file shard_1_of_N.dat

或Compute Node触发加载:
  ./dvstor --load-index --index-prefix /path/to/dvstor_index
  → 发送MN_COMMAND_LOAD_INDEX到所有Memory Node
```

`load_index_file()` 使用`mmap` + `vmtouch`将shard文件映射到HugePage区域

### 7. 关键配置参数
```
--data-path:        数据集路径
--memory-nodes:     shard数量
--partition-strategy: balanced|bfs|metis
--threads:          构建线程数
--R:                最大出度
--beam-width-construction: 构建时束宽
--alpha:            RobustPrune α参数
--rabitq-bits:      RaBitQ量化位数
--output-prefix:    输出文件前缀
```

## 课后任务
1. 用一个小数据集（如SIFT10K）运行离线构建器并检查shard文件
2. 对比三种分区策略的跨shard边比例
3. 分析：离线构建vs在线构建的性能差异来源

## 参考文件
- `tools/vamana_offline_builder.cc`
- `tools/vamana_offline/config.hh`
- `tools/vamana_offline/graph.hh`、`graph.cc`
- `tools/vamana_offline/partitioning.hh`、`partitioning.cc`
- `tools/vamana_offline/shard_writer.hh`、`shard_writer.cc`
