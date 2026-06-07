# 第15课：GPU内核实现——距离计算

## 学习目标
- 理解CUDA distance kernels的模板化设计
- 掌握Typed Pair Distance的多种精度组合
- 理解间接指针路径的GPU实现

## 内容大纲

### 1. 内核文件组织
```
src/gpu/
├── gpu_kernel_launcher.hh     ← 主机端声明（C++编译器）
├── gpu_kernel_launcher.cu     ← 主机端实现 + 内核启动（nvcc）
└── kernels/
    └── distance_kernels.cuh   ← GPU内核定义（nvcc，.cuh文件）
```

### 2. 模板化内核设计
```cuda
template <uint32_t TILE_SIZE, typename QueryT, typename CandidateT>
__global__ void batch_l2_typed_pair_distance_kernel(
    const QueryT* __restrict__ query,
    const CandidateT* __restrict__ candidates,
    float* __restrict__ distances,
    uint32_t n_candidates, uint32_t dim)
{
    // 每个线程处理TILE_SIZE个元素
    // 使用cooperative groups进行warp级规约
}
```
支持9种类型组合 (QueryT × CandidateT ∈ {float, uint8_t, int8_t})

### 3. 内核启动配置
```cpp
constexpr uint32_t TILE_SIZE = 4;      // 每线程处理4个元素
constexpr uint32_t BLOCK_SIZE = 512;   // 每block 512线程

uint32_t total_threads = n_candidates * TILE_SIZE;
uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(...);
```

### 4. 直接路径 vs 间接路径

**直接路径** (`batch_l2_typed_pair_distance`):
```cuda
// 候选向量连续存储在d_candidates中
// candidates[i] 位于 d_candidates + i*dim*sizeof(CandidateT)
```

**间接路径** (`batch_l2_typed_pair_distance_indirect`):
```cuda
// 候选向量通过指针表间接访问
// candidates[i] 位于 d_candidate_ptrs[i]
// 用于GPUDirect RDMA写入非连续地址的场景
```

### 5. ID距离计算 (`batch_l2_id_distance_kernel`)
```cuda
// d_base_vectors: 完整数据集在GPU上（离线构建场景）
// 通过candidate_id索引基向量
template <typename T>
__global__ void batch_l2_id_distance_kernel(
    const T* d_base_vectors, uint32_t query_id,
    const uint32_t* d_candidate_ids, float* d_distances, ...)
{
    // query = d_base_vectors[query_id]
    // for each candidate: dist(query, d_base_vectors[candidate_ids[i]])
}
```
用于离线构建时所有向量都在GPU上的场景

### 6. 内核启动器中的类型分发
```cpp
void launch_batch_typed_query_l2_distances(stream, event, d_query, query_dtype,
                                            d_candidates, candidate_dtype,
                                            d_distances, n_candidates, dim) {
    // float×float → 快速路径
    if (query_dtype == 0 && candidate_dtype == 0) {
        launch_batch_l2_distances(stream, event, ...);
        return;
    }
    // 否则根据类型组合分发
    if (query_dtype == 0 && candidate_dtype == 1) {
        launch_typed_pair_distance<float, uint8_t>(...);
    } else if (...) { ... }
}
```

### 7. GPU事件同步
```cpp
// 每次内核启动后记录事件
CUDA_CHECK(cudaEventRecord(event, stream));

// 调度器轮询
cudaError_t err = cudaEventQuery(event);
if (err == cudaSuccess) {
    --gpu_post_balances[coro_id];  // GPU完成
}
```
事件用于异步完成通知，不阻塞CPU

## 课后任务
1. 阅读`kernels/distance_kernels.cuh`（如果存在），画出GPU线程映射图
2. 分析：TILE_SIZE=4 vs TILE_SIZE=8对SM占用率的影响
3. 思考：为什么float×float有快速路径而其他类型组合没有？

## 参考文件
- `src/gpu/gpu_kernel_launcher.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/kernels/distance_kernels.cuh`
