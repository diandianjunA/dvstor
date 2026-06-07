# 第16课：GPU内核实现——RobustPrune

## 学习目标
- 理解RobustPrune的GPU并行化策略
- 掌握单block + 共享内存的实现方案
- 理解typed版本的多种精度支持

## 内容大纲

### 1. RobustPrune的并行挑战
```
算法本质: 顺序依赖
  for v in sorted_candidates:
      for s in selected:
          if α*dist(source, v) > dist(s, v): reject v

挑战:
  - outer loop有数据依赖（选中的v影响后续判断）
  - inner loop的dist(s,v)可以并行计算
  - 需要动态大小的selected集合
```

### 2. GPU实现策略：单Block + 共享内存
```cpp
size_t smem_size = n_candidates * sizeof(bool);  // 每个候选的selected标志
uint32_t block_size = std::min(BLOCK_SIZE, n_candidates);

gpu_kernels::robust_prune_kernel
    <<<1, block_size, smem_size, stream>>>(...);
```

设计决策：
- **单个Block**: 保证顺序执行outer loop（利用`__syncthreads()`）
- **共享内存selected[]**: 线程可访问所有已选候选的标志
- **每线程一个候选**: `threadIdx.x` 对应候选索引

### 3. 内核伪代码
```cuda
__global__ void robust_prune_kernel(...) {
    extern __shared__ bool selected[];  // 共享内存

    // 初始化
    if (threadIdx.x == 0) *d_pruned_count = 0;
    for (int i = threadIdx.x; i < n_candidates; i += blockDim.x)
        selected[i] = false;

    __syncthreads();

    // 按距离顺序处理候选（由d_candidate_order决定顺序，或dists已排好序）
    for (int rank = 0; rank < n_candidates && *d_pruned_count < R; rank++) {
        int idx = d_candidate_order ? d_candidate_order[rank] : rank;

        bool keep = true;
        // 检查是否被任何已选候选"覆盖"
        for (int s = 0; s < n_candidates; s++) {
            if (selected[s] && threadIdx.x == idx) {
                float alpha_d = alpha * d_candidate_dists[idx];
                float d_s_v = compute_distance(candidate_vecs[s], candidate_vecs[idx]);
                if (alpha_d > d_s_v) keep = false;
            }
        }
        __syncthreads();

        if (keep && threadIdx.x == idx) {
            selected[idx] = true;
            int slot = atomicAdd(d_pruned_count, 1);
            d_pruned_indices[slot] = idx;
        }
        __syncthreads();
    }
}
```

### 4. Typed版本
```cpp
template <typename T>  // T = uint8_t 或 int8_t
__global__ void robust_prune_typed_kernel(
    const T* d_candidate_vecs,  // 量化存储的向量
    const float* d_candidate_dists,
    const uint32_t* d_candidate_order,
    uint32_t n_candidates, uint32_t dim,
    float alpha, uint32_t R,
    uint32_t* d_pruned_indices, uint32_t* d_pruned_count)
```
支持直接在量化向量上计算距离，避免反量化开销

### 5. 内核启动器
```cpp
void launch_robust_prune_typed(stream, event, d_candidate_vecs,
                                candidate_dtype, d_candidate_dists,
                                d_candidate_order, n_candidates, dim,
                                alpha, R, d_pruned_indices, d_pruned_count) {
    if (candidate_dtype == 0) {
        // float → 使用原始内核
        launch_robust_prune(stream, event, ...);
    } else if (candidate_dtype == 1) {
        robust_prune_typed_kernel<uint8_t><<<1, block_size, smem, stream>>>(...);
    } else if (candidate_dtype == 2) {
        robust_prune_typed_kernel<int8_t><<<1, block_size, smem, stream>>>(...);
    }
}
```

### 6. 性能特性
- **共享内存瓶颈**: 需要`n_candidates * sizeof(bool)` SMEM，限制候选数上限
- **Block Size自适应**: `std::min(BLOCK_SIZE, n_candidates)`，小候选集用更少线程
- **原子操作**: `atomicAdd(d_pruned_count)` 控制输出写入位置

## 课后任务
1. 分析：为什么RobustPrune只用1个block？多个block的可行性？
2. 计算：SMEM=48KB时，最大支持的candidate数量
3. 思考：如何改进内核以支持更大的候选集？

## 参考文件
- `src/gpu/gpu_kernel_launcher.cu`（`launch_robust_prune`相关函数）
- `src/gpu/kernels/distance_kernels.cuh`（如果包含prune内核）
