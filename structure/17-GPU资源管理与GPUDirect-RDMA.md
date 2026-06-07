# 第17课：GPU资源管理与GPUDirect RDMA

## 学习目标
- 掌握`GpuBufferManager`的完整资源管理
- 理解GPUDirect RDMA的注册与使用流程
- 理解双缓冲避免流水线停顿

## 内容大纲

### 1. GpuBufferManager资源清单
每个协程分配的资源：
```
Per-Coroutine GPU Resources:
  CUDA Stream         ×1  (cudaStreamNonBlocking)
  CUDA Event          ×1  (cudaEventDisableTiming)
  Pinned Host Memory:
    h_query           [query_vector_bytes]
    h_candidate_vecs  [max_batch × candidate_vector_bytes]
    h_candidate_dists [max_batch × float]
    h_candidate_order [max_batch × uint32_t]
    h_distances       [max_batch × float]
    h_candidate_ptrs  [max_batch × void*]
    h_pruned_indices  [max_R × uint32_t]
    h_pruned_count    [1 × uint32_t]
  Device Memory:
    d_query, d_candidate_vecs, d_candidate_vecs_alt (双缓冲!),
    d_candidate_dists, d_candidate_order, d_distances,
    d_candidate_ptrs, d_pruned_indices, d_pruned_count
  GPUDirect RDMA MR (可选):
    d_candidate_vecs_mr, d_candidate_vecs_alt_mr
```

### 2. 初始化流程
```cpp
void GpuBufferManager::init(num_coroutines, dim, max_batch, max_R,
                             query_vector_bytes, candidate_vector_bytes,
                             rdma_pd, enable_gpudirect_rdma) {
    for each coroutine:
        cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming);
        cudaMallocHost(...) × 8;  // pinned host memory
        cudaMalloc(...) × 9;       // device memory

    if (try_gpudirect_rdma):
        for each coroutine:
            ibv_reg_mr(pd, d_candidate_vecs, bytes, IBV_ACCESS_LOCAL_WRITE);
            ibv_reg_mr(pd, d_candidate_vecs_alt, bytes, IBV_ACCESS_LOCAL_WRITE);
        gpudirect_candidate_ready_ = true;
}
```

### 3. GPUDirect RDMA详解
```
传统路径:
  RDMA NIC → Host Memory (PCIe) → GPU Memory (PCIe)  [两次PCIe穿越]

GPUDirect路径:
  RDMA NIC → GPU Memory (PCIe P2P)  [一次PCIe穿越]
```

注册要求：
- CUDA Context必须已创建
- 需要`nvidia-peermem`内核模块（跨PCIe root complex时）
- GPU内存必须通过`ibv_reg_mr`注册到RDMA PD
- 注册标志: `IBV_ACCESS_LOCAL_WRITE`

### 4. 双缓冲设计
```cpp
// 查询中交替使用两个候选向量缓冲区
d_candidate_vecs      // buffer 0
d_candidate_vecs_alt  // buffer 1

void flip_query_candidate_buffer() {
    query_candidate_buffer_index ^= 1u;
}

uint8_t* current_query_candidate_vecs() const {
    return index == 0 ? d_candidate_vecs : d_candidate_vecs_alt;
}
```
**目的**: 避免一个查询的RDMA写入与下一个查询的GPU Kernel读取冲突

### 5. 间接指针路径
当GPUDirect可用但向量不连续时：
```cpp
// 每个向量独立RDMA到GPU的不同位置
for each unvisited node:
    staging_ptr = current_query_candidate_vecs() + i * vec_bytes;
    h_candidate_ptrs[i] = staging_ptr;     // 记录GPU地址
    destinations.push_back({reinterpret_cast<u64>(staging_ptr), lkey, ...});

// 批量RDMA到各个独立位置
batch_read_vectors(unvisited, thread, &destinations);

// 上传指针表到GPU
cudaMemcpyAsync(d_candidate_ptrs, h_candidate_ptrs, n_batch*sizeof(void*), H2D, stream);

// GPU kernel通过指针间接访问向量
launch_batch_typed_query_l2_distances_indirect(stream, event, d_query, query_dtype,
    d_candidate_ptrs, candidate_dtype, d_distances, n_batch, dim);
```

### 6. 销毁与清理
```cpp
void GpuBufferManager::destroy() {
    for each coroutine:
        cudaFree(d_*); cudaFreeHost(h_*);
        if (d_candidate_vecs_mr) ibv_dereg_mr(...);
        cudaEventDestroy(event); cudaStreamDestroy(stream);
    delete[] states_;
}
```

## 课后任务
1. 验证：使用`nvidia-smi topo -m`查看GPU与NIC的PCIe拓扑
2. 实验：比较GPUDirect vs Host Staging的吞吐量差异
3. 分析：双缓冲设计为什么只需要2个buffer而不是N个？

## 参考文件
- `src/gpu/gpu_buffer_manager.hh`
- `src/gpu/gpu_buffer_manager.cu`
- `src/gpu/gpu_awaitable.hh`
- `src/gpu/compute_thread_gpu.cc`
