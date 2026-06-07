# 第10课：内存管理与BufferAllocator

## 学习目标
- 理解Compute Node和Memory Node的内存管理策略
- 掌握`BufferAllocator`的bump分配器设计
- 理解大页内存的使用与RDMA注册

## 内容大纲

### 1. Compute Node内存架构

```
Compute Node 内存布局:
┌────────────────────────────────────────────┐
│         HugePage Buffer (60GB)             │
│  ┌──────────────────────────────────────┐  │
│  │    Bump Allocator (BufferAllocator)  │  │
│  │  ┌────┬────┬──────┬────┬──────┬───┐  │  │
│  │  │NODE│VEC │VEC   │PTR │NBRS  │...│  │  │
│  │  └────┴────┴──────┴────┴──────┴───┘  │  │
│  │  bump_pointer ──────────────→       │  │
│  │  FreeList: {size→[ptr1, ptr2, ...]} │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  PINNED Host Memory (per coroutine):        │
│  ┌──────────────────────────────────────┐  │
│  │ h_query | h_candidate_vecs | h_dists │  │
│  │ ... (cudaMallocHost)                 │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

### 2. BufferAllocator (`src/buffer_allocator.hh`)
```cpp
class BufferAllocator {
    // Bump分配器: 单调递增的指针
    byte_t* allocate(size_t size) {
        byte_t* ptr = buffer_ptr_ + bump_pointer_.fetch_add(align(size));
        return ptr;
    }

    // 释放: 放入size对应的freelist
    void free_buffer(byte_t* ptr, size_t size) {
        freelists_by_size_[align(size)].enqueue(ptr);
    }

    // 分配: 先查freelist，miss则bump allocate
    byte_t* allocate_buffer(size_t size) {
        if (auto it = freelists_by_size_.find(aligned_size); it != freelists_by_size_.end()) {
            if (it->second.try_dequeue(ptr)) return ptr;
        }
        return allocate(size);
    }
};
```

设计要点：
- **Bump分配器**: O(1)分配，无碎片问题
- **Freelist缓存**: 回收内存按大小分类，优先复用
- **缓存行对齐**: `align(size)` 向上取整到64B
- **无锁设计**: 使用`std::atomic` bump pointer + `concurrent_queue` freelist

### 3. Memory Node内存架构
```
Memory Node 内存布局 (HugePage):
┌────────────────────────────────────────────┐
│         Index Buffer (260GB max)           │
│  ┌──────────────────────────────────────┐  │
│  │ [0,8):  free_ptr (u64)       ← FAA  │  │
│  │ [8,16): medoid_ptr (u64)     ← CAS  │  │
│  │ [16,):  node_0 | node_1 | ...       │  │
│  │         ├─ header(8B)                │  │
│  │         ├─ id(4B)+edge_count(1B)+pad │  │
│  │         ├─ vector(dim*sizeof)        │  │
│  │         └─ neighbors(R*8B)           │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Peer Scratch Buffer (RDMA staging):        │
│  ┌──────────────────────────────────────┐  │
│  │ 用于跨Memory Node的RDMA读/写暂存    │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

### 4. 远程内存分配
```cpp
// Compute Node端调用:
RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);

// 底层操作: RDMA FAA on Memory Node's free_ptr
qp->post_FAA(local_addr, lkey, remote_mrt, 0/*offset=free_ptr*/, node_size, ...);
// 返回: RemotePtr{memory_node, old_free_ptr_value}
```
- 分配是跨网络的原子操作
- Compute Node直接通过RDMA FAA"抢"空间
- 无中心化分配器——每个Compute Node独立分配

### 5. 大页内存 (HugePage)
```cpp
template <typename T>
class HugePage {
    void allocate(size_t elements) {
        // mmap with MAP_HUGETLB
        // 减少TLB miss，提高RDMA访问性能
    }
    void touch_memory() {
        // 预触所有页面，确保物理内存分配
    }
};
```
- Memory Node用1GB大页（减少页表开销）
- Compute Node因大小原因可能用2MB大页

### 6. GPU内存管理 (`gpu_buffer_manager.cu`)
每个协程的GPU资源：
- **Device Memory**: d_query, d_candidate_vecs(×2双缓冲), d_candidate_dists, d_distances, d_pruned_indices, d_pruned_count
- **Pinned Host Memory**: 对应的h_* staging buffers (cudaMallocHost)
- **CUDA Stream + Event**: 每协程独立stream实现overlap
- **GPUDirect MR**: d_candidate_vecs注册为RDMA可访问

## 课后任务
1. 计算：dim=128, dtype=float32, R=64时，每个VamanaNode多少字节？
2. 分析：BufferAllocator中freelist vs bump allocate的性能权衡
3. 模拟：如果10个compute thread × 4个coroutines同时分配，Memory Node的FAA是否成为瓶颈？

## 参考文件
- `src/buffer_allocator.hh`
- `rdma-library/library/hugepage.hh`
- `src/gpu/gpu_buffer_manager.hh`、`gpu_buffer_manager.cu`
