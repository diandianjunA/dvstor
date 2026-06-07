# 第4课：Vamana节点与邻居列表的数据结构

## 学习目标
- 掌握`VamanaNode`的内存布局与序列化格式
- 理解`VamanaNeighborlist`的视图模式设计
- 熟悉节点锁、Medoid标志等并发控制机制

## 内容大纲

### 1. VamanaNode内存布局 (`src/vamana/vamana_node.hh`)
```
总大小: HEADER_SIZE(8B) + META_SIZE(8B) + vector_bytes + NEIGHBORS_SIZE(R*8B)

偏移量布局:
[0, 8):   header (u64)      — 锁标志 + medoid标志
[8, 12):  id (u32)          — 节点全局ID
[12, 13): edge_count (u8)   — 当前活跃邻居数
[13, 16): padding (3B)      — 对齐填充
[16, 16+vec_bytes): vector  — 向量数据（可能量化）
[16+vec_bytes, ...): neighbors — R个RemotePtr槽位
```

### 2. Header位标志
```cpp
HEADER_NODE_LOCK      = 0b01          // 节点锁（bit 0）
HEADER_MEDOID_LOCK    = 0b100000000   // Medoid锁（bit 8），用于CAS Race
HEADER_IS_MEDOID      = 0b10000000000000000  // 是否为Medoid（bit 16）
```
- **节点锁**: CAS操作设置，用于插入时的互斥
- **Medoid锁**: 用于Medoid指针交换时的CAS保护
- **Medoid标志**: 标识此节点为图的入口点

### 3. 静态存储初始化
```cpp
static void init_static_storage(u32 dim, u32 max_degree, VectorDType vector_dtype) {
    DIM = dim; R = max_degree; VECTOR_DTYPE = vector_dtype;
    VECTOR_COMPONENT_SIZE = vector_dtype_component_size(vector_dtype);
    VECTOR_BYTES = vector_dtype_bytes(vector_dtype, dim);
    NEIGHBORS_SIZE = max_degree * sizeof(u64);
}
```
设计要点：所有节点共享相同的维度、最大出度和数据类型——这些是类静态成员

### 4. VamanaNeighborlist (`src/vamana/vamana_neighborlist.hh`)
**视图模式**：不拥有内存，而是指向RDMA读取的缓冲区
```cpp
class VamanaNeighborlist {
    // 缓冲区布局: [edge_count(1B) | neighbor_ptrs(R*8B)]
    u8 num_neighbors() const;    // 读取活跃邻居数
    span<RemotePtr> view() const; // 活跃邻居视图（仅edge_count个）
    span<RemotePtr> all_slots() const; // 全部R个槽位
    void add(const RemotePtr&);  // 追加邻居
    void reset();                // 清空
};
```

### 5. RDMA分层读取策略
邻居列表的读取使用**两次RDMA读**（在`vamana_rdma_reads.hh`中）：
1. 读取`edge_count` (1B) + `neighbor_slots` (R*8B) 
   - 第一次读: edge_count (1B)
   - 第二次读: neighbors (R*8B)
   - 这种分离允许只读取需要的字节量

向量读取使用**单次RDMA读**，定位到`offset_vector()`偏移

### 6. 内存分配与释放
```cpp
// 节点析构时自动释放RDMA缓冲区
VamanaNode::~VamanaNode() {
    if (buffer_slice_ != nullptr) {
        owner_->buffer_allocator.free_buffer(buffer_slice_, buffer_size_);
    }
}
```
使用RAII模式：节点持有`buffer_slice_`，析构时归还给`BufferAllocator`

## 课后任务
1. 画一张VamanaNode的完整内存布局图（标注每个字段的偏移和大小）
2. 计算：对于dim=128, R=64, dtype=float32，一个节点占用多少字节？
3. 设计一种支持可变长度向量的节点布局方案

## 参考文件
- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_node.cc`
- `src/vamana/vamana_neighborlist.hh`
- `src/vamana/vamana_neighborlist.cc`
