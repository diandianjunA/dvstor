# 第 03 课：公共类型、ID、指针和向量 dtype

## 本课目标

本课建立阅读整个项目必须掌握的基础词典：节点 ID、距离类型、远程指针、向量存储类型、距离计算函数。后续所有 RDMA、Vamana、storage-owner、metadata、RaBitQ 逻辑都依赖这些基础类型。

## 代码证据

必须阅读：

- `src/common/types.hh`
- `src/common/constants.hh`
- `src/common/vector_dtype.hh`
- `src/common/distance.hh`
- `src/remote_pointer.hh`
- `src/vamana/vamana_node.hh`

## 核心类型词典

`src/common/types.hh` 定义：

| 类型 | 实际类型 | 语义 |
| --- | --- | --- |
| `node_t` | `u32` | 业务层向量 ID 或图节点 ID |
| `element_t` | `f32` | compute API 中 float 查询/插入向量元素 |
| `distance_t` | `f32` | 距离值 |
| `filepath_t` | `std::filesystem::path` | 路径 |
| `hashset_t<T>` | `std::unordered_set<T>` | 当前默认 hash set |
| `hashmap_t<K,V>` | `std::unordered_map<K,V>` | 当前默认 hash map |

这几个别名很简单，但要注意：`node_t` 和 `RemotePtr` 是两个不同层面的标识。

- `node_t` 是逻辑 ID。
- `RemotePtr` 是物理位置。

同一个 `node_t` 在 upsert 后可能对应新的 `RemotePtr`，旧的 `RemotePtr` 被标记 deleted。

## RemotePtr 编码

`src/remote_pointer.hh` 定义：

```text
raw_address: u64
[ memory node: 16 bits | byte offset: 48 bits ]
```

核心方法：

- `memory_node()`: `raw_address >> 48`
- `byte_offset()`: 低 48 位
- `is_null()`: `raw_address == 0`
- `store_address(memory_node, byte_offset)`

这表示项目把 shard ID 和远端内存 offset 压成一个 64 位值。几乎所有图边、medoid、idmap、hot graph、storage-owner response 都用这个物理指针。

## RemotePtr 的读法

当你看到：

```cpp
RemotePtr{2, 4096}
```

它表示：

```text
memory node 2 的 RDMA index region 中，从 byte offset 4096 开始的一条 VamanaNode 记录
```

当你看到：

```cpp
rptr.raw_address
```

它常常是在写 wire format、metadata sidecar 或 RDMA payload。

## RemotePtr 的风险

`RemotePtr` 没有内置边界检查。边界检查分散在：

- `StorageLayoutResolver::ptr_in_bounds`
- `MemoryNode::read_node_snapshot`
- `MemoryNode::remote_read_bytes`
- `rdma::vamana::allocate_vamana_node` 的 `MEMORY_NODE_MAX_MEMORY` assert
- metadata 和 `VamanaNode::total_size()` 校验

后续做重构时，一个方向是引入更明确的 typed address 或 region capability，减少到处手算 offset。

## 向量 dtype

`VectorDType` 支持：

```cpp
enum class VectorDType : u32 {
  float32 = 0,
  uint8 = 1,
  int8 = 2,
};
```

关键函数：

- `vector_dtype_name`
- `parse_vector_dtype`
- `infer_vector_dtype_from_path`
- `resolve_vector_dtype_config`
- `vector_dtype_component_size`
- `vector_dtype_bytes`
- `vector_component_as_float`
- `encode_float_vector_to_storage`
- `decode_storage_vector_to_float`

在线 API 插入通常接收 `vec<element_t>`，也就是 float。写入节点时会调用 `encode_float_vector_to_storage` 转成 `VamanaNode::vector_dtype()` 对应的存储格式。

查询路径也分两类：

- `search(const vec<element_t>& query, u32 k)`：float query。
- `search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k)`：raw dtype query。

## dtype 对距离计算的影响

CPU 距离：

- `L2Distance::dist`
- `IPDistance::dist`
- `typed_l2_distance`
- `typed_ip_distance`
- `typed_distance_float_query`

GPU 距离：

- `launch_batch_typed_query_l2_distances`
- `launch_batch_typed_multi_query_l2_distances`
- `launch_batch_typed_query_l2_distances_indirect`

你要注意项目中有两套距离语义：

1. `Distance` 模板参数决定高层使用 L2 还是 IP。
2. GPU kernel 当前重点实现 typed L2 路径，在线 Vamana 查询和插入中大量调用 typed L2 launcher。

如果要系统支持 IP 距离，需要检查所有 GPU 路径、offline builder、metadata、benchmark 是否一致。

## 数据从 API 到存储的转换

普通插入路径中：

```text
ComputeService::insert
  InsertItem.values: vec<float>
  InsertRequest.components: vec<float>
  Vamana::insert
  rdma::vamana::write_vamana_node
  encode_float_vector_to_storage
  写入 VamanaNode::offset_vector()
```

storage-owner 插入路径中：

```text
ComputeService::post_storage_owner_batch
  encode_float_vector_to_storage
  request payload vectors
MemoryNode::process_storage_owner_insert_tasks
  decode_storage_vector_to_float
  execute storage-owner update
```

这说明 storage-owner wire payload 发送的是存储 dtype，不是 float API dtype。memory node 为了本地 distance 又会 decode 成 float。

## VamanaNode 静态 dtype 状态

`VamanaNode` 中有静态变量：

- `DIM`
- `R`
- `VECTOR_DTYPE`
- `VECTOR_COMPONENT_SIZE`
- `VECTOR_BYTES`
- `STORAGE_FORMAT`

这些通过 `VamanaNode::init_static_storage(dim, R, vector_dtype)` 设置。

这带来一个重要事实：同一进程内默认只能安全服务一种维度、度数和向量存储类型。如果未来想一个进程加载多个索引，这套静态状态会成为主要障碍。

## RemotePtr 到节点字段的解析

不要直接手写 offset。项目引入了 `StorageLayoutResolver`：

- `header(ptr)`
- `id(ptr)`
- `generation(ptr)`
- `edge_count(ptr)`
- `vector(ptr)`
- `rabitq(ptr)`
- `neighbor_read(ptr)`
- `neighbor_slots(ptr)`

正确思路是：

```text
RemotePtr
  -> StorageLayoutResolver
  -> VamanaNode static layout
  -> RDMA offset and size
```

这样可以同时支持 AoS 和 compact storage。

## 性能影响

基础类型会直接影响性能：

- `node_t = u32` 限制单个逻辑 ID 空间，但减少 wire payload 和 idmap 大小。
- `RemotePtr = u64` 让 neighbor list 每条边固定 8 字节。
- hot graph compact pointer 使用 5 字节编码，减少邻居读取字节数。
- `uint8/int8` 存储 dtype 减少 vector RDMA 字节，但会增加 decode 或 typed kernel 复杂度。
- `std::unordered_set` 用于 visited set，查询热路径可能有分配和 cache miss 开销。

## 设计异味

1. `RemotePtr` 是裸物理地址，没有类型区分。medoid pointer、node pointer、anchor pointer 都是同一个类型。
2. `VamanaNode` 的静态布局状态让多索引、多维度、多 dtype 进程很难实现。
3. `hashset_t` 和 `hashmap_t` 注释里已经提示需要替换为更快实现，但目前全局使用标准容器。
4. dtype 编码和距离计算散在 `vector_dtype.hh`、`distance.hh`、GPU launcher、VamanaNode RaBitQ 中。
5. `RemotePtr::is_null` 把 raw 0 当空指针，意味着 memory node 0 offset 0 不能作为合法节点地址。项目实际把 offset 0/8 作为 header slots，节点从 offset 16 开始，刚好避开。

## 可验证问题

- `RemotePtr` 最多能表达多少 memory node？
- 为什么节点从 offset 16 开始分配？
- `node_t` 和 `RemotePtr` 是否一一对应？
- upsert 后旧 `RemotePtr` 怎么处理？
- `vector_data_type=auto` 是如何从路径后缀推断 dtype 的？

## 学习任务

1. 在代码中搜索 `raw_address`，列出所有直接序列化 `RemotePtr` 的位置。
2. 在代码中搜索 `VamanaNode::init_static_storage`，列出所有设置静态布局的入口。
3. 画一张 `node_t -> idmap -> RemotePtr -> VamanaNode` 的关系图。
4. 手算一个 `RemotePtr{3, 1024}` 的 `raw_address` 编码。
5. 思考：如果把 `RemotePtr` 改成带 region 类型的结构，哪些文件会首先受影响？

