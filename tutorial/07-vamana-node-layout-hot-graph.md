# 第 07 课：VamanaNode 存储布局与 hot graph

## 本课目标

本课讲清一个节点在远端内存中的字节布局，以及 AoS storage、compact storage、hot graph plane、RaBitQ entry 之间的关系。学完后，你应该能根据 `RemotePtr` 和 `VamanaNode` 静态参数手算 header、vector、neighbor、rabitq 的 offset。

## 代码证据

必须阅读：

- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_node.cc`
- `src/vamana/storage_format.hh`
- `src/vamana/hot_graph.hh`
- `src/vamana/storage_layout_resolver.hh`
- `tools/vamana_offline/shard_writer.cc`

## 两种 storage format

`StorageFormat` 有两种：

```cpp
enum class StorageFormat : u8 {
  aos_v1 = 1,
  compact_v1 = 2,
};
```

名称：

- `vamana_aos_v1`
- `vamana_compact_v1`

`aos_v1` 把 header、id、edge_count、neighbors、vector、RaBitQ 都放在固定节点记录里。

`compact_v1` 把固定节点中的 authoritative neighbor list 移到 hot graph plane 中，固定节点保留 header、id、generation、vector、可选 RaBitQ。

## AoS 节点布局

根据 `VamanaNode` 注释和 offset 函数，AoS 大致是：

```text
node base
  header: 8B
  id: 4B
  edge_count: 1B
  padding: 3B
  neighbors: R * 8B
  graph padding: align to 64B
  vector: vector_bytes, padded to 64B
  optional rabitq_code
  optional rabitq_norm: 4B
  optional rabitq_error: 4B
  rabitq padding: align to 64B
```

关键 offset：

- `offset_id() = HEADER_SIZE`
- `offset_edge_count() = HEADER_SIZE + ID_SIZE`
- `offset_neighbors() = NODE_PREFIX_SIZE`
- `graph_hot_bytes() = align_storage(offset_neighbors() + NEIGHBORS_SIZE)`
- `offset_vector() = graph_hot_bytes()`
- `offset_rabitq_code() = offset_vector() + vector_storage_bytes()`

## compact 节点布局

compact storage 的固定节点：

```text
node base
  header: 8B
  id: 4B
  generation: 4B
  vector: vector_bytes, align8
  optional rabitq entry
  align compact
```

neighbor list 不在 fixed node 的 `offset_neighbors()` 里，而在 hot graph plane。`StorageLayoutResolver::neighbor_read(ptr)` 会返回：

```text
if compact:
  offset = VamanaNode::hot_graph_entry_offset(ptr)
  size = VamanaNode::hot_graph_entry_size()
  compact = true
else:
  offset = ptr.byte_offset() + VamanaNode::neighbor_read_offset()
  size = VamanaNode::neighbor_read_size()
  compact = false
```

## header bit

`VamanaNode` header 是 64 位，目前关键 bit：

- `HEADER_NODE_LOCK`
- `HEADER_MEDOID_LOCK`
- `HEADER_IS_MEDOID`
- `HEADER_DELETED`

普通 compute-side 插入会通过 CAS 设置 `HEADER_NODE_LOCK`，通过 RDMA WRITE 清锁。delete/upsert 会设置 `HEADER_DELETED`。

## hot graph entry

`src/vamana/hot_graph.hh` 定义 hot graph compact pointer：

- `kCompactPointerBytes = 5`
- `kNodeBaseOffset = 16`
- `kNullCompactPointer = (1 << 40) - 1`

每个 neighbor 用 5 字节表示：

```text
[ shard bits | offset units ]
offset_units = byte_offset / 8
```

entry size：

```cpp
align8(8 + R * 5)
```

version 2 entry 中：

- byte 0: edge count
- byte 1: flags，比如 deleted
- byte 2..3: checksum16
- byte 4..7: generation
- byte 8..: neighbors

## hot graph 的读写转换

读 neighbor：

```text
RDMA READ hot graph entry
NeighborReadAwaitable::await_resume
  decode_hot_graph_entry(compact, neighbor_read_buffer)
  返回 VamanaNeighborlist
```

写 neighbor：

```text
write_vamana_neighbors
  if compact_storage:
    encode_hot_graph_entry
    RDMA WRITE hot graph entry
  else:
    RDMA WRITE edge_count
    RDMA WRITE neighbor slots
```

因此算法层继续看 `VamanaNeighborlist`，storage 层负责格式转换。

## RaBitQ entry

如果 `HAS_RABITQ_CODE` 为 true，节点固定记录后面还带：

- `rabitq_code`
- `rabitq_norm`
- `rabitq_error`

关键函数：

- `rabitq_code_bits()`: next power of two of `DIM`，至少 8。
- `rabitq_code_size()`
- `rabitq_entry_size()`
- `compute_rotated_query`
- `compute_rabitq_code`
- `compute_rabitq_entry`

注意：RaBitQ sidecar cache 和节点内 RaBitQ entry 不是同一个东西。sidecar 是查询 gate 使用的预算化 RFQ5 cache；节点内 entry 是节点布局的一部分。

## offline shard 写出中的布局

`tools/vamana_offline/shard_writer.cc` 写 shard 时：

1. 设置 `VamanaNode::STORAGE_FORMAT`。
2. 如果启用 RaBitQ，设置 centroid 并 `enable_rabitq`。
3. 计算 `node_size` 和 `aligned_size`。
4. 根据 partition 生成每个 node 的 `{memory_node, offset}`。
5. 每个 shard 文件 offset 0 写 shard size。
6. shard 0 offset 8 写 medoid `RemotePtr`。
7. 从 offset 16 开始写节点。
8. compact 模式下追加 hot graph header、entry region 和 dynamic base offset。
9. 写 metadata，记录所有关键 offset 和 size。

这就是在线 `load_index` 时校验 metadata 的依据。

## 字节级布局示例

假设：

- `R = 64`
- `dim = 128`
- `vector_dtype = float32`
- `vector_bytes = 512`

AoS 中：

```text
HEADER_SIZE = 8
META_SIZE = 8
NODE_PREFIX_SIZE = 16
NEIGHBORS_SIZE = 64 * 8 = 512
graph_hot_bytes = align64(16 + 512) = 576
offset_vector = 576
vector_storage_bytes = align64(512) = 512
```

不启用 RaBitQ 时 `total_size = align64(576 + 512) = 1088`。

compact 中 fixed node 会省掉 512B neighbor slots，但额外有 hot graph entry：

```text
hot_graph_entry_size = align8(8 + 64 * 5) = align8(328) = 328
```

所以 compact 主要减少远端 neighbor read 字节和 fixed node 大小，但引入 hot graph plane 的一致性和 checksum。

## 性能影响

- AoS neighbor read 大小是 `8 + R*8` 附近，R=64 时约 520 字节。
- compact hot graph neighbor read 大小是 `align8(8 + R*5)`，R=64 时约 328 字节。
- compact 减少 neighbor RDMA 字节，但 decode 和 checksum 增加 CPU 成本。
- vector offset 与 graph hot bytes 分离，使查询可以只读 vector，不必读整个节点。
- RaBitQ entry 增加节点大小，但可以减少 exact vector RDMA。

## 设计异味

1. `VamanaNode` 混合了布局计算、buffer view、RaBitQ 计算、hot graph 编解码。
2. 静态全局布局让多索引进程困难。
3. header bit 没有封装成强类型，容易写错 bit 操作。
4. compact hot graph metadata 校验分散在 compute 和 memory node 启动路径。
5. `VamanaNode` 析构依赖 `owner_->buffer_allocator`，这让 node view 和内存管理耦合。

## 可验证问题

- AoS 中 neighbor slots 的 offset 如何计算？
- compact 中 neighbor list 存在哪里？
- hot graph compact pointer 为什么要求 `byte_offset % 8 == 0`？
- metadata 中哪些字段用于验证运行时 layout？
- `HEADER_DELETED` 同时影响 fixed node 和 hot graph entry 吗？

## 学习任务

1. 手算你常用配置下的 `VamanaNode::total_size()`。
2. 找出所有调用 `configure_hot_graph` 的位置，解释每个位置的用途。
3. 画出 shard 文件 offset 0、8、16 之后的内容。
4. 对比 AoS 和 compact 模式下 `read_vamana_neighbors` 实际 RDMA READ 字节数。
5. 思考：如果拆分 `VamanaNode`，应该拆成 layout、view、codec、quantization 哪几类？

