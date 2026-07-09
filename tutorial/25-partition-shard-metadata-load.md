# 第 25 课：分区、shard 写出、metadata 与 index load

## 本课目标

本课学习离线构建结果如何变成 memory node 可加载的 shard 文件，以及 compute node 如何用 metadata 恢复 runtime layout。学完后，你需要能够：

1. 理解 balanced、BFS、METIS 三种分区策略的代码边界。
2. 解释 shard 文件开头、medoid、node 区、hot graph 区、RaBitQ sidecar、idmap sidecar、anchor sidecar 的关系。
3. 说明 metadata 字段如何被 runtime 校验。
4. 判断改 layout、改 metadata、改分区策略会影响哪些模块。

代码入口：

- `tools/vamana_offline/partitioning.hh`
- `tools/vamana_offline/partitioning.cc`
- `tools/vamana_offline/shard_writer.hh`
- `tools/vamana_offline/shard_writer.cc`
- `src/vamana/storage_layout_resolver.hh`
- `src/service/index_metadata.cc`
- `src/service/compute_service/index_commands.ipp`

## 1. NodePlacement

分区和写 shard 的核心输出是：

```cpp
struct NodePlacement {
  u32 memory_node;
  u64 offset;
};
```

`memory_node` 表示节点应该写入哪个 shard。

`offset` 表示该节点在 shard 文件中的 byte offset。

在线运行时，节点用 `RemotePtr{memory_node, offset}` 表示。也就是说，离线 placement 直接决定 runtime 远端指针。

## 2. offset 约定

分区代码中 shard offset 从 16 开始：

```cpp
vec<u64> shard_offsets(num_memory_nodes, 16);
```

这个约定来自 Vamana memory layout：

- offset 0：free pointer 或 shard size。
- offset 8：medoid pointer。
- offset 16：第一个节点起始位置。

offline shard writer 会：

1. 创建每个 shard 文件。
2. 在 offset 0 写 shard size。
3. 在 shard 0 的 offset 8 写 medoid pointer。
4. 从 offset 16 开始写 node。

memory node load 时会把 shard 文件加载到远端内存区域，因此这些 offset 会直接成为 RDMA 读写地址。

## 3. balanced placement

`assign_nodes_to_shards_balanced()` 的策略：

1. 每个 shard offset 初始为 16。
2. 遍历 node id。
3. 找当前 offset 最小的 shard。
4. 将 node 放入该 shard 当前 offset。
5. 该 shard offset 增加 aligned node size。

特点：

- 只平衡容量。
- 不考虑图边。
- 不考虑 query locality。
- 实现简单且 deterministic。

性能影响：

- 如果图边随机跨 shard，查询扩展会产生更多跨 memory node RDMA。
- 但每个 shard 大小较均衡，memory 使用均衡。

适合场景：

- 单 memory node。
- 测试基础流程。
- 对 partition locality 暂不敏感的实验。

## 4. BFS partition

`compute_bfs_partition()` 和 shard writer 中的 `compute_bfs_partition_graph()` 实现了多源 BFS 分区。

主要步骤：

1. 从 start node 选择第一个 seed。
2. 用 farthest-point heuristic 选择剩余 seeds。
3. 多源 BFS 扩张。
4. 如果 preferred shard 已达到 target size，则分配到当前最小 shard。
5. 对未访问的孤立节点，分配到最小 shard。
6. 统计 edge cut 和 cross-shard ratio。

BFS partition 的目标是保留图局部性：

- 相邻节点更可能分到同一 shard。
- 降低 active neighbor cross-shard ratio。
- 减少查询时跨 memory node neighbor/vector read。

但它也有风险：

- 对非连通图或弱连通图，seed 选择可能不稳定。
- load balancing 是粗粒度 target size，不保证严格均匀。
- 分区质量依赖图当前质量。

## 5. METIS partition

METIS 支持由 CMake 选项控制：

- `DVSTOR_METIS_PARTITION`
- 代码中宏 `DVSTOR_HAVE_METIS`

如果未编译 METIS，`metis_partitioning_available()` 返回 false，调用 `compute_metis_partition()` 会抛出 unavailable reason。

METIS partition 流程：

1. 收集 undirected packed edges：
   - `pack_undirected_edge(a, b)`
   - `append_partition_edges(...)`
2. 去除 0 edge。
3. sort + unique。
4. 构造 CSR：
   - `xadj`
   - `adjncy`
5. 调用 `METIS_PartGraphKway(...)`。
6. 返回 parts。
7. 统计 edge cut。

METIS 的目标是更系统地降低 edge cut，同时控制 imbalance。

风险：

- 编译期依赖。
- `idx_t` 容量限制。
- 大图 CSR 内存开销。
- METIS 输出不一定与 online workload 最优一致，因为 workload query 分布也重要。

## 6. shard writer 主流程

`write_vamana_shards(...)` 的主流程：

1. 解析 storage format。
2. 设置 `VamanaNode` storage format。
3. 如果 use RaBitQ：
   - 计算全局 centroid。
   - `VamanaNode::enable_rabitq()`
   - `set_rabitq_centroid(...)`
4. 如果 use RaBitQ cache sidecar：
   - 选择 sidecar entry bytes。
   - 计算 code bits。
   - 扫描 dataset 计算 norm min/max。
5. 计算 node size 和 aligned size。
6. `place_nodes(...)` 得到 placements 和 partition stats。
7. `write_anchor_sidecar(...)` 写 anchor。
8. 计算每个 shard size 和 entry count。
9. 如果 compact format：
   - 计算 hot graph header offset。
   - 计算 hot graph entry offset。
   - 计算 dynamic base offset。
   - `VamanaNode::configure_hot_graph(...)`
10. 创建 shard 文件并设定大小。
11. 打开 shard 文件和 RaBitQ sidecar 文件。
12. 写 shard size。
13. 写 hot graph header。
14. 写 RaBitQ sidecar header。
15. 在 shard 0 offset 8 写 medoid pointer。
16. 遍历每个 node 写 node data。
17. flush shard/cache。
18. 写 metadata json。
19. 写 owner idmap sidecar。

这条链路说明 offline writer 是 index format 的真实来源。runtime 只是按 metadata 和固定 layout 读取它。

## 7. node 写出内容

每个 node 的 buffer：

1. header：
   - 如果是 medoid，设置 `HEADER_IS_MEDOID`。
2. id：
   - 写入 `dataset.id(i)`。
3. edge count 或 generation：
   - AoS layout 写 edge count。
   - compact hot graph layout 写 generation。
4. vector：
   - `memcpy` raw dataset vector 到 `VamanaNode::offset_vector()`。
5. neighbors：
   - AoS layout 将 `RemotePtr.raw_address` 写入 node 内 neighbors 区。
   - compact layout 不写入 node 内 neighbors，而写 hot graph entry。
6. RaBitQ：
   - 写 code、norm、error 到 node 内对应 offset。
   - sidecar 额外写 RFQ cache entry。
7. compact hot graph：
   - 调用 `VamanaNode::encode_hot_graph_entry(...)`。
   - 写到 hot graph entry 区。

这说明 compact format 下，邻居热路径不再在 node 本体里，而是在独立 hot graph 区。在线读邻居时必须通过 `StorageLayoutResolver` 或 `VamanaNode` 的 hot graph offset 计算。

## 8. medoid pointer

writer 构造：

```cpp
const RemotePtr medoid_ptr{placements[graph.medoid].memory_node,
                           placements[graph.medoid].offset};
```

然后：

```cpp
shard_files[0].seekp(8);
write medoid_ptr.raw_address
```

注意 medoid pointer 总是写在 shard 0 offset 8，即使 medoid 节点本身可能在其他 memory node。

在线搜索时，compute node 读 medoid pointer 的路径也是从 memory node 0 offset 8 开始。

如果这里被破坏：

- search 初始化会失败或从错误节点开始。
- routing centroid 计算也会出错，因为它读 medoid probe。

## 9. RaBitQ sidecar

如果 `config.use_rabitq`：

writer 会为每个 shard 创建 RaBitQ cache sidecar：

- header：
  - entry size
  - code bits
  - node size
  - raw vector bytes
  - entry count
  - cache budget bytes
  - quantization
- entry：
  - 每个 node placement slot 对应一个 encoded cache entry。

metadata 也会写入：

- `rabitq_centroid`
- `rabitq_code_bits`
- `rabitq_entry_size`
- `rabitq_cache_bits`
- `rabitq_cache_entry_size`
- `rabitq_cache_norm_min/max`
- `rabitq_cache_error_min/max`

compute node 启动时如果 `use_rabitq`，会：

- 校验 index 是 RaBitQ layout。
- 加载 `vamana::rabitq::Cache`。
- 校验 cache ratio。
- 将 cache 指针设置给 Vamana。

所以 RaBitQ 是跨 writer、metadata、runtime search 三者的功能。

## 10. metadata 写出字段

writer 写 `.meta.json`，关键字段包括：

- `data_file`
- `output_prefix`
- `distance`
- `num_vectors`
- `dim`
- `R`
- `beam_width`
- `beam_width_construction`
- `alpha`
- `num_memory_nodes`
- `medoid`
- `node_size`
- `node_layout`
- `storage_format`
- `schema_version`
- `graph_hot_bytes`
- `vector_offset`
- `neighbors_offset`
- `rabitq_offset`
- `vector_storage_bytes`
- `offline_builder_version`
- `vector_data_type`
- `vector_component_size`
- `vector_bytes`
- `partition_strategy`
- `partition_max_degree`
- `partition_imbalance`
- `partition_edge_cut`
- `partition_cross_shard_ratio`
- `idmap_format`
- `anchor_format`
- `anchor_count_per_shard`

compact layout 额外包含：

- `hot_graph_neighbor_read_bytes`
- `hot_graph_neighbor_update_bytes`
- `hot_graph_entry_size`
- `hot_graph_pointer_bytes`
- `hot_graph_shard_bits`
- `hot_graph_offsets`
- `hot_graph_header_offsets`
- `hot_graph_entry_counts`
- `hot_graph_dynamic_base_offsets`
- `hot_graph_dynamic_record_bytes`
- `hot_graph_dynamic_hot_offset`
- `allocation_size`

这些字段不是“说明文档”，而是 runtime validation 的输入。

## 11. metadata load 与 validation

`src/service/index_metadata.cc` 使用 nlohmann json 读取字段，填充 `Metadata`。

`ComputeService::validate_index_metadata()` 会校验：

- schema version 必须是 13。
- storage format 可解析。
- dim、R、node size、vector bytes、offset 都匹配 runtime。
- memory node 数匹配当前连接。
- RaBitQ code layout 匹配 runtime dim。
- compact hot graph arrays 尺寸匹配 memory node 数。
- dynamic record 参数合法。
- construction beam width 如果非 0，必须匹配。

这说明 metadata 是运行时防错机制。改 writer 时必须改 reader/validator；改 runtime layout 时必须改 writer。

## 12. owner idmap sidecar

writer 最后写 owner idmap：

1. 为每个 owner 准备 `owner_entries`。
2. owner 计算方式是：

```cpp
owner = dataset.id(i) % num_memory_nodes
```

3. 每个 entry 包含：
   - id
   - RemotePtr raw address
   - flags
   - reserved
4. 每个 owner 写一个 `owner_idmap_file`。

运行时：

- compute-side idmap 用于非 storage-owner 模式的 upsert/delete。
- memory node storage-owner 也会加载 owner idmap，用于维护 freshness。

注意：owner shard 和 physical placement shard 可以不同。id 的 owner 由 id hash 决定，节点实际存放位置由 partition placement 决定。

这个分离是 storage-owner 模式的基础，也会增加跨 peer reverse update 的复杂度。

## 13. anchor sidecar

writer 调用 `write_anchor_sidecar(...)`。metadata 中：

- 如果 `anchor_count_per_shard == 0`，`anchor_format` 为空。
- 否则 `anchor_format = "owner_anchor_v1"`。

compute node 如果启用 storage-owner local-stitch，会要求：

- metadata anchor format 是 `owner_anchor_v1`。
- anchor index 能加载。

否则启动失败。

这说明 anchor sidecar 对 local-stitch 不是可选优化，而是必要依赖。

## 14. 性能影响

分区和 layout 影响 runtime 性能：

1. cross-shard ratio：
   - 越高，查询扩展越可能读多个 memory node。
   - 影响 QP 利用、RDMA latency、credit wait。

2. compact hot graph：
   - 邻居热路径读更小。
   - 但写入和动态更新更复杂。

3. vector dtype：
   - 决定 vector bytes。
   - 影响 RDMA vector read bytes、GPU kernel输入、cache ratio。

4. RaBitQ sidecar：
   - 可减少 exact vector read。
   - 但增加 CPU gate、cache 内存和 recall 风险。

5. shard size 平衡：
   - 影响 memory node 容量和负载。

6. idmap owner：
   - owner 与 placement 不一致时，更新可能走 peer RPC/RDMA。

## 15. 设计异味

1. schema version 硬编码为 13：
   - 没有 migration 层。

2. writer 与 runtime 都依赖 `VamanaNode` 静态状态：
   - layout 变更要跨模块同步。

3. metadata 字段很多但缺少强类型 schema：
   - 读取时很多字段用默认值。
   - 有些缺字段可能直到 runtime 才失败。

4. owner 由 `id % num_memory_nodes` 固定：
   - 对 skewed id 分布可能不均衡。

5. partition 只看图边：
   - 没有使用真实 query 分布。

6. shard writer 既做 partition、layout、sidecar、metadata：
   - 职责过宽。

## 16. 可验证问题

1. metadata 与 runtime dim 不一致：
   - compute node 是否拒绝 load。

2. metadata `num_memory_nodes` 与实际连接不一致：
   - 是否拒绝 load。

3. compact layout 缺少 hot graph arrays：
   - 是否拒绝。

4. shard 0 offset 8 medoid pointer 错误：
   - search 初始化表现如何。

5. BFS/METIS 分区：
   - `partition_cross_shard_ratio` 是否比 balanced 低。
   - runtime RDMA active nodes 是否下降。

6. owner idmap 缺失：
   - upsert/delete 是否降级或失败。

## 17. 学习任务

1. 画一张 shard 文件布局图：offset 0、8、16、node 区、hot graph header、hot graph entries、dynamic base。
2. 画一张 metadata 字段到 runtime 校验函数的映射表。
3. 用三种 partition strategy 构建同一小数据集，比较 metadata 中的 cross-shard ratio。
4. 设计一个测试：手动修改 metadata 的 `vector_bytes`，确认 compute node 拒绝 load。
5. 设计一个重构方案：将 shard writer 拆成 partitioner、layout encoder、sidecar writer、metadata writer 四个模块。

