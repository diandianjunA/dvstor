# 第 29 课 离线索引构建与迁移

> 课号 29 / 30 · 课题：离线索引构建与迁移
> 涉及代码：`tools/vamana_offline/{graph.cc,graph.hh,shard_writer.cc,shard_writer.hh,pq_indexer.cc,pq_indexer.hh,partitioning.cc,partitioning.hh,anchor_builder.cc,anchor_builder.hh,dataset_io.cc,dataset_io.hh,config.cc,config.hh,recall_check.cc,recall_check.hh,progress.hh,progress.cc}`、`tools/vamana_offline_builder.cc`、`tools/vamana_pq_indexer.cc`、`tools/vamana_anchor_sidecar_builder.cc`、`tools/legacy_index/migrator.cc`、`tools/legacy_index/migrator.hh`、`tools/gpunetio_probe.cc`、`tools/gpunetio_loopback_probe.cc`，辅以 `src/common/index_path.hh`、`src/vamana/hot_graph.hh`、`src/vamana/vamana_node.hh`、`src/gpu_search/index_format.hh`。

## 29.1 本课目标与涉及文件

dvstor 是一个 GPU 中心化的存算分离向量检索系统。线上跑的是**持久化 kernel**（见第 17–21 课）和**存储节点**（见第 23–26 课），它们读取的是一份"已经构图、已经分区、已经 PQ 编码、已经写好 anchor"的 schema-15 索引（见第 7 课）。这份索引从哪里来？就是本课讲的**离线工具链**。

离线工具不跑在 GPU 上，不进存储节点进程，而是一组独立的可执行文件：读原始向量文件（`.fbin`/`.u8bin`/`.i8bin`），在 CPU 上建 Vamana compact 图，按内存节点分区，写出 schema-14 的 `.dat`/`.idmap`/`.anchors`/`.meta.json`，再训练 OPQ/PQ 生成 `.pq16` 模型和 `.pq16.codes` 码流，把元数据升级到 schema-15。对历史遗留的 schema-13 索引，另有一个 migrator 做只读字段压缩迁移。最后还有两个 GPUNetIO probe 工具，用来在真正部署前确认硬件链路（GPUDirect RDMA + DOCA GPUNetIO）可用。

本课按"构图 → 分区 → PQ sidecar → anchor → 迁移 → probe 工具"分组讲解，每组讲清 `main` 流程和关键函数。涉及文件按职责分组如下：

| 分组 | 文件 | 行数 | 职责 |
| --- | --- | --- | --- |
| 入口 | `tools/vamana_offline_builder.cc` | ~57 | 构建主流程：读数据 → 建图 → 写 shard |
| 入口 | `tools/vamana_pq_indexer.cc` | ~56 | PQ 训练/编码/布局升级 CLI |
| 入口 | `tools/vamana_anchor_sidecar_builder.cc` | ~269 | 独立 anchor sidecar 重建 CLI |
| 配置/IO | `tools/vamana_offline/config.{cc,hh}` | ~93/~33 | `VamanaBuildConfig` + boost::program_options |
| 配置/IO | `tools/vamana_offline/dataset_io.{cc,hh}` | ~140/~31 | 数据集读取与距离函数 |
| 配置/IO | `tools/vamana_offline/progress.{hh,cc}` | ~135/~26 | `ProgressReporter` + `parallel_for` |
| 构图 | `tools/vamana_offline/graph.{cc,hh}` | ~564/~60 | `VamanaGraph`、`beam_search`、`robust_prune`、`build_vamana_graph` |
| 分区 | `tools/vamana_offline/partitioning.{cc,hh}` | ~404/~50 | balanced/bfs/metis 三种分区 |
| 写 shard | `tools/vamana_offline/shard_writer.{cc,hh}` | ~496/~14 | 写 `.dat`/`.idmap`/`.meta.json` + hot graph 表 |
| PQ | `tools/vamana_offline/pq_indexer.{cc,hh}` | ~539/~34 | OPQ/PQ 训练、编码、布局升级 |
| anchor | `tools/vamana_offline/anchor_builder.{cc,hh}` | ~128/~15 | 内嵌的 anchor sidecar 写出 |
| recall | `tools/vamana_offline/recall_check.{cc,hh}` | ~115/~13 | 离线 recall 校验 |
| 迁移 | `tools/legacy_index/migrator.{cc,hh}` | ~549/~24 | schema-13 → schema-14 只读迁移 |
| probe | `tools/gpunetio_probe.cc` | ~200 | DOCA GPUNetIO UMEM/MR 注册探针 |
| probe | `tools/gpunetio_loopback_probe.cc` | ~191 | GPUNetIO kernel 端 loopback 读探针 |

读完后你应当能回答：

- 一份原始 `.fbin` 数据集，经过哪些步骤变成存储节点可加载的 schema-15 索引？每一步的输出是什么文件？
- Vamana 图是怎么建的？为什么 `graph.cc` 里 `build_alpha = 1.0f` 而不是用配置里的 `alpha`？
- 三种分区策略（balanced/bfs/metis）各自怎么选 shard？METIS 没链接进来会怎样？
- PQ sidecar 的 `code_bytes`、`remote_offset`、`dynamic_record_bytes` 是怎么算出来的？为什么要在 metadata 里存这些偏移？
- `vamana_pq_indexer` 的 `--upgrade-layout-only` 是干什么的？为什么不重新编码也能升级 schema？
- schema-13 → schema-14 的迁移为什么是"只读"的？`translate_pointer` 在做什么？
- `gpunetio_probe` 和 `gpunetio_loopback_probe` 测的是不是同一件事？它们和第 22 课的 transport probe 有什么区别？

## 29.2 离线构建流水线总览

先把整条流水线画出来，后面每节展开一个环节。

```
                       ┌──────────────────────────┐
   base.fbin/u8bin ──▶ │ read_dataset (dataset_io)│ ──▶ Dataset{raw_vectors,dim,dtype}
                       └──────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────┐
                │ build_vamana_graph (graph.cc)       │
                │   compute_medoid                    │
                │   init random R-regular graph       │
                │   per-node: beam_search + robust_prune │
                │   consolidate_reverse_edges         │
                │   in-memory recall sanity check     │
                └────────────────────────────────────┘
                                    │
                                    ▼  (VamanaGraph: neighbors[], degrees[])
                ┌────────────────────────────────────┐
                │ run_optional_recall_check          │  （query_path/groundtruth_path）
                └────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────┐
                │ write_vamana_shards (shard_writer) │
                │   place_nodes: balanced/bfs/metis  │
                │   write_anchor_sidecar (内嵌)      │
                │   写 .dat (含 hot_graph 表)        │
                │   写 .idmap (owner 分片)           │
                │   写 .meta.json (schema_version=14)│
                └────────────────────────────────────┘
                                    │
                                    ▼  schema-14 plain compact L2 索引
                ┌────────────────────────────────────┐
                │ build_pq_index (pq_indexer.cc)     │  CLI: vamana_pq_indexer
                │   sample_training_vectors          │
                │   train_model: OPQ + PQ32 (Faiss)  │
                │   encode_shard: 写 .pq16.codes     │
                │   make_persistent_layout           │
                │   apply_persistent_layout → schema-15 │
                │   synthesize_distributed_view      │
                └────────────────────────────────────┘
                                    │
                                    ▼  schema-15 opq_pq_graph_v1 索引
                ┌────────────────────────────────────┐
                │ (可选) vamana_anchor_sidecar_builder│  独立重建 .anchors
                └────────────────────────────────────┘
                                    │
                                    ▼
                          存储节点加载（见第 23 课）
```

迁移流程则是另一条独立链路：

```
   schema-13 索引 (rabitq, 含 RaBitQ payload)
            │
            ▼
   migrate_schema13_index (legacy_index/migrator.cc)
     ├─ parse_layout: 校验 schema-13 + 计算新布局
     ├─ translate_pointer: 旧 RemotePtr → 新 RemotePtr
     ├─ migrate_shard: 压缩 fixed record + 重写 hot graph + 校验 checksum
     ├─ migrate_idmap: 重写 owner idmap 里的 rptr
     ├─ migrate_anchors: 重写 anchor 里的 rptr
     ├─ write_metadata: 抹掉 rabitq 字段 → schema-14
     └─ publish_outputs: 原子 rename 全部临时文件
            │
            ▼
   schema-14 plain compact L2 索引（再走 PQ 流程升级到 schema-15）
```

## 29.3 入口 `vamana_offline_builder.cc`：构建主流程

`tools/vamana_offline_builder.cc:17-57` 是整条构建链的总入口。`main` 非常薄，只做四件事：

```cpp
int main(int argc, char** argv) {
  const VamanaBuildConfig config = parse_configuration(argc, argv);
  const Dataset dataset = read_dataset(config);
  const filepath_t output_prefix =
      config.output_prefix.empty()
          ? default_vamana_prefix(dataset.source_file, config.R, config.beam_width)
          : config.output_prefix;
  // ...
  VamanaNode::init_static_storage(dataset.dim, config.R, dataset.dtype);
  VamanaGraph graph;
  build_vamana_graph(graph, dataset, config);
  run_optional_recall_check(graph, dataset, config);
  write_vamana_shards(graph, dataset, config, output_prefix);
  // ...
}
```

几个要点：

- `output_prefix` 默认是 `<dataset_dir>/dump/vamana_R{R}_bw{beam_width}`（`config.cc:14-17` 的 `default_vamana_prefix`）。所有 shard 文件名都由这个前缀加后缀生成，后缀规则在 `src/common/index_path.hh` 里集中定义：`_node{i}_of{n}.dat`、`_node{i}_of{n}.idmap`、`.anchors`、`.pq{subquantizers}`、`_node{i}_of{n}.pq{subquantizers}.codes`、`.meta.json`。
- `VamanaNode::init_static_storage` 把 `dim/R/dtype` 注入到 `VamanaNode` 的静态布局（见第 7 课），后续 `VamanaNode::total_size()` / `offset_vector()` / `hot_graph_entry_size()` 等都依赖这一步。离线工具和线上存储节点共享同一个 `VamanaNode` 头，保证布局一致。
- 流程严格按"建图 → 可选 recall → 写 shard"三步，中间没有任何持久化中间产物。`VamanaGraph` 完全在内存里（见 29.4），写 shard 时才一次性落盘。
- `offline distance execution: cpu-avx2`（`vamana_offline_builder.cc:41`）是一句日志，提示距离计算走的是 `dataset_io.cc` 里的标量 L2（uint8/int8 走整数差平方和，float32 走 `typed_l2_distance`），并没有真的链接 AVX2 intrinsics——这是历史日志，保留作占位。

## 29.4 构图：`graph.cc` / `graph.hh`

这是整个离线工具链最重的一块。`VamanaGraph` 是一个 CSR 风格的紧凑图结构，`build_vamana_graph` 是 Vamana 算法的 CPU 实现。

### 29.4.1 `VamanaGraph` 数据结构

`graph.hh:12-33`：

```cpp
struct VamanaGraph {
  static constexpr u32 kEmptyNeighbor = std::numeric_limits<u32>::max();
  size_t num_nodes{0};
  u32 dim{0};
  u32 R{0};
  size_t medoid{0};
  vec<u32> neighbors;
  vec<u8> degrees;
  std::unique_ptr<std::atomic_flag[]> lock_stripes;
  size_t lock_stripe_count{0};
  void init(size_t n, u32 d, u32 max_degree, size_t requested_lock_stripes = 1 << 20);
  size_t offset(size_t node) const { return node * static_cast<size_t>(R); }
  u8 degree(size_t node) const { return degrees[node]; }
  // ...
};
```

- `neighbors` 是一个 `n*R` 的扁平数组，`offset(node) = node*R`。每个节点预留 `R` 个 `u32` 邻居槽，实际有效邻居数存在 `degrees[node]`（`u8`，所以 `R <= 255`，`init` 里有 `lib_assert(max_degree <= std::numeric_limits<u8>::max())`）。
- `kEmptyNeighbor = 0xffffffff` 是空槽标记，和 `RemotePtr` 的 null 编码一致（见第 6 课）。
- `lock_stripes` 是一个分段自旋锁数组，`init` 里把 `requested_lock_stripes` 向上取整到 2 的幂（`graph.cc:112-118`）。构图是多线程的，每次改某个节点的邻居前要 `lock_node`，避免 `set_neighbors` 和 `try_append_neighbor_unlocked` 竞争。`NodeLockGuard`（`graph.hh:35-40`）是 RAII 包装。
- `init` 还会打一行内存占用日志（`graph.cc:119-121`），方便估计算 1 亿向量需要多少 RAM。

### 29.4.2 `LocalIdSet`：自定义开放寻址哈希集合

`graph.cc:19-66` 是一个内联的 `LocalIdSet`，开放寻址 + 线性探测：

```cpp
explicit LocalIdSet(size_t expected_items) {
  size_t capacity = 1;
  while (capacity < expected_items * 2) capacity <<= 1;
  table_.assign(capacity, kEmpty);
  mask_ = capacity - 1;
}
```

容量取 ≥ `2*expected_items` 的最小 2 的幂，装载因子 ≤ 0.5。`kEmpty = 0xffffffff`。哈希函数用的是 SplitMix64 风格（`graph.cc:54-62`）。`beam_search` 会高频调用 `visited.insert` / `expanded.contains`，`std::unordered_set<u32>` 太慢，这个手写集合是为了让构图阶段跑得动。

### 29.4.3 `compute_medoid`：选入口点

`graph.cc:181-211`。medoid 是"离质心最近的实际向量"。算法是：

1. 抽样最多 10000 个点（`n <= 10000` 时全量），算算术平均得到 `centroid`（float32，不管原始 dtype 是什么，都走 `dataset_decode_vector` 解码到 float 再累加）。
2. 全量扫描 `n` 个点，找离 `centroid` L2 最近的那个，返回其下标。

注意 `dataset_distance_float_query(dataset, centroid.data(), i)` —— 这里 query 是 float 数组，rhs 是存储格式（可能是 uint8/int8），距离函数在 `dataset_io.cc:25-32` 的 `l2_float_query_to_raw` 里逐分量 `vector_component_as_float` 做差平方和。medoid 选好后存在 `graph.medoid`，后续所有 `beam_search` 都从它出发。

### 29.4.4 `beam_search` 与 `beam_search_float_query`

`graph.cc:213-264` 是构图用的 beam search（query 是数据集里某个向量的下标），`graph.cc:266-306` 是 recall 校验用的版本（query 是任意 float 数组）。两者结构一致：

```cpp
vec<std::pair<float, u32>> beam_search(VamanaGraph& graph, const Dataset& dataset,
                                       u32 query_id, u32 beam_width) {
  vec<std::pair<float, u32>> all_visited;
  vec<std::pair<float, u32>> beam;
  LocalIdSet visited(...);
  LocalIdSet expanded(...);
  const float medoid_dist = dataset_distance(dataset, query_id, graph.medoid);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  all_visited.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  vec<u32> nbrs;
  vec<u32> unvisited;
  while (true) {
    ssize_t best_pos = -1;
    for (size_t i = 0; i < beam.size(); ++i) {
      if (!expanded.contains(beam[i].second)) { best_pos = ...; break; }
    }
    if (best_pos < 0) break;
    const u32 best_node = beam[best_pos].second;
    expanded.insert(best_node);
    {
      NodeLockGuard lock(graph, best_node);
      graph.copy_neighbors(best_node, nbrs);
    }
    unvisited.clear();
    for (u32 nbr : nbrs) {
      if (!visited.insert(nbr)) continue;
      unvisited.push_back(nbr);
    }
    for (u32 nbr : unvisited) {
      const float d = dataset_distance(dataset, query_id, nbr);
      insert_sorted_beam(beam, {d, nbr}, beam_width);
      all_visited.push_back({d, nbr});
    }
  }
  std::sort(all_visited.begin(), all_visited.end(), candidate_id_less);
  return all_visited;
}
```

要点：

- `beam` 始终是按距离升序排列的"待展开候选"，`insert_sorted_beam`（`graph.cc:93-100`）用 `lower_bound` 插入并截断到 `beam_width`。
- `expanded` 记录"已经拉过邻居的节点"，避免重复展开；`visited` 记录"已经算过距离的节点"，避免重复算距离。这两个集合都是 `LocalIdSet`。
- 拷贝邻居时持 `NodeLockGuard`，因为构图阶段别的线程可能在 `set_neighbors`。
- 返回的是 `all_visited`（所有算过距离的点，按 `(dist, id)` 升序），不是 `beam`。这是 Vamana 论文里 `search` 的输出，接下来交给 `robust_prune` 做剪枝。

### 29.4.5 `robust_prune`：RobustPrune 剪枝

`graph.cc:308-330` 是 Vamana 的核心剪枝逻辑：

```cpp
vec<u32> robust_prune(const Dataset& dataset, u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha, u32 R) {
  vec<u32> selected;
  selected.reserve(R);
  for (const auto& [cand_dist, cand_id] : sorted_candidates) {
    if (cand_id == source) continue;
    if (selected.size() >= R) break;
    bool pruned = false;
    for (u32 sel_id : selected) {
      const float d_sel_cand = dataset_distance(dataset, sel_id, cand_id);
      if (alpha * d_sel_cand <= cand_dist) { pruned = true; break; }
    }
    if (!pruned) selected.push_back(cand_id);
  }
  return selected;
}
```

输入 `sorted_candidates` 必须已按 `(dist_to_source, id)` 升序排好。遍历每个候选 `c`：如果存在已选中的 `p`，使得 `alpha * d(p, c) <= d(source, c)`，就剪掉 `c`（因为从 source 出发，先到 p 再到 c 不会比直接到 c 远太多，c 这条边冗余）。`alpha >= 1` 时剪得更狠，图更稀疏；`alpha = 1` 退化为标准剪枝。

**关键细节**：`build_vamana_graph` 里 `build_alpha = 1.0f` 是写死的（`graph.cc:478`），即使配置里 `alpha > 1` 也只用 1.0 构图，配置值只存进 metadata：

```cpp
const float build_alpha = 1.0f;
if (alpha > 1.0f + 1e-6f) {
  std::cerr << "note: using alpha=1.0 for construction (config alpha="
            << alpha << " stored in metadata)\n";
}
```

这是和原版 Vamana 的一个偏差：本系统构图阶段固定用 `alpha=1`，把 `alpha>1` 留给后续可能的在线增量阶段。构图用 alpha=1 能得到更稠密的图，配合后面的 `consolidate_reverse_edges` 反向边整合，保证连通性。

### 29.4.6 `build_vamana_graph`：完整构图主循环

`graph.cc:417-562`。分四步：

**第 1 步：初始化**（`graph.cc:426-433`）。`graph.init(n, dim, R)` 分配 `neighbors`/`degrees`/`lock_stripes`，算 medoid，生成随机访问顺序 `order`（`std::shuffle`，seed 来自 `config.seed`，`-1` 表示用 `random_device`）。

**第 2 步：随机 R-regular 初始图**（`graph.cc:438-476`）。每个节点 `i` 用一个 per-node PRNG（`mix_seed(seed ^ i)` 派生，`graph.cc:439-441`）随机选 `R` 个不重复邻居。这里有个完整性检查（`graph.cc:457-476`）：抽样 4096 个节点，对它们的邻居列表算 FNV 风格的签名，要求唯一签名比例 ≥ 99%，否则 `lib_assert` 失败——防止 PRNG 退化导致初始图高度同质。

**第 3 步：主循环 + 反向边维护**（`graph.cc:485-513`）。对每个节点 `node_idx = order[step]`：

1. `beam_search(graph, dataset, node_idx, beam_width)` 找到候选集。
2. 把 `node_idx` 当前已有的邻居也加进候选（`graph.cc:491-498`），`sort_and_unique_candidates` 去重排序。
3. `robust_prune(..., build_alpha, R)` 剪枝得到 `new_neighbors`，持锁 `set_neighbors`。
4. 对每个新邻居 `nbr`，持 `nbr` 的锁 `try_append_neighbor_unlocked(nbr, node_idx)`——把反向边"便宜地"追加进去（如果 `nbr` 的度已满 `R` 或已存在该边则跳过）。

这就是 `graph.cc:483` 那行日志说的"reverse edge maintenance: cheap append plus bulk consolidation"——反向边先懒追加，等所有节点处理完再批量整合。

**第 4 步：批量反向边整合 `consolidate_reverse_edges`**（`graph.cc:332-415`）。这是构图质量的关键。流程：

1. 两遍 `parallel_for` 构建 reverse-edge CSR：先 `incoming_counts[target]++`，再前缀和得到 `incoming_offsets`，再第二遍把每条边 `(source, target)` 写进 `incoming_edges[incoming_offsets[target] + slot]`。
2. 对每个节点 `node`：把它当前的出边 + 所有入边合并成候选集，去重排序，`robust_prune(..., build_alpha, graph.R)` 重新剪枝，`set_neighbors` 覆盖。这一步保证图的"对称性"——如果 A 选了 B 做邻居，B 在整合时也会把 A 纳入候选，最终双向边都会被 RobustPrune 重新评估。

**第 5 步：统计 + 可选 sanity check**（`graph.cc:517-561`）。打平均/最大/最小度。如果没传 `--skip-sanity-check`，随机抽 200 个 query 做 brute-force top-10，再用 `beam_search` 跑一遍算 in-memory recall@10，打日志。这是个朴素但有用的健全性检查——如果构图逻辑坏了，这里 recall 会很低。

### 29.4.7 `consolidate_reverse_edges` 的内存账

`graph.cc:361-364` 会打：

```
bulk reverse consolidation memory: incoming_edges=... edge_bytes=... offset_bytes=... count_bytes=...
```

`incoming_edges` 是 `vec<u32>`，大小等于总边数（每条有向边一条记录）。对 1 亿向量、R=64 的图，这是约 `1e8 * 64 * 4 = 25.6 GB` 的临时内存——这是离线工具 RAM 消耗的大头之一，规划机器时要留够。

## 29.5 分区：`partitioning.cc` / `partitioning.hh`

构图完成后，要把每个节点分配到一个 `memory_node`（shard）和一个 `offset`（在该 shard 文件里的字节偏移）。`NodePlacement`（`partitioning.hh:10-13`）就是 `{memory_node, offset}`。三种策略在 `shard_writer.cc:205-237` 的 `place_nodes` 里分发。

### 29.5.1 `PartitionOptions` / `PartitionStats`

`partitioning.hh:15-27`：

```cpp
struct PartitionOptions {
  u32 num_parts{1};
  u32 max_degree{16};
  double imbalance{1.03};
};
struct PartitionStats {
  size_t input_edges{0};
  size_t unique_edges{0};
  size_t edge_cut{0};
  double partition_cross_shard_ratio{0.0};
  vec<size_t> part_node_counts;
};
```

`max_degree` 是 METIS 输入图的截断度数（每个节点最多取前 `max_degree` 个邻居喂给 METIS，避免图太密）。`imbalance` 是 METIS 的 `ubvec`，1.03 表示允许 3% 的不均衡。`PartitionStats` 里的 `edge_cut` 和 `partition_cross_shard_ratio` 是分区质量指标——跨 shard 边越多，查询时跨节点 RDMA 读越多。

### 29.5.2 `balanced`：最小堆贪心

`partitioning.cc:208-225` 的 `assign_nodes_to_shards_balanced`：

```cpp
vec<u64> shard_offsets(num_memory_nodes, 16);
vec<NodePlacement> placements(num_vectors);
for (size_t i = 0; i < num_vectors; ++i) {
  const auto min_it = std::min_element(shard_offsets.begin(), shard_offsets.end());
  const u32 shard = static_cast<u32>(std::distance(shard_offsets.begin(), min_it));
  placements[i] = {shard, *min_it};
  *min_it += aligned_node_size;
}
```

每个 shard 起始偏移 16（`kNodeBaseOffset`，留给文件头的 size+medoid 两个字段），每次把当前节点放进"最空"的 shard。这是 O(n·m) 的（m = shard 数），但因为 m 通常很小（几个到十几个），实际很快。它完全不考虑图拓扑，跨 shard 边比例会接近 `1 - 1/m`，但对一些均匀分布的数据集已经够用。

`shard_writer.cc:27-31` 的 `assign_nodes_to_shards` 是它的薄包装，用 `VamanaNode::total_size()` 算 `node_size` 并 8 字节对齐。

### 29.5.3 `bfs`：多源 BFS + 负载均衡

`partitioning.cc:227-364`（也内联在 `shard_writer.cc:47-187` 里有一个直接吃 `VamanaGraph` 的版本 `compute_bfs_partition_graph`，逻辑一致）。三步：

**Step 1：farthest-point 选种子**（`partitioning.cc:257-295`）。从 `start_node`（通常是 medoid）出发 BFS 算距离，然后贪心选"离已选种子最远"的节点作为下一个种子，重复 `num_parts` 次。这是 k-center greedy 的标准做法，保证种子在图上分散。

**Step 2：多源 BFS + 负载均衡**（`partitioning.cc:297-332`）。一个 FIFO 队列，每项是 `(node, preferred_shard)`。弹出时如果 `preferred_shard` 已经满了（`>= target = n/num_parts`），就改派到当前最空的 shard。这样既保证拓扑连续（邻居尽量同 shard），又保证负载均衡。

**Step 3：兜底**（`partitioning.cc:334-342`）。BFS 没到达的孤立节点（极少见）派到最空的 shard。

**Step 4：算统计**（`partitioning.cc:344-361`）。遍历所有边，数跨 shard 边比例。

BFS 策略不需要外部库，跨 shard 比例通常比 balanced 显著低，是默认推荐。

### 29.5.4 `metis`：METIS k-way 划分

`partitioning.cc:101-206`。这是质量最高但需要外部依赖的方案。

**METIS 链接校验**（`partitioning.cc:9-19, 70-80`）：

```cpp
#ifndef DVSTOR_HAVE_METIS
#define DVSTOR_HAVE_METIS 0
#endif
#if DVSTOR_HAVE_METIS
#define idx_t metis_idx_t
#define real_t metis_real_t
#include <metis.h>
#undef real_t
#undef idx_t
#endif
```

`DVSTOR_HAVE_METIS` 由 CMake 控制（`-DDVSTOR_METIS_PARTITION=ON` 且找到 libmetis 才置 1）。`metis_partitioning_available()` 返回它，`metis_unavailable_reason()` 给出安装提示。如果用户传 `--partition-strategy=metis` 但没链接 METIS，`compute_metis_partition` 会在 `partitioning.cc:119-121` 抛 `"METIS support is not built..."`。

**边打包**（`partitioning.cc:82-99`）：`pack_undirected_edge(a, b)` 把无向边规范化成 `(min, max)` 并打包成 `u64`，`a == b` 返回 0（自环丢弃）。`append_partition_edges` 对每个节点的邻居截断到 `max_degree` 后打包。

**预处理**（`partitioning.cc:128-134`）：去零、排序、去重，得到无向边集合。`stats->unique_edges` 在这里更新。

**idx_t 容量校验**（`partitioning.cc:24-33, 136-137`）：`align_checked_idx` 检查节点数和邻接表条数不超过 `metis_idx_t` 上限。如果 METIS 用的是 32 位 `idx_t`，超过 2^31-1 个节点会直接报错，提示"rebuild METIS with 64-bit idx_t"。

**构建 CSR**（`partitioning.cc:139-161`）：把无向边转成 METIS 需要的 `xadj`/`adjncy` CSR。每条无向边 `(u,v)` 在 `adjncy` 里出现两次（`v` 在 `u` 的邻接表，`u` 在 `v` 的邻接表）。

**调用 METIS**（`partitioning.cc:163-188`）：

```cpp
metis_idx_t nvtxs = static_cast<metis_idx_t>(num_nodes);
metis_idx_t ncon = 1;
metis_idx_t nparts = static_cast<metis_idx_t>(options.num_parts);
metis_idx_t objval = 0;
metis_real_t ubvec = static_cast<metis_real_t>(options.imbalance);
vec<metis_idx_t> part(num_nodes, 0);
metis_idx_t metis_options[METIS_NOPTIONS];
METIS_SetDefaultOptions(metis_options);
metis_options[METIS_OPTION_NUMBERING] = 0;  // C 风格 0-based
const int rc = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                   nullptr, nullptr, nullptr,
                                   &nparts, nullptr, &ubvec,
                                   metis_options, &objval, part.data());
```

`METIS_OPTION_NUMBERING = 0` 是关键——METIS 默认是 Fortran 风格 1-based，这里显式改成 C 风格 0-based，和我们的节点下标一致。`objval` 返回的是跨分区边数（cut edge count），直接存进 `stats->edge_cut`。

`assign_nodes_to_shards_from_partition`（`partitioning.cc:366-383`）把 `parts[]` 转成 `NodePlacement[]`：每个 shard 维护一个游标 `shard_offsets[shard]`（起始 16），按节点顺序依次分配。

### 29.5.5 `place_nodes` 分发与跨 shard 比例

`shard_writer.cc:205-237` 的 `place_nodes` 根据 `config.partition_strategy` 选上面三种之一，最后调 `compute_cross_shard_ratio_graph`（`shard_writer.cc:189-203`）算"实际跨 shard 边比例"——注意这和 `PartitionStats::partition_cross_shard_ratio` 不完全一样：METIS 那个是无向去重边，这里是有向边遍历。`print_partition_stats`（`shard_writer.cc:239-256`）把两者都打出来，便于判断分区质量。

## 29.6 写 shard：`shard_writer.cc`

`write_vamana_shards`（`shard_writer.cc:270-493`）是 schema-14 plain compact L2 索引的落盘函数。它要把内存里的 `VamanaGraph` + `Dataset` 转成一组存储节点可以直接 mmap 的二进制文件。

### 29.6.1 布局规划

`shard_writer.cc:276-307` 先算每个 shard 文件的总大小和三段布局：

```
┌──────────────────────────────────────────────────────────┐
│ [0, 16)            file header: size(u64) + medoid(u64)  │
│ [16, static_end)   fixed VamanaNode records              │
│ [static_end, ...)  hot_graph::Header + hot graph entries │
│ [hg_end, ...)      (预留 dynamic region base)            │
└──────────────────────────────────────────────────────────┘
```

- `node_size = VamanaNode::total_size()`，`aligned_size = (node_size + 7) & ~7`（8 字节对齐）。
- `shard_sizes[shard]` 初始为 16，遍历 placements 取每个 shard 的最大 `offset + aligned_size`。
- `shard_entry_counts[shard]`：`(offset - 16) / aligned_size + 1`，即该 shard 的节点数。
- hot graph 段：`hot_graph_header_offsets[shard] = align_storage(shard_sizes[shard])`，`hot_graph_entry_offsets[shard] = align_storage(header_offset + sizeof(Header))`，`hot_graph_dynamic_base_offsets[shard] = align_storage(entry_offset + entry_count * entry_size)`。最后把 `shard_sizes` 抬到 dynamic base，给后续 schema-15 的 PQ 码区/control 区留位置。

`VamanaNode::configure_hot_graph`（`shard_writer.cc:308-314`）把这些偏移注入 `VamanaNode` 静态状态，后面 `encode_hot_graph_entry` 要用 `hot_graph_shard_bits` 编码邻居指针。

`hot_graph_shard_bits = vamana::hot_graph::shard_bits_for(num_memory_nodes)`（`hot_graph.hh:48`），决定 compact pointer 里 shard ID 占几个 bit。

### 29.6.2 创建文件 + 写文件头与 hot graph header

`shard_writer.cc:319-342`：`create_sized_file`（`shard_writer.cc:258-266`）用 `seekp(size-1); put(0)` 的经典手法预分配文件大小（sparse file）。然后每个 shard 写两段：

- 文件头 `[0, 16)`：`shard_sizes[shard]`（u64）写在 `[0,8)`；`[8,16)` 留给 medoid 指针，但只 shard 0 写（`shard_writer.cc:345-346`）。
- hot graph header：`vamana::hot_graph::Header`（`hot_graph.hh`），字段包括 `version=kVersion2`、`entry_bytes`、`max_degree=R`、`compact_pointer_shard_bits`、`entry_count`、`reserved0=dynamic_base`、`reserved1=allocation_size`、`reserved2=total_size`。

### 29.6.3 写每个节点

`shard_writer.cc:352-393` 是主循环。对每个节点 `i`：

```cpp
u64 header = 0;
if (i == graph.medoid) header |= VamanaNode::HEADER_IS_MEDOID;
*reinterpret_cast<u64*>(buf) = header;
*reinterpret_cast<u32*>(buf + VamanaNode::HEADER_SIZE) = dataset.id(i);
graph.copy_neighbors(i, nbrs);
const u8 edge_count = static_cast<u8>(std::min<size_t>(nbrs.size(), config.R));
*reinterpret_cast<u32*>(buf + VamanaNode::offset_generation()) = 0;
std::memcpy(buf + VamanaNode::offset_vector(), dataset.raw_vector(i), dataset.vector_bytes);
```

- `header` 里只标了 `HEADER_IS_MEDOID`（medoid 节点），其余 bit 留给运行时。
- `dataset.id(i)`：注意 `dataset.id(i) == i`（`dataset_io.hh:17`），离线构建时 ID 就是行号。
- `generation = 0`：离线索引的初始 generation，运行时存储节点会递增。
- 原始向量按 `offset_vector` 偏移拷进去，dtype 保持存储格式（uint8/int8/float32）。

接着把邻居编码成 `RemotePtr` 数组并写 hot graph entry：

```cpp
std::fill(hot_neighbors.begin(), hot_neighbors.end(), RemotePtr{});
for (u8 j = 0; j < edge_count; ++j) {
  const u32 nbr = nbrs[j];
  RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
  hot_neighbors[j] = nbr_ptr;
}
// 写 fixed record
file.seekp(placement.offset); file.write(node_buf...);
// 编码并写 hot graph entry
VamanaNode::encode_hot_graph_entry(hot_graph_entry.data(), edge_count,
                                   hot_neighbors.data(), edge_count,
                                   hot_graph_shard_bits, 0);
const u64 slot = (placement.offset - 16) / aligned_size;
const u64 hot_offset = hot_graph_entry_offsets[placement.memory_node] + slot * hot_graph_entry_size;
file.seekp(hot_offset); file.write(hot_graph_entry...);
```

`RemotePtr` 是 8 字节结构（shard ID + byte offset，见第 6 课），`encode_hot_graph_entry` 把它压缩成 5 字节 compact pointer（`kCompactPointerBytes = 5`，`hot_graph.hh:15`）。`slot = (offset - 16) / aligned_size` 是该节点在 shard 内的序号，hot graph entry 按这个序号排列——这样运行时拿到一个节点的 `RemotePtr`，能 O(1) 算出它的 hot graph entry 位置。

### 29.6.4 写 metadata `.meta.json`

`shard_writer.cc:401-460`。这是一份巨大的 JSON，字段涵盖：

- 数据源信息：`data_file`、`num_vectors`、`dim`、`vector_data_type`、`vector_bytes`。
- 构图参数：`R`、`beam_width`、`alpha`、`num_memory_nodes`。
- 布局信息：`node_size`、`node_layout="plain"`、`storage_format="vamana_compact_v1"`、`schema_version=14`、`vector_offset`、`vector_storage_bytes`、`graph_hot_bytes`。
- hot graph 信息：`hot_graph_entry_size`、`hot_graph_pointer_bytes=5`、`hot_graph_shard_bits`、`hot_graph_offsets`、`hot_graph_header_offsets`、`hot_graph_entry_counts`、`hot_graph_dynamic_base_offsets`、`hot_graph_dynamic_record_bytes`、`hot_graph_dynamic_hot_offset`、`allocation_size`。
- medoid：`{"memory_node":..., "offset":...}`。
- 分区信息：`partition_strategy`、`partition_max_degree`、`partition_imbalance`、`partition_edge_cut`、`partition_cross_shard_ratio`。
- 导航/PQ 占位字段：`navigation_quantizer=""`、`pq_subquantizers=0`、`pq_bits=0`、`navigation_format=""`、`navigation_code_remote_offsets=[]`、`navigation_code_region_bytes=[]`、`navigation_code_materialization=""`、`navigation_graph_source="storage_compact_graph"`、`navigation_execution=""`。这些字段在 schema-14 阶段是空的，等 PQ indexer 跑完才会填上（见 29.7）。
- idmap/anchor 格式标记：`idmap_format="owner_sharded_v1"`、`anchor_format`（如果有 anchor 则 `"owner_anchor_v1"`）、`anchor_count_per_shard`。

`offline_builder_version=2`、`random_graph_seed_scope="per_node"` 是版本/审计字段。

### 29.6.5 写 `.idmap` owner 分片

`shard_writer.cc:462-491`。注意这里有个微妙的语义：**owner 分片和 placement 分片是两个不同的概念**。

```cpp
vec<vec<vamana::idmap::Entry>> owner_entries(config.num_memory_nodes);
for (size_t i = 0; i < n; ++i) {
  const u32 owner = config.num_memory_nodes == 0
    ? 0
    : static_cast<u32>(dataset.id(i) % config.num_memory_nodes);
  owner_entries[owner].push_back(vamana::idmap::Entry{
    dataset.id(i),
    RemotePtr{placements[i].memory_node, placements[i].offset}.raw_address,
    0, 0});
}
```

- **placement shard**（`placements[i].memory_node`）：节点 `i` 物理上落在哪个 shard 文件，由分区策略决定。
- **owner shard**（`dataset.id(i) % num_memory_nodes`）：节点 `i` 的 ID 归哪个 owner 管，按 ID 哈希。

两者可能不同。idmap 是"owner → 它管的所有 ID 的物理位置"的反向索引，所以按 owner 分片写。`idmap::Entry` 是 `{id, rptr_raw, generation, reserved}`（见第 8 课）。每个 owner 一个文件 `_node{owner+1}_of{n}.idmap`，含一个 `idmap::Header`（`owner_shard`/`shard_count`/`entry_count`）+ entry 数组。

这个设计让"按 ID 查位置"可以在 owner 节点本地完成，不需要广播给所有 shard。运行时见第 23/28 课。

## 29.7 PQ sidecar：`pq_indexer.cc`

`build_pq_index`（`pq_indexer.cc:353-441`）把 schema-14 索引升级成 schema-15。它要训练 OPQ+PQ 模型、给每个 shard 的每个向量编码、把码流写进 sidecar 文件、再把 metadata 里的导航字段填满。

### 29.7.1 `parse_layout`：校验 schema-14 元数据

`pq_indexer.cc:46-79`。硬性要求：

```cpp
if ((schema_version != 14 && schema_version != gpu_search::format::kMetadataSchemaVersion) ||
    metadata.value("node_layout", str{}) != "plain" ||
    metadata.value("storage_format", str{}) != "vamana_compact_v1" ||
    metadata.value("distance", str{"l2"}) != "l2") {
  throw std::runtime_error("PQ indexer requires a schema-14 plain compact L2 index");
}
```

注意它**也接受 schema-15 输入**（`kMetadataSchemaVersion == 15`）——这是为了支持 `--upgrade-layout-only` 路径（29.7.5）。接着读 `dim/shards/node_bytes/vector_offset/vector_bytes/dtype/node_count/counts/dynamic_offsets`，并做一堆一致性校验：`vector_offset + vector_bytes <= node_bytes`、`vector_dtype_bytes(dtype, dim) == vector_bytes`、`counts` 和 `dynamic_offsets` 长度都等于 `shards`、`sum(counts) == node_count`。任何一条不过都抛异常，防止用坏掉的 metadata 训练。

### 29.7.2 `sample_training_vectors`：分层抽样训练集

`pq_indexer.cc:178-215`。OPQ/PQ 训练需要一份子集，不能全量（太慢）。算法：

```cpp
const u64 sample_count = std::min<u64>(requested, layout.node_count);
if (sample_count < gpu_search::pq::kCentroidsPerSubquantizer) {
  throw std::runtime_error("PQ training requires at least 256 samples");
}
vec<u64> ordinal_bases(layout.shards + 1, 0);
for (u32 shard = 0; shard < layout.shards; ++shard) {
  ordinal_bases[shard + 1] = ordinal_bases[shard] + layout.counts[shard];
}
// ...
const u64 phase = mix64(seed) % layout.node_count;
for (u64 sample = 0; sample < sample_count; ++sample) {
  const u64 ordinal = (phase + sample * layout.node_count / sample_count) % layout.node_count;
  const auto upper = std::upper_bound(ordinal_bases.begin(), ordinal_bases.end(), ordinal);
  const u32 shard = static_cast<u32>(upper - ordinal_bases.begin() - 1);
  const u64 slot = ordinal - ordinal_bases[shard];
  inputs[shard].seekg(kNodeBaseOffset + slot * layout.node_bytes + layout.vector_offset);
  inputs[shard].read(raw.data(), raw.size());
  decode_storage_vector_to_float(raw.data(), layout.dtype, layout.dim,
                                 samples.data() + sample * layout.dim);
}
```

要点：

- `kCentroidsPerSubquantizer = 256`（PQ 每个子量化器 256 个聚类心，k-means 至少要 256 个点），少于这个直接报错。
- `ordinal_bases` 是每个 shard 在全局序号空间里的起始点。用 `upper_bound` 把全局 ordinal 映射回 `(shard, slot)`。
- 抽样是"等间距 + 随机相位"：`phase = mix64(seed) % node_count` 是随机起点，然后每隔 `node_count/sample_count` 取一个。这比纯随机抽样更均匀，且可复现。
- 读出的原始向量用 `decode_storage_vector_to_float` 解码成 float32（uint8/int8 都转 float），因为 Faiss 训练要 float32。

### 29.7.3 `train_model`：OPQ + PQ32 训练

`pq_indexer.cc:217-248`：

```cpp
faiss::OPQMatrix opq(layout.dim, options.subquantizers);
opq.niter = static_cast<int>(options.opq_iterations);
opq.niter_pq = std::max<int>(1, static_cast<int>(options.pq_iterations / 4));
opq.niter_pq_0 = static_cast<int>(options.pq_iterations);
opq.max_train_points = count;
opq.verbose = true;
opq.train(static_cast<faiss::idx_t>(count), samples.data());

vec<f32> transformed(samples.size());
opq.apply_noalloc(static_cast<faiss::idx_t>(count), samples.data(), transformed.data());
faiss::ProductQuantizer product(layout.dim, options.subquantizers, 8);
product.cp.niter = static_cast<int>(options.pq_iterations);
product.cp.seed = static_cast<int>(options.seed);
product.verbose = true;
product.train(count, transformed.data());

gpu_search::pq::Model model;
model.dim = layout.dim;
model.subquantizers = options.subquantizers;
model.rotation = opq.A;
model.centroids = product.centroids;
// ...
```

- `OPQMatrix(dim, subquantizers)`：OPQ 旋转矩阵，`subquantizers` 就是 PQ 的子量化器数（默认 16，常见 32）。`opq.A` 是 `dim x dim` 旋转矩阵。
- OPQ 训练内部会反复训 PQ 优化旋转，`niter_pq_0` 是第一次 PQ 的迭代数（用满 `pq_iterations`），`niter_pq` 是后续每次的迭代数（`pq_iterations/4`，因为旋转稳定后 PQ 收敛快）。
- OPQ 训完后 `apply_noalloc` 把训练集旋转一遍，再用 `ProductQuantizer(dim, subquantizers, 8)` 训 8-bit PQ。`product.centroids` 是 `[subquantizers * 256 * dim/subquantizers]` 的浮点数组。
- `gpu_search::pq::Model`（见第 9 课）是和 Faiss 解耦的运行时模型结构，存 `rotation` 和 `centroids`。`gpu_search::pq::validate` 校验模型完整性。

**BLAS 单线程防嵌套**：`build_pq_index` 在 `pq_indexer.cc:367-369` 设了：

```cpp
omp_set_dynamic(0);
omp_set_max_active_levels(1);
omp_set_num_threads(static_cast<int>(training_threads));
```

`omp_set_max_active_levels(1)` 是关键——Faiss 内部会用 OpenMP 并行，如果不限制嵌套层级，外层 `parallel_for` + 内层 Faiss OMP 会爆炸成 `threads^2` 个线程。这里强制只允许一层 OMP，并把线程数钉死（`omp_set_dynamic(0)` 防止运行时动态调整）。`training_threads` 默认取 `min(hardware_concurrency, 32)`，避免 128 核机器开 128 线程把 BLAS 搞坏。

### 29.7.4 `encode_shard`：分块编码 + 审计

`pq_indexer.cc:250-349`。每个 shard 单独编码，输出 `<prefix>_node{i}_of{n}.pq{subquantizers}.codes`。

```cpp
gpu_search::format::CodeHeader header;
output.write(reinterpret_cast<const char*>(&header), sizeof(header));  // 占位，最后回填

faiss::LinearTransform transform(layout.dim, layout.dim, false);
transform.A = model.rotation;
transform.is_trained = true;
transform.is_orthonormal = true;
faiss::ProductQuantizer product(layout.dim, model.subquantizers, 8);
product.centroids = model.centroids;

const u64 count = layout.counts[shard];
const u32 chunk_vectors = std::max<u32>(1, options.chunk_vectors);
// ...
for (u64 base = 0; base < count; base += chunk_vectors) {
  // 1. 读 chunk_vectors 个 VamanaNode 的向量段
  // 2. decode_storage_vector_to_float 解码
  // 3. transform.apply_noalloc 旋转（如果有 rotation）
  // 4. product.compute_codes 编码成 PQ 码
  // 5. 审计（base==0 时）
  // 6. 写码流 + 更新 checksum
}
```

关键点：

- **CodeHeader 占位**：文件开头先写一个空 header，编完码再 `write_code_header` 回填（`pq_indexer.cc:335-346`）。header 含 `memory_node/code_bytes/node_size/entry_count/remote_offset/payload_bytes/model_checksum/payload_checksum`。`payload_checksum` 是整段码流的 64-bit checksum，运行时加载时校验。
- **分块**：`chunk_vectors` 默认 32768，控制单次内存占用（`chunk_vectors * (node_bytes + dim*4 + dim*4 + code_bytes)`）。
- **审计**（`pq_indexer.cc:300-326`）：每个 shard 第一个 chunk 的前 64 个向量，用 `gpu_search::pq::encode`（运行时编码器）独立编一遍，和 Faiss 的 `product.compute_codes` 结果逐字节比对。如果 mismatches 超过审计组件数的 1% 就抛异常。这是为了抓住"Faiss 编码器和运行时编码器实现不一致"的 bug——PQ 码必须 bit-exact 一致，否则运行时召回会塌。
- **进度条**：`\rPQ encoding shard i/n: base+batch/count` 用 `\r` 原地刷新。

### 29.7.5 `make_persistent_layout` + `apply_persistent_layout`：schema-15 布局

`make_persistent_layout`（`pq_indexer.cc:81-138`）算 PQ 码区在 shard 文件里的物理位置。每个 shard 的 dynamic region 布局是：

```
[dynamic_base, control_offset)              预留
[control_offset, control_offset + 4096)     StorageControlBlock（见第 7 课）
[code_offset, code_offset + count*code_bytes)  PQ 码区
后面按 dynamic_record_bytes 对齐            预留给运行时 dynamic node
```

- `control_offset = align_up(dynamic_offsets[shard], 64)`。
- `code_offset = control_offset + kStorageControlBytes`（4096）。
- `region_bytes = counts[shard] * code_bytes`。
- `dynamic_node_offsets[shard] = dynamic_offsets[shard] + align_up(code_end - dynamic_offsets[shard], dynamic_record_bytes)`。
- `dynamic_code_offset = dynamic_hot_offset + graph_entry_bytes`（dynamic record 内部，PQ 码紧跟 hot graph entry 之后）。
- `dynamic_record_bytes = align_up(dynamic_code_offset + code_bytes, 16)`。

这一堆偏移的意义在于：运行时存储节点加载 shard 文件时，知道 PQ 码区在哪、control block 在哪、dynamic node 从哪开始分配。`apply_persistent_layout`（`pq_indexer.cc:140-155`）把这些写进 metadata：

```cpp
metadata["schema_version"] = gpu_search::format::kMetadataSchemaVersion;  // 15
metadata["navigation_code_bytes"] = code_bytes;
metadata["navigation_code_remote_offsets"] = layout.code_offsets;
metadata["navigation_code_region_bytes"] = layout.code_region_bytes;
metadata["storage_control_remote_offsets"] = layout.control_offsets;
metadata["dynamic_node_base_offsets"] = layout.dynamic_node_offsets;
metadata["hot_graph_dynamic_record_bytes"] = layout.dynamic_record_bytes;
metadata["allocation_size"] = layout.dynamic_record_bytes;
metadata["dynamic_navigation_code_offset"] = layout.dynamic_code_offset;
metadata["navigation_code_materialization"] = "storage_startup_sidecar";
metadata["navigation_graph_source"] = "storage_compact_graph";
metadata["navigation_execution"] = "gpu_beam_v1";
```

`navigation_code_materialization = "storage_startup_sidecar"` 告诉存储节点：PQ 码不在主 shard 文件内嵌，而是从 sidecar `.pq16.codes` 启动时加载（见第 23 课）。`navigation_execution = "gpu_beam_v1"` 标识 GPU 端的导航执行路径（见第 20 课）。

### 29.7.6 `write_metadata_atomic`：原子写 metadata

`pq_indexer.cc:157-169`：

```cpp
void write_metadata_atomic(const filepath_t& path, const nlohmann::json& metadata) {
  const filepath_t temporary{path.string() + ".schema15.tmp"};
  {
    std::ofstream output(temporary, std::ios::trunc);
    output << std::setw(2) << metadata << '\n';
    // ...
  }
  std::filesystem::rename(temporary, path);
}
```

写 `.schema15.tmp` 再 `rename` 覆盖 `.meta.json`。`rename` 在同一文件系统上是原子的，保证升级过程中如果崩溃，metadata 要么是旧 schema-14 要么是新 schema-15，不会出现半写状态。这是离线工具链唯一的"检查点"语义——PQ 编码完成 + 码文件写好 + metadata 原子换版，这三步构成一个事务。

### 29.7.7 `synthesize_distributed_view` 与入口点

`pq_indexer.cc:427-434`：

```cpp
gpu_search::format::View manifest;
bool used_anchors = false;
if (!gpu_search::format::synthesize_distributed_view(
      options.index_prefix, manifest,
      {.entry_points = options.entry_points, .seed = options.seed},
      &used_anchors, &error)) {
  throw std::runtime_error(error);
}
```

这是在 PQ 模型就位后，合成一份"分布式视图"——把 anchor sidecar 和 PQ 模型组合成 GPU 查询启动时需要的入口点列表（`entry_points`，默认 256）。`used_anchors` 标记是否用了 anchor 做入口；如果 anchor sidecar 不存在，就退化到 medoid。这一步只是验证和日志，不写文件——运行时存储节点会重新合成（见第 23 课）。

### 29.7.8 `upgrade_pq_layout`：只升级布局不重新编码

`pq_indexer.cc:443-537`。这是 `--upgrade-layout-only` 走的路径，用于"已有 schema-14 + PQ sidecar，但 metadata 还是旧布局"的场景。它不重新训练、不重新编码，只做：

1. 校验已有 PQ sidecar 的 header（`memory_node/code_bytes/node_size/entry_count/payload_bytes/model_checksum` 全部匹配）。
2. 用 `make_persistent_layout` 算新偏移。
3. 改写每个 sidecar 的 `CodeHeader.remote_offset`（`pq_indexer.cc:502-509`）。
4. 备份旧 metadata 到 `.schema14.bak`（`pq_indexer.cc:513-516`），`apply_persistent_layout` + `write_metadata_atomic`。
5. 如果中途任何一步抛异常，把已改写的 header 回滚（`pq_indexer.cc:520-528` 的 catch 块）。

`--local-shard` 选项支持只重写某一个 shard 的 sidecar（`local_shard != 0` 时跳过其他 shard），适合分布式重建场景下某个 shard 损坏的单点修复。

### 29.7.9 CLI：`vamana_pq_indexer.cc`

`tools/vamana_pq_indexer.cc:8-56`。CLI 用 boost::program_options，关键参数：

- `--index-prefix`（必填）：schema-14 索引前缀。
- `--reuse-model`：复用已有 `.pq16` 模型，跳过训练（`build_pq_index` 里 `options.reuse_model` 非空时走 `gpu_search::pq::read_model`）。
- `--subquantizers`（默认 16）：PQ 子量化器数，必须是 `dim` 的因子。
- `--train-samples`（默认 262144）：训练抽样数。
- `--opq-iterations`（默认 20）/ `--pq-iterations`（默认 25）。
- `--chunk-vectors`（默认 32768）：编码分块大小。
- `--entry-points`（默认 256）：GPU 搜索入口点数。
- `--threads`（默认 0 = min(hw, 32)）。
- `--overwrite`：覆盖已有输出。
- `--upgrade-layout-only`：走 `upgrade_pq_layout` 而非 `build_pq_index`。
- `--local-shard`：配合 `--upgrade-layout-only`，只处理某个 shard。

`main` 根据 `upgrade_layout_only` 分发到两条路径，异常都 catch 成一行错误 + `EXIT_FAILURE`。

## 29.8 anchor sidecar

anchor sidecar 有两个产出路径：内嵌在 `write_vamana_shards` 里的 `write_anchor_sidecar`（`anchor_builder.cc`），和独立的 `vamana_anchor_sidecar_builder.cc` CLI。两者写出的文件格式一致（`vamana::anchor::Header` + per-shard `ShardHeader` + centroid + entries，见第 6 课），但选样逻辑和数据来源不同。

### 29.8.1 内嵌路径：`anchor_builder.cc`

`anchor_builder.cc:35-126`。在 `write_vamana_shards` 里，分区完成后立刻调（`shard_writer.cc:285`）。选样：

```cpp
vec<std::priority_queue<Sample>> samples(config.num_memory_nodes);
for (u32 node = 0; node < graph.num_nodes; ++node) {
  const u32 shard = placements[node].memory_node;
  const u64 priority = mix64(static_cast<u64>(dataset.id(node)) ^
                             (static_cast<u64>(config.seed) << 32));
  auto& heap = samples[shard];
  if (heap.size() < target) {
    heap.push(Sample{priority, node});
  } else if (priority < heap.top().priority) {
    heap.pop();
    heap.push(Sample{priority, node});
  }
}
```

每个物理 shard 维护一个最大堆（大小 `anchor_count_per_shard`，默认 4096）。遍历所有节点，按 `mix64(id ^ (seed<<32))` 算优先级，保留优先级最小的 `target` 个（最大堆 + "比 top 小才替换"= top-K 最小）。这是确定性的、可复现的 top-K 采样。

选完后每个 shard 排序（`std::sort(nodes.begin(), nodes.end())` 按 node id），算 centroid（平均向量），写 header/shard_header/centroid/entries。`EntryHeader` 含 `rptr_raw`（节点的 RemotePtr）、`id`、`degree`。每个 entry 后跟原始向量字节。

如果 `config.anchor_count_per_shard == 0`，`write_anchor_sidecar` 直接 return（`anchor_builder.cc:41-43`），不生成 anchor 文件，metadata 里 `anchor_format=""`。

### 29.8.2 独立 CLI：`vamana_anchor_sidecar_builder.cc`

`tools/vamana_anchor_sidecar_builder.cc:66-269`。这个工具用于"已有 schema-14/15 索引，但想重建 anchor sidecar"的场景——比如最初建索引时没加 anchor，后来想启用 anchor 路由。

它不能依赖内存里的 `VamanaGraph` 和 `Dataset`，而是从磁盘读：

1. 读 `.meta.json` 拿 `dim/shard_count/vector_offset/dtype/vector_bytes`。
2. 检查是否有全部 shard 文件，或者源数据集文件（`data_file`）。至少有一个才能读向量（`vamana_anchor_sidecar_builder.cc:85-93`）。
3. 遍历所有 owner idmap（`_node{i}_of{n}.idmap`），按 `RemotePtr.memory_node()` 把 entry 派到对应物理 shard 的堆里（`vamana_anchor_sidecar_builder.cc:95-132`）。这里的采样优先级是 `mix64(entry.id ^ config.seed)`，和内嵌路径公式一致但 seed 来源不同（内嵌是 `config.seed`，这里是 CLI `--seed`）。
4. 对每个选中的 entry，用 `read_vector` lambda 从 shard 文件（`ptr.byte_offset() + vector_offset`）或源数据集（`2*sizeof(u32) + id * vector_bytes`）读原始向量。
5. 写 `.anchors.tmp`，`rename` 成 `.anchors`。
6. 用 `vamana::anchor::Index` 重新加载做 validation（`vamana_anchor_sidecar_builder.cc:236-246`），确认 `anchor_count() == total` 且 `route` 能返回非空 hints。
7. 更新 metadata 的 `anchor_format`/`anchor_count_per_shard`，原子写回（`.anchor.tmp` → rename）。

注意这里采样基于 **idmap 而非原始节点遍历**——idmap 是 owner 视角的完整 ID 列表，独立 CLI 用它作为"全量节点清单"的来源，避免重新解析 shard 文件里的 fixed record 区。

## 29.9 recall 校验：`recall_check.cc`

`recall_check.cc:87-113` 的 `run_optional_recall_check` 是可选的离线 recall 测试，只在 `--query-path` 和 `--groundtruth-path` 都传了时才跑。

```cpp
const QuerySet queries = read_queries(config.query_path, dataset.dim);
const GroundTruth groundtruth = read_groundtruth(config.groundtruth_path, queries.count);
for (u32 eval_k : {1u, 5u, 10u}) {
  if (eval_k > groundtruth.topk) continue;
  const size_t total_hits = count_hits(graph, dataset, queries, groundtruth, eval_k, config.beam_width);
  const double recall = static_cast<double>(total_hits) / (queries.count * eval_k);
  std::cerr << "recall@" << eval_k << " = " << recall << ...;
}
```

- `read_queries` / `read_groundtruth` 读标准的 `.fbin`/`.bin` 格式（count, dim, then payload）。
- `count_hits`（`recall_check.cc:63-83`）对每个 query 调 `beam_search_float_query`（`graph.cc:266`，query 是 float 数组版本），取前 `eval_k` 个结果，和 groundtruth set 比对。
- 这是**内存图** recall，不是磁盘 shard recall——它在 `write_vamana_shards` 之前跑，测的是 `VamanaGraph` 本身的质量，不涉及序列化/反序列化误差。

`recall@1/5/10` 三档都打。如果 recall 很低，要么是 `R/beam_width` 太小，要么是构图逻辑有 bug。这个和 `build_vamana_graph` 末尾的 in-memory sanity check（200 个随机 query）互补——sanity check 用数据集自身做 query，这个用真实查询集。

## 29.10 配置与 IO：`config.cc` / `dataset_io.cc` / `progress.hh`

### 29.10.1 `VamanaBuildConfig` 与 `parse_configuration`

`config.hh:10-28` 定义所有构建参数，`config.cc:19-90` 用 boost::program_options 解析。几个非显然的校验：

- `--vector-data-type` 必须是 `auto/float32/uint8/int8`（`config.cc:74-80`）。
- `--partition-strategy` 必须是 `balanced/bfs/metis`（`config.cc:81-84`）。
- `--partition-imbalance >= 1.0`（METIS 的 ubvec 物理含义，<1.0 无意义）。
- `--beam-width-construction` 和 `--ef-construction` 是 `--beam-width` 的别名（`config.cc:35-38`），方便和 FAISS/HNSW 生态的参数名对齐。

`--max-vectors` 默认 `u32::max()`，可以用来在小数据集上做调试构建。

### 29.10.2 `read_dataset` 与距离函数

`dataset_io.cc:48-95`。读 `.fbin`/`.u8bin`/`.i8bin` 头（`total_vectors` u32 + `dim` u32），按 `max_vectors` 截断，分块读（`rows_per_chunk = max(1, 64MB / vector_bytes)`）。

距离函数有三档（`dataset_io.cc:97-126`）：

- `uint8`：`int diff = au[i] - bu[i]`，`u32 sum += diff*diff`。用 `int` 而非 `u8` 做差避免下溢。
- `int8`：同上。
- `float32`：`typed_l2_distance`（来自 `common/vector_dtype.hh`）。

`dataset_distance_float_query`（`dataset_io.cc:132-134`）是 query 为 float、rhs 为存储格式的混合距离，medoid 搜索和 recall 校验都用它。

`resolve_dataset_file`（`dataset_io.cc:36-46`）支持传目录，自动找 `base.fbin`/`base.u8bin`/`base.i8bin`/`base.bin`。

### 29.10.3 `ProgressReporter` 与 `parallel_for`

`progress.hh:23-101`。`ProgressReporter` 是一个带后台线程的进度条：

- 构造时启动后台线程，每 250ms `render` 一次（`progress.hh:46-50`）。
- `interactive_` 检测 stderr 是不是 TTY。是 TTY 用 `\r` 原地刷新进度条（`[====...] 50% (n/m) elapsed 1m2s eta 1m0s`），非 TTY（重定向到文件）每 15s 或每 5% 打一行（`progress.hh:80-89`）。
- ETA 用 `elapsed / ratio` 估。
- `finish()` 把 `current_` 设成 `total_`，join 线程，最后一次 `render(true)` 打 "done"。
- 析构函数自动 `finish()`，异常安全。

`parallel_for`（`progress.hh:103-132`）是简单的原子计数器任务窃取：

```cpp
std::atomic<size_t> current{begin};
for (size_t tid = 0; tid < num_threads; ++tid) {
  threads.emplace_back([&, tid]() {
    for (;;) {
      const size_t i = current.fetch_add(1);
      if (i >= end) return;
      try { fn(i, tid); }
      catch (...) { last_exception = ...; current.store(end); return; }
    }
  });
}
```

`fn(i, tid)` 的 `tid` 可用于 per-thread 缓存。异常处理：第一个异常存下来，把 `current` 推到 `end` 让其他线程尽快退出，最后 `rethrow_exception`。这是离线工具链里唯一的多原语——构图、反向边整合、PQ 编码都用它。

## 29.11 旧索引迁移：`legacy_index/migrator.cc`

`migrate_schema13_index`（`migrator.cc:447-547`）把 schema-13 的 rabitq 索引迁移到 schema-14 plain compact。这是一次性历史工具，但它的设计很值得讲——典型的"只读迁移 + 原子发布"模式。

### 29.11.1 schema-13 → schema-14 的区别

schema-13（rabitq）的 VamanaNode 里嵌入了 RaBitQ 量化载荷（`rabitq_offset` 字段）。schema-14 移除了 RaBitQ，改用外部 PQ sidecar（见第 7、9 课）。迁移就是把每个节点的 fixed record 从"含 RaBitQ"压缩成"不含 RaBitQ"，并相应调整所有指针。

### 29.11.2 `parse_layout`：严格校验 + 新布局计算

`migrator.cc:89-169`。先硬性校验：

```cpp
if (metadata.value("schema_version", 0u) != 13 ||
    metadata.value("node_layout", str{}) != "rabitq" ||
    metadata.value("storage_format", str{}) != "vamana_compact_v1" ||
    metadata.value("distance", str{"l2"}) != "l2") {
  throw std::runtime_error("legacy migration requires a schema-13 compact L2 index with embedded RaBitQ");
}
```

然后读旧布局所有字段（`old_node_bytes/old_graph_header_offsets/old_graph_offsets/old_dynamic_offsets`），用 `VamanaNode::init_static_storage` 算新布局（`new_node_bytes = VamanaNode::total_size()`，`new_dynamic_record_bytes = align_compact(new_node_bytes + graph_entry_bytes)`）。

校验项很多（`migrator.cc:124-139`）：`vector_offset == VamanaNode::offset_vector()`、`vector_bytes == VamanaNode::vector_bytes()`、`legacy_payload_offset == vector_offset + vector_storage_bytes`、`old_node_bytes > new_node_bytes`（迁移必须压缩）、`graph_entry_bytes == entry_bytes(degree)`、`shard_bits == shard_bits_for(shards)`、`hot_graph_pointer_bytes == 5`。任何一条不过就抛 "unsupported byte layout"。

接着算每个 shard 的新偏移（`migrator.cc:141-167`）：`new_graph_header_offsets = align_up(new_static_end, 64)`，`new_graph_offsets = align_up(header_offset + sizeof(Header), 64)`，`new_dynamic_offsets = align_up(graph_offset + count*entry_bytes, 64)`。全部 64 对齐。

### 29.11.3 `translate_pointer`：旧 RemotePtr → 新 RemotePtr

`migrator.cc:171-193`。这是迁移的核心难点——所有指向节点的指针（hot graph 邻居、idmap、anchor）都按旧的 `old_node_bytes` 步长编码，迁移后步长变了，必须重算。

```cpp
RemotePtr translate_pointer(RemotePtr pointer, const Layout& layout) {
  if (pointer.is_null()) return pointer;
  const u32 shard = pointer.memory_node();
  if (shard >= layout.shards) throw ...;
  const u64 offset = pointer.byte_offset();
  if (offset < vamana::hot_graph::kNodeBaseOffset) throw ...;
  const u64 relative = offset - vamana::hot_graph::kNodeBaseOffset;
  if (relative % layout.old_node_bytes != 0) {
    throw std::runtime_error(
      "legacy index contains dynamic or unaligned pointers; persist a static snapshot first");
  }
  const u64 slot = relative / layout.old_node_bytes;
  if (slot >= layout.counts[shard]) {
    throw std::runtime_error(
      "legacy index contains dynamic pointers; static schema migration cannot preserve them");
  }
  return RemotePtr{shard, vamana::hot_graph::kNodeBaseOffset + slot * layout.new_node_bytes};
}
```

逻辑：`offset - 16` 得到相对节点区起点的字节，除以 `old_node_bytes` 得到 slot 序号，再乘 `new_node_bytes` 加回 16 得到新偏移。`slot` 不变——节点在 shard 内的顺序不变，只是每条 record 变窄了。

**关键约束**：如果 `relative % old_node_bytes != 0` 或 `slot >= counts[shard]`，说明这个指针指向 dynamic region（运行时分配的动态节点），不是静态节点。migrator 拒绝处理这种指针——"static schema migration cannot preserve them"。这就是为什么 `migrate_shard` 开头要检查源文件大小必须等于 `old_dynamic_offsets[shard]`（`migrator.cc:204-209`）：如果有 dynamic 记录追加在后面，文件会比 `old_dynamic_offsets` 大，迁移会拒绝。**必须先 persist 一个 static snapshot**（让运行时把 dynamic 节点合并进静态区）再迁移。

### 29.11.4 `migrate_shard`：压缩节点 + 重写 hot graph

`migrator.cc:195-303`。每个 shard 单独迁移：

1. **打开源文件 + 创建临时文件**（`migrator.cc:211-220`）：临时文件 `.migration.tmp`，预分配 `new_dynamic_offsets[shard]` 大小。
2. **写文件头**（`migrator.cc:222-225`）：`output_bytes`（新文件大小）+ `zero`（medoid 占位，后面由 metadata 覆盖）。
3. **压缩节点**（`migrator.cc:227-248`）：分 chunk 读旧节点，每个节点 `memset(new, 0)` 清零，然后 `memcpy(new, old, legacy_payload_offset)`——只拷贝 `[0, legacy_payload_offset)` 这一段，即 header + id + generation + vector，**不拷贝 RaBitQ 载荷**。新节点剩余部分保持零。写回新文件对应位置。

   ```cpp
   std::memset(new_nodes.data() + index * layout.new_node_bytes, 0, layout.new_node_bytes);
   std::memcpy(new_nodes.data() + index * layout.new_node_bytes,
               old_nodes.data() + index * layout.old_node_bytes,
               layout.legacy_payload_offset);
   ```

4. **写 hot graph header**（`migrator.cc:250-260`）：新 header 的 `reserved0/1/2` 指向新的 dynamic 布局。
5. **重写 hot graph entries**（`migrator.cc:262-297`）：每个 entry 读出来，先校验 `entry[0] <= degree` 和 `checksum16`（防止源文件损坏），再对每个邻居指针调 `translate_pointer` + `encode_remote_ptr` 重写，最后重算 `checksum16` 写回。这是迁移里最热的循环——每个节点的每个邻居都要 decode/encode 一次。

   ```cpp
   for (u32 neighbor = 0; neighbor < layout.degree; ++neighbor) {
     byte_t* encoded = entry + vamana::hot_graph::neighbor_offset(neighbor);
     const RemotePtr old_pointer = vamana::hot_graph::decode_remote_ptr(encoded, layout.shard_bits);
     if (old_pointer.is_null()) continue;
     const RemotePtr new_pointer = translate_pointer(old_pointer, layout);
     if (!vamana::hot_graph::encode_remote_ptr(new_pointer, layout.shard_bits, encoded)) {
       throw std::runtime_error("translated graph pointer does not fit compact encoding");
     }
   }
   vamana::hot_graph::store_u16_le(entry + 2, vamana::hot_graph::checksum16(entry, layout.graph_entry_bytes));
   ```

   `encode_remote_ptr` 可能失败（新偏移超过 5 字节 compact pointer 能表示的范围），失败就抛异常。

### 29.11.5 `migrate_idmap` 与 `migrate_anchors`

`migrate_idmap`（`migrator.cc:305-342`）：读 idmap header（校验 magic/version/owner_shard/shard_count/文件大小），按 chunk 读 entries，每个非零 `rptr_raw` 调 `translate_pointer` 重写，写临时文件。

`migrate_anchors`（`migrator.cc:344-390`）：读 anchor header（校验 magic/version/dim/shard_count/dtype/vector_bytes），逐 shard 处理：读 `ShardHeader` + centroid + entries，每个 entry 的 `rptr_raw` 调 `translate_pointer` 重写，向量字节原样拷贝。最后校验 `entries == total_anchors` 且 `input.peek() == EOF`（没有多余字节）。

### 29.11.6 `write_metadata`：抹掉 rabitq + schema-14 字段

`migrator.cc:392-437`：

```cpp
for (auto iterator = metadata.begin(); iterator != metadata.end();) {
  if (iterator.key().find("rabitq") != str::npos) iterator = metadata.erase(iterator);
  else ++iterator;
}
metadata["schema_version"] = 14;
metadata["node_layout"] = "plain";
metadata["node_size"] = layout.new_node_bytes;
metadata["graph_hot_bytes"] = VamanaNode::graph_hot_bytes();
metadata["vector_storage_bytes"] = VamanaNode::vector_storage_bytes();
metadata.erase("neighbors_offset");
metadata["medoid"] = {{"memory_node", medoid.memory_node()}, {"offset", medoid.byte_offset()}};
// ... 一堆 hot_graph_* 偏移更新 ...
metadata["navigation_quantizer"] = "";
metadata["navigation_code_bytes"] = 0;
// ... 导航字段全置空 ...
metadata["migration"] = {
  {"source_schema", 13},
  {"source_prefix", options.source_prefix.string()},
  {"method", "static_stride_compaction_v1"},
};
```

关键动作：

- 删掉所有 key 含 "rabitq" 的字段。
- `node_layout` 从 `"rabitq"` 改成 `"plain"`。
- `node_size` 改成新的（更小的）`new_node_bytes`。
- `medoid` 用 `translate_pointer(old_medoid)` 重算。
- 导航/PQ 字段全置空，等后续 `vamana_pq_indexer` 填。
- 追加一个 `migration` 对象记录迁移来源，便于审计。

### 29.11.7 `publish_outputs`：原子发布

`migrator.cc:439-443` + 调用点 `migrator.cc:506-534`。所有输出文件（`shards * 2 + 2`，即 shard.dat + shard.idmap 每个一份 + anchor + metadata）都先写到 `.migration.tmp`，全部成功后才一次性 `rename` 成正式文件名。

```cpp
void publish_outputs(const vec<PendingOutput>& outputs) {
  for (const PendingOutput& output : outputs) {
    std::filesystem::rename(output.temporary, output.final);
  }
}
```

注意 `rename` 不是跨文件系统原子的，但同目录下是原子的。这里所有临时文件和目标文件都在同一目录，所以是原子的——迁移要么全部成功，要么原索引完全不动。这是"检查点语义"的最强形式：**整个迁移是一个原子事务**，中途崩溃不会留下半写状态。

但有个细节：`publish_outputs` 里 rename 是逐个的，不是真原子。如果在 rename 中间崩溃，会有一部分文件是新版、一部分是旧版。不过因为源文件在 `source_prefix`（不同前缀），目标在 `output_prefix`，源不会被改，重跑迁移即可。这是"源只读 + 目标原子"的折中。

### 29.11.8 多线程迁移

`migrator.cc:475-498`：shard 级并行。线程数 `min(shards, requested_threads)`，每个线程 `fetch_add` 抢 shard 任务。`migrate_idmap` 和 `migrate_anchors` 是串行的（idmap 和 anchor 通常很小，不需要并行）。异常用 `std::exception_ptr` 捕获，第一个异常重抛。

## 29.12 GPUNetIO probe 工具

两个独立的 probe 工具，用于部署前确认 GPUDirect RDMA 链路可用。它们和第 22 课讲的 transport probe 不是一回事——第 22 课是 transport 库内部的探针，这两个是独立的可执行文件，从零开始拉起 DOCA/verbs，做最小化的注册和读测试。

### 29.12.1 `gpunetio_probe.cc`：DOCA UMEM + mlx5 MR 注册探针

`tools/gpunetio_probe.cc:94-200`。这个工具不连存储节点，不跑 kernel，只测"能不能在 GPU 内存上注册一个 RDMA MR"。CLI 很简单：

```
gpunetio_probe [gpu_index=0] [ibdev_name=""] [alloc_bytes=64K] [reg_bytes=alloc] [mode=dmabuf|peer]
```

流程：

1. `cudaSetDevice(gpu_index)` + `cudaFree(0)` 初始化 CUDA context + `cudaDeviceGetPCIBusId` 拿 GPU 的 PCI bus ID（`gpunetio_probe.cc:114-118`）。
2. `find_device`（`gpunetio_probe.cc:72-90`）枚举 DOCA 设备，按 IB 设备名过滤。
3. `doca_verbs_context_create` + `doca_verbs_query_device` + 查 `is_gpu_external_datapath_supported`（GPUDirect RDMA 关键能力）+ 查 RC QP 支持。
4. `doca_verbs_pd_create` + `doca_verbs_pd_as_doca_dev` 拿 PD 和 dev handle。
5. `doca_gpu_create(gpu_bus_id)` 创建 DOCA GPU 句柄。
6. `doca_gpu_mem_alloc` 在 GPU 上分配 `alloc_bytes`（`gpunetio_probe.cc:144-146`）。
7. 如果 `mode == "dmabuf"`，`doca_gpu_dmabuf_fd` 拿 dmabuf 文件描述符（`gpunetio_probe.cc:147-151`）。
8. `doca_umem_gpu_create` 注册 DOCA UMEM，然后立即 `doca_umem_destroy`（只测能不能注册，不留着用）（`gpunetio_probe.cc:152-159`）。
9. 通过 `doca_verbs_bridge_verbs_pd_get_ibv_pd` 拿到底层 `ibv_pd*`。
10. 分段注册 MR（`gpunetio_probe.cc:165-181`）：每段 `reg_bytes`（必须 64K 倍数），`dmabuf` 模式用 `mlx5dv_reg_dmabuf_mr`，`peer` 模式用 `ibv_reg_mr`（直接传 GPU 指针）。access flags 是 `LOCAL_WRITE | REMOTE_READ | REMOTE_WRITE`。

成功打 `GPUNetIO probe passed` 和一堆诊断信息（GPU bus、ibdev、首段 lkey）。失败打 `GPUNetIO probe failed: <error>`。

这个 probe 对应记忆里 "g201 DOCA peermem missing" 的排查场景——`doca_umem_gpu_create` 失败通常是因为 `nvidia-peermem` 内核模块没加载，或者 GPU 和 NIC 不在同一 NUMA/PCIe root complex。在 `.201` 上会失败，在 `.202` 上才能过。

### 29.12.2 `gpunetio_loopback_probe.cc`：kernel 端 loopback 读探针

`tools/gpunetio_loopback_probe.cc:41-191`。这个工具更进一步——它真的连上存储节点（用 `ClientConnectionManager`），拉起 `GpuNetioPersistentTransport`，发一个 kernel probe。

流程：

1. `configuration::IndexConfiguration config{argc, argv}` + `Context` + `ClientConnectionManager::connect` 拉起 RDMA 连接（`gpunetio_loopback_probe.cc:42-45`）。
2. 给每个 server QP 发一个 `configuration::Parameters{num_threads=1, gpu_rdma_qps=...}`，让对端知道用几个 QP（`gpunetio_loopback_probe.cc:47-54`）。
3. 收 `MemoryRegionToken`（对端授权的远端 MR 信息）（`gpunetio_loopback_probe.cc:56-62`）。
4. 从环境变量读 stress 参数：`DVSTOR_GPUNETIO_STRESS_BLOCKS`（默认 64）、`DVSTOR_GPUNETIO_STRESS_ITERATIONS`（默认 32）、`DVSTOR_GPUNETIO_BATCH_READS`（默认 1）（`gpunetio_loopback_probe.cc:64-69`）。
5. 构造 `gpu::GpuNetioPersistentTransport`，拿 `view`（含 `remote_regions/qps/local_mkey/local_iova_base` 等）（`gpunetio_loopback_probe.cc:74-77`）。
6. 分配 GPU 状态 buffer：`stop/disabled/error/statuses/completed` + 一个 CUDA stream（`gpunetio_loopback_probe.cc:84-96`）。
7. 填 `PersistentKernelParams`，把 transport view 的字段塞进 `direct_*` 字段（`gpunetio_loopback_probe.cc:97-115`）。
8. 根据 `batch_reads` 选 kernel：`batch_reads == 1` 调 `launch_gpunetio_locked_read_probe`（逐次 locked read），否则调 `launch_gpunetio_batched_read_probe`（批量读）（`gpunetio_loopback_probe.cc:117-125`）。
9. `cudaStreamSynchronize` 等完成，`cudaMemcpy` 把 `statuses/completed/disabled/error` 拷回 host（`gpunetio_loopback_probe.cc:126-140`）。
10. 校验：`launch_status == cudaSuccess && sync_status == cudaSuccess && host_error == 0 && host_completed == expected && all statuses == 0`（`gpunetio_loopback_probe.cc:147-151`）。失败打详细诊断（前 16 个 status）。
11. 成功打 `GPUNetIO locked-read stress passed: operations=... rate=... ops/s`。
12. 额外做一次 `storage_startup::Request`/`Response` 握手，确认对端存储节点 ready（`gpunetio_loopback_probe.cc:175-187`）。

和 `gpunetio_probe.cc` 的区别：`gpunetio_probe` 只测 DOCA UMEM/MR 注册（不连存储节点），`gpunetio_loopback_probe` 测的是 kernel 端真的能通过 GPUNetIO 发 RDMA Read（连存储节点）。前者是"硬件 + 驱动"层探针，后者是"kernel + transport"层探针。和第 22 课的 transport probe 相比，这两个工具是面向运维的独立可执行文件，不依赖完整的计算服务配置。

## 29.13 与其他模块的关系

- **第 6 课**（Vamana 图与 anchor/idmap）：本课的 `VamanaGraph` 是内存版，第 6 课讲的是磁盘版 `vamana_compact_v1` 格式。`RemotePtr`、`vamana::hot_graph::Header`、`vamana::idmap::Entry`、`vamana::anchor::Header` 都在第 6 课定义，本课只是写出方。
- **第 7 课**（schema-15 索引格式）：本课产出 schema-14（`write_vamana_shards`）和 schema-15（`build_pq_index` 的 `apply_persistent_layout`）。`kMetadataSchemaVersion=15`、`kStorageControlBytes=4096`、`kNodeBaseOffset=16`、`kCompactPointerBytes=5` 都在第 7 课的 `index_format.hh` 定义。
- **第 8 课**（元数据/owner map/存储协议）：`.idmap` 的 owner 分片语义（`id % num_memory_nodes`）在第 8 课详述，本课是产出方。
- **第 9 课**（GPU 类型/遥测/PQ 模型）：`gpu_search::pq::Model`、`gpu_search::pq::encode`、`gpu_search::pq::validate`、`gpu_search::pq::read_model`/`write_model` 都在第 9 课。本课 `train_model` 把 Faiss 的 `OPQMatrix.A` 和 `ProductQuantizer.centroids` 拷进 `Model`，运行时编码器（`pq::encode`）和 Faiss 编码器（`product.compute_codes`）必须 bit-exact 一致——`encode_shard` 里的审计就是抓这个。
- **第 12/13 课**（construction）：在线增量构图走的是另一条路径（见第 12/13 课），本课是离线全量构图。两者都用 RobustPrune，但离线版 `build_alpha=1.0` 固定，在线版可能用 `alpha>1`。
- **第 17/20/21 课**（kernel 启动器/查询遍历/角色调度）：PQ 码区布局（`navigation_code_remote_offsets`、`dynamic_navigation_code_offset`）是 kernel 端 GPUNetIO 读的物理位置，必须和本课写出的偏移一致。
- **第 22 课**（GPUNetIO 传输/probe）：第 22 课讲的是 transport 库内部的 probe 机制，本课的 `gpunetio_probe.cc`/`gpunetio_loopback_probe.cc` 是独立的部署前探针，用最小化配置验证链路。
- **第 23 课**（存储节点主体）：存储节点启动时加载本课产出的 `.dat`/`.idmap`/`.anchors`/`.pq16.codes`/`.meta.json`，按 `navigation_code_materialization="storage_startup_sidecar"` 从 sidecar 加载 PQ 码。
- **第 30 课**（breakdown benchmark/实验脚本）：实验脚本会调本课的三个 CLI（`vamana_offline_builder`、`vamana_pq_indexer`、`vamana_anchor_sidecar_builder`）和 migrator 生成测试索引。

## 29.14 小结

本课讲了 dvstor 的离线工具链，它是"原始向量数据"到"线上可加载索引"之间的桥梁。核心要点：

1. **构图（`graph.cc`）**：`VamanaGraph` 是 CSR 风格内存图，`build_vamana_graph` 走"随机 R-regular 初始化 → per-node beam_search + robust_prune → cheap append 反向边 → bulk consolidate"四步。**构图固定 `build_alpha=1.0`**，配置里的 `alpha` 只存 metadata。`LocalIdSet` 是手写开放寻址哈希集合，`beam_search` 高频用它去重。
2. **分区（`partitioning.cc`）**：三种策略——balanced（最小堆贪心，不考虑拓扑）、bfs（多源 BFS + 负载均衡，拓扑连续）、metis（METIS k-way，质量最高但需外部库，有 `DVSTOR_HAVE_METIS` 链接校验和 `idx_t` 容量校验）。`place_nodes` 分发，`compute_cross_shard_ratio_graph` 算跨 shard 边比例。
3. **写 shard（`shard_writer.cc`）**：每个 shard 文件分三段（file header + fixed VamanaNode records + hot graph header/entries + dynamic 预留）。`encode_hot_graph_entry` 把 `RemotePtr` 压成 5 字节 compact pointer，按 `slot = (offset-16)/aligned_size` 排列。metadata 是巨大的 JSON，schema-14 阶段导航字段全空。**idmap 按 owner 分片（`id % num_memory_nodes`），与 placement shard 是两个概念**。
4. **PQ sidecar（`pq_indexer.cc`）**：schema-14 → schema-15。`sample_training_vectors` 等间距 + 随机相位抽样，`train_model` 用 Faiss OPQ+PQ32（`omp_set_max_active_levels(1)` 防 OMP 嵌套爆炸），`encode_shard` 分块编码 + 64 向量审计（Faiss vs 运行时编码器 bit-exact 校验），`make_persistent_layout` 算码区/control 区/dynamic 区偏移，`apply_persistent_layout` 填 metadata，`write_metadata_atomic` 原子换版。`--upgrade-layout-only` 支持只改布局不重新编码（带 header 回滚）。
5. **anchor sidecar**：内嵌路径（`anchor_builder.cc`，构图后直接调）和独立 CLI（`vamana_anchor_sidecar_builder.cc`，从 idmap 重建）两种。都是按 `mix64(id ^ seed)` 优先级 top-K 采样，每 shard 一个最大堆。
6. **recall 校验（`recall_check.cc`）**：可选，用真实 query/groundtruth 测内存图 recall@1/5/10，和 `build_vamana_graph` 末尾的随机 sanity check 互补。
7. **迁移（`migrator.cc`）**：schema-13 rabitq → schema-14 plain compact。`translate_pointer` 按 `slot = (offset-16)/old_node_bytes` 重算所有 RemotePtr，`migrate_shard` 压缩 fixed record（只拷贝 `[0, legacy_payload_offset)`，丢弃 RaBitQ 载荷）+ 重写 hot graph entries（含 checksum16 校验和重算）。**拒绝处理 dynamic 指针**——必须先 persist static snapshot。`publish_outputs` 一次性 rename 全部 `.migration.tmp` 成正式文件，源索引只读不动。
8. **GPUNetIO probe**：`gpunetio_probe.cc` 测 DOCA UMEM/MR 注册（不连存储节点，对应 "g201 peermem missing" 排查），`gpunetio_loopback_probe.cc` 测 kernel 端 RDMA Read loopback（连存储节点，跑 `launch_gpunetio_locked_read_probe`/`launch_gpunetio_batched_read_probe`）。两者都是部署前运维探针，和第 22 课的 transport 内部 probe 不同。

整条工具链的检查点语义：构图无中间产物（内存图直接写 shard），PQ 编码以 `.meta.json` 原子换版为检查点，迁移以 `publish_outputs` 批量 rename 为检查点。这些检查点保证了"崩溃可重跑"——任何一步失败都不会留下半写状态污染下一阶段。
