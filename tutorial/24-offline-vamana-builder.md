# 第 24 课：离线 Vamana 构建器

## 本课目标

本课学习离线 Vamana builder 的实现。学完后，你需要能够：

1. 从 `tools/vamana_offline_builder.cc` 跟踪完整 build graph 流程。
2. 理解离线 `VamanaGraph` 的数据结构和锁模型。
3. 对比离线构建和在线插入的算法、并发与性能路径差异。
4. 判断离线 builder 的哪些部分适合作为后续优化实验点。

代码入口：

- `tools/vamana_offline_builder.cc`
- `tools/vamana_offline/config.hh`
- `tools/vamana_offline/dataset_io.hh`
- `tools/vamana_offline/graph.hh`
- `tools/vamana_offline/graph.cc`
- `tools/vamana_offline/recall_check.cc`
- `tools/vamana_offline/shard_writer.cc`

## 1. 主流程

`tools/vamana_offline_builder.cc` 的 main 很短，但串起了所有关键步骤：

1. `parse_configuration(argc, argv)` 解析构建配置。
2. `read_dataset(config)` 读取 dataset。
3. 决定 `output_prefix`。
4. 打印构建参数：
   - memory nodes
   - thread count
   - R
   - construction beam width
   - alpha
   - vector dtype
   - partition strategy
5. 初始化 `VamanaNode` 静态 storage：
   - `disable_rabitq()`
   - `init_static_storage(dataset.dim, config.R, dataset.dtype)`
6. 构造 `VamanaGraph graph`。
7. `build_vamana_graph(graph, dataset, config)`。
8. `run_optional_recall_check(graph, dataset, config)`。
9. `write_vamana_shards(graph, dataset, config, output_prefix)`。
10. 输出总耗时。

这个 main 说明离线 builder 分成三个阶段：

- 内存中构图。
- 可选 recall 检查。
- 写出 shard 和 metadata。

离线构建不连接 RDMA，不初始化 GPU，不启动 service worker。

## 2. VamanaGraph 数据结构

`tools/vamana_offline/graph.hh` 定义：

```cpp
struct VamanaGraph {
  size_t num_nodes;
  u32 dim;
  u32 R;
  size_t medoid;
  vec<u32> neighbors;
  vec<u8> degrees;
  unique_ptr<atomic_flag[]> lock_stripes;
  size_t lock_stripe_count;
};
```

邻居存储方式：

- `neighbors` 是扁平数组。
- 每个 node 占 `R` 个 slot。
- `offset(node) = node * R`。
- `degrees[node]` 记录当前有效邻居数。
- 空邻居用 `kEmptyNeighbor = UINT32_MAX`。

锁模型：

- `lock_stripes` 是一组 `atomic_flag`。
- `lock_node(node)` 用 `node & (lock_stripe_count - 1)` 映射到锁 stripe。
- 多个 node 可能共享同一把 stripe lock。
- lock 循环中在 x86 上调用 pause 指令。

这个设计的优点：

- 内存紧凑。
- copy/write 邻居容易。
- 锁数量可控，不需要每个 node 一个 mutex。

缺点：

- stripe 冲突会导致无关节点互相阻塞。
- `R` 必须能放入 `u8 degree`，代码要求 `max_degree <= UINT8_MAX`。
- 扁平数组更新适合固定最大度，不适合高动态度。

## 3. 初始化随机 R-regular graph

`build_vamana_graph()` 首先：

1. `graph.init(n, dataset.dim, R)`。
2. `graph.medoid = compute_medoid(...)`。
3. 创建 `order = [0, 1, ..., n-1]`。
4. 按 seed shuffle order。
5. 并行初始化每个 node 的随机邻居。

随机初始化逻辑：

- 每个 node 使用 `mix_seed(seed ^ i)` 得到自己的随机种子。
- 随机选不同于自己的 node。
- 避免重复。
- 初始化到 R 个邻居。

初始化后有一个 sanity check：

- 对前 `min(n, 4096)` 个节点计算邻居 signature。
- 统计 unique signature ratio。
- 要求 ratio >= 0.99。

这个检查是为了防止随机图初始化出现大量重复邻居列表。它是一个实现级质量护栏，不是算法理论的一部分。

## 4. compute_medoid

`compute_medoid()` 的逻辑：

1. 取 sample size：
   - `min(n, 10000)`
2. 如果 n 大于 sample size：
   - 用 seed 42 随机采样 index。
3. decode sample vector 到 float。
4. 计算 sample centroid。
5. 遍历所有 node，找到离 centroid 最近的 node。
6. 返回该 node 作为 medoid。

对 L2 来说，这是合理的近似中心点。

对 IP 距离，需要结合 `dataset_distance_float_query(..., ip_distance)` 的实现理解“最近”语义。不要只根据函数名推断。

性能影响：

- decode sample 成本是 `O(sample_size * dim)`。
- 遍历所有 node 成本是 `O(n * dim)`。
- 对大数据集，medoid 计算会是显著前置成本。

可优化方向：

- 并行计算 centroid。
- 并行扫描 best medoid。
- 对超大 dataset 做分块 IO。
- 允许采样候选而不是全量扫描。

## 5. beam_search

离线 `beam_search()` 输入：

- graph
- dataset
- query id
- beam width
- distance mode

核心状态：

- `all_visited`
- `beam`
- `visited`
- `expanded`

流程：

1. 以 medoid 作为初始 beam。
2. 循环寻找 beam 中第一个未 expanded 的节点。
3. 锁该节点并复制邻居。
4. 对未访问邻居计算距离。
5. 插入 sorted beam。
6. 记录 all_visited。
7. 没有未 expanded 节点时结束。
8. sort all_visited 后返回。

它和在线搜索的区别：

- 离线邻居在本地内存数组，不需要 RDMA。
- 距离计算在 CPU 上直接访问 dataset。
- 没有 GPU H2D/D2H。
- 没有 RaBitQ gate。
- 没有 credit-aware expansion。
- 没有 hot graph 读。

因此离线 beam search 更容易理解，也适合做算法单元测试。

## 6. robust_prune

离线 `robust_prune()` 输入：

- dataset
- source
- sorted candidates
- alpha
- R
- distance mode

逻辑：

1. 初始化 selected。
2. 遍历按距离排序的 candidate。
3. 跳过 source 自己。
4. selected 达到 R 后停止。
5. 对每个已选邻居 `sel_id`：
   - 计算 `d(sel_id, cand_id)`。
   - 如果 `alpha * d_sel_cand <= cand_dist`，则 cand 被 prune。
6. 未被 prune 的 candidate 加入 selected。

离线构建中有一个重要实现细节：

```cpp
const float build_alpha = 1.0f;
if (alpha > 1.0f + 1e-6f) {
  std::cerr << "note: using alpha=1.0 for construction ..."
}
```

也就是说，离线构建实际使用 `build_alpha = 1.0`，但 metadata 中仍保存 config alpha。这一点必须写入学习笔记，否则你会误以为构建使用了命令行 alpha。

## 7. 主构建循环

`build_vamana_graph()` 的核心 parallel loop：

1. 从 shuffled order 中取 `node_idx`。
2. 对该 node 执行 `beam_search(...)` 得到 candidates。
3. 锁 node，复制现有邻居。
4. 把现有邻居也加入 candidates。
5. sort and unique candidates。
6. `robust_prune(...)` 得到 new neighbors。
7. 锁 node，设置新邻居。
8. 对每个 new neighbor：
   - 锁 neighbor。
   - 尝试 append reverse edge。

这里的 reverse edge 维护是“cheap append”：

- 如果 neighbor degree 未满，则 append source。
- 如果满了，暂时不处理。

之后再做 bulk consolidation。

并发风险：

- node 自身和 neighbor 更新使用 stripe lock。
- beam search 复制邻居时也加 lock。
- 但图在构建过程中持续变化，beam_search 看到的是动态快照。

这和在线 Vamana 插入类似：图在构建过程中不是全局静止的。

## 8. reverse-edge consolidation

`consolidate_reverse_edges()` 做批量反向边整理：

1. 统计每个 node 的 incoming edge count。
2. 构造 incoming offsets。
3. 填充 incoming_edges CSR。
4. 对每个 node：
   - 收集 outgoing neighbors。
   - 收集 incoming sources。
   - 去重。
   - 对候选计算距离。
   - robust prune。
   - 设置最终邻居。

这一步相比 cheap append 更完整：

- 它将 outgoing 和 incoming 合并。
- 再重新 prune 到 R。
- 提升图的互连质量。

性能成本：

- 需要额外 CSR 内存。
- 每个 node 要对候选重新计算距离。
- 并行执行，但内存带宽压力大。

优化方向：

- 统计候选数量分布。
- 减少重复距离计算。
- 分块构建 incoming_edges。
- 对高维向量优化距离计算。

## 9. recall check

`run_optional_recall_check()` 只有在配置提供 query 和 groundtruth path 时执行。

它会：

1. 读取 query `.fbin`。
2. 读取 groundtruth `.bin`。
3. 对 eval_k 取 1、5、10。
4. 对每个 query 调用 `beam_search_float_query(...)`。
5. 与 groundtruth row 计算 hit。
6. 打印 recall。

注意这是离线内存图 recall，不是最终 RDMA/GPU runtime recall。它不能覆盖：

- shard writer 编码错误。
- memory node load 错误。
- hot graph 读错误。
- RaBitQ gate 误筛。
- online insert 后一致性。

所以它是 builder 阶段的 sanity check，不是完整系统验证。

## 10. 离线与在线路径差异表

| 维度 | 离线 builder | 在线 insert/search |
|---|---|---|
| 数据位置 | 本地 dataset 和本地图数组 | memory node 远端内存 |
| 邻居读取 | 本地数组 copy | RDMA read 或 hot graph read |
| 距离计算 | CPU dataset distance | GPU typed L2/IP 或 CPU 辅助 |
| 并发模型 | parallel_for + stripe lock | service worker + coroutine + RDMA/GPU event |
| 节点分配 | shard writer 静态 placement | RDMA FAA 分配 remote ptr |
| reverse update | cheap append + bulk consolidation | 对邻居 CAS lock，写 neighbor list，storage-owner 可跨 peer |
| layout | 构建结束写出 | runtime 根据 metadata 解析 |
| recall check | 内存图查询 | benchmark 或线上 query |
| 失败模型 | 构建失败可直接退出 | 可能部分写入远端图 |

## 11. 性能影响

离线构建主要瓶颈：

1. 距离计算：
   - beam search 中大量 `dataset_distance`。
   - robust prune 中候选两两距离。

2. graph lock contention：
   - 热点 medoid 附近 node 更容易被多线程锁住。
   - stripe lock 会放大冲突。

3. memory bandwidth：
   - dataset decode。
   - neighbors 扁平数组访问。
   - incoming_edges CSR 构建。

4. random initialization：
   - n 很大时初始化 R 个邻居也是 `O(n * R)`。

5. reverse consolidation：
   - incoming count、prefix offsets、incoming_edges 都可能占大量内存。

可观测指标：

- build 总耗时。
- medoid 计算耗时。
- graph build loop 耗时。
- reverse consolidation 耗时。
- avg/max/min degree。
- incoming_edges 数量。
- recall@k。
- peak RSS。
- lock contention 采样。

## 12. 设计异味

1. 离线构建和在线插入的 Vamana 逻辑重复：
   - beam search、robust prune 思路相同，但实现完全分离。

2. `build_alpha` 固定为 1.0：
   - metadata 保存 config alpha，实际构建不使用，容易造成理解偏差。

3. CPU distance hardcoded：
   - main 打印 `offline distance execution: cpu-avx2`，但接口没有清晰策略抽象。

4. graph lock stripe 是低层细节：
   - 没有暴露 contention 指标。

5. recall check 与 builder 耦合：
   - 只能测内存图，不测写出后的 shard。

6. memory layout 状态依赖 `VamanaNode` 静态配置：
   - offline writer 和 runtime 使用同一静态 layout 类。

## 13. 可验证问题

1. 随机初始化：
   - 不同 seed 是否产生不同 graph。
   - 相同 seed 是否可复现。

2. medoid：
   - sample size 小于 n 时是否固定 seed 42。
   - IP distance 下 medoid 选择是否符合预期。

3. build alpha：
   - config alpha 改变时，构建输出 graph 是否实际变化。

4. reverse consolidation：
   - consolidation 前后 avg degree、edge symmetry、recall 是否变化。

5. lock stripes：
   - 减少 lock stripe count 是否显著降低 build throughput。

6. recall check：
   - 内存图 recall 和最终 runtime recall 是否一致。

## 14. 学习任务

1. 画一张离线 build graph 流程图：dataset -> random graph -> medoid -> beam/prune -> reverse consolidation -> recall -> shard writer。
2. 对 `VamanaGraph` 写一张内存布局图，标出 `neighbors` 和 `degrees` 如何定位 node。
3. 对比离线 `robust_prune()` 与在线插入 prune kernel 的输入输出。
4. 设计一个小规模测试：n=1000、dim=16、R=16，记录不同 beam width 的 recall/build time。
5. 设计一个重构方案：将离线和在线可共享的 prune policy 抽成纯函数或策略对象。

