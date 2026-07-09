# 第 06 课：Vamana 图索引算法在代码中的边界

## 本课目标

本课从 `Vamana<Distance>` 模板类出发，建立在线索引的算法边界。你需要明确哪些逻辑属于 Vamana 算法，哪些逻辑是 RDMA 操作，哪些逻辑是 GPU 操作，哪些只是服务层调度。

## 代码证据

必须阅读：

- `src/vamana/vamana.hh`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`
- `src/vamana/vamana_helpers.ipp`
- `src/vamana/vamana_neighborlist.hh`
- `src/vamana/vamana_neighborlist.cc`
- `src/vamana/vamana_node.hh`

## Vamana 类的定位

`Vamana<Distance>` 是在线 compute side 的图索引算法载体。它不是完整服务，也不持有队列、RPC、HTTP 或 memory node 连接。它依赖外部传入的 `ComputeThread`，通过 `ComputeThread` 再访问 RDMA context、GPU buffer、statistics 和 breakdown sample。

构造参数：

```cpp
Vamana(u32 R,
       u32 beam_width,
       u32 beam_width_construction,
       f64 alpha,
       u32 k,
       u32 dim,
       VectorDType vector_dtype)
```

关键成员：

- `R_`
- `beam_width_`
- `beam_width_construction_`
- `alpha_`
- `k_`
- `dim_`
- `direct_node_reads_`
- `expansion_batch_`
- `credit_aware_expansion_`
- `query_batch_size_`
- `use_rabitq_`
- `rabitq_cache_`
- `rabitq_gate_width_`
- `rabitq_gate_max_width_`
- `rabitq_gate_margin_`

构造函数会调用：

```cpp
VamanaNode::init_static_storage(dim, R, vector_dtype)
```

这再次说明 `Vamana` 与 `VamanaNode` 的静态布局强耦合。

## include .ipp 的结构

`Vamana` 类内部 include：

```cpp
#include "vamana/vamana_search.ipp"
#include "vamana/vamana_insert.ipp"
#include "vamana/vamana_helpers.ipp"
```

这不是普通头文件依赖，而是把函数实现直接展开进模板类定义。这样做的目的通常是让模板函数可见并可实例化。代价是代码边界不清晰，阅读时要记住 `.ipp` 中的函数都是 `Vamana<Distance>` 的成员。

## 在线查询入口

查询相关成员：

- `knn`
- `knn_raw`
- `knn_batch`

`knn` 接收 float span，然后转成 raw：

```text
knn(q_id, span<float>)
  -> knn_raw(q_id, byte*, VectorDType::float32)
```

`knn_raw` 是单查询主路径。它负责：

- 初始化 query 计数。
- 准备 query H2D 或 RaBitQ rotated query。
- 初始化 beam 和 visited。
- 选择 entry points 或 medoid。
- 执行批量 beam expansion。
- RDMA 读取 neighbor list 和 candidate vectors。
- GPU 距离计算。
- 更新 beam。
- 最终读取 node id 并写入 `thread->query_results[q_id]`。

## 在线插入入口

插入相关成员：

- `insert`

`insert` 负责：

- 读取 medoid。
- 处理空索引 first insert。
- 使用 beam search 找候选。
- GPU RobustPrune 选择新节点邻居。
- 远程分配新节点。
- 写新节点。
- 对每个 selected neighbor 做反向边更新。

普通 compute-side insert 是强一致性较重的路径，因为它会远程 lock neighbor，读邻居列表，必要时 overflow prune，再写邻居列表。

## helper 函数

`vamana_helpers.ipp` 中有：

- `insert_into_beam`
- `upsert_beam`
- H2D/D2H 统计函数
- breakdown 计时辅助函数
- GPU kernel timing 辅助函数
- `read_node` MinorCoroutine

这些 helper 看似简单，但经常处在热路径。例如 `insert_into_beam` 使用 vector lower_bound + insert，beam 大小时会带来移动成本。

## Neighborlist 抽象

`VamanaNeighborlist` 是邻居读缓冲区的 view。它认为 buffer 格式是：

```text
id(4B) + edge_count(1B) + padding + R * RemotePtr(8B)
```

核心方法：

- `num_neighbors`
- `set_num_neighbors`
- `view`
- `all_slots`
- `add`
- `reset`

对于 compact storage，远端实际存的是 hot graph entry，但 `rdma::vamana::NeighborReadAwaitable::await_resume` 会把 compact hot graph decode 成传统 neighbor read buffer，再交给 `VamanaNeighborlist`。

因此算法层可以统一使用 `VamanaNeighborlist::view()`，不关心 AoS 或 compact 存储格式。

## 算法概念到源码函数

| 算法概念 | 代码位置 |
| --- | --- |
| entry point | `knn_raw` 中 `entry_points` 或 `read_medoid_ptr` |
| beam | `VamanaCoroutine::beam` |
| visited set | `VamanaCoroutine::visited_nodes` |
| expand best unexpanded | `select_best` lambda |
| neighbor fetch | `rdma::vamana::read_vamana_neighbors` |
| candidate vector fetch | `rdma::vamana::batch_read_vectors` |
| exact distance | `gpu::launch_batch_typed_query_l2_distances` |
| RaBitQ gate | `rabitq_cache_->estimate_batch_lut` 和 `select_gate_into` |
| RobustPrune | `gpu::launch_robust_prune_typed` |
| reverse edge update | `Vamana::insert` 后半段 |

## 与 HNSW 的差异

代码注释提到它替换了 HNSW，但你应该从实现看差异：

- `VamanaNode` 是单层图，没有 HNSW level。
- 每个节点最多 `R` 个邻居。
- 搜索用 beam，而不是多层 greedy descent。
- 插入后使用 RobustPrune 控制度数。
- entry point 是 medoid 或 anchor hints。

不要从名称推断行为，要以 `vamana_search.ipp` 和 `vamana_insert.ipp` 为准。

## 性能影响

Vamana 算法边界内的主要开销：

- beam selection：线性扫描 `beam` 找最小未展开节点。
- visited set：`unordered_set<RemotePtr>` 查重。
- neighbor fetch：小块 RDMA READ，延迟敏感。
- vector fetch：大块批量 RDMA READ，带宽和 QP credit 敏感。
- GPU distance：kernel launch 和 D2H 同步敏感。
- beam update：vector insert 可能搬移元素。
- final id read：逐个结果读取 id，会增加尾部 RDMA 小读。

## 设计异味

1. `Vamana<Distance>` 同时了解 RDMA、GPU、breakdown、statistics，算法和系统实现耦合。
2. `direct_node_reads_` 是常量 true，但仍保留分支，说明历史设计残留。
3. 查询和插入都在 `.ipp` 中，文件较长，难以局部测试。
4. `VamanaNode::init_static_storage` 在 `Vamana` 构造中调用，算法对象会修改全局静态布局。
5. `Neighborlist` 的 buffer ownership 依赖 `ComputeThread* owner_`，这让数据结构不纯粹。

## 可验证问题

- `Vamana<Distance>` 是否持有 RDMA QP？
- `VamanaNeighborlist` 是否知道 compact hot graph？
- `knn_raw` 的结果写到哪里？
- `insert_into_beam` 如何保持 beam 有序？
- `beam_width` 和 `beam_width_construction` 分别在哪条路径使用？

## 学习任务

1. 读完 `Vamana` 类定义，列出所有 setter，并写明对应配置项。
2. 在 `vamana_search.ipp` 中标记每一个 `co_await` 的子系统：RDMA 或 GPU。
3. 在 `vamana_insert.ipp` 中标记 search、prune、write、reverse update 四个阶段。
4. 画出 `VamanaNeighborlist` buffer 格式。
5. 思考：如果要把算法和 RDMA 分离，需要定义哪些接口？

