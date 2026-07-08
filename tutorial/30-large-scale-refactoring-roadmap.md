# 第 30 课：大规模重构路线图设计

## 本课目标

这是最后一课。目标不是马上重写项目，而是给未来的大规模重构建立路线图。学完后，你需要能够：

1. 识别当前架构的结构性问题。
2. 明确哪些重构会改变性能、索引格式、网络协议或实验可比性。
3. 设计渐进式重构路线：先可测，再抽边界，再替换实现。
4. 给每个阶段定义行为不变量和回归测试。

代码入口：

- `src/service/compute_service.hh`
- `src/vamana/vamana.hh`
- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`
- `src/memory_node/memory_node.hh`
- `src/common/configuration.hh`
- `src/http/vamana_service_scheduler.hh`
- `src/rdma/vector_batch_planner.hh`
- `src/service/storage_owner_protocol.hh`

## 1. 当前架构的主要问题

基于前 29 课源码阅读，可以归纳出七个结构性问题。

### 1.1 ComputeService 过大

`ComputeService` 同时负责：

- RDMA context 和 connection manager。
- remote token。
- worker pool。
- Vamana algorithm object。
- RaBitQ cache。
- anchor index。
- service queues。
- RPC routing。
- storage-owner insert runtime。
- compute-side idmap。
- metadata validation。
- breakdown report。
- load/store command。

这导致：

- 单元测试困难。
- 构造函数过重。
- 模板类膨胀。
- 改一个子系统容易触碰其他子系统。

### 1.2 VamanaNode 静态全局 layout

`VamanaNode` 保存：

- DIM。
- R。
- VECTOR_DTYPE。
- STORAGE_FORMAT。
- RaBitQ 状态。
- hot graph offsets。
- node size/offset。

这些是进程级静态状态。

后果：

- 同一进程多 index 不现实。
- metadata validation 有副作用。
- load index 必须暂停 worker。
- 测试之间容易污染。

### 1.3 算法、RDMA、GPU 交织

`vamana_search.ipp` 和 `vamana_insert.ipp` 同时包含：

- Vamana beam/visited 算法。
- RDMA read/write。
- GPU buffer staging。
- CUDA kernel launch。
- breakdown 埋点。
- RaBitQ gate。
- credit-aware expansion。

这使得算法优化和系统优化很难独立验证。

### 1.4 protocol 与业务逻辑耦合

协议散落在：

- `service/storage_owner_protocol.hh`
- `ComputeService::RpcHeader`
- memory node peer RPC。
- command protocol。

编码方式多为 memcpy struct + raw payload。

问题：

- 缺少统一 codec。
- 缺少协议版本。
- 很难做 fuzz/test。
- 改字段容易破坏兼容。

### 1.5 配置膨胀

`configuration::IndexConfiguration` 覆盖：

- index 参数。
- RDMA 参数。
- GPU 参数。
- benchmark/服务参数。
- routing 参数。
- RaBitQ 参数。
- storage-owner 参数。
- thread/core 参数。

配置对象传遍全局，导致模块边界不清晰。

### 1.6 scheduler 与 progress 耦合

service scheduler 同时做：

- queue dequeue。
- coroutine resume。
- RDMA CQ poll。
- GPU event poll。
- pause/shutdown。
- breakdown start/finish。

问题：

- 难以单测。
- 难以替换 progress 策略。
- 空转 yield 行为分散。

### 1.7 offline/runtime 格式强耦合

offline writer 和 runtime 都依赖 `VamanaNode` 静态 layout。

改格式要同步：

- shard writer。
- metadata writer。
- metadata loader。
- compute validation。
- memory node load。
- RDMA layout resolver。
- hot graph。
- RaBitQ cache。

这类改动必须放在后期。

## 2. 重构基本原则

1. 不先改 hot path。
2. 不先改 on-disk format。
3. 不先改 RDMA protocol。
4. 先补测试和 observability。
5. 每次只抽一个边界。
6. 新旧实现并存一段时间。
7. 所有性能重构都必须保留 baseline JSON。
8. 所有行为重构都必须有回归测试。

## 3. 阶段 A：可测性和边界固化

目标：

- 不显著改变行为。
- 补足测试。
- 提取纯逻辑。
- 明确不变量。

### A1. 建立 test target

工作：

- 创建 `test/CMakeLists.txt`。
- 添加纯函数测试可执行。
- 接入 CTest。

首批测试：

- vector batch planner。
- partitioning。
- metadata parser。
- vector dtype。
- storage-owner protocol offset helper。

不变量：

- planner 输出不改变。
- partition placement offset 从 16 开始。
- metadata parser 字段语义不变。

### A2. 固化 benchmark baseline

工作：

- 保存标准 service config。
- 保存标准 workload 参数。
- 固定小/中/大数据集。
- 固定 JSON report 输出目录。
- 编写 report compare 脚本。

指标：

- query p50/p95/p99。
- insert p50/p95/p99。
- recall@k。
- RDMA bytes/ops。
- GPU kernel busy ratio。
- throughput。

不变量：

- 优化后 recall 不低于阈值。
- p99 不允许无解释恶化。

### A3. 抽 metadata 纯校验

当前：

- `validate_index_metadata()` 同时 parse、校验、修改 `VamanaNode` 静态状态。

目标：

- 拆成：
  - `load_metadata(...)`
  - `validate_metadata_against_config(...)`
  - `apply_metadata_layout(...)`

收益：

- 可以单测 metadata validation。
- 降低 load index 风险。

不变量：

- 错误 message 不必完全一致，但接受/拒绝结果必须一致。

### A4. 抽 RPC codec

当前：

- `RpcHeader` 是 `ComputeService` 私有结构。
- payload memcpy 散落。

目标：

- 新建 compute routing protocol/codec。
- encode/decode header。
- 校验 magic/type/payload_count。

不改变：

- wire format。
- message size。
- routing 行为。

测试：

- register centroid。
- search request。
- search response。
- invalid magic。
- payload too large。

### A5. 给 service scheduler 增加小型可观测指标

目标：

- 统计每个 worker 空转、pause wait、active coroutine 数分布。
- 不改变调度逻辑。

收益：

- 后续优化 scheduler 有 baseline。

风险：

- 统计开销。

## 4. 阶段 B：低风险模块拆分和配置收敛

目标：

- 拆出模块边界。
- 减少 `ComputeService` 膨胀。
- 不改核心 hot path 行为。

### B1. 拆分 ComputeService 子组件

候选组件：

1. `IndexLifecycleManager`
   - load/store command。
   - metadata validation。
   - startup load/store。

2. `RoutingRpcRuntime`
   - rpc buffer。
   - rpc loop。
   - centroid registration。
   - pending query。

3. `StorageOwnerClientRuntime`
   - storage insert owners。
   - sender/completion loop。
   - response slots。

4. `BreakdownCollector`
   - sample vector。
   - reset/collect report。

5. `ComputeSideIdMap`
   - idmap load。
   - lookup/publish/delete。

拆分方式：

- 第一阶段只移动代码，不改行为。
- 保持 `ComputeService` public API 不变。
- 子组件先作为 private member。

测试：

- 现有 benchmark smoke。
- load/store smoke。
- routing smoke。

### B2. 配置分组

将大配置对象逻辑分组：

- `IndexLayoutConfig`
- `RdmaConfig`
- `GpuConfig`
- `SearchConfig`
- `InsertConfig`
- `RoutingConfig`
- `RabitqConfig`
- `StorageOwnerConfig`
- `WorkerConfig`

第一步不一定改 `IndexConfiguration` 类型，可以先添加 getter 或 view：

- `config_.search_options()`
- `config_.rdma_options()`

收益：

- 减少函数参数污染。
- 新模块只看到需要的配置。

风险：

- 配置默认值和解析逻辑不能改变。

### B3. layout context 只读封装

目标不是马上移除 `VamanaNode` 静态状态，而是先引入只读封装：

```cpp
struct VamanaLayoutView {
  dim;
  R;
  dtype;
  offsets;
  hot_graph;
  rabitq;
};
```

第一阶段：

- 从 `VamanaNode` 静态状态读取。
- 传给新模块作为 const view。

收益：

- 为后续实例化 layout 做准备。
- 让函数签名表达对 layout 的依赖。

不变量：

- 所有 offset/size 与 `VamanaNode` 返回值一致。

### B4. RDMA semantic ops 分层

当前 RDMA helper 已有一定语义封装：

- read medoid。
- read node。
- read vectors。
- allocate node。
- write node。
- spinlock。

可以进一步分为：

1. verb wrapper：
   - QueuePair。
   - MemoryRegion。

2. batch planner：
   - pure plan。

3. Vamana layout-aware operations：
   - read neighbors。
   - read vector。
   - write node。

4. algorithm-facing interface：
   - `GraphReader`
   - `GraphWriter`

先做命名和文件边界，不改实现。

### B5. storage-owner protocol codec

将 `storage_owner_protocol.hh` 中的 offset helper、header、result encoding 拆出 codec 测试。

收益：

- 降低 peer RPC/storage owner runtime 改动风险。
- 为协议版本化准备。

## 5. 阶段 C：高收益性能路径重写

只有阶段 A/B 完成后，才建议进入阶段 C。

### C1. 查询 pipeline 重构

目标：

- 将 Vamana search 拆为：
  - frontier selection。
  - neighbor expansion。
  - candidate scoring。
  - beam update。
  - finalization。

引入接口：

- `NeighborFetcher`
- `VectorFetcher`
- `DistanceEngine`
- `BeamState`
- `SearchMetrics`

收益：

- 可以独立优化 RDMA、GPU、beam。
- 可以替换 RaBitQ gate。
- 可以测试 beam 逻辑。

风险：

- hot path 性能回退。
- coroutine awaitable 逻辑复杂。
- recall 改变。

必须测试：

- exact search baseline recall。
- RaBitQ recall。
- p95/p99。
- RDMA bytes。

### C2. 插入 pipeline 重构

目标：

- 将 insert 拆成：
  - candidate search。
  - prune。
  - allocate/write new node。
  - reverse update。
  - consistency publish。

收益：

- storage-owner 和 compute-side insert 可共享更多逻辑。
- RobustPrune 可独立测试。

风险：

- 图一致性。
- deleted/generation。
- reverse edge 丢失。

必须测试：

- insert 后 query recall。
- upsert/delete freshness。
- concurrent insert consistency。

### C3. routing 多 destination merge

当前 routing 是单 destination。未来可做 fanout：

1. initiator 选择 top M destinations。
2. 并发发送 search request。
3. response 携带 id + distance。
4. initiator merge top-k。
5. 返回 origin。

需要改：

- RPC payload。
- pending query 聚合状态。
- result distance 保留。
- timeout/failure handling。

风险：

- 协议变更。
- 网络流量上升。
- tail latency 上升。

只有在 partition/routing recall 明显不足时才值得做。

### C4. VamanaNode layout 实例化

最终目标：

- 移除 `VamanaNode` 静态全局 layout。
- 将 layout 作为对象传递。

收益：

- 多 index。
- 更好测试。
- load 切换更安全。

风险极高：

- 影响所有 offset。
- 影响 RDMA wrapper。
- 影响 shard writer。
- 影响 memory node。
- 影响 GPU buffer size。
- 影响 RaBitQ/hot graph。

必须分步：

1. 引入 read-only view。
2. 新代码使用 view。
3. 老代码仍读静态。
4. 逐步替换。
5. 最后移除静态。

## 6. 不应立即重构的区域

以下区域不要作为第一批重构对象：

1. RemotePtr raw layout。
   - 影响磁盘、RDMA、idmap、protocol。

2. shard file offset 0/8/16 约定。
   - 影响 memory node load 和 search medoid。

3. storage format schema。
   - 没有 migration 前不要改。

4. RDMA QueuePair 基础封装。
   - 硬件相关，回归成本高。

5. GPU kernel 数值逻辑。
   - 容易引入 recall 或精度变化。

6. storage-owner peer consistency。
   - 并发和跨节点状态复杂。

7. coroutine awaitable 协议。
   - RDMA/GPU pending balance 依赖它。

这些区域可以加测试、加注释、加观测，但不建议先改结构。

## 7. 推荐路线图

### 第 1 阶段：1-2 周

目标：可测性。

工作：

- 建 test target。
- 补 vector planner/partition/metadata/vector dtype 测试。
- 固化 benchmark baseline。
- 增加 report compare 脚本。

完成标准：

- 纯函数测试可本地一键运行。
- baseline report 可复现。

### 第 2 阶段：2-4 周

目标：边界固化。

工作：

- metadata validation 拆分。
- RPC codec 抽出。
- storage-owner protocol codec 测试。
- service scheduler 小重构。

完成标准：

- public API 不变。
- benchmark 无明显回退。
- 单测覆盖新增 codec。

### 第 3 阶段：4-8 周

目标：模块拆分。

工作：

- RoutingRpcRuntime。
- IndexLifecycleManager。
- BreakdownCollector。
- ComputeSideIdMap。
- StorageOwnerClientRuntime。

完成标准：

- `ComputeService` 字段数量明显减少。
- 构造函数时序更清晰。
- benchmark smoke 通过。

### 第 4 阶段：8 周以上

目标：性能路径重写。

工作：

- 查询 pipeline 分层。
- 插入 pipeline 分层。
- RDMA/GPU engine 接口。
- 可选 fanout routing。
- 可选 layout 实例化。

完成标准：

- 性能提升有 JSON 证据。
- recall 保持。
- 新旧路径可对比。

## 8. 每个 PR 的检查清单

1. 是否改变 on-disk format。
2. 是否改变 network protocol。
3. 是否改变 RemotePtr/layout offset。
4. 是否改变 recall。
5. 是否改变 public API。
6. 是否改变 benchmark 可比性。
7. 是否有测试。
8. 是否有 baseline 对比。
9. 是否能回滚。
10. 是否拆分过大。

如果答案中有任何一个“是”，PR 描述必须明确说明。

## 9. 最终学习成果

完成 30 课后，你应该具备以下能力：

1. 从 `main.cc` 追踪 compute/memory node 启动。
2. 理解 RDMA remote pointer 和 Vamana node layout。
3. 看懂 search 的 RDMA/GPU/coroutine 交织。
4. 看懂 insert 的 allocation/prune/reverse update。
5. 理解 storage-owner 模式和 peer update。
6. 看懂 offline builder、partition、shard writer、metadata。
7. 读懂 benchmark JSON。
8. 设计性能优化实验。
9. 判断哪些重构风险高。
10. 制定渐进式重构路线。

## 10. 结课任务

1. 画一张全项目架构图：
   - compute service。
   - worker pool。
   - Vamana。
   - RDMA。
   - GPU。
   - memory node。
   - offline builder。
   - benchmark。

2. 写一份性能优化 proposal：
   - 选择一个 query 或 insert 瓶颈。
   - 给出假设。
   - 给出代码位置。
   - 给出实验设计。
   - 给出失败判据。

3. 写一份重构 proposal：
   - 选择一个低风险边界。
   - 列出行为不变量。
   - 列出测试。
   - 列出迁移步骤。

4. 从源码中找一个你认为设计不合理的点，用本课路线图判断它属于阶段 A、B 还是 C。

