# 第 28 课：测试现状、可测性与安全重构护栏

## 本课目标

本课不是教你写某个具体测试，而是帮助你为后续性能优化和重构建立安全护栏。学完后，你需要能够：

1. 基于代码确认当前测试入口和缺口。
2. 识别哪些逻辑可以先补单元测试。
3. 设计 recall、latency、throughput、consistency 的回归护栏。
4. 给大规模重构制定最小测试集。

代码入口：

- `CMakeLists.txt`
- `src/rdma/vector_batch_planner.hh`
- `tools/vamana_offline/partitioning.hh`
- `tools/vamana_offline/partitioning.cc`
- `tools/vamana_offline/recall_check.hh`
- `tools/vamana_offline/recall_check.cc`
- `src/service/index_metadata.cc`
- `src/http/service_types.hh`

## 1. 当前测试入口

`CMakeLists.txt` 中有：

```cmake
option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)

if (DVSTOR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  enable_testing()
  add_subdirectory(test)
endif()
```

这说明：

- 项目预留了 test 目录接入。
- 但只有当 `test/CMakeLists.txt` 存在时才会启用。
- 如果当前仓库没有顶层 `test/` 目录，则没有实际 CTest 测试被构建。

因此，后续重构前不能假设已有自动化测试保护。必须先补最小测试集。

## 2. 测试金字塔

建议为这个项目建立五层测试：

1. 纯函数单元测试：
   - 不依赖 RDMA。
   - 不依赖 CUDA。
   - 不依赖真实文件。
   - 速度最快。

2. 文件格式测试：
   - metadata JSON。
   - shard writer 小数据集。
   - idmap/anchor/RaBitQ sidecar。

3. 算法小图测试：
   - 离线 VamanaGraph。
   - partition。
   - recall check。

4. 单机集成测试：
   - 本地 memory node + compute node。
   - 小数据集 load/search/insert/upsert/delete。

5. 性能回归测试：
   - benchmark JSON 指标。
   - recall threshold。
   - latency/throughput threshold。

先补前两层，再做大重构。

## 3. 优先补的纯函数测试

### 3.1 vector_batch_planner

`src/rdma/vector_batch_planner.hh` 注释明确写着：

```cpp
Pure planning helper kept independent from verbs so the balancing policy can be tested without RDMA hardware.
```

这是最适合先补的测试。

需要覆盖：

1. 空 request：
   - chunks 空。
   - active_nodes = 0。
   - active_qps = 0。

2. invalid node id：
   - request node 超过 qp_counts size 时被忽略。

3. non-adaptive round robin：
   - 同一 node 的 requests 按 `i % qp_count` 分配。
   - active_qps 正确。
   - request_order 正确。

4. adaptive bulk QP：
   - qp_count > 1 时 QP0 作为 control lane 被跳过。
   - 请求分配到 QP1..N。

5. max_chain_wrs：
   - chunks 的 request_count 不超过 chain limit。
   - plan.max_chain_wrs 正确。

6. outstanding_wrs：
   - adaptive 模式优先选择 projected_load 小的 QP。

7. tie_breaker：
   - 相同 load 下按 tie breaker 改变选择顺序。

这个测试能保护 RDMA batch planner 优化。

### 3.2 partitioning

`tools/vamana_offline/partitioning.cc` 里有多个纯函数：

- `pack_undirected_edge`
- `append_partition_edges`
- `assign_nodes_to_shards_balanced`
- `compute_bfs_partition`
- `assign_nodes_to_shards_from_partition`
- `compute_cross_shard_ratio`

需要覆盖：

1. `pack_undirected_edge(a,b)` 对称。
2. 自环返回 0。
3. balanced placement 从 offset 16 开始。
4. placement offset 按 aligned node size 增加。
5. invalid num_memory_nodes 抛异常。
6. BFS num_parts=1 时所有节点 part=0。
7. BFS 对孤立节点能分配。
8. cross-shard ratio 对简单图计算正确。

这类测试能保护 shard placement 和后续分区优化。

### 3.3 metadata parser

`src/service/index_metadata.cc` 可以用临时 JSON 文件测试：

1. 缺文件返回 false。
2. 必需字段缺失返回 false。
3. `schema_version` default。
4. `vector_data_type` 解析。
5. hot graph arrays 读取。
6. RaBitQ centroid 读取。

更进一步可以测试 `ComputeService::validate_index_metadata()`，但它依赖完整 service，难度更高。先测 parser。

### 3.4 vector dtype

如果要改 dtype 或 raw query 路径，需要测试：

- `parse_vector_dtype`
- `vector_dtype_bytes`
- `decode_storage_vector_to_float`
- `vector_component_as_float`

这些函数通常是纯函数，适合快速覆盖。

## 4. 文件格式测试

文件格式测试建议用小数据集：

- n = 16 或 64。
- dim = 4 或 8。
- R = 4。
- memory nodes = 1 或 2。

测试内容：

1. offline builder 能写出 shard。
2. `.meta.json` 字段完整。
3. shard 0 offset 0 是 shard size。
4. shard 0 offset 8 是 medoid raw pointer。
5. node 从 offset 16 开始。
6. owner idmap entry 数量等于输入数据量。
7. compact layout 下 hot graph offsets 存在。
8. RaBitQ 模式下 sidecar header 与 metadata 一致。

这类测试可在不启动 RDMA 的情况下验证 writer。

## 5. 算法小图测试

离线图适合做算法回归：

1. 构建小 dataset。
2. `build_vamana_graph(...)`。
3. 检查：
   - 每个 degree <= R。
   - 没有自环。
   - 邻居 id 不越界。
   - medoid < n。
4. 执行 `beam_search`。
5. 对小 n 计算 brute-force groundtruth。
6. 验证 recall 不低于低阈值。

这些测试不保证大规模性能，但能防止基础逻辑被重构破坏。

## 6. 单机集成测试

单机集成测试更难，因为依赖：

- RDMA verbs。
- CUDA/GPU。
- memory node 进程。
- compute node 进程。
- 配置文件。

如果环境允许，应建立最小流程：

1. 用 offline builder 生成小 index。
2. 启动 memory node。
3. 启动 compute service。
4. load index。
5. query 10 条。
6. insert 10 条。
7. upsert 几条。
8. delete 几条。
9. store index。
10. 重新 load 并 query。

如果 CI 没有 RDMA/GPU，则把这类测试标记为 hardware/integration，不作为默认必跑。

## 7. 性能回归护栏

性能优化不能只看“跑得更快”。必须同时检查：

1. correctness：
   - recall 不下降。
   - insert/upsert/delete 结果一致。

2. latency：
   - p50/p95/p99。
   - queue wait。
   - service time。

3. throughput：
   - query ops/s。
   - write ops/s。
   - mixed total ops/s。

4. resource：
   - RDMA bytes/ops。
   - H2D/D2H bytes。
   - GPU kernel busy ratio。
   - CPU utilization。
   - memory usage。

5. tail behavior：
   - p99 不应因优化平均值而显著恶化。

建议将每次优化的 benchmark JSON 保存为 artifact，并写一个比较脚本。

## 8. 重构前最小测试集

在做大规模重构前，至少补这些测试：

1. vector batch planner 单元测试。
2. partitioning 单元测试。
3. metadata parser 单元测试。
4. vector dtype decode 单元测试。
5. offline shard writer 小数据集测试。
6. VamanaGraph 小图 invariants。
7. benchmark smoke：
   - 小 workload。
   - 输出 JSON 可解析。
8. 如果硬件允许：
   - load/search/insert smoke。

这些测试不需要覆盖所有性能路径，但能保护基础行为。

## 9. 行为不变量

重构时要明确这些不变量：

1. RemotePtr raw layout：
   - memory node 16 bit。
   - byte offset 48 bit。

2. shard offset：
   - offset 0 free ptr/shard size。
   - offset 8 medoid pointer。
   - offset 16 第一个 node。

3. metadata schema：
   - 当前 schema_version 13。

4. Vamana node layout：
   - vector offset。
   - neighbors offset。
   - hot graph offset。
   - RaBitQ offset。

5. Query result order：
   - 按 distance 排序。
   - public API 返回 id。

6. Insert semantics：
   - insert 新 id。
   - upsert 标记旧 ptr deleted 并发布新 ptr。
   - erase 标记 deleted。

7. routing semantics：
   - 单 destination search。
   - response 只返回 id。

8. breakdown semantics：
   - queue wait、service、end-to-end 的定义保持一致。

## 10. 可测性重构优先级

低风险优先：

1. 给纯函数加 test target。
2. 把 metadata validation 中纯校验逻辑拆出。
3. 给 protocol offset helper 加测试。
4. 给 shard writer 增加小数据 fixture。

中风险：

1. 将 service scheduler 的 request dispatch 提取为可测试小函数。
2. 将 RPC header encode/decode 抽成独立 codec。
3. 将 storage-owner protocol 编码解码抽成测试able 函数。

高风险：

1. 把 `VamanaNode` 静态 layout 变成实例状态。
2. 替换 coroutine scheduler。
3. 改 RDMA QP 管理。
4. 改 RPC response payload 以支持 result merge。

## 11. 学习任务

1. 检查当前仓库是否存在 `test/` 目录，并记录 CMake 行为。
2. 为 `plan_vector_read_batch()` 写出至少 8 个测试用例描述。
3. 为 partitioning 写出输入图和期望输出。
4. 制定一份重构前最小测试清单，按优先级排序。
5. 选择一个你最想优化的路径，为它设计 correctness、latency、throughput 三类回归护栏。

