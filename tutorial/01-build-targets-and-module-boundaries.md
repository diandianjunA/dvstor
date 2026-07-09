# 第 01 课：从构建系统恢复真实模块边界

## 本课目标

学完本课后，你应该能够不依赖 README，而是直接从 `CMakeLists.txt` 判断这个项目有哪些可执行程序、哪些库、哪些源码属于在线服务、哪些源码属于离线构建、哪些源码只在 GPU 版本中参与编译。

本课的核心方法是：先看构建目标，再看入口文件，最后再看目录名。目录名可能表达设计意图，但构建目标才决定真实二进制边界。

## 代码证据

必须阅读：

- `CMakeLists.txt`
- `cmake/DvstorDependencies.cmake`
- `cmake/DvstorTargetHelpers.cmake`
- `src/main.cc`
- `src/memory_node_main.cc`

辅助阅读：

- `src/service/compute_service.cc`
- `src/memory_node/memory_node.cc`
- `tools/vamana_offline_builder.cc`
- `tools/dvstor_breakdown_benchmark.cc`

## 先从 target 看项目

顶层 `CMakeLists.txt` 定义了几个关键选项：

- `DVSTOR_STORAGE_NODE_ONLY`
- `DVSTOR_BUILD_EXECUTABLES`
- `DVSTOR_BUILD_TESTS`
- `DVSTOR_USE_NATIVE_ARCH`
- `DVSTOR_METIS_PARTITION`

这些选项比目录结构更重要。比如 `src/gpu/` 明明存在，但当 `DVSTOR_STORAGE_NODE_ONLY=ON` 时，项目只以 `CXX` 语言配置，不启用 CUDA，也不会构建 `dvstor_runtime` 中的 GPU 路径。

真实 target 可以分为五类：

| target | 类型 | 主要用途 | 是否依赖 CUDA |
| --- | --- | --- | --- |
| `rdma_library` | library | RDMA verbs 封装、连接管理、内存注册 | 否 |
| `dvstor_gpu_kernels` | static library | `.cu` kernel 编译产物 | 是 |
| `dvstor_runtime` | static library | compute node 在线运行时 | 是 |
| `dvstor` | executable | 在线 compute node 服务入口 | 是 |
| `dvstor_memory_node` | executable | memory node 服务入口 | 否 |
| `vamana_offline_builder` | executable | 离线 Vamana 图构建和 shard 写出 | 否 |
| `vamana_metis_repartitioner` | executable | 基于 METIS 重分区 | 否 |
| `vamana_bfs_repartitioner` | executable | 基于 BFS 重分区 | 否 |
| `vamana_anchor_sidecar_builder` | executable | anchor sidecar 构建 | 否 |
| `vamana_rabitq_sidecar_converter` | executable | RaBitQ sidecar 转换 | 否 |
| `generate_sift101m_recall_data` | executable | 数据生成工具 | 否 |
| `dvstor_breakdown_benchmark` | executable | 调用 compute service 的性能拆解工具 | 是 |

注意：`dvstor_memory_node` 在 storage-node-only 和完整构建模式下都会生成。它依赖 `rdma_library`，但不依赖 CUDA。

## 源码分组

`CMakeLists.txt` 中手工列出了几个源码集合。

`DVSTOR_MEMORY_NODE_SOURCES` 包括：

- `src/memory_node/memory_node.cc`
- `src/memory_node/peer_rdma.cc`
- `src/memory_node/peer_rpc.cc`
- `src/memory_node/storage_owner_anchor.cc`
- `src/memory_node/storage_owner_index.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/memory_node/storage_owner_runtime.cc`
- `src/vamana/anchor_index.cc`

这说明 memory node 并不是简单的被动 RDMA 内存池，它还拥有 storage-owner 插入、peer RDMA/RPC、anchor 本地搜索和维护逻辑。

`DVSTOR_OFFLINE_BUILDER_SOURCES` 包括：

- `tools/vamana_offline/anchor_builder.cc`
- `tools/vamana_offline/config.cc`
- `tools/vamana_offline/dataset_io.cc`
- `tools/vamana_offline/graph.cc`
- `tools/vamana_offline/partitioning.cc`
- `tools/vamana_offline/progress.cc`
- `tools/vamana_offline/recall_check.cc`
- `tools/vamana_offline/shard_writer.cc`

这说明离线构建器是独立于在线 `Vamana<Distance>` 实现的另一套 CPU 构建路径。后续读代码时不要误以为在线插入和离线构建共享同一套 `Vamana::insert`。

`DVSTOR_RUNTIME_SOURCES` 通过 `file(GLOB_RECURSE src/*.cc)` 自动收集，然后排除：

- `src/main.cc`
- `src/memory_node_main.cc`

因此完整构建下，`dvstor_runtime` 会把 `src/service/`、`src/vamana/`、`src/rdma/`、`src/gpu/*.cc`、`src/router/`、`src/http/`、部分 `src/memory_node/` 实现一起编入一个大静态库。

## 编译模式对架构理解的影响

完整模式：

```text
project(rdma-vamana CXX CUDA)
dvstor_runtime -> rdma_library + CUDA::cudart + dvstor_gpu_kernels
dvstor -> dvstor_runtime
dvstor_breakdown_benchmark -> dvstor_runtime
dvstor_memory_node -> rdma_library
```

storage-node-only 模式：

```text
project(rdma-vamana CXX)
dvstor_memory_node -> rdma_library
vamana_offline_builder -> rdma_library
其他离线工具 -> rdma_library 或普通 C++
```

这个模式说明：memory node 的代码必须在没有 CUDA 编译器、没有 GPU 的机器上工作；compute node 的查询和在线普通插入则强绑定 GPU buffer 和 kernel launcher。

## 目录结构和真实边界的差异

不要简单把目录当成模块边界。举例：

- `src/vamana/anchor_index.cc` 被 memory node target 使用，也被 compute side 的 local-stitch routing 使用。
- `src/memory_node/` 下的 storage-owner 代码不是只服务 `dvstor_memory_node`，它还定义了和 compute side RPC 协议紧密相关的行为。
- `src/rdma/` 不是底层 RDMA library，而是 Vamana 索引语义层的 RDMA 操作；真正 verbs 封装在 `rdma-library/library/`。
- `tools/vamana_offline/` 的 `graph.cc` 和 `src/vamana/vamana_insert.ipp` 都有 Vamana 构建逻辑，但实现完全不同。

## 你应该建立的第一张架构图

建议画成如下层级：

```text
可执行程序
  dvstor
    src/main.cc
    ComputeService<L2Distance/IPDistance>
    dvstor_runtime

  dvstor_memory_node
    src/memory_node_main.cc
    MemoryNode
    rdma_library

  vamana_offline_builder
    tools/vamana_offline_builder.cc
    tools/vamana_offline/*

  dvstor_breakdown_benchmark
    tools/dvstor_breakdown_benchmark.cc
    ComputeService<L2Distance/IPDistance>

运行时库
  dvstor_runtime
    service
    vamana
    rdma semantic wrappers
    gpu managers
    router
    http scheduler

底层库
  rdma_library
    Context
    QueuePair
    ConnectionManager
    MemoryRegion
```

## 关键设计事实

1. 在线 compute service 和 benchmark 共用 `dvstor_runtime`。因此 benchmark 不是外部黑盒，它直接构造 `ComputeService`。
2. memory node target 没有链接 `dvstor_runtime`，所以 memory node 不能直接调用 compute-side `Vamana<Distance>`。
3. `src/rdma/vamana_rdma_*` 是索引语义层，依赖 `ComputeThread`、`VamanaNode`、`RemotePtr`。
4. `rdma-library/library/*` 是 verbs 封装层，不知道 Vamana 图索引。
5. 离线 builder 用 `tools/vamana_offline/VamanaGraph`，不是 `src/vamana/Vamana<Distance>`。

## 性能影响

构建边界直接影响性能分析方式：

- `dvstor_runtime` 静态链接了大量模板和 `.ipp`，编译器能内联，但也扩大了二进制和编译时间。
- `dvstor_gpu_kernels` 单独作为 static library 编译，host 侧通过 `gpu_kernel_launcher.hh` 调用，不把 CUDA 类型暴露给所有 C++ 文件。
- `dvstor_memory_node` 不依赖 CUDA，memory node 性能主要来自 RDMA、CPU、内存布局和 peer RPC。
- 离线构建器不依赖 GPU，构建性能主要由 CPU distance、图锁、候选剪枝和 shard 写出决定。

## 设计异味

本课只从构建系统能看到几个明显风险：

1. `dvstor_runtime` 通过 glob 收集 `src/*.cc`，新增 `.cc` 文件时容易被意外编入 runtime。
2. `DVSTOR_MEMORY_NODE_SOURCES` 和 runtime glob 之间可能重复包含部分 memory node 语义，需要持续检查目标边界。
3. `CMakeLists.txt` 同时承担构建、功能开关、工具定义、第三方链接，规模已经偏大。
4. `src/vamana/*.ipp` 通过 include 拼进模板类，编译边界不清晰，改动容易触发大范围重编译。
5. `DVSTOR_CONDA_LIB_DIR` 带有本地开发机路径，这是可移植性风险。

## 可验证问题

阅读本课代码后，应该能回答：

- `dvstor_memory_node` 是否依赖 CUDA？为什么？
- `dvstor_breakdown_benchmark` 是黑盒压测还是直接调用服务对象？
- 在线插入和离线构建是否使用同一套 Vamana 实现？
- `rdma-library` 和 `src/rdma` 的职责有什么区别？
- storage-node-only 构建下会生成哪些工具？

## 学习任务

1. 用 `cmake --build build --target help` 查看当前 build 目录实际 target，并和本课表格对照。
2. 在纸上画出 `dvstor`、`dvstor_memory_node`、`vamana_offline_builder` 三个可执行程序的源码依赖边界。
3. 阅读 `CMakeLists.txt` 中 `DVSTOR_STORAGE_NODE_ONLY` 分支，标出哪些源码只在完整模式出现。
4. 找出所有通过 `add_executable` 生成的工具，写下每个工具的入口 `.cc` 文件。
5. 思考：如果要把 GPU 查询路径拆成独立库，需要移动哪些 target 依赖？

