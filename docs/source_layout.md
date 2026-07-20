# Source Layout

## 模块边界

运行时按职责划分，而不是按历史功能堆叠：

- `src/gpu_search/persistent_engine.cc`：GPU 查询引擎 PImpl 的装配入口；
- `src/gpu_search/persistent_engine/`：构造、生命周期、查询、路由、增量发布、
  存储回收、完成处理和状态；
- `src/gpu_search/persistent_kernel.cu`：单一持久化 CUDA translation unit；
- `src/gpu_search/persistent_kernel/`：PQ 评分、GPUNetIO/RDMA 读取、anchor 路由记录、
  图遍历和 kernel runtime；
- `src/memory_node/storage_owner_index/`：存储分配、图访问、候选搜索和图修改；
- `src/memory_node/peer_rpc/`：peer RPC 生命周期、请求处理、worker 和客户端请求；
- `src/memory_node/storage_owner_maintenance/`：维护队列、worker 和图任务；
- `src/memory_node/storage_owner_runtime/`：更新 runtime 生命周期、批执行和 wire protocol；
- `src/service/compute_service/storage_owner/`：计算侧更新入口、发送和完成处理；
- `tools/breakdown_benchmark/`：数据集、进度、报表和工作负载编排。

## 实现单元

CPU 运行时使用普通 `.hh`/`.cc`：每个 `.cc` 是可独立编译、可被 clangd 直接解析的
职责单元。模块共享的非公开声明放在本目录的 `detail.hh`；GPU 查询引擎通过
`impl.hh` 定义 PImpl 状态，公开门面只保留装配和转发。项目不使用 `.ipp` 文本片段，
也不依赖 `__INCLUDE_LEVEL__` 改变源码语义。

CUDA 设备代码仍由 `persistent_kernel.cu` 形成一个 translation unit，以保留设备端
内联与常量传播；目录内 `.cuh` 是带 `#pragma once`、显式依赖和命名空间的正常 CUDA
头文件，而不是可独立切换行为的 include 片段。

## 动态更新语义

GPU delta 是可见性 overlay，不是第二套静态索引，也不存在虚假的在线 base
compaction API。存储维护完成后，计算节点依据 durable watermark 和 query ticket
安全退休旧 overlay 记录；遥测使用 `delta_reclaim_batches` 表达该动作。静态
generation 的重建与切换属于独立的离线/控制面过程。

## 变更约束

- 不在查询热路径分配内存、创建 stream、创建 QP 或同步等待 CPU fallback；
- 不新增 CPU 图导航或隐式传输回退；
- 索引格式和 RPC wire protocol 必须由显式 schema/version 保护；
- 结构重构保持公开 ABI，并通过构建与性能测试确认跨实现单元边界；
- 修改职责边界时同步更新单元测试与本文档。
