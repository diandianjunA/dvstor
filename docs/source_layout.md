# Source Layout

## 模块边界

运行时按职责划分，而不是按历史功能堆叠：

- `src/gpu_search/persistent_engine.cc`：GPU 查询引擎 PImpl 的装配入口；
- `src/gpu_search/persistent_engine/`：构造、生命周期、查询、路由、质心发布、
  存储回收、完成处理和状态；
- `src/gpu_search/persistent_kernel.cu`：单一持久化 CUDA translation unit；
- `src/gpu_search/persistent_kernel/`：PQ 评分、GPUNetIO/RDMA 读取、版本化 centroid
  route 快照、图遍历和 kernel runtime；
- `src/memory_node/storage_owner_index/`：存储分配、图访问、候选搜索和图修改；
- `src/memory_node/peer_rpc/`：peer RPC 生命周期、请求处理、worker 和客户端请求；
- `src/memory_node/storage_owner_maintenance/`：维护队列、worker 和图任务；
- `src/memory_node/storage_owner_runtime/`：更新 runtime 生命周期、批执行和 wire protocol；
- `src/service/compute_service/storage_owner/`：计算侧更新入口、发送和完成处理；
- `src/vamana/centroid_router.*`：物理分片补偿式 FP64 sum/count、1--4 个 live entries 与
  immutable publication；
- `src/vamana/centroid_state.hh`：离线 `.centroid` checkpoint 的版本化格式；
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

存储节点维护唯一的权威动态图。Stage1 完成本地插入和本地反向边后，新节点即可
通过正常图遍历被查询；Stage2 延续同一搜索上下文完成远端候选、最终剪枝、物理
home 选择和最终反向边。计算节点不维护独立的更新 overlay，也不复制一份动态
索引状态。动态查询采用 incarnation-tagged read-committed 语义：旧存储记录在
durable watermark 后回收，复用时递增 incarnation 并最后发布新 header；系统不提供
跨计算节点 snapshot-RCU。静态 generation 的重建与切换属于独立的离线/控制面过程。

路由状态与 ID authority 分离。每个物理分片从 `.centroid` checkpoint 恢复精确
sum/count 和真实图入口，mutation 完成物理 membership 变更后由 `CentroidRouter`
批量产生新的 immutable snapshot。存储节点再通过带 descriptor、magic/version、
checksum 和 sequence bracket 的 variable-length publication 暴露该版本。计算节点
从同一组 storage-canonical publication 构造 CPU physical-home selector，并通过
control CTA 更新 GPU centroid route；查询只消费完整的 per-shard transaction。

在线布局合成不读取 metadata 中的 medoid 或离线采样 entry-point，也不生成独立
静态路由表。首份完整 centroid route 安装失败时查询引擎 fail-stop，
不会退回另一套离线路由或 CPU 导航。

## 变更约束

- 不在查询热路径分配内存、创建 stream、创建 QP 或同步等待 CPU fallback；
- 不新增 CPU 图导航或隐式传输回退；
- 索引格式、centroid publication 和 RPC wire protocol 必须由显式 schema/version
  保护；
- 结构重构保持公开 ABI，并通过构建与性能测试确认跨实现单元边界；
- 修改职责边界时同步更新单元测试与本文档。
