# Source Layout

## 模块边界

运行时按职责划分，而不是按历史功能堆叠：

- `src/gpu_search/persistent_engine.cc`：GPU 查询引擎 PImpl 的装配入口；
- `src/gpu_search/persistent_engine/`：构造、生命周期、查询、路由、增量发布、
  存储回收、完成处理和状态；
- `src/gpu_search/persistent_kernel.cu`：单一持久化 CUDA translation unit；
- `src/gpu_search/persistent_kernel/`：PQ 评分、RDMA/cache、图遍历和 kernel runtime；
- `src/memory_node/storage_owner_index/`：存储分配、图访问、候选搜索和图修改；
- `src/memory_node/peer_rpc/`：peer RPC 生命周期、请求处理、worker 和客户端请求；
- `src/memory_node/storage_owner_maintenance/`：维护队列、worker 和图任务；
- `src/memory_node/storage_owner_runtime/`：更新 runtime 生命周期、批执行和 wire protocol；
- `src/service/compute_service/storage_owner/`：计算侧更新入口、发送和完成处理；
- `tools/breakdown_benchmark/`：数据集、进度、报表和工作负载编排。

## 同编译单元拆分

查询和更新热路径使用 `.ipp`/`.cuh` 责任片段，由一个 `.cc`/`.cu` 聚合编译。这一
选择是有意的：既把数千行实现拆成可导航模块，又不增加跨 translation unit 调用、
不暴露 PImpl 私有状态，也不阻断编译器对热路径的内联和常量传播。

新代码应放入最小责任片段。聚合文件只保留 include、共享类型和装配顺序；不得把
业务实现重新写回聚合文件。只有需要跨 subsystem 复用且拥有稳定接口的能力，才
提取为普通 `.hh`/`.cc`，例如 `mapped_ring.hh`、benchmark dataset 和 report。

## 动态更新语义

GPU delta 是可见性 overlay，不是第二套静态索引，也不存在虚假的在线 base
compaction API。存储维护完成后，计算节点依据 durable watermark 和 query ticket
安全退休旧 overlay 记录；遥测使用 `delta_reclaim_batches` 表达该动作。静态
generation 的重建与切换属于独立的离线/控制面过程。

## 变更约束

- 不在查询热路径分配内存、创建 stream、创建 QP 或同步等待 CPU fallback；
- 不新增 CPU 图导航或隐式传输回退；
- 索引格式和 RPC wire protocol 必须由显式 schema/version 保护；
- 结构重构优先保持同一编译单元和公开 ABI；
- 修改职责边界时同步更新单元测试与本文档。
