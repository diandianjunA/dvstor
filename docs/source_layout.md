# 源码结构

## 运行时

| 路径 | 职责 |
|---|---|
| `src/common/` | 配置、公共类型与格式辅助函数 |
| `src/service/compute_service/` | 服务生命周期、索引校验、查询和更新入口 |
| `src/gpu_search/search_engine.hh` | 两种查询后端共用的接口 |
| `src/gpu_search/host_orchestrated_engine/` | CPU 图导航、RDMA 波次和 CUDA 距离计算的编排 |
| `src/gpu_search/persistent_engine/` | Persistent 引擎资源、提交/完成环、路由同步和健康检查 |
| `src/gpu_search/persistent_kernel/` | GPU 候选评分、图遍历、前沿管理、RDMA 读取和精排 |
| `src/gpu/` | DOCA GPUNetIO 传输、QP owner 和硬件探针 |
| `src/memory_node/storage_owner_index/` | 存储分配、图访问、候选搜索和图修改 |
| `src/memory_node/storage_owner_runtime/` | 更新协议、事务执行和 worker 生命周期 |
| `src/memory_node/storage_owner_maintenance/` | Stage2、反向边、迁移和回收任务 |
| `src/memory_node/peer_rpc/` | 存储节点之间的 RPC 请求、响应和 worker |
| `src/vamana/` | 图、质心发布、centroid checkpoint 和 idmap 格式 |

引擎选择位于 `src/service/compute_service/lifecycle.cc`，配置模式解析位于
`src/common/configuration.hh`。修改 baseline/full 时先核对模式解析后的实际参数，
再核对后端实现；实验脚本可以通过 `manual` 模式覆盖下层参数。

CPU 模块使用独立编译的 `.cc` 和声明头 `.hh`。模块内部共享声明放在 `detail.hh`，
查询引擎内部状态放在 `impl.hh`，公开门面负责构造和转发。
CUDA 设备代码由 `persistent_kernel.cu` 编译为单个翻译单元，其 `.cuh` 按职责拆分，
保留设备端内联与常量传播。

## 工具与实验

- `tools/vamana_offline/`：数据读取、Vamana 构图、分片、PQ 编码和格式转换。
- `tools/breakdown_benchmark/`：负载编排、数据集读取、进度和报表。
- `test/`：CPU 单元测试和 CUDA/RDMA 测试；硬件测试需对应运行环境。
- `cmake/`：依赖发现与目标配置。
- `experiment/common/`：数据集共用的绘图、汇总和存储启动实现；从各数据集入口调用。
- `experiment/{sift100m,deep100m,spacev100m}/`：数据/profile 配置、索引准备和实验入口。
- `experiment/mixed_test/`：跨数据集混合负载。

## 修改边界

Persistent 查询热路径使用预分配资源，避免引入动态分配、stream/QP 创建或 CPU 同步导航。
HostOrchestrated 独立维护 CPU 导航流程；两者均不做隐式后端回退。
图、向量、ID authority 和 mutation 状态归存储节点，计算侧只持有查询状态与可验证缓存。

索引、centroid publication 和 RPC wire protocol 的布局由显式 schema/version 保护。
结构调整应保留公开接口、序列化布局、排序和浮点语义；涉及查询热路径时核对 GPU 资源占用、
召回率和端到端性能。执行协议见 [查询与动态更新](gpu_persistent_engine.md)。
