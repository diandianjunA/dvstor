# DVSTOR

DVSTOR 是面向动态向量检索的存算分离系统。存储节点维护 Vamana 图、精确向量和更新状态，
计算节点通过 RDMA 访问索引，使用 OPQ/PQ 导航和原始向量 L2 精排。

项目包含两种查询引擎，共用 schema-16 索引：

| 查询推进模式 | 引擎 | 执行方式 |
|---|---|---|
| `coupled` | HostOrchestrated | CPU 维护 beam/visited、提交和轮询 RDMA，有限生命周期 CUDA kernel 计算距离 |
| `decoupled` | Persistent | GPU 常驻查询状态，通过 DOCA GPUNetIO 读取远端记录，并提前发起下一轮图读取 |
| `manual` | Persistent | 用于单独控制提前读取、批量宽度等实验参数 |

正式 baseline/full profile 分别使用前两种模式；动机和消融脚本会覆盖部分模式，具体见
[实验说明](experiment/README.md)。HostOrchestrated 是显式选择的后端，运行失败不会触发引擎切换。

## 构建

需要 CMake 3.18+、支持 C++20 的编译器（GCC 11+）、AVX2 CPU、RDMA verbs、Boost 和 TBB。
计算节点还需要 CUDA 11.8+ 和 DOCA GPUNetIO，DOCA 默认安装路径为 `/opt/mellanox/doca`。
离线工具需要 CPU Faiss、BLAS、LAPACK 和 OpenMP，METIS 为可选分片依赖。

计算节点构建：

```bash
cmake -S . -B build-compute -DCMAKE_BUILD_TYPE=Release
cmake --build build-compute -j16
```

无 GPU 的存储/索引构建节点：

```bash
cmake -S . -B build-storage -DCMAKE_BUILD_TYPE=Release \
  -DDVSTOR_STORAGE_NODE_ONLY=ON
cmake --build build-storage -j16
```

仅运行存储服务时，可再设置 `-DDVSTOR_BUILD_OFFLINE_TOOLS=OFF`，无需安装 Faiss、CUDA 或 DOCA。
计算和存储构建使用各自的目录；实验脚本通过 `BUILD_DIR`、`STORAGE_BUILD_DIR` 选择对应产物。

`DVSTOR_METIS_PARTITION=AUTO` 会检查 METIS/GKlib 的实际链接兼容性，并尝试回退到系统库。
需要 METIS 时设置 `-DDVSTOR_METIS_PARTITION=ON`；自定义安装路径使用
`-DDVSTOR_METIS_ROOT=/path/to/metis`。没有兼容 METIS 时仍可使用 `balanced` 或 `bfs` 分片。

| 可执行文件 | 用途 |
|---|---|
| `dvstor_compute_node` | 计算服务与更新客户端 |
| `dvstor_memory_node` | 存储服务 |
| `dvstor_breakdown_benchmark` | 查询/更新负载、吞吐、延迟、召回率与分解统计 |
| `vamana_offline_builder` | CPU 构图和物理分片 |
| `vamana_pq_indexer` | OPQ/PQ 训练、编码和 schema-16 产物生成 |
| `vamana_graph_extent_indexer` | 校验图分片并生成读取长度 sidecar |
| `vamana_metis_repartitioner` | 保留逻辑图，将现有索引重新做 METIS 物理分片 |
| `vamana_legacy_index_converter` | 将支持的 schema-15 旧索引转换为 schema 16 |

## 索引与部署

运行索引使用 schema 16、L2、`plain` 精确向量记录和 incarnation-tagged 紧凑图。
`M` 表示 PQ 子空间数：SIFT100M、Deep100M 使用 32，SPACEV100M 使用 20。

| 文件 | 计算节点 | 存储节点 X | 内容 |
|---|---|---|---|
| `<prefix>.meta.json` | 必需 | 必需 | 格式、分片、远端 offset 和构建指纹 |
| `<prefix>.pqM` | 必需 | 无需 | OPQ 矩阵和 PQ codebook |
| `<prefix>.gextent8` | 正式 profile 必需 | 无需 | 每个基础节点一个字节的图读取长度档 |
| `<prefix>_nodeX_ofN.dat` | 无需 | 必需 | 精确向量和图记录 |
| `<prefix>_nodeX_ofN.idmap` | 无需 | 必需 | 该 authority 的 ID、代际和物理目录 |
| `<prefix>_nodeX_ofN.centroid` | 无需 | 必需 | 物理分片中心状态与启动入口 |
| `<prefix>_nodeX_ofN.pqM.codes` | 无需 | 必需 | 注册到远端内存的基础 PQ 码流 |

`pqM` 是文件名示意，例如 `.pq32` 或 `.pq20`。metadata、分片和 sidecar 必须来自同一次构建，
启动时会校验格式、尺寸、指纹和 checksum。计算 GPU 从存储节点导入基础 PQ 码流；图、
精确向量和 ID 目录保留在存储侧。Persistent 按物理槽位和 incarnation 在 GPU 中缓存动态
PQ 数据；HostOrchestrated 每次读取动态 PQ 到主机 scratch，再复制到对应 CUDA lane。
权威状态均由存储节点持有。

首次运行先按数据集说明准备数据和索引，再启动实验入口。以 SIFT100M 为例：

```bash
BUILD_DIR="$PWD/build-storage" ./experiment/sift100m/build_sift100m_index.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program1/run_program1.sh
```

实验脚本在每个 case 前打印存储节点命令并等待就绪。数据路径、节点地址和运行参数需按机器
配置，详见 [SIFT100M](experiment/sift100m/README.md)、[Deep100M](experiment/deep100m/README.md)
和 [SPACEV100M](experiment/spacev100m/README.md)。三者均提供 Program 1–3；SIFT 另有累积消融，
跨数据集混合负载见 [mixed_test](experiment/mixed_test/README.md)。

## 验证与开发

```bash
ctest --test-dir build-storage --output-on-failure
ctest --test-dir build-compute --output-on-failure
```

本地测试覆盖格式、PQ、路由、更新协议、内存预算与查询策略。吞吐、召回率和 GPU/RDMA
运行情况需要在实际存储节点和计算节点上验证，测试通过不能代替硬件实验。

- [源码结构](docs/source_layout.md)：模块职责和修改边界。
- [查询与动态更新](docs/gpu_persistent_engine.md)：执行流程、路由、一致性和故障处理。
- [Live-Extent RDMA](docs/live_extent_rdma.md)：读取长度、回退和统计口径。
- [实验说明](experiment/README.md)：数据准备、Program 1–3 和结果复用。
