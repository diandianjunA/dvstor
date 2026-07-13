# DVSTOR

DVSTOR 是面向动态向量检索的存算分离系统。`dev` 分支只保留一条查询路径：
GPU 常驻 OPQ/PQ32 导航码，持久化 CUDA kernel 维护查询状态，并通过 DOCA
GPUNetIO 直接读取存储节点上的紧凑图记录与精确向量。CPU 仅负责请求准入、
启动阶段批量传输、控制面和动态更新 RPC，不参与稳态图遍历。

历史实现保存在 `main` 分支；`dev` 不提供旧查询引擎或 CPU 查询回退。

## 查询路径

1. CPU 将请求写入有界提交队列，并按微批分配 GPU 查询槽。
2. 持久化 GPU block 完成 OPQ 变换并构建 PQ 查找表。
3. GPU 对常驻入口点和动态 delta 入口进行打分，初始化 beam。
4. GPU 通过 GPUNetIO 并发拉取远端 512 字节以内的紧凑图记录。
5. GPU 使用常驻 PQ code 对邻居做近似距离计算并更新 beam。
6. GPU 仅为最终候选拉取精确向量，计算 L2 距离并返回 top-k。

查询过程中没有 CPU 驱动的逐轮 RDMA、主机向量中转或本地图副本。GPUNetIO、
持久化 kernel 或更新发布发生系统性错误时，引擎进入 fail-stop 状态，不会把
降级路径的结果伪装成 GPU 查询结果。

## 动态更新

存储节点继续拥有 insert、upsert 和 delete 的图维护协议：

- storage owner 负责 idmap、代际、紧凑图记录和反向边维护；
- 提交成功的 mutation 以 epoch 批次发布到 GPU delta；
- CPU 将原始存储格式向量写入映射固定内存，专用常驻 control CTA 批量完成
  OPQ/PQ 编码、hash/bucket 链接和 epoch 发布；
- GPU delta 保存原始精确向量、PQ code、删除标记和动态候选桶；
- 基础图缓存保持不可变，动态可见性不依赖逐写缓存失效；
- delta 超过容量或维护失败时停止接收新查询，避免静默返回陈旧结果。

## 索引文件

新索引固定为 schema 14、L2、`plain` vector record、compact graph 和
OPQ/PQ 导航。默认 profile 使用 32 个 8-bit 子空间。运行时不读取任何计算节点图清单文件。

| 文件 | 计算节点 | 存储节点 X | 作用 |
| --- | --- | --- | --- |
| `<prefix>.meta.json` | 必需 | 必需 | 分片、远端 offset 和格式契约 |
| `<prefix>.pq32` | 必需 | 不需要 | OPQ 矩阵与 PQ codebook |
| `<prefix>.anchors` | 必需 | 必需 | 查询入口与动态更新 anchor |
| `<prefix>_nodeX_ofN.dat` | 不需要 | 必需 | 精确向量、固定记录和紧凑图 |
| `<prefix>_nodeX_ofN.idmap` | 不需要 | 必需 | storage-owner ID 映射 |
| `<prefix>_nodeX_ofN.pq32.codes` | 不需要 | 必需 | 启动时注册到远端内存的 PQ32 码流 |

计算节点本地只保存 metadata、PQ 模型和 anchors；不会保存 `.gpu.idx`、图分片、
精确向量或全量导航码。

PQ32 每个向量占 32 字节：SIFT100M 为 3.2 GB，SIFT1B 为 32 GB
（约 29.8 GiB）。预算器优先保留 PQ 与 2 GiB delta，再按剩余显存自适应缩放
图缓存和精确向量缓存；显式分配上限为 36 GiB，并为 CUDA/DOCA 保留 4 GiB。
计算节点本地文件远低于 50 GB。

## 依赖与构建

计算节点需要 C++20、CUDA、RDMA verbs、DOCA GPUNetIO、Boost、TBB、OpenMP
和 Faiss；离线 METIS 分片可选。默认 DOCA 路径为 `/opt/mellanox/doca`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

主要目标：

- `dvstor_compute_node`：GPU 查询与更新客户端；
- `dvstor_memory_node`：存储节点；
- `dvstor_breakdown_benchmark`：吞吐、延迟、召回率和分解统计；
- `vamana_offline_builder`：在 CPU 上构建 compact Vamana 图，并执行
  `balanced`、`bfs` 或可选的 `metis` 分片；
- `vamana_pq_indexer`：训练 OPQ/PQ 并生成分片码流；
- `vamana_legacy_index_converter`：复用旧图与分片布局进行迁移。

无 GPU 的存储节点可同时构建存储服务和 CPU 离线索引工具：

```bash
cmake -S . -B build-storage \
  -DCMAKE_BUILD_TYPE=Release \
  -DDVSTOR_STORAGE_NODE_ONLY=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
cmake --build build-storage -j --target \
  dvstor_memory_node vamana_offline_builder vamana_anchor_sidecar_builder \
  vamana_pq_indexer vamana_legacy_index_converter
```

存储服务本身只依赖 CPU、RDMA、Boost 和 TBB。离线工具使用 CPU Faiss、BLAS、
LAPACK、OpenMP 和可选 METIS，但不依赖 CUDA 或 DOCA。CMake 直接链接
`libfaiss` CPU 库，不加载可能包含 `faiss_gpu_objs` 的 Faiss CMake 导出。
动态更新运行时使用标准 C++20 coroutine 和 `atomic_ref`，因此要求 GCC 11+
或等价的现代 Clang。若节点只运行存储服务，可增加
`-DDVSTOR_BUILD_OFFLINE_TOOLS=OFF`。

必须使用独立的 `build-storage` 目录；不要把已经配置为 CUDA 计算节点的 `build`
目录切换成存储模式。`DVSTOR_METIS_PARTITION=AUTO` 会先验证 METIS/GKlib 能否
在本机真实链接；若仓库内预编译库与本机 glibc 不兼容，会先回退到系统 CPU
METIS，没有兼容版本时才保留 `balanced/bfs` 并禁用 `metis`。需要强制 METIS
时，安装本机 CPU 版本并设置
`-DDVSTOR_METIS_ROOT=/path/to/metis -DDVSTOR_METIS_PARTITION=ON`。

新建索引的分片由 `vamana_offline_builder --partition-strategy ...` 完成，不需要
GPU，也不需要额外重分片可执行文件。`vamana_legacy_index_converter` 的职责是
低成本复用旧图，因此保持旧索引的分片数与节点布局；已删除的 schema-13
`vamana_bfs_repartitioner`/`vamana_metis_repartitioner` 依赖旧 RaBitQ 记录格式，
不能用于 schema 14。若要改变旧索引的分片布局，应先在旧格式下完成分片，再
执行迁移；不要把旧重分片器重新链接进新运行时。

## 生成或迁移索引

完整构建：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

该命令直接产出 schema-14 compact 分片、owner idmap、anchors、OPQ/PQ32 模型和
每分片 PQ32 码流，不需要再运行重编码脚本。默认目标已存在时脚本会在昂贵构建前
拒绝覆盖；建议通过 `PQ_INDEX_PREFIX=/new/prefix` 构建新版本，确认需要原地重建时
才设置 `OVERWRITE_INDEX=1`。覆盖模式会先解除目标 prefix 下可能存在的迁移软链接，
避免误写其源分片。

复用已有 Vamana/Metis 图，不重新构图或分区：

```bash
SOURCE_PREFIX=/path/to/legacy/index_prefix \
./experiment/convert_legacy_sift100m_index.sh 04_gpu_persistent_gpunetio
```

已有 schema-14 PQ16 索引时，只重编码为默认 PQ32，不复制或重建图：

```bash
./experiment/reencode_sift100m_pq.sh 04_gpu_persistent_gpunetio
```

迁移会顺序压缩旧 fixed record、重写紧凑图中的 RemotePtr、迁移 idmap 与
anchors；脚本随后在独立进程中训练/复用 PQ 模型并编码所有向量。两个阶段以
schema-14 metadata 为持久化检查点：迁移完成后 PQ 失败，重新执行脚本不会再次
迁移 65 GB 分片。`PQ_THREADS` 默认 32，CPU BLAS 固定为非嵌套运行；若已有不完整
的 `.pq32`/`.pq32.codes`，使用 `OVERWRITE_PQ=1` 重做 PQ 阶段。源 prefix 与输出
prefix 必须不同；迁移器不会修改或删除旧索引。

## 运行 SIFT100M

先在 `experiment/profiles/04_gpu_persistent_gpunetio.env` 或环境变量中配置
`HOSTS`、`INDEX_DIR`、GPU 与内存预算。在存储节点启动对应分片后，在计算节点运行：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_recall.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

Benchmark 并发使用独立的 `BENCHMARK_CLIENT_THREADS`，不写入索引/系统 profile；
例如 `BENCHMARK_CLIENT_THREADS=128 ./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio`。
标准的 10K `query.u8bin` 只用于 recall 检查。吞吐阶段使用独立的
`PERFORMANCE_QUERY_FILE`，并从 warmup 到 measure 单遍消费；文件耗尽会直接失败，
不会回绕重复。默认性能查询集是 `bigann_base.bvecs` 的 `[100M,103M)`，插入集是
不重叠的 `[103M,105M)`；生成的 `.u8bin` 文件位于 SIFT1B 数据集目录。

分布式部署时，每台存储节点只需其自身的 `.dat`、`.idmap`、`.pq32.codes`，
再加共享 metadata 与 anchors。详细流程见 `experiment/README.md`。

## 验证

```bash
ctest --test-dir build --output-on-failure
bash -n experiment/*.sh experiment/profiles/*.env
```

`gpu_memory_budget_test` 覆盖 SIFT100M/SIFT1B 预算；格式、PQ、delta、迁移和
提交环均有独立测试。硬件吞吐和召回率仍需在真实存储节点启动后通过 profile
进行验证。

## 代码结构

- `src/gpu_search/`：PQ 模型、索引布局、持久化引擎与 CUDA kernel；
- `src/gpu/`：DOCA GPUNetIO 传输和探针；
- `src/memory_node/`：分片加载、更新、维护与 storage peer RPC；
- `src/service/`：计算服务、索引契约和统计；
- `src/vamana/`：compact graph、anchor 和 idmap 格式；
- `tools/legacy_index/`：隔离的旧索引只读迁移器；
- `tools/vamana_offline/`：离线构图与 PQ sidecar 生成；
- `experiment/`：唯一支持的 SIFT100M profile。
