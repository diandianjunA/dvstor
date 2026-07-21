# DVSTOR

DVSTOR 是面向动态向量检索的存算分离系统。`dev` 分支只保留一条查询路径：
GPU 常驻 OPQ/PQ32 导航码，持久化 CUDA kernel 维护查询状态，并通过 DOCA
GPUNetIO 直接读取存储节点上的紧凑图记录与精确向量。CPU 仅负责请求准入、
启动阶段批量传输、控制面和动态更新 RPC，不参与稳态图遍历。

## 查询路径

1. CPU 将请求写入有界提交队列，并按微批分配 GPU 查询槽。
2. 持久化 GPU block 完成 OPQ 变换并构建 PQ 查找表。
3. GPU 读取由最新完整 centroid publication 安装的版本化分片路由，按分片中心
   排序，并只从最近的一个物理分片选择实时存活入口初始化 beam；后续跨分片工作
   只能由真实图边触发。
4. GPU 通过 GPUNetIO 并发直接拉取远端 2048 字节以内的紧凑图记录；路由入口
   只保存指针和代际，静态节点使用常驻 base PQ code，动态节点的权威 PQ code
   随记录按需读取，不常驻计算侧动态索引副本。
5. GPU 使用对应 PQ code 对邻居做近似距离计算并更新 beam。
6. GPU 仅为最终候选拉取精确向量，计算 L2 距离并返回 top-k。

## 动态更新

存储节点继续拥有 insert、upsert 和 delete 的图维护协议：

- storage owner 负责 idmap、代际、紧凑图记录和反向边维护；
- Stage1 在物理 home 写入节点、出边和本地反向边，使新节点立即进入权威图；
- 查询直接遍历该权威动态图，动态 PQ code、精确向量、tombstone 和 generation
  都从存储记录读取，不经过计算节点广播或二次发布；
- Stage2 延续 Stage1 搜索上下文，仅补充远端候选，完成统一剪枝、最终物理
  placement 和最终反向边；
- 质心路由由存储节点低频发布完整事务，GPU control CTA 只安装路由状态，不编码
  或缓存 mutation 数据；
- 动态查询采用 incarnation-tagged read-committed 语义：durable watermark
  允许旧地址复用，递增 incarnation 和 header-last 发布使旧指针无法命中新对象；
  系统不提供跨计算节点 snapshot-RCU，也不维护固定客户端 ACK 数组。

ID 的逻辑 authority 与向量的物理 home 是两件事。基础和动态 ID 的 authority 都由
确定性的 `ID % N` 选择，负责串行化代际和目录更新；该存储节点自己的 idmap 保存
真实物理 placement。计算节点从 storage-canonical centroid 快照中选择 Stage1 的
物理 home，再由 authority 在本地执行或转发。这样任意数量的计算节点都无需复制
全量 ID 目录，也不会产生分裂的 ID 所有权，物理放置仍可随数据分布动态变化。
逻辑 ID 必须位于配置的 `[0, max-vectors)` 命名空间；计算和存储两侧都在创建事务
状态前校验该边界，使 generation/tombstone 目录的最坏空间由声明容量而非请求历史决定。

每个物理存储分片维护一个 `CentroidRouter`：成员变化以补偿式 FP64 sum/count 更新
中心，并保留 1--4 个由调用方确认仍然存活的真实图入口。离线 `.centroid` sidecar
提供启动时的精确状态；维护线程把一批 mutation 作为新的不可变版本发布，再通过
RDMA 可读的 variable-length centroid publication 公布
`{version, count, centroid, live entries}`。计算节点用 sequence/checksum 校验完整
publication，并把同一份版本化 FP32 centroid 同时安装到 CPU home selector 和 GPU
路由表；两者采用相同的 FP32 FMA 累加和分片 tie-break，而 FP64 sum/count 仅留在
存储侧用于稳定维护。
查询没有离线静态入口兜底：首份完整 centroid 快照安装成功后才开放准入。

唯一支持的两阶段插入语义为：

1. 计算节点先把向量编码为索引的 canonical dtype，并从已完整安装的
   storage-canonical centroid 快照选择物理 home；authority 验证 ID/代际、该 home
   和当前 placement；所选物理 home 从自身当前 centroid
   live entries 执行完整的宽度 `L` 本地搜索，写入 provisional 出边，并在单独的
   provisional 邻接区安装最多两个、按 tagged 动态槽位轮转的本地可达桥（至少一个
   ACK 才提交 Stage1）；普通查询同时遍历 stable/provisional 邻接，因而 Stage1
   ACK 后即可发现新节点，而热点容量不会随出度 `R` 倍增消耗。
2. Stage2 仍在该物理 home 上从 Stage1 保存的 beam、visited 和 remote frontier
   继续搜索，不重启逐分片搜索；只有沿图边跨分片时才通过 one-sided RDMA 拉取远端
   记录，并继续使用原始 dtype 的精确距离。收敛后统一执行一次 alpha RobustPrune，
   选择使最终跨分片边最少的物理 placement（只有严格改善才迁移），再以 ID/generation
   幂等协调最终反向边：先把一个临时可达桥原子提升为 `R` 有界的 stable 边，再发布
   其余普通反向边，最后清除全部 Stage1 protected 临时边；随后更新新旧 centroid
   membership 并推进 durable watermark。

maintenance 日志直接记录这条机制的工作量与收益：Stage2 continuation 的 remote
frontier、实际远端展开和评分记录数，初选 home 命中率、迁移数，以及迁移前后的
跨分片边数；Stage1/Stage2 搜索预算耗尽次数也单独报告，任何有界工作量造成的
质量风险都不会被吞掉。benchmark 按测量窗口对累积计数器做差，因此可在不同数据集、dtype、
并发度和更新率下同时检验实时开销与长期局部性，而不是依赖短时采样。

## 索引文件

运行索引固定为 schema 16、L2、`plain` vector record、tagged compact graph、持久化
storage control block 和
OPQ/PQ 导航。默认 profile 使用 32 个 8-bit 子空间。运行时不读取任何计算节点图清单文件。

| 文件                              | 计算节点             | 存储节点 X | 作用                     |
| ------------------------------- | ---------------- | ------ | ---------------------- |
| `<prefix>.meta.json`            | 必需               | 必需     | 分片、远端 offset 和格式契约     |
| `<prefix>.pq32`                 | 必需               | 不需要    | OPQ 矩阵与 PQ codebook    |
| `<prefix>_nodeX_ofN.dat`        | 不需要              | 必需     | 精确向量、固定记录和紧凑图          |
| `<prefix>_nodeX_ofN.idmap`      | 不需要              | 必需     | 与构建/owner 强绑定的 ID/代际/物理目录 |
| `<prefix>_nodeX_ofN.centroid`   | 不需要              | 必需     | 物理分片 FP64 sum/count 与启动入口  |
| `<prefix>_nodeX_ofN.pq32.codes` | 不需要              | 必需     | 启动时注册到远端内存的 PQ32 码流    |

计算节点本地保存 metadata 和 PQ 模型；metadata 只合成并校验不可变分片布局，
不读取 medoid 或离线采样 entry-point。METIS 只决定物理 placement；逻辑 authority
始终可由 ID 推导，因此查询和更新模式的计算节点都不加载 `.idmap`。计算节点不会
保存 `.gpu.idx`、图分片、精确向量、ID 目录或全量导航码。

每个 PQ code sidecar 的 header 同时绑定向量 dtype、整次索引构建指纹和物理分片
指纹；存储节点会与 metadata 以及分片文件中的指纹交叉校验，拒绝同尺寸的跨数据集
或跨分片码流。PQ32 每个向量占 32 字节：SIFT100M 为 3.2 GB，SIFT1B 为 32 GB
（约 29.8 GiB）。计算 GPU 只为每个分片常驻中心和最多 4 个实时入口，不保存
计算侧 mutable overlay、基础图记录或精确向量；稳态查询通过
GPUNetIO/RDMA 按需直接读取它们。

每个 owner idmap 使用 `owner_sharded_v2_bound`：header 同时绑定整次构建指纹、
owner 对应的分片指纹、owner/shard 数量、tagged `RemotePtr` 布局以及 payload/header
checksum。存储节点逐块流式校验精确文件长度、`ID % N`、静态记录范围和重复 ID；
旧的 owner-sharded v1 不会被兼容加载。基础目录在内存中使用紧凑的
`ID -> RemotePtr` 表；generation=0 等不可变状态在查找时物化，只有参与动态更新的
ID 才进入完整事务状态 overlay，因此基础规模不会乘上 mutation receipt 的体积。

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
- `vamana_pq_indexer`：训练 OPQ/PQ 并生成分片码流。

无 GPU 的存储节点可同时构建存储服务和 CPU 离线索引工具：

```bash
cmake -S . -B build-storage \
  -DCMAKE_BUILD_TYPE=Release \
  -DDVSTOR_STORAGE_NODE_ONLY=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
cmake --build build-storage -j --target \
  dvstor_memory_node vamana_offline_builder vamana_pq_indexer \
  vamana_legacy_index_converter
```

存储服务本身只依赖 CPU、RDMA、Boost 和 TBB。离线工具使用 CPU Faiss、BLAS、
LAPACK、OpenMP 和可选 METIS，但不依赖 CUDA 或 DOCA。CMake 直接链接
`libfaiss` CPU 库，不加载可能包含 `faiss_gpu_objs` 的 Faiss CMake 导出。
动态更新运行时使用标准 C++20 coroutine 和 `atomic_ref`，因此要求 GCC 11+
或等价的现代 Clang。若节点只运行存储服务，可增加
`-DDVSTOR_BUILD_OFFLINE_TOOLS=OFF`。

`DVSTOR_METIS_PARTITION=AUTO` 会先验证 METIS/GKlib 能否
在本机真实链接；若仓库内预编译库与本机 glibc 不兼容，会先回退到系统 CPU
METIS，没有兼容版本时才保留 `balanced/bfs` 并禁用 `metis`。需要强制 METIS
时，安装本机 CPU 版本并设置
`-DDVSTOR_METIS_ROOT=/path/to/metis -DDVSTOR_METIS_PARTITION=ON`。

新建索引的分片由 `vamana_offline_builder --partition-strategy ...` 完成，不需要
GPU，也不需要额外重分片可执行文件。

## 生成索引

完整构建：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

该命令先产出 schema-15 tagged graph 中间态及每个物理分片的精确 `.centroid`
sidecar，再由 PQ indexer 原子升级并产出最终 schema-16 分片契约、owner idmap、
OPQ/PQ32 模型和
每分片 PQ32 码流，不需要再运行重编码脚本。默认目标已存在时脚本会在昂贵构建前
拒绝覆盖；建议通过 `PQ_INDEX_PREFIX=/new/prefix` 构建新版本，确认需要原地重建时
才设置 `OVERWRITE_INDEX=1`。

若已有完整、静态的 schema-15 `vamana_compact_v1` 分片，可用
`vamana_legacy_index_converter` 保留原图与 METIS placement 并流式升级，无需重新
构图或训练 PQ；输入要求、只读校验和命令见
[`experiment/README.md`](experiment/README.md#转换旧-compact-v1-索引)。

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
不会回绕重复。默认性能查询集是 `bigann_base.bvecs` 的 `[100M,105M)`，插入集是
`[103M,105M)`；生成的 `.u8bin` 文件位于 SIFT1B 数据集目录。两个流按你的当前
压测设置有重叠，适合直接复用预生成数据做吞吐测试，但不能当作严格 held-out 的
质量证据。计算节点只需这两份生成文件，不需要完整的 `bigann_base.bvecs`。

分布式部署时，每台存储节点只需其自身的 `.dat`、`.idmap`、`.centroid`、
`.pq32.codes`，再加共享 metadata；计算节点不需要额外的路由 sidecar。详细流程见
`experiment/README.md`。

## 验证

```bash
ctest --test-dir build --output-on-failure
bash -n experiment/*.sh experiment/profiles/*.env
```

`gpu_memory_budget_test` 覆盖 SIFT100M/SIFT1B 预算；格式、PQ、centroid 路由和
提交环均有独立测试。硬件吞吐和召回率仍需在真实存储节点启动后通过 profile
进行验证。

## 代码结构

- `src/gpu_search/`：PQ 模型、索引布局、持久化引擎与 CUDA kernel；
- `src/gpu/`：DOCA GPUNetIO 传输和探针；
- `src/memory_node/`：分片加载、更新、维护与 storage peer RPC；
- `src/service/`：计算服务、索引契约和统计；
- `src/vamana/`：compact graph、CentroidRouter、centroid state 和 idmap 格式；
- `tools/vamana_offline/`：离线构图与 PQ sidecar 生成；
- `experiment/`：唯一支持的 SIFT100M profile。
