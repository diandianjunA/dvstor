# 第 1 课 项目总览、架构与构建系统

> 本课是 30 课源码教程的第一课。目标是让读者在不打开任何源文件的情况下，先建立
> 对 dvstor 项目的整体认知：它解决什么问题、为什么这样切分模块、CMake 怎样把
> 一份源码编译成"计算节点 / 存储节点 / 离线工具"三类产物。后续每一课都会回到
> 本课给出的目录导览图和 6 步查询路径，因此本课的名词和边界请务必读熟。

## 1. 本课目标与涉及文件

本课只讲"项目骨架"，不进入任何 CUDA kernel 或 RDMA 细节。具体目标：

1. 理解 dvstor 的定位：存算分离、GPU 中心化查询、DOCA GPUNetIO、**无 CPU fallback**。
2. 记住"单查询路径 6 步"，这是后续第 14、18–22 课的纲领。
3. 掌握仓库的目录边界：`src/gpu_search`、`src/gpu`、`src/memory_node`、`src/service`、
   `src/vamana`、`src/common`、`tools`、`rdma-library`、`experiment` 各自承担什么。
4. 能逐段读懂 `CMakeLists.txt`，包括三个关键构建选项
   （`DVSTOR_STORAGE_NODE_ONLY`、`DVSTOR_BUILD_OFFLINE_TOOLS`、`DVSTOR_METIS_PARTITION`）
   以及"为什么存储节点可以不依赖 CUDA/DOCA"。
5. 知道 `.clangd` 的作用，以及 `docs/source_layout.md` 中"变更约束"为什么是硬约束。
6. 通过两张总览图把 30 课的脉络挂到具体模块上。

本课需要完整阅读的文件（你已经读过它们，下面会逐段引用行号）：

- `README.md`（仓库总入口，226 行）
- `CMakeLists.txt`（顶层构建脚本，398 行）
- `.clangd`（clangd 配置，32 行）
- `docs/source_layout.md`（源码布局与变更约束，44 行）
- `docs/gpu_persistent_engine.md`（GPU 中心化引擎设计文档，188 行）
- `experiment/README.md`（实验流程说明，240 行）

另外用 `find`/`wc -l` 盘点了目录结构。下文引用的所有行号都来自真实文件。

## 2. 项目定位：从 README 第一段说起

`README.md:1-8` 给出了 dvstor 的自我定位：

```
1	# DVSTOR
2	
3	DVSTOR 是面向动态向量检索的存算分离系统。`dev` 分支只保留一条查询路径：
4	GPU 常驻 OPQ/PQ32 导航码，持久化 CUDA kernel 维护查询状态，并通过 DOCA
5	GPUNetIO 直接读取存储节点上的紧凑图记录与精确向量。CPU 仅负责请求准入、
6	启动阶段批量传输、控制面和动态更新 RPC，不参与稳态图遍历。
7	
8	历史实现保存在 `main` 分支；`dev` 不提供旧查询引擎或 CPU 查询回退。
```

逐句拆解：

- **"面向动态向量检索"**：系统支持 insert/upsert/delete 在线写入，不是只读 ANN。
  这一点会在第 10、15、23–26 课反复出现。
- **"存算分离"**：计算节点（GPU）与存储节点（持有 `.dat`/`.idmap`/`.pq32.codes`）物理
  分离，靠 RDMA 互联。第 22、23 课讲传输与存储主体。
- **"GPU 常驻 OPQ/PQ32 导航码"**：PQ 编码常驻 GPU 显存，不在查询路径上做 H2D。
  `docs/gpu_persistent_engine.md:9-16` 明确"图和精确向量由存储节点持有，PQ code 常驻
  计算 GPU"。
- **"持久化 CUDA kernel 维护查询状态"**：查询 beam、visited set、PQ lookup table 不
  在每次查询时重建，而由一个长期存活的 kernel block 持有。第 11、17、20、21 课讲
  这套机制。
- **"通过 DOCA GPUNetIO 直接读取存储节点"**：远端 RDMA read 终点直接落在 GPU
  显存，不经过主机内存。第 22 课专题。
- **"CPU 仅负责请求准入、启动批量传输、控制面和动态更新 RPC"**：CPU 角色严格受限。
  `docs/gpu_persistent_engine.md:34-48` 把 CPU admission thread 的工作列得很死：领
  slot、拷 query、发 ring descriptor，"不等待人为凑批"。
- **"不参与稳态图遍历"**：稳态=稳态查询。`docs/source_layout.md:39-40` 的"变更约束"
  把"不新增 CPU 图导航或隐式传输回退"列为硬约束。
- **"`dev` 不提供旧查询引擎或 CPU 查询回退"**：这是关键设计决定。`docs/gpu_persistent_engine.md:177-188` 把它叫做"fail-stop"——任何 GPU/GPUNetIO/delta
  失败都直接把引擎标记为 unhealthy，"系统没有 CPU 查询 fallback。这样 benchmark
  不会在硬件路径失效后悄悄测量另一套慢路径，也不会以低召回结果继续运行。"

### 2.1 单查询路径 6 步

`README.md:10-17` 是全教程最重要的一张清单：

```
10	## 查询路径
11	
12	1. CPU 将请求写入有界提交队列，并按微批分配 GPU 查询槽。
13	2. 持久化 GPU block 完成 OPQ 变换并构建 PQ 查找表。
14	3. GPU 对常驻入口点和动态 delta 入口进行打分，初始化 beam。
15	4. GPU 通过 GPUNetIO 并发拉取远端 512 字节以内的紧凑图记录。
16	5. GPU 使用常驻 PQ code 对邻居做近似距离计算并更新 beam。
16	6. GPU 仅为最终候选拉取精确向量，计算 L2 距离并返回 top-k。
```

这 6 步对应后续课程的落点：

| 步骤 | 关键模块 | 课程 |
| --- | --- | --- |
| 1 提交队列、slot | `src/gpu_search/mapped_ring.hh`、`device_ring.cuh`、`src/gpu_search/persistent_engine/query_execution.cc` | 第 14、17 课 |
| 2 OPQ/PQ LUT | `src/gpu_search/persistent_kernel/candidate_scoring.cuh`、`pq_index.cc` | 第 9、18 课 |
| 3 入口打分 | `navigation_bootstrapper.cc`、`dynamic_route_overlay.cc`、`adaptive_route_table.cc` | 第 6、10 课 |
| 4 GPUNetIO 远端读 | `src/gpu/gpunetio_transport.cc`、`src/gpu/gpunetio_probe.cu` | 第 22 课 |
| 5 邻居 PQ 评分 | `persistent_kernel/query_traversal.cuh`、`candidate_scoring.cuh` | 第 18、20 课 |
| 6 精确重排 | `persistent_kernel/query_traversal.cuh`、`rdma_cache.cuh` | 第 19、20 课 |

`README.md:19-21` 紧跟着强调："查询过程中没有 CPU 驱动的逐轮 RDMA、主机向量中转
或本地图副本。"——这就是第 14 课"查询执行/路由/完成"和第 20 课"查询遍历主循环"
要逐行实现的约束。

### 2.2 动态更新的两条原则

`README.md:23-43` 描述动态更新。两段最关键：

- **storage owner 拥有图维护协议**（`README.md:25-33`）：idmap、代际、紧凑图记录、
  反向边都在存储节点；mutation 以 epoch 批次发布到 GPU delta；CPU 把原始向量写入
  映射固定内存，由"专用常驻 control CTA"完成 OPQ/PQ 编码。这对应第 23–26 课。
- **路由不是静态 anchor 的延续**（`README.md:35-43`）：每个存储分片维护 8 个固定
  容量的 EMA 中心/活代表，通过既有 RDMA control page 发布 canonical 快照；GPU 查询
  侧拉取这份快照，与离线静态入口竞争初始 beam。`docs/gpu_persistent_engine.md:62-69`
  补充：每槽同时携带 32B 权威 PQ code，"动态表内存固定，不会随运行时间或 mutation
  数量增长"。这对应第 10 课"delta/动态路由/预算"。

### 2.3 索引文件契约

`README.md:67-90` 给出索引文件表，这是第 6–8 课的索引格式契约的浓缩：

```
73	| 文件 | 计算节点 | 存储节点 X | 作用 |
74	| --- | --- | --- | --- |
75	| `<prefix>.meta.json` | 必需 | 必需 | 分片、远端 offset 和格式契约 |
76	| `<prefix>.pq32` | 必需 | 不需要 | OPQ 矩阵与 PQ codebook |
77	| `<prefix>.anchors` | 必需 | 不需要 | GPU 静态冷启动/召回兜底入口 |
78	| `<prefix>_nodeX_ofN.dat` | 不需要 | 必需 | 精确向量、固定记录和紧凑图 |
79	| `<prefix>_nodeX_ofN.idmap` | 更新模式需全部分片；纯查询不需要 | 必需 | base ID 的真实 owner/版本映射 |
80	| `<prefix>_nodeX_ofN.pq32.codes` | 不需要 | 必需 | 启动时注册到远端内存的 PQ32 码流 |
```

要点：计算节点**不持有** `.dat`、`.pq32.codes` 或任何图分片；存储节点**不需要**
`.pq32` 模型或 `.anchors`。这种"非对称部署"是 CMake 分裂构建（storage-only vs
compute）的根本动因，下文第 4 节会回到这一点。

`README.md:86-90` 还给出内存预算硬上限：SIFT1B PQ32 = 32 GB、mutable L0 默认
256 MiB、显式分配上限 36 GiB、CUDA/DOCA reserve 4 GiB。第 9 课"GPU 类型/遥测/PQ
模型"和第 11 课"持久化引擎 PImpl/生命周期"会逐项核对。

### 2.4 构建目标与依赖速览

`README.md:92-129` 把构建依赖和目标列得很清楚。先看依赖（`README.md:93-95`）：

> 计算节点需要 C++20、CUDA、RDMA verbs、DOCA GPUNetIO、Boost、TBB、OpenMP 和
> Faiss；离线 METIS 分片可选。默认 DOCA 路径为 `/opt/mellanox/doca`。

再看 5 个主要目标（`README.md:103-110`）：

- `dvstor_compute_node`：GPU 查询与更新客户端（`src/main.cc`）；
- `dvstor_memory_node`：存储节点（`src/memory_node_main.cc`）；
- `dvstor_breakdown_benchmark`：吞吐、延迟、召回率和分解统计；
- `vamana_offline_builder`：CPU 上构建 compact Vamana 图，支持 `balanced`/`bfs`/可选
  `metis` 分片；
- `vamana_pq_indexer`：训练 OPQ/PQ 并生成分片码流；
- `vamana_legacy_index_converter`：复用旧图与分片布局进行迁移。

注意 `README.md:112-129` 的存储节点构建段：无 GPU 的存储节点可以同时构建存储服务
和 CPU 离线工具，"存储服务本身只依赖 CPU、RDMA、Boost 和 TBB。离线工具使用 CPU
Faiss、BLAS、LAPACK、OpenMP 和可选 METIS，但不依赖 CUDA 或 DOCA。" 这条边界决定了
`CMakeLists.txt` 的整体结构，第 4 节细讲。

`README.md:131-143` 还给出三条"踩坑警告"：

1. 必须用独立 `build-storage` 目录，不要把计算节点的 `build` 切成存储模式；
2. `DVSTOR_METIS_PARTITION=AUTO` 会先验证 METIS/GKlib 能否真实链接，bundled 库与
   本机 glibc 不兼容时回退系统 METIS，再不行才保留 `balanced/bfs` 并禁用 `metis`；
3. `vamana_legacy_index_converter` 只复用旧图，不重新分片；schema-13 的
   `vamana_bfs_repartitioner`/`vamana_metis_repartitioner` 已删除，不能用于 schema 14+。

### 2.5 运行与验证

`README.md:190-224` 给出 SIFT100M 的运行流程和验证命令。运行流程是：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_recall.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

`README.md:215-224` 的验证命令只有两条：

```bash
ctest --test-dir build --output-on-failure
bash -n experiment/*.sh experiment/profiles/*.env
```

`bash -n` 只做语法检查，不执行。`ctest` 覆盖预算、格式、PQ、delta、迁移、提交环
等。`README.md:222-224` 明确："硬件吞吐和召回率仍需在真实存储节点启动后通过 profile
进行验证。"——单元测试不能替代真实 RDMA/GPU 测试，这点在第 30 课会再讲。

## 3. 目录与模块边界

`README.md:226-236` 给出了"代码结构"清单，但只是入口名。本节用实际 `find`/`wc -l`
盘点的结果补充每个模块的规模和内部子目录。

### 3.1 顶层目录

仓库根目录（`ls /home/xjs/experiment/dvstor`）：

```
build  cmake  CMakeLists.txt  docs  experiment  LICENSE  motivation
rdma-library  README.md  scripts  src  test  thirdparty  tools  tutorial
```

- `build/`：计算节点的 CMake 构建产物（gitignore）。
- `cmake/`：自定义 CMake 模块，3 个文件，下文第 4.4 节详讲。
- `docs/`：2 个设计文档（`source_layout.md`、`gpu_persistent_engine.md`）。
- `experiment/`：唯一支持的 SIFT100M profile 与所有 shell 脚本。
- `motivation/`：历史 RaBitQ pipeline 探索目录（3 个子目录），不在运行路径上。
- `rdma-library/`：自研 RDMA 传输库，第 4、5 课专题。
- `scripts/`：仅 `build_metis64.sh` 一个脚本，用于编译 thirdparty METIS。
- `src/`：运行时源码主体（约 29.7k 行 cc/cu + 7.4k 行 hh/cuh）。
- `test/`：24 个 GoogleTest 用例 + `CMakeLists.txt`。
- `thirdparty/`：ankerl（hash map）、httplib、metis64、nlohmann（json）、xoshiro（RNG）。
- `tools/`：离线构建、benchmark、迁移工具（约 8.7k 行 cc）。
- `tutorial/`：本教程所在目录。

### 3.2 `src/` 内部子树与规模

```
src
├── coroutine.hh          # C++20 coroutine 基础原语（第 3 课）
├── main.cc               # dvstor_compute_node 入口（第 27 课）
├── memory_node_main.cc   # dvstor_memory_node 入口（第 23 课）
├── remote_pointer.hh     # 5-byte RemotePtr 编码（第 7、8 课）
├── common/               # 175 行 cc + 1846 行 hh，15 文件
├── gpu/                  # 884 行 cc/cu + 87 行 hh，4 文件（第 22 课）
├── gpu_search/           # 1774 行 cc + 1573 行 hh，22 文件（第 9–11、14–16、19–21 课）
│   ├── persistent_engine/    # 3081 行 cc + 493 行 hh，10 文件（第 11、14、15、16 课）
│   └── persistent_kernel/    # 0 行 cc + 3760 行 cuh，5 文件（第 17–21 课）
├── memory_node/          # 1210 行 cc + 1115 行 hh，7 文件（第 23 课）
│   ├── peer_rpc/              # 1730 行 cc + 579 行 hh，6 文件（第 24 课）
│   ├── storage_owner_index/   # 1998 行 cc + 660 行 hh，10 文件（第 25 课）
│   ├── storage_owner_maintenance/  # 2141 行 cc + 1237 行 hh，9 文件（第 26 课）
│   └── storage_owner_runtime/ # 945 行 cc + 21 行 hh，5 文件（第 26 课）
├── service/              # 268 行 cc + 853 行 hh，9 文件（第 2、8、27、28 课）
│   ├── breakdown/             # 0 行 cc + 362 行 hh，5 文件（第 30 课）
│   └── compute_service/       # 379 行 cc + 35 行 hh，4 文件 + storage_owner/ 子目录
│       └── storage_owner/     # 1013 行 cc + 38 行 hh，5 文件（第 28 课）
└── vamana/               # 539 行 cc + 687 行 hh，8 文件（第 6、7、10 课）
```

下面逐模块说明职责（行号引用 `docs/source_layout.md:5-17` 的"模块边界"段，并与
`README.md:226-235` 对照）。

#### `src/common/` — 公共原语与配置

15 个文件，1846 行头文件 + 175 行实现。包含：

- `atomic_utils.hh`、`bounded_queue.hh`、`completion_pool.hh`、`sliding_completion_ring.hh`：
  并发原语，第 3 课。
- `configuration.hh`、`constants.hh`、`core_assignment.{cc,hh}`、`core_partition.hh`：
  配置与 CPU 核绑定，第 2 课。
- `distance.hh`、`vector_dtype.hh`、`types.hh`、`index_path.hh`、`timing.{cc,hh}`：
  距离函数、dtype、通用类型、路径、计时。

注意 `CMakeLists.txt:267-268` 把 `core_assignment.cc` 和 `timing.cc` 单独列进 runtime
源码列表，说明这两个是计算与存储共用的"基础工具"，不能依赖任何 GPU/RDMA 符号。

#### `src/gpu_search/` — GPU 查询引擎主体

22 个文件，是计算节点最重的子树。又分两层：

- **装配层**（`src/gpu_search/*.cc`，22 文件中的非目录文件）：
  - `persistent_engine.{cc,hh}`：PImpl 装配入口，第 11 课。
  - `persistent_engine/`：构造、生命周期、查询、路由、增量发布、存储回收、完成、
    健康、状态，第 11、14–16 课。
  - `types.{cc,hh}`、`index_format.{cc,hh}`、`pq_index.{cc,hh}`、`delta_index.{cc,hh}`、
    `dynamic_route_overlay.{cc,hh}`、`navigation_bootstrapper.{cc,hh}`：GPU 侧类型、
    索引格式、PQ 模型、delta overlay、入口引导，第 9、10、14 课。
  - `memory_budget.hh`、`delta_scan_budget.hh`、`initial_seed_budget.hh`：预算头，第 9、11 课。
  - `mapped_ring.hh`、`device_ring.cuh`：提交/完成 ring，第 17 课。
  - `dynamic_route_consistency.hh`：路由一致性，第 10 课。
- **kernel 层**（`src/gpu_search/persistent_kernel*`）：
  - `persistent_kernel.cu`：单一 CUDA translation unit，第 17–21 课的根。
  - `persistent_kernel/*.cuh`（3760 行）：`context.cuh`（kernel 上下文，第 17 课）、
    `candidate_scoring.cuh`（评分，第 18 课）、`rdma_cache.cuh`（第 19 课）、
    `query_traversal.cuh`（遍历主循环，第 20 课）、`runtime.cuh`（kernel 运行时/角色
    调度，第 21 课）。

`docs/source_layout.md:11` 强调"`src/gpu_search/persistent_kernel.cu`：单一持久化
CUDA translation unit"，`docs/source_layout.md:27-28` 解释原因："CUDA 设备代码仍由
`persistent_kernel.cu` 形成一个 translation unit，以保留设备端内联与常量传播"。这
意味着第 17–21 课会把多个 `.cuh` 在同一个 `.cu` 里 include，而不是分散编译。

#### `src/gpu/` — DOCA GPUNetIO 传输与探针

4 个文件：`gpunetio_probe.{cu,hh}`、`gpunetio_transport.{cc,hh}`。第 22 课专题。
`CMakeLists.txt:238-239` 在检测到 GPUNetIO 时把 `gpunetio_probe.cu` 加进
`DVSTOR_GPU_SOURCES`，`CMakeLists.txt:297-300` 把 `gpunetio_transport.cc` 加进
`DVSTOR_RUNTIME_SOURCES` 并关掉 deprecated/pedantic 警告（DOCA 头文件较老）。

#### `src/memory_node/` — 存储节点主体

7 个直接文件 + 4 个子目录。直接文件：

- `memory_node.{cc,hh}`：存储服务主类，第 23 课。
- `peer_rdma.{cc,hh}`：peer 间 RDMA，第 23 课。
- `startup_protocol.hh`、`storage_owner_cpu_plan.hh`、`storage_owner_state.hh`、
  `storage_reclaim.hh`：启动协议、CPU 计划、状态、回收头。

子目录对应 `docs/source_layout.md:12-15`：

- `storage_owner_index/`：存储分配、图访问、候选搜索、图修改，第 25 课。
- `peer_rpc/`：peer RPC 生命周期、请求处理、worker、客户端请求，第 24 课。
- `storage_owner_maintenance/`：维护队列、worker、图任务，第 26 课。
- `storage_owner_runtime/`：更新 runtime 生命周期、批执行、wire protocol，第 26 课。

#### `src/service/` — 计算服务、索引契约、统计

9 个直接文件 + 2 个子目录：

- `base_owner_map.{cc,hh}`：base ID → owner 映射，第 8、28 课。
- `index_metadata.{cc,hh}`：metadata 解析，第 8 课。
- `storage_owner_protocol.hh`、`storage_owner_client_helpers.hh`：存储 owner 协议，
  第 8、24 课。
- `compute_service.hh`、`query_result.hh`、`breakdown.hh`：服务门面、结果、分解遥测。
- `compute_service/`：4 文件 + `storage_owner/` 子目录，对应 `docs/source_layout.md:16`：
  - `index_commands.cc`、`lifecycle.cc`、`search.cc`：第 27 课。
  - `storage_owner/`：计算侧更新入口、发送、完成，第 28 课。
- `breakdown/`：5 个头文件，第 30 课遥测定义。

#### `src/vamana/` — Vamana 图格式与 anchor/idmap/路由表

8 个文件：

- `anchor_index.{cc,hh}`：anchor 文件格式，第 6 课。
- `adaptive_route_table.{cc,hh}`：自适应路由表（8 槽 EMA 中心），第 10 课。
- `idmap.hh`、`hot_graph.hh`、`storage_layout_resolver.hh`、`vamana_node.hh`：
  idmap、热图、布局解析、节点布局，第 6、7 课。

#### `tools/` — 离线工具与 benchmark

```
tools
├── dvstor_breakdown_benchmark.cc     # 入口（第 30 课）
├── dvstor_sift101m_long_insert_recall.cc  # 长插入召回（第 30 课）
├── generate_sift101m_recall_data.cc  # 召回数据生成（第 30 课）
├── gpunetio_probe.cc / gpunetio_loopback_probe.cc  # GPUNetIO 探针（第 22 课）
├── vamana_offline_builder.cc / vamana_anchor_sidecar_builder.cc
├── vamana_pq_indexer.cc / vamana_legacy_index_converter.cc
├── breakdown_benchmark/   # 2856 行 cc + 332 行 hh，12 文件（第 30 课）
├── legacy_index/          # 549 行 cc + 24 行 hh，2 文件（第 29 课）
└── vamana_offline/        # 2504 行 cc + 385 行 hh，18 文件（第 29 课）
```

`docs/source_layout.md:17` 把 `tools/breakdown_benchmark/` 概括为"数据集、进度、报表
和工作负载编排"。第 29 课讲 `vamana_offline/` 和 `legacy_index/`，第 30 课讲
`breakdown_benchmark/`。

#### `rdma-library/` — 自研 RDMA 传输库

`rdma-library/library/` 下 20 个文件，4379 行 `extern/concurrentqueue.hh` 是第三方
无锁队列。自研部分包含 `context.{cc,hh}`、`queue_pair.{cc,hh}`、`memory_region.{cc,hh}`、
`connection_manager.{cc,hh}`、`configuration.{cc,hh}`、`batched_read.hh`、
`detached_qp.hh`、`hugepage.hh`、`latch.hh`、`thread.hh`、`utils.{cc,hh}`、
`dynamic_region_allocator.hh`、`types.hh`。第 4、5 课专题。

`rdma-library/CMakeLists.txt`（20 行）构建 `rdma_library` 静态库，依赖 IBVerbs、
Boost::program_options、TBB、Threads，被顶层 `CMakeLists.txt:96` `add_subdirectory`
引入。

#### `experiment/` — SIFT100M 实验目录

`experiment/README.md:1-6` 开宗明义："实验目录只保留 `04_gpu_persistent_gpunetio`：
持久化 GPU OPQ/PQ32 图导航、GPUNetIO 远端读取和 storage-owner 动态更新。旧
profile、旧 sidecar 转换器和历史输出不属于 `dev` 的运行接口。"

主要脚本：`build_sift100m_index.sh`、`convert_legacy_sift100m_index.sh`、
`reencode_sift100m_pq.sh`、`upgrade_pq_schema15.sh`、`start_all_memory_nodes.sh`、
`start_memory_node.sh`、`stop_memory_nodes.sh`、`run_recall.sh`、`run_breakdown.sh`、
`run_sift101m_long_insert_recall.sh`。辅助：`sift100m_common.sh`、`common.sh`、
`compare_reports.py`、`convert_sift100m.py`、`prepare_sift100m_data.sh`、
`prepare_sift_benchmark_data.py`、`profiles/`、`reports/`、`logs/`、`pids/`。

第 30 课会逐脚本讲，本课只把入口记住：`experiment/README.md:118-126` 的启动流程和
`experiment/README.md:191-205` 的召回/性能流程。

## 4. CMakeLists.txt 逐段讲解

顶层 `CMakeLists.txt` 共 398 行，结构清晰。下文按段拆解。

### 4.1 项目声明与构建选项（1-62 行）

```
1	cmake_minimum_required(VERSION 3.18)
2	
3	option(DVSTOR_STORAGE_NODE_ONLY
4	    "Build storage-host runtime and CPU offline tools without CUDA/GPU dependencies" OFF)
5	if (DVSTOR_STORAGE_NODE_ONLY)
6	    project(dvstor CXX)
7	else ()
8	    project(dvstor CXX CUDA)
9	endif ()
```

`cmake_minimum_required(VERSION 3.18)` 是 CUDA as first-class language 的最低要求。
`DVSTOR_STORAGE_NODE_ONLY` 是整套构建分裂的核心开关：开启时 `project` 只声明 CXX，
不声明 CUDA，下文所有 CUDA 相关代码段被 `if (NOT DVSTOR_STORAGE_NODE_ONLY)` 包住。

```
11	set(CMAKE_CXX_STANDARD 20)
12	set(CMAKE_CXX_STANDARD_REQUIRED ON)
13	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
14	set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")
```

C++20 是硬要求（coroutines、atomic_ref、concepts）。`CMAKE_EXPORT_COMPILE_COMMANDS`
为 clangd 生成 `compile_commands.json`（第 5 节会回来）。Debug 构建启用 ASan 与
`-fno-inline`，便于调试 kernel 之外的 CPU 路径。

```
16	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
17	    message(FATAL_ERROR
18	        "DVSTOR requires GCC 11 or newer for standard C++20 coroutines and atomic_ref. "
19	        "Select a newer compiler with -DCMAKE_CXX_COMPILER=/path/to/g++.")
20	endif ()
```

显式校验 GCC 11+，对应 `README.md:127-129` 的"动态更新运行时使用标准 C++20 coroutine
和 `atomic_ref`，因此要求 GCC 11+ 或等价的现代 Clang"。

```
22	# Use RPATH (not RUNPATH) so that custom METIS/GKlib paths take priority
23	# over system library paths at runtime.
24	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
25	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
```

`--disable-new-dtags` 把 `RUNPATH` 改回 `RPATH`，使 `DVSTOR_METIS_ROOT` 指定的库
优先于系统库被加载。这与 `cmake/DvstorDependencies.cmake` 的 METIS 处理配套。

```
27	if (NOT DVSTOR_STORAGE_NODE_ONLY)
28	    # CUDA standard
29	    set(CMAKE_CUDA_STANDARD 17)
30	    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
31	
32	    # Auto-detect GPU architecture if not specified
33	    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
34	        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
35	    endif ()
36	endif ()
```

注意一个不对称：**C++ 用 20，CUDA 用 17**。原因是 nvcc 对 C++20 的支持滞后且部分
device 端特性不稳定；device 代码用 C++17 足够，host 端 C++20 的 coroutine/atomic_ref
也不进入 device。默认架构 80（A100）/86/89/90，覆盖 Ampere 与 Hopper。

```
38	# set default build type to Release
39	if (NOT CMAKE_BUILD_TYPE)
40	    set(CMAKE_BUILD_TYPE Release)
41	endif ()
42	
43	option(DVSTOR_BUILD_EXECUTABLES "Build standalone DVSTOR executables and tools" ON)
44	option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)
45	option(DVSTOR_BUILD_OFFLINE_TOOLS
46	    "Build CPU-only index construction, partitioning, conversion, and PQ tools" ON)
47	option(DVSTOR_USE_NATIVE_ARCH "Compile C++ sources with -march=native" OFF)
```

四个二级开关：

- `DVSTOR_BUILD_EXECUTABLES`：是否构建 `dvstor_compute_node`/`dvstor_breakdown_benchmark`
  等可执行目标（库仍构建），便于只想要静态库做联调时关闭。
- `DVSTOR_BUILD_TESTS`：是否 `add_subdirectory(test)`。
- `DVSTOR_BUILD_OFFLINE_TOOLS`：是否构建 `vamana_offline_builder`/`vamana_pq_indexer`/
  `vamana_legacy_index_converter`/`vamana_anchor_sidecar_builder`/
  `generate_sift101m_recall_data`。**这个开关在 storage-only 构建里也能用**，所以
  无 GPU 的存储主机可以同时跑离线构图——这是 `README.md:112-122` 的 storage 构建
  示例同时构建存储服务和离线工具的实现机制。
- `DVSTOR_USE_NATIVE_ARCH`：默认关，避免在 heterogenous 集群上编译出 AVX-512 之类的
  指令在别处 illegal instruction。

```
48	if (DVSTOR_STORAGE_NODE_ONLY)
49	    set(Boost_NO_BOOST_CMAKE ON CACHE BOOL
50	        "Use CMake's FindBoost module on CPU-only storage hosts to avoid Conda RPATH injection")
51	endif ()
52	if (NOT DVSTOR_STORAGE_NODE_ONLY)
53	    set(DVSTOR_DOCA_ROOT "/opt/mellanox/doca" CACHE PATH "DOCA SDK installation root")
54	endif ()
```

storage-only 模式下显式禁用 Boost 的 CMake config 模式，强制走 FindBoost module，
"避免 Conda RPATH 注入"——这是开发机上 Conda 环境 Boost 与系统 Boost 冲突时的对策。
DOCA 路径只在非 storage-only 时定义。

```
55	set(DVSTOR_METIS_PARTITION "AUTO" CACHE STRING "METIS graph partitioning support: AUTO, ON, or OFF")
56	set_property(CACHE DVSTOR_METIS_PARTITION PROPERTY STRINGS AUTO ON OFF)
57	string(TOUPPER "${DVSTOR_METIS_PARTITION}" DVSTOR_METIS_PARTITION_MODE)
58	if (NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "AUTO" AND
59	    NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "ON" AND
60	    NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "OFF")
61	    message(FATAL_ERROR "DVSTOR_METIS_PARTITION must be AUTO, ON, or OFF")
62	endif ()
```

METIS 三态：`AUTO`（默认，尝试链接，失败就降级）、`ON`（强制要求，失败就
FATAL_ERROR）、`OFF`（完全不找）。`cmake/DvstorDependencies.cmake` 实现具体逻辑，
第 4.4 节展开。

### 4.2 编译选项与路径常量（64-97 行）

```
64	# CXX compiler flags (not applied to CUDA files)
65	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
66	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
67	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
68	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wfatal-errors>)
69	if (DVSTOR_USE_NATIVE_ARCH)
70	    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
71	endif ()
72	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
73	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
74	add_compile_options(-DNOHUGEPAGES)
75	#add_compile_options(-g -fno-omit-frame-pointer -fno-inline)  # for profiling
```

所有警告标志都包在 `$<$<COMPILE_LANGUAGE:CXX>:...>` generator expression 里，确保
**不应用到 CUDA 文件**——nvcc 对 `-Wpedantic` 这类 GCC 选项兼容性差。`-Wfatal-errors`
让第一个错误就停，避免错误雪崩。`-ffast-math -mavx2` 是为距离计算做的浮点/向量指令
优化。`-DNOHUGEPAGES` 是给 thirdparty（如 metis）用的宏。

```
77	set(DVSTOR_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
78	set(DVSTOR_CONDA_LIB_DIR "/home/xjs/anaconda3/lib" CACHE PATH
79	    "Optional local libstdc++ search path used on the development machine")
80	set(DVSTOR_METIS_ROOT "" CACHE PATH "Optional METIS installation root")
81	set(DVSTOR_FAISS_BLAS_VENDOR "Generic" CACHE STRING
82	    "BLAS vendor for CPU Faiss offline tools; Generic avoids nested MKL/OpenMP runtimes")
```

`DVSTOR_CONDA_LIB_DIR` 是开发机特定的 libstdc++ 路径（硬编码 `/home/xjs/anaconda3/lib`
作为默认值），下文 `cmake/DvstorDependencies.cmake:28-31` 把它加到 linker rpath。
`DVSTOR_FAISS_BLAS_VENDOR=Generic` 故意不用 MKL，避免和 Faiss 自带的 OpenMP 嵌套
（`README.md:124-126` 与 `experiment/README.md:64` 都强调 BLAS 线程为 1）。

```
84	list(APPEND CMAKE_MODULE_PATH
85	    "${DVSTOR_SOURCE_ROOT}"
86	    "${DVSTOR_SOURCE_ROOT}/cmake")
87	
88	include(DvstorDependencies)
89	include(DvstorTargetHelpers)
90	if (DVSTOR_BUILD_OFFLINE_TOOLS)
91	    find_package(FaissCPU REQUIRED)
92	endif ()
93	
94	include_directories(${DVSTOR_SOURCE_ROOT}/src)
95	include_directories(${DVSTOR_SOURCE_ROOT}/thirdparty)
96	add_subdirectory(rdma-library)
97	add_subdirectory(thirdparty)
```

把仓库根 + `cmake/` 加进 module path，include 两个自定义模块。`find_package(FaissCPU)`
用的是 `cmake/FindFaissCPU.cmake`（35 行）。`include_directories` 是全局的，所有
target 默认能 include `src/` 和 `thirdparty/`——这虽然不够"现代 CMake"，但与
`dvstor_target_common_includes`（在 `cmake/DvstorTargetHelpers.cmake:3-14`）的显式
include 形成双保险。`rdma-library` 和 `thirdparty` 各自 `add_subdirectory`。

### 4.3 源码列表与目标定义（99-189 行）

这一段把所有 `.cc` 显式列出，没有任何 `file(GLOB)`。这是刻意的——`docs/source_layout.md`
的"变更约束"要求"结构重构保持公开 ABI，并通过构建与性能测试确认跨实现单元边界"，
显式列表让任何源文件增删都必须改 CMake，触发 reviewer 注意。

#### `DVSTOR_MEMORY_NODE_SOURCES`（99-123 行）

```
99	set(DVSTOR_MEMORY_NODE_SOURCES
100	    src/gpu_search/index_format.cc
101	    src/gpu_search/pq_index.cc
102	    src/memory_node/memory_node.cc
103	    src/memory_node/peer_rdma.cc
104	    src/memory_node/peer_rpc/client_requests.cc
105	    src/memory_node/peer_rpc/request_handlers.cc
106	    src/memory_node/peer_rpc/runtime.cc
107	    src/memory_node/peer_rpc/workers.cc
108	    src/memory_node/storage_owner_index/allocation.cc
109	    src/memory_node/storage_owner_index/candidate_search.cc
110	    src/memory_node/storage_owner_index/graph_access.cc
111	    src/memory_node/storage_owner_index/graph_mutation.cc
112	    src/memory_node/storage_owner_index/reverse_batch.cc
113	    src/memory_node/storage_owner_maintenance/graph_tasks.cc
114	    src/memory_node/storage_owner_maintenance/queue.cc
115	    src/memory_node/storage_owner_maintenance/runtime.cc
116	    src/memory_node/storage_owner_maintenance/worker.cc
117	    src/memory_node/storage_owner_runtime/batch_execution.cc
118	    src/memory_node/storage_owner_runtime/lifecycle.cc
119	    src/memory_node/storage_owner_runtime/wire_protocol.cc
120	    src/memory_node/storage_owner_runtime/workers.cc
121	    src/vamana/anchor_index.cc
122	    src/vamana/adaptive_route_table.cc
123	)
```

存储节点复用 `gpu_search/index_format.cc`、`gpu_search/pq_index.cc`、
`vamana/anchor_index.cc`、`vamana/adaptive_route_table.cc`——这 4 个文件是"格式与
模型"层，不含 GPU/CUDA 符号，所以能在 storage-only 构建里编译。**注意
`DVSTOR_MEMORY_NODE_SOURCES` 里没有任何 `.cu` 或 `persistent_engine/*.cc`**——存储
节点不跑 GPU 查询引擎。

#### 离线工具源码列表（125-153 行）

```
125	set(DVSTOR_OFFLINE_BUILDER_SOURCES
126	    tools/vamana_offline/anchor_builder.cc
127	    tools/vamana_offline/config.cc
128	    tools/vamana_offline/dataset_io.cc
129	    tools/vamana_offline/graph.cc
130	    tools/vamana_offline/partitioning.cc
131	    tools/vamana_offline/progress.cc
132	    tools/vamana_offline/recall_check.cc
133	    tools/vamana_offline/shard_writer.cc
134	)
135	
136	set(DVSTOR_PQ_INDEXER_SOURCES
137	    src/gpu_search/index_format.cc
138	    src/gpu_search/pq_index.cc
139	    src/vamana/anchor_index.cc
140	    tools/vamana_offline/pq_indexer.cc
141	)
142	
143	set(DVSTOR_BREAKDOWN_SUPPORT_SOURCES
144	    tools/breakdown_benchmark/dataset.cc
145	    tools/breakdown_benchmark/maintenance_log.cc
146	    tools/breakdown_benchmark/report.cc
147	)
148	
149	set(DVSTOR_BREAKDOWN_BENCHMARK_SOURCES
150	    tools/breakdown_benchmark/args.cc
151	    tools/breakdown_benchmark/progress.cc
152	    tools/breakdown_benchmark/workload.cc
153	)
```

四个列表对应五类离线工具（offline builder、anchor sidecar builder、pq indexer、
legacy converter、generate_sift101m_recall_data）和 benchmark。`DVSTOR_PQ_INDEXER_SOURCES`
复用 `gpu_search/pq_index.cc` 与 `index_format.cc`——PQ 训练逻辑在 GPU 侧声明，但
**实现是纯 CPU Faiss**，因此可以离线跑。

#### `dvstor_add_offline_tools` 函数（155-189 行）

```
155	function(dvstor_add_offline_tools)
156	    add_executable(vamana_offline_builder
157	        tools/vamana_offline_builder.cc
158	        ${DVSTOR_OFFLINE_BUILDER_SOURCES}
159	    )
160	    dvstor_target_tool_includes(vamana_offline_builder)
161	    target_link_libraries(vamana_offline_builder rdma_library)
162	    target_link_metis(vamana_offline_builder)
163	
164	    add_executable(vamana_anchor_sidecar_builder
165	        tools/vamana_anchor_sidecar_builder.cc
166	        src/vamana/anchor_index.cc
167	    )
168	    dvstor_target_tool_includes(vamana_anchor_sidecar_builder)
169	    target_link_libraries(vamana_anchor_sidecar_builder rdma_library)
170	
171	    add_executable(vamana_pq_indexer
172	        tools/vamana_pq_indexer.cc
173	        ${DVSTOR_PQ_INDEXER_SOURCES}
174	    )
175	    dvstor_target_tool_includes(vamana_pq_indexer)
176	    target_link_libraries(vamana_pq_indexer rdma_library FaissCPU::FaissCPU)
177	
178	    add_executable(vamana_legacy_index_converter
179	        tools/vamana_legacy_index_converter.cc
180	        tools/legacy_index/migrator.cc
181	    )
182	    dvstor_target_tool_includes(vamana_legacy_index_converter)
183	    target_link_libraries(vamana_legacy_index_converter rdma_library)
184	
185	    add_executable(generate_sift101m_recall_data
186	        tools/generate_sift101m_recall_data.cc
187	    )
188	    dvstor_target_tool_includes(generate_sift101m_recall_data)
189	endfunction()
```

五个离线可执行文件：

| 目标 | 入口 | 链接 | METIS | Faiss |
| --- | --- | --- | --- | --- |
| `vamana_offline_builder` | `tools/vamana_offline_builder.cc` | `rdma_library` | 是（`target_link_metis`） | 否 |
| `vamana_anchor_sidecar_builder` | `tools/vamana_anchor_sidecar_builder.cc` | `rdma_library` | 否 | 否 |
| `vamana_pq_indexer` | `tools/vamana_pq_indexer.cc` | `rdma_library` + `FaissCPU::FaissCPU` | 否 | 是 |
| `vamana_legacy_index_converter` | `tools/vamana_legacy_index_converter.cc` | `rdma_library` | 否 | 否 |
| `generate_sift101m_recall_data` | `tools/generate_sift101m_recall_data.cc` | （无显式链接） | 否 | 否 |

注意 `dvstor_target_tool_includes`（`cmake/DvstorTargetHelpers.cmake:16-19`）会先
`target_include_directories(... PRIVATE ${DVSTOR_SOURCE_ROOT})`（让工具能 include
`tools/...` 自身路径），再调用 `dvstor_target_common_includes` 加 `src/`、
`rdma-library/`、`thirdparty/`。`target_link_metis`（`cmake/DvstorTargetHelpers.cmake:21-32`）
根据 `DVSTOR_HAVE_METIS` 决定是否链接 METIS/GKlib，并定义宏
`DVSTOR_HAVE_METIS=0/1`，让 C++ 代码用 `#if DVSTOR_HAVE_METIS` 选择性编译 metis 路径。

### 4.4 storage-only 分支（191-204 行）

```
191	if (DVSTOR_STORAGE_NODE_ONLY)
192	    message(STATUS "Storage-node build: CUDA and DOCA are disabled; CPU offline tools=${DVSTOR_BUILD_OFFLINE_TOOLS}")
193	    add_executable(dvstor_memory_node
194	        src/memory_node_main.cc
195	        ${DVSTOR_MEMORY_NODE_SOURCES}
196	        src/common/core_assignment.cc
197	        src/common/timing.cc
198	        src/service/index_metadata.cc
199	    )
200	    dvstor_target_common_includes(dvstor_memory_node)
201	    target_link_libraries(dvstor_memory_node rdma_library)
202	    if (DVSTOR_BUILD_OFFLINE_TOOLS)
203	        dvstor_add_offline_tools()
204	    endif ()
```

storage-only 构建产物就两类：`dvstor_memory_node`（链接 `rdma_library`，无 CUDA、
无 DOCA）和可选的离线工具。`dvstor_memory_node` 源码 = `DVSTOR_MEMORY_NODE_SOURCES`
+ `src/common/core_assignment.cc` + `src/common/timing.cc` + `src/service/index_metadata.cc`
+ 入口 `memory_node_main.cc`。这正好印证 `README.md:124-125`："存储服务本身只依赖
CPU、RDMA、Boost 和 TBB。"

### 4.5 compute 分支：DOCA 探测（205-231 行）

```
205	else ()
206	    find_path(DVSTOR_DOCA_INCLUDE_DIR doca_gpunetio.h
207	        HINTS "${DVSTOR_DOCA_ROOT}/include")
208	    find_library(DVSTOR_DOCA_GPUNETIO_LIBRARY doca_gpunetio
209	        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
210	    find_library(DVSTOR_DOCA_VERBS_LIBRARY doca_verbs
211	        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
212	    find_library(DVSTOR_DOCA_COMMON_LIBRARY doca_common
213	        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
214	    find_library(DVSTOR_DOCA_RDMA_LIBRARY doca_rdma
215	        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
216	    find_library(DVSTOR_MLX5_LIBRARY mlx5)
217	    if (NOT DVSTOR_DOCA_INCLUDE_DIR OR
218	        NOT DVSTOR_DOCA_GPUNETIO_LIBRARY OR
219	        NOT DVSTOR_DOCA_VERBS_LIBRARY OR
220	        NOT DVSTOR_DOCA_COMMON_LIBRARY OR
221	        NOT DVSTOR_DOCA_RDMA_LIBRARY OR
222	        NOT DVSTOR_MLX5_LIBRARY)
223	        message(FATAL_ERROR
224	            "The compute-node runtime requires a complete DOCA GPUNetIO installation")
225	    endif ()
226	    set(DVSTOR_HAVE_GPUNETIO ON)
227	    list(FIND CMAKE_CUDA_ARCHITECTURES 52 DVSTOR_CUDA_SM52_INDEX)
228	    if (NOT DVSTOR_CUDA_SM52_INDEX EQUAL -1)
229	        message(WARNING "GPUNetIO requires sm_60+; replacing cached sm_52 with sm_80")
230	        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
231	    endif ()
```

DOCA 探测找 6 个库：`doca_gpunetio`、`doca_verbs`、`doca_common`、`doca_rdma`、
`mlx5`（mlx5 用户态驱动）。任意一个缺失就 FATAL_ERROR，呼应 `README.md:21` 的"系统
没有 CPU 查询 fallback"——DOCA 缺失就没法跑计算节点。`DVSTOR_HAVE_GPUNETIO` 后面
用来条件编译 GPUNetIO 路径。`sm_52` 被强制替换为 `sm_80`，因为 GPUNetIO 要求
sm_60+（`README.md:229` 的警告）。

### 4.6 compute 分支：GPU kernel 库（233-262 行）

```
233	    # GPU-centric query engine kernels. Keep this list explicit so retired
234	    # CPU-driven CUDA paths cannot silently re-enter the runtime.
235	    set(DVSTOR_GPU_SOURCES
236	        src/gpu_search/persistent_kernel.cu
237	    )
238	    if (DVSTOR_HAVE_GPUNETIO)
239	        list(APPEND DVSTOR_GPU_SOURCES src/gpu/gpunetio_probe.cu)
240	    endif ()
241	    if (DVSTOR_GPU_SOURCES)
242	        add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
243	        dvstor_target_common_includes(dvstor_gpu_kernels PUBLIC)
244	        target_link_libraries(dvstor_gpu_kernels
245	            CUDA::cudart
246	            rdma_library
247	        )
248	        if (DVSTOR_HAVE_GPUNETIO)
249	            target_include_directories(dvstor_gpu_kernels PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
250	            target_compile_definitions(dvstor_gpu_kernels PUBLIC DVSTOR_HAVE_GPUNETIO=1)
251	        endif ()
252	        set_target_properties(dvstor_gpu_kernels PROPERTIES
253	            CUDA_SEPARABLE_COMPILATION ON
254	            CUDA_RESOLVE_DEVICE_SYMBOLS ON
255	            POSITION_INDEPENDENT_CODE ON
256	        )
257	        target_compile_options(dvstor_gpu_kernels PRIVATE
258	            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
259	            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
260	            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
261	        )
262	    endif ()
```

`dvstor_gpu_kernels` 是一个静态库，源码只有 1-2 个 `.cu`：`persistent_kernel.cu`
（必选）和 `gpunetio_probe.cu`（GPUNetIO 时）。`CUDA_SEPARABLE_COMPILATION ON` +
`CUDA_RESOLVE_DEVICE_SYMBOLS ON` + `POSITION_INDEPENDENT_CODE ON` 三件套是为了让
device linker 在静态库里能正确解析 device symbol（持久化 kernel 用了 extended
lambda 和 rdc）。注释 `233-234` 是关键纪律："Keep this list explicit so retired
CPU-driven CUDA paths cannot silently re-enter the runtime."——禁止用 GLOB，避免
被删除的旧 CUDA 路径悄悄回来。

### 4.7 compute 分支：runtime 库（264-322 行）

```
264	    # Compute-node runtime. This is intentionally explicit: the dev branch has
265	    # one query engine, with CPU threads limited to admission and update RPCs.
266	    set(DVSTOR_RUNTIME_SOURCES
267	        src/common/core_assignment.cc
268	        src/common/timing.cc
269	        src/gpu_search/delta_index.cc
270	        src/gpu_search/dynamic_route_overlay.cc
271	        src/gpu_search/navigation_bootstrapper.cc
272	        src/gpu_search/index_format.cc
273	        src/gpu_search/persistent_engine.cc
274	        src/gpu_search/persistent_engine/completion.cc
275	        src/gpu_search/persistent_engine/construction.cc
276	        src/gpu_search/persistent_engine/delta_publication.cc
277	        src/gpu_search/persistent_engine/health.cc
278	        src/gpu_search/persistent_engine/lifecycle.cc
279	        src/gpu_search/persistent_engine/query_execution.cc
280	        src/gpu_search/persistent_engine/routing.cc
281	        src/gpu_search/persistent_engine/storage_reclaim.cc
282	        src/gpu_search/pq_index.cc
283	        src/gpu_search/types.cc
284	        src/service/base_owner_map.cc
285	        src/service/compute_service/index_commands.cc
286	        src/service/compute_service/lifecycle.cc
287	        src/service/compute_service/search.cc
288	        src/service/compute_service/storage_owner/completion.cc
289	        src/service/compute_service/storage_owner/lifecycle.cc
290	        src/service/compute_service/storage_owner/public_mutations.cc
291	        src/service/compute_service/storage_owner/sender.cc
292	        src/service/index_metadata.cc
293	        src/vamana/anchor_index.cc
294	        src/vamana/adaptive_route_table.cc
295	    )
296	    if (DVSTOR_HAVE_GPUNETIO)
297	        list(APPEND DVSTOR_RUNTIME_SOURCES src/gpu/gpunetio_transport.cc)
298	        set_source_files_properties(src/gpu/gpunetio_transport.cc PROPERTIES
299	            COMPILE_OPTIONS "-Wno-deprecated-declarations;-Wno-pedantic")
300	    endif ()
301	
302	    add_library(dvstor_runtime STATIC ${DVSTOR_RUNTIME_SOURCES})
303	    dvstor_target_common_includes(dvstor_runtime PUBLIC)
304	    target_link_libraries(dvstor_runtime
305	        rdma_library
306	        CUDA::cudart
307	    )
308	    if (DVSTOR_HAVE_GPUNETIO)
309	        target_include_directories(dvstor_runtime PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
310	        target_compile_definitions(dvstor_runtime PUBLIC DVSTOR_HAVE_GPUNETIO=1)
311	        target_link_libraries(dvstor_runtime
312	            ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
313	            ${DVSTOR_DOCA_VERBS_LIBRARY}
314	            ${DVSTOR_DOCA_COMMON_LIBRARY}
315	            ${DVSTOR_DOCA_RDMA_LIBRARY}
316	            ${DVSTOR_MLX5_LIBRARY}
317	        )
318	    endif ()
319	    if (DVSTOR_GPU_SOURCES)
320	        target_link_libraries(dvstor_runtime dvstor_gpu_kernels)
321	    endif ()
322	    target_compile_options(dvstor_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)
```

`dvstor_runtime` 是计算节点的核心静态库。对照 `DVSTOR_MEMORY_NODE_SOURCES`，可以看到
**两份源码列表完全不重叠**：runtime 有 `persistent_engine.*`、`compute_service/*`、
`base_owner_map.cc`、`gpunetio_transport.cc`（GPUNetIO 时），存储节点一个都没有；
存储节点的 `peer_rpc/`、`storage_owner_index/`、`storage_owner_maintenance/`、
`storage_owner_runtime/`，runtime 也一个都没有。共享的只有 `index_format.cc`、
`pq_index.cc`、`anchor_index.cc`、`adaptive_route_table.cc`、`index_metadata.cc`、
`core_assignment.cc`、`timing.cc`——这些是纯 CPU 的"格式与模型"层。

`-fcoroutines` 显式开启 C++20 coroutine（GCC 11+ 在 `-std=c++20` 下仍需此 flag，
GCC 12+ 才默认）。

### 4.8 compute 分支：可执行目标（324-381 行）

```
324	    if (DVSTOR_BUILD_EXECUTABLES OR DVSTOR_BUILD_TESTS)
325	        add_library(dvstor_breakdown_support STATIC ${DVSTOR_BREAKDOWN_SUPPORT_SOURCES})
326	        dvstor_target_tool_includes(dvstor_breakdown_support)
327	        target_link_libraries(dvstor_breakdown_support PUBLIC dvstor_runtime)
328	    endif ()
329	
330	    add_executable(dvstor_memory_node
331	        src/memory_node_main.cc
332	        ${DVSTOR_MEMORY_NODE_SOURCES}
333	        src/common/core_assignment.cc
334	        src/common/timing.cc
335	        src/service/index_metadata.cc
336	    )
337	    dvstor_target_common_includes(dvstor_memory_node)
338	    target_link_libraries(dvstor_memory_node rdma_library)
339	
340	    if (DVSTOR_BUILD_EXECUTABLES)
341	        add_executable(dvstor_compute_node src/main.cc)
342	        target_link_libraries(dvstor_compute_node dvstor_runtime)
343	
344	        if (DVSTOR_HAVE_GPUNETIO)
345	            add_executable(dvstor_gpunetio_probe tools/gpunetio_probe.cc)
346	            dvstor_target_common_includes(dvstor_gpunetio_probe)
347	            target_include_directories(dvstor_gpunetio_probe PRIVATE ${DVSTOR_DOCA_INCLUDE_DIR})
348	            target_compile_definitions(dvstor_gpunetio_probe PRIVATE ALLOW_EXPERIMENTAL_API=1)
349	            target_compile_options(dvstor_gpunetio_probe PRIVATE
350	                -Wno-deprecated-declarations
351	                -Wno-pedantic
352	            )
353	            target_link_libraries(dvstor_gpunetio_probe
354	                rdma_library
355	                CUDA::cudart
356	                ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
357	                ${DVSTOR_DOCA_VERBS_LIBRARY}
358	                ${DVSTOR_DOCA_COMMON_LIBRARY}
359	                ${DVSTOR_MLX5_LIBRARY}
360	            )
361	
362	            add_executable(dvstor_gpunetio_loopback_probe tools/gpunetio_loopback_probe.cc)
363	            dvstor_target_common_includes(dvstor_gpunetio_loopback_probe)
364	            target_link_libraries(dvstor_gpunetio_loopback_probe dvstor_runtime)
365	        endif ()
366	
367	        add_executable(dvstor_breakdown_benchmark
368	            tools/dvstor_breakdown_benchmark.cc
369	            ${DVSTOR_BREAKDOWN_BENCHMARK_SOURCES}
370	        )
371	        dvstor_target_tool_includes(dvstor_breakdown_benchmark)
372	        target_link_libraries(dvstor_breakdown_benchmark dvstor_breakdown_support)
373	
374	        add_executable(dvstor_sift101m_long_insert_recall
375	            tools/dvstor_sift101m_long_insert_recall.cc
376	            tools/breakdown_benchmark/args.cc
377	            tools/breakdown_benchmark/progress.cc
378	        )
379	        dvstor_target_tool_includes(dvstor_sift101m_long_insert_recall)
380	        target_link_libraries(dvstor_sift101m_long_insert_recall dvstor_runtime)
381	    endif ()
```

注意几件事：

1. **`dvstor_memory_node` 在 compute 分支里也构建一次**（330-338 行），源码与
   storage-only 分支完全一致。这样在 GPU 机器上也能跑存储节点（一体化部署）。
2. `dvstor_compute_node` 入口只有 `src/main.cc` 一个文件，所有逻辑在 `dvstor_runtime`
   库里——这是 PImpl 的体现（第 11 课）。
3. `dvstor_gpunetio_probe` 是独立可执行文件（`tools/gpunetio_probe.cc`），不依赖
   runtime，直接链接 DOCA 库做 QP probe。它的 `ALLOW_EXPERIMENTAL_API=1` 是 DOCA
   实验性 API 必需的宏。`dvstor_gpunetio_loopback_probe` 则链接 runtime，做回环测试。
4. `dvstor_breakdown_benchmark` 链接 `dvstor_breakdown_support`，后者再 PUBLIC 链接
   `dvstor_runtime`——所以 benchmark 传递依赖整个 runtime。
5. `dvstor_sift101m_long_insert_recall` 复用 `args.cc`/`progress.cc` 但不走
   `dvstor_breakdown_support`，是更轻量的长跑召回工具。

### 4.9 测试与状态打印（383-397 行）

```
383	    if (DVSTOR_BUILD_OFFLINE_TOOLS)
384	        dvstor_add_offline_tools()
385	    endif ()
386	
387	    if (DVSTOR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
388	        enable_testing()
389	        add_subdirectory(test)
390	    endif ()
391	
392	    # status prints
393	    get_target_property(compile_options dvstor_runtime COMPILE_OPTIONS)
394	    message(STATUS "Compile options: ${compile_options}")
395	    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
396	    message(STATUS "DOCA GPUNetIO backend: ${DVSTOR_HAVE_GPUNETIO}")
397	endif ()
```

compute 分支结尾：可选构建离线工具、可选启用测试、打印三条状态信息（compile
options、CUDA arch、GPUNetIO 后端是否启用）。这三条 `message(STATUS)` 是用户验证
CMake 配置是否符合预期的最快途径。

### 4.10 `cmake/` 模块补充

`cmake/DvstorDependencies.cmake`（135 行）的核心是 `dvstor_check_metis_link`
函数（6-23 行）：用 `check_cxx_source_compiles` 真实编译一段调用
`METIS_SetDefaultOptions` 的代码，验证 METIS/GKlib 能否在本机链接。`AUTO` 模式下
（44-114 行），先找 bundled `thirdparty/metis64`，链接失败且库确实来自 bundled
路径时（57-60 行的 `_DVSTOR_BUNDLED_METIS_INDEX` 判断），回退到系统 `/usr`/`/usr/local`
的 METIS，再失败就 WARNING 并保留 `balanced/bfs`。`ON` 模式下找不到或链接失败都
FATAL_ERROR。这正是 `README.md:131-136` 描述的行为。

`cmake/DvstorTargetHelpers.cmake`（33 行）三个函数上文已用：
`dvstor_target_common_includes`（默认 PRIVATE，可传 PUBLIC）、
`dvstor_target_tool_includes`（先加根目录再调 common）、`target_link_metis`（条件
链接 METIS/GKlib + 定义 `DVSTOR_HAVE_METIS` 宏）。

`cmake/FindFaissCPU.cmake`（35 行）是自定义 Find 模块，故意只找 CPU 版 libfaiss，
避免加载 Faiss CMake 导出里可能包含的 `faiss_gpu_objs`（`README.md:126`）。

## 5. `.clangd` 的作用

`.clangd` 全文 32 行：

```
1	CompileFlags:
2	  CompilationDatabase: build
3	
4	---
5	If:
6	  PathMatch: 'src/.*\.(cu|cuh)'
7	CompileFlags:
8	  Compiler: clang++
9	  Remove:
10	    - -forward-unknown-to-host-compiler
11	    - --options-file
12	    - 'CMakeFiles/*'
13	    - '--generate-code=*'
14	    - '-Xcompiler=*'
15	    - --expt-relaxed-constexpr
16	    - --expt-extended-lambda
17	    - '-rdc=*'
18	    - -x
19	  Add:
20	    - -xcuda
21	    - --cuda-path=/usr/local/cuda
22	    - --cuda-gpu-arch=sm_80
23	    - --no-cuda-version-check
24	    - -I../src
25	    - -I../thirdparty
26	    - -I../rdma-library
27	    - -I../rdma-library/.
28	    - -I/opt/mellanox/doca/include
29	    - -I/usr/local/cuda/include
30	    - -isystem
31	    - /home/xjs/anaconda3/include
```

作用：让 clangd（IDE/编辑器的语义高亮、跳转、补全后端）正确处理 CUDA 文件。

- 第 2 行：默认从 `build/compile_commands.json` 读取编译选项（CMake 由
  `CMAKE_EXPORT_COMPILE_COMMANDS=ON` 生成）。
- 第 5-6 行：对 `src/` 下的 `.cu`/`.cuh` 文件单独覆盖。
- 第 8 行：强制用 `clang++` 而不是 `nvcc`——clangd 不能直接消费 nvcc 的
  `--generate-code`/`-Xcompiler`/`-rdc` 等 flag，必须用 clang 自己的 CUDA frontend。
- 第 9-18 行 `Remove`：从 `compile_commands.json` 里剥掉 nvcc 专属 flag，避免 clang
  报错。
- 第 19-31 行 `Add`：手动给 clang 加上 CUDA mode、CUDA path、sm_80 arch、所有 include
  路径（src、thirdparty、rdma-library、DOCA、CUDA、Conda）。

这条配置是开发流的重要一环：没有它，编辑器里 `.cu`/`.cuh` 会一片红。第 17–21 课
读者在阅读 `persistent_kernel*` 时如果遇到 IDE 报错，多半是 `.clangd` 没生效或
`build/compile_commands.json` 过期。

## 6. `docs/` 两份文档的定位

`docs/` 只有两个文件，但都是设计契约。

### 6.1 `docs/source_layout.md` — 模块边界与变更约束

44 行。结构：

- **模块边界**（5-17 行）：把运行时按职责而非历史功能堆叠划分，列出 8 个目录的
  一句话职责。这是第 3 节"目录与模块边界"的权威来源。
- **实现单元**（19-28 行）：规定 CPU 运行时用普通 `.hh`/`.cc`，每个 `.cc` 是可独立
  编译、可被 clangd 直接解析的职责单元；模块共享的非公开声明放本目录 `detail.hh`；
  GPU 引擎用 `impl.hh` 定义 PImpl 状态，公开门面只保留装配和转发。"项目不使用
  `.ipp` 文本片段，也不依赖 `__INCLUDE_LEVEL__` 改变源码语义"——禁止预处理器魔法。
  CUDA 设备代码由 `persistent_kernel.cu` 形成单一 translation unit，目录内 `.cuh`
  是正常头文件而非 include 片段。
- **动态更新语义**（30-35 行）：GPU delta 是"可见性 overlay"而不是第二套静态索引，
  "也不存在虚假的在线 base compaction API"。退休用 `delta_reclaim_batches` 遥测
  表达。
- **变更约束**（37-43 行）：这是**硬约束**，5 条：
  1. 不在查询热路径分配内存、创建 stream、创建 QP 或同步等待 CPU fallback；
  2. 不新增 CPU 图导航或隐式传输回退；
  3. 索引格式和 RPC wire protocol 必须由显式 schema/version 保护；
  4. 结构重构保持公开 ABI，并通过构建与性能测试确认跨实现单元边界；
  5. 修改职责边界时同步更新单元测试与本文档。

第 1、2 条呼应 `README.md:19-21` 和 `docs/gpu_persistent_engine.md:177-188` 的
"无 CPU fallback"。第 3 条是第 7、8、26 课的 schema/version 体系根据。第 4、5 条
是 contributor 工作流：改结构要跑构建+性能测试，改边界要同步更新测试和这份文档。

### 6.2 `docs/gpu_persistent_engine.md` — GPU 中心化引擎设计

188 行，是引擎的"宪法"。分 8 节：

- **设计边界**（1-18 行）：数据面只有 GPU；固定契约 schema 15、L2、`header+id+generation+exact`、
  紧凑图记录 ≤512 字节、5 字节指针、OPQ+PQ32。
- **启动**（20-31 行）：6 步启动校验（metadata → 存储节点加载 `.dat` 与
  `.pq32.codes` → 批量 RDMA PQ code → 抽样首中尾校验 → GPUNetIO QP probe → 全部
  验证后才启动持久化 kernel）。启动期 CPU-posted GPUDirect RDMA 只用于连续码流导入。
- **请求调度**（33-48 行）：提交/完成环有界；查询 CTA、更新 control CTA、QP-owner
  warp 是隔离的长期资源；CPU admission 只领 slot、拷 query、发 ring descriptor。
- **GPU 图遍历**（50-75 行）：8 阶段（OPQ → PQ LUT → 入口打分 → 选未展开候选 →
  并发远端图读 → 解码 RemotePtr → PQ 评分 → 裁剪/收敛/精排）。静态入口在启动时
  聚集成连续兜底层；每分片另有 8 个 storage-owner 权威固定槽，通过 4 KiB control
  page 发布带 checksum/seqlock 的快照。图读 miss 落入每查询 scratch；默认 graph
  cache 容量为 0。
- **精确重排**（77-88 行）：5 步（选 `gpu-final-rerank-width` 候选 → GPUNetIO 拉
  fixed record → dtype 解码 → 精确 L2 + delta 合并 + 过滤 → top-k）。"PQ 误差不会
  直接污染最终距离，但会影响候选是否进入精排。"
- **动态一致性**（90-134 行）：storage-owner commit + GPU epoch publish 的 6 步
  （stage1 发布 → ACK → mapped pinned staging → control CTA 编码 → 原子发布 delta
  count/epoch → stage2 反向边 + durable sequence）。查询 admission 绑定 snapshot
  epoch；RCU 退休；schema-15 反向边无 generation，动态物理地址不复用。两阶段最终
  等价边界明确写在 132-134 行。
- **内存预算**（136-159 行）：显式预算 = `gpu-memory-limit-gb - gpu-memory-reserve-gb`；
  启动前统一核算 7 类数组；SIFT1B 预算表。
- **性能原则**（161-174 行）：6 条性能原则 + "判断达标必须同时检查 QPS/recall/GPU
  util/QP 错误/direct-path failure/图读数/精确读数"。
- **故障策略**（176-188 行）：5 类 fail-stop + "系统没有 CPU 查询 fallback"。

这份文档的每一节都对应后续课程的具体源码段，读者在阅读第 11–22 课时应随时回查
这份文档。

## 7. 启动/运行/验证流程串起后续课程

把 `experiment/README.md` 和 `README.md` 的运行流程合在一起，对应到课程：

```
[离线构建索引]
  experiment/build_sift100m_index.sh
    ├─ vamana_offline_builder       → 第 29 课
    ├─ vamana_pq_indexer            → 第 9、29 课
    └─ vamana_anchor_sidecar_builder → 第 6、29 课

[转换旧索引]
  experiment/convert_legacy_sift100m_index.sh
    └─ vamana_legacy_index_converter → 第 29 课

[启动存储节点]
  experiment/start_all_memory_nodes.sh
    └─ dvstor_memory_node           → 第 23 课
        ├─ memory_node.cc            → 第 23 课
        ├─ peer_rdma.cc              → 第 23 课
        ├─ peer_rpc/*                → 第 24 课
        ├─ storage_owner_index/*     → 第 25 课
        ├─ storage_owner_maintenance/* → 第 26 课
        └─ storage_owner_runtime/*   → 第 26 课

[启动计算节点]
  dvstor_compute_node (src/main.cc)  → 第 27 课
    └─ dvstor_runtime               → 第 11–22、27、28 课
        ├─ persistent_engine*        → 第 11、14、15、16 课
        ├─ persistent_kernel*        → 第 17–21 课
        ├─ gpu/gpunetio_*            → 第 22 课
        ├─ compute_service/*         → 第 27 课
        ├─ compute_service/storage_owner/* → 第 28 课
        ├─ vamana/adaptive_route_table → 第 10 课
        └─ service/base_owner_map   → 第 8、28 课

[运行 benchmark]
  experiment/run_recall.sh / run_breakdown.sh
    └─ dvstor_breakdown_benchmark   → 第 30 课
        └─ tools/breakdown_benchmark/* → 第 30 课

[停止]
  experiment/stop_memory_nodes.sh
```

`experiment/README.md:114-129` 的启动校验流程也值得记住：启动脚本会"验证 schema、
分片数、R、dtype、PQ checksum 和角色所需文件，不兼容时在申请大块注册内存前退出"。
这是 `docs/gpu_persistent_engine.md:20-31` 启动 6 步的 shell 实现。

`experiment/README.md:131-136` 还有一条 schema-15 部署约束：反向边请求只携带物理
指针、没有 generation，因此存储端不复用已删除动态节点的物理地址，每次 insert/upsert
都消耗新空间，部署必须预留 memory-node 容量。这与 `docs/gpu_persistent_engine.md:122-127`
一致。

## 8. 关键数据结构与流程图

### 8.1 系统总体架构图

```
                      ┌─────────────────────────────────────────────────────┐
                      │             计算节点（GPU 中心化查询）                 │
                      │                                                     │
                      │  CPU admission thread                               │
                      │    │  1. 领 slot / 拷 query / 发 ring descriptor    │
                      │    ▼                                                 │
                      │  ┌───────────────────────────────────────────┐      │
                      │  │ 持久化 CUDA kernel (persistent_kernel.cu)  │      │
                      │  │  ┌─────────────────────────────────────┐  │      │
                      │  │  │ 查询 CTA  : beam / visited / LUT     │  │      │
                      │  │  │ 更新 control CTA : OPQ/PQ 编码       │  │      │
                      │  │  │ QP-owner warp : GPUNetIO 发包        │  │      │
                      │  │  └─────────────────────────────────────┘  │      │
                      │  │  常驻 PQ32 code (32B/vec)                  │      │
                      │  │  delta overlay (L0, 256 MiB 默认)          │      │
                      │  │  动态路由 8 槽 (storage-canonical 快照)    │      │
                      │  │  可选 graph cache / exact cache (默认关)   │      │
                      │  └───────────────┬───────────────────────────┘      │
                      │                  │ GPUNetIO RDMA read               │
                      │                  │ (GPU 显存 ← 存储节点注册内存)     │
                      │  CPU completion thread                             │
                      │    │  组装 top-k / 归还 slot                       │
                      └────┼────────────────┬───────────────────────────────┘
                           │                │
                           │ control page   │ 紧凑图记录 (≤512B) / fixed record
                           │ (4 KiB, 8 槽   │ (header+id+generation+exact vector)
                           │  canonical     │
                           │  route 快照)   │
                           ▼                ▼
        ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
        │  存储节点 1         │  │  存储节点 2         │  │  存储节点 N         │
        │  .dat / .idmap     │  │  .dat / .idmap     │  │  .dat / .idmap     │
        │  .pq32.codes       │  │  .pq32.codes       │  │  .pq32.codes       │
        │  storage owner     │  │  storage owner     │  │  storage owner     │
        │   ├ index/         │  │   ├ index/         │  │   ├ index/         │
        │   ├ maintenance/   │  │   ├ maintenance/   │  │   ├ maintenance/   │
        │   ├ runtime/       │  │   ├ runtime/       │  │   ├ runtime/       │
        │   └ peer_rpc/      │  │   └ peer_rpc/      │  │   └ peer_rpc/      │
        │  peer RDMA ◄──────►│  │  peer RDMA ◄──────►│  │  peer RDMA ◄──────►│
        └────────────────────┘  └────────────────────┘  └────────────────────┘
                  ▲                       ▲                       ▲
                  └───────────────────────┴───────────────────────┘
                          peer RPC（stage2 跨分片反向边、idmap 查询）
```

### 8.2 单查询路径 6 步时序

```
 CPU admission          持久化 kernel                    存储节点
 ─────────────          ──────────────                    ────────
 1. 领 slot
    拷 query ───────►  提交 ring
                      2. OPQ + PQ LUT
                      3. 入口打分
                         (静态 anchors + 8 动态槽)
                      4. GPUNetIO 并发拉图记录  ─────►  RDMA read
                                                  ◄────  紧凑图记录 (≤512B)
                      5. PQ 评分 + beam 更新
                         (循环 4-5 至收敛/max-exp)
                      6. GPUNetIO 拉精确向量     ─────►  RDMA read
                                                  ◄────  fixed record
                         精确 L2 + delta 合并
                         top-k
    ◄──────────────  完成 ring
  组装结果 / 归还 slot
```

### 8.3 30 课如何对应模块（导览图）

```
Part I  入门与基础设施（第 1–4 课）
  1 本课               仓库总览 / CMake / 目录边界
  2 公共类型与配置      src/common/{types,configuration,constants}.hh
  3 并发原语与协程      src/coroutine.hh, src/common/{atomic_utils,bounded_queue,
                        completion_pool,sliding_completion_ring}.hh
  4 RDMA 传输库(上)     rdma-library/library/{context,queue_pair,memory_region}.hh

Part II 传输层与索引格式（第 5–8 课）
  5 RDMA 传输库(下)     rdma-library/library/{connection_manager,batched_read,
                        dynamic_region_allocator,hugepage}.hh
  6 Vamana 图格式/anchor/idmap  src/vamana/{anchor_index,idmap,vamana_node}.hh
  7 schema-15 索引格式  src/remote_pointer.hh, src/gpu_search/index_format.hh
  8 元数据/owner map/存储协议  src/service/{index_metadata,base_owner_map,
                        storage_owner_protocol}.hh

Part III GPU 搜索引擎（第 9–16 课）
  9 GPU 类型/遥测/PQ 模型  src/gpu_search/{types,pq_index,memory_budget}.hh
  10 delta/动态路由/预算   src/gpu_search/{delta_index,dynamic_route_overlay,
                        delta_scan_budget,initial_seed_budget}.hh
                        src/vamana/adaptive_route_table.hh
  11 持久化引擎 PImpl/生命周期  src/gpu_search/{persistent_engine,persistent_engine/
                        {lifecycle,health,impl}}.hh
  12 construction(上)   src/gpu_search/persistent_engine/construction.cc
  13 construction(下)   同上
  14 查询执行/路由/完成  src/gpu_search/persistent_engine/{query_execution,routing,
                        completion}.cc
  15 增量发布           src/gpu_search/persistent_engine/delta_publication.cc
  16 存储回收 RCU       src/gpu_search/persistent_engine/storage_reclaim.cc

Part IV 持久化 CUDA Kernel（第 17–21 课）
  17 kernel 启动器/上下文/device ring  src/gpu_search/persistent_kernel.cu,
                        persistent_kernel/{context,runtime}.cuh,
                        src/gpu_search/{mapped_ring,device_ring}.cuh
  18 候选评分           persistent_kernel/candidate_scoring.cuh
  19 RDMA cache         persistent_kernel/rdma_cache.cuh
  20 查询遍历主循环     persistent_kernel/query_traversal.cuh
  21 kernel 运行时/角色调度  persistent_kernel/runtime.cuh

Part V DOCA GPUNetIO（第 22 课）
  22 GPUNetIO 传输/probe  src/gpu/{gpunetio_transport,gpunetio_probe}.{cc,cu}

Part VI 存储节点（第 23–26 课）
  23 存储节点主体/peer RDMA  src/memory_node/{memory_node,peer_rdma}.cc
  24 peer RPC           src/memory_node/peer_rpc/*
  25 索引访问/图修改    src/memory_node/storage_owner_index/*
  26 维护/wire protocol src/memory_node/{storage_owner_maintenance,storage_owner_runtime}/*

Part VII 计算服务（第 27–28 课）
  27 计算服务主体       src/service/compute_service/{lifecycle,search,index_commands}.cc,
                        src/main.cc
  28 计算侧 storage owner 更新  src/service/compute_service/storage_owner/*

Part VIII 离线工具与实验（第 29–30 课）
  29 离线构建/迁移      tools/{vamana_offline,legacy_index}/*,
                        tools/vamana_*_builder.cc, vamana_pq_indexer.cc
  30 breakdown benchmark/实验脚本  tools/breakdown_benchmark/*,
                        tools/dvstor_breakdown_benchmark.cc, experiment/*
```

## 9. 与其他模块的关系

本课是"地图"，不深入任何模块。后续课程对本课的依赖关系：

- **第 2 课**（公共类型与配置）会展开 `src/common/` 15 个文件，本课只点出名。
- **第 3 课**（并发原语与协程）讲 `src/coroutine.hh` 与 `src/common/` 的 ring/pool，
  本课的 6 步路径第 1 步"领 slot / 发 ring descriptor"在那里落地。
- **第 4–5 课**（RDMA 传输库）讲 `rdma-library/`，本课 4.4 节提到的
  `rdma_library` 静态库依赖在那里展开。
- **第 6–8 课**（索引格式与协议）讲 `src/vamana/`、`src/remote_pointer.hh`、
  `src/service/index_metadata.hh` 等，本课 2.3 节的文件契约表是入口。
- **第 9–16 课**（GPU 搜索引擎）讲 `src/gpu_search/`，本课 2.1 节的 6 步路径与
  `docs/gpu_persistent_engine.md` 是纲领。
- **第 17–21 课**（持久化 CUDA kernel）讲 `persistent_kernel*`，本课 4.6 节的
  `dvstor_gpu_kernels` 库与 `docs/source_layout.md:27-28` 的"单一 translation unit"
  是根据。
- **第 22 课**（GPUNetIO）讲 `src/gpu/`，本课 4.5 节 DOCA 探测与 4.8 节 probe 目标
  是构建侧入口。
- **第 23–26 课**（存储节点）讲 `src/memory_node/`，本课 4.4 节 storage-only 构建与
  `DVSTOR_MEMORY_NODE_SOURCES` 列表是边界。
- **第 27–28 课**（计算服务）讲 `src/service/compute_service/`，本课 4.7 节
  `DVSTOR_RUNTIME_SOURCES` 列表是入口。
- **第 29–30 课**（离线工具与实验）讲 `tools/` 与 `experiment/`，本课 4.3 节
  `dvstor_add_offline_tools` 函数与第 7 节运行流程是入口。

## 10. 小结

本课做了四件事：

1. **定位**：dvstor 是存算分离、GPU 中心化、DOCA GPUNetIO 直读、无 CPU fallback 的
   动态向量检索系统。`dev` 分支只保留一条查询路径，CPU 仅做准入/启动/控制面/更新 RPC。
2. **目录**：`src/gpu_search`（查询引擎 + 持久化 kernel）、`src/gpu`（GPUNetIO）、
   `src/memory_node`（存储主体 + 4 个子目录）、`src/service`（计算服务 + 协议）、
   `src/vamana`（图格式）、`src/common`（原语）、`tools`（离线 + benchmark）、
   `rdma-library`（自研 RDMA）、`experiment`（SIFT100M profile）。约 38k 行 C++ + 4k 行
   RDMA 库。
3. **构建**：`CMakeLists.txt` 用 `DVSTOR_STORAGE_NODE_ONLY` 分裂出无 CUDA/DOCA 的
   存储节点构建；`DVSTOR_BUILD_OFFLINE_TOOLS` 控制离线工具；`DVSTOR_METIS_PARTITION`
   三态管理 METIS 链接。计算节点构建 6 类目标：`dvstor_compute_node`、
   `dvstor_memory_node`、`dvstor_breakdown_benchmark`、`dvstor_gpunetio_probe`、
   `vamana_*` 离线工具、`dvstor_gpu_kernels`/`dvstor_runtime` 静态库。C++20 + CUDA 17
   + GCC 11+；显式源码列表，禁 GLOB。
4. **约束**：`.clangd` 让 clangd 能解析 CUDA；`docs/source_layout.md` 的 5 条变更约束
   是硬规则；`docs/gpu_persistent_engine.md` 是引擎宪法；`experiment/README.md` 给出
   SIFT100M 唯一支持的 profile 与运行流程。

下一课（第 2 课）进入 `src/common/` 的公共类型与配置头文件，开始真正的逐文件源码
阅读。读者在进入第 2 课前应能默写本课的 6 步查询路径与 9 个目录边界。
