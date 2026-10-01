# 实验

SIFT100M、Deep100M 和 SPACEV100M 均提供 Program 1–3。数据与机器配置保留在各数据集目录，
通用绘图、Program 2 汇总和存储启动实现在 `common/`，原数据集入口通过相对符号链接调用。
从数据集路径运行脚本，以便正确找到其 profile、日志和 PID 目录。

| 数据集 | 配置与数据说明 | 向量 / PQ | 额外实验 |
|---|---|---|---|
| SIFT100M | [sift100m](sift100m/README.md) | 128 维 uint8 / PQ32 | [累积消融](sift100m/ablation/README.md) |
| Deep100M | [deep100m](deep100m/README.md) | 96 维 float32 / PQ32 | — |
| SPACEV100M | [spacev100m](spacev100m/README.md) | 100 维 int8 / PQ20 | — |

跨数据集查询/插入混合压力测试见 [mixed_test](mixed_test/README.md)。下文命令从仓库根目录运行，
以 SIFT100M 为例；Program 1–3 可替换为另外两个数据集路径。

## 配置与索引构建

先按 [根 README](../README.md#构建) 编译计算节点和存储节点，再配置数据集的
`*_common.sh` 或用环境变量覆盖。常用项：

| 配置 | 用途 |
|---|---|
| `DATASET_DIR`、`WORK_DIR` | 原始数据与索引/派生数据位置 |
| `PQ_INDEX_PREFIX` | 最终 schema-16 索引前缀 |
| `BUILD_DIR`、`STORAGE_BUILD_DIR` | 当前脚本与存储 case 启动脚本的构建目录 |
| `HOSTS`、`BASE_PORT`、`SHARDS` | 逻辑分片、地址和端口 |
| `IB_DEVICE`、`IB_PORT`、`GPU_DEVICE` | RDMA 与 GPU 设备 |
| `R`、`BUILD_BEAM`、`PARTITION_STRATEGY` | 构图与物理分片参数 |

三组数据默认均为 100M、L2、R96、build beam 128、5 shards、METIS。计算和存储两侧必须
使用匹配的索引前缀、分片与 profile。`profiles/04_gpu_persistent_gpunetio_common.sh`
保存 baseline/full 共用的质量、容量和硬件参数；两个 `.env` 选择贡献模式。

数据准备完成后，可先检查构建输入与计划：

```bash
BUILD_DIR="$PWD/build-storage" VALIDATE_ONLY=1 \
  ./experiment/sift100m/build_sift100m_index.sh
```

该模式需要已有离线工具，只做预检、索引状态校验与命令打印。正式构建使用同一入口，去掉
`VALIDATE_ONLY=1`。三个构建脚本均按 graph → PQ → extent 提交：

1. graph 阶段生成 schema-15 中间图和物理分片 centroid；
2. PQ 阶段训练/编码，生成绑定构建的模型、码流、owner idmap 和 schema-16 metadata；
3. extent 阶段全量校验图并生成 `.gextent8`。

中断后重跑会复用已提交阶段；完整索引只校验后退出。`REBUILD_GRAPH=1` 才会从头重建，
`REBUILD_PQ=1`、`REBUILD_EXTENT=1` 分别重做后处理；`OVERWRITE_INDEX=1` 不会删除有效图。
构建并发 `BUILD_JOBS`、`BUILD_THREADS`、`PQ_THREADS` 默认 16，最大 32。

Deep/SPACEV 还提供 `build.sh`，支持 `BUILD_ROLE=storage|offline|compute|all|auto`，默认
`storage`；`auto` 根据已有 CMake cache 选目标。`BUILD_INDEX=0` 可跳过索引构建。
已有 storage-only cache 不能用于 compute 目标。若使用 Conda，避免混合其头文件与系统
不同版本的 Boost/TBB 库；可用 CMake 的包路径参数指定一致的一套依赖。

## 运行与恢复

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program1/run_program1.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program2/run_program2.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program3/run_program3.sh
```

每个 case 前计算脚本会暂停，并打印准确的存储节点命令。在存储机器运行提示命令，看到
`storage ready` 后继续。独立构建目录示例：

```bash
STORAGE_BUILD_DIR="$PWD/build-storage" \
  ./experiment/sift100m/program2/start_storage_case.sh fixed
```

每个 case 都应重启存储进程，从同一静态索引开始，避免上个 case 的更新影响下一组。
节点启动脚本不会远程 SSH 到 `HOSTS`；分布部署时，在对应机器使用数据集的
`start_memory_node.sh <node_id> [profile]`，并确保所需逻辑分片全部就绪。

结果默认在各 program 的 `results/programN_时间戳/`，包含原始报告、日志、manifest、汇总
和 SVG。新生成的结果、日志与 PID 默认只保存在本地，已有参考结果保留在仓库。
同一次实验复用 `RUN_ROOT` 可续跑或重新汇总；输入、profile、并发和测量参数应保持一致：

```bash
RUN_ROOT=/path/to/program2_TIMESTAMP ./experiment/sift100m/program2/run_program2.sh live
RUN_ROOT=/path/to/program2_TIMESTAMP ./experiment/sift100m/program2/run_program2.sh summarize
```

## Program 1：两阶段更新

依次运行 `baseline`、`solution` 和 `quality`：前两者比较同步完成与 Stage1 ACK 的更新
开销，质量阶段测量新节点的短期可见性和 Stage2 收敛。两组更新实验均显式使用 Persistent
查询栈，只切换更新完成模式；这里的 baseline 不等于直接运行正式 baseline profile。

默认 16 个客户端线程、128 次预热操作、1000 次测量操作。质量阶段默认插入 1000 个向量，
对 1000 个查询检查 self-recall，Stage2 延迟 15 秒、等待收敛 30 秒。对应覆盖项为
`CLIENT_THREADS`、`WARMUP_OPS`、`MEASURE_OPS`、`QUALITY_INSERT_COUNT`、
`QUALITY_RECALL_QUERIES`、`QUALITY_STAGE2_DELAY_MS` 和 `QUALITY_SETTLE_SECONDS`。

```bash
RUN_ROOT=/path/to/program1_TIMESTAMP ./experiment/sift100m/program1/run_program1.sh quality
RUN_ROOT=/path/to/program1_TIMESTAMP ./experiment/sift100m/program1/run_program1.sh summarize
```

`baseline` 和 `solution` 也可单独补跑。质量数据来自专用验证程序，与常规静态 Recall 分开解读。

## Program 2：动态长度读取

在相同 mixed workload 下比较三种实际策略：Fixed 完整读、Header→Neighbor 串行两读、
ClassExtent 按长度档一次读。R96 图记录为 832 B，header 为 16 B。三组使用相同动态 extent
发布方式、查询/插入文件和固定写入压力，其余查询线程闭环饱和。

默认预热 5 秒、测量 20 秒、16 个写线程、500 insert/s。汇总要求写入达成率至少 95%，
实际展开至少 100 个动态版本，且动态展开占比至少 0.1%。可通过 `TARGET_WRITE_QPS`、
`WRITE_THREADS`、`MIN_WRITE_ATTAINMENT`、`MIN_DYNAMIC_EXPANDED` 和 `MIN_DYNAMIC_SHARE`
调整。小规模连通性检查可缩短时间并显式降低动态样本门槛：

```bash
WARMUP_SECONDS=2 MEASURE_SECONDS=5 RECALL_QUERIES=100 \
MIN_DYNAMIC_EXPANDED=10 MIN_DYNAMIC_SHARE=0 \
  ./experiment/sift100m/program2/run_program2.sh
```

Oracle 由已经校验并权威展开的动态节点离线计算：

```text
Oracle bytes = 16 × dynamic_parents + 8 × live_neighbor_sum
Oracle WQEs  = dynamic_parents
```

这是提前知道精确长度的理论下界，不参与请求准备，也没有端到端 QPS。可选 `probe` 仅检查
单次精确长度 RDMA 协议。结果包含 `summary.json/csv`、动态 degree histogram、动机图和效果图。
详细校验与统计口径见 [Live-Extent RDMA](../docs/live_extent_rdma.md)。

## Program 3：提前发起精确前沿读取

先扫描 `C=1/4/8/16` 的候选和 Merge 开销，再在 C16 比较 Late/Early，共 6 个 case。
默认每组预热 5 秒，动机测量 10 秒，效果测量 20 秒，每个 case 一次。动机阶段只查询；
效果阶段默认只查询，可用 `PERF_WORKLOAD=mixed` 加入固定写入压力。

两组均使用 Persistent、Stable-Run 和 Live/DynaExtent，并固定 issue width 等于 commit
width，关闭 speculative tail。差别仅为 critical reads 在完整 beam 发布后发出，还是在
exact frontier certificate 就绪后提前发出。因此使用 progression=`manual`，不能替换成
会选择 HostOrchestrated 的 `coupled`。

```bash
MOTIVATION_SECONDS=3 PERFORMANCE_SECONDS=5 RECALL_QUERIES=100 \
  ./experiment/sift100m/program3/run_program3.sh
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/sift100m/program3/run_program3.sh motivation
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/sift100m/program3/run_program3.sh performance
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/sift100m/program3/run_program3.sh summarize
```

存储启动命令会带 `GPU_COMMIT_WIDTH`，需与当前 case 一致。结果包含汇总 JSON、宽度扫描 CSV、
动机图、查询/候选维护分解图、效果图及 Markdown 报告。汇总会检查 certificate 是否实际发布，
以及 tail 读取是否为零。
