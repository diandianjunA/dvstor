# SPACEV100M Motivation 实验

本目录接入 `/data/xjs/datasets/spacev1b/spacev100m`，提供当前系统三项
Motivation 实验所需的数据准备、schema-16 索引构建、运行 profile 和节点启动脚本。

已按数据文件头部固定以下契约：

| 项目 | 配置 |
|---|---|
| Base | `spacev100m_base.i8bin`，100,000,000 × 100 |
| dtype / metric | `int8` / L2 |
| Query | `query.i8bin`，29,316 × 100 |
| Ground truth | `msspacev-gt-100M`，29,316 × top-100 |
| 图 / 分片 | R=96，build beam=128，balanced，5 shards |
| OPQ/PQ | PQ20（100 维可整除，每子空间 5 维） |

## 1. 修改机器配置

集中修改 `spacev100m_common.sh` 中的 `HOSTS`、`BASE_PORT`、`GPU_DEVICE`、
`IB_DEVICE`，或用同名环境变量覆盖。默认使用 5 个逻辑分片和当前系统硬件资源
参数。索引默认写到：

```text
/data/xjs/index/dvstor_spacev100m/index/
```

可用 `WORK_DIR` 或 `PQ_INDEX_PREFIX` 改到其他磁盘。计算节点与存储节点必须使用相同
的 profile、索引前缀和分片数。

## 2. 准备 benchmark 输入

原始 query 的 `[0,10000)` 默认用于 recall query，并同步切出对应 top-100 GT。性能查询和插入默认使用从官方 SPACEV1B base 提取、且未进入 100M 索引的两段：

- `[100000000,110000000)`：`spacev100m_to_110m_query.i8bin`；
- `[110000000,120000000)`：`spacev110m_to_120m_insert.i8bin`。

```bash
./experiment/spacev100m/prepare_spacev100m_data.sh
```

输出默认位于 `/data/xjs/index/dvstor_spacev100m/prepared/`。脚本严格校验三个源文件
的 header 与精确大小，重复运行会原子替换小型派生文件。

## 3. 构建

所有构建并发默认使用 16、最多为 32：`BUILD_JOBS=16` 控制 CMake，`BUILD_THREADS=16`
控制离线构图，`PQ_THREADS=16` 控制 OPQ/PQ 训练与编码。可以调低；设置为大于 32
会直接报错。运行期的 `SERVICE_THREADS` 不属于构建线程。

`build.sh` 支持 `BUILD_ROLE=auto|storage|offline|compute|all`。`auto` 会读取现有 CMake cache；
若 `DVSTOR_STORAGE_NODE_ONLY=ON`，只构建存储服务和离线索引工具，不会请求 compute target。

存储/离线索引节点（当前 `build` 会自动识别为此角色）：

```bash
./experiment/spacev100m/build.sh
```

计算节点必须使用单独的非 storage-only 构建目录：

```bash
cmake -S . -B build-compute -DCMAKE_BUILD_TYPE=Release \
  -DDVSTOR_STORAGE_NODE_ONLY=OFF
BUILD_DIR="$PWD/build-compute" BUILD_ROLE=compute BUILD_INDEX=0 \
  ./experiment/spacev100m/build.sh
```

如果存储工程已经编译，只构建索引：

```bash
./experiment/spacev100m/build_spacev100m_index.sh 04_gpu_persistent_gpunetio
```

脚本会自动识别并续跑三个提交阶段：schema-15 graph、schema-16 PQ 和 extent。PQ 或 extent 失败后直接重跑同一命令，不会删除或重建已完成的 graph；完整索引会直接校验后退出。

`OVERWRITE_INDEX=1` 不再删除 graph。只有明确需要从零重建时才使用 `REBUILD_GRAPH=1`；仅重做 PQ 或 extent 可分别使用 `REBUILD_PQ=1`、`REBUILD_EXTENT=1`。每次正式构建前还会运行 LocalIdSet/beam-search 回归、数据头、分片参数、磁盘和内存预检。仅编译程序和准备输入、不构建昂贵索引时使用
`BUILD_INDEX=0 ./experiment/spacev100m/build.sh`。

## 4. 运行 Motivation

三项入口均为本目录内的独立实现：

```bash
./experiment/spacev100m/program1/run_program1.sh
./experiment/spacev100m/program2/run_program2.sh
./experiment/spacev100m/program3/run_program3.sh
```

也可以使用统一入口：

```bash
./experiment/spacev100m/run_motivation.sh program1
./experiment/spacev100m/run_motivation.sh program2
./experiment/spacev100m/run_motivation.sh program3
# 顺序运行三项（每个 case 仍会等待存储节点就绪）
./experiment/spacev100m/run_motivation.sh all
```

计算侧脚本会在每个 case 前打印对应的存储侧命令。例如方案三某个宽度应在存储节点运行：

```bash
GPU_COMMIT_WIDTH=16 ./experiment/spacev100m/program3/start_storage_case.sh early
```

可用各 program 的 `start_storage_case.sh status` / `stop` 查看或停止存储进程。结果分别
写入 `program1/results`、`program2/results`、`program3/results`。

底层单次 benchmark 入口也可直接使用：

```bash
WORKLOAD=query ./experiment/spacev100m/run_breakdown.sh 04_gpu_persistent_gpunetio
```

## 5. 常用覆盖项

```bash
# 更换构建目录或索引位置
BUILD_DIR=/path/to/build PQ_INDEX_PREFIX=/path/to/index_prefix \
  ./experiment/spacev100m/build_spacev100m_index.sh

# 调整节点地址和 RDMA 设备（计算/存储两侧保持一致）
HOSTS="host1 host2 host3 host4 host5" IB_DEVICE=mlx5_0 \
  ./experiment/spacev100m/program2/start_storage_case.sh live
```

正式对比应保持 baseline/full 的下层 GPU、RDMA 和 storage capacity 参数一致；两个
profile 只切换贡献级 mode。运行前脚本会校验 metadata 的 schema、dtype、维度、分片、
PQ 模型、extent sidecar 和各 storage artifact。
