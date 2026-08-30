# Deep100M Motivation 实验

本目录是独立的 Deep100M 实验实现，不 source、exec 或引用其他数据集目录中的脚本。
数据集默认位于 `/data/xjs/datasets/deep1b/deep100m`。

## 数据与索引契约

| 项目 | 配置 |
|---|---|
| Base | `100M.fbin`，100,000,000 × 96 float32 |
| Query | `queries.fbin`，10,000 × 96 float32 |
| Ground truth | `100M_gt.bin`，10,000 × top-100 |
| metric | L2 |
| 图 / 分片 | R=96，build beam=128，METIS，5 shards |
| OPQ/PQ | PQ32（每个子空间3维） |
| 最终索引 | schema-16 tagged graph + centroid/idmap/PQ/extent sidecars |

`queries.fbin` 的 `[0,3334)` 默认用于 recall query，并同步切出对应 GT。性能查询和插入默认使用从官方 DEEP1B base 提取、且未进入 100M 索引的两段：

- `[100000000,110000000)`：`deep100m_to_110m_query.fbin`；
- `[110000000,120000000)`：`deep110m_to_120m_insert.fbin`。

recall 派生文件默认写入 `/data/xjs/index/dvstor_deep100m/prepared`，38.4 GB base 不复制。

## 配置

集中修改 `deep100m_common.sh`，或使用同名环境变量覆盖：

- `HOSTS`、`BASE_PORT`、`IB_DEVICE`、`IB_PORT`、`GPU_DEVICE`；
- `SHARDS`、`PARTITION_STRATEGY`、`R`、`BUILD_BEAM`；
- `WORK_DIR` 或 `PQ_INDEX_PREFIX`。

默认索引前缀为：

```text
/data/xjs/index/dvstor_deep100m/index/deep100m_R96_bw128_metis_pmd32_pq32_schema16
```

构建并发默认使用16、最多32：`BUILD_JOBS` 控制 CMake，`BUILD_THREADS` 控制构图，
`PQ_THREADS` 控制 OPQ/PQ；超过32会直接拒绝。

## 数据准备与命令验证

```bash
./experiment/deep100m/prepare_deep100m_data.sh
VALIDATE_ONLY=1 ./experiment/deep100m/build_deep100m_index.sh
```

`VALIDATE_ONLY=1` 只验证文件、profile、目标和最终命令，不启动100M构图。

## 存储/离线索引节点构建

`build.sh` 支持 `BUILD_ROLE=auto|storage|offline|compute|all`。已有 CMake cache 若为
`DVSTOR_STORAGE_NODE_ONLY=ON`，`auto` 会只请求存储和离线工具目标。

```bash
nohup env \
  BUILD_ROLE=storage BUILD_JOBS=16 BUILD_THREADS=16 PQ_THREADS=16 \
  ./experiment/deep100m/build.sh \
  > experiment/deep100m/build.log 2>&1 &
```

只构建索引：

```bash
BUILD_THREADS=16 PQ_THREADS=16 \
  ./experiment/deep100m/build_deep100m_index.sh
```

脚本会自动识别 schema-15 graph、schema-16 PQ 和 extent 三个阶段。PQ 或 extent 失败后直接重跑同一命令，只续失败阶段，已完成 graph 保持不变。完整索引会直接校验后退出。

`OVERWRITE_INDEX=1` 不再删除 graph；完整重建使用 `REBUILD_GRAPH=1`，只重做后处理使用 `REBUILD_PQ=1` 或 `REBUILD_EXTENT=1`。正式构建前会运行回归测试以及数据、分片参数、磁盘和内存预检。

## 将已有 balanced 索引转换为 METIS

本目录的 `repartition_to_metis.env` 和 `repartition_to_metis.sh` 是 Deep100M
独立实现，不引用 SPACEV 或其他数据集目录。转换保留逻辑图，但物理分片发生变化，所以
shard、idmap、物理分片 centroid、PQ codes 和 graph extent 都会重建；OPQ/PQ 模型经
checksum 校验后复用。

不要与 SPACEV100M 的 100M 转换并行。完整转换、校验并删除 balanced 版本：

```bash
DELETE_BALANCED_AFTER_SUCCESS=1 \
  ./experiment/deep100m/repartition_to_metis.sh
```

删除是 opt-in，且只在新 schema-16 METIS 索引通过全部校验后发生。默认先保留 balanced；
稍后用上面的环境变量重跑时，转换器只校验已完成目标，然后精确删除旧前缀的索引文件，
不会删除 `100M.fbin`。中断后可直接重跑，已提交的 graph/PQ 阶段会复用。

## 计算节点构建

storage-only cache 没有 compute targets，计算侧应使用独立目录：

```bash
cmake -S . -B build-compute -DCMAKE_BUILD_TYPE=Release \
  -DDVSTOR_STORAGE_NODE_ONLY=OFF
BUILD_DIR="$PWD/build-compute" BUILD_ROLE=compute BUILD_INDEX=0 \
  ./experiment/deep100m/build.sh
```

## Motivation 运行

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/program1/run_program1.sh
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/program2/run_program2.sh
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/program3/run_program3.sh
```

统一入口：

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/run_motivation.sh program1
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/run_motivation.sh program2
BUILD_DIR="$PWD/build-compute" ./experiment/deep100m/run_motivation.sh program3
```

每个 case 会打印存储节点需要运行的本目录命令。节点生命周期入口为
`start_memory_node.sh`、`start_all_memory_nodes.sh` 和 `stop_memory_nodes.sh`；结果、日志、
PID 分别留在 Deep100M 自己的目录中。
