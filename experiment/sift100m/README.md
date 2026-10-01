# SIFT100M

使用 100,000,000 × 128 uint8 向量，L2、PQ32。公共实验流程见
[实验说明](../README.md)，构图默认 R96、beam 128、METIS、5 shards。
配置入口为 `sift100m_common.sh`，可通过同名环境变量覆盖机器和数据路径。

## 数据前提

默认原始数据目录为 `/data/xjs/datasets/sift1b`，工作目录为
`/data/xjs/index/dvstor_sift100m`。运行需要以下已经转换好的文件：

| 位置 | 文件 | 用途 |
|---|---|---|
| `$WORK_DIR/converted/` | `base_100000000.u8bin` | 构图使用的 100M 基础向量 |
| `$WORK_DIR/converted/` | `query.u8bin` | 10,000 个 recall query |
| `$WORK_DIR/converted/` | `groundtruth_100M.bin` | 对应 top-10 ground truth |
| `$DATASET_DIR/` | `sift100m_to_110m_query.u8bin` | 性能查询，原始 base 的 `[100M,110M)` |
| `$DATASET_DIR/` | `sift110m_to_120m_insert.u8bin` | 插入，原始 base 的 `[110M,120M)` |

`CONVERTED_DIR`、`PERFORMANCE_QUERY_FILE`、`INSERT_FILE` 可覆盖相应位置。仓库未包含
`prepare_sift100m_data.sh` 所依赖的两个转换工具，因此这里以现成输入为前提。
`run_breakdown.sh` 默认不生成数据；保持 `PREPARE_QUERY`、`PREPARE_GROUNDTRUTH` 和
`PREPARE_BENCHMARK_DATA` 为默认值 0。

## 构建与运行

```bash
BUILD_DIR="$PWD/build-storage" VALIDATE_ONLY=1 \
  ./experiment/sift100m/build_sift100m_index.sh
BUILD_DIR="$PWD/build-storage" ./experiment/sift100m/build_sift100m_index.sh
```

最终前缀默认为
`$WORK_DIR/index/sift100m_R96_bw128_metis_pmd32_pq32_schema16`，可用 `PQ_INDEX_PREFIX` 覆盖。
三阶段续跑和重建开关见 [索引构建](../README.md#配置与索引构建)。

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program1/run_program1.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program2/run_program2.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/program3/run_program3.sh
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/ablation/run_ablation.sh
```

每个 case 按计算脚本打印的命令重启存储节点。[Program 1–3](../README.md#运行与恢复)
的参数与结果说明共用；[累积消融](ablation/README.md) 有单独的模式矩阵和运行参数。
