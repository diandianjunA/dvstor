# SPACEV100M

使用 100,000,000 × 100 int8 向量，L2、PQ20（每个子空间 5 维）。公共实验流程见
[实验说明](../README.md)，构图默认 R96、beam 128、METIS、5 shards。
配置入口为 `spacev100m_common.sh`。

## 数据

默认数据目录为 `/data/xjs/datasets/spacev1b/spacev100m`，工作目录为
`/data/xjs/index/dvstor_spacev100m`。

| 文件 | 内容 |
|---|---|
| `spacev100m_base.i8bin` | 100M × 100 基础向量 |
| `query.i8bin` | 29,316 × 100 官方查询 |
| `msspacev-gt-100M` | 29,316 × top-100 ground truth |
| `spacev100m_to_110m_query.i8bin` | 性能查询，原始 base 的 `[100M,110M)` |
| `spacev110m_to_120m_insert.i8bin` | 插入，原始 base 的 `[110M,120M)` |

后两个文件需事先准备，也可通过 `PERFORMANCE_QUERY_FILE` 和 `INSERT_FILE` 指定。
准备脚本校验原始 base/query/GT 的 header 与大小，将官方 query 的 `[0,10000)` 和对应
GT 写入 `$WORK_DIR/prepared/`：

```bash
./experiment/spacev100m/prepare_spacev100m_data.sh
```

脚本还生成其余官方 query 的小型 `performance_query.i8bin` 与 `insert.i8bin` 切片；主实验
默认仍读取上表的两段 10M 文件，不会自动使用这些小切片。base 不复制。

## 构建与运行

```bash
BUILD_DIR="$PWD/build-storage" BUILD_ROLE=storage ./experiment/spacev100m/build.sh
BUILD_DIR="$PWD/build-compute" BUILD_ROLE=compute BUILD_INDEX=0 \
  ./experiment/spacev100m/build.sh
```

`build.sh` 会编译、准备派生输入，并按角色决定是否构建索引。已有二进制时可单独运行
`build_spacev100m_index.sh`，支持 `VALIDATE_ONLY=1` 和断点续建。最终前缀默认为
`$WORK_DIR/index/spacev100m_R96_bw128_metis_pmd32_pq20_schema16`，可用 `PQ_INDEX_PREFIX` 覆盖。

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/spacev100m/run_motivation.sh all
```

也可将 `all` 换成 `program1`、`program2`、`program3`，或直接调用各 program 的
`run_programN.sh`。参数、存储启动与结果复用见 [共享流程](../README.md#运行与恢复)。

## 已有 balanced 索引转 METIS

先检查 `repartition_to_metis.env` 的源/目标前缀，再执行：

```bash
./experiment/spacev100m/repartition_to_metis.sh
```

转换保留逻辑图和向量 ID，重建物理 shard、idmap、centroid、PQ codes 与 extent，校验后
复用 OPQ/PQ 模型。中断后重跑会复用已提交阶段，默认保留源索引。只有明确需要在成功后
删除旧版本时才设置 `DELETE_BALANCED_AFTER_SUCCESS=1`；删除范围限于旧前缀索引产物。
两个 100M METIS 转换应串行运行，避免 SPACEV 与 Deep 同时占用峰值内存。
