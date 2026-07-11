# SIFT100M Experiment

实验目录只保留 `04_gpu_persistent_gpunetio`：持久化 GPU OPQ/PQ16 图导航、
GPUNetIO 远端读取和 storage-owner 动态更新。旧 profile、旧 sidecar 转换器和
历史输出不属于 `dev` 的运行接口。

## 配置

默认路径定义在 `sift100m_common.sh`，架构参数定义在
`profiles/04_gpu_persistent_gpunetio.env`。常用覆盖项：

```bash
export HOSTS="192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202 192.168.6.202"
export INDEX_DIR=/data/xjs/index/dvstor_sift100m/index
export GPU_DEVICE=1
export GPU_MEMORY_LIMIT_GB=40
export GPU_MEMORY_RESERVE_GB=4
```

profile 的 `LEGACY_INDEX_PREFIX` 指向旧索引，`INDEX_PREFIX` 指向新的 `_pq16`
输出。二者不能相同。

## 构建新索引

存储节点脚本默认使用独立的 `build-storage`，先按根目录 README 配置 CPU-only
构建，不能与计算节点的 `build` 共用。

该命令先构建 compact Vamana/Metis 分片，再训练 OPQ/PQ16 并写码流：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

## 转换旧索引

已有 Vamana/Metis 索引无需重新构图或分区：

```bash
./experiment/convert_legacy_sift100m_index.sh 04_gpu_persistent_gpunetio
```

转换和 PQ 编码是两个独立进程。最终 schema-14 metadata 是迁移检查点；如果迁移
已经完成而 OPQ/PQ 训练失败，原命令会跳过迁移并直接继续 PQ，不会重复扫描和
改写全部旧分片。`OVERWRITE_INDEX=1` 仅用于明确要求重新迁移；若 PQ 输出不完整，
使用 `OVERWRITE_PQ=1`。

可使用已训练模型减少迁移时间：

```bash
PQ_REUSE_MODEL=/path/to/compatible.pq16 \
./experiment/convert_legacy_sift100m_index.sh 04_gpu_persistent_gpunetio
```

模型必须与维度、子空间数和 dtype 对应。转换仍需顺序扫描全部向量生成 16-byte
code，但不执行昂贵的 Vamana construction 或 METIS partition。PQ 默认使用
`PQ_THREADS=32`，也可显式覆盖；BLAS 线程保持为 1，避免和 Faiss OpenMP 嵌套。

## 部署文件

计算节点：

```text
<prefix>.meta.json
<prefix>.pq16
<prefix>.anchors
```

存储节点 X：

```text
<prefix>.meta.json
<prefix>.anchors
<prefix>_nodeX_ofN.dat
<prefix>_nodeX_ofN.idmap
<prefix>_nodeX_ofN.pq16.codes
```

计算节点不需要 `.dat`、`.idmap`、`.pq16.codes` 或 `.gpu.idx`。存储节点运行时
不需要 `.pq16` 模型。

## 启动

在各存储节点准备对应文件后启动服务：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
```

如果每个分片位于不同主机，可分别执行：

```bash
./experiment/start_memory_node.sh 1 04_gpu_persistent_gpunetio
```

启动脚本会验证 schema、分片数、R、dtype、PQ16 checksum 和角色所需文件，
不兼容时在申请大块注册内存前退出。

## 召回率与性能

先做 query-only 召回验证：

```bash
RECALL_QUERIES=1000 \
./experiment/run_recall.sh 04_gpu_persistent_gpunetio
```

再运行读写混合负载：

```bash
WORKLOAD=mixed READ_RATIO=0.5 \
WARMUP_SECONDS=30 MEASURE_SECONDS=120 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

短跑示例：

```bash
WORKLOAD=query RECALL_QUERIES=100 \
WARMUP_SECONDS=1 MEASURE_SECONDS=5 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

结果写入 `experiment/reports/04_gpu_persistent_gpunetio/`。有效结果应同时满足：

- `gpu_persistent.direct_path_failures == 0`；
- recall 达到论文设定阈值；
- 没有 unhealthy/fail-stop 日志；
- GPU 和 RDMA 指标显示多查询并发，而非单查询串行等待。

可与 OdinANN 或历史 JSON 比较：

```bash
python3 experiment/compare_reports.py \
  --baseline /path/to/odinann.json \
  --candidate experiment/reports/04_gpu_persistent_gpunetio/latest.json \
  --min-query-speedup 1.0 \
  --max-recall-loss 0.01
```

停止本机启动的存储进程：

```bash
./experiment/stop_memory_nodes.sh
```
