# SIFT100M Experiment

实验目录只保留 `04_gpu_persistent_gpunetio`：持久化 GPU OPQ/PQ32 图导航、
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
export GPU_RESIDENT_PQ_BUDGET_MB=4096
```

旧索引只在迁移命令中通过 `SOURCE_PREFIX` 显式指定；运行 profile 只维护当前
`INDEX_PREFIX`，不携带旧索引状态。

## 构建新索引

存储节点脚本默认使用独立的 `build-storage`，先按根目录 README 配置 CPU-only
构建，不能与计算节点的 `build` 共用。

该命令先构建 compact Vamana/Metis 分片，再训练 OPQ/PQ32 并写码流：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

完整构建以 schema-14 compact 图为中间态，最终直接生成 schema-15
OPQ/PQ32 运行索引，无需随后重编码。
推荐使用 `PQ_INDEX_PREFIX=/new/prefix` 保留已有索引；只有明确要删除目标 prefix
下旧产物并原地重建时才设置 `OVERWRITE_INDEX=1`。

## 转换旧索引

已有 Vamana/Metis 索引无需重新构图或分区：

```bash
SOURCE_PREFIX=/path/to/legacy/index_prefix \
./experiment/convert_legacy_sift100m_index.sh 04_gpu_persistent_gpunetio
```

转换和 PQ 编码是两个独立进程。schema-14 metadata 是离线迁移检查点，PQ 阶段
完成后会原子写成 schema-15；如果迁移
已经完成而 OPQ/PQ 训练失败，原命令会跳过迁移并直接继续 PQ，不会重复扫描和
改写全部旧分片。`OVERWRITE_INDEX=1` 仅用于明确要求重新迁移；若 PQ 输出不完整，
使用 `OVERWRITE_PQ=1`。

可使用已训练模型减少迁移时间：

```bash
PQ_REUSE_MODEL=/path/to/compatible.pq32 \
./experiment/convert_legacy_sift100m_index.sh 04_gpu_persistent_gpunetio
```

模型必须与维度、子空间数和 dtype 对应。转换仍需顺序扫描全部向量生成 32-byte
code，但不执行昂贵的 Vamana construction 或 METIS partition。PQ 默认使用
`PQ_THREADS=32`，也可显式覆盖；BLAS 线程保持为 1，避免和 Faiss OpenMP 嵌套。

已有 schema-14 PQ16 索引时可直接复用 `.dat`、`.idmap` 和 anchors，仅生成
默认 PQ32 模型与码流：

```bash
./experiment/reencode_sift100m_pq.sh 04_gpu_persistent_gpunetio
```

已有 schema-14 OPQ/PQ32 sidecar 时可原地升级，不读取 `.dat` payload，也不重新
训练或编码。计算节点运行 metadata-only 模式；每个存储节点传自己的分片号：

```bash
INDEX_ROLE=compute ./experiment/upgrade_pq_schema15.sh 04_gpu_persistent_gpunetio
INDEX_ROLE=storage LOCAL_SHARD=1 \
  ./experiment/upgrade_pq_schema15.sh 04_gpu_persistent_gpunetio
```

## 部署文件

计算节点：

```text
<prefix>.meta.json
<prefix>.pq32
<prefix>.anchors
<prefix>_node1_ofN.idmap ... <prefix>_nodeN_ofN.idmap
```

存储节点 X：

```text
<prefix>.meta.json
<prefix>.anchors
<prefix>_nodeX_ofN.dat
<prefix>_nodeX_ofN.idmap
<prefix>_nodeX_ofN.pq32.codes
```

计算节点不需要 `.dat`、`.pq32.codes` 或 `.gpu.idx`，但更新路由必须能读取全部
owner-sharded `.idmap`。已有 schema-15 索引只需把各存储分片的 idmap sidecar
复制到计算节点的同一 prefix，无需重建索引。存储节点运行时不需要 `.pq32` 模型。

## 启动

在各存储节点准备对应文件后启动服务：

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
```

如果每个分片位于不同主机，可分别执行：

```bash
./experiment/start_memory_node.sh 1 04_gpu_persistent_gpunetio
```

启动脚本会验证 schema、分片数、R、dtype、PQ checksum 和角色所需文件，
不兼容时在申请大块注册内存前退出。

schema-15 存储控制区使用版本 2：每个计算节点拥有独立 reclaim ACK。升级后必须
重新编译并重启全部计算、存储节点；PQ code 无需重编码。当前 schema-15 的反向边
请求只携带物理指针、没有 generation，因此存储端不会复用已删除动态节点的物理
地址，避免迟到重试修改另一个节点。每次成功 insert/upsert 都会消耗新的节点/向量
空间，部署时必须为预期写入量预留 memory-node 容量；这项限制要等独立的协议升级
后才能解除。

新插入或 upsert 产生的 PQ code 在发布时由 GPU 编码一次，并进入独立的常驻
dynamic-PQ 层。短期 L0 中的原始向量和可变图记录退休后，该 PQ code 仍留在
GPU，查询导航不会退化为逐 code RDMA。只有对应版本被 upsert/delete 淘汰、旧查询
RCU 屏障退出后才回收常驻 PQ 槽。stage2 context、GPU delta 元数据和被淘汰的 PQ
都保持有界并可回收，但上述存储节点/真实向量地址在 schema-15 内保持 generation
稳定。PQ 容量由 `GPU_RESIDENT_PQ_BUDGET_MB` 显式限制，报告中的
`resident_pq_entries/peak/capacity/reclaimed` 用于观察长期运行水位。

## 召回率与性能

测试负载参数不放在索引/系统 profile 中。`BENCHMARK_CLIENT_THREADS`、
`WORKLOAD`、`READ_RATIO`、`WARMUP_SECONDS`、`MEASURE_SECONDS` 和
`RECALL_QUERIES` 由运行脚本读取。`SERVICE_THREADS` 是计算服务 CPU 线程数，
不等于 benchmark 客户端并发数。

`query.u8bin` 的 10K 标准查询仅供 recall 使用。性能阶段由
`PERFORMANCE_QUERY_FILE` 提供独立查询流，warmup 与 measure 共用一个单遍游标，
同一行不会再次执行；查询池耗尽时 benchmark 会失败而不是取模回绕。当前默认
性能查询池为 `[100M,105M)` 的 500 万行，插入向量池为 `[105M,107M)` 的
200 万行。生成文件默认位于
`/data/xjs/datasets/sift1b`：

```text
sift100m_to_105m_query.u8bin
sift105m_to_107m_insert.u8bin
```

`prepare_sift100m_data.sh` 会按需生成并校验这两个文件；已有且头部、大小正确时会
直接复用，此时计算节点不需要 `bigann_base.bvecs`。只有文件缺失或设置
`OVERWRITE_BENCHMARK_DATA=1` 时才需要完整源数据。可通过
`PERFORMANCE_QUERY_FILE`、`INSERT_FILE` 覆盖路径，或用以下变量
调整源区间：`PERFORMANCE_QUERY_START`、`PERFORMANCE_QUERY_END`、
`INSERT_VECTOR_START`、`INSERT_VECTOR_END`。例如：

```bash
PERFORMANCE_QUERY_FILE=/data/xjs/datasets/sift/perf_queries_2m.u8bin \
INSERT_FILE=/data/xjs/datasets/sift/inserts_2m.u8bin \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

先做 query-only 召回验证：

```bash
RECALL_QUERIES=1000 \
./experiment/run_recall.sh 04_gpu_persistent_gpunetio
```

该脚本使用 `--recall-only`，不会执行 warmup/measure，也不会加载性能查询池。

再运行读写混合负载：

```bash
BENCHMARK_CLIENT_THREADS=128 WORKLOAD=mixed READ_RATIO=0.5 \
WARMUP_SECONDS=30 MEASURE_SECONDS=120 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

两阶段更新验收使用固定 profile，且必须先把同一新二进制部署到全部五个存储节点，
并确认五个进程均已启动完成。
每轮独立 profile 前都要重启五个存储进程，以同一份 schema-15 基础文件恢复内存图；
这只是重新加载现有索引，不需要重建或升级索引。query-only 基线和 `mixed15m` 都
必须从这份干净基础状态启动，两者之间不得运行写负载。基线 JSON 会绑定索引、查询
文件、dtype、GPU 搜索参数和冷缓存设置；`mixed15m` 不接受手填的裸 QPS：

```bash
# 先重启全部五个存储进程，确认均已从同一份基础索引启动
UPDATE_ACCEPTANCE_PROFILE=insert24 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio

# 重启全部五个存储进程，恢复同一份基础索引

UPDATE_ACCEPTANCE_PROFILE=insert64 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio

# 再次重启全部五个存储进程；随后先跑基线且不要插入其他写负载
UPDATE_ACCEPTANCE_PROFILE=querybaseline \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio

# 基线退出后再次重启五个存储进程，从同一份基础索引启动 mixed15m
QUERY_BASELINE_REPORT=/path/to/querybaseline.json \
STORAGE_MAINTENANCE_LOGS="/path/node1.log /path/node2.log /path/node3.log /path/node4.log /path/node5.log" \
UPDATE_ACCEPTANCE_PROFILE=mixed15m \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

三个更新 profile 会 fail-closed 检查各自的吞吐和零完成窗口；15 分钟混合负载还会
检查 5K query/s、1K insert/s、query-only 基线的 90%、base-only recall、
stage2 p99/积压/排空、10ms GPU 可见性、容量拒绝、最终 delta 和迟到 RPC。

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
