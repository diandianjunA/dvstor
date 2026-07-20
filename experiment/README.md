# SIFT100M Experiment

实验目录只保留 `04_gpu_persistent_gpunetio`：持久化 GPU OPQ/PQ32 图导航、
GPUNetIO 远端读取和 storage-owner 动态更新。

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

## 部署文件

计算节点（查询与更新）：

```text
<prefix>.meta.json
<prefix>.pq32
<prefix>.anchors
<prefix>_node1_ofN.idmap ... <prefix>_nodeN_ofN.idmap
```

纯查询配置使用 `enable-updates = false`，需要 `<prefix>.meta.json`、
`<prefix>.pq32` 和 `<prefix>.anchors`；不会加载 owner idmap，也不会启动更新执行器。
这里的 anchors 只作为 GPU 冷启动/召回兜底。在线 mutation 会持续更新 storage
owner 的固定容量动态入口；每个计算节点从 control page 拉取同一 canonical 快照，
因此其他计算节点写入的新代表节点同样可见。

存储节点 X：

```text
<prefix>.meta.json
<prefix>_nodeX_ofN.dat
<prefix>_nodeX_ofN.idmap
<prefix>_nodeX_ofN.pq32.codes
```

计算节点不需要 `.dat`、`.pq32.codes` 或 `.gpu.idx`。启用更新时，它必须能读取全部
owner-sharded `.idmap`：METIS 分片的 owner 不能由 `ID % N` 推导，基础 ID 的
upsert/delete 和重复写必须先找到真实 owner。每个存储节点仍只需自己的 idmap。
已有 schema-15 索引只需复制 sidecar，无需重建索引。存储节点运行时不需要
`.pq32` 模型或 `.anchors`。

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

stage2 finalized 的等价边界是同一逻辑快照下的分片在线 reference：每个分片完成
相同宽度 `L` 的构建搜索，合并全部 beam 后执行一次相同 RobustPrune，并等待本次
insert 所选邻居的反向边完成。它不等价于离线 builder 的全候选构图。当前也没有
完整入边索引，所以 delete/upsert 不能同步清除所有历史未知入边；报告中的 durable
或 drained 仅表示已声明的 maintenance 任务完成，不应解释为全图整理已经完成。

同一 4 KiB 控制页的 offset 1024 还发布固定 8 槽 canonical route 快照；它不改变
`StorageControlBlock`、索引文件或任何 RPC 布局。旧存储二进制没有该运行时扩展，
因此新计算节点会在启动校验时拒绝混合部署；同步升级二进制即可，不需要重建索引。

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
性能查询池为 `[100M,105M)` 的 500 万行。为便于当前机器直接预跑，默认插入池
为已有的 `[103M,105M)` 200 万行。文件默认位于
`/data/xjs/datasets/sift1b`：

```text
sift100m_to_105m_query.u8bin
sift103m_to_105m_insert.u8bin
```

`run_breakdown.sh` 默认只校验并读取预生成文件，不会在计算节点寻找
`bigann_base.bvecs`。只有显式设置 `PREPARE_BENCHMARK_DATA=1` 时才会调用数据准备。
可通过
`PERFORMANCE_QUERY_FILE`、`INSERT_FILE` 覆盖路径，或用以下变量
调整源区间：`PERFORMANCE_QUERY_START`、`PERFORMANCE_QUERY_END`、
`INSERT_VECTOR_START`、`INSERT_VECTOR_END`。常用选项可直接查看
`./experiment/run_breakdown.sh --help`。例如：

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

如需比较不同运行，保持相同的索引、查询/插入文件和 GPU 参数即可。
报告只提供吞吐、延迟、召回、GPU 内存与 stage2 遥测；不包含自动验收结论。

短跑示例：

```bash
WORKLOAD=query RECALL_QUERIES=100 \
WARMUP_SECONDS=1 MEASURE_SECONDS=5 \
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
```

结果写入 `experiment/reports/04_gpu_persistent_gpunetio/`。报告保留下列原始指标，
由实验者结合目标负载自行分析：

- `gpu_persistent.direct_path_failures == 0`；
- 前后 recall 及其变化；
- 没有 unhealthy/fail-stop 日志；
- GPU 和 RDMA 指标显示多查询并发，而非单查询串行等待。

可与 OdinANN 或历史 JSON 比较：

```bash
python3 experiment/compare_reports.py \
  --baseline /path/to/odinann.json \
  --candidate experiment/reports/04_gpu_persistent_gpunetio/latest.json
```

比较工具只输出原始吞吐、延迟、加速比和 recall 差值，不给出自动通过/失败结论。

停止本机启动的存储进程：

```bash
./experiment/stop_memory_nodes.sh
```
