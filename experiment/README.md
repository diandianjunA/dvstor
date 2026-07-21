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
```

## 构建新索引

存储节点脚本默认使用独立的 `build-storage`，先按根目录 README 配置 CPU-only
构建，不能与计算节点的 `build` 共用。

该命令先构建 compact Vamana/Metis 分片，再训练 OPQ/PQ32 并写码流：

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

完整构建以 schema-15 tagged graph 为中间态，最终直接生成 schema-16
OPQ/PQ32 运行索引，无需随后重编码。
推荐使用 `PQ_INDEX_PREFIX=/new/prefix` 保留已有索引；只有明确要删除目标 prefix
下旧产物并原地重建时才设置 `OVERWRITE_INDEX=1`。

## 转换旧 compact-v1 索引

完整、未包含在线 mutation 的旧 schema-15 `vamana_compact_v1` 基图可以流式
转换，不需要重新运行 Vamana 或 METIS，也不需要重新训练 OPQ/PQ。转换保持每个
物理分片的 slot 顺序，因此保留原始向量字节、图拓扑、METIS placement 和跨分片边；
它会重写 fixed record、5-byte compact edge、所有 `RemotePtr`，并重新生成 bound
idmap v2 和物理分片 centroid v2。旧 PQ model 被复用，base PQ codes 从精确向量
重新编码。

先执行只读的全量校验：

```bash
./build/vamana_legacy_index_converter \
  --input-prefix "$OLD_PREFIX" \
  --output-prefix "$NEW_PREFIX" \
  --dry-run
```

校验通过后写到新的 prefix：

```bash
./build/vamana_legacy_index_converter \
  --input-prefix "$OLD_PREFIX" \
  --output-prefix "$NEW_PREFIX" \
  --chunk-vectors 65536 \
  --threads 32
```

转换器禁止原地执行或覆盖已有输出，并在所有分片、idmap 和 centroid 完成后最后
发布 metadata。输入至少需要 metadata、全部旧 `.dat` 和旧 `.pqM` model；旧
`.codes`、`.anchors`、原始 dataset 和旧 idmap 都不是恢复静态基图所必需的。
`--graph-only` 可停在新的 tagged schema-15 中间态。转换器拒绝 deleted、非零
generation 或含动态 slot 的运行时快照；这类在线状态没有足够的旧持久化语义可安全
映射到 incarnation/provisional/centroid 新契约，只能从一致的静态快照重新生成。

## 部署文件

计算节点（查询与更新）：

```text
<prefix>.meta.json
<prefix>.pq32
```

纯查询配置使用 `enable-updates = false`，需要 `<prefix>.meta.json` 和
`<prefix>.pq32`；不会启动更新执行器。在线 mutation 会持续
更新 storage owner 的 centroid publication；每个计算节点从 storage 拉取同一版本化
快照，因此其他计算节点写入的新代表节点同样可见。

存储节点 X：

```text
<prefix>.meta.json
<prefix>_nodeX_ofN.dat
<prefix>_nodeX_ofN.idmap
<prefix>_nodeX_ofN.centroid
<prefix>_nodeX_ofN.pq32.codes
```

计算节点不需要 `.dat`、`.idmap`、`.pq32.codes` 或 `.gpu.idx`。METIS 只决定物理
placement；基础和动态 ID 的逻辑 authority 都由 `ID % N` 确定，其存储端 idmap
负责解析当前物理记录。因而增加计算节点不会复制一份 O(N) 的 ID 目录；每个存储
节点只加载自己的 `owner_sharded_v2_bound` idmap。该文件与整次构建和 owner
分片指纹强绑定，并校验完整长度、payload/header checksum、`ID % N`、tagged
`RemotePtr` 静态范围及重复 ID；旧 v1 会被直接拒绝。
加载后基础项只保留紧凑的 `ID -> RemotePtr`，完整的代际、提交回执和迁移状态仅为
实际参与 mutation 的 ID 分配。离线 writer 同样以每 owner 临时流单遍分桶，不在
内存中复制一份全量 idmap payload。
旧索引不能通过复制或改名 sidecar 升级：schema-16 运行格式、8-byte tagged
`RemotePtr`、构建/分片指纹、centroid sidecar v2 和 PQ code header 是同一次构建的
绑定契约。完整的 compact-v1 静态 `.dat` 可使用上面的转换器重写这些契约；缺失
`.dat` 或只有 anchor/idmap/PQ model 时信息不足，才必须重新构图。存储节点运行时
不需要 `.pq32` 模型。

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

运行格式固定为 schema 16，peer RPC 协议固定为版本 11。所有动态图指针携带
`{shard, offset, incarnation}`，记录头在读取和修改前都校验 incarnation；删除地址只在
对应维护序列 durable 后进入复用流程。查询采用 incarnation-tagged read-committed
语义，不要求动态加入的计算节点参与全局 ACK；incarnation 耗尽的槽位永久退休而不回绕。
动态目标分配使用无超时驱逐的 receipt：只有源记录进入终态且目标记录的精确身份已
确认后才结算，避免迟到重试把新对象误认为旧对象。混用旧二进制、旧分片、旧
centroid 或旧 PQ code 会在启动校验时失败。

Stage2 finalized 的等价边界是同一逻辑快照下延续 Stage1 的宽度 `L`
beam/visited/frontier，沿图中实际跨分片边完成 one-sided-RDMA 扩展后执行一次相同
RobustPrune，并等待本次 insert 所选邻居的反向边完成；它不会为每个分片重启独立
搜索，也不等价于离线 builder 的全候选构图。当前也没有
完整入边索引，所以 delete/upsert 不能同步清除所有历史未知入边；报告中的 durable
或 drained 仅表示已声明的 maintenance 任务完成，不应解释为全图整理已经完成。

控制页通过 descriptor 指向独立、可变长度的 centroid route publication；容量由
维度、标量类型和 live-entry 上限计算，不受 4 KiB 控制页固定槽位约束。旧存储
二进制没有该运行时扩展，因此新计算节点会在启动校验时拒绝混合部署；必须同步
升级全部二进制，并使用新 builder 重建或用上述工具完整转换索引。

maintenance observation 同时输出窗口可差分的 locality 计数器：Stage2 continuation
次数、远端 frontier/展开/评分记录数、迁移数，以及以 Stage1 home 和最终 home 计算的
跨分片边数。benchmark 只对测量窗口前后的单调累积值做差，报告
`home_match_rate`、`cross_edge_reduction_ratio` 和每次 continuation 的平均远端工作量；
这些指标包含窗口内全部请求，不使用请求抽样或数据集专用捷径。

动态节点的 PQ code 和图记录都以存储节点上的权威记录为准，GPU 查询通过
one-sided RDMA 按需读取，不维护需要广播、同步或回收的计算侧 dynamic-PQ 副本。
stage2 context 有界回收；存储节点记录同时保持逻辑 generation 与物理 incarnation
稳定，并在 durable watermark 后以 header-last 方式发布复用后的新 incarnation。

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
