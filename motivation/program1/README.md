# 方案一动机实验

本目录对应“可达性优先的续跑式两阶段插入与更新边界局部性维护”的三组动机实验。实验目标不是直接证明完整设计优于 baseline，而是依次回答三个更基础的问题：同步全局插入为什么慢、只做局部插入为什么不够、在线更新为什么需要局部性修复。

## 实验矩阵

| 编号 | 问题 | 对照 | 主要指标 | 当前可运行性 |
| --- | --- | --- | --- | --- |
| M1.1 | storage-owner offload 后，ACK 前是否仍被跨分片依赖主导？ | `coupled` synchronous global vs. `decoupled` Stage1 ACK | ACK P50/P99/P999、RDMA/CPU 时间占比、搜索/剪枝/反向边子项、write QPS、混合负载 query QPS/P99 | 可直接运行 |
| M1.2 | 永久停在局部图是否虽快但损害长期质量？ | local-only vs. synchronous global vs. two-stage | ACK latency、insert QPS、动态 GT Recall@10、不可达动态节点比例、remote frontier、cross-edge ratio | 缺 local-only 产品开关与动态可达性扫描器 |
| M1.3 | centroid-only placement 是否随更新量积累 locality drift？ | relocation off vs. mutation-boundary repair on | 全图/动态图 cross-edge ratio、graph shard batches/query、graph bytes/query、QPS/P99、容量倾斜、迁移字节 | 缺 relocation-only 开关与全图 checkpoint 扫描器 |

现有 `coupled` 路径是 append-only、同步完成全局 RDMA 搜索/剪枝/稳定写入/反向边后才 ACK，适合作为 M1.1 的 synchronous-global 对照。它没有 Stage1/Stage2、maintenance debt 或 migration。因此它不能代替 M1.2 的 local-only，也不能用于 M1.3 的“同一 Stage2、只关闭 relocation”对照。

## M1.1：同步 one-stage ACK 关键路径

`run_m1_1_ack_path.sh` 对每轮使用 AB/BA 交错顺序：奇数轮 `coupled -> decoupled`，偶数轮反向。每个 case 都重启全部 memory node，从同一静态索引重新开始。除更新完成模式外，固定：

- `GPU_DYNAMIC_GRAPH_ACCESS_MODE=adaptive`
- `GPU_RDMA_SEARCH_PROGRESSION_MODE=decoupled`
- 同一 profile、索引、插入文件、ID 起点、线程和容量参数
- `ENABLE_BREAKDOWN=true`

默认包含两个 scenario：

1. `ack`：insert-only、单客户端，隔离 authority-side ACK 服务时间和细粒度分解。
2. `mixed`：固定 caller 数的 50/50 读写混合负载，观察更新协议对 query QPS/P99 的干扰。它是饱和闭环测试，`READ_RATIO` 表示 caller 比例，不保证完成操作比例。

默认假设 benchmark 在计算节点运行、memory node 由实验者在存储节点启动。每个 case 前脚本会打印本轮所需 mode，并等待确认；存储端必须停止上一 case、从同一静态索引重新启动，不能沿用上一 case 的动态内存状态。正式运行：

```bash
cd /home/xjs/experiment/dvstor
REPEATS=10 ACK_SECONDS=60 MIXED_SECONDS=120 \
./motivation/program1/run_m1_1_ack_path.sh
```

计算节点不需要 storage `.dat`，也不会启动或停止 memory node。短程校验：

```bash
SMOKE=1 \
./motivation/program1/run_m1_1_ack_path.sh
```

存储节点每轮使用配套脚本启动。它会先校验全部 shard artifact，再停止上一 case，并固定使用 CPU-only `build-storage`：

```bash
# 计算端提示 coupled_one_stage 时
./motivation/program1/start_storage_case.sh coupled_one_stage

# 计算端提示 two_stage 时
./motivation/program1/start_storage_case.sh two_stage

# 辅助操作
./motivation/program1/start_storage_case.sh status
./motivation/program1/start_storage_case.sh stop
```

存储构建目录不在默认位置时设置 `STORAGE_BUILD_DIR=/path/to/build-storage`。每次启动的日志保存在 `motivation/program1/storage_logs/<timestamp>_<case>/`。若以后希望由外部编排器自动重启，可以设置 `STORAGE_BEFORE_CASE_HOOK=/absolute/hook`；hook 会收到 `repeat scenario order case case_root` 五个参数和三个 mode 环境变量。

仅在计算、存储共置且本机确实具有全部 shard `.dat` 时，才使用：

```bash
STORAGE_NODE_MODE=local ALLOW_SERVICE_RESTART=1 SMOKE=1 \
./motivation/program1/run_m1_1_ack_path.sh
```

只查看将要执行的矩阵，不改变服务状态：

```bash
DRY_RUN=1 ./motivation/program1/run_m1_1_ack_path.sh
```

结果位于 `results/m1_1_<timestamp>/`。每个 case 有独立的 `reports/`、`logs/`、`driver.log` 和窗口前后的 `nic_before.tsv`/`nic_after.tsv`；根目录保存 `manifest.tsv` 与 `provenance.txt`。NIC 文件是本机端口原始累计计数器，使用时应按相同 counter 做 after-before，并注明共置 shard 共享端口。汇总：

```bash
python3 motivation/program1/summarize_m1_1.py \
  motivation/program1/results/m1_1_<timestamp>
```

汇总器会拒绝 mode 不一致、缺少 fine-grained breakdown 或零 insert 的报告，并生成 `summary.csv` 和 `summary.md`。论文图建议分成两张：

- 图 M1.1a：ACK P50/P99/P999 与 insert throughput。
- 图 M1.1b：同步版本每次插入的 CPU/RDMA 堆叠分解；将 search neighbor/snapshot read 与 prune snapshot read 合并为 cross-shard dependency，另画 mixed query QPS/P99。

不要把 compute-side `end_to_end_ns` 的所有时间都命名为 authority execution。权威路径细分来自 storage owner 返回的 breakdown counters；客户端排队、sender queue 和完成唤醒应单独列出。

当前 wire telemetry 能报告 owner RPC batch/item/wall time，细粒度 breakdown 能报告远端 neighbor/snapshot/prune read 时间；它还不能把 synchronous-global 的 graph WQE、vector WQE、payload bytes 和 dependency waves 分成独立计数器。正式写作若需要这些字段，应先在 coupled search 的 RDMA posting/completion 边界增加窗口可差分计数器；不能用 NIC 总包数反推逻辑 graph/vector WQE。

## M1.2：local-only 的速度—质量负结果

正式对照必须从相同静态快照、相同 mutation trace 开始，并使用三种明确语义：

- `local-only`：完成 Stage1 的本地搜索、provisional edges 与 protected backlinks 后 ACK，永久不执行 Stage2。
- `synchronous-global`：ACK 前完成全局 refinement。
- `two-stage`：Stage1 ACK，Stage2 后台完成并 drain 到固定 maintenance prefix。

在累计插入 1%、5%、10% 后，分别测 ACK 后立即、1 s、10 s、30 s 的动态 GT Recall@10。GT 必须包含新 ID，不能继续使用静态 SIFT100M ground truth。另需扫描动态节点的正常入口可达率；“查询结果中没有命中”不能替代不可达性检查。

当前代码不应通过把 maintenance worker 设为 0、堵满队列或杀掉后台线程模拟 local-only：这些做法违反“ACK 前 maintenance debt 已获准入”的协议，还会引入背压和失败语义。`check_capabilities.sh` 会报告缺失的代码接口。

建议新增的最小实验接口：

```text
--storage-owner-update-completion-mode=local-only
--mutation-trace-row-offset=N
--recall-delay-ms=0,1000,10000,30000
--dynamic-reachability-report=<json>
```

## M1.3：长期 locality drift

从同一 METIS 基图分别运行 `repair=off/on`，在累计更新比例 0%、1%、5%、10%、20% 建立逻辑 checkpoint。每个 checkpoint 先固定并 drain 已接纳的 maintenance prefix，再运行同一只读 query trace。推荐以“累计插入数”定义横轴；若包含 upsert/delete，需同时报告 live dynamic-node 数。

最重要的控制变量是：off/on 两组都必须运行完全相同的 Stage2 continuation、final RobustPrune 和反向边协议，只允许 placement decision 不同。不能用 `coupled` 充当 repair-off，因为它连 Stage2、owner-side fusion 和更新语义都一起改变了。

建议新增的最小实验接口：

```text
--storage-owner-locality-repair-mode=off|mutation-boundary
--storage-owner-locality-snapshot=<json>
--mutation-trace-row-offset=N
```

locality snapshot 至少包含：base/dynamic/live 节点数、全图与动态图 edge 数、cross-shard edge 数、每 shard live bytes/slots、migration bytes。查询报告已有 `average_graph_shard_batches_per_query`、`average_graph_read_bytes_per_query`、QPS/P99；Stage2 报告已有 current-final-neighbor 上的 counterfactual cross-edge reduction，但它不能替代全图 drift 曲线。

## 共同有效性要求

- 每轮 case 都重启全部 storage owner；禁止在上一个 case 的动态内存状态上继续跑另一个 case。
- AB/BA 至少 10 轮，报告均值、标准差、95% CI 和 paired difference；不要只挑最好的一轮。
- 正式吞吐运行可关闭细粒度计时，但 M1.1 分解运行必须开启；两类结果分开呈现。
- 保存 resolved modes、schema/build fingerprint、索引 prefix、git commit、机器/GPU/RDMA 信息和完整命令。
- NIC counters 需要在窗口前后读取端口计数器并做差；若多个 shard 共用同一 NIC，只能报告 NIC 级总量，不能伪装成 per-owner 独立计数。
- M1.2/M1.3 的 mutation trace 必须使用不重叠 ID/row range，并记录每个 checkpoint 的精确范围。
