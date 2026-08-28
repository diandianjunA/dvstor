# Program 2：动态长度读取动机与有效性实验

主实验现在只在动态 mixed workload 下比较三种真实可实现策略：

- `Fixed`：每次读取完整的 832 B 图记录；
- `Header→Neighbor`：先读 16 B header，再串行读取精确 neighbor body；
- `ClassExtent`（配置名 `live-extent`）：根据长度 class 一次读取，低估时 fallback。

三个 case 使用同样的查询/插入文件、目标查询速率、目标更新速率和动态
extent 发布逻辑。每个 case 都必须从重新启动的干净存储节点开始。默认测量
为 5 秒预热、20 秒正式阶段、500 update/s。16 个专用写线程通过独立 pacer
维持更新到达率，其余查询线程闭环饱和。脚本要求更新速率达成率至少为
95%，并要求查询真正展开至少
100 个动态版本，而且动态版本占权威展开节点至少 0.1%，否则拒绝汇总，避免
出现“名义 mixed，实际几乎没访问动态节点”。

## Oracle 的含义

Oracle 不是第四种端到端策略。GPU 只在某个动态版本已经被读取、校验并
正式展开以后，记录它的真实邻居数。汇总器据此离线计算：

```text
Oracle bytes = 16 × dynamic_parents + 8 × live_neighbor_sum
Oracle WQEs  = 1 × dynamic_parents
```

这等价于“查询发起前零开销知道精确长度”的理论下界。Oracle 不参与请求
准备，也没有端到端 QPS 柱子。可选的 RDMA probe 只是单次精确长度读取的
协议层 sanity check，不能表述成可部署系统性能。

## 运行方法

在计算节点：

```bash
cd /home/xjs/experiment/dvstor
./experiment/sift100m/program2/run_program2.sh
```

计算节点会依次提示三个命令。在存储节点分别运行：

```bash
./experiment/sift100m/program2/start_storage_case.sh fixed
./experiment/sift100m/program2/start_storage_case.sh header
./experiment/sift100m/program2/start_storage_case.sh live
```

每个新命令都会停止并重新启动全部 memory node，所以不要复用上一个 case
已经发生更新的存储状态。

时间更紧时可以先做 smoke test：

```bash
WARMUP_SECONDS=2 MEASURE_SECONDS=5 RECALL_QUERIES=100 \
MIN_DYNAMIC_EXPANDED=10 \
MIN_DYNAMIC_SHARE=0 \
  ./experiment/sift100m/program2/run_program2.sh
```

正式测试可调整固定更新压力，但三个 case 必须使用同一个 `RUN_ROOT` 和相同
参数：

```bash
TARGET_WRITE_QPS=500 WRITE_THREADS=16 \
WARMUP_SECONDS=10 MEASURE_SECONDS=30 \
  ./experiment/sift100m/program2/run_program2.sh
```

某个 case 失败后可在同一结果目录补跑：

```bash
RUN_ROOT=/path/to/program2_TIMESTAMP ./experiment/sift100m/program2/run_program2.sh live
RUN_ROOT=/path/to/program2_TIMESTAMP ./experiment/sift100m/program2/run_program2.sh summarize
```

可选协议 probe（不是主实验必需）：

```bash
RUN_ROOT=/path/to/program2_TIMESTAMP ./experiment/sift100m/program2/run_program2.sh probe
```

结果包括 `summary.json`、`summary.csv`、`dynamic_degree_histogram.csv`、
`program2_motivation.svg` 和 `program2_effectiveness.svg`。

图 1 只展示两种传统方法与 Oracle 理论下界，解释为什么“固定全读”和
“Header 串行两读”都不理想；图 2 展示三种真实策略的动态端到端吞吐，以及
实际动态读取字节数相对 Oracle 下界的差距、fallback 比例和动态访问占比。
