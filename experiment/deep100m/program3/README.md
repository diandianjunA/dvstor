# Program 3：精确前沿驱动的 GPU–RDMA 推进解耦

Program 3 只保留一套动机实验，与 Program 1/2 一样使用统一入口
`run_program3.sh`。实验先扫描扩展批量 `C=1/4/8/16`，采集每轮候选槽位、完整
Merge pipeline 时间以及 exact-prefix 占比；随后在 C=16 下各运行一次 Late/Early
性能对照，共 6 个 case。

## 运行

计算节点：

```bash
cd /home/xjs/experiment/dvstor
./experiment/deep100m/program3/run_program3.sh
```

计算节点会逐个提示存储节点命令，例如：

```bash
cd /home/xjs/experiment/dvstor
GPU_COMMIT_WIDTH=4 ./experiment/deep100m/program3/start_storage_case.sh early
```

默认动机点预热 5 秒、测量 10 秒，最终效果点预热 5 秒、测量 20 秒，每个 case
只运行一次。快速检查：

```bash
MOTIVATION_SECONDS=3 PERFORMANCE_SECONDS=5 RECALL_QUERIES=100 \
  ./experiment/deep100m/program3/run_program3.sh
```

## 分阶段与断点续跑

```bash
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/deep100m/program3/run_program3.sh motivation
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/deep100m/program3/run_program3.sh performance
RUN_ROOT=/path/to/program3_TIMESTAMP ./experiment/deep100m/program3/run_program3.sh summarize
```

输出包括：

- `summary.json` 和 `motivation_width_sweep.csv`；
- `方案三_动机与性能实验报告.md`；
- `program3_motivation.svg`；
- `program3_query_time_breakdown.svg`；
- `program3_candidate_maintenance_breakdown.svg`；
- `program3_effectiveness.svg`。

Late/Early 均固定为 Persistent GPU + GPUNetIO、Stable-Run、Live/DynaExtent，且
`issue width == commit width`、speculative tail 关闭。两者只比较 mandatory graph
reads 是在完整 Beam 发布后发出，还是在 exact frontier certificate 就绪后提前发出。
不要使用正式 profile 的 `coupled` 模式作为方案三 baseline，因为它会切换到
HostOrchestrated 查询引擎，无法隔离方案三。
