# Program 3：精确前沿驱动的 GPU–RDMA 推进解耦

## 推荐：按论文故事运行精简实验

新的 story runner 先扫描扩展批量 `C=1/4/8/16`，采集每轮候选槽位、完整
Merge pipeline 时间、exact-prefix 时间及其占比；随后只运行一次 C=16 的
Late/Early 性能对照。总共 6 个 case：

```bash
cd /home/xjs/experiment/dvstor
./motivation/program3/run_story.sh
```

计算节点会逐个提示存储节点命令，例如：

```bash
GPU_COMMIT_WIDTH=4 ./motivation/program3/start_storage_case.sh early
```

默认动机点测量 10 秒，最终性能点测量 20 秒。若只想快速检查：

```bash
MOTIVATION_SECONDS=3 PERFORMANCE_SECONDS=5 RECALL_QUERIES=100 \
  ./motivation/program3/run_story.sh
```

可分阶段运行或在失败后复用结果目录：

```bash
RUN_ROOT=/path/to/program3_story_TIMESTAMP ./motivation/program3/run_story.sh motivation
RUN_ROOT=/path/to/program3_story_TIMESTAMP ./motivation/program3/run_story.sh performance
RUN_ROOT=/path/to/program3_story_TIMESTAMP ./motivation/program3/run_story.sh summarize
```

输出的主报告是 `方案三_动机与性能实验报告.md`，主图是
`program3_story_motivation.svg` 和 `program3_story_effectiveness.svg`。

下面的 `run_program3.sh` 保留为多次稳定性验证脚本，不再是论文故事的首选入口。

该实验只比较一个变量：相同的 mandatory graph reads 是在完整 Beam 发布后发出，
还是在 exact frontier certificate 生成后提前发出。

两组均固定为：

- Persistent GPU + GPUNetIO；
- Stable-Run Beam merge；
- Live/DynaExtent；
- `issue width == commit width == 16`；
- speculative tail 关闭。

因此不要使用正式 profile 的 `coupled` 模式作为 baseline；该模式会切换到
HostOrchestrated 查询引擎，无法隔离方案三。

## 正式运行

计算节点：

```bash
cd /home/xjs/experiment/dvstor
REPEATS=3 WARMUP_SECONDS=10 MEASURE_SECONDS=30 \
  ./motivation/program3/run_program3.sh
```

脚本每个 case 都会暂停并提示存储节点命令。存储节点按提示运行：

```bash
cd /home/xjs/experiment/dvstor
./motivation/program3/start_storage_case.sh late
# 或
./motivation/program3/start_storage_case.sh early
```

默认运行 query-only 和固定 500 update/s 的 mixed workload，各做 3 组 AB/BA
配对重复。每个 mixed case 都重启存储节点，避免不同 case 复用已经更新过的状态。

## 快速检查

```bash
REPEATS=1 WORKLOADS=query WARMUP_SECONDS=2 MEASURE_SECONDS=5 \
RECALL_QUERIES=100 ./motivation/program3/run_program3.sh
```

若等价性测试使用的物理 GPU 不是 1，可单独设置
`VERIFY_CUDA_VISIBLE_DEVICES=0`；该变量不会改变正式 benchmark 的 GPU 配置。

只运行某一类负载：

```bash
./motivation/program3/run_program3.sh query
./motivation/program3/run_program3.sh mixed
```

失败后复用结果目录补跑或重新汇总：

```bash
RUN_ROOT=/path/to/program3_TIMESTAMP ./motivation/program3/run_program3.sh query
RUN_ROOT=/path/to/program3_TIMESTAMP ./motivation/program3/run_program3.sh summarize
```

输出包括 `summary.json`、`summary.csv`、中文分析报告、动机图和效果图。
