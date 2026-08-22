# Program 3：精确前沿驱动的 GPU–RDMA 推进解耦

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
