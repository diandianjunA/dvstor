# 累积消融

四个 case 依次启用两阶段更新、GPU/RDMA 推进解耦和动态长度读取，共用 SIFT100M 索引
与质量、容量参数。运行准备见 [实验说明](../../README.md)。

| case | 编码 | 更新完成模式 | 图访问模式 | 查询推进模式 |
|---|---|---|---|---|
| `baseline` | 000 | coupled | fixed | manual |
| `program1` | 100 | decoupled | fixed | manual |
| `program3` | 101 | decoupled | fixed | decoupled |
| `full` | 111 | decoupled | adaptive | decoupled |

此处 baseline 显式使用 Persistent + GPUNetIO、Stable-Run，关闭提前推进；直接运行
正式 baseline profile 则会选择 HostOrchestrated，两者不能混用。manual case 的
commit/issue width 均为 16；decoupled case 的 issue 上限由运行时自动解析。

默认预热 15 秒、测量 120 秒、查询/插入各占 50% 线程，客户端并发自动选择，上限 1024。
可以覆盖 `WARMUP_SECONDS`、`MEASURE_SECONDS`、`READ_RATIO`、`CLIENT_THREADS` 和
`CLIENT_THREAD_CAP`，同一次比较的四组保持一致。

```bash
BUILD_DIR="$PWD/build-compute" ./experiment/sift100m/ablation/run_ablation.sh
```

每个 case 前脚本暂停；在存储节点运行提示中的命令，例如：

```bash
STORAGE_BUILD_DIR="$PWD/build-storage" \
  ./experiment/sift100m/ablation/start_storage_case.sh baseline
```

看到 `storage ready` 后继续，随后依次启动 `program1`、`program3`、`full`。
存储脚本会停止上一组进程，再从同一静态索引重新加载。

输出位于 `results/ablation_时间戳/`，包含原始 JSON/TXT、manifest、`summary.csv`、
`ablation_performance.svg` 和分析报告。中断后补跑或重新汇总：

```bash
RUN_ROOT=/path/to/ablation_TIMESTAMP ./experiment/sift100m/ablation/run_ablation.sh program3
RUN_ROOT=/path/to/ablation_TIMESTAMP ./experiment/sift100m/ablation/run_ablation.sh summarize
```
