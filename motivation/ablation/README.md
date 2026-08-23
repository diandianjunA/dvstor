# 三项方案累积消融

四个配置只改变三个正式的贡献级模式，其他配置完全继承 `experiment` 下的正式
SIFT100M profile：

| case | 编码 | 方案一 | 方案二 | 方案三 |
|---|---:|---|---|---|
| `baseline` | 000 | 关 | 关 | 关 |
| `program1` | 100 | 开 | 关 | 关 |
| `program3` | 101 | 开 | 关 | 开 |
| `full` | 111 | 开 | 开 | 开 |

Baseline 也是 Persistent GPU + GPUNetIO，只关闭方案三的 early/ahead-of-commit
progression；HostOrchestrated CPU-posted 模式不参与贡献消融。累积顺序为
`Baseline → +P1 → +P3 → +P2 (Full)`，让方案二在高性能 GPU 查询路径上体现收益。

`full` 直接使用 `04_gpu_persistent_gpunetio` 正式 profile；默认 warmup 15 秒、测量
120 秒、50% 查询线程和 50% 插入线程、自动 512 并发，与指定的主实验报告一致。

## 运行

计算节点：

```bash
cd /home/xjs/experiment/dvstor
./motivation/ablation/run_ablation.sh
```

脚本每到一个 case 会暂停。此时在存储节点按提示运行，例如：

```bash
cd /home/xjs/experiment/dvstor
./motivation/ablation/start_storage_case.sh baseline
```

看到 `storage ready` 后回到计算节点按 Enter。随后依次按提示启动 `program1`、
`program3` 和 `full`。存储脚本会停止上一组存储进程，再从同一静态索引重新启动，
避免前一组动态更新污染下一组。

结果写入 `motivation/ablation/results/ablation_时间戳/`，包括原始 JSON/TXT、
`summary.csv`、`ablation_performance.svg` 和 `消融实验分析报告.md`。

若某一组中断，可复用原结果目录补跑：

```bash
RUN_ROOT=/home/xjs/experiment/dvstor/motivation/ablation/results/ablation_YYYYmmdd_HHMMSS \
  ./motivation/ablation/run_ablation.sh program3
```

四组完成后重新汇总：

```bash
RUN_ROOT=/home/xjs/experiment/dvstor/motivation/ablation/results/ablation_YYYYmmdd_HHMMSS \
  ./motivation/ablation/run_ablation.sh summarize
```

存储节点若使用独立构建目录，可在启动时指定，例如：

```bash
STORAGE_BUILD_DIR=/home/xjs/experiment/dvstor/build-storage \
  ./motivation/ablation/start_storage_case.sh full
```
