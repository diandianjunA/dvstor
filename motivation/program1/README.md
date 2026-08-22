# 方案一动机实验（快速版）

五个问题压缩成三次启动、两段计算实验：

| 存储 case | 得到的数据 |
| --- | --- |
| `baseline` | 同步完整更新的本地 Stage1、跨分片全局 refinement、远程反向更新等关键路径分解；baseline 更新吞吐 |
| `solution` | 两阶段更新吞吐、Stage2 平均/P99 延迟、迁移前后跨分片边比例 |
| `quality` | 延迟 Stage2 后的 Stage1-only 自命中率，以及 Stage2 完成后的自命中率 |

修复后的 baseline JSON 在 `coupled_insert_critical_path.stack` 中提供可直接绘制堆叠柱的互斥阶段；solution 会在 maintenance drain 后等待控制页发布覆盖全部测量更新，避免把未发布的 Stage2/locality 计数误报成 0。

默认只做一轮固定操作数实验（128 次预热、1000 次测量），不做耗时的十轮重复。

## 运行

计算节点：

```bash
cd /home/xjs/experiment/dvstor
./motivation/program1/run_program1.sh
```

脚本每次暂停时，在存储节点执行它打印的 `start_storage_case.sh` 命令。前两组完成后，计算节点继续：

```bash
RUN_ROOT=/上一步打印的/results/program1_时间戳 \
./motivation/program1/run_quality.sh
```

精度实验默认插入 1000 个点并查询全部 1000 个新点，Stage2 延迟 15 秒并等待 30 秒；同时报告 self-hit@10 和 Stage1/最终 Top-10 结果重合率。若插入与第一次查询合计超过 15 秒，把计算端和存储端的 `QUALITY_STAGE2_DELAY_MS` 同时调大。

汇总与画一张四联动机图：

```bash
python3 motivation/program1/summarize_program1.py "$RUN_ROOT"
python3 motivation/program1/plot_program1.py "$RUN_ROOT"
```

输出包括 `summary.json`、`summary.csv` 和无需额外 Python 包的矢量图 `program1_motivation.svg`。正式论文前建议再将 `MEASURE_OPS` 提高到 5000，并重复 3 次；快速判断方案动机时默认配置足够。
