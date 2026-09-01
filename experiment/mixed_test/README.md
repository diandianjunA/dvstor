# 固定查询线程、扫描更新线程的混合负载测试

这个测试使用各数据集现有的 `run_breakdown.sh`，为查询和更新分配独立的
闭环线程。查询线程数固定，更新线程数按列表变化；`baseline` 和 `full` 分别是：

- `04_gpu_persistent_gpunetio_baseline`
- `04_gpu_persistent_gpunetio`

## 快速运行

在计算节点启动扫描：

```bash
cd /home/xjs/experiment/dvstor
./experiment/mixed_test/run_mixed_test.sh \
  --query-threads 256 \
  --update-threads 0,16,32,64,128,256 \
  --profiles baseline,full \
  --repeats 3
```

每个 case 开始前，脚本会打印一条带唯一 token 的存储节点命令并暂停。到存储
节点执行该命令，看到 `storage ready` 后回到计算节点按 Enter。每个 case 都必须
重启存储服务，使动态索引恢复到相同的静态起点；不能让上一个更新测试的插入
残留到下一个点。

默认测试 SIFT100M，预热 15 秒、测量 120 秒。其他数据集可指定：

```bash
./experiment/mixed_test/run_mixed_test.sh \
  --dataset deep100m --query-threads 128 --update-threads 0,8,16,32 \
  --profiles full
```

先检查解析出的矩阵而不创建目录：

```bash
./experiment/mixed_test/run_mixed_test.sh \
  --query-threads 256 --update-threads 0,32,64 --dry-run
```

## 结果与续跑

结果默认写入 `experiment/mixed_test/results/mixed_<时间戳>/`：

- `runs/<case>/report.json`：benchmark 原始报告；
- `runs/<case>/driver.log`：完整运行日志；
- `raw_results.csv`：每次重复一行；
- `summary.csv`：按数据集、profile、查询线程和更新线程聚合的均值与 95% CI。

更新线程列表中包含 `0` 时，同一 profile 的纯查询点会作为
`normalized_query_qps` 的 1.0 基线。

中断后使用同一个目录续跑，已有 `DONE` 的 case 会自动复用：

```bash
./experiment/mixed_test/run_mixed_test.sh \
  --query-threads 256 --update-threads 0,16,32,64,128,256 \
  --profiles baseline,full --repeats 3 \
  --result-root /path/to/previous/result
```

只重新汇总已有结果：

```bash
RESULT_ROOT=/path/to/result \
  ./experiment/mixed_test/run_mixed_test.sh --summarize-only
```

自动化调度可加 `--no-storage-prompt`，但此时调用者必须保证每个 case 前已用匹配
profile 重置所有 memory node；脚本仍会从 benchmark 报告校验实际 profile 和线程划分。
