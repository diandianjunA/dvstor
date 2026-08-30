# Program 1：两阶段更新动机实验

唯一入口为 `run_program1.sh`。它依次测试同步更新基线、两阶段更新方案，以及
Stage1 短期质量与 Stage2 收敛，不再需要单独运行质量脚本。

```bash
cd /home/xjs/experiment/dvstor
./experiment/deep100m/program1/run_program1.sh
```

计算节点暂停时，在存储节点执行提示中的
`./experiment/deep100m/program1/start_storage_case.sh <case>`。断点续跑时复用 `RUN_ROOT`，
也可单独运行 `baseline`、`solution`、`quality` 或 `summarize`。
