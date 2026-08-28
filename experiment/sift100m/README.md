# DVStor 论文实验

本目录只提供三项方案的动机实验和一套累积消融实验：

| 实验 | 计算节点入口 | 内容 |
|---|---|---|
| 方案一 | `program1/run_program1.sh` | RDMA 更新开销、两阶段更新收益、短期质量与 Stage2 收敛 |
| 方案二 | `program2/run_program2.sh` | 动态邻接长度、三种远程读取方式和端到端收益 |
| 方案三 | `program3/run_program3.sh` | 批量扩展引起的 Merge 膨胀、可重叠窗口和方案效果 |
| 消融实验 | `ablation/run_ablation.sh` | GPU-centric Baseline → +P1 → +P3 → +P2 |

四套实验的详细参数和断点续跑方式见各自目录中的 `README.md`。

## 公共运行支撑

根目录下其余脚本不是额外实验，而是四套实验共同调用的最小基础设施：

- `run_breakdown.sh`：统一 benchmark 与 JSON/TXT 报告生成；
- `common.sh`、`sift100m_common.sh`、`profiles/`：数据、索引和运行配置；
- `start_memory_node.sh`、`start_all_memory_nodes.sh`、`stop_memory_nodes.sh`：存储节点生命周期；
- `prepare_sift100m_data.sh`、`build_sift100m_index.sh`：首次部署时的数据和索引准备。

## 快速入口

计算节点从仓库根目录执行其中一个：

```bash
./experiment/sift100m/program1/run_program1.sh
./experiment/sift100m/program2/run_program2.sh
./experiment/sift100m/program3/run_program3.sh
./experiment/sift100m/ablation/run_ablation.sh
```

每个脚本会暂停并打印对应的存储节点启动命令。结果写入各实验自己的
`results/`，不会再写入另一套平行的实验目录。

## 公共配置

默认使用 SIFT100M schema-16 索引。数据路径、存储地址、GPU、RDMA 和内存预算集中在
`sift100m_common.sh`；Baseline/Full 的公共参数位于
`profiles/04_gpu_persistent_gpunetio_common.sh`。首次部署可按需运行：

```bash
./experiment/sift100m/prepare_sift100m_data.sh
./experiment/sift100m/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```
