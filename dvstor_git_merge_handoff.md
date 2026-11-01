# dvstor Git 合并与多节点代码交接

日期：2026-11-01

## 1. 当前状态与边界

- 201 `/home/xjs/experiment/dvstor` 是当前约定的规范树。检查时 `main` 与 `origin/main` 一致，HEAD 为 `0d4a55a4917947bd8bf0609f9cca6a75198d2057`（`update rdma connection`）。
- 202 `/home/xjs/experiment/dvstor` 正处于 `git pull` 产生的合并冲突中，尚未执行 `merge --abort`、`reset` 或冲突解决。
- 202 本地 HEAD 为 `4dcf03111a2e60193373f5a4e4ddfe565863a848`（`stage`），远端合并头为上述 `0d4a55a`。共同祖先是 `cdd60a0f17f5980968374159a83a8b3f4f85d8d0`。
- 提交关系：202 ahead 1、behind 2。远端路径为 `632ae09 checkpoint` -> `0d4a55a update rdma connection`，202 是独立提交 `4dcf031 stage`。
- `git cherry MERGE_HEAD HEAD` 显示 `4dcf031` 不是远端已有补丁的等价提交。
- 127 `/home/xjs/dvstor` 是 Git 工作树，但有本地修改且 ahead 1、behind 1。128、129 的 `/home/xjs/dvstor` 不是 Git 工作树，不能直接在那里 `git pull`。

本文件只做交接，不授权丢弃任何提交。先备份，再合并；不要直接全选 ours/theirs。

## 2. 冲突原因

`add/add` 是因为共同祖先没有相应文件，而 202 和 201 分别新增了不同版本，例如：

- `tools/vamana_offline/checkpoint.cc/.hh`
- `experiment/sift1b/gen_cn_config.sh`
- `experiment/sift1b/sift1b_common.sh`

`content` 是因为双方修改了同一区域，例如：

- `rdma-library/library/connection_manager.cc/.hh`
- `src/memory_node/memory_node.cc`
- `tools/vamana_offline/graph.cc/.hh`
- `tools/vamana_offline/shard_writer.cc`

“201 是规范树”不代表 Git 会自动用 201 覆盖另一个本地提交，也不代表时间较新的提交自动获胜。

## 3. 202 必须保留的已验证构建修复

以下差异来自直接比较 `4dcf031` 与 `0d4a55a`。它们用于产生当前高召回 DEEP1B 参考索引，不能用旧实现覆盖。

### 图构建语义

文件：`tools/vamana_offline/graph.cc/.hh`、`config.cc/.hh`

- 空图增量链接；随机初始图仅保留为诊断选项。
- 默认 1.3R 临时构建容量，最终度数仍为 R。
- 反向边达到临时容量时立即剪枝，不能静默丢边。
- robust prune 使用渐进 alpha。
- 建图剪枝候选池仅包含实际展开节点；这是全量构建的重要加速，不要回退成全部访问节点。
- 默认不再做旧的全局反向边合并，最终仅清理超过 R 的节点。
- 保留图健康统计：双向边、静默丢边、medoid 自可达性等。

### Checkpoint v2

文件：`tools/vamana_offline/checkpoint.cc/.hh`

- 元数据记录 build capacity、beam、alpha、slack、构建语义开关、数据类型等构建身份。
- 支持 `final_prune` 阶段恢复。
- 202 版本没有 201 的远程 checkpoint 支持。合并时必须同时保留“202 的 v2 元数据/恢复语义”和“201 的远程 checkpoint 能力”，不能简单二选一。

### 召回闸口与导出

文件：`tools/vamana_offline/recall_check.cc`、`shard_writer.cc`

- recall@1/5/10 共享搜索结果计算。
- `min-recall-at-10` 未达标时拒绝分片导出。
- 热图按 64K 条一批顺序写入，避免逐条随机写。
- 每个分片同步校验并执行 ship hook，防止全部写完后才分发导致磁盘打满。

### 需要筛选而非全部接收的 202 独有实验代码

- `tools/vamana_graph_repair.cc`
- `tools/vamana_offline/reverse_repair.cc/.hh`
- `test/early_prune_equivalence.cc`
- `test/vamana_offline_graph_test.cc`
- `test/vector_dtype_test.cc`

旧 DEEP 图的“64 原边+32 反向边”抢救已失败，不要把 `reverse_repair` 当成正式构建路径。测试和诊断工具可单独评估保留。

## 4. 201 也有 202 缺少的内容

不要用 202 整体覆盖 201。至少包括：

- 最新 RDMA 重连相关实现和 `test/server_connection_reconnect_test.cc`。
- 远程 checkpoint 支持。
- `test/kmeans_partition_test.cc`、PQ/OPQ 测试和 microbenchmark。
- 部分运行时脚本及 GPU/存储生命周期更新。

MN 复用目前只通过过编译/契约检查，双会话实际验证没有成功，正式实验未启用。最新提交不等于该功能已经验收。

## 5. 四台存储节点并不一致

对 `src`、`rdma-library`、`tools`、`experiment`、`test` 中源码/脚本（排除 build、日志、`.bak` 和 `__pycache__`）做 SHA-256 清单比较，结果如下：

| 节点 | 清单文件数 | 与 201 不同数 | 与 202 不同数 |
| --- | ---: | ---: | ---: |
| 127 | 430 | 45 | 57 |
| 128 | 338 | 142 | 151 |
| 129 | 362 | 119 | 128 |

另外，127/128/129 三者共同存在的文件中仍有 14 个内容不一致；它们还有各自缺失的文件，不能仅凭部分核心进程能运行就认定代码一致。

四台 `dvstor_memory_node` 二进制的 SHA-256 也全部不同：

- 202：`69502ef795d10de57bcc53f1332fc8a73982b9279a7ec5180767b0dace0eb81c`
- 127：`195b30851b11affe8d3120ba00b635d4310df3cb6f8dff2ffc60c3ca740f0ba5`
- 128：`b405ef1cdbbcb0f6c989b41a1d05029592674d4b23b1b4dd984a50aaea56c39d`
- 129：`a0ab54483741bedc3547d87624a2014b1aa8105d8016e0fde1b5a2d8a48f4cd9`

二进制哈希不同不能单独证明行为不同，但证明它们不是同一份构建产物。合并后必须重新核对源码哈希并分别构建、测试，不能整树 rsync。

## 6. 当前冲突已影响实验启动

202 的 `experiment/sift1b/sift1b_common.sh` 含冲突内容，启动 MN 时实际报错：

```text
experiment/sift1b/sift1b_common.sh: line 333: syntax error near unexpected token `<<<'
```

这发生在 DEEP1B beam54 探针启动前，未产生该点召回结果。

为不干扰其他 agent 合并，当前实验临时使用 `127:/home/xjs/dvstor_runtime_v2/common.sh` 中的 `USE_EXISTING_MN_BINARIES=1`：直接启动各节点既有、此前已跑通的 MN 二进制，不 source 仓库启动脚本，也不重新编译。该方式默认关闭，只用于当前 DEEP 探针/正式实验。

合并完成后应：

1. 先验证正常启动脚本和统一二进制；
2. 再停止设置 `USE_EXISTING_MN_BINARIES=1`，恢复正常构建/启动路径；
3. 不要在当前实验运行期间替换既有 MN 二进制。

其他 agent 请只处理各仓库源码，不要修改 `/home/xjs/dvstor_runtime_v2/`，以免与正在进行的实验冲突。

## 7. 推荐合并顺序

1. 在任何 abort/reset 前，备份 202 的 `4dcf031`（备份分支及补丁）。
2. 以 201 为基线，按功能移植第 3 节的已验证构建修复，保留第 4 节的 201 新能力。
3. 对 GPU、存储生命周期和测试文件的其余差异逐项审查，不要默认某一边全部正确。
4. 不纳入运行日志、`.bak*`、实验二进制和临时数据文件。
5. 在 202 完成编译与测试后，再由用户决定是否同步/推送到 201 规范树。
6. 对 127/128/129 采用明确文件清单同步并保留备份，分别构建；最后重新做源码哈希核对。

## 8. 合并后最低验证要求

- `git diff --check`，并确认没有遗留 `<<<<<<<`、`=======`、`>>>>>>>` 冲突标记。
- `bash -n experiment/sift1b/sift1b_common.sh`，先消除当前 line 333 的语法错误。
- 分别构建 202 MN、201 CN 和离线 builder；不要覆盖正在实验使用的二进制，使用独立构建目录或等实验窗口结束。
- 运行现有图构建、checkpoint、分片和 RDMA 契约测试。
- 用小规模差分测试确认：反向边不静默丢弃、候选池只含展开节点、checkpoint v2 可恢复、最终度数不超过 R、热图顺序写和逐片同步 hook 均保留。
- 不要用小规模高召回直接外推 1B，也不要为了验证合并重新跑 20 小时全量构建；当前 DEEP 参考索引已完成且召回已验收。

## 9. 当前实验状态摘要

- DEEP 参考索引路径：`202:/data/xjs/index/dvstor_deep1b_reference/index/deep1b_R96_bw160_kmeans_ref_pmd32_pq48_schema16`。
- 全精度 recall@10 = 0.9795；真实 PQ probe 在 beam256/exp640 时 recall@10 = 0.9885。
- 当前目标是完成 12 个精确 beam 的召回探针，然后正式性能测试复用探针召回，不再重复计算召回。
- 正式协议：双 CN、每 CN 256 读+256 写、15s 预热+120s 测量、QP96；MN 复用关闭。
- SPACEV1B 暂不构建或启动。
