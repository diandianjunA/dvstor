# DVSTOR 源码逐行教程（30 课）

本教程基于 `dev` 分支的**实际源码**逐行讲解 dvstor 项目。每一课都打开真实的源文件，按
`file_path:line_number` 的粒度解释类型、内存归属、线程模型、CUDA/RDMA 语义与设计取舍。
教程不依赖注释或文档，而是以代码本身为准。

> 项目定位：面向动态向量检索的**存算分离**系统。计算节点用持久化 CUDA kernel 做图遍历 +
> PQ 评分 + 精确重排，通过 DOCA GPUNetIO 让 GPU 直接 RDMA 读取存储节点的图记录与精确向量；
> CPU 不参与稳态查询。存储节点持有图与向量，并执行 `local_stitch` 两阶段在线更新。

## 如何使用

- **按顺序读**：从第 1 课到第 30 课是自底向上的递进（基础设施 → 传输 → 索引 → GPU 引擎 →
  Kernel → DOCA → 存储节点 → 计算服务 → 离线工具）。跨课引用写作"见第 X 课"。
- **配合源码**：每课都会贴出带行号的代码片段，建议同时打开对应源文件对照。
- **图示**：数据流、状态机、内存布局用 ASCII/Mermaid 图辅助。

## 前置知识

- C++20（coroutine、`atomic_ref`、concepts）、CUDA（persistent kernel、`__shared__`、
  `__threadfence_system`、device-side RDMA/umem）、InfiniBand/RDMA verbs、DOCA GPUNetIO。
- 向量检索：Vamana/HNSW 图、OPQ/PQ 量化、RobustPrune、beam search。

## 课程大纲与文件映射

### Part I · 入门与基础设施
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 01 | 项目总览、架构与构建系统 | `README.md` `CMakeLists.txt` `.clangd` `docs/*` 目录布局 |
| 02 | 公共类型与配置 | `src/common/types.hh` `constants.hh` `vector_dtype.hh` `distance.hh` `configuration.hh` `index_path.hh` |
| 03 | 并发原语与协程 | `src/common/atomic_utils.hh` `bounded_queue.hh` `completion_pool.hh` `sliding_completion_ring.hh` `core_assignment.hh/cc` `src/coroutine.hh` `src/remote_pointer.hh` `src/common/timing.hh/cc` |
| 04 | RDMA 传输库（上） | `rdma-library/library/context.hh/cc` `configuration.hh/cc` `types.hh` `utils.hh/cc` `thread.hh` `memory_region.hh/cc` `dynamic_region_allocator.hh` `hugepage.hh` |

### Part II · 传输层与索引格式
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 05 | RDMA 传输库（下）：QP 与连接管理 | `queue_pair.hh/cc` `detached_qp.hh` `connection_manager.hh/cc` `batched_read.hh` |
| 06 | Vamana 图格式与 anchor/idmap | `src/vamana/vamana_node.hh` `hot_graph.hh` `anchor_index.hh/cc` `idmap.hh` `storage_layout_resolver.hh` `adaptive_route_table.hh/cc` |
| 07 | 索引格式契约 schema-15 | `src/gpu_search/index_format.hh` `index_format.cc` |
| 08 | 元数据、owner map 与存储协议 | `src/service/index_metadata.hh/cc` `base_owner_map.hh/cc` `storage_owner_protocol.hh` `storage_owner_client_helpers.hh` `query_result.hh` |

### Part III · GPU 搜索引擎
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 09 | GPU 引擎类型、遥测与 PQ 模型 | `src/gpu_search/types.hh/cc` `pq_index.hh/cc` |
| 10 | Delta 索引、动态路由 overlay 与预算 | `delta_index.hh/cc` `dynamic_route_overlay.hh/cc` `dynamic_route_consistency.hh` `delta_scan_budget.hh` `initial_seed_budget.hh` `memory_budget.hh` `navigation_bootstrapper.hh/cc` |
| 11 | 持久化引擎 PImpl 与生命周期 | `persistent_engine.hh/cc` `persistent_engine/impl.hh` `lifecycle.cc` `health.cc` `cuda_helpers.hh` |
| 12 | 引擎构造与资源分配（上） | `persistent_engine/construction.cc`（前半：内存布局与 PQ 引导） |
| 13 | 引擎构造与资源分配（下） | `persistent_engine/construction.cc`（后半：QP 装配与 kernel 启动） |
| 14 | 查询执行、路由与完成处理 | `query_execution.cc` `routing.cc` `completion.cc` |
| 15 | 增量发布 | `delta_publication.cc` |
| 16 | 存储回收 RCU | `storage_reclaim.cc` |

### Part IV · 持久化 CUDA Kernel
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 17 | Kernel 启动器、上下文与 device ring | `persistent_kernel.cu` `persistent_kernel.hh` `persistent_kernel/context.cuh` `device_ring.cuh` `mapped_ring.hh` |
| 18 | 候选评分（PQ ADC） | `persistent_kernel/candidate_scoring.cuh` |
| 19 | RDMA cache 与请求合并 | `persistent_kernel/rdma_cache.cuh` |
| 20 | 查询遍历主循环 | `persistent_kernel/query_traversal.cuh` |
| 21 | Kernel 运行时与角色调度 | `persistent_kernel/runtime.cuh` |

### Part V · DOCA GPUNetIO
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 22 | GPUNetIO 传输与 probe | `src/gpu/gpunetio_transport.cc` `gpunetio_probe.cu` `gpunetio_probe.hh` |

### Part VI · 存储节点
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 23 | 存储节点主体与 peer RDMA | `memory_node.cc/hh` `peer_rdma.cc` `startup_protocol.hh` `storage_owner_state.hh` `storage_owner_cpu_plan.hh` `storage_reclaim.hh` |
| 24 | 存储侧 peer RPC | `peer_rpc/runtime.cc` `request_handlers.cc` `workers.cc` `client_requests.cc` `async_response.hh` |
| 25 | 存储侧索引访问与图修改 | `storage_owner_index/allocation.cc` `graph_access.cc` `candidate_search.cc` `graph_mutation.cc` `reverse_batch.cc` `robust_prune_policy.hh` `two_stage_insert_oracle.hh` `partition_local_search.hh` `detail.hh` |
| 26 | 存储侧维护与 runtime wire protocol | `storage_owner_maintenance/queue.cc` `worker.cc` `runtime.cc` `stage2_tracker.hh` `reverse_outbox.hh` `graph_tasks.cc` `cleanup_policy.hh` `detail.hh`；`storage_owner_runtime/lifecycle.cc` `workers.cc` `batch_execution.cc` `wire_protocol.cc` |

### Part VII · 计算服务
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 27 | 计算服务主体 | `compute_service.hh` `lifecycle.cc` `search.cc` `index_commands.cc` `detail.hh`；`src/service/breakdown/*` `breakdown.hh` |
| 28 | 计算侧 storage owner 更新 | `compute_service/storage_owner/lifecycle.cc` `sender.cc` `completion.cc` `public_mutations.cc` `response_validation.hh` |

### Part VIII · 离线工具与实验
| 课 | 主题 | 涉及文件 |
|---|---|---|
| 29 | 离线索引构建与迁移 | `tools/vamana_offline/*` `vamana_offline_builder.cc` `vamana_pq_indexer.cc` `vamana_anchor_sidecar_builder.cc` `tools/legacy_index/migrator.cc` `tools/gpunetio_probe.cc` `gpunetio_loopback_probe.cc` |
| 30 | Breakdown benchmark 与实验脚本 | `tools/breakdown_benchmark/*` `dvstor_breakdown_benchmark.cc` `dvstor_sift101m_long_insert_recall.cc` `generate_sift101m_recall_data.cc` `experiment/*.sh` `experiment/profiles/*.env` |

## 约定

- 叙述用简体中文，代码与标识符保留英文。
- 行号引用格式：`src/gpu_search/types.hh:15`。
- 代码片段前会标明出处；讲解紧贴片段。
- 本教程随 `dev` 分支代码生成；若代码演进，以源码为准。
