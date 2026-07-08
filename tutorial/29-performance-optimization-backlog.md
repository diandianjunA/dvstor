# 第 29 课：性能优化候选点系统梳理

## 本课目标

本课把前 28 课的代码理解转化为性能优化 backlog。每个候选点都要包含：

- 假设。
- 涉及代码。
- 预期收益。
- 风险。
- 验证指标。

代码入口：

- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`
- `src/rdma/vector_batch_planner.hh`
- `src/shared_context.hh`
- `src/gpu/gpu_buffer_manager.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/vamana/rabitq_cache.hh`
- `src/memory_node/storage_owner_runtime.cc`
- `tools/dvstor_breakdown_benchmark.cc`

## 1. 优化方法论

不要先改代码。对这个项目，优化应遵守以下顺序：

1. 选路径：
   - query。
   - insert。
   - mixed。
   - offline build。

2. 建 baseline：
   - 固定数据集。
   - 固定配置。
   - 固定 warmup/measure。
   - 保存 report JSON。

3. 定指标：
   - latency。
   - throughput。
   - recall。
   - RDMA bytes/ops。
   - GPU kernel busy。
   - queue wait。

4. 提假设：
   - 例如“vector RDMA active QPs 太低导致尾延迟高”。

5. 做最小改动。

6. A/B 对比。

7. 如果 recall 或 p99 恶化，回滚或重新设计。

## 2. 查询路径拆分

在线查询路径可以拆成：

1. 请求入队：
   - `ComputeService::search_local_result`
   - `QueryQueue`

2. scheduler：
   - `vamana_service_schedule_queries`

3. 初始化：
   - entry points。
   - medoid read。
   - query H2D。

4. beam loop：
   - CPU select。
   - neighbor fetch。
   - candidate filter。
   - RaBitQ gate。
   - vector RDMA read。
   - GPU distance。
   - D2H。
   - beam update。

5. finalize：
   - sort。
   - read ids。
   - set promise。

每段都对应 breakdown 字段，优化时要先定位是哪段慢。

## 3. 优化项 Q1：vector RDMA batch planner

假设：

- 查询中 `rdma_vector_fetch_ns` 高，且 `vector_rdma_active_qps` 低。
- 当前 planner 没有充分利用 QP pool 或 memory node 并行度。

涉及代码：

- `src/rdma/vector_batch_planner.hh`
- `src/rdma/vamana_rdma_reads.hh`
- `src/shared_context.hh`

预期收益：

- 提高每轮 vector read 并行度。
- 降低 RDMA completion wait。
- 降低 p95/p99 query latency。

风险：

- WR 链太长导致 CQ completion 粒度变差。
- QP0 control lane 保护不足，影响 neighbor read/metadata read。
- outstanding WR 估计不准确导致某些 QP 过载。

验证指标：

- `rdma_vector_fetch_ns`
- `vector_rdma_mean_active_qps_per_batch`
- `vector_rdma_mean_chain_wrs`
- `vector_rdma_credit_wait_ns`
- `vector_rdma_post_send_retries`
- p95/p99 latency
- recall 不变

实验：

- 比较 adaptive/non-adaptive。
- sweep `rdma_read_chain_size`。
- sweep QP pool size。

## 4. 优化项 Q2：credit-aware expansion 参数

假设：

- 当前 expansion batch 过大或过小。
- credit-aware controller 能减少无效 RDMA 或改善尾延迟。

涉及代码：

- `src/vamana/vamana_search.ipp`
- `CreditExpansionController`
- `ComputeService` 中 `set_credit_aware_expansion(...)`

预期收益：

- 减少过量候选 vector read。
- 降低 RDMA stalls。
- 改善查询尾延迟。

风险：

- 过早收缩 expansion 会损伤 recall。
- 参数过多，容易调出偶然结果。
- cost guard 可能导致 no-progress rounds。

验证指标：

- `credit_rounds`
- `credit_mean_issue_k`
- `credit_precommit_ratio`
- `credit_credit_stalls`
- `credit_no_progress_rounds`
- `visited_nodes`
- `exact_reranks`
- recall@k
- p99 latency

实验：

- 固定 beam width，sweep min/max k。
- 开关 cost guard。
- 比较不同 query distribution。

## 5. 优化项 Q3：RaBitQ gate

假设：

- RaBitQ 可以减少 exact vector reads，但 gate width/margin 未调优。

涉及代码：

- `src/vamana/rabitq_cache.hh`
- `src/vamana/vamana_search.ipp`
- `tools/vamana_offline/shard_writer.cc`
- `tools/vamana_rabitq_sidecar_converter.cc`

预期收益：

- 降低 `rdma_vector_fetch_ns`。
- 降低 `vector_rdma_bytes`。
- 提高 query throughput。

风险：

- gate 太窄损伤 recall。
- strict/audit 增加额外 exact read。
- cache miss 或 dynamic overflow 降低收益。

验证指标：

- `rabitq_l0_candidates`
- `rabitq_gate_passes`
- `rabitq_exact_vector_reads`
- `rabitq_cache_misses`
- `rabitq_forced_widen`
- `rabitq_audit_expansions`
- `rdma_vector_fetch_ns`
- recall@k

实验：

- sweep gate width/max width/margin。
- 开关 strict recall。
- 对比 static index 与 mixed update 后的 cache overflow。

## 6. 优化项 Q4：GPU distance kernel batch

假设：

- GPU kernel launch overhead 或 batch size 不合理。
- 查询候选数量小，GPU 利用不足。

涉及代码：

- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/gpu_buffer_manager.hh`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`

预期收益：

- 提高 GPU busy ratio。
- 降低 per-candidate distance cost。

风险：

- 增大 batch 可能增加等待，恶化 tail latency。
- H2D/D2H 成本可能抵消 kernel 收益。
- query batch size 改动影响 scheduler 和 buffer capacity。

验证指标：

- `gpu_query_distance_ns`
- `gpu_kernel_busy_ratio`
- `transfer_query_h2d_ns`
- `transfer_distance_d2h_ns`
- `query_h2d_bytes`
- `query_d2h_bytes`
- p50/p99 latency

实验：

- sweep `query_batch_size`。
- 比较不同 dim/dtype。
- 观察 small k 和 large k。

## 7. 优化项 Q5：D2H 和 beam update

假设：

- distance D2H 后 CPU beam update 成为瓶颈。
- 候选数量大时 sort/insert 成本高。

涉及代码：

- `src/vamana/vamana_search.ipp`
- beam 数据结构。
- visited set。

预期收益：

- 降低 `transfer_distance_d2h_ns`。
- 降低 `cpu_query_beam_update_ns`。
- 降低 CPU runtime overhead。

风险：

- 改 beam 结构可能影响搜索顺序和 recall。
- GPU 上做更多 selection 会增加 kernel 复杂度。

验证指标：

- `cpu_query_beam_update_ns`
- `cpu_query_select_ns`
- `cpu_query_filter_ns`
- `transfer_distance_d2h_ns`
- visited nodes
- recall

实验：

- 比较不同 beam width。
- 记录 candidate count 分布。
- 尝试 top-k partial selection。

## 8. 优化项 Q6：neighbor read 与 hot graph

假设：

- compact hot graph 能减少 neighbor read bytes，但当前动态更新或 fallback 导致收益不足。

涉及代码：

- `src/vamana/vamana_node.hh`
- `src/vamana/storage_layout_resolver.hh`
- `src/rdma/vamana_rdma_reads.hh`
- `tools/vamana_offline/shard_writer.cc`

预期收益：

- 降低 neighbor RDMA bytes。
- 降低 first expansion latency。

风险：

- hot graph encoding/decoding 错误会破坏搜索。
- dynamic hot graph 与 node header generation 一致性复杂。

验证指标：

- `neighbor_rdma_bytes`
- `neighbor_rdma_read_ops`
- `rdma_neighbor_fetch_ns`
- `visited_neighborlists`
- recall

实验：

- AoS vs compact。
- static index vs mixed updates。
- cross-shard ratio 不同的 index。

## 9. 优化项 I1：insert candidate search

假设：

- 在线插入大部分时间花在候选搜索。

涉及代码：

- `src/vamana/vamana_insert.ipp`
- 查询子路径复用。

预期收益：

- 降低 insert service time。
- 提高 write throughput。

风险：

- 降低 construction beam width 会降低图质量，长期影响 query recall。
- candidate 太少会导致 neighbor quality 差。

验证指标：

- `cpu_insert_select_ns`
- `rdma_neighbor_fetch_ns`
- `rdma_vector_fetch_ns`
- `gpu_insert_distance_ns`
- insert latency
- 后续 query recall

实验：

- sweep `beam_width_construction`。
- 插入后跑 recall。
- 比较不同 R。

## 10. 优化项 I2：RobustPrune GPU kernel

假设：

- 插入 prune 或 overflow prune 成为写路径瓶颈。

涉及代码：

- `src/vamana/vamana_insert.ipp`
- `src/gpu/gpu_kernel_launcher.cu`
- GPU prune kernel。

预期收益：

- 降低 `gpu_insert_prune_ns`。
- 降低 overflow prune 成本。

风险：

- prune 结果变化会影响 graph quality。
- kernel 优化可能引入 dtype/dim corner case。

验证指标：

- `gpu_insert_prune_ns`
- `transfer_prune_h2d_ns`
- `transfer_prune_d2h_ns`
- `overflow_prunes`
- `overflow_prune_avg_candidates`
- recall

实验：

- 构造高冲突插入，让 overflow prune 频繁触发。
- 对比 kernel blocks/threads。

## 11. 优化项 I3：neighbor lock 和 reverse update

假设：

- 插入时 CAS lock 和 neighbor list write 导致写尾延迟高。

涉及代码：

- `src/vamana/vamana_insert.ipp`
- `src/rdma/vamana_rdma_atomics.hh`
- `src/rdma/vamana_rdma_writes.hh`

预期收益：

- 降低 insert p99。
- 降低 CAS failures。

风险：

- 图更新并发安全性风险高。
- 锁粒度改变可能引入数据损坏。

验证指标：

- `rdma_neighbor_lock_ns`
- `rdma_neighbor_unlock_ns`
- `lock_attempts`
- `lock_retries`
- `cas_failures`
- insert p99
- consistency check

实验：

- 多线程 mixed 写入。
- 热点 id/upsert workload。
- 对比 lock backoff 策略。

## 12. 优化项 S1：storage-owner batching

假设：

- storage-owner insert 模式下 batch 等待、request prepare 或 response wait 是瓶颈。

涉及代码：

- `src/service/compute_service/storage_owner_insert.ipp`
- `src/memory_node/storage_owner_runtime.cc`
- `src/service/storage_owner_protocol.hh`

预期收益：

- 提高写吞吐。
- 降低 owner RPC 次数。

风险：

- batch 太大会增加单请求延迟。
- timeout/backpressure 复杂。
- response slot 管理容易出错。

验证指标：

- `cpu_storage_owner_sender_queue_wait_ns`
- `cpu_storage_owner_batch_wait_ns`
- `cpu_storage_owner_request_prepare_ns`
- `rdma_storage_owner_send_ns`
- write throughput
- write p99

实验：

- sweep owner batch size。
- probability vs fixed_threads mixed。
- skewed id owner 分布。

## 13. 优化项 S2：peer reverse update

假设：

- owner 与 placement 不一致导致 peer reverse update 成本高。

涉及代码：

- `src/memory_node/peer_rdma.cc`
- `src/memory_node/peer_rpc.cc`
- `src/memory_node/storage_owner_maintenance.cc`
- `src/memory_node/storage_owner_runtime.cc`

预期收益：

- 降低 remote reverse update latency。
- 提高 mixed write throughput。

风险：

- 跨 memory node 一致性复杂。
- credit 限制和 peer QP 调整可能导致死锁或 starvation。

验证指标：

- `cpu_storage_owner_remote_reverse_ns`
- `cpu_storage_owner_peer_reverse_apply_ns`
- peer credit wait。
- write p99。
- query recall after updates。

实验：

- balanced vs BFS/METIS partition。
- owner skew。
- local_stitch on/off。

## 14. 优化项 O1：offline partition

假设：

- BFS/METIS 降低 cross-shard ratio，能降低 runtime query RDMA。

涉及代码：

- `tools/vamana_offline/partitioning.cc`
- `tools/vamana_offline/shard_writer.cc`

预期收益：

- 降低 active nodes per vector batch。
- 降低 cross-node RDMA。
- 改善 query latency。

风险：

- 分区构建时间变长。
- shard size imbalance。
- owner/placement mismatch 加剧写路径 peer update。

验证指标：

- metadata `partition_cross_shard_ratio`
- `vector_rdma_mean_active_nodes_per_batch`
- `rdma_vector_fetch_ns`
- query p99
- write remote reverse ns

实验：

- same dataset，balanced/BFS/METIS 三组 index。
- query-only 和 mixed 都跑。

## 15. 优化项 O2：offline builder CPU 距离

假设：

- offline build 时间主要由 CPU distance 计算决定。

涉及代码：

- `tools/vamana_offline/graph.cc`
- `tools/vamana_offline/dataset_io.hh`

预期收益：

- 降低构建时间。

风险：

- 改距离计算可能改变排序稳定性。
- AVX/并行优化需要处理 dtype。

验证指标：

- build total time。
- medoid time。
- graph build time。
- reverse consolidation time。
- recall。

实验：

- dim sweep。
- thread count sweep。
- dtype sweep。

## 16. 优化优先级建议

建议优先级：

1. 先做观测增强：
   - 补 query-only 并发 benchmark。
   - 补 vector planner 单测。

2. 低风险性能项：
   - RDMA planner 参数。
   - credit-aware 参数。
   - RaBitQ gate 参数。

3. 中风险实现项：
   - request pool。
   - beam update 数据结构。
   - storage-owner batch 参数。

4. 高风险重写：
   - GPU selection。
   - lock-free reverse update。
   - multi-destination routing merge。
   - VamanaNode layout 实例化。

## 17. 学习任务

1. 从一次 benchmark JSON 中选出最大 subcategory，写一个优化假设。
2. 为 Q1-Q6、I1-I3、S1-S2 各补一个可观测指标。
3. 设计一个三天内可完成的低风险优化实验。
4. 设计一个必须先补测试才能做的高风险优化实验。
5. 把本课 backlog 整理成 issue 列表，每个 issue 包含假设、代码、指标、失败判据。

