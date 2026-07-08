# 第 20 课：RaBitQ cache、sidecar 与 gate

## 本课目标

本课讲 RaBitQ 在项目中的实现：节点内 RaBitQ entry、离线 sidecar、runtime cache、query LUT、distance estimate、lower bound、gate selection、warmup/audit/strict recall。学完后，你应该能解释 RaBitQ 如何减少 exact vector RDMA，以及它带来的 recall 风险。

## 代码证据

必须阅读：

- `src/vamana/rabitq_cache.hh`
- `src/vamana/vamana_node.hh`
- `src/vamana/vamana_search.ipp`
- `tools/vamana_offline/shard_writer.cc`
- `tools/vamana_rabitq_sidecar_converter.cc`
- `src/service/compute_service/lifecycle.ipp`

## RaBitQ 启用条件

compute service 构造时：

```text
vamana_->set_use_rabitq(config_.use_rabitq)
```

如果 `use_rabitq` 且 `load_index`：

1. metadata 必须是 rabitq layout。
2. metadata 中 centroid、code bits、entry size 必须匹配 runtime。
3. 加载 `vamana::rabitq::Cache`。
4. 检查 cache size ratio 不超过配置。
5. `vamana_->set_rabitq_cache(rabitq_cache_.get())`。

如果索引不是 RaBitQ layout 但配置 `--use-rabitq`，会直接失败。

## 节点内 RaBitQ entry

`VamanaNode` 中：

- `HAS_RABITQ_CODE`
- `rabitq_centroid`
- `compute_rotated_query`
- `compute_rabitq_code`
- `compute_rabitq_entry`

基本过程：

1. 向量减全局 centroid。
2. 通过 deterministic sign flip。
3. 做 Hadamard-like rotation。
4. 根据 rotated 分量符号生成 binary code。
5. 记录 norm 和 error factor。

节点内 entry 存在固定 node layout 中。

## RFQ5 sidecar cache

`rabitq_cache.hh` 定义 sidecar：

- magic: `RFQ5`
- version: 5
- entry_size
- code_bits
- node_size
- raw_vector_bytes
- entry_count
- cache_budget_bytes
- quantization

sidecar entry 更小，目标是控制在 raw vector bytes 的一定比例内。默认 `choose_entry_bytes` 会按 `kDefaultCacheRatio = 0.10` 选择预算。

## Query LUT

查询时，RaBitQ 不直接对每个 candidate 做完整旋转。`knn_raw` 先：

```text
VamanaNode::compute_rotated_query(query)
rabitq::build_query_lut(rotated_query, cache_code_bits)
```

`QueryLut` 包含：

- `signed_dot`
- `mismatch_energy`
- `code_bits`
- `code_bytes`

每个 byte 有 256 种可能 code，因此 LUT 大小是 `code_bytes * 256`。

## estimate distance

`estimate_distance_lut`：

1. 对 candidate code 每个 byte 查 `signed_dot`。
2. 反量化 norm。
3. 估计 inner product。
4. 返回 L2 distance estimate。

`lower_bound_lut` 使用 mismatch energy 和 norm quantization 给下界，用于更保守的筛选。

## gate selection

查询 expansion 中，如果非 warmup/audit：

```text
rabitq_cache_->estimate_batch_lut(...)
select_gate_into(approximate_distances, cache_miss_indices,
                 gate_width, gate_max_width, margin,
                 selected, cached_order, is_miss)
exact_ptrs = selected candidates
```

gate 的目标：

- cache miss 必须进入 exact。
- 估计距离最好的候选进入 exact。
- margin 内候选可扩展到 max width。
- strict recall 下如果 selected 少于 gate width，会强制 widen。

## warmup 和 audit

RaBitQ 查询还有两类 exact 模式：

- warmup exact：前 `rabitq_warmup_exact_expansions` 次 expansion 不 gate，全部 exact。
- audit exact：每隔 `rabitq_audit_period` expansion，对整批候选 exact。

统计：

- `query_rabitq_forced_widen`
- `query_rabitq_audit_expansions`
- `query_rabitq_audit_candidates`

这些机制用于降低 gate 误杀好候选导致 recall 降低的风险。

## dynamic cache

普通 compute-side insert/upsert 成功后：

- `rabitq_cache_->upsert_dynamic(new_ptr, stored_vector, dtype)`
- 删除旧 ptr 时 `erase_dynamic(old_ptr)`

这说明 sidecar cache 不是纯静态，它还有 dynamic override/slot，用于在线变更。

storage-owner 路径也需要关注 response 中 invalidated neighbors 和 dynamic cache 更新，否则 RaBitQ gate 可能看到过期候选。

## 离线 sidecar 写出

`shard_writer.cc` 中：

1. 如果 `use_rabitq`，计算全局 centroid。
2. 选择 sidecar entry bytes 和 code bits。
3. 扫描所有向量确定 norm min/max。
4. 为每个 shard 创建 rabitq cache file。
5. 写 `SidecarHeader`。
6. 对每个 node 写 cache entry。
7. metadata 写入 rabitq fields。

在线加载时会校验这些字段。

## 查询数据流

```text
query raw bytes
  -> rotated query + norm2
  -> LUT
neighbor expansion candidates
  -> all_unvisited RemotePtr
  -> rabitq cache lookup
  -> estimate distances
  -> gate selected exact_ptrs
  -> exact vector RDMA
  -> GPU exact L2
  -> beam update
```

RaBitQ 只决定哪些候选进入 exact distance。最终写入 beam 的距离仍然是 exact GPU L2。

## 性能影响

收益：

- 减少 full vector RDMA bytes。
- 减少 GPU exact distance candidate 数。
- 减少 H2D candidate bytes。

成本：

- CPU LUT 和 gate 计算。
- sidecar cache memory。
- cache miss 强制 exact。
- warmup/audit 会周期性放大 exact 工作。
- strict recall widen 可能降低过滤收益。

关键指标：

- `query_rabitq_l0_candidates`
- `query_rabitq_cache_misses`
- `query_rabitq_l1_candidates`
- `query_rabitq_l2_candidates`
- `query_exact_reranks`
- vector RDMA bytes
- recall

## 设计异味

1. RaBitQ 逻辑分散在 `VamanaNode`、`rabitq_cache.hh`、`vamana_search.ipp`、offline writer。
2. gate policy 是手写逻辑，缺少独立测试。
3. cache dynamic update 和 idmap/delete 语义耦合。
4. strict/audit/warmup 参数较多，调参空间大。
5. RaBitQ 当前主要是 CPU gate，GPU 和 RDMA pipeline 交织复杂。

## 可验证问题

- RaBitQ gate 是否直接返回最终结果？
- cache miss 为什么必须 exact？
- warmup exact 和 audit exact 的区别是什么？
- sidecar entry size 如何受 raw vector bytes 影响？
- dynamic cache 如何处理 upsert/delete？

## 学习任务

1. 画出 RaBitQ 查询 gate 到 exact rerank 的数据流。
2. 手算一个 sidecar entry 的 code bits 和 entry bytes。
3. 找出所有 `query_rabitq_*` 统计字段的写入位置。
4. 设计实验：固定 recall，调整 gate width 观察 RDMA bytes。
5. 思考：如果 gate 误杀 true nearest neighbor，哪些参数能降低风险？

