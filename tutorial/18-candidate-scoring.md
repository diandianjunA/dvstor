# 第 18 课：候选评分（PQ ADC）

## 本课目标与涉及文件

第 17 课我们讲解了 kernel 启动器、`PersistentKernelParams` 这块庞大的“设备侧上下文”，以及 query CTA / control CTA / dispatcher 的角色分工。从本课起我们进入持久化 kernel 的“计算内核”本身。

这一课聚焦**候选评分（candidate scoring）**这一 GPU 上最热的路径：给定一个查询向量 `query` 和一组候选节点 handle，需要为每个候选算出一个近似距离，以便 beam search 决定保留谁、淘汰谁。dvstor 在这里采用了经典的 **PQ ADC（Asymmetric Distance Computation，非对称距离计算）**：

- **非对称**：查询保持原始 float 表示，候选只用 8-bit PQ code 表示。
- **LUT（Look-Up Table）**：在查询进入 CTA 时，先把 query 投影到 PQ 子空间，与每个子量化的 256 个中心点算平方距离，得到一张 `[pq_subquantizers][256]` 的表；评分时只做 `distance += lut[sub][code[sub]]` 这种表查表累加。
- **OPQ 旋转**：若 PQ 模型带 OPQ 旋转矩阵（`Model::rotation`，见第 9 课），query 在算 LUT 之前先做一次 `R·query`。

本课完整读取的文件：

- `src/gpu_search/persistent_kernel/candidate_scoring.cuh`（742 行，本课主线）
- 辅以 `src/gpu_search/persistent_kernel/context.cuh`（编译期常量与 cub 排序类型别名）
- `src/gpu_search/persistent_kernel/query_traversal.cuh`（LUT 构建、anchor/delta/dynamic-route 调用点）
- `src/gpu_search/pq_index.hh` / `pq_index.cc`（host 侧 PQ 模型与参考 LUT 实现）
- `src/gpu_search/types.hh`、`src/gpu_search/persistent_kernel.hh`（结构体定义）

涉及的关键函数（按出现顺序）：

| 函数 | 位置 | 作用 |
|---|---|---|
| `query_component` | `candidate_scoring.cuh:129` | dtype 分支读 query 元素 |
| `approximate_entry` | `candidate_scoring.cuh:431` | 单个 PQ code 的 ADC 累加 |
| `approximate_handle` | `candidate_scoring.cuh:443` | 给 handle 加一层静态/delta/resident 分支 |
| `beam_insert` | `candidate_scoring.cuh:479` | 把候选插入 beam（最朴素版） |
| `candidate_less` / `candidate_sort_capacity` / `sort_candidates` | `candidate_scoring.cuh:501–553` | 自定义位序网络排序 |
| `merge_approximate_radix` | `candidate_scoring.cuh:555` | 用 cub::BlockRadixSort 合并候选与 beam |
| `merge_approximate_compact` | `candidate_scoring.cuh:610` | 两遍 radix + 终局合并的紧凑版本 |
| `merge_approximate_into_beam` | `candidate_scoring.cuh:705` | 上面两个的总分发器 |

LUT 构建本身在 `query_traversal.cuh:317–333`（见第 20 课），本课为了把“LUT → 评分”闭环讲清楚，会顺带引用。delta 扫描、anchor 评分、dynamic-route 评分也都在 `query_traversal.cuh` 里调用本课的 `approximate_entry` / `approximate_handle`，因此本课会一并讲解这些分支，但调用点的完整流程留给第 20 课。

---

## 文件全景：`candidate_scoring.cuh` 的分层

`candidate_scoring.cuh` 内部其实分成三个相对独立的层，理解这一点对读这份代码至关重要：

1. **基础设施层（1–208 行）**：workspace union、`unlink_mutable_delta`、各种图缓存状态常量、哈希函数、`load_cg`（cg = cache-global 一致性加载）、`global_time_ns`、`query_component`、GPUNetIO 的 `poll_direct_cq` / `lock_direct_qp`（这部分由第 22 课讲）、`insert_visited`、`shard_for_ordinal`、`static_handle_from_raw` / `delta_slot_from_raw` / `resident_pq_slot_from_raw` 等 handle 解析工具。这一层大部分是被其它模块复用的“公共子程序”，本课只挑与评分紧密相关的讲。
2. **handle 与可见性层（234–449 行）**：`handle_from_raw` / `resolve_handle` 把 u32 handle 解析成远端 raw 地址 + shard + 图偏移；`base_overridden` / `delta_visible` / `delta_code_visible` 判断一个静态节点是否被 delta 覆盖、一条 delta 记录在 snapshot_epoch 下是否可见；最后是本课的主角 `approximate_entry` / `approximate_handle`。
3. **beam 合并层（479–740 行）**：`beam_insert` 的朴素插入、`sort_candidates` 的自写位序网络、`merge_approximate_*` 的 cub radix 版合并。这一层负责把刚算好的一批候选距离并入当前 beam。

下面按这三层依次深入。

---

## 基础设施层：评分要用到的底层原语

### `load_cg`：全局可见的“cached load”

```cpp
// candidate_scoring.cuh:80
__device__ __forceinline__ u64 load_cg(const u64* address) {
  u64 value = 0;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ u32 load_cg(const u32* address) {
  u32 value = 0;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}
```

`ld.global.cg` 是 PTX 中的 "cache global" 加载语义：结果可被 L1 缓存，但在 **device 范围内**保证对其它线程的松散/释放写可见。dvstor 的 query CTA 与 control CTA / dispatcher CTA 是不同的 block，它们之间的同步不能靠 `__syncthreads()`（那是 block 内的），而要用 `ld.global.cg` + `st.global.cg` + `__threadfence()` 组合出 device-scope 的可见性。这里的两个重载覆盖了 u32 / u64 两种最常见状态字（`delta_count`、`delta_bucket_heads[]`、`delta_remote_keys[]`、`resident_pq_keys[]` 等）。

在评分路径里，`approximate_handle` 会通过 `load_cg(params.delta_count)` 读 delta 高水位、`load_cg(params.resident_pq_keys + position)` 读 resident PQ 哈希表，全部走 `cg` 语义，这是为了能及时看到 control CTA 发布的新记录，又不至于像 `ld.global.cv`（cache volatile）那样强制打穿 L1。

### `query_component`：dtype 0/1/2 的查询向量读取

```cpp
// candidate_scoring.cuh:129
__device__ __forceinline__ f32 query_component(const u8* query, u8 dtype,
                                                u32 index) {
  switch (dtype) {
    case 0:
      return reinterpret_cast<const f32*>(query)[index];
    case 1:
      return static_cast<f32>(query[index]);
    case 2:
      return static_cast<f32>(reinterpret_cast<const std::int8_t*>(query)[index]);
    default:
      return 0.0f;
  }
}
```

查询向量的 dtype 与索引/存储的 dtype 一致（见第 9 课）：

- `dtype == 0`：float32，每元素 4 字节，`reinterpret_cast` 后直接索引。
- `dtype == 1`：uint8，1 字节，转 float。
- `dtype == 2`：int8，1 字节有符号，转 float。

这个函数在 `query_traversal.cuh:299` 被 `process_query` 用来把 `descriptor.query_device_address` 指向的原始字节解码为 `params.decoded_queries[query_slot]` 这块 float 缓冲。它在评分路径里**不直接**出现，但所有 LUT 构建都来自这块解码后的 float 查询，因此是评分链路的真正入口。

### `shard_for_ordinal` / `static_handle_from_raw` / `handle_from_raw` / `resolve_handle`

这几个函数解决了“怎么把一个 u32 handle 转成实际能读到 PQ code 的指针”这一根本问题。完整细节属于第 10 课（delta）和第 17 课（context），这里只讲评分路径需要的那部分：

```cpp
// candidate_scoring.cuh:343
__device__ u32 handle_from_raw(const PersistentKernelParams& params, u64 raw) {
  u32 handle = UINT32_MAX;
  if (static_handle_from_raw(params, raw, handle)) return handle;
  // ... 否则解析成 dynamic delta handle:
  //   kDeltaHandleBit | (shard << slot_bits) | slot
}
```

handle 的最高位 `kDeltaHandleBit = 0x80000000u` 区分两类：

- **最高位 = 0**：静态节点，handle 就是 ordinal，`approximate_handle` 直接走 `params.pq_codes + handle * pq_code_bytes`。
- **最高位 = 1**：动态 delta 节点，低 31 位再拆成 `shard | slot`，需要走 `delta_slot_from_raw` 或 `resident_pq_slot_from_raw` 找到常驻的 PQ code。

`resolve_handle`（`candidate_scoring.cuh:363`）是反操作：handle → `(raw, shard, graph_offset)`，其中 `raw` 是 64 位远端地址（高 16 位 shard、低 48 位偏移），`graph_offset` 是该节点在远端 graph 文件里的偏移。这个反操作在 `approximate_handle` 里用来取回 raw，再去 delta 表里查 slot。

### `delta_visible` / `delta_code_visible`：snapshot 一致性

```cpp
// candidate_scoring.cuh:416
__device__ bool delta_visible(const DeviceDeltaRecord& record, u64 snapshot_epoch) {
  const u64 superseded = load_cg(&record.superseded_epoch);
  return record.epoch <= snapshot_epoch &&
    (superseded == 0 || superseded > snapshot_epoch) &&
    (record.flags & (kDeltaDeleted | kDeltaDurable)) == 0;
}

__device__ bool delta_code_visible(const DeviceDeltaRecord& record,
                                   u64 snapshot_epoch) {
  const u64 superseded = load_cg(&record.superseded_epoch);
  return record.epoch <= snapshot_epoch &&
    (superseded == 0 || superseded > snapshot_epoch) &&
    (record.flags & kDeltaDeleted) == 0;
}
```

这两个函数是评分路径的**一致性闸门**。`snapshot_epoch` 来自 `QueryDescriptor`，是查询提交时记录的存储 epoch（见第 14 课）。一条 delta 记录对查询可见的充要条件：

1. `record.epoch <= snapshot_epoch`：记录的发布 epoch 不晚于查询快照。
2. `superseded == 0 || superseded > snapshot_epoch`：没有被另一条更新在同一快照前取代。
3. `flags` 不能带 `kDeltaDeleted`（已删除）；`delta_visible` 还额外排除 `kDeltaDurable`（已落盘归档，转入静态层，见第 15 课）。

**两者的区别**只在 `kDeltaDurable`：

- `delta_visible`：要求记录**仍是 mutable delta**，durable 的记录算不可见（要回退到静态层去查）。
- `delta_code_visible`：只要没被删，PQ code 就可以读——因为 durable 记录的 PQ code 仍是有效的近似，只是语义上它已经迁移到静态层了。

`approximate_handle`（下面会讲）走的就是 `delta_code_visible`，它只需要“这条 PQ code 能不能代表这个节点”这一条信息，而不在乎节点是不是已经 durable。

---

## 主角登场：PQ ADC 评分

### `approximate_entry`：单条 PQ code 的 ADC 累加

```cpp
// candidate_scoring.cuh:431
__device__ f32 approximate_entry(const PersistentKernelParams& params,
                                 const f32* query_lut,
                                 const u8* code) {
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < params.pq_subquantizers;
       ++subquantizer) {
    distance += query_lut[static_cast<size_t>(subquantizer) * 256 +
                          code[subquantizer]];
  }
  return distance;
}
```

这是整个评分体系最核心的 6 行。逐行解读：

- `params.pq_subquantizers`：子量化器个数 M，运行期由 `PersistentKernelParams` 传入（`persistent_kernel.hh:97`），编译期上限 `kPersistentMaxSubquantizers = 32`（`persistent_kernel.hh:15`）。典型值 16，因此循环 16 次。
- `code`：指向 `pq_code_bytes = subquantizers` 字节的 PQ code，每字节一个 0–255 的子空间中心编号。
- `query_lut`：查询侧预构建的距离表，布局是 `[subquantizer][centroid]`，共 `subquantizers * 256` 个 float。`static_cast<size_t>(subquantizer) * 256 + code[subquantizer]` 这个下标就是把二维表拍平。
- 累加结果就是 ADC 距离：`D(q, x) ≈ Σ_m ||q_m - c_{m, code[m]}||²`。

**为什么不展开循环？** M 在编译期未知（`params.pq_subquantizers` 是运行期 u32），但 nvcc 通常能把这种小常整数循环自动展开。代码刻意写 `subquantizer * 256` 而不是 `query_lut += 256` 这种步进，是为了在 `code[subquantizer]` 的索引下保持可读性，并且让编译器看清 stride=256 这一不变量。

**内存访问模式**：

- `query_lut` 在整个 block 内对所有候选共享，且只读——它会被 L1 缓存命中得非常好。
- `code` 是连续 `pq_code_bytes` 字节，单条 code 评分时是 16 次 1 字节加载，但批量评分时（见下文 `approximate_handles_batch`）多个线程会按 `index` 并行处理不同 code，相邻线程读相邻 code，合并成合并的全局访问。

**与 host 侧参考实现的对照**：`pq_index.cc:226` 的 `asymmetric_distance` 是 host 侧同构实现：

```cpp
// pq_index.cc:226
f32 asymmetric_distance(const Model& model, std::span<const f32> table,
                        std::span<const u8> code) {
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    distance += table[static_cast<size_t>(subquantizer) *
      kCentroidsPerSubquantizer + code[subquantizer]];
  }
  return distance;
}
```

注意 `kCentroidsPerSubquantizer = 256`（`pq_index.hh:16`），与 device 侧的硬编码 `256` 对应。device 侧不直接用常量名，是为了避免在 `__device__` 代码里依赖 host 头文件的符号；两处一致是约定。

### `approximate_handle`：handle 到 code 的三路分发

```cpp
// candidate_scoring.cuh:443
__device__ f32 approximate_handle(const PersistentKernelParams& params,
                                  const f32* query_lut,
                                  u32 handle, u64 snapshot_epoch) {
  if ((handle & kDeltaHandleBit) == 0) {
    if (handle >= params.num_nodes) return FLT_MAX;
    return approximate_entry(params, query_lut,
      params.pq_codes + static_cast<size_t>(handle) * params.pq_code_bytes);
  }
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return FLT_MAX;
  const u32 slot = delta_slot_from_raw(params, raw);
  if (slot < min(load_cg(params.delta_count), params.delta_capacity) &&
      params.delta_records[slot].remote_node == raw) {
    const DeviceDeltaRecord& record = params.delta_records[slot];
    if (delta_code_visible(record, snapshot_epoch)) {
      return approximate_entry(params, query_lut,
        params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
    }
    const u64 superseded = load_cg(&record.superseded_epoch);
    if (record.epoch <= snapshot_epoch &&
        ((record.flags & kDeltaDeleted) != 0 ||
         (superseded != 0 && superseded <= snapshot_epoch))) {
      return FLT_MAX;
    }
  }
  const u32 resident_slot = resident_pq_slot_from_raw(params, raw);
  if (resident_slot == UINT32_MAX || params.resident_pq_codes == nullptr) {
    return FLT_MAX;
  }
  return approximate_entry(params, query_lut,
    params.resident_pq_codes +
      static_cast<size_t>(resident_slot) * params.pq_code_bytes);
}
```

这是**单点评分**的完整决策树，理解它就理解了 dvstor “同一份 handle 对应多种 PQ code 源”的设计。逐段拆：

**第 1 段：静态节点（最高位 0）**

```cpp
if ((handle & kDeltaHandleBit) == 0) {
  if (handle >= params.num_nodes) return FLT_MAX;
  return approximate_entry(params, query_lut,
    params.pq_codes + static_cast<size_t>(handle) * params.pq_code_bytes);
}
```

- `kDeltaHandleBit = 0x80000000u`（`persistent_kernel.hh:25`）。
- 静态节点 handle 就是 ordinal，直接索引 `params.pq_codes`，这是构建时一次性写入的、与 base 索引同生命周期的 PQ code 数组（第 9 课）。
- 越界返回 `FLT_MAX`，这是评分体系的“哨兵距离”——后面 `beam_insert` 会据此丢弃候选。

**第 2 段：解析 dynamic handle → raw**

```cpp
u64 raw = 0;
u64 graph_offset = 0;
u32 shard = 0;
if (!resolve_handle(params, handle, raw, shard, graph_offset)) return FLT_MAX;
```

`resolve_handle`（第 363 行）对 dynamic handle 拆出 `shard`、远端节点偏移 `raw`、图偏移 `graph_offset`。注意 `raw` 不仅是地址，还是 delta 表的 key——delta 与 resident PQ 表都以 `raw` 作为查找键。

**第 3 段：mutable delta 命中**

```cpp
const u32 slot = delta_slot_from_raw(params, raw);
if (slot < min(load_cg(params.delta_count), params.delta_capacity) &&
    params.delta_records[slot].remote_node == raw) {
  const DeviceDeltaRecord& record = params.delta_records[slot];
  if (delta_code_visible(record, snapshot_epoch)) {
    return approximate_entry(params, query_lut,
      params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
  }
  const u64 superseded = load_cg(&record.superseded_epoch);
  if (record.epoch <= snapshot_epoch &&
      ((record.flags & kDeltaDeleted) != 0 ||
       (superseded != 0 && superseded <= snapshot_epoch))) {
    return FLT_MAX;
  }
}
```

- `delta_slot_from_raw`（`candidate_scoring.cuh:249`）在 `params.delta_remote_keys[]` 这张开放寻址哈希表里查 `raw`，返回 `delta_remote_slots[position]`。表容量 `delta_remote_capacity` 总是 2 的幂，掩码 `mask = capacity - 1`，探测用线性 `position = (position + 1) & mask`，终止条件是遇到 `kDeltaRemoteEmpty = 0`。注意 raw=0 被视为不存在（第 250 行的特判），因为 0 是 empty 哨兵。
- `delta_count` 是已用 slot 的高水位，`min(delta_count, delta_capacity)` 防止高水位越过容量（race 下可能短暂出现）。
- 二次确认 `params.delta_records[slot].remote_node == raw`：哈希表只给出 slot 编号，但 slot 可能已被重用写入新的 raw，必须比对 `record.remote_node` 才算真正命中。这是无锁哈希表的经典 ABA 防护。
- `delta_code_visible` 通过后，直接从 `params.delta_pq_codes`（delta 区的 PQ code 连续数组）取码评分。
- 若不可见，但原因是“被删除或被取代且都在 snapshot 内”，则返回 `FLT_MAX`——这条记录是“曾经存在过、现在没了”的墓碑，应当从候选中剔除。
- 若 `delta_code_visible` 失败但不是墓碑（比如 `epoch > snapshot_epoch`，即记录是查询快照之后才发布的），则**不返回**，继续向下走 resident PQ 回退。这是“未来不可见但可能本来就在 resident 里”的情况。

**第 4 段：resident PQ 回退**

```cpp
const u32 resident_slot = resident_pq_slot_from_raw(params, raw);
if (resident_slot == UINT32_MAX || params.resident_pq_codes == nullptr) {
  return FLT_MAX;
}
return approximate_entry(params, query_lut,
  params.resident_pq_codes +
    static_cast<size_t>(resident_slot) * params.pq_code_bytes);
```

- `resident_pq_slot_from_raw`（第 262 行）结构与 `delta_slot_from_raw` 完全一致，只是查的是 `resident_pq_keys / resident_pq_slots` 这张表。
- “resident PQ” 是 dvstor 为那些**已经被 durable 归档、不再 mutable** 的动态节点保留的常驻 PQ code 缓存（见第 15 课增量发布、第 16 课存储回收）。它让查询在不必远端拉取的情况下仍能对这些节点做近似评分。
- 命中则评分，未命中返回 `FLT_MAX`。

**为什么不在 mutable delta 命中后就直接返回？** 因为 `delta_code_visible` 失败的语义是“这条 PQ code 在当前快照下不能代表这个节点”——可能是记录已被删，也可能是记录是未来的。前者应返回 `FLT_MAX`（已处理），后者要回退到 resident。代码通过 `if (... delta_code_visible ...) return; if (... 墓碑 ...) return FLT_MAX;` 的两段判断精确区分了这两种情况，只有都不是才继续走 resident。

**三路优先级总结**：

```
handle 最高位 == 0     →  params.pq_codes[handle]          (静态)
handle 最高位 == 1:
  ├─ delta_remote 命中且 code_visible  →  params.delta_pq_codes[slot]
  ├─ delta_remote 命中但是墓碑         →  FLT_MAX
  └─ 否则:
       ├─ resident_pq 命中              →  params.resident_pq_codes[slot]
       └─ 未命中                         →  FLT_MAX
```

这条链路是 dvstor “delta / resident / 静态”三层节点存储在评分侧的统一投影。第 10 课在 host/协议层讲过这三层的发布与回收，本课是它们在 GPU 评分代码里的实际消费者。

---

## LUT 构建：把 query 变成 256×M 的距离表

LUT 构建本身写在 `query_traversal.cuh` 里（第 20 课详解），但它是评分的前置步骤，必须在本课交代清楚。相关代码在 `process_query` 的 prepare 阶段：

### OPQ 旋转

```cpp
// query_traversal.cuh:302
f32* transformed = params.transformed_queries +
  static_cast<size_t>(query_slot) * params.dim;
for (u32 row = threadIdx.x; row < params.dim; row += blockDim.x) {
  if (params.opq_matrix == nullptr) {
    transformed[row] = query[row];
    continue;
  }
  f32 value = 0.0f;
  const f32* matrix_row = params.opq_matrix + static_cast<size_t>(row) * params.dim;
  for (u32 column = 0; column < params.dim; ++column) {
    value += matrix_row[column] * query[column];
  }
  transformed[row] = value;
}
```

- `params.opq_matrix` 是 `dim × dim` 的行主序矩阵（host 侧 `pq_model.rotation`，见 `construction.cc:418` 的 `cudaMalloc(OPQ matrix)` 和 `:425` 的 `cudaMemcpy`）。若模型没有旋转（`Model::rotation.empty()`，见 `pq_index.hh:56`），这里就是 `nullptr`，直接拷贝 `query` 到 `transformed`。
- 每个线程负责 `transformed` 的一行（一个输出维度），内层循环对 `query` 全维度做点积。这是标准的矩阵-向量乘法 `y = R·x`，等价于 host 侧 `pq_index.cc:151` 的 `transform`。
- `blockDim.x` 通常是 `kPersistentQueryThreads = 256`（见第 17 课），对 128 维查询来说每线程一行，刚好一轮完成。

**为什么 OPQ 旋转只在 query 侧做？** 因为 OPQ 是对向量空间的正交变换 `R`，满足 `||R x - R y||² = ||x - y||²`。在 PQ 编码时，host 已经对所有 base 向量先做 `R·x` 再划分子空间量化（见 `pq_index.cc:170` 的 `encode`，先 `transform` 再量化）。因此查询侧也要做同样的 `R·q`，子空间距离才对齐。device 侧只需要乘矩阵，不需要再做别的——OPQ 的“优化”已经烤进了 `pq_centroids` 与 `pq_codes` 里。

### LUT 构建：dtype 在这一步已经归一

```cpp
// query_traversal.cuh:317
f32* query_lut = params.query_luts +
  static_cast<size_t>(query_slot) * params.pq_subquantizers * 256;
const u32 table_entries = params.pq_subquantizers * 256;
for (u32 index = threadIdx.x; index < table_entries; index += blockDim.x) {
  const u32 subquantizer = index / 256;
  const f32* query_subvector = transformed +
    static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
  const f32* centroid_subvector = params.pq_centroids +
    static_cast<size_t>(index) * params.pq_subvector_dim;
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
    const f32 difference = query_subvector[dimension] - centroid_subvector[dimension];
    distance += difference * difference;
  }
  query_lut[index] = distance;
}
```

逐行：

- `params.query_luts` 是 host 侧 `construction.cc:513` 分配的 `query_slots * subquantizers * 256` 个 float，每个 query slot 独占 `subquantizers * 256` 项。
- `table_entries = M * 256`，典型 M=16 时是 4096 项。`blockDim.x = 256`，每线程处理 16 项。
- `index` 拆成 `(subquantizer, centroid)`：`subquantizer = index / 256`，`centroid = index % 256`。
- `query_subvector = transformed + subquantizer * subvector_dim`：取旋转后查询的第 `subquantizer` 个子段。
- `centroid_subvector = params.pq_centroids + index * subvector_dim`：注意这里是用 `index`（不是 `centroid`）做步长，因为 `params.pq_centroids` 的布局是 `[subquantizer][centroid][dimension]`，拍平后第 `index` 项的起始位置就是 `index * subvector_dim`（因为 `index = subquantizer * 256 + centroid`）。
- 内层 `dimension` 循环算 `||q_sub - c_sub||²`，与 host 侧 `pq_index.cc:199` 的 `build_distance_table` 完全同构。

**dtype 分支去哪了？** 这一步的输入 `transformed` 已经是 float，dtype 只在 `query_component` 阶段（`candidate_scoring.cuh:129`）起作用。也就是说，PQ ADC 在 LUT 构建后**完全与 dtype 无关**——uint8/int8/float32 的查询向量只在最前面的 decode 步骤区分，之后所有评分都走同一份 float LUT。这是 dvstor 把 dtype 复杂度隔离到 prepare 阶段的关键设计。

**host 侧 `build_distance_table` 的参考实现**（`pq_index.cc:199`）：

```cpp
void build_distance_table(const Model& model, std::span<const f32> input,
                          std::span<f32> table,
                          std::span<f32> transformed_scratch) {
  // ... shape checks ...
  transform(model, input, transformed_scratch);
  const u32 dsub = model.subvector_dim();
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    const f32* value = transformed_scratch.data() + ...;
    const f32* centroids = model.centroids.data() + ...;
    for (u32 centroid = 0; centroid < kCentroidsPerSubquantizer; ++centroid) {
      f32 distance = 0.0f;
      const f32* candidate = centroids + static_cast<size_t>(centroid) * dsub;
      for (u32 dimension = 0; dimension < dsub; ++dimension) {
        const f32 difference = value[dimension] - candidate[dimension];
        distance += difference * difference;
      }
      table[static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer + centroid] =
        distance;
    }
  }
}
```

device 版是它的数据并行版：host 用两层串行循环填表，device 把 4096 项压成一个 `index` 维度并行。但布局完全相同：`table[subquantizer * 256 + centroid]`，这也是 `approximate_entry` 能直接用 `subquantizer * 256 + code[subquantizer]` 索引的原因。

### coalesced 与 bank conflict 分析

LUT 构建的内存布局对 GPU 友好：

- 写 `query_lut[index]`：相邻线程的 `index` 相差 1，写相邻 4 字节，完美 coalesced。
- 读 `params.pq_centroids[index * subvector_dim + ...]`：相邻线程的 `index` 相差 1，读相隔 `subvector_dim * 4` 字节的位置——**这并不 coalesced**！但 `pq_centroids` 是只读的常驻数组，在第一次访问后会被 L2 强缓存，后续命中 L2 即可。若需更优，可以用 `__ldg`（`candidate_scoring.cuh:149` 的 GPUNetIO 路径里就用了 `__ldg`）强制走纹理/只读缓存。

LUT 本身在评分时被反复读：

- `query_lut[subquantizer * 256 + code[subquantizer]]`：对同一 `subquantizer`，相邻候选若 `code` 相近，访问会落在 shared/L1 cache line 上。
- LUT 在 `process_query` 里是 `__shared__` 还是 global？代码里 `params.query_luts` 是 global，每个 query slot 独占一块。同一 block 内所有线程共享 query slot，因此 L1 命中率极高；不进 shared memory 是因为 `M*256*4 = 16KB`（M=16），对 shared memory 来说占比太大，且 L1 已经够用。

`shared memory` 的 bank conflict：LUT 不在 shared 里，所以没有 bank conflict 问题。但下面 `merge_approximate_*` 的 workspace 在 shared 里，那里需要专门考虑（见下文）。

---

## LUT → code 查表 → 距离累加 → beam 更新：数据流图

把 LUT 构建、ADC 评分、beam 合并串起来，整个评分链路的数据流如下：

```
                QueryDescriptor (dtype, dim, query_device_address)
                            │
        ┌───────────────────┴───────────────────┐
        │  query_component (dtype 0/1/2 → f32)  │  candidate_scoring.cuh:129
        │  decoded_queries[query_slot]          │  query_traversal.cuh:298
        └───────────────────┬───────────────────┘
                          f32[dim]
                            │
        ┌───────────────────┴───────────────────┐
        │  OPQ 旋转:  transformed = R · query   │  query_traversal.cuh:302
        │  (opq_matrix == nullptr 时直通)        │
        └───────────────────┬───────────────────┘
                          f32[dim]
                            │
        ┌───────────────────┴───────────────────┐
        │  LUT 构建 (M × 256 项)                │  query_traversal.cuh:317
        │  for index in [0, M*256):              │
        │    sub = index / 256                   │
        │    centroid = index % 256              │
        │    lut[index] = ‖transformed_sub -     │
        │                 pq_centroids[index]‖²  │
        └───────────────────┬───────────────────┘
                          f32[M*256]  (global, per query_slot)
                            │
        ┌───────────────────┴───────────────────┐
        │  approximate_entry / approximate_handle │  candidate_scoring.cuh:431/443
        │  distance = Σ_sub lut[sub*256 + code[sub]] │
        └───────────────────┬───────────────────┘
                          f32  距离
                            │
        ┌───────────────────┴───────────────────┐
        │  merge_approximate_into_beam          │  candidate_scoring.cuh:705
        │  (cub::BlockRadixSort 合并 beam+候选) │
        │  保留前 beam_capacity 个最近候选       │
        └───────────────────┬───────────────────┘
                          更新后的 beam
```

### Shared memory 布局

评分链路里 shared memory 主要被三块占用：

```
__shared__ u32 shared_beam_handles[kPersistentMaxBeam];      // 128 * 4 = 512 B
__shared__ u32 shared_beam_ids[kPersistentMaxBeam];          // 128 * 4 = 512 B
__shared__ f32 shared_beam_distances[kPersistentMaxBeam];    // 128 * 4 = 512 B
__shared__ u8  shared_beam_expanded[kPersistentMaxBeam];     // 128 * 1 = 128 B
__shared__ CandidateWorkspace candidate_workspace;           // 见下
```

`CandidateWorkspace`（`candidate_scoring.cuh:20`）包含：

```cpp
struct CandidateWorkspaceArrays {
  u32 handles[kPersistentMaxExact * 2];     // 256*2 * 4 = 2048 B
  f32 distances[kPersistentMaxExact * 2];   // 256*2 * 4 = 2048 B
  u32 ids[kPersistentMaxExact * 2];         // 2048 B
  u8  expanded[kPersistentMaxExact * 2];    // 512 B
};
union CandidateSortWorkspace {
  ApproximateBlockSortWide::TempStorage radix_sort_wide;            // cub 块排序临时空间
  ApproximateBlockSortCompactPass::TempStorage radix_sort_compact_pass;
  ApproximateBlockSortCompactFinal::TempStorage radix_sort_compact_final;
};
```

- `kPersistentMaxExact = 256`，`kPersistentMaxMergeCandidates = 2048`（`persistent_kernel.hh:14,20`）。
- `CandidateWorkspaceArrays` 约 6.6 KB，是 merge 阶段的“候选+beam 合并缓冲”。
- `CandidateSortWorkspace` 是 union，三种 cub 排序临时空间择一使用——三种排序不会同时发生，union 节省 shared memory。这一点对 occupancy 至关重要：cub::BlockRadixSort 的 TempStorage 可以大到几 KB，三个并列会让 shared memory 溢出。

`kApproximateSortThreadsWide = 256`、`kApproximateSortItemsWide = kPersistentMaxMergeCandidates / 256 = 8`（`candidate_scoring.cuh:25–27`），对应 `ApproximateBlockSortWide = cub::BlockRadixSort<f32, 256, 8, u64>`：256 线程每线程 8 项，总容量 2048，正好 `kPersistentMaxMergeCandidates`。这里的 `u64` 是伴随值（carry-along），低 32 位放 handle、第 32 位放 expanded 标志，详见 `merge_approximate_radix`。

**bank conflict 分析**：`CandidateWorkspaceArrays` 的四个数组都是 4 字节对齐的 u32/f32，相邻线程访问相邻 `index` 时正好落在不同 bank（每 bank 4 字节，32 bank 一组），无冲突。`u8 expanded[]` 是 1 字节数组，访问时相邻线程的 `index` 也相差 1，32 个线程落在 32 个不同的字节但同一组 32 字节内——这会有部分 bank conflict，但 `expanded` 仅在排序搬运时用，不在热路径。

---

## 评分路径的三种调用场景

`approximate_entry` / `approximate_handle` 在 `query_traversal.cuh` 里有三种调用场景，对应 dvstor 候选来源的三个维度：

### 场景 1：静态 anchor 评分

```cpp
// query_traversal.cuh:390
for (u32 anchor = threadIdx.x; anchor < params.anchor_count; anchor += blockDim.x) {
  const u32 handle = params.anchor_handles[anchor];
  const f32 distance = approximate_entry(
    params, query_lut,
    params.anchor_pq_codes + static_cast<size_t>(anchor) * params.pq_code_bytes);
  // ... 每线程保留局部最佳 local_anchors[2] ...
}
```

- `anchor_pq_codes` 是 anchor 集合的预取 PQ code（第 6 课 anchor/idmap），每条 `pq_code_bytes` 字节。
- 每个线程遍历 `anchor_count` 个 anchor，每人维护 `local_anchor_candidates = 2` 个局部最佳（`candidates_per_thread = 2` 当 `blockDim.x == kApproximateSortThreadsCompact = 128`，否则 1）。
- 然后写回 `merge_handles/merge_distances` 做 `sort_candidates` 全局排序，取前 `selected_anchor_count`。
- 注意 anchor 阶段还会做一次 `exact_anchor_distance` 精确重排（`query_traversal.cuh:434`），因为 anchor 数量少（`anchor_count` 通常 ≤ 几百），可以负担得起 L2 距离计算。这是 dvstor “先粗筛后精排”的典型模式。

### 场景 2：dynamic route 槽评分

```cpp
// query_traversal.cuh:482 (调用)
//   score_dynamic_route_slot(params, slot, snapshot_epoch, query_lut, ...)
//
// query_traversal.cuh:83 (实现内部)
const f32 candidate_distance = approximate_entry(
  params, query_lut,
  params.dynamic_route_pq_codes +
    static_cast<size_t>(slot_index) * params.pq_code_bytes);
```

- `dynamic_route_pq_codes` 是 dynamic route overlay 的常驻 PQ code（第 10 课动态路由）。
- `score_dynamic_route_slot`（`query_traversal.cuh:45`）使用 seqseq 风格的 sequence + acquire 重试，保证评分时 PQ code 与元数据属于同一快照。评分后还会再读一次 `sequence` 校验窗口稳定（`query_traversal.cuh:90`），防止“旧 code 配新元数据”的撕裂。
- 评分结果与静态 seed 合并去重，形成初始 beam（`query_traversal.cuh:522–562`）。

### 场景 3：delta 扫描评分

```cpp
// query_traversal.cuh:217 (在 add_delta_candidates 内)
const f32 approximation = approximate_entry(
  params, query_lut,
  params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
```

- `delta_pq_codes` 是 mutable delta 区的 PQ code，每个 slot 一份。
- `add_delta_candidates`（`query_traversal.cuh:99`）扫描选中 anchor 桶里的 delta 链表（或全表扫描高水位），对每条 `delta_visible` 的记录做 `approximate_entry`。
- 每个线程跟踪自己的局部最佳 `local_slot / local_approximation`，最后汇总到 `candidate_slots/handles/distances` 由 thread 0 单线程 `beam_insert`。这是 dvstor 对 delta 的“每线程一票”归约模式，避免对 beam 数组的并发写入。

### 场景 4：邻居批量评分（图遍历主循环）

```cpp
// query_traversal.cuh:732
if (!approximate_handles_batch(params, descriptor, query_lut,
                               navigation_handles,
                               candidate_count,
                               navigation_distances)) { ... }
```

- `approximate_handles_batch`（`rdma_cache.cuh:328`）对一组刚从图里取出来的邻居 handle 做批量评分。
- 它内部走的就是 `approximate_handle` 的三路分发：静态直接评、mutable delta 直接评、resident 直接评、都未命中则**发起 GPUNetIO 远端拉取** `dynamic_code_records`，拉回来后再 `approximate_entry`。
- 这是评分路径里唯一会触发远端 IO 的分支，与第 19 课（RDMA cache）、第 22 课（GPUNetIO）紧密耦合。

---

## Beam 合并：从候选到 beam 的归并排序

评分给出 `(handle, distance)` 列表后，下一步是把它并入当前 beam，保留最近的 `beam_capacity` 个。dvstor 提供了三套实现：

### `beam_insert`：朴素线性插入

```cpp
// candidate_scoring.cuh:479
__device__ void beam_insert(u32* handles, u32* ids, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u32 handle, u32 id, f32 distance) {
  if (handle == UINT32_MAX || !isfinite(distance) || distance == FLT_MAX) return;
  if (count < capacity) {
    handles[count] = handle;
    ids[count] = id;
    distances[count] = distance;
    expanded[count] = 0;
    ++count;
    return;
  }
  u32 worst = 0;
  for (u32 index = 1; index < count; ++index) {
    if (distances[index] > distances[worst]) worst = index;
  }
  if (distance >= distances[worst]) return;
  handles[worst] = handle;
  ids[worst] = id;
  distances[worst] = distance;
  expanded[worst] = 0;
}
```

- 哨兵：`handle == UINT32_MAX`、`!isfinite(distance)`、`distance == FLT_MAX` 都直接丢弃，这是评分侧与 beam 侧的约定。
- 未满时直接 append。
- 已满时线性找最大（worst），比它近才替换。
- 复杂度 O(count)，只适合**单线程小批量**插入。`add_delta_candidates` 里 thread 0 用它把每个线程的局部最佳逐个插入 beam（`query_traversal.cuh:247`），就是这种场景：候选数 ≤ blockDim.x = 256，线性扫可接受。

### `candidate_less` / `sort_candidates`：自写位序网络排序

```cpp
// candidate_scoring.cuh:501
__device__ __forceinline__ bool candidate_less(u32 lhs_handle, f32 lhs_distance,
                                               u32 rhs_handle, f32 rhs_distance) {
  return lhs_distance < rhs_distance ||
    (lhs_distance == rhs_distance && lhs_handle < rhs_handle);
}
```

比较函数：距离优先，距离相等按 handle 升序——这是为了在排序后做去重时确定的顺序，避免两条同距离记录随机排前后。

```cpp
// candidate_scoring.cuh:513
__device__ void sort_candidates(u32* handles, u32* ids, f32* distances,
                                u8* expanded, u32 count) {
  const u32 capacity = candidate_sort_capacity(max(1u, count));
  for (u32 index = count + threadIdx.x; index < capacity; index += blockDim.x) {
    handles[index] = UINT32_MAX;
    if (ids != nullptr) ids[index] = UINT32_MAX;
    distances[index] = FLT_MAX;
    expanded[index] = 0;
  }
  __syncthreads();
  for (u32 sequence = 2; sequence <= capacity; sequence <<= 1) {
    for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (u32 index = threadIdx.x; index < capacity; index += blockDim.x) {
        const u32 partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        const bool exchange = ascending
          ? candidate_less(handles[partner], distances[partner],
                           handles[index], distances[index])
          : candidate_less(handles[index], distances[index],
                           handles[partner], distances[partner]);
        if (!exchange) continue;
        // 交换 handles/ids/distances/expanded[index, partner]
      }
      __syncthreads();
    }
  }
}
```

这是经典的 **bitonic sort（双调排序）** 网络：

- `capacity` 向上取整为 2 的幂（`candidate_sort_capacity`，第 507 行）。
- 先把 `[count, capacity)` 填充哨兵（`UINT32_MAX` / `FLT_MAX`），让数组长度对齐到 2 的幂。
- 外层 `sequence` 控制双调子序列长度（2, 4, 8, ...）。
- 中层 `stride` 是 partner 距离（sequence/2, sequence/4, ..., 1）。
- 内层 `index` 由线程并行处理：每个线程负责一对 `(index, index ^ stride)`，根据 `ascending` 决定交换方向。
- `partner <= index` 的过滤避免重复交换（每对只处理一次）。
- 交换时四个数组同步搬动，保持 `handles[i] / ids[i] / distances[i] / expanded[i]` 一致。

**为什么不用 cub 排序？** 这里 `sort_candidates` 处理的是 shared memory 里的小数组（最多 `kPersistentMaxExact * 2 = 512` 项），且需要保持多个伴随数组的同步。cub 的 `BlockRadixSort` 只能携带一个伴随值（见 `merge_approximate_radix` 的 `u64` 拼接），无法一次带四个数组。自写 bitonic 虽然理论复杂度 O(n log²n) 高于 radix 的 O(n)，但在 n=512 的小规模下，shared memory 内的位序网络没有全局内存往返，实际延迟更低，且代码简洁可控。

### `merge_approximate_radix`：cub 块基数排序合并

当 beam 与候选合并规模较大时（图遍历主循环每轮可能产生上百候选），dvstor 切换到 cub 的 `BlockRadixSort`：

```cpp
// candidate_scoring.cuh:555
template <class BlockSort, u32 ItemsPerThread>
__device__ void merge_approximate_radix(
    u32* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32 existing_count, u32 merge_count,
    typename BlockSort::TempStorage& radix_storage) {
  f32 local_distances[ItemsPerThread];
  u64 local_values[ItemsPerThread];
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u32 handle = UINT32_MAX;
    u32 expanded = 0;
    f32 distance = FLT_MAX;
    if (index < existing_count) {
      handle = beam_handles[index];
      expanded = beam_expanded[index];
      distance = beam_distances[index];
    } else if (index < merge_count) {
      const u32 candidate = index - existing_count;
      handle = candidate_handles[candidate];
      distance = candidate_distances[candidate];
    }
    if (handle == UINT32_MAX || !isfinite(distance)) {
      handle = UINT32_MAX;
      expanded = 0;
      distance = FLT_MAX;
    }
    local_distances[item] = distance;
    local_values[item] = static_cast<u64>(handle) |
      (static_cast<u64>(expanded != 0) << 32);
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(local_distances, local_values);
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = static_cast<u32>(local_values[item]);
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = local_distances[item];
    beam_expanded[output] = static_cast<u8>((local_values[item] >> 32) != 0);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    const u32 limit = min(merge_count, beam_capacity);
    while (valid < limit && beam_handles[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}
```

逐段拆解：

**第 1 段：把 beam + 候选拼成 `[0, merge_count)` 的统一数组**

- `existing_count = beam_count`，`merge_count = existing_count + candidate_count`。
- 线程 `t` 负责的项是 `index = t * ItemsPerThread + item`，对 `ItemsPerThread = 8`、`blockDim.x = 256`，总覆盖 `256 * 8 = 2048 = kPersistentMaxMergeCandidates` 项。
- `index < existing_count`：从当前 beam 取（保留 `expanded` 标志，这是已展开标记，不能丢）。
- `index < merge_count`：从候选数组取（候选没有 `expanded`，置 0）。
- 否则保持哨兵 `UINT32_MAX / FLT_MAX`——cub 排序需要 2 的幂项数，剩余位置用哨兵填充。
- `handle == UINT32_MAX || !isfinite(distance)` 的脏数据统一清成哨兵，避免排序后污染有效段。

**第 2 段：把 handle 与 expanded 打包成 u64**

```cpp
local_values[item] = static_cast<u64>(handle) |
  (static_cast<u64>(expanded != 0) << 32);
```

cub::BlockRadixSort 的 `Sort(keys, values)` 只能携带一个 value。dvstor 把 handle 放低 32 位、expanded 放第 32 位，打包成 u64。排序后从 u64 拆回：

```cpp
beam_handles[output] = static_cast<u32>(local_values[item]);
beam_expanded[output] = static_cast<u8>((local_values[item] >> 32) != 0);
```

这是一个非常常见的 GPU 排序技巧：用位打包把多个标量塞进一个 64 位伴随值。注意 `expanded` 只是 1 位，理论上可以放第 31 位（handle 的最高位是 `kDeltaHandleBit`，但 handle 可能就是 `UINT32_MAX`，所以不能借位）；放第 32 位是安全的。

**第 3 段：cub 块排序**

```cpp
BlockSort(radix_storage).Sort(local_distances, local_values);
```

`BlockSort = ApproximateBlockSortWide = cub::BlockRadixSort<f32, 256, 8, u64>`（`context.cuh:32`）。256 线程 × 8 项/线程 = 2048 项，按 `f32` 键升序排，`u64` 值跟随。cub 内部使用 radix sort（按位桶排），对 float 的排序通过位反转处理负数（cub 内部已处理）。

**第 4 段：写回 beam**

- 排序后 `local_distances[item]` 升序，前 `beam_capacity` 项就是新的 beam。
- `output >= beam_capacity` 的项丢弃——这是 beam 容量约束。
- `beam_ids[output] = UINT32_MAX`：合并阶段不知道 id，后续 `exactify_into_beam` 会重新解析。

**第 5 段：统计有效 beam 数**

thread 0 单线程扫前 `limit = min(merge_count, beam_capacity)` 项，遇到第一个哨兵就停。这一步是 O(beam_capacity) 的串行扫描，但 beam_capacity ≤ 128，可接受。

### `merge_approximate_compact`：两遍 radix 的紧凑版

当 `blockDim.x = kApproximateSortThreadsCompact = 128` 时，单次 radix 只能覆盖 `128 * 8 = 1024` 项，不够 `kPersistentMaxMergeCandidates = 2048`。代码用两遍排序 + 终局合并解决：

```cpp
// candidate_scoring.cuh:610
__device__ void merge_approximate_compact(...) {
  constexpr u32 pass_items =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;  // 128 * 8 = 1024
  for (u32 pass = 0; pass < 2; ++pass) {
    // 第 pass 遍: 处理 [pass*1024, (pass+1)*1024) 这一段
    // 排序后写入 scratch_handles/scratch_distances/scratch_expanded
    // destination = pass * beam_capacity + output
  }
  // 终局: 把两段各 beam_capacity 项合并排序
  //   final 用 kApproximateSortItemsCompactFinal = 2 (128*2 = 256 项)
  //   排序后取前 beam_capacity 写回 beam
}
```

两遍 radix 各排 1024 项，各自保留前 `beam_capacity` 项到 scratch；终局把 2×`beam_capacity` 项做一次 256 项的 radix 排序，取前 `beam_capacity`。这是“分治 + 归并”思路：单次排序容量不够，就分两段排，每段只留 top-K，再合并。

**为什么不用 cub::BlockMergeSort？** MergeSort 的临时空间与 radix 不同，且对 float 的 NaN 处理不如 radix 稳定。dvstor 选择统一用 radix，通过分遍来规避容量问题，代码更一致。

### `merge_approximate_into_beam`：总分发器

```cpp
// candidate_scoring.cuh:705
__device__ void merge_approximate_into_beam(
    u32* candidate_handles, u32* candidate_distances, u32 candidate_count,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32* merge_handles, u32* merge_ids, f32* merge_distances,
    u8* merge_expanded, u32* compact_scratch_handles,
    u32* compact_scratch_expanded, f32* compact_scratch_distances,
    CandidateWorkspace& workspace) {
  const u32 existing_count = beam_count;
  const u32 merge_count = existing_count + candidate_count;
  if (blockDim.x != kApproximateSortThreadsWide &&
      blockDim.x != kApproximateSortThreadsCompact) {
    if (threadIdx.x == 0) beam_count = 0;
    __syncthreads();
    return;
  }
  if (blockDim.x == kApproximateSortThreadsWide) {
    merge_approximate_radix<ApproximateBlockSortWide,
                            kApproximateSortItemsWide>(
      candidate_handles, candidate_distances, candidate_count,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, existing_count, merge_count,
      workspace.sort.radix_sort_wide);
  } else {
    merge_approximate_compact(
      candidate_handles, candidate_distances,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, existing_count, merge_count,
      compact_scratch_handles, compact_scratch_expanded,
      compact_scratch_distances, workspace);
  }
  (void)merge_handles;
  (void)merge_ids;
  (void)merge_distances;
  (void)merge_expanded;
}
```

- **块大小守卫**：`blockDim.x` 必须是 256 或 128，否则清空 beam_count 并返回。这是防止配置错误的硬约束——cub 排序类型在编译期绑定线程数，运行期 `blockDim.x` 不匹配会导致越界。
- **256 线程走 wide 路径**：单遍 radix 覆盖 2048 项，直接合并。
- **128 线程走 compact 路径**：两遍 radix + 终局合并。
- 末尾的 `(void)merge_handles; ...` 是为了消除“未使用参数”警告——这些参数在 `merge_approximate_radix` 里确实没用，但保留接口对称。

**调用点**（`query_traversal.cuh:745`）：

```cpp
merge_approximate_into_beam(
    navigation_handles, navigation_distances,
    candidate_count, beam_handles, beam_ids, beam_distances,
    beam_expanded, beam_count, traversal_capacity,
    merge_handles, merge_ids, merge_distances, merge_expanded,
    rerank_handles, rerank_ids, rerank_distances,
    candidate_workspace);
```

这是图遍历主循环每轮的核心：把刚评分完的 `candidate_count` 个邻居（`navigation_handles/navigation_distances`）并入当前 beam（`shared_beam_*`），保留 `traversal_capacity = min(kPersistentMaxBeam, traversal_beam_width)` 个最近候选。`rerank_handles/distances` 在这里被借作 compact 路径的 scratch buffer。

---

## 与其它模块的关系

- **第 9 课（PQ 模型）**：本课所有 PQ 常量（`kCentroidsPerSubquantizer = 256`、`kBitsPerCode = 8`、`kDefaultSubquantizers = 16`）都来自 `pq_index.hh`。host 侧 `build_distance_table` / `asymmetric_distance` 是 device 侧 `query_lut` 构建 / `approximate_entry` 的参考实现，两者布局完全一致。OPQ 旋转矩阵的 host 端来源（`Model::rotation`）也在第 9 课。
- **第 10 课（delta/动态路由/预算）**：`approximate_handle` 的三路分发（静态 / mutable delta / resident PQ）直接对应第 10 课讲的三层节点存储。`delta_visible` / `delta_code_visible` 是 delta 可见性规则在 device 侧的实现。`score_dynamic_route_slot` 的 seqlock 重试是动态路由一致性的 device 侧投影。
- **第 14 课（查询执行/路由/完成）**：`process_query` 是 query CTA 的主入口，本课的评分是其中的 score 阶段。`CompletionDescriptor.score_cycles` / `beam_cycles` 正是从 `query_traversal.cuh` 的 `score_phase_cycles` / `beam_phase_cycles` 累加而来。
- **第 17 课（kernel 启动器/上下文/device ring）**：`PersistentKernelParams` 的所有评分相关字段（`pq_codes`、`pq_centroids`、`opq_matrix`、`query_luts`、`delta_pq_codes`、`resident_pq_codes`、`dynamic_route_pq_codes`、`anchor_pq_codes`）都在第 17 课的 `construction.cc` 里分配和填充。
- **第 19 课（RDMA cache）**：`approximate_handles_batch` 在 resident/delta/静态三层都未命中时，会触发 GPUNetIO 远端拉取 PQ code。这条路径的细节（`direct_fetch_batch`、`wait_direct_batch`、`dynamic_code_records` 缓存）属于第 19 课。
- **第 20 课（查询遍历主循环）**：本课的 LUT 构建和 anchor/delta/dynamic-route 调用点都在 `query_traversal.cuh` 里，第 20 课会完整串起 `process_query` 的整个 while 循环。本课只聚焦“评分”这一个原语。
- **第 22 课（GPUNetIO 传输/probe）**：`candidate_scoring.cuh:143` 的 `poll_direct_cq` / `lock_direct_qp` 是 GPUNetIO 的设备端verbs，由第 22 课详解。本课只在 `approximate_handles_batch` 触发远端拉取时间接用到。

---

## 小结

本课讲解了 dvstor GPU 评分路径的核心 `candidate_scoring.cuh`，要点如下：

1. **PQ ADC 三步走**：query 解码 → OPQ 旋转 → LUT 构建 → ADC 累加。dtype 复杂度被隔离在 `query_component` 一步，之后所有评分都走同一份 float LUT。
2. **LUT 布局**：`[subquantizer][centroid]` 拍平为 `M * 256` 项 float，与 host 侧 `build_distance_table` 完全一致。构建在 `process_query` prepare 阶段一次性完成，整个 block 共享。
3. **`approximate_entry`**：6 行代码完成 `Σ_sub lut[sub*256 + code[sub]]`，是评分体系的最内层。coalesced 访问、L1 缓存命中。
4. **`approximate_handle`**：handle 到 code 的三路分发——静态 `pq_codes`、mutable delta `delta_pq_codes`、resident `resident_pq_codes`——通过 `delta_code_visible` 与 snapshot_epoch 保证一致性。这是 dvstor “三层节点存储”在评分侧的统一投影。
5. **三种调用场景**：静态 anchor 评分（`anchor_pq_codes`）、dynamic route 评分（`dynamic_route_pq_codes`，带 seqlock 重试）、delta 扫描评分（`delta_pq_codes`，每线程局部最佳归约）。批量邻居评分 `approximate_handles_batch` 还会触发 GPUNetIO 远端拉取。
6. **Beam 合并**：`beam_insert` 适合小批量单线程插入；`sort_candidates` 是自写 bitonic sort，适合多伴随数组的小规模排序；`merge_approximate_radix` / `merge_approximate_compact` 是 cub::BlockRadixSort 的大规模合并，通过 u64 打包 handle+expanded 携带伴随值。`merge_approximate_into_beam` 按 `blockDim.x` 分发到 wide/compact 两路。
7. **shared memory**：`CandidateWorkspace`（约 6.6 KB arrays + union 的 cub TempStorage）是评分路径的主要 shared 占用，union 节省了三种排序临时空间。bank conflict 在 4 字节对齐数组上不存在，`u8 expanded[]` 有轻度冲突但不在热路径。

下一课（第 19 课）将进入 RDMA cache，讲解 `approximate_handles_batch` 触发远端拉取时，PQ code 是如何通过 GPUNetIO 从存储节点拉到 GPU 的，以及 `dynamic_code_records` 这块缓冲是如何被 cache 与 reuse 的。
