# 第 9 课 GPU 引擎类型、遥测与 PQ 模型

本课是 Part III「GPU 搜索引擎」的开篇。从这一课开始，我们正式进入 dvstor 的 GPU 端：前 8 课建立的所有类型、索引格式、传输协议，都要在 GPU 上以一个常驻的持久化 kernel 的形式跑起来。要做到这一点，CPU 和 GPU 之间必须有一组**精确对齐、定长、可静态断言**的数据结构来交换命令、状态和遥测——这就是 `src/gpu_search/types.hh` 与 `src/gpu_search/types.cc` 的全部职责。同时，向量检索引擎的核心近似评分依赖一套 PQ（乘积量化）模型，它在 CPU 端由离线训练器写入磁盘、在引擎启动时被解析并上传到 GPU 常驻显存——这是 `src/gpu_search/pq_index.{hh,cc}` 的职责。

本课先建立 GPU 引擎的「语言」（types），再建立 GPU 引擎的「度量」（telemetry），最后建立 GPU 引擎的「近似评分基底」（PQ）。这三者合起来，为后续第 11 课的持久化引擎生命周期、第 14 课的查询执行、第 18 课的候选评分、第 20 课的查询遍历主循环铺好地基。

## 本课目标与涉及文件

读完本课你应当能够：

1. 逐字段背出 `QueryDescriptor`（40 字节）与 `CompletionDescriptor`（128 字节，有 `static_assert` 兜底）的布局，并说明每个字段在查询生命周期中的写入时机。
2. 理解 delta/动态路由 overlay 所用的一组小定长结构（`DeltaSupersedeUpdate`、`DeltaOverrideUpdate`、`DeltaDurableUpdate`、`ResidentPqEraseUpdate`、`DynamicRouteUpdate`、`DeviceDynamicRouteSlot`），尤其是 `DeviceDynamicRouteSlot` 的 device-scope seqlock 语义。
3. 解释 `kDynamicRouteSlotsPerShard=8`、`kDeltaCommandReset`、`kDeltaCommandPromoteOverrides` 等常量的含义。
4. 看懂 `TelemetrySnapshot` 七十余个字段按「显存占用 / 查询吞吐 / 阶段耗时 / RDMA / 图遍历 / 动态路由 / delta 发布 / resident PQ / mutation 容量 / 可见性」十类的分类法，并理解 `Telemetry::snapshot()` 与 `Telemetry::reset()` 中 acquire/relaxed 的取舍。
5. 理解 PQ 模型文件（`.pq16`/`.pq32`）的二进制格式：`ModelHeader` → 可选 rotation → centroids，FNV-1a 风格校验和，OPQ 旋转矩阵与 codebook 的形状约束。
6. 在脑海里画出「PQ code 常驻 GPU + 查询时构建 256×M LUT」的完整流程图，并能指出 codebook 在 GPU 上的布局（`d_pq_centroids[subquantizer * 256 * dsub + centroid * dsub + dim]`）与 LUT 的布局（`query_luts[query_slot * M * 256 + subquantizer * 256 + centroid]`）。

涉及文件（全部按真实代码逐行讲解）：

- `/home/xjs/experiment/dvstor/src/gpu_search/types.hh`：GPU/CPU 共享的全部 POD 结构与常量。
- `/home/xjs/experiment/dvstor/src/gpu_search/types.cc`：`Telemetry::snapshot()` / `Telemetry::reset()` 的实现。
- `/home/xjs/experiment/dvstor/src/gpu_search/pq_index.hh`：PQ 模型的结构、常量、函数声明。
- `/home/xjs/experiment/dvstor/src/gpu_search/pq_index.cc`：PQ 模型的校验、读写、编码、LUT 构造、ADC 距离计算。

为了讲清 PQ 在 GPU 上的实际使用方式，本课还会引用以下文件中的片段（但不会展开讲，留到对应课程）：

- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/impl.hh`：`pq::Model pq_model;`、`f32* d_pq_centroids{};`、`byte_t* d_pq_codes{};` 等 GPU 指针成员。
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/construction.cc`：`pq::read_model(...)` 上传 codebook、`device_allocate(d_pq_centroids, ...)`、`device_allocate(d_query_luts, ...)`。
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/query_traversal.cuh`：查询时在 device 端构建 256×M LUT 的 kernel。
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/candidate_scoring.cuh`：`approximate_entry` 用 LUT 做近似评分。
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/runtime.cuh`：delta 上线时在 device 端对新增向量做 PQ 编码并写回 `d_delta_pq_codes` / `d_resident_pq_codes`。
- `/home/xjs/experiment/dvstor/src/common/index_path.hh`：`.pq16`/`.pq32` 文件名生成规则。

---

## 1. `types.hh` 总览：POD 优先的设计哲学

`types.hh` 开头先定义一组固定宽度的类型别名，全文件随后只使用这些别名，禁止 `int`/`unsigned`/`float` 裸类型，以确保 CPU/GPU 两端的布局一致：

```cpp
// src/gpu_search/types.hh:8-13
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;
```

整个头文件没有任何虚函数、没有任何 `std::string`、没有任何 `std::vector`——所有结构都是 POD（Plain Old Data），可以直接 `cudaMemcpy` 到 device、可以直接放进 RDMA 报文、可以用 `static_assert(sizeof(...) == N)` 锁死大小。这是 dvstor 在 CPU/GPU/网络三端共享数据的基本纪律。

下面按「查询路径 / delta 发布路径 / 动态路由 / 遥测」四个语义簇逐块讲解。

---

## 2. 查询路径的描述符

### 2.1 `QueryDescriptor`：一次查询的入参

```cpp
// src/gpu_search/types.hh:15-27
struct QueryDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 query_device_address{};
  u64 result_device_address{};
  u32 query_slot{};
  u32 result_capacity{};
  u16 dim{};
  u16 k{};
  u8 query_dtype{};
  u8 flags{};
  u16 reserved{};
};
```

字段布局（按声明顺序，自然对齐）：

| 偏移 | 字段 | 类型 | 字节 | 含义 |
|---:|---|---|---:|---|
| 0  | `request_id`            | u64 | 8 | 全局唯一请求 ID，用于在完成回报里把结果与请求配对。 |
| 8  | `snapshot_epoch`        | u64 | 8 | 查询的快照纪元。delta/动态路由的可见性判定都以此为基准（见第 10 课、第 20 课）。 |
| 16 | `query_device_address`  | u64 | 8 | 查询向量的 **device 虚拟地址**。注意是 device 地址，不是 host 地址——query 向量在提交前就已经被 `cudaMemcpy` 到 GPU。 |
| 24 | `result_device_address` | u64 | 8 | 结果缓冲区的 device 地址。kernel 把 top-k 写到这里。 |
| 32 | `query_slot`            | u32 | 4 | 查询在持久化 kernel 的 per-slot 工作区里的下标（decoded query、transformed query、LUT、beam 都按 slot 索引，见第 17 课、第 20 课）。 |
| 36 | `result_capacity`       | u32 | 4 | 结果缓冲区最多能容纳多少条 `(id, distance)`。 |
| 40 | `dim`                   | u16 | 2 | 向量维度。 |
| 42 | `k`                     | u16 | 2 | top-k。 |
| 44 | `query_dtype`           | u8  | 1 | 查询向量 dtype（fp32/fp16/...），由 `query_component` 在 device 端解释（见第 18 课）。 |
| 45 | `flags`                 | u8  | 1 | 查询标志位。 |
| 46 | `reserved`              | u16 | 2 | 保留，置 0，保证结构对齐到 8 字节边界。 |

总计 48 字节（不是 16 字节——这是一个常见的误读；`QueryDescriptor` 比 16 字节大得多，因为它要装两个 device 地址和若干元数据）。结构自然对齐到 8 字节，可以直接放进 device ring 的一个 slot 里被 kernel 原子读取。

设计要点：

- **两段式地址**：`query_device_address` + `result_device_address` 让 kernel 不需要知道 host 的内存布局，只看 device 指针。这是 GPUNetIO 异步流水线的前提（见第 22 课）。
- **`snapshot_epoch` 是查询的「时间戳」**：所有 delta 记录和动态路由槽的可见性都靠它判定。它使得「查询提交时已发布但尚未可见的写入」不会污染本次查询，实现快照隔离。
- **`query_slot` 解耦提交与执行**：CPU 把 descriptor 写进 slot N，kernel 持续轮询所有 slot；slot 决定了 LUT/beam 等共享显存的哪一块属于这次查询。

### 2.2 `CompletionDescriptor`：一次查询的回报

```cpp
// src/gpu_search/types.hh:29-55
struct CompletionDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 gpu_cycles{};
  u64 prepare_cycles{};
  u64 graph_cycles{};
  u64 score_cycles{};
  u64 beam_cycles{};
  u64 exact_cycles{};
  u64 delta_scan_cycles{};
  u32 query_slot{};
  u32 result_count{};
  i32 status{};
  u32 remote_pages{};
  u32 remote_batches{};
  u32 graph_rounds{};
  u32 exact_vectors{};
  u32 cache_hits{};
  u32 route_hits{};
  u32 exact_cache_hits{};
  u32 delta_scan_records{};
  u32 delta_scan_scored{};
  u32 delta_scan_truncated_buckets{};
  u32 graph_read_retries{};
};

static_assert(sizeof(CompletionDescriptor) == 128);
```

这是本课里最关键的结构之一。`static_assert(sizeof(CompletionDescriptor) == 128);` 把它钉死在 128 字节——这是 device ring 里完成槽的大小，也是 RDMA 完成报文的最大 payload 长度。任何字段的增删都必须同步更新这个断言或者调整保留字段。

逐字段拆解：

**身份与时间（0–71 字节，9 个 u64）**

| 偏移 | 字段 | 含义 |
|---:|---|---|
| 0  | `request_id`      | 与 `QueryDescriptor::request_id` 一致，用于配对。 |
| 8  | `snapshot_epoch`  | 回报本次查询实际使用的快照纪元（通常与请求一致，留作诊断）。 |
| 16 | `gpu_cycles`      | kernel 从开始处理这个 slot 到写完成的总 cycle 数（用 device 端 `clock64()` 测量）。 |
| 24 | `prepare_cycles`  | prepare 阶段：解码 query、OPQ 旋转、构建 256×M LUT 的 cycle 数。 |
| 32 | `graph_cycles`    | 图遍历主循环的 cycle 数（见第 20 课）。 |
| 40 | `score_cycles`    | 候选评分（LUT 查表 + ADC）的 cycle 数（见第 18 课）。 |
| 48 | `beam_cycles`     | beam 维护（插入、截断、去重）的 cycle 数。 |
| 56 | `exact_cycles`    | 精确重排阶段的 cycle 数（见第 18 课）。 |
| 64 | `delta_scan_cycles` | delta 扫描阶段的 cycle 数（见第 10 课）。 |

**结果与状态（72–83 字节）**

| 偏移 | 字段 | 类型 | 含义 |
|---:|---|---|---|
| 72 | `query_slot`     | u32 | 回报这个完成来自哪个 slot，便于 CPU 端回收 slot。 |
| 76 | `result_count`   | u32 | 实际写入 `result_device_address` 的条数（≤ k，可能 < k 若候选不足）。 |
| 80 | `status`         | i32 | 0 表示成功，负值是 `-errno`（例如 `-ENOSPC` 表示 resident PQ 容量不足，见 `runtime.cuh`）。 |
| 84 | `remote_pages`   | u32 | 本次查询触发的 RDMA 远程图页读次数。 |
| 88 | `remote_batches` | u32 | 这些 RDMA 读被合并成多少个 batch（合并率高说明 `rdma_cache` 工作良好，见第 19 课）。 |

**图遍历统计（92–103 字节）**

| 偏移 | 字段 | 含义 |
|---:|---|---|
| 92  | `graph_rounds`                   | 图遍历主循环的轮数（每轮展开 beam 中一个未展开节点）。 |
| 96  | `exact_vectors`                  | 精确重排阶段实际读取的向量数。 |
| 100 | `cache_hits`                     | 图页缓存命中次数（见第 19 课）。 |
| 104 | `route_hits`                     | 动态路由命中次数（命中即跳过静态 anchor 路由，见第 10 课）。 |
| 108 | `exact_cache_hits`               | 精确向量缓存命中次数。 |
| 112 | `delta_scan_records`             | delta 扫描阶段读到的记录总数。 |
| 116 | `delta_scan_scored`              | 其中真正被 PQ 评分的记录数。 |
| 120 | `delta_scan_truncated_buckets`   | delta bucket 被截断的次数（见第 10 课）。 |
| 124 | `graph_read_retries`             | 图页 RDMA 读重试次数（反映网络抖动）。 |

字节布局图：

```
CompletionDescriptor (128 B)
+--------+--------+--------+--------+--------+--------+--------+--------+
| 0      request_id                | 8      snapshot_epoch            |  8
+--------+--------+--------+--------+--------+--------+--------+--------+
| 16     gpu_cycles                | 24     prepare_cycles            | 16
+--------+--------+--------+--------+--------+--------+--------+--------+
| 32     graph_cycles              | 40     score_cycles              | 24
+--------+--------+--------+--------+--------+--------+--------+--------+
| 48     beam_cycles               | 56     exact_cycles              | 32
+--------+--------+--------+--------+--------+--------+--------+--------+
| 64     delta_scan_cycles         | 72     query_slot | 76 result_count | 40
+--------+--------+--------+--------+--------+--------+--------+--------+
| 80     status   | 84     remote_pages            | 88     remote_batches | 48
+--------+--------+--------+--------+--------+--------+--------+--------+
| 92     graph_rounds              | 96     exact_vectors            | 56
+--------+--------+--------+--------+--------+--------+--------+--------+
| 100    cache_hits                 | 104    route_hits               | 64
+--------+--------+--------+--------+--------+--------+--------+--------+
| 108    exact_cache_hits           | 112    delta_scan_records       | 72
+--------+--------+--------+--------+--------+--------+--------+--------+
| 116    delta_scan_scored          | 120    delta_scan_truncated_buckets | 80
+--------+--------+--------+--------+--------+--------+--------+--------+
| 124    graph_read_retries         |                                 pad | 88
+--------+--------+--------+--------+--------+--------+--------+--------+
                                                                                  (128 B total)
```

注意末尾没有显式 padding：`graph_read_retries` 占到偏移 127，结构正好 128 字节。这就是 `static_assert` 能过的原因。

---

## 3. Delta 发布路径的小定长结构

第 15 课会专门讲增量发布协议，本课只介绍它在 `types.hh` 里的数据结构。这些结构都是 delta publication 的 **command payload**——被装在 `DeltaPublishDescriptor` 描述的一次发布里，批量从存储节点传到计算节点。

### 3.1 三种 delta 状态变更

```cpp
// src/gpu_search/types.hh:57-73
struct DeltaSupersedeUpdate {
  u32 slot{};
  u32 reserved{};
  u64 epoch{};
};

struct DeltaOverrideUpdate {
  u32 ordinal{};
  u32 reserved{};
  u64 epoch{};
};

struct DeltaDurableUpdate {
  u32 slot{};
  u32 reserved{};
  u64 epoch{};
};
```

三个结构都是 16 字节、形状几乎一致（`u32 id + u32 reserved + u64 epoch`），但语义不同：

- `DeltaSupersedeUpdate`：某个 delta slot 被一个更新的记录**取代**了，`epoch` 是取代者的纪元。kernel 据此把 `superseded_epoch` 写进 `DeviceDeltaRecord`，使旧记录在 `snapshot_epoch < superseded_epoch` 的查询里仍然可见（快照隔离）。
- `DeltaOverrideUpdate`：某个 base 向量（按 `ordinal` 索引）被 delta 永久覆盖，`epoch` 是覆盖生效纪元。`ordinal` 是 base 层的向量序号，不是 delta slot。
- `DeltaDurableUpdate`：某个 delta slot 已经落盘持久化（`epoch` 是落盘纪元）。此后该记录即使在内存中被回收也仍然算「可见」。

`reserved` 字段不是装饰：它把结构顶到 16 字节，方便 publication payload 里所有 update 类型等长对齐，kernel 可以按 16 字节步长批量拷贝。

### 3.2 Resident PQ 槽回收

```cpp
// src/gpu_search/types.hh:75-79
struct ResidentPqEraseUpdate {
  u64 remote_node{};
  u32 slot{};
  u32 reserved{};
};
```

`remote_node` 是远端存储节点 ID（delta 记录的来源），`slot` 是它在 GPU resident PQ 表里的 slot。当一条 delta 被回收，kernel 需要把 resident PQ 表中对应 slot 标记为空。这个结构就是回收指令的 payload（16 字节）。详见第 16 课（存储回收 RCU）和第 15 课。

### 3.3 动态路由 overlay

dvstor 的查询路由有两层：静态 anchor 是 bootstrap/回退路径；动态路由是 overlay，由存储节点发布「活的代表向量」给计算节点。注释把意图说得很清楚：

```cpp
// src/gpu_search/types.hh:81-86
// The dynamic query-route overlay is deliberately tiny and fixed-capacity.
// Static anchors remain the bootstrap/fallback. Storage owners publish the
// canonical live representatives, so every compute node installs identical
// slot identities even when mutations originate from different clients.
inline constexpr u32 kDynamicRouteSlotsPerShard = 8;
inline constexpr u32 kDynamicRouteLive = 1u;
```

- `kDynamicRouteSlotsPerShard = 8`：每个 shard 最多 8 个动态路由槽。这是「deliberately tiny」——动态路由只覆盖热路径，冷路径仍走静态 anchor。
- `kDynamicRouteLive = 1u`：`flags` 字段里的「该槽位是活的代表」位。`DeviceDynamicRouteSlot::flags` 检查它来判断槽是否有效。

```cpp
// src/gpu_search/types.hh:88-97
struct DynamicRouteUpdate {
  u64 epoch{};
  u64 remote_node{};
  u32 slot{};
  u32 shard{};
  u32 id{};
  u32 generation{};
  u32 flags{};
  u32 reserved{};
};
```

`DynamicRouteUpdate` 是 CPU→GPU 的发布指令（40 字节，由 `static_assert(sizeof(DynamicRouteUpdate) == 40);` 锁死）。字段含义：

- `epoch`：发布纪元，用于新旧版本判定。
- `remote_node`：该代表向量所在的远端节点 ID。
- `slot` / `shard`：目标 shard 内的 slot 下标。
- `id`：代表向量在 base 层的逻辑 ID。
- `generation`：该 slot 的代数，每次替换 +1，便于诊断重复发布。
- `flags`：`kDynamicRouteLive` 等标志。
- `reserved`：对齐。

### 3.4 `DeviceDynamicRouteSlot` 与 seqlock 语义

这是本课最精巧的结构。它是 **device 端**的路由槽，被 control CTA 写、被 query CTA 读，跨 CTA 无锁：

```cpp
// src/gpu_search/types.hh:99-115
// sequence is a device-scope seqlock.  The control CTA is the only writer:
// odd means an update is in progress, even means the remaining fields form a
// stable snapshot.  Query CTAs never wait for a writer; they skip an unstable
// dynamic seed and continue with the static route.
struct DeviceDynamicRouteSlot {
  u64 sequence{};
  u64 command_id{};
  u64 epoch{};
  u64 remote_node{};
  u32 id{};
  u32 generation{};
  u32 shard{};
  u32 flags{};
};

static_assert(sizeof(DynamicRouteUpdate) == 40);
static_assert(sizeof(DeviceDynamicRouteSlot) == 48);
```

`sequence` 是 device-scope seqlock 的版本号：

- **奇数**：control CTA 正在写这个槽，剩余字段处于半更新状态，不可读。
- **偶数**：剩余字段构成一个稳定快照，可读；偶数值同时也是版本号，每次写完 +1。

读端（query CTA）的标准 seqlock 读法：

```
seq1 = load(sequence)
if (seq1 is odd) → 不稳定，跳过这个动态槽，回退到静态 anchor
... 读其它字段 ...
seq2 = load(sequence)
if (seq1 != seq2) → 读期间发生写，放弃，回退到静态 anchor
```

关键设计取舍在注释里点明了：「Query CTAs never wait for a writer; they skip an unstable dynamic seed and continue with the static route.」查询永远不等写者，宁可降级到静态路由也不阻塞。这是 GPU 持久化 kernel 的核心信条——查询路径上不能有等待。

48 字节布局：

```
DeviceDynamicRouteSlot (48 B)
+--------+--------+--------+--------+--------+--------+--------+--------+
| 0      sequence                  | 8      command_id                |
+--------+--------+--------+--------+--------+--------+--------+--------+
| 16     epoch                     | 24     remote_node               |
+--------+--------+--------+--------+--------+--------+--------+--------+
| 32     id        | 36     generation | 40     shard    | 44     flags  |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

`command_id` 把这个槽的当前内容与某次 `DeltaPublishDescriptor::command_id` 关联起来，便于在发布完成回报里确认安装了哪个版本。

---

## 4. Delta 发布命令的描述符与完成回报

```cpp
// src/gpu_search/types.hh:117-118
inline constexpr u32 kDeltaCommandReset = 1u;
inline constexpr u32 kDeltaCommandPromoteOverrides = 1u << 1;
```

这两个常量是 `DeltaPublishDescriptor::flags` 的位：

- `kDeltaCommandReset`：本次发布是 reset（清空所有 delta，重新装载）。
- `kDeltaCommandPromoteOverrides`：本次发布要把 override 提升为 base（永久化）。

```cpp
// src/gpu_search/types.hh:120-132
struct DeltaPublishDescriptor {
  u64 command_id{};
  u32 first_slot{};
  u32 record_count{};
  u32 final_count{};
  u32 invalidation_count{};
  u32 superseded_count{};
  u32 override_count{};
  u32 durable_count{};
  u32 resident_pq_erase_count{};
  u32 dynamic_route_count{};
  u32 flags{};
};
```

这是 CPU 交给 GPU control CTA 的一次「发布命令」的头部。注意它本身只描述**数量**，真正的 payload（`DeltaSupersedeUpdate` 数组、`DynamicRouteUpdate` 数组等）跟在它后面，由 control CTA 按数量解析。字段含义：

- `command_id`：单调递增的命令 ID，用于完成回报配对。
- `first_slot`：本次发布的 delta 记录写入 device 端 delta 表的起始 slot。
- `record_count`：本次发布的 delta 记录数（base payload）。
- `final_count`：发布完成后 device 端 delta 表应有的活记录总数（用于一致性校验）。
- `invalidation_count`：失效条数（被删除的 base 向量）。
- `superseded_count` / `override_count` / `durable_count`：对应三种 update 的数量。
- `resident_pq_erase_count`：要回收的 resident PQ 槽数。
- `dynamic_route_count`：要安装的动态路由槽数。
- `flags`：`kDeltaCommandReset` / `kDeltaCommandPromoteOverrides`。

```cpp
// src/gpu_search/types.hh:134-138
struct DeltaPublishCompletion {
  u64 command_id{};
  i32 status{};
  u32 final_count{};
};
```

完成回报只有 16 字节：`command_id` 配对、`status` 报错（0 或 `-errno`）、`final_count` 是发布完成后 device 端 delta 表的实际活记录数，与 descriptor 里的 `final_count` 比对可以检测不一致。

整条发布流水线是：CPU 端组装 `DeltaPublishDescriptor` + 各种 update payload → 写到 device ring 的 control 槽 → control CTA 解析并安装 → 写 `DeltaPublishCompletion` 到完成 ring → CPU 回收。详见第 15 课。

---

## 5. `TelemetrySnapshot` 与 `Telemetry` 类

### 5.1 `TelemetrySnapshot`：七十多个字段的全景视图

`TelemetrySnapshot` 是一个纯 POD 结构，所有字段都是 `u64`，按语义分十类。它的作用是给上层（compute service、benchmark）提供一个**一致**的遥测快照——所有字段在一次 `snapshot()` 调用里被读取，避免字段之间跨时间不一致。

```cpp
// src/gpu_search/types.hh:140-206
struct TelemetrySnapshot {
  // —— 显存占用（7 个）——
  u64 gpu_memory_explicit_bytes{};        // 显式分配的总字节数（不含共享 buffer）
  u64 gpu_memory_base_pq_bytes{};         // base PQ code 区大小
  u64 gpu_memory_resident_pq_bytes{};     // resident PQ（delta 的 PQ code）区大小
  u64 gpu_memory_route_graph_bytes{};     // 路由图（anchor + 动态路由）大小
  u64 gpu_memory_delta_reserved_bytes{};  // delta 表预留大小
  u64 gpu_memory_graph_cache_bytes{};     // 图页缓存大小
  u64 gpu_memory_exact_cache_bytes{};     // 精确向量缓存大小

  // —— 查询吞吐（4 个）——
  u64 queries_submitted{};     // 提交总数
  u64 queries_completed{};     // 完成总数
  u64 batches{};               // 批次总数
  u64 batch_queries{};         // 批次内查询总数（含批内合并）

  // —— 提交/完成等待（2 个）——
  u64 submission_wait_ns{};    // CPU 等待提交槽的总 ns
  u64 completion_wait_ns{};    // CPU 等待完成的总 ns

  // —— GPU 阶段耗时（7 个）——
  u64 gpu_active_ns{};         // GPU 活跃总时长
  u64 gpu_prepare_ns{};        // prepare 阶段（解码+OPQ+LUT）
  u64 gpu_graph_ns{};          // 图遍历
  u64 gpu_score_ns{};          // 候选评分
  u64 gpu_beam_ns{};           // beam 维护
  u64 gpu_exact_ns{};          // 精确重排
  u64 gpu_delta_scan_ns{};     // delta 扫描

  // —— RDMA（3 个）——
  u64 rdma_read_ops{};         // RDMA 读操作数
  u64 rdma_read_bytes{};       // RDMA 读字节数
  u64 rdma_merged_requests{};  // 被合并的 RDMA 请求数

  // —— 图遍历细节（6 个）——
  u64 direct_path_failures{};          // 直连路径失败数（GPUNetIO 回退）
  u64 graph_page_requests{};           // 图页请求总数
  u64 graph_read_retries{};            // 图页读重试数
  u64 graph_dependency_rounds{};       // 依赖轮数（见第 19 课 rdma_cache）
  u64 graph_page_cache_hits{};         // 图页缓存命中
  u64 graph_route_hits{};              // 路由命中（跳过远端读）
  u64 graph_route_refreshes{};         // 路由刷新次数

  // —— 动态路由（4 个）——
  u64 dynamic_route_publications{};    // 动态路由发布次数
  u64 dynamic_route_slot_updates{};    // 槽更新次数
  u64 dynamic_route_live_slots{};      // 当前活槽数（gauge）
  u64 dynamic_route_snapshot_skips{};  // seqlock 不稳定跳过次数

  // —— 缓存与精确向量（3 个）——
  u64 graph_cache_invalidations{};     // 图页缓存失效次数
  u64 exact_vector_reads{};            // 精确向量读取次数
  u64 exact_vector_cache_hits{};       // 精确向量缓存命中

  // —— Delta 扫描与发布（4 + 5 = 9 个）——
  u64 delta_queries{};                 // 含 delta 扫描的查询数
  u64 delta_scan_records{};            // delta 扫描记录数
  u64 delta_scan_scored{};             // 其中评分数
  u64 delta_scan_truncated_buckets{};  // 截断 bucket 数
  u64 mutations_published{};           // 发布的 mutation 数
  u64 delta_publications{};            // 发布次数
  u64 delta_reclaim_batches{};         // 回收批次
  u64 delta_entries_retired{};         // 退役条数
  u64 storage_reclaim_ack_writes{};    // storage reclaim ack 写次数
  u64 storage_reclaim_ack_sequence{};  // 最新 ack 序列号

  // —— Delta 表存量（4 个 gauge）——
  u64 delta_live_entries{};            // 活记录数
  u64 delta_physical_entries{};        // 物理记录数（含未回收）
  u64 delta_mutable_entries{};         // 可变记录数
  u64 delta_durable_entries{};         // 已持久化记录数

  // —— Resident PQ（4 个）——
  u64 resident_pq_capacity{};          // 容量
  u64 resident_pq_entries{};           // 当前条数
  u64 resident_pq_peak_entries{};      // 峰值条数
  u64 resident_pq_reclaimed{};         // 累计回收条数

  // —— Mutation 容量管理（5 个）——
  u64 mutation_capacity_rejections{};   // 因容量不足被拒的 mutation 数
  u64 mutation_capacity_wait_events{};  // 等待事件数
  u64 mutation_capacity_wait_ns{};      // 等待总 ns
  u64 mutation_capacity_reserved{};     // 当前预留量（gauge）
  u64 mutation_capacity_reserved_max{}; // 预留峰值

  // —— 可见性与发布耗时（6 个）——
  u64 visibility_ns_total{};            // 可见性延迟总和
  u64 visibility_ns_max{};              // 可见性延迟峰值
  u64 publication_queue_ns_total{};     // 发布排队耗时
  u64 publication_prepare_ns_total{};   // 发布 prepare 耗时
  u64 publication_command_ns_total{};   // 发布 command 耗时
};
```

字段分类的记忆口诀：**「显存、吞吐、等待、阶段、RDMA、图遍历、动态路由、缓存、delta、resident PQ、容量、可见性」**。每类对应一个子系统，调试某个子系统时只看对应那一类即可。

注意几个字段是 **gauge**（瞬时值，如 `delta_live_entries`、`resident_pq_entries`、`mutation_capacity_reserved`），其余是 **counter**（累计值）。`reset()` 对这两类采取不同策略，见 5.3。

### 5.2 `Telemetry` 类：原子计数器集合

```cpp
// src/gpu_search/types.hh:208-278
class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();

  std::atomic<u64> gpu_memory_explicit_bytes{0};
  // ... 70+ 个 std::atomic<u64> 成员，与 TelemetrySnapshot 一一对应 ...
};
```

`Telemetry` 类除了 `snapshot()` 和 `reset()` 两个方法，就是一个巨大的 `std::atomic<u64>` 成员集合。所有成员都内联初始化为 0。选择 `std::atomic<u64>` 而非裸 `u64` 的原因：

- **多线程读写**：CPU 端的提交线程、完成线程、发布线程都会写不同的计数器；查询线程会读 gauge。无锁原子是最低开销的同步方式。
- **与 GPU 解耦**：这些计数器全在 CPU 端。GPU 端的统计（如 `gpu_cycles`）是写在 `CompletionDescriptor` 里随完成回报传回的，CPU 端收到后聚合到 `Telemetry`。
- **`std::memory_order_relaxed` 够用**：见下文。

### 5.3 `Telemetry::snapshot()` 的实现

```cpp
// src/gpu_search/types.cc:5-85
TelemetrySnapshot Telemetry::snapshot() const {
  // A zero reservation observed with acquire means every mutation publisher
  // whose reservation contributed to that zero has completed its telemetry
  // updates. release_mutation_capacity() serializes publishers under the
  // delta mutex and publishes the final reservation count with release.
  const u64 reserved = mutation_capacity_reserved.load(
    std::memory_order_acquire);
  return {
    .gpu_memory_explicit_bytes = gpu_memory_explicit_bytes.load(std::memory_order_relaxed),
    // ... 其余字段全部 relaxed ...
    .mutation_capacity_reserved = reserved,
    // ...
  };
}
```

几乎每个字段都用 `std::memory_order_relaxed`，**唯独** `mutation_capacity_reserved` 用 `acquire`。注释解释了原因：

- mutation publisher 在写自己的遥测字段前会先 `reserve_mutation_capacity()` 增加 `mutation_capacity_reserved`，写完后在 `release_mutation_capacity()` 里（持 delta mutex）把 `mutation_capacity_reserved` 用 `release` 序写回最终值。
- `snapshot()` 用 `acquire` 读 `mutation_capacity_reserved`：如果读到 0，意味着所有 publisher 都已经完成了遥测更新——因为 release/acquire 配对保证了 publisher 在增加 reservation 之前的遥测写都对读者可见。
- 换句话说，`mutation_capacity_reserved` 是一个「发布者在线指示器」，它用 acquire/relaxed 的不对称把 publisher 的遥测更新与 snapshot 串联起来，而其余字段只需 relaxed（因为它们要么是单调 counter，要么已经受 mutex 保护）。

这是个非常精细的设计：用单一字段的 acquire/release 替代给所有 70+ 字段都加 seqlock 或 mutex。代价是 snapshot 里其余字段可能比 `reserved` 略旧，但对遥测来说完全可以接受。

返回值用 C++20 designated initializers 一次性构造 `TelemetrySnapshot`，字段顺序与结构声明严格一致——这是为什么 `TelemetrySnapshot` 的字段顺序不能随便调（一调 `snapshot()` 的初始化列表就要跟着改）。

### 5.4 `Telemetry::reset()` 的实现

```cpp
// src/gpu_search/types.cc:87-145
void Telemetry::reset() {
  queries_submitted.store(0, std::memory_order_relaxed);
  // ... 把所有 counter 清 0 ...
  resident_pq_peak_entries.store(
    resident_pq_entries.load(std::memory_order_relaxed),
    std::memory_order_relaxed);
  resident_pq_reclaimed.store(0, std::memory_order_relaxed);
  // ...
  mutation_capacity_reserved_max.store(
    mutation_capacity_reserved.load(std::memory_order_relaxed),
    std::memory_order_relaxed);
  // ...
}
```

`reset()` 的关键细节：

- **counter 清 0**：所有累计型计数器（`queries_submitted`、`rdma_read_ops` 等）直接 `store(0)`。
- **gauge 保留**：`delta_live_entries`、`resident_pq_entries`、`mutation_capacity_reserved` 等 gauge **不清零**——它们是系统当前状态，清零会让快照与现实脱节。注意 `reset()` 函数体里没有这些 gauge 的 `store(0)` 调用。
- **peak 重置为当前值**：`resident_pq_peak_entries` 被重置为当前的 `resident_pq_entries`（而不是 0），`mutation_capacity_reserved_max` 同理。这样 reset 后 peak 仍然有意义——「从现在起的峰值」。
- **不 reset 显存占用字段**：`gpu_memory_*` 字段在 `reset()` 里完全不动，因为它们描述的是已分配显存，与计数器无关。`reset()` 是给 benchmark 用的「清零计数器开始新一轮测量」，不应影响显存状态。

注意 `reset()` 没有清 `mutation_capacity_reserved`（gauge），但 `snapshot()` 用 acquire 读它——这是自洽的：reset 不改变 publisher 的在席状态。

---

## 6. PQ 模型：`pq_index.hh`

PQ（Product Quantization）是 dvstor 近似评分的数学基础。本节先讲模型结构和磁盘格式，下一节讲实现。

### 6.1 常量与文件格式

```cpp
// src/gpu_search/pq_index.hh:11-18
namespace gpu_search::pq {

inline constexpr std::array<char, 8> kModelMagic{'D', 'V', 'P', 'Q', '1', '6', '\0', '\0'};
inline constexpr u32 kModelVersion = 1;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kCentroidsPerSubquantizer = 256;
inline constexpr u32 kBitsPerCode = 8;
inline constexpr u32 kDefaultSubquantizers = 16;
```

- `kModelMagic` = `"DVPQ16\0\0"`：文件头 magic。注意末尾的 `16`——这是文件格式版本号的一部分，不是 subquantizer 数。文件名里的 `.pq16`/`.pq32` 才是 subquantizer 数（见下文）。
- `kModelVersion = 1`：格式版本。
- `kEndianMarker = 0x01020304`：端序标记。读取时检查它是否仍是 `0x01020304`，若变成 `0x04030201` 则说明文件是大端写的，本机是小端，直接拒绝（dvstor 不做端序转换）。
- `kCentroidsPerSubquantizer = 256`：每个子量化器有 256 个聚类中心。这是 8-bit 编码的必然结果（`2^8 = 256`）。
- `kBitsPerCode = 8`：每个子量化器编码 8 bit。
- `kDefaultSubquantizers = 16`：默认 16 个子量化器，即每向量 16 字节 code。但 dvstor 也支持 32（每向量 32 字节，文件名 `.pq32`），最大 32（`kPersistentMaxSubquantizers`，见 `persistent_kernel.hh:15`）。

文件名规则在 `src/common/index_path.hh:43-45`：

```cpp
inline filepath_t navigation_model_file(const filepath_t& prefix, u32 subquantizers) {
  return filepath_t(prefix.string() + ".pq" + std::to_string(subquantizers));
}
```

所以 `.pq32` 表示「subquantizers=32 的 PQ 模型文件」，每向量 32 字节 code。模型文件只装 codebook + 可选 rotation，**不装** code——code 是离线 indexer 编码后单独存的（见 `navigation_code_file`，本课不展开，见第 12 课）。

### 6.2 `ModelHeader`：磁盘头

```cpp
// src/gpu_search/pq_index.hh:20-39
struct ModelHeader {
  std::array<char, 8> magic{kModelMagic};
  u32 version{kModelVersion};
  u32 header_bytes{sizeof(ModelHeader)};
  u32 endian_marker{kEndianMarker};
  u32 dim{};
  u32 subquantizers{};
  u32 bits_per_code{kBitsPerCode};
  u32 subvector_dim{};
  u32 code_bytes{};
  u32 flags{};
  u32 reserved0{};
  u64 rotation_offset{};
  u64 rotation_bytes{};
  u64 centroids_offset{};
  u64 centroids_bytes{};
  u64 file_bytes{};
  u64 payload_checksum{};
  std::array<u64, 4> reserved{};
};
```

这是文件头，所有 offset 都是绝对偏移。字段含义：

- `magic` / `version` / `header_bytes` / `endian_marker`：格式校验四件套。
- `dim`：原始向量维度 D。
- `subquantizers`：子量化器数 M（如 16 或 32）。
- `bits_per_code`：恒为 8（`kBitsPerCode`）。
- `subvector_dim`：每个子量化器负责的子向量维度 `dsub = D / M`。
- `code_bytes`：每向量的 code 字节数 = M（因为每子量化器 8 bit = 1 字节）。
- `flags`：`kFlagHasRotation` 表示有 OPQ 旋转矩阵。
- `rotation_offset` / `rotation_bytes`：旋转矩阵的位置和大小（`D*D*4` 字节，若无则为 0）。
- `centroids_offset` / `centroids_bytes`：codebook 的位置和大小（`M * 256 * dsub * 4` 字节）。
- `file_bytes`：整文件大小，用于截断检测。
- `payload_checksum`：rotation + centroids 的 FNV-1a 风格校验和。
- `reserved[4]`：未来扩展。

### 6.3 `Model` 与 `kFlagHasRotation`

```cpp
// src/gpu_search/pq_index.hh:41
inline constexpr u32 kFlagHasRotation = 1u << 0;

// src/gpu_search/pq_index.hh:43-57
struct Model {
  u32 dim{};
  u32 subquantizers{kDefaultSubquantizers};
  u32 bits_per_code{kBitsPerCode};
  std::vector<f32> rotation;
  std::vector<f32> centroids;

  u32 subvector_dim() const {
    return subquantizers == 0 ? 0 : dim / subquantizers;
  }
  u32 code_bytes() const { return subquantizers; }
  u64 checksum() const;
  bool has_rotation() const { return !rotation.empty(); }
};
```

`Model` 是内存态的模型表示。关键方法：

- `subvector_dim()`：`dim / subquantizers`，要求整除（否则 `validate` 报错）。
- `code_bytes()`：等于 `subquantizers`（每子量化器 1 字节）。
- `has_rotation()`：`rotation` 非空即视为有 OPQ 旋转。
- `checksum()`：见下节。

注意 `Model` 是唯一含 `std::vector` 的结构——它只在 CPU 端存活，从磁盘读上来后立刻被 `cudaMemcpy` 到 GPU（`d_pq_centroids` 和 `d_opq_matrix`），之后 `Model` 对象本身在热路径上不再被引用。详见第 12 课 construction。

### 6.4 函数声明

```cpp
// src/gpu_search/pq_index.hh:59-73
bool validate(const Model& model, std::string* error = nullptr);
bool write_model(const std::filesystem::path& path, const Model& model,
                 std::string* error = nullptr);
bool read_model(const std::filesystem::path& path, Model& model,
                std::string* error = nullptr);

void transform(const Model& model, std::span<const f32> input,
               std::span<f32> output);
void encode(const Model& model, std::span<const f32> input,
            std::span<u8> code, std::span<f32> transformed_scratch);
void build_distance_table(const Model& model, std::span<const f32> input,
                          std::span<f32> table,
                          std::span<f32> transformed_scratch);
f32 asymmetric_distance(const Model& model, std::span<const f32> table,
                        std::span<const u8> code);
```

- `validate`：检查模型形状与数值合法性。
- `write_model` / `read_model`：磁盘 I/O。
- `transform`：应用 OPQ 旋转（或恒等）。
- `encode`：把一个向量编码成 M 字节 code（离线 indexer 用，GPU 上线 delta 时也用同样的逻辑）。
- `build_distance_table`：对一个查询向量构建 256×M 的距离表（LUT）——这是 CPU 参考实现，GPU 端在 `query_traversal.cuh` 里并行化实现。
- `asymmetric_distance`：用 LUT + code 计算非对称距离（ADC）。

---

## 7. PQ 模型：`pq_index.cc` 实现

### 7.1 FNV-1a 风格校验和

```cpp
// src/gpu_search/pq_index.cc:13-28
constexpr u64 kChecksumOffset = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
constexpr u64 kChecksumPrime = 1099511628211ULL;          // FNV-1a 64-bit prime

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

u64 checksum_update(u64 state, const void* data, size_t bytes) {
  const auto* source = static_cast<const u8*>(data);
  for (size_t index = 0; index < bytes; ++index) {
    state ^= source[index];
    state *= kChecksumPrime;
  }
  return state;
}
```

这是标准 FNV-1a 64-bit：`state = (state ^ byte) * prime`。offset basis 和 prime 都是 FNV-1a 的规范常量。

```cpp
// src/gpu_search/pq_index.cc:37-46
u64 Model::checksum() const {
  u64 state = kChecksumOffset;
  if (!rotation.empty()) {
    state = checksum_update(state, rotation.data(), rotation.size() * sizeof(f32));
  }
  if (!centroids.empty()) {
    state = checksum_update(state, centroids.data(), centroids.size() * sizeof(f32));
  }
  return state;
}
```

`checksum()` 把 rotation 和 centroids 拼接做 FNV-1a。注意**不**含 header 字段——这样改 header（如调整 reserved）不会影响 checksum，但改任何 float 都会。这个 checksum 同时被写进 `ModelHeader::payload_checksum` 和持久化索引 layout 的 `model_checksum`（见 `construction.cc:124` 的 `index.layout.model_checksum != pq_model.checksum()` 校验），用于确保 GPU 引擎加载的模型与离线 indexer 用的是同一份。

### 7.2 `validate`：形状与数值校验

```cpp
// src/gpu_search/pq_index.cc:48-71
bool validate(const Model& model, std::string* error) {
  if (model.dim == 0 || model.subquantizers == 0 ||
      model.dim % model.subquantizers != 0) {
    return fail(error, "PQ model dimension must be divisible by its subquantizer count");
  }
  if (model.bits_per_code != kBitsPerCode) {
    return fail(error, "PQ runtime supports exactly 8 bits per subquantizer");
  }
  const size_t expected_rotation = static_cast<size_t>(model.dim) * model.dim;
  if (!model.rotation.empty() && model.rotation.size() != expected_rotation) {
    return fail(error, "PQ model rotation matrix has an invalid shape");
  }
  const size_t expected_centroids = static_cast<size_t>(model.subquantizers) *
    kCentroidsPerSubquantizer * model.subvector_dim();
  if (model.centroids.size() != expected_centroids) {
    return fail(error, "PQ model centroid table has an invalid shape");
  }
  const auto finite = [](f32 value) { return std::isfinite(value); };
  if (!std::all_of(model.rotation.begin(), model.rotation.end(), finite) ||
      !std::all_of(model.centroids.begin(), model.centroids.end(), finite)) {
    return fail(error, "PQ model contains non-finite values");
  }
  return true;
}
```

四道检查：

1. `dim % subquantizers == 0`：子向量必须能整除。
2. `bits_per_code == 8`：runtime 只支持 8-bit。
3. rotation 形状 `D*D`（若存在）。
4. centroids 形状 `M * 256 * dsub`。

最后用 `std::isfinite` 排除 NaN/Inf——这是离线训练可能的副产品，绝不能让它流到 GPU（会污染整个 LUT）。

### 7.3 `write_model`：序列化

```cpp
// src/gpu_search/pq_index.cc:73-101
bool write_model(const std::filesystem::path& path, const Model& model,
                 std::string* error) {
  if (!validate(model, error)) return false;
  ModelHeader header;
  header.dim = model.dim;
  header.subquantizers = model.subquantizers;
  header.bits_per_code = model.bits_per_code;
  header.subvector_dim = model.subvector_dim();
  header.code_bytes = model.code_bytes();
  header.flags = model.has_rotation() ? kFlagHasRotation : 0;
  header.rotation_offset = sizeof(ModelHeader);
  header.rotation_bytes = model.rotation.size() * sizeof(f32);
  header.centroids_offset = header.rotation_offset + header.rotation_bytes;
  header.centroids_bytes = model.centroids.size() * sizeof(f32);
  header.file_bytes = header.centroids_offset + header.centroids_bytes;
  header.payload_checksum = model.checksum();

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.good()) return fail(error, "failed to create PQ model: " + path.string());
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!model.rotation.empty()) {
    output.write(reinterpret_cast<const char*>(model.rotation.data()),
                 static_cast<std::streamsize>(header.rotation_bytes));
  }
  output.write(reinterpret_cast<const char*>(model.centroids.data()),
               static_cast<std::streamsize>(header.centroids_bytes));
  if (!output.good()) return fail(error, "failed to write PQ model: " + path.string());
  return true;
}
```

文件布局：

```
+----------------------+
| ModelHeader          |  rotation_offset = sizeof(ModelHeader)
+----------------------+
| rotation (D*D*4 B)   |  可选；若无则 rotation_bytes=0
+----------------------+
| centroids            |  centroids_offset = rotation_offset + rotation_bytes
| (M*256*dsub*4 B)     |
+----------------------+
| (无 padding)         |
+----------------------+
```

`file_bytes = centroids_offset + centroids_bytes`，紧密排列无 padding。`payload_checksum` 在写 header 前算好。

### 7.4 `read_model`：反序列化与多重校验

```cpp
// src/gpu_search/pq_index.cc:103-149
bool read_model(const std::filesystem::path& path, Model& model,
                std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) return fail(error, "missing PQ model: " + path.string());
  ModelHeader header;
  if (!read_exact(input, &header, sizeof(header)) || header.magic != kModelMagic ||
      header.version != kModelVersion || header.header_bytes != sizeof(ModelHeader) ||
      header.endian_marker != kEndianMarker || header.bits_per_code != kBitsPerCode ||
      header.code_bytes != header.subquantizers || header.subvector_dim == 0 ||
      header.dim != header.subquantizers * header.subvector_dim ||
      header.rotation_offset != sizeof(ModelHeader) ||
      header.centroids_offset != header.rotation_offset + header.rotation_bytes ||
      header.file_bytes != header.centroids_offset + header.centroids_bytes) {
    return fail(error, "invalid PQ model header: " + path.string());
  }
```

header 校验极其严格，一个表达式里串了 12 个条件：

- magic / version / header_bytes / endian_marker：格式四件套。
- `bits_per_code == 8` / `code_bytes == subquantizers` / `subvector_dim != 0` / `dim == subquantizers * subvector_dim`：内部一致性。
- `rotation_offset == sizeof(ModelHeader)`：rotation 必须紧跟 header。
- `centroids_offset == rotation_offset + rotation_bytes`：centroids 必须紧跟 rotation。
- `file_bytes == centroids_offset + centroids_bytes`：文件总长必须等于内容总长。

任何一个不符直接拒载，避免脏文件把后续 GPU 加载搞崩。

```cpp
  const bool has_rotation = (header.flags & kFlagHasRotation) != 0;
  const u64 expected_rotation_bytes = has_rotation
    ? static_cast<u64>(header.dim) * header.dim * sizeof(f32) : 0;
  const u64 expected_centroid_bytes = static_cast<u64>(header.subquantizers) *
    kCentroidsPerSubquantizer * header.subvector_dim * sizeof(f32);
  if (header.rotation_bytes != expected_rotation_bytes ||
      header.centroids_bytes != expected_centroid_bytes) {
    return fail(error, "invalid PQ model payload shape: " + path.string());
  }
  input.seekg(0, std::ios::end);
  if (static_cast<u64>(input.tellg()) != header.file_bytes) {
    return fail(error, "truncated PQ model: " + path.string());
  }
```

第二道校验：根据 `has_rotation` 标志重算预期的 payload 字节数，与 header 声明的对比；再用 `seekg(0, end)` + `tellg()` 比对真实文件大小与 `file_bytes`，防截断。

```cpp
  input.seekg(static_cast<std::streamoff>(header.rotation_offset));
  Model loaded;
  loaded.dim = header.dim;
  loaded.subquantizers = header.subquantizers;
  loaded.bits_per_code = header.bits_per_code;
  loaded.rotation.resize(static_cast<size_t>(header.rotation_bytes / sizeof(f32)));
  loaded.centroids.resize(static_cast<size_t>(header.centroids_bytes / sizeof(f32)));
  if ((!loaded.rotation.empty() &&
       !read_exact(input, loaded.rotation.data(), header.rotation_bytes)) ||
      !read_exact(input, loaded.centroids.data(), header.centroids_bytes)) {
    return fail(error, "failed to read PQ model payload: " + path.string());
  }
  if (loaded.checksum() != header.payload_checksum) {
    return fail(error, "PQ model checksum mismatch: " + path.string());
  }
  if (!validate(loaded, error)) return false;
  model = std::move(loaded);
  return true;
}
```

读到 `loaded` 临时对象，做 checksum 比对（防静默损坏），再跑一遍 `validate`（防 header 撒谎但 checksum 碰巧对的极端情况），全部通过才 `std::move` 给 `model`。这是「先校验后暴露」模式——失败时 `model` 保持原状。

### 7.5 `transform`：OPQ 旋转

```cpp
// src/gpu_search/pq_index.cc:151-168
void transform(const Model& model, std::span<const f32> input,
               std::span<f32> output) {
  if (input.size() != model.dim || output.size() != model.dim) {
    throw std::invalid_argument("PQ transform dimension mismatch");
  }
  if (!model.has_rotation()) {
    std::copy(input.begin(), input.end(), output.begin());
    return;
  }
  for (u32 row = 0; row < model.dim; ++row) {
    f32 value = 0.0f;
    const f32* matrix_row = model.rotation.data() + static_cast<size_t>(row) * model.dim;
    for (u32 column = 0; column < model.dim; ++column) {
      value += matrix_row[column] * input[column];
    }
    output[row] = value;
  }
}
```

若无旋转矩阵，直接拷贝（恒等变换）。否则做矩阵-向量乘 `output = rotation * input`，`rotation` 行主序存储。这是 OPQ（Optimized PQ）的随机旋转预处理——让各子空间的能量均匀分布，提升 PQ 精度。

### 7.6 `encode`：把向量编码成 M 字节

```cpp
// src/gpu_search/pq_index.cc:170-197
void encode(const Model& model, std::span<const f32> input,
            std::span<u8> code, std::span<f32> transformed_scratch) {
  if (code.size() != model.code_bytes() || transformed_scratch.size() != model.dim) {
    throw std::invalid_argument("PQ encode buffer shape mismatch");
  }
  transform(model, input, transformed_scratch);
  const u32 dsub = model.subvector_dim();
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    const f32* value = transformed_scratch.data() + static_cast<size_t>(subquantizer) * dsub;
    const f32* table = model.centroids.data() +
      static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer * dsub;
    f32 best_distance = std::numeric_limits<f32>::max();
    u32 best = 0;
    for (u32 centroid = 0; centroid < kCentroidsPerSubquantizer; ++centroid) {
      f32 distance = 0.0f;
      const f32* candidate = table + static_cast<size_t>(centroid) * dsub;
      for (u32 dimension = 0; dimension < dsub; ++dimension) {
        const f32 difference = value[dimension] - candidate[dimension];
        distance += difference * difference;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best = centroid;
      }
    }
    code[subquantizer] = static_cast<u8>(best);
  }
}
```

流程：

1. `transform` 把输入旋转变换到 `transformed_scratch`。
2. 对每个子量化器 `m`：
   - 取子向量 `value = transformed[m*dsub : (m+1)*dsub]`。
   - 取该子量化器的 256 个聚类中心 `table = centroids[m*256*dsub : ...]`。
   - 找出与 `value` 欧氏距离平方最小的聚类中心 `best`。
   - 写 `code[m] = best`（1 字节）。

最终 `code` 是 M 字节。这就是 base 向量的离线编码（见第 12 课 indexer）和 delta 上线时 device 端编码（见 `runtime.cuh:330-360`）共用的逻辑——前者在 CPU 跑这版，后者在 GPU 跑等价 kernel。

### 7.7 `build_distance_table`：构建 256×M LUT（CPU 参考实现）

```cpp
// src/gpu_search/pq_index.cc:199-224
void build_distance_table(const Model& model, std::span<const f32> input,
                          std::span<f32> table,
                          std::span<f32> transformed_scratch) {
  const size_t table_size = static_cast<size_t>(model.subquantizers) *
    kCentroidsPerSubquantizer;
  if (table.size() != table_size || transformed_scratch.size() != model.dim) {
    throw std::invalid_argument("PQ distance-table buffer shape mismatch");
  }
  transform(model, input, transformed_scratch);
  const u32 dsub = model.subvector_dim();
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    const f32* value = transformed_scratch.data() + static_cast<size_t>(subquantizer) * dsub;
    const f32* centroids = model.centroids.data() +
      static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer * dsub;
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

这是 **ADC（Asymmetric Distance Computation）** 的核心：查询向量被「非对称」地保留为浮点（不编码），对每个子量化器的 256 个中心算欧氏距离平方，存进 `table[m*256 + c]`。得到的表大小 `M*256` 个 float（M=32 时即 32KB），就是查询时的 LUT。

布局是 **行优先，subquantizer 为慢索引**：`table[subquantizer * 256 + centroid]`。这个布局在 GPU 端被原样复用（`query_luts[query_slot * M * 256 + subquantizer * 256 + centroid]`），保证 device 端查表时 coalesced。

### 7.8 `asymmetric_distance`：用 LUT + code 算距离

```cpp
// src/gpu_search/pq_index.cc:226-238
f32 asymmetric_distance(const Model& model, std::span<const f32> table,
                        std::span<const u8> code) {
  if (table.size() != static_cast<size_t>(model.subquantizers) *
        kCentroidsPerSubquantizer || code.size() != model.code_bytes()) {
    throw std::invalid_argument("PQ asymmetric-distance buffer shape mismatch");
  }
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    distance += table[static_cast<size_t>(subquantizer) *
      kCentroidsPerSubquantizer + code[subquantizer]];
  }
  return distance;
}
```

ADC 距离 = `Σ_m table[m*256 + code[m]]`。M 次查表 + M 次加法，零浮点乘法。这就是 PQ 在检索时如此快的原因：把 D 维浮点距离降成 M 次表查找。

device 端等价实现是 `approximate_entry`（`candidate_scoring.cuh:431-440`）：

```cpp
// src/gpu_search/persistent_kernel/candidate_scoring.cuh:431-440
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

完全同构，只是 `model.subquantizers` 换成 `params.pq_subquantizers`、`table` 换成 `query_lut`、`code` 来自 `params.pq_codes` 等设备指针。

---

## 8. 关键数据结构与流程图

### 8.1 `QueryDescriptor` 字节布局

```
QueryDescriptor (48 B, 自然对齐到 8)
+--------+--------+--------+--------+--------+--------+--------+--------+
| 0      request_id                | 8      snapshot_epoch            |
+--------+--------+--------+--------+--------+--------+--------+--------+
| 16     query_device_address      | 24     result_device_address     |
+--------+--------+--------+--------+--------+--------+--------+--------+
| 32     query_slot        | 36     result_capacity  |
+--------+--------+--------+--------+--------+--------+--------+--------+
| 40 dim  | 42 k    | 44 dtype| 45 flags| 46 reserved              |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

### 8.2 `CompletionDescriptor` 字节布局

见第 2.2 节末尾的图，128 字节，9 个 u64 + 13 个 u32/i32，无 padding。

### 8.3 PQ 在 GPU 上的端到端流程

```
                        ┌─────────────────────────────────────────────┐
                        │  离线阶段（第 12 课、第 29 课）              │
                        │  tools/vamana_offline/pq_indexer.cc          │
                        └─────────────────────────────────────────────┘
                                          │
                  train_model → Model{dim, M, rotation, centroids}
                                          │
                          write_model → prefix.pq32  (只含 codebook)
                          encode each base vector → prefix*.pq32.codes
                                          │
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  引擎启动（第 11/12 课 construction.cc）    │
                        └─────────────────────────────────────────────┘
                                          │
   pq::read_model(prefix.pq32, pq_model)  │   device_allocate(d_pq_centroids, M*256*dsub)
   校验 magic/version/checksum/shape      │   cudaMemcpy(d_pq_centroids, pq_model.centroids)
                                          │   d_pq_codes = d_remote_buffer  (GPUNetIO 区)
                                          │   d_query_luts = alloc(slot_count * M*256*4)
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  查询时（第 20 课 query_traversal.cuh）       │
                        └─────────────────────────────────────────────┘
                                          │
   for each query_slot:                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 1. decode query       : query_luts 区域之外的 decoded_queries   │
   │ 2. OPQ transform      : transformed = opq_matrix * query        │
   │ 3. build LUT (256×M)  : for idx in [0, M*256):                  │
   │      sub = idx / 256                                            │
   │      diff = transformed[sub*dsub..] - d_pq_centroids[idx*dsub..]│
   │      query_luts[slot*M*256 + idx] = dot(diff, diff)            │
   └────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  候选评分（第 18 课 candidate_scoring.cuh）  │
                        └─────────────────────────────────────────────┘
                                          │
   for each candidate handle h:           ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ code = d_pq_codes[h * M]   (base)                                │
   │      或 d_delta_pq_codes[slot * M]   (delta)                     │
   │      或 d_resident_pq_codes[slot * M] (resident)                 │
   │ dist = Σ_m query_lut[m*256 + code[m]]   ← approximate_entry     │
   └────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  精确重排（第 18 课）                         │
                        │  取 top beam 的精确向量做真 L2，重排得到 top-k │
                        └─────────────────────────────────────────────┘
```

关键点：

- **codebook 常驻 GPU**：`d_pq_centroids` 在引擎启动时一次性分配并拷贝，查询时只读。
- **PQ code 也常驻 GPU**：base code 在 `d_pq_codes`（实际上是 GPUNetIO 的 remote buffer 区，见第 12 课 construction），delta code 在 `d_delta_pq_codes` 和 `d_resident_pq_codes`。
- **LUT 每查询重建**：每个 query slot 拥有独立的 `M*256` float LUT，在查询开始时由 kernel 并行构建。
- **评分零浮点乘法**：`approximate_entry` 只做 M 次表查 + M 次加法，这是 PQ 速度的来源。

### 8.4 Codebook 在 GPU 上的内存布局

```
d_pq_centroids  (f32 数组，长度 = M * 256 * dsub)

  subquantizer 0                subquantizer 1            ...  subquantizer M-1
 ┌──────────────────────────┐ ┌──────────────────────────┐     ┌──────────────────────────┐
 │ centroid 0  (dsub floats)│ │ centroid 0  (dsub floats)│ ... │ centroid 0  (dsub floats)│
 │ centroid 1  (dsub floats)│ │ centroid 1  (dsub floats)│ ... │ centroid 1  (dsub floats)│
 │ ...                      │ │ ...                      │     │ ...                      │
 │ centroid 255(dsub floats)│ │ centroid 255(dsub floats)│ ... │ centroid 255(dsub floats)│
 └──────────────────────────┘ └──────────────────────────┘     └──────────────────────────┘

寻址：d_pq_centroids[subquantizer * 256 * dsub + centroid * dsub + dim]
```

这个布局让 `build_distance_table` 的 GPU kernel 可以用 `index = subquantizer * 256 + centroid` 一次 coalesced 读出 `dsub` 个 float（连续），与 `query_lut[index]` 的写入方向一致。

### 8.5 LUT 在 GPU 上的内存布局

```
d_query_luts  (f32 数组，长度 = slot_count * M * 256)

  slot 0                       slot 1                      ...  slot (slot_count-1)
 ┌──────────────────────────┐ ┌──────────────────────────┐     ┌──────────────────────────┐
 │ sub 0: 256 floats        │ │ sub 0: 256 floats        │ ... │ sub 0: 256 floats        │
 │ sub 1: 256 floats        │ │ sub 1: 256 floats        │ ... │ sub 1: 256 floats        │
 │ ...                      │ │ ...                      │     │ ...                      │
 │ sub M-1: 256 floats      │ │ sub M-1: 256 floats      │ ... │ sub M-1: 256 floats      │
 └──────────────────────────┘ └──────────────────────────┘     └──────────────────────────┘

寻址：d_query_luts[query_slot * M * 256 + subquantizer * 256 + centroid]
```

`approximate_entry` 用 `query_lut[subquantizer * 256 + code[subquantizer]]` 查表，每子量化器一次 256-way 选中，M 次累加。

---

## 9. 与其他模块的关系

本课的内容在 dvstor 全局中被以下课程消费：

- **第 2 课（公共类型与配置）**：`u8/u16/u32/u64/i32/f32` 别名其实来自 `common/types.hh`，本课在 `types.hh:8-13` 重新定义（避免头文件循环依赖）。两处定义必须一致。
- **第 7 课（schema-15 索引格式）**：PQ 模型文件 `.pq32` 是 schema-15 manifest 引用的组件之一，`construction.cc:113` 用 `index_path::navigation_model_file` 拼出路径。
- **第 10 课（delta/动态路由/预算）**：`DeltaSupersedeUpdate` / `DeltaOverrideUpdate` / `DeltaDurableUpdate` / `ResidentPqEraseUpdate` / `DynamicRouteUpdate` / `DeviceDynamicRouteSlot` 全部在那里被实际驱动；`kDynamicRouteSlotsPerShard` 的容量限制在那里体现。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`Telemetry` 作为 `PersistentSearchEngine::Impl` 的成员持有，`TelemetrySnapshot` 通过 PImpl 边界暴露给上层。
- **第 12 课（construction 上）**：`pq::read_model` 在那里被调用，`d_pq_centroids` / `d_pq_codes` / `d_query_luts` 在那里分配和上传。
- **第 14 课（查询执行/路由/完成）**：`QueryDescriptor` 是提交入口，`CompletionDescriptor` 是完成出口，两者都通过 device ring 流转。
- **第 15 课（增量发布）**：`DeltaPublishDescriptor` + 各种 update payload + `DeltaPublishCompletion` 是发布协议的全部数据结构。
- **第 16 课（存储回收 RCU）**：`ResidentPqEraseUpdate` 在那里被批量生成和安装。
- **第 17 课（kernel 启动器/上下文/device ring）**：`QueryDescriptor` / `CompletionDescriptor` / `DeltaPublishDescriptor` / `DeltaPublishCompletion` 都在 device ring 里有对应 slot，大小由本课的 `static_assert` 锁定。
- **第 18 课（候选评分）**：`approximate_entry` 用本课的 LUT 布局；`build_distance_table` 的 GPU 版在 `query_traversal.cuh` 实现。
- **第 19 课（RDMA cache）**：`CompletionDescriptor` 里的 `remote_pages` / `remote_batches` / `cache_hits` / `graph_read_retries` 反映 cache 行为；`TelemetrySnapshot` 里的 RDMA 类字段是 cache 的长期聚合。
- **第 20 课（查询遍历主循环）**：LUT 构建代码（`query_traversal.cuh:317-333`）就在主循环开头，是 prepare 阶段的核心。
- **第 22 课（GPUNetIO 传输/probe）**：`QueryDescriptor` 的 device 地址字段是 GPUNetIO 异步流水线的前提。
- **第 28 课（计算侧 storage owner 更新）**：`DynamicRouteUpdate` / `DeltaPublishDescriptor` 在那里被计算节点接收并安装。
- **第 30 课（breakdown benchmark/实验脚本）**：`TelemetrySnapshot` 的字段是 benchmark 报告的主要数据源，`reset()` 用于在测量前清零。

---

## 10. 小结

本课建立了 GPU 引擎的三块基石：

1. **类型语言**：`QueryDescriptor`（48 B）与 `CompletionDescriptor`（128 B，`static_assert` 锁死）是查询路径的入参/出参；delta 发布路径有一组 16/40 字节的小定长 update 结构；`DeviceDynamicRouteSlot`（48 B）用 device-scope seqlock 实现无锁的 control-CTA 写、query-CTA 读，查询永不等待写者。所有结构都是 POD，可被 `static_assert` 锁大小、可直接 `cudaMemcpy`、可进 RDMA 报文。

2. **遥测体系**：`Telemetry` 是 70+ 个 `std::atomic<u64>` 的集合，`TelemetrySnapshot` 是它的纯 POD 视图。`snapshot()` 用单一字段的 acquire/release 串联 publisher 的更新与读者，其余字段全 relaxed——以最低开销换取「近似一致」的快照。`reset()` 区分 counter（清零）与 gauge（保留），并把 peak 重置为当前值。字段按「显存/吞吐/等待/阶段/RDMA/图遍历/动态路由/缓存/delta/resident PQ/容量/可见性」十二类组织，每个子系统调试时只看对应那一类。

3. **PQ 模型**：`.pq16`/`.pq32` 文件 = `ModelHeader` + 可选 OPQ rotation + codebook，用 FNV-1a checksum 防损坏，`read_model` 做 12 项 header 校验 + payload 形状校验 + 文件大小校验 + checksum 校验 + `validate` 五道关卡。codebook 在引擎启动时上传到 `d_pq_centroids`，PQ code 常驻 `d_pq_codes` / `d_delta_pq_codes` / `d_resident_pq_codes`。查询时 kernel 在 `d_query_luts` 里为每个 slot 构建 256×M 的 LUT，候选评分退化为 M 次表查 + M 次加法（`approximate_entry`），这是 dvstor 高吞吐近似检索的算术基础。

下一课（第 10 课）将把这些 delta/动态路由结构真正运转起来——讲清 delta 的可见性判定、override 提升、resident PQ 槽分配与回收、动态路由的发布与 seqlock 读路径，以及 mutation 容量预算如何用 `mutation_capacity_reserved` 这个 gauge 节流。
