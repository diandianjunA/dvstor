# 第 7 课：索引格式契约 schema-15

> 课号 7 / 30 ｜ 课题：索引格式契约 schema-15
> 项目根目录：`/home/xjs/experiment/dvstor`
> 涉及代码：`src/gpu_search/index_format.hh`、`src/gpu_search/index_format.cc`
> 辅助引用：`src/service/index_metadata.{hh,cc}`（第 8 课深讲）、`src/remote_pointer.hh`、`src/common/vector_dtype.hh`、`src/common/index_path.hh`、`src/vamana/anchor_index.hh`

## 7.1 本课目标与涉及文件

第 6 课我们讲了 vamana 图本身的三件套：紧凑图记录的字节布局（`vamana_node.hh`）、热点图视图（`hot_graph.hh`）、以及 anchor/idmap 的二进制格式。这些都是**单条记录、单文件**层面的契约。但 dvstor 是一个**多存储节点、多 shard、带动态回收与路由发布**的系统——一台 GPU 上的持久化 kernel 需要知道：

1. 整张索引在远端 RDMA 内存里**怎么排布**：哪个 shard 在哪个 memory node 上，图记录从哪个 byte offset 开始，控制页在哪，PQ code 区在哪，动态节点区从哪开始。
2. 这些布局信息**怎么校验**：远程读到的字节流是否还是离线构建时写入的那一份，有没有被半截写入撕裂，schema 是否匹配当前进程期望的版本。
3. **运行时元数据**（比如 canonical route 快照）怎么塞进**固定的 4 KiB 控制页**里而不破坏 schema-15 的磁盘契约。
4. 一个全局 ordinal 怎么映射成 `RemotePtr`，反之一个 `RemotePtr` 怎么还原成 ordinal——这是 GPU kernel 跨 shard 跳转的基础。

这一课就是回答这四个问题的"格式总账"。它由两个文件组成：

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/gpu_search/index_format.hh` | 193 | schema/version 常量、`NavigationLayout`/`ShardRegion`/`StorageControlBlock`/`StorageRoutePublication`/`CodeHeader` 等 POD 结构定义、所有 `static_assert`、对外 API 声明 |
| `src/gpu_search/index_format.cc` | 534 | FNV-1a 风格 checksum、`.meta.json` 解析与分布式 view 合成、`View`/`CodeHeader` 校验、anchor 侧入口点合成、ordinal↔RemotePtr 双向映射 |

注意这两个文件**只在 CPU 侧运行**。GPU kernel 不直接 `#include` 它们——kernel 消费的是这些结构导出到 `types.hh` 的镜像（见第 9、17 课）。本课讲的是"磁盘契约 + CPU 侧解析"，是连接第 6 课（vamana 图格式）与第 11 课（持久化引擎生命周期）/第 17 课（kernel 启动器）/第 23 课（存储节点启动校验）的桥梁。

读完本课你应当能够：

1. 逐字段背出 `NavigationLayout`（80 字节）、`ShardRegion`（88 字节）、`StorageControlBlock`（640 字节，64 对齐）、`StorageRoutePublication`（448 字节）、`CodeHeader`（120 字节）的布局，并能指出每个字段被哪一课的哪个模块消费。
2. 解释 schema-15 契约为什么是 **fail-stop**（不自动升级、不降级），以及 `kMetadataSchemaVersion == 15` 这个硬性等式在 `synthesize_distributed_view` 里如何守住整条解析路径。
3. 画出"计算节点文件 vs 存储节点文件"的索引布局表，以及 fixed record / compact graph record / control page 的字节级布局图。
4. 说明 4 KiB 控制页 offset 1024 处的 canonical route 快照区是怎么"塞进"既有 schema-15 而不移动任何 dynamic node offset 的，以及它的 begin/end 序列号 + body checksum 三重防撕裂机制。
5. 理解 `RemotePtr` 的 8 字节 16+48 编码，以及 compact graph record 里 5 字节指针如何再压缩这 8 字节。
6. 跟着 `synthesize_distributed_view` 走完一遍从 `.meta.json` 到 `View` 的合成流程，包括 anchor 侧入口点的 rank-轮转填充算法。

## 7.2 `index_format.hh`：schema 契约常量与结构定义

文件路径：`/home/xjs/experiment/dvstor/src/gpu_search/index_format.hh`

### 7.2.1 头部 include 与 namespace

```cpp
// index_format.hh:1-15
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"

namespace gpu_search::format {
```

注意它 include 的是 `remote_pointer.hh`（不是 `gpu_search/remote_pointer.hh`），因为 `RemotePtr` 是项目根级类型（第 4 课已讲过它的 16+48 编码）。`common/types.hh` 提供 `u8/u32/u64/byte_t` 等固定宽度别名（第 2 课）。`common/vector_dtype.hh` 提供 `VectorDType` 枚举与 `parse_vector_dtype`（见 7.3.4）。`<iosfwd>` 而不是 `<ostream>` 是为了在头文件里只前向声明 `std::ostream`，把 `write_code_header(std::ostream&, ...)` 的实现留在 `.cc`。

### 7.2.2 schema/version 常量与 magic 数字

```cpp
// index_format.hh:17-37
inline constexpr std::array<char, 8> kCodeMagic{'D', 'V', 'G', 'P', 'U', 'C', '5', '\0'};
inline constexpr u32 kVersion = 5;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kMaxEntryPoints = 512;
inline constexpr u32 kGraphCacheLineBytes = 512;
inline constexpr u32 kCompactPointerBytes = 5;
inline constexpr u64 kNodeBaseOffset = 16;
inline constexpr u32 kMetadataSchemaVersion = 15;
inline constexpr u32 kStorageControlBytes = 4096;
inline constexpr u64 kStorageControlMagic = 0x314c525443565344ULL;  // "DSVCTRL1"
inline constexpr u32 kStorageControlVersion = 2;
inline constexpr u32 kMaxComputeClients = 64;
// The route publication lives in the unused tail of the existing 4 KiB
// storage control page.  It is runtime metadata, not an on-disk index record,
// so adding it neither changes schema-15 nor moves any dynamic node offset.
inline constexpr u32 kStorageRoutePublicationOffset = 1024;
inline constexpr u64 kStorageRoutePublicationMagic =
  0x3154554f52565344ULL;  // "DSVROUT1"
inline constexpr u32 kStorageRoutePublicationVersion = 1;
inline constexpr u32 kStorageRouteSlots = 8;
inline constexpr u32 kStorageRouteMaxCodeBytes = 32;
```

逐个看：

- **`kCodeMagic`** = `"DVGPUC5\0"`（8 字节，含末尾 NUL）。这是 PQ code sidecar 文件（`*.pq*.codes`）的 magic，标识"DVstor GPU Code v5"。`kVersion = 5` 是这个 sidecar 文件头的格式版本，**与 `kMetadataSchemaVersion` 是两套独立的版本号**——前者管 PQ code 文件头，后者管 `.meta.json` 全局 schema。不要混淆。
- **`kEndianMarker = 0x01020304`**：写入文件头后再读回来，如果读成 `0x04030201` 就说明跨端序了，直接拒绝。dvstor 假定小端序，这个 marker 只是 fail-fast 的额外保险。
- **`kMaxEntryPoints = 512`**：单个 `View` 的 entry point 表上限。GPU kernel 启动时会从 entry points 开始图遍历，512 是显存预算与召回率之间的折中（第 17 课的 kernel 上下文会把这个表原样拷到 device）。
- **`kGraphCacheLineBytes = 512`**：compact graph record 一条记录最多 512 字节。这是第 6 课讲过的"一条邻居列表落在一个 cache line"约束的**硬上限**，`validate_layout` 会强制 `graph_entry_bytes <= 512`。
- **`kCompactPointerBytes = 5`**：compact graph record 里每个邻居指针只占 5 字节，而不是 `RemotePtr` 的 8 字节。5 字节编码见 7.2.9。
- **`kNodeBaseOffset = 16`**：每个 shard 的 fixed record 区从该 shard 内存段的第 16 字节开始。前 16 字节留给 shard-local 的元数据头（与第 6 课的 `VamanaNode` 对齐），所有 shard 的 `node_base_offset` 必须都等于这个值，`validate_view` 会强制（见 7.3.2）。
- **`kMetadataSchemaVersion = 15`**：本课的标题。`.meta.json` 里的 `schema_version` 字段必须**严格等于 15**，任何不匹配都 fail-stop。schema-14 → 15 的"升级路径"在 7.4 节专门讨论。
- **`kStorageControlBytes = 4096`**：每个 shard 的存储控制页固定 4 KiB。这个页同时承载 `StorageControlBlock`（前 640 字节）和 `StorageRoutePublication`（offset 1024 起，448 字节），剩余空间预留。
- **`kStorageControlMagic = 0x314c525443565344`**：ASCII 解读为 `"DSVCTRL1"`（注意 little-endian，所以读出来字符顺序是 D-S-V-C-T-R-L-1）。这是 `StorageControlBlock` 的 magic。
- **`kStorageControlVersion = 2`**：`StorageControlBlock` 的版本号。version 1 是 schema-14 时代的，没有 `reclaim_ack_sequences` 数组；version 2 加了 64 路 compute client 的 reclaim ACK 序列（见 7.2.5）。这就是 schema-14 → 15 升级在**控制页层面**的体现。
- **`kMaxComputeClients = 64`**：一个存储 shard 最多服务 64 个 compute client，对应 `reclaim_ack_sequences[64]`。
- **`kStorageRoutePublicationOffset = 1024`**：canonical route 快照在 4 KiB 控制页内的起始偏移。注释明确指出这是"塞进既有 4 KiB 控制页的空闲尾部"，所以它**不是磁盘契约的一部分**，加它不改变 schema-15 也不移动任何 dynamic node offset。这是非常重要的设计约束：存储节点重启后看到的 dynamic node 布局必须与离线构建时一致，route 快照只是运行时叠加层。
- **`kStorageRoutePublicationMagic = 0x3154554f52565344`**：ASCII 为 `"DSVROUT1"`。
- **`kStorageRoutePublicationVersion = 1`**：route 快照格式版本 1。
- **`kStorageRouteSlots = 8`**：每个 shard 的 canonical route 表固定 8 槽。这与第 10 课 dynamic route overlay、第 21 课 `DeviceDynamicRouteSlot` 的 8 槽对应——GPU 端的 `kDynamicRouteSlotsPerShard=8` 就是这个值的镜像。
- **`kStorageRouteMaxCodeBytes = 32`**：每个 route slot 最多带 32 字节的 navigation code（PQ 编码的 representative 向量），用于 kernel 端做近似路由评分。

### 7.2.3 `QuantizerKind` 枚举

```cpp
// index_format.hh:39-41
enum class QuantizerKind : u32 {
  opq_pq = 1,
};
```

目前唯一支持的量化器是 OPQ+PQ（optimal product quantization + product quantization）。值从 1 开始（0 留作"未设置"哨兵），`NavigationLayout::quantizer_kind` 和 `CodeHeader::quantizer_kind` 都用 `static_cast<u32>(QuantizerKind::opq_pq)` 初始化。`validate_layout` 会强制 `quantizer_kind == opq_pq`，所以加新量化器（比如纯 PQ、scalar quant）必须同时改这里、改 `validate_layout`、改 `synthesize_distributed_view` 的 `navigation_format` 白名单——三处一致才放行。

### 7.2.4 `NavigationLayout`：全局导航参数

```cpp
// index_format.hh:43-60
struct NavigationLayout {
  u32 dim{};
  u32 graph_degree{};
  u32 vector_dtype{};
  u32 quantizer_kind{static_cast<u32>(QuantizerKind::opq_pq)};
  u32 pq_subquantizers{};
  u32 pq_bits{};
  u32 code_bytes{};
  u32 num_shards{};
  u32 graph_entry_bytes{};
  u32 graph_pointer_bytes{kCompactPointerBytes};
  u32 graph_shard_bits{};
  u32 medoid_ordinal{};
  u32 reserved0{};
  u64 num_nodes{};
  u64 base_generation{1};
  u64 model_checksum{};
};
```

这是"整张索引的全局参数"，所有 shard 共享。逐字段：

- `dim`：向量维度。
- `graph_degree`：vamana 图的度数 R（第 6 课 `VamanaNode` 的 R）。
- `vector_dtype`：`static_cast<u32>(VectorDType)`，取值 0/1/2 对应 float32/uint8/int8（见 `common/vector_dtype.hh:15-19`）。
- `quantizer_kind`：见 7.2.3。
- `pq_subquantizers`：PQ 子量化器个数 M。`validate_layout` 强制 `dim % pq_subquantizers == 0`。
- `pq_bits`：PQ 每段比特数，**强制为 8**（即每段 256 个 centroid）。这是 GPU kernel LUT 实现的硬约束。
- `code_bytes`：单个向量的 PQ code 字节数，强制等于 `pq_subquantizers`（因为 pq_bits=8 → 每段 1 字节）。
- `num_shards`：shard 个数，等于存储节点数 `num_memory_nodes`。
- `graph_entry_bytes`：一条 compact graph record 的字节数，强制 `8 + R*5 <= graph_entry_bytes <= 512`。
- `graph_pointer_bytes`：邻居指针字节数，强制等于 `kCompactPointerBytes = 5`。
- `graph_shard_bits`：邻居指针里 shard 编号的位宽，由 `shard_bits_for(num_shards)` 推导（见 7.3.1）。
- `medoid_ordinal`：整张图的 medoid 在全局 ordinal 空间里的下标。注意它是 ordinal 而不是 RemotePtr——`synthesize_distributed_view` 会把 `.meta.json` 里的 `(memory_node, offset)` 反解成 ordinal。
- `reserved0`：4 字节对齐填充。
- `num_nodes`：全局节点总数，强制 `< 2^30`（GPU 端用 30 位 ordinal）。
- `base_generation`：schema-15 immutable base 节点的 generation。**默认 1**——注释里说 base 节点 generation 为 0，但在线版本从 1 开始，两者都是合法的 canonical route representative（见 `validate_storage_route_publication` 的 7.3.3）。这里默认 1 是因为 `synthesize_distributed_view` 显式赋 `synthesized.layout.base_generation = 1`（`index_format.cc:351`）。
- `model_checksum`：PQ 模型（`.pqM` 文件）的校验和，必须非零。这个值同时写进 `CodeHeader`，用于 cross-check PQ code sidecar 与 metadata 描述的是同一套模型。

注意 `NavigationLayout` **没有 `alignas`**，靠字段顺序自然对齐：12 个 u32（48 字节）+ 1 个 u32 reserved（4 字节）+ 3 个 u64（24 字节）= 76 字节。没有 `static_assert(sizeof(NavigationLayout) == ...)`——这是有意的，因为它是 `View` 的内嵌字段，不单独序列化，不需要固定字节布局。

### 7.2.5 `ShardRegion`：单 shard 的远端布局描述

```cpp
// index_format.hh:62-78
struct ShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{kNodeBaseOffset};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 control_remote_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};

  bool operator==(const ShardRegion&) const = default;
};
```

一个 `ShardRegion` 描述一个 shard 在某个 memory node 上的**远端内存布局**。所有 offset 都是相对于该 shard 所在 RDMA 内存段的字节偏移。逐字段：

- `ordinal_base`：这个 shard 的第一个节点在全局 ordinal 空间里的起点。shard 0 的 `ordinal_base = 0`，shard i 的 `ordinal_base = sum(shard[0..i-1].node_count)`。`validate_view` 会强制这个累加关系（7.3.2）。
- `node_count`：本 shard 的节点数。
- `node_base_offset`：fixed record 区起点，**强制等于 `kNodeBaseOffset = 16`**（7.3.2）。
- `node_stride`：一条 fixed record 的步长（含 header + id + generation + 向量本体，见第 6 课 `VamanaNode`）。
- `graph_base_offset`：compact graph 区起点。fixed record 区结束后紧跟 graph 区。
- `dynamic_base_offset`：动态节点区起点。这个区是 schema-15 的核心——它允许在线插入新节点而不移动既有节点。`dynamic_base_offset` 必须 `>= code_remote_offset + code_bytes`（7.3.2）。
- `control_remote_offset`：4 KiB 控制页起点。`align_up(dynamic_offsets[shard], 64)` 算出来的（7.3.4）。
- `code_remote_offset`：PQ code 区起点，**强制等于 `control_remote_offset + kStorageControlBytes`**（7.3.2）。控制页紧跟在 dynamic 区后面，PQ code 区紧跟在控制页后面。
- `code_bytes`：PQ code 区字节数，强制等于 `node_count * layout.code_bytes`。
- `memory_node`：本 shard 所在的 memory node 编号，**强制等于 shard index**（7.3.2）。这是 dvstor 的硬约束：shard i 必须在 memory node i 上。这个约束简化了 kernel 端的寻址逻辑（ordinal → memory_node 是确定的）。
- `dynamic_record_bytes`：动态节点区里一条记录的字节数。
- `dynamic_hot_offset`：动态记录里"hot 段"（compact graph 部分）的偏移。
- `dynamic_code_offset`：动态记录里 PQ code 段的偏移。

`operator==` 用 `= default` 是 C++20 的语法，逐成员比较。这用在 `validate_view` 的 shard 一致性检查里。

`sizeof(ShardRegion) == 88` 由 `static_assert` 守护（7.2.11）。9 个 u64（72 字节）+ 4 个 u32（16 字节）= 88。

### 7.2.6 `StorageControlBlock`：4 KiB 控制页的主结构

```cpp
// index_format.hh:80-98
struct alignas(64) StorageControlBlock {
  u64 magic{kStorageControlMagic};
  u32 version{kStorageControlVersion};
  u32 header_bytes{sizeof(StorageControlBlock)};
  u32 shard_id{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
  u32 code_bytes{};
  u32 compute_client_count{};
  u32 reserved0{};
  u64 next_maintenance_sequence{1};
  u64 durable_maintenance_sequence{};
  u64 dynamic_high_watermark{};
  u64 reclaim_pending_nodes{};
  u64 reclaim_reused_nodes{};
  u64 reserved1{};
  std::array<u64, kMaxComputeClients> reclaim_ack_sequences{};
};
```

这是 4 KiB 控制页的**主结构**，64 字节对齐（`alignas(64)`）。逐字段：

- `magic` = `kStorageControlMagic`："DSVCTRL1"。
- `version` = `kStorageControlVersion = 2`。**version 2 相比 version 1 加的就是末尾的 `reclaim_ack_sequences[64]` 数组**——这是 schema-14 → 15 升级在控制页层面的体现（7.4 节详述）。
- `header_bytes` = `sizeof(StorageControlBlock)`：自描述大小，让老版本代码能跳过未知的尾部字段。
- `shard_id`：本控制页所属的 shard。
- `dynamic_record_bytes` / `dynamic_hot_offset` / `dynamic_code_offset`：与 `ShardRegion` 同名字段对应，存储节点启动时会 cross-check 这两边一致（第 23 课）。
- `code_bytes`：PQ code 区字节数。
- `compute_client_count`：当前连到这个 shard 的 compute client 数。
- `reserved0`：4 字节对齐填充。
- `next_maintenance_sequence`：下次维护操作的序列号，从 1 开始（第 26 课 wire protocol 用）。
- `durable_maintenance_sequence`：已持久化的维护序列号。RCU 回收（第 16 课）推进这个值。
- `dynamic_high_watermark`：动态节点区已分配到的最高水位线。
- `reclaim_pending_nodes`：待回收节点数。
- `reclaim_reused_nodes`：已复用节点数。
- `reserved1`：8 字节对齐填充。
- `reclaim_ack_sequences[64]`：每个 compute client 最近一次确认的 reclaim 序列号。**这就是 version 2 加的字段**——schema-14 时这里没有，回收只能保守等所有 client 超时；schema-15 加了显式 ACK 数组，让存储节点能精确知道哪个 client 已经看到了哪次回收（第 16 课 RCU 详述）。

`sizeof(StorageControlBlock) == 640`，由 `static_assert` 守护：1+1+1+1+1+1+1+1+1+1 = 10 个 u32（40 字节）+ 1 个 u64 magic + 6 个 u64（48 字节）+ 64 个 u64（512 字节）= 8 + 40 + 48 + 512 = 648？不对——重新数：magic(8) + version(4) + header_bytes(4) + shard_id(4) + dynamic_record_bytes(4) + dynamic_hot_offset(4) + dynamic_code_offset(4) + code_bytes(4) + compute_client_count(4) + reserved0(4) = 40 字节，然后 next_maintenance_sequence(8) + durable_maintenance_sequence(8) + dynamic_high_watermark(8) + reclaim_pending_nodes(8) + reclaim_reused_nodes(8) + reserved1(8) = 48 字节，最后 `array<u64,64>` = 512 字节。40+48+512 = 600，加上头部 magic 8 = 608，不到 640。这是因为 `alignas(64)` 强制结构体大小是 64 的倍数，编译器在末尾补了 32 字节填充到 640。`static_assert(sizeof(StorageControlBlock) == 640)` 守护这个值。

`static_assert(sizeof(StorageControlBlock) <= kStorageControlBytes)` 确保 640 <= 4096，主结构能放进控制页。

### 7.2.7 `StorageRouteSlot` 与 `StorageRoutePublication`：canonical route 快照

```cpp
// index_format.hh:100-124
struct StorageRouteSlot {
  u64 remote_node{};
  u32 id{};
  u32 generation{};
  std::array<u8, kStorageRouteMaxCodeBytes> navigation_code{};
};

// A begin/end sequence plus a body checksum makes a torn body detectable.
// Compute readers additionally bracket the body RDMA with two completed reads
// of sequence_begin, which rules out a coherent old body paired with a newly
// observed sequence. On any mismatch they keep the previous snapshot; no
// query or update thread waits for a route refresh.
struct alignas(64) StorageRoutePublication {
  u64 sequence_begin{};
  u64 magic{kStorageRoutePublicationMagic};
  u32 version{kStorageRoutePublicationVersion};
  u32 header_bytes{sizeof(StorageRoutePublication)};
  u32 shard_id{};
  u32 slot_count{kStorageRouteSlots};
  u32 code_bytes{};
  u32 reserved{};
  u64 body_checksum{};
  std::array<StorageRouteSlot, kStorageRouteSlots> slots{};
  u64 sequence_end{};
};
```

这是 4 KiB 控制页 offset 1024 处的 canonical route 快照。注释非常关键——它解释了**三重防撕裂机制**：

1. `sequence_begin` + `sequence_end`：发布前 `sequence_begin` 自增成偶数，写完 body 后 `sequence_end = sequence_begin`。读者先读 `sequence_begin`，读 body，再读 `sequence_end`，如果不等就说明 body 被撕裂。
2. `body_checksum`：FNV-1a 风格校验和（7.3.3），覆盖 magic 到 reserved 的头部 + slots 数组。即使 begin/end 相等，checksum 不对也拒绝。
3. **compute reader 额外做两次 `sequence_begin` 读**：开头读一次，body 读完再读一次，三次都相等才接受。这规则了"读者看到了新 sequence_begin，但 body 还没写完"的窗口——这种情况下第二次读到的 `sequence_begin` 会再变，读者就丢弃。

注释最后一句很重要："On any mismatch they keep the previous snapshot; no query or update thread waits for a route refresh."——route 快照更新是**完全异步**的，查询和更新路径都不阻塞。这与第 10 课 dynamic route overlay 的"无锁 seqlock"思路一致。

`StorageRouteSlot` 逐字段：
- `remote_node`：representative 节点的 `RemotePtr.raw_address`（16+48 编码）。`validate_storage_route_publication` 会检查高 16 位（memory_node）等于 `expected_shard`（7.3.3）。
- `id`：representative 的外部 ID。
- `generation`：representative 的 generation。schema-15 immutable base 节点 generation=0，在线版本从 1 开始，两者都合法。
- `navigation_code[32]`：representative 的 PQ code，最多 32 字节。kernel 端用它做近似路由评分。

`StorageRoutePublication` 逐字段：
- `sequence_begin` / `sequence_end`：见上。
- `magic` = `kStorageRoutePublicationMagic`："DSVROUT1"。
- `version` = 1。
- `header_bytes` = `sizeof(StorageRoutePublication)` = 448。
- `shard_id`：本快照所属 shard。
- `slot_count` = 8。
- `code_bytes`：每个 slot 实际用的 code 字节数（<= 32）。
- `reserved`：对齐填充。
- `body_checksum`：见 7.3.3。
- `slots[8]`：8 个 representative。
- `sequence_end`：见上。

`sizeof(StorageRouteSlot) == 48`（8+4+4+32）。`sizeof(StorageRoutePublication) == 448`：8+8+4+4+4+4+4+4+8 = 48 字节头部 + 8*48 = 384 字节 slots + 8 字节 sequence_end = 440，再补到 64 的倍数 = 448。`alignas(64)` 守护。

### 7.2.8 `CodeHeader`：PQ code sidecar 文件头

```cpp
// index_format.hh:126-143
struct CodeHeader {
  std::array<char, 8> magic{kCodeMagic};
  u32 version{kVersion};
  u32 header_bytes{sizeof(CodeHeader)};
  u32 endian_marker{kEndianMarker};
  u32 memory_node{};
  u32 quantizer_kind{static_cast<u32>(QuantizerKind::opq_pq)};
  u32 code_bytes{};
  u32 node_size{};
  u32 reserved0{};
  u64 entry_count{};
  u64 remote_offset{};
  u64 payload_bytes{};
  u64 model_checksum{};
  u64 payload_checksum{};
  u64 header_checksum{};
  std::array<u64, 4> reserved{};
};
```

这是 `.pq*.codes` 文件（PQ code sidecar）的 120 字节文件头。每个 shard 一个 sidecar 文件（`index_path::navigation_code_file`，见 7.3.4）。逐字段：

- `magic` = `kCodeMagic`："DVGPUC5\0"。
- `version` = `kVersion = 5`。
- `header_bytes` = 120。
- `endian_marker` = `kEndianMarker`。
- `memory_node`：本 sidecar 所属 memory node。
- `quantizer_kind`：强制 `opq_pq`。
- `code_bytes`：单条 code 字节数（= `pq_subquantizers`）。
- `node_size`：一条 fixed record 的步长（与 `ShardRegion::node_stride` 对应）。
- `reserved0`。
- `entry_count`：本 sidecar 的 code 条数（= shard `node_count`）。
- `remote_offset`：本 sidecar 在远端 RDMA 内存里的偏移（= `ShardRegion::code_remote_offset`）。
- `payload_bytes`：code 区字节数 = `entry_count * code_bytes`。
- `model_checksum`：与 `NavigationLayout::model_checksum` cross-check。
- `payload_checksum`：code 区的 FNV-1a 校验和。
- `header_checksum`：文件头自身的校验和（计算时 `header_checksum` 字段置 0，见 7.3.6）。
- `reserved[4]`：32 字节预留。

`sizeof(CodeHeader) == 120`：8+4+4+4+4+4+4+4+4+4 = 44 字节前部 + 6 个 u64 = 48 字节中部 + 4 个 u64 = 32 字节尾部 reserved = 44+48+32 = 124？重新数：magic(8) + version(4) + header_bytes(4) + endian_marker(4) + memory_node(4) + quantizer_kind(4) + code_bytes(4) + node_size(4) + reserved0(4) = 40 字节，然后 entry_count(8) + remote_offset(8) + payload_bytes(8) + model_checksum(8) + payload_checksum(8) + header_checksum(8) = 48 字节，最后 reserved[4] = 32 字节。40+48+32 = 120。✓

### 7.2.9 RemotePtr 5 字节编码与 `kCompactPointerBytes`

`RemotePtr`（`src/remote_pointer.hh:7-29`）是 8 字节：

```cpp
// remote_pointer.hh:7-22
struct RemotePtr {
  static constexpr size_t SIZE = sizeof(u64);
  u64 raw_address{};  // [ memory node (16b) | byte offset (48b) ]

  u32 memory_node() const { return raw_address >> 48; }
  u64 byte_offset() const { return (raw_address << 16) >> 16; }
  ...
};
```

`raw_address` 高 16 位是 memory node，低 48 位是 byte offset。48 位 offset 上限是 `1 << 48 = 256 TiB`，足够任何 RDMA 内存段。16 位 memory node 上限是 65536 个节点。

但 compact graph record 里**每个邻居指针只占 5 字节**（`kCompactPointerBytes = 5`），不是 8 字节。5 字节 = 40 位，怎么编码 16+48=64 位信息？答案是：**shard 内邻居指针省掉 memory node，跨 shard 邻居指针用 `graph_shard_bits` 位编码 shard index**。

具体编码规则由 `NavigationLayout` 的两个字段决定：
- `graph_shard_bits`：shard 编号的位宽，由 `shard_bits_for(num_shards)` 推导（7.3.1）。比如 4 个 shard → 2 位，8 个 shard → 3 位。
- `graph_pointer_bytes = 5`：固定 5 字节。

5 字节 = 40 位，扣除 `graph_shard_bits` 位后剩下 `40 - graph_shard_bits` 位编码 byte offset。因为 `node_base_offset = 16` 且 `node_stride` 是固定的，offset 实际上由 (shard, slot) 决定，slot 可以用 `40 - graph_shard_bits` 位编码，足够 `2^(40-graph_shard_bits)` 个 slot——即使 `graph_shard_bits = 16`（上限），还有 24 位 = 1600 万 slot，远超单 shard 节点数上限 `2^30` 的实际使用量。

`validate_layout` 强制 `graph_shard_bits < 16`（7.3.1），保证 offset 至少有 24 位。具体的 5 字节编解码在第 6 课 `hot_graph.hh` 里实现，本课只定义契约常量。

### 7.2.10 `View` 与 `SynthesisOptions`

```cpp
// index_format.hh:160-169
struct View {
  NavigationLayout layout{};
  std::vector<ShardRegion> shards;
  std::vector<u32> entry_points;
};

struct SynthesisOptions {
  u32 entry_points{};
  u64 seed{1234};
};
```

`View` 是 `synthesize_distributed_view` 的输出，也是 GPU 引擎启动时持有的"索引视图"。它包含：
- `layout`：全局导航参数。
- `shards`：每个 shard 的远端布局描述。
- `entry_points`：冷启动 entry point 的全局 ordinal 列表（medoid + anchor + 随机采样填充）。

`SynthesisOptions`：
- `entry_points`：期望的 entry point 数，0 表示用 metadata 里的 `navigation_entry_points`（默认 256）。
- `seed`：随机采样 entry point 的种子，默认 1234。

### 7.2.11 `static_assert` 与 magic 常量守护

```cpp
// index_format.hh:145-153
static_assert(sizeof(ShardRegion) == 88);
static_assert(sizeof(StorageControlBlock) == 640);
static_assert(sizeof(StorageControlBlock) <= kStorageControlBytes);
static_assert(sizeof(StorageRouteSlot) == 48);
static_assert(sizeof(StorageRoutePublication) == 448);
static_assert(kStorageRoutePublicationOffset >= sizeof(StorageControlBlock));
static_assert(kStorageRoutePublicationOffset +
                sizeof(StorageRoutePublication) <= kStorageControlBytes);
static_assert(sizeof(CodeHeader) == 120);
```

这 8 条 `static_assert` 是 schema-15 的**编译期契约**：

1. `ShardRegion == 88`：shard 描述定长，任何字段加减都会编译失败。
2. `StorageControlBlock == 640`：控制页主结构定长。
3. `StorageControlBlock <= 4096`：主结构能放进控制页。
4. `StorageRouteSlot == 48`：route slot 定长。
5. `StorageRoutePublication == 448`：route 快照定长。
6. `kStorageRoutePublicationOffset(1024) >= sizeof(StorageControlBlock)(640)`：route 快照区在主结构之后，不重叠。
7. `1024 + 448 = 1472 <= 4096`：route 快照区整体在控制页内，还有 2624 字节预留。
8. `CodeHeader == 120`：PQ code sidecar 文件头定长。

这些断言**在编译期就锁死了字节布局**。任何想改字段的人都会立刻撞到编译错误，必须同时改 `static_assert`——这是一种"格式变更必须显式"的代码审查触发器。

### 7.2.12 对外 API 声明

```cpp
// index_format.hh:155-191
u64 storage_route_body_checksum(const StorageRoutePublication& publication);
bool validate_storage_route_publication(
  const StorageRoutePublication& publication, u32 expected_shard,
  std::string* error = nullptr);

...

u64 align_up(u64 value, u64 alignment);
u64 checksum64(const byte_t* data, size_t bytes);
u64 checksum64_update(u64 state, const byte_t* data, size_t bytes);
u64 checksum64_initial();

bool validate_layout(const NavigationLayout& layout, std::string* error = nullptr);
bool validate_view(const View& view, std::string* error = nullptr);
bool synthesize_distributed_view(
  const std::filesystem::path& index_prefix, View& view,
  const SynthesisOptions& options = {},
  bool* used_anchor_entry_points = nullptr,
  std::string* error = nullptr);

bool validate_code_header(const CodeHeader& header, std::string* error = nullptr);
bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error = nullptr);
bool write_code_header(std::ostream& output, const CodeHeader& header,
                       std::string* error = nullptr);

bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer);
bool remote_to_ordinal(const View& view, RemotePtr pointer, u32& ordinal);
```

API 分五组：
1. **route 快照校验**：`storage_route_body_checksum` + `validate_storage_route_publication`。
2. **通用工具**：`align_up`、`checksum64` 系列。
3. **layout/view 校验**：`validate_layout`、`validate_view`。
4. **view 合成**：`synthesize_distributed_view`（核心入口）。
5. **code sidecar IO**：`validate_code_header`、`read_code_header`、`write_code_header`。
6. **双向映射**：`ordinal_to_remote`、`remote_to_ordinal`。

所有函数都走 `bool + std::string* error` 的 fail-stop 模式，不抛异常（`synthesize_distributed_view` 内部 try/catch 把异常转成 error，7.3.4 详述）。

## 7.3 `index_format.cc`：实现逐函数讲解

文件路径：`/home/xjs/experiment/dvstor/src/gpu_search/index_format.cc`

### 7.3.1 namespace 内的常量与工具函数

文件开头先把 `storage_route_body_checksum` 和 `validate_storage_route_publication` 放在匿名 namespace 之前（因为头文件声明了它们，需要外部链接）。然后才是匿名 namespace：

```cpp
// index_format.cc:70-88
namespace {

constexpr u64 kChecksumOffset = 1469598103934665603ULL;
constexpr u64 kChecksumPrime = 1099511628211ULL;
constexpr u64 kRemoteOffsetLimit = 1ull << 48;

void set_error(std::string* error, const std::string& value) {
  if (error != nullptr) *error = value;
}

u32 shard_bits_for(u32 shard_count) {
  u32 bits = 0;
  u32 capacity = 1;
  while (capacity < shard_count && bits < 31) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}
```

- `kChecksumOffset = 1469598103934665603` 和 `kChecksumPrime = 1099511628211`：这是 FNV-1a 64 位的标准偏移基与素数。`checksum64_initial()` 返回 `kChecksumOffset`，`checksum64_update` 每字节 `state ^= data[i]; state *= prime`。dvstor 的所有 checksum（code payload、code header、route body）都用这个算法。
- `kRemoteOffsetLimit = 1 << 48`：与 `RemotePtr` 的 48 位 offset 上限对齐。`validate_view` 用它检查 shard 区不溢出（7.3.2）。
- `set_error`：把错误字符串写进 `error` 指针（如果非空）。这是 fail-stop 模式的标准 helper。
- `shard_bits_for`：算 `shard_count` 需要多少位编码。比如 `shard_count=4` → capacity 从 1 涨到 2（bits=1）、再涨到 4（bits=2），返回 2。`bits < 31` 是防御性上限，避免 `shard_count=0` 或负数时死循环。注意它返回的是**最少需要的位数**，不是 `ceil(log2(shard_count))`——比如 `shard_count=3` 也返回 2（capacity=4 >= 3）。`validate_layout` 会强制 `graph_shard_bits == shard_bits_for(num_shards)`（7.3.1）。

#### `mix64`：entry point 采样的哈希混合

```cpp
// index_format.cc:90-95
u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}
```

这是 Stafford 的 Mix13 变种（基于 splitmix64），用于把 `(seed, shard, sample_rank)` 混合成均匀分布的 u64，再 `% node_count` 取 slot。用于 7.3.4 的 anchor-free fallback 采样。

#### `read_exact_or_throw`：定长读取

```cpp
// index_format.cc:97-103
void read_exact_or_throw(std::istream& input, void* destination, size_t bytes,
                         const std::filesystem::path& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}
```

读不满就抛异常。这是 anchor sidecar 解析（7.3.4）和 `synthesize_distributed_view` 内部 try/catch 的基础——所有解析错误都转成 `runtime_error`，最后被 catch 成 `error` 字符串。

### 7.3.2 `validate_layout`：全局导航参数校验

```cpp
// index_format.cc:197-225
bool validate_layout(const NavigationLayout& layout, std::string* error) {
  if (layout.dim == 0 || layout.graph_degree == 0 || layout.num_shards == 0 ||
      layout.num_nodes == 0 || layout.num_nodes >= (1ull << 30) ||
      layout.num_nodes > std::numeric_limits<u32>::max() || layout.base_generation == 0) {
    set_error(error, "GPU navigation layout has invalid dimensions");
    return false;
  }
  if (layout.quantizer_kind != static_cast<u32>(QuantizerKind::opq_pq) ||
      layout.pq_bits != 8 || layout.pq_subquantizers == 0 ||
      layout.dim % layout.pq_subquantizers != 0 ||
      layout.code_bytes != layout.pq_subquantizers || layout.model_checksum == 0) {
    set_error(error, "GPU navigation layout has an invalid PQ configuration");
    return false;
  }
  if (layout.graph_pointer_bytes != kCompactPointerBytes ||
      layout.graph_entry_bytes <
        8 + static_cast<u64>(layout.graph_degree) * kCompactPointerBytes ||
      layout.graph_entry_bytes > kGraphCacheLineBytes ||
      layout.graph_shard_bits != shard_bits_for(layout.num_shards) ||
      layout.graph_shard_bits >= 16) {
    set_error(error, "GPU navigation requires one-cache-line compact graph records");
    return false;
  }
  if (layout.medoid_ordinal >= layout.num_nodes) {
    set_error(error, "GPU navigation layout has an invalid medoid");
    return false;
  }
  return true;
}
```

四组校验：

**第一组（基本维度）**：`dim`、`graph_degree`、`num_shards`、`num_nodes` 都非零；`num_nodes < 2^30`（GPU 端 30 位 ordinal 上限）；`num_nodes <= u32::max`（虽然 `num_nodes` 是 u64，但 ordinal 在 kernel 端是 u32）；`base_generation != 0`。

**第二组（PQ 配置）**：`quantizer_kind == opq_pq`；`pq_bits == 8`（硬约束，GPU LUT 实现）；`pq_subquantizers != 0`；`dim % pq_subquantizers == 0`（每段等分）；`code_bytes == pq_subquantizers`（因为 8 比特 = 1 字节）；`model_checksum != 0`。

**第三组（compact graph 契约）**：`graph_pointer_bytes == 5`；`graph_entry_bytes >= 8 + R*5`（8 字节 header + R 个 5 字节指针，与第 6 课 `VamanaNode` 的布局对应）；`graph_entry_bytes <= 512`（一个 cache line）；`graph_shard_bits == shard_bits_for(num_shards)`（位宽精确匹配 shard 数）；`graph_shard_bits < 16`（5 字节 = 40 位，至少留 24 位给 offset）。

**第四组（medoid）**：`medoid_ordinal < num_nodes`。

任一组失败都 fail-stop。注意这里**没有 cross-shard 一致性检查**——那是 `validate_view` 的职责。

### 7.3.3 `validate_view`：跨 shard 布局一致性校验

```cpp
// index_format.cc:227-285
bool validate_view(const View& view, std::string* error) {
  if (view.shards.size() != view.layout.num_shards || view.entry_points.empty() ||
      view.entry_points.size() > kMaxEntryPoints || !validate_layout(view.layout, error)) {
    if (error != nullptr && error->empty()) *error = "GPU navigation view cardinality mismatch";
    return false;
  }
  u64 next_ordinal = 0;
  for (size_t shard_index = 0; shard_index < view.shards.size(); ++shard_index) {
    const ShardRegion& shard = view.shards[shard_index];
    const bool node_range_overflows = shard.node_base_offset > kRemoteOffsetLimit ||
      (shard.node_stride != 0 && shard.node_count >
       (kRemoteOffsetLimit - shard.node_base_offset) / shard.node_stride);
    const bool graph_range_overflows = shard.graph_base_offset > kRemoteOffsetLimit ||
      (view.layout.graph_entry_bytes != 0 && shard.node_count >
       (kRemoteOffsetLimit - shard.graph_base_offset) / view.layout.graph_entry_bytes);
    const bool code_range_overflows = shard.code_remote_offset > kRemoteOffsetLimit ||
      shard.code_bytes > kRemoteOffsetLimit - shard.code_remote_offset;
    const u64 node_end = node_range_overflows ? kRemoteOffsetLimit :
      shard.node_base_offset + shard.node_count * shard.node_stride;
    const u64 graph_end = graph_range_overflows ? kRemoteOffsetLimit :
      shard.graph_base_offset + shard.node_count * view.layout.graph_entry_bytes;
    if (shard.memory_node != shard_index || shard.ordinal_base != next_ordinal ||
        shard.node_count == 0 || shard.node_base_offset != kNodeBaseOffset ||
        shard.node_stride == 0 || shard.graph_base_offset == 0 ||
        shard.dynamic_base_offset == 0 || shard.control_remote_offset == 0 ||
        shard.dynamic_record_bytes == 0 ||
        shard.dynamic_hot_offset == 0 || node_range_overflows || graph_range_overflows ||
        code_range_overflows || node_end > shard.graph_base_offset ||
        graph_end > shard.control_remote_offset ||
        shard.dynamic_hot_offset < shard.node_stride ||
        shard.dynamic_hot_offset > shard.dynamic_record_bytes ||
        view.layout.graph_entry_bytes >
          shard.dynamic_record_bytes - shard.dynamic_hot_offset ||
        shard.dynamic_code_offset <
          shard.dynamic_hot_offset + view.layout.graph_entry_bytes ||
        shard.dynamic_code_offset > shard.dynamic_record_bytes ||
        view.layout.code_bytes >
          shard.dynamic_record_bytes - shard.dynamic_code_offset ||
        shard.code_remote_offset !=
          shard.control_remote_offset + kStorageControlBytes ||
        shard.dynamic_base_offset < shard.code_remote_offset + shard.code_bytes ||
        shard.code_bytes != shard.node_count * view.layout.code_bytes) {
      set_error(error, "GPU navigation layout contains an invalid shard region");
      return false;
    }
    next_ordinal += shard.node_count;
  }
  if (next_ordinal != view.layout.num_nodes) {
    set_error(error, "GPU navigation shard ranges do not cover all nodes");
    return false;
  }
  for (u32 entry : view.entry_points) {
    if (entry >= view.layout.num_nodes) {
      set_error(error, "GPU navigation layout contains an invalid entry point");
      return false;
    }
  }
  return true;
}
```

这是 schema-15 最复杂的一个函数。逐段拆：

**前置校验**：`shards.size() == num_shards`；`entry_points` 非空且 `<= 512`；调 `validate_layout`。

**溢出检查**：对每个 shard 算三个 overflow flag：
- `node_range_overflows`：`node_base_offset + node_count * node_stride > 2^48`。用除法避免乘法溢出。
- `graph_range_overflows`：`graph_base_offset + node_count * graph_entry_bytes > 2^48`。
- `code_range_overflows`：`code_remote_offset + code_bytes > 2^48`。

**主校验（一个巨大的 if）**：列举所有必须成立的条件，任一不成立就 fail。条件分几类：

1. **shard 索引一致性**：`shard.memory_node == shard_index`（shard i 必须在 memory node i）；`shard.ordinal_base == next_ordinal`（ordinal 累加）。
2. **非零性**：`node_count`、`node_stride`、`graph_base_offset`、`dynamic_base_offset`、`control_remote_offset`、`dynamic_record_bytes`、`dynamic_hot_offset` 都非零。
3. **区间不重叠**：`node_end <= graph_base_offset`（fixed record 区后紧跟 graph 区，不重叠）；`graph_end <= control_remote_offset`（graph 区后是控制页）；`code_remote_offset == control_remote_offset + 4096`（PQ code 区紧跟控制页）；`dynamic_base_offset >= code_remote_offset + code_bytes`（动态区在 PQ code 区之后）。
4. **动态记录内部布局**：`dynamic_hot_offset >= node_stride`（hot 段在 fixed record 之后，且至少能容纳一条 fixed record？这里语义是 dynamic 记录里 hot 段偏移要 >= node_stride，确保 dynamic 节点的 hot 段不与 fixed record 段重叠）；`dynamic_hot_offset <= dynamic_record_bytes`；`graph_entry_bytes <= dynamic_record_bytes - dynamic_hot_offset`（hot 段能装下一条 graph 记录）；`dynamic_code_offset >= dynamic_hot_offset + graph_entry_bytes`（code 段在 hot 段之后）；`dynamic_code_offset <= dynamic_record_bytes`；`code_bytes <= dynamic_record_bytes - dynamic_code_offset`（code 段能装下一条 code）。
5. **尺寸一致**：`code_bytes == node_count * layout.code_bytes`（PQ code 区总字节数 = 节点数 × 单条 code 字节数）。

**全局覆盖**：`next_ordinal == num_nodes`（所有 shard 的节点数加起来等于全局节点数）。

**entry point 校验**：每个 entry point `< num_nodes`。

这个函数是 schema-15 的**几何契约守护者**。它确保：
- shard 之间不重叠、不留洞。
- fixed record → graph → control → code → dynamic 五个区在 RDMA 内存里顺序排列、不重叠。
- 动态记录内部的 hot/code 段布局自洽。
- 所有 offset 都在 48 位 RDMA 寻址范围内。

### 7.3.4 `storage_route_body_checksum` 与 `validate_storage_route_publication`

```cpp
// index_format.cc:16-27
u64 storage_route_body_checksum(
    const StorageRoutePublication& publication) {
  u64 checksum = checksum64_initial();
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(&publication.magic),
    offsetof(StorageRoutePublication, body_checksum) -
      offsetof(StorageRoutePublication, magic));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(publication.slots.data()),
    publication.slots.size() * sizeof(StorageRouteSlot));
  return checksum;
}
```

注意 checksum 的覆盖范围：从 `magic` 字段到 `body_checksum` 之前（用 `offsetof` 算偏移差），加上 `slots` 数组。**不覆盖 `sequence_begin` 和 `sequence_end`**——这两个字段是发布过程中的并发标记，不能进 checksum（否则写完 `sequence_end` 又得重算 checksum，破坏 seqlock 语义）。也不覆盖 `body_checksum` 自身。

```cpp
// index_format.cc:29-69
bool validate_storage_route_publication(
    const StorageRoutePublication& publication, u32 expected_shard,
    std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (publication.sequence_begin == 0 ||
      (publication.sequence_begin & 1u) != 0 ||
      publication.sequence_begin != publication.sequence_end) {
    return fail("storage route snapshot overlaps publication");
  }
  if (publication.magic != kStorageRoutePublicationMagic ||
      publication.version != kStorageRoutePublicationVersion ||
      publication.header_bytes != sizeof(StorageRoutePublication) ||
      publication.shard_id != expected_shard ||
      publication.slot_count != kStorageRouteSlots ||
      publication.code_bytes == 0 ||
      publication.code_bytes > kStorageRouteMaxCodeBytes) {
    return fail("storage route publication header mismatch");
  }
  if (publication.body_checksum !=
      storage_route_body_checksum(publication)) {
    return fail("storage route publication checksum mismatch");
  }
  for (const StorageRouteSlot& slot : publication.slots) {
    if (slot.remote_node == 0) {
      if (slot.generation == 0 && slot.id != 0) {
        return fail("storage route publication contains an invalid empty slot");
      }
      continue;
    }
    // Schema-15 immutable base nodes store generation zero; online versions
    // start at one. Both are valid canonical route representatives.
    if (static_cast<u32>(slot.remote_node >> 48) != expected_shard) {
      return fail("storage route publication contains an invalid live slot");
    }
  }
  if (error != nullptr) error->clear();
  return true;
}
```

三段校验：

**第一段（seqlock 完整性）**：`sequence_begin != 0`（0 是未初始化）；`sequence_begin` 是偶数（`& 1u == 0`）——发布完成后 begin/end 都是偶数，发布过程中 begin 自增成奇数，读者看到奇数就知道在发布中；`sequence_begin == sequence_end`。

**第二段（头部契约）**：magic、version、header_bytes、shard_id、slot_count、code_bytes 范围全部匹配。

**第三段（checksum）**：重算 body_checksum 比对。

**第四段（slot 语义）**：每个 slot 要么是空槽（`remote_node == 0`，此时 `generation==0 && id==0` 才合法——允许 `generation==0, id==0` 的"全零空槽"，但 `generation==0, id!=0` 是非法的，因为那样表示"有 id 但没 remote_node"的脏槽），要么是 live slot（`remote_node != 0`，此时 `remote_node >> 48` 必须等于 `expected_shard`——representative 必须在本 shard 内）。

注释特别指出："Schema-15 immutable base nodes store generation zero; online versions start at one."——这是 schema-15 与 schema-14 的另一个细微差别：schema-14 的 base 节点 generation 可能是任意值，schema-15 明确 base 节点 generation=0，在线插入的节点 generation 从 1 开始。两者都合法，所以这里不校验 generation 的具体值。

### 7.3.5 `align_up` 与 `checksum64` 系列

```cpp
// index_format.cc:173-195
u64 align_up(u64 value, u64 alignment) {
  if (alignment == 0) return value;
  const u64 remainder = value % alignment;
  if (remainder == 0) return value;
  if (value > std::numeric_limits<u64>::max() - (alignment - remainder)) return 0;
  return value + alignment - remainder;
}

u64 checksum64_initial() {
  return kChecksumOffset;
}

u64 checksum64_update(u64 state, const byte_t* data, size_t bytes) {
  for (size_t index = 0; index < bytes; ++index) {
    state ^= static_cast<u64>(data[index]);
    state *= kChecksumPrime;
  }
  return state;
}

u64 checksum64(const byte_t* data, size_t bytes) {
  return checksum64_update(checksum64_initial(), data, bytes);
}
```

- `align_up`：向上对齐。`alignment == 0` 直接返回（防御）；已对齐直接返回；溢出返回 0（`align_up` 返回 0 是"溢出哨兵"，调用方需要检查，但 `synthesize_distributed_view` 里 `align_up(dynamic_offsets[shard], 64)` 不会溢出，因为 dynamic_offsets 来自 metadata，远小于 `u64::max`）。
- `checksum64_initial` / `update` / `checksum64`：标准 FNV-1a 64 位。`initial` 返回 offset basis，`update` 逐字节 `state ^= data[i]; state *= prime`，`checksum64` 是 `update(initial(), data, bytes)` 的简写。

### 7.3.6 `validate_code_header` / `read_code_header` / `write_code_header`

```cpp
// index_format.cc:449-470
bool validate_code_header(const CodeHeader& header, std::string* error) {
  if (header.magic != kCodeMagic || header.version != kVersion ||
      header.header_bytes != sizeof(CodeHeader) || header.endian_marker != kEndianMarker) {
    set_error(error, "invalid GPU PQ code sidecar header");
    return false;
  }
  if (header.quantizer_kind != static_cast<u32>(QuantizerKind::opq_pq) ||
      header.entry_count == 0 || header.node_size == 0 || header.remote_offset == 0 ||
      header.code_bytes == 0 || header.model_checksum == 0 ||
      header.payload_bytes != header.entry_count * header.code_bytes) {
    set_error(error, "invalid GPU PQ code sidecar dimensions");
    return false;
  }
  CodeHeader copy = header;
  const u64 stored_checksum = copy.header_checksum;
  copy.header_checksum = 0;
  if (checksum64(reinterpret_cast<const byte_t*>(&copy), sizeof(copy)) != stored_checksum) {
    set_error(error, "GPU PQ code sidecar header checksum mismatch");
    return false;
  }
  return true;
}
```

三段校验：

**第一段（magic/version/endian）**：magic、version、header_bytes、endian_marker 全部匹配。

**第二段（维度）**：`quantizer_kind == opq_pq`；`entry_count`、`node_size`、`remote_offset`、`code_bytes`、`model_checksum` 都非零；`payload_bytes == entry_count * code_bytes`（一致性）。

**第三段（header checksum）**：把 `header_checksum` 字段置 0，重算整个 header 的 FNV-1a，与存储的 checksum 比对。注意这里**copy 一份**再清零，不修改原 header——因为 `validate_code_header` 可能被调用方用 const 引用调用（虽然签名是值传递，但 copy 后清零是防御性的）。

```cpp
// index_format.cc:472-486
bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(error, "GPU PQ code sidecar does not exist: " + path.string());
    return false;
  }
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || !validate_code_header(header, error)) return false;
  if (std::filesystem::file_size(path) != sizeof(CodeHeader) + header.payload_bytes) {
    set_error(error, "GPU PQ code sidecar file size mismatch: " + path.string());
    return false;
  }
  return true;
}
```

`read_code_header` 三步：打开文件读 120 字节 header；`validate_code_header` 校验；检查文件总大小 == `sizeof(CodeHeader) + payload_bytes`（防截断/拼接）。

```cpp
// index_format.cc:488-506
bool write_code_header(std::ostream& output, const CodeHeader& source,
                       std::string* error) {
  CodeHeader header = source;
  header.magic = kCodeMagic;
  header.version = kVersion;
  header.header_bytes = sizeof(CodeHeader);
  header.endian_marker = kEndianMarker;
  header.header_checksum = 0;
  header.header_checksum = checksum64(
    reinterpret_cast<const byte_t*>(&header), sizeof(header));
  if (!validate_code_header(header, error)) return false;
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!output.good()) {
    set_error(error, "failed to write GPU PQ code sidecar header");
    return false;
  }
  return true;
}
```

`write_code_header` 强制覆盖 magic/version/header_bytes/endian_marker（即使调用方传错也纠正），然后：先置 `header_checksum = 0`，算 checksum，写回 `header_checksum`，再 `validate_code_header` 自校验（写之前先验证，避免写出脏 header），最后 `seekp(0)` 写到流开头。这个函数是 schema-15 PQ code sidecar 的**唯一写入点**——第 12/13 课 construction 完成后用它写出每个 shard 的 `.pq*.codes` 文件。

### 7.3.7 `synthesize_distributed_view`：从 `.meta.json` 合成 `View`

这是 `index_format.cc` 最长的函数，也是 schema-15 的核心解析入口。分段讲：

#### 头部与 schema-15 严格匹配

```cpp
// index_format.cc:287-311
bool synthesize_distributed_view(
    const std::filesystem::path& index_prefix, View& view,
    const SynthesisOptions& options, bool* used_anchor_entry_points,
    std::string* error) {
  if (used_anchor_entry_points != nullptr) *used_anchor_entry_points = false;
  try {
    const std::filesystem::path metadata_path{index_prefix.string() + ".meta.json"};
    std::ifstream metadata_input(metadata_path);
    if (!metadata_input.good()) {
      throw std::runtime_error("missing index metadata: " + metadata_path.string());
    }
    nlohmann::json metadata;
    metadata_input >> metadata;
    const std::string quantizer = metadata.value("navigation_quantizer", std::string{});
    const std::string navigation_format = metadata.value("navigation_format", std::string{});
    if (metadata.value("schema_version", 0u) != kMetadataSchemaVersion ||
        metadata.value("distance", std::string{"l2"}) != "l2" ||
        metadata.value("node_layout", std::string{}) != "plain" ||
        metadata.value("storage_format", std::string{}) != "vamana_compact_v1" ||
        (quantizer != "opq_pq" && quantizer != "opq_pq16") ||
        (navigation_format != "opq_pq_graph_v1" &&
         navigation_format != "opq_pq16_graph_v1")) {
      throw std::runtime_error(
        "GPU navigation requires schema-15 compact L2 metadata with persistent dynamic PQ codes");
    }
```

**schema-15 fail-stop 在这里**。一组严格等式：
- `schema_version == 15`：必须是 schema-15，schema-14 直接拒绝（不自动升级）。
- `distance == "l2"`：只支持 L2 距离。
- `node_layout == "plain"`：节点布局必须是 plain（不是 tiled/blocked）。
- `storage_format == "vamana_compact_v1"`：存储格式必须是 vamana compact v1（第 6 课讲过）。
- `quantizer ∈ {"opq_pq", "opq_pq16"}`：OPQ+PQ 或 OPQ+PQ16。
- `navigation_format ∈ {"opq_pq_graph_v1", "opq_pq16_graph_v1"}`：导航格式。

任一不匹配就抛异常，最后被 catch 成 error 字符串返回 false。这就是 schema-14 → 15 的"升级路径"——**没有自动升级，必须用离线工具重新构建索引**（第 29 课离线构建/迁移）。这个设计是刻意的：自动升级在分布式系统里极其危险，因为不同节点可能在不同时间看到不同 schema，导致跨 shard 不一致。fail-stop 强制运维人员显式迁移。

#### 读取 shard 数组并校验长度

```cpp
// index_format.cc:313-333
    const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
    const std::vector<u64> counts =
      metadata.at("hot_graph_entry_counts").get<std::vector<u64>>();
    const std::vector<u64> graph_offsets =
      metadata.at("hot_graph_offsets").get<std::vector<u64>>();
    const std::vector<u64> dynamic_offsets =
      metadata.at("hot_graph_dynamic_base_offsets").get<std::vector<u64>>();
    const std::vector<u64> control_offsets =
      metadata.at("storage_control_remote_offsets").get<std::vector<u64>>();
    const std::vector<u64> dynamic_node_offsets =
      metadata.at("dynamic_node_base_offsets").get<std::vector<u64>>();
    const std::vector<u64> code_offsets =
      metadata.at("navigation_code_remote_offsets").get<std::vector<u64>>();
    const std::vector<u64> code_sizes =
      metadata.at("navigation_code_region_bytes").get<std::vector<u64>>();
    if (shard_count == 0 || counts.size() != shard_count ||
        graph_offsets.size() != shard_count || dynamic_offsets.size() != shard_count ||
        control_offsets.size() != shard_count || dynamic_node_offsets.size() != shard_count ||
        code_offsets.size() != shard_count || code_sizes.size() != shard_count) {
      throw std::runtime_error("GPU navigation metadata has invalid shard arrays");
    }
```

读取 7 个长度为 `shard_count` 的数组：节点数、graph offset、dynamic offset、控制页 offset、动态节点 offset、PQ code offset、PQ code 字节数。任一数组长度不等于 `shard_count` 就 fail。这些字段对应 `service::index_metadata::Metadata`（第 8 课）的同名字段——`.meta.json` 是 construction 阶段写出的（第 12/13 课），运行时由 `index_metadata::load_metadata` 读入 `Metadata` 结构，再由 `synthesize_distributed_view` 从 `Metadata` 合成 `View`。本课直接读 JSON 是为了让 GPU 引擎不依赖 `service::index_metadata`（解耦）。

#### 合成 `NavigationLayout`

```cpp
// index_format.cc:335-352
    View synthesized;
    synthesized.layout.dim = metadata.at("dim").get<u32>();
    synthesized.layout.graph_degree = metadata.at("R").get<u32>();
    const VectorDType dtype = parse_vector_dtype(
      metadata.value("vector_data_type", std::string{"float32"}));
    synthesized.layout.vector_dtype = static_cast<u32>(dtype);
    synthesized.layout.quantizer_kind = static_cast<u32>(QuantizerKind::opq_pq);
    synthesized.layout.pq_subquantizers = metadata.at("pq_subquantizers").get<u32>();
    synthesized.layout.pq_bits = metadata.at("pq_bits").get<u32>();
    synthesized.layout.code_bytes = metadata.at("navigation_code_bytes").get<u32>();
    synthesized.layout.model_checksum = metadata.at("navigation_model_checksum").get<u64>();
    synthesized.layout.num_shards = shard_count;
    synthesized.layout.graph_entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
    synthesized.layout.graph_pointer_bytes =
      metadata.at("hot_graph_pointer_bytes").get<u32>();
    synthesized.layout.graph_shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
    synthesized.layout.base_generation = 1;
    synthesized.shards.resize(shard_count);
```

逐字段从 JSON 取值填进 `NavigationLayout`。`base_generation = 1` 是硬编码——schema-15 的 base 节点 generation 在 `View` 层面视为 1（虽然 `StorageRouteSlot` 允许 generation=0，但 `View` 合成时统一用 1，避免下游代码处理 0/1 两种情况）。`vector_dtype` 用 `parse_vector_dtype` 解析字符串（`"float32"` / `"uint8"` / `"int8"`，见 `common/vector_dtype.hh:33-44`）。

#### 合成 `ShardRegion`

```cpp
// index_format.cc:354-388
    const u64 node_stride = metadata.at("node_size").get<u64>();
    const u32 dynamic_record_bytes = metadata.at("hot_graph_dynamic_record_bytes").get<u32>();
    const u32 dynamic_hot_offset = metadata.at("hot_graph_dynamic_hot_offset").get<u32>();
    const u32 dynamic_code_offset = metadata.at("dynamic_navigation_code_offset").get<u32>();
    u64 node_count = 0;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      const u64 expected_control_offset = align_up(dynamic_offsets[shard], 64);
      const u64 expected_code_offset = expected_control_offset + kStorageControlBytes;
      const u64 expected_code_bytes = counts[shard] * synthesized.layout.code_bytes;
      if (counts[shard] == 0 || graph_offsets[shard] == 0 ||
          dynamic_offsets[shard] == 0 ||
          control_offsets[shard] != expected_control_offset ||
          code_offsets[shard] != expected_code_offset ||
          code_sizes[shard] != expected_code_bytes ||
          dynamic_node_offsets[shard] < expected_code_offset + expected_code_bytes ||
          counts[shard] >= (1ull << 30) - node_count) {
        throw std::runtime_error("GPU navigation metadata contains an invalid shard");
      }
      synthesized.shards[shard] = {
        .ordinal_base = node_count,
        .node_count = counts[shard],
        .node_base_offset = kNodeBaseOffset,
        .node_stride = node_stride,
        .graph_base_offset = graph_offsets[shard],
        .dynamic_base_offset = dynamic_node_offsets[shard],
        .control_remote_offset = control_offsets[shard],
        .code_remote_offset = code_offsets[shard],
        .code_bytes = code_sizes[shard],
        .memory_node = shard,
        .dynamic_record_bytes = dynamic_record_bytes,
        .dynamic_hot_offset = dynamic_hot_offset,
        .dynamic_code_offset = dynamic_code_offset,
      };
      node_count += counts[shard];
    }
```

对每个 shard：
- `expected_control_offset = align_up(dynamic_offsets[shard], 64)`：控制页必须 64 对齐（`StorageControlBlock` 是 `alignas(64)`）。
- `expected_code_offset = expected_control_offset + 4096`：PQ code 区紧跟控制页。
- `expected_code_bytes = counts[shard] * code_bytes`：PQ code 区字节数。

校验：`control_offsets[shard] == expected_control_offset`（控制页 offset 必须是 dynamic offset 64 对齐）；`code_offsets[shard] == expected_code_offset`；`code_sizes[shard] == expected_code_bytes`；`dynamic_node_offsets[shard] >= expected_code_offset + expected_code_bytes`（动态节点区在 PQ code 区之后）；`counts[shard] < 2^30 - node_count`（全局节点数不超 2^30）。

然后用 C++20 designated initializer 填 `ShardRegion`。`ordinal_base` 用累加的 `node_count`，`memory_node = shard`（shard i 在 memory node i）。

#### 全局节点数与 medoid

```cpp
// index_format.cc:389-401
    if (node_count != metadata.at("num_vectors").get<u64>() ||
        node_count == 0 || node_count >= (1ull << 30)) {
      throw std::runtime_error("GPU navigation metadata has an invalid node count");
    }
    synthesized.layout.num_nodes = node_count;

    const auto& medoid = metadata.at("medoid");
    const RemotePtr medoid_pointer{
      medoid.at("memory_node").get<u32>(), medoid.at("offset").get<u64>()};
    if (!remote_to_ordinal(synthesized, medoid_pointer,
                           synthesized.layout.medoid_ordinal)) {
      throw std::runtime_error("GPU navigation metadata has an invalid medoid");
    }
```

- 累加的 `node_count` 必须等于 metadata 的 `num_vectors`。
- `node_count` 非零且 `< 2^30`。
- medoid 从 metadata 的 `(memory_node, offset)` 反解成 ordinal，用 `remote_to_ordinal`（7.3.8）。如果反解失败（medoid 不在任何 shard 内）就 fail。

#### entry point 合成：anchor + 随机采样 + 顺序兜底

```cpp
// index_format.cc:403-435
    const u32 requested_entry_points = options.entry_points == 0
      ? metadata.value("navigation_entry_points", 256u) : options.entry_points;
    if (requested_entry_points == 0 || requested_entry_points > kMaxEntryPoints) {
      throw std::runtime_error("GPU entry-point count must be in [1, 512]");
    }
    const u32 target = static_cast<u32>(std::min<u64>(requested_entry_points, node_count));
    std::unordered_set<u32> selected;
    selected.insert(synthesized.layout.medoid_ordinal);
    synthesized.entry_points.push_back(synthesized.layout.medoid_ordinal);
    const bool used_anchors = append_anchor_entry_points(
      index_prefix, synthesized.layout.dim, dtype,
      metadata.at("vector_bytes").get<u32>(), synthesized, target,
      selected, synthesized.entry_points);
    // The anchor-free fallback must not fill the table from the first shard
    // before later shards get a chance to contribute.  Walk shards at every
    // sample rank so the fixed entry set remains balanced even when the
    // requested count is smaller than one shard's sampling budget.
    const u32 quota = (target + shard_count - 1) / shard_count;
    for (u32 sample = 0; sample < quota * 16 &&
         synthesized.entry_points.size() < target; ++sample) {
      for (u32 shard = 0; shard < shard_count &&
           synthesized.entry_points.size() < target; ++shard) {
        const u64 slot = mix64(options.seed ^
          (static_cast<u64>(shard) << 32) ^ sample) % counts[shard];
        const u32 ordinal = static_cast<u32>(
          synthesized.shards[shard].ordinal_base + slot);
        if (selected.insert(ordinal).second) synthesized.entry_points.push_back(ordinal);
      }
    }
    for (u32 ordinal = 0; synthesized.entry_points.size() < target &&
         ordinal < node_count; ++ordinal) {
      if (selected.insert(ordinal).second) synthesized.entry_points.push_back(ordinal);
    }
```

三级填充：

**第 0 级（medoid）**：medoid ordinal 永远是第一个 entry point。

**第 1 级（anchor 侧）**：调 `append_anchor_entry_points`（7.3.9）从 `.anchors` 文件读 anchor 向量，按 rank 轮转填入。

**第 2 级（随机采样）**：如果 anchor 不够，用 `mix64(seed ^ (shard << 32) ^ sample)` 哈希采样。注释解释了为什么外层循环是 `sample`、内层是 `shard`：避免先填满第一个 shard 再填第二个，保证 entry point 在 shard 间均衡。`quota = ceil(target / shard_count)` 是每个 shard 应贡献的数量上限，`quota * 16` 是采样轮数上限（16 倍冗余应对去重失败）。

**第 3 级（顺序兜底）**：如果随机采样还不够（极端情况，比如 node_count < target），从 ordinal 0 开始顺序填。

`selected` 是去重集合，保证每个 ordinal 只进一次。

#### 最终校验与返回

```cpp
// index_format.cc:436-447
    std::string validation_error;
    if (!validate_view(synthesized, &validation_error)) {
      throw std::runtime_error(validation_error);
    }
    if (used_anchor_entry_points != nullptr) *used_anchor_entry_points = used_anchors;
    view = std::move(synthesized);
    return true;
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
    return false;
  }
}
```

合成完调 `validate_view` 做几何契约校验（7.3.2）。通过后 `std::move` 给输出参数。整个函数体包在 try/catch 里，任何异常都转成 error 字符串返回 false——这是 fail-stop 模式的标准做法。

### 7.3.8 `append_anchor_entry_points`：anchor 侧入口点合成

```cpp
// index_format.cc:105-169
bool append_anchor_entry_points(
    const std::filesystem::path& prefix, u32 dim, VectorDType dtype,
    u32 vector_bytes, const View& view, u32 target,
    std::unordered_set<u32>& selected, std::vector<u32>& entry_points) {
  const std::filesystem::path path = index_path::anchor_file(prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) return false;
  vamana::anchor::Header header;
  read_exact_or_throw(input, &header, sizeof(header), path);
  if (header.magic != vamana::anchor::kMagic ||
      header.version != vamana::anchor::kVersion || header.dim != dim ||
      header.shard_count != view.shards.size() ||
      header.vector_dtype != static_cast<u32>(dtype) ||
      header.vector_bytes != vector_bytes || header.total_anchors > (1u << 24)) {
    throw std::runtime_error("invalid anchor sidecar for GPU entry points: " + path.string());
  }
```

打开 `.anchors` 文件（`index_path::anchor_file(prefix)` = `prefix + ".anchors"`，见 `common/index_path.hh:36-38`）。读 `vamana::anchor::Header`（第 6 课讲过，magic="ADLANCH1"、version=1）。校验 magic、version、dim、shard_count、vector_dtype、vector_bytes 都与 `View` 一致；`total_anchors <= 2^24`（防御上限）。

```cpp
// index_format.cc:121-152
  std::vector<std::vector<u32>> anchor_ordinals(view.shards.size());
  std::vector<f32> shard_centroid(dim);
  std::vector<byte_t> vector(vector_bytes);
  u64 loaded = 0;
  for (u32 shard = 0; shard < view.shards.size(); ++shard) {
    vamana::anchor::ShardHeader shard_header;
    read_exact_or_throw(input, &shard_header, sizeof(shard_header), path);
    if (shard_header.shard != shard ||
        shard_header.anchor_count > header.anchors_per_shard ||
        loaded + shard_header.anchor_count > header.total_anchors) {
      throw std::runtime_error("invalid anchor shard for GPU entry points: " + path.string());
    }
    read_exact_or_throw(input, shard_centroid.data(),
                        shard_centroid.size() * sizeof(f32), path);
    anchor_ordinals[shard].reserve(shard_header.anchor_count);
    for (u32 index = 0; index < shard_header.anchor_count; ++index) {
      vamana::anchor::EntryHeader entry;
      read_exact_or_throw(input, &entry, sizeof(entry), path);
      read_exact_or_throw(input, vector.data(), vector.size(), path);
      const RemotePtr pointer{entry.rptr_raw};
      u32 ordinal = 0;
      if (pointer.is_null() || pointer.memory_node() != shard ||
          !remote_to_ordinal(view, pointer, ordinal)) {
        throw std::runtime_error("anchor points outside its static GPU shard");
      }
      anchor_ordinals[shard].push_back(ordinal);
      ++loaded;
    }
  }
  if (loaded != header.total_anchors) {
    throw std::runtime_error("anchor sidecar count mismatch for GPU entry points");
  }
```

逐 shard 读：`ShardHeader`（shard 编号 + anchor 数）→ shard centroid（`dim` 个 f32）→ 每个 anchor 的 `EntryHeader`（rptr_raw + id + degree）+ 向量本体（`vector_bytes` 字节）。

每个 anchor 的 `rptr_raw` 解析成 `RemotePtr`，校验：非空、`memory_node == shard`（anchor 必须在它声明的 shard 内）、`remote_to_ordinal` 能反解（offset 在 shard 范围内且对齐 `node_stride`）。任一失败就抛"anchor points outside its static GPU shard"。这个校验很重要——它确保 anchor 侧文件与 `View` 的 shard 布局一致，不会有 anchor 指向错误的 shard。

最后 `loaded == total_anchors` 校验总数。

```cpp
// index_format.cc:153-168
  bool appended = false;
  for (u32 rank = 0; entry_points.size() < target; ++rank) {
    bool have_rank = false;
    for (u32 shard = 0; shard < anchor_ordinals.size() &&
         entry_points.size() < target; ++shard) {
      if (rank >= anchor_ordinals[shard].size()) continue;
      have_rank = true;
      const u32 ordinal = anchor_ordinals[shard][rank];
      if (selected.insert(ordinal).second) {
        entry_points.push_back(ordinal);
        appended = true;
      }
    }
    if (!have_rank) break;
  }
  return appended;
}
```

rank-轮转填充：外层 `rank`，内层 `shard`。每个 rank 取每个 shard 的第 rank 个 anchor。这保证 entry point 在 shard 间均衡（与第 2 级随机采样的均衡思路一致）。`have_rank` 标记本 rank 是否还有 anchor，全空就退出。`selected` 去重，`appended` 标记是否真的加了新 entry point。

返回值 `appended` 传给 `used_anchor_entry_points` 输出参数，让调用方知道是否用了 anchor（影响第 17 课 kernel 启动时的 telemetry）。

### 7.3.9 `ordinal_to_remote` / `remote_to_ordinal`：双向映射

```cpp
// index_format.cc:508-520
bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer) {
  if (ordinal >= view.layout.num_nodes) return false;
  const auto it = std::upper_bound(
    view.shards.begin(), view.shards.end(), ordinal,
    [](u32 value, const ShardRegion& shard) { return value < shard.ordinal_base; });
  if (it == view.shards.begin()) return false;
  const ShardRegion& shard = *(it - 1);
  const u64 slot = static_cast<u64>(ordinal) - shard.ordinal_base;
  if (slot >= shard.node_count) return false;
  pointer = RemotePtr{shard.memory_node,
    shard.node_base_offset + slot * shard.node_stride};
  return true;
}
```

`ordinal_to_remote`：全局 ordinal → `RemotePtr`。
- `ordinal >= num_nodes` → fail。
- `upper_bound` 找第一个 `ordinal_base > ordinal` 的 shard，前一个 shard 就是 ordinal 所在的 shard。因为 shard 按 `ordinal_base` 升序排列（`validate_view` 强制），二分查找正确。
- `it == begin()` 不可能（shard 0 的 `ordinal_base = 0`），但防御性检查。
- `slot = ordinal - shard.ordinal_base`，`slot >= node_count` → fail。
- 构造 `RemotePtr{memory_node, node_base_offset + slot * node_stride}`。

```cpp
// index_format.cc:522-532
bool remote_to_ordinal(const View& view, RemotePtr pointer, u32& ordinal) {
  if (pointer.is_null() || pointer.memory_node() >= view.shards.size()) return false;
  const ShardRegion& shard = view.shards[pointer.memory_node()];
  if (pointer.byte_offset() < shard.node_base_offset || shard.node_stride == 0) return false;
  const u64 relative = pointer.byte_offset() - shard.node_base_offset;
  if (relative % shard.node_stride != 0) return false;
  const u64 slot = relative / shard.node_stride;
  if (slot >= shard.node_count || shard.ordinal_base + slot >= (1ull << 30)) return false;
  ordinal = static_cast<u32>(shard.ordinal_base + slot);
  return true;
}
```

`remote_to_ordinal`：`RemotePtr` → 全局 ordinal。逆向操作。
- 空指针或 `memory_node >= shards.size()` → fail。
- 取 `shards[memory_node]`（因为 `memory_node == shard_index`，可以直接索引）。
- `byte_offset < node_base_offset` → fail（offset 在 fixed record 区之前）。
- `(byte_offset - node_base_offset) % node_stride != 0` → fail（offset 必须对齐到 record 步长）。
- `slot = (byte_offset - node_base_offset) / node_stride`，`slot >= node_count` → fail。
- `ordinal_base + slot < 2^30`（30 位 ordinal 上限）。
- 输出 `ordinal = ordinal_base + slot`。

这两个函数是 GPU kernel 跨 shard 跳转的基础——kernel 读到一条 compact graph record，里面的 5 字节邻居指针先解码成 (shard, slot)，再调 `ordinal_to_remote` 得到 `RemotePtr` 发起下一次 RDMA 读。不过 kernel 端用的是 device 版本（第 17/20 课），CPU 侧这两个函数主要供 construction（第 12/13 课）和 anchor 解析（7.3.8）使用。

## 7.4 schema-14 → 15 升级路径（与 fail-stop 哲学）

schema-15 与 schema-14 的差别，从本课代码看，体现在三处：

### 7.4.1 `.meta.json` 层面：严格等式而非范围

`synthesize_distributed_view` 在 `index_format.cc:302-311` 用一组严格等式校验 metadata：

```cpp
if (metadata.value("schema_version", 0u) != kMetadataSchemaVersion ||  // 必须 == 15
    metadata.value("distance", std::string{"l2"}) != "l2" ||
    metadata.value("node_layout", std::string{}) != "plain" ||
    metadata.value("storage_format", std::string{}) != "vamana_compact_v1" ||
    (quantizer != "opq_pq" && quantizer != "opq_pq16") ||
    (navigation_format != "opq_pq_graph_v1" &&
     navigation_format != "opq_pq16_graph_v1")) {
  throw std::runtime_error(
    "GPU navigation requires schema-15 compact L2 metadata with persistent dynamic PQ codes");
}
```

schema-14 的 metadata（`schema_version=14`）会直接被拒绝。**没有自动升级代码**——`index_format.cc` 里没有任何 `if (schema_version == 14) { migrate(); }` 的逻辑。这是刻意的 fail-stop：

- 自动升级需要处理旧字段缺失、旧布局不对齐、旧 checksum 算法等大量边界情况，任何一处出错都可能导致跨 shard 不一致。
- 分布式系统里，不同节点在不同时间升级，自动升级会让"老节点看到 schema-14、新节点看到 schema-15"的窗口期出现 mixed-schema 查询，极难调试。
- 显式迁移（用离线工具重建索引，第 29 课）虽然麻烦，但保证了 schema 切换的原子性。

对比 `service::index_metadata::load_metadata`（`src/service/index_metadata.cc:32`）：

```cpp
metadata.schema_version = json.value("schema_version", 1u);
```

`load_metadata` **不校验** schema_version，只是读进来。校验职责在 `synthesize_distributed_view`——这是解耦设计：`service::index_metadata` 只负责"把 JSON 读成结构体"，`gpu_search::format` 负责"判断这个结构体能不能被 GPU 引擎消费"。第 8 课会详讲 `load_metadata`。

### 7.4.2 控制页层面：version 2 的 reclaim ACK 数组

`StorageControlBlock::version = 2`（`index_format.hh:27`）。version 2 相比 version 1 加的就是末尾的 `reclaim_ack_sequences[64]` 数组（`index_format.hh:97`）。这个数组让存储节点能精确知道每个 compute client 已经看到了哪次回收，从而安全地复用动态节点槽位（第 16 课 RCU 详述）。

schema-14 的控制页是 version 1，没有这个数组，回收只能保守等所有 client 超时。schema-15 升级到 version 2，但**控制页是运行时结构，不是磁盘契约**——存储节点重启时重新初始化控制页，所以从 version 1 到 version 2 不需要磁盘迁移，只需要存储节点二进制升级。`StorageControlBlock` 的 `header_bytes` 字段让老版本代码能跳过未知的尾部字段，但 dvstor 的部署模型是"全集群同时升级"，所以这个前向兼容只是防御性的。

### 7.4.3 路由快照层面：全新的 version 1

`StorageRoutePublication::version = 1`（`index_format.hh:35`）。这是 schema-15 **新增**的运行时结构，schema-14 没有。它塞在 4 KiB 控制页的 offset 1024 处（`kStorageRoutePublicationOffset`），不占用任何磁盘契约空间。注释（`index_format.hh:29-32`）明确：

> The route publication lives in the unused tail of the existing 4 KiB storage control page. It is runtime metadata, not an on-disk index record, so adding it neither changes schema-15 nor moves any dynamic node offset.

所以 schema-14 → 15 在路由快照层面的"升级"是**新增**，不是迁移。schema-14 的控制页 offset 1024 之后是空闲的，schema-15 把 route publication 塞进去，老代码不读这片区域，新代码读它——前向兼容自然成立。

### 7.4.4 节点 generation 语义

`validate_storage_route_publication` 的注释（`index_format.cc:61-62`）：

> Schema-15 immutable base nodes store generation zero; online versions start at one. Both are valid canonical route representatives.

schema-15 明确了 base 节点 generation=0、在线节点 generation>=1 的语义。schema-14 没有这个明确约定。这个差别不影响磁盘布局（generation 字段一直都在），但影响 route 快照的校验逻辑——schema-15 接受 generation=0 和 generation>=1 两种 representative，不把它们当成错误。

## 7.5 关键数据结构与字节级布局图

### 7.5.1 计算节点文件 vs 存储节点文件

dvstor 是存算分离的，索引文件在两类节点上的分布不同：

| 文件/结构 | 计算节点 | 存储节点 | 说明 |
|---|---|---|---|
| `<prefix>.meta.json` | 读（合成 `View`） | 读（启动校验） | schema-15 全局元数据，第 8 课深讲 |
| `<prefix>.anchors` | 读（合成 entry points） | 不读 | anchor 侧文件，第 6 课讲过格式 |
| `<prefix>.pq<M>` | 读（上传 GPU） | 不读 | PQ 模型，第 9 课讲过 |
| `<prefix>_node<i>_of<N>.dat` | 不读 | **加载进 RDMA 内存** | fixed record + compact graph，第 6 课讲过 |
| `<prefix>_node<i>_of<N>.pq<M>.codes` | 读（校验 header） | **加载进 RDMA 内存** | PQ code sidecar |
| `<prefix>_node<i>_of<N>.idmap` | 不读 | 读（启动校验） | ID 映射，第 6 课讲过 |
| RDMA 内存段（per shard） | 不持有 | **持有并导出** | fixed record → graph → control → code → dynamic 五区 |
| `StorageControlBlock` | 远程 RDMA 读 | 本地写 | 4 KiB 控制页主结构 |
| `StorageRoutePublication` | 远程 RDMA 读 | 本地写 | 4 KiB 控制页 offset 1024 处的 route 快照 |

计算节点通过 `synthesize_distributed_view` 从 `.meta.json` + `.anchors` 合成 `View`，`View` 描述了"远端 RDMA 内存怎么布局"。GPU kernel 持有 `View` 的 device 镜像，通过 RDMA 读远端存储节点的内存段。

存储节点启动时加载 `.dat` / `.pq*.codes` / `.idmap` 进 RDMA 内存段，初始化 `StorageControlBlock` 和 `StorageRoutePublication`，然后等计算节点的 RDMA 读请求。第 23 课详述存储节点启动流程。

### 7.5.2 fixed record 字节级布局

一条 fixed record（`node_stride` 字节）的布局（与第 6 课 `VamanaNode` 对齐）：

```
偏移        字段                  字节数    说明
0           header                8         u64，位域：NODE_LOCK / IS_MEDOID / DELETED / ...
8           id                    4         u32，外部 ID
12          generation            4         u32，schema-15 base 节点 = 0，在线 >= 1
16          exact vector          vector_bytes  原始向量，dtype 由 vector_dtype 决定
16+vb       (padding to stride)  ...        对齐到 node_stride
```

`node_base_offset = 16`，所以 shard 内存段前 16 字节是 shard-local 头（不在任何 fixed record 内），fixed record 从 offset 16 开始。

### 7.5.3 compact graph record 字节级布局

一条 compact graph record（`graph_entry_bytes` 字节，<=512）：

```
偏移        字段                  字节数    说明
0           header                8         u64，与 fixed record header 共享位定义
8           neighbor[0]           5         5 字节紧凑指针（shard_bits + offset bits）
13          neighbor[1]           5
...
8 + R*5     (padding to entry_bytes)  ...   对齐到 graph_entry_bytes
```

5 字节邻居指针编码（与 7.2.9 对应）：

```
位           字段                  位宽      说明
[39 : 40-sb) slot/offset           40-sb     shard 内 slot 编码（sb = graph_shard_bits）
[40-sb : 40) shard index           sb        跨 shard 邻居的 shard 编号
```

例如 `num_shards=4` → `graph_shard_bits=2` → 5 字节 = 40 位 = 2 位 shard + 38 位 slot。38 位 slot 上限 `2^38 = 2.7e11`，远超单 shard 节点数上限 `2^30`。

### 7.5.4 4 KiB 控制页字节级布局

```
偏移        结构                          字节数    说明
0           StorageControlBlock           640       主结构（alignas(64)），含 reclaim_ack_sequences[64]
640         (reserved)                    384       预留空间，未使用
1024        StorageRoutePublication       448       canonical route 快照（offset = kStorageRoutePublicationOffset）
1472        (reserved)                    2624      预留空间，未使用
4096        (end of control page)
```

`static_assert` 守护（7.2.11）：
- `640 <= 4096`（主结构能放进控制页）。
- `1024 >= 640`（route 快照在主结构之后）。
- `1024 + 448 = 1472 <= 4096`（route 快照整体在控制页内）。

### 7.5.5 `StorageControlBlock` 字节级布局

```
偏移        字段                          字节数    说明
0           magic                         8         "DSVCTRL1"
8           version                       4         = 2
12          header_bytes                  4         = 640
16          shard_id                      4
20          dynamic_record_bytes          4
24          dynamic_hot_offset            4
28          dynamic_code_offset           4
32          code_bytes                    4
36          compute_client_count          4
40          reserved0                     4
44          (padding to 8-align)          4
48          next_maintenance_sequence     8         = 1 初始
56          durable_maintenance_sequence  8
64          dynamic_high_watermark        8
72          reclaim_pending_nodes         8
80          reclaim_reused_nodes          8
88          reserved1                     8
96          reclaim_ack_sequences[64]     512       version 2 新增，64 路 compute client ACK
608         (padding to 640)              32        alignas(64) 填充
640         (end)
```

### 7.5.6 `StorageRoutePublication` 字节级布局

```
偏移        字段                          字节数    说明
0           sequence_begin                8         seqlock 起始（偶数）
8           magic                         8         "DSVROUT1"
16          version                       4         = 1
20          header_bytes                  4         = 448
24          shard_id                      4
28          slot_count                    4         = 8
32          code_bytes                    4         <= 32
36          reserved                      4
40          body_checksum                 8         FNV-1a，覆盖 magic..reserved + slots
48          slots[8]                      384       8 × StorageRouteSlot (48 字节)
432         sequence_end                  8         = sequence_begin
440         (padding to 448)              8         alignas(64) 填充
448         (end)
```

### 7.5.7 `CodeHeader` 字节级布局

```
偏移        字段                          字节数    说明
0           magic                         8         "DVGPUC5\0"
8           version                       4         = 5
12          header_bytes                  4         = 120
16          endian_marker                 4         = 0x01020304
20          memory_node                   4
24          quantizer_kind                4         = 1 (opq_pq)
28          code_bytes                    4
32          node_size                     4
36          reserved0                     4
40          entry_count                   8
48          remote_offset                 8
56          payload_bytes                 8         = entry_count * code_bytes
64          model_checksum                8
72          payload_checksum              8         FNV-1a over payload
80          header_checksum               8         FNV-1a over header (self = 0)
88          reserved[4]                   32
120         (end)
```

### 7.5.8 索引布局流程图

```
                    离线构建（第 12/13 课）
                            |
                            v
        +-------------------------------------------+
        |  生成 .meta.json (schema_version=15)     |
        |  生成 .anchors                            |
        |  生成 .pq<M> (PQ 模型)                    |
        |  生成 _node<i>_of<N>.dat (fixed+graph)    |
        |  生成 _node<i>_of<N>.pq<M>.codes          |
        |  生成 _node<i>_of<N>.idmap                |
        +-------------------------------------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
        计算节点启动                  存储节点启动（第 23 课）
              |                           |
              v                           v
    synthesize_distributed_view    加载 .dat / .codes / .idmap
    读 .meta.json + .anchors      进 RDMA 内存段
              |                           |
              v                           v
        得到 View                   初始化 StorageControlBlock
        (NavigationLayout +         初始化 StorageRoutePublication
         shards[] +                 (offset 1024 处)
         entry_points[])                     |
              |                           |
              v                           v
        上传 View 镜像到 GPU         等待 RDMA 读请求
        (第 17 课 kernel 上下文)             |
              |                           |
              v                           v
        GPU kernel 通过 RDMA 读远端内存段
        (第 20 课查询遍历主循环)
```

## 7.6 与其他模块的关系

本课定义的 schema-15 契约被以下课程消费：

- **第 6 课（vamana 图格式）**：`VamanaNode` 的 fixed record 布局、`hot_graph.hh` 的 compact graph record 布局，是本课 `ShardRegion::node_stride` / `graph_entry_bytes` 的具体实现。本课定义"shard 整体布局"，第 6 课定义"单条记录内部布局"。
- **第 8 课（元数据/owner map/存储协议）**：`service::index_metadata::load_metadata` 是 `synthesize_distributed_view` 的"前置步骤"——`load_metadata` 把 `.meta.json` 读成 `Metadata` 结构（不校验 schema），`synthesize_distributed_view` 直接读 JSON 并校验 schema-15。两者字段一一对应（`Metadata::schema_version` ↔ JSON `schema_version`，等等）。第 8 课会详讲 `Metadata` 结构与 `.meta.json` 的完整字段列表。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：PQ 模型文件 `.pq<M>` 的 `model_checksum` 写进 `NavigationLayout::model_checksum` 和 `CodeHeader::model_checksum`，用于 cross-check PQ code sidecar 与 metadata 描述的是同一套模型。
- **第 12/13 课（construction）**：construction 完成后用 `write_code_header` 写出每个 shard 的 `.pq*.codes` 文件头，用 `validate_view` 自校验合成的 `View`。construction 写出的 `.meta.json` 必须满足 `synthesize_distributed_view` 的 schema-15 严格等式。
- **第 17 课（kernel 启动器/上下文/device ring）**：kernel 启动时把 `View`（`NavigationLayout` + `shards[]` + `entry_points[]`）拷到 device 显存。device 侧的 `DeviceView` 镜像就是本课 `View` 的 device 版本。
- **第 20 课（查询遍历主循环）**：kernel 遍历图时用 `ordinal_to_remote` 的 device 版本把 ordinal 转成 `RemotePtr`，发 RDMA 读。compact graph record 里的 5 字节邻居指针解码成 (shard, slot)，再转成 ordinal。
- **第 16 课（存储回收 RCU）**：`StorageControlBlock::reclaim_ack_sequences[64]` 是 RCU 回收的核心数据结构——存储节点等所有 compute client 的 ACK 序列追上 `durable_maintenance_sequence` 后才能复用动态节点槽位。
- **第 23 课（存储节点主体/peer RDMA）**：存储节点启动时加载 `.dat` / `.codes` 进 RDMA 内存段，初始化 `StorageControlBlock` 和 `StorageRoutePublication`，并 cross-check `StorageControlBlock` 的 `dynamic_record_bytes` / `dynamic_hot_offset` / `dynamic_code_offset` 与 `ShardRegion` 的同名字段一致。
- **第 24 课（peer RPC）**：canonical route 快照（`StorageRoutePublication`）通过 peer RPC 在存储节点间传播，让每个存储节点都知道其他 shard 的 representative。
- **第 29 课（离线构建/迁移）**：schema-14 → 15 的"升级"必须用离线工具重建索引。本课的 fail-stop 设计强制了这一点——没有任何运行时自动升级代码。

## 7.7 小结

本课讲解了 dvstor 索引格式的 schema-15 契约，核心要点：

1. **两套版本号**：`kMetadataSchemaVersion = 15` 管 `.meta.json` 全局 schema；`kVersion = 5` 管 PQ code sidecar 文件头；`kStorageControlVersion = 2` 管控制页主结构；`kStorageRoutePublicationVersion = 1` 管路由快照。四者独立，不要混淆。

2. **fail-stop 哲学**：`synthesize_distributed_view` 用一组严格等式校验 metadata，schema-14 直接拒绝，不自动升级。schema-14 → 15 的"升级路径"是用离线工具重建索引（第 29 课），不是运行时迁移。控制页和路由快照的 version 升级是运行时结构升级，不需要磁盘迁移。

3. **5 个 POD 结构 + 8 条 static_assert**：`NavigationLayout`（全局参数）、`ShardRegion`（单 shard 布局）、`StorageControlBlock`（控制页主结构，640 字节，64 对齐）、`StorageRoutePublication`（路由快照，448 字节）、`CodeHeader`（PQ code sidecar 文件头，120 字节）。`static_assert` 在编译期锁死字节布局。

4. **三重防撕裂的 route 快照**：`sequence_begin` + `sequence_end` + `body_checksum`，加上 compute reader 额外的双 `sequence_begin` 读，保证并发发布的 route 快照要么读到完整的旧版本、要么读到完整的新版本，不会读到撕裂的中间态。查询和更新路径都不阻塞 route 刷新。

5. **5 字节紧凑指针**：compact graph record 的邻居指针只占 5 字节（40 位），编码 `graph_shard_bits` 位 shard + 剩余位 slot，比 `RemotePtr` 的 8 字节省 3 字节/邻居。在 R=64 的图里，一条记录省 192 字节，让一条邻居列表能塞进一个 512 字节 cache line。

6. **ordinal ↔ RemotePtr 双向映射**：`ordinal_to_remote` 用 `upper_bound` 二分查找 shard，`remote_to_ordinal` 直接索引 `shards[memory_node]`（因为 `memory_node == shard_index`）。这是 GPU kernel 跨 shard 跳转的基础。

7. **三级 entry point 填充**：medoid → anchor 侧 rank-轮转 → 随机采样 shard-轮转 → 顺序兜底。两级轮转保证 entry point 在 shard 间均衡，即使请求的 entry point 数小于一个 shard 的 anchor 数。

8. **几何契约守护者 `validate_view`**：一个巨大的 if 列举所有必须成立的条件——shard 索引一致性、非零性、区间不重叠、动态记录内部布局自洽、尺寸一致、全局覆盖、entry point 范围。任一不成立就 fail-stop。这是 schema-15 在运行时的最后防线。

下一课（第 8 课）我们将深入 `service/index_metadata.{hh,cc}`，看 `.meta.json` 的完整字段列表、`load_metadata` 的容错解析、以及 owner map 与存储协议如何依赖 schema-15 契约。
