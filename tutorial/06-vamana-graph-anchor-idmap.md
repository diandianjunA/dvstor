# 第 6 课：Vamana 图格式与 anchor / idmap

> 课号 6 / 30 ｜ 课题：Vamana 图格式与 anchor / idmap
> 项目根目录：`/home/xjs/experiment/dvstor`
> 涉及代码：`src/vamana/` 下的 8 个文件

## 6.1 本课目标与涉及文件

在第 5 课我们把 RDMA 传输库讲完了，从这一课开始我们正式进入"索引格式"的世界。dvstor 是一个 GPU 中心化的存算分离向量检索系统，它的检索路径并不是把整张图拷到 GPU 显存里跑，而是把图当作一个**远端 RDMA 可寻址的数据结构**：GPU 端持有一张"路由表 + anchor 入口"，通过 RDMA 读远端存储节点上的紧凑图记录，再沿着记录里的邻居指针继续跳。

要让这套机制成立，必须解决三个问题：

1. **图记录怎么在字节流里布局？** 越紧凑越好，因为每跳都要走 RDMA 读；又要可校验，因为 RDMA 链路上不能假设对端永远正确。
2. **冷启动时从哪里开始？** 图检索需要一个起点（medoid/anchor）。anchor 索引就是这套起点表。
3. **一个向量 ID 在 METIS 分区后到底归谁？** 离线分区之后，ID 的归属已经不是 `id % N` 那么简单，必须有一张显式的映射表。

本课围绕这三个问题，逐文件讲解 `src/vamana/` 目录下的 8 个文件：

| 文件 | 作用 | 是否含逻辑实现 |
|---|---|---|
| `vamana/vamana_node.hh` | 紧凑图记录的字节布局与静态参数 | 仅头文件（静态配置 + 编解码助手） |
| `vamana/hot_graph.hh` | "热点图"视图：紧凑指针编码、校验和、文件头 | 仅头文件（inline 工具函数） |
| `vamana/anchor_index.hh` | 静态 anchor 入口索引接口 | 仅声明 |
| `vamana/anchor_index.cc` | anchor 索引加载、最近邻 shard/anchor 路由 | 实现 |
| `vamana/idmap.hh` | base ID → 真实 owner / generation 映射的磁盘格式 | 仅头文件（格式常量 + 结构体） |
| `vamana/storage_layout_resolver.hh` | 把 RemotePtr 解析成具体的 (memory_node, offset, size) | 仅头文件 |
| `vamana/adaptive_route_table.hh` | 自适应路由表接口（storage-owner canonical 8 槽） | 仅声明 |
| `vamana/adaptive_route_table.cc` | 自适应路由表的 EMA 中心、representative 选举、路由 | 实现 |

读完本课你会理解：紧凑图记录为什么是 ≤512B、5 字节指针是怎么编码 16+40=56 位信息的、anchor `.anchors` 文件二进制格式、`.idmap` 为什么不能由 ID 推导 owner、以及自适应路由表与第 10 课 dynamic route overlay、第 21 课 kernel 侧 `DeviceDynamicRouteSlot` 的对应关系。

## 6.2 `vamana_node.hh`：紧凑图记录的字节布局

文件路径：`src/vamana/vamana_node.hh`

这是整个 vamana 模块的"格式总账"。它本身不存任何图数据，而是定义**一条图记录在 RDMA 内存里的字节布局**以及一组**进程级静态参数**（dim、R、dtype）。所有读写图记录的代码（kernel、construction、storage node、resolver）都通过这个类的静态方法取得偏移量。

### 6.2.1 文件头注释与设计意图

```cpp
// vamana_node.hh:12-16
/**
 * Fixed records contain header, id, generation, and exact vector. The compact
 * graph plane stores authoritative neighbors and is addressed deterministically
 * from the record's RemotePtr.
 */
class VamanaNode {
```

关键设计点：一条记录**固定包含 header + id + generation + 向量本体**，而**邻居（compact graph plane）是另一个平面**，通过对记录所在槽位的确定性偏移推导出来。这种分离让"读向量"和"读邻居"可以走两条独立的 RDMA 读路径，互不阻塞——这一点在 6.5 节讲 `StorageLayoutResolver` 时会再次出现。

### 6.2.2 header 位定义与尺寸常量

```cpp
// vamana_node.hh:19-26
static constexpr size_t HEADER_NODE_LOCK = 0b01;
static constexpr size_t HEADER_IS_MEDOID = 0b10000000000000000;
static constexpr size_t HEADER_DELETED = 0b1000000000000000000000000;
static constexpr u8 HOT_GRAPH_DELETED = 1u << 0;
static constexpr size_t HEADER_SIZE = sizeof(u64);
static constexpr size_t ID_SIZE = sizeof(u32);
static constexpr size_t GENERATION_SIZE = sizeof(u32);
static constexpr size_t COMPACT_META_SIZE = ID_SIZE + GENERATION_SIZE;
```

`header` 是一个 64 位字段，三段语义：

- `HEADER_NODE_LOCK = 0b01`（bit 0）：节点锁。读者通过 CAS 抢这个位实现"读时锁定"。
- `HEADER_IS_MEDOID = 1<<16`（bit 16）：标记该节点是某 shard 的 medoid，是 anchor 入口的候选。
- `HEADER_DELETED = 1<<24`（bit 24）：墓碑位，逻辑删除。

`HOT_GRAPH_DELETED` 是另一种 deleted 标记，存在 compact graph entry 的 flag 字节里（见 6.2.8）。

固定尺寸：`HEADER_SIZE = 8B`、`ID_SIZE = 4B`、`GENERATION_SIZE = 4B`，所以 `COMPACT_META_SIZE = 8B`（id + generation）。

### 6.2.3 进程级静态参数

```cpp
// vamana_node.hh:30-34
inline static u32 DIM{};
inline static u32 R{};
inline static VectorDType VECTOR_DTYPE{VectorDType::float32};
inline static u32 VECTOR_COMPONENT_SIZE{sizeof(element_t)};
inline static u32 VECTOR_BYTES{0};
```

这些都是 `inline static`，意味着整个进程共享一份。`element_t` 在 `common/types.hh:10` 定义为 `f32`。`DIM` 是向量维度，`R` 是最大出度（max degree）。**`R` 必须 ≤ 255**，因为 compact graph entry 用一个字节存 edge count——这条约束在 `init_static_storage` 里被显式 assert：

```cpp
// vamana_node.hh:36-51
static void init_static_storage(u32 dim,
                                u32 max_degree,
                                VectorDType vector_dtype = VectorDType::float32) {
  lib_assert(dim > 0, "Vamana dimension must be > 0");
  lib_assert(max_degree > 0, "Vamana max degree R must be > 0");
  lib_assert(max_degree <= std::numeric_limits<u8>::max(),
             "Vamana max degree R must be <= 255 because edge_count is stored in one byte");
  const size_t bytes = vector_dtype_bytes(vector_dtype, dim);
  lib_assert(bytes <= std::numeric_limits<u32>::max(),
             "Vamana vector byte width exceeds the runtime layout limit");
  DIM = dim;
  R = max_degree;
  VECTOR_DTYPE = vector_dtype;
  VECTOR_COMPONENT_SIZE = static_cast<u32>(vector_dtype_component_size(vector_dtype));
  VECTOR_BYTES = static_cast<u32>(bytes);
}
```

支持三种 dtype：float32 / uint8 / int8（与 anchor 索引的允许集合一致，见 6.3.2）。`vector_dtype_bytes(dtype, dim)` 给出该 dtype 在该维度下的字节数。这一步必须在任何读写图记录之前完成，否则所有 `offset_*` 方法都会基于 0 维度算出错误结果。

### 6.2.4 对齐与偏移量函数

```cpp
// vamana_node.hh:58-71
static constexpr size_t STORAGE_ALIGNMENT = 64;
static constexpr size_t COMPACT_ALIGNMENT = 16;

static size_t align_storage(size_t value) {
  return (value + STORAGE_ALIGNMENT - 1) & ~(STORAGE_ALIGNMENT - 1);
}
static size_t align_compact(size_t value) {
  return (value + COMPACT_ALIGNMENT - 1) & ~(COMPACT_ALIGNMENT - 1);
}
static size_t align8(size_t value) {
  return (value + 7) & ~size_t{7};
}
```

三档对齐：

- `STORAGE_ALIGNMENT = 64`：用于整个 shard 内存区域的分配对齐（cache line / GPU page）。
- `COMPACT_ALIGNMENT = 16`：单条记录的尾部对齐，也是 hot graph entry 的对齐。
- `align8`：8 字节对齐，给向量存储用——因为 compact pointer 以 8 字节为单位编码偏移（见 6.2.6）。

偏移量函数把字节布局钉死：

```cpp
// vamana_node.hh:73-79
static size_t offset_id() { return HEADER_SIZE; }                       // 8
static size_t offset_generation() { return HEADER_SIZE + ID_SIZE; }     // 12
static size_t graph_hot_bytes() { return HEADER_SIZE + COMPACT_META_SIZE; } // 16
static size_t offset_vector() { return graph_hot_bytes(); }             // 16
static size_t vector_storage_bytes() { return align8(vector_bytes()); }
static size_t vector_bytes() { return VECTOR_BYTES; }
static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
```

`graph_hot_bytes()` 这个名字有点 misleading，它其实是"固定记录的 metadata 头部大小"（header + id + generation = 16B），向量紧跟其后从 offset 16 开始。

### 6.2.5 neighbor 读取缓冲区布局

```cpp
// vamana_node.hh:80-86
static size_t neighbor_read_size() { return 8 + static_cast<size_t>(R) * sizeof(RemotePtr); }
static constexpr size_t neighbor_count_offset_in_read() { return ID_SIZE; }
static constexpr size_t neighbor_payload_offset_in_read() { return 8; }
static size_t total_size() {
  const size_t end = offset_vector() + vector_storage_bytes();
  return align_compact(end);
}
```

注意 `neighbor_read_size()` 返回的不是磁盘上的 compact entry 大小，而是**解码后写入读缓冲的大小**：

```
[0..4)    : (保留)
[4..8)    : u8 edge_count（放在 ID_SIZE 偏移处）
[8..8+R*8): R 个 RemotePtr（每个 8 字节）
```

这里用的是 `sizeof(RemotePtr) = 8`（见 `remote_pointer.hh:6`，`RemotePtr::SIZE = sizeof(u64)`），所以读缓冲是"宽松版"——每条邻居占 8 字节，而磁盘上的 compact 版本每条只占 5 字节（见 6.2.6）。`decode_hot_graph_entry`（6.2.8）就是把 5 字节紧凑版展开成 8 字节宽松版。

`total_size()` 是**一条固定记录（含向量）在 RDMA 内存里占用的总字节数**，按 16 字节对齐。例如 dim=128, float32：`VECTOR_BYTES=512`，`offset_vector()=16`，`size_until_vector_end=528`，`align_compact(528)=528`（已对齐），`total_size()=528`。这就是"≤512B"的来源（实际上带 header 后稍大于 512，但向量本身 ≤512B 满足常见 dim=128 的 float32 场景）。

### 6.2.6 RemotePtr 与 compact pointer 的关系

在讲 hot graph 之前先看 `RemotePtr` 本身（`src/remote_pointer.hh:4-25`）：

```cpp
struct RemotePtr {
  static constexpr size_t SIZE = sizeof(u64);
  u64 raw_address{};  // [ memory node (16b) | byte offset (48b) ]

  u32 memory_node() const { return raw_address >> 48; }
  u64 byte_offset() const { return (raw_address << 16) >> 16; }
  ...
};
```

`RemotePtr` 是 8 字节的"宽松指针"：高 16 位是 memory_node（shard id），低 48 位是 byte_offset。48 位 byte offset 上限是 256 TB，对单个 shard 足够。但 8 字节 × R 个邻居 × 数亿节点，磁盘和 RDMA 带宽上吃不消，于是有了 compact pointer。

`hot_graph.hh:15-17` 给出 compact 指针的参数：

```cpp
constexpr u32 kCompactPointerBytes = 5;            // 40 bit
constexpr u64 kNodeBaseOffset = 16;
constexpr u64 kNullCompactPointer = (1ull << 40) - 1ull;  // 全 1
```

5 字节 = 40 位。`kNullCompactPointer` 是全 1，作为 null sentinel。`kNodeBaseOffset = 16` 表示 shard 内节点数据从偏移 16 开始（前面留 16 字节给 shard header）。

compact pointer 的编解码见 6.3.3，核心思想是：把 16+48=64 位的 RemotePtr 压成 `shard_bits + (40 - shard_bits)` 位，shard 数少时偏移量位数多，反之亦然。

### 6.2.7 hot graph 静态配置

`vamana_node.hh:90-99` 定义了一组 hot graph 相关的静态参数：

```cpp
inline static bool HAS_HOT_GRAPH = false;
inline static u32 HOT_GRAPH_ENTRY_BYTES = 0;
inline static u32 HOT_GRAPH_SHARD_BITS = 0;
inline static vec<u64> HOT_GRAPH_ENTRY_OFFSETS;
inline static vec<u64> HOT_GRAPH_ENTRY_COUNTS;
inline static vec<u64> HOT_GRAPH_DYNAMIC_BASE_OFFSETS;
inline static u32 HOT_GRAPH_DYNAMIC_RECORD_BYTES = 0;
inline static u32 HOT_GRAPH_DYNAMIC_HOT_OFFSET = 0;
inline static u32 HOT_GRAPH_DYNAMIC_CODE_OFFSET = 0;
inline static u32 HOT_GRAPH_DYNAMIC_CODE_BYTES = 0;
```

"hot graph"在 dvstor 里指**与固定记录同区分配的紧凑邻居平面**。两条路径：

- **静态布局**：每个 shard 的 compact graph 入口表位于一个固定偏移 `HOT_GRAPH_ENTRY_OFFSETS[node]`，每条入口 `HOT_GRAPH_ENTRY_BYTES` 字节，共 `HOT_GRAPH_ENTRY_COUNTS[node]` 条。`hot_graph_entry_offset(ptr)` 通过 `ptr.byte_offset()` 反推槽位序号 `slot = (offset - 16) / total_size()`，再 `entry_base + slot * entry_bytes`。
- **动态布局**：当节点是动态分配的（增量发布场景，见第 15 课），compact graph entry 紧贴在固定记录后面，偏移量是 `ptr.byte_offset() + HOT_GRAPH_DYNAMIC_HOT_OFFSET`。

`configure_hot_graph` 是配置入口（`vamana_node.hh:122-157`），它做了一系列校验：

```cpp
// vamana_node.hh:148-156
HAS_HOT_GRAPH = entry_bytes >= hot_graph_entry_size() &&
  HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() == HOT_GRAPH_ENTRY_OFFSETS.size() &&
  HOT_GRAPH_DYNAMIC_RECORD_BYTES >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
  HOT_GRAPH_DYNAMIC_HOT_OFFSET >= total_size() &&
  (HOT_GRAPH_DYNAMIC_CODE_BYTES == 0 ||
   (HOT_GRAPH_DYNAMIC_CODE_OFFSET >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
    HOT_GRAPH_DYNAMIC_RECORD_BYTES >=
      HOT_GRAPH_DYNAMIC_CODE_OFFSET + HOT_GRAPH_DYNAMIC_CODE_BYTES));
if (!HAS_HOT_GRAPH) disable_hot_graph();
```

校验逻辑确保：entry_bytes 至少能装下一条 entry；动态基址数组与静态数组一一对应；hot offset 在固定记录之后；可选的"navigation code"区域（用于自适应路由的 PQ 量化码，见第 9 课）也在 hot entry 之后且不越界。任何一项不满足就整体回退到 `disable_hot_graph()`，保证不会进入半残状态。

`hot_graph_entry_offset` 的双路径逻辑：

```cpp
// vamana_node.hh:179-189
static u64 hot_graph_entry_offset(RemotePtr ptr) {
  const u64 relative = ptr.byte_offset() - vamana::hot_graph::kNodeBaseOffset;
  const u64 node_size = total_size();
  if (node_size != 0 && relative % node_size == 0) {
    const u64 slot = relative / node_size;
    if (slot < HOT_GRAPH_ENTRY_COUNTS[ptr.memory_node()]) {
      return HOT_GRAPH_ENTRY_OFFSETS[ptr.memory_node()] + slot * HOT_GRAPH_ENTRY_BYTES;
    }
  }
  return ptr.byte_offset() + HOT_GRAPH_DYNAMIC_HOT_OFFSET;
}
```

先尝试按静态布局解析（relative 必须能整除 `total_size()`，且 slot 在范围内）；失败就按动态布局回退到 `ptr.byte_offset() + HOT_GRAPH_DYNAMIC_HOT_OFFSET`。这是"同一份代码服务两种部署形态"的关键。

### 6.2.8 compact entry 的编解码

`encode_hot_graph_entry`（`vamana_node.hh:198-219`）把内存里的 `RemotePtr[]` 编码进 5 字节紧凑指针：

```cpp
static void encode_hot_graph_entry(byte_t* out,
                                   u8 edge_count,
                                   const RemotePtr* neighbors,
                                   size_t neighbor_count,
                                   u32 shard_bits = HOT_GRAPH_SHARD_BITS,
                                   u32 generation = 0,
                                   bool deleted = false) {
  std::memset(out, 0, hot_graph_entry_size());
  out[0] = deleted ? 0 : static_cast<u8>(std::min<size_t>(edge_count, R));
  out[1] = deleted ? HOT_GRAPH_DELETED : 0;
  vamana::hot_graph::store_u32_le(out + 4, generation);
  for (u32 i = 0; i < R; ++i) {
    byte_t* encoded = out + vamana::hot_graph::neighbor_offset(i);
    if (!deleted && i < neighbor_count) {
      (void)vamana::hot_graph::encode_remote_ptr(neighbors[i], shard_bits, encoded);
    } else {
      (void)vamana::hot_graph::encode_remote_ptr(RemotePtr{}, shard_bits, encoded);
    }
  }
  const u16 checksum = vamana::hot_graph::checksum16(out, hot_graph_entry_size());
  vamana::hot_graph::store_u16_le(out + 2, checksum);
}
```

compact entry 内部布局（与 `hot_graph.hh` 对应）：

```
[0]    edge_count（deleted 时写 0）
[1]    flags（bit0 = HOT_GRAPH_DELETED）
[2..4) checksum16（FNV-1a，跳过 byte 2/3 自身）
[4..8) generation（u32 LE）
[8..8+R*5) R 条 5 字节 compact pointer
```

注意 `out[0] = deleted ? 0 : min(edge_count, R)`——deleted 时强制 edge_count=0，这样解码端即使忽略 flag 也不会读到脏邻居。`generation` 写在 byte 4-7，让 RDMA 单次 8 字节读就能同时拿到 edge_count + flags + checksum + generation 的一半，方便快速过滤。

解码侧 `decode_hot_graph_entry`（`vamana_node.hh:221-240`）做严格校验：

```cpp
static bool decode_hot_graph_entry(const byte_t* compact, byte_t* neighbor_read_buffer) {
  std::memset(neighbor_read_buffer, 0, neighbor_read_size());
  const u8 edge_count = compact[0];
  if (edge_count > R) return false;                                  // 越界
  const u16 expected = vamana::hot_graph::load_u16_le(compact + 2);
  const u16 actual = vamana::hot_graph::checksum16(compact, hot_graph_entry_size());
  if (expected != actual) return false;                              // 校验和失败
  if ((compact[1] & HOT_GRAPH_DELETED) != 0) {                       // 墓碑
    *reinterpret_cast<u8*>(neighbor_read_buffer + neighbor_count_offset_in_read()) = 0;
    return true;
  }
  *reinterpret_cast<u8*>(neighbor_read_buffer + neighbor_count_offset_in_read()) = edge_count;
  auto* out = reinterpret_cast<RemotePtr*>(neighbor_read_buffer + neighbor_payload_offset_in_read());
  for (u32 i = 0; i < edge_count; ++i) {
    out[i] = vamana::hot_graph::decode_remote_ptr(
      compact + vamana::hot_graph::neighbor_offset(i), HOT_GRAPH_SHARD_BITS);
  }
  return true;
}
```

返回 `bool` 表示解码是否成功。三种"成功"情况：正常 entry、墓碑 entry（edge_count 写 0）、空 entry。失败情况：edge_count 超过 R、checksum 不对。校验和失败时直接丢弃整条 entry——这是 RDMA 路径上对抗"对端写了半条记录"的核心防线。第 16 课讲 RCU 存储回收时还会再回到这个校验和。

### 6.2.9 allocation_size 与 dynamic navigation code

```cpp
// vamana_node.hh:102-107
static size_t hot_graph_entry_size() { return vamana::hot_graph::entry_bytes(R); }
static size_t dynamic_record_size() {
  return align_compact(total_size() + hot_graph_entry_size());
}
static size_t allocation_size() {
  return HAS_HOT_GRAPH ? HOT_GRAPH_DYNAMIC_RECORD_BYTES : total_size();
}
```

- `hot_graph_entry_size()` = `align8(8 + R*5)`。例如 R=64：`8 + 320 = 328`，`align8(328) = 328`。
- `dynamic_record_size()` = 固定记录 + 紧邻其后的 hot entry，整体 16 字节对齐。
- `allocation_size()` 是分配单条记录（含 hot entry）时实际要的字节数。

```cpp
// vamana_node.hh:191-196
static u64 dynamic_navigation_code_offset(RemotePtr ptr) {
  lib_assert(HAS_HOT_GRAPH && HOT_GRAPH_DYNAMIC_CODE_BYTES != 0 &&
               hot_graph_entry_available(ptr),
             "dynamic navigation code requested for an invalid node");
  return ptr.byte_offset() + HOT_GRAPH_DYNAMIC_CODE_OFFSET;
}
```

"navigation code"是该节点用于自适应路由的 PQ 量化码（见第 9 课 PQ 模型），紧贴在 hot entry 之后。这条 API 给 kernel 侧的"按节点读导航码"提供偏移量。

### 6.2.10 字节布局图

把 6.2 节汇总成一张图：

```
┌─────────────────────────── VamanaNode 固定记录 (total_size) ───────────────────────────┐
│                                                                                          │
│  offset 0                  8        12        16                          16+Vbytes      │
│  ┌──────────┬─────────┬───────────┬──────────────────────────────────┬────────────────┐  │
│  │ header   │   id    │ generation │           vector (Vbytes)         │  pad to 16B   │  │
│  │ u64      │  u32    │   u32      │  (float32/uint8/int8, dim 维)    │  (align_compact)│  │
│  │ lock|med │         │            │                                  │                │  │
│  │ |deleted │         │            │                                  │                │  │
│  └──────────┴─────────┴───────────┴──────────────────────────────────┴────────────────┘  │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

       ↓ 同一 RDMA 区，紧贴在固定记录之后（动态布局）
       或在独立 entry table 中按 slot 索引（静态布局）

┌──────────────────────── hot graph entry (hot_graph_entry_size = align8(8 + R*5)) ───────┐
│  0     1       2..4        4..8                  8..8+R*5                                │
│  ┌─────┬───────┬───────────┬──────────┬──────────────────────────────────────────────┐  │
│  │edge │flags  │ checksum  │generation│  R × 5B compact pointer                      │  │
│  │count│       │  u16 LE   │  u32 LE  │  [shard_bits | (40-shard_bits) offset units] │  │
│  │ u8  │ u8    │ FNV-1a    │          │  null = 0xFFFFFFFFFF                         │  │
│  └─────┴───────┴───────────┴──────────┴──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

       ↓ 可选：navigation code (PQ 量化码, dynamic_code_bytes)
       offset = ptr.byte_offset() + HOT_GRAPH_DYNAMIC_CODE_OFFSET
```

`allocation_size` = HAS_HOT_GRAPH 时为 `HOT_GRAPH_DYNAMIC_RECORD_BYTES`（含 hot entry + 可选 nav code），否则仅 `total_size()`。

## 6.3 `hot_graph.hh`：热点图视图与紧凑指针编码

文件路径：`src/vamana/hot_graph.hh`

这个头文件是 `vamana::hot_graph` 命名空间下一组 `inline` 工具函数和常量，给 `vamana_node.hh` 的编解码提供底层原语。它定义了 compact graph 的文件头、5 字节指针编解码、FNV-1a 校验和、小端整数读写。

### 6.3.1 文件头与 magic

```cpp
// hot_graph.hh:12-18
constexpr u32 kMagic = 0x31474844;  // DHG1
constexpr u16 kVersion = 1;
constexpr u16 kVersion2 = 2;
constexpr u32 kCompactPointerBytes = 5;
constexpr u64 kNodeBaseOffset = 16;
constexpr u64 kNullCompactPointer = (1ull << 40) - 1ull;
constexpr u32 kV2NeighborBaseOffset = 8;
```

`kMagic = 0x31474844`，ASCII 是 "DHG1"（小端读为 "1GHD"，大端读为 "DHG1"）。`kVersion = 1`、`kVersion2 = 2` 两个版本并存——v2 把邻居区的起始偏移从隐式改为显式 `kV2NeighborBaseOffset = 8`，给后续兼容留了余地。

`kNullCompactPointer = (1<<40) - 1 = 0xFFFFFFFFFF`，全 1。这跟 5 字节指针的"全 1"对齐，编解码两端都用这个值标记 null。

### 6.3.2 文件头结构体

```cpp
// hot_graph.hh:20-38
#pragma pack(push, 1)
struct Header {
  u32 magic{kMagic};
  u16 version{kVersion};
  u16 header_bytes{64};
  u32 entry_bytes{};
  u32 max_degree{};
  u32 compact_pointer_bytes{kCompactPointerBytes};
  u32 compact_pointer_shard_bits{};
  u32 flags{};
  u64 entry_count{};
  u64 node_base_offset{kNodeBaseOffset};
  u64 reserved0{};
  u64 reserved1{};
  u32 reserved2{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 64);
```

`#pragma pack(push, 1)` + `static_assert(sizeof(Header) == 64)` 保证磁盘上头是 64 字节紧凑布局，没有 padding。字段含义：

- `magic` / `version`：版本握手。
- `header_bytes = 64`：自描述头部大小，将来扩展不会破坏旧解析器。
- `entry_bytes`：一条 compact entry 的字节数（= `align8(8 + R*5)`）。
- `max_degree`：即 R。
- `compact_pointer_bytes = 5`：固定 5。
- `compact_pointer_shard_bits`：shard 位宽，决定了 compact pointer 里 shard 字段占几位、offset 字段占几位。
- `flags`：保留位。
- `entry_count`：本 shard 的 entry 总数。
- `node_base_offset = 16`：节点数据起始偏移，跟 compact pointer 解码时减基址对齐。

### 6.3.3 compact pointer 编解码

```cpp
// hot_graph.hh:58-77
inline bool encode_remote_ptr(RemotePtr ptr, u32 shard_bits, byte_t* out) {
  if (ptr.is_null() || ptr.byte_offset() % 8 != 0 || shard_bits >= 16) {
    std::memset(out, 0xff, kCompactPointerBytes);
    return ptr.is_null();
  }

  const u32 offset_bits = 40 - shard_bits;
  const u64 max_shards = 1ull << shard_bits;
  const u64 offset_units = ptr.byte_offset() / 8;
  if (ptr.memory_node() >= max_shards || offset_units >= (1ull << offset_bits)) {
    std::memset(out, 0xff, kCompactPointerBytes);
    return false;
  }

  const u64 packed = (static_cast<u64>(ptr.memory_node()) << offset_bits) | offset_units;
  for (u32 i = 0; i < kCompactPointerBytes; ++i) {
    out[i] = static_cast<byte_t>((packed >> (8 * i)) & 0xffu);
  }
  return true;
}
```

逐步解析：

1. **三个不可编码情况**：null 指针、byte_offset 不按 8 对齐、shard_bits ≥ 16。任一发生都写全 1（null sentinel）。返回值区分"原本就是 null"（true）和"非法但被强制 null"（false）。
2. **位宽分配**：`offset_bits = 40 - shard_bits`。shard_bits=4 时 offset 有 36 位，对应 `2^36 * 8 = 512 GB` 的 shard 内可寻址空间；shard_bits=8 时 offset 有 32 位，对应 `32 GB`。
3. **单位转换**：`offset_units = byte_offset / 8`，因为所有合法 offset 都已对齐到 8 字节，所以可以无损除 8。
4. **范围检查**：`memory_node < 2^shard_bits`、`offset_units < 2^offset_bits`。任何一边越界都不可编码，写 null。
5. **打包**：`packed = (memory_node << offset_bits) | offset_units`，小端逐字节写出。

解码是对称的：

```cpp
// hot_graph.hh:79-92
inline RemotePtr decode_remote_ptr(const byte_t* in, u32 shard_bits) {
  u64 packed = 0;
  for (u32 i = 0; i < kCompactPointerBytes; ++i) {
    packed |= static_cast<u64>(in[i]) << (8 * i);
  }
  if (packed == kNullCompactPointer || shard_bits >= 16) {
    return RemotePtr{};
  }
  const u32 offset_bits = 40 - shard_bits;
  const u64 offset_mask = (1ull << offset_bits) - 1ull;
  const u32 shard = static_cast<u32>(packed >> offset_bits);
  const u64 offset = (packed & offset_mask) * 8;
  return RemotePtr{shard, offset};
}
```

读 5 字节拼成 40 位，先判 null，再按位宽切分 shard / offset，offset 乘 8 还原成字节偏移。注意编码端做 `byte_offset / 8`、解码端做 `* 8`，这一对操作要求所有图记录地址严格 8 字节对齐——这也是 6.2.4 节 `align8` 在向量存储上必须用的原因之一。

### 6.3.4 校验和：FNV-1a 变种

```cpp
// hot_graph.hh:98-107
inline u16 checksum16(const byte_t* data, size_t bytes) {
  u32 hash = 2166136261u;
  for (size_t i = 0; i < bytes; ++i) {
    if (i == 2 || i == 3) continue;     // 跳过 checksum 自身
    hash ^= data[i];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}
```

标准 FNV-1a（offset basis `2166136261`、prime `16777619`）的 16 位折叠版本。两个特殊点：

1. **跳过 byte 2/3**：这两字节是 checksum 自身的位置，必须跳过否则循环依赖。
2. **末尾 `hash ^= hash >> 16` + 截断到 u16**：把 32 位 hash 折叠成 16 位，比直接取低 16 位的雪崩性更好。

校验和只覆盖单条 entry（几十到几百字节），FNV-1a 在这个尺寸上够用，且实现简单、对 GPU 友好（kernel 端可复用同样的常量）。代价是不能防恶意篡改，但 dvstor 假设对端存储节点可信，校验和只防"半写"和"读错地址"。

### 6.3.5 小端整数读写

```cpp
// hot_graph.hh:109-127
inline u32 load_u32_le(const byte_t* data) {
  u32 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}
inline void store_u32_le(byte_t* data, u32 value) {
  std::memcpy(data, &value, sizeof(value));
}
inline u16 load_u16_le(const byte_t* data) { ... }
inline void store_u16_le(byte_t* data, u16 value) { ... }
```

注意函数名带 `_le`，但实现是直接 `memcpy`——这隐含假设**目标平台是小端**（x86 / ARM little-endian）。如果在 PDP-11 之类的怪平台上跑会出错，但 dvstor 只在 x86 + ARM little-endian 部署，所以无伤大雅。命名上的 `_le` 是文档性提示：磁盘格式是小端。

`neighbor_offset(i) = 8 + i * 5`（`hot_graph.hh:94-96`）给出第 i 条邻居在 entry 内的偏移。

`entry_bytes(max_degree) = align8(8 + max_degree * 5)`（`hot_graph.hh:44-46`）：8 字节头（edge_count + flags + checksum + generation）+ R × 5 字节指针，再 8 字节对齐。

`shard_bits_for(shard_count)`（`hot_graph.hh:48-56`）计算至少需要多少位表示 shard_count 个 shard——向上取整到 2 的幂。例如 shard_count=12 → bits=4（2^4=16 ≥ 12）。

## 6.4 `anchor_index.hh` / `anchor_index.cc`：静态 anchor 入口索引

文件路径：`src/vamana/anchor_index.hh`、`src/vamana/anchor_index.cc`

图检索必须有个起点。在 Vamana 算法里起点是 medoid（离质心最近的实际节点）。anchor 索引就是离线计算好的"每 shard 取若干 medoid 作为入口候选"的静态表，存成 `.anchors` 文件。它在两种场景下被使用：

1. **冷启动**：计算侧刚加载完索引，自适应路由表还空着，必须靠 anchor 兜底。
2. **召回兜底**：自适应路由表（6.6 节）覆盖不全时，anchor 提供确定性的几何最近入口。

### 6.4.1 文件格式常量与结构体

```cpp
// anchor_index.hh:11-13
constexpr u64 kMagic = 0x3148434e414c4441ull;  // "ADLANCH1"
constexpr u32 kVersion = 1;
```

`kMagic` 是 8 字节 ASCII "ADLANCH1"（A-D-L-A-N-C-H-1，注意 dvstor 把它当成 u64 小端整数，所以字节序读出来是 `1HNLADA`...，但作为磁盘 magic 只关心它是个唯一常量）。

```cpp
// anchor_index.hh:14-24
struct Header {
  u64 magic{kMagic};
  u32 version{kVersion};
  u32 dim{};
  u32 shard_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchors_per_shard{};
  u32 reserved{};
  u64 total_anchors{};
};
```

全局头部 48 字节（注意没有 `#pragma pack`，但所有字段都是 4/8 字节自然对齐，sizeof 在常见平台上是 48）。字段含义自明。

```cpp
// anchor_index.hh:26-36
struct ShardHeader {
  u32 shard{};
  u32 anchor_count{};
};

struct EntryHeader {
  u64 rptr_raw{};     // RemotePtr 的 raw_address
  u32 id{};
  u16 degree{};
  u16 reserved{};
};
```

每个 shard 一个 ShardHeader（8 字节），随后是 `anchor_count` 个 `(EntryHeader, vector)` 对。`EntryHeader` 16 字节，`rptr_raw` 直接存 `RemotePtr::raw_address`（8 字节 16+48 拆分），`id` 是该 anchor 节点的 ID，`degree` 是它在图里的出度（仅供调试/分析用，路由时不读）。

```cpp
// anchor_index.hh:38-42
struct Route {
  u32 owner{};
  vec<RemotePtr> hints;
  RemotePtr bucket_hint;
};
```

`Route` 是 anchor 索交给调用方的"路由建议"：`owner` 是推荐的 owner shard，`hints` 是一组 anchor RemotePtr（按几何距离排序），`bucket_hint` 是第一个（最近）的 anchor。

### 6.4.2 Index 类接口

```cpp
// anchor_index.hh:44-74
class Index {
public:
  bool load(const filepath_t& index_prefix,
            u32 expected_dim,
            u32 expected_shards,
            str* error_message = nullptr);

  bool empty() const { return shards_.empty(); }
  size_t anchor_count() const { return total_anchors_; }
  size_t memory_bytes() const;

  Route route(const span<const element_t> query,
              u32 hint_count,
              std::optional<u32> owner_override = std::nullopt) const;
  vec<u32> nearest_shards(const span<const element_t> query, u32 count) const;
  u32 nearest_shard(const span<const element_t> query) const;
  vec<RemotePtr> nearest_anchors(const span<const element_t> query,
                                 u32 shard,
                                 u32 count) const;

private:
  struct Shard {
    vec<element_t> centroid;        // shard 质心（dim 维 float32）
    vec<element_t> vectors;         // flatten 的 anchor 向量，anchor_count * dim
    vec<RemotePtr> pointers;        // 每个 anchor 的 RemotePtr
  };

  u32 dim_{};
  size_t total_anchors_{};
  vec<Shard> shards_;
};
```

内部表示很直接：每个 shard 存质心 + anchor 向量 + anchor 指针。所有向量都解码成 `element_t = f32`（即使 dtype 是 uint8/int8），这样查询时不用每次转换。

### 6.4.3 load：`.anchors` 文件加载

`anchor_index.cc:24-104` 是加载实现。逐段看：

```cpp
// anchor_index.cc:24-36
bool Index::load(const filepath_t& index_prefix,
                 u32 expected_dim,
                 u32 expected_shards,
                 str* error_message) {
  dim_ = 0;
  total_anchors_ = 0;
  shards_.clear();

  const filepath_t path = index_path::anchor_file(index_prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return fail(error_message, "missing anchor sidecar: " + path.string());
  }
```

`index_path::anchor_file(prefix)` 在 `common/index_path.hh:36` 定义，把 prefix 拼成 `.anchors` 路径（具体拼接规则见第 7 课 schema-15 索引格式）。先重置状态，打开文件，失败时通过 `fail` 助手写错误消息并返回 false。

```cpp
// anchor_index.cc:38-43
Header header;
input.read(reinterpret_cast<char*>(&header), sizeof(header));
if (!input.good() || header.magic != kMagic || header.version != kVersion ||
    header.dim != expected_dim || header.shard_count != expected_shards) {
  return fail(error_message, "invalid or incompatible anchor sidecar: " + path.string());
}
```

读 48 字节全局头，做五个一致性检查：流状态、magic、version、dim 与调用方期望一致、shard_count 与调用方期望一致。`expected_dim/expected_shards` 来自索引前缀的主格式（schema-15），这保证 anchor sidecar 不会跟主索引对不上。

```cpp
// anchor_index.cc:45-63
VectorDType dtype;
try {
  switch (static_cast<VectorDType>(header.vector_dtype)) {
    case VectorDType::float32:
    case VectorDType::uint8:
    case VectorDType::int8:
      dtype = static_cast<VectorDType>(header.vector_dtype);
      break;
    default:
      return fail(error_message, "invalid anchor sidecar dtype: " + path.string());
  }
  if (vector_dtype_bytes(dtype, header.dim) != header.vector_bytes) {
    return fail(error_message, "anchor sidecar vector layout mismatch: " + path.string());
  }
} catch (const std::exception& e) {
  return fail(error_message, "invalid anchor sidecar dtype: " + str{e.what()});
}
```

dtype 校验：只允许 float32 / uint8 / int8 三种。`vector_dtype_bytes(dtype, dim)` 算出的字节数必须等于 header 里声明的 `vector_bytes`，否则布局不一致。`try/catch` 是因为 `vector_dtype_bytes` 在某些非法组合下会抛异常。

```cpp
// anchor_index.cc:63-97
dim_ = header.dim;
shards_.resize(header.shard_count);
vec<byte_t> raw(header.vector_bytes);
size_t loaded = 0;
for (u32 expected_shard = 0; expected_shard < header.shard_count; ++expected_shard) {
  ShardHeader shard_header;
  input.read(reinterpret_cast<char*>(&shard_header), sizeof(shard_header));
  if (!input.good() || shard_header.shard != expected_shard) {
    return fail(error_message, "invalid anchor shard header: " + path.string());
  }
  if (shard_header.anchor_count > header.anchors_per_shard ||
      loaded + shard_header.anchor_count > header.total_anchors) {
    return fail(error_message, "invalid anchor shard count: " + path.string());
  }

  auto& shard = shards_[expected_shard];
  shard.centroid.resize(dim_);
  input.read(reinterpret_cast<char*>(shard.centroid.data()),
             static_cast<std::streamsize>(dim_ * sizeof(element_t)));
  shard.vectors.resize(static_cast<size_t>(shard_header.anchor_count) * dim_);
  shard.pointers.reserve(shard_header.anchor_count);
  for (u32 i = 0; i < shard_header.anchor_count; ++i) {
    EntryHeader entry;
    input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    input.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(raw.size()));
    if (!input.good()) {
      return fail(error_message, "truncated anchor sidecar: " + path.string());
    }
    decode_storage_vector_to_float(raw.data(), dtype, dim_,
                                   shard.vectors.data() + static_cast<size_t>(i) * dim_);
    shard.pointers.emplace_back(entry.rptr_raw);
    ++loaded;
  }
}
```

外层循环遍历 shard，每个 shard 先读 8 字节 ShardHeader，校验 `shard` 字段等于循环下标（防止文件里 shard 乱序）和 anchor_count 不超过 `anchors_per_shard` 上限、累计不超过 `total_anchors`。

然后读 `dim * 4` 字节的 shard 质心（`element_t = f32`）。注意：质心**总是 float32**，与 `vector_dtype` 无关——这是磁盘格式约定。

接着逐个 anchor：读 16 字节 EntryHeader + `vector_bytes` 字节原始向量，用 `decode_storage_vector_to_float` 把 uint8/int8/float32 解码成 float32 存进 `shard.vectors`（连续内存，layout 是 `[anchor0_dim0, anchor0_dim1, ..., anchor1_dim0, ...]`）。`entry.rptr_raw` 直接构造 `RemotePtr`（`RemotePtr(u64)` 构造器）放进 `pointers`。

```cpp
// anchor_index.cc:99-103
if (loaded != header.total_anchors) {
  return fail(error_message, "anchor sidecar count mismatch: " + path.string());
}
total_anchors_ = loaded;
return true;
```

最后做总数对账，确保实际读到的 anchor 数等于 header 声明的总数。这一步是抗文件截断/损坏的最后一道防线。

### 6.4.4 查询接口：nearest_shard / nearest_shards / nearest_anchors

```cpp
// anchor_index.cc:116-130
u32 Index::nearest_shard(const span<const element_t> query) const {
  u32 best_shard = 0;
  distance_t best_distance = std::numeric_limits<distance_t>::max();
  for (u32 shard = 0; shard < shards_.size(); ++shard) {
    if (shards_[shard].centroid.size() != dim_) {
      continue;
    }
    const distance_t distance = L2Distance::dist(query, shards_[shard].centroid, dim_);
    if (distance < best_distance) {
      best_distance = distance;
      best_shard = shard;
    }
  }
  return best_shard;
}
```

线性扫描所有 shard 质心，找 L2 距离最小的。shard 数通常几十量级，O(shard_count * dim) 完全可接受。`L2Distance::dist` 在 `common/distance.hh`（见第 18 课候选评分）。

`nearest_shards(query, count)`（`anchor_index.cc:132-157`）是 top-k 版本：用 `std::priority_queue<Candidate>` 维护大小为 count 的最大堆，最后按距离升序输出。注意是最大堆——堆顶是当前最远的，新候选距离更近就替换堆顶。

`nearest_anchors(query, shard, count)`（`anchor_index.cc:159-186`）在指定 shard 内对所有 anchor 做同样的 top-k，但返回的是 `RemotePtr` 而非索引。`shard.pointers[nearest.top().second]` 从堆里取出索引再查指针。

### 6.4.5 route：综合路由建议

```cpp
// anchor_index.cc:188-217
Route Index::route(const span<const element_t> query,
                   u32 hint_count,
                   std::optional<u32> owner_override) const {
  Route route;
  if (shards_.empty() || query.size() != dim_) {
    return route;
  }
  const u32 semantic_shard = nearest_shard(query);
  route.owner = owner_override.has_value() ? *owner_override : semantic_shard;
  vec<RemotePtr> semantic = nearest_anchors(
    query, semantic_shard, std::max<u32>(1, hint_count));
  if (!semantic.empty()) route.bucket_hint = semantic.front();
  if (route.owner == semantic_shard) {
    route.hints.assign(semantic.begin(), semantic.begin() +
      std::min<size_t>(semantic.size(), hint_count));
    return route;
  }

  const u32 local_count = (hint_count + 1) / 2;
  route.hints = nearest_anchors(query, route.owner, local_count);
  const size_t semantic_count = std::min<size_t>(
    semantic.size(), hint_count - local_count);
  for (size_t index = 0; index < semantic_count; ++index) {
    const RemotePtr hint = semantic[index];
    if (std::find(route.hints.begin(), route.hints.end(), hint) == route.hints.end()) {
      route.hints.push_back(hint);
    }
  }
  return route;
}
```

这是 anchor 索引的核心 API，给调用方一个完整路由建议。逻辑：

1. 计算语义最近的 shard（`semantic_shard`，按几何质心最近）。
2. `owner` 由调用方覆盖（`owner_override`）或默认等于 semantic_shard。
3. 取 semantic_shard 的 top anchors，第一个作为 `bucket_hint`。
4. **如果 owner == semantic_shard**：直接把 semantic anchors 截断到 hint_count 返回。
5. **如果 owner != semantic_shard**（即调用方强制把请求路由到了一个非几何最近的 shard，例如因为负载均衡）：hints 分两半，一半（`(hint_count+1)/2`）来自 owner shard 的几何最近 anchor，另一半补 semantic shard 的 anchor（去重）。这种"半本地半语义"的混合策略保证：即使请求被强制路由到非几何最优的 owner，搜索起点仍然包含几何最近的入口。

这个 `route()` 函数跟第 14 课查询执行/路由、第 24 课 peer RPC 里的"路由决策"直接对接。`owner_override` 来自计算侧的 storage owner 表（第 28 课），它表示"这个 query 应该由哪个 storage shard 主导"。

### 6.4.6 与 construction 的衔接

`anchor_index.cc` 里的 `Index` 是**读取侧**。写入侧在 `gpu_search/persistent_engine/construction.cc:19`（见第 12 课 construction 上）调用 `index_path::anchor_file(prefix)` 写出 `.anchors` 文件。construction 阶段会：

1. 用 METIS 把图分区到各 shard。
2. 每个 shard 计算 K-medoid，选出 `anchors_per_shard` 个 anchor。
3. 计算 shard 质心。
4. 按 6.4.1 的格式序列化到 `.anchors`。

construction 阶段还有个 `gather_anchor_codes` 步骤（见第 12 课），把 anchor 的 PQ 量化码聚集成"navigation code"区域，写进 `vamana_node.hh` 里 `HOT_GRAPH_DYNAMIC_CODE_OFFSET` 指向的位置——这就是 6.2.9 节 `dynamic_navigation_code_offset` 的数据来源。anchor 的 RemotePtr 既是图搜索起点，也是 navigation code 的索引键。

## 6.5 `idmap.hh`：base ID → 真实 owner / generation 映射

文件路径：`src/vamana/idmap.hh`

这个文件只有 32 行，但概念上极其重要。它定义了 `.idmap` 文件格式，回答"一个原始向量 ID 在分区后到底归哪个 shard 管"。

### 6.5.1 为什么需要 idmap

朴素的分布式向量库常常用 `owner = id % shard_count` 来分配 owner。这在 dvstor 行不通，原因有三：

1. **METIS 分区是几何感知的**。METIS 根据图结构和向量几何把"互相近"的节点放进同一 shard，让图检索尽量在 shard 内收敛。如果用 `id % N`，相邻 ID 的向量会被打散到所有 shard，每个 query 都得跨多 shard 跳，带宽和延迟都炸。METIS 分区后，相邻向量可能都在同一 shard，但 ID 不连续，无法用模运算反推 owner。
2. **delta / 重分区**。第 10 课 delta/动态路由会引入增量重分区：某个节点从 shard A 物理迁移到 shard B，但它的 base ID 不变。这时 owner 必须显式记录，不能由 ID 推导。
3. **generation**。同一 ID 在不同时刻可能对应多版本（旧版本待回收、新版本已上线），idmap 记录当前权威 generation，让 reader 能识别 stale 读。

所以 dvstor 用一张显式的映射表：`id → (rptr_raw, generation, flags)`。`rptr_raw` 直接给出该 ID 当前权威存储位置的 RemotePtr，`generation` 是版本号，`flags` 含墓碑位。

### 6.5.2 文件格式

```cpp
// idmap.hh:7-9
constexpr u32 kMagic = 0x504d4444;  // DDMP
constexpr u32 kVersion = 1;
constexpr u32 kDeleted = 1u << 0;
```

`kMagic = 0x504d4444`，ASCII 是 "DDMP"（小端读为 "PMDD"，大端读为 "DDMP"），"Delta Delta Map" 或 "Data Distribution Map" 的缩写。`kDeleted` 是 Entry.flags 的 bit0，标记该 ID 已逻辑删除（但 entry 还保留以便识别 stale 查询）。

```cpp
// idmap.hh:11-26
#pragma pack(push, 1)
struct Header {
  u32 magic{kMagic};
  u32 version{kVersion};
  u32 owner_shard{};
  u32 shard_count{};
  u64 entry_count{};
};

struct Entry {
  node_t id{};
  u64 rptr_raw{};
  u32 generation{};
  u32 flags{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 24);
static_assert(sizeof(Entry) == 20);
```

`#pragma pack(push, 1)` + `static_assert` 保证磁盘布局确定：Header 24 字节，Entry 20 字节。注意 Header 里有个 `owner_shard` 字段——这说明 `.idmap` 是**按 shard 切片**的：每个 shard 写自己的 `.idmap.<shard>`，里面只放归自己管的 ID。`shard_count` 是全局 shard 总数，`entry_count` 是本 shard 的 entry 数。

Entry 4 个字段：

- `id`：`node_t = u32`，原始向量 ID。
- `rptr_raw`：8 字节 RemotePtr raw_address，指向该 ID 当前权威图记录。
- `generation`：u32 版本号，每次该 ID 被重写时 +1。
- `flags`：u32 标志位，目前只用 bit0 = kDeleted。

### 6.5.3 查找语义

`idmap.hh` 只定义格式，不提供加载/查找 API——那些在 `gpu_search/index_format.cc` 等使用方里实现（见第 7 课 schema-15 索引格式、第 25 课索引访问）。查找语义约定：

1. 按 `id` 二分查找（`.idmap` 离线构建时按 id 排序）。
2. 命中且 `flags & kDeleted == 0`：返回 `(rptr_raw, generation)` 作为权威定位。
3. 命中且 `kDeleted` 置位：ID 已删除，调用方应返回 not-found。
4. 未命中：ID 不存在或不在本 shard，调用方应跨 shard 查找或返回 not-found。

`generation` 用于跟第 16 课 RCU 回收配合：reader 读到 generation G 的记录后，必须确认 idmap 里该 ID 的当前 generation 仍是 G，才能使用读到的数据；否则说明期间该 ID 被改写，读到的可能是 stale 数据，必须重试。

## 6.6 `storage_layout_resolver.hh`：RemotePtr → 具体存储位置

文件路径：`src/vamana/storage_layout_resolver.hh`

这个文件是个纯函数集合，把"RemotePtr + 想读什么字段"翻译成 `(memory_node, offset, size)` 三元组，给 RDMA 读请求用。它是 vamana 字节布局的"读侧 view"。

### 6.6.1 Address 与 NeighborRead

```cpp
// storage_layout_resolver.hh:11-21
class StorageLayoutResolver {
public:
  struct Address {
    u32 memory_node{};
    u64 offset{};
    size_t size{};
  };

  struct NeighborRead {
    Address address;
  };
```

`Address` 就是 RDMA 读请求的三要素：哪个 memory node（shard）、从什么 offset 开始、读多少字节。`NeighborRead` 是个薄包装，目前只有 address 一个字段，保留这个嵌套结构是为了将来扩展（比如邻居区可能拆成多次读）。

### 6.6.2 字段解析函数

```cpp
// storage_layout_resolver.hh:23-39
static Address header(RemotePtr ptr) {
  return {ptr.memory_node(), ptr.byte_offset(), VamanaNode::HEADER_SIZE};
}

static Address id(RemotePtr ptr) {
  return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_id(), VamanaNode::ID_SIZE};
}

static Address generation(RemotePtr ptr) {
  return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_generation(),
          VamanaNode::GENERATION_SIZE};
}

static Address vector(RemotePtr ptr) {
  return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_vector(),
          VamanaNode::vector_bytes()};
}
```

四个字段解析，完全对应 6.2.4 的偏移量：

| 字段 | offset | size |
|---|---|---|
| header | 0 | 8 |
| id | 8 | 4 |
| generation | 12 | 4 |
| vector | 16 | VECTOR_BYTES |

调用方可以这样发 RDMA 读：先 `header(ptr)` 读 8 字节，看 lock/deleted 位；通过后 `id(ptr) + generation(ptr)` 一次 8 字节读（offset 8, size 8）拿到 id+generation；再 `vector(ptr)` 读向量。**这种"按需读字段"避免了一次读整条记录的浪费**——比如查询时只关心向量，就不必读 header/id/generation（虽然实际上 kernel 通常一次读完整记录以减少 RDMA round trip）。

### 6.6.3 neighbor 解析：双布局路由

```cpp
// storage_layout_resolver.hh:41-49
static NeighborRead neighbor_read(RemotePtr ptr) {
  return {{ptr.memory_node(), VamanaNode::hot_graph_entry_offset(ptr),
           VamanaNode::hot_graph_entry_size()}};
}

static Address neighbor_slots(RemotePtr ptr) {
  return {ptr.memory_node(), VamanaNode::hot_graph_entry_offset(ptr),
          VamanaNode::hot_graph_entry_size()};
}
```

`neighbor_read` 和 `neighbor_slots` 都返回同一个 Address——读邻居就是读 hot graph entry。重点在 `VamanaNode::hot_graph_entry_offset(ptr)`（6.2.7），它会自动走静态或动态布局路径。所以 resolver 调用方完全不必关心目标节点是静态分配还是动态分配，这对 kernel 侧的统一逻辑至关重要。

### 6.6.4 边界检查与分配大小

```cpp
// storage_layout_resolver.hh:51-55
static u64 allocation_size() { return VamanaNode::allocation_size(); }

static bool ptr_in_bounds(RemotePtr ptr, u64 shard_cap) {
  return !ptr.is_null() && ptr.byte_offset() + VamanaNode::total_size() <= shard_cap;
}
```

`allocation_size()` 透传，给分配器用。`ptr_in_bounds` 是个安全检查：ptr 非 null，且从该 ptr 开始还能装下一条 `total_size()` 大小的记录，没有越过 shard 内存上限 `shard_cap`。这条检查在解析"从磁盘或网络拿到的 RemotePtr"时很重要——防止被恶意/损坏的指针诱导读越界。

### 6.6.5 ID → owner → 远端记录的解析流程图

把 idmap + resolver + vamana_node 串起来，一个完整的"按 ID 读图记录"流程如下：

```
                  ┌────────────────────────────────────────────────┐
                  │           输入：base ID（node_t）              │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  1. 查 idmap（每 shard 一份 .idmap.<shard>）   │
                  │     - 二分查找 id                              │
                  │     - 命中 → (rptr_raw, generation, flags)    │
                  │     - flags & kDeleted → not-found            │
                  │     - 未命中 → 跨 shard 查找 / not-found      │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            │ rptr_raw → RemotePtr
                                            │ (memory_node, byte_offset)
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  2. StorageLayoutResolver 解析字段偏移         │
                  │     header(ptr)      → (mn, off+0,  8)        │
                  │     id(ptr)          → (mn, off+8,  4)        │
                  │     generation(ptr)  → (mn, off+12, 4)        │
                  │     vector(ptr)      → (mn, off+16, Vbytes)   │
                  │     neighbor_read(ptr)                        │
                  │       → (mn, hot_graph_entry_offset(ptr),     │
                  │          hot_graph_entry_size())              │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            │ RDMA READ
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  3. 远端 storage node 上的 VamanaNode 记录     │
                  │     ┌────────┬────┬────┬────────┬───────────┐  │
                  │     │ header │ id │gen │ vector │ hot entry │  │
                  │     └────────┴────┴────┴────────┴───────────┘  │
                  │     ← total_size →   ← hot_graph_entry_size →  │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  4. generation 校验                            │
                  │     读到的 generation 必须 == idmap generation │
                  │     否则 stale，重试                          │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  5. 解码 hot entry → RemotePtr[] 邻居          │
                  │     decode_hot_graph_entry()                  │
                  │     - checksum16 校验                         │
                  │     - 5B compact ptr → 8B RemotePtr           │
                  │     - 跳过 deleted 邻居                       │
                  └─────────────────────────┬──────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────┐
                  │  6. 对每个邻居 RemotePtr 回到步骤 2           │
                  │     （图检索的主循环，见第 20 课）            │
                  └────────────────────────────────────────────────┘
```

这个流程图把本课所有组件串起来了：idmap 提供 ID→ptr 映射，resolver 提供 ptr→字段偏移，vamana_node 提供字段定义和编解码，hot_graph 提供 compact pointer 的底层编码。第 20 课查询遍历主循环就是按这个流程一跳一跳地走。

## 6.7 `adaptive_route_table.hh` / `.cc`：自适应路由表

文件路径：`src/vamana/adaptive_route_table.hh`、`src/vamana/adaptive_route_table.cc`

这是 vamana 模块里最"算法密集"的部分。它在内存里维护一张"每个 shard 8 个槽"的路由表，根据实时观察到的写入/迁移动态调整槽里的 representative，让查询时能快速找到一个几何近的入口。

### 6.7.1 设计目标与定位

```cpp
// adaptive_route_table.hh:17-22
// A small, live-mutation-driven routing table.  Capacity and adaptation are
// algorithm constants on purpose: routing cost and memory do not grow with the
// number of mutations, and deployment scripts cannot silently change routing
// quality.  Reads take a shared lock; observe/invalidate are serialized by an
// exclusive lock, so callers do not need an external single-writer protocol.
class AdaptiveRouteTable {
public:
  static constexpr u32 kSlotsPerShard = 8;
  static constexpr element_t kCenterEmaWeight = 0.125F;
```

注释把设计意图说得很清楚：

- **容量是算法常量**：每 shard 固定 8 槽，不随 mutation 数量增长。这跟 idmap（每 ID 一条）形成对比——idmap 是"全集映射"，route table 是"采样入口"。
- **不能被部署脚本偷偷改**：路由质量是 SLO，不能因为运维改了配置就变。
- **锁协议**：读用 shared_lock，observe/invalidate 用 unique_lock，调用方不需要外部单写者协议。

`kSlotsPerShard = 8` 和 `kCenterEmaWeight = 0.125` 是两个魔法常量。8 槽意味着每 shard 最多记 8 个 representative；EMA 权重 0.125 = 1/8，意思是新观察对 center 的拉动是 1/8，旧 center 保留 7/8——收敛速度适中。

### 6.7.2 Route / SlotSnapshot / RouteSlotSnapshot

```cpp
// adaptive_route_table.hh:23-62
struct Route {
  u32 shard{};
  node_t id{};
  u32 generation{};
  RemotePtr entry;
  distance_t shard_distance{};
  distance_t entry_distance{};
};

struct SlotSnapshot {
  u32 shard{};
  u32 slot{};
  bool initialized{};
  bool live{};
  node_t id{};
  u32 generation{};
  RemotePtr entry;
  u64 observations{};
  vec<element_t> center;
  vec<element_t> representative;
};

struct RouteSlotSnapshot {
  u32 shard{};
  u32 slot{};
  bool initialized{};
  bool live{};
  node_t id{};
  u32 generation{};
  RemotePtr entry;
};
```

- `Route`：路由结果，包含目标 shard、节点 ID、generation、entry RemotePtr，以及两个距离（shard_distance = 到该 shard 最近 center 的距离；entry_distance = 到 representative 的距离）。这两个距离让调用方可以做混合排序（先按 shard 近，再按 entry 近）。
- `SlotSnapshot`：完整槽位快照，含 center 和 representative 向量。用于调试/序列化。
- `RouteSlotSnapshot`：精简快照，**故意不含向量**。注释说"Centers and representative vectors deliberately stay inside the table"——这是给 GPU 路由发布器用的：GPU 端只需要 (shard, slot, id, generation, entry) 这组元数据，向量留在 CPU 端做距离计算。这跟第 21 课 kernel 侧 `DeviceDynamicRouteSlot` 直接对应。

### 6.7.3 Slot 与 Shard 内部结构

```cpp
// adaptive_route_table.hh:116-129
struct Slot {
  bool initialized{};
  bool live{};
  node_t id{};
  u32 generation{};
  RemotePtr entry;
  u64 observations{};
  vec<element_t> center;
  vec<element_t> representative;
};

struct Shard {
  std::array<Slot, kSlotsPerShard> slots;
};
```

`Slot` 是路由表的最小单元。关键字段：

- `initialized`：是否曾被初始化过（即使后来 invalidate 也保留 true，用于区分"从未用过"和"用过但已删"）。
- `live`：当前是否活跃（未 invalidate）。
- `id` / `generation` / `entry`：该槽当前代表的节点。
- `observations`：该槽累计观察次数（单调增）。
- `center`：该槽的 EMA 中心向量。一个槽的 center 是所有被路由到该槽的观察向量的指数移动平均，代表"这个槽覆盖的子空间中心"。
- `representative`：该槽当前的代表向量——是观察到的那次实际向量，不是 center 的几何投影。center 用于路由（找最近槽），representative 用于返回 entry。

`Shard` 就是 `std::array<Slot, 8>`，固定 8 槽。

### 6.7.4 构造函数：预分配所有内存

```cpp
// adaptive_route_table.cc:12-26
AdaptiveRouteTable::AdaptiveRouteTable(u32 dim, u32 shard_count)
    : dim_(dim), shard_count_(shard_count), shards_(shard_count) {
  if (dim == 0 || shard_count == 0) {
    throw std::invalid_argument(
      "adaptive route table requires non-zero dimension and shard count");
  }
  for (Shard& shard : shards_) {
    for (Slot& slot : shard.slots) {
      // All mutation-time storage is allocated here.  observe() never grows
      // route state, even after an unbounded mutation stream.
      slot.center.resize(dim_);
      slot.representative.resize(dim_);
    }
  }
}
```

注释强调"所有 mutation-time storage 在这里预分配，observe() 永不增长路由状态"。这是热路径的关键保证：observe 在 unbounded mutation 流下也不会触发 allocation，避免 GC 抖动。每个 Slot 预分配 `2 * dim * 4` 字节（center + representative），全表 `shard_count * 8 * 2 * dim * 4` 字节。例如 shard_count=16, dim=128：16*8*2*128*4 = 128 KB，常驻内存可忽略。

### 6.7.5 initialize_slot 与 update_slot

```cpp
// adaptive_route_table.cc:28-42
void AdaptiveRouteTable::initialize_slot(
    Slot& slot,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector) {
  std::copy(vector.begin(), vector.end(), slot.center.begin());
  std::copy(vector.begin(), vector.end(), slot.representative.begin());
  slot.initialized = true;
  slot.live = true;
  slot.id = id;
  slot.generation = generation;
  slot.entry = entry;
  slot.observations = 1;
}
```

`initialize_slot` 用于"空槽/废槽重置"：center 和 representative 都初始化为观察向量本身，observations=1。这是个全新开始，没有历史包袱。

```cpp
// adaptive_route_table.cc:44-71
void AdaptiveRouteTable::update_slot(
    Slot& slot,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector,
    bool force_representative) {
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    const element_t old_center = slot.center[dimension];
    slot.center[dimension] = old_center +
      kCenterEmaWeight * (vector[dimension] - old_center);
  }
  if (slot.observations != std::numeric_limits<u64>::max()) {
    ++slot.observations;
  }

  const distance_t current_distance = L2Distance::dist(
    slot.center, slot.representative, dim_);
  const distance_t candidate_distance = L2Distance::dist(
    slot.center, vector, dim_);
  if (force_representative || candidate_distance <= current_distance) {
    std::copy(vector.begin(), vector.end(), slot.representative.begin());
    slot.id = id;
    slot.generation = generation;
    slot.entry = entry;
    slot.live = true;
  }
}
```

`update_slot` 是核心更新逻辑，分三步：

1. **EMA 更新 center**：`new_center = old_center + 0.125 * (vector - old_center)`。这是标准 EMA 公式，等价于 `new_center = 0.875 * old_center + 0.125 * vector`。逐维做，无 SIMD 优化（dim 不大时可接受）。
2. **observations++**（防溢出：到 u64 上限就停）。
3. **representative 选举**：算当前 representative 到新 center 的距离 `current_distance`，以及候选 vector 到新 center 的距离 `candidate_distance`。如果 `force_representative`（强制覆盖，用于 upsert 自身）或候选更近，就把 representative 换成新 vector。这条策略保证 representative 始终是"离 center 最近的实际观察向量"——既能几何代表该子空间，又是真实可达的节点（有 entry 指针）。

注意 `force_representative` 参数：当 observe 发现"被 upsert 的是该槽当前的 representative 自身"时，无论几何关系如何都必须更新（因为旧 entry 已失效）。这个分支在 6.7.6 的 observe 里被触发。

### 6.7.6 observe：观察一次提交的 mutation

`observe` 是路由表的写入入口（`adaptive_route_table.cc:73-151`）。逐步解析：

```cpp
// adaptive_route_table.cc:73-82
bool AdaptiveRouteTable::observe(
    u32 shard,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector) {
  if (shard >= shard_count_ || vector.size() != dim_ || entry.is_null() ||
      entry.memory_node() != shard) {
    return false;
  }
```

入参校验：shard 在范围、向量维度对、entry 非 null、entry 的 memory_node 等于 shard（防止"路由表说在 shard A，但 entry 指向 shard B"的脏数据）。

```cpp
// adaptive_route_table.cc:84-101
std::unique_lock<std::shared_mutex> lock(mutex_);
bool identity_found = false;
u32 newest_generation = 0;
Slot* same_shard_representative = nullptr;
for (u32 current_shard = 0; current_shard < shard_count_; ++current_shard) {
  for (Slot& slot : shards_[current_shard].slots) {
    if (!slot.initialized || slot.id != id) continue;
    if (!identity_found || slot.generation > newest_generation) {
      newest_generation = slot.generation;
      identity_found = true;
    }
    if (current_shard == shard && slot.live &&
        (same_shard_representative == nullptr ||
         slot.generation > same_shard_representative->generation)) {
      same_shard_representative = &slot;
    }
  }
}
if (identity_found && generation <= newest_generation) {
  return false;
}
```

加写锁。然后**全局扫描所有 shard 的所有槽**，找两个东西：

1. 该 ID 在全表中的最新 generation（`newest_generation`）。如果当前观察的 generation 不大于它，说明这是 stale 观察（旧版本事件迟到），直接丢弃。注释里强调："callers must admit only authoritative committed-current observations for non-representatives"——路由表不维护 per-ID 历史，所以 stale 检测只能靠 generation 比较。
2. 该 ID 在**目标 shard** 内当前 live 的 representative 槽（`same_shard_representative`）。如果存在，说明这次 observe 是对已有 representative 的 upsert。

```cpp
// adaptive_route_table.cc:106-112
if (same_shard_representative != nullptr) {
  // The representative itself was upserted.  Its old vector/pointer cannot
  // remain a route entry even if another observation would be geometrically
  // closer to the center.
  update_slot(*same_shard_representative, id, generation, entry, vector, true);
  return true;
}
```

**分支 A：upsert 自身**。直接 `update_slot(..., force_representative=true)`，强制更新 entry 和 vector。注释解释了为什么 force：representative 的旧 entry 已失效（被新 generation 覆盖），即使另一个观察几何上更近也不能用旧 entry。

```cpp
// adaptive_route_table.cc:116-125
if (identity_found) {
  for (Shard& current_shard : shards_) {
    for (Slot& slot : current_shard.slots) {
      if (slot.initialized && slot.live && slot.id == id) {
        slot.live = false;
        slot.entry.reset();
      }
    }
  }
}
```

**分支 B：ID 在别的 shard 已存在，但目标 shard 没有 live 槽**。说明该 ID 从别的 shard 迁移过来了。先 invalidate 所有 shard 里该 ID 的旧 live 槽（置 live=false, entry.reset()），再在目标 shard 安装新槽。这一步保证同一 ID 在全表只有一个 live entry——避免路由器把读者送到旧物理位置。

```cpp
// adaptive_route_table.cc:127-135
Shard& destination = shards_[shard];
for (Slot& slot : destination.slots) {
  if (!slot.live) {
    // Empty/deleted slots are reset instead of slowly dragging a stale
    // center across the vector space.
    initialize_slot(slot, id, generation, entry, vector);
    return true;
  }
}
```

在目标 shard 找一个非 live 槽（空槽或废槽）。如果找到，`initialize_slot` 全新安装——注释解释为什么不走 update_slot：废槽的 center 可能是迁走前的旧几何中心，继续 EMA 会把新数据"拖"向错误位置，重置反而干净。

```cpp
// adaptive_route_table.cc:137-150
Slot* nearest = &destination.slots.front();
distance_t nearest_distance = L2Distance::dist(
  vector, nearest->center, dim_);
for (u32 slot_index = 1; slot_index < kSlotsPerShard; ++slot_index) {
  Slot& candidate = destination.slots[slot_index];
  const distance_t distance = L2Distance::dist(vector, candidate.center, dim_);
  if (distance < nearest_distance) {
    nearest = &candidate;
    nearest_distance = distance;
  }
}
update_slot(*nearest, id, generation, entry, vector, false);
return true;
}
```

**所有 8 槽都 live 的情况**：找 center 离新 vector 最近的槽，调用 `update_slot(..., force_representative=false)`。这一步是"几何就近合并"——新 vector 进入离它最近的子空间，更新该子空间的 center（EMA），并可能取代该子空间的 representative（如果新 vector 比旧 representative 更近新 center）。

整个 observe 的语义可以总结为一张表：

| 情况 | 动作 |
|---|---|
| generation stale | 丢弃 |
| upsert 自身（同 shard 有 live 同 ID 槽） | force update 该槽 |
| ID 在别 shard live，目标 shard 无 live 槽 | invalidate 旧 + initialize 新 |
| 目标 shard 有空/废槽 | initialize 该槽 |
| 目标 shard 全 live | 找最近 center 槽，update（可能不换 representative） |

### 6.7.7 invalidate：墓碑

```cpp
// adaptive_route_table.cc:153-170
bool AdaptiveRouteTable::invalidate(node_t id, u32 generation) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  bool changed = false;
  for (Shard& shard : shards_) {
    for (Slot& slot : shard.slots) {
      if (!slot.initialized || slot.id != id || generation < slot.generation) {
        continue;
      }
      if (slot.live || generation > slot.generation) {
        changed = true;
      }
      slot.live = false;
      slot.generation = generation;
      slot.entry.reset();
    }
  }
  return changed;
}
```

`invalidate(id, generation)` 是删除通知。逻辑：

- 跳过 uninitialized、不同 ID、generation 小于槽当前 generation 的（迟到的 invalidate 不能撤销较新的状态）。
- `generation == slot.generation` 且 live：置 live=false，changed=true。
- `generation > slot.generation`：即使不 live 也更新 generation（防止更老的 observe 复活），changed=true。

返回 `changed` 表示"表状态是否真变了"。调用方用这个返回值决定是否需要通知 GPU 侧重新发布路由快照（见第 28 课 storage owner 更新）。

### 6.7.8 route：全局路由

```cpp
// adaptive_route_table.cc:172-201
std::optional<AdaptiveRouteTable::Route>
AdaptiveRouteTable::route_in_shard_locked(
    const span<const element_t>& query, u32 shard) const {
  if (shard >= shard_count_ || query.size() != dim_) return std::nullopt;

  const Shard& selected = shards_[shard];
  distance_t shard_distance = std::numeric_limits<distance_t>::max();
  const Slot* nearest_entry = nullptr;
  distance_t entry_distance = std::numeric_limits<distance_t>::max();
  for (const Slot& slot : selected.slots) {
    if (!slot.live) continue;
    shard_distance = std::min(
      shard_distance, L2Distance::dist(query, slot.center, dim_));
    const distance_t distance = L2Distance::dist(
      query, slot.representative, dim_);
    if (nearest_entry == nullptr || distance < entry_distance) {
      nearest_entry = &slot;
      entry_distance = distance;
    }
  }
  if (nearest_entry == nullptr) return std::nullopt;
  return Route{
    .shard = shard,
    .id = nearest_entry->id,
    .generation = nearest_entry->generation,
    .entry = nearest_entry->entry,
    .shard_distance = shard_distance,
    .entry_distance = entry_distance,
  };
}
```

`route_in_shard_locked` 是内部助手（调用方持锁），在指定 shard 内：

- 遍历所有 live 槽，`shard_distance` 取所有 center 到 query 距离的最小值（代表"该 shard 整体有多近"）。
- `nearest_entry` 取 representative 到 query 距离最小的槽。
- 返回 Route，含两个距离。

```cpp
// adaptive_route_table.cc:203-222
std::optional<AdaptiveRouteTable::Route> AdaptiveRouteTable::route(
    const span<const element_t>& query) const {
  if (query.size() != dim_) return std::nullopt;
  std::shared_lock<std::shared_mutex> lock(mutex_);

  std::optional<u32> nearest_shard;
  distance_t nearest_distance = std::numeric_limits<distance_t>::max();
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (const Slot& slot : shards_[shard].slots) {
      if (!slot.live) continue;
      const distance_t distance = L2Distance::dist(query, slot.center, dim_);
      if (!nearest_shard.has_value() || distance < nearest_distance) {
        nearest_shard = shard;
        nearest_distance = distance;
      }
    }
  }
  if (!nearest_shard.has_value()) return std::nullopt;
  return route_in_shard_locked(query, *nearest_shard);
}
```

`route` 是全局路由（共享锁）：

1. 遍历所有 shard 的所有 live 槽，找 center 离 query 最近的 shard。注意这是"槽级"比较，不是"shard 质心级"——一个 shard 有 8 个 center，取最近的那个作为该 shard 的代表。
2. 选定 shard 后调 `route_in_shard_locked` 返回该 shard 内最近的 representative。

这是"两阶段最近邻"：先选 shard（粗），再选 entry（细）。跟 anchor 索引的 `route()` 策略一致，但 anchor 是静态质心，route table 是动态 EMA center。

### 6.7.9 route_in_shard 与 routes_in_shard

```cpp
// adaptive_route_table.cc:224-229
std::optional<AdaptiveRouteTable::Route> AdaptiveRouteTable::route_in_shard(
    const span<const element_t>& query, u32 shard) const {
  if (query.size() != dim_ || shard >= shard_count_) return std::nullopt;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return route_in_shard_locked(query, shard);
}
```

公开 API，加 shared_lock 后调内部助手。注释说"This is the stage-2/peer-search form"——这是第 14 课查询路由的 stage-2 阶段（已经决定了 owner shard，在该 shard 内选入口）和 peer 搜索（peer 节点帮你在它本地选入口）用的接口。

```cpp
// adaptive_route_table.cc:231-258
vec<AdaptiveRouteTable::Route> AdaptiveRouteTable::routes_in_shard(
    const span<const element_t>& query, u32 shard) const {
  vec<Route> routes;
  if (query.size() != dim_ || shard >= shard_count_) return routes;
  std::shared_lock<std::shared_mutex> lock(mutex_);

  routes.reserve(kSlotsPerShard);
  for (const Slot& slot : shards_[shard].slots) {
    if (!slot.live) continue;
    routes.push_back(Route{
      .shard = shard,
      .id = slot.id,
      .generation = slot.generation,
      .entry = slot.entry,
      .shard_distance = L2Distance::dist(query, slot.center, dim_),
      .entry_distance = L2Distance::dist(
        query, slot.representative, dim_),
    });
  }
  std::sort(routes.begin(), routes.end(),
            [](const Route& lhs, const Route& rhs) {
              if (lhs.entry_distance != rhs.entry_distance) {
                return lhs.entry_distance < rhs.entry_distance;
              }
              return lhs.entry.raw_address < rhs.entry.raw_address;
            });
  return routes;
}
```

`routes_in_shard` 返回该 shard **所有 live 槽**的 Route，按 entry_distance 升序排序（相同距离按 raw_address 破平局保证确定性）。注释强调：

> Construction search starts from this complete fixed-capacity route set; graph-search convergence is governed only by construction beam width L, never by an expansion/depth cap.

这是给 **construction 阶段**用的入口集——从所有 8 个 representative 同时开始搜索，覆盖整个 shard 子空间。construction 的收敛只受 beam width L 控制（见第 12 课 construction），不受路由表大小限制。这条不变量很重要：它保证 construction 召回率不会因为 route table 容量被 cap 而下降。

### 6.7.10 快照与统计

```cpp
// adaptive_route_table.cc:260-266
size_t AdaptiveRouteTable::live_count(u32 shard) const {
  if (shard >= shard_count_) return 0;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return static_cast<size_t>(std::count_if(
    shards_[shard].slots.begin(), shards_[shard].slots.end(),
    [](const Slot& slot) { return slot.live; }));
}
```

`live_count` 返回某 shard 的 live 槽数（0-8），用于监控路由覆盖率。

```cpp
// adaptive_route_table.cc:268-290
void AdaptiveRouteTable::snapshot_route_slots(
    span<RouteSlotSnapshot> output) const {
  if (output.size() != capacity()) {
    throw std::invalid_argument(
      "adaptive route metadata snapshot has the wrong capacity");
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (u32 slot_index = 0; slot_index < kSlotsPerShard; ++slot_index) {
      const Slot& slot = shards_[shard].slots[slot_index];
      output[static_cast<size_t>(shard) * kSlotsPerShard + slot_index] =
        RouteSlotSnapshot{
          .shard = shard,
          .slot = slot_index,
          .initialized = slot.initialized,
          .live = slot.live,
          .id = slot.id,
          .generation = slot.generation,
          .entry = slot.entry,
        };
    }
  }
}
```

`snapshot_route_slots` 把全表导出成扁平数组（`shard_count * 8` 个 `RouteSlotSnapshot`），**故意不含向量**（6.7.2 已述）。这个数组就是发布到 GPU 端的"路由元数据"——GPU kernel 侧的 `DeviceDynamicRouteSlot`（见第 21 课 kernel 运行时/角色调度）跟 `RouteSlotSnapshot` 字段一一对应。`output.size() != capacity()` 时抛异常，强制调用方准备好正确大小的缓冲区。

`snapshot()`（`adaptive_route_table.cc:292-318`）是完整快照（含 center/representative 向量），主要用于调试和持久化。

### 6.7.11 与第 10 课 dynamic_route_overlay、第 21 课 DeviceDynamicRouteSlot 的关系

`AdaptiveRouteTable` 是 **CPU 侧、storage-owner canonical** 的路由表。它跟其他几个组件的关系：

| 层 | 组件 | 角色 | 课号 |
|---|---|---|---|
| CPU storage 侧 | `vamana::routing::AdaptiveRouteTable` | 权威路由表，由 observe/invalidate 维护 | 本课 |
| CPU 计算侧 | dynamic_route_overlay | 缓存从 storage 收到的路由快照，叠加本地 query-aware 路由 | 第 10 课 |
| GPU kernel 侧 | `DeviceDynamicRouteSlot` | kernel 读的扁平路由数组，由 `snapshot_route_slots` 发布 | 第 21 课 |

数据流：

1. storage 节点 observe mutation → 更新本地 `AdaptiveRouteTable`。
2. storage 节点 `snapshot_route_slots` → 通过 RPC（第 24 课）把快照发给计算节点。
3. 计算节点收到后写进 `dynamic_route_overlay`（第 10 课），叠加 anchor 路由、query 几何路由形成最终入口集。
4. 计算 kernel 启动前，`dynamic_route_overlay` 把路由元数据拷到 GPU 显存的 `DeviceDynamicRouteSlot[]` 数组（第 21 课）。
5. kernel 在查询遍历主循环（第 20 课）里读这个数组选起点。

所以本课的 `AdaptiveRouteTable` 是这条链的**源头**，8 槽容量、EMA 权重、representative 选举策略是整个系统路由质量的算法根基。任何在 6.7.5 / 6.7.6 里的调整都会通过这条链影响 kernel 的查询召回。

## 6.8 与其他模块的关系

- **第 2 课（公共类型与配置）**：`node_t`、`element_t`、`distance_t`、`VectorDType` 都来自 `common/types.hh` 和 `common/vector_dtype.hh`，本课直接消费。
- **第 4-5 课（RDMA 传输库）**：`RemotePtr` 是 8 字节宽松指针，本课的 compact pointer 是它的 5 字节压缩形态。
- **第 7 课（schema-15 索引格式）**：`.anchors` 和 `.idmap` 都是 schema-15 索引的 sidecar 文件，主格式声明 dim/shard_count/dtype，sidecar 必须与之对齐（6.4.3 的 `expected_dim/expected_shards` 校验）。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`dynamic_navigation_code_offset`（6.2.9）指向的 PQ 量化码由 PQ 模型定义，construction 阶段写入。
- **第 10 课（delta/动态路由/预算）**：`AdaptiveRouteTable` 是 dynamic_route_overlay 的数据源，delta mutation 通过 observe 进入路由表。
- **第 12 课（construction 上）**：construction 写出 `.anchors` 文件，调用 `gather_anchor_codes` 生成 navigation code 区域，配置 `VamanaNode::configure_hot_graph`。
- **第 14 课（查询执行/路由/完成）**：anchor `route()` 和 adaptive `route()` 是查询路由决策的两条并行路径，前者兜底后者动态。
- **第 15 课（增量发布）**：新发布的节点通过 observe 进入路由表，HOT_GRAPH_DYNAMIC_* 字段服务于增量发布的动态布局。
- **第 16 课（存储回收 RCU）**：generation 字段、`HEADER_DELETED`、`HOT_GRAPH_DELETED`、`idmap::kDeleted` 都是 RCU 回收链上的标记；checksum16 是 RCU 读侧的合法性防线。
- **第 18 课（候选评分）**：`L2Distance::dist` 是本课所有距离计算的底层。
- **第 20 课（查询遍历主循环）**：6.6.5 流程图的"步骤 6"就是主循环的一跳。
- **第 21 课（kernel 运行时/角色调度）**：`DeviceDynamicRouteSlot` 与 `RouteSlotSnapshot` 字段一一对应。
- **第 25 课（索引访问/图修改）**：idmap 的查找 API、图记录的写入/修改都在这里实现。
- **第 28 课（计算侧 storage owner 更新）**：`snapshot_route_slots` 的输出通过 storage owner 更新协议下发到计算侧。

## 6.9 小结

本课讲解了 dvstor 索引格式的"骨架"层：

1. **`vamana_node.hh`** 把图记录钉成"header + id + generation + vector + 可选 hot entry + 可选 navigation code"的字节布局，固定记录按 16 字节对齐，hot entry 按 8 字节对齐。所有偏移量通过静态方法暴露，进程级参数（dim, R, dtype）由 `init_static_storage` 一次性配置。
2. **`hot_graph.hh`** 提供 compact pointer 编解码：5 字节 = shard_bits 位 shard + (40 - shard_bits) 位 offset（以 8 字节为单位），null 用全 1 表示。FNV-1a checksum16 防半写。
3. **`anchor_index.hh/cc`** 是静态 anchor 入口索引，`.anchors` 文件格式 = 全局 Header + 每 shard (ShardHeader + centroid + N × (EntryHeader + vector))。`route()` 提供"语义最近 shard + 几何最近 anchor"的混合路由建议，支持 owner_override。
4. **`idmap.hh`** 定义 `.idmap` 文件格式，回答"base ID 在 METIS 分区后归谁管"。每 shard 一个切片，Entry = (id, rptr_raw, generation, flags)。因为 METIS 几何分区 + delta 迁移，owner 不能由 `id % N` 推导。
5. **`storage_layout_resolver.hh`** 是字节布局的读侧 view，把 RemotePtr + 字段名翻译成 RDMA 读请求三元组 (memory_node, offset, size)。`neighbor_read` 自动走静态/动态双布局。
6. **`adaptive_route_table.hh/cc`** 是 storage-owner canonical 的 8 槽自适应路由表。observe 用 EMA(0.125) 更新 center、用几何就近选举 representative；invalidate 用 generation 防迟到墓碑；route 是两阶段最近邻（先 shard 后 entry）；`snapshot_route_slots` 导出扁平元数据给 GPU 侧 `DeviceDynamicRouteSlot`。

这三组数据结构——静态 anchor 索引、显式 idmap、动态 adaptive route table——分别覆盖"冷启动兜底"、"全集 ID 定位"、"热数据入口"三个层次，加上紧凑图记录本身的字节布局，构成了 dvstor 检索路径的格式基石。下一课（第 7 课 schema-15 索引格式）会把这四个 sidecar 文件以及主索引文件统一在一个 schema 版本号下管理。
