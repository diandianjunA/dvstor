# 第 8 课：元数据、owner map 与存储协议

## 8.1 本课目标与涉及文件

至此我们走完了"传输层"（第 4–5 课）和"磁盘索引格式"（第 6–7 课）。本课是连接"静态索引文件"和"运行时存储节点"之间的桥：

- **计算节点如何从 `.meta.json` 知道索引在远端怎么布局**——这是 `index_metadata.hh/cc` 的事。
- **计算节点如何为每个 logical ID 决定它该被路由到哪个存储节点**——这是 `base_owner_map.hh/cc` 的事。
- **计算↔存储之间用什么 wire format 传输 mutation（insert/upsert/erase）请求和响应**——这是 `storage_owner_protocol.hh` 的事。
- **客户端构造 mutation 请求时如何拼装字节、把存储节点回传的 breakdown counters 折算到本地 `Sample` 上**——这是 `storage_owner_client_helpers.hh` 的事。
- **一次查询最终以什么形状返回给上层**——这是 `query_result.hh` 的事。

这五个文件构成了"存算分离"契约的写侧（mutation）和读侧（query result）的最底层。读侧的查询执行细节见第 14、18–20 课；存储节点如何消费这些 wire 消息见第 23–26 课；离线如何产出 `.meta.json` 和 `.idmap` 见第 12–13、29 课。

涉及文件（均为绝对路径）：

- `/home/xjs/experiment/dvstor/src/service/index_metadata.hh`
- `/home/xjs/experiment/dvstor/src/service/index_metadata.cc`
- `/home/xjs/experiment/dvstor/src/service/base_owner_map.hh`
- `/home/xjs/experiment/dvstor/src/service/base_owner_map.cc`
- `/home/xjs/experiment/dvstor/src/service/storage_owner_protocol.hh`
- `/home/xjs/experiment/dvstor/src/service/storage_owner_client_helpers.hh`
- `/home/xjs/experiment/dvstor/src/service/query_result.hh`

辅助/被引用的文件：

- `/home/xjs/experiment/dvstor/src/vamana/idmap.hh`（owner-sharded idmap 的 on-disk header/entry）
- `/home/xjs/experiment/dvstor/src/common/index_path.hh`（idmap 文件命名）
- `/home/xjs/experiment/dvstor/src/vamana/vamana_node.hh`（`VamanaNode::vector_bytes()`/`R`）
- `/home/xjs/experiment/dvstor/src/common/vector_dtype.hh`（dtype 解析与字节数）
- `/home/xjs/experiment/dvstor/src/common/types.hh`（`node_t`/`distance_t`/`element_t`）
- `/home/xjs/experiment/dvstor/src/service/breakdown/names.hh`（`Subcategory` 枚举）
- `/home/xjs/experiment/dvstor/src/service/breakdown/sample.hh`（`Sample`）
- `/home/xjs/experiment/dvstor/rdma-library/library/utils.hh`（`encode_64bit`）
- `/home/xjs/experiment/dvstor/src/service/compute_service/lifecycle.cc`（计算节点启动加载元数据 + base owner map）
- `/home/xjs/experiment/dvstor/src/service/compute_service/index_commands.cc`（`claim_storage_owner_for_mutation` 三段式决策）
- `/home/xjs/experiment/dvstor/src/service/compute_service/storage_owner/public_mutations.cc`（新 ID 的 `id % num_servers_` 确定性分片）

---

## 8.2 `index_metadata.hh/cc`：把 `.meta.json` 翻译成内存布局表

### 8.2.1 `Metadata` 结构体

文件 `src/service/index_metadata.hh:8-47` 定义了一个 POD 风格的 `Metadata` 结构体。它是 `.meta.json` 在内存中的镜像。我们逐组讲解字段。

```cpp
struct Metadata {
  u32 schema_version{15};
  u32 dim{};
  u32 R{};
  u32 beam_width_construction{};
  u32 partition_max_degree{};
  double partition_cross_shard_ratio{};
  u32 num_memory_nodes{};
  u32 node_size{};
  str node_layout{"plain"};
  str storage_format{};
  ...
};
```

**索引基本几何参数组**（第 9–14 行）：

- `schema_version`：默认 15。这是第 7 课讲述的 schema-15 索引格式的版本号；计算节点和存储节点都必须按这个版本解释 on-disk 字节。
- `dim`：向量维度。
- `R`：Vamana 图的最大出度（见第 6、12 课）。
- `beam_width_construction`：构建期 beam width，仅离线用，运行时一般不读，但写到 metadata 里方便对账。
- `partition_max_degree` / `partition_cross_shard_ratio`：分片构建时的裁剪参数（见第 13 课）。

**分片存储拓扑组**（第 15–19 行）：

```cpp
u32 num_memory_nodes{};
u32 node_size{};
str node_layout{"plain"};
str storage_format{};
u32 graph_hot_bytes{};
```

- `num_memory_nodes`：存储节点（memory node）数量。这个数字非常关键——它就是 owner 数量 `N`，后面 `BaseOwnerMap::load` 要用它。
- `node_size`：每个 storage shard 上"一段"的总字节数（详见第 7 课 index_format）。
- `node_layout`：默认 `"plain"`。其他取值（如 `"hot_dynamic"`）会触发不同的图布局路径。
- `storage_format`：例如 `"hot_dynamic_v1"`，决定热图/动态区域的解释方式。
- `graph_hot_bytes`：热图区域字节数。

**向量数据组**（第 20–23 行）：

```cpp
u32 vector_offset{};
VectorDType vector_dtype{VectorDType::float32};
u32 vector_component_size{sizeof(element_t)};
u32 vector_bytes{};
```

- `vector_offset`：向量数据在 `VamanaNode` 内的相对偏移（紧跟 header 之后）。
- `vector_dtype`：默认 `float32`。在 `.cc` 里用 `parse_vector_dtype` 解析（见 `src/common/vector_dtype.hh:33-44`，支持 `float32/uint8/int8`）。
- `vector_component_size`：单分量字节数，默认取 `sizeof(element_t)`（`f32` = 4 字节，`src/common/types.hh:9-11`）。
- `vector_bytes`：一条完整向量的字节数 = `dim * component_size`。

**PQ / 导航模型组**（第 24–28 行）：

```cpp
str navigation_quantizer{};
u32 navigation_code_bytes{};
u32 pq_subquantizers{};
u32 pq_bits{};
u64 navigation_model_checksum{};
```

这些是 PQ 导航模型的元信息，会喂给 GPU 持久化 kernel（见第 9、18、19 课）。`navigation_model_checksum` 用于启动时校验远端模型与本地的版本一致。

**热图布局组**（第 29–39 行）：

```cpp
u32 hot_graph_entry_size{};
u32 hot_graph_pointer_bytes{};
u32 hot_graph_shard_bits{};
vec<u64> hot_graph_offsets;
vec<u64> hot_graph_entry_counts;
vec<u64> hot_graph_dynamic_base_offsets;
vec<u64> storage_control_remote_offsets;
vec<u64> dynamic_node_base_offsets;
u32 hot_graph_dynamic_record_bytes{};
u32 hot_graph_dynamic_hot_offset{};
u32 dynamic_navigation_code_offset{};
u32 allocation_size{};
```

这一组是**本课最核心的"远端 offset/长度表"**。计算节点本身不持有索引数据，所有图/向量字节都在存储节点。这些 `vec<u64>` 是**每个存储 shard 一个条目**的远端偏移表（长度等于 `num_memory_nodes`）：

- `hot_graph_offsets[s]`：第 `s` 个 shard 上热图区域的起始字节偏移。
- `hot_graph_entry_counts[s]`：该 shard 上热图条目数。
- `hot_graph_dynamic_base_offsets[s]`：动态节点区域基址。
- `storage_control_remote_offsets[s]`：该 shard 的 control page 起始偏移（用于 RDMA 读 control page，见第 19 课 rdma_cache、第 23 课 storage 主体）。
- `dynamic_node_base_offsets[s]`：动态节点基址。
- `hot_graph_dynamic_record_bytes` / `hot_graph_dynamic_hot_offset`：动态记录字节大小、热区在动态区内的偏移。
- `dynamic_navigation_code_offset`：动态导航码偏移。
- `allocation_size`：每个 shard 的总分配字节数，默认等于 `node_size`。

`hot_graph_shard_bits` 决定"用 logical ID 的高位选择 shard 内的 hot sub-shard"——这是热图内部的二级分片，与 `num_memory_nodes` 的横向分片是两件事。

**idmap/anchor/navigation 元信息组**（第 41–46 行）：

```cpp
str idmap_format{};
str anchor_format{};
u32 anchor_count_per_shard{};
str navigation_format{};
vec<u64> navigation_code_remote_offsets;
vec<u64> navigation_code_region_bytes;
```

- `idmap_format`：当且仅当为 `"owner_sharded_v1"` 时，`BaseOwnerMap` 才会加载（见 8.3.2）。这是 base idmap 的格式开关。
- `anchor_format` / `anchor_count_per_shard`：anchor 元数据格式与每个 shard 的 anchor 数量。anchor 是 Vamana 的入口点候选（见第 6 课）。
- `navigation_format`：导航码格式。
- `navigation_code_remote_offsets[s]` / `navigation_code_region_bytes[s]`：每个 shard 上 PQ 导航码区域的远端偏移和字节数。

### 8.2.2 `load_metadata` 函数

`src/service/index_metadata.cc:21-100`。这是**纯函数**：给定 `index_prefix`（不含扩展名的索引前缀路径），把 `<prefix>.meta.json` 反序列化进 `Metadata`。

```cpp
bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message) {
  const filepath_t metadata_file = filepath_t(index_prefix.string() + ".meta.json");
  std::ifstream input(metadata_file);
  if (!input.good()) {
    return fail(error_message, "missing index metadata file: " + metadata_file.string());
  }
  ...
}
```

第 22 行把前缀拼成 `.meta.json` 路径。第 23–26 行打开失败则返回错误。`fail` 是个文件内 helper（第 12–17 行）：把错误消息写进 `error_message`（若非空）并返回 `false`，这样调用方既可以选择拿到错误字符串也可以忽略。

主体在 `try`/`catch` 里（第 28–97 行），用 `nlohmann::json` 解析：

```cpp
nlohmann::json json;
input >> json;

metadata.schema_version = json.value("schema_version", 1u);
metadata.dim = json.at("dim").get<u32>();
metadata.R = json.at("R").get<u32>();
metadata.beam_width_construction = json.value("beam_width_construction", 0u);
metadata.partition_max_degree = json.value("partition_max_degree", metadata.R);
metadata.partition_cross_shard_ratio =
  json.value("partition_cross_shard_ratio", 0.0);
metadata.num_memory_nodes = json.at("num_memory_nodes").get<u32>();
metadata.node_size = json.at("node_size").get<u32>();
metadata.node_layout = json.value("node_layout", str{"plain"});
metadata.storage_format = json.at("storage_format").get<str>();
```

注意两种取值风格的区别：

- **`json.at(k).get<T>()`**：键必须存在，否则抛异常（被外层 catch 捕获）。`dim`、`R`、`num_memory_nodes`、`node_size`、`storage_format` 都是强制的——它们没有合理默认值。
- **`json.value(k, default)`**：键缺失就用默认。`schema_version`（默认 1）、`beam_width_construction`（默认 0）、`partition_max_degree`（默认等于 `R`）、`partition_cross_shard_ratio`（默认 0.0）、`node_layout`（默认 `"plain"`）都是可选的。

第 43–49 行处理向量 dtype，配合默认值链：

```cpp
metadata.graph_hot_bytes = json.value("graph_hot_bytes", 0u);
metadata.vector_offset = json.value("vector_offset", 0u);
metadata.vector_dtype = parse_vector_dtype(json.value("vector_data_type", str{"float32"}));
metadata.vector_component_size = json.value(
  "vector_component_size", static_cast<u32>(vector_dtype_component_size(metadata.vector_dtype)));
metadata.vector_bytes = json.value(
  "vector_bytes", static_cast<u32>(vector_dtype_bytes(metadata.vector_dtype, metadata.dim)));
```

这里的默认值链很巧妙：先解析 `vector_data_type` → 得到 `vector_dtype`，再用 dtype 推 `vector_component_size` 和 `vector_bytes`。如果 JSON 显式给了 `vector_component_size` / `vector_bytes`，会覆盖推算值——这给 int8/uint8 量化向量留了自定义空间。

第 58–75 行处理那五个 `vec<u64>` 数组：

```cpp
if (json.contains("hot_graph_offsets")) {
  metadata.hot_graph_offsets = json["hot_graph_offsets"].get<vec<u64>>();
}
if (json.contains("hot_graph_entry_counts")) {
  metadata.hot_graph_entry_counts = json["hot_graph_entry_counts"].get<vec<u64>>();
}
...
```

用 `contains` 判断而非 `value`，因为 `nlohmann::json::value` 不能直接返回复杂容器。这些数组缺失时就保持默认的空 `vec`，调用方需要自行处理。

最后第 95–97 行：

```cpp
} catch (const std::exception& e) {
  return fail(error_message, "failed to parse index metadata " + metadata_file.string() + ": " + e.what());
}
```

任何解析异常都被翻译成"failed to parse ..."错误字符串，函数返回 `false`。这让 `load_metadata` 永不抛异常——调用方（如 `src/service/compute_service/lifecycle.cc:36-38`）可以用 `lib_assert(load_metadata(...), error_message)` 直接断言。

### 8.2.3 `Metadata` 与第 7 课 index_format 的关系

`Metadata` 几乎是 schema-15 索引格式的目录页（见第 7 课）：

- `schema_version`/`storage_format`/`node_layout` 决定如何解释 on-disk 字节。
- `hot_graph_*`、`vector_offset`、`dynamic_*` 字段定义了**单 shard 内的字节布局**（这是第 7 课的内容）。
- `hot_graph_offsets`/`storage_control_remote_offsets`/`navigation_code_remote_offsets` 等 `vec<u64>` 定义了**多 shard 之间的远端偏移表**——这是本课独有的、用于存算分离的拓扑信息：计算节点据此向存储节点 `s` 发起 RDMA 读，偏移量就是这些表里的对应元素。RDMA cache（第 19 课）和 query traversal（第 20 课）会反复查这些表。
- `idmap_format="owner_sharded_v1"` 是触发 owner map 加载的开关——这就是 8.3 的入口。

`Metadata` 加载之后，下游还会做一个 `validate_index_metadata` 校验（见 `lifecycle.cc:34`），再喂给 `PersistentSearchEngine`（`lifecycle.cc:63-64`）。`Metadata` 不可变（没有 setter），在整个进程生命周期内被多处只读引用。

---

## 8.3 `base_owner_map.hh/cc`：base ID → owner 映射

### 8.3.1 类接口

`src/service/base_owner_map.hh:13-37`：

```cpp
// Immutable after load. A two-level byte table keeps the common dense-ID
// representation at one byte per ID without making a sparse/corrupt high ID
// allocate a multi-gigabyte flat vector.
class BaseOwnerMap {
public:
  bool load(const filepath_t& index_prefix,
            u32 owner_count,
            const str& idmap_format,
            str* error_message = nullptr);

  std::optional<u32> owner_for(node_t id) const;
  size_t entry_count() const { return entry_count_; }
  size_t memory_bytes() const;
  bool empty() const { return entry_count_ == 0; }

private:
  static constexpr u8 kMissingOwner = std::numeric_limits<u8>::max();
  static constexpr u32 kPageBits = 16;
  static constexpr size_t kPageSize = size_t{1} << kPageBits;
  using Page = std::array<u8, kPageSize>;

  std::vector<std::unique_ptr<Page>> pages_;
  size_t allocated_pages_{};
  size_t entry_count_{};
};
```

注释点明设计意图：**这是一个不可变的、每 ID 一字节的查找表，但用二级页表避免稀疏/坏 ID 一次性分配几个 G 的扁平 vector**。

- `kMissingOwner = 255`：缺失 ID 的哨兵值（因此 owner 数最多 255，见下文校验）。
- `kPageBits = 16`、`kPageSize = 65536`：每页 64 KiB，恰好容纳 65536 个 owner 字节，即一页覆盖 65536 个连续 logical ID。
- `pages_`：`vector<unique_ptr<Page>>`。`pages_[id >> 16]` 指向那一页；未分配的页指针为空（节省内存）。
- `allocated_pages_` / `entry_count_`：实际分配的页数与已登记的 ID 数，用于 `memory_bytes()` 和日志。

为什么是"二级"：高 16 位选页，低 16 位页内偏移。这样如果一个 ID 是 2^32 量级的稀疏值，只要它周围的邻居都不存在，那一页就永远不分配——内存占用与实际 ID 密度成正比，而不是与最大 ID 成正比。

### 8.3.2 `load` 的前置校验

`src/service/base_owner_map.cc:24-41`：

```cpp
bool BaseOwnerMap::load(const filepath_t& index_prefix,
                        u32 owner_count,
                        const str& idmap_format,
                        str* error_message) {
  if (idmap_format != "owner_sharded_v1") {
    return fail(error_message,
                "compute mutations require idmap_format=owner_sharded_v1; "
                "index metadata reports '" + idmap_format + "'");
  }
  if (index_prefix.empty()) {
    return fail(error_message, "owner idmap index prefix is empty");
  }
  // Values 0..254 are owners and 255 is the absent-ID sentinel.
  if (owner_count == 0 ||
      owner_count > static_cast<u32>(std::numeric_limits<u8>::max())) {
    return fail(error_message,
                "owner-sharded idmap requires between 1 and 255 owners");
  }
  ...
}
```

三个前置校验：

1. **`idmap_format` 必须是 `"owner_sharded_v1"`**。这是 `Metadata.idmap_format` 直接传进来的。如果是别的格式（比如不分片的扁平 idmap），owner map 拒绝加载——这会连锁地让计算节点 `lib_assert` 失败退出。也就是说，**离线构建必须用 owner-sharded idmap 才能开 updates**（见第 13、29 课）。
2. **前缀非空**：防止配置错误。
3. **`owner_count ∈ [1, 255]`**：因为 owner 编码进 `u8`，且 255 保留给哨兵，所以最多 255 个 owner。这把存储节点数硬性上限设到 255。`num_memory_nodes` 必须落在这个区间。

### 8.3.3 `load` 的主体：扫描所有 owner shard

`src/service/base_owner_map.cc:43-148` 是核心。先看骨架：

```cpp
BaseOwnerMap loaded;
try {
  for (u32 owner = 0; owner < owner_count; ++owner) {
    const filepath_t path = index_path::owner_idmap_file(
      index_prefix, static_cast<size_t>(owner) + 1, owner_count);
    ...
    // 读 header、校验、读 entries、写入 loaded.pages_
  }
} catch (const std::bad_alloc&) {
  return fail(error_message,
              "insufficient memory while loading owner-sharded idmaps");
}

*this = std::move(loaded);
return true;
```

关键设计：

1. **构造临时 `loaded`，最后 `std::move` 给 `*this`**。这样如果中途任何一步失败，`*this` 保持原状（要么空、要么上一次成功加载的状态），不会留下半成品。这是一种"事务性加载"。
2. **外层只捕获 `bad_alloc`**。其他错误（文件缺失、header 坏、重复 ID）都用 `return fail(...)` 显式返回，错误字符串更精确。
3. **遍历顺序 `owner = 0..owner_count-1`**。每个 owner 对应一个独立的 `.idmap` 文件。

#### 8.3.3.1 文件命名

```cpp
const filepath_t path = index_path::owner_idmap_file(
  index_prefix, static_cast<size_t>(owner) + 1, owner_count);
```

`index_path::owner_idmap_file` 在 `src/common/index_path.hh:31-34`：

```cpp
inline filepath_t owner_idmap_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
  return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
                    std::to_string(num_nodes) + ".idmap");
}
```

所以 `owner=0` 对应 `<prefix>_node1_of4.idmap`（1-based ordinal），`owner=1` 对应 `_node2_of4.idmap`，依此类推。**文件名里写死了 num_nodes**——这是防呆：如果你换了一组存储节点数不同的索引，文件名对不上，立即就能发现。

#### 8.3.3.2 文件大小预检与 header 读取

`src/service/base_owner_map.cc:48-78`：

```cpp
std::error_code size_error;
const std::uintmax_t actual_bytes =
  std::filesystem::file_size(path, size_error);
if (size_error) {
  return fail(error_message,
              "missing owner idmap sidecar: " + path.string());
}
if (actual_bytes < sizeof(vamana::idmap::Header)) {
  return fail(error_message,
              "truncated owner idmap header: " + path.string());
}

std::ifstream input(path, std::ios::binary);
if (!input.good()) {
  return fail(error_message,
              "failed to open owner idmap sidecar: " + path.string());
}
vamana::idmap::Header header{};
input.read(reinterpret_cast<char*>(&header), sizeof(header));
if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
  return fail(error_message,
              "truncated owner idmap header: " + path.string());
}
if (header.magic != vamana::idmap::kMagic ||
    header.version != vamana::idmap::kVersion ||
    header.owner_shard != owner ||
    header.shard_count != owner_count) {
  return fail(error_message,
              "invalid owner idmap header (magic/version/owner/shard): " +
              path.string());
}
```

`vamana::idmap::Header` 在 `src/vamana/idmap.hh:11-17`：

```cpp
constexpr u32 kMagic = 0x504d4444;  // DDMP
constexpr u32 kVersion = 1;
constexpr u32 kDeleted = 1u << 0;

#pragma pack(push, 1)
struct Header {
  u32 magic{kMagic};
  u32 version{kVersion};
  u32 owner_shard{};
  u32 shard_count{};
  u64 entry_count{};
};
```

四重校验：magic（`0x504d4444` = `"DDMP"` ASCII，注意小端字节序）、version、本文件归属的 owner 编号、shard 总数。最后两项是**跨文件一致性检查**——确保所有 sidecar 文件是同一次构建产出的、且对应到正确的 owner。

第 79–94 行做文件大小公式校验：

```cpp
constexpr std::uintmax_t kHeaderBytes = sizeof(vamana::idmap::Header);
constexpr std::uintmax_t kEntryBytes = sizeof(vamana::idmap::Entry);
if (header.entry_count >
    (std::numeric_limits<std::uintmax_t>::max() - kHeaderBytes) /
      kEntryBytes) {
  return fail(error_message,
              "owner idmap entry count overflows file size: " +
              path.string());
}
const std::uintmax_t expected_bytes =
  kHeaderBytes + header.entry_count * kEntryBytes;
if (actual_bytes != expected_bytes) {
  return fail(error_message,
              "owner idmap file size mismatch (truncated or trailing data): " +
              path.string());
}
```

`expected = sizeof(Header) + entry_count * sizeof(Entry)`，必须和 OS 报告的文件大小**完全相等**。这能抓到两类问题：截断（actual < expected）和尾部垃圾（actual > expected）。先做溢出检查（第 81–87 行）防止 `entry_count * kEntryBytes` 在 `uintmax_t` 上溢出。

#### 8.3.3.3 分批读取 entries

`src/service/base_owner_map.cc:96-139`：

```cpp
std::array<vamana::idmap::Entry, 4096> entry_buffer{};
u64 entries_remaining = header.entry_count;
while (entries_remaining != 0) {
  const size_t entries_to_read = static_cast<size_t>(std::min<u64>(
    entries_remaining, entry_buffer.size()));
  const size_t bytes_to_read =
    entries_to_read * sizeof(vamana::idmap::Entry);
  input.read(reinterpret_cast<char*>(entry_buffer.data()),
             static_cast<std::streamsize>(bytes_to_read));
  if (input.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
    return fail(error_message,
                "truncated owner idmap entry: " + path.string());
  }

  for (size_t entry_index = 0; entry_index < entries_to_read;
       ++entry_index) {
    const auto& entry = entry_buffer[entry_index];
    ...
  }
  entries_remaining -= entries_to_read;
}
```

用 4096 个 `Entry` 的栈缓冲（约 64 KiB）分批读取，避免一次性把上百万 entry 全 `read` 进堆。每批读完立刻在 `for` 循环里逐条写入 `loaded.pages_`。`vamana::idmap::Entry` 在 `src/vamana/idmap.hh:19-24`：

```cpp
struct Entry {
  node_t id{};
  u64 rptr_raw{};
  u32 generation{};
  u32 flags{};
};
```

注意 `Entry` 里其实带了 `rptr_raw`/`generation`/`flags`，但 `BaseOwnerMap` **只关心 `id`**——它要做的是 ID → owner 映射，不是 ID → rptr 映射。`rptr_raw`/`generation` 这些信息由其他渠道（control page、compute_side_idmap）在运行时获得（见第 28 课）。所以 base idmap 在这里是"被剪裁了用途"的：离线产物完整，但运行时只用 `id` 那一列。

#### 8.3.3.4 写入 page table

`src/service/base_owner_map.cc:113-137`：

```cpp
const auto& entry = entry_buffer[entry_index];
const size_t page_index =
  static_cast<size_t>(entry.id >> kPageBits);
const size_t page_offset =
  static_cast<size_t>(entry.id & (kPageSize - 1));
if (loaded.pages_.size() <= page_index) {
  loaded.pages_.resize(page_index + 1);
}
if (!loaded.pages_[page_index]) {
  loaded.pages_[page_index] = std::make_unique_for_overwrite<Page>();
  loaded.pages_[page_index]->fill(kMissingOwner);
  ++loaded.allocated_pages_;
}
u8& existing = (*loaded.pages_[page_index])[page_offset];
if (existing != kMissingOwner) {
  const str duplicate_kind = existing == static_cast<u8>(owner)
    ? "duplicate ID in owner idmap"
    : "conflicting owner for duplicate ID in owner idmaps";
  return fail(error_message,
              duplicate_kind + ": id=" + std::to_string(entry.id) +
              " previous_owner=" + std::to_string(existing) +
              " owner=" + std::to_string(owner));
}
existing = static_cast<u8>(owner);
++loaded.entry_count_;
```

逻辑：

1. `page_index = id >> 16`、`page_offset = id & 0xFFFF`：二级页表的两次位移。
2. `pages_` 容量不够就 `resize`——`resize` 只增加 `unique_ptr` 槽位（每个 8 字节），新槽位是空指针，**不分配页内存**。
3. 第一次命中某页时 `make_unique_for_overwrite<Page>()` 分配 64 KiB，用 `fill(kMissingOwner)` 把整页填成 255（表示"未映射"）。`make_unique_for_overwrite` 比 `make_unique` 略快，因为它不调用构造函数（`array<u8, N>` 默认构造本来也不做任何事，但写法上更明确）。
4. 检查 `existing`：如果已经被某个 owner 占了，说明同一 ID 出现两次。这里分两种错误信息：
   - **`existing == owner`**：同一 owner 文件里有重复 ID → `"duplicate ID in owner idmap"`。这是构建 bug。
   - **`existing != owner`**：不同 owner 文件里出现了同一 ID → `"conflicting owner for duplicate ID in owner idmaps"`。这是 owner 分配 bug，会导致所有权分裂。
   
   两种都是致命错误，必须重新构建索引。
5. 写入 owner 编号，`entry_count_++`。

第 141–144 行捕获 `bad_alloc`：

```cpp
} catch (const std::bad_alloc&) {
  return fail(error_message,
              "insufficient memory while loading owner-sharded idmaps");
}
```

`make_unique_for_overwrite<Page>()` 在内存不足时抛 `bad_alloc`，这里翻译成可读错误。

最后第 146–147 行：

```cpp
*this = std::move(loaded);
return true;
```

`std::move` 把 `loaded.pages_` 整个搬进 `*this`，零拷贝。如果中途失败，`loaded` 在栈上析构，自动释放已分配的页。

### 8.3.4 `owner_for` 查询

`src/service/base_owner_map.cc:150-159`：

```cpp
std::optional<u32> BaseOwnerMap::owner_for(node_t id) const {
  const size_t page_index = static_cast<size_t>(id >> kPageBits);
  if (page_index >= pages_.size() || !pages_[page_index]) {
    return std::nullopt;
  }
  const size_t page_offset = static_cast<size_t>(id & (kPageSize - 1));
  const u8 owner = (*pages_[page_index])[page_offset];
  if (owner == kMissingOwner) return std::nullopt;
  return static_cast<u32>(owner);
}
```

三段式：

1. 高 16 位越界 → `nullopt`（ID 比已加载的任何页都大）。
2. 页指针为空 → `nullopt`（这一段 ID 区间没有任何 base idmap 条目）。
3. 页内值是 `kMissingOwner` → `nullopt`（页是稀疏填充的，这个具体 ID 未映射）。

返回 `optional<u32>` 的语义是"这个 ID 在 base idmap 里有 owner 吗"。返回 `nullopt` 不代表"这个 ID 不存在"，而是"base idmap 不知道它"——这正好对应"新插入的 ID"：它还没进 base idmap（base idmap 是离线产物，运行时新增的 ID 不会回写到 base idmap 文件）。这就是 8.4 要讲的"新 ID 的确定性分片"路径的触发条件。

### 8.3.5 `memory_bytes`

`src/service/base_owner_map.cc:161-164`：

```cpp
size_t BaseOwnerMap::memory_bytes() const {
  return pages_.size() * sizeof(std::unique_ptr<Page>) +
         allocated_pages_ * sizeof(Page);
}
```

两部分：`pages_` 的指针数组（`vector` 容量 × 8 字节）+ 实际分配的页数 × 64 KiB。`lifecycle.cc:43-45` 打印这个值，方便估算 owner map 的内存占用。

### 8.3.6 base idmap vs 新 ID 的确定性分片——`placement` 策略

把 base idmap 和"新 ID 路由"放在一起看。`src/service/compute_service/lifecycle.cc:39-55`：

```cpp
if (config_.enable_updates) {
  lib_assert(base_owner_map_.load(startup_prefix, num_servers_,
                                  metadata.idmap_format, &metadata_error),
             metadata_error);
  print_status("storage-owner base idmap: entries=" +
               std::to_string(base_owner_map_.entry_count()) + " memory=" +
               std::to_string(base_owner_map_.memory_bytes()) + " bytes");
  // Logical-ID placement must be identical on every compute node. Dynamic
  // routes are intentionally used for graph/search entry selection, not for
  // authoritative identity ownership; otherwise independently evolving
  // compute-local centers could create the same generation on two owners.
  print_status(
    "storage-owner placement: base idmap for existing IDs; "
    "deterministic ID shard for new IDs");
} else {
  print_status("compute updates disabled: owner idmaps and update executor are not loaded");
}
```

注释和日志是这一课最重要的设计声明：

> Logical-ID placement must be identical on every compute node. Dynamic routes are intentionally used for graph/search entry selection, not for authoritative identity ownership.

——**logical ID → owner 的决策必须在所有计算节点上一致**。否则两个计算节点可能同时把同一个新 ID 路由到不同 owner，造成"所有权分裂"。查询时的 dynamic route（动态路由，见第 10、14 课）是用来选搜索入口/邻居的，不是用来决定所有权的。

placement 策略有两段：

1. **已存在的 ID（base idmap 覆盖）**：`owner_for(id)` 返回非空，直接用。
2. **新 ID（base idmap 没覆盖）**：用 `id % num_servers_` 确定性分片。

`src/service/compute_service/storage_owner/public_mutations.cc:64-80` 给出了实际代码：

```cpp
for (const auto& item : items) {
  const auto operation_started = std::chrono::steady_clock::now();
  u32 owner_storage = 0;
  const std::optional<u32> known_owner =
    known_storage_owner_for_id(item.id);
  if (known_owner.has_value()) {
    owner_storage = *known_owner;
  } else {
    // Every compute node proposes the same owner for an unseen ID. The
    // process-local claim still serializes racing mutations here, while the
    // deterministic proposal prevents split ownership across compute nodes.
    const u32 proposed_owner = num_servers_ == 0
      ? 0
      : static_cast<u32>(item.id % num_servers_);
    owner_storage = claim_storage_owner_for_mutation(
      item.id, proposed_owner);
  }
  ...
}
```

注释明确说："Every compute node proposes the same owner for an unseen ID."——`id % num_servers_` 是个纯函数，所有计算节点输入相同 ID 得到相同 owner。这就是"确定性 ID shard"的含义。`claim_storage_owner_for_mutation` 在 `src/service/compute_service/index_commands.cc:54-75`：

```cpp
u32 ComputeService::claim_storage_owner_for_mutation(
  node_t id, u32 proposed_owner) {
  auto& shard =
    compute_side_idmap_[static_cast<size_t>(id) % kComputeSideIdShardCount];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto existing = shard.entries.find(id);
  if (existing != shard.entries.end()) {
    return existing->second.owner_storage;
  }
  // An immutable base owner is authoritative even before this compute
  // process has observed a runtime mutation for the ID.
  if (const auto base_owner = base_owner_map_.owner_for(id)) {
    return *base_owner;
  }
  // Generation zero is a local routing claim, not a published mutation. The
  // first successful storage response starts at generation one and replaces
  // it. This closes the window in which concurrent first mutations for the
  // same ID could choose different owners on this compute service.
  shard.entries.emplace(
    id, ComputeSideIdEntry{RemotePtr{}, true, proposed_owner, 0});
  return proposed_owner;
}
```

这是一个**三段式决策**：

1. **compute_side_idmap 已有运行时记录** → 直接用记录里的 owner。这覆盖"本进程已经发起过 mutation 并收到了响应"的情况。
2. **base idmap 有记录** → 用 base owner。这覆盖"索引离线构建时已存在、本进程还没发过 mutation"的情况。注释强调："An immutable base owner is authoritative even before this compute process has observed a runtime mutation for the ID."
3. **两者都没有** → 用 `proposed_owner`（即 `id % num_servers_`），并在 `compute_side_idmap` 里写一个 `generation=0` 的占位。`generation=0` 是个本地路由 claim，不是已发布的 mutation；存储节点第一次成功响应会以 `generation=1` 覆盖它。

这样三段叠加，保证：**同一个 ID 在同一个计算进程内的多次 mutation 一定路由到同一个 owner；不同计算进程对同一个新 ID 也提议同一个 owner**。这就是日志"storage-owner placement: base idmap for existing IDs; deterministic ID shard for new IDs"的全部含义。

`known_storage_owner_for_id`（`index_commands.cc:42-52`）是配套的只读查询：

```cpp
std::optional<u32> ComputeService::known_storage_owner_for_id(
  node_t id) const {
  {
    const auto& shard =
      compute_side_idmap_[static_cast<size_t>(id) % kComputeSideIdShardCount];
    std::lock_guard<std::mutex> lock(shard.mutex);
    const auto it = shard.entries.find(id);
    if (it != shard.entries.end()) return it->second.owner_storage;
  }
  return base_owner_map_.owner_for(id);
}
```

它先查 compute_side_idmap（运行时已确认的），再回退到 base_owner_map（不可变基线）。它和 `claim_storage_owner_for_mutation` 的差别仅在于：未命中时不写 `generation=0` 占位——所以它适合"只读路由"，而 `claim_*` 适合"准备发起 mutation"。

---

## 8.4 `storage_owner_protocol.hh`：计算↔存储 RPC wire protocol

这个文件定义了**计算节点向存储节点发起 mutation 请求、接收响应**的全部字节布局。它是纯头文件，全是 POD struct + inline 函数，没有成员函数——所有"操作"都是对 `void* payload` 的 `reinterpret_cast`。这种风格在零拷贝 RDMA 代码里很常见：消息就是一段连续字节，header + 紧随其后的数组。

### 8.4.1 magic 与版本

`src/service/storage_owner_protocol.hh:10-13`：

```cpp
constexpr u32 kInsertMagic = 0x53494e54;  // "SINT"
constexpr u32 kMutationMagic = 0x4d555444;  // D T U M / "DUTM"
constexpr u32 kPeerRpcMagic = 0x53505250;  // "SPRP"
constexpr u32 kPeerRpcVersion = 3;
```

三种 magic 对应三类消息：

- `kInsertMagic`（`"SINT"`）：旧式 insert 批量请求/响应。仍保留是为了向后兼容。
- `kMutationMagic`（`"DUTM"`）：**本课重点**。insert/upsert/erase 三种 mutation 的统一批量请求。
- `kPeerRpcMagic`（`"SPRP"`）：**存储节点之间**的 peer RPC（reverse update / stitch search / cleanup deleted）。这是第 24 课 peer RPC 的内容，本课只做结构介绍。

`kPeerRpcVersion = 3`：peer RPC 有版本号，便于演进。mutation/insert RPC 没有显式 version 字段，而是用 `static_assert` 锁死布局（见 8.4.3）。

### 8.4.2 状态与类型枚举

`src/service/storage_owner_protocol.hh:15-42`：

```cpp
enum class InsertStatus : u32 {
  ok = 0,
  failed = 1,
  overloaded = 2,
};

enum class MutationKind : u32 {
  insert = 1,
  upsert = 2,
  erase = 3,
};

enum class MutationStatus : u32 {
  ok = 0,
  not_found = 1,
  already_exists = 2,
  already_deleted = 3,
  failed = 4,
};

enum class PeerRpcType : u32 {
  reverse_update_request = 1,
  reverse_update_response = 2,
  cleanup_deleted_request = 3,
  stitch_search_request = 4,
  stitch_search_response = 5,
  cleanup_deleted_response = 6,
};
```

`MutationKind` 是请求侧的"做什么"，`MutationStatus` 是响应侧的"做得怎么样"。注意 `MutationKind::insert=1`、`upsert=2`、`erase=3`——和 `MutationStatus::ok=0` 的取值空间不冲突，但它们是不同 enum 类型，不会被混用。

`MutationStatus::already_deleted` 是 erase 路径特有的：本计算节点以为 ID 还在，但远端已经标删除了——这通常意味着删除 mutation 比 erase 请求先到，是合法竞态。

`PeerRpcType` 有 6 个值，三对 request/response：reverse_update（反向边更新）、stitch_search（跨 shard 缝合搜索）、cleanup_deleted（已删节点清理）。详见第 24 课。

### 8.4.3 请求 header 与 schema-15 兼容性

`src/service/storage_owner_protocol.hh:44-75`：

```cpp
struct InsertBatchRequestHeader {
  u32 magic{kInsertMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchor_hint_count{};
  u64 batch_id{};
};

struct MutationBatchRequestHeader {
  u32 magic{kMutationMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchor_hint_count{};
  u64 batch_id{};
};

// Schema 15 compatibility: anchor_hint_count remains in both request headers
// at the original byte offset, but this implementation only accepts zero.
static_assert(sizeof(InsertBatchRequestHeader) == 40);
static_assert(sizeof(MutationBatchRequestHeader) == 40);
static_assert(offsetof(InsertBatchRequestHeader, anchor_hint_count) == 28);
static_assert(offsetof(MutationBatchRequestHeader, anchor_hint_count) == 28);
static_assert(offsetof(InsertBatchRequestHeader, batch_id) == 32);
static_assert(offsetof(MutationBatchRequestHeader, batch_id) == 32);
```

两个 header 字段完全一样，只是 magic 不同：

- `magic`：版本/类型标识，校验时必须匹配。
- `dim`：向量维度，存储节点据此解释紧随其后的 vectors 段。
- `owner_storage`：目标存储节点的 owner 编号（就是 `BaseOwnerMap` 给出的那个）。**注意：请求里再写一遍 owner_storage**——这是因为 RDMA send/recv 不一定保留"发到哪个 QP"的信息，header 里再写一遍方便存储节点内部路由。
- `source_client`：发起请求的计算节点编号，方便存储节点回溯/记账。
- `item_count`：本批次包含多少个 mutation item。
- `vector_dtype` / `vector_bytes`：dtype 编码（`VectorDType` 的 u32 表示）和单向量字节数。
- `anchor_hint_count`：**schema-15 兼容字段，本实现只接受 0**。注释明确："anchor_hint_count remains in both request headers at the original byte offset, but this implementation only accepts zero."——历史上这里曾用来传 anchor hint，现在 anchor 由存储节点自己决定，字段保留只为不破坏 wire 布局。
- `batch_id`：批次 ID，用于关联响应和请求、做超时统计。

下面的 `static_assert` 是 schema-15 兼容性的核心**编译期护栏**：

- 两个 header 都恰好 40 字节。
- `anchor_hint_count` 在 offset 28。
- `batch_id` 在 offset 32。

任何人改动这两个 header（加字段、调顺序、改类型）都会触发编译错误。这是 wire protocol 演进的纪律：**字段可以废弃，但字节偏移不能动**，否则老存储节点和新计算节点会读错位。

### 8.4.4 响应 header 与 breakdown counters

`src/service/storage_owner_protocol.hh:77-83`：

```cpp
struct InsertBatchResponseHeader {
  u32 magic{kInsertMagic};
  u32 owner_storage{};
  u32 item_count{};
  u32 reserved{};
  u64 batch_id{};
};
```

响应 header 也是 16 字节（u32 × 4 + u64）。注意它用 `kInsertMagic`——也就是说**响应的 magic 和 insert 请求相同**（不论请求是 insert 还是 mutation）。响应通过 `batch_id` 与请求配对，而不是通过 magic 区分类型。

`src/service/storage_owner_protocol.hh:85-91`：

```cpp
struct MutationResult {
  u64 new_rptr_raw{};
  u64 old_rptr_raw{};
  u32 generation{};
  u32 reserved{};
  u64 maintenance_sequence{};
};
```

每个 item 一个 `MutationResult`：

- `new_rptr_raw`：mutation 后的新远端指针（指向存储节点上该节点的新位置/新版本）。
- `old_rptr_raw`：被替换的旧远端指针（用于 RCU 回收，见第 16 课）。
- `generation`：该 ID 的新 generation 号（见第 10 课 delta/动态路由）。
- `maintenance_sequence`：本次 mutation 触发的维护任务序号（见第 26 课维护）。

`src/service/storage_owner_protocol.hh:93-143` 是 `InsertBreakdownCounters`，存储节点回传的细分耗时统计：

```cpp
struct InsertBreakdownCounters {
  u64 storage_owner_queue_wait_ns{};
  u64 storage_owner_medoid_ns{};
  u64 storage_owner_search_ns{};
  u64 storage_owner_prune_ns{};
  u64 storage_owner_write_node_ns{};
  u64 storage_owner_local_reverse_ns{};
  u64 storage_owner_remote_reverse_ns{};
  u64 storage_owner_peer_reverse_apply_ns{};
  u64 storage_owner_response_send_ns{};
  u64 storage_owner_prepare_mutation_ns{};
  u64 storage_owner_allocate_node_ns{};
  u64 storage_owner_publish_mutation_ns{};
  u64 storage_owner_schedule_maintenance_ns{};
  u64 storage_owner_response_build_ns{};

  u64 storage_owner_search_select_ns{};
  u64 storage_owner_search_neighbor_read_ns{};
  u64 storage_owner_search_snapshot_read_ns{};
  u64 storage_owner_search_distance_ns{};
  u64 storage_owner_search_beam_update_ns{};
  u64 storage_owner_search_result_sort_ns{};
  u64 storage_owner_prune_snapshot_read_ns{};
  u64 storage_owner_prune_distance_ns{};
  u64 storage_owner_prune_sort_ns{};
  u64 storage_owner_prune_pair_distance_ns{};

  // Preserve the schema-15 response size and all following field offsets.
  // These four words used to carry anchor-hint telemetry and are now ignored.
  u64 reserved_schema15[4]{};

  u64 total() const {
    return storage_owner_queue_wait_ns +
           storage_owner_medoid_ns +
           ... +
           storage_owner_response_build_ns;
  }
};

static_assert(sizeof(InsertBreakdownCounters) == 224);
static_assert(offsetof(InsertBreakdownCounters, reserved_schema15) == 192);
```

分组：

- 第 1 组（前 14 个）：粗粒度阶段——queue wait、medoid 选择、search、prune、write_node、local/remote reverse、peer reverse apply、response send、prepare/allocate/publish/schedule_maintenance/response_build。
- 第 2 组（10 个）：search 和 prune 阶段的细粒度子项——select、neighbor_read、snapshot_read、distance、beam_update、result_sort、prune_pair_distance 等。
- `reserved_schema15[4]`：又是 schema-15 兼容——4 个 u64（32 字节）原本是 anchor-hint 遥测，现已废弃但保留字节位置。

`total()` 只累加粗粒度阶段，不累加细粒度子项——避免双重计数。这和 `storage_owner_client_helpers.hh` 里的 `add_storage_owner_breakdown` 用减法避免双重计数的逻辑对应（见 8.5.2）。

`static_assert(sizeof(InsertBreakdownCounters) == 224)`：224 = 14 × 8 + 10 × 8 + 4 × 8 = (14+10+4) × 8。这个尺寸必须和存储节点的写入侧一致，否则读越界。`offsetof(reserved_schema15) == 192` = (14+10) × 8，保证 reserved 区域在末尾。

### 8.4.5 peer RPC header 与 op 结构

`src/service/storage_owner_protocol.hh:145-171`：

```cpp
struct PeerRpcHeader {
  u32 magic{kPeerRpcMagic};
  u32 version{kPeerRpcVersion};
  u32 type{};
  u32 source_shard{};
  u32 item_count{};
  u64 request_id{};
  u32 status{static_cast<u32>(InsertStatus::failed)};
  u32 reserved{};
};

struct ReverseUpdateOp {
  u64 target_raw{};
  u64 candidate_raw{};
};

struct StitchSearchItem {
  u64 target_raw{};
  u32 id{};
  u32 generation{};
};

struct StitchSearchCandidate {
  u64 raw{};
  u32 generation{};
  u32 reserved{};
};
```

`PeerRpcHeader`（32 字节）是所有 peer RPC 的公共前缀：

- `magic` + `version`：双保险。
- `type`：`PeerRpcType` 枚举值，决定后续 payload 怎么解释。
- `source_shard`：发起方 shard 编号。
- `item_count`：本批次 op 数。
- `request_id`：配对 request/response。
- `status`：响应状态，默认 `failed`——这是个防御性默认，未填写的响应会被当作失败。

`ReverseUpdateOp`：反向边更新——"target 节点应该把 candidate 加进它的邻居表"。`target_raw`/`candidate_raw` 是远端指针，详见第 23–24 课。

`StitchSearchItem`：跨 shard 缝合搜索的请求项——"在 target_raw 这个入口附近，找 id 这个 logical ID 的真实节点"。`generation` 用来过滤过期版本。

`StitchSearchCandidate`：缝合搜索的响应候选项——一个候选邻居的远端指针 + generation。

### 8.4.6 对齐工具与请求/响应字节计算

`src/service/storage_owner_protocol.hh:173-184`：

```cpp
constexpr size_t align_wire_u64(size_t value) {
  return (value + alignof(u64) - 1) & ~(alignof(u64) - 1);
}

static_assert(align_wire_u64(1) == 8);
static_assert(align_wire_u64(8) == 8);

inline size_t insert_batch_request_bytes(u32 item_count) {
  return sizeof(InsertBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}
```

`align_wire_u64` 把任意字节数向上对齐到 8 字节——这是 RDMA/DMA 友好对齐。peer RPC 里有变长对齐需求（stitch search 的 vectors 区域必须 8 字节对齐起），所以需要它。两个 `static_assert` 验证：1 对齐到 8、8 对齐到 8。

`insert_batch_request_bytes`：insert 请求的总字节数 = header + item_count 个 `node_t`（ID 数组）+ item_count 个 vector（每个 `VamanaNode::vector_bytes()` 字节）。`VamanaNode::vector_bytes()` 是个 inline static（`src/vamana/vamana_node.hh:78-79`），在初始化时由 `Metadata.vector_bytes` 设置。

`src/service/storage_owner_protocol.hh:186-200`：

```cpp
inline size_t mutation_batch_request_bytes(u32 item_count) {
  return sizeof(MutationBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(u32) +
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}

inline size_t insert_batch_response_bytes(u32 item_count) {
  return sizeof(InsertBatchResponseHeader) +
         static_cast<size_t>(item_count) * sizeof(u32) +
         static_cast<size_t>(item_count) * sizeof(MutationResult) +
         sizeof(InsertBreakdownCounters) +
         sizeof(u32) +
         static_cast<size_t>(item_count) * VamanaNode::R * sizeof(u64);
}
```

`mutation_batch_request_bytes` 比 insert 版多了一段 `item_count * sizeof(u32)`——这是 `MutationKind` 数组，每个 item 一个 kind（insert/upsert/erase）。这是 mutation 比 insert 多携带的信息。

`insert_batch_response_bytes` 是响应总字节：

- `InsertBatchResponseHeader`：16 字节。
- `item_count * sizeof(u32)`：每个 item 的 `MutationStatus`。
- `item_count * sizeof(MutationResult)`：每个 item 的 `MutationResult`（new/old rptr、generation 等）。
- `sizeof(InsertBreakdownCounters)`：一整个批次的 breakdown counters（不是每 item 一份）。
- `sizeof(u32)`：invalidation count——本次 mutation 让多少个旧节点失效（用于 RCU/缓存失效）。
- `item_count * VamanaNode::R * sizeof(u64)`：invalidated raws——失效节点的远端指针数组，每 item 最多 `R` 个（因为一次 mutation 最多影响 `R` 个邻居）。

这个布局是**变长响应**：`invalidation_count` 字段告诉接收方后面实际有多少个 invalidated raws（上限是 `item_count * R`）。

### 8.4.7 字段访问器：把 `void*` 切片

`src/service/storage_owner_protocol.hh:202-292` 是一组 inline 访问器。它们都是 `reinterpret_cast` + 字节偏移，把 `void* payload` 切成 header / kinds / ids / vectors / statuses / results / breakdown / invalidations。看几个代表：

```cpp
inline node_t* request_ids(void* payload) {
  return reinterpret_cast<node_t*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline u32* mutation_request_kinds(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(MutationBatchRequestHeader));
}

inline node_t* mutation_request_ids(void* payload) {
  return reinterpret_cast<node_t*>(mutation_request_kinds(payload) +
                                   reinterpret_cast<MutationBatchRequestHeader*>(payload)->item_count);
}

inline byte_t* mutation_request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(mutation_request_ids(payload) + item_count);
}
```

`mutation_request_*` 系列展示了 mutation 请求 payload 的三段式：header → kinds 数组 → ids 数组 → vectors 数组。每个访问器都基于前一个的末尾地址 + item_count 计算，形成链式偏移。

响应侧（第 252–292 行）：

```cpp
inline u32* response_statuses(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline MutationResult* response_mutation_results(void* payload, u32 item_count) {
  return reinterpret_cast<MutationResult*>(response_statuses(payload) + item_count);
}

inline InsertBreakdownCounters* response_breakdown(void* payload, u32 item_count) {
  return reinterpret_cast<InsertBreakdownCounters*>(
    reinterpret_cast<byte_t*>(response_mutation_results(payload, item_count) + item_count));
}

inline u32* response_invalidation_count(void* payload, u32 item_count) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(response_breakdown(payload, item_count) + 1));
}

inline u64* response_invalidated_raws(void* payload, u32 item_count) {
  return reinterpret_cast<u64*>(response_invalidation_count(payload, item_count) + 1);
}

inline u32 response_invalidation_capacity(u32 item_count) {
  return item_count * VamanaNode::R;
}
```

响应 payload 布局：header → statuses → mutation_results → breakdown（1 个） → invalidation_count（1 个 u32） → invalidated_raws（最多 `item_count * R` 个 u64）。

`response_invalidation_capacity` 是上限——预分配响应缓冲时按这个上限来，实际有效数由 `response_invalidation_count` 给出。

### 8.4.8 peer RPC 字节计算与访问器

`src/service/storage_owner_protocol.hh:298-394` 是 peer RPC 的字节布局。peer RPC 因为有变长 vectors 和对齐需求，比 mutation 更复杂。看几个关键函数：

```cpp
inline size_t reverse_update_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) * sizeof(ReverseUpdateOp);
}

inline size_t reverse_update_response_bytes() {
  return sizeof(PeerRpcHeader);
}
```

reverse update 请求 = header + N 个 `ReverseUpdateOp`；响应只有 header（带 status）。这是最简单的 peer RPC。

```cpp
inline size_t stitch_search_vectors_offset(u32 item_count) {
  return align_wire_u64(sizeof(PeerRpcHeader) +
                        static_cast<size_t>(item_count) * sizeof(StitchSearchItem));
}

inline size_t stitch_search_request_bytes(u32 item_count) {
  return stitch_search_vectors_offset(item_count) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}
```

stitch search 请求要带 query vectors，所以 layout = header + N × StitchSearchItem + （对齐填充）+ N × vector_bytes。`align_wire_u64` 确保 vectors 起始地址 8 字节对齐——DMA 友好。

```cpp
inline size_t stitch_search_candidate_vectors_offset(u32 item_count,
                                                      u32 candidate_capacity) {
  return align_wire_u64(stitch_search_candidates_offset(item_count) +
                        static_cast<size_t>(item_count) * candidate_capacity *
                          sizeof(StitchSearchCandidate));
}

inline size_t stitch_search_response_bytes(u32 item_count,
                                           u32 candidate_capacity) {
  return stitch_search_candidate_vectors_offset(item_count, candidate_capacity) +
         static_cast<size_t>(item_count) * candidate_capacity *
           VamanaNode::vector_bytes();
}
```

stitch search 响应更复杂：每个 item 最多返回 `candidate_capacity` 个候选，每个候选带 `StitchSearchCandidate` 元数据 + 一个完整 vector。响应字节 = header + counts + candidates + candidate_vectors，层层对齐。

剩下的 `reverse_update_ops`、`stitch_search_items`、`stitch_search_vectors`、`stitch_search_response_counts`、`stitch_search_response_candidates`、`stitch_search_response_candidate_vectors` 都是对应的 `void*` 切片访问器，模式与 mutation 侧一致。peer RPC 的实际收发逻辑见第 24 课。

### 8.4.9 与第 26 课 wire_protocol 的关系

本文件的 `kPeerRpcMagic` / `PeerRpcHeader` 是 peer RPC 的**协议层**定义——消息长什么样、字段怎么排。而第 26 课的 `wire_protocol.cc`（`src/memory_node/storage_owner_runtime/wire_protocol.cc`）是**编解码层**实现：它包含如何 `recv` 一个 mutation 请求、如何 dispatch 到对应 handler、如何 `send` 响应。本文件定义"语法"，第 26 课定义"语法分析器+调度器"。

调用关系：`src/memory_node/storage_owner_runtime/wire_protocol.cc:55,123` 多次调用 `mutation_batch_request_bytes`；`src/memory_node/storage_owner_runtime/lifecycle.cc:13` 在初始化时用它来预分配缓冲区。计算侧 `src/service/compute_service/storage_owner/sender.cc:151` 同样用它来计算发送缓冲大小。

---

## 8.5 `storage_owner_client_helpers.hh`：客户端构造 mutation 请求的辅助

这个头文件是计算侧（client）的纯函数工具箱，全部 `inline`。它做两件事：(1) 时间统计的算术辅助；(2) 把存储节点回传的 `InsertBreakdownCounters` 折算成本地 `breakdown::Sample` 的子分类。

### 8.5.1 时间算术辅助

`src/service/storage_owner_client_helpers.hh:14-37`：

```cpp
inline u64 per_item_ns(u64 total, u32 item_count) {
  return item_count == 0 ? 0 : total / item_count;
}

inline u64 saturating_sub(u64 lhs, u64 rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

inline u64 duration_ns(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline u64 duration_ns_clamped(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
  if (end <= start) {
    return 0;
  }
  return duration_ns(start, end);
}

inline u64 storage_owner_wr_id(u32 owner_storage, u32 slot_id) {
  return encode_64bit(owner_storage, slot_id);
}
```

- `per_item_ns`：把总耗时摊到每 item。除零保护。
- `saturating_sub`：饱和减法，防止 `u64` 下溢成 `UINT64_MAX`。这在 breakdown 拆分时很关键——细粒度子项之和可能因测量噪声略大于粗粒度父项，直接减会爆。
- `duration_ns`：两个 `steady_clock` 时间点的纳秒差。
- `duration_ns_clamped`：clamp 版，`end <= start` 返回 0（避免负数 wrap）。
- `storage_owner_wr_id`：把 `(owner_storage, slot_id)` 编码进一个 u64，用作 work request ID。`encode_64bit(a, b) = (a << 32) | b`（`rdma-library/library/utils.hh:70`）。这样 RDMA completion 里取出的 `wr_id` 可以拆回 `(owner, slot)`，路由到对应的 callback。这是 RDMA 编程的常见技巧——`wr_id` 是 completion 与请求配对的唯一线索。

### 8.5.2 `add_storage_owner_breakdown`：把 counters 折算进 Sample

`src/service/storage_owner_client_helpers.hh:39-99`：

```cpp
inline void add_storage_owner_breakdown(
    service::breakdown::Sample* sample,
    const service::storage_owner::InsertBreakdownCounters& counters,
    u32 item_count) {
  if (!sample || !sample->collects_breakdown()) {
    return;
  }
  const u64 explained_search_ns =
    counters.storage_owner_search_select_ns +
    counters.storage_owner_search_distance_ns +
    counters.storage_owner_search_beam_update_ns +
    counters.storage_owner_search_result_sort_ns;
  const u64 explained_prune_ns =
    counters.storage_owner_prune_distance_ns +
    counters.storage_owner_prune_sort_ns +
    counters.storage_owner_prune_pair_distance_ns;
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_queue_wait,
                          per_item_ns(counters.storage_owner_queue_wait_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_medoid,
                          per_item_ns(counters.storage_owner_medoid_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search,
                          per_item_ns(saturating_sub(counters.storage_owner_search_ns, explained_search_ns),
                                      item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune,
                          per_item_ns(saturating_sub(counters.storage_owner_prune_ns, explained_prune_ns),
                                      item_count));
  ...
}
```

设计要点：

1. **空 sample 或不采集 breakdown 时直接返回**——避免没开 breakdown 时的开销。
2. **`explained_search_ns` / `explained_prune_ns`**：把细粒度子项之和算出来，然后从粗粒度父项里**减掉**，得到"未解释"部分。这正是 8.4.4 里 `total()` 不累加细粒度的原因——细粒度是粗粒度的"子集"，不能直接相加。
3. **`saturating_sub`**：防止测量噪声导致 explained > parent。
4. **`per_item_ns`**：所有值都摊到每 item，便于跨批次比较。

剩下的 `add_subcategory` 调用一一对应 `InsertBreakdownCounters` 的字段和 `Subcategory` 枚举值（见 `src/service/breakdown/names.hh:21-57`）。每个粗粒度阶段、每个细粒度子项都有自己的 Subcategory。计算侧的 `Sample` 累加这些值后，最终汇入 `breakdown::Report`，供 `/breakdown` 接口查询（见第 30 课 breakdown benchmark）。

### 8.5.3 `add_storage_owner_sender_breakdown`：发送侧耗时

`src/service/storage_owner_client_helpers.hh:101-119`：

```cpp
inline void add_storage_owner_sender_breakdown(
    service::breakdown::Sample* sample,
    u64 sender_queue_wait_ns,
    u64 request_prepare_ns,
    u64 send_ns,
    u64 response_wait_unaccounted_ns,
    u32 item_count) {
  if (!sample || !sample->collects_breakdown()) {
    return;
  }
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_sender_queue_wait,
                          sender_queue_wait_ns);
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_request_prepare,
                          per_item_ns(request_prepare_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_send,
                          per_item_ns(send_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_response_wait_unaccounted,
                          per_item_ns(response_wait_unaccounted_ns, item_count));
}
```

这一组是**计算侧自己测量**的耗时（不来自存储节点响应）：

- `sender_queue_wait_ns`：请求在计算侧发送队列里等待的时间。
- `request_prepare_ns`：构造请求字节的时间。
- `send_ns`：RDMA send 本身的时间。
- `response_wait_unaccounted_ns`：等待响应但未被存储节点 breakdown 解释的部分（用总等待时间减去存储节点报的阶段和）。

注意 `sender_queue_wait_ns` 没有 `per_item_ns`——因为它通常按批次而非按 item 统计。

这两个 helper 加在一起，把一次 mutation 的端到端时间拆成"计算侧发送 + 存储侧处理 + 计算侧等待"三大块，每块再细分成十几个子项。这是第 30 课 breakdown benchmark 的数据基础。

---

## 8.6 `query_result.hh`：查询结果结构

`src/service/query_result.hh` 全文：

```cpp
#pragma once

#include "common/types.hh"

namespace service {

struct QueryResultItem {
  node_t id{};
  distance_t distance{};
};

using QueryResult = vec<QueryResultItem>;

}  // namespace service
```

只有 15 行，但它是整个查询路径的"出口契约"。

`QueryResultItem` 是单个 top-k 结果：

- `id`：`node_t` = `u32`（`src/common/types.hh:9`），即 logical ID。客户端用这个 ID 去业务侧映射回原始数据。
- `distance`：`distance_t` = `f32`（`src/common/types.hh:11`），即查询向量与该 ID 对应向量的距离（通常是 L2 或内积，由查询参数决定）。

`QueryResult = vec<QueryResultItem>`：一个查询返回 top-k 个 item，按 distance 升序（最近的在前）。k 由查询参数指定，通常等于 GPU kernel 的 beam 宽度或最终 rerank 上限。

### 8.6.1 与第 9 课 CompletionDescriptor、第 14 课 completion 的关系

`QueryResult` 是**纯数据**，不携带任何执行状态。执行状态由第 9 课的 `CompletionDescriptor` 和第 14 课的 completion 流程管理：

- 查询从客户端发起 → 进入持久化引擎（第 11、17 课）→ kernel 启动（第 17 课）→ RDMA cache 取数（第 19 课）→ query traversal 主循环（第 20 课）→ 候选评分（第 18 课）→ completion（第 14 课）→ 产出 `QueryResult`。
- `CompletionDescriptor` 描述"这个查询在 device ring 上的哪个 slot、用哪个 beam buffer、最终结果写到哪段 GPU 内存"（见第 9 课）。
- completion 流程（第 14 课）把 GPU 上的 top-k 结果 copy 回 host、组装成 `QueryResult`、通过 callback 返回给调用方。

`QueryResult` 的极简设计是有意的：它不绑定任何执行细节，所以可以穿过 host/device 边界、穿过 RDMA、穿过协程（第 3 课），而无需为每种载体做适配。第 22 课 GPUNetIO 传输、第 27 课计算服务主体都会以 `QueryResult` 作为最终返回类型。

---

## 8.7 关键数据结构与流程图

### 8.7.1 数据结构总览

```
+-----------------------------------+   <-- service::index_metadata::Metadata
| schema_version, dim, R, ...       |      (src/service/index_metadata.hh:8-47)
| num_memory_nodes (= owner_count)  |
| hot_graph_offsets: vec<u64>       |   每 shard 一个远端偏移
| storage_control_remote_offsets    |
| navigation_code_remote_offsets    |
| idmap_format = "owner_sharded_v1" |   触发 BaseOwnerMap 加载
+-----------------------------------+
              |
              | load_metadata()
              v
+-----------------------------------+   <-- service::BaseOwnerMap
| pages_: vector<unique_ptr<Page>>  |      (src/service/base_owner_map.hh:16-37)
|   pages_[id>>16][id&0xFFFF] = u8  |   二级页表，1 byte/ID
|   kMissingOwner = 255             |
| entry_count_, allocated_pages_    |
+-----------------------------------+
              |
              | owner_for(id) -> optional<u32>
              v
        路由决策（compute_service/index_commands.cc:42-75）
        1. compute_side_idmap[id] (runtime)
        2. base_owner_map.owner_for(id) (immutable)
        3. id % num_servers_ (deterministic new ID)
              |
              v
+-----------------------------------+   <-- service::storage_owner::wire protocol
| MutationBatchRequestHeader        |      (src/service/storage_owner_protocol.hh:56-66)
|   magic, dim, owner_storage, ...  |
| payload layout:                   |
|   [header][kinds][ids][vectors]   |
+-----------------------------------+
              |
              | RDMA send
              v
        存储节点（第 23-26 课）
              |
              | RDMA recv
              v
+-----------------------------------+
| InsertBatchResponseHeader         |      (src/service/storage_owner_protocol.hh:77-83)
|   magic, owner_storage, batch_id  |
| payload layout:                   |
|   [header][statuses][results]     |
|   [breakdown][inv_count][inv_raws]|
+-----------------------------------+
              |
              | add_storage_owner_breakdown()
              v
+-----------------------------------+
| service::breakdown::Sample        |      (src/service/breakdown/sample.hh)
|   subcategory_ns[]                |   每个阶段/子项一个槽
+-----------------------------------+
```

### 8.7.2 交互图：从 metadata 加载到 mutation ack

```
计算节点启动                                存储节点
============                                ========

[load_metadata(prefix)]
  读 .meta.json
  -> Metadata{num_memory_nodes=N,
              idmap_format="owner_sharded_v1",
              hot_graph_offsets[s],
              storage_control_remote_offsets[s], ...}

[if enable_updates]
  [base_owner_map_.load(prefix, N, "owner_sharded_v1")]
    for owner in 0..N-1:
      读 <prefix>_node{owner+1}_of{N}.idmap
      校验 magic/version/owner_shard/shard_count
      校验 文件大小 == sizeof(Header)+entry_count*sizeof(Entry)
      分批读 entries，写入 pages_[id>>16][id&0xFFFF]
    print "base idmap: entries=.. memory=.."
    print "placement: base idmap for existing; deterministic for new"

[启动 persistent_search_, 连接存储节点]
  (用 Metadata 里的远端偏移表初始化 RDMA cache)

------------------ 运行时 ------------------

客户端调用 mutate(item)
  |
  v
known_storage_owner_for_id(item.id)
  1. compute_side_idmap 命中 -> 用其 owner
  2. base_owner_map.owner_for(id) 命中 -> 用 base owner
  3. 都未命中 -> proposed = id % N
     claim_storage_owner_for_mutation(id, proposed)
       在 compute_side_idmap 写 generation=0 占位
  |
  v  owner_storage = <决定出来的 owner>
  
[构造 MutationBatchRequest]
  mutation_batch_request_bytes(item_count)
  填 header: magic=kMutationMagic, dim, owner_storage, source_client,
            item_count, vector_dtype, vector_bytes, anchor_hint_count=0, batch_id
  填 kinds[item_count]: insert/upsert/erase
  填 ids[item_count]:   logical IDs
  填 vectors[item_count*vector_bytes]: 向量数据
  
[RDMA send]  ------------------------------>  [存储节点 recv]
                                              dispatch by magic
                                              执行 mutation（第 23-26 课）
                                              构造响应：
                                                [InsertBatchResponseHeader]
                                                [statuses[item_count]]
                                                [MutationResult[item_count]]
                                                  new_rptr_raw, old_rptr_raw,
                                                  generation, maintenance_seq
                                                [InsertBreakdownCounters]
                                                [invalidation_count]
                                                [invalidated_raws[<= item_count*R]]
[RDMA recv]  <-------------------------------  [RDMA send]

[解析响应]
  response_statuses(payload) -> MutationStatus[item_count]
  response_mutation_results(payload, item_count) -> MutationResult[item_count]
  response_breakdown(payload, item_count) -> InsertBreakdownCounters*
  response_invalidation_count -> u32
  response_invalidated_raws -> u64[]

[add_storage_owner_breakdown(sample, counters, item_count)]
  把存储侧细分耗时折算进本地 Sample（每 item 纳秒）

[add_storage_owner_sender_breakdown(sample, ...)]
  把本地发送侧耗时折算进 Sample

[更新 compute_side_idmap[id] = {new_rptr, deleted=false, owner, generation}]
  generation 从 0 升到存储节点给的值
  rptr 从空变成 new_rptr_raw

[处理 invalidated_raws]
  这些是因本次 mutation 而失效的旧 rptr
  -> 通知 RDMA cache 失效（第 19 课）
  -> 进入 RCU 回收（第 16 课）

[breakdown 上报] -> 第 30 课
```

### 8.7.3 关键不变量

1. **owner 一致性**：同一 logical ID 在所有计算节点上路由到同一 owner。来源：base idmap 不可变 + 新 ID 用 `id % N` 确定性分片。
2. **wire 布局不可变**：`static_assert` 锁死 header 尺寸和关键字段偏移。schema-15 字段（anchor_hint_count、reserved_schema15）保留字节但废弃语义。
3. **事务性加载**：`BaseOwnerMap::load` 失败时 `*this` 保持原状。
4. **breakdown 不双重计数**：细粒度子项从粗粒度父项里 `saturating_sub` 出来，`total()` 只累加粗粒度。
5. **generation 单调**：compute_side_idmap 用 `existing.generation >= generation` 拒绝旧响应（`index_commands.cc:22-25`）。

---

## 8.8 与其他模块的关系

- **第 2 课（公共类型与配置）**：`Metadata` 用到的 `u32/u64/str/vec/filepath_t`、`node_t/element_t/distance_t`、`VectorDType` 都来自 `common/types.hh` 和 `common/vector_dtype.hh`。
- **第 4–5 课（RDMA 传输库）**：`storage_owner_wr_id` 用 `encode_64bit` 编码 `(owner, slot)`，是 RDMA work request ID 的标准玩法。mutation 请求/响应通过 RDMA send/recv 传输。
- **第 6 课（Vamana 图格式与 anchor/idmap）**：`BaseOwnerMap` 直接消费 `vamana::idmap::Header`/`Entry`，这是 owner-sharded idmap 的 on-disk 格式。`anchor_count_per_shard`、`anchor_format` 在 `Metadata` 里。
- **第 7 课（schema-15 索引格式）**：`Metadata` 是 schema-15 索引的目录页。`schema_version`、`storage_format`、`node_layout`、所有 `hot_graph_*`/`dynamic_*` 字段都对应 schema-15 的 on-disk 布局。本课的 `static_assert` 是 schema-15 wire 兼容性的编译期护栏。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`Metadata` 的 PQ 字段（`pq_subquantizers`/`pq_bits`/`navigation_*`）喂给 GPU 持久化 kernel。`InsertBreakdownCounters` 是遥测数据结构。
- **第 10 课（delta/动态路由/预算）**：`MutationResult.generation` 是动态路由的基础——每次 mutation 递增 generation，旧 generation 的节点被失效。`maintenance_sequence` 触发维护任务。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`PersistentSearchEngine` 在构造时接收 `Metadata`，据此初始化 RDMA cache 的远端偏移表。
- **第 14 课（查询执行/路由/完成）**：`QueryResult` 是 completion 流程的产物。completion 把 GPU 上的 top-k 组装成 `QueryResult`。
- **第 16 课（存储回收 RCU）**：`MutationResult.old_rptr_raw` 和响应里的 `invalidated_raws` 进入 RCU 回收路径。
- **第 17 课（kernel 启动器/上下文/device ring）**：kernel 启动器从 `Metadata` 拿 `dim`/`R`/PQ 参数配置 device ring。
- **第 19 课（RDMA cache）**：`Metadata.storage_control_remote_offsets[s]` 是 RDMA cache 读 control page 的偏移来源。`invalidated_raws` 触发 cache 失效。
- **第 20 课（查询遍历主循环）**：遍历用 `Metadata.hot_graph_offsets[s]` 等表发起 RDMA 读。
- **第 23 课（存储节点主体/peer RDMA）**：存储节点用本课的 wire protocol 解析收到的 mutation 请求，调用图修改 API（第 25 课）执行实际 mutation，再按本课的响应格式回传。
- **第 24 课（peer RPC）**：本课定义的 `PeerRpcHeader`/`ReverseUpdateOp`/`StitchSearchItem`/`StitchSearchCandidate` 是 peer RPC 的协议层。
- **第 26 课（维护/wire protocol）**：本课的 `MutationBatchRequestHeader`/`InsertBatchResponseHeader` 在第 26 课的 `wire_protocol.cc` 里被解析和构造。`maintenance_sequence` 触发第 26 课的维护任务。
- **第 27 课（计算服务主体）**：`ComputeService` 构造函数（`lifecycle.cc`）是本课 metadata + base_owner_map 加载的调用方。
- **第 28 课（计算侧 storage owner 更新）**：`compute_side_idmap`、`claim_storage_owner_for_mutation`、`known_storage_owner_for_id` 是第 28 课的核心数据结构，本课的 `BaseOwnerMap` 是它们的不可变基线。
- **第 29 课（离线构建/迁移）**：离线构建产出 `.meta.json` 和 `<prefix>_node{i}_of{N}.idmap`，必须满足 `idmap_format="owner_sharded_v1"` 才能在运行时开 updates。
- **第 30 课（breakdown benchmark）**：`add_storage_owner_breakdown`/`add_storage_owner_sender_breakdown` 把 `InsertBreakdownCounters` 折算进 `Sample`，是 breakdown 报告的数据来源。

---

## 8.9 小结

本课覆盖了存算分离契约的"元数据 + 路由 + wire 协议 + 结果"四件套：

1. **`index_metadata`**：把 `.meta.json` 翻译成内存中的 `Metadata` 结构，其中远端偏移表（`hot_graph_offsets`/`storage_control_remote_offsets`/`navigation_code_remote_offsets`）是计算节点发起 RDMA 读的依据；`idmap_format` 字段是触发 owner map 加载的开关。

2. **`BaseOwnerMap`**：用二级页表（64 KiB 一页、1 byte/ID）把 owner-sharded idmap 全量加载进内存，提供 `O(1)` 的 `owner_for(id)` 查询。加载是事务性的（失败不留半成品），并做四重 header 校验和文件大小公式校验。它与 `compute_side_idmap` + `id % N` 一起构成三段式 owner 决策，保证所有计算节点对同一 ID 路由一致。

3. **`storage_owner_protocol`**：定义计算↔存储的 mutation wire format。核心是 `MutationBatchRequestHeader`（40 字节，schema-15 兼容）和变长响应（header + statuses + results + breakdown + invalidations）。所有字段偏移用 `static_assert` 锁死，废弃字段保留字节。peer RPC（`PeerRpcHeader` + 三对 request/response）是第 24 课的协议层。

4. **`storage_owner_client_helpers`**：客户端工具箱——时间算术（`per_item_ns`/`saturating_sub`/`duration_ns`/`encode_64bit`）和 breakdown 折算（`add_storage_owner_breakdown`/`add_storage_owner_sender_breakdown`）。关键技巧是用 `saturating_sub` 把细粒度子项从粗粒度父项里减出来，避免双重计数。

5. **`query_result`**：极简的 `QueryResult = vec<{id, distance}>`，是整个查询路径的出口契约，与第 9 课 `CompletionDescriptor`（执行状态）和第 14 课 completion（组装流程）配合使用。

下一课（第 9 课）将进入 GPU 侧，讲解 GPU 类型系统、遥测结构和 PQ 模型——其中 PQ 模型的元数据正是本课 `Metadata` 里 `pq_subquantizers`/`pq_bits`/`navigation_*` 字段的消费者。
