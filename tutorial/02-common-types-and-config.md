# 第 2 课 公共类型与配置

## 2.1 本课目标与涉及文件

第 1 课我们俯瞰了 dvstor 的整体目录结构和构建系统，知道这是一个 GPU 中心化的存算分离向量检索系统：计算节点上跑常驻 CUDA Kernel，通过 GPUNetIO 直接 RDMA 读存储节点上的图分片与 PQ 编码；存储节点上跑 RDMA 服务端 + 在线图维护。

本课聚焦于 `src/common/` 下的六个头文件——它们是整个项目"说话的方式"：所有模块共享的类型别名、编译期常量、向量 dtype 解析、距离函数、CLI 配置对象以及索引文件路径拼装。这些头文件几乎被项目中每个 `.cc`/`.cu`/`.hh`/`.cuh` 直接或间接 include，理解它们是阅读后续每一课的前置条件。

涉及文件（均为绝对路径）：

| 文件 | 行数 | 作用 |
|------|------|------|
| `/home/xjs/experiment/dvstor/src/common/types.hh` | 23 | 项目级类型别名（`node_t`/`element_t`/`distance_t`/容器别名） |
| `/home/xjs/experiment/dvstor/src/common/constants.hh` | 9 | 编译期常量（缓存行、最大 peer QP 数） |
| `/home/xjs/experiment/dvstor/src/common/vector_dtype.hh` | 499 | `VectorDType` 枚举、解析/编解码、AVX2 加速 L2 距离 |
| `/home/xjs/experiment/dvstor/src/common/distance.hh` | 89 | 旧的 float32 专用 L2 距离（hnswlib 风格 AVX 路径） |
| `/home/xjs/experiment/dvstor/src/common/configuration.hh` | 451 | `IndexConfiguration`：所有 CLI 选项、校验、打印 |
| `/home/xjs/experiment/dvstor/src/common/index_path.hh` | 61 | 索引文件路径拼接辅助函数 |
| `/home/xjs/experiment/dvstor/src/common/timing.hh`（辅助） | 53 | `Timing`/`Interval` 计时器 |

本课在最后会用一张流程图展示"配置如何流向计算节点与存储节点"，并标注每个配置项在哪一课被消费。

> **前置说明**：`src/common/types.hh` 第 5 行 `#include <library/types.hh>` 指向的是 `rdma-library/library/types.hh`（见第 4 课 RDMA 传输库上）。该文件定义了 `u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/byte_t/str/vec/span/s_ptr/func/concurrent_vec/concurrent_queue` 等全套别名。本课在用到这些别名时会显式说明其底层类型，但不对该文件本身逐行讲解——那是第 4 课的内容。`configuration.hh` 第 9 行 `#include <library/configuration.hh>` 同理指向 `rdma-library/library/configuration.hh` 中的基类 `Configuration`，本课会在 2.6 节给出该基类的关键成员，完整讲解留到第 4 课。

---

## 2.2 `types.hh`：项目级类型别名

完整文件只有 23 行，但定义了项目内最常用的几个语义别名。

```cpp
// /home/xjs/experiment/dvstor/src/common/types.hh
1   #pragma once
2
3   #include <filesystem>
4   #include <unordered_map>
5   #include <library/types.hh>
6
7   #include "ankerl/unordered_dense.h"
8
9   using node_t = u32;
10  using element_t = f32;
11  using distance_t = f32;
12
13  using filepath_t = std::filesystem::path;
14
15  template <typename T>
16  using hashset_t = ankerl::unordered_dense::set<T>;
17
18  template <typename K, typename V>
19  using hashmap_t = std::unordered_map<K, V>;
20
21  template <typename K, typename V>
22  using dense_hashmap_t = ankerl::unordered_dense::map<K, V>;
```

逐行讲解：

- **第 1 行 `#pragma once`**：标准的 include guard，比传统的 `#ifndef ... #define ... #endif` 更简洁，被主流编译器支持。
- **第 3–4 行**：引入 `std::filesystem::path` 和 `std::unordered_map`。
- **第 5 行**：`<library/types.hh>` 带来 `u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/byte_t/str/vec/span/s_ptr/func/concurrent_vec/concurrent_queue` 等基础别名（详见第 4 课）。这是整个项目的"类型地基"。
- **第 7 行**：引入 `ankerl::unordered_dense`——一个比 `std::unordered_map` 快得多（常常 2–3 倍）的哈希容器库，基于"_metadata 平铺 + 开放寻址"实现。项目里高频路径用它。
- **第 9 行 `using node_t = u32;`**：图节点 ID。注意是 `u32` 而非 `u64`——这意味着 dvstor 的 Vamana 图最多支持 2³² ≈ 42 亿个节点。这与 `configuration.hh` 第 31 行 `u32 max_vectors{1'000'000};` 的类型选择一致：`max_vectors` 表示"逻辑向量 ID 范围上限"，也是 `u32`。后续在 GPU kernel 中图遍历、`anchor/idmap`（见第 6 课）等都以 `node_t` 为节点标识。
- **第 10 行 `using element_t = f32;`**：向量分量类型。**注意这是"查询侧"和"距离计算浮点侧"的类型**，而**不是**存储侧的类型——存储侧可以是 `uint8`/`int8`/`float32` 三选一（见 2.4 节 `VectorDType`）。`element_t` 始终是 `f32`，因为查询向量在 GPU/CPU 上都先被解码成 float 再做距离计算。
- **第 11 行 `using distance_t = f32;`**：距离值类型。L2 距离的累加结果一律用 `f32`（即使在 `vector_dtype.hh` 的 AVX2 整型路径里，最终也是把整型累加器 cast 成 `float`，见 2.4.6 节）。
- **第 13 行 `using filepath_t = std::filesystem::path;`**：文件路径类型。整个项目用 `filepath_t` 而非 `std::string` 表示路径，好处是天然支持路径拼接（`/` 运算符）和扩展名操作（`has_extension()`/`replace_extension()`），`index_path.hh` 大量依赖这些。
- **第 15–16 行 `hashset_t`**：基于 `ankerl::unordered_dense::set` 的哈希集合别名。例如图遍历的 visited 集合可以用它。
- **第 18–19 行 `hashmap_t`**：标准 `std::unordered_map` 别名。注意它**没有**用 ankerl——可能是历史原因或某些场景下需要 `std::unordered_map` 的迭代器稳定性。新代码建议优先用 `dense_hashmap_t`。
- **第 21–22 行 `dense_hashmap_t`**：基于 `ankerl::unordered_dense::map` 的哈希映射别名，是性能敏感场景的首选。

> **设计观察**：`types.hh` 故意保持极薄——它只定义 dvstor 特有的语义别名（`node_t`/`element_t`/`distance_t`/`filepath_t`）和容器别名，**不**重新定义基础整数类型，而是直接复用 `library/types.hh`。这种"分层别名"避免了重复定义，也让 dvstor 的代码在视觉上与 rdma-library 的代码风格一致。

---

## 2.3 `constants.hh`：编译期常量

完整文件只有 9 行，但这两个常量都是性能敏感的"硬约束"。

```cpp
// /home/xjs/experiment/dvstor/src/common/constants.hh
1   #pragma once
2
3   #include <cstddef>
4
5   #include <library/types.hh>
6
7   inline constexpr size_t kCacheLineBytes = 64;
8   inline constexpr u32 kMaxPeerQps = 4;
```

逐行讲解：

- **第 7 行 `kCacheLineBytes = 64`**：CPU 缓存行大小，固定为 64 字节。这是 x86/ARM 主流平台的 L1/L2 缓存行宽度。项目里所有"避免 false sharing"的对齐分配都引用它——例如存储节点的 per-QP 状态、worker 协程上下文等会按 `kCacheLineBytes` 对齐，确保两个线程写的字段不在同一缓存行上。`inline constexpr` 让它在每个翻译单元都是同一份定义（C++17 起 `inline` 函数模板/变量可以在头文件中定义而不会违反 ODR）。
- **第 8 行 `kMaxPeerQps = 4`**：每个 peer（对等存储节点）允许的最大 RDMA QP 数。注意这是"peer 维度"的上限，与 `configuration.hh` 第 57 行 `u32 gpu_rdma_qps{4};`（"GPU 发起的 GPUNetIO QPs per storage node"）以及校验规则 `gpu_rdma_qps <= 32`（第 315 行）形成层级关系：单个 GPU 客户端最多 32 个 QP 去访问一个存储节点，但 peer 之间（存储节点互访，见第 23 课）最多 4 个 QP。这个常量被存储节点的 peer RDMA 子系统消费（见第 23 课"存储节点主体/peer RDMA"）。

> **为什么常量这么少？** dvstor 把绝大部分可调参数放在 `IndexConfiguration` 里（运行时 CLI 可改），只把"绝对不该改的硬约束"放在 `constants.hh`。`kCacheLineBytes` 改了会导致 false sharing 回归；`kMaxPeerQps` 改了会与存储节点 peer 侧的 QP 池布局耦合。这两个值都是"改了就要同步改代码"的，所以锁死在编译期。

---

## 2.4 `vector_dtype.hh`：向量 dtype 解析与编解码

这是本课最厚的文件，499 行，承担三个职责：

1. 定义 `VectorDType` 枚举与字符串互转；
2. 提供 dtype 自动推断（从文件扩展名）与显式配置的解析；
3. 提供存储 dtype ↔ float 的编解码，以及 AVX2 加速的 L2 距离函数。

### 2.4.1 枚举与命名

```cpp
// /home/xjs/experiment/dvstor/src/common/vector_dtype.hh
1   #pragma once
2
3   #include <algorithm>
4   #include <cmath>
5   #include <cstring>
6   #include <optional>
7   #include <stdexcept>
8
9   #ifdef __AVX2__
10  #include <immintrin.h>
11  #endif
12
13  #include "common/types.hh"
14
15  enum class VectorDType : u32 {
16    float32 = 0,
17    uint8 = 1,
18    int8 = 2,
19  };
```

- **第 9–11 行**：仅当编译器定义了 `__AVX2__` 时才引入 `<immintrin.h>`（AVX2/AVX intrinsics 总头）。这让文件在非 AVX2 平台上也能编译——所有 SIMD 代码都被 `#ifdef __AVX2__` 包裹，缺失时退化为标量路径。
- **第 15–19 行 `enum class VectorDType : u32`**：强类型枚举，底层类型 `u32`。三个值：
  - `float32 = 0`：每个分量 4 字节 IEEE-754 单精度浮点；
  - `uint8 = 1`：每个分量 1 字节无符号整数（范围 0–255），常用于 SIFT/Deep 等数据集；
  - `int8 = 2`：每个分量 1 字节有符号整数（范围 -128–127），常用于量化后的向量。
  
  数值从 0 开始递增是为了让枚举可以方便地用作数组索引或序列化标签。

```cpp
21  inline str vector_dtype_name(VectorDType dtype) {
22    switch (dtype) {
23      case VectorDType::float32:
24        return "float32";
25      case VectorDType::uint8:
26        return "uint8";
27      case VectorDType::int8:
28        return "int8";
29    }
30    return "unknown";
31  }
```

`vector_dtype_name` 是反向映射，用于日志/序列化。注意第 30 行的 `return "unknown"` 是为了消除"控制流到达函数末尾"的编译警告——三个 case 已覆盖所有合法枚举值，但编译器不知道。

### 2.4.2 字符串解析 `parse_vector_dtype`

```cpp
33  inline VectorDType parse_vector_dtype(const str& name) {
34    if (name == "float32" || name == "float" || name == "f32") {
35      return VectorDType::float32;
36    }
37    if (name == "uint8" || name == "u8") {
38      return VectorDType::uint8;
39    }
40    if (name == "int8" || name == "i8") {
41      return VectorDType::int8;
42    }
43    throw std::invalid_argument("unknown vector dtype: " + name);
44  }
```

接受三种风格的名字：

- `float32` / `float` / `f32` → `VectorDType::float32`
- `uint8` / `u8` → `VectorDType::uint8`
- `int8` / `i8` → `VectorDType::int8`

其他字符串抛 `std::invalid_argument`。这个宽容的解析是为了兼容 CLI 用户的不同习惯（NumPy 风格 `f32`、C++ 风格 `float`、正式名 `float32` 都行）。它被 `configuration.hh` 第 294 行的校验和第 102 行的 `resolved_vector_dtype()` 调用。

### 2.4.3 从文件路径推断 `infer_vector_dtype_from_path`

```cpp
46  inline std::optional<VectorDType> infer_vector_dtype_from_path(const filepath_t& path) {
47    const str ext = path.extension().string();
48    if (ext == ".u8bin") {
49      return VectorDType::uint8;
50    }
51    if (ext == ".i8bin") {
52      return VectorDType::int8;
53    }
54    if (ext == ".fbin" || ext == ".bin") {
55      return VectorDType::float32;
56    }
57    return std::nullopt;
58  }
```

这是 **auto 解析的核心**。它遵循一个事实上的"ANN 数据集文件扩展名约定"（来自 Microsoft SPTAG、hnswlib、NMSLIB 等项目）：

- `.u8bin` → uint8
- `.i8bin` → int8
- `.fbin` 或 `.bin` → float32

注意第 54 行 `.bin` 也被当作 float32——这是为了兼容一些不带 dtype 后缀的老数据集。如果扩展名都不匹配，返回 `std::nullopt`，调用方可以 fallback 到默认值。

### 2.4.4 配置解析 `resolve_vector_dtype_config`

```cpp
60  inline VectorDType resolve_vector_dtype_config(const str& value, const filepath_t& path) {
61    if (value.empty() || value == "auto") {
62      return infer_vector_dtype_from_path(path).value_or(VectorDType::float32);
63    }
64    return parse_vector_dtype(value);
65  }
```

这是配置解析的统一入口：

- 如果 `value` 是空字符串或 `"auto"`，则从 `path` 推断；推断失败时 fallback 到 `float32`（第 62 行 `.value_or(VectorDType::float32)`）。
- 否则用 `parse_vector_dtype` 显式解析（会校验字符串合法性，非法则抛异常）。

> **注意与 `IndexConfiguration::resolved_vector_dtype()` 的差异**：`configuration.hh` 第 100–103 行的 `resolved_vector_dtype()` 在 `"auto"` 时**直接返回 `float32`**，并不查文件扩展名——因为配置对象构造时 `index_prefix` 可能只是一个目录前缀而非数据文件路径。而 `resolve_vector_dtype_config` 是更完整的版本，会在确有数据文件路径的场景（如离线构建、见第 12/13/29 课）使用。

### 2.4.5 字节数计算

```cpp
67  inline size_t vector_dtype_component_size(VectorDType dtype) {
68    switch (dtype) {
69      case VectorDType::float32:
70        return sizeof(float);
71      case VectorDType::uint8:
72        return sizeof(u8);
73      case VectorDType::int8:
74        return sizeof(i8);
75    }
76    return sizeof(float);
77  }
78
79  inline size_t vector_dtype_bytes(VectorDType dtype, u32 dim) {
80    return static_cast<size_t>(dim) * vector_dtype_component_size(dtype);
81  }
```

- `vector_dtype_component_size`：返回单个分量的字节数（4/1/1）。第 76 行的 fallback 同样是为了消除编译警告。
- `vector_dtype_bytes`：返回一条 `dim` 维向量的总字节数。注意第 80 行 `static_cast<size_t>(dim)`——先转 `size_t` 再乘，避免 `u32 * size_t` 在某些平台上的有符号提升问题。这个函数在计算 RDMA 读大小、GPU 缓存预算、PQ 编码大小等场景被反复调用。

### 2.4.6 单分量取值 `vector_component_as_float`

```cpp
83  inline float vector_component_as_float(const byte_t* data, VectorDType dtype, size_t index) {
84    switch (dtype) {
85      case VectorDType::float32:
86        return reinterpret_cast<const float*>(data)[index];
87      case VectorDType::uint8:
88        return static_cast<float>(reinterpret_cast<const u8*>(data)[index]);
89      case VectorDType::int8:
90        return static_cast<float>(reinterpret_cast<const i8*>(data)[index]);
91    }
92    return 0.0f;
93  }
```

把存储里的第 `index` 个分量读出来转成 `float`。这是"统一到 float 计算"的基础原语——标量 L2 距离路径（第 471–477 行）就直接调用它。

注意三种 dtype 的读取都用 `reinterpret_cast` 把 `byte_t*` 转成对应类型指针再索引——这要求 `data` 的对齐满足目标类型（float 需要 4 字节对齐）。在 dvstor 里，RDMA 接收缓冲区和 GPU 缓存分配时都会保证至少 4 字节对齐，所以这里是安全的。

### 2.4.7 编码 `encode_float_vector_to_storage`

```cpp
95  inline void encode_float_vector_to_storage(const float* src, u32 dim, VectorDType dtype, byte_t* dst) {
96    switch (dtype) {
97      case VectorDType::float32:
98        std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
99        return;
100     case VectorDType::uint8: {
101       auto* out = reinterpret_cast<u8*>(dst);
102       for (u32 i = 0; i < dim; ++i) {
103         const long rounded = std::lround(src[i]);
104         out[i] = static_cast<u8>(std::clamp<long>(rounded, 0, 255));
105       }
106       return;
107     }
108     case VectorDType::int8: {
109       auto* out = reinterpret_cast<i8*>(dst);
110       for (u32 i = 0; i < dim; ++i) {
111         const long rounded = std::lround(src[i]);
112         out[i] = static_cast<u8>(std::clamp<long>(rounded, -128, 127));
113       }
114       return;
115     }
116   }
117 }
118
119 inline vec<byte_t> encode_float_vector_to_storage(const span<const element_t> src, VectorDType dtype) {
120   vec<byte_t> out(vector_dtype_bytes(dtype, static_cast<u32>(src.size())));
121   encode_float_vector_to_storage(src.data(), static_cast<u32>(src.size()), dtype, out.data());
122   return out;
123 }
```

把 `float` 向量编码到存储 dtype：

- **float32**：直接 `memcpy`，因为源和目标都是 IEEE-754 float。
- **uint8**：用 `std::lround` 四舍五入到 `long`，再用 `std::clamp<long>(rounded, 0, 255)` 钳位到 uint8 范围，最后 cast 成 `u8`。注意这里先转 `long` 再钳位——避免直接 cast 到 `u8` 时负数变成 255、超 255 的数回绕。
- **int8**：同理，钳位到 `[-128, 127]`。

> **第 112 行的细节**：`out[i] = static_cast<u8>(...)` —— 等等，这里声明的是 `i8* out`，但赋值却 cast 成 `u8`？这其实是一个 **隐式依赖两阶段转换**：`static_cast<u8>(clamp<long>(...))` 先把 clamp 后的 long 转成 `u8`（值在 [-128,127] 内时位模式与 i8 相同），再赋给 `i8` 数组元素。C++ 标准下 `u8` → `i8` 是 implementation-defined（当 u8 值 ≤ 127 时结果定义良好）。这在二进制层面正确，但严格来说写成 `static_cast<i8>(...)` 更清晰。这是一个可以改进的小点，但不影响功能。

第 119–123 行是 span 版本的重载，方便调用方直接传 `span<const element_t>` 而不用手动算 dim 和分配 buffer。

### 2.4.8 解码 `decode_storage_vector_to_float`

```cpp
125 inline void decode_storage_vector_to_float(const byte_t* src, VectorDType dtype, u32 dim, float* dst) {
126   for (u32 i = 0; i < dim; ++i) {
127     dst[i] = vector_component_as_float(src, dtype, i);
128   }
129 }
130
131 inline vec<float> decode_storage_vector_to_float(const byte_t* src, VectorDType dtype, u32 dim) {
132   vec<float> out(dim);
133   decode_storage_vector_to_float(src, dtype, dim, out.data());
134   return out;
135 }
```

逐分量调用 `vector_component_as_float` 把存储向量解码成 float 向量。两个重载：一个写入调用方提供的 `dst`，一个返回新分配的 `vec<float>`。注意这两个函数**没有 SIMD 加速**——它们用于离线构建/调试路径，热路径用的是 2.4.10 节的 `typed_l2_distance_float_query`（直接在距离计算里做转换，不单独解码）。

### 2.4.9 byte L2 精确性断言 `integral_byte_l2_sum_exact_in_float`

```cpp
137 // For byte vectors, every squared component difference is at most 255^2.
138 // Integer sums through this dimension remain exactly representable in IEEE-754
139 // float, so integer and decoded-float L2 paths cannot differ by reduction
140 // rounding. Wider dimensions must keep the established reduction order.
141 inline constexpr bool integral_byte_l2_sum_exact_in_float(u32 dim) {
142   return static_cast<u64>(dim) * 255ull * 255ull <= (1ull << 24) - 1;
143 }
```

这是一个**数值正确性断言函数**，理解它需要一点 IEEE-754 知识：

- `f32`（IEEE-754 binary32）的尾数是 23 位显式 + 1 位隐式 = 24 位精度，能精确表示的整数范围是 `[0, 2²⁴ - 1] = [0, 16777215]`。
- byte 向量每对分量的平方差最大是 `255² = 65025`。
- 如果 `dim` 个这样的平方差之和不超过 `2²⁴ - 1`，那么整个累加过程在 `f32` 里**精确可表示**，不会因为浮点累加顺序不同而产生差异。

这个函数返回 `true` 当且仅当 `dim * 255² ≤ 2²⁴ - 1`，即 `dim ≤ 258`。也就是说：**当 byte 向量维度 ≤ 258 时，无论用整型 SIMD 累加还是先解码成 float 再用 float 累加，L2 距离结果在位级别完全相同**。

这个保证的意义在于：存储侧（用 `typed_l2_distance` 的 uint8/int8 SIMD 路径，整型累加）和查询侧（用 `typed_l2_distance_float_query` 的 float 累加）对同一条 byte 向量算出的距离可以无歧义比较。如果维度更大，注释说"必须保持既定的归约顺序"——即两端必须用相同的累加顺序，否则会有 ULP 级别的差异。

> 这个函数在本课的代码里其实**没有被调用**——它是为后续 kernel 实现者提供的"安全边界"判断工具。在 GPU kernel（第 18–20 课）里会用到类似的思想。

### 2.4.10 AVX2 加速 L2 距离

这是文件最长也最硬核的部分。整体策略是：

- 同 dtype 对（uint8↔uint8、int8↔int8）走"整型差 → 整型平方和 → 转 float"路径，用 4 路并行累加；
- float 查询 vs byte 存储走"byte 解码成 float → float 差平方和"路径，用 4 路 float 累加；
- 两者都有 32 元素展开主循环 + 尾部标量处理。

#### 2.4.10.1 `typed_l2_distance_uint8_simd`（第 151–251 行）

```cpp
151 inline float typed_l2_distance_uint8_simd(const byte_t* lhs, const byte_t* rhs, u32 dim) {
152   const u8* a = reinterpret_cast<const u8*>(lhs);
153   const u8* b = reinterpret_cast<const u8*>(rhs);
154
155   __m256i sum0 = _mm256_setzero_si256();
156   __m256i sum1 = _mm256_setzero_si256();
157   __m256i sum2 = _mm256_setzero_si256();
158   __m256i sum3 = _mm256_setzero_si256();
159
160   u32 i = 0;
161
162   for (; i + 128 <= dim; i += 128) {
163     // Unrolled x4: each iteration directly targets its own accumulator
164     // to avoid runtime branch on k inside an already hot loop.
```

- 第 152–153 行：把 `byte_t*` 当成 `u8*` 处理。
- 第 155–158 行：**4 路独立累加器** `sum0/sum1/sum2/sum3`，每个是 `__m256i`（256 位，含 8 个 int32）。4 路独立的目的是打破数据依赖，让 CPU 乱序执行引擎能并行处理 4 条独立链路——这是 SIMD 累加的常用吞吐优化。
- 第 162 行：主循环每步处理 **128 个 uint8 分量**（4 路 × 32 字节/路）。注释（163–164 行）解释了为什么不写内层 `for k=0..3` 循环：避免热循环里的分支。

接下来看一路（k=0）的处理（第 167–176 行）：

```cpp
167     { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
168       __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
169       __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
170       __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
171       __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
172       __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
173       __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
174       __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
175       sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_lo, diff_lo));
176       sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_hi, diff_hi)); }
```

每 32 字节（32 个 u8）的处理逻辑：

1. 第 167/168 行：`_mm256_loadu_si256` 读取 32 个 u8（不需要对齐）。
2. 第 169/170 行：`_mm256_castsi256_si256(va)` 取 va 的低 128 位（16 个 u8），`_mm256_cvtepu8_epi16` 把这 16 个 u8 零扩展成 16 个 `int16`（放进 256 位寄存器）。同理 `vb_lo`。
3. 第 171 行：`_mm256_sub_epi16` 做 16 路 int16 减法得到 `diff_lo`。差值范围 `[-255, 255]`，在 int16 范围内。
4. 第 172–174 行：对 va/vb 的高 128 位做同样的事，得到 `diff_hi`。
5. 第 175 行：`_mm256_madd_epi16(diff_lo, diff_lo)`——这是关键指令：对 16 对 int16 做乘法（每对是同一个 diff，相当于平方），再把相邻两对相加成 8 个 int32。然后 `_mm256_add_epi32` 累加到 `sum0`。
6. 第 176 行：对 `diff_hi` 同样处理。

所以一路（32 个 u8）产生 8 个 int32 平方差的部分和，累加到 `sum0`。4 路（k=0..3）共 128 个 u8 一次循环处理完。

循环结束后（第 215–232 行）处理剩余的 32 倍数：

```cpp
215   for (; i + 32 <= dim; i += 32) {
216     __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
217     __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
...
230     sum0 = _mm256_add_epi32(sum0, sq_lo);
231     sum0 = _mm256_add_epi32(sum0, sq_hi);
232   }
```

逻辑同上，但只处理 32 个 u8，累加到 `sum0`。

然后归约 4 路累加器（第 234–242 行）：

```cpp
234   sum0 = _mm256_add_epi32(sum0, sum1);
235   sum2 = _mm256_add_epi32(sum2, sum3);
236   sum0 = _mm256_add_epi32(sum0, sum2);
237
238   __m128i lo = _mm256_castsi256_si128(sum0);
239   __m128i hi = _mm256_extracti128_si256(sum0, 1);
240   __m128i combined = _mm_add_epi32(lo, hi);
241   combined = _mm_hadd_epi32(combined, combined);
242   combined = _mm_hadd_epi32(combined, combined);
243   float result = static_cast<float>(_mm_cvtsi128_si32(combined));
```

- 234–236：4 路 → 1 路。
- 238：取低 128 位（4 个 int32）。
- 239：取高 128 位（4 个 int32）。
- 240：上下相加得 4 个 int32。
- 241–242：两次水平加法（`_mm_hadd_epi32`）把 4 个 int32 折叠成 1 个（标号 0 位置）。
- 243：`_mm_cvtsi128_si32` 取标号 0 的 int32，cast 成 float——**这里整型累加结果转 float，因为 dim×255² 在 f32 精度范围内（见 2.4.9）所以无损**。

最后尾部标量（第 245–248 行）处理 `dim % 32` 个剩余分量：

```cpp
245   for (; i < dim; ++i) {
246     const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
247     result += diff * diff;
248   }
249   return result;
```

> **注意**：尾部用 float 累加，主循环用 int32 累加——但因为 2.4.9 的精度保证，两者在 byte 维度 ≤ 258 时位级别一致；超过时按既定归约顺序也一致。

#### 2.4.10.2 `typed_l2_distance_int8_simd`（第 253–350 行）

逻辑与 uint8 版本完全对称，唯一区别是用 `_mm256_cvtepi8_epi16`（有符号扩展）代替 `_mm256_cvtepu8_epi16`（零扩展）。int8 差值范围 `[-255, 255]`，仍在 int16 范围内，所以 `_mm256_madd_epi16` 同样安全。

#### 2.4.10.3 `horizontal_sum_ps`（第 354–361 行）

```cpp
354 inline float horizontal_sum_ps(__m256 value) {
355   __m128 lo = _mm256_castps256_ps128(value);
356   __m128 hi = _mm256_extractf128_ps(value, 1);
357   __m128 sum = _mm_add_ps(lo, hi);
358   sum = _mm_hadd_ps(sum, sum);
359   sum = _mm_hadd_ps(sum, sum);
360   return _mm_cvtss_f32(sum);
361 }
```

把 8 个 float（`__m256`）水平求和成 1 个 float。这是 float 路径的归约工具。逻辑：

- 取低 128（4 float）+ 高 128（4 float）→ 4 float。
- 两次 `_mm_hadd_ps` 把 4 float 折叠成 1 float。
- `_mm_cvtss_f32` 取标号 0 的 float。

#### 2.4.10.4 `typed_l2_distance_float_query_uint8_simd`（第 363–407 行）

这是**查询侧**的距离函数：查询向量是 float（`span<const element_t>`），存储向量是 uint8。

```cpp
363 inline float typed_l2_distance_float_query_uint8_simd(const span<const element_t> query,
364                                                       const byte_t* stored,
365                                                       u32 dim) {
366   const float* q = query.data();
367   const u8* s = reinterpret_cast<const u8*>(stored);
368   __m256 acc0 = _mm256_setzero_ps();
369   __m256 acc1 = _mm256_setzero_ps();
370   __m256 acc2 = _mm256_setzero_ps();
371   __m256 acc3 = _mm256_setzero_ps();
372
373   u32 i = 0;
374   for (; i + 32 <= dim; i += 32) {
375     { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i));
376       __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
377       __m256 qv = _mm256_loadu_ps(q + i);
378       __m256 diff = _mm256_sub_ps(qv, sv);
379       acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff, diff)); }
```

每步处理 32 个分量（4 路 × 8）。一路的处理：

1. 第 375 行：`_mm_loadl_epi64` 读取 8 个 u8（64 位）。
2. 第 376 行：`_mm256_cvtepu8_epi32` 把 8 个 u8 零扩展成 8 个 int32，`_mm256_cvtepi32_ps` 再转成 8 个 float。即把 byte 存储分量解码成 float。
3. 第 377 行：读取 8 个查询 float。
4. 第 378 行：float 减法。
5. 第 379 行：`diff * diff` 累加到 `acc0`。

注意这里**全程 float 累加**——因为查询侧本身是 float，没有整型累加的选项。累加器是 `__m256`（8 float），4 路独立。

循环结束后用 `horizontal_sum_ps` 归约（第 397–400 行），尾部标量处理 `dim % 32`（第 402–405 行）。

#### 2.4.10.5 `typed_l2_distance_float_query_int8_simd`（第 408–452 行）

与 uint8 版本对称，把 `_mm256_cvtepu8_epi32` 换成 `_mm256_cvtepi8_epi32`。

#### 2.4.10.6 顶层分发 `typed_l2_distance` / `typed_l2_distance_float_query`

```cpp
456 inline float typed_l2_distance(const byte_t* lhs,
457                                VectorDType lhs_dtype,
458                                const byte_t* rhs,
459                                VectorDType rhs_dtype,
460                                u32 dim) {
461 #ifdef __AVX2__
462   if (lhs_dtype == rhs_dtype) {
463     if (lhs_dtype == VectorDType::uint8) {
464       return typed_l2_distance_uint8_simd(lhs, rhs, dim);
465     }
466     if (lhs_dtype == VectorDType::int8) {
467       return typed_l2_distance_int8_simd(lhs, rhs, dim);
468     }
469   }
470 #endif
471   float sum = 0.0f;
472   for (u32 i = 0; i < dim; ++i) {
473     const float diff = vector_component_as_float(lhs, lhs_dtype, i) -
474                        vector_component_as_float(rhs, rhs_dtype, i);
475     sum += diff * diff;
476   }
477   return sum;
478 }
```

`typed_l2_distance` 是**同/异 dtype 通用 L2 距离**：

- 如果 `__AVX2__` 且两侧 dtype 相同且是 uint8/int8，走 SIMD 整型路径。
- 否则（float32、或 dtype 不同、或无 AVX2）走标量路径：逐分量解码成 float 再算。

注意 float32↔float32 没有专门 SIMD 路径——它走的是标量循环。这是因为旧的 `distance.hh`（2.5 节）已有 float32 专用 AVX 实现，新代码里 float32↔float32 通常用 GPU 算或调用 `distance.hh::l2`。

```cpp
480 inline float typed_l2_distance_float_query(const span<const element_t> query,
481                                           const byte_t* stored,
482                                           VectorDType stored_dtype,
483                                           u32 dim) {
484 #ifdef __AVX2__
485   if (stored_dtype == VectorDType::uint8) {
486     return typed_l2_distance_float_query_uint8_simd(query, stored, dim);
487   }
488   if (stored_dtype == VectorDType::int8) {
489     return typed_l2_distance_float_query_int8_simd(query, stored, dim);
490   }
491 #endif
492   float sum = 0.0f;
493   for (u32 i = 0; i < dim; ++i) {
494     const float diff = query[i] - vector_component_as_float(stored, stored_dtype, i);
495     sum += diff * diff;
496   }
497   return sum;
498 }
```

`typed_l2_distance_float_query` 是**查询侧专用**：查询是 float，存储是任意 dtype。这是 CPU 侧 rerank 的核心函数（GPU 侧有自己的 kernel，见第 18 课）。

> **设计观察**：`vector_dtype.hh` 把"dtype 解析、编解码、距离计算"全部内联在头文件里——因为这些都是热路径，内联可以消除函数调用开销，且让编译器在编译期根据 dtype 常量折叠分支。代价是编译时间增加和代码膨胀，但对向量检索这种距离计算占总时间 80%+ 的场景是值得的。

---

## 2.5 `distance.hh`：旧版 float32 专用 L2

这个文件是项目早期从 hnswlib/flann 借来的 float32 L2 实现，现在主要被离线构建路径（见第 12/13 课）和某些 CPU 侧 rerank 调用。

```cpp
1   #pragma once
2
3   #include "common/types.hh"
4
5   #ifdef __AVX__
6   #include <x86intrin.h>
7   #endif
8
9   #ifdef __AVX__
10  // taken from https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h#L61
11  static f32 L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
12    f32* pVect1 = (f32*)pVect1v;
13    f32* pVect2 = (f32*)pVect2v;
14    size_t qty = *((size_t*)qty_ptr);
15    f32 __attribute__((aligned(32))) TmpRes[8];
16    size_t qty16 = qty >> 4;
17
18    const f32* pEnd1 = pVect1 + (qty16 << 4);
19
20    __m256 diff, v1, v2;
21    __m256 sum = _mm256_set1_ps(0);
22
23    while (pVect1 < pEnd1) {
24      v1 = _mm256_loadu_ps(pVect1);
25      pVect1 += 8;
26      v2 = _mm256_loadu_ps(pVect2);
27      pVect2 += 8;
28      diff = _mm256_sub_ps(v1, v2);
29      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
30
31      v1 = _mm256_loadu_ps(pVect1);
32      pVect1 += 8;
32      v2 = _mm256_loadu_ps(pVect2);
33      pVect2 += 8;
34      diff = _mm256_sub_ps(v1, v2);
35      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
36    }
37
38    _mm256_store_ps(TmpRes, sum);
39    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
40  }
41
42  #endif
```

- **第 5/9 行**：注意是 `__AVX__`（不是 `__AVX2__`）——AVX1 即可，因为只用了 256 位 float 运算（`_mm256_*ps`），不需要 AVX2 的整数/置换指令。
- **第 11 行 `L2SqrSIMD16ExtAVX`**：来自 hnswlib 的 `space_l2.h`，注释里贴了源 URL。每次循环处理 16 个 float（2 个 `__m256`）。
- **第 15 行**：`TmpRes` 数组按 32 字节对齐，因为第 38 行 `_mm256_store_ps` 要求对齐存储。
- **第 39 行**：手动水平求和（8 个 float 相加）——比 `_mm_hadd_ps` 链路更直观，但可能略慢。
- **第 16 行 `qty >> 4`**：把数量右移 4 位 = 除以 16，得到 16-float 块的数量。然后第 18 行 `qty16 << 4` 乘回 16 得到对齐后的结束指针。

```cpp
46  static f32 l2(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) {
47    const f32* a = lhs.data();
48    const f32* b = rhs.data();
49
50    const f32* last = a + dim;
51    f32 result = 0.;
52
53  #ifdef __AVX__
54    f32 diff0;
55    const size_t qty16 = dim >> 4 << 4;
56    result = L2SqrSIMD16ExtAVX(a, b, &qty16);
57
58    a += qty16;
59    b += qty16;
60
61  #else
62    // taken from https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/dist.h
63    f32 diff0, diff1, diff2, diff3;
64    const f32* unroll_group = last - 3;
65
66    /* Process 4 items with each loop for efficiency. */
67    while (a < unroll_group) {
68      diff0 = a[0] - b[0];
69      diff1 = a[1] - b[1];
70      diff2 = a[2] - b[2];
71      diff3 = a[3] - b[3];
72      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
73      a += 4;
74      b += 4;
75    }
76  #endif
77    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
78    while (a < last) {
79      diff0 = *a++ - *b++;
80      result += diff0 * diff0;
81    }
82
83    return result;
84  }
85
86  struct L2Distance {
87    static f32 dist(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) { return l2(lhs, rhs, dim); }
88  };
```

- **`l2`**：float32 L2 距离。AVX 路径调 `L2SqrSIMD16ExtAVX` 处理 16 的倍数，剩余 0–15 个走尾部标量循环。非 AVX 路径用 4 路展开标量循环（来自 flann）。
- **第 78–81 行尾部**：处理 16 的倍数之外的剩余分量（标量）。注释说"Not needed for standard vector lengths"——常见向量维度（128/256/384/512/768/960）都是 16 的倍数，但代码仍然兜底。
- **第 86–88 行 `L2Distance`**：一个静态函数包装器，用作模板策略类（一些通用代码会用 `Distance::dist(...)` 调用模式）。

> **`vector_dtype.hh` vs `distance.hh` 的关系**：`vector_dtype.hh` 是新一代多 dtype 实现，`distance.hh` 是旧的 float32 专用实现。新代码优先用 `vector_dtype.hh::typed_l2_distance_float_query`；`distance.hh` 主要服务于离线构建（那里数据已经是 float32）和测试。两者并存是渐进迁移的结果。

---

## 2.6 `configuration.hh`：`IndexConfiguration` 详解

这是本课最重要的文件，定义了整个系统所有可调参数。451 行，分四部分：成员字段、构造函数、`add_options()`、`validate()`、`operator<<`。

### 2.6.1 基类 `Configuration` 回顾

`IndexConfiguration` 继承自 `Configuration`（定义在 `rdma-library/library/configuration.hh`，详见第 4 课）。这里给出关键成员：

```cpp
// rdma-library/library/configuration.hh（节选，详见第 4 课）
class Configuration {
public:
  i32 max_send_queue_wr{1024};      // RDMA 发送队列深度
  i32 max_recv_queue_wr{1024};      // RDMA 接收队列深度
  i32 max_poll_cqes{16};            // 每次 poll CQ 的最大条目数
  u32 port{1234};                   // RDMA CM 端口
  str ib_device;                    // IB 设备名
  u32 device_port{1};               // IB 物理端口
  bool is_server{false};            // 是否是存储服务端
  vec<str> server_nodes;            // 存储节点端点列表
  vec<str> client_nodes;            // 客户端节点列表
  u32 num_clients{1};               // 每个存储节点服务的客户端数
  bool is_initiator{false};         // 是否是发起查询的计算端
protected:
  po::options_description desc{"Allowed options"};  // boost::program_options 描述符
public:
  u32 num_server_nodes() const { return server_nodes.size(); }
  u32 num_client_nodes() const { return client_nodes.size(); }
protected:
  static void exit_with_help_message(char** argv);
  void process_program_options(int argc, char** argv);
};
```

关键点：

- **角色判断**：`is_server`（存储节点）、`is_initiator`（计算节点发起方）。`Configuration` 的校验保证 `is_server` 和 `is_initiator` 互斥——一个进程要么是存储端要么是计算端。
- **`server_nodes`**：所有存储节点端点列表。`num_server_nodes()` 返回其大小。
- **`desc`**：boost::program_options 的选项描述符，子类通过 `desc.add_options()(...)` 添加自己的选项。`process_program_options` 会解析 argv 填充字段并校验基类选项。
- **`exit_with_help_message`**：打印帮助信息并 `exit(1)`。

### 2.6.2 `IndexConfiguration` 字段定义

```cpp
// /home/xjs/experiment/dvstor/src/common/configuration.hh
15  namespace configuration {
16
17  struct Parameters {
18    u32 num_threads{};
19    u32 gpu_rdma_qps{};
20  };
21
22  class IndexConfiguration : public Configuration {
23  public:
24    filepath_t index_prefix{};
25    filepath_t server_index_file{};
26    u32 num_threads{};
27    i32 seed{1234};
28    bool disable_thread_pinning{};
29
30    u32 dim{};
31    u32 max_vectors{1'000'000};
32    u32 k{};
33    u32 R{64};
34    u32 beam_width_construction{200};
35    f64 alpha{1.2};
36    str vector_data_type{"auto"};
37
38    u32 gpu_device{};
39    bool enable_breakdown{true};
40    u32 gpu_query_slots{256};
41    u32 gpu_memory_limit_gb{40};
42    u32 gpu_memory_reserve_gb{4};
43    u32 gpu_resident_pq_budget_mb{4096};
44    u32 gpu_adjacency_cache_mb{0};
45    u32 gpu_adjacency_cache_ways{4};
46    u32 gpu_exact_cache_mb{0};
47    u32 gpu_exact_cache_ways{4};
48    u32 gpu_bootstrap_window_mb{64};
49    u32 gpu_bootstrap_windows{2};
50    u32 gpu_graph_prefetch_depth{32};
51    u32 gpu_graph_cache_ttl_us{5000};
52    u32 gpu_traversal_beam_width{128};
53    u32 gpu_final_rerank_width{64};
54    u32 gpu_max_expansions{384};
55    u32 gpu_entry_seed_count{16};
56    u32 gpu_delta_anchor_probes{32};
57    u32 gpu_rdma_qps{4};
58    u32 gpu_persistent_blocks_per_sm{4};
59    // Compute-side mutation support is optional for query-only deployments.
60    // Keep it enabled by default so existing service configurations preserve
61    // their insert/upsert/erase behavior.
62    bool enable_updates{true};
63    u32 update_visibility_us{10'000};
64    u32 delta_budget_mb{256};
65    u32 gpu_delta_maintenance_period_ms{10};
66
67    u32 storage_id{};
68    vec<str> storage_peers;
69    u32 storage_owner_coroutines{4};
70    u32 storage_owner_batch_max{16};
71    u32 storage_owner_peer_rdma_tokens{8};
72    u32 storage_owner_rpc_depth{8};
73    u32 storage_owner_rpc_timeout_ms{30'000};
74    u32 storage_owner_search_snapshot_batch{64};
75    str storage_owner_update_mode{"exact"};
76    str storage_owner_maintenance_mode{"off"};
77    u32 storage_owner_maintenance_workers{};
78    u32 storage_owner_maintenance_queue_depth{65'536};
79    str storage_owner_reverse_mode{"async"};
80    u32 storage_owner_reverse_queue_depth{65'536};
81    u32 storage_owner_reverse_coalesce_max{256};
82
83    u32 mn_memory_gb{10};
```

逐字段说明（按逻辑分组）：

#### 索引与基础参数（第 24–36 行）

| 字段 | 默认值 | 含义 | 消费课 |
|------|--------|------|--------|
| `index_prefix` | 空 | 索引文件前缀（目录+文件名前缀），所有 metadata/graph shard/PQ model/PQ code 都共享此前缀 | 第 6/7/8/15 课 |
| `server_index_file` | 空 | 本存储节点加载的本地 graph shard 文件路径 | 第 23 课 |
| `num_threads` | 0 | CPU 控制/更新线程数 | 第 3/14/27 课 |
| `seed` | 1234 | 随机种子（构造期 Vamana 图采样、路由采样等） | 第 12/13 课 |
| `disable_thread_pinning` | false | 是否禁用 CPU 线程绑核 | 第 3/27 课 |
| `dim` | 0 | 向量维度 | 全项目 |
| `max_vectors` | 1,000,000 | 逻辑向量 ID 范围上限（注意是范围上限，不是实际数量） | 第 6/8/10 课 |
| `k` | 0 | 用户请求的最近邻个数 | 第 14/27 课 |
| `R` | 64 | Vamana 图最大出度（被 `R <= 255` 校验，因为 schema-15 用 u8 存出度） | 第 7/12/13 课 |
| `beam_width_construction` | 200 | 存储侧在线图维护的 beam width | 第 15/25 课 |
| `alpha` | 1.2 | Vamana RobustPrune 的多样性参数（>1 倾向更长边，更准确） | 第 12/13/25 课 |
| `vector_data_type` | "auto" | 存储向量 dtype，见 2.4 节 | 全项目 |

#### GPU 查询引擎参数（第 38–65 行）

| 字段 | 默认值 | 含义 | 消费课 |
|------|--------|------|--------|
| `gpu_device` | 0 | CUDA 设备 ID | 第 11/17/27 课 |
| `enable_breakdown` | true | 是否收集 per-request 拆解计时样本 | 第 30 课 |
| `gpu_query_slots` | 256 | 最大并发 GPU 查询槽位（被校验 ≤ 4096） | 第 11/14/27 课 |
| `gpu_memory_limit_gb` | 40 | 查询引擎显式 GPU 分配硬上限 | 第 11/27 课 |
| `gpu_memory_reserve_gb` | 4 | 给 CUDA runtime/transport 预留的显存（必须 < limit） | 第 11/27 课 |
| `gpu_resident_pq_budget_mb` | 4096 | 常驻 GPU 的动态插入向量 PQ 编码预算 | 第 9/10/11 课 |
| `gpu_adjacency_cache_mb` | 0 | GPU 紧凑图缓存预算 | 第 19 课 |
| `gpu_adjacency_cache_ways` | 4 | 紧凑图缓存组相联度（必须 = 4） | 第 19 课 |
| `gpu_exact_cache_mb` | 0 | GPU 精确向量缓存预算 | 第 19 课 |
| `gpu_exact_cache_ways` | 4 | 精确向量缓存组相联度（必须 = 4） | 第 19 课 |
| `gpu_bootstrap_window_mb` | 64 | 一次性 PQ bootstrap 单次 RDMA 读上限 | 第 9/22 课 |
| `gpu_bootstrap_windows` | 2 | 并发 bootstrap 读数（≤ 16） | 第 9/22 课 |
| `gpu_graph_prefetch_depth` | 32 | 单次查询并发获取的图记录数（≤ 32） | 第 19/20 课 |
| `gpu_graph_cache_ttl_us` | 5000 | 图缓存最大 age（μs）；0 表示基线图常驻 | 第 19 课 |
| `gpu_traversal_beam_width` | 128 | GPU 图导航的 OPQ/PQ beam width（≥ k，≤ 256） | 第 20 课 |
| `gpu_final_rerank_width` | 64 | 终态 rerank 拉取的精确向量数（≥ k，≤ 256） | 第 18/20 课 |
| `gpu_max_expansions` | 384 | 单查询最大图扩展次数（≥ traversal beam，≤ 4096） | 第 20 课 |
| `gpu_entry_seed_count` | 16 | 查询起点评分的常驻入口点数（≤ 512） | 第 20 课 |
| `gpu_delta_anchor_probes` | 32 | 每查询探测的动态 anchor 桶数（≤ 64） | 第 10/20 课 |
| `gpu_rdma_qps` | 4 | 每存储节点的 GPUNetIO QP 数（≤ 32） | 第 22 课 |
| `gpu_persistent_blocks_per_sm` | 4 | 每 SM 启动的持久化查询 block 数（≤ 16） | 第 21 课 |
| `enable_updates` | true | 是否启用计算侧 insert/upsert/erase | 第 14/27/28 课 |
| `update_visibility_us` | 10,000 | 更新发布到 GPU delta 的最大延迟（μs） | 第 10/14 课 |
| `delta_budget_mb` | 256 | GPU delta 索引显存预算 | 第 10 课 |
| `gpu_delta_maintenance_period_ms` | 10 | GPU delta 退役/存储水印轮询周期（ms） | 第 10 课 |

#### 存储节点更新参数（第 67–81 行）

| 字段 | 默认值 | 含义 | 消费课 |
|------|--------|------|--------|
| `storage_id` | 0 | 本存储节点 shard ID（从 0 开始） | 第 23/24 课 |
| `storage_peers` | 空 | 有序存储节点端点列表（必须与 `num_server_nodes()` 相等） | 第 22/23 课 |
| `storage_owner_coroutines` | 4 | 每存储端更新 worker 的协程数 | 第 24/25 课 |
| `storage_owner_batch_max` | 16 | 单次存储 RPC 批量变更上限 | 第 24/25 课 |
| `storage_owner_peer_rdma_tokens` | 8 | 每存储数据 QP 的 peer RDMA 读令牌数 | 第 23 课 |
| `storage_owner_rpc_depth` | 8 | 每存储节点的在飞变更批次数 | 第 24 课 |
| `storage_owner_rpc_timeout_ms` | 30,000 | 变更 RPC 超时 | 第 24 课 |
| `storage_owner_search_snapshot_batch` | 64 | 更新搜索时并发节点快照数 | 第 25 课 |
| `storage_owner_update_mode` | "exact" | 动态图更新策略：`exact` 或 `local_stitch` | 第 25 课 |
| `storage_owner_maintenance_mode` | "off" | 后台图维护：`off` 或 `finalize` | 第 25 课 |
| `storage_owner_maintenance_workers` | 0 | 后台精确 finalization worker 数 | 第 25 课 |
| `storage_owner_maintenance_queue_depth` | 65,536 | 有界图维护积压上限；写者在满时反压 | 第 25 课 |
| `storage_owner_reverse_mode` | "async" | 反向更新完成模式：`async` 或 `sync` | 第 25 课 |
| `storage_owner_reverse_queue_depth` | 65,536 | 排队反向更新上限 | 第 25 课 |
| `storage_owner_reverse_coalesce_max` | 256 | 单次合并批次的反向更新上限 | 第 25 课 |

#### 内存参数（第 83 行）

| 字段 | 默认值 | 含义 | 消费课 |
|------|--------|------|--------|
| `mn_memory_gb` | 10 | 存储节点注册内存容量（GiB） | 第 23 课 |

#### `Parameters` 结构（第 17–20 行）

```cpp
17  struct Parameters {
18    u32 num_threads{};
19    u32 gpu_rdma_qps{};
20  };
```

这是一个"子集"结构，用于把配置中的两个字段传递给不需要完整 `IndexConfiguration` 的子系统（如 kernel 启动器）。在 2.6.6 节会看到它被 kernel 上下文使用（见第 17 课）。

### 2.6.3 构造函数

```cpp
85  IndexConfiguration(int argc, char** argv) {
86    add_options();
87    process_program_options(argc, argv);
88    vector_data_type = normalize_mode(vector_data_type);
89    storage_owner_update_mode = normalize_mode(storage_owner_update_mode);
90    storage_owner_maintenance_mode = normalize_mode(storage_owner_maintenance_mode);
91    storage_owner_reverse_mode = normalize_mode(storage_owner_reverse_mode);
92    validate(argv);
93    operator<<(std::cerr, *this);
94  }
```

执行顺序：

1. **第 86 行 `add_options()`**：注册所有 CLI 选项到 `desc`（见 2.6.4）。
2. **第 87 行 `process_program_options(argc, argv)`**：基类方法，用 boost::program_options 解析 argv，填充字段，处理 `--help`/`--version`，并校验基类选项（`is_server`/`is_initiator` 互斥等）。
3. **第 88–91 行**：把四个模式字符串归一化为小写——用户可能在 CLI 写 `EXACT`/`Async`/`Finalize`，统一成小写后便于后续比较。
4. **第 92 行 `validate(argv)`**：项目特有的校验（见 2.6.5）。
5. **第 93 行**：把最终配置打印到 stderr，方便运维核对。

### 2.6.4 解析辅助函数

```cpp
96  filepath_t resolved_index_prefix() const {
97    return index_prefix;
98  }
99
100 VectorDType resolved_vector_dtype() const {
101   return vector_data_type == "auto"
102     ? VectorDType::float32 : parse_vector_dtype(vector_data_type);
103 }
104
105 u32 resolved_storage_owner_construction_width() const {
106   return beam_width_construction;
107 }
108
109 private:
110   static str normalize_mode(str value) {
111     std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
112       return static_cast<char>(std::tolower(ch));
113     });
114     return value;
115   }
```

- **`resolved_index_prefix`**：当前直接返回 `index_prefix`，是个语义包装器，未来可能根据角色拼路径。
- **`resolved_vector_dtype`**：`"auto"` 时返回 `float32`（注意：**不**查文件扩展名，这与 `vector_dtype.hh::resolve_vector_dtype_config` 不同，见 2.4.4 节）。非 auto 时调 `parse_vector_dtype`。
- **`resolved_storage_owner_construction_width`**：返回 `beam_width_construction`，是"存储侧在线图维护用的 beam width"语义包装器。
- **`normalize_mode`**：私有静态方法，把字符串转小写。注意第 111 行 lambda 参数是 `unsigned char`——`std::tolower` 接收 int，传负数 char 会 UB，所以先转 unsigned char。

### 2.6.5 `add_options()`：CLI 选项注册

第 117–274 行用 boost::program_options 注册所有选项。每个选项格式是 `("name", po::value<T>(&field)->default_value(field), "description")`。注意 `default_value(field)` 用的是字段在类声明时的默认值——这意味着默认值有两处来源（类内默认 + CLI 默认），它们必须保持一致。

下表按字段顺序汇总所有选项（含基类选项来自 `Configuration::create_rdma_options()`，详见第 4 课）：

#### 索引/基础选项（第 119–142 行）

```cpp
118   desc.add_options()
119     ("index-prefix", po::value<filepath_t>(&index_prefix),
120      "Prefix shared by metadata, graph shards, PQ model, and PQ code shards.")
121     ("server-index-file", po::value<filepath_t>(&server_index_file),
122      "Local graph shard loaded by this storage node before serving requests.")
123     ("threads,t", po::value<u32>(&num_threads),
124      "CPU control/update threads represented by this process.")
125     ("disable-thread-pinning,p",
126      po::bool_switch(&disable_thread_pinning)->default_value(false),
127      "Disable CPU thread pinning.")
128     ("seed", po::value<i32>(&seed)->default_value(seed),
129      "Deterministic random seed.")
130     ("dim", po::value<u32>(&dim), "Vector dimension.")
131     ("max-vectors", po::value<u32>(&max_vectors)->default_value(max_vectors),
132      "Maximum logical vector identifier range.")
133     ("k", po::value<u32>(&k), "Requested nearest-neighbor count.")
134     ("R", po::value<u32>(&R)->default_value(R), "Maximum graph out-degree.")
135     ("beam-width-construction",
136      po::value<u32>(&beam_width_construction)->default_value(beam_width_construction),
137      "Beam width used by storage-side online graph maintenance.")
138     ("alpha", po::value<f64>(&alpha)->default_value(alpha),
139      "RobustPrune diversity factor.")
140     ("vector-data-type",
141      po::value<str>(&vector_data_type)->default_value(vector_data_type),
142      "Exact-vector storage type: auto, float32, uint8, or int8.")
```

要点：

- **`index-prefix`、`server-index-file`、`dim`、`k`、`threads`**：没有 `default_value`，意味着必须显式提供（`validate()` 会强制校验，见 2.6.6）。
- **`threads,t`**：短选项 `-t`，方便快速指定。
- **`disable-thread-pinning,p`**：用 `po::bool_switch` 而非 `po::value<bool>`——前者是"出现即为 true"的 flag 风格（`--disable-thread-pinning` 而不需要 `--disable-thread-pinning true`）。
- **`seed`、`max-vectors`、`R`、`beam-width-construction`、`alpha`、`vector-data-type`**：有默认值，可省略。

#### GPU 选项（第 144–219 行）

```cpp
144     ("gpu-device", po::value<u32>(&gpu_device)->default_value(gpu_device),
145      "CUDA device used by the persistent query engine.")
146     ("enable-breakdown",
147      po::value<bool>(&enable_breakdown)->default_value(enable_breakdown),
148      "Collect per-request breakdown samples.")
149     ("gpu-query-slots",
150      po::value<u32>(&gpu_query_slots)->default_value(gpu_query_slots),
151      "Maximum concurrent GPU query slots.")
...
201     ("gpu-rdma-qps",
202      po::value<u32>(&gpu_rdma_qps)->default_value(gpu_rdma_qps),
203      "GPU-initiated GPUNetIO QPs per storage node.")
204     ("gpu-persistent-blocks-per-sm",
205      po::value<u32>(&gpu_persistent_blocks_per_sm)->default_value(gpu_persistent_blocks_per_sm),
206      "Persistent query blocks launched per GPU SM.")
207     ("enable-updates",
208      po::value<bool>(&enable_updates)->default_value(enable_updates),
209      "Enable compute-side insert, upsert, and erase submission.")
...
216     ("gpu-delta-maintenance-period-ms",
217      po::value<u32>(&gpu_delta_maintenance_period_ms)
218         ->default_value(gpu_delta_maintenance_period_ms),
219      "GPU delta retirement and storage-watermark polling period.")
```

注意几个细节：

- **`enable-breakdown`、`enable-updates`** 用 `po::value<bool>`（带值）而非 `bool_switch`——这意味着 CLI 要写 `--enable-updates false`。这与 `disable-thread-pinning` 的风格不一致，是历史原因。
- **`gpu-rdma-qps`** 与 `constants.hh::kMaxPeerQps` 不同：前者是 GPU 客户端到存储节点的 QP 数（≤ 32），后者是存储节点之间的 peer QP 数（= 4）。
- 所有 GPU 选项都有默认值，意味着一个"开箱即用"的配置可以省略它们。

#### 存储选项（第 221–273 行）

```cpp
221     ("storage-id", po::value<u32>(&storage_id)->default_value(storage_id),
222      "Zero-based storage shard identifier.")
223     ("storage-peers", po::value<vec<str>>(&storage_peers)->multitoken(),
224      "Ordered storage-node endpoints.")
...
272     ("mn-memory", po::value<u32>(&mn_memory_gb)->default_value(mn_memory_gb),
273      "Storage-node registered-memory capacity in GiB.");
274   }
```

- **`storage-peers`** 用 `multitoken()`——意味着 `--storage-peers 10.0.0.1:1234 10.0.0.2:1234 10.0.0.3:1234` 可以一次传多个值。
- 第 273 行末尾的分号标志着 `desc.add_options()(...)()` 链式调用的结束。

### 2.6.6 `validate()`：校验规则

第 276–372 行是项目的核心校验逻辑。先看辅助函数和必填校验：

```cpp
276 void validate(char** argv) const {
277   const auto fail = [&](const str& message) {
278     std::cerr << "[ERROR]: " << message << std::endl;
279     exit_with_help_message(argv);
280   };
281
282   if (index_prefix.empty()) fail("--index-prefix is required");
283   if (num_threads == 0 || dim == 0 || max_vectors == 0 || k == 0 ||
284       R == 0 || beam_width_construction == 0 || mn_memory_gb == 0) {
285     fail("threads, dim, max-vectors, k, R, beam-width-construction, and mn-memory must be > 0");
286   }
287   if (R > std::numeric_limits<u8>::max()) {
288     fail("--R must be <= 255");
289   }
290   if (k > gpu_final_rerank_width) {
291     fail("--k must not exceed --gpu-final-rerank-width");
292   }
293   try {
294     if (vector_data_type != "auto") (void)parse_vector_dtype(vector_data_type);
295   } catch (const std::exception& error) {
296     fail(str{"invalid --vector-data-type: "} + error.what());
297   }
```

- **第 277–280 行 `fail` lambda**：打印错误并调用基类的 `exit_with_help_message`（打印帮助后 `exit(1)`）。
- **第 282 行**：`--index-prefix` 必填。
- **第 283–286 行**：6 个数值字段必须 > 0。注意 `mn_memory_gb` 也在其中——存储节点必须显式给内存。
- **第 287–289 行**：`R <= 255`，因为 schema-15 索引格式（见第 7 课）用 `u8` 存出度。
- **第 290–292 行**：`k <= gpu_final_rerank_width`——终态 rerank 拉的精确向量数必须 ≥ 用户要的近邻数，否则拿不到 top-k。
- **第 293–297 行**：如果 dtype 不是 auto，调 `parse_vector_dtype` 验证合法性。

接下来是 GPU 配置校验（第 299–321 行），用一个大 `if` 把所有 GPU 边界条件塞在一起：

```cpp
299   if (gpu_query_slots == 0 || gpu_query_slots > 4096 ||
300       gpu_memory_limit_gb == 0 ||
301       gpu_memory_reserve_gb >= gpu_memory_limit_gb ||
302       gpu_resident_pq_budget_mb == 0 ||
303       gpu_adjacency_cache_ways != 4 ||
304       gpu_exact_cache_ways != 4 ||
305       gpu_bootstrap_window_mb == 0 || gpu_bootstrap_windows == 0 ||
306       gpu_bootstrap_windows > 16 ||
307       gpu_graph_prefetch_depth == 0 ||
308       gpu_graph_prefetch_depth > 32 ||
309       gpu_traversal_beam_width < k || gpu_traversal_beam_width > 256 ||
310       gpu_final_rerank_width < k || gpu_final_rerank_width > 256 ||
311       gpu_max_expansions < gpu_traversal_beam_width ||
312       gpu_max_expansions > 4096 ||
313       gpu_entry_seed_count == 0 || gpu_entry_seed_count > 512 ||
314       gpu_delta_anchor_probes == 0 || gpu_delta_anchor_probes > 64 ||
315       gpu_rdma_qps == 0 || gpu_rdma_qps > 32 ||
316       gpu_persistent_blocks_per_sm == 0 ||
317       gpu_persistent_blocks_per_sm > 16 ||
318       update_visibility_us == 0 || delta_budget_mb == 0 ||
319       gpu_delta_maintenance_period_ms == 0) {
320     fail("invalid persistent GPU query configuration");
321   }
```

逐条解读：

| 条件 | 含义 |
|------|------|
| `gpu_query_slots == 0 \|\| > 4096` | 并发槽位在 [1, 4096]，受 GPU 资源限制 |
| `gpu_memory_limit_gb == 0` | 必须有显存预算 |
| `gpu_memory_reserve_gb >= gpu_memory_limit_gb` | 预留必须小于上限（否则可用为负） |
| `gpu_resident_pq_budget_mb == 0` | PQ 预算必须非 0 |
| `gpu_adjacency_cache_ways != 4` | 缓存组相联度固定为 4（kernel 实现 hardcode） |
| `gpu_exact_cache_ways != 4` | 同上 |
| `gpu_bootstrap_window_mb == 0 \|\| windows == 0` | bootstrap 必须有窗口和并发数 |
| `gpu_bootstrap_windows > 16` | 并发 bootstrap 最多 16（受 QP/资源限制） |
| `gpu_graph_prefetch_depth == 0 \|\| > 32` | prefetch depth 在 [1, 32]，受 kernel 内 warp 调度限制 |
| `gpu_traversal_beam_width < k \|\| > 256` | traversal beam ≥ k（否则找不到 top-k），≤ 256（kernel 内数组大小） |
| `gpu_final_rerank_width < k \|\| > 256` | 同上 |
| `gpu_max_expansions < gpu_traversal_beam_width` | 扩展次数至少够一轮 beam |
| `gpu_max_expansions > 4096` | 上限受 kernel 内循环计数器位宽 |
| `gpu_entry_seed_count == 0 \|\| > 512` | 入口种子在 [1, 512] |
| `gpu_delta_anchor_probes == 0 \|\| > 64` | anchor 探测在 [1, 64] |
| `gpu_rdma_qps == 0 \|\| > 32` | GPUNetIO QP 在 [1, 32] |
| `gpu_persistent_blocks_per_sm == 0 \|\| > 16` | 每 SM block 数在 [1, 16]（受 SM 资源限制） |
| `update_visibility_us == 0` | 可见性延迟必须非 0 |
| `delta_budget_mb == 0` | delta 预算必须非 0 |
| `gpu_delta_maintenance_period_ms == 0` | 维护周期必须非 0 |

> **设计观察**：把所有 GPU 校验塞进一个 `if` 是为了"快速失败"——任何一个不满足就立即报错。缺点是错误消息只有一个通用串"invalid persistent GPU query configuration"，用户得自己对照源码找哪个条件挂了。这是一个可改进点（拆成多个 fail 调用，每个带具体消息）。

接下来是存储配置校验（第 323–371 行）：

```cpp
323   if (storage_peers.size() != num_server_nodes()) {
324     fail("--storage-peers must list exactly one endpoint per storage node");
325   }
326   if (storage_id >= num_server_nodes() || storage_owner_coroutines == 0 ||
327       storage_owner_batch_max == 0 ||
328       storage_owner_peer_rdma_tokens == 0 ||
329       storage_owner_rpc_depth == 0 ||
330       storage_owner_rpc_timeout_ms == 0 ||
331       storage_owner_search_snapshot_batch == 0 ||
332       storage_owner_maintenance_queue_depth == 0 ||
333       storage_owner_reverse_queue_depth == 0 ||
334       storage_owner_reverse_coalesce_max == 0) {
335     fail("invalid storage-side update configuration");
336   }
337   if (storage_owner_batch_max > std::numeric_limits<u32>::max() / R) {
338     fail("storage-owner batch invalidation capacity exceeds u32");
339   }
```

- **第 323–325 行**：`storage_peers` 数量必须等于 `num_server_nodes()`（即 `server_nodes.size()`，来自基类）。这保证 `--storage-peers` 和 `--servers`（基类选项）一致。
- **第 326–336 行**：存储侧字段非 0 校验。
- **第 337–339 行**：`storage_owner_batch_max * R` 不能溢出 `u32`——因为图维护时一个 batch 里每个变更可能要失效 R 条边，乘积要能放进 u32。这是预防算术溢出的防御性校验。

然后是模式字符串校验（第 340–368 行）：

```cpp
340   if (storage_owner_update_mode != "exact" &&
341       storage_owner_update_mode != "local_stitch") {
342     fail("--storage-owner-update-mode must be exact or local_stitch");
343   }
344   if (storage_owner_reverse_mode != "async" &&
345       storage_owner_reverse_mode != "sync") {
346     fail("--storage-owner-reverse-mode must be async or sync");
347   }
348   if (storage_owner_maintenance_mode != "off" &&
349       storage_owner_maintenance_mode != "finalize") {
350     fail("--storage-owner-maintenance-mode must be off or finalize");
351   }
352   if (storage_owner_update_mode == "local_stitch" &&
353       storage_owner_maintenance_mode != "finalize") {
354     fail("local_stitch requires finalize maintenance");
355   }
356   if (storage_owner_maintenance_mode == "finalize" &&
357       storage_owner_update_mode != "local_stitch") {
358     fail("finalize maintenance is the stage2 of local_stitch updates");
359   }
360   if (storage_owner_maintenance_mode == "finalize" &&
361       storage_owner_maintenance_workers == 0) {
362     fail("finalize maintenance requires storage-owner-maintenance-workers > 0");
363   }
364   if (storage_owner_maintenance_mode == "finalize" &&
365       static_cast<u64>(storage_owner_maintenance_queue_depth) <
366         static_cast<u64>(storage_owner_batch_max) * 2) {
367     fail("finalize maintenance queue depth must cover two intents per RPC batch");
368   }
```

- 第 340–351 行：三个模式字符串取值校验。注意 `update_mode` 有 `exact`（精确重算）和 `local_stitch`（局部缝合）两种策略，`maintenance_mode` 有 `off` 和 `finalize` 两种。
- 第 352–359 行：**`local_stitch` 和 `finalize` 必须同时启用**——`local_stitch` 是在线快速缝合（不准确），`finalize` 是后台精确重算。两者是同一更新策略的两阶段，单独启用任一无意义。
- 第 360–363 行：`finalize` 模式需要至少 1 个 worker。
- 第 364–368 行：`finalize` 模式的维护队列深度必须 ≥ 2 × `batch_max`——因为一个 RPC batch 可能产生两个 intent（前向 + 反向），队列要能容纳。

最后是角色相关校验（第 369–371 行）：

```cpp
369   if (is_server && server_index_file.empty()) {
370     fail("storage node requires --server-index-file");
371   }
372 }
```

存储节点必须指定 `--server-index-file`，因为每个存储节点要加载自己的本地 graph shard。

### 2.6.7 `operator<<`：配置摘要打印

第 375–447 行实现 `operator<<`，根据角色打印不同的摘要。

```cpp
375 friend std::ostream& operator<<(
376     std::ostream& output, const IndexConfiguration& config) {
377   output << static_cast<const Configuration&>(config);
378   constexpr i32 width = 34;
379   constexpr i32 line_width = 68;
380   output << std::left << std::setfill(' ');
381
382   if (config.is_initiator) {
383     output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
...
436     output << std::setfill('=') << std::setw(line_width) << "" << '\n';
437   } else if (config.is_server) {
438     output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
439     output << std::setw(width) << "storage shard: "
440            << config.server_index_file << '\n';
441     output << std::setw(width) << "storage id: " << config.storage_id << '\n';
442     output << std::setw(width) << "registered memory GiB: "
443            << config.mn_memory_gb << '\n';
444     output << std::setfill('=') << std::setw(line_width) << "" << '\n';
445   }
446   return output;
447 }
```

- **第 377 行**：先打印基类 `Configuration` 的摘要（RDMA 设备、端口、节点列表等，见第 4 课）。
- **第 378–379 行**：`width = 34` 是字段名列宽，`line_width = 68` 是分隔线长度。
- **第 382–436 行**：计算节点（`is_initiator`）打印完整摘要：索引信息、查询引擎类型、GPU 配置、存储更新配置等。
- **第 437–444 行**：存储节点（`is_server`）只打印索引前缀、shard 文件、storage id、注册内存——存储节点不关心 GPU 配置。
- 注意用 `else if`——一个进程要么是计算端要么是存储端，不会两者都是。

计算节点摘要的关键行（节选）：

```cpp
393     output << std::setw(width) << "query engine: "
394            << "persistent_gpu_opq_pq" << '\n';
395     output << std::setw(width) << "remote transport: "
396            << "GPU-initiated GPUNetIO" << '\n';
```

这两行是"硬编码的架构标识"——告诉运维当前查询引擎是"持久化 GPU + OPQ/PQ"，远端传输是"GPU 发起的 GPUNetIO"。这是 dvstor 区别于传统 CPU ANN 系统的核心特征（见第 11/22 课）。

---

## 2.7 `index_path.hh`：索引文件路径拼装

这个文件提供 7 个内联函数，把"索引前缀 + 节点信息"拼成具体文件路径。所有函数都在 `namespace index_path` 里。

```cpp
// /home/xjs/experiment/dvstor/src/common/index_path.hh
1   #pragma once
2
3   #include <library/utils.hh>
4
5   #include "types.hh"
6
7   namespace index_path {
8
9   inline filepath_t base_directory(const filepath_t& data_path) {
10    lib_assert(!data_path.empty(), "data path must not be empty");
11    return data_path.has_extension() ? data_path.parent_path() : data_path;
12  }
13
14  inline filepath_t default_prefix(const filepath_t& data_path, u32 m, u32 ef_construction) {
15    return base_directory(data_path) / "dump" /
16           ("index_m" + std::to_string(m) + "_efc" + std::to_string(ef_construction));
17  }
18
19  inline filepath_t resolve_prefix(const filepath_t& data_path,
20                                   const filepath_t& explicit_prefix,
21                                   u32 m,
22                                   u32 ef_construction) {
23    return explicit_prefix.empty() ? default_prefix(data_path, m, ef_construction) : explicit_prefix;
24  }
25
26  inline filepath_t shard_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
27    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" + std::to_string(num_nodes) +
28                      ".dat");
29  }
30
31  inline filepath_t owner_idmap_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
32    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
33                      std::to_string(num_nodes) + ".idmap");
34  }
35
36  inline filepath_t anchor_file(const filepath_t& prefix) {
37    return filepath_t(prefix.string() + ".anchors");
38  }
39
40  inline filepath_t navigation_model_file(const filepath_t& prefix, u32 subquantizers) {
41    return filepath_t(prefix.string() + ".pq" + std::to_string(subquantizers));
42  }
43
44  inline filepath_t navigation_code_file(const filepath_t& prefix,
45                                         size_t node_ordinal,
46                                         size_t num_nodes,
47                                         u32 subquantizers) {
48    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
49                      std::to_string(num_nodes) + ".pq" +
50                      std::to_string(subquantizers) + ".codes");
51  }
52
53  inline filepath_t navigation_code_for_shard(const filepath_t& shard_file_path,
54                                             u32 subquantizers) {
55    filepath_t result = shard_file_path;
56    result.replace_extension(".pq" + std::to_string(subquantizers) + ".codes");
57    return result;
58  }
59
60  }  // namespace index_path
```

逐函数讲解：

#### `base_directory`（第 9–12 行）

```cpp
9   inline filepath_t base_directory(const filepath_t& data_path) {
10    lib_assert(!data_path.empty(), "data path must not be empty");
11    return data_path.has_extension() ? data_path.parent_path() : data_path;
12  }
```

- **第 3 行 `#include <library/utils.hh>`**：引入 `lib_assert` 宏（定义在 `rdma-library/library/utils.hh`）：
  ```cpp
  // rdma-library/library/utils.hh
  #define lib_assert(cond, msg)        \
    do {                               \
      if (!(cond)) {                   \
        std::cerr << msg << std::endl; \
        std::exit(EXIT_FAILURE);       \
      }                                \
    } while (0)
  ```
  即"条件不满足就打印消息并 `exit(EXIT_FAILURE)`"。比标准 `assert` 更友好（不会被 `NDEBUG` 关掉），适合做"不该发生"的防御性检查。
- **第 10 行**：`data_path` 不能为空。
- **第 11 行**：如果 `data_path` 有扩展名（例如 `/data/sift100m.fbin`），返回其父目录（`/data`）；否则（例如 `/data`）直接返回它自己。这是一个"统一目录"的辅助——无论用户传文件还是目录，最终都得到目录。

#### `default_prefix`（第 14–17 行）

```cpp
14  inline filepath_t default_prefix(const filepath_t& data_path, u32 m, u32 ef_construction) {
15    return base_directory(data_path) / "dump" /
16           ("index_m" + std::to_string(m) + "_efc" + std::to_string(ef_construction));
17  }
```

拼默认索引前缀：`<数据目录>/dump/index_m{m}_efc{ef_construction}`。例如 `/data/dump/index_m64_efc200`。这里 `m` 是 Vamana 的 `R`（最大出度），`ef_construction` 是构建时的 ef 参数。`/` 运算符是 `std::filesystem::path` 的路径拼接。

#### `resolve_prefix`（第 19–24 行）

```cpp
19  inline filepath_t resolve_prefix(const filepath_t& data_path,
20                                   const filepath_t& explicit_prefix,
21                                   u32 m,
22                                   u32 ef_construction) {
23    return explicit_prefix.empty() ? default_prefix(data_path, m, ef_construction) : explicit_prefix;
24  }
```

如果用户显式提供了 `explicit_prefix`，直接用；否则用 `default_prefix` 生成。这是"显式优先，自动生成兜底"的常见模式。

#### `shard_file`（第 26–29 行）

```cpp
26  inline filepath_t shard_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
27    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" + std::to_string(num_nodes) +
28                      ".dat");
29  }
```

拼图分片文件：`{prefix}_node{ordinal}_of{num_nodes}.dat`。例如 `index_m64_efc200_node0_of4.dat`。这是 schema-15 索引格式（见第 7 课）的图分片文件命名约定。`node_ordinal` 是 0-based 的节点序号，`num_nodes` 是总节点数。

#### `owner_idmap_file`（第 31–34 行）

```cpp
31  inline filepath_t owner_idmap_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
32    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
33                      std::to_string(num_nodes) + ".idmap");
34  }
```

拼 owner idmap 文件：`{prefix}_node{ordinal}_of{num_nodes}.idmap`。`.idmap` 文件存储"本 shard 的局部节点序号 → 全局向量 ID"映射，详见第 6 课"Vamana 图格式与 anchor/idmap"。

#### `anchor_file`（第 36–38 行）

```cpp
36  inline filepath_t anchor_file(const filepath_t& prefix) {
37    return filepath_t(prefix.string() + ".anchors");
38  }
```

拼 anchor 文件：`{prefix}.anchors`。anchor 是 Vamana 图的全局入口点集合，详见第 6 课。

#### `navigation_model_file`（第 40–42 行）

```cpp
40  inline filepath_t navigation_model_file(const filepath_t& prefix, u32 subquantizers) {
41    return filepath_t(prefix.string() + ".pq" + std::to_string(subquantizers));
42  }
```

拼 PQ 模型文件：`{prefix}.pq{subquantizers}`。`subquantizers` 是 OPQ/PQ 的子量化器数量（即把向量切成几段）。例如 `index_m64_efc200.pq32`。这是 OPQ 模型文件，详见第 9 课"GPU 类型/遥测/PQ 模型"。

#### `navigation_code_file`（第 44–51 行）

```cpp
44  inline filepath_t navigation_code_file(const filepath_t& prefix,
45                                         size_t node_ordinal,
46                                         size_t num_nodes,
47                                         u32 subquantizers) {
48    return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
49                      std::to_string(num_nodes) + ".pq" +
50                      std::to_string(subquantizers) + ".codes");
51  }
```

拼 PQ 编码分片文件：`{prefix}_node{ordinal}_of{num_nodes}.pq{subquantizers}.codes`。例如 `index_m64_efc200_node0_of4.pq32.codes`。这是按 shard 切分的 PQ 编码，存储节点持有自己的那份，GPU 通过 RDMA 读取，详见第 9/22 课。

#### `navigation_code_for_shard`（第 53–58 行）

```cpp
53  inline filepath_t navigation_code_for_shard(const filepath_t& shard_file_path,
54                                             u32 subquantizers) {
55    filepath_t result = shard_file_path;
56    result.replace_extension(".pq" + std::to_string(subquantizers) + ".codes");
57    return result;
58  }
```

从一个已知的 shard 文件路径推导其对应的 PQ 编码文件路径：把 `.dat` 扩展名替换成 `.pq{subquantizers}.codes`。例如 `..._node0_of4.dat` → `..._node0_of4.pq32.codes`。这是"从 shard 文件反推 PQ 编码文件"的便捷函数，避免调用方手动拼字符串。

> **设计观察**：`index_path.hh` 把所有路径约定集中在一个头文件里，让"文件命名规则"成为项目的单一事实来源。如果未来要改命名（例如加版本号），只改这一个文件。第 6/7/8/12/13/29 课都会反复用到这些函数。

---

## 2.8 `timing.hh`（辅助）：`Timing`/`Interval` 计时器

虽然不在本课主任务列表，但 `configuration.hh` 的 `enable_breakdown` 字段和第 30 课的 breakdown benchmark 都依赖它，所以简要介绍。

```cpp
// /home/xjs/experiment/dvstor/src/common/timing.hh
1   #pragma once
2
3   #include <library/types.hh>
4   #include <ostream>
5
6   #include "nlohmann/json.hh"
7
8   namespace timing {
9
10  class Timing {
11  public:
12    struct Interval {
13      str descriptor_;
14
15      clockid_t clock_id_;
16      timespec time_{};
17      timespec time_start_{};
18
19      explicit Interval(str&& descriptor);
20
21      void start();
22      void stop();
23      void clear();
24      void add(const s_ptr<Interval>& t2);
25
26      f64 get_ms() const;
27    };
28
29  public:
30    using IntervalPtr = s_ptr<Interval>;
31    using json = nlohmann::json;
32
33    IntervalPtr create_enroll(str&& descriptor);
34    static void start(const IntervalPtr& interval) { interval->start(); }
35    static void stop(const IntervalPtr& interval) { interval->stop(); }
36    static void clear(const IntervalPtr& interval) { interval->clear(); }
37
38    json to_json() const;
39    static f64 get_ms(const timespec t) { return t.tv_nsec / 1000000.0 + t.tv_sec * 1000.0; }  // NOLINT
40    friend std::ostream& operator<<(std::ostream& os, const Timing& timing);
41
42  private:
43    vec<IntervalPtr> intervals_;
44  };
45
46  timespec operator+(const timespec& ts1, const timespec& ts2);
47  timespec operator-(const timespec& ts1, const timespec& ts2);
48  std::ostream& operator<<(std::ostream& os, const timespec& ts);
49
50  nlohmann::json get_timestamp();
51
52  }  // namespace timing
```

要点：

- **`Interval`**：单个计时区间，持有描述符、时钟 ID、累计时间 `time_`、本次启动时间 `time_start_`。`start()`/`stop()` 配对使用，`add()` 合并另一个 Interval。
- **`Timing`**：管理多个 `Interval` 的集合。`create_enroll` 创建并注册一个新区间。
- **第 39 行 `get_ms`**：把 `timespec` 转成毫秒（`f64`）。
- **第 46–48 行**：`timespec` 的加减运算符和打印运算符。
- **第 50 行 `get_timestamp`**：返回当前时间的 JSON 表示，用于报告时间戳。

`enable_breakdown = true` 时（默认），查询引擎会在关键路径上创建 Interval 并 start/stop，最终通过 `to_json()` 输出 per-request 拆解数据，详见第 30 课。

---

## 2.9 配置如何流向计算节点与存储节点

下图展示 `IndexConfiguration` 的字段如何被两个角色消费：

```
                       CLI argv (用户传入)
                              │
                              ▼
                  IndexConfiguration(argc, argv)
                  ┌───────────────────────────┐
                  │ add_options()             │  注册选项
                  │ process_program_options() │  基类解析（含 --is-server/-i）
                  │ normalize_mode() x4       │  小写化模式串
                  │ validate()                │  项目校验
                  │ operator<<(cerr, *this)   │  打印摘要
                  └───────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
        is_initiator=true                is_server=true
        (计算节点)                       (存储节点)
              │                               │
              ▼                               ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ GPU 查询引擎消费:    │      │ 存储服务端消费:      │
   │  gpu_device          │      │  server_index_file   │
   │  gpu_query_slots     │      │  storage_id          │
   │  gpu_memory_*        │      │  mn_memory_gb        │
   │  gpu_resident_pq_*   │      │  storage_peers       │
   │  gpu_adjacency_cache │      │  storage_owner_*     │
   │  gpu_exact_cache     │      │  (coroutines/batch/  │
   │  gpu_bootstrap_*     │      │   rpc_*/reverse_*/   │
   │  gpu_graph_*         │      │   maintenance_*)     │
   │  gpu_traversal_*     │      │  beam_width_constr   │
   │  gpu_final_rerank_*  │      │  alpha               │
   │  gpu_max_expansions  │      │  R                   │
   │  gpu_entry_seed_*    │      │  storage_owner_      │
   │  gpu_delta_anchor_*  │      │   update_mode        │
   │  gpu_rdma_qps ───────────►  │  (exact/local_stitch)│
   │  gpu_persistent_*    │      │                      │
   │  enable_updates      │      │                      │
   │  update_visibility   │      │                      │
   │  delta_budget        │      │                      │
   │  gpu_delta_maint_*   │      │                      │
   │  enable_breakdown    │      │                      │
   └──────────────────────┘      └──────────────────────┘
              │                               │
              │   GPUNetIO RDMA (第22课)      │
              │ ─────────────────────────────►│
              │   peer RPC (第24课)           │
              │ ◄─────────────────────────────│
              │                               │
   共享字段（两端都用）:
     index_prefix, dim, k, max_vectors, vector_data_type,
     num_threads, seed, R, alpha, beam_width_construction
```

### 字段消费课表（按课号排序）

| 课号 | 课题 | 主要消费字段 |
|------|------|--------------|
| 第 3 课 | 并发原语与协程 | `num_threads`, `disable_thread_pinning` |
| 第 4/5 课 | RDMA 传输库 | 基类 `Configuration` 全部字段（`max_send_queue_wr` 等） |
| 第 6 课 | Vamana 图格式/anchor/idmap | `index_prefix`, `max_vectors`, `R`, `dim` |
| 第 7 课 | schema-15 索引格式 | `R`（u8 出度）, `dim`, `index_prefix` |
| 第 8 课 | 元数据/owner map/存储协议 | `index_prefix`, `storage_peers`, `storage_id` |
| 第 9 课 | GPU 类型/遥测/PQ 模型 | `gpu_resident_pq_budget_mb`, `gpu_bootstrap_*`, `vector_data_type` |
| 第 10 课 | delta/动态路由/预算 | `gpu_delta_anchor_probes`, `delta_budget_mb`, `update_visibility_us`, `gpu_delta_maintenance_period_ms`, `gpu_resident_pq_budget_mb` |
| 第 11 课 | 持久化引擎 PImpl/生命周期 | `gpu_device`, `gpu_query_slots`, `gpu_memory_*`, `enable_breakdown` |
| 第 12/13 课 | construction | `alpha`, `R`, `beam_width_construction`, `seed`, `dim`, `vector_data_type` |
| 第 14 课 | 查询执行/路由/完成 | `k`, `gpu_query_slots`, `enable_updates`, `update_visibility_us` |
| 第 15 课 | 增量发布 | `beam_width_construction`, `storage_owner_*` |
| 第 17 课 | kernel 启动器/上下文/device ring | `Parameters{num_threads, gpu_rdma_qps}`, `gpu_persistent_blocks_per_sm`, `gpu_device` |
| 第 18 课 | 候选评分 | `gpu_final_rerank_width`, `dim`, `vector_data_type` |
| 第 19 课 | RDMA cache | `gpu_adjacency_cache_*`, `gpu_exact_cache_*`, `gpu_graph_prefetch_depth`, `gpu_graph_cache_ttl_us` |
| 第 20 课 | 查询遍历主循环 | `gpu_traversal_beam_width`, `gpu_max_expansions`, `gpu_entry_seed_count`, `gpu_delta_anchor_probes`, `gpu_graph_prefetch_depth` |
| 第 21 课 | kernel 运行时/角色调度 | `gpu_persistent_blocks_per_sm`, `gpu_query_slots` |
| 第 22 课 | GPUNetIO 传输/probe | `gpu_rdma_qps`, `gpu_bootstrap_*`, `storage_peers`, `kMaxPeerQps` |
| 第 23 课 | 存储节点主体/peer RDMA | `server_index_file`, `storage_id`, `mn_memory_gb`, `storage_owner_peer_rdma_tokens`, `kMaxPeerQps`, `kCacheLineBytes` |
| 第 24 课 | peer RPC | `storage_owner_coroutines`, `storage_owner_batch_max`, `storage_owner_rpc_depth`, `storage_owner_rpc_timeout_ms` |
| 第 25 课 | 索引访问/图修改 | `storage_owner_update_mode`, `storage_owner_maintenance_*`, `storage_owner_reverse_*`, `storage_owner_search_snapshot_batch`, `alpha`, `R` |
| 第 27 课 | 计算服务主体 | `gpu_*`（全部）, `enable_updates`, `num_threads` |
| 第 28 课 | 计算侧 storage owner 更新 | `enable_updates`, `storage_owner_*`, `storage_peers` |
| 第 29 课 | 离线构建/迁移 | `index_prefix`, `alpha`, `R`, `beam_width_construction`, `seed`, `vector_data_type`, `index_path::*` |
| 第 30 课 | breakdown benchmark/实验脚本 | `enable_breakdown`, `timing::Timing` |

---

## 2.10 小结

本课讲解了 `src/common/` 下七个公共头文件，它们共同构成 dvstor 项目的"类型与配置地基"：

1. **`types.hh`**（23 行）：定义 `node_t`/`element_t`/`distance_t`/`filepath_t` 以及 `hashset_t`/`hashmap_t`/`dense_hashmap_t` 容器别名。极薄，只承担 dvstor 特有语义别名，基础整数类型复用 `library/types.hh`（第 4 课）。

2. **`constants.hh`**（9 行）：两个编译期硬约束——`kCacheLineBytes = 64`（缓存行）和 `kMaxPeerQps = 4`（peer RDMA QP 上限）。其余可调参数全部在 `IndexConfiguration`。

3. **`vector_dtype.hh`**（499 行）：最厚的文件。定义 `VectorDType` 枚举（float32/uint8/int8）、字符串解析（`parse_vector_dtype`，支持 `f32`/`u8`/`i8` 等多种风格）、文件扩展名推断（`.u8bin`/`.i8bin`/`.fbin`/`.bin`）、配置解析（`resolve_vector_dtype_config`，auto 模式从扩展名推断）、编解码（`encode_float_vector_to_storage`/`decode_storage_vector_to_float`）、字节数计算，以及四条 AVX2 加速 L2 距离路径（uint8/int8 同 dtype 对走整型累加，float 查询 vs byte 存储走 float 累加）。`integral_byte_l2_sum_exact_in_float` 提供"byte 维度 ≤ 258 时整型/float 累加位级别一致"的数值保证。

4. **`distance.hh`**（89 行）：旧版 float32 专用 L2，来自 hnswlib/flann，主要服务离线构建路径。新代码优先用 `vector_dtype.hh`。

5. **`configuration.hh`**（451 行）：`IndexConfiguration` 继承 `Configuration`（第 4 课），定义 5 大类约 40 个字段：索引基础（`index_prefix`/`dim`/`k`/`R`/`alpha`/`vector_data_type` 等）、GPU 查询引擎（`gpu_device`/`gpu_query_slots`/`gpu_memory_*`/`gpu_*_cache_*`/`gpu_traversal_*`/`gpu_persistent_*` 等）、动态更新（`enable_updates`/`update_visibility_us`/`delta_budget_mb`）、存储更新（`storage_owner_*`）、内存（`mn_memory_gb`）。`add_options()` 注册所有 CLI 选项，`validate()` 实施约 30 条边界校验（含 `R <= 255`、`k <= gpu_final_rerank_width`、`gpu_adjacency_cache_ways == 4`、`local_stitch` 与 `finalize` 必须同时启用等），`operator<<` 按角色打印摘要。

6. **`index_path.hh`**（61 行）：7 个内联函数拼装索引文件路径，统一管理 `{prefix}_node{o}_of{n}.dat`、`.idmap`、`.anchors`、`.pq{M}`、`.pq{M}.codes` 等命名约定。第 6/7/8/12/13/29 课反复使用。

7. **`timing.hh`**（53 行，辅助）：`Timing`/`Interval` 计时器，配合 `enable_breakdown` 收集 per-request 拆解数据（第 30 课）。

理解本课是阅读后续每一课的前提——后续每一课都会以"本模块消费了哪些 `IndexConfiguration` 字段"开篇。下一课（第 3 课）将讲解 dvstor 的并发原语与协程，主要消费 `num_threads` 和 `disable_thread_pinning`。
