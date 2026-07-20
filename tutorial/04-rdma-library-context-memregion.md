# 第 4 课：RDMA 传输库（上）—— Context / Configuration / MemoryRegion

> 本课是 30 课源码教程的第 4 课。我们开始进入 dvstor 系统的“传输层”。dvstor 是 GPU 中心化的存算分离向量检索系统：GPU 持久化引擎（第 17–22 课）需要直接发起 RDMA 读把存储节点上的向量/PQ 码拉到显存或主机内存；存储节点（第 23–26 课）也要通过 RDMA 把自己注册的内存暴露给对端。两边共用的 verbs 封装就是本课要讲的 `rdma-library`。
>
> 本课只覆盖“无 QP”的部分：库的定位、配置、Context（设备/PD/CQ 握手）、MemoryRegion（注册与 token 交换）、动态 MR 分配器、hugepage 辅助、以及通用的工具/线程原语。**QP 的创建、状态机、 verbs 投递**留到第 5 课；**GPU 直接发起的 GPUNetIO 传输**留到第 22 课。

---

## 4.1 本课目标与涉及文件

读完本课你应当能回答：

1. dvstor 计算节点和存储节点为什么共用同一个 verbs 封装？它和“原生 ibverbs”相比封装了什么？
2. 一个 `Configuration` 对象怎么从命令行被填满？为什么它有一个空的 `ib_device`？它和 `src/common/configuration.hh` 里的 `IndexConfiguration` 是什么继承关系？
3. `Context` 的构造函数里 `ibv_get_device_list → ibv_open_device → ibv_alloc_pd → ibv_query_port → ibv_create_cq → ibv_create_srq` 这条链路为什么不能乱序？析构为什么反向？
4. `MemoryRegion` 怎样把一段裸内存注册成 RNIC 可寻址的区域？`lkey`/`rkey`/`address` 三件套如何被打包成 `MemoryRegionToken` 用于远端交换？
5. `DynamicRegionAllocator` 如何按块切分出可复用的接收缓冲 MR？`HugePage` 怎么保证 RDMA 缓冲落在对齐的大页上？
6. 客户端/服务端怎么通过一条**临时 TCP socket** 交换 QP 编号和 LID，然后立刻把 TCP 关掉，只用 RDMA 通信？

涉及文件（全部在 `/home/xjs/experiment/dvstor/rdma-library/library/` 下）：

| 文件 | 角色 | 行数 |
|---|---|---|
| `types.hh` | 库内全局类型别名 + `span` + 并发容器别名 | 95 |
| `utils.hh` / `utils.cc` | `lib_assert`/`lib_debug` 宏、`Endpoint` 解析、吞吐/延迟统计、位运算工具 | 85 / 77 |
| `thread.hh` | `Thread` RAII 封装 + CPU 亲和绑定辅助 | 78 |
| `configuration.hh` / `configuration.cc` | `configuration::Configuration` 基类（program_options） | 49 / 149 |
| `context.hh` / `context.cc` | `Context`：device/PD/CQ/SRQ + TCP 握手 + CQ 轮询 | 98 / 338 |
| `memory_region.hh` / `memory_region.cc` | `MemoryRegion` / `LocalMemoryRegion` / `MemoryRegionToken` | 63 / 52 |
| `dynamic_region_allocator.hh` | 模板化的动态 MR 分配器 | 77 |
| `hugepage.hh` | 大页对齐 bump 分配器 | 138 |

辅助阅读：`rdma-library/README.md`（库定位）、`rdma-library/CMakeLists.txt`（依赖：ibverbs/Boost/TBB/Threads）、`src/common/configuration.hh`（`IndexConfiguration` 继承本库 `Configuration`，见第 2 课）、`src/common/types.hh`（继承本库 `types.hh`，加 `node_t`/`element_t`/`filepath_t`）。

---

## 4.2 库的整体定位

`rdma-library/README.md:1-7` 自我描述：

> "High-level" library to connect machines, connect queue pairs, register memory regions, post RDMA verbs, etc. The goal of this library is to conveniently wrap the [ibverbs library].

它是一个“比 libibverbs 高一层、比业务逻辑低一层”的静态库。`rdma-library/CMakeLists.txt:17-20` 把 `library/**.cc` 全部 glob 进一个静态库 `rdma_library`，对外暴露 `target_include_directories(rdma_library PUBLIC .)`，也就是说上层只需 `#include <library/context.hh>` 就能用。

依赖（CMake 第 11–14、20 行）：

- `IBVerbs`（libibverbs，提供 `ibv_*` 系列 C API）
- `Boost.program_options`（命令行解析）
- `TBB`（`concurrent_vector`）
- `Threads`（pthread，用于 `pthread_setaffinity_np`）
- 额外链接 `${NL_ROUTE_LIBRARY}` / `${NL_3_LIBRARY}`（这些在仓库根的 CMake 里给出，主要给 GPUNetIO 路径用，本课不展开）

dvstor 里有两类消费者：

1. **存储节点主体**（第 23 课 `storage_node`）：起一个 `Context`，调用 `bind_to_port` + `wait_for_connection` 接受计算节点的 TCP 握手，握手成功后 `post_shared_receive` 在 SRQ 上挂接收缓冲，靠 `MemoryRegion` 把图分片、PQ 码、向量注册成 RNIC 可寻址内存。
2. **计算节点的 RDMA cache / 查询遍历**（第 19、20 课）：用 `connect_to_server` 主动连存储节点，拿到对端 `MemoryRegionToken` 后用 RDMA READ 把图/向量拉到本地或 GPU 显存。
3. **GPU 持久化引擎的 GPUNetIO 路径**（第 22 课）走的是 DOCA GPUNetIO，不直接用本库的 QP，但**配置和 token 数据结构沿用了本课的 `Configuration` 与 `MemoryRegionToken`**。这就是为什么本课要先把这些底座讲透。

第 5 课会讲 `queue_pair.hh/cc`（QP 状态机 RTR/RTS、`ibv_post_send`/`ibv_post_recv`），它是本课 `Context::wait_for_connection` 里 `make_unique<QueuePair>(this)` 背后的对象。本课在调用点只点到为止。

---

## 4.3 `types.hh`：库的“公共别名字典”

`rdma-library/library/types.hh` 是整个 rdma-library 的类型基石。`src/common/types.hh:5` 直接 `#include <library/types.hh>` 把它继承下来，再追加业务别名。先看核心片段：

```cpp
// types.hh:16-33
using i8  = int8_t;   using u8  = uint8_t;
using i16 = int16_t;  using u16 = uint16_t;
using i32 = int32_t;  using u32 = uint32_t;
using i64 = int64_t;  using u64 = uint64_t;
using f32 = float;    using f64 = double;

using byte_t = uint8_t;
using str    = std::string;
using size_t = std::size_t;
using idx_t  = std::size_t;
using intptr_t = std::intptr_t;
```

这里没有用 `std::int32_t` 这种长名，而是统一短别名。注意：

- `byte_t` 是裸字节类型，`hugepage.hh`、`memory_region.cc` 都用它做指针运算。
- `str`/`vec`/`u_ptr`/`s_ptr`/`func` 是 STL 的短写（38–48 行），贯穿全库，例如 `vec<MRT>`（`memory_region.hh:17`）、`u_ptr<QueuePair>`（`context.hh:53`）、`func<void(u64)>`（`context.hh:72`）。
- `idx_t = std::size_t`：给“下标/索引”语义一个独立名字，避免和“字节大小”的 `size_t` 混用。

**并发容器别名**（87–92 行）：

```cpp
template <typename T> using concurrent_vec    = oneapi::tbb::concurrent_vector<T>;
template <typename T> using concurrent_queue  = moodycamel::ConcurrentQueue<T>;
// using concurrent_queue = oneapi::tbb::concurrent_queue<T>;
```

注意那行被注释掉的 tbb 版本——作者实测后改用了 moodycamel 的无锁队列。`DynamicRegionAllocator`（4.9 节）的 `free_list_` 就是 `concurrent_queue<u32>`；`region_buffers_`/`memory_regions_` 是 `concurrent_vec<u_ptr<...>>`，因为这两个表会在持有 `std::mutex` 的临界区里 `emplace_back`，但读者会并发按 `region_id` 下标读，`concurrent_vector` 的“元素一旦插入就地址稳定”保证正好满足这一点。

**`span` 轻量视图**（50–85 行）是一个手写的 C++17 `std::span` 替代（仓库并不要求 C++20）：

```cpp
template <typename T>
class span {
public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using pointer      = T*;
  using reference    = T&;
  using iterator     = pointer;
  constexpr span() noexcept = default;
  constexpr span(pointer data, size_t size) noexcept : data_(data), size_(size) {}
  template <typename Allocator, typename U = T,
            std::enable_if_t<std::is_convertible_v<value_type*, U*>, int> = 0>
  span(std::vector<value_type, Allocator>& values) noexcept
      : data_(values.data()), size_(values.size()) {}
  // ... const vector 版本、可转换 span 版本 ...
  constexpr pointer  data()  const noexcept { return data_; }
  constexpr size_t   size()  const noexcept { return size_; }
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end()   const noexcept { return data_ + size_; }
  constexpr reference operator[](size_t index) const noexcept { return data_[index]; }
private:
  pointer data_{};
  size_t  size_{};
};
```

`span` 提供“指针 + 长度”的不拥有视图，能在不拷贝的前提下把 `std::vector` 传给 C 接口。第 17 课 kernel 启动器、第 19 课 RDMA cache 都大量用它把 host 端的 `vec<...>` 喂给 verbs 或 CUDA。注意 73 行的转换构造允许 `span<const U>` 从 `span<U>` 隐式加 const——这是把“可写视图”降级成“只读视图”的标准技巧。

> **小坑**：`dynamic_region_allocator.hh:8` 有一行 `#include "span.hh"`，但仓库里**根本没有 `span.hh` 文件**（`find` 验证）。`span` 其实定义在本 `types.hh` 里。这是一处历史遗留 include；编译能过是因为 rdma-library 的 `target_include_directories` 在头文件搜索路径里恰好没找到 `span.hh` 就被忽略——`#include "span.hh"` 失败时（在 GCC/Clang 默认行为下）只有当文件确实不存在才会报错，但这里的真实情况是 `dynamic_region_allocator.hh` 同时 `#include "types.hh"`（第 9 行），所以 `span` 定义已经可见，多余 include 被预处理跳过。读者读到这一行不要被迷惑。

---

## 4.4 `utils.hh` / `utils.cc`：错误处理、端点解析、统计

### 4.4.1 `lib_assert` / `lib_debug` 宏

verbs 调用几乎每一条都可能失败，且失败语义是“返回非零 errno”，没有异常。库作者选择了一个简单粗暴的策略：**失败就打印并 `std::exit(EXIT_FAILURE)`**。

```cpp
// utils.hh:14-23
// why using macros? for std::string&& (rvalues) always _M_dispose() is called,
// even if the body is empty; using const char* also does not help because we
// cannot concatenate strings then
#define lib_assert(cond, msg)        \
  do {                               \
    if (!(cond)) {                   \
      std::cerr << msg << std::endl; \
      std::exit(EXIT_FAILURE);       \
    }                                \
  } while (0)
```

注释解释了为什么用宏而不用 `inline void lib_assert(bool, const str&)`：如果用函数，即便在非 debug 构建里，调用方也会构造 `str` 消息（可能拼接 `std::to_string(...)` 等），构造完再传给函数——字符串的 `_M_dispose()` 总会被调用，浪费性能。用宏则让 `msg` 只在 `cond` 为假时被求值。`lib_debug` 同理（25–34 行），仅在 `LIB_DEBUG` 宏定义时打印。

```cpp
// utils.cc:8-11
void lib_failure(const str&& message) {
  std::cerr << "[ERROR]: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}
```

`lib_failure` 是给“非断言型失败”（如 `poll_recv_cq` 返回负数）用的函数版本。注意参数是 `const str&&`——只接受右值，强制调用方写 `lib_failure("..." + x)` 这种临时值。

### 4.4.2 `Endpoint` 与 `parse_endpoint`

`utils.hh:39-45`：

```cpp
struct Endpoint {
  str host;
  str address;
  u32 port;
};
Endpoint parse_endpoint(const str& endpoint, u32 default_port);
```

`Endpoint` 把一个 `host:port` 字符串拆成三件：`host`（原始名，可能是 `cluster3` 这种别名）、`address`（解析后的 IPv4 字符串）、`port`。`utils.cc:26-56` 实现：

```cpp
Endpoint parse_endpoint(const str& endpoint, u32 default_port) {
  lib_assert(!endpoint.empty(), "Endpoint must not be empty");
  str host = endpoint;
  u32 port = default_port;

  const auto colon_pos = endpoint.find(':');
  if (colon_pos != str::npos) {
    host = endpoint.substr(0, colon_pos);
    const str port_str = endpoint.substr(colon_pos + 1);
    lib_assert(!host.empty(),    "Endpoint host must not be empty: " + endpoint);
    lib_assert(!port_str.empty(), "Endpoint port must not be empty: " + endpoint);
    lib_assert(std::all_of(port_str.begin(), port_str.end(),
                           [](unsigned char ch) { return std::isdigit(ch) != 0; }),
               "Endpoint port must be numeric: " + endpoint);
    try {
      const auto parsed_port = std::stoul(port_str);
      lib_assert(parsed_port <= 65535, "Endpoint port out of range: " + endpoint);
      port = static_cast<u32>(parsed_port);
    } catch (const std::exception&) {
      lib_assert(false, "Invalid endpoint port: " + endpoint);
    }
  }

  str address = host;
  if (std::count(host.begin(), host.end(), '.') != 3) {
    address = get_ip(host);   // 别名查表
  }
  return Endpoint{host, address, port};
}
```

逻辑：
1. 没有 `:` 就用 `default_port`，整个字符串当 host。
2. 有 `:` 就拆分，端口必须全数字且 ≤65535。
3. 如果 host 里 `.` 的数量不是 3（即不是 IPv4 形如 `192.168.6.202`），就当别名查 `get_ip` 表。

`get_ip`（`utils.cc:13-24`）是一张硬编码别名表：

```cpp
std::map<str, str> node_to_ip{
  {"cluster1", "127.0.0.1"},
  {"cluster2", "192.168.6.201"},
  {"cluster3", "192.168.6.202"},
};
```

这正是 memory 里 [dvstor-cluster-topology](dvstor-cluster-topology.md) 记录的拓扑：`.202` 是 RDMA 集群节点，`.201` 跑不起来。**这是库的硬编码假设**——如果你要换集群，要么改这张表，要么命令行直接传 `192.168.6.202:1234` 这种带点 IP。`parse_endpoint` 在 `IndexConfiguration`（第 2 课）解析 `--storage-peers` 时被调用，把 `cluster3 cluster4` 这种列表拆成可连的 `Endpoint`。

### 4.4.3 统计与位运算工具

```cpp
// utils.hh:47-55
f64 compute_throughput(i32 message_size, i32 repeats, Timepoint start, Timepoint end);
f64 compute_latency(i32 repeats, Timepoint start, Timepoint end,
                    bool is_read_or_atomic);
```

`utils.cc:58-72`：

```cpp
f64 compute_throughput(i32 message_size, i32 repeats, Timepoint start, Timepoint end) {
  return message_size / (ToSeconds(end - start).count() / repeats) / std::pow(1000, 2);
}
f64 compute_latency(i32 repeats, Timepoint start, Timepoint end, bool is_read_or_atomic) {
  i32 rtt_factor = is_read_or_atomic ? 1 : 2;
  return ToMicroSeconds(end - start).count() / repeats / rtt_factor;
}
```

- `compute_throughput`：吞吐 = 总字节 / 总秒数，除以 `1000^2` 把 B/s 换成 MB/s。
- `compute_latency`：单边延迟。注意 `rtt_factor`：RDMA READ/ATOMIC 是单边操作（发起方一次 verb 完成），算单程；SEND/RECV 是双边，要算往返，所以除 2。这两个函数主要给第 30 课 breakdown benchmark 用。

位运算工具（`utils.hh:70-77`）：

```cpp
inline u64 encode_64bit(u64 a, u64 b) { return (a << 32) | b; }
inline std::pair<u32, u32> decode_64bit(u64 word) {
  u32 a = word >> 32;
  u32 b = (word << 32) >> 32;
  return {a, b};
}
```

把两个 `u32` 打包进一个 `u64`，常用于塞进 verbs 的 `wr_id`（work request ID，完成时回传）。`context.cc:208` 的 `work_request.wr_id = reinterpret_cast<u64>(&region)` 是另一种用法——直接塞指针。

`punning`（65–68 行）是严格同大小的 type-pun，比 `memcpy` 快，用于把 `float` 当 `u32` 看等场景。`touch_memory`（58–62 行）逐字节写零，强迫物理页落实（page fault → 真正分配），避免 RDMA 首次访问时触发缺页延迟。

---

## 4.5 `thread.hh`：线程 RAII 与 CPU 亲和

`rdma-library/library/thread.hh` 提供两类东西：`Thread` 类和两个自由函数。

### 4.5.1 `Thread` 类

```cpp
// thread.hh:13-53
class Thread {
public:
  explicit Thread(u32 id) : thread_id_(id){};

  template <typename... Args>
  void start(Args&&... args) {
    t_ptr_ =
      std::make_unique<std::thread>(std::forward<Args>(args)..., thread_id_);
  }
  void join() const { if (t_ptr_) t_ptr_->join(); }
  void set_done() { done_ = true; }
  bool is_done() const { return done_; }
  u32  get_id() const { return thread_id_; }

#ifdef __unix__
  void set_affinity(u32 core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    lib_assert(pthread_setaffinity_np(
                 t_ptr_->native_handle(), sizeof(cpu_set_t), &cpuset) == 0,
               "cannot pin thread " + std::to_string(thread_id_));
  }
#else
  void set_affinity(u32) {}
#endif

private:
  const u32 thread_id_;
  u_ptr<std::thread> t_ptr_;
  std::atomic<bool> done_{false};
};
```

要点：

- `start` 模板把 `thread_id_` **追加**到用户传入的可调用对象参数列表末尾——即 worker 函数签名最后必须有一个 `u32 thread_id` 参数。这是库约定的 worker 协议。
- `done_` 用 `std::atomic<bool>` 而不是 `volatile bool`。注释（49–52 行）解释得很清楚：跨线程可见性必须靠原子（或 memory barrier），普通 `volatile` 在 x86 上“碰巧能工作”但不保证，且会被编译器/缓存优化掉。worker 在 spin loop 里查 `is_done()`，主控线程 `set_done()` 通知退出。
- `set_affinity` 用 `pthread_setaffinity_np` 把线程钉到指定核。dvstor 在第 21 课的 kernel 运行时角色调度、第 23 课存储节点 owner 协程里都依赖这个——RDMA 轮询线程必须独占核，否则 CQ poll 会被抢占导致延迟毛刺。

### 4.5.2 自由函数：钉主线程/任意线程

```cpp
// thread.hh:55-75
#ifdef __unix__
inline void pin_main_thread(u32 core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  lib_assert(sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0,
             "cannot pin main thread");
}
inline void pin_thread(std::thread& thread, u32 core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  lib_assert(pthread_setaffinity_np(
               thread.native_handle(), sizeof(cpu_set_t), &cpuset) == 0,
             "cannot pin worker thread");
}
#else
inline void pin_main_thread(u32) {}
inline void pin_thread(std::thread&, u32) {}
#endif
```

`pin_main_thread` 用 `sched_setaffinity(0, ...)`（`0` 表示当前线程），`pin_thread` 用 `pthread_setaffinity_np`。两者都是 `inline` 头文件函数，避免给一个简单的系统调用再开一个 `.cc`。

---

## 4.6 `configuration.hh` / `configuration.cc`：命令行配置基类

`Configuration` 是 dvstor 整个配置体系的根。`src/common/configuration.hh:22` 的 `IndexConfiguration` 公开继承它（`class IndexConfiguration : public Configuration`），并在 `add_options()` 里继续往 `desc` 里 `add_options()`——这是 boost::program_options 的标准扩展手法：基类先注册 RDMA 选项，子类再追加业务选项。

### 4.6.1 字段定义

```cpp
// configuration.hh:12-27
namespace configuration {
namespace po = boost::program_options;

class Configuration {
public:
  i32 max_send_queue_wr{1024};   // send CQ 容量
  i32 max_recv_queue_wr{1024};   // recv CQ 容量
  i32 max_poll_cqes{16};         // 一次 ibv_poll_cq 最多取多少个 WC
  u32 port{1234};                // TCP 握手端口
  str ib_device;                 // 空 = 自动选第一个设备
  u32 device_port{1};            // IB 物理端口（1 或 2）
  bool is_server{false};         // 服务端 flag
  vec<str> server_nodes;         // 客户端要连的服务端列表
  vec<str> client_nodes;         // initiator 要反向通知的客户端列表
  u32 num_clients{1};            // 每个服务端期望接的客户端数
  bool is_initiator{false};      // 发起方 flag

protected:
  po::options_description desc{"Allowed options"};
  ...
};
```

关键字段语义：

- `max_send_queue_wr` / `max_recv_queue_wr`：这两个值会传给 `ibv_create_cq`（`context.cc:68-70`）和 SRQ（`context.cc:77`）。它们定义了“在途 WR 数上限”——超过这个数的 `ibv_post_send` 会阻塞或失败。dvstor 存储节点默认 1024，计算节点 GPU 路径会用更大的值（见第 22 课）。
- `max_poll_cqes`：`ibv_poll_cq` 的批量大小。第 30 课 breakdown benchmark 会调它影响统计粒度。
- `ib_device`：空字符串是合法值，`Context` 构造时会 fallback 到 `device_idx`（默认 0）。这很重要：在多 RNIC 机器上，dvstor 用 `--ib-device mlx5_0` 显式指定；单卡机器可以不传。
- `device_port`：IB 卡通常有两个物理端口（port 1/2），对应两条链路。dvstor 在 `.202` 上固定用 port 1。
- `is_server` / `is_initiator`：互斥的角色 flag。`is_server` 表示“我是存储节点，被动 accept”；`is_initiator` 表示“我是发起方客户端，主动 connect 并可能反向连 client_nodes”。两者都不是时就是普通客户端。
- `server_nodes` / `client_nodes`：字符串列表，可以是 `cluster3` 或 `192.168.6.202:1235`，由 `parse_endpoint` 拆解。

### 4.6.2 选项注册

```cpp
// configuration.cc:16-54
void Configuration::create_rdma_options() {
  desc.add_options()
    ("help,h", "Show help message")
    ("is-server,s",
     po::bool_switch(&is_server)->default_value(is_server),
     "Program acts as server if set")
    ("servers",
     po::value<vec<str>>(&server_nodes)->multitoken(),
     "A list of server endpoints to which a client connects, e.g., \"cluster3\" or \"127.0.0.1:1235\"")
    ("clients",
     po::value<vec<str>>(&client_nodes)->multitoken(),
     "A list of client endpoints to which the initiator connects, e.g., "
     "\"cluster4 cluster5\" or \"127.0.0.1:2234 127.0.0.1:2235\"")
    ("initiator,i",
     po::bool_switch(&is_initiator)->default_value(is_initiator),
     "Program acts as initiating client if set")
    ("num-clients,c",
     po::value<u32>(&num_clients)->default_value(num_clients),
     "Number of clients that connect to each server (relevant only for server nodes)");

  // configuration options
  desc.add_options()
    ("port", po::value<u32>(&port)->default_value(port), "TCP port")
    ("ib-device",
     po::value<str>(&ib_device),
     "InfiniBand/RDMA device name, for example mlx5_0. Empty selects the first device.")
    ("ib-port",
     po::value<u32>(&device_port)->default_value(device_port),
     "Port of infiniband device")
    ("max-poll-cqes",
     po::value<i32>(&max_poll_cqes)->default_value(max_poll_cqes),
     "Number of outstanding RDMA operations allowed (hardware-specific)")
    ("max-send-wrs",
     po::value<i32>(&max_send_queue_wr)->default_value(max_send_queue_wr),
     "Maximum number of outstanding send work requests")
    ("max-receive-wrs",
     po::value<i32>(&max_recv_queue_wr)->default_value(max_recv_queue_wr),
     "Maximum number of outstanding receive work requests");
}
```

`po::bool_switch(&is_server)` 是 boost 的“开关选项”绑定——`--is-server` 出现就置 true，不出现保持默认。`po::value<vec<str>>(&server_nodes)->multitoken()` 接受多个 token（空格分隔），让 `--servers cluster3 cluster4` 一次填两个端点。`po::value<str>(&ib_device)` 没有默认值——不传时 `ib_device` 保持构造时初始化的空字符串，这正是“自动选设备”的信号。

### 4.6.3 构造与解析

```cpp
// configuration.cc:9-14
Configuration::Configuration() { create_rdma_options(); }
Configuration::Configuration(int argc, char** argv) : Configuration() {
  process_program_options(argc, argv);
  operator<<(std::cerr, *this);
}
```

委托构造：先调无参构造把所有选项注册进 `desc`，再 `process_program_options` 解析命令行，最后把配置 dump 到 stderr。`src/common/configuration.hh:86-94` 的 `IndexConfiguration` 构造函数复用这个模式：先 `add_options()`（追加业务选项），再 `process_program_options(argc, argv)`，再 `validate(argv)`，再 dump。

```cpp
// configuration.cc:61-96
void Configuration::process_program_options(int argc, char** argv) {
  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cerr << desc << std::endl;
      std::exit(EXIT_FAILURE);
    }
    po::notify(vm);

    if (!is_server && server_nodes.empty()) {
      std::cerr << "[ERROR]: --servers <arg-list> must be given if "
                   "--is-server is not set" << std::endl;
      exit_with_help_message(argv);
    }
    if (is_server && is_initiator) {
      std::cerr << "[ERROR]: a server cannot be the initiator" << std::endl;
      exit_with_help_message(argv);
    }
    if (!is_initiator && !client_nodes.empty()) {
      std::cerr << "[ERROR]: --clients <arg-list> is only required by the "
                   "initiating client" << std::endl;
      exit_with_help_message(argv);
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    exit_with_help_message(argv);
  }
}
```

三道语义校验：
1. **非服务端必须给 `--servers`**：客户端没目标地址没法连。
2. **服务端不能同时是 initiator**：角色互斥。
3. **`--clients` 只有 initiator 能用**：避免普通客户端误传。

`po::notify(vm)` 触发 `notifier`，把解析结果写回绑定的成员变量。`po::store` 只是填 `vm`，`notify` 才落地——这是 boost 的两阶段设计，让你在 `notify` 前有机会检查 `vm.count("help")` 并提前退出。

### 4.6.4 配置打印

`configuration.cc:98-146` 的 `operator<<` 是个格式化 dump：开头打一行 `=` 填充的横线，中间显示角色（SERVER/CLIENT）、连接目标、TCP 端口、IB 设备/端口、CQ 容量等。`src/common/configuration.hh:377` 的 `IndexConfiguration::operator<<` 第一行就是 `output << static_cast<const Configuration&>(config);`——先 dump 基类，再追加业务字段。这是经典的“基类打印 + 子类增量打印”模式。

### 4.6.5 与 `IndexConfiguration` 的关系

`src/common/configuration.hh:22-94` 的 `IndexConfiguration` 公开继承 `Configuration`，它：

- 复用 `desc`、`process_program_options`、`exit_with_help_message`、`operator<<`。
- 追加约 40 个业务选项（`--index-prefix`、`--dim`、`--gpu-device`、`--storage-peers` 等）。
- 在 `validate(argv)` 里加了大量业务校验，例如 `storage_peers.size() != num_server_nodes()` 必须匹配（323 行）——`num_server_nodes()` 就是 `Configuration::num_server_nodes()`（`configuration.hh:33`，返回 `server_nodes.size()`）。

这意味着任何 dvstor 进程的命令行都是**先解析 RDMA 通用选项，再解析业务选项**。这就是为什么本课要把 `Configuration` 讲透——它定义了所有进程共用的“RDMA 启动词典”。详见第 2 课。

---

## 4.7 `context.hh` / `context.cc`：verbs 上下文与 TCP 握手

`Context` 是 rdma-library 的“工厂对象”：它持有 device、PD、CQ、SRQ，提供 QP 工厂（`wait_for_connection`/`connect_to_server`）和 CQ 轮询接口。一个进程通常只有一个 `Context`。

### 4.7.1 类声明

```cpp
// context.hh:18-21
struct ReceiveInfo {
  MemoryRegion* mr{nullptr};
  u32 bytes_written{};
};
```

`ReceiveInfo` 是接收完成时返回给上层的“收据”：哪个 MR 收到了、收了多少字节。`mr` 是从 `wr_id` 反解出来的（见 4.7.6 节）。

```cpp
// context.hh:23-49
class Context {
public:
  using IBDeviceList = ibv_device**;
  using Configuration = configuration::Configuration;

public:
  explicit Context(Configuration& config,
                   i32 device_idx = 0,
                   bool create_shared_rcq = false);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ibv_context* get_raw_context()      { return context_; }
  ibv_pd*      get_protection_domain(){ return protection_domain_; }
  ibv_cq*      get_send_cq()          { return send_cq_; }
  ibv_cq*      get_receive_cq()       { return receive_cq_; }
  ibv_srq*     get_shared_receive_cq(){ return shared_receive_cq_; }
  Configuration& get_config() const   { return config_; }
  u16 get_lid() const                 { return port_attributes_.lid; }
  u32 max_qp_read_atomic() const {
    return std::max<u32>(1, std::min<u32>(16, device_attributes_.max_qp_init_rd_atom));
  }
  u32 max_qp_dest_read_atomic() const {
    return std::max<u32>(1, std::min<u32>(16, device_attributes_.max_qp_rd_atom));
  }
  ...
};
```

设计要点：

- **禁拷贝**（34–35 行）：verbs 对象是资源句柄，拷贝会导致双重释放。
- **`device_idx` 参数**：当 `config.ib_device` 为空时按索引选设备，默认 0。
- **`create_shared_rcq` 参数**：是否创建 SRQ（Shared Receive Queue）。SRQ 让多个 QP 共享一个接收缓冲池，dvstor 存储节点用它服务多客户端（见第 23 课）。
- **`max_qp_read_atomic` / `max_qp_dest_read_atomic`**：QP 发起/目标端的 read/atomic in-flight 上限，钳制在 `[1, 16]`。这两个值会在第 5 课 QP 创建时传给 `ibv_modify_qp` 的 `max_dest_rd_atomic` / `max_rd_atomic` 字段。
- **`get_lid`**：返回 IB 端口的 LID（Local Identifier）。在 IB 网络里，QP 寻址靠 `(lid, qp_number, psn)` 三元组；RoCE 网络用 GID，但本库走的是 IB LID 路径。

### 4.7.2 构造：device → context → PD → port → CQ → SRQ

```cpp
// context.cc:25-86
Context::Context(Configuration& config,
                 const i32 device_idx,
                 bool create_shared_rcq)
    : config_(config) {
  i32 num_devices = 0;
  IBDeviceList device_list = ibv_get_device_list(&num_devices);

  lib_assert(num_devices > 0, "No InfiniBand devices found");
  lib_assert(device_list != nullptr, "Device list is null");
  if (!config_.ib_device.empty()) {
    for (i32 i = 0; i < num_devices; ++i) {
      if (config_.ib_device == ibv_get_device_name(device_list[i])) {
        device_ = device_list[i];
        break;
      }
    }
    lib_assert(device_ != nullptr, "RDMA device " + config_.ib_device + " not found");
  } else {
    lib_assert(0 <= device_idx && device_idx < num_devices,
               "Device " + std::to_string(device_idx) + " not found");
    device_ = device_list[device_idx];
  }

  std::cerr << num_devices << " device(s) found" << std::endl;
  std::cerr << "Selected device: " << ibv_get_device_name(device_) << std::endl;

  context_ = ibv_open_device(device_);
  lib_assert(device_ && context_, "Cannot open device");
  lib_assert(ibv_query_device(context_, &device_attributes_) == 0,
             "Cannot query RDMA device capabilities");

  // allocate protection domain
  protection_domain_ = ibv_alloc_pd(context_);

  // query port
  lib_assert(
    ibv_query_port(context_, config_.device_port, &port_attributes_) == 0,
    "Cannot query port " + std::to_string(config_.device_port));
  std::cerr << "Selected port state: " << port_attributes_.state
            << ", lid: " << port_attributes_.lid << std::endl;

  // create completion queues
  send_cq_ =
    ibv_create_cq(context_, config_.max_send_queue_wr, nullptr, nullptr, 0);
  receive_cq_ =
    ibv_create_cq(context_, config_.max_recv_queue_wr, nullptr, nullptr, 0);

  lib_assert(send_cq_ && receive_cq_, "Cannot create completion queues");

  if (create_shared_rcq) {
    ibv_srq_init_attr attributes{};
    attributes.srq_context = context_;
    attributes.attr.max_wr = config_.max_recv_queue_wr;
    attributes.attr.max_sge = 1;
    shared_receive_cq_ = ibv_create_srq(protection_domain_, &attributes);
    lib_assert(shared_receive_cq_,
               "Cannot create shared receive completion queue");
  }

  ibv_free_device_list(device_list);
}
```

逐段：

**设备选择**（29–46 行）。`ibv_get_device_list` 返回系统所有 RNIC 的列表。如果有 `--ib-device`，按名字匹配；否则按 `device_idx` 取。注意循环里 `break` 后 `device_` 指向设备——但 **device_list 持有所有权**，所以最后 85 行必须 `ibv_free_device_list` 释放列表（但 `device_` 指向的 `ibv_device` 在 `ibv_open_device` 之后已经不再需要，释放列表是安全的）。

**打开设备并查能力**（51–54 行）。`ibv_open_device` 创建 `ibv_context`——这是 verbs 的“连接点”，后续所有 `ibv_*` 调用都要它。`ibv_query_device` 填充 `device_attributes_`（`ibv_device_attr`），里面包含 `max_qp`、`max_cq`、`max_qp_init_rd_atom` 等硬件上限。`Context::max_qp_read_atomic()` 就是基于这里的 `max_qp_init_rd_atom` 算的。

**分配 PD**（57 行）。Protection Domain 是 MR/QP 的命名空间。同一个 PD 下的 MR 才能被同一个 PD 下的 QP 访问。dvstor 单进程单 PD，简单粗暴。

**查询端口**（60–64 行）。`ibv_query_port` 填 `port_attributes_`（`ibv_port_attr`），关键字段：
- `state`：端口状态（`IBV_PORT_ACTIVE` 才能用）。
- `lid`：Local Identifier，IB 寻址用。这个值会被 `get_lid()` 返回，进而塞进 `QPInfo` 用于握手交换。
- `active_width` / `active_speed`：链路速率，第 30 课 benchmark 用。

**创建 CQ**（67–72 行）。`ibv_create_cq(context, cqe, cq_context, comp_channel, comp_vector)`：`cqe` 是 CQ 容量（来自 `max_send_queue_wr`/`max_recv_queue_wr`），后三个 `nullptr/0` 表示不用事件通道（dvstor 用轮询，不用事件）。send/recv 分两个 CQ 是惯例——分开轮询避免互相干扰。

**可选 SRQ**（74–83 行）。`ibv_srq_init_attr` 里 `max_wr` 是 SRQ 容量，`max_sge = 1` 表示每个接收 WR 只用 1 个 scatter-gather entry（即接收连续一段内存，不跨页）。SRQ 必须在 PD 下创建。

**释放设备列表**（85 行）。这一步必须在 `ibv_open_device` 之后——打开后 `context_` 持有自己的引用，列表可以释放。

### 4.7.3 析构：严格反向

```cpp
// context.cc:88-100
Context::~Context() {
  lib_assert(!shared_receive_cq_ || ibv_destroy_srq(shared_receive_cq_) == 0,
             "Cannot destroy shared receive completion queue");
  lib_assert(ibv_destroy_cq(receive_cq_) == 0,
             "Cannot destroy receive completion queue");
  lib_assert(ibv_destroy_cq(send_cq_) == 0,
             "Cannot destroy send completion queue");
  lib_assert(ibv_dealloc_pd(protection_domain_) == 0,
             "Cannot deallocate protection domain");
  lib_assert(ibv_close_device(context_) == 0, "Cannot close device.");

  close_server_socket();
}
```

顺序：SRQ → recv CQ → send CQ → PD → device。**严格反向**，因为 verbs 要求：QP/MR 必须在 PD 销毁前销毁（QP 在第 5 课 `QueuePair` 析构里销毁，MR 在 `MemoryRegion` 析构里销毁），CQ 必须在 PD 销毁前销毁，SRQ 必须在 PD 销毁前销毁。如果顺序错，`ibv_dealloc_pd` 会返回非零（PD 非空），`lib_assert` 直接 abort。

注意 `lib_assert` 用在析构里其实有风险——析构里 `std::exit` 会跳过其他析构。但库作者选择“宁可硬失败也不静默泄漏”，符合 verbs 资源管理的严格性。

`close_server_socket`（99 行）兜底关 TCP 监听 socket（如果 `bind_to_port` 被调过）。

### 4.7.4 TCP 握手：服务端

```cpp
// context.cc:102-124
void Context::bind_to_port(u32 tcp_port) {
  server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  lib_assert(server_socket_ >= 0, "Cannot open socket.");

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(tcp_port);

  i32 option_val = 1;
  lib_assert(setsockopt(server_socket_,
                        SOL_SOCKET,
                        SO_REUSEADDR,
                        &option_val,
                        sizeof(option_val)) == 0,
             "Cannot set socket option to reuse address");

  lib_assert(
    bind(server_socket_, (sockaddr*)&address, sizeof(sockaddr_in)) == 0,
    "Cannot bind to port " + std::to_string(tcp_port));
  lib_assert(listen(server_socket_, 128) == 0, "Cannot listen on socket");
}
```

`bind_to_port` 创建一个 IPv4 TCP 监听 socket。`sin_addr` 没填，等价于 `INADDR_ANY`——监听所有网卡。`SO_REUSEADDR` 让服务端重启时能立刻复用端口（TIME_WAIT 状态下也能 bind）。`listen(..., 128)` 设 backlog 为 128。

```cpp
// context.cc:132-158
std::pair<QP, u32> Context::wait_for_connection() {
  QP queue_pair = std::make_unique<QueuePair>(this);

  QPInfo receive_buffer{}, send_buffer{get_lid(), queue_pair->get_qp_num()};
  ssize_t qp_size = sizeof(QPInfo);

  i32 tcp_socket = accept(server_socket_, nullptr, nullptr);
  lib_assert(tcp_socket >= 0, "Cannot open socket.");

  lib_debug("Exchange QP information with client");
  lib_assert(recv(tcp_socket, &receive_buffer, qp_size, 0) == qp_size,
             "Received an incorrect number of bytes");
  lib_assert(send(tcp_socket, &send_buffer, qp_size, 0) == qp_size,
             "Transmitted an incorrect number of bytes");

  std::cerr << "pairing: " << queue_pair->get_qp_num() << " -- "
            << receive_buffer.qp_number << std::endl;

  queue_pair->transition_to_rtr(receive_buffer);
  queue_pair->transition_to_rts();

  // TODO: set remote user data

  close(tcp_socket);

  return {std::move(queue_pair), receive_buffer.node_id};
}
```

握手协议：

1. **先建 QP**（133 行）。`QueuePair` 构造时会调 `ibv_create_qp`，此时 QP 处于 RESET 状态。QP 的 `qp_num`（QP 编号）是硬件分配的，立即可用。
2. **准备交换报文**（135 行）。`QPInfo`（第 5 课讲）含 `lid`、`qp_number`、`node_id`。服务端把自己的 `lid`（`get_lid()`）和新建 QP 的 `qp_num` 填进 `send_buffer`。
3. **accept TCP**（138 行）。阻塞等客户端连进来。
4. **先 recv 后 send**（142–145 行）。注意顺序：服务端先收客户端的 `QPInfo`，再发自己的。客户端那边顺序相反（见 4.7.5）。这样双方都不会因为 send/recv 顺序死锁——其实是经典的“双向交换”，任一顺序都行，只要双方互补。
5. **迁移 QP 状态**（150–151 行）。`RESET → RTR (Ready to Receive) → RTS (Ready to Send)`。RTR 需要对端的 `lid` + `qp_num`（在 `receive_buffer` 里），RTS 不需要远端信息。这两个迁移在第 5 课细讲。
6. **关 TCP**（155 行）。握手完成，TCP 不再需要——之后所有通信走 RDMA。
7. **返回 QP 和对端 node_id**（157 行）。`node_id` 是客户端在 `QPInfo` 里自报的，服务端用它区分多客户端（第 23 课存储节点用 `node_id` 路由）。

### 4.7.5 TCP 握手：客户端

```cpp
// context.cc:160-195
QP Context::connect_to_server(const str& address, u32 tcp_port, u32 node_id) {
  QP queue_pair = std::make_unique<QueuePair>(this);

  QPInfo send_buffer{get_lid(), queue_pair->get_qp_num(), node_id},
    receive_buffer{};
  ssize_t qp_size = sizeof(QPInfo);

  sockaddr_in remote_address{};
  remote_address.sin_family = AF_INET;
  remote_address.sin_port = htons(tcp_port);
  inet_pton(AF_INET, address.c_str(), &(remote_address.sin_addr));

  i32 tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
  lib_assert(tcp_socket >= 0, "Cannot open socket.");

  lib_debug("Connect to server with address " + address);
  while (connect(tcp_socket, (sockaddr*)&remote_address, sizeof(sockaddr_in)) != 0) {
    // wait until server opens a connection
  }

  lib_debug("Exchange QP information with server");
  lib_assert(send(tcp_socket, &send_buffer, qp_size, 0) == qp_size,
             "Transmitted an incorrect number of bytes");
  lib_assert(recv(tcp_socket, &receive_buffer, qp_size, 0) == qp_size,
             "Received an incorrect number of bytes");

  std::cerr << "pairing: " << queue_pair->get_qp_num() << " -- "
            << receive_buffer.qp_number << std::endl;

  queue_pair->transition_to_rtr(receive_buffer);
  queue_pair->transition_to_rts();
  close(tcp_socket);

  return queue_pair;
}
```

和服务端镜像：

- `inet_pton(AF_INET, address.c_str(), ...)` 把点分 IP 字符串转成 `in_addr`。注意这里 `address` 必须是 IP（`parse_endpoint` 已经把 `cluster3` 解析成 `192.168.6.202`）。
- `connect` 在 `while` 里重试——服务端可能还没 `accept`，`connect` 失败立刻重试。这是粗暴的忙等，但握手只在启动时发生一次，可以接受。
- **先 send 后 recv**（182–185 行），与服务端顺序互补。
- 同样 `transition_to_rtr(receive_buffer)` + `transition_to_rts()`，关 TCP，返回 QP。

这就是 dvstor 的“TCP 握手 + RDMA 数据”双通道设计：**TCP 只用来交换 24 字节的 QPInfo，一旦 QP 进 RTS 态就完全切到 RDMA**。这种设计避免了在 RDMA 上实现连接建立的复杂度（verbs 本身没有连接建立原语，QP 状态迁移需要的对端信息必须带外交换）。

### 4.7.6 SRQ 接收投递

```cpp
// context.cc:197-217
void Context::post_shared_receive(MemoryRegion& region) {
  ibv_recv_wr work_request{};
  ibv_sge scatter_gather_entry{};
  ibv_recv_wr* bad_work_request{nullptr};

  lib_assert(shared_receive_cq_, "No shared receive CQ exists");

  scatter_gather_entry.addr = region.get_address();
  scatter_gather_entry.length = region.get_size_in_bytes();
  scatter_gather_entry.lkey = region.get_lkey();

  work_request.wr_id = reinterpret_cast<u64>(&region);
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  lib_assert(ibv_post_srq_recv(
               get_shared_receive_cq(), &work_request, &bad_work_request) == 0,
             "Cannot post shared receive request");
  lib_debug("Shared receive request successfully posted");
}
```

这是 SRQ 模式的接收投递。`ibv_recv_wr` 是接收工作请求，`ibv_sge` 是 scatter-gather entry（描述接收数据写到哪段内存）：

- `addr` / `length` / `lkey` 来自 `MemoryRegion`——RNIC 拿到数据后用 `lkey` 验证“这段内存确实被注册过”，然后 DMA 写入。
- `wr_id = &region`：把 MR 指针塞进 `wr_id`。完成时 `ibv_poll_cq` 会回传这个 `wr_id`，`poll_recv_cq` 里 `reinterpret_cast<MemoryRegion*>(wr_id)` 反解出 MR，填进 `ReceiveInfo.mr`（见 4.7.7）。这是“零拷贝元数据”技巧——不用查表，直接拿到接收缓冲对象。
- `num_sge = 1`：单个 SGE，对应 SRQ 创建时的 `max_sge = 1`。
- `ibv_post_srq_recv` 投递到 SRQ 而不是某个 QP 的 RQ——SRQ 上的缓冲可以被任意 QP 的接收填充。

### 4.7.7 CQ 轮询：接收

```cpp
// context.cc:220-246  (静态版本)
i32 Context::poll_recv_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* recv_cq,
                          ReceiveInfo* recv_info) {
  // caution: work_completion and recv_info must be arrays of size max_cqes
  i32 num_entries = ibv_poll_cq(recv_cq, max_cqes, work_completion);

  if (num_entries > 0) {
    for (i32 i = 0; i < num_entries; ++i) {
      lib_assert(work_completion[i].status == IBV_WC_SUCCESS,
                 describe_wc_failure("Receive", work_completion[i]));
      lib_debug("Receive request completed");

      if (recv_info && work_completion[i].opcode == IBV_WC_RECV) {
        recv_info[i].mr =
          reinterpret_cast<MemoryRegion*>(work_completion[i].wr_id);
        recv_info[i].bytes_written = work_completion[i].byte_len;
      }
    }
  } else if (num_entries < 0) {
    lib_failure("Cannot poll receive completion queue");
  }

  return num_entries;
}
```

`ibv_poll_cq(cq, max_cqes, wc)` 从 CQ 取最多 `max_cqes` 个完成项，返回实际取到的数量：

- `>0`：成功取到若干个。逐个检查 `status`，非 `IBV_WC_SUCCESS` 就 abort（`describe_wc_failure` 在 15–21 行格式化错误信息，含 status/opcode/vendor_err/wr_id，便于调试）。
- `=0`：CQ 空，没有完成。上层会 spin。
- `<0`：`lib_failure` 直接退出——`ibv_poll_cq` 返回负数表示系统错误。

对于 `IBV_WC_RECV` 完成项，从 `wr_id` 反解 MR 指针，把实际收到的字节数 `byte_len` 填进 `ReceiveInfo.bytes_written`。这就是 4.7.6 里 `wr_id = &region` 的回收点。

```cpp
// context.cc:248-255
i32 Context::poll_recv_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ReceiveInfo* recv_info) {
  lib_assert(max_cqes <= config_.max_recv_queue_wr,
             "expected number of WCs exceeds number of max WRs");
  return poll_recv_cq(work_completion, max_cqes, receive_cq_, recv_info);
}
```

成员版本封装静态版本，多一个“别超过 CQ 容量”的断言，并隐式用 `receive_cq_`。

```cpp
// context.cc:257-267
ReceiveInfo Context::receive() {
  ibv_wc work_completion{};
  ReceiveInfo recv_info{};
  i32 num_entries;
  do {
    num_entries = poll_recv_cq(&work_completion, 1, &recv_info);
  } while (num_entries == 0);
  return recv_info;
}

// context.cc:269-277
void Context::receive(i32 n) {
  vec<ibv_wc> work_completions(n);
  i32 num_entries = 0;
  do {
    num_entries += poll_recv_cq(work_completions.data(), n);
  } while (num_entries < n);
}
```

两个阻塞接收封装：`receive()` 收一个，`receive(n)` 收 n 个。都是 spin 直到凑够。第 23 课存储节点会在 owner 协程里用这个等对端的 mutation RPC。

### 4.7.8 CQ 轮询：发送

```cpp
// context.cc:280-302
i32 Context::poll_send_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* send_cq,
                          const func<void(u64)>& id_handler) {
  i32 num_entries = ibv_poll_cq(send_cq, max_cqes, work_completion);
  if (num_entries > 0) {
    for (i32 i = 0; i < num_entries; ++i) {
      lib_assert(work_completion[i].status == IBV_WC_SUCCESS,
                 describe_wc_failure("Send", work_completion[i]));
      id_handler(work_completion[i].wr_id);
    }
    lib_debug("Send request completed");
  } else if (num_entries < 0) {
    lib_failure("Cannot poll completion queue");
  }
  return num_entries;
}

i32 Context::poll_send_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* send_cq) {
  return poll_send_cq(work_completion, max_cqes, send_cq, [](u64) {});
}
```

发送完成多了一个 `id_handler` 回调——上层可以注册“这个 WR 完成了，回收它关联的资源”逻辑（例如回收 RDMA cache 槽位，见第 19 课）。无 handler 版本传一个空 lambda。

```cpp
// context.cc:311-337
i32 Context::poll_send_cq(ibv_wc* work_completion, const i32 max_cqes) {
  lib_assert(max_cqes <= config_.max_send_queue_wr,
             "expected number of WCs exceeds number of max WRs");
  return poll_send_cq(work_completion, max_cqes, send_cq_);
}

i32 Context::poll_send_cq_until_completion() {
  ibv_wc work_completion{};
  i32 num_entries;
  do {
    num_entries = poll_send_cq(&work_completion, 1);
  } while (num_entries == 0);
  return num_entries;
}

void Context::poll_send_cq_until_completion(i32 n) {
  vec<ibv_wc> work_completions(n);
  i32 num_entries = 0;
  do {
    num_entries += poll_send_cq(work_completions.data(), n);
  } while (num_entries < n);
}
```

`poll_send_cq_until_completion` 是 spin 直到至少一个发送完成；带 `n` 参数的版本等 n 个。第 19 课 RDMA cache 用它确认 READ 完成后再读 buffer。

---

## 4.8 `memory_region.hh` / `memory_region.cc`：内存注册与 token

### 4.8.1 `MemoryRegionToken`：远端访问三件套

```cpp
// memory_region.hh:9-17
struct MemoryRegionToken {
  u64 address;
  u32 lkey;
  u32 rkey;
};

// must be on the heap s.t. the address does not change after vector movements
using MRT = u_ptr<MemoryRegionToken>;
using MemoryRegionTokens = vec<MRT>;
```

`MemoryRegionToken` 是 MR 的“对外名片”——通过 TCP 或 RPC 发给对端，对端就能用 `address` + `rkey` 发起 RDMA READ/WRITE。三件套语义：

- `address`：MR 在**本地**的虚拟地址（u64）。
- `lkey`（local key）：**本地**访问 MR 时用的 key（send/recv WR 的 SGE 里填它）。
- `rkey`（remote key）：**远端**访问 MR 时用的 key（RDMA READ/WRITE 的 RQE 里填它）。

注释“must be on the heap”很重要：`MemoryRegionToken` 必须用 `u_ptr` 放堆上，不能直接放 `vec<MemoryRegionToken>`——因为 vector 扩容会搬家，地址就变了，对端拿到的旧 `address` 失效。`vec<u_ptr<MemoryRegionToken>>` 里存的是指针，vector 扩容只搬指针，token 本体不动。

### 4.8.2 `MemoryRegion` 类

```cpp
// memory_region.hh:22-51
class MemoryRegion {
protected:
  MemoryRegion(Context& context,
               void* data,
               size_t size_in_bytes,
               bool remote_access);

public:
  MemoryRegion(Context& context, void* data, size_t size_in_bytes);
  explicit MemoryRegion(Context& context);

  ~MemoryRegion();
  MemoryRegion(const MemoryRegion&) = delete;
  MemoryRegion& operator=(const MemoryRegion&) = delete;

  void register_memory(void* data, size_t size_in_bytes, bool remote_access);
  MemoryRegionToken createToken() const;

  u64 get_address() const { return reinterpret_cast<u64>(data_); }
  size_t get_size_in_bytes() const { return size_in_bytes_; }
  u32 get_lkey() const { return memory_region_->lkey; }
  u32 get_rkey() const { return memory_region_->rkey; }

private:
  Context& context_;
  void* data_{nullptr};
  size_t size_in_bytes_{0};
  ibv_mr* memory_region_{nullptr};
  bool is_registered_{false};
};
```

三个构造函数对应三种用法：

- **protected 4 参数版**（24–27 行）：基类内部用，带 `remote_access` 开关。
- **public 3 参数版**（30 行）：默认 `remote_access = true`，对外暴露 rkey——这是“给对端读的 MR”。
- **public 1 参数版**（31 行）：只绑 Context，不立即注册。配合 `register_memory` 延迟注册用。

`get_lkey`/`get_rkey` 直接读 `ibv_mr` 的字段——`ibv_mr` 是 verbs 返回的结构，`lkey`/`rkey` 由硬件分配，全局唯一。

### 4.8.3 `register_memory`：核心注册逻辑

```cpp
// memory_region.cc:15-30
void MemoryRegion::register_memory(void* data,
                                   const size_t size_in_bytes,
                                   bool remote_access) {
  int access = (remote_access)
                 ? IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE
                 : IBV_ACCESS_LOCAL_WRITE;
  memory_region_ =
    ibv_reg_mr(context_.get_protection_domain(), data, size_in_bytes, access);

  lib_assert(memory_region_, "Cannot register memory region");

  data_ = data;
  size_in_bytes_ = size_in_bytes;
  is_registered_ = true;
}
```

`ibv_reg_mr(pd, data, size, access)` 把 `[data, data+size)` 这段虚拟内存注册到 RNIC，RNIC 会 pin 住底层物理页（防止被换出，因为 DMA 用物理地址），并分配 `lkey`/`rkey`。

access flags 两种模式：

- **`remote_access = true`**（默认）：`REMOTE_READ | REMOTE_WRITE | REMOTE_ATOMIC | LOCAL_WRITE`。对端可以 RDMA READ/WRITE/ATOMIC，本地也能写（为什么本地写需要 `LOCAL_WRITE`？因为 verbs 要求“要被远端写的 MR 必须本地也可写”，硬件一致性约束）。
- **`remote_access = false`**：只有 `LOCAL_WRITE`。这种 MR 只能本地写，对端不能访问——用于纯本地缓冲（如接收缓冲在 SRQ 模式下其实需要 `LOCAL_WRITE` 给硬件写，不需要 remote）。

注意没有 `IBV_ACCESS_LOCAL_READ`——本地读是默认隐含的，所有 MR 都能本地读。

### 4.8.4 两个子类

```cpp
// memory_region.cc:32-40
MemoryRegion::MemoryRegion(Context& context,
                           void* data,
                           const size_t size_in_bytes)
    : MemoryRegion(context, data, size_in_bytes, true) {}

LocalMemoryRegion::LocalMemoryRegion(Context& context,
                                     void* data,
                                     const size_t size_in_bytes)
    : MemoryRegion(context, data, size_in_bytes, false) {}
```

- `MemoryRegion`（3 参数版）委托给 4 参数 protected 版，`remote_access = true`。
- `LocalMemoryRegion` 委托给 4 参数 protected 版，`remote_access = false`。

`memory_region.hh:53-59`：

```cpp
class LocalMemoryRegion : public MemoryRegion {
public:
  LocalMemoryRegion(Context& context, void* data, size_t size_in_bytes);
};
using LocalMemoryRegions = vec<u_ptr<LocalMemoryRegion>>;
using MemoryRegions       = vec<u_ptr<MemoryRegion>>;
```

`LocalMemoryRegion` 用于“只在本机 verbs 操作中用的缓冲”（如 SRQ 接收缓冲、QP recv 缓冲），不需要被远端访问。`DynamicRegionAllocator`（4.9 节）就用 `LocalMemoryRegion`。

### 4.8.5 析构与 token 创建

```cpp
// memory_region.cc:42-47
MemoryRegion::~MemoryRegion() {
  if (is_registered_) {
    lib_assert(ibv_dereg_mr(memory_region_) == 0,
               "Cannot deregister memory region.");
  }
}

MemoryRegionToken MemoryRegion::createToken() const {
  return MemoryRegionToken{get_address(), get_lkey(), get_rkey()};
}
```

- 析构：`ibv_dereg_mr` 解除注册，RNIC 释放 pin 的物理页映射。`is_registered_` guard 处理“1 参数构造但没调 `register_memory`”的情况——避免 dereg 一个空指针。
- `createToken`：把 `address/lkey/rkey` 打包成 `MemoryRegionToken`，用于发送给对端。这就是第 19 课 RDMA cache 的 `rkey` 来源：存储节点启动时把图/PQ 码 MR 的 token 通过 RPC 发给计算节点，计算节点缓存后用 `address + rkey` 发起 RDMA READ。

### 4.8.6 MR 生命周期与对象关系

一个典型的 dvstor 存储节点 MR 生命周期：

1. `HugePage::allocate(N)` 分配一段大页内存（4.10 节）。
2. `MemoryRegion(ctx, hugepage.get_full_buffer(), N)` 注册成 RNIC 可寻址区域，`remote_access = true`。
3. `mr.createToken()` 得到 `{address, lkey, rkey}`。
4. 通过 RPC（第 24 课）把 token 发给计算节点。
5. 计算节点把 token 存进 RDMA cache（第 19 课），用 `address + rkey` 发起 `ibv_post_send` RDMA READ。
6. 存储节点退出时 `MemoryRegion` 析构 `ibv_dereg_mr`，`HugePage` 析构 `munmap`。

---

## 4.9 `dynamic_region_allocator.hh`：动态 MR 分配器

存储节点要服务大量并发请求，每个请求需要一个接收缓冲。如果每个请求都 `new + ibv_reg_mr`，开销巨大（`ibv_reg_mr` 要 pin 物理页，毫秒级）。`DynamicRegionAllocator` 是“预分配 + 按需扩容 + freelist 复用”的池化方案。

```cpp
// dynamic_region_allocator.hh:12-23
template <typename BufferEntryType>
class DynamicRegionAllocator {
public:
  DynamicRegionAllocator(Context& context, u32 preallocate, size_t region_size)
      : context_(context),
        region_size_(region_size),
        region_length_(region_size / sizeof(BufferEntryType)) {
    // pre-allocate some regions
    for (u32 i = 0; i < preallocate; ++i) {
      allocate_region(true);
    }
  }
```

模板参数 `BufferEntryType` 是缓冲的元素类型（如某个 RPC 响应结构体）。构造时预分配 `preallocate` 个 region，每个 region 大小 `region_size` 字节，能装 `region_size / sizeof(BufferEntryType)` 个元素。`allocate_region(true)` 的 `true` 表示 touch 内存（强制落实物理页）。

### 4.9.1 申请与归还

```cpp
// dynamic_region_allocator.hh:25-37
u32 get_free_region_id() {
  u32 id;
  while (!free_list_.try_dequeue(id)) {
    allocate_region();
    lib_debug("allocated response regions: " + std::to_string(region_buffers_.size()));
  }
  return id;
}

void free_region(u32 region_id) { free_list_.enqueue(region_id); }
```

- `get_free_region_id`：从 freelist 取一个 region id。如果空了，就 `allocate_region()` 扩容，再试。这是无界增长——上层需要自己限流。
- `free_region`：用完归还，不释放内存，只把 id 塞回 freelist。

`free_list_` 是 `concurrent_queue<u32>`（moodycamel 无锁队列），多线程并发申请/归还无锁。

### 4.9.2 访问 region

```cpp
// dynamic_region_allocator.hh:39-47
LocalMemoryRegion* get_memory_region(u32 region_id) {
  return memory_regions_[region_id].get();
}
BufferEntryType* get_region_buffer(u32 region_id) {
  return region_buffers_[region_id].get();
}
size_t allocated_regions() const { return memory_regions_.size(); }
```

按 id 取 MR 指针和缓冲指针。注意 `region_id` 就是 vector 下标——`allocate_region` 里 `free_list_.enqueue(region_buffers_.size() - 1)` 保证这一点。`concurrent_vec` 的“插入后地址稳定”保证下标访问安全。

### 4.9.3 扩容

```cpp
// dynamic_region_allocator.hh:50-63
void allocate_region(bool touch = false) {
  std::scoped_lock lock{mutex_};
  // note that emplace_back of concurrent_vec returns an iterator
  auto buffer_ptr =
    region_buffers_.emplace_back(new BufferEntryType[region_length_]);
  memory_regions_.emplace_back(std::make_unique<LocalMemoryRegion>(
    context_, buffer_ptr->get(), region_size_));

  if (touch) {
    touch_memory(*buffer_ptr, region_length_);
  }

  free_list_.enqueue(region_buffers_.size() - 1);
}
```

关键步骤：

1. **加锁 `std::scoped_lock`**：保护扩容的临界区。虽然 `concurrent_vec` 本身线程安全，但“emplace_back buffer + emplace_back MR + enqueue id”这三步必须原子——否则别的线程可能看到 `region_buffers_` 多了一项但 `memory_regions_` 还没跟上，或者 id 还没入队。锁内同时只能一个线程扩容。
2. **`new BufferEntryType[region_length_]`**：堆分配元素数组。
3. **`std::make_unique<LocalMemoryRegion>(context_, buffer_ptr->get(), region_size_)`**：把这段内存注册成 `LocalMemoryRegion`（`remote_access = false`，只给本地 SRQ 接收用）。
4. **`touch_memory`**：可选，逐元素写零。预分配时 touch 一次，避免运行时首次访问缺页。
5. **`free_list_.enqueue(region_buffers_.size() - 1)`**：把新 region 的 id（= 当前 vector 长度 - 1）入队。

`buffer_ptr->get()` 这里 `buffer_ptr` 是 `concurrent_vec::emplace_back` 返回的迭代器，指向 `u_ptr<BufferEntryType[]>`，`->get()` 取裸指针。注意 `LocalMemoryRegion` 持有 `data_` 指针但**不拥有**内存——内存所有权在 `region_buffers_` 的 `u_ptr`。析构顺序：先 `memory_regions_` 析构（dereg MR），再 `region_buffers_` 析构（delete[]）。C++ 成员析构顺序是声明逆序，正好满足：`free_list_` → `region_buffers_` → `memory_regions_` → 其他。等等，看声明：

```cpp
// dynamic_region_allocator.hh:70-73
concurrent_vec<u_ptr<LocalMemoryRegion>> memory_regions_;
concurrent_vec<u_ptr<BufferEntryType[]>> region_buffers_;
concurrent_queue<u32> free_list_;
std::mutex mutex_;
```

成员析构顺序是声明逆序：`mutex_` → `free_list_` → `region_buffers_` → `memory_regions_`。这意味着 `memory_regions_`（MR）**先**析构，`region_buffers_`（buffer）**后**析构——正好是安全的顺序（先 dereg MR，再 free buffer）。这是作者刻意安排的声明顺序。

### 4.9.4 使用场景

第 23 课存储节点用 `DynamicRegionAllocator<RpcResponse>` 预分配一组接收缓冲，每来一个请求 `get_free_region_id()` 拿一个，`post_shared_receive(*mr)` 投递到 SRQ，处理完 `free_region(id)` 归还。这样高频接收不需要动态 `ibv_reg_mr`。

---

## 4.10 `hugepage.hh`：大页对齐分配

RDMA 对内存对齐敏感：RNIC DMA 引擎对页边界有偏好，普通 `malloc` 的 4KB 页会导致 TLB miss 频繁，影响吞吐。`HugePage` 用 `mmap` + `MAP_HUGETLB` 直接拿 2MB 或 1GB 大页。

### 4.10.1 类骨架

```cpp
// hugepage.hh:11-12
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
```

这两个宏在 `<sys/mman.h>` 没定义时提供 fallback。`MAP_HUGE_SHIFT` 是内核定义的（26），`21`/`30` 是 log2(2MB)/log2(1GB) 对应的页大小编码。

```cpp
// hugepage.hh:14-22
template <typename T, bool HUGE_1GB = true>
class HugePage {
public:
  HugePage() = default;
  explicit HugePage(size_t size) { allocate(size); }
  ~HugePage() { deallocate(); }

  HugePage(HugePage&) = delete;
  HugePage& operator=(HugePage&) = delete;
```

模板参数：
- `T`：元素类型，`buffer_[i]` 返回 `T&`。
- `HUGE_1GB`：默认用 1GB 大页；`false` 则用 2MB。

### 4.10.2 `allocate`

```cpp
// hugepage.hh:24-54
void allocate(size_t size) {
  lib_assert(buffer_size == 0, "Buffer has been already allocated");
  buffer_size = size;
  buffer_length = size / sizeof(T);
  size_left_ = size;

#ifdef NOHUGEPAGES
  buffer_ = static_cast<T*>(std::aligned_alloc(64, buffer_size));
  lib_assert(reinterpret_cast<u64>(buffer_) % 64 == 0,
             "Not cache-line aligned");
  std::cerr << "allocated ALIGNED MEM (no hugepage) at "
            << reinterpret_cast<u64>(buffer_) << " with buffer size "
            << buffer_size << std::endl;
#else
  print_status("map huge page");
  void* ptr = mmap(NULL,
                   buffer_size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
                     (HUGE_1GB ? MAP_HUGE_1GB : MAP_HUGE_2MB),
                   -1,
                   0);
  lib_assert(ptr != MAP_FAILED, "Allocating huge-pages failed");
  lib_assert(reinterpret_cast<u64>(ptr) % 64 == 0, "alignment failed");
  buffer_ = static_cast<T*>(ptr);
  std::cerr << "allocated HUGEPAGE at " << reinterpret_cast<u64>(buffer_)
            << " with buffer size " << buffer_size << std::endl;
#endif

  bump_pointer_ = buffer_;
}
```

两条路径：

- **`NOHUGEPAGES` 宏定义时**：用 `std::aligned_alloc(64, size)` 拿 64 字节对齐的普通内存（fallback，开发/测试环境用，不需要 hugepage 配额）。
- **默认**：`mmap` + `MAP_HUGETLB` + `MAP_HUGE_1GB/2MB`。`MAP_PRIVATE | MAP_ANONYMOUS` 是私有匿名映射。`-1, 0` 表示不基于文件。

注意 `mmap(NULL, size, ...)` 的 `size` **必须是页大小的整数倍**，否则会失败。调用方要保证——`HugePage` 自己不检查。dvstor 存储节点 `--mn-memory 10`（10 GiB）正好是 1GB 页的整数倍。

`bump_pointer_ = buffer_`：初始化 bump allocator 的指针。

### 4.10.3 Bump 分配

```cpp
// hugepage.hh:56-65
T* get_slice_unaligned(size_t size_in_bytes) {
  lib_assert(size_left_ >= size_in_bytes,
             "Pre-allocated hugepage memory exhausted");
  T* slice = static_cast<T*>(bump_pointer_);
  bump_pointer_ = static_cast<byte_t*>(bump_pointer_) + size_in_bytes;
  size_left_ -= size_in_bytes;
  return slice;
}

T* get_slice(size_t size_in_bytes) {
  lib_assert(size_left_ >= size_in_bytes,
             "Pre-allocated hugepage memory exhausted");
  lib_assert(
    std::align(64, size_in_bytes, bump_pointer_, size_left_) != nullptr,
    "alignment failed");
  T* slice = static_cast<T*>(bump_pointer_);
  bump_pointer_ = static_cast<byte_t*>(bump_pointer_) + size_in_bytes;
  size_left_ -= size_in_bytes;
  lib_assert(reinterpret_cast<u64>(slice) % 64 == 0, "alignment failed");
  return slice;
}
```

两个版本：

- **`get_slice_unaligned`**：直接 bump，不补齐。适合任意大小。
- **`get_slice`**：先用 `std::align(64, size, ptr, space)` 把 `bump_pointer_` 向上对齐到 64 字节边界，再 bump。这是 cache-line 对齐版本，用于 MR 切片——RNIC 对 cache-line 对齐的地址访问更快。

`std::align` 是 `<memory>` 的标准函数，它就地修改 `ptr` 和 `space`，返回对齐后的指针（或 `nullptr` 表示空间不够）。这里用 `lib_assert` 兜底。

bump 分配是单向的——没有 `free_slice`。整块 hugepage 在 `HugePage` 析构时一次性 `munmap`。这适合“启动时一次性切分，运行时永不回收”的场景，例如存储节点把图/PQ 码/向量各切一片。

### 4.10.4 辅助方法

```cpp
// hugepage.hh:101-117
size_t get_num_hugepages() const {
  std::ifstream is("/proc/sys/vm/nr_hugepages");
  size_t num_hugepages;
  lib_assert(is.good(), "Cannot read the number of available hugepages");
  lib_assert((is >> num_hugepages), "Cannot get the number of hugepages");
  return num_hugepages;
}

size_t get_memory_size() const {
#ifdef NOHUGEPAGES
  return 500UL * 1024UL * 1024UL * 1024UL;  // 500 GB
#else
  return get_num_hugepages() * 1024UL * (HUGE_1GB ? 1024UL * 1024UL : 2048UL);
#endif
}
```

- `get_num_hugepages` 读 `/proc/sys/vm/nr_hugepages` 拿系统配额。运维要 `echo N > /proc/sys/vm/nr_hugepages` 预留大页。
- `get_memory_size`：1GB 页模式 = `nr_hugepages * 1GB`，2MB 页模式 = `nr_hugepages * 2MB`。注意 `nr_hugepages` 在 1GB 和 2MB 模式下含义不同——内核有独立的 `nr_hugepages_1GB`，但这里读的是默认的 `nr_hugepages`，所以 1GB 模式下其实是个粗略估计。`NOHUGEPAGES` 时硬编码 500GB 上限。

### 4.10.5 其他

```cpp
// hugepage.hh:86-99
void deallocate() {
  if (buffer_ != nullptr) {
#ifdef NOHUGEPAGES
    std::free(buffer_);
#else
    munmap(static_cast<void*>(buffer_), buffer_size);
#endif
    buffer_ = nullptr;
    bump_pointer_ = nullptr;
  }
  buffer_length = 0;
  buffer_size = 0;
}
```

`deallocate` 用 `munmap` 释放（或 `std::free` 在 NOHUGEPAGES 模式）。`if (buffer_ != nullptr)` 防止双重释放。

```cpp
// hugepage.hh:119-125
T& operator[](size_t idx) { return *(buffer_ + idx); }
void touch_memory() {
  for (size_t i = 0; i < buffer_length; ++i) {
    buffer_[i] = 0;
  }
}
```

`operator[]` 提供数组访问。`touch_memory` 逐元素写零，落实物理页——RDMA MR 注册前 touch 一次，避免运行时缺页。

---

## 4.11 关键数据结构与对象关系图

### 4.11.1 Context → PD → MemoryRegion → 远端 token 交换

```
┌────────────────────────  Process (storage node)  ────────────────────────┐
│                                                                          │
│   Configuration                                                          │
│   (ib_device, device_port, max_*_queue_wr, is_server, server_nodes...)   │
│         │                                                                │
│         │ ref                                                            │
│         ▼                                                                │
│   ┌───────────────────── Context ─────────────────────┐                  │
│   │ ibv_device*  device_                               │                  │
│   │ ibv_context* context_  ◄── ibv_open_device         │                  │
│   │ ibv_pd*      protection_domain_  ◄── ibv_alloc_pd  │                  │
│   │ ibv_cq*      send_cq_          ◄── ibv_create_cq   │                  │
│   │ ibv_cq*      receive_cq_       ◄── ibv_create_cq   │                  │
│   │ ibv_srq*     shared_receive_cq_ (optional)         │                  │
│   │ ibv_port_attr port_attributes_ (lid, state, ...)   │                  │
│   │ ibv_device_attr device_attributes_                 │                  │
│   └──────────────┬─────────────────────────────────────┘                  │
│                  │ PD ref                                                │
│                  ▼                                                       │
│   ┌──────────────────── MemoryRegion ───────────────────┐                │
│   │ Context& context_                                    │                │
│   │ void*  data_  ───────────► [HugePage slice / new[]] │                │
│   │ size_t size_in_bytes_                                │                │
│   │ ibv_mr* memory_region_  ◄── ibv_reg_mr(pd, data,    │                │
│   │                                   size, access)      │                │
│   │   memory_region_->lkey  (local access key)          │                │
│   │   memory_region_->rkey  (remote access key)         │                │
│   │ bool is_registered_                                  │                │
│   └──────────────┬───────────────────────────────────────┘                │
│                  │ createToken()                                         │
│                  ▼                                                       │
│   ┌──────────────────── MemoryRegionToken ──────────────┐                │
│   │ u64 address  ◄── get_address()                       │                │
│   │ u32 lkey     ◄── get_lkey()                          │                │
│   │ u32 rkey     ◄── get_rkey()                          │                │
│   └──────────────┬───────────────────────────────────────┘                │
│                  │                                                       │
└──────────────────┼───────────────────────────────────────────────────────┘
                   │  TCP handshake / RPC (第 24 课)
                   ▼
┌──────────────────  Process (compute node)  ──────────────────────────────┐
│   RDMA Cache (第 19 课)                                                   │
│   {remote_address, remote_rkey}  ──► ibv_post_send RDMA READ             │
│                                          │                                │
│                                          ▼                                │
│                          RNIC DMA reads from storage node's MR            │
│                          using (address, rkey)                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 4.11.2 TCP 握手时序

```
   Compute (client)                        Storage (server)
   ─────────────────                       ─────────────────
   connect_to_server(addr, port, node_id)
   │  socket() + connect()                 │  bind_to_port(port)
   │  ───────────────────────────────────► │  listen()
   │                                       │  wait_for_connection()
   │                                       │    accept()
   │  send(QPInfo{my_lid, my_qp_num,       │
   │              node_id})                │
   │  ───────────────────────────────────► │  recv() → receive_buffer
   │                                       │
   │                       recv() ◄──────  │  send(QPInfo{srv_lid, srv_qp_num})
   │  ◄─────────────────────────────────── │
   │  receive_buffer ready                 │
   │                                       │
   │  transition_to_rtr(receive_buffer)    │  transition_to_rtr(receive_buffer)
   │  transition_to_rts()                  │  transition_to_rts()
   │                                       │
   │  close(tcp_socket)                    │  close(tcp_socket)
   │                                       │
   │  ★ RDMA path now ready ★             │  ★ RDMA path now ready ★
```

### 4.11.3 DynamicRegionAllocator 内部

```
   DynamicRegionAllocator<ResponseT>
   ┌─────────────────────────────────────────────────────────────┐
   │ region_size_ = N bytes                                      │
   │ region_length_ = N / sizeof(ResponseT)                      │
   │                                                             │
   │  memory_regions_  (concurrent_vec<u_ptr<LocalMemoryRegion>>) │
   │   [0] → LocalMemoryRegion ─► ibv_mr{lkey}                   │
   │   [1] → LocalMemoryRegion ─► ibv_mr{lkey}                   │
   │   [2] → LocalMemoryRegion ─► ibv_mr{lkey}                   │
   │   ...                                                       │
   │  region_buffers_  (concurrent_vec<u_ptr<ResponseT[]>>)      │
   │   [0] ──► [ ResponseT | ResponseT | ... ]                   │
   │   [1] ──► [ ResponseT | ResponseT | ... ]                   │
   │   [2] ──► [ ResponseT | ResponseT | ... ]                   │
   │   ...                                                       │
   │  free_list_  (concurrent_queue<u32>)                        │
   │   ──► [2, 0, 1, ...]  (region_id 的复用池)                  │
   │                                                             │
   │  mutex_  (扩容时持有)                                       │
   └─────────────────────────────────────────────────────────────┘

   get_free_region_id():
     try_dequeue from free_list_ ──── success ──► return id
                    │ fail
                    ▼
     allocate_region()  (持有 mutex_)
       new ResponseT[region_length_]            ─► region_buffers_.emplace_back
       LocalMemoryRegion(ctx, ptr, region_size_) ─► memory_regions_.emplace_back
       free_list_.enqueue(region_buffers_.size() - 1)
     retry try_dequeue
```

### 4.11.4 内存归属表

| 资源 | 持有者 | 释放方式 | verbs 销毁调用 |
|---|---|---|---|
| `ibv_device` 列表 | `ibv_get_device_list` 返回 | `Context` 构造末尾 `ibv_free_device_list` | `ibv_free_device_list` |
| `ibv_context` | `Context::context_` | `Context` 析构 | `ibv_close_device` |
| `ibv_pd` | `Context::protection_domain_` | `Context` 析构 | `ibv_dealloc_pd` |
| `ibv_cq` | `Context::send_cq_` / `receive_cq_` | `Context` 析构 | `ibv_destroy_cq` |
| `ibv_srq` | `Context::shared_receive_cq_` | `Context` 析构 | `ibv_destroy_srq` |
| `ibv_mr` | `MemoryRegion::memory_region_` | `MemoryRegion` 析构 | `ibv_dereg_mr` |
| 大页内存 | `HugePage::buffer_` | `HugePage` 析构 | `munmap` |
| `ibv_qp` | `QueuePair`（第 5 课） | `QueuePair` 析构 | `ibv_destroy_qp` |
| RPC 接收缓冲 | `DynamicRegionAllocator::region_buffers_` | `DynamicRegionAllocator` 析构 | `delete[]` |

---

## 4.12 与其他模块的关系

- **第 2 课**：`src/common/configuration.hh` 的 `IndexConfiguration` 公开继承本课的 `Configuration`，复用 `desc`/`process_program_options`/`exit_with_help_message`/`operator<<`。`src/common/types.hh` `#include <library/types.hh>` 继承所有别名，再追加 `node_t`/`element_t`/`distance_t`/`filepath_t`/`hashset_t`/`hashmap_t`。
- **第 3 课**：`thread.hh` 的 `Thread` 和并发容器别名（`concurrent_vec`/`concurrent_queue`）是第 3 课并发原语的一部分。本课讲的是库内版本，第 3 课会讲 dvstor 业务侧的协程/RCU 等更高层原语。
- **第 5 课**：`context.hh:53-56` 的 `wait_for_connection`/`connect_to_server` 返回 `u_ptr<QueuePair>`，`QueuePair` 类在 `queue_pair.hh/cc`。第 5 课讲 QP 状态机（RESET→INIT→RTR→RTS）、`ibv_post_send`/`ibv_post_recv`、`QPInfo` 结构、`transition_to_rtr`/`transition_to_rts` 的 verbs 调用细节。
- **第 17–22 课**：GPU 持久化引擎。本课的 `Configuration`（`gpu_rdma_qps` 等在 `IndexConfiguration` 里）、`MemoryRegionToken`（计算节点缓存远端 MR 的 token）、CQ 轮询模式都被 GPU 路径复用。第 22 课 GPUNetIO 不直接用 `Context`/`QueuePair`，但 token 数据结构和握手思路一致。
- **第 19 课 RDMA cache**：计算节点缓存对端 `MemoryRegionToken`，用 `address + rkey` 发起 RDMA READ。本课的 `MemoryRegionToken` 是它的输入。
- **第 23 课存储节点**：起 `Context`（`is_server=true`）、`bind_to_port`、循环 `wait_for_connection` 接受计算节点。用 `HugePage` 分配大块注册内存，用 `DynamicRegionAllocator` 池化接收缓冲，用 `post_shared_receive` 投递到 SRQ。
- **第 24 课 peer RPC**：存储节点之间互连也用本课的 `connect_to_server`/`wait_for_connection`，token 交换走 RPC。
- **第 30 课 breakdown benchmark**：用 `compute_throughput`/`compute_latency` 统计 RDMA 吞吐和延迟。

---

## 4.13 小结

本课讲清了 dvstor RDMA 传输库的“底座”：

1. **`types.hh`** 提供全库类型别名字典（`u32`/`str`/`vec`/`u_ptr`/`span`/`concurrent_vec`/`concurrent_queue`），被 `src/common/types.hh` 继承。
2. **`utils.hh/cc`** 提供错误处理宏 `lib_assert`/`lib_debug`（用宏避免临时字符串开销）、`Endpoint`/`parse_endpoint`（host:port 解析 + cluster 别名查表）、吞吐/延迟统计、位运算工具。
3. **`thread.hh`** 提供 `Thread` RAII 封装（worker 协议：`u32 thread_id` 尾参；`atomic<bool> done_` 跨线程可见性）和 CPU 亲和绑定辅助。
4. **`configuration.hh/cc`** 的 `Configuration` 用 boost::program_options 解析 RDMA 通用选项（IB device/port、CQ/WR 容量、TCP 端口、角色 flags、节点列表），是 `IndexConfiguration` 的基类。
5. **`context.hh/cc`** 的 `Context` 封装 verbs 资源链：`ibv_get_device_list → ibv_open_device → ibv_query_device → ibv_alloc_pd → ibv_query_port → ibv_create_cq → (ibv_create_srq)`，析构严格反向。提供 TCP 握手（`bind_to_port`/`wait_for_connection`/`connect_to_server`）交换 `QPInfo` 后切到 RDMA、SRQ 接收投递、CQ 轮询（`poll_recv_cq`/`poll_send_cq` 及 spin 封装）。
6. **`memory_region.hh/cc`** 的 `MemoryRegion` 包装 `ibv_reg_mr`，按 `remote_access` 开关区分“可被远端访问”与“纯本地”两类 access flags；`MemoryRegionToken{address, lkey, rkey}` 是远端访问三件套；`LocalMemoryRegion` 子类用于 SRQ 接收缓冲等纯本地场景。
7. **`dynamic_region_allocator.hh`** 是模板化 MR 池：预分配 + freelist 复用 + 持锁扩容，避免高频 `ibv_reg_mr`。成员声明顺序刻意安排让 MR 先析构、buffer 后析构。
8. **`hugepage.hh`** 用 `mmap + MAP_HUGETLB` 拿 1GB/2MB 大页，提供 cache-line 对齐的 bump 分配器，是 MR 注册前的内存来源。

下一课（第 5 课）会补上 `queue_pair.hh/cc`——QP 状态机、`QPInfo` 结构、`ibv_post_send`/`ibv_post_recv` 的封装，把本课的 `Context` 和 `MemoryRegion` 串成完整的 verbs 数据通路。
