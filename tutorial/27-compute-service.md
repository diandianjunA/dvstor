# 第 27 课 计算服务主体

> 本课是 Part VII 的第一课。Part VI 讲完了"存储节点（第 23–26 课）"这一侧：peer RDMA、peer RPC、索引访问、维护协议。从本课开始，我们把视角切回计算节点，看 `ComputeService` 这个对外的"门面类"是如何把 RDMA 通道、GPU 持久化搜索引擎、storage-owner RPC、breakdown 采样全部串起来的。第 28 课会接着讲计算侧的 storage owner 更新子系统（即 `src/service/compute_service/storage_owner/` 下的 `sender.cc`/`completion.cc`/`public_mutations.cc`/`lifecycle.cc`），所以本课在讲 `index_commands.cc` 里 `ComputeSideIdEntry`/`BaseOwnerMap` 等共享状态时，会特意把"运行时更新"的部分留给第 28 课。

---

## 27.1 本课目标与涉及文件

本课要回答的问题是：

1. 进程从 `main` 进入后，`ComputeService` 是如何被构造的？构造期完成了哪些"启动握手"？
2. 计算节点对外暴露的 `search` / `search_raw` / `insert` / `upsert` / `erase` / `status` 接口，分别落到哪段代码？一次 query 是怎么从 API 调用走到 `PersistentSearchEngine::search`，再被 breakdown 采样捕获的？
3. 启动期的"索引命令"具体在做什么？`validate_index_metadata` 这个 70 行的校验函数，到底校验了 schema-15 索引的哪些不变量？
4. 计算服务内部的 `detail.hh` 共享了哪些常量与工具？
5. breakdown 子系统（`service/breakdown.hh` + `service/breakdown/{names,sample,aggregate,json,text}.hh`）是如何在 per-request 采样、聚合、报表三个层次上组织起来的？它和第 9 课的 `CompletionDescriptor`、第 30 课的 benchmark 是如何衔接的？

涉及文件（均为绝对路径，行号引用即对应该文件）：

- `/home/xjs/experiment/dvstor/src/main.cc`（进程入口，35 行）
- `/home/xjs/experiment/dvstor/src/service/compute_service.hh`（`ComputeService` 类声明，226 行）
- `/home/xjs/experiment/dvstor/src/service/compute_service/detail.hh`（模块内部头，36 行）
- `/home/xjs/experiment/dvstor/src/service/compute_service/lifecycle.cc`（构造/析构，79 行）
- `/home/xjs/experiment/dvstor/src/service/compute_service/search.cc`（查询入口，78 行）
- `/home/xjs/experiment/dvstor/src/service/compute_service/index_commands.cc`（status + compute-side idmap + 索引元数据校验 + 启动握手，222 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown.hh`（汇总 include，7 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown/names.hh`（Operation/Category/Subcategory 枚举与名字表，108 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown/sample.hh`（per-request `Sample`，60 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown/aggregate.hh`（`Aggregate`/`Report` + 聚合函数，94 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown/json.hh`（JSON 报表，67 行）
- `/home/xjs/experiment/dvstor/src/service/breakdown/text.hh`（文本摘要，33 行）

辅助参考：

- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine.hh`（引擎接口，见第 11 课）
- `/home/xjs/experiment/dvstor/src/common/core_assignment.hh`（核分配，见第 3 课）
- `/home/xjs/experiment/dvstor/rdma-library/library/connection_manager.hh` / `memory_region.hh`（见第 4、5 课）

---

## 27.2 进程入口：`main.cc`

整个计算节点的可执行文件入口只有 35 行（`src/main.cc`）：

```cpp
int main(int argc, char** argv) {
  configuration::IndexConfiguration config{argc, argv};

  if (config.is_server) {
    std::cerr << "use dvstor_memory_node for a storage process\n";
    return EXIT_FAILURE;
  }
  ComputeService service{config};
  wait_for_shutdown_signal();

  return EXIT_SUCCESS;
}
```

逐行讲：

- `configuration::IndexConfiguration config{argc, argv};`：用 boost.program_options 解析命令行（见第 2 课）。`is_server` 是一个布尔字段，用来区分"这个进程应该是计算节点还是存储节点"。同一个二进制在 Makefile 里被同时部署成 `dvstor_compute` 和 `dvstor_memory_node`（见第 1 课构建系统），所以这里用 `config.is_server` 做了一次"自我身份检查"：如果用户错用计算二进制启动存储进程，立刻报错退出。
- `ComputeService service{config};`：构造服务对象。**这一行几乎做了所有启动期工作**——RDMA 连接、token 交换、元数据校验、引擎构造、storage 节点唤醒、storage insert 运行时启动，全部塞在构造函数里（见 27.4）。所以构造完之后，服务已经是"就绪"状态。
- `wait_for_shutdown_signal()`：阻塞主线程直到 `SIGINT`/`SIGTERM`。注意它先用 `pthread_sigmask` 屏蔽这两个信号，再用 `sigwait` 同步等待——这是多线程程序里推荐的写法，避免信号被任意工作线程捕获造成不可预测的行为。返回后 `service` 析构、进程退出。

`wait_for_shutdown_signal` 的实现也很短：

```cpp
void wait_for_shutdown_signal() {
  sigset_t block_set;
  sigemptyset(&block_set);
  sigaddset(&block_set, SIGINT);
  sigaddset(&block_set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &block_set, nullptr);

  int sig = 0;
  sigwait(&block_set, &sig);
  print_status("received signal " + std::to_string(sig) + ", shutting down...");
}
```

关键点是 `pthread_sigmask(SIG_BLOCK, ...)` 必须在任何工作线程创建之前调用，否则新建线程会继承未屏蔽的信号掩码，可能抢走 `sigwait` 的信号。在 `ComputeService` 的构造函数里会创建 `storage_insert_progress_thread_` 和 `storage_insert_completion_thread_`（见 27.4），而 `main` 在构造 `ComputeService` 之前就完成了屏蔽，所以这一点是成立的。

---

## 27.3 `ComputeService` 类总览（`compute_service.hh`）

`ComputeService` 是一个不可拷贝的"胖门面"对象（`compute_service.hh:29-225`）。它持有：

```cpp
Configuration config_;                 // IndexConfiguration 的副本
Context context_;                      // RDMA Context（IBV 上下文 + PD）
ClientConnectionManager cm_;           // 所有 client↔server QP 的管理器
const u32 num_servers_;                // 存储节点数
MemoryRegionTokens remote_access_tokens_;   // 每个存储节点的 MR token
Assignment core_assignment_;           // CoreAssignment<interleaved>
std::atomic<size_t> vectors_inserted_{0};
std::unique_ptr<gpu_search::PersistentSearchEngine> persistent_search_;
std::thread storage_insert_progress_thread_;
std::thread storage_insert_completion_thread_;
std::atomic<bool> storage_insert_shutdown_{false};
...
std::unique_ptr<bounded::Queue<StorageOwnerReadySlot>> storage_ready_slots_;
std::unique_ptr<bounded::Queue<StorageOwnerReleasedSlot>> storage_released_slots_;
std::unique_ptr<bounded::CompletionPool> storage_completion_pool_;
std::unique_ptr<service::breakdown::Sample[]> storage_completion_samples_;
vec<std::unique_ptr<StorageOwnerSenderState>> storage_insert_owners_;
std::array<ComputeSideIdShard, kComputeSideIdShardCount> compute_side_idmap_;
service::BaseOwnerMap base_owner_map_;
std::atomic<u64> next_request_id_{1};
mutable std::mutex breakdown_mutex_;
std::atomic<bool> breakdown_enabled_{false};
service::breakdown::Report completed_breakdown_report_;
```

几个关键点：

- **类型别名**（`compute_service.hh:31-32`）：
  ```cpp
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;
  ```
  `interleaved` 来自 `common/core_assignment.hh:32`，对应"8,0,9,1,…,24,16,25,17,…"这种跨 NUMA 的交错核序（见第 3 课）。`core_assignment_` 作为成员意味着构造 `ComputeService` 时就已经把当前进程的 CPU 亲和性限制到了一个 partition 上。

- **不可拷贝**（`compute_service.hh:56-57`）：它持有 RDMA QP、GPU 引擎、多个 `std::thread`，拷贝在语义上无意义，所以显式 `delete` 拷贝构造和赋值。

- **对外接口**（`compute_service.hh:59-77`）：
  ```cpp
  size_t insert(const vec<InsertItem>& batch);
  size_t upsert(const vec<InsertItem>& batch);
  size_t erase(const vec<node_t>& ids);
  vec<node_t> search(const vec<element_t>& query, u32 k);
  vec<node_t> search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k);
  Status status() const;
  void reset_breakdown_state();
  void clear_thread_statistics();
  service::breakdown::Report collect_breakdown_report() const;
  gpu_search::TelemetrySnapshot gpu_search_telemetry() const { ... }
  u64 late_storage_owner_rpc_completions() const { ... }
  const Configuration& config() const { return config_; }
  ```
  注意本课只讲 `search` / `search_raw` / `status` / breakdown 相关的接口；`insert`/`upsert`/`erase` 的实现位于 `storage_owner/sender.cc` 和 `storage_owner/completion.cc`，**留到第 28 课**讲。本课会涉及到这些接口所依赖的共享状态（`compute_side_idmap_`、`base_owner_map_`、`StorageOwnerSenderState` 等），因为它们的声明都在 `compute_service.hh` 里，且 `index_commands.cc` 实现了 `publish_compute_side_id`/`lookup_compute_side_id`/`known_storage_owner_for_id`/`claim_storage_owner_for_mutation` 这几个状态操作。

- **内嵌数据结构**（`compute_service.hh:35-142`）：本课需要认识的几个：

  - `InsertItem`（35-38 行）：`{ node_t id; vec<element_t> values; }`，是 `insert`/`upsert` 的输入单元。
  - `Status`（40-45 行）：`status()` 的返回值，对外的运行时状态快照。
  - `LocalMainSearchOutput`（47-50 行）：`{ QueryResult results; shared_ptr<Sample> sample; }`，这是 `search_local_result` 的返回——查询结果与 breakdown 采样成对返回，再由 `search_local` 把采样合并进 `completed_breakdown_report_`。
  - `StorageInsertTask`（79-85 行）、`StorageOwnerRpcSlot`（87-114 行）、`StorageOwnerResponseSlot`（116-121 行）、`StorageOwnerSenderState`（123-131 行）：这些都是 storage-owner RPC 运行时的内部结构，**第 28 课**详细讲。这里只需要知道：每个存储节点对应一个 `StorageOwnerSenderState`，里面有一组 slot（slot 池）和一组 response slot（接收池），用 `bounded::Queue` 串联 task 就绪和 slot 释放两个方向。
  - `ComputeSideIdEntry`（206-211 行）和 `ComputeSideIdShard`（213-216 行）：计算侧的"逻辑 ID → {RemotePtr, deleted, owner_storage, generation}"映射，分 256 片加锁（`kComputeSideIdShardCount = 256`，212 行）。这是第 28 课的核心数据结构之一，本课在 27.6 节会讲它的"声明侧"语义。

---

## 27.4 生命周期：`lifecycle.cc`

`lifecycle.cc` 只有 79 行，但承载了整个计算节点的启动/停止流程。它定义了构造函数和析构函数。

### 27.4.1 构造函数

```cpp
ComputeService::ComputeService(const Configuration& config)
    : config_(config),
      context_(config_),
      cm_(context_, config_),
      num_servers_(config_.num_server_nodes()) {
  init_remote_tokens();
  cm_.connect();

  // Do not pin the constructor thread. POSIX threads inherit their creator's
  // affinity mask, so pinning here used to serialize every later query,
  // update, and benchmark worker onto the same CPU. Dedicated runtime threads
  // are pinned after creation by their owning subsystems instead.

  if (cm_.is_initiator) {
    const u32 gpu_rdma_qps = config_.gpu_rdma_qps * 2u;
    configuration::Parameters parameters{
      config_.num_threads,
      gpu_rdma_qps,
    };
    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&parameters, sizeof(parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  receive_remote_access_tokens();

  str metadata_error;
  const filepath_t startup_prefix = config_.resolved_index_prefix();
  lib_assert(validate_index_metadata(startup_prefix, &metadata_error), metadata_error);

  service::index_metadata::Metadata metadata;
  lib_assert(service::index_metadata::load_metadata(
               startup_prefix, metadata, &metadata_error), metadata_error);
  if (config_.enable_updates) {
    lib_assert(base_owner_map_.load(startup_prefix, num_servers_,
                                    metadata.idmap_format, &metadata_error),
               metadata_error);
    print_status("storage-owner base idmap: entries=" +
                 std::to_string(base_owner_map_.entry_count()) + " memory=" +
                 std::to_string(base_owner_map_.memory_bytes()) + " bytes");
    print_status(
      "storage-owner placement: base idmap for existing IDs; "
      "deterministic ID shard for new IDs");
  } else {
    print_status("compute updates disabled: owner idmaps and update executor are not loaded");
  }

  const cudaError_t cuda_status = cudaSetDevice(static_cast<int>(config_.gpu_device));
  lib_assert(cuda_status == cudaSuccess,
             str{"failed to select GPU: "} + cudaGetErrorString(cuda_status));
  print_status("search: GPU-persistent OPQ/PQ" +
               std::to_string(metadata.pq_subquantizers) +
               " beam + final RDMA exact rerank");
  persistent_search_ = std::make_unique<gpu_search::PersistentSearchEngine>(
    config_, context_, cm_, remote_access_tokens_);
  print_status("query engine: persistent GPU + GPUNetIO slots=" +
               std::to_string(config_.gpu_query_slots));

  cm_.synchronize();
  start_storage_nodes();
  synchronize_clients_after_startup();
  if (config_.enable_updates) start_storage_insert_runtime();
}
```

逐段讲：

**成员初始化列表（5-9 行）**：

```cpp
: config_(config),
  context_(config_),
  cm_(context_, config_),
  num_servers_(config_.num_server_nodes()) {
```

- `config_` 拷贝一份 `IndexConfiguration`，之后 `config()` 接口返回常引用。
- `context_(config_)`：构造 RDMA `Context`（打开 IBV device、分配 PD 等，见第 4 课）。
- `cm_(context_, config_)`：构造 `ClientConnectionManager`，它持有所有 `server_qps`（与存储节点的 QP）、`client_qps`（initiator 与其他计算节点的 QP）、`initiator_qp`、以及 `is_initiator` 标志（见 `rdma-library/library/connection_manager.hh:39-53`）。注意构造 `cm_` 本身并不建立连接，连接在下面 `cm_.connect()` 完成。
- `num_servers_`：存储节点数，从配置取，常量成员（`const u32`，`compute_service.hh:186`）。

**`init_remote_tokens()` + `cm_.connect()`（10-11 行）**：

`init_remote_tokens()` 的实现在 `index_commands.cc:94-99`：

```cpp
void ComputeService::init_remote_tokens() {
  remote_access_tokens_.resize(num_servers_);
  for (auto& token : remote_access_tokens_) {
    token = std::make_unique<MemoryRegionToken>();
  }
}
```

`MemoryRegionTokens` 就是 `vec<u_ptr<MemoryRegionToken>>`（`rdma-library/library/memory_region.hh:16-17`）。这里只是为每个存储节点预留一个空 token 指针，真正的填充在 `receive_remote_access_tokens()` 里。`cm_.connect()` 才真正完成所有 QP 的 RTS/RRTR 握手（见第 4 课）。

**"不要 pin 构造线程"的注释（13-16 行）**：

这段注释很重要，它解释了为什么 `core_assignment_` 虽然是成员（构造时会执行 `CoreAssignment` 的构造函数，里面会调用 `restrict_current_thread_to_partition()`，见 `core_assignment.hh:81-113`），但构造函数体本身不再调用任何 `pin_thread`。原因写在注释里：POSIX 线程会继承创建者的亲和性掩码，如果在构造期把"构造线程"pin 到某个核，那么之后所有由构造线程派生的工作线程（包括 benchmark 客户端线程）都会被继承到同一个核上，导致所有负载串行化。所以策略改为：`core_assignment_` 只是把当前进程限制到一个 partition（partition 内仍可多核），具体的工作线程 pinning 由各子系统自己负责。`CoreAssignment` 的 `get_available_core()` 提供 partition 内的核号（`core_assignment.hh:44`）。

**initiator 广播参数（18-28 行）**：

```cpp
if (cm_.is_initiator) {
  const u32 gpu_rdma_qps = config_.gpu_rdma_qps * 2u;
  configuration::Parameters parameters{
    config_.num_threads,
    gpu_rdma_qps,
  };
  for (const QP& qp : cm_.server_qps) {
    qp->post_send_inlined(&parameters, sizeof(parameters), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
}
```

计算节点之间有"initiator / non-initiator"的区分（见第 4 课 `ClientConnectionManager::is_initiator`）。initiator 负责把 GPU 的 `num_threads` 和 `gpu_rdma_qps*2`（QP 数 ×2，估计是 send/recv 各一）通过 RDMA send 广播给所有存储节点，让存储节点知道该为这个计算节点分配多少 recv buffer。`post_send_inlined` 是 inline send（小消息零拷贝，见第 4 课）。每个 QP 单独 `poll_send_cq_until_completion()` 串行确认。

**`receive_remote_access_tokens()`（30 行）**：

实现在 `index_commands.cc:101-110`：

```cpp
void ComputeService::receive_remote_access_tokens() {
  print_status("receive access tokens of remote memory regions");
  for (u32 memory_node = 0; memory_node < num_servers_; ++memory_node) {
    const QP& qp = cm_.server_qps[memory_node];
    MRT& token = remote_access_tokens_[memory_node];
    LocalMemoryRegion token_region{context_, token.get(), sizeof(MemoryRegionToken)};
    qp->post_receive(token_region);
    context_.receive();
  }
}
```

每个存储节点会通过 RDMA send 把自己的 `MemoryRegionToken`（包含远端 MR 的 rkey/地址等，见第 4 课）发过来。这里逐节点 `post_receive` 一个本地 MR 包住 token 指针，再 `context_.receive()` 阻塞等待。这之后 `remote_access_tokens_` 就有了所有存储节点的远端访问凭据，`PersistentSearchEngine` 才能对存储节点做 RDMA read。

**索引元数据校验与加载（32-55 行）**：

```cpp
str metadata_error;
const filepath_t startup_prefix = config_.resolved_index_prefix();
lib_assert(validate_index_metadata(startup_prefix, &metadata_error), metadata_error);

service::index_metadata::Metadata metadata;
lib_assert(service::index_metadata::load_metadata(
             startup_prefix, metadata, &metadata_error), metadata_error);
if (config_.enable_updates) {
  lib_assert(base_owner_map_.load(startup_prefix, num_servers_,
                                  metadata.idmap_format, &metadata_error),
             metadata_error);
  ...
} else {
  print_status("compute updates disabled: owner idmaps and update executor are not loaded");
}
```

`validate_index_metadata` 是 70 行的校验函数，详见 27.6 节。`lib_assert` 是带字符串的断言宏——失败时抛异常终止。注意两次读 metadata：先 `validate_index_metadata` 内部读一次做校验，再 `load_metadata` 读一次拿到 `Metadata` 对象。这里没合并是因为 `validate_*` 还要负责"配置 `VamanaNode` 的静态布局"等副作用（见 27.6）。

`base_owner_map_` 是 `service::BaseOwnerMap`（见第 8 课 index_metadata/owner map），只在 `enable_updates` 时加载。它给出"逻辑 ID → owner_storage"的不可变基线，用于第一眼就能确定一个已有 ID 属于哪个存储节点。注释里强调："Logical-ID placement must be identical on every compute node. Dynamic routes are intentionally used for graph/search entry selection, not for authoritative identity ownership"——逻辑 ID 的归属必须在所有计算节点上一致，否则两个计算节点可能把同一个 generation 的更新发到两个 owner。这条不变量是第 28 课 storage owner 更新的基石。

**GPU 设备选择与引擎构造（57-66 行）**：

```cpp
const cudaError_t cuda_status = cudaSetDevice(static_cast<int>(config_.gpu_device));
lib_assert(cuda_status == cudaSuccess, ...);
print_status("search: GPU-persistent OPQ/PQ" + ...);
persistent_search_ = std::make_unique<gpu_search::PersistentSearchEngine>(
  config_, context_, cm_, remote_access_tokens_);
```

`cudaSetDevice` 选 GPU；`PersistentSearchEngine` 的构造签名见 `gpu_search/persistent_engine.hh:27-30`，它接受 config/context/cm/remote_tokens 四件套，内部用 PImpl（`struct Impl`，第 11 课）。引擎构造时会启动 GPU persistent kernel（见第 17、20、21 课）。

**最后握手（68-71 行）**：

```cpp
cm_.synchronize();
start_storage_nodes();
synchronize_clients_after_startup();
if (config_.enable_updates) start_storage_insert_runtime();
```

- `cm_.synchronize()`：所有计算节点之间的 barrier（见第 4 课）。
- `start_storage_nodes()`：见 27.6 节，initiator 给每个存储节点发一个 `storage_startup::Request`，等回 `storage_startup::Response{ready=true}`。
- `synchronize_clients_after_startup()`：见 27.6 节，initiator 给所有非 initiator 计算节点发 `ready=true`，非 initiator 收到才算"全集群就绪"。
- `start_storage_insert_runtime()`：启动 storage-owner 更新运行时（两条线程 + 各类 bounded queue），**第 28 课**详细讲。

构造函数返回后，主线程进入 `wait_for_shutdown_signal()` 阻塞。

### 27.4.2 析构函数

```cpp
ComputeService::~ComputeService() {
  if (config_.enable_updates) stop_storage_insert_runtime();
  persistent_search_.reset();
  cm_.server_qps.clear();
  if (config_.enable_updates) release_storage_insert_runtime();
}
```

顺序很讲究：

1. `stop_storage_insert_runtime()`：先通知 storage insert 的两条线程退出（`storage_insert_shutdown_.store(true)`），join 它们。这必须在引擎还活着的时候做，因为 runtime 里的完成回调会调用 `persistent_search_->publish_mutations`（见第 28 课）。
2. `persistent_search_.reset()`：销毁引擎，GPU persistent kernel 被取消（见第 11 课）。
3. `cm_.server_qps.clear()`：关掉与存储节点的 QP。`unique_ptr` 的析构会走 ibv destroy_qp。
4. `release_storage_insert_runtime()`：释放 runtime 占的 slot 池、queue、buffer。放在最后是因为前几步可能还要访问这些结构（比如析构 QP 时还有未完成的 WR 要 drain）。

`stop_storage_insert_runtime` 和 `release_storage_insert_runtime` 的实现都在 `storage_owner/lifecycle.cc`，**第 28 课**讲。

---

## 27.5 查询入口：`search.cc`

`search.cc` 78 行，定义了 `search_local_result` / `search_local_raw_result` / `search_local` / `search_local_raw` / `search_raw` / `search` 六个函数。它们的关系是一个层层包装的漏斗。

### 27.5.1 `search_local_result`：真正调引擎的地方

```cpp
ComputeService::LocalMainSearchOutput
ComputeService::search_local_result(const vec<element_t>& query, u32 k) {
  if (query.size() != config_.dim) {
    throw std::invalid_argument("search dimension mismatch");
  }
  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query,
    breakdown_enabled_.load(std::memory_order_acquire));
  const auto started = std::chrono::steady_clock::now();
  sample->enqueued_at = started;
  sample->mark_started(started, started);
  service::QueryResult results = persistent_search_->search(
    span<const element_t>{query.data(), query.size()}, k);
  sample->mark_finished(std::chrono::steady_clock::now());
  return {.results = std::move(results), .sample = std::move(sample)};
}
```

逐行讲：

- 维度检查：`query.size() != config_.dim` 抛 `std::invalid_argument`。注意这是**同步入口**的检查；`search_raw` 里还有一次 `dim != config_.dim` 的检查（见 27.5.4）。
- `auto sample = std::make_shared<service::breakdown::Sample>(Operation::query, breakdown_enabled_.load(acquire));`：创建一次 breakdown 采样。`Sample` 的构造函数签名是 `Sample(Operation, bool collect_fine_grained = true)`（`breakdown/sample.hh:14-15`）。这里第二个实参 `breakdown_enabled_` 是个 atomic bool，用 acquire 读——它对应"是否启用细粒度 breakdown"。`reset_breakdown_state()` 会用 release 写它（见 27.6），acquire/release 配对保证 reset 后新创建的 Sample 一定能看到新的开关状态。
- `started = steady_clock::now()`：记录开始时刻。
- `sample->enqueued_at = started;` + `sample->mark_started(started, started);`：这里有个**重要简化**。`Sample::mark_started(dequeued, started)`（`sample.hh:31-37`）会计算 `queue_wait_ns = dequeued - enqueued_at`。在真正的多线程 admission 队列里（第 14 课），`enqueued_at` 是入队时刻、`dequeued` 是被 worker 取走的时刻、`started` 是真正开始执行的时刻，三者通常不同。但 `ComputeService::search` 是**同步调用**——调用者直接阻塞等结果，没有 admission 队列——所以这里把 `enqueued_at == dequeued == started` 全设成 `started`，`queue_wait_ns = 0`。也就是说，在计算服务这一层看到的 `queue_wait_ns` 永远是 0；真正能测到排队等待的是第 30 课的 benchmark harness（它有自己的 admission 队列，会单独写 `Sample::enqueued_at`）。
- `persistent_search_->search(span<const element_t>{query.data(), query.size()}, k)`：调引擎。引擎有两个 `search` 重载（`persistent_engine.hh:36-37`）：一个接 `VectorDType + byte_t*`（raw），一个接 `std::span<const element_t>`。本函数走后者。引擎内部会把 query 上传 GPU、走 admission（见第 14 课）、走 persistent kernel 主循环（第 20 课）、走 RDMA exact rerank，最后返回 `QueryResult`（即 `vec<QueryResultItem>`，`service/query_result.hh`）。
- `sample->mark_finished(now)`：记录结束时刻，计算 `service_ns` 和 `end_to_end_ns`（`sample.hh:39-46`）。
- 返回 `{results, sample}`：结果和采样成对返回。`LocalMainSearchOutput` 是 `compute_service.hh:47-50` 定义的结构体。用 `std::move` 避免拷贝。

### 27.5.2 `search_local_raw_result`：raw 指针版本

```cpp
ComputeService::LocalMainSearchOutput
ComputeService::search_local_raw_result(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (query_data == nullptr) throw std::invalid_argument("raw query pointer is null");
  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query,
    breakdown_enabled_.load(std::memory_order_acquire));
  const auto started = std::chrono::steady_clock::now();
  sample->enqueued_at = started;
  sample->mark_started(started, started);
  service::QueryResult results = persistent_search_->search(query_dtype, query_data, k);
  sample->mark_finished(std::chrono::steady_clock::now());
  return {.results = std::move(results), .sample = std::move(sample)};
}
```

与 `search_local_result` 几乎完全对称，区别只有两点：

1. 入参是 `(VectorDType, const byte_t*)`，绕开 `vec<element_t>` 的拷贝，benchmark 直接喂原始 buffer（见第 30 课）。
2. 调 `persistent_search_->search(query_dtype, query_data, k)` 这个重载（`persistent_engine.hh:36`）。引擎内部会按 `query_dtype`（fp32/fp16/int8 等，见第 2 课 `vector_dtype.hh`）解析 buffer。

`search_local_result` 和 `search_local_raw_result` 是**唯一两处真正调引擎的地方**，其余 `search*` 都是它们的包装。

### 27.5.3 `search_local` / `search_local_raw`：抽出 ID 并合并采样

```cpp
vec<node_t> ComputeService::search_local(const vec<element_t>& query, u32 k) {
  LocalMainSearchOutput output = search_local_result(query, k);
  vec<node_t> ids;
  ids.reserve(std::min<size_t>(k, output.results.size()));
  for (const service::QueryResultItem& result : output.results) {
    if (ids.size() == k) break;
    ids.push_back(result.id);
  }
  if (output.sample && output.sample->finished_flag) {
    std::lock_guard<std::mutex> lock(breakdown_mutex_);
    service::breakdown::add_sample(
      completed_breakdown_report_.query, *output.sample);
  }
  return ids;
}
```

逻辑：

- 调 `search_local_result` 拿到 `{results, sample}`。
- `reserve(min(k, results.size()))`：避免多次扩容。
- 遍历 `QueryResultItem`，只取 `id`（丢掉 `distance`），最多取 k 个。`QueryResultItem` 在 `service/query_result.hh` 是 `{ node_t id; distance_t distance; }`。
- **采样合并**：`if (output.sample && output.sample->finished_flag)`——`finished_flag` 在 `mark_finished` 里被置 true（`sample.hh:40`）。加锁 `breakdown_mutex_` 后调 `service::breakdown::add_sample(completed_breakdown_report_.query, *output.sample)`。`add_sample` 在 `breakdown/aggregate.hh:54-81`，它把 `Sample` 累加进 `Aggregate`（count+1、累加 latency、更新 reservoir、累加 subcategory）。`completed_breakdown_report_` 是 `Report`，含 `query`/`insert` 两个 `Aggregate`（`aggregate.hh:46-52`）。这里只更新 `query` 那个。
- 返回 `ids`。

`search_local_raw` 与 `search_local` 完全对称，只是调 `search_local_raw_result`。

### 27.5.4 `search_raw` / `search`：对外 API

```cpp
vec<node_t> ComputeService::search_raw(
    VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k) {
  if (dim != config_.dim) throw std::invalid_argument("raw search dimension mismatch");
  return search_local_raw(query_dtype, query_data, k);
}

vec<node_t> ComputeService::search(const vec<element_t>& query, u32 k) {
  return search_local(query, k);
}
```

- `search_raw` 多一道维度检查（`dim != config_.dim`），然后转 `search_local_raw`。注意 `search_local_raw` 内部**不再**重复检查维度，因为 `search_local_raw_result` 调的是 raw 重载，引擎那边会按字节解释，所以必须在入口拦。
- `search` 就是 `search_local` 的别名，提供给"已经持有 `vec<element_t>`"的调用方。

### 27.5.5 与第 14 课 admission 的衔接

`search_local_result` 调 `persistent_search_->search(...)` 是同步阻塞调用，但引擎内部并不是真的"同步"——它把 query 投递给 GPU persistent kernel 的 admission 队列（第 14 课 `CompletionDescriptor`），然后等 kernel 把结果写回 completion pool 才返回。所以：

- `ComputeService` 这一层看到的 `service_ns` ≈ admission 排队 + kernel 执行 + RDMA rerank + completion 回收。
- 真正细粒度的阶段分解（各阶段 cycle 数）由 `CompletionDescriptor` 在引擎内部记录（第 9 课），再被第 30 课的 benchmark harness 通过 `gpu_search_telemetry()` 或更底层的 telemetry 拉出来。
- `Sample` 在这一层只能拿到"端到端"的 `service_ns` / `end_to_end_ns`；`subcategory_ns` 在 query 路径上其实不会被填充（query 路径没调 `Sample::add_subcategory`），细粒度子项主要服务于 storage-owner RPC（见第 28 课）。

### 27.5.6 请求处理时序图

```
调用方                ComputeService            PersistentSearchEngine         GPU persistent kernel
  |                         |                            |                              |
  | search(query, k)        |                            |                              |
  |------------------------>|                            |                              |
  |                         | search_local:              |                              |
  |                         |  Sample(enqueued=now)      |                              |
  |                         |  mark_started(now, now)    |                              |
  |                         |  search(span, k) ---------->|                              |
  |                         |                            | admission enqueue            |
  |                         |                            |----------------------------->|
  |                         |                            |                              | kernel picks up,
  |                         |                            |                              | graph traversal (第20课)
  |                         |                            |                              | RDMA rerank (第19课)
  |                         |                            |<-----------------------------|
  |                         |                            | completion pool signal       |
  |                         |  mark_finished(now)        |                              |
  |                         |  add_sample(report.query)  |                              |
  |  <--- ids --------------|                            |                              |
  |                         |                            |                              |
```

---

## 27.6 `index_commands.cc`：状态、compute-side idmap、元数据校验、启动握手

`index_commands.cc` 222 行，承载了五件不直接相关但都属于"非查询命令"的事：

1. `status()`：对外状态。
2. `publish_compute_side_id` / `lookup_compute_side_id` / `known_storage_owner_for_id` / `claim_storage_owner_for_mutation`：compute-side idmap 的四把操作。
3. `reset_breakdown_state` / `clear_thread_statistics` / `collect_breakdown_report`：breakdown 状态管理。
4. `init_remote_tokens` / `receive_remote_access_tokens` / `start_storage_nodes` / `synchronize_clients_after_startup`：启动握手。
5. `validate_index_metadata`：索引元数据校验（最长，70 行）。

### 27.6.1 `status()`

```cpp
ComputeService::Status ComputeService::status() const {
  return {
    .state = "running",
    .vectors_inserted = vectors_inserted_.load(std::memory_order_relaxed),
    .dimension = config_.dim,
    .threads = config_.num_threads,
  };
}
```

返回 `Status`（`compute_service.hh:40-45`）。`vectors_inserted_` 是 atomic，用 relaxed 读——它只是个统计计数，不做同步。`state` 字段恒为 `"running"`（构造完即 running，析构即销毁，没有中间状态）。

### 27.6.2 compute-side idmap 四把操作

`compute_side_idmap_` 是 `std::array<ComputeSideIdShard, 256>`，每个 shard 是 `{ mutex; hashmap_t<node_t, ComputeSideIdEntry>; }`（`compute_service.hh:213-217`）。`ComputeSideIdEntry` 是 `{ RemotePtr ptr; bool deleted; u32 owner_storage; u32 generation; }`（206-211 行）。这是第 28 课"计算侧 storage owner 更新"的核心状态，本课讲四把操作的语义，运行时如何驱动它们留到第 28 课。

**`publish_compute_side_id`**（`index_commands.cc:14-29`）：

```cpp
bool ComputeService::publish_compute_side_id(node_t id,
                                             RemotePtr ptr,
                                             bool deleted,
                                             u32 owner_storage,
                                             u32 generation) {
  auto& shard = compute_side_idmap_[static_cast<size_t>(id) % kComputeSideIdShardCount];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto existing = shard.entries.find(id);
  if (existing != shard.entries.end() &&
      existing->second.generation >= generation) {
    return false;
  }
  shard.entries[id] = ComputeSideIdEntry{ptr, deleted, owner_storage, generation};
  return true;
}
```

- 按 `id % 256` 分片。
- 加锁后查现有条目；若现有 `generation >= 新 generation`，**拒绝**（返回 false）。这是"单调 generation"不变量——同一个 ID 的更新只能按 generation 递增顺序生效。这就解释了 `lifecycle.cc:46-52` 那条注释为什么要求"逻辑 ID 归属在所有计算节点上一致"：如果两个计算节点对同一个 ID 各自递增 generation 并发到不同 owner，就会在合并时出现 generation 冲突。
- 否则覆盖，返回 true。

**`lookup_compute_side_id`**（31-40 行）：纯读，加锁后查到就填 `ptr`/`deleted`，返回 true；否则 false。注意 `deleted` 字段表示"这个 ID 被软删除了"——查询路径在拼远端指针时要避开 deleted 条目。

**`known_storage_owner_for_id`**（42-52 行）：

```cpp
std::optional<u32> ComputeService::known_storage_owner_for_id(
    node_t id) const {
  {
    const auto& shard = compute_side_idmap_[...];
    std::lock_guard<std::mutex> lock(shard.mutex);
    const auto it = shard.entries.find(id);
    if (it != shard.entries.end()) return it->second.owner_storage;
  }
  return base_owner_map_.owner_for(id);
}
```

先查 runtime map（`compute_side_idmap_`），没有再回退到 `base_owner_map_`（不可变基线，第 8 课）。这给出了"这个 ID 当前归谁管"的 best-effort 答案。返回 `optional<u32>`，可能为空（ID 完全未知）。

**`claim_storage_owner_for_mutation`**（54-75 行）：

```cpp
u32 ComputeService::claim_storage_owner_for_mutation(node_t id, u32 proposed_owner) {
  auto& shard = compute_side_idmap_[...];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto existing = shard.entries.find(id);
  if (existing != shard.entries.end()) {
    return existing->second.owner_storage;
  }
  if (const auto base_owner = base_owner_map_.owner_for(id)) {
    return *base_owner;
  }
  // Generation zero is a local routing claim, not a published mutation.
  shard.emplace(
    id, ComputeSideIdEntry{RemotePtr{}, true, proposed_owner, 0});
  return proposed_owner;
}
```

这是"为一次 mutation 选定 owner"的逻辑：

1. 如果 runtime map 已有，用现有 owner。
2. 否则查 base owner map；有就用 base owner。
3. 否则（全新 ID），用调用方提议的 `proposed_owner`，并写一条 `generation=0` 的占位条目（`RemotePtr{}` 空、`deleted=true`）。

注释（63-72 行）解释了 generation=0 的含义：它是一个"本地路由占位"，不是真正的发布——第一次成功的 storage 响应会以 generation=1 覆盖它。这一步关闭了"同一个 ID 的并发首次 mutation 在本计算节点上选到不同 owner"的窗口：因为 `claim` 是加锁的，第一个 claimer 写了 generation=0 占位后，第二个 claimer 走分支 1 拿到同一个 `proposed_owner`。

这四把操作是第 28 课"计算侧 storage owner 更新"子系统的入口，本课点到为止。

### 27.6.3 breakdown 状态管理

```cpp
void ComputeService::reset_breakdown_state() {
  std::lock_guard<std::mutex> lock(breakdown_mutex_);
  completed_breakdown_report_ = {};
  breakdown_enabled_.store(config_.enable_breakdown,
                           std::memory_order_release);
  persistent_search_->reset_telemetry();
  storage_insert_late_rpc_completions_.store(0, std::memory_order_relaxed);
}

void ComputeService::clear_thread_statistics() {
}

service::breakdown::Report ComputeService::collect_breakdown_report() const {
  std::lock_guard<std::mutex> lock(breakdown_mutex_);
  return completed_breakdown_report_;
}
```

- `reset_breakdown_state`：清空 report，把 `breakdown_enabled_` 重置为 `config_.enable_breakdown`（默认 true，见第 2 课 `configuration.hh:39`），调引擎的 `reset_telemetry()`（`persistent_engine.hh:49`，第 11 课），清零"迟到的 storage-owner RPC completion"计数。这是第 30 课 benchmark 在每个测量区间开始时调的——清掉 warmup 期间的统计。
- `clear_thread_statistics`：当前是空实现，留作未来扩展（比如清线程本地计数器）。
- `collect_breakdown_report`：加锁拷贝返回当前 report。注意 `Report` 含两个 `Aggregate`，每个 `Aggregate` 含两个 `std::vector<u64>`（latency reservoir），拷贝不便宜但 benchmark 只在测量结束时调一次。

### 27.6.4 `init_remote_tokens` / `receive_remote_access_tokens`

已在 27.4.1 讲过，这里补一句：`MemoryRegionToken` 在 `rdma-library/library/memory_region.hh:9` 定义，是远端 MR 的访问凭据（rkey + 地址 + 长度），`MRT = u_ptr<MemoryRegionToken>`，`MemoryRegionTokens = vec<MRT>`。`receive_remote_access_tokens` 用 RDMA recv 接收存储节点发来的 token，**不是** RDMA read——token 是元数据，必须先通过 send/recv 交换，之后才能用 RDMA read 读存储节点上的实际数据。

### 27.6.5 `start_storage_nodes`：唤醒存储节点

```cpp
void ComputeService::start_storage_nodes() {
  if (!cm_.is_initiator) return;
  for (u32 server = 0; server < cm_.server_qps.size(); ++server) {
    storage_startup::Request request{};
    const QP& qp = cm_.server_qps[server];
    qp->post_send_inlined(&request, sizeof(request), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
  for (u32 server = 0; server < cm_.server_qps.size(); ++server) {
    storage_startup::Response response{};
    LocalMemoryRegion response_region{context_, &response, sizeof(response)};
    cm_.server_qps[server]->post_receive(response_region);
    context_.receive();
    lib_assert(response.ready,
               "storage startup failed on node " + std::to_string(server));
  }
}
```

只有 initiator 执行（`if (!cm_.is_initiator) return;`）。两轮：

1. 给每个存储节点发一个 `storage_startup::Request`（空结构，只是一个触发信号），同步等 send 完成。
2. 从每个存储节点 recv 一个 `storage_startup::Response`，断言 `response.ready == true`。

这对应第 23 课存储节点主体的"等 initiator 唤醒 → 加载索引 → 回 ready"流程。注意非 initiator 计算节点不参与这一步——它们只等 initiator 的 `synchronize_clients_after_startup`。

### 27.6.6 `synchronize_clients_after_startup`：计算节点间 barrier

```cpp
void ComputeService::synchronize_clients_after_startup() {
  constexpr bool ready = true;
  if (cm_.is_initiator) {
    for (const QP& qp : cm_.client_qps) {
      qp->post_send_inlined(&ready, sizeof(ready), IBV_WR_SEND);
    }
    if (!cm_.client_qps.empty()) {
      context_.poll_send_cq_until_completion(
        static_cast<i32>(cm_.client_qps.size()));
    }
  } else {
    bool initiator_ready{};
    LocalMemoryRegion region{context_, &initiator_ready, sizeof(initiator_ready)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();
    lib_assert(initiator_ready, "initiator startup synchronization failed");
  }
}
```

- initiator：给所有非 initiator 计算节点 send 一个 `ready=true`，然后**一次性** poll 所有 send 完成（`poll_send_cq_until_completion(n)` 一次等 n 个 CQE，比循环 n 次更高效）。
- 非 initiator：从 `initiator_qp` recv 一个 bool，断言为 true。

这是一个"initiator 等所有存储节点 ready 后再放行所有计算节点"的二级握手——确保所有计算节点在同一时刻进入"可服务"状态。在多计算节点 benchmark（第 30 课）里这一步很关键，否则先就绪的计算节点会抢先发 query，污染统计。

### 27.6.7 `validate_index_metadata`：schema-15 不变量校验

这是本文件最长也最关键的函数（130-203 行，70 余行）。它做两件事：**校验**索引元数据与当前配置/二进制兼容；**副作用**——配置 `VamanaNode` 的静态布局。

```cpp
bool ComputeService::validate_index_metadata(
    const filepath_t& index_prefix, str* error_message) {
  service::index_metadata::Metadata metadata;
  if (!service::index_metadata::load_metadata(index_prefix, metadata, error_message)) {
    return false;
  }
  const bool compatible_quantizer = metadata.navigation_quantizer == "opq_pq" ||
    metadata.navigation_quantizer == "opq_pq16";
  const bool compatible_navigation = metadata.navigation_format == "opq_pq_graph_v1" ||
    metadata.navigation_format == "opq_pq16_graph_v1";
  if (metadata.schema_version != gpu_search::format::kMetadataSchemaVersion ||
      metadata.node_layout != "plain" ||
      metadata.storage_format != "vamana_compact_v1" ||
      !compatible_quantizer || !compatible_navigation ||
      metadata.navigation_code_bytes == 0 ||
      metadata.navigation_code_bytes != metadata.pq_subquantizers ||
      metadata.pq_bits != 8 || metadata.navigation_model_checksum == 0 ||
      metadata.dim != config_.dim || metadata.R != config_.R ||
      metadata.num_memory_nodes != num_servers_) {
    if (error_message != nullptr) {
      *error_message = "index is not a compatible schema-15 OPQ/PQ GPU index";
    }
    return false;
  }
  ...
}
```

第一段校验：

- `schema_version == kMetadataSchemaVersion`：schema 版本（第 7 课 schema-15）。
- `node_layout == "plain"`、`storage_format == "vamana_compact_v1"`：节点布局与存储格式（第 6 课 Vamana、第 7 课 schema-15）。
- `navigation_quantizer ∈ {"opq_pq", "opq_pq16"}`、`navigation_format ∈ {"opq_pq_graph_v1", "opq_pq16_graph_v1"}`：导航量化器与图格式必须是 OPQ/PQ 系列。
- `navigation_code_bytes == pq_subquantizers`、`pq_bits == 8`：每个子量化器 1 字节、8 bit。
- `navigation_model_checksum != 0`：必须有 PQ 模型。
- `dim == config_.dim`、`R == config_.R`：维度与图度数必须与配置一致。
- `num_memory_nodes == num_servers_`：索引构建时的存储节点数必须与当前集群一致——这是 schema-15 索引"分片到 N 个存储节点"的硬性约束。

任何一条不满足，填错误信息返回 false，构造函数里的 `lib_assert` 会抛异常终止。

第二段校验 dtype：

```cpp
if (config_.vector_data_type != "auto" &&
    config_.resolved_vector_dtype() != metadata.vector_dtype) {
  if (error_message != nullptr) *error_message = "index vector dtype mismatch";
  return false;
}
config_.vector_data_type = vector_dtype_name(metadata.vector_dtype);
```

如果配置显式指定了 dtype 且与索引不符，拒绝；否则把配置的 dtype 改写为索引的 dtype（"auto" 模式以索引为准）。

第三段——配置 `VamanaNode` 静态布局：

```cpp
VamanaNode::disable_hot_graph();
VamanaNode::init_static_storage(config_.dim, config_.R, metadata.vector_dtype);
```

`disable_hot_graph()` 先关掉 hot graph（确保 `init_static_storage` 用基础布局初始化），再用 dim/R/dtype 初始化 `VamanaNode` 的静态成员（节点大小、向量偏移等，第 6 课）。

第四段——存储布局校验（162-185 行）：把 `metadata` 里所有"布局数字"与 `VamanaNode::xxx()` 静态方法返回值逐项比对：

```cpp
if (metadata.vector_component_size != VamanaNode::vector_component_size() ||
    metadata.vector_bytes != VamanaNode::vector_bytes() ||
    metadata.node_size != VamanaNode::total_size() ||
    metadata.graph_hot_bytes != VamanaNode::graph_hot_bytes() ||
    metadata.vector_offset != VamanaNode::offset_vector() ||
    metadata.hot_graph_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
    metadata.hot_graph_entry_size != VamanaNode::hot_graph_entry_size() ||
    metadata.hot_graph_offsets.size() != num_servers_ ||
    metadata.hot_graph_entry_counts.size() != num_servers_ ||
    metadata.hot_graph_dynamic_base_offsets.size() != num_servers_ ||
    metadata.storage_control_remote_offsets.size() != num_servers_ ||
    metadata.dynamic_node_base_offsets.size() != num_servers_ ||
    metadata.navigation_code_remote_offsets.size() != num_servers_ ||
    metadata.navigation_code_region_bytes.size() != num_servers_ ||
    metadata.hot_graph_dynamic_record_bytes <
      metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size ||
    metadata.hot_graph_dynamic_hot_offset < VamanaNode::total_size() ||
    metadata.dynamic_navigation_code_offset <
      metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size ||
    metadata.hot_graph_dynamic_record_bytes <
      metadata.dynamic_navigation_code_offset + metadata.navigation_code_bytes) {
  if (error_message != nullptr) *error_message = "index storage layout mismatch";
  return false;
}
```

这一长串校验的是 schema-15 的存储布局不变量（第 7、8 课）：

- 每个分片相关的 offset/size 数组都必须长度 == `num_servers_`。
- hot graph 的紧凑指针字节数、entry size 必须与二进制编译期常量一致。
- hot graph 的 dynamic region 布局必须自洽：`hot_graph_dynamic_record_bytes >= hot_graph_dynamic_hot_offset + hot_graph_entry_size`、`hot_graph_dynamic_hot_offset >= VamanaNode::total_size()`、`dynamic_navigation_code_offset >= hot_graph_dynamic_hot_offset + hot_graph_entry_size`、`hot_graph_dynamic_record_bytes >= dynamic_navigation_code_offset + navigation_code_bytes`。

任何一条不满足都报"index storage layout mismatch"。

第五段——配置 hot graph：

```cpp
VamanaNode::configure_hot_graph(
  metadata.hot_graph_offsets, metadata.hot_graph_entry_counts,
  metadata.hot_graph_entry_size, metadata.hot_graph_shard_bits,
  metadata.dynamic_node_base_offsets,
  metadata.hot_graph_dynamic_record_bytes,
  metadata.hot_graph_dynamic_hot_offset,
  metadata.dynamic_navigation_code_offset,
  metadata.navigation_code_bytes);
if (!VamanaNode::HAS_HOT_GRAPH) {
  if (error_message != nullptr) *error_message = "failed to enable compact graph layout";
  return false;
}
```

把元数据里的 hot graph 几何喂给 `VamanaNode::configure_hot_graph`，让 `VamanaNode::HAS_HOT_GRAPH` 变 true。如果配置失败，报错返回 false。

最后打印一行就绪信息，返回 true。

这个函数本质上是"schema-15 索引与运行时二进行的兼容性契约"——任何一项 layout 数字对不上，运行时访问远端节点就会读到错位的内存，所以必须在启动期一次性校验。

---

## 27.7 `detail.hh`：模块内部共享声明

`detail.hh`（36 行）是 `compute_service/*.cc` 共用的内部头：

```cpp
#pragma once

#include "service/compute_service.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "service/storage_owner_client_helpers.hh"
#include "vamana/idmap.hh"

namespace compute_service_detail {

inline constexpr u32 kRpcMagic = 0x53484e57;
inline constexpr u32 kRpcVersion = 1;
inline constexpr u32 kInitialRpcRecvsPerPeer = 8;
inline constexpr u32 kMaxRpcResults = 512;

using service::storage_owner_client::add_storage_owner_breakdown;
using service::storage_owner_client::add_storage_owner_sender_breakdown;
using service::storage_owner_client::duration_ns;
using service::storage_owner_client::duration_ns_clamped;
using service::storage_owner_client::per_item_ns;
using service::storage_owner_client::storage_owner_wr_id;

}  // namespace compute_service_detail
```

关键点：

- 它 include 了 `compute_service.hh`，所以所有 `.cc` 只需要 `#include "service/compute_service/detail.hh"` 就能拿到完整 `ComputeService` 定义 + 标准库 + cuda_runtime + 一组 storage-owner 客户端辅助（`storage_owner_client_helpers.hh`）+ idmap。
- 常量：
  - `kRpcMagic = 0x53484e47`：ASCII "SHNW"（"Storage Handler NetWork"?），storage-owner RPC 的魔数，第 24 课 peer RPC 协议头里会校验。
  - `kRpcVersion = 1`：协议版本。
  - `kInitialRpcRecvsPerPeer = 8`：每个 peer 初始 post 的 recv 数。
  - `kMaxRpcResults = 512`：单次 RPC 最多返回的结果数。
- using 声明：把 `storage_owner_client` 命名空间下的几个工具函数（duration 计算、breakdown 累加、WR ID 编码）拉到 `compute_service_detail` 命名空间里，让 `storage_owner/*.cc` 写起来短一点。这些工具的实现和语义在第 28 课讲。

`lifecycle.cc`、`search.cc`、`index_commands.cc` 都在文件顶部 `using namespace compute_service_detail;`，所以可以直接写 `kRpcMagic`、`duration_ns` 等而不用加前缀。

---

## 27.8 breakdown 子系统

breakdown 子系统是"per-request 采样 → 聚合 → 报表"三层结构，全部在 header-only 的小文件里实现。本课讲它的骨架，第 30 课讲 benchmark 如何驱动它。

### 27.8.1 `breakdown.hh`：汇总入口

```cpp
#pragma once

#include "service/breakdown/names.hh"
#include "service/breakdown/sample.hh"
#include "service/breakdown/aggregate.hh"
#include "service/breakdown/json.hh"
#include "service/breakdown/text.hh"
```

7 行，纯 include。`ComputeService` 在 `compute_service.hh:24` include 了这个头，所以一次性拿到 `Sample`/`Aggregate`/`Report`/`add_sample`/`report_to_json`/`aggregate_text_summary` 全部符号。

### 27.8.2 `names.hh`：Operation/Category/Subcategory 枚举

```cpp
enum class Operation : u8 { query = 0, insert = 1 };

enum class Category : u8 { cpu = 0, rdma, count };
constexpr size_t kCategoryCount = static_cast<size_t>(Category::count);

inline constexpr std::array<std::string_view, kCategoryCount> kCategoryNames = {
  "cpu_ns",
  "rdma_ns",
};

enum class Subcategory : u8 {
  cpu_storage_owner_queue_wait = 0,
  cpu_storage_owner_search,
  ...
  rdma_storage_owner_medoid,
  rdma_storage_owner_send,
  ...
  count
};
```

- `Operation`：只有 query/insert 两种。
- `Category`：cpu/rdma 两类，`count` 是哨兵。
- `Subcategory`：33 个子类（CPU 29 个 + RDMA 5 个），全部围绕"storage-owner"操作细分。注意 **query 路径上的细粒度子项几乎不在这里**——query 的细粒度阶段在第 9 课 `CompletionDescriptor` 里用 GPU cycle 记录，走 telemetry；这里的 subcategory 服务于 storage-owner RPC（第 24、28 课）。这也解释了为什么 `search_local` 里 `Sample` 没调 `add_subcategory`。

`parent_category`（102-106 行）：subcategory ≥ `rdma_storage_owner_medoid` 且 < `count` 归 rdma，否则归 cpu。这依赖枚举值的排列顺序（CPU 子类在前，RDMA 在后）。

`kSubcategoryNames`（61-96 行）：每个 subcategory 对应一个字符串名，用于 JSON/text 报表输出。

### 27.8.3 `sample.hh`：per-request `Sample`

```cpp
struct Sample {
  Sample() : Sample(Operation::insert, false) {}
  explicit Sample(Operation operation, bool collect_fine_grained = true)
      : operation(operation), collect_fine_grained_breakdown(collect_fine_grained) {}

  Operation operation;
  bool collect_fine_grained_breakdown{};
  Clock::time_point enqueued_at{};
  Clock::time_point dequeued_at{};
  Clock::time_point started_at{};
  Clock::time_point finished_at{};
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};
  u64 queue_wait_ns{};
  u64 service_ns{};
  u64 end_to_end_ns{};
  bool started_flag{};
  bool finished_flag{};

  void mark_started(Clock::time_point dequeued, Clock::time_point started) {
    dequeued_at = dequeued;
    started_at = started;
    started_flag = true;
    queue_wait_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(dequeued_at - enqueued_at).count());
  }

  void mark_finished(Clock::time_point finished) {
    finished_at = finished;
    finished_flag = true;
    service_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(finished_at - started_at).count());
    end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(finished_at - enqueued_at).count());
  }

  [[nodiscard]] bool collects_breakdown() const {
    return collect_fine_grained_breakdown;
  }

  void add_subcategory(Subcategory subcategory, u64 nanoseconds) {
    if (!collect_fine_grained_breakdown) return;
    subcategory_ns[static_cast<size_t>(subcategory)] += nanoseconds;
    category_ns[static_cast<size_t>(parent_category(subcategory))] += nanoseconds;
  }
};
```

- 四个时间点：`enqueued_at`（入队）/`dequeued_at`（被 worker 取走）/`started_at`（开始执行）/`finished_at`（完成）。在 `ComputeService::search_local_result` 里前三个都设成 `started`，所以 `queue_wait_ns = 0`；只有第 30 课的 benchmark harness 会真正区分它们。
- 三个累计 ns：`queue_wait_ns`（排队等待）、`service_ns`（服务时间）、`end_to_end_ns`（端到端）。
- `category_ns` / `subcategory_ns`：细粒度分解。只有 `collect_fine_grained_breakdown == true` 时才累加（`add_subcategory` 内部短路）。
- `mark_started` / `mark_finished`：状态机，置 flag 并计算 ns。
- 默认构造是 `Sample(insert, false)`——给 storage-owner RPC 用，默认不采细粒度（性能优先）。query 路径显式传 `Operation::query` 和 `breakdown_enabled_`。

### 27.8.4 `aggregate.hh`：`Aggregate`/`Report` + 聚合

```cpp
inline constexpr size_t kLatencyReservoirCapacity = 1u << 18;   // 262144

inline u64 reservoir_hash(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct Aggregate {
  Operation operation{Operation::query};
  size_t count{};
  u64 total_queue_wait_ns{};
  u64 total_service_ns{};
  u64 total_end_to_end_ns{};
  bool fine_grained_breakdown_observed{};
  std::vector<u64> end_to_end_latencies_ns;
  std::vector<u64> service_latencies_ns;
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};

  [[nodiscard]] u64 cpu_other_ns() const {
    u64 explicit_cpu = 0;
    for (size_t index = 0; index < subcategory_ns.size(); ++index) {
      if (parent_category(static_cast<Subcategory>(index)) == Category::cpu) {
        explicit_cpu += subcategory_ns[index];
      }
    }
    const u64 rdma = category_ns[static_cast<size_t>(Category::rdma)];
    const u64 cpu_total = total_service_ns > rdma ? total_service_ns - rdma : 0;
    return cpu_total > explicit_cpu ? cpu_total - explicit_cpu : 0;
  }
};

struct Report {
  Aggregate query{};
  Aggregate insert{};

  [[nodiscard]] bool has_query() const { return query.count > 0; }
  [[nodiscard]] bool has_insert() const { return insert.count > 0; }
};
```

- `kLatencyReservoirCapacity = 1<<18 = 262144`：reservoir 采样容量。一旦 latency 样本超过这个数，就用 `reservoir_hash` 做替换（见 `add_sample`）。
- `reservoir_hash`：一个整数 hash（splitmix64 风格），用于 reservoir sampling 的"随机替换位置"。
- `Aggregate`：单个 Operation 的聚合。count、总 latency、reservoir（两个 vector，分别存 end_to_end 和 service 的 latency 样本）、category/subcategory 累加。
- `cpu_other_ns()`：把"总 service 时间减去 rdma 时间"得到 cpu 总时间，再减去所有显式 cpu 子类累加，得到"未分类的 cpu 开销"——这是 query 路径上唯一能拿到的"cpu runtime overhead"数字，因为 query 不填 subcategory。
- `Report`：query + insert 两个 `Aggregate`。

`add_sample`（54-81 行）：

```cpp
inline void add_sample(Aggregate& aggregate, const Sample& sample) {
  if (!sample.finished_flag) return;
  aggregate.operation = sample.operation;
  ++aggregate.count;
  aggregate.total_queue_wait_ns += sample.queue_wait_ns;
  aggregate.total_service_ns += sample.service_ns;
  aggregate.total_end_to_end_ns += sample.end_to_end_ns;
  if (aggregate.end_to_end_latencies_ns.size() < kLatencyReservoirCapacity) {
    aggregate.end_to_end_latencies_ns.push_back(sample.end_to_end_ns);
    aggregate.service_latencies_ns.push_back(sample.service_ns);
  } else {
    const size_t replacement = static_cast<size_t>(
      reservoir_hash(static_cast<u64>(aggregate.count)) % aggregate.count);
    if (replacement < kLatencyReservoirCapacity) {
      aggregate.end_to_end_latencies_ns[replacement] = sample.end_to_end_ns;
      aggregate.service_latencies_ns[replacement] = sample.service_ns;
    }
  }
  aggregate.fine_grained_breakdown_observed =
    aggregate.fine_grained_breakdown_observed || sample.collects_breakdown();
  if (!sample.collects_breakdown()) return;
  for (size_t index = 0; index < aggregate.category_ns.size(); ++index) {
    aggregate.category_ns[index] += sample.category_ns[index];
  }
  for (size_t index = 0; index < aggregate.subcategory_ns.size(); ++index) {
    aggregate.subcategory_ns[index] += sample.subcategory_ns[index];
  }
}
```

经典 reservoir sampling：前 262144 个样本直接 push；之后对第 k 个样本（k > capacity），用 `reservoir_hash(k) % k` 决定是否替换——如果落在 [0, capacity) 就替换对应位置。这保证了任意时刻 reservoir 都是总体 latency 的均匀采样，可用于 p50/p95/p99 估计。

`percentile_ns`（87-92 行）：排序后按 `percentile * (n-1)` 取下标。注意它**按值传 vector**——每次调用都拷贝再排序，所以调用方应该缓存结果。`ns_to_ms`（83-85 行）：简单除 1e6。

### 27.8.5 `json.hh`：JSON 报表

```cpp
inline nlohmann::json aggregate_to_json(const Aggregate& aggregate) {
  using json = nlohmann::json;
  json output;
  output["operation"] = operation_name(aggregate.operation);
  output["count"] = aggregate.count;
  output["latency"] = {
    {"queue_wait_ns", aggregate.total_queue_wait_ns},
    {"service_ns", aggregate.total_service_ns},
    {"end_to_end_ns", aggregate.total_end_to_end_ns},
    {"mean_queue_wait_ns", aggregate.count == 0 ? 0 :
      aggregate.total_queue_wait_ns / aggregate.count},
    {"mean_service_ns", aggregate.count == 0 ? 0 :
      aggregate.total_service_ns / aggregate.count},
    {"mean_end_to_end_ns", aggregate.count == 0 ? 0 :
      aggregate.total_end_to_end_ns / aggregate.count},
    {"p50_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.50)},
    {"p95_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.95)},
    {"p99_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)},
    {"p50_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.50)},
    {"p95_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.95)},
    {"p99_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.99)},
  };
  output["fine_grained_breakdown_observed"] =
    aggregate.fine_grained_breakdown_observed;
  if (!aggregate.fine_grained_breakdown_observed) return output;

  const u64 rdma_ns = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  output["breakdown"] = {
    {"cpu_ns", aggregate.total_service_ns > rdma_ns
      ? aggregate.total_service_ns - rdma_ns : 0},
    {"rdma_ns", rdma_ns},
  };

  json subcategories = json::object();
  for (size_t category = 0; category < kCategoryCount; ++category) {
    subcategories[std::string{kCategoryNames[category]}] = json::object();
  }
  for (size_t index = 0; index < kSubcategoryCount; ++index) {
    const auto subcategory = static_cast<Subcategory>(index);
    subcategories[std::string{kCategoryNames[
      static_cast<size_t>(parent_category(subcategory))]}]
      [std::string{kSubcategoryNames[index]}] = aggregate.subcategory_ns[index];
  }
  subcategories["cpu_ns"][aggregate.operation == Operation::query
    ? "cpu_query_runtime_overhead_ns" : "cpu_insert_runtime_overhead_ns"] =
      aggregate.cpu_other_ns();
  output["sub_breakdown"] = std::move(subcategories);
  return output;
}

inline nlohmann::json report_to_json(const Report& report) {
  nlohmann::json output = nlohmann::json::object();
  if (report.has_query()) output["query_breakdown"] = aggregate_to_json(report.query);
  if (report.has_insert()) output["insert_breakdown"] = aggregate_to_json(report.insert);
  return output;
}
```

输出结构：

```
{
  "query_breakdown": {
    "operation": "query",
    "count": N,
    "latency": {
      "queue_wait_ns", "service_ns", "end_to_end_ns",
      "mean_*_ns", "p50/p95/p99_end_to_end_ns", "p50/p95/p99_service_ns"
    },
    "fine_grained_breakdown_observed": true/false,
    "breakdown": {"cpu_ns", "rdma_ns"},            // 仅当 fine_grained_observed
    "sub_breakdown": {
      "cpu_ns": { "<subcategory>_ns": ..., "cpu_query_runtime_overhead_ns": ... },
      "rdma_ns": { "<subcategory>_ns": ... }
    }
  },
  "insert_breakdown": { ... 同构 ... }
}
```

注意 `cpu_query_runtime_overhead_ns` / `cpu_insert_runtime_overhead_ns` 这一项是 `cpu_other_ns()`，即"未分类的 cpu 开销"，对应 query 路径上 `service_ns - rdma_ns`（因为 query 没填任何 cpu 子类）。这一项是第 30 课 benchmark 评估"GPU runtime 自身开销"的关键指标。

`report_to_json` 只在有样本时才输出对应 key，避免空对象。

### 27.8.6 `text.hh`：文本摘要

```cpp
inline std::string aggregate_text_summary(const Aggregate& aggregate) {
  std::ostringstream output;
  output << operation_name(aggregate.operation) << " breakdown\n";
  output << "  count: " << aggregate.count << '\n';
  output << "  latency_ms: mean="
         << ns_to_ms(aggregate.count == 0 ? 0 :
              aggregate.total_end_to_end_ns / aggregate.count)
         << " p50=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.50))
         << " p95=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.95))
         << " p99=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.99))
         << '\n';
  if (!aggregate.fine_grained_breakdown_observed) {
    output << "  fine_grained_breakdown: disabled\n";
    return output.str();
  }
  const u64 rdma_ns = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  const u64 cpu_ns = aggregate.total_service_ns > rdma_ns
    ? aggregate.total_service_ns - rdma_ns : 0;
  output << "  cpu_ms: " << ns_to_ms(cpu_ns) << '\n';
  output << "  rdma_ms: " << ns_to_ms(rdma_ns) << '\n';
  return output.str();
}
```

比 JSON 简短得多，只输出 count、latency 的 mean/p50/p95/p99（毫秒），以及 fine_grained_observed 时的 cpu/rdma 总时间。这是给 stdout 日志用的快速摘要；完整数据走 JSON 落盘（第 30 课 benchmark 脚本解析 JSON 生成图表）。

---

## 27.9 关键数据结构与流程图

### 27.9.1 组件关系图

```
                +----------------------------- main.cc --------------------------------+
                |  config = IndexConfiguration(argc, argv)                              |
                |  if (config.is_server) abort()  // 必须是计算节点                      |
                |  ComputeService service{config}  -- 构造期完成所有启动                  |
                |  wait_for_shutdown_signal()      -- SIGINT/SIGTERM 阻塞                |
                +------------------------------------------------------------------------+
                                       |
                                       v
+---------------------- ComputeService (compute_service.hh) -----------------------+
|                                                                                 |
|  config_        context_       cm_               num_servers_                   |
|  core_assignment_               remote_access_tokens_                            |
|                                                                                 |
|  persistent_search_ : unique_ptr<PersistentSearchEngine>  (第 11/14/20 课)        |
|                                                                                 |
|  breakdown:  completed_breakdown_report_ : service::breakdown::Report            |
|              breakdown_enabled_ : atomic<bool>                                   |
|              breakdown_mutex_ : mutex                                            |
|                                                                                 |
|  storage-owner 更新运行时 (第 28 课):                                            |
|    storage_insert_progress_thread_ / storage_insert_completion_thread_           |
|    storage_ready_slots_ / storage_released_slots_ / storage_completion_pool_     |
|    storage_insert_owners_ : vec<unique_ptr<StorageOwnerSenderState>>             |
|    compute_side_idmap_ : array<ComputeSideIdShard, 256>                          |
|    base_owner_map_ : BaseOwnerMap  (第 8 课)                                     |
|                                                                                 |
|  对外 API:                                                                       |
|    search(query, k)            -> search_local -> search_local_result            |
|    search_raw(dtype, ptr, dim, k) -> search_local_raw -> search_local_raw_result |
|    insert/upsert/erase          -> 第 28 课                                      |
|    status / reset_breakdown_state / collect_breakdown_report                    |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+----------- PersistentSearchEngine (gpu_search/persistent_engine.hh) -------------+
|  search(span<element_t>, k)          -> service::QueryResult                    |
|  search(VectorDType, byte_t*, k)     -> service::QueryResult                    |
|  publish_mutations(...)             (第 16/28 课)                                |
|  telemetry() / reset_telemetry()    (第 9/11 课)                                 |
|  内部 PImpl + DeltaCoordinator + Telemetry + GPU persistent kernel               |
+---------------------------------------------------------------------------------+
```

### 27.9.2 启动时序图

```
main                     ComputeService ctor              storage node         other compute nodes
  |                            |                               |                        |
  | Construct(config)--------->|                               |                        |
  |                            | init_remote_tokens()          |                        |
  |                            | cm_.connect()  (QP RTS/RRTR)  |                        |
  |                            |  if initiator: broadcast      |                        |
  |                            |    Parameters{threads,qps*2}->|                        |
  |                            | receive_remote_access_tokens()|                        |
  |                            |   <-- MR tokens --------------|                        |
  |                            | validate_index_metadata()     |                        |
  |                            |   (schema-15 + VamanaNode)    |                        |
  |                            | load_metadata()               |                        |
  |                            |  if enable_updates:           |                        |
  |                            |   base_owner_map_.load()      |                        |
  |                            | cudaSetDevice()               |                        |
  |                            | persistent_search_ = make_unique<...>()                |
  |                            | cm_.synchronize()  (barrier among compute nodes)       |
  |                            | start_storage_nodes():        |                        |
  |                            |   Request ------------------->|                        |
  |                            |   <-- Response{ready=true} ---|                        |
  |                            | synchronize_clients_after_startup():                  |
  |                            |   initiator: send ready=true to all other computes    |
  |                            |   non-initiator: recv ready=true                      |
  |                            |  if enable_updates: start_storage_insert_runtime()    |
  |<-- ctor returns ----------|                               |                        |
  | wait_for_shutdown_signal()|                               |                        |
  |     (block on SIGINT/SIGTERM)                              |                        |
```

### 27.9.3 请求处理时序图（query）

见 27.5.6。

---

## 27.10 与其他模块的关系

- **第 1 课（构建系统）**：`main.cc` 同一个二进制被部署成 `dvstor_compute` 和 `dvstor_memory_node`，`config.is_server` 做身份分流。
- **第 2 课（配置）**：`ComputeService` 几乎所有行为都受 `IndexConfiguration` 控制——`enable_updates`、`enable_breakdown`、`gpu_query_slots`、`gpu_rdma_qps`、`num_threads`、`dim`、`R`、`gpu_device`、`resolved_index_prefix()`、`num_server_nodes()` 等。
- **第 3 课（并发原语与协程）**：`core_assignment_` 是 `CoreAssignment<interleaved>`，构造时调用 `restrict_current_thread_to_partition()` 把进程限制到一个 partition；`bounded::Queue`、`CompletionPool` 是第 3 课讲过的并发原语。构造函数注释解释了"不在构造线程 pin 具体核"的原因。
- **第 4、5 课（RDMA 传输库）**：`Context`、`ClientConnectionManager`、`QP`、`LocalMemoryRegion`、`MemoryRegionToken` 全部来自 `rdma-library/library/`。`cm_.connect()`、`post_send_inlined`、`post_receive`、`poll_send_cq_until_completion`、`context_.receive()` 这些是第 4、5 课讲过的原语。
- **第 6 课（Vamana 图格式）**：`VamanaNode::init_static_storage` / `configure_hot_graph` / `disable_hot_graph` / `HAS_HOT_GRAPH` 等，以及 `metadata.vector_dtype`、`R`、`dim` 的消费。
- **第 7 课（schema-15 索引格式）**：`validate_index_metadata` 是 schema-15 不变量的运行时校验者；`gpu_search::format::kMetadataSchemaVersion` 是版本常量。
- **第 8 课（元数据/owner map/存储协议）**：`service::index_metadata::Metadata` / `load_metadata`、`service::BaseOwnerMap`、`storage_startup::Request/Response`。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：`gpu_search::TelemetrySnapshot`、`persistent_search_->telemetry()` / `reset_telemetry()`；`metadata.pq_subquantizers`/`pq_bits`/`navigation_model_checksum` 的校验。`CompletionDescriptor` 各阶段 cycle 与本课 `Sample` 的关系：query 路径上 `Sample` 只记端到端 ns，细粒度阶段走 telemetry。
- **第 10 课（delta/动态路由/预算）**：`PersistentSearchEngine::publish_mutations` / `try_reserve_mutation_capacity` 等（构造期不直接调，但 `persistent_search_` 持有 `DeltaCoordinator`）。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`PersistentSearchEngine` 的 PImpl（`struct Impl`）、构造/析构、`search` 接口。
- **第 14 课（查询执行/路由/完成）**：`search` 调引擎内部走 admission → completion pool；本课 `Sample::enqueued_at == dequeued == started` 反映了"计算服务层无 admission 队列"。
- **第 17、20、21 课（kernel 启动器/查询遍历主循环/角色调度）**：`persistent_search_` 构造时启动的 GPU persistent kernel。
- **第 19 课（RDMA cache）**：引擎内部 search 时用 RDMA cache 做邻居读和 exact rerank。
- **第 23 课（存储节点主体）**：`start_storage_nodes` 的对端——存储节点等 initiator 的 `Request`、回 `Response{ready}`。
- **第 24 课（peer RPC）**：`detail.hh` 里的 `kRpcMagic`/`kRpcVersion`、`storage_owner_wr_id` 等。
- **第 28 课（计算侧 storage owner 更新）**：`insert/upsert/erase` 的实现、`storage_insert_*` 运行时、`compute_side_idmap_` 的运行时驱动、`StorageOwnerSenderState` 等。本课只讲它们的声明和 `index_commands.cc` 里的状态操作。
- **第 30 课（breakdown benchmark）**：`reset_breakdown_state` / `collect_breakdown_report` / `report_to_json` / `aggregate_text_summary` 是 benchmark harness 的调用目标；`cpu_query_runtime_overhead_ns` 是评估 GPU runtime 开销的关键指标。

---

## 27.11 小结

`ComputeService` 是计算节点的"门面 + 启动器 + breakdown 收集器"三合一：

1. **启动期**：构造函数串起了 RDMA 连接 → token 交换 → 索引元数据校验 → `VamanaNode` 静态布局配置 → GPU 设备选择 → 引擎构造 → 存储节点唤醒 → 计算节点间 barrier → storage insert 运行时启动。其中 `validate_index_metadata` 是 schema-15 索引与二进制兼容性的最后一道防线，校验了 schema 版本、量化器、维度、分片数、所有 layout offset/size 不变量。
2. **查询期**：`search` / `search_raw` 是同步阻塞入口，唯一真正调引擎的地方是 `search_local_result` / `search_local_raw_result`。每个 query 创建一个 `Sample`，由于同步调用没有 admission 队列，`queue_wait_ns = 0`；细粒度 subcategory 在 query 路径上不填，走 telemetry。
3. **breakdown**：三层 header-only 结构——`Sample`（per-request）→ `Aggregate`（reservoir sampling + category/subcategory 累加）→ `Report`（query+insert）。JSON 输出含 mean/p50/p95/p99 + cpu/rdma 总时间 + `cpu_*_runtime_overhead_ns`，文本输出是 stdout 快速摘要。
4. **storage-owner 状态**：`compute_side_idmap_`（256 片分片锁）+ `base_owner_map_`（不可变基线）共同回答"逻辑 ID 归哪个 owner"，`publish/lookup/known/claim` 四把操作维护单调 generation 不变量；真正的 RPC 运行时在第 28 课。

下一课（第 28 课）会接着讲 `src/service/compute_service/storage_owner/` 下的 `lifecycle.cc`/`sender.cc`/`completion.cc`/`public_mutations.cc`，把 `insert`/`upsert`/`erase` 的完整链路、`StorageOwnerSenderState` 的 slot 池机制、以及 `compute_side_idmap_` 在运行时如何被驱动讲完。
