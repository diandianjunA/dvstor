# 第 22 课 · GPUNetIO 传输与 probe

> 本课是 Part V 的唯一一课，也是整部教程的"分水岭"之一。从第 4、5 课的 CPU verbs 传输库出发，本课把 RDMA 的 datapath 彻底搬到 GPU 上：CQ / WQ / doorbell 全部分配在 GPU 显存里，由 CUDA kernel 直接写 WQE、直接 poll CQE、直接敲 doorbell，CPU 在查询热路径上完全不参与。这一套机制由 NVIDIA DOCA GPUNetIO 提供，dvstor 在其之上封装了 `GpuNetioPersistentTransport`，并用一个启动期 `gpunetio_read_probe_kernel` 来验证每一个 export 出来的 GPU QP 都能真正完成一次 RDMA Read。

## 本课目标与涉及文件

读完本课你应该能够回答：

1. 为什么 dvstor 要把 QP/CQ 的 datapath 搬到 GPU 显存？CPU verbs QP 在查询热路径上卡在哪里？
2. `GpuNetioPersistentTransport::Impl` 构造函数那一长串 `doca_*` 调用，每一步创建了什么对象、为什么必须按这个顺序？
3. `doca_umem_gpu_create` 为什么需要 `nvidia-peermem`？为什么 g201 节点上 `doca_umem_gpu_create` 会失败（见记忆 `g201-doca-peermem-missing.md`）？
4. `doca_gpu_verbs_export_qp` 之后 GPU 侧拿到的是什么句柄？它如何被 kernel 消费（与第 19 课 `rdma_cache.cuh` 的 `direct_fetch` 衔接）？
5. 启动期 `gpunetio_read_probe_kernel` 做了什么？为什么必须在持久化 kernel 启动之前跑？它和第 11 课的 lifecycle 校验如何衔接？
6. `tools/gpunetio_probe.cc` 与 `tools/gpunetio_loopback_probe.cc` 这两个独立工具各自验证什么？

涉及文件（绝对路径）：

- `/home/xjs/experiment/dvstor/src/gpu/gpunetio_transport.cc`（约 712 行，本课主干）
- `/home/xjs/experiment/dvstor/src/gpu/gpunetio_transport.hh`（PImpl 接口 + `GpuNetioPersistentView`）
- `/home/xjs/experiment/dvstor/src/gpu/gpunetio_probe.cu`（启动期 RDMA read probe kernel）
- `/home/xjs/experiment/dvstor/src/gpu/gpunetio_probe.hh`（probe 参数结构体 `GpuNetioReadProbeParams`）
- `/home/xjs/experiment/dvstor/tools/gpunetio_probe.cc`（独立 capability probe 工具）
- `/home/xjs/experiment/dvstor/tools/gpunetio_loopback_probe.cc`（独立 loopback 压力 probe 工具）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/rdma_cache.cuh`（消费侧：`direct_fetch` / `direct_fetch_batch`）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_engine/construction.cc`（装配点：`direct_transport` 构造与 `PersistentKernelParams` 字段填充）

---

## 逐文件逐函数讲解

### 1. `gpunetio_transport.hh`：PImpl 接口与 `GpuNetioPersistentView`

整个 GPUNetIO 传输层用 PImpl 隔离 DOCA 头文件，对上层只暴露一个 POD 视图 `GpuNetioPersistentView`（`gpunetio_transport.hh:16-27`）：

```cpp
struct GpuNetioPersistentView {
  void** qp_array{};
  void* remote_regions{};
  uint32_t remote_region_count{};
  uint32_t qps_per_node{};
  int* qp_locks{};
  uint32_t local_mkey{};
  uint64_t local_iova_base{};
  unsigned char* data{};
  size_t data_bytes{};
  unsigned char* dump{};
};
```

这些字段全部是**设备指针**（指向 GPU 显存或 GPU可寻址的注册区），构造完成后会被原样塞进 `PersistentKernelParams`（见 `construction.cc:908-920`）：

- `qp_array`：`void**`，元素是 `doca_gpu_dev_verbs_qp*`（export 之后由 `doca_gpu_verbs_get_qp_dev` 返回的 device-side 句柄），kernel 里 `reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index])` 直接使用（`rdma_cache.cuh:48`）。
- `remote_regions`：`GpuNetioRemoteMemoryRegion*`，每个远端存储节点一项，含 `address` 与 `rkey`（已经做过字节序处理，下面会讲）。
- `qp_locks`：每个 QP 一个 `int` 信号量，由 kernel 里的 `lock_direct_qp` / `unlock_direct_qp` 自旋获取（`rdma_cache.cuh:59,102`）。
- `local_mkey`：本端 GPU MR 的 lkey，**已经经过 `byte_swap32`** —— 因为 mlx5 WQE 里的 mkey 字段是大端。
- `local_iova_base`：本端 GPU MR 的 IOVA 基址。peer_memory 模式下为 0（NIC 直接用 GPU 虚拟地址），dmabuf 模式下等于 `registered_base`（NIC 用 0 基 IOVA + offset）。
- `data` / `data_bytes`：本端 GPU 显存里那块"持久化数据缓冲区"（PQ codes、graph cache、exact cache、control snapshots 等都布局在这块里，见 `construction.cc:387-402`）。
- `dump`：probe/错误转储用的 8 字节 GPU 缓冲。

`GpuNetioPersistentTransport` 本身（`gpunetio_transport.hh:29-48`）只暴露构造、析构、`view()` 三个接口，构造函数签名是：

```cpp
GpuNetioPersistentTransport(
  const configuration::IndexConfiguration& config,
  size_t data_bytes,
  Context& context,
  ClientConnectionManager& connection_manager,
  const MemoryRegionTokens& remote_regions);
```

`data_bytes` 就是上层的 `remote_buffer_bytes`（`construction.cc:374,376-377`），即 PQ codes + graph cache + exact cache + control snapshots 的总和；`context` 与 `connection_manager` 复用第 4、5 课的 CPU verbs 通道，用来在 QP 建链阶段交换 `QPInfo`（lid、qpn）和接收远端 MR token。

### 2. `gpunetio_transport.cc`：匿名命名空间里的工具函数

文件开头 `namespace gpu { namespace { ... } }`（`gpunetio_transport.cc:40-207`）定义了一批构造期用的小工具。

#### 2.1 常量与对齐工具（`gpunetio_transport.cc:44-51`）

```cpp
constexpr uint32_t kQueryQueueEntries = 1024;
constexpr size_t kGpuPageSize = 64 * 1024;
constexpr size_t kExternalQueueBytes = 128 * 1024;
constexpr size_t kExternalDbrBytes = 4 * 1024;
```

- `kQueryQueueEntries`：CQ / SQ / RQ 的深度，1024 个 WQE。这是每个 (lane × server) QP 的容量。
- `kGpuPageSize`：DOCA GPU 内存分配的页大小 64 KiB，所有 `doca_gpu_mem_alloc` 都用它对齐。
- `kExternalQueueBytes`：每块 external umem 的大小 128 KiB，足以容纳 1024 个 64 字节的 CQE 或 WQE。
- `kExternalDbrBytes`：doorbell record 缓冲 4 KiB。

`align_up`（`gpunetio_transport.cc:49-51`）是常规向上取整；`byte_swap32`（`gpunetio_transport.cc:53-56`）把 host 字节序的 32 位值翻成大端 —— **mlx5 硬件用大端**，WQE 里所有 32/64 位字段都要大端。后面 `local_mkey_wqe = byte_swap32(local_mkey)` 就是为此。

#### 2.2 DOCA / CUDA 错误检查（`gpunetio_transport.cc:58-72`）

```cpp
[[noreturn]] void throw_doca(const char* what, doca_error_t status);
void check_doca(const char* what, doca_error_t status);
void check_cuda(const char* what, cudaError_t status);
```

`check_doca` 在 `status != DOCA_SUCCESS` 时抛 `std::runtime_error`，错误描述用 `doca_error_get_descr`。`check_cuda` 类似用 `cudaGetErrorString`。整个构造函数几乎每一行 DOCA/CUDA 调用都被 `check_doca` / `check_cuda` 包裹，失败即抛异常 —— 因为 GPUNetIO 装配任何一步失败都不可能降级运行。

#### 2.3 CQ owner 位初始化（`gpunetio_transport.cc:74-81`）

```cpp
void initialize_cq_owner_bits(void* cq_buffer, const size_t bytes) {
  std::vector<unsigned char> initial(bytes, 0);
  for (size_t offset = 63; offset < bytes; offset += 64) {
    initial[offset] =
      (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
  }
  check_cuda("cudaMemcpy(cq owner init)",
             cudaMemcpy(cq_buffer, initial.data(), bytes, cudaMemcpyHostToDevice));
}
```

这是 GPUNetIO 的一个关键细节。mlx5 CQE 是 64 字节，最后一字节 `op_own` 的高 4 位是 opcode、低 1 位是 owner 轮转位。CQ 初始化时**所有 64 字节槽位的 op_own 都必须写成 `MLX5_CQE_INVALID << 4 | MLX5_CQE_OWNER_MASK`**，否则 kernel 第一次 poll CQ 会误以为已经有完成事件。注意这里用 `offset = 63; offset += 64` —— 第 63 字节正好是第一个 CQE 的 `op_own` 字节位置（0-indexed）。这块 GPU 显存由 CPU 通过 PCIe 写入初值（`cudaMemcpy` HostToDevice），之后完全由 GPU kernel 读写。

#### 2.4 NIC handler 命名（`gpunetio_transport.cc:89-102`）

```cpp
const char* nic_handler_name(const doca_gpu_dev_verbs_nic_handler handler) {
  switch (handler) {
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO:       return "AUTO";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY:  return "CPU_PROXY";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB:  return "GPU_SM_DB";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF:  return "GPU_SM_BF";
    ...
  }
}
```

GPUNetIO 有四种 doorbell handler 模式：

- `CPU_PROXY`：GPU 写 WQE，但 doorbell 由 CPU 代理敲 —— 这会退化成 CPU 参与 datapath，dvstor 明确禁止（见 `gpunetio_transport.cc:385-388`）。
- `GPU_SM_DB`：GPU SM 直接写内存映射的 doorbell 寄存器。dvstor 选这种（`gpunetio_transport.cc:226-227`）。
- `GPU_SM_BF`：GPU SM 写 BlueFlame（WQE 内联到 doorbell 寄存器）。性能更高但要求 NIC 支持。

#### 2.5 QPInfo 交换（`gpunetio_transport.cc:104-110`）

```cpp
void exchange_qp_info(Context& channel_context, QueuePair& channel_qp,
                      const QPInfo& local_info, QPInfo& remote_info) {
  LocalMemoryRegion region{channel_context, &remote_info, sizeof(remote_info)};
  channel_qp.post_receive(region);
  channel_qp.post_send_inlined(&local_info, sizeof(local_info), IBV_WR_SEND);
  channel_context.poll_send_cq_until_completion();
  channel_context.receive();
}
```

这是**复用第 4 课的 CPU verbs 通道**来交换 GPU QP 的建链信息（lid + qpn）。注意这里 `channel_qp` 是 CPU QP（`cm.server_qps[server]`），不是 GPU QP。GPU QP 自己只能做 RDMA Read/Write，不能做 Send/Recv 建链握手。

#### 2.6 GPU PCI 地址解析（`gpunetio_transport.cc:112-115`）

```cpp
char* gpu_pci_address(const uint32_t gpu_device, char (&bus_id)[32]) {
  check_cuda("cudaDeviceGetPCIBusId",
             cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), static_cast<int>(gpu_device)));
  return bus_id;
}
```

`doca_gpu_create` 需要 GPU 的 PCI DBDF 字符串（如 `0000:01:00.0`），用 `cudaDeviceGetPCIBusId` 取。这一步在 `tools/gpunetio_probe.cc:117-118` 里也有，是 GPUNetIO 装配的固定前置。

#### 2.7 `DocaDevinfoList`：按 ibdev 名查 devinfo（`gpunetio_transport.cc:117-142`）

```cpp
class DocaDevinfoList {
public:
  DocaDevinfoList() {
    check_doca("doca_devinfo_create_list", doca_devinfo_create_list(&infos_, &count_));
  }
  ~DocaDevinfoList() {
    if (infos_ != nullptr) doca_devinfo_destroy_list(infos_);
  }
  doca_devinfo* find(const char* ibdev_name) const {
    for (uint32_t i = 0; i < count_; ++i) {
      char current_ibdev[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
      if (doca_devinfo_get_ibdev_name(infos_[i], current_ibdev, sizeof(current_ibdev)) != DOCA_SUCCESS) continue;
      if (std::strcmp(current_ibdev, ibdev_name) == 0) return infos_[i];
    }
    throw std::runtime_error(std::string("failed to find DOCA device for ibdev ") + ibdev_name);
  }
  ...
};
```

RAII 封装 `doca_devinfo_create_list` / `doca_devinfo_destroy_list`，按 ibdev 名（如 `mlx5_0`）找到对应的 `doca_devinfo*`。这个 ibdev 名来自第 4 课的 `Context::get_raw_context()->device`（`gpunetio_transport.cc:222`）—— 也就是说 **GPUNetIO 用的 NIC 必须和 CPU verbs 通道用的是同一块卡**。

#### 2.8 QP 状态机迁移（`gpunetio_transport.cc:144-205`）

三个函数对应 verbs QP 的标准三段式状态机：`RESET → INIT → RTR → RTS`。

**`qp_modify_to_init`（`gpunetio_transport.cc:144-160`）**：

```cpp
doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT);
doca_verbs_qp_attr_set_allow_remote_write(attr, 1);
doca_verbs_qp_attr_set_allow_remote_read(attr, 1);
doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
doca_verbs_qp_attr_set_pkey_index(attr, 0);
doca_verbs_qp_attr_set_port_num(attr, 1);
```

允许远端 Read/Write（dvstor 只用 RDMA Read，但 Write 也开着以备未来用），原子操作走 IB spec，pkey 索引 0，端口 1。

**`qp_modify_to_rtr`（`gpunetio_transport.cc:162-187`）**：进入 Ready-to-Receive，需要远端 QP 信息：

```cpp
doca_verbs_ah_attr_set_addr_type(ah_attr, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH);  // 不用 GRH（同子网）
doca_verbs_ah_attr_set_dlid(ah_attr, remote_info.lid);                       // 目标 LID
doca_verbs_ah_attr_set_sl(ah_attr, 0);                                       // Service Level 0
doca_verbs_qp_attr_set_path_mtu(attr, DOCA_MTU_SIZE_4K_BYTES);               // 4K MTU
doca_verbs_qp_attr_set_dest_qp_num(attr, remote_info.qp_number);             // 对端 QPN
doca_verbs_qp_attr_set_rq_psn(attr, 0);                                      // 期望的 PSN 从 0 开始
doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, 16);                         // 最多 16 个 outstanding RDMA Read
doca_verbs_qp_attr_set_min_rnr_timer(attr, 12);                              // RNR 重传定时器
```

`max_dest_rd_atomic=16` 表示作为目标端允许对端同时有 16 个未完成的 RDMA Read 请求；`min_rnr_timer=12` 是 RNR（Receiver Not Ready）的等待时间。

**`qp_modify_to_rts`（`gpunetio_transport.cc:189-205`）**：进入 Ready-to-Send：

```cpp
doca_verbs_qp_attr_set_sq_psn(attr, 0);                  // 发送 PSN 从 0 开始
doca_verbs_qp_attr_set_ack_timeout(attr, 14);            // ACK 超时
doca_verbs_qp_attr_set_retry_cnt(attr, 7);               // 重传次数 7
doca_verbs_qp_attr_set_rnr_retry(attr, 7);               // RNR 重传次数 7
doca_verbs_qp_attr_set_max_rd_atomic(attr, 16);          // 本端最多 16 个 outstanding RDMA Read
```

这些参数和第 5 课 CPU verbs QP 的取值是一致的 —— GPUNetIO QP 在协议层和 CPU QP 没有区别，区别只在 datapath（WQE/CQE/doorbell）的物理位置和谁来读写。

### 3. `Impl` 构造函数：核心装配逻辑（`gpunetio_transport.cc:209-580`）

这是本课最长也最关键的一段。它分四大块：(A) 设备/GPU 初始化，(B) 每 (lane × server) 创建 GPU-backed CQ+QP 并 export，(C) 注册本端 GPU MR 并切分配子区域，(D) 启动期 probe。

#### 3.1 构造函数签名与字段初始化（`gpunetio_transport.cc:209-219`）

```cpp
struct GpuNetioPersistentTransport::Impl {
  Impl(const configuration::IndexConfiguration& config,
       const size_t data_bytes,
       Context& context,
       ClientConnectionManager& cm,
       const MemoryRegionTokens& remote_regions)
      : qps_per_node(std::max<u32>(1, config.gpu_rdma_qps)),
        remote_region_count(static_cast<uint32_t>(remote_regions.size())) {
    if (data_bytes == 0 || remote_regions.empty()) {
      throw std::invalid_argument("GPUNetIO transport requires non-empty data and remote regions");
    }
```

`qps_per_node` 来自配置 `gpu_rdma_qps`（每存储节点多少条 GPU QP，用于并行度）；`remote_region_count` 就是存储节点数。两者至少为 1。

#### 3.2 块 A：设备与 GPU 初始化（`gpunetio_transport.cc:221-243`）

```cpp
char pci_bus_id[32] = {0};
const char* ibdev_name = ibv_get_device_name(context.get_raw_context()->device);
check_cuda("cudaSetDevice", cudaSetDevice(static_cast<int>(config.gpu_device)));
check_cuda("cudaFree(0)", cudaFree(nullptr));
const char* gpu_pci = gpu_pci_address(config.gpu_device, pci_bus_id);
constexpr doca_gpu_dev_verbs_nic_handler nic_handler =
  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

{
  DocaDevinfoList devinfos;
  check_doca("doca_verbs_context_create",
             doca_verbs_context_create(devinfos.find(ibdev_name),
                                       DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                                       &verbs_context));
}
check_doca("doca_verbs_pd_create", doca_verbs_pd_create(verbs_context, &pd));
ibv_pd* ibv_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(pd);
if (ibv_pd == nullptr) {
  throw std::runtime_error("doca_verbs_bridge_verbs_pd_get_ibv_pd returned null");
}
check_doca("doca_rdma_bridge_open_dev_from_pd",
           doca_rdma_bridge_open_dev_from_pd(ibv_pd, &dev));
check_doca("doca_gpu_create", doca_gpu_create(gpu_pci, &gpu));
```

逐行讲解：

1. `ibv_get_device_name(context.get_raw_context()->device)` —— 从 CPU verbs context 拿到 ibdev 名字（如 `mlx5_0`）。
2. `cudaSetDevice(config.gpu_device)` —— 选定 GPU。**这一步必须在任何 `doca_gpu_*` 调用之前完成**，否则 DOCA 不知道当前 CUDA context。
3. `cudaFree(nullptr)` —— 这是一个老把戏：`cudaSetDevice` 本身是惰性的，不会立即创建 primary context；`cudaFree(0)` 强制初始化 primary context，确保后续 `cudaMalloc` / `cudaDeviceGetPCIBusId` 等都生效。
4. `gpu_pci_address(...)` —— 拿到 GPU 的 PCI DBDF 字符串。
5. `nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB` —— 选定 GPU SM 直接敲 doorbell 模式。
6. `DocaDevinfoList devinfos;` 后 `doca_verbs_context_create(devinfos.find(ibdev_name), ...)` —— 创建 DOCA verbs context。`devinfos` 在这个 block 结束时析构，释放 devinfo list（context 已经 hold 住必要的句柄）。
7. `doca_verbs_pd_create(verbs_context, &pd)` —— 创建 Protection Domain。所有 QP、MR 都必须挂在同一个 PD 下。
8. `doca_verbs_bridge_verbs_pd_get_ibv_pd(pd)` —— 从 DOCA PD 拿到 libibverbs 的 `ibv_pd*`。这个 bridge 是为了让我们能用标准的 `ibv_reg_mr` / `mlx5dv_reg_dmabuf_mr` 来注册 GPU MR（下面块 C 会用到）。
9. `doca_rdma_bridge_open_dev_from_pd(ibv_pd, &dev)` —— 从 `ibv_pd` 反向打开一个 `doca_dev*`。后续 `doca_umem_gpu_create` 需要 `doca_dev*` 来绑定底层 RDMA 设备。
10. `doca_gpu_create(gpu_pci, &gpu)` —— 创建 `doca_gpu*` 句柄，这是 GPUNetIO 的"GPU 端锚点"，所有 `doca_gpu_mem_alloc` / `doca_umem_gpu_create` / `doca_gpu_verbs_export_qp` 都需要它。

这十步的顺序是有依赖的：必须先有 `verbs_context` 才能建 `pd`；必须有 `pd` 才能拿 `ibv_pd`；必须有 `ibv_pd` 才能开 `dev`；`dev` 和 `gpu` 都有了才能创建 GPU umem 和 export QP。

#### 3.3 块 B：每 (lane × server) 创建一对 GPU-backed CQ + QP（`gpunetio_transport.cc:245-411`）

这是整个构造函数最长的一段。外层两层循环：

```cpp
for (uint32_t lane = 0; lane < std::max<uint32_t>(1, qps_per_node); ++lane) {
  for (uint32_t server = 0; server < remote_region_count; ++server) {
    ...
  }
}
```

每个 (lane, server) 组合创建一对独立的 CQ + QP，总共 `qps_per_node × remote_region_count` 条 QP。kernel 侧的 QP 索引计算是 `qp_index = (lane % direct_qps_per_node) * direct_region_count + memory_node`（`rdma_cache.cuh:45-46`，`rdma_cache.cuh:134-135`）—— 正好对应这里的双重循环顺序。

循环体内部又分五小步。

##### 3.3.1 分配 GPU 显存并初始化 CQ owner 位（`gpunetio_transport.cc:264-277`）

```cpp
check_doca("doca_verbs_cq_attr_create", doca_verbs_cq_attr_create(&cq_attr));
check_doca("doca_verbs_cq_attr_set_entry_size",
           doca_verbs_cq_attr_set_entry_size(cq_attr, DOCA_VERBS_CQ_ENTRY_SIZE_64));
check_doca("doca_verbs_cq_attr_set_cq_size",
           doca_verbs_cq_attr_set_cq_size(cq_attr, kQueryQueueEntries));
check_doca("doca_verbs_cq_attr_set_cq_overrun",
           doca_verbs_cq_attr_set_cq_overrun(cq_attr, 1));
check_doca("doca_gpu_mem_alloc(send_cq_umem)",
           doca_gpu_mem_alloc(
             gpu, kExternalQueueBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &send_cq_umem_buf, nullptr));
check_doca("doca_gpu_mem_alloc(recv_cq_umem)",
           doca_gpu_mem_alloc(
             gpu, kExternalQueueBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &recv_cq_umem_buf, nullptr));
initialize_cq_owner_bits(send_cq_umem_buf, kExternalQueueBytes);
initialize_cq_owner_bits(recv_cq_umem_buf, kExternalQueueBytes);
```

- CQ entry size 64 字节（标准 mlx5 CQE 大小）。
- CQ 深度 1024。
- `cq_overrun=1` 允许 CQ 溢出（生产环境常用，避免硬件卡死）。
- 用 `doca_gpu_mem_alloc` 在 GPU 显存里分配 128 KiB 给 send_cq 和 recv_cq。
- 调用 2.3 节的 `initialize_cq_owner_bits` 把每个 CQE 的 owner 位设好。

**为什么 send_cq 和 recv_cq 都要？** dvstor 的查询热路径只用 RDMA Read（单向），但 QP 是 RC 类型，建链时需要双向能力；并且 `doca_verbs_qp_init_attr` 要求同时提供 send_cq 和 recv_cq（即使 recv 永远不会被用）。

##### 3.3.2 `doca_umem_gpu_create`：把 GPU 显存注册成 RDMA 可见（`gpunetio_transport.cc:278-297`）

```cpp
check_doca("doca_umem_gpu_create(send_cq)",
           doca_umem_gpu_create(gpu,
                                dev,
                                send_cq_umem_buf,
                                kExternalQueueBytes,
                                DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                  DOCA_ACCESS_FLAG_RDMA_WRITE |
                                  DOCA_ACCESS_FLAG_RDMA_READ |
                                  DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                &send_cq_umem));
```

`doca_umem_gpu_create` 是 GPUNetIO 的核心 API 之一：它把一块 GPU 显存注册成 NIC 可以直接 DMA 访问的 umem。**这一步需要 `nvidia-peermem` 内核模块**（或新的 `nvidia-peermem-ld`），它为 GPU 显存建立 PCIe→NIC 的 P2P 映射，让 NIC 的 DMA 引擎能直接读写 GPU 显存而不经过 CPU 内存。

记忆 `g201-doca-peermem-missing.md` 里记的就是这一步在 g201 节点上的失败：g201 上 `nvidia-peermem` 没加载（`lsmod | grep peermem` 空），`doca_umem_gpu_create` 会返回 `DOCA_ERROR_NOT_PERMITTED` 之类的错误，因为内核拒绝为 GPU 内存建立 P2P 映射。这就是为什么 RDMA 集群必须在 .202 上跑、g201 不能跑 GPUNetIO 的根本原因。

access flags 同时允许本地读写、远端 RDMA 写、远端 RDMA 读、远端原子 —— 对 CQ umem 来说其实只需要本地读写（NIC 写 CQE、GPU 读 CQE），但这里全开是为了和 QP WQ umem 复用同一套 flag。

recv_cq umem 走完全相同的流程（`gpunetio_transport.cc:288-297`）。

##### 3.3.3 创建 CQ 并绑定 external umem（`gpunetio_transport.cc:298-305`）

```cpp
check_doca("doca_verbs_cq_attr_set_external_datapath_en",
           doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1));
check_doca("doca_verbs_cq_attr_set_external_umem(send)",
           doca_verbs_cq_attr_set_external_umem(cq_attr, send_cq_umem, 0));
check_doca("doca_verbs_cq_create(send)", doca_verbs_cq_create(verbs_context, cq_attr, &send_cq));
check_doca("doca_verbs_cq_attr_set_external_umem(recv)",
           doca_verbs_cq_attr_set_external_umem(cq_attr, recv_cq_umem, 0));
check_doca("doca_verbs_cq_create(recv)", doca_verbs_cq_create(verbs_context, cq_attr, &recv_cq));
```

- `set_external_datapath_en(1)` 开启 external datapath —— 告诉 DOCA 这块 CQ 的 backing memory 在 GPU 上、由 GPU 而非 CPU 管理。
- `set_external_umem(send_cq_umem, 0)` 把刚才 `doca_umem_gpu_create` 出来的 umem 绑到 CQ 的 offset 0。
- `doca_verbs_cq_create` 真正创建 CQ 对象。
- recv_cq 复用同一个 `cq_attr`（只改 umem 绑定），再 create 一次。

注意 `cq_attr` 是个 builder，两次 create 之间只换了 external umem，其它字段（entry size、size、overrun、external datapath）都保持。

##### 3.3.4 创建 QP 并绑 external umem / dbr / uar（`gpunetio_transport.cc:306-362`）

```cpp
check_doca("doca_verbs_qp_init_attr_create", doca_verbs_qp_init_attr_create(&qp_init));
check_doca("doca_verbs_qp_init_attr_set_pd", doca_verbs_qp_init_attr_set_pd(qp_init, pd));
check_doca("doca_verbs_qp_init_attr_set_send_cq", doca_verbs_qp_init_attr_set_send_cq(qp_init, send_cq));
check_doca("doca_verbs_qp_init_attr_set_receive_cq", doca_verbs_qp_init_attr_set_receive_cq(qp_init, recv_cq));
check_doca("doca_verbs_qp_init_attr_set_sq_wr", doca_verbs_qp_init_attr_set_sq_wr(qp_init, kQueryQueueEntries));
check_doca("doca_verbs_qp_init_attr_set_rq_wr", doca_verbs_qp_init_attr_set_rq_wr(qp_init, kQueryQueueEntries));
check_doca("doca_verbs_qp_init_attr_set_send_max_sges",
           doca_verbs_qp_init_attr_set_send_max_sges(qp_init, 1));
check_doca("doca_verbs_qp_init_attr_set_receive_max_sges",
           doca_verbs_qp_init_attr_set_receive_max_sges(qp_init, 1));
check_doca("doca_verbs_qp_init_attr_set_max_inline_data",
           doca_verbs_qp_init_attr_set_max_inline_data(qp_init, 0));
check_doca("doca_verbs_qp_init_attr_set_qp_type",
           doca_verbs_qp_init_attr_set_qp_type(qp_init, DOCA_VERBS_QP_TYPE_RC));
```

QP 初始化属性：

- 挂到本 PD。
- send/recv CQ 分别绑上面创建的两个 CQ。
- SQ/RQ 深度都是 1024。
- Send/Recv 各最多 1 个 SGE（scatter-gather entry）—— dvstor 每次 RDMA Read 用一段连续 buffer。
- `max_inline_data=0` —— 不用 inline data（RDMA Read 不发送 payload，所以这个字段对 Read 没意义）。
- **QP 类型 RC（Reliable Connection）** —— 这是 RDMA Read 必须的（UC/UD 不支持 RDMA Read）。

接下来分配 QP 的 WQ umem 和 doorbell record umem：

```cpp
check_doca("doca_gpu_mem_alloc(qp_wq_umem)",
           doca_gpu_mem_alloc(gpu, kExternalQueueBytes, kGpuPageSize,
                              DOCA_GPU_MEM_TYPE_GPU, &qp_wq_umem_buf, nullptr));
check_doca("doca_gpu_mem_alloc(qp_dbr_umem)",
           doca_gpu_mem_alloc(gpu, kExternalDbrBytes, kGpuPageSize,
                              DOCA_GPU_MEM_TYPE_GPU, &qp_dbr_umem_buf, nullptr));
check_cuda("cudaMemset(qp_wq_umem)", cudaMemset(qp_wq_umem_buf, 0, kExternalQueueBytes));
check_cuda("cudaMemset(qp_dbr_umem)", cudaMemset(qp_dbr_umem_buf, 0, kExternalDbrBytes));
check_doca("doca_umem_gpu_create(qp_wq)", ...);
check_doca("doca_umem_gpu_create(qp_dbr)", ...);
```

- `qp_wq_umem`：SQ 的工作队列，1024 个 WQE 的存储。GPU kernel 会把 WQE 写到这里。
- `qp_dbr_umem`：doorbell record，记录 SQ 的 producer index。GPU kernel 敲 doorbell 时会更新这里的 32 位计数器。
- 两块都 `cudaMemset` 清零（WQ 必须清零，否则 NIC 可能把残留数据当 WQE 处理）。
- 两块都 `doca_umem_gpu_create` 注册成 RDMA 可见。

然后绑到 QP init attr：

```cpp
check_doca("doca_verbs_qp_init_attr_set_external_datapath_en",
           doca_verbs_qp_init_attr_set_external_datapath_en(qp_init, 1));
check_doca("doca_verbs_qp_init_attr_set_external_umem",
           doca_verbs_qp_init_attr_set_external_umem(qp_init, qp_wq_umem, 0));
check_doca("doca_verbs_qp_init_attr_set_external_dbr_umem",
           doca_verbs_qp_init_attr_set_external_dbr_umem(qp_init, qp_dbr_umem, 0));
```

`external_datapath_en(1)` 让 QP 的 SQ datapath 走 GPU umem。

接下来分配 UAR（User Access Region）——doorbell 寄存器的用户态映射：

```cpp
doca_error_t uar_status = doca_uar_create(
  dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar);
if (uar_status != DOCA_SUCCESS) {
  uar_status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &external_uar);
}
check_doca("doca_uar_create(GPU doorbell)", uar_status);
check_doca("doca_verbs_qp_init_attr_set_external_uar",
           doca_verbs_qp_init_attr_set_external_uar(qp_init, external_uar));
check_doca("doca_verbs_qp_create", doca_verbs_qp_create(verbs_context, qp_init, &qp));
```

- 优先 `NONCACHE_DEDICATED`（专属、不可缓存）UAR —— 性能最好。
- 失败则退化到 `NONCACHE`（共享不可缓存）。
- UAR 绑到 QP 后，`doca_verbs_qp_create` 创建 QP 对象。

至此 QP 的所有 backing resource（CQ umem、WQ umem、dbr umem、UAR）都在 GPU 上或绑定到 GPU。

##### 3.3.5 `doca_gpu_verbs_export_qp`：导出给 GPU kernel（`gpunetio_transport.cc:367-388`）

```cpp
send_cqs.push_back(send_cq);
recv_cqs.push_back(recv_cq);
qps.push_back(qp);
std::cerr << "[STATUS]: exporting GPUNetIO QP resource=" << 0
          << " lane=" << lane
          << " server=" << server << " qpn=" << doca_verbs_qp_get_qpn(qp)
          << " gpu_pci=" << gpu_pci << " ibdev=" << ibdev_name
          << " handler=" << nic_handler_name(nic_handler) << std::endl;
check_doca("doca_gpu_verbs_export_qp",
           doca_gpu_verbs_export_qp(gpu,
                                    dev,
                                    qp,
                                    nic_handler,
                                    qp_wq_umem_buf,
                                    send_cq,
                                    recv_cq,
                                    &gpu_qp));
check_doca("doca_gpu_verbs_get_qp_dev", doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_qp_dev));
uint8_t cpu_proxy_enabled = 0;
check_doca("doca_gpu_verbs_cpu_proxy_enabled",
           doca_gpu_verbs_cpu_proxy_enabled(gpu_qp, &cpu_proxy_enabled));
if (cpu_proxy_enabled != 0) {
  throw std::runtime_error(
    "GPUNetIO exported a CPU-proxy QP; the GPU-only query engine requires GPU doorbells");
}
```

`doca_gpu_verbs_export_qp` 是把 QP "导出"成 GPU 可直接操作的句柄 `doca_gpu_verbs_qp* gpu_qp`。它的入参：

- `gpu` / `dev`：GPU 与 RDMA 设备句柄。
- `qp`：上面创建的 `doca_verbs_qp*`。
- `nic_handler`：doorbell handler 模式（这里是 `GPU_SM_DB`）。
- `qp_wq_umem_buf`：SQ 的 GPU 显存指针（kernel 要直接写 WQE 到这里）。
- `send_cq` / `recv_cq`：CQ 对象（kernel 要直接 poll CQE）。
- 出参 `gpu_qp`：GPU 侧句柄。

紧接着 `doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_qp_dev)` 拿到的是 **device-side 句柄**（`doca_gpu_dev_verbs_qp*`）—— 这是真正会被传给 kernel 的指针。`gpu_qp`（host-side export 句柄）只用于管理（销毁、查询属性）；`gpu_qp_dev`（device-side 句柄）才是 kernel 里 `doca_gpu_dev_verbs_get_wqe_ptr` / `doca_gpu_dev_verbs_submit` 等内联函数要操作的对象。

下面立即检查 `cpu_proxy_enabled`：如果 export 出来的是 CPU-proxy QP（GPU 敲 doorbell 退化成 CPU 代理），直接抛异常。这是 dvstor 的硬性要求 —— **GPU-only 查询引擎必须用 GPU doorbells，不允许 CPU 介入 datapath**。

##### 3.3.6 QP 状态机迁移与 QPInfo 交换（`gpunetio_transport.cc:390-395`）

```cpp
qp_modify_to_init(qp);
const QPInfo local_info{context.get_lid(), doca_verbs_qp_get_qpn(qp)};
QPInfo remote_info{};
exchange_qp_info(context, *cm.server_qps[server], local_info, remote_info);
qp_modify_to_rtr(verbs_context, qp, remote_info);
qp_modify_to_rts(qp);
```

这就是 2.8 节那三个函数的应用。注意 `exchange_qp_info` 用的 `cm.server_qps[server]` 是第 5 课的 CPU verbs QP —— GPU QP 自己不能做 Send/Recv 握手。和第 5 课的 CPU QP 建链流程完全对称，只是 QP 本体的 datapath 在 GPU 上。

##### 3.3.7 收集句柄（`gpunetio_transport.cc:397-409`）

```cpp
gpu_qps.push_back(gpu_qp);
gpu_qp_devices_host.push_back(gpu_qp_dev);
external_uars.push_back(external_uar);
external_umems.push_back(send_cq_umem);
external_umems.push_back(recv_cq_umem);
external_umems.push_back(qp_wq_umem);
external_umems.push_back(qp_dbr_umem);
external_umem_buffers.push_back(send_cq_umem_buf);
external_umem_buffers.push_back(recv_cq_umem_buf);
external_umem_buffers.push_back(qp_wq_umem_buf);
external_umem_buffers.push_back(qp_dbr_umem_buf);
check_doca("doca_verbs_qp_init_attr_destroy", doca_verbs_qp_init_attr_destroy(qp_init));
check_doca("doca_verbs_cq_attr_destroy", doca_verbs_cq_attr_destroy(cq_attr));
```

`gpu_qp_devices_host` 是 `vec<void*>`，元素是 device-side 句柄。稍后块 D 会把它 `cudaMemcpy` 到设备指针 `d_qp_array`，再通过 `view().qp_array` 暴露给 kernel。

attr builder 用完即销毁。

#### 3.4 块 C：注册本端 GPU MR 并切分配子区域（`gpunetio_transport.cc:414-478`）

前面 export 的 QP 只能"发送"RDMA Read 请求，但 Read 的目标 buffer（本端 GPU 显存）还必须注册成 MR，让 NIC 能 DMA 写入。

##### 3.4.1 分配注册区（`gpunetio_transport.cc:414-424`）

```cpp
const size_t control_bytes =
  2 * sizeof(uint64_t) + sizeof(int) +
  kGpuNetioProbeDebugValueCount * sizeof(uint64_t) + 256;
const size_t registered_bytes =
  align_up(control_bytes + kGpuPageSize, kGpuPageSize) +
  align_up(data_bytes, kGpuPageSize);

check_doca("doca_gpu_mem_alloc",
           doca_gpu_mem_alloc(
             gpu, registered_bytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU,
             &registered_base, nullptr));
```

`registered_bytes` = 控制区（probe value + probe status + probe debug 数组 + 256 字节余量）+ `data_bytes`（持久化数据区）。两块都按 64 KiB 对齐。这块 GPU 显存就是整个查询引擎的本端数据中枢。

##### 3.4.2 注册 MR：peer_memory 优先，dmabuf 兜底（`gpunetio_transport.cc:425-453`）

```cpp
const int mr_access = IBV_ACCESS_LOCAL_WRITE |
  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
errno = 0;
registered_mr = ibv_reg_mr(
  ibv_pd, registered_base, registered_bytes, mr_access);
const int peer_memory_error = errno;
if (registered_mr != nullptr) {
  local_iova_base = 0;
  std::cerr << "[STATUS]: GPUNetIO GPU MR registration=peer_memory bytes="
            << registered_bytes << std::endl;
} else {
  check_doca("doca_gpu_dmabuf_fd",
             doca_gpu_dmabuf_fd(
               gpu, registered_base, registered_bytes, &dmabuf_fd));
  registered_mr = mlx5dv_reg_dmabuf_mr(
    ibv_pd, 0, registered_bytes, 0, dmabuf_fd, mr_access, 0);
  local_iova_base = reinterpret_cast<uint64_t>(registered_base);
}
if (registered_mr == nullptr) {
  throw std::runtime_error(
    std::string("GPU MR registration failed: peer_memory=") +
    std::strerror(peer_memory_error) + ", dmabuf=" + std::strerror(errno));
}
if (local_iova_base != 0) {
  std::cerr << "[STATUS]: GPUNetIO GPU MR registration=dmabuf bytes="
            << registered_bytes << " peer_memory_error="
            << std::strerror(peer_memory_error) << std::endl;
}
local_mkey = registered_mr->lkey;
local_mkey_wqe = byte_swap32(local_mkey);
```

这里有两条注册路径：

**路径 1：`ibv_reg_mr` + nvidia-peermem（peer_memory 模式）**

直接对 GPU 显存指针调用标准的 `ibv_reg_mr`。这要求 `nvidia-peermim` 内核模块加载，它实现了 `get_dma_peer` / `get_page_peer` 回调，让 ib_core 把 GPU page 当作可 DMA 的 peer page。注册成功后，NIC 用 GPU 虚拟地址做 DMA，`local_iova_base = 0` —— WQE 里直接填 GPU 虚拟地址（`destination - 0`）。

**路径 2：`mlx5dv_reg_dmabuf_mr`（dmabuf 模式）**

如果 peer_memory 失败（peermem 没装、或内核不支持），改走 dmabuf：`doca_gpu_dmabuf_fd` 拿到 GPU 内存的 dma-buf 文件描述符，`mlx5dv_reg_dmabuf_mr` 用 fd 注册。注册成功后 NIC 用 0 基 IOVA + offset 做 DMA，`local_iova_base = registered_base` —— WQE 里填 `destination - registered_base`（即 offset）。

`tools/gpunetio_probe.cc` 提供了独立验证这两种路径的工具（见第 5 节）。

**`local_mkey_wqe = byte_swap32(local_mkey)`**：lkey 是 host 字节序，WQE 里需要大端，所以预先翻一次。`GpuNetioPersistentView::local_mkey` 直接返回翻转后的值（`gpunetio_transport.cc:704`）。

##### 3.4.3 切分配子区域（`gpunetio_transport.cc:456-478`）

```cpp
size_t offset = 0;
auto allocate = [&](const size_t bytes, const size_t alignment) -> void* {
  offset = align_up(offset, alignment);
  auto* pointer = static_cast<unsigned char*>(registered_base) + offset;
  offset += bytes;
  return pointer;
};

d_probe_value = static_cast<uint64_t*>(allocate(sizeof(uint64_t), alignof(uint64_t)));
d_dump = static_cast<unsigned char*>(allocate(sizeof(uint64_t), alignof(uint64_t)));
d_probe_status = static_cast<int*>(allocate(sizeof(int), alignof(int)));
d_probe_debug = static_cast<uint64_t*>(
  allocate(kGpuNetioProbeDebugValueCount * sizeof(uint64_t), alignof(uint64_t)));
persistent_data = static_cast<unsigned char*>(allocate(data_bytes, kGpuPageSize));
persistent_data_size = data_bytes;
if (offset > registered_bytes) {
  throw std::logic_error("GPUNetIO registered allocation layout overflow");
}
```

用 lambda `allocate` 在 `registered_base` 上做 bump 分配，依次切出 probe value、dump、probe status、probe debug 数组、持久化数据区。`persistent_data` 就是 `view().data`（`gpunetio_transport.cc:706`），上层 `construction.cc:382` 把它当作 `d_remote_buffer` 用，所有 PQ codes、graph cache、exact cache 都布局在这块里（见第 11、13 课）。

#### 3.5 块 D：远端 region 表 + 启动期 probe（`gpunetio_transport.cc:480-579`）

##### 3.5.1 远端 region 表（`gpunetio_transport.cc:480-504`）

```cpp
remote_regions_host.resize(remote_region_count);
for (uint32_t i = 0; i < remote_region_count; ++i) {
  remote_regions_host[i] = {
    .address = remote_regions[i]->address,
    .rkey = byte_swap32(remote_regions[i]->rkey),
    .reserved = remote_regions[i]->rkey,
  };
}
check_cuda("cudaMalloc(remote_regions)",
           cudaMalloc(&d_remote_regions,
                      remote_regions_host.size() * sizeof(GpuNetioRemoteMemoryRegion)));
check_cuda("cudaMemcpy(remote_regions)", cudaMemcpy(d_remote_regions, ...));

check_cuda("cudaMalloc(qp_array)",
           cudaMalloc(&d_qp_array, gpu_qp_devices_host.size() * sizeof(void*)));
check_cuda("cudaMemcpy(qp_array)", cudaMemcpy(d_qp_array, gpu_qp_devices_host.data(), ...));

check_cuda("cudaMalloc(qp_locks)",
           cudaMalloc(&d_qp_locks, gpu_qp_devices_host.size() * sizeof(int)));
check_cuda("cudaMemset(qp_locks)", cudaMemset(d_qp_locks, 0, ...));
```

注意 `GpuNetioRemoteMemoryRegion`（`gpunetio_probe.hh:13-17`）：

```cpp
struct GpuNetioRemoteMemoryRegion {
  uint64_t address;
  uint32_t rkey;
  uint32_t reserved;
};
```

`rkey` 字段是 **大端**（`byte_swap32(remote_regions[i]->rkey)`），因为 WQE 里直接用这个值。`reserved` 保留 host 字节序的 rkey 便于调试。

`d_qp_array` 和 `d_qp_locks` 是设备指针数组，每个元素对应一个 (lane, server) QP。kernel 侧通过 `params.direct_qps[qp_index]` 和 `params.direct_qp_locks[qp_index]` 访问。

##### 3.5.2 启动期 probe 循环（`gpunetio_transport.cc:506-575`）

```cpp
check_cuda("cudaStreamCreate", cudaStreamCreate(&stream));
if (data_bytes > 0) {
  for (uint32_t qp_index = 0; qp_index < gpu_qp_devices_host.size(); ++qp_index) {
    check_cuda("cudaMemset(GPUNetIO probe status)",
               cudaMemset(d_probe_status, 0, sizeof(int)));
    check_cuda("cudaMemset(GPUNetIO probe debug)",
               cudaMemset(d_probe_debug, 0,
                          kGpuNetioProbeDebugValueCount * sizeof(uint64_t)));
    launch_gpunetio_read_probe(stream, GpuNetioReadProbeParams{
      .local_mkey = local_mkey_wqe,
      .local_iova_base = local_iova_base,
      .remote_regions = d_remote_regions,
      .remote_region_count = remote_region_count,
      .qp_array = d_qp_array,
      .qp_count = static_cast<uint32_t>(gpu_qp_devices_host.size()),
      .qp_index = qp_index,
      .remote_region = qp_index % remote_region_count,
      .destination = reinterpret_cast<unsigned char*>(d_probe_value),
      .dump_ptr = d_dump,
      .status_code = d_probe_status,
      .debug_values = d_probe_debug,
    });
    check_cuda("launch_gpunetio_read_probe", cudaGetLastError());
    check_cuda("cudaStreamSynchronize(GPUNetIO probe)", cudaStreamSynchronize(stream));
    int probe_status = 0;
    uint64_t probe_debug[kGpuNetioProbeDebugValueCount]{};
    check_cuda("cudaMemcpy(GPUNetIO probe status)", cudaMemcpy(&probe_status, ...));
    check_cuda("cudaMemcpy(GPUNetIO probe debug)", cudaMemcpy(probe_debug, ...));
    if (probe_status != 0) {
      throw std::runtime_error(
        "GPUNetIO startup RDMA read probe failed: qp=" + ...
        ... 一长串 debug 字段 ...
      );
    }
  }
  std::cerr << "[STATUS]: GPUNetIO startup RDMA read probe passed for "
            << gpu_qp_devices_host.size() << " QPs\n";
  std::cerr << "[STATUS]: GPUNetIO RDMA Read implementation=manual_wqe_locked\n";
}
```

对每条 QP 跑一次 `launch_gpunetio_read_probe`，目标远端是 `qp_index % remote_region_count`（轮询分配，确保每条 QP 都能命中它对应的存储节点）。如果 probe 失败，抛带完整 debug 信息的异常 —— debug 信息覆盖了 remote addr/rkey、local iova/lkey、ticket、CQE consumer index、op_own、SQ pre/post 三元组、WQE 16 字节 hex dump 等等，几乎可以还原 kernel 内部状态。

**为什么必须 probe？** GPUNetIO 装配链路极长：GPU mem → umem → external datapath → QP → export → device handle。任何一步静默失败都可能让后续 kernel 跑出"看起来成功但数据是错的"或者死循环 poll CQ。probe 强制每条 QP 至少完成一次真实的 RDMA Read（从远端读 8 字节到本端 GPU），验证整条链路通畅，然后才允许上层启动持久化 kernel。这与第 11 课 lifecycle 的"装配→校验→启动"三段式呼应 —— probe 是校验阶段的核心动作。

`"GPUNetIO RDMA Read implementation=manual_wqe_locked"` 这行日志说明 dvstor 用的是 **manual WQE + locked（每 QP 自旋锁）** 实现，而不是 GPUNetIO 提供的高层 `doca_gpu_dev_verbs_post_read` API —— 原因是 dvstor 需要在 batch read 时精细控制 WQE flag（如 `CQ_ERROR_UPDATE` vs `CQ_UPDATE`，见 `rdma_cache.cuh:82-83,215-217`）。

#### 3.6 `view()` 与析构（`gpunetio_transport.cc:582-710`）

析构（`gpunetio_transport.cc:582-651`）按装配逆序销毁：先销 stream / cudaFree 设备数组 / dereg MR / close dmabuf_fd / free 注册区，再逐个 `doca_gpu_verbs_unexport_qp` → `doca_verbs_qp_destroy` → `doca_uar_destroy` → `doca_verbs_cq_destroy` × 2 → `doca_umem_destroy` × 4 → `doca_gpu_mem_free` × 4，最后 `doca_verbs_pd_destroy` → `doca_verbs_context_destroy` → `doca_gpu_destroy` → `doca_dev_close`。每一步都判空，避免双重释放。

`view()`（`gpunetio_transport.cc:697-710`）把内部设备指针打包成 POD：

```cpp
return {
  .qp_array = impl_->d_qp_array,
  .remote_regions = impl_->d_remote_regions,
  .remote_region_count = impl_->remote_region_count,
  .qps_per_node = impl_->qps_per_node,
  .qp_locks = impl_->d_qp_locks,
  .local_mkey = impl_->local_mkey_wqe,
  .local_iova_base = impl_->local_iova_base,
  .data = impl_->persistent_data,
  .data_bytes = impl_->persistent_data_size,
  .dump = impl_->d_dump,
};
```

这个 view 被 `construction.cc:378` 取走，字段被填进 `PersistentKernelParams`（`construction.cc:908-920`），最终被持久化 kernel 消费。

### 4. `gpunetio_probe.cu` / `gpunetio_probe.hh`：启动期 probe kernel

`gpunetio_probe.hh` 定义参数结构体（`gpunetio_probe.hh:13-32`）：

```cpp
struct GpuNetioRemoteMemoryRegion {
  uint64_t address;
  uint32_t rkey;
  uint32_t reserved;
};

struct GpuNetioReadProbeParams {
  uint32_t local_mkey;
  uint64_t local_iova_base;
  const GpuNetioRemoteMemoryRegion* remote_regions;
  uint32_t remote_region_count;
  void* const* qp_array;
  uint32_t qp_count;
  uint32_t qp_index;
  uint32_t remote_region;
  unsigned char* destination;
  unsigned char* dump_ptr;
  int* status_code;
  uint64_t* debug_values;
};
```

`GpuNetioReadProbeParams` 是一个 by-value 的 POD，会作为 kernel 参数传给 `__global__` 函数。注意 `qp_array` 是 `void* const*` —— 指向设备侧 `void*` 数组的 host 指针（数组本身在设备内存）。

#### 4.1 `poll_cq_at_with_timeout`（`gpunetio_probe.cu:20-53`）

```cpp
template <enum doca_gpu_dev_verbs_resource_sharing_mode sharing_mode>
__device__ inline int poll_cq_at_with_timeout(struct doca_gpu_dev_verbs_cq* cq,
                                              const uint64_t ticket,
                                              uint64_t* cqe_debug) {
  auto* cqe_base = reinterpret_cast<struct mlx5_cqe64*>(__ldg((uintptr_t*)&cq->cqe_daddr));
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  const uint32_t idx = ticket & (cqe_num - 1);
  auto* cqe64 = &cqe_base[idx];

  uint64_t curr_cons_index = 0;
  uint8_t opown = 0;
  for (uint64_t spins = 0; spins < kPollSpinLimit; ++spins) {
    curr_cons_index =
      doca_gpu_dev_verbs_load_relaxed<sharing_mode>(&cq->cqe_ci);
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global(reinterpret_cast<uint8_t*>(&cqe64->op_own));
    if (!((curr_cons_index <= ticket) && ((opown & MLX5_CQE_OWNER_MASK) ^ !!(ticket & cqe_num)))) {
      const uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      if (opcode == MLX5_CQE_REQ_ERR) return -EIO;
      doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      doca_gpu_dev_verbs_atomic_max<uint64_t, sharing_mode>(
        &cq->cqe_ci, ticket + 1);
      return 0;
    }
  }
  if (cqe_debug != nullptr) {
    cqe_debug[0] = 0x54494d454f5554ULL;   // "TIMEOUT" 的 ASCII
    cqe_debug[1] = (ticket << 32) | idx;
    cqe_debug[2] = curr_cons_index;
    cqe_debug[3] = opown;
  }
  return kPollTimeoutStatus;
}
```

这是 GPUNetIO 用户态 poll CQ 的标准模式，要点几乎每行都值得讲：

- `__ldg((uintptr_t*)&cq->cqe_daddr)` —— 用 `__ldg`（load through read-only cache）读 CQE 基地址 `cqe_daddr`。这是 device-side `doca_gpu_dev_verbs_cq` 结构体里的 GPU 虚拟地址。
- `cqe_num` 是 CQ 深度（2 的幂），`ticket & (cqe_num - 1)` 算出 CQE 索引。
- 自旋 `kPollSpinLimit = 100000000` 次（约 1 亿次），每次：
  - `doca_gpu_dev_verbs_load_relaxed` 松弛读 `cqe_ci`（consumer index）。
  - `doca_gpu_dev_verbs_load_relaxed_sys_global` 系统级松弛读 CQE 的 `op_own` 字节 —— 系统级 fence 确保 GPU 看到 NIC 写入的 CQE。
  - 判断 CQE 是否就绪：`!((curr_cons_index <= ticket) && ((opown & OWNER) ^ !!(ticket & cqe_num)))` —— 这是 mlx5 的 owner 轮转算法。当 CQE 还没就绪时，`opown & OWNER` 与 `ticket & cqe_num` 的奇偶性一致（异或为 0），条件成立，继续 spin；就绪后异或变 1，条件不成立，跳出。
- 跳出后检查 opcode：如果是 `MLX5_CQE_REQ_ERR` 返回 `-EIO`（请求错误）。
- 正常完成则 `doca_gpu_dev_verbs_fence_acquire`（fence 保证后续操作看到 CQE 内容）+ `doca_gpu_dev_verbs_atomic_max(&cq->cqe_ci, ticket + 1)` 原子推进 consumer index。
- 超时则把 4 个 debug 值写到 `cqe_debug`（magic `0x54494d454f5554` 是 `"TIMEOUT"` 的 ASCII），返回 `-110`（`kPollTimeoutStatus = -110`，对应 Linux `ETIMEDOUT`）。

#### 4.2 `gpunetio_read_probe_kernel`（`gpunetio_probe.cu:54-163`）

```cpp
__global__ void gpunetio_read_probe_kernel(GpuNetioReadProbeParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (params.remote_region_count == 0 || params.qp_array == nullptr ||
      params.qp_index >= params.qp_count ||
      params.remote_region >= params.remote_region_count ||
      params.qp_array[params.qp_index] == nullptr) {
    *params.status_code = -EINVAL;
    return;
  }
  ...
}
```

单线程 kernel（`<<<1, 1>>>`，见 `gpunetio_probe.cu:169`）。开头是参数合法性检查，任何一项非法直接写 `-EINVAL` 到 status_code 返回。

接下来是大段 debug 值采集（`gpunetio_probe.cu:63-82`），把远端 region 信息、QP 内部状态（`need_dump`、`nic_handler`、`mem_type`、`sq_rsvd_index`、`sq_ready_index`、`sq_wqe_pi`、`sq_wqe_daddr`、`sq_dbrec`、`sq_db`、`cqe_daddr`、`sq_wqe_num/mask`、`cqe_num`）全部写到 `debug_values`。这些值在 probe 失败时会被 host 端打印出来，用于诊断。

##### 4.2.1 提交第一次 RDMA Read（`gpunetio_probe.cu:83-99`）

```cpp
doca_gpu_dev_verbs_ticket_t ticket =
  doca_gpu_dev_verbs_atomic_read<uint64_t,
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_wqe_pi);
auto* probe_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
doca_gpu_dev_verbs_wqe_prepare_read(
  qp,
  probe_wqe_ptr,
  ticket,
  DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
  region.address,
  region.rkey,
  reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base,
  params.local_mkey,
  sizeof(uint64_t));
doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
  qp, ticket + 1);
```

- `doca_gpu_dev_verbs_atomic_read(&qp->sq_wqe_pi)` 原子读当前 producer index（ticket）。
- `doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket)` 算出 WQE 的 GPU 虚拟地址。
- `doca_gpu_dev_verbs_wqe_prepare_read` 把 RDMA Read 的 WQE 字段填到 WQE 内存里：远端 addr/rkey、本端 IOVA offset、本端 mkey、读 8 字节。`DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE` flag 要求 NIC 完成后写 CQE。
- `doca_gpu_dev_verbs_submit<EXCLUSIVE>(qp, ticket + 1)` 把 producer index 推到 `ticket+1` 并敲 doorbell。`EXCLUSIVE` 模式表示当前线程独占该 QP（已经持锁，probe 是单线程所以天然独占）。

注意本端地址 `reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base`：peer_memory 模式下 `local_iova_base=0`，这就是 GPU 虚拟地址；dmabuf 模式下 `local_iova_base=registered_base`，这就是相对偏移。和 3.4.2 节描述的两种 IOVA 模型完全对应。

##### 4.2.2 采集 post-submit 状态并 poll CQ（`gpunetio_probe.cu:100-118`）

```cpp
params.debug_values[4] = ticket;
params.debug_values[18] = qp->sq_rsvd_index;
params.debug_values[19] = qp->sq_ready_index;
params.debug_values[20] = qp->sq_wqe_pi;
params.debug_values[21] = *qp->sq_dbrec;
const auto* probe_wqe = reinterpret_cast<const uint64_t*>(
  qp->sq_wqe_daddr + ((ticket & qp->sq_wqe_mask) << DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT));
for (uint32_t i = 0; i < 8; ++i) {
  params.debug_values[22 + i] = probe_wqe[i];
  params.debug_values[30 + i] = probe_wqe[i];
}
const int status = poll_cq_at_with_timeout<
  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
    cq, ticket, params.debug_values + 6);
params.debug_values[39] = static_cast<uint32_t>(status);
if (status != 0) {
  params.debug_values[38] = 1;
  *params.status_code = status;
  return;
}
```

post-submit 再采集一遍 SQ 状态（`sq_rsvd_index`、`sq_ready_index`、`sq_wqe_pi`、`sq_dbrec`），并 hex dump WQE 的前 8 个 uint64（64 字节，正好一个完整 mlx5 WQE）。`debug_values[30..37]` 是 `debug_values[22..29]` 的副本，留给第二次 read 之后覆盖 —— 这样 host 端能看到两次 WQE 的对比。

`poll_cq_at_with_timeout` 用 ticket 等待这次 read 的 CQE。失败则写 `debug_values[38]=1`（probe 第一阶段）+ status_code，返回。

##### 4.2.3 第二次 Read（带 optional dump）（`gpunetio_probe.cu:120-163`）

```cpp
const doca_gpu_dev_verbs_ticket_t read_ticket = qp->sq_wqe_pi;
auto* read_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, read_ticket);
doca_gpu_dev_verbs_wqe_prepare_read(
  qp, read_wqe_ptr, read_ticket,
  DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
  region.address, region.rkey,
  reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base,
  params.local_mkey, sizeof(uint64_t));
doca_gpu_dev_verbs_ticket_t final_ticket = read_ticket;
if (qp->need_dump) {
  final_ticket = read_ticket + 1;
  auto* dump_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_ticket);
  doca_gpu_dev_verbs_prepare_dump(
    qp, dump_wqe_ptr, final_ticket,
    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    reinterpret_cast<uint64_t>(params.dump_ptr) - params.local_iova_base,
    params.local_mkey, 1);
}
doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
  qp, final_ticket + 1);
int dump_status = poll_cq_at_with_timeout<...>(cq, read_ticket, params.debug_values + 6);
if (dump_status == 0 && final_ticket != read_ticket) {
  dump_status = poll_cq_at_with_timeout<...>(cq, final_ticket, params.debug_values + 6);
}
params.debug_values[4] = final_ticket;
params.debug_values[38] = 2;
params.debug_values[39] = static_cast<uint32_t>(dump_status);
*params.status_code = dump_status;
```

第二次 read 和第一次几乎一样，目的是验证 QP 能连续完成多次 read（排空 WQE 之后的复用）。区别在于：如果 `qp->need_dump` 为真（NIC 需要 dump 模式），会在 read 之后追加一个 dump WQE（`doca_gpu_dev_verbs_wqe_prepare_dump`），把 NIC 内部状态 dump 到 `params.dump_ptr`。`final_ticket` 取决于是否追加了 dump WQE。

`poll_cq_at_with_timeout` 要 poll 两次：先等 read 的 CQE（`read_ticket`），如果还有 dump 则再等 dump 的 CQE（`final_ticket`）。

`debug_values[38]=2` 标记 probe 第二阶段，status_code 最终反映第二次 read 的状态。

这就是 probe 的全貌：两次 read + 可选 dump，全部完成且 status=0 才算 QP 可用。

### 5. `tools/gpunetio_probe.cc`：独立 capability probe 工具

这个工具（`tools/gpunetio_probe.cc:94-200`）和 `gpunetio_read_probe_kernel` 不是一回事 —— 它**不创建 QP、不做 RDMA Read**，只验证 GPUNetIO 的"装配能力"：

1. `cudaSetDevice` + `cudaFree(0)` + `cudaDeviceGetPCIBusId`（`gpunetio_probe.cc:114-118`）。
2. `find_device` 找到指定 ibdev 的 devinfo（`gpunetio_probe.cc:120-121`）。
3. `doca_verbs_context_create` + `doca_verbs_query_device` 查设备能力（`gpunetio_probe.cc:122-128`）。
4. `doca_verbs_device_attr_get_is_gpu_external_datapath_supported` 验证 NIC 支持 GPU external datapath（`gpunetio_probe.cc:129-131`）—— 这一项不支持就根本没法做 GPUNetIO QP。
5. `doca_verbs_device_attr_get_is_qp_type_supported(RC)` 验证 RC QP 支持（`gpunetio_probe.cc:132-134`）。
6. `doca_verbs_pd_create` + `doca_verbs_pd_as_doca_dev`（`gpunetio_probe.cc:135-140`）。
7. `doca_gpu_create`（`gpunetio_probe.cc:141`）。
8. `doca_gpu_mem_alloc` 分配 GPU 内存（`gpunetio_probe.cc:144-146`）。
9. `doca_gpu_dmabuf_fd` 拿 dmabuf fd（如果 mode=dmabuf）（`gpunetio_probe.cc:147-151`）。
10. `doca_umem_gpu_create` 验证 GPU umem 注册（`gpunetio_probe.cc:152-157`）—— **这一步就是 g201 上会失败的那一步**。
11. 两种 MR 注册路径（`gpunetio_probe.cc:165-181`）：
    - `dmabuf` 模式：`mlx5dv_reg_dmabuf_mr`。
    - `peer` 模式：`ibv_reg_mr`（依赖 nvidia-peermem）。

工具按 64 KiB 的 chunk 分段注册（`registration_bytes` 参数，必须是 64 KiB 的倍数），验证大块 GPU 内存能否分段注册。最后打印一条总结：GPU PCI、ibdev、external datapath supported、RC supported、umem registration passed、MR registration passed、mode、字节数、MR 数、第一个 lkey。

用法（从 `gpunetio_probe.cc:96-102`）：

```
gpunetio_probe <gpu_index> <ibdev_name> <allocation_bytes> <registration_bytes> <registration_mode>
```

典型用法：`gpunetio_probe 0 mlx5_0 65536 65536 peer` 验证 peermem 路径；`gpunetio_probe 0 mlx5_0 65536 65536 dmabuf` 验证 dmabuf 路径。

### 6. `tools/gpunetio_loopback_probe.cc`：loopback 压力 probe

这个工具更进一步，做真实的 loopback RDMA Read 压力测试。流程（`gpunetio_loopback_probe.cc:41-191`）：

1. 构造 `IndexConfiguration`、`Context`、`ClientConnectionManager`，连接存储节点（`gpunetio_loopback_probe.cc:42-45`）。
2. 发送 `Parameters{num_threads=1, gpu_rdma_qps}` 给存储端（`gpunetio_loopback_probe.cc:47-54`）。
3. 接收每条 server QP 的 `MemoryRegionToken`（`gpunetio_loopback_probe.cc:56-62`）—— 即远端 MR 的 addr/rkey。
4. 读环境变量 `DVSTOR_GPUNETIO_STRESS_BLOCKS` / `ITERATIONS` / `BATCH_READS`（`gpunetio_loopback_probe.cc:64-66`）。
5. 构造 `GpuNetioPersistentTransport`（`gpunetio_loopback_probe.cc:74-76`）—— **这一步会跑启动期 probe，和第 3.5 节描述一致**。
6. 分配 `stop` / `disabled` / `error` / `statuses` / `completed` 设备标志（`gpunetio_loopback_probe.cc:78-96`）。
7. 填 `PersistentKernelParams`（`gpunetio_loopback_probe.cc:97-115`），其中 `direct_*` 字段来自 `transport.view()`。
8. 根据 `batch_reads` 选 `launch_gpunetio_locked_read_probe`（单读）或 `launch_gpunetio_batched_read_probe`（批读）（`gpunetio_loopback_probe.cc:117-125`）。
9. 同步后回读 statuses / completed / disabled / error，校验 `host_completed == expected` 且所有 status==0（`gpunetio_loopback_probe.cc:127-163`）。
10. 通过则打印 `rate=... ops/s`（`gpunetio_loopback_probe.cc:165-169`）。
11. 与存储端 synchronize，发 `storage_startup::Request` 收 `Response`（`gpunetio_loopback_probe.cc:175-187`），确认 loopback 链路完全 OK。

这个工具的价值在于：它跑的是**真实的持久化 kernel**（`launch_gpunetio_locked_read_probe` / `launch_gpunetio_batched_read_probe`，与查询热路径用的 kernel 是同一套），只是把数据替换成一个 `u64`。相当于一次"干跑"，在不跑真实查询的前提下验证整条 GPU→NIC→远端 MR→CQ→kernel 链路的吞吐和正确性。

---

## 关键数据结构与流程图

### 对象创建时序图

```
host (CPU)                                GPU device
==========                                ==========
cudaSetDevice + cudaFree(0)
  |
cudaDeviceGetPCIBusId ----+
                          |
doca_devinfo_create_list  |
  |                       |
doca_verbs_context_create |
  |                       |
doca_verbs_pd_create      |
  |                       |
doca_verbs_bridge_verbs_pd_get_ibv_pd
  |                       |
doca_rdma_bridge_open_dev_from_pd ---> dev
  |                       |
doca_gpu_create(gpu_pci) ---------> gpu
  |
  +-- for each (lane, server):
  |     |
  |     doca_gpu_mem_alloc(gpu, 128KiB) x4  ---> [GPU] send_cq_buf / recv_cq_buf / qp_wq_buf / qp_dbr_buf
  |     |                                              (GPU 显存)
  |     initialize_cq_owner_bits (cudaMemcpy H2D)
  |     |
  |     doca_umem_gpu_create(gpu, dev, send_cq_buf, ...)  ---> send_cq_umem
  |     doca_umem_gpu_create(gpu, dev, recv_cq_buf, ...)  ---> recv_cq_umem
  |     doca_umem_gpu_create(gpu, dev, qp_wq_buf, ...)    ---> qp_wq_umem
  |     doca_umem_gpu_create(gpu, dev, qp_dbr_buf, ...)   ---> qp_dbr_umem
  |     |        ^ 需要 nvidia-peermem；否则失败（g201 问题）
  |     |
  |     doca_verbs_cq_create(...external_umem=send_cq_umem...)  ---> send_cq
  |     doca_verbs_cq_create(...external_umem=recv_cq_umem...)  ---> recv_cq
  |     doca_uar_create(dev, NONCACHE_DEDICATED)          ---> external_uar
  |     doca_verbs_qp_create(...qp_wq_umem, qp_dbr_umem, external_uar, send_cq, recv_cq, RC)
  |                                            ---> qp (qpn = doca_verbs_qp_get_qpn)
  |     |
  |     doca_gpu_verbs_export_qp(gpu, dev, qp, GPU_SM_DB, qp_wq_buf, send_cq, recv_cq)
  |                                            ---> gpu_qp (host handle)
  |     doca_gpu_verbs_get_qp_dev(gpu_qp)      ---> gpu_qp_dev (device handle)  [传给 kernel]
  |     doca_gpu_verbs_cpu_proxy_enabled(gpu_qp) == 0  (硬性要求)
  |     |
  |     qp_modify_to_init(qp)
  |     exchange_qp_info via CPU QP (cm.server_qps[server])
  |     qp_modify_to_rtr(qp, remote_info)
  |     qp_modify_to_rts(qp)
  |
  doca_gpu_mem_alloc(gpu, registered_bytes)  ---> [GPU] registered_base
  ibv_reg_mr(ibv_pd, registered_base, ...)   -- peer_memory 路径 (local_iova_base=0)
     | 失败则
     doca_gpu_dmabuf_fd + mlx5dv_reg_dmabuf_mr -- dmabuf 路径 (local_iova_base=registered_base)
  |
  bump 分配子区域: d_probe_value / d_dump / d_probe_status / d_probe_debug / persistent_data
  |
  cudaMemcpy remote_regions_host -> d_remote_regions
  cudaMemcpy gpu_qp_devices_host -> d_qp_array
  cudaMemset  d_qp_locks = 0
  |
  for each qp_index:
     launch_gpunetio_read_probe (<<<1,1>>>) on stream
        |  kernel 内部: 取 ticket -> prepare_read WQE -> submit -> poll_cq_at_with_timeout
        |              -> 第二次 read (+ optional dump) -> poll CQ
     cudaStreamSynchronize
     if status != 0: throw (打印 39 个 debug 值)
  |
  view() -> GpuNetioPersistentView  (传给 PersistentKernelParams)
```

### CPU/GPU 数据通路图

```
                      查询热路径（kernel 内）
                      =====================

   GPU SM (warp/lane)
      |
      |  1. lock_direct_qp(qp_locks[qp_index])   # 自旋锁
      |
      |  2. ticket = qp->sq_wqe_pi               # 读 producer index
      |     wqe = get_wqe_ptr(qp, ticket)        # 算 WQE 地址
      |     wqe_prepare_read(wqe, remote_addr, rkey,
      |                       local_iova, mkey, bytes)
      |
      |  3. submit(qp, ticket+1)                 # 写 sq_dbrec + 敲 doorbell (UAR)
      |
      v
   GPU 显存 (QP WQ umem) ----PCIe BAR----> NIC doorbell 寄存器
                                              |
                                              | NIC 解析 WQE,
                                              | 发起 RDMA Read
                                              v
                                          IB 网络
                                              |
                                              v
                                       远端存储节点 MR (见第 23 课)
                                              |
                                              | 数据 DMA 回来
                                              v
   GPU 显存 (registered_base 子区域) <---NIC DMA 写入---  (本端 GPU MR)
      |
      |  4. poll_cq_at_with_timeout(cq, ticket)
      |     - 松弛读 cqe->op_own
      |     - owner 轮转判断
      |     - fence_acquire + atomic_max(cq->cqe_ci, ticket+1)
      |
      |  5. unlock_direct_qp(qp_locks[qp_index])
      v
   GPU SM 继续 traversal (见第 20 课)


   建链/控制路径（CPU 参与）
   ========================
   CPU verbs QP (第 4/5 课)
      - exchange QPInfo (lid, qpn)
      - receive MemoryRegionToken (远端 addr/rkey)
      - storage_startup handshake
```

关键点：**热路径上 CPU 完全不参与**。GPU SM 直接写 WQE、敲 doorbell、poll CQE、消费数据。CPU 只在建链、probe、错误处理时介入。这与第 5 课 CPU verbs QP（每条 RDMA Read 都要 CPU post_recv / poll_cq）形成鲜明对比。

---

## 与其他模块的关系

### 与第 5 课（CPU verbs QP）的对比

| 维度 | 第 5 课 CPU QP | 第 22 课 GPU QP |
|---|---|---|
| CQ/WQ 内存位置 | host 内存 | GPU 显存（external umem） |
| WQE 写入者 | CPU（`ibv_post_send`） | GPU SM（`doca_gpu_dev_verbs_wqe_prepare_*`） |
| Doorbell 敲入者 | CPU（`ibv_post_send` 内部） | GPU SM（`doca_gpu_dev_verbs_submit` 写 UAR） |
| CQE 轮询者 | CPU（`ibv_poll_cq`） | GPU SM（`poll_cq_at_with_timeout` 松弛读 op_own） |
| 热路径 CPU 开销 | 每条 Read 一次系统调用 | 零 CPU |
| 建链方式 | CPU verbs Send/Recv | **复用** CPU verbs 通道（`cm.server_qps`）交换 QPInfo |
| MR 注册 | `ibv_reg_mr` host 内存 | `ibv_reg_mr` GPU mem（peermem）或 `mlx5dv_reg_dmabuf_mr`（dmabuf） |
| 状态机 | RESET→INIT→RTR→RTS | 同（`qp_modify_to_init/rtr/rts`） |
| 协议层 | RC / IB | RC / IB（完全相同） |

简单说：**协议层完全一致，datapath 物理位置从 host 搬到 GPU**。CPU verbs 的所有控制平面知识（PD、CQ、QP 状态机、QPInfo 交换）在第 22 课全部复用。

### 与第 19 课（RDMA cache 消费侧）的衔接

`rdma_cache.cuh:37-113` 的 `direct_fetch` 和 `rdma_cache.cuh:115-262` 的 `direct_fetch_batch` 是 GPUNetIO 的核心消费者。它们做的事和 `gpunetio_read_probe_kernel` 几乎一样：

1. `qp_index = (lane % direct_qps_per_node) * direct_region_count + memory_node`（`rdma_cache.cuh:45-46`）—— 和本课 `gpunetio_transport.cc:245-246` 的双重循环顺序对齐。
2. `lock_direct_qp` 获取 QP 自旋锁（`rdma_cache.cuh:59`）。
3. `read_ticket = qp->sq_wqe_pi` → `get_wqe_ptr` → `wqe_prepare_read` → 可选 `wqe_prepare_dump` → `submit<EXCLUSIVE>`（`rdma_cache.cuh:72-94`）。
4. `poll_direct_cq` 等待 CQE（`rdma_cache.cuh:95-97`）—— 实现和 `poll_cq_at_with_timeout` 同源。
5. `unlock_direct_qp` 释放（`rdma_cache.cuh:102`）。

区别在于：消费侧支持 batch（一次提交多个 read WQE，最后一个才带 `CQ_UPDATE` flag，见 `rdma_cache.cuh:215-217`），并且有 `direct_batch_queues` 异步路径（`rdma_cache.cuh:137-195`，由第 17 课的 device ring + 第 21 课的角色调度消费）。

### 与第 11 课（持久化引擎 lifecycle）的衔接

第 11 课描述的 PImpl 生命周期三段式：

1. **装配**：`construction.cc:376-377` 构造 `GpuNetioPersistentTransport`，期间完成所有 QP 创建、export、状态机迁移、MR 注册。
2. **校验**：构造函数末尾的 probe 循环（本课 3.5.2）—— 每条 QP 必须完成一次真实 RDMA Read，否则抛异常。这是 lifecycle 校验阶段的硬性 gate。
3. **启动**：校验通过后，`construction.cc:908-920` 把 `direct_view` 字段填进 `PersistentKernelParams`，持久化 kernel 启动后即可用。

probe 失败会直接抛 `std::runtime_error`，整个引擎不会启动 —— 这是 dvstor 的 fail-stop 设计：宁可启动失败也不要运行时才发现 QP 不可用。

### 与第 17/21 课（kernel 启动器/角色调度）的关系

`PersistentKernelParams.direct_*` 字段（`construction.cc:908-920`）是第 17 课 kernel 启动器构造的参数块的一部分。第 21 课的角色调度把 query worker / batch owner / delta worker 等角色映射到 warp，每个角色都可能调用 `direct_fetch` / `direct_fetch_batch`。QP 自旋锁（`direct_qp_locks`）保证同一 QP 同时只有一个 warp 操作 —— 这是第 21 课角色调度能并发跑多个 query 的前提。

### 与第 23 课（存储节点主体/peer RDMA）的关系

GPUNetIO QP 的对端是第 23 课的存储节点。存储节点用第 5 课的 CPU verbs 注册 MR，把 addr/rkey 通过 `MemoryRegionToken` 发给计算节点（`gpunetio_loopback_probe.cc:56-62` 的接收流程）。计算节点的 GPUNetIO QP 用这些 token 作为 `remote_regions`，kernel 发起的 RDMA Read 直接打到存储节点的 CPU MR。存储节点不需要 GPUNetIO —— 它只被动响应 Read，CPU verbs 足够。

---

## 小结

本课拆解了 dvstor 把 RDMA datapath 搬到 GPU 的完整装配过程：

1. **设备/GPU 初始化**（块 A）：ibdev 选择、CUDA context 建立、`doca_verbs_context_create` → `pd` → `ibv_pd` → `dev` → `gpu` 的依赖链。
2. **GPU-backed CQ + QP 创建**（块 B）：每 (lane × server) 一对，4 块 GPU umem（send_cq/recv_cq/qp_wq/qp_dbr）+ 1 个 UAR + 1 个 RC QP，`doca_gpu_verbs_export_qp` 导出 device 句柄，CPU-proxy 模式被硬性拒绝。
3. **本端 GPU MR 注册**（块 C）：peer_memory 优先（`ibv_reg_mr` + nvidia-peermem），dmabuf 兜底（`mlx5dv_reg_dmabuf_mr`），两种 IOVA 模型（`local_iova_base=0` vs `=registered_base`）。
4. **启动期 probe**（块 D）：每条 QP 跑 `gpunetio_read_probe_kernel`，完成两次真实 RDMA Read + 可选 dump，验证整条链路，失败即抛异常阻止引擎启动。
5. **独立工具**：`tools/gpunetio_probe.cc` 验证装配能力（umem 注册、MR 注册两种路径）；`tools/gpunetio_loopback_probe.cc` 跑真实持久化 kernel 的 loopback 压力测试。

核心要点：

- **nvidia-peermem 是 GPUNetIO 的硬性依赖**。`doca_umem_gpu_create` 需要它建立 GPU→NIC 的 P2P 映射；g201 缺 peermem 导致这一步失败，这就是 RDMA 集群必须在 .202 上跑的根本原因（见记忆 `g201-doca-peermem-missing.md`）。
- **热路径零 CPU**。GPU SM 直接写 WQE、敲 doorbell、poll CQE；CPU 只在建链、probe、错误处理时介入。
- **probe 是 fail-stop 的关键**。装配链路太长，任何一步静默失败都会让 kernel 跑出错误数据或死循环；probe 强制每条 QP 完成真实 Read，把问题挡在启动期。
- **与第 5 课 CPU verbs 的关系是"协议层复用、datapath 替换"**。PD、CQ、QP 状态机、QPInfo 交换全部复用 CPU verbs 知识；只有 WQE/CQE/doorbell 的物理位置和读写者变了。
- **manual WQE + locked 是 dvstor 的实现选择**，而非 GPUNetIO 高层 API。原因是为了精细控制 batch read 的 WQE flag（`CQ_UPDATE` vs `CQ_ERROR_UPDATE`）和 QP 互斥。

下一课（第 23 课）将转向存储节点主体，看对端如何用 CPU verbs 注册 MR、响应 RDMA Read、并通过 peer RPC 与计算节点协作。
