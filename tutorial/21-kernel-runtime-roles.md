# 第 21 课：Kernel 运行时与角色调度

> 本课是 Part IV "持久化 CUDA Kernel" 的收官。第 17 课介绍了 device ring 与 launch 握手的基础设施，第 18 课讲解了候选评分，第 19 课讲解了 RDMA cache 与 GPUNetIO 直读原语，第 20 课讲解了查询遍历主循环。本课把这些零件组装起来——一个 `__global__` kernel 如何按 `blockIdx` 分流成 "owner / query / dispatcher / control-delta" 四种角色，每个角色如何在自己的主循环里工作，delta 命令如何被逐字段校验并发布成快照。这是把"GPU 当成一个常驻服务进程"的最后一块拼图。

## 本课目标与涉及文件

本课只讲一件事：**`persistent_search_kernel` 这个 kernel 的运行时控制流**。读完本课你应当能回答：

1. 同一个 `__global__` 函数，如何让不同的 CTA 同时充当 GPUNetIO owner warp、查询 CTA、提交 ring 分发器、delta 控制路径？
2. CPU 侧怎么知道 GPU kernel 已经准备好接收请求？非 unified 模式与 unified 模式的握手有什么差别？
3. dispatcher 角色如何把 CPU 提交的 `QueryDescriptor` 从 `submissions` ring 转发到 device-local 的 `device_submissions` ring？为什么要这样转发？
4. delta 控制路径如何处理 `kDeltaCommandReset` 与 `kDeltaCommandPromoteOverrides` 两种命令？那条长达几十行的校验链在防什么？
5. delta 编码、resident_pq 表、dynamic_route 槽的发布顺序为什么必须是"先奇后偶"的 seqlock？发布失败和发布成功分别如何回写 `DeltaPublishCompletion`？
6. owner warp 怎么批量提交 DOCA verbs WQE、怎么 poll CQ、怎么用 `direct_owner_phases` 把内部状态暴露给 CPU 调试？

涉及文件：

- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/runtime.cuh`（约 1048 行，本课主角）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel.cu`（薄薄的 launch 包装）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel.hh`（`PersistentKernelParams` 这个大参数结构，第 17 课已介绍）
- `/home/xjs/experiment/dvstor/src/gpu_search/device_ring.cuh`（`DeviceRingView` 与 `device_ring_try_pop/push/relax`，第 17 课已介绍）
- `/home/xjs/experiment/dvstor/src/gpu_search/types.hh`（`DeltaPublishDescriptor`、`kDeltaCommand*`、`DeviceDynamicRouteSlot` 等）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/candidate_scoring.cuh`（`hash32/hash64/anchor_graph_slot/erase_resident_pq/insert_resident_pq/unlink_mutable_delta/poll_direct_cq`）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/rdma_cache.cuh`（`direct_fetch/direct_fetch_batch`，第 19 课已介绍）
- `/home/xjs/experiment/dvstor/src/gpu_search/persistent_kernel/query_traversal.cuh`（`process_query`，第 20 课已介绍）

## 逐文件逐函数讲解

### 1. `persistent_search_kernel` 主体：按 blockIdx 分流角色

整个持久化引擎的入口是一个普通的 `__global__` 函数（`runtime.cuh:11`）：

```cpp
__global__ void persistent_search_kernel(PersistentKernelParams params) {
  const bool unified_dispatch = params.direct_owner_block_count != 0;
  if (unified_dispatch && blockIdx.x < params.direct_owner_block_count) {
    direct_read_owner_loop(params, params.direct_batch_queue_count, blockIdx.x);
    return;
  }
  ...
}
```

注意 `PersistentKernelParams` 是按值传递的。这很关键：每个 CTA 拿到的是结构体的一份私有拷贝，对其中字段的写不会泄漏到别的 CTA（参数结构体本身没有改动，里面的指针仍指向同一块 device/global 显存）。按值传还让编译器可以把 `params` 放到 local memory / register，减少对 global memory 的反复 load。

第一行 `unified_dispatch` 的判据是 `params.direct_owner_block_count != 0`。这是一个"两合一"开关：

- **非 unified 模式**（`direct_owner_block_count == 0`）：GPUNetIO owner 由一个独立的 kernel `direct_read_owner_kernel` 启动（见 `runtime.cuh:940`），`persistent_search_kernel` 只负责查询与 delta。这种模式用于没有 GPUNetIO 直读能力、或者想把直读与查询放在不同 stream 调度的场景。
- **unified 模式**（`direct_owner_block_count != 0`）：所有角色共用一个 kernel launch。前 `direct_owner_block_count` 个 block 当 owner，剩下的 block 在 query / dispatcher / delta 三种角色里再分流。好处是只 launch 一次、只握手一次、共享 `stop` 探测和 idle 退避状态机；坏处是 owner block 与 query block 的 threadIdx 配置必须兼容——本课后面会看到 owner 内部按 warp 分工，要求 `blockDim.x` 是 32 的倍数且不超过 256（8 warp）。

`runtime.cuh:13-16` 的早退分支：unified 模式下，前 `direct_owner_block_count` 个 block 直接跳进 `direct_read_owner_loop` 并 `return`，根本不走后面的角色分流逻辑。这等价于"在同一个 kernel 里嵌入了另一个 kernel"。

接下来是角色分流（`runtime.cuh:18-37`）：

```cpp
bool enable_queries = true;
bool enable_dispatcher = false;
bool enable_delta = true;
if (unified_dispatch) {
  const u32 role_block = blockIdx.x - params.direct_owner_block_count;
  enable_queries = role_block < params.query_block_count;
  enable_dispatcher = role_block == params.query_block_count;
  enable_delta = role_block == params.query_block_count + 1;
  if (!enable_queries && !enable_dispatcher && !enable_delta) return;
  if (threadIdx.x == 0) {
    u32* ready_count = enable_queries ? params.query_kernel_ready_count
      : enable_dispatcher ? params.dispatcher_kernel_ready_count
                         : params.control_kernel_ready_count;
    if (ready_count != nullptr) atomicAdd(ready_count, 1u);
    __threadfence_system();
  }
} else if (threadIdx.x == 0 && params.kernel_ready_count != nullptr) {
  atomicAdd(params.kernel_ready_count, 1u);
  __threadfence_system();
}
```

这段揭示了 unified 模式的 block 布局：

```
blockIdx.x 区间                         角色
[0, direct_owner_block_count)          direct_read_owner_loop (已早退)
[direct_owner_block_count,
 direct_owner_block_count + query_block_count)   查询 CTA (enable_queries)
direct_owner_block_count + query_block_count     dispatcher (单 block)
direct_owner_block_count + query_block_count + 1 control / delta (单 block)
再往后                                            return（多余 block 直接退出）
```

也就是说 dispatcher 与 delta 各自只占一个 block。这是有意为之——它们都是"串行状态机"角色，多 block 同时跑会争抢同一个 ring 的 producer/consumer 而互相打架；而查询是天然可并行的，所以分配 `query_block_count` 个 block。

注意默认值的不对称：`enable_queries = true`、`enable_delta = true`、`enable_dispatcher = false`。这是为了非 unified 模式下，每个查询 block 都默认同时承担 delta 控制职责（后面会看到，主循环里 delta 优先于 query 被检查）。unified 模式则把 delta 收敛到唯一一个 block，避免多 block 同时操作 `delta_count`、`base_override` 表、`dynamic_route_slots`。这也是为什么 unified 模式必须保证 `role_block == query_block_count + 1` 的那个 block 真的存在——否则 delta 永远没人处理。

`threadIdx.x == 0` 的那段 `atomicAdd(ready_count, 1u)` + `__threadfence_system()` 是 CPU 侧启动握手的关键（见第 17 课）。CPU launch kernel 后会自旋等待三个 ready 计数器（或非 unified 模式下的单一 `kernel_ready_count`）达到预期值，确认 GPU 已进入主循环才开始往 ring 里 push 请求。`__threadfence_system()` 保证"我已 ready"这个写对 CPU 可见——`atomicAdd` 本身只是 device-scope，不跨 PCIe，所以必须显式 fence。这也是整个文件里反复出现 `__threadfence_system()` 的原因：凡是要让 CPU 看到的状态变更，都得跟一条 system fence。

### 2. `__shared__` 协调变量与 `__threadfence_system`

`runtime.cuh:38-51` 声明了一组 `__shared__` 变量，它们是 CTA 内所有线程协作的"黑板"：

```cpp
__shared__ QueryDescriptor descriptor;
__shared__ QueryDescriptor dispatch_descriptor;
__shared__ DeltaPublishDescriptor delta_descriptor;
__shared__ u32 have_submission;
__shared__ u32 dispatch_pending;
__shared__ u32 have_delta_submission;
__shared__ u32 stop_requested;
__shared__ u32 idle_cycles;
__shared__ i32 delta_status;
if (threadIdx.x == 0) {
  dispatch_pending = 0;
  idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
}
__syncthreads();
```

这里有个值得注意的设计：**每个 shared 变量都只由 `threadIdx.x == 0` 写、由其它线程读**。这是一种简化的一致性模型——不用 `atomicAdd`/`atomicCAS` 在 shared memory 上做无锁同步，而是用"单写者 + `__syncthreads()`"的模式。代价是同一个 CTA 内的所有线程必须以同一节奏前进（每个分支后都有 `__syncthreads()`），收益是逻辑极度清晰，且 shared atomic 在某些 GPU 架构上比 global atomic 慢。

`idle_cycles` 的初值 `256u + ((blockIdx.x * 131u) & 1023u)` 是有意的"去同步"——不同 block 的初始退避时长错开（131 是素数，与 1024 互质，能均匀打散），避免所有查询 CTA 在空转时同步醒来同时去抢同一个 ring。这是一个微小的优化，但在高 QPS 场景下能显著减少 ring 的竞争热点。最大值被钳到 `16384u`（`runtime.cuh:77`、`runtime.cuh:689`），对应 `__nanosleep` 的最大有意义值。

`stop_requested` 是唯一会被外部（CPU 写 `params.stop`）改变的 shared 变量。它在主循环顶部由 thread 0 读：

```cpp
for (;;) {
  if (threadIdx.x == 0) {
    stop_requested = *reinterpret_cast<volatile u32*>(params.stop);
  }
  __syncthreads();
  if (stop_requested != 0) return;
  ...
}
```

`volatile` 强制每次都真的从 global memory 读（不走 cache），保证 CPU 一旦把 `*params.stop` 写成非 0，最多在一轮主循环内被 GPU 发现。这是优雅停机的核心机制——CPU 想停 kernel 时不用 cudaDeviceSynchronize 这种粗暴手段，只要写一个 u32。

### 3. 主循环骨架与角色优先级

主循环的结构是（去掉角色分支后的骨架）：

```
loop:
  读 stop -> 若停则 return
  if enable_dispatcher: 取 submissions -> push device_submissions；continue
  取 delta_submissions -> 若有则处理 delta 命令；continue
  取 query_queue -> 若有则 process_query；continue
  否则 idle 退避
```

**角色优先级是 delta > query**（dispatcher 角色已经 `continue` 走了，不参与后续）。这个顺序很重要：delta 命令改变 `delta_count` 和 `snapshot_epoch`，如果先处理 query 再处理 delta，query 可能基于旧快照跑完后才看到新 delta，导致同一批查询结果不一致。先 drain delta，能保证后续 query 看到的快照是"截至本轮循环开始时最新的"。

注意 dispatcher 角色在 `runtime.cuh:58-82` 完全独立，它进入分支后直接 `continue`，不会触碰 delta 或 query 路径。这是因为它操作的是另一个 ring（`submissions` -> `device_submissions`），与查询消费的 ring 没有交集，混在一起反而容易出 bug。

### 4. dispatcher 角色：从 submissions 到 device_submissions

dispatcher 的完整逻辑（`runtime.cuh:58-82`）：

```cpp
if (enable_dispatcher) {
  if (threadIdx.x == 0) {
    bool progressed = false;
    if (dispatch_pending == 0 && params.submissions.entries != nullptr &&
        device_ring_try_pop(params.submissions, dispatch_descriptor)) {
      dispatch_pending = 1;
      progressed = true;
    }
    if (dispatch_pending != 0 &&
        params.device_submissions.entries != nullptr &&
        device_ring_try_push(params.device_submissions,
                             dispatch_descriptor)) {
      dispatch_pending = 0;
      progressed = true;
    }
    if (progressed) {
      idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
    } else {
      device_ring_relax(idle_cycles);
      idle_cycles = min(idle_cycles * 2, 16384u);
    }
  }
  __syncthreads();
  continue;
}
```

这是经典的"搬运工"模式：从 CPU 可写的 `submissions` ring pop 一个 `QueryDescriptor`，push 到 device-local 的 `device_submissions` ring。两步分离是因为：

1. `submissions` 的 producer 是 CPU（通过 PCIe 写入），consumer 是 GPU。`device_ring_try_pop` 内部用 `ld.acquire.sys`（见 `device_ring.cuh:32`）读取 sequence，跨 PCIe 保证 CPU 的 release 写对 GPU 可见。
2. `device_submissions` 的 producer 和 consumer 都是 GPU（dispatcher block 与 query block），不需要 system-scope 的 acquire/release，但仍要 device-scope 的顺序保证。
3. 两步分离让 dispatcher 可以"积压"一个 `dispatch_pending=1` 的 descriptor——如果 device ring 暂时满了，descriptor 留在 shared memory 里等下一轮再 push，而 submissions ring 的 slot 已经被 release（`try_pop` 成功时已 store_release sequence），CPU 可以继续往 submissions 里写下一个。

`progressed` 这个布尔标志驱动退避状态机：只要这一轮 pop 或 push 任一成功，`idle_cycles` 立即复位到 256~1280 的初始区间；只有两步都失败时才翻倍退避。这避免了 ring 有流量但偶发空轮询时进入长睡眠。

注意 `dispatch_descriptor` 是 `__shared__`，跨 `__syncthreads()` 持久存活。但 dispatcher 分支里只有 thread 0 在干活，其它线程在 `__syncthreads()` 后立即 `continue`——这是因为 dispatcher 本质是单线程状态机，没必要 warp 并行。

### 5. control/delta 角色：取 DeltaPublishDescriptor

`runtime.cuh:83-88` 是 delta 入口：

```cpp
if (threadIdx.x == 0) {
  have_delta_submission = enable_delta &&
    params.delta_submissions.entries != nullptr &&
    device_ring_try_pop(params.delta_submissions, delta_descriptor) ? 1u : 0u;
}
__syncthreads();
if (have_delta_submission != 0) {
  ... // 几百行 delta 处理
  continue;
}
```

`enable_delta` 的短路保证：非 unified 模式下，所有查询 block 都 `enable_delta = true`，但只有第一个成功 `try_pop` 的 block 会真正处理 delta；其它 block 在 `try_pop` 失败时 `have_delta_submission = 0`，跳过整段。这是 lock-free MPSC 模式——多个 block 争抢同一个 ring 的消费权，`device_ring_try_pop` 内部用 `atomicCAS(dequeue_position, position, position+1)` 保证只有一个 block 抢到。

抢到的 block 进入下一段：在 thread 0 上做命令校验。

### 6. delta 命令的合法性与 flags 解析

`runtime.cuh:90-168` 是整个文件最长的单段校验逻辑。先看头部（`runtime.cuh:90-95`）：

```cpp
if (threadIdx.x == 0) {
  delta_status = 0;
  const bool reset = (delta_descriptor.flags & kDeltaCommandReset) != 0;
  const bool promote =
    (delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0;
  constexpr u32 known_flags =
    kDeltaCommandReset | kDeltaCommandPromoteOverrides;
```

`kDeltaCommandReset = 1u`、`kDeltaCommandPromoteOverrides = 1u << 1`（见 `types.hh:117-118`）。两个 bit 互斥（后面 `reset && promote` 会被判非法）。`known_flags` 是个掩码，用来检测"未知 flag bit"——任何不在已知集合内的 bit 都让命令非法，这是前向兼容保护：未来加新 flag 时，旧 GPU kernel 会拒绝而不是默默忽略。

接下来是一条长达 70 行的布尔表达式（`runtime.cuh:97-165`），形式是：

```cpp
if ((delta_descriptor.flags & ~known_flags) != 0 ||
    (reset && promote) ||
    (reset && (... 一堆 reset 模式下的非法字段检查 ...)) ||
    (!reset && (... 一堆 publish 模式下的字段/指针检查 ...))) {
  delta_status = -EINVAL;
}
```

它把校验分成三块：

**a. 通用校验**：未知 flag、reset 与 promote 同时出现。

**b. reset 模式校验**（`runtime.cuh:99-120`）：reset 命令应当清空 delta 区，因此：
- `first_slot != 0`：reset 必须从 0 开始扫，不允许部分清。
- `record_count > params.delta_capacity`：要清的记录数不能超过容量。
- `final_count / invalidation_count / superseded_count / override_count / durable_count / resident_pq_erase_count / dynamic_route_count`：reset 不应携带任何业务更新，这些字段必须全为 0。
- 一堆 `params.* == nullptr` 检查：reset 要触碰 delta_records / delta_next / delta_prev / delta_remote_positions / delta_count / base_override_keys / base_override_epochs / delta_remote_keys / delta_remote_slots 等表，任何一块缺失都无法完成 reset。
- `params.anchor_count != 0 && params.delta_bucket_heads == nullptr`：如果有 anchor 但没有 bucket_heads 数组，无法清 bucket 链表头。

**c. publish 模式校验**（`runtime.cuh:121-165`）：非 reset 命令要发布 delta，必须保证：
- `final_count <= delta_capacity`、`record_count <= delta_capacity`：不能越界。
- 如果 `record_count != 0`，则 `delta_staging_slots`、`delta_staging_records` 必须非空（要有 staging 数据）。
- 如果 `record_count != 0` 且 `params.delta_remote_positions == nullptr || params.delta_remote_capacity == 0`：发布 record 必然要更新 remote 表，所以 remote 表必须就绪。
- 如果有 record 且 `vector_bytes != 0`：staging 与目标 vector 缓冲都必须在。
- 如果有 record 且 `pq_code_bytes != 0`：PQ 编码所需的 centroids、scratch、resident_pq 表等一整套都必须就绪。
- `durable_count != 0 && delta_durable_updates == nullptr`：durable 更新数组必须存在。
- `resident_pq_erase_count != 0`：resident_pq 表与 erase updates 都必须就绪。
- `dynamic_route_count != 0`：dynamic_route_updates / code_updates / slots / pq_codes / capacity 都必须就绪，且 `dynamic_route_count <= dynamic_route_capacity`。
- `promote && override_count != 0`：promote 模式要写 `permanent_override_bits`，所以必须存在。
- `!promote && override_count != 0`：非 promote 模式要写 base_override 表，所以 keys/epochs/capacity 必须就绪。

这条校验链的目的是**把"kernel 中段 panic"转成"早期 EINVAL 返回"**。GPU kernel 不能抛异常也不能优雅 unwind，一旦中段访问空指针就是非法内存访问 → cuda launch fail → 整个持久化引擎崩溃。所以在动任何指针前把所有"业务约束 + 指针就绪"一次性查完，是务实的选择。校验由 thread 0 独自完成（`runtime.cuh:90`），其它线程在 `__syncthreads()` 后才参与工作（`runtime.cuh:169`）。

### 7. kDeltaCommandReset 处理：清空 delta 区

校验通过后（或校验失败也走这条路径，因为 reset 失败也要回 completion），`runtime.cuh:171-229` 处理 reset：

```cpp
if ((delta_descriptor.flags & kDeltaCommandReset) != 0) {
  if (delta_status == 0) {
    for (u32 index = threadIdx.x; index < delta_descriptor.record_count;
         index += blockDim.x) {
      const DeviceDeltaRecord record = params.delta_records[index];
      const u32 remote_position = params.delta_remote_positions[index];
      if (record.remote_node != 0 &&
          remote_position < params.delta_remote_capacity &&
          load_cg(params.delta_remote_slots + remote_position) == index) {
        atomicCAS(reinterpret_cast<unsigned long long*>(
                    params.delta_remote_keys + remote_position),
                  record.remote_node, kDeltaRemoteTombstone);
        atomicExch(params.delta_remote_slots + remote_position, UINT32_MAX);
      }
      if (record.base_ordinal < params.num_nodes) {
        const u32 mask = params.base_override_capacity - 1;
        u32 position = hash32(record.base_ordinal) & mask;
        for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
          const u32 key = load_cg(params.base_override_keys + position);
          if (key == record.base_ordinal) {
            if (atomicCAS(params.base_override_keys + position,
                          record.base_ordinal,
                          kBaseOverrideTombstone) == record.base_ordinal) {
              params.base_override_epochs[position] = 0;
            }
            break;
          }
          if (key == kBaseOverrideEmpty) break;
          position = (position + 1) & mask;
        }
      }
      params.delta_records[index] = {};
      params.delta_records[index].base_ordinal = kBaseOverrideEmpty;
      params.delta_next[index] = UINT32_MAX;
      params.delta_prev[index] = UINT32_MAX;
      params.delta_remote_positions[index] = UINT32_MAX;
    }
    for (u32 index = threadIdx.x; index < params.anchor_count;
         index += blockDim.x) {
      params.delta_bucket_heads[index] = UINT32_MAX;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && delta_status == 0) {
    __threadfence();
    atomicExch(params.delta_count, 0u);
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    device_ring_push(params.delta_completions, DeltaPublishCompletion{
      .command_id = delta_descriptor.command_id,
      .status = delta_status,
      .final_count = 0,
    });
  }
  __syncthreads();
  continue;
}
```

逐段看：

**a. 清 delta_remote 表**（`runtime.cuh:177-184`）：对每个 delta record，若它登记过 remote_node，就用 `atomicCAS` 把 `delta_remote_keys[position]` 从 `remote_node` 改成 `kDeltaRemoteTombstone`（墓碑）。`load_cg` 是 cache-bypass load（`candidate_scoring.cuh:80-90`），避免把整个 keys 表读进 L2——reset 是冷数据全扫。`atomicCAS` + `atomicExch` 的组合是经典的"标记 + 清"两步：先把 key 标成 tombstone 让查询侧的 lookup miss，再把 slot 指针清成 `UINT32_MAX`。查询侧（见 `candidate_scoring.cuh:249-260` 的 `delta_slot_from_raw`）遇到 tombstone 会继续 probe，遇到 `kDeltaRemoteEmpty` 会停止——所以这里不能用 Empty，否则后续 probe 会被错误截断。

**b. 清 base_override 表**（`runtime.cuh:185-201`）：若 record 的 `base_ordinal` 是有效节点（`< num_nodes`），在 open-addressing 哈希表里线性探测找到它的位置，CAS 成 tombstone 并把 epoch 清 0。`hash32` 是一个 32-bit 的混合哈希（`candidate_scoring.cuh:62-69`，使用 `0x7feb352d` / `0x846ca68b` 两个 magic），分布均匀。`kBaseOverrideEmpty` 的早退保证不会越过真正的空槽——空槽意味着这条 key 从未插入过。

**c. 重置 delta_records/next/prev/remote_positions**（`runtime.cuh:202-206`）：把每个 slot 恢复成"未使用"状态，`base_ordinal` 设为 `kBaseOverrideEmpty` 作为哨兵。

**d. 重置 bucket_heads**（`runtime.cuh:208-211`）：每个 anchor bucket 的链表头设为 `UINT32_MAX`，整个 delta 链表就空了。

**e. 发布 delta_count = 0**（`runtime.cuh:214-218`）：`__threadfence()` 保证前面的写对全 GPU 可见，然后 `atomicExch(params.delta_count, 0u)` 发布新计数，`__threadfence_system()` 让 CPU 也能看到。查询侧用 `delta_count` 判断要不要扫 delta，0 就直接跳过。这一对 fence 是必须的：少了 `__threadfence()`，别的 GPU thread 可能先看到 `delta_count=0` 再看到残留的 `delta_records`；少了 `__threadfence_system()`，CPU 监控会拿到旧值。

**f. 写 completion**（`runtime.cuh:220-226`）：`device_ring_push` 是阻塞 push（`device_ring.cuh:78-86`），内部自旋等 sequence 就绪。这保证 CPU 一定能收到 completion——但代价是如果 completion ring 满了，delta block 会卡死，所以 CPU 必须及时 drain `delta_completions`。`final_count = 0` 是 reset 的语义结果。

注意整段在 `delta_status != 0` 时会跳过清理直接写失败 completion（`runtime.cuh:172` 的 `if (delta_status == 0)` 保护了清理循环；但 `runtime.cuh:220` 的 completion 写无条件执行）。这是"fail fast"设计：校验失败就立即回报，不试图做部分清理。

### 8. publish 模式：staging slot 范围校验

`runtime.cuh:231-241` 是 publish 路径的第一道校验：

```cpp
if (delta_status == 0) {
  for (u32 index = threadIdx.x;
       index < delta_descriptor.record_count;
       index += blockDim.x) {
    const u32 slot = params.delta_staging_slots[index];
    if (slot >= delta_descriptor.final_count || slot >= params.delta_capacity) {
      atomicExch(&delta_status, -EINVAL);
    }
  }
  __syncthreads();
}
```

`delta_staging_slots[index]` 是 CPU 在构建 delta 时为第 `index` 条 staging record 分配的目标 slot。这里并行检查所有 slot 都落在 `[0, final_count)` 且 `< delta_capacity`——任何一个越界就让整个命令失败。`atomicExch(&delta_status, -EINVAL)` 而不是 `delta_status = -EINVAL` 是因为多线程并发写，必须用原子；用 `atomicExch` 而不是 `atomicMin` 是因为所有失败都写同一个值，无所谓谁赢。

### 9. publish 模式：dynamic_route 命令的预校验

`runtime.cuh:243-281` 是 dynamic_route 的"前置一致性检查"，在真正写入前完成：

```cpp
if (delta_status == 0) {
  for (u32 index = threadIdx.x;
       index < delta_descriptor.dynamic_route_count;
       index += blockDim.x) {
    const DynamicRouteUpdate update = params.dynamic_route_updates[index];
    const bool live = (update.flags & kDynamicRouteLive) != 0;
    bool duplicate_slot = false;
    for (u32 prior = 0; prior < index; ++prior) {
      duplicate_slot = duplicate_slot ||
        params.dynamic_route_updates[prior].slot == update.slot;
    }
    if (update.slot >= params.dynamic_route_capacity ||
        update.shard >= params.num_shards ||
        update.epoch == 0 ||
        update.slot / kDynamicRouteSlotsPerShard != update.shard ||
        (update.flags & ~kDynamicRouteLive) != 0 ||
        (live &&
         (update.remote_node == 0 ||
          static_cast<u32>(update.remote_node >> 48) != update.shard)) ||
        (!live && update.remote_node != 0) || duplicate_slot) {
      atomicExch(&delta_status, -EINVAL);
      continue;
    }
    const DeviceDynamicRouteSlot& current =
      params.dynamic_route_slots[update.slot];
    const u64 current_command = dynamic_route_atomic_load(current.command_id);
    const u32 current_id = dynamic_route_atomic_load(current.id);
    const u32 current_generation =
      dynamic_route_atomic_load(current.generation);
    if (current_command >= delta_descriptor.command_id ||
        (current_id == update.id &&
         current_generation > update.generation)) {
      atomicExch(&delta_status, -ESTALE);
    }
  }
  __syncthreads();
}
```

校验项解析：

- `slot >= dynamic_route_capacity`：越界。
- `shard >= num_shards`：shard 编号非法。
- `epoch == 0`：epoch=0 是保留值（见 `runtime.cuh:257` 的 `score_dynamic_route_slot` 也判 `epoch == 0` 为非法）。
- `slot / kDynamicRouteSlotsPerShard != update.shard`：slot 必须落在它声明的 shard 内。`kDynamicRouteSlotsPerShard = 8`（`types.hh:85`），所以 slot 0-7 属于 shard 0，8-15 属于 shard 1，依此类推。这个约束让查询侧可以由 slot 反推 shard，简化路由逻辑。
- `flags & ~kDynamicRouteLive`：除 live bit 外不能有其它 bit。
- live 模式下 `remote_node == 0`：live 路由必须指向一个真实节点。
- live 模式下 `remote_node >> 48 != shard`：remote_node 的高 16 位编码 shard（见 `runtime.cuh:262`），必须与声明的 shard 一致。这是 dvstor 的"remote_node 打包格式"约定。
- `!live && remote_node != 0`：非 live（即"下线"）路由必须把 remote_node 清零。
- `duplicate_slot`：同一批 update 里不能对同一个 slot 写两次（O(n²) 检查，但 `dynamic_route_count` 很小，最多 `dynamic_route_capacity`，typical 几十个）。

第二段是"版本校验"：读当前 slot 的 `command_id / id / generation`，如果当前 `command_id >= 本次 command_id` 说明已有更新的命令覆盖了本命令（command_id 单调递增），或者同 id 但 generation 更高（旧 generation 不能覆盖新 generation），则返回 `-ESTALE`。`-ESTALE` 是"过期"语义，CPU 收到后知道这不是真的失败，而是被并发更新抢先了，可以放弃或重试。

`dynamic_route_atomic_load` 是 `cuda::atomic_ref<T>::load(relaxed)` 的封装（`query_traversal.cuh:32-37`）。这里用 relaxed 是因为 seqlock（见后面 `runtime.cuh:600-657`）已经在外层保证一致性，单字段读不需要额外 ordering。

### 10. publish 模式：写 delta_records / vectors

`runtime.cuh:283-301` 把 staging 数据搬到正式 slot：

```cpp
if (delta_status == 0) {
  for (u32 index = threadIdx.x;
       index < delta_descriptor.record_count;
       index += blockDim.x) {
    const u32 slot = params.delta_staging_slots[index];
    params.delta_records[slot] = params.delta_staging_records[index];
    params.delta_next[slot] = UINT32_MAX;
    params.delta_prev[slot] = UINT32_MAX;
  }
  for (u64 index = threadIdx.x;
       index < static_cast<u64>(delta_descriptor.record_count) * params.vector_bytes;
       index += blockDim.x) {
    const u32 record_index = static_cast<u32>(index / params.vector_bytes);
    const u32 byte = static_cast<u32>(index % params.vector_bytes);
    const u32 slot = params.delta_staging_slots[record_index];
    params.delta_vectors[static_cast<u64>(slot) * params.vector_bytes + byte] =
      params.delta_staging_vectors[index];
  }
  __syncthreads();
  ...
}
```

注意两层循环的结构：

- 第一层按 record 粒度并行，每个 thread 处理若干 record，拷贝 `DeviceDeltaRecord` 结构体（48 字节）并把链表指针初始化成 `UINT32_MAX`（暂时不在任何 bucket 链表里）。
- 第二层按字节粒度并行，把 `record_count * vector_bytes` 个字节从 staging 拷到目标。每个 thread 处理一个字节，但通过 `record_index = index / vector_bytes` 和 `byte = index % vector_bytes` 反推所属 record 和字节偏移。这种"扁平化"循环比"外层 record / 内层 byte"的双层循环对 GPU 更友好——单层 grid-stride loop 的 occupancy 更高。

`__syncthreads()` 在两层循环后保证所有 thread 都写完，才进入下面的 PQ 编码阶段（编码要读刚写的 vectors）。

### 11. publish 模式：OPQ 变换与 PQ 编码

`runtime.cuh:303-328` 是 OPQ 变换：

```cpp
for (u64 index = threadIdx.x;
     index < static_cast<u64>(delta_descriptor.record_count) * params.dim;
     index += blockDim.x) {
  const u32 record_index = static_cast<u32>(index / params.dim);
  const u32 row = static_cast<u32>(index % params.dim);
  const u32 slot = params.delta_staging_slots[record_index];
  const DeviceDeltaRecord record = params.delta_records[slot];
  f32 transformed = 0.0f;
  if ((record.flags & kDeltaDeleted) == 0) {
    const u8* vector = params.delta_vectors +
      static_cast<size_t>(slot) * params.vector_bytes;
    if (params.opq_matrix == nullptr) {
      transformed = storage_component(params, vector, row);
    } else {
      const f32* matrix_row = params.opq_matrix +
        static_cast<size_t>(row) * params.dim;
      for (u32 column = 0; column < params.dim; ++column) {
        transformed += matrix_row[column] *
          storage_component(params, vector, column);
      }
    }
  }
  params.delta_encode_scratch[index] = transformed;
}
```

`storage_component(params, vector, row)`（见 `rdma_cache.cuh:7`）按 `params.vector_dtype` 把向量的第 `row` 维解析成 f32。`opq_matrix` 是可选的旋转矩阵：若存在，对向量做 `M * v` 变换再编码；若不存在，直接用原始分量。`kDeltaDeleted` 的 record 跳过（transformed=0），后面 PQ 编码会用 0 填充——这其实是"占位"，删除记录的 PQ code 永远不会被查询读到（查询侧通过 flags 过滤）。

`delta_encode_scratch` 是临时缓冲，存 `record_count * dim` 个 f32。后面 PQ 编码从这里读。

`runtime.cuh:329-370` 是 PQ 编码：

```cpp
for (u64 index = threadIdx.x;
     index < static_cast<u64>(delta_descriptor.record_count) * params.pq_code_bytes;
     index += blockDim.x) {
  const u32 record_index = static_cast<u32>(index / params.pq_code_bytes);
  const u32 subquantizer = static_cast<u32>(index % params.pq_code_bytes);
  const u32 slot = params.delta_staging_slots[record_index];
  u8 best_code = 0;
  if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
    const f32* transformed = params.delta_encode_scratch +
      static_cast<size_t>(record_index) * params.dim +
      static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
    const f32* centroids = params.pq_centroids +
      static_cast<size_t>(subquantizer) * 256 * params.pq_subvector_dim;
    f32 best_distance = FLT_MAX;
    for (u32 centroid = 0; centroid < 256; ++centroid) {
      f32 distance = 0.0f;
      for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
        const f32 difference = transformed[dimension] -
          centroids[static_cast<size_t>(centroid) * params.pq_subvector_dim + dimension];
        distance += difference * difference;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best_code = static_cast<u8>(centroid);
      }
    }
  }
  params.delta_pq_codes[
    static_cast<size_t>(slot) * params.pq_code_bytes + subquantizer] = best_code;
  const u32 resident_slot = params.delta_records[slot].resident_pq_slot;
  if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
    if (resident_slot >= params.resident_pq_capacity) {
      atomicExch(&delta_status, -ENOSPC);
    } else {
      params.resident_pq_codes[
        static_cast<size_t>(resident_slot) * params.pq_code_bytes +
        subquantizer] = best_code;
    }
  }
}
__threadfence();
__syncthreads();
```

每个 thread 负责一个 (record, subquantizer) 对，在 256 个 centroid 里找 L2 距离最近的，作为该 subquantizer 的 PQ code。这是经典 PQ 编码的 brute-force 实现——`256 * pq_subvector_dim` 次乘加，对 dim=128/pq_subvector_dim=16 的工作量约 4K FLOPs/thread，完全可以接受。

`resident_pq_codes` 是 resident PQ 表（见第 19 课），存"当前 live"节点的 PQ code 用于快速近似评分。`resident_pq_slot` 是 CPU 在构建 delta 时为该 record 分配的 resident 槽位。如果 `resident_slot >= resident_pq_capacity`，说明 CPU 分配错了（或容量不够），返回 `-ENOSPC`。注意这里用 `atomicExch` 而非 `delta_status = -ENOSPC`，原因同上。

最后的 `__threadfence()` + `__syncthreads()` 保证 PQ code 全部写完且对全 GPU 可见——下一阶段是图缓存失效，可能被查询侧并发读到。

### 12. publish 模式：图缓存失效

`runtime.cuh:372-415` 处理 `invalidation_count` 个图缓存失效：

```cpp
for (u32 index = threadIdx.x;
     index < delta_descriptor.invalidation_count;
     index += blockDim.x) {
  const u64 key = params.graph_invalidation_keys[index];
  const u32 route_slot = anchor_graph_slot(params, key);
  if (route_slot != UINT32_MAX &&
      params.anchor_graph_states != nullptr) {
    atomicCAS(params.anchor_graph_states + route_slot,
              kGraphCacheReady, kGraphCacheStale);
  }
  if (params.graph_cache_sets == 0 ||
      params.graph_cache_states == nullptr ||
      params.graph_cache_keys == nullptr) {
    continue;
  }
  const u32 set = hash64(key) % params.graph_cache_sets;
  for (u32 way = 0; way < params.graph_cache_ways; ++way) {
    const u32 slot = set * params.graph_cache_ways + way;
    for (;;) {
      const u32 state = *reinterpret_cast<volatile u32*>(
        params.graph_cache_states + slot);
      if (load_cg(params.graph_cache_keys + slot) != key ||
          state == kGraphCacheEmpty || state == kGraphCacheStale ||
          state == kGraphCacheFillInvalidated) {
        break;
      }
      if (state == kGraphCacheReady) {
        if (atomicCAS(params.graph_cache_states + slot, kGraphCacheReady,
                      kGraphCacheStale) == kGraphCacheReady) {
          break;
        }
        continue;
      }
      if (state == kGraphCacheFilling) {
        if (atomicCAS(params.graph_cache_states + slot, kGraphCacheFilling,
                      kGraphCacheFillInvalidated) == kGraphCacheFilling) {
          break;
        }
        continue;
      }
      break;
    }
  }
}
```

图缓存是组相联（`graph_cache_sets` 组，每组 `graph_cache_ways` 路）。失效逻辑：

1. **route 表失效**（`runtime.cuh:375-381`）：`anchor_graph_slot` 是二分查找（`candidate_scoring.cuh:92-107`），在 `anchor_graph_keys` 排序数组里找 key。找到则把对应 state 从 `kGraphCacheReady` CAS 到 `kGraphCacheStale`。
2. **组相联缓存失效**（`runtime.cuh:386-414`）：对 key 所在组的每一路，根据当前 state 做不同处理：
   - key 不匹配 / state 是 Empty/Stale/FillInvalidated：跳过（已经失效或不是这条 key）。
   - Ready：CAS 到 Stale。CAS 失败说明并发改了，重读。
   - Filling：CAS 到 FillInvalidated。这告诉正在 fill 的 thread："你 fill 完后别 publish，数据已经过期了"。
   - 其它（implicit break）：保守退出。

内层 `for(;;)` + CAS 失败 `continue` 是无锁状态机的标准范式——并发状态下任何观测都可能瞬时失效，必须重读直到 CAS 成功或确认不需要操作。`volatile u32*` 读保证看到最新值，`load_cg` 读 key 是 cache-bypass 避免污染 L2。

### 13. publish 模式：supersede / override / durable / resident_pq_erase

`runtime.cuh:417-529` 是一组"小批量、单线程"操作，全部在 `threadIdx.x == 0` 上串行执行。原因是这些操作之间有顺序依赖（例如 supersede 要 unlink，durable 也要 unlink），并行化收益小、复杂度高。

**a. supersede**（`runtime.cuh:417-428`）：

```cpp
if (threadIdx.x == 0) {
  for (u32 index = 0; index < delta_descriptor.superseded_count; ++index) {
    const DeltaSupersedeUpdate update = params.delta_supersede_updates[index];
    if (update.slot >= delta_descriptor.final_count) {
      delta_status = -EINVAL;
      continue;
    }
    DeviceDeltaRecord& record = params.delta_records[update.slot];
    record.superseded_epoch = update.epoch;
    unlink_mutable_delta(params, update.slot);
  }
}
```

`unlink_mutable_delta`（`candidate_scoring.cuh:25-45`）把 slot 从 bucket 链表里摘下来——用 `atomicCAS` 修改前驱的 next 和后继的 prev，把 slot 的 prev/next 都设为 `UINT32_MAX`。supersede 的语义是"这个 delta record 已被新版本覆盖，不再参与候选生成"，所以要从 bucket 链表移除。

**b. override 处理**（`runtime.cuh:430-479`）分两种模式：

promote 模式（`runtime.cuh:430-441`）：

```cpp
if ((delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0) {
  for (u32 index = threadIdx.x;
       index < delta_descriptor.override_count;
       index += blockDim.x) {
    const u32 ordinal = params.delta_override_updates[index].ordinal;
    if (ordinal >= params.num_nodes) {
      atomicExch(&delta_status, -EINVAL);
      continue;
    }
    atomicOr(params.permanent_override_bits + ordinal / 32,
             1u << (ordinal % 32));
  }
}
```

promote 是把 delta override "提升"成 permanent override——用 bitset 标记某个 ordinal 永久被覆盖（不再走 base 索引）。`atomicOr` 在 bitset 上设位，每个 thread 处理一个 ordinal，可并行。

非 promote 模式（`runtime.cuh:442-479`）是普通 delta override，写到 `base_override` 哈希表：

```cpp
} else if (threadIdx.x == 0) {
  const u32 mask = params.base_override_capacity - 1;
  for (u32 index = 0; index < delta_descriptor.override_count; ++index) {
    const DeltaOverrideUpdate update = params.delta_override_updates[index];
    u32 position = hash32(update.ordinal) & mask;
    u32 first_tombstone = UINT32_MAX;
    bool inserted = false;
    for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
      const u32 key = params.base_override_keys[position];
      if (key == update.ordinal) {
        params.base_override_epochs[position] = min(
          params.base_override_epochs[position], update.epoch);
        inserted = true;
        break;
      }
      if (key == kBaseOverrideTombstone && first_tombstone == UINT32_MAX) {
        first_tombstone = position;
      }
      if (key == kBaseOverrideEmpty) {
        const u32 destination = first_tombstone == UINT32_MAX
          ? position : first_tombstone;
        params.base_override_epochs[destination] = update.epoch;
        __threadfence();
        params.base_override_keys[destination] = update.ordinal;
        inserted = true;
        break;
      }
      position = (position + 1) & mask;
    }
    if (!inserted && first_tombstone != UINT32_MAX) {
      params.base_override_epochs[first_tombstone] = update.epoch;
      __threadfence();
      params.base_override_keys[first_tombstone] = update.ordinal;
      inserted = true;
    }
    if (!inserted) delta_status = -ENOSPC;
  }
}
```

这是标准的开放寻址哈希表插入，带 tombstone 复用：
- 线性探测找 key；找到则 `min(epoch)` 更新（旧 epoch 优先，因为旧 epoch 表示更早的版本，查询快照 <= epoch 才能看到）。
- 遇 tombstone 记下第一个，继续探测（key 可能在后面）。
- 遇 Empty：插到这里（或第一个 tombstone，优先复用 tombstone 节省空间）。
- 探测完仍没找到也没 Empty：表满，`-ENOSPC`。

**epoch 先写、key 后写**（`runtime.cuh:463-465`）的顺序很关键：查询侧通过读 key 判断"这条 slot 是否有效"，若先写 key 再写 epoch，查询侧可能读到新 key 但旧 epoch（=0），误判为"未初始化"。先写 epoch 再 fence 再写 key，保证查询侧读到 key 时 epoch 一定就位。`__threadfence()` 保证写顺序对全 GPU 可见。

**c. durable**（`runtime.cuh:481-524`）：

```cpp
if (threadIdx.x == 0) {
  for (u32 index = 0; index < delta_descriptor.durable_count; ++index) {
    const DeltaDurableUpdate update = params.delta_durable_updates[index];
    if (update.slot >= delta_descriptor.final_count) {
      delta_status = -EINVAL;
      continue;
    }
    DeviceDeltaRecord& record = params.delta_records[update.slot];
    if (record.epoch == update.epoch) {
      if (record.superseded_epoch == 0) {
        record.superseded_epoch = update.epoch;
      }
      unlink_mutable_delta(params, update.slot);
      // 清 delta_remote 表
      // 设 permanent_override_bits
      // 清 base_override 表
    }
  }
  for (u32 index = 0;
       index < delta_descriptor.resident_pq_erase_count; ++index) {
    erase_resident_pq(params, params.resident_pq_erase_updates[index]);
  }
}
```

durable 处理"delta record 已经被持久化到 base 索引"：把 record 从 delta 链表 unlink、清掉 remote 表项、把 base_ordinal 标记成 permanent override（这样查询侧遇到这个 ordinal 时走 base 而不是 delta）、清掉 base_override 表里临时 override。`record.epoch == update.epoch` 是幂等性检查——只有 record 还在它声明时的 epoch 才执行，避免并发 durable 处理同一 slot。

**d. resident_pq_erase**（`runtime.cuh:525-528`）：调用 `erase_resident_pq`（`candidate_scoring.cuh:325-340`），从 resident PQ 表删一个 entry（CAS key 到 tombstone、清 slot/position）。这用于"某节点下线，从 resident PQ 移除"。

### 14. publish 模式：delta_remote 表与 bucket 链表发布

`runtime.cuh:533-598` 是 delta record "上线"的核心——把它们登记到 remote 表和 bucket 链表：

```cpp
if (delta_status == 0) {
  if (threadIdx.x == 0) {
    const u32 mask = params.delta_remote_capacity - 1;
    for (u32 index = 0; index < delta_descriptor.record_count; ++index) {
      const u32 slot = params.delta_staging_slots[index];
      const DeviceDeltaRecord record = params.delta_records[slot];
      if ((record.flags & kDeltaDeleted) == 0 &&
          !insert_resident_pq(params, record.remote_node, record.resident_pq_slot)) {
        delta_status = -ENOSPC;
        break;
      }
      params.delta_remote_positions[slot] = UINT32_MAX;
      if (record.remote_node != 0 && params.delta_remote_capacity != 0) {
        u32 position = hash64(record.remote_node) & mask;
        u32 first_tombstone = UINT32_MAX;
        bool inserted = false;
        for (u32 probe = 0; probe < params.delta_remote_capacity; ++probe) {
          const u64 key = params.delta_remote_keys[position];
          if (key == record.remote_node) {
            params.delta_remote_slots[position] = slot;
            params.delta_remote_positions[slot] = position;
            inserted = true;
            break;
          }
          if (key == kDeltaRemoteTombstone && first_tombstone == UINT32_MAX) {
            first_tombstone = position;
          }
          if (key == kDeltaRemoteEmpty) {
            const u32 destination = first_tombstone == UINT32_MAX
              ? position : first_tombstone;
            params.delta_remote_slots[destination] = slot;
            __threadfence();
            params.delta_remote_keys[destination] = record.remote_node;
            params.delta_remote_positions[slot] = destination;
            inserted = true;
            break;
          }
          position = (position + 1) & mask;
        }
        if (!inserted && first_tombstone != UINT32_MAX) {
          params.delta_remote_slots[first_tombstone] = slot;
          __threadfence();
          params.delta_remote_keys[first_tombstone] = record.remote_node;
          params.delta_remote_positions[slot] = first_tombstone;
          inserted = true;
        }
        if (!inserted) {
          delta_status = -ENOSPC;
          break;
        }
      }
      if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
          record.superseded_epoch == 0 &&
          params.delta_bucket_heads != nullptr) {
        const u32 old_head = params.delta_bucket_heads[record.anchor_bucket];
        params.delta_prev[slot] = UINT32_MAX;
        params.delta_next[slot] = old_head;
        if (old_head < params.delta_capacity) {
          params.delta_prev[old_head] = slot;
        }
        params.delta_bucket_heads[record.anchor_bucket] = slot;
      }
    }
  }
}
__syncthreads();
```

这段在 thread 0 上串行，因为：

1. `insert_resident_pq`（`candidate_scoring.cuh:282-323`）写 resident_pq 表，可能被并发 delta record 争抢同一 tombstone。
2. delta_remote 表的开放寻址插入需要"原子地占有 slot"，多线程并发插入时 tombstone 复用会冲突。
3. bucket 链表的 head 插入需要 `old_head = bucket_heads[bucket]; bucket_heads[bucket] = slot; prev[old_head] = slot`，这三步对同一 bucket 非原子，必须串行。

串行的代价是 record_count 大时慢，但 record_count typical 几十个，可接受。

注意 `delta_remote_positions[slot]` 在 `runtime.cuh:545` 被先设为 `UINT32_MAX`，然后在插入成功后被设为实际 position。这是"先失活再激活"模式——如果插入失败，slot 处于"无 remote 关联"状态，查询侧不会找到它；插入成功后才激活。

bucket 链表插入（`runtime.cuh:585-595`）是经典的"head insert"：把 slot 接到链表头，更新旧 head 的 prev 指针。`if (old_head < params.delta_capacity)` 是防御——链表空时 head 是 `UINT32_MAX`，不能当数组下标。`kDeltaDeleted | kDeltaDurable` 的 record 不入链表（已删 / 已持久化的不该被扫到），`superseded_epoch == 0` 的才入（被 supersede 的不入）。

### 15. publish 模式：dynamic_route 的 seqlock 发布

`runtime.cuh:600-657` 是 dynamic_route 的"seqlock 发布"，最精妙的一段：

```cpp
if (delta_status == 0) {
  // Canonical storage-route codes become visible before a route slot can
  // point at them. Mark every changing slot odd before touching either
  // its code or metadata; query scoring rechecks the same sequence after
  // consuming both.
  for (u32 index = threadIdx.x;
       index < delta_descriptor.dynamic_route_count;
       index += blockDim.x) {
    const DynamicRouteUpdate update = params.dynamic_route_updates[index];
    DeviceDynamicRouteSlot& destination = params.dynamic_route_slots[update.slot];
    cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
      destination.sequence);
    sequence.fetch_add(1, cuda::memory_order_acq_rel);
  }
  __syncthreads();
  for (u64 byte = threadIdx.x;
       byte < static_cast<u64>(delta_descriptor.dynamic_route_count) *
                params.pq_code_bytes;
       byte += blockDim.x) {
    const u32 update_index = static_cast<u32>(byte / params.pq_code_bytes);
    const u32 code_byte = static_cast<u32>(byte % params.pq_code_bytes);
    const DynamicRouteUpdate update = params.dynamic_route_updates[update_index];
    if ((update.flags & kDynamicRouteLive) != 0) {
      params.dynamic_route_pq_codes[
        static_cast<size_t>(update.slot) * params.pq_code_bytes +
        code_byte] = params.dynamic_route_code_updates[byte];
    }
  }
  __threadfence();
  __syncthreads();
  for (u32 index = threadIdx.x;
       index < delta_descriptor.dynamic_route_count;
       index += blockDim.x) {
    const DynamicRouteUpdate update = params.dynamic_route_updates[index];
    DeviceDynamicRouteSlot& destination = params.dynamic_route_slots[update.slot];
    cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
      destination.sequence);
    dynamic_route_atomic_store(destination.command_id, delta_descriptor.command_id);
    dynamic_route_atomic_store(destination.epoch, update.epoch);
    dynamic_route_atomic_store(destination.remote_node, update.remote_node);
    dynamic_route_atomic_store(destination.id, update.id);
    dynamic_route_atomic_store(destination.generation, update.generation);
    dynamic_route_atomic_store(destination.shard, update.shard);
    dynamic_route_atomic_store(destination.flags, update.flags);
    __threadfence();
    sequence.fetch_add(1, cuda::memory_order_release);
  }
}
```

`DeviceDynamicRouteSlot` 的 `sequence` 字段是 seqlock（见 `types.hh:99-112` 的注释）：奇数表示"正在更新"，偶数表示"稳定快照"。查询侧（`query_traversal.cuh:55-97` 的 `score_dynamic_route_slot`）读 sequence 两次（读字段前后），如果两次都是同一个偶数，说明读到的是一致快照。

发布分三步：

**a. 第一步：所有 slot 的 sequence 加 1（变奇）**（`runtime.cuh:605-615`）。`acq_rel` ordering 保证这一步与之前的 delta_record/vector 写入建立 happens-before。`__syncthreads()` 保证所有 slot 都变奇后才进入下一步。

**b. 第二步：写 PQ code**（`runtime.cuh:617-632`）。每个 thread 处理一个 (update, code_byte) 对，把 canonical PQ code 写到 `dynamic_route_pq_codes`。只 live 路由才写（非 live 的 PQ code 保留旧值，反正查询侧也会因 `!live` 跳过）。`__threadfence()` + `__syncthreads()` 保证 PQ code 全部写完且对全 GPU 可见。

**c. 第三步：写 metadata 并 sequence 加 1（变偶）**（`runtime.cuh:635-656`）。逐字段 relaxed store 写 command_id/epoch/remote_node/id/generation/shard/flags，然后 `__threadfence()` 保证字段写对全 GPU 可见，最后 `sequence.fetch_add(1, release)` 变偶。

注意第二步和第三步之间没有 `__syncthreads()`——它们都在第三步循环开始前由 `__threadfence()` + `__syncthreads()`（`runtime.cuh:633-634`）隔开。第三步内部不同 thread 处理不同 slot，互不干扰；同一 slot 内的 7 个字段都由同一个 thread 写（因为 `index` 循环里每个 thread 独占一个 update），所以字段间顺序由 program order 保证。

这套 seqlock 保证：查询侧要么读到完全旧的 (code, metadata)，要么读到完全新的，绝不会读到新 code + 旧 metadata 或反之。这是"无锁 publish"的核心——查询侧从不阻塞 writer，writer 从不阻塞 reader。

### 16. publish 模式：发布 delta_count 与写 completion

`runtime.cuh:659-676` 是 publish 的最后一步：

```cpp
if (threadIdx.x == 0 && delta_status == 0) {
  __threadfence();
  atomicExch(params.delta_count, delta_descriptor.final_count);
  __threadfence_system();
}
__syncthreads();
if (threadIdx.x == 0) {
  device_ring_push(params.delta_completions, DeltaPublishCompletion{
    .command_id = delta_descriptor.command_id,
    .status = delta_status,
    .final_count = delta_status == 0 ? delta_descriptor.final_count : 0u,
  });
}
__syncthreads();
if (threadIdx.x == 0) idle_cycles = 256u;
__syncthreads();
continue;
```

`delta_count` 是查询侧判断"delta 区有多少有效 record"的唯一信号。`__threadfence()` 保证前面的所有写（records / vectors / pq_codes / remote 表 / bucket 链表 / dynamic_route）对全 GPU 可见，然后 `atomicExch` 发布 `final_count`，`__threadfence_system()` 让 CPU 也能看到。这一对 fence 的角色与 reset 路径完全一致。

completion 的 `final_count` 在失败时写 0（`runtime.cuh:669`）——告诉 CPU"这次没发布任何东西"，CPU 可以据此决定是否重试。成功时写 `final_count`，CPU 用它跟踪当前 delta 区大小。

最后 `idle_cycles = 256u`（`runtime.cuh:673`）把退避复位——刚处理完一个 delta，下一个可能很快就来，不要睡太久。

### 17. query 角色：消费 device_submissions

如果 delta ring 是空的（或者 `enable_delta = false`），主循环走到 query 路径（`runtime.cuh:678-699`）：

```cpp
if (threadIdx.x == 0) {
  const DeviceRingView<QueryDescriptor> query_queue =
    params.device_submissions.entries != nullptr
      ? params.device_submissions : params.submissions;
  have_submission = enable_queries && query_queue.entries != nullptr &&
    device_ring_try_pop(query_queue, descriptor) ? 1u : 0u;
}
__syncthreads();
if (have_submission == 0) {
  if (threadIdx.x == 0) {
    device_ring_relax(idle_cycles);
    idle_cycles = min(idle_cycles * 2, 16384u);
  }
  __syncthreads();
  continue;
}
if (threadIdx.x == 0) {
  idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
}
__syncthreads();
process_query(params, descriptor);
__syncthreads();
```

关键的 ternary（`runtime.cuh:679-681`）：
- unified 模式下 `device_submissions.entries != nullptr`，从 device_submissions 取（dispatcher 已经把 CPU 提交的 descriptor 转发到这里）。
- 非 unified 模式下 `device_submissions.entries == nullptr`，直接从 `submissions` 取（CPU 提交的 ring）。

这实现了"双 ring 透明回退"——同一份 query 消费逻辑既能工作于 unified 模式（经 dispatcher 转发），也能工作于非 unified 模式（直读 CPU ring）。

`enable_queries &&` 短路保护：dispatcher block 与 delta block 都会执行到这一行（它们没有 `continue` 走出主循环？错——dispatcher 在 `runtime.cuh:81` `continue` 了，所以只有 delta block 会走到这里）。等等，让我重新梳理：

实际上 dispatcher 在 `runtime.cuh:58-82` 完整地 `continue`，不会走到 query 路径。delta block 在 `runtime.cuh:89-676` 处理完 delta 后 `continue`。所以走到 `runtime.cuh:678` 的只有"没有 delta 可取"的 block：
- 非 unified 模式：所有 query block（`enable_delta = true` 但 `device_ring_try_pop` 失败）。
- unified 模式：`enable_queries = true` 的 block（即 `role_block < query_block_count`）。delta block 不会走到这里（它要么在处理 delta，要么 `enable_queries = false` 且 `enable_delta = true`，但它的 `enable_queries = false` 会让 `have_submission = 0`，进入 idle 退避，永远不调 `process_query`）。

所以 `enable_queries &&` 这个短路对 delta block 是必须的——否则 delta block 在没 delta 时会去抢 query ring，造成与 query block 的争抢。这是一个很 subtle 的保护。

`process_query` 是第 20 课的主题，这里只提一句：它接收 descriptor、跑完整个图遍历 + 候选评分 + 精确重排，把结果写到 `result_ids`/`result_distances`，并通过 completion ring 回报。

### 18. `direct_read_owner_loop`：owner warp 的 GPUNetIO 主循环

`runtime.cuh:710-938` 是 unified 模式下 owner block 的实现，也是第 19 课 RDMA cache 的"服务端"——查询 CTA 通过 `direct_fetch_batch` 把读请求 push 到 `direct_batch_queues`，owner warp 从这个 ring 取请求、组装 DOCA verbs WQE、提交、poll CQ、写 completion。整个流程见第 19 课与第 22 课，本课只讲 owner loop 的运行时结构。

**a. warp 分工**（`runtime.cuh:714-741`）：

```cpp
constexpr u32 warp_width = 32;
constexpr u32 max_warps_per_block = 8;
constexpr u32 max_submit_batches = 8;
const u32 lane = threadIdx.x % warp_width;
const u32 warps_per_block = blockDim.x / warp_width;
const u32 warp_in_block = threadIdx.x / warp_width;
const u32 warp = owner_block * warps_per_block + warp_in_block;
if (warps_per_block == 0 || warps_per_block > max_warps_per_block ||
    warp >= queue_count) return;
```

每个 block 含最多 8 个 warp（256 thread），每个 warp 独占一个 `direct_batch_queues[warp]`。`warp` 是全局 warp 编号（`owner_block * warps_per_block + warp_in_block`），用作 queue 索引。`warp >= queue_count` 早退：多余的 warp 不干活。

`max_warps_per_block = 8` 与 `runtime.cuh:743-750` 的 `__shared__` 数组维度对应——shared memory 数组按 `[max_warps_per_block][max_submit_batches]` 分配，超过 8 warp 会越界。

**b. 参数健全性检查**（`runtime.cuh:727-741`）用位掩码 `invalid` 收集所有错误：

```cpp
u32 invalid = 0;
invalid |= params.direct_batch_queues == nullptr ? 1u : 0u;
invalid |= params.direct_qps == nullptr ? 2u : 0u;
invalid |= params.direct_regions == nullptr ? 4u : 0u;
invalid |= params.direct_disabled == nullptr ? 8u : 0u;
invalid |= params.direct_region_count == 0 ? 16u : 0u;
invalid |= params.direct_qps_per_node == 0 ? 32u : 0u;
invalid |= warp >= params.direct_batch_queue_count ? 64u : 0u;
if (invalid != 0) {
  if (lane == 0 && params.direct_owner_phases != nullptr) {
    params.direct_owner_phases[warp] = 0x100u | invalid;
    __threadfence_system();
  }
  return;
}
```

每个错误对应一个 bit，组合成 `0x100 | invalid` 写入 `direct_owner_phases[warp]`。CPU 读这个数组就能精确知道哪个 warp 因为什么原因退出（例如 `0x105` = QP 与 regions 都为 null）。`0x100u` 前缀表示"参数错误退出"，与正常 phase 编号（1/2/3/4/5/6）区分。

`direct_owner_phases` 是 CPU 与 GPU 之间的调试通道——CPU 启动 owner 后会轮询这个数组，看每个 warp 走到了哪个阶段（phase 1 = 初始化完成、phase 2 = 取到第一个 batch、phase 3 = 提交 WQE、phase 4 = 等 CQ、phase 5 = 出错、phase 6 = 成功完成）。这是远程调试 GPU kernel 的常用技巧。

**c. shared memory 工作区**（`runtime.cuh:743-750`）：

```cpp
__shared__ DirectBatchDescriptor shared_batches
  [max_warps_per_block][max_submit_batches];
__shared__ u32 shared_matching_counts
  [max_warps_per_block][max_submit_batches];
__shared__ u32 shared_wqe_offsets
  [max_warps_per_block][max_submit_batches];
__shared__ u32 shared_batch_counts[max_warps_per_block];
__shared__ u32 shared_total_wqes[max_warps_per_block];
```

每个 warp 在 shared memory 里有自己的 8-slot "batch 工作区"，存放一次 submit 最多 8 个 batch 的 descriptor / matching count / WQE offset。warp 内所有 lane 通过 `__syncwarp()` 共享这些数据——lane 0 生产，所有 lane 消费。

**d. 取 batch 与流量控制**（`runtime.cuh:778-851`）：

主循环顶部先检查 stop（用 `__shfl_sync` 把 lane 0 读的 stop 广播给所有 lane），然后 lane 0 进入取 batch 循环：

```cpp
if (lane == 0) {
  u32 batch_count = 0;
  u32 total_wqes = 0;
  while (batch_count < max_submit_batches) {
    DirectBatchDescriptor descriptor{};
    u32 matching = 0;
    if (have_deferred) {
      descriptor = deferred;
      matching = deferred_matching;
      have_deferred = false;
    } else if (!device_ring_try_pop(queue, descriptor)) {
      break;
    } else {
      ... // 校验 descriptor，计算 matching
    }
    ...
    const u32 needed = matching + (need_dump ? 1u : 0u);
    if (needed > qp->sq_wqe_num) {
      complete_direct_batch(descriptor, -E2BIG);
      continue;
    }
    if (batch_count != 0 && total_wqes + needed > qp->sq_wqe_num) {
      deferred = descriptor;
      deferred_matching = matching;
      have_deferred = true;
      break;
    }
    shared_batches[warp_in_block][batch_count] = descriptor;
    shared_matching_counts[warp_in_block][batch_count] = matching;
    shared_wqe_offsets[warp_in_block][batch_count] = total_wqes;
    ++batch_count;
    total_wqes += needed;
  }
  shared_batch_counts[warp_in_block] = batch_count;
  shared_total_wqes[warp_in_block] = total_wqes;
}
__syncwarp();
```

这是"批量合并提交"的核心：lane 0 从 ring 连续 pop batch，累计 WQE 数，直到：
- 凑满 8 个 batch（`max_submit_batches`）；
- 或下一个 batch 会超过 QP 的 WQE 容量（`total_wqes + needed > qp->sq_wqe_num`）——此时把这条 batch "deferred" 到下一轮，本批先提交。
- 或 ring 空（`try_pop` 失败）。

`deferred` 机制保证：即使 batch 太大塞不下，也不会丢——留在 shared memory 里下一轮优先处理。`-E2BIG` 是单个 batch 就超过 QP 容量的硬错误，直接 fail completion。

**e. WQE 准备与提交**（`runtime.cuh:854-910`）：

```cpp
const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
const doca_gpu_dev_verbs_ticket_t first_completion =
  doca_gpu_dev_verbs_load_relaxed<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      &completion_queue->cqe_ci);
for (u32 batch = 0; batch < batch_count; ++batch) {
  const DirectBatchDescriptor descriptor = shared_batches[warp_in_block][batch];
  const u32 matching = shared_matching_counts[warp_in_block][batch];
  const u32 batch_offset = shared_wqe_offsets[warp_in_block][batch];
  u32 matched_before = 0;
  for (u32 base = 0; base < descriptor.request_count; base += warp_width) {
    const u32 index = base + lane;
    const bool matching_request = index < descriptor.request_count &&
      descriptor.request_shards[index] == memory_node;
    const u32 matching_mask = __ballot_sync(0xffffffffu, matching_request);
    if (matching_request) {
      const u32 lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      const u32 rank = __popc(matching_mask & lower_lanes);
      const u32 matched = matched_before + rank;
      const doca_gpu_dev_verbs_ticket_t ticket = first_wqe + batch_offset + matched;
      auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
      const bool last_read = matched + 1 == matching;
      const auto flags = !need_dump && last_read
        ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
        : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
      doca_gpu_dev_verbs_wqe_prepare_read(
        qp, wqe, ticket, flags,
        region.address + descriptor.remote_offsets[index], region.rkey,
        descriptor.local_iova_offsets[index], params.direct_local_mkey,
        descriptor.bytes);
    }
    matched_before += __popc(matching_mask);
  }
  if (need_dump && lane == 0) {
    ... // 准备 dump WQE
  }
}
__syncwarp();
if (lane == 0) {
  ... // submit, poll CQ, complete batches
}
```

这是 warp 协作的精髓：

- **`__ballot_sync` + `__popc`** 算每个 matching lane 在 batch 内的"rank"——即这条 read 请求对应 batch 内第几个 WQE。例如 batch 有 5 个 matching request，分布在 lane 0/1/4/5/7，那么 lane 0 的 rank=0、lane 1 的 rank=1、lane 4 的 rank=2，依此类推。这样每个 lane 知道自己该往 `first_wqe + batch_offset + rank` 这个 WQE slot 写。
- **CQ_UPDATE flag 只在 batch 内最后一个 read** 上设（`last_read = matched + 1 == matching`）——这样一个 batch 只产生一个 CQE，poll CQ 时只需等一个 completion。其它 read 用 `CQ_ERROR_UPDATE` flag，只有出错时才产生 CQE。这是性能优化：每 batch 一个 CQE 而不是每 read 一个。
- **`need_dump`** 是调试模式，每个 batch 末尾额外塞一个 "dump WQE"（把 GPU 内存内容 dump 到网卡 trace buffer）。

lane 0 在所有 WQE 准备好后调 `doca_gpu_dev_verbs_submit` 一次性提交整批 WQE，然后串行 poll 每个 batch 的 CQ（`runtime.cuh:912-920`），把 status 写回每个 batch 的 `completion_status`。poll 失败时把 `direct_disabled` 设为 1，让后续所有 batch 立即 fail（`runtime.cuh:926-929`）。

`complete_direct_batch`（`runtime.cuh:703-708`）的实现：

```cpp
__device__ void complete_direct_batch(const DirectBatchDescriptor& descriptor,
                                      i32 status) {
  if (descriptor.completion_status == nullptr) return;
  __threadfence_system();
  atomicExch(descriptor.completion_status, status);
}
```

`__threadfence_system()` 保证 owner warp 之前的所有 RDMA write（WQE 的 read 结果写到 local memory）对 CPU 可见——完成状态写早了的话，CPU 可能读到未完成的 RDMA 数据。`atomicExch` 让等待的查询 CTA 看到 status 变化（查询 CTA 在 `direct_fetch_batch` 里自旋等 `*completion_status`）。

### 19. 独立的 owner / probe kernel

`runtime.cuh:940-966` 是非 unified 模式下的独立 owner kernel：

```cpp
__global__ void direct_read_owner_kernel(PersistentKernelParams params,
                                         u32 queue_count) {
  direct_read_owner_loop(params, queue_count, blockIdx.x);
}
```

仅仅是 `direct_read_owner_loop` 的薄包装，让 CPU 可以单独 launch。

`runtime.cuh:945-966`、`runtime.cuh:968-996`、`runtime.cuh:998-1029` 是三个 probe kernel，用于 GPUNetIO 链路健康检查（见第 22 课）：
- `gpunetio_locked_read_probe_kernel`：单 lane 独占 QP，做 N 次 8 字节 read。
- `gpunetio_batched_read_probe_kernel`：单 CTA 做一次 batched read（最多 `kPersistentMaxExact = 256` 个请求）。
- `gpunetio_owner_read_probe_kernel`：走完整的 owner 路径（`direct_fetch_batch` with `completion_status`），验证 owner 机制本身。

`runtime.cuh:1031-1046` 的 `gather_anchor_codes_kernel` 是个工具 kernel，把 anchor 对应的 base PQ code 收集到连续数组，用于离线 / 启动时构建 anchor 码本。

### 20. launch 包装

`persistent_kernel.cu` 是所有 kernel 的 launch 包装：

```cpp
void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}

void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads) {
  const u32 warps_per_block = max(1u, threads / 32);
  const u32 blocks = (queue_count + warps_per_block - 1) / warps_per_block;
  direct_read_owner_kernel<<<blocks, threads, 0, stream>>>(params, queue_count);
}
```

注意 `launch_direct_read_owners` 的 blocks 计算：`warps_per_block = threads / 32`，`blocks = ceil(queue_count / warps_per_block)`。例如 queue_count=32、threads=128 → warps_per_block=4、blocks=8。每个 block 处理 4 个 queue（warp），共 32 个 queue。

`launch_persistent_search` 在 unified 模式下，`blocks = direct_owner_block_count + query_block_count + 2`（+2 是 dispatcher 与 delta）；非 unified 模式下 `blocks = query_block_count`。CPU 侧（见第 13 课 launch）会预先算好这些值并填入 `PersistentKernelParams`。

## 关键数据结构 / 流程图

### Block 角色分流图（unified 模式）

```
                     persistent_search_kernel<<<blocks, threads>>>(params)
                                       │
                                       ▼
                 ┌─────────────────────────────────────┐
                 │ unified_dispatch = direct_owner_   │
                 │                     block_count!=0 │
                 └──────┬──────────────────────┬──────┘
                        │ 是                   │ 否
                        ▼                      ▼
        ┌───────────────────────────┐   ┌────────────────────────┐
        │ blockIdx < owner_count ?  │   │ enable_queries=true    │
        │  是 → direct_read_owner_  │   │ enable_delta=true      │
        │       loop; return        │   │ enable_dispatcher=false│
        │  否 ↓                     │   │ (所有 query block 都   │
        └─────────────┬─────────────┘   │  enable_delta)         │
                      ▼                  └───────────┬────────────┘
        role_block = blockIdx - owner_count          │
                      │                              │
        ┌─────────────┼──────────────┬──────────────┐│
        ▼             ▼              ▼              ▼│
  < query_count   ==query_count  ==query_count+1  其它│
   query CTA      dispatcher       control/delta   退出│
   enable_queries  enable_         enable_delta       │
                   dispatcher                        │
        │             │              │                │
        │             │              │                │
        ▼             ▼              ▼                │
   atomicAdd      atomicAdd     atomicAdd             │
   (query_ready)  (dispatcher_  (control_ready)       │
                   ready)                             │
        └─────────────┴──────────────┴────────────────┘
                      │
                      ▼
              进入主循环（见下）
```

### 主循环控制流

```
┌─────────────────────────────────────────────────────────────────┐
│ for (;;) {                                                       │
│   if (threadIdx==0) stop_requested = *params.stop (volatile);   │
│   __syncthreads();                                               │
│   if (stop_requested) return;                                    │
│                                                                  │
│   ┌── enable_dispatcher ?                                        │
│   │   是: lane0 try_pop submissions → try_push device_sub       │
│   │       progressed ? reset idle : idle*=2 (≤16384)            │
│   │       __syncthreads(); continue;                             │
│   │                                                              │
│   ├── lane0: have_delta_submission =                            │
│   │       enable_delta && try_pop(delta_submissions, delta_desc)│
│   │   __syncthreads();                                           │
│   │   have_delta_submission ?                                    │
│   │     是: ┌─ delta 处理（见下）─┐                              │
│   │       │  continue;            │                              │
│   │       └────────────────────────┘                              │
│   │                                                              │
│   └── lane0: query_queue = device_submissions ?: submissions    │
│       have_submission = enable_queries && try_pop(query_queue)  │
│       __syncthreads();                                           │
│       have_submission ?                                          │
│         否: idle 退避; continue;                                 │
│         是: idle 复位; process_query(descriptor);                │
│             __syncthreads();                                     │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### delta 命令处理控制流

```
┌─ thread0: delta_status = 0; 解析 reset/promote flags ───────────┐
│                                                                  │
│  巨大布尔表达式校验 flags / 字段合法性 / 指针非空                 │
│  → delta_status = -EINVAL 若任何一项失败                         │
│  __syncthreads();                                                │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌── kDeltaCommandReset ? ──
        │   是:
        │     if (delta_status==0):
        │       并行清 delta_remote / base_override
        │       并行清 delta_records / next / prev / positions
        │       并行清 bucket_heads
        │       __syncthreads();
        │       thread0: __threadfence; delta_count=0;
        │                 __threadfence_system
        │     thread0: push DeltaPublishCompletion(final_count=0)
        │     continue;
        │
        ▼ 否（publish 模式）
┌─ 并行校验 staging slot 范围 → delta_status=-EINVAL ─┐
│  __syncthreads();                                    │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ 并行预校验 dynamic_route updates（flags/shard/epoch/┐
│   remote_node/重复 slot/版本号）→ -EINVAL 或 -ESTALE │
│  __syncthreads();                                    │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ delta_status==0 时：                              ─┐
│  并行写 delta_records / delta_vectors               │
│  __syncthreads();                                   │
│  并行 OPQ 变换 → delta_encode_scratch               │
│  __syncthreads();                                   │
│  并行 PQ 编码 → delta_pq_codes + resident_pq_codes  │
│  __threadfence(); __syncthreads();                  │
│                                                     │
│  并行图缓存失效（route + 组相联 cache）              │
│                                                     │
│  thread0 串行:                                       │
│    supersede（unlink）                               │
│    override（promote bitset 或 base_override 表）    │
│    durable（unlink + remote 清 + permanent bit +     │
│            base_override 清）                        │
│    resident_pq_erase                                 │
│  __syncthreads();                                   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ thread0: 发布 delta_remote 表 + bucket 链表 ──────┐
│   insert_resident_pq / delta_remote 插入           │
│   bucket head insert                               │
│  __syncthreads();                                  │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ delta_status==0 时：dynamic_route seqlock 发布 ─┐
│  Step 1: 并行 sequence.fetch_add(1, acq_rel) 变奇  │
│  __syncthreads();                                  │
│  Step 2: 并行写 dynamic_route_pq_codes             │
│  __threadfence(); __syncthreads();                 │
│  Step 3: 并行写 metadata + __threadfence +         │
│          sequence.fetch_add(1, release) 变偶       │
│  __syncthreads();                                  │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ thread0: __threadfence; delta_count = final_count;─┐
│          __threadfence_system                       │
│  __syncthreads();                                   │
│  thread0: push DeltaPublishCompletion               │
│  __syncthreads();                                   │
│  thread0: idle_cycles = 256                         │
│  __syncthreads();                                   │
│  continue;                                          │
└─────────────────────────────────────────────────────┘
```

## 与其他模块的关系

- **与第 13 课（construction/launch）**：CPU 侧的 `launch_persistent_search` / `launch_direct_read_owners` 在 `persistent_kernel.cu` 里。第 13 课讲的是如何准备 `PersistentKernelParams`、决定 block 数量、做启动握手。本课是这些决定的"接收端"——kernel 怎么用这些参数分流角色、怎么回报 ready。
- **与第 14 课（admission/completion）**：query 角色调用的 `process_query` 会写 completion 到 `completions` ring，CPU 侧从该 ring drain 结果。本课的主循环决定了"何时调 process_query"与"何时不能调"（delta 优先、stop 优先）。
- **与第 15 课（增量发布）**：CPU 侧的发布协议把 `DeltaPublishDescriptor` push 到 `delta_submissions` ring、从 `delta_completions` ring 读结果。本课是发布协议的"GPU 端实现"——校验、执行、原子发布 `delta_count`、写 completion。`kDeltaCommandReset` / `kDeltaCommandPromoteOverrides` 两种命令的语义在本课落实。
- **与第 17 课（kernel 启动器/上下文/device ring）**：`DeviceRingView`、`device_ring_try_pop/push/relax`、ready count 握手都在第 17 课介绍。本课大量使用这些原语，特别是 dispatcher 的"双 ring 搬运"完全建立在 device ring 之上。
- **与第 18 课（候选评分）**：`candidate_scoring.cuh` 提供 `hash32/hash64/anchor_graph_slot/insert_resident_pq/erase_resident_pq/unlink_mutable_delta` 等工具，本课的 delta 处理路径直接调用。
- **与第 19 课（RDMA cache）**：`direct_fetch` / `direct_fetch_batch` 是查询侧发起 GPUNetIO 读的接口，它们把请求 push 到 `direct_batch_queues`，本课的 `direct_read_owner_loop` 从这个 ring 取请求并执行。两者通过 `DirectBatchDescriptor.completion_status` 同步。
- **与第 20 课（查询遍历主循环）**：`process_query` 是第 20 课的主题。本课只讲"主循环何时调用 process_query"，不涉及内部。
- **与第 22 课（GPUNetIO 传输/probe）**：`direct_read_owner_loop` 的 DOCA verbs 细节、三个 probe kernel 的链路诊断用法在第 22 课展开。本课只讲运行时结构（warp 分工、批量提交、phase 暴露）。
- **与第 28 课（计算侧 storage owner 更新）**：CPU 侧如何决定发 `kDeltaCommandReset` vs 普通 publish、如何处理 `-ESTALE` 重试，在第 28 课讲。本课是这些决策的"执行端"。

## 小结

本课讲解了 `runtime.cuh` 的核心：一个 `__global__` kernel 如何通过 `blockIdx` 分流成四种角色（owner / query / dispatcher / control-delta），每种角色如何在自己的主循环里协作。

关键设计模式：

1. **单 kernel 多角色**（unified 模式）：一次 launch 覆盖所有功能，简化 CPU 侧调度，但要求 block 布局严格按 `[owner | query... | dispatcher | delta]` 排列。
2. **shared memory 单写者 + `__syncthreads`**：CTA 内一致性靠"thread 0 写 + fence + sync"而非 shared atomic，逻辑清晰。
3. **`__threadfence_system` 跨 PCIe 可见性**：凡是要让 CPU 看到的状态（stop / ready / delta_count / direct_disabled / owner phase）都必须配 system fence。
4. **idle 退避状态机**：`256 + (blockIdx * 131 & 1023)` 的去同步初值，翻倍到 16384 上限，progress 后复位。这是无锁 ring 的标准配套。
5. **delta 校验链**：70 行布尔表达式把所有"业务约束 + 指针就绪"一次性检查，把 GPU kernel 中段 panic 转成早期 EINVAL。
6. **seqlock 发布 dynamic_route**：奇/偶 sequence + acq_rel/release，查询侧无锁读、写者不阻塞读者。
7. **批量 WQE 合并提交**：owner warp 用 `__ballot_sync` + `__popc` 算 rank，每 batch 只产生一个 CQE，最大化 RDMA 吞吐。
8. **`direct_owner_phases` 调试通道**：每个 warp 把内部状态写到 device memory，CPU 远程观测，无需 cuda-gdb。

这些模式共同构成了 dvstor 的"GPU 当服务进程"运行时——一个永不退出的 kernel，靠 ring 与 CPU 通信，靠 shared memory 协调 CTA 内线程，靠 seqlock 与 atomic 实现无锁并发。第 22 课将深入 GPUNetIO verbs 的细节，第 23 课起转向存储节点主体。
