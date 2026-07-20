# 第 5 课 · RDMA 传输库（下）：QP 与连接管理

> 本课是第 4 课的延续。第 4 课讲的是 RDMA verbs 的“资源侧”：`Context` 打开设备、分配 PD、创建 CQ/SRQ，`MemoryRegion` 注册内存并产出 `MemoryRegionToken{address, lkey, rkey}`。本课讲“通道侧”：在已就绪的 `Context` 之上，如何把一个 `ibv_qp*` 包成 `QueuePair` 类、如何用 TCP 把两端的 QP 信息搬到对端、再把 QP 推过 `RESET → INIT → RTR → RTS` 状态机让它真正能跑 RDMA，最后如何把这套流程编排成 `ServerConnectionManager` / `ClientConnectionManager` 两个角色，让计算节点和存储节点复用同一套连接代码。

## 5.1 本课目标与涉及文件

读完本课你应当能够：

1. 解释 `QueuePair` 对 `ibv_qp*` 的封装边界：构造即 `RESET → INIT`，`transition_to_rtr/transition_to_rts` 各自修改了哪些 QP 属性，PSN/timeout/retry/rnr_retry/max_rd_atomic/min_rnr_timer 这些魔法数字的物理含义。
2. 写出 `post_send` / `post_receive` / `post_CAS` / `post_FAA` 的 WR + SGE 组织方式，并说出 `IBV_SEND_INLINE` / `IBV_SEND_SIGNALED` 两个标志对 CQE 产出与零拷贝路径的影响。
3. 解释 `DetachedQP` 为什么要“自带 CQ”这一变体，以及它如何复用一条已经连通的 QP 作为“信道”来给新 QP 做 pairing。
4. 描绘 `ServerConnectionManager::connect_to_clients` / `ClientConnectionManager::connect`（三步：`connect_among_clients` → `distribute_client_ids` → `connect_to_servers`）的拓扑：谁是 initiator、谁 listen、谁 connect、`QPInfo` 如何在 TCP 上往返。
5. 画出“TCP 握手 → 交换 lid/qpn/node_id → RTR → RTS → post_send/poll_cq”的时序图，并指出该库被第 11/22 课（计算节点）与第 23 课（存储节点）如何复用。
6. 理解 `BatchedREAD` 这种 CPU 侧把多个 `IBV_WR_RDMA_READ` 通过 `wr.next` 串成链一次 `ibv_post_send` 的合并下发工具（注意：这是 CPU 侧 verbs 的批合并，与第 19 课 GPU 侧 `rdma_cache.cuh` 的合并是两条独立路径）。

涉及文件（全部位于 `rdma-library/library/`）：

| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `queue_pair.hh` | 129 | `QueuePair` 类声明、`QPInfo` 结构、常量 `INLINE_SIZE/MESSAGE_SIZE` |
| `queue_pair.cc` | 390 | 全部实现：构造、状态机、post_send/recv/CAS/FAA |
| `detached_qp.hh` | 90 | 独立 QP + 可选自带 CQ；用已有 QP 当信道完成 pairing |
| `connection_manager.hh` | 57 | `ConnectionManager` 基类 + Server/Client 两个子类 |
| `connection_manager.cc` | 126 | 三段式连接编排、client_id 分发、同步屏障 |
| `batched_read.hh` | 74 | CPU 侧批量 RDMA read 合并下发 |
| `context.hh/cc` | 97/338 | （支撑材料，第 4 课已讲）`wait_for_connection` / `connect_to_server` 是 TCP 握手的真正落点 |

> 关于“access token”术语：代码里 `MemoryRegionToken{address, lkey, rkey}` 就是让对端能远程访问本地 MR 的三元组。计算节点在 `src/service/compute_service/index_commands.cc:102` 打印 `"receive access tokens of remote memory regions"`，存储节点在 `src/memory_node/memory_node.cc:163` 打印 `"register memory and distribute access token"` 并在第 167 行附近把 token 发给所有计算节点。本课讲的 `QP` 与 `ConnectionManager` 是承载这些 token 交换的“管道”，token 本身的语义在第 4 课已讲。

## 5.2 `queue_pair.hh`：接口与常量

```cpp
// rdma-library/library/queue_pair.hh:15-22
constexpr u32 INLINE_SIZE = 256;
constexpr u32 MESSAGE_SIZE = 1073741824;  // = 1GB (is max on our machines)

struct QPInfo {
  u16 lid{0};
  u32 qp_number{0};
  u32 node_id{0};
};
```

- `INLINE_SIZE = 256`：内联发送的最大字节数。verbs 的 `IBV_SEND_INLINE` 允许把数据直接嵌在 WQE 里下发，绕过“先注册 MR、再用 lkey 引用”的路径，但单条 WR 能塞的数据量受 `cap.max_inline_data` 限制。256 是个保守值，适配 ConnectX 系列网卡。
- `MESSAGE_SIZE = 1GB`：注释写明“is max on our machines”。这是一条 SEND/READ/WRITe 的硬上限，`post_send` 里会做 `lib_assert(size <= MESSAGE_SIZE, "Message size too large")`。
- `QPInfo{lid, qp_number, node_id}`：TCP 握手要交换的三元组。`lid` 是 LID（Local Identifier，IB 子网内 16-bit 地址）；`qp_number` 是对端 QP 的 QPN；`node_id` 是逻辑节点编号，**只有 client → server 方向会带**（见 5.4 节 `connect_to_server` 第三个参数），server → client 方向是 0。

```cpp
// rdma-library/library/queue_pair.hh:24-37
class QueuePair {
public:
  explicit QueuePair(Context* context, bool use_shared_receive_cq = false);
  QueuePair(Context* context,
            ibv_cq* send_cq,
            ibv_cq* recv_cq,
            bool use_shared_receive_cq = false);

  QueuePair(const QueuePair&) = delete;
  QueuePair& operator=(const QueuePair&) = delete;
  ~QueuePair();

  u32 get_qp_num() { return queue_pair_->qp_num; }
  ibv_qp* get_ibv_qp() { return queue_pair_; }
  u32 max_send_wr() const { return max_send_wr_; }
```

- 两个构造函数：第一个用 `Context` 自带的 `send_cq_` / `receive_cq_`（共享 CQ，绝大多数场景用这个，`Context::wait_for_connection` / `connect_to_server` 都走它）；第二个允许外部传 CQ 进来——这是给 `DetachedQP` 的“自带 CQ”变体留的口子（5.3 节）。
- `delete` 拷贝构造/赋值：`QueuePair` 持有 `ibv_qp*`，是独占资源，禁止拷贝（否则析构会 double destroy）。
- `get_qp_num()` 直接读 `queue_pair_->qp_num`：QPN 在 `ibv_create_qp` 后由硬件分配，构造期就已就绪，所以 `Context::wait_for_connection` 能立刻把它塞进 `QPInfo` 发给对端。
- `get_ibv_qp()` 返回裸指针：`BatchedREAD::post_batch` 需要它来调用 `ibv_post_send(qp, wr_list, &bad)`（5.6 节），因为批量下发是 verbs 原生 API，`QueuePair` 没必要再包一层。
- `max_send_wr()`：保存构造时硬件实际批准的发送队列深度。`cap.max_send_wr` 在 `ibv_create_qp` 后可能被驱动上调或下调，这里读的是返回值。

```cpp
// rdma-library/library/queue_pair.hh:40-42
void transition_to_init();
void transition_to_rtr(const QPInfo& remote_buffer);
void transition_to_rts();
```

三个状态迁移函数对应 verbs QP 状态机的三跳（`RESET → INIT` 已在构造函数里做掉）：

- `RESET → INIT`：设置端口、PKey、access flags。INIT 之后才能 post recv。
- `INIT → RTR`（Ready To Receive）：设置对端 QPN、LID、PSN、path MTU、min_rnr_timer、max_dest_rd_atomic。RTR 之后才能收。
- `RTR → RTS`（Ready To Send）：设置 sq_psn、timeout、retry_cnt、rnr_retry、max_rd_atomic。RTS 之后才能发。

```cpp
// rdma-library/library/queue_pair.hh:44-49
void post_receive(MemoryRegion& region);
void post_receive(MemoryRegion& region,
                  u32 size_in_bytes,
                  u64 wr_id = 0,
                  u64 local_offset = 0);
u32 receive_u32(Context& context);
```

接收侧两个重载 + 一个糖水函数 `receive_u32`：前者用 `region` 的全尺寸，后者允许指定 `size_in_bytes` / `wr_id` / `local_offset`。`wr_id` 会在 CQE 里原样返回，poll 时用来反查“这条完成是谁的”——`Context::post_shared_receive` 里就把 `wr_id` 设成 `&region` 指针，poll 时直接 reinterpret_cast 回 `MemoryRegion*`（见 `context.cc:208` 与 `context.cc:236`）。

```cpp
// rdma-library/library/queue_pair.hh:51-79
void post_send_inlined(const void* address,
                       u32 size_in_bytes,
                       enum ibv_wr_opcode opcode,
                       bool signaled = true,
                       MemoryRegionToken* token = nullptr,
                       u64 remote_offset = 0,
                       u64 wr_id = 0);
void post_send_u32(u32& value, bool signaled);
void post_send(MemoryRegion& region,
               enum ibv_wr_opcode opcode,
               bool signaled = true,
               MemoryRegionToken* token = nullptr,
               u64 remote_offset = 0,
               u64 local_offset = 0);
void post_send(MemoryRegion& region,
               u32 size_in_bytes,
               enum ibv_wr_opcode opcode,
               bool signaled = true,
               MemoryRegionToken* token = nullptr,
               u64 remote_offset = 0,
               u64 local_offset = 0);
void post_send_with_id(MemoryRegion& region,
                       u32 size_in_bytes,
                       enum ibv_wr_opcode opcode,
                       u64 wr_id,
                       bool signaled = true,
                       MemoryRegionToken* token = nullptr,
                       u64 remote_offset = 0,
                       u64 local_offset = 0);
```

发送侧五个公开重载，全部委托给私有 10 参数版 `post_send(u64 address, u32 size, u32 lkey, ...)`：

- `post_send_inlined`：用裸指针（不需要 MR），强制 inline，`lkey` 传 0。
- `post_send_u32`：发 4 字节，inline + SEND，专给 `distribute_client_ids` 这种“传个 8 字节元数据”场景用。
- `post_send(region, opcode, ...)`：用 MR 全尺寸。
- `post_send(region, size_in_bytes, opcode, ...)`：用 MR 但限定 size（offset 仍由 `local_offset` 控制）。
- `post_send_with_id`：带 `wr_id`，让上层在批量 poll 时能区分完成事件。

```cpp
// rdma-library/library/queue_pair.hh:92-115
void post_CAS(MemoryRegion& local_region,
              MemoryRegionToken* remote_token,
              u64 remote_offset,
              u64 compare_to,
              u64 swap_with,
              bool signaled = true,
              u64 wr_id = 0);

void post_CAS(u64 laddr,
              u32 lkey,
              MemoryRegionToken* remote_token,
              u64 remote_offset,
              u64 compare_to,
              u64 swap_with,
              bool signaled = true,
              u64 wr_id = 0);

void post_FAA(u64 laddr,
              u32 lkey,
              MemoryRegionToken* remote_token,
              u64 remote_offset,
              u64 to_add,
              bool signaled = true,
              u64 wr_id = 0);
```

两个原子操作：CAS（`IBV_WR_ATOMIC_CMP_AND_SWP`）和 FAA（`IBV_WR_ATOMIC_FETCH_AND_ADD`）。CAS 有 MR 版和裸地址版；FAA 只有裸地址版。`remote_offset` 必须按 8 字节对齐（`post_CAS` 里 `lib_assert(remote_offset % 8 == 0, ...)`），这是 verbs 对原子操作的硬约束。这两个原语是后面 RCU 回收、引用计数等无锁数据结构的基础（见第 16 课）。

私有成员：

```cpp
// rdma-library/library/queue_pair.hh:117-124
private:
  Context* context_;
  const u16 lid_;
  const bool use_shared_receive_cq_;

  ibv_qp* queue_pair_{nullptr};
  u32 max_send_wr_{};
};
```

- `context_`：回指针，用来在状态迁移里访问 `Context` 的配置/CQ/SRQ/device attributes。
- `lid_`：构造时 `context->get_lid()` 缓存，但实际代码里没用到这个字段（RTR 用的是 `remote_buffer.lid`，不是本地 lid）。保留它是为了调试或未来扩展。
- `queue_pair_`：verbs 句柄，析构时 `ibv_destroy_qp`。
- `max_send_wr_`：构造后从 `init_attributes.cap.max_send_wr` 回填。

文件末尾两个别名：

```cpp
// rdma-library/library/queue_pair.hh:126-127
using QP = u_ptr<QueuePair>;
using QPs = vec<QP>;
```

`QP = std::unique_ptr<QueuePair>`，`QPs = std::vector<QP>`。`ConnectionManager` 的 `client_qps` / `server_qps` / `initiator_qp` 全是这两个类型，独占所有权。

## 5.3 `queue_pair.cc`：逐函数实现

### 5.3.1 构造与析构

```cpp
// rdma-library/library/queue_pair.cc:6-10
// delegating ctor
QueuePair::QueuePair(Context* context, bool use_shared_receive_cq)
    : QueuePair(context,
                context->get_send_cq(),
                context->get_receive_cq(),
                use_shared_receive_cq) {}
```

单参数构造是“委派构造”（delegating constructor，C++11）：把 `Context` 自带的 `send_cq_` / `receive_cq_` 取出来，转发给四参数版。这样两条入口最终走同一段创建逻辑。

```cpp
// rdma-library/library/queue_pair.cc:12-27
QueuePair::QueuePair(Context* context,
                     ibv_cq* send_cq,
                     ibv_cq* recv_cq,
                     bool use_shared_receive_cq)
    : context_(context),
      lid_(context->get_lid()),
      use_shared_receive_cq_(use_shared_receive_cq) {
  ibv_qp_init_attr init_attributes =
    get_qp_initial_attributes(send_cq, recv_cq);
  queue_pair_ =
    ibv_create_qp(context->get_protection_domain(), &init_attributes);
  lib_assert(queue_pair_, "Cannot create queue pair");
  max_send_wr_ = init_attributes.cap.max_send_wr;

  transition_to_init();
}
```

四件事按顺序做：

1. `get_qp_initial_attributes` 拼出 `ibv_qp_init_attr`（5.3.2）。
2. `ibv_create_qp(pd, &attr)` 在 `Context` 的 protection domain 里创建 QP。**这一步硬件就把 QPN 分配好了**，所以 `get_qp_num()` 立刻可用。
3. `lib_assert` 检查非空——失败直接 `std::exit`（`utils.hh:17-23` 的宏语义）。
4. 回填 `max_send_wr_`：`ibv_create_qp` 会把驱动实际批准的深度写回 `attr.cap.max_send_wr`，可能比请求值小。
5. `transition_to_init()`：构造即把 QP 推到 INIT。**这是关键设计决定**——调用者拿到 `QueuePair` 对象时，QP 已经能 post recv 了，不需要再手动调一次 `transition_to_init`。`Context::wait_for_connection` 里 `make_unique<QueuePair>(this)` 之后立刻 `get_qp_num()` 拿 QPN 去 TCP 交换，就是这个保证在起作用。

析构：

```cpp
// rdma-library/library/queue_pair.cc:29-31
QueuePair::~QueuePair() {
  lib_assert(ibv_destroy_qp(queue_pair_) == 0, "Cannot destroy queue pair.");
}
```

只销毁 QP 本身，不碰 CQ/SRQ——它们由 `Context` 拥有（共享 CQ 场景）或 `DetachedQP` 拥有（自带 CQ 场景）。verbs 的销毁顺序要求：先销毁 QP，再销毁它用过的 CQ/SRQ。`Context` 的析构里 `ibv_destroy_srq → ibv_destroy_cq × 2 → ibv_dealloc_pd → ibv_close_device` 正好遵守这个顺序（`context.cc:88-100`）。

### 5.3.2 `get_qp_initial_attributes`：QP 创建属性

```cpp
// rdma-library/library/queue_pair.cc:33-55
ibv_qp_init_attr QueuePair::get_qp_initial_attributes(ibv_cq* send_cq,
                                                      ibv_cq* recv_cq) {
  ibv_qp_init_attr attributes{};
  const i32 max_sge_elements = 1;

  // FYI: if a shared rcq is used, no normal receive request RR can be posted
  if (use_shared_receive_cq_) {
    attributes.srq = context_->get_shared_receive_cq();
  }

  attributes.send_cq = send_cq;
  attributes.recv_cq = recv_cq;
  attributes.cap.max_send_wr = context_->get_config().max_send_queue_wr;
  attributes.cap.max_send_sge = max_sge_elements;
  attributes.cap.max_recv_wr = context_->get_config().max_recv_queue_wr;
  attributes.cap.max_recv_sge = max_sge_elements;
  attributes.cap.max_inline_data = INLINE_SIZE;
  attributes.qp_type = IBV_QPT_RC;
  // if 1, all WRs will generate CQEs, if 0, only flagged WRs generate CQEs
  attributes.sq_sig_all = 0;

  return attributes;
}
```

逐字段：

- `attributes{}`：值初始化，所有字段清零（C 风格 struct 用 `{}` 比较安全）。
- `max_sge_elements = 1`：每条 WR 只挂一个 SGE。这是个保守选择——多数 RDMA 代码都用单 SGE + 连续 buffer，避免 SGE 链带来的额外解析开销。如果要做 gather/scatter，需要扩到多 SGE。
- `use_shared_receive_cq_` 为真时挂上 SRQ。注释提示：“if a shared rcq is used, no normal receive request RR can be posted”——一旦用 SRQ，receive queue 就被 SRQ 取代，`ibv_post_recv` 不再可用，必须 `ibv_post_srq_recv`。本课里 `post_receive` 走的是 RR 路径，所以默认不开 SRQ。
- `send_cq` / `recv_cq`：verbs 要求 RC QP 必须挂 send/recv 两个 CQ。可以共用一个 CQ（指针相同），这里默认分开。
- `cap.max_send_wr = config.max_send_queue_wr`（默认 1024，`configuration.hh:14`）：发送队列深度。
- `cap.max_recv_wr = config.max_recv_queue_wr`（默认 1024）：接收队列深度（或 SRQ 的深度）。
- `cap.max_inline_data = INLINE_SIZE = 256`：内联发送上限。
- `qp_type = IBV_QPT_RC`：**可靠连接**（Reliable Connection）。RC 模式下，一个 QP 绑定一个对端 QP，硬件保证顺序、重传、完整性。与之对照的 UD（不可靠数据报）本课不用。
- `sq_sig_all = 0`：不强制每条 SEND WR 都产生 CQE。配合 `IBV_SEND_SIGNALED` 标志，只有显式 signaled 的 WR 才会出 CQE。这是性能关键——批量 SEND 时只给最后一条 signaled，能大幅减少 CQE 风暴。`BatchedREAD::post_batch` 就是这么做的（5.6 节）。

### 5.3.3 状态机：`transition_to_init`

```cpp
// rdma-library/library/queue_pair.cc:57-74
// transition state of queue pair from RESET to INIT:
// basic information set, ready for posting to receive queue.
void QueuePair::transition_to_init() {
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_INIT;
  attributes.pkey_index = 0;
  attributes.port_num = context_->get_config().device_port;
  attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_ATOMIC;
  lib_assert(ibv_modify_qp(queue_pair_,
                           &attributes,
                           IBV_QP_STATE | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS |
                             IBV_QP_PKEY_INDEX) == 0,
             "Cannot change state of queue pair to INIT");
  lib_debug("Transitioned state to INIT successfully");
}
```

`ibv_modify_qp` 的第三个参数是“mask”，指明 `attributes` 里哪些字段是有效的——没在 mask 里的字段被驱动忽略。这里 mask = `STATE | PORT | ACCESS_FLAGS | PKEY_INDEX`，对应四个字段：

- `qp_state = IBV_QPS_INIT`：目标状态。
- `port_num = config.device_port`（默认 1）：QP 走物理端口 1（多数 IB 卡 port 1 是主口）。
- `pkey_index = 0`：Partition Key 表的索引 0，默认分区。
- `qp_access_flags`：声明“本 QP 允许对端做哪些远程操作”。四个 flag 全开：
  - `IBV_ACCESS_REMOTE_WRITE`：对端可以 RDMA WRITE 进来。
  - `IBV_ACCESS_REMOTE_READ`：对端可以 RDMA READ 出去。
  - `IBV_ACCESS_LOCAL_WRITE`：本地 CPU 可写（这是 RDMA WRITE 的前提——如果本地不可写，对端也没法写进来）。
  - `IBV_ACCESS_REMOTE_ATOMIC`：对端可以发原子操作（CAS/FAA）。

注释“ready for posting to receive queue”说明 INIT 状态的语义：可以 post recv，但还不能收（要 RTR）、不能发（要 RTS）。

### 5.3.4 状态机：`transition_to_rtr`

```cpp
// rdma-library/library/queue_pair.cc:76-99
void QueuePair::transition_to_rtr(const QPInfo& remote_buffer) {
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_RTR;
  attributes.path_mtu = IBV_MTU_4096;
  attributes.dest_qp_num = remote_buffer.qp_number;
  attributes.rq_psn = 0;
  attributes.max_dest_rd_atomic = context_->max_qp_dest_read_atomic();
  attributes.min_rnr_timer = 12;
  attributes.ah_attr.is_global = 0;
  attributes.ah_attr.dlid = remote_buffer.lid;
  attributes.ah_attr.sl = 0;
  attributes.ah_attr.src_path_bits = 0;
  attributes.ah_attr.port_num = context_->get_config().device_port;

  lib_assert(
    ibv_modify_qp(queue_pair_,
                  &attributes,
                  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER |
                    IBV_QP_MAX_DEST_RD_ATOMIC) == 0,
    "Cannot change state of queue pair to RTR");
  lib_debug("Transitioned state to RTR successfully");
}
```

`remote_buffer` 就是 TCP 收到的对端 `QPInfo`。逐字段：

- `qp_state = IBV_QPS_RTR`：目标状态。
- `path_mtu = IBV_MTU_4096`：路径 MTU 设为 4096（4K 字节），这是 IB 的最大 MTU。两端 MTU 必须一致或不大于路径 MTU，写 4096 意味着“尽量用大包”。RoCE 网卡通常也支持 4096。
- `dest_qp_num = remote_buffer.qp_number`：**对端的 QPN**。RC QP 是一对一绑定的，这一步把“我这个 QP”和“对端那个 QPN”锁死。
- `rq_psn = 0`：接收侧 PSN（Packet Sequence Number）。PSN 是 IB 包头里的序号，用来检测丢包/乱序。设 0 意味着对端发来的第一个包 PSN 应该是 0（发送侧 `sq_psn` 也设 0，对应）。两端 PSN 不一定要相同，只要双方知道对方起始 PSN 即可，这里都用 0 简化。
- `max_dest_rd_atomic = context_->max_qp_dest_read_atomic()`：本 QP 作为“被读方”时，允许同时未完成的 RDMA READ / ATOMIC 请求数上限。`Context` 的实现（`context.hh:47-49`）是 `std::max<u32>(1, std::min<u32>(16, device_attributes_.max_qp_rd_atom))`，即取硬件能力和 16 的较小值，至少 1。这个数字直接影响对端能并发多少 outstanding read。
- `min_rnr_timer = 12`：RNR（Receiver Not Ready）NACK 的重试定时器。对端发 SEND 而本地 RQ 还没 post recv 时，硬件会回 RNR NACK，告诉对端“我还没准备好收”。`min_rnr_timer` 控制 NACK 里携带的“多久之后再试”的指数退避值。12 是经验值，约对应几百微秒。
- `ah_attr`（Address Handle）：路由信息。
  - `is_global = 0`：不走 Global Routing Header（GRH）。IB 局域网内 LID 路由就够，不需要 GRH（GRH 主要给 RoCEV2 跨子网用）。
  - `dlid = remote_buffer.lid`：目的 LID。
  - `sl = 0`：Service Level 0，默认优先级。
  - `src_path_bits = 0`：源路径位（LID 掩码的低位，用于 LMC 多路径），这里不用。
  - `port_num`：从哪个物理端口发。

mask = `STATE | AV | PATH_MTU | DEST_QPN | RQ_PSN | MIN_RNR_TIMER | MAX_DEST_RD_ATOMIC`，对应上面所有有效字段。注意 `ah_attr` 整体由 `IBV_QP_AV` 这一个 mask bit 控制。

### 5.3.5 状态机：`transition_to_rts`

```cpp
// rdma-library/library/queue_pair.cc:101-118
void QueuePair::transition_to_rts() {
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_RTS;
  attributes.timeout = 14;
  attributes.retry_cnt = 7;
  attributes.rnr_retry = 7;
  attributes.sq_psn = 0;
  attributes.max_rd_atomic = context_->max_qp_read_atomic();

  lib_assert(ibv_modify_qp(queue_pair_,
                           &attributes,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                             IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                             IBV_QP_MAX_QP_RD_ATOMIC) == 0,
             "Cannot change state of queue pair to RTS");
  lib_debug("Transitioned state to RTS successfully");
}
```

RTS 是发送侧的最后一跳。逐字段：

- `qp_state = IBV_QPS_RTS`：目标状态。
- `timeout = 14`：发送侧 ACK 超时定时器。这是个 4.096 μs × 2^timeout 的公式，14 对应约 67ms。超过这个时间没收到 ACK，硬件重传。值过小会在长延迟链路上误重传，过大会让真丢包的恢复变慢。
- `retry_cnt = 7`：包级重传次数上限（针对 ACK 超时或 NAK）。7 是verbs 允许的最大值。达到上限后 QP 进入 ERROR 态。
- `rnr_retry = 7`：RNR NACK 的重试次数上限。7 = 无限重试（verbs 里 7 是特殊值，表示一直试）。配合 `min_rnr_timer` 让对端“慢慢准备好”。
- `sq_psn = 0`：发送侧起始 PSN，与对端 `rq_psn = 0` 对齐。
- `max_rd_atomic = context_->max_qp_read_atomic()`：本 QP 作为“发起方”时，允许同时未完成的 RDMA READ / ATOMIC 数上限。`Context` 实现（`context.hh:44-46`）取 `max_qp_init_rd_atom` 与 16 的较小值。

mask = `STATE | TIMEOUT | RETRY_CNT | RNR_RETRY | SQ_PSN | MAX_QP_RD_ATOMIC`。

**到这里 QP 才真正能发能收**。`Context::wait_for_connection` / `connect_to_server` 都是 `transition_to_rtr(...)` 紧接 `transition_to_rts()`，两步之间不留窗口。

### 5.3.6 接收：`post_receive`

```cpp
// rdma-library/library/queue_pair.cc:120-122
void QueuePair::post_receive(MemoryRegion& region) {
  post_receive(region, region.get_size_in_bytes());
}
```

全尺寸重载，转给四参数版，`wr_id=0`、`local_offset=0`。

```cpp
// rdma-library/library/queue_pair.cc:124-147
void QueuePair::post_receive(MemoryRegion& region,
                             u32 size_in_bytes,
                             u64 wr_id,
                             u64 local_offset) {
  ibv_recv_wr work_request{};
  ibv_sge scatter_gather_entry{};

  // points to the RR that failed to be posted (if not successful)
  ibv_recv_wr* bad_work_request{nullptr};

  scatter_gather_entry.addr = region.get_address() + local_offset;
  scatter_gather_entry.length = size_in_bytes;
  scatter_gather_entry.lkey = region.get_lkey();

  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  // post receive request to receive queue
  lib_assert(ibv_post_recv(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post receive request");
  lib_debug("Receive request successfully posted");
}
```

 RECEIVE 的 WR 组织：

- `ibv_recv_wr work_request{}` + `ibv_sge scatter_gather_entry{}`：值初始化清零，避免脏字段干扰驱动。
- `bad_work_request`：输出参数，`ibv_post_recv` 失败时指向失败的那条 WR。这里没用，只是占位。
- SGE 三件套：
  - `addr = region.get_address() + local_offset`：数据落地起始地址。`local_offset` 让你能在一个大 MR 里复用多个偏移。
  - `length = size_in_bytes`：期望收到的最大字节数。实际收到的可能更少（用 CQE 的 `byte_len` 反查，`Context::poll_recv_cq` 里就是 `recv_info[i].bytes_written = work_completion[i].byte_len`）。
  - `lkey = region.get_lkey()`：本地访问 key。verbs 用它做 DMA 映射校验。
- WR：
  - `wr_id`：自定义 64-bit ID，CQE 里原样返回。`Context::post_shared_receive` 用 `&region` 作为 wr_id，poll 时反查 MR。
  - `next = nullptr`：单条 WR，链尾。批量 post 可以串成链，`BatchedREAD` 就是这么做的（5.6 节，但那是 send 侧）。
  - `sg_list` / `num_sge`：指向 SGE 数组，1 个元素。
- `ibv_post_recv`：把 WR 推到 RQ。返回 0 表示成功。

### 5.3.7 接收糖水：`receive_u32`

```cpp
// rdma-library/library/queue_pair.cc:149-157
u32 QueuePair::receive_u32(Context& context) {
  u32 value;

  LocalMemoryRegion region{context, std::addressof(value), sizeof(u32)};
  post_receive(region);
  context.receive();

  return value;
}
```

同步收 4 字节：栈上开 `u32 value`，用 `LocalMemoryRegion` 临时注册（`LocalMemoryRegion` 是 `MemoryRegion` 的子类，构造即 `ibv_reg_mr`，析构即 `ibv_dereg_mr`，见第 4 课），post recv，然后 `context.receive()` 阻塞 poll 直到拿到 CQE。返回 `value`。这是握手期“传个整数”的便利函数，热路径不用（频繁 reg/dereg MR 代价高）。

### 5.3.8 发送：五个重载 → 一个私有 10 参数版

五个公开重载都是把参数整理成“地址/大小/lkey/标志”四元组，转给私有版：

```cpp
// rdma-library/library/queue_pair.cc:159-176
void QueuePair::post_send_inlined(const void* address,
                                  u32 size_in_bytes,
                                  enum ibv_wr_opcode opcode,
                                  bool signaled,
                                  MemoryRegionToken* token,
                                  u64 remote_offset,
                                  u64 wr_id) {
  post_send(reinterpret_cast<u64>(address),
            size_in_bytes,
            0,
            opcode,
            signaled,
            true,
            token,
            remote_offset,
            0,
            wr_id);
}
```

`post_send_inlined`：`lkey=0`（inline 不需要 lkey，数据嵌在 WQE 里），`inlined=true`，`local_offset=0`（inline 不支持 offset，整段下发）。

```cpp
// rdma-library/library/queue_pair.cc:178-189
void QueuePair::post_send_u32(u32& value, bool signaled) {
  post_send(reinterpret_cast<u64>(std::addressof(value)),
            sizeof(u32),
            0,
            IBV_WR_SEND,
            signaled,
            true,
            nullptr,
            0,
            0,
            0);
}
```

`post_send_u32`：4 字节 SEND，inline，无 token（SEND 不需要 remote addr）。注意它发的是栈上 `value` 的引用——inline 模式下 verbs 在 `ibv_post_send` 调用期间把数据拷进 WQE，所以函数返回后栈帧销毁也没关系。

```cpp
// rdma-library/library/queue_pair.cc:191-207
void QueuePair::post_send(MemoryRegion& region,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset) {
  post_send(region.get_address(),
            region.get_size_in_bytes(),
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            0);
}
```

`post_send(region, opcode, ...)`：用 MR 全尺寸，`inlined=false`，`wr_id=0`。

```cpp
// rdma-library/library/queue_pair.cc:209-226
void QueuePair::post_send(MemoryRegion& region,
                          u32 size_in_bytes,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset) {
  post_send(region.get_address(),
            size_in_bytes,
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            0);
}
```

带 `size_in_bytes` 的重载：用 MR 的 lkey 但只发指定字节数（offset 由 `local_offset` 控制）。`distribute_client_ids` 里发 16 字节（2×u64）就是走这条路径……不，实际上它走的是 `post_send_inlined(content.data(), 2*sizeof(u64), IBV_WR_SEND)`，因为 content 是 `vec<u64>`，没注册 MR。看代码要精确。

```cpp
// rdma-library/library/queue_pair.cc:228-246
void QueuePair::post_send_with_id(MemoryRegion& region,
                                  u32 size_in_bytes,
                                  enum ibv_wr_opcode opcode,
                                  u64 wr_id,
                                  bool signaled,
                                  MemoryRegionToken* token,
                                  u64 remote_offset,
                                  u64 local_offset) {
  post_send(region.get_address(),
            size_in_bytes,
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            wr_id);
}
```

带 `wr_id` 的重载：唯一区别是 `wr_id` 透传。批量场景（`BatchedREAD`）需要 wr_id 区分完成事件，但 `BatchedREAD` 直接调 `ibv_post_send` 没走这个函数——`post_send_with_id` 是给“单条但需要 wr_id”的场景留的。

### 5.3.9 发送：私有 10 参数实现

```cpp
// rdma-library/library/queue_pair.cc:248-303
void QueuePair::post_send(u64 address,
                          u32 size,
                          u32 lkey,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          bool inlined,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset,
                          u64 wr_id) {
  lib_assert(!inlined || size <= INLINE_SIZE, "Request cannot be inlined");
  lib_assert(size <= MESSAGE_SIZE, "Message size too large");

  ibv_send_wr work_request{};
  ibv_sge scatter_gather_entry{};

  // points to the SR that failed to be posted (if not successful)
  struct ibv_send_wr* bad_work_request;

  scatter_gather_entry.addr = address + local_offset;
  scatter_gather_entry.length = size;
  scatter_gather_entry.lkey = lkey;

  work_request.opcode = opcode;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.send_flags |= inlined ? IBV_SEND_INLINE : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  if (opcode != IBV_WR_SEND) {
    lib_assert(token, "MemoryRegionToken does not exist");
    work_request.wr.rdma.remote_addr = token->address + remote_offset;
    work_request.wr.rdma.rkey = token->rkey;
  }

  // post send request to send queue
  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post send request");

  switch (opcode) {
  case IBV_WR_SEND:
    lib_debug("SEND request successfully posted");
    break;
  case IBV_WR_RDMA_READ:
    lib_debug("RDMA_READ request successfully posted");
    break;
  case IBV_WR_RDMA_WRITE:
    lib_debug("RDMA_WRITE request successfully posted");
    break;
  default:
    lib_failure("Unknown request posted");
    break;
  }
}
```

逐段：

- 两个前置断言：
  - `!inlined || size <= INLINE_SIZE`：inline 模式下 size 不能超过 256。这是硬件 `cap.max_inline_data` 的硬约束。
  - `size <= MESSAGE_SIZE`：单条 WR 不超过 1GB。
- SGE 三件套：和 `post_receive` 同构。`addr = address + local_offset`，inline 模式下 `local_offset` 必须是 0（调用者保证，`post_send_inlined` 传的就是 0）。
- `send_flags`：
  - `IBV_SEND_SIGNALED`：这条 WR 完成时产生 CQE。配合 `sq_sig_all=0`，只有显式 signaled 的 WR 才出 CQE。
  - `IBV_SEND_INLINE`：inline 模式，数据嵌在 WQE 里。
- `opcode != IBV_WR_SEND` 时必须有 `token`：RDMA READ/WRITE 需要远端地址+rkey，这两者来自 `MemoryRegionToken`。`SEND` 不需要（数据进对端 RQ，由对端 post recv 决定落地位置）。
  - `work_request.wr.rdma.remote_addr = token->address + remote_offset`：远端落地地址。
  - `work_request.wr.rdma.rkey = token->rkey`：远端访问 key。
  - 注意：CAS/FAA 用的是 `work_request.wr.atomic`（不是 `wr.rdma`），所以这条路径只覆盖 SEND/READ/WRITE。原子操作有独立的 `post_CAS` / `post_FAA`。
- `ibv_post_send`：推到 SQ。
- 最后 switch 只是 debug 日志，`lib_failure` 在非 debug 模式下也不退出（看 `utils.cc` 实现），主要是给 default 分支兜底。

### 5.3.10 原子：`post_CAS`

```cpp
// rdma-library/library/queue_pair.cc:304-319
void QueuePair::post_CAS(MemoryRegion& local_region,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 compare_to,
                         u64 swap_with,
                         bool signaled,
                         u64 wr_id) {
  post_CAS(local_region.get_address(),
           local_region.get_lkey(),
           remote_token,
           remote_offset,
           compare_to,
           swap_with,
           signaled,
           wr_id);
}

void QueuePair::post_CAS(u64 laddr,
                         u32 lkey,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 compare_to,
                         u64 swap_with,
                         bool signaled,
                         u64 wr_id) {
  lib_assert(remote_offset % 8 == 0, "CAS address must be 8B aligned");

  ibv_send_wr work_request{};
  ibv_sge sge{};

  struct ibv_send_wr* bad_work_request;

  sge.addr = laddr;
  sge.length = 8;
  sge.lkey = lkey;

  work_request.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &sge;
  work_request.num_sge = 1;

  auto& atomic = work_request.wr.atomic;
  atomic.remote_addr = remote_token->address + remote_offset;
  atomic.rkey = remote_token->rkey;

  atomic.compare_add = compare_to;
  atomic.swap = swap_with;

  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post CAS request");
}
```

CAS（Compare-And-Swap）：

- `remote_offset % 8 == 0`：原子操作要求 8 字节对齐，verbs 硬约束。
- SGE：`length=8`（64-bit 原子），`lkey` 是本地结果落地 buffer 的 key（CAS 把“旧值”写回本地 SGE 指向的地址）。
- `opcode = IBV_WR_ATOMIC_CMP_AND_SWP`。
- `wr.atomic`（不是 `wr.rdma`）：
  - `remote_addr` / `rkey`：远端原子操作目标。
  - `compare_add = compare_to`：期望值（语义：如果远端当前值 == compare_to，就写入 swap_with）。
  - `swap = swap_with`：新值。
- 完成后 CQE 产生，本地 SGE 指向的 8 字节被写入“操作前的远端旧值”，可以用来判断 CAS 是否成功。

### 5.3.11 原子：`post_FAA`

```cpp
// rdma-library/library/queue_pair.cc:358-389
void QueuePair::post_FAA(u64 laddress,
                         u32 lkey,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 to_add,
                         bool signaled,
                         u64 wr_id) {
  ibv_send_wr work_request{};
  ibv_sge sge{};

  struct ibv_send_wr* bad_work_request;

  sge.addr = laddress;
  sge.length = 8;
  sge.lkey = lkey;

  work_request.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &sge;
  work_request.num_sge = 1;

  auto& atomic = work_request.wr.atomic;
  atomic.remote_addr = remote_token->address + remote_offset;
  atomic.rkey = remote_token->rkey;

  atomic.compare_add = to_add;

  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post FAA request");
}
```

FAA（Fetch-And-Add）：

- 结构和 CAS 几乎一样，区别：
  - `opcode = IBV_WR_ATOMIC_FETCH_AND_ADD`。
  - `atomic.compare_add = to_add`：要加的值（有符号）。
  - 不设 `atomic.swap`（FAA 不需要）。
- 完成后本地 SGE 指向的 8 字节被写入“操作前的远端旧值”。

> 注意 `post_FAA` 没有像 `post_CAS` 那样显式断言 `remote_offset % 8 == 0`——这是代码不一致的地方，调用者必须自己保证对齐，否则 verbs 会在运行时返回错误。这种小瑕疵在阅读真实代码时要留意。

## 5.4 `detached_qp.hh`：独立 QP

`DetachedQP` 解决的场景：你需要一个“不绑定 `Context` 共享 CQ”的 QP——比如某个独立线程想自己 poll 自己的 CQ，避免和主线程的 CQ 抢锁；或者测试场景下想隔离 completion。它有两种构造方式。

```cpp
// rdma-library/library/detached_qp.hh:11-34
class DetachedQP {
public:
  // ctor with delegated completion queues
  DetachedQP(Context& context, ibv_cq* send_cq, ibv_cq* recv_cq)
      : send_cq_(send_cq), recv_cq_(recv_cq), owns_cqs_(false) {
    qp = std::make_unique<QueuePair>(&context, send_cq, recv_cq);
  }

  // ctor with own completion queues
  explicit DetachedQP(Context& context) : owns_cqs_(true) {
    send_cq_ = ibv_create_cq(context.get_raw_context(),
                             context.get_config().max_send_queue_wr,
                             nullptr,
                             nullptr,
                             0);
    recv_cq_ = ibv_create_cq(context.get_raw_context(),
                             context.get_config().max_recv_queue_wr,
                             nullptr,
                             nullptr,
                             0);
    lib_assert(send_cq_ && recv_cq_, "Cannot create completion queues");

    qp = std::make_unique<QueuePair>(&context, send_cq_, recv_cq_);
  }
```

- 第一种构造：外部传 CQ 进来，`owns_cqs_=false`，析构不销毁 CQ（CQ 由外部拥有）。
- 第二种构造：自己 `ibv_create_cq` 两个独立 CQ，`owns_cqs_=true`，析构要销毁。
  - `ibv_create_cq(context, cqe, cq_context, channel, comp_vector)`：`cqe` 是 CQ 深度（用 `max_send_queue_wr` / `max_recv_queue_wr`），`cq_context` 是用户上下文（nullptr），`channel` 是完成通道（nullptr，用轮询不用事件），`comp_vector` 是中断向量（0）。
  - 这两个 CQ 不进 `Context` 的管理，纯 `DetachedQP` 私有。

两种构造都把 CQ 透传给 `QueuePair` 的四参数构造（5.3.1），QP 本身的创建/状态机逻辑不变。

```cpp
// rdma-library/library/detached_qp.hh:36-44
  ~DetachedQP() {
    if (owns_cqs_) {
      qp.reset();  // destroy qp first
      lib_assert(ibv_destroy_cq(recv_cq_) == 0,
                 "Cannot destroy receive completion queue");
      lib_assert(ibv_destroy_cq(send_cq_) == 0,
                 "Cannot destroy send completion queue");
    }
  }
```

析构顺序很关键：

1. `qp.reset()` 先销毁 QP（`QueuePair` 析构调 `ibv_destroy_qp`）。
2. 再销毁 recv_cq，再销毁 send_cq。

verbs 要求：**QP 必须先于它使用的 CQ 销毁**，否则 `ibv_destroy_cq` 会因为“CQ 还被 QP 引用”而失败。`qp.reset()` 显式先做这一步，是 detached 场景下必须的手工顺序控制（`Context` 共享 CQ 场景下，`Context` 析构时所有 QP 已经先析构完，自然满足顺序）。

```cpp
// rdma-library/library/detached_qp.hh:49-58
  i32 poll_send_cq(ibv_wc* work_completion, const i32 max_cqes) const {
    return Context::poll_send_cq(work_completion, max_cqes, send_cq_);
  }

  i32 poll_recv_cq(ibv_wc* work_completion,
                   i32 max_cqes,
                   ReceiveInfo* recv_info = nullptr) const {
    return Context::poll_recv_cq(
      work_completion, max_cqes, recv_cq_, recv_info);
  }
```

`DetachedQP` 自己提供 poll 入口，调用 `Context::poll_send_cq` / `poll_recv_cq` 的静态重载（`context.hh:60-66`、`context.hh:73-75`），把自己持有的 CQ 指针传进去。这样调用者不需要通过 `Context` 就能 poll 自己的 CQ——这正是“detached”的核心价值。

### 5.4.1 `DetachedQP::connect`：用已有 QP 当信道做 pairing

```cpp
// rdma-library/library/detached_qp.hh:60-78
  // channel_context is the context we use for communication
  // (the context to which other_qp belongs to)
  void connect(Context& channel_context, u16 lid, QP& other_qp) const {
    QPInfo send_buffer{lid, qp->get_qp_num()}, receive_buffer{};
    LocalMemoryRegion region{channel_context, &receive_buffer, sizeof(QPInfo)};

    // other_qp is the qp we use to exchange information
    other_qp->post_receive(region);
    other_qp->post_send_inlined(&send_buffer, sizeof(QPInfo), IBV_WR_SEND);

    channel_context.poll_send_cq_until_completion();
    channel_context.receive();

    std::cerr << "pairing: " << qp->get_qp_num() << " -- "
              << receive_buffer.qp_number << std::endl;

    qp->transition_to_rtr(receive_buffer);
    qp->transition_to_rts();
  }
```

这是个很巧的复用模式：

- **不通过 TCP**，而是通过一条**已经连通的 QP**（`other_qp`）来交换新 QP 的 `QPInfo`。
- 场景：两个节点之间已经有一条 QP 连着（比如由 `ConnectionManager` 建立），现在想再开一条独立 QP（detached，自带 CQ），不想再走 TCP 握手。
- 步骤：
  1. `send_buffer = {lid, qp->get_qp_num()}`：新 QP 的本端信息。
  2. `LocalMemoryRegion region{channel_context, &receive_buffer, sizeof(QPInfo)}`：临时注册一个 MR 装收到的对端信息。
  3. `other_qp->post_receive(region)`：在旧 QP 上 post recv。
  4. `other_qp->post_send_inlined(&send_buffer, IBV_WR_SEND)`：在旧 QP 上 SEND 自己的新 QP 信息。
  5. `channel_context.poll_send_cq_until_completion()`：等发送完成。
  6. `channel_context.receive()`：等接收完成（阻塞 poll）。
  7. `qp->transition_to_rtr(receive_buffer)` / `transition_to_rts()`：用收到的对端信息把新 QP 推到 RTS。
- 对端要对称做同样的事（post recv + post send），两端都拿到对方的 QPInfo 后各自 RTR/RTS。

这个模式在“已经有一条管理信道、想按需开数据信道”的场景下很实用——避免了重复的 TCP 端口管理。第 22 课的 GPUNetIO transport 在某些初始化路径上也会借道已连通的 CPU QP 做信息交换。

## 5.5 `connection_manager.hh/cc`：连接编排

### 5.5.1 类层次

```cpp
// rdma-library/library/connection_manager.hh:10-23
class ConnectionManager {
public:
  using Configuration = configuration::Configuration;

public:
  Context& get_context() const { return context_; }

protected:
  ConnectionManager(Context& context, const Configuration& config);

protected:
  Context& context_;
  const Configuration& config_;
};
```

基类只持有 `context_` 和 `config_` 两个引用——纯粹的“共享上下文”角色，没有连接逻辑。protected 构造说明它只能被子类构造。

```cpp
// rdma-library/library/connection_manager.hh:25-34
class ServerConnectionManager : public ConnectionManager {
public:
  ServerConnectionManager(Context& context, const Configuration& config);
  void connect_to_clients();
  void synchronize() const;

public:
  QPs client_qps;
  QP& initiator_qp;
};
```

`ServerConnectionManager`（存储节点用）：

- `client_qps`：所有连进来的 client 的 QP 集合，按下标 = `client_id` 寻址。
- `initiator_qp`：引用 `client_qps.front()`——注意是 `QP&`（引用），不是 `QP`。构造函数里 `initiator_qp(client_qps.front())` 把它绑到第 0 号 client（也就是 initiator）。这是个语义捷径：“server 视角下的 initiator”就是 `client_qps[0]`。
- `connect_to_clients()`：listen + accept 所有 client。
- `synchronize()`：向所有 client 广播一个 `true`，作为同步屏障。

```cpp
// rdma-library/library/connection_manager.hh:36-54
class ClientConnectionManager : public ConnectionManager {
public:
  ClientConnectionManager(Context& context, const Configuration& config);
  void connect();
  bool synchronize() const;

private:
  void connect_among_clients();
  void distribute_client_ids();
  void connect_to_servers();

public:
  const bool is_initiator;
  u32 client_id{};
  u32 num_total_clients{};
  QPs server_qps;
  QPs client_qps;  // relevant only for the initiator
  QP initiator_qp;  // relevant only for non-initiators
};
```

`ClientConnectionManager`（计算节点用）：

- `is_initiator`：是否是发起者（const，构造期定）。initiator 负责给其他 client 编号。
- `client_id`：本 client 的编号（initiator 是 0，其他由 initiator 分配）。
- `num_total_clients`：client 总数。
- `server_qps`：连到所有 server 的 QP 集合。
- `client_qps`：**只有 initiator 用**——它要连到所有其他 client 来分发 id。
- `initiator_qp`：**只有非 initiator 用**——它连到 initiator 来收 id。
- `connect()`：三步连接编排。
- `synchronize()`：从所有 server 收一个 `bool`，AND 起来返回。

### 5.5.2 构造函数

```cpp
// rdma-library/library/connection_manager.cc:7-15
ConnectionManager::ConnectionManager(Context& context,
                                     const Configuration& config)
    : context_(context), config_(config) {}

ServerConnectionManager::ServerConnectionManager(Context& context,
                                                 const Configuration& config)
    : ConnectionManager(context, config),
      client_qps(config.num_clients),
      initiator_qp(client_qps.front()) {}
```

`ServerConnectionManager` 的成员初始化很有意思：

- `client_qps(config.num_clients)`：预分配 `num_clients` 个空 `QP`（`unique_ptr` 默认 nullptr）。这是为了后面按下标 `client_qps[client_id] = std::move(qp)` 直接填充。
- `initiator_qp(client_qps.front())`：**在 `client_qps` 还是空指针时就把引用绑上去**。这之所以合法，是因为 `vector` 的 `front()` 返回的是引用，绑定的不是当时指向的对象，而是 `client_qps[0]` 这个槽位本身——后面 `client_qps[0] = std::move(qp)` 之后，`initiator_qp` 自然指向新填入的 QP。这是 C++ 引用语义的一个常用技巧，但前提是 `client_qps` 之后不会触发 reallocation（已 reserve 且不扩容）。这里 `client_qps` 大小固定为 `num_clients`，不再 push_back，所以安全。

```cpp
// rdma-library/library/connection_manager.cc:17-23
ClientConnectionManager::ClientConnectionManager(Context& context,
                                                 const Configuration& config)
    : ConnectionManager(context, config), is_initiator(config.is_initiator) {
  // reserve memory for queue pairs
  server_qps.reserve(config.num_server_nodes());
  client_qps.reserve(config.num_client_nodes());  // legal no-op if 0
}
```

`ClientConnectionManager` 只 reserve 不 resize，因为 `server_qps` / `client_qps` 要 `emplace_back`（不知道实际连成几个之前不能预填 nullptr）。`client_qps.reserve(0)` 是合法 no-op，注释专门点出——非 initiator 不需要 client_qps。

### 5.5.3 `ServerConnectionManager::connect_to_clients`：listen + accept 循环

```cpp
// rdma-library/library/connection_manager.cc:25-35
void ServerConnectionManager::connect_to_clients() {
  context_.bind_to_port(config_.port);

  // connect queue pairs and order them by client ids
  for (u32 i = 0; i < config_.num_clients; ++i) {
    auto [qp, client_id] = context_.wait_for_connection();
    client_qps[client_id] = std::move(qp);
  }

  context_.close_server_socket();
}
```

server 侧的连接循环：

1. `context_.bind_to_port(config_.port)`：在 `config_.port`（默认 1234）上 listen（`context.cc:102-124`：socket → SO_REUSEADDR → bind → listen，backlog 128）。
2. 循环 `num_clients` 次，每次 `context_.wait_for_connection()`：accept 一条 TCP 连接，在 TCP 上交换 `QPInfo`，把新 QP 推到 RTR/RTS，返回 `{QP, client_id}`（`context.cc:132-158`）。
3. `client_qps[client_id] = std::move(qp)`：**按对端自报的 client_id 填槽**，而不是按 accept 顺序。这保证 `client_qps[i]` 一定是 client_id=i 的那条 QP，与 `initiator_qp` 的引用约定一致。
4. 全部连完后 `close_server_socket()`：关掉 listen socket，不再接受新连接。

`wait_for_connection` 的实现（`context.cc:132-158`）是 TCP 握手的真正落点：

```cpp
// rdma-library/library/context.cc:132-158
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

注意：`send_buffer` 不带 `node_id`（用默认 0），`receive_buffer` 里的 `node_id` 是 client 自报的——这就是为什么 server 能按 `client_id` 填槽。`// TODO: set remote user data` 是个未完成点，未来可能扩展握手协议携带更多元数据。

### 5.5.4 `ClientConnectionManager::connect`：三步编排

```cpp
// rdma-library/library/connection_manager.cc:37-41
void ClientConnectionManager::connect() {
  connect_among_clients();
  distribute_client_ids();
  connect_to_servers();
}
```

三步：

1. **client 之间互连**（`connect_among_clients`）：initiator 连所有其他 client，其他 client listen 等 initiator。
2. **分发 client_id**（`distribute_client_ids`）：initiator 给每个 client 发编号 + 总数。
3. **连 server**（`connect_to_servers`）：每个 client 连所有 server。

#### 5.5.4.1 `connect_among_clients`

```cpp
// rdma-library/library/connection_manager.cc:43-61
void ClientConnectionManager::connect_among_clients() {
  if (is_initiator) {
    for (const str& node : config_.client_nodes) {
      const auto endpoint = parse_endpoint(node, config_.port);
      std::cerr << "connect to client " << endpoint.host << ":" << endpoint.port << std::endl;
      // clients act as "server" (they wait for a connection)
      client_qps.emplace_back(context_.connect_to_server(endpoint.address, endpoint.port));
    }

  } else {
    std::cerr << "connect to initiator" << std::endl;
    context_.bind_to_port(config_.port);

    // connect queue pair to initiator
    initiator_qp = context_.wait_for_connection().first;

    context_.close_server_socket();
  }
}
```

- initiator 路径：遍历 `config_.client_nodes`（其他 client 的地址列表），对每个调 `context_.connect_to_server(address, port)`。注释点明：“clients act as server”——非 initiator 的 client 在自己端口上 listen，所以 initiator 调 `connect_to_server` 反向连过去。这是个角色反转：`connect_to_server` 名字里的 “server” 指的是 TCP accept 侧，不是 dvstor 的 storage server。
- 非 initiator 路径：`bind_to_port` listen，`wait_for_connection().first` 接受一条连接（只接受一条，就是 initiator 来的），填进 `initiator_qp`。然后 close listen socket。

`connect_to_server` 的实现（`context.cc:160-195`）：

```cpp
// rdma-library/library/context.cc:160-195
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
  while (connect(tcp_socket, (sockaddr*)&remote_address, sizeof(sockaddr_in)) !=
         0) {
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

- `send_buffer` 带 `node_id`：client → server 方向会带上自己的编号。
- `while (connect(...) != 0)`：忙等重试 connect，直到对端 listen 起来。这是处理“client 启动比 server 早”的竞态——没有退避，纯忙等，简单粗暴但够用（局域网内启动时序差不会太大）。
- 握手顺序：先 send 后 recv（client 先发自报信息），对称地 `wait_for_connection` 是先 recv 后 send（server 先收）。两端顺序匹配。

#### 5.5.4.2 `distribute_client_ids`

```cpp
// rdma-library/library/connection_manager.cc:63-92
void ClientConnectionManager::distribute_client_ids() {
  // initiator distributes client ids, number of clients, and threshold
  if (is_initiator) {
    client_id = 0;
    num_total_clients = config_.num_client_nodes() + 1;  // incl. initiator

    for (u32 i = 0; i < config_.num_client_nodes(); ++i) {
      QP& qp = client_qps[i];
      u32 id = i + 1;
      vec<u64> content{id, num_total_clients};
      qp->post_send_inlined(content.data(), 2 * sizeof(u64), IBV_WR_SEND);
    }

    context_.poll_send_cq_until_completion(
      static_cast<i32>(config_.num_client_nodes()));

  } else {
    vec<u64> content(2);
    LocalMemoryRegion region{context_, content.data(), 2 * sizeof(u64)};
    initiator_qp->post_receive(region);
    context_.receive();

    // unpack content
    client_id = content[0];
    num_total_clients = content[1];
  }

  std::cerr << "client id: " << client_id << std::endl;
  std::cerr << "number of clients: " << num_total_clients << std::endl;
}
```

initiator 分发编号：

- initiator 自报 `client_id = 0`，`num_total_clients = num_client_nodes() + 1`（其他 client 数 + 自己）。
- 对每条已连通的 client QP，发 16 字节（`id, num_total_clients`），用 `post_send_inlined`（inline + SEND，无 MR）。
- `poll_send_cq_until_completion(num_client_nodes())`：等所有 SEND 完成（5.3.5 提过，`sq_sig_all=0` 但默认 `signaled=true`，所以每条都出 CQE，这里 poll 直到拿到 N 个）。

非 initiator 收编号：

- 栈上 `vec<u64> content(2)`，临时注册成 MR。
- `initiator_qp->post_receive(region)` + `context_.receive()`：同步收。
- 解包：`client_id = content[0]`，`num_total_clients = content[1]`。

**注意**：这一步只发 `id` 和 `num_total_clients` 两个 u64，没有发 threshold——注释里写“and threshold”但代码没发。可能是历史遗留或未来扩展点。

#### 5.5.4.3 `connect_to_servers`

```cpp
// rdma-library/library/connection_manager.cc:94-100
void ClientConnectionManager::connect_to_servers() {
  for (const str& node : config_.server_nodes) {
    const auto endpoint = parse_endpoint(node, config_.port);
    std::cerr << "connect to server " << endpoint.host << ":" << endpoint.port << std::endl;
    server_qps.emplace_back(context_.connect_to_server(endpoint.address, endpoint.port, client_id));
  }
}
```

每个 client 连所有 server：

- 遍历 `config_.server_nodes`。
- `connect_to_server(address, port, client_id)`：注意第三个参数 `client_id`——这就是 `QPInfo::node_id` 的来源，server 端 `wait_for_connection` 拿到后用来按 id 填槽 `client_qps[client_id]`。
- `emplace_back` 进 `server_qps`：`server_qps` 是计算节点持连所有存储节点的 QP 集合。

#### 5.5.4.4 `ClientConnectionManager::synchronize` 与 `ServerConnectionManager::synchronize`

```cpp
// rdma-library/library/connection_manager.cc:102-115
bool ClientConnectionManager::synchronize() const {
  bool success = true;

  for (const QP& qp : server_qps) {
    bool ready{};
    LocalMemoryRegion region(context_, &ready, sizeof(bool));
    qp->post_receive(region);
    context_.receive();

    success &= ready;
  }

  return success;
}

void ServerConnectionManager::synchronize() const {
  constexpr bool ready = true;

  for (const QP& qp : client_qps) {
    qp->post_send_inlined(&ready, sizeof(bool), IBV_WR_SEND);
  }

  context_.poll_send_cq_until_completion(static_cast<i32>(client_qps.size()));
}
```

一对配套的同步屏障：

- server 侧 `synchronize()`：向所有 client 广播一个 `true`（1 字节 bool，inline SEND）。
- client 侧 `synchronize()`：从所有 server 各收一个 `bool`，AND 起来返回。

语义：所有 server 都“ready”之后，所有 client 才会从 `synchronize()` 返回。这是“所有节点就绪后再进入下一阶段”的经典 barrier。`success` 变量预留了“未来 server 可能回报 false”的扩展（比如某个 server 初始化失败），目前 server 总发 `true`。

### 5.5.5 整体拓扑

把 `connect()` 三步和 `connect_to_clients` 拼起来，整个集群的连接拓扑是：

```
            ┌─────────────── initiator (client_id=0) ───────────────┐
            │   connect_among_clients: 主动 connect 每个 other client │
            │   distribute_client_ids:  给每个 other client 发 (id,N) │
            └────────────────────────────────────────────────────────┘
                                  │ QP (client↔client)
                                  ▼
        ┌─────────── other client (client_id=1..N-1) ───────────┐
        │ connect_among_clients:  listen 等 initiator            │
        │ distribute_client_ids:  从 initiator 收 (id,N)          │
        └────────────────────────────────────────────────────────┘
                                  │
                                  │ connect_to_servers: 每个 client
                                  │ 连所有 server（带 client_id）
                                  ▼
        ┌─────────────────── storage server ────────────────────┐
        │ connect_to_clients:  listen, accept num_clients 条连接 │
        │ 按 client_id 填 client_qps[id]                          │
        │ initiator_qp == client_qps[0]                          │
        └────────────────────────────────────────────────────────┘
```

关键点：

- **client↔client** 只有 initiator 与其他 client 之间的星形连接（其他 client 之间不互连）。
- **client↔server** 是全连接：每个 client 连每个 server。
- **server↔server** 在这一层不连——server 之间的 peer RDMA 在第 23 课的 `memory_node` 里另起一套（用 `peer_remote_tokens_`，见 `memory_node.hh:585`）。

## 5.6 `batched_read.hh`：CPU 侧批量 RDMA read

```cpp
// rdma-library/library/batched_read.hh:7-19
struct BatchedREAD {
  const u32 max_size;
  u32 requests{0};
  u64 total_size{0};

  vec<ibv_send_wr> work_requests;
  vec<ibv_sge> scatter_gather_entries;
  ibv_send_wr* bad_work_request{nullptr};

  explicit BatchedREAD(size_t max_batch_size)
      : max_size(max_batch_size),
        work_requests(max_batch_size),
        scatter_gather_entries(max_batch_size) {}
```

`BatchedREAD` 是个纯结构体（不是类），意图简单：预分配 `max_batch_size` 条 WR 和 SGE，逐条填，最后一次性 `ibv_post_send` 链式下发。

- `max_size`：批量上限（const）。
- `requests`：当前已添加条数。
- `total_size`：累计字节数（统计用）。
- `work_requests` / `scatter_gather_entries`：两个 vector 预分配，避免每条 WR 现场分配。
- `bad_work_request`：`ibv_post_send` 失败时的输出指针。

### 5.6.1 `add_to_batch`

```cpp
// rdma-library/library/batched_read.hh:21-54
  void add_to_batch(u64 local_address,
                    u64 remote_address,
                    u32 length,
                    u32 lkey,
                    u32 rkey,
                    u64 wr_id,
                    bool signaled = true) {
    lib_assert(length > 0, "Cannot READ 0 bytes");
    lib_assert(requests < max_size, "Batch exceeds maximum batch size");

    auto& sge = scatter_gather_entries[requests];
    auto& wr = work_requests[requests];

    sge.addr = local_address;
    sge.length = length;
    sge.lkey = lkey;

    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    wr.next = nullptr;
    if (requests > 0) {
      work_requests[requests - 1].next = &wr;
    }

    wr.wr.rdma.remote_addr = remote_address;
    wr.wr.rdma.rkey = rkey;

    ++requests;
    total_size += length;
  }
```

逐条添加：

- 两个断言：长度>0、不超批量上限。
- 取当前槽位的 SGE 和 WR 引用（`auto&`，避免拷贝）。
- SGE：`local_address` / `length` / `lkey`。
- WR：
  - `opcode = IBV_WR_RDMA_READ`。
  - `send_flags`：signaled 控制（默认 true）。
  - `wr_id`：完成时回传。
  - `sg_list` / `num_sge`：单 SGE。
  - `next = nullptr`：先把当前条设为链尾。
  - **链表回填**：`if (requests > 0) work_requests[requests-1].next = &wr`——把上一条的 `next` 指向当前条，串成链。
  - `wr.rdma.remote_addr` / `rkey`：远端地址和 key。
- `++requests`，`total_size += length`。

关键设计：**用 `vec` 预分配保证 SGE/WR 地址稳定**。如果用 `std::vector` 边 push_back 边取地址，reallocation 会让之前填好的 `next` 指针失效。这里构造期 reserve 固定大小，之后只改内容不增删，指针稳定。

### 5.6.2 `post_batch`

```cpp
// rdma-library/library/batched_read.hh:56-70
  void post_batch(QP& qp) {
    lib_assert(requests > 0, "Empty READ batch");

    // send final WR signaled in any case
    work_requests[requests - 1].send_flags = IBV_SEND_SIGNALED;

    lib_assert(
      ibv_post_send(
        qp->get_ibv_qp(), work_requests.data(), &bad_work_request) == 0,
      "Cannot post send request");

    // reset batch
    requests = 0;
    total_size = 0;
  }
};
```

下发：

- 断言非空。
- **强制最后一条 signaled**：即使前面有些条没 signaled，最后一条必须 signaled。原因：verbs 保证 signaled WR 完成时，**同 QP 上之前所有未 signaled 的 WR 也都已完成**。所以只需 poll 最后一条的 CQE，就能确认整批完成。这是 RDMA 批量下发的标准优化——大幅减少 CQE 数量。
- `ibv_post_send(qp, work_requests.data(), &bad)`：传链头，verbs 沿 `next` 链遍历一次性 post 所有 WR。
- reset：`requests=0`，`total_size=0`，复用 `BatchedREAD` 对象。

> **与第 19 课的区别**：`BatchedREAD` 是 CPU 侧 verbs 的批合并——把多条 `IBV_WR_RDMA_READ` 通过 WR 链一次 post。第 19 课讲的 `rdma_cache.cuh` 是 GPU 侧的合并——在 GPU kernel 里把多个 read 请求合并成更少的 GPUNetIO 请求。两条路径独立，互不替代：CPU 侧批合并减少 `ibv_post_send` 调用次数，GPU 侧批合并减少 kernel↔host 的同步开销。

## 5.7 关键数据结构与流程图

### 5.7.1 数据结构速查

| 结构 | 字段 | 含义 |
| --- | --- | --- |
| `QPInfo` | `lid, qp_number, node_id` | TCP 握手交换的三元组 |
| `MemoryRegionToken` | `address, lkey, rkey` | 远端访问本地 MR 的三元组（第 4 课） |
| `ibv_qp_init_attr` | `send_cq, recv_cq, srq, cap, qp_type, sq_sig_all` | QP 创建属性 |
| `ibv_qp_attr` | `qp_state, path_mtu, dest_qp_num, rq_psn, ...` | QP 修改属性（状态机用） |
| `ibv_send_wr` | `opcode, send_flags, wr_id, next, sg_list, num_sge, wr.rdma/wr.atomic` | 发送 WR |
| `ibv_recv_wr` | `wr_id, next, sg_list, num_sge` | 接收 WR |
| `ibv_sge` | `addr, length, lkey` | Scatter/Gather 元素 |
| `QueuePair` | `context_, lid_, use_shared_receive_cq_, queue_pair_, max_send_wr_` | verbs QP 封装 |
| `DetachedQP` | `qp, send_cq_, recv_cq_, owns_cqs_` | 独立 QP + 可选自带 CQ |
| `ConnectionManager` | `context_, config_` | 基类 |
| `ServerConnectionManager` | `client_qps, initiator_qp` | 存储侧，按 client_id 寻址 |
| `ClientConnectionManager` | `is_initiator, client_id, num_total_clients, server_qps, client_qps, initiator_qp` | 计算侧 |
| `BatchedREAD` | `max_size, requests, total_size, work_requests, scatter_gather_entries` | CPU 侧批量 read |

### 5.7.2 QP 状态机流程图

```
   ibv_create_qp
        │
        ▼
     RESET
        │  transition_to_init()
        │  设置 port / pkey / access_flags
        ▼
      INIT  ← 可 post_recv，但还不能收/发
        │  transition_to_rtr(remote)
        │  设置 dest_qp_num / path_mtu / rq_psn /
        │       max_dest_rd_atomic / min_rnr_timer / ah_attr
        ▼
      RTR   ← 可收
        │  transition_to_rts()
        │  设置 timeout / retry_cnt / rnr_retry /
        │       sq_psn / max_rd_atomic
        ▼
      RTS   ← 可发可收
        │
        ▼
   post_send / post_recv / post_CAS / post_FAA
        │
        ▼
   poll_send_cq / poll_recv_cq
        │
        ▼
   ibv_destroy_qp（QueuePair 析构）
```

### 5.7.3 连接建立时序图（client → server 方向）

```
   Client                                    Server
   │                                          │
   │  bind_to_port(port)  ◄───────────────────┤  (server 先 listen)
   │                                          │
   │  make_unique<QueuePair>(context)         │
   │   → ibv_create_qp → RESET → INIT         │
   │   → qp_num_c 已就绪                       │
   │                                          │
   │  TCP connect ───────────────────────────►│  accept
   │                                          │  make_unique<QueuePair>(context)
   │                                          │   → ibv_create_qp → RESET → INIT
   │                                          │   → qp_num_s 已就绪
   │                                          │
   │  send(QPInfo{lid_c, qp_num_c, node_id})─►│
   │                                          │
   │◄──────────────── recv(QPInfo{lid_s, qp_num_s, 0})
   │                                          │
   │  transition_to_rtr(remote={lid_s, qp_num_s})
   │  transition_to_rts()
   │                                          │  transition_to_rtr(remote={lid_c, qp_num_c})
   │                                          │  transition_to_rts()
   │                                          │
   │  close(tcp_socket)                       │  close(tcp_socket)
   │                                          │
   │  QP 已就绪，可 post_send / post_recv     │
   │                                          │
```

`DetachedQP::connect` 用已连通 QP 当信道的时序：

```
   端 A (新 QP)               端 B (新 QP)
   │                          │
   │  已有一条 other_qp 连通   │
   │  ◄════════════════════►  │
   │                          │
   │  other_qp.post_recv      │  other_qp.post_recv
   │  other_qp.post_send(     │  other_qp.post_send(
   │    {lid_a, qp_num_a})    │    {lid_b, qp_num_b})
   │                          │
   │  poll_send_cq_until_done │  poll_send_cq_until_done
   │  context.receive()       │  context.receive()
   │                          │
   │  收到 {lid_b, qp_num_b}  │  收到 {lid_a, qp_num_a}
   │                          │
   │  new_qp.rtr(remote_b)    │  new_qp.rtr(remote_a)
   │  new_qp.rts()            │  new_qp.rts()
   │                          │
   │  new_qp 已就绪            │  new_qp 已就绪
```

## 5.8 与其他模块的关系

本课的 QP/ConnectionManager 是整个 dvstor RDMA 通信的 CPU 侧底座，被以下模块复用：

- **第 4 课（RDMA 库上）**：`Context` 提供 CQ/PD/SRQ 和 `wait_for_connection`/`connect_to_server` 的 TCP 落点；`MemoryRegion` 提供 `MemoryRegionToken`。本课在它们之上建 QP 和连接编排。
- **第 9 课（GPU 类型/遥测）**：遥测计数器会统计 QP 的 outstanding read、CQE 吞吐等。本课的 `max_rd_atomic` / `max_dest_rd_atomic` 是这些计数器的硬上限来源。
- **第 11 课（持久化引擎 PImpl/生命周期）**：计算节点的持久化引擎在初始化时通过 `ClientConnectionManager` 拿到所有 `server_qps`，再用这些 QP 发起 RDMA read/write。引擎对象持有 `cm_` 引用。
- **第 14 课（查询执行/路由/完成）**：查询路由需要把请求送到正确的 server QP，完成路径需要 poll send/recv CQE。`Context::poll_send_cq` / `poll_recv_cq` 是热路径。
- **第 16 课（存储回收 RCU）**：RCU 的引用计数和无锁回收依赖 `post_CAS` / `post_FAA` 这两个原子原语。
- **第 19 课（RDMA cache）**：GPU 侧的 read 合并路径。与 `BatchedREAD` 是两条独立路径（5.6 节已对比）。
- **第 22 课（GPUNetIO 传输/probe）**：GPU 侧的 QP 用 DOCA GPUNetIO API，不走本课的 `ibv_create_qp`。但 GPUNetIO transport 的初始化有时会借道 CPU QP 交换元数据（类似 `DetachedQP::connect` 的模式）。本课是 CPU verbs QP，第 22 课是 GPU GPUNetIO QP，两者并存。
- **第 23 课（存储节点主体/peer RDMA）**：存储节点用 `ServerConnectionManager cm_` 接受所有计算节点的连接（`memory_node.hh:562`），同时维护 `MemoryRegionTokens peer_remote_tokens_`（`memory_node.hh:585`）用于 server↔server 的 peer RDMA——后者不走 `ConnectionManager`，是另一套独立的 QP 配对（用 `DetachedQP` 或类似机制）。
- **第 27 课（计算服务主体）**：`compute_service.hh:185` 的 `ClientConnectionManager cm_` 是计算节点连接所有存储节点的入口。
- **第 28 课（计算侧 storage owner 更新）**：owner map 的更新依赖 `cm_.server_qps` 把更新请求送到对应 server。
- **第 30 课（breakdown benchmark）**：实验脚本会配置 `server_nodes` / `client_nodes` / `is_initiator` / `num_clients` 等，触发 `connect()` 三步编排。

### 复用要点

计算节点和存储节点复用同一套 `ConnectionManager` 代码，区别只在角色：

- 计算节点：`ClientConnectionManager`，`is_initiator` 由配置决定，`connect()` 三步走完拿到 `server_qps`。
- 存储节点：`ServerConnectionManager`，`connect_to_clients()` listen + accept 所有 client，拿到 `client_qps` 按 `client_id` 寻址。

`QueuePair` 本身角色无关——它就是个 verbs 句柄的 RAII 封装，谁都能用。`DetachedQP` 给需要独立 CQ 的场景留口子（如 peer RDMA、测试探针）。

## 5.9 小结

本课覆盖了 dvstor CPU 侧 RDMA 传输库的“通道层”：

1. **`QueuePair`** 把 `ibv_qp*` 包成 RAII 类，构造即 `RESET → INIT`，`transition_to_rtr/rtk` 完成状态机最后两跳。PSN=0、timeout=14、retry_cnt=7、rnr_retry=7、min_rnr_timer=12、MTU=4096、access flags 全开、`sq_sig_all=0` 这些参数是 dvstor 在自己集群上的稳定默认值。
2. **post_send / post_recv / post_CAS / post_FAA** 统一通过 SGE+WR 组织，inline/signaled 两个标志控制零拷贝和 CQE 产出。五个 send 重载收敛到一个私有 10 参数实现，避免重复。
3. **`DetachedQP`** 提供“独立 CQ”变体，析构顺序严格遵守 QP→CQ。它的 `connect` 方法复用已连通 QP 当信道，避免重复 TCP 握手。
4. **`ConnectionManager`** 把 TCP 握手 + QP pairing 编排成 `ServerConnectionManager`（listen + accept，按 `client_id` 填槽）和 `ClientConnectionManager`（三步：client 互连 → 分发 id → 连 server）两个角色。`QPInfo{lid, qp_number, node_id}` 是 TCP 上的握手载荷，`node_id` 让 server 能按对端自报编号寻址。
5. **`BatchedREAD`** 是 CPU 侧 verbs 的批量 read 合并工具，用 WR 链 + “最后一条 signaled” 减少调用次数和 CQE 数量。
6. 整套库被计算节点（`ClientConnectionManager`）和存储节点（`ServerConnectionManager`）对称复用，是 dvstor 存算分离架构的 CPU 侧通信底座。GPU 侧的 GPUNetIO QP（第 22 课）是另一条独立路径，与本文不冲突。

下一课（第 6 课）将离开传输层，进入索引格式：Vamana 图格式与 anchor/idmap。
