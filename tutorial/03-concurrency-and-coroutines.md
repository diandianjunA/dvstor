# 第 3 课 并发原语与协程

> 本课目标：把 dvstor 在 CPU 侧用到的"基础设施级"并发原语一次性讲透。包括原子工具、Vyukov 有界 MPMC 队列、两方完成池、滑动完成环、CPU 核心亲和性分配、C++20 协程桥接、远端指针编码，以及计时工具。它们既被存储节点的 runtime 使用（见第 21、23、26 课），也被计算服务侧的同步 API 使用（见第 27、28 课），是后续课程反复出现的"积木"。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `src/common/atomic_utils.hh` | 原子操作小工具：relaxed 比较-交换更新最大值、RAII 计数减一守卫 |
| `src/common/bounded_queue.hh` | 预分配 Vyukov 风格有界 MPMC 队列（含 `push_wait`/`pop_wait`/`notify_all`） |
| `src/common/completion_pool.hh` | 两方完成池（producer/consumer 各持一引用），用于同步 RPC 等 |
| `src/common/sliding_completion_ring.hh` | 滑动完成环：有序预留、乱序完成、前缀连续推进 |
| `src/common/core_assignment.hh` + `.cc` | CPU 核心亲和性分配（strict / interleaved 两种策略） |
| `src/common/core_partition.hh` | 把已拓扑排序的 CPU 序列在多个同机进程间划分 |
| `src/coroutine.hh` | C++20 协程返回对象与 promise_type（`MinorCoroutine`/`VamanaCoroutine`/`StorageOwnerInsertCoroutine`） |
| `src/remote_pointer.hh` | 远端指针 `RemotePtr`：16b 节点号 + 48b 字节偏移，以及哈希特化 |
| `src/common/timing.hh` + `.cc` | `clock_gettime(CLOCK_MONOTONIC)` 累计式计时器，支持 json 序列化 |

## 0. 总体定位

dvstor 是 GPU 中心化的存算分离系统：存储节点持有图索引与向量数据，GPU 在计算节点上做检索。为了让 GPU 不被 CPU 侧的锁/IO 阻塞，CPU 侧的并发原语必须满足三个特征：

1. **预分配、无热路径分配**：队列/池在构造期一次性 `new`，热路径只做原子操作。
2. **避免 ABA 与重用竞争**：完成池用"双方各持一引用"防止超时消费者把还在用的槽位还回 free list。
3. **NUMA 与超线程感知**：NIC 在 NUMA node 1，存储线程优先绑定到 node 1 的物理核。

本课的所有文件都在围绕这三点做工程化。下面逐文件讲解。

---

## 1. `atomic_utils.hh`：原子工具

文件：`src/common/atomic_utils.hh:1-30`

整个文件只有两个工具，但都是高频被复用的"片段"，所以要单独封装。

### 1.1 `update_max_relaxed`

```cpp
template <class T>
void update_max_relaxed(std::atomic<T>& target, T value) noexcept {
  T observed = target.load(std::memory_order_relaxed);
  while (observed < value &&
         !target.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}
```

- 这是一段经典的"无锁更新最大值"循环。`compare_exchange_weak` 在失败时会把当前值写回 `observed`，所以循环里不需要重新 `load`。
- 全程使用 `memory_order_relaxed`：因为这只是**统计/遥测**用途（例如延迟分位的最大值），既不参与发布-订阅，也不保护任何非原子数据，所以不需要 acquire/release。relaxed 既足够正确，又最便宜。
- "observed < value"作为短路：一旦发现当前最大值已经不小于 `value`，直接退出，不做无谓的 CAS。
- 注意签名是 `noexcept` 但不是 `[[nodiscard]]`，因为它没有有意义的返回值——成功与否不影响调用方语义。

### 1.2 `CounterDecrementGuard`

```cpp
template <class T>
class CounterDecrementGuard {
public:
  explicit CounterDecrementGuard(std::atomic<T>& counter) : counter_(counter) {}
  ~CounterDecrementGuard() {
    counter_.fetch_sub(1, std::memory_order_acq_rel);
  }
  ...
};
```

- RAII：构造时不增不减，析构时 `fetch_sub(1)`。这个语义有点反直觉（不是"加一构造/减一析构"的 scope guard，而是"我承诺会在作用域结束时把这个计数器减一"），所以它适合这种用法：
  ```cpp
  inflight_.fetch_add(1, std::memory_order_release);
  atomic_utils::CounterDecrementGuard<u32> guard{inflight_};
  // ... 中间可能 return / throw ...
  ```
  保证无论从哪个出口离开作用域，`inflight_` 都会被减回去。
- 用 `acq_rel` 而不是 `relaxed`：因为这种"in-flight 计数"通常配合"等所有 in-flight 工作完成"的检查使用——观察者 `load(acquire)` 看到 0 时，要能看到被保护工作的全部写。所以减一操作必须 release；同时它也要 acquire，因为减一发生前可能读过被保护数据，这等于"获取"该数据上的发布。

两个工具都很短，但它们在整个项目里被用作"标准化片段"，避免每个模块自己手写 CAS 循环或 RAII 减计数。

---

## 2. `bounded_queue.hh`：Vyukov 有界 MPMC 队列

文件：`src/common/bounded_queue.hh:1-165`

这是项目里最核心的 CPU 并发原语之一。它的设计目标是：

- **预分配**：构造后 `push`/`pop` 都不再 `new`。
- **MPMC 安全**：多生产者多消费者都能用；在更新 runtime 里也常被降级当 MPSC/SPSC 用（注释 `bounded_queue.hh:17-20`）。
- **支持阻塞 + 唤醒**：除了 `try_push`/`try_pop`，还有 `push_wait`/`pop_wait`，用 C++20 的 `atomic::wait`/`notify_one` 实现低开销阻塞。
- **可观测停机**：`notify_all()` 把所有阻塞的 producer/consumer 唤醒，配合外部 stop 标志优雅退出。

### 2.1 数据布局

```cpp
struct alignas(kCacheLineBytes) Cell {
  std::atomic<u64> sequence{};
  T value{};
};
...
const size_t capacity_;
const size_t mask_;
std::unique_ptr<Cell[]> cells_;
alignas(kCacheLineBytes) std::atomic<u64> enqueue_position_{0};
alignas(kCacheLineBytes) std::atomic<u64> dequeue_position_{0};
alignas(kCacheLineBytes) std::atomic<u64> push_epoch_{0};
alignas(kCacheLineBytes) std::atomic<u64> pop_epoch_{0};
```

- `kCacheLineBytes = 64`（见 `src/common/constants.hh:7`），每个 `Cell` 与每个位置计数器都按 64 字节对齐，避免 false sharing。
- `capacity_` 在构造时被 `normalize_capacity` 向上取整为 2 的幂（`bounded_queue.hh:150-154`），所以 `mask_ = capacity_ - 1`，槽位下标可以用 `position & mask_` 直接计算，省一次除法。
- 经典的 Vyukov 三件套：每个槽位自带一个 `sequence`，而不是用"头尾 mod 容量"判断空满。这把空/满判断从两个全局计数器转移到了槽位本地，CAS 失败的代价更小。
- `enqueue_position_`/`dequeue_position_` 是单调递增的全局游标，**永不回卷**，回卷只发生在 `position & mask_` 取槽位时。这避免了 ABA。
- `push_epoch_`/`pop_epoch_` 不参与空满判断，只服务于 `wait`/`notify`：每次成功 push/pop 都 `fetch_add(1)` 后 `notify_one`，让阻塞的对端有机会醒来。

### 2.2 构造与初始化

```cpp
explicit Queue(size_t requested_capacity)
    : capacity_(normalize_capacity(requested_capacity)),
      mask_(capacity_ - 1),
      cells_(std::make_unique<Cell[]>(capacity_)) {
  for (u64 index = 0; index < capacity_; ++index) {
    cells_[index].sequence.store(index, std::memory_order_relaxed);
  }
}
```

- 初始化时第 `index` 个槽的 `sequence = index`。这正是 Vyukov 的初始不变量：`enqueue_position = 0` 时，cell[0].sequence = 0，与 `enqueue_position` 相等，所以 `difference = 0`，第一个 push 能直接命中 cell[0]。
- 初始化用 `relaxed` 是安全的：构造期没有并发访问。
- `normalize_capacity`：
  ```cpp
  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 62)) return size_t{1} << 62;
    return std::bit_ceil(requested);
  }
  ```
  最小 2（保证有意义的 MPMC），最大 2^62（防止溢出 `u64` 的算术），中间用 `std::bit_ceil` 向上取 2 的幂。

### 2.3 入队：`emplace`

```cpp
template <class U>
bool emplace(U&& value) {
  u64 position = enqueue_position_.load(std::memory_order_relaxed);
  Cell* cell = nullptr;
  for (;;) {
    cell = &cells_[static_cast<size_t>(position) & mask_];
    const u64 sequence = cell->sequence.load(std::memory_order_acquire);
    const i64 difference = static_cast<i64>(sequence - position);
    if (difference == 0) {
      if (enqueue_position_.compare_exchange_weak(
            position, position + 1,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
        break;
      }
    } else if (difference < 0) {
      return false;
    } else {
      position = enqueue_position_.load(std::memory_order_relaxed);
    }
  }

  cell->value = std::forward<U>(value);
  cell->sequence.store(position + 1, std::memory_order_release);
  push_epoch_.fetch_add(1, std::memory_order_release);
  push_epoch_.notify_one();
  return true;
}
```

逐行解读这段是本课的重点：

1. `position = enqueue_position_.load(relaxed)`：拿到当前尾游标。relaxed 足够，因为下面要 CAS，CAS 失败会自动重新读取。
2. `cell->sequence.load(acquire)`：读槽位的"期望生产者位置"。acquire 是为了同步消费端 release 写入的 `sequence = position + capacity_`（见 2.4）——也就是说，只有当上一个消费者把该槽"归还"为可写状态后，我们才能看到 `sequence` 等于当前 `position`。
3. `difference = sequence - position`：
   - `== 0`：槽正好等着这个 `position` 写入，可以抢。
   - `< 0`：槽的 sequence 落后于 `position`，意味着队列已满（消费者还没把槽还回来）。返回 false。
   - `> 0`：槽的 sequence 跑到 `position` 前面了，说明我们拿到的 `position` 已经过期（其它生产者已经推进了 enqueue_position_），重新 load。
4. CAS 成功后，**先写 value 再更新 sequence**：
   ```cpp
   cell->value = std::forward<U>(value);
   cell->sequence.store(position + 1, std::memory_order_release);
   ```
   顺序至关重要：消费者是先看到 `sequence` 变成 `position + 1`，再去读 `value` 的。release 保证消费者 acquire 看到 sequence 时，也能看到我们写入的 `value`。
5. 最后 `push_epoch_.fetch_add(1, release)` + `notify_one()` 唤醒一个阻塞的消费者。

### 2.4 出队：`try_pop`

```cpp
bool try_pop(T& value) {
  u64 position = dequeue_position_.load(std::memory_order_relaxed);
  Cell* cell = nullptr;
  for (;;) {
    cell = &cells_[static_cast<size_t>(position) & mask_];
    const u64 sequence = cell->sequence.load(std::memory_order_acquire);
    const i64 difference = static_cast<i64>(sequence - (position + 1));
    if (difference == 0) {
      if (dequeue_position_.compare_exchange_weak(
            position, position + 1,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
        break;
      }
    } else if (difference < 0) {
      return false;
    } else {
      position = dequeue_position_.load(std::memory_order_relaxed);
    }
  }

  value = std::move(cell->value);
  cell->sequence.store(position + capacity_, std::memory_order_release);
  pop_epoch_.fetch_add(1, std::memory_order_release);
  pop_epoch_.notify_one();
  return true;
}
```

注意 `difference = sequence - (position + 1)`：消费者期望的 sequence 是 `position + 1`，与生产者写入的 `position + 1` 对齐。出队成功后写 `sequence = position + capacity_`，这把槽位标记为"等待第 `position + capacity_` 个生产者写入"，正好对应下一轮复用。这是 Vyukov 队列的核心技巧：通过 sequence 的递增实现自然回卷，不需要额外的"满/空"标志。

### 2.5 阻塞与唤醒：`push_wait` / `pop_wait`

```cpp
template <class U>
void push_wait(U&& value) {
  for (;;) {
    if (emplace(std::forward<U>(value))) return;
    const u64 observed = pop_epoch_.load(std::memory_order_acquire);
    if (emplace(std::forward<U>(value))) return;
    pop_epoch_.wait(observed, std::memory_order_relaxed);
  }
}
```

经典的"双重检查 wait"模式：
1. 先尝试 emplace，成功就走。
2. 失败后记录当前 `pop_epoch_`（消费者每次成功 pop 都会 +1）。
3. **再试一次**：因为在 load epoch 和 wait 之间，消费者可能已经 pop 了若干个。如果不重试，就会错过这期间的 notify。
4. 还没成功就 `wait`，等被唤醒。

`pop_wait` 对称：
```cpp
void pop_wait(T& value) {
  for (;;) {
    if (try_pop(value)) return;
    const u64 observed = push_epoch_.load(std::memory_order_acquire);
    if (try_pop(value)) return;
    push_epoch_.wait(observed, std::memory_order_relaxed);
  }
}
```

还有一个带 stop 标志的版本：
```cpp
bool pop_wait(T& value, const std::atomic<bool>& stop) {
  for (;;) {
    if (try_pop(value)) return true;
    if (stop.load(std::memory_order_acquire)) return false;
    const u64 observed = push_epoch_.load(std::memory_order_acquire);
    if (try_pop(value)) return true;
    if (stop.load(std::memory_order_acquire)) return false;
    push_epoch_.wait(observed, std::memory_order_relaxed);
  }
}
```
这是给 worker 线程主循环用的：每次 wait 前后都检查 stop 标志，确保停机时不会卡死在 wait 上。但 `wait` 本身可能阻塞，所以还需要 `notify_all` 在停机时唤醒：

```cpp
void notify_all() noexcept {
  push_epoch_.fetch_add(1, std::memory_order_release);
  pop_epoch_.fetch_add(1, std::memory_order_release);
  push_epoch_.notify_all();
  pop_epoch_.notify_all();
}
```

`fetch_add(1)` 改变 epoch 值，使所有 `wait(observed)` 中 `observed` 不再等于当前值，从而全部返回——这是 `std::atomic::wait` 的标准唤醒手法。

### 2.6 大小观测

```cpp
[[nodiscard]] size_t approximate_size() const noexcept {
  const u64 pushed = enqueue_position_.load(std::memory_order_acquire);
  const u64 popped = dequeue_position_.load(std::memory_order_acquire);
  return static_cast<size_t>(std::min<u64>(capacity_, pushed - popped));
}
```

注意函数名是 **approximate**：两次独立 load 之间，生产者/消费者都可能推进，所以 `pushed - popped` 可能瞬时为负或偏大，最后用 `min(capacity_, ...)` 钳制。这只能用作遥测，不能用于并发决策。这个函数在 `src/memory_node/storage_owner_runtime/lifecycle.cc:28` 等地方被用来构造 `bounded::Queue<StorageOwnerInsertTask>`，给 storage owner 的 worker 池做任务队列（见第 21 课）。

---

## 3. `completion_pool.hh`：两方完成池

文件：`src/common/completion_pool.hh:1-129`

这个组件解决一个具体问题：**同步 RPC**。调用方线程发起一个写请求后要阻塞等结果，但响应可能从另一个线程（响应执行器）回来。需要一个"完成回调汇聚点"。

最朴素的实现是"每请求 new 一个 promise/future"，但 dvstor 要求热路径无分配。`CompletionPool` 给出一套预分配方案。

### 3.1 设计核心：两方引用计数

```cpp
struct Cell {
  std::atomic<u32> state{static_cast<u32>(Result::pending)};
  std::atomic<u32> references{0};
};
```

注释（`completion_pool.hh:13-16`）说明了为什么是"两方"：
> A producer (the response executor) and a consumer (the synchronous public API) each own one reference. A timed out/abandoned consumer therefore cannot create an ABA reuse while an RPC is still in flight.

每个完成槽有两个持有者：
- **consumer**：调用 `wait()` 的同步 API 线程。
- **producer**：响应执行器，调用 `complete()` 写入结果。

只有当两方都"放手"后，槽位才能被回收到 free 队列。这避免了"消费者超时返回，槽位被回收并复用，而生产者的响应还在路上写老槽位"这种 ABA。

### 3.2 构造

```cpp
explicit CompletionPool(u32 capacity)
    : capacity_(capacity),
      cells_(std::make_unique<Cell[]>(capacity_)),
      free_(capacity_) {
  if (capacity_ == 0) {
    throw std::invalid_argument("completion pool capacity must be positive");
  }
  for (u32 id = 0; id < capacity_; ++id) {
    if (!free_.try_push(id)) {
      throw std::runtime_error("failed to initialize completion pool");
    }
  }
}
```

- 所有 cell 预分配。
- `free_` 是上一节的 `bounded::Queue<u32>`，容量与 pool 相同，初始化时把所有 id 入队。这复用了一整套无锁基础设施。
- 注意 `free_` 容量 = `capacity_`：因为同一时刻一个 id 最多在 free 队列里出现一次，所以不需要更大的队列。

### 3.3 申请与准备：`acquire` / `try_acquire` / `prepare`

```cpp
u32 acquire() {
  u32 id = 0;
  free_.pop_wait(id);
  prepare(id);
  return id;
}

bool try_acquire(u32& id) {
  if (!free_.try_pop(id)) return false;
  prepare(id);
  return true;
}

void prepare(u32 id) {
  Cell& cell = cells_[id];
  cell.state.store(static_cast<u32>(Result::pending),
                   std::memory_order_relaxed);
  cell.references.store(2, std::memory_order_release);
}
```

- `acquire` 阻塞地从 free 队列拿一个 id；`try_acquire` 是非阻塞版本。
- `prepare` 重置 cell：
  - `state = pending`（relaxed 足够，因为 `references.store(release)` 是这条路径上后续操作的同步点）。
  - `references = 2`：消费方和生产方各占一引用。**release** 在这里发布"这个槽位已经准备好被等待和完成了"。

### 3.4 等待与完成

```cpp
Result wait(u32 id) const {
  validate(id);
  auto& state = cells_[id].state;
  u32 observed = state.load(std::memory_order_acquire);
  while (observed == static_cast<u32>(Result::pending)) {
    state.wait(observed, std::memory_order_relaxed);
    observed = state.load(std::memory_order_acquire);
  }
  return static_cast<Result>(observed);
}

void complete(u32 id, bool success) {
  validate(id);
  Cell& cell = cells_[id];
  const u32 desired = static_cast<u32>(
    success ? Result::success : Result::failure);
  u32 expected = static_cast<u32>(Result::pending);
  if (!cell.state.compare_exchange_strong(
        expected, desired, std::memory_order_release,
        std::memory_order_acquire)) {
    throw std::logic_error("completion cell completed more than once");
  }
  cell.state.notify_all();
  release_reference(id);
}
```

- `wait` 是标准的"双重检查 wait"：load(acquire) → 不满足则 wait → 再 load。acquire 保证看到 `complete` 里 release 写入的 state。
- `complete` 用 CAS 把 `pending` 改成 `success`/`failure`：如果 CAS 失败说明这个槽被完成过两次，是逻辑错误，直接抛异常。
- `release_reference(id)`：producer 完成后释放自己的那一引用。consumer 在 `wait` 返回后通过 `release_consumer` 释放自己那一引用。两个引用都释放后，`release_reference` 把 id 推回 `free_` 队列。

### 3.5 引用回收

```cpp
void release_reference(u32 id) {
  Cell& cell = cells_[id];
  const u32 previous = cell.references.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 0) {
    throw std::logic_error("completion cell reference underflow");
  }
  if (previous != 1) return;
  if (!free_.try_push(id)) {
    throw std::logic_error("completion pool free queue overflow");
  }
}
```

- `fetch_sub(acq_rel)`：acq_rel 既保证减一可见性（release 给下一个观察者），又保证看到之前所有对 state 的写（acquire 给自己）。
- `previous == 1` 才回收：说明这是最后一个引用。`previous == 2` 表示对方还持有，不能回收。
- `previous == 0` 是不可能的（除非双重 release），抛异常。

`release_consumer`：
```cpp
void release_consumer(u32 id) {
  validate(id);
  release_reference(id);
}
```

只是 `release_reference` 的语义化别名。注意：**即使消费者超时放弃了等待，也必须调用 `release_consumer`**，否则槽位永远不会被回收。调用方在 `src/service/compute_service/storage_owner/public_mutations.cc:60` 就是这样做的：`wait` 返回（无论结果如何）后立即 `release_consumer`。

### 3.6 实际使用场景

在 `src/service/compute_service/storage_owner/lifecycle.cc:49`，计算服务为 storage owner 的同步 mutation API 创建了 `CompletionPool`。`public_mutations.cc:85-87` 展示了一个典型流程：先 `try_acquire`，失败则降级到阻塞 `acquire`（因为容量满了说明在途 RPC 太多）。这对接到第 28 课"计算侧 storage owner 更新"。`storage_completion_pool_->complete(...)` 则在 `src/service/compute_service/storage_owner/completion.cc:421` 被 storage owner 的响应处理逻辑调用（见第 27 课）。

---

## 4. `sliding_completion_ring.hh`：滑动完成环

文件：`src/common/sliding_completion_ring.hh:1-170`

这是 dvstor 自研的最复杂的一个并发原语，服务于**维护流水线的反压**。它的设计目标：

- **有序预留**：每个 mutation 在开始前拿到一个单调递增的 sequence。
- **乱序完成**：实际工作可以乱序结束。
- **前缀推进**：`finalized()` 水位只跨过连续完成的前缀，保证持久化顺序与预留顺序一致。
- **有界容量**：在途（已预留但未完成）的工作量受 capacity 限制，防止维护积压压垮存储控制块。

注释（`sliding_completion_ring.hh:14-16`）一句话总结：
> Reservations are ordered, work may finish out of order, and finalized() only advances across a contiguous completed prefix. Capacity is acquired before a mutation becomes visible.

### 4.1 数据结构

```cpp
struct Cell {
  std::atomic<u64> sequence{0};
  std::atomic<u32> remaining{0};
};

const size_t capacity_;
std::unique_ptr<Cell[]> cells_;
std::atomic<u64> next_;
std::atomic<u64> finalized_;
```

- 与 `bounded_queue` 一样，每个槽位自带 `sequence`，用来判断"这个槽当前持有的是哪个 sequence"。
- `next_`：下一个要分配的 sequence（单增）。
- `finalized_`：已连续完成的最大 sequence。
- `remaining`：该 sequence 还有多少 work item 未完成（一个 sequence 可以对应多个 work item，支持 batch 预留）。

构造要求 `next_sequence == finalized_sequence + 1`（`sliding_completion_ring.hh:25-28`），保证初始窗口为空。默认 `next_sequence = 1, finalized_sequence = 0`，所以 sequence 0 被保留为"无效/未预留"哨兵——`complete` 与 `remaining` 都对 `sequence == 0` 直接返回。

### 4.2 索引计算

```cpp
size_t index(u64 sequence) const noexcept {
  return static_cast<size_t>((sequence - 1) % capacity_);
}
```

`sequence` 从 1 开始，槽位映射是 `(sequence - 1) % capacity`。注意这里**不要求 capacity 是 2 的幂**（不像 `bounded_queue` 用 mask），因为这里的"槽位"是逻辑上的，且 capacity 可能由配置指定。这带来一次 mod，但维护路径不在最热的关键路径上。

### 4.3 `reserve_batch`：批量预留

```cpp
u64 reserve_batch(span<const u32> work_items, size_t admission_limit) {
  if (work_items.empty()) {
    throw std::invalid_argument("completion ring batch must not be empty");
  }
  if (admission_limit == 0 || admission_limit > capacity_ ||
      work_items.size() > admission_limit) {
    throw std::invalid_argument(
      "completion ring batch exceeds its admission window");
  }

  const u64 count = static_cast<u64>(work_items.size());
  u64 sequence = 0;
  for (;;) {
    const u64 done = finalized_.load(std::memory_order_acquire);
    sequence = next_.load(std::memory_order_acquire);
    if (sequence <= done) continue;
    if (sequence > std::numeric_limits<u64>::max() - count) {
      throw std::overflow_error("completion ring sequence overflow");
    }
    const u64 next_after_batch = sequence + count;
    if (next_after_batch - done - 1 > admission_limit) {
      finalized_.wait(done, std::memory_order_relaxed);
      continue;
    }
    if (next_.compare_exchange_weak(
          sequence, next_after_batch,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }

  for (size_t item = 0; item < work_items.size(); ++item) {
    const u64 item_sequence = sequence + static_cast<u64>(item);
    Cell& cell = cells_[index(item_sequence)];
    cell.remaining.store(work_items[item], std::memory_order_relaxed);
    cell.sequence.store(item_sequence, std::memory_order_release);
  }
  advance();
  return sequence;
}
```

这是文件里最复杂的一段，逐行讲解：

1. **参数校验**：`admission_limit` 可以小于物理 capacity（注释 `sliding_completion_ring.hh:64-68`）。这个分离允许"descriptor/intent 分配很大，但可见的在途工作受更严格限制"——例如维护 intent 表有 1024 项，但只允许 256 个 stage2 工作同时在途。
2. **加载顺序很关键**：先 load `finalized_` 再 load `next_`。注释 `sliding_completion_ring.hh:83-86` 解释：`finalized_` 推进意味着 `next_` 已经先推进了。先 load finalized 再 load next，能避免"finalized 看到了新值，但 next 看到了旧值"的错乱快照（那会导致 unsigned 下溢）。
3. **`sequence <= done` 时 continue**：理论上 `next_ > finalized_` 永远成立（因为 next 从 finalized + 1 开始），但并发推进中可能撞上瞬时不一致，continue 重读即可。
4. **溢出检查**：`next + count` 不能溢出 `u64`。维护 sequence 是单调递增的，长期运行可能撞天花板，所以直接抛异常。
5. **admission 检查**：`next_after_batch - done - 1 > admission_limit` 表示"如果分配这批，在途工作量会超过许可"。`-1` 是因为 `next_after_batch` 是下一个未分配位置，`next_after_batch - done - 1` 才是"已分配未完成"的数量。超限时 `finalized_.wait(done, relaxed)`——等水位推进。注意这里 wait 的语义：等任何推进，所以水位一变就醒来重试。
6. **CAS 推进 `next_`**：acq_rel 既发布我们即将写入的 cell 数据（release 给后续 `complete`），又看到之前其他 producer 写入的 cell 状态（acquire 给自己）。
7. **逐项初始化 cell**：先写 `remaining`（relaxed，因为下面 sequence.store(release) 是同步点），再写 `sequence`（release，发布这个槽位现在持有 `item_sequence`）。
8. **`advance()`**：尝试推进 `finalized_`。注释 `sliding_completion_ring.hh:109-111` 提到，这能跨过"零工作前缀"——如果某个 cell 的 `remaining == 0`（例如某个 work item 在 batch 里就是 0，或某种特殊预留），它的完成由"使前驱就绪的那次完成"触发。

为什么是 `reserve_batch` 而不是逐项 `reserve`？注释 `sliding_completion_ring.hh:55-59` 解释：如果多个 worker 各自只拿到 batch 的一部分，会形成**部分窗口死锁**——每个 worker 都在等剩余 capacity 释放，但因为没有任何一个 sequence 完整完成，`finalized_` 永远不推进。原子地一次性预留整批，避免这个死锁。

### 4.4 `complete` 与 `advance`

```cpp
void complete(u64 sequence, u32 work_items = 1) {
  if (sequence == 0 || work_items == 0) return;
  Cell& cell = cells_[index(sequence)];
  if (cell.sequence.load(std::memory_order_acquire) != sequence) {
    throw std::logic_error("completion ring stale or unknown sequence");
  }
  const u32 previous = cell.remaining.fetch_sub(
    work_items, std::memory_order_acq_rel);
  if (previous < work_items) {
    cell.remaining.fetch_add(work_items, std::memory_order_relaxed);
    throw std::logic_error("completion ring work counter underflow");
  }
  if (previous == work_items) advance();
}
```

- `cell.sequence != sequence` 检查非常重要：因为槽位是 `% capacity` 复用的，如果我们对"已经被回收并重新分配"的旧 sequence 调用 complete，会错误地减一个新 batch 的 remaining。这个检查防御性地抛异常。
- `fetch_sub(acq_rel)`：减去 work_items。如果 `previous == work_items`，说明这个 sequence 的工作全部完成，调用 `advance()` 尝试推进水位。
- 如果 `previous < work_items`，说明完成次数超过了预留次数——回滚并抛异常。

```cpp
void advance() {
  for (;;) {
    u64 watermark = finalized_.load(std::memory_order_acquire);
    const u64 candidate = watermark + 1;
    Cell& cell = cells_[index(candidate)];
    if (cell.sequence.load(std::memory_order_acquire) != candidate ||
        cell.remaining.load(std::memory_order_acquire) != 0) {
      return;
    }
    if (finalized_.compare_exchange_weak(
          watermark, candidate,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
      finalized_.notify_all();
    }
  }
}
```

- `advance` 是个循环：从当前水位 +1 开始，连续地把"sequence 匹配且 remaining == 0"的槽位跨过。
- 一旦遇到不匹配的槽（要么 sequence 对不上，要么 remaining 还没归零），就停止——这保证 finalized 总是连续前缀。
- CAS 成功后 `notify_all()`，唤醒在 `reserve_batch` 里 `wait` 的生产者。

### 4.5 `outstanding` 与读侧

```cpp
[[nodiscard]] size_t outstanding() const noexcept {
  const u64 done = finalized_.load(std::memory_order_acquire);
  const u64 next = next_.load(std::memory_order_acquire);
  if (next <= done) return 0;
  return static_cast<size_t>(next - done - 1);
}
```

同样是"近似"观测，因为两次独立 load 之间状态会变。注释 `sliding_completion_ring.hh:44-49` 强调先 load finalized 再 load next 的顺序，与 `reserve_batch` 一致，防止下溢。

### 4.6 实际使用

`SlidingCompletionRing` 在存储节点的维护流水线里被用作**反压阀门**。看 `src/memory_node/storage_owner_maintenance/queue.cc:88-93`：
```cpp
const u64 sequence =
  storage_owner_maintenance_completion_ring_->reserve_batch(
    work_items, storage_owner_maintenance_admission_limit_);
publish_storage_owner_maintenance_watermarks();
```
预留后立即把 `next_`/`finalized_` 水位发布到 GPU 可见的 `StorageControlBlock`（`queue.cc:95-118`）。GPU kernel 通过读这个控制块决定哪些维护 sequence 已经稳定可见。`complete_storage_owner_maintenance_sequence`（`queue.cc:124-133`）完成后再发布一次水位并 `notify_all`。这套机制衔接第 16 课（存储回收 RCU）和第 26 课（维护/wire protocol）。

---

## 5. CPU 核心亲和性：`core_assignment.{hh,cc}` + `core_partition.hh`

文件：`src/common/core_assignment.hh:1-123`、`src/common/core_assignment.cc:1-62`、`src/common/core_partition.hh:1-49`

dvstor 的存储节点对延迟极度敏感：NIC 在 NUMA node 1，跨 NUMA 访问会显著增加尾延迟。这套代码负责在启动时确定每个 worker 线程绑到哪个核。

### 5.1 硬件假设

注释 `core_assignment.hh:17-30` 明确写了机器拓扑：
```
w/out hyper-threading:
NUMA node0 CPU(s):    0-7
NUMA node1 CPU(s):    8-15

w/ hyper-threading
NUMA node0 CPU(s):   0-7,16-23
NUMA node1 CPU(s):   8-15,24-31

Strict policy: pin threads in the following order: 8-15, 0-7, 24-31, 16-23
Interleaved policy: 8,0,9,1,...,24,16,25,17,...
```

关键观察：
- 物理核 0-15（每个 NUMA 8 个），超线程 sibling 是 16-31。
- NIC 在 node 1（CPU 8-15），所以优先把 IO 线程绑到 8-15。
- strict 策略：先把 node1 物理核用完，再 node0 物理核，再 node1 超线程，最后 node0 超线程。
- interleaved 策略：node1 和 node0 交替，让两类线程均衡分布。

### 5.2 `CoreAssignment` 类骨架

```cpp
enum AssignmentPolicy { interleaved, strict };

template <AssignmentPolicy>
class CoreAssignment {
public:
  CoreAssignment() : cores_(num_cores_) {
    set_core_sequence();
    apply_local_process_partition();
    restrict_current_thread_to_partition();
    print_hardware_info();
  }

  u32 get_available_core() { return cores_[assigned_cores_++ % cores_.size()]; }
  u32 available_core_count() const { return static_cast<u32>(cores_.size()); }
  bool hyperthreading_enabled() const {
    return num_cores_ == physical_cores_per_socket_ * num_sockets_ * 2;
  }
  void reset() { assigned_cores_ = 0; }
  ...
};
```

- 模板参数是策略，`set_core_sequence` 是特化的（见 5.3）。
- 构造函数里四步：① 生成 core 序列；② 应用本进程的分区（多进程共享一台机器时）；③ 把当前线程（构造 CoreAssignment 的主线程）限制到分区；④ 打印信息。
- `get_available_core()`：每次调用返回下一个核，循环复用。注意 `assigned_cores_++` 不是原子的——**这个类设计上只允许单线程调用**，由调用方在主线程初始化期间把所有 worker 的核分配好，然后 worker 启动时各自 `pin_thread`。

成员变量：
```cpp
const u32 num_cores_{std::thread::hardware_concurrency()};
const u32 num_sockets_{2};
const u32 physical_cores_per_socket_{
  num_cores_ > 16 ? num_cores_ / (2 * num_sockets_) : num_cores_ / num_sockets_};
```
- 假设 2 socket。
- 超线程判断：如果总逻辑核数 > 16，假设开了 HT，物理核 = 逻辑核 / 4（2 socket × 2 HT）；否则物理核 = 逻辑核 / 2。这个启发式对应注释里 16 核和 32 核两种机器配置。

### 5.3 `set_core_sequence`：策略特化

strict 版本（`core_assignment.cc:5-25`）：
```cpp
template <>
void CoreAssignment<strict>::set_core_sequence() {
  std::iota(cores_.begin() + 0 * physical_cores_per_socket_,
            cores_.begin() + 1 * physical_cores_per_socket_,
            1 * physical_cores_per_socket_);   // cores_[0..7] = 8..15  (node1 物理)
  std::iota(cores_.begin() + 1 * physical_cores_per_socket_,
            cores_.begin() + 2 * physical_cores_per_socket_,
            0 * physical_cores_per_socket_);   // cores_[8..15] = 0..7  (node0 物理)
  if (hyperthreading_enabled()) {
    std::iota(cores_.begin() + 2 * physical_cores_per_socket_,
              cores_.begin() + 3 * physical_cores_per_socket_,
              3 * physical_cores_per_socket_); // cores_[16..23] = 24..31 (node1 HT)
    std::iota(cores_.begin() + 3 * physical_cores_per_socket_,
              cores_.begin() + 4 * physical_cores_per_socket_,
              2 * physical_cores_per_socket_); // cores_[24..31] = 16..23 (node0 HT)
  }
}
```
对应注释里的 strict 顺序：8-15, 0-7, 24-31, 16-23。`std::iota(begin, end, start)` 把 `[begin, end)` 填成 `start, start+1, ...`。

interleaved 版本（`core_assignment.cc:27-48`）：
```cpp
vec<u32> node1_cores(num_cores_ / num_sockets_);
vec<u32> node0_cores(num_cores_ / num_sockets_);

std::iota(node1_cores.begin(), node1_cores.begin() + physical_cores_per_socket_,
          1 * physical_cores_per_socket_);   // node1 物理: 8..15
std::iota(node0_cores.begin(), node0_cores.begin() + physical_cores_per_socket_,
          0 * physical_cores_per_socket_);   // node0 物理: 0..7
if (hyperthreading_enabled()) {
  std::iota(node1_cores.begin() + physical_cores_per_socket_, node1_cores.end(),
            3 * physical_cores_per_socket_);  // node1 HT: 24..31
  std::iota(node0_cores.begin() + physical_cores_per_socket_, node0_cores.end(),
            2 * physical_cores_per_socket_);  // node0 HT: 16..23
}
for (u32 i = 0, j = 0; i < num_cores_ / num_sockets_; ++i) {
  cores_[j++] = node1_cores[i];
  cores_[j++] = node0_cores[i];
}
```
先把两个 node 的核按"物理在前、HT 在后"放进临时数组，然后交替写入 `cores_`，得到 8,0,9,1,...,24,16,25,17,...。

### 5.4 进程内分区：`apply_local_process_partition`

当一台机器上跑多个 dvstor 进程（例如多存储 shard），需要把 core 在进程间划分。环境变量 `DVSTOR_LOCAL_PROCESS_RANK` / `DVSTOR_LOCAL_PROCESS_COUNT` 指定本进程的 rank 和总数：

```cpp
void apply_local_process_partition() {
  const auto rank = read_partition_environment("DVSTOR_LOCAL_PROCESS_RANK");
  const auto count = read_partition_environment("DVSTOR_LOCAL_PROCESS_COUNT");
  if (!rank.has_value() && !count.has_value()) return;
  if (!rank.has_value() || !count.has_value() || *count == 0 || *rank >= *count) {
    throw std::invalid_argument(
      "DVSTOR_LOCAL_PROCESS_RANK/COUNT must describe one valid partition");
  }
  local_process_rank_ = *rank;
  local_process_count_ = *count;
  if (*count == 1) return;
  cores_ = core_assignment_detail::partition_ordered_cores(
    cores_, hyperthreading_enabled(), *rank, *count);
}
```

- 两个变量要么都不设（单进程），要么都设。
- `count == 1` 时直接返回，不调分区函数（保持原 cores_）。
- 否则调用 `partition_ordered_cores` 把已经按拓扑排好的 `cores_` 切出本 rank 的那一部分。

### 5.5 `partition_ordered_cores`：SMT 感知的均分

文件 `src/common/core_partition.hh:15-47`：

```cpp
inline std::vector<std::uint32_t> partition_ordered_cores(
    const std::vector<std::uint32_t>& ordered,
    bool paired_smt_halves,
    std::uint32_t rank,
    std::uint32_t count) {
  if (ordered.empty() || count == 0 || rank >= count) {
    throw std::invalid_argument("invalid local process CPU partition");
  }
  const std::size_t group_count = paired_smt_halves
    ? ordered.size() / 2 : ordered.size();
  if (group_count == 0 || count > group_count ||
      (paired_smt_halves && ordered.size() % 2 != 0)) {
    throw std::invalid_argument("local process CPU partition exceeds core groups");
  }
  const std::size_t base = group_count / count;
  const std::size_t extra = group_count % count;
  const std::size_t begin = static_cast<std::size_t>(rank) * base +
                            std::min<std::size_t>(rank, extra);
  const std::size_t groups = base + (rank < extra ? 1 : 0);
  const std::size_t end = begin + groups;

  std::vector<std::uint32_t> result;
  result.reserve(groups * (paired_smt_halves ? 2 : 1));
  result.insert(result.end(), ordered.begin() + begin, ordered.begin() + end);
  if (paired_smt_halves) {
    result.insert(result.end(),
                  ordered.begin() + group_count + begin,
                  ordered.begin() + group_count + end);
  }
  return result;
}
```

关键设计：**当 `paired_smt_halves == true`，假设 `ordered` 的前半是物理核、后半是对应的超线程 sibling**（这正是 5.3 里 strict/interleaved 生成的顺序）。

- `group_count = ordered.size() / 2`：把"一个物理核 + 它的 HT sibling"看作一个不可分割的组。注释 `core_partition.hh:11-14` 解释：避免两个 shard 在同一物理核的 sibling 上互相竞争，而其他物理核却闲置。
- `base`/`extra`：均分组数，余数前 `extra` 个 rank 各多分一组。`begin`/`end` 用 `min(rank, extra)` 处理这个偏移。
- 输出：先插入前半段 `[begin, end)`（物理核），再插入后半段对应位置 `[group_count + begin, group_count + end)`（HT sibling）。这样本进程拿到的是若干**完整的物理核及其 HT sibling**，而不是一堆散乱的逻辑核。

`paired_smt_halves == false`（未开 HT）时直接均分逻辑核。

### 5.6 `restrict_current_thread_to_partition`：尊重外层 taskset/cgroup

```cpp
void restrict_current_thread_to_partition() {
#ifdef __linux__
  cpu_set_t inherited;
  CPU_ZERO(&inherited);
  if (sched_getaffinity(0, sizeof(inherited), &inherited) != 0) {
    throw std::runtime_error(
      std::string("sched_getaffinity failed: ") + std::strerror(errno));
  }
  vec<u32> allowed;
  allowed.reserve(cores_.size());
  for (const u32 cpu : cores_) {
    if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &inherited)) {
      allowed.push_back(cpu);
    }
  }
  if (allowed.empty()) {
    throw std::runtime_error(
      "local process CPU partition does not intersect inherited affinity");
  }
  cpu_set_t partition;
  CPU_ZERO(&partition);
  for (const u32 cpu : allowed) CPU_SET(cpu, &partition);
  if (sched_setaffinity(0, sizeof(partition), &partition) != 0) {
    throw std::runtime_error(
      std::string("sched_setaffinity failed: ") + std::strerror(errno));
  }
  cores_ = std::move(allowed);
#endif
}
```

- `sched_getaffinity` 读取继承的亲和性掩码（来自 taskset/cgroup/systemd）。
- 取 `cores_` 与 inherited 的**交集**：如果运维已经用 taskset 把这个进程限制到某些核，CoreAssignment 不能突破这个限制。
- 用 `sched_setaffinity` 把当前主线程固定到交集。后续 `get_available_core()` 返回的核都在这个集合里，所以 worker `pin_thread` 也不会越界。
- 注释 `core_assignment.hh:90-92` 总结这个意图：" Respect an outer taskset/cgroup mask. The intersection also ensures that later pin_thread calls cannot escape the local shard partition."

`get_available_core` 之后会被 `src/memory_node/storage_owner_runtime/lifecycle.cc`、`src/memory_node/peer_rpc/runtime.cc` 等地方用来给每个 worker 线程选核（见第 21、23 课）。

### 5.7 `read_partition_environment`：严谨的环境变量解析

```cpp
static std::optional<u32> read_partition_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return std::nullopt;
  u32 parsed = 0;
  const std::string_view text{value};
  const auto [end, error] = std::from_chars(
    text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}
```

- 空字符串视为未设置（返回 nullopt）。
- 用 `std::from_chars` 而不是 `atoi`/`strtol`：更快、不依赖 locale、能精确检测"部分解析"（如 `"3a"`）。
- `end != text.data() + text.size()` 拒绝"3a"这类含尾随垃圾的输入。

---

## 6. `coroutine.hh`：C++20 协程桥接

文件：`src/coroutine.hh:1-93`

dvstor 在存储 owner 的图修改路径（插入/搜索）里用 C++20 协程来表达"异步读远端节点 → 处理 → 再异步读"的流水线。本文件定义了三个协程返回对象类型。

### 6.1 协程基础回顾

C++20 协程通过 `promise_type` 定义行为：
- `get_return_object()`：构造给调用方的返回对象。
- `initial_suspend()`：协程体执行前的暂停点。`suspend_never` = 立即执行；`suspend_always` = 创建即挂起。
- `final_suspend() noexcept`：协程体执行完后的暂停点。`suspend_never` = 自动销毁帧；`suspend_always` = 留给调用方销毁。
- `return_void()`/`return_value()`：`co_return` 时调用。
- `unhandled_exception()`：异常路径。

`awaitable` 通过 `await_ready`/`await_suspend`/`await_resume` 三个方法控制 `co_await` 行为。

### 6.2 `MinorCoroutine`

```cpp
struct MinorCoroutine {
  struct promise_type {
    MinorCoroutine get_return_object() {
      return MinorCoroutine{Handle::from_promise(*this)};
    }
    static std::suspend_never initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;

  explicit MinorCoroutine(Handle handle) : handle(handle) {}

  ~MinorCoroutine() {
    if (handle) {
      handle.destroy();
    }
  }
  ...
  Handle handle;
};
```

- `initial_suspend` 是 `suspend_never`：协程一旦创建就**立即执行到第一个 `co_await` 或结束**。注释 `coroutine.hh:14` 说"the object is created after first suspend"，意思是返回对象拿到时协程已经跑过一段了。
- `final_suspend` 是 `suspend_always`：协程结束后**不自动销毁帧**，由 `~MinorCoroutine` 调 `handle.destroy()` 显式销毁。注释 `coroutine.hh:10` 说"Handle is destroyed by the destructor to prevent memory leaks."
- 禁用拷贝和移动：协程帧生命周期与 wrapper 一一绑定，避免双重 destroy。
- `unhandled_exception` 重新抛出：让异常逃逸到调用方。

"Minor" 的语义是"被其他协程调用的协程"——它没有自己的状态字段，只是一个轻量的执行包装器。

### 6.3 `VamanaCoroutine`

```cpp
struct VamanaCoroutine {
  struct promise_type {
    VamanaCoroutine get_return_object() {
      return VamanaCoroutine{Handle::from_promise(*this)};
    }
    static std::suspend_always initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;
  Handle handle;

  struct BeamEntry {
    RemotePtr rptr;
    distance_t distance;
    bool expanded{false};
  };

  vec<BeamEntry> beam{};
  hashset_t<RemotePtr> visited_nodes{};
  vec<RemotePtr> scratch_unvisited{};
  vec<RemotePtr> reserved_ptrs_a{};
  vec<RemotePtr> reserved_ptrs_b{};
  vec<RemotePtr> indirect_candidate_ptrs{};
  vec<u32> indirect_candidate_indices{};
  vec<float> scratch_distances{};
  vec<const byte_t*> scratch_entry_ptrs{};
  vec<u32> scratch_indices_a{};
  vec<u32> scratch_indices_b{};
  vec<u8> scratch_flags{};

  bool gpu_pending{false};
};
```

- 与 `MinorCoroutine` 不同，`initial_suspend` 是 `suspend_always`：协程创建后**不立即执行**，要等调用方 `handle.resume()` 才开始。这适合"创建一批协程，然后统一调度"的模式。
- **注意**：成员变量（`beam`、`visited_nodes`、各种 scratch）虽然写在 struct 里，但它们**实际位于协程帧上**（因为编译器把 promise 放在帧里，struct 就是 promise 的基类布局）。这些是 Vamana 图搜索的 beam search 状态：候选节点、已访问集合、临时缓冲。
- 注释 `coroutine.hh:40-42` 说它取代了旧的 `HNSWCoroutine`，因为 dvstor 用 Vamana 图而非 HNSW，搜索状态是 beam 而非 HNSW 的堆。
- `gpu_pending`：标记是否有一个 GPU 异步操作在途，调度器据此决定是否 resume 这个协程。
- 没有显式析构调 `destroy`：这意味着 `VamanaCoroutine` 的生命周期由调度器管理（在 storage owner runtime 里，见第 21、26 课），调用方负责在适当时候 destroy。

### 6.4 `StorageOwnerInsertCoroutine`

```cpp
struct StorageOwnerInsertCoroutine {
  struct promise_type {
    StorageOwnerInsertCoroutine get_return_object() {
      return StorageOwnerInsertCoroutine{Handle::from_promise(*this)};
    }
    static std::suspend_always initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;
  Handle handle;
};
```

最简形式，只有 handle。这是 storage owner 处理一条 insert mutation 的协程返回类型，实际实现见 `src/memory_node/storage_owner_index/candidate_search.cc:147` 和 `graph_mutation.cc:311`。

### 6.5 与 awaitable 的配合

虽然本文件只定义了 promise_type，真正的异步是通过 awaitable 实现的。看 `src/memory_node/storage_owner_index/detail.hh:77-91`：

```cpp
struct MemoryNode::GlobalMedoidReadAwaitable {
  bool ready{};
  byte_t* buffer{};
  MemoryNode* node{};

  bool await_ready() const { return ready; }
  static void await_suspend(std::coroutine_handle<>) {}
  RemotePtr await_resume() const {
    if (node->storage_id_ == 0) {
      return RemotePtr{
        *reinterpret_cast<u64*>(node->index_buffer_.get_full_buffer() + 8)};
    }
    return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
  }
};
```

- `await_ready`：如果 `ready == true`，直接调 `await_resume`，不挂起。这是"结果已经就地可用"的快路径。
- `await_suspend`：空实现 + 返回 void，意味着协程**立即被挂起**，控制权返回调用方。实际异步操作的完成由外部代码（在 buffer 就绪后）调 `handle.resume()` 触发。这是"手动调度"的协程模式——没有内置的 executor，所有调度都显式。
- `await_resume`：从 buffer 解析出 `RemotePtr`。

`NodeSnapshotReadAwaitable`、`NodeSnapshotsReadAwaitable`、`NeighborListReadAwaitable`（`detail.hh:93-159`）结构类似，都遵循"`ready` 快路径 + 挂起等外部 resume + resume 时解析"的模式。在 `candidate_search.cc:162` 可以看到它们被 `co_await` 调用：

```cpp
NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
```

这套 awaitable + 协程的完整调度循环在第 21 课（kernel 运行时/角色调度）和第 26 课（维护/wire protocol）里详细讲。

---

## 7. `remote_pointer.hh`：远端指针编码

文件：`src/remote_pointer.hh:1-62`

`RemotePtr` 是 dvstor 的"全局地址"——在 16b 节点号 + 48b 偏移的格式里编码"哪个存储节点的哪个字节偏移"。这种编码让图里的邻居指针、anchor、idmap 项都能用一个 64 位整数表达，节省内存且对 RDMA 友好。

### 7.1 编码布局

```cpp
struct RemotePtr {
  static constexpr size_t SIZE = sizeof(u64);
  u64 raw_address{};  // [ memory node (16b) | byte offset (48b) ]

  RemotePtr() = default;
  explicit RemotePtr(u64 raw_address) : raw_address(raw_address) {}
  RemotePtr(u32 memory_node, u64 byte_offset) { store_address(memory_node, byte_offset); }
  ...
};
```

- `raw_address` 是一个 `u64`，高 16 位是 memory node id，低 48 位是字节偏移。
- `SIZE = sizeof(u64) = 8` 字节，这个常量在 GPU 侧代码里被用来对齐缓冲布局（见第 17、20 课）。
- 三种构造：默认（null）、从原始 u64、从 node + offset。

### 7.2 解码

```cpp
u32 memory_node() const { return raw_address >> 48; }
u64 byte_offset() const { return (raw_address << 16) >> 16; }
bool is_null() const { return raw_address == 0; }
void reset() { raw_address = 0; }
```

- `memory_node()`：右移 48 取高 16 位。
- `byte_offset()`：先左移 16（把高 16 位移出），再右移 16（无符号右移，高位补 0）。等价于 `raw_address & 0x0000FFFFFFFFFFFF`，但用移位实现可能更便宜。
- `is_null()`：约定 `raw_address == 0` 表示 null（node 0 + offset 0 是无效组合，因为 offset 0 通常是控制块而非数据）。

### 7.3 编码

```cpp
void store_address(u32 memory_node, u64 byte_offset) {
  raw_address = (static_cast<u64>(memory_node) << 48) | byte_offset;
}
```

把 node 放到高 16 位，与 offset OR 起来。注意这里**没有掩码掉 byte_offset 的高 16 位**——如果调用方传了超过 48 位的 offset，会污染 node 字段。这是有意的"信任调用方"设计，因为 48 位偏移覆盖 256TB 地址空间，远超任何单节点内存。

### 7.4 哈希特化

`RemotePtr` 经常作为 hashset/hashmap 的 key（例如 `VamanaCoroutine` 的 `visited_nodes`、`scratch_unvisited`）。文件提供了两个特化：

**std::hash 特化（`remote_pointer.hh:31-52`）**：
```cpp
template <>
struct std::hash<RemotePtr> {
  size_t operator()(const RemotePtr& r) const noexcept {
    u64 h = std::hash<u64>{}(r.raw_address);
    // murmur64
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    ...
    return h;
  }
};
```

- 先用 `std::hash<u64>` 取一个初值，再用 murmur64 finalizer 做雪崩。Murmur 的常量 `0xff51afd7ed558ccd` / `0xc4ceb9fe1a85ec53` 是官方推荐的 64 位 finalizer。
- 这一步是必要的，因为 `raw_address` 的高 16 位（node id）变化很慢——如果直接用 `std::hash<u64>`（通常是恒等或简单乘法），同一个 node 的所有指针会聚集在相邻桶，破坏哈希分布。

**ankerl::unordered_dense::hash 特化（`remote_pointer.hh:54-61`）**：
```cpp
template <>
struct ankerl::unordered_dense::hash<RemotePtr> {
  using is_avalanching = void;
  size_t operator()(const RemotePtr& r) const noexcept {
    return ankerl::unordered_dense::hash<u64>{}(r.raw_address);
  }
};
```

- `is_avalanching = void` 是 ankerl 库的标记，告诉调用方"这个 hash 已经做了雪崩，不需要再 mix"。
- 直接复用 `unordered_dense::hash<u64>`（它本身就是 avalanching 的），比 std::hash 版本更简洁高效。

### 7.5 比较与输出

```cpp
bool operator==(const RemotePtr&) const = default;  // compares raw_address

friend std::ostream& operator<<(std::ostream& os, const RemotePtr& r) {
  return os << "[node: " << r.memory_node() << " | offset: " << r.byte_offset() << "]";
}
```

- `= default` 让编译器生成逐成员比较，这里就是比较 `raw_address`。
- 输出格式 `[node: X | offset: Y]` 方便调试日志。

### 7.6 GPU 侧镜像

注意本文件是 CPU 侧定义。GPU kernel 里也有对应的 5 字节（40 位）/ 8 字节解码逻辑——`RemotePtr` 在 CPU-GPU 之间是按 8 字节 `u64` 传输的，但 GPU kernel 在某些紧凑结构里会只用低 40 位 offset 以节省寄存器。这部分在第 17 课（kernel 启动器/上下文/device ring）和第 20 课（查询遍历主循环）讲。

---

## 8. `timing.hh` + `timing.cc`：高精度计时

文件：`src/common/timing.hh:1-52`、`src/common/timing.cc:1-114`

dvstor 的性能分析高度依赖分阶段计时（construction 的各阶段、查询的各阶段、维护的各阶段）。这套 `Timing` 类提供"注册若干 Interval 句柄，每个句柄可重复 start/stop 累计"的模式。

### 8.1 `Interval` 结构

```cpp
struct Interval {
  str descriptor_;
  clockid_t clock_id_;
  timespec time_{};
  timespec time_start_{};
  explicit Interval(str&& descriptor);
  void start();
  void stop();
  void clear();
  void add(const s_ptr<Interval>& t2);
  f64 get_ms() const;
};
```

- `descriptor_`：人类可读的名字，会作为 json 的 key。
- `clock_id_`：构造时固定为 `CLOCK_MONOTONIC`（`timing.cc:10`），不受系统时间调整影响，适合测量区间。
- `time_`：累计的已测量时间。
- `time_start_`：最近一次 `start` 的时刻。

### 8.2 start/stop

```cpp
void Timing::Interval::start() {
  lib_assert(clock_gettime(clock_id_, &time_start_) == 0,
             "calling clock_gettime failed");
}

void Timing::Interval::stop() {
  timespec time_now;
  lib_assert(clock_gettime(clock_id_, &time_now) == 0,
             "calling clock_gettime failed");
  time_ = time_now - time_start_ + time_;
}
```

- `start` 只记录起点，`stop` 算 `(now - start)` 然后累加到 `time_`。
- 用 `lib_assert` 而不是抛异常：`clock_gettime` 几乎不会失败，失败说明系统有严重问题，直接 abort 更合理。
- **不是 rdtsc**：虽然文件名叫 timing，但用的是 `clock_gettime(CLOCK_MONOTONIC)`。注释在大纲里提到"rdtsc/clock"，实际代码用 clock_gettime——这是 vDSO 加速的，足够快且不需要校准 TSC 频率。在 GPU 侧的 kernel 计时才用 `clock64()`（见第 18、19 课）。

### 8.3 `timespec` 运算符

```cpp
timespec operator-(const timespec& ts1, const timespec& ts2) {
  struct timespec res;
  if (ts1.tv_sec >= 0 && ts2.tv_sec >= 0) {
    res.tv_sec = ts1.tv_sec - ts2.tv_sec;
    res.tv_nsec = ts1.tv_nsec - ts2.tv_nsec;
    if (res.tv_nsec < 0) {
      res.tv_nsec += 1000000000;
      res.tv_sec -= 1;
    }
  } else {
    std::perror("timing call to operator- failed");
    std::abort();
  }
  return res;
}
```

- 手动处理 nanosecond 借位：`tv_nsec` 范围是 `[0, 1e9)`，减法为负时从 `tv_sec` 借 1。
- 拒绝负数输入：`CLOCK_MONOTONIC` 不会返回负数，所以这是防御性检查。
- `operator+` 对称处理进位。

### 8.4 注册与汇总

```cpp
Timing::IntervalPtr Timing::create_enroll(str&& descriptor) {
  auto interval = std::make_shared<Interval>(std::move(descriptor));
  intervals_.push_back(interval);
  return interval;
}
```

- `create_enroll` 创建一个 `Interval`，注册到 `intervals_` 列表，返回 `shared_ptr` 给调用方持有。
- 调用方拿到 `shared_ptr` 后可以反复 `start()`/`stop()`；`Timing` 也持有一份副本用于最终汇总。

```cpp
Timing::json Timing::to_json() const {
  json out;
  for (auto& interval : intervals_) {
    out[interval->descriptor_] = interval->get_ms();
  }
  return out;
}
```

- `to_json` 遍历所有注册的 interval，输出 `{descriptor: ms}` 字典。
- 这就是 breakdown benchmark 报告的核心数据源（见第 30 课）。

### 8.5 静态便捷方法与时间戳

```cpp
static void start(const IntervalPtr& interval) { interval->start(); }
static void stop(const IntervalPtr& interval) { interval->stop(); }
static void clear(const IntervalPtr& interval) { interval->clear(); }
```

这些静态方法是语法糖，允许 `Timing::start(handle)` 这种调用风格。

```cpp
nlohmann::json get_timestamp() {
  nlohmann::json json_obj;
  const std::time_t now = std::time(nullptr);
  const std::tm* time = std::localtime(&now);
  json_obj["$date"] = (std::stringstream{} << std::put_time(time, "%Y-%m-%dT%H:%M:%SZ")).str();
  return json_obj;
}
```

- `get_timestamp` 返回当前时间的 ISO 8601 字符串，包在 `$date` 字段里（MongoDB 风格）。
- 注意用 `std::localtime`（非线程安全）——所以这个函数不应该在多线程并发调用；它是启动/报告生成时单线程用的。

### 8.6 使用模式

调用方典型用法：
```cpp
Timing timing;
auto t_stage1 = timing.create_enroll("stage1");
auto t_stage2 = timing.create_enroll("stage2");
Timing::start(t_stage1);
// ... do work ...
Timing::stop(t_stage1);
Timing::start(t_stage2);
// ... do work ...
Timing::stop(t_stage2);
std::cerr << timing << std::endl;  // 输出 json
```

这个模式在 construction（第 12、13 课）、维护、breakdown benchmark（第 30 课）里反复出现。

---

## 9. 关键数据结构与流程图

### 9.1 数据结构关系

```
+---------------------------+
|   bounded::Queue<T>       |   Vyukov MPMC，预分配 Cell[]
|   - enqueue/dequeue pos   |<---------+
|   - push/pop epoch (wait) |          |
+---------------------------+          |
                                       |
+---------------------------+          |
|   CompletionPool          |          | free_ 队列复用
|   - Cell{state, refs}     |----------+  (Queue<u32>)
|   - 两方引用计数          |
+---------------------------+

+---------------------------+
| SlidingCompletionRing     |   独立的 Cell{sequence, remaining}
| - next_ / finalized_      |   前缀连续推进
| - reserve_batch / complete|
+---------------------------+

+---------------------------+
| RemotePtr                 |   u64: [node 16b | offset 48b]
| - memory_node()/offset()  |   跨 CPU/GPU/网络的统一地址
| - std::hash / ankerl hash |
+---------------------------+
```

### 9.2 CPU 线程模型 + 完成环 + 协程调度

```
                    [主线程]
                       |
            CoreAssignment<strict>
            决定每个 worker 绑哪个核
                       |
        +--------------+--------------+
        |              |              |
   [SO worker 0]  [SO worker 1]  [peer RPC worker]
   pin core 8     pin core 9      pin core 10
        |              |              |
        v              v              v
   +---------------------------------------+
   | bounded::Queue<StorageOwnerInsertTask>|  MPSC 任务队列
   |  try_pop / pop_wait(stop)             |
   +---------------------------------------+
        |              |
        | 每个 task 启动一个 StorageOwnerInsertCoroutine
        v
   +---------------------------------------+
   | Coroutine frame (promise + locals)    |
   |  co_await async_read_node_snapshot    |
   |  co_await async_read_neighbor_list    |
   +---------------------------------------+
        |
        | await_suspend 把协程挂起，控制权回 worker
        | worker 继续处理其他 task 或 pop_wait
        |
        | 异步 RDMA 完成后，外部代码调 handle.resume()
        v
   +---------------------------------------+
   | 协程恢复，继续执行                     |
   +---------------------------------------+
        |
        | 维护任务完成时
        v
   +---------------------------------------+
   | SlidingCompletionRing.complete(seq)   |
   |  -> advance() 推进 finalized_         |
   |  -> notify_all() 唤醒 reserve_batch   |
   +---------------------------------------+
        |
        | 同步 mutation API 路径
        v
   +---------------------------------------+
   | CompletionPool                        |
   |  consumer: wait(id)                   |
   |  producer: complete(id, success)      |
   |  双方 release_reference 后回收         |
   +---------------------------------------+
```

这个图展示了三个并发原语如何协作：
- `bounded::Queue` 把任务从生产者（API 线程）传给 worker。
- 协程在 worker 上跑，遇到异步 IO 时挂起，让出 worker 给其他任务。
- `SlidingCompletionRing` 控制维护路径的反压，防止在途工作爆炸。
- `CompletionPool` 服务于同步 API 调用方，把异步完成转成同步等待。

### 9.3 内存序选择对照

| 操作 | 内存序 | 原因 |
| --- | --- | --- |
| `update_max_relaxed` | relaxed | 统计用途，无发布-订阅 |
| `CounterDecrementGuard` 析构 | acq_rel | 配合"等 in-flight 归零"的 acquire 观察 |
| `Queue::emplace` 写 sequence | release | 发布 value 给消费者 |
| `Queue::try_pop` 读 sequence | acquire | 同步生产者的 value 写入 |
| `Queue::emplace` CAS enqueue_position | relaxed | CAS 本身是原子的，同步靠 cell->sequence |
| `CompletionPool::wait` 读 state | acquire | 看到 complete 的 release |
| `CompletionPool::complete` CAS state | release/acquire | 发布结果 / 重读 pending |
| `SlidingCompletionRing::reserve_batch` CAS next_ | acq_rel | 发布 cell 初始化 + 看到其他 producer |
| `SlidingCompletionRing::advance` CAS finalized_ | acq_rel | 发布水位推进 + 看到其他 advance |

---

## 10. 与其他模块的关系

- **第 4-5 课（RDMA 传输库）**：RDMA 完成回调会唤醒 `bounded::Queue` 的 `pop_wait`，并把响应送到 `CompletionPool::complete`。
- **第 8 课（元数据/owner map/存储协议）**：owner map 用 `RemotePtr` 标识每个数据块的全局地址。
- **第 9 课（GPU 类型/遥测/PQ 模型）**：GPU 侧的 `RemotePtr` 镜像定义在那里，与 CPU 侧的 8 字节布局对齐。
- **第 11 课（持久化引擎 PImpl/生命周期）**：`Timing` 被用来测量引擎各阶段耗时。
- **第 16 课（存储回收 RCU）**：`SlidingCompletionRing` 的 `finalized_` 水位决定哪些旧版本可以回收。
- **第 17 课（kernel 启动器/上下文/device ring）**：GPU 侧的 device ring 是 `bounded::Queue` 的 GPU 版本，思路类似但实现不同。
- **第 18-20 课（候选评分/RDMA cache/查询遍历）**：GPU kernel 用 `RemotePtr` 解码邻居地址，发起 RDMA 读。
- **第 21 课（kernel 运行时/角色调度）**：CPU worker 从 `bounded::Queue` 取任务，调度协程 resume。
- **第 23 课（存储节点主体/peer RDMA）**：`bounded::Queue<PeerReverseUpdateResponse>` 用本课的队列实现 peer 间反向更新。
- **第 26 课（维护/wire protocol）**：`SlidingCompletionRing` + `StorageOwnerInsertCoroutine` 是维护流水线的核心。
- **第 27 课（计算服务主体）**：计算服务的同步 API 用 `CompletionPool` 等存储节点响应。
- **第 28 课（计算侧 storage owner 更新）**：`public_mutations.cc` 是 `CompletionPool` 的主要使用者。
- **第 30 课（breakdown benchmark）**：`Timing` 的 `to_json` 输出是 benchmark 报告的数据源。

---

## 11. 小结

本课讲解了 dvstor 在 CPU 侧的七组并发基础设施：

1. **`atomic_utils`**：两段高频复用的原子片段（relaxed 更新最大值、RAII 减计数）。
2. **`bounded::Queue`**：Vyukov 风格 MPMC 队列，槽位自带 sequence 实现无 ABA 的空满判断，配合 C++20 `atomic::wait/notify` 实现低开销阻塞。
3. **`CompletionPool`**：两方引用计数的完成池，专门解决"同步 API 等异步响应"场景，预分配避免热路径 `new`，双引用防止超时消费者造成 ABA。
4. **`SlidingCompletionRing`**：滑动完成环，有序预留 + 乱序完成 + 前缀推进，批量预留避免部分窗口死锁，admission_limit 分离物理容量与逻辑反压。
5. **`CoreAssignment` + `partition_ordered_cores`**：NUMA/HT 感知的核分配，strict/interleaved 两种策略，支持多进程分区，尊重外层 taskset/cgroup。
6. **`coroutine.hh`**：C++20 协程的 promise_type 桥接，配合外部 awaitable 实现"手动调度"的异步图搜索。
7. **`RemotePtr`**：16b+48b 全局地址编码，murmur64 哈希保证分布均匀。
8. **`Timing`**：`CLOCK_MONOTONIC` 累计式分阶段计时，输出 json 供 breakdown 分析。

这些原语的设计共同体现了 dvstor 的工程哲学：**预分配、无锁/无分配热路径、NUMA 感知、显式同步**。它们不炫技，每个内存序选择和字段布局都有具体的并发场景在驱动。掌握这些原语之后，后续课程里看到的 worker 池、维护流水线、同步 API 都会变得透明——它们只是把这些积木按特定方式组装起来。
