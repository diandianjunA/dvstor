# 第 19 课：CUDA kernel 与距离计算实现

## 本课目标

本课深入 CUDA kernel launcher 和 distance kernels。你需要理解 typed distance 的分支、single-query、multi-query、id-based、indirect pointer 路径，以及 RobustPrune kernel 的基本实现和性能风险。

## 代码证据

必须阅读：

- `src/gpu/gpu_kernel_launcher.hh`
- `src/gpu/gpu_kernel_launcher.cu`
- `src/gpu/kernels/distance_kernels.cuh`
- `src/common/distance.hh`
- `src/common/vector_dtype.hh`

## launcher API 分类

distance launcher：

- `launch_batch_l2_distances`
- `launch_batch_typed_l2_distances`
- `launch_batch_typed_query_l2_distances`
- `launch_batch_typed_multi_query_l2_distances`
- `launch_batch_id_l2_distances`
- `launch_batch_typed_query_l2_distances_indirect`

prune launcher：

- `launch_robust_prune`
- `launch_robust_prune_typed`

资源 API：

- `gpu_init`
- `gpu_malloc`
- `gpu_malloc_host`
- stream/event create/destroy
- async memcpy

## dtype pair 分支

`launch_batch_typed_query_l2_distances` 根据 query dtype 和 candidate dtype 选择模板：

| query | candidate | path |
| --- | --- | --- |
| float32 | float32 | `launch_batch_l2_distances` |
| float32 | uint8 | typed pair |
| float32 | int8 | typed pair |
| uint8 | float32 | typed pair |
| uint8 | uint8 | integral pair |
| uint8 | int8 | integral pair |
| int8 | float32 | typed pair |
| int8 | uint8 | integral pair |
| int8 | int8 | integral pair |

integral pair 会判断 int32 accumulator 是否安全，否则使用 int64。

## TILE_SIZE 和 BLOCK_SIZE

launcher 中：

```cpp
TILE_SIZE = 4
BLOCK_SIZE = 512
total_threads = n_candidates * TILE_SIZE
num_blocks = ceil(total_threads / BLOCK_SIZE)
```

kernel 中一个 tile 计算一个 candidate distance。tile 内每个线程处理维度上的 stride，最后 cooperative groups reduce。

优点：

- 实现简单。
- 一个 candidate 的 dim 由少量线程并行。

风险：

- n_candidates 小时 GPU 利用率差。
- dim 很大时 4 线程可能不足。
- 不同 dtype 的内存加载 coalescing 和转换成本不同。

## float32 kernel

`batch_l2_squared_distance_kernel` 对 float32 做了 `uint4` 向量化读取：

```text
query -> uint4*
candidate -> uint4*
每次处理 4 floats
尾部处理 dim % 4
tile reduce
```

这比逐 float 读取更利于 memory coalescing，但要求地址对齐和访问模式合理。

## typed pair kernel

`batch_l2_typed_pair_distance_kernel`：

- `typed_component_to_float` 把组件转 float。
- 如果 query/candidate 都是 integral，用 integer accumulator。
- 否则用 float accumulator。

这避免 uint8/int8 路径先解码成 float buffer，减少 H2D 和存储成本。

## multi-query kernel

`batch_l2_typed_multi_query_distance_kernel` 多了：

```text
candidate_query_ids[tile_id]
query = queries + candidate_query_id * dim
candidate = candidates + tile_id * dim
```

它允许多个 query 的候选合并成一个 GPU launch，但需要一个 candidate 到 query 的映射数组。

在 `knn_batch` 中用于 query batch path。

## indirect pointer kernel

`batch_l2_typed_pair_distance_indirect_kernel`：

```text
cand_vec = candidate_ptrs[tile_id]
```

适合候选向量不连续时使用。代价是多一次 pointer array H2D 和 kernel 中间接读取。

## id-based kernel

`batch_l2_id_distance_kernel`：

```text
query = base_vectors + query_id * dim
candidate = base_vectors + candidate_id * dim
```

这适合 base vectors 常驻 GPU 的场景。当前在线 RDMA 查询主要不是这条路径，因为候选向量来自远端 memory node。

## RobustPrune kernel

`robust_prune_kernel` 和 typed 版本：

1. shared memory 中 `is_valid[n_candidates]`。
2. 如果 `n_candidates <= max_R`，直接输出。
3. 否则从排序候选中依次选择 pstar。
4. 对后续候选计算 pstar 到 candidate 的距离。
5. 如果 `alpha * dist_pstar_pprime <= candidate_dists[i]`，标记 invalid。
6. 直到 selected 数达到 R。

风险：

- shared memory 大小随 n_candidates。
- pair distance 是 O(R * n_candidates * dim) 上界。
- 线程块内串行选择 pstar，算法并行度有限。

## CPU distance 与 GPU distance

CPU：

- `L2SqrSIMD16ExtAVX`
- `InnerProductSIMD16ExtAVX`
- typed CPU distance functions in `vector_dtype.hh`

GPU：

- 主要是 L2 typed kernels。

读代码时要确认路径：

- entry point 初始距离走 CPU。
- 查询候选 exact 距离走 GPU。
- 插入搜索和 prune 距离走 GPU。
- storage-owner 路径多为 CPU 距离。
- offline builder 路径是 CPU distance。

## 性能影响

可优化点：

- 小 batch 下 kernel launch overhead。
- TILE_SIZE 固定为 4 是否适合所有 dim。
- float32 vectorized load 和 uint8/int8 typed load 的吞吐差异。
- D2H distances 大小为 `n_candidates * 4B`，通常不是带宽大头，但同步点重要。
- RobustPrune kernel 可能成为插入和 overflow prune 主要 GPU 开销。

## 设计异味

1. launcher 分支手写，dtype 增加时容易膨胀。
2. TILE_SIZE 和 BLOCK_SIZE 是固定常量，没有按 dim 调优。
3. IP distance GPU 路径不如 L2 路径完整。
4. RobustPrune kernel 和 distance kernel 同处 launcher 文件，职责混合。
5. kernel 错误检查主要靠 `CUDA_CHECK(cudaEventRecord)`，缺少 launch 后 `cudaGetLastError`。

## 可验证问题

- 一个 candidate distance 由多少 CUDA thread 计算？
- uint8/int8 何时用 int64 accumulator？
- multi-query kernel 如何知道每个 candidate 属于哪个 query？
- indirect pointer path 的额外成本是什么？
- RobustPrune 的复杂度主要由哪些参数决定？

## 学习任务

1. 为 dim=128、n_candidates=1024 手算 blocks 数。
2. 列出所有 GPU launcher 和它们的调用点。
3. 设计实验比较 TILE_SIZE=4 和 TILE_SIZE=8 的影响。
4. 检查 IP distance 是否在在线 GPU 查询中完整支持。
5. 思考：如果把 vectors 常驻 GPU，哪些 RDMA/GPU kernel 路径会被替换？

