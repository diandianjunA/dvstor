# 第 26 课：实验工具与 benchmark 主路径

## 本课目标

本课学习项目现有 benchmark 如何驱动 `ComputeService`，以及它能支持哪些性能优化验证。学完后，你需要能够：

1. 跟踪 `dvstor_breakdown_benchmark` 从命令行到请求执行的完整路径。
2. 理解 query、insert、mixed workload 的构造方式。
3. 判断 warmup、measure、client threads、read ratio 对结果的影响。
4. 识别现有 benchmark 覆盖不足的场景。

代码入口：

- `tools/dvstor_breakdown_benchmark.cc`
- `tools/breakdown_benchmark/args.hh`
- `tools/breakdown_benchmark/args.cc`
- `tools/breakdown_benchmark/workload.hh`
- `tools/breakdown_benchmark/workload.cc`
- `tools/breakdown_benchmark/progress.hh`
- `tools/run_recall_test.sh`

## 1. benchmark main

`tools/dvstor_breakdown_benchmark.cc` 是 benchmark 可执行入口。主流程：

1. 注册 SIGSEGV handler，发生段错误时输出 backtrace。
2. `parse_args(argc, argv)` 解析 benchmark 自己的参数。
3. `build_service_argv(args.service_config_path)` 从 service config 构造 `IndexConfiguration` 需要的 argv。
4. `configuration::IndexConfiguration config(...)` 创建服务配置。
5. 根据 `config.ip_distance` 选择：
   - `ComputeService<IPDistance>`
   - `ComputeService<L2Distance>`
6. 构造 `ComputeService`。
7. 调用 `run_benchmark(service, args)`。
8. 捕获异常，失败时输出错误并返回 `EXIT_FAILURE`。

重要结论：

- benchmark 不是模拟器，它会真正启动 `ComputeService`。
- 它会连接 memory node、初始化 RDMA/GPU、启动 worker。
- 它测到的是端到端服务路径，而不是单函数 microbenchmark。

## 2. Args 参数模型

`tools/breakdown_benchmark/args.hh` 中 `Args` 包含：

```cpp
std::string service_config_path;
std::string workload{"both"};
size_t warmup_ops{100};
size_t measure_ops{1000};
size_t warmup_seconds{0};
size_t measure_seconds{0};
size_t client_threads{4};
double read_ratio{0.5};
std::string mixed_mode{"probability"};
std::string query_file;
std::string insert_file;
std::string groundtruth_file;
size_t recall_queries{1000};
uint32_t recall_k{0};
double min_recall{-1.0};
bool synthetic{false};
std::string report_json_path;
std::string report_text_path;
uint32_t insert_start_id{0};
double write_insert_ratio{0.5};
double write_upsert_ratio{0.4};
double write_delete_ratio{0.1};
```

参数可以分为六类：

1. service 配置：
   - `service_config_path`

2. workload 类型：
   - `workload`
   - query / insert / both / mixed

3. 运行规模：
   - `warmup_ops`
   - `measure_ops`
   - `warmup_seconds`
   - `measure_seconds`
   - `client_threads`

4. mixed workload：
   - `read_ratio`
   - `mixed_mode`
   - `write_insert_ratio`
   - `write_upsert_ratio`
   - `write_delete_ratio`

5. 数据与 recall：
   - `query_file`
   - `insert_file`
   - `groundtruth_file`
   - `recall_queries`
   - `recall_k`
   - `min_recall`

6. 输出：
   - `report_json_path`
   - `report_text_path`

理解这些参数是做优化实验的前提。尤其是 `workload` 和 `mixed_mode`，它们会改变请求产生方式。

## 3. run_benchmark 的整体结构

`tools/breakdown_benchmark/workload.cc` 中 `run_benchmark()` 是主函数。它做的事情：

1. 构造 JSON root。
2. 写入 meta 信息：
   - workload
   - warmup/measure 参数
   - vector dtype
   - node size
   - search mode
   - credit-aware 参数
   - RaBitQ 参数
   - client threads
   - write mix
3. 准备 insert vectors。
4. 准备 query vectors。
5. 如有 groundtruth，先做性能前 recall check。
6. 根据 workload 执行 warmup。
7. 清空 thread statistics 和 breakdown state。
8. 执行 measure。
9. 收集 breakdown report。
10. 计算 throughput。
11. 如有 groundtruth，做性能后 recall check。
12. 生成 text summary 和 JSON report。
13. 写 report file。
14. 如果 recall 低于阈值，抛异常。

这说明 benchmark 同时承担两个职责：

- 性能测量。
- recall 回归检查。

优化实验时应同时关注两者，避免只看吞吐延迟而牺牲 recall。

## 4. synthetic vector 生成

如果没有提供 insert file，benchmark 用 `make_deterministic_vector(id, dim)` 生成向量。

逻辑：

1. 用 id 和固定常量初始化 64-bit state。
2. 每个维度通过 xorshift-like 更新产生值。
3. 值映射到 `[0, 1)`。
4. 对两个维度加额外偏置：
   - `vector[seed % dim] += 4.0f`
   - `vector[(seed * 17 + 3) % dim] += 1.0f`

这让 synthetic vector 具备一定可区分性，并且可复现。

注意：

- 它不是随机高斯或真实 embedding。
- 它的分布可能和生产数据完全不同。
- 它适合 smoke/perf 对比，但不能代表真实 recall。

## 5. insert data 路径

如果 `insert_file` 非空：

1. `read_vector_rows(args.insert_file)` 读取 binary vector file。
2. `resolve_vector_dtype_config("auto", path)` 判断 dtype。
3. 读取 raw bytes。
4. decode 到 float。
5. 校验 dim 与 service config 一致。

如果没有 insert file：

- 使用 deterministic synthetic vector。

`get_insert_vector(id)` 返回 `vec<element_t>`：

- file 模式：按 `id % insert_rows.count` 选择一行 decoded vector。
- synthetic 模式：按 id 生成。

upsert 使用：

```cpp
get_update_vector(target_id, version)
```

它对 id 和 version 做混合，生成更新后的 vector。

## 6. query data 路径

如果 `query_file` 非空：

1. 检查文件存在。
2. `read_vector_rows(query_file)`。
3. 校验 dim。

如果没有 query file：

1. 根据 workload 和 measure 参数决定 `query_count`。
2. 使用 deterministic vector 生成 synthetic query。
3. 用 `make_float_query_rows(...)` 组织成 raw float32 rows。

query 执行时使用：

```cpp
service.search_raw(query_rows.dtype, query_rows.raw_row(idx), dim, service.config().k)
```

这意味着 benchmark 查询路径默认走 `search_raw()`，可以测试非 float32 query dtype。但如果 service routing enabled，前面课程已经说明 raw query 可能会被 decode 后走 float routing。

## 7. bootstrap 数据

如果 workload 需要 query data 且 service 没有 `load_index`：

1. benchmark 生成 `bootstrap_count` 个 synthetic vectors。
2. 调用 `service.insert(bootstrap_batch)` 插入。
3. 在 meta 中记录 `bootstrap_vectors`。

`bootstrap_count` 的计算：

- time mode：`max(4096, client_threads * 256)`
- ops mode：`max(2048, measure_ops)`

这个设计保证在没有预加载 index 时，query workload 也有基础数据可查。

风险：

- bootstrap 是一次批量 insert，会改变 index 构建状态。
- bootstrap vector 分布是 synthetic。
- 如果你想测纯离线索引查询性能，应使用 `load_index`，不要依赖 bootstrap。

## 8. recall check

如果提供 `groundtruth_file`：

1. 读取 groundtruth bin。
2. 校验 query count。
3. 决定 recall k：
   - 如果 `args.recall_k == 0`，取 `min(service.config().k, gt.top_k)`。
4. 对前 `recall_queries` 个 query：
   - 调用 `service.search_raw(...)`。
   - 取返回 id。
   - 与 groundtruth row 计算 recall。
5. 写入 JSON：
   - phase
   - groundtruth file
   - query count
   - k
   - recall
   - min_recall
   - passed

measure 前做一次 recall：

- key 是 `"recall"`。
- 如果低于 `min_recall`，会设置失败标记。
- 可选择 reset stats。

measure 后做一次 recall：

- key 是 `"static_gt_post_recall"`。
- 不强制 threshold。

后置 recall 对 mixed workload 很重要：如果 measure 中有 insert/upsert/delete，静态 groundtruth 不一定适用，结果只能作为参考。

## 9. query workload

ops mode：

```cpp
for op in [0, ops):
  idx = op % query_count
  service.search_raw(...)
```

seconds mode：

1. 设 deadline。
2. 用平均 query duration 估算是否还能开始下一次操作。
3. 循环调用 search。

当前 query phase 是单线程循环执行。虽然 `client_threads` 出现在 benchmark 参数中，但 query-only 的 `run_query_phase_ops/seconds` 没有启动多个 client threads。多线程主要用于 mixed workload。

这个细节非常重要：如果你以为 query workload 用 `client_threads` 并发压测，会误读结果。要测并发查询，需要检查或扩展 workload 实现。

## 10. insert workload

insert phase 与 query phase 类似：

- ops mode：循环执行 `service.insert(...)`。
- seconds mode：deadline 前持续 insert。

每次 insert 都构造一个单元素 batch：

```cpp
vec<ComputeService<Distance>::InsertItem> insert_items;
insert_items.push_back({id, values});
service.insert(insert_items);
```

因此 benchmark 的 insert granularity 是 single vector。它不会测试大 batch insert 的 amortization。

meta 中也明确记录：

```cpp
"operation_granularity": "single_vector"
```

这对性能解释很重要：如果生产场景使用大 batch，benchmark 的插入性能不一定代表真实吞吐。

## 11. mixed workload

mixed workload 有两种模式：

1. `probability`
   - 每个 client thread 按 `read_ratio` 抽样决定读或写。

2. `fixed_threads`
   - 按 `read_ratio` 将线程固定分成 read threads 和 write threads。

写操作类型由三个 ratio 决定：

- insert
- upsert
- delete

代码会先归一化：

```cpp
normalized_insert_ratio
normalized_upsert_ratio
normalized_delete_ratio
```

mixed ops mode：

- 多线程共享 `next_op`。
- 每个线程从 barrier 同步开始。
- 每个 op 根据模式决定读或写。
- 读调用 `service.search_raw(...)`。
- 写调用 insert/upsert/erase。

mixed seconds mode：

- 多线程跑到 deadline。
- 最后 join。

mixed 是当前 benchmark 中最接近真实并发的路径。

## 12. throughput 计算

measure 后：

```cpp
report = service.collect_breakdown_report()
root.update(report_to_json(report))
```

throughput 对 time mode 有意义：

- `duration_seconds = measure_seconds`
- query ops：
  - mixed：`measure_mixed_stats.completed_reads`
  - 非 mixed：如果 enable_breakdown，用 `report.query.count`，否则用 measured count
- write ops 类似。

如果不是 time mode，throughput duration 为 0，ops/sec 也为 0。

因此：

- ops mode 适合收集 latency/breakdown。
- seconds mode 适合看 throughput。

## 13. report 输出

输出包括：

1. JSON report：
   - meta
   - recall
   - query_breakdown
   - insert_breakdown
   - throughput
   - bottleneck_summary
   - system_counters

2. text report：
   - throughput summary
   - recall summary
   - aggregate text summary

`report_json_path` 是必需输出路径。`report_text_path` 为空时不写 text 文件，但仍会打印 text summary 到 stdout。

## 14. 性能优化验证能力

现有 benchmark 适合验证：

1. 搜索路径 breakdown：
   - RDMA neighbor/vector read。
   - GPU distance。
   - D2H。
   - beam update。
   - RaBitQ gate。

2. 插入路径 breakdown：
   - candidate search。
   - RobustPrune。
   - node write。
   - neighbor lock/write。
   - storage-owner send。

3. mixed 读写干扰：
   - 读写比例。
   - upsert/delete 对查询的影响。

4. recall 回归：
   - 优化前后 recall 是否下降。

5. RaBitQ 和 credit-aware 参数：
   - meta 中会记录对应配置。
   - counters 中有相关指标。

## 15. 覆盖不足

现有 benchmark 不足包括：

1. query-only 不是多 client thread 并发。
2. insert-only 不是多 client thread 并发。
3. insert granularity 固定为单 vector。
4. 不直接测 memory node CPU 使用率。
5. 不测网络层真实带宽、NIC counter。
6. 不覆盖 memory node crash 或部分失败。
7. 不覆盖 load/store 期间请求行为。
8. 不覆盖多 destination result merge，因为当前 routing 本身不 merge。
9. 静态 groundtruth 对 mixed 写入后的 recall 不一定有效。
10. synthetic data 分布不代表真实 embedding。

## 16. 实验建议

如果你准备优化某个路径，建议至少跑：

1. query ops mode：
   - 收集 breakdown 和 latency percentiles。

2. query seconds mode：
   - 收集 throughput。

3. mixed seconds mode：
   - 看读写干扰。

4. recall before/after：
   - 用真实 query 和 groundtruth。

5. 参数 sweep：
   - beam width
   - expansion batch
   - query batch size
   - rdma read batch mode
   - qp pool size
   - RaBitQ gate width

每次实验必须保存：

- service config。
- benchmark args。
- git commit 或 diff。
- report JSON。
- memory node 数、compute node 数、GPU 型号、NIC 型号。

## 17. 学习任务

1. 画一张 benchmark 参数到实际请求流的映射表。
2. 找出 query-only 为什么不是多线程并发，并设计一个扩展方案。
3. 用 report JSON 列出优化前后必须比较的字段。
4. 设计一个 mixed workload：80% read、20% write，其中写入按 insert/upsert/delete=2/7/1。
5. 设计一个 recall 回归流程，要求性能优化后 recall 不能低于 baseline 的 99%。

