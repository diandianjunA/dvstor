# 第 30 课：Breakdown Benchmark 与实验脚本

本课是整个 30 课教程的最后一课，定位在“如何把系统跑起来并采集可信的实验数据”。前 29 课分别拆解了传输层、索引格式、GPU 持久化引擎、存储节点、计算服务、离线构建等子系统的代码实现，但一个工程上可复现的向量检索系统不止需要正确代码，还需要一套**可重复执行的负载脚本**与**不会自欺欺人的报表**：否则任何性能数字都既无法被自己复现，也无法被他人审视。本课讲解 `dvstor` 仓库 `tools/breakdown_benchmark/` 下的 benchmark 框架与 `experiment/` 目录下的实验脚本，这两部分共同回答“一次端到端实验从索引构建到报告对比应该怎么做”。

## 本课目标与涉及文件

读完本课你应该能够：

1. 理解 `dvstor_breakdown_benchmark` 的入口、参数解析、`ComputeService` 的构造方式，以及它如何把一个 `.ini` 服务配置文件转成 `argv`。
2. 掌握 `workload.cc` 中 query / insert / mixed 三类负载的 warmup-measure 双阶段编排，理解 `single_pass_no_reuse` 查询游标、`PacedOperationDispatcher` 限速器、固定读写线程模式与概率读写模式的差异。
3. 看懂 recall 校验流程（含 `base_only` 模式与“前后两次 recall”），以及 GPU breakdown 报表、`stage2` 存储侧 maintenance 日志聚合、stability 稳定性窗口的产出方式。
4. 读懂 `report.cc` / `progress.cc` / `maintenance_log.cc` / `dataset.cc` 四个支撑模块的职责边界。
5. 逐段理解 `experiment/*.sh` 与 `profiles/*.env` 的环境变量、流程与“`pkill` 自杀陷阱”这类工程坑，能够独立完成 build → start storage → run recall → run breakdown → compare 的完整链路。
6. 明白验收一份报告时应该看哪些原始字段（`direct_path_failures`、recall 变化、`zero_completion_windows`、`average_batch_size` 等），以及为什么脚本只记录原始结果、不做通过/失败判断。

本课涉及的全部源码文件（路径均为绝对路径）：

- 入口与编排：
  - `/home/xjs/experiment/dvstor/tools/dvstor_breakdown_benchmark.cc`
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/workload.cc`
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/workload.hh`（结构声明，与 `args.hh` 配套）
- 参数与配置：
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/args.cc`、`args.hh`
- 报表与遥测：
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/report.cc`、`report.hh`
- 进度与限速器：
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/progress.cc`、`progress.hh`
- 存储侧 maintenance 日志聚合：
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/maintenance_log.cc`、`maintenance_log.hh`
- 数据集读取：
  - `/home/xjs/experiment/dvstor/tools/breakdown_benchmark/dataset.cc`、`dataset.hh`
- 长插入 recall 与数据生成工具：
  - `/home/xjs/experiment/dvstor/tools/dvstor_sift101m_long_insert_recall.cc`
  - `/home/xjs/experiment/dvstor/tools/generate_sift101m_recall_data.cc`
- 实验脚本：
  - `/home/xjs/experiment/dvstor/experiment/common.sh`
  - `/home/xjs/experiment/dvstor/experiment/sift100m_common.sh`
  - `/home/xjs/experiment/dvstor/experiment/profiles/04_gpu_persistent_gpunetio.env`
  - `/home/xjs/experiment/dvstor/experiment/run_breakdown.sh`
  - `/home/xjs/experiment/dvstor/experiment/run_recall.sh`
  - `/home/xjs/experiment/dvstor/experiment/start_memory_node.sh`
  - `/home/xjs/experiment/dvstor/experiment/start_all_memory_nodes.sh`
  - `/home/xjs/experiment/dvstor/experiment/stop_memory_nodes.sh`
  - `/home/xjs/experiment/dvstor/experiment/build_sift100m_index.sh`
  - `/home/xjs/experiment/dvstor/experiment/compare_reports.py`
  - `/home/xjs/experiment/dvstor/experiment/README.md`

下面按文件逐段讲解。

## 一、入口：`tools/dvstor_breakdown_benchmark.cc`

这是 benchmark 进程的 `main`，体量很小（45 行），但交代了三件关键事项。

```cpp
void segfault_handler(int signal) {
  void* frames[64];
  const int count = backtrace(frames, 64);
  const char header[] = "\n[breakdown] fatal signal, backtrace:\n";
  const ssize_t ignored = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  _exit(128 + signal);
}
```

`segfault_handler`（`tools/dvstor_breakdown_benchmark.cc:14`）注册在 `SIGSEGV` 上。benchmark 是一个把 GPU kernel、RDMA、多线程客户端揉在一起的高并发进程，一旦崩溃，最重要的不是错误码而是**栈**。这里用 `backtrace_symbols_fd` 直接写到 `STDERR_FILENO`，再 `_exit(128 + signal)`。之所以用 `::write` 而不是 `std::cerr`，是因为信号处理函数里不能调用非异步信号安全的函数；`write` 是安全的，`iostream` 不是。`_exit` 而不是 `exit` 同样是为了避免运行 atexit 析构（那可能再次崩溃）。这个模式在 `dvstor_sift101m_long_insert_recall.cc:106` 的 `signal_handler` 中会再次出现，差别是后者额外捕获了 `SIGABRT`。

```cpp
int main(int argc, char** argv) {
  signal(SIGSEGV, segfault_handler);
  try {
    const Args args = parse_args(argc, argv);
    auto service_args = build_service_argv(args.service_config_path);
    auto service_argv = make_argv(service_args);
    configuration::IndexConfiguration config(
      static_cast<int>(service_argv.size()), service_argv.data());
    ComputeService service(config);
    (void)run_benchmark(service, args);
  } catch (const std::exception& e) {
    std::cerr << "breakdown benchmark failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
```

`main` 的关键流程是：`parse_args` 解析 benchmark 自己的命令行（负载参数、报告路径），`build_service_argv` 把 `--service-config` 指向的 `.ini` 文件**展开成 `ComputeService` 的命令行参数**，再交给 `configuration::IndexConfiguration` 解析。这一点很容易被忽略：benchmark 本身**不是**一个 RPC 客户端，它直接在同一进程内构造 `ComputeService`（`tools/dvstor_breakdown_benchmark.cc:36`），所以它和第 27 课（见第 27 课）讲的计算服务主体共享同一份代码、同一个 `IndexConfiguration`、同一份 GPU/RDMA 运行时。benchmark 客户端线程调用 `service.search_raw` / `service.insert` 等公共 API 时，就和真实计算节点接收外部请求时的代码路径完全一致。这种“in-process benchmark”设计避免了 IPC 噪声，也让 breakdown 采样可以直接在进程内取。

`run_benchmark` 的返回值 `nlohmann::json` 被 `(void)` 丢弃——它的副作用是把 JSON 报表写到 `args.report_json_path`、把文本报表写到 `args.report_text_path` 并打印到 `stdout`，所有这些都在 `workload.cc` 内部完成。

## 二、参数解析：`tools/breakdown_benchmark/args.cc` / `args.hh`

`Args` 结构体（`args.hh:14`）是 benchmark 的全部配置面，字段大致可分四组：

1. **服务接入**：`service_config_path`（必填，指向 `.ini`）。
2. **负载形状**：`workload`（`query`/`insert`/`both`/`mixed`，默认 `both`）、`warmup_ops`/`measure_ops`（按次数跑）、`warmup_seconds`/`measure_seconds`（按时间跑，二者必须同时为正）、`client_threads`。
3. **混合负载策略**：`mixed_mode`（`probability`/`fixed_threads`/`rate_limited`）、`read_ratio` ∈ [0,1]、`target_query_qps`/`target_write_qps`、写操作三比例 `write_insert_ratio`/`write_upsert_ratio`/`write_delete_ratio`。
4. **数据文件与 recall**：`recall_query_file`、`performance_query_file`、`insert_file`、`groundtruth_file`、`recall_queries`、`recall_k`、`recall_mode`（`all`/`base_only`）、`recall_base_id_limit`、`storage_maintenance_logs`、`recall_only`、`insert_start_id`。
5. **输出**：`report_json_path`（必填）、`report_text_path`。

### 2.1 `.ini` 配置到 argv 的转换

`read_config`（`args.cc:22`）是一个最小的 INI 解析器。它逐行读取：

```cpp
const auto comment_pos = line.find_first_of("#;");
if (comment_pos != std::string::npos) {
  line.erase(comment_pos);
}
```

`#` 和 `;` 都被视为注释起始；这是为了兼容 INI 与 shell 两种注释风格。空行跳过，`key=value` 通过 `find('=')` 拆分，两侧 `trim` 后写入 `ConfigMap`。

`build_service_argv`（`args.cc:78`）把 `ConfigMap` 转成 `ComputeService` 能识别的 `--key value` 序列，它把键分为三类：

- `multi_keys`（`args.cc:83`）：`servers`、`clients`、`storage-peers`。这些键的值是空格或逗号分隔的多 token，调用 `split_tokens` 把 `,` 替换成空格再 `stringstream` 切分，最终展开成 `--servers host1:port1 host2:port2 ...`。
- `flag_keys`（`args.cc:84`）：`initiator`、`disable-thread-pinning`。这些键只在值为 `1/true/on/yes`（`is_truthy`，`args.cc:53`）时追加裸 `--flag`，不追加值。
- `benchmark_only_keys`（`args.cc:86`）：`insert-start-id`、`write-id-base`。这两个键**只供 benchmark 使用**，不传给 `ComputeService`，所以直接 `continue` 跳过。
- 其余键一律按 `--key value` 追加。

`make_argv`（`args.cc:69`）把 `vector<string>` 转成 `vector<char*>`，因为 `IndexConfiguration` 接收的是传统的 `int argc, char** argv`。注意它返回的 `argv` 引用了 `args` 内部 `string` 的 `data()`，所以调用方必须保证 `args` 的生命周期长于 `argv` 的使用——`main` 里 `service_args` 和 `service_argv` 都在同一作用域，安全。

### 2.2 命令行解析

`parse_args`（`args.cc:117`）用一个 `for` 循环 + `require_value` lambda 手写解析。每个 `--flag` 对应一个分支，没有用第三方库（如 gflags），原因是 benchmark 参数面不大、且需要精细控制校验顺序。几个值得注意的细节：

- `--query-file` 是 `--recall-query-file` 的旧别名，二者不能同时指定（`args.cc:161`）。
- `--target-write-qps` 和 `--target-insert-qps` 是同一参数的两个名字（`args.cc:148`），后者是为了兼容旧脚本。
- `--storage-maintenance-log` 可重复出现，追加到 `vector`（`args.cc:181`），脚本里就是为每个分片各传一次。
- `--recall-base-id-limit` 被显式限制在 `uint32_t` 范围（`args.cc:175`）。

解析完后的校验（`args.cc:205` 起）非常重要，它们是“脚本只记录原始结果、不自动验收”这条原则的前置守门：

```cpp
if (args.read_ratio < 0.0 || args.read_ratio > 1.0) {
  throw std::runtime_error("--read-ratio must be in [0, 1]");
}
if (args.write_insert_ratio + args.write_upsert_ratio + args.write_delete_ratio <= 0.0) {
  throw std::runtime_error("at least one write mutation ratio must be > 0");
}
```

写操作三比例的**绝对值**无关，`workload.cc` 内部会归一化（见后文），但至少要有一个为正。`recall_mode == "base_only"` 必须同时给出正的 `recall_base_id_limit`（`args.cc:256`），且必须有 groundtruth——`base_only` 模式用于评估“只看 base ID 范围内的召回”，避免在线 insert 产生的 ID 污染 recall 评估。

`rate_limited` 模式有最强约束（`args.cc:309`）：必须 `workload == mixed` 且用时间模式，且 `target_query_qps` 与 `target_write_qps` 至少一个为正；反之非 `rate_limited` 模式下这两个 target 必须为 0（`args.cc:318`），避免静默误用。

`insert_start_id` 有三级回退（`args.cc:295`）：命令行 > 配置文件 `insert-start-id` > 配置文件 `write-id-base` > 0（0 会在 `workload.cc` 内部根据 `max_vectors` 推算默认值）。

最后是“recall 与 performance 查询文件必须不同”的硬约束（`args.cc:277`、`args.cc:286`），用 `std::filesystem::equivalent` 比较 inode。这是为了杜绝“用 10K recall 查询集跑吞吐”这种自欺欺人的做法——recall 查询集只有 1 万行，跑吞吐会瞬间耗尽或反复命中同一批向量缓存，性能数字完全不可信。

## 三、工作负载编排：`tools/breakdown_benchmark/workload.cc`

这是 benchmark 的核心，1247 行，包含三类负载（query / insert / mixed）、两套时序（按 ops / 按 seconds）、recall 校验、GPU publication 等待、stage2 日志聚合、吞吐与稳定性统计。函数入口是 `run_benchmark`（`workload.cc:60`）。

### 3.1 元信息收集与 meta JSON

函数开头先决定 `workload_has_queries`（`workload.cc:65`）：

```cpp
const bool workload_has_queries =
  !args.recall_only &&
  (args.workload == "query" || args.workload == "both" ||
   (args.workload == "mixed" &&
    (args.mixed_mode == "rate_limited" ? args.target_query_qps > 0.0
                                       : args.read_ratio > 0.0)));
```

这个布尔量决定后续是否加载 performance 查询文件、是否运行 query 阶段。注意 `rate_limited` 模式看 `target_query_qps`，其它模式看 `read_ratio`——因为 `rate_limited` 模式下读写比例由两个 target QPS 决定，`read_ratio` 不再有意义。

随后 `root["meta"]`（`workload.cc:73`）把负载形状、向量 dtype、节点大小、GPU 参数等全部记录进 JSON。这一段看似平淡，但它是报告可读性的基石：任何一份 JSON 报表都自带“我是用什么参数跑出来的”完整声明，不需要翻日志。其中几条值得点出：

- `VamanaNode::vector_dtype_name()` / `vector_bytes()` / `total_size()`（`workload.cc:88-91`）：这些来自第 6 课（见第 6 课）讲的 `VamanaNode` 类型，反映了 schema-15 节点布局。
- `time_issue_policy`（`workload.cc:82`）：根据 `mixed_mode` 取 `fixed_read_write_threads_until_deadline`、`shared_two_stream_pacer_until_deadline` 或 `probabilistic_read_write_per_thread_until_deadline`。这三个字符串会出现在报告里，让人一眼看出限速策略。
- `effective_bytes_per_vector`（`workload.cc:93`）：声明“按单向量粒度计费”，方便换算带宽。

接下来归一化写操作比例（`workload.cc:124`）：

```cpp
const double write_ratio_sum = args.write_insert_ratio + args.write_upsert_ratio + args.write_delete_ratio;
const double normalized_insert_ratio = args.write_insert_ratio / write_ratio_sum;
const double normalized_upsert_ratio = args.write_upsert_ratio / write_ratio_sum;
const double normalized_delete_ratio = args.write_delete_ratio / write_ratio_sum;
```

后面 `choose_write_kind`（`workload.cc:575`）用一个 `uniform_real_distribution<double>(0.0, 1.0)` 把 `[0,1)` 区间按三个归一化比例切分，决定每次写是 insert / upsert / erase。

`fixed_threads` 模式下还会预先算好读写线程数（`workload.cc:133`）：

```cpp
if (args.read_ratio <= 0.0) {
  fixed_read_threads = 0;
} else if (args.read_ratio >= 1.0) {
  fixed_read_threads = args.client_threads;
} else {
  fixed_read_threads = static_cast<size_t>(std::llround(static_cast<double>(args.client_threads) * args.read_ratio));
  fixed_read_threads = std::clamp<size_t>(fixed_read_threads, 1, args.client_threads - 1);
}
fixed_write_threads = args.client_threads - fixed_read_threads;
```

`clamp` 保证至少 1 个读线程和 1 个写线程，避免 `read_ratio` 过小或过大时某一类线程数为 0。

### 3.2 数据加载与 `SinglePassRowStream`

`get_insert_vector`（`workload.cc:172`）是 insert 向量来源的抽象：若指定了 `--insert-file`，则从文件按 `id % insert_rows.count` 取行（允许重复，因为 insert 文件通常只有几百万行而 benchmark 可能要插入更多）；否则用 `make_deterministic_vector(id, dim)`（`dataset.cc:10`）合成。合成向量基于 xorshift64* 伪随机数，并在 `seed % dim` 和 `(seed*17+3) % dim` 两个位置分别加 4.0 和 1.0，保证每个 ID 有可区分的“尖峰”，避免所有向量过于均匀导致 ANN 退化。

`get_update_vector`（`workload.cc:182`）用 `target_id ^ (0x9e3779b9u * (version + 1u))` 作为 seed 合成 upsert 向量，`0x9e3779b9` 是黄金分割常量，确保不同 version 的向量差异足够大。

performance 查询数据的加载在 `workload.cc:313`：调用 `read_vector_rows(args.performance_query_file, false)`，`false` 表示不解码（performance 查询直接以原始 dtype 喂给 `service.search_raw`，不需要 float 中间态）。日志里特意打了 `policy=single_pass_no_reuse`（`workload.cc:322`），这是 benchmark 的核心设计原则之一：**查询游标永不回绕**。`SinglePassRowStream`（`dataset.cc:80`）实现如下：

```cpp
std::optional<size_t> SinglePassRowStream::try_claim() {
  if (exhausted_.load(std::memory_order_acquire)) return std::nullopt;
  const size_t row = next_row_.fetch_add(1, std::memory_order_relaxed);
  if (row >= row_count_) {
    exhausted_.store(true, std::memory_order_release);
    return std::nullopt;
  }
  return row;
}
```

这是一个无锁的原子行号分配器：每个线程 `fetch_add` 拿到唯一行号，行号超过 `row_count_` 即视为耗尽。一旦耗尽，`exhausted_` 置位，后续 `try_claim` 立即返回 `nullopt`。`workload.cc:366` 的 `throw_if_performance_queries_exhausted` 会在每个阶段结束后检查：如果游标在阶段中途耗尽，直接抛异常终止 benchmark，而不是悄悄回绕或停止计数。`rate_limited` 模式还会在加载时预估所需行数（`workload.cc:324`）：

```cpp
const uint64_t required_rows = PacedOperationDispatcher::scheduled_count(
    args.target_query_qps, args.warmup_seconds) +
  PacedOperationDispatcher::scheduled_count(
    args.target_query_qps, args.measure_seconds);
if (required_rows > performance_query_rows.count) {
  throw std::runtime_error("rate-limited workload requires ... rows but the file contains ...");
}
```

提前报错比跑到一半耗尽好得多。

### 3.3 recall 校验流程

`run_recall_check`（`workload.cc:377`）是一个 lambda，被调用两次：一次在 performance 阶段之前（`workload.cc:492`，键 `"recall"`），一次在之后（`workload.cc:1189`，键 `"static_gt_post_recall"`）。前者反映“未受性能负载影响”的召回，后者反映“经历 warmup + measure 后”的召回；两者的差值能暴露在线 insert/upsert/delete 是否破坏了搜索质量。

它的流程：

1. 读 groundtruth（`workload.cc:386`），校验行数与 recall 查询集一致。
2. 计算 `recall_k`：若用户未指定，取 `min(service.config().k, gt.top_k)`（`workload.cc:390`）。
3. `base_only` 模式下，`recall_search_width` 取 `gpu_final_rerank_width`（`workload.cc:398`），并校验每个 groundtruth ID 都在 `recall_base_id_limit` 之内（`workload.cc:404`）——否则 `base_only` 评估本身就无意义。`all` 模式下 `recall_search_width = recall_k`，即直接用 `k` 作为搜索宽度。
4. 多线程跑 recall：每个 worker 用 `next_recall_query.fetch_add` 抢查询索引，调用 `service.search_raw` 拿回 `vec<node_t>`，再用 `recall_at`（`dataset.cc:134`）算单个查询的召回率。`base_only` 模式会先用 `filter_base_only_recall_ids`（`report.cc:24`）过滤掉 `>= base_id_limit` 的 ID，若过滤后不足 `recall_k` 则记一次 `insufficient_base_results`。
5. 汇总：`root[key]` 写入 `recall`、`queries`、`mode`、`base_id_limit`、`search_result_width`、`queries_with_insufficient_base_results`、`result_set_complete`、`recall`。`result_set_complete` 是 `insufficient_queries == 0`，这是验收时必须为 `true` 的字段之一。
6. 若 `reset_after` 为真，调用 `service.clear_thread_statistics()` 与 `service.reset_breakdown_state()`（`workload.cc:488`），把 recall 期间的统计从后续 measure 阶段剥离。第一次调用 `reset_after=true`，第二次 `reset_after=false`（因为 benchmark 即将结束）。

`recall_at`（`dataset.cc:134`）实现简单：把 groundtruth 前 `k` 个 ID 放进 `unordered_set`，再统计结果中在前 `k` 个的命中数。注意它只看前 `min(results.size(), k)` 个结果，所以结果不足 `k` 会直接降低召回率——这就是 `insufficient_base_results` 必须为 0 的原因。

### 3.4 insert 阶段

`run_insert_phase_ops`（`workload.cc:192`）是按次数跑的 insert 阶段。它用 `std::barrier` 让所有 worker 同时开始（`workload.cc:202`），每个 worker 用 `next_op.fetch_add` 抢任务，构造单条 `InsertItem` 调用 `service.insert`。如果 `insert` 返回值不为 1（即被拒绝），抛 `"singleton insert was rejected"`。`failed` 原子量让其它 worker 在下一次循环退出，`error` 保存第一个异常，最终 `std::rethrow_exception` 抛出。`ProgressReporter` 在后台线程每 5 秒打印进度，并在 `label` 以 `"measure-"` 开头时收集 `measure_windows`（`workload.cc:234`）。

`run_insert_phase_seconds`（`workload.cc:238`）是按时间跑的版本，多了一个 `can_start_timed_operation` 判断（`progress.cc:13`）：

```cpp
bool can_start_timed_operation(const std::chrono::steady_clock::time_point deadline,
                               const std::chrono::nanoseconds avg_duration,
                               size_t completed_ops) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return false;
  if (completed_ops == 0 || avg_duration.count() <= 0) return true;
  const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
  return remaining >= avg_duration;
}
```

这是一个“提前停止”启发式：用滑动平均的 insert 耗时估计“现在开始一个新 insert 能否在 deadline 前完成”，若不能就不开始。`update_avg_duration`（`progress.cc:28`）用 `(old*7 + new) / 8` 的指数滑动平均，避免长尾样本扰动估计。这保证了 `measure_seconds` 结束时**没有在途 insert**，从而吞吐统计的分子分母一致。注意它不是强制的——即使估计能完成，实际可能超时；真正的 deadline 检查在 `workload.cc:263`：`if (std::chrono::steady_clock::now() >= deadline) break;`。`drain_seconds`（`workload.cc:287`）记录 deadline 之后到所有 worker join 的时间，这是“排水”时间，会写进报告的 `client_drain_seconds`。

### 3.5 query 阶段

`run_query_phase_ops`（`workload.cc:494`）与 `run_query_phase_seconds`（`workload.cc:523`）结构对称。核心循环：

```cpp
for (;;) {
  if (performance_query_stream.exhausted()) break;
  const size_t op = next_op.fetch_add(1, std::memory_order_relaxed);
  if (op >= ops) break;
  const auto query_row = performance_query_stream.try_claim();
  if (!query_row.has_value()) break;
  (void)service.search_raw(
    performance_query_rows.dtype,
    performance_query_rows.raw_row(*query_row), dim, service.config().k);
  completed_ops.fetch_add(1, std::memory_order_relaxed);
}
```

三重退出条件：游标耗尽、达到 op 上限、`try_claim` 失败（也是游标耗尽）。`search_raw` 的返回值被 `(void)` 丢弃——query 阶段只关心吞吐和延迟，不关心结果（结果正确性由 recall 阶段验证）。`raw_row`（`dataset.cc:36`）返回 `raw.data() + index * vector_bytes`，零拷贝。

时间模式下（`workload.cc:523`），所有 worker 在 `start_barrier` 等待主线程设置 deadline 后同时开始（`workload.cc:549`），这是为了让“所有线程同时施压”这个条件可复现。

### 3.6 mixed 阶段与三种调度模式

mixed 阶段是 benchmark 最复杂的部分，它有三种调度模式：

**`probability` 模式**（`run_mixed_phase_ops` / `run_mixed_phase_seconds` 的非 `fixed_threads` 分支）：每个线程每次循环用 `choose_mixed_read(rng)`（`workload.cc:563`）按 `read_ratio` 伯努利抽样决定读写，写操作再按归一化比例选 insert/upsert/erase。每个线程有独立 `mt19937_64`，seed 由 `tid`、`label` 哈希与一个常数异或得到（`workload.cc:657`），保证可复现且线程间不相关。

**`fixed_threads` 模式**：线程按 `tid < fixed_read_threads` 静态划分读写职责（`workload.cc:667`）。读线程只跑查询，写线程只跑写。这种模式适合评估“读写在各自独占线程上的极限吞吐”，避免了 probability 模式下线程频繁切换读写角色带来的缓存抖动。

**`rate_limited` 模式**（`workload.cc:734` 起）：用 `PacedOperationDispatcher` 做共享双流限速。`PacedOperationDispatcher`（`progress.cc:42`）维护两个 `Stream`（query / write），每个流有 `rate` 和 `next_ordinal`。`claim`（`progress.cc:79`）在锁内计算两个流下一次“应触发”的时间（`scheduled_at`，`progress.cc:67`）：

```cpp
const long double seconds =
  static_cast<long double>(stream.next_ordinal) /
  static_cast<long double>(stream.rate);
return start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<long double>(seconds));
```

即第 `n` 个操作应在 `start + n/rate` 时刻触发。`claim` 选两个流中较早的那个，`next_ordinal++`，然后 `sleep_until(claim.scheduled_at)`。这种设计让多线程共享一个限速器：哪个线程先拿到锁就先 claim，但实际执行时间由 `scheduled_at` 决定，从而精确控制总 QPS。`scheduled_count`（`progress.cc:119`）返回 `ceil(rate * seconds)`，用于预估所需查询行数。

`issue_mixed_write`（`workload.cc:590`）是写操作的统一入口，按 `choose_write_kind` 分派：

- `insert`：`next_insert_id.fetch_add` 拿新 ID，调 `service.insert`。
- `upsert`：`sample_existing_id` 在 `[0, max_vectors)` 随机选 ID，`next_update_version.fetch_add` 拿版本号，合成新向量调 `service.upsert`。
- `erase`：`sample_existing_id` 选 ID 调 `service.erase`。

注意 upsert/erase 的目标 ID 是**随机采样**的，不是遍历，这模拟了真实工作负载中“热点 ID 被反复修改”的场景。`issue_mixed_write` 返回 `bool` 表示是否成功，失败（如 insert 被 mutation capacity 拒绝）只计 issued 不计 completed，不抛异常——mixed 负载容忍个别写失败，但会在报告里体现为 `issued/completed` 差值。

mixed 阶段结束后有一段强校验（`workload.cc:1001`）：

```cpp
if (reads_expected) {
  lib_assert(measure_mixed_stats.completed_reads > 0, "mixed benchmark completed zero reads");
}
if (writes_expected) {
  lib_assert(measure_mixed_stats.completed_writes > 0, "mixed benchmark completed zero writes");
}
```

`lib_assert` 在条件不满足时抛异常，这能捕获“配置错了但没报错”的情况，比如 `read_ratio=0.5` 但实际完成的读为 0（可能是 performance query 文件为空或游标立即耗尽）。

### 3.7 GPU publication 等待与 stage2 日志聚合

measure 阶段结束后，benchmark 不会立即生成报告，而是先做两件收尾：

**等待 GPU mutation publication 排空**（`wait_for_gpu_publications`，`workload.cc:844`）：

```cpp
for (;;) {
  const auto telemetry = service.gpu_search_telemetry();
  if (telemetry.mutation_capacity_reserved == 0) {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  }
  if (std::chrono::steady_clock::now() >= deadline) {
    throw std::runtime_error(
      std::string{"GPU mutation publication did not drain during "} + phase);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
```

`mutation_capacity_reserved` 是 GPU 侧为未发布的 mutation 预留的容量（见第 10 课 delta/动态路由/预算）。如果它不为 0，说明还有 insert/upsert 没有发布到 GPU 搜索路径，此时采集的 GPU 遥测不准。超时时间取 `storage_owner_rpc_timeout_ms * 3`，clamp 到 `[5000, 60000]` 毫秒。这个等待在 warmup 后（`workload.cc:896`）和 measure 后（`workload.cc:949`）各做一次，分别保证 recall 与 measure 遥测的干净。

**stage2 maintenance 日志聚合**：`snapshot_maintenance_logs`（`maintenance_log.cc:380`）在每个 storage 节点的 maintenance 日志文件上记录当前文件大小作为 cursor，并在读到的最后一条 observation 作为 baseline。measure 结束后再 snapshot 一次，调 `summarize_maintenance_log_window`（`maintenance_log.cc:416`）计算窗口内的 delta。这部分稍后在 maintenance_log 节细讲。

### 3.8 吞吐与稳定性统计

吞吐计算（`workload.cc:1026` 起）有几个细节值得点出。首先是 `throughput_duration` 的定义：

```cpp
const double throughput_duration = has_throughput_duration
  ? static_cast<double>(args.measure_seconds) + measure_client_drain_seconds
  : 0.0;
```

有效测量时长 = 配置时长 + 客户端排水时长。这是因为即使 `can_start_timed_operation` 做了提前停止，仍可能有少量在途操作在 deadline 后完成，它们的完成时间应该计入分母，否则吞吐会被高估。但对 `rate_limited` 模式，吞吐用配置时长作分母（`workload.cc:1049`）：

```cpp
const double query_throughput = rate_limited_measurement
  ? static_cast<double>(throughput_query_ops) / configured_measure_duration
  : effective_query_throughput;
```

因为 `rate_limited` 模式的目标就是“在配置窗口内完成 N 个操作”，用配置时长作分母更能反映“是否达成了目标 QPS”。`query_rate_attainment_ratio`（`workload.cc:1081`）= `实际完成 / 计划调度`，是限速模式的核心验收指标。

稳定性统计（`workload.cc:1100` 起）把 measure 窗口内的 `ProgressSample` 序列转成 JSON，并计算 head/tail QPS。`stability_window_deadline = measure_seconds + 0.5`（`workload.cc:1121`），超出的样本不纳入稳定性统计——这是为了避免“deadline 后的排水间隔”被误判为“负载尾部”。`edge_mean`（`workload.cc:1151`）取前 3 或后 3 个有效窗口的均值，`tail_to_head_ratio` 反映负载是否衰减：接近 1 说明稳态，明显小于 1 说明系统在 measure 后期变慢（可能是 GC、cache miss、mutation 积压）。`zero_completion_windows` / `zero_query_windows` / `zero_write_windows` 统计“某窗口内完成数为 0”的次数，理想情况全为 0。

最后是 `format_report`（`workload.cc:1227`）生成文本报告，`bottleneck_summary` 字段（`workload.cc:1228`）来自 `service::breakdown::aggregate_text_summary`，即第 27 课（见第 27 课）讲的细粒度 breakdown 文本。JSON 与文本分别写到 `report_json_path` 与 `report_text_path`，文本还打印到 `stdout`。

## 四、报表：`tools/breakdown_benchmark/report.cc` / `report.hh`

`report.cc` 有三个职责：路径规范化、recall ID 过滤、遥测转 JSON 与文本格式化。

### 4.1 路径规范化与 recall 过滤

`normalize_path`（`report.cc:11`）用 `std::filesystem::absolute` + `weakly_canonical` 把相对路径转成规范绝对路径，失败时退化到 `lexically_normal`。这个函数用于 meta JSON 里的 `index_prefix`、`performance_query.canonical_source` 等字段，保证不同机器、不同 CWD 下跑出的报告里路径可比较。

`filter_base_only_recall_ids`（`report.cc:24`）在 `base_only` recall 模式下使用：把 `>= base_id_limit` 的 ID 过滤掉，只保留前 `result_limit` 个。这个过滤发生在搜索结果已经返回之后，所以它不改变搜索行为，只改变 recall 计算。

### 4.2 `telemetry_to_json`：GPU 遥测全量序列化

`telemetry_to_json`（`report.cc:38`）把 `gpu_search::TelemetrySnapshot`（见第 9 课 GPU 类型/遥测/PQ 模型）逐字段转成 JSON，共约 70 个字段。几组关键字段：

- **GPU 内存**（`report.cc:40-46`）：`gpu_memory_explicit_bytes`（显式分配）、`base_pq`（base 向量 PQ 码）、`resident_pq`（常驻 dynamic PQ）、`route_graph`（路由图）、`delta_reserved`（delta 预留）、`graph_cache` / `exact_cache`（图/精确向量缓存）。这些字段直接对应报告文本里的 `GPU memory ... GiB` 行，是判断内存水位的核心。
- **批量与提交**（`report.cc:47-56`）：`queries_submitted` / `queries_completed` / `batches` / `batch_queries`，`average_batch_size = batch_queries / batches`。`average_batch_size` 是验收字段之一——理想情况下应远大于 1（多查询并发），若接近 1 说明查询串行化，GPU 持久化 kernel 没有发挥批量优势。
- **GPU 阶段耗时**（`report.cc:58-80`）：`gpu_query_residence_ns`、`gpu_prepare_ns`、`gpu_graph_ns`、`gpu_score_ns`、`gpu_beam_ns`、`gpu_exact_ns`、`gpu_delta_scan_ns`，每个都除以 `queries_completed` 得到平均微秒。这些字段对应 GPU kernel 内部各阶段（见第 18-20 课候选评分与查询遍历主循环），能定位瓶颈在图遍历、评分还是精确重排。
- **RDMA**（`report.cc:81-90`）：`rdma_read_ops`、`rdma_read_bytes`、`rdma_merged_requests`、`direct_path_failures`。`direct_path_failures` 是**最重要的验收字段**，必须为 0——它记录了 RDMA direct path（GPU 直接读远端内存）失败的次数，非零说明 GPUNetIO 路径有问题（见第 22 课 GPUNetIO 传输/probe）。
- **图缓存与路由**（`report.cc:85-110`）：`graph_page_requests`、`graph_page_cache_hits`、`graph_route_hits`、`graph_route_refreshes`、`dynamic_route_publications` / `slot_updates` / `live_slots` / `snapshot_skips`。这些字段对应第 10 课动态路由与第 19 课 RDMA cache。
- **delta 与 mutation**（`report.cc:119-170`）：`delta_queries`、`delta_scan_records`、`mutations_published`、`delta_publications`、`delta_reclaim_batches`、`delta_entries_retired`、`mutation_capacity_rejections` / `wait_events` / `wait_ns` / `reserved` / `reserved_max`、`visibility_ns_total` / `max`。`average_visibility_us` 是 insert 到查询可见的平均延迟，反映动态路由发布的实时性。

### 4.3 `format_report`：文本报告

`format_report`（`report.cc:173`）生成人类可读的文本报告，结构对应 JSON 的 `throughput` / `stage2` / `recall` / `static_gt_post_recall` / `gpu_persistent` / insert breakdown / query breakdown。几处实现细节：

- `bytes_per_gib = 1024.0 * 1024.0 * 1024.0`（`report.cc:312`），用 `static_cast<double>` 转换避免整数溢出。
- GPU 内存行（`report.cc:314`）把 7 个内存字段拼成一行，用 `/` 分隔，方便 diff。
- `stage2` 段（`report.cc:246`）只在 `requested_logs != 0` 时输出，即只有传了 `--storage-maintenance-log` 才会有。`p99_stitch_delay_available` 为 false 时打印 `unavailable`，而不是打印 0——避免误导。
- `recall` 与 `static_gt_post_recall` 段（`report.cc:280`、`report.cc:295`）格式相同，`mode/base_id_limit/search_width/insufficient_queries` 四个值用 `/` 拼在一行。
- 最后调用 `service::breakdown::aggregate_text_summary(report.insert)` 与 `report.query`（`report.cc:381-388`），把第 27 课的细粒度 breakdown 文本追加进来。这两段文本就是报告结尾的 `insert breakdown` / `query breakdown` 块，包含 `count`、`latency_ms: mean/p50/p95/p99`、`cpu_ms`、`rdma_ms`。

## 五、进度报告与限速器：`tools/breakdown_benchmark/progress.cc` / `progress.hh`

这个文件有三个独立组件：`can_start_timed_operation` / `update_avg_duration`（已在 3.4 节介绍）、`PacedOperationDispatcher`（已在 3.6 节介绍）、`ProgressReporter`。

`ProgressReporter`（`progress.cc:134`）是一个后台线程，每 5 秒（默认 `report_interval`）采样一次 `completed_ops` / `completed_reads` / `completed_writes` 原子量，计算 interval 速率，打印到 `stderr`，并把 `ProgressSample` 存入 `samples_`。它的设计有几个关键点：

```cpp
const auto report_at = timed
  ? std::min(next_report, measurement_deadline)
  : next_report;
{
  std::unique_lock<std::mutex> lock(finish_mutex_);
  if (finish_cv_.wait_until(lock, report_at, [&] {
        return finished_.load(std::memory_order_acquire);
      })) {
    break;
  }
}
```

在时间模式下，每次 `wait_until` 的超时点取 `min(next_report, measurement_deadline)`，即**最多等到测量 deadline**。这样在 deadline 时刻会做最后一次采样，之后即使 worker 还在排水，reporter 也不会再采样——避免排水间隔被误记为“负载尾部零完成窗口”。deadline 采样后（`progress.cc:263`）：

```cpp
if (timed && report_at == measurement_deadline) {
  deadline_sampled = true;
  std::unique_lock<std::mutex> lock(finish_mutex_);
  finish_cv_.wait(lock, [&] {
    return finished_.load(std::memory_order_acquire);
  });
  break;
}
```

reporter 挂起在 `finish_cv_` 上，直到主线程调 `finish()`。这保证了 `samples()` 返回的序列恰好覆盖 `[start, deadline]`，不含排水期。

`finish()`（`progress.cc:152`）设置 `finished_`，通知 CV，join 线程。析构函数自动调 `finish()`，所以即使主线程忘记调也不会泄漏线程。`samples()` 用 `samples_mutex_` 保护，主线程在阶段结束后调 `reporter.samples()` 拿到 `measure_windows`。

## 六、存储侧 maintenance 日志聚合：`tools/breakdown_benchmark/maintenance_log.cc` / `maintenance_log.hh`

这个模块的作用是：从每个 storage 节点的 maintenance 日志文件中，提取 `[baseline_offset, end_offset)` 窗口内的 `MaintenanceObservation`，计算 backlog 斜率、p99 stitch 延迟、failure delta 等指标，写进报告的 `stage2` 段。它让 benchmark 能在计算侧看到存储侧的 maintenance 健康度，而不需要额外跑日志分析工具。

### 6.1 observation 解析

`parse_observation`（`maintenance_log.cc:82`）解析一行日志。它先找标记 `"storage-owner maintenance "`，再要求同一行包含 `"observation:"` 或 `"summary:"`，否则返回 `nullopt`。然后从第一个 `:` 后开始，按空格切分 token，每个 token 按 `=` 拆 key/value，存入 `Fields`。最后用 `parse_u64` / `parse_double` / `parse_histogram` 提取字段：

- `stitch_enqueued` / `stitched_live` / `stale` / `remaining` / `peer_reverse_remaining`：stitch 队列状态。
- `failed` / `peer_reverse_failed`：失败计数。
- `admission_window` / `completion_outstanding`：admission control 窗口。
- `p99_stitch_delay_upper_ms` / `p99_stitch_delay_over_30s`：p99 延迟。
- `stitch_delay_histogram`：18 桶直方图（`kMaintenanceLatencyBucketUpperMs`，`maintenance_log.cc:19`），桶上界从 1ms 到 30s 再到 `infinity`。

`backlog()`（`maintenance_log.cc:371`）计算“积压”：

```cpp
uint64_t MaintenanceObservation::backlog() const {
  const uint64_t completed = stitched_live + stale;
  const uint64_t unfinished_stitches =
    stitch_enqueued > completed ? stitch_enqueued - completed : 0;
  return std::max({unfinished_stitches, remaining, peer_reverse_remaining});
}
```

取三个量的最大值，避免重复计数（这些计数在 in-flight work 上有重叠）。`backlog_slope`（`maintenance_log.cc:211`）用最小二乘法对 `backlog` 序列做线性回归，斜率反映积压是增长还是收敛。

### 6.2 窗口聚合

`summarize_impl`（`maintenance_log.cc:293`）遍历每个 cursor，调 `read_log_slice` 读窗口内的日志行，解析出 observation 序列，然后：

- `remaining += latest.backlog()`：所有分片最终剩余积压之和。
- `max_backlog_observed`：窗口内单分片最大积压。
- `backlog_slope_per_sec += backlog_slope(...)`：所有分片斜率之和（正数表示积压在增长）。
- `failures`：用 `counter_delta` 计算 `failed + peer_reverse_failed` 的 delta，若 `latest < baseline`（计数器回卷）则跳过。
- `p99_stitch_delay_upper_ms`：用 `histogram_delta` 计算直方图 delta，再 `include_histogram_p99` 求 p99。p99 的求法（`maintenance_log.cc:258`）是：找到累计计数达到 `samples - samples/100` 的桶，取该桶上界。`over_30s` 标志在最后一个桶（`infinity`）时置位。

`snapshot_maintenance_logs`（`maintenance_log.cc:380`）在阶段开始时调用，记录每个日志文件的当前大小作为 cursor offset，并把最后一条 observation 作为 baseline。若文件为空但可读，则 baseline 设为“全零且 available”（`maintenance_log.cc:399`），这样后续 delta 就是窗口内全部新增。`summarize_maintenance_log_window`（`maintenance_log.cc:416`）在阶段结束时调用，传入 begin/end cursors，计算窗口 delta。

`backlog_slope_available` / `failure_delta_available` / `completion_window_available` / `p99_stitch_delay_available` 这些 `*_available` 布尔（`maintenance_log.cc:357-365`）只在**所有请求的日志都成功计算出对应指标**时为 true。这是为了防止“部分日志缺失导致指标偏小”被误读为“系统健康”。报告文本里 `unavailable` 就是这些布尔为 false 时打印的。

## 七、数据集读取：`tools/breakdown_benchmark/dataset.cc` / `dataset.hh`

这个模块负责读取向量文件、groundtruth 文件，并提供 `SinglePassRowStream` 游标。

`read_vector_rows`（`dataset.cc:40`）读取 `.u8bin` / `.i8bin` / `.fbin` / `.bin` 格式：前 8 字节是 `uint32_t count` + `uint32_t dim`，后接 `count * vector_bytes` 的 payload。`vector_bytes` 由 `vector_dtype_bytes(dtype, dim)` 决定（见第 2 课公共类型与配置）。`resolve_vector_dtype_config("auto", filepath_t{path})` 根据文件扩展名推断 dtype（`.u8bin` → uint8，`.i8bin` → int8，`.fbin`/`.bin` → float32）。`decode_rows` 为 true 时把每行解码成 float 存入 `decoded`，供 insert 使用（insert 需要float 中间态来编码 PQ）。

`read_groundtruth_bin`（`dataset.cc:109`）读取项目自定义的 groundtruth 格式：`uint32_t rows` + `uint32_t top_k` + `rows * top_k` 个 `uint32_t` ID。这与 `.ivecs` 格式（每行前有 `top_k` 字节）不同，`dvstor_sift101m_long_insert_recall.cc` 里会看到两种格式的分别处理。

`recall_at`（`dataset.cc:134`）已在 3.3 节介绍。`make_deterministic_vector` / `make_dataset`（`dataset.cc:10` / `dataset.cc:26`）已在 3.2 节介绍。

## 八、长插入 recall 与数据生成工具

### 8.1 `tools/dvstor_sift101m_long_insert_recall.cc`

这个工具（765 行）用于评估“在 1 亿 base 上额外插入 100 万向量后，recall 是否保持”。它的流程与 `dvstor_breakdown_benchmark` 类似但更专注：

1. 读 insert 向量文件、query 文件、（可选）baseline groundtruth、post groundtruth。
2. 跑 baseline recall（若有 baseline GT）：`run_recall(service, "baseline-100m", ...)`。
3. 跑 insert 阶段：`run_insert_phase`（`dvstor_sift101m_long_insert_recall.cc:480`），多线程批量 insert，每 `reset_breakdown_every` 次重置一次 breakdown 采样（避免长跑累积采样失真）。
4. `wait_for_settle`：等 `settle_seconds`（默认 300 秒），让存储侧 maintenance 把新插入的向量 stitch 进图。
5. 跑 post recall：`run_recall(service, "post-101m", ...)`。
6. 计算 `recall_delta = baseline - post`，输出 JSON 与文本。

`run_insert_phase` 的批量 insert（`dvstor_sift101m_long_insert_recall.cc:517`）与 `workload.cc` 的单条 insert 不同：

```cpp
const size_t begin = next_row.fetch_add(args.insert_batch_size, std::memory_order_relaxed);
if (begin >= effective_insert_count) break;
const size_t end = std::min(begin + args.insert_batch_size, effective_insert_count);
batch.clear();
for (size_t row = begin; row < end; ++row) {
  const uint64_t id64 = args.insert_start_id + row;
  batch.push_back({static_cast<node_t>(id64), insert_rows.decode_row(args.insert_row_offset + row)});
}
const size_t ok = service.insert(batch);
```

`insert_batch_size`（默认 16）控制每次 `service.insert` 的批量大小，这比单条 insert 更接近真实在线写入的批量模式。`reset_breakdown_every`（默认 50000）的逻辑（`dvstor_sift101m_long_insert_recall.cc:539`）：

```cpp
if (args.reset_breakdown_every > 0 &&
    before / args.reset_breakdown_every != after / args.reset_breakdown_every) {
  std::lock_guard<std::mutex> lock(reset_mutex);
  service.clear_thread_statistics();
  service.reset_breakdown_state();
}
```

当累计 attempted 跨过 `reset_breakdown_every` 的倍数时，重置 breakdown。这让长跑的 breakdown 采样始终反映“最近一批”的延迟分布，而不是整个 100 万 insert 的平均。

这个工具支持多种文件格式：`.bvecs` / `.fvecs`（带行前导 dim）与 `.u8bin` / `.i8bin` / `.fbin` / `.bin`（带文件头 count+dim），通过 `read_vector_rows`（`dvstor_sift101m_long_insert_recall.cc:255`）按扩展名分派。groundtruth 也支持 `.ivecs` 与项目自定义 `.bin` 两种格式（`read_groundtruth`，`dvstor_sift101m_long_insert_recall.cc:327`）。

### 8.2 `tools/generate_sift101m_recall_data.cc`

这个工具（423 行）用于从 SIFT1B 原始数据生成 101M 的 recall 评估数据：把 `bigann_query.bvecs` 转成 `.fbin`，并计算“在 100M base + 100 万 insert”上的 groundtruth。它的核心是 `build_groundtruth101`（`generate_sift101m_recall_data.cc:293`）：

1. 读 100M 的 groundtruth（`idx_100M.ivecs` + `dis_100M.fvecs`），取每个 query 的 top-k 候选。
2. 对每个 query，把 100M 的 top-k 作为初始堆，然后扫描全部 100 万 insert 向量，用 `l2_u8_bounded`（`generate_sift101m_recall_data.cc:283`）计算 L2 距离，若优于堆顶则替换。
3. 最终堆里的 k 个 ID 就是 101M 的 groundtruth。

`l2_u8_bounded` 有一个 `limit` 参数：一旦 `sum >= limit` 就提前返回，避免完整计算。`limit` 取当前堆顶距离，这样大部分远距离向量几个维度就能剪枝。这是暴力 kNN 的标准优化。多线程并行：每个 worker 独立打开 GT 文件，用 `next_query.fetch_add` 抢 query，互不干扰。

输出 query 文件是 `.fbin`（float32），因为 SIFT1B 的 query 是 uint8，但项目计算召回时用 float 距离，所以预先转成 float。groundtruth 是项目自定义 `.bin` 格式，ID 是 `insert_start + offset`（即 100000000 + offset），与 `dvstor_sift101m_long_insert_recall` 的 `--insert-start-id` 默认值一致。

## 九、实验脚本

实验脚本位于 `/home/xjs/experiment/dvstor/experiment/`，它们把上面这些工具串成一条可复现的端到端流程。

### 9.1 `common.sh` 与 `sift100m_common.sh`

`common.sh`（35 行）定义 `EXPERIMENT_DIR`、`PROJECT_DIR`、`REPORT_DIR`、`LOG_DIR`、`PID_DIR`，并 source `sift100m_common.sh`。它还定义 `load_experiment_profile`（`common.sh:13`）：

```bash
load_experiment_profile() {
  local profile="${1:?profile name is required}"
  local profile_env="$EXPERIMENT_DIR/profiles/${profile}.env"
  if [[ ! -f "$profile_env" ]]; then
    echo "unknown experiment profile: $profile" >&2
    echo "available profiles:" >&2
    find "$EXPERIMENT_DIR/profiles" -maxdepth 1 -name '*.env' -printf '  %f\n' \
      | sed 's/\.env$//' >&2
    return 1
  fi
  PROFILE="$profile"
  source "$profile_env"
}
```

profile 就是 `profiles/` 目录下的一个 `.env` 文件，被 source 进来覆盖 `sift100m_common.sh` 里的默认值。目前 `dev` 分支只保留 `04_gpu_persistent_gpunetio.env` 一个 profile。

`sift100m_common.sh`（353 行）是全部实验脚本的共享配置中心，按职责分段：

**路径与目录**（`sift100m_common.sh:13-19`）：

- `DATASET_DIR`：原始数据集（SIFT1B）。
- `WORK_DIR`：工作目录，下面分 `converted`（转换后的 `.u8bin`）、`index`（构建产物）。
- `REPORT_DIR` / `LOG_DIR` / `PID_DIR`：报告、日志、pid 文件。

**索引与系统参数**（`sift100m_common.sh:21-36`）：

- `SHARDS=5`：存储分片数。
- `PARTITION_STRATEGY=metis`：分区策略（见第 29 课离线构建/迁移）。
- `R=96`：Vamana 图度数。
- `BUILD_BEAM=128`、`ALPHA=1.2`：构建参数。
- `K=10`、`DIM=128`、`VECTOR_DATA_TYPE=uint8`：数据参数。
- `BUILD_THREADS=112`、`SERVICE_THREADS=64`：构建与服务线程数。
- `GPU_DEVICE=1`：使用的 GPU 编号。
- `PQ_SUBQUANTIZERS=32`：PQ 子量化器数。
- `MAX_VECTORS=100000000`、`MAX_QUERIES=10000`：base 与 recall 查询规模。
- `GROUNDTRUTH_LABEL=100M`、`GROUNDTRUTH_TOPK=10`：groundtruth 文件名标签与 top-k。

**benchmark 输入文件**（`sift100m_common.sh:41-47`）：这是搬机器时最常改的一块：

- `BENCHMARK_VECTOR_SOURCE`：原始 `bigann_base.bvecs`，只在 `PREPARE_BENCHMARK_DATA=1` 时用。
- `PERFORMANCE_QUERY_FILE`：性能查询池，默认 `sift100m_to_105m_query.u8bin`（500 万行，`[100M, 105M)`）。
- `INSERT_FILE`：插入池，默认 `sift103m_to_105m_insert.u8bin`（200 万行，`[103M, 105M)`）。
- `PERFORMANCE_QUERY_START/END`、`INSERT_VECTOR_START/END`：声明每个 u8bin 是从 base 的哪段切出来的，只用于记录，不影响运行。

**RDMA 与网络**（`sift100m_common.sh:52-58`）：

- `BASE_PORT=1234`：存储节点起始端口，第 `i` 个分片用 `BASE_PORT + i - 1`。
- `HOSTS`：5 个分片的 host，默认全是 `192.168.6.202`（单机 5 分片，集群拓扑见 dvstor 集群拓扑记忆）。
- `IB_DEVICE` / `IB_PORT`、`MAX_SEND_WRS` / `MAX_RECEIVE_WRS` / `MAX_POLL_CQES`：RDMA 参数（见第 4-5 课 RDMA 传输库）。

**索引前缀**（`sift100m_common.sh:60-61`）：

```bash
INDEX_PREFIX="${INDEX_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pq${PQ_SUBQUANTIZERS}}"
```

索引前缀编码了 R、BUILD_BEAM、分区策略、PQ 子量化器数，改任一参数都会产生不同前缀，避免覆盖旧索引。

**内存估算**（`sift100m_common.sh:63-85`）：`estimate_node_bytes` 按 `VECTOR_DATA_TYPE` 估算单节点字节数（fixed 部分按 16 字节对齐，graph 部分按 8 字节对齐），`estimate_mn_memory_gb` 估算每个 storage 节点需要的内存（`vectors_per_shard * node_bytes * 1.2 + 4GiB`，向上取整到 GiB，最小 8GiB）。这个估算用于 `mn-memory` 参数，storage 节点启动时申请这么多注册内存（见第 23 课存储节点主体/peer RDMA）。

**文件名辅助函数**（`sift100m_common.sh:89-126`）：`base_bin` / `query_bin` / `groundtruth_bin` / `insert_bin` / `performance_query_bin` / `metadata_file` / `model_file` / `shard_file` / `idmap_file` / `navigation_code_file`。这些函数把“索引前缀 + 后缀”的命名规则集中管理，避免脚本间不一致。

**`validate_index_metadata`**（`sift100m_common.sh:128-255`）：这是一个内嵌的 Python 校验脚本，读 `.meta.json`，校验 schema_version、distance、R、dim、num_vectors、num_shards、dtype、PQ 参数、分区策略、idmap 格式、anchor 格式、动态 PQ code 偏移不重叠等。校验失败时在“申请大块注册内存前”退出，避免 storage 节点启动到一半才崩溃。`role=compute` 时额外校验 `.pq32` 模型与 `.anchors` 存在，`require_update_sidecars=true` 时校验全部 5 个 `.idmap` 存在；`role=storage` 时校验对应分片的 `.dat`、`.idmap`、`.pq32.codes` 都存在。这个函数在 `run_breakdown.sh`、`run_recall.sh`、`start_memory_node.sh`、`build_sift100m_index.sh` 里都会被调用，是实验流程的“schema 守门人”。

**`server_endpoints`**（`sift100m_common.sh:257`）：把 `HOSTS` 与 `BASE_PORT` 拼成 `host:port` 列表，校验数量等于 `SHARDS`。

**`common_rdma_args`**（`sift100m_common.sh:272`）：拼 RDMA 相关的 `--ib-port`、`--max-send-wrs` 等参数，用 `%q` 转义便于 `read -r -a` 还原。

**`ensure_built`**（`sift100m_common.sh:279`）：检查 `BUILD_DIR/CMakeCache.txt` 存在，然后 `cmake --build` 指定 target。注意这里假设 `BUILD_DIR` 已经 configure 过，不会自动 configure。

**`write_service_config`**（`sift100m_common.sh:287`）：生成 benchmark 用的 `.ini` 文件。它先调 `validate_index_metadata compute 0 "$enable_updates"` 再写文件，内容覆盖 servers、initiator、num-clients、port、RDMA 参数、index-prefix、threads、向量参数、GPU 参数（query slots、memory limit、resident PQ budget、bootstrap window、graph prefetch depth、traversal beam width、final rerank width、max expansions、entry seed count、delta anchor probes、rdma qps、persistent blocks per SM）、update 参数（visibility、delta budget、maintenance period）、storage owner 参数（batch max、peer rdma tokens、rpc depth/timeout、search snapshot batch、update mode、maintenance mode/workers、reverse mode/queue depth/coalesce max）。这个 `.ini` 会被 `args.cc::build_service_argv` 解析成 `ComputeService` 的 argv。`ENABLE_UPDATES` 控制是否启用更新执行器（纯查询负载设为 false，避免初始化 mutation 路由）。

### 9.2 `profiles/04_gpu_persistent_gpunetio.env`

这个 profile（35 行）覆盖 `sift100m_common.sh` 的默认值，定义当前唯一支持的运行时架构：持久化 GPU OPQ/PQ32 图导航 + GPUNetIO 远端读取 + storage-owner 动态更新。它的关键覆盖：

```bash
PARTITION_STRATEGY=metis
PARTITION_MAX_DEGREE="${PARTITION_MAX_DEGREE:-32}"
PQ_SUBQUANTIZERS="${PQ_SUBQUANTIZERS:-32}"
INDEX_PREFIX="${PQ_INDEX_PREFIX:-$INDEX_DIR/sift100m_R${R}_bw${BUILD_BEAM}_${PARTITION_STRATEGY}_pmd${PARTITION_MAX_DEGREE}_pq${PQ_SUBQUANTIZERS}}"
```

注意 `INDEX_PREFIX` 在 profile 里多了 `_pmd${PARTITION_MAX_DEGREE}` 段，与 `sift100m_common.sh` 默认值不同——profile 优先级更高，所以实际跑用的是带 `pmd32` 的前缀。`PQ_INDEX_PREFIX` 允许完全覆盖前缀，用于保留旧索引。

其余字段都是 GPU 与 storage owner 参数的默认值，与 `write_service_config` 里的 `${VAR:-default}` 一一对应。`ENABLE_BREAKDOWN=true` 默认开启细粒度 breakdown（见第 27 课）。

### 9.3 `run_breakdown.sh`：主 benchmark 脚本

`run_breakdown.sh`（277 行）是日常实验的主入口。它的流程：

**1. 解析 profile 与默认参数**（`run_breakdown.sh:39-57`）：

```bash
PROFILE="${1:-${PROFILE:-04_gpu_persistent_gpunetio}}"
load_experiment_profile "$PROFILE"

WORKLOAD="${WORKLOAD:-mixed}"
BENCHMARK_CLIENT_THREADS="${BENCHMARK_CLIENT_THREADS:-128}"
READ_RATIO="${READ_RATIO:-0.5}"
MIXED_MODE="${MIXED_MODE:-fixed_threads}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RECALL_K="${RECALL_K:-$K}"
TARGET_QUERY_QPS="${TARGET_QUERY_QPS:-0}"
TARGET_WRITE_QPS="${TARGET_WRITE_QPS:-0}"
RECALL_MODE="${RECALL_MODE:-all}"
RECALL_BASE_ID_LIMIT="${RECALL_BASE_ID_LIMIT:-0}"
```

默认是 `mixed` + `fixed_threads` + `read_ratio=0.5` + 30s warmup + 120s measure + 1000 条 recall。这些默认值与 README 的示例一致。

**2. 决定需要哪些数据**（`run_breakdown.sh:73-100`）：根据 `WORKLOAD` 与 `MIXED_MODE` 设置 `needs_performance_query` / `needs_insert_data` / `needs_recall_data`。`mixed` + `rate_limited` 模式下，按 `TARGET_QUERY_QPS` / `TARGET_WRITE_QPS` 是否为正决定；其它模式下按 `READ_RATIO` 决定。`needs_recall_data` 默认为 1，即默认都跑 recall。

**3. 设置 `ENABLE_UPDATES`**（`run_breakdown.sh:104-108`）：有写操作的负载设为 true，纯查询设为 false。这控制 `write_service_config` 是否在 `.ini` 里写 `enable-updates = true`，进而控制 `ComputeService` 是否初始化 mutation 路由与 owner idmap。

**4. 收集 maintenance 日志路径**（`run_breakdown.sh:110-124`）：默认为每个分片自动拼 `$LOG_DIR/memory_node_${node_id}_${PROFILE}.log`，即 `start_memory_node.sh` 写日志的路径。也支持 `STORAGE_MAINTENANCE_LOGS` 环境变量覆盖。

**5. 校验索引**（`run_breakdown.sh:130`）：`validate_index_metadata compute 0 "$ENABLE_UPDATES"`。

**6. 拼数据文件路径**（`run_breakdown.sh:132-141`）：用 `query_bin` / `groundtruth_bin` / `performance_query_bin` / `insert_bin` 函数。

**7. 可选数据准备**（`run_breakdown.sh:145-160`）：`PREPARE_BENCHMARK_DATA=1` / `PREPARE_QUERY=1` / `PREPARE_GROUNDTRUTH=1` 时调 `prepare_sift100m_data.sh`。默认全为 0，即只读预生成文件。

**8. 校验数据文件存在且不冲突**（`run_breakdown.sh:162-203`）：recall 与 performance 文件不能相同（`readlink -f` 比较），performance 与 insert 文件不能相同。

**9. 计算 `effective_insert_start_id`**（`run_breakdown.sh:205-227`）：默认 `MAX_VECTORS + 1000000`；`mixed` 模式下取 `max(MAX_VECTORS, PERFORMANCE_QUERY_END) + 1000000`，避免 insert ID 与 performance query 行号重叠。校验是 uint32 且不与 base ID 重叠。

**10. 构建 benchmark 二进制**（`run_breakdown.sh:229`）：`ensure_built dvstor_breakdown_benchmark`。

**11. 生成运行时配置并拼命令行**（`run_breakdown.sh:231-274`）：时间戳命名输出文件，`write_service_config` 生成 `.ini`，拼出完整的 `dvstor_breakdown_benchmark` 命令，按需追加 recall / performance / insert / maintenance log 参数，最后 `printf` 命令行并执行。

脚本结尾的两行 `echo` 把 JSON 与文本报告路径打到 stdout，方便后续脚本或 CI 抓取。

### 9.4 `run_recall.sh`：纯 recall 脚本

`run_recall.sh`（60 行）是 `run_breakdown.sh` 的简化版，只跑 recall 不跑性能负载：

```bash
cmd=("$BUILD_DIR/dvstor_breakdown_benchmark"
  --service-config "$RUNTIME_CONFIG"
  --workload query
  --recall-only
  --warmup-ops 0
  --measure-ops 0
  --client-threads "$RECALL_CLIENT_THREADS"
  --recall-query-file "$RECALL_QUERY_FILE"
  --groundtruth-file "$GROUNDTRUTH_FILE"
  --recall-queries "$RECALL_QUERIES"
  --recall-k "$RECALL_K"
  --report-json "$JSON_REPORT"
  --report-text "$TEXT_REPORT")
```

`--recall-only` 让 `workload.cc` 跳过所有 warmup/measure 阶段，只跑 recall。`--warmup-ops 0 --measure-ops 0` 是非时间模式的占位（`args.cc` 要求 `warmup_seconds` 与 `measure_seconds` 同时为正才能用时间模式，所以这里用 ops 模式且 ops 为 0）。`ENABLE_UPDATES=false` 因为纯查询不需要 mutation 路由。这个脚本用于“先验证 recall 没问题再跑性能”的两步流程。

### 9.5 存储节点生命周期脚本

**`start_memory_node.sh`**（90 行）启动单个 storage 节点：

1. 校验 `NODE_ID` 在 `[1, SHARDS]`。
2. `ensure_built dvstor_memory_node`。
3. `validate_index_metadata storage "$NODE_ID"`：校验该分片的 `.dat`、`.idmap`、`.pq32.codes` 都存在。
4. 检查 pid 文件，若进程还活着则拒绝重启。
5. 拼命令行：`--is-server`、`--num-clients 1`、`--servers`、`--port`、RDMA 参数、`--server-index-file`、`--index-prefix`、向量参数、`--storage-id $((NODE_ID - 1))`、`--storage-peers`、storage owner 参数。
6. **CPU 分区**（`start_memory_node.sh:35-50`）：当多个逻辑分片共享同一 host 时，通过 `DVSTOR_LOCAL_PROCESS_RANK` / `DVSTOR_LOCAL_PROCESS_COUNT` 环境变量让每个进程知道自己是本机第几个进程，从而划分 CPU 核心。这是单机多分片部署的关键——否则所有进程都会绑定到同一组核心，互相抢 CPU。
7. `nohup ... > "$LOG_FILE" 2>&1 &` 后台启动，`echo $! > "$PID_FILE"`。

**`start_all_memory_nodes.sh`**（11 行）就是循环调 `start_memory_node.sh`。

**`stop_memory_node.sh`**（16 行）：

```bash
for pid_file in "$PID_DIR"/memory_node_*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping $pid_file pid=$pid"
    kill "$pid"
  fi
  rm -f "$pid_file"
done
```

它遍历所有 `memory_node_*.pid`，`kill` 进程并删除 pid 文件。这里有一个经典的“`pkill` 自杀陷阱”：如果用 `pkill -f dvstor_memory_node` 来停进程，`pkill` 会匹配自己的命令行（因为 `pkill -f dvstor_memory_node` 这个命令行里就包含 `dvstor_memory_node` 字符串），把自己也杀掉。所以脚本用 pid 文件 + `kill` 的方式，避免 `pkill`。这是 dvstor 集群拓扑记忆里提到的坑。

### 9.6 `build_sift100m_index.sh`：索引构建脚本

`build_sift100m_index.sh`（96 行）是离线索引构建入口（见第 29 课离线构建/迁移）。它的流程：

1. 加载 profile，设置 PQ 构建参数（`PQ_TRAIN_SAMPLES`、`PQ_OPQ_ITERATIONS`、`PQ_ITERATIONS` 等）。
2. `ensure_built vamana_offline_builder vamana_pq_indexer`。
3. `PREPARE_BENCHMARK_DATA=0 prepare_sift100m_data.sh`：准备 base 数据（但不准备 benchmark 数据）。
4. 检查索引产物是否已存在：若存在且 `OVERWRITE_INDEX != 1`，拒绝覆盖；否则删除旧产物。
5. 跑 `vamana_offline_builder`：构建 schema-14 compact Vamana/Metis 分片。
6. 跑 `vamana_pq_indexer`：训练 OPQ/PQ32 并编码。`OMP_NUM_THREADS=$PQ_THREADS`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OMP_DYNAMIC=FALSE`：限制 BLAS 线程为 1，避免与 Faiss OpenMP 嵌套（这会爆炸式创建线程）。
7. `validate_index_metadata storage`：构建完成后校验 schema。

这个脚本与第 29 课的离线构建工具对接，产出的索引被 `start_memory_node.sh` 与 `run_breakdown.sh` 使用。

### 9.7 `compare_reports.py`：报告对比工具

`compare_reports.py`（76 行）把两份 JSON 报告（baseline 与 candidate）对比，输出 query/write QPS 加速比、p99 延迟比、recall delta。它的 `metrics` 函数（`compare_reports.py:21`）用 `nested` 辅助函数从多种可能的 JSON 路径取值：

```python
def nested(document, *paths, default=0.0):
    for path in paths:
        value = document
        try:
            for key in path.split("."):
                value = value[key]
            return float(value)
        except (KeyError, TypeError, ValueError):
            continue
    return float(default)
```

这让它能同时处理 dvstor 与 OdinANN 两种 JSON 格式：dvstor 的 query QPS 在 `throughput.query_ops_per_sec`，OdinANN 可能在 `read_qps`；recall 在 `static_gt_post_recall.recall` 或 `recall.recall` 或 `post_recall`。p99 延迟优先取 `read_p99_us` / `write_p99_us`，取不到则从 `query_breakdown.latency.p99_end_to_end_ns` / 1000 计算。

输出示例（来自 `compare_reports.py:63`）：

```
query QPS : 10000.00 -> 24000.00 (2.400x)
write QPS : 1000.00 -> 3200.00 (3.200x)
recall    : 0.940000 -> 0.939600 (-0.000400)
query p99 : 5000.00 -> 3200.00 us
```

`--output` 可选写出 JSON。这个工具**只输出原始数字，不给出通过/失败结论**——这是整个实验脚本的统一原则：脚本负责采集与对比，是否达标由实验者根据目标负载自行判断。

### 9.8 `experiment/README.md`：实验手册

`README.md`（240 行）是实验目录的说明书，覆盖配置、构建、转换、部署、启动、召回率与性能、停止等全部流程。几条关键指引：

- **部署文件清单**：计算节点需要 `.meta.json`、`.pq32`、`.anchors`、全部 5 个 `.idmap`；存储节点 X 需要 `.meta.json`、`_nodeX_ofN.dat`、`.idmap`、`.pq32.codes`。纯查询配置 `enable-updates = false` 时计算节点不需要 `.idmap`。
- **stage2 finalized 的边界**（`README.md:138`）：明确说明“stage2 finalized 仅表示已声明的 maintenance 任务完成，不等价于全图整理已经完成”，避免误读 `durable` / `drained` 字段。
- **schema-15 的限制**（`README.md:131`）：反向边请求只携带物理指针没有 generation，所以每次 insert/upsert 都消耗新节点/向量空间，部署时必须预留 memory-node 容量。
- **验收要点**（`README.md:220-223`）：`direct_path_failures == 0`、前后 recall 及其变化、没有 unhealthy/fail-stop 日志、GPU 与 RDMA 指标显示多查询并发而非单查询串行等待。这四条是实验者审报告时的 checklist。
- **recall 与性能查询文件分离**（`README.md:163`）：`query.u8bin` 的 10K 标准查询仅供 recall，性能阶段用独立的 `PERFORMANCE_QUERY_FILE`，warmup 与 measure 共用一个单遍游标，同一行不会再次执行，查询池耗尽时 benchmark 失败而不是取模回绕。

## 十、关键数据结构与流程图

### 10.1 benchmark 内部线程模型

```
dvstor_breakdown_benchmark 进程
├── main 线程
│   ├── parse_args → Args
│   ├── build_service_argv(.ini) → argv
│   ├── IndexConfiguration(argv) → ComputeService
│   └── run_benchmark(service, args)
│       ├── [recall 阶段] recall_workers × N → service.search_raw → recall_at
│       ├── [warmup 阶段] client_threads × N
│       │   ├── query: SinglePassRowStream.try_claim → service.search_raw
│       │   ├── insert: get_insert_vector → service.insert
│       │   └── mixed: choose_mixed_read / issue_mixed_write / PacedOperationDispatcher.claim
│       ├── wait_for_gpu_publications (轮询 mutation_capacity_reserved == 0)
│       ├── snapshot_maintenance_logs (记录 begin cursors)
│       ├── service.clear_thread_statistics / reset_breakdown_state
│       ├── [measure 阶段] 同 warmup，label="measure-*"
│       ├── snapshot_maintenance_logs (记录 end cursors)
│       ├── summarize_maintenance_log_window(begin, end) → stage2
│       ├── wait_for_gpu_publications
│       ├── service.collect_breakdown_report → Report
│       ├── service.gpu_search_telemetry → TelemetrySnapshot
│       ├── 计算 throughput / stability
│       ├── run_recall_check("after_performance", "static_gt_post_recall")
│       ├── format_report → text
│       └── 写 JSON / text 报告
├── ProgressReporter 线程 × 1 (每 5s 采样 completed_ops/reads/writes)
└── ComputeService 内部线程池 (第 27 课)
    ├── service threads (SERVICE_THREADS)
    ├── GPU persistent kernel (第 17-21 课)
    └── storage owner RPC / maintenance (第 23-26 课)
```

关键点：benchmark 客户端线程（`client_threads`）与 `ComputeService` 内部服务线程（`SERVICE_THREADS`）是两组独立的线程池。客户端线程调用 `service.search_raw` 等同步 API，请求被投递到服务线程池执行，GPU 持久化 kernel 在后台持续消费查询。`ProgressReporter` 是第三个独立线程，只做采样不参与施压。

### 10.2 脚本编排端到端流程

```
[1] build_sift100m_index.sh
    │  vamana_offline_builder → schema-14 compact 分片
    │  vamana_pq_indexer → OPQ/PQ32 训练 + 编码 → schema-15 索引
    │  validate_index_metadata storage
    ▼
[2] start_all_memory_nodes.sh
    │  for node in 1..SHARDS: start_memory_node.sh $node
    │    ├─ validate_index_metadata storage $node
    │    ├─ DVSTOR_LOCAL_PROCESS_RANK/COUNT 划分 CPU
    │    ├─ nohup dvstor_memory_node --is-server ... > log &
    │    └─ echo $! > pid_file
    ▼
[3a] run_recall.sh (可选，先验证 recall)
    │  --recall-only --warmup-ops 0 --measure-ops 0
    │  → recall_04_.../sift100m_recall_*.json/txt
    ▼
[3b] run_breakdown.sh (主性能实验)
    │  决定 needs_performance_query/needs_insert_data/needs_recall_data
    │  设置 ENABLE_UPDATES
    │  收集 maintenance_logs (每分片一个 .log)
    │  validate_index_metadata compute 0 $ENABLE_UPDATES
    │  write_service_config → service_*.ini
    │  dvstor_breakdown_benchmark --service-config .ini ...
    │  → 04_.../sift100m_04_*.json/txt
    ▼
[4] compare_reports.py
    │  --baseline odinann.json --candidate latest.json
    │  → query/write QPS 加速比、recall delta、p99 比值
    ▼
[5] stop_memory_nodes.sh
       for pid_file in memory_node_*.pid: kill $pid; rm pid_file
```

这个流程与第 1 课 README 的运行流程（见第 1 课）一致：构建 → 启动存储 → 跑 recall 验证 → 跑 breakdown 性能 → 对比 → 停止。

## 十一、与其他模块的关系

- **第 27 课 ComputeService/breakdown**：benchmark 直接 in-process 构造 `ComputeService`，调用 `search_raw` / `insert` / `upsert` / `erase` / `collect_breakdown_report` / `gpu_search_telemetry` / `clear_thread_statistics` / `reset_breakdown_state`。报告里的 `insert breakdown` / `query breakdown` 段直接来自 `service::breakdown::aggregate_text_summary`，即第 27 课的细粒度 breakdown 聚合。
- **第 23 课存储节点主体/peer RDMA**：`start_memory_node.sh` 启动的 `dvstor_memory_node` 就是第 23 课讲的存储节点主体。benchmark 通过 `--storage-maintenance-log` 读取它的日志，聚合出 stage2 指标。
- **第 29 课离线构建/迁移**：`build_sift100m_index.sh` 调用的 `vamana_offline_builder` 与 `vamana_pq_indexer` 是第 29 课的离线构建工具。产出的 schema-15 索引被本课的 benchmark 与 storage 节点使用。
- **第 22 课 GPUNetIO 传输/probe**：报告里的 `direct_path_failures` 字段直接反映 GPUNetIO direct path 的健康度，是验收时必须为 0 的字段。
- **第 10 课 delta/动态路由/预算**：`wait_for_gpu_publications` 等待的 `mutation_capacity_reserved` 与报告里的 `dynamic_route_publications` / `delta_publications` / `mutation_capacity_*` 字段都来自第 10 课的动态路由与 delta 预算机制。
- **第 9 课 GPU 类型/遥测/PQ 模型**：`telemetry_to_json` 序列化的 `TelemetrySnapshot` 就是第 9 课定义的 GPU 遥测结构。
- **第 6 课 Vamana 图格式**：`VamanaNode::vector_dtype_name` / `vector_bytes` / `total_size` 来自第 6 课的节点布局。
- **第 2 课公共类型与配置**：`VectorDType` / `vector_dtype_bytes` / `resolve_vector_dtype_config` / `decode_storage_vector_to_float` 都是第 2 课的公共类型工具。
- **第 1 课项目总览**：本课的脚本编排流程与第 1 课 README 的运行流程一致，是 README 在代码层面的具体实现。

## 十二、小结

本课讲解了 dvstor 项目的 benchmark 框架与实验脚本，核心要点：

1. **benchmark 是 in-process 的**：`dvstor_breakdown_benchmark` 直接构造 `ComputeService`，客户端线程调用同步 API，避免了 IPC 噪声，让 breakdown 采样与性能测量都发生在同一进程内。

2. **查询游标永不回绕**：`SinglePassRowStream` 用原子行号分配，耗尽即终止。recall 查询集（10K）与 performance 查询集（500 万）必须分离，否则性能数字不可信。

3. **三种混合调度模式**：`probability`（按概率伯努利抽样）、`fixed_threads`（静态线程划分）、`rate_limited`（`PacedOperationDispatcher` 共享双流限速器）。每种模式对应不同的实验问题。

4. **warmup-measure 双阶段 + 前后两次 recall**：warmup 让系统进入稳态，measure 采集稳态数据；performance 前后的两次 recall 反映在线写入对搜索质量的影响。

5. **GPU publication 等待与 stage2 日志聚合**：measure 结束后等 `mutation_capacity_reserved == 0` 再采集 GPU 遥测；同时从 storage 节点日志窗口计算 backlog 斜率、p99 stitch 延迟、failure delta，让计算侧报告能反映存储侧健康度。

6. **脚本只记录原始结果，不自动验收**：验收四条（`direct_path_failures == 0`、recall 变化、无 unhealthy/fail-stop、多查询并发而非单查询串行）由实验者根据报告字段人工判断，`compare_reports.py` 也只输出数字不结论。

7. **工程坑**：`pkill` 自杀陷阱（用 pid 文件 + `kill` 替代）、BLAS 与 OpenMP 嵌套（`OPENBLAS_NUM_THREADS=1`）、单机多分片 CPU 分区（`DVSTOR_LOCAL_PROCESS_RANK`）、recall 与 performance 文件必须不同（`std::filesystem::equivalent` 校验）、`rate_limited` 模式预估查询行数避免中途耗尽。

至此，30 课教程结束。从第 1 课的项目总览到本课的实验脚本，我们依次拆解了 dvstor 的传输层、索引格式、GPU 持久化引擎、存储节点、计算服务、离线构建与实验框架。一个工程上可复现的 GPU 中心化存算分离向量检索系统，不止需要正确的算法实现，还需要本课讲的这种“可重复执行的负载脚本 + 不会自欺欺人的报表”——这才是把研究原型变成可被他人审视的系统的最后一公里。
