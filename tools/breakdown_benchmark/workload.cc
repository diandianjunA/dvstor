#include "tools/breakdown_benchmark/workload.hh"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/distance.hh"
#include "service/breakdown.hh"
#include "tools/breakdown_benchmark/progress.hh"

namespace tools::breakdown_benchmark {

std::vector<float> make_deterministic_vector(uint32_t seed, size_t dim) {
  std::vector<float> vector(dim, 0.0f);
  uint64_t state = 1469598103934665603ull ^ static_cast<uint64_t>(seed);
  for (size_t i = 0; i < dim; ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const uint32_t value = static_cast<uint32_t>((state * 2685821657736338717ull) >> 32);
    vector[i] = static_cast<float>(value % 10000) / 10000.0f;
  }
  vector[seed % dim] += 4.0f;
  vector[(seed * 17 + 3) % dim] += 1.0f;
  return vector;
}

std::vector<float> make_dataset(const std::vector<uint32_t>& ids, size_t dim) {
  std::vector<float> vectors;
  vectors.reserve(ids.size() * dim);
  for (uint32_t id : ids) {
    auto vec = make_deterministic_vector(id, dim);
    vectors.insert(vectors.end(), vec.begin(), vec.end());
  }
  return vectors;
}

std::vector<float> read_fbin(const std::string& path, uint32_t* dim_out, size_t* count_out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  uint32_t count = 0;
  uint32_t dim = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) {
    throw std::runtime_error("failed to read fbin header: " + path);
  }
  std::vector<float> data(static_cast<size_t>(count) * dim);
  input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("failed to read fbin payload: " + path);
  }
  if (dim_out) {
    *dim_out = dim;
  }
  if (count_out) {
    *count_out = count;
  }
  return data;
}


template <class Distance>
nlohmann::json run_benchmark(ComputeService<Distance>& service, const Args& args) {
  using SampleReport = service::breakdown::Report;
  using service::breakdown::aggregate_text_summary;
  using service::breakdown::report_to_json;

  nlohmann::json root;
  root["meta"] = {
    {"workload", args.workload},
    {"warmup_ops", args.warmup_ops},
    {"measure_ops", args.measure_ops},
    {"warmup_seconds", args.warmup_seconds},
    {"measure_seconds", args.measure_seconds},
    {"run_mode", (args.warmup_seconds > 0 || args.measure_seconds > 0) ? "time" : "ops"},
    {"time_completion_policy", "drain"},
    {"time_issue_policy", "dedicated_read_write_threads_until_deadline"},
    {"mixed_dispatch_policy", "thread_pool_split"},
    {"operation_granularity", "single_vector"},
    {"client_threads", args.client_threads},
    {"read_ratio", args.read_ratio},
    {"insert_start_id", args.insert_start_id},
    {"dim", service.config().dim},
    {"threads", service.config().num_threads},
    {"coroutines", service.config().num_coroutines},
    {"search_mode", service.config().search_mode},
  };
  const size_t dim = service.config().dim;
  const size_t bootstrap_work = args.measure_seconds > 0
                                  ? std::max<size_t>(4096, args.client_threads * 256)
                                  : std::max<size_t>(2048, args.measure_ops);
  const size_t bootstrap_count = bootstrap_work;
  std::cerr << "[breakdown] preparing workload: bootstrap_count=" << bootstrap_count
            << ", workload=" << args.workload << std::endl;
  const bool needs_query_data = (args.workload == "query" || args.workload == "both" || args.workload == "mixed");
  const bool requires_rabitq_artifacts = service.config().use_rabitq_search() &&
                                         (args.workload == "insert" || args.workload == "both" || args.workload == "mixed");

  if (requires_rabitq_artifacts && !service.config().load_index) {
    throw std::runtime_error(
      "mixed/insert benchmark with search-mode=rabitq_gpu requires a preloaded offline index. "
      "Enable --load-index and provide a valid index-prefix so the .meta.json and .rotation.bin artifacts are loaded.");
  }

  std::vector<uint32_t> bootstrap_ids(bootstrap_count);
  std::iota(bootstrap_ids.begin(), bootstrap_ids.end(), 1);
  const auto bootstrap_vectors = make_dataset(bootstrap_ids, dim);
  std::cerr << "[breakdown] bootstrap vectors ready" << std::endl;

  if (needs_query_data && !service.config().load_index) {
    vec<typename ComputeService<Distance>::InsertItem> bootstrap_batch;
    bootstrap_batch.reserve(bootstrap_count);
    for (size_t i = 0; i < bootstrap_count; ++i) {
      const auto begin = bootstrap_vectors.begin() + static_cast<std::ptrdiff_t>(i * dim);
      bootstrap_batch.push_back({bootstrap_ids[i], vec<element_t>(begin, begin + static_cast<std::ptrdiff_t>(dim))});
    }
    service.insert(bootstrap_batch);
    root["meta"]["bootstrap_vectors"] = bootstrap_count;
  }

  auto run_insert_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, ops, 0);
    for (size_t op = 0; op < ops; ++op) {
      const uint32_t id = start_id + static_cast<uint32_t>(op);
      auto values = make_deterministic_vector(id, dim);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
      service.insert(insert_items);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
    }
    reporter.finish();
  };

  auto run_insert_phase_seconds = [&](const std::string& label, size_t seconds, uint32_t start_id) -> uint32_t {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    uint32_t current_id = start_id;
    std::chrono::nanoseconds avg_insert_duration{0};
    size_t local_completed = 0;
    while (can_start_timed_operation(deadline, avg_insert_duration, local_completed)) {
      const uint32_t id = current_id++;
      auto values = make_deterministic_vector(id, dim);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
      const auto started_at = std::chrono::steady_clock::now();
      service.insert(insert_items);
      update_avg_duration(avg_insert_duration, started_at, local_completed);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++local_completed;
    }
    reporter.finish();
    return current_id;
  };

  std::vector<float> query_data;
  size_t query_count = 0;
  if (!args.query_file.empty()) {
    std::ifstream probe(args.query_file, std::ios::binary);
    if (!probe.good()) {
      throw std::runtime_error("query file does not exist: " + args.query_file);
    }
    uint32_t file_dim = 0;
    query_data = read_fbin(args.query_file, &file_dim, &query_count);
    if (file_dim != dim) {
      throw std::runtime_error("query dim mismatch with service config");
    }
  } else {
    query_count = std::max<size_t>(
      bootstrap_count,
      args.measure_seconds > 0 ? args.client_threads * 4096 : args.measure_ops * args.client_threads);
    std::vector<uint32_t> query_ids(query_count);
    std::iota(query_ids.begin(), query_ids.end(), bootstrap_count + 1);
    query_data = make_dataset(query_ids, dim);
    root["meta"]["synthetic_query_vectors"] = query_count;
  }
  std::cerr << "[breakdown] query data ready: count=" << query_count << std::endl;

  auto run_query_phase_ops = [&](const std::string& label, size_t ops) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, ops, 0);
    for (size_t op = 0; op < ops; ++op) {
      const size_t idx = op % query_count;
      std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(idx * dim),
                               query_data.begin() + static_cast<std::ptrdiff_t>((idx + 1) * dim));
      (void)service.search(query, service.config().k);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
    }
    reporter.finish();
  };

  auto run_query_phase_seconds = [&](const std::string& label, size_t seconds) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    size_t op = 0;
    std::chrono::nanoseconds avg_query_duration{0};
    while (can_start_timed_operation(deadline, avg_query_duration, op)) {
      const size_t idx = op % query_count;
      std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(idx * dim),
                               query_data.begin() + static_cast<std::ptrdiff_t>((idx + 1) * dim));
      const auto started_at = std::chrono::steady_clock::now();
      (void)service.search(query, service.config().k);
      update_avg_duration(avg_query_duration, started_at, op);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++op;
    }
    reporter.finish();
  };

  const auto mixed_read_thread_count = [&]() -> size_t {
    if (args.client_threads == 0) {
      return 0;
    }
    if (args.read_ratio <= 0.0) {
      return 0;
    }
    if (args.read_ratio >= 1.0) {
      return args.client_threads;
    }
    const size_t rounded = static_cast<size_t>(std::llround(static_cast<double>(args.client_threads) * args.read_ratio));
    return std::clamp<size_t>(rounded, 1, args.client_threads - 1);
  };

  auto run_mixed_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    const size_t read_target = static_cast<size_t>(std::llround(static_cast<double>(ops) * args.read_ratio));
    const size_t write_target = ops >= read_target ? (ops - read_target) : 0;
    std::atomic<size_t> next_read{0};
    std::atomic<size_t> next_write{0};
    const size_t read_threads = mixed_read_thread_count();
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, ops, 0);

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        start_barrier.arrive_and_wait();
        const bool do_read = tid < read_threads;
        if (do_read) {
          for (;;) {
            const size_t read_index = next_read.fetch_add(1, std::memory_order_relaxed);
            if (read_index >= read_target) {
              break;
            }
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(query_idx * dim),
                                     query_data.begin() + static_cast<std::ptrdiff_t>((query_idx + 1) * dim));
            (void)service.search(query, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
            completed_ops.fetch_add(1, std::memory_order_relaxed);
          }
        } else {
          for (;;) {
            const size_t write_index = next_write.fetch_add(1, std::memory_order_relaxed);
            if (write_index >= write_target) {
              break;
            }
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
            auto values = make_deterministic_vector(id, dim);
            vec<typename ComputeService<Distance>::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
            (void)service.insert(insert_items);
            completed_writes.fetch_add(1, std::memory_order_relaxed);
            completed_ops.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    reporter.finish();
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
    };
  };

  auto run_mixed_phase_seconds = [&](const std::string& label, size_t seconds, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    const size_t read_threads = mixed_read_thread_count();
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        start_barrier.arrive_and_wait();
        const bool do_read = tid < read_threads;
        for (;;) {
          if (std::chrono::steady_clock::now() >= deadline) {
            break;
          }

          if (do_read) {
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(query_idx * dim),
                                     query_data.begin() + static_cast<std::ptrdiff_t>((query_idx + 1) * dim));
            (void)service.search(query, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
            auto values = make_deterministic_vector(id, dim);
            vec<typename ComputeService<Distance>::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
            (void)service.insert(insert_items);
            completed_writes.fetch_add(1, std::memory_order_relaxed);
          }
          completed_ops.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    reporter.finish();
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
    };
  };

  uint32_t next_insert_id = args.insert_start_id;
  if (next_insert_id == 0) {
    const uint64_t default_start = service.config().load_index
                                     ? static_cast<uint64_t>(service.config().max_vectors) + 10'000ull
                                     : static_cast<uint64_t>(bootstrap_count) + 10'000ull;
    if (default_start > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("default insert start id exceeds uint32_t; pass --insert-start-id explicitly");
    }
    next_insert_id = static_cast<uint32_t>(default_start);
  }
  root["meta"]["effective_insert_start_id"] = next_insert_id;
  std::cerr << "[breakdown] effective insert start id=" << next_insert_id << std::endl;
  const bool use_time_mode = args.warmup_seconds > 0 || args.measure_seconds > 0;
  MixedPhaseStats warmup_mixed_stats{};
  MixedPhaseStats measure_mixed_stats{};

  if (args.workload == "insert" || args.workload == "both") {
    std::cerr << "[breakdown] starting warmup insert" << std::endl;
    if (use_time_mode) {
      next_insert_id = run_insert_phase_seconds("warmup-insert", args.warmup_seconds, next_insert_id) + 1024;
    } else {
      run_insert_phase_ops("warmup-insert", args.warmup_ops, next_insert_id);
      next_insert_id += static_cast<uint32_t>(args.warmup_ops + 1024);
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    std::cerr << "[breakdown] starting warmup query" << std::endl;
    if (use_time_mode) {
      run_query_phase_seconds("warmup-query", args.warmup_seconds);
    } else {
      run_query_phase_ops("warmup-query", args.warmup_ops);
    }
  }
  if (args.workload == "mixed") {
    std::cerr << "[breakdown] starting warmup mixed" << std::endl;
    if (use_time_mode) {
      warmup_mixed_stats = run_mixed_phase_seconds("warmup-mixed", args.warmup_seconds, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    } else {
      warmup_mixed_stats = run_mixed_phase_ops("warmup-mixed", args.warmup_ops, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    }
  }

  service.clear_thread_statistics();
  service.reset_breakdown_state();
  std::cerr << "[breakdown] starting measure phase" << std::endl;

  if (args.workload == "insert" || args.workload == "both") {
    if (use_time_mode) {
      next_insert_id = run_insert_phase_seconds("measure-insert", args.measure_seconds, next_insert_id);
    } else {
      run_insert_phase_ops("measure-insert", args.measure_ops, next_insert_id);
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    if (use_time_mode) {
      run_query_phase_seconds("measure-query", args.measure_seconds);
    } else {
      run_query_phase_ops("measure-query", args.measure_ops);
    }
  }
  if (args.workload == "mixed") {
    if (use_time_mode) {
      measure_mixed_stats = run_mixed_phase_seconds("measure-mixed", args.measure_seconds, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
    } else {
      measure_mixed_stats = run_mixed_phase_ops("measure-mixed", args.measure_ops, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
    }
  }

  if (args.workload == "mixed") {
    root["meta"]["warmup_mixed"] = {
      {"issued_reads", warmup_mixed_stats.issued_reads},
      {"issued_writes", warmup_mixed_stats.issued_writes},
      {"completed_reads", warmup_mixed_stats.completed_reads},
      {"completed_writes", warmup_mixed_stats.completed_writes},
    };
    root["meta"]["measure_mixed"] = {
      {"issued_reads", measure_mixed_stats.issued_reads},
      {"issued_writes", measure_mixed_stats.issued_writes},
      {"completed_reads", measure_mixed_stats.completed_reads},
      {"completed_writes", measure_mixed_stats.completed_writes},
    };
    std::cerr << "[breakdown][measure-mixed] reads issued/completed=" << measure_mixed_stats.issued_reads << "/"
              << measure_mixed_stats.completed_reads << ", writes issued/completed="
              << measure_mixed_stats.issued_writes << "/" << measure_mixed_stats.completed_writes << std::endl;
    lib_assert(measure_mixed_stats.completed_reads > 0, "mixed benchmark completed zero reads");
    lib_assert(measure_mixed_stats.completed_writes > 0, "mixed benchmark completed zero writes");
  }

  const SampleReport report = service.collect_breakdown_report();
  root.update(report_to_json(report));

  const bool has_throughput_duration = use_time_mode && args.measure_seconds > 0;
  const double throughput_duration = has_throughput_duration ? static_cast<double>(args.measure_seconds) : 0.0;
  const double query_throughput = has_throughput_duration
                                    ? static_cast<double>(report.query.count) / throughput_duration
                                    : 0.0;
  const double insert_throughput = has_throughput_duration
                                     ? static_cast<double>(report.insert.count) / throughput_duration
                                     : 0.0;
  const double total_throughput = query_throughput + insert_throughput;
  root["throughput"] = {
    {"duration_seconds", throughput_duration},
    {"total_ops", report.query.count + report.insert.count},
    {"total_ops_per_sec", total_throughput},
    {"query_ops", report.query.count},
    {"query_ops_per_sec", query_throughput},
    {"insert_ops", report.insert.count},
    {"insert_ops_per_sec", insert_throughput},
  };

  nlohmann::json summaries = nlohmann::json::object();
  std::ostringstream text_summary;
  if (has_throughput_duration) {
    text_summary << "throughput\n";
    text_summary << "  duration_seconds: " << throughput_duration << '\n';
    text_summary << "  total_ops_per_sec: " << total_throughput
                 << " (ops=" << (report.query.count + report.insert.count) << ")\n";
    text_summary << "  query_ops_per_sec: " << query_throughput
                 << " (ops=" << report.query.count << ")\n";
    text_summary << "  insert_ops_per_sec: " << insert_throughput
                 << " (ops=" << report.insert.count << ")\n";
  }
  if (report.has_insert()) {
    const auto summary = aggregate_text_summary(report.insert);
    summaries["insert"] = summary;
    text_summary << summary;
  }
  if (report.has_query()) {
    const auto summary = aggregate_text_summary(report.query);
    summaries["query"] = summary;
    text_summary << summary;
  }
  root["bottleneck_summary"] = std::move(summaries);
  root["system_counters"] = {
    {"rdma_read_bytes", report.query.counters.rdma_read_bytes + report.insert.counters.rdma_read_bytes},
    {"rdma_write_bytes", report.query.counters.rdma_write_bytes + report.insert.counters.rdma_write_bytes},
    {"h2d_bytes", report.query.counters.h2d_bytes + report.insert.counters.h2d_bytes},
    {"d2h_bytes", report.query.counters.d2h_bytes + report.insert.counters.d2h_bytes},
  };

  std::ofstream json_output(args.report_json_path);
  json_output << root.dump(2) << '\n';
  if (!json_output) {
    throw std::runtime_error("failed to write report json");
  }

  if (!args.report_text_path.empty()) {
    std::ofstream text_output(args.report_text_path);
    text_output << text_summary.str();
  }

  std::cout << text_summary.str();
  return root;
}



template nlohmann::json run_benchmark<L2Distance>(ComputeService<L2Distance>& service, const Args& args);
template nlohmann::json run_benchmark<IPDistance>(ComputeService<IPDistance>& service, const Args& args);

}  // namespace tools::breakdown_benchmark
