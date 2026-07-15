#include "tools/breakdown_benchmark/workload.hh"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "common/vector_dtype.hh"
#include "service/breakdown.hh"
#include "tools/breakdown_benchmark/dataset.hh"
#include "tools/breakdown_benchmark/maintenance_log.hh"
#include "tools/breakdown_benchmark/progress.hh"
#include "tools/breakdown_benchmark/report.hh"
#include "vamana/vamana_node.hh"

namespace tools::breakdown_benchmark {

struct MixedPhaseStats {
  uint32_t next_insert_id{};
  size_t issued_reads{};
  size_t issued_writes{};
  size_t issued_inserts{};
  size_t issued_upserts{};
  size_t issued_deletes{};
  size_t completed_reads{};
  size_t completed_writes{};
  size_t completed_inserts{};
  size_t completed_upserts{};
  size_t completed_deletes{};
  uint64_t scheduled_reads{};
  uint64_t scheduled_writes{};
  double drain_seconds{};
};

struct InsertPhaseStats {
  uint32_t next_insert_id{};
  size_t completed{};
  double drain_seconds{};
};

struct QueryPhaseStats {
  size_t completed{};
  double drain_seconds{};
};

nlohmann::json run_benchmark(ComputeService& service, const Args& args) {
  using SampleReport = service::breakdown::Report;
  using service::breakdown::report_to_json;

  const bool use_insert_file = !args.insert_file.empty();
  const bool workload_has_queries =
    !args.recall_only &&
    (args.workload == "query" || args.workload == "both" ||
     (args.workload == "mixed" &&
      (args.mixed_mode == "rate_limited" ? args.target_query_qps > 0.0
                                         : args.read_ratio > 0.0)));

  nlohmann::json root;
  root["meta"] = {
    {"workload", args.workload},
    {"warmup_ops", args.warmup_ops},
    {"measure_ops", args.measure_ops},
    {"warmup_seconds", args.warmup_seconds},
    {"measure_seconds", args.measure_seconds},
    {"run_mode", (args.warmup_seconds > 0 || args.measure_seconds > 0) ? "time" : "ops"},
    {"recall_only", args.recall_only},
    {"time_completion_policy", "drain"},
    {"time_issue_policy", args.mixed_mode == "fixed_threads"
       ? "fixed_read_write_threads_until_deadline"
       : (args.mixed_mode == "rate_limited"
            ? "shared_two_stream_pacer_until_deadline"
            : "probabilistic_read_write_per_thread_until_deadline")},
    {"mixed_dispatch_policy", args.mixed_mode},
    {"vector_data_type", VamanaNode::vector_dtype_name()},
    {"vector_component_size", VamanaNode::vector_component_size()},
    {"vector_bytes", VamanaNode::vector_bytes()},
    {"node_size", VamanaNode::total_size()},
    {"candidate_vector_rdma_bytes", VamanaNode::vector_bytes()},
    {"effective_bytes_per_vector", VamanaNode::vector_bytes()},
    {"operation_granularity", "single_vector"},
    {"insert_vector_source", use_insert_file ? args.insert_file : "deterministic_synthetic_from_insert_id"},
    {"client_threads", args.client_threads},
    {"read_ratio", args.read_ratio},
    {"target_query_qps", args.target_query_qps},
    {"target_write_qps", args.target_write_qps},
    {"write_insert_ratio", args.write_insert_ratio},
    {"write_upsert_ratio", args.write_upsert_ratio},
    {"write_delete_ratio", args.write_delete_ratio},
    {"recall_mode", args.recall_mode},
    {"recall_base_id_limit", args.recall_base_id_limit},
    {"insert_start_id", args.insert_start_id},
    {"index_prefix", normalize_acceptance_path(
       service.config().resolved_index_prefix().string())},
    {"dim", service.config().dim},
    {"threads", service.config().num_threads},
    {"fine_grained_breakdown_enabled", service.config().enable_breakdown},
    {"search", "gpu_persistent_opq_pq"},
    {"navigation_quantizer", "opq_pq"},
    {"traversal_beam_width", service.config().gpu_traversal_beam_width},
    {"final_rerank_width", service.config().gpu_final_rerank_width},
    {"max_expansions", service.config().gpu_max_expansions},
    {"entry_seed_count", service.config().gpu_entry_seed_count},
    {"gpu_query_slots", service.config().gpu_query_slots},
    {"gpu_rdma_qps", service.config().gpu_rdma_qps},
    {"gpu_graph_prefetch_depth", service.config().gpu_graph_prefetch_depth},
    {"gpu_adjacency_cache_mb", service.config().gpu_adjacency_cache_mb},
    {"gpu_exact_cache_mb", service.config().gpu_exact_cache_mb},
  };
  const size_t dim = service.config().dim;
  const double write_ratio_sum = args.write_insert_ratio + args.write_upsert_ratio + args.write_delete_ratio;
  const double normalized_insert_ratio = args.write_insert_ratio / write_ratio_sum;
  const double normalized_upsert_ratio = args.write_upsert_ratio / write_ratio_sum;
  const double normalized_delete_ratio = args.write_delete_ratio / write_ratio_sum;
  root["meta"]["normalized_write_mix"] = {
    {"insert", normalized_insert_ratio},
    {"upsert", normalized_upsert_ratio},
    {"delete", normalized_delete_ratio},
  };
  size_t fixed_read_threads = 0;
  size_t fixed_write_threads = 0;
  if (args.workload == "mixed" && args.mixed_mode == "fixed_threads") {
    if (args.read_ratio <= 0.0) {
      fixed_read_threads = 0;
    } else if (args.read_ratio >= 1.0) {
      fixed_read_threads = args.client_threads;
    } else {
      fixed_read_threads = static_cast<size_t>(std::llround(static_cast<double>(args.client_threads) * args.read_ratio));
      fixed_read_threads = std::clamp<size_t>(fixed_read_threads, 1, args.client_threads - 1);
    }
    fixed_write_threads = args.client_threads - fixed_read_threads;
    root["meta"]["mixed_fixed_threads"] = {
      {"read_threads", fixed_read_threads},
      {"write_threads", fixed_write_threads},
    };
    std::cerr << "[breakdown] mixed fixed thread split: reads=" << fixed_read_threads
              << ", writes=" << fixed_write_threads << std::endl;
  }
  const size_t bootstrap_work = args.measure_seconds > 0
                                  ? std::max<size_t>(4096, args.client_threads * 256)
                                  : std::max<size_t>(2048, args.measure_ops);
  const size_t bootstrap_count = bootstrap_work;
  std::cerr << "[breakdown] preparing workload: bootstrap_count=" << bootstrap_count
            << ", workload=" << args.workload << std::endl;

  // Load insert vectors from file if specified, otherwise use synthetic data.
  VectorRows insert_rows;
  if (use_insert_file) {
    insert_rows = read_vector_rows(args.insert_file, true);
    if (insert_rows.dim != dim) {
      throw std::runtime_error("insert-file dim mismatch: " + std::to_string(insert_rows.dim)
                               + " vs config " + std::to_string(dim));
    }
    std::cerr << "[breakdown] insert-file vectors: count=" << insert_rows.count
              << " dim=" << insert_rows.dim << " dtype=" << vector_dtype_name(insert_rows.dtype)
              << std::endl;
  }

  auto get_insert_vector = [&](uint32_t id) -> vec<element_t> {
    if (use_insert_file) {
      size_t row = id % insert_rows.count;
      return vec<element_t>(insert_rows.decoded.begin() + row * dim,
                            insert_rows.decoded.begin() + (row + 1) * dim);
    }
    auto gen = make_deterministic_vector(id, dim);
    return vec<element_t>(gen.begin(), gen.end());
  };

  auto get_update_vector = [&](uint32_t target_id, uint32_t version) -> vec<element_t> {
    return get_insert_vector(target_id ^ (0x9e3779b9u * (version + 1u)));
  };

  std::vector<uint32_t> bootstrap_ids(bootstrap_count);
  std::iota(bootstrap_ids.begin(), bootstrap_ids.end(), 1);
  const auto bootstrap_vectors = make_dataset(bootstrap_ids, dim);
  std::cerr << "[breakdown] bootstrap vectors ready (synthetic)" << std::endl;
  std::vector<ProgressSample> measure_windows;

  auto run_insert_phase_ops = [&](const std::string& label, size_t ops,
                                  uint32_t start_id) -> size_t {
    std::atomic<size_t> completed_ops{0};
    std::atomic<size_t> next_op{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    ProgressReporter reporter(label, completed_ops, ops, 0,
                              nullptr, &completed_ops);
    const size_t worker_count = args.client_threads;
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(worker_count));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        start_barrier.arrive_and_wait();
        try {
          while (!failed.load(std::memory_order_acquire)) {
            const size_t op = next_op.fetch_add(1, std::memory_order_relaxed);
            if (op >= ops) break;
            const uint32_t id = start_id + static_cast<uint32_t>(op);
            vec<element_t> values = get_insert_vector(id);
            vec<ComputeService::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, std::move(values)});
            if (service.insert(insert_items) != 1) {
              throw std::runtime_error("singleton insert was rejected");
            }
            completed_ops.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error == nullptr) error = std::current_exception();
          }
          failed.store(true, std::memory_order_release);
        }
      });
    }
    for (auto& worker : workers) worker.join();
    reporter.finish();
    if (error != nullptr) std::rethrow_exception(error);
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    return completed_ops.load(std::memory_order_relaxed);
  };

  auto run_insert_phase_seconds = [&](const std::string& label, size_t seconds,
                                      uint32_t start_id) -> InsertPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> current_id{start_id};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    std::chrono::steady_clock::time_point deadline;
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads + 1));
    std::vector<std::thread> workers;
    workers.reserve(args.client_threads);
    for (size_t worker = 0; worker < args.client_threads; ++worker) {
      workers.emplace_back([&]() {
        start_barrier.arrive_and_wait();
        std::chrono::nanoseconds avg_insert_duration{0};
        size_t local_completed = 0;
        try {
          while (!failed.load(std::memory_order_acquire) &&
                 can_start_timed_operation(
                   deadline, avg_insert_duration, local_completed)) {
            const uint32_t id = current_id.fetch_add(1, std::memory_order_relaxed);
            vec<element_t> values = get_insert_vector(id);
            vec<ComputeService::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, std::move(values)});
            if (std::chrono::steady_clock::now() >= deadline) break;
            const auto started_at = std::chrono::steady_clock::now();
            if (service.insert(insert_items) != 1) {
              throw std::runtime_error("singleton insert was rejected");
            }
            update_avg_duration(avg_insert_duration, started_at, local_completed);
            completed_ops.fetch_add(1, std::memory_order_relaxed);
            ++local_completed;
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error == nullptr) error = std::current_exception();
          }
          failed.store(true, std::memory_order_release);
        }
      });
    }
    ProgressReporter reporter(label, completed_ops, 0, seconds,
                              nullptr, &completed_ops);
    const auto phase_start = std::chrono::steady_clock::now();
    deadline = phase_start + std::chrono::seconds(seconds);
    start_barrier.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    const double drain_seconds = std::max(
      0.0, std::chrono::duration<double>(
             std::chrono::steady_clock::now() - deadline).count());
    reporter.finish();
    if (error != nullptr) std::rethrow_exception(error);
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    return {
      .next_insert_id = current_id.load(std::memory_order_relaxed),
      .completed = completed_ops.load(std::memory_order_relaxed),
      .drain_seconds = drain_seconds,
    };
  };

  VectorRows recall_query_rows;
  if (!args.recall_query_file.empty()) {
    recall_query_rows = read_vector_rows(args.recall_query_file, false);
    if (recall_query_rows.dim != dim) {
      throw std::runtime_error("recall-query-file dim mismatch with service config");
    }
    std::cerr << "[breakdown] recall query data ready: count=" << recall_query_rows.count
              << " dtype=" << vector_dtype_name(recall_query_rows.dtype)
              << " vector_bytes=" << recall_query_rows.vector_bytes << std::endl;
  }

  VectorRows performance_query_rows;
  uint64_t rate_limited_required_query_rows = 0;
  if (workload_has_queries) {
    performance_query_rows = read_vector_rows(args.performance_query_file, false);
    if (performance_query_rows.dim != dim) {
      throw std::runtime_error("performance-query-file dim mismatch with service config");
    }
    std::cerr << "[breakdown] performance query data ready: count="
              << performance_query_rows.count
              << " dtype=" << vector_dtype_name(performance_query_rows.dtype)
              << " vector_bytes=" << performance_query_rows.vector_bytes
              << " policy=single_pass_no_reuse" << std::endl;
    if (args.workload == "mixed" && args.mixed_mode == "rate_limited") {
      const uint64_t required_rows = PacedOperationDispatcher::scheduled_count(
        args.target_query_qps, args.warmup_seconds) +
        PacedOperationDispatcher::scheduled_count(
          args.target_query_qps, args.measure_seconds);
      if (required_rows > performance_query_rows.count) {
        throw std::runtime_error(
          "rate-limited workload requires " + std::to_string(required_rows) +
          " unique performance query rows but the file contains " +
          std::to_string(performance_query_rows.count));
      }
      rate_limited_required_query_rows = required_rows;
    }
  }

  if (!args.recall_query_file.empty() && workload_has_queries &&
      std::filesystem::canonical(args.recall_query_file) ==
        std::filesystem::canonical(args.performance_query_file)) {
    throw std::runtime_error(
      "recall and performance query files must be different; the 10K recall set "
      "must not be reused for throughput measurement");
  }

  root["meta"]["recall_query"] = {
    {"source", args.recall_query_file},
    {"rows", recall_query_rows.count},
    {"data_type", recall_query_rows.count == 0 ? "" : vector_dtype_name(recall_query_rows.dtype)},
    {"vector_bytes", recall_query_rows.vector_bytes},
    {"purpose", "recall_only"},
  };
  root["meta"]["performance_query"] = {
    {"source", args.performance_query_file},
    {"canonical_source", workload_has_queries
      ? normalize_acceptance_path(args.performance_query_file) : ""},
    {"rows", performance_query_rows.count},
    {"data_type", performance_query_rows.count == 0 ? "" : vector_dtype_name(performance_query_rows.dtype)},
    {"vector_bytes", performance_query_rows.vector_bytes},
    {"row_reuse_policy", "single_pass_no_reuse"},
    {"row_reuse_count", 0},
    {"rate_limited_required_rows", rate_limited_required_query_rows},
  };

  const bool cold_cache = service.config().gpu_adjacency_cache_mb == 0 &&
    service.config().gpu_exact_cache_mb == 0;
  const nlohmann::json acceptance_fingerprint = {
    {"version", 1},
    {"index_prefix", root["meta"]["index_prefix"]},
    {"performance_query_file",
     root["meta"]["performance_query"]["canonical_source"]},
    {"dim", service.config().dim},
    {"vector_data_type", VamanaNode::vector_dtype_name()},
    {"graph_degree", service.config().R},
    {"query_k", service.config().k},
    {"gpu_traversal_beam_width",
     service.config().gpu_traversal_beam_width},
    {"gpu_final_rerank_width", service.config().gpu_final_rerank_width},
    {"gpu_max_expansions", service.config().gpu_max_expansions},
    {"gpu_entry_seed_count", service.config().gpu_entry_seed_count},
    {"gpu_graph_prefetch_depth",
     service.config().gpu_graph_prefetch_depth},
    {"gpu_query_slots", service.config().gpu_query_slots},
    {"gpu_rdma_qps", service.config().gpu_rdma_qps},
    {"client_threads", args.client_threads},
    {"gpu_adjacency_cache_mb",
     service.config().gpu_adjacency_cache_mb},
    {"gpu_exact_cache_mb", service.config().gpu_exact_cache_mb},
    {"cold_cache", cold_cache},
  };
  root["meta"]["acceptance_fingerprint"] = acceptance_fingerprint;

  std::optional<VerifiedQueryBaseline> verified_query_baseline;
  if (!args.query_baseline_report.empty()) {
    verified_query_baseline = load_verified_query_baseline(
      args.query_baseline_report, acceptance_fingerprint);
    root["meta"]["query_baseline"] = {
      {"source", "verified_report"},
      {"report_path", verified_query_baseline->report_path},
      {"effective_query_ops_per_sec",
       verified_query_baseline->effective_query_qps},
      {"fingerprint_verified", true},
      {"bare_query_baseline_qps_ignored",
       args.query_baseline_qps >= 0.0},
    };
  } else {
    root["meta"]["query_baseline"] = {
      {"source", args.query_baseline_qps >= 0.0 ? "bare_qps" : "disabled"},
      {"effective_query_ops_per_sec", args.query_baseline_qps},
      {"fingerprint_verified", false},
    };
  }

  SinglePassRowStream performance_query_stream(performance_query_rows.count);
  auto throw_if_performance_queries_exhausted = [&](const std::string& phase) {
    if (!workload_has_queries || !performance_query_stream.exhausted()) {
      return;
    }
    throw std::runtime_error(
      "performance query file exhausted during " + phase + " after " +
      std::to_string(performance_query_stream.capacity()) +
      " rows; rows are never reused. Provide a larger --performance-query-file "
      "or shorten the warmup/measurement duration");
  };

  bool recall_below_threshold = false;
  bool performance_below_threshold = false;
  std::optional<double> recall_before_performance;
  std::optional<double> recall_after_performance;
  auto run_recall_check = [&](const char* phase,
                              const char* key,
                              bool reset_after,
                              bool enforce_threshold) {
    if (args.groundtruth_file.empty()) {
      return;
    }
    if (recall_query_rows.count == 0) {
      throw std::runtime_error("recall requires query vectors");
    }
    const GroundTruth gt = read_groundtruth_bin(args.groundtruth_file);
    if (gt.rows != recall_query_rows.count) {
      throw std::runtime_error("recall-query/groundtruth row count mismatch");
    }
    const uint32_t recall_k = args.recall_k == 0 ? std::min<uint32_t>(service.config().k, gt.top_k) : args.recall_k;
    if (recall_k == 0 || recall_k > gt.top_k) {
      throw std::runtime_error("invalid recall k");
    }
    const size_t recall_queries = args.recall_queries == 0
      ? recall_query_rows.count
      : std::min<size_t>(args.recall_queries, recall_query_rows.count);
    const bool base_only = args.recall_mode == "base_only";
    const uint32_t recall_search_width = base_only
      ? service.config().gpu_final_rerank_width : recall_k;
    if (recall_search_width < recall_k) {
      throw std::runtime_error(
        "base-only recall requires gpu-final-rerank-width >= recall-k");
    }
    if (base_only) {
      for (size_t qi = 0; qi < recall_queries; ++qi) {
        const uint32_t* truth = gt.row(qi);
        for (uint32_t rank = 0; rank < recall_k; ++rank) {
          if (truth[rank] >= args.recall_base_id_limit) {
            throw std::runtime_error(
              "base-only recall ground truth contains an ID outside the "
              "configured base range");
          }
        }
      }
    }
    std::atomic<size_t> recall_completed{0};
    std::atomic<size_t> next_recall_query{0};
    std::atomic<bool> recall_failed{false};
    std::vector<double> per_query_recall(recall_queries, 0.0);
    std::atomic<size_t> insufficient_base_results{0};
    std::exception_ptr recall_error;
    std::mutex recall_error_mutex;
    ProgressReporter recall_reporter(key, recall_completed, recall_queries, 0);
    const size_t recall_workers = std::max<size_t>(
      1, std::min<size_t>(args.client_threads, recall_queries));
    std::vector<std::thread> workers;
    workers.reserve(recall_workers);
    for (size_t worker = 0; worker < recall_workers; ++worker) {
      workers.emplace_back([&]() {
        try {
          while (!recall_failed.load(std::memory_order_acquire)) {
            const size_t qi = next_recall_query.fetch_add(1, std::memory_order_relaxed);
            if (qi >= recall_queries) break;
            const auto results = service.search_raw(
              recall_query_rows.dtype, recall_query_rows.raw_row(qi), dim,
              recall_search_width);
            std::vector<uint32_t> result_ids;
            if (base_only) {
              result_ids = filter_base_only_recall_ids(
                results, args.recall_base_id_limit, recall_k);
              if (result_ids.size() < recall_k) {
                insufficient_base_results.fetch_add(
                  1, std::memory_order_relaxed);
              }
            } else {
              result_ids.reserve(results.size());
              for (const auto id : results) {
                result_ids.push_back(static_cast<uint32_t>(id));
              }
            }
            per_query_recall[qi] = recall_at(result_ids, gt.row(qi), recall_k);
            recall_completed.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(recall_error_mutex);
            if (recall_error == nullptr) recall_error = std::current_exception();
          }
          recall_failed.store(true, std::memory_order_release);
        }
      });
    }
    for (auto& worker : workers) worker.join();
    if (recall_error != nullptr) std::rethrow_exception(recall_error);
    recall_reporter.finish();
    const double total_recall = std::accumulate(
      per_query_recall.begin(), per_query_recall.end(), 0.0);
    const double recall = recall_queries > 0 ? total_recall / static_cast<double>(recall_queries) : 0.0;
    const size_t insufficient_queries =
      insufficient_base_results.load(std::memory_order_relaxed);
    const bool result_set_complete = insufficient_queries == 0;
    const bool recall_passed = result_set_complete &&
      (args.min_recall < 0.0 || recall >= args.min_recall);
    if (std::string_view(phase) == "before_performance") {
      recall_before_performance = recall;
    } else if (std::string_view(phase) == "after_performance") {
      recall_after_performance = recall;
    }
    root[key] = {
      {"phase", phase},
      {"query_file", args.recall_query_file},
      {"groundtruth_file", args.groundtruth_file},
      {"queries", recall_queries},
      {"k", recall_k},
      {"mode", base_only ? "base_only" : "all"},
      {"base_id_limit", args.recall_base_id_limit},
      {"search_result_width", recall_search_width},
      {"queries_with_insufficient_base_results", insufficient_queries},
      {"result_set_complete", result_set_complete},
      {"recall", recall},
      {"min_recall", args.min_recall},
      {"passed", recall_passed},
    };
    std::cerr << "[breakdown][recall] " << phase << " recall@" << recall_k << "=" << recall
              << " queries=" << recall_queries << std::endl;
    if (enforce_threshold && !recall_passed) {
      recall_below_threshold = true;
    }

    if (reset_after) {
      service.clear_thread_statistics();
      service.reset_breakdown_state();
    }
  };
  run_recall_check("before_performance", "recall", true, true);

  auto run_query_phase_ops = [&](const std::string& label, size_t ops) -> size_t {
    std::atomic<size_t> completed_ops{0};
    std::atomic<size_t> next_op{0};
    ProgressReporter reporter(label, completed_ops, ops, 0, &completed_ops, nullptr);
    const size_t worker_count = std::max<size_t>(1, std::min(args.client_threads, ops));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
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
      });
    }
    for (auto& worker : workers) worker.join();
    reporter.finish();
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    throw_if_performance_queries_exhausted(label);
    return completed_ops.load(std::memory_order_relaxed);
  };

  auto run_query_phase_seconds = [&](const std::string& label,
                                     size_t seconds) -> QueryPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::chrono::steady_clock::time_point deadline;
    std::barrier start_barrier(
      static_cast<std::ptrdiff_t>(args.client_threads + 1));
    std::vector<std::thread> workers;
    workers.reserve(args.client_threads);
    for (size_t worker = 0; worker < args.client_threads; ++worker) {
      workers.emplace_back([&]() {
        start_barrier.arrive_and_wait();
        while (!performance_query_stream.exhausted() &&
               std::chrono::steady_clock::now() < deadline) {
          const auto query_row = performance_query_stream.try_claim();
          if (!query_row.has_value()) break;
          (void)service.search_raw(
            performance_query_rows.dtype,
            performance_query_rows.raw_row(*query_row), dim, service.config().k);
          completed_ops.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    ProgressReporter reporter(label, completed_ops, 0, seconds,
                              &completed_ops, nullptr);
    const auto phase_start = std::chrono::steady_clock::now();
    deadline = phase_start + std::chrono::seconds(seconds);
    start_barrier.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    const double drain_seconds = std::max(
      0.0, std::chrono::duration<double>(
             std::chrono::steady_clock::now() - deadline).count());
    reporter.finish();
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    throw_if_performance_queries_exhausted(label);
    return {
      .completed = completed_ops.load(std::memory_order_relaxed),
      .drain_seconds = drain_seconds,
    };
  };

  auto choose_mixed_read = [&](std::mt19937_64& rng) {
    if (args.read_ratio <= 0.0) {
      return false;
    }
    if (args.read_ratio >= 1.0) {
      return true;
    }
    std::bernoulli_distribution read_dist(args.read_ratio);
    return read_dist(rng);
  };

  enum class WriteKind { insert, upsert, erase };
  auto choose_write_kind = [&](std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double pick = dist(rng);
    if (pick < normalized_insert_ratio) return WriteKind::insert;
    if (pick < normalized_insert_ratio + normalized_upsert_ratio) return WriteKind::upsert;
    return WriteKind::erase;
  };

  const uint32_t base_id_count = static_cast<uint32_t>(
    std::max<size_t>(1, service.config().max_vectors));
  auto sample_existing_id = [&](std::mt19937_64& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, base_id_count - 1);
    return dist(rng);
  };

  auto issue_mixed_write = [&](std::mt19937_64& rng,
                               std::atomic<uint32_t>& next_insert_id,
                               std::atomic<uint32_t>& next_update_version,
                               std::atomic<size_t>& issued_inserts,
                               std::atomic<size_t>& issued_upserts,
                               std::atomic<size_t>& issued_deletes,
                               std::atomic<size_t>& completed_inserts,
                               std::atomic<size_t>& completed_upserts,
                               std::atomic<size_t>& completed_deletes) -> bool {
    switch (choose_write_kind(rng)) {
      case WriteKind::insert: {
        issued_inserts.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
        vec<element_t> values = get_insert_vector(id);
        vec<ComputeService::InsertItem> items;
        items.push_back({id, std::move(values)});
        const bool succeeded = service.insert(items) == 1;
        if (succeeded) completed_inserts.fetch_add(1, std::memory_order_relaxed);
        return succeeded;
      }
      case WriteKind::upsert: {
        issued_upserts.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = sample_existing_id(rng);
        const uint32_t version = next_update_version.fetch_add(1, std::memory_order_relaxed);
        vec<element_t> values = get_update_vector(id, version);
        vec<ComputeService::InsertItem> items;
        items.push_back({id, std::move(values)});
        const bool succeeded = service.upsert(items) == 1;
        if (succeeded) completed_upserts.fetch_add(1, std::memory_order_relaxed);
        return succeeded;
      }
      case WriteKind::erase: {
        issued_deletes.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = sample_existing_id(rng);
        vec<node_t> ids;
        ids.push_back(id);
        const bool succeeded = service.erase(ids) == 1;
        if (succeeded) completed_deletes.fetch_add(1, std::memory_order_relaxed);
        return succeeded;
      }
    }
    return false;
  };

  auto run_mixed_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> issued_inserts{0};
    std::atomic<size_t> issued_upserts{0};
    std::atomic<size_t> issued_deletes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::atomic<size_t> completed_inserts{0};
    std::atomic<size_t> completed_upserts{0};
    std::atomic<size_t> completed_deletes{0};
    std::atomic<size_t> next_op{0};
    std::atomic<uint32_t> next_update_version{0};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, ops, 0,
                              &completed_reads, &completed_writes);

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        std::mt19937_64 rng(0x9e3779b97f4a7c15ull ^
                            (static_cast<uint64_t>(tid) << 32) ^
                            static_cast<uint64_t>(std::hash<std::string>{}(label)));
        start_barrier.arrive_and_wait();
        for (;;) {
          const size_t op_index = next_op.fetch_add(1, std::memory_order_relaxed);
          if (op_index >= ops) {
            break;
          }

          const bool read_op = args.mixed_mode == "fixed_threads" ? tid < fixed_read_threads : choose_mixed_read(rng);
          bool succeeded = true;
          if (read_op) {
            const auto query_row = performance_query_stream.try_claim();
            if (!query_row.has_value()) break;
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            (void)service.search_raw(
              performance_query_rows.dtype,
              performance_query_rows.raw_row(*query_row), dim, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            succeeded = issue_mixed_write(
              rng, next_insert_id, next_update_version,
              issued_inserts, issued_upserts, issued_deletes,
              completed_inserts, completed_upserts, completed_deletes);
            if (succeeded) completed_writes.fetch_add(1, std::memory_order_relaxed);
          }
          if (succeeded) completed_ops.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    reporter.finish();
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    throw_if_performance_queries_exhausted(label);
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .issued_inserts = issued_inserts.load(std::memory_order_relaxed),
      .issued_upserts = issued_upserts.load(std::memory_order_relaxed),
      .issued_deletes = issued_deletes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
      .completed_inserts = completed_inserts.load(std::memory_order_relaxed),
      .completed_upserts = completed_upserts.load(std::memory_order_relaxed),
      .completed_deletes = completed_deletes.load(std::memory_order_relaxed),
    };
  };

  auto run_mixed_phase_seconds = [&](const std::string& label, size_t seconds,
                                     uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> issued_inserts{0};
    std::atomic<size_t> issued_upserts{0};
    std::atomic<size_t> issued_deletes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::atomic<size_t> completed_inserts{0};
    std::atomic<size_t> completed_upserts{0};
    std::atomic<size_t> completed_deletes{0};
    std::atomic<uint32_t> next_update_version{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    std::barrier start_barrier(
      static_cast<std::ptrdiff_t>(args.client_threads + 1));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    std::chrono::steady_clock::time_point deadline;
    std::optional<PacedOperationDispatcher> pacer;
    if (args.mixed_mode == "rate_limited") {
      pacer.emplace(args.target_query_qps, args.target_write_qps);
    }

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        std::mt19937_64 rng(0xd1b54a32d192ed03ull ^
                            (static_cast<uint64_t>(tid) << 32) ^
                            static_cast<uint64_t>(std::hash<std::string>{}(label)));
        start_barrier.arrive_and_wait();
        try {
          while (!failed.load(std::memory_order_acquire)) {
            bool read_op = false;
            if (pacer.has_value()) {
              const auto claim = pacer->claim();
              if (!claim.has_value()) break;
              read_op = claim->kind == PacedOperationKind::query;
            } else {
              if (std::chrono::steady_clock::now() >= deadline) break;
              read_op = args.mixed_mode == "fixed_threads"
                ? tid < fixed_read_threads : choose_mixed_read(rng);
            }

            bool succeeded = true;
            if (read_op) {
              const auto query_row = performance_query_stream.try_claim();
              if (!query_row.has_value()) break;
              issued_reads.fetch_add(1, std::memory_order_relaxed);
              (void)service.search_raw(
                performance_query_rows.dtype,
                performance_query_rows.raw_row(*query_row), dim,
                service.config().k);
              completed_reads.fetch_add(1, std::memory_order_relaxed);
            } else {
              issued_writes.fetch_add(1, std::memory_order_relaxed);
              succeeded = issue_mixed_write(
                rng, next_insert_id, next_update_version,
                issued_inserts, issued_upserts, issued_deletes,
                completed_inserts, completed_upserts, completed_deletes);
              if (succeeded) {
                completed_writes.fetch_add(1, std::memory_order_relaxed);
              }
            }
            if (succeeded) completed_ops.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error == nullptr) error = std::current_exception();
          }
          failed.store(true, std::memory_order_release);
        }
      });
    }

    ProgressReporter reporter(label, completed_ops, 0, seconds,
                              &completed_reads, &completed_writes);
    const auto phase_start = std::chrono::steady_clock::now();
    deadline = phase_start + std::chrono::seconds(seconds);
    if (pacer.has_value()) pacer->start(phase_start, deadline);
    start_barrier.arrive_and_wait();
    for (auto& thread : threads) {
      thread.join();
    }
    const double drain_seconds = std::max(
      0.0, std::chrono::duration<double>(
             std::chrono::steady_clock::now() - deadline).count());
    reporter.finish();
    if (error != nullptr) std::rethrow_exception(error);
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    throw_if_performance_queries_exhausted(label);
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .issued_inserts = issued_inserts.load(std::memory_order_relaxed),
      .issued_upserts = issued_upserts.load(std::memory_order_relaxed),
      .issued_deletes = issued_deletes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
      .completed_inserts = completed_inserts.load(std::memory_order_relaxed),
      .completed_upserts = completed_upserts.load(std::memory_order_relaxed),
      .completed_deletes = completed_deletes.load(std::memory_order_relaxed),
      .scheduled_reads = args.mixed_mode == "rate_limited"
        ? PacedOperationDispatcher::scheduled_count(args.target_query_qps, seconds)
        : 0,
      .scheduled_writes = args.mixed_mode == "rate_limited"
        ? PacedOperationDispatcher::scheduled_count(args.target_write_qps, seconds)
        : 0,
      .drain_seconds = drain_seconds,
    };
  };

  uint32_t next_insert_id = args.insert_start_id;
  if (next_insert_id == 0) {
    const uint64_t default_start =
      static_cast<uint64_t>(service.config().max_vectors) + 10'000ull;
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
  double measure_client_drain_seconds = 0.0;
  std::vector<MaintenanceLogCursor> maintenance_log_cursors;
  const auto wait_for_gpu_publications = [&](const char* phase) {
    const auto started = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(std::clamp<u64>(
      static_cast<u64>(service.config().storage_owner_rpc_timeout_ms) * 3,
      5000, 60000));
    const auto deadline = started + timeout;
    for (;;) {
      const auto telemetry = service.gpu_search_telemetry();
      if (telemetry.mutation_capacity_reserved == 0) {
        return std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
          std::string{"GPU mutation publication did not drain during "} +
          phase);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  if (!args.recall_only && (args.workload == "insert" || args.workload == "both")) {
    std::cerr << "[breakdown] starting warmup insert" << std::endl;
    if (use_time_mode) {
      const auto warmup = run_insert_phase_seconds(
        "warmup-insert", args.warmup_seconds, next_insert_id);
      next_insert_id = warmup.next_insert_id + 1024;
    } else {
      (void)run_insert_phase_ops("warmup-insert", args.warmup_ops, next_insert_id);
      next_insert_id += static_cast<uint32_t>(args.warmup_ops + 1024);
    }
  }
  if (!args.recall_only && (args.workload == "query" || args.workload == "both")) {
    std::cerr << "[breakdown] starting warmup query" << std::endl;
    if (use_time_mode) {
      (void)run_query_phase_seconds("warmup-query", args.warmup_seconds);
    } else {
      (void)run_query_phase_ops("warmup-query", args.warmup_ops);
    }
  }
  if (!args.recall_only && args.workload == "mixed") {
    std::cerr << "[breakdown] starting warmup mixed" << std::endl;
    if (use_time_mode) {
      warmup_mixed_stats = run_mixed_phase_seconds("warmup-mixed", args.warmup_seconds, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    } else {
      warmup_mixed_stats = run_mixed_phase_ops("warmup-mixed", args.warmup_ops, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    }
  }

  const size_t performance_queries_after_warmup = performance_query_stream.consumed();
  (void)wait_for_gpu_publications("warmup");
  if (!args.storage_maintenance_logs.empty()) {
    maintenance_log_cursors = snapshot_maintenance_logs(
      args.storage_maintenance_logs);
  }
  service.clear_thread_statistics();
  service.reset_breakdown_state();
  std::cerr << "[breakdown] starting measure phase" << std::endl;
  size_t measured_query_operations = 0;
  size_t measured_insert_operations = 0;

  if (!args.recall_only && (args.workload == "insert" || args.workload == "both")) {
    if (use_time_mode) {
      const auto measured = run_insert_phase_seconds(
        "measure-insert", args.measure_seconds, next_insert_id);
      next_insert_id = measured.next_insert_id;
      measured_insert_operations = measured.completed;
      measure_client_drain_seconds = measured.drain_seconds;
    } else {
      measured_insert_operations = run_insert_phase_ops(
        "measure-insert", args.measure_ops, next_insert_id);
    }
  }
  if (!args.recall_only && (args.workload == "query" || args.workload == "both")) {
    if (use_time_mode) {
      const auto measured = run_query_phase_seconds(
        "measure-query", args.measure_seconds);
      measured_query_operations = measured.completed;
      measure_client_drain_seconds = measured.drain_seconds;
    } else {
      measured_query_operations = run_query_phase_ops("measure-query", args.measure_ops);
    }
  }
  if (!args.recall_only && args.workload == "mixed") {
    if (use_time_mode) {
      measure_mixed_stats = run_mixed_phase_seconds("measure-mixed", args.measure_seconds, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
      measure_client_drain_seconds = measure_mixed_stats.drain_seconds;
    } else {
      measure_mixed_stats = run_mixed_phase_ops("measure-mixed", args.measure_ops, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
    }
  }

  MaintenanceLogSummary load_maintenance_summary;
  MaintenanceLogSummary drain_maintenance_summary;
  MaintenanceLogSummary full_maintenance_summary;
  std::vector<MaintenanceLogCursor> post_stop_cursors;
  double stage2_drain_seconds = 0.0;
  bool stage2_drain_timed_out = false;
  if (!maintenance_log_cursors.empty()) {
    // Freeze the load window before waiting for stage2 drain. In particular,
    // post-stop observations must never change the load backlog slope.
    post_stop_cursors = snapshot_maintenance_logs(
      args.storage_maintenance_logs);
    load_maintenance_summary = summarize_maintenance_log_window(
      maintenance_log_cursors, post_stop_cursors);
  }

  const double gpu_publication_drain_seconds =
    wait_for_gpu_publications("measurement");

  if (!maintenance_log_cursors.empty()) {
    const auto drain_started = std::chrono::steady_clock::now();
    const uint64_t remaining_target = args.max_stage2_remaining < 0
      ? 0 : static_cast<uint64_t>(args.max_stage2_remaining);
    for (;;) {
      drain_maintenance_summary = summarize_maintenance_logs(post_stop_cursors);
      const bool observed_every_log =
        drain_maintenance_summary.logs_with_observations ==
          drain_maintenance_summary.requested_logs &&
        drain_maintenance_summary.requested_logs != 0;
      if (observed_every_log &&
          drain_maintenance_summary.remaining <= remaining_target) {
        break;
      }
      stage2_drain_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - drain_started).count();
      if (args.stage2_drain_timeout_seconds == 0) {
        // Zero retains the old non-waiting behavior, but lack of a post-stop
        // observation is now an explicit failure instead of a remaining=0
        // false pass.
        stage2_drain_timed_out = true;
        break;
      }
      if (stage2_drain_seconds >=
          static_cast<double>(args.stage2_drain_timeout_seconds)) {
        stage2_drain_timed_out = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    stage2_drain_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - drain_started).count();
    // Failures and stitch-delay samples cover the complete measured lifetime,
    // including work issued during load that only finalizes while draining.
    // Their cumulative counters are still differenced from the pre-load cursor.
    full_maintenance_summary = summarize_maintenance_logs(
      maintenance_log_cursors);
  }

  const size_t expected_gpu_mutations = args.workload == "mixed"
    ? measure_mixed_stats.completed_writes : measured_insert_operations;
  const bool gpu_visibility_drain_requested =
    args.max_gpu_visibility_ms >= 0.0 && expected_gpu_mutations != 0;
  const bool gpu_final_state_drain_requested =
    gpu_visibility_drain_requested ||
    args.max_final_mutation_capacity_reserved >= 0 ||
    args.max_final_delta_mutable_entries >= 0;
  double gpu_final_state_drain_seconds = 0.0;
  bool gpu_final_state_drain_timed_out = false;
  gpu_search::TelemetrySnapshot final_gpu_telemetry =
    service.gpu_search_telemetry();
  if (gpu_final_state_drain_requested) {
    const auto drain_started = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(std::clamp<u64>(
      static_cast<u64>(service.config().storage_owner_rpc_timeout_ms) * 3,
      5000, 60000));
    const auto deadline = drain_started + timeout;
    for (;;) {
      final_gpu_telemetry = service.gpu_search_telemetry();
      const bool capacity_ready =
        args.max_final_mutation_capacity_reserved < 0 ||
        final_gpu_telemetry.mutation_capacity_reserved <=
          static_cast<u64>(args.max_final_mutation_capacity_reserved);
      const bool mutable_ready = args.max_final_delta_mutable_entries < 0 ||
        final_gpu_telemetry.delta_mutable_entries <=
          static_cast<u64>(args.max_final_delta_mutable_entries);
      const bool visibility_ready = !gpu_visibility_drain_requested ||
        (final_gpu_telemetry.mutation_capacity_reserved == 0 &&
         final_gpu_telemetry.mutations_published == expected_gpu_mutations);
      if (capacity_ready && mutable_ready && visibility_ready) break;
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        gpu_final_state_drain_timed_out = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gpu_final_state_drain_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - drain_started).count();
  }

  const size_t total_performance_queries = performance_query_stream.consumed();
  root["meta"]["performance_query"]["warmup_rows_consumed"] =
    performance_queries_after_warmup;
  root["meta"]["performance_query"]["measure_rows_consumed"] =
    total_performance_queries - performance_queries_after_warmup;
  root["meta"]["performance_query"]["total_rows_consumed"] =
    total_performance_queries;
  root["meta"]["performance_query"]["remaining_rows"] =
    performance_query_stream.capacity() - total_performance_queries;

  if (!args.recall_only && args.workload == "mixed") {
    root["meta"]["warmup_mixed"] = {
      {"issued_reads", warmup_mixed_stats.issued_reads},
      {"issued_writes", warmup_mixed_stats.issued_writes},
      {"issued_inserts", warmup_mixed_stats.issued_inserts},
      {"issued_upserts", warmup_mixed_stats.issued_upserts},
      {"issued_deletes", warmup_mixed_stats.issued_deletes},
      {"completed_reads", warmup_mixed_stats.completed_reads},
      {"completed_writes", warmup_mixed_stats.completed_writes},
      {"completed_inserts", warmup_mixed_stats.completed_inserts},
      {"completed_upserts", warmup_mixed_stats.completed_upserts},
      {"completed_deletes", warmup_mixed_stats.completed_deletes},
      {"scheduled_reads", warmup_mixed_stats.scheduled_reads},
      {"scheduled_writes", warmup_mixed_stats.scheduled_writes},
      {"drain_seconds", warmup_mixed_stats.drain_seconds},
    };
    root["meta"]["measure_mixed"] = {
      {"issued_reads", measure_mixed_stats.issued_reads},
      {"issued_writes", measure_mixed_stats.issued_writes},
      {"issued_inserts", measure_mixed_stats.issued_inserts},
      {"issued_upserts", measure_mixed_stats.issued_upserts},
      {"issued_deletes", measure_mixed_stats.issued_deletes},
      {"completed_reads", measure_mixed_stats.completed_reads},
      {"completed_writes", measure_mixed_stats.completed_writes},
      {"completed_inserts", measure_mixed_stats.completed_inserts},
      {"completed_upserts", measure_mixed_stats.completed_upserts},
      {"completed_deletes", measure_mixed_stats.completed_deletes},
      {"scheduled_reads", measure_mixed_stats.scheduled_reads},
      {"scheduled_writes", measure_mixed_stats.scheduled_writes},
      {"drain_seconds", measure_mixed_stats.drain_seconds},
    };
    std::cerr << "[breakdown][measure-mixed] reads issued/completed=" << measure_mixed_stats.issued_reads << "/"
              << measure_mixed_stats.completed_reads << ", writes issued/completed="
              << measure_mixed_stats.issued_writes << "/" << measure_mixed_stats.completed_writes
              << " (insert=" << measure_mixed_stats.completed_inserts
              << ", upsert=" << measure_mixed_stats.completed_upserts
              << ", delete=" << measure_mixed_stats.completed_deletes << ")" << std::endl;
    const bool reads_expected = args.mixed_mode == "rate_limited"
      ? args.target_query_qps > 0.0 : args.read_ratio > 0.0;
    const bool writes_expected = args.mixed_mode == "rate_limited"
      ? args.target_write_qps > 0.0 : args.read_ratio < 1.0;
    if (reads_expected) {
      lib_assert(measure_mixed_stats.completed_reads > 0, "mixed benchmark completed zero reads");
    }
    if (writes_expected) {
      lib_assert(measure_mixed_stats.completed_writes > 0, "mixed benchmark completed zero writes");
    }
  }

  const SampleReport report = service.collect_breakdown_report();
  root.update(report_to_json(report));
  // Keep the post-drain snapshot used by the final-state gates. Later recall
  // queries may change query counters, but cannot make a mutation final-state
  // failure pass retroactively.
  root["gpu_persistent"] = telemetry_to_json(final_gpu_telemetry);
  const u64 late_storage_owner_rpcs =
    service.late_storage_owner_rpc_completions();
  root["storage_owner_runtime"] = {
    {"late_rpc_completions", late_storage_owner_rpcs},
    {"late_rpc_threshold_ms", service.config().storage_owner_rpc_timeout_ms},
  };


  const bool has_throughput_duration = use_time_mode && args.measure_seconds > 0;
  const double throughput_duration = has_throughput_duration
    ? static_cast<double>(args.measure_seconds) + measure_client_drain_seconds
    : 0.0;
  const size_t throughput_query_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_reads
    : (service.config().enable_breakdown ? report.query.count : measured_query_operations);
  const size_t throughput_write_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_writes
    : (service.config().enable_breakdown ? report.insert.count : measured_insert_operations);
  const size_t throughput_insert_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_inserts
    : (service.config().enable_breakdown ? report.insert.count : measured_insert_operations);
  const double effective_query_throughput = has_throughput_duration
    ? static_cast<double>(throughput_query_ops) / throughput_duration : 0.0;
  const double effective_write_throughput = has_throughput_duration
    ? static_cast<double>(throughput_write_ops) / throughput_duration : 0.0;
  const double effective_insert_throughput = has_throughput_duration
    ? static_cast<double>(throughput_insert_ops) / throughput_duration : 0.0;
  const bool rate_limited_measurement = args.workload == "mixed" &&
    args.mixed_mode == "rate_limited" && has_throughput_duration;
  const double configured_measure_duration =
    static_cast<double>(args.measure_seconds);
  const double query_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_query_ops) / configured_measure_duration
    : effective_query_throughput;
  const double write_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_write_ops) / configured_measure_duration
    : effective_write_throughput;
  const double insert_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_insert_ops) / configured_measure_duration
    : effective_insert_throughput;
  const double total_throughput = query_throughput + write_throughput;
  root["throughput"] = {
    {"configured_measure_seconds", args.measure_seconds},
    {"effective_measure_seconds", throughput_duration},
    {"duration_seconds", throughput_duration},
    {"client_drain_seconds", measure_client_drain_seconds},
    {"gpu_publication_drain_seconds", gpu_publication_drain_seconds},
    {"gpu_final_state_drain_seconds", gpu_final_state_drain_seconds},
    {"gpu_final_state_drain_timed_out", gpu_final_state_drain_timed_out},
    {"total_ops", throughput_query_ops + throughput_write_ops},
    {"total_ops_per_sec", total_throughput},
    {"query_ops", throughput_query_ops},
    {"query_ops_per_sec", query_throughput},
    {"nominal_query_ops_per_sec", query_throughput},
    {"effective_query_ops_per_sec", effective_query_throughput},
    {"write_ops", throughput_write_ops},
    {"write_ops_per_sec", write_throughput},
    {"nominal_write_ops_per_sec", write_throughput},
    {"effective_write_ops_per_sec", effective_write_throughput},
    {"insert_ops", throughput_insert_ops},
    {"insert_ops_per_sec", insert_throughput},
    {"nominal_insert_ops_per_sec", insert_throughput},
    {"effective_insert_ops_per_sec", effective_insert_throughput},
    {"scheduled_query_ops", measure_mixed_stats.scheduled_reads},
    {"scheduled_write_ops", measure_mixed_stats.scheduled_writes},
    {"query_rate_attainment_ratio", measure_mixed_stats.scheduled_reads == 0
      ? 1.0 : static_cast<double>(throughput_query_ops) /
          static_cast<double>(measure_mixed_stats.scheduled_reads)},
    {"write_rate_attainment_ratio", measure_mixed_stats.scheduled_writes == 0
      ? 1.0 : static_cast<double>(throughput_write_ops) /
          static_cast<double>(measure_mixed_stats.scheduled_writes)},
    {"nominal_rate_basis", rate_limited_measurement
      ? "configured_schedule_window" : "effective_wall_clock"},
    {"effective_rate_basis", "wall_clock_including_client_drain"},
    {"upsert_ops", args.workload == "mixed" ? measure_mixed_stats.completed_upserts : 0},
    {"upsert_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_upserts) / throughput_duration
      : 0.0},
    {"delete_ops", args.workload == "mixed" ? measure_mixed_stats.completed_deletes : 0},
    {"delete_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_deletes) / throughput_duration
      : 0.0},
  };

  nlohmann::json window_json = nlohmann::json::array();
  std::vector<double> total_window_rates;
  std::vector<double> query_window_rates;
  std::vector<double> write_window_rates;
  size_t zero_completion_windows = 0;
  size_t zero_query_windows = 0;
  size_t zero_write_windows = 0;
  const bool query_windows_expected = args.workload == "query" ||
    args.workload == "both" ||
    (args.workload == "mixed" &&
     (args.mixed_mode == "rate_limited" ? args.target_query_qps > 0.0
                                        : args.read_ratio > 0.0));
  const bool write_windows_expected = args.workload == "insert" ||
    args.workload == "both" ||
    (args.workload == "mixed" &&
     (args.mixed_mode == "rate_limited" ? args.target_write_qps > 0.0
                                        : args.read_ratio < 1.0));
  // The reporter is stopped only after synchronous calls already issued at
  // the deadline return. Keep those completions in throughput/drain counters,
  // but do not let a long final drain interval masquerade as the load tail or
  // create a false zero-completion window.
  const double stability_window_deadline =
    static_cast<double>(args.measure_seconds) + 0.5;
  for (const ProgressSample& sample : measure_windows) {
    const bool within_measurement_window = !has_throughput_duration ||
      sample.elapsed_seconds <= stability_window_deadline;
    window_json.push_back({
      {"elapsed_seconds", sample.elapsed_seconds},
      {"interval_seconds", sample.interval_seconds},
      {"completed_ops", sample.completed_ops},
      {"interval_ops", sample.interval_ops},
      {"interval_reads", sample.interval_reads},
      {"interval_writes", sample.interval_writes},
      {"total_ops_per_sec", sample.total_ops_per_sec},
      {"query_ops_per_sec", sample.query_ops_per_sec},
      {"write_ops_per_sec", sample.write_ops_per_sec},
      {"within_measurement_window", within_measurement_window},
    });
    if (within_measurement_window && sample.interval_seconds >= 2.5) {
      total_window_rates.push_back(sample.total_ops_per_sec);
      query_window_rates.push_back(sample.query_ops_per_sec);
      write_window_rates.push_back(sample.write_ops_per_sec);
      if (sample.interval_ops == 0) ++zero_completion_windows;
      if (query_windows_expected && sample.interval_reads == 0) {
        ++zero_query_windows;
      }
      if (write_windows_expected && sample.interval_writes == 0) {
        ++zero_write_windows;
      }
    }
  }
  auto edge_mean = [](const std::vector<double>& values, bool tail) {
    if (values.empty()) return 0.0;
    const size_t count = std::min<size_t>(3, values.size());
    const size_t begin = tail ? values.size() - count : 0;
    return std::accumulate(values.begin() + begin,
                           values.begin() + begin + count, 0.0) /
      static_cast<double>(count);
  };
  const double total_head_qps = edge_mean(total_window_rates, false);
  const double total_tail_qps = edge_mean(total_window_rates, true);
  const double query_head_qps = edge_mean(query_window_rates, false);
  const double query_tail_qps = edge_mean(query_window_rates, true);
  const double write_head_qps = edge_mean(write_window_rates, false);
  const double write_tail_qps = edge_mean(write_window_rates, true);
  root["stability"] = {
    {"window_seconds", 5},
    {"windows", std::move(window_json)},
    {"zero_completion_windows", zero_completion_windows},
    {"zero_query_windows", zero_query_windows},
    {"zero_write_windows", zero_write_windows},
    {"total_head_ops_per_sec", total_head_qps},
    {"total_tail_ops_per_sec", total_tail_qps},
    {"total_tail_to_head_ratio", total_head_qps == 0.0 ? 0.0
      : total_tail_qps / total_head_qps},
    {"query_head_ops_per_sec", query_head_qps},
    {"query_tail_ops_per_sec", query_tail_qps},
    {"query_tail_to_head_ratio", query_head_qps == 0.0 ? 0.0
      : query_tail_qps / query_head_qps},
    {"write_head_ops_per_sec", write_head_qps},
    {"write_tail_ops_per_sec", write_tail_qps},
    {"write_tail_to_head_ratio", write_head_qps == 0.0 ? 0.0
      : write_tail_qps / write_head_qps},
    {"single_pass_no_reuse", true},
    {"cache_independent_baseline",
      service.config().gpu_adjacency_cache_mb == 0 &&
      service.config().gpu_exact_cache_mb == 0},
  };
  if (!args.recall_only) {
    run_recall_check("after_performance", "static_gt_post_recall", false, true);
  }

  auto unreadable_logs_json = [](const MaintenanceLogSummary& summary) {
    nlohmann::json paths = nlohmann::json::array();
    for (const auto& path : summary.unreadable_logs) paths.push_back(path);
    return paths;
  };
  root["stage2"] = {
    // Compatibility fields retain their old names, but now deliberately use
    // load-only backlog slope, post-stop remaining, and full-window deltas.
    {"requested_logs", full_maintenance_summary.requested_logs},
    {"readable_logs", full_maintenance_summary.readable_logs},
    {"logs_with_observations",
     full_maintenance_summary.logs_with_observations},
    {"logs_with_slope_observations",
     load_maintenance_summary.logs_with_slope_observations},
    {"observations", full_maintenance_summary.observations},
    {"unreadable_logs", unreadable_logs_json(full_maintenance_summary)},
    {"remaining", drain_maintenance_summary.remaining},
    {"max_backlog_observed",
     load_maintenance_summary.max_backlog_observed},
    {"backlog_slope_per_sec",
     load_maintenance_summary.backlog_slope_per_sec},
    {"backlog_slope_available",
     load_maintenance_summary.backlog_slope_available},
    {"p99_stitch_delay_upper_ms",
     full_maintenance_summary.p99_stitch_delay_upper_ms},
    {"p99_stitch_delay_over_30s",
     full_maintenance_summary.p99_stitch_delay_over_30s},
    {"p99_stitch_delay_samples",
     full_maintenance_summary.p99_stitch_delay_samples},
    {"p99_stitch_delay_available",
     full_maintenance_summary.p99_stitch_delay_available},
    {"failures", full_maintenance_summary.failures},
    {"failure_delta_available",
     full_maintenance_summary.failure_delta_available},
    {"drain_seconds", stage2_drain_seconds},
    {"drain_timed_out", stage2_drain_timed_out},
    {"observation_period_seconds_assumed", 5.0},
    {"load", {
      {"requested_logs", load_maintenance_summary.requested_logs},
      {"readable_logs", load_maintenance_summary.readable_logs},
      {"logs_with_observations",
       load_maintenance_summary.logs_with_observations},
      {"logs_with_slope_observations",
       load_maintenance_summary.logs_with_slope_observations},
      {"observations", load_maintenance_summary.observations},
      {"unreadable_logs", unreadable_logs_json(load_maintenance_summary)},
      {"remaining_at_stop", load_maintenance_summary.remaining},
      {"max_backlog_observed",
       load_maintenance_summary.max_backlog_observed},
      {"backlog_slope_per_sec",
       load_maintenance_summary.backlog_slope_per_sec},
      {"backlog_slope_available",
       load_maintenance_summary.backlog_slope_available},
      {"p99_stitch_delay_upper_ms",
       load_maintenance_summary.p99_stitch_delay_upper_ms},
      {"p99_stitch_delay_samples",
       load_maintenance_summary.p99_stitch_delay_samples},
      {"p99_stitch_delay_available",
       load_maintenance_summary.p99_stitch_delay_available},
      {"failures", load_maintenance_summary.failures},
      {"failure_delta_available",
       load_maintenance_summary.failure_delta_available},
    }},
    {"post_stop_drain", {
      {"requested_logs", drain_maintenance_summary.requested_logs},
      {"readable_logs", drain_maintenance_summary.readable_logs},
      {"logs_with_observations",
       drain_maintenance_summary.logs_with_observations},
      {"observations", drain_maintenance_summary.observations},
      {"unreadable_logs", unreadable_logs_json(drain_maintenance_summary)},
      {"remaining", drain_maintenance_summary.remaining},
      {"max_backlog_observed",
       drain_maintenance_summary.max_backlog_observed},
      {"p99_stitch_delay_upper_ms",
       drain_maintenance_summary.p99_stitch_delay_upper_ms},
      {"p99_stitch_delay_samples",
       drain_maintenance_summary.p99_stitch_delay_samples},
      {"p99_stitch_delay_available",
       drain_maintenance_summary.p99_stitch_delay_available},
      {"failures", drain_maintenance_summary.failures},
      {"failure_delta_available",
       drain_maintenance_summary.failure_delta_available},
    }},
  };

  const bool query_acceptance_applies =
    has_throughput_duration && query_windows_expected;
  const double query_stability_ratio = query_head_qps == 0.0
    ? 0.0 : query_tail_qps / query_head_qps;
  const double write_stability_ratio = write_head_qps == 0.0
    ? 0.0 : write_tail_qps / write_head_qps;
  const bool query_qps_passed = !query_acceptance_applies || args.min_query_qps < 0.0 ||
    query_throughput >= args.min_query_qps;
  const bool insert_acceptance_applies =
    has_throughput_duration && write_windows_expected;
  const bool insert_qps_passed = !insert_acceptance_applies || args.min_insert_qps < 0.0 ||
    insert_throughput >= args.min_insert_qps;
  const bool query_stability_passed = !query_acceptance_applies ||
    args.min_stability_ratio < 0.0 ||
    (query_stability_ratio >= args.min_stability_ratio && zero_completion_windows == 0);
  const bool write_acceptance_applies =
    has_throughput_duration && write_windows_expected;
  const bool write_stability_passed = !write_acceptance_applies ||
    args.min_write_stability_ratio < 0.0 ||
    write_stability_ratio >= args.min_write_stability_ratio;
  const bool zero_completion_windows_passed =
    args.max_zero_completion_windows < 0 ||
    zero_completion_windows <=
      static_cast<size_t>(args.max_zero_completion_windows);
  const bool zero_query_windows_passed =
    args.max_zero_query_windows < 0 ||
    zero_query_windows <= static_cast<size_t>(args.max_zero_query_windows);
  const bool zero_write_windows_passed =
    args.max_zero_write_windows < 0 ||
    zero_write_windows <= static_cast<size_t>(args.max_zero_write_windows);
  const bool writes_all_succeeded = !write_acceptance_applies ||
    args.workload != "mixed" ||
    measure_mixed_stats.issued_writes == measure_mixed_stats.completed_writes;
  const u64 mutation_capacity_rejections = root["gpu_persistent"].value(
    "mutation_capacity_rejections", 0ULL);
  const bool mutation_capacity_passed = !write_acceptance_applies ||
    mutation_capacity_rejections == 0;
  const double max_gpu_visibility_ms =
    static_cast<double>(final_gpu_telemetry.visibility_ns_max) / 1e6;
  const bool gpu_visibility_coverage_passed = !write_windows_expected ||
    (expected_gpu_mutations != 0 &&
     final_gpu_telemetry.mutations_published == expected_gpu_mutations);
  const bool gpu_visibility_available = gpu_visibility_coverage_passed;
  const bool gpu_visibility_passed = args.max_gpu_visibility_ms < 0.0 ||
    (gpu_visibility_available &&
     max_gpu_visibility_ms <= args.max_gpu_visibility_ms);
  const bool final_mutation_capacity_passed =
    args.max_final_mutation_capacity_reserved < 0 ||
    final_gpu_telemetry.mutation_capacity_reserved <=
      static_cast<u64>(args.max_final_mutation_capacity_reserved);
  const bool final_delta_mutable_passed =
    args.max_final_delta_mutable_entries < 0 ||
    final_gpu_telemetry.delta_mutable_entries <=
      static_cast<u64>(args.max_final_delta_mutable_entries);
  const bool gpu_final_state_drain_passed =
    !gpu_final_state_drain_requested || !gpu_final_state_drain_timed_out;
  const bool late_storage_owner_rpcs_passed =
    args.max_late_storage_owner_rpcs < 0 ||
    late_storage_owner_rpcs <=
      static_cast<u64>(args.max_late_storage_owner_rpcs);
  const bool gpu_final_state_checks_requested =
    args.max_gpu_visibility_ms >= 0.0 ||
    args.max_final_mutation_capacity_reserved >= 0 ||
    args.max_final_delta_mutable_entries >= 0 ||
    args.max_late_storage_owner_rpcs >= 0;
  const double query_rate_attainment = root["throughput"].value(
    "query_rate_attainment_ratio", 1.0);
  const double write_rate_attainment = root["throughput"].value(
    "write_rate_attainment_ratio", 1.0);
  const bool rate_attainment_passed = !rate_limited_measurement ||
    args.min_rate_attainment_ratio < 0.0 ||
    ((!query_windows_expected ||
      query_rate_attainment >= args.min_rate_attainment_ratio) &&
     (!write_windows_expected ||
      write_rate_attainment >= args.min_rate_attainment_ratio));
  const double baseline_effective_query_qps = verified_query_baseline
    ? verified_query_baseline->effective_query_qps : args.query_baseline_qps;
  const bool query_baseline_requested =
    args.min_query_baseline_ratio >= 0.0;
  const bool query_baseline_verified =
    verified_query_baseline.has_value();
  const double query_baseline_ratio =
    baseline_effective_query_qps > 0.0
      ? effective_query_throughput / baseline_effective_query_qps : 0.0;
  const bool query_baseline_passed = !query_baseline_requested ||
    ((args.workload != "mixed" || query_baseline_verified) &&
     baseline_effective_query_qps > 0.0 &&
     query_baseline_ratio >= args.min_query_baseline_ratio);
  const bool client_drain_passed = args.max_drain_seconds < 0.0 ||
    measure_client_drain_seconds <= args.max_drain_seconds;
  const bool stage2_checks_requested =
    args.max_stage2_p99_ms >= 0.0 ||
    args.max_stage2_backlog_slope >= 0.0 ||
    args.max_stage2_remaining >= 0 ||
    args.stage2_drain_timeout_seconds != 0;
  const bool stage2_load_observations_passed = !stage2_checks_requested ||
    (load_maintenance_summary.logs_with_observations ==
       load_maintenance_summary.requested_logs &&
     load_maintenance_summary.requested_logs != 0);
  const bool stage2_drain_observations_passed = !stage2_checks_requested ||
    (drain_maintenance_summary.logs_with_observations ==
       drain_maintenance_summary.requested_logs &&
     drain_maintenance_summary.requested_logs != 0);
  const bool stage2_observations_passed = !stage2_checks_requested ||
    (stage2_load_observations_passed &&
     stage2_drain_observations_passed);
  const bool stage2_p99_passed = args.max_stage2_p99_ms < 0.0 ||
    (stage2_observations_passed &&
     full_maintenance_summary.p99_stitch_delay_available &&
     !full_maintenance_summary.p99_stitch_delay_over_30s &&
     full_maintenance_summary.p99_stitch_delay_upper_ms <=
       args.max_stage2_p99_ms);
  const bool stage2_slope_passed = args.max_stage2_backlog_slope < 0.0 ||
    (stage2_load_observations_passed &&
     load_maintenance_summary.backlog_slope_available &&
     load_maintenance_summary.backlog_slope_per_sec <=
       args.max_stage2_backlog_slope);
  const bool stage2_remaining_passed = args.max_stage2_remaining < 0 ||
    (stage2_drain_observations_passed &&
     drain_maintenance_summary.remaining <=
       static_cast<uint64_t>(args.max_stage2_remaining));
  const bool stage2_failures_passed = !stage2_checks_requested ||
    (full_maintenance_summary.failure_delta_available &&
     full_maintenance_summary.failures == 0);
  const bool stage2_drain_passed = !stage2_checks_requested ||
    (stage2_drain_observations_passed && !stage2_drain_timed_out);
  const bool recall_drop_passed = args.max_recall_drop < 0.0 ||
    (recall_before_performance.has_value() &&
     recall_after_performance.has_value() &&
     *recall_before_performance - *recall_after_performance <=
       args.max_recall_drop);
  performance_below_threshold = !query_qps_passed || !insert_qps_passed ||
    !query_stability_passed || !write_stability_passed ||
    !zero_completion_windows_passed || !zero_query_windows_passed ||
    !zero_write_windows_passed || !writes_all_succeeded ||
    !mutation_capacity_passed || !gpu_visibility_passed ||
    !final_mutation_capacity_passed || !final_delta_mutable_passed ||
    !gpu_final_state_drain_passed || !late_storage_owner_rpcs_passed ||
    !rate_attainment_passed ||
    !query_baseline_passed || !client_drain_passed ||
    !stage2_observations_passed || !stage2_p99_passed ||
    !stage2_slope_passed || !stage2_remaining_passed ||
    !stage2_failures_passed || !stage2_drain_passed ||
    !recall_drop_passed;
  root["acceptance"] = {
    {"applies", query_acceptance_applies || write_acceptance_applies ||
                stage2_checks_requested || gpu_final_state_checks_requested},
    {"min_query_ops_per_sec", args.min_query_qps},
    {"observed_query_ops_per_sec", query_throughput},
    {"query_ops_per_sec_passed", query_qps_passed},
    {"min_insert_ops_per_sec", args.min_insert_qps},
    {"observed_insert_ops_per_sec", insert_throughput},
    {"insert_ops_per_sec_passed", insert_qps_passed},
    {"min_query_tail_to_head_ratio", args.min_stability_ratio},
    {"observed_query_tail_to_head_ratio", query_stability_ratio},
    {"query_stability_passed", query_stability_passed},
    {"min_write_tail_to_head_ratio", args.min_write_stability_ratio},
    {"observed_write_tail_to_head_ratio", write_stability_ratio},
    {"write_stability_passed", write_stability_passed},
    {"zero_completion_windows", zero_completion_windows},
    {"zero_query_windows", zero_query_windows},
    {"zero_write_windows", zero_write_windows},
    {"zero_completion_windows_passed", zero_completion_windows_passed},
    {"zero_query_windows_passed", zero_query_windows_passed},
    {"zero_write_windows_passed", zero_write_windows_passed},
    {"writes_all_succeeded", writes_all_succeeded},
    {"mutation_capacity_rejections", mutation_capacity_rejections},
    {"mutation_capacity_passed", mutation_capacity_passed},
    {"max_gpu_visibility_ms", args.max_gpu_visibility_ms},
    {"observed_max_gpu_visibility_ms", max_gpu_visibility_ms},
    {"expected_gpu_mutations", expected_gpu_mutations},
    {"observed_gpu_mutations_published",
     final_gpu_telemetry.mutations_published},
    {"gpu_visibility_coverage_passed", gpu_visibility_coverage_passed},
    {"gpu_visibility_available", gpu_visibility_available},
    {"gpu_visibility_passed", gpu_visibility_passed},
    {"max_final_mutation_capacity_reserved",
     args.max_final_mutation_capacity_reserved},
    {"observed_final_mutation_capacity_reserved",
     final_gpu_telemetry.mutation_capacity_reserved},
    {"final_mutation_capacity_passed",
     final_mutation_capacity_passed},
    {"max_final_delta_mutable_entries",
     args.max_final_delta_mutable_entries},
    {"observed_final_delta_mutable_entries",
     final_gpu_telemetry.delta_mutable_entries},
    {"final_delta_mutable_passed", final_delta_mutable_passed},
    {"gpu_final_state_drain_requested", gpu_final_state_drain_requested},
    {"gpu_final_state_drain_seconds", gpu_final_state_drain_seconds},
    {"gpu_final_state_drain_timed_out", gpu_final_state_drain_timed_out},
    {"gpu_final_state_drain_passed", gpu_final_state_drain_passed},
    {"max_late_storage_owner_rpcs", args.max_late_storage_owner_rpcs},
    {"observed_late_storage_owner_rpcs", late_storage_owner_rpcs},
    {"late_storage_owner_rpcs_passed", late_storage_owner_rpcs_passed},
    {"min_rate_attainment_ratio", args.min_rate_attainment_ratio},
    {"observed_query_rate_attainment_ratio", query_rate_attainment},
    {"observed_write_rate_attainment_ratio", write_rate_attainment},
    {"rate_attainment_passed", rate_attainment_passed},
    {"query_baseline_source", verified_query_baseline
      ? "verified_report" :
        (args.query_baseline_qps >= 0.0 ? "bare_qps" : "disabled")},
    {"query_baseline_report", verified_query_baseline
      ? verified_query_baseline->report_path : ""},
    {"query_baseline_fingerprint_verified", query_baseline_verified},
    {"query_baseline_effective_ops_per_sec",
     baseline_effective_query_qps},
    {"query_baseline_qps", baseline_effective_query_qps},
    {"observed_effective_query_ops_per_sec",
     effective_query_throughput},
    {"observed_query_baseline_ratio", query_baseline_ratio},
    {"query_baseline_comparison_basis", "effective_wall_clock"},
    {"min_query_baseline_ratio", args.min_query_baseline_ratio},
    {"query_baseline_passed", query_baseline_passed},
    {"max_client_drain_seconds", args.max_drain_seconds},
    {"observed_client_drain_seconds", measure_client_drain_seconds},
    {"client_drain_passed", client_drain_passed},
    {"stage2_load_observations_passed",
     stage2_load_observations_passed},
    {"stage2_post_stop_observations_passed",
     stage2_drain_observations_passed},
    {"stage2_observations_passed", stage2_observations_passed},
    {"stage2_p99_available",
     full_maintenance_summary.p99_stitch_delay_available},
    {"stage2_p99_passed", stage2_p99_passed},
    {"stage2_backlog_slope_passed", stage2_slope_passed},
    {"stage2_remaining_passed", stage2_remaining_passed},
    {"stage2_failure_delta_available",
     full_maintenance_summary.failure_delta_available},
    {"stage2_failures_passed", stage2_failures_passed},
    {"stage2_drain_passed", stage2_drain_passed},
    {"max_recall_drop", args.max_recall_drop},
    {"observed_recall_drop",
     recall_before_performance.has_value() && recall_after_performance.has_value()
       ? *recall_before_performance - *recall_after_performance : 0.0},
    {"recall_drop_passed", recall_drop_passed},
    {"passed", !performance_below_threshold},
  };

  FormattedReport formatted_report = format_report(root, report);
  root["bottleneck_summary"] = std::move(formatted_report.bottleneck_summary);
  std::ofstream json_output(args.report_json_path);
  json_output << root.dump(2) << '\n';
  if (!json_output) {
    throw std::runtime_error("failed to write report json");
  }

  if (!args.report_text_path.empty()) {
    std::ofstream text_output(args.report_text_path);
    text_output << formatted_report.text;
  }

  std::cout << formatted_report.text;
  if (recall_below_threshold) {
    throw std::runtime_error("recall below threshold");
  }
  if (performance_below_threshold) {
    throw std::runtime_error("throughput or stability below threshold");
  }
  return root;
}




}  // namespace tools::breakdown_benchmark
