#include "tools/breakdown_benchmark/workload.hh"

#include <algorithm>
#include <array>
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
#include "gpu_search/index_format.hh"
#include "gpu_search/persistent_kernel.hh"
#include "service/breakdown.hh"
#include "service/index_metadata.hh"
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

struct FixedWorkPhaseStats {
  size_t completed{};
  double duration_seconds{};
};

nlohmann::json run_benchmark(ComputeService& service, const Args& args) {
  using SampleReport = service::breakdown::Report;
  using service::breakdown::report_to_json;

  const bool use_insert_file = !args.insert_file.empty();
  const bool shared_rate_limited = args.mixed_mode == "rate_limited";
  const bool write_rate_limited =
    args.mixed_mode == "write_rate_limited";
  const bool workload_has_queries =
    !args.recall_only &&
    (args.workload == "query" || args.workload == "both" ||
     (args.workload == "mixed" &&
      (shared_rate_limited ? args.target_query_qps > 0.0 :
       write_rate_limited ? true : args.read_ratio > 0.0)));
  const bool mixed_has_writes = args.workload == "mixed" &&
    ((shared_rate_limited || write_rate_limited)
       ? args.target_write_qps > 0.0 : args.read_ratio < 1.0);
  if (service.config().synchronous_exact_updates_enabled() &&
      mixed_has_writes &&
      (args.write_upsert_ratio > 0.0 || args.write_delete_ratio > 0.0)) {
    throw std::invalid_argument(
      "coupled update mode is append-only: mixed workloads must set "
      "--write-upsert-ratio=0 and --write-delete-ratio=0");
  }

  const auto index_prefix = service.config().resolved_index_prefix();
  service::index_metadata::Metadata index_metadata;
  str index_metadata_error;
  if (!service::index_metadata::load_metadata(
        index_prefix, index_metadata, &index_metadata_error)) {
    throw std::runtime_error(
      "failed to reload validated index metadata for benchmark report: " +
      index_metadata_error);
  }

  nlohmann::json root;
  root["meta"] = {
    {"system_variant", {
      {"profile_name", args.profile_name},
      {"label", args.system_variant_label},
      {"update_mutation_api",
        service::storage_owner::mutation_api_name_for_completion_mode(
          service.config().synchronous_exact_updates_enabled())},
      {"resolved_modes", {
        {"storage_owner_update_completion_mode",
          service.config().storage_owner_update_completion_mode},
        {"gpu_dynamic_graph_access_mode",
          service.config().dynamic_graph_access_mode},
        {"gpu_rdma_search_progression_mode",
          service.config().gpu_rdma_search_progression_mode},
      }},
      {"index", {
        {"prefix", normalize_path(index_prefix.string())},
        {"schema_version", index_metadata.schema_version},
        {"build_fingerprint", index_metadata.index_build_fingerprint},
      }},
    }},
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
       : (shared_rate_limited
            ? "shared_two_stream_pacer_until_deadline"
          : write_rate_limited
            ? "closed_loop_query_threads_plus_paced_write_threads"
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
    {"write_threads", args.write_threads},
    {"read_ratio", args.read_ratio},
    {"target_query_qps", args.target_query_qps},
    {"target_write_qps", args.target_write_qps},
    {"write_insert_ratio", args.write_insert_ratio},
    {"write_upsert_ratio", args.write_upsert_ratio},
    {"write_delete_ratio", args.write_delete_ratio},
    {"recall_mode", args.recall_mode},
    {"recall_base_id_limit", args.recall_base_id_limit},
    {"insert_start_id", args.insert_start_id},
    {"index_prefix", normalize_path(index_prefix.string())},
    {"dim", service.config().dim},
    {"threads", service.config().num_threads},
    {"fine_grained_breakdown_enabled", service.config().enable_breakdown},
    {"search", "gpu_persistent_opq_pq"},
    {"navigation_quantizer", "opq_pq"},
    {"traversal_beam_width", service.config().gpu_traversal_beam_width},
    {"final_rerank_width", service.config().gpu_final_rerank_width},
    {"max_expansions", service.config().gpu_max_expansions},
    {"entry_seed_policy", "nearest_centroid_shard_live_entries"},
    {"entry_seed_shards", 1},
    {"entry_seed_capacity", gpu_search::kCentroidRouteMaxLiveEntries},
    {"gpu_query_slots", service.config().gpu_query_slots},
    {"gpu_rdma_qps", service.config().gpu_rdma_qps},
    {"gpu_graph_prefetch_depth", service.config().gpu_graph_prefetch_depth},
    {"gpu_graph_commit_width", service.config().gpu_graph_commit_width},
    {"gpu_graph_issue_width", service.config().gpu_graph_issue_width},
    {"gpu_exact_frontier_early_issue",
      service.config().gpu_exact_frontier_early_issue},
    {"gpu_frontier_execution_mode",
      service.config().decoupled_gpu_rdma_search_progression_enabled()
        ? (service.config().gpu_graph_issue_width >
              service.config().gpu_graph_commit_width
            ? "adaptive_decoupled" : "exact_core_early_issue")
        : "persistent_late_issue"},
    {"gpu_query_graph_read_policy",
      service.config().gpu_query_graph_read_policy},
    {"gpu_dynamic_graph_extent",
      service.config().gpu_query_graph_read_policy == "live-extent" &&
        service.config().gpu_dynamic_graph_extent},
    {"gpu_dynamic_graph_extent_source",
      service.config().gpu_query_graph_read_policy == "live-extent" &&
          service.config().gpu_dynamic_graph_extent
        ? "incarnation_tagged_live_extent"
        : "full_physical_record"},
    {"gpu_graph_physical_record_bytes", VamanaNode::hot_graph_entry_size()},
    {"gpu_graph_entry_capacity", VamanaNode::graph_entry_capacity()},
    {"gpu_graph_extent_quantum_edges",
      gpu_search::format::kGraphExtentQuantum},
    {"gpu_graph_extent_sidecar_format", "global_ordinal_u8_gextent8_v1"},
    {"gpu_query_beam_merge_policy",
      service.config().gpu_query_beam_merge_policy},
    {"storage_owner_stage2_score_many",
      service.config().storage_owner_stage2_score_many},
    {"storage_owner_stage2_graph_issue_width",
      service.config().storage_owner_stage2_graph_issue_width},
    {"storage_owner_stage2_home_rpc_combining",
      service.config().storage_owner_stage2_home_rpc_combining},
    {"storage_owner_maintenance_queue_depth",
      service.config().storage_owner_maintenance_queue_depth},
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
  if (args.workload == "mixed" &&
      (args.mixed_mode == "fixed_threads" || write_rate_limited)) {
    if (write_rate_limited) {
      fixed_write_threads = args.write_threads;
      fixed_read_threads = args.client_threads - fixed_write_threads;
    } else {
      if (args.read_ratio <= 0.0) {
        fixed_read_threads = 0;
      } else if (args.read_ratio >= 1.0) {
        fixed_read_threads = args.client_threads;
      } else {
        fixed_read_threads = static_cast<size_t>(std::llround(
          static_cast<double>(args.client_threads) * args.read_ratio));
        fixed_read_threads = std::clamp<size_t>(
          fixed_read_threads, 1, args.client_threads - 1);
      }
    }
    if (!write_rate_limited) {
      fixed_write_threads = args.client_threads - fixed_read_threads;
    }
    root["meta"]["mixed_fixed_threads"] = {
      {"read_threads", fixed_read_threads},
      {"write_threads", fixed_write_threads},
    };
    std::cerr << "[breakdown] mixed "
              << (write_rate_limited ? "write-rate-limited" : "fixed")
              << " thread split: reads=" << fixed_read_threads
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
  // `both` executes insert and query as separate sequential phases. Preserve
  // the first phase instead of letting the later query reporter overwrite it,
  // so stability never interprets a query-only window as a write stall.
  std::vector<ProgressSample> measure_insert_windows;

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
    if (label.starts_with("measure-")) {
      measure_windows = reporter.samples();
      measure_insert_windows = measure_windows;
    }
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
    if (label.starts_with("measure-")) {
      measure_windows = reporter.samples();
      measure_insert_windows = measure_windows;
    }
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
      ? normalize_path(args.performance_query_file) : ""},
    {"rows", performance_query_rows.count},
    {"data_type", performance_query_rows.count == 0 ? "" : vector_dtype_name(performance_query_rows.dtype)},
    {"vector_bytes", performance_query_rows.vector_bytes},
    {"row_reuse_policy", "single_pass_no_reuse"},
    {"row_reuse_count", 0},
    {"rate_limited_required_rows", rate_limited_required_query_rows},
  };

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

  auto run_recall_check = [&](const char* phase,
                              const char* key,
                              bool reset_after) {
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
    };
    std::cerr << "[breakdown][recall] " << phase << " recall@" << recall_k << "=" << recall
              << " queries=" << recall_queries << std::endl;
    if (reset_after) {
      service.clear_thread_statistics();
      service.reset_breakdown_state();
    }
  };
  run_recall_check("before_performance", "recall", true);

  auto run_query_phase_ops = [&](const std::string& label,
                                 size_t ops) -> FixedWorkPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<size_t> next_op{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    ProgressReporter reporter(label, completed_ops, ops, 0, &completed_ops, nullptr);
    const size_t worker_count = std::max<size_t>(1, std::min(args.client_threads, ops));
    std::barrier start_barrier(
      static_cast<std::ptrdiff_t>(worker_count + 1));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        start_barrier.arrive_and_wait();
        try {
          while (!failed.load(std::memory_order_acquire)) {
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
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error == nullptr) error = std::current_exception();
          }
          failed.store(true, std::memory_order_release);
        }
      });
    }
    const auto started = std::chrono::steady_clock::now();
    start_barrier.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    const double duration_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    reporter.finish();
    if (error != nullptr) std::rethrow_exception(error);
    if (label.starts_with("measure-")) measure_windows = reporter.samples();
    throw_if_performance_queries_exhausted(label);
    return {
      .completed = completed_ops.load(std::memory_order_relaxed),
      .duration_seconds = duration_seconds,
    };
  };

  auto run_query_phase_seconds = [&](const std::string& label,
                                     size_t seconds) -> QueryPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    std::chrono::steady_clock::time_point deadline;
    std::barrier start_barrier(
      static_cast<std::ptrdiff_t>(args.client_threads + 1));
    std::vector<std::thread> workers;
    workers.reserve(args.client_threads);
    for (size_t worker = 0; worker < args.client_threads; ++worker) {
      workers.emplace_back([&]() {
        start_barrier.arrive_and_wait();
        try {
          while (!failed.load(std::memory_order_acquire) &&
                 !performance_query_stream.exhausted() &&
                 std::chrono::steady_clock::now() < deadline) {
            const auto query_row = performance_query_stream.try_claim();
            if (!query_row.has_value()) break;
            (void)service.search_raw(
              performance_query_rows.dtype,
              performance_query_rows.raw_row(*query_row), dim, service.config().k);
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
    if (error != nullptr) std::rethrow_exception(error);
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
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
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
        try {
          while (!failed.load(std::memory_order_acquire)) {
            const size_t op_index = next_op.fetch_add(1, std::memory_order_relaxed);
            if (op_index >= ops) break;

            const bool read_op = args.mixed_mode == "fixed_threads"
              ? tid < fixed_read_threads : choose_mixed_read(rng);
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
              if (!issue_mixed_write(
                    rng, next_insert_id, next_update_version,
                    issued_inserts, issued_upserts, issued_deletes,
                    completed_inserts, completed_upserts, completed_deletes)) {
                throw std::runtime_error(
                  "mixed mutation was rejected or timed out");
              }
              completed_writes.fetch_add(1, std::memory_order_relaxed);
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

    for (auto& thread : threads) {
      thread.join();
    }
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
    std::optional<PacedOperationDispatcher> write_pacer;
    if (shared_rate_limited) {
      pacer.emplace(args.target_query_qps, args.target_write_qps);
    } else if (write_rate_limited) {
      write_pacer.emplace(0.0, args.target_write_qps);
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
            } else if (write_rate_limited) {
              read_op = tid < fixed_read_threads;
              if (read_op) {
                if (std::chrono::steady_clock::now() >= deadline) break;
              } else {
                const auto claim = write_pacer->claim();
                if (!claim.has_value()) break;
                lib_assert(
                  claim->kind == PacedOperationKind::write,
                  "write-only pacer returned a query claim");
              }
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
              if (!succeeded) {
                throw std::runtime_error(
                  "mixed mutation was rejected or timed out");
              }
              completed_writes.fetch_add(1, std::memory_order_relaxed);
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
    if (write_pacer.has_value()) write_pacer->start(phase_start, deadline);
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
      .scheduled_reads = shared_rate_limited
        ? PacedOperationDispatcher::scheduled_count(args.target_query_qps, seconds)
        : 0,
      .scheduled_writes = (shared_rate_limited || write_rate_limited)
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
  const bool workload_has_writes = !args.recall_only &&
    (args.workload == "insert" || args.workload == "both" ||
     (args.workload == "mixed" &&
      ((shared_rate_limited || write_rate_limited)
         ? args.target_write_qps > 0.0 : args.read_ratio < 1.0)));
  MixedPhaseStats warmup_mixed_stats{};
  MixedPhaseStats measure_mixed_stats{};
  double measure_client_drain_seconds = 0.0;
  double measure_query_client_drain_seconds = 0.0;
  double measure_write_client_drain_seconds = 0.0;
  double measure_maintenance_drain_seconds = 0.0;
  vec<u64> maintenance_target_sequences;
  vec<u64> maintenance_durable_sequences;
  const auto drain_storage_maintenance = [&](const char* phase) {
    if (!workload_has_writes) return 0.0;
    const auto started = std::chrono::steady_clock::now();
    vec<u64> targets;
    vec<u64> durable;
    const bool complete = service.wait_for_storage_maintenance(
      std::chrono::milliseconds(
        std::max<u32>(1000, service.config().storage_owner_rpc_timeout_ms)),
      &targets, &durable);
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    std::cerr << "[breakdown][" << phase
              << "] maintenance drain elapsed=" << elapsed
              << "s complete=" << (complete ? "yes" : "no")
              << std::endl;
    if (!complete) {
      throw std::runtime_error(
        std::string("storage Stage2 maintenance did not reach its durable ") +
        "watermark during " + phase);
    }
    maintenance_target_sequences = std::move(targets);
    maintenance_durable_sequences = std::move(durable);
    return elapsed;
  };
  std::vector<MaintenanceLogCursor> maintenance_log_cursors;
  std::vector<std::optional<gpu_search::maintenance_telemetry::Snapshot>>
    maintenance_snapshot_begin;
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
  (void)drain_storage_maintenance("warmup");

  const size_t performance_queries_after_warmup = performance_query_stream.consumed();
  if (workload_has_writes) {
    try {
      maintenance_snapshot_begin = service.storage_maintenance_telemetry();
    } catch (const std::exception& error) {
      std::cerr << "[breakdown] in-band Stage2 telemetry baseline unavailable: "
                << error.what() << std::endl;
    }
  }
  if (!args.storage_maintenance_logs.empty()) {
    maintenance_log_cursors = snapshot_maintenance_logs(
      args.storage_maintenance_logs);
  }
  service.clear_thread_statistics();
  service.reset_breakdown_state();
  std::cerr << "[breakdown] starting measure phase" << std::endl;
  size_t measured_query_operations = 0;
  size_t measured_insert_operations = 0;
  double measure_query_wall_seconds = 0.0;
  double measure_write_wall_seconds = 0.0;
  double measure_total_wall_seconds = 0.0;

  if (!args.recall_only && (args.workload == "insert" || args.workload == "both")) {
    if (use_time_mode) {
      const auto measured = run_insert_phase_seconds(
        "measure-insert", args.measure_seconds, next_insert_id);
      next_insert_id = measured.next_insert_id;
      measured_insert_operations = measured.completed;
      measure_write_client_drain_seconds = measured.drain_seconds;
      measure_client_drain_seconds += measured.drain_seconds;
    } else {
      const auto started = std::chrono::steady_clock::now();
      measured_insert_operations = run_insert_phase_ops(
        "measure-insert", args.measure_ops, next_insert_id);
      measure_write_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
      measure_total_wall_seconds += measure_write_wall_seconds;
    }
  }
  if (!args.recall_only && (args.workload == "query" || args.workload == "both")) {
    if (use_time_mode) {
      const auto measured = run_query_phase_seconds(
        "measure-query", args.measure_seconds);
      measured_query_operations = measured.completed;
      measure_query_client_drain_seconds = measured.drain_seconds;
      measure_client_drain_seconds += measured.drain_seconds;
    } else {
      const FixedWorkPhaseStats measured =
        run_query_phase_ops("measure-query", args.measure_ops);
      measured_query_operations = measured.completed;
      measure_query_wall_seconds = measured.duration_seconds;
      measure_total_wall_seconds += measure_query_wall_seconds;
    }
  }
  if (!args.recall_only && args.workload == "mixed") {
    if (use_time_mode) {
      measure_mixed_stats = run_mixed_phase_seconds("measure-mixed", args.measure_seconds, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
      measure_client_drain_seconds = measure_mixed_stats.drain_seconds;
      measure_query_client_drain_seconds = measure_client_drain_seconds;
      measure_write_client_drain_seconds = measure_client_drain_seconds;
    } else {
      const auto started = std::chrono::steady_clock::now();
      measure_mixed_stats = run_mixed_phase_ops("measure-mixed", args.measure_ops, next_insert_id);
      measure_total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
      measure_query_wall_seconds = measure_total_wall_seconds;
      measure_write_wall_seconds = measure_total_wall_seconds;
      next_insert_id = measure_mixed_stats.next_insert_id;
    }
  }
  measure_maintenance_drain_seconds =
    drain_storage_maintenance("measure");

  MaintenanceLogSummary maintenance_summary;
  MaintenanceLogSummary maintenance_storage_log_summary;
  bool in_band_maintenance_telemetry = false;
  if (!maintenance_snapshot_begin.empty()) {
    try {
      auto maintenance_snapshot_end =
        service.storage_maintenance_telemetry();
      // The durable watermark can advance just before the storage node's
      // periodic control-page telemetry publication. For short fixed-work
      // runs this used to produce a fully drained run with zero Stage2 and
      // locality deltas. Poll only the diagnostic snapshot (not the measured
      // workload/drain interval) until it accounts for the measured writes or
      // one publication period has elapsed.
      const uint64_t expected_stage2_completions =
        static_cast<uint64_t>(measured_insert_operations) +
        static_cast<uint64_t>(measure_mixed_stats.completed_writes);
      const auto observed_stage2_completions = [&]() {
        uint64_t completed = 0;
        const size_t shards = std::min(
          maintenance_snapshot_begin.size(),
          maintenance_snapshot_end.size());
        for (size_t shard = 0; shard < shards; ++shard) {
          if (!maintenance_snapshot_begin[shard].has_value() ||
              !maintenance_snapshot_end[shard].has_value()) {
            continue;
          }
          const auto& first = *maintenance_snapshot_begin[shard];
          const auto& latest = *maintenance_snapshot_end[shard];
          const uint64_t first_done =
            first.stage2_finalized_live + first.stale;
          const uint64_t latest_done =
            latest.stage2_finalized_live + latest.stale;
          if (latest_done >= first_done) completed += latest_done - first_done;
        }
        return completed;
      };
      if (!service.config().synchronous_exact_updates_enabled() &&
          expected_stage2_completions != 0) {
        const auto telemetry_deadline = std::chrono::steady_clock::now() +
          std::chrono::seconds(7);
        while (observed_stage2_completions() <
                 expected_stage2_completions &&
               std::chrono::steady_clock::now() < telemetry_deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          maintenance_snapshot_end =
            service.storage_maintenance_telemetry();
        }
        std::cerr << "[breakdown] Stage2 telemetry publication observed="
                  << observed_stage2_completions()
                  << " expected=" << expected_stage2_completions
                  << std::endl;
        if (observed_stage2_completions() < expected_stage2_completions) {
          throw std::runtime_error(
            "Stage2 control-page telemetry did not publish all measured "
            "completions after maintenance drain");
        }
      }
      MaintenanceLogSummary in_band_summary =
        summarize_maintenance_snapshot_window(
          maintenance_snapshot_begin, maintenance_snapshot_end);
      if (in_band_summary.requested_logs != 0 &&
          in_band_summary.logs_with_observations ==
            in_band_summary.requested_logs) {
        maintenance_summary = std::move(in_band_summary);
        in_band_maintenance_telemetry = true;
      }
    } catch (const std::exception& error) {
      std::cerr << "[breakdown] in-band Stage2 telemetry final snapshot "
                   "unavailable: " << error.what() << std::endl;
    }
  }
  if (!maintenance_log_cursors.empty()) {
    const auto measurement_end = snapshot_maintenance_logs(
      args.storage_maintenance_logs);
    maintenance_storage_log_summary = summarize_maintenance_log_window(
      maintenance_log_cursors, measurement_end);
    if (!in_band_maintenance_telemetry) {
      maintenance_summary = maintenance_storage_log_summary;
    }
  }

  const gpu_search::TelemetrySnapshot final_gpu_telemetry =
    service.gpu_search_telemetry();

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
    const bool reads_expected = shared_rate_limited
      ? args.target_query_qps > 0.0
      : write_rate_limited ? true : args.read_ratio > 0.0;
    const bool writes_expected = shared_rate_limited || write_rate_limited
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
  // Keep one post-publication snapshot for the raw report. Later recall
  // queries may change query counters but not mutation-state telemetry.
  root["gpu_persistent"] = telemetry_to_json(final_gpu_telemetry);
  const u64 late_storage_owner_rpcs =
    service.late_storage_owner_rpc_completions();
  const auto storage_owner_sender =
    service.storage_owner_sender_telemetry();
  root["storage_owner_runtime"] = {
    {"late_rpc_completions", late_storage_owner_rpcs},
    {"late_rpc_threshold_ms", service.config().storage_owner_rpc_timeout_ms},
    {"maintenance_drain_seconds", measure_maintenance_drain_seconds},
    {"maintenance_target_sequences", maintenance_target_sequences},
    {"maintenance_durable_sequences", maintenance_durable_sequences},
    {"submitted_batches", storage_owner_sender.submitted_batches},
    {"submitted_items", storage_owner_sender.submitted_items},
    {"completed_batches", storage_owner_sender.completed_batches},
    {"completed_items", storage_owner_sender.completed_items},
    {"completed_rpc_wall_ns",
      storage_owner_sender.completed_rpc_wall_ns},
    {"max_rpc_wall_ns", storage_owner_sender.max_rpc_wall_ns},
    {"average_submitted_batch_size",
      storage_owner_sender.submitted_batches == 0 ? 0.0
      : static_cast<double>(storage_owner_sender.submitted_items) /
          static_cast<double>(storage_owner_sender.submitted_batches)},
    {"average_completed_rpc_wall_us",
      storage_owner_sender.completed_batches == 0 ? 0.0
      : static_cast<double>(storage_owner_sender.completed_rpc_wall_ns) /
          static_cast<double>(storage_owner_sender.completed_batches) /
          1000.0},
  };


  const bool has_throughput_duration =
    (use_time_mode && args.measure_seconds > 0) ||
    (!use_time_mode && measure_total_wall_seconds > 0.0);
  const double configured_phase_duration =
    static_cast<double>(args.measure_seconds);
  const double configured_total_measure_duration = has_throughput_duration
    ? configured_phase_duration * (args.workload == "both" ? 2.0 : 1.0)
    : 0.0;
  // `both` runs insert and query as two sequential full-duration phases.  Its
  // aggregate denominator must therefore be their combined makespan, while
  // each per-operation rate uses only the phase in which that operation was
  // offered. Mixed/query/insert retain their single shared window.
  const double throughput_duration = use_time_mode
    ? configured_total_measure_duration + measure_client_drain_seconds
    : measure_total_wall_seconds;
  const double query_throughput_duration = use_time_mode
    ? configured_phase_duration +
        (args.workload == "both" ? measure_query_client_drain_seconds
                                  : measure_client_drain_seconds)
    : measure_query_wall_seconds;
  const double write_throughput_duration = use_time_mode
    ? configured_phase_duration +
        (args.workload == "both" ? measure_write_client_drain_seconds
                                  : measure_client_drain_seconds)
    : measure_write_wall_seconds;
  const double durable_throughput_duration = has_throughput_duration
    ? throughput_duration + measure_maintenance_drain_seconds : 0.0;
  const size_t throughput_query_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_reads
    : (service.config().enable_breakdown ? report.query.count : measured_query_operations);
  const size_t throughput_write_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_writes
    : (service.config().enable_breakdown ? report.insert.count : measured_insert_operations);
  const size_t throughput_insert_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_inserts
    : (service.config().enable_breakdown ? report.insert.count : measured_insert_operations);
  const double effective_query_throughput = query_throughput_duration > 0.0
    ? static_cast<double>(throughput_query_ops) / query_throughput_duration
    : 0.0;
  const double effective_write_throughput = write_throughput_duration > 0.0
    ? static_cast<double>(throughput_write_ops) / write_throughput_duration
    : 0.0;
  const double effective_insert_throughput = write_throughput_duration > 0.0
    ? static_cast<double>(throughput_insert_ops) / write_throughput_duration
    : 0.0;
  const bool rate_limited_measurement = args.workload == "mixed" &&
    (shared_rate_limited || write_rate_limited) && has_throughput_duration;
  const double configured_measure_duration = configured_phase_duration;
  const double query_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_query_ops) / configured_measure_duration
    : effective_query_throughput;
  const double write_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_write_ops) / configured_measure_duration
    : effective_write_throughput;
  const double insert_throughput = rate_limited_measurement
    ? static_cast<double>(throughput_insert_ops) / configured_measure_duration
    : effective_insert_throughput;
  const double total_throughput = has_throughput_duration
    ? static_cast<double>(throughput_query_ops + throughput_write_ops) /
        throughput_duration
    : 0.0;
  const double durable_write_throughput = durable_throughput_duration > 0.0
    ? static_cast<double>(throughput_write_ops) / durable_throughput_duration
    : 0.0;
  const double durable_total_throughput = durable_throughput_duration > 0.0
    ? static_cast<double>(throughput_query_ops + throughput_write_ops) /
        durable_throughput_duration
    : 0.0;
  const double mixed_write_subtype_duration = rate_limited_measurement
    ? configured_measure_duration : throughput_duration;
  root["throughput"] = {
    {"measurement_mode", use_time_mode ? "time" : "fixed_work"},
    {"configured_measure_seconds", args.measure_seconds},
    {"configured_measure_ops", args.measure_ops},
    {"configured_total_measure_seconds", configured_total_measure_duration},
    {"effective_measure_seconds", throughput_duration},
    {"duration_seconds", throughput_duration},
    {"client_drain_seconds", measure_client_drain_seconds},
    {"query_client_drain_seconds", measure_query_client_drain_seconds},
    {"write_client_drain_seconds", measure_write_client_drain_seconds},
    {"query_duration_seconds", query_throughput_duration},
    {"write_duration_seconds", write_throughput_duration},
    {"maintenance_drain_seconds", measure_maintenance_drain_seconds},
    {"durable_effective_measure_seconds", durable_throughput_duration},
    {"total_ops", throughput_query_ops + throughput_write_ops},
    {"total_ops_per_sec", total_throughput},
    {"query_ops", throughput_query_ops},
    {"query_ops_per_sec", query_throughput},
    {"nominal_query_ops_per_sec", query_throughput},
    {"effective_query_ops_per_sec", effective_query_throughput},
    {"write_ops", throughput_write_ops},
    {"write_ops_per_sec", write_throughput},
    {"durable_write_ops_per_sec", durable_write_throughput},
    {"nominal_write_ops_per_sec", write_throughput},
    {"effective_write_ops_per_sec", effective_write_throughput},
    {"insert_ops", throughput_insert_ops},
    {"insert_ops_per_sec", insert_throughput},
    {"nominal_insert_ops_per_sec", insert_throughput},
    {"effective_insert_ops_per_sec", effective_insert_throughput},
    {"durable_total_ops_per_sec", durable_total_throughput},
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
    {"durable_rate_basis",
     "wall_clock_including_client_and_stage2_watermark_drain"},
    {"upsert_ops", args.workload == "mixed" ? measure_mixed_stats.completed_upserts : 0},
    {"upsert_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_upserts) /
          mixed_write_subtype_duration
      : 0.0},
    {"delete_ops", args.workload == "mixed" ? measure_mixed_stats.completed_deletes : 0},
    {"delete_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_deletes) /
          mixed_write_subtype_duration
      : 0.0},
  };

  nlohmann::json window_json = nlohmann::json::array();
  std::vector<double> total_window_rates;
  std::vector<double> both_insert_total_window_rates;
  std::vector<double> both_query_total_window_rates;
  std::vector<double> query_window_rates;
  std::vector<double> write_window_rates;
  size_t zero_completion_windows = 0;
  size_t zero_query_windows = 0;
  size_t zero_write_windows = 0;
  const bool query_windows_expected = args.workload == "query" ||
    args.workload == "both" ||
    (args.workload == "mixed" &&
     (shared_rate_limited ? args.target_query_qps > 0.0
      : write_rate_limited ? true : args.read_ratio > 0.0));
  const bool write_windows_expected = args.workload == "insert" ||
    args.workload == "both" ||
    (args.workload == "mixed" &&
     ((shared_rate_limited || write_rate_limited)
       ? args.target_write_qps > 0.0 : args.read_ratio < 1.0));
  // The reporter is stopped only after synchronous calls already issued at
  // the deadline return. Keep those completions in throughput/drain counters,
  // but do not let a long final drain interval masquerade as the load tail or
  // create a false zero-completion window.
  const double stability_window_deadline =
    static_cast<double>(args.measure_seconds) + 0.5;
  struct StabilityPhase {
    const char* name;
    const std::vector<ProgressSample>* samples;
    bool expects_queries;
    bool expects_writes;
  };
  std::vector<StabilityPhase> stability_phases;
  if (args.workload == "both") {
    stability_phases.push_back(
      {"insert", &measure_insert_windows, false, true});
    stability_phases.push_back(
      {"query", &measure_windows, true, false});
  } else {
    stability_phases.push_back({
      args.workload.c_str(), &measure_windows,
      query_windows_expected, write_windows_expected});
  }
  for (const StabilityPhase& phase : stability_phases) {
    for (const ProgressSample& sample : *phase.samples) {
      const bool within_measurement_window = !has_throughput_duration ||
        sample.elapsed_seconds <= stability_window_deadline;
      window_json.push_back({
        {"phase", phase.name},
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
      if (!within_measurement_window || sample.interval_seconds < 2.5) {
        continue;
      }
      if (args.workload == "both") {
        (phase.expects_queries ? both_query_total_window_rates
                               : both_insert_total_window_rates)
          .push_back(sample.total_ops_per_sec);
      } else {
        total_window_rates.push_back(sample.total_ops_per_sec);
      }
      if (phase.expects_queries) {
        query_window_rates.push_back(sample.query_ops_per_sec);
      }
      if (phase.expects_writes) {
        write_window_rates.push_back(sample.write_ops_per_sec);
      }
      if (sample.interval_ops == 0) ++zero_completion_windows;
      if (phase.expects_queries && sample.interval_reads == 0) {
        ++zero_query_windows;
      }
      if (phase.expects_writes && sample.interval_writes == 0) {
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
  const auto minimum_rate = [](const std::vector<double>& values) {
    return values.empty() ? 0.0
      : *std::min_element(values.begin(), values.end());
  };
  const double total_head_qps = args.workload == "both"
    ? (edge_mean(both_insert_total_window_rates, false) +
       edge_mean(both_query_total_window_rates, false)) / 2.0
    : edge_mean(total_window_rates, false);
  const double total_tail_qps = args.workload == "both"
    ? (edge_mean(both_insert_total_window_rates, true) +
       edge_mean(both_query_total_window_rates, true)) / 2.0
    : edge_mean(total_window_rates, true);
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
    {"query_min_window_ops_per_sec", minimum_rate(query_window_rates)},
    {"query_tail_to_head_ratio", query_head_qps == 0.0 ? 0.0
      : query_tail_qps / query_head_qps},
    {"write_head_ops_per_sec", write_head_qps},
    {"write_tail_ops_per_sec", write_tail_qps},
    {"write_min_window_ops_per_sec", minimum_rate(write_window_rates)},
    {"write_tail_to_head_ratio", write_head_qps == 0.0 ? 0.0
      : write_tail_qps / write_head_qps},
    {"single_pass_no_reuse", true},
    {"direct_remote_read_baseline", true},
  };
  if (!args.recall_only) {
    run_recall_check("after_performance", "static_gt_post_recall", false);
  }

  auto unreadable_logs_json = [](const MaintenanceLogSummary& summary) {
    nlohmann::json paths = nlohmann::json::array();
    for (const auto& path : summary.unreadable_logs) paths.push_back(path);
    return paths;
  };
  constexpr std::array<const char*, kStage2TimingPhaseCount>
    stage2_phase_names{
      "search", "freeze_prune", "reverse_prepare", "placement_authority",
      "completion_handoff", "finalize",
    };
  nlohmann::json stage2_phase_timing = nlohmann::json::object();
  for (size_t phase = 0; phase < stage2_phase_names.size(); ++phase) {
    const uint64_t tasks =
      maintenance_summary.stage2_phase_task_attempts[phase];
    stage2_phase_timing[stage2_phase_names[phase]] = {
      {"attempts", maintenance_summary.stage2_phase_attempts[phase]},
      {"task_attempts", tasks},
      {"elapsed_ns", maintenance_summary.stage2_phase_elapsed_ns[phase]},
      {"avg_us_per_task", tasks == 0 ? 0.0 :
        static_cast<double>(
          maintenance_summary.stage2_phase_elapsed_ns[phase]) /
        static_cast<double>(tasks) / 1e3},
    };
  }
  const auto stage1_average_us = [&](uint64_t elapsed_ns) {
    return maintenance_summary.physical_stage1_items == 0 ? 0.0 :
      static_cast<double>(elapsed_ns) /
      static_cast<double>(maintenance_summary.physical_stage1_items) / 1e3;
  };
  const uint64_t graph_home_rpc_batches =
    maintenance_summary.stage2_home_rpc_batches >=
        maintenance_summary.stage2_home_score_rpc_batches
      ? maintenance_summary.stage2_home_rpc_batches -
          maintenance_summary.stage2_home_score_rpc_batches
      : 0;
  const uint64_t graph_home_rpc_items =
    maintenance_summary.stage2_home_rpc_items >=
        maintenance_summary.stage2_home_score_rpc_items
      ? maintenance_summary.stage2_home_rpc_items -
          maintenance_summary.stage2_home_score_rpc_items
      : 0;
  const auto stage1_average_count = [&](uint64_t count) {
    return maintenance_summary.physical_stage1_items == 0 ? 0.0 :
      static_cast<double>(count) /
      static_cast<double>(maintenance_summary.physical_stage1_items);
  };
  const uint64_t exact_remote_dependency_ns =
    maintenance_summary.exact_insert_remote_read_ns +
    maintenance_summary.exact_insert_remote_reverse_ns;
  const uint64_t exact_rdma_wait_ns =
    maintenance_summary.exact_insert_rdma_wait_ns;
  const uint64_t exact_cpu_and_local_ns =
    maintenance_summary.exact_insert_total_ns >= exact_rdma_wait_ns
      ? maintenance_summary.exact_insert_total_ns - exact_rdma_wait_ns
      : 0;
  const uint64_t exact_local_ns =
    maintenance_summary.exact_insert_total_ns >= exact_remote_dependency_ns
      ? maintenance_summary.exact_insert_total_ns - exact_remote_dependency_ns
      : 0;
  const uint64_t exact_deferred_stage2_ns =
    maintenance_summary.exact_insert_stage2_global_continuation_ns +
    maintenance_summary.exact_insert_final_candidate_snapshot_ns +
    maintenance_summary.exact_insert_remote_reverse_ns;
  const uint64_t exact_known_stack_ns =
    maintenance_summary.exact_insert_stage1_local_search_ns +
    exact_deferred_stage2_ns + maintenance_summary.exact_insert_prune_ns +
    maintenance_summary.exact_insert_allocate_write_ns +
    maintenance_summary.exact_insert_local_reverse_ns;
  const uint64_t exact_metadata_other_ns =
    maintenance_summary.exact_insert_total_ns >= exact_known_stack_ns
      ? maintenance_summary.exact_insert_total_ns - exact_known_stack_ns
      : 0;
  root["coupled_insert_critical_path"] = {
    {"counter_delta_available",
     maintenance_summary.exact_insert_counter_delta_available},
    {"items", maintenance_summary.exact_insert_items},
    {"total_ns", maintenance_summary.exact_insert_total_ns},
    {"remote_read_ns", maintenance_summary.exact_insert_remote_read_ns},
    {"remote_reverse_ns",
     maintenance_summary.exact_insert_remote_reverse_ns},
    {"remote_dependency_ns", exact_remote_dependency_ns},
    {"rdma_wait_ns", exact_rdma_wait_ns},
    {"cpu_and_local_ns", exact_cpu_and_local_ns},
    {"rdma_wait_ratio",
     maintenance_summary.exact_insert_total_ns == 0 ? 0.0 :
       static_cast<double>(exact_rdma_wait_ns) /
       static_cast<double>(maintenance_summary.exact_insert_total_ns)},
    {"avg_rdma_wait_us",
     maintenance_summary.exact_insert_items == 0 ? 0.0 :
       static_cast<double>(exact_rdma_wait_ns) /
       static_cast<double>(maintenance_summary.exact_insert_items) / 1e3},
    {"local_and_protocol_ns", exact_local_ns},
    {"remote_dependency_ratio",
     maintenance_summary.exact_insert_total_ns == 0 ? 0.0 :
       static_cast<double>(exact_remote_dependency_ns) /
       static_cast<double>(maintenance_summary.exact_insert_total_ns)},
    {"avg_total_us",
     maintenance_summary.exact_insert_items == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.exact_insert_total_ns) /
       static_cast<double>(maintenance_summary.exact_insert_items) / 1e3},
    {"avg_remote_dependency_us",
     maintenance_summary.exact_insert_items == 0 ? 0.0 :
       static_cast<double>(exact_remote_dependency_ns) /
       static_cast<double>(maintenance_summary.exact_insert_items) / 1e3},
    {"search_ns", maintenance_summary.exact_insert_search_ns},
    {"prune_ns", maintenance_summary.exact_insert_prune_ns},
    {"allocate_write_ns",
     maintenance_summary.exact_insert_allocate_write_ns},
    {"local_reverse_ns",
     maintenance_summary.exact_insert_local_reverse_ns},
    {"stage1_local_search_ns",
     maintenance_summary.exact_insert_stage1_local_search_ns},
    {"stage2_global_continuation_ns",
     maintenance_summary.exact_insert_stage2_global_continuation_ns},
    {"final_candidate_snapshot_ns",
     maintenance_summary.exact_insert_final_candidate_snapshot_ns},
    {"deferred_stage2_ns", exact_deferred_stage2_ns},
    {"deferred_stage2_ratio",
     maintenance_summary.exact_insert_total_ns == 0 ? 0.0 :
       static_cast<double>(exact_deferred_stage2_ns) /
       static_cast<double>(maintenance_summary.exact_insert_total_ns)},
    {"stack", {
      {"stage1_local_search_ns",
       maintenance_summary.exact_insert_stage1_local_search_ns},
      {"global_continuation_ns",
       maintenance_summary.exact_insert_stage2_global_continuation_ns},
      {"final_candidate_snapshot_ns",
       maintenance_summary.exact_insert_final_candidate_snapshot_ns},
      {"remote_reverse_ns",
       maintenance_summary.exact_insert_remote_reverse_ns},
      {"final_prune_ns", maintenance_summary.exact_insert_prune_ns},
      {"allocate_write_ns",
       maintenance_summary.exact_insert_allocate_write_ns},
      {"local_reverse_ns",
       maintenance_summary.exact_insert_local_reverse_ns},
      {"metadata_and_other_ns", exact_metadata_other_ns},
    }},
  };
  root["stage2"] = {
    {"source", in_band_maintenance_telemetry
      ? "in_band_control_page" : "storage_logs"},
    {"requested_logs", maintenance_summary.requested_logs},
    {"readable_logs", maintenance_summary.readable_logs},
    {"logs_with_observations", maintenance_summary.logs_with_observations},
    {"logs_with_slope_observations",
     maintenance_summary.logs_with_slope_observations},
    {"observations", maintenance_summary.observations},
    {"unreadable_logs", unreadable_logs_json(maintenance_summary)},
    {"remaining", maintenance_summary.remaining},
    {"max_backlog_observed", maintenance_summary.max_backlog_observed},
    {"backlog_slope_per_sec", maintenance_summary.backlog_slope_per_sec},
    {"backlog_slope_available", maintenance_summary.backlog_slope_available},
    {"p99_stage2_delay_upper_ms",
     maintenance_summary.p99_stage2_delay_upper_ms},
    {"p99_stage2_delay_over_30s",
     maintenance_summary.p99_stage2_delay_over_30s},
    {"p99_stage2_delay_samples", maintenance_summary.p99_stage2_delay_samples},
    {"p99_stage2_delay_available",
     maintenance_summary.p99_stage2_delay_available},
    {"latency_sum_delta_available",
     maintenance_summary.stage2_latency_sum_delta_available},
    {"finalize_latency_ns",
     maintenance_summary.stage2_finalize_latency_ns},
    {"avg_stage2_delay_ms",
     maintenance_summary.stage2_finalized_live == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_finalize_latency_ns) /
       static_cast<double>(maintenance_summary.stage2_finalized_live) / 1e6},
    {"failures", maintenance_summary.failures},
    {"failure_delta_available", maintenance_summary.failure_delta_available},
    {"peer_reverse_retry_attempts",
     maintenance_summary.peer_reverse_retry_attempts},
    {"peer_reverse_retry_delta_available",
     maintenance_summary.peer_reverse_retry_delta_available},
    {"admission_window", maintenance_summary.admission_window},
    {"completion_outstanding", maintenance_summary.completion_outstanding},
    {"max_completion_outstanding_per_shard",
     maintenance_summary.max_completion_outstanding_per_shard},
    {"completion_incomplete", maintenance_summary.completion_incomplete},
    {"max_completion_incomplete_per_shard",
     maintenance_summary.max_completion_incomplete_per_shard},
    {"completed_behind_hole",
     maintenance_summary.completion_outstanding >=
         maintenance_summary.completion_incomplete
       ? maintenance_summary.completion_outstanding -
           maintenance_summary.completion_incomplete
       : 0},
    {"exact_completion_credit_available",
     maintenance_summary.exact_completion_credit_available},
    {"completion_logical_full_failures",
     maintenance_summary.completion_logical_full_failures},
    {"completion_physical_full_failures",
     maintenance_summary.completion_physical_full_failures},
    {"completion_admission_failure_delta_available",
     maintenance_summary.completion_admission_failure_delta_available},
    {"active_stage2_contexts_latest_sum",
     maintenance_summary.active_stage2_contexts_latest_sum},
    {"active_stage2_context_limit_sum",
     maintenance_summary.active_stage2_context_limit_sum},
    {"completion_window_available",
     maintenance_summary.completion_window_available},
    {"locality_delta_available",
     maintenance_summary.locality_delta_available},
    {"stage2_finalized_live_delta",
     maintenance_summary.stage2_finalized_live},
    {"stage2_continuations", maintenance_summary.stage2_continuations},
    {"stage2_remote_frontier_items",
     maintenance_summary.stage2_remote_frontier_items},
    {"avg_stage2_remote_frontier",
     maintenance_summary.stage2_continuations == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_remote_frontier_items) /
       static_cast<double>(maintenance_summary.stage2_continuations)},
    {"stage2_remote_expansions",
     maintenance_summary.stage2_remote_expansions},
    {"avg_stage2_remote_expansions",
     maintenance_summary.stage2_continuations == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_remote_expansions) /
       static_cast<double>(maintenance_summary.stage2_continuations)},
    {"stage2_scored_candidates",
     maintenance_summary.stage2_scored_candidates},
    {"avg_stage2_scored_candidates",
     maintenance_summary.stage2_continuations == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_scored_candidates) /
       static_cast<double>(maintenance_summary.stage2_continuations)},
    {"stage2_migrations", maintenance_summary.stage2_migrations},
    {"home_match_rate",
     maintenance_summary.stage2_finalized_live == 0 ? 0.0 :
       1.0 - static_cast<double>(maintenance_summary.stage2_migrations) /
       static_cast<double>(maintenance_summary.stage2_finalized_live)},
    {"stage2_final_edges", maintenance_summary.stage2_final_edges},
    {"stage2_cross_edges_stage1_home",
     maintenance_summary.stage2_cross_edges_stage1_home},
    {"stage2_cross_edges_final_home",
     maintenance_summary.stage2_cross_edges_final_home},
    {"cross_edge_reduction_ratio",
     maintenance_summary.stage2_cross_edges_stage1_home == 0 ? 0.0 :
       1.0 - static_cast<double>(
         maintenance_summary.stage2_cross_edges_final_home) /
       static_cast<double>(
         maintenance_summary.stage2_cross_edges_stage1_home)},
    {"search_budget_delta_available",
     maintenance_summary.search_budget_delta_available},
    {"stage1_search_budget_exhausted",
     maintenance_summary.stage1_search_budget_exhausted},
    {"stage2_search_budget_exhausted",
     maintenance_summary.stage2_search_budget_exhausted},
    {"execution_counter_delta_available",
     maintenance_summary.execution_counter_delta_available},
    {"pressure_yields", maintenance_summary.pressure_yields},
    {"stage2_batches", maintenance_summary.stage2_batches},
    {"stage2_batched_items", maintenance_summary.stage2_batched_items},
    {"avg_stage2_batch_size",
     maintenance_summary.stage2_batches == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_batched_items) /
       static_cast<double>(maintenance_summary.stage2_batches)},
    {"stage2_packing", {
      {"counter_delta_available",
       maintenance_summary.packing_delta_available},
      {"target_batch_max", maintenance_summary.packing_target_batch_max},
      {"arrival_interval_us_max",
       maintenance_summary.packing_arrival_interval_us_max},
      {"waited_batches", maintenance_summary.packing_waited_batches},
      {"wait_ns", maintenance_summary.packing_wait_ns},
      {"target_flushes", maintenance_summary.packing_target_flushes},
      {"deadline_flushes", maintenance_summary.packing_deadline_flushes},
      {"cleanup_flushes", maintenance_summary.packing_cleanup_flushes},
    }},
    {"stage2_graph_read_waves",
     maintenance_summary.stage2_graph_read_waves},
    {"stage2_graph_unique_reads",
     maintenance_summary.stage2_graph_unique_reads},
    {"avg_stage2_graph_reads_per_wave",
     maintenance_summary.stage2_graph_read_waves == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_graph_unique_reads) /
       static_cast<double>(maintenance_summary.stage2_graph_read_waves)},
    {"ordered_graph_issue", {
      {"issued", maintenance_summary.stage2_graph_prefetch_issued},
      {"hits", maintenance_summary.stage2_graph_prefetch_hits},
      {"wasted", maintenance_summary.stage2_graph_prefetch_wasted},
      {"promotion_ratio",
       maintenance_summary.stage2_graph_prefetch_hits +
           maintenance_summary.stage2_graph_prefetch_wasted == 0 ? 0.0 :
         static_cast<double>(
           maintenance_summary.stage2_graph_prefetch_hits) /
         static_cast<double>(
           maintenance_summary.stage2_graph_prefetch_hits +
           maintenance_summary.stage2_graph_prefetch_wasted)},
    }},
    {"stage2_vector_read_waves",
     maintenance_summary.stage2_vector_read_waves},
    {"stage2_vector_unique_reads",
     maintenance_summary.stage2_vector_unique_reads},
    {"avg_stage2_vector_reads_per_wave",
     maintenance_summary.stage2_vector_read_waves == 0 ? 0.0 :
       static_cast<double>(maintenance_summary.stage2_vector_unique_reads) /
       static_cast<double>(maintenance_summary.stage2_vector_read_waves)},
    {"home_rpc_wire", {
      {"counter_delta_available",
       maintenance_summary.home_rpc_wire_counter_delta_available},
      {"batches", maintenance_summary.stage2_home_rpc_batches},
      {"items", maintenance_summary.stage2_home_rpc_items},
      {"avg_items_per_rpc",
       maintenance_summary.stage2_home_rpc_batches == 0 ? 0.0 :
         static_cast<double>(maintenance_summary.stage2_home_rpc_items) /
         static_cast<double>(maintenance_summary.stage2_home_rpc_batches)},
      {"graph_batches", graph_home_rpc_batches},
      {"graph_items", graph_home_rpc_items},
      {"avg_graph_items_per_rpc",
       graph_home_rpc_batches == 0 ? 0.0 :
         static_cast<double>(graph_home_rpc_items) /
         static_cast<double>(graph_home_rpc_batches)},
      {"scored_neighbors",
       maintenance_summary.stage2_home_scored_neighbors},
      {"avg_scored_neighbors_per_home_item",
       maintenance_summary.stage2_home_rpc_items == 0 ? 0.0 :
         static_cast<double>(
           maintenance_summary.stage2_home_scored_neighbors) /
         static_cast<double>(maintenance_summary.stage2_home_rpc_items)},
    }},
    {"score_rpc_wire", {
      {"counter_delta_available",
       maintenance_summary.score_rpc_wire_counter_delta_available},
      {"batches", maintenance_summary.stage2_home_score_rpc_batches},
      {"items", maintenance_summary.stage2_home_score_rpc_items},
      {"query_vectors", maintenance_summary.stage2_home_score_rpc_queries},
      {"request_bytes",
       maintenance_summary.stage2_home_score_rpc_request_bytes},
      {"response_bytes",
       maintenance_summary.stage2_home_score_rpc_response_bytes},
      {"avg_items_per_rpc",
       maintenance_summary.stage2_home_score_rpc_batches == 0 ? 0.0 :
         static_cast<double>(
           maintenance_summary.stage2_home_score_rpc_items) /
         static_cast<double>(
           maintenance_summary.stage2_home_score_rpc_batches)},
      {"query_dedup_ratio",
       maintenance_summary.stage2_home_score_rpc_items == 0 ? 0.0 :
         1.0 - static_cast<double>(
           maintenance_summary.stage2_home_score_rpc_queries) /
         static_cast<double>(
           maintenance_summary.stage2_home_score_rpc_items)},
    }},
    {"timing_counter_delta_available",
     maintenance_summary.timing_counter_delta_available},
    {"phase_timing", stage2_phase_timing},
    {"maintenance_worker_idle_waits",
     maintenance_summary.maintenance_worker_idle_waits},
    {"maintenance_worker_idle_ns",
     maintenance_summary.maintenance_worker_idle_ns},
    {"maintenance_lost_wake_avoided",
     maintenance_summary.maintenance_lost_wake_avoided},
    {"wake_counter_delta_available",
     maintenance_summary.wake_counter_delta_available},
    {"maintenance_targeted_wakes",
     maintenance_summary.maintenance_targeted_wakes},
    {"maintenance_generic_wakes",
     maintenance_summary.maintenance_generic_wakes},
    {"maintenance_broadcast_wakes",
     maintenance_summary.maintenance_broadcast_wakes},
    {"maintenance_context_slots_scanned",
     maintenance_summary.maintenance_context_slots_scanned},
    {"physical_stage1", {
      {"items", maintenance_summary.physical_stage1_items},
      {"total_ns", maintenance_summary.physical_stage1_total_ns},
      {"search_ns", maintenance_summary.physical_stage1_search_ns},
      {"prune_ns", maintenance_summary.physical_stage1_prune_ns},
      {"allocate_write_ns",
       maintenance_summary.physical_stage1_allocate_write_ns},
      {"backlink_ns", maintenance_summary.physical_stage1_backlink_ns},
      {"candidates", maintenance_summary.physical_stage1_candidates},
      {"remote_frontier_items",
       maintenance_summary.physical_stage1_remote_frontier_items},
      {"neighbors", maintenance_summary.physical_stage1_neighbors},
      {"avg_total_us", stage1_average_us(
        maintenance_summary.physical_stage1_total_ns)},
      {"avg_search_us", stage1_average_us(
        maintenance_summary.physical_stage1_search_ns)},
      {"avg_prune_us", stage1_average_us(
        maintenance_summary.physical_stage1_prune_ns)},
      {"avg_allocate_write_us", stage1_average_us(
        maintenance_summary.physical_stage1_allocate_write_ns)},
      {"avg_backlink_us", stage1_average_us(
        maintenance_summary.physical_stage1_backlink_ns)},
      {"avg_candidates", stage1_average_count(
        maintenance_summary.physical_stage1_candidates)},
      {"avg_remote_frontier", stage1_average_count(
        maintenance_summary.physical_stage1_remote_frontier_items)},
      {"avg_neighbors", stage1_average_count(
        maintenance_summary.physical_stage1_neighbors)},
    }},
    {"observation_period_seconds_assumed", 5.0},
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
  return root;
}




}  // namespace tools::breakdown_benchmark
