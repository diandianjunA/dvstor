#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "common/statistics.hh"
#include "common/types.hh"
#include "nlohmann/json.hh"

namespace service::breakdown {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

enum class Operation : u8 { query = 0, insert = 1 };

enum class Category : u8 {
  cpu = 0,
  gpu,
  rdma,
  transfer,
  count
};

constexpr size_t kCategoryCount = static_cast<size_t>(Category::count);

inline constexpr std::array<std::string_view, kCategoryCount> kCategoryNames = {
  "cpu_ns",
  "gpu_ns",
  "rdma_ns",
  "transfer_ns",
};

enum class Subcategory : u8 {
  // CPU
  cpu_cache_lookup = 0,
  cpu_query_select,
  cpu_query_filter,
  cpu_query_stage_candidates,
  cpu_query_beam_update,
  cpu_query_rerank_collect,
  cpu_query_rerank_prepare,
  cpu_query_rerank_update,
  cpu_query_beam_sort,
  cpu_query_result_ids,
  cpu_query_finalize,
  cpu_insert_init,
  cpu_insert_select,
  cpu_insert_filter,
  cpu_insert_stage_candidates,
  cpu_insert_preprune_sort,
  cpu_insert_candidate_collect,
  cpu_insert_beam_update,
  cpu_insert_candidate_sort,
  cpu_insert_prune_prepare,
  cpu_insert_quantize_prepare,
  cpu_insert_neighbor_collect,
  cpu_insert_finalize,
  cpu_insert_neighbor_prepare,
  cpu_insert_pruned_neighbor_collect,
  cpu_insert_overflow_prepare,

  // GPU
  gpu_query_prepare,
  gpu_query_distance,
  gpu_query_rerank,
  gpu_insert_distance,
  gpu_insert_prune,
  gpu_insert_quantize,
  gpu_insert_overflow_distance,
  gpu_insert_overflow_prune,

  // RDMA
  rdma_medoid_ptr,
  rdma_neighbor_fetch,
  rdma_rabitq_fetch,
  rdma_vector_fetch,
  rdma_rerank_fetch,
  rdma_alloc,
  rdma_new_node_write,
  rdma_medoid_update,
  rdma_header_write,
  rdma_candidate_fetch,
  rdma_neighbor_node_read,
  rdma_neighbor_lock,
  rdma_neighbor_list_read,
  rdma_neighbor_list_write,
  rdma_overflow_vec_fetch,
  rdma_pruned_neighbor_write,
  rdma_neighbor_unlock,

  // Transfer
  transfer_query_h2d,
  transfer_rabitq_h2d,
  transfer_candidate_h2d,
  transfer_distance_d2h,
  transfer_rerank_h2d,
  transfer_rerank_d2h,
  transfer_insert_query_h2d,
  transfer_prune_h2d,
  transfer_prune_d2h,
  transfer_quantize_d2h,
  transfer_overflow_query_h2d,
  transfer_overflow_candidate_h2d,
  transfer_overflow_dist_d2h,
  transfer_overflow_prune_inputs_h2d,
  transfer_overflow_prune_d2h,
  count
};

constexpr size_t kSubcategoryCount = static_cast<size_t>(Subcategory::count);

inline constexpr std::array<std::string_view, kSubcategoryCount> kSubcategoryNames = {
  "cpu_cache_lookup_ns",
  "cpu_query_select_ns",
  "cpu_query_filter_ns",
  "cpu_query_stage_candidates_ns",
  "cpu_query_beam_update_ns",
  "cpu_query_rerank_collect_ns",
  "cpu_query_rerank_prepare_ns",
  "cpu_query_rerank_update_ns",
  "cpu_query_beam_sort_ns",
  "cpu_query_result_ids_ns",
  "cpu_query_finalize_ns",
  "cpu_insert_init_ns",
  "cpu_insert_select_ns",
  "cpu_insert_filter_ns",
  "cpu_insert_stage_candidates_ns",
  "cpu_insert_preprune_sort_ns",
  "cpu_insert_candidate_collect_ns",
  "cpu_insert_beam_update_ns",
  "cpu_insert_candidate_sort_ns",
  "cpu_insert_prune_prepare_ns",
  "cpu_insert_quantize_prepare_ns",
  "cpu_insert_neighbor_collect_ns",
  "cpu_insert_finalize_ns",
  "cpu_insert_neighbor_prepare_ns",
  "cpu_insert_pruned_neighbor_collect_ns",
  "cpu_insert_overflow_prepare_ns",
  "gpu_query_prepare_ns",
  "gpu_query_distance_ns",
  "gpu_query_rerank_ns",
  "gpu_insert_distance_ns",
  "gpu_insert_prune_ns",
  "gpu_insert_quantize_ns",
  "gpu_insert_overflow_distance_ns",
  "gpu_insert_overflow_prune_ns",
  "rdma_medoid_ptr_ns",
  "rdma_neighbor_fetch_ns",
  "rdma_rabitq_fetch_ns",
  "rdma_vector_fetch_ns",
  "rdma_rerank_fetch_ns",
  "rdma_alloc_ns",
  "rdma_new_node_write_ns",
  "rdma_medoid_update_ns",
  "rdma_header_write_ns",
  "rdma_candidate_fetch_ns",
  "rdma_neighbor_node_read_ns",
  "rdma_neighbor_lock_ns",
  "rdma_neighbor_list_read_ns",
  "rdma_neighbor_list_write_ns",
  "rdma_overflow_vec_fetch_ns",
  "rdma_pruned_neighbor_write_ns",
  "rdma_neighbor_unlock_ns",
  "transfer_query_h2d_ns",
  "transfer_rabitq_h2d_ns",
  "transfer_candidate_h2d_ns",
  "transfer_distance_d2h_ns",
  "transfer_rerank_h2d_ns",
  "transfer_rerank_d2h_ns",
  "transfer_insert_query_h2d_ns",
  "transfer_prune_h2d_ns",
  "transfer_prune_d2h_ns",
  "transfer_quantize_d2h_ns",
  "transfer_overflow_query_h2d_ns",
  "transfer_overflow_candidate_h2d_ns",
  "transfer_overflow_dist_d2h_ns",
  "transfer_overflow_prune_inputs_h2d_ns",
  "transfer_overflow_prune_d2h_ns",
};

inline constexpr std::string_view operation_name(const Operation operation) {
  return operation == Operation::query ? "query" : "insert";
}

inline constexpr Category parent_category(const Subcategory subcategory) {
  switch (subcategory) {
    case Subcategory::cpu_cache_lookup:
    case Subcategory::cpu_query_select:
    case Subcategory::cpu_query_filter:
    case Subcategory::cpu_query_stage_candidates:
    case Subcategory::cpu_query_beam_update:
    case Subcategory::cpu_query_rerank_collect:
    case Subcategory::cpu_query_rerank_prepare:
    case Subcategory::cpu_query_rerank_update:
    case Subcategory::cpu_query_beam_sort:
    case Subcategory::cpu_query_result_ids:
    case Subcategory::cpu_query_finalize:
    case Subcategory::cpu_insert_init:
    case Subcategory::cpu_insert_select:
    case Subcategory::cpu_insert_filter:
    case Subcategory::cpu_insert_stage_candidates:
    case Subcategory::cpu_insert_preprune_sort:
    case Subcategory::cpu_insert_candidate_collect:
    case Subcategory::cpu_insert_beam_update:
    case Subcategory::cpu_insert_candidate_sort:
    case Subcategory::cpu_insert_prune_prepare:
    case Subcategory::cpu_insert_quantize_prepare:
    case Subcategory::cpu_insert_neighbor_collect:
    case Subcategory::cpu_insert_finalize:
    case Subcategory::cpu_insert_neighbor_prepare:
    case Subcategory::cpu_insert_pruned_neighbor_collect:
    case Subcategory::cpu_insert_overflow_prepare:
      return Category::cpu;
    case Subcategory::gpu_query_prepare:
    case Subcategory::gpu_query_distance:
    case Subcategory::gpu_query_rerank:
    case Subcategory::gpu_insert_distance:
    case Subcategory::gpu_insert_prune:
    case Subcategory::gpu_insert_quantize:
    case Subcategory::gpu_insert_overflow_distance:
    case Subcategory::gpu_insert_overflow_prune:
      return Category::gpu;
    case Subcategory::rdma_medoid_ptr:
    case Subcategory::rdma_neighbor_fetch:
    case Subcategory::rdma_rabitq_fetch:
    case Subcategory::rdma_vector_fetch:
    case Subcategory::rdma_rerank_fetch:
    case Subcategory::rdma_alloc:
    case Subcategory::rdma_new_node_write:
    case Subcategory::rdma_medoid_update:
    case Subcategory::rdma_header_write:
    case Subcategory::rdma_candidate_fetch:
    case Subcategory::rdma_neighbor_node_read:
    case Subcategory::rdma_neighbor_lock:
    case Subcategory::rdma_neighbor_list_read:
    case Subcategory::rdma_neighbor_list_write:
    case Subcategory::rdma_overflow_vec_fetch:
    case Subcategory::rdma_pruned_neighbor_write:
    case Subcategory::rdma_neighbor_unlock:
      return Category::rdma;
    case Subcategory::transfer_query_h2d:
    case Subcategory::transfer_rabitq_h2d:
    case Subcategory::transfer_candidate_h2d:
    case Subcategory::transfer_distance_d2h:
    case Subcategory::transfer_rerank_h2d:
    case Subcategory::transfer_rerank_d2h:
    case Subcategory::transfer_insert_query_h2d:
    case Subcategory::transfer_prune_h2d:
    case Subcategory::transfer_prune_d2h:
    case Subcategory::transfer_quantize_d2h:
    case Subcategory::transfer_overflow_query_h2d:
    case Subcategory::transfer_overflow_candidate_h2d:
    case Subcategory::transfer_overflow_dist_d2h:
    case Subcategory::transfer_overflow_prune_inputs_h2d:
    case Subcategory::transfer_overflow_prune_d2h:
      return Category::transfer;
    case Subcategory::count:
      return Category::cpu;
  }
  return Category::cpu;
}

struct ThreadCounterDelta {
  u64 rdma_read_bytes{};
  u64 rdma_write_bytes{};
  u64 neighbor_rdma_bytes{};
  u64 vector_rdma_bytes{};
  u64 rabitq_rdma_bytes{};
  u64 h2d_bytes{};
  u64 d2h_bytes{};
  u64 l2_kernels{};
  u64 prune_kernels{};
  u64 rabitq_kernels{};
  u64 exact_reranks{};
  u64 visited_nodes{};
  u64 visited_neighborlists{};
  u64 remote_allocations{};
  u64 overflow_prunes{};
  u64 cache_hits{};
  u64 cache_misses{};
  u64 query_rdma_to_staging_bytes{};
  u64 query_host_staging_fallback_bytes{};
  u64 neighbor_cache_hits{};
  u64 neighbor_cache_misses{};
  u64 gpu_rabitq_cache_hits{};
  u64 gpu_rabitq_cache_misses{};
  u64 gpu_rabitq_cache_fills{};
  u64 gpu_rabitq_cache_fill_bytes{};
  u64 gpu_rabitq_cache_duplicate_fills{};
  u64 gpu_rabitq_cache_fallback_batches{};
};


inline ThreadCounterDelta diff_thread_counters(const statistics::ThreadStatistics& end,
                                               const statistics::ThreadStatistics& start,
                                               const Operation operation) {
  ThreadCounterDelta out{};

  if (operation == Operation::query) {
    out.rdma_read_bytes = end.query_rdma_reads_in_bytes - start.query_rdma_reads_in_bytes;
    out.rdma_write_bytes = end.query_rdma_writes_in_bytes - start.query_rdma_writes_in_bytes;
    out.neighbor_rdma_bytes = end.query_neighbor_rdma_reads_in_bytes - start.query_neighbor_rdma_reads_in_bytes;
    out.vector_rdma_bytes = end.query_vector_rdma_reads_in_bytes - start.query_vector_rdma_reads_in_bytes;
    out.rabitq_rdma_bytes = end.query_rabitq_rdma_reads_in_bytes - start.query_rabitq_rdma_reads_in_bytes;
    out.h2d_bytes = end.query_h2d_bytes - start.query_h2d_bytes;
    out.d2h_bytes = end.query_d2h_bytes - start.query_d2h_bytes;
    out.rabitq_kernels = end.query_rabitq_kernels - start.query_rabitq_kernels;
    out.exact_reranks = end.query_exact_reranks - start.query_exact_reranks;
    out.visited_nodes =
      (end.visited_nodes - start.visited_nodes) + (end.visited_nodes_l0 - start.visited_nodes_l0);
    out.visited_neighborlists = end.visited_neighborlists - start.visited_neighborlists;
    out.cache_hits = end.cache_hits - start.cache_hits;
    out.cache_misses = end.cache_misses - start.cache_misses;
    out.query_rdma_to_staging_bytes = end.query_rdma_to_staging_bytes - start.query_rdma_to_staging_bytes;
    out.query_host_staging_fallback_bytes = end.query_host_staging_fallback_bytes - start.query_host_staging_fallback_bytes;
    out.neighbor_cache_hits = end.neighbor_cache_hits - start.neighbor_cache_hits;
    out.neighbor_cache_misses = end.neighbor_cache_misses - start.neighbor_cache_misses;
    out.gpu_rabitq_cache_hits = end.gpu_rabitq_cache_hits - start.gpu_rabitq_cache_hits;
    out.gpu_rabitq_cache_misses = end.gpu_rabitq_cache_misses - start.gpu_rabitq_cache_misses;
    out.gpu_rabitq_cache_fills = end.gpu_rabitq_cache_fills - start.gpu_rabitq_cache_fills;
    out.gpu_rabitq_cache_fill_bytes = end.gpu_rabitq_cache_fill_bytes - start.gpu_rabitq_cache_fill_bytes;
    out.gpu_rabitq_cache_duplicate_fills =
      end.gpu_rabitq_cache_duplicate_fills - start.gpu_rabitq_cache_duplicate_fills;
    out.gpu_rabitq_cache_fallback_batches =
      end.gpu_rabitq_cache_fallback_batches - start.gpu_rabitq_cache_fallback_batches;
    return out;
  }

  out.rdma_read_bytes = end.build_rdma_reads_in_bytes - start.build_rdma_reads_in_bytes;
  out.rdma_write_bytes = end.build_rdma_writes_in_bytes - start.build_rdma_writes_in_bytes;
  out.neighbor_rdma_bytes = end.build_neighbor_rdma_reads_in_bytes - start.build_neighbor_rdma_reads_in_bytes;
  out.vector_rdma_bytes = end.build_vector_rdma_reads_in_bytes - start.build_vector_rdma_reads_in_bytes;
  out.rabitq_rdma_bytes = end.build_rabitq_rdma_reads_in_bytes - start.build_rabitq_rdma_reads_in_bytes;
  out.h2d_bytes = end.build_h2d_bytes - start.build_h2d_bytes;
  out.d2h_bytes = end.build_d2h_bytes - start.build_d2h_bytes;
  out.l2_kernels = end.build_l2_kernels - start.build_l2_kernels;
  out.prune_kernels = end.build_prune_kernels - start.build_prune_kernels;
  out.remote_allocations = end.remote_allocations - start.remote_allocations;
  out.overflow_prunes = end.build_overflow_prunes - start.build_overflow_prunes;
  out.cache_hits = end.cache_hits - start.cache_hits;
  out.cache_misses = end.cache_misses - start.cache_misses;
  return out;
}

struct Sample {
  explicit Sample(Operation op) : operation(op) {}

  Operation operation;
  Clock::time_point enqueued_at{};
  Clock::time_point dequeued_at{};
  Clock::time_point started_at{};
  Clock::time_point finished_at{};
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};
  statistics::ThreadStatistics start_counters{};
  statistics::ThreadStatistics end_counters{};
  u64 queue_wait_ns{};
  u64 service_ns{};
  u64 end_to_end_ns{};
  u64 lock_attempts{};
  u64 lock_retries{};
  u64 cas_failures{};
  bool started_flag{};
  bool finished_flag{};

  void mark_started(const Clock::time_point dequeued,
                    const Clock::time_point started,
                    const statistics::ThreadStatistics& counters) {
    dequeued_at = dequeued;
    started_at = started;
    start_counters = counters;
    started_flag = true;
    queue_wait_ns =
      static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(dequeued_at - enqueued_at).count());
  }

  void mark_finished(const Clock::time_point finished, const statistics::ThreadStatistics& counters) {
    finished_at = finished;
    end_counters = counters;
    finished_flag = true;
    service_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - started_at).count());
    end_to_end_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - enqueued_at).count());
  }

  void add_subcategory(const Subcategory subcategory, const u64 ns) {
    subcategory_ns[static_cast<size_t>(subcategory)] += ns;
    category_ns[static_cast<size_t>(parent_category(subcategory))] += ns;
  }

  ThreadCounterDelta counters() const { return diff_thread_counters(end_counters, start_counters, operation); }
};

struct Aggregate {
  Operation operation{Operation::query};
  size_t count{};
  u64 total_queue_wait_ns{};
  u64 total_service_ns{};
  u64 total_end_to_end_ns{};
  std::vector<u64> end_to_end_latencies_ns{};
  std::vector<u64> service_latencies_ns{};
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};
  ThreadCounterDelta counters{};
  u64 lock_attempts{};
  u64 lock_retries{};
  u64 cas_failures{};

  [[nodiscard]] u64 measured_total_ns() const {
    u64 total = 0;
    for (const u64 value : category_ns) {
      total += value;
    }
    return total;
  }

  [[nodiscard]] u64 cpu_other_ns() const {
    u64 explicit_cpu = 0;
    for (size_t i = 0; i < subcategory_ns.size(); ++i) {
      const auto sub = static_cast<Subcategory>(i);
      if (parent_category(sub) == Category::cpu) {
        explicit_cpu += subcategory_ns[i];
      }
    }
    const u64 cpu_total = total_service_ns > (category_ns[1] + category_ns[2] + category_ns[3])
                            ? total_service_ns - (category_ns[1] + category_ns[2] + category_ns[3])
                            : 0;
    return cpu_total > explicit_cpu ? cpu_total - explicit_cpu : 0;
  }
};

struct Report {
  Aggregate query{};
  Aggregate insert{};

  [[nodiscard]] bool has_query() const { return query.count > 0; }
  [[nodiscard]] bool has_insert() const { return insert.count > 0; }
};

inline void add_sample(Aggregate& aggregate, const Sample& sample) {
  if (!sample.finished_flag) {
    return;
  }

  aggregate.operation = sample.operation;
  ++aggregate.count;
  aggregate.total_queue_wait_ns += sample.queue_wait_ns;
  aggregate.total_service_ns += sample.service_ns;
  aggregate.total_end_to_end_ns += sample.end_to_end_ns;
  aggregate.end_to_end_latencies_ns.push_back(sample.end_to_end_ns);
  aggregate.service_latencies_ns.push_back(sample.service_ns);
  for (size_t i = 0; i < aggregate.category_ns.size(); ++i) {
    aggregate.category_ns[i] += sample.category_ns[i];
  }
  for (size_t i = 0; i < aggregate.subcategory_ns.size(); ++i) {
    aggregate.subcategory_ns[i] += sample.subcategory_ns[i];
  }

  const ThreadCounterDelta delta = sample.counters();
  aggregate.counters.rdma_read_bytes += delta.rdma_read_bytes;
  aggregate.counters.rdma_write_bytes += delta.rdma_write_bytes;
  aggregate.counters.neighbor_rdma_bytes += delta.neighbor_rdma_bytes;
  aggregate.counters.vector_rdma_bytes += delta.vector_rdma_bytes;
  aggregate.counters.rabitq_rdma_bytes += delta.rabitq_rdma_bytes;
  aggregate.counters.h2d_bytes += delta.h2d_bytes;
  aggregate.counters.d2h_bytes += delta.d2h_bytes;
  aggregate.counters.l2_kernels += delta.l2_kernels;
  aggregate.counters.prune_kernels += delta.prune_kernels;
  aggregate.counters.rabitq_kernels += delta.rabitq_kernels;
  aggregate.counters.exact_reranks += delta.exact_reranks;
  aggregate.counters.visited_nodes += delta.visited_nodes;
  aggregate.counters.visited_neighborlists += delta.visited_neighborlists;
  aggregate.counters.remote_allocations += delta.remote_allocations;
  aggregate.counters.overflow_prunes += delta.overflow_prunes;
  aggregate.counters.cache_hits += delta.cache_hits;
  aggregate.counters.cache_misses += delta.cache_misses;
  aggregate.counters.query_rdma_to_staging_bytes += delta.query_rdma_to_staging_bytes;
  aggregate.counters.query_host_staging_fallback_bytes += delta.query_host_staging_fallback_bytes;
  aggregate.counters.neighbor_cache_hits += delta.neighbor_cache_hits;
  aggregate.counters.neighbor_cache_misses += delta.neighbor_cache_misses;
  aggregate.counters.gpu_rabitq_cache_hits += delta.gpu_rabitq_cache_hits;
  aggregate.counters.gpu_rabitq_cache_misses += delta.gpu_rabitq_cache_misses;
  aggregate.counters.gpu_rabitq_cache_fills += delta.gpu_rabitq_cache_fills;
  aggregate.counters.gpu_rabitq_cache_fill_bytes += delta.gpu_rabitq_cache_fill_bytes;
  aggregate.counters.gpu_rabitq_cache_duplicate_fills += delta.gpu_rabitq_cache_duplicate_fills;
  aggregate.counters.gpu_rabitq_cache_fallback_batches += delta.gpu_rabitq_cache_fallback_batches;
  aggregate.lock_attempts += sample.lock_attempts;
  aggregate.lock_retries += sample.lock_retries;
  aggregate.cas_failures += sample.cas_failures;
}

inline double ns_to_ms(const u64 ns) { return static_cast<double>(ns) / 1'000'000.0; }

inline u64 percentile_ns(std::vector<u64> values, const double percentile) {
  if (values.empty()) {
    return 0;
  }
  std::sort(values.begin(), values.end());
  const double idx = percentile * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(idx)];
}

inline nlohmann::json aggregate_to_json(const Aggregate& aggregate) {
  using json = nlohmann::json;
  json out;
  out["operation"] = operation_name(aggregate.operation);
  out["count"] = aggregate.count;
  out["latency"] = {
    {"queue_wait_ns", aggregate.total_queue_wait_ns},
    {"service_ns", aggregate.total_service_ns},
    {"end_to_end_ns", aggregate.total_end_to_end_ns},
    {"mean_queue_wait_ns", aggregate.count == 0 ? 0 : aggregate.total_queue_wait_ns / aggregate.count},
    {"mean_service_ns", aggregate.count == 0 ? 0 : aggregate.total_service_ns / aggregate.count},
    {"mean_end_to_end_ns", aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count},
    {"p50_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.50)},
    {"p95_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.95)},
    {"p99_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)},
    {"p50_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.50)},
    {"p95_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.95)},
    {"p99_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.99)},
  };

  const u64 cpu_total = aggregate.total_service_ns > (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          ? aggregate.total_service_ns - (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          : 0;

  json categories = json::object();
  categories["cpu_ns"] = cpu_total;
  categories["gpu_ns"] = aggregate.category_ns[static_cast<size_t>(Category::gpu)];
  categories["rdma_ns"] = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  categories["transfer_ns"] = aggregate.category_ns[static_cast<size_t>(Category::transfer)];
  out["breakdown"] = std::move(categories);

  json sub = json::object();
  for (size_t c = 0; c < kCategoryCount; ++c) {
    sub[std::string{kCategoryNames[c]}] = json::object();
  }
  for (size_t i = 0; i < kSubcategoryCount; ++i) {
    const auto subcat = static_cast<Subcategory>(i);
    sub[std::string{kCategoryNames[static_cast<size_t>(parent_category(subcat))]}]
       [std::string{kSubcategoryNames[i]}] = aggregate.subcategory_ns[i];
  }
  if (aggregate.operation == Operation::query) {
    sub["cpu_ns"]["cpu_query_runtime_overhead_ns"] = aggregate.cpu_other_ns();
  } else {
    sub["cpu_ns"]["cpu_insert_runtime_overhead_ns"] = aggregate.cpu_other_ns();
  }
  out["sub_breakdown"] = std::move(sub);

  out["counters"] = {
    {"rdma_read_bytes", aggregate.counters.rdma_read_bytes},
    {"rdma_write_bytes", aggregate.counters.rdma_write_bytes},
    {"neighbor_rdma_bytes", aggregate.counters.neighbor_rdma_bytes},
    {"vector_rdma_bytes", aggregate.counters.vector_rdma_bytes},
    {"rabitq_rdma_bytes", aggregate.counters.rabitq_rdma_bytes},
    {"h2d_bytes", aggregate.counters.h2d_bytes},
    {"d2h_bytes", aggregate.counters.d2h_bytes},
    {"l2_kernels", aggregate.counters.l2_kernels},
    {"prune_kernels", aggregate.counters.prune_kernels},
    {"rabitq_kernels", aggregate.counters.rabitq_kernels},
    {"exact_reranks", aggregate.counters.exact_reranks},
    {"visited_nodes", aggregate.counters.visited_nodes},
    {"visited_neighborlists", aggregate.counters.visited_neighborlists},
    {"remote_allocations", aggregate.counters.remote_allocations},
    {"overflow_prunes", aggregate.counters.overflow_prunes},
    {"cache_hits", aggregate.counters.cache_hits},
    {"cache_misses", aggregate.counters.cache_misses},
    {"query_rdma_to_staging_bytes", aggregate.counters.query_rdma_to_staging_bytes},
    {"query_host_staging_fallback_bytes", aggregate.counters.query_host_staging_fallback_bytes},
    {"neighbor_cache_hits", aggregate.counters.neighbor_cache_hits},
    {"neighbor_cache_misses", aggregate.counters.neighbor_cache_misses},
    {"gpu_rabitq_cache_hits", aggregate.counters.gpu_rabitq_cache_hits},
    {"gpu_rabitq_cache_misses", aggregate.counters.gpu_rabitq_cache_misses},
    {"gpu_rabitq_cache_fills", aggregate.counters.gpu_rabitq_cache_fills},
    {"gpu_rabitq_cache_fill_bytes", aggregate.counters.gpu_rabitq_cache_fill_bytes},
    {"gpu_rabitq_cache_duplicate_fills", aggregate.counters.gpu_rabitq_cache_duplicate_fills},
    {"gpu_rabitq_cache_fallback_batches", aggregate.counters.gpu_rabitq_cache_fallback_batches},
    {"lock_attempts", aggregate.lock_attempts},
    {"lock_retries", aggregate.lock_retries},
    {"cas_failures", aggregate.cas_failures},
  };
  return out;
}

inline std::string aggregate_text_summary(const Aggregate& aggregate) {
  std::ostringstream os;
  os << operation_name(aggregate.operation) << " breakdown\n";
  os << "  count: " << aggregate.count << '\n';
  os << "  latency_ms: mean=" << ns_to_ms(aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count)
     << " p50=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.50))
     << " p95=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.95))
     << " p99=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)) << '\n';

  const u64 cpu_total = aggregate.total_service_ns > (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          ? aggregate.total_service_ns - (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          : 0;

  std::vector<std::pair<std::string, u64>> ranked = {
    {"cpu_ns", cpu_total},
    {"gpu_ns", aggregate.category_ns[static_cast<size_t>(Category::gpu)]},
    {"rdma_ns", aggregate.category_ns[static_cast<size_t>(Category::rdma)]},
    {"transfer_ns", aggregate.category_ns[static_cast<size_t>(Category::transfer)]},
  };
  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  os << "  top_categories:\n";
  for (const auto& [name, value] : ranked) {
    const double ratio = aggregate.total_service_ns == 0
                           ? 0.0
                           : static_cast<double>(value) / static_cast<double>(aggregate.total_service_ns);
    os << "    " << name << ": " << ns_to_ms(value) << " ms (" << ratio * 100.0 << "%)\n";
  }
  return os.str();
}

inline nlohmann::json report_to_json(const Report& report) {
  nlohmann::json out;
  if (report.has_query()) {
    out["query_breakdown"] = aggregate_to_json(report.query);
  }
  if (report.has_insert()) {
    out["insert_breakdown"] = aggregate_to_json(report.insert);
  }
  return out;
}

}  // namespace service::breakdown
