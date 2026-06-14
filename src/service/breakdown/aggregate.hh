#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "service/breakdown/sample.hh"

namespace service::breakdown {

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
  aggregate.counters.rdma_read_ops += delta.rdma_read_ops;
  aggregate.counters.rdma_write_ops += delta.rdma_write_ops;
  aggregate.counters.neighbor_rdma_bytes += delta.neighbor_rdma_bytes;
  aggregate.counters.vector_rdma_bytes += delta.vector_rdma_bytes;
  aggregate.counters.neighbor_rdma_read_ops += delta.neighbor_rdma_read_ops;
  aggregate.counters.vector_rdma_read_ops += delta.vector_rdma_read_ops;
  aggregate.counters.vector_rdma_batch_calls += delta.vector_rdma_batch_calls;
  aggregate.counters.vector_rdma_cqes += delta.vector_rdma_cqes;
  aggregate.counters.h2d_bytes += delta.h2d_bytes;
  aggregate.counters.d2h_bytes += delta.d2h_bytes;
  aggregate.counters.l2_kernels += delta.l2_kernels;
  aggregate.counters.prune_kernels += delta.prune_kernels;
  aggregate.counters.exact_reranks += delta.exact_reranks;
  aggregate.counters.rabitq_l0_candidates += delta.rabitq_l0_candidates;
  aggregate.counters.rabitq_cache_misses += delta.rabitq_cache_misses;
  aggregate.counters.rabitq_l1_candidates += delta.rabitq_l1_candidates;
  aggregate.counters.rabitq_l2_candidates += delta.rabitq_l2_candidates;
  aggregate.counters.rabitq_forced_widen += delta.rabitq_forced_widen;
  aggregate.counters.rabitq_audit_expansions += delta.rabitq_audit_expansions;
  aggregate.counters.rabitq_audit_candidates += delta.rabitq_audit_candidates;
  aggregate.counters.rabitq_safe_skips += delta.rabitq_safe_skips;
  aggregate.counters.rabitq_exact_fallbacks += delta.rabitq_exact_fallbacks;
  aggregate.counters.rabitq_prefetch_issued += delta.rabitq_prefetch_issued;
  aggregate.counters.rabitq_prefetch_hits += delta.rabitq_prefetch_hits;
  aggregate.counters.rabitq_prefetch_misses += delta.rabitq_prefetch_misses;
  aggregate.counters.rabitq_prefetch_disabled_queries +=
    delta.rabitq_prefetch_disabled_queries;
  aggregate.counters.qir_qcode_rdma_ops += delta.qir_qcode_rdma_ops;
  aggregate.counters.qir_qcode_rdma_bytes += delta.qir_qcode_rdma_bytes;
  aggregate.counters.qir_qcode_cache_hits += delta.qir_qcode_cache_hits;
  aggregate.counters.qir_qcode_cache_misses += delta.qir_qcode_cache_misses;
  aggregate.counters.qir_exact_reads += delta.qir_exact_reads;
  aggregate.counters.qir_exact_reads_avoided += delta.qir_exact_reads_avoided;
  aggregate.counters.qir_uncertain_candidates += delta.qir_uncertain_candidates;
  aggregate.counters.qir_prune_fallbacks += delta.qir_prune_fallbacks;
  aggregate.counters.qir_repair_intents += delta.qir_repair_intents;
  aggregate.counters.qir_repair_queue_delay_ns += delta.qir_repair_queue_delay_ns;
  aggregate.counters.qir_repair_applied_edges += delta.qir_repair_applied_edges;
  aggregate.counters.qir_repair_stale_skips += delta.qir_repair_stale_skips;
  aggregate.counters.qir_sync_repair_fallbacks += delta.qir_sync_repair_fallbacks;
  aggregate.counters.qir_audit_samples += delta.qir_audit_samples;
  aggregate.counters.qir_audit_disagreements += delta.qir_audit_disagreements;
  aggregate.counters.visited_nodes += delta.visited_nodes;
  aggregate.counters.visited_neighborlists += delta.visited_neighborlists;
  aggregate.counters.remote_allocations += delta.remote_allocations;
  aggregate.counters.overflow_prunes += delta.overflow_prunes;
  aggregate.counters.overflow_prune_candidates += delta.overflow_prune_candidates;
  aggregate.counters.overflow_prune_max_candidates =
    std::max(aggregate.counters.overflow_prune_max_candidates, delta.overflow_prune_max_candidates);
  aggregate.counters.overflow_prune_max_candidates =
    std::max(aggregate.counters.overflow_prune_max_candidates, sample.overflow_prune_max_candidates);
  aggregate.counters.overflow_prune_pair_checks_upper_bound += delta.overflow_prune_pair_checks_upper_bound;
  aggregate.counters.overflow_prune_global_load_bytes_upper_bound +=
    delta.overflow_prune_global_load_bytes_upper_bound;
  aggregate.counters.overflow_prune_kernel_blocks += delta.overflow_prune_kernel_blocks;
  aggregate.counters.overflow_prune_kernel_threads += delta.overflow_prune_kernel_threads;
  aggregate.counters.overflow_prune_max_kernel_threads =
    std::max(aggregate.counters.overflow_prune_max_kernel_threads, delta.overflow_prune_max_kernel_threads);
  aggregate.counters.overflow_prune_max_kernel_threads =
    std::max(aggregate.counters.overflow_prune_max_kernel_threads, sample.overflow_prune_max_kernel_threads);
  aggregate.counters.query_rdma_to_staging_bytes += delta.query_rdma_to_staging_bytes;
  aggregate.counters.query_host_staging_fallback_bytes += delta.query_host_staging_fallback_bytes;
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

}  // namespace service::breakdown
