#pragma once

#include <algorithm>
#include <chrono>

#include "common/statistics.hh"
#include "service/breakdown/names.hh"

namespace service::breakdown {

struct ThreadCounterDelta {
  u64 rdma_read_bytes{};
  u64 rdma_write_bytes{};
  u64 rdma_read_ops{};
  u64 rdma_write_ops{};
  u64 neighbor_rdma_bytes{};
  u64 vector_rdma_bytes{};
  u64 neighbor_rdma_read_ops{};
  u64 vector_rdma_read_ops{};
  u64 vector_rdma_batch_calls{};
  u64 vector_rdma_cqes{};
  u64 h2d_bytes{};
  u64 d2h_bytes{};
  u64 l2_kernels{};
  u64 prune_kernels{};
  u64 exact_reranks{};
  u64 rabitq_l0_candidates{};
  u64 rabitq_cache_misses{};
  u64 rabitq_l1_candidates{};
  u64 rabitq_l2_candidates{};
  u64 rabitq_forced_widen{};
  u64 rabitq_audit_expansions{};
  u64 rabitq_audit_candidates{};
  u64 rabitq_safe_skips{};
  u64 rabitq_exact_fallbacks{};
  u64 rabitq_prefetch_issued{};
  u64 rabitq_prefetch_hits{};
  u64 rabitq_prefetch_misses{};
  u64 rabitq_prefetch_disabled_queries{};
  u64 visited_nodes{};
  u64 visited_neighborlists{};
  u64 remote_allocations{};
  u64 overflow_prunes{};
  u64 overflow_prune_candidates{};
  u64 overflow_prune_max_candidates{};
  u64 overflow_prune_pair_checks_upper_bound{};
  u64 overflow_prune_global_load_bytes_upper_bound{};
  u64 overflow_prune_kernel_blocks{};
  u64 overflow_prune_kernel_threads{};
  u64 overflow_prune_max_kernel_threads{};
  u64 query_rdma_to_staging_bytes{};
  u64 query_host_staging_fallback_bytes{};
  u64 storage_owner_handoff_requests{};
  u64 storage_owner_handoff_successes{};
  u64 storage_owner_handoff_queue_full{};
  u64 storage_owner_handoff_timeouts{};
  u64 storage_owner_handoff_overloaded{};
  u64 storage_owner_handoff_failed{};
  u64 storage_owner_handoff_request_bytes{};
  u64 storage_owner_handoff_response_bytes{};
  u64 storage_owner_handoff_remote_handler_ns{};
  u64 storage_owner_handoff_remote_expanded_nodes{};
  u64 storage_owner_handoff_remote_snapshot_reads{};
  u64 storage_owner_handoff_remote_neighbor_reads{};
  u64 storage_owner_handoff_response_beam_entries{};
  u64 storage_owner_handoff_response_visited_entries{};
  u64 storage_owner_handoff_response_visited_truncated{};
  u64 storage_owner_qdi_requests{};
  u64 storage_owner_qdi_successes{};
  u64 storage_owner_qdi_queue_full{};
  u64 storage_owner_qdi_timeouts{};
  u64 storage_owner_qdi_overloaded{};
  u64 storage_owner_qdi_failed{};
  u64 storage_owner_qdi_request_bytes{};
  u64 storage_owner_qdi_response_bytes{};
  u64 storage_owner_qdi_remote_handler_ns{};
  u64 storage_owner_qdi_remote_expanded_nodes{};
  u64 storage_owner_qdi_remote_approx_scores{};
  u64 storage_owner_qdi_remote_exact_reads{};
  u64 storage_owner_qdi_remote_neighbor_reads{};
  u64 storage_owner_qdi_response_candidates{};
};

inline void add_counter_delta(ThreadCounterDelta& lhs, const ThreadCounterDelta& rhs) {
  lhs.rdma_read_bytes += rhs.rdma_read_bytes;
  lhs.rdma_write_bytes += rhs.rdma_write_bytes;
  lhs.rdma_read_ops += rhs.rdma_read_ops;
  lhs.rdma_write_ops += rhs.rdma_write_ops;
  lhs.neighbor_rdma_bytes += rhs.neighbor_rdma_bytes;
  lhs.vector_rdma_bytes += rhs.vector_rdma_bytes;
  lhs.neighbor_rdma_read_ops += rhs.neighbor_rdma_read_ops;
  lhs.vector_rdma_read_ops += rhs.vector_rdma_read_ops;
  lhs.vector_rdma_batch_calls += rhs.vector_rdma_batch_calls;
  lhs.vector_rdma_cqes += rhs.vector_rdma_cqes;
  lhs.h2d_bytes += rhs.h2d_bytes;
  lhs.d2h_bytes += rhs.d2h_bytes;
  lhs.l2_kernels += rhs.l2_kernels;
  lhs.prune_kernels += rhs.prune_kernels;
  lhs.exact_reranks += rhs.exact_reranks;
  lhs.rabitq_l0_candidates += rhs.rabitq_l0_candidates;
  lhs.rabitq_cache_misses += rhs.rabitq_cache_misses;
  lhs.rabitq_l1_candidates += rhs.rabitq_l1_candidates;
  lhs.rabitq_l2_candidates += rhs.rabitq_l2_candidates;
  lhs.rabitq_forced_widen += rhs.rabitq_forced_widen;
  lhs.rabitq_audit_expansions += rhs.rabitq_audit_expansions;
  lhs.rabitq_audit_candidates += rhs.rabitq_audit_candidates;
  lhs.rabitq_safe_skips += rhs.rabitq_safe_skips;
  lhs.rabitq_exact_fallbacks += rhs.rabitq_exact_fallbacks;
  lhs.rabitq_prefetch_issued += rhs.rabitq_prefetch_issued;
  lhs.rabitq_prefetch_hits += rhs.rabitq_prefetch_hits;
  lhs.rabitq_prefetch_misses += rhs.rabitq_prefetch_misses;
  lhs.rabitq_prefetch_disabled_queries += rhs.rabitq_prefetch_disabled_queries;
  lhs.visited_nodes += rhs.visited_nodes;
  lhs.visited_neighborlists += rhs.visited_neighborlists;
  lhs.remote_allocations += rhs.remote_allocations;
  lhs.overflow_prunes += rhs.overflow_prunes;
  lhs.overflow_prune_candidates += rhs.overflow_prune_candidates;
  lhs.overflow_prune_max_candidates =
    std::max(lhs.overflow_prune_max_candidates, rhs.overflow_prune_max_candidates);
  lhs.overflow_prune_pair_checks_upper_bound += rhs.overflow_prune_pair_checks_upper_bound;
  lhs.overflow_prune_global_load_bytes_upper_bound += rhs.overflow_prune_global_load_bytes_upper_bound;
  lhs.overflow_prune_kernel_blocks += rhs.overflow_prune_kernel_blocks;
  lhs.overflow_prune_kernel_threads += rhs.overflow_prune_kernel_threads;
  lhs.overflow_prune_max_kernel_threads =
    std::max(lhs.overflow_prune_max_kernel_threads, rhs.overflow_prune_max_kernel_threads);
  lhs.query_rdma_to_staging_bytes += rhs.query_rdma_to_staging_bytes;
  lhs.query_host_staging_fallback_bytes += rhs.query_host_staging_fallback_bytes;
  lhs.storage_owner_handoff_requests += rhs.storage_owner_handoff_requests;
  lhs.storage_owner_handoff_successes += rhs.storage_owner_handoff_successes;
  lhs.storage_owner_handoff_queue_full += rhs.storage_owner_handoff_queue_full;
  lhs.storage_owner_handoff_timeouts += rhs.storage_owner_handoff_timeouts;
  lhs.storage_owner_handoff_overloaded += rhs.storage_owner_handoff_overloaded;
  lhs.storage_owner_handoff_failed += rhs.storage_owner_handoff_failed;
  lhs.storage_owner_handoff_request_bytes += rhs.storage_owner_handoff_request_bytes;
  lhs.storage_owner_handoff_response_bytes += rhs.storage_owner_handoff_response_bytes;
  lhs.storage_owner_handoff_remote_handler_ns += rhs.storage_owner_handoff_remote_handler_ns;
  lhs.storage_owner_handoff_remote_expanded_nodes += rhs.storage_owner_handoff_remote_expanded_nodes;
  lhs.storage_owner_handoff_remote_snapshot_reads += rhs.storage_owner_handoff_remote_snapshot_reads;
  lhs.storage_owner_handoff_remote_neighbor_reads += rhs.storage_owner_handoff_remote_neighbor_reads;
  lhs.storage_owner_handoff_response_beam_entries += rhs.storage_owner_handoff_response_beam_entries;
  lhs.storage_owner_handoff_response_visited_entries += rhs.storage_owner_handoff_response_visited_entries;
  lhs.storage_owner_handoff_response_visited_truncated += rhs.storage_owner_handoff_response_visited_truncated;
  lhs.storage_owner_qdi_requests += rhs.storage_owner_qdi_requests;
  lhs.storage_owner_qdi_successes += rhs.storage_owner_qdi_successes;
  lhs.storage_owner_qdi_queue_full += rhs.storage_owner_qdi_queue_full;
  lhs.storage_owner_qdi_timeouts += rhs.storage_owner_qdi_timeouts;
  lhs.storage_owner_qdi_overloaded += rhs.storage_owner_qdi_overloaded;
  lhs.storage_owner_qdi_failed += rhs.storage_owner_qdi_failed;
  lhs.storage_owner_qdi_request_bytes += rhs.storage_owner_qdi_request_bytes;
  lhs.storage_owner_qdi_response_bytes += rhs.storage_owner_qdi_response_bytes;
  lhs.storage_owner_qdi_remote_handler_ns += rhs.storage_owner_qdi_remote_handler_ns;
  lhs.storage_owner_qdi_remote_expanded_nodes += rhs.storage_owner_qdi_remote_expanded_nodes;
  lhs.storage_owner_qdi_remote_approx_scores += rhs.storage_owner_qdi_remote_approx_scores;
  lhs.storage_owner_qdi_remote_exact_reads += rhs.storage_owner_qdi_remote_exact_reads;
  lhs.storage_owner_qdi_remote_neighbor_reads += rhs.storage_owner_qdi_remote_neighbor_reads;
  lhs.storage_owner_qdi_response_candidates += rhs.storage_owner_qdi_response_candidates;
}


inline ThreadCounterDelta diff_thread_counters(const statistics::ThreadStatistics& end,
                                               const statistics::ThreadStatistics& start,
                                               const Operation operation) {
  ThreadCounterDelta out{};

  if (operation == Operation::query) {
    out.rdma_read_bytes = end.query_rdma_reads_in_bytes - start.query_rdma_reads_in_bytes;
    out.rdma_write_bytes = end.query_rdma_writes_in_bytes - start.query_rdma_writes_in_bytes;
    out.rdma_read_ops = end.query_rdma_read_ops - start.query_rdma_read_ops;
    out.rdma_write_ops = end.query_rdma_write_ops - start.query_rdma_write_ops;
    out.neighbor_rdma_bytes = end.query_neighbor_rdma_reads_in_bytes - start.query_neighbor_rdma_reads_in_bytes;
    out.vector_rdma_bytes = end.query_vector_rdma_reads_in_bytes - start.query_vector_rdma_reads_in_bytes;
    out.neighbor_rdma_read_ops = end.query_neighbor_rdma_read_ops - start.query_neighbor_rdma_read_ops;
    out.vector_rdma_read_ops = end.query_vector_rdma_read_ops - start.query_vector_rdma_read_ops;
    out.vector_rdma_batch_calls =
      end.query_vector_rdma_batch_calls - start.query_vector_rdma_batch_calls;
    out.vector_rdma_cqes = end.query_vector_rdma_cqes - start.query_vector_rdma_cqes;
    out.h2d_bytes = end.query_h2d_bytes - start.query_h2d_bytes;
    out.d2h_bytes = end.query_d2h_bytes - start.query_d2h_bytes;
    out.exact_reranks = end.query_exact_reranks - start.query_exact_reranks;
    out.rabitq_l0_candidates = end.query_rabitq_l0_candidates - start.query_rabitq_l0_candidates;
    out.rabitq_cache_misses = end.query_rabitq_cache_misses - start.query_rabitq_cache_misses;
    out.rabitq_l1_candidates = end.query_rabitq_l1_candidates - start.query_rabitq_l1_candidates;
    out.rabitq_l2_candidates = end.query_rabitq_l2_candidates - start.query_rabitq_l2_candidates;
    out.rabitq_forced_widen = end.query_rabitq_forced_widen - start.query_rabitq_forced_widen;
    out.rabitq_audit_expansions =
      end.query_rabitq_audit_expansions - start.query_rabitq_audit_expansions;
    out.rabitq_audit_candidates =
      end.query_rabitq_audit_candidates - start.query_rabitq_audit_candidates;
    out.rabitq_safe_skips = end.query_rabitq_safe_skips - start.query_rabitq_safe_skips;
    out.rabitq_exact_fallbacks =
      end.query_rabitq_exact_fallbacks - start.query_rabitq_exact_fallbacks;
    out.rabitq_prefetch_issued =
      end.query_rabitq_prefetch_issued - start.query_rabitq_prefetch_issued;
    out.rabitq_prefetch_hits =
      end.query_rabitq_prefetch_hits - start.query_rabitq_prefetch_hits;
    out.rabitq_prefetch_misses =
      end.query_rabitq_prefetch_misses - start.query_rabitq_prefetch_misses;
    out.rabitq_prefetch_disabled_queries =
      end.query_rabitq_prefetch_disabled_queries -
      start.query_rabitq_prefetch_disabled_queries;
    out.visited_nodes =
      (end.visited_nodes - start.visited_nodes) + (end.visited_nodes_l0 - start.visited_nodes_l0);
    out.visited_neighborlists = end.visited_neighborlists - start.visited_neighborlists;
    out.query_rdma_to_staging_bytes = end.query_rdma_to_staging_bytes - start.query_rdma_to_staging_bytes;
    out.query_host_staging_fallback_bytes = end.query_host_staging_fallback_bytes - start.query_host_staging_fallback_bytes;
    return out;
  }

  out.rdma_read_bytes = end.build_rdma_reads_in_bytes - start.build_rdma_reads_in_bytes;
  out.rdma_write_bytes = end.build_rdma_writes_in_bytes - start.build_rdma_writes_in_bytes;
  out.rdma_read_ops = end.build_rdma_read_ops - start.build_rdma_read_ops;
  out.rdma_write_ops = end.build_rdma_write_ops - start.build_rdma_write_ops;
  out.neighbor_rdma_bytes = end.build_neighbor_rdma_reads_in_bytes - start.build_neighbor_rdma_reads_in_bytes;
  out.vector_rdma_bytes = end.build_vector_rdma_reads_in_bytes - start.build_vector_rdma_reads_in_bytes;
  out.neighbor_rdma_read_ops = end.build_neighbor_rdma_read_ops - start.build_neighbor_rdma_read_ops;
  out.vector_rdma_read_ops = end.build_vector_rdma_read_ops - start.build_vector_rdma_read_ops;
  out.h2d_bytes = end.build_h2d_bytes - start.build_h2d_bytes;
  out.d2h_bytes = end.build_d2h_bytes - start.build_d2h_bytes;
  out.l2_kernels = end.build_l2_kernels - start.build_l2_kernels;
  out.prune_kernels = end.build_prune_kernels - start.build_prune_kernels;
  out.remote_allocations = end.remote_allocations - start.remote_allocations;
  out.overflow_prunes = end.build_overflow_prunes - start.build_overflow_prunes;
  out.overflow_prune_candidates =
    end.build_overflow_prune_candidates - start.build_overflow_prune_candidates;
  out.overflow_prune_max_candidates =
    end.build_overflow_prune_max_candidates > start.build_overflow_prune_max_candidates
      ? end.build_overflow_prune_max_candidates
      : 0;
  out.overflow_prune_pair_checks_upper_bound =
    end.build_overflow_prune_pair_checks_upper_bound - start.build_overflow_prune_pair_checks_upper_bound;
  out.overflow_prune_global_load_bytes_upper_bound =
    end.build_overflow_prune_global_load_bytes_upper_bound - start.build_overflow_prune_global_load_bytes_upper_bound;
  out.overflow_prune_kernel_blocks =
    end.build_overflow_prune_kernel_blocks - start.build_overflow_prune_kernel_blocks;
  out.overflow_prune_kernel_threads =
    end.build_overflow_prune_kernel_threads - start.build_overflow_prune_kernel_threads;
  out.overflow_prune_max_kernel_threads =
    end.build_overflow_prune_max_kernel_threads > start.build_overflow_prune_max_kernel_threads
      ? end.build_overflow_prune_max_kernel_threads
      : 0;
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
  u64 overflow_prune_max_candidates{};
  u64 overflow_prune_max_kernel_threads{};
  ThreadCounterDelta extra_counters{};
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

  void add_counters(const ThreadCounterDelta& delta) {
    add_counter_delta(extra_counters, delta);
  }

  ThreadCounterDelta counters() const {
    ThreadCounterDelta out = diff_thread_counters(end_counters, start_counters, operation);
    add_counter_delta(out, extra_counters);
    return out;
  }
};

}  // namespace service::breakdown
