#pragma once

#include <chrono>
#include <memory>

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
  u64 vector_rdma_active_nodes{};
  u64 vector_rdma_active_qps{};
  u64 vector_rdma_chain_wrs{};
  u64 vector_rdma_max_chain_wrs{};
  u64 vector_rdma_qp_high_water_wrs{};
  u64 vector_rdma_credit_waits{};
  u64 vector_rdma_credit_wait_ns{};
  u64 vector_rdma_completion_token_waits{};
  u64 vector_rdma_post_send_calls{};
  u64 vector_rdma_post_send_retries{};
  u64 vector_rdma_post_send_errors{};
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
  u64 credit_rounds{};
  u64 credit_expansions_issued{};
  u64 credit_precommit_expansions{};
  u64 credit_postcommit_expansions{};
  u64 credit_grow_events{};
  u64 credit_shrink_events{};
  u64 credit_credit_stalls{};
  u64 credit_no_progress_rounds{};
  u64 credit_underfilled_rounds{};
  u64 credit_overfilled_rounds{};
  u64 credit_cost_guard_events{};
  u64 credit_cost_growth_blocked{};
  u64 credit_cost_baseline_samples{};
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
  u64 storage_owner_anchor_hints{};
  u64 storage_owner_anchor_valid_hints{};
  u64 storage_owner_anchor_expansions{};
  u64 storage_owner_anchor_remote_expansions{};
  u64 storage_owner_anchor_fallbacks{};
  u64 storage_owner_anchor_audits{};
  u64 storage_owner_anchor_audit_failures{};
};

struct StorageOwnerAnchorCounters {
  u64 hints{};
  u64 valid_hints{};
  u64 expansions{};
  u64 remote_expansions{};
  u64 fallbacks{};
  u64 audits{};
  u64 audit_failures{};
};


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
      end.vector_rdma_batch_calls - start.vector_rdma_batch_calls;
    out.vector_rdma_cqes = end.vector_rdma_chunks - start.vector_rdma_chunks;
    out.vector_rdma_active_nodes =
      end.vector_rdma_active_nodes - start.vector_rdma_active_nodes;
    out.vector_rdma_active_qps =
      end.vector_rdma_active_qps - start.vector_rdma_active_qps;
    out.vector_rdma_chain_wrs = end.vector_rdma_chain_wrs - start.vector_rdma_chain_wrs;
    out.vector_rdma_max_chain_wrs = end.vector_rdma_max_chain_wrs;
    out.vector_rdma_qp_high_water_wrs = end.vector_rdma_qp_high_water_wrs;
    out.vector_rdma_credit_waits =
      end.vector_rdma_credit_waits - start.vector_rdma_credit_waits;
    out.vector_rdma_credit_wait_ns =
      end.vector_rdma_credit_wait_ns - start.vector_rdma_credit_wait_ns;
    out.vector_rdma_completion_token_waits =
      end.vector_rdma_completion_token_waits - start.vector_rdma_completion_token_waits;
    out.vector_rdma_post_send_calls =
      end.vector_rdma_post_send_calls - start.vector_rdma_post_send_calls;
    out.vector_rdma_post_send_retries =
      end.vector_rdma_post_send_retries - start.vector_rdma_post_send_retries;
    out.vector_rdma_post_send_errors =
      end.vector_rdma_post_send_errors - start.vector_rdma_post_send_errors;
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
    out.credit_rounds = end.query_credit_rounds - start.query_credit_rounds;
    out.credit_expansions_issued =
      end.query_credit_expansions_issued - start.query_credit_expansions_issued;
    out.credit_precommit_expansions =
      end.query_credit_precommit_expansions - start.query_credit_precommit_expansions;
    out.credit_postcommit_expansions =
      end.query_credit_postcommit_expansions - start.query_credit_postcommit_expansions;
    out.credit_grow_events = end.query_credit_grow_events - start.query_credit_grow_events;
    out.credit_shrink_events = end.query_credit_shrink_events - start.query_credit_shrink_events;
    out.credit_credit_stalls =
      end.query_credit_credit_stalls - start.query_credit_credit_stalls;
    out.credit_no_progress_rounds =
      end.query_credit_no_progress_rounds - start.query_credit_no_progress_rounds;
    out.credit_underfilled_rounds =
      end.query_credit_underfilled_rounds - start.query_credit_underfilled_rounds;
    out.credit_overfilled_rounds =
      end.query_credit_overfilled_rounds - start.query_credit_overfilled_rounds;
    out.credit_cost_guard_events =
      end.query_credit_cost_guard_events - start.query_credit_cost_guard_events;
    out.credit_cost_growth_blocked =
      end.query_credit_cost_growth_blocked - start.query_credit_cost_growth_blocked;
    out.credit_cost_baseline_samples =
      end.query_credit_cost_baseline_samples - start.query_credit_cost_baseline_samples;
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
  out.vector_rdma_batch_calls =
    end.vector_rdma_batch_calls - start.vector_rdma_batch_calls;
  out.vector_rdma_cqes = end.vector_rdma_chunks - start.vector_rdma_chunks;
  out.vector_rdma_active_nodes =
    end.vector_rdma_active_nodes - start.vector_rdma_active_nodes;
  out.vector_rdma_active_qps =
    end.vector_rdma_active_qps - start.vector_rdma_active_qps;
  out.vector_rdma_chain_wrs = end.vector_rdma_chain_wrs - start.vector_rdma_chain_wrs;
  out.vector_rdma_max_chain_wrs = end.vector_rdma_max_chain_wrs;
  out.vector_rdma_qp_high_water_wrs = end.vector_rdma_qp_high_water_wrs;
  out.vector_rdma_credit_waits =
    end.vector_rdma_credit_waits - start.vector_rdma_credit_waits;
  out.vector_rdma_credit_wait_ns =
    end.vector_rdma_credit_wait_ns - start.vector_rdma_credit_wait_ns;
  out.vector_rdma_completion_token_waits =
    end.vector_rdma_completion_token_waits - start.vector_rdma_completion_token_waits;
  out.vector_rdma_post_send_calls =
    end.vector_rdma_post_send_calls - start.vector_rdma_post_send_calls;
  out.vector_rdma_post_send_retries =
    end.vector_rdma_post_send_retries - start.vector_rdma_post_send_retries;
  out.vector_rdma_post_send_errors =
    end.vector_rdma_post_send_errors - start.vector_rdma_post_send_errors;
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
  explicit Sample(Operation op, bool collect_fine_grained = true)
      : operation(op), collect_fine_grained_breakdown(collect_fine_grained) {}

  Operation operation;
  bool collect_fine_grained_breakdown{};
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
  // RDMA wait is a query-level software occupancy proxy: it measures time
  // spent awaiting RDMA completions, not physical NIC line-rate utilization.
  u64 rdma_wait_ns{};
  // Measured with CUDA events around distance kernels; excludes CPU launch,
  // stream queueing, and D2H copies.
  u64 gpu_kernel_ns{};
  u64 lock_attempts{};
  u64 lock_retries{};
  u64 cas_failures{};
  u64 overflow_prune_max_candidates{};
  u64 overflow_prune_max_kernel_threads{};
  std::shared_ptr<StorageOwnerAnchorCounters> storage_owner_anchor;
  bool started_flag{};
  bool finished_flag{};
  bool device_utilization_observed{};

  void mark_started(const Clock::time_point dequeued,
                    const Clock::time_point started,
                    const statistics::ThreadStatistics& counters) {
    dequeued_at = dequeued;
    started_at = started;
    if (collect_fine_grained_breakdown) {
      start_counters = counters;
    }
    started_flag = true;
    queue_wait_ns =
      static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(dequeued_at - enqueued_at).count());
  }

  void mark_finished(const Clock::time_point finished, const statistics::ThreadStatistics& counters) {
    finished_at = finished;
    if (collect_fine_grained_breakdown) {
      end_counters = counters;
    }
    finished_flag = true;
    service_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - started_at).count());
    end_to_end_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - enqueued_at).count());
  }

  [[nodiscard]] bool collects_breakdown() const { return collect_fine_grained_breakdown; }

  void add_subcategory(const Subcategory subcategory, const u64 ns) {
    if (!collect_fine_grained_breakdown) return;
    subcategory_ns[static_cast<size_t>(subcategory)] += ns;
    category_ns[static_cast<size_t>(parent_category(subcategory))] += ns;
    if (operation == Operation::query && parent_category(subcategory) == Category::rdma) {
      rdma_wait_ns += ns;
    }
  }

  void add_gpu_kernel_time(const u64 ns) {
    if (!collect_fine_grained_breakdown) return;
    if (operation == Operation::query) gpu_kernel_ns += ns;
  }

  void set_device_utilization_observed() {
    if (collect_fine_grained_breakdown) device_utilization_observed = true;
  }

  ThreadCounterDelta counters() const {
    if (!collect_fine_grained_breakdown) return {};
    ThreadCounterDelta out = diff_thread_counters(end_counters, start_counters, operation);
    if (storage_owner_anchor != nullptr) {
      out.storage_owner_anchor_hints = storage_owner_anchor->hints;
      out.storage_owner_anchor_valid_hints = storage_owner_anchor->valid_hints;
      out.storage_owner_anchor_expansions = storage_owner_anchor->expansions;
      out.storage_owner_anchor_remote_expansions = storage_owner_anchor->remote_expansions;
      out.storage_owner_anchor_fallbacks = storage_owner_anchor->fallbacks;
      out.storage_owner_anchor_audits = storage_owner_anchor->audits;
      out.storage_owner_anchor_audit_failures = storage_owner_anchor->audit_failures;
    }
    return out;
  }
};

}  // namespace service::breakdown
