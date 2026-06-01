#pragma once

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
  u64 rabitq_rdma_bytes{};
  u64 neighbor_rdma_read_ops{};
  u64 vector_rdma_read_ops{};
  u64 rabitq_rdma_read_ops{};
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
  u64 overflow_prune_candidates{};
  u64 overflow_prune_max_candidates{};
  u64 overflow_prune_pair_checks_upper_bound{};
  u64 overflow_prune_global_load_bytes_upper_bound{};
  u64 overflow_prune_kernel_blocks{};
  u64 overflow_prune_kernel_threads{};
  u64 overflow_prune_max_kernel_threads{};
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
  u64 gpu_rabitq_cache_gather_batches{};
  u64 gpu_rabitq_cache_gather_bytes{};
  u64 gpu_rabitq_cache_loading_fallbacks{};
  u64 gpu_rabitq_cache_evictions{};
  u64 gpu_rabitq_cache_duplicate_fills{};
  u64 gpu_rabitq_cache_fallback_batches{};
  u64 gpu_rabitq_cache_nursery_hits{};
  u64 gpu_rabitq_cache_hot_hits{};
  u64 gpu_rabitq_cache_promotions{};
  u64 gpu_rabitq_cache_nursery_overwrites{};
  u64 gpu_rabitq_cache_hot_tile_evictions{};
  u64 gpu_rabitq_cache_unique_tiles{};
  u64 gpu_rabitq_cache_tile_tasks{};
  u64 gpu_rabitq_cache_miss_ranges{};
  u64 gpu_rabitq_cache_resolve_ns{};
  u64 gpu_rabitq_cache_victim_ns{};
  u64 gpu_rabitq_cache_kernel_ns{};
  u64 gpu_rabitq_cache_rdma_ns{};
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
    out.rabitq_rdma_bytes = end.query_rabitq_rdma_reads_in_bytes - start.query_rabitq_rdma_reads_in_bytes;
    out.neighbor_rdma_read_ops = end.query_neighbor_rdma_read_ops - start.query_neighbor_rdma_read_ops;
    out.vector_rdma_read_ops = end.query_vector_rdma_read_ops - start.query_vector_rdma_read_ops;
    out.rabitq_rdma_read_ops = end.query_rabitq_rdma_read_ops - start.query_rabitq_rdma_read_ops;
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
    out.gpu_rabitq_cache_gather_batches =
      end.gpu_rabitq_cache_gather_batches - start.gpu_rabitq_cache_gather_batches;
    out.gpu_rabitq_cache_gather_bytes =
      end.gpu_rabitq_cache_gather_bytes - start.gpu_rabitq_cache_gather_bytes;
    out.gpu_rabitq_cache_loading_fallbacks =
      end.gpu_rabitq_cache_loading_fallbacks - start.gpu_rabitq_cache_loading_fallbacks;
    out.gpu_rabitq_cache_evictions =
      end.gpu_rabitq_cache_evictions - start.gpu_rabitq_cache_evictions;
    out.gpu_rabitq_cache_duplicate_fills =
      end.gpu_rabitq_cache_duplicate_fills - start.gpu_rabitq_cache_duplicate_fills;
    out.gpu_rabitq_cache_fallback_batches =
      end.gpu_rabitq_cache_fallback_batches - start.gpu_rabitq_cache_fallback_batches;
    out.gpu_rabitq_cache_nursery_hits = end.gpu_rabitq_cache_nursery_hits - start.gpu_rabitq_cache_nursery_hits;
    out.gpu_rabitq_cache_hot_hits = end.gpu_rabitq_cache_hot_hits - start.gpu_rabitq_cache_hot_hits;
    out.gpu_rabitq_cache_promotions = end.gpu_rabitq_cache_promotions - start.gpu_rabitq_cache_promotions;
    out.gpu_rabitq_cache_nursery_overwrites =
      end.gpu_rabitq_cache_nursery_overwrites - start.gpu_rabitq_cache_nursery_overwrites;
    out.gpu_rabitq_cache_hot_tile_evictions =
      end.gpu_rabitq_cache_hot_tile_evictions - start.gpu_rabitq_cache_hot_tile_evictions;
    out.gpu_rabitq_cache_unique_tiles = end.gpu_rabitq_cache_unique_tiles - start.gpu_rabitq_cache_unique_tiles;
    out.gpu_rabitq_cache_tile_tasks = end.gpu_rabitq_cache_tile_tasks - start.gpu_rabitq_cache_tile_tasks;
    out.gpu_rabitq_cache_miss_ranges = end.gpu_rabitq_cache_miss_ranges - start.gpu_rabitq_cache_miss_ranges;
    out.gpu_rabitq_cache_resolve_ns = end.gpu_rabitq_cache_resolve_ns - start.gpu_rabitq_cache_resolve_ns;
    out.gpu_rabitq_cache_victim_ns = end.gpu_rabitq_cache_victim_ns - start.gpu_rabitq_cache_victim_ns;
    out.gpu_rabitq_cache_kernel_ns = end.gpu_rabitq_cache_kernel_ns - start.gpu_rabitq_cache_kernel_ns;
    out.gpu_rabitq_cache_rdma_ns = end.gpu_rabitq_cache_rdma_ns - start.gpu_rabitq_cache_rdma_ns;
    return out;
  }

  out.rdma_read_bytes = end.build_rdma_reads_in_bytes - start.build_rdma_reads_in_bytes;
  out.rdma_write_bytes = end.build_rdma_writes_in_bytes - start.build_rdma_writes_in_bytes;
  out.rdma_read_ops = end.build_rdma_read_ops - start.build_rdma_read_ops;
  out.rdma_write_ops = end.build_rdma_write_ops - start.build_rdma_write_ops;
  out.neighbor_rdma_bytes = end.build_neighbor_rdma_reads_in_bytes - start.build_neighbor_rdma_reads_in_bytes;
  out.vector_rdma_bytes = end.build_vector_rdma_reads_in_bytes - start.build_vector_rdma_reads_in_bytes;
  out.rabitq_rdma_bytes = end.build_rabitq_rdma_reads_in_bytes - start.build_rabitq_rdma_reads_in_bytes;
  out.neighbor_rdma_read_ops = end.build_neighbor_rdma_read_ops - start.build_neighbor_rdma_read_ops;
  out.vector_rdma_read_ops = end.build_vector_rdma_read_ops - start.build_vector_rdma_read_ops;
  out.rabitq_rdma_read_ops = end.build_rabitq_rdma_read_ops - start.build_rabitq_rdma_read_ops;
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
  u64 overflow_prune_max_candidates{};
  u64 overflow_prune_max_kernel_threads{};
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

}  // namespace service::breakdown
