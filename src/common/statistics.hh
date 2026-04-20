#pragma once

#include <iostream>
#include <library/types.hh>
#include <ostream>

#include "nlohmann/json.hh"

namespace statistics {

class Statistics {
public:
  using json = nlohmann::json;

  void add_timings(const json& timings) { stats_["timings"] = timings; }

  template <typename T>
  void add_meta_stat(const str&& key, T v) {
    stats_["meta"][key] = v;
  }

  template <typename T>
  void add_meta_stats(T pair) {
    add_meta_stat(pair.first, pair.second);
  }

  template <typename T, typename... Args>
  void add_meta_stats(T pair, Args... args) {
    add_meta_stat(pair.first, pair.second);
    add_meta_stats(args...);
  }

  void output_all(const json& timings) {
    add_timings(timings);

    std::cerr << std::endl << "statistics:" << std::endl;
    std::cout << *this << std::endl;
  }

  template <typename T>
  void add_static_stat(const str& key, T v) {
    stats_[key] = v;
  }

  template <typename T>
  void add_nested_static_stat(const str& group, const str&& key, T v) {
    stats_[group][key] = v;
  }

  template <typename T>
  void add_nested_static_stat(const str& g1, const str& g2, const str&& key, T v) {
    stats_[g1][g2][key] = v;
  }

  friend std::ostream& operator<<(std::ostream& os, const Statistics& s) { return os << s.stats_.dump(2); }

private:
  json stats_;
};

struct CNStatistics {
  size_t build_distcomps{};
  size_t build_rdma_reads{};
  size_t build_rdma_writes{};
  size_t build_neighbor_rdma_reads{};
  size_t build_vector_rdma_reads{};
  size_t build_rabitq_rdma_reads{};
  size_t build_h2d_bytes{};
  size_t build_d2h_bytes{};
  size_t build_l2_kernels{};
  size_t build_prune_kernels{};
  size_t build_overflow_prunes{};
  size_t remote_allocations{};
  size_t total_allocation_size{};
  u32 max_level{};

  size_t query_distcomps{};
  size_t query_rdma_reads{};
  size_t query_rdma_writes{};
  size_t query_neighbor_rdma_reads{};
  size_t query_vector_rdma_reads{};
  size_t query_rabitq_rdma_reads{};
  size_t query_h2d_bytes{};
  size_t query_d2h_bytes{};
  size_t query_rabitq_kernels{};
  size_t query_exact_reranks{};

  size_t query_rdma_to_staging_bytes{};
  size_t query_host_staging_fallback_bytes{};

  size_t query_visited_nodes{};
  size_t query_visited_nodes_l0{};
  size_t query_visited_neighborlists{};

  size_t processed_queries{};
  size_t processed_inserts{};
  u64 query_queue_wait_ns{};
  u64 insert_queue_wait_ns{};
  size_t local_allocation_size{};

  f64 rolling_recall{};
  timespec build_time{};
  timespec query_time{};

  void combine(const CNStatistics& other) {
    total_allocation_size += other.total_allocation_size;
    remote_allocations += other.remote_allocations;
    rolling_recall += other.rolling_recall;
    max_level = std::max(max_level, other.max_level);

    build_distcomps += other.build_distcomps;
    build_rdma_reads += other.build_rdma_reads;
    build_rdma_writes += other.build_rdma_writes;
    build_neighbor_rdma_reads += other.build_neighbor_rdma_reads;
    build_vector_rdma_reads += other.build_vector_rdma_reads;
    build_rabitq_rdma_reads += other.build_rabitq_rdma_reads;
    build_h2d_bytes += other.build_h2d_bytes;
    build_d2h_bytes += other.build_d2h_bytes;
    build_l2_kernels += other.build_l2_kernels;
    build_prune_kernels += other.build_prune_kernels;
    build_overflow_prunes += other.build_overflow_prunes;

    processed_queries += other.processed_queries;
    processed_inserts += other.processed_inserts;
    query_queue_wait_ns += other.query_queue_wait_ns;
    insert_queue_wait_ns += other.insert_queue_wait_ns;
    local_allocation_size += other.local_allocation_size;

    query_distcomps += other.query_distcomps;
    query_rdma_reads += other.query_rdma_reads;
    query_rdma_writes += other.query_rdma_writes;
    query_neighbor_rdma_reads += other.query_neighbor_rdma_reads;
    query_vector_rdma_reads += other.query_vector_rdma_reads;
    query_rabitq_rdma_reads += other.query_rabitq_rdma_reads;
    query_h2d_bytes += other.query_h2d_bytes;
    query_d2h_bytes += other.query_d2h_bytes;
    query_rabitq_kernels += other.query_rabitq_kernels;
    query_exact_reranks += other.query_exact_reranks;
    query_rdma_to_staging_bytes += other.query_rdma_to_staging_bytes;
    query_host_staging_fallback_bytes += other.query_host_staging_fallback_bytes;
    query_visited_nodes += other.query_visited_nodes;
    query_visited_nodes_l0 += other.query_visited_nodes_l0;
    query_visited_neighborlists += other.query_visited_neighborlists;
  }

  void convert(Statistics& statistics) const {
    const str build_group = "build";
    const str query_group = "queries";

    statistics.add_nested_static_stat(build_group, "dist_comps", build_distcomps);
    statistics.add_nested_static_stat(build_group, "rdma_reads_in_bytes", build_rdma_reads);
    statistics.add_nested_static_stat(build_group, "rdma_writes_in_bytes", build_rdma_writes);
    statistics.add_nested_static_stat(build_group, "neighbor_rdma_reads_in_bytes", build_neighbor_rdma_reads);
    statistics.add_nested_static_stat(build_group, "vector_rdma_reads_in_bytes", build_vector_rdma_reads);
    statistics.add_nested_static_stat(build_group, "rabitq_rdma_reads_in_bytes", build_rabitq_rdma_reads);
    statistics.add_nested_static_stat(build_group, "h2d_in_bytes", build_h2d_bytes);
    statistics.add_nested_static_stat(build_group, "d2h_in_bytes", build_d2h_bytes);
    statistics.add_nested_static_stat(build_group, "l2_kernels", build_l2_kernels);
    statistics.add_nested_static_stat(build_group, "prune_kernels", build_prune_kernels);
    statistics.add_nested_static_stat(build_group, "overflow_prunes", build_overflow_prunes);
    statistics.add_nested_static_stat(build_group, "remote_allocations", remote_allocations);
    statistics.add_nested_static_stat(build_group, "index_size", total_allocation_size);
    statistics.add_nested_static_stat(build_group, "max_level", max_level);
    statistics.add_nested_static_stat(build_group, "processed", processed_inserts);
    statistics.add_nested_static_stat(build_group, "queue_wait_ns", insert_queue_wait_ns);

    statistics.add_nested_static_stat(query_group, "dist_comps", query_distcomps);
    statistics.add_nested_static_stat(query_group, "rdma_reads_in_bytes", query_rdma_reads);
    statistics.add_nested_static_stat(query_group, "rdma_writes_in_bytes", query_rdma_writes);
    statistics.add_nested_static_stat(query_group, "neighbor_rdma_reads_in_bytes", query_neighbor_rdma_reads);
    statistics.add_nested_static_stat(query_group, "vector_rdma_reads_in_bytes", query_vector_rdma_reads);
    statistics.add_nested_static_stat(query_group, "rabitq_rdma_reads_in_bytes", query_rabitq_rdma_reads);
    statistics.add_nested_static_stat(query_group, "h2d_in_bytes", query_h2d_bytes);
    statistics.add_nested_static_stat(query_group, "d2h_in_bytes", query_d2h_bytes);
    statistics.add_nested_static_stat(query_group, "rabitq_kernels", query_rabitq_kernels);
    statistics.add_nested_static_stat(query_group, "exact_reranks", query_exact_reranks);
    statistics.add_nested_static_stat(query_group, "recall", rolling_recall);
    statistics.add_nested_static_stat(query_group, "visited_nodes", query_visited_nodes);
    statistics.add_nested_static_stat(query_group, "visited_nodes_l0", query_visited_nodes_l0);
    statistics.add_nested_static_stat(query_group, "visited_neighborlists", query_visited_neighborlists);
    statistics.add_nested_static_stat(query_group, "processed", processed_queries);
    statistics.add_nested_static_stat(query_group, "queue_wait_ns", query_queue_wait_ns);

    statistics.add_static_stat("actual_total_local_buffer_size", local_allocation_size);
  }
};

struct ThreadStatistics {
  size_t distcomps{0};
  size_t query_distcomps{0};
  size_t build_distcomps{0};
  size_t rdma_reads_in_bytes{0};
  size_t rdma_writes_in_bytes{0};
  size_t query_rdma_reads_in_bytes{0};
  size_t build_rdma_reads_in_bytes{0};
  size_t query_rdma_writes_in_bytes{0};
  size_t build_rdma_writes_in_bytes{0};
  size_t query_neighbor_rdma_reads_in_bytes{0};
  size_t query_vector_rdma_reads_in_bytes{0};
  size_t query_rabitq_rdma_reads_in_bytes{0};
  size_t build_neighbor_rdma_reads_in_bytes{0};
  size_t build_vector_rdma_reads_in_bytes{0};
  size_t build_rabitq_rdma_reads_in_bytes{0};
  size_t query_h2d_bytes{0};
  size_t query_d2h_bytes{0};
  size_t build_h2d_bytes{0};
  size_t build_d2h_bytes{0};
  size_t query_rabitq_kernels{0};
  size_t query_exact_reranks{0};
  size_t build_l2_kernels{0};
  size_t build_prune_kernels{0};
  size_t build_overflow_prunes{0};
  size_t processed{0};
  size_t processed_queries{0};
  size_t processed_inserts{0};
  size_t remote_allocations{0};
  size_t allocation_size{0};
  size_t visited_nodes{0};
  size_t visited_nodes_l0{0};
  size_t visited_neighborlists{0};
  u32 max_level{0};
  u64 query_queue_wait_ns{0};
  u64 insert_queue_wait_ns{0};

  size_t cache_hits{0};
  size_t cache_misses{0};

  size_t query_rdma_to_staging_bytes{0};
  size_t query_host_staging_fallback_bytes{0};
  size_t neighbor_cache_hits{0};
  size_t neighbor_cache_misses{0};
  size_t gpu_rabitq_cache_hits{0};
  size_t gpu_rabitq_cache_misses{0};
  size_t gpu_rabitq_cache_fills{0};
  size_t gpu_rabitq_cache_fill_bytes{0};
  size_t gpu_rabitq_cache_duplicate_fills{0};
  size_t gpu_rabitq_cache_fallback_batches{0};

  void inc_visited_nodes(u32 level) {
    if (level > 0) {
      ++visited_nodes;
    } else {
      ++visited_nodes_l0;
    }
  }

  f64 cache_hit_rate() const {
    return cache_hits + cache_misses == 0 ? 0
                                          : static_cast<f64>(cache_hits) / static_cast<f64>(cache_hits + cache_misses);
  }
};

}  // namespace statistics
