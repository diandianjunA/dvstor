#pragma once

#include <atomic>
#include <cstdint>

namespace gpu_search {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;

struct QueryDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 query_device_address{};
  u64 result_device_address{};
  u32 query_slot{};
  u32 result_capacity{};
  u16 dim{};
  u16 k{};
  u8 query_dtype{};
  u8 flags{};
  u16 reserved{};
};

struct CompletionDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 gpu_cycles{};
  u64 prepare_cycles{};
  u64 graph_cycles{};
  u64 score_cycles{};
  u64 beam_cycles{};
  u64 exact_cycles{};
  u32 query_slot{};
  u32 result_count{};
  i32 status{};
  u32 remote_pages{};
  u32 exact_vectors{};
  u32 cache_hits{};
  u32 exact_cache_hits{};
};

struct TelemetrySnapshot {
  u64 queries_submitted{};
  u64 queries_completed{};
  u64 batches{};
  u64 batch_queries{};
  u64 submission_wait_ns{};
  u64 completion_wait_ns{};
  u64 gpu_active_ns{};
  u64 rdma_read_ops{};
  u64 rdma_read_bytes{};
  u64 rdma_merged_requests{};
  u64 direct_path_failures{};
  u64 graph_page_requests{};
  u64 graph_page_cache_hits{};
  u64 exact_vector_reads{};
  u64 exact_vector_cache_hits{};
  u64 delta_queries{};
  u64 mutations_published{};
  u64 delta_compactions{};
  u64 base_entries_merged{};
  u64 delta_live_entries{};
  u64 visibility_ns_total{};
  u64 visibility_ns_max{};
};

class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();

  std::atomic<u64> queries_submitted{0};
  std::atomic<u64> queries_completed{0};
  std::atomic<u64> batches{0};
  std::atomic<u64> batch_queries{0};
  std::atomic<u64> submission_wait_ns{0};
  std::atomic<u64> completion_wait_ns{0};
  std::atomic<u64> gpu_active_ns{0};
  std::atomic<u64> rdma_read_ops{0};
  std::atomic<u64> rdma_read_bytes{0};
  std::atomic<u64> rdma_merged_requests{0};
  std::atomic<u64> direct_path_failures{0};
  std::atomic<u64> graph_page_requests{0};
  std::atomic<u64> graph_page_cache_hits{0};
  std::atomic<u64> exact_vector_reads{0};
  std::atomic<u64> exact_vector_cache_hits{0};
  std::atomic<u64> delta_queries{0};
  std::atomic<u64> mutations_published{0};
  std::atomic<u64> delta_compactions{0};
  std::atomic<u64> base_entries_merged{0};
  std::atomic<u64> delta_live_entries{0};
  std::atomic<u64> visibility_ns_total{0};
  std::atomic<u64> visibility_ns_max{0};
};

}  // namespace gpu_search
