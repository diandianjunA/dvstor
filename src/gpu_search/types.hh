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
  u32 remote_batches{};
  u32 graph_rounds{};
  u32 exact_vectors{};
  u32 cache_hits{};
  u32 route_hits{};
  u32 exact_cache_hits{};
};

struct DeltaSupersedeUpdate {
  u32 slot{};
  u32 reserved{};
  u64 epoch{};
};

struct DeltaOverrideUpdate {
  u32 ordinal{};
  u32 reserved{};
  u64 epoch{};
};

struct DeltaDurableUpdate {
  u32 slot{};
  u32 reserved{};
  u64 epoch{};
};

struct ResidentPqEraseUpdate {
  u64 remote_node{};
  u32 slot{};
  u32 reserved{};
};

inline constexpr u32 kDeltaCommandReset = 1u;
inline constexpr u32 kDeltaCommandPromoteOverrides = 1u << 1;

struct DeltaPublishDescriptor {
  u64 command_id{};
  u32 first_slot{};
  u32 record_count{};
  u32 final_count{};
  u32 invalidation_count{};
  u32 superseded_count{};
  u32 override_count{};
  u32 durable_count{};
  u32 resident_pq_erase_count{};
  u32 flags{};
};

struct DeltaPublishCompletion {
  u64 command_id{};
  i32 status{};
  u32 final_count{};
};

struct TelemetrySnapshot {
  u64 gpu_memory_explicit_bytes{};
  u64 gpu_memory_base_pq_bytes{};
  u64 gpu_memory_resident_pq_bytes{};
  u64 gpu_memory_route_graph_bytes{};
  u64 gpu_memory_delta_reserved_bytes{};
  u64 gpu_memory_graph_cache_bytes{};
  u64 gpu_memory_exact_cache_bytes{};
  u64 queries_submitted{};
  u64 queries_completed{};
  u64 batches{};
  u64 batch_queries{};
  u64 submission_wait_ns{};
  u64 completion_wait_ns{};
  u64 gpu_active_ns{};
  u64 gpu_prepare_ns{};
  u64 gpu_graph_ns{};
  u64 gpu_score_ns{};
  u64 gpu_beam_ns{};
  u64 gpu_exact_ns{};
  u64 rdma_read_ops{};
  u64 rdma_read_bytes{};
  u64 rdma_merged_requests{};
  u64 direct_path_failures{};
  u64 graph_page_requests{};
  u64 graph_dependency_rounds{};
  u64 graph_page_cache_hits{};
  u64 graph_route_hits{};
  u64 graph_route_refreshes{};
  u64 graph_cache_invalidations{};
  u64 exact_vector_reads{};
  u64 exact_vector_cache_hits{};
  u64 delta_queries{};
  u64 mutations_published{};
  u64 delta_publications{};
  u64 delta_reclaim_batches{};
  u64 delta_entries_retired{};
  u64 storage_reclaim_ack_writes{};
  u64 storage_reclaim_ack_sequence{};
  u64 delta_live_entries{};
  u64 delta_physical_entries{};
  u64 delta_mutable_entries{};
  u64 delta_durable_entries{};
  u64 resident_pq_capacity{};
  u64 resident_pq_entries{};
  u64 resident_pq_peak_entries{};
  u64 resident_pq_reclaimed{};
  u64 mutation_capacity_rejections{};
  u64 mutation_capacity_reserved{};
  u64 mutation_capacity_reserved_max{};
  u64 visibility_ns_total{};
  u64 visibility_ns_max{};
  u64 publication_queue_ns_total{};
  u64 publication_prepare_ns_total{};
  u64 publication_command_ns_total{};
};

class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();

  std::atomic<u64> gpu_memory_explicit_bytes{0};
  std::atomic<u64> gpu_memory_base_pq_bytes{0};
  std::atomic<u64> gpu_memory_resident_pq_bytes{0};
  std::atomic<u64> gpu_memory_route_graph_bytes{0};
  std::atomic<u64> gpu_memory_delta_reserved_bytes{0};
  std::atomic<u64> gpu_memory_graph_cache_bytes{0};
  std::atomic<u64> gpu_memory_exact_cache_bytes{0};
  std::atomic<u64> queries_submitted{0};
  std::atomic<u64> queries_completed{0};
  std::atomic<u64> batches{0};
  std::atomic<u64> batch_queries{0};
  std::atomic<u64> submission_wait_ns{0};
  std::atomic<u64> completion_wait_ns{0};
  std::atomic<u64> gpu_active_ns{0};
  std::atomic<u64> gpu_prepare_ns{0};
  std::atomic<u64> gpu_graph_ns{0};
  std::atomic<u64> gpu_score_ns{0};
  std::atomic<u64> gpu_beam_ns{0};
  std::atomic<u64> gpu_exact_ns{0};
  std::atomic<u64> rdma_read_ops{0};
  std::atomic<u64> rdma_read_bytes{0};
  std::atomic<u64> rdma_merged_requests{0};
  std::atomic<u64> direct_path_failures{0};
  std::atomic<u64> graph_page_requests{0};
  std::atomic<u64> graph_dependency_rounds{0};
  std::atomic<u64> graph_page_cache_hits{0};
  std::atomic<u64> graph_route_hits{0};
  std::atomic<u64> graph_route_refreshes{0};
  std::atomic<u64> graph_cache_invalidations{0};
  std::atomic<u64> exact_vector_reads{0};
  std::atomic<u64> exact_vector_cache_hits{0};
  std::atomic<u64> delta_queries{0};
  std::atomic<u64> mutations_published{0};
  std::atomic<u64> delta_publications{0};
  std::atomic<u64> delta_reclaim_batches{0};
  std::atomic<u64> delta_entries_retired{0};
  std::atomic<u64> storage_reclaim_ack_writes{0};
  std::atomic<u64> storage_reclaim_ack_sequence{0};
  std::atomic<u64> delta_live_entries{0};
  std::atomic<u64> delta_physical_entries{0};
  std::atomic<u64> delta_mutable_entries{0};
  std::atomic<u64> delta_durable_entries{0};
  std::atomic<u64> resident_pq_capacity{0};
  std::atomic<u64> resident_pq_entries{0};
  std::atomic<u64> resident_pq_peak_entries{0};
  std::atomic<u64> resident_pq_reclaimed{0};
  std::atomic<u64> mutation_capacity_rejections{0};
  std::atomic<u64> mutation_capacity_reserved{0};
  std::atomic<u64> mutation_capacity_reserved_max{0};
  std::atomic<u64> visibility_ns_total{0};
  std::atomic<u64> visibility_ns_max{0};
  std::atomic<u64> publication_queue_ns_total{0};
  std::atomic<u64> publication_prepare_ns_total{0};
  std::atomic<u64> publication_command_ns_total{0};
};

}  // namespace gpu_search
