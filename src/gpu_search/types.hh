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
  u64 query_device_address{};
  u64 result_device_address{};
  u32 query_slot{};
  u32 result_capacity{};
  // The runtime/index dimension is u32. Keeping only 16 bits here silently
  // truncated otherwise valid layouts before the device-side shape check.
  u32 dim{};
  u16 k{};
  u8 query_dtype{};
  u8 flags{};
};

static_assert(sizeof(QueryDescriptor) == 40);

enum class QueryFailureReason : u32 {
  none = 0,
  invalid_descriptor = 1,
  route_snapshot_timeout = 2,
  route_no_seed = 3,
  graph_fetch = 4,
  dynamic_code_fetch = 5,
  exact_rerank_empty = 6,
};

inline constexpr u32 kQueryFailureReasonBits = 8;
inline constexpr u32 kQueryFailureReasonMask =
  (u32{1} << kQueryFailureReasonBits) - 1;

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 make_query_diagnostic(QueryFailureReason reason,
                                           u32 route_snapshot_retries = 0) {
  const u32 bounded_retries = route_snapshot_retries > 0x00ffffffu
    ? 0x00ffffffu : route_snapshot_retries;
  return (bounded_retries << kQueryFailureReasonBits) |
    static_cast<u32>(reason);
}

inline constexpr QueryFailureReason query_failure_reason(u32 diagnostic) {
  return static_cast<QueryFailureReason>(
    diagnostic & kQueryFailureReasonMask);
}

inline constexpr u32 query_route_snapshot_retries(u32 diagnostic) {
  return diagnostic >> kQueryFailureReasonBits;
}

struct CompletionDescriptor {
  u64 request_id{};
  u64 gpu_cycles{};
  u64 prepare_cycles{};
  u64 graph_cycles{};
  u64 score_cycles{};
  u64 beam_cycles{};
  u64 exact_cycles{};
  u64 dynamic_code_cycles{};
  u64 beam_selection_cycles{};
  u64 rdma_issue_cycles{};
  u64 rdma_wait_cycles{};
  u64 graph_validation_cycles{};
  u64 neighbor_decode_cycles{};
  u64 pq_score_cycles{};
  u64 visited_cycles{};
  u64 beam_merge_cycles{};
  u64 beam_merge_prepare_cycles{};
  u64 beam_merge_sort_cycles{};
  u64 beam_merge_materialize_cycles{};
  // Actual graph-record RDMA payload issued by this query. Unlike
  // remote_pages, this includes snapshot retries and live-extent fallback
  // reads and therefore cannot be reconstructed from the physical record
  // size when variable-length reads are enabled.
  u64 graph_read_bytes{};
  u32 query_slot{};
  u32 result_count{};
  i32 status{};
  u32 remote_pages{};
  u32 remote_batches{};
  u32 graph_rounds{};
  u32 exact_vectors{};
  u32 route_hits{};
  u32 graph_read_retries{};
  u32 graph_live_extent_reads{};
  u32 graph_full_record_reads{};
  u32 graph_extent_fallback_reads{};
  u32 graph_extent_underhint_reads{};
  u32 graph_extent_hint_promotions{};
  u32 expansion_policy{};
  u32 sum_selected_parents{};
  u32 sum_feedback_horizon{};
  u32 sum_hardware_credit_tiles{};
  u32 minimum_selected_batch{};
  u32 maximum_selected_batch{};
  u32 minimum_feedback_horizon{};
  u32 maximum_feedback_horizon{};
  u32 extra_parent_count{};
  u32 qp_lease_claim_count{};
  u32 qp_lease_reject_count{};
  u32 qp_lease_rollback_count{};
  u32 compute_allowance_tile_sum{};
  u32 marginal_probe_pass_count{};
  u32 marginal_probe_fail_count{};
  u32 reserved_expansion{};
  u32 dynamic_code_candidates{};
  u32 dynamic_code_reads{};
  u32 dynamic_code_incarnation_rejects{};
  u32 dynamic_code_cache_hits{};
  u32 dynamic_code_batch_deduplicated{};
  u32 dynamic_code_cache_publish_successes{};
  u32 dynamic_code_cache_publish_races{};
  u32 dynamic_code_cache_lookup_probe_exhaustions{};
  u32 dynamic_code_cache_publish_probe_exhaustions{};
  u32 dynamic_code_cache_lookup_probes{};
  u32 dynamic_code_cache_max_lookup_probes{};
  u32 trace_event_count{};
  u32 trace_overflow{};
  u32 adjacency_oracle_event_count{};
  u32 adjacency_oracle_overflow{};
  // Low 8 bits encode QueryFailureReason; the high 24 bits count complete
  // centroid-route snapshot retries caused by a concurrent publication.
  u32 diagnostic{};
};

struct CentroidRoutePublishDescriptor {
  u64 command_id{};
  u32 update_count{};
  u32 reserved{};
};

struct CentroidRoutePublishCompletion {
  u64 command_id{};
  i32 status{};
  u32 update_count{};
};

struct TelemetrySnapshot {
  u64 gpu_memory_explicit_bytes{};
  u64 gpu_memory_base_pq_bytes{};
  u64 gpu_memory_route_graph_bytes{};
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
  u64 gpu_beam_selection_ns{};
  u64 gpu_rdma_issue_ns{};
  u64 gpu_rdma_wait_ns{};
  u64 gpu_graph_validation_ns{};
  u64 gpu_neighbor_decode_ns{};
  u64 gpu_pq_score_ns{};
  u64 gpu_visited_ns{};
  u64 gpu_beam_merge_ns{};
  u64 gpu_beam_merge_prepare_ns{};
  u64 gpu_beam_merge_sort_ns{};
  u64 gpu_beam_merge_materialize_ns{};
  u64 feedback_hunger_queries{};
  u64 expansion_sum_selected_parents{};
  u64 expansion_sum_feedback_horizon{};
  u64 expansion_sum_hardware_credit_tiles{};
  u64 expansion_minimum_selected_batch{};
  u64 expansion_maximum_selected_batch{};
  u64 expansion_minimum_feedback_horizon{};
  u64 expansion_maximum_feedback_horizon{};
  u64 expansion_extra_parents{};
  u64 expansion_qp_lease_claims{};
  u64 expansion_qp_lease_rejects{};
  u64 expansion_qp_lease_rollbacks{};
  u64 expansion_compute_allowance_tiles{};
  u64 expansion_marginal_probe_passes{};
  u64 expansion_marginal_probe_failures{};
  u64 expansion_pressure_active_queries{};
  u64 expansion_pressure_active_queries_peak{};
  u64 expansion_pressure_credit_current{};
  u64 expansion_pressure_credit_max_observed{};
  u64 expansion_pressure_maximum_credit_tiles{};
  u64 expansion_pressure_hunger_grants{};
  u64 expansion_pressure_idle_owner_episodes{};
  u64 expansion_pressure_congestion_clears{};
  u64 expansion_pressure_ring_backpressure_events{};
  u64 expansion_pressure_sq_defer_events{};
  u64 expansion_qp_lease_available_wqes{};
  u64 expansion_qp_lease_offers{};
  u64 expansion_qp_lease_claimed_wqes{};
  u64 expansion_qp_lease_owner_rejects{};
  u64 expansion_qp_lease_returns{};
  u64 expansion_qp_lease_revocations{};
  u64 expansion_qp_lease_stale_returns{};
  u64 rdma_read_ops{};
  u64 rdma_read_bytes{};
  u64 rdma_merged_requests{};
  u64 direct_path_failures{};
  u64 graph_page_requests{};
  u64 graph_shard_batches{};
  u64 graph_read_retries{};
  u64 graph_read_bytes{};
  u64 graph_live_extent_reads{};
  u64 graph_full_record_reads{};
  u64 graph_extent_fallback_reads{};
  u64 graph_extent_underhint_reads{};
  u64 graph_extent_hint_promotions{};
  u64 graph_dependency_rounds{};
  u64 graph_route_hits{};
  u64 graph_route_refreshes{};
  u64 centroid_route_publications{};
  u64 centroid_route_shard_updates{};
  u64 centroid_route_live_entries{};
  u64 centroid_route_snapshot_skips{};
  u64 centroid_route_probe_reads{};
  u64 centroid_route_body_reads{};
  u64 centroid_route_unchanged_polls{};
  u64 centroid_route_poll_delay_us{};
  u64 centroid_route_query_retries{};
  u64 centroid_route_query_timeouts{};
  u64 exact_vector_reads{};
  u64 dynamic_code_candidates{};
  u64 dynamic_code_reads{};
  u64 dynamic_code_read_bytes{};
  u64 dynamic_code_incarnation_rejects{};
  u64 dynamic_code_wait_ns{};
  u64 dynamic_code_cache_hits{};
  u64 dynamic_code_batch_deduplicated{};
  u64 dynamic_code_cache_publish_successes{};
  u64 dynamic_code_cache_publish_races{};
  u64 dynamic_code_cache_lookup_probe_exhaustions{};
  u64 dynamic_code_cache_publish_probe_exhaustions{};
  u64 dynamic_code_cache_lookup_probes{};
  u64 dynamic_code_cache_max_lookup_probes{};
  u64 dynamic_code_cache_occupied{};
  u64 dynamic_code_cache_capacity{};
};

class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();

  std::atomic<u64> gpu_memory_explicit_bytes{0};
  std::atomic<u64> gpu_memory_base_pq_bytes{0};
  std::atomic<u64> gpu_memory_route_graph_bytes{0};
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
  std::atomic<u64> gpu_beam_selection_ns{0};
  std::atomic<u64> gpu_rdma_issue_ns{0};
  std::atomic<u64> gpu_rdma_wait_ns{0};
  std::atomic<u64> gpu_graph_validation_ns{0};
  std::atomic<u64> gpu_neighbor_decode_ns{0};
  std::atomic<u64> gpu_pq_score_ns{0};
  std::atomic<u64> gpu_visited_ns{0};
  std::atomic<u64> gpu_beam_merge_ns{0};
  std::atomic<u64> gpu_beam_merge_prepare_ns{0};
  std::atomic<u64> gpu_beam_merge_sort_ns{0};
  std::atomic<u64> gpu_beam_merge_materialize_ns{0};
  std::atomic<u64> feedback_hunger_queries{0};
  std::atomic<u64> expansion_sum_selected_parents{0};
  std::atomic<u64> expansion_sum_feedback_horizon{0};
  std::atomic<u64> expansion_sum_hardware_credit_tiles{0};
  std::atomic<u64> expansion_minimum_selected_batch{UINT64_MAX};
  std::atomic<u64> expansion_maximum_selected_batch{0};
  std::atomic<u64> expansion_minimum_feedback_horizon{UINT64_MAX};
  std::atomic<u64> expansion_maximum_feedback_horizon{0};
  std::atomic<u64> expansion_extra_parents{0};
  std::atomic<u64> expansion_qp_lease_claims{0};
  std::atomic<u64> expansion_qp_lease_rejects{0};
  std::atomic<u64> expansion_qp_lease_rollbacks{0};
  std::atomic<u64> expansion_compute_allowance_tiles{0};
  std::atomic<u64> expansion_marginal_probe_passes{0};
  std::atomic<u64> expansion_marginal_probe_failures{0};
  std::atomic<u64> rdma_read_ops{0};
  std::atomic<u64> rdma_read_bytes{0};
  std::atomic<u64> rdma_merged_requests{0};
  std::atomic<u64> direct_path_failures{0};
  std::atomic<u64> graph_page_requests{0};
  std::atomic<u64> graph_shard_batches{0};
  std::atomic<u64> graph_read_retries{0};
  std::atomic<u64> graph_read_bytes{0};
  std::atomic<u64> graph_live_extent_reads{0};
  std::atomic<u64> graph_full_record_reads{0};
  std::atomic<u64> graph_extent_fallback_reads{0};
  std::atomic<u64> graph_extent_underhint_reads{0};
  std::atomic<u64> graph_extent_hint_promotions{0};
  std::atomic<u64> graph_dependency_rounds{0};
  std::atomic<u64> graph_route_hits{0};
  std::atomic<u64> graph_route_refreshes{0};
  std::atomic<u64> centroid_route_publications{0};
  std::atomic<u64> centroid_route_shard_updates{0};
  std::atomic<u64> centroid_route_live_entries{0};
  std::atomic<u64> centroid_route_snapshot_skips{0};
  std::atomic<u64> centroid_route_probe_reads{0};
  std::atomic<u64> centroid_route_body_reads{0};
  std::atomic<u64> centroid_route_unchanged_polls{0};
  std::atomic<u64> centroid_route_poll_delay_us{0};
  std::atomic<u64> centroid_route_query_retries{0};
  std::atomic<u64> centroid_route_query_timeouts{0};
  std::atomic<u64> exact_vector_reads{0};
  std::atomic<u64> dynamic_code_candidates{0};
  std::atomic<u64> dynamic_code_reads{0};
  std::atomic<u64> dynamic_code_read_bytes{0};
  std::atomic<u64> dynamic_code_incarnation_rejects{0};
  std::atomic<u64> dynamic_code_wait_ns{0};
  std::atomic<u64> dynamic_code_cache_hits{0};
  std::atomic<u64> dynamic_code_batch_deduplicated{0};
  std::atomic<u64> dynamic_code_cache_publish_successes{0};
  std::atomic<u64> dynamic_code_cache_publish_races{0};
  std::atomic<u64> dynamic_code_cache_lookup_probe_exhaustions{0};
  std::atomic<u64> dynamic_code_cache_publish_probe_exhaustions{0};
  std::atomic<u64> dynamic_code_cache_lookup_probes{0};
  std::atomic<u64> dynamic_code_cache_max_lookup_probes{0};
  // Occupancy is lifetime state, not an interval counter. reset() deliberately
  // preserves it so a post-warmup benchmark reports the cache it actually uses.
  std::atomic<u64> dynamic_code_cache_occupied{0};
  std::atomic<u64> dynamic_code_cache_capacity{0};
};

}  // namespace gpu_search
