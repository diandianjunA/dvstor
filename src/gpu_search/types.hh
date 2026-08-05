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
  exact_fetch = 7,
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
  u64 frontier_preview_cycles{};
  u64 frontier_prepare_cycles{};
  u64 frontier_enqueue_cycles{};
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
  // Physical graph bytes fetched for dynamic handles only. This is kept
  // separate from graph_read_bytes so DynaExtent savings are not hidden by
  // the much larger immutable base-node population.
  u64 dynamic_graph_read_bytes{};
  // Adaptive Speculative Frontier Execution (ASFE) keeps communication issue
  // width independent from authoritative graph-expansion commit width. These
  // counters are query-local and are reduced into Telemetry by the host
  // completion loop. In particular, graph_page_requests remains sourced from
  // remote_pages; the critical/speculative split below is not used to
  // reconstruct or reinterpret that existing counter.
  u64 critical_graph_bytes{};
  u64 speculative_graph_bytes{};
  // Bytes fetched speculatively whose records were ultimately discarded
  // before the authoritative commit frontier.  This is intentionally
  // separate from speculative_stale (a request count): a single request can
  // have a variable Live-Extent payload.
  u64 speculative_wasted_bytes{};
  // Full fixed-record bytes fetched by the terminal exact cache but not
  // promoted into the final authoritative exact Beam.
  u64 terminal_exact_cache_wasted_bytes{};
  // Sum of submission-group completion latencies measured on the GPU.  A
  // submission group is one descriptor group that produces one final CQE;
  // keeping both the sum and count makes the metric independent of query
  // batching and avoids presenting a request-count approximation as latency.
  u64 rdma_completion_latency_ns{};
  u64 speculative_completion_latency_ns{};
  u64 rdma_completion_groups{};
  u64 speculative_completion_groups{};
  u64 issue_width_sum{};
  // Sum of the controller's issue-width capacity for every issue epoch.  The
  // ratio issue_width_sum / issue_width_capacity_sum is the realized
  // outstanding-frontier utilization.
  u64 issue_width_capacity_sum{};
  u64 commit_width_sum{};
  u64 speculative_wait_cycles{};
  // Core-prefetch waves are the guaranteed next commit frontier.  They share
  // the critical owner queue with authoritative misses, but remain
  // asynchronous until the next commit phase.  These counters make the
  // overlap visible without folding core traffic into speculative waste.
  u64 core_prefetch_bytes{};
  u32 query_slot{};
  u32 result_count{};
  i32 status{};
  u32 remote_pages{};
  u32 remote_batches{};
  u32 graph_rounds{};
  u32 logical_expansions{};
  u32 critical_graph_reads{};
  u32 speculative_graph_reads{};
  u32 speculative_arrived{};
  u32 speculative_promoted{};
  u32 speculative_stale{};
  u32 speculative_queue_rejects{};
  u32 core_prefetch_reads{};
  u32 core_prefetch_arrived{};
  u32 core_prefetch_promoted{};
  u32 core_prefetch_stale{};
  u32 core_prefetch_queue_rejects{};
  u32 core_prefetch_waves{};
  u32 core_ready_waves{};
  // Terminal-horizon exact-cache diagnostics. attempted_queries is 0/1 for a
  // single completion and is widened by the host aggregation path. Records
  // are counted independently so promotion and fail-soft miss behavior remain
  // visible without reusing the frontier-certificate diagnostic slots.
  u32 terminal_exact_cache_attempted_queries{};
  u32 terminal_exact_cache_issued_records{};
  u32 terminal_exact_cache_promoted_records{};
  u32 terminal_exact_cache_queue_rejects{};
  u32 terminal_exact_cache_miss_records{};
  // PQ phase-boundary diagnostics.  Reuse the four retired production-DEEC
  // slots so adding these counters does not enlarge CompletionDescriptor (and
  // therefore does not enlarge the persistent CTA's shared completion object).
  // A batch is counted only when its candidate count is nonzero.
  u32 completion_score_batches{};
  u32 completion_score_candidates{};
  u32 frontier_telemetry_reserved0{};
  u32 frontier_telemetry_reserved1{};
  // Exact Issue-Frontier certificates derived from the already-required
  // Stable-Run leaves.  Their bounded tree prefix is retained and reused by
  // the authoritative merge; DEEC remains only a focused test primitive.
  u32 frontier_reusable_certificates{};
  // Streaming/ordered-score/SRFC work counters.  These are deliberately
  // query-local u32 values: the completion loop widens them to u64 before
  // process-wide aggregation.
  u32 frontier_streamed_candidate_runs{};
  u32 ordered_score_batches{};
  u32 ordered_score_candidates{};
  u32 frontier_reusable_prefix_ranks{};
  u32 frontier_reusable_full_prefix_certificates{};
  u32 frontier_reusable_issued_certificates{};
  u32 issue_epochs{};
  u32 commit_epochs{};
  u32 max_issue_width{};
  u32 max_commit_width{};
  u32 critical_rob_hits{};
  u32 critical_misses{};
  u32 exact_vectors{};
  // Populated-shard attempts to issue one fenced full-record/trailer train,
  // and the disjoint subset that had to use the correctness-preserving
  // two-batch fallback (normally only an SQ-capacity/compatibility event).
  // Therefore successful trains = batches - fallbacks.
  u32 exact_snapshot_train_batches{};
  u32 exact_snapshot_train_fallbacks{};
  u32 route_hits{};
  u32 graph_read_retries{};
  u32 graph_live_extent_reads{};
  u32 graph_full_record_reads{};
  u32 graph_extent_fallback_reads{};
  u32 graph_extent_underhint_reads{};
  u32 graph_extent_hint_promotions{};
  // DynaExtent raw physical snapshot-attempt and repair counters. A fallback
  // contributes one short attempt plus one full attempt, but checksum retries
  // add further physical attempts, so logical graph reads cannot be inferred
  // from these fields. The host widens the query-local counters unchanged.
  u32 dynamic_graph_short_reads{};
  u32 dynamic_graph_full_reads{};
  u32 dynamic_graph_fallback_reads{};
  u32 dynamic_graph_hint_promotions{};
  u32 dynamic_graph_hint_demotions{};
  u32 dynamic_code_candidates{};
  u32 dynamic_code_reads{};
  // Schema-compatible name: this is now the complete rejected-snapshot
  // count, including incarnation mismatch, arena replacement overlap, and
  // trailer-checksum failure. It is not a pure incarnation-mismatch metric.
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
  // Low 8 bits encode QueryFailureReason; the high 24 bits count complete
  // centroid-route snapshot retries caused by a concurrent publication.
  u32 diagnostic{};
  // Successful 0 -> occupied arena transitions. Appending this field consumes
  // the descriptor's former four-byte tail padding, preserving every existing
  // field offset and the mapped-ring ABI size.
  u32 dynamic_code_cache_first_occupancies{};
};

// CompletionDescriptor is embedded once in persistent-kernel shared memory
// and is also the mapped device-to-host ring ABI. Keep the explicit size check
// synchronized with both sides whenever production telemetry extends it.
static_assert(sizeof(CompletionDescriptor) == 584);
static_assert(alignof(CompletionDescriptor) == alignof(u64));

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
  u64 gpu_frontier_preview_ns{};
  u64 gpu_frontier_prepare_ns{};
  u64 gpu_frontier_enqueue_ns{};
  u64 gpu_rdma_wait_ns{};
  u64 gpu_graph_validation_ns{};
  u64 gpu_neighbor_decode_ns{};
  u64 gpu_pq_score_ns{};
  u64 gpu_visited_ns{};
  u64 gpu_beam_merge_ns{};
  u64 gpu_beam_merge_prepare_ns{};
  u64 gpu_beam_merge_sort_ns{};
  u64 gpu_beam_merge_materialize_ns{};
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
  u64 dynamic_graph_short_reads{};
  u64 dynamic_graph_full_reads{};
  u64 dynamic_graph_read_bytes{};
  u64 dynamic_graph_fallback_reads{};
  u64 dynamic_graph_hint_promotions{};
  u64 dynamic_graph_hint_demotions{};
  u64 logical_expansions{};
  u64 critical_graph_reads{};
  u64 critical_graph_bytes{};
  u64 speculative_graph_reads{};
  u64 speculative_graph_bytes{};
  u64 speculative_wasted_bytes{};
  u64 terminal_exact_cache_wasted_bytes{};
  u64 rdma_completion_latency_ns{};
  u64 speculative_completion_latency_ns{};
  u64 rdma_completion_groups{};
  u64 speculative_completion_groups{};
  u64 speculative_arrived{};
  u64 speculative_promoted{};
  u64 speculative_stale{};
  u64 speculative_queue_rejects{};
  u64 issue_epochs{};
  u64 commit_epochs{};
  u64 issue_width_sum{};
  u64 issue_width_capacity_sum{};
  u64 commit_width_sum{};
  u64 core_prefetch_bytes{};
  u64 max_issue_width{};
  u64 max_commit_width{};
  u64 critical_rob_hits{};
  u64 critical_misses{};
  u64 speculative_wait_ns{};
  u64 core_prefetch_reads{};
  u64 core_prefetch_arrived{};
  u64 core_prefetch_promoted{};
  u64 core_prefetch_stale{};
  u64 core_prefetch_queue_rejects{};
  u64 core_prefetch_waves{};
  u64 core_ready_waves{};
  u64 terminal_exact_cache_attempted_queries{};
  u64 terminal_exact_cache_issued_records{};
  u64 terminal_exact_cache_promoted_records{};
  u64 terminal_exact_cache_queue_rejects{};
  u64 terminal_exact_cache_miss_records{};
  u64 completion_score_batches{};
  u64 completion_score_candidates{};
  u64 frontier_reusable_certificates{};
  u64 frontier_streamed_candidate_runs{};
  u64 ordered_score_batches{};
  u64 ordered_score_candidates{};
  u64 ooo_bypassed_parents{};
  u64 frontier_reusable_prefix_ranks{};
  u64 frontier_reusable_full_prefix_certificates{};
  u64 frontier_reusable_issued_certificates{};
  u64 frontier_certificate_rejects{};
  u64 owner_submitted_wqes{};
  u64 owner_submission_wqe_capacity{};
  u64 owner_critical_batches{};
  u64 owner_speculative_batches{};
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
  u64 exact_snapshot_train_batches{};
  u64 exact_snapshot_train_fallbacks{};
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
  // Static kernel resource facts are exported with every interval report.
  // They are not reset between warmup and measurement.
  u64 gpu_kernel_threads{};
  u64 gpu_registers_per_thread{};
  u64 gpu_static_shared_bytes{};
  u64 gpu_active_blocks_per_sm{};
  u64 gpu_effective_blocks_per_sm{};
  u64 gpu_query_blocks{};
  u64 gpu_owner_blocks{};
  u64 gpu_total_persistent_blocks{};
};

class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();
  void set_gpu_occupancy(
      u64 kernel_threads,
      u64 registers_per_thread,
      u64 static_shared_bytes,
      u64 active_blocks_per_sm,
      u64 effective_blocks_per_sm,
      u64 query_blocks,
      u64 owner_blocks,
      u64 total_persistent_blocks);

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
  std::atomic<u64> gpu_frontier_preview_ns{0};
  std::atomic<u64> gpu_frontier_prepare_ns{0};
  std::atomic<u64> gpu_frontier_enqueue_ns{0};
  std::atomic<u64> gpu_rdma_wait_ns{0};
  std::atomic<u64> gpu_graph_validation_ns{0};
  std::atomic<u64> gpu_neighbor_decode_ns{0};
  std::atomic<u64> gpu_pq_score_ns{0};
  std::atomic<u64> gpu_visited_ns{0};
  std::atomic<u64> gpu_beam_merge_ns{0};
  std::atomic<u64> gpu_beam_merge_prepare_ns{0};
  std::atomic<u64> gpu_beam_merge_sort_ns{0};
  std::atomic<u64> gpu_beam_merge_materialize_ns{0};
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
  std::atomic<u64> dynamic_graph_short_reads{0};
  std::atomic<u64> dynamic_graph_full_reads{0};
  std::atomic<u64> dynamic_graph_read_bytes{0};
  std::atomic<u64> dynamic_graph_fallback_reads{0};
  std::atomic<u64> dynamic_graph_hint_promotions{0};
  std::atomic<u64> dynamic_graph_hint_demotions{0};
  std::atomic<u64> logical_expansions{0};
  std::atomic<u64> critical_graph_reads{0};
  std::atomic<u64> critical_graph_bytes{0};
  std::atomic<u64> speculative_graph_reads{0};
  std::atomic<u64> speculative_graph_bytes{0};
  std::atomic<u64> speculative_wasted_bytes{0};
  std::atomic<u64> terminal_exact_cache_wasted_bytes{0};
  std::atomic<u64> rdma_completion_latency_ns{0};
  std::atomic<u64> speculative_completion_latency_ns{0};
  std::atomic<u64> rdma_completion_groups{0};
  std::atomic<u64> speculative_completion_groups{0};
  std::atomic<u64> speculative_arrived{0};
  std::atomic<u64> speculative_promoted{0};
  std::atomic<u64> speculative_stale{0};
  std::atomic<u64> speculative_queue_rejects{0};
  std::atomic<u64> issue_epochs{0};
  std::atomic<u64> commit_epochs{0};
  std::atomic<u64> issue_width_sum{0};
  std::atomic<u64> issue_width_capacity_sum{0};
  std::atomic<u64> commit_width_sum{0};
  std::atomic<u64> core_prefetch_bytes{0};
  std::atomic<u64> max_issue_width{0};
  std::atomic<u64> max_commit_width{0};
  std::atomic<u64> critical_rob_hits{0};
  std::atomic<u64> critical_misses{0};
  std::atomic<u64> speculative_wait_ns{0};
  std::atomic<u64> core_prefetch_reads{0};
  std::atomic<u64> core_prefetch_arrived{0};
  std::atomic<u64> core_prefetch_promoted{0};
  std::atomic<u64> core_prefetch_stale{0};
  std::atomic<u64> core_prefetch_queue_rejects{0};
  std::atomic<u64> core_prefetch_waves{0};
  std::atomic<u64> core_ready_waves{0};
  std::atomic<u64> terminal_exact_cache_attempted_queries{0};
  std::atomic<u64> terminal_exact_cache_issued_records{0};
  std::atomic<u64> terminal_exact_cache_promoted_records{0};
  std::atomic<u64> terminal_exact_cache_queue_rejects{0};
  std::atomic<u64> terminal_exact_cache_miss_records{0};
  std::atomic<u64> completion_score_batches{0};
  std::atomic<u64> completion_score_candidates{0};
  std::atomic<u64> frontier_reusable_certificates{0};
  std::atomic<u64> frontier_streamed_candidate_runs{0};
  std::atomic<u64> ordered_score_batches{0};
  std::atomic<u64> ordered_score_candidates{0};
  std::atomic<u64> ooo_bypassed_parents{0};
  std::atomic<u64> frontier_reusable_prefix_ranks{0};
  std::atomic<u64> frontier_reusable_full_prefix_certificates{0};
  std::atomic<u64> frontier_reusable_issued_certificates{0};
  std::atomic<u64> frontier_certificate_rejects{0};
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
  std::atomic<u64> exact_snapshot_train_batches{0};
  std::atomic<u64> exact_snapshot_train_fallbacks{0};
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
  std::atomic<u64> gpu_kernel_threads{0};
  std::atomic<u64> gpu_registers_per_thread{0};
  std::atomic<u64> gpu_static_shared_bytes{0};
  std::atomic<u64> gpu_active_blocks_per_sm{0};
  std::atomic<u64> gpu_effective_blocks_per_sm{0};
  std::atomic<u64> gpu_query_blocks{0};
  std::atomic<u64> gpu_owner_blocks{0};
  std::atomic<u64> gpu_total_persistent_blocks{0};
};

}  // namespace gpu_search
