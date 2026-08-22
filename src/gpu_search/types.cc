#include "gpu_search/types.hh"

namespace gpu_search {

TelemetrySnapshot Telemetry::snapshot() const {
  TelemetrySnapshot snapshot{
    .gpu_memory_explicit_bytes = gpu_memory_explicit_bytes.load(std::memory_order_relaxed),
    .gpu_memory_base_pq_bytes = gpu_memory_base_pq_bytes.load(std::memory_order_relaxed),
    .gpu_memory_route_graph_bytes = gpu_memory_route_graph_bytes.load(std::memory_order_relaxed),
    .queries_submitted = queries_submitted.load(std::memory_order_relaxed),
    .queries_completed = queries_completed.load(std::memory_order_relaxed),
    .batches = batches.load(std::memory_order_relaxed),
    .batch_queries = batch_queries.load(std::memory_order_relaxed),
    .submission_wait_ns = submission_wait_ns.load(std::memory_order_relaxed),
    .completion_wait_ns = completion_wait_ns.load(std::memory_order_relaxed),
    .gpu_active_ns = gpu_active_ns.load(std::memory_order_relaxed),
    .gpu_prepare_ns = gpu_prepare_ns.load(std::memory_order_relaxed),
    .gpu_graph_ns = gpu_graph_ns.load(std::memory_order_relaxed),
    .gpu_score_ns = gpu_score_ns.load(std::memory_order_relaxed),
    .gpu_beam_ns = gpu_beam_ns.load(std::memory_order_relaxed),
    .gpu_exact_ns = gpu_exact_ns.load(std::memory_order_relaxed),
    .gpu_beam_selection_ns = gpu_beam_selection_ns.load(std::memory_order_relaxed),
    .gpu_rdma_issue_ns = gpu_rdma_issue_ns.load(std::memory_order_relaxed),
    .gpu_frontier_preview_ns =
      gpu_frontier_preview_ns.load(std::memory_order_relaxed),
    .gpu_frontier_prepare_ns =
      gpu_frontier_prepare_ns.load(std::memory_order_relaxed),
    .gpu_frontier_enqueue_ns =
      gpu_frontier_enqueue_ns.load(std::memory_order_relaxed),
    .gpu_rdma_wait_ns = gpu_rdma_wait_ns.load(std::memory_order_relaxed),
    .gpu_graph_validation_ns = gpu_graph_validation_ns.load(std::memory_order_relaxed),
    .gpu_neighbor_decode_ns = gpu_neighbor_decode_ns.load(std::memory_order_relaxed),
    .gpu_pq_score_ns = gpu_pq_score_ns.load(std::memory_order_relaxed),
    .gpu_visited_ns = gpu_visited_ns.load(std::memory_order_relaxed),
    .gpu_beam_merge_ns = gpu_beam_merge_ns.load(std::memory_order_relaxed),
    .gpu_beam_merge_prepare_ns =
      gpu_beam_merge_prepare_ns.load(std::memory_order_relaxed),
    .gpu_beam_merge_sort_ns =
      gpu_beam_merge_sort_ns.load(std::memory_order_relaxed),
    .gpu_beam_merge_materialize_ns =
      gpu_beam_merge_materialize_ns.load(std::memory_order_relaxed),
    .rdma_read_ops = rdma_read_ops.load(std::memory_order_relaxed),
    .rdma_read_bytes = rdma_read_bytes.load(std::memory_order_relaxed),
    .rdma_merged_requests = rdma_merged_requests.load(std::memory_order_relaxed),
    .direct_path_failures = direct_path_failures.load(std::memory_order_relaxed),
    .graph_page_requests = graph_page_requests.load(std::memory_order_relaxed),
    .graph_shard_batches = graph_shard_batches.load(std::memory_order_relaxed),
    .graph_read_retries = graph_read_retries.load(std::memory_order_relaxed),
    .graph_read_bytes = graph_read_bytes.load(std::memory_order_relaxed),
    .graph_live_extent_reads =
      graph_live_extent_reads.load(std::memory_order_relaxed),
    .graph_full_record_reads =
      graph_full_record_reads.load(std::memory_order_relaxed),
    .graph_extent_fallback_reads =
      graph_extent_fallback_reads.load(std::memory_order_relaxed),
    .graph_extent_underhint_reads =
      graph_extent_underhint_reads.load(std::memory_order_relaxed),
    .graph_extent_hint_promotions =
      graph_extent_hint_promotions.load(std::memory_order_relaxed),
    .expanded_parent_count =
      expanded_parent_count.load(std::memory_order_relaxed),
    .expanded_neighbor_count_sum =
      expanded_neighbor_count_sum.load(std::memory_order_relaxed),
    .dynamic_graph_short_reads =
      dynamic_graph_short_reads.load(std::memory_order_relaxed),
    .dynamic_graph_full_reads =
      dynamic_graph_full_reads.load(std::memory_order_relaxed),
    .dynamic_graph_read_bytes =
      dynamic_graph_read_bytes.load(std::memory_order_relaxed),
    .dynamic_graph_fallback_reads =
      dynamic_graph_fallback_reads.load(std::memory_order_relaxed),
    .dynamic_graph_hint_promotions =
      dynamic_graph_hint_promotions.load(std::memory_order_relaxed),
    .dynamic_graph_hint_demotions =
      dynamic_graph_hint_demotions.load(std::memory_order_relaxed),
    .logical_expansions =
      logical_expansions.load(std::memory_order_relaxed),
    .critical_graph_reads =
      critical_graph_reads.load(std::memory_order_relaxed),
    .critical_graph_bytes =
      critical_graph_bytes.load(std::memory_order_relaxed),
    .speculative_graph_reads =
      speculative_graph_reads.load(std::memory_order_relaxed),
    .speculative_graph_bytes =
      speculative_graph_bytes.load(std::memory_order_relaxed),
    .speculative_wasted_bytes =
      speculative_wasted_bytes.load(std::memory_order_relaxed),
    .terminal_exact_cache_wasted_bytes =
      terminal_exact_cache_wasted_bytes.load(std::memory_order_relaxed),
    .rdma_completion_latency_ns =
      rdma_completion_latency_ns.load(std::memory_order_relaxed),
    .speculative_completion_latency_ns =
      speculative_completion_latency_ns.load(std::memory_order_relaxed),
    .rdma_completion_groups =
      rdma_completion_groups.load(std::memory_order_relaxed),
    .speculative_completion_groups =
      speculative_completion_groups.load(std::memory_order_relaxed),
    .speculative_arrived =
      speculative_arrived.load(std::memory_order_relaxed),
    .speculative_promoted =
      speculative_promoted.load(std::memory_order_relaxed),
    .speculative_stale =
      speculative_stale.load(std::memory_order_relaxed),
    .speculative_queue_rejects =
      speculative_queue_rejects.load(std::memory_order_relaxed),
    .issue_epochs = issue_epochs.load(std::memory_order_relaxed),
    .commit_epochs = commit_epochs.load(std::memory_order_relaxed),
    .issue_width_sum = issue_width_sum.load(std::memory_order_relaxed),
    .issue_width_capacity_sum =
      issue_width_capacity_sum.load(std::memory_order_relaxed),
    .commit_width_sum = commit_width_sum.load(std::memory_order_relaxed),
    .core_prefetch_bytes =
      core_prefetch_bytes.load(std::memory_order_relaxed),
    .max_issue_width = max_issue_width.load(std::memory_order_relaxed),
    .max_commit_width = max_commit_width.load(std::memory_order_relaxed),
    .critical_rob_hits = critical_rob_hits.load(std::memory_order_relaxed),
    .critical_misses = critical_misses.load(std::memory_order_relaxed),
    .speculative_wait_ns =
      speculative_wait_ns.load(std::memory_order_relaxed),
    .core_prefetch_reads =
      core_prefetch_reads.load(std::memory_order_relaxed),
    .core_prefetch_arrived =
      core_prefetch_arrived.load(std::memory_order_relaxed),
    .core_prefetch_promoted =
      core_prefetch_promoted.load(std::memory_order_relaxed),
    .core_prefetch_stale =
      core_prefetch_stale.load(std::memory_order_relaxed),
    .core_prefetch_queue_rejects =
      core_prefetch_queue_rejects.load(std::memory_order_relaxed),
    .core_prefetch_waves =
      core_prefetch_waves.load(std::memory_order_relaxed),
    .core_ready_waves =
      core_ready_waves.load(std::memory_order_relaxed),
    .terminal_exact_cache_attempted_queries =
      terminal_exact_cache_attempted_queries.load(
        std::memory_order_relaxed),
    .terminal_exact_cache_issued_records =
      terminal_exact_cache_issued_records.load(std::memory_order_relaxed),
    .terminal_exact_cache_promoted_records =
      terminal_exact_cache_promoted_records.load(std::memory_order_relaxed),
    .terminal_exact_cache_queue_rejects =
      terminal_exact_cache_queue_rejects.load(std::memory_order_relaxed),
    .terminal_exact_cache_miss_records =
      terminal_exact_cache_miss_records.load(std::memory_order_relaxed),
    .completion_score_batches =
      completion_score_batches.load(std::memory_order_relaxed),
    .completion_score_candidates =
      completion_score_candidates.load(std::memory_order_relaxed),
    .frontier_reusable_certificates =
      frontier_reusable_certificates.load(std::memory_order_relaxed),
    .frontier_streamed_candidate_runs =
      frontier_streamed_candidate_runs.load(std::memory_order_relaxed),
    .ordered_score_batches =
      ordered_score_batches.load(std::memory_order_relaxed),
    .ordered_score_candidates =
      ordered_score_candidates.load(std::memory_order_relaxed),
    .ooo_bypassed_parents =
      ooo_bypassed_parents.load(std::memory_order_relaxed),
    .frontier_reusable_prefix_ranks =
      frontier_reusable_prefix_ranks.load(std::memory_order_relaxed),
    .frontier_reusable_full_prefix_certificates =
      frontier_reusable_full_prefix_certificates.load(
        std::memory_order_relaxed),
    .frontier_reusable_issued_certificates =
      frontier_reusable_issued_certificates.load(
        std::memory_order_relaxed),
    .frontier_certificate_rejects =
      frontier_certificate_rejects.load(std::memory_order_relaxed),
    .graph_dependency_rounds = graph_dependency_rounds.load(std::memory_order_relaxed),
    .graph_route_hits = graph_route_hits.load(std::memory_order_relaxed),
    .graph_route_refreshes = graph_route_refreshes.load(std::memory_order_relaxed),
    .centroid_route_publications =
      centroid_route_publications.load(std::memory_order_relaxed),
    .centroid_route_shard_updates =
      centroid_route_shard_updates.load(std::memory_order_relaxed),
    .centroid_route_live_entries =
      centroid_route_live_entries.load(std::memory_order_relaxed),
    .centroid_route_snapshot_skips =
      centroid_route_snapshot_skips.load(std::memory_order_relaxed),
    .centroid_route_probe_reads =
      centroid_route_probe_reads.load(std::memory_order_relaxed),
    .centroid_route_body_reads =
      centroid_route_body_reads.load(std::memory_order_relaxed),
    .centroid_route_unchanged_polls =
      centroid_route_unchanged_polls.load(std::memory_order_relaxed),
    .centroid_route_poll_delay_us =
      centroid_route_poll_delay_us.load(std::memory_order_relaxed),
    .centroid_route_query_retries =
      centroid_route_query_retries.load(std::memory_order_relaxed),
    .centroid_route_query_timeouts =
      centroid_route_query_timeouts.load(std::memory_order_relaxed),
    .exact_vector_reads = exact_vector_reads.load(std::memory_order_relaxed),
    .exact_snapshot_train_batches =
      exact_snapshot_train_batches.load(std::memory_order_relaxed),
    .exact_snapshot_train_fallbacks =
      exact_snapshot_train_fallbacks.load(std::memory_order_relaxed),
    .dynamic_code_candidates =
      dynamic_code_candidates.load(std::memory_order_relaxed),
    .dynamic_code_reads = dynamic_code_reads.load(std::memory_order_relaxed),
    .dynamic_code_read_bytes =
      dynamic_code_read_bytes.load(std::memory_order_relaxed),
    .dynamic_code_incarnation_rejects =
      dynamic_code_incarnation_rejects.load(std::memory_order_relaxed),
    .dynamic_code_wait_ns =
      dynamic_code_wait_ns.load(std::memory_order_relaxed),
    .dynamic_code_cache_hits =
      dynamic_code_cache_hits.load(std::memory_order_relaxed),
    .dynamic_code_batch_deduplicated =
      dynamic_code_batch_deduplicated.load(std::memory_order_relaxed),
    .dynamic_code_cache_publish_successes =
      dynamic_code_cache_publish_successes.load(std::memory_order_relaxed),
    .dynamic_code_cache_publish_races =
      dynamic_code_cache_publish_races.load(std::memory_order_relaxed),
    .dynamic_code_cache_lookup_probe_exhaustions =
      dynamic_code_cache_lookup_probe_exhaustions.load(
        std::memory_order_relaxed),
    .dynamic_code_cache_publish_probe_exhaustions =
      dynamic_code_cache_publish_probe_exhaustions.load(
        std::memory_order_relaxed),
    .dynamic_code_cache_lookup_probes =
      dynamic_code_cache_lookup_probes.load(std::memory_order_relaxed),
    .dynamic_code_cache_max_lookup_probes =
      dynamic_code_cache_max_lookup_probes.load(std::memory_order_relaxed),
    .dynamic_code_cache_occupied =
      dynamic_code_cache_occupied.load(std::memory_order_relaxed),
    .dynamic_code_cache_capacity =
      dynamic_code_cache_capacity.load(std::memory_order_relaxed),
    .gpu_kernel_threads =
      gpu_kernel_threads.load(std::memory_order_relaxed),
    .gpu_registers_per_thread =
      gpu_registers_per_thread.load(std::memory_order_relaxed),
    .gpu_static_shared_bytes =
      gpu_static_shared_bytes.load(std::memory_order_relaxed),
    .gpu_active_blocks_per_sm =
      gpu_active_blocks_per_sm.load(std::memory_order_relaxed),
    .gpu_effective_blocks_per_sm =
      gpu_effective_blocks_per_sm.load(std::memory_order_relaxed),
    .gpu_query_blocks = gpu_query_blocks.load(std::memory_order_relaxed),
    .gpu_owner_blocks = gpu_owner_blocks.load(std::memory_order_relaxed),
    .gpu_total_persistent_blocks =
      gpu_total_persistent_blocks.load(std::memory_order_relaxed),
  };
  for (u32 bucket = 0; bucket < kGraphDegreeHistogramBuckets; ++bucket) {
    snapshot.expanded_degree_histogram[bucket] =
      expanded_degree_histogram[bucket].load(std::memory_order_relaxed);
  }
  return snapshot;
}

void Telemetry::set_gpu_occupancy(
    u64 kernel_threads,
    u64 registers_per_thread,
    u64 static_shared_bytes,
    u64 active_blocks_per_sm,
    u64 effective_blocks_per_sm,
    u64 query_blocks,
    u64 owner_blocks,
    u64 total_persistent_blocks) {
  gpu_kernel_threads.store(kernel_threads, std::memory_order_relaxed);
  gpu_registers_per_thread.store(
    registers_per_thread, std::memory_order_relaxed);
  gpu_static_shared_bytes.store(
    static_shared_bytes, std::memory_order_relaxed);
  gpu_active_blocks_per_sm.store(
    active_blocks_per_sm, std::memory_order_relaxed);
  gpu_effective_blocks_per_sm.store(
    effective_blocks_per_sm, std::memory_order_relaxed);
  gpu_query_blocks.store(query_blocks, std::memory_order_relaxed);
  gpu_owner_blocks.store(owner_blocks, std::memory_order_relaxed);
  gpu_total_persistent_blocks.store(
    total_persistent_blocks, std::memory_order_relaxed);
}

void Telemetry::reset() {
  queries_submitted.store(0, std::memory_order_relaxed);
  queries_completed.store(0, std::memory_order_relaxed);
  batches.store(0, std::memory_order_relaxed);
  batch_queries.store(0, std::memory_order_relaxed);
  submission_wait_ns.store(0, std::memory_order_relaxed);
  completion_wait_ns.store(0, std::memory_order_relaxed);
  gpu_active_ns.store(0, std::memory_order_relaxed);
  gpu_prepare_ns.store(0, std::memory_order_relaxed);
  gpu_graph_ns.store(0, std::memory_order_relaxed);
  gpu_score_ns.store(0, std::memory_order_relaxed);
  gpu_beam_ns.store(0, std::memory_order_relaxed);
  gpu_exact_ns.store(0, std::memory_order_relaxed);
  gpu_beam_selection_ns.store(0, std::memory_order_relaxed);
  gpu_rdma_issue_ns.store(0, std::memory_order_relaxed);
  gpu_frontier_preview_ns.store(0, std::memory_order_relaxed);
  gpu_frontier_prepare_ns.store(0, std::memory_order_relaxed);
  gpu_frontier_enqueue_ns.store(0, std::memory_order_relaxed);
  gpu_rdma_wait_ns.store(0, std::memory_order_relaxed);
  gpu_graph_validation_ns.store(0, std::memory_order_relaxed);
  gpu_neighbor_decode_ns.store(0, std::memory_order_relaxed);
  gpu_pq_score_ns.store(0, std::memory_order_relaxed);
  gpu_visited_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_prepare_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_sort_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_materialize_ns.store(0, std::memory_order_relaxed);
  rdma_read_ops.store(0, std::memory_order_relaxed);
  rdma_read_bytes.store(0, std::memory_order_relaxed);
  rdma_merged_requests.store(0, std::memory_order_relaxed);
  direct_path_failures.store(0, std::memory_order_relaxed);
  graph_page_requests.store(0, std::memory_order_relaxed);
  graph_shard_batches.store(0, std::memory_order_relaxed);
  graph_read_retries.store(0, std::memory_order_relaxed);
  graph_read_bytes.store(0, std::memory_order_relaxed);
  graph_live_extent_reads.store(0, std::memory_order_relaxed);
  graph_full_record_reads.store(0, std::memory_order_relaxed);
  graph_extent_fallback_reads.store(0, std::memory_order_relaxed);
  graph_extent_underhint_reads.store(0, std::memory_order_relaxed);
  graph_extent_hint_promotions.store(0, std::memory_order_relaxed);
  expanded_parent_count.store(0, std::memory_order_relaxed);
  expanded_neighbor_count_sum.store(0, std::memory_order_relaxed);
  for (auto& bucket : expanded_degree_histogram) {
    bucket.store(0, std::memory_order_relaxed);
  }
  dynamic_graph_short_reads.store(0, std::memory_order_relaxed);
  dynamic_graph_full_reads.store(0, std::memory_order_relaxed);
  dynamic_graph_read_bytes.store(0, std::memory_order_relaxed);
  dynamic_graph_fallback_reads.store(0, std::memory_order_relaxed);
  dynamic_graph_hint_promotions.store(0, std::memory_order_relaxed);
  dynamic_graph_hint_demotions.store(0, std::memory_order_relaxed);
  logical_expansions.store(0, std::memory_order_relaxed);
  critical_graph_reads.store(0, std::memory_order_relaxed);
  critical_graph_bytes.store(0, std::memory_order_relaxed);
  speculative_graph_reads.store(0, std::memory_order_relaxed);
  speculative_graph_bytes.store(0, std::memory_order_relaxed);
  speculative_wasted_bytes.store(0, std::memory_order_relaxed);
  terminal_exact_cache_wasted_bytes.store(0, std::memory_order_relaxed);
  rdma_completion_latency_ns.store(0, std::memory_order_relaxed);
  speculative_completion_latency_ns.store(0, std::memory_order_relaxed);
  rdma_completion_groups.store(0, std::memory_order_relaxed);
  speculative_completion_groups.store(0, std::memory_order_relaxed);
  speculative_arrived.store(0, std::memory_order_relaxed);
  speculative_promoted.store(0, std::memory_order_relaxed);
  speculative_stale.store(0, std::memory_order_relaxed);
  speculative_queue_rejects.store(0, std::memory_order_relaxed);
  issue_epochs.store(0, std::memory_order_relaxed);
  commit_epochs.store(0, std::memory_order_relaxed);
  issue_width_sum.store(0, std::memory_order_relaxed);
  issue_width_capacity_sum.store(0, std::memory_order_relaxed);
  commit_width_sum.store(0, std::memory_order_relaxed);
  core_prefetch_bytes.store(0, std::memory_order_relaxed);
  max_issue_width.store(0, std::memory_order_relaxed);
  max_commit_width.store(0, std::memory_order_relaxed);
  critical_rob_hits.store(0, std::memory_order_relaxed);
  critical_misses.store(0, std::memory_order_relaxed);
  speculative_wait_ns.store(0, std::memory_order_relaxed);
  core_prefetch_reads.store(0, std::memory_order_relaxed);
  core_prefetch_arrived.store(0, std::memory_order_relaxed);
  core_prefetch_promoted.store(0, std::memory_order_relaxed);
  core_prefetch_stale.store(0, std::memory_order_relaxed);
  core_prefetch_queue_rejects.store(0, std::memory_order_relaxed);
  core_prefetch_waves.store(0, std::memory_order_relaxed);
  core_ready_waves.store(0, std::memory_order_relaxed);
  terminal_exact_cache_attempted_queries.store(0, std::memory_order_relaxed);
  terminal_exact_cache_issued_records.store(0, std::memory_order_relaxed);
  terminal_exact_cache_promoted_records.store(0, std::memory_order_relaxed);
  terminal_exact_cache_queue_rejects.store(0, std::memory_order_relaxed);
  terminal_exact_cache_miss_records.store(0, std::memory_order_relaxed);
  completion_score_batches.store(0, std::memory_order_relaxed);
  completion_score_candidates.store(0, std::memory_order_relaxed);
  frontier_reusable_certificates.store(0, std::memory_order_relaxed);
  frontier_streamed_candidate_runs.store(0, std::memory_order_relaxed);
  ordered_score_batches.store(0, std::memory_order_relaxed);
  ordered_score_candidates.store(0, std::memory_order_relaxed);
  ooo_bypassed_parents.store(0, std::memory_order_relaxed);
  frontier_reusable_prefix_ranks.store(0, std::memory_order_relaxed);
  frontier_reusable_full_prefix_certificates.store(
    0, std::memory_order_relaxed);
  frontier_reusable_issued_certificates.store(
    0, std::memory_order_relaxed);
  frontier_certificate_rejects.store(0, std::memory_order_relaxed);
  graph_dependency_rounds.store(0, std::memory_order_relaxed);
  graph_route_hits.store(0, std::memory_order_relaxed);
  graph_route_refreshes.store(0, std::memory_order_relaxed);
  centroid_route_publications.store(0, std::memory_order_relaxed);
  centroid_route_shard_updates.store(0, std::memory_order_relaxed);
  centroid_route_snapshot_skips.store(0, std::memory_order_relaxed);
  centroid_route_probe_reads.store(0, std::memory_order_relaxed);
  centroid_route_body_reads.store(0, std::memory_order_relaxed);
  centroid_route_unchanged_polls.store(0, std::memory_order_relaxed);
  centroid_route_query_retries.store(0, std::memory_order_relaxed);
  centroid_route_query_timeouts.store(0, std::memory_order_relaxed);
  exact_vector_reads.store(0, std::memory_order_relaxed);
  exact_snapshot_train_batches.store(0, std::memory_order_relaxed);
  exact_snapshot_train_fallbacks.store(0, std::memory_order_relaxed);
  dynamic_code_candidates.store(0, std::memory_order_relaxed);
  dynamic_code_reads.store(0, std::memory_order_relaxed);
  dynamic_code_read_bytes.store(0, std::memory_order_relaxed);
  dynamic_code_incarnation_rejects.store(0, std::memory_order_relaxed);
  dynamic_code_wait_ns.store(0, std::memory_order_relaxed);
  dynamic_code_cache_hits.store(0, std::memory_order_relaxed);
  dynamic_code_batch_deduplicated.store(0, std::memory_order_relaxed);
  dynamic_code_cache_publish_successes.store(0, std::memory_order_relaxed);
  dynamic_code_cache_publish_races.store(0, std::memory_order_relaxed);
  dynamic_code_cache_lookup_probe_exhaustions.store(
    0, std::memory_order_relaxed);
  dynamic_code_cache_publish_probe_exhaustions.store(
    0, std::memory_order_relaxed);
  dynamic_code_cache_lookup_probes.store(0, std::memory_order_relaxed);
  dynamic_code_cache_max_lookup_probes.store(0, std::memory_order_relaxed);
}

}  // namespace gpu_search
