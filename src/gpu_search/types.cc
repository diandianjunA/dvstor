#include "gpu_search/types.hh"

namespace gpu_search {

TelemetrySnapshot Telemetry::snapshot() const {
  return {
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
    .feedback_hunger_queries =
      feedback_hunger_queries.load(std::memory_order_relaxed),
    .expansion_sum_selected_parents =
      expansion_sum_selected_parents.load(std::memory_order_relaxed),
    .expansion_sum_feedback_horizon =
      expansion_sum_feedback_horizon.load(std::memory_order_relaxed),
    .expansion_sum_hardware_credit_tiles =
      expansion_sum_hardware_credit_tiles.load(std::memory_order_relaxed),
    .expansion_minimum_selected_batch =
      expansion_minimum_selected_batch.load(std::memory_order_relaxed),
    .expansion_maximum_selected_batch =
      expansion_maximum_selected_batch.load(std::memory_order_relaxed),
    .expansion_minimum_feedback_horizon =
      expansion_minimum_feedback_horizon.load(std::memory_order_relaxed),
    .expansion_maximum_feedback_horizon =
      expansion_maximum_feedback_horizon.load(std::memory_order_relaxed),
    .expansion_extra_parents =
      expansion_extra_parents.load(std::memory_order_relaxed),
    .expansion_qp_lease_claims =
      expansion_qp_lease_claims.load(std::memory_order_relaxed),
    .expansion_qp_lease_rejects =
      expansion_qp_lease_rejects.load(std::memory_order_relaxed),
    .expansion_qp_lease_rollbacks =
      expansion_qp_lease_rollbacks.load(std::memory_order_relaxed),
    .expansion_compute_allowance_tiles =
      expansion_compute_allowance_tiles.load(std::memory_order_relaxed),
    .expansion_marginal_probe_passes =
      expansion_marginal_probe_passes.load(std::memory_order_relaxed),
    .expansion_marginal_probe_failures =
      expansion_marginal_probe_failures.load(std::memory_order_relaxed),
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
  };
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
  gpu_rdma_wait_ns.store(0, std::memory_order_relaxed);
  gpu_graph_validation_ns.store(0, std::memory_order_relaxed);
  gpu_neighbor_decode_ns.store(0, std::memory_order_relaxed);
  gpu_pq_score_ns.store(0, std::memory_order_relaxed);
  gpu_visited_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_prepare_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_sort_ns.store(0, std::memory_order_relaxed);
  gpu_beam_merge_materialize_ns.store(0, std::memory_order_relaxed);
  feedback_hunger_queries.store(0, std::memory_order_relaxed);
  expansion_sum_selected_parents.store(0, std::memory_order_relaxed);
  expansion_sum_feedback_horizon.store(0, std::memory_order_relaxed);
  expansion_sum_hardware_credit_tiles.store(0, std::memory_order_relaxed);
  expansion_minimum_selected_batch.store(
    UINT64_MAX, std::memory_order_relaxed);
  expansion_maximum_selected_batch.store(0, std::memory_order_relaxed);
  expansion_minimum_feedback_horizon.store(
    UINT64_MAX, std::memory_order_relaxed);
  expansion_maximum_feedback_horizon.store(0, std::memory_order_relaxed);
  expansion_extra_parents.store(0, std::memory_order_relaxed);
  expansion_qp_lease_claims.store(0, std::memory_order_relaxed);
  expansion_qp_lease_rejects.store(0, std::memory_order_relaxed);
  expansion_qp_lease_rollbacks.store(0, std::memory_order_relaxed);
  expansion_compute_allowance_tiles.store(0, std::memory_order_relaxed);
  expansion_marginal_probe_passes.store(0, std::memory_order_relaxed);
  expansion_marginal_probe_failures.store(0, std::memory_order_relaxed);
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
