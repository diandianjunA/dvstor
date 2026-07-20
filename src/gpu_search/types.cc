#include "gpu_search/types.hh"

namespace gpu_search {

TelemetrySnapshot Telemetry::snapshot() const {
  // A zero reservation observed with acquire means every mutation publisher
  // whose reservation contributed to that zero has completed its telemetry
  // updates. release_mutation_capacity() serializes publishers under the
  // delta mutex and publishes the final reservation count with release.
  const u64 reserved = mutation_capacity_reserved.load(
    std::memory_order_acquire);
  return {
    .gpu_memory_explicit_bytes = gpu_memory_explicit_bytes.load(std::memory_order_relaxed),
    .gpu_memory_base_pq_bytes = gpu_memory_base_pq_bytes.load(std::memory_order_relaxed),
    .gpu_memory_resident_pq_bytes =
      gpu_memory_resident_pq_bytes.load(std::memory_order_relaxed),
    .gpu_memory_route_graph_bytes = gpu_memory_route_graph_bytes.load(std::memory_order_relaxed),
    .gpu_memory_delta_reserved_bytes = gpu_memory_delta_reserved_bytes.load(std::memory_order_relaxed),
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
    .gpu_delta_scan_ns = gpu_delta_scan_ns.load(std::memory_order_relaxed),
    .rdma_read_ops = rdma_read_ops.load(std::memory_order_relaxed),
    .rdma_read_bytes = rdma_read_bytes.load(std::memory_order_relaxed),
    .rdma_merged_requests = rdma_merged_requests.load(std::memory_order_relaxed),
    .direct_path_failures = direct_path_failures.load(std::memory_order_relaxed),
    .graph_page_requests = graph_page_requests.load(std::memory_order_relaxed),
    .graph_read_retries = graph_read_retries.load(std::memory_order_relaxed),
    .graph_dependency_rounds = graph_dependency_rounds.load(std::memory_order_relaxed),
    .graph_route_hits = graph_route_hits.load(std::memory_order_relaxed),
    .graph_route_refreshes = graph_route_refreshes.load(std::memory_order_relaxed),
    .dynamic_route_publications =
      dynamic_route_publications.load(std::memory_order_relaxed),
    .dynamic_route_slot_updates =
      dynamic_route_slot_updates.load(std::memory_order_relaxed),
    .dynamic_route_live_slots =
      dynamic_route_live_slots.load(std::memory_order_relaxed),
    .dynamic_route_snapshot_skips =
      dynamic_route_snapshot_skips.load(std::memory_order_relaxed),
    .graph_route_invalidations = graph_route_invalidations.load(std::memory_order_relaxed),
    .exact_vector_reads = exact_vector_reads.load(std::memory_order_relaxed),
    .delta_queries = delta_queries.load(std::memory_order_relaxed),
    .delta_scan_records = delta_scan_records.load(std::memory_order_relaxed),
    .delta_scan_scored = delta_scan_scored.load(std::memory_order_relaxed),
    .delta_scan_truncated_buckets =
      delta_scan_truncated_buckets.load(std::memory_order_relaxed),
    .mutations_published = mutations_published.load(std::memory_order_relaxed),
    .delta_publications = delta_publications.load(std::memory_order_relaxed),
    .delta_reclaim_batches = delta_reclaim_batches.load(std::memory_order_relaxed),
    .delta_entries_retired = delta_entries_retired.load(std::memory_order_relaxed),
    .storage_reclaim_ack_writes = storage_reclaim_ack_writes.load(std::memory_order_relaxed),
    .storage_reclaim_ack_sequence = storage_reclaim_ack_sequence.load(std::memory_order_relaxed),
    .delta_live_entries = delta_live_entries.load(std::memory_order_relaxed),
    .delta_physical_entries = delta_physical_entries.load(std::memory_order_relaxed),
    .delta_mutable_entries = delta_mutable_entries.load(std::memory_order_relaxed),
    .delta_durable_entries = delta_durable_entries.load(std::memory_order_relaxed),
    .resident_pq_capacity = resident_pq_capacity.load(std::memory_order_relaxed),
    .resident_pq_entries = resident_pq_entries.load(std::memory_order_relaxed),
    .resident_pq_peak_entries = resident_pq_peak_entries.load(std::memory_order_relaxed),
    .resident_pq_reclaimed = resident_pq_reclaimed.load(std::memory_order_relaxed),
    .mutation_capacity_rejections = mutation_capacity_rejections.load(std::memory_order_relaxed),
    .mutation_capacity_wait_events = mutation_capacity_wait_events.load(std::memory_order_relaxed),
    .mutation_capacity_wait_ns = mutation_capacity_wait_ns.load(std::memory_order_relaxed),
    .mutation_capacity_reserved = reserved,
    .mutation_capacity_reserved_max = mutation_capacity_reserved_max.load(std::memory_order_relaxed),
    .visibility_ns_total = visibility_ns_total.load(std::memory_order_relaxed),
    .visibility_ns_max = visibility_ns_max.load(std::memory_order_relaxed),
    .publication_queue_ns_total = publication_queue_ns_total.load(std::memory_order_relaxed),
    .publication_prepare_ns_total = publication_prepare_ns_total.load(std::memory_order_relaxed),
    .publication_command_ns_total = publication_command_ns_total.load(std::memory_order_relaxed),
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
  gpu_delta_scan_ns.store(0, std::memory_order_relaxed);
  rdma_read_ops.store(0, std::memory_order_relaxed);
  rdma_read_bytes.store(0, std::memory_order_relaxed);
  rdma_merged_requests.store(0, std::memory_order_relaxed);
  direct_path_failures.store(0, std::memory_order_relaxed);
  graph_page_requests.store(0, std::memory_order_relaxed);
  graph_read_retries.store(0, std::memory_order_relaxed);
  graph_dependency_rounds.store(0, std::memory_order_relaxed);
  graph_route_hits.store(0, std::memory_order_relaxed);
  graph_route_refreshes.store(0, std::memory_order_relaxed);
  dynamic_route_publications.store(0, std::memory_order_relaxed);
  dynamic_route_slot_updates.store(0, std::memory_order_relaxed);
  dynamic_route_snapshot_skips.store(0, std::memory_order_relaxed);
  graph_route_invalidations.store(0, std::memory_order_relaxed);
  exact_vector_reads.store(0, std::memory_order_relaxed);
  delta_queries.store(0, std::memory_order_relaxed);
  delta_scan_records.store(0, std::memory_order_relaxed);
  delta_scan_scored.store(0, std::memory_order_relaxed);
  delta_scan_truncated_buckets.store(0, std::memory_order_relaxed);
  mutations_published.store(0, std::memory_order_relaxed);
  delta_publications.store(0, std::memory_order_relaxed);
  delta_reclaim_batches.store(0, std::memory_order_relaxed);
  delta_entries_retired.store(0, std::memory_order_relaxed);
  storage_reclaim_ack_writes.store(0, std::memory_order_relaxed);
  delta_live_entries.store(0, std::memory_order_relaxed);
  delta_physical_entries.store(0, std::memory_order_relaxed);
  delta_mutable_entries.store(0, std::memory_order_relaxed);
  delta_durable_entries.store(0, std::memory_order_relaxed);
  resident_pq_peak_entries.store(
    resident_pq_entries.load(std::memory_order_relaxed),
    std::memory_order_relaxed);
  resident_pq_reclaimed.store(0, std::memory_order_relaxed);
  mutation_capacity_rejections.store(0, std::memory_order_relaxed);
  mutation_capacity_wait_events.store(0, std::memory_order_relaxed);
  mutation_capacity_wait_ns.store(0, std::memory_order_relaxed);
  mutation_capacity_reserved_max.store(
    mutation_capacity_reserved.load(std::memory_order_relaxed),
    std::memory_order_relaxed);
  visibility_ns_total.store(0, std::memory_order_relaxed);
  visibility_ns_max.store(0, std::memory_order_relaxed);
  publication_queue_ns_total.store(0, std::memory_order_relaxed);
  publication_prepare_ns_total.store(0, std::memory_order_relaxed);
  publication_command_ns_total.store(0, std::memory_order_relaxed);
}

}  // namespace gpu_search
