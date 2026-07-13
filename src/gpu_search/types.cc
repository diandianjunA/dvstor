#include "gpu_search/types.hh"

namespace gpu_search {

TelemetrySnapshot Telemetry::snapshot() const {
  return {
    .gpu_memory_explicit_bytes = gpu_memory_explicit_bytes.load(std::memory_order_relaxed),
    .gpu_memory_base_pq_bytes = gpu_memory_base_pq_bytes.load(std::memory_order_relaxed),
    .gpu_memory_route_graph_bytes = gpu_memory_route_graph_bytes.load(std::memory_order_relaxed),
    .gpu_memory_delta_reserved_bytes = gpu_memory_delta_reserved_bytes.load(std::memory_order_relaxed),
    .gpu_memory_graph_cache_bytes = gpu_memory_graph_cache_bytes.load(std::memory_order_relaxed),
    .gpu_memory_exact_cache_bytes = gpu_memory_exact_cache_bytes.load(std::memory_order_relaxed),
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
    .rdma_read_ops = rdma_read_ops.load(std::memory_order_relaxed),
    .rdma_read_bytes = rdma_read_bytes.load(std::memory_order_relaxed),
    .rdma_merged_requests = rdma_merged_requests.load(std::memory_order_relaxed),
    .direct_path_failures = direct_path_failures.load(std::memory_order_relaxed),
    .graph_page_requests = graph_page_requests.load(std::memory_order_relaxed),
    .graph_dependency_rounds = graph_dependency_rounds.load(std::memory_order_relaxed),
    .graph_page_cache_hits = graph_page_cache_hits.load(std::memory_order_relaxed),
    .graph_route_hits = graph_route_hits.load(std::memory_order_relaxed),
    .graph_route_refreshes = graph_route_refreshes.load(std::memory_order_relaxed),
    .graph_cache_invalidations = graph_cache_invalidations.load(std::memory_order_relaxed),
    .exact_vector_reads = exact_vector_reads.load(std::memory_order_relaxed),
    .exact_vector_cache_hits = exact_vector_cache_hits.load(std::memory_order_relaxed),
    .delta_queries = delta_queries.load(std::memory_order_relaxed),
    .mutations_published = mutations_published.load(std::memory_order_relaxed),
    .delta_publications = delta_publications.load(std::memory_order_relaxed),
    .delta_compactions = delta_compactions.load(std::memory_order_relaxed),
    .delta_entries_retired = delta_entries_retired.load(std::memory_order_relaxed),
    .storage_reclaim_ack_writes = storage_reclaim_ack_writes.load(std::memory_order_relaxed),
    .storage_reclaim_ack_sequence = storage_reclaim_ack_sequence.load(std::memory_order_relaxed),
    .delta_live_entries = delta_live_entries.load(std::memory_order_relaxed),
    .delta_physical_entries = delta_physical_entries.load(std::memory_order_relaxed),
    .delta_mutable_entries = delta_mutable_entries.load(std::memory_order_relaxed),
    .delta_durable_entries = delta_durable_entries.load(std::memory_order_relaxed),
    .mutation_capacity_rejections = mutation_capacity_rejections.load(std::memory_order_relaxed),
    .mutation_capacity_reserved = mutation_capacity_reserved.load(std::memory_order_relaxed),
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
  rdma_read_ops.store(0, std::memory_order_relaxed);
  rdma_read_bytes.store(0, std::memory_order_relaxed);
  rdma_merged_requests.store(0, std::memory_order_relaxed);
  direct_path_failures.store(0, std::memory_order_relaxed);
  graph_page_requests.store(0, std::memory_order_relaxed);
  graph_dependency_rounds.store(0, std::memory_order_relaxed);
  graph_page_cache_hits.store(0, std::memory_order_relaxed);
  graph_route_hits.store(0, std::memory_order_relaxed);
  graph_route_refreshes.store(0, std::memory_order_relaxed);
  graph_cache_invalidations.store(0, std::memory_order_relaxed);
  exact_vector_reads.store(0, std::memory_order_relaxed);
  exact_vector_cache_hits.store(0, std::memory_order_relaxed);
  delta_queries.store(0, std::memory_order_relaxed);
  mutations_published.store(0, std::memory_order_relaxed);
  delta_publications.store(0, std::memory_order_relaxed);
  delta_compactions.store(0, std::memory_order_relaxed);
  delta_entries_retired.store(0, std::memory_order_relaxed);
  storage_reclaim_ack_writes.store(0, std::memory_order_relaxed);
  delta_live_entries.store(0, std::memory_order_relaxed);
  delta_physical_entries.store(0, std::memory_order_relaxed);
  delta_mutable_entries.store(0, std::memory_order_relaxed);
  delta_durable_entries.store(0, std::memory_order_relaxed);
  mutation_capacity_rejections.store(0, std::memory_order_relaxed);
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
