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
    .rdma_read_ops = rdma_read_ops.load(std::memory_order_relaxed),
    .rdma_read_bytes = rdma_read_bytes.load(std::memory_order_relaxed),
    .rdma_merged_requests = rdma_merged_requests.load(std::memory_order_relaxed),
    .direct_path_failures = direct_path_failures.load(std::memory_order_relaxed),
    .graph_page_requests = graph_page_requests.load(std::memory_order_relaxed),
    .graph_read_retries = graph_read_retries.load(std::memory_order_relaxed),
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
    .exact_vector_reads = exact_vector_reads.load(std::memory_order_relaxed),
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
  graph_read_retries.store(0, std::memory_order_relaxed);
  graph_dependency_rounds.store(0, std::memory_order_relaxed);
  graph_route_hits.store(0, std::memory_order_relaxed);
  graph_route_refreshes.store(0, std::memory_order_relaxed);
  centroid_route_publications.store(0, std::memory_order_relaxed);
  centroid_route_shard_updates.store(0, std::memory_order_relaxed);
  centroid_route_snapshot_skips.store(0, std::memory_order_relaxed);
  centroid_route_probe_reads.store(0, std::memory_order_relaxed);
  centroid_route_body_reads.store(0, std::memory_order_relaxed);
  centroid_route_unchanged_polls.store(0, std::memory_order_relaxed);
  exact_vector_reads.store(0, std::memory_order_relaxed);
}

}  // namespace gpu_search
