#include "gpu_search/types.hh"

namespace gpu_search {

TelemetrySnapshot Telemetry::snapshot() const {
  return {
    .queries_submitted = queries_submitted.load(std::memory_order_relaxed),
    .queries_completed = queries_completed.load(std::memory_order_relaxed),
    .batches = batches.load(std::memory_order_relaxed),
    .batch_queries = batch_queries.load(std::memory_order_relaxed),
    .submission_wait_ns = submission_wait_ns.load(std::memory_order_relaxed),
    .completion_wait_ns = completion_wait_ns.load(std::memory_order_relaxed),
    .gpu_active_ns = gpu_active_ns.load(std::memory_order_relaxed),
    .rdma_read_ops = rdma_read_ops.load(std::memory_order_relaxed),
    .rdma_read_bytes = rdma_read_bytes.load(std::memory_order_relaxed),
    .rdma_merged_requests = rdma_merged_requests.load(std::memory_order_relaxed),
    .direct_path_failures = direct_path_failures.load(std::memory_order_relaxed),
    .graph_page_requests = graph_page_requests.load(std::memory_order_relaxed),
    .graph_page_cache_hits = graph_page_cache_hits.load(std::memory_order_relaxed),
    .exact_vector_reads = exact_vector_reads.load(std::memory_order_relaxed),
    .exact_vector_cache_hits = exact_vector_cache_hits.load(std::memory_order_relaxed),
    .delta_queries = delta_queries.load(std::memory_order_relaxed),
    .mutations_published = mutations_published.load(std::memory_order_relaxed),
    .delta_compactions = delta_compactions.load(std::memory_order_relaxed),
    .base_entries_merged = base_entries_merged.load(std::memory_order_relaxed),
    .delta_live_entries = delta_live_entries.load(std::memory_order_relaxed),
    .visibility_ns_total = visibility_ns_total.load(std::memory_order_relaxed),
    .visibility_ns_max = visibility_ns_max.load(std::memory_order_relaxed),
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
  rdma_read_ops.store(0, std::memory_order_relaxed);
  rdma_read_bytes.store(0, std::memory_order_relaxed);
  rdma_merged_requests.store(0, std::memory_order_relaxed);
  direct_path_failures.store(0, std::memory_order_relaxed);
  graph_page_requests.store(0, std::memory_order_relaxed);
  graph_page_cache_hits.store(0, std::memory_order_relaxed);
  exact_vector_reads.store(0, std::memory_order_relaxed);
  exact_vector_cache_hits.store(0, std::memory_order_relaxed);
  delta_queries.store(0, std::memory_order_relaxed);
  mutations_published.store(0, std::memory_order_relaxed);
  delta_compactions.store(0, std::memory_order_relaxed);
  base_entries_merged.store(0, std::memory_order_relaxed);
  delta_live_entries.store(0, std::memory_order_relaxed);
  visibility_ns_total.store(0, std::memory_order_relaxed);
  visibility_ns_max.store(0, std::memory_order_relaxed);
}

}  // namespace gpu_search
