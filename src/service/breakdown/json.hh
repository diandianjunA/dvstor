#pragma once

#include <string>

#include "nlohmann/json.hh"
#include "service/breakdown/aggregate.hh"
#include "vamana/vamana_node.hh"

namespace service::breakdown {

inline nlohmann::json aggregate_to_json(const Aggregate& aggregate) {
  using json = nlohmann::json;
  json out;
  out["operation"] = operation_name(aggregate.operation);
  out["count"] = aggregate.count;
  out["latency"] = {
    {"queue_wait_ns", aggregate.total_queue_wait_ns},
    {"service_ns", aggregate.total_service_ns},
    {"end_to_end_ns", aggregate.total_end_to_end_ns},
    {"mean_queue_wait_ns", aggregate.count == 0 ? 0 : aggregate.total_queue_wait_ns / aggregate.count},
    {"mean_service_ns", aggregate.count == 0 ? 0 : aggregate.total_service_ns / aggregate.count},
    {"mean_end_to_end_ns", aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count},
    {"p50_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.50)},
    {"p95_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.95)},
    {"p99_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)},
    {"p50_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.50)},
    {"p95_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.95)},
    {"p99_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.99)},
  };
  out["fine_grained_breakdown_observed"] = aggregate.fine_grained_breakdown_observed;

  if (!aggregate.fine_grained_breakdown_observed) {
    out["utilization"] = {
      {"device_utilization_observed", false},
      {"gpu_kernel_busy_ns", 0},
      {"gpu_kernel_busy_ratio", 0.0},
      {"gpu_kernel_idle_ratio", 0.0},
      {"rdma_completion_wait_ns", 0},
      {"rdma_completion_wait_ratio", 0.0},
      {"rdma_payload_bytes_per_service_s", 0.0},
    };
    return out;
  }

  const u64 cpu_total = aggregate.total_service_ns > (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          ? aggregate.total_service_ns - (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          : 0;

  json categories = json::object();
  categories["cpu_ns"] = cpu_total;
  categories["gpu_ns"] = aggregate.category_ns[static_cast<size_t>(Category::gpu)];
  categories["rdma_ns"] = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  categories["transfer_ns"] = aggregate.category_ns[static_cast<size_t>(Category::transfer)];
  out["breakdown"] = std::move(categories);

  // These values are deliberately separate from the category totals. RDMA
  // scopes measure software time blocked on completions, while CUDA events
  // measure device execution time only. Both are normalized by the sum of
  // query service windows, so use a single-query workload for a per-query
  // pipeline-utilization interpretation.
  const double service_ns = static_cast<double>(aggregate.total_service_ns);
  const bool observed = aggregate.device_utilization_observed;
  out["utilization"] = {
    {"device_utilization_observed", observed},
    {"gpu_kernel_busy_ns", observed ? aggregate.total_gpu_kernel_ns : 0},
    {"gpu_kernel_busy_ratio", !observed || service_ns == 0.0 ? 0.0 :
      static_cast<double>(aggregate.total_gpu_kernel_ns) / service_ns},
    {"gpu_kernel_idle_ratio", !observed || service_ns == 0.0 ? 0.0 :
      std::max(0.0, 1.0 - static_cast<double>(aggregate.total_gpu_kernel_ns) / service_ns)},
    {"rdma_completion_wait_ns", observed ? aggregate.total_rdma_wait_ns : 0},
    {"rdma_completion_wait_ratio", !observed || service_ns == 0.0 ? 0.0 :
      static_cast<double>(aggregate.total_rdma_wait_ns) / service_ns},
    {"rdma_payload_bytes_per_service_s", !observed || service_ns == 0.0 ? 0.0 :
      static_cast<double>(aggregate.counters.rdma_read_bytes) * 1e9 / service_ns},
  };

  json sub = json::object();
  for (size_t c = 0; c < kCategoryCount; ++c) {
    sub[std::string{kCategoryNames[c]}] = json::object();
  }
  for (size_t i = 0; i < kSubcategoryCount; ++i) {
    const auto subcat = static_cast<Subcategory>(i);
    sub[std::string{kCategoryNames[static_cast<size_t>(parent_category(subcat))]}]
       [std::string{kSubcategoryNames[i]}] = aggregate.subcategory_ns[i];
  }
  if (aggregate.operation == Operation::query) {
    sub["cpu_ns"]["cpu_query_runtime_overhead_ns"] = aggregate.cpu_other_ns();
  } else {
    sub["cpu_ns"]["cpu_insert_runtime_overhead_ns"] = aggregate.cpu_other_ns();
  }
  out["sub_breakdown"] = std::move(sub);

  out["counters"] = {
    {"rdma_read_bytes", aggregate.counters.rdma_read_bytes},
    {"rdma_write_bytes", aggregate.counters.rdma_write_bytes},
    {"rdma_read_ops", aggregate.counters.rdma_read_ops},
    {"rdma_write_ops", aggregate.counters.rdma_write_ops},
    {"rdma_read_avg_bytes",
     aggregate.counters.rdma_read_ops == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.rdma_read_bytes) /
           static_cast<double>(aggregate.counters.rdma_read_ops)},
    {"rdma_write_avg_bytes",
     aggregate.counters.rdma_write_ops == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.rdma_write_bytes) /
           static_cast<double>(aggregate.counters.rdma_write_ops)},
    {"neighbor_rdma_bytes", aggregate.counters.neighbor_rdma_bytes},
    {"vector_rdma_bytes", aggregate.counters.vector_rdma_bytes},
    {"neighbor_rdma_read_ops", aggregate.counters.neighbor_rdma_read_ops},
    {"vector_rdma_read_ops", aggregate.counters.vector_rdma_read_ops},
    {"vector_rdma_batch_calls", aggregate.counters.vector_rdma_batch_calls},
    {"vector_rdma_cqes", aggregate.counters.vector_rdma_cqes},
    {"vector_rdma_active_nodes", aggregate.counters.vector_rdma_active_nodes},
    {"vector_rdma_active_qps", aggregate.counters.vector_rdma_active_qps},
    {"vector_rdma_mean_active_nodes_per_batch",
     aggregate.counters.vector_rdma_batch_calls == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_active_nodes) /
           static_cast<double>(aggregate.counters.vector_rdma_batch_calls)},
    {"vector_rdma_mean_active_qps_per_batch",
     aggregate.counters.vector_rdma_batch_calls == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_active_qps) /
           static_cast<double>(aggregate.counters.vector_rdma_batch_calls)},
    {"vector_rdma_mean_chain_wrs",
     aggregate.counters.vector_rdma_cqes == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_chain_wrs) /
           static_cast<double>(aggregate.counters.vector_rdma_cqes)},
    {"vector_rdma_max_chain_wrs", aggregate.counters.vector_rdma_max_chain_wrs},
    {"vector_rdma_qp_high_water_wrs",
     aggregate.counters.vector_rdma_qp_high_water_wrs},
    {"vector_rdma_credit_waits", aggregate.counters.vector_rdma_credit_waits},
    {"vector_rdma_credit_wait_ns", aggregate.counters.vector_rdma_credit_wait_ns},
    {"vector_rdma_completion_token_waits",
     aggregate.counters.vector_rdma_completion_token_waits},
    {"vector_rdma_post_send_calls", aggregate.counters.vector_rdma_post_send_calls},
    {"vector_rdma_post_send_retries", aggregate.counters.vector_rdma_post_send_retries},
    {"vector_rdma_post_send_errors", aggregate.counters.vector_rdma_post_send_errors},
    {"vector_rdma_reads_per_batch",
     aggregate.counters.vector_rdma_batch_calls == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_read_ops) /
           static_cast<double>(aggregate.counters.vector_rdma_batch_calls)},
    {"vector_rdma_reads_per_cqe",
     aggregate.counters.vector_rdma_cqes == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_read_ops) /
           static_cast<double>(aggregate.counters.vector_rdma_cqes)},
    {"neighbor_rdma_read_avg_bytes",
     aggregate.counters.neighbor_rdma_read_ops == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.neighbor_rdma_bytes) /
           static_cast<double>(aggregate.counters.neighbor_rdma_read_ops)},
    {"vector_rdma_read_avg_bytes",
     aggregate.counters.vector_rdma_read_ops == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.vector_rdma_bytes) /
           static_cast<double>(aggregate.counters.vector_rdma_read_ops)},
    {"h2d_bytes", aggregate.counters.h2d_bytes},
    {"d2h_bytes", aggregate.counters.d2h_bytes},
    {"l2_kernels", aggregate.counters.l2_kernels},
    {"prune_kernels", aggregate.counters.prune_kernels},
    {"exact_reranks", aggregate.counters.exact_reranks},
    {"rabitq_l0_candidates", aggregate.counters.rabitq_l0_candidates},
    {"rabitq_cache_misses", aggregate.counters.rabitq_cache_misses},
    {"rabitq_l1_candidates", aggregate.counters.rabitq_l1_candidates},
    {"rabitq_l2_candidates", aggregate.counters.rabitq_l2_candidates},
    {"rabitq_forced_widen", aggregate.counters.rabitq_forced_widen},
    {"rabitq_audit_expansions", aggregate.counters.rabitq_audit_expansions},
    {"rabitq_audit_candidates", aggregate.counters.rabitq_audit_candidates},
    {"rabitq_safe_skips", aggregate.counters.rabitq_safe_skips},
    {"rabitq_safe_skip_vector_bytes",
     aggregate.counters.rabitq_safe_skips * VamanaNode::vector_bytes()},
    {"rabitq_exact_fallbacks", aggregate.counters.rabitq_exact_fallbacks},
    {"rabitq_prefetch_issued", aggregate.counters.rabitq_prefetch_issued},
    {"rabitq_prefetch_hits", aggregate.counters.rabitq_prefetch_hits},
    {"rabitq_prefetch_misses", aggregate.counters.rabitq_prefetch_misses},
    {"rabitq_prefetch_disabled_queries",
     aggregate.counters.rabitq_prefetch_disabled_queries},
    {"credit_rounds", aggregate.counters.credit_rounds},
    {"credit_expansions_issued", aggregate.counters.credit_expansions_issued},
    {"credit_precommit_expansions", aggregate.counters.credit_precommit_expansions},
    {"credit_postcommit_expansions", aggregate.counters.credit_postcommit_expansions},
    {"credit_grow_events", aggregate.counters.credit_grow_events},
    {"credit_shrink_events", aggregate.counters.credit_shrink_events},
    {"credit_credit_stalls", aggregate.counters.credit_credit_stalls},
    {"credit_no_progress_rounds", aggregate.counters.credit_no_progress_rounds},
    {"credit_underfilled_rounds", aggregate.counters.credit_underfilled_rounds},
    {"credit_overfilled_rounds", aggregate.counters.credit_overfilled_rounds},
    {"credit_cost_guard_events", aggregate.counters.credit_cost_guard_events},
    {"credit_cost_growth_blocked", aggregate.counters.credit_cost_growth_blocked},
    {"credit_cost_baseline_samples", aggregate.counters.credit_cost_baseline_samples},
    {"credit_mean_issue_k",
     aggregate.counters.credit_rounds == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.credit_expansions_issued) /
           static_cast<double>(aggregate.counters.credit_rounds)},
    {"credit_precommit_ratio",
     aggregate.counters.credit_expansions_issued == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.credit_precommit_expansions) /
           static_cast<double>(aggregate.counters.credit_expansions_issued)},
    {"storage_owner_anchor_hints", aggregate.counters.storage_owner_anchor_hints},
    {"storage_owner_anchor_valid_hints", aggregate.counters.storage_owner_anchor_valid_hints},
    {"storage_owner_anchor_expansions", aggregate.counters.storage_owner_anchor_expansions},
    {"storage_owner_anchor_remote_expansions",
     aggregate.counters.storage_owner_anchor_remote_expansions},
    {"storage_owner_anchor_fallbacks", aggregate.counters.storage_owner_anchor_fallbacks},
    {"storage_owner_anchor_audits", aggregate.counters.storage_owner_anchor_audits},
    {"storage_owner_anchor_audit_failures",
     aggregate.counters.storage_owner_anchor_audit_failures},
    {"rabitq_prefetch_hit_ratio",
     aggregate.counters.rabitq_prefetch_issued == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.rabitq_prefetch_hits) /
           static_cast<double>(aggregate.counters.rabitq_prefetch_issued)},
    {"rabitq_local_scores", aggregate.counters.rabitq_l0_candidates},
    {"rabitq_gate_passes", aggregate.counters.rabitq_l1_candidates},
    {"rabitq_exact_vector_reads", aggregate.counters.rabitq_l2_candidates},
    {"visited_nodes", aggregate.counters.visited_nodes},
    {"visited_neighborlists", aggregate.counters.visited_neighborlists},
    {"remote_allocations", aggregate.counters.remote_allocations},
    {"overflow_prunes", aggregate.counters.overflow_prunes},
    {"overflow_prune_candidates", aggregate.counters.overflow_prune_candidates},
    {"overflow_prune_avg_candidates",
     aggregate.counters.overflow_prunes == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.overflow_prune_candidates) /
           static_cast<double>(aggregate.counters.overflow_prunes)},
    {"overflow_prune_max_candidates", aggregate.counters.overflow_prune_max_candidates},
    {"overflow_prune_pair_checks_upper_bound", aggregate.counters.overflow_prune_pair_checks_upper_bound},
    {"overflow_prune_global_load_bytes_upper_bound",
     aggregate.counters.overflow_prune_global_load_bytes_upper_bound},
    {"overflow_prune_kernel_blocks", aggregate.counters.overflow_prune_kernel_blocks},
    {"overflow_prune_avg_kernel_blocks",
     aggregate.counters.overflow_prunes == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.overflow_prune_kernel_blocks) /
           static_cast<double>(aggregate.counters.overflow_prunes)},
    {"overflow_prune_kernel_threads", aggregate.counters.overflow_prune_kernel_threads},
    {"overflow_prune_avg_kernel_threads",
     aggregate.counters.overflow_prunes == 0
       ? 0.0
       : static_cast<double>(aggregate.counters.overflow_prune_kernel_threads) /
           static_cast<double>(aggregate.counters.overflow_prunes)},
    {"overflow_prune_max_kernel_threads", aggregate.counters.overflow_prune_max_kernel_threads},
    {"query_rdma_to_staging_bytes", aggregate.counters.query_rdma_to_staging_bytes},
    {"query_host_staging_fallback_bytes", aggregate.counters.query_host_staging_fallback_bytes},
    {"lock_attempts", aggregate.lock_attempts},
    {"lock_retries", aggregate.lock_retries},
    {"cas_failures", aggregate.cas_failures},
  };
  return out;
}


inline nlohmann::json report_to_json(const Report& report) {
  nlohmann::json out = nlohmann::json::object();
  if (report.has_query()) {
    out["query_breakdown"] = aggregate_to_json(report.query);
  }
  if (report.has_insert()) {
    out["insert_breakdown"] = aggregate_to_json(report.insert);
  }
  return out;
}

}  // namespace service::breakdown
