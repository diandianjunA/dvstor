#pragma once

#include <string>

#include "nlohmann/json.hh"
#include "service/breakdown/aggregate.hh"

namespace service::breakdown {

inline nlohmann::json aggregate_to_json(const Aggregate& aggregate) {
  using json = nlohmann::json;
  json output;
  output["operation"] = operation_name(aggregate.operation);
  output["count"] = aggregate.count;
  output["latency"] = {
    {"queue_wait_ns", aggregate.total_queue_wait_ns},
    {"service_ns", aggregate.total_service_ns},
    {"end_to_end_ns", aggregate.total_end_to_end_ns},
    {"mean_queue_wait_ns", aggregate.count == 0 ? 0 :
      aggregate.total_queue_wait_ns / aggregate.count},
    {"mean_service_ns", aggregate.count == 0 ? 0 :
      aggregate.total_service_ns / aggregate.count},
    {"mean_end_to_end_ns", aggregate.count == 0 ? 0 :
      aggregate.total_end_to_end_ns / aggregate.count},
    {"p50_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.50)},
    {"p95_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.95)},
    {"p99_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)},
    {"p50_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.50)},
    {"p95_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.95)},
    {"p99_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.99)},
  };
  output["fine_grained_breakdown_observed"] =
    aggregate.fine_grained_breakdown_observed;
  if (!aggregate.fine_grained_breakdown_observed) return output;

  const u64 rdma_ns = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  output["breakdown"] = {
    {"cpu_ns", aggregate.total_service_ns > rdma_ns
      ? aggregate.total_service_ns - rdma_ns : 0},
    {"rdma_ns", rdma_ns},
  };

  json subcategories = json::object();
  for (size_t category = 0; category < kCategoryCount; ++category) {
    subcategories[std::string{kCategoryNames[category]}] = json::object();
  }
  for (size_t index = 0; index < kSubcategoryCount; ++index) {
    const auto subcategory = static_cast<Subcategory>(index);
    subcategories[std::string{kCategoryNames[
      static_cast<size_t>(parent_category(subcategory))]}]
      [std::string{kSubcategoryNames[index]}] = aggregate.subcategory_ns[index];
  }
  subcategories["cpu_ns"][aggregate.operation == Operation::query
    ? "cpu_query_runtime_overhead_ns" : "cpu_insert_runtime_overhead_ns"] =
      aggregate.cpu_other_ns();
  output["sub_breakdown"] = std::move(subcategories);
  output["counters"] = {
    {"storage_owner_anchor_hints", aggregate.counters.storage_owner_anchor_hints},
    {"storage_owner_anchor_valid_hints",
     aggregate.counters.storage_owner_anchor_valid_hints},
    {"storage_owner_anchor_expansions",
     aggregate.counters.storage_owner_anchor_expansions},
    {"storage_owner_anchor_remote_expansions",
     aggregate.counters.storage_owner_anchor_remote_expansions},
  };
  return output;
}

inline nlohmann::json report_to_json(const Report& report) {
  nlohmann::json output = nlohmann::json::object();
  if (report.has_query()) output["query_breakdown"] = aggregate_to_json(report.query);
  if (report.has_insert()) output["insert_breakdown"] = aggregate_to_json(report.insert);
  return output;
}

}  // namespace service::breakdown
