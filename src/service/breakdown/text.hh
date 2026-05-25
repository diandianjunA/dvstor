#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "service/breakdown/aggregate.hh"

namespace service::breakdown {

inline std::string aggregate_text_summary(const Aggregate& aggregate) {
  std::ostringstream os;
  os << operation_name(aggregate.operation) << " breakdown\n";
  os << "  count: " << aggregate.count << '\n';
  os << "  latency_ms: mean=" << ns_to_ms(aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count)
     << " p50=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.50))
     << " p95=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.95))
     << " p99=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)) << '\n';

  const u64 cpu_total = aggregate.total_service_ns > (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                      aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          ? aggregate.total_service_ns - (aggregate.category_ns[static_cast<size_t>(Category::gpu)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::rdma)] +
                                                          aggregate.category_ns[static_cast<size_t>(Category::transfer)])
                          : 0;

  std::vector<std::pair<std::string, u64>> ranked = {
    {"cpu_ns", cpu_total},
    {"gpu_ns", aggregate.category_ns[static_cast<size_t>(Category::gpu)]},
    {"rdma_ns", aggregate.category_ns[static_cast<size_t>(Category::rdma)]},
    {"transfer_ns", aggregate.category_ns[static_cast<size_t>(Category::transfer)]},
  };
  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  os << "  top_categories:\n";
  for (const auto& [name, value] : ranked) {
    const double ratio = aggregate.total_service_ns == 0
                           ? 0.0
                           : static_cast<double>(value) / static_cast<double>(aggregate.total_service_ns);
    os << "    " << name << ": " << ns_to_ms(value) << " ms (" << ratio * 100.0 << "%)\n";
  }
  return os.str();
}

}  // namespace service::breakdown
