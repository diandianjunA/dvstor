#pragma once

#include <sstream>
#include <string>

#include "service/breakdown/aggregate.hh"

namespace service::breakdown {

inline std::string aggregate_text_summary(const Aggregate& aggregate) {
  std::ostringstream output;
  output << operation_name(aggregate.operation) << " breakdown\n";
  output << "  count: " << aggregate.count << '\n';
  output << "  latency_ms: mean="
         << ns_to_ms(aggregate.count == 0 ? 0 :
              aggregate.total_end_to_end_ns / aggregate.count)
         << " p50=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.50))
         << " p95=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.95))
         << " p99=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.99))
         << '\n';
  if (!aggregate.fine_grained_breakdown_observed) {
    output << "  fine_grained_breakdown: disabled\n";
    return output.str();
  }
  const u64 rdma_ns = aggregate.category_ns[static_cast<size_t>(Category::rdma)];
  const u64 cpu_ns = aggregate.total_service_ns > rdma_ns
    ? aggregate.total_service_ns - rdma_ns : 0;
  output << "  cpu_ms: " << ns_to_ms(cpu_ns) << '\n';
  output << "  rdma_ms: " << ns_to_ms(rdma_ns) << '\n';
  return output.str();
}

}  // namespace service::breakdown
