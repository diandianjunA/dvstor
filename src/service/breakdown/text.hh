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

  if (!aggregate.fine_grained_breakdown_observed) {
    os << "  fine_grained_breakdown: disabled\n";
    return os.str();
  }

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
  if (aggregate.operation == Operation::query) {
    if (!aggregate.device_utilization_observed) {
      os << "  utilization: disabled (set --observe-device-utilization to enable)\n";
      return os.str();
    }
    const double service_ns = static_cast<double>(aggregate.total_service_ns);
    const double gpu_busy = service_ns == 0.0 ? 0.0
      : static_cast<double>(aggregate.total_gpu_kernel_ns) / service_ns;
    const double rdma_wait = service_ns == 0.0 ? 0.0
      : static_cast<double>(aggregate.total_rdma_wait_ns) / service_ns;
    const double rdma_payload_gbps = service_ns == 0.0 ? 0.0
      : static_cast<double>(aggregate.counters.rdma_read_bytes) * 8.0 / service_ns;
    os << "  utilization (query software view):\n";
    os << "    gpu_kernel_busy: " << gpu_busy * 100.0 << "%"
       << " (CUDA event, excludes launch/copies)\n";
    os << "    rdma_completion_wait: " << rdma_wait * 100.0 << "%"
       << " (awaited-completion time)\n";
    os << "    rdma_payload: " << rdma_payload_gbps << " Gb/s per summed service window\n";
  }
  return os.str();
}

}  // namespace service::breakdown
