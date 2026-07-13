#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "service/breakdown/sample.hh"

namespace service::breakdown {

inline constexpr size_t kLatencyReservoirCapacity = 1u << 18;

inline u64 reservoir_hash(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct Aggregate {
  Operation operation{Operation::query};
  size_t count{};
  u64 total_queue_wait_ns{};
  u64 total_service_ns{};
  u64 total_end_to_end_ns{};
  bool fine_grained_breakdown_observed{};
  std::vector<u64> end_to_end_latencies_ns;
  std::vector<u64> service_latencies_ns;
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};
  SampleCounters counters{};

  [[nodiscard]] u64 cpu_other_ns() const {
    u64 explicit_cpu = 0;
    for (size_t index = 0; index < subcategory_ns.size(); ++index) {
      if (parent_category(static_cast<Subcategory>(index)) == Category::cpu) {
        explicit_cpu += subcategory_ns[index];
      }
    }
    const u64 rdma = category_ns[static_cast<size_t>(Category::rdma)];
    const u64 cpu_total = total_service_ns > rdma ? total_service_ns - rdma : 0;
    return cpu_total > explicit_cpu ? cpu_total - explicit_cpu : 0;
  }
};

struct Report {
  Aggregate query{};
  Aggregate insert{};

  [[nodiscard]] bool has_query() const { return query.count > 0; }
  [[nodiscard]] bool has_insert() const { return insert.count > 0; }
};

inline void add_sample(Aggregate& aggregate, const Sample& sample) {
  if (!sample.finished_flag) return;
  aggregate.operation = sample.operation;
  ++aggregate.count;
  aggregate.total_queue_wait_ns += sample.queue_wait_ns;
  aggregate.total_service_ns += sample.service_ns;
  aggregate.total_end_to_end_ns += sample.end_to_end_ns;
  if (aggregate.end_to_end_latencies_ns.size() < kLatencyReservoirCapacity) {
    aggregate.end_to_end_latencies_ns.push_back(sample.end_to_end_ns);
    aggregate.service_latencies_ns.push_back(sample.service_ns);
  } else {
    const size_t replacement = static_cast<size_t>(
      reservoir_hash(static_cast<u64>(aggregate.count)) % aggregate.count);
    if (replacement < kLatencyReservoirCapacity) {
      aggregate.end_to_end_latencies_ns[replacement] = sample.end_to_end_ns;
      aggregate.service_latencies_ns[replacement] = sample.service_ns;
    }
  }
  aggregate.fine_grained_breakdown_observed =
    aggregate.fine_grained_breakdown_observed || sample.collects_breakdown();
  if (!sample.collects_breakdown()) return;
  for (size_t index = 0; index < aggregate.category_ns.size(); ++index) {
    aggregate.category_ns[index] += sample.category_ns[index];
  }
  for (size_t index = 0; index < aggregate.subcategory_ns.size(); ++index) {
    aggregate.subcategory_ns[index] += sample.subcategory_ns[index];
  }
  const SampleCounters counters = sample.counters();
  aggregate.counters.storage_owner_anchor_hints += counters.storage_owner_anchor_hints;
  aggregate.counters.storage_owner_anchor_valid_hints +=
    counters.storage_owner_anchor_valid_hints;
  aggregate.counters.storage_owner_anchor_expansions +=
    counters.storage_owner_anchor_expansions;
  aggregate.counters.storage_owner_anchor_remote_expansions +=
    counters.storage_owner_anchor_remote_expansions;
}

inline double ns_to_ms(u64 nanoseconds) {
  return static_cast<double>(nanoseconds) / 1'000'000.0;
}

inline u64 percentile_ns(std::vector<u64> values, double percentile) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const double index = percentile * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(index)];
}

}  // namespace service::breakdown
