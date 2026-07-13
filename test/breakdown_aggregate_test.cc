#include <cassert>
#include <cstddef>

#include "service/breakdown/aggregate.hh"

int main() {
  service::breakdown::Aggregate aggregate;
  constexpr size_t sample_count =
    service::breakdown::kLatencyReservoirCapacity + 4096;
  u64 expected_end_to_end = 0;
  u64 expected_service = 0;

  for (size_t index = 1; index <= sample_count; ++index) {
    service::breakdown::Sample sample{
      service::breakdown::Operation::query, false};
    sample.finished_flag = true;
    sample.end_to_end_ns = static_cast<u64>(index);
    sample.service_ns = static_cast<u64>(index * 2);
    expected_end_to_end += sample.end_to_end_ns;
    expected_service += sample.service_ns;
    service::breakdown::add_sample(aggregate, sample);
  }

  assert(aggregate.count == sample_count);
  assert(aggregate.total_end_to_end_ns == expected_end_to_end);
  assert(aggregate.total_service_ns == expected_service);
  assert(aggregate.end_to_end_latencies_ns.size() ==
         service::breakdown::kLatencyReservoirCapacity);
  assert(aggregate.service_latencies_ns.size() ==
         service::breakdown::kLatencyReservoirCapacity);
  for (size_t index = 0; index < aggregate.end_to_end_latencies_ns.size(); ++index) {
    assert(aggregate.service_latencies_ns[index] ==
           aggregate.end_to_end_latencies_ns[index] * 2);
  }
  return 0;
}
