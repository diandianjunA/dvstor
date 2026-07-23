#include <algorithm>
#include <cassert>
#include <cstddef>
#include <thread>
#include <vector>

#include "service/breakdown/aggregate.hh"

namespace {

void test_serial_aggregate_reservoir() {
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
}

void test_concurrent_aggregate_exact_totals_and_bounded_reservoir() {
  service::breakdown::ConcurrentAggregate aggregate{
    service::breakdown::Operation::query};
  constexpr size_t thread_count = 32;
  constexpr size_t samples_per_thread = 10'000;
  constexpr size_t sample_count = thread_count * samples_per_thread;

  std::vector<std::thread> producers;
  producers.reserve(thread_count);
  for (size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    producers.emplace_back([&, thread_index] {
      for (size_t local_index = 0;
           local_index < samples_per_thread; ++local_index) {
        const u64 value = static_cast<u64>(
          thread_index * samples_per_thread + local_index + 1);
        service::breakdown::Sample sample{
          service::breakdown::Operation::query, true};
        sample.finished_flag = true;
        sample.queue_wait_ns = 3;
        sample.end_to_end_ns = value;
        sample.service_ns = value * 2;
        for (size_t index = 0; index < sample.category_ns.size(); ++index) {
          sample.category_ns[index] = static_cast<u64>(index + 1);
        }
        for (size_t index = 0; index < sample.subcategory_ns.size(); ++index) {
          sample.subcategory_ns[index] = static_cast<u64>(index + 1);
        }
        aggregate.add(sample);
      }
    });
  }

  // Taking snapshots while producers are live exercises coherent collect()
  // without weakening exact totals. Constant queue/category contributions
  // make every intermediate snapshot independently checkable.
  while (true) {
    const auto snapshot = aggregate.collect();
    assert(snapshot.total_queue_wait_ns == snapshot.count * 3);
    for (size_t index = 0; index < snapshot.category_ns.size(); ++index) {
      assert(snapshot.category_ns[index] ==
             snapshot.count * static_cast<u64>(index + 1));
    }
    assert(snapshot.end_to_end_latencies_ns.size() ==
           std::min(snapshot.count,
                    service::breakdown::kLatencyReservoirCapacity));
    assert(snapshot.service_latencies_ns.size() ==
           snapshot.end_to_end_latencies_ns.size());
    // std::thread remains joinable after completion, so the final count is
    // the reliable termination condition for this closed producer set.
    if (snapshot.count == sample_count) break;
    std::this_thread::yield();
  }
  for (auto& producer : producers) producer.join();

  const auto snapshot = aggregate.collect();
  const u64 expected_end_to_end =
    static_cast<u64>(sample_count) * (sample_count + 1) / 2;
  assert(snapshot.operation == service::breakdown::Operation::query);
  assert(snapshot.count == sample_count);
  assert(snapshot.total_queue_wait_ns == sample_count * 3);
  assert(snapshot.total_end_to_end_ns == expected_end_to_end);
  assert(snapshot.total_service_ns == expected_end_to_end * 2);
  assert(snapshot.fine_grained_breakdown_observed);
  for (size_t index = 0; index < snapshot.category_ns.size(); ++index) {
    assert(snapshot.category_ns[index] ==
           sample_count * static_cast<u64>(index + 1));
  }
  for (size_t index = 0; index < snapshot.subcategory_ns.size(); ++index) {
    assert(snapshot.subcategory_ns[index] ==
           sample_count * static_cast<u64>(index + 1));
  }
  assert(snapshot.end_to_end_latencies_ns.size() ==
         service::breakdown::kLatencyReservoirCapacity);
  assert(snapshot.service_latencies_ns.size() ==
         service::breakdown::kLatencyReservoirCapacity);
  for (size_t index = 0; index < snapshot.end_to_end_latencies_ns.size(); ++index) {
    assert(snapshot.service_latencies_ns[index] ==
           snapshot.end_to_end_latencies_ns[index] * 2);
  }

  aggregate.reset();
  const auto reset_snapshot = aggregate.collect();
  assert(reset_snapshot.operation == service::breakdown::Operation::query);
  assert(reset_snapshot.count == 0);
  assert(reset_snapshot.total_queue_wait_ns == 0);
  assert(reset_snapshot.total_service_ns == 0);
  assert(reset_snapshot.total_end_to_end_ns == 0);
  assert(!reset_snapshot.fine_grained_breakdown_observed);
  assert(reset_snapshot.end_to_end_latencies_ns.empty());
  assert(reset_snapshot.service_latencies_ns.empty());

  service::breakdown::Sample post_reset_sample{
    service::breakdown::Operation::query, false};
  post_reset_sample.finished_flag = true;
  post_reset_sample.end_to_end_ns = 41;
  post_reset_sample.service_ns = 37;
  aggregate.add(post_reset_sample);
  const auto post_reset_snapshot = aggregate.collect();
  assert(post_reset_snapshot.count == 1);
  assert(post_reset_snapshot.end_to_end_latencies_ns.size() == 1);
  assert(post_reset_snapshot.service_latencies_ns.size() == 1);
  assert(post_reset_snapshot.end_to_end_latencies_ns[0] == 41);
  assert(post_reset_snapshot.service_latencies_ns[0] == 37);
}

void test_concurrent_report_operations_remain_separate() {
  service::breakdown::ConcurrentReport report;
  service::breakdown::Sample query{
    service::breakdown::Operation::query, false};
  query.finished_flag = true;
  query.end_to_end_ns = 11;
  query.service_ns = 7;
  service::breakdown::Sample insert{
    service::breakdown::Operation::insert, false};
  insert.finished_flag = true;
  insert.end_to_end_ns = 17;
  insert.service_ns = 13;
  service::breakdown::add_sample(report.query, query);
  service::breakdown::add_sample(report.insert, insert);

  const auto snapshot = report.collect();
  assert(snapshot.query.operation == service::breakdown::Operation::query);
  assert(snapshot.insert.operation == service::breakdown::Operation::insert);
  assert(snapshot.query.count == 1);
  assert(snapshot.insert.count == 1);
  assert(snapshot.query.total_end_to_end_ns == 11);
  assert(snapshot.insert.total_end_to_end_ns == 17);

  report.reset();
  const auto reset_snapshot = report.collect();
  assert(!reset_snapshot.has_query());
  assert(!reset_snapshot.has_insert());
}

}  // namespace

int main() {
  test_serial_aggregate_reservoir();
  test_concurrent_aggregate_exact_totals_and_bounded_reservoir();
  test_concurrent_report_operations_remain_separate();
  return 0;
}
