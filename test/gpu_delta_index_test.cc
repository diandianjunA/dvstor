#include <cassert>
#include <chrono>

#include "gpu_search/delta_index.hh"

int main() {
  using namespace std::chrono_literals;
  gpu_search::DeltaCoordinator delta(7);
  gpu_search::DeltaMutation insert;
  insert.id = 42;
  insert.kind = service::storage_owner::MutationKind::insert;
  insert.vector = {1, 2, 3, 4};
  delta.enqueue(insert);

  auto pending = delta.take_pending(16, 0us);
  assert(pending.size() == 1);
  const u64 first_epoch = pending.front().epoch;
  assert(delta.publish(std::move(pending), first_epoch));
  assert(delta.published_epoch() == first_epoch);
  assert(delta.delta_size() == 1);
  auto version = delta.version(42);
  assert(version.has_value());
  assert(!version->deleted);
  assert(version->in_delta);

  gpu_search::DeltaMutation erase;
  erase.id = 42;
  erase.kind = service::storage_owner::MutationKind::erase;
  erase.epoch = delta.reserve_epoch();
  const u64 erase_epoch = erase.epoch;
  assert(delta.publish({erase}, erase_epoch));
  version = delta.version(42);
  assert(version->deleted);
  assert(version->epoch == erase_epoch);

  assert(delta.should_consolidate(10, 1u << 20, 0.01, 0.7, 60s));
  const auto snapshot = delta.begin_consolidation();
  assert(snapshot.base_generation == 7);
  assert(snapshot.mutations.size() == 1);
  delta.complete_partial_consolidation({42}, 8, snapshot.epoch);
  assert(delta.base_generation() == 8);
  assert(delta.delta_size() == 0);
  assert(!delta.version(42)->in_delta);

  gpu_search::DeltaMutation next;
  next.id = 43;
  next.kind = service::storage_owner::MutationKind::insert;
  const u64 next_epoch = delta.reserve_epoch();
  assert(delta.publish({next}, next_epoch));
  delta.complete_consolidation(9, next_epoch);
  assert(delta.base_generation() == 9);
  assert(delta.delta_size() == 0);

  gpu_search::DeltaCoordinator durable_delta(1);
  gpu_search::DeltaMutation durable;
  durable.id = 99;
  durable.kind = service::storage_owner::MutationKind::insert;
  durable.owner_storage = 1;
  durable.maintenance_sequence = 7;
  durable.remote_node = 12345;
  gpu_search::DeltaMutation second_durable = durable;
  second_durable.id = 100;
  second_durable.remote_node = 54321;
  const u64 durable_epoch = durable_delta.reserve_epoch();
  assert(durable_delta.publish({durable, second_durable}, durable_epoch));
  assert(durable_delta.delta_size() == 2);
  const std::vector<u64> incomplete_watermarks{100, 6};
  assert(durable_delta.retire_durable(incomplete_watermarks).empty());
  assert(durable_delta.delta_size() == 2);
  const std::vector<u64> complete_watermarks{100, 7};
  const auto newly_durable = durable_delta.retire_durable(complete_watermarks, 1);
  assert(newly_durable.size() == 1);
  assert(newly_durable.front().durable);
  assert(durable_delta.delta_size() == 1);
  const auto final_durable = durable_delta.retire_durable(complete_watermarks, 1);
  assert(final_durable.size() == 1);
  assert(final_durable.front().durable);
  assert(final_durable.front().id != newly_durable.front().id);
  assert(durable_delta.retire_durable(complete_watermarks, 1).empty());
  assert(durable_delta.delta_size() == 0);
  const auto durable_snapshot = durable_delta.snapshot();
  assert(durable_snapshot.mutations.empty());
  assert(!durable_delta.version(99)->in_delta);

  gpu_search::DeltaCoordinator ordered_delta(1);
  gpu_search::DeltaMutation superseded;
  superseded.id = 200;
  superseded.owner_storage = 0;
  superseded.maintenance_sequence = 10;
  const u64 superseded_epoch = ordered_delta.reserve_epoch();
  assert(ordered_delta.publish({superseded}, superseded_epoch));
  superseded.maintenance_sequence = 20;
  const u64 latest_epoch = ordered_delta.reserve_epoch();
  assert(ordered_delta.publish({superseded}, latest_epoch));
  assert(ordered_delta.retire_durable(std::vector<u64>{10}).empty());
  const auto latest = ordered_delta.retire_durable(std::vector<u64>{20});
  assert(latest.size() == 1);
  assert(latest.front().epoch == latest_epoch);

  gpu_search::DeltaMutation later;
  later.id = 201;
  later.owner_storage = 0;
  later.maintenance_sequence = 30;
  const u64 later_epoch = ordered_delta.reserve_epoch();
  assert(ordered_delta.publish({later}, later_epoch));
  gpu_search::DeltaMutation earlier = later;
  earlier.id = 202;
  earlier.maintenance_sequence = 25;
  const u64 earlier_epoch = ordered_delta.reserve_epoch();
  assert(ordered_delta.publish({earlier}, earlier_epoch));
  const auto out_of_order = ordered_delta.retire_durable(std::vector<u64>{25});
  assert(out_of_order.size() == 1);
  assert(out_of_order.front().id == earlier.id);
  assert(ordered_delta.retire_durable(std::vector<u64>{29}).empty());
  const auto final_ordered = ordered_delta.retire_durable(std::vector<u64>{30});
  assert(final_ordered.size() == 1);
  assert(final_ordered.front().id == later.id);

  gpu_search::DeltaCoordinator stress_delta(1);
  std::vector<gpu_search::DeltaMutation> stress_mutations;
  stress_mutations.reserve(50000);
  for (u32 index = 0; index < 50000; ++index) {
    gpu_search::DeltaMutation mutation;
    mutation.id = 1000 + index;
    mutation.owner_storage = index % 5;
    mutation.maintenance_sequence = index / 5 + 1;
    stress_mutations.push_back(std::move(mutation));
  }
  const u64 stress_epoch = stress_delta.reserve_epoch();
  assert(stress_delta.publish(std::move(stress_mutations), stress_epoch));
  const std::vector<u64> stress_watermarks(5, 10000);
  size_t stress_retired = 0;
  for (;;) {
    auto batch = stress_delta.retire_durable(stress_watermarks, 137);
    if (batch.empty()) break;
    stress_retired += batch.size();
  }
  assert(stress_retired == 50000);
  assert(stress_delta.delta_size() == 0);
  return 0;
}
