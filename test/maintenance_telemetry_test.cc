#include <array>
#include <cassert>
#include <cstring>

#include "gpu_search/maintenance_telemetry.hh"

int main() {
  namespace telemetry = gpu_search::maintenance_telemetry;

  alignas(64) std::array<byte_t, gpu_search::format::kStorageControlBytes>
    control_page{};
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    control_page.data());
  *control = gpu_search::format::StorageControlBlock{.shard_id = 3};
  const auto original_header = *control;

  telemetry::Snapshot first{
    .shard_id = 3,
    .published_steady_ns = 100,
    .stage2_enqueued = 20,
    .stage2_finalized_live = 12,
    .remaining = 8,
    .stage2_continuations = 12,
    .stage2_remote_expansions = 44,
    .stage2_graph_prefetch_issued = 12,
    .stage2_graph_prefetch_hits = 8,
    .stage2_graph_prefetch_wasted = 2,
    .stage2_score_prefetch_issued = 30,
    .stage2_score_prefetch_hits = 24,
    .stage2_score_prefetch_wasted = 3,
    .stage2_home_rpc_batches = 7,
    .stage2_home_rpc_items = 19,
    .maintenance_lost_wake_avoided = 23,
  };
  first.stage2_delay_histogram[6] = 12;
  telemetry::publish(control_page.data(), first);

  telemetry::Snapshot copied;
  std::memcpy(&copied, telemetry::snapshot_from_control_page(
                         control_page.data()), sizeof(copied));
  const u64 sequence_after = telemetry::snapshot_from_control_page(
    control_page.data())->sequence;
  assert(telemetry::validate(copied, sequence_after, 3));
  assert(!telemetry::validate(copied, sequence_after, 2));
  assert(!telemetry::validate(copied, sequence_after + 2, 3));
  assert(copied.stage2_enqueued == 20);
  assert(copied.stage2_delay_histogram[6] == 12);
  assert(copied.stage2_graph_prefetch_issued == 12);
  assert(copied.stage2_graph_prefetch_hits == 8);
  assert(copied.stage2_score_prefetch_issued == 30);
  assert(copied.stage2_score_prefetch_hits == 24);
  assert(copied.stage2_home_rpc_batches == 7);
  assert(copied.stage2_home_rpc_items == 19);
  assert(copied.maintenance_lost_wake_avoided == 23);

  telemetry::Snapshot second = first;
  second.published_steady_ns = 200;
  second.stage2_enqueued = 25;
  second.stage2_finalized_live = 18;
  telemetry::publish(control_page.data(), second);
  const auto* published = telemetry::snapshot_from_control_page(
    control_page.data());
  assert((published->sequence & 1u) == 0);
  assert(published->sequence == sequence_after + 2);
  assert(published->stage2_enqueued == 25);

  // The extension must not alter the stable 192-byte control header.
  assert(std::memcmp(control, &original_header, sizeof(original_header)) == 0);
  return 0;
}
