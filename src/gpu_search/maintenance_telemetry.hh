#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>

#include "common/types.hh"
#include "gpu_search/index_format.hh"

namespace gpu_search::maintenance_telemetry {

// StorageControlBlock is a stable 192-byte ABI header at the beginning of a
// reserved 4 KiB page.  Telemetry is an optional extension in the otherwise
// unused tail of that page, so adding it neither changes header_bytes/version
// nor requires rebuilding an existing index.
inline constexpr u64 kMagic = 0x31544d43565344ULL;  // "DSVCMT1"
inline constexpr u32 kVersion = 1;
inline constexpr u32 kValidCounters = 1u << 0;
inline constexpr size_t kLatencyBucketCount = 18;
inline constexpr size_t kSnapshotOffset = sizeof(format::StorageControlBlock);

// sequence is a seqlock. A writer publishes odd before replacing the body and
// even after it. An RDMA reader fetches the complete record and then fetches
// sequence once more; it accepts only an identical, nonzero, even sequence.
// All counters are cumulative since the storage-node process started. This
// lets a compute node obtain a measurement-window delta with two bounded
// control-page reads, without per-mutation telemetry traffic.
struct alignas(64) Snapshot {
  u64 sequence{};
  u64 magic{kMagic};
  u32 version{kVersion};
  u32 snapshot_bytes{sizeof(Snapshot)};
  u32 shard_id{};
  u32 flags{kValidCounters};
  u64 published_steady_ns{};

  u64 stage2_enqueued{};
  u64 stage2_finalized_live{};
  u64 stale{};
  u64 remaining{};
  u64 peer_reverse_remaining{};
  u64 failed{};
  u64 peer_reverse_failed{};
  u64 admission_window{};
  u64 completion_outstanding{};
  u64 max_backlog{};
  u64 stage1_search_budget_exhausted{};
  u64 stage2_search_budget_exhausted{};
  u64 stage2_continuations{};
  u64 stage2_remote_frontier_items{};
  u64 stage2_remote_expansions{};
  u64 stage2_scored_candidates{};
  u64 stage2_migrations{};
  u64 stage2_final_edges{};
  u64 stage2_cross_edges_stage1_home{};
  u64 stage2_cross_edges_final_home{};
  u64 pressure_yields{};
  u64 stage2_batches{};
  u64 stage2_batched_items{};
  u64 stage2_graph_read_waves{};
  u64 stage2_graph_unique_reads{};
  u64 stage2_vector_read_waves{};
  u64 stage2_vector_unique_reads{};
  std::array<u64, kLatencyBucketCount> stage2_delay_histogram{};
  std::array<u64, 6> reserved{};
};

static_assert(kSnapshotOffset == 192);
static_assert(alignof(Snapshot) == 64);
static_assert(kSnapshotOffset % alignof(Snapshot) == 0);
static_assert(kSnapshotOffset + sizeof(Snapshot) <=
              format::kStorageControlBytes);

inline Snapshot* snapshot_from_control_page(byte_t* control_page) {
  return reinterpret_cast<Snapshot*>(control_page + kSnapshotOffset);
}

inline const Snapshot* snapshot_from_control_page(
    const byte_t* control_page) {
  return reinterpret_cast<const Snapshot*>(control_page + kSnapshotOffset);
}

inline void publish(byte_t* control_page, const Snapshot& source) {
  Snapshot* destination = snapshot_from_control_page(control_page);
  std::atomic_ref<u64> sequence(destination->sequence);
  u64 previous = sequence.load(std::memory_order_relaxed);
  if ((previous & 1u) != 0) ++previous;
  const u64 writing = previous + 1;
  const u64 committed = previous + 2;
  // acquire on the exchange prevents body stores from being hoisted before
  // the odd marker; release on the final store publishes the complete body.
  sequence.exchange(writing, std::memory_order_acq_rel);
  std::memcpy(reinterpret_cast<byte_t*>(destination) + sizeof(u64),
              reinterpret_cast<const byte_t*>(&source) + sizeof(u64),
              sizeof(Snapshot) - sizeof(u64));
  std::atomic_thread_fence(std::memory_order_release);
  sequence.store(committed, std::memory_order_release);
}

inline bool validate(const Snapshot& snapshot, u64 sequence_after,
                     u32 expected_shard) {
  return snapshot.sequence != 0 && (snapshot.sequence & 1u) == 0 &&
    snapshot.sequence == sequence_after && snapshot.magic == kMagic &&
    snapshot.version == kVersion &&
    snapshot.snapshot_bytes == sizeof(Snapshot) &&
    snapshot.shard_id == expected_shard &&
    (snapshot.flags & kValidCounters) != 0;
}

}  // namespace gpu_search::maintenance_telemetry
