#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace vamana::routing {

// Mutation-driven shard centroids with compensated FP64 sums and a small caller-selected set of
// real graph entry points.  The router deliberately does not choose entries:
// callers must supply current live nodes rather than sampled substitutes.
//
// Writers are serialized but do not rebuild the read snapshot on the mutation
// path.  A maintenance thread explicitly calls publish() after a time- or
// count-based batch.  Readers load one shared_ptr atomically, so they observe
// either the complete previous publication or the complete new publication.
class CentroidRouter {
public:
  static constexpr u32 kMinLiveEntries = 1;
  static constexpr u32 kMaxLiveEntries = 4;

  struct LiveEntry {
    RemotePtr pointer;
    u32 generation{};

    bool operator==(const LiveEntry&) const = default;
  };

  struct ShardSnapshot {
    u32 shard{};
    u64 version{};
    u64 count{};
    vec<f64> sum;
    vec<f64> centroid;
    std::array<LiveEntry, kMaxLiveEntries> live_entries{};
    u32 live_entry_count{};

    span<const LiveEntry> entries() const {
      return {live_entries.data(), live_entry_count};
    }
  };

  struct Snapshot {
    u32 dim{};
    u32 shard_count{};
    u64 version{};
    vec<ShardSnapshot> shards;
  };

  CentroidRouter(u32 dim, u32 shard_count);

  CentroidRouter(const CentroidRouter&) = delete;
  CentroidRouter& operator=(const CentroidRouter&) = delete;

  u32 dim() const { return dim_; }
  u32 shard_count() const { return shard_count_; }

  // Restore one exact physical-shard checkpoint without replaying its member
  // vectors. This startup-only operation copies O(dim + entries) state and may
  // be called at most once for each shard while the restoration window is
  // open. The first successful ordinary mutation, or any explicit publish(),
  // closes that window for the complete router.
  //
  // version must be non-zero. A non-empty shard must provide 1..4 unique,
  // non-null live entries owned by that physical shard (and no more entries
  // than vectors). An empty shard must provide an all-zero sum and no entries.
  // The restored state remains invisible to readers until publish().
  bool restore_shard_state(u32 shard,
                           u64 count,
                           span<const f64> sum,
                           span<const LiveEntry> live_entries,
                           u64 version);

  // The caller is authoritative for membership and must supply the exact old
  // vector to erase/upsert.  This component intentionally keeps no per-vector
  // history.  Successful operations update only the authoritative FP64 state,
  // its versions and a dirty bit; they never construct a read snapshot.
  bool insert(u32 shard, span<const f32> vector);
  bool erase(u32 shard, span<const f32> vector);
  bool upsert(u32 shard,
              span<const f32> old_vector,
              span<const f32> new_vector);
  bool move(u32 source_shard,
            u32 destination_shard,
            span<const f32> vector);

  // Replace the complete entry set for one non-empty shard.  Entries must be
  // unique, non-null physical pointers owned by that shard.  Their ordering is
  // preserved for the caller; the centroid router does not sample or elect a
  // representative.  Empty shards clear their entries automatically.
  bool replace_live_entries(u32 shard, span<const LiveEntry> entries);

  // Publish every successful mutation since the previous publication as one
  // immutable snapshot.  Returns true only when dirty authoritative state was
  // published.  This is the batching boundary intended for a maintenance
  // thread; mutation callers may issue any number of operations before it.
  bool publish();

  // A returned snapshot remains immutable and valid while later mutations
  // publish newer snapshots.
  std::shared_ptr<const Snapshot> snapshot() const;

  // Exact writer-side cardinality, including mutations not yet published.
  // This is intentionally separate from snapshot(): maintenance code uses it
  // to validate a partially idempotent membership batch without confusing
  // the previous immutable publication with current authoritative state.
  u64 authoritative_count(u32 shard) const;

  // Exact writer-side centroid after all successful mutations, including
  // those not yet published. Storage maintenance uses this to keep the tiny
  // live-root set near the centroid in the same publication transaction.
  // Empty shards return an empty vector.
  vec<f64> authoritative_centroid(u32 shard) const;
  bool copy_authoritative_centroid(u32 shard, span<f64> destination) const;

private:
  struct ShardState {
    u64 version{};
    u64 count{};
    vec<f64> sum;
    // Neumaier compensation keeps long insert/erase streams stable even when
    // vector magnitudes differ by many orders. Publications expose sum plus
    // this correction, while checkpoints restart from that canonical value.
    vec<f64> sum_compensation;
    std::array<LiveEntry, kMaxLiveEntries> live_entries{};
    u32 live_entry_count{};
    bool restored{};
  };

  bool valid_vector(span<const f32> vector) const;
  static void add_sum_component(ShardState& state, u32 dimension,
                                f64 delta);
  static f64 compensated_sum_component(const ShardState& state,
                                       u32 dimension);
  std::shared_ptr<const Snapshot> build_snapshot_locked() const;

  u32 dim_{};
  u32 shard_count_{};
  vec<ShardState> shards_;
  u64 state_version_{};
  bool dirty_{};
  bool restoration_closed_{};
  mutable std::mutex writer_mutex_;
  // Use the shared_ptr atomic free functions rather than
  // atomic<shared_ptr<T>> so this component remains compatible with the
  // repository's GCC 11/libstdc++ baseline.
  std::shared_ptr<const Snapshot> published_;
};

}  // namespace vamana::routing
