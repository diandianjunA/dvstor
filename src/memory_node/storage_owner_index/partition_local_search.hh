#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// Algorithm-only state for construction search inside one storage partition.
// The beam is always sorted and never grows beyond L.  Search completion is
// defined by the absence of an unexpanded item in this beam; there is no
// independent expansion/depth limit.
struct PartitionLocalSearchEntry {
  RemotePtr rptr;
  distance_t distance{};
  bool expanded{false};
};

class PartitionLocalSearchBeam {
public:
  PartitionLocalSearchBeam(u32 partition_id, u32 beam_width)
      : partition_id_(partition_id), beam_width_(beam_width) {
    reset(partition_id, beam_width);
  }

  void reset(u32 partition_id, u32 beam_width) {
    if (beam_width == 0) {
      throw std::invalid_argument("partition-local construction beam width must be positive");
    }
    partition_id_ = partition_id;
    beam_width_ = beam_width;
    visited_.clear();
    beam_.clear();
    expansion_count_ = 0;
    visited_.reserve(beam_width_);
    beam_.reserve(beam_width_);
  }

  // Returns true exactly once for a non-null pointer owned by this partition.
  // Rejected/deleted candidates should remain visited for this search, so
  // visitation and scoring are deliberately separate operations.
  bool try_visit(RemotePtr pointer) {
    if (pointer.is_null() || pointer.memory_node() != partition_id_) {
      return false;
    }
    return visited_.insert(pointer).second;
  }

  // Precondition: try_visit(pointer) returned true.  A live scored candidate
  // is admitted only if it belongs in the current fixed-width beam.
  void add_visited(RemotePtr pointer, distance_t distance) {
    const PartitionLocalSearchEntry candidate{pointer, distance, false};
    const auto position = std::lower_bound(
      beam_.begin(), beam_.end(), candidate, entry_less);
    beam_.insert(position, candidate);
    if (beam_.size() > beam_width_) {
      beam_.resize(beam_width_);
    }
  }

  // Marks and returns the closest item that has not yet been expanded.
  // nullopt is the construction-search convergence condition.
  std::optional<RemotePtr> take_closest_unexpanded() {
    for (PartitionLocalSearchEntry& entry : beam_) {
      if (!entry.expanded) {
        entry.expanded = true;
        ++expansion_count_;
        return entry.rptr;
      }
    }
    return std::nullopt;
  }

  const vec<PartitionLocalSearchEntry>& final_beam() const { return beam_; }
  vec<PartitionLocalSearchEntry>& mutable_final_beam() { return beam_; }

  size_t visited_count() const { return visited_.size(); }
  u64 expansion_count() const { return expansion_count_; }
  u32 beam_width() const { return beam_width_; }

private:
  static bool entry_less(const PartitionLocalSearchEntry& lhs,
                         const PartitionLocalSearchEntry& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.rptr.raw_address < rhs.rptr.raw_address;
  }

  u32 partition_id_{};
  u32 beam_width_{};
  hashset_t<RemotePtr> visited_;
  vec<PartitionLocalSearchEntry> beam_;
  u64 expansion_count_{};
};

// Runs a complete, partition-local, multi-entry construction search.
//
// score(pointer) returns nullopt for a stale/deleted candidate, otherwise its
// distance to the query. expand(pointer, visit) enumerates its graph neighbors
// by invoking visit(neighbor). Both callbacks only observe pointers accepted by
// the partition boundary enforced above.
template <typename Score, typename Expand>
vec<PartitionLocalSearchEntry>& partition_local_construction_search_into(
    PartitionLocalSearchBeam& search,
    span<const RemotePtr> entry_points,
    u32 partition_id,
    u32 beam_width,
    Score&& score,
    Expand&& expand) {
  search.reset(partition_id, beam_width);

  auto consider = [&](RemotePtr pointer) {
    if (!search.try_visit(pointer)) {
      return;
    }
    const std::optional<distance_t> distance = std::invoke(score, pointer);
    if (distance.has_value()) {
      search.add_visited(pointer, *distance);
    }
  };

  for (const RemotePtr entry : entry_points) {
    consider(entry);
  }

  while (const std::optional<RemotePtr> current =
           search.take_closest_unexpanded()) {
    std::invoke(expand, *current, consider);
  }

  return search.mutable_final_beam();
}

template <typename Score, typename Expand>
vec<PartitionLocalSearchEntry> partition_local_construction_search(
    span<const RemotePtr> entry_points,
    u32 partition_id,
    u32 beam_width,
    Score&& score,
    Expand&& expand) {
  PartitionLocalSearchBeam search(partition_id, beam_width);
  const vec<PartitionLocalSearchEntry>& final_beam =
    partition_local_construction_search_into(
      search, entry_points, partition_id, beam_width,
      std::forward<Score>(score), std::forward<Expand>(expand));
  return final_beam;
}

// Concurrent mutation can tombstone a node after it was scored. This helper
// performs the final point-in-time validation without embedding storage-node
// liveness policy in the algorithm-only search primitive. Callers that later
// mutate graph records must still validate again at that mutation boundary.
template <typename IsCurrent>
void filter_final_partition_local_beam(
    vec<PartitionLocalSearchEntry>& beam, IsCurrent&& is_current) {
  beam.erase(
    std::remove_if(
      beam.begin(), beam.end(),
      [&](const PartitionLocalSearchEntry& entry) {
        return !std::invoke(is_current, entry.rptr);
      }),
    beam.end());
}

}  // namespace memory_node_storage_owner_index_detail
