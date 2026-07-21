#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

struct PartitionSearchBudget {
  u64 max_expansions{};
  size_t max_candidate_visits{};
  size_t max_remote_frontier{};

  static constexpr PartitionSearchBudget unbounded() {
    return {
      .max_expansions = std::numeric_limits<u64>::max(),
      .max_candidate_visits = std::numeric_limits<size_t>::max(),
      .max_remote_frontier = std::numeric_limits<size_t>::max(),
    };
  }
};

inline size_t saturating_search_budget_add(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs
    ? std::numeric_limits<size_t>::max() : lhs + rhs;
}

inline size_t saturating_search_budget_multiply(size_t lhs, size_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs
    ? std::numeric_limits<size_t>::max() : lhs * rhs;
}

// Stage1 owns at most 2L graph expansions and exports at most 2L boundary
// candidates. Stage2 below owns another L expansions, so one insertion has a
// dataset-independent 3L expansion bound. The visit limits include every
// possible edge examined from a bounded expansion; they are correctness
// guards, not empirical tuning parameters.
inline PartitionSearchBudget stage1_partition_search_budget(
    u32 beam_width, size_t entry_count, u32 graph_entry_capacity) {
  const u64 expansions = static_cast<u64>(beam_width) * 2;
  return {
    .max_expansions = expansions,
    .max_candidate_visits = saturating_search_budget_add(
      entry_count, saturating_search_budget_multiply(
        static_cast<size_t>(expansions), graph_entry_capacity)),
    .max_remote_frontier = static_cast<size_t>(beam_width) * 2,
  };
}

inline PartitionSearchBudget stage2_partition_search_budget(
    u32 beam_width, u32 graph_entry_capacity) {
  const size_t frontier = static_cast<size_t>(beam_width) * 2;
  return {
    .max_expansions = beam_width,
    .max_candidate_visits = saturating_search_budget_add(
      static_cast<size_t>(beam_width),
      saturating_search_budget_add(
        frontier, saturating_search_budget_multiply(
          beam_width, graph_entry_capacity))),
    .max_remote_frontier = frontier,
  };
}

// Algorithm-only state for construction search inside one storage partition.
// The beam is always sorted and never grows beyond L. Production callers also
// provide explicit expansion, candidate-visit, and boundary-frontier limits;
// algorithm-only callers may deliberately request the unbounded policy.
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

  void reset(u32 partition_id, u32 beam_width,
             PartitionSearchBudget budget =
               PartitionSearchBudget::unbounded()) {
    if (beam_width == 0) {
      throw std::invalid_argument("partition-local construction beam width must be positive");
    }
    partition_id_ = partition_id;
    beam_width_ = beam_width;
    budget_ = budget;
    visited_.clear();
    remote_priorities_.clear();
    remote_frontier_.clear();
    beam_.clear();
    expansion_count_ = 0;
    current_expansion_distance_ =
      std::numeric_limits<distance_t>::infinity();
    budget_exhausted_ = false;
    const size_t visit_reserve = budget_.max_candidate_visits ==
        std::numeric_limits<size_t>::max()
      ? beam_width_
      : std::max<size_t>(beam_width_, budget_.max_candidate_visits);
    visited_.reserve(visit_reserve);
    const size_t frontier_reserve = std::min<size_t>(
      budget_.max_remote_frontier, static_cast<size_t>(beam_width_) * 2);
    remote_priorities_.reserve(frontier_reserve);
    remote_frontier_.reserve(frontier_reserve);
    beam_.reserve(beam_width_);
  }

  // Returns true exactly once for a non-null pointer owned by this partition.
  // Rejected/deleted candidates should remain visited for this search, so
  // visitation and scoring are deliberately separate operations.
  bool try_visit(RemotePtr pointer) {
    if (pointer.is_null()) {
      return false;
    }
    if (pointer.memory_node() != partition_id_) {
      admit_remote_frontier(pointer, current_expansion_distance_);
      return false;
    }
    if (visited_.contains(pointer)) return false;
    if (visited_.size() >= budget_.max_candidate_visits) {
      budget_exhausted_ = true;
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
        if (expansion_count_ >= budget_.max_expansions) {
          budget_exhausted_ = true;
          return std::nullopt;
        }
        entry.expanded = true;
        ++expansion_count_;
        current_expansion_distance_ = entry.distance;
        return entry.rptr;
      }
    }
    return std::nullopt;
  }

  const vec<PartitionLocalSearchEntry>& final_beam() const { return beam_; }
  vec<PartitionLocalSearchEntry>& mutable_final_beam() { return beam_; }
  const vec<RemotePtr>& remote_frontier() const { return remote_frontier_; }

  size_t visited_count() const { return visited_.size(); }
  u64 expansion_count() const { return expansion_count_; }
  u32 beam_width() const { return beam_width_; }
  bool budget_exhausted() const { return budget_exhausted_; }

private:
  static bool entry_less(const PartitionLocalSearchEntry& lhs,
                         const PartitionLocalSearchEntry& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.rptr.raw_address < rhs.rptr.raw_address;
  }

  void admit_remote_frontier(RemotePtr pointer,
                             distance_t parent_distance) {
    if (budget_.max_remote_frontier == 0) {
      budget_exhausted_ = true;
      return;
    }
    const auto key_less = [](distance_t lhs_distance, RemotePtr lhs,
                             distance_t rhs_distance, RemotePtr rhs) {
      if (lhs_distance != rhs_distance) return lhs_distance < rhs_distance;
      return lhs.raw_address < rhs.raw_address;
    };
    const auto existing = remote_priorities_.find(pointer);
    if (existing != remote_priorities_.end()) {
      if (key_less(parent_distance, pointer, existing->second, pointer)) {
        existing->second = parent_distance;
        std::sort(remote_frontier_.begin(), remote_frontier_.end(),
                  [&](RemotePtr lhs, RemotePtr rhs) {
                    return key_less(remote_priorities_.at(lhs), lhs,
                                    remote_priorities_.at(rhs), rhs);
                  });
      }
      return;
    }

    const auto insert_sorted = [&]() {
      remote_priorities_.emplace(pointer, parent_distance);
      const auto position = std::lower_bound(
        remote_frontier_.begin(), remote_frontier_.end(), pointer,
        [&](RemotePtr lhs, RemotePtr rhs) {
          return key_less(remote_priorities_.at(lhs), lhs,
                          parent_distance, rhs);
        });
      remote_frontier_.insert(position, pointer);
    };
    if (remote_frontier_.size() < budget_.max_remote_frontier) {
      insert_sorted();
      return;
    }

    const RemotePtr worst = remote_frontier_.back();
    const distance_t worst_distance = remote_priorities_.at(worst);
    if (!key_less(parent_distance, pointer, worst_distance, worst)) {
      budget_exhausted_ = true;
      return;
    }
    remote_frontier_.pop_back();
    remote_priorities_.erase(worst);
    insert_sorted();
    budget_exhausted_ = true;
  }

  u32 partition_id_{};
  u32 beam_width_{};
  PartitionSearchBudget budget_{PartitionSearchBudget::unbounded()};
  hashset_t<RemotePtr> visited_;
  dense_hashmap_t<RemotePtr, distance_t> remote_priorities_;
  vec<RemotePtr> remote_frontier_;
  vec<PartitionLocalSearchEntry> beam_;
  u64 expansion_count_{};
  distance_t current_expansion_distance_{
    std::numeric_limits<distance_t>::infinity()};
  bool budget_exhausted_{};
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
    PartitionSearchBudget budget,
    Score&& score,
    Expand&& expand) {
  search.reset(partition_id, beam_width, budget);

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
vec<PartitionLocalSearchEntry>& partition_local_construction_search_into(
    PartitionLocalSearchBeam& search,
    span<const RemotePtr> entry_points,
    u32 partition_id,
    u32 beam_width,
    Score&& score,
    Expand&& expand) {
  return partition_local_construction_search_into(
    search, entry_points, partition_id, beam_width,
    PartitionSearchBudget::unbounded(),
    std::forward<Score>(score), std::forward<Expand>(expand));
}

template <typename Score, typename Expand>
vec<PartitionLocalSearchEntry> partition_local_construction_search(
    span<const RemotePtr> entry_points,
    u32 partition_id,
    u32 beam_width,
    PartitionSearchBudget budget,
    Score&& score,
    Expand&& expand) {
  PartitionLocalSearchBeam search(partition_id, beam_width);
  const vec<PartitionLocalSearchEntry>& final_beam =
    partition_local_construction_search_into(
      search, entry_points, partition_id, beam_width, budget,
      std::forward<Score>(score), std::forward<Expand>(expand));
  return final_beam;
}


template <typename Score, typename Expand>
vec<PartitionLocalSearchEntry> partition_local_construction_search(
    span<const RemotePtr> entry_points,
    u32 partition_id,
    u32 beam_width,
    Score&& score,
    Expand&& expand) {
  return partition_local_construction_search(
    entry_points, partition_id, beam_width,
    PartitionSearchBudget::unbounded(),
    std::forward<Score>(score), std::forward<Expand>(expand));
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

// Fixed-width continuation state used by Stage2. Stage1-local candidates are
// seeded as already expanded. Only pointers outside the Stage1 partition are
// admitted for further expansion, so Stage2 never repeats local work.
class PartitionContinuationBeam {
public:
  PartitionContinuationBeam(u32 partition_id, u32 beam_width)
      : partition_id_(partition_id), beam_width_(beam_width) {
    reset(partition_id, beam_width);
  }

  void reset(u32 partition_id, u32 beam_width,
             PartitionSearchBudget budget =
               PartitionSearchBudget::unbounded()) {
    if (beam_width == 0) {
      throw std::invalid_argument(
        "partition continuation beam width must be positive");
    }
    partition_id_ = partition_id;
    beam_width_ = beam_width;
    budget_ = budget;
    visited_.clear();
    beam_.clear();
    expansion_count_ = 0;
    budget_exhausted_ = false;
    const size_t visit_reserve = budget_.max_candidate_visits ==
        std::numeric_limits<size_t>::max()
      ? beam_width_
      : std::max<size_t>(beam_width_, budget_.max_candidate_visits);
    visited_.reserve(visit_reserve);
    beam_.reserve(beam_width_);
  }

  void seed_local(span<const PartitionLocalSearchEntry> local_beam) {
    for (const PartitionLocalSearchEntry& entry : local_beam) {
      if (entry.rptr.is_null() ||
          entry.rptr.memory_node() != partition_id_ ||
          !visited_.insert(entry.rptr).second) {
        continue;
      }
      add(entry.rptr, entry.distance, true);
    }
  }

  bool try_visit_remote(RemotePtr pointer) {
    if (pointer.is_null() || pointer.memory_node() == partition_id_ ||
        visited_.contains(pointer)) {
      return false;
    }
    if (visited_.size() >= budget_.max_candidate_visits) {
      budget_exhausted_ = true;
      return false;
    }
    return visited_.insert(pointer).second;
  }

  void add_remote(RemotePtr pointer, distance_t distance) {
    add(pointer, distance, false);
  }

  std::optional<RemotePtr> take_closest_unexpanded() {
    for (PartitionLocalSearchEntry& entry : beam_) {
      if (!entry.expanded) {
        if (expansion_count_ >= budget_.max_expansions) {
          budget_exhausted_ = true;
          return std::nullopt;
        }
        entry.expanded = true;
        ++expansion_count_;
        return entry.rptr;
      }
    }
    return std::nullopt;
  }

  const vec<PartitionLocalSearchEntry>& final_beam() const { return beam_; }
  u64 expansion_count() const { return expansion_count_; }
  bool budget_exhausted() const { return budget_exhausted_; }
  void mark_budget_exhausted() { budget_exhausted_ = true; }

private:
  void add(RemotePtr pointer, distance_t distance, bool expanded) {
    const PartitionLocalSearchEntry candidate{pointer, distance, expanded};
    const auto position = std::lower_bound(
      beam_.begin(), beam_.end(), candidate,
      [](const PartitionLocalSearchEntry& lhs,
         const PartitionLocalSearchEntry& rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        return lhs.rptr.raw_address < rhs.rptr.raw_address;
      });
    beam_.insert(position, candidate);
    if (beam_.size() > beam_width_) beam_.resize(beam_width_);
  }

  u32 partition_id_{};
  u32 beam_width_{};
  PartitionSearchBudget budget_{PartitionSearchBudget::unbounded()};
  hashset_t<RemotePtr> visited_;
  vec<PartitionLocalSearchEntry> beam_;
  u64 expansion_count_{};
  bool budget_exhausted_{};
};

// Continue Stage1 using its exact local beam and the cross-partition frontier
// discovered during local expansion. score_batch must invoke emit(ptr, dist)
// for every live input pointer. expand enumerates one remote node's neighbors.
// Both callbacks may batch RDMA internally; the algorithm itself performs no
// per-shard restart and never grows beyond L candidates.
template <typename ScoreBatch, typename Expand>
const vec<PartitionLocalSearchEntry>& continue_partition_construction_search_into(
    span<const PartitionLocalSearchEntry> local_beam,
    span<const RemotePtr> remote_frontier,
    u32 partition_id,
    u32 beam_width,
    PartitionSearchBudget budget,
    ScoreBatch&& score_batch,
    Expand&& expand,
    bool* budget_exhausted = nullptr,
    u64* expansion_count = nullptr) {
  // Stage2 does not suspend or interleave another task on the same OS worker.
  // Reuse its O(L*R) visited table and small edge buffers instead of allocating
  // them once per high-frequency insertion.
  thread_local PartitionContinuationBeam search(0, 1);
  search.reset(partition_id, beam_width, budget);
  search.seed_local(local_beam);

  thread_local vec<RemotePtr> pending;
  pending.clear();
  pending.reserve(remote_frontier.size());
  const auto consider_batch = [&](span<const RemotePtr> candidates) {
    pending.clear();
    for (const RemotePtr pointer : candidates) {
      if (search.try_visit_remote(pointer)) pending.push_back(pointer);
    }
    if (pending.empty()) return;
    std::invoke(score_batch, span<const RemotePtr>{pending},
                [&](RemotePtr pointer, distance_t distance) {
                  search.add_remote(pointer, distance);
                });
  };

  const size_t admitted_frontier = std::min(
    remote_frontier.size(), budget.max_remote_frontier);
  if (admitted_frontier != remote_frontier.size()) {
    search.mark_budget_exhausted();
  }
  consider_batch(span<const RemotePtr>{
    remote_frontier.data(), admitted_frontier});
  thread_local vec<RemotePtr> neighbors;
  neighbors.clear();
  while (const std::optional<RemotePtr> current =
           search.take_closest_unexpanded()) {
    neighbors.clear();
    std::invoke(expand, *current, [&](RemotePtr pointer) {
      neighbors.push_back(pointer);
    });
    consider_batch(span<const RemotePtr>{neighbors});
  }
  if (budget_exhausted != nullptr) {
    *budget_exhausted = search.budget_exhausted();
  }
  if (expansion_count != nullptr) {
    *expansion_count = search.expansion_count();
  }
  return search.final_beam();
}

template <typename ScoreBatch, typename Expand>
vec<PartitionLocalSearchEntry> continue_partition_construction_search(
    span<const PartitionLocalSearchEntry> local_beam,
    span<const RemotePtr> remote_frontier,
    u32 partition_id,
    u32 beam_width,
    PartitionSearchBudget budget,
    ScoreBatch&& score_batch,
    Expand&& expand,
    bool* budget_exhausted = nullptr,
    u64* expansion_count = nullptr) {
  return continue_partition_construction_search_into(
    local_beam, remote_frontier, partition_id, beam_width, budget,
    std::forward<ScoreBatch>(score_batch), std::forward<Expand>(expand),
    budget_exhausted, expansion_count);
}

template <typename ScoreBatch, typename Expand>
vec<PartitionLocalSearchEntry> continue_partition_construction_search(
    span<const PartitionLocalSearchEntry> local_beam,
    span<const RemotePtr> remote_frontier,
    u32 partition_id,
    u32 beam_width,
    ScoreBatch&& score_batch,
    Expand&& expand) {
  return continue_partition_construction_search_into(
    local_beam, remote_frontier, partition_id, beam_width,
    PartitionSearchBudget::unbounded(),
    std::forward<ScoreBatch>(score_batch),
    std::forward<Expand>(expand));
}

// Minimizing cross-shard outgoing edges is equivalent to placing the node on
// the shard containing the largest number of its final neighbors. Prefer the
// Stage1 home on ties so migration happens only when it strictly improves
// physical locality.
inline u32 choose_min_cross_shard_home(
    span<const RemotePtr> neighbors,
    u32 shard_count,
    u32 stage1_home) {
  if (shard_count == 0 || stage1_home >= shard_count) {
    throw std::invalid_argument("invalid shard domain for final placement");
  }
  if (shard_count > 64) {
    throw std::invalid_argument("final placement supports at most 64 shards");
  }
  thread_local std::array<u32, 64> local_edges{};
  std::array<u32, 64> touched{};
  size_t touched_count = 0;
  for (const RemotePtr neighbor : neighbors) {
    if (!neighbor.is_null() && neighbor.memory_node() < shard_count) {
      if (local_edges[neighbor.memory_node()] == 0) {
        touched[touched_count++] = neighbor.memory_node();
      }
      ++local_edges[neighbor.memory_node()];
    }
  }
  u32 selected = stage1_home;
  std::sort(touched.begin(), touched.begin() + touched_count);
  for (size_t index = 0; index < touched_count; ++index) {
    const u32 shard = touched[index];
    if (local_edges[shard] > local_edges[selected]) selected = shard;
  }
  for (size_t index = 0; index < touched_count; ++index) {
    local_edges[touched[index]] = 0;
  }
  return selected;
}

}  // namespace memory_node_storage_owner_index_detail
