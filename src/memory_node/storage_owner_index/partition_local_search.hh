#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
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

// Both construction stages run their fixed-width beams to natural
// convergence. Stage1 must also preserve the complete, de-duplicated boundary
// exposed by that converged local search: Stage2 cannot know which remote
// pointer belongs in the beam until it has read and scored the vector. A 2L (or
// any other fixed) handoff cap therefore changes the search result rather than
// merely bounding temporary work. The generic policy retains explicit limits
// for algorithm-only diagnostics, while both production policies are fully
// unbounded.
inline PartitionSearchBudget stage1_partition_search_budget(
    u32, size_t, u32) {
  return PartitionSearchBudget::unbounded();
}

inline PartitionSearchBudget stage2_partition_search_budget(
    u32, u32) {
  // Stage1 has preserved the complete phase-boundary frontier. Stage2 must
  // not impose a new truncation on that exact handoff.
  return PartitionSearchBudget::unbounded();
}

// Algorithm-only state for construction search inside one storage partition.
// The beam is always sorted and never grows beyond L. Production callers run
// expansion/visitation to convergence and retain the complete phase-boundary
// frontier; algorithm-only callers may still inject a diagnostic policy.
struct PartitionLocalSearchEntry {
  RemotePtr rptr;
  distance_t distance{};
  bool expanded{false};
};

// Distance callbacks may observe malformed input data or an arithmetic NaN.
// A NaN cannot participate in a strict weak ordering, which would make beam
// membership depend on insertion order.  Treat it as the worst possible
// distance at the search boundary; finite and infinite distances retain their
// normal ordering and the full RemotePtr remains the deterministic tie break.
inline distance_t normalize_partition_search_distance(distance_t distance) {
  return std::isnan(distance)
    ? std::numeric_limits<distance_t>::infinity()
    : distance;
}

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
    remote_frontier_truncated_ = false;
    remote_frontier_dirty_ = false;
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
    const PartitionLocalSearchEntry candidate{
      pointer, normalize_partition_search_distance(distance), false};
    // Once the beam is full, most graph neighbors are farther than its
    // current boundary. Reject those in O(1) before lower_bound/insert moves
    // up to L entries. This is exactly equivalent to inserting and truncating
    // the last element, including the deterministic full-handle tie break.
    if (beam_.size() == beam_width_ &&
        !entry_less(candidate, beam_.back())) {
      return;
    }
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

  void finalize_remote_frontier() {
    if (!remote_frontier_dirty_) return;
    std::sort(remote_frontier_.begin(), remote_frontier_.end(),
              [&](RemotePtr lhs, RemotePtr rhs) {
                return frontier_key_less(
                  remote_priorities_.at(lhs), lhs,
                  remote_priorities_.at(rhs), rhs);
              });
    remote_frontier_dirty_ = false;
  }

  size_t visited_count() const { return visited_.size(); }
  u64 expansion_count() const { return expansion_count_; }
  u32 beam_width() const { return beam_width_; }
  bool budget_exhausted() const { return budget_exhausted_; }
  bool remote_frontier_truncated() const {
    return remote_frontier_truncated_;
  }

  // Precondition: the caller has copied final_beam()/remote_frontier() and
  // the search is no longer active. This is a retention policy only: it never
  // caps an in-flight naturally converging search. Normal-sized allocations
  // stay reserved for reuse, while an exceptional O(N) high-water mark is
  // released instead of becoming permanent thread-local memory.
  void trim_oversized_capacity(size_t max_retained_capacity) {
    visited_.clear();
    if (visited_.values().capacity() > max_retained_capacity) {
      visited_.rehash(0);
    }
    remote_priorities_.clear();
    if (remote_priorities_.values().capacity() > max_retained_capacity) {
      remote_priorities_.rehash(0);
    }
    remote_frontier_.clear();
    if (remote_frontier_.capacity() > max_retained_capacity) {
      vec<RemotePtr>{}.swap(remote_frontier_);
    }
    beam_.clear();
    if (beam_.capacity() > max_retained_capacity) {
      vec<PartitionLocalSearchEntry>{}.swap(beam_);
    }
  }

private:
  static bool entry_less(const PartitionLocalSearchEntry& lhs,
                         const PartitionLocalSearchEntry& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.rptr.raw_address < rhs.rptr.raw_address;
  }

  static bool frontier_key_less(distance_t lhs_distance, RemotePtr lhs,
                                distance_t rhs_distance, RemotePtr rhs) {
    if (lhs_distance != rhs_distance) return lhs_distance < rhs_distance;
    return lhs.raw_address < rhs.raw_address;
  }

  void admit_remote_frontier(RemotePtr pointer,
                             distance_t parent_distance) {
    if (budget_.max_remote_frontier == 0) {
      remote_frontier_truncated_ = true;
      return;
    }
    const auto existing = remote_priorities_.find(pointer);
    if (existing != remote_priorities_.end()) {
      if (frontier_key_less(
            parent_distance, pointer, existing->second, pointer)) {
        existing->second = parent_distance;
        if (budget_.max_remote_frontier ==
            std::numeric_limits<size_t>::max()) {
          remote_frontier_dirty_ = true;
        } else {
          std::sort(remote_frontier_.begin(), remote_frontier_.end(),
                    [&](RemotePtr lhs, RemotePtr rhs) {
                      return frontier_key_less(
                        remote_priorities_.at(lhs), lhs,
                        remote_priorities_.at(rhs), rhs);
                    });
        }
      }
      return;
    }

    if (budget_.max_remote_frontier ==
        std::numeric_limits<size_t>::max()) {
      remote_priorities_.emplace(pointer, parent_distance);
      remote_frontier_.push_back(pointer);
      remote_frontier_dirty_ = true;
      return;
    }

    const auto insert_sorted = [&]() {
      remote_priorities_.emplace(pointer, parent_distance);
      const auto position = std::lower_bound(
        remote_frontier_.begin(), remote_frontier_.end(), pointer,
        [&](RemotePtr lhs, RemotePtr rhs) {
          return frontier_key_less(remote_priorities_.at(lhs), lhs,
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
    if (!frontier_key_less(
          parent_distance, pointer, worst_distance, worst)) {
      remote_frontier_truncated_ = true;
      return;
    }
    remote_frontier_.pop_back();
    remote_priorities_.erase(worst);
    insert_sorted();
    remote_frontier_truncated_ = true;
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
  bool remote_frontier_truncated_{};
  bool remote_frontier_dirty_{};
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

  // Production keeps the complete boundary, so admission is O(1) and the
  // deterministic ordering cost is paid once rather than after every edge.
  search.finalize_remote_frontier();

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

// Fixed-width continuation state used by Stage2. Stage1's final local beam is
// seeded as already expanded, so pointers already carried across the phase
// boundary are never repeated. Stage2 is deliberately remote-only: Stage1 has
// already run the home-shard graph to natural convergence and exported its
// complete cross-shard boundary. Following a remote edge back into the home
// shard would restart local search, duplicate work, and invalidate the
// locality argument of the two-stage design.
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
    if (pointer.is_null() ||
        pointer.memory_node() == partition_id_ ||
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

  // Append, without mutating the beam, the closest candidates that could be
  // selected after the current authoritative expansion. Callers may fetch
  // their adjacency speculatively, but only take_closest_unexpanded() is
  // allowed to mark an entry or consume the expansion budget.
  size_t append_closest_unexpanded(
      u32 shard, size_t limit, vec<RemotePtr>& output) const {
    if (limit == 0 || expansion_count_ >= budget_.max_expansions) return 0;
    const u64 budget_remaining = budget_.max_expansions - expansion_count_;
    limit = static_cast<size_t>(std::min<u64>(limit, budget_remaining));
    const size_t before = output.size();
    for (const PartitionLocalSearchEntry& entry : beam_) {
      if (output.size() - before == limit) break;
      if (!entry.expanded && entry.rptr.memory_node() == shard) {
        output.push_back(entry.rptr);
      }
    }
    return output.size() - before;
  }

  // Global ordered preview.  The beam is already nearest-first, so callers
  // that own a cross-peer transport scheduler can choose the best next
  // expansions without first partitioning the ranking by shard.  Like the
  // shard-filtered overload this is observational only: it neither marks an
  // entry expanded nor consumes the expansion budget.
  size_t append_closest_unexpanded(
      size_t limit, vec<RemotePtr>& output) const {
    if (limit == 0 || expansion_count_ >= budget_.max_expansions) return 0;
    const u64 budget_remaining = budget_.max_expansions - expansion_count_;
    limit = static_cast<size_t>(std::min<u64>(limit, budget_remaining));
    const size_t before = output.size();
    for (const PartitionLocalSearchEntry& entry : beam_) {
      if (output.size() - before == limit) break;
      if (!entry.expanded) output.push_back(entry.rptr);
    }
    return output.size() - before;
  }

  const vec<PartitionLocalSearchEntry>& final_beam() const { return beam_; }
  u64 expansion_count() const { return expansion_count_; }
  bool budget_exhausted() const { return budget_exhausted_; }
  void mark_budget_exhausted() { budget_exhausted_ = true; }

  // Precondition: the final beam has been copied and this continuation is no
  // longer active. See PartitionLocalSearchBeam::trim_oversized_capacity().
  void trim_oversized_capacity(size_t max_retained_capacity) {
    visited_.clear();
    if (visited_.values().capacity() > max_retained_capacity) {
      visited_.rehash(0);
    }
    beam_.clear();
    if (beam_.capacity() > max_retained_capacity) {
      vec<PartitionLocalSearchEntry>{}.swap(beam_);
    }
  }

private:
  static bool entry_less(const PartitionLocalSearchEntry& lhs,
                         const PartitionLocalSearchEntry& rhs) {
    if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
    return lhs.rptr.raw_address < rhs.rptr.raw_address;
  }

  void add(RemotePtr pointer, distance_t distance, bool expanded) {
    const PartitionLocalSearchEntry candidate{
      pointer, normalize_partition_search_distance(distance), expanded};
    if (beam_.size() == beam_width_ &&
        !entry_less(candidate, beam_.back())) {
      return;
    }
    const auto position = std::lower_bound(
      beam_.begin(), beam_.end(), candidate, entry_less);
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

// One logical Stage2 continuation in a worker-local batch. initialize() copies
// both spans into owned search state, so their storage need not survive a
// paused continuation.
struct PartitionContinuationSeed {
  span<const PartitionLocalSearchEntry> local_beam;
  span<const RemotePtr> remote_frontier;
};

struct PartitionContinuationScoreRequest {
  size_t search_index{};
  RemotePtr pointer;
  // Generation identifies one dependency wave of one logical search.  It is
  // deliberately the last aggregate member so existing synchronous callers
  // that initialize {search_index, pointer} remain source compatible.
  u64 generation{};
};

struct PartitionContinuationExpandRequest {
  size_t search_index{};
  RemotePtr pointer;
  u64 generation{};
};

struct PartitionContinuationScoreResult {
  size_t search_index{};
  RemotePtr pointer;
  distance_t distance{};
  u64 generation{};
};

struct PartitionContinuationExpandResult {
  size_t search_index{};
  RemotePtr pointer;
  u64 generation{};
};

enum class PartitionContinuationWave {
  uninitialized,
  score,
  expand,
  complete,
};

// Interleave independent Stage2 continuations at their natural dependency
// boundary. Each logical search owns its phase and monotonically increasing
// generation. A search expands exactly one closest beam entry, then resolves
// every newly discovered vector in that score generation before it may select
// its next entry. Consequently it is algorithmically identical to the serial
// continuation, while a retryable vector in search A cannot prevent search B
// from completing further score/expand generations.
//
// The zero-argument pending/consume methods and wave() are compatibility views
// for synchronous callers: they gather all searches currently in one phase.
// Asynchronous callers should use the indexed pending/resolve methods. A stale
// generation is rejected without mutating the search. resolve_score_request()
// accepts nullopt to record one coherent terminal/missing vector observation;
// only the final resolution in that search's generation advances its beam.
class PartitionContinuationBatch {
public:
  // Configure an empty batch whose searches may be activated independently.
  // This lets a retryable Stage1 handoff delay only its own continuation.
  void initialize(
      size_t search_count,
      u32 partition_id,
      u32 beam_width,
      PartitionSearchBudget budget) {
    if (beam_width == 0) {
      throw std::invalid_argument(
        "partition continuation batch beam width must be positive");
    }
    search_count_ = search_count;
    partition_id_ = partition_id;
    beam_width_ = beam_width;
    budget_ = budget;
    initialized_ = true;
    activated_count_ = 0;
    while (searches_.size() < search_count_) {
      searches_.emplace_back();
    }
    results_.resize(search_count_);
    exhausted_results_.assign(search_count_, false);
    expansion_results_.assign(search_count_, 0);
    for (size_t search_index = 0; search_index < search_count_;
         ++search_index) {
      searches_[search_index].reset_metadata();
      results_[search_index].clear();
    }
    synchronous_score_results_.clear();
    synchronous_expand_results_.clear();
    request_cache_dirty_ = true;
  }

  void initialize(
      span<const PartitionContinuationSeed> seeds,
      u32 partition_id,
      u32 beam_width,
      PartitionSearchBudget budget) {
    initialize(seeds.size(), partition_id, beam_width, budget);
    for (size_t search_index = 0; search_index < search_count_;
         ++search_index) {
      initialize_search(search_index, seeds[search_index]);
    }
  }

  void initialize_search(size_t search_index,
                         const PartitionContinuationSeed& seed) {
    require_initialized();
    require_search_index(search_index);
    SearchState& state = searches_[search_index];
    if (state.phase != PartitionContinuationWave::uninitialized) {
      throw std::logic_error("continuation search activated more than once");
    }

    state.search.reset(partition_id_, beam_width_, budget_);
    state.search.seed_local(seed.local_beam);
    const size_t admitted_frontier = std::min(
      seed.remote_frontier.size(), budget_.max_remote_frontier);
    if (admitted_frontier != seed.remote_frontier.size()) {
      state.search.mark_budget_exhausted();
    }
    for (size_t item = 0; item < admitted_frontier; ++item) {
      const RemotePtr pointer = seed.remote_frontier[item];
      if (state.search.try_visit_remote(pointer)) {
        state.pending_scores.push_back({search_index, pointer, 0});
      }
    }
    ++activated_count_;
    if (!state.pending_scores.empty()) {
      enter_score_phase(search_index, state);
    } else {
      prepare_next_expand_or_finish(search_index, state);
    }
    request_cache_dirty_ = true;
  }

  PartitionContinuationWave wave() const {
    if (!initialized_) return PartitionContinuationWave::uninitialized;
    // Compatibility callers can process mixed per-search phases by draining
    // all score-ready searches first, then all expand-ready searches.
    for (size_t index = 0; index < search_count_; ++index) {
      if (searches_[index].phase == PartitionContinuationWave::score) {
        return PartitionContinuationWave::score;
      }
    }
    for (size_t index = 0; index < search_count_; ++index) {
      if (searches_[index].phase == PartitionContinuationWave::expand) {
        return PartitionContinuationWave::expand;
      }
    }
    return all_complete() ? PartitionContinuationWave::complete
                          : PartitionContinuationWave::uninitialized;
  }

  bool search_active(size_t search_index) const {
    require_search_index(search_index);
    return searches_[search_index].phase !=
           PartitionContinuationWave::uninitialized;
  }

  bool search_complete(size_t search_index) const {
    require_search_index(search_index);
    return searches_[search_index].phase ==
           PartitionContinuationWave::complete;
  }

  bool all_complete() const {
    if (!initialized_ || activated_count_ != search_count_) return false;
    for (size_t index = 0; index < search_count_; ++index) {
      if (searches_[index].phase != PartitionContinuationWave::complete) {
        return false;
      }
    }
    return true;
  }

  bool complete() const { return all_complete(); }

  PartitionContinuationWave search_wave(size_t search_index) const {
    require_search_index(search_index);
    return searches_[search_index].phase;
  }

  u64 generation(size_t search_index) const {
    require_search_index(search_index);
    return searches_[search_index].generation;
  }

  span<const PartitionContinuationScoreRequest>
  pending_score_requests(size_t search_index) const {
    require_search_index(search_index);
    const SearchState& state = searches_[search_index];
    if (state.phase != PartitionContinuationWave::score) return {};
    return span<const PartitionContinuationScoreRequest>{
      state.pending_scores};
  }

  std::optional<PartitionContinuationExpandRequest>
  pending_expand_request(size_t search_index) const {
    require_search_index(search_index);
    const SearchState& state = searches_[search_index];
    if (state.phase != PartitionContinuationWave::expand) {
      return std::nullopt;
    }
    return state.pending_expand;
  }

  size_t append_expand_prefetch_candidates(
      size_t search_index, u32 shard, size_t limit,
      vec<RemotePtr>& output) const {
    require_search_index(search_index);
    const SearchState& state = searches_[search_index];
    if (state.phase != PartitionContinuationWave::expand ||
        !state.pending_expand.has_value()) {
      return 0;
    }
    return state.search.append_closest_unexpanded(
      shard, limit, output);
  }

  size_t append_expand_prefetch_candidates(
      size_t search_index, size_t limit, vec<RemotePtr>& output) const {
    require_search_index(search_index);
    const SearchState& state = searches_[search_index];
    if (state.phase != PartitionContinuationWave::expand ||
        !state.pending_expand.has_value()) {
      return 0;
    }
    return state.search.append_closest_unexpanded(limit, output);
  }

  span<const PartitionContinuationScoreRequest>
  pending_score_requests() const {
    rebuild_request_cache();
    return span<const PartitionContinuationScoreRequest>{score_requests_};
  }

  span<const PartitionContinuationExpandRequest>
  pending_expand_requests() const {
    rebuild_request_cache();
    return span<const PartitionContinuationExpandRequest>{expand_requests_};
  }

  // Resolve one vector snapshot. distance=nullopt is a stable terminal/missing
  // observation. Retryable observations must not call this method; keeping the
  // request unresolved leaves only this logical search paused.
  bool resolve_score_request(
      size_t search_index,
      u64 generation,
      RemotePtr pointer,
      std::optional<distance_t> distance) {
    if (!valid_phase_generation(
          search_index, PartitionContinuationWave::score, generation) ||
        pointer.is_null()) {
      return false;
    }
    SearchState& state = searches_[search_index];
    const auto found = state.pending_score_indices.find(pointer);
    if (found == state.pending_score_indices.end()) return false;

    const size_t remove_index = found->second;
    const size_t last_index = state.pending_scores.size() - 1;
    state.pending_score_indices.erase(found);
    if (remove_index != last_index) {
      state.pending_scores[remove_index] = state.pending_scores[last_index];
      state.pending_score_indices.at(
        state.pending_scores[remove_index].pointer) = remove_index;
    }
    state.pending_scores.pop_back();
    if (distance.has_value()) {
      state.search.add_remote(pointer, *distance);
    }

    if (state.pending_scores.empty()) {
      prepare_next_expand_or_finish(search_index, state);
    }
    request_cache_dirty_ = true;
    return true;
  }

  // Atomically complete one search's score generation. Results omitted from
  // scores are terminal/missing candidates. This preserves the complete
  // score-wave dependency within that search while allowing other searches to
  // be in unrelated generations and phases.
  bool consume_score_results(
      size_t search_index,
      u64 generation,
      span<const PartitionContinuationScoreResult> scores) {
    if (!valid_phase_generation(
          search_index, PartitionContinuationWave::score, generation)) {
      return false;
    }
    for (const PartitionContinuationScoreResult& score : scores) {
      if (score.search_index != search_index ||
          (score.generation != 0 && score.generation != generation)) {
        continue;
      }
      resolve_score_request(
        search_index, generation, score.pointer, score.distance);
    }
    // resolve_score_request() swap-erases, so the back pointer is always a
    // valid unresolved request until the generation advances.
    while (valid_phase_generation(
             search_index, PartitionContinuationWave::score, generation)) {
      SearchState& state = searches_[search_index];
      if (state.pending_scores.empty()) break;
      const RemotePtr missing = state.pending_scores.back().pointer;
      resolve_score_request(search_index, generation, missing, std::nullopt);
    }
    return true;
  }

  void consume_score_results(
      span<const PartitionContinuationScoreResult> scores) {
    if (wave() != PartitionContinuationWave::score) {
      throw std::logic_error("score results consumed outside a score wave");
    }
    for (size_t search_index = 0; search_index < search_count_;
         ++search_index) {
      if (searches_[search_index].phase !=
          PartitionContinuationWave::score) {
        continue;
      }
      const u64 current_generation = searches_[search_index].generation;
      consume_score_results(
        search_index, current_generation, scores);
    }
  }

  bool resolve_expand_request(
      size_t search_index,
      u64 generation,
      RemotePtr expanded_pointer,
      span<const RemotePtr> neighbors) {
    if (!valid_phase_generation(
          search_index, PartitionContinuationWave::expand, generation) ||
        expanded_pointer.is_null()) {
      return false;
    }
    SearchState& state = searches_[search_index];
    if (!state.pending_expand.has_value() ||
        state.pending_expand->pointer != expanded_pointer) {
      return false;
    }
    state.pending_expand.reset();
    for (const RemotePtr neighbor : neighbors) {
      if (state.search.try_visit_remote(neighbor)) {
        state.pending_scores.push_back({search_index, neighbor, 0});
      }
    }
    if (!state.pending_scores.empty()) {
      enter_score_phase(search_index, state);
    } else {
      prepare_next_expand_or_finish(search_index, state);
    }
    request_cache_dirty_ = true;
    return true;
  }

  bool resolve_expand_request(
      size_t search_index,
      u64 generation,
      span<const RemotePtr> neighbors) {
    if (!valid_phase_generation(
          search_index, PartitionContinuationWave::expand, generation)) {
      return false;
    }
    const SearchState& state = searches_[search_index];
    if (!state.pending_expand.has_value()) return false;
    return resolve_expand_request(
      search_index, generation, state.pending_expand->pointer, neighbors);
  }

  bool consume_expand_results(
      size_t search_index,
      u64 generation,
      span<const PartitionContinuationExpandResult> neighbors) {
    if (!valid_phase_generation(
          search_index, PartitionContinuationWave::expand, generation)) {
      return false;
    }
    synchronous_neighbor_pointers_.clear();
    for (const PartitionContinuationExpandResult& neighbor : neighbors) {
      if (neighbor.search_index != search_index ||
          (neighbor.generation != 0 &&
           neighbor.generation != generation)) {
        continue;
      }
      synchronous_neighbor_pointers_.push_back(neighbor.pointer);
    }
    return resolve_expand_request(
      search_index, generation,
      span<const RemotePtr>{synchronous_neighbor_pointers_});
  }

  void consume_expand_results(
      span<const PartitionContinuationExpandResult> neighbors) {
    if (wave() != PartitionContinuationWave::expand) {
      throw std::logic_error(
        "expansion results consumed outside an expand wave");
    }
    for (size_t search_index = 0; search_index < search_count_;
         ++search_index) {
      if (searches_[search_index].phase !=
          PartitionContinuationWave::expand) {
        continue;
      }
      const u64 current_generation = searches_[search_index].generation;
      consume_expand_results(
        search_index, current_generation, neighbors);
    }
  }

  const vec<PartitionLocalSearchEntry>& result(size_t search_index) const {
    require_search_index(search_index);
    if (!search_complete(search_index)) {
      throw std::logic_error(
        "continuation search result requested before completion");
    }
    return results_[search_index];
  }

  bool budget_exhausted_result(size_t search_index) const {
    require_completed_search(search_index);
    return exhausted_results_[search_index];
  }

  u64 expansion_count_result(size_t search_index) const {
    require_completed_search(search_index);
    return expansion_results_[search_index];
  }

  const vec<vec<PartitionLocalSearchEntry>>& results() const {
    require_complete();
    return results_;
  }

  const vec<bool>& budget_exhausted_results() const {
    require_complete();
    return exhausted_results_;
  }

  const vec<u64>& expansion_count_results() const {
    require_complete();
    return expansion_results_;
  }

  // Call only after copying results and counters from a completed batch.
  // This invalidates those views, preserves ordinary reusable allocations,
  // and releases only capacities above the caller's retention threshold.
  void trim_oversized_capacity(size_t max_retained_capacity) {
    require_complete();
    for (SearchState& state : searches_) {
      state.search.trim_oversized_capacity(max_retained_capacity);
      state.pending_scores.clear();
      if (state.pending_scores.capacity() > max_retained_capacity) {
        vec<PartitionContinuationScoreRequest>{}.swap(
          state.pending_scores);
      }
      state.pending_score_indices.clear();
      if (state.pending_score_indices.values().capacity() >
          max_retained_capacity) {
        state.pending_score_indices.rehash(0);
      }
    }
    if (searches_.capacity() > max_retained_capacity) {
      vec<SearchState>{}.swap(searches_);
    }

    const auto trim_vector = [max_retained_capacity](auto& values) {
      values.clear();
      if (values.capacity() > max_retained_capacity) {
        using Vector = std::remove_reference_t<decltype(values)>;
        Vector{}.swap(values);
      }
    };
    trim_vector(score_requests_);
    trim_vector(expand_requests_);
    trim_vector(synchronous_score_results_);
    trim_vector(synchronous_expand_results_);
    trim_vector(synchronous_neighbor_pointers_);
    for (vec<PartitionLocalSearchEntry>& result : results_) {
      trim_vector(result);
    }
    if (results_.capacity() > max_retained_capacity) {
      vec<vec<PartitionLocalSearchEntry>>{}.swap(results_);
    }
    trim_vector(exhausted_results_);
    trim_vector(expansion_results_);
    search_count_ = 0;
    activated_count_ = 0;
    initialized_ = false;
    request_cache_dirty_ = true;
  }

  template <typename ScoreBatch, typename ExpandBatch>
  const vec<vec<PartitionLocalSearchEntry>>& run(
      span<const PartitionContinuationSeed> seeds,
      u32 partition_id,
      u32 beam_width,
      PartitionSearchBudget budget,
      ScoreBatch&& score_batch,
      ExpandBatch&& expand_batch,
      vec<bool>* budget_exhausted = nullptr,
      vec<u64>* expansion_counts = nullptr) {
    initialize(seeds, partition_id, beam_width, budget);
    while (!complete()) {
      if (wave() == PartitionContinuationWave::score) {
        synchronous_score_results_.clear();
        std::invoke(
          score_batch, pending_score_requests(),
          [&](size_t search_index, RemotePtr pointer, distance_t distance) {
            synchronous_score_results_.push_back(
              {search_index, pointer, distance});
          });
        consume_score_results(
          span<const PartitionContinuationScoreResult>{
            synchronous_score_results_});
        continue;
      }
      if (wave() != PartitionContinuationWave::expand) {
        throw std::logic_error(
          "continuation entered an invalid synchronous wave");
      }
      synchronous_expand_results_.clear();
      std::invoke(
        expand_batch, pending_expand_requests(),
        [&](size_t search_index, RemotePtr pointer) {
          synchronous_expand_results_.push_back({search_index, pointer});
        });
      consume_expand_results(
        span<const PartitionContinuationExpandResult>{
          synchronous_expand_results_});
    }
    if (budget_exhausted != nullptr) {
      *budget_exhausted = exhausted_results_;
    }
    if (expansion_counts != nullptr) {
      *expansion_counts = expansion_results_;
    }
    return results_;
  }

private:
  struct SearchState {
    SearchState() : search(0, 1) {}

    void reset_metadata() {
      phase = PartitionContinuationWave::uninitialized;
      generation = 0;
      pending_scores.clear();
      pending_score_indices.clear();
      pending_expand.reset();
    }

    PartitionContinuationBeam search;
    PartitionContinuationWave phase{
      PartitionContinuationWave::uninitialized};
    u64 generation{};
    vec<PartitionContinuationScoreRequest> pending_scores;
    dense_hashmap_t<RemotePtr, size_t> pending_score_indices;
    std::optional<PartitionContinuationExpandRequest> pending_expand;
  };

  void require_initialized() const {
    if (!initialized_) {
      throw std::logic_error("continuation batch is not initialized");
    }
  }

  void require_search_index(size_t search_index) const {
    if (!initialized_ || search_index >= search_count_) {
      throw std::out_of_range("continuation search index is out of range");
    }
  }

  void require_complete() const {
    if (!complete()) {
      throw std::logic_error("continuation results requested before completion");
    }
  }

  void require_completed_search(size_t search_index) const {
    require_search_index(search_index);
    if (!search_complete(search_index)) {
      throw std::logic_error(
        "continuation search counter requested before completion");
    }
  }

  bool valid_phase_generation(
      size_t search_index,
      PartitionContinuationWave phase,
      u64 generation) const {
    return initialized_ && search_index < search_count_ && generation != 0 &&
           searches_[search_index].phase == phase &&
           searches_[search_index].generation == generation;
  }

  static void advance_generation(SearchState& state) {
    ++state.generation;
    // Generation zero is reserved for legacy result aggregates that do not
    // carry an asynchronous token.
    if (state.generation == 0) ++state.generation;
  }

  void enter_score_phase(size_t search_index, SearchState& state) {
    advance_generation(state);
    state.phase = PartitionContinuationWave::score;
    state.pending_expand.reset();
    state.pending_score_indices.clear();
    state.pending_score_indices.reserve(state.pending_scores.size());
    for (size_t request_index = 0;
         request_index < state.pending_scores.size(); ++request_index) {
      PartitionContinuationScoreRequest& request =
        state.pending_scores[request_index];
      request.search_index = search_index;
      request.generation = state.generation;
      state.pending_score_indices.emplace(request.pointer, request_index);
    }
  }

  void prepare_next_expand_or_finish(
      size_t search_index, SearchState& state) {
    state.pending_scores.clear();
    state.pending_score_indices.clear();
    state.pending_expand.reset();
    advance_generation(state);
    if (const std::optional<RemotePtr> current =
          state.search.take_closest_unexpanded()) {
      state.phase = PartitionContinuationWave::expand;
      state.pending_expand = PartitionContinuationExpandRequest{
        search_index, *current, state.generation};
      return;
    }
    state.phase = PartitionContinuationWave::complete;
    const auto& beam = state.search.final_beam();
    results_[search_index].assign(beam.begin(), beam.end());
    exhausted_results_[search_index] = state.search.budget_exhausted();
    expansion_results_[search_index] = state.search.expansion_count();
  }

  void rebuild_request_cache() const {
    if (!request_cache_dirty_) return;
    score_requests_.clear();
    expand_requests_.clear();
    for (size_t search_index = 0; search_index < search_count_;
         ++search_index) {
      const SearchState& state = searches_[search_index];
      if (state.phase == PartitionContinuationWave::score) {
        score_requests_.insert(score_requests_.end(),
                               state.pending_scores.begin(),
                               state.pending_scores.end());
      } else if (state.phase == PartitionContinuationWave::expand &&
                 state.pending_expand.has_value()) {
        expand_requests_.push_back(*state.pending_expand);
      }
    }
    request_cache_dirty_ = false;
  }

  size_t search_count_{};
  size_t activated_count_{};
  u32 partition_id_{};
  u32 beam_width_{};
  PartitionSearchBudget budget_{PartitionSearchBudget::unbounded()};
  bool initialized_{};
  vec<SearchState> searches_;
  mutable vec<PartitionContinuationScoreRequest> score_requests_;
  mutable vec<PartitionContinuationExpandRequest> expand_requests_;
  mutable bool request_cache_dirty_{true};
  vec<PartitionContinuationScoreResult> synchronous_score_results_;
  vec<PartitionContinuationExpandResult> synchronous_expand_results_;
  vec<RemotePtr> synchronous_neighbor_pointers_;
  vec<vec<PartitionLocalSearchEntry>> results_;
  vec<bool> exhausted_results_;
  vec<u64> expansion_results_;
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
