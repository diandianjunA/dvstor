#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "common/types.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_maintenance_detail {

// One Stage2 search lane owns this state while a context is suspended at an
// RDMA dependency boundary.  The vectors retain their high-water capacity when
// the lane is reused, so allocation is bounded by the small lane pool rather
// than by the number or history of Stage2 contexts.
enum class Stage2SearchIoPhase : std::uint8_t {
  idle,
  score_body_ready,
  score_body_pending,
  score_header_ready,
  score_header_pending,
  graph_ready,
  graph_pending,
};

enum class Stage2SearchAdvanceResult : std::uint8_t {
  waiting_rdma,
  posted_rdma,
  complete,
};

struct Stage2PendingVectorRead {
  std::size_t group_index{};
  RemotePtr pointer;
  byte_t* buffer{};
  byte_t* after_header{};
  u64 before{};
  u32 slot_incarnation{};
  u32 attempt{};
};

struct Stage2PendingGraphRead {
  std::size_t unique_index{};
  RemotePtr pointer;
  byte_t* buffer{};
  u32 attempt{};
};

struct Stage2ScoreConsumer {
  std::size_t search_index{};
  u64 generation{};
  RemotePtr pointer;
};

struct Stage2GraphConsumer {
  std::size_t search_index{};
  u64 generation{};
  RemotePtr pointer;
};

struct Stage2GraphRetryState {
  RemotePtr pointer;
  u32 attempt{};
};

// A score generation may expose more candidates than fit in one transport
// dispatch. Retryable snapshots remain unresolved, so restarting at element
// zero on every dispatch can repeatedly select the same candidate and starve
// the rest of that logical search. Keep one cursor per search across
// dispatches, while selections_in_dispatch prevents wraparound from selecting
// one request twice in the same finite transport batch.
//
// pending_score_requests() swap-erases resolved requests. Normalize the
// cursor against the current request count on every take. This changes only
// transport scheduling order, not the continuation's beam semantics.
struct Stage2ScoreRoundRobinCursor {
  std::size_t next_position{};
  std::size_t selections_in_dispatch{};

  void begin_dispatch() { selections_in_dispatch = 0; }

  [[nodiscard]] std::optional<std::size_t> take(
      std::size_t request_count) {
    if (request_count == 0 ||
        selections_in_dispatch >= request_count) {
      return std::nullopt;
    }
    next_position %= request_count;
    const std::size_t selected = next_position;
    next_position = selected + 1;
    if (next_position == request_count) next_position = 0;
    ++selections_in_dispatch;
    return selected;
  }
};

struct Stage2SearchIoState {
  bool initialized{};
  Stage2SearchIoPhase phase{Stage2SearchIoPhase::idle};
  bool ordered_snapshot_pairs{};
  bool prefer_graph{};
  std::size_t round_robin_search{};

  // The continuation is lane-owned because it contains the private beam and
  // visited set for every task in the compacted context.  A worker may have
  // several lanes suspended on independent CQ completions at once.
  memory_node_storage_owner_index_detail::PartitionContinuationBatch
    continuation;
  vec<vec<memory_node_storage_owner_index_detail::PartitionLocalSearchEntry>>
    local_beams;
  vec<memory_node_storage_owner_index_detail::PartitionContinuationSeed>
    seeds;
  vec<u8> search_seeded;
  vec<Stage2ScoreRoundRobinCursor> score_collect_cursors;

  // A dispatch is a finite transport batch, not an algorithmic wave.  Each
  // consumer carries the continuation generation that produced it, while the
  // sorted group table reads one physical pointer and scatters the stable or
  // terminal outcome to every matching search.
  vec<Stage2ScoreConsumer> score_consumers;
  vec<std::size_t> score_order;
  vec<std::size_t> score_group_offsets;
  vec<RemotePtr> score_unique;
  vec<Stage2PendingVectorRead> pending_vectors;

  vec<Stage2GraphConsumer> graph_consumers;
  vec<std::size_t> graph_order;
  vec<std::size_t> graph_group_offsets;
  vec<RemotePtr> graph_unique;
  vec<Stage2PendingGraphRead> pending_graph;
  vec<vec<RemotePtr>> graph_neighbors;
  vec<Stage2GraphRetryState> graph_retry_state;

  void reset() {
    initialized = false;
    phase = Stage2SearchIoPhase::idle;
    ordered_snapshot_pairs = false;
    prefer_graph = false;
    round_robin_search = 0;
    for (auto& beam : local_beams) beam.clear();
    seeds.clear();
    search_seeded.clear();
    score_collect_cursors.clear();
    score_consumers.clear();
    score_order.clear();
    score_group_offsets.clear();
    score_unique.clear();
    pending_vectors.clear();
    graph_consumers.clear();
    graph_order.clear();
    graph_group_offsets.clear();
    graph_unique.clear();
    pending_graph.clear();
    for (vec<RemotePtr>& neighbors : graph_neighbors) neighbors.clear();
    graph_retry_state.clear();
  }

  // The continuation must be complete and its results already copied.  This
  // bounds only lane-cache retention after an exceptional search; it is not
  // an in-flight candidate/frontier budget.
  void reset_completed(std::size_t max_retained_capacity) {
    continuation.trim_oversized_capacity(max_retained_capacity);
    reset();

    const auto trim = [max_retained_capacity](auto& values) {
      if (values.capacity() > max_retained_capacity) {
        using Vector = std::remove_reference_t<decltype(values)>;
        Vector{}.swap(values);
      }
    };
    for (auto& beam : local_beams) trim(beam);
    trim(local_beams);
    trim(seeds);
    trim(search_seeded);
    trim(score_collect_cursors);
    trim(score_consumers);
    trim(score_order);
    trim(score_group_offsets);
    trim(score_unique);
    trim(pending_vectors);
    trim(graph_consumers);
    trim(graph_order);
    trim(graph_group_offsets);
    trim(graph_unique);
    trim(pending_graph);
    for (auto& neighbors : graph_neighbors) trim(neighbors);
    trim(graph_neighbors);
    trim(graph_retry_state);
  }

  [[nodiscard]] bool idle() const {
    return phase == Stage2SearchIoPhase::idle && !initialized;
  }
};

}  // namespace memory_node_storage_owner_maintenance_detail
