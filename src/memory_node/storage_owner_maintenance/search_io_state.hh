#pragma once

#include <algorithm>
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

// One Stage2 context owns this state for the lifetime of its continuation.
// Registered RDMA scratch remains lane-owned, but the logical beam, pending
// home RPCs, and retry state must survive releasing that lane while the task is
// waiting on a dependency that does not reference the scratch buffers.
enum class Stage2SearchIoPhase : std::uint8_t {
  idle,
  score_body_ready,
  score_body_pending,
  score_header_ready,
  score_header_pending,
  graph_ready,
  graph_pending,
  score_home_pending,
  graph_home_pending,
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
  bool requires_after_header{};
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

// Before paying to return speculative adjacency data with a home expansion,
// measure the strongest candidates already present in that response.  The
// predictor is observation-only: it neither resolves a continuation request
// nor changes beam order.  Keeping the best two distinct pointers lets the
// benchmark evaluate the wire-cost break-even point for depth one and two.
struct Stage2GraphPrefetchPrediction {
  RemotePtr first;
  RemotePtr second;
  distance_t first_distance{std::numeric_limits<distance_t>::max()};
  distance_t second_distance{std::numeric_limits<distance_t>::max()};

  void clear() {
    first = RemotePtr{};
    second = RemotePtr{};
    first_distance = std::numeric_limits<distance_t>::max();
    second_distance = std::numeric_limits<distance_t>::max();
  }

  [[nodiscard]] bool empty() const { return first.is_null(); }

  void observe(RemotePtr pointer, distance_t distance) {
    if (pointer.is_null()) return;
    distance = memory_node_storage_owner_index_detail::
      normalize_partition_search_distance(distance);
    const auto better = [](RemotePtr lhs, distance_t lhs_distance,
                           RemotePtr rhs, distance_t rhs_distance) {
      return rhs.is_null() || lhs_distance < rhs_distance ||
        (lhs_distance == rhs_distance &&
         lhs.raw_address < rhs.raw_address);
    };
    if (pointer == first) {
      first_distance = std::min(first_distance, distance);
      return;
    }
    if (pointer == second) {
      second_distance = std::min(second_distance, distance);
      if (better(second, second_distance, first, first_distance)) {
        std::swap(first, second);
        std::swap(first_distance, second_distance);
      }
      return;
    }
    if (better(pointer, distance, first, first_distance)) {
      second = first;
      second_distance = first_distance;
      first = pointer;
      first_distance = distance;
    } else if (better(pointer, distance, second, second_distance)) {
      second = pointer;
      second_distance = distance;
    }
  }

  [[nodiscard]] u32 rank(RemotePtr pointer) const {
    if (!pointer.is_null() && pointer == first) return 1;
    if (!pointer.is_null() && pointer == second) return 2;
    return 0;
  }
};

struct Stage2HomeExpandRpc {
  u32 target_shard{};
  u32 item_count{};
  u64 request_id{};
  u64 deadline_ns{};
  bool posted{};
  bool complete{};
  vec<byte_t> request;
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

// Registered lane scratch is consumed only by a distinct remote physical
// READ.  Local/terminal work and another logical consumer of a pointer already
// selected in this dispatch need no additional scratch record.  Keep this
// predicate separate from transport-credit admission: both limits must accept
// a distinct remote pointer before the collector records it.
constexpr bool stage2_consumer_fits_physical_scratch(
    bool distinct_remote,
    std::size_t physical_reads,
    std::size_t physical_read_capacity) {
  return !distinct_remote || physical_reads < physical_read_capacity;
}

// score-many carries logical score operations in a bounded two-sided message;
// it neither consumes one registered snapshot slot per pointer nor posts one
// RDMA READ WR per pointer.  Reusing the one-sided quota here fragmented a
// 256-item wire message into roughly 30 items/RPC in production. Admit at most
// one full RPC per remote peer in a dispatch, while local/terminal work remains
// unbounded because it creates no outbound message.
struct Stage2ScoreManyDispatchQuota {
  std::span<u32> items_by_peer{};
  u32 items_per_peer{};

  void reset(std::span<u32> new_items_by_peer,
             u32 new_items_per_peer) {
    items_by_peer = new_items_by_peer;
    items_per_peer = new_items_per_peer;
    std::fill(items_by_peer.begin(), items_by_peer.end(), 0);
  }

  [[nodiscard]] bool try_accept(u32 peer, bool remote) {
    if (!remote) return true;
    if (items_per_peer == 0 || peer >= items_by_peer.size() ||
        items_by_peer[peer] >= items_per_peer) {
      return false;
    }
    ++items_by_peer[peer];
    return true;
  }
};

// A half-full message is the conservative break-even gate for enabling the
// two-sided path. Small dependency-generated waves are latency dominated in
// the SIFT100M result even though their byte count is lower; they remain on
// the one-sided path. Large frontier waves can use score-many without letting
// one sparse destination force small RPCs to every peer.
constexpr u32 stage2_score_many_min_items(u32 message_capacity) {
  return message_capacity == 0 ? 0 : (message_capacity + 1) / 2;
}

constexpr bool stage2_score_many_peer_eligible(
    std::size_t pending_items, u32 message_capacity) {
  const u32 minimum = stage2_score_many_min_items(message_capacity);
  return minimum != 0 && pending_items >= minimum;
}

struct Stage2SearchIoState {
  bool initialized{};
  Stage2SearchIoPhase phase{Stage2SearchIoPhase::idle};
  bool ordered_snapshot_pairs{};
  bool score_many_dispatch{};
  bool prefer_graph{};
  std::size_t round_robin_search{};

  // The continuation is context-owned because it contains the private beam
  // and visited set for every task in the compacted context. A context may
  // release and later reacquire a physical scratch lane while its home RPCs
  // are in flight without restarting or changing the logical search.
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
  hashset_t<RemotePtr> score_selected_remote;
  vec<std::size_t> score_order;
  vec<std::size_t> score_group_offsets;
  vec<RemotePtr> score_unique;
  vec<Stage2PendingVectorRead> pending_vectors;
  vec<Stage2HomeExpandRpc> score_home_rpcs;
  std::size_t score_home_rpc_count{};

  vec<Stage2GraphConsumer> graph_consumers;
  hashset_t<RemotePtr> graph_selected_remote;
  vec<std::size_t> graph_order;
  vec<std::size_t> graph_group_offsets;
  vec<RemotePtr> graph_unique;
  vec<Stage2PendingGraphRead> pending_graph;
  vec<vec<RemotePtr>> graph_neighbors;
  vec<Stage2GraphRetryState> graph_retry_state;
  vec<Stage2GraphPrefetchPrediction> graph_prefetch_predictions;
  vec<Stage2HomeExpandRpc> home_expand_rpcs;
  std::size_t home_expand_rpc_count{};

  void reset() {
    initialized = false;
    phase = Stage2SearchIoPhase::idle;
    ordered_snapshot_pairs = false;
    score_many_dispatch = false;
    prefer_graph = false;
    round_robin_search = 0;
    for (auto& beam : local_beams) beam.clear();
    seeds.clear();
    search_seeded.clear();
    score_collect_cursors.clear();
    score_consumers.clear();
    score_selected_remote.clear();
    score_order.clear();
    score_group_offsets.clear();
    score_unique.clear();
    pending_vectors.clear();
    score_home_rpc_count = 0;
    for (auto& rpc : score_home_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.request.clear();
    }
    graph_consumers.clear();
    graph_selected_remote.clear();
    graph_order.clear();
    graph_group_offsets.clear();
    graph_unique.clear();
    pending_graph.clear();
    for (vec<RemotePtr>& neighbors : graph_neighbors) neighbors.clear();
    graph_retry_state.clear();
    graph_prefetch_predictions.clear();
    home_expand_rpc_count = 0;
    for (auto& rpc : home_expand_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.request.clear();
    }
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
    // The selected-pointer sets can never grow beyond the fixed physical
    // scratch capacity, unlike the logical continuation vectors trimmed here.
    trim(score_order);
    trim(score_group_offsets);
    trim(score_unique);
    trim(pending_vectors);
    trim(score_home_rpcs);
    trim(graph_consumers);
    trim(graph_order);
    trim(graph_group_offsets);
    trim(graph_unique);
    trim(pending_graph);
    for (auto& neighbors : graph_neighbors) trim(neighbors);
    trim(graph_neighbors);
    trim(graph_prefetch_predictions);
    // At most one request buffer per peer is retained. Its capacity is
    // bounded by storage_owner_batch_max * (metadata + vector bytes), and
    // retaining it avoids an allocator round trip for every graph wave.
    trim(home_expand_rpcs);
    trim(graph_retry_state);
  }

  [[nodiscard]] bool idle() const {
    return phase == Stage2SearchIoPhase::idle && !initialized;
  }

  // A lane may be rebound only when no live object in this state points into
  // its registered scratch. Home-executed graph expansion owns request and
  // response payloads independently, so waiting for those RPCs does not pin
  // the RDMA lane. This increases latency-hiding concurrency without adding a
  // search budget or acknowledging a task before durable completion.
  [[nodiscard]] bool scratch_rebindable() const {
    return idle() ||
      ((phase == Stage2SearchIoPhase::graph_home_pending ||
        phase == Stage2SearchIoPhase::score_home_pending) &&
       pending_vectors.empty() && pending_graph.empty());
  }
};

}  // namespace memory_node_storage_owner_maintenance_detail
