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
  bool speculative{};
  RemotePtr expansion_pointer;
};

struct Stage2GraphConsumer {
  std::size_t search_index{};
  u64 generation{};
  RemotePtr pointer;
  bool speculative{};
};

struct Stage2GraphRetryState {
  RemotePtr pointer;
  u32 attempt{};
};

struct Stage2HomeExpandRpc {
  u32 target_shard{};
  u32 item_count{};
  u64 request_id{};
  u64 deadline_ns{};
  bool posted{};
  bool complete{};
  vec<byte_t> request;
  // Graph responses need to distinguish authoritative items from ordered
  // speculative items without changing the wire protocol.
  vec<std::size_t> graph_consumer_indexes;
  // Score responses use request order on the wire. Keep the corresponding
  // logical consumer so piggybacked speculative scores never advance a beam.
  vec<std::size_t> score_consumer_indexes;
};

struct Stage2PrefetchedGraphNeighbor {
  RemotePtr pointer;
  distance_t distance{};
  u32 disposition{};
  bool score_prefetched{};
  u32 score_prefetch_issues{};
};

struct Stage2PrefetchedGraphExpansion {
  RemotePtr pointer;
  u32 disposition{};
  vec<Stage2PrefetchedGraphNeighbor> neighbors;
};

// Global promotion feedback is intentionally conservative. Width four is a
// bounded warm-up; only a measured >=70% promotion rate unlocks the configured
// width. Below that threshold new speculation stops, while already cached
// records remain eligible for exact ordered commit. At p=0.70, width 16 does
// only 2.6% more graph work than width 8, but halves its remaining graph RPCs.
constexpr u32 stage2_ordered_issue_width(
    u64 hits, u64 wasted, u32 configured_max_width) {
  const u32 maximum = std::max<u32>(1, configured_max_width);
  if (maximum == 1) return 1;
  const u64 outcomes = hits + wasted;
  if (outcomes < 512) return std::min<u32>(4, maximum);
  const auto promotion_ratio =
      static_cast<long double>(hits) / static_cast<long double>(outcomes);
  if (promotion_ratio < 0.70L) return 1;
  return maximum;
}

constexpr bool stage2_score_prefetch_enabled(u64 hits, u64 wasted) {
  const u64 outcomes = hits + wasted;
  if (outcomes < 512) return true;
  return static_cast<long double>(hits) /
           static_cast<long double>(outcomes) >= 0.70L;
}

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
  vec<Stage2HomeExpandRpc> home_expand_rpcs;
  std::size_t home_expand_rpc_count{};
  vec<vec<Stage2PrefetchedGraphExpansion>> graph_prefetch_cache;

  [[nodiscard]] bool graph_prefetch_contains(
      std::size_t search_index, RemotePtr pointer) const {
    if (search_index >= graph_prefetch_cache.size()) return false;
    const auto& cache = graph_prefetch_cache[search_index];
    return std::any_of(cache.begin(), cache.end(), [&](const auto& entry) {
      return entry.pointer == pointer;
    });
  }

  [[nodiscard]] std::size_t graph_prefetch_size(
      std::size_t search_index) const {
    return search_index < graph_prefetch_cache.size()
      ? graph_prefetch_cache[search_index].size() : 0;
  }

  bool insert_graph_prefetch(
      std::size_t search_index, Stage2PrefetchedGraphExpansion entry,
      std::size_t capacity) {
    if (search_index >= graph_prefetch_cache.size() || capacity == 0) {
      return false;
    }
    auto& cache = graph_prefetch_cache[search_index];
    if (cache.size() >= capacity || graph_prefetch_contains(
          search_index, entry.pointer)) {
      return false;
    }
    cache.push_back(std::move(entry));
    return true;
  }

  bool resolve_graph_prefetch_score(
      std::size_t search_index, RemotePtr expansion_pointer,
      RemotePtr neighbor_pointer, std::optional<distance_t> distance,
      u32 disposition) {
    if (search_index >= graph_prefetch_cache.size()) return false;
    auto& cache = graph_prefetch_cache[search_index];
    const auto expansion = std::find_if(
      cache.begin(), cache.end(), [&](const auto& entry) {
        return entry.pointer == expansion_pointer;
      });
    if (expansion == cache.end()) return false;
    const auto neighbor = std::find_if(
      expansion->neighbors.begin(), expansion->neighbors.end(),
      [&](const auto& entry) { return entry.pointer == neighbor_pointer; });
    if (neighbor == expansion->neighbors.end()) return false;
    neighbor->score_prefetched = true;
    neighbor->disposition = disposition;
    if (distance.has_value()) {
      neighbor->distance = *distance;
    }
    return true;
  }

  [[nodiscard]] u64 graph_prefetched_score_count() const {
    u64 count = 0;
    for (const auto& cache : graph_prefetch_cache) {
      for (const auto& expansion : cache) {
        for (const auto& neighbor : expansion.neighbors) {
          count += neighbor.score_prefetch_issues;
        }
      }
    }
    return count;
  }

  std::optional<Stage2PrefetchedGraphExpansion> take_graph_prefetch(
      std::size_t search_index, RemotePtr pointer) {
    if (search_index >= graph_prefetch_cache.size()) return std::nullopt;
    auto& cache = graph_prefetch_cache[search_index];
    const auto found = std::find_if(
      cache.begin(), cache.end(), [&](const auto& entry) {
        return entry.pointer == pointer;
      });
    if (found == cache.end()) return std::nullopt;
    Stage2PrefetchedGraphExpansion result = std::move(*found);
    if (found != cache.end() - 1) *found = std::move(cache.back());
    cache.pop_back();
    return result;
  }

  [[nodiscard]] u64 graph_prefetch_entry_count() const {
    u64 count = 0;
    for (const auto& cache : graph_prefetch_cache) count += cache.size();
    return count;
  }

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
      rpc.score_consumer_indexes.clear();
    }
    graph_consumers.clear();
    graph_selected_remote.clear();
    graph_order.clear();
    graph_group_offsets.clear();
    graph_unique.clear();
    pending_graph.clear();
    for (vec<RemotePtr>& neighbors : graph_neighbors) neighbors.clear();
    graph_retry_state.clear();
    for (auto& cache : graph_prefetch_cache) cache.clear();
    home_expand_rpc_count = 0;
    for (auto& rpc : home_expand_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.request.clear();
      rpc.graph_consumer_indexes.clear();
      rpc.score_consumer_indexes.clear();
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
    // At most one request buffer per peer is retained. Its capacity is
    // bounded by storage_owner_batch_max * (metadata + vector bytes), and
    // retaining it avoids an allocator round trip for every graph wave.
    trim(home_expand_rpcs);
    trim(graph_retry_state);
    for (auto& cache : graph_prefetch_cache) {
      for (auto& entry : cache) trim(entry.neighbors);
      trim(cache);
    }
    trim(graph_prefetch_cache);
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
