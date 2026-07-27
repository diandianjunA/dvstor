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

struct Stage2SearchIoState {
  bool initialized{};
  Stage2SearchIoPhase phase{Stage2SearchIoPhase::idle};
  bool ordered_snapshot_pairs{};
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
