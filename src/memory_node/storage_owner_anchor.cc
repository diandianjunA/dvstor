#include "memory_node/memory_node.hh"

#include <algorithm>
#include <limits>

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

u32 snapshot_batch_limit(const Configuration& config, const StorageOwnerThread* thread) {
  const u32 configured = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0, "storage-owner anchor search scratch cannot hold a snapshot");
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

i32 closest_unexpanded(const vec<BeamEntry>& beam) {
  i32 best = -1;
  distance_t best_distance = std::numeric_limits<distance_t>::max();
  for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
    if (!beam[i].expanded && beam[i].distance < best_distance) {
      best = i;
      best_distance = beam[i].distance;
    }
  }
  return best;
}

vec<RemotePtr> sorted_candidates(vec<BeamEntry>& beam) {
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  for (const BeamEntry& entry : beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}

}  // namespace

vec<RemotePtr> MemoryNode::anchor_search_candidates(const span<const element_t> query,
                                                     const vec<RemotePtr>& anchor_hints,
                                                     const Configuration& config,
                                                     InsertBreakdownCounters* breakdown,
                                                     bool local_only) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = snapshot_batch_limit(config, current_storage_owner_thread_);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    vec<RemotePtr> batch;
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(hint.memory_node())) &&
          visited.insert(hint).second) {
        batch.push_back(hint);
      }
    }
    auto started = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
    }
    for (const NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) continue;
      started = std::chrono::steady_clock::now();
      const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        ++breakdown->storage_owner_anchor_valid_hints;
      }
      insert_into_beam(beam, snapshot.rptr, distance, beam_width);
    }
  }

  u32 expansions = 0;
  u32 remote_expansions = 0;
  while (expansions < config.storage_owner_anchor_expand_cap) {
    auto started = std::chrono::steady_clock::now();
    const i32 best = closest_unexpanded(beam);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(started);
    }
    if (best < 0) break;

    BeamEntry& entry = beam[best];
    entry.expanded = true;
    const bool remote = !local_shard(entry.rptr.memory_node());
    if (local_only && remote) {
      continue;
    }
    if (remote && remote_expansions >= config.storage_owner_anchor_remote_rescue_cap) {
      continue;
    }
    ++expansions;
    if (remote) ++remote_expansions;

    started = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = read_neighbor_list(entry.rptr);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    vec<RemotePtr> unvisited;
    unvisited.reserve(neighbors.size());
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() && neighbor.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(neighbor.memory_node())) &&
          visited.insert(neighbor).second) {
        unvisited.push_back(neighbor);
      }
    }

    for (size_t begin = 0; begin < unvisited.size(); begin += batch_limit) {
      const size_t end = std::min(unvisited.size(), begin + batch_limit);
      vec<RemotePtr> batch(unvisited.begin() + begin, unvisited.begin() + end);
      started = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) continue;
        started = std::chrono::steady_clock::now();
        const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        }
        insert_into_beam(beam, snapshot.rptr, distance, beam_width);
      }
    }
  }

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_expansions += expansions;
    breakdown->storage_owner_anchor_remote_expansions += remote_expansions;
  }
  auto started = std::chrono::steady_clock::now();
  vec<RemotePtr> candidates = sorted_candidates(beam);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  return candidates;
}
