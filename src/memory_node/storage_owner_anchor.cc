#include "memory_node/memory_node.hh"
#include "memory_node/storage_owner_helpers.hh"

#include <algorithm>
#include <chrono>
#include <limits>

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

using memory_node_detail::storage_owner_snapshot_batch_size;

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
                                                     InsertBreakdownCounters* breakdown) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    vec<RemotePtr> batch;
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ && visited.insert(hint).second) {
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

auto MemoryNode::anchor_search_candidates_async(const span<const element_t> query,
                                                const vec<RemotePtr>& anchor_hints,
                                                const Configuration& config,
                                                StorageOwnerThread& thread,
                                                InsertBreakdownCounters* breakdown)
  -> StorageOwnerInsertCoroutine {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = storage_owner_snapshot_batch_size(config, &thread);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    vec<RemotePtr> batch;
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ && visited.insert(hint).second) {
        batch.push_back(hint);
      }
    }
    auto started = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
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
    if (remote && remote_expansions >= config.storage_owner_anchor_remote_rescue_cap) {
      continue;
    }
    ++expansions;
    if (remote) ++remote_expansions;

    started = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(entry.rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    vec<RemotePtr> unvisited;
    unvisited.reserve(neighbors.size());
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() && neighbor.memory_node() < num_storage_nodes_ &&
          visited.insert(neighbor).second) {
        unvisited.push_back(neighbor);
      }
    }

    for (size_t begin = 0; begin < unvisited.size(); begin += batch_limit) {
      const size_t end = std::min(unvisited.size(), begin + batch_limit);
      vec<RemotePtr> batch(unvisited.begin() + begin, unvisited.begin() + end);
      started = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
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
  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto started = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  for (const BeamEntry& entry : beam) out.push_back(entry.rptr);
}
