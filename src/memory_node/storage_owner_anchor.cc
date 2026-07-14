#include "memory_node/memory_node.hh"

#include <algorithm>
#include <limits>

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerCoroutineScratch = memory_node_detail::StorageOwnerCoroutineScratch;
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

void MemoryNode::score_local_candidates_into_beam(
    const span<const element_t> query,
    const vec<RemotePtr>& candidates,
    const Configuration& config,
    vec<BeamEntry>& beam,
    u32 beam_width,
    InsertBreakdownCounters* breakdown,
    bool count_valid_hints) const {
  for (const RemotePtr candidate : candidates) {
    auto started = std::chrono::steady_clock::now();
    const byte_t* vector = local_live_vector(candidate);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
    }
    if (vector == nullptr) {
      continue;
    }

    started = std::chrono::steady_clock::now();
    const distance_t distance = distance_to_stored_vector(query, vector, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
      if (count_valid_hints) {
        ++breakdown->storage_owner_anchor_valid_hints;
      }
    }
    started = std::chrono::steady_clock::now();
    insert_into_beam(beam, candidate, distance, beam_width);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(started);
    }
  }
}

vec<RemotePtr> MemoryNode::anchor_search_candidates(const span<const element_t> query,
                                                     const vec<RemotePtr>& anchor_hints,
                                                     const Configuration& config,
                                                     InsertBreakdownCounters* breakdown,
                                                     bool local_only) {
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  hashset_t<RemotePtr> local_visited;
  vec<BeamEntry> local_beam;
  vec<RemotePtr> local_neighbors;
  vec<RemotePtr> local_batch;
  vec<RemotePtr> local_unvisited;
  vec<byte_t> local_neighbor_entry;
  vec<byte_t> local_neighbor_decoded;
  if (scratch != nullptr) {
    scratch->clear_search();
  }
  hashset_t<RemotePtr>& visited = scratch != nullptr ? scratch->visited : local_visited;
  vec<BeamEntry>& beam = scratch != nullptr ? scratch->beam : local_beam;
  vec<RemotePtr>& neighbors = scratch != nullptr ? scratch->neighbors : local_neighbors;
  vec<RemotePtr>& batch = scratch != nullptr ? scratch->batch : local_batch;
  vec<RemotePtr>& unvisited = scratch != nullptr ? scratch->unvisited : local_unvisited;
  vec<byte_t>& neighbor_entry = scratch != nullptr
                                  ? scratch->neighbor_entry
                                  : local_neighbor_entry;
  vec<byte_t>& neighbor_decoded = scratch != nullptr
                                    ? scratch->neighbor_decoded
                                    : local_neighbor_decoded;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = snapshot_batch_limit(config, current_storage_owner_thread_);
  visited.reserve(anchor_hints.size() +
                  static_cast<size_t>(config.storage_owner_anchor_expand_cap) *
                    std::max<u32>(1, config.R));
  beam.reserve(beam_width);
  batch.reserve(batch_limit);
  unvisited.reserve(config.R);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.clear();
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(hint.memory_node())) &&
          visited.insert(hint).second) {
        batch.push_back(hint);
      }
    }
    if (local_only) {
      score_local_candidates_into_beam(query, batch, config, beam, beam_width,
                                       breakdown, true);
    } else {
      auto started = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) continue;
        started = std::chrono::steady_clock::now();
        const distance_t distance = distance_to_stored_vector(
          query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
          ++breakdown->storage_owner_anchor_valid_hints;
        }
        started = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, distance, beam_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(started);
        }
      }
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
    if (local_only) {
      read_local_neighbor_list(entry.rptr, neighbors, neighbor_entry,
                               neighbor_decoded);
    } else {
      neighbors = read_neighbor_list(entry.rptr);
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    unvisited.clear();
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() && neighbor.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(neighbor.memory_node())) &&
          visited.insert(neighbor).second) {
        unvisited.push_back(neighbor);
      }
    }

    for (size_t begin = 0; begin < unvisited.size(); begin += batch_limit) {
      const size_t end = std::min(unvisited.size(), begin + batch_limit);
      batch.clear();
      batch.insert(batch.end(), unvisited.begin() + begin, unvisited.begin() + end);
      if (local_only) {
        score_local_candidates_into_beam(query, batch, config, beam, beam_width,
                                         breakdown, false);
      } else {
        started = std::chrono::steady_clock::now();
        vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
        }
        for (const NodeSnapshot& snapshot : snapshots) {
          if (snapshot.deleted) continue;
          started = std::chrono::steady_clock::now();
          const distance_t distance = distance_to_stored_vector(
            query, snapshot.vector_data.data(), config);
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
          }
          started = std::chrono::steady_clock::now();
          insert_into_beam(beam, snapshot.rptr, distance, beam_width);
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(started);
          }
        }
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
