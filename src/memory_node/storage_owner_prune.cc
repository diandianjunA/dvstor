#include "memory_node/memory_node.hh"
#include "memory_node/storage_owner_helpers.hh"

#include <algorithm>
#include <chrono>

namespace {

using Configuration = configuration::IndexConfiguration;
using NodeSnapshot = memory_node_detail::NodeSnapshot;

using memory_node_detail::storage_owner_prune_candidate_limit;
using memory_node_detail::storage_owner_snapshot_batch_size;

}  // namespace

double MemoryNode::storage_owner_candidate_overlap(const vec<RemotePtr>& lhs,
                                                   const vec<RemotePtr>& rhs,
                                                   u32 limit) {
  const size_t lhs_size = std::min<size_t>(lhs.size(), limit);
  const size_t rhs_size = std::min<size_t>(rhs.size(), limit);
  const size_t denominator = std::max<size_t>(1, std::min(lhs_size, rhs_size));
  hashset_t<RemotePtr> left;
  left.reserve(lhs_size);
  for (size_t i = 0; i < lhs_size; ++i) left.insert(lhs[i]);
  size_t matches = 0;
  for (size_t i = 0; i < rhs_size; ++i) {
    if (left.contains(rhs[i])) ++matches;
  }
  return static_cast<double>(matches) / static_cast<double>(denominator);
}

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown) {
  struct CandidateInfo {
    RemotePtr rptr;
    distance_t dist{};
    vec<byte_t> vector_data;
  };

  vec<CandidateInfo> infos;
  infos.reserve(candidates.size());
  vec<RemotePtr> filtered;
  filtered.reserve(std::min<size_t>(candidates.size(), storage_owner_prune_candidate_limit(config)));
  const u32 prune_candidate_limit = storage_owner_prune_candidate_limit(config);
  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
    if (filtered.size() >= prune_candidate_limit) {
      break;
    }
  }

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
  for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
    const size_t end = std::min(filtered.size(), begin + snapshot_batch);
    vec<RemotePtr> batch;
    batch.reserve(end - begin);
    batch.insert(batch.end(), filtered.begin() + begin, filtered.begin() + end);
    auto t_snapshot = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) {
        continue;
      }
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_between_vectors(source, source_dtype,
                                                       snapshot.vector_data.data(), VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({snapshot.rptr, dist, std::move(snapshot.vector_data)});
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  vec<RemotePtr> selected;
  selected.reserve(config.R);
  vec<const byte_t*> selected_vectors;
  selected_vectors.reserve(config.R);

  for (const auto& candidate : infos) {
    if (selected.size() >= config.R) {
      break;
    }

    bool pruned = false;
    for (idx_t i = 0; i < selected_vectors.size(); ++i) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_between_vectors(candidate.vector_data.data(), VamanaNode::vector_dtype(),
                                                           selected_vectors[i], VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      if (config.alpha * pair_dist <= candidate.dist) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      selected.push_back(candidate.rptr);
      selected_vectors.push_back(candidate.vector_data.data());
    }
  }

  return selected;
}
