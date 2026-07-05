#include "memory_node/memory_node.hh"
#include "memory_node/storage_owner_helpers.hh"

#include <chrono>
#include <cstring>

#include "vamana/storage_layout_resolver.hh"

namespace {

using Configuration = configuration::IndexConfiguration;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

using memory_node_detail::anchored_update_enabled;

}  // namespace

void MemoryNode::publish_mutation(node_t id, RemotePtr ptr, u32 generation, bool deleted) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  idmap_[id] = FreshnessEntry{ptr, generation, deleted};
  mutations_inflight_.erase(id);
}

service::storage_owner::MutationStatus MemoryNode::prepare_mutation(
    node_t id,
    service::storage_owner::MutationKind kind,
    FreshnessEntry* old_entry,
    u32* new_generation) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  if (mutations_inflight_.contains(id)) {
    return service::storage_owner::MutationStatus::failed;
  }
  auto it = idmap_.find(id);
  const bool exists = it != idmap_.end();
  const bool live = exists && !it->second.deleted;
  if (old_entry != nullptr) {
    *old_entry = exists ? it->second : FreshnessEntry{};
  }
  const u32 previous_generation = exists ? it->second.generation : 0;
  if (new_generation != nullptr) {
    *new_generation = previous_generation + 1;
  }
  if (kind == service::storage_owner::MutationKind::insert && live) {
    return service::storage_owner::MutationStatus::already_exists;
  }
  if (kind == service::storage_owner::MutationKind::erase) {
    if (!exists) return service::storage_owner::MutationStatus::not_found;
    if (!live) return service::storage_owner::MutationStatus::already_deleted;
  }
  mutations_inflight_.insert(id);
  return service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::mark_node_deleted(RemotePtr rptr, u32 generation) {
  if (rptr.is_null()) return true;
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  const bool remote = !local_shard(rptr.memory_node());
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED), std::memory_order_acq_rel);
  } else {
    lock_node(rptr);
    u64 header = 0;
    remote_read_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
    header |= static_cast<u64>(VamanaNode::HEADER_DELETED);
    remote_write_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
  }
  if (VamanaNode::compact_storage()) {
    vec<byte_t> entry(VamanaNode::hot_graph_entry_size(), 0);
    VamanaNode::encode_hot_graph_entry(entry.data(), 0, 0, nullptr, 0,
      VamanaNode::HOT_GRAPH_SHARD_BITS, generation, 2, true);
    const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
    if (local_shard(rptr.memory_node())) {
      std::memcpy(index_buffer_.get_full_buffer() + hot_offset, entry.data(), entry.size());
    } else {
      remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
    }
  }
  if (remote) {
    unlock_node(rptr);
  }
  return true;
}

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            std::unordered_map<u64, vec<RemotePtr>>& local_updates,
                                            std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                                 VamanaNode::DIM};
  FreshnessEntry old_entry{};
  u32 generation = 0;
  const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
  job.old_ptr = old_entry.current.is_null() ? job.old_ptr_hint : old_entry.current;
  if (old_entry.current.is_null() && job.old_generation_hint >= generation) {
    generation = job.old_generation_hint + 1;
  }
  job.generation = generation;
  if (status != service::storage_owner::MutationStatus::ok) {
    job.status = status;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::erase) {
    job.ok = mark_node_deleted(job.old_ptr, generation);
    job.status = job.ok ? service::storage_owner::MutationStatus::ok
                        : service::storage_owner::MutationStatus::failed;
    if (job.ok) {
      publish_mutation(job.id, job.old_ptr, generation, true);
    }
    co_return;
  }
  const bool low_confidence =
    (job.route_flags & service::storage_owner::kRouteFlagLowConfidence) != 0 ||
    job.route_confidence < static_cast<f32>(config.storage_owner_route_confidence_threshold);
  const bool use_anchors = anchored_update_enabled(config, job.anchor_hints) && !low_confidence;
  if (anchored_update_enabled(config, job.anchor_hints) && low_confidence) {
    ++breakdown.storage_owner_anchor_fallbacks;
  }
  RemotePtr medoid_ptr{};
  bool medoid_loaded = false;
  vec<RemotePtr> owned_candidates;
  const vec<RemotePtr>* candidates = nullptr;
  vec<RemotePtr> audit_exact_candidates;

  if (use_anchors) {
    auto t_search = std::chrono::steady_clock::now();
    auto search = anchor_search_candidates_async(components, job.anchor_hints, config, thread, &breakdown);
    co_await std::suspend_always{};
    while (!search.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        search.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    search.handle.destroy();
    breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
    owned_candidates = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
    candidates = &owned_candidates;

    const u64 sequence = storage_owner_anchor_insert_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool audit = config.storage_owner_anchor_audit_rate != 0 &&
                       sequence % config.storage_owner_anchor_audit_rate == 0;
    const bool insufficient = owned_candidates.size() < config.R;
    if (audit || insufficient) {
      auto t_medoid = std::chrono::steady_clock::now();
      medoid_ptr = co_await async_read_global_medoid(thread);
      medoid_loaded = true;
      breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
      if (!medoid_ptr.is_null()) {
        t_search = std::chrono::steady_clock::now();
        auto exact_search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
        co_await std::suspend_always{};
        while (!exact_search.handle.done()) {
          if (thread.is_ready(thread.running_coroutine)) {
            exact_search.handle.resume();
          } else {
            co_await std::suspend_always{};
          }
        }
        exact_search.handle.destroy();
        breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
        const vec<RemotePtr>& exact = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
        if (audit) {
          ++breakdown.storage_owner_anchor_audits;
        }
        if (insufficient) {
          owned_candidates = exact;
          candidates = &owned_candidates;
          ++breakdown.storage_owner_anchor_fallbacks;
        } else if (audit) {
          audit_exact_candidates = exact;
        }
      }
    }
  }

  if (!use_anchors) {
    auto t_medoid = std::chrono::steady_clock::now();
    medoid_ptr = co_await async_read_global_medoid(thread);
    medoid_loaded = true;
    breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  }
  if (medoid_loaded && medoid_ptr.is_null()) {
    const RemotePtr new_ptr = allocate_local_node();
    job.new_ptr = new_ptr;
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, {}, generation);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    RemotePtr observed;
    if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
      job.ok = true;
      job.status = service::storage_owner::MutationStatus::ok;
      if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
        const RemotePtr previous_ptr = old_entry.current.is_null() ? job.old_ptr_hint : old_entry.current;
        const u32 previous_generation = old_entry.current.is_null()
          ? job.old_generation_hint
          : old_entry.generation;
        if (!previous_ptr.is_null()) {
          mark_node_deleted(previous_ptr, previous_generation);
        }
      }
      publish_mutation(job.id, new_ptr, generation, false);
      co_return;
    }
    medoid_ptr = observed;
  }

  if (!use_anchors) {
    auto t_search = std::chrono::steady_clock::now();
    auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
    co_await std::suspend_always{};
    while (!search.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        search.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    search.handle.destroy();
    breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
    candidates = &storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  }

  lib_assert(candidates != nullptr, "storage-owner insert search produced no candidate set");
  hashset_t<RemotePtr> empty_skip;
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                                                       VectorDType::float32, *candidates, empty_skip, config, &breakdown);
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  if (!audit_exact_candidates.empty()) {
    t_prune = std::chrono::steady_clock::now();
    vec<RemotePtr> exact_selected = robust_prune_cpu(
      reinterpret_cast<const byte_t*>(components.data()), VectorDType::float32,
      audit_exact_candidates, empty_skip, config, &breakdown);
    breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
    if (storage_owner_candidate_overlap(selected_neighbors, exact_selected, config.R) <
        config.storage_owner_anchor_min_overlap) {
      selected_neighbors = std::move(exact_selected);
      ++breakdown.storage_owner_anchor_fallbacks;
      ++breakdown.storage_owner_anchor_audit_failures;
    }
  }
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
  if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
    const RemotePtr previous_ptr = old_entry.current.is_null() ? job.old_ptr_hint : old_entry.current;
    const u32 previous_generation = old_entry.current.is_null()
      ? job.old_generation_hint
      : old_entry.generation;
    if (!previous_ptr.is_null()) {
      mark_node_deleted(previous_ptr, previous_generation);
    }
  }
  publish_mutation(job.id, new_ptr, generation, false);

  for (const RemotePtr& neighbor_ptr : selected_neighbors) {
    if (local_shard(neighbor_ptr.memory_node())) {
      local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
    } else {
      remote_updates[neighbor_ptr.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
    }
  }
  job.ok = true;
  job.status = service::storage_owner::MutationStatus::ok;
}
