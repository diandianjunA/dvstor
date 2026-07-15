#include "gpu_search/delta_index.hh"

#include <algorithm>

namespace gpu_search {
u64 DeltaCoordinator::reserve_epoch() {
  return next_epoch_.fetch_add(1, std::memory_order_relaxed);
}

bool DeltaCoordinator::publish(std::vector<DeltaMutation> mutations, u64 epoch) {
  return publish_impl(std::span<DeltaMutation>{mutations}, epoch, true);
}

bool DeltaCoordinator::publish_metadata(
    std::span<DeltaMutation> mutations, u64 epoch) {
  return publish_impl(mutations, epoch, false);
}

bool DeltaCoordinator::publish_impl(std::span<DeltaMutation> mutations,
                                    u64 epoch,
                                    bool retain_vectors) {
  if (mutations.empty() || epoch == 0) return false;
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (DeltaMutation& mutation : mutations) {
    mutation.epoch = epoch;
    const auto current = versions_.find(mutation.id);
    const u32 current_generation = current == versions_.end()
                                     ? 0 : current->second.generation;
    if (mutation.generation == 0) {
      mutation.generation = current_generation + 1;
    } else if (mutation.generation <= current_generation) {
      continue;
    }
    const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
    versions_[mutation.id] = VersionEntry{
      .generation = mutation.generation,
      .epoch = epoch,
      .deleted = deleted,
      .in_delta = true,
    };
    const node_t id = mutation.id;
    DeltaMutation* stored_ptr = nullptr;
    if (retain_vectors) {
      stored_ptr = &(delta_[id] = std::move(mutation));
    } else {
      // The raw vector is required only by the synchronous GPU upload
      // preceding this coordinator handoff. Keeping another owned copy until
      // stage2 retirement would prevent the RPC-slot buffer from being reused.
      DeltaMutation metadata;
      metadata.id = mutation.id;
      metadata.kind = mutation.kind;
      metadata.generation = mutation.generation;
      metadata.epoch = mutation.epoch;
      metadata.remote_node = mutation.remote_node;
      metadata.old_remote_node = mutation.old_remote_node;
      metadata.anchor_hint = mutation.anchor_hint;
      metadata.maintenance_sequence = mutation.maintenance_sequence;
      metadata.owner_storage = mutation.owner_storage;
      metadata.durable = mutation.durable;
      metadata.enqueued_at = mutation.enqueued_at;
      stored_ptr = &(delta_[id] = std::move(metadata));
    }
    DeltaMutation& stored = *stored_ptr;
    if (stored.maintenance_sequence != 0) {
      if (durable_candidates_.size() <= stored.owner_storage) {
        durable_candidates_.resize(static_cast<size_t>(stored.owner_storage) + 1);
      }
      durable_candidates_[stored.owner_storage].push(DurableCandidate{
        .maintenance_sequence = stored.maintenance_sequence,
        .epoch = stored.epoch,
        .id = stored.id,
        .generation = stored.generation,
      });
    }
  }
  publish_barrier(epoch);
  return true;
}

void DeltaCoordinator::publish_barrier(u64 epoch) {
  if (epoch == 0) {
    throw std::invalid_argument("delta publication barrier requires a non-zero epoch");
  }
  u64 current = published_epoch_.load(std::memory_order_relaxed);
  while (current < epoch &&
         !published_epoch_.compare_exchange_weak(current, epoch,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed)) {}
}

u64 DeltaCoordinator::published_epoch() const {
  return published_epoch_.load(std::memory_order_acquire);
}

size_t DeltaCoordinator::delta_size() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return delta_.size();
}

std::optional<VersionEntry> DeltaCoordinator::version(node_t id) const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  const auto it = versions_.find(id);
  return it == versions_.end() ? std::nullopt : std::optional<VersionEntry>{it->second};
}

std::vector<DeltaMutation> DeltaCoordinator::retire_durable(
    std::span<const u64> durable_sequences, size_t max_items) {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  std::vector<DeltaMutation> retired;
  if (max_items == 0 || durable_sequences.empty() ||
      durable_candidates_.empty()) return retired;
  retired.reserve(std::min(max_items, delta_.size()));
  const size_t owner_count = std::min(
    durable_sequences.size(), durable_candidates_.size());
  const size_t first_owner = durable_owner_cursor_ % owner_count;
  for (size_t offset = 0;
       offset < owner_count && retired.size() < max_items; ++offset) {
    const size_t owner = (first_owner + offset) % owner_count;
    DurableQueue& candidates = durable_candidates_[owner];
    const u64 durable_sequence = durable_sequences[owner];
    while (!candidates.empty() && retired.size() < max_items &&
           candidates.top().maintenance_sequence <= durable_sequence) {
      const DurableCandidate candidate = candidates.top();
      candidates.pop();
      const auto mutation_iterator = delta_.find(candidate.id);
      if (mutation_iterator == delta_.end()) continue;
      DeltaMutation& mutation = mutation_iterator->second;
      if (mutation.durable || mutation.owner_storage != owner ||
          mutation.maintenance_sequence != candidate.maintenance_sequence ||
          mutation.epoch != candidate.epoch ||
          mutation.generation != candidate.generation) {
        continue;
      }
      mutation.durable = true;
      const auto version = versions_.find(mutation.id);
      if (version != versions_.end() &&
          version->second.epoch <= mutation.epoch) {
        version->second.in_delta = false;
      }
      retired.push_back(std::move(mutation));
      delta_.erase(mutation_iterator);
    }
  }
  durable_owner_cursor_ = (first_owner + 1) % owner_count;
  return retired;
}

}  // namespace gpu_search
