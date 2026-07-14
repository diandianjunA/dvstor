#include "gpu_search/delta_index.hh"

#include <algorithm>

namespace gpu_search {
u64 DeltaCoordinator::reserve_epoch() {
  return next_epoch_.fetch_add(1, std::memory_order_relaxed);
}

bool DeltaCoordinator::publish(std::vector<DeltaMutation> mutations, u64 epoch) {
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
    DeltaMutation& stored = delta_[id] = std::move(mutation);
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
  u64 current = published_epoch_.load(std::memory_order_relaxed);
  while (current < epoch &&
         !published_epoch_.compare_exchange_weak(current, epoch,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed)) {}
  return true;
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
