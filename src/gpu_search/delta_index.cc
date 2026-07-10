#include "gpu_search/delta_index.hh"

#include <algorithm>

namespace gpu_search {
namespace {

size_t mutation_bytes(const DeltaMutation& mutation) {
  return sizeof(DeltaMutation) + mutation.vector.size() +
         mutation.neighbors.size() * sizeof(node_t);
}

}  // namespace

DeltaCoordinator::DeltaCoordinator(u64 base_generation)
    : base_generation_(std::max<u64>(1, base_generation)),
      last_consolidation_(std::chrono::steady_clock::now()) {}

u64 DeltaCoordinator::reserve_epoch() {
  return next_epoch_.fetch_add(1, std::memory_order_relaxed);
}

void DeltaCoordinator::enqueue(DeltaMutation mutation) {
  if (mutation.epoch == 0) mutation.epoch = reserve_epoch();
  if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) {
    mutation.enqueued_at = std::chrono::steady_clock::now();
  }
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_.push_back(std::move(mutation));
  }
  pending_cv_.notify_one();
}

std::vector<DeltaMutation> DeltaCoordinator::take_pending(
    size_t max_items, std::chrono::microseconds max_wait) {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  if (pending_.empty() && max_wait.count() > 0) {
    pending_cv_.wait_for(lock, max_wait, [&] { return !pending_.empty(); });
  }
  const size_t count = std::min(max_items, pending_.size());
  std::vector<DeltaMutation> batch;
  batch.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    batch.push_back(std::move(pending_.front()));
    pending_.pop_front();
  }
  return batch;
}

bool DeltaCoordinator::publish(std::vector<DeltaMutation> mutations, u64 epoch,
                               std::chrono::steady_clock::time_point now) {
  if (mutations.empty() || epoch == 0) return false;
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (DeltaMutation& mutation : mutations) {
    mutation.epoch = epoch;
    mutation.published_at = now;
    auto existing = delta_.find(mutation.id);
    if (existing != delta_.end()) delta_bytes_ -= mutation_bytes(existing->second);
    mutation.generation = std::max<u32>(
      mutation.generation,
      versions_.contains(mutation.id) ? versions_[mutation.id].generation + 1 : 1);
    const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
    versions_[mutation.id] = VersionEntry{
      .generation = mutation.generation,
      .epoch = epoch,
      .deleted = deleted,
      .in_delta = true,
    };
    delta_bytes_ += mutation_bytes(mutation);
    delta_[mutation.id] = std::move(mutation);
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

u64 DeltaCoordinator::base_generation() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return base_generation_;
}

size_t DeltaCoordinator::delta_size() const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  return delta_.size();
}

size_t DeltaCoordinator::pending_size() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  return pending_.size();
}

std::optional<VersionEntry> DeltaCoordinator::version(node_t id) const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  const auto it = versions_.find(id);
  return it == versions_.end() ? std::nullopt : std::optional<VersionEntry>{it->second};
}

DeltaSnapshot DeltaCoordinator::snapshot(u64 epoch) const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  DeltaSnapshot result;
  result.epoch = epoch == 0 ? published_epoch_.load(std::memory_order_acquire) : epoch;
  result.base_generation = base_generation_;
  result.mutations.reserve(delta_.size());
  for (const auto& [id, mutation] : delta_) {
    (void)id;
    if (mutation.epoch <= result.epoch) result.mutations.push_back(mutation);
  }
  std::sort(result.mutations.begin(), result.mutations.end(),
            [](const DeltaMutation& lhs, const DeltaMutation& rhs) {
              return lhs.id < rhs.id;
            });
  return result;
}

bool DeltaCoordinator::should_consolidate(u64 base_nodes, size_t delta_budget_bytes,
                                          f64 max_ratio, f64 budget_high_watermark,
                                          std::chrono::milliseconds max_age) const {
  std::shared_lock<std::shared_mutex> lock(state_mutex_);
  const bool ratio_reached = base_nodes > 0 && max_ratio > 0.0 &&
    static_cast<f64>(delta_.size()) >= static_cast<f64>(base_nodes) * max_ratio;
  const bool budget_reached = delta_budget_bytes > 0 && budget_high_watermark > 0.0 &&
    static_cast<f64>(delta_bytes_) >=
      static_cast<f64>(delta_budget_bytes) * budget_high_watermark;
  const bool age_reached = max_age.count() > 0 && !delta_.empty() &&
    std::chrono::steady_clock::now() - last_consolidation_ >= max_age;
  return ratio_reached || budget_reached || age_reached;
}

DeltaSnapshot DeltaCoordinator::begin_consolidation() {
  return snapshot();
}

void DeltaCoordinator::complete_consolidation(u64 new_base_generation, u64 through_epoch) {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  if (new_base_generation <= base_generation_) return;
  for (auto it = delta_.begin(); it != delta_.end();) {
    if (it->second.epoch <= through_epoch) {
      delta_bytes_ -= mutation_bytes(it->second);
      auto version_it = versions_.find(it->first);
      if (version_it != versions_.end() && version_it->second.epoch <= through_epoch) {
        version_it->second.in_delta = false;
      }
      it = delta_.erase(it);
    } else {
      ++it;
    }
  }
  base_generation_ = new_base_generation;
  last_consolidation_ = std::chrono::steady_clock::now();
}

void DeltaCoordinator::complete_partial_consolidation(
    const std::vector<node_t>& merged_ids,
    u64 new_base_generation,
    u64 through_epoch) {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (const node_t id : merged_ids) {
    const auto it = delta_.find(id);
    if (it == delta_.end() || it->second.epoch > through_epoch) continue;
    delta_bytes_ -= mutation_bytes(it->second);
    const auto version_it = versions_.find(id);
    if (version_it != versions_.end() && version_it->second.epoch <= through_epoch) {
      version_it->second.in_delta = false;
    }
    delta_.erase(it);
  }
  base_generation_ = std::max(base_generation_, new_base_generation);
  last_consolidation_ = std::chrono::steady_clock::now();
}

void DeltaCoordinator::mark_compacted() {
  std::unique_lock<std::shared_mutex> lock(state_mutex_);
  last_consolidation_ = std::chrono::steady_clock::now();
}

}  // namespace gpu_search
