#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "common/types.hh"

namespace memory_node_storage_owner_maintenance_detail {

struct MaintenanceDependencyKey {
  node_t id{};
  std::uint32_t generation{};
  std::uint64_t target_raw{};

  bool operator==(const MaintenanceDependencyKey&) const = default;
};

struct MaintenanceDependencyKeyHash {
  std::size_t operator()(const MaintenanceDependencyKey& key) const {
    std::uint64_t value = key.target_raw ^
      (static_cast<std::uint64_t>(key.id) << 32) ^ key.generation;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94b3a7ce15db9a35ULL;
    value ^= value >> 31;
    return static_cast<std::size_t>(value);
  }
};

// Tracks only dependencies that can affect the same physical incarnation.
// Unrelated cleanup never waits for the shard-wide contiguous durable prefix.
class MaintenanceDependencyTracker {
 public:
  bool register_finalize(std::uint64_t sequence, node_t id,
                         std::uint32_t generation, std::uint64_t target_raw) {
    if (sequence == 0 || target_raw == 0) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const MaintenanceDependencyKey key{id, generation, target_raw};
    if (sequences_.contains(sequence) || keys_.contains(key)) return false;
    sequences_.emplace(sequence, key);
    keys_.emplace(key, sequence);
    return true;
  }

  bool complete_finalize(std::uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = sequences_.find(sequence);
    if (found == sequences_.end()) return false;
    keys_.erase(found->second);
    sequences_.erase(found);
    return true;
  }

  bool cleanup_ready(node_t id, std::uint32_t generation,
                     std::uint64_t target_raw) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !keys_.contains(MaintenanceDependencyKey{id, generation, target_raw});
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sequences_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    sequences_.clear();
    keys_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, MaintenanceDependencyKey> sequences_;
  std::unordered_map<MaintenanceDependencyKey, std::uint64_t,
                     MaintenanceDependencyKeyHash> keys_;
};

}  // namespace memory_node_storage_owner_maintenance_detail
