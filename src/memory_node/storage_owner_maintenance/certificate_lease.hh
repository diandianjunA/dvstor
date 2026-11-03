#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/types.hh"

namespace memory_node_storage_owner_maintenance_detail {

enum class CertificateAcquireResult : std::uint8_t {
  acquired,
  replay,
  capacity_exhausted,
  conflict,
};

// Target-local ownership of the stable incoming edges that must survive until
// every temporary bridge for the same child has been removed. The table only
// stores protocol state; graph adjacency remains authoritative for the edge.
class CertificateLeaseTable {
 public:
  explicit CertificateLeaseTable(std::size_t degree_limit)
      : degree_limit_(degree_limit) {}

  CertificateAcquireResult acquire(std::uint64_t target_raw,
                                   std::uint64_t child_raw,
                                   std::uint64_t sequence) {
    if (target_raw == 0 || child_raw == 0 || sequence == 0) {
      return CertificateAcquireResult::conflict;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& target = leases_[target_raw];
    const auto found = target.find(child_raw);
    if (found != target.end()) {
      return found->second == sequence ? CertificateAcquireResult::replay
                                       : CertificateAcquireResult::conflict;
    }
    if (target.size() >= degree_limit_) {
      return CertificateAcquireResult::capacity_exhausted;
    }
    target.emplace(child_raw, sequence);
    ++size_;
    return CertificateAcquireResult::acquired;
  }

  bool release(std::uint64_t target_raw, std::uint64_t child_raw,
               std::uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    if (target == leases_.end()) return true;
    const auto found = target->second.find(child_raw);
    if (found == target->second.end()) return true;
    if (sequence != 0 && found->second != sequence) return false;
    target->second.erase(found);
    --size_;
    if (target->second.empty()) leases_.erase(target);
    return true;
  }

  bool commit(
      std::uint64_t target_raw,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& acquired,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& released) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = leases_.find(target_raw);
    std::unordered_map<std::uint64_t, std::uint64_t> updated =
      existing == leases_.end()
        ? std::unordered_map<std::uint64_t, std::uint64_t>{}
        : existing->second;
    for (const auto& [child, sequence] : released) {
      const auto found = updated.find(child);
      if (found != updated.end()) {
        if (sequence != 0 && found->second != sequence) return false;
        updated.erase(found);
      }
    }
    for (const auto& [child, sequence] : acquired) {
      if (child == 0 || sequence == 0) return false;
      const auto found = updated.find(child);
      if (found != updated.end() && found->second != sequence) return false;
      if (found == updated.end() && updated.size() >= degree_limit_) {
        return false;
      }
      updated[child] = sequence;
    }
    const std::size_t previous =
      existing == leases_.end() ? 0 : existing->second.size();
    const std::size_t next = updated.size();
    if (updated.empty()) {
      if (existing != leases_.end()) leases_.erase(existing);
    } else {
      leases_[target_raw] = std::move(updated);
    }
    size_ = size_ - previous + next;
    return true;
  }

  bool contains(std::uint64_t target_raw, std::uint64_t child_raw,
                std::uint64_t sequence = 0) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    if (target == leases_.end()) return false;
    const auto found = target->second.find(child_raw);
    return found != target->second.end() &&
      (sequence == 0 || found->second == sequence);
  }

  std::size_t active_at(std::uint64_t target_raw) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    return target == leases_.end() ? 0 : target->second.size();
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> leases_at(
      std::uint64_t target_raw) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    const auto target = leases_.find(target_raw);
    if (target == leases_.end()) return result;
    result.reserve(target->second.size());
    for (const auto& [child, sequence] : target->second) {
      result.emplace_back(child, sequence);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    leases_.clear();
    size_ = 0;
  }

 private:
  std::size_t degree_limit_;
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t,
                     std::unordered_map<std::uint64_t, std::uint64_t>> leases_;
  std::size_t size_{};
};

}  // namespace memory_node_storage_owner_maintenance_detail
