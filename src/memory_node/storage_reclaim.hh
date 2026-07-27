#pragma once

#include <cstddef>
#include <iterator>
#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_detail {

class StorageReclaimQueue {
public:
  bool retire(RemotePtr pointer, u64 maintenance_sequence) {
    if (pointer.is_null() || maintenance_sequence == 0) return false;
    // Cleanup activation and transport retries are deliberately idempotent.
    // Keep exactly one retirement record per physical slot so a duplicate
    // completion can never put the same address in the ready list twice.
    if (!retired_.insert(pointer.physical_address_raw()).second) return false;
    pending_[maintenance_sequence].push_back(pointer);
    ++size_;
    return true;
  }

  std::optional<RemotePtr> acquire(u64 durable_sequence) {
    while (!pending_.empty() && pending_.begin()->first <= durable_sequence) {
      auto nodes = std::move(pending_.begin()->second);
      pending_.erase(pending_.begin());
      ready_.insert(ready_.end(),
                    std::make_move_iterator(nodes.begin()),
                    std::make_move_iterator(nodes.end()));
    }
    if (ready_.empty()) return std::nullopt;
    const RemotePtr pointer = ready_.back();
    ready_.pop_back();
    const size_t erased = retired_.erase(pointer.physical_address_raw());
    (void)erased;
    --size_;
    ++reused_;
    return pointer;
  }

  size_t size() const { return size_; }
  size_t ready_size() const { return ready_.size(); }
  u64 reused() const { return reused_; }

private:
  std::map<u64, std::vector<RemotePtr>> pending_;
  std::vector<RemotePtr> ready_;
  std::unordered_set<u64> retired_;
  size_t size_{};
  u64 reused_{};
};

}  // namespace memory_node_detail
