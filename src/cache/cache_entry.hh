#pragma once

#include "remote_pointer.hh"
#include "tagged_ptr.hh"
#include "vamana/vamana_node.hh"

namespace cache {

struct CacheEntry {
  RemotePtr key{};
  std::atomic<s_ptr<VamanaNode>> value{nullptr};

  tagged_ptr<CacheEntry> next{};
  std::atomic<u16> tag{};
  std::atomic<bool> cooling{};

  // occurs under lock
  void update_value(const s_ptr<VamanaNode>& val) { value = val; }

  // occurs under lock
  u16 evict() {
    const u16 old_tag = tag.fetch_add(1, std::memory_order_release);

    key.reset();
    next.invalidate();

    cooling = false;
    value.store(nullptr, std::memory_order_release);  // force deallocation

    return old_tag + 1;  // return new tag
  }
};

}  // namespace cache
