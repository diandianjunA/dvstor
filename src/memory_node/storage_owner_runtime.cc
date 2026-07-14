#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

bool storage_owner_local_stitch_mode(const configuration::IndexConfiguration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

bool storage_owner_batch_prefers_sync_local_stitch(
    const configuration::IndexConfiguration& config,
    const vec<service::storage_owner::MutationKind>& kinds,
    const vec<u64>& anchor_hints,
    u32 anchor_hint_count,
    size_t item_count) {
  if (!storage_owner_local_stitch_mode(config) ||
      !config.storage_owner_local_stitch_sync_fast_path ||
      anchor_hint_count == 0) {
    return false;
  }
  if (anchor_hints.size() < item_count * static_cast<size_t>(anchor_hint_count)) {
    return false;
  }
  for (size_t item = 0; item < item_count; ++item) {
    if (item < kinds.size() &&
        kinds[item] == service::storage_owner::MutationKind::erase) {
      continue;
    }
    bool has_hint = false;
    const size_t base = item * static_cast<size_t>(anchor_hint_count);
    for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
      if (!RemotePtr{anchor_hints[base + hint]}.is_null()) {
        has_hint = true;
        break;
      }
    }
    if (!has_hint) {
      return false;
    }
  }
  return true;
}

}  // namespace

#include "memory_node/storage_owner_runtime/lifecycle.ipp"
#include "memory_node/storage_owner_runtime/workers.ipp"
#include "memory_node/storage_owner_runtime/batch_execution.ipp"
#include "memory_node/storage_owner_runtime/wire_protocol.ipp"
