#pragma once

#include "common/types.hh"

namespace memory_node_storage_owner_index_detail {

// Zero means "no explicit bound": the dynamic region may grow to the end of
// the registered shard region.
[[nodiscard]] constexpr u64 resolve_dynamic_allocation_limit(
    u64 configured_limit, u64 region_bytes) noexcept {
  return configured_limit == 0 ? region_bytes : configured_limit;
}

// Overflow-safe half-open bound check: [offset, offset + node_size) must fit
// within [0, allocation_limit). A full region reports "does not fit" so the
// caller can propagate a retryable allocation failure instead of faulting.
[[nodiscard]] constexpr bool dynamic_slot_allocation_fits(
    u64 offset, u64 node_size, u64 allocation_limit) noexcept {
  return offset <= allocation_limit &&
         node_size <= allocation_limit - offset;
}

}  // namespace memory_node_storage_owner_index_detail
