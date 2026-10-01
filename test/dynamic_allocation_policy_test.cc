#include <cassert>
#include <limits>

#include "memory_node/storage_owner_index/dynamic_allocation_policy.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

void test_zero_limit_resolves_to_region_bytes() {
  assert(detail::resolve_dynamic_allocation_limit(0, 4096) == 4096);
  assert(detail::resolve_dynamic_allocation_limit(
           0, std::numeric_limits<u64>::max()) ==
         std::numeric_limits<u64>::max());
  assert(detail::resolve_dynamic_allocation_limit(1024, 4096) == 1024);
}

void test_slot_fit_boundary() {
  constexpr u64 kNodeSize = 256;
  constexpr u64 kLimit = 1024;
  // The last byte of the allocation may touch the limit exactly.
  assert(detail::dynamic_slot_allocation_fits(768, kNodeSize, kLimit));
  assert(detail::dynamic_slot_allocation_fits(0, kNodeSize, kLimit));
  // One slot past the boundary and the OOM path must report "does not fit"
  // so the caller can return a null RemotePtr instead of terminating.
  assert(!detail::dynamic_slot_allocation_fits(769, kNodeSize, kLimit));
  assert(!detail::dynamic_slot_allocation_fits(1024, kNodeSize, kLimit));
  assert(!detail::dynamic_slot_allocation_fits(1025, kNodeSize, kLimit));
}

void test_fit_check_cannot_overflow() {
  constexpr u64 kNodeSize = 256;
  // An exhausted bump allocator keeps handing out ever-larger offsets. The
  // subtraction form must stay exact when offset approaches u64 max.
  const u64 limit = std::numeric_limits<u64>::max() - 100;
  assert(!detail::dynamic_slot_allocation_fits(
    std::numeric_limits<u64>::max(), kNodeSize, limit));
  assert(!detail::dynamic_slot_allocation_fits(
    std::numeric_limits<u64>::max() - 50, kNodeSize, limit));
  assert(detail::dynamic_slot_allocation_fits(
    std::numeric_limits<u64>::max() - 256, kNodeSize,
    std::numeric_limits<u64>::max()));
}

}  // namespace

int main() {
  test_zero_limit_resolves_to_region_bytes();
  test_slot_fit_boundary();
  test_fit_check_cannot_overflow();
  return 0;
}
