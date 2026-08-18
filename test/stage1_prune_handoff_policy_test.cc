#include <cassert>
#include <vector>

#include "memory_node/storage_owner_index/stage1_prune_handoff_policy.hh"

using memory_node_storage_owner_index_detail::
  deferred_stage1_provisional_neighbors;
using memory_node_storage_owner_index_detail::stage2_observed_reverse_delta;

namespace {

RemotePtr pointer(u64 value) {
  return RemotePtr{value};
}

}  // namespace

int main() {
  const vec<RemotePtr> ordered{
    pointer(11), pointer(22), pointer(33), pointer(44)};
  assert((deferred_stage1_provisional_neighbors(ordered, 0) ==
          vec<RemotePtr>{}));
  assert((deferred_stage1_provisional_neighbors(ordered, 2) ==
          vec<RemotePtr>{pointer(11), pointer(22)}));
  assert(deferred_stage1_provisional_neighbors(ordered, 8) == ordered);

  const vec<RemotePtr> observed{
    pointer(11), pointer(55), pointer(22), pointer(66)};
  assert((stage2_observed_reverse_delta(observed, ordered) ==
          vec<RemotePtr>{pointer(55), pointer(66)}));
  assert(stage2_observed_reverse_delta(observed, {}).size() ==
         observed.size());
  return 0;
}
