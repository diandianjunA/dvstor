#include <cassert>

#include "memory_node/storage_reclaim.hh"

int main() {
  memory_node_detail::StorageReclaimQueue queue;
  const RemotePtr first{1, 4096};
  const RemotePtr reincarnated_first{1, 4096, 7};
  const RemotePtr second{1, 8192};
  const RemotePtr third{1, 12288};
  assert(queue.retire(second, 4));
  assert(queue.retire(first, 2));
  assert(queue.retire(third, 4));
  assert(!queue.retire(first, 9));
  // A delayed cleanup may carry a different logical incarnation of the same
  // slot. Reclamation is keyed by physical storage, so it must not enqueue
  // that address twice.
  assert(!queue.retire(reincarnated_first, 9));
  assert(queue.size() == 3);
  assert(!queue.acquire(1).has_value());

  const auto reclaimed_first = queue.acquire(2);
  assert(reclaimed_first.has_value());
  assert(*reclaimed_first == first);
  assert(queue.size() == 2);

  const auto reclaimed_third = queue.acquire(4);
  const auto reclaimed_second = queue.acquire(4);
  assert(reclaimed_third.has_value());
  assert(reclaimed_second.has_value());
  assert(*reclaimed_third == third);
  assert(*reclaimed_second == second);
  assert(queue.size() == 0);
  assert(queue.reused() == 3);
  assert(!queue.acquire(100).has_value());
  return 0;
}
