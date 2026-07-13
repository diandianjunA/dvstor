#include <cassert>

#include "memory_node/storage_reclaim.hh"

int main() {
  memory_node_detail::StorageReclaimQueue queue;
  const RemotePtr first{1, 4096};
  const RemotePtr second{1, 8192};
  const RemotePtr third{1, 12288};
  queue.retire(second, 4);
  queue.retire(first, 2);
  queue.retire(third, 4);
  assert(queue.size() == 3);
  assert(!queue.acquire(4, 1).has_value());

  const auto reclaimed_first = queue.acquire(4, 2);
  assert(reclaimed_first.has_value());
  assert(*reclaimed_first == first);
  assert(queue.size() == 2);

  const auto reclaimed_third = queue.acquire(4, 4);
  const auto reclaimed_second = queue.acquire(4, 4);
  assert(reclaimed_third.has_value());
  assert(reclaimed_second.has_value());
  assert(*reclaimed_third == third);
  assert(*reclaimed_second == second);
  assert(queue.size() == 0);
  assert(queue.reused() == 3);
  assert(!queue.acquire(100, 100).has_value());
  return 0;
}
