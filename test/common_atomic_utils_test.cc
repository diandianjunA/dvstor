#include <atomic>
#include <cassert>

#include "common/atomic_utils.hh"

int main() {
  std::atomic<unsigned> maximum{3};
  atomic_utils::update_max_relaxed(maximum, 2u);
  assert(maximum.load(std::memory_order_relaxed) == 3);
  atomic_utils::update_max_relaxed(maximum, 5u);
  assert(maximum.load(std::memory_order_relaxed) == 5);

  std::atomic<unsigned> active{2};
  {
    atomic_utils::CounterDecrementGuard guard(active);
    assert(active.load(std::memory_order_relaxed) == 2);
  }
  assert(active.load(std::memory_order_relaxed) == 1);
  return 0;
}
