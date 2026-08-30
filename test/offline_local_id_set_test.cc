#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "tools/vamana_offline/local_id_set.hh"

using tools::vamana_offline::LocalIdSet;

int main() {
  // Production failure shape: the old table had 2,048 slots and spun forever
  // after a difficult search expanded more entries than that.
  LocalIdSet expanded(1024);
  const size_t initial_capacity = expanded.capacity();
  for (uint32_t id = 0; id < 100000; ++id) {
    assert(expanded.insert(id));
  }
  assert(expanded.size() == 100000);
  assert(expanded.capacity() > initial_capacity);
  for (uint32_t id = 0; id < 100000; ++id) {
    assert(expanded.contains(id));
    assert(!expanded.insert(id));
  }
  assert(!expanded.contains(200000));

  LocalIdSet tiny(0);
  for (uint32_t id = 0; id < 10000; ++id) {
    assert(tiny.insert(id * 17U + 3U));
  }
  assert(tiny.size() == 10000);
  for (uint32_t id = 0; id < 10000; ++id) {
    assert(tiny.contains(id * 17U + 3U));
  }

  bool rejected_sentinel = false;
  try {
    (void)tiny.insert(std::numeric_limits<uint32_t>::max());
  } catch (const std::invalid_argument &) {
    rejected_sentinel = true;
  }
  assert(rejected_sentinel);
  assert(!tiny.contains(std::numeric_limits<uint32_t>::max()));

  bool rejected_overflow = false;
  try {
    LocalIdSet impossible(std::numeric_limits<size_t>::max());
    (void)impossible;
  } catch (const std::length_error &) {
    rejected_overflow = true;
  }
  assert(rejected_overflow);

  std::cout << "offline LocalIdSet growth test passed\n";
  return 0;
}
