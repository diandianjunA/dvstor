#include <atomic>
#include <cassert>
#include <thread>

#include "memory_node/storage_owner_index/incarnation_lock.hh"

namespace {

using memory_node_storage_owner_index_detail::IncarnationLockResult;
using memory_node_storage_owner_index_detail::try_lock_header_once;

void unlock(u64& header) {
  std::atomic_ref<u64>(header).fetch_and(
    ~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK),
    std::memory_order_release);
}

void test_current_incarnation_locks_and_contention_is_distinct() {
  alignas(u64) u64 header = VamanaNode::make_header(7);
  assert(try_lock_header_once(header, 7) ==
         IncarnationLockResult::locked);
  assert((std::atomic_ref<u64>(header).load(std::memory_order_acquire) &
          VamanaNode::HEADER_NODE_LOCK) != 0);
  assert(try_lock_header_once(header, 7) ==
         IncarnationLockResult::busy);
  unlock(header);
  assert(try_lock_header_once(header, 7) ==
         IncarnationLockResult::locked);
  unlock(header);
}

void test_check_then_reuse_cannot_lock_or_modify_new_incarnation() {
  constexpr u32 old_incarnation = 11;
  constexpr u32 new_incarnation = 12;
  alignas(u64) u64 header = VamanaNode::make_header(
    old_incarnation, VamanaNode::HEADER_PROVISIONAL);
  std::atomic<bool> old_identity_checked{false};
  std::atomic<bool> slot_reused{false};
  IncarnationLockResult delayed_result = IncarnationLockResult::busy;

  std::thread delayed_reader([&] {
    const u64 observed = std::atomic_ref<u64>(header).load(
      std::memory_order_acquire);
    assert(VamanaNode::header_incarnation(observed) == old_incarnation);
    old_identity_checked.store(true, std::memory_order_release);
    while (!slot_reused.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    delayed_result = try_lock_header_once(header, old_incarnation);
  });

  while (!old_identity_checked.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  const u64 replacement = VamanaNode::make_header(
    new_incarnation, VamanaNode::HEADER_CENTROID_ACCOUNTED);
  std::atomic_ref<u64>(header).store(replacement, std::memory_order_release);
  slot_reused.store(true, std::memory_order_release);
  delayed_reader.join();

  assert(delayed_result == IncarnationLockResult::stale);
  assert(std::atomic_ref<u64>(header).load(std::memory_order_acquire) ==
         replacement);
  assert(try_lock_header_once(header, new_incarnation) ==
         IncarnationLockResult::locked);
  unlock(header);
}

void test_stale_attempt_does_not_unlock_a_new_occupant() {
  alignas(u64) u64 header = VamanaNode::make_header(22);
  assert(try_lock_header_once(header, 22) ==
         IncarnationLockResult::locked);
  const u64 replacement_locked =
    std::atomic_ref<u64>(header).load(std::memory_order_acquire);

  assert(try_lock_header_once(header, 21) ==
         IncarnationLockResult::stale);
  assert(std::atomic_ref<u64>(header).load(std::memory_order_acquire) ==
         replacement_locked);
  unlock(header);
}

}  // namespace

int main() {
  test_current_incarnation_locks_and_contention_is_distinct();
  test_check_then_reuse_cannot_lock_or_modify_new_incarnation();
  test_stale_attempt_does_not_unlock_a_new_occupant();
  return 0;
}
