#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "common/bounded_queue.hh"
#include "common/completion_pool.hh"
#include "common/sliding_completion_ring.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"

namespace {

void test_queue_wrap_and_capacity() {
  bounded::Queue<unsigned> queue(3);
  assert(queue.capacity() == 4);
  assert(queue.try_push(1));
  assert(queue.try_push(2));
  assert(queue.try_push(3));
  assert(queue.try_push(4));
  assert(!queue.try_push(5));

  unsigned value = 0;
  assert(queue.try_pop(value) && value == 1);
  assert(queue.try_pop(value) && value == 2);
  assert(queue.try_push(5));
  assert(queue.try_push(6));
  for (unsigned expected = 3; expected <= 6; ++expected) {
    assert(queue.try_pop(value) && value == expected);
  }
  assert(!queue.try_pop(value));
}

void test_queue_multiple_producers() {
  constexpr unsigned kProducers = 4;
  constexpr unsigned kItems = 10'000;
  bounded::Queue<unsigned> queue(256);
  std::array<std::thread, kProducers> producers;
  for (unsigned producer = 0; producer < kProducers; ++producer) {
    producers[producer] = std::thread([&, producer]() {
      for (unsigned item = 0; item < kItems; ++item) {
        queue.push_wait(producer * kItems + item);
      }
    });
  }

  std::vector<unsigned> received;
  received.reserve(kProducers * kItems);
  for (unsigned item = 0; item < kProducers * kItems; ++item) {
    unsigned value = 0;
    queue.pop_wait(value);
    received.push_back(value);
  }
  for (auto& producer : producers) producer.join();
  std::sort(received.begin(), received.end());
  for (unsigned item = 0; item < received.size(); ++item) {
    assert(received[item] == item);
  }
}

void test_completion_pool_reuse_and_abandon() {
  bounded::CompletionPool pool(2);
  const u32 first = pool.acquire();
  const u32 second = pool.acquire();
  u32 unavailable = 0;
  assert(!pool.try_acquire(unavailable));

  std::thread producer([&]() { pool.complete(first, true); });
  assert(pool.wait(first) == bounded::CompletionPool::Result::success);
  pool.release_consumer(first);
  producer.join();

  const u32 reused = pool.acquire();
  assert(reused == first);
  pool.release_consumer(reused);  // A timed-out/abandoned public caller.
  assert(!pool.try_acquire(unavailable));
  pool.complete(reused, false);
  assert(pool.try_acquire(unavailable));
  pool.complete(unavailable, true);
  pool.release_consumer(unavailable);

  pool.complete(second, false);
  assert(pool.wait(second) == bounded::CompletionPool::Result::failure);
  pool.release_consumer(second);
}

void test_sliding_completion_ring() {
  bounded::SlidingCompletionRing ring(3);
  const u64 one = ring.reserve(1);
  const u64 two = ring.reserve(2);
  const u64 three = ring.reserve(1);
  assert(one == 1 && two == 2 && three == 3);
  assert(ring.outstanding() == 3);

  ring.complete(two);
  ring.complete(three);
  assert(ring.finalized() == 0);
  ring.complete(one);
  assert(ring.finalized() == 1);
  ring.complete(two);
  assert(ring.finalized() == 3);

  const u64 four = ring.reserve(0);
  assert(four == 4);
  assert(ring.finalized() == 4);

  const u64 five = ring.reserve(1);
  const u64 six = ring.reserve(1);
  const u64 seven = ring.reserve(1);
  std::atomic<bool> reserve_started{false};
  auto blocked = std::async(std::launch::async, [&]() {
    reserve_started.store(true, std::memory_order_release);
    return ring.reserve(1);
  });
  while (!reserve_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  assert(blocked.wait_for(std::chrono::milliseconds(10)) ==
         std::future_status::timeout);
  ring.complete(six);
  ring.complete(seven);
  assert(ring.finalized() == four);
  ring.complete(five);
  assert(blocked.wait_for(std::chrono::seconds(1)) ==
         std::future_status::ready);
  assert(blocked.get() == 8);
  ring.complete(8);
  assert(ring.finalized() == 8);
}

void test_sliding_completion_ring_atomic_batch_admission() {
  bounded::SlidingCompletionRing ring(4);
  const std::array<u32, 3> first_work{1, 1, 1};
  assert(ring.reserve_batch(
           span<const u32>{first_work.data(), first_work.size()}) == 1);
  assert(ring.next_sequence() == 4);

  std::atomic<bool> reserve_started{false};
  const std::array<u32, 2> second_work{1, 1};
  auto blocked = std::async(std::launch::async, [&]() {
    reserve_started.store(true, std::memory_order_release);
    return ring.reserve_batch(
      span<const u32>{second_work.data(), second_work.size()});
  });
  while (!reserve_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  assert(blocked.wait_for(std::chrono::milliseconds(10)) ==
         std::future_status::timeout);
  // The blocked producer must not retain a partial reservation.
  assert(ring.next_sequence() == 4);

  ring.complete(2);
  ring.complete(3);
  assert(ring.finalized() == 0);
  assert(ring.next_sequence() == 4);
  ring.complete(1);
  assert(blocked.wait_for(std::chrono::seconds(1)) ==
         std::future_status::ready);
  assert(blocked.get() == 4);
  assert(ring.next_sequence() == 6);
  ring.complete(5);
  assert(ring.finalized() == 3);
  ring.complete(4);
  assert(ring.finalized() == 5);

  const std::array<u32, 2> zero_work{0, 0};
  assert(ring.reserve_batch(
           span<const u32>{zero_work.data(), zero_work.size()}) == 6);
  assert(ring.finalized() == 7);
}

void test_stale_stitch_sequence_handoff_to_bounded_repair() {
  struct RepairDescriptor {
    u64 sequence{};
  };

  bounded::SlidingCompletionRing ring(4);
  bounded::Queue<RepairDescriptor> repairs(2);

  // A stale insert transfers its only completion unit to repair. Merely
  // discovering staleness must not advance the finalized watermark.
  const u64 insert_sequence = ring.reserve(1);
  assert(repairs.try_push({insert_sequence}));
  assert(ring.finalized() == 0);

  // An upsert owns stitch+cleanup units. Its ordinary cleanup may finish out
  // of order, but the sequence remains blocked while stale stitch repair is
  // pending and while the prior insert sequence has not finalized.
  const u64 upsert_sequence = ring.reserve(2);
  ring.complete(upsert_sequence);
  assert(ring.finalized() == 0);
  assert(repairs.try_push({upsert_sequence}));
  assert(!repairs.try_push({upsert_sequence + 1}));

  // A not-yet-ready repair can be popped and returned without loss or heap
  // growth. Production uses finalized >= sequence-1 as the readiness gate.
  RepairDescriptor repair;
  assert(repairs.try_pop(repair));
  assert(repair.sequence == insert_sequence);
  assert(repairs.try_push(std::move(repair)));
  assert(ring.finalized() < upsert_sequence - 1);

  assert(repairs.try_pop(repair));
  assert(repair.sequence == upsert_sequence);
  assert(repairs.try_push(std::move(repair)));

  // Queue order is not semantically significant: completing the later repair
  // first still cannot jump the earlier sequence.
  assert(repairs.try_pop(repair));
  assert(repair.sequence == insert_sequence);
  ring.complete(insert_sequence);
  assert(ring.finalized() == insert_sequence);
  assert(repairs.try_pop(repair));
  assert(repair.sequence == upsert_sequence);
  ring.complete(upsert_sequence);
  assert(ring.finalized() == upsert_sequence);
  assert(repairs.empty());
}

void test_stale_stitch_repair_keeps_schema15_payload_bound() {
  constexpr size_t kR = 4;
  const vec<RemotePtr> preserved{
    RemotePtr{1, 0x1000}, RemotePtr{1, 0x2000},
    RemotePtr{1, 0x3000}, RemotePtr{1, 0x4000}};
  const vec<RemotePtr> supplemental{
    RemotePtr{2, 0x1000}, RemotePtr{2, 0x2000},
    RemotePtr{2, 0x3000}, RemotePtr{2, 0x4000}};

  // The stale stitch transfers its completion unit to a repair descriptor.
  // Even with disjoint R-sized preserved and supplemental sets, that repair
  // sends only the supplemental backlinks attempted by the stitch.
  bounded::SlidingCompletionRing ring(2);
  const u64 stale_stitch_sequence = ring.reserve(1);
  const u64 successor_cleanup_sequence = ring.reserve(1);
  const vec<RemotePtr> repair =
    memory_node_storage_owner_maintenance_detail::select_cleanup_neighbors(
      true,
      span<const RemotePtr>{preserved.data(), preserved.size()},
      span<const RemotePtr>{supplemental.data(), supplemental.size()});
  assert(repair == supplemental);
  assert(repair.size() == kR);
  assert(ring.finalized() == 0);

  // Repair completion releases the stale sequence. The later ordinary
  // delete/upsert cleanup independently consumes the preserved adjacency.
  ring.complete(stale_stitch_sequence);
  assert(ring.finalized() == stale_stitch_sequence);
  const vec<RemotePtr> ordinary =
    memory_node_storage_owner_maintenance_detail::select_cleanup_neighbors(
      false,
      span<const RemotePtr>{preserved.data(), preserved.size()},
      span<const RemotePtr>{});
  assert(ordinary == preserved);
  ring.complete(successor_cleanup_sequence);
  assert(ring.finalized() == successor_cleanup_sequence);
}

void test_stage2_admission_yields_only_for_live_foreground_pressure() {
  using memory_node_storage_owner_maintenance_detail::
    Stage2AdmissionDecision;
  using memory_node_storage_owner_maintenance_detail::
    decide_stage2_admission;

  bool pressure_probe_called = false;
  const auto pressure_probe = [&]() {
    pressure_probe_called = true;
    return true;
  };

  assert(decide_stage2_admission(true, false, pressure_probe) ==
         Stage2AdmissionDecision::unavailable);
  assert(!pressure_probe_called);
  assert(decide_stage2_admission(false, true, pressure_probe) ==
         Stage2AdmissionDecision::unavailable);
  assert(!pressure_probe_called);
  assert(decide_stage2_admission(false, false, pressure_probe) ==
         Stage2AdmissionDecision::foreground_pressure);
  assert(pressure_probe_called);

  pressure_probe_called = false;
  assert(decide_stage2_admission(false, false, [&]() {
           pressure_probe_called = true;
           return false;
         }) == Stage2AdmissionDecision::admit);
  assert(pressure_probe_called);
}

void test_reverse_candidate_is_revalidated_at_locked_write_boundary() {
  const RemotePtr candidate{1, 4096};
  const vec<RemotePtr> current_neighbors;
  const vec<RemotePtr> candidates{candidate};

  // This models the unsafe pre-lock cache seeing a live source, followed by
  // delete cleanup completing before the reverse target lock is acquired.
  const bool prelock_cached_live = true;
  bool live_at_locked_boundary = false;
  vec<RemotePtr> selected;
  memory_node_storage_owner_index_detail::
    select_fresh_reverse_candidates_locked(
      current_neighbors, candidates,
      [&](const RemotePtr& observed) {
        assert(observed == candidate);
        return live_at_locked_boundary;
      },
      selected);
  assert(prelock_cached_live);
  assert(selected.empty());

  // If deletion starts after this check, production still holds the target
  // lock, so cleanup must run after the backlink write and remove it.
  live_at_locked_boundary = true;
  memory_node_storage_owner_index_detail::
    select_fresh_reverse_candidates_locked(
      current_neighbors, candidates,
      [&](const RemotePtr&) { return live_at_locked_boundary; }, selected);
  assert(selected.size() == 1 && selected.front() == candidate);
}

}  // namespace

int main() {
  test_queue_wrap_and_capacity();
  test_queue_multiple_producers();
  test_completion_pool_reuse_and_abandon();
  test_sliding_completion_ring();
  test_sliding_completion_ring_atomic_batch_admission();
  test_stale_stitch_sequence_handoff_to_bounded_repair();
  test_stale_stitch_repair_keeps_schema15_payload_bound();
  test_stage2_admission_yields_only_for_live_foreground_pressure();
  test_reverse_candidate_is_revalidated_at_locked_write_boundary();
  return 0;
}
