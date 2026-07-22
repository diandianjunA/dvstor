#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#include "common/bounded_queue.hh"
#include "common/completion_pool.hh"
#include "common/sliding_completion_ring.hh"
#include "common/vector_dtype.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"
#include "memory_node/storage_owner_index/robust_prune_policy.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"

namespace {

struct PausedQueueValue {
  unsigned value{};
  std::atomic<bool>* assignment_entered{};
  std::atomic<bool>* assignment_release{};
  bool pause_assignment{};

  PausedQueueValue() = default;
  PausedQueueValue(
      unsigned value_in,
      std::atomic<bool>* entered,
      std::atomic<bool>* release,
      bool pause)
      : value(value_in),
        assignment_entered(entered),
        assignment_release(release),
        pause_assignment(pause) {}

  PausedQueueValue(const PausedQueueValue&) = default;
  PausedQueueValue(PausedQueueValue&&) = default;
  PausedQueueValue& operator=(const PausedQueueValue&) = default;

  PausedQueueValue& operator=(PausedQueueValue&& other) noexcept {
    value = other.value;
    assignment_entered = other.assignment_entered;
    assignment_release = other.assignment_release;
    const bool pause = other.pause_assignment;
    // A cell blocks only while the producer is publishing into it. Moving the
    // already-published cell into a consumer must not block a second time.
    pause_assignment = false;
    other.pause_assignment = false;
    if (pause) {
      assignment_entered->store(true, std::memory_order_release);
      assignment_entered->notify_all();
      while (!assignment_release->load(std::memory_order_acquire)) {
        assignment_release->wait(false, std::memory_order_relaxed);
      }
    }
    return *this;
  }
};

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

void test_queue_later_publication_can_be_blocked_by_reserved_head() {
  bounded::Queue<PausedQueueValue> queue(4);
  std::atomic<bool> first_assignment_entered{false};
  std::atomic<bool> release_first_assignment{false};

  std::thread first([&]() {
    assert(queue.try_push(PausedQueueValue{
      1, &first_assignment_entered, &release_first_assignment, true}));
  });
  while (!first_assignment_entered.load(std::memory_order_acquire)) {
    first_assignment_entered.wait(false, std::memory_order_relaxed);
  }

  // Producer one has reserved FIFO position zero but has not published its
  // sequence. Producer two can reserve and fully publish position one.
  assert(queue.try_push(PausedQueueValue{2, nullptr, nullptr, false}));
  PausedQueueValue value;
  assert(!queue.try_pop(value));

  release_first_assignment.store(true, std::memory_order_release);
  release_first_assignment.notify_all();
  first.join();
  assert(queue.try_pop(value) && value.value == 1);
  assert(queue.try_pop(value) && value.value == 2);
  assert(!queue.try_pop(value));
}

void test_queue_stopped_producer_does_not_overwrite_full_queue() {
  bounded::Queue<unsigned> queue(2);
  assert(queue.try_push(1));
  assert(queue.try_push(2));

  std::atomic<bool> stop{false};
  auto producer = std::async(std::launch::async, [&]() {
    return queue.push_wait(3, stop);
  });
  assert(producer.wait_for(std::chrono::milliseconds(10)) ==
         std::future_status::timeout);
  stop.store(true, std::memory_order_release);
  queue.notify_all();
  assert(producer.get() == false);

  unsigned value = 0;
  assert(queue.try_pop(value) && value == 1);
  assert(queue.try_pop(value) && value == 2);
  assert(!queue.try_pop(value));

  assert(queue.push_wait(4, stop) == false);
  assert(!queue.try_pop(value));
}

void test_queue_publication_precedes_slot_reuse() {
  constexpr unsigned kSlots = 8;
  constexpr unsigned kThreads = 8;
  constexpr unsigned kIterations = 20'000;
  bounded::Queue<unsigned> free_slots(kSlots);
  std::array<std::atomic<unsigned>, kSlots> phases{};
  for (unsigned slot = 0; slot < kSlots; ++slot) {
    phases[slot].store(0, std::memory_order_relaxed);
    assert(free_slots.try_push(slot));
  }

  std::atomic<bool> failed{false};
  std::array<std::thread, kThreads> workers;
  for (auto& worker : workers) {
    worker = std::thread([&]() {
      for (unsigned iteration = 0; iteration < kIterations; ++iteration) {
        unsigned slot = 0;
        free_slots.pop_wait(slot);
        unsigned expected = 0;
        if (!phases[slot].compare_exchange_strong(
              expected, 1, std::memory_order_acquire,
              std::memory_order_relaxed)) {
          failed.store(true, std::memory_order_relaxed);
        }
        // This is the query-slot release order: publish free state first,
        // then make its index visible to another queue consumer.
        phases[slot].store(0, std::memory_order_release);
        free_slots.push_wait(slot);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  assert(!failed.load(std::memory_order_relaxed));
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

void test_completion_pool_timed_wait_preserves_producer_reference() {
  bounded::CompletionPool pool(1);
  const u32 id = pool.acquire();
  assert(pool.wait_for(id, std::chrono::milliseconds(1)) ==
         bounded::CompletionPool::Result::pending);
  pool.release_consumer(id);
  pool.complete(id, true);

  const u32 reused = pool.acquire();
  assert(reused == id);
  pool.complete(reused, false);
  assert(pool.wait(reused) == bounded::CompletionPool::Result::failure);
  pool.release_consumer(reused);
}

void test_completion_pool_timed_wait_is_completion_driven() {
  bounded::CompletionPool pool(1);
  const u32 id = pool.acquire();
  std::thread producer([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    pool.complete(id, true);
  });
  assert(pool.wait_for(id, std::chrono::seconds(1)) ==
         bounded::CompletionPool::Result::success);
  pool.release_consumer(id);
  producer.join();
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

void test_sliding_completion_ring_bounded_smooth_admission() {
  {
    bounded::SlidingCompletionRing ring(8);
    const std::array<u32, 2> initial_work{1, 1};
    assert(ring.reserve_batch(
             span<const u32>{initial_work.data(), initial_work.size()}, 2) == 1);
    auto waiter = std::async(std::launch::async, [&]() {
      const std::array<u32, 1> work{1};
      return ring.reserve_batch(span<const u32>{work.data(), work.size()}, 2);
    });
    assert(waiter.wait_for(std::chrono::milliseconds(10)) ==
           std::future_status::timeout);
    // Out-of-order completion does not release admission credit.
    ring.complete(2);
    assert(waiter.wait_for(std::chrono::milliseconds(10)) ==
           std::future_status::timeout);
    ring.complete(1);
    assert(waiter.wait_for(std::chrono::seconds(1)) ==
           std::future_status::ready);
    ring.complete(waiter.get());
  }

  {
    bounded::SlidingCompletionRing ring(8);
    const std::array<u32, 2> initial_work{1, 1};
    assert(ring.reserve_batch(
             span<const u32>{initial_work.data(), initial_work.size()}, 2) == 1);
    auto first_waiter = std::async(std::launch::async, [&]() {
      const std::array<u32, 1> work{1};
      return ring.reserve_batch(span<const u32>{work.data(), work.size()}, 2);
    });
    auto second_waiter = std::async(std::launch::async, [&]() {
      const std::array<u32, 1> work{1};
      return ring.reserve_batch(span<const u32>{work.data(), work.size()}, 2);
    });
    assert(first_waiter.wait_for(std::chrono::milliseconds(10)) ==
           std::future_status::timeout);
    assert(second_waiter.wait_for(std::chrono::milliseconds(10)) ==
           std::future_status::timeout);

    // One newly available sequence can be claimed by only one producer even
    // though notify_all keeps mixed-size admission work-conserving.
    ring.complete(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool first_ready =
      first_waiter.wait_for(std::chrono::milliseconds(0)) ==
      std::future_status::ready;
    const bool second_ready =
      second_waiter.wait_for(std::chrono::milliseconds(0)) ==
      std::future_status::ready;
    assert(first_ready != second_ready);

    const u64 admitted = first_ready ? first_waiter.get() : second_waiter.get();
    ring.complete(2);
    auto& remaining_waiter = first_ready ? second_waiter : first_waiter;
    assert(remaining_waiter.wait_for(std::chrono::seconds(1)) ==
           std::future_status::ready);
    ring.complete(admitted);
    ring.complete(remaining_waiter.get());
  }
}

void test_sliding_completion_ring_concurrent_batches_never_partially_admit() {
  constexpr size_t kBatches = 4;
  constexpr size_t kBatchItems = 2;
  constexpr size_t kAdmissionWindow = kBatchItems;
  bounded::SlidingCompletionRing ring(kBatches * kBatchItems);
  const std::array<u32, kBatchItems> work{1, 1};

  // All producers enter reserve_batch together.  The admission window can
  // hold exactly one foreground RPC batch, so a per-item reservation scheme
  // could let two producers retain one sequence each and deadlock forever.
  // Atomic batch admission must instead publish exactly one complete pair.
  std::barrier start(static_cast<std::ptrdiff_t>(kBatches + 1));
  std::array<std::future<u64>, kBatches> reservations;
  for (auto& reservation : reservations) {
    reservation = std::async(std::launch::async, [&]() {
      start.arrive_and_wait();
      return ring.reserve_batch(
        span<const u32>{work.data(), work.size()}, kAdmissionWindow);
    });
  }
  start.arrive_and_wait();

  std::array<bool, kBatches> consumed{};
  for (size_t admitted_batches = 0;
       admitted_batches < kBatches;
       ++admitted_batches) {
    const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
    size_t ready_count = 0;
    size_t ready_index = kBatches;
    do {
      ready_count = 0;
      ready_index = kBatches;
      for (size_t index = 0; index < kBatches; ++index) {
        if (consumed[index]) continue;
        if (reservations[index].wait_for(std::chrono::milliseconds(0)) ==
            std::future_status::ready) {
          ++ready_count;
          ready_index = index;
        }
      }
      if (ready_count == 0) std::this_thread::yield();
    } while (ready_count == 0 &&
             std::chrono::steady_clock::now() < deadline);

    // Exactly one producer can commit while the previous pair is live.  In
    // particular, no blocked producer may advance next_sequence by one.
    assert(ready_count == 1);
    const u64 first_sequence = reservations[ready_index].get();
    consumed[ready_index] = true;
    assert(first_sequence == admitted_batches * kBatchItems + 1);
    assert(ring.next_sequence() == first_sequence + kBatchItems);
    assert(ring.outstanding() == kBatchItems);
    for (size_t index = 0; index < kBatches; ++index) {
      if (consumed[index]) continue;
      assert(reservations[index].wait_for(std::chrono::milliseconds(0)) ==
             std::future_status::timeout);
    }

    // Completion may arrive out of order, but admission credit is released
    // only after the complete contiguous pair has finalized.  That release
    // must wake one of the remaining batches and guarantee forward progress.
    ring.complete(first_sequence + 1);
    assert(ring.finalized() == first_sequence - 1);
    assert(ring.outstanding() == kBatchItems);
    ring.complete(first_sequence);
    assert(ring.finalized() == first_sequence + 1);
  }
  assert(ring.finalized() == kBatches * kBatchItems);
  assert(ring.outstanding() == 0);
}

void test_integral_raw_stage2_distance_is_exact() {
  constexpr u32 dim = 128;

  auto verify = [](const std::array<byte_t, dim>& lhs,
                   const std::array<byte_t, dim>& rhs,
                   VectorDType dtype) {
    std::array<float, dim> decoded{};
    decode_storage_vector_to_float(lhs.data(), dtype, dim, decoded.data());
    const float raw = typed_l2_distance(
      lhs.data(), dtype, rhs.data(), dtype, dim);
    const float established = typed_l2_distance_float_query(
      span<const float>{decoded.data(), decoded.size()},
      rhs.data(), dtype, dim);
    assert(raw == established);
  };

  std::array<byte_t, dim> lhs{};
  std::array<byte_t, dim> rhs{};
  for (u32 i = 0; i < dim; ++i) {
    lhs[i] = static_cast<byte_t>((i & 1u) == 0 ? 0 : 255);
    rhs[i] = static_cast<byte_t>((i & 1u) == 0 ? 255 : 0);
  }
  verify(lhs, rhs, VectorDType::uint8);

  auto* signed_lhs = reinterpret_cast<i8*>(lhs.data());
  auto* signed_rhs = reinterpret_cast<i8*>(rhs.data());
  for (u32 i = 0; i < dim; ++i) {
    signed_lhs[i] = (i & 1u) == 0 ? static_cast<i8>(-128)
                                  : static_cast<i8>(127);
    signed_rhs[i] = (i & 1u) == 0 ? static_cast<i8>(127)
                                  : static_cast<i8>(-128);
  }
  verify(lhs, rhs, VectorDType::int8);

  u32 state = 0x9e3779b9u;
  for (u32 sample = 0; sample < 32; ++sample) {
    for (u32 i = 0; i < dim; ++i) {
      state = state * 1664525u + 1013904223u;
      lhs[i] = static_cast<byte_t>(state >> 24);
      state = state * 1664525u + 1013904223u;
      rhs[i] = static_cast<byte_t>(state >> 24);
    }
    verify(lhs, rhs, VectorDType::uint8);
    verify(lhs, rhs, VectorDType::int8);
  }
}

void test_wide_integral_simd_distance_never_overflows() {
  constexpr u32 dim = 100003;
  vec<byte_t> lhs(dim, 0);
  vec<byte_t> rhs(dim, static_cast<byte_t>(255));
  const f64 exact = static_cast<f64>(dim) * 255.0 * 255.0;
  const auto verify = [&](VectorDType dtype) {
    const f32 actual = typed_l2_distance(
      lhs.data(), dtype, rhs.data(), dtype, dim);
    assert(std::isfinite(actual) && actual > 0.0F);
    assert(actual == static_cast<f32>(exact));
  };
  verify(VectorDType::uint8);

  auto* signed_lhs = reinterpret_cast<i8*>(lhs.data());
  auto* signed_rhs = reinterpret_cast<i8*>(rhs.data());
  std::fill(signed_lhs, signed_lhs + dim, static_cast<i8>(-128));
  std::fill(signed_rhs, signed_rhs + dim, static_cast<i8>(127));
  verify(VectorDType::int8);
}

void test_stale_stage2_sequence_handoff_to_bounded_repair() {
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

  // An upsert owns Stage2-finalize+cleanup units. Its ordinary cleanup may finish out
  // of order, but the sequence remains blocked while stale Stage2 finalization repair is
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

void test_stale_stage2_repair_keeps_wire_payload_bound() {
  constexpr size_t kR = 4;
  const vec<RemotePtr> preserved{
    RemotePtr{1, 0x1000}, RemotePtr{1, 0x2000},
    RemotePtr{1, 0x3000}, RemotePtr{1, 0x4000}};
  const vec<RemotePtr> supplemental{
    RemotePtr{2, 0x1000}, RemotePtr{2, 0x2000},
    RemotePtr{2, 0x3000}, RemotePtr{2, 0x4000}};

  // The stale Stage2 finalization transfers its completion unit to a repair descriptor.
  // Even with disjoint R-sized preserved and supplemental sets, that repair
  // sends only the supplemental backlinks attempted by the Stage2 finalization.
  bounded::SlidingCompletionRing ring(2);
  const u64 stale_stage2_sequence = ring.reserve(1);
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
  ring.complete(stale_stage2_sequence);
  assert(ring.finalized() == stale_stage2_sequence);
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

void test_stage2_pressure_retains_a_dedicated_progress_floor() {
  using memory_node_storage_owner_maintenance_detail::
    stage2_context_admission_limit;

  // The normal path can hide peer latency with the configured RPC depth.
  assert(stage2_context_admission_limit(2, 16, false) == 32);
  // Foreground pressure throttles that fanout but cannot stop both dedicated
  // Stage2 workers, otherwise a full completion window deadlocks Stage1 ACKs.
  assert(stage2_context_admission_limit(2, 16, true) == 2);
  assert(stage2_context_admission_limit(1, 16, true) == 1);
  assert(stage2_context_admission_limit(0, 0, true) == 1);
  assert(stage2_context_admission_limit(0, 0, false) == 1);
}

void test_stage2_sequence_window_tracks_service_capacity_not_lane_rebalance() {
  using memory_node_storage_owner_maintenance_detail::
    saturating_admission_multiply;
  using memory_node_storage_owner_maintenance_detail::
    stage2_sequence_admission_limit;

  // The current colocated deployment keeps exactly four wire batches of
  // bounded debt whether two or four Stage2 workers service it.  Reassigning
  // idle CPUs therefore cannot manufacture short-term ACK throughput by
  // silently doubling the backlog window.
  assert(stage2_sequence_admission_limit(2, 16, 32) == 128);
  assert(stage2_sequence_admission_limit(4, 16, 32) == 128);
  assert(stage2_sequence_admission_limit(8, 16, 32) == 128);

  // A larger wire batch used to expose a hidden dependency on the actual
  // post-rebalance worker count (2 workers -> 128, 4 workers -> 256).  Runtime
  // passes the CPU plan's two-worker admission baseline, so the window stays
  // at the legacy two-worker bound for every batch size.
  assert(stage2_sequence_admission_limit(2, 16, 64) == 128);

  // A genuinely larger executor/context population may expose one task per
  // context, while tiny settings retain a nonzero, batch-safe bound.
  assert(stage2_sequence_admission_limit(16, 16, 32) == 256);
  assert(stage2_sequence_admission_limit(1, 1, 32) == 32);
  assert(stage2_sequence_admission_limit(4, 16, 1) == 64);
  assert(stage2_sequence_admission_limit(0, 0, 0) == 4);

  // Every finite policy point is no larger than the previous four-task-per-
  // context bound (except that one complete wire batch is always admitted),
  // and hostile configuration arithmetic saturates instead of wrapping to a
  // tiny, unsafe window.
  for (std::size_t workers = 0; workers <= 32; ++workers) {
    for (std::size_t depth = 0; depth <= 32; ++depth) {
      for (std::size_t batch = 0; batch <= 128; batch += 8) {
        const std::size_t normalized_workers = std::max<std::size_t>(1, workers);
        const std::size_t normalized_depth = std::max<std::size_t>(1, depth);
        const std::size_t normalized_batch = std::max<std::size_t>(1, batch);
        const std::size_t legacy_limit = std::max(
          normalized_batch,
          saturating_admission_multiply(
            saturating_admission_multiply(
              normalized_workers, normalized_depth),
            4));
        const std::size_t limit = stage2_sequence_admission_limit(
          workers, depth, batch);
        assert(limit >= normalized_batch);
        assert(limit <= legacy_limit);
      }
    }
  }
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  assert(stage2_sequence_admission_limit(max_size, max_size, max_size) ==
         max_size);
  assert(stage2_sequence_admission_limit(max_size, 2, 1) == max_size);
}

void test_stage1_arm_queue_permit_cannot_be_stolen() {
  using memory_node_storage_owner_maintenance_detail::
    maintenance_queue_permit_available;

  assert(maintenance_queue_permit_available(3, 0, 4));
  // Once arm owns the last slot, a generic cleanup/repair producer observes
  // the queue as full even though the arm task is not enqueued yet.
  assert(!maintenance_queue_permit_available(3, 1, 4));
  assert(!maintenance_queue_permit_available(2, 2, 4));
  assert(maintenance_queue_permit_available(2, 1, 4));
  assert(!maintenance_queue_permit_available(0, 0, 0));
}

void test_stage1_arm_batch_queue_permit_is_atomic_and_bounded() {
  using memory_node_storage_owner_maintenance_detail::
    maintenance_queue_batch_permit_available;
  using memory_node_storage_owner_maintenance_detail::
    maintenance_queue_permit_available;

  constexpr size_t capacity = 8;
  constexpr size_t runnable = 3;
  constexpr size_t reserved = 2;
  constexpr size_t remaining = capacity - runnable - reserved;
  static_assert(remaining == 3);

  // An entire control-RPC batch can claim the exact remaining capacity, but
  // it cannot partially reserve a batch that is one item too large.
  assert(maintenance_queue_batch_permit_available(
    runnable, reserved, remaining, capacity));
  assert(!maintenance_queue_batch_permit_available(
    runnable, reserved, remaining + 1, capacity));
  assert(!maintenance_queue_batch_permit_available(
    runnable, reserved, 0, capacity));

  // Reject inconsistent snapshots before performing any unsigned capacity
  // subtraction: runnable work cannot exceed the queue, and reservations
  // cannot exceed the capacity left by runnable work.
  assert(!maintenance_queue_batch_permit_available(9, 0, 1, capacity));
  assert(!maintenance_queue_batch_permit_available(6, 3, 1, capacity));
  assert(!maintenance_queue_batch_permit_available(8, 1, 1, capacity));
  assert(!maintenance_queue_batch_permit_available(0, 0, 1, 0));

  // Batch and legacy single-slot permits share one reserved-slots account.
  // One pre-existing single-slot permit plus a two-item batch exactly fills
  // this queue; after accounting for that batch no producer can steal a slot.
  constexpr size_t shared_capacity = 5;
  constexpr size_t shared_runnable = 2;
  constexpr size_t single_reserved = 1;
  assert(maintenance_queue_permit_available(
    shared_runnable, single_reserved, shared_capacity));
  assert(maintenance_queue_batch_permit_available(
    shared_runnable, single_reserved, 2, shared_capacity));
  constexpr size_t all_reserved = single_reserved + 2;
  assert(!maintenance_queue_permit_available(
    shared_runnable, all_reserved, shared_capacity));
  assert(!maintenance_queue_batch_permit_available(
    shared_runnable, all_reserved, 1, shared_capacity));

  // Releasing either kind of permit exposes the same single slot again.
  assert(maintenance_queue_permit_available(
    shared_runnable, all_reserved - 1, shared_capacity));
  assert(maintenance_queue_batch_permit_available(
    shared_runnable, all_reserved - 1, 1, shared_capacity));
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

void test_reverse_overflow_uses_alpha_robust_prune_not_nearest_r() {
  struct Candidate {
    RemotePtr rptr;
    distance_t source_distance;
    distance_t coordinate;
  };

  // A and B are the two nearest candidates to the reverse target, but are
  // almost duplicates of one another.  Alpha RobustPrune must retain the
  // diverse C instead of producing nearest-R = {A, B}.
  const Candidate a{RemotePtr{0, 4096}, 1.0f, 0.0f};
  const Candidate b{RemotePtr{0, 8192}, 1.1f, 0.1f};
  const Candidate c{RemotePtr{0, 12288}, 2.0f, 10.0f};
  const Candidate d{RemotePtr{0, 16384}, 3.0f, 20.0f};
  const vec<Candidate> sorted{a, b, c, d};
  vec<RemotePtr> selected;
  vec<size_t> selected_indices;

  memory_node_storage_owner_index_detail::
    select_alpha_robust_pruned_sorted(
      span<const Candidate>{sorted.data(), sorted.size()},
      2,
      1.2,
      [](const Candidate& candidate) { return candidate.rptr; },
      [](const Candidate& candidate) {
        return candidate.source_distance;
      },
      [](const Candidate& lhs, const Candidate& rhs) {
        const distance_t delta = lhs.coordinate - rhs.coordinate;
        return delta * delta;
      },
      selected,
      selected_indices);

  assert(selected.size() == 2);
  assert(selected[0] == a.rptr);
  assert(selected[1] == c.rptr);
}

void test_stage2_rebase_preserves_post_stage1_reverse_edge() {
  const RemotePtr stage1_a{0, 4096};
  const RemotePtr stage1_b{0, 8192};
  const RemotePtr globally_selected{1, 4096};
  const RemotePtr concurrent_reverse{0, 12288};
  const vec<RemotePtr> stage1_neighbors{stage1_a, stage1_b};
  const vec<RemotePtr> global_neighbors{globally_selected};
  const vec<RemotePtr> observed_neighbors{
    stage1_a, concurrent_reverse, globally_selected};

  const vec<RemotePtr> rebased =
    memory_node_storage_owner_maintenance_detail::
      merge_stage2_rebase_candidates(
        span<const RemotePtr>{global_neighbors.data(),
                              global_neighbors.size()},
        span<const RemotePtr>{stage1_neighbors.data(),
                              stage1_neighbors.size()},
        span<const RemotePtr>{observed_neighbors.data(),
                              observed_neighbors.size()});

  // The old stage1-only edge may be replaced by global prune, while the edge
  // acknowledged after stage1 must enter the final locked re-prune.
  assert(rebased.size() == 2);
  assert(rebased[0] == globally_selected);
  assert(rebased[1] == concurrent_reverse);
}

void test_cleanup_identity_fences_both_reused_slots() {
  using namespace memory_node_storage_owner_maintenance_detail;
  assert(cleanup_deleted_candidate_matches(7, 3, 7, 3, true));
  assert(!cleanup_deleted_candidate_matches(7, 3, 8, 3, true));
  assert(!cleanup_deleted_candidate_matches(7, 3, 7, 4, true));
  assert(!cleanup_deleted_candidate_matches(7, 3, 7, 3, false));

  assert(cleanup_reverse_target_matches(11, 5, 11, 5, false));
  // A delayed cleanup is an idempotent no-op when the target address has
  // been tombstoned or reused by either a different ID or generation.
  assert(!cleanup_reverse_target_matches(11, 5, 12, 5, false));
  assert(!cleanup_reverse_target_matches(11, 5, 11, 6, false));
  assert(!cleanup_reverse_target_matches(11, 5, 11, 5, true));
}

void test_protected_reparent_order_is_local_bounded_and_deterministic() {
  using memory_node_storage_owner_maintenance_detail::
    order_protected_reparent_candidates;
  const RemotePtr child{2, 0x1000};
  const RemotePtr retiring{2, 0x2000};
  const RemotePtr local_high{2, 0x5000};
  const RemotePtr local_low{2, 0x3000};
  const RemotePtr remote_low{0, 0x1000};
  const RemotePtr remote_high{3, 0x7000};
  const vec<RemotePtr> neighbors{
    remote_high, retiring, local_high, remote_low, child, local_low,
    local_high};
  const vec<RemotePtr> ordered = order_protected_reparent_candidates(
    child, retiring, span<const RemotePtr>{neighbors});
  assert((ordered == vec<RemotePtr>{
                       local_low, local_high, remote_low, remote_high}));
  assert(ordered.size() <= neighbors.size());
}

void test_protected_reparent_capacity_never_evicts_existing_work() {
  using memory_node_storage_owner_maintenance_detail::
    protected_reparent_target_has_capacity;
  const RemotePtr child{1, 0x1000};
  const RemotePtr protected_other{1, 0x3000};
  const vec<RemotePtr> one_protected{protected_other};
  const vec<RemotePtr> full_protected{
    protected_other, RemotePtr{1, 0x4000}};
  assert(protected_reparent_target_has_capacity(
    child, one_protected, 2));
  assert(!protected_reparent_target_has_capacity(
    child, full_protected, 2));
  const vec<RemotePtr> already_reserved{protected_other, child};
  assert(protected_reparent_target_has_capacity(
    child, already_reserved, 2));
}

}  // namespace

int main() {
  test_queue_wrap_and_capacity();
  test_queue_multiple_producers();
  test_queue_later_publication_can_be_blocked_by_reserved_head();
  test_queue_stopped_producer_does_not_overwrite_full_queue();
  test_queue_publication_precedes_slot_reuse();
  test_completion_pool_reuse_and_abandon();
  test_completion_pool_timed_wait_preserves_producer_reference();
  test_completion_pool_timed_wait_is_completion_driven();
  test_sliding_completion_ring();
  test_sliding_completion_ring_atomic_batch_admission();
  test_sliding_completion_ring_bounded_smooth_admission();
  test_sliding_completion_ring_concurrent_batches_never_partially_admit();
  test_integral_raw_stage2_distance_is_exact();
  test_wide_integral_simd_distance_never_overflows();
  test_stale_stage2_sequence_handoff_to_bounded_repair();
  test_stale_stage2_repair_keeps_wire_payload_bound();
  test_stage2_admission_yields_only_for_live_foreground_pressure();
  test_stage2_pressure_retains_a_dedicated_progress_floor();
  test_stage2_sequence_window_tracks_service_capacity_not_lane_rebalance();
  test_stage1_arm_queue_permit_cannot_be_stolen();
  test_stage1_arm_batch_queue_permit_is_atomic_and_bounded();
  test_reverse_candidate_is_revalidated_at_locked_write_boundary();
  test_reverse_overflow_uses_alpha_robust_prune_not_nearest_r();
  test_stage2_rebase_preserves_post_stage1_reverse_edge();
  test_cleanup_identity_fences_both_reused_slots();
  test_protected_reparent_order_is_local_bounded_and_deterministic();
  test_protected_reparent_capacity_never_evicts_existing_work();
  return 0;
}
