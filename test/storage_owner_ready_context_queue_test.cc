#include <cassert>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "memory_node/storage_owner_maintenance/ready_context_queue.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;

namespace {

void test_coalesces_reasons_and_fences_generation() {
  detail::Stage2ReadyContextQueue queue(11, 3, 4);
  const auto first = queue.activate(1);
  assert(queue.notify(first, detail::Stage2ContextReadyReason::rpc_response));
  assert(queue.notify(first,
                      detail::Stage2ContextReadyReason::rdma_completion));

  detail::Stage2ReadyContextEvent event;
  assert(queue.try_pop(event));
  assert(event.owner == first);
  assert((event.reasons & detail::stage2_ready_reason_bits(
            detail::Stage2ContextReadyReason::rpc_response)) != 0);
  assert((event.reasons & detail::stage2_ready_reason_bits(
            detail::Stage2ContextReadyReason::rdma_completion)) != 0);
  assert(!queue.try_pop(event));

  assert(queue.deactivate(first));
  const auto second = queue.activate(1);
  assert(!queue.notify(first,
                       detail::Stage2ContextReadyReason::rpc_response));
  assert(queue.notify(second,
                      detail::Stage2ContextReadyReason::reverse_completion));
  assert(queue.try_pop(event));
  assert(event.owner == second);
}

void test_concurrent_notifications_do_not_lose_edges() {
  detail::Stage2ReadyContextQueue queue(22, 1, 2);
  const auto owner = queue.activate(0);
  std::thread rpc([&] {
    for (int i = 0; i < 10'000; ++i) {
      assert(queue.notify(
        owner, detail::Stage2ContextReadyReason::rpc_response));
    }
  });
  std::thread rdma([&] {
    for (int i = 0; i < 10'000; ++i) {
      assert(queue.notify(
        owner, detail::Stage2ContextReadyReason::rdma_completion));
    }
  });
  rpc.join();
  rdma.join();

  detail::Stage2ReadyContextEvent event;
  assert(queue.try_pop(event));
  assert(event.owner == owner);
  assert(event.reasons != 0);
  assert(!queue.try_pop(event));
}

void test_concurrent_publish_consume_never_loses_a_ready_generation() {
  constexpr std::uint32_t kSlots = 16;
  constexpr std::uint32_t kRounds = 50'000;
  detail::Stage2ReadyContextQueue queue(23, 5, kSlots);
  std::vector<detail::Stage2ContextOwnerKey> owners;
  owners.reserve(kSlots);
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    owners.push_back(queue.activate(slot));
  }

  std::atomic<std::uint32_t> published{0};
  std::atomic<std::uint32_t> consumed{0};
  std::thread producer([&] {
    for (std::uint32_t round = 1; round <= kRounds; ++round) {
      while (consumed.load(std::memory_order_acquire) + 1 != round) {
        std::this_thread::yield();
      }
      const auto& owner = owners[round % kSlots];
      assert(queue.notify(
        owner, detail::Stage2ContextReadyReason::rpc_response));
      assert(queue.notify(
        owner, detail::Stage2ContextReadyReason::rdma_completion));
      published.store(round, std::memory_order_release);
    }
  });

  for (std::uint32_t round = 1; round <= kRounds; ++round) {
    while (published.load(std::memory_order_acquire) != round) {
      std::this_thread::yield();
    }
    detail::Stage2ReadyContextEvent event;
    while (!queue.try_pop(event)) std::this_thread::yield();
    assert(event.owner == owners[round % kSlots]);
    assert((event.reasons & detail::stage2_ready_reason_bits(
              detail::Stage2ContextReadyReason::rpc_response)) != 0);
    assert((event.reasons & detail::stage2_ready_reason_bits(
              detail::Stage2ContextReadyReason::rdma_completion)) != 0);
    consumed.store(round, std::memory_order_release);
  }
  producer.join();

  for (const auto& owner : owners) assert(queue.deactivate(owner));
}

void test_activation_rejects_live_slot_and_geometry_before_allocation() {
  detail::Stage2ReadyContextQueue queue(33, 0, 1);
  const auto owner = queue.activate(0);
  bool duplicate_rejected = false;
  try {
    (void)queue.activate(0);
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  assert(duplicate_rejected);
  assert(queue.deactivate(owner));

  if constexpr (std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<std::uint32_t>::max()) {
    bool geometry_rejected = false;
    try {
      detail::Stage2ReadyContextQueue invalid(
        44, 0,
        static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()) + 1);
    } catch (const std::invalid_argument&) {
      geometry_rejected = true;
    }
    assert(geometry_rejected);
  }
}

void test_stale_ticket_pressure_recovers_current_generation() {
  detail::Stage2ReadyContextQueue queue(55, 2, 1);
  const auto first = queue.activate(0);
  assert(queue.notify(first, detail::Stage2ContextReadyReason::timer));
  assert(queue.deactivate(first));
  const auto second = queue.activate(0);
  assert(queue.notify(second, detail::Stage2ContextReadyReason::timer));
  assert(queue.deactivate(second));
  const auto current = queue.activate(0);
  // The two-slot ticket ring still contains both stale generations. The live
  // event is retained in its slot and recovered after those tickets drain.
  assert(queue.notify(
    current, detail::Stage2ContextReadyReason::rdma_completion));
  assert(queue.overflowed());

  detail::Stage2ReadyContextEvent event;
  assert(!queue.try_pop(event));
  std::vector<detail::Stage2ReadyContextEvent> recovered;
  recovered.reserve(1);
  assert(queue.recover_overflow(recovered) == 1);
  assert(recovered.size() == 1);
  assert(recovered.front().owner == current);
  assert(queue.deactivate(current));
}

}  // namespace

int main() {
  test_coalesces_reasons_and_fences_generation();
  test_concurrent_notifications_do_not_lose_edges();
  test_concurrent_publish_consume_never_loses_a_ready_generation();
  test_activation_rejects_live_slot_and_geometry_before_allocation();
  test_stale_ticket_pressure_recovers_current_generation();
  return 0;
}
