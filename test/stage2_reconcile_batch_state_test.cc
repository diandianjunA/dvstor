#include <cassert>
#include <cstdint>

#include "memory_node/storage_owner_maintenance/reconcile_batch_state.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;
namespace protocol = service::storage_owner;

namespace {

protocol::ReconcileReverseOp op(std::uint64_t target,
                                std::uint64_t sequence,
                                std::uint32_t id) {
  return protocol::ReconcileReverseOp{
    .target_raw = target,
    .old_candidate_raw = target + 1,
    .new_candidate_raw = target + 2,
    .placement_sequence = sequence,
    .id = id,
    .generation = id + 10,
    .kind = static_cast<std::uint32_t>(
      protocol::ReconcileReverseOpKind::replace_or_add),
  };
}

void test_exact_payload_and_out_of_order_completion() {
  detail::Stage2ReconcileBatchState state;
  state.reserve(8, 4);
  const detail::Stage2ContextHandle context{3, 7};
  state.begin(context, detail::Stage2ReconcileBarrier::promotion);
  const std::uint32_t epoch = state.epoch();

  protocol::ReconcileReverseOp first[] = {
    op(100, 9, 1), op(200, 10, 2)};
  const protocol::ReconcileReverseOp second[] = {op(300, 11, 3)};
  assert(state.append_chunk(41, 1, first));
  assert(state.append_chunk(42, 4, second));
  first[0].target_raw = 999;
  assert(state.remaining() == 2);

  const auto first_payload = state.payload(state.chunks()[0]);
  assert(first_payload.size() == 2);
  assert(first_payload[0].target_raw == 100);
  assert(first_payload[0].placement_sequence == 9);
  assert(first_payload[1].target_raw == 200);
  assert(first_payload[1].id == 2);

  // ACK order is unrelated to post order and duplicate completion is
  // idempotent inside the same generation/epoch.
  assert(state.mark_complete(1, context, epoch));
  assert(state.remaining() == 1);
  assert(state.mark_complete(1, context, epoch));
  assert(state.remaining() == 1);
  assert(state.mark_complete(0, context, epoch));
  assert(state.complete());
}

void test_epoch_and_context_generation_fence_late_ack() {
  detail::Stage2ReconcileBatchState state;
  const detail::Stage2ContextHandle old_context{1, 4};
  state.begin(old_context, detail::Stage2ReconcileBarrier::promotion);
  const protocol::ReconcileReverseOp payload[] = {op(10, 1, 2)};
  assert(state.append_chunk(7, 2, payload));
  const std::uint32_t promotion_epoch = state.epoch();

  state.clear();
  state.begin(old_context, detail::Stage2ReconcileBarrier::stable);
  assert(state.append_chunk(8, 2, payload));
  assert(state.epoch() != promotion_epoch);
  assert(!state.mark_complete(0, old_context, promotion_epoch));
  assert(state.remaining() == 1);

  const std::uint32_t stable_epoch = state.epoch();
  state.clear();
  const detail::Stage2ContextHandle reused_context{1, 5};
  state.begin(reused_context, detail::Stage2ReconcileBarrier::removal);
  assert(state.append_chunk(9, 2, payload));
  assert(!state.mark_complete(0, old_context, stable_epoch));
  assert(state.remaining() == 1);
}

void test_empty_barrier_completes_without_transport() {
  detail::Stage2ReconcileBatchState state;
  const detail::Stage2ContextHandle context{2, 9};
  state.begin(context, detail::Stage2ReconcileBarrier::stable);
  assert(state.active());
  assert(state.complete());
  assert(state.remaining() == 0);
  assert(state.chunks().empty());
  assert(state.barrier() == detail::Stage2ReconcileBarrier::stable);
}

void test_barrier_subphases_release_search_lane() {
  using Barrier = detail::Stage2ReconcileBarrier;
  using Subphase = detail::Stage2FinalizeSubphase;
  assert(detail::stage2_reconcile_wait_subphase(Barrier::promotion) ==
         Subphase::promotion_wait);
  assert(detail::stage2_reconcile_wait_subphase(Barrier::stable) ==
         Subphase::stable_wait);
  assert(detail::stage2_reconcile_wait_subphase(Barrier::removal) ==
         Subphase::removal_wait);
  assert(detail::stage2_finalize_subphase_needs_lane(Subphase::prepare));
  assert(detail::stage2_finalize_subphase_needs_lane(
    Subphase::placement_ready));
  assert(!detail::stage2_finalize_subphase_needs_lane(
    Subphase::promotion_wait));
  assert(!detail::stage2_finalize_subphase_needs_lane(Subphase::stable_wait));
  assert(!detail::stage2_finalize_subphase_needs_lane(Subphase::removal_wait));
}

}  // namespace

int main() {
  test_exact_payload_and_out_of_order_completion();
  test_epoch_and_context_generation_fence_late_ack();
  test_empty_barrier_completes_without_transport();
  test_barrier_subphases_release_search_lane();
  return 0;
}
