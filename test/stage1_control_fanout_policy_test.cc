#include <array>
#include <cassert>

#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"

namespace {

using namespace service::storage_owner;
using memory_node_peer_rpc_detail::Stage1ControlHomeProgress;
using memory_node_peer_rpc_detail::Stage1ControlResponseDisposition;
using memory_node_peer_rpc_detail::classify_stage1_control_response;
using memory_node_peer_rpc_detail::
  defer_fused_stage1_success_for_atomic_retry;
using memory_node_peer_rpc_detail::stage1_execute_success_has_expected_fence;
using memory_node_peer_rpc_detail::stage1_execute_uses_fused_arm;
using memory_node_peer_rpc_detail::valid_fused_stage1_execute_item;

Stage1ArmItem arm_item(u32 item_index, Stage1ArmAction action) {
  return Stage1ArmItem{
    .token = AuthorityOperationToken{
      .source_client = 7,
      .item_index = item_index,
      .client_batch_id = 99,
    },
    .target_raw = 0x1000 + item_index * 0x100,
    .initial_placement_version =
      action == Stage1ArmAction::arm ||
          action == Stage1ArmAction::release
        ? u64{4} : u64{0},
    .id = 100 + item_index,
    .generation = 3,
    .action = static_cast<u32>(action),
  };
}

template <size_t N>
Stage1ControlResponseDisposition classify(
    const std::array<Stage1ArmItem, N>& inputs,
    const std::array<Stage1ArmResult, N>& outputs) {
  return classify_stage1_control_response(
    span<const Stage1ArmItem>{inputs.data(), inputs.size()},
    span<const Stage1ArmResult>{outputs.data(), outputs.size()});
}

Stage1ArmResult ok_result(const Stage1ArmItem& input) {
  return Stage1ArmResult{
    .token = input.token,
    .target_raw = input.target_raw,
    .maintenance_sequence = 1000 + input.token.item_index,
    .status = static_cast<u32>(MutationStatus::ok),
  };
}

void test_atomic_home_arm_response() {
  const std::array inputs{
    arm_item(0, Stage1ArmAction::arm),
    arm_item(1, Stage1ArmAction::arm),
  };
  std::array outputs{ok_result(inputs[0]), ok_result(inputs[1])};
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::resolved);

  // A generic failure is terminal: it represents structural/identity
  // conflict and must not be converted into an unbounded retry.
  outputs[1].status = static_cast<u32>(MutationStatus::failed);
  outputs[1].maintenance_sequence = 0;
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::malformed);

  // Only the explicit internal retry status denotes transient preparation,
  // arming, or bounded maintenance-credit pressure.
  outputs[1].status = static_cast<u32>(MutationStatus::retry);
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::retry);

  outputs[1].status = static_cast<u32>(MutationStatus::not_found);
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::malformed);
}

void test_release_is_an_idempotent_ordered_watermark() {
  const std::array inputs{
    arm_item(2, Stage1ArmAction::release),
    arm_item(3, Stage1ArmAction::release),
  };
  std::array outputs{ok_result(inputs[0]), ok_result(inputs[1])};
  // A replayed release may find no receipt and therefore returns no target or
  // sequence; matching token + ok is the authoritative postcondition.
  outputs[0].target_raw = 0;
  outputs[0].maintenance_sequence = 0;
  outputs[1].maintenance_sequence = 0;
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::resolved);
}

void test_structural_corruption_never_becomes_retry() {
  const std::array inputs{arm_item(4, Stage1ArmAction::arm)};
  std::array outputs{ok_result(inputs[0])};

  outputs[0].token.client_batch_id++;
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::malformed);
  outputs[0] = ok_result(inputs[0]);
  outputs[0].target_raw++;
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::malformed);
  outputs[0] = ok_result(inputs[0]);
  outputs[0].maintenance_sequence = 0;
  assert(classify(inputs, outputs) ==
         Stage1ControlResponseDisposition::malformed);
}

void test_each_home_advances_independently() {
  std::array<Stage1ControlHomeProgress, 3> homes;
  for (auto& home : homes) home.mark_posted();

  // Home 1 ACKs and can be committed while homes 0 and 2 remain in flight.
  assert(homes[1].mark_resolved());
  assert(homes[1].resolved());
  assert(homes[0].posted());
  assert(homes[2].posted());

  // Only the timed-out home 0 becomes eligible for repost. A late timeout on
  // committed home 1 cannot reopen its token or duplicate authority commit.
  homes[0].mark_retry();
  homes[1].mark_retry();
  assert(homes[0].needs_post());
  assert(!homes[1].needs_post());
  assert(!homes[2].needs_post());
  assert(!homes[1].mark_resolved());
}

void test_home_commit_releases_credit_before_next_home_arm() {
  // Model the minimum completion window that exposed the distributed cycle:
  // one remote fused home has armed and owns the only credit while a local
  // home still needs to arm. Consuming the remote ACK must invoke its commit
  // and ordered receipt release before the coordinator may enter the
  // potentially blocking local ARM.
  size_t available_completion_credits = 1;
  size_t available_receipt_slots = 1;
  std::array<bool, 2> authority_committed{};
  std::array<bool, 2> receipt_released{};
  std::array<Stage1ControlHomeProgress, 2> homes;

  const auto try_arm = [&](size_t home) {
    if (available_completion_credits == 0 ||
        available_receipt_slots == 0) {
      return false;
    }
    --available_completion_credits;
    --available_receipt_slots;
    homes[home].mark_posted();
    return true;
  };
  const auto resolve_and_commit = [&](size_t home) {
    if (!homes[home].mark_resolved()) return false;
    authority_committed[home] = true;
    // This represents Stage2 crossing its authority gate and retiring the
    // bounded completion sequence. With an all-home commit barrier this
    // release cannot happen and the second arm has no legal transition.
    ++available_completion_credits;
    return true;
  };
  const auto release_receipt = [&](size_t home) {
    if (!authority_committed[home] || receipt_released[home]) return false;
    receipt_released[home] = true;
    ++available_receipt_slots;
    return true;
  };

  assert(try_arm(0));
  assert(!try_arm(1));
  assert(resolve_and_commit(0));
  assert(authority_committed[0]);
  // Authority commit alone frees Stage2 completion credit, but an all-home
  // receipt barrier can still block the next prepare forever.
  assert(!try_arm(1));
  assert(release_receipt(0));
  assert(try_arm(1));
  assert(resolve_and_commit(1));
  assert(authority_committed[1]);
  assert(release_receipt(1));
  assert(available_completion_credits == 1);
  assert(available_receipt_slots == 1);

  // A duplicated/late response or release cannot manufacture capacity.
  assert(!resolve_and_commit(0));
  assert(!release_receipt(0));
  assert(available_completion_credits == 1);
  assert(available_receipt_slots == 1);
}

void test_fused_execute_requires_fresh_insert_and_runnable_fence() {
  Stage1ExecuteItem input{
    .client_batch_id = 99,
    .old_raw = 0,
    .initial_placement_version = 4,
    .source_client = 7,
    .item_index = 1,
    .id = 101,
    .generation = 3,
    .kind = static_cast<u32>(MutationKind::insert),
    .authority_shard = 2,
  };
  Stage1ExecuteResult output{
    .client_batch_id = input.client_batch_id,
    .target_raw = 0x1000,
    .maintenance_sequence = 77,
    .source_client = input.source_client,
    .item_index = input.item_index,
    .status = static_cast<u32>(MutationStatus::ok),
  };
  assert(stage1_execute_uses_fused_arm(input));
  assert(valid_fused_stage1_execute_item(input));
  assert(stage1_execute_success_has_expected_fence(input, output));

  output.maintenance_sequence = 0;
  assert(!stage1_execute_success_has_expected_fence(input, output));
  output.status = static_cast<u32>(MutationStatus::retry);
  assert(stage1_execute_success_has_expected_fence(input, output));

  input.kind = static_cast<u32>(MutationKind::upsert);
  assert(!valid_fused_stage1_execute_item(input));
  output.status = static_cast<u32>(MutationStatus::ok);
  output.maintenance_sequence = 77;
  assert(!stage1_execute_success_has_expected_fence(input, output));

  input.kind = static_cast<u32>(MutationKind::insert);
  input.old_raw = 0x2000;
  assert(!valid_fused_stage1_execute_item(input));

  // Legacy prepare succeeds before standalone cleanup/arm and must not claim
  // a Stage2 sequence in its Execute response.
  input.old_raw = 0;
  input.initial_placement_version = 0;
  output.maintenance_sequence = 0;
  assert(!stage1_execute_uses_fused_arm(input));
  assert(stage1_execute_success_has_expected_fence(input, output));
  output.maintenance_sequence = 77;
  assert(!stage1_execute_success_has_expected_fence(input, output));
}

void test_atomic_retry_never_hides_terminal_prepare_failure() {
  Stage1ExecuteResult success{
    .maintenance_sequence = 77,
    .status = static_cast<u32>(MutationStatus::ok),
  };
  defer_fused_stage1_success_for_atomic_retry(success);
  assert(success.status == static_cast<u32>(MutationStatus::retry));
  assert(success.maintenance_sequence == 0);

  Stage1ExecuteResult terminal{
    .status = static_cast<u32>(MutationStatus::failed),
  };
  defer_fused_stage1_success_for_atomic_retry(terminal);
  assert(terminal.status == static_cast<u32>(MutationStatus::failed));
  assert(terminal.maintenance_sequence == 0);

  Stage1ExecuteResult transient{
    .status = static_cast<u32>(MutationStatus::retry),
  };
  defer_fused_stage1_success_for_atomic_retry(transient);
  assert(transient.status == static_cast<u32>(MutationStatus::retry));
  assert(transient.maintenance_sequence == 0);
}

}  // namespace

int main() {
  test_atomic_home_arm_response();
  test_release_is_an_idempotent_ordered_watermark();
  test_structural_corruption_never_becomes_retry();
  test_each_home_advances_independently();
  test_home_commit_releases_credit_before_next_home_arm();
  test_fused_execute_requires_fresh_insert_and_runnable_fence();
  test_atomic_retry_never_hides_terminal_prepare_failure();
}
