#include <array>
#include <cassert>

#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"

namespace {

using namespace service::storage_owner;
using memory_node_peer_rpc_detail::Stage1ControlHomeProgress;
using memory_node_peer_rpc_detail::Stage1ControlResponseDisposition;
using memory_node_peer_rpc_detail::classify_stage1_control_response;

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

}  // namespace

int main() {
  test_atomic_home_arm_response();
  test_release_is_an_idempotent_ordered_watermark();
  test_structural_corruption_never_becomes_retry();
  test_each_home_advances_independently();
}
