#include <array>
#include <cassert>
#include <vector>

#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"

namespace {

using namespace service::storage_owner;
using memory_node_peer_rpc_detail::Stage1ControlHomeProgress;
using memory_node_peer_rpc_detail::Stage1ControlResponseDisposition;
using memory_node_peer_rpc_detail::Stage1HomeRetryBackoff;
using memory_node_peer_rpc_detail::classify_stage1_control_response;
using memory_node_peer_rpc_detail::dequeue_stage2_home_first;
using memory_node_peer_rpc_detail::make_fused_stage1_release_item;
using memory_node_peer_rpc_detail::partition_stage1_control_response;
using memory_node_peer_rpc_detail::partition_stage1_execute_response;
using memory_node_peer_rpc_detail::stage1_peer_attempt_timeout;
using memory_node_peer_rpc_detail::write_stage1_retry_response;
using memory_node_peer_rpc_detail::stage1_execute_success_has_expected_fence;
using memory_node_peer_rpc_detail::stage1_execute_tokens_unique;
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

void test_stage2_home_queue_cannot_starve_behind_stage1() {
  u32 stage1_streak = 0;
  for (u32 iteration = 0;
       iteration < memory_node_peer_rpc_detail::kStage1MaximumDequeueBurst;
       ++iteration) {
    assert(!dequeue_stage2_home_first(true, true, stage1_streak));
  }
  assert(dequeue_stage2_home_first(true, true, stage1_streak));
  assert(stage1_streak == 0);

  // Either queue remains independently work-conserving.
  assert(dequeue_stage2_home_first(false, true, stage1_streak));
  assert(!dequeue_stage2_home_first(true, false, stage1_streak));

  // A long Stage1-only interval retains a saturated fairness baton, so newly
  // arrived completion-producing work is served immediately.
  for (u32 iteration = 0; iteration < 100; ++iteration) {
    assert(!dequeue_stage2_home_first(true, false, stage1_streak));
  }
  assert(dequeue_stage2_home_first(true, true, stage1_streak));
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

  // A sibling Execute may still be live while the first receipt is already
  // quiescent. Consume/release the resolved token once and carry only the
  // unfinished token into the next control RPC.
  outputs[1].status = static_cast<u32>(MutationStatus::retry);
  std::vector<u32> resolved_slots;
  std::vector<u32> retry_slots;
  assert(partition_stage1_control_response(
    span<const Stage1ArmItem>{inputs.data(), inputs.size()},
    span<const Stage1ArmResult>{outputs.data(), outputs.size()},
    resolved_slots, retry_slots));
  assert((resolved_slots == std::vector<u32>{0}));
  assert((retry_slots == std::vector<u32>{1}));

  // Retrying the second release after quiescence is idempotent even if its
  // receipt was independently erased by a previous ACK replay.
  const std::array retry_input{inputs[1]};
  auto replay = ok_result(retry_input[0]);
  replay.target_raw = 0;
  replay.maintenance_sequence = 0;
  const std::array retry_output{replay};
  assert(partition_stage1_control_response(
    span<const Stage1ArmItem>{retry_input.data(), retry_input.size()},
    span<const Stage1ArmResult>{retry_output.data(), retry_output.size()},
    resolved_slots, retry_slots));
  assert((resolved_slots == std::vector<u32>{0}));
  assert(retry_slots.empty());
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

void test_stage1_retry_backoff_is_bounded_and_phase_local() {
  using namespace std::chrono_literals;
  Stage1HomeRetryBackoff backoff;
  const auto epoch = Stage1HomeRetryBackoff::Clock::time_point{} + 1s;

  assert(backoff.ready(epoch));
  assert(backoff.next_delay_us() == 250);
  backoff.schedule(epoch);
  assert(!backoff.ready(epoch + 249us));
  assert(backoff.ready(epoch + 250us));
  assert(backoff.next_delay_us() == 500);

  auto retry_at = epoch + 250us;
  for (u32 expected : {1000u, 2000u, 4000u, 8000u, 10000u, 10000u}) {
    const u32 scheduled_delay = backoff.next_delay_us();
    backoff.schedule(retry_at);
    assert(!backoff.ready(
      retry_at + std::chrono::microseconds(scheduled_delay - 1)));
    retry_at += std::chrono::microseconds(scheduled_delay);
    assert(backoff.ready(retry_at));
    assert(backoff.next_delay_us() == expected);
  }

  // Execute resolution starts a distinct ordered release phase. Its first
  // retry must begin at the minimum delay rather than inheriting Execute's
  // saturated-window backoff.
  backoff.reset();
  assert(backoff.ready(retry_at));
  assert(backoff.next_delay_us() == 250);
}

void test_peer_attempt_timeout_leaves_public_retry_headroom() {
  using namespace std::chrono_literals;
  assert(stage1_peer_attempt_timeout(30'000) == 500ms);
  assert(stage1_peer_attempt_timeout(500) == 125ms);
  assert(stage1_peer_attempt_timeout(73) == 18ms);
  assert(stage1_peer_attempt_timeout(0) == 1ms);
}

void test_retry_response_preserves_execute_and_control_tokens() {
  constexpr u32 response_source = 4;
  constexpr u64 request_id = 1234;

  std::vector<byte_t> execute_request(stage1_execute_request_bytes(2));
  auto* execute_header = reinterpret_cast<PeerRpcHeader*>(
    execute_request.data());
  execute_header->magic = kPeerRpcMagic;
  execute_header->version = kPeerRpcVersion;
  execute_header->type = static_cast<u32>(
    PeerRpcType::stage1_execute_request);
  execute_header->source_shard = 2;
  execute_header->item_count = 2;
  execute_header->request_id = request_id;
  Stage1ExecuteItem* execute_inputs = stage1_execute_items(
    execute_request.data());
  execute_inputs[0].client_batch_id = 77;
  execute_inputs[0].source_client = 8;
  execute_inputs[0].item_index = 3;
  execute_inputs[1].client_batch_id = 78;
  execute_inputs[1].source_client = 9;
  execute_inputs[1].item_index = 4;
  std::vector<byte_t> execute_response(stage1_execute_response_bytes(2));
  assert(write_stage1_retry_response(
    response_source, *execute_header,
    span<const byte_t>{execute_request}, span<byte_t>{execute_response}));
  const auto* execute_response_header =
    reinterpret_cast<const PeerRpcHeader*>(execute_response.data());
  assert(execute_response_header->source_shard == response_source);
  assert(execute_response_header->request_id == request_id);
  assert(execute_response_header->status ==
         static_cast<u32>(InsertStatus::overloaded));
  const Stage1ExecuteResult* execute_outputs = stage1_execute_results(
    execute_response.data());
  for (u32 index = 0; index < 2; ++index) {
    assert(execute_outputs[index].client_batch_id ==
           execute_inputs[index].client_batch_id);
    assert(execute_outputs[index].source_client ==
           execute_inputs[index].source_client);
    assert(execute_outputs[index].item_index ==
           execute_inputs[index].item_index);
    assert(execute_outputs[index].maintenance_sequence == 0);
    assert(execute_outputs[index].status ==
           static_cast<u32>(MutationStatus::retry));
  }

  std::vector<byte_t> control_request(stage1_arm_request_bytes(2));
  auto* control_header = reinterpret_cast<PeerRpcHeader*>(
    control_request.data());
  control_header->magic = kPeerRpcMagic;
  control_header->version = kPeerRpcVersion;
  control_header->type = static_cast<u32>(PeerRpcType::stage1_arm_request);
  control_header->source_shard = 2;
  control_header->item_count = 2;
  control_header->request_id = request_id + 1;
  Stage1ArmItem* control_inputs = stage1_arm_items(control_request.data());
  control_inputs[0] = arm_item(5, Stage1ArmAction::release);
  control_inputs[1] = arm_item(6, Stage1ArmAction::release);
  std::vector<byte_t> control_response(stage1_arm_response_bytes(2));
  assert(write_stage1_retry_response(
    response_source, *control_header,
    span<const byte_t>{control_request}, span<byte_t>{control_response}));
  const auto* control_response_header =
    reinterpret_cast<const PeerRpcHeader*>(control_response.data());
  assert(control_response_header->status ==
         static_cast<u32>(InsertStatus::ok));
  const Stage1ArmResult* control_outputs = stage1_arm_results(
    control_response.data());
  for (u32 index = 0; index < 2; ++index) {
    assert(control_outputs[index].token.source_client ==
           control_inputs[index].token.source_client);
    assert(control_outputs[index].token.item_index ==
           control_inputs[index].token.item_index);
    assert(control_outputs[index].token.client_batch_id ==
           control_inputs[index].token.client_batch_id);
    assert(control_outputs[index].target_raw ==
           control_inputs[index].target_raw);
    assert(control_outputs[index].maintenance_sequence == 0);
    assert(control_outputs[index].status ==
           static_cast<u32>(MutationStatus::retry));
  }
  assert(classify_stage1_control_response(
           span<const Stage1ArmItem>{control_inputs, 2},
           span<const Stage1ArmResult>{control_outputs, 2}) ==
         Stage1ControlResponseDisposition::retry);
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

Stage1ExecuteItem fused_execute_item(u32 item_index) {
  return Stage1ExecuteItem{
    .client_batch_id = 99,
    .old_raw = 0,
    .initial_placement_version = 4,
    .source_client = 7,
    .item_index = item_index,
    .id = 100 + item_index,
    .generation = 3,
    .kind = static_cast<u32>(MutationKind::insert),
    .authority_shard = 2,
  };
}

Stage1ExecuteResult execute_result(
    const Stage1ExecuteItem& input, MutationStatus status, u32 home) {
  return Stage1ExecuteResult{
    .client_batch_id = input.client_batch_id,
    .target_raw = status == MutationStatus::ok
      ? RemotePtr{home, 0x1000 + input.item_index * 0x100, 1}.raw_address
      : 0,
    .maintenance_sequence = status == MutationStatus::ok
      ? 77 + input.item_index : 0,
    .source_client = input.source_client,
    .item_index = input.item_index,
    .status = static_cast<u32>(status),
  };
}

void test_execute_mixed_progress_compacts_only_retry_tokens() {
  constexpr u32 home = 4;
  const std::array inputs{
    fused_execute_item(3),
    fused_execute_item(7),
    fused_execute_item(11),
    fused_execute_item(13),
  };
  // Model four ready tokens with only two completion credits. The receiver's
  // per-token admission returns two durable sequences, one transient retry,
  // and preserves an independent terminal failure instead of rolling back the
  // successful prefix.
  const std::array outputs{
    execute_result(inputs[0], MutationStatus::ok, home),
    execute_result(inputs[1], MutationStatus::ok, home),
    execute_result(inputs[2], MutationStatus::retry, home),
    execute_result(inputs[3], MutationStatus::failed, home),
  };
  std::vector<u32> resolved_slots;
  std::vector<u32> retry_slots;
  assert(partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{inputs.data(), inputs.size()},
    span<const Stage1ExecuteResult>{outputs.data(), outputs.size()}, home,
    resolved_slots, retry_slots));
  assert((resolved_slots == std::vector<u32>{0, 1, 3}));
  assert((retry_slots == std::vector<u32>{2}));

  // The already-runnable token is released immediately and is absent from the
  // retry wave. Its token/target fence is reproduced exactly in the ordered
  // release record.
  Stage1ArmItem release;
  assert(make_fused_stage1_release_item(inputs[0], outputs[0], release));
  assert(release.token.source_client == inputs[0].source_client);
  assert(release.token.item_index == inputs[0].item_index);
  assert(release.token.client_batch_id == inputs[0].client_batch_id);
  assert(release.target_raw == outputs[0].target_raw);
  assert(static_cast<Stage1ArmAction>(release.action) ==
         Stage1ArmAction::release);
  assert(make_fused_stage1_release_item(inputs[1], outputs[1], release));
  assert(!make_fused_stage1_release_item(inputs[2], outputs[2], release));

  // Replaying the compact subset preserves the semantic token even though it
  // uses a fresh transport request ID. Resolution cannot cause tokens 3, 7,
  // or 13 to be recomputed or committed twice because none is in this wave.
  const std::array replay_inputs{inputs[2]};
  const std::array replay_outputs{
    execute_result(replay_inputs[0], MutationStatus::ok, home),
  };
  assert(partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{replay_inputs.data(), replay_inputs.size()},
    span<const Stage1ExecuteResult>{
      replay_outputs.data(), replay_outputs.size()}, home,
    resolved_slots, retry_slots));
  assert((resolved_slots == std::vector<u32>{0}));
  assert(retry_slots.empty());
  assert(replay_outputs[0].client_batch_id == inputs[2].client_batch_id);
  assert(replay_outputs[0].source_client == inputs[2].source_client);
  assert(replay_outputs[0].item_index == inputs[2].item_index);

  // A terminal ARM result may retain its same-home provisional target for
  // authority abort, but may never redirect recovery to another home.
  auto terminal_with_target = outputs;
  terminal_with_target[3].target_raw =
    RemotePtr{home, 0x9000, 1}.raw_address;
  assert(partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{inputs.data(), inputs.size()},
    span<const Stage1ExecuteResult>{
      terminal_with_target.data(), terminal_with_target.size()}, home,
    resolved_slots, retry_slots));
  terminal_with_target[3].target_raw =
    RemotePtr{home + 1, 0x9000, 1}.raw_address;
  assert(!partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{inputs.data(), inputs.size()},
    span<const Stage1ExecuteResult>{
      terminal_with_target.data(), terminal_with_target.size()}, home,
    resolved_slots, retry_slots));

  auto retry_with_target = replay_outputs;
  retry_with_target[0].status = static_cast<u32>(MutationStatus::retry);
  retry_with_target[0].maintenance_sequence = 0;
  retry_with_target[0].target_raw =
    RemotePtr{home, 0xa000, 1}.raw_address;
  assert(partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{replay_inputs.data(), replay_inputs.size()},
    span<const Stage1ExecuteResult>{
      retry_with_target.data(), retry_with_target.size()}, home,
    resolved_slots, retry_slots));
  assert(resolved_slots.empty());
  assert((retry_slots == std::vector<u32>{0}));
  retry_with_target[0].target_raw =
    RemotePtr{home + 1, 0xa000, 1}.raw_address;
  assert(!partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{replay_inputs.data(), replay_inputs.size()},
    span<const Stage1ExecuteResult>{
      retry_with_target.data(), retry_with_target.size()}, home,
    resolved_slots, retry_slots));

  auto malformed = replay_outputs;
  malformed[0].item_index = inputs[0].item_index;
  assert(!partition_stage1_execute_response(
    span<const Stage1ExecuteItem>{replay_inputs.data(), replay_inputs.size()},
    span<const Stage1ExecuteResult>{malformed.data(), malformed.size()}, home,
    resolved_slots, retry_slots));
  assert(resolved_slots.empty());
  assert(retry_slots.empty());

  auto duplicated = inputs;
  duplicated[3].client_batch_id = duplicated[0].client_batch_id;
  duplicated[3].source_client = duplicated[0].source_client;
  duplicated[3].item_index = duplicated[0].item_index;
  assert(!stage1_execute_tokens_unique(
    span<const Stage1ExecuteItem>{duplicated.data(), duplicated.size()}));
  assert(stage1_execute_tokens_unique(
    span<const Stage1ExecuteItem>{inputs.data(), inputs.size()}));
}

}  // namespace

int main() {
  test_stage2_home_queue_cannot_starve_behind_stage1();
  test_atomic_home_arm_response();
  test_release_is_an_idempotent_ordered_watermark();
  test_structural_corruption_never_becomes_retry();
  test_each_home_advances_independently();
  test_stage1_retry_backoff_is_bounded_and_phase_local();
  test_peer_attempt_timeout_leaves_public_retry_headroom();
  test_retry_response_preserves_execute_and_control_tokens();
  test_home_commit_releases_credit_before_next_home_arm();
  test_fused_execute_requires_fresh_insert_and_runnable_fence();
  test_execute_mixed_progress_compacts_only_retry_tokens();
}
