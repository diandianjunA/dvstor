#pragma once

#include "service/storage_owner_protocol.hh"

namespace memory_node_peer_rpc_detail {

inline bool stage1_execute_uses_fused_arm(
    const service::storage_owner::Stage1ExecuteItem& item) noexcept {
  return item.initial_placement_version != 0;
}

// Fresh inserts have no old-generation cleanup dependency.  They may
// therefore reserve their bounded Stage2 sequence in the same physical-home
// transaction that prepares the local graph node.  Upserts must keep the
// standalone cleanup -> arm ordering even if a malformed sender supplies a
// non-zero placement version.
inline bool valid_fused_stage1_execute_item(
    const service::storage_owner::Stage1ExecuteItem& item) noexcept {
  using namespace service::storage_owner;
  return stage1_execute_uses_fused_arm(item) && item.old_raw == 0 &&
    static_cast<MutationKind>(item.kind) == MutationKind::insert;
}

// A successful legacy prepare owns no maintenance sequence.  Conversely, a
// successful fused prepare is authority-committable only when the response
// proves that the physical home already owns a runnable bounded Stage2 task.
// Retry/terminal prepare results are checked by their caller and intentionally
// carry no such proof.
inline bool stage1_execute_success_has_expected_fence(
    const service::storage_owner::Stage1ExecuteItem& input,
    const service::storage_owner::Stage1ExecuteResult& output) noexcept {
  using namespace service::storage_owner;
  if (output.status != static_cast<u32>(MutationStatus::ok)) {
    return output.maintenance_sequence == 0;
  }
  if (stage1_execute_uses_fused_arm(input)) {
    return valid_fused_stage1_execute_item(input) &&
      output.maintenance_sequence != 0;
  }
  return output.maintenance_sequence == 0;
}

// When one prepare in a fused home batch is transiently unready, already
// successful prepares must wait for the batch-atomic ARM retry. A terminal
// prepare failure is not transient and must remain visible; rewriting it to
// retry could turn an allocation or graph-publication failure into an
// unbounded retry loop.
inline void defer_fused_stage1_success_for_atomic_retry(
    service::storage_owner::Stage1ExecuteResult& output) noexcept {
  using namespace service::storage_owner;
  if (output.status != static_cast<u32>(MutationStatus::ok)) return;
  output.maintenance_sequence = 0;
  output.status = static_cast<u32>(MutationStatus::retry);
}

enum class Stage1ControlResponseDisposition : u8 {
  resolved,
  retry,
  malformed,
};

// Classify one physical home's atomic Stage1 control response. Only the
// explicit internal `retry` status is transient. `failed`, not-found, and
// already-* statuses are terminal protocol/identity failures and must never be
// hidden by an unbounded retry loop because they invalidate the token/target
// proof consumed by the authority commit.
inline Stage1ControlResponseDisposition classify_stage1_control_response(
    span<const service::storage_owner::Stage1ArmItem> inputs,
    span<const service::storage_owner::Stage1ArmResult> outputs) {
  using namespace service::storage_owner;
  if (inputs.empty() || inputs.size() != outputs.size()) {
    return Stage1ControlResponseDisposition::malformed;
  }

  bool retry = false;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const Stage1ArmItem& input = inputs[index];
    const Stage1ArmResult& output = outputs[index];
    const auto action = static_cast<Stage1ArmAction>(input.action);
    const bool known_action = action == Stage1ArmAction::arm ||
      action == Stage1ArmAction::abort ||
      action == Stage1ArmAction::release;
    const bool same_token =
      output.token.source_client == input.token.source_client &&
      output.token.item_index == input.token.item_index &&
      output.token.client_batch_id == input.token.client_batch_id;
    if (!known_action || !same_token || output.reserved != 0 ||
        output.status > static_cast<u32>(MutationStatus::retry) ||
        (action == Stage1ArmAction::arm &&
         output.target_raw != input.target_raw)) {
      return Stage1ControlResponseDisposition::malformed;
    }

    if (output.status == static_cast<u32>(MutationStatus::retry)) {
      retry = true;
      continue;
    }
    if (output.status != static_cast<u32>(MutationStatus::ok)) {
      return Stage1ControlResponseDisposition::malformed;
    }
    if (action == Stage1ArmAction::arm &&
        output.maintenance_sequence == 0) {
      return Stage1ControlResponseDisposition::malformed;
    }
  }
  return retry ? Stage1ControlResponseDisposition::retry
               : Stage1ControlResponseDisposition::resolved;
}

// Per-home progress is intentionally independent.  A retry transitions only
// its own home back to ready; a resolved home can never be reopened by a late
// timeout.  The foreground coordinator uses this property to commit an arm
// response immediately instead of building a cross-home hold-and-wait barrier.
class Stage1ControlHomeProgress {
public:
  [[nodiscard]] bool needs_post() const noexcept {
    return !posted_ && !resolved_;
  }
  [[nodiscard]] bool posted() const noexcept { return posted_; }
  [[nodiscard]] bool resolved() const noexcept { return resolved_; }

  void mark_posted() noexcept {
    if (!resolved_) posted_ = true;
  }

  void mark_retry() noexcept {
    if (!resolved_) posted_ = false;
  }

  [[nodiscard]] bool mark_resolved() noexcept {
    if (resolved_) return false;
    posted_ = false;
    resolved_ = true;
    return true;
  }

private:
  bool posted_{};
  bool resolved_{};
};

}  // namespace memory_node_peer_rpc_detail
