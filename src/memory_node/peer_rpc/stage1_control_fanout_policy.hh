#pragma once

#include <algorithm>
#include <chrono>
#include <limits>

#include "service/storage_owner_protocol.hh"

namespace memory_node_peer_rpc_detail {

// Stage1 publication and Stage2 home expansion share a CPU pool. Strictly
// preferring Stage1 creates a dependency cycle under sustained load: every
// newly published update adds Stage2 work, but that work runs only after the
// Stage1 queue becomes empty, so durable completion credit can never return.
// Bound the Stage1 dequeue burst while keeping both queues work-conserving.
// This is scheduler fairness only; it neither drops work nor changes either
// stage's acknowledgement boundary.
// A Stage2-home request performs the expansion/scoring work that dominates
// insert service time.  Letting every shared worker dequeue four new Stage1
// publications before one Stage2 request amplifies acknowledged maintenance
// debt under a sustained insert stream.  Alternate when both queues are
// runnable.  Stage1 still has a strict progress guarantee and consumes the
// whole pool whenever Stage2 is empty.
constexpr u64 kStage2HomeBorrowAgeNs = 250'000;
constexpr u64 kStage1UrgentAgeNs = 1'000'000;

inline bool stage1_worker_has_eligible_task(
    bool stage1_available,
    bool stage2_home_available) noexcept {
  return stage1_available || stage2_home_available;
}

// Keep the CQ response router and the Stage2 caller on one definition of the
// home-search response family.  A response type omitted from the router is
// otherwise silently reposted, leaving an already completed home RPC pending
// until its caller retries forever.
inline bool is_stage2_home_response(
    service::storage_owner::PeerRpcType type) noexcept {
  using service::storage_owner::PeerRpcType;
  return type == PeerRpcType::stage2_expand_score_response ||
    type == PeerRpcType::stage2_score_many_response;
}

inline bool dequeue_stage2_home_first(
    bool stage1_available,
    bool stage2_home_available,
    u64 stage1_oldest_age_ns,
    u64 stage2_oldest_age_ns) noexcept {
  if (!stage2_home_available) return false;
  if (!stage1_available) return true;
  // Foreground publication is urgent above 1 ms. Otherwise Stage2 may borrow
  // a shared lane only after its oldest request has waited 250 us. A
  // dedicated Stage2 lane remains active independently, so neither class can
  // starve and idle lanes stay work-conserving.
  if (stage1_oldest_age_ns >= kStage1UrgentAgeNs) return false;
  return stage2_oldest_age_ns >= kStage2HomeBorrowAgeNs;
}

// Build a fully self-describing transient Stage1 response directly from the
// validated request bytes. CQ progress uses this when a same-ID request is
// already executing or bounded queue admission fails. An explicit response
// lets the authority back off immediately instead of waiting for an attempt
// timeout, while all semantic fields remain available for token validation.
inline bool write_stage1_retry_response(
    u32 response_source_shard,
    const service::storage_owner::PeerRpcHeader& request_header,
    span<const byte_t> request,
    span<byte_t> response) noexcept {
  using namespace service::storage_owner;
  if (request_header.item_count == 0 || request_header.request_id == 0 ||
      request_header.reserved != 0) {
    return false;
  }
  const auto request_type = static_cast<PeerRpcType>(request_header.type);
  const bool execute = request_type == PeerRpcType::stage1_execute_request;
  const bool control = request_type == PeerRpcType::stage1_arm_request;
  if (!execute && !control) return false;

  const size_t expected_request_bytes = execute
    ? stage1_execute_request_bytes(request_header.item_count)
    : stage1_arm_request_bytes(request_header.item_count);
  const size_t expected_response_bytes = execute
    ? stage1_execute_response_bytes(request_header.item_count)
    : stage1_arm_response_bytes(request_header.item_count);
  if (request.size() != expected_request_bytes ||
      response.size() != expected_response_bytes) {
    return false;
  }

  std::fill(response.begin(), response.end(), byte_t{});
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(execute
    ? PeerRpcType::stage1_execute_response
    : PeerRpcType::stage1_arm_response);
  response_header->source_shard = response_source_shard;
  response_header->item_count = request_header.item_count;
  response_header->request_id = request_header.request_id;
  // Execute's existing aggregate convention marks any per-item retry as
  // overloaded. Stage1 control keeps a consumable envelope and carries retry
  // solely per item so the strict control parser can validate every token.
  response_header->status = static_cast<u32>(execute
    ? InsertStatus::overloaded : InsertStatus::ok);

  if (execute) {
    const Stage1ExecuteItem* inputs = stage1_execute_items(request.data());
    Stage1ExecuteResult* outputs = stage1_execute_results(response.data());
    for (u32 index = 0; index < request_header.item_count; ++index) {
      outputs[index].client_batch_id = inputs[index].client_batch_id;
      outputs[index].source_client = inputs[index].source_client;
      outputs[index].item_index = inputs[index].item_index;
      outputs[index].status = static_cast<u32>(MutationStatus::retry);
    }
  } else {
    const Stage1ArmItem* inputs = stage1_arm_items(request.data());
    Stage1ArmResult* outputs = stage1_arm_results(response.data());
    for (u32 index = 0; index < request_header.item_count; ++index) {
      outputs[index].token = inputs[index].token;
      outputs[index].target_raw = inputs[index].target_raw;
      outputs[index].status = static_cast<u32>(MutationStatus::retry);
    }
  }
  return true;
}

// The public mutation timeout bounds the whole authority transaction.  A
// peer Execute/control response is only one idempotent transport attempt
// inside that transaction, so using the same deadline for both layers leaves
// no time to recover a request that was deliberately dropped when a bounded
// peer queue was transiently full.  Keep attempts long enough for a loaded
// physical-home search, but always leave multiple retry opportunities before
// the public caller's deadline.
inline std::chrono::milliseconds stage1_peer_attempt_timeout(
    u32 public_timeout_ms) noexcept {
  constexpr u32 kMaximumAttemptMs = 500;
  const u32 retry_headroom_ms = std::max<u32>(1, public_timeout_ms / 4);
  return std::chrono::milliseconds(
    std::min(retry_headroom_ms, kMaximumAttemptMs));
}

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

inline bool stage1_execute_tokens_unique(
    span<const service::storage_owner::Stage1ExecuteItem> items) noexcept {
  for (size_t item = 0; item < items.size(); ++item) {
    for (size_t previous = 0; previous < item; ++previous) {
      if (items[item].client_batch_id == items[previous].client_batch_id &&
          items[item].source_client == items[previous].source_client &&
          items[item].item_index == items[previous].item_index) {
        return false;
      }
    }
  }
  return true;
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

enum class Stage1ExecuteItemDisposition : u8 {
  resolved,
  retry,
  malformed,
};

// Execute is a physical-home batch only for transport and queue efficiency;
// each public mutation token remains an independent authority transaction.
// Validate one result without letting a transient sibling hide an already
// runnable token.  This is also the single definition used while compacting a
// retry wave, so a malformed token can never be silently moved to a new RPC.
inline Stage1ExecuteItemDisposition classify_stage1_execute_item(
    const service::storage_owner::Stage1ExecuteItem& input,
    const service::storage_owner::Stage1ExecuteResult& output,
    u32 physical_home) noexcept {
  using namespace service::storage_owner;
  const bool same_token = output.client_batch_id == input.client_batch_id &&
    output.source_client == input.source_client &&
    output.item_index == input.item_index;
  if (!same_token || output.reserved != 0 ||
      output.status > static_cast<u32>(MutationStatus::retry) ||
      !stage1_execute_success_has_expected_fence(input, output)) {
    return Stage1ExecuteItemDisposition::malformed;
  }
  if (output.target_raw != 0 &&
      RemotePtr{output.target_raw}.memory_node() != physical_home) {
    // A prepare failure has no target, while a later ARM failure/retry retains
    // the same-home provisional target so the authority can abort it. Reject
    // pointers that could make that recovery cross physical ownership.
    return Stage1ExecuteItemDisposition::malformed;
  }
  if (output.status == static_cast<u32>(MutationStatus::retry)) {
    return Stage1ExecuteItemDisposition::retry;
  }
  if (output.status == static_cast<u32>(MutationStatus::ok)) {
    const RemotePtr target{output.target_raw};
    if (target.is_null() || target.memory_node() != physical_home) {
      return Stage1ExecuteItemDisposition::malformed;
    }
  }
  return Stage1ExecuteItemDisposition::resolved;
}

// Partition the current wire wave by slot. Resolved slots may be committed
// immediately; retry slots are the only records copied into the next wire
// wave. Original semantic tokens are carried by the input records and never
// rewritten. The caller must use a fresh transport request ID for the compact
// wave so a late response for the previous item_count cannot alias it.
inline bool partition_stage1_execute_response(
    span<const service::storage_owner::Stage1ExecuteItem> inputs,
    span<const service::storage_owner::Stage1ExecuteResult> outputs,
    u32 physical_home,
    vec<u32>& resolved_slots,
    vec<u32>& retry_slots) {
  resolved_slots.clear();
  retry_slots.clear();
  if (inputs.empty() || inputs.size() != outputs.size() ||
      inputs.size() > std::numeric_limits<u32>::max()) {
    return false;
  }
  resolved_slots.reserve(inputs.size());
  retry_slots.reserve(inputs.size());
  for (u32 slot = 0; slot < static_cast<u32>(inputs.size()); ++slot) {
    switch (classify_stage1_execute_item(
              inputs[slot], outputs[slot], physical_home)) {
      case Stage1ExecuteItemDisposition::resolved:
        resolved_slots.push_back(slot);
        break;
      case Stage1ExecuteItemDisposition::retry:
        retry_slots.push_back(slot);
        break;
      case Stage1ExecuteItemDisposition::malformed:
        resolved_slots.clear();
        retry_slots.clear();
        return false;
    }
  }
  return true;
}

inline bool make_fused_stage1_release_item(
    const service::storage_owner::Stage1ExecuteItem& input,
    const service::storage_owner::Stage1ExecuteResult& output,
    service::storage_owner::Stage1ArmItem& release) noexcept {
  using namespace service::storage_owner;
  if (!stage1_execute_uses_fused_arm(input) ||
      output.status != static_cast<u32>(MutationStatus::ok) ||
      !stage1_execute_success_has_expected_fence(input, output)) {
    return false;
  }
  release = Stage1ArmItem{
    .token = {
      .source_client = input.source_client,
      .item_index = input.item_index,
      .client_batch_id = input.client_batch_id,
    },
    .target_raw = output.target_raw,
    .initial_placement_version = input.initial_placement_version,
    .id = input.id,
    .generation = input.generation,
    .action = static_cast<u32>(Stage1ArmAction::release),
  };
  return true;
}

enum class Stage1ControlResponseDisposition : u8 {
  resolved,
  retry,
  malformed,
};

inline Stage1ControlResponseDisposition classify_stage1_control_item(
    const service::storage_owner::Stage1ArmItem& input,
    const service::storage_owner::Stage1ArmResult& output) noexcept {
  using namespace service::storage_owner;
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
    return Stage1ControlResponseDisposition::retry;
  }
  if (output.status != static_cast<u32>(MutationStatus::ok) ||
      (action == Stage1ArmAction::arm &&
       output.maintenance_sequence == 0)) {
    return Stage1ControlResponseDisposition::malformed;
  }
  return Stage1ControlResponseDisposition::resolved;
}

// Translate the receiver-local fused ARM attempt back into the enclosing
// Execute result without turning an uncertain handoff into a semantic
// rejection.  Once prepare has succeeded, an absent or malformed ARM result
// does not prove that admission failed: the task may already own a durable
// maintenance sequence and the exact-token replay is the only operation that
// can recover that sequence safely.  arm_local_stage1_items() reports a
// separate structural-validity bit; only that explicit identity/generation
// conflict is terminal.  Ordinary capacity, duplicate-in-progress, and
// shutdown admission remain retryable.
inline service::storage_owner::MutationStatus
propagate_fused_stage1_arm_result(
    bool structurally_valid,
    const service::storage_owner::Stage1ArmItem& input,
    const service::storage_owner::Stage1ArmResult* output,
    service::storage_owner::Stage1ExecuteResult& execute) noexcept {
  using namespace service::storage_owner;
  execute.maintenance_sequence = 0;
  if (!structurally_valid) {
    execute.status = static_cast<u32>(MutationStatus::failed);
    return MutationStatus::failed;
  }
  if (output == nullptr) {
    execute.status = static_cast<u32>(MutationStatus::retry);
    return MutationStatus::retry;
  }
  switch (classify_stage1_control_item(input, *output)) {
    case Stage1ControlResponseDisposition::resolved:
      execute.maintenance_sequence = output->maintenance_sequence;
      execute.status = static_cast<u32>(MutationStatus::ok);
      return MutationStatus::ok;
    case Stage1ControlResponseDisposition::retry:
      execute.status = static_cast<u32>(MutationStatus::retry);
      return MutationStatus::retry;
    case Stage1ControlResponseDisposition::malformed:
      // The semantic receipt is still keyed by input.token.  Discard the
      // untrusted result and replay that exact token instead of aborting a
      // task whose ARM may already have linearized.
      execute.status = static_cast<u32>(MutationStatus::retry);
      return MutationStatus::retry;
  }
  execute.status = static_cast<u32>(MutationStatus::retry);
  return MutationStatus::retry;
}

inline bool partition_stage1_control_response(
    span<const service::storage_owner::Stage1ArmItem> inputs,
    span<const service::storage_owner::Stage1ArmResult> outputs,
    vec<u32>& resolved_slots,
    vec<u32>& retry_slots) {
  resolved_slots.clear();
  retry_slots.clear();
  if (inputs.empty() || inputs.size() != outputs.size() ||
      inputs.size() > std::numeric_limits<u32>::max()) {
    return false;
  }
  resolved_slots.reserve(inputs.size());
  retry_slots.reserve(inputs.size());
  for (u32 slot = 0; slot < static_cast<u32>(inputs.size()); ++slot) {
    switch (classify_stage1_control_item(inputs[slot], outputs[slot])) {
      case Stage1ControlResponseDisposition::resolved:
        resolved_slots.push_back(slot);
        break;
      case Stage1ControlResponseDisposition::retry:
        retry_slots.push_back(slot);
        break;
      case Stage1ControlResponseDisposition::malformed:
        resolved_slots.clear();
        retry_slots.clear();
        return false;
    }
  }
  return true;
}

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
    const auto disposition = classify_stage1_control_item(
      inputs[index], outputs[index]);
    if (disposition == Stage1ControlResponseDisposition::malformed) {
      return Stage1ControlResponseDisposition::malformed;
    }
    retry |= disposition == Stage1ControlResponseDisposition::retry;
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

// A bounded per-home retry delay keeps a temporarily full Stage2 completion
// window from turning an Execute `retry` response into a tight request storm.
// The semantic token and transport request ID remain unchanged across retries;
// only a successful phase transition resets the delay.
class Stage1HomeRetryBackoff {
public:
  using Clock = std::chrono::steady_clock;

  // A full Stage2 completion window normally advances on the millisecond
  // scale.  Retrying every 2 ms at saturation multiplied peer Stage1 traffic
  // by nearly 6x in the 100M mixed run and stole CPU from the work that would
  // release the window.  Keep the first retry responsive, then converge to the
  // existing 10 ms Stage2 batching horizon instead of polling it aggressively.
  static constexpr u32 kInitialDelayUs = 250;
  static constexpr u32 kMaximumDelayUs = 10'000;

  [[nodiscard]] bool ready(Clock::time_point now) const noexcept {
    return retry_not_before_ == Clock::time_point{} ||
      now >= retry_not_before_;
  }

  void schedule(Clock::time_point now) noexcept {
    retry_not_before_ = now + std::chrono::microseconds(next_delay_us_);
    next_delay_us_ = std::min<u32>(
      next_delay_us_ * 2, kMaximumDelayUs);
  }

  void reset() noexcept {
    retry_not_before_ = {};
    next_delay_us_ = kInitialDelayUs;
  }

  [[nodiscard]] u32 next_delay_us() const noexcept {
    return next_delay_us_;
  }

private:
  Clock::time_point retry_not_before_{};
  u32 next_delay_us_{kInitialDelayUs};
};

}  // namespace memory_node_peer_rpc_detail
