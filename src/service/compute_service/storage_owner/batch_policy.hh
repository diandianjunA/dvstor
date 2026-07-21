#pragma once

#include <algorithm>
#include <cstdint>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool saturated{};
  bool tail_escape{};
  u32 take{};
};

// A synchronous caller population forms a closed queueing loop. If every RPC
// slot consumes the first item it sees, N callers are permanently fragmented
// across rpc_depth tiny requests (N / rpc_depth items each). This policy keeps
// isolated writes immediate, then latches a saturated epoch once every lane is
// busy or a full batch is already ready. During that epoch only full batches
// open additional lanes; the last active lane is the progress escape for a
// finite tail. No timer, scheduler yield, or dataset-specific threshold is
// involved, and the CQ progress thread never waits for a producer.
inline StorageOwnerBatchDecision decide_storage_owner_batch(
    bool saturated,
    u32 ready_tasks,
    u32 active_rpcs,
    u32 free_rpc_slots,
    u32 pending_producers,
    u32 batch_max) {
  if (batch_max == 0) return {};
  if (!saturated &&
      (ready_tasks >= batch_max ||
       (free_rpc_slots == 0 && ready_tasks != 0))) {
    saturated = true;
  }
  if (saturated && active_rpcs == 0 && ready_tasks == 0 &&
      pending_producers == 0) {
    saturated = false;
  }
  if (free_rpc_slots == 0 || ready_tasks == 0) {
    return {.saturated = saturated};
  }
  // A producer increments pending_producers before publishing its queue
  // entry.  When no RPC is active, wait for those already-announced entries
  // to become visible instead of prematurely sending the currently visible
  // tail.  Once every producer closes its announcement, the tail escape below
  // guarantees progress without a timer.
  if (saturated && ready_tasks < batch_max &&
      (active_rpcs != 0 || pending_producers != 0)) {
    return {.saturated = true};
  }
  const bool tail_escape = saturated && active_rpcs == 0 &&
    ready_tasks < batch_max;
  return {
    .saturated = saturated,
    .tail_escape = tail_escape,
    .take = std::min(ready_tasks, batch_max),
  };
}

}  // namespace compute_service_detail
