#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// Stage1 needs only a small incoming-edge certificate to become immediately
// query reachable.  Replicating the provisional edge to every one of the R
// outgoing neighbors turns a local cluster into an R-fold protected-slot
// hotspot; two independently placed bridges tolerate one concurrent parent
// retirement without making that cost grow with R.
inline constexpr u32 kStage1ReachabilityBridgeGoal = 2;

enum class Stage1BridgeInstallDisposition : u8 {
  rejected,
  busy,
  installed,
};

// Dynamic records are allocated at a fixed stride.  Dividing the tagged
// pointer's byte offset by that stride therefore advances by one for adjacent
// allocations, even when the dynamic arena itself is not stride-aligned.
// Incarnation participates as well so reuse of the same physical slot does
// not repeatedly select the same parents.  Unlike choosing targets[0], this
// gives identical hot neighbor sets a deterministic round-robin start.
inline size_t stage1_bridge_rotation(const RemotePtr candidate,
                                     const size_t target_count,
                                     const size_t dynamic_record_bytes) {
  if (target_count == 0) return 0;
  const u64 stride = std::max<u64>(1, dynamic_record_bytes);
  const u64 slot_ordinal = candidate.byte_offset() / stride;
  const u64 incarnation_phase =
    static_cast<u64>(candidate.incarnation()) * 0x9e3779b97f4a7c15ULL;
  const u64 shard_phase =
    static_cast<u64>(candidate.memory_node()) * 0xbf58476d1ce4e5b9ULL;
  return static_cast<size_t>(
    (slot_ordinal + incarnation_phase + shard_phase) % target_count);
}

// Scans the O(R) candidate list once, beginning at the rotating offset, and
// records only callbacks that actually establish (or idempotently observe)
// the provisional backlink. Since accepted is capped at two, duplicate ACKs
// are suppressed with constant bounded work and no per-insert hash allocation.
template <class TryInstall>
vec<RemotePtr> select_stage1_reachability_bridges(
    const RemotePtr candidate,
    const span<const RemotePtr> targets,
    const size_t dynamic_record_bytes,
    TryInstall&& try_install) {
  vec<RemotePtr> accepted;
  if (candidate.is_null() || targets.empty()) {
    return accepted;
  }
  accepted.reserve(std::min<size_t>(
    kStage1ReachabilityBridgeGoal, targets.size()));

  const size_t start = stage1_bridge_rotation(
    candidate, targets.size(), dynamic_record_bytes);
  for (size_t step = 0; step < targets.size(); ++step) {
    const RemotePtr target = targets[(start + step) % targets.size()];
    if (target.is_null() || target == candidate ||
        std::find(accepted.begin(), accepted.end(), target) !=
          accepted.end()) {
      continue;
    }
    if (!try_install(target)) continue;
    accepted.push_back(target);
    if (accepted.size() == kStage1ReachabilityBridgeGoal) break;
  }
  return accepted;
}

// Retry a complete candidate sweep only when at least one target still has a
// live but busy identity. Permanent rejection cannot improve by spinning, and
// one installed bridge is already a valid bounded reachability certificate.
// wait_for_retry() is the caller's shutdown/backoff boundary and returns false
// when no further sweep should be attempted.
template <class TryInstall, class WaitForRetry>
vec<RemotePtr> select_stage1_reachability_bridges_retry_busy(
    const RemotePtr candidate,
    const span<const RemotePtr> targets,
    const size_t dynamic_record_bytes,
    TryInstall&& try_install,
    WaitForRetry&& wait_for_retry) {
  for (;;) {
    bool saw_busy = false;
    vec<RemotePtr> accepted = select_stage1_reachability_bridges(
      candidate, targets, dynamic_record_bytes,
      [&](const RemotePtr target) {
        const Stage1BridgeInstallDisposition disposition =
          try_install(target);
        saw_busy = saw_busy ||
          disposition == Stage1BridgeInstallDisposition::busy;
        return disposition == Stage1BridgeInstallDisposition::installed;
      });
    if (!accepted.empty() || !saw_busy || !wait_for_retry()) {
      return accepted;
    }
  }
}

}  // namespace memory_node_storage_owner_index_detail
