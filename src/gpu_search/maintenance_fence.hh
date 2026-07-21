#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "gpu_search/types.hh"

namespace gpu_search::maintenance_fence {

// Capture a fixed per-shard prefix of every maintenance sequence that had
// already been reserved when the control blocks were read.  This is stronger
// than tracking only the sequence returned to one compute process: a
// cross-shard upsert reserves old-generation cleanup on its old physical home
// and Stage2 on its new home, while the public mutation result carries only
// the latter sequence.  Waiting for all captured prefixes covers both without
// adding a second sequence to every mutation response.
inline std::vector<u64> capture_targets(
    std::span<const u64> requested_targets,
    std::span<const u64> next_sequences) {
  if (requested_targets.size() != next_sequences.size()) {
    throw std::invalid_argument(
      "maintenance target and control shard counts differ");
  }
  std::vector<u64> targets(requested_targets.begin(),
                           requested_targets.end());
  for (size_t shard = 0; shard < targets.size(); ++shard) {
    if (next_sequences[shard] == 0) {
      throw std::runtime_error(
        "storage maintenance next sequence is invalid");
    }
    targets[shard] = std::max(targets[shard], next_sequences[shard] - 1);
  }
  return targets;
}

}  // namespace gpu_search::maintenance_fence
