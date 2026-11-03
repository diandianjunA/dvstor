#pragma once

#include <algorithm>
#include <chrono>

#include "memory_node/storage_owner_maintenance/reconcile_batch_state.hh"

namespace memory_node_storage_owner_maintenance_detail {

enum class Stage2ItemPipelineAction : std::uint8_t {
  wait,
  start_install,
  poll_reconcile,
  finish_placement,
  complete,
};

inline Stage2ItemPipelineAction stage2_item_pipeline_action(
    Stage2FinalizeSubphase subphase,
    std::chrono::steady_clock::time_point retry_not_before,
    std::chrono::steady_clock::time_point now) {
  if (subphase == Stage2FinalizeSubphase::complete) {
    return Stage2ItemPipelineAction::complete;
  }
  if (retry_not_before > now) return Stage2ItemPipelineAction::wait;
  switch (subphase) {
    case Stage2FinalizeSubphase::prepare:
      return Stage2ItemPipelineAction::start_install;
    case Stage2FinalizeSubphase::install_wait:
    case Stage2FinalizeSubphase::removal_wait:
    case Stage2FinalizeSubphase::release_wait:
      return Stage2ItemPipelineAction::poll_reconcile;
    case Stage2FinalizeSubphase::placement_ready:
      return Stage2ItemPipelineAction::finish_placement;
    case Stage2FinalizeSubphase::complete:
      break;
  }
  return Stage2ItemPipelineAction::wait;
}

template <class SubphaseRange>
bool stage2_context_items_complete(const SubphaseRange& subphases) {
  return std::all_of(subphases.begin(), subphases.end(),
    [](Stage2FinalizeSubphase subphase) {
      return subphase == Stage2FinalizeSubphase::complete;
    });
}

}  // namespace memory_node_storage_owner_maintenance_detail
