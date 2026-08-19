#pragma once

#include <chrono>
#include <memory>

#include <library/utils.hh>

#include "common/types.hh"
#include "service/breakdown.hh"
#include "service/storage_owner_protocol.hh"

namespace service::storage_owner_client {

inline u64 per_item_ns(u64 total, u32 item_count) {
  return item_count == 0 ? 0 : total / item_count;
}

inline u64 saturating_sub(u64 lhs, u64 rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

inline u64 duration_ns(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline u64 duration_ns_clamped(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
  if (end <= start) {
    return 0;
  }
  return duration_ns(start, end);
}

inline u64 storage_owner_wr_id(u32 owner_storage, u32 slot_id) {
  return encode_64bit(owner_storage, slot_id);
}

inline constexpr u32 kStorageOwnerCompletionWrBit = 0x80000000u;

inline u64 storage_owner_completion_wr_id(u32 owner_storage, u32 slot_id) {
  return encode_64bit(owner_storage | kStorageOwnerCompletionWrBit, slot_id);
}

inline bool storage_owner_is_completion_wr(u32 encoded_owner) {
  return (encoded_owner & kStorageOwnerCompletionWrBit) != 0;
}

inline u32 storage_owner_wr_owner(u32 encoded_owner) {
  return encoded_owner & ~kStorageOwnerCompletionWrBit;
}

// A successful coupled mutation is already exact and publishes no background
// maintenance debt. Every successful decoupled mutation, including erase and
// replay, commits only after obtaining its non-zero runnable Stage2 fence.
inline constexpr bool valid_success_maintenance_sequence(
    bool synchronous_exact, u64 maintenance_sequence) {
  return synchronous_exact
    ? maintenance_sequence == 0 : maintenance_sequence != 0;
}

inline void add_storage_owner_breakdown(
    service::breakdown::Sample* sample,
    const service::storage_owner::InsertBreakdownCounters& counters,
    u32 item_count) {
  if (!sample || !sample->collects_breakdown()) {
    return;
  }
  const u64 explained_search_ns =
    counters.storage_owner_search_select_ns +
    counters.storage_owner_search_distance_ns +
    counters.storage_owner_search_beam_update_ns +
    counters.storage_owner_search_result_sort_ns;
  const u64 explained_prune_ns =
    counters.storage_owner_prune_distance_ns +
    counters.storage_owner_prune_sort_ns +
    counters.storage_owner_prune_pair_distance_ns;
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_queue_wait,
                          per_item_ns(counters.storage_owner_queue_wait_ns, item_count));
  sample->add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_stage1_execute_wait,
    per_item_ns(counters.storage_owner_stage1_execute_wait_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search,
                          per_item_ns(saturating_sub(counters.storage_owner_search_ns, explained_search_ns),
                                      item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune,
                          per_item_ns(saturating_sub(counters.storage_owner_prune_ns, explained_prune_ns),
                                      item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_write_node,
                          per_item_ns(counters.storage_owner_write_node_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_local_reverse,
                          per_item_ns(counters.storage_owner_local_reverse_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_remote_reverse,
                          per_item_ns(counters.storage_owner_remote_reverse_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_peer_reverse_apply,
                          per_item_ns(counters.storage_owner_peer_reverse_apply_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_response_send,
                          per_item_ns(counters.storage_owner_response_send_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prepare_mutation,
                          per_item_ns(counters.storage_owner_prepare_mutation_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_allocate_node,
                          per_item_ns(counters.storage_owner_allocate_node_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_publish_mutation,
                          per_item_ns(counters.storage_owner_publish_mutation_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_schedule_maintenance,
                          per_item_ns(counters.storage_owner_schedule_maintenance_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_response_build,
                          per_item_ns(counters.storage_owner_response_build_ns, item_count));
  sample->add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_stage1_arm_wait,
    per_item_ns(counters.storage_owner_stage1_arm_wait_ns, item_count));
  sample->add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_stage1_release_wait,
    per_item_ns(counters.storage_owner_stage1_release_wait_ns, item_count));
  sample->add_subcategory(
    service::breakdown::Subcategory::cpu_storage_owner_cleanup_control_wait,
    per_item_ns(counters.storage_owner_cleanup_control_wait_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_select,
                          per_item_ns(counters.storage_owner_search_select_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_distance,
                          per_item_ns(counters.storage_owner_search_distance_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_beam_update,
                          per_item_ns(counters.storage_owner_search_beam_update_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_result_sort,
                          per_item_ns(counters.storage_owner_search_result_sort_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_distance,
                          per_item_ns(counters.storage_owner_prune_distance_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_sort,
                          per_item_ns(counters.storage_owner_prune_sort_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_pair_distance,
                          per_item_ns(counters.storage_owner_prune_pair_distance_ns, item_count));
}

inline void add_storage_owner_sender_breakdown(
    service::breakdown::Sample* sample,
    u64 sender_queue_wait_ns,
    u64 request_prepare_ns,
    u64 send_ns,
    u64 response_wait_unaccounted_ns,
    u32 item_count) {
  if (!sample || !sample->collects_breakdown()) {
    return;
  }
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_sender_queue_wait,
                          sender_queue_wait_ns);
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_request_prepare,
                          per_item_ns(request_prepare_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_send,
                          per_item_ns(send_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_response_wait_unaccounted,
                          per_item_ns(response_wait_unaccounted_ns, item_count));
}

}  // namespace service::storage_owner_client
