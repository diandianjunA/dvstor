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

inline void add_storage_owner_breakdown(
    const std::shared_ptr<service::breakdown::Sample>& sample,
    const service::storage_owner::InsertBreakdownCounters& counters,
    u32 item_count) {
  if (!sample) {
    return;
  }
  const u64 explained_search_ns =
    counters.storage_owner_search_select_ns +
    counters.storage_owner_search_neighbor_read_ns +
    counters.storage_owner_search_snapshot_read_ns +
    counters.storage_owner_search_distance_ns +
    counters.storage_owner_search_beam_update_ns +
    counters.storage_owner_search_result_sort_ns;
  const u64 explained_prune_ns =
    counters.storage_owner_prune_snapshot_read_ns +
    counters.storage_owner_prune_distance_ns +
    counters.storage_owner_prune_sort_ns +
    counters.storage_owner_prune_pair_distance_ns;
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_queue_wait,
                          per_item_ns(counters.storage_owner_queue_wait_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_medoid,
                          per_item_ns(counters.storage_owner_medoid_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search,
                          per_item_ns(saturating_sub(counters.storage_owner_search_ns,
                                                     explained_search_ns),
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
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_select,
                          per_item_ns(counters.storage_owner_search_select_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_search_neighbor_read,
                          per_item_ns(counters.storage_owner_search_neighbor_read_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_search_snapshot_read,
                          per_item_ns(counters.storage_owner_search_snapshot_read_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_distance,
                          per_item_ns(counters.storage_owner_search_distance_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_beam_update,
                          per_item_ns(counters.storage_owner_search_beam_update_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_search_result_sort,
                          per_item_ns(counters.storage_owner_search_result_sort_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_prune_snapshot_read,
                          per_item_ns(counters.storage_owner_prune_snapshot_read_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_distance,
                          per_item_ns(counters.storage_owner_prune_distance_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_sort,
                          per_item_ns(counters.storage_owner_prune_sort_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_prune_pair_distance,
                          per_item_ns(counters.storage_owner_prune_pair_distance_ns, item_count));
}

inline void add_storage_owner_counters(
    const std::shared_ptr<service::breakdown::Sample>& sample,
    const service::storage_owner::InsertBreakdownCounters& counters) {
  if (!sample) {
    return;
  }
  service::breakdown::ThreadCounterDelta delta{};
  delta.qir_qcode_rdma_ops = counters.qir_qcode_rdma_ops;
  delta.qir_qcode_rdma_bytes = counters.qir_qcode_rdma_bytes;
  delta.qir_qcode_cache_hits = counters.qir_qcode_cache_hits;
  delta.qir_qcode_cache_misses = counters.qir_qcode_cache_misses;
  delta.qir_exact_reads = counters.qir_exact_reads;
  delta.qir_exact_reads_avoided = counters.qir_exact_reads_avoided;
  delta.qir_uncertain_candidates = counters.qir_uncertain_candidates;
  delta.qir_prune_fallbacks = counters.qir_prune_fallbacks;
  delta.qir_repair_intents = counters.qir_repair_intents;
  delta.qir_repair_queue_delay_ns = counters.qir_repair_queue_delay_ns;
  delta.qir_repair_applied_edges = counters.qir_repair_applied_edges;
  delta.qir_repair_stale_skips = counters.qir_repair_stale_skips;
  delta.qir_sync_repair_fallbacks = counters.qir_sync_repair_fallbacks;
  delta.qir_audit_samples = counters.qir_audit_samples;
  delta.qir_audit_disagreements = counters.qir_audit_disagreements;
  sample->add_counters(delta);
}

inline void add_storage_owner_sender_breakdown(
    const std::shared_ptr<service::breakdown::Sample>& sample,
    u64 sender_queue_wait_ns,
    u64 batch_wait_ns,
    u64 request_prepare_ns,
    u64 send_ns,
    u64 response_wait_unaccounted_ns,
    u32 item_count) {
  if (!sample) {
    return;
  }
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_sender_queue_wait,
                          sender_queue_wait_ns);
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_batch_wait,
                          batch_wait_ns);
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_request_prepare,
                          per_item_ns(request_prepare_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::rdma_storage_owner_send,
                          per_item_ns(send_ns, item_count));
  sample->add_subcategory(service::breakdown::Subcategory::cpu_storage_owner_response_wait_unaccounted,
                          per_item_ns(response_wait_unaccounted_ns, item_count));
}

}  // namespace service::storage_owner_client
