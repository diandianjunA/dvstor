#pragma once

#include <array>
#include <chrono>
#include <string_view>

#include "common/types.hh"

namespace service::breakdown {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

enum class Operation : u8 { query = 0, insert = 1 };

enum class Category : u8 {
  cpu = 0,
  gpu,
  rdma,
  transfer,
  count
};

constexpr size_t kCategoryCount = static_cast<size_t>(Category::count);

inline constexpr std::array<std::string_view, kCategoryCount> kCategoryNames = {
  "cpu_ns",
  "gpu_ns",
  "rdma_ns",
  "transfer_ns",
};

enum class Subcategory : u8 {
  // CPU
  cpu_node_read = 0,
  cpu_query_select,
  cpu_query_filter,
  cpu_query_rabitq_gate,
  cpu_query_stage_candidates,
  cpu_query_beam_update,
  cpu_query_rerank_collect,
  cpu_query_rerank_prepare,
  cpu_query_rerank_update,
  cpu_query_beam_sort,
  cpu_query_result_ids,
  cpu_query_finalize,
  cpu_insert_init,
  cpu_insert_select,
  cpu_insert_filter,
  cpu_insert_stage_candidates,
  cpu_insert_preprune_sort,
  cpu_insert_candidate_collect,
  cpu_insert_beam_update,
  cpu_insert_candidate_sort,
  cpu_insert_prune_prepare,
  cpu_insert_neighbor_collect,
  cpu_insert_finalize,
  cpu_insert_neighbor_prepare,
  cpu_insert_pruned_neighbor_collect,
  cpu_insert_overflow_prepare,
  cpu_storage_owner_queue_wait,
  cpu_storage_owner_search,
  cpu_storage_owner_prune,
  cpu_storage_owner_write_node,
  cpu_storage_owner_local_reverse,
  cpu_storage_owner_remote_reverse,
  cpu_storage_owner_peer_reverse_apply,
  cpu_storage_owner_response_send,
  cpu_storage_owner_sender_queue_wait,
  cpu_storage_owner_batch_wait,
  cpu_storage_owner_request_prepare,
  cpu_storage_owner_response_wait_unaccounted,
  cpu_storage_owner_search_select,
  cpu_storage_owner_search_distance,
  cpu_storage_owner_search_beam_update,
  cpu_storage_owner_search_result_sort,
  cpu_storage_owner_prune_distance,
  cpu_storage_owner_prune_sort,
  cpu_storage_owner_prune_pair_distance,

  // GPU
  gpu_query_prepare,
  gpu_query_distance,
  gpu_query_rerank,
  gpu_insert_distance,
  gpu_insert_prune,
  gpu_insert_overflow_distance,
  gpu_insert_overflow_prune,

  // RDMA
  rdma_medoid_ptr,
  rdma_neighbor_fetch,
  rdma_vector_fetch,
  rdma_rerank_fetch,
  rdma_alloc,
  rdma_new_node_write,
  rdma_medoid_update,
  rdma_header_write,
  rdma_candidate_fetch,
  rdma_neighbor_node_read,
  rdma_neighbor_lock,
  rdma_neighbor_list_read,
  rdma_neighbor_list_write,
  rdma_overflow_vec_fetch,
  rdma_pruned_neighbor_write,
  rdma_neighbor_unlock,
  rdma_storage_owner_medoid,
  rdma_storage_owner_send,
  rdma_storage_owner_search_neighbor_read,
  rdma_storage_owner_search_snapshot_read,
  rdma_storage_owner_prune_snapshot_read,

  // Transfer
  transfer_query_h2d,
  transfer_candidate_h2d,
  transfer_distance_d2h,
  transfer_rerank_h2d,
  transfer_rerank_d2h,
  transfer_insert_query_h2d,
  transfer_prune_h2d,
  transfer_prune_d2h,
  transfer_overflow_query_h2d,
  transfer_overflow_candidate_h2d,
  transfer_overflow_dist_d2h,
  transfer_overflow_prune_inputs_h2d,
  transfer_overflow_prune_d2h,
  count
};

constexpr size_t kSubcategoryCount = static_cast<size_t>(Subcategory::count);

inline constexpr std::array<std::string_view, kSubcategoryCount> kSubcategoryNames = {
  "cpu_node_read_ns",
  "cpu_query_select_ns",
  "cpu_query_filter_ns",
  "cpu_query_rabitq_gate_ns",
  "cpu_query_stage_candidates_ns",
  "cpu_query_beam_update_ns",
  "cpu_query_rerank_collect_ns",
  "cpu_query_rerank_prepare_ns",
  "cpu_query_rerank_update_ns",
  "cpu_query_beam_sort_ns",
  "cpu_query_result_ids_ns",
  "cpu_query_finalize_ns",
  "cpu_insert_init_ns",
  "cpu_insert_select_ns",
  "cpu_insert_filter_ns",
  "cpu_insert_stage_candidates_ns",
  "cpu_insert_preprune_sort_ns",
  "cpu_insert_candidate_collect_ns",
  "cpu_insert_beam_update_ns",
  "cpu_insert_candidate_sort_ns",
  "cpu_insert_prune_prepare_ns",
  "cpu_insert_neighbor_collect_ns",
  "cpu_insert_finalize_ns",
  "cpu_insert_neighbor_prepare_ns",
  "cpu_insert_pruned_neighbor_collect_ns",
  "cpu_insert_overflow_prepare_ns",
  "cpu_storage_owner_queue_wait_ns",
  "cpu_storage_owner_search_ns",
  "cpu_storage_owner_prune_ns",
  "cpu_storage_owner_write_node_ns",
  "cpu_storage_owner_local_reverse_ns",
  "cpu_storage_owner_remote_reverse_ns",
  "cpu_storage_owner_peer_reverse_apply_ns",
  "cpu_storage_owner_response_send_ns",
  "cpu_storage_owner_sender_queue_wait_ns",
  "cpu_storage_owner_batch_wait_ns",
  "cpu_storage_owner_request_prepare_ns",
  "cpu_storage_owner_response_wait_unaccounted_ns",
  "cpu_storage_owner_search_select_ns",
  "cpu_storage_owner_search_distance_ns",
  "cpu_storage_owner_search_beam_update_ns",
  "cpu_storage_owner_search_result_sort_ns",
  "cpu_storage_owner_prune_distance_ns",
  "cpu_storage_owner_prune_sort_ns",
  "cpu_storage_owner_prune_pair_distance_ns",
  "gpu_query_prepare_ns",
  "gpu_query_distance_ns",
  "gpu_query_rerank_ns",
  "gpu_insert_distance_ns",
  "gpu_insert_prune_ns",
  "gpu_insert_overflow_distance_ns",
  "gpu_insert_overflow_prune_ns",
  "rdma_medoid_ptr_ns",
  "rdma_neighbor_fetch_ns",
  "rdma_vector_fetch_ns",
  "rdma_rerank_fetch_ns",
  "rdma_alloc_ns",
  "rdma_new_node_write_ns",
  "rdma_medoid_update_ns",
  "rdma_header_write_ns",
  "rdma_candidate_fetch_ns",
  "rdma_neighbor_node_read_ns",
  "rdma_neighbor_lock_ns",
  "rdma_neighbor_list_read_ns",
  "rdma_neighbor_list_write_ns",
  "rdma_overflow_vec_fetch_ns",
  "rdma_pruned_neighbor_write_ns",
  "rdma_neighbor_unlock_ns",
  "rdma_storage_owner_medoid_ns",
  "rdma_storage_owner_send_ns",
  "rdma_storage_owner_search_neighbor_read_ns",
  "rdma_storage_owner_search_snapshot_read_ns",
  "rdma_storage_owner_prune_snapshot_read_ns",
  "transfer_query_h2d_ns",
  "transfer_candidate_h2d_ns",
  "transfer_distance_d2h_ns",
  "transfer_rerank_h2d_ns",
  "transfer_rerank_d2h_ns",
  "transfer_insert_query_h2d_ns",
  "transfer_prune_h2d_ns",
  "transfer_prune_d2h_ns",
  "transfer_overflow_query_h2d_ns",
  "transfer_overflow_candidate_h2d_ns",
  "transfer_overflow_dist_d2h_ns",
  "transfer_overflow_prune_inputs_h2d_ns",
  "transfer_overflow_prune_d2h_ns",
};

inline constexpr std::string_view operation_name(const Operation operation) {
  return operation == Operation::query ? "query" : "insert";
}

inline constexpr Category parent_category(const Subcategory subcategory) {
  switch (subcategory) {
    case Subcategory::cpu_node_read:
    case Subcategory::cpu_query_select:
    case Subcategory::cpu_query_filter:
    case Subcategory::cpu_query_rabitq_gate:
    case Subcategory::cpu_query_stage_candidates:
    case Subcategory::cpu_query_beam_update:
    case Subcategory::cpu_query_rerank_collect:
    case Subcategory::cpu_query_rerank_prepare:
    case Subcategory::cpu_query_rerank_update:
    case Subcategory::cpu_query_beam_sort:
    case Subcategory::cpu_query_result_ids:
    case Subcategory::cpu_query_finalize:
    case Subcategory::cpu_insert_init:
    case Subcategory::cpu_insert_select:
    case Subcategory::cpu_insert_filter:
    case Subcategory::cpu_insert_stage_candidates:
    case Subcategory::cpu_insert_preprune_sort:
    case Subcategory::cpu_insert_candidate_collect:
    case Subcategory::cpu_insert_beam_update:
    case Subcategory::cpu_insert_candidate_sort:
    case Subcategory::cpu_insert_prune_prepare:
    case Subcategory::cpu_insert_neighbor_collect:
    case Subcategory::cpu_insert_finalize:
    case Subcategory::cpu_insert_neighbor_prepare:
    case Subcategory::cpu_insert_pruned_neighbor_collect:
    case Subcategory::cpu_insert_overflow_prepare:
    case Subcategory::cpu_storage_owner_queue_wait:
    case Subcategory::cpu_storage_owner_search:
    case Subcategory::cpu_storage_owner_prune:
    case Subcategory::cpu_storage_owner_write_node:
    case Subcategory::cpu_storage_owner_local_reverse:
    case Subcategory::cpu_storage_owner_remote_reverse:
    case Subcategory::cpu_storage_owner_peer_reverse_apply:
    case Subcategory::cpu_storage_owner_response_send:
    case Subcategory::cpu_storage_owner_sender_queue_wait:
    case Subcategory::cpu_storage_owner_batch_wait:
    case Subcategory::cpu_storage_owner_request_prepare:
    case Subcategory::cpu_storage_owner_response_wait_unaccounted:
    case Subcategory::cpu_storage_owner_search_select:
    case Subcategory::cpu_storage_owner_search_distance:
    case Subcategory::cpu_storage_owner_search_beam_update:
    case Subcategory::cpu_storage_owner_search_result_sort:
    case Subcategory::cpu_storage_owner_prune_distance:
    case Subcategory::cpu_storage_owner_prune_sort:
    case Subcategory::cpu_storage_owner_prune_pair_distance:
      return Category::cpu;
    case Subcategory::gpu_query_prepare:
    case Subcategory::gpu_query_distance:
    case Subcategory::gpu_query_rerank:
    case Subcategory::gpu_insert_distance:
    case Subcategory::gpu_insert_prune:
    case Subcategory::gpu_insert_overflow_distance:
    case Subcategory::gpu_insert_overflow_prune:
      return Category::gpu;
    case Subcategory::rdma_medoid_ptr:
    case Subcategory::rdma_neighbor_fetch:
    case Subcategory::rdma_vector_fetch:
    case Subcategory::rdma_rerank_fetch:
    case Subcategory::rdma_alloc:
    case Subcategory::rdma_new_node_write:
    case Subcategory::rdma_medoid_update:
    case Subcategory::rdma_header_write:
    case Subcategory::rdma_candidate_fetch:
    case Subcategory::rdma_neighbor_node_read:
    case Subcategory::rdma_neighbor_lock:
    case Subcategory::rdma_neighbor_list_read:
    case Subcategory::rdma_neighbor_list_write:
    case Subcategory::rdma_overflow_vec_fetch:
    case Subcategory::rdma_pruned_neighbor_write:
    case Subcategory::rdma_neighbor_unlock:
    case Subcategory::rdma_storage_owner_medoid:
    case Subcategory::rdma_storage_owner_send:
    case Subcategory::rdma_storage_owner_search_neighbor_read:
    case Subcategory::rdma_storage_owner_search_snapshot_read:
    case Subcategory::rdma_storage_owner_prune_snapshot_read:
      return Category::rdma;
    case Subcategory::transfer_query_h2d:
    case Subcategory::transfer_candidate_h2d:
    case Subcategory::transfer_distance_d2h:
    case Subcategory::transfer_rerank_h2d:
    case Subcategory::transfer_rerank_d2h:
    case Subcategory::transfer_insert_query_h2d:
    case Subcategory::transfer_prune_h2d:
    case Subcategory::transfer_prune_d2h:
    case Subcategory::transfer_overflow_query_h2d:
    case Subcategory::transfer_overflow_candidate_h2d:
    case Subcategory::transfer_overflow_dist_d2h:
    case Subcategory::transfer_overflow_prune_inputs_h2d:
    case Subcategory::transfer_overflow_prune_d2h:
      return Category::transfer;
    case Subcategory::count:
      return Category::cpu;
  }
  return Category::cpu;
}

}  // namespace service::breakdown
