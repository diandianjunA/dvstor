#pragma once

#include <array>
#include <string_view>

#include "common/types.hh"

namespace service::breakdown {

enum class Operation : u8 { query = 0, insert = 1 };

enum class Category : u8 { cpu = 0, rdma, count };

constexpr size_t kCategoryCount = static_cast<size_t>(Category::count);

inline constexpr std::array<std::string_view, kCategoryCount> kCategoryNames = {
  "cpu_ns",
  "rdma_ns",
};

enum class Subcategory : u8 {
  cpu_storage_owner_queue_wait = 0,
  cpu_storage_owner_search,
  cpu_storage_owner_prune,
  cpu_storage_owner_write_node,
  cpu_storage_owner_local_reverse,
  cpu_storage_owner_remote_reverse,
  cpu_storage_owner_peer_reverse_apply,
  cpu_storage_owner_response_send,
  cpu_storage_owner_prepare_mutation,
  cpu_storage_owner_allocate_node,
  cpu_storage_owner_publish_mutation,
  cpu_storage_owner_schedule_maintenance,
  cpu_storage_owner_response_build,
  cpu_storage_owner_sender_queue_wait,
  cpu_storage_owner_request_prepare,
  cpu_storage_owner_route,
  cpu_storage_owner_response_wait_unaccounted,
  cpu_storage_owner_dequeue_to_post,
  cpu_storage_owner_cq_progress_gap,
  cpu_storage_owner_response_executor_queue,
  cpu_storage_owner_response_process,
  cpu_storage_owner_caller_wake,
  cpu_storage_owner_search_select,
  cpu_storage_owner_search_distance,
  cpu_storage_owner_search_beam_update,
  cpu_storage_owner_search_result_sort,
  cpu_storage_owner_prune_distance,
  cpu_storage_owner_prune_sort,
  cpu_storage_owner_prune_pair_distance,
  rdma_storage_owner_send,
  rdma_storage_owner_search_neighbor_read,
  rdma_storage_owner_search_snapshot_read,
  rdma_storage_owner_prune_snapshot_read,
  count
};

constexpr size_t kSubcategoryCount = static_cast<size_t>(Subcategory::count);

inline constexpr std::array<std::string_view, kSubcategoryCount> kSubcategoryNames = {
  "cpu_storage_owner_queue_wait_ns",
  "cpu_storage_owner_search_ns",
  "cpu_storage_owner_prune_ns",
  "cpu_storage_owner_write_node_ns",
  "cpu_storage_owner_local_reverse_ns",
  "cpu_storage_owner_remote_reverse_ns",
  "cpu_storage_owner_peer_reverse_apply_ns",
  "cpu_storage_owner_response_send_ns",
  "cpu_storage_owner_prepare_mutation_ns",
  "cpu_storage_owner_allocate_node_ns",
  "cpu_storage_owner_publish_mutation_ns",
  "cpu_storage_owner_schedule_maintenance_ns",
  "cpu_storage_owner_response_build_ns",
  "cpu_storage_owner_sender_queue_wait_ns",
  "cpu_storage_owner_request_prepare_ns",
  "cpu_storage_owner_route_ns",
  "cpu_storage_owner_response_wait_unaccounted_ns",
  "cpu_storage_owner_dequeue_to_post_ns",
  "cpu_storage_owner_cq_progress_gap_ns",
  "cpu_storage_owner_response_executor_queue_ns",
  "cpu_storage_owner_response_process_ns",
  "cpu_storage_owner_caller_wake_ns",
  "cpu_storage_owner_search_select_ns",
  "cpu_storage_owner_search_distance_ns",
  "cpu_storage_owner_search_beam_update_ns",
  "cpu_storage_owner_search_result_sort_ns",
  "cpu_storage_owner_prune_distance_ns",
  "cpu_storage_owner_prune_sort_ns",
  "cpu_storage_owner_prune_pair_distance_ns",
  "rdma_storage_owner_send_ns",
  "rdma_storage_owner_search_neighbor_read_ns",
  "rdma_storage_owner_search_snapshot_read_ns",
  "rdma_storage_owner_prune_snapshot_read_ns",
};

inline constexpr std::string_view operation_name(Operation operation) {
  return operation == Operation::query ? "query" : "insert";
}

inline constexpr Category parent_category(Subcategory subcategory) {
  return subcategory >= Subcategory::rdma_storage_owner_send &&
         subcategory < Subcategory::count
    ? Category::rdma : Category::cpu;
}

}  // namespace service::breakdown
