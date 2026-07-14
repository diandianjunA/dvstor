#include "memory_node/storage_owner_runtime/detail.hh"

using namespace memory_node_storage_owner_runtime_detail;

bool MemoryNode::execute_storage_owner_batch_items_async(const node_t* ids,
                                             const service::storage_owner::MutationKind* kinds,
                                             const element_t* vectors,
                                             const u64* anchor_hints,
                                             u32 anchor_hint_count,
                                             size_t item_count,
                                             StorageOwnerThread& thread,
                                             InsertBreakdownCounters& breakdown,
                                             const Configuration& config,
                                             vec<vec<u64>>* invalidated_neighbors,
                                             vec<u32>* statuses,
                                             vec<service::storage_owner::MutationResult>* results) {
  if (item_count == 0) {
    return true;
  }

  vec<StorageOwnerInsertJob> jobs;
  jobs.reserve(item_count);
  for (size_t idx = 0; idx < item_count; ++idx) {
    StorageOwnerInsertJob job;
    job.id = ids[idx];
    job.kind = kinds == nullptr ? service::storage_owner::MutationKind::insert : kinds[idx];
    job.vector_data.resize(static_cast<size_t>(VamanaNode::DIM) * sizeof(element_t));
    std::memcpy(job.vector_data.data(),
                vectors + idx * VamanaNode::DIM,
                static_cast<size_t>(VamanaNode::DIM) * sizeof(element_t));
    if (anchor_hints != nullptr) {
      job.anchor_hints.reserve(anchor_hint_count);
      for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
        const RemotePtr ptr{anchor_hints[idx * anchor_hint_count + hint]};
        if (!ptr.is_null()) {
          job.anchor_hints.push_back(ptr);
        }
      }
    }
    jobs.push_back(std::move(job));
  }

  dense_hashmap_t<u64, vec<RemotePtr>> local_updates;
  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;

  const u32 coroutine_count = static_cast<u32>(std::max<size_t>(1, thread.post_balances.size()));
  lib_assert(thread.id < storage_owner_async_candidates_.size(),
             "storage_owner async candidate slots not initialized for worker");
  lib_assert(storage_owner_async_candidates_[thread.id].size() >= coroutine_count,
             "storage_owner async candidate slots not initialized for coroutines");

  thread.coroutines.clear();
  thread.coroutines.reserve(coroutine_count);
  for (u32 i = 0; i < coroutine_count; ++i) {
    thread.coroutines.emplace_back(std::make_unique<StorageOwnerInsertCoroutine>(dummy_storage_owner_insert_coroutine()));
  }

  size_t next_job = 0;
  for (;;) {
    bool all_done = true;
    poll_peer_send_cq();

    for (u32 coroutine_id = 0; coroutine_id < coroutine_count; ++coroutine_id) {
      auto& coroutine = *thread.coroutines[coroutine_id];
      if (coroutine.handle.done()) {
        if (next_job < jobs.size()) {
          coroutine.handle.destroy();
          thread.set_current_coroutine(coroutine_id);
          coroutine.handle = execute_storage_owner_insert_job_async(
            thread, jobs[next_job++], local_updates, remote_updates, breakdown, config).handle;
          all_done = false;
        }
      } else if (thread.is_ready(coroutine_id)) {
        thread.set_current_coroutine(coroutine_id);
        coroutine.handle.resume();
        all_done = false;
      } else {
        all_done = false;
      }
    }

    if (all_done) {
      break;
    }
  }

  for (const auto& coroutine : thread.coroutines) {
    lib_assert(coroutine->handle.done(), "storage-owner insert coroutine not done yet");
    coroutine->handle.destroy();
  }
  thread.coroutines.clear();

  bool ok = true;
  if (statuses != nullptr) {
    statuses->assign(item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  }
  if (results != nullptr) {
    results->assign(item_count, {});
  }
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->assign(item_count, {});
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& job = jobs[i];
    if (statuses != nullptr) {
      (*statuses)[i] = static_cast<u32>(job.status);
    }
    if (results != nullptr) {
      (*results)[i] = service::storage_owner::MutationResult{
        job.new_ptr.raw_address,
        job.old_ptr.raw_address,
        job.generation,
        0,
        job.maintenance_sequence};
    }
    if (invalidated_neighbors != nullptr) {
      (*invalidated_neighbors)[i] = job.invalidated_neighbors;
    }
  }
  auto t_local_reverse = std::chrono::steady_clock::now();
  for (auto& [target_raw, candidates] : local_updates) {
    ok &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
  }
  breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
  auto t_remote_reverse = std::chrono::steady_clock::now();
  for (auto& [target_shard, ops] : remote_updates) {
    ok &= send_reverse_update_batch(target_shard, ops, config);
  }
  breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
  return ok;
}

StorageOwnerInsertCoroutine MemoryNode::dummy_storage_owner_insert_coroutine() {
  co_return;
}

size_t MemoryNode::insert_request_slot_offset(u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return slot_index * insert_runtime_.request_bytes;
}

size_t MemoryNode::insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return insert_runtime_.response_offset + slot_index * response_slot_bytes(config);
}
