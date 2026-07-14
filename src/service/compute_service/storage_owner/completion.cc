#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::run_storage_insert_completion_loop() {
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
  const auto poll_completions = [&]() {
    bool progressed = false;
    const i32 send_count = Context::poll_send_cq(
      send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
        const auto [owner_storage, slot_id] = decode_64bit(wr_id);
        handle_storage_owner_send_completion(owner_storage, slot_id);
      });
    progressed = send_count > 0;

    const i32 recv_count =
      context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    progressed = progressed || recv_count > 0;
    for (i32 i = 0; i < recv_count; ++i) {
      const auto [owner_storage, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      handle_storage_owner_response(owner_storage, slot_id);
    }
    return progressed;
  };

  for (;;) {
    bool progressed = poll_completions();
    if (progressed) {
      complete_ready_storage_owner_slots();
    }

    if (storage_insert_shutdown_.load(std::memory_order_acquire) &&
        storage_insert_senders_done_.load(std::memory_order_acquire) &&
        storage_insert_inflight_.load(std::memory_order_acquire) == 0) {
      storage_insert_completion_done_.store(true, std::memory_order_release);
      storage_insert_publication_cv_.notify_all();
      return;
    }
    if (!progressed) {
      std::this_thread::yield();
    }
  }
}

void ComputeService::run_storage_insert_publication_loop() {
  const u32 publication_capacity = std::max<u32>(1, config_.gpu_query_slots);
  const auto aggregation_budget = std::chrono::microseconds(
    std::max<u32>(1, config_.update_visibility_us / 2));
  for (;;) {
    StorageOwnerPublicationBatch publication;
    {
      std::unique_lock<std::mutex> lock(storage_insert_publication_mutex_);
      storage_insert_publication_cv_.wait(lock, [&]() {
        return !storage_insert_publication_queue_.empty() ||
               storage_insert_completion_done_.load(std::memory_order_acquire);
      });
      if (storage_insert_publication_queue_.empty()) {
        return;
      }
      publication = std::move(storage_insert_publication_queue_.front());
      storage_insert_publication_queue_.pop_front();
      auto oldest = std::chrono::steady_clock::now();
      for (const auto& mutation : publication.mutations) {
        if (mutation.enqueued_at != std::chrono::steady_clock::time_point{}) {
          oldest = std::min(oldest, mutation.enqueued_at);
        }
      }
      const auto deadline = oldest + aggregation_budget;
      for (;;) {
        while (!storage_insert_publication_queue_.empty() &&
               publication.mutations.size() +
                   storage_insert_publication_queue_.front().mutations.size() <=
                 publication_capacity) {
          auto next = std::move(storage_insert_publication_queue_.front());
          storage_insert_publication_queue_.pop_front();
          publication.reserved_items += next.reserved_items;
          publication.mutations.insert(
            publication.mutations.end(),
            std::make_move_iterator(next.mutations.begin()),
            std::make_move_iterator(next.mutations.end()));
          publication.invalidated_graph_nodes.insert(
            publication.invalidated_graph_nodes.end(),
            std::make_move_iterator(next.invalidated_graph_nodes.begin()),
            std::make_move_iterator(next.invalidated_graph_nodes.end()));
        }
        const bool next_batch_does_not_fit =
          !storage_insert_publication_queue_.empty() &&
          publication.mutations.size() +
              storage_insert_publication_queue_.front().mutations.size() >
            publication_capacity;
        if (publication.mutations.size() >= publication_capacity ||
            next_batch_does_not_fit ||
            std::chrono::steady_clock::now() >= deadline ||
            storage_insert_completion_done_.load(std::memory_order_acquire)) {
          break;
        }
        storage_insert_publication_cv_.wait_until(lock, deadline, [&]() {
          return !storage_insert_publication_queue_.empty() ||
                 storage_insert_completion_done_.load(std::memory_order_acquire);
        });
      }
    }
    publish_storage_owner_mutations(std::move(publication));
  }
}

void ComputeService::handle_storage_owner_send_completion(u32 owner_storage, u32 slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) {
    return;
  }
  auto& state = *storage_insert_owners_[owner_storage];
  std::lock_guard<std::mutex> lock(state.mutex);
  if (slot_id >= state.slots.size()) {
    return;
  }
  auto& slot = state.slots[slot_id];
  if (!slot.in_use) {
    return;
  }
  slot.send_done = true;
  slot.send_completed_at = std::chrono::steady_clock::now();
}

void ComputeService::handle_storage_owner_response(u32 owner_storage, u32 slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) {
    return;
  }
  auto& state = *storage_insert_owners_[owner_storage];
  if (slot_id >= state.response_slots.size()) {
    return;
  }

  auto& response_slot = state.response_slots[slot_id];
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto* response =
      reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(response_slot.buffer.data());
    const bool header_ok = (response->magic == service::storage_owner::kInsertMagic ||
                            response->magic == service::storage_owner::kMutationMagic) &&
                           response->owner_storage == owner_storage;
    auto slot_it = header_ok ? state.batch_to_slot.find(response->batch_id) : state.batch_to_slot.end();
    if (!header_ok || slot_it == state.batch_to_slot.end() || slot_it->second >= state.slots.size()) {
      static std::atomic<u32> unknown_response_logs{0};
      const u32 log_index = unknown_response_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        std::cerr << "[storage-owner] unmatched insert response"
                  << " owner=" << owner_storage
                  << " response_slot=" << slot_id
                  << " magic=0x" << std::hex << response->magic << std::dec
                  << " response_owner=" << response->owner_storage
                  << " batch_id=" << response->batch_id
                  << " item_count=" << response->item_count
                  << std::endl;
      }
    } else {
      auto& slot = state.slots[slot_it->second];
      const size_t response_size = service::storage_owner::insert_batch_response_bytes(response->item_count);
      if (slot.in_use && response_size <= slot.response_buffer.size()) {
        std::memcpy(slot.response_buffer.data(), response_slot.buffer.data(), response_size);
        slot.response_done = true;
        slot.response_completed_at = std::chrono::steady_clock::now();
      }
    }
  }
  post_storage_owner_response_receive(owner_storage, slot_id);
}

void ComputeService::post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) {
    return;
  }
  auto& state = *storage_insert_owners_[owner_storage];
  if (response_slot_id >= state.response_slots.size()) {
    return;
  }
  auto& response_slot = state.response_slots[response_slot_id];
  const u64 wr_id = storage_owner_wr_id(owner_storage, response_slot_id);
  cm_.server_qps[owner_storage]->post_receive(
    *response_slot.region,
    static_cast<u32>(response_slot.buffer.size()),
    wr_id);
}

void ComputeService::complete_ready_storage_owner_slots() {
  const u32 publication_capacity = std::max<u32>(1, config_.gpu_query_slots);
  for (;;) {
    vec<std::pair<u32, u32>> ready_slots;
    u32 ready_items = 0;
    for (u32 owner_storage = 0;
         owner_storage < storage_insert_owners_.size() && ready_items < publication_capacity;
         ++owner_storage) {
      auto& state = *storage_insert_owners_[owner_storage];
      std::lock_guard<std::mutex> lock(state.mutex);
      for (auto& slot : state.slots) {
        if (!slot.in_use || !slot.send_done || !slot.response_done ||
            slot.results_completed || slot.completion_claimed) {
          continue;
        }
        if (!ready_slots.empty() &&
            ready_items + slot.item_count > publication_capacity) {
          continue;
        }
        slot.completion_claimed = true;
        ready_items += slot.item_count;
        ready_slots.emplace_back(owner_storage, slot.slot_id);
        if (ready_items >= publication_capacity) break;
      }
    }
    if (ready_slots.empty()) return;

    StorageOwnerPublicationBatch publication;
    publication.mutations.reserve(ready_items);
    publication.invalidated_graph_nodes.reserve(
      static_cast<size_t>(ready_items) * config_.R);
    commit_ready_storage_owner_slots(ready_slots, publication);
    if (publication.mutations.empty()) {
      if (persistent_search_ != nullptr && publication.reserved_items != 0) {
        persistent_search_->release_mutation_capacity(publication.reserved_items);
      }
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(storage_insert_publication_mutex_);
      storage_insert_publication_queue_.push_back(std::move(publication));
    }
    storage_insert_publication_cv_.notify_one();
  }
}

void ComputeService::commit_ready_storage_owner_slots(
    const vec<std::pair<u32, u32>>& ready_slots,
    StorageOwnerPublicationBatch& publication) {
  for (const auto& [owner_storage, slot_id] : ready_slots) {
    auto& state = *storage_insert_owners_[owner_storage];
    u32 release_reserved_items = 0;
    std::lock_guard<std::mutex> lock(state.mutex);
    auto& slot = state.slots[slot_id];
    if (!slot.in_use || !slot.send_done || !slot.response_done ||
        !slot.completion_claimed) {
      continue;
    }

    const auto* response =
      reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(
        slot.response_buffer.data());
    const auto* request =
      reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(
        slot.request_buffer.data());
    bool response_ok =
      (response->magic == service::storage_owner::kInsertMagic ||
       response->magic == service::storage_owner::kMutationMagic) &&
      response->magic == request->magic &&
      response->owner_storage == slot.owner_storage &&
      response->batch_id == slot.batch_id &&
      response->item_count == slot.item_count;
    u32 invalidation_count = 0;
    if (response_ok) {
      invalidation_count = *service::storage_owner::response_invalidation_count(
        slot.response_buffer.data(), slot.item_count);
      response_ok = invalidation_count <=
        service::storage_owner::response_invalidation_capacity(slot.item_count);
      if (!response_ok) invalidation_count = 0;
    }
    const u32* statuses = service::storage_owner::response_statuses(
      slot.response_buffer.data());
    const auto* results = service::storage_owner::response_mutation_results(
      slot.response_buffer.data(), slot.item_count);
    const bool mutation_request = request->magic == service::storage_owner::kMutationMagic;
    const byte_t* request_vectors = mutation_request
      ? service::storage_owner::mutation_request_vectors(
          slot.request_buffer.data(), slot.item_count)
      : service::storage_owner::request_vectors(
          slot.request_buffer.data(), slot.item_count);
    bool collect_breakdown = false;
    for (const auto& sample : slot.samples) {
      if (sample && sample->collects_breakdown()) {
        collect_breakdown = true;
        break;
      }
    }
    const auto* breakdown = collect_breakdown && response_ok
      ? service::storage_owner::response_breakdown(
          slot.response_buffer.data(), slot.item_count)
      : nullptr;
    if (!response_ok) {
      static std::atomic<u32> bad_response_logs{0};
      const u32 log_index = bad_response_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        std::cerr << "[storage-owner] invalid insert response"
                  << " owner=" << slot.owner_storage
                  << " slot=" << slot.slot_id
                  << " magic=0x" << std::hex << response->magic << std::dec
                  << " response_owner=" << response->owner_storage
                  << " expected_owner=" << slot.owner_storage
                  << " batch_id=" << response->batch_id
                  << " expected_batch_id=" << slot.batch_id
                  << " item_count=" << response->item_count
                  << " expected_item_count=" << slot.item_count
                  << std::endl;
      }
    }
    const u64 memory_breakdown_ns = breakdown == nullptr ? 0 : breakdown->total();
    const u64 send_ns = collect_breakdown
      ? duration_ns_clamped(slot.send_posted_at, slot.send_completed_at)
      : 0;
    const u64 response_wait_ns = collect_breakdown
      ? duration_ns_clamped(slot.send_completed_at, slot.response_completed_at)
      : 0;
    const u64 response_wait_unaccounted_ns = collect_breakdown &&
        response_wait_ns > memory_breakdown_ns
      ? response_wait_ns - memory_breakdown_ns : 0;
    if (persistent_search_ != nullptr && response_ok) {
      const u64* invalidated_raws =
        service::storage_owner::response_invalidated_raws(
          slot.response_buffer.data(), slot.item_count);
      for (u32 index = 0; index < invalidation_count; ++index) {
        if (invalidated_raws[index] != 0) {
          publication.invalidated_graph_nodes.push_back(invalidated_raws[index]);
        }
      }
    }

    u32 committed_items = 0;
    for (u32 i = 0; i < slot.item_count; ++i) {
      const bool committed = response_ok && statuses[i] == 0;
      if (response_ok && !committed) {
        static std::atomic<u32> failed_status_logs{0};
        const u32 log_index = failed_status_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 16) {
          std::cerr << "[storage-owner] insert failed"
                    << " owner=" << slot.owner_storage
                    << " slot=" << slot.slot_id
                    << " batch_id=" << slot.batch_id
                    << " item=" << i
                    << " status=" << statuses[i]
                    << std::endl;
        }
      }
      if (slot.samples[i] && slot.samples[i]->collects_breakdown()) {
        add_storage_owner_sender_breakdown(
          slot.samples[i],
          duration_ns_clamped(slot.tasks[i]->enqueued_at, slot.tasks[i]->sender_dequeued_at),
          per_item_ns(slot.batch_wait_ns, slot.item_count),
          slot.request_prepare_ns,
          send_ns,
          response_wait_unaccounted_ns,
          slot.item_count);
        if (breakdown) {
          add_storage_owner_breakdown(slot.samples[i], *breakdown, slot.item_count);
          if (i == 0) add_storage_owner_counters(slot.samples[i], *breakdown);
        }
      }
      if (slot.samples[i]) {
        slot.samples[i]->mark_finished(slot.response_completed_at);
      }
      if (committed) {
        const auto& result = results[i];
        if (slot.tasks[i]->kind == service::storage_owner::MutationKind::erase) {
          publish_compute_side_id(slot.tasks[i]->item.id,
                                  RemotePtr{result.old_rptr_raw}, true,
                                  slot.owner_storage, result.generation);
        } else {
          publish_compute_side_id(slot.tasks[i]->item.id,
                                  RemotePtr{result.new_rptr_raw}, false,
                                  slot.owner_storage, result.generation);
        }
        if (persistent_search_ != nullptr) {
          gpu_search::DeltaMutation mutation;
          mutation.id = slot.tasks[i]->item.id;
          mutation.kind = slot.tasks[i]->kind;
          mutation.generation = result.generation;
          mutation.remote_node = result.new_rptr_raw;
          mutation.old_remote_node = result.old_rptr_raw;
          mutation.anchor_hint = slot.tasks[i]->anchor_bucket_hint.raw_address;
          mutation.maintenance_sequence = result.maintenance_sequence;
          mutation.owner_storage = owner_storage;
          mutation.enqueued_at = slot.response_completed_at;
          if (mutation.kind != service::storage_owner::MutationKind::erase) {
            const byte_t* vector = request_vectors +
              static_cast<size_t>(i) * VamanaNode::vector_bytes();
            mutation.vector.assign(vector, vector + VamanaNode::vector_bytes());
          }
          publication.mutations.push_back(std::move(mutation));
          ++committed_items;
        }
      }
      slot.tasks[i]->result.set_value(committed);
    }

    lib_assert(committed_items <= slot.gpu_reserved_items,
               "committed storage mutations exceeded reserved GPU capacity");
    publication.reserved_items += committed_items;
    release_reserved_items = slot.gpu_reserved_items - committed_items;

    const u64 batch_id = slot.batch_id;
    slot.in_use = false;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.completion_claimed = false;
    slot.gpu_reserved_items = 0;
    slot.item_count = 0;
    slot.batch_id = 0;
    slot.batch_wait_ns = 0;
    slot.request_prepare_ns = 0;
    slot.request_size = 0;
    slot.response_size = 0;
    slot.send_posted_at = {};
    slot.send_completed_at = {};
    slot.response_completed_at = {};
    slot.tasks.clear();
    slot.samples.clear();
    state.batch_to_slot.erase(batch_id);
    state.free_slots.push_back(slot_id);
    storage_insert_inflight_.fetch_sub(1, std::memory_order_acq_rel);
    state.cv.notify_one();

    if (persistent_search_ != nullptr && release_reserved_items != 0) {
      persistent_search_->release_mutation_capacity(release_reserved_items);
    }
  }
}

void ComputeService::publish_storage_owner_mutations(
    StorageOwnerPublicationBatch&& publication) {
  if (persistent_search_ == nullptr || publication.mutations.empty()) {
    if (persistent_search_ != nullptr && publication.reserved_items != 0) {
      persistent_search_->release_mutation_capacity(publication.reserved_items);
    }
    return;
  }

  std::sort(publication.invalidated_graph_nodes.begin(),
            publication.invalidated_graph_nodes.end());
  publication.invalidated_graph_nodes.erase(
    std::unique(publication.invalidated_graph_nodes.begin(),
                publication.invalidated_graph_nodes.end()),
    publication.invalidated_graph_nodes.end());

  try {
    const u64 epoch = persistent_search_->delta().reserve_epoch();
    if (!persistent_search_->publish_mutations(
          std::move(publication.mutations), epoch,
          publication.invalidated_graph_nodes)) {
      persistent_search_->mark_committed_mutation_gap(
        "persistent GPU mutation publication returned false");
    }
  } catch (const std::exception& error) {
    persistent_search_->mark_committed_mutation_gap(error.what());
    static std::atomic<u32> gpu_delta_failure_logs{0};
    const u32 log_index = gpu_delta_failure_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] committed mutation batch was not published to GPU delta: "
                << error.what() << std::endl;
    }
  }
  persistent_search_->release_mutation_capacity(publication.reserved_items);
}

void ComputeService::fail_storage_owner_tasks(vec<std::unique_ptr<StorageInsertTask>>& tasks) {
  const auto finished_at = std::chrono::steady_clock::now();
  for (auto& task : tasks) {
    if (!task) {
      continue;
    }
    if (task->sample) {
      task->sample->mark_finished(finished_at);
    }
    task->result.set_value(false);
  }
  tasks.clear();
}
