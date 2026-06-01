template <class Distance>
size_t ComputeService<Distance>::insert(const vec<InsertItem>& batch) {
  if (config_.use_storage_owner_insert()) {
    if (storage_insert_owners_.empty()) {
      throw std::runtime_error("storage_owner insert runtime is not initialized");
    }
    vec<std::future<bool>> futures;
    vec<std::shared_ptr<service::breakdown::Sample>> samples;
    futures.reserve(batch.size());
    samples.reserve(batch.size());
    for (const auto& item : batch) {
      if (item.values.size() != config_.dim) {
        throw std::invalid_argument("insert dimension mismatch");
      }

      std::shared_ptr<service::breakdown::Sample> sample;
      if (breakdown_enabled_) {
        sample = std::make_shared<service::breakdown::Sample>(service::breakdown::Operation::insert);
        const auto now = std::chrono::steady_clock::now();
        sample->enqueued_at = now;
        sample->mark_started(now, now, statistics::ThreadStatistics{});
      }

      auto task = std::make_unique<StorageInsertTask>();
      task->item = item;
      task->sample = sample;
      task->enqueued_at = std::chrono::steady_clock::now();
      futures.push_back(task->result.get_future());
      samples.push_back(sample);
      const u32 owner_storage = num_servers_ == 0 ? 0 : static_cast<u32>(item.id % num_servers_);
      auto& state = *storage_insert_owners_[owner_storage];
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.queue.push_back(std::move(task));
      }
      state.cv.notify_one();
    }

    size_t inserted = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
    for (size_t i = 0; i < futures.size(); ++i) {
      auto& future = futures[i];
      if (future.wait_until(deadline) != std::future_status::ready) {
        const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 8) {
          std::cerr << "[storage-owner] insert RPC timed out after "
                    << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
        }
        if (samples[i] && !samples[i]->finished_flag) {
          samples[i]->mark_finished(std::chrono::steady_clock::now(), statistics::ThreadStatistics{});
        }
        continue;
      }
      inserted += future.get() ? 1u : 0u;
    }
    for (const auto& sample : samples) {
      if (sample && sample->finished_flag) {
        std::lock_guard<std::mutex> lock(breakdown_mutex_);
        completed_insert_samples_.push_back(*sample);
      }
    }
    vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
    note_graph_insertions(inserted);
    return inserted;
  }

  vec<service::InsertRequest*> requests;
  vec<std::future<bool>> futures;
  vec<std::shared_ptr<service::breakdown::Sample>> samples;
  requests.reserve(batch.size());
  futures.reserve(batch.size());
  samples.reserve(batch.size());

  for (const auto& item : batch) {
    if (item.values.size() != config_.dim) {
      throw std::invalid_argument("insert dimension mismatch");
    }

    std::shared_ptr<service::breakdown::Sample> sample;
    if (breakdown_enabled_) {
      sample = std::make_shared<service::breakdown::Sample>(service::breakdown::Operation::insert);
      sample->enqueued_at = std::chrono::steady_clock::now();
    }

    auto* request = new service::InsertRequest{item.id, item.values, {}, std::chrono::steady_clock::now(), sample};
    futures.push_back(request->result.get_future());
    requests.push_back(request);
    samples.push_back(sample);
    insert_queue_.enqueue(request);
  }

  size_t inserted = 0;
  for (size_t i = 0; i < futures.size(); ++i) {
    const bool ok = futures[i].get();
    if (ok) {
      ++inserted;
    }
    if (samples[i] && samples[i]->finished_flag) {
      std::lock_guard<std::mutex> lock(breakdown_mutex_);
      completed_insert_samples_.push_back(*samples[i]);
    }
  }
  vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
  note_graph_insertions(inserted);

  for (auto* request : requests) {
    delete request;
  }

  return inserted;
}

template <class Distance>
void ComputeService<Distance>::start_storage_insert_runtime() {
  if (!storage_insert_owners_.empty()) {
    return;
  }

  const u32 owner_count = std::max<u32>(1, num_servers_);
  const u32 rpc_depth = std::max<u32>(1, config_.storage_owner_rpc_depth);
  const size_t request_bytes =
    service::storage_owner::insert_batch_request_bytes(config_.storage_owner_batch_max, config_.dim);
  const size_t response_bytes =
    service::storage_owner::insert_batch_response_bytes(config_.storage_owner_batch_max);
  const size_t max_inflight = static_cast<size_t>(owner_count) * rpc_depth;
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_send_queue_wr),
             "storage_owner RPC depth exceeds compute send CQ capacity");
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_recv_queue_wr),
             "storage_owner RPC depth exceeds compute receive CQ capacity");

  storage_insert_shutdown_.store(false, std::memory_order_release);
  storage_insert_senders_done_.store(false, std::memory_order_release);
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_insert_owners_.reserve(owner_count);
  for (u32 owner = 0; owner < owner_count; ++owner) {
    auto state = std::make_unique<StorageOwnerSenderState>();
    state->slots.resize(rpc_depth);
    for (u32 slot_id = 0; slot_id < rpc_depth; ++slot_id) {
      auto& slot = state->slots[slot_id];
      slot.owner_storage = owner;
      slot.slot_id = slot_id;
      slot.request_buffer.assign(request_bytes, 0);
      slot.response_buffer.assign(response_bytes, 0);
      slot.request_region =
        std::make_unique<LocalMemoryRegion>(context_, slot.request_buffer.data(), slot.request_buffer.size());
      slot.tasks.reserve(config_.storage_owner_batch_max);
      slot.samples.reserve(config_.storage_owner_batch_max);
      state->free_slots.push_back(slot_id);
    }
    state->response_slots.resize(rpc_depth);
    for (u32 response_slot_id = 0; response_slot_id < rpc_depth; ++response_slot_id) {
      auto& response_slot = state->response_slots[response_slot_id];
      response_slot.owner_storage = owner;
      response_slot.slot_id = response_slot_id;
      response_slot.buffer.assign(response_bytes, 0);
      response_slot.region =
        std::make_unique<LocalMemoryRegion>(context_, response_slot.buffer.data(), response_slot.buffer.size());
    }
    storage_insert_owners_.push_back(std::move(state));
  }
  for (u32 owner = 0; owner < owner_count; ++owner) {
    for (u32 response_slot_id = 0; response_slot_id < rpc_depth; ++response_slot_id) {
      post_storage_owner_response_receive(owner, response_slot_id);
    }
  }

  storage_insert_completion_thread_ =
    std::thread([this]() { run_storage_insert_completion_loop(); });
  for (u32 owner = 0; owner < owner_count; ++owner) {
    storage_insert_owners_[owner]->thread =
      std::thread([this, owner]() { run_storage_insert_sender(owner); });
  }
}

template <class Distance>
void ComputeService<Distance>::stop_storage_insert_runtime() {
  storage_insert_shutdown_.store(true, std::memory_order_release);
  for (auto& state : storage_insert_owners_) {
    if (state) {
      state->cv.notify_all();
    }
  }

  for (auto& state : storage_insert_owners_) {
    if (state && state->thread.joinable()) {
      state->thread.join();
    }
  }

  for (auto& state : storage_insert_owners_) {
    if (!state) {
      continue;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    while (!state->queue.empty()) {
      vec<std::unique_ptr<StorageInsertTask>> task;
      task.push_back(std::move(state->queue.front()));
      state->queue.pop_front();
      fail_storage_owner_tasks(task);
    }
    for (auto& slot : state->slots) {
      if (slot.in_use && !slot.results_completed) {
        fail_storage_owner_tasks(slot.tasks);
      }
      slot.in_use = false;
      slot.send_done = true;
      slot.response_done = true;
      slot.results_completed = true;
    }
  }
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_insert_senders_done_.store(true, std::memory_order_release);
  if (storage_insert_completion_thread_.joinable()) {
    storage_insert_completion_thread_.join();
  }

  for (auto& state : storage_insert_owners_) {
    if (!state) {
      continue;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& slot : state->slots) {
      slot = StorageOwnerRpcSlot{};
    }
    state->batch_to_slot.clear();
    state->free_slots.clear();
  }
  storage_insert_owners_.clear();
}

template <class Distance>
void ComputeService<Distance>::run_storage_insert_sender(u32 owner_storage) {
  auto& state = *storage_insert_owners_[owner_storage];
  for (;;) {
    vec<std::unique_ptr<StorageInsertTask>> owned_tasks;
    u64 batch_wait_ns = 0;
    auto owner_selected_at = std::chrono::steady_clock::now();
    u32 slot_id = 0;
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      state.cv.wait(lock, [&]() {
        return (storage_insert_shutdown_.load(std::memory_order_acquire) && state.queue.empty()) ||
               (!state.queue.empty() && !state.free_slots.empty());
      });

      if (storage_insert_shutdown_.load(std::memory_order_acquire) && state.queue.empty()) {
        return;
      }
      if (state.queue.empty() || state.free_slots.empty()) {
        continue;
      }
      owner_selected_at = std::chrono::steady_clock::now();

      if (config_.storage_owner_batch_wait_us > 0 &&
          state.queue.size() < config_.storage_owner_batch_max &&
          !storage_insert_shutdown_.load(std::memory_order_acquire)) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(config_.storage_owner_batch_wait_us);
        const auto batch_wait_start = std::chrono::steady_clock::now();
        while (state.queue.size() < config_.storage_owner_batch_max &&
               !storage_insert_shutdown_.load(std::memory_order_acquire)) {
          if (state.cv.wait_until(lock, deadline) == std::cv_status::timeout) {
            break;
          }
        }
        batch_wait_ns = duration_ns(batch_wait_start, std::chrono::steady_clock::now());
      }

      slot_id = state.free_slots.front();
      state.free_slots.pop_front();
      const size_t batch_size = std::min<size_t>(state.queue.size(), config_.storage_owner_batch_max);
      owned_tasks.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        state.queue.front()->sender_dequeued_at = owner_selected_at;
        owned_tasks.push_back(std::move(state.queue.front()));
        state.queue.pop_front();
      }
    }

    if (owned_tasks.empty()) {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.free_slots.push_back(slot_id);
      state.cv.notify_one();
      continue;
    }
    post_storage_owner_batch(owner_storage, slot_id, std::move(owned_tasks), batch_wait_ns);
  }
}

template <class Distance>
void ComputeService<Distance>::post_storage_owner_batch(
    u32 owner_storage,
    u32 slot_id,
    vec<std::unique_ptr<StorageInsertTask>>&& tasks,
    u64 batch_wait_ns) {
  if (tasks.empty()) {
    return;
  }

  const u32 item_count = static_cast<u32>(tasks.size());
  const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  const auto prepare_start = std::chrono::steady_clock::now();
  const size_t request_size = service::storage_owner::insert_batch_request_bytes(item_count, config_.dim);
  const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
  auto& state = *storage_insert_owners_[owner_storage];
  auto& slot = state.slots[slot_id];
  slot.tasks = std::move(tasks);
  slot.samples.clear();
  slot.samples.reserve(slot.tasks.size());
  for (const auto& task : slot.tasks) {
    slot.samples.push_back(task->sample);
  }

  auto* request = reinterpret_cast<service::storage_owner::InsertBatchRequestHeader*>(slot.request_buffer.data());
  request->magic = service::storage_owner::kInsertMagic;
  request->dim = config_.dim;
  request->owner_storage = owner_storage;
  request->source_client = cm_.client_id;
  request->item_count = item_count;
  request->batch_id = batch_id;

  node_t* ids = service::storage_owner::request_ids(slot.request_buffer.data());
  element_t* vectors = service::storage_owner::request_vectors(slot.request_buffer.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    ids[i] = slot.tasks[i]->item.id;
    std::memcpy(vectors + static_cast<size_t>(i) * config_.dim,
                slot.tasks[i]->item.values.data(),
                static_cast<size_t>(config_.dim) * sizeof(element_t));
  }

  const u64 request_prepare_ns = duration_ns(prepare_start, std::chrono::steady_clock::now());
  std::memset(slot.response_buffer.data(), 0, response_size);

  const u64 wr_id = storage_owner_wr_id(owner_storage, slot_id);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    slot.in_use = true;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.item_count = item_count;
    slot.batch_id = batch_id;
    slot.batch_wait_ns = batch_wait_ns;
    slot.request_prepare_ns = request_prepare_ns;
    slot.request_size = request_size;
    slot.response_size = response_size;
    slot.send_posted_at = std::chrono::steady_clock::now();
    state.batch_to_slot[batch_id] = slot_id;
  }
  storage_insert_inflight_.fetch_add(1, std::memory_order_acq_rel);
  auto& qp = *cm_.server_qps[owner_storage];
  qp.post_send_with_id(*slot.request_region,
                       static_cast<u32>(request_size),
                       IBV_WR_SEND,
                       wr_id,
                       true,
                       nullptr,
                       0,
                       0);
}

template <class Distance>
void ComputeService<Distance>::run_storage_insert_completion_loop() {
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));

  for (;;) {
    bool progressed = false;
    const i32 send_count = Context::poll_send_cq(
      send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
        const auto [owner_storage, slot_id] = decode_64bit(wr_id);
        handle_storage_owner_send_completion(owner_storage, slot_id);
      });
    progressed = progressed || send_count > 0;

    const i32 recv_count =
      context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    progressed = progressed || recv_count > 0;
    for (i32 i = 0; i < recv_count; ++i) {
      const auto [owner_storage, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      handle_storage_owner_response(owner_storage, slot_id);
    }

    if (storage_insert_shutdown_.load(std::memory_order_acquire) &&
        storage_insert_senders_done_.load(std::memory_order_acquire) &&
        storage_insert_inflight_.load(std::memory_order_acquire) == 0) {
      return;
    }
    if (!progressed) {
      std::this_thread::yield();
    }
  }
}

template <class Distance>
void ComputeService<Distance>::handle_storage_owner_send_completion(u32 owner_storage, u32 slot_id) {
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
  maybe_release_storage_owner_slot_locked(state, slot);
}

template <class Distance>
void ComputeService<Distance>::handle_storage_owner_response(u32 owner_storage, u32 slot_id) {
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
    const bool header_ok = response->magic == service::storage_owner::kInsertMagic &&
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
        maybe_release_storage_owner_slot_locked(state, slot);
      }
    }
  }
  post_storage_owner_response_receive(owner_storage, slot_id);
}

template <class Distance>
void ComputeService<Distance>::post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id) {
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

template <class Distance>
void ComputeService<Distance>::maybe_release_storage_owner_slot_locked(
    StorageOwnerSenderState& state,
    StorageOwnerRpcSlot& slot) {
  if (!slot.in_use || !slot.send_done || !slot.response_done) {
    return;
  }

  if (!slot.results_completed) {
    const auto* response =
      reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(slot.response_buffer.data());
    const u32* statuses = service::storage_owner::response_statuses(slot.response_buffer.data());
    const bool response_ok = response->magic == service::storage_owner::kInsertMagic &&
                             response->owner_storage == slot.owner_storage &&
                             response->batch_id == slot.batch_id &&
                             response->item_count == slot.item_count;
    const auto* breakdown = response_ok
                              ? service::storage_owner::response_breakdown(slot.response_buffer.data(), slot.item_count)
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
    const u64 send_ns = duration_ns_clamped(slot.send_posted_at, slot.send_completed_at);
    const u64 response_wait_ns = duration_ns_clamped(slot.send_completed_at, slot.response_completed_at);
    const u64 response_wait_unaccounted_ns =
      response_wait_ns > memory_breakdown_ns ? response_wait_ns - memory_breakdown_ns : 0;
    const auto finished_at = slot.response_completed_at;

    for (u32 i = 0; i < slot.item_count; ++i) {
      const bool ok = response_ok &&
                      statuses[i] == static_cast<u32>(service::storage_owner::InsertStatus::ok);
      if (response_ok && !ok) {
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
      if (slot.samples[i]) {
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
        }
        slot.samples[i]->mark_finished(finished_at, statistics::ThreadStatistics{});
      }
      slot.tasks[i]->result.set_value(ok);
    }
    slot.results_completed = true;
  }

  const u32 slot_id = slot.slot_id;
  const u64 batch_id = slot.batch_id;
  slot.in_use = false;
  slot.send_done = false;
  slot.response_done = false;
  slot.results_completed = false;
  slot.item_count = 0;
  slot.batch_id = 0;
  slot.batch_wait_ns = 0;
  slot.request_prepare_ns = 0;
  slot.request_size = 0;
  slot.response_size = 0;
  slot.tasks.clear();
  slot.samples.clear();
  state.batch_to_slot.erase(batch_id);
  state.free_slots.push_back(slot_id);
  storage_insert_inflight_.fetch_sub(1, std::memory_order_acq_rel);
  state.cv.notify_one();
}

template <class Distance>
void ComputeService<Distance>::fail_storage_owner_tasks(vec<std::unique_ptr<StorageInsertTask>>& tasks) {
  const auto finished_at = std::chrono::steady_clock::now();
  for (auto& task : tasks) {
    if (!task) {
      continue;
    }
    if (task->sample) {
      task->sample->mark_finished(finished_at, statistics::ThreadStatistics{});
    }
    task->result.set_value(false);
  }
  tasks.clear();
}
