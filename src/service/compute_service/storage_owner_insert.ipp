size_t ComputeService::insert(const vec<InsertItem>& batch) {
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

      auto sample = std::make_shared<service::breakdown::Sample>(
        service::breakdown::Operation::insert, breakdown_enabled_);
      const auto now = std::chrono::steady_clock::now();
      sample->enqueued_at = now;
      sample->mark_started(now, now);

      auto task = std::make_unique<StorageInsertTask>();
      task->item = item;
      task->kind = service::storage_owner::MutationKind::insert;
      task->sample = sample;
      task->enqueued_at = sample->enqueued_at;
      futures.push_back(task->result.get_future());
      samples.push_back(sample);
      const auto route = route_storage_owner_update(item);
      task->anchor_hints = route.hints;
      task->anchor_bucket_hint = route.bucket_hint;
      const u32 owner_storage = route.owner;
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
          samples[i]->mark_finished(std::chrono::steady_clock::now());
        }
        continue;
      }
      inserted += future.get() ? 1u : 0u;
    }
    for (const auto& sample : samples) {
      if (sample && sample->finished_flag) {
        std::lock_guard<std::mutex> lock(breakdown_mutex_);
        service::breakdown::add_sample(
          completed_breakdown_report_.insert, *sample);
      }
    }
    vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
    return inserted;
}

size_t ComputeService::upsert(const vec<InsertItem>& batch) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
  }
  vec<std::future<bool>> futures;
  futures.reserve(batch.size());
  for (const auto& item : batch) {
    if (item.values.size() != config_.dim) {
      throw std::invalid_argument("upsert dimension mismatch");
    }
    auto task = std::make_unique<StorageInsertTask>();
    task->item = item;
    task->kind = service::storage_owner::MutationKind::upsert;
    futures.push_back(task->result.get_future());
    const u32 owner_storage = storage_owner_for_id(item.id);
    const auto route = route_storage_owner_update(item, owner_storage);
    task->anchor_hints = route.hints;
    task->anchor_bucket_hint = route.bucket_hint;
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.queue.push_back(std::move(task));
    }
    state.cv.notify_one();
  }
  size_t updated = 0;
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
  for (auto& future : futures) {
    if (future.wait_until(deadline) != std::future_status::ready) {
      const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] upsert RPC timed out after "
                  << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
      }
      continue;
    }
    updated += future.get() ? 1u : 0u;
  }
  vectors_inserted_.fetch_add(updated, std::memory_order_relaxed);
  return updated;
}

size_t ComputeService::erase(const vec<node_t>& ids) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
  }
  vec<std::future<bool>> futures;
  futures.reserve(ids.size());
  for (const node_t id : ids) {
    auto task = std::make_unique<StorageInsertTask>();
    task->item.id = id;
    task->item.values.assign(config_.dim, 0.0f);
    task->kind = service::storage_owner::MutationKind::erase;
    futures.push_back(task->result.get_future());
    const u32 owner_storage = storage_owner_for_id(id);
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.queue.push_back(std::move(task));
    }
    state.cv.notify_one();
  }
  size_t erased = 0;
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
  for (auto& future : futures) {
    if (future.wait_until(deadline) != std::future_status::ready) {
      const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] delete RPC timed out after "
                  << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
      }
      continue;
    }
    erased += future.get() ? 1u : 0u;
  }
  return erased;
}

void ComputeService::start_storage_insert_runtime() {
  if (!storage_insert_owners_.empty()) {
    return;
  }

  const u32 owner_count = std::max<u32>(1, num_servers_);
  const u32 rpc_depth = std::max<u32>(1, config_.storage_owner_rpc_depth);
  const bool anchor_mode = config_.storage_owner_update_mode == "local_stitch";
  const size_t request_bytes = std::max(
    service::storage_owner::insert_batch_request_bytes(
      config_.storage_owner_batch_max, config_.dim,
      anchor_mode ? config_.storage_owner_anchor_hints : 0),
    service::storage_owner::mutation_batch_request_bytes(
      config_.storage_owner_batch_max, config_.dim,
      anchor_mode ? config_.storage_owner_anchor_hints : 0));
  const size_t response_bytes =
    service::storage_owner::insert_batch_response_bytes(config_.storage_owner_batch_max);
  lib_assert(request_bytes <= std::numeric_limits<u32>::max() &&
             response_bytes <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs; reduce batch size or vector dimension");
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

void ComputeService::stop_storage_insert_runtime() {
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
      if (persistent_search_ != nullptr && slot.gpu_reserved_items != 0) {
        persistent_search_->release_mutation_capacity(slot.gpu_reserved_items);
        slot.gpu_reserved_items = 0;
      }
      slot.in_use = false;
      slot.send_done = true;
      slot.response_done = true;
      slot.results_completed = true;
      slot.completion_claimed = false;
    }
  }
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_insert_senders_done_.store(true, std::memory_order_release);
  if (storage_insert_completion_thread_.joinable()) {
    storage_insert_completion_thread_.join();
  }

}

void ComputeService::release_storage_insert_runtime() {
  for (auto& state : storage_insert_owners_) {
    if (!state) {
      continue;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& slot : state->slots) {
      slot.request_region.reset();
      slot.response_region.reset();
    }
    for (auto& response_slot : state->response_slots) {
      response_slot.region.reset();
    }
    state->slots.clear();
    state->response_slots.clear();
    state->batch_to_slot.clear();
    state->free_slots.clear();
  }
  storage_insert_owners_.clear();
}

void ComputeService::run_storage_insert_sender(u32 owner_storage) {
  auto& state = *storage_insert_owners_[owner_storage];
  for (;;) {
    vec<std::unique_ptr<StorageInsertTask>> owned_tasks;
    u64 batch_wait_ns = 0;
    std::chrono::steady_clock::time_point owner_selected_at{};
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
      if (breakdown_enabled_) {
        owner_selected_at = std::chrono::steady_clock::now();
      }

      if (config_.storage_owner_batch_wait_us > 0 &&
          state.queue.size() < config_.storage_owner_batch_max &&
          !storage_insert_shutdown_.load(std::memory_order_acquire)) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(config_.storage_owner_batch_wait_us);
        std::chrono::steady_clock::time_point batch_wait_start{};
        if (breakdown_enabled_) {
          batch_wait_start = std::chrono::steady_clock::now();
        }
        while (state.queue.size() < config_.storage_owner_batch_max &&
               !storage_insert_shutdown_.load(std::memory_order_acquire)) {
          if (state.cv.wait_until(lock, deadline) == std::cv_status::timeout) {
            break;
          }
        }
        if (breakdown_enabled_) {
          batch_wait_ns = duration_ns(batch_wait_start, std::chrono::steady_clock::now());
        }
      }

      slot_id = state.free_slots.front();
      state.free_slots.pop_front();
      const size_t batch_size = std::min<size_t>(state.queue.size(), config_.storage_owner_batch_max);
      owned_tasks.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        if (state.queue.front()->sample &&
            state.queue.front()->sample->collects_breakdown()) {
          state.queue.front()->sender_dequeued_at = owner_selected_at;
        }
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

void ComputeService::post_storage_owner_batch(
    u32 owner_storage,
    u32 slot_id,
    vec<std::unique_ptr<StorageInsertTask>>&& tasks,
    u64 batch_wait_ns) {
  if (tasks.empty()) {
    return;
  }

  const u32 item_count = static_cast<u32>(tasks.size());
  if (persistent_search_ != nullptr &&
      !persistent_search_->try_reserve_mutation_capacity(item_count)) {
    static std::atomic<u32> capacity_rejection_logs{0};
    const u32 log_index = capacity_rejection_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] write rejected before remote commit: "
                << "bounded GPU mutation tier is full" << std::endl;
    }
    fail_storage_owner_tasks(tasks);
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.free_slots.push_back(slot_id);
    }
    state.cv.notify_one();
    return;
  }
  const bool anchor_mode = config_.storage_owner_update_mode == "local_stitch";
  const u32 anchor_hint_count = anchor_mode
                                  ? config_.storage_owner_anchor_hints : 0;
  const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  bool collect_breakdown = false;
  for (const auto& task : tasks) {
    if (task->sample && task->sample->collects_breakdown()) {
      collect_breakdown = true;
      break;
    }
  }
  const auto prepare_start = collect_breakdown
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point{};
  bool mutation_request = false;
  for (const auto& task : tasks) {
    mutation_request = mutation_request || task->kind != service::storage_owner::MutationKind::insert;
  }
  const size_t request_size = mutation_request
    ? service::storage_owner::mutation_batch_request_bytes(item_count, config_.dim, anchor_hint_count)
    : service::storage_owner::insert_batch_request_bytes(item_count, config_.dim, anchor_hint_count);
  const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
  auto& state = *storage_insert_owners_[owner_storage];
  auto& slot = state.slots[slot_id];
  lib_assert(request_size <= slot.request_buffer.size() &&
             response_size <= slot.response_buffer.size(),
             "storage_owner RPC slot buffer is too small for this batch");
  lib_assert(request_size <= std::numeric_limits<u32>::max() &&
             response_size <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs");
  slot.tasks = std::move(tasks);
  slot.samples.clear();
  slot.samples.reserve(slot.tasks.size());
  for (const auto& task : slot.tasks) {
    slot.samples.push_back(task->sample);
  }

  auto* request = reinterpret_cast<service::storage_owner::InsertBatchRequestHeader*>(slot.request_buffer.data());
  request->magic = mutation_request ? service::storage_owner::kMutationMagic
                                    : service::storage_owner::kInsertMagic;
  request->dim = config_.dim;
  request->owner_storage = owner_storage;
  request->source_client = cm_.client_id;
  request->item_count = item_count;
  request->vector_dtype = static_cast<u32>(VamanaNode::vector_dtype());
  request->vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
  request->anchor_hint_count = anchor_hint_count;
  request->batch_id = batch_id;

  node_t* ids = mutation_request
    ? service::storage_owner::mutation_request_ids(slot.request_buffer.data())
    : service::storage_owner::request_ids(slot.request_buffer.data());
  byte_t* vectors = mutation_request
    ? service::storage_owner::mutation_request_vectors(slot.request_buffer.data(), item_count)
    : service::storage_owner::request_vectors(slot.request_buffer.data(), item_count);
  u32* kinds = mutation_request ? service::storage_owner::mutation_request_kinds(slot.request_buffer.data())
                                : nullptr;
  u64* anchor_hints = mutation_request
    ? service::storage_owner::mutation_request_anchor_hints(slot.request_buffer.data(), item_count)
    : service::storage_owner::request_anchor_hints(slot.request_buffer.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    ids[i] = slot.tasks[i]->item.id;
    if (kinds != nullptr) {
      kinds[i] = static_cast<u32>(slot.tasks[i]->kind);
    }
    byte_t* vector_output = vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
    if (slot.tasks[i]->kind == service::storage_owner::MutationKind::erase) {
      std::memset(vector_output, 0, VamanaNode::vector_bytes());
    } else {
      encode_float_vector_to_storage(slot.tasks[i]->item.values.data(), config_.dim,
                                     VamanaNode::vector_dtype(), vector_output);
    }
    for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
      anchor_hints[static_cast<size_t>(i) * anchor_hint_count + hint] =
        hint < slot.tasks[i]->anchor_hints.size()
          ? slot.tasks[i]->anchor_hints[hint].raw_address : 0;
    }
  }

  const u64 request_prepare_ns = collect_breakdown
    ? duration_ns(prepare_start, std::chrono::steady_clock::now())
    : 0;
  std::memset(slot.response_buffer.data(), 0, response_size);

  const u64 wr_id = storage_owner_wr_id(owner_storage, slot_id);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    slot.in_use = true;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.completion_claimed = false;
    slot.gpu_reserved_items = persistent_search_ == nullptr ? 0 : item_count;
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
      const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::microseconds(config_.update_visibility_us);
      while (std::chrono::steady_clock::now() < deadline) {
        if (!poll_completions()) std::this_thread::yield();
      }
      complete_ready_storage_owner_slots();
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

    std::vector<gpu_search::DeltaMutation> mutations;
    mutations.reserve(ready_items);
    std::vector<u64> invalidated_graph_nodes;
    invalidated_graph_nodes.reserve(static_cast<size_t>(ready_items) * config_.R);
    for (const auto& [owner_storage, slot_id] : ready_slots) {
      const auto& slot = storage_insert_owners_[owner_storage]->slots[slot_id];
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
      if (response_ok) {
        const u32 invalidation_count =
          *service::storage_owner::response_invalidation_count(
            slot.response_buffer.data(), slot.item_count);
        response_ok = invalidation_count <=
          service::storage_owner::response_invalidation_capacity(slot.item_count);
      }
      if (!response_ok) continue;
      const u32 invalidation_count =
        *service::storage_owner::response_invalidation_count(
          slot.response_buffer.data(), slot.item_count);
      const u64* invalidated_raws =
        service::storage_owner::response_invalidated_raws(
          slot.response_buffer.data(), slot.item_count);
      for (u32 index = 0; index < invalidation_count; ++index) {
        if (invalidated_raws[index] != 0) {
          invalidated_graph_nodes.push_back(invalidated_raws[index]);
        }
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
      for (u32 item = 0; item < slot.item_count; ++item) {
        if (statuses[item] != 0) continue;
        gpu_search::DeltaMutation mutation;
        mutation.id = slot.tasks[item]->item.id;
        mutation.kind = slot.tasks[item]->kind;
        mutation.generation = results[item].generation;
        mutation.remote_node = results[item].new_rptr_raw;
        mutation.old_remote_node = results[item].old_rptr_raw;
        mutation.anchor_hint = slot.tasks[item]->anchor_bucket_hint.raw_address;
        mutation.maintenance_sequence = results[item].maintenance_sequence;
        mutation.owner_storage = owner_storage;
        mutation.enqueued_at = slot.response_completed_at;
        if (mutation.kind != service::storage_owner::MutationKind::erase) {
          const byte_t* vector = request_vectors +
            static_cast<size_t>(item) * VamanaNode::vector_bytes();
          mutation.vector.assign(vector, vector + VamanaNode::vector_bytes());
        }
        mutations.push_back(std::move(mutation));
      }
    }

    bool gpu_visible = true;
    if (persistent_search_ != nullptr && !mutations.empty()) {
      try {
        const u64 epoch = persistent_search_->delta().reserve_epoch();
        gpu_visible = persistent_search_->publish_mutations(
          std::move(mutations), epoch, invalidated_graph_nodes);
      } catch (const std::exception& error) {
        gpu_visible = false;
        persistent_search_->mark_committed_mutation_gap(error.what());
        static std::atomic<u32> gpu_delta_failure_logs{0};
        const u32 log_index = gpu_delta_failure_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 16) {
          std::cerr << "[storage-owner] committed mutation batch was not published to GPU delta: "
                    << error.what() << std::endl;
        }
      }
    }

    for (const auto& [owner_storage, slot_id] : ready_slots) {
      auto& state = *storage_insert_owners_[owner_storage];
      std::lock_guard<std::mutex> lock(state.mutex);
      maybe_release_storage_owner_slot_locked(state, state.slots[slot_id], gpu_visible);
    }
  }
}

void ComputeService::maybe_release_storage_owner_slot_locked(
    StorageOwnerSenderState& state,
    StorageOwnerRpcSlot& slot,
    bool gpu_visible) {
  if (!slot.in_use || !slot.send_done || !slot.response_done) {
    return;
  }

  if (!slot.results_completed) {
    const auto* response =
      reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(slot.response_buffer.data());
    const auto* request =
      reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(slot.request_buffer.data());
    const u32* statuses = service::storage_owner::response_statuses(slot.response_buffer.data());
    const auto* mutation_results = service::storage_owner::response_mutation_results(
      slot.response_buffer.data(), slot.item_count);
    bool response_ok = (response->magic == service::storage_owner::kInsertMagic ||
                        response->magic == service::storage_owner::kMutationMagic) &&
                       response->magic == request->magic &&
                       response->owner_storage == slot.owner_storage &&
                       response->batch_id == slot.batch_id &&
                       response->item_count == slot.item_count;
    u32 invalidation_count = 0;
    if (response_ok) {
      invalidation_count = *service::storage_owner::response_invalidation_count(
        slot.response_buffer.data(), slot.item_count);
      if (invalidation_count >
          service::storage_owner::response_invalidation_capacity(slot.item_count)) {
        response_ok = false;
        invalidation_count = 0;
      }
    }
    bool collect_breakdown = false;
    for (const auto& sample : slot.samples) {
      if (sample && sample->collects_breakdown()) {
        collect_breakdown = true;
        break;
      }
    }
    const auto* breakdown = collect_breakdown && response_ok
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
    const u64 send_ns = collect_breakdown
      ? duration_ns_clamped(slot.send_posted_at, slot.send_completed_at)
      : 0;
    const u64 response_wait_ns = collect_breakdown
      ? duration_ns_clamped(slot.send_completed_at, slot.response_completed_at)
      : 0;
    const u64 response_wait_unaccounted_ns = collect_breakdown &&
        response_wait_ns > memory_breakdown_ns
      ? response_wait_ns - memory_breakdown_ns : 0;
    vec<bool> storage_ok(slot.item_count, false);
    for (u32 i = 0; i < slot.item_count; ++i) {
      storage_ok[i] = response_ok && statuses[i] == 0;
    }
    const auto finished_at = persistent_search_ == nullptr
      ? slot.response_completed_at : std::chrono::steady_clock::now();

    for (u32 i = 0; i < slot.item_count; ++i) {
      const bool committed = storage_ok[i];
      const bool ok = committed && gpu_visible;
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
        slot.samples[i]->mark_finished(finished_at);
      }
      if (committed) {
        const auto& result = mutation_results[i];
        if (slot.tasks[i]->kind == service::storage_owner::MutationKind::erase) {
          publish_compute_side_id(slot.tasks[i]->item.id,
                                  RemotePtr{result.old_rptr_raw}, true, slot.owner_storage);
        } else {
          publish_compute_side_id(slot.tasks[i]->item.id,
                                  RemotePtr{result.new_rptr_raw}, false, slot.owner_storage);
        }
      }
      slot.tasks[i]->result.set_value(ok);
    }
    slot.results_completed = true;
  }

  const u32 slot_id = slot.slot_id;
  const u64 batch_id = slot.batch_id;
  if (persistent_search_ != nullptr && slot.gpu_reserved_items != 0) {
    persistent_search_->release_mutation_capacity(slot.gpu_reserved_items);
  }
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
