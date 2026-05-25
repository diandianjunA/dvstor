template <class Distance>
void ComputeService<Distance>::start_rpc() {
  if (!routing_enabled()) {
    return;
  }

  const size_t peer_count = cm_.is_initiator ? cm_.client_qps.size() : 1;
  const size_t buffer_entries = std::max<size_t>(16, peer_count * (kInitialRpcRecvsPerPeer + 8));
  const size_t msg_size = rpc_message_size();

  rpc_buffer_ = std::make_unique<byte_t[]>(buffer_entries * msg_size);
  std::memset(rpc_buffer_.get(), 0, buffer_entries * msg_size);
  rpc_region_ = std::make_unique<LocalMemoryRegion>(context_, rpc_buffer_.get(), buffer_entries * msg_size);
  rpc_freelist_.reserve(buffer_entries);
  for (idx_t i = 0; i < buffer_entries; ++i) {
    rpc_freelist_.push_back(i * msg_size);
  }

  rpc_thread_ = std::thread([this]() { run_rpc_loop(); });
}

template <class Distance>
void ComputeService<Distance>::stop_rpc() {
  if (!routing_enabled()) {
    return;
  }

  rpc_shutdown_.store(true, std::memory_order_release);
  resume_rpc();
  if (rpc_thread_.joinable()) {
    rpc_thread_.join();
  }

  RpcOutbound* leftover = nullptr;
  while (outbound_rpc_queue_.try_dequeue(leftover)) {
    delete leftover;
  }
}

template <class Distance>
void ComputeService<Distance>::pause_rpc() {
  if (!routing_enabled()) {
    return;
  }

  rpc_paused_.store(true, std::memory_order_release);
  while (!rpc_idle_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

template <class Distance>
void ComputeService<Distance>::resume_rpc() {
  rpc_paused_.store(false, std::memory_order_release);
}

template <class Distance>
void ComputeService<Distance>::post_rpc_receive(u32 peer_client) {
  if (rpc_freelist_.empty()) {
    return;
  }

  const idx_t offset = rpc_freelist_.back();
  rpc_freelist_.pop_back();
  qp_for_client(peer_client)
    .post_receive(*rpc_region_, static_cast<u32>(rpc_message_size()), encode_64bit(peer_client, offset), offset);
}

template <class Distance>
void ComputeService<Distance>::post_initial_rpc_receives() {
  if (cm_.is_initiator) {
    for (u32 remote = 1; remote < cm_.num_total_clients; ++remote) {
      for (u32 i = 0; i < kInitialRpcRecvsPerPeer; ++i) {
        post_rpc_receive(remote);
      }
    }
  } else {
    for (u32 i = 0; i < kInitialRpcRecvsPerPeer; ++i) {
      post_rpc_receive(0);
    }
  }
}

template <class Distance>
QueuePair& ComputeService<Distance>::qp_for_client(u32 client_id) {
  if (cm_.is_initiator) {
    lib_assert(client_id > 0 && client_id <= cm_.client_qps.size(), "invalid destination client id");
    return *cm_.client_qps[client_id - 1];
  }

  lib_assert(client_id == 0, "non-initiator can only communicate with initiator");
  return *cm_.initiator_qp;
}

template <class Distance>
void ComputeService<Distance>::enqueue_rpc(RpcOutbound* outbound) {
  outbound_rpc_queue_.enqueue(outbound);
}

template <class Distance>
void ComputeService<Distance>::flush_outbound_rpc() {
  RpcOutbound* outbound = nullptr;
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));

  while (outbound_rpc_queue_.try_dequeue(outbound)) {
    while (rpc_freelist_.empty()) {
      Context::poll_send_cq(send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
        const auto [_, offset] = decode_64bit(wr_id);
        rpc_freelist_.push_back(offset);
      });
      std::this_thread::yield();
    }

    const idx_t offset = rpc_freelist_.back();
    rpc_freelist_.pop_back();
    byte_t* slot = rpc_buffer_.get() + offset;

    RpcHeader header{};
    header.magic = kRpcMagic;
    header.type = outbound->type;
    header.source_client = cm_.client_id;
    header.origin_client = outbound->origin_client;
    header.request_id = outbound->request_id;
    header.top_k = outbound->top_k;

    size_t payload_bytes = 0;
    if (outbound->type == rpc_search_response) {
      header.payload_count = static_cast<u32>(outbound->id_payload.size());
      payload_bytes = outbound->id_payload.size() * sizeof(node_t);
      std::memcpy(slot + sizeof(RpcHeader), outbound->id_payload.data(), payload_bytes);

    } else {
      header.payload_count = static_cast<u32>(outbound->float_payload.size());
      payload_bytes = outbound->float_payload.size() * sizeof(element_t);
      if (payload_bytes > 0) {
        std::memcpy(slot + sizeof(RpcHeader), outbound->float_payload.data(), payload_bytes);
      }
    }

    std::memcpy(slot, &header, sizeof(header));
    qp_for_client(outbound->destination_client)
      .post_send_with_id(*rpc_region_,
                         static_cast<u32>(sizeof(RpcHeader) + payload_bytes),
                         IBV_WR_SEND,
                         encode_64bit(outbound->destination_client, offset),
                         true,
                         nullptr,
                         0,
                         offset);
    delete outbound;
  }
}

template <class Distance>
u32 ComputeService<Distance>::choose_destination(const vec<element_t>& query) const {
  if (!routing_enabled() || !cm_.is_initiator) {
    return cm_.client_id;
  }

  std::lock_guard<std::mutex> lock(routing_mutex_);
  float best_score = std::numeric_limits<float>::max();
  u32 best_client = cm_.client_id;

  for (u32 client = 0; client < routing_centroids_.size(); ++client) {
    if (routing_centroids_[client].empty()) {
      continue;
    }

    const float distance = Distance::dist(query, routing_centroids_[client], config_.dim);
    const float load_penalty = 1.0f + 0.2f * static_cast<float>(routing_inflight_[client]);
    const float score = distance * load_penalty;
    if (score < best_score) {
      best_score = score;
      best_client = client;
    }
  }

  return best_client;
}

template <class Distance>
void ComputeService<Distance>::handle_register_centroid(const RpcHeader& header, const byte_t* payload) {
  if (!cm_.is_initiator) {
    return;
  }

  vec<element_t> centroid(header.payload_count, 0.0f);
  if (!centroid.empty()) {
    std::memcpy(centroid.data(), payload, centroid.size() * sizeof(element_t));
  }

  bool first_registration = false;
  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (header.source_client < routing_centroids_.size()) {
      first_registration = routing_centroids_[header.source_client].empty();
      routing_centroids_[header.source_client] = std::move(centroid);
    }
  }

  if (first_registration) {
    registered_remote_clients_.fetch_add(1, std::memory_order_acq_rel);
    routing_cv_.notify_all();
  }

  auto* ack = new RpcOutbound{};
  ack->destination_client = header.source_client;
  ack->type = rpc_register_ack;
  ack->request_id = header.request_id;
  ack->origin_client = header.source_client;
  enqueue_rpc(ack);
}

template <class Distance>
void ComputeService<Distance>::handle_register_ack(const RpcHeader& header) {
  std::shared_ptr<std::promise<void>> promise;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_registration_acks_.find(header.request_id);
    if (it == pending_registration_acks_.end()) {
      return;
    }
    promise = it->second;
    pending_registration_acks_.erase(it);
  }

  promise->set_value();
}

template <class Distance>
void ComputeService<Distance>::handle_search_proxy(const RpcHeader& header, const byte_t* payload) {
  if (!cm_.is_initiator) {
    return;
  }

  vec<element_t> query(header.payload_count, 0.0f);
  if (!query.empty()) {
    std::memcpy(query.data(), payload, query.size() * sizeof(element_t));
  }

  const u32 destination = choose_destination(query);
  if (destination == cm_.client_id) {
    vec<node_t> ids = search_local(query, header.top_k);
    auto* response = new RpcOutbound{};
    response->destination_client = header.source_client;
    response->type = rpc_search_response;
    response->request_id = header.request_id;
    response->origin_client = header.source_client;
    response->id_payload = std::move(ids);
    enqueue_rpc(response);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (destination < routing_inflight_.size()) {
      ++routing_inflight_[destination];
    }
  }

  auto* forwarded = new RpcOutbound{};
  forwarded->destination_client = destination;
  forwarded->type = rpc_search_request;
  forwarded->request_id = header.request_id;
  forwarded->origin_client = header.source_client;
  forwarded->top_k = header.top_k;
  forwarded->float_payload = std::move(query);
  enqueue_rpc(forwarded);
}

template <class Distance>
void ComputeService<Distance>::handle_search_request(const RpcHeader& header, const byte_t* payload) {
  vec<element_t> query(header.payload_count, 0.0f);
  if (!query.empty()) {
    std::memcpy(query.data(), payload, query.size() * sizeof(element_t));
  }

  vec<node_t> ids = search_local(query, header.top_k);
  auto* response = new RpcOutbound{};
  response->destination_client = cm_.is_initiator ? header.origin_client : 0;
  response->type = rpc_search_response;
  response->request_id = header.request_id;
  response->origin_client = header.origin_client;
  response->id_payload = std::move(ids);
  enqueue_rpc(response);
}

template <class Distance>
void ComputeService<Distance>::handle_search_response(const RpcHeader& header, const byte_t* payload) {
  vec<node_t> ids(header.payload_count, 0);
  if (!ids.empty()) {
    std::memcpy(ids.data(), payload, ids.size() * sizeof(node_t));
  }

  if (cm_.is_initiator && header.source_client != cm_.client_id && header.origin_client != cm_.client_id) {
    {
      std::lock_guard<std::mutex> lock(routing_mutex_);
      if (header.source_client < routing_inflight_.size() && routing_inflight_[header.source_client] > 0) {
        --routing_inflight_[header.source_client];
      }
    }

    auto* forwarded = new RpcOutbound{};
    forwarded->destination_client = header.origin_client;
    forwarded->type = rpc_search_response;
    forwarded->request_id = header.request_id;
    forwarded->origin_client = header.origin_client;
    forwarded->id_payload = std::move(ids);
    enqueue_rpc(forwarded);
    return;
  }

  if (cm_.is_initiator && header.source_client != cm_.client_id) {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (header.source_client < routing_inflight_.size() && routing_inflight_[header.source_client] > 0) {
      --routing_inflight_[header.source_client];
    }
  }

  std::shared_ptr<std::promise<vec<node_t>>> promise;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_queries_.find(header.request_id);
    if (it == pending_queries_.end()) {
      return;
    }
    promise = it->second;
    pending_queries_.erase(it);
  }

  promise->set_value(std::move(ids));
}

template <class Distance>
void ComputeService<Distance>::handle_rpc_receive(const RpcHeader& header, const byte_t* payload) {
  if (header.magic != kRpcMagic) {
    return;
  }

  switch (header.type) {
    case rpc_register_centroid:
      handle_register_centroid(header, payload);
      break;
    case rpc_register_ack:
      handle_register_ack(header);
      break;
    case rpc_search_proxy:
      handle_search_proxy(header, payload);
      break;
    case rpc_search_request:
      handle_search_request(header, payload);
      break;
    case rpc_search_response:
      handle_search_response(header, payload);
      break;
    default:
      break;
  }
}

template <class Distance>
void ComputeService<Distance>::run_rpc_loop() {
  if (!routing_enabled()) {
    return;
  }

  post_initial_rpc_receives();

  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));

  for (;;) {
    if (rpc_shutdown_.load(std::memory_order_acquire)) {
      break;
    }

    if (rpc_paused_.load(std::memory_order_acquire)) {
      rpc_idle_.store(true, std::memory_order_release);
      std::this_thread::yield();
      continue;
    }
    rpc_idle_.store(false, std::memory_order_release);

    flush_outbound_rpc();

    const i32 num_received =
      Context::poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()), context_.get_receive_cq());
    for (i32 i = 0; i < num_received; ++i) {
      const auto [peer_client, offset] = decode_64bit(recv_wcs[i].wr_id);
      const byte_t* slot = rpc_buffer_.get() + offset;
      const auto* header = reinterpret_cast<const RpcHeader*>(slot);
      handle_rpc_receive(*header, slot + sizeof(RpcHeader));
      rpc_freelist_.push_back(offset);
      post_rpc_receive(static_cast<u32>(peer_client));
    }

    Context::poll_send_cq(send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
      const auto [_, offset] = decode_64bit(wr_id);
      rpc_freelist_.push_back(offset);
    });

    if (num_received == 0) {
      std::this_thread::yield();
    }
  }
}

template <class Distance>
void ComputeService<Distance>::refresh_routing_state(bool wait_for_remote_registration) {
  if (!routing_enabled()) {
    return;
  }

  if (cm_.is_initiator) {
    if (wait_for_remote_registration) {
      std::unique_lock<std::mutex> lock(routing_mutex_);
      routing_cv_.wait(lock, [&]() {
        return registered_remote_clients_.load(std::memory_order_acquire) >= cm_.num_total_clients - 1;
      });
    }
    return;
  }

  const u64 request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_registration_acks_[request_id] = promise;
  }

  auto* outbound = new RpcOutbound{};
  outbound->destination_client = 0;
  outbound->type = rpc_register_centroid;
  outbound->request_id = request_id;
  outbound->origin_client = cm_.client_id;
  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    outbound->float_payload = routing_centroids_[cm_.client_id];
  }
  enqueue_rpc(outbound);

  if (wait_for_remote_registration) {
    future.get();
  }
}

