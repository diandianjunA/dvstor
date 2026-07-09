template <class Distance>
vec<RemotePtr> ComputeService<Distance>::storage_owner_query_entry_points(
  const span<const element_t> query) const {
  vec<RemotePtr> entries;
  if (config_.storage_owner_update_mode != "local_stitch" ||
      anchor_index_ == nullptr || anchor_index_->empty()) {
    return entries;
  }

  constexpr u32 kQueryShardEntryCount = 2;
  const vec<u32> shards = anchor_index_->nearest_shards(query, kQueryShardEntryCount);
  entries.reserve(static_cast<size_t>(shards.size()) * config_.storage_owner_anchor_hints);
  hashset_t<RemotePtr> seen;
  for (const u32 shard : shards) {
    vec<RemotePtr> anchors =
      anchor_index_->nearest_anchors(query, shard, config_.storage_owner_anchor_hints);
    for (const RemotePtr& anchor : anchors) {
      if (!anchor.is_null() && seen.insert(anchor).second) {
        entries.push_back(anchor);
      }
    }
  }
  return entries;
}

template <class Distance>
typename ComputeService<Distance>::LocalMainSearchOutput ComputeService<Distance>::search_local_result(
  const vec<element_t>& query, u32 k) {
  if (query.size() != config_.dim) {
    throw std::invalid_argument("search dimension mismatch");
  }

  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query, breakdown_enabled_);
  const auto enqueued_at = std::chrono::steady_clock::now();
  sample->enqueued_at = enqueued_at;

  auto* request = new service::QueryRequest{};
  request->components = query;
  request->entry_points = storage_owner_query_entry_points(
    span<const element_t>{query.data(), query.size()});
  request->query_dtype = VectorDType::float32;
  request->k = k;
  request->enqueued_at = enqueued_at;
  request->breakdown_sample = sample;
  auto future = request->result.get_future();
  query_queue_.enqueue(request);

  service::QueryResult results = future.get();
  delete request;

  return {.results = std::move(results), .sample = std::move(sample)};
}

template <class Distance>
vec<node_t> ComputeService<Distance>::search_local(const vec<element_t>& query, u32 k) {
  LocalMainSearchOutput main = search_local_result(query, k);
  service::QueryResult& main_results = main.results;
  vec<node_t> ids;
  ids.reserve(std::min<size_t>(k, main_results.size()));
  for (size_t i = 0; i < main_results.size() && ids.size() < k; ++i) {
    ids.push_back(main_results[i].id);
  }
  if (main.sample && main.sample->finished_flag) {
    const auto finished_at = std::chrono::steady_clock::now();
    main.sample->finished_at = finished_at;
    main.sample->service_ns = static_cast<u64>(
      std::chrono::duration_cast<service::breakdown::Nanoseconds>(finished_at - main.sample->started_at).count());
    main.sample->end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<service::breakdown::Nanoseconds>(finished_at - main.sample->enqueued_at).count());
    std::lock_guard<std::mutex> lock(breakdown_mutex_);
    completed_query_samples_.push_back(*main.sample);
  }
  return ids;
}

template <class Distance>
typename ComputeService<Distance>::LocalMainSearchOutput ComputeService<Distance>::search_local_raw_result(
  VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (query_data == nullptr) {
    throw std::invalid_argument("raw query pointer is null");
  }

  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query, breakdown_enabled_);
  const auto enqueued_at = std::chrono::steady_clock::now();
  sample->enqueued_at = enqueued_at;

  auto* request = new service::QueryRequest{};
  request->raw_components.assign(query_data, query_data + vector_dtype_bytes(query_dtype, config_.dim));
  if (config_.storage_owner_update_mode == "local_stitch" &&
      anchor_index_ != nullptr && !anchor_index_->empty()) {
    vec<element_t> decoded(config_.dim);
    decode_storage_vector_to_float(query_data, query_dtype, config_.dim, decoded.data());
    request->entry_points = storage_owner_query_entry_points(
      span<const element_t>{decoded.data(), decoded.size()});
  }
  request->query_dtype = query_dtype;
  request->k = k;
  request->enqueued_at = enqueued_at;
  request->breakdown_sample = sample;
  auto future = request->result.get_future();
  query_queue_.enqueue(request);

  service::QueryResult results = future.get();
  delete request;

  return {.results = std::move(results), .sample = std::move(sample)};
}

template <class Distance>
vec<node_t> ComputeService<Distance>::search_local_raw(VectorDType query_dtype, const byte_t* query_data, u32 k) {
  LocalMainSearchOutput main = search_local_raw_result(query_dtype, query_data, k);
  service::QueryResult& main_results = main.results;
  vec<node_t> ids;
  ids.reserve(std::min<size_t>(k, main_results.size()));
  for (size_t i = 0; i < main_results.size() && ids.size() < k; ++i) {
    ids.push_back(main_results[i].id);
  }
  if (main.sample && main.sample->finished_flag) {
    const auto finished_at = std::chrono::steady_clock::now();
    main.sample->finished_at = finished_at;
    main.sample->service_ns = static_cast<u64>(
      std::chrono::duration_cast<service::breakdown::Nanoseconds>(finished_at - main.sample->started_at).count());
    main.sample->end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<service::breakdown::Nanoseconds>(finished_at - main.sample->enqueued_at).count());
    std::lock_guard<std::mutex> lock(breakdown_mutex_);
    completed_query_samples_.push_back(*main.sample);
  }
  return ids;
}

template <class Distance>
vec<node_t> ComputeService<Distance>::search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k) {
  if (dim != config_.dim) {
    throw std::invalid_argument("raw search dimension mismatch");
  }
  if (query_data == nullptr) {
    throw std::invalid_argument("raw query pointer is null");
  }

  if (routing_enabled()) {
    vec<element_t> decoded(config_.dim);
    decode_storage_vector_to_float(query_data, query_dtype, config_.dim, decoded.data());
    return search(decoded, k);
  }

  return search_local_raw(query_dtype, query_data, k);
}

template <class Distance>
vec<node_t> ComputeService<Distance>::search(const vec<element_t>& query, u32 k) {
  if (!routing_enabled()) {
    return search_local(query, k);
  }

  if (cm_.is_initiator) {
    const u32 destination = choose_destination(query);
    if (destination == cm_.client_id) {
      return search_local(query, k);
    }
  }

  const u64 request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  auto promise = std::make_shared<std::promise<vec<node_t>>>();
  auto future = promise->get_future();

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_queries_[request_id] = promise;
  }

  auto* outbound = new RpcOutbound{};
  outbound->request_id = request_id;
  outbound->top_k = std::min<u32>(k, kMaxRpcResults);
  outbound->float_payload = query;

  if (cm_.is_initiator) {
    const u32 destination = choose_destination(query);
    if (destination == cm_.client_id) {
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_queries_.erase(request_id);
      }
      delete outbound;
      return search_local(query, k);
    }

    outbound->destination_client = destination;
    outbound->type = rpc_search_request;
    outbound->origin_client = cm_.client_id;

    {
      std::lock_guard<std::mutex> lock(routing_mutex_);
      if (destination < routing_inflight_.size()) {
        ++routing_inflight_[destination];
      }
    }

  } else {
    outbound->destination_client = 0;
    outbound->type = rpc_search_proxy;
    outbound->origin_client = cm_.client_id;
  }

  enqueue_rpc(outbound);
  return future.get();
}
