template <class Distance>
typename ComputeService<Distance>::LocalMainSearchOutput ComputeService<Distance>::search_local_result(
  const vec<element_t>& query, u32 k) {
  if (query.size() != config_.dim) {
    throw std::invalid_argument("search dimension mismatch");
  }

  std::shared_ptr<service::breakdown::Sample> sample;
  if (breakdown_enabled_) {
    sample = std::make_shared<service::breakdown::Sample>(service::breakdown::Operation::query);
    sample->enqueued_at = std::chrono::steady_clock::now();
  }

  auto* request = new service::QueryRequest{query, k, {}, std::chrono::steady_clock::now(), sample};
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

