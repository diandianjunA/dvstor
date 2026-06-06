template <class Distance>
bool ComputeService<Distance>::load_index(const std::string& path, str* error_message) {
  pause_workers();
  pause_rpc();
  const auto results = send_index_command(mn_command::LOAD, path);

  for (const auto& result : results) {
    if (!result.success) {
      if (error_message) {
        *error_message = result.message;
      }
      resume_rpc();
      resume_workers();
      return false;
    }
  }

  if (!validate_index_metadata(filepath_t{path}, error_message)) {
    resume_rpc();
    resume_workers();
    return false;
  }

  if (routing_enabled()) {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    routing_centroids_[cm_.client_id] = compute_local_routing_centroid();
  }

  resume_rpc();
  refresh_routing_state(false);
  resume_workers();
  return true;
}

template <class Distance>
bool ComputeService<Distance>::store_index(const std::string& path, str* error_message) {
  pause_workers();
  pause_rpc();
  const auto results = send_index_command(mn_command::STORE, path);
  resume_rpc();
  resume_workers();

  for (const auto& result : results) {
    if (!result.success) {
      if (error_message) {
        *error_message = result.message;
      }
      return false;
    }
  }
  return true;
}

template <class Distance>
typename ComputeService<Distance>::Status ComputeService<Distance>::status() const {
  return {
    .state = "running",
    .vectors_inserted = vectors_inserted_.load(std::memory_order_relaxed),
    .dimension = config_.dim,
    .threads = config_.num_threads,
  };
}

template <class Distance>
void ComputeService<Distance>::reset_breakdown_state() {
  std::lock_guard<std::mutex> lock(breakdown_mutex_);
  completed_query_samples_.clear();
  completed_insert_samples_.clear();
  breakdown_enabled_ = true;
}

template <class Distance>
void ComputeService<Distance>::clear_thread_statistics() {
  for (auto& thread : compute_threads()) {
    thread->stats = statistics::ThreadStatistics{};
  }
}

template <class Distance>
service::breakdown::Report ComputeService<Distance>::collect_breakdown_report() const {
  service::breakdown::Report report;
  std::lock_guard<std::mutex> lock(breakdown_mutex_);
  for (const auto& sample : completed_query_samples_) {
    service::breakdown::add_sample(report.query, sample);
  }
  for (const auto& sample : completed_insert_samples_) {
    service::breakdown::add_sample(report.insert, sample);
  }
  return report;
}

template <class Distance>
bool ComputeService<Distance>::routing_enabled() const {
  return config_.routing && cm_.num_total_clients > 1;
}

template <class Distance>
size_t ComputeService<Distance>::rpc_message_size() const {
  const size_t payload_bytes =
    std::max<size_t>(config_.dim * sizeof(element_t), std::max<u32>(config_.k, kMaxRpcResults) * sizeof(node_t));
  return sizeof(RpcHeader) + payload_bytes;
}

template <class Distance>
vec<element_t> ComputeService<Distance>::compute_local_routing_centroid() const {
  vec<element_t> centroid(config_.dim, 0.0f);
  auto& thread = compute_threads()[0];
  thread->set_current_coroutine(0);

  RemotePtr medoid_ptr;
  s_ptr<VamanaNode> medoid_node;
  auto probe = read_medoid_probe(medoid_ptr, medoid_node, thread);
  while (!probe.handle.done()) {
    thread->poll_cq();
    if (thread->is_ready(0)) {
      probe.handle.resume();
    }
  }

  if (medoid_node) {
    for (idx_t i = 0; i < config_.dim; ++i) {
      centroid[i] = medoid_node->component_as_float(i);
    }
  }
  return centroid;
}

template <class Distance>
void ComputeService<Distance>::init_remote_tokens() {
  remote_access_tokens_.resize(num_servers_);
  for (auto& mrt : remote_access_tokens_) {
    mrt = std::make_unique<MemoryRegionToken>();
  }
}

template <class Distance>
void ComputeService<Distance>::receive_remote_access_tokens() {
  print_status("receive access tokens of remote memory regions");
  for (u32 memory_node = 0; memory_node < num_servers_; ++memory_node) {
    const QP& qp = cm_.server_qps[memory_node];
    MRT& mrt = remote_access_tokens_[memory_node];

    LocalMemoryRegion token_region{context_, mrt.get(), sizeof(MemoryRegionToken)};
    qp->post_receive(token_region);
    context_.receive();
  }
}

template <class Distance>
void ComputeService<Distance>::wait_for_load_or_store() {
  if (!cm_.is_initiator) return;

  mn_command::Command cmd = mn_command::NOOP;

  if (config_.load_index) {
    cmd = mn_command::LOAD;
  } else if (config_.store_index) {
    cmd = mn_command::STORE;
  }

  const size_t num_memory_servers = cm_.server_qps.size();
  const filepath_t index_prefix = cmd == mn_command::NOOP ? filepath_t{} : config_.resolved_index_prefix();

  for (idx_t i = 0; i < num_memory_servers; ++i) {
    std::string path;
    if (cmd != mn_command::NOOP) {
      path = index_path::shard_file(index_prefix, i + 1, num_memory_servers).string();
    }

    mn_command::Request req{cmd, path.size()};
    const QP& qp = cm_.server_qps[i];

    qp->post_send_inlined(&req, sizeof(mn_command::Request), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();

    if (!path.empty()) {
      qp->post_send_inlined(path.data(), path.size(), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  for (idx_t i = 0; i < num_memory_servers; ++i) {
    mn_command::Response resp{};
    LocalMemoryRegion region{context_, &resp, sizeof(mn_command::Response)};
    cm_.server_qps[i]->post_receive(region);
    context_.receive();

    str msg;
    if (resp.message_length > 0) {
      msg.resize(resp.message_length);
      LocalMemoryRegion msg_region{context_, msg.data(), resp.message_length};
      cm_.server_qps[i]->post_receive(msg_region);
      context_.receive();
    }

    const str detail = msg.empty() ? "" : ": " + msg;
    lib_assert(resp.success, "startup load/store failed on memory server " + std::to_string(i) + detail);
  }
}

template <class Distance>
typename ComputeService<Distance>::ServiceProfile ComputeService<Distance>::resolve_service_profile() const {
  ServiceProfile profile{};
  const u32 num_threads = config_.num_threads;

  if (config_.use_storage_owner_insert()) {
    profile.insert_workers = 0;
    profile.query_workers = num_threads;
    profile.insert_coroutines = 0;
    profile.query_coroutines =
      config_.query_coroutines == 0 ? std::min<u32>(config_.num_coroutines, 4) : config_.query_coroutines;
    lib_assert(profile.query_coroutines > 0 && profile.query_coroutines <= config_.num_coroutines,
               "invalid query coroutine count");
    return profile;
  }

  if (config_.insert_workers == 0 && config_.query_workers == 0) {
    profile.insert_workers = num_threads <= 1 ? 1 : std::clamp<u32>(num_threads / 2, 1, num_threads - 1);
    profile.query_workers = num_threads - profile.insert_workers;
  } else if (config_.insert_workers == 0) {
    profile.query_workers = config_.query_workers;
    profile.insert_workers = num_threads - profile.query_workers;
  } else if (config_.query_workers == 0) {
    profile.insert_workers = config_.insert_workers;
    profile.query_workers = num_threads - profile.insert_workers;
  } else {
    profile.insert_workers = config_.insert_workers;
    profile.query_workers = config_.query_workers;
  }

  lib_assert(profile.insert_workers <= num_threads, "insert worker split exceeds total threads");
  lib_assert(profile.query_workers <= num_threads, "query worker split exceeds total threads");
  lib_assert(profile.insert_workers + profile.query_workers == num_threads, "invalid worker split");
  lib_assert(profile.insert_workers > 0, "service profile requires at least one insert worker");
  lib_assert(profile.query_workers > 0, "service profile requires at least one query worker");

  profile.insert_coroutines = config_.insert_coroutines == 0 ? config_.num_coroutines : config_.insert_coroutines;
  profile.query_coroutines =
    config_.query_coroutines == 0 ? std::min<u32>(config_.num_coroutines, 4) : config_.query_coroutines;

  lib_assert(profile.insert_coroutines > 0 && profile.insert_coroutines <= config_.num_coroutines,
             "invalid insert coroutine count");
  lib_assert(profile.query_coroutines > 0 && profile.query_coroutines <= config_.num_coroutines,
             "invalid query coroutine count");
  return profile;
}

template <class Distance>
bool ComputeService<Distance>::validate_index_metadata(const filepath_t& index_prefix, str* error_message) {
  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
  if (index_prefix.empty() || !std::filesystem::exists(meta_file)) {
    VamanaNode::init_static_storage(config_.dim, config_.R, config_.resolved_vector_dtype());
    return true;
  }

  service::index_metadata::Metadata metadata;
  if (!service::index_metadata::load_metadata(index_prefix, metadata, error_message)) {
    return false;
  }

  if (config_.vector_data_type != "auto" && config_.resolved_vector_dtype() != metadata.vector_dtype) {
    if (error_message) {
      *error_message = "index vector dtype mismatch: runtime=" + config_.vector_data_type +
                       ", metadata=" + vector_dtype_name(metadata.vector_dtype);
    }
    return false;
  }

  VamanaNode::init_static_storage(config_.dim, config_.R, metadata.vector_dtype);
  config_.vector_data_type = vector_dtype_name(metadata.vector_dtype);

  if (metadata.dim != config_.dim || metadata.R != config_.R ||
      metadata.vector_component_size != VamanaNode::vector_component_size() ||
      metadata.vector_bytes != VamanaNode::vector_bytes() ||
      metadata.node_size != VamanaNode::total_size() ||
      metadata.num_memory_nodes != num_servers_) {
    if (error_message) {
      *error_message = "index metadata does not match runtime Vamana configuration";
    }
    return false;
  }
  if (metadata.beam_width_construction != 0 &&
      metadata.beam_width_construction != config_.beam_width_construction) {
    if (error_message) {
      *error_message = "index construction beam-width mismatch: expected " +
                       std::to_string(config_.beam_width_construction) + ", got " +
                       std::to_string(metadata.beam_width_construction);
    }
    return false;
  }
  if (metadata.node_layout != "standard") {
    if (error_message) {
      *error_message = "unsupported index node_layout=" + metadata.node_layout +
                       "; rebuild the index with the standard exact-vector layout";
    }
    return false;
  }

  print_status("loaded index metadata from " + index_prefix.string() +
               " (layout=" + VamanaNode::layout_name() +
               ", vector_data_type=" + VamanaNode::vector_dtype_name() + ")");
  return true;
}

template <class Distance>
void ComputeService<Distance>::synchronize_clients_after_startup() {
  constexpr bool ready = true;

  if (cm_.is_initiator) {
    for (const QP& qp : cm_.client_qps) {
      qp->post_send_inlined(&ready, sizeof(bool), IBV_WR_SEND);
    }

    if (!cm_.client_qps.empty()) {
      context_.poll_send_cq_until_completion(static_cast<i32>(cm_.client_qps.size()));
    }

  } else {
    bool initiator_ready{};
    LocalMemoryRegion region{context_, &initiator_ready, sizeof(bool)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();
    lib_assert(initiator_ready, "initiator startup synchronization failed");
  }
}

template <class Distance>
auto ComputeService<Distance>::send_index_command(mn_command::Command cmd, const std::string& path)
    -> vec<CommandResult> {
  std::lock_guard<std::mutex> lock(mn_command_mutex_);

  const size_t num_memory_servers = cm_.server_qps.size();
  vec<CommandResult> results(num_memory_servers);

  for (idx_t i = 0; i < num_memory_servers; ++i) {
    std::string node_path;
    if (!path.empty()) {
      node_path = index_path::shard_file(filepath_t{path}, i + 1, num_memory_servers).string();
    }

    mn_command::Request req{cmd, node_path.size()};
    const QP& qp = cm_.server_qps[i];

    qp->post_send_inlined(&req, sizeof(mn_command::Request), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();

    if (!node_path.empty()) {
      qp->post_send_inlined(node_path.data(), node_path.size(), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  for (idx_t i = 0; i < num_memory_servers; ++i) {
    mn_command::Response resp{};
    LocalMemoryRegion region{context_, &resp, sizeof(mn_command::Response)};
    cm_.server_qps[i]->post_receive(region);
    context_.receive();

    str msg;
    if (resp.message_length > 0) {
      msg.resize(resp.message_length);
      LocalMemoryRegion msg_region{context_, msg.data(), resp.message_length};
      cm_.server_qps[i]->post_receive(msg_region);
      context_.receive();
    }

    results[i] = {resp.success, std::move(msg)};
  }

  return results;
}
