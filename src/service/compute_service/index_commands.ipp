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

  if (!maybe_load_rabitq_artifacts(filepath_t{path}, error_message)) {
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
      centroid[i] = medoid_node->components()[i];
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

  if (cmd == mn_command::LOAD && config_.use_rabitq_search()) {
    str artifact_error;
    lib_assert(maybe_load_rabitq_artifacts(index_prefix, &artifact_error), artifact_error);
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
bool ComputeService<Distance>::maybe_load_rabitq_artifacts(const filepath_t& index_prefix, str* error_message) {
  if (!config_.use_rabitq_search()) {
    return true;
  }

  rabitq_artifacts_ready_ = false;

  if (index_prefix.empty()) {
    if (error_message) {
      *error_message = "rabitq_gpu search requires a non-empty index prefix";
    }
    return false;
  }

  service::rabitq::Artifacts artifacts;
  if (!service::rabitq::load_artifacts(index_prefix, artifacts, error_message)) {
    return false;
  }

  if (artifacts.dim != config_.dim) {
    if (error_message) {
      *error_message = "RaBitQ artifact dim mismatch: expected " + std::to_string(config_.dim) +
                       ", got " + std::to_string(artifacts.dim);
    }
    return false;
  }
  if (artifacts.rabitq_bits != config_.rabitq_bits) {
    if (error_message) {
      *error_message = "RaBitQ artifact bits mismatch: expected " + std::to_string(config_.rabitq_bits) +
                       ", got " + std::to_string(artifacts.rabitq_bits);
    }
    return false;
  }
  if (artifacts.rabitq_size != VamanaNode::RABITQ_SIZE) {
    if (error_message) {
      *error_message = "RaBitQ artifact size mismatch: expected " + std::to_string(VamanaNode::RABITQ_SIZE) +
                       ", got " + std::to_string(artifacts.rabitq_size);
    }
    return false;
  }
  if (artifacts.R != config_.R) {
    if (error_message) {
      *error_message = "index R mismatch: expected " + std::to_string(config_.R) +
                       ", got " + std::to_string(artifacts.R) +
                       ". Runtime R must match the offline index metadata.";
    }
    return false;
  }
  if (artifacts.beam_width_construction != config_.beam_width_construction) {
    if (error_message) {
      *error_message = "index construction beam-width mismatch: expected " +
                       std::to_string(config_.beam_width_construction) + ", got " +
                       std::to_string(artifacts.beam_width_construction);
    }
    return false;
  }
  if (artifacts.node_size != VamanaNode::total_size()) {
    if (error_message) {
      *error_message = "index node size mismatch: expected " + std::to_string(VamanaNode::total_size()) +
                       ", got " + std::to_string(artifacts.node_size) +
                       ". Runtime layout must match the offline index metadata.";
    }
    return false;
  }
  if (artifacts.num_memory_nodes != num_servers_) {
    if (error_message) {
      *error_message = "RaBitQ artifact memory-node count mismatch: expected " + std::to_string(num_servers_) +
                       ", got " + std::to_string(artifacts.num_memory_nodes);
    }
    return false;
  }

  upload_rabitq_artifacts(artifacts);
  rabitq_artifacts_ready_ = true;
  print_status("loaded RaBitQ artifacts from " + index_prefix.string() +
               " (dim=" + std::to_string(artifacts.dim) +
               ", bits=" + std::to_string(artifacts.rabitq_bits) + ")");
  return true;
}

template <class Distance>
void ComputeService<Distance>::upload_rabitq_artifacts(const service::rabitq::Artifacts& artifacts) {
  for (auto& thread : compute_threads()) {
    thread->gpu_buffers.configure_rabitq(
      artifacts.rotation_matrix.data(),
      artifacts.rotated_centroid.data(),
      artifacts.dim,
      artifacts.t_const);
  }
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

