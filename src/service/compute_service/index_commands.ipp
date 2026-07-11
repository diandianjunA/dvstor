ComputeService::Status ComputeService::status() const {
  return {
    .state = "running",
    .vectors_inserted = vectors_inserted_.load(std::memory_order_relaxed),
    .dimension = config_.dim,
    .threads = config_.num_threads,
  };
}

void ComputeService::publish_compute_side_id(node_t id,
                                                       RemotePtr ptr,
                                                       bool deleted,
                                                       u32 owner_storage) {
  std::lock_guard<std::mutex> lock(compute_side_idmap_mutex_);
  compute_side_idmap_[id] = ComputeSideIdEntry{ptr, deleted, owner_storage};
}

bool ComputeService::lookup_compute_side_id(
    node_t id, RemotePtr* ptr, bool* deleted) const {
  std::lock_guard<std::mutex> lock(compute_side_idmap_mutex_);
  const auto it = compute_side_idmap_.find(id);
  if (it == compute_side_idmap_.end()) return false;
  if (ptr != nullptr) *ptr = it->second.ptr;
  if (deleted != nullptr) *deleted = it->second.deleted;
  return true;
}

u32 ComputeService::storage_owner_for_id(node_t id) const {
  {
    std::lock_guard<std::mutex> lock(compute_side_idmap_mutex_);
    const auto it = compute_side_idmap_.find(id);
    if (it != compute_side_idmap_.end()) return it->second.owner_storage;
  }
  return num_servers_ == 0 ? 0 : static_cast<u32>(id % num_servers_);
}

vamana::anchor::Route ComputeService::route_storage_owner_update(
    const InsertItem& item, std::optional<u32> owner_override) const {
  if (anchor_index_ == nullptr || anchor_index_->empty()) {
    vamana::anchor::Route route;
    route.owner = owner_override.value_or(
      num_servers_ == 0 ? 0 : static_cast<u32>(item.id % num_servers_));
    return route;
  }
  return anchor_index_->route(item.values, config_.storage_owner_anchor_hints,
                              owner_override);
}

void ComputeService::reset_breakdown_state() {
  std::lock_guard<std::mutex> lock(breakdown_mutex_);
  completed_query_samples_.clear();
  completed_insert_samples_.clear();
  breakdown_enabled_ = config_.enable_breakdown;
  persistent_search_->reset_telemetry();
}

void ComputeService::clear_thread_statistics() {
}

service::breakdown::Report ComputeService::collect_breakdown_report() const {
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

void ComputeService::init_remote_tokens() {
  remote_access_tokens_.resize(num_servers_);
  for (auto& token : remote_access_tokens_) {
    token = std::make_unique<MemoryRegionToken>();
  }
}

void ComputeService::receive_remote_access_tokens() {
  print_status("receive access tokens of remote memory regions");
  for (u32 memory_node = 0; memory_node < num_servers_; ++memory_node) {
    const QP& qp = cm_.server_qps[memory_node];
    MRT& token = remote_access_tokens_[memory_node];
    LocalMemoryRegion token_region{context_, token.get(), sizeof(MemoryRegionToken)};
    qp->post_receive(token_region);
    context_.receive();
  }
}

void ComputeService::start_storage_nodes() {
  if (!cm_.is_initiator) return;
  for (u32 server = 0; server < cm_.server_qps.size(); ++server) {
    storage_startup::Request request{};
    const QP& qp = cm_.server_qps[server];
    qp->post_send_inlined(&request, sizeof(request), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
  for (u32 server = 0; server < cm_.server_qps.size(); ++server) {
    storage_startup::Response response{};
    LocalMemoryRegion response_region{context_, &response, sizeof(response)};
    cm_.server_qps[server]->post_receive(response_region);
    context_.receive();
    lib_assert(response.ready,
               "storage startup failed on node " + std::to_string(server));
  }
}

bool ComputeService::validate_index_metadata(
    const filepath_t& index_prefix, str* error_message) {
  service::index_metadata::Metadata metadata;
  if (!service::index_metadata::load_metadata(index_prefix, metadata, error_message)) {
    return false;
  }
  const bool compatible_quantizer = metadata.navigation_quantizer == "opq_pq" ||
    metadata.navigation_quantizer == "opq_pq16";
  const bool compatible_navigation = metadata.navigation_format == "opq_pq_graph_v1" ||
    metadata.navigation_format == "opq_pq16_graph_v1";
  if (metadata.schema_version != 14 || metadata.node_layout != "plain" ||
      metadata.storage_format != "vamana_compact_v1" ||
      !compatible_quantizer || !compatible_navigation ||
      metadata.navigation_code_bytes == 0 ||
      metadata.navigation_code_bytes != metadata.pq_subquantizers ||
      metadata.pq_bits != 8 || metadata.navigation_model_checksum == 0 ||
      metadata.dim != config_.dim || metadata.R != config_.R ||
      metadata.num_memory_nodes != num_servers_) {
    if (error_message != nullptr) {
      *error_message = "index is not a compatible schema-14 OPQ/PQ GPU index";
    }
    return false;
  }
  if (config_.vector_data_type != "auto" &&
      config_.resolved_vector_dtype() != metadata.vector_dtype) {
    if (error_message != nullptr) *error_message = "index vector dtype mismatch";
    return false;
  }
  config_.vector_data_type = vector_dtype_name(metadata.vector_dtype);
  VamanaNode::disable_hot_graph();
  VamanaNode::init_static_storage(config_.dim, config_.R, metadata.vector_dtype);
  if (metadata.vector_component_size != VamanaNode::vector_component_size() ||
      metadata.vector_bytes != VamanaNode::vector_bytes() ||
      metadata.node_size != VamanaNode::total_size() ||
      metadata.graph_hot_bytes != VamanaNode::graph_hot_bytes() ||
      metadata.vector_offset != VamanaNode::offset_vector() ||
      metadata.hot_graph_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
      metadata.hot_graph_entry_size != VamanaNode::hot_graph_entry_size() ||
      metadata.hot_graph_offsets.size() != num_servers_ ||
      metadata.hot_graph_entry_counts.size() != num_servers_ ||
      metadata.hot_graph_dynamic_base_offsets.size() != num_servers_ ||
      metadata.navigation_code_remote_offsets.size() != num_servers_ ||
      metadata.navigation_code_region_bytes.size() != num_servers_ ||
      metadata.hot_graph_dynamic_record_bytes <
        metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size ||
      metadata.hot_graph_dynamic_hot_offset < VamanaNode::total_size()) {
    if (error_message != nullptr) *error_message = "index storage layout mismatch";
    return false;
  }
  VamanaNode::configure_hot_graph(
    metadata.hot_graph_offsets, metadata.hot_graph_entry_counts,
    metadata.hot_graph_entry_size, metadata.hot_graph_shard_bits,
    metadata.hot_graph_dynamic_base_offsets,
    metadata.hot_graph_dynamic_record_bytes,
    metadata.hot_graph_dynamic_hot_offset);
  if (!VamanaNode::HAS_HOT_GRAPH) {
    if (error_message != nullptr) *error_message = "failed to enable compact graph layout";
    return false;
  }
  print_status("loaded schema-14 GPU index metadata from " + index_prefix.string() +
               " (OPQ/PQ" + std::to_string(metadata.pq_subquantizers) +
               ", vector_data_type=" +
               VamanaNode::vector_dtype_name() + ")");
  return true;
}

void ComputeService::synchronize_clients_after_startup() {
  constexpr bool ready = true;
  if (cm_.is_initiator) {
    for (const QP& qp : cm_.client_qps) {
      qp->post_send_inlined(&ready, sizeof(ready), IBV_WR_SEND);
    }
    if (!cm_.client_qps.empty()) {
      context_.poll_send_cq_until_completion(
        static_cast<i32>(cm_.client_qps.size()));
    }
  } else {
    bool initiator_ready{};
    LocalMemoryRegion region{context_, &initiator_ready, sizeof(initiator_ready)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();
    lib_assert(initiator_ready, "initiator startup synchronization failed");
  }
}
