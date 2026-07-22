#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

ComputeService::Status ComputeService::status() const {
  return {
    .state = "running",
    .vectors_inserted = vectors_inserted_.load(std::memory_order_relaxed),
    .dimension = config_.dim,
    .threads = config_.num_threads,
  };
}

void ComputeService::reset_breakdown_state() {
  completed_breakdown_report_.reset();
  breakdown_enabled_.store(config_.enable_breakdown,
                           std::memory_order_release);
  persistent_search_->reset_telemetry();
  storage_insert_late_rpc_completions_.store(0, std::memory_order_relaxed);
  storage_owner_submitted_batches_.store(0, std::memory_order_relaxed);
  storage_owner_submitted_items_.store(0, std::memory_order_relaxed);
  storage_owner_completed_batches_.store(0, std::memory_order_relaxed);
  storage_owner_completed_items_.store(0, std::memory_order_relaxed);
  storage_owner_completed_rpc_wall_ns_.store(0, std::memory_order_relaxed);
  storage_owner_max_rpc_wall_ns_.store(0, std::memory_order_relaxed);
}

void ComputeService::clear_thread_statistics() {
}

service::breakdown::Report ComputeService::collect_breakdown_report() const {
  return completed_breakdown_report_.collect();
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
    lib_assert(response.vector_id_namespace_size ==
                 config_.vector_id_namespace_size,
               "vector ID namespace mismatch on storage node " +
                 std::to_string(server) + ": compute=" +
                 std::to_string(config_.vector_id_namespace_size) +
                 " storage=" +
                 std::to_string(response.vector_id_namespace_size));
  }
}

bool ComputeService::validate_index_metadata(
    const filepath_t& index_prefix, str* error_message) {
  service::index_metadata::Metadata metadata;
  if (!service::index_metadata::load_metadata(index_prefix, metadata, error_message)) {
    return false;
  }
  if (num_servers_ == 0 ||
      num_servers_ > RemotePtr::MEMORY_NODE_MASK + 1 ||
      metadata.schema_version != gpu_search::format::kMetadataSchemaVersion ||
      metadata.node_layout != "plain" ||
      metadata.storage_format != "vamana_tagged_v2" ||
      metadata.centroid_state_format !=
        "physical_shard_centroid_v2_bound" ||
      metadata.navigation_quantizer != "opq_pq" ||
      metadata.navigation_format != "opq_pq_graph_v1" ||
      metadata.navigation_code_bytes == 0 ||
      metadata.navigation_code_bytes != metadata.pq_subquantizers ||
      metadata.pq_bits != 8 || metadata.navigation_model_checksum == 0 ||
      metadata.index_build_fingerprint == 0 ||
      metadata.shard_build_fingerprints.size() != num_servers_ ||
      std::find(metadata.shard_build_fingerprints.begin(),
                metadata.shard_build_fingerprints.end(), 0) !=
        metadata.shard_build_fingerprints.end() ||
      metadata.dim != config_.dim || metadata.R != config_.R ||
      metadata.num_vectors != config_.max_vectors ||
      metadata.num_memory_nodes != num_servers_) {
    if (error_message != nullptr) {
      *error_message = "index is not a compatible schema-16 OPQ/PQ GPU index";
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
      metadata.slot_incarnation_offset !=
        VamanaNode::offset_slot_incarnation() ||
      metadata.remote_ptr_format !=
        "tagged_inc24_shard6_off34x16_v1" ||
      metadata.hot_graph_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
      metadata.hot_graph_entry_size != VamanaNode::hot_graph_entry_size() ||
      metadata.hot_graph_offsets.size() != num_servers_ ||
      metadata.hot_graph_entry_counts.size() != num_servers_ ||
      metadata.hot_graph_dynamic_base_offsets.size() != num_servers_ ||
      metadata.storage_control_remote_offsets.size() != num_servers_ ||
      metadata.dynamic_node_base_offsets.size() != num_servers_ ||
      metadata.navigation_code_remote_offsets.size() != num_servers_ ||
      metadata.navigation_code_region_bytes.size() != num_servers_ ||
      metadata.hot_graph_dynamic_record_bytes <
        metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size ||
      metadata.hot_graph_dynamic_hot_offset < VamanaNode::total_size() ||
      metadata.dynamic_navigation_code_offset <
        metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size ||
      metadata.dynamic_navigation_code_validation_bytes !=
        VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES ||
      metadata.hot_graph_dynamic_record_bytes <
        metadata.dynamic_navigation_code_offset +
          metadata.dynamic_navigation_code_validation_bytes +
          metadata.navigation_code_bytes) {
    if (error_message != nullptr) *error_message = "index storage layout mismatch";
    return false;
  }
  VamanaNode::configure_hot_graph(
    metadata.hot_graph_offsets, metadata.hot_graph_entry_counts,
    metadata.hot_graph_entry_size, metadata.hot_graph_shard_bits,
    metadata.dynamic_node_base_offsets,
    metadata.hot_graph_dynamic_record_bytes,
    metadata.hot_graph_dynamic_hot_offset,
    metadata.dynamic_navigation_code_offset,
    metadata.navigation_code_bytes);
  if (!VamanaNode::HAS_HOT_GRAPH) {
    if (error_message != nullptr) *error_message = "failed to enable compact graph layout";
    return false;
  }
  print_status("loaded schema-16 GPU index metadata from " + index_prefix.string() +
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
