ComputeService::ComputeService(const Configuration& config)
    : config_(config),
      context_(config_),
      cm_(context_, config_),
      num_servers_(config_.num_server_nodes()) {
  init_remote_tokens();
  cm_.connect();

  if (!config_.disable_thread_pinning) {
    const u32 core = core_assignment_.get_available_core();
    pin_main_thread(core);
    print_status("pinned main thread to core " + std::to_string(core));
  }

  if (cm_.is_initiator) {
    const u32 gpu_rdma_qps = config_.gpu_rdma_qps * 2u;
    configuration::Parameters parameters{
      config_.num_threads,
      gpu_rdma_qps,
    };
    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&parameters, sizeof(parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  receive_remote_access_tokens();

  str metadata_error;
  const filepath_t startup_prefix = config_.resolved_index_prefix();
  lib_assert(validate_index_metadata(startup_prefix, &metadata_error), metadata_error);

  service::index_metadata::Metadata metadata;
  lib_assert(service::index_metadata::load_metadata(
               startup_prefix, metadata, &metadata_error), metadata_error);
  anchor_index_ = std::make_unique<vamana::anchor::Index>();
  str anchor_error;
  lib_assert(metadata.anchor_format == "owner_anchor_v1" &&
               anchor_index_->load(startup_prefix, config_.dim, num_servers_, &anchor_error),
             "storage-owner anchor sidecar unavailable: " + anchor_error);
  print_status("storage-owner anchors: entries=" +
               std::to_string(anchor_index_->anchor_count()) + " memory=" +
               std::to_string(anchor_index_->memory_bytes()) + " bytes");

  const cudaError_t cuda_status = cudaSetDevice(static_cast<int>(config_.gpu_device));
  lib_assert(cuda_status == cudaSuccess,
             str{"failed to select GPU: "} + cudaGetErrorString(cuda_status));
  print_status("search: GPU-persistent OPQ/PQ16 beam + final RDMA exact rerank");
  persistent_search_ = std::make_unique<gpu_search::PersistentSearchEngine>(
    config_, context_, cm_, remote_access_tokens_);
  print_status("query engine: persistent GPU + GPUNetIO batch=" +
               std::to_string(config_.query_batch_min) + "/" +
               std::to_string(config_.query_batch_target) + "/" +
               std::to_string(config_.query_batch_max));

  cm_.synchronize();
  start_storage_nodes();
  synchronize_clients_after_startup();
  start_storage_insert_runtime();
}

ComputeService::~ComputeService() {
  stop_storage_insert_runtime();
  persistent_search_.reset();
}
