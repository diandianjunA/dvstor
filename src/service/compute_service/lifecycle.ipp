template <class Distance>
ComputeService<Distance>::ComputeService(const Configuration& config, bool shutdown_remote_on_stop)
    : config_(config),
      context_(config_),
      cm_(context_, config_),
      num_servers_(config_.num_server_nodes()),
      shutdown_remote_on_stop_(shutdown_remote_on_stop) {
  init_remote_tokens();
  cm_.connect();

  if (!config_.disable_thread_pinning) {
    const u32 core = core_assignment_.get_available_core();
    pin_main_thread(core);
    print_status("pinned main thread to core " + std::to_string(core));
  }

  if (cm_.is_initiator) {
    const bool gpu_rdma = config_.use_gpu_persistent_search() &&
      config_.gpu_rdma_backend != "local";
    const u32 gpu_rdma_qps = gpu_rdma
      ? config_.gpu_rdma_qps * (config_.gpu_rdma_backend == "gpunetio" ? 2u : 1u)
      : 0;
    configuration::Parameters p{
      config_.num_threads,
      config_.use_gpu_persistent_search(),
      config_.routing,
      config_.effective_rdma_qp_pool_size(),
      gpu_rdma_qps};
    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&p, sizeof(configuration::Parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  receive_remote_access_tokens();

  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(vamana::StorageFormat::aos_v1);
  service::index_metadata::Metadata startup_metadata;
  bool have_startup_metadata = false;
  if (config_.load_index) {
    const filepath_t startup_prefix = config_.resolved_index_prefix();
    const filepath_t meta_file = filepath_t(startup_prefix.string() + ".meta.json");
    if (!startup_prefix.empty() && std::filesystem::exists(meta_file)) {
      service::index_metadata::Metadata metadata;
      str metadata_error;
      lib_assert(service::index_metadata::load_metadata(startup_prefix, metadata, &metadata_error), metadata_error);
      startup_metadata = metadata;
      have_startup_metadata = true;
      if (config_.vector_data_type != "auto" && config_.resolved_vector_dtype() != metadata.vector_dtype) {
        lib_failure("configured vector-data-type=" + config_.vector_data_type +
                    " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
      }
      config_.vector_data_type = vector_dtype_name(metadata.vector_dtype);
      const auto storage_format = vamana::parse_storage_format(metadata.storage_format);
      lib_assert(storage_format.has_value() && metadata.schema_version == 13,
                 "index storage format is obsolete; rebuild with the current offline builder");
      VamanaNode::set_storage_format(*storage_format);
      VamanaNode::init_static_storage(config_.dim, config_.R, metadata.vector_dtype);
      if (metadata.node_layout == "rabitq") {
        lib_assert(metadata.rabitq_centroid.size() == metadata.dim,
                   "RaBitQ index metadata has a missing or invalid centroid");
        VamanaNode::enable_rabitq();
        VamanaNode::set_rabitq_centroid(metadata.rabitq_centroid);
        lib_assert(metadata.rabitq_code_bits == VamanaNode::rabitq_code_bits() &&
                   metadata.rabitq_entry_size == VamanaNode::rabitq_entry_size(),
                   "RaBitQ index code layout does not match the runtime dimension");
      } else if (config_.use_rabitq) {
        lib_failure("--use-rabitq requires an index built with --use-rabitq");
      }
      lib_assert(metadata.vector_component_size == VamanaNode::vector_component_size(),
                 "index metadata vector component size mismatch on compute node");
      lib_assert(metadata.vector_bytes == VamanaNode::vector_bytes(),
                 "index metadata vector byte size mismatch on compute node");
      lib_assert(metadata.node_size == VamanaNode::total_size(),
                 "index metadata node size mismatch on compute node");
      lib_assert(metadata.graph_hot_bytes == VamanaNode::graph_hot_bytes() &&
                 metadata.vector_offset == VamanaNode::offset_vector() &&
                 metadata.neighbors_offset == VamanaNode::offset_neighbors() &&
                 metadata.rabitq_offset == (VamanaNode::HAS_RABITQ_CODE ? VamanaNode::offset_rabitq_code() : 0),
                 "index metadata storage offsets mismatch on compute node");
      if (*storage_format == vamana::StorageFormat::compact_v1) {
        lib_assert(metadata.hot_graph_pointer_bytes == vamana::hot_graph::kCompactPointerBytes &&
                   metadata.hot_graph_entry_size == VamanaNode::hot_graph_entry_size() &&
                   metadata.hot_graph_offsets.size() == num_servers_ &&
                   metadata.hot_graph_entry_counts.size() == num_servers_,
                   "index hot graph metadata mismatch on compute node");
        lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_servers_ &&
                   metadata.hot_graph_dynamic_record_bytes >=
                     metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
                   metadata.hot_graph_dynamic_hot_offset >= VamanaNode::total_size(),
                   "index dynamic hot graph metadata mismatch on compute node");
        VamanaNode::configure_hot_graph(metadata.hot_graph_offsets,
                                        metadata.hot_graph_entry_counts,
                                        metadata.hot_graph_entry_size,
                                        metadata.hot_graph_shard_bits,
                                        2u,
                                        metadata.hot_graph_dynamic_base_offsets,
                                        metadata.hot_graph_dynamic_record_bytes,
                                        metadata.hot_graph_dynamic_hot_offset);
        lib_assert(VamanaNode::HAS_HOT_GRAPH, "failed to enable compact hot graph on compute node");
      }
    }
  }

  // Initialize GPU
  gpu::gpu_init(static_cast<int>(config_.gpu_device));
  service_profile_ = resolve_service_profile();

  // Construct Vamana index
  vamana_ = std::make_unique<vamana::Vamana<Distance>>(
    config_.R, config_.beam_width, config_.beam_width_construction,
    config_.alpha, config_.k, config_.dim, config_.resolved_vector_dtype());
  vamana_->set_expansion_batch(config_.expansion_batch);
  vamana_->set_credit_aware_expansion(config_.credit_aware_expansion,
                                      config_.credit_aware_min_k,
                                      config_.credit_aware_max_k,
                                      config_.credit_aware_target_candidates,
                                      config_.credit_aware_max_lookahead,
                                      config_.credit_aware_cost_guard,
                                      static_cast<f32>(config_.credit_aware_cost_max_extra_ratio),
                                      config_.credit_aware_cost_probe_rounds);
  vamana_->set_observe_device_utilization(
    config_.enable_breakdown && config_.observe_device_utilization);
  vamana_->set_query_batch_size(config_.query_batch_size);
  vamana_->set_rabitq_gate(config_.rabitq_gate_width,
                           config_.rabitq_gate_max_width,
                           static_cast<f32>(config_.rabitq_gate_margin));
  vamana_->set_rabitq_runtime(config_.rabitq_coalesce_min,
                              config_.rabitq_warmup_exact_expansions,
                              config_.rabitq_audit_period,
                              config_.rabitq_strict_recall);
  vamana_->set_use_rabitq(config_.use_rabitq);
  if (vamana_->use_rabitq() && config_.load_index && !config_.use_gpu_persistent_search()) {
    const filepath_t startup_prefix = config_.resolved_index_prefix();
    rabitq_cache_ = std::make_unique<vamana::rabitq::Cache>();
    str cache_error;
    lib_assert(rabitq_cache_->load(startup_prefix, num_servers_,
                                  static_cast<u32>(VamanaNode::total_size()),
                                  static_cast<size_t>(config_.rabitq_dynamic_budget_mb) << 20,
                                  &cache_error),
               cache_error);
    vamana_->set_rabitq_cache(rabitq_cache_.get());
    print_status("RaBitQ RFQ5 cache: format " +
                 str(rabitq_cache_->full_layout() ? "full" : "budget") +
                 ", entry " + std::to_string(rabitq_cache_->entry_bytes()) +
                 " bytes, code " + std::to_string(rabitq_cache_->code_bits()) +
                 " bits, static " +
                 std::to_string(rabitq_cache_->size_bytes()) + " bytes, dynamic " +
                 std::to_string(rabitq_cache_->dynamic_size_bytes()) + " bytes, overrides " +
                 std::to_string(rabitq_cache_->override_bitmap_bytes()) + " bytes, decode " +
                 std::to_string(rabitq_cache_->decode_table_bytes()) + " bytes, NUMA " +
                 (rabitq_cache_->numa_interleaved() ? "interleaved" : "local"));
  }
  if (config_.use_storage_owner_insert() &&
      config_.storage_owner_update_mode == "local_stitch") {
    anchor_index_ = std::make_unique<vamana::anchor::Index>();
    str anchor_error;
    if (!have_startup_metadata || startup_metadata.anchor_format != "owner_anchor_v1" ||
        !anchor_index_->load(config_.resolved_index_prefix(), config_.dim, num_servers_, &anchor_error)) {
      anchor_index_.reset();
      throw std::runtime_error(
        "local-stitch storage-owner sidecar unavailable; refusing to run ALDI without anchors: " +
        anchor_error);
    } else {
      print_status("storage-owner anchors: entries=" +
                   std::to_string(anchor_index_->anchor_count()) + " memory=" +
                   std::to_string(anchor_index_->memory_bytes()) + " bytes");
    }
  }
  const str search_path = config_.use_gpu_persistent_search()
    ? "GPU-persistent RaBitQ gate + RDMA exact-distance beam"
    : (vamana_->use_rabitq() ? "RFQ5 RaBitQ cpu_gate + GPUDirect exact beam" : "exact");
  print_status(str(config_.credit_aware_expansion ? "search: credit-aware " : "search: ") +
               search_path);

  if (!config_.use_gpu_persistent_search()) {
    worker_pool_ = std::make_unique<WorkerPool>(
      config_.num_threads, config_.max_send_queue_wr,
      static_cast<u64>(config_.cn_memory_gb) * 1073741824ul);
    const RdmaReadBatchOptions rdma_batch_options{
      config_.rdma_read_batch_mode == "adaptive",
      config_.rdma_read_chain_size,
      config_.rdma_read_max_inflight_wrs,
    };
    worker_pool_->allocate_worker_threads(
      context_, cm_, remote_access_tokens_, config_.num_coroutines,
      config_.effective_rdma_qp_pool_size(), rdma_batch_options);
    const u32 query_batch_factor = std::max<u32>(1, config_.query_batch_size);
    const u64 query_frontier_batch =
      static_cast<u64>(std::max(config_.R, config_.beam_width)) *
      config_.expansion_batch * query_batch_factor;
    const u64 construction_batch = config_.beam_width_construction;
    const u64 overflow_prune_batch = static_cast<u64>(config_.R) + 1u;
    const u64 max_batch_u64 =
      std::max(std::max(query_frontier_batch, construction_batch), overflow_prune_batch);
    lib_assert(max_batch_u64 <= std::numeric_limits<u32>::max(),
               "GPU candidate batch capacity exceeds u32; reduce R, expansion-batch, or query-batch-size");
    const u32 max_batch = static_cast<u32>(max_batch_u64);
    const size_t query_buffer_bytes = std::max(
      static_cast<size_t>(config_.dim) * sizeof(element_t) * query_batch_factor,
      static_cast<size_t>(VamanaNode::rabitq_code_bits()) * sizeof(float));
    const size_t candidate_buffer_bytes = std::max(
      VamanaNode::vector_bytes(), VamanaNode::rabitq_entry_size());
    for (u32 tid = 0; tid < compute_threads().size(); ++tid) {
      auto& thread = compute_threads()[tid];
      thread->gpu_buffers.init(config_.num_coroutines,
                               config_.dim,
                               max_batch,
                               config_.R,
                               query_buffer_bytes,
                               candidate_buffer_bytes,
                               thread->ctx->context.get_protection_domain(),
                               config_.gpudirect_rdma,
                               config_.enable_breakdown && config_.observe_device_utilization);
    }
  }
  if (config_.use_gpu_persistent_search()) {
    persistent_search_ = std::make_unique<gpu_search::PersistentSearchEngine>(
      config_, context_, cm_, remote_access_tokens_);
    print_status("query engine: gpu_persistent backend=" + config_.gpu_rdma_backend +
                 " batch=" + std::to_string(config_.query_batch_min) + "/" +
                 std::to_string(config_.query_batch_target) + "/" +
                 std::to_string(config_.query_batch_max));
  }
  cm_.synchronize();
  if (have_startup_metadata && !config_.use_storage_owner_insert()) {
    (void)initialize_compute_side_idmap(config_.resolved_index_prefix(), startup_metadata);
  }

  wait_for_load_or_store();
  synchronize_clients_after_startup();
  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    routing_centroids_.assign(cm_.num_total_clients, vec<element_t>{});
    routing_inflight_.assign(cm_.num_total_clients, 0);
    if (routing_enabled()) {
      routing_centroids_[cm_.client_id] = compute_local_routing_centroid();
    }
  }

  start_workers();
  start_rpc();
  if (config_.use_storage_owner_insert()) {
    start_storage_insert_runtime();
  }
  refresh_routing_state(true);
}

template <class Distance>
ComputeService<Distance>::~ComputeService() {
  stop_storage_insert_runtime();
  stop_rpc();
  stop_workers();
  persistent_search_.reset();
  shutdown_remote_if_requested();
  if (worker_pool_ != nullptr) {
    for (auto& thread : compute_threads()) {
      thread->gpu_buffers.destroy();
    }
  }
  gpu::gpu_shutdown();
}


template <class Distance>
void ComputeService<Distance>::start_workers() {
  const u32 dim = config_.dim;
  const u32 num_insert_workers = service_profile_.insert_workers;
  const u32 num_query_workers = config_.use_gpu_persistent_search()
    ? 0 : service_profile_.query_workers;
  const u32 insert_coroutines = service_profile_.insert_coroutines;
  const u32 query_coroutines = service_profile_.query_coroutines;

  const u32 active_worker_count = num_insert_workers + num_query_workers;
  print_status("starting " + std::to_string(active_worker_count) + " service worker threads (Vamana)");
  print_status("worker split: inserts=" + std::to_string(num_insert_workers) +
               ", queries=" + std::to_string(num_query_workers) +
               " | coroutines: insert=" + std::to_string(insert_coroutines) +
               ", query=" + std::to_string(query_coroutines));
  workers_.reserve(active_worker_count);

  for (u32 tid = 0; tid < num_insert_workers; ++tid) {
    workers_.emplace_back([this, insert_coroutines, dim, tid]() {
      gpu::gpu_init(static_cast<int>(config_.gpu_device));
      service::vamana_service_schedule_inserts<Distance>(
        *vamana_, insert_queue_, shutdown_, insert_coroutines, compute_threads()[tid], dim, workers_paused_, workers_idle_count_);
    });
  }

  for (u32 tid = num_insert_workers; tid < num_insert_workers + num_query_workers; ++tid) {
    workers_.emplace_back([this, query_coroutines, dim, tid]() {
      gpu::gpu_init(static_cast<int>(config_.gpu_device));
      service::vamana_service_schedule_queries<Distance>(
        *vamana_, query_queue_, shutdown_, query_coroutines, compute_threads()[tid], dim, workers_paused_, workers_idle_count_);
    });
  }

  if (!config_.disable_thread_pinning) {
    for (u32 tid = 0; tid < workers_.size(); ++tid) {
      const u32 core = core_assignment_.get_available_core();
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core, &cpuset);
      pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t), &cpuset);
      print_status("pinned worker " + std::to_string(tid) + " to core " + std::to_string(core));
    }
  }
}

template <class Distance>
void ComputeService<Distance>::stop_workers() {
  if (stopped_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  shutdown_.store(true, std::memory_order_relaxed);
  resume_workers();

  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

template <class Distance>
void ComputeService<Distance>::pause_workers() {
  workers_paused_.store(true, std::memory_order_release);
  while (workers_idle_count_.load(std::memory_order_acquire) < workers_.size()) {
    std::this_thread::yield();
  }
}

template <class Distance>
void ComputeService<Distance>::resume_workers() {
  workers_paused_.store(false, std::memory_order_release);
}


template <class Distance>
void ComputeService<Distance>::shutdown_remote_if_requested() {
  if (!shutdown_remote_on_stop_ || !cm_.is_initiator) {
    return;
  }

  if (config_.use_storage_owner_insert()) {
    return;
  }

  send_index_command(mn_command::SHUTDOWN, "");
}
