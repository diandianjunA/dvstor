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
    configuration::Parameters p{config_.num_threads, false, config_.routing, config_.rdma_qp_pool_size};
    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&p, sizeof(configuration::Parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  receive_remote_access_tokens();

  if (config_.load_index) {
    const filepath_t startup_prefix = config_.resolved_index_prefix();
    const filepath_t meta_file = filepath_t(startup_prefix.string() + ".meta.json");
    if (!startup_prefix.empty() && std::filesystem::exists(meta_file)) {
      service::index_metadata::Metadata metadata;
      str metadata_error;
      lib_assert(service::index_metadata::load_metadata(startup_prefix, metadata, &metadata_error), metadata_error);
      if (config_.vector_data_type != "auto" && config_.resolved_vector_dtype() != metadata.vector_dtype) {
        lib_failure("configured vector-data-type=" + config_.vector_data_type +
                    " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
      }
      config_.vector_data_type = vector_dtype_name(metadata.vector_dtype);
    }
  }

  // Initialize GPU
  gpu::gpu_init(static_cast<int>(config_.gpu_device));
  service_profile_ = resolve_service_profile();
  print_status("search: exact");

  // Construct Vamana index
  vamana_ = std::make_unique<vamana::Vamana<Distance>>(
    config_.R, config_.beam_width, config_.beam_width_construction,
    config_.alpha, config_.k, config_.dim, config_.resolved_vector_dtype());
  vamana_->set_expansion_batch(config_.expansion_batch);
  vamana_->set_query_batch_size(config_.query_batch_size);

  worker_pool_ = std::make_unique<WorkerPool>(config_.num_threads,
                                              config_.max_send_queue_wr,
                                              static_cast<u64>(config_.cn_memory_gb) * 1073741824ul);
  worker_pool_->allocate_worker_threads(context_, cm_, remote_access_tokens_, config_.num_coroutines,
                                         config_.rdma_qp_pool_size);
  // Initialize GPU buffers for each compute thread
  const u32 max_batch = std::max(config_.beam_width * config_.expansion_batch,
                                   config_.beam_width_construction);
  for (u32 tid = 0; tid < compute_threads().size(); ++tid) {
    auto& thread = compute_threads()[tid];
    thread->gpu_buffers.init(config_.num_coroutines,
                             config_.dim,
                             max_batch,
                             config_.R,
                             static_cast<size_t>(config_.dim) * sizeof(element_t),
                             VamanaNode::vector_bytes(),
                             thread->ctx->context.get_protection_domain(),
                             config_.gpudirect_rdma);
  }
  cm_.synchronize();

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
  shutdown_remote_if_requested();
  for (auto& thread : compute_threads()) {
    thread->gpu_buffers.destroy();
  }
  gpu::gpu_shutdown();
}


template <class Distance>
void ComputeService<Distance>::start_workers() {
  const u32 num_threads = config_.num_threads;
  const u32 dim = config_.dim;
  const u32 num_insert_workers = service_profile_.insert_workers;
  const u32 num_query_workers = service_profile_.query_workers;
  const u32 insert_coroutines = service_profile_.insert_coroutines;
  const u32 query_coroutines = service_profile_.query_coroutines;

  print_status("starting " + std::to_string(num_threads) + " service worker threads (Vamana)");
  print_status("worker split: inserts=" + std::to_string(num_insert_workers) +
               ", queries=" + std::to_string(num_query_workers) +
               " | coroutines: insert=" + std::to_string(insert_coroutines) +
               ", query=" + std::to_string(query_coroutines));
  workers_.reserve(num_threads);

  for (u32 tid = 0; tid < num_insert_workers; ++tid) {
    workers_.emplace_back([this, insert_coroutines, dim, tid]() {
      gpu::gpu_init(static_cast<int>(config_.gpu_device));
      service::vamana_service_schedule_inserts<Distance>(
        *vamana_, insert_queue_, shutdown_, insert_coroutines, compute_threads()[tid], dim, workers_paused_, workers_idle_count_);
    });
  }

  for (u32 tid = num_insert_workers; tid < num_threads; ++tid) {
    workers_.emplace_back([this, query_coroutines, dim, tid]() {
      gpu::gpu_init(static_cast<int>(config_.gpu_device));
      service::vamana_service_schedule_queries<Distance>(
        *vamana_, query_queue_, shutdown_, query_coroutines, compute_threads()[tid], dim, workers_paused_, workers_idle_count_);
    });
  }

  if (!config_.disable_thread_pinning) {
    for (u32 tid = 0; tid < num_threads; ++tid) {
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
  while (workers_idle_count_.load(std::memory_order_acquire) < config_.num_threads) {
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
