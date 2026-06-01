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
    configuration::Parameters p{config_.num_threads, config_.use_cache, config_.routing};
    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&p, sizeof(configuration::Parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  receive_remote_access_tokens();

  // Initialize GPU
  gpu::gpu_init(static_cast<int>(config_.gpu_device));
  service_profile_ = resolve_service_profile();
  print_status("search mode: " + config_.search_mode +
               ", cache=" + (config_.use_cache ? str{"on"} : str{"off"}));

  // Construct Vamana index
  vamana_ = std::make_unique<vamana::Vamana<Distance>>(
    config_.R, config_.beam_width, config_.beam_width_construction,
    config_.alpha, config_.k, config_.rabitq_bits, config_.dim, config_.use_cache, config_.use_rabitq_search());

  const size_t estimated_index_size = config_.max_vectors * VamanaNode::total_size();
  const size_t cache_size = static_cast<f32>(estimated_index_size) / 100. * config_.cache_size_ratio;

  if (config_.use_cache) {
    print_status("max cache size: " + std::to_string(cache_size));
  }

  const size_t num_cache_buckets = cache_size / VamanaNode::total_size();
  const size_t num_cooling_table_buckets = std::ceil(cache_size / VamanaNode::total_size() /
                                                     cache::COOLING_TABLE_BUCKET_ENTRIES * cache::COOLING_TABLE_RATIO);

  worker_pool_ = std::make_unique<WorkerPool>(config_.num_threads,
                                              config_.max_send_queue_wr,
                                              cache_size,
                                              num_cache_buckets,
                                              num_cooling_table_buckets,
                                              config_.use_cache,
                                              static_cast<u64>(config_.cn_memory_gb) * 1073741824ul);
  worker_pool_->allocate_worker_threads(context_, cm_, remote_access_tokens_, config_.num_coroutines);
  for (auto& thread : compute_threads()) {
    thread->set_graph_epoch_source(&graph_epoch_);
  }

  // Initialize GPU buffers for each compute thread
  const u32 search_batch =
      config_.use_rabitq_search() ? (config_.beam_width + kRabitqSearchBeamSlack) : config_.beam_width;
  const u32 max_batch = std::max(search_batch, config_.beam_width_construction);
  const u64 neighbor_cache_total_bytes = static_cast<u64>(config_.neighbor_cache_mb) * 1024ull * 1024ull;
  const u64 gpu_rabitq_cache_total_bytes = static_cast<u64>(config_.gpu_rabitq_cache_mb) * 1024ull * 1024ull;
  const u64 neighbor_cache_bytes_per_query_worker =
    service_profile_.query_workers == 0 ? 0 : neighbor_cache_total_bytes / service_profile_.query_workers;
  const u64 gpu_rabitq_cache_bytes_per_query_worker =
    service_profile_.query_workers == 0 ? 0 : gpu_rabitq_cache_total_bytes / service_profile_.query_workers;
  for (u32 tid = 0; tid < compute_threads().size(); ++tid) {
    auto& thread = compute_threads()[tid];
    const bool query_worker = tid >= service_profile_.insert_workers;
    if (query_worker && neighbor_cache_bytes_per_query_worker > 0) {
      thread->neighbor_cache.init(neighbor_cache_bytes_per_query_worker);
      print_status("neighbor cache worker " + std::to_string(tid) +
                   ": slots=" + std::to_string(thread->neighbor_cache.slot_count()));
    }
    thread->gpu_buffers.init(config_.num_coroutines,
                             config_.dim,
                             max_batch,
                             config_.R,
                             config_.rabitq_bits,
                             thread->ctx->context.get_protection_domain(),
                             config_.gpudirect_rdma,
                             query_worker && config_.use_rabitq_search() ? gpu_rabitq_cache_bytes_per_query_worker : 0,
                             config_.rabitq_cache_mode.c_str(),
                             config_.gentile_tile_slots,
                             config_.gentile_nursery_ratio,
                             config_.gentile_promotion_threshold,
                             config_.gentile_enable_promotion,
                             config_.gentile_enable_value_bin,
                             config_.gentile_enable_hit_tile_grouping);
  }
  cm_.synchronize();

  wait_for_load_or_store();
  synchronize_clients_after_startup();
  if (config_.load_index && config_.use_rabitq_search() && !rabitq_artifacts_ready_) {
    str artifact_error;
    lib_assert(maybe_load_rabitq_artifacts(config_.resolved_index_prefix(), &artifact_error), artifact_error);
  }

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
void ComputeService<Distance>::force_publish_graph_epoch() {
  std::lock_guard<std::mutex> lock(neighbor_cache_epoch_mutex_);
  pending_neighbor_cache_inserts_ = 0;
  last_neighbor_cache_epoch_publish_ = std::chrono::steady_clock::now();
  graph_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

template <class Distance>
void ComputeService<Distance>::note_graph_insertions(size_t inserted) {
  if (inserted == 0) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(neighbor_cache_epoch_mutex_);
  pending_neighbor_cache_inserts_ += inserted;

  const bool insert_threshold_met =
    pending_neighbor_cache_inserts_ >= static_cast<size_t>(config_.neighbor_cache_invalidation_inserts);
  const bool time_threshold_met =
    config_.neighbor_cache_invalidation_ms > 0 &&
    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_neighbor_cache_epoch_publish_).count() >=
      static_cast<int64_t>(config_.neighbor_cache_invalidation_ms);

  if (!insert_threshold_met && !time_threshold_met) {
    return;
  }

  pending_neighbor_cache_inserts_ = 0;
  last_neighbor_cache_epoch_publish_ = now;
  graph_epoch_.fetch_add(1, std::memory_order_acq_rel);
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
