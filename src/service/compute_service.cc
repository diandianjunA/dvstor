#include "service/compute_service.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "common/debug.hh"
#include "coroutine.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"

#include <cuda_runtime.h>

namespace {

constexpr u32 kRpcMagic = 0x53484e57;  // "SHNW"
constexpr u32 kRpcVersion = 1;
constexpr u32 kInitialRpcRecvsPerPeer = 8;
constexpr u32 kMaxRpcResults = 512;
constexpr u32 kRabitqSearchBeamSlack = 64;

MinorCoroutine read_medoid_probe(RemotePtr& medoid_ptr, s_ptr<VamanaNode>& node, const u_ptr<ComputeThread>& thread) {
  medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
  if (!medoid_ptr.is_null()) {
    node = co_await rdma::vamana::read_vamana_node(medoid_ptr, thread);
  }
}

}  // namespace

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
                             query_worker && config_.use_rabitq_search() ? gpu_rabitq_cache_bytes_per_query_worker : 0);
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
  refresh_routing_state(true);
}

template <class Distance>
ComputeService<Distance>::~ComputeService() {
  stop_rpc();
  stop_workers();
  shutdown_remote_if_requested();
  for (auto& thread : compute_threads()) {
    thread->gpu_buffers.destroy();
  }
  gpu::gpu_shutdown();
}

template <class Distance>
size_t ComputeService<Distance>::insert(const vec<InsertItem>& batch) {
  vec<service::InsertRequest*> requests;
  vec<std::future<bool>> futures;
  vec<std::shared_ptr<service::breakdown::Sample>> samples;
  requests.reserve(batch.size());
  futures.reserve(batch.size());
  samples.reserve(batch.size());

  for (const auto& item : batch) {
    if (item.values.size() != config_.dim) {
      throw std::invalid_argument("insert dimension mismatch");
    }

    std::shared_ptr<service::breakdown::Sample> sample;
    if (breakdown_enabled_) {
      sample = std::make_shared<service::breakdown::Sample>(service::breakdown::Operation::insert);
      sample->enqueued_at = std::chrono::steady_clock::now();
    }

    auto* request = new service::InsertRequest{item.id, item.values, {}, std::chrono::steady_clock::now(), sample};
    futures.push_back(request->result.get_future());
    requests.push_back(request);
    samples.push_back(sample);
    insert_queue_.enqueue(request);
  }

  size_t inserted = 0;
  for (size_t i = 0; i < futures.size(); ++i) {
    const bool ok = futures[i].get();
    if (ok) {
      ++inserted;
    }
    if (samples[i] && samples[i]->finished_flag) {
      std::lock_guard<std::mutex> lock(breakdown_mutex_);
      completed_insert_samples_.push_back(*samples[i]);
    }
  }
  vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);

  for (auto* request : requests) {
    delete request;
  }

  return inserted;
}

template <class Distance>
vec<node_t> ComputeService<Distance>::search_local(const vec<element_t>& query, u32 k) {
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

  vec<node_t> results = future.get();
  if (sample && sample->finished_flag) {
    std::lock_guard<std::mutex> lock(breakdown_mutex_);
    completed_query_samples_.push_back(*sample);
  }
  delete request;

  if (results.size() > k) {
    results.resize(k);
  }
  return results;
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
      service::vamana_service_schedule_inserts<Distance>(
        *vamana_, insert_queue_, shutdown_, insert_coroutines, compute_threads()[tid], dim, workers_paused_, workers_idle_count_);
    });
  }

  for (u32 tid = num_insert_workers; tid < num_threads; ++tid) {
    workers_.emplace_back([this, query_coroutines, dim, tid]() {
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
void ComputeService<Distance>::start_rpc() {
  if (!routing_enabled()) {
    return;
  }

  const size_t peer_count = cm_.is_initiator ? cm_.client_qps.size() : 1;
  const size_t buffer_entries = std::max<size_t>(16, peer_count * (kInitialRpcRecvsPerPeer + 8));
  const size_t msg_size = rpc_message_size();

  rpc_buffer_ = std::make_unique<byte_t[]>(buffer_entries * msg_size);
  std::memset(rpc_buffer_.get(), 0, buffer_entries * msg_size);
  rpc_region_ = std::make_unique<LocalMemoryRegion>(context_, rpc_buffer_.get(), buffer_entries * msg_size);
  rpc_freelist_.reserve(buffer_entries);
  for (idx_t i = 0; i < buffer_entries; ++i) {
    rpc_freelist_.push_back(i * msg_size);
  }

  rpc_thread_ = std::thread([this]() { run_rpc_loop(); });
}

template <class Distance>
void ComputeService<Distance>::stop_rpc() {
  if (!routing_enabled()) {
    return;
  }

  rpc_shutdown_.store(true, std::memory_order_release);
  resume_rpc();
  if (rpc_thread_.joinable()) {
    rpc_thread_.join();
  }

  RpcOutbound* leftover = nullptr;
  while (outbound_rpc_queue_.try_dequeue(leftover)) {
    delete leftover;
  }
}

template <class Distance>
void ComputeService<Distance>::pause_rpc() {
  if (!routing_enabled()) {
    return;
  }

  rpc_paused_.store(true, std::memory_order_release);
  while (!rpc_idle_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

template <class Distance>
void ComputeService<Distance>::resume_rpc() {
  rpc_paused_.store(false, std::memory_order_release);
}

template <class Distance>
void ComputeService<Distance>::post_rpc_receive(u32 peer_client) {
  if (rpc_freelist_.empty()) {
    return;
  }

  const idx_t offset = rpc_freelist_.back();
  rpc_freelist_.pop_back();
  qp_for_client(peer_client)
    .post_receive(*rpc_region_, static_cast<u32>(rpc_message_size()), encode_64bit(peer_client, offset), offset);
}

template <class Distance>
void ComputeService<Distance>::post_initial_rpc_receives() {
  if (cm_.is_initiator) {
    for (u32 remote = 1; remote < cm_.num_total_clients; ++remote) {
      for (u32 i = 0; i < kInitialRpcRecvsPerPeer; ++i) {
        post_rpc_receive(remote);
      }
    }
  } else {
    for (u32 i = 0; i < kInitialRpcRecvsPerPeer; ++i) {
      post_rpc_receive(0);
    }
  }
}

template <class Distance>
QueuePair& ComputeService<Distance>::qp_for_client(u32 client_id) {
  if (cm_.is_initiator) {
    lib_assert(client_id > 0 && client_id <= cm_.client_qps.size(), "invalid destination client id");
    return *cm_.client_qps[client_id - 1];
  }

  lib_assert(client_id == 0, "non-initiator can only communicate with initiator");
  return *cm_.initiator_qp;
}

template <class Distance>
void ComputeService<Distance>::enqueue_rpc(RpcOutbound* outbound) {
  outbound_rpc_queue_.enqueue(outbound);
}

template <class Distance>
void ComputeService<Distance>::flush_outbound_rpc() {
  RpcOutbound* outbound = nullptr;
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));

  while (outbound_rpc_queue_.try_dequeue(outbound)) {
    while (rpc_freelist_.empty()) {
      Context::poll_send_cq(send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
        const auto [_, offset] = decode_64bit(wr_id);
        rpc_freelist_.push_back(offset);
      });
      std::this_thread::yield();
    }

    const idx_t offset = rpc_freelist_.back();
    rpc_freelist_.pop_back();
    byte_t* slot = rpc_buffer_.get() + offset;

    RpcHeader header{};
    header.magic = kRpcMagic;
    header.type = outbound->type;
    header.source_client = cm_.client_id;
    header.origin_client = outbound->origin_client;
    header.request_id = outbound->request_id;
    header.top_k = outbound->top_k;

    size_t payload_bytes = 0;
    if (outbound->type == rpc_search_response) {
      header.payload_count = static_cast<u32>(outbound->id_payload.size());
      payload_bytes = outbound->id_payload.size() * sizeof(node_t);
      std::memcpy(slot + sizeof(RpcHeader), outbound->id_payload.data(), payload_bytes);

    } else {
      header.payload_count = static_cast<u32>(outbound->float_payload.size());
      payload_bytes = outbound->float_payload.size() * sizeof(element_t);
      if (payload_bytes > 0) {
        std::memcpy(slot + sizeof(RpcHeader), outbound->float_payload.data(), payload_bytes);
      }
    }

    std::memcpy(slot, &header, sizeof(header));
    qp_for_client(outbound->destination_client)
      .post_send_with_id(*rpc_region_,
                         static_cast<u32>(sizeof(RpcHeader) + payload_bytes),
                         IBV_WR_SEND,
                         encode_64bit(outbound->destination_client, offset),
                         true,
                         nullptr,
                         0,
                         offset);
    delete outbound;
  }
}

template <class Distance>
u32 ComputeService<Distance>::choose_destination(const vec<element_t>& query) const {
  if (!routing_enabled() || !cm_.is_initiator) {
    return cm_.client_id;
  }

  std::lock_guard<std::mutex> lock(routing_mutex_);
  float best_score = std::numeric_limits<float>::max();
  u32 best_client = cm_.client_id;

  for (u32 client = 0; client < routing_centroids_.size(); ++client) {
    if (routing_centroids_[client].empty()) {
      continue;
    }

    const float distance = Distance::dist(query, routing_centroids_[client], config_.dim);
    const float load_penalty = 1.0f + 0.2f * static_cast<float>(routing_inflight_[client]);
    const float score = distance * load_penalty;
    if (score < best_score) {
      best_score = score;
      best_client = client;
    }
  }

  return best_client;
}

template <class Distance>
void ComputeService<Distance>::handle_register_centroid(const RpcHeader& header, const byte_t* payload) {
  if (!cm_.is_initiator) {
    return;
  }

  vec<element_t> centroid(header.payload_count, 0.0f);
  if (!centroid.empty()) {
    std::memcpy(centroid.data(), payload, centroid.size() * sizeof(element_t));
  }

  bool first_registration = false;
  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (header.source_client < routing_centroids_.size()) {
      first_registration = routing_centroids_[header.source_client].empty();
      routing_centroids_[header.source_client] = std::move(centroid);
    }
  }

  if (first_registration) {
    registered_remote_clients_.fetch_add(1, std::memory_order_acq_rel);
    routing_cv_.notify_all();
  }

  auto* ack = new RpcOutbound{};
  ack->destination_client = header.source_client;
  ack->type = rpc_register_ack;
  ack->request_id = header.request_id;
  ack->origin_client = header.source_client;
  enqueue_rpc(ack);
}

template <class Distance>
void ComputeService<Distance>::handle_register_ack(const RpcHeader& header) {
  std::shared_ptr<std::promise<void>> promise;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_registration_acks_.find(header.request_id);
    if (it == pending_registration_acks_.end()) {
      return;
    }
    promise = it->second;
    pending_registration_acks_.erase(it);
  }

  promise->set_value();
}

template <class Distance>
void ComputeService<Distance>::handle_search_proxy(const RpcHeader& header, const byte_t* payload) {
  if (!cm_.is_initiator) {
    return;
  }

  vec<element_t> query(header.payload_count, 0.0f);
  if (!query.empty()) {
    std::memcpy(query.data(), payload, query.size() * sizeof(element_t));
  }

  const u32 destination = choose_destination(query);
  if (destination == cm_.client_id) {
    vec<node_t> ids = search_local(query, header.top_k);
    auto* response = new RpcOutbound{};
    response->destination_client = header.source_client;
    response->type = rpc_search_response;
    response->request_id = header.request_id;
    response->origin_client = header.source_client;
    response->id_payload = std::move(ids);
    enqueue_rpc(response);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (destination < routing_inflight_.size()) {
      ++routing_inflight_[destination];
    }
  }

  auto* forwarded = new RpcOutbound{};
  forwarded->destination_client = destination;
  forwarded->type = rpc_search_request;
  forwarded->request_id = header.request_id;
  forwarded->origin_client = header.source_client;
  forwarded->top_k = header.top_k;
  forwarded->float_payload = std::move(query);
  enqueue_rpc(forwarded);
}

template <class Distance>
void ComputeService<Distance>::handle_search_request(const RpcHeader& header, const byte_t* payload) {
  vec<element_t> query(header.payload_count, 0.0f);
  if (!query.empty()) {
    std::memcpy(query.data(), payload, query.size() * sizeof(element_t));
  }

  vec<node_t> ids = search_local(query, header.top_k);
  auto* response = new RpcOutbound{};
  response->destination_client = cm_.is_initiator ? header.origin_client : 0;
  response->type = rpc_search_response;
  response->request_id = header.request_id;
  response->origin_client = header.origin_client;
  response->id_payload = std::move(ids);
  enqueue_rpc(response);
}

template <class Distance>
void ComputeService<Distance>::handle_search_response(const RpcHeader& header, const byte_t* payload) {
  vec<node_t> ids(header.payload_count, 0);
  if (!ids.empty()) {
    std::memcpy(ids.data(), payload, ids.size() * sizeof(node_t));
  }

  if (cm_.is_initiator && header.source_client != cm_.client_id && header.origin_client != cm_.client_id) {
    {
      std::lock_guard<std::mutex> lock(routing_mutex_);
      if (header.source_client < routing_inflight_.size() && routing_inflight_[header.source_client] > 0) {
        --routing_inflight_[header.source_client];
      }
    }

    auto* forwarded = new RpcOutbound{};
    forwarded->destination_client = header.origin_client;
    forwarded->type = rpc_search_response;
    forwarded->request_id = header.request_id;
    forwarded->origin_client = header.origin_client;
    forwarded->id_payload = std::move(ids);
    enqueue_rpc(forwarded);
    return;
  }

  if (cm_.is_initiator && header.source_client != cm_.client_id) {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (header.source_client < routing_inflight_.size() && routing_inflight_[header.source_client] > 0) {
      --routing_inflight_[header.source_client];
    }
  }

  std::shared_ptr<std::promise<vec<node_t>>> promise;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_queries_.find(header.request_id);
    if (it == pending_queries_.end()) {
      return;
    }
    promise = it->second;
    pending_queries_.erase(it);
  }

  promise->set_value(std::move(ids));
}

template <class Distance>
void ComputeService<Distance>::handle_rpc_receive(const RpcHeader& header, const byte_t* payload) {
  if (header.magic != kRpcMagic) {
    return;
  }

  switch (header.type) {
    case rpc_register_centroid:
      handle_register_centroid(header, payload);
      break;
    case rpc_register_ack:
      handle_register_ack(header);
      break;
    case rpc_search_proxy:
      handle_search_proxy(header, payload);
      break;
    case rpc_search_request:
      handle_search_request(header, payload);
      break;
    case rpc_search_response:
      handle_search_response(header, payload);
      break;
    default:
      break;
  }
}

template <class Distance>
void ComputeService<Distance>::run_rpc_loop() {
  if (!routing_enabled()) {
    return;
  }

  post_initial_rpc_receives();

  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));

  for (;;) {
    if (rpc_shutdown_.load(std::memory_order_acquire)) {
      break;
    }

    if (rpc_paused_.load(std::memory_order_acquire)) {
      rpc_idle_.store(true, std::memory_order_release);
      std::this_thread::yield();
      continue;
    }
    rpc_idle_.store(false, std::memory_order_release);

    flush_outbound_rpc();

    const i32 num_received =
      Context::poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()), context_.get_receive_cq());
    for (i32 i = 0; i < num_received; ++i) {
      const auto [peer_client, offset] = decode_64bit(recv_wcs[i].wr_id);
      const byte_t* slot = rpc_buffer_.get() + offset;
      const auto* header = reinterpret_cast<const RpcHeader*>(slot);
      handle_rpc_receive(*header, slot + sizeof(RpcHeader));
      rpc_freelist_.push_back(offset);
      post_rpc_receive(static_cast<u32>(peer_client));
    }

    Context::poll_send_cq(send_wcs.data(), static_cast<i32>(send_wcs.size()), context_.get_send_cq(), [&](u64 wr_id) {
      const auto [_, offset] = decode_64bit(wr_id);
      rpc_freelist_.push_back(offset);
    });

    if (num_received == 0) {
      std::this_thread::yield();
    }
  }
}

template <class Distance>
void ComputeService<Distance>::refresh_routing_state(bool wait_for_remote_registration) {
  if (!routing_enabled()) {
    return;
  }

  if (cm_.is_initiator) {
    if (wait_for_remote_registration) {
      std::unique_lock<std::mutex> lock(routing_mutex_);
      routing_cv_.wait(lock, [&]() {
        return registered_remote_clients_.load(std::memory_order_acquire) >= cm_.num_total_clients - 1;
      });
    }
    return;
  }

  const u64 request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_registration_acks_[request_id] = promise;
  }

  auto* outbound = new RpcOutbound{};
  outbound->destination_client = 0;
  outbound->type = rpc_register_centroid;
  outbound->request_id = request_id;
  outbound->origin_client = cm_.client_id;
  {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    outbound->float_payload = routing_centroids_[cm_.client_id];
  }
  enqueue_rpc(outbound);

  if (wait_for_remote_registration) {
    future.get();
  }
}

template <class Distance>
void ComputeService<Distance>::shutdown_remote_if_requested() {
  if (!shutdown_remote_on_stop_ || !cm_.is_initiator) {
    return;
  }

  send_index_command(mn_command::SHUTDOWN, "");
}

template class ComputeService<L2Distance>;
template class ComputeService<IPDistance>;
