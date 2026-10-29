#include "service/compute_service/detail.hh"

#include "gpu_search/host_orchestrated_engine.hh"
#include "gpu_search/persistent_engine.hh"

using namespace compute_service_detail;

ComputeService::ComputeService(const Configuration& config)
    : config_(config),
      context_(config_),
      cm_(context_, config_),
      num_servers_(config_.num_server_nodes()) {
  // Select and validate the configured device before any storage connection or
  // startup protocol side effect. Invalid ordinals now fail locally.
  const cudaError_t cuda_status =
    cudaSetDevice(static_cast<int>(config_.gpu_device));
  lib_assert(cuda_status == cudaSuccess,
             str{"failed to select GPU: "} + cudaGetErrorString(cuda_status));

  str metadata_error;
  const filepath_t startup_prefix = config_.resolved_index_prefix();
  lib_assert(validate_index_metadata(startup_prefix, &metadata_error),
             metadata_error);
  service::index_metadata::Metadata metadata;
  lib_assert(service::index_metadata::load_metadata(
               startup_prefix, metadata, &metadata_error),
             metadata_error);

  init_remote_tokens();
  cm_.connect();

  // Do not pin the constructor thread. POSIX threads inherit their creator's
  // affinity mask, so pinning here used to serialize every later query,
  // update, and benchmark worker onto the same CPU. Dedicated runtime threads
  // are pinned after creation by their owning subsystems instead.

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

  // Multi-compute-node startup serialization: each storage node performs its
  // GPU/bootstrap QP handshakes in client-ID order, so the initiator must
  // finish its search-engine construction (which carries those handshakes)
  // before any other client begins its own. Without this barrier the
  // concurrent control-channel handshakes race and startup fails.
  if (!cm_.is_initiator && cm_.num_total_clients > 1) {
    bool initiator_engine_ready{};
    LocalMemoryRegion region{context_, &initiator_engine_ready,
                             sizeof(initiator_engine_ready)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();
    lib_assert(initiator_engine_ready,
               "initiator search-engine startup synchronization failed");
  }

  if (config_.enable_updates) {
    print_status(
      "storage-owner authority: deterministic ID shard; "
      "physical placement: storage directory + centroid home");
  } else {
    print_status("compute updates disabled: update executor is not started");
  }

  if (config_.gpu_rdma_search_progression_mode == "coupled") {
    print_status("search: CPU-orchestrated schema-v16 OPQ/PQ" +
                 std::to_string(metadata.pq_subquantizers) +
                 " strict RDMA waves + finite CUDA scoring + exact rerank");
    search_engine_ =
      std::make_unique<gpu_search::HostOrchestratedSearchEngine>(
        config_, context_, cm_, remote_access_tokens_);
    print_status("query engine: host-orchestrated RDMA lanes=" +
                 std::to_string(config_.gpu_rdma_qps) +
                 " persistent-kernel=off GPUNetIO-query=off");
  } else {
    print_status("search: GPU-persistent OPQ/PQ" +
                 std::to_string(metadata.pq_subquantizers) +
                 " beam + final RDMA exact rerank");
    search_engine_ = std::make_unique<gpu_search::PersistentSearchEngine>(
      config_, context_, cm_, remote_access_tokens_);
    print_status("query engine: persistent GPU + GPUNetIO slots=" +
                 std::to_string(config_.gpu_query_slots));
  }

  if (cm_.is_initiator && cm_.num_total_clients > 1) {
    constexpr bool engine_ready = true;
    for (const QP& qp : cm_.client_qps) {
      qp->post_send_inlined(&engine_ready, sizeof(engine_ready), IBV_WR_SEND);
    }
    if (!cm_.client_qps.empty()) {
      context_.poll_send_cq_until_completion(
        static_cast<i32>(cm_.client_qps.size()));
    }
  }

  cm_.synchronize();
  start_storage_nodes();
  synchronize_clients_after_startup();
  if (config_.enable_updates) start_storage_insert_runtime();
}

ComputeService::~ComputeService() {
  if (config_.enable_updates) stop_storage_insert_runtime();
  search_engine_.reset();
  cm_.server_qps.clear();
  if (config_.enable_updates) release_storage_insert_runtime();
}
