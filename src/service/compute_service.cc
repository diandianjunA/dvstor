#include "service/compute_service.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "common/debug.hh"
#include "coroutine.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"
#include "service/storage_owner_client_helpers.hh"

#include <cuda_runtime.h>

namespace {

constexpr u32 kRpcMagic = 0x53484e57;  // "SHNW"
constexpr u32 kRpcVersion = 1;
constexpr u32 kInitialRpcRecvsPerPeer = 8;
constexpr u32 kMaxRpcResults = 512;

MinorCoroutine read_medoid_probe(RemotePtr& medoid_ptr, s_ptr<VamanaNode>& node, const u_ptr<ComputeThread>& thread) {
  medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
  if (!medoid_ptr.is_null()) {
    node = co_await rdma::vamana::read_vamana_node(medoid_ptr, thread);
  }
}

using service::storage_owner_client::add_storage_owner_breakdown;
using service::storage_owner_client::add_storage_owner_sender_breakdown;
using service::storage_owner_client::duration_ns;
using service::storage_owner_client::duration_ns_clamped;
using service::storage_owner_client::per_item_ns;
using service::storage_owner_client::storage_owner_wr_id;

}  // namespace

#include "service/compute_service/lifecycle.ipp"
#include "service/compute_service/storage_owner_insert.ipp"
#include "service/compute_service/search.ipp"
#include "service/compute_service/index_commands.ipp"
#include "service/compute_service/rpc_routing.ipp"

template class ComputeService<L2Distance>;
template class ComputeService<IPDistance>;
