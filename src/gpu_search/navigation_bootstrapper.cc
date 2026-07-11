#include "gpu_search/navigation_bootstrapper.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <library/detached_qp.hh>
#include <library/utils.hh>

namespace gpu_search {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

struct BootstrapContext {
  configuration::IndexConfiguration& config;
  Context& channel_context;
  ClientConnectionManager& connection_manager;
  const MemoryRegionTokens& remote_regions;
  void* gpu_destination_base{};
  size_t gpu_destination_bytes{};
};

}  // namespace

struct NavigationBootstrapper::Impl {
  explicit Impl(const BootstrapContext& context)
      : config_(context.config), channel_context_(context.channel_context),
        connection_manager_(context.connection_manager), remote_regions_(context.remote_regions),
        data_context_(config_), gpu_region_(data_context_, context.gpu_destination_base,
                                           context.gpu_destination_bytes),
        gpu_base_(reinterpret_cast<u64>(context.gpu_destination_base)),
        gpu_bytes_(context.gpu_destination_bytes), next_qp_(remote_regions_.size(), 0) {
    const u32 qps_per_node = std::max<u32>(1, config_.gpu_rdma_qps);
    qps_.resize(remote_regions_.size());
    for (u32 node = 0; node < qps_.size(); ++node) {
      qps_[node].reserve(qps_per_node);
      for (u32 index = 0; index < qps_per_node; ++index) {
        auto qp = std::make_unique<DetachedQP>(data_context_);
        qp->connect(channel_context_, data_context_.get_lid(),
                    connection_manager_.server_qps[node]);
        qps_[node].push_back(std::move(qp));
      }
    }
    check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
               "cudaSetDevice(PQ bootstrap init)");
    int flush_options = 0;
    check_cuda(cudaDeviceGetAttribute(
                 &flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                 static_cast<int>(config_.gpu_device)),
               "cudaDeviceGetAttribute(PQ bootstrap GPUDirect flush)");
    flush_supported_ =
      (flush_options & cudaFlushGPUDirectRDMAWritesOptionHost) != 0;
  }

  void read(std::span<const NavigationRead> requests,
            std::span<i32> statuses) {
    if (requests.size() != statuses.size()) {
      throw std::invalid_argument("PQ bootstrap status cardinality mismatch");
    }
    if (failed_) throw std::runtime_error("PQ bootstrap RDMA backend is unavailable");
    struct QpBatch {
      DetachedQP* qp{};
      std::vector<size_t> request_indices;
    };
    std::unordered_map<DetachedQP*, size_t> batch_by_qp;
    std::vector<QpBatch> batches;
    for (size_t i = 0; i < requests.size(); ++i) {
      const NavigationRead& request = requests[i];
      statuses[i] = -EINVAL;
      const u64 destination_offset = request.destination_address >= gpu_base_
        ? request.destination_address - gpu_base_ : gpu_bytes_;
      if (request.memory_node >= qps_.size() || request.bytes == 0 ||
          request.destination_address < gpu_base_ ||
          destination_offset > gpu_bytes_ || request.bytes > gpu_bytes_ - destination_offset) {
        continue;
      }
      auto& node_qps = qps_[request.memory_node];
      DetachedQP* qp = node_qps[next_qp_[request.memory_node]++ % node_qps.size()].get();
      auto [it, inserted] = batch_by_qp.emplace(qp, batches.size());
      if (inserted) batches.push_back(QpBatch{.qp = qp, .request_indices = {}});
      batches[it->second].request_indices.push_back(i);
    }

    for (QpBatch& batch : batches) {
      for (size_t request_index : batch.request_indices) {
        const NavigationRead& request = requests[request_index];
        batch.qp->qp->post_send(
          request.destination_address, request.bytes, gpu_region_.get_lkey(),
          IBV_WR_RDMA_READ, true, false, remote_regions_[request.memory_node].get(),
          request.remote_offset, 0, request_index + 1);
      }
    }
    std::vector<ibv_wc> completions(64);
    for (QpBatch& batch : batches) {
      size_t remaining = batch.request_indices.size();
      const u32 timeout_ms = std::min<u32>(config_.storage_owner_rpc_timeout_ms, 1000);
      const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
      while (remaining > 0) {
        const i32 count = batch.qp->poll_send_cq(
          completions.data(), static_cast<i32>(std::min<size_t>(completions.size(), remaining)));
        if (count == 0) {
          if (std::chrono::steady_clock::now() >= deadline) {
            failed_ = true;
            throw std::runtime_error("PQ bootstrap RDMA read timed out");
          }
          std::this_thread::yield();
          continue;
        }
        if (count < 0) {
          failed_ = true;
          throw std::runtime_error("PQ bootstrap CQ polling failed");
        }
        remaining -= static_cast<size_t>(count);
        for (i32 i = 0; i < count; ++i) {
          const size_t request_index = static_cast<size_t>(completions[i].wr_id - 1);
          if (request_index < statuses.size()) {
            statuses[request_index] = completions[i].status == IBV_WC_SUCCESS ? 1 : -EIO;
          }
        }
      }
    }
    thread_local int selected_gpu = -1;
    if (selected_gpu != static_cast<int>(config_.gpu_device)) {
      check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
                 "cudaSetDevice(PQ bootstrap fetch)");
      selected_gpu = static_cast<int>(config_.gpu_device);
    }
    if (flush_supported_) {
      check_cuda(cudaDeviceFlushGPUDirectRDMAWrites(
                   cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                   cudaFlushGPUDirectRDMAWritesToOwner),
                 "cudaDeviceFlushGPUDirectRDMAWrites(PQ bootstrap fetch)");
    }
  }

  configuration::IndexConfiguration& config_;
  Context& channel_context_;
  ClientConnectionManager& connection_manager_;
  const MemoryRegionTokens& remote_regions_;
  Context data_context_;
  LocalMemoryRegion gpu_region_;
  u64 gpu_base_{};
  size_t gpu_bytes_{};
  std::vector<std::vector<std::unique_ptr<DetachedQP>>> qps_;
  std::vector<u32> next_qp_;
  bool flush_supported_{};
  bool failed_{};
};

NavigationBootstrapper::NavigationBootstrapper(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions,
    void* gpu_destination_base,
    const size_t gpu_destination_bytes)
    : impl_(std::make_unique<Impl>(BootstrapContext{
        .config = config,
        .channel_context = channel_context,
        .connection_manager = connection_manager,
        .remote_regions = remote_regions,
        .gpu_destination_base = gpu_destination_base,
        .gpu_destination_bytes = gpu_destination_bytes,
      })) {}

NavigationBootstrapper::~NavigationBootstrapper() = default;

void NavigationBootstrapper::read(
    const std::span<const NavigationRead> requests,
    const std::span<i32> statuses) {
  impl_->read(requests, statuses);
}

}  // namespace gpu_search
