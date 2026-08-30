#include "gpu_search/navigation_bootstrapper.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <memory>
#include <limits>
#include <mutex>
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
        data_context_(config_),
        gpu_base_(reinterpret_cast<u64>(context.gpu_destination_base)),
        gpu_bytes_(context.gpu_destination_bytes), next_qp_(remote_regions_.size(), 0) {
    if (gpu_base_ == 0 || gpu_bytes_ == 0 ||
        gpu_base_ > std::numeric_limits<u64>::max() -
          (static_cast<u64>(gpu_bytes_) - 1u)) {
      throw std::invalid_argument(
        "PQ bootstrap requires a non-empty representable GPU destination");
    }
    if (remote_regions_.empty() ||
        remote_regions_.size() != connection_manager_.server_qps.size()) {
      throw std::invalid_argument(
        "PQ bootstrap remote-region and storage-connection counts differ");
    }
    for (size_t node = 0; node < remote_regions_.size(); ++node) {
      if (remote_regions_[node] == nullptr ||
          !remote_regions_[node]->address_range_valid() ||
          connection_manager_.server_qps[node] == nullptr) {
        throw std::invalid_argument(
          "PQ bootstrap received an invalid remote region at node " +
          std::to_string(node));
      }
    }
    check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
               "cudaSetDevice(PQ bootstrap init)");
    cudaDeviceProp properties{};
    check_cuda(
      cudaGetDeviceProperties(&properties, static_cast<int>(config_.gpu_device)),
      "cudaGetDeviceProperties(PQ bootstrap)");
    int gdr_supported = 0;
    check_cuda(
      cudaDeviceGetAttribute(&gdr_supported,
                             cudaDevAttrGPUDirectRDMASupported,
                             static_cast<int>(config_.gpu_device)),
      "cudaDeviceGetAttribute(PQ bootstrap GPUDirect RDMA support)");
    if (properties.unifiedAddressing == 0 || gdr_supported == 0) {
      throw std::runtime_error(
        "PQ bootstrap requires unified addressing and GPUDirect RDMA; device=" +
        std::string(properties.name) +
        " uva=" + std::to_string(properties.unifiedAddressing) +
        " gdr=" + std::to_string(gdr_supported));
    }
    int flush_options = 0;
    int writes_ordering = cudaGPUDirectRDMAWritesOrderingNone;
    check_cuda(cudaDeviceGetAttribute(
                 &flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                 static_cast<int>(config_.gpu_device)),
               "cudaDeviceGetAttribute(PQ bootstrap GPUDirect flush)");
    check_cuda(cudaDeviceGetAttribute(
                 &writes_ordering, cudaDevAttrGPUDirectRDMAWritesOrdering,
                 static_cast<int>(config_.gpu_device)),
               "cudaDeviceGetAttribute(PQ bootstrap GPUDirect ordering)");
    flush_required_ =
      writes_ordering < cudaGPUDirectRDMAWritesOrderingOwner;
    if (flush_required_ &&
        (flush_options & cudaFlushGPUDirectRDMAWritesOptionHost) == 0) {
      throw std::runtime_error(
        "GPU has no owner-visible GPUDirect RDMA write ordering and no "
        "host flush capability");
    }

    gpu_region_ = std::make_unique<LocalMemoryRegion>(
      data_context_, context.gpu_destination_base,
      context.gpu_destination_bytes);

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
  }

  void read(std::span<const NavigationRead> requests,
            std::span<i32> statuses) {
    std::lock_guard<std::mutex> lock(io_mutex_);
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
          remote_regions_[request.memory_node] == nullptr ||
          request.remote_offset > remote_regions_[request.memory_node]->bytes ||
          request.bytes >
            remote_regions_[request.memory_node]->bytes - request.remote_offset ||
          request.destination_address < gpu_base_ ||
          destination_offset > gpu_bytes_ ||
          request.bytes > gpu_bytes_ - destination_offset) {
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
          request.destination_address, request.bytes, gpu_region_->get_lkey(),
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
    check_cuda(cudaSetDevice(static_cast<int>(config_.gpu_device)),
               "cudaSetDevice(PQ bootstrap fetch)");
    if (flush_required_) {
      check_cuda(cudaDeviceFlushGPUDirectRDMAWrites(
                   cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                   cudaFlushGPUDirectRDMAWritesToOwner),
                 "cudaDeviceFlushGPUDirectRDMAWrites(PQ bootstrap fetch)");
    }
  }

  void write(std::span<const NavigationWrite> requests,
             std::span<i32> statuses) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (requests.size() != statuses.size()) {
      throw std::invalid_argument("navigation write status cardinality mismatch");
    }
    if (failed_) throw std::runtime_error("navigation RDMA backend is unavailable");
    struct QpBatch {
      DetachedQP* qp{};
      std::vector<size_t> request_indices;
    };
    std::unordered_map<DetachedQP*, size_t> batch_by_qp;
    std::vector<QpBatch> batches;
    for (size_t i = 0; i < requests.size(); ++i) {
      const NavigationWrite& request = requests[i];
      statuses[i] = -EINVAL;
      const u64 source_offset = request.source_address >= gpu_base_
        ? request.source_address - gpu_base_ : gpu_bytes_;
      if (request.memory_node >= qps_.size() || request.bytes == 0 ||
          remote_regions_[request.memory_node] == nullptr ||
          request.remote_offset > remote_regions_[request.memory_node]->bytes ||
          request.bytes >
            remote_regions_[request.memory_node]->bytes - request.remote_offset ||
          request.source_address < gpu_base_ ||
          source_offset > gpu_bytes_ ||
          request.bytes > gpu_bytes_ - source_offset) {
        continue;
      }
      auto& node_qps = qps_[request.memory_node];
      DetachedQP* qp = node_qps[next_qp_[request.memory_node]++ % node_qps.size()].get();
      auto [iterator, inserted] = batch_by_qp.emplace(qp, batches.size());
      if (inserted) batches.push_back(QpBatch{.qp = qp, .request_indices = {}});
      batches[iterator->second].request_indices.push_back(i);
    }

    for (QpBatch& batch : batches) {
      for (size_t request_index : batch.request_indices) {
        const NavigationWrite& request = requests[request_index];
        batch.qp->qp->post_send(
          request.source_address, request.bytes, gpu_region_->get_lkey(),
          IBV_WR_RDMA_WRITE, true, false, remote_regions_[request.memory_node].get(),
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
            throw std::runtime_error("navigation RDMA write timed out");
          }
          std::this_thread::yield();
          continue;
        }
        if (count < 0) {
          failed_ = true;
          throw std::runtime_error("navigation write CQ polling failed");
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
  }

  configuration::IndexConfiguration& config_;
  Context& channel_context_;
  ClientConnectionManager& connection_manager_;
  const MemoryRegionTokens& remote_regions_;
  Context data_context_;
  std::unique_ptr<LocalMemoryRegion> gpu_region_;
  u64 gpu_base_{};
  size_t gpu_bytes_{};
  std::vector<std::vector<std::unique_ptr<DetachedQP>>> qps_;
  std::vector<u32> next_qp_;
  std::mutex io_mutex_;
  bool flush_required_{};
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

void NavigationBootstrapper::write(
    const std::span<const NavigationWrite> requests,
    const std::span<i32> statuses) {
  impl_->write(requests, statuses);
}

}  // namespace gpu_search
