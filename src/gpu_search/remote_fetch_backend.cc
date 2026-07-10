#include "gpu_search/remote_fetch_backend.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <library/detached_qp.hh>
#include <library/utils.hh>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"

namespace gpu_search {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class LocalFetchBackend final : public RemoteFetchBackend {
public:
  explicit LocalFetchBackend(const RemoteFetchBackendContext& context)
      : config_(context.config) {
    const filepath_t prefix = config_.resolved_index_prefix();
    shard_files_.reserve(config_.num_server_nodes());
    graph_page_files_.reserve(config_.num_server_nodes());
    graph_page_headers_.reserve(config_.num_server_nodes());
    for (u32 shard = 0; shard < config_.num_server_nodes(); ++shard) {
      auto input = std::make_unique<std::ifstream>(
        index_path::shard_file(prefix, shard + 1, config_.num_server_nodes()),
        std::ios::binary);
      if (!input->good()) {
        throw std::runtime_error("failed to open local fetch shard " +
          index_path::shard_file(prefix, shard + 1, config_.num_server_nodes()).string());
      }
      shard_files_.push_back(std::move(input));

      const auto pages_path = index_path::gpu_graph_pages_file(
        prefix, shard + 1, config_.num_server_nodes());
      auto pages = std::make_unique<std::ifstream>(pages_path, std::ios::binary);
      format::ShardPageFileHeader header;
      pages->read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!pages->good() || header.magic != format::kShardPagesMagic ||
          header.version != format::kVersion || header.memory_node != shard) {
        throw std::runtime_error("failed to open local graph-page shard " +
                                 pages_path.string());
      }
      graph_page_headers_.push_back(header);
      graph_page_files_.push_back(std::move(pages));
    }
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(local fetch)");
  }

  ~LocalFetchBackend() override {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  RemoteBackendKind kind() const override { return RemoteBackendKind::local; }

  void fetch(std::span<const FetchDescriptor> requests,
             std::span<i32> statuses) override {
    if (requests.size() != statuses.size()) {
      throw std::invalid_argument("local fetch status cardinality mismatch");
    }
    std::vector<std::vector<byte_t>> staging(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
      const FetchDescriptor& request = requests[i];
      statuses[i] = -EINVAL;
      if (request.memory_node >= shard_files_.size() || request.bytes == 0) continue;
      staging[i].resize(request.bytes);
      const bool graph_page = request.kind == static_cast<u8>(FetchKind::graph_page);
      auto& input = graph_page ? *graph_page_files_[request.memory_node]
                               : *shard_files_[request.memory_node];
      u64 file_offset = request.remote_offset;
      if (graph_page) {
        const auto& header = graph_page_headers_[request.memory_node];
        if (request.remote_offset < header.remote_offset ||
            request.remote_offset - header.remote_offset > header.data_bytes ||
            request.bytes > header.data_bytes -
              (request.remote_offset - header.remote_offset)) {
          continue;
        }
        file_offset = sizeof(format::ShardPageFileHeader) +
          request.remote_offset - header.remote_offset;
      }
      {
        std::lock_guard<std::mutex> lock(file_mutex_);
        input.clear();
        input.seekg(static_cast<std::streamoff>(file_offset));
        input.read(reinterpret_cast<char*>(staging[i].data()), request.bytes);
      }
      if (!input.good()) {
        statuses[i] = -EIO;
        continue;
      }
      const cudaError_t copy_status = cudaMemcpyAsync(
        reinterpret_cast<void*>(request.destination_address), staging[i].data(), request.bytes,
        cudaMemcpyHostToDevice, stream_);
      statuses[i] = copy_status == cudaSuccess ? 1 : -EIO;
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(local fetch)");
  }

private:
  configuration::IndexConfiguration& config_;
  std::vector<std::unique_ptr<std::ifstream>> shard_files_;
  std::vector<std::unique_ptr<std::ifstream>> graph_page_files_;
  std::vector<format::ShardPageFileHeader> graph_page_headers_;
  std::mutex file_mutex_;
  cudaStream_t stream_{};
};

class VerbsProxyFetchBackend final : public RemoteFetchBackend {
public:
  explicit VerbsProxyFetchBackend(const RemoteFetchBackendContext& context)
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
  }

  RemoteBackendKind kind() const override { return RemoteBackendKind::verbs_proxy; }

  void fetch(std::span<const FetchDescriptor> requests,
             std::span<i32> statuses) override {
    if (requests.size() != statuses.size()) {
      throw std::invalid_argument("verbs proxy status cardinality mismatch");
    }
    struct QpBatch {
      DetachedQP* qp{};
      std::vector<size_t> request_indices;
    };
    std::unordered_map<DetachedQP*, size_t> batch_by_qp;
    std::vector<QpBatch> batches;
    for (size_t i = 0; i < requests.size(); ++i) {
      const FetchDescriptor& request = requests[i];
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
        const FetchDescriptor& request = requests[request_index];
        batch.qp->qp->post_send(
          request.destination_address, request.bytes, gpu_region_.get_lkey(),
          IBV_WR_RDMA_READ, true, false, remote_regions_[request.memory_node].get(),
          request.remote_offset, 0, request_index + 1);
      }
    }
    std::vector<ibv_wc> completions(64);
    for (QpBatch& batch : batches) {
      size_t remaining = batch.request_indices.size();
      while (remaining > 0) {
        const i32 count = batch.qp->poll_send_cq(
          completions.data(), static_cast<i32>(std::min<size_t>(completions.size(), remaining)));
        if (count == 0) {
          std::this_thread::yield();
          continue;
        }
        if (count < 0) throw std::runtime_error("verbs proxy CQ polling failed");
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

private:
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
};

}  // namespace

std::unique_ptr<RemoteFetchBackend> create_remote_fetch_backend(
    RemoteBackendKind kind, const RemoteFetchBackendContext& context) {
  switch (kind) {
    case RemoteBackendKind::local:
      return std::make_unique<LocalFetchBackend>(context);
    case RemoteBackendKind::verbs_proxy:
      return std::make_unique<VerbsProxyFetchBackend>(context);
    case RemoteBackendKind::gpunetio:
#ifdef DVSTOR_HAVE_GPUNETIO
      throw std::runtime_error("GPUNetIO backend must be created by the direct GPU backend factory");
#else
      throw std::runtime_error("DVSTOR was built without DOCA GPUNetIO support");
#endif
  }
  throw std::runtime_error("unknown remote fetch backend");
}

}  // namespace gpu_search
