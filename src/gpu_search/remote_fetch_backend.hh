#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "gpu_search/types.hh"

namespace gpu_search {

class RemoteFetchBackend {
public:
  virtual ~RemoteFetchBackend() = default;
  virtual RemoteBackendKind kind() const = 0;
  virtual bool gpu_initiated() const { return false; }
  virtual void fetch(std::span<const FetchDescriptor> requests,
                     std::span<i32> statuses) = 0;
};

struct RemoteFetchBackendContext {
  configuration::IndexConfiguration& config;
  Context& channel_context;
  ClientConnectionManager& connection_manager;
  const MemoryRegionTokens& remote_regions;
  void* gpu_destination_base{};
  size_t gpu_destination_bytes{};
};

std::unique_ptr<RemoteFetchBackend> create_remote_fetch_backend(
  RemoteBackendKind kind, const RemoteFetchBackendContext& context);
std::unique_ptr<RemoteFetchBackend> create_control_qp_bootstrap_backend(
  const RemoteFetchBackendContext& context);

}  // namespace gpu_search
