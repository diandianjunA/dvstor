#pragma once

#include <cstddef>
#include <memory>

#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/types.hh"

class Context;
class ClientConnectionManager;

namespace gpu {

struct GpuNetioPersistentView {
  void** qp_array{};
  void* remote_regions{};
  uint32_t remote_region_count{};
  uint32_t qps_per_node{};
  int* qp_locks{};
  uint32_t local_mkey{};
  uint64_t local_iova_base{};
  unsigned char* data{};
  size_t data_bytes{};
  unsigned char* dump{};
};

class GpuNetioPersistentTransport {
public:
  GpuNetioPersistentTransport(
    const configuration::IndexConfiguration& config,
    size_t data_bytes,
    Context& context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions);
  ~GpuNetioPersistentTransport();

  GpuNetioPersistentTransport(const GpuNetioPersistentTransport&) = delete;
  GpuNetioPersistentTransport& operator=(
    const GpuNetioPersistentTransport&) = delete;

  GpuNetioPersistentView view() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
