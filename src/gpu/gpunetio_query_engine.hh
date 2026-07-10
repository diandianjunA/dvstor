#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/types.hh"
#include "service/breakdown.hh"

class Context;
class ClientConnectionManager;

namespace gpu {

struct GpuNetioPersistentView {
  void** qp_array{};
  void* remote_regions{};
  uint32_t remote_region_count{};
  uint32_t qps_per_node{};
  uint32_t local_mkey{};
  uint64_t local_iova_base{};
  unsigned char* data{};
  size_t data_bytes{};
  unsigned char* dump{};
};

class GpuNetioPersistentTransport {
public:
  GpuNetioPersistentTransport(const configuration::IndexConfiguration& config,
                              size_t data_bytes,
                              Context& context,
                              ClientConnectionManager& cm,
                              const MemoryRegionTokens& remote_regions);
  ~GpuNetioPersistentTransport();

  GpuNetioPersistentTransport(const GpuNetioPersistentTransport&) = delete;
  GpuNetioPersistentTransport& operator=(const GpuNetioPersistentTransport&) = delete;

  GpuNetioPersistentView view() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class GpuNetioQueryPool {
public:
  struct Resource;

  GpuNetioQueryPool(const configuration::IndexConfiguration& config,
                    u32 resource_count,
                    Context& context,
                    ClientConnectionManager& cm,
                    const MemoryRegionTokens& remote_regions);
  ~GpuNetioQueryPool();

  GpuNetioQueryPool(const GpuNetioQueryPool&) = delete;
  GpuNetioQueryPool& operator=(const GpuNetioQueryPool&) = delete;

  vec<node_t> search(const vec<element_t>& query, u32 k, service::breakdown::Sample* sample);

private:
  const configuration::IndexConfiguration& config_;
  std::vector<std::unique_ptr<Resource>> resources_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<bool> busy_;
  size_t next_resource_{0};
};

}  // namespace gpu
