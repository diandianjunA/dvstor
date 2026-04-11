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

class GpuNetioQueryPool {
public:
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
  struct Resource;

  const configuration::IndexConfiguration& config_;
  std::vector<std::unique_ptr<Resource>> resources_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<bool> busy_;
  size_t next_resource_{0};
};

}  // namespace gpu
