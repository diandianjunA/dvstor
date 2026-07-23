#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/types.hh"

namespace gpu_search {

struct NavigationRead {
  u64 remote_offset{};
  u64 destination_address{};
  u32 bytes{};
  u16 memory_node{};
};

struct NavigationWrite {
  u64 remote_offset{};
  u64 source_address{};
  u32 bytes{};
  u16 memory_node{};
};

class NavigationBootstrapper {
public:
  NavigationBootstrapper(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions,
    void* gpu_destination_base,
    size_t gpu_destination_bytes);
  ~NavigationBootstrapper();

  NavigationBootstrapper(const NavigationBootstrapper&) = delete;
  NavigationBootstrapper& operator=(const NavigationBootstrapper&) = delete;

  void read(std::span<const NavigationRead> requests,
            std::span<i32> statuses);
  void write(std::span<const NavigationWrite> requests,
             std::span<i32> statuses);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu_search
