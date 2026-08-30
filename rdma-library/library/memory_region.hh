#ifndef RDMA_LIBRARY_MEMORY_REGION_HH
#define RDMA_LIBRARY_MEMORY_REGION_HH

#include <infiniband/verbs.h>

#include <bit>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "context.hh"
#include "types.hh"

struct MemoryRegionToken {
  static constexpr u32 kWireMagic = 0x4456524du;  // "DVRM"
  static constexpr u16 kWireVersion = 1;
  static constexpr u16 kWireBytes = 32;

  u64 address;
  u32 lkey;
  u32 rkey;
  u64 bytes;
  u32 wire_magic;
  u16 wire_version;
  u16 wire_bytes;

  constexpr bool address_range_valid() const {
    return wire_magic == kWireMagic && wire_version == kWireVersion &&
      wire_bytes == kWireBytes && address != 0 &&
      bytes != 0 && address <=
      std::numeric_limits<u64>::max() - (bytes - 1);
  }

  constexpr bool contains(u64 offset, u64 length) const {
    return address_range_valid() && length != 0 && offset <= bytes &&
      length <= bytes - offset;
  }
};
static_assert(sizeof(MemoryRegionToken) == MemoryRegionToken::kWireBytes);
static_assert(std::endian::native == std::endian::little,
              "MemoryRegionToken wire protocol requires little-endian hosts");
static_assert(std::is_standard_layout_v<MemoryRegionToken>);
static_assert(std::is_trivially_copyable_v<MemoryRegionToken>);
static_assert(offsetof(MemoryRegionToken, address) == 0);
static_assert(offsetof(MemoryRegionToken, rkey) == 12);
static_assert(offsetof(MemoryRegionToken, bytes) == 16);
static_assert(offsetof(MemoryRegionToken, wire_magic) == 24);
static_assert(offsetof(MemoryRegionToken, wire_bytes) == 30);

// must be on the heap s.t. the address does not change after vector movements
using MRT = u_ptr<MemoryRegionToken>;
using MemoryRegionTokens = vec<MRT>;

// forward declaration
class Context;

class MemoryRegion {
protected:
  MemoryRegion(Context& context,
               void* data,
               size_t size_in_bytes,
               bool remote_access);

public:
  MemoryRegion(Context& context, void* data, size_t size_in_bytes);
  explicit MemoryRegion(Context& context);

  ~MemoryRegion();
  MemoryRegion(const MemoryRegion&) = delete;
  MemoryRegion& operator=(const MemoryRegion&) = delete;

  void register_memory(void* data, size_t size_in_bytes, bool remote_access);
  MemoryRegionToken createToken() const;

  u64 get_address() const { return reinterpret_cast<u64>(data_); }
  size_t get_size_in_bytes() const { return size_in_bytes_; }
  u32 get_lkey() const { return memory_region_->lkey; }
  u32 get_rkey() const { return memory_region_->rkey; }

private:
  Context& context_;
  void* data_{nullptr};
  size_t size_in_bytes_{0};
  ibv_mr* memory_region_{nullptr};
  bool is_registered_{false};
};

class LocalMemoryRegion : public MemoryRegion {
public:
  LocalMemoryRegion(Context& context, void* data, size_t size_in_bytes);
};

// must be on the heap s.t. the address does not change after vector movements
using LocalMemoryRegions = vec<u_ptr<LocalMemoryRegion>>;
using MemoryRegions = vec<u_ptr<MemoryRegion>>;

#endif  // RDMA_LIBRARY_MEMORY_REGION_HH
