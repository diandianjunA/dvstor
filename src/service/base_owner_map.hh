#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "common/types.hh"

namespace service {

// Immutable after load. A two-level byte table keeps the common dense-ID
// representation at one byte per ID without making a sparse/corrupt high ID
// allocate a multi-gigabyte flat vector.
class BaseOwnerMap {
public:
  bool load(const filepath_t& index_prefix,
            u32 owner_count,
            const str& idmap_format,
            str* error_message = nullptr);

  std::optional<u32> owner_for(node_t id) const;
  size_t entry_count() const { return entry_count_; }
  size_t memory_bytes() const;
  bool empty() const { return entry_count_ == 0; }

private:
  static constexpr u8 kMissingOwner = std::numeric_limits<u8>::max();
  static constexpr u32 kPageBits = 16;
  static constexpr size_t kPageSize = size_t{1} << kPageBits;
  using Page = std::array<u8, kPageSize>;

  std::vector<std::unique_ptr<Page>> pages_;
  size_t allocated_pages_{};
  size_t entry_count_{};
};

}  // namespace service
