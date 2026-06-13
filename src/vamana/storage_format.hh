#pragma once

#include <optional>

#include "common/types.hh"

namespace vamana {

enum class StorageFormat : u8 {
  aos_v1 = 1,
  compact_v1 = 2,
};

inline constexpr const char* storage_format_name(StorageFormat format) {
  switch (format) {
    case StorageFormat::aos_v1:
      return "vamana_aos_v1";
    case StorageFormat::compact_v1:
      return "vamana_compact_v1";
  }
  return "unknown";
}

inline std::optional<StorageFormat> parse_storage_format(const str& name) {
  if (name == "vamana_aos_v1") return StorageFormat::aos_v1;
  if (name == "vamana_compact_v1") return StorageFormat::compact_v1;
  return std::nullopt;
}

}  // namespace vamana
