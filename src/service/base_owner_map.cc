#include "service/base_owner_map.hh"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>

#include "common/index_path.hh"
#include "vamana/idmap.hh"

namespace service {

namespace {

bool fail(str* error_message, const str& message) {
  if (error_message != nullptr) *error_message = message;
  return false;
}

}  // namespace

bool BaseOwnerMap::load(const filepath_t& index_prefix,
                        u32 owner_count,
                        const str& idmap_format,
                        str* error_message) {
  if (idmap_format != "owner_sharded_v1") {
    return fail(error_message,
                "compute mutations require idmap_format=owner_sharded_v1; "
                "index metadata reports '" + idmap_format + "'");
  }
  if (index_prefix.empty()) {
    return fail(error_message, "owner idmap index prefix is empty");
  }
  // Values 0..254 are owners and 255 is the absent-ID sentinel.
  if (owner_count == 0 ||
      owner_count > static_cast<u32>(std::numeric_limits<u8>::max())) {
    return fail(error_message,
                "owner-sharded idmap requires between 1 and 255 owners");
  }

  BaseOwnerMap loaded;
  try {
    for (u32 owner = 0; owner < owner_count; ++owner) {
      const filepath_t path = index_path::owner_idmap_file(
        index_prefix, static_cast<size_t>(owner) + 1, owner_count);
      std::error_code size_error;
      const std::uintmax_t actual_bytes =
        std::filesystem::file_size(path, size_error);
      if (size_error) {
        return fail(error_message,
                    "missing owner idmap sidecar: " + path.string());
      }
      if (actual_bytes < sizeof(vamana::idmap::Header)) {
        return fail(error_message,
                    "truncated owner idmap header: " + path.string());
      }

      std::ifstream input(path, std::ios::binary);
      if (!input.good()) {
        return fail(error_message,
                    "failed to open owner idmap sidecar: " + path.string());
      }
      vamana::idmap::Header header{};
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        return fail(error_message,
                    "truncated owner idmap header: " + path.string());
      }
      if (header.magic != vamana::idmap::kMagic ||
          header.version != vamana::idmap::kVersion ||
          header.owner_shard != owner ||
          header.shard_count != owner_count) {
        return fail(error_message,
                    "invalid owner idmap header (magic/version/owner/shard): " +
                    path.string());
      }
      constexpr std::uintmax_t kHeaderBytes = sizeof(vamana::idmap::Header);
      constexpr std::uintmax_t kEntryBytes = sizeof(vamana::idmap::Entry);
      if (header.entry_count >
          (std::numeric_limits<std::uintmax_t>::max() - kHeaderBytes) /
            kEntryBytes) {
        return fail(error_message,
                    "owner idmap entry count overflows file size: " +
                    path.string());
      }
      const std::uintmax_t expected_bytes =
        kHeaderBytes + header.entry_count * kEntryBytes;
      if (actual_bytes != expected_bytes) {
        return fail(error_message,
                    "owner idmap file size mismatch (truncated or trailing data): " +
                    path.string());
      }

      std::array<vamana::idmap::Entry, 4096> entry_buffer{};
      u64 entries_remaining = header.entry_count;
      while (entries_remaining != 0) {
        const size_t entries_to_read = static_cast<size_t>(std::min<u64>(
          entries_remaining, entry_buffer.size()));
        const size_t bytes_to_read =
          entries_to_read * sizeof(vamana::idmap::Entry);
        input.read(reinterpret_cast<char*>(entry_buffer.data()),
                   static_cast<std::streamsize>(bytes_to_read));
        if (input.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
          return fail(error_message,
                      "truncated owner idmap entry: " + path.string());
        }

        for (size_t entry_index = 0; entry_index < entries_to_read;
             ++entry_index) {
          const auto& entry = entry_buffer[entry_index];
          const size_t page_index =
            static_cast<size_t>(entry.id >> kPageBits);
          const size_t page_offset =
            static_cast<size_t>(entry.id & (kPageSize - 1));
          if (loaded.pages_.size() <= page_index) {
            loaded.pages_.resize(page_index + 1);
          }
          if (!loaded.pages_[page_index]) {
            loaded.pages_[page_index] = std::make_unique_for_overwrite<Page>();
            loaded.pages_[page_index]->fill(kMissingOwner);
            ++loaded.allocated_pages_;
          }
          u8& existing = (*loaded.pages_[page_index])[page_offset];
          if (existing != kMissingOwner) {
            const str duplicate_kind = existing == static_cast<u8>(owner)
              ? "duplicate ID in owner idmap"
              : "conflicting owner for duplicate ID in owner idmaps";
            return fail(error_message,
                        duplicate_kind + ": id=" + std::to_string(entry.id) +
                        " previous_owner=" + std::to_string(existing) +
                        " owner=" + std::to_string(owner));
          }
          existing = static_cast<u8>(owner);
          ++loaded.entry_count_;
        }
        entries_remaining -= entries_to_read;
      }
    }
  } catch (const std::bad_alloc&) {
    return fail(error_message,
                "insufficient memory while loading owner-sharded idmaps");
  }

  *this = std::move(loaded);
  return true;
}

std::optional<u32> BaseOwnerMap::owner_for(node_t id) const {
  const size_t page_index = static_cast<size_t>(id >> kPageBits);
  if (page_index >= pages_.size() || !pages_[page_index]) {
    return std::nullopt;
  }
  const size_t page_offset = static_cast<size_t>(id & (kPageSize - 1));
  const u8 owner = (*pages_[page_index])[page_offset];
  if (owner == kMissingOwner) return std::nullopt;
  return static_cast<u32>(owner);
}

size_t BaseOwnerMap::memory_bytes() const {
  return pages_.size() * sizeof(std::unique_ptr<Page>) +
         allocated_pages_ * sizeof(Page);
}

}  // namespace service
