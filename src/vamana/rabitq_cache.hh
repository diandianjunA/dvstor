#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "common/index_path.hh"
#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace vamana::rabitq {

constexpr u32 kCacheBits = 80;
constexpr u32 kCacheCodeBytes = kCacheBits / 8;
constexpr u32 kCacheEntryBytes = 12;
constexpr u32 kCacheMagic = 0x31514652;  // RFQ1

#pragma pack(push, 1)
struct CompactEntry {
  std::array<byte_t, kCacheCodeBytes> code{};
  u8 norm_q{};
  u8 error_q{};
};
#pragma pack(pop)

static_assert(sizeof(CompactEntry) == kCacheEntryBytes);

struct Quantization {
  f32 norm_min{};
  f32 norm_max{};
  f32 error_min{};
  f32 error_max{};
};

struct Estimate {
  f32 distance{};
  f32 lower_bound{};
  f32 upper_bound{};
};

using QueryLut = std::array<f32, kCacheCodeBytes * 256>;

struct SidecarHeader {
  u32 magic{kCacheMagic};
  u32 version{1};
  u32 entry_size{kCacheEntryBytes};
  u32 code_bits{kCacheBits};
  u32 node_size{};
  u32 reserved{};
  u64 entry_count{};
  Quantization quantization{};
};

inline u8 quantize(f32 value, f32 min_value, f32 max_value) {
  if (!(max_value > min_value)) return 0;
  const f32 scaled = (value - min_value) * (255.0f / (max_value - min_value));
  return static_cast<u8>(std::clamp(std::lround(scaled), 0l, 255l));
}

inline f32 dequantize(u8 value, f32 min_value, f32 max_value) {
  if (!(max_value > min_value)) return min_value;
  return min_value + static_cast<f32>(value) * ((max_value - min_value) / 255.0f);
}

inline void compute_values(const byte_t* vector, VectorDType dtype,
                           std::array<byte_t, kCacheCodeBytes>* code,
                           f32* norm, f32* error) {
  vec<f32> rotated(VamanaNode::rabitq_code_bits());
  f32 norm2 = 0.0f;
  VamanaNode::compute_rotated_query(vector, dtype, rotated.data(), &norm2);
  code->fill(0);
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < kCacheBits; ++bit) {
    const bool positive = rotated[bit] > 0.0f;
    if (positive) {
      (*code)[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
    }
    signed_dot += positive ? rotated[bit] : -rotated[bit];
  }
  *norm = std::sqrt(norm2);
  *error = *norm <= 1e-15f
    ? 1.0f
    : std::max(signed_dot / (*norm * std::sqrt(static_cast<f32>(kCacheBits))), 1e-15f);
}

inline CompactEntry encode(const byte_t* vector, VectorDType dtype,
                           const Quantization& quantization) {
  CompactEntry entry;
  f32 norm = 0.0f;
  f32 error = 0.0f;
  compute_values(vector, dtype, &entry.code, &norm, &error);
  entry.norm_q = quantize(norm, quantization.norm_min, quantization.norm_max);
  entry.error_q = quantize(error, quantization.error_min, quantization.error_max);
  return entry;
}

inline f32 estimate_distance(const f32* rotated_query, f32 query_norm2,
                             const CompactEntry& entry,
                             const Quantization& quantization) {
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < kCacheBits; ++bit) {
    const bool positive = (entry.code[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    signed_dot += positive ? rotated_query[bit] : -rotated_query[bit];
  }
  const f32 norm = dequantize(entry.norm_q, quantization.norm_min, quantization.norm_max);
  const f32 error = dequantize(entry.error_q, quantization.error_min, quantization.error_max);
  const f32 inner_product = error > 1e-12f
    ? norm * signed_dot / (std::sqrt(static_cast<f32>(kCacheBits)) * error)
    : 0.0f;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

inline QueryLut build_query_lut(const f32* rotated_query) {
  QueryLut lut{};
  for (u32 byte = 0; byte < kCacheCodeBytes; ++byte) {
    for (u32 code = 0; code < 256; ++code) {
      f32 sum = 0.0f;
      for (u32 bit = 0; bit < 8; ++bit) {
        const bool positive = (code & (1u << (7u - bit))) != 0;
        const f32 value = rotated_query[byte * 8 + bit];
        sum += positive ? value : -value;
      }
      lut[byte * 256 + code] = sum;
    }
  }
  return lut;
}

inline f32 estimate_distance_lut(const QueryLut& lut, f32 query_norm2,
                                 const CompactEntry& entry,
                                 const Quantization& quantization) {
  f32 signed_dot = 0.0f;
  for (u32 byte = 0; byte < kCacheCodeBytes; ++byte) {
    signed_dot += lut[byte * 256 + entry.code[byte]];
  }
  const f32 norm = dequantize(entry.norm_q, quantization.norm_min, quantization.norm_max);
  const f32 error = dequantize(entry.error_q, quantization.error_min, quantization.error_max);
  const f32 inner_product = error > 1e-12f
    ? norm * signed_dot / (std::sqrt(static_cast<f32>(kCacheBits)) * error)
    : 0.0f;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

inline Estimate estimate_interval(const f32* rotated_query, f32 query_norm2,
                                  const CompactEntry& entry,
                                  const Quantization& quantization,
                                  f32 epsilon = 1.9f) {
  const f32 distance = estimate_distance(rotated_query, query_norm2, entry, quantization);
  const f32 norm = dequantize(entry.norm_q, quantization.norm_min, quantization.norm_max);
  const f32 correlation = std::max(
    dequantize(entry.error_q, quantization.error_min, quantization.error_max), 1e-6f);
  const f32 angular_error = 2.0f * norm * epsilon *
    std::sqrt(std::max(1.0f / (correlation * correlation) - 1.0f, 0.0f) /
              static_cast<f32>(kCacheBits - 1)) * std::sqrt(query_norm2);
  const f32 norm_step = (quantization.norm_max - quantization.norm_min) / 255.0f;
  const f32 quantization_error = 2.0f * norm_step *
    (std::sqrt(query_norm2) + norm + norm_step);
  const f32 error = angular_error + quantization_error;
  return {distance, std::max(distance - error, 0.0f), distance + error};
}

inline Estimate estimate_interval_lut(const QueryLut& lut, f32 query_norm2,
                                      const CompactEntry& entry,
                                      const Quantization& quantization,
                                      f32 epsilon = 1.9f) {
  const f32 distance = estimate_distance_lut(lut, query_norm2, entry, quantization);
  const f32 norm = dequantize(entry.norm_q, quantization.norm_min, quantization.norm_max);
  const f32 correlation = std::max(
    dequantize(entry.error_q, quantization.error_min, quantization.error_max), 1e-6f);
  const f32 angular_error = 2.0f * norm * epsilon *
    std::sqrt(std::max(1.0f / (correlation * correlation) - 1.0f, 0.0f) /
              static_cast<f32>(kCacheBits - 1)) * std::sqrt(query_norm2);
  const f32 norm_step = (quantization.norm_max - quantization.norm_min) / 255.0f;
  const f32 quantization_error = 2.0f * norm_step *
    (std::sqrt(query_norm2) + norm + norm_step);
  const f32 error = angular_error + quantization_error;
  return {distance, std::max(distance - error, 0.0f), distance + error};
}

inline f32 estimate_full_entry(const f32* rotated_query, f32 query_norm2,
                               const byte_t* entry) {
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < VamanaNode::rabitq_code_bits(); ++bit) {
    const bool positive = (entry[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    signed_dot += positive ? rotated_query[bit] : -rotated_query[bit];
  }
  const u32 scalar_offset = static_cast<u32>(VamanaNode::rabitq_code_storage_size());
  f32 norm = 0.0f;
  f32 error = 0.0f;
  std::memcpy(&norm, entry + scalar_offset, sizeof(norm));
  std::memcpy(&error, entry + scalar_offset + sizeof(norm), sizeof(error));
  const f32 inner_product = error > 1e-12f
    ? norm * signed_dot /
        (std::sqrt(static_cast<f32>(VamanaNode::rabitq_code_bits())) * error)
    : 0.0f;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

class Cache {
public:
  bool load(const filepath_t& prefix, u32 num_nodes, u32 expected_node_size, str* error) {
    shards_.clear();
    shards_.resize(num_nodes);
    size_bytes_ = 0;
    entry_count_ = 0;
    for (u32 node = 0; node < num_nodes; ++node) {
      const filepath_t path = index_path::rabitq_cache_file(prefix, node + 1, num_nodes);
      std::ifstream input(path, std::ios::binary);
      if (!input.good()) return fail(error, "missing RaBitQ cache sidecar: " + path.string());
      SidecarHeader header;
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!input.good() || header.magic != kCacheMagic || header.version != 1 ||
          header.entry_size != kCacheEntryBytes || header.code_bits != kCacheBits ||
          header.node_size != expected_node_size) {
        return fail(error, "invalid RaBitQ cache sidecar header: " + path.string());
      }
      if (node == 0) quantization_ = header.quantization;
      if (std::memcmp(&quantization_, &header.quantization, sizeof(Quantization)) != 0) {
        return fail(error, "RaBitQ cache quantization differs across shards");
      }
      auto& entries = shards_[node];
      entries.resize(header.entry_count);
      input.read(reinterpret_cast<char*>(entries.data()),
                 static_cast<std::streamsize>(entries.size() * sizeof(CompactEntry)));
      if (!input.good()) return fail(error, "truncated RaBitQ cache sidecar: " + path.string());
      size_bytes_ += entries.size() * sizeof(CompactEntry);
      entry_count_ += entries.size();
    }
    node_size_ = expected_node_size;
    return true;
  }

  const CompactEntry* find(RemotePtr pointer) const {
    if (pointer.memory_node() >= shards_.size() || pointer.byte_offset() < 16 || node_size_ == 0) {
      return nullptr;
    }
    const u64 relative = pointer.byte_offset() - 16;
    if (relative % node_size_ != 0) return nullptr;
    const u64 slot = relative / node_size_;
    const auto& entries = shards_[pointer.memory_node()];
    return slot < entries.size() ? &entries[slot] : nullptr;
  }

  const Quantization& quantization() const { return quantization_; }
  size_t size_bytes() const { return size_bytes_; }
  size_t entry_count() const { return entry_count_; }

private:
  static bool fail(str* error, const str& message) {
    if (error != nullptr) *error = message;
    return false;
  }

  vec<vec<CompactEntry>> shards_;
  Quantization quantization_{};
  u32 node_size_{};
  size_t size_bytes_{};
  size_t entry_count_{};
};

}  // namespace vamana::rabitq
