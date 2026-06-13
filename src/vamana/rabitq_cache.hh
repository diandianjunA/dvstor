#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <linux/mempolicy.h>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>

#include "common/index_path.hh"
#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace vamana::rabitq {

class ScopedNumaInterleave {
public:
  ScopedNumaInterleave() {
    std::ifstream input("/sys/devices/system/node/online");
    str spec;
    if (!input.good() || !std::getline(input, spec)) return;
    u32 max_node = 0;
    std::stringstream ranges(spec);
    str range;
    vec<u32> nodes;
    while (std::getline(ranges, range, ',')) {
      const auto dash = range.find('-');
      const u32 first = static_cast<u32>(std::stoul(range.substr(0, dash)));
      const u32 last = dash == str::npos
        ? first : static_cast<u32>(std::stoul(range.substr(dash + 1)));
      for (u32 node = first; node <= last; ++node) nodes.push_back(node);
      max_node = std::max(max_node, last);
    }
    if (nodes.size() < 2) return;
    constexpr u32 word_bits = sizeof(unsigned long) * 8;
    mask_.assign(max_node / word_bits + 1, 0);
    for (u32 node : nodes) mask_[node / word_bits] |= 1ul << (node % word_bits);
    enabled_ = syscall(SYS_set_mempolicy, MPOL_INTERLEAVE,
                       mask_.data(), max_node + 1) == 0;
  }

  ~ScopedNumaInterleave() {
    if (enabled_) syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
  }

  bool enabled() const { return enabled_; }

private:
  vec<unsigned long> mask_;
  bool enabled_{};
};

constexpr u32 kCodeBits = 80;
constexpr u32 kCodeBytes = kCodeBits / 8;
constexpr u32 kEntryBytes = 12;
constexpr u32 kSidecarMagic = 0x34514652;  // RFQ4
constexpr u32 kSidecarVersion = 4;

#pragma pack(push, 1)
struct CompactEntry {
  std::array<byte_t, kCodeBytes> code{};
  u8 norm_q{};
  u8 error_q{};
};
#pragma pack(pop)

static_assert(sizeof(CompactEntry) == kEntryBytes);

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

using QueryLut = std::array<f32, kCodeBytes * 256>;

struct SidecarHeader {
  u32 magic{kSidecarMagic};
  u32 version{kSidecarVersion};
  u32 entry_size{kEntryBytes};
  u32 code_bits{kCodeBits};
  u32 node_size{};
  u32 raw_vector_bytes{};
  u64 entry_count{};
  u64 cache_budget_bytes{};
  Quantization quantization{};
};

struct DynamicSlot {
  u64 raw{};
  CompactEntry entry{};
  u8 state{};  // 0=empty, 1=live, 2=deleted
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

inline void validate_dimension() {
  lib_assert(VamanaNode::rabitq_code_bits() >= kCodeBits,
             "RaBitQ budget gate requires at least 72 rotated dimensions");
}

inline void compute_values(const byte_t* vector, VectorDType dtype,
                           std::array<byte_t, kCodeBytes>* code,
                           f32* norm, f32* error) {
  validate_dimension();
  thread_local vec<f32> rotated;
  rotated.resize(VamanaNode::rabitq_code_bits());
  f32 norm2 = 0.0f;
  VamanaNode::compute_rotated_query(vector, dtype, rotated.data(), &norm2);
  code->fill(0);
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < kCodeBits; ++bit) {
    const bool positive = rotated[bit] > 0.0f;
    if (positive) (*code)[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
    signed_dot += positive ? rotated[bit] : -rotated[bit];
  }
  *norm = std::sqrt(norm2);
  *error = *norm <= 1e-15f
    ? 1.0f
    : std::max(signed_dot / (*norm * std::sqrt(static_cast<f32>(kCodeBits))), 1e-15f);
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

inline QueryLut build_query_lut(const f32* rotated_query) {
  QueryLut lut{};
  for (u32 byte = 0; byte < kCodeBytes; ++byte) {
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
  for (u32 byte = 0; byte < kCodeBytes; ++byte) {
    signed_dot += lut[byte * 256 + entry.code[byte]];
  }
  const f32 norm = dequantize(entry.norm_q, quantization.norm_min, quantization.norm_max);
  const f32 error = dequantize(entry.error_q, quantization.error_min, quantization.error_max);
  const f32 inner_product = error > 1e-12f
    ? norm * signed_dot / (std::sqrt(static_cast<f32>(kCodeBits)) * error)
    : 0.0f;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

inline Estimate estimate_interval_lut(const QueryLut& lut, f32 query_norm2,
                                      const CompactEntry& entry,
                                      const Quantization& quantization,
                                      f32) {
  const f32 distance = estimate_distance_lut(lut, query_norm2, entry, quantization);
  return {distance, distance, distance};
}

inline f32 estimate_full_entry(const f32* rotated_query, f32 query_norm2,
                               const byte_t* entry) {
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < VamanaNode::rabitq_code_bits(); ++bit) {
    const bool positive = (entry[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    signed_dot += positive ? rotated_query[bit] : -rotated_query[bit];
  }
  f32 norm = 0.0f;
  f32 error = 0.0f;
  const size_t scalar_offset = VamanaNode::rabitq_code_storage_size();
  std::memcpy(&norm, entry + scalar_offset, sizeof(norm));
  std::memcpy(&error, entry + scalar_offset + sizeof(norm), sizeof(error));
  const f32 inner_product = error > 1e-12f
    ? norm * signed_dot /
        (std::sqrt(static_cast<f32>(VamanaNode::rabitq_code_bits())) * error)
    : 0.0f;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

inline void select_gate_into(const vec<f32>& distances,
                             const vec<u32>& cache_miss_indices,
                             u32 width, u32 max_width, f32 margin,
                             vec<u32>& selected,
                             vec<u32>& cached,
                             vec<u8>& is_miss) {
  selected.clear();
  cached.clear();
  is_miss.assign(distances.size(), 0);
  for (u32 index : cache_miss_indices) {
    if (index < is_miss.size() && !is_miss[index]) {
      is_miss[index] = 1;
      selected.push_back(index);
    }
  }
  for (u32 i = 0; i < distances.size(); ++i) {
    if (!is_miss[i]) cached.push_back(i);
  }
  const auto less_by_distance = [&](u32 lhs, u32 rhs) {
    if (distances[lhs] != distances[rhs]) return distances[lhs] < distances[rhs];
    return lhs < rhs;
  };
  const u32 base = std::min<u32>(width, cached.size());
  const u32 limit = std::max(width, max_width);
  const u32 bounded_limit = std::min<u32>(limit, cached.size());
  if (bounded_limit == 0) return;
  if (cached.size() > bounded_limit) {
    std::nth_element(cached.begin(), cached.begin() + bounded_limit,
                     cached.end(), less_by_distance);
    cached.resize(bounded_limit);
  }
  std::sort(cached.begin(), cached.end(), less_by_distance);
  for (u32 i = 0; i < base; ++i) selected.push_back(cached[i]);
  if (base > 0) {
    const f32 cutoff = distances[cached[base - 1]];
    const f32 margin_cutoff = cutoff + std::abs(cutoff) * std::max(margin, 0.0f);
    for (u32 i = base; i < bounded_limit; ++i) {
      if (distances[cached[i]] > margin_cutoff) break;
      selected.push_back(cached[i]);
    }
  }
}

inline vec<u32> select_gate(const vec<f32>& distances,
                            const vec<u32>& cache_miss_indices,
                            u32 width, u32 max_width, f32 margin) {
  vec<u32> selected;
  vec<u32> cached;
  vec<u8> is_miss;
  select_gate_into(distances, cache_miss_indices, width, max_width, margin,
                   selected, cached, is_miss);
  return selected;
}

class Cache {
public:
  bool load(const filepath_t& prefix, u32 num_nodes, u32 expected_node_size,
            size_t dynamic_budget_bytes, str* error) {
    validate_dimension();
    ScopedNumaInterleave interleave;
    numa_interleaved_ = interleave.enabled();
    shards_.assign(num_nodes, {});
    size_bytes_ = 0;
    entry_count_ = 0;
    for (u32 node = 0; node < num_nodes; ++node) {
      const filepath_t path = index_path::rabitq_cache_file(prefix, node + 1, num_nodes);
      std::ifstream input(path, std::ios::binary);
      if (!input.good()) return fail(error, "missing RFQ4 RaBitQ sidecar: " + path.string());
      SidecarHeader header;
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!input.good() || header.magic != kSidecarMagic ||
          header.version != kSidecarVersion || header.entry_size != kEntryBytes ||
          header.code_bits != kCodeBits || header.node_size != expected_node_size) {
        return fail(error, "invalid RFQ4 RaBitQ sidecar header: " + path.string());
      }
      if (node == 0) quantization_ = header.quantization;
      if (std::memcmp(&quantization_, &header.quantization, sizeof(Quantization)) != 0) {
        return fail(error, "RaBitQ gate quantization differs across shards");
      }
      auto& entries = shards_[node];
      entries.resize(header.entry_count);
      input.read(reinterpret_cast<char*>(entries.data()),
                 static_cast<std::streamsize>(entries.size() * sizeof(CompactEntry)));
      if (!input.good()) return fail(error, "truncated RFQ4 RaBitQ sidecar: " + path.string());
      size_bytes_ += entries.size() * sizeof(CompactEntry);
      entry_count_ += entries.size();
    }
    node_size_ = expected_node_size;
    rebuild_decode_tables();
    init_dynamic(dynamic_budget_bytes);
    prewarm();
    return true;
  }

  f32 estimate_distance_lut(const QueryLut& lut, f32 query_norm2,
                            const CompactEntry& entry) const {
    f32 signed_dot = 0.0f;
    for (u32 byte = 0; byte < kCodeBytes; ++byte) {
      signed_dot += lut[byte * 256 + entry.code[byte]];
    }
    const u32 scale_index =
      (static_cast<u32>(entry.norm_q) << 8u) | static_cast<u32>(entry.error_q);
    return std::max(query_norm2 + norm2_table_[entry.norm_q] -
                    scale_table_[scale_index] * signed_dot, 0.0f);
  }

  const CompactEntry* find(RemotePtr pointer) const {
    if (pointer.memory_node() < shards_.size() && pointer.byte_offset() >= 16 && node_size_ != 0) {
      const u64 relative = pointer.byte_offset() - 16;
      if (relative % node_size_ == 0) {
        const u64 slot = relative / node_size_;
        const auto& entries = shards_[pointer.memory_node()];
        if (slot < entries.size()) return &entries[slot];
      }
    }
    return find_dynamic(pointer);
  }

  bool upsert_dynamic(RemotePtr pointer, const byte_t* vector, VectorDType dtype) {
    if (dynamic_slots_.empty() || pointer.is_null()) return false;
    const CompactEntry entry = encode(vector, dtype, quantization_);
    std::unique_lock lock(dynamic_mutex_);
    const u64 raw = pointer.raw_address;
    const u64 mask = static_cast<u64>(dynamic_slots_.size() - 1);
    size_t first_deleted = dynamic_slots_.size();
    for (u64 probe = 0; probe < dynamic_slots_.size(); ++probe) {
      const size_t slot_index = (hash_raw(raw) + probe) & mask;
      auto& slot = dynamic_slots_[slot_index];
      if (slot.state == 1 && slot.raw == raw) {
        slot.entry = entry;
        return true;
      }
      if (slot.state == 2 && first_deleted == dynamic_slots_.size()) {
        first_deleted = slot_index;
      }
      if (slot.state == 0) {
        auto& target = dynamic_slots_[first_deleted == dynamic_slots_.size()
          ? slot_index : first_deleted];
        target.raw = raw;
        target.entry = entry;
        target.state = 1;
        ++dynamic_live_;
        return true;
      }
    }
    if (first_deleted != dynamic_slots_.size()) {
      auto& target = dynamic_slots_[first_deleted];
      target.raw = raw;
      target.entry = entry;
      target.state = 1;
      ++dynamic_live_;
      return true;
    }
    ++dynamic_overflow_;
    return false;
  }

  bool erase_dynamic(RemotePtr pointer) {
    if (dynamic_slots_.empty() || pointer.is_null()) return false;
    std::unique_lock lock(dynamic_mutex_);
    const u64 raw = pointer.raw_address;
    const u64 mask = static_cast<u64>(dynamic_slots_.size() - 1);
    for (u64 probe = 0; probe < dynamic_slots_.size(); ++probe) {
      auto& slot = dynamic_slots_[(hash_raw(raw) + probe) & mask];
      if (slot.state == 0) return false;
      if (slot.state == 1 && slot.raw == raw) {
        slot.state = 2;
        --dynamic_live_;
        return true;
      }
    }
    return false;
  }

  const Quantization& quantization() const { return quantization_; }
  size_t size_bytes() const { return size_bytes_; }
  size_t dynamic_size_bytes() const { return dynamic_slots_.size() * sizeof(DynamicSlot); }
  size_t decode_table_bytes() const { return sizeof(norm2_table_) + sizeof(scale_table_); }
  size_t total_size_bytes() const { return size_bytes_ + dynamic_size_bytes() + decode_table_bytes(); }
  size_t entry_count() const { return entry_count_; }
  size_t dynamic_capacity() const { return dynamic_slots_.size(); }
  size_t dynamic_live() const { return dynamic_live_; }
  size_t dynamic_overflow() const { return dynamic_overflow_; }
  bool numa_interleaved() const { return numa_interleaved_; }

private:
  static u64 hash_raw(u64 value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
  }

  void rebuild_decode_tables() {
    for (u32 norm_q = 0; norm_q < 256; ++norm_q) {
      const f32 norm = dequantize(static_cast<u8>(norm_q),
                                  quantization_.norm_min, quantization_.norm_max);
      norm2_table_[norm_q] = norm * norm;
      for (u32 error_q = 0; error_q < 256; ++error_q) {
        const f32 error = dequantize(static_cast<u8>(error_q),
                                     quantization_.error_min, quantization_.error_max);
        const u32 index = (norm_q << 8u) | error_q;
        scale_table_[index] = error > 1e-12f
          ? 2.0f * norm / (std::sqrt(static_cast<f32>(kCodeBits)) * error)
          : 0.0f;
      }
    }
  }

  void init_dynamic(size_t budget_bytes) {
    if (budget_bytes < sizeof(DynamicSlot) * 2) return;
    size_t capacity = 1;
    const size_t requested = budget_bytes / sizeof(DynamicSlot);
    while (capacity * 2 <= requested) capacity *= 2;
    dynamic_slots_.assign(capacity, {});
  }

  const CompactEntry* find_dynamic(RemotePtr pointer) const {
    if (dynamic_slots_.empty() || pointer.is_null()) return nullptr;
    std::shared_lock lock(dynamic_mutex_);
    const u64 raw = pointer.raw_address;
    const u64 mask = static_cast<u64>(dynamic_slots_.size() - 1);
    for (u64 probe = 0; probe < dynamic_slots_.size(); ++probe) {
      const auto& slot = dynamic_slots_[(hash_raw(raw) + probe) & mask];
      if (slot.state == 0) return nullptr;
      if (slot.raw == raw) return slot.state == 1 ? &slot.entry : nullptr;
    }
    return nullptr;
  }

  void prewarm() const {
    volatile u8 sink = 0;
    for (const auto& shard : shards_) {
      const auto* bytes = reinterpret_cast<const u8*>(shard.data());
      const size_t bytes_size = shard.size() * sizeof(CompactEntry);
      for (size_t offset = 0; offset < bytes_size; offset += 4096) sink ^= bytes[offset];
    }
    (void)sink;
  }

  static bool fail(str* error, const str& message) {
    if (error != nullptr) *error = message;
    return false;
  }

  vec<vec<CompactEntry>> shards_;
  mutable std::shared_mutex dynamic_mutex_;
  vec<DynamicSlot> dynamic_slots_;
  std::array<f32, 256> norm2_table_{};
  std::array<f32, 256 * 256> scale_table_{};
  Quantization quantization_{};
  u32 node_size_{};
  size_t size_bytes_{};
  size_t entry_count_{};
  size_t dynamic_live_{};
  size_t dynamic_overflow_{};
  bool numa_interleaved_{};
};

}  // namespace vamana::rabitq
