#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <linux/mempolicy.h>
#include <mutex>
#include <numeric>
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

constexpr u32 kSidecarMagic = 0x35514652;  // RFQ5
constexpr u32 kSidecarVersion = 5;
constexpr u32 kDefaultEntryBytes = 12;
constexpr u32 kEntryBytes = kDefaultEntryBytes;  // compatibility for older metadata/report code
constexpr f64 kDefaultCacheRatio = 0.10;

struct Quantization {
  f32 norm_min{};
  f32 norm_max{};
  f32 error_min{};  // retained for metadata compatibility; unused by RFQ5
  f32 error_max{};  // retained for metadata compatibility; unused by RFQ5
};

struct Estimate {
  f32 distance{};
  f32 lower_bound{};
  f32 upper_bound{};
};

struct QueryLut {
  vec<f32> signed_dot;
  vec<f32> mismatch_energy;
  u32 code_bits{};
  u32 code_bytes{};
};

struct SidecarHeader {
  u32 magic{kSidecarMagic};
  u32 version{kSidecarVersion};
  u32 entry_size{};
  u32 code_bits{};
  u32 node_size{};
  u32 raw_vector_bytes{};
  u64 entry_count{};
  u64 cache_budget_bytes{};
  Quantization quantization{};
};

struct DynamicSlot {
  u64 raw{};
  u8 state{};  // 0=empty, 1=live, 2=deleted
};

inline u32 entry_code_bytes(u32 entry_bytes) {
  return entry_bytes > 0 ? entry_bytes - 1 : 0;
}

inline u32 entry_code_bits(u32 entry_bytes) {
  return entry_code_bytes(entry_bytes) * 8u;
}

inline u32 choose_entry_bytes(u32 raw_vector_bytes, f64 ratio = kDefaultCacheRatio) {
  if (raw_vector_bytes == 0 || ratio <= 0.0) return 0;
  const auto budget = static_cast<u32>(std::floor(raw_vector_bytes * ratio));
  if (budget < 2) return 0;
  const u32 full_code_bytes = (VamanaNode::rabitq_code_bits() + 7u) / 8u;
  return std::min<u32>(budget, full_code_bytes + 1u);
}

inline u8 quantize(f32 value, f32 min_value, f32 max_value) {
  if (!(max_value > min_value)) return 0;
  const f32 scaled = (value - min_value) * (255.0f / (max_value - min_value));
  return static_cast<u8>(std::clamp(std::lround(scaled), 0l, 255l));
}

inline f32 dequantize(u8 value, f32 min_value, f32 max_value) {
  if (!(max_value > min_value)) return min_value;
  return min_value + static_cast<f32>(value) * ((max_value - min_value) / 255.0f);
}

inline f32 dequantize_norm_lower(u8 value, const Quantization& q) {
  if (!(q.norm_max > q.norm_min)) return q.norm_min;
  const f32 step = (q.norm_max - q.norm_min) / 255.0f;
  return q.norm_min + std::max(0.0f, static_cast<f32>(value) - 0.5f) * step;
}

inline f32 dequantize_norm_upper(u8 value, const Quantization& q) {
  if (!(q.norm_max > q.norm_min)) return q.norm_min;
  const f32 step = (q.norm_max - q.norm_min) / 255.0f;
  return q.norm_min + std::min(255.0f, static_cast<f32>(value) + 0.5f) * step;
}

inline void validate_dimension() {
  lib_assert(VamanaNode::rabitq_code_bits() >= 8,
             "RaBitQ requires at least one rotated byte");
}

inline bool compute_values(const byte_t* vector, VectorDType dtype,
                           u32 code_bits, byte_t* code,
                           f32* norm) {
  validate_dimension();
  if (code_bits == 0 || code_bits > VamanaNode::rabitq_code_bits() ||
      (code_bits % 8u) != 0) {
    return false;
  }
  const u32 code_bytes = code_bits / 8u;
  thread_local vec<f32> rotated;
  rotated.resize(VamanaNode::rabitq_code_bits());
  f32 norm2 = 0.0f;
  VamanaNode::compute_rotated_query(vector, dtype, rotated.data(), &norm2);
  std::memset(code, 0, code_bytes);
  for (u32 bit = 0; bit < code_bits; ++bit) {
    if (rotated[bit] > 0.0f) code[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
  }
  *norm = std::sqrt(norm2);
  return true;
}

inline bool encode_into(const byte_t* vector, VectorDType dtype,
                        const Quantization& quantization,
                        u32 code_bits, u32 entry_bytes,
                        byte_t* entry) {
  if (entry_bytes < 2 || code_bits != entry_code_bits(entry_bytes)) return false;
  f32 norm = 0.0f;
  if (!compute_values(vector, dtype, code_bits, entry, &norm)) return false;
  if (norm < quantization.norm_min || norm > quantization.norm_max) {
    return false;
  }
  entry[entry_code_bytes(entry_bytes)] =
      quantize(norm, quantization.norm_min, quantization.norm_max);
  return true;
}

inline vec<byte_t> encode(const byte_t* vector, VectorDType dtype,
                          const Quantization& quantization,
                          u32 code_bits, u32 entry_bytes) {
  vec<byte_t> entry(entry_bytes, 0);
  (void)encode_into(vector, dtype, quantization, code_bits, entry_bytes, entry.data());
  return entry;
}

inline QueryLut build_query_lut(const f32* rotated_query, u32 code_bits) {
  QueryLut lut;
  lut.code_bits = code_bits;
  lut.code_bytes = code_bits / 8u;
  lut.signed_dot.assign(static_cast<size_t>(lut.code_bytes) * 256u, 0.0f);
  lut.mismatch_energy.assign(static_cast<size_t>(lut.code_bytes) * 256u, 0.0f);
  for (u32 byte = 0; byte < lut.code_bytes; ++byte) {
    for (u32 code = 0; code < 256; ++code) {
      f32 dot = 0.0f;
      f32 mismatch = 0.0f;
      for (u32 bit = 0; bit < 8; ++bit) {
        const u32 global_bit = byte * 8u + bit;
        if (global_bit >= code_bits) break;
        const bool positive = (code & (1u << (7u - bit))) != 0;
        const f32 value = rotated_query[global_bit];
        dot += positive ? value : -value;
        if ((value > 0.0f) != positive) mismatch += value * value;
      }
      lut.signed_dot[byte * 256u + code] = dot;
      lut.mismatch_energy[byte * 256u + code] = mismatch;
    }
  }
  return lut;
}

inline QueryLut build_query_lut(const f32* rotated_query) {
  return build_query_lut(rotated_query, VamanaNode::rabitq_code_bits());
}

inline f32 estimate_distance_lut(const QueryLut& lut, f32 query_norm2,
                                 const byte_t* entry,
                                 const Quantization& quantization) {
  f32 signed_dot = 0.0f;
  for (u32 byte = 0; byte < lut.code_bytes; ++byte) {
    signed_dot += lut.signed_dot[byte * 256u + entry[byte]];
  }
  const f32 norm = dequantize(entry[lut.code_bytes],
                              quantization.norm_min, quantization.norm_max);
  const f32 denom = std::sqrt(static_cast<f32>(std::max<u32>(1, lut.code_bits)));
  const f32 inner_product = norm * signed_dot / denom;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

inline f32 lower_bound_lut(const QueryLut& lut, f32 query_norm2,
                           const byte_t* entry,
                           const Quantization& quantization) {
  f32 mismatch = 0.0f;
  for (u32 byte = 0; byte < lut.code_bytes; ++byte) {
    mismatch += lut.mismatch_energy[byte * 256u + entry[byte]];
  }
  const f32 projection_norm = std::sqrt(std::max(query_norm2 - mismatch, 0.0f));
  const u8 norm_q = entry[lut.code_bytes];
  const f32 norm_lo = dequantize_norm_lower(norm_q, quantization);
  const f32 norm_hi = dequantize_norm_upper(norm_q, quantization);
  const f32 r = std::clamp(projection_norm, norm_lo, norm_hi);
  return std::max(query_norm2 + r * r - 2.0f * projection_norm * r, 0.0f);
}

inline Estimate estimate_interval_lut(const QueryLut& lut, f32 query_norm2,
                                      const byte_t* entry,
                                      const Quantization& quantization,
                                      f32) {
  const f32 distance = estimate_distance_lut(lut, query_norm2, entry, quantization);
  const f32 lower = lower_bound_lut(lut, query_norm2, entry, quantization);
  return {distance, lower, std::numeric_limits<f32>::infinity()};
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
  const auto less_by_distance = [&](u32 lhs, u32 rhs) {
    if (distances[lhs] != distances[rhs]) return distances[lhs] < distances[rhs];
    return lhs < rhs;
  };
  if (cache_miss_indices.empty()) {
    is_miss.clear();
    cached.resize(distances.size());
    std::iota(cached.begin(), cached.end(), 0u);
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
    selected.insert(selected.end(), cached.begin(), cached.begin() + base);
    if (base > 0) {
      const f32 cutoff = distances[cached[base - 1]];
      const f32 margin_cutoff = cutoff + std::abs(cutoff) * std::max(margin, 0.0f);
      for (u32 i = base; i < bounded_limit; ++i) {
        if (distances[cached[i]] > margin_cutoff) break;
        selected.push_back(cached[i]);
      }
    }
    return;
  }
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
            size_t dynamic_budget_bytes, str* error,
            f64 max_cache_ratio = kDefaultCacheRatio) {
    validate_dimension();
    ScopedNumaInterleave interleave;
    numa_interleaved_ = interleave.enabled();
    shards_.assign(num_nodes, {});
    override_bits_.assign(num_nodes, {});
    has_dynamic_entries_.store(false, std::memory_order_relaxed);
    size_bytes_ = 0;
    entry_count_ = 0;
    for (u32 node = 0; node < num_nodes; ++node) {
      const filepath_t path = index_path::rabitq_cache_file(prefix, node + 1, num_nodes);
      std::ifstream input(path, std::ios::binary);
      if (!input.good()) return fail(error, "missing RFQ5 RaBitQ sidecar: " + path.string());
      SidecarHeader header;
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!input.good() || header.magic != kSidecarMagic ||
          header.version != kSidecarVersion || header.node_size != expected_node_size ||
          header.entry_size < 2 || header.code_bits != entry_code_bits(header.entry_size) ||
          header.raw_vector_bytes != VamanaNode::vector_bytes()) {
        return fail(error, "invalid RFQ5 RaBitQ sidecar header: " + path.string());
      }
      if (node == 0) {
        entry_bytes_ = header.entry_size;
        code_bits_ = header.code_bits;
        code_bytes_ = entry_code_bytes(entry_bytes_);
        quantization_ = header.quantization;
      }
      if (entry_bytes_ != header.entry_size || code_bits_ != header.code_bits ||
          std::memcmp(&quantization_, &header.quantization, sizeof(Quantization)) != 0) {
        return fail(error, "RaBitQ RFQ5 sidecar layout differs across shards");
      }
      auto& entries = shards_[node];
      entries.resize(header.entry_count * entry_bytes_);
      override_bits_[node].assign((header.entry_count + 63u) / 64u, 0);
      input.read(reinterpret_cast<char*>(entries.data()),
                 static_cast<std::streamsize>(entries.size()));
      if (!input.good()) return fail(error, "truncated RFQ5 RaBitQ sidecar: " + path.string());
      size_bytes_ += entries.size();
      entry_count_ += header.entry_count;
    }
    node_size_ = expected_node_size;
    const size_t raw_bytes = entry_count_ * VamanaNode::vector_bytes();
    const size_t total_budget = max_cache_ratio > 0.0
      ? static_cast<size_t>(std::floor(static_cast<f64>(raw_bytes) * max_cache_ratio))
      : 0;
    override_bitmap_bytes_ = 0;
    for (const auto& bits : override_bits_) {
      override_bitmap_bytes_ += bits.size() * sizeof(u64);
    }
    const size_t static_budget_bytes = size_bytes_ + override_bitmap_bytes_;
    const size_t capped_dynamic_budget =
      total_budget > static_budget_bytes
        ? std::min(dynamic_budget_bytes, total_budget - static_budget_bytes)
        : 0;
    init_dynamic(capped_dynamic_budget);
    prewarm();
    return true;
  }

  f32 estimate_distance_lut(const QueryLut& lut, f32 query_norm2,
                            const byte_t* entry) const {
    return rabitq::estimate_distance_lut(lut, query_norm2, entry, quantization_);
  }

  f32 lower_bound_lut(const QueryLut& lut, f32 query_norm2,
                      const byte_t* entry) const {
    return rabitq::lower_bound_lut(lut, query_norm2, entry, quantization_);
  }

  void estimate_batch_lut(const QueryLut& lut, f32 query_norm2,
                          const vec<RemotePtr>& pointers,
                          u32 begin, u32 count,
                          vec<f32>& distances,
                          vec<u32>& cache_miss_indices,
                          vec<const byte_t*>& entries) const {
    constexpr u32 prefetch_distance = 8;
    distances.resize(count);
    cache_miss_indices.clear();
    entries.resize(count);
    for (u32 step = 0; step < count + prefetch_distance; ++step) {
      if (step < count) {
        const byte_t* entry = find(pointers[begin + step]);
        entries[step] = entry;
        if (entry == nullptr) {
          cache_miss_indices.push_back(step);
        } else {
          __builtin_prefetch(entry, 0, 1);
        }
      }
      if (step >= prefetch_distance) {
        const u32 score_index = step - prefetch_distance;
        const byte_t* entry = entries[score_index];
        if (entry != nullptr) {
          distances[score_index] =
            rabitq::estimate_distance_lut(lut, query_norm2, entry, quantization_);
        }
      }
    }
  }

  const byte_t* find(RemotePtr pointer) const {
    if (pointer.memory_node() < shards_.size() && pointer.byte_offset() >= 16 && node_size_ != 0) {
      const u64 relative = pointer.byte_offset() - 16;
      if (relative % node_size_ == 0) {
        const u64 slot = relative / node_size_;
        const auto& entries = shards_[pointer.memory_node()];
        const u64 offset = slot * entry_bytes_;
        if (entry_bytes_ != 0 && offset + entry_bytes_ <= entries.size()) {
          if (!has_dynamic_entries_.load(std::memory_order_acquire)) {
            return entries.data() + offset;
          }
          const auto& bits = override_bits_[pointer.memory_node()];
          const size_t word = static_cast<size_t>(slot >> 6u);
          const u64 mask = 1ull << (slot & 63u);
          if (word < bits.size() &&
              (std::atomic_ref<const u64>(bits[word]).load(std::memory_order_acquire) & mask) != 0) {
            return find_dynamic(pointer);
          }
          return entries.data() + offset;
        }
      }
    }
    if (!has_dynamic_entries_.load(std::memory_order_acquire)) return nullptr;
    return find_dynamic(pointer);
  }

  bool upsert_dynamic(RemotePtr pointer, const byte_t* vector, VectorDType dtype) {
    if (dynamic_slots_.empty() || pointer.is_null() || entry_bytes_ < 2) return false;
    thread_local vec<byte_t> entry;
    entry.assign(entry_bytes_, 0);
    if (!encode_into(vector, dtype, quantization_, code_bits_, entry_bytes_, entry.data())) {
      ++dynamic_overflow_;
      return false;
    }
    std::unique_lock lock(dynamic_mutex_);
    const u64 raw = pointer.raw_address;
    const u64 mask = static_cast<u64>(dynamic_slots_.size() - 1);
    size_t first_deleted = dynamic_slots_.size();
    for (u64 probe = 0; probe < dynamic_slots_.size(); ++probe) {
      const size_t slot_index = (hash_raw(raw) + probe) & mask;
      auto& slot = dynamic_slots_[slot_index];
      if (slot.state == 1 && slot.raw == raw) {
        std::memcpy(dynamic_entries_.data() + slot_index * entry_bytes_,
                    entry.data(), entry_bytes_);
        has_dynamic_entries_.store(true, std::memory_order_release);
        mark_static_override(pointer);
        return true;
      }
      if (slot.state == 2 && first_deleted == dynamic_slots_.size()) {
        first_deleted = slot_index;
      }
      if (slot.state == 0) {
        const size_t target_index = first_deleted == dynamic_slots_.size()
          ? slot_index : first_deleted;
        auto& target = dynamic_slots_[target_index];
        target.raw = raw;
        target.state = 1;
        std::memcpy(dynamic_entries_.data() + target_index * entry_bytes_,
                    entry.data(), entry_bytes_);
        ++dynamic_live_;
        has_dynamic_entries_.store(true, std::memory_order_release);
        mark_static_override(pointer);
        return true;
      }
    }
    if (first_deleted != dynamic_slots_.size()) {
      auto& target = dynamic_slots_[first_deleted];
      target.raw = raw;
      target.state = 1;
      std::memcpy(dynamic_entries_.data() + first_deleted * entry_bytes_,
                  entry.data(), entry_bytes_);
      ++dynamic_live_;
      has_dynamic_entries_.store(true, std::memory_order_release);
      mark_static_override(pointer);
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
  size_t dynamic_size_bytes() const {
    return dynamic_slots_.size() * sizeof(DynamicSlot) + dynamic_entries_.size();
  }
  size_t decode_table_bytes() const { return 0; }
  size_t override_bitmap_bytes() const { return override_bitmap_bytes_; }
  size_t total_size_bytes() const {
    return size_bytes_ + override_bitmap_bytes_ + dynamic_size_bytes();
  }
  size_t entry_count() const { return entry_count_; }
  size_t entry_bytes() const { return entry_bytes_; }
  size_t code_bits() const { return code_bits_; }
  size_t code_bytes() const { return code_bytes_; }
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

  void mark_static_override(RemotePtr pointer) {
    if (pointer.memory_node() >= override_bits_.size() ||
        pointer.byte_offset() < 16 || node_size_ == 0) {
      return;
    }
    const u64 relative = pointer.byte_offset() - 16;
    if (relative % node_size_ != 0) return;
    const u64 slot = relative / node_size_;
    auto& bits = override_bits_[pointer.memory_node()];
    const size_t word = static_cast<size_t>(slot >> 6u);
    if (word >= bits.size()) return;
    const u64 mask = 1ull << (slot & 63u);
    std::atomic_ref<u64>(bits[word]).fetch_or(mask, std::memory_order_release);
  }

  void init_dynamic(size_t budget_bytes) {
    if (entry_bytes_ == 0) return;
    const size_t per_slot = sizeof(DynamicSlot) + entry_bytes_;
    if (budget_bytes < per_slot * 2) return;
    size_t capacity = 1;
    const size_t requested = budget_bytes / per_slot;
    while (capacity * 2 <= requested) capacity *= 2;
    dynamic_slots_.assign(capacity, {});
    dynamic_entries_.assign(capacity * entry_bytes_, 0);
  }

  const byte_t* find_dynamic(RemotePtr pointer) const {
    if (dynamic_slots_.empty() || pointer.is_null() || entry_bytes_ == 0) return nullptr;
    std::shared_lock lock(dynamic_mutex_);
    const u64 raw = pointer.raw_address;
    const u64 mask = static_cast<u64>(dynamic_slots_.size() - 1);
    for (u64 probe = 0; probe < dynamic_slots_.size(); ++probe) {
      const size_t slot_index = (hash_raw(raw) + probe) & mask;
      const auto& slot = dynamic_slots_[slot_index];
      if (slot.state == 0) return nullptr;
      if (slot.raw == raw) {
        return slot.state == 1
          ? dynamic_entries_.data() + slot_index * entry_bytes_
          : nullptr;
      }
    }
    return nullptr;
  }

  void prewarm() const {
    volatile u8 sink = 0;
    for (const auto& shard : shards_) {
      for (size_t offset = 0; offset < shard.size(); offset += 4096) sink ^= shard[offset];
    }
    (void)sink;
  }

  static bool fail(str* error, const str& message) {
    if (error != nullptr) *error = message;
    return false;
  }

  vec<vec<byte_t>> shards_;
  vec<vec<u64>> override_bits_;
  std::atomic<bool> has_dynamic_entries_{false};
  mutable std::shared_mutex dynamic_mutex_;
  vec<DynamicSlot> dynamic_slots_;
  vec<byte_t> dynamic_entries_;
  Quantization quantization_{};
  u32 node_size_{};
  u32 entry_bytes_{};
  u32 code_bits_{};
  u32 code_bytes_{};
  size_t size_bytes_{};
  size_t override_bitmap_bytes_{};
  size_t entry_count_{};
  size_t dynamic_live_{};
  size_t dynamic_overflow_{};
  bool numa_interleaved_{};
};

}  // namespace vamana::rabitq
