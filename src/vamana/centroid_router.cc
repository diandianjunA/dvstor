#include "vamana/centroid_router.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vamana::routing {

namespace {

// Validate IEEE-754 exponent bits directly at every external
// centroid/vector boundary. Besides being branch-light, this keeps the wire
// validity contract independent of compiler and standard-library choices.
bool finite_f32(f32 value) {
  return (std::bit_cast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

bool finite_f64(f64 value) {
  return (std::bit_cast<u64>(value) & 0x7ff0000000000000ull) !=
         0x7ff0000000000000ull;
}

}  // namespace

CentroidRouter::CentroidRouter(u32 dim, u32 shard_count)
    : dim_(dim), shard_count_(shard_count), shards_(shard_count) {
  if (dim == 0 || shard_count == 0 || shard_count > (1u << 16)) {
    throw std::invalid_argument(
      "centroid router requires a non-zero dimension and 1..65536 shards");
  }
  std::atomic_store_explicit(
    &published_, build_snapshot_locked(), std::memory_order_release);
}

bool CentroidRouter::valid_vector(span<const f32> vector) const {
  if (vector.size() != dim_) return false;
  return std::all_of(vector.begin(), vector.end(), [](f32 value) {
    return finite_f32(value);
  });
}

void CentroidRouter::add_sum_component(ShardState& state, u32 dimension,
                                       f64 delta) {
  const f64 current = state.sum[dimension];
  const f64 next = current + delta;
  if (std::abs(current) >= std::abs(delta)) {
    state.sum_compensation[dimension] +=
      (current - next) + delta;
  } else {
    state.sum_compensation[dimension] +=
      (delta - next) + current;
  }
  state.sum[dimension] = next;
}

f64 CentroidRouter::compensated_sum_component(
    const ShardState& state, u32 dimension) {
  return state.sum[dimension] + state.sum_compensation[dimension];
}

bool CentroidRouter::restore_shard_state(
    u32 shard,
    u64 count,
    span<const f64> sum,
    span<const LiveEntry> live_entries,
    u64 version) {
  if (shard >= shard_count_ || version == 0 || sum.size() != dim_ ||
      live_entries.size() > kMaxLiveEntries ||
      live_entries.size() > count ||
      ((count == 0) != live_entries.empty())) {
    return false;
  }

  vec<f64> restored_sum(sum.begin(), sum.end());
  for (const f64 value : restored_sum) {
    if (!finite_f64(value) || (count == 0 && value != 0.0)) {
      return false;
    }
  }

  std::array<LiveEntry, kMaxLiveEntries> restored_entries{};
  for (size_t index = 0; index < live_entries.size(); ++index) {
    const LiveEntry& entry = live_entries[index];
    if (entry.pointer.is_null() || entry.pointer.memory_node() != shard) {
      return false;
    }
    for (size_t prior = 0; prior < index; ++prior) {
      if (live_entries[prior].pointer == entry.pointer) return false;
    }
    restored_entries[index] = entry;
  }

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& destination = shards_[shard];
  if (restoration_closed_ || destination.restored ||
      destination.version != 0 || destination.count != 0 ||
      destination.live_entry_count != 0) {
    return false;
  }
  destination.version = version;
  destination.count = count;
  destination.sum = std::move(restored_sum);
  destination.sum_compensation.assign(dim_, 0.0);
  destination.live_entries = restored_entries;
  destination.live_entry_count = static_cast<u32>(live_entries.size());
  destination.restored = true;
  state_version_ = std::max(state_version_, version);
  dirty_ = true;
  return true;
}

std::shared_ptr<const CentroidRouter::Snapshot>
CentroidRouter::build_snapshot_locked() const {
  auto result = std::make_shared<Snapshot>();
  result->dim = dim_;
  result->shard_count = shard_count_;
  result->version = state_version_;
  result->shards.reserve(shard_count_);
  for (u32 shard_index = 0; shard_index < shard_count_; ++shard_index) {
    const ShardState& source = shards_[shard_index];
    ShardSnapshot destination{
      .shard = shard_index,
      .version = source.version,
      .count = source.count,
      .sum = source.sum.empty() ? vec<f64>{} : vec<f64>(dim_, 0.0),
      .centroid = source.count == 0 ? vec<f64>{} : vec<f64>(dim_, 0.0),
      .live_entries = source.live_entries,
      .live_entry_count = source.live_entry_count,
    };
    if (!source.sum.empty()) {
      for (u32 dimension = 0; dimension < dim_; ++dimension) {
        destination.sum[dimension] =
          compensated_sum_component(source, dimension);
      }
    }
    if (source.count != 0) {
      const f64 inverse_count = 1.0 / static_cast<f64>(source.count);
      for (u32 dimension = 0; dimension < dim_; ++dimension) {
        destination.centroid[dimension] =
          destination.sum[dimension] * inverse_count;
      }
    }
    result->shards.push_back(std::move(destination));
  }
  return result;
}

bool CentroidRouter::insert(u32 shard, span<const f32> vector) {
  if (shard >= shard_count_ || !valid_vector(vector)) return false;

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& destination = shards_[shard];
  if (destination.count == std::numeric_limits<u64>::max() ||
      destination.version == std::numeric_limits<u64>::max() ||
      state_version_ == std::numeric_limits<u64>::max()) {
    return false;
  }
  if (destination.sum.empty()) {
    destination.sum.resize(dim_, 0.0);
    destination.sum_compensation.resize(dim_, 0.0);
  }
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    add_sum_component(
      destination, dimension, static_cast<f64>(vector[dimension]));
  }
  restoration_closed_ = true;
  ++destination.count;
  ++destination.version;
  ++state_version_;
  dirty_ = true;
  return true;
}

bool CentroidRouter::erase(u32 shard, span<const f32> vector) {
  if (shard >= shard_count_ || !valid_vector(vector)) return false;

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& source = shards_[shard];
  if (source.count == 0 ||
      source.version == std::numeric_limits<u64>::max() ||
      state_version_ == std::numeric_limits<u64>::max()) {
    return false;
  }
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    add_sum_component(
      source, dimension, -static_cast<f64>(vector[dimension]));
  }
  restoration_closed_ = true;
  --source.count;
  if (source.count == 0) {
    // Canonicalize an empty shard instead of retaining floating-point residue
    // from a long insert/erase stream.
    std::fill(source.sum.begin(), source.sum.end(), 0.0);
    std::fill(source.sum_compensation.begin(),
              source.sum_compensation.end(), 0.0);
    source.live_entries = {};
    source.live_entry_count = 0;
  }
  ++source.version;
  ++state_version_;
  dirty_ = true;
  return true;
}

bool CentroidRouter::upsert(
    u32 shard,
    span<const f32> old_vector,
    span<const f32> new_vector) {
  if (shard >= shard_count_ || !valid_vector(old_vector) ||
      !valid_vector(new_vector)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& destination = shards_[shard];
  if (destination.count == 0 ||
      destination.version == std::numeric_limits<u64>::max() ||
      state_version_ == std::numeric_limits<u64>::max()) {
    return false;
  }
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    add_sum_component(
      destination, dimension,
      static_cast<f64>(new_vector[dimension]) -
        static_cast<f64>(old_vector[dimension]));
  }
  restoration_closed_ = true;
  ++destination.version;
  ++state_version_;
  dirty_ = true;
  return true;
}

bool CentroidRouter::move(
    u32 source_shard,
    u32 destination_shard,
    span<const f32> vector) {
  if (source_shard >= shard_count_ || destination_shard >= shard_count_ ||
      source_shard == destination_shard || !valid_vector(vector)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& source = shards_[source_shard];
  ShardState& destination = shards_[destination_shard];
  if (source.count == 0 ||
      destination.count == std::numeric_limits<u64>::max() ||
      source.version == std::numeric_limits<u64>::max() ||
      destination.version == std::numeric_limits<u64>::max() ||
      state_version_ == std::numeric_limits<u64>::max()) {
    return false;
  }
  if (destination.sum.empty()) {
    destination.sum.resize(dim_, 0.0);
    destination.sum_compensation.resize(dim_, 0.0);
  }

  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    const f64 value = static_cast<f64>(vector[dimension]);
    add_sum_component(source, dimension, -value);
    add_sum_component(destination, dimension, value);
  }
  restoration_closed_ = true;
  --source.count;
  ++destination.count;
  if (source.count == 0) {
    std::fill(source.sum.begin(), source.sum.end(), 0.0);
    std::fill(source.sum_compensation.begin(),
              source.sum_compensation.end(), 0.0);
    source.live_entries = {};
    source.live_entry_count = 0;
  }
  ++source.version;
  ++destination.version;
  ++state_version_;
  dirty_ = true;
  return true;
}

bool CentroidRouter::replace_live_entries(
    u32 shard, span<const LiveEntry> entries) {
  if (shard >= shard_count_ ||
      entries.size() < kMinLiveEntries ||
      entries.size() > kMaxLiveEntries) {
    return false;
  }
  for (size_t index = 0; index < entries.size(); ++index) {
    const LiveEntry& entry = entries[index];
    if (entry.pointer.is_null() || entry.pointer.memory_node() != shard) {
      return false;
    }
    for (size_t prior = 0; prior < index; ++prior) {
      if (entries[prior].pointer == entry.pointer) return false;
    }
  }

  std::lock_guard<std::mutex> lock(writer_mutex_);
  ShardState& destination = shards_[shard];
  if (destination.count == 0 || entries.size() > destination.count ||
      destination.version == std::numeric_limits<u64>::max() ||
      state_version_ == std::numeric_limits<u64>::max()) {
    return false;
  }

  bool changed = destination.live_entry_count != entries.size();
  if (!changed) {
    for (size_t index = 0; index < entries.size(); ++index) {
      if (!(destination.live_entries[index] == entries[index])) {
        changed = true;
        break;
      }
    }
  }
  if (!changed) return false;

  destination.live_entries = {};
  std::copy(entries.begin(), entries.end(), destination.live_entries.begin());
  destination.live_entry_count = static_cast<u32>(entries.size());
  restoration_closed_ = true;
  ++destination.version;
  ++state_version_;
  dirty_ = true;
  return true;
}

bool CentroidRouter::publish() {
  std::lock_guard<std::mutex> lock(writer_mutex_);
  restoration_closed_ = true;
  if (!dirty_) return false;
  const std::shared_ptr<const Snapshot> next = build_snapshot_locked();
  std::atomic_store_explicit(
    &published_, next, std::memory_order_release);
  dirty_ = false;
  return true;
}

std::shared_ptr<const CentroidRouter::Snapshot>
CentroidRouter::snapshot() const {
  return std::atomic_load_explicit(&published_, std::memory_order_acquire);
}

u64 CentroidRouter::authoritative_count(u32 shard) const {
  if (shard >= shard_count_) {
    throw std::out_of_range("centroid shard is outside the router");
  }
  std::lock_guard<std::mutex> lock(writer_mutex_);
  return shards_[shard].count;
}

vec<f64> CentroidRouter::authoritative_centroid(u32 shard) const {
  if (shard >= shard_count_) {
    throw std::out_of_range("centroid shard is outside the router");
  }
  std::lock_guard<std::mutex> lock(writer_mutex_);
  const ShardState& state = shards_[shard];
  if (state.count == 0) return {};
  vec<f64> centroid(dim_);
  const f64 inverse_count = 1.0 / static_cast<f64>(state.count);
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    centroid[dimension] =
      compensated_sum_component(state, dimension) * inverse_count;
  }
  return centroid;
}

bool CentroidRouter::copy_authoritative_centroid(
    u32 shard, span<f64> destination) const {
  if (shard >= shard_count_) {
    throw std::out_of_range("centroid shard is outside the router");
  }
  if (destination.size() != dim_) {
    throw std::invalid_argument("centroid destination dimension mismatch");
  }
  std::lock_guard<std::mutex> lock(writer_mutex_);
  const ShardState& state = shards_[shard];
  if (state.count == 0) return false;
  const f64 inverse_count = 1.0 / static_cast<f64>(state.count);
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    destination[dimension] =
      compensated_sum_component(state, dimension) * inverse_count;
  }
  return true;
}

}  // namespace vamana::routing
