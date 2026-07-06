#include "vamana/anchor_index.hh"

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>

#include "common/distance.hh"
#include "common/index_path.hh"

namespace vamana::anchor {

namespace {

bool fail(str* error_message, const str& message) {
  if (error_message != nullptr) {
    *error_message = message;
  }
  return false;
}

}  // namespace

bool Index::load(const filepath_t& index_prefix,
                 u32 expected_dim,
                 u32 expected_shards,
                 str* error_message) {
  dim_ = 0;
  total_anchors_ = 0;
  shards_.clear();

  const filepath_t path = index_path::anchor_file(index_prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return fail(error_message, "missing anchor sidecar: " + path.string());
  }

  Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || header.magic != kMagic || header.version != kVersion ||
      header.dim != expected_dim || header.shard_count != expected_shards) {
    return fail(error_message, "invalid or incompatible anchor sidecar: " + path.string());
  }

  VectorDType dtype;
  try {
    switch (static_cast<VectorDType>(header.vector_dtype)) {
      case VectorDType::float32:
      case VectorDType::uint8:
      case VectorDType::int8:
        dtype = static_cast<VectorDType>(header.vector_dtype);
        break;
      default:
        return fail(error_message, "invalid anchor sidecar dtype: " + path.string());
    }
    if (vector_dtype_bytes(dtype, header.dim) != header.vector_bytes) {
      return fail(error_message, "anchor sidecar vector layout mismatch: " + path.string());
    }
  } catch (const std::exception& e) {
    return fail(error_message, "invalid anchor sidecar dtype: " + str{e.what()});
  }

  dim_ = header.dim;
  shards_.resize(header.shard_count);
  vec<byte_t> raw(header.vector_bytes);
  size_t loaded = 0;
  for (u32 expected_shard = 0; expected_shard < header.shard_count; ++expected_shard) {
    ShardHeader shard_header;
    input.read(reinterpret_cast<char*>(&shard_header), sizeof(shard_header));
    if (!input.good() || shard_header.shard != expected_shard) {
      return fail(error_message, "invalid anchor shard header: " + path.string());
    }
    if (shard_header.anchor_count > header.anchors_per_shard ||
        loaded + shard_header.anchor_count > header.total_anchors) {
      return fail(error_message, "invalid anchor shard count: " + path.string());
    }

    auto& shard = shards_[expected_shard];
    shard.centroid.resize(dim_);
    input.read(reinterpret_cast<char*>(shard.centroid.data()),
               static_cast<std::streamsize>(dim_ * sizeof(element_t)));
    shard.vectors.resize(static_cast<size_t>(shard_header.anchor_count) * dim_);
    shard.pointers.reserve(shard_header.anchor_count);
    for (u32 i = 0; i < shard_header.anchor_count; ++i) {
      EntryHeader entry;
      input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
      input.read(reinterpret_cast<char*>(raw.data()),
                 static_cast<std::streamsize>(raw.size()));
      if (!input.good()) {
        return fail(error_message, "truncated anchor sidecar: " + path.string());
      }
      decode_storage_vector_to_float(raw.data(), dtype, dim_,
                                     shard.vectors.data() + static_cast<size_t>(i) * dim_);
      shard.pointers.emplace_back(entry.rptr_raw);
      ++loaded;
    }
  }

  if (loaded != header.total_anchors) {
    return fail(error_message, "anchor sidecar count mismatch: " + path.string());
  }
  total_anchors_ = loaded;
  return true;
}

size_t Index::memory_bytes() const {
  size_t bytes = 0;
  for (const auto& shard : shards_) {
    bytes += shard.centroid.size() * sizeof(element_t);
    bytes += shard.vectors.size() * sizeof(element_t);
    bytes += shard.pointers.size() * sizeof(RemotePtr);
  }
  return bytes;
}

u32 Index::nearest_shard(const span<const element_t> query) const {
  u32 best_shard = 0;
  distance_t best_distance = std::numeric_limits<distance_t>::max();
  for (u32 shard = 0; shard < shards_.size(); ++shard) {
    if (shards_[shard].centroid.size() != dim_) {
      continue;
    }
    const distance_t distance = L2Distance::dist(query, shards_[shard].centroid, dim_);
    if (distance < best_distance) {
      best_distance = distance;
      best_shard = shard;
    }
  }
  return best_shard;
}

vec<u32> Index::nearest_shards(const span<const element_t> query, u32 count) const {
  vec<u32> result;
  if (count == 0 || shards_.empty() || query.size() != dim_) {
    return result;
  }
  using Candidate = std::pair<distance_t, u32>;
  std::priority_queue<Candidate> nearest;
  for (u32 shard = 0; shard < shards_.size(); ++shard) {
    if (shards_[shard].centroid.size() != dim_) {
      continue;
    }
    const distance_t distance = L2Distance::dist(query, shards_[shard].centroid, dim_);
    if (nearest.size() < count) {
      nearest.emplace(distance, shard);
    } else if (distance < nearest.top().first) {
      nearest.pop();
      nearest.emplace(distance, shard);
    }
  }
  result.resize(nearest.size());
  for (size_t pos = result.size(); pos > 0; --pos) {
    result[pos - 1] = nearest.top().second;
    nearest.pop();
  }
  return result;
}

vec<RemotePtr> Index::nearest_anchors(const span<const element_t> query,
                                      u32 shard_id,
                                      u32 count) const {
  vec<RemotePtr> result;
  if (count == 0 || shard_id >= shards_.size()) {
    return result;
  }
  const auto& shard = shards_[shard_id];
  using Candidate = std::pair<distance_t, u32>;
  std::priority_queue<Candidate> nearest;
  for (u32 i = 0; i < shard.pointers.size(); ++i) {
    const span<const element_t> anchor{
      shard.vectors.data() + static_cast<size_t>(i) * dim_, dim_};
    const distance_t distance = L2Distance::dist(query, anchor, dim_);
    if (nearest.size() < count) {
      nearest.emplace(distance, i);
    } else if (distance < nearest.top().first) {
      nearest.pop();
      nearest.emplace(distance, i);
    }
  }
  result.resize(nearest.size());
  for (size_t pos = result.size(); pos > 0; --pos) {
    result[pos - 1] = shard.pointers[nearest.top().second];
    nearest.pop();
  }
  return result;
}

Route Index::route(const span<const element_t> query,
                   u32 hint_count,
                   std::optional<u32> owner_override) const {
  Route route;
  if (shards_.empty() || query.size() != dim_) {
    return route;
  }
  const u32 semantic_shard = nearest_shard(query);
  route.owner = owner_override.has_value() ? *owner_override : semantic_shard;
  if (route.owner == semantic_shard) {
    route.hints = nearest_anchors(query, semantic_shard, hint_count);
    return route;
  }

  const u32 local_count = (hint_count + 1) / 2;
  route.hints = nearest_anchors(query, route.owner, local_count);
  vec<RemotePtr> semantic = nearest_anchors(query, semantic_shard, hint_count - local_count);
  for (const RemotePtr hint : semantic) {
    if (std::find(route.hints.begin(), route.hints.end(), hint) == route.hints.end()) {
      route.hints.push_back(hint);
    }
  }
  return route;
}

}  // namespace vamana::anchor
