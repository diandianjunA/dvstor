#include "vamana/anchor_index.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>

#include "common/distance.hh"
#include "common/index_path.hh"
#include "service/storage_owner_protocol.hh"

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
  std::unique_lock lock(mutex_);
  dim_ = 0;
  version_ = kVersion;
  route_epoch_ = kVersion;
  total_anchors_ = 0;
  shards_.clear();

  const filepath_t path = index_path::anchor_file(index_prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return fail(error_message, "missing anchor sidecar: " + path.string());
  }

  Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || header.magic != kMagic ||
      (header.version != kVersion && header.version != kLegacyVersion) ||
      header.dim != expected_dim || header.shard_count != expected_shards) {
    return fail(error_message, "invalid or incompatible anchor sidecar: " + path.string());
  }
  version_ = header.version;
  route_epoch_ = header.version;

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
    shard.static_anchor_count = shard_header.anchor_count;
  }

  if (loaded != header.total_anchors) {
    return fail(error_message, "anchor sidecar count mismatch: " + path.string());
  }
  total_anchors_ = loaded;
  return true;
}

size_t Index::memory_bytes() const {
  std::shared_lock lock(mutex_);
  size_t bytes = 0;
  for (const auto& shard : shards_) {
    bytes += shard.centroid.size() * sizeof(element_t);
    bytes += shard.vectors.size() * sizeof(element_t);
    bytes += shard.pointers.size() * sizeof(RemotePtr);
  }
  return bytes;
}

bool Index::add_runtime_anchor(u32 shard_id,
                               const span<const element_t> vector,
                               RemotePtr pointer,
                               u32 max_runtime_anchors_per_shard) {
  if (max_runtime_anchors_per_shard == 0 || pointer.is_null() ||
      shard_id >= shards_.size() || vector.size() != dim_) {
    return false;
  }

  std::unique_lock lock(mutex_);
  auto& shard = shards_[shard_id];
  const size_t dynamic_count = shard.pointers.size() >= shard.static_anchor_count
                                 ? shard.pointers.size() - shard.static_anchor_count
                                 : 0;
  if (dynamic_count >= max_runtime_anchors_per_shard) {
    const size_t evict_index = shard.static_anchor_count;
    shard.pointers.erase(shard.pointers.begin() + static_cast<idx_t>(evict_index));
    const auto vec_begin = shard.vectors.begin() + static_cast<idx_t>(evict_index * dim_);
    shard.vectors.erase(vec_begin, vec_begin + dim_);
    --total_anchors_;
  }

  shard.pointers.push_back(pointer);
  shard.vectors.insert(shard.vectors.end(), vector.begin(), vector.end());
  ++total_anchors_;
  ++route_epoch_;
  return true;
}

distance_t Index::shard_distance(const span<const element_t> query, u32 shard_id) const {
  if (shard_id >= shards_.size()) {
    return std::numeric_limits<distance_t>::max();
  }
  const auto& shard = shards_[shard_id];
  distance_t best_distance = std::numeric_limits<distance_t>::max();
  if (shard.centroid.size() == dim_) {
    best_distance = L2Distance::dist(query, shard.centroid, dim_);
  }
  for (u32 i = 0; i < shard.pointers.size(); ++i) {
    const span<const element_t> anchor{
      shard.vectors.data() + static_cast<size_t>(i) * dim_, dim_};
    best_distance = std::min(best_distance, L2Distance::dist(query, anchor, dim_));
  }
  return best_distance;
}

Index::ShardScore Index::nearest_shard(const span<const element_t> query,
                                       distance_t* second_distance) const {
  ShardScore best{0, std::numeric_limits<distance_t>::max()};
  distance_t second = std::numeric_limits<distance_t>::max();
  if (second_distance != nullptr) {
    *second_distance = second;
  }
  if (shards_.empty()) {
    return best;
  }
  distance_t best_distance = std::numeric_limits<distance_t>::max();
  for (u32 shard = 0; shard < shards_.size(); ++shard) {
    const distance_t distance = shard_distance(query, shard);
    if (distance < best_distance) {
      second = best_distance;
      best_distance = distance;
      best = ShardScore{shard, distance};
    } else if (distance < second) {
      second = distance;
    }
  }
  if (second_distance != nullptr) {
    *second_distance = second;
  }
  return best;
}

vec<Index::ShardScore> Index::top_shards(const span<const element_t> query, u32 count) const {
  vec<ShardScore> scores;
  if (count == 0 || shards_.empty()) {
    return scores;
  }
  scores.reserve(shards_.size());
  for (u32 shard = 0; shard < shards_.size(); ++shard) {
    scores.push_back(ShardScore{shard, shard_distance(query, shard)});
  }
  const u32 keep = std::min<u32>(count, static_cast<u32>(scores.size()));
  std::partial_sort(scores.begin(), scores.begin() + keep, scores.end(),
                    [](const ShardScore& lhs, const ShardScore& rhs) {
                      return lhs.distance < rhs.distance;
                    });
  scores.resize(keep);
  return scores;
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
                   u32 top_owners,
                   std::optional<u32> owner_override) const {
  std::shared_lock lock(mutex_);
  Route route;
  if (shards_.empty() || query.size() != dim_) {
    return route;
  }
  const u32 owner_count = std::max<u32>(1, top_owners);
  vec<ShardScore> ranked_shards = top_shards(query, std::max<u32>(2, owner_count));
  if (ranked_shards.empty()) {
    return route;
  }
  const ShardScore semantic = ranked_shards.front();
  const distance_t second_distance = ranked_shards.size() > 1
                                       ? ranked_shards[1].distance
                                       : std::numeric_limits<distance_t>::max();
  const u32 semantic_shard = semantic.shard;
  route.semantic_owner = semantic_shard;
  route.anchor_version = route_epoch_;
  if (std::isfinite(second_distance) && second_distance > 0.0f) {
    route.confidence = static_cast<f32>(
      std::clamp((second_distance - semantic.distance) / second_distance, 0.0f, 1.0f));
  } else {
    route.confidence = shards_.size() <= 1 ? 1.0f : 0.0f;
  }
  route.owner = owner_override.has_value() ? *owner_override : semantic_shard;
  if (owner_override.has_value() && *owner_override != semantic_shard) {
    route.flags |= service::storage_owner::kRouteFlagOwnerOverride;
  }

  vec<u32> hint_shards;
  hint_shards.reserve(ranked_shards.size() + 1);
  hint_shards.push_back(route.owner);
  const u32 hint_owner_count = std::min<u32>(owner_count, static_cast<u32>(ranked_shards.size()));
  for (u32 i = 0; i < hint_owner_count; ++i) {
    const ShardScore& score = ranked_shards[i];
    if (std::find(hint_shards.begin(), hint_shards.end(), score.shard) == hint_shards.end()) {
      hint_shards.push_back(score.shard);
    }
  }

  const u32 per_shard_min = std::max<u32>(1, hint_count / std::max<u32>(1, hint_shards.size()));
  for (const u32 shard : hint_shards) {
    if (route.hints.size() >= hint_count) {
      break;
    }
    const u32 remaining = hint_count - static_cast<u32>(route.hints.size());
    const u32 shard_budget = std::max<u32>(per_shard_min, remaining == hint_count ? (hint_count + 1) / 2 : 1);
    vec<RemotePtr> shard_hints = nearest_anchors(query, shard, std::min(remaining, shard_budget));
    for (const RemotePtr hint : shard_hints) {
      if (std::find(route.hints.begin(), route.hints.end(), hint) == route.hints.end()) {
        route.hints.push_back(hint);
        if (route.hints.size() >= hint_count) {
          break;
        }
      }
    }
  }
  if (route.hints.size() < hint_count) {
    for (const u32 shard : hint_shards) {
      vec<RemotePtr> shard_hints = nearest_anchors(query, shard, hint_count);
      for (const RemotePtr hint : shard_hints) {
        if (std::find(route.hints.begin(), route.hints.end(), hint) == route.hints.end()) {
          route.hints.push_back(hint);
          if (route.hints.size() >= hint_count) {
            return route;
          }
        }
      }
    }
  }
  return route;
}

}  // namespace vamana::anchor
