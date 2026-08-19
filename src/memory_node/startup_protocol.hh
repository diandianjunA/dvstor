#pragma once

#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>

#include "common/types.hh"

namespace storage_startup {

inline constexpr u32 kMagic = 0x44565354;  // DVST
inline constexpr u16 kVersion = 2;
inline constexpr u16 kRequestBytes = 48;
inline constexpr u16 kResponseBytes = 56;

inline constexpr u32 kModeFieldMask = 0x3u;
inline constexpr u32 kUpdateModeShift = 0;
inline constexpr u32 kGraphAccessModeShift = 2;
inline constexpr u32 kSearchProgressionModeShift = 4;
inline constexpr u32 kKnownFeatureMask =
  (kModeFieldMask << kUpdateModeShift) |
  (kModeFieldMask << kGraphAccessModeShift) |
  (kModeFieldMask << kSearchProgressionModeShift);
inline constexpr u32 kInvalidFeatureMask =
  std::numeric_limits<u32>::max();

enum class UpdateCompletionMode : u32 {
  coupled = 0,
  decoupled = 1,
};

enum class GraphAccessMode : u32 {
  fixed = 0,
  adaptive = 1,
  manual = 2,
};

enum class SearchProgressionMode : u32 {
  coupled = 0,
  decoupled = 1,
  manual = 2,
};

inline constexpr u32 encode_feature_modes(
    std::string_view update_completion,
    std::string_view graph_access,
    std::string_view search_progression) {
  const u32 update = update_completion == "coupled"
    ? static_cast<u32>(UpdateCompletionMode::coupled)
    : update_completion == "decoupled"
      ? static_cast<u32>(UpdateCompletionMode::decoupled)
      : kInvalidFeatureMask;
  const u32 graph = graph_access == "fixed"
    ? static_cast<u32>(GraphAccessMode::fixed)
    : graph_access == "adaptive"
      ? static_cast<u32>(GraphAccessMode::adaptive)
      : graph_access == "manual"
        ? static_cast<u32>(GraphAccessMode::manual)
        : kInvalidFeatureMask;
  const u32 search = search_progression == "coupled"
    ? static_cast<u32>(SearchProgressionMode::coupled)
    : search_progression == "decoupled"
      ? static_cast<u32>(SearchProgressionMode::decoupled)
      : search_progression == "manual"
        ? static_cast<u32>(SearchProgressionMode::manual)
        : kInvalidFeatureMask;
  if (update == kInvalidFeatureMask || graph == kInvalidFeatureMask ||
      search == kInvalidFeatureMask) {
    return kInvalidFeatureMask;
  }
  return (update << kUpdateModeShift) |
    (graph << kGraphAccessModeShift) |
    (search << kSearchProgressionModeShift);
}

inline constexpr bool valid_feature_modes(u32 mask) {
  if ((mask & ~kKnownFeatureMask) != 0) return false;
  const u32 update = (mask >> kUpdateModeShift) & kModeFieldMask;
  const u32 graph = (mask >> kGraphAccessModeShift) & kModeFieldMask;
  const u32 search = (mask >> kSearchProgressionModeShift) & kModeFieldMask;
  return update <= static_cast<u32>(UpdateCompletionMode::decoupled) &&
    graph <= static_cast<u32>(GraphAccessMode::manual) &&
    search <= static_cast<u32>(SearchProgressionMode::manual);
}

enum MismatchFlag : u32 {
  request_envelope_mismatch = 1u << 0,
  feature_modes_mismatch = 1u << 1,
  schema_mismatch = 1u << 2,
  shard_identity_mismatch = 1u << 3,
  shard_count_mismatch = 1u << 4,
  index_build_mismatch = 1u << 5,
  shard_build_mismatch = 1u << 6,
  vector_namespace_mismatch = 1u << 7,
};

struct Request {
  u32 magic{kMagic};
  u16 version{kVersion};
  u16 bytes{kRequestBytes};
  u32 feature_modes{};
  u32 schema_version{};
  u32 expected_shard{};
  u32 expected_shard_count{};
  u32 expected_vector_id_namespace_size{};
  u32 reserved{};
  u64 index_build_fingerprint{};
  u64 shard_build_fingerprint{};
};

struct Response {
  u32 magic{kMagic};
  u16 version{kVersion};
  u16 bytes{kResponseBytes};
  u32 ready{};
  u32 mismatch_flags{};
  u32 feature_modes{};
  u32 schema_version{};
  u32 shard{};
  u32 shard_count{};
  u32 vector_id_namespace_size{};
  u32 reserved{};
  u64 index_build_fingerprint{};
  u64 shard_build_fingerprint{};
};

inline constexpr Response evaluate_request(
    const Request& request,
    u32 local_feature_modes,
    u32 local_schema_version,
    u32 local_shard,
    u32 local_shard_count,
    u32 local_vector_id_namespace_size,
    u64 local_index_build_fingerprint,
    u64 local_shard_build_fingerprint) {
  Response response{
    .feature_modes = local_feature_modes,
    .schema_version = local_schema_version,
    .shard = local_shard,
    .shard_count = local_shard_count,
    .vector_id_namespace_size = local_vector_id_namespace_size,
    .index_build_fingerprint = local_index_build_fingerprint,
    .shard_build_fingerprint = local_shard_build_fingerprint,
  };
  if (request.magic != kMagic || request.version != kVersion ||
      request.bytes != kRequestBytes || request.reserved != 0 ||
      !valid_feature_modes(request.feature_modes) ||
      !valid_feature_modes(local_feature_modes)) {
    response.mismatch_flags |= request_envelope_mismatch;
  }
  if (request.feature_modes != local_feature_modes) {
    response.mismatch_flags |= feature_modes_mismatch;
  }
  if (request.schema_version != local_schema_version) {
    response.mismatch_flags |= schema_mismatch;
  }
  if (request.expected_shard != local_shard) {
    response.mismatch_flags |= shard_identity_mismatch;
  }
  if (request.expected_shard_count != local_shard_count) {
    response.mismatch_flags |= shard_count_mismatch;
  }
  if (request.expected_vector_id_namespace_size !=
      local_vector_id_namespace_size) {
    response.mismatch_flags |= vector_namespace_mismatch;
  }
  if (request.index_build_fingerprint == 0 ||
      local_index_build_fingerprint == 0 ||
      request.index_build_fingerprint != local_index_build_fingerprint) {
    response.mismatch_flags |= index_build_mismatch;
  }
  if (request.shard_build_fingerprint == 0 ||
      local_shard_build_fingerprint == 0 ||
      request.shard_build_fingerprint != local_shard_build_fingerprint) {
    response.mismatch_flags |= shard_build_mismatch;
  }
  response.ready = response.mismatch_flags == 0 ? 1u : 0u;
  return response;
}

inline constexpr bool valid_response_envelope(const Response& response) {
  return response.magic == kMagic && response.version == kVersion &&
    response.bytes == kResponseBytes &&
    (response.ready == 0 || response.ready == 1) &&
    response.reserved == 0 &&
    valid_feature_modes(response.feature_modes);
}

static_assert(std::is_standard_layout_v<Request> &&
              std::is_trivially_copyable_v<Request>);
static_assert(std::is_standard_layout_v<Response> &&
              std::is_trivially_copyable_v<Response>);
static_assert(sizeof(Request) == 48);
static_assert(offsetof(Request, feature_modes) == 8);
static_assert(offsetof(Request, reserved) == 28);
static_assert(offsetof(Request, index_build_fingerprint) == 32);
static_assert(sizeof(Response) == 56);
static_assert(offsetof(Response, ready) == 8);
static_assert(offsetof(Response, reserved) == 36);
static_assert(offsetof(Response, index_build_fingerprint) == 40);

}  // namespace storage_startup
