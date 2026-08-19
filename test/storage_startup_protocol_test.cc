#include <cassert>

#include "memory_node/startup_protocol.hh"

namespace {

using storage_startup::Request;

constexpr u32 kBaselineModes = storage_startup::encode_feature_modes(
  "coupled", "fixed", "coupled");
constexpr u32 kFullModes = storage_startup::encode_feature_modes(
  "decoupled", "adaptive", "decoupled");

Request matching_request() {
  return Request{
    .feature_modes = kFullModes,
    .schema_version = 16,
    .expected_shard = 3,
    .expected_shard_count = 5,
    .expected_vector_id_namespace_size = 200'000'000,
    .index_build_fingerprint = 0x1122334455667788ull,
    .shard_build_fingerprint = 0x8877665544332211ull,
  };
}

storage_startup::Response evaluate(const Request& request) {
  return storage_startup::evaluate_request(
    request, kFullModes, 16, 3, 5, 200'000'000,
    0x1122334455667788ull, 0x8877665544332211ull);
}

void expect_mismatch(Request request, u32 flag) {
  const auto response = evaluate(request);
  assert(storage_startup::valid_response_envelope(response));
  assert(response.ready == 0);
  assert((response.mismatch_flags & flag) != 0);
  assert(response.feature_modes == kFullModes);
  assert(response.schema_version == 16);
  assert(response.shard == 3);
  assert(response.shard_count == 5);
  assert(response.vector_id_namespace_size == 200'000'000);
  assert(response.index_build_fingerprint == 0x1122334455667788ull);
  assert(response.shard_build_fingerprint == 0x8877665544332211ull);
}

}  // namespace

int main() {
  static_assert(kBaselineModes == 0);
  static_assert(kFullModes == 21);
  static_assert(storage_startup::valid_feature_modes(kBaselineModes));
  static_assert(storage_startup::valid_feature_modes(kFullModes));
  static_assert(storage_startup::valid_feature_modes(
    storage_startup::encode_feature_modes(
      "decoupled", "manual", "manual")));
  static_assert(storage_startup::encode_feature_modes(
    "invalid", "fixed", "coupled") ==
    storage_startup::kInvalidFeatureMask);
  static_assert(!storage_startup::valid_feature_modes(
    storage_startup::kInvalidFeatureMask));

  const auto accepted = evaluate(matching_request());
  assert(storage_startup::valid_response_envelope(accepted));
  assert(accepted.ready == 1);
  assert(accepted.mismatch_flags == 0);
  auto malformed_response = accepted;
  malformed_response.reserved = 1;
  assert(!storage_startup::valid_response_envelope(malformed_response));

  auto request = matching_request();
  request.magic = 0;
  expect_mismatch(request, storage_startup::request_envelope_mismatch);
  request = matching_request();
  request.version = 1;
  expect_mismatch(request, storage_startup::request_envelope_mismatch);
  request = matching_request();
  request.bytes = 0;
  expect_mismatch(request, storage_startup::request_envelope_mismatch);
  request = matching_request();
  request.reserved = 1;
  expect_mismatch(request, storage_startup::request_envelope_mismatch);
  request = matching_request();
  request.feature_modes = kBaselineModes;
  expect_mismatch(request, storage_startup::feature_modes_mismatch);
  request = matching_request();
  request.schema_version = 15;
  expect_mismatch(request, storage_startup::schema_mismatch);
  request = matching_request();
  request.expected_shard = 2;
  expect_mismatch(request, storage_startup::shard_identity_mismatch);
  request = matching_request();
  request.expected_shard_count = 4;
  expect_mismatch(request, storage_startup::shard_count_mismatch);
  request = matching_request();
  request.expected_vector_id_namespace_size = 100;
  expect_mismatch(request, storage_startup::vector_namespace_mismatch);
  request = matching_request();
  request.index_build_fingerprint ^= 1;
  expect_mismatch(request, storage_startup::index_build_mismatch);
  request = matching_request();
  request.shard_build_fingerprint ^= 1;
  expect_mismatch(request, storage_startup::shard_build_mismatch);

  return 0;
}
