#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/vector_dtype.hh"
#include "gpu_search/centroid_home_selector.hh"

namespace {

template <typename Function>
bool throws_invalid_argument(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  namespace home = gpu_search::centroid_home;

  const std::vector<f32> scalar_query{0.0f};
  assert(!home::select(scalar_query, {}).has_value());

  const home::Snapshot two_dimensional{
    {.vector_count = 1, .centroid = {0.0, 0.0}, .live_entry_count = 1},
  };
  assert(throws_invalid_argument([&] {
    (void)home::select(scalar_query, two_dimensional);
  }));

  const std::vector<f32> finite_shape_query{0.0f, 0.0f};
  for (const f32 invalid : {
         std::numeric_limits<f32>::quiet_NaN(),
         std::numeric_limits<f32>::infinity(),
         -std::numeric_limits<f32>::infinity()}) {
    const std::vector<f32> query{invalid, 0.0f};
    assert(throws_invalid_argument([&] {
      (void)home::select(query, two_dimensional);
    }));
  }

  // Empty shards and shards without a live entry are not eligible, even when
  // their centroid would otherwise be nearest.
  const home::Snapshot sparse{
    {.vector_count = 0, .centroid = {0.0, 0.0}, .live_entry_count = 1},
    {.vector_count = 8, .centroid = {0.0, 0.0}, .live_entry_count = 0},
    {.vector_count = 9, .centroid = {4.0, 3.0}, .live_entry_count = 2},
  };
  const auto sparse_home = home::select(finite_shape_query, sparse);
  assert(sparse_home.has_value() && *sparse_home == 2);

  const home::Snapshot no_eligible_shard{
    {.vector_count = 0, .centroid = {1.0, 2.0}, .live_entry_count = 0},
    {.vector_count = 7, .centroid = {3.0, 4.0}, .live_entry_count = 0},
  };
  assert(!home::select(finite_shape_query, no_eligible_shard).has_value());

  // Storage maintains exact FP64 sums, then deliberately publishes FP32. Both
  // offsets disappear in that canonical representation, so CPU insert routing
  // must retain shard zero on the same tie as GPU query routing (rather than
  // privately using the unpublished FP64 precision and choosing shard one).
  const f32 large_query_value = 1.0e20f;
  const double large_query_as_double = static_cast<double>(large_query_value);
  assert(large_query_as_double + 1.0e12 != large_query_as_double);
  const std::vector<f32> large_query{large_query_value};
  const f32 published_far = static_cast<f32>(
    large_query_as_double + 2.0e12);
  const f32 published_near = static_cast<f32>(
    large_query_as_double + 1.0e12);
  assert(published_far == large_query_value);
  assert(published_near == large_query_value);
  const home::Snapshot canonical_precision{
    {.vector_count = 1,
     .centroid = {published_far},
     .live_entry_count = 1},
    {.vector_count = 1,
     .centroid = {published_near},
     .live_entry_count = 1},
  };
  const auto canonical_home = home::select(
    large_query, canonical_precision);
  assert(canonical_home.has_value() && *canonical_home == 0);

  // Guard the accumulation precision independently of centroid publication.
  // The second coordinate changes the mathematical/FP64 distance, but its
  // square is below half an FP32 ulp at 1.0 and is therefore lost by the same
  // fmaf recurrence used in the GPU kernel. Both routes must see the FP32 tie.
  constexpr f32 sub_ulp_coordinate = 1.0f / 8192.0f;
  const home::Snapshot accumulation_precision{
    {.vector_count = 1,
     .centroid = {1.0f, sub_ulp_coordinate},
     .live_entry_count = 1},
    {.vector_count = 1,
     .centroid = {1.0f, 0.0f},
     .live_entry_count = 1},
  };
  const double fp64_first_distance = 1.0 +
    static_cast<double>(sub_ulp_coordinate) * sub_ulp_coordinate;
  assert(fp64_first_distance > 1.0);
  const auto accumulation_home = home::select(
    finite_shape_query, accumulation_precision);
  assert(accumulation_home.has_value() && *accumulation_home == 0);

  const home::Snapshot exact_tie{
    {.vector_count = 1, .centroid = {-1.0}, .live_entry_count = 1},
    {.vector_count = 1, .centroid = {1.0}, .live_entry_count = 1},
  };
  const auto tie_home = home::select(scalar_query, exact_tie);
  assert(tie_home.has_value() && *tie_home == 0);

  // Integer indexes route the exact canonical value that will be stored, not
  // the caller's pre-quantized float. Otherwise Stage1 can select a different
  // home from the centroid space used by the physical index.
  const home::Snapshot quantized_centroids{
    {.vector_count = 10, .centroid = {0.0}, .live_entry_count = 1},
    {.vector_count = 10, .centroid = {0.6}, .live_entry_count = 1},
  };
  const std::vector<f32> fractional{0.4f};
  assert(home::select(fractional, quantized_centroids) == 1);
  const vec<byte_t> encoded_u8 = encode_float_vector_to_storage(
    span<const element_t>{fractional}, VectorDType::uint8);
  const vec<f32> canonical_u8 = decode_storage_vector_to_float(
    encoded_u8.data(), VectorDType::uint8, 1);
  assert(canonical_u8[0] == 0.0f);
  assert(home::select(canonical_u8, quantized_centroids) == 0);

  const home::Snapshot signed_centroids{
    {.vector_count = 10, .centroid = {-1.0}, .live_entry_count = 1},
    {.vector_count = 10, .centroid = {-0.4}, .live_entry_count = 1},
  };
  const std::vector<f32> signed_fractional{-0.6f};
  assert(home::select(signed_fractional, signed_centroids) == 1);
  const vec<byte_t> encoded_i8 = encode_float_vector_to_storage(
    span<const element_t>{signed_fractional}, VectorDType::int8);
  const vec<f32> canonical_i8 = decode_storage_vector_to_float(
    encoded_i8.data(), VectorDType::int8, 1);
  assert(canonical_i8[0] == -1.0f);
  assert(home::select(canonical_i8, signed_centroids) == 0);

  return 0;
}
