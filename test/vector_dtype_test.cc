#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "common/vector_dtype.hh"

namespace {

bool encoding_rejects(f32 value, VectorDType dtype) {
  std::array<f32, 1> input{value};
  std::array<byte_t, sizeof(f32)> output{};
  try {
    encode_float_vector_to_storage(
      input.data(), 1, dtype, output.data());
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

f32 scalar_same_dtype_distance(const vec<byte_t>& lhs,
                               const vec<byte_t>& rhs,
                               VectorDType dtype,
                               u32 dim) {
  f64 sum = 0.0;
  for (u32 index = 0; index < dim; ++index) {
    const f64 difference =
      static_cast<f64>(vector_component_as_float(lhs.data(), dtype, index)) -
      static_cast<f64>(vector_component_as_float(rhs.data(), dtype, index));
    sum += difference * difference;
  }
  return saturate_squared_l2(sum);
}

void test_integer_simd_matches_canonical_distance_for_arbitrary_dimensions() {
  constexpr std::array<u32, 11> dimensions{
    1, 7, 31, 32, 127, 128, 255, 256, 257, 4097, 65537};
  for (const VectorDType dtype :
       {VectorDType::uint8, VectorDType::int8}) {
    for (const u32 dim : dimensions) {
      vec<byte_t> lhs(dim);
      vec<byte_t> rhs(dim);
      for (u32 index = 0; index < dim; ++index) {
        if (dtype == VectorDType::uint8) {
          lhs[index] = static_cast<byte_t>((index * 131u + 17u) & 0xffu);
          rhs[index] = static_cast<byte_t>((index * 29u + 251u) & 0xffu);
        } else {
          reinterpret_cast<i8*>(lhs.data())[index] = static_cast<i8>(
            static_cast<i32>((index * 131u + 17u) & 0xffu) - 128);
          reinterpret_cast<i8*>(rhs.data())[index] = static_cast<i8>(
            static_cast<i32>((index * 29u + 251u) & 0xffu) - 128);
        }
      }
      const f32 expected =
        scalar_same_dtype_distance(lhs, rhs, dtype, dim);
      const f32 actual = typed_l2_distance(
        lhs.data(), dtype, rhs.data(), dtype, dim);
      assert(actual == expected);

      vec<f32> fractional_query(dim);
      for (u32 index = 0; index < dim; ++index) {
        fractional_query[index] =
          vector_component_as_float(lhs.data(), dtype, index) +
          (index % 2 == 0 ? 0.25F : -0.5F);
      }
      const f32 scalar_query = typed_l2_distance_float_query_scalar(
        span<const f32>{fractional_query}, rhs.data(), dtype, dim);
      const f32 vectorized_query = typed_l2_distance_float_query(
        span<const f32>{fractional_query}, rhs.data(), dtype, dim);
      const f32 tolerance =
        std::max(1.0F, scalar_query) * 2.0e-5F;
      assert(std::abs(vectorized_query - scalar_query) <= tolerance);
    }
  }
}

}  // namespace

int main() {
  for (const f32 invalid : {
         std::numeric_limits<f32>::quiet_NaN(),
         std::numeric_limits<f32>::infinity(),
         -std::numeric_limits<f32>::infinity()}) {
    assert(encoding_rejects(invalid, VectorDType::float32));
    assert(encoding_rejects(invalid, VectorDType::uint8));
    assert(encoding_rejects(invalid, VectorDType::int8));
  }

  const std::array<f32, 2> extremes{
    std::numeric_limits<f32>::max(),
    -std::numeric_limits<f32>::max()};
  std::array<byte_t, 2> encoded{};
  encode_float_vector_to_storage(
    extremes.data(), 2, VectorDType::uint8, encoded.data());
  assert(static_cast<u8>(encoded[0]) == 255);
  assert(static_cast<u8>(encoded[1]) == 0);
  encode_float_vector_to_storage(
    extremes.data(), 2, VectorDType::int8, encoded.data());
  assert(reinterpret_cast<const i8*>(encoded.data())[0] == 127);
  assert(reinterpret_cast<const i8*>(encoded.data())[1] == -128);

  std::array<byte_t, sizeof(extremes)> raw_extremes{};
  std::memcpy(raw_extremes.data(), extremes.data(), sizeof(extremes));
  const std::array<f32, 2> opposite{
    -std::numeric_limits<f32>::max(),
    std::numeric_limits<f32>::max()};
  std::array<byte_t, sizeof(opposite)> raw_opposite{};
  std::memcpy(raw_opposite.data(), opposite.data(), sizeof(opposite));
  const f32 extreme_pair = typed_l2_distance(
    raw_extremes.data(), VectorDType::float32,
    raw_opposite.data(), VectorDType::float32, 2);
  assert(floating_value_is_finite(extreme_pair));
  assert(extreme_pair == kMaxValidSquaredL2);

  const std::array<byte_t, 2> zero_bytes{};
  const f32 extreme_query = typed_l2_distance_float_query(
    span<const f32>{extremes.data(), extremes.size()},
    zero_bytes.data(), VectorDType::uint8, 2);
  assert(floating_value_is_finite(extreme_query));
  assert(extreme_query == kMaxValidSquaredL2);

  if constexpr (sizeof(size_t) > sizeof(u32)) {
    const f32 value = 0.0f;
    const span<const f32> oversized{
      &value, static_cast<size_t>(std::numeric_limits<u32>::max()) + 1};
    bool rejected = false;
    try {
      (void)encode_float_vector_to_storage(
        oversized, VectorDType::float32);
    } catch (const std::length_error&) {
      rejected = true;
    }
    assert(rejected);
  }
  test_integer_simd_matches_canonical_distance_for_arbitrary_dimensions();
  return 0;
}
