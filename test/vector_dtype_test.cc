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

void test_integral_alpha_threshold_matches_complete_distance() {
  constexpr u32 dim = 128;
  vec<byte_t> lhs(dim);
  vec<byte_t> rhs(dim);

  const auto verify = [&](VectorDType dtype,
                          f64 alpha,
                          distance_t source_distance) {
    const distance_t complete = typed_l2_distance(
      lhs.data(), dtype, rhs.data(), dtype, dim);
    const distance_t scalar = scalar_same_dtype_distance(
      lhs, rhs, dtype, dim);
    assert(complete == scalar);
    const bool expected =
      alpha * static_cast<f64>(scalar) <=
      static_cast<f64>(source_distance);
    const bool actual = typed_l2_distance_alpha_leq_source(
      lhs.data(), dtype, rhs.data(), dtype, dim, alpha, source_distance);
    assert(actual == expected);
  };

  for (const VectorDType dtype :
       {VectorDType::uint8, VectorDType::int8}) {
    u32 state = dtype == VectorDType::uint8 ? 0x12345678u : 0x87654321u;
    for (u32 sample = 0; sample < 512; ++sample) {
      for (u32 index = 0; index < dim; ++index) {
        state = state * 1664525u + 1013904223u;
        lhs[index] = static_cast<byte_t>(state >> 24);
        state = state * 1664525u + 1013904223u;
        rhs[index] = static_cast<byte_t>(state >> 24);
      }
      const distance_t complete = typed_l2_distance(
        lhs.data(), dtype, rhs.data(), dtype, dim);
      verify(dtype, 1.0, complete);
      verify(dtype, 1.0,
             std::nextafter(complete,
                            -std::numeric_limits<distance_t>::infinity()));
      verify(dtype, 1.0,
             std::nextafter(complete,
                            std::numeric_limits<distance_t>::infinity()));
      verify(dtype, 1.2, static_cast<distance_t>(
        1.2 * static_cast<f64>(complete)));
      verify(dtype, 0.73, static_cast<distance_t>(
        0.73 * static_cast<f64>(complete)));
      verify(dtype, 2.5, static_cast<distance_t>(state & 0x00ffffffu));
    }
  }
}

void test_integral_alpha_threshold_exits_early_and_fallbacks_are_unchanged() {
  constexpr u32 dim = 128;
  vec<byte_t> lhs(dim, static_cast<byte_t>(0));
  vec<byte_t> rhs(dim, static_cast<byte_t>(255));

  for (const VectorDType dtype :
       {VectorDType::uint8, VectorDType::int8}) {
    if (dtype == VectorDType::int8) {
      std::fill(reinterpret_cast<i8*>(lhs.data()),
                reinterpret_cast<i8*>(lhs.data()) + dim,
                static_cast<i8>(-128));
      std::fill(reinterpret_cast<i8*>(rhs.data()),
                reinterpret_cast<i8*>(rhs.data()) + dim,
                static_cast<i8>(127));
    }
    u32 evaluated = 0;
    assert(!typed_l2_distance_alpha_leq_source(
      lhs.data(), dtype, rhs.data(), dtype, dim, 1.2, 1.0F,
      &evaluated));
    assert(evaluated == 32);
  }

  std::array<f32, 4> float_lhs{0.0F, 1.0F, 2.0F, 3.0F};
  std::array<f32, 4> float_rhs{3.0F, 2.0F, 1.0F, 0.0F};
  u32 evaluated = 0;
  const bool expected = 1.2 * static_cast<f64>(typed_l2_distance(
    reinterpret_cast<const byte_t*>(float_lhs.data()),
    VectorDType::float32,
    reinterpret_cast<const byte_t*>(float_rhs.data()),
    VectorDType::float32,
    static_cast<u32>(float_lhs.size()))) <= 10.0;
  const bool actual = typed_l2_distance_alpha_leq_source(
    reinterpret_cast<const byte_t*>(float_lhs.data()),
    VectorDType::float32,
    reinterpret_cast<const byte_t*>(float_rhs.data()),
    VectorDType::float32,
    static_cast<u32>(float_lhs.size()), 1.2, 10.0F, &evaluated);
  assert(actual == expected);
  assert(evaluated == float_lhs.size());

  vec<byte_t> mixed_lhs(dim, static_cast<byte_t>(200));
  vec<byte_t> mixed_rhs(dim, static_cast<byte_t>(-50));
  const distance_t mixed_distance = typed_l2_distance(
    mixed_lhs.data(), VectorDType::uint8,
    mixed_rhs.data(), VectorDType::int8, dim);
  evaluated = 0;
  const distance_t mixed_source = std::nextafter(
    mixed_distance, std::numeric_limits<distance_t>::infinity());
  const bool mixed_expected =
    1.0 * static_cast<f64>(mixed_distance) <= mixed_source;
  const bool mixed_actual = typed_l2_distance_alpha_leq_source(
    mixed_lhs.data(), VectorDType::uint8,
    mixed_rhs.data(), VectorDType::int8,
    dim, 1.0, mixed_source, &evaluated);
  assert(mixed_actual == mixed_expected);
  assert(evaluated == dim);
}

}  // namespace

void test_unaligned_float_component_load() {
  alignas(float) std::array<byte_t, sizeof(float) + 1> storage{};
  const float expected = -123.5f;
  std::memcpy(storage.data() + 1, &expected, sizeof(expected));
  const float actual = vector_component_as_float(
    storage.data() + 1, VectorDType::float32, 0);
  assert(actual == expected);
}

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
  test_unaligned_float_component_load();
  test_integer_simd_matches_canonical_distance_for_arbitrary_dimensions();
  test_integral_alpha_threshold_matches_complete_distance();
  test_integral_alpha_threshold_exits_early_and_fallbacks_are_unchanged();
  return 0;
}
