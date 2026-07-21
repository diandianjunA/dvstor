#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "common/types.hh"

enum class VectorDType : u32 {
  float32 = 0,
  uint8 = 1,
  int8 = 2,
};

inline str vector_dtype_name(VectorDType dtype) {
  switch (dtype) {
    case VectorDType::float32:
      return "float32";
    case VectorDType::uint8:
      return "uint8";
    case VectorDType::int8:
      return "int8";
  }
  return "unknown";
}

inline VectorDType parse_vector_dtype(const str& name) {
  if (name == "float32" || name == "float" || name == "f32") {
    return VectorDType::float32;
  }
  if (name == "uint8" || name == "u8") {
    return VectorDType::uint8;
  }
  if (name == "int8" || name == "i8") {
    return VectorDType::int8;
  }
  throw std::invalid_argument("unknown vector dtype: " + name);
}

inline std::optional<VectorDType> infer_vector_dtype_from_path(const filepath_t& path) {
  const str ext = path.extension().string();
  if (ext == ".u8bin") {
    return VectorDType::uint8;
  }
  if (ext == ".i8bin") {
    return VectorDType::int8;
  }
  if (ext == ".fbin" || ext == ".bin") {
    return VectorDType::float32;
  }
  return std::nullopt;
}

inline VectorDType resolve_vector_dtype_config(const str& value, const filepath_t& path) {
  if (value.empty() || value == "auto") {
    return infer_vector_dtype_from_path(path).value_or(VectorDType::float32);
  }
  return parse_vector_dtype(value);
}

inline size_t vector_dtype_component_size(VectorDType dtype) {
  switch (dtype) {
    case VectorDType::float32:
      return sizeof(float);
    case VectorDType::uint8:
      return sizeof(u8);
    case VectorDType::int8:
      return sizeof(i8);
  }
  return sizeof(float);
}

inline size_t vector_dtype_bytes(VectorDType dtype, u32 dim) {
  return static_cast<size_t>(dim) * vector_dtype_component_size(dtype);
}

// Inspect IEEE-754 exponent bits at external-data boundaries. This is cheap
// and keeps validation stable across supported compilers and math-library
// implementations without depending on optimizer treatment of isfinite.
inline bool floating_value_is_finite(f32 value) {
  return (std::bit_cast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

inline bool floating_value_is_finite(f64 value) {
  return (std::bit_cast<u64>(value) & 0x7ff0000000000000ull) !=
    0x7ff0000000000000ull;
}

inline bool vector_component_is_finite(float value) {
  return floating_value_is_finite(value);
}

// FLT_MAX remains the internal invalid/uninitialized sentinel in the GPU
// beam.  Saturate legal squared-L2 results to the immediately preceding float
// so even finite extreme inputs remain sortable and cannot be mistaken for an
// RDMA or record-validation failure.
inline constexpr f32 kMaxValidSquaredL2 = 0x1.fffffcp+127f;

inline f32 saturate_squared_l2(f64 value) {
  if (!(value < static_cast<f64>(kMaxValidSquaredL2))) {
    return kMaxValidSquaredL2;
  }
  return value <= 0.0 ? 0.0f : static_cast<f32>(value);
}

inline float vector_component_as_float(const byte_t* data, VectorDType dtype, size_t index) {
  switch (dtype) {
    case VectorDType::float32:
      return reinterpret_cast<const float*>(data)[index];
    case VectorDType::uint8:
      return static_cast<float>(reinterpret_cast<const u8*>(data)[index]);
    case VectorDType::int8:
      return static_cast<float>(reinterpret_cast<const i8*>(data)[index]);
  }
  return 0.0f;
}

inline void encode_float_vector_to_storage(const float* src, u32 dim, VectorDType dtype, byte_t* dst) {
  if (src == nullptr || dst == nullptr) {
    throw std::invalid_argument("vector encoding requires non-null storage");
  }
  switch (dtype) {
    case VectorDType::float32:
      for (u32 i = 0; i < dim; ++i) {
        if (!vector_component_is_finite(src[i])) {
          throw std::invalid_argument("vector components must be finite");
        }
      }
      std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
      return;
    case VectorDType::uint8: {
      auto* out = reinterpret_cast<u8*>(dst);
      for (u32 i = 0; i < dim; ++i) {
        if (!vector_component_is_finite(src[i])) {
          throw std::invalid_argument("vector components must be finite");
        }
        // Clamp in float before lround. A finite float can still exceed the
        // range of long, for which lround has a domain error.
        const float bounded = std::clamp(src[i], 0.0f, 255.0f);
        out[i] = static_cast<u8>(std::lround(bounded));
      }
      return;
    }
    case VectorDType::int8: {
      auto* out = reinterpret_cast<i8*>(dst);
      for (u32 i = 0; i < dim; ++i) {
        if (!vector_component_is_finite(src[i])) {
          throw std::invalid_argument("vector components must be finite");
        }
        const float bounded = std::clamp(src[i], -128.0f, 127.0f);
        out[i] = static_cast<i8>(std::lround(bounded));
      }
      return;
    }
  }
  throw std::invalid_argument("unknown vector dtype");
}

inline vec<byte_t> encode_float_vector_to_storage(const span<const element_t> src, VectorDType dtype) {
  if (src.size() > std::numeric_limits<u32>::max()) {
    throw std::length_error("vector dimension exceeds the storage layout limit");
  }
  const u32 dim = static_cast<u32>(src.size());
  vec<byte_t> out(vector_dtype_bytes(dtype, dim));
  encode_float_vector_to_storage(src.data(), dim, dtype, out.data());
  return out;
}

inline void decode_storage_vector_to_float(const byte_t* src, VectorDType dtype, u32 dim, float* dst) {
  for (u32 i = 0; i < dim; ++i) {
    dst[i] = vector_component_as_float(src, dtype, i);
  }
}

inline vec<float> decode_storage_vector_to_float(const byte_t* src, VectorDType dtype, u32 dim) {
  vec<float> out(dim);
  decode_storage_vector_to_float(src, dtype, dim, out.data());
  return out;
}

inline float typed_l2_distance_float_query_scalar(
    const span<const element_t> query, const byte_t* stored,
    VectorDType stored_dtype, u32 dim) {
  f64 sum = 0.0;
  for (u32 i = 0; i < dim; ++i) {
    const f64 diff = static_cast<f64>(query[i]) -
      static_cast<f64>(vector_component_as_float(stored, stored_dtype, i));
    sum += diff * diff;
  }
  return saturate_squared_l2(sum);
}

// =========================================================================
// SIMD-accelerated distance functions for same-dtype vector pairs (AVX2)
// =========================================================================

#ifdef __AVX2__

inline float typed_l2_distance_uint8_simd_chunk(
    const byte_t* lhs, const byte_t* rhs, u32 dim) {
  const u8* a = reinterpret_cast<const u8*>(lhs);
  const u8* b = reinterpret_cast<const u8*>(rhs);

  __m256i sum0 = _mm256_setzero_si256();
  __m256i sum1 = _mm256_setzero_si256();
  __m256i sum2 = _mm256_setzero_si256();
  __m256i sum3 = _mm256_setzero_si256();

  u32 i = 0;

  for (; i + 128 <= dim; i += 128) {
    // Unrolled x4: each iteration directly targets its own accumulator
    // to avoid runtime branch on k inside an already hot loop.

    // k = 0
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_lo, diff_lo));
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 1
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff_lo, diff_lo));
      sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 2
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff_lo, diff_lo));
      sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 3
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(diff_lo, diff_lo));
      sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(diff_hi, diff_hi)); }
  }

  for (; i + 32 <= dim; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

    __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);

    __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
    __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

    __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
    __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

    sum0 = _mm256_add_epi32(sum0, sq_lo);
    sum0 = _mm256_add_epi32(sum0, sq_hi);
  }

  sum0 = _mm256_add_epi32(sum0, sum1);
  sum2 = _mm256_add_epi32(sum2, sum3);
  sum0 = _mm256_add_epi32(sum0, sum2);

  __m128i lo = _mm256_castsi256_si128(sum0);
  __m128i hi = _mm256_extracti128_si256(sum0, 1);
  __m128i combined = _mm_add_epi32(lo, hi);
  combined = _mm_hadd_epi32(combined, combined);
  combined = _mm_hadd_epi32(combined, combined);
  float result = static_cast<float>(_mm_cvtsi128_si32(combined));

  for (; i < dim; ++i) {
    const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    result += diff * diff;
  }

  return result;
}

inline float typed_l2_distance_int8_simd_chunk(
    const byte_t* lhs, const byte_t* rhs, u32 dim) {
  const i8* a = reinterpret_cast<const i8*>(lhs);
  const i8* b = reinterpret_cast<const i8*>(rhs);

  __m256i sum0 = _mm256_setzero_si256();
  __m256i sum1 = _mm256_setzero_si256();
  __m256i sum2 = _mm256_setzero_si256();
  __m256i sum3 = _mm256_setzero_si256();

  u32 i = 0;

  for (; i + 128 <= dim; i += 128) {
    // k = 0
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_lo, diff_lo));
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 1
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff_lo, diff_lo));
      sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 2
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff_lo, diff_lo));
      sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff_hi, diff_hi)); }

    // k = 3
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
      sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(diff_lo, diff_lo));
      sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(diff_hi, diff_hi)); }
  }

  for (; i + 32 <= dim; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

    __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
    __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
    __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);

    __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
    __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

    __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
    __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

    sum0 = _mm256_add_epi32(sum0, sq_lo);
    sum0 = _mm256_add_epi32(sum0, sq_hi);
  }

  sum0 = _mm256_add_epi32(sum0, sum1);
  sum2 = _mm256_add_epi32(sum2, sum3);
  sum0 = _mm256_add_epi32(sum0, sum2);

  __m128i lo = _mm256_castsi256_si128(sum0);
  __m128i hi = _mm256_extracti128_si256(sum0, 1);
  __m128i combined = _mm_add_epi32(lo, hi);
  combined = _mm_hadd_epi32(combined, combined);
  combined = _mm_hadd_epi32(combined, combined);
  float result = static_cast<float>(_mm_cvtsi128_si32(combined));

  for (; i < dim; ++i) {
    const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    result += diff * diff;
  }

  return result;
}

// Keep every signed epi32 reduction and its float conversion exact.
// 256*255^2=16,646,400 is below both INT32_MAX and float's exact-integer
// range. Wider vectors retain SIMD throughput, accumulate exact chunk totals
// in FP64, and round only once at the public float result.
inline constexpr u32 kIntegralByteSimdExactChunk = 256;

inline float typed_l2_distance_uint8_simd(
    const byte_t* lhs, const byte_t* rhs, u32 dim) {
  f64 total = 0.0;
  for (u32 offset = 0; offset < dim;) {
    const u32 chunk = std::min<u32>(
      kIntegralByteSimdExactChunk, dim - offset);
    total += typed_l2_distance_uint8_simd_chunk(
      lhs + offset, rhs + offset, chunk);
    offset += chunk;
  }
  return static_cast<float>(total);
}

inline float typed_l2_distance_int8_simd(
    const byte_t* lhs, const byte_t* rhs, u32 dim) {
  f64 total = 0.0;
  for (u32 offset = 0; offset < dim;) {
    const u32 chunk = std::min<u32>(
      kIntegralByteSimdExactChunk, dim - offset);
    total += typed_l2_distance_int8_simd_chunk(
      lhs + offset, rhs + offset, chunk);
    offset += chunk;
  }
  return static_cast<float>(total);
}

inline float horizontal_sum_ps(__m256 value) {
  __m128 lo = _mm256_castps256_ps128(value);
  __m128 hi = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

inline float typed_l2_distance_float_query_uint8_simd(const span<const element_t> query,
                                                      const byte_t* stored,
                                                      u32 dim) {
  const float* q = query.data();
  const u8* s = reinterpret_cast<const u8*>(stored);
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  u32 i = 0;
  for (; i + 32 <= dim; i += 32) {
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 8));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 8);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 16));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 16);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 24));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 24);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(diff, diff)); }
  }

  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  acc0 = _mm256_add_ps(acc0, acc2);
  float sum = horizontal_sum_ps(acc0);

  for (; i < dim; ++i) {
    const float diff = q[i] - static_cast<float>(s[i]);
    sum += diff * diff;
  }
  return floating_value_is_finite(sum)
    ? std::min(sum, kMaxValidSquaredL2)
    : typed_l2_distance_float_query_scalar(
        query, stored, VectorDType::uint8, dim);
}
inline float typed_l2_distance_float_query_int8_simd(const span<const element_t> query,
                                                     const byte_t* stored,
                                                     u32 dim) {
  const float* q = query.data();
  const i8* s = reinterpret_cast<const i8*>(stored);
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  u32 i = 0;
  for (; i + 32 <= dim; i += 32) {
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 8));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 8);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 16));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 16);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(diff, diff)); }
    { __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + 24));
      __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
      __m256 qv = _mm256_loadu_ps(q + i + 24);
      __m256 diff = _mm256_sub_ps(qv, sv);
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(diff, diff)); }
  }

  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  acc0 = _mm256_add_ps(acc0, acc2);
  float sum = horizontal_sum_ps(acc0);

  for (; i < dim; ++i) {
    const float diff = q[i] - static_cast<float>(s[i]);
    sum += diff * diff;
  }
  return floating_value_is_finite(sum)
    ? std::min(sum, kMaxValidSquaredL2)
    : typed_l2_distance_float_query_scalar(
        query, stored, VectorDType::int8, dim);
}

#endif  // __AVX2__

inline float typed_l2_distance(const byte_t* lhs,
                               VectorDType lhs_dtype,
                               const byte_t* rhs,
                               VectorDType rhs_dtype,
                               u32 dim) {
#ifdef __AVX2__
  if (lhs_dtype == rhs_dtype) {
    if (lhs_dtype == VectorDType::uint8) {
      return typed_l2_distance_uint8_simd(lhs, rhs, dim);
    }
    if (lhs_dtype == VectorDType::int8) {
      return typed_l2_distance_int8_simd(lhs, rhs, dim);
    }
  }
#endif
  f64 sum = 0.0;
  for (u32 i = 0; i < dim; ++i) {
    const f64 diff =
      static_cast<f64>(vector_component_as_float(lhs, lhs_dtype, i)) -
      static_cast<f64>(vector_component_as_float(rhs, rhs_dtype, i));
    sum += diff * diff;
  }
  return saturate_squared_l2(sum);
}

// Evaluate the exact predicate used by alpha RobustPrune while avoiding the
// rest of an integral byte-vector distance once the predicate can no longer
// hold.  For equal uint8/int8 dtypes every squared component difference is a
// non-negative integer.  Consequently the exact partial sum and its rounded
// float representation are monotone.  With a finite positive alpha,
//
//   alpha * float(partial_sum) > source_distance
//
// proves that alpha * float(final_sum) <= source_distance is impossible.
// The completed path rounds the exact integer sum once, exactly as
// typed_l2_distance() does.  Float, mixed-dtype, and unusual-alpha cases use
// typed_l2_distance() directly so their established behavior is unchanged.
// evaluated_components is optional test/telemetry output; it has no bearing
// on the result.
inline bool typed_l2_distance_alpha_leq_source(
    const byte_t* lhs,
    VectorDType lhs_dtype,
    const byte_t* rhs,
    VectorDType rhs_dtype,
    u32 dim,
    f64 alpha,
    distance_t source_distance,
    u32* evaluated_components = nullptr) {
  if (evaluated_components != nullptr) {
    *evaluated_components = 0;
  }
  const bool same_integral_dtype =
    lhs_dtype == rhs_dtype &&
    (lhs_dtype == VectorDType::uint8 || lhs_dtype == VectorDType::int8);
  if (!same_integral_dtype || !(alpha > 0.0) ||
      !floating_value_is_finite(alpha)) {
    if (evaluated_components != nullptr) {
      *evaluated_components = dim;
    }
    return alpha * static_cast<f64>(
                     typed_l2_distance(lhs, lhs_dtype, rhs, rhs_dtype, dim)) <=
      static_cast<f64>(source_distance);
  }

  // A 32-component chunk is small enough that its largest possible sum
  // (32 * 255^2) is represented exactly by float.  This preserves exact
  // accumulation while providing four early-exit opportunities for the
  // common 128-dimensional vectors.
  constexpr u32 kThresholdChunkComponents = 32;
  f64 exact_sum = 0.0;
  u32 offset = 0;
  while (offset < dim) {
    const u32 chunk = std::min<u32>(
      kThresholdChunkComponents, dim - offset);
#ifdef __AVX2__
    exact_sum += lhs_dtype == VectorDType::uint8
      ? static_cast<f64>(typed_l2_distance_uint8_simd_chunk(
          lhs + offset, rhs + offset, chunk))
      : static_cast<f64>(typed_l2_distance_int8_simd_chunk(
          lhs + offset, rhs + offset, chunk));
#else
    if (lhs_dtype == VectorDType::uint8) {
      const auto* a = reinterpret_cast<const u8*>(lhs + offset);
      const auto* b = reinterpret_cast<const u8*>(rhs + offset);
      for (u32 index = 0; index < chunk; ++index) {
        const i32 diff = static_cast<i32>(a[index]) -
          static_cast<i32>(b[index]);
        exact_sum += static_cast<f64>(diff * diff);
      }
    } else {
      const auto* a = reinterpret_cast<const i8*>(lhs + offset);
      const auto* b = reinterpret_cast<const i8*>(rhs + offset);
      for (u32 index = 0; index < chunk; ++index) {
        const i32 diff = static_cast<i32>(a[index]) -
          static_cast<i32>(b[index]);
        exact_sum += static_cast<f64>(diff * diff);
      }
    }
#endif
    offset += chunk;
    if (evaluated_components != nullptr) {
      *evaluated_components = offset;
    }

    const distance_t rounded_partial = static_cast<distance_t>(exact_sum);
    if (alpha * static_cast<f64>(rounded_partial) >
        static_cast<f64>(source_distance)) {
      return false;
    }
  }

  const distance_t rounded_distance = static_cast<distance_t>(exact_sum);
  return alpha * static_cast<f64>(rounded_distance) <=
    static_cast<f64>(source_distance);
}

inline float typed_l2_distance_float_query(const span<const element_t> query,
                                           const byte_t* stored,
                                           VectorDType stored_dtype,
                                           u32 dim) {
#ifdef __AVX2__
  if (stored_dtype == VectorDType::uint8) {
    return typed_l2_distance_float_query_uint8_simd(query, stored, dim);
  }
  if (stored_dtype == VectorDType::int8) {
    return typed_l2_distance_float_query_int8_simd(query, stored, dim);
  }
#endif
  return typed_l2_distance_float_query_scalar(
    query, stored, stored_dtype, dim);
}
