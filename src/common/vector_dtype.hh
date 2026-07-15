#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
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
  switch (dtype) {
    case VectorDType::float32:
      std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
      return;
    case VectorDType::uint8: {
      auto* out = reinterpret_cast<u8*>(dst);
      for (u32 i = 0; i < dim; ++i) {
        const long rounded = std::lround(src[i]);
        out[i] = static_cast<u8>(std::clamp<long>(rounded, 0, 255));
      }
      return;
    }
    case VectorDType::int8: {
      auto* out = reinterpret_cast<i8*>(dst);
      for (u32 i = 0; i < dim; ++i) {
        const long rounded = std::lround(src[i]);
        out[i] = static_cast<i8>(std::clamp<long>(rounded, -128, 127));
      }
      return;
    }
  }
}

inline vec<byte_t> encode_float_vector_to_storage(const span<const element_t> src, VectorDType dtype) {
  vec<byte_t> out(vector_dtype_bytes(dtype, static_cast<u32>(src.size())));
  encode_float_vector_to_storage(src.data(), static_cast<u32>(src.size()), dtype, out.data());
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

// For byte vectors, every squared component difference is at most 255^2.
// Integer sums through this dimension remain exactly representable in IEEE-754
// float, so integer and decoded-float L2 paths cannot differ by reduction
// rounding. Wider dimensions must keep the established reduction order.
inline constexpr bool integral_byte_l2_sum_exact_in_float(u32 dim) {
  return static_cast<u64>(dim) * 255ull * 255ull <= (1ull << 24) - 1;
}

// =========================================================================
// SIMD-accelerated distance functions for same-dtype vector pairs (AVX2)
// =========================================================================

#ifdef __AVX2__

inline float typed_l2_distance_uint8_simd(const byte_t* lhs, const byte_t* rhs, u32 dim) {
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

inline float typed_l2_distance_int8_simd(const byte_t* lhs, const byte_t* rhs, u32 dim) {
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
  return sum;
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
  return sum;
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
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float diff = vector_component_as_float(lhs, lhs_dtype, i) -
                       vector_component_as_float(rhs, rhs_dtype, i);
    sum += diff * diff;
  }
  return sum;
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
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float diff = query[i] - vector_component_as_float(stored, stored_dtype, i);
    sum += diff * diff;
  }
  return sum;
}
