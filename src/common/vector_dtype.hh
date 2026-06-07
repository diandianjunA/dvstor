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

inline float typed_ip_distance_uint8_simd(const byte_t* lhs, const byte_t* rhs, u32 dim) {
  const u8* a = reinterpret_cast<const u8*>(lhs);
  const u8* b = reinterpret_cast<const u8*>(rhs);

  __m256i dot0 = _mm256_setzero_si256();
  __m256i dot1 = _mm256_setzero_si256();
  __m256i dot2 = _mm256_setzero_si256();
  __m256i dot3 = _mm256_setzero_si256();

  u32 i = 0;

  for (; i + 128 <= dim; i += 128) {
    // k = 0
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot0 = _mm256_add_epi32(dot0, prod_lo);
      dot0 = _mm256_add_epi32(dot0, prod_hi); }

    // k = 1
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot1 = _mm256_add_epi32(dot1, prod_lo);
      dot1 = _mm256_add_epi32(dot1, prod_hi); }

    // k = 2
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot2 = _mm256_add_epi32(dot2, prod_lo);
      dot2 = _mm256_add_epi32(dot2, prod_hi); }

    // k = 3
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
      __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot3 = _mm256_add_epi32(dot3, prod_lo);
      dot3 = _mm256_add_epi32(dot3, prod_hi); }
  }

  for (; i + 32 <= dim; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

    __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    dot0 = _mm256_add_epi32(dot0, _mm256_madd_epi16(va_lo, vb_lo));

    __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
    dot0 = _mm256_add_epi32(dot0, _mm256_madd_epi16(va_hi, vb_hi));
  }

  dot0 = _mm256_add_epi32(dot0, dot1);
  dot2 = _mm256_add_epi32(dot2, dot3);
  dot0 = _mm256_add_epi32(dot0, dot2);

  __m128i lo = _mm256_castsi256_si128(dot0);
  __m128i hi = _mm256_extracti128_si256(dot0, 1);
  __m128i combined = _mm_add_epi32(lo, hi);
  combined = _mm_hadd_epi32(combined, combined);
  combined = _mm_hadd_epi32(combined, combined);
  float dot = static_cast<float>(_mm_cvtsi128_si32(combined));

  for (; i < dim; ++i) {
    dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }

  return 1.0f - dot;
}

inline float typed_ip_distance_int8_simd(const byte_t* lhs, const byte_t* rhs, u32 dim) {
  const i8* a = reinterpret_cast<const i8*>(lhs);
  const i8* b = reinterpret_cast<const i8*>(rhs);

  __m256i dot0 = _mm256_setzero_si256();
  __m256i dot1 = _mm256_setzero_si256();
  __m256i dot2 = _mm256_setzero_si256();
  __m256i dot3 = _mm256_setzero_si256();

  u32 i = 0;

  for (; i + 128 <= dim; i += 128) {
    // k = 0
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot0 = _mm256_add_epi32(dot0, prod_lo);
      dot0 = _mm256_add_epi32(dot0, prod_hi); }

    // k = 1
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot1 = _mm256_add_epi32(dot1, prod_lo);
      dot1 = _mm256_add_epi32(dot1, prod_hi); }

    // k = 2
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot2 = _mm256_add_epi32(dot2, prod_lo);
      dot2 = _mm256_add_epi32(dot2, prod_hi); }

    // k = 3
    { __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
      __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
      __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
      __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
      __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
      __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
      __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
      dot3 = _mm256_add_epi32(dot3, prod_lo);
      dot3 = _mm256_add_epi32(dot3, prod_hi); }
  }

  for (; i + 32 <= dim; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

    __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
    __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
    dot0 = _mm256_add_epi32(dot0, _mm256_madd_epi16(va_lo, vb_lo));

    __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
    dot0 = _mm256_add_epi32(dot0, _mm256_madd_epi16(va_hi, vb_hi));
  }

  dot0 = _mm256_add_epi32(dot0, dot1);
  dot2 = _mm256_add_epi32(dot2, dot3);
  dot0 = _mm256_add_epi32(dot0, dot2);

  __m128i lo = _mm256_castsi256_si128(dot0);
  __m128i hi = _mm256_extracti128_si256(dot0, 1);
  __m128i combined = _mm_add_epi32(lo, hi);
  combined = _mm_hadd_epi32(combined, combined);
  combined = _mm_hadd_epi32(combined, combined);
  float dot = static_cast<float>(_mm_cvtsi128_si32(combined));

  for (; i < dim; ++i) {
    dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }

  return 1.0f - dot;
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

inline float typed_ip_distance(const byte_t* lhs,
                               VectorDType lhs_dtype,
                               const byte_t* rhs,
                               VectorDType rhs_dtype,
                               u32 dim) {
#ifdef __AVX2__
  if (lhs_dtype == rhs_dtype) {
    if (lhs_dtype == VectorDType::uint8) {
      return typed_ip_distance_uint8_simd(lhs, rhs, dim);
    }
    if (lhs_dtype == VectorDType::int8) {
      return typed_ip_distance_int8_simd(lhs, rhs, dim);
    }
  }
#endif
  float dot = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    dot += vector_component_as_float(lhs, lhs_dtype, i) *
           vector_component_as_float(rhs, rhs_dtype, i);
  }
  return 1.0f - dot;
}
inline float typed_l2_distance_float_query(const span<const element_t> query,
                                           const byte_t* stored,
                                           VectorDType stored_dtype,
                                           u32 dim) {
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float diff = query[i] - vector_component_as_float(stored, stored_dtype, i);
    sum += diff * diff;
  }
  return sum;
}

inline float typed_ip_distance_float_query(const span<const element_t> query,
                                           const byte_t* stored,
                                           VectorDType stored_dtype,
                                           u32 dim) {
  float dot = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    dot += query[i] * vector_component_as_float(stored, stored_dtype, i);
  }
  return 1.0f - dot;
}

inline float typed_distance_float_query(const span<const element_t> query,
                                        const byte_t* stored,
                                        VectorDType stored_dtype,
                                        u32 dim,
                                        bool ip_distance) {
  return ip_distance ? typed_ip_distance_float_query(query, stored, stored_dtype, dim)
                     : typed_l2_distance_float_query(query, stored, stored_dtype, dim);
}
