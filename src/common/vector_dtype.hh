#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>

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

inline float typed_l2_distance(const byte_t* lhs,
                               VectorDType lhs_dtype,
                               const byte_t* rhs,
                               VectorDType rhs_dtype,
                               u32 dim) {
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
