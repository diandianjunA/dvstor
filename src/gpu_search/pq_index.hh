#pragma once

#include <array>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "common/types.hh"

namespace gpu_search::pq {

inline constexpr std::array<char, 8> kModelMagic{'D', 'V', 'P', 'Q', '1', '6', '\0', '\0'};
inline constexpr u32 kModelVersion = 1;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kCentroidsPerSubquantizer = 256;
inline constexpr u32 kBitsPerCode = 8;
inline constexpr u32 kDefaultSubquantizers = 16;

struct ModelHeader {
  std::array<char, 8> magic{kModelMagic};
  u32 version{kModelVersion};
  u32 header_bytes{sizeof(ModelHeader)};
  u32 endian_marker{kEndianMarker};
  u32 dim{};
  u32 subquantizers{};
  u32 bits_per_code{kBitsPerCode};
  u32 subvector_dim{};
  u32 code_bytes{};
  u32 flags{};
  u32 reserved0{};
  u64 rotation_offset{};
  u64 rotation_bytes{};
  u64 centroids_offset{};
  u64 centroids_bytes{};
  u64 file_bytes{};
  u64 payload_checksum{};
  std::array<u64, 4> reserved{};
};

inline constexpr u32 kFlagHasRotation = 1u << 0;

struct Model {
  u32 dim{};
  u32 subquantizers{kDefaultSubquantizers};
  u32 bits_per_code{kBitsPerCode};
  std::vector<f32> rotation;
  std::vector<f32> centroids;

  u32 subvector_dim() const {
    return subquantizers == 0 ? 0 : dim / subquantizers;
  }

  u32 code_bytes() const { return subquantizers; }
  u64 checksum() const;
  bool has_rotation() const { return !rotation.empty(); }
};

bool validate(const Model& model, std::string* error = nullptr);
bool write_model(const std::filesystem::path& path, const Model& model,
                 std::string* error = nullptr);
bool read_model(const std::filesystem::path& path, Model& model,
                std::string* error = nullptr);

void transform(const Model& model, std::span<const f32> input,
               std::span<f32> output);
void encode(const Model& model, std::span<const f32> input,
            std::span<u8> code, std::span<f32> transformed_scratch);
void build_distance_table(const Model& model, std::span<const f32> input,
                          std::span<f32> table,
                          std::span<f32> transformed_scratch);
f32 asymmetric_distance(const Model& model, std::span<const f32> table,
                        std::span<const u8> code);

}  // namespace gpu_search::pq
