#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

enum class GpuRabitqSource {
  automatic,
  sidecar,
  nodes,
};

struct GpuSidecarConversionOptions {
  filepath_t index_prefix;
  u32 entry_points{256};
  u32 threads{};
  u64 seed{1234};
  GpuRabitqSource rabitq_source{GpuRabitqSource::automatic};
  bool manifest_only{};
  bool overwrite{};
};

struct GpuSidecarConversionResult {
  filepath_t index_file;
  vec<filepath_t> code_files;
  vec<u64> code_remote_offsets;
  vec<u64> code_bytes;
  u64 node_count{};
  u32 entry_point_count{};
  bool used_rabitq_sidecars{};
};

GpuRabitqSource parse_gpu_rabitq_source(const str& value);
const char* gpu_rabitq_source_name(GpuRabitqSource source);

GpuSidecarConversionResult convert_gpu_sidecars(
  const GpuSidecarConversionOptions& options);

}
