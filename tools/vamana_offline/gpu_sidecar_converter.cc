#include "tools/vamana_offline/gpu_sidecar_converter.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <unistd.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "tools/vamana_offline/progress.hh"
#include "vamana/anchor_index.hh"
#include "vamana/hot_graph.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/storage_format.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

constexpr u64 kShardHeaderBytes = 16;
constexpr size_t kIoChunkBytes = 64ull << 20;

struct Layout {
  u32 dim{};
  u32 degree{};
  VectorDType dtype{VectorDType::float32};
  u64 node_bytes{};
  u64 vector_bytes{};
  u64 rabitq_offset{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_bytes{};
  u32 graph_entry_bytes{};
  u32 graph_shard_bits{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
};

struct ShardInfo {
  filepath_t path;
  u64 file_bytes{};
  u64 free_pointer{};
  u64 medoid_raw{};
  u64 node_count{};
  u64 graph_offset{};
  u64 dynamic_base_offset{};
};

struct EntrySample {
  u64 priority{};
  u32 ordinal{};
  bool operator<(const EntrySample& other) const { return priority < other.priority; }
};

struct TemporaryPath {
  filepath_t final_path;
  filepath_t temporary_path;
  bool published{};

  TemporaryPath() = default;
  TemporaryPath(const TemporaryPath&) = delete;
  TemporaryPath& operator=(const TemporaryPath&) = delete;
  TemporaryPath(TemporaryPath&& other) noexcept
      : final_path(std::move(other.final_path)),
        temporary_path(std::move(other.temporary_path)),
        published(other.published) {
    other.published = true;
  }
  TemporaryPath& operator=(TemporaryPath&& other) noexcept {
    if (this == &other) return *this;
    if (!published && !temporary_path.empty()) {
      std::error_code error;
      std::filesystem::remove(temporary_path, error);
    }
    final_path = std::move(other.final_path);
    temporary_path = std::move(other.temporary_path);
    published = other.published;
    other.published = true;
    return *this;
  }
  ~TemporaryPath() {
    if (!published && !temporary_path.empty()) {
      std::error_code error;
      std::filesystem::remove(temporary_path, error);
    }
  }
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void read_exact(std::istream& input, void* destination, size_t bytes,
                const filepath_t& path);

bool append_anchor_entry_points(
    const filepath_t& prefix, const Layout& layout, u32 shard_count,
    const vec<u64>& counts, const vec<u64>& ordinal_bases, u32 target,
    std::unordered_set<u32>& selected, vec<u32>& entry_points) {
  const filepath_t path = index_path::anchor_file(prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) return false;
  vamana::anchor::Header header;
  read_exact(input, &header, sizeof(header), path);
  if (header.magic != vamana::anchor::kMagic ||
      header.version != vamana::anchor::kVersion || header.dim != layout.dim ||
      header.shard_count != shard_count ||
      header.vector_dtype != static_cast<u32>(layout.dtype) ||
      header.vector_bytes != layout.vector_bytes ||
      header.total_anchors > (1u << 24)) {
    throw std::runtime_error("invalid anchor sidecar for GPU V4 entry points: " +
                             path.string());
  }
  vec<vec<u32>> anchor_ordinals(shard_count);
  vec<byte_t> vector(header.vector_bytes);
  u64 loaded = 0;
  for (u32 shard = 0; shard < shard_count; ++shard) {
    vamana::anchor::ShardHeader shard_header;
    read_exact(input, &shard_header, sizeof(shard_header), path);
    if (shard_header.shard != shard ||
        shard_header.anchor_count > header.anchors_per_shard ||
        loaded + shard_header.anchor_count > header.total_anchors) {
      throw std::runtime_error("invalid anchor shard for GPU V4 entry points: " +
                               path.string());
    }
    vec<f32> shard_centroid(layout.dim);
    read_exact(input, shard_centroid.data(),
               shard_centroid.size() * sizeof(f32), path);
    anchor_ordinals[shard].reserve(shard_header.anchor_count);
    for (u32 index = 0; index < shard_header.anchor_count; ++index) {
      vamana::anchor::EntryHeader entry;
      read_exact(input, &entry, sizeof(entry), path);
      read_exact(input, vector.data(), vector.size(), path);
      const RemotePtr pointer{entry.rptr_raw};
      if (pointer.is_null() || pointer.memory_node() != shard ||
          pointer.byte_offset() < kShardHeaderBytes) {
        throw std::runtime_error("anchor points outside its static GPU V4 shard");
      }
      const u64 relative = pointer.byte_offset() - kShardHeaderBytes;
      if (relative % layout.node_bytes != 0 ||
          relative / layout.node_bytes >= counts[shard]) {
        throw std::runtime_error("anchor points outside its static GPU V4 range");
      }
      anchor_ordinals[shard].push_back(static_cast<u32>(
        ordinal_bases[shard] + relative / layout.node_bytes));
      ++loaded;
    }
  }
  if (loaded != header.total_anchors) {
    throw std::runtime_error("anchor sidecar count mismatch for GPU V4 entry points");
  }
  bool appended = false;
  for (u32 rank = 0; entry_points.size() < target; ++rank) {
    bool have_rank = false;
    for (u32 shard = 0; shard < shard_count && entry_points.size() < target; ++shard) {
      if (rank >= anchor_ordinals[shard].size()) continue;
      have_rank = true;
      const u32 ordinal = anchor_ordinals[shard][rank];
      if (selected.insert(ordinal).second) {
        entry_points.push_back(ordinal);
        appended = true;
      }
    }
    if (!have_rank) break;
  }
  return appended;
}

void read_exact(std::istream& input, void* destination, size_t bytes,
                const filepath_t& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void write_exact(std::ostream& output, const void* source, size_t bytes,
                 const filepath_t& path) {
  output.write(reinterpret_cast<const char*>(source), static_cast<std::streamsize>(bytes));
  if (!output.good()) throw std::runtime_error("failed to write " + path.string());
}

u64 read_u64(const byte_t* source) {
  u64 value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

u32 read_u32(const byte_t* source) {
  u32 value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

TemporaryPath make_temporary_path(const filepath_t& final_path) {
  TemporaryPath result;
  result.final_path = final_path;
  result.temporary_path = filepath_t(
    final_path.string() + ".tmp." + std::to_string(static_cast<unsigned long>(::getpid())));
  std::error_code error;
  std::filesystem::remove(result.temporary_path, error);
  return result;
}

void publish(TemporaryPath& path) {
  std::error_code error;
  std::filesystem::rename(path.temporary_path, path.final_path, error);
  if (error) {
    throw std::runtime_error(
      "failed to publish " + path.final_path.string() + ": " + error.message());
  }
  path.published = true;
}

Layout configure_layout(const nlohmann::json& metadata) {
  if (metadata.value("schema_version", 0u) != 13) {
    throw std::runtime_error("V4 converter requires schema_version 13");
  }
  if (metadata.value("distance", str{"l2"}) != "l2") {
    throw std::runtime_error("GPU V4 currently supports L2 indexes only");
  }
  if (metadata.value("node_layout", str{"standard"}) != "rabitq") {
    throw std::runtime_error("GPU V4 requires full RaBitQ entries in the storage index");
  }
  const auto storage_format = vamana::parse_storage_format(
    metadata.value("storage_format", str{}));
  if (!storage_format || *storage_format != vamana::StorageFormat::compact_v1) {
    throw std::runtime_error(
      "GPU V4 requires vamana_compact_v1 so graph records can be fetched directly");
  }

  Layout layout;
  layout.dim = metadata.at("dim").get<u32>();
  layout.degree = metadata.at("R").get<u32>();
  layout.dtype = parse_vector_dtype(metadata.value("vector_data_type", str{"float32"}));
  VamanaNode::disable_hot_graph();
  VamanaNode::disable_rabitq();
  VamanaNode::set_storage_format(*storage_format);
  VamanaNode::init_static_storage(layout.dim, layout.degree, layout.dtype);
  VamanaNode::enable_rabitq();
  layout.node_bytes = VamanaNode::total_size();
  layout.vector_bytes = VamanaNode::vector_bytes();
  layout.rabitq_offset = VamanaNode::offset_rabitq_code();
  layout.rabitq_code_bits = VamanaNode::rabitq_code_bits();
  layout.rabitq_entry_bytes = static_cast<u32>(VamanaNode::rabitq_entry_size());
  layout.graph_entry_bytes = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  layout.graph_shard_bits = vamana::hot_graph::shard_bits_for(
    metadata.at("num_memory_nodes").get<u32>());
  layout.dynamic_record_bytes = metadata.value(
    "hot_graph_dynamic_record_bytes", static_cast<u32>(VamanaNode::dynamic_record_size()));
  layout.dynamic_hot_offset = metadata.value(
    "hot_graph_dynamic_hot_offset", static_cast<u32>(VamanaNode::total_size()));

  const auto require_equal = [&](const char* name, u64 expected) {
    const u64 actual = metadata.value(name, std::numeric_limits<u64>::max());
    if (actual != expected) {
      throw std::runtime_error(str{"metadata layout mismatch for "} + name);
    }
  };
  require_equal("node_size", layout.node_bytes);
  require_equal("vector_bytes", layout.vector_bytes);
  require_equal("rabitq_offset", layout.rabitq_offset);
  require_equal("rabitq_code_bits", layout.rabitq_code_bits);
  require_equal("rabitq_entry_size", layout.rabitq_entry_bytes);
  require_equal("hot_graph_entry_size", layout.graph_entry_bytes);
  require_equal("hot_graph_pointer_bytes", vamana::hot_graph::kCompactPointerBytes);
  require_equal("hot_graph_shard_bits", layout.graph_shard_bits);
  if (layout.graph_entry_bytes > gpu_search::format::kGraphCacheLineBytes) {
    throw std::runtime_error("compact graph entry exceeds the GPU V4 512-byte cache line");
  }
  return layout;
}

vec<ShardInfo> inspect_shards(const filepath_t& prefix,
                              const nlohmann::json& metadata,
                              const Layout& layout) {
  const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
  const vec<u64> counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  const vec<u64> graph_offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
  const vec<u64> header_offsets = metadata.at("hot_graph_header_offsets").get<vec<u64>>();
  const vec<u64> dynamic_offsets =
    metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
  if (counts.size() != shard_count || graph_offsets.size() != shard_count ||
      header_offsets.size() != shard_count || dynamic_offsets.size() != shard_count) {
    throw std::runtime_error("compact graph metadata has an invalid shard count");
  }

  vec<ShardInfo> shards(shard_count);
  u64 total_nodes = 0;
  for (u32 shard_id = 0; shard_id < shard_count; ++shard_id) {
    ShardInfo& shard = shards[shard_id];
    shard.path = index_path::shard_file(prefix, shard_id + 1, shard_count);
    std::ifstream input(shard.path, std::ios::binary);
    if (!input.good()) throw std::runtime_error("missing storage shard: " + shard.path.string());
    std::array<byte_t, kShardHeaderBytes> fixed_header{};
    read_exact(input, fixed_header.data(), fixed_header.size(), shard.path);
    shard.free_pointer = read_u64(fixed_header.data());
    shard.medoid_raw = read_u64(fixed_header.data() + sizeof(u64));
    shard.file_bytes = std::filesystem::file_size(shard.path);
    shard.node_count = counts[shard_id];
    shard.graph_offset = graph_offsets[shard_id];
    shard.dynamic_base_offset = dynamic_offsets[shard_id];
    if (shard.node_count == 0 || shard.free_pointer != shard.dynamic_base_offset ||
        kShardHeaderBytes + shard.node_count * layout.node_bytes > shard.file_bytes ||
        shard.graph_offset + shard.node_count * layout.graph_entry_bytes > shard.file_bytes) {
      throw std::runtime_error(
        "storage shard is truncated or contains persisted dynamic records; compact it first: " +
        shard.path.string());
    }
    vamana::hot_graph::Header graph_header;
    input.seekg(static_cast<std::streamoff>(header_offsets[shard_id]));
    read_exact(input, &graph_header, sizeof(graph_header), shard.path);
    if (graph_header.magic != vamana::hot_graph::kMagic ||
        graph_header.version != vamana::hot_graph::kVersion2 ||
        graph_header.entry_bytes != layout.graph_entry_bytes ||
        graph_header.max_degree != layout.degree ||
        graph_header.compact_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
        graph_header.compact_pointer_shard_bits != layout.graph_shard_bits ||
        graph_header.entry_count != shard.node_count) {
      throw std::runtime_error("invalid compact graph header: " + shard.path.string());
    }
    total_nodes += shard.node_count;
  }
  if (total_nodes != metadata.at("num_vectors").get<u64>() ||
      total_nodes == 0 || total_nodes >= (1ull << 30)) {
    throw std::runtime_error("GPU V4 requires 1..2^30-1 static nodes");
  }
  return shards;
}

bool static_ordinal(RemotePtr pointer, const vec<ShardInfo>& shards,
                    const Layout& layout, const vec<u64>& ordinal_bases,
                    u32& ordinal) {
  if (pointer.is_null() || pointer.memory_node() >= shards.size() ||
      pointer.byte_offset() < kShardHeaderBytes) {
    return false;
  }
  const u64 relative = pointer.byte_offset() - kShardHeaderBytes;
  if (relative % layout.node_bytes != 0) return false;
  const u64 slot = relative / layout.node_bytes;
  if (slot >= shards[pointer.memory_node()].node_count) return false;
  ordinal = static_cast<u32>(ordinal_bases[pointer.memory_node()] + slot);
  return true;
}

bool full_sidecars_available(const filepath_t& prefix,
                             const vec<ShardInfo>& shards,
                             const Layout& layout) {
  const u32 code_bytes = layout.rabitq_code_bits / 8u;
  for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
    const filepath_t sidecar_path = index_path::rabitq_cache_file(
      prefix, shard_id + 1, shards.size());
    std::ifstream sidecar(sidecar_path, std::ios::binary);
    if (!sidecar.good()) return false;
    vamana::rabitq::SidecarHeader header;
    try {
      read_exact(sidecar, &header, sizeof(header), sidecar_path);
    } catch (...) {
      return false;
    }
    if (header.magic != vamana::rabitq::kSidecarMagic ||
        header.version != vamana::rabitq::kSidecarVersion ||
        !vamana::rabitq::is_full_layout(header.entry_size, header.code_bits) ||
        header.code_bits != layout.rabitq_code_bits ||
        header.node_size != layout.node_bytes ||
        header.raw_vector_bytes != layout.vector_bytes ||
        header.entry_count != shards[shard_id].node_count ||
        std::filesystem::file_size(sidecar_path) !=
          sizeof(header) + header.entry_count * header.entry_size) {
      return false;
    }
    std::ifstream nodes(shards[shard_id].path, std::ios::binary);
    const std::array<u64, 3> samples{
      0, header.entry_count / 2, header.entry_count - 1};
    vec<byte_t> source(header.entry_size);
    vec<byte_t> stored(layout.rabitq_entry_bytes);
    for (size_t sample = 0; sample < samples.size(); ++sample) {
      if (sample != 0 && samples[sample] == samples[sample - 1]) continue;
      sidecar.clear();
      sidecar.seekg(static_cast<std::streamoff>(
        sizeof(header) + samples[sample] * header.entry_size));
      read_exact(sidecar, source.data(), source.size(), sidecar_path);
      nodes.clear();
      nodes.seekg(static_cast<std::streamoff>(
        kShardHeaderBytes + samples[sample] * layout.node_bytes + layout.rabitq_offset));
      read_exact(nodes, stored.data(), stored.size(), shards[shard_id].path);
      if (std::memcmp(source.data(), stored.data(), code_bytes) != 0 ||
          std::memcmp(source.data() + code_bytes,
                      stored.data() + gpu_search::format::rabitq_norm_offset(
                        layout.rabitq_code_bits), 2 * sizeof(f32)) != 0) {
        return false;
      }
    }
  }
  return true;
}

void validate_entry(const byte_t* entry, const Layout& layout) {
  f32 norm = 0.0f;
  f32 error = 0.0f;
  std::memcpy(&norm, entry + gpu_search::format::rabitq_norm_offset(
    layout.rabitq_code_bits), sizeof(norm));
  std::memcpy(&error, entry + gpu_search::format::rabitq_error_offset(
    layout.rabitq_code_bits), sizeof(error));
  if (!std::isfinite(norm) || norm < 0.0f || !std::isfinite(error) || error <= 0.0f) {
    throw std::runtime_error("storage index contains an invalid full RaBitQ entry");
  }
}

gpu_search::format::CodeHeader write_code_sidecar(
    const filepath_t& prefix, u32 shard_id, const vec<ShardInfo>& shards,
    const Layout& layout, bool from_sidecar, TemporaryPath& output_path,
    ProgressReporter& progress) {
  gpu_search::format::CodeHeader output_header;
  output_header.memory_node = shard_id;
  output_header.code_bits = layout.rabitq_code_bits;
  output_header.entry_bytes = layout.rabitq_entry_bytes;
  output_header.node_size = static_cast<u32>(layout.node_bytes);
  output_header.entry_count = shards[shard_id].node_count;
  output_header.remote_offset = gpu_search::format::align_up(
    shards[shard_id].dynamic_base_offset, 64);
  output_header.payload_bytes = output_header.entry_count * output_header.entry_bytes;

  std::ofstream output(output_path.temporary_path, std::ios::binary | std::ios::trunc);
  if (!output.good()) {
    throw std::runtime_error("failed to create " + output_path.temporary_path.string());
  }
  gpu_search::format::CodeHeader placeholder;
  write_exact(output, &placeholder, sizeof(placeholder), output_path.temporary_path);
  u64 checksum = gpu_search::format::checksum64_initial();

  if (from_sidecar) {
    const filepath_t source_path = index_path::rabitq_cache_file(
      prefix, shard_id + 1, shards.size());
    std::ifstream input(source_path, std::ios::binary);
    vamana::rabitq::SidecarHeader source_header;
    read_exact(input, &source_header, sizeof(source_header), source_path);
    const size_t entries_per_chunk = std::max<size_t>(1, kIoChunkBytes / source_header.entry_size);
    vec<byte_t> source(entries_per_chunk * source_header.entry_size);
    vec<byte_t> converted(entries_per_chunk * layout.rabitq_entry_bytes, 0);
    const u32 code_bytes = layout.rabitq_code_bits / 8u;
    for (u64 begin = 0; begin < output_header.entry_count; begin += entries_per_chunk) {
      const size_t count = static_cast<size_t>(std::min<u64>(
        entries_per_chunk, output_header.entry_count - begin));
      read_exact(input, source.data(), count * source_header.entry_size, source_path);
      std::fill(converted.begin(), converted.begin() + count * layout.rabitq_entry_bytes, 0);
      for (size_t index = 0; index < count; ++index) {
        const byte_t* source_entry = source.data() + index * source_header.entry_size;
        byte_t* destination = converted.data() + index * layout.rabitq_entry_bytes;
        std::memcpy(destination, source_entry, code_bytes);
        std::memcpy(destination + gpu_search::format::rabitq_norm_offset(
                      layout.rabitq_code_bits),
                    source_entry + code_bytes, 2 * sizeof(f32));
        validate_entry(destination, layout);
      }
      const size_t bytes = count * layout.rabitq_entry_bytes;
      write_exact(output, converted.data(), bytes, output_path.temporary_path);
      checksum = gpu_search::format::checksum64_update(checksum, converted.data(), bytes);
      progress.increment(count);
    }
  } else {
    std::ifstream input(shards[shard_id].path, std::ios::binary);
    const size_t nodes_per_chunk = std::max<size_t>(1, kIoChunkBytes / layout.node_bytes);
    vec<byte_t> nodes(nodes_per_chunk * layout.node_bytes);
    vec<byte_t> entries(nodes_per_chunk * layout.rabitq_entry_bytes);
    for (u64 begin = 0; begin < output_header.entry_count; begin += nodes_per_chunk) {
      const size_t count = static_cast<size_t>(std::min<u64>(
        nodes_per_chunk, output_header.entry_count - begin));
      input.seekg(static_cast<std::streamoff>(
        kShardHeaderBytes + begin * layout.node_bytes));
      read_exact(input, nodes.data(), count * layout.node_bytes, shards[shard_id].path);
      for (size_t index = 0; index < count; ++index) {
        const byte_t* node = nodes.data() + index * layout.node_bytes;
        if ((read_u64(node) & VamanaNode::HEADER_DELETED) != 0 ||
            read_u32(node + VamanaNode::offset_generation()) != 0) {
          throw std::runtime_error(
            "persisted mutations require storage compaction before V4 conversion");
        }
        byte_t* entry = entries.data() + index * layout.rabitq_entry_bytes;
        std::memcpy(entry, node + layout.rabitq_offset, layout.rabitq_entry_bytes);
        validate_entry(entry, layout);
      }
      const size_t bytes = count * layout.rabitq_entry_bytes;
      write_exact(output, entries.data(), bytes, output_path.temporary_path);
      checksum = gpu_search::format::checksum64_update(checksum, entries.data(), bytes);
      progress.increment(count);
    }
  }
  output_header.payload_checksum = checksum;
  str error;
  if (!gpu_search::format::write_code_header(output, output_header, &error)) {
    throw std::runtime_error(error);
  }
  output.flush();
  if (!output.good()) throw std::runtime_error("failed to flush GPU V4 code sidecar");
  return output_header;
}

void ensure_outputs_available(const filepath_t& prefix, u32 shard_count, bool overwrite) {
  vec<filepath_t> outputs{index_path::gpu_tiered_file(prefix)};
  for (u32 shard = 0; shard < shard_count; ++shard) {
    outputs.push_back(index_path::gpu_code_file(prefix, shard + 1, shard_count));
  }
  if (overwrite) return;
  for (const filepath_t& output : outputs) {
    if (std::filesystem::exists(output)) {
      throw std::runtime_error("output exists; pass --overwrite to replace " + output.string());
    }
  }
}

GpuSidecarConversionResult write_manifest_only(
    const GpuSidecarConversionOptions& options,
    const filepath_t& metadata_path,
    const nlohmann::json& metadata,
    const Layout& layout) {
  const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
  const vec<u64> counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  const vec<u64> graph_offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
  const vec<u64> dynamic_offsets =
    metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
  if (counts.size() != shard_count || graph_offsets.size() != shard_count ||
      dynamic_offsets.size() != shard_count) {
    throw std::runtime_error("GPU V4 manifest metadata has invalid shard arrays");
  }
  const filepath_t manifest_path = index_path::gpu_tiered_file(options.index_prefix);
  if (!options.overwrite && std::filesystem::exists(manifest_path)) {
    throw std::runtime_error(
      "output exists; pass --overwrite to replace " + manifest_path.string());
  }
  const vec<f32> centroid = metadata.at("rabitq_centroid").get<vec<f32>>();
  if (centroid.size() != layout.dim ||
      !std::all_of(centroid.begin(), centroid.end(), [](f32 value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error("metadata contains an invalid RaBitQ centroid");
  }

  gpu_search::format::View manifest;
  manifest.header.dim = layout.dim;
  manifest.header.graph_degree = layout.degree;
  manifest.header.vector_dtype = static_cast<u32>(layout.dtype);
  manifest.header.rabitq_code_bits = layout.rabitq_code_bits;
  manifest.header.rabitq_entry_bytes = layout.rabitq_entry_bytes;
  manifest.header.num_shards = shard_count;
  manifest.header.graph_entry_bytes = layout.graph_entry_bytes;
  manifest.header.graph_pointer_bytes = vamana::hot_graph::kCompactPointerBytes;
  manifest.header.graph_shard_bits = layout.graph_shard_bits;
  manifest.header.base_generation = 1;
  manifest.centroid = centroid;
  manifest.shards.resize(shard_count);
  vec<u64> ordinal_bases(shard_count, 0);
  u64 node_count = 0;
  for (u32 shard = 0; shard < shard_count; ++shard) {
    if (counts[shard] == 0 || dynamic_offsets[shard] == 0) {
      throw std::runtime_error("GPU V4 does not support an empty or invalid shard");
    }
    ordinal_bases[shard] = node_count;
    manifest.shards[shard] = {
      .ordinal_base = node_count,
      .node_count = counts[shard],
      .node_base_offset = kShardHeaderBytes,
      .node_stride = layout.node_bytes,
      .graph_base_offset = graph_offsets[shard],
      .dynamic_base_offset = dynamic_offsets[shard],
      .code_remote_offset = gpu_search::format::align_up(dynamic_offsets[shard], 64),
      .code_bytes = counts[shard] * layout.rabitq_entry_bytes,
      .memory_node = shard,
      .dynamic_record_bytes = layout.dynamic_record_bytes,
      .dynamic_hot_offset = layout.dynamic_hot_offset,
    };
    node_count += counts[shard];
  }
  if (node_count != metadata.at("num_vectors").get<u64>() ||
      node_count >= (1ull << 30)) {
    throw std::runtime_error("GPU V4 manifest node count is invalid");
  }
  manifest.header.num_nodes = node_count;

  const auto& medoid = metadata.at("medoid");
  const u32 medoid_shard = medoid.at("memory_node").get<u32>();
  const u64 medoid_offset = medoid.at("offset").get<u64>();
  if (medoid_shard >= shard_count || medoid_offset < kShardHeaderBytes ||
      (medoid_offset - kShardHeaderBytes) % layout.node_bytes != 0) {
    throw std::runtime_error("metadata contains an invalid medoid pointer");
  }
  const u64 medoid_slot = (medoid_offset - kShardHeaderBytes) / layout.node_bytes;
  if (medoid_slot >= counts[medoid_shard]) {
    throw std::runtime_error("metadata medoid exceeds its static shard");
  }
  manifest.header.medoid_ordinal = static_cast<u32>(
    ordinal_bases[medoid_shard] + medoid_slot);

  const u32 target = static_cast<u32>(std::min<u64>(options.entry_points, node_count));
  const u32 quota = (target + shard_count - 1) / shard_count;
  std::unordered_set<u32> selected;
  selected.insert(manifest.header.medoid_ordinal);
  manifest.entry_points.push_back(manifest.header.medoid_ordinal);
  const bool used_anchor_entry_points = append_anchor_entry_points(
    options.index_prefix, layout, shard_count, counts, ordinal_bases, target,
    selected, manifest.entry_points);
  for (u32 shard = 0; shard < shard_count && manifest.entry_points.size() < target; ++shard) {
    for (u32 sample = 0; sample < quota * 16 &&
         manifest.entry_points.size() < target; ++sample) {
      const u64 slot = mix64(options.seed ^
        (static_cast<u64>(shard) << 32) ^ sample) % counts[shard];
      const u32 ordinal = static_cast<u32>(ordinal_bases[shard] + slot);
      if (selected.insert(ordinal).second) manifest.entry_points.push_back(ordinal);
    }
  }
  for (u32 ordinal = 0; manifest.entry_points.size() < target; ++ordinal) {
    if (selected.insert(ordinal).second) manifest.entry_points.push_back(ordinal);
  }

  TemporaryPath manifest_output = make_temporary_path(manifest_path);
  str error;
  if (!gpu_search::format::write_file(manifest_output.temporary_path, manifest, &error)) {
    throw std::runtime_error(error);
  }
  GpuSidecarConversionResult result;
  result.index_file = manifest_path;
  result.node_count = node_count;
  result.entry_point_count = static_cast<u32>(manifest.entry_points.size());
  for (u32 shard = 0; shard < shard_count; ++shard) {
    result.code_remote_offsets.push_back(manifest.shards[shard].code_remote_offset);
    result.code_bytes.push_back(manifest.shards[shard].code_bytes);
  }

  nlohmann::json updated_metadata = metadata;
  updated_metadata["gpu_tiered_format"] = "gpu_tiered_v4";
  updated_metadata["gpu_tiered_file"] = manifest_path.string();
  updated_metadata["gpu_entry_points"] = result.entry_point_count;
  updated_metadata["gpu_code_files"] = vec<str>{};
  updated_metadata["gpu_code_remote_offsets"] = result.code_remote_offsets;
  updated_metadata["gpu_code_region_bytes"] = result.code_bytes;
  updated_metadata["gpu_code_materialization"] = "storage_startup";
  updated_metadata["gpu_graph_source"] = "storage_compact_plane";
  updated_metadata["gpu_tiered_source"] = "distributed_manifest_v1";
  updated_metadata["gpu_tiered_rabitq_source"] = "authoritative_nodes";
  updated_metadata["gpu_entry_point_source"] = used_anchor_entry_points
    ? "anchors_then_shard_hash" : "shard_hash";
  updated_metadata["gpu_hot_degree"] = 0;
  updated_metadata["gpu_graph_page_bytes"] = 0;
  updated_metadata["gpu_graph_page_offsets"] = vec<u64>{};
  updated_metadata["gpu_graph_page_region_bytes"] = vec<u64>{};
  TemporaryPath metadata_output = make_temporary_path(metadata_path);
  {
    std::ofstream output(metadata_output.temporary_path, std::ios::trunc);
    output << std::setw(2) << updated_metadata << '\n';
    if (!output.good()) throw std::runtime_error("failed to write updated metadata");
  }
  publish(manifest_output);
  publish(metadata_output);
  return result;
}

}  // namespace

GpuRabitqSource parse_gpu_rabitq_source(const str& value) {
  if (value == "auto") return GpuRabitqSource::automatic;
  if (value == "sidecar") return GpuRabitqSource::sidecar;
  if (value == "nodes") return GpuRabitqSource::nodes;
  throw std::invalid_argument("rabitq-source must be auto, sidecar, or nodes");
}

const char* gpu_rabitq_source_name(GpuRabitqSource source) {
  switch (source) {
    case GpuRabitqSource::automatic: return "auto";
    case GpuRabitqSource::sidecar: return "sidecar";
    case GpuRabitqSource::nodes: return "nodes";
  }
  return "unknown";
}

GpuSidecarConversionResult convert_gpu_sidecars(
    const GpuSidecarConversionOptions& options) {
  if (options.index_prefix.empty()) throw std::invalid_argument("index-prefix is required");
  if (options.entry_points == 0 ||
      options.entry_points > gpu_search::format::kMaxEntryPoints) {
    throw std::invalid_argument("gpu-entry-points must be in [1, 512]");
  }
  const filepath_t metadata_path{options.index_prefix.string() + ".meta.json"};
  std::ifstream metadata_input(metadata_path);
  if (!metadata_input.good()) {
    throw std::runtime_error("missing old index metadata: " + metadata_path.string());
  }
  nlohmann::json metadata;
  metadata_input >> metadata;
  const Layout layout = configure_layout(metadata);
  if (options.manifest_only) {
    return write_manifest_only(options, metadata_path, metadata, layout);
  }
  const vec<ShardInfo> shards = inspect_shards(options.index_prefix, metadata, layout);
  ensure_outputs_available(options.index_prefix, shards.size(), options.overwrite);

  const vec<f32> centroid = metadata.at("rabitq_centroid").get<vec<f32>>();
  if (centroid.size() != layout.dim ||
      !std::all_of(centroid.begin(), centroid.end(), [](f32 value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error("metadata contains an invalid RaBitQ centroid");
  }
  VamanaNode::set_rabitq_centroid(centroid);

  const bool sidecars_available = full_sidecars_available(
    options.index_prefix, shards, layout);
  const bool use_sidecars = options.rabitq_source == GpuRabitqSource::sidecar ||
    (options.rabitq_source == GpuRabitqSource::automatic && sidecars_available);
  if (options.rabitq_source == GpuRabitqSource::sidecar && !sidecars_available) {
    throw std::runtime_error("full RaBitQ sidecars are missing, stale, or incompatible");
  }

  u64 node_count = 0;
  vec<u64> ordinal_bases(shards.size());
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    ordinal_bases[shard] = node_count;
    node_count += shards[shard].node_count;
  }
  gpu_search::format::View manifest;
  manifest.header.dim = layout.dim;
  manifest.header.graph_degree = layout.degree;
  manifest.header.vector_dtype = static_cast<u32>(layout.dtype);
  manifest.header.rabitq_code_bits = layout.rabitq_code_bits;
  manifest.header.rabitq_entry_bytes = layout.rabitq_entry_bytes;
  manifest.header.num_shards = static_cast<u32>(shards.size());
  manifest.header.graph_entry_bytes = layout.graph_entry_bytes;
  manifest.header.graph_pointer_bytes = vamana::hot_graph::kCompactPointerBytes;
  manifest.header.graph_shard_bits = layout.graph_shard_bits;
  manifest.header.num_nodes = node_count;
  manifest.header.base_generation = 1;
  manifest.centroid = centroid;
  manifest.shards.resize(shards.size());

  RemotePtr medoid;
  for (const ShardInfo& shard : shards) {
    if (shard.medoid_raw != 0) {
      medoid = RemotePtr{shard.medoid_raw};
      break;
    }
  }
  if (!static_ordinal(medoid, shards, layout, ordinal_bases,
                      manifest.header.medoid_ordinal)) {
    throw std::runtime_error("storage shard headers contain an invalid medoid pointer");
  }

  const u32 target_entries = static_cast<u32>(std::min<u64>(options.entry_points, node_count));
  const u32 quota = (target_entries + shards.size() - 1) / shards.size();
  vec<std::priority_queue<EntrySample>> samples(shards.size());
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    for (u64 slot = 0; slot < shards[shard].node_count; ++slot) {
      const u32 ordinal = static_cast<u32>(ordinal_bases[shard] + slot);
      EntrySample sample{mix64(static_cast<u64>(ordinal) ^ options.seed), ordinal};
      auto& heap = samples[shard];
      if (heap.size() < quota) heap.push(sample);
      else if (sample.priority < heap.top().priority) {
        heap.pop();
        heap.push(sample);
      }
    }
  }
  vec<EntrySample> selected;
  for (auto& heap : samples) {
    while (!heap.empty()) {
      selected.push_back(heap.top());
      heap.pop();
    }
  }
  std::sort(selected.begin(), selected.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.priority < rhs.priority;
  });
  manifest.entry_points.push_back(manifest.header.medoid_ordinal);
  for (const EntrySample& sample : selected) {
    if (manifest.entry_points.size() >= target_entries) break;
    if (sample.ordinal != manifest.header.medoid_ordinal) {
      manifest.entry_points.push_back(sample.ordinal);
    }
  }

  ProgressReporter progress{"Writing GPU V4 RaBitQ streams", node_count};
  vamana::rabitq::ScopedNumaInterleave numa_policy;
  (void)numa_policy;
  vec<TemporaryPath> code_outputs;
  code_outputs.reserve(shards.size());
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    code_outputs.push_back(make_temporary_path(
      index_path::gpu_code_file(options.index_prefix, shard + 1, shards.size())));
  }
  vec<gpu_search::format::CodeHeader> code_headers(shards.size());
  parallel_for(0, shards.size(),
               std::min<size_t>(options.threads == 0 ? shards.size() : options.threads,
                                shards.size()),
    [&](size_t shard, size_t) {
      code_headers[shard] = write_code_sidecar(
        options.index_prefix, static_cast<u32>(shard), shards, layout,
        use_sidecars, code_outputs[shard], progress);
    });

  for (u32 shard = 0; shard < shards.size(); ++shard) {
    manifest.shards[shard] = {
      .ordinal_base = ordinal_bases[shard],
      .node_count = shards[shard].node_count,
      .node_base_offset = kShardHeaderBytes,
      .node_stride = layout.node_bytes,
      .graph_base_offset = shards[shard].graph_offset,
      .dynamic_base_offset = shards[shard].dynamic_base_offset,
      .code_remote_offset = code_headers[shard].remote_offset,
      .code_bytes = code_headers[shard].payload_bytes,
      .memory_node = shard,
      .dynamic_record_bytes = layout.dynamic_record_bytes,
      .dynamic_hot_offset = layout.dynamic_hot_offset,
    };
  }

  TemporaryPath manifest_output = make_temporary_path(
    index_path::gpu_tiered_file(options.index_prefix));
  str error;
  if (!gpu_search::format::write_file(manifest_output.temporary_path, manifest, &error)) {
    throw std::runtime_error(error);
  }

  GpuSidecarConversionResult result;
  result.index_file = manifest_output.final_path;
  result.node_count = node_count;
  result.entry_point_count = static_cast<u32>(manifest.entry_points.size());
  result.used_rabitq_sidecars = use_sidecars;
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    result.code_files.push_back(code_outputs[shard].final_path);
    result.code_remote_offsets.push_back(code_headers[shard].remote_offset);
    result.code_bytes.push_back(code_headers[shard].payload_bytes);
  }

  nlohmann::json updated_metadata = metadata;
  updated_metadata["gpu_tiered_format"] = "gpu_tiered_v4";
  updated_metadata["gpu_tiered_file"] = result.index_file.string();
  updated_metadata["gpu_entry_points"] = result.entry_point_count;
  updated_metadata["gpu_code_files"] = vec<str>{};
  for (const filepath_t& path : result.code_files) {
    updated_metadata["gpu_code_files"].push_back(path.string());
  }
  updated_metadata["gpu_code_remote_offsets"] = result.code_remote_offsets;
  updated_metadata["gpu_code_region_bytes"] = result.code_bytes;
  updated_metadata["gpu_code_materialization"] = "sidecar_or_storage_startup";
  updated_metadata["gpu_graph_source"] = "storage_compact_plane";
  updated_metadata["gpu_tiered_source"] = "legacy_sidecar_conversion_v2";
  updated_metadata["gpu_tiered_rabitq_source"] = use_sidecars ? "full_sidecars" : "nodes";
  updated_metadata["gpu_hot_degree"] = 0;
  updated_metadata["gpu_graph_page_bytes"] = 0;
  updated_metadata["gpu_graph_page_offsets"] = vec<u64>{};
  updated_metadata["gpu_graph_page_region_bytes"] = vec<u64>{};
  TemporaryPath metadata_output = make_temporary_path(metadata_path);
  {
    std::ofstream output(metadata_output.temporary_path, std::ios::trunc);
    output << std::setw(2) << updated_metadata << '\n';
    if (!output.good()) throw std::runtime_error("failed to write updated metadata");
  }

  for (TemporaryPath& output : code_outputs) publish(output);
  publish(manifest_output);
  publish(metadata_output);
  progress.finish();
  return result;
}

}  // namespace tools::vamana_offline
