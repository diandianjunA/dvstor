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
#include <utility>
#include <unistd.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/progress.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/storage_format.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

constexpr u64 kShardHeaderBytes = 16;
constexpr size_t kIoChunkBytes = 64ull << 20;
constexpr u32 kUnsetId = std::numeric_limits<u32>::max();

struct Layout {
  vamana::StorageFormat storage_format{vamana::StorageFormat::aos_v1};
  u32 dim{};
  u32 degree{};
  VectorDType dtype{VectorDType::float32};
  u64 node_bytes{};
  u64 vector_offset{};
  u64 vector_bytes{};
  u64 neighbors_offset{};
  u64 rabitq_offset{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_bytes{};
  u64 hot_entry_bytes{};
  u32 hot_shard_bits{};
};

struct ShardInfo {
  filepath_t path;
  u64 file_bytes{};
  u64 free_pointer{};
  u64 medoid_raw{};
  u64 node_count{};
  u64 hot_offset{};
  u64 hot_header_offset{};
  u64 dynamic_base_offset{};
  u16 hot_version{};
};

struct EntrySample {
  u64 priority{};
  u32 id{};

  bool operator<(const EntrySample& other) const {
    return priority < other.priority;
  }
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
                const filepath_t& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void write_exact(std::ostream& output, const void* source, size_t bytes,
                 const filepath_t& path) {
  output.write(reinterpret_cast<const char*>(source), static_cast<std::streamsize>(bytes));
  if (!output.good()) {
    throw std::runtime_error("failed to write " + path.string());
  }
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

u64 checked_file_bytes(u64 header_bytes, u64 count, u64 entry_bytes,
                       const filepath_t& path) {
  if (entry_bytes != 0 &&
      count > (std::numeric_limits<u64>::max() - header_bytes) / entry_bytes) {
    throw std::runtime_error("declared entry count overflows file size: " + path.string());
  }
  return header_bytes + count * entry_bytes;
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
    throw std::runtime_error("converter requires schema_version 13");
  }
  if (metadata.value("distance", str{"l2"}) != "l2") {
    throw std::runtime_error("GPU RaBitQ sidecars currently require an L2 index");
  }
  if (metadata.value("node_layout", str{"standard"}) != "rabitq") {
    throw std::runtime_error(
      "old index must contain full RaBitQ entries; standard nodes would require rewriting .dat files");
  }
  const auto storage_format = vamana::parse_storage_format(
    metadata.value("storage_format", str{}));
  if (!storage_format.has_value()) {
    throw std::runtime_error("unsupported storage_format in metadata");
  }

  Layout layout;
  layout.storage_format = *storage_format;
  layout.dim = metadata.at("dim").get<u32>();
  layout.degree = metadata.at("R").get<u32>();
  layout.dtype = parse_vector_dtype(
    metadata.value("vector_data_type", str{"float32"}));
  VamanaNode::disable_hot_graph();
  VamanaNode::disable_rabitq();
  VamanaNode::set_storage_format(layout.storage_format);
  VamanaNode::init_static_storage(layout.dim, layout.degree, layout.dtype);
  VamanaNode::enable_rabitq();

  layout.node_bytes = VamanaNode::total_size();
  layout.vector_offset = VamanaNode::offset_vector();
  layout.vector_bytes = VamanaNode::vector_bytes();
  layout.neighbors_offset = VamanaNode::offset_neighbors();
  layout.rabitq_offset = VamanaNode::offset_rabitq_code();
  layout.rabitq_code_bits = VamanaNode::rabitq_code_bits();
  layout.rabitq_entry_bytes = static_cast<u32>(VamanaNode::rabitq_entry_size());
  layout.hot_entry_bytes = VamanaNode::hot_graph_entry_size();
  layout.hot_shard_bits = vamana::hot_graph::shard_bits_for(
    metadata.at("num_memory_nodes").get<u32>());

  const auto require_equal = [&](const char* name, u64 expected) {
    const u64 actual = metadata.value(name, std::numeric_limits<u64>::max());
    if (actual != expected) {
      throw std::runtime_error(
        str{"metadata layout mismatch for "} + name + ": metadata=" +
        std::to_string(actual) + " runtime=" + std::to_string(expected));
    }
  };
  require_equal("node_size", layout.node_bytes);
  require_equal("vector_offset", layout.vector_offset);
  require_equal("vector_bytes", layout.vector_bytes);
  require_equal("neighbors_offset", layout.neighbors_offset);
  require_equal("rabitq_offset", layout.rabitq_offset);
  require_equal("rabitq_code_bits", layout.rabitq_code_bits);
  require_equal("rabitq_entry_size", layout.rabitq_entry_bytes);
  if (layout.storage_format == vamana::StorageFormat::compact_v1) {
    require_equal("hot_graph_entry_size", layout.hot_entry_bytes);
    require_equal("hot_graph_pointer_bytes", vamana::hot_graph::kCompactPointerBytes);
    require_equal("hot_graph_shard_bits", layout.hot_shard_bits);
  }
  return layout;
}

vec<ShardInfo> inspect_shards(const filepath_t& prefix,
                              const nlohmann::json& metadata,
                              const Layout& layout) {
  const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
  vec<u64> compact_counts;
  vec<u64> compact_offsets;
  vec<u64> compact_header_offsets;
  vec<u64> compact_dynamic_offsets;
  if (layout.storage_format == vamana::StorageFormat::compact_v1) {
    compact_counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
    compact_offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
    compact_header_offsets = metadata.at("hot_graph_header_offsets").get<vec<u64>>();
    compact_dynamic_offsets = metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
    if (compact_counts.size() != shard_count || compact_offsets.size() != shard_count ||
        compact_header_offsets.size() != shard_count ||
        compact_dynamic_offsets.size() != shard_count) {
      throw std::runtime_error("compact graph metadata has an invalid shard count");
    }
  }

  vec<ShardInfo> shards(shard_count);
  u64 total_nodes = 0;
  for (u32 shard_id = 0; shard_id < shard_count; ++shard_id) {
    auto& shard = shards[shard_id];
    shard.path = index_path::shard_file(prefix, shard_id + 1, shard_count);
    std::ifstream input(shard.path, std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error(
        "missing old shard required for graph conversion: " + shard.path.string());
    }
    std::array<byte_t, kShardHeaderBytes> header{};
    read_exact(input, header.data(), header.size(), shard.path);
    shard.free_pointer = read_u64(header.data());
    shard.medoid_raw = read_u64(header.data() + sizeof(u64));
    shard.file_bytes = std::filesystem::file_size(shard.path);
    if (shard.free_pointer < kShardHeaderBytes || shard.free_pointer > shard.file_bytes) {
      throw std::runtime_error("invalid free pointer in " + shard.path.string());
    }

    if (layout.storage_format == vamana::StorageFormat::compact_v1) {
      shard.node_count = compact_counts[shard_id];
      shard.hot_offset = compact_offsets[shard_id];
      shard.hot_header_offset = compact_header_offsets[shard_id];
      shard.dynamic_base_offset = compact_dynamic_offsets[shard_id];
      vamana::hot_graph::Header hot_header;
      input.seekg(static_cast<std::streamoff>(shard.hot_header_offset));
      read_exact(input, &hot_header, sizeof(hot_header), shard.path);
      if (hot_header.magic != vamana::hot_graph::kMagic ||
          (hot_header.version != vamana::hot_graph::kVersion &&
           hot_header.version != vamana::hot_graph::kVersion2) ||
          hot_header.header_bytes != sizeof(vamana::hot_graph::Header) ||
          hot_header.entry_bytes != layout.hot_entry_bytes ||
          hot_header.max_degree != layout.degree ||
          hot_header.compact_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
          hot_header.compact_pointer_shard_bits != layout.hot_shard_bits ||
          hot_header.entry_count != shard.node_count) {
        throw std::runtime_error("invalid compact graph header in " + shard.path.string());
      }
      shard.hot_version = hot_header.version;
      const u64 fixed_end = kShardHeaderBytes + shard.node_count * layout.node_bytes;
      const u64 hot_end = shard.hot_offset + shard.node_count * layout.hot_entry_bytes;
      if (fixed_end > shard.file_bytes || hot_end > shard.file_bytes) {
        throw std::runtime_error("truncated compact graph shard: " + shard.path.string());
      }
      if (shard.free_pointer != shard.dynamic_base_offset) {
        throw std::runtime_error(
          "persisted dynamic records are not supported; compact the old index first: " +
          shard.path.string());
      }
    } else {
      const u64 payload_bytes = shard.free_pointer - kShardHeaderBytes;
      if (payload_bytes % layout.node_bytes != 0) {
        throw std::runtime_error("AoS shard payload is not node aligned: " + shard.path.string());
      }
      shard.node_count = payload_bytes / layout.node_bytes;
    }
    if (shard.node_count == 0) {
      throw std::runtime_error("GPU tiered format does not support an empty shard");
    }
    total_nodes += shard.node_count;
  }
  if (total_nodes != metadata.at("num_vectors").get<u64>()) {
    throw std::runtime_error(
      "old shard node counts do not match metadata num_vectors");
  }
  if (total_nodes == 0 || total_nodes > std::numeric_limits<u32>::max()) {
    throw std::runtime_error("GPU tiered format requires 1..2^32-1 dense node IDs");
  }
  return shards;
}

std::pair<u32, u64> pointer_slot(RemotePtr pointer,
                                 const vec<ShardInfo>& shards,
                                 const Layout& layout) {
  if (pointer.is_null() || pointer.memory_node() >= shards.size()) {
    throw std::runtime_error("graph contains a null or invalid remote pointer");
  }
  const auto& shard = shards[pointer.memory_node()];
  if (pointer.byte_offset() < kShardHeaderBytes) {
    throw std::runtime_error("remote pointer precedes the fixed-node plane");
  }
  const u64 relative = pointer.byte_offset() - kShardHeaderBytes;
  if (relative % layout.node_bytes != 0) {
    throw std::runtime_error("remote pointer is not aligned to node_size");
  }
  const u64 slot = relative / layout.node_bytes;
  if (slot >= shard.node_count) {
    throw std::runtime_error("remote pointer exceeds the static node plane");
  }
  return {pointer.memory_node(), slot};
}

class GraphPageWriter {
public:
  GraphPageWriter(const filepath_t& path, u32 shard, u32 page_bytes, u64 remote_offset)
      : path_(path), shard_(shard), page_bytes_(page_bytes), remote_offset_(remote_offset),
        page_(page_bytes, 0), cursor_(sizeof(gpu_search::format::PageHeader)),
        stream_buffer_(4ull << 20) {
    output_.rdbuf()->pubsetbuf(stream_buffer_.data(),
                              static_cast<std::streamsize>(stream_buffer_.size()));
    output_.open(path_, std::ios::binary | std::ios::trunc);
    if (!output_.good()) {
      throw std::runtime_error("failed to create GPU graph pages: " + path_.string());
    }
    gpu_search::format::ShardPageFileHeader header;
    header.page_bytes = page_bytes_;
    header.memory_node = shard_;
    header.remote_offset = remote_offset_;
    write_exact(output_, &header, sizeof(header), path_);
  }

  void append(u32 node_id, const vec<u32>& neighbors,
              gpu_search::format::IdEncoding encoding,
              gpu_search::format::NodeRecord& node_record) {
    const size_t id_bytes = static_cast<size_t>(encoding);
    if (neighbors.size() > std::numeric_limits<u16>::max()) {
      throw std::runtime_error("graph degree exceeds GPU page format capacity");
    }
    const size_t record_bytes = sizeof(gpu_search::format::PageNodeHeader) +
      neighbors.size() * id_bytes;
    const size_t padded_record_bytes = gpu_search::format::align_up(
      record_bytes, alignof(gpu_search::format::PageNodeHeader));
    if (padded_record_bytes + sizeof(gpu_search::format::PageHeader) > page_bytes_) {
      throw std::runtime_error("one adjacency list does not fit in a GPU graph page");
    }
    if (cursor_ + padded_record_bytes > page_bytes_ ||
        page_node_count_ == std::numeric_limits<u16>::max()) {
      flush_page();
    }

    node_record.cold_record_offset = static_cast<u32>(cursor_);
    node_record.cold_page_offset = remote_offset_ + page_index_ * page_bytes_;
    gpu_search::format::PageNodeHeader header;
    header.node_id = node_id;
    header.degree = static_cast<u16>(neighbors.size());
    header.flags = 0;
    std::memcpy(page_.data() + cursor_, &header, sizeof(header));
    byte_t* encoded = page_.data() + cursor_ + sizeof(header);
    for (size_t index = 0; index < neighbors.size(); ++index) {
      gpu_search::format::encode_id(
        encoded + index * id_bytes, neighbors[index], encoding);
    }
    cursor_ += padded_record_bytes;
    ++page_node_count_;
  }

  u64 finish() {
    if (page_node_count_ != 0) flush_page();
    gpu_search::format::ShardPageFileHeader header;
    header.page_bytes = page_bytes_;
    header.memory_node = shard_;
    header.remote_offset = remote_offset_;
    header.data_bytes = page_index_ * page_bytes_;
    header.checksum = gpu_search::format::checksum64(
      reinterpret_cast<const byte_t*>(&header),
      offsetof(gpu_search::format::ShardPageFileHeader, checksum));
    output_.seekp(0);
    write_exact(output_, &header, sizeof(header), path_);
    output_.flush();
    if (!output_.good()) {
      throw std::runtime_error("failed to flush GPU graph pages: " + path_.string());
    }
    return header.data_bytes;
  }

private:
  void flush_page() {
    gpu_search::format::PageHeader header;
    header.node_count = page_node_count_;
    header.payload_bytes = static_cast<u32>(cursor_ - sizeof(header));
    std::memcpy(page_.data(), &header, sizeof(header));
    output_.seekp(static_cast<std::streamoff>(
      sizeof(gpu_search::format::ShardPageFileHeader) + page_index_ * page_bytes_));
    write_exact(output_, page_.data(), page_.size(), path_);
    ++page_index_;
    std::fill(page_.begin(), page_.end(), byte_t{0});
    cursor_ = sizeof(gpu_search::format::PageHeader);
    page_node_count_ = 0;
  }

  filepath_t path_;
  u32 shard_{};
  u32 page_bytes_{};
  u64 remote_offset_{};
  std::ofstream output_;
  vec<byte_t> page_;
  size_t cursor_{};
  u16 page_node_count_{};
  u64 page_index_{};
  vec<char> stream_buffer_;
};

bool full_rabitq_sidecars_available(const filepath_t& prefix,
                                    const vec<ShardInfo>& shards,
                                    const Layout& layout) {
  for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
    const filepath_t path = index_path::rabitq_cache_file(
      prefix, shard_id + 1, shards.size());
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) return false;
    vamana::rabitq::SidecarHeader header;
    try {
      read_exact(input, &header, sizeof(header), path);
    } catch (const std::exception&) {
      return false;
    }
    if (header.magic != vamana::rabitq::kSidecarMagic ||
        header.version != vamana::rabitq::kSidecarVersion ||
        !vamana::rabitq::is_full_layout(header.entry_size, header.code_bits) ||
        header.code_bits != layout.rabitq_code_bits ||
        header.node_size != layout.node_bytes ||
        header.raw_vector_bytes != layout.vector_bytes ||
        header.entry_count != shards[shard_id].node_count) {
      return false;
    }
    const u64 expected_bytes = checked_file_bytes(
      sizeof(header), header.entry_count, header.entry_size, path);
    if (std::filesystem::file_size(path) != expected_bytes) return false;

    std::ifstream shard_input(shards[shard_id].path, std::ios::binary);
    if (!shard_input.good()) return false;
    const std::array<u64, 3> sample_slots{
      0, shards[shard_id].node_count / 2, shards[shard_id].node_count - 1};
    vec<byte_t> sidecar_entry(header.entry_size);
    vec<byte_t> node_entry(layout.rabitq_entry_bytes);
    const u32 code_bytes = layout.rabitq_code_bits / 8u;
    for (size_t sample = 0; sample < sample_slots.size(); ++sample) {
      if (sample != 0 && sample_slots[sample] == sample_slots[sample - 1]) continue;
      const u64 slot = sample_slots[sample];
      input.clear();
      input.seekg(static_cast<std::streamoff>(sizeof(header) + slot * header.entry_size));
      read_exact(input, sidecar_entry.data(), sidecar_entry.size(), path);
      shard_input.clear();
      shard_input.seekg(static_cast<std::streamoff>(
        kShardHeaderBytes + slot * layout.node_bytes + layout.rabitq_offset));
      read_exact(shard_input, node_entry.data(), node_entry.size(), shards[shard_id].path);
      if (std::memcmp(sidecar_entry.data(), node_entry.data(), code_bytes) != 0 ||
          std::memcmp(sidecar_entry.data() + code_bytes,
                      node_entry.data() + gpu_search::format::rabitq_norm_offset(
                        layout.rabitq_code_bits),
                      2 * sizeof(f32)) != 0) {
        return false;
      }
    }
  }
  return true;
}

void validate_rabitq_entry(const byte_t* entry, u32 code_bits) {
  f32 norm = 0.0f;
  f32 error = 0.0f;
  std::memcpy(&norm, entry + gpu_search::format::rabitq_norm_offset(code_bits), sizeof(norm));
  std::memcpy(&error, entry + gpu_search::format::rabitq_error_offset(code_bits), sizeof(error));
  if (!std::isfinite(norm) || norm < 0.0f || !std::isfinite(error) || error <= 0.0f) {
    throw std::runtime_error("old index contains an invalid full RaBitQ entry");
  }
}

void load_rabitq_from_sidecars(const filepath_t& prefix,
                               const vec<ShardInfo>& shards,
                               const vec<vec<u32>>& slot_ids,
                               const Layout& layout,
                               gpu_search::format::View& view,
                               ProgressReporter& progress,
                               u32 threads) {
  parallel_for(0, shards.size(), std::min<size_t>(threads == 0 ? shards.size() : threads,
                                                  shards.size()),
    [&](size_t shard_id, size_t) {
      const filepath_t path = index_path::rabitq_cache_file(
        prefix, shard_id + 1, shards.size());
      std::ifstream input(path, std::ios::binary);
      vamana::rabitq::SidecarHeader header;
      read_exact(input, &header, sizeof(header), path);
      const size_t entries_per_chunk = std::max<size_t>(1, kIoChunkBytes / header.entry_size);
      vec<byte_t> buffer(entries_per_chunk * header.entry_size);
      const u32 code_bytes = layout.rabitq_code_bits / 8u;
      for (u64 begin = 0; begin < shards[shard_id].node_count;
           begin += entries_per_chunk) {
        const size_t count = static_cast<size_t>(std::min<u64>(
          entries_per_chunk, shards[shard_id].node_count - begin));
        read_exact(input, buffer.data(), count * header.entry_size, path);
        for (size_t local = 0; local < count; ++local) {
          const u64 slot = begin + local;
          const u32 id = slot_ids[shard_id][slot];
          const byte_t* source = buffer.data() + local * header.entry_size;
          byte_t* destination = view.rabitq_entries.data() +
            static_cast<size_t>(id) * view.header.rabitq_entry_bytes;
          std::memcpy(destination, source, code_bytes);
          std::memcpy(
            destination + gpu_search::format::rabitq_norm_offset(layout.rabitq_code_bits),
            source + code_bytes, 2 * sizeof(f32));
          validate_rabitq_entry(destination, layout.rabitq_code_bits);
        }
        progress.increment(count);
      }
    });
}

void load_rabitq_from_nodes(const vec<ShardInfo>& shards,
                            const vec<vec<u32>>& slot_ids,
                            const Layout& layout,
                            gpu_search::format::View& view,
                            ProgressReporter& progress,
                            u32 threads) {
  parallel_for(0, shards.size(), std::min<size_t>(threads == 0 ? shards.size() : threads,
                                                  shards.size()),
    [&](size_t shard_id, size_t) {
      const auto& shard = shards[shard_id];
      std::ifstream input(shard.path, std::ios::binary);
      const size_t nodes_per_chunk = std::max<size_t>(1, kIoChunkBytes / layout.node_bytes);
      vec<byte_t> buffer(nodes_per_chunk * layout.node_bytes);
      for (u64 begin = 0; begin < shard.node_count; begin += nodes_per_chunk) {
        const size_t count = static_cast<size_t>(std::min<u64>(
          nodes_per_chunk, shard.node_count - begin));
        input.seekg(static_cast<std::streamoff>(
          kShardHeaderBytes + begin * layout.node_bytes));
        read_exact(input, buffer.data(), count * layout.node_bytes, shard.path);
        for (size_t local = 0; local < count; ++local) {
          const u64 slot = begin + local;
          const byte_t* node = buffer.data() + local * layout.node_bytes;
          const u32 id = slot_ids[shard_id][slot];
          if (read_u32(node + VamanaNode::offset_id()) != id) {
            throw std::runtime_error("idmap and fixed-node IDs disagree in " + shard.path.string());
          }
          if ((read_u64(node) & VamanaNode::HEADER_DELETED) != 0 ||
              (layout.storage_format == vamana::StorageFormat::compact_v1 &&
               read_u32(node + VamanaNode::offset_generation()) != 0)) {
            throw std::runtime_error(
              "persisted node mutations are not supported; compact the old index first");
          }
          byte_t* destination = view.rabitq_entries.data() +
            static_cast<size_t>(id) * view.header.rabitq_entry_bytes;
          std::memcpy(destination, node + layout.rabitq_offset,
                      view.header.rabitq_entry_bytes);
          validate_rabitq_entry(destination, layout.rabitq_code_bits);
        }
        progress.increment(count);
      }
    });
}

void add_entry_sample(vec<std::priority_queue<EntrySample>>& samples,
                      u32 shard, u32 id, u32 quota, u64 seed) {
  EntrySample sample{
    mix64(static_cast<u64>(id) ^
          (static_cast<u64>(static_cast<u32>(seed)) << 32)),
    id};
  auto& heap = samples[shard];
  if (heap.size() < quota) {
    heap.push(sample);
  } else if (sample.priority < heap.top().priority) {
    heap.pop();
    heap.push(sample);
  }
}

u32 resolve_neighbor_id(RemotePtr pointer,
                        const vec<ShardInfo>& shards,
                        const vec<vec<u32>>& slot_ids,
                        const Layout& layout) {
  const auto [shard, slot] = pointer_slot(pointer, shards, layout);
  const u32 id = slot_ids[shard][slot];
  if (id == kUnsetId) {
    throw std::runtime_error("remote pointer resolves to an unmapped node slot");
  }
  return id;
}

void populate_graph_record(u32 source_id,
                           const vec<RemotePtr>& pointers,
                           const vec<ShardInfo>& shards,
                           const vec<vec<u32>>& slot_ids,
                           const Layout& layout,
                           const GpuSidecarConversionOptions& options,
                           gpu_search::format::IdEncoding encoding,
                           GraphPageWriter& page_writer,
                           gpu_search::format::View& view,
                           u64& hot_edges,
                           u64& graph_edges) {
  vec<u32> neighbors;
  neighbors.reserve(pointers.size());
  for (RemotePtr pointer : pointers) {
    if (pointer.is_null()) {
      throw std::runtime_error("adjacency contains a null pointer inside its declared degree");
    }
    neighbors.push_back(resolve_neighbor_id(pointer, shards, slot_ids, layout));
  }
  auto& record = view.nodes[source_id];
  record.hot_neighbor_begin = source_id * options.hot_degree;
  record.hot_neighbor_count = static_cast<u16>(
    std::min<size_t>(neighbors.size(), options.hot_degree));
  const size_t hot_begin = static_cast<size_t>(record.hot_neighbor_begin);
  for (u32 index = 0; index < record.hot_neighbor_count; ++index) {
    view.hot_neighbors[hot_begin + index] = neighbors[index];
  }
  hot_edges += record.hot_neighbor_count;
  graph_edges += neighbors.size();
  page_writer.append(source_id, neighbors, encoding, record);
}

void convert_compact_graph_shard(size_t shard_id,
                                 const vec<ShardInfo>& shards,
                                 const vec<vec<u32>>& slot_ids,
                                 const Layout& layout,
                                 const GpuSidecarConversionOptions& options,
                                 gpu_search::format::IdEncoding encoding,
                                 GraphPageWriter& page_writer,
                                 gpu_search::format::View& view,
                                 ProgressReporter& progress,
                                 u64& hot_edges,
                                 u64& graph_edges) {
  const auto& shard = shards[shard_id];
  std::ifstream input(shard.path, std::ios::binary);
  const size_t entries_per_chunk = std::max<size_t>(1, kIoChunkBytes / layout.hot_entry_bytes);
  vec<byte_t> buffer(entries_per_chunk * layout.hot_entry_bytes);
  vec<RemotePtr> pointers;
  pointers.reserve(layout.degree);
  for (u64 begin = 0; begin < shard.node_count; begin += entries_per_chunk) {
    const size_t count = static_cast<size_t>(std::min<u64>(
      entries_per_chunk, shard.node_count - begin));
    input.seekg(static_cast<std::streamoff>(shard.hot_offset + begin * layout.hot_entry_bytes));
    read_exact(input, buffer.data(), count * layout.hot_entry_bytes, shard.path);
    for (size_t local = 0; local < count; ++local) {
      const u64 slot = begin + local;
      const u32 source_id = slot_ids[shard_id][slot];
      const byte_t* entry = buffer.data() + local * layout.hot_entry_bytes;
      if (shard.hot_version >= vamana::hot_graph::kVersion2) {
        const u16 stored = vamana::hot_graph::load_u16_le(entry + 2);
        const u16 actual = vamana::hot_graph::checksum16(entry, layout.hot_entry_bytes);
        if (stored != actual) {
          throw std::runtime_error("compact graph checksum mismatch in " + shard.path.string());
        }
        if ((entry[1] & VamanaNode::HOT_GRAPH_DELETED) != 0 ||
            vamana::hot_graph::load_u32_le(entry + 4) != 0) {
          throw std::runtime_error(
            "persisted graph mutations are not supported; compact the old index first");
        }
      } else if (vamana::hot_graph::load_u32_le(entry) != source_id) {
        throw std::runtime_error("compact v1 graph ID disagrees with idmap");
      }
      const u32 degree = shard.hot_version >= vamana::hot_graph::kVersion2
        ? entry[0] : entry[sizeof(u32)];
      if (degree > layout.degree) {
        throw std::runtime_error("compact graph degree exceeds metadata R");
      }
      pointers.clear();
      for (u32 index = 0; index < degree; ++index) {
        pointers.push_back(vamana::hot_graph::decode_remote_ptr(
          entry + vamana::hot_graph::neighbor_offset(index), layout.hot_shard_bits));
      }
      populate_graph_record(source_id, pointers, shards, slot_ids, layout, options,
                            encoding, page_writer, view, hot_edges, graph_edges);
    }
    progress.increment(count);
  }
}

void convert_aos_graph_shard(size_t shard_id,
                             const vec<ShardInfo>& shards,
                             const vec<vec<u32>>& slot_ids,
                             const Layout& layout,
                             const GpuSidecarConversionOptions& options,
                             gpu_search::format::IdEncoding encoding,
                             GraphPageWriter& page_writer,
                             gpu_search::format::View& view,
                             ProgressReporter& progress,
                             u64& hot_edges,
                             u64& graph_edges) {
  const auto& shard = shards[shard_id];
  std::ifstream input(shard.path, std::ios::binary);
  const size_t nodes_per_chunk = std::max<size_t>(1, kIoChunkBytes / layout.node_bytes);
  vec<byte_t> buffer(nodes_per_chunk * layout.node_bytes);
  vec<RemotePtr> pointers;
  pointers.reserve(layout.degree);
  for (u64 begin = 0; begin < shard.node_count; begin += nodes_per_chunk) {
    const size_t count = static_cast<size_t>(std::min<u64>(
      nodes_per_chunk, shard.node_count - begin));
    input.seekg(static_cast<std::streamoff>(kShardHeaderBytes + begin * layout.node_bytes));
    read_exact(input, buffer.data(), count * layout.node_bytes, shard.path);
    for (size_t local = 0; local < count; ++local) {
      const u64 slot = begin + local;
      const u32 source_id = slot_ids[shard_id][slot];
      const byte_t* node = buffer.data() + local * layout.node_bytes;
      if (read_u32(node + VamanaNode::offset_id()) != source_id) {
        throw std::runtime_error("idmap and AoS node IDs disagree");
      }
      if ((read_u64(node) & VamanaNode::HEADER_DELETED) != 0) {
        throw std::runtime_error(
          "persisted node deletions are not supported; compact the old index first");
      }
      const u32 degree = node[VamanaNode::offset_edge_count()];
      if (degree > layout.degree) {
        throw std::runtime_error("AoS graph degree exceeds metadata R");
      }
      pointers.clear();
      for (u32 index = 0; index < degree; ++index) {
        pointers.emplace_back(read_u64(
          node + layout.neighbors_offset + index * sizeof(u64)));
      }
      populate_graph_record(source_id, pointers, shards, slot_ids, layout, options,
                            encoding, page_writer, view, hot_edges, graph_edges);
    }
    progress.increment(count);
  }
}

void validate_options(const GpuSidecarConversionOptions& options,
                      const Layout& layout, u64 node_count) {
  if (options.hot_degree == 0 || options.hot_degree > layout.degree ||
      options.hot_degree > gpu_search::format::kMaxHotDegree) {
    throw std::invalid_argument("gpu-hot-degree must be in [1, min(R, 32)]");
  }
  if (options.entry_points == 0 || options.entry_points > 512) {
    throw std::invalid_argument("gpu-entry-points must be in [1, 512]");
  }
  if (options.page_bytes < gpu_search::format::kDefaultPageBytes ||
      (options.page_bytes & (options.page_bytes - 1)) != 0) {
    throw std::invalid_argument("gpu-graph-page-bytes must be a power of two >= 4096");
  }
  if (node_count > std::numeric_limits<u32>::max() / options.hot_degree) {
    throw std::runtime_error("GPU hot-neighbor offsets exceed uint32 capacity");
  }
}

void ensure_outputs_available(const filepath_t& prefix, u32 shard_count, bool overwrite) {
  vec<filepath_t> outputs{index_path::gpu_tiered_file(prefix)};
  for (u32 shard = 0; shard < shard_count; ++shard) {
    outputs.push_back(index_path::gpu_graph_pages_file(prefix, shard + 1, shard_count));
  }
  if (!overwrite) {
    for (const auto& output : outputs) {
      if (std::filesystem::exists(output)) {
        throw std::runtime_error("output exists; pass --overwrite: " + output.string());
      }
    }
  }
}

void verify_written_header(const filepath_t& path) {
  std::ifstream input(path, std::ios::binary);
  gpu_search::format::Header header;
  read_exact(input, &header, sizeof(header), path);
  str error;
  if (!gpu_search::format::validate_header(header, &error)) {
    throw std::runtime_error(error);
  }
  const u64 stored_checksum = header.checksum;
  header.checksum = 0;
  if (gpu_search::format::checksum64(
        reinterpret_cast<const byte_t*>(&header), sizeof(header)) != stored_checksum) {
    throw std::runtime_error("generated GPU tiered index header checksum mismatch");
  }
  if (std::filesystem::file_size(path) != header.file_bytes) {
    throw std::runtime_error("generated GPU tiered index file size mismatch");
  }
}

}

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
  if (options.index_prefix.empty()) {
    throw std::invalid_argument("index-prefix is required");
  }
  const filepath_t metadata_path{options.index_prefix.string() + ".meta.json"};
  std::ifstream metadata_input(metadata_path);
  if (!metadata_input.good()) {
    throw std::runtime_error("missing old index metadata: " + metadata_path.string());
  }
  nlohmann::json metadata;
  metadata_input >> metadata;

  const Layout layout = configure_layout(metadata);
  const vec<ShardInfo> shards = inspect_shards(options.index_prefix, metadata, layout);
  const u64 node_count = metadata.at("num_vectors").get<u64>();
  validate_options(options, layout, node_count);
  ensure_outputs_available(options.index_prefix, shards.size(), options.overwrite);

  const vec<f32> centroid = metadata.at("rabitq_centroid").get<vec<f32>>();
  if (centroid.size() != layout.dim ||
      !std::all_of(centroid.begin(), centroid.end(), [](f32 value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error("metadata contains an invalid RaBitQ centroid");
  }
  VamanaNode::set_rabitq_centroid(centroid);

  const u64 work_items = node_count * 3;
  ProgressReporter progress{"Converting old index to GPU sidecars", work_items};
  vamana::rabitq::ScopedNumaInterleave numa_policy;

  gpu_search::format::View view;
  view.header.page_bytes = options.page_bytes;
  view.header.dim = layout.dim;
  view.header.graph_degree = layout.degree;
  view.header.hot_degree = options.hot_degree;
  view.header.vector_dtype = static_cast<u32>(layout.dtype);
  view.header.rabitq_code_bits = layout.rabitq_code_bits;
  view.header.rabitq_entry_bytes = gpu_search::format::rabitq_entry_bytes(
    layout.rabitq_code_bits);
  if (view.header.rabitq_entry_bytes != layout.rabitq_entry_bytes) {
    throw std::runtime_error("GPU and storage RaBitQ entry layouts disagree");
  }
  view.header.id_encoding_bytes = node_count <= 0x00ffffffULL ? 3 : 4;
  view.header.num_shards = static_cast<u32>(shards.size());
  view.header.num_nodes = node_count;
  view.header.base_generation = 1;
  view.nodes.resize(node_count);
  for (auto& record : view.nodes) record.generation = 0;
  view.hot_neighbors.resize(static_cast<size_t>(node_count) * options.hot_degree, 0);
  view.rabitq_entries.resize(static_cast<size_t>(node_count) * layout.rabitq_entry_bytes, 0);
  view.shards.resize(shards.size());
  view.centroid = centroid;

  vec<vec<u32>> slot_ids(shards.size());
  for (size_t shard = 0; shard < shards.size(); ++shard) {
    slot_ids[shard].assign(shards[shard].node_count, kUnsetId);
  }
  const u32 target_entry_points = static_cast<u32>(
    std::min<u64>(options.entry_points, node_count));
  const u32 entry_quota = (target_entry_points + shards.size() - 1) / shards.size();
  vec<std::priority_queue<EntrySample>> entry_samples(shards.size());

  u64 mapped_nodes = 0;
  for (u32 owner = 0; owner < shards.size(); ++owner) {
    const filepath_t idmap_path = index_path::owner_idmap_file(
      options.index_prefix, owner + 1, shards.size());
    std::ifstream input(idmap_path, std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error("missing owner idmap: " + idmap_path.string());
    }
    vamana::idmap::Header header;
    read_exact(input, &header, sizeof(header), idmap_path);
    if (header.magic != vamana::idmap::kMagic ||
        header.version != vamana::idmap::kVersion ||
        header.owner_shard != owner || header.shard_count != shards.size() ||
        header.entry_count > node_count) {
      throw std::runtime_error("invalid owner idmap header: " + idmap_path.string());
    }
    const u64 expected_bytes = checked_file_bytes(
      sizeof(header), header.entry_count, sizeof(vamana::idmap::Entry), idmap_path);
    if (std::filesystem::file_size(idmap_path) != expected_bytes) {
      throw std::runtime_error("invalid owner idmap file size: " + idmap_path.string());
    }
    constexpr size_t entries_per_chunk = 1u << 18;
    vec<vamana::idmap::Entry> entries(entries_per_chunk);
    for (u64 begin = 0; begin < header.entry_count; begin += entries_per_chunk) {
      const size_t count = static_cast<size_t>(std::min<u64>(
        entries_per_chunk, header.entry_count - begin));
      read_exact(input, entries.data(), count * sizeof(entries.front()), idmap_path);
      for (size_t index = 0; index < count; ++index) {
        const auto& entry = entries[index];
        if (entry.id >= node_count || entry.id % shards.size() != owner ||
            entry.generation != 0 || entry.flags != 0) {
          throw std::runtime_error(
            "idmap contains a non-static, deleted, or non-dense entry: " + idmap_path.string());
        }
        auto& record = view.nodes[entry.id];
        if (record.generation != 0) {
          throw std::runtime_error("duplicate dense ID in owner idmaps");
        }
        const RemotePtr pointer{entry.rptr_raw};
        const auto [shard, slot] = pointer_slot(pointer, shards, layout);
        if (slot_ids[shard][slot] != kUnsetId) {
          throw std::runtime_error("multiple IDs map to the same storage node slot");
        }
        slot_ids[shard][slot] = entry.id;
        record.remote_node = entry.rptr_raw;
        record.generation = 1;
        record.shard = static_cast<u16>(shard);
        record.flags = 0;
        add_entry_sample(entry_samples, shard, entry.id, entry_quota, options.seed);
        ++mapped_nodes;
      }
      progress.increment(count);
    }
  }
  if (mapped_nodes != node_count ||
      std::any_of(view.nodes.begin(), view.nodes.end(), [](const auto& record) {
        return record.generation == 0;
      })) {
    throw std::runtime_error("owner idmaps do not cover every dense node ID exactly once");
  }
  for (const auto& ids : slot_ids) {
    if (std::find(ids.begin(), ids.end(), kUnsetId) != ids.end()) {
      throw std::runtime_error("owner idmaps do not cover every static shard slot");
    }
  }

  const RemotePtr medoid_pointer{shards.front().medoid_raw};
  const auto [medoid_shard, medoid_slot] = pointer_slot(medoid_pointer, shards, layout);
  view.header.medoid_id = slot_ids[medoid_shard][medoid_slot];

  const bool sidecars_available = full_rabitq_sidecars_available(
    options.index_prefix, shards, layout);
  bool use_sidecars = false;
  if (options.rabitq_source == GpuRabitqSource::sidecar) {
    if (!sidecars_available) {
      throw std::runtime_error("full RaBitQ sidecars were requested but are missing or incompatible");
    }
    use_sidecars = true;
  } else if (options.rabitq_source == GpuRabitqSource::automatic) {
    use_sidecars = sidecars_available;
  }
  if (use_sidecars) {
    load_rabitq_from_sidecars(options.index_prefix, shards, slot_ids, layout,
                              view, progress, options.threads);
  } else {
    load_rabitq_from_nodes(shards, slot_ids, layout, view, progress, options.threads);
  }

  vec<EntrySample> selected_entries;
  for (auto& samples : entry_samples) {
    while (!samples.empty()) {
      selected_entries.push_back(samples.top());
      samples.pop();
    }
  }
  std::sort(selected_entries.begin(), selected_entries.end(),
            [](const EntrySample& lhs, const EntrySample& rhs) {
              return lhs.priority < rhs.priority;
            });
  if (selected_entries.size() > target_entry_points) {
    selected_entries.resize(target_entry_points);
  }
  view.entry_points.reserve(target_entry_points);
  view.entry_points.push_back(view.header.medoid_id);
  for (const auto& sample : selected_entries) {
    if (sample.id != view.header.medoid_id &&
        view.entry_points.size() < target_entry_points) {
      view.entry_points.push_back(sample.id);
    }
  }

  TemporaryPath index_output = make_temporary_path(
    index_path::gpu_tiered_file(options.index_prefix));
  vec<std::unique_ptr<TemporaryPath>> page_outputs;
  page_outputs.reserve(shards.size());
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    page_outputs.push_back(std::make_unique<TemporaryPath>(make_temporary_path(
      index_path::gpu_graph_pages_file(options.index_prefix, shard + 1, shards.size()))));
  }

  vec<u64> shard_hot_edges(shards.size(), 0);
  vec<u64> shard_graph_edges(shards.size(), 0);
  const auto encoding = static_cast<gpu_search::format::IdEncoding>(
    view.header.id_encoding_bytes);
  parallel_for(0, shards.size(),
               std::min<size_t>(options.threads == 0 ? shards.size() : options.threads,
                                shards.size()),
    [&](size_t shard_id, size_t) {
      const u64 remote_offset = gpu_search::format::align_up(
        shards[shard_id].file_bytes, options.page_bytes);
      auto& region = view.shards[shard_id];
      region.graph_pages_offset = remote_offset;
      region.vector_region_offset = kShardHeaderBytes + layout.vector_offset;
      region.vector_stride = layout.node_bytes;
      region.node_count = shards[shard_id].node_count;
      region.memory_node = static_cast<u32>(shard_id);
      GraphPageWriter writer(page_outputs[shard_id]->temporary_path,
                             static_cast<u32>(shard_id), options.page_bytes, remote_offset);
      if (layout.storage_format == vamana::StorageFormat::compact_v1) {
        convert_compact_graph_shard(
          shard_id, shards, slot_ids, layout, options, encoding, writer, view,
          progress, shard_hot_edges[shard_id], shard_graph_edges[shard_id]);
      } else {
        convert_aos_graph_shard(
          shard_id, shards, slot_ids, layout, options, encoding, writer, view,
          progress, shard_hot_edges[shard_id], shard_graph_edges[shard_id]);
      }
      region.graph_pages_bytes = writer.finish();
    });

  str write_error;
  if (!gpu_search::format::write_file(index_output.temporary_path, view, &write_error)) {
    throw std::runtime_error(write_error);
  }
  verify_written_header(index_output.temporary_path);

  GpuSidecarConversionResult result;
  result.index_file = index_output.final_path;
  result.node_count = node_count;
  result.hot_edge_count = 0;
  result.graph_edge_count = 0;
  result.entry_point_count = static_cast<u32>(view.entry_points.size());
  result.used_rabitq_sidecars = use_sidecars;
  result.graph_page_files.reserve(shards.size());
  result.graph_page_offsets.reserve(shards.size());
  result.graph_page_bytes.reserve(shards.size());
  for (u32 shard = 0; shard < shards.size(); ++shard) {
    result.hot_edge_count += shard_hot_edges[shard];
    result.graph_edge_count += shard_graph_edges[shard];
    result.graph_page_files.push_back(page_outputs[shard]->final_path);
    result.graph_page_offsets.push_back(view.shards[shard].graph_pages_offset);
    result.graph_page_bytes.push_back(view.shards[shard].graph_pages_bytes);
  }

  nlohmann::json updated_metadata = metadata;
  updated_metadata["gpu_tiered_format"] = "gpu_tiered_v3";
  updated_metadata["gpu_tiered_file"] = result.index_file.string();
  updated_metadata["gpu_hot_degree"] = options.hot_degree;
  updated_metadata["gpu_entry_points"] = result.entry_point_count;
  updated_metadata["gpu_graph_page_bytes"] = options.page_bytes;
  updated_metadata["gpu_graph_page_offsets"] = result.graph_page_offsets;
  updated_metadata["gpu_graph_page_region_bytes"] = result.graph_page_bytes;
  updated_metadata["gpu_tiered_source"] = "legacy_sidecar_conversion_v1";
  updated_metadata["gpu_tiered_rabitq_source"] = use_sidecars ? "full_sidecars" : "nodes";
  TemporaryPath metadata_output = make_temporary_path(metadata_path);
  {
    std::ofstream output(metadata_output.temporary_path, std::ios::trunc);
    if (!output.good()) {
      throw std::runtime_error("failed to create temporary metadata file");
    }
    output << std::setw(2) << updated_metadata << '\n';
    if (!output.good()) {
      throw std::runtime_error("failed to write updated metadata");
    }
  }

  for (auto& page_output : page_outputs) publish(*page_output);
  publish(index_output);
  publish(metadata_output);
  progress.finish();
  return result;
}

}
