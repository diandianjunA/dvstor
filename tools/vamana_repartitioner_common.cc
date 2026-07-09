#include "tools/vamana_repartitioner_common.hh"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#include "common/index_path.hh"
#include "remote_pointer.hh"
#include "vamana/anchor_index.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/storage_format.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_repartition {
namespace {

constexpr u64 kShardHeaderBytes = 16;

size_t align64(size_t value) {
  return (value + 63) & ~size_t{63};
}

void read_exact(std::istream& input, void* destination, size_t bytes, const filepath_t& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void write_exact(std::ostream& output, const void* source, size_t bytes, const filepath_t& path) {
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

void write_u64(std::ostream& output, u64 value, const filepath_t& path) {
  write_exact(output, &value, sizeof(value), path);
}

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

struct Layout {
  vamana::StorageFormat storage_format{vamana::StorageFormat::aos_v1};
  bool rabitq{};
  u32 dim{};
  u32 R{};
  VectorDType dtype{VectorDType::float32};
  size_t graph_hot_bytes{};
  size_t vector_offset{};
  size_t neighbors_offset{};
  size_t rabitq_offset{};
  size_t vector_bytes{};
  size_t vector_storage_bytes{};
  size_t rabitq_code_bits{};
  size_t rabitq_code_bytes{};
  size_t rabitq_code_storage_bytes{};
  size_t rabitq_entry_size{};
  size_t rabitq_entry_storage_size{};
  size_t node_size{};
  size_t hot_graph_entry_size{};
  u32 hot_graph_shard_bits{};
  size_t allocation_size{};
};

Layout make_layout(vamana::StorageFormat storage_format,
                   bool rabitq,
                   u32 dim,
                   u32 R,
                   VectorDType dtype,
                   u32 shard_count) {
  VamanaNode::disable_hot_graph();
  VamanaNode::disable_rabitq();
  VamanaNode::set_storage_format(storage_format);
  VamanaNode::init_static_storage(dim, R, dtype);
  if (rabitq) VamanaNode::enable_rabitq();

  Layout layout;
  layout.storage_format = storage_format;
  layout.rabitq = rabitq;
  layout.dim = dim;
  layout.R = R;
  layout.dtype = dtype;
  layout.graph_hot_bytes = VamanaNode::graph_hot_bytes();
  layout.vector_offset = VamanaNode::offset_vector();
  layout.neighbors_offset = VamanaNode::offset_neighbors();
  layout.rabitq_offset = rabitq ? VamanaNode::offset_rabitq_code() : 0;
  layout.vector_bytes = VamanaNode::vector_bytes();
  layout.vector_storage_bytes = VamanaNode::vector_storage_bytes();
  layout.rabitq_code_bits = rabitq ? VamanaNode::rabitq_code_bits() : 0;
  layout.rabitq_code_bytes = rabitq ? VamanaNode::rabitq_code_size() : 0;
  layout.rabitq_code_storage_bytes = rabitq ? VamanaNode::rabitq_code_storage_size() : 0;
  layout.rabitq_entry_size = rabitq ? VamanaNode::rabitq_entry_size() : 0;
  layout.rabitq_entry_storage_size = rabitq ? VamanaNode::rabitq_entry_storage_size() : 0;
  layout.node_size = VamanaNode::total_size();
  if (storage_format == vamana::StorageFormat::compact_v1) {
    layout.hot_graph_entry_size = VamanaNode::hot_graph_entry_size();
    layout.hot_graph_shard_bits = vamana::hot_graph::shard_bits_for(shard_count);
    layout.allocation_size = VamanaNode::align_compact(
      layout.node_size + layout.hot_graph_entry_size);
  } else {
    layout.allocation_size = layout.node_size;
  }
  return layout;
}

struct AnchorCandidate {
  u64 priority{};
  u32 vertex{};
  node_t id{};
  u16 degree{};
  RemotePtr pointer;

  bool operator<(const AnchorCandidate& other) const {
    return priority < other.priority;
  }
};

struct OutputFile {
  filepath_t final_path;
  filepath_t temp_path;
  std::fstream stream;
};

void replace_file(const filepath_t& temp, const filepath_t& output) {
  if (std::filesystem::exists(output)) std::filesystem::remove(output);
  std::filesystem::rename(temp, output);
}

}  // namespace

struct Index::Impl {
  struct Shard {
    filepath_t path;
    u64 free_ptr{};
    u64 medoid_raw{};
    u64 node_count{};
    u64 base_vertex{};
    u64 hot_graph_offset{};
    u16 hot_graph_version{};
  };

  struct Node {
    u64 header{};
    node_t id{};
    u32 generation{};
    vec<byte_t> vector;
    vec<byte_t> rabitq;
    vec<RemotePtr> neighbors;
  };

  Options options;
  nlohmann::json metadata;
  Layout input_layout;
  Layout output_layout;
  vec<Shard> shards;
  mutable vec<std::ifstream> shard_streams;
  mutable vec<byte_t> fixed_scratch;
  mutable vec<byte_t> hot_scratch;
  size_t total_nodes{};
  u32 medoid{};
  vamana::rabitq::Quantization rabitq_quantization{};

  explicit Impl(Options supplied) : options(std::move(supplied)) {
    load_metadata();
    configure_options();
    inspect_shards();
    locate_medoid();
    if (input_layout.rabitq) scan_rabitq_quantization();
  }

  void load_metadata() {
    const filepath_t path{options.input_prefix.string() + ".meta.json"};
    std::ifstream input(path);
    if (!input.good()) {
      throw std::runtime_error("failed to open metadata: " + path.string());
    }
    input >> metadata;
  }

  void configure_options() {
    if (metadata.value("schema_version", 0u) != 13) {
      throw std::runtime_error("input index must use schema version 13");
    }
    if (options.memory_nodes == 0) options.memory_nodes = metadata.value("num_memory_nodes", 0u);
    if (options.dim == 0) options.dim = metadata.value("dim", 0u);
    if (options.R == 0) options.R = metadata.value("R", 0u);
    if (options.memory_nodes == 0 || options.dim == 0 || options.R == 0) {
      throw std::runtime_error("missing memory-node, dimension, or degree metadata");
    }
    if (metadata.value("num_memory_nodes", 0u) != options.memory_nodes ||
        metadata.value("dim", 0u) != options.dim ||
        metadata.value("R", 0u) != options.R) {
      throw std::runtime_error("command-line layout parameters do not match metadata");
    }

    const VectorDType metadata_dtype =
      parse_vector_dtype(metadata.value("vector_data_type", str{"float32"}));
    if (options.vector_dtype_set && options.vector_dtype != metadata_dtype) {
      throw std::runtime_error("--vector-data-type does not match metadata");
    }
    options.vector_dtype = metadata_dtype;

    const str input_format_name = metadata.value("storage_format", str{});
    const auto input_format = vamana::parse_storage_format(input_format_name);
    if (!input_format) {
      throw std::runtime_error("unsupported input storage format: " + input_format_name);
    }
    const bool rabitq = metadata.value("node_layout", str{"standard"}) == "rabitq";
    if (!rabitq && metadata.value("node_layout", str{"standard"}) != "standard") {
      throw std::runtime_error("unsupported node layout");
    }
    input_layout = make_layout(
      *input_format, rabitq, options.dim, options.R, options.vector_dtype, options.memory_nodes);
    validate_layout(input_layout, true);

    const str output_format_name =
      options.storage_format == "auto" ? input_format_name : options.storage_format;
    const auto output_format = vamana::parse_storage_format(output_format_name);
    if (!output_format) {
      throw std::runtime_error(
        "--storage-format must be auto, vamana_aos_v1, or vamana_compact_v1");
    }
    options.storage_format = output_format_name;
    output_layout = make_layout(
      *output_format, rabitq, options.dim, options.R, options.vector_dtype, options.memory_nodes);

    if (!options.anchors_per_shard_set) {
      options.anchors_per_shard = metadata.value("anchor_count_per_shard", 4096u);
    }
    if (options.rabitq_cache_format == "auto") {
      options.rabitq_cache_format =
        rabitq ? metadata.value("rabitq_cache_format", str{"budget"}) : "budget";
    }
    if (options.rabitq_cache_format != "budget" &&
        options.rabitq_cache_format != "full") {
      throw std::runtime_error("--rabitq-cache-format must be auto, budget, or full");
    }
    if (metadata.value("distance", str{"l2"}) != "l2" && options.anchors_per_shard != 0) {
      throw std::runtime_error("anchor sidecars currently require an L2 index");
    }
  }

  void validate_layout(const Layout& layout, bool input) const {
    const str prefix = input ? "input " : "output ";
    if (metadata.value("node_size", 0u) != layout.node_size ||
        metadata.value("graph_hot_bytes", 0u) != layout.graph_hot_bytes ||
        metadata.value("vector_offset", 0u) != layout.vector_offset ||
        metadata.value("neighbors_offset", 0u) != layout.neighbors_offset ||
        metadata.value("rabitq_offset", 0u) != layout.rabitq_offset ||
        metadata.value("vector_bytes", 0u) != layout.vector_bytes) {
      throw std::runtime_error(prefix + "metadata storage layout does not match schema 13");
    }
    if (layout.rabitq &&
        (metadata.value("rabitq_code_bits", 0u) != layout.rabitq_code_bits ||
         metadata.value("rabitq_entry_size", 0u) != layout.rabitq_entry_size ||
         metadata.value("rabitq_centroid", vec<float>{}).size() != layout.dim)) {
      throw std::runtime_error(prefix + "RaBitQ metadata is incomplete or incompatible");
    }
    if (layout.storage_format == vamana::StorageFormat::compact_v1) {
      if (metadata.value("hot_graph_entry_size", 0u) != layout.hot_graph_entry_size ||
          metadata.value("hot_graph_pointer_bytes", 0u) !=
            vamana::hot_graph::kCompactPointerBytes ||
          metadata.value("hot_graph_shard_bits", std::numeric_limits<u32>::max()) !=
            layout.hot_graph_shard_bits) {
        throw std::runtime_error(prefix + "compact hot-graph metadata is incompatible");
      }
    }
  }

  void inspect_shards() {
    vec<u64> compact_counts;
    vec<u64> compact_offsets;
    vec<u64> compact_header_offsets;
    vec<u64> compact_dynamic_bases;
    if (input_layout.storage_format == vamana::StorageFormat::compact_v1) {
      compact_counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
      compact_offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
      compact_header_offsets = metadata.at("hot_graph_header_offsets").get<vec<u64>>();
      compact_dynamic_bases =
        metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
      if (compact_counts.size() != options.memory_nodes ||
          compact_offsets.size() != options.memory_nodes ||
          compact_header_offsets.size() != options.memory_nodes ||
          compact_dynamic_bases.size() != options.memory_nodes) {
        throw std::runtime_error("compact metadata shard arrays have the wrong size");
      }
    }

    shards.resize(options.memory_nodes);
    u64 base = 0;
    for (u32 shard_id = 0; shard_id < options.memory_nodes; ++shard_id) {
      Shard shard;
      shard.path = index_path::shard_file(
        options.input_prefix, shard_id + 1, options.memory_nodes);
      std::ifstream input(shard.path, std::ios::binary);
      if (!input.good()) {
        throw std::runtime_error("missing input shard: " + shard.path.string());
      }
      std::array<byte_t, kShardHeaderBytes> header{};
      read_exact(input, header.data(), header.size(), shard.path);
      shard.free_ptr = read_u64(header.data());
      shard.medoid_raw = read_u64(header.data() + sizeof(u64));
      const u64 file_size = std::filesystem::file_size(shard.path);
      if (shard.free_ptr < kShardHeaderBytes || shard.free_ptr > file_size) {
        throw std::runtime_error("invalid shard free pointer: " + shard.path.string());
      }

      if (input_layout.storage_format == vamana::StorageFormat::compact_v1) {
        shard.node_count = compact_counts[shard_id];
        shard.hot_graph_offset = compact_offsets[shard_id];
        vamana::hot_graph::Header hot_header;
        input.seekg(static_cast<std::streamoff>(compact_header_offsets[shard_id]));
        read_exact(input, &hot_header, sizeof(hot_header), shard.path);
        if (hot_header.magic != vamana::hot_graph::kMagic ||
            (hot_header.version != vamana::hot_graph::kVersion &&
             hot_header.version != vamana::hot_graph::kVersion2) ||
            hot_header.entry_bytes != input_layout.hot_graph_entry_size ||
            hot_header.entry_count != shard.node_count) {
          throw std::runtime_error("invalid compact hot-graph header: " + shard.path.string());
        }
        shard.hot_graph_version = hot_header.version;
        const u64 fixed_end = kShardHeaderBytes + shard.node_count * input_layout.node_size;
        const u64 hot_end = shard.hot_graph_offset +
          shard.node_count * input_layout.hot_graph_entry_size;
        if (fixed_end > file_size || hot_end > file_size) {
          throw std::runtime_error("truncated compact shard: " + shard.path.string());
        }
        if (shard.free_ptr != compact_dynamic_bases[shard_id]) {
          throw std::runtime_error(
            "compact shard contains persisted dynamic records; "
            "repartitioning dynamic inserts is not supported: " +
            shard.path.string());
        }
      } else {
        const u64 payload = shard.free_ptr - kShardHeaderBytes;
        if (payload % input_layout.node_size != 0) {
          throw std::runtime_error("AoS shard payload is not node aligned: " + shard.path.string());
        }
        shard.node_count = payload / input_layout.node_size;
      }
      shard.base_vertex = base;
      base += shard.node_count;
      if (base > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("repartitioner supports at most 2^32-1 nodes");
      }
      shards[shard_id] = shard;
    }
    total_nodes = static_cast<size_t>(base);
    shard_streams.resize(options.memory_nodes);
    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      shard_streams[shard].open(shards[shard].path, std::ios::binary);
      if (!shard_streams[shard].good()) {
        throw std::runtime_error("failed to open input shard: " + shards[shard].path.string());
      }
    }
    fixed_scratch.resize(input_layout.node_size);
    hot_scratch.resize(input_layout.hot_graph_entry_size);
  }

  size_t vertex_from_ptr(RemotePtr pointer) const {
    if (pointer.is_null() || pointer.memory_node() >= shards.size()) {
      throw std::runtime_error("neighbor pointer has an invalid shard");
    }
    const auto& shard = shards[pointer.memory_node()];
    if (pointer.byte_offset() < kShardHeaderBytes) {
      throw std::runtime_error("neighbor pointer is before the node plane");
    }
    const u64 relative = pointer.byte_offset() - kShardHeaderBytes;
    if (relative % input_layout.node_size != 0) {
      throw std::runtime_error("neighbor pointer is not aligned to the fixed-node size");
    }
    const u64 local = relative / input_layout.node_size;
    if (local >= shard.node_count) {
      throw std::runtime_error("neighbor pointer is outside the static node plane");
    }
    return static_cast<size_t>(shard.base_vertex + local);
  }

  std::pair<u32, u64> location_for_vertex(size_t vertex) const {
    if (vertex >= total_nodes) throw std::runtime_error("vertex is outside the index");
    const auto it = std::upper_bound(
      shards.begin(), shards.end(), vertex,
      [](size_t value, const Shard& shard) { return value < shard.base_vertex; });
    const size_t shard_index = it == shards.begin()
      ? 0
      : static_cast<size_t>(std::distance(shards.begin(), it) - 1);
    const auto& shard = shards[shard_index];
    const u64 local = vertex - shard.base_vertex;
    return {static_cast<u32>(shard_index), local};
  }

  void read_node(u32 shard_id, u64 local, bool include_payload, Node& node) const {
    const auto& shard = shards.at(shard_id);
    if (local >= shard.node_count) throw std::runtime_error("node slot is outside shard");
    auto& input = shard_streams.at(shard_id);
    input.clear();

    const u64 fixed_offset = kShardHeaderBytes + local * input_layout.node_size;
    input.seekg(static_cast<std::streamoff>(fixed_offset));
    read_exact(input, fixed_scratch.data(), fixed_scratch.size(), shard.path);

    node.header = read_u64(fixed_scratch.data());
    node.id = read_u32(fixed_scratch.data() + VamanaNode::HEADER_SIZE);
    node.generation = input_layout.storage_format == vamana::StorageFormat::compact_v1
      ? read_u32(fixed_scratch.data() + VamanaNode::offset_generation())
      : 0;
    if (include_payload) {
      node.vector.resize(input_layout.vector_bytes);
      std::memcpy(
        node.vector.data(),
        fixed_scratch.data() + input_layout.vector_offset,
        input_layout.vector_bytes);
      if (input_layout.rabitq) {
        node.rabitq.resize(input_layout.rabitq_entry_size);
        std::memcpy(
          node.rabitq.data(),
          fixed_scratch.data() + input_layout.rabitq_offset,
          input_layout.rabitq_entry_size);
      }
    }
    node.neighbors.clear();

    if (input_layout.storage_format == vamana::StorageFormat::compact_v1) {
      input.seekg(static_cast<std::streamoff>(
        shard.hot_graph_offset + local * input_layout.hot_graph_entry_size));
      read_exact(input, hot_scratch.data(), hot_scratch.size(), shard.path);
      if (shard.hot_graph_version >= vamana::hot_graph::kVersion2) {
        const u16 expected =
          vamana::hot_graph::checksum16(hot_scratch.data(), hot_scratch.size());
        const u16 stored = vamana::hot_graph::load_u16_le(hot_scratch.data() + 2);
        if (expected != stored) {
          throw std::runtime_error("compact hot-graph checksum mismatch: " + shard.path.string());
        }
        node.generation = vamana::hot_graph::load_u32_le(hot_scratch.data() + 4);
      }
      const u8 edge_count = shard.hot_graph_version >= vamana::hot_graph::kVersion2
        ? hot_scratch[0]
        : hot_scratch[sizeof(u32)];
      const size_t count = std::min<size_t>(edge_count, input_layout.R);
      node.neighbors.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        const RemotePtr pointer = vamana::hot_graph::decode_remote_ptr(
          hot_scratch.data() +
            vamana::hot_graph::neighbor_offset(static_cast<u32>(i)),
          input_layout.hot_graph_shard_bits);
        if (!pointer.is_null()) node.neighbors.push_back(pointer);
      }
    } else {
      const u8 edge_count =
        fixed_scratch[VamanaNode::HEADER_SIZE + sizeof(u32)];
      const size_t count = std::min<size_t>(edge_count, input_layout.R);
      node.neighbors.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        const RemotePtr pointer{
          read_u64(
            fixed_scratch.data() +
            input_layout.neighbors_offset +
            i * sizeof(u64))};
        if (!pointer.is_null()) node.neighbors.push_back(pointer);
      }
    }
  }

  Node read_node(size_t vertex, bool include_payload = true) const {
    const auto [shard, local] = location_for_vertex(vertex);
    Node node;
    read_node(shard, local, include_payload, node);
    return node;
  }

  void locate_medoid() {
    const RemotePtr pointer{shards.front().medoid_raw};
    if (!pointer.is_null()) {
      medoid = static_cast<u32>(vertex_from_ptr(pointer));
      return;
    }
    for (size_t vertex = 0; vertex < total_nodes; ++vertex) {
      if ((read_node(vertex, false).header & VamanaNode::HEADER_IS_MEDOID) != 0) {
        medoid = static_cast<u32>(vertex);
        return;
      }
    }
    throw std::runtime_error("could not locate input medoid");
  }

  void scan_rabitq_quantization() {
    if (options.rabitq_cache_format != "full" &&
        metadata.contains("rabitq_cache_norm_min") &&
        metadata.contains("rabitq_cache_norm_max")) {
      rabitq_quantization.norm_min = metadata.at("rabitq_cache_norm_min").get<f32>();
      rabitq_quantization.norm_max = metadata.at("rabitq_cache_norm_max").get<f32>();
      rabitq_quantization.error_min =
        metadata.value("rabitq_cache_error_min", 0.0f);
      rabitq_quantization.error_max =
        metadata.value("rabitq_cache_error_max", 0.0f);
      return;
    }
    rabitq_quantization.norm_min = std::numeric_limits<f32>::max();
    rabitq_quantization.norm_max = std::numeric_limits<f32>::lowest();
    rabitq_quantization.error_min = std::numeric_limits<f32>::max();
    rabitq_quantization.error_max = std::numeric_limits<f32>::lowest();
    Node node;
    for (size_t vertex = 0; vertex < total_nodes; ++vertex) {
      const auto [shard, local] = location_for_vertex(vertex);
      read_node(shard, local, true, node);
      f32 norm = 0.0f;
      f32 error = 0.0f;
      std::memcpy(&norm,
                  node.rabitq.data() + input_layout.rabitq_code_storage_bytes,
                  sizeof(norm));
      std::memcpy(&error,
                  node.rabitq.data() + input_layout.rabitq_code_storage_bytes + sizeof(norm),
                  sizeof(error));
      rabitq_quantization.norm_min = std::min(rabitq_quantization.norm_min, norm);
      rabitq_quantization.norm_max = std::max(rabitq_quantization.norm_max, norm);
      rabitq_quantization.error_min = std::min(rabitq_quantization.error_min, error);
      rabitq_quantization.error_max = std::max(rabitq_quantization.error_max, error);
    }
    if (total_nodes == 0) rabitq_quantization = {};
  }

  vec<vec<u32>> read_neighbor_lists(CrossShardStats* stats) const {
    if (stats) *stats = {};
    vec<vec<u32>> result(total_nodes);
    Node node;
    for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
      const auto& shard = shards[shard_id];
      for (u64 local = 0; local < shard.node_count; ++local) {
        const size_t vertex = shard.base_vertex + local;
        read_node(shard_id, local, false, node);
        auto& neighbors = result[vertex];
        neighbors.reserve(node.neighbors.size());
        for (RemotePtr pointer : node.neighbors) {
          neighbors.push_back(static_cast<u32>(vertex_from_ptr(pointer)));
          if (stats) {
            ++stats->total_edges;
            if (pointer.memory_node() != shard_id) ++stats->cross_edges;
          }
        }
      }
    }
    return result;
  }

  vec<u64> read_partition_edges(u32 max_degree, CrossShardStats* stats) const {
    if (stats) *stats = {};
    vec<u64> edges;
    edges.reserve(std::min<size_t>(
      total_nodes * static_cast<size_t>(max_degree),
      std::numeric_limits<u32>::max()));
    Node node;
    for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
      const auto& shard = shards[shard_id];
      for (u64 local = 0; local < shard.node_count; ++local) {
        const u32 vertex = static_cast<u32>(shard.base_vertex + local);
        read_node(shard_id, local, false, node);
        for (size_t i = 0; i < node.neighbors.size(); ++i) {
          const RemotePtr pointer = node.neighbors[i];
          const u32 neighbor = static_cast<u32>(vertex_from_ptr(pointer));
          if (stats) {
            ++stats->total_edges;
            if (pointer.memory_node() != shard_id) ++stats->cross_edges;
          }
          if (i < max_degree) {
            const u64 edge = tools::vamana_offline::pack_undirected_edge(vertex, neighbor);
            if (edge != 0) edges.push_back(edge);
          }
        }
      }
    }
    return edges;
  }

  void ensure_outputs_available(bool write_rabitq, bool write_anchors) const {
    if (options.input_prefix == options.output_prefix) {
      throw std::runtime_error("input-prefix and output-prefix must be different");
    }
    vec<filepath_t> outputs;
    outputs.push_back(filepath_t(options.output_prefix.string() + ".meta.json"));
    if (write_anchors) outputs.push_back(index_path::anchor_file(options.output_prefix));
    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      outputs.push_back(index_path::shard_file(
        options.output_prefix, shard + 1, options.memory_nodes));
      outputs.push_back(index_path::owner_idmap_file(
        options.output_prefix, shard + 1, options.memory_nodes));
      if (write_rabitq) {
        outputs.push_back(index_path::rabitq_cache_file(
          options.output_prefix, shard + 1, options.memory_nodes));
      }
    }
    for (const auto& output : outputs) {
      if (std::filesystem::exists(output) && !options.overwrite) {
        throw std::runtime_error("output exists, pass --overwrite: " + output.string());
      }
    }
  }

  WriteResult write(const vec<u32>& parts,
                    const str& partition_strategy,
                    const tools::vamana_offline::PartitionStats& partition_stats,
                    const CrossShardStats& before_stats,
                    const nlohmann::json& partition_metadata) const {
    if (parts.size() != total_nodes) {
      throw std::runtime_error("partition size does not match the index node count");
    }
    const bool write_rabitq = output_layout.rabitq;
    const bool write_anchors = options.anchors_per_shard != 0;
    ensure_outputs_available(write_rabitq, write_anchors);
    if (options.overwrite && !write_anchors) {
      std::filesystem::remove(index_path::anchor_file(options.output_prefix));
    }
    if (options.overwrite && !write_rabitq) {
      for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
        std::filesystem::remove(index_path::rabitq_cache_file(
          options.output_prefix, shard + 1, options.memory_nodes));
      }
    }

    const vec<tools::vamana_offline::NodePlacement> placements =
      tools::vamana_offline::assign_nodes_to_shards_from_partition(
        parts, options.memory_nodes, output_layout.node_size);
    vec<u64> counts(options.memory_nodes, 0);
    for (const auto& placement : placements) ++counts[placement.memory_node];

    vec<u64> fixed_ends(options.memory_nodes, kShardHeaderBytes);
    vec<u64> hot_header_offsets(options.memory_nodes, 0);
    vec<u64> hot_offsets(options.memory_nodes, 0);
    vec<u64> dynamic_base_offsets(options.memory_nodes, 0);
    vec<u64> file_sizes(options.memory_nodes, kShardHeaderBytes);
    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      fixed_ends[shard] = kShardHeaderBytes + counts[shard] * output_layout.node_size;
      if (output_layout.storage_format == vamana::StorageFormat::compact_v1) {
        hot_header_offsets[shard] = align64(fixed_ends[shard]);
        hot_offsets[shard] = align64(
          hot_header_offsets[shard] + sizeof(vamana::hot_graph::Header));
        dynamic_base_offsets[shard] = align64(
          hot_offsets[shard] + counts[shard] * output_layout.hot_graph_entry_size);
        file_sizes[shard] = dynamic_base_offsets[shard];
      } else {
        file_sizes[shard] = fixed_ends[shard];
      }
    }

    const filepath_t output_dir = options.output_prefix.parent_path();
    if (!output_dir.empty()) std::filesystem::create_directories(output_dir);

    vec<OutputFile> shard_files(options.memory_nodes);
    vec<OutputFile> idmap_files(options.memory_nodes);
    vec<OutputFile> rabitq_files(write_rabitq ? options.memory_nodes : 0);
    vec<u64> idmap_counts(options.memory_nodes, 0);
    const bool full_rabitq_cache =
      write_rabitq && options.rabitq_cache_format == "full";
    const u32 rabitq_entry_bytes = write_rabitq
      ? (full_rabitq_cache
          ? vamana::rabitq::full_entry_bytes()
          : vamana::rabitq::choose_entry_bytes(static_cast<u32>(output_layout.vector_bytes)))
      : 0;
    const u32 rabitq_code_bits = write_rabitq
      ? (full_rabitq_cache
          ? static_cast<u32>(output_layout.rabitq_code_bits)
          : vamana::rabitq::entry_code_bits(rabitq_entry_bytes))
      : 0;
    const u32 rabitq_code_bytes = write_rabitq
      ? rabitq_code_bits / 8u
      : 0;
    if (write_rabitq && rabitq_entry_bytes < 2) {
      throw std::runtime_error("RFQ5 sidecar cannot fit within the current vector budget");
    }

    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      auto& output = shard_files[shard];
      output.final_path = index_path::shard_file(
        options.output_prefix, shard + 1, options.memory_nodes);
      output.temp_path = filepath_t(output.final_path.string() + ".tmp");
      std::filesystem::remove(output.temp_path);
      output.stream.open(
        output.temp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
      if (!output.stream.good()) {
        throw std::runtime_error("failed to create output shard: " + output.temp_path.string());
      }
      if (file_sizes[shard] != 0) {
        output.stream.seekp(static_cast<std::streamoff>(file_sizes[shard] - 1));
        output.stream.put(0);
      }
      output.stream.seekp(0);
      write_u64(output.stream, file_sizes[shard], output.temp_path);
      write_u64(output.stream, 0, output.temp_path);
      if (output_layout.storage_format == vamana::StorageFormat::compact_v1) {
        vamana::hot_graph::Header header;
        header.version = vamana::hot_graph::kVersion2;
        header.entry_bytes = static_cast<u32>(output_layout.hot_graph_entry_size);
        header.max_degree = output_layout.R;
        header.compact_pointer_shard_bits = output_layout.hot_graph_shard_bits;
        header.entry_count = counts[shard];
        header.reserved0 = dynamic_base_offsets[shard];
        header.reserved1 = output_layout.allocation_size;
        header.reserved2 = static_cast<u32>(output_layout.node_size);
        output.stream.seekp(static_cast<std::streamoff>(hot_header_offsets[shard]));
        write_exact(output.stream, &header, sizeof(header), output.temp_path);
      }

      auto& idmap = idmap_files[shard];
      idmap.final_path = index_path::owner_idmap_file(
        options.output_prefix, shard + 1, options.memory_nodes);
      idmap.temp_path = filepath_t(idmap.final_path.string() + ".tmp");
      std::filesystem::remove(idmap.temp_path);
      idmap.stream.open(
        idmap.temp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
      if (!idmap.stream.good()) {
        throw std::runtime_error("failed to create idmap: " + idmap.temp_path.string());
      }
      vamana::idmap::Header idmap_header;
      idmap_header.owner_shard = shard;
      idmap_header.shard_count = options.memory_nodes;
      write_exact(idmap.stream, &idmap_header, sizeof(idmap_header), idmap.temp_path);

      if (write_rabitq) {
        auto& cache = rabitq_files[shard];
        cache.final_path = index_path::rabitq_cache_file(
          options.output_prefix, shard + 1, options.memory_nodes);
        cache.temp_path = filepath_t(cache.final_path.string() + ".tmp");
        std::filesystem::remove(cache.temp_path);
        cache.stream.open(
          cache.temp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!cache.stream.good()) {
          throw std::runtime_error(
            "failed to create RaBitQ sidecar: " + cache.temp_path.string());
        }
        vamana::rabitq::SidecarHeader cache_header;
        cache_header.entry_size = rabitq_entry_bytes;
        cache_header.code_bits = rabitq_code_bits;
        cache_header.node_size = static_cast<u32>(output_layout.node_size);
        cache_header.raw_vector_bytes = static_cast<u32>(output_layout.vector_bytes);
        cache_header.entry_count = counts[shard];
        cache_header.cache_budget_bytes =
          sizeof(vamana::rabitq::SidecarHeader) + counts[shard] * rabitq_entry_bytes;
        cache_header.quantization = rabitq_quantization;
        write_exact(cache.stream, &cache_header, sizeof(cache_header), cache.temp_path);
      }
    }

    vec<std::priority_queue<AnchorCandidate>> anchor_heaps(options.memory_nodes);
    CrossShardStats after_stats;
    vec<byte_t> fixed(output_layout.node_size, 0);
    vec<byte_t> hot(output_layout.hot_graph_entry_size, 0);
    vec<byte_t> rabitq_cache_entry(rabitq_entry_bytes, 0);
    vec<RemotePtr> rewritten_neighbors;
    Node node;

    for (size_t vertex = 0; vertex < total_nodes; ++vertex) {
      const auto [input_shard, input_local] = location_for_vertex(vertex);
      read_node(input_shard, input_local, true, node);
      const auto& placement = placements[vertex];
      const RemotePtr new_pointer{placement.memory_node, placement.offset};
      rewritten_neighbors.clear();
      rewritten_neighbors.reserve(node.neighbors.size());
      for (RemotePtr old_neighbor : node.neighbors) {
        const size_t neighbor_vertex = vertex_from_ptr(old_neighbor);
        const auto& neighbor_placement = placements[neighbor_vertex];
        const RemotePtr rewritten{neighbor_placement.memory_node, neighbor_placement.offset};
        rewritten_neighbors.push_back(rewritten);
        ++after_stats.total_edges;
        if (rewritten.memory_node() != placement.memory_node) ++after_stats.cross_edges;
      }

      std::fill(fixed.begin(), fixed.end(), 0);
      std::memcpy(fixed.data(), &node.header, sizeof(node.header));
      std::memcpy(
        fixed.data() + VamanaNode::HEADER_SIZE, &node.id, sizeof(node.id));
      if (output_layout.storage_format == vamana::StorageFormat::compact_v1) {
        std::memcpy(
          fixed.data() + VamanaNode::HEADER_SIZE + sizeof(node.id),
          &node.generation,
          sizeof(node.generation));
      } else {
        fixed[VamanaNode::HEADER_SIZE + sizeof(node.id)] =
          static_cast<byte_t>(std::min<size_t>(rewritten_neighbors.size(), output_layout.R));
        for (size_t i = 0; i < rewritten_neighbors.size() && i < output_layout.R; ++i) {
          std::memcpy(
            fixed.data() + output_layout.neighbors_offset + i * sizeof(u64),
            &rewritten_neighbors[i].raw_address,
            sizeof(u64));
        }
      }
      std::memcpy(
        fixed.data() + output_layout.vector_offset,
        node.vector.data(),
        output_layout.vector_bytes);
      if (output_layout.rabitq) {
        std::memcpy(
          fixed.data() + output_layout.rabitq_offset,
          node.rabitq.data(),
          output_layout.rabitq_entry_size);
      }

      auto& shard_output = shard_files[placement.memory_node];
      shard_output.stream.seekp(static_cast<std::streamoff>(placement.offset));
      write_exact(
        shard_output.stream, fixed.data(), fixed.size(), shard_output.temp_path);
      const u64 slot = (placement.offset - kShardHeaderBytes) / output_layout.node_size;
      if (output_layout.storage_format == vamana::StorageFormat::compact_v1) {
        VamanaNode::disable_rabitq();
        VamanaNode::set_storage_format(vamana::StorageFormat::compact_v1);
        VamanaNode::init_static_storage(
          output_layout.dim, output_layout.R, output_layout.dtype);
        VamanaNode::encode_hot_graph_entry(
          hot.data(),
          node.id,
          static_cast<u8>(std::min<size_t>(rewritten_neighbors.size(), output_layout.R)),
          rewritten_neighbors.data(),
          rewritten_neighbors.size(),
          output_layout.hot_graph_shard_bits,
          node.generation,
          vamana::hot_graph::kVersion2,
          (node.header & VamanaNode::HEADER_DELETED) != 0);
        shard_output.stream.seekp(static_cast<std::streamoff>(
          hot_offsets[placement.memory_node] + slot * output_layout.hot_graph_entry_size));
        write_exact(
          shard_output.stream, hot.data(), hot.size(), shard_output.temp_path);
      }

      const u32 owner = node.id % options.memory_nodes;
      vamana::idmap::Entry idmap_entry{
        node.id,
        new_pointer.raw_address,
        node.generation,
        (node.header & VamanaNode::HEADER_DELETED) != 0 ? vamana::idmap::kDeleted : 0};
      auto& idmap = idmap_files[owner];
      write_exact(idmap.stream, &idmap_entry, sizeof(idmap_entry), idmap.temp_path);
      ++idmap_counts[owner];

      if (write_rabitq) {
        std::fill(
          rabitq_cache_entry.begin(), rabitq_cache_entry.end(), byte_t{0});
        std::memcpy(
          rabitq_cache_entry.data(), node.rabitq.data(), rabitq_code_bytes);
        f32 norm = 0.0f;
        f32 error = 0.0f;
        std::memcpy(
          &norm,
          node.rabitq.data() + input_layout.rabitq_code_storage_bytes,
          sizeof(norm));
        std::memcpy(
          &error,
          node.rabitq.data() + input_layout.rabitq_code_storage_bytes + sizeof(norm),
          sizeof(error));
        if (full_rabitq_cache) {
          std::memcpy(rabitq_cache_entry.data() + rabitq_code_bytes,
                      &norm, sizeof(norm));
          std::memcpy(rabitq_cache_entry.data() + rabitq_code_bytes + sizeof(norm),
                      &error, sizeof(error));
        } else {
          rabitq_cache_entry[rabitq_code_bytes] = vamana::rabitq::quantize(
            norm, rabitq_quantization.norm_min, rabitq_quantization.norm_max);
        }
        auto& cache = rabitq_files[placement.memory_node];
        cache.stream.seekp(static_cast<std::streamoff>(
          sizeof(vamana::rabitq::SidecarHeader) + slot * rabitq_entry_bytes));
        write_exact(
          cache.stream,
          rabitq_cache_entry.data(),
          rabitq_cache_entry.size(),
          cache.temp_path);
      }

      if (write_anchors) {
        AnchorCandidate candidate{
          mix64(static_cast<u64>(node.id) ^ options.anchor_seed),
          static_cast<u32>(vertex),
          node.id,
          static_cast<u16>(std::min<size_t>(
            rewritten_neighbors.size(), std::numeric_limits<u16>::max())),
          new_pointer};
        auto& heap = anchor_heaps[placement.memory_node];
        if (heap.size() < options.anchors_per_shard) {
          heap.push(candidate);
        } else if (candidate.priority < heap.top().priority) {
          heap.pop();
          heap.push(candidate);
        }
      }
    }

    const RemotePtr new_medoid{
      placements[medoid].memory_node, placements[medoid].offset};
    shard_files[0].stream.seekp(sizeof(u64));
    write_u64(shard_files[0].stream, new_medoid.raw_address, shard_files[0].temp_path);

    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      auto& idmap = idmap_files[shard];
      vamana::idmap::Header header;
      header.owner_shard = shard;
      header.shard_count = options.memory_nodes;
      header.entry_count = idmap_counts[shard];
      idmap.stream.seekp(0);
      write_exact(idmap.stream, &header, sizeof(header), idmap.temp_path);
    }

    filepath_t anchor_temp;
    if (write_anchors) {
      const filepath_t anchor_path = index_path::anchor_file(options.output_prefix);
      anchor_temp = filepath_t(anchor_path.string() + ".tmp");
      std::filesystem::remove(anchor_temp);
      std::ofstream output(anchor_temp, std::ios::binary | std::ios::trunc);
      if (!output.good()) {
        throw std::runtime_error("failed to create anchor sidecar: " + anchor_temp.string());
      }
      vec<vec<AnchorCandidate>> selected(options.memory_nodes);
      u64 total_anchors = 0;
      for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
        auto& heap = anchor_heaps[shard];
        while (!heap.empty()) {
          selected[shard].push_back(heap.top());
          heap.pop();
        }
        std::sort(
          selected[shard].begin(), selected[shard].end(),
          [](const auto& lhs, const auto& rhs) {
            return lhs.pointer.raw_address < rhs.pointer.raw_address;
          });
        total_anchors += selected[shard].size();
      }

      vamana::anchor::Header header;
      header.dim = options.dim;
      header.shard_count = options.memory_nodes;
      header.vector_dtype = static_cast<u32>(options.vector_dtype);
      header.vector_bytes = static_cast<u32>(output_layout.vector_bytes);
      header.anchors_per_shard = options.anchors_per_shard;
      header.total_anchors = total_anchors;
      write_exact(output, &header, sizeof(header), anchor_temp);

      vec<float> decoded(options.dim);
      for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
        const auto& candidates = selected[shard];
        const vamana::anchor::ShardHeader shard_header{
          shard, static_cast<u32>(candidates.size())};
        write_exact(output, &shard_header, sizeof(shard_header), anchor_temp);
        vec<float> centroid(options.dim, 0.0f);
        for (const auto& candidate : candidates) {
          const Node node = read_node(candidate.vertex);
          decode_storage_vector_to_float(
            node.vector.data(), options.vector_dtype, options.dim, decoded.data());
          for (u32 d = 0; d < options.dim; ++d) centroid[d] += decoded[d];
        }
        if (!candidates.empty()) {
          const float scale = 1.0f / static_cast<float>(candidates.size());
          for (float& value : centroid) value *= scale;
        }
        write_exact(
          output, centroid.data(), centroid.size() * sizeof(float), anchor_temp);
        for (const auto& candidate : candidates) {
          const Node node = read_node(candidate.vertex);
          vamana::anchor::EntryHeader entry;
          entry.rptr_raw = candidate.pointer.raw_address;
          entry.id = candidate.id;
          entry.degree = candidate.degree;
          write_exact(output, &entry, sizeof(entry), anchor_temp);
          write_exact(
            output, node.vector.data(), node.vector.size(), anchor_temp);
        }
      }
    }

    for (auto& output : shard_files) output.stream.close();
    for (auto& output : idmap_files) output.stream.close();
    for (auto& output : rabitq_files) output.stream.close();
    for (const auto& output : shard_files) replace_file(output.temp_path, output.final_path);
    for (const auto& output : idmap_files) replace_file(output.temp_path, output.final_path);
    for (const auto& output : rabitq_files) replace_file(output.temp_path, output.final_path);
    if (write_anchors) {
      replace_file(anchor_temp, index_path::anchor_file(options.output_prefix));
    }

    nlohmann::json output_metadata = metadata;
    output_metadata["output_prefix"] = options.output_prefix.string();
    output_metadata["num_memory_nodes"] = options.memory_nodes;
    output_metadata["dim"] = options.dim;
    output_metadata["R"] = options.R;
    output_metadata["schema_version"] = 13;
    output_metadata["storage_format"] =
      vamana::storage_format_name(output_layout.storage_format);
    output_metadata["node_layout"] = output_layout.rabitq ? "rabitq" : "standard";
    output_metadata["node_size"] = output_layout.node_size;
    output_metadata["graph_hot_bytes"] = output_layout.graph_hot_bytes;
    output_metadata["vector_offset"] = output_layout.vector_offset;
    output_metadata["neighbors_offset"] = output_layout.neighbors_offset;
    output_metadata["rabitq_offset"] = output_layout.rabitq_offset;
    output_metadata["vector_data_type"] = vector_dtype_name(options.vector_dtype);
    output_metadata["vector_component_size"] =
      vector_dtype_component_size(options.vector_dtype);
    output_metadata["vector_bytes"] = output_layout.vector_bytes;
    output_metadata["vector_storage_bytes"] = output_layout.vector_storage_bytes;
    output_metadata["medoid"] = {
      {"memory_node", new_medoid.memory_node()},
      {"offset", new_medoid.byte_offset()}};
    output_metadata["partition_strategy"] = partition_strategy;
    output_metadata["partition_edge_cut"] = partition_stats.edge_cut;
    output_metadata["partition_cross_shard_ratio"] = after_stats.ratio();
    output_metadata["partition_source_prefix"] = options.input_prefix.string();
    output_metadata["partition_before_cross_shard_ratio"] = before_stats.ratio();
    output_metadata["partition_before_cross_shard_edges"] = before_stats.cross_edges;
    output_metadata["partition_before_total_edges"] = before_stats.total_edges;
    output_metadata["partition_after_cross_shard_edges"] = after_stats.cross_edges;
    output_metadata["partition_after_total_edges"] = after_stats.total_edges;
    for (const auto& [key, value] : partition_metadata.items()) {
      output_metadata[key] = value;
    }
    output_metadata["idmap_format"] = "owner_sharded_v1";
    output_metadata["anchor_format"] = write_anchors ? "owner_anchor_v1" : "";
    output_metadata["anchor_count_per_shard"] =
      write_anchors ? options.anchors_per_shard : 0;

    if (output_layout.storage_format == vamana::StorageFormat::compact_v1) {
      output_metadata["hot_graph_neighbor_read_bytes"] =
        output_layout.hot_graph_entry_size;
      output_metadata["hot_graph_neighbor_update_bytes"] =
        output_layout.hot_graph_entry_size;
      output_metadata["hot_graph_entry_size"] = output_layout.hot_graph_entry_size;
      output_metadata["hot_graph_pointer_bytes"] =
        vamana::hot_graph::kCompactPointerBytes;
      output_metadata["hot_graph_shard_bits"] = output_layout.hot_graph_shard_bits;
      output_metadata["hot_graph_offsets"] = hot_offsets;
      output_metadata["hot_graph_header_offsets"] = hot_header_offsets;
      output_metadata["hot_graph_entry_counts"] = counts;
      output_metadata["hot_graph_dynamic_base_offsets"] = dynamic_base_offsets;
      output_metadata["hot_graph_dynamic_record_bytes"] = output_layout.allocation_size;
      output_metadata["hot_graph_dynamic_hot_offset"] = output_layout.node_size;
      output_metadata["allocation_size"] = output_layout.allocation_size;
    } else {
      static constexpr std::array<const char*, 12> stale_keys{
        "hot_graph_neighbor_read_bytes",
        "hot_graph_neighbor_update_bytes",
        "hot_graph_entry_size",
        "hot_graph_pointer_bytes",
        "hot_graph_shard_bits",
        "hot_graph_offsets",
        "hot_graph_header_offsets",
        "hot_graph_entry_counts",
        "hot_graph_dynamic_base_offsets",
        "hot_graph_dynamic_record_bytes",
        "hot_graph_dynamic_hot_offset",
        "allocation_size"};
      for (const char* key : stale_keys) output_metadata.erase(key);
      output_metadata["allocation_size"] = output_layout.node_size;
    }
    if (write_rabitq) {
      output_metadata["rabitq_entry_storage_size"] =
        output_layout.rabitq_entry_storage_size;
      output_metadata["rabitq_cache_bits"] = rabitq_code_bits;
      output_metadata["rabitq_cache_entry_size"] = rabitq_entry_bytes;
      output_metadata["rabitq_cache_format"] = options.rabitq_cache_format;
      output_metadata["rabitq_cache_norm_min"] = rabitq_quantization.norm_min;
      output_metadata["rabitq_cache_norm_max"] = rabitq_quantization.norm_max;
      output_metadata["rabitq_cache_error_min"] = rabitq_quantization.error_min;
      output_metadata["rabitq_cache_error_max"] = rabitq_quantization.error_max;
    } else {
      static constexpr std::array<const char*, 12> stale_rabitq_keys{
        "rabitq_centroid",
        "rabitq_code_bits",
        "rabitq_entry_size",
        "rabitq_entry_storage_size",
        "rabitq_cache_bits",
        "rabitq_cache_entry_size",
        "rabitq_cache_format",
        "rabitq_cache_norm_min",
        "rabitq_cache_norm_max",
        "rabitq_cache_error_min",
        "rabitq_cache_error_max",
        "rabitq_offset"};
      for (const char* key : stale_rabitq_keys) output_metadata.erase(key);
      output_metadata["rabitq_offset"] = 0;
    }

    const filepath_t metadata_path{options.output_prefix.string() + ".meta.json"};
    const filepath_t metadata_temp{metadata_path.string() + ".tmp"};
    {
      std::ofstream output(metadata_temp, std::ios::trunc);
      if (!output.good()) {
        throw std::runtime_error("failed to create metadata: " + metadata_temp.string());
      }
      output << std::setw(2) << output_metadata << '\n';
      if (!output.good()) {
        throw std::runtime_error("failed to write metadata: " + metadata_temp.string());
      }
    }
    replace_file(metadata_temp, metadata_path);
    return {after_stats, total_nodes};
  }
};

Index::Index(Options options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;

const Options& Index::options() const {
  return impl_->options;
}

size_t Index::node_count() const {
  return impl_->total_nodes;
}

u32 Index::medoid_vertex() const {
  return impl_->medoid;
}

const str& Index::input_storage_format() const {
  return impl_->metadata.at("storage_format").get_ref<const str&>();
}

const str& Index::output_storage_format() const {
  return impl_->options.storage_format;
}

vec<vec<u32>> Index::read_neighbor_lists(CrossShardStats* stats) const {
  return impl_->read_neighbor_lists(stats);
}

vec<u64> Index::read_partition_edges(u32 max_degree, CrossShardStats* stats) const {
  return impl_->read_partition_edges(max_degree, stats);
}

WriteResult Index::write(
    const vec<u32>& parts,
    const str& partition_strategy,
    const tools::vamana_offline::PartitionStats& partition_stats,
    const CrossShardStats& before_stats,
    const nlohmann::json& partition_metadata) const {
  return impl_->write(
    parts, partition_strategy, partition_stats, before_stats, partition_metadata);
}

}  // namespace tools::vamana_repartition
